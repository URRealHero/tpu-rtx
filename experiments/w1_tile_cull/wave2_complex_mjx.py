"""Wave 2: Massive Dynamic MJX Primitive Scaling (50 -> 200 -> 1,000) & Cluttered Warehouse Benchmark on w1 TPU."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import re
import time
import traceback
from typing import Any, Dict, List

import jax
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np

from tpu_mjx_render.renderer import RenderSceneSpec, render_batch
from tpu_mjx_render.roofline import inspect_hlo_ops

from v3_hierarchical import pallas_raycast_hierarchical
from warehouse_scene import build_warehouse_mjcf


def _check_hlo_zero_copy(hlo_text: str) -> Dict[str, int]:
  info = inspect_hlo_ops(hlo_text)
  info["send_recv_ops"] = len(re.findall(r"\bsend\b|\brecv\b", hlo_text))
  return info


def _make_warehouse_batch(n_target_prims: int, batch_size: int, height: int, width: int, cam_name: str = "overview"):
  xml = build_warehouse_mjcf(n_target_prims=n_target_prims)
  m = mujoco.MjModel.from_xml_string(xml)
  mx = mjx.put_model(m, impl="jax")
  cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
  spec = RenderSceneSpec.from_mjmodel(m, cam_id=cam_id, height=height, width=width, use_shadows=True)

  dx0 = mjx.make_data(m, impl="jax")
  dx0 = mjx.forward(mx, dx0)
  dx_batch = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x[None, ...], (batch_size, *x.shape)).copy(), dx0)

  # Dynamic actuator targets so dual arms + rotating racks + sliding trays all move in MJX!
  ctrl_np = np.zeros((batch_size, m.nu), dtype=np.float32)
  t_phase = np.linspace(0.0, 2.0 * np.pi, batch_size, endpoint=False, dtype=np.float32)
  for j in range(m.nu):
    ctrl_np[:, j] = 0.35 * np.sin(t_phase + 0.5 * j)
  ctrl = jnp.asarray(ctrl_np)
  return m, mx, spec, dx_batch, ctrl


def run_parity_wave2() -> List[Dict[str, Any]]:
  """Verify exact parity of pallas_raycast_hierarchical vs v2_soa on G~50 and G~200 warehouse scenes on live TPU."""
  pallas_interp = (jax.default_backend() == "cpu")
  results = []
  for n_prims in (50, 200):
    m, mx, spec, dx_batch, ctrl = _make_warehouse_batch(n_prims, batch_size=8, height=64, width=64)
    dx_batch = jax.device_put(dx_batch, jax.devices()[0])
    ctrl = jax.device_put(ctrl, jax.devices()[0])
    # Step 5 times so objects/arms/racks are in dynamic non-zero motion
    step_fn = jax.jit(lambda d, u: jax.vmap(lambda di, ui: mjx.step(mx, di.replace(ctrl=ui)))(d, u))
    for _ in range(5):
      dx_batch = step_fn(dx_batch, ctrl)
    jax.block_until_ready(dx_batch.geom_xpos)

    ref_fn = jax.jit(lambda d: render_batch(spec, d, mode="v2_soa"))
    ref_gb = ref_fn(dx_batch)
    jax.block_until_ready((ref_gb.rgb, ref_gb.depth, ref_gb.geom_id))

    for cm in ("flat_loop", "cluster1", "hier2"):
      fn = jax.jit(lambda d, _cm=cm: pallas_raycast_hierarchical(
          spec, d, interpret=pallas_interp, block_b=8, block_r=512, cull_mode=_cm
      ))
      hlo_info = _check_hlo_zero_copy(fn.lower(dx_batch).as_text())
      gb = fn(dx_batch)
      jax.block_until_ready((gb.rgb, gb.depth, gb.geom_id))

      geom_match = float(jnp.mean(gb.geom_id == ref_gb.geom_id))
      depth_mae = float(jnp.mean(jnp.abs(gb.depth - ref_gb.depth)))
      max_rgb_diff = float(jnp.max(jnp.abs(gb.rgb - ref_gb.rgb)))
      passed = (
          geom_match == 1.0
          and depth_mae < 1e-5
          and max_rgb_diff < 5e-4
          and hlo_info["host_callbacks"] == 0
          and hlo_info["send_recv_ops"] == 0
      )
      rec = {
          "n_target_prims": n_prims,
          "n_enabled_geoms": len(spec.enabled_geom_ids),
          "cull_mode": cm,
          "geom_match": geom_match,
          "depth_mae_m": depth_mae,
          "max_rgb_diff": max_rgb_diff,
          "host_callbacks": hlo_info["host_callbacks"],
          "send_recv_ops": hlo_info["send_recv_ops"],
          "passed": passed,
      }
      print(f"[W2 PARITY] G={len(spec.enabled_geom_ids)} {cm} -> pass={passed} geom={geom_match} depth_mae={depth_mae:.3e} rgb_max={max_rgb_diff:.3e}", flush=True)
      results.append(rec)
  return results


def bench_1c_case(n_prims: int, batch_size: int, res: int, mode: str, n_warmup: int = 2, n_steps: int = 12) -> Dict[str, Any]:
  pallas_interp = (jax.default_backend() == "cpu")
  m, mx, spec, dx_batch, ctrl = _make_warehouse_batch(n_prims, batch_size, res, res)
  dx_batch = jax.device_put(dx_batch, jax.devices()[0])
  ctrl = jax.device_put(ctrl, jax.devices()[0])
  rec: Dict[str, Any] = {
      "n_chips": 1,
      "n_target_prims": n_prims,
      "n_enabled_geoms": len(spec.enabled_geom_ids),
      "nq": int(m.nq),
      "nv": int(m.nv),
      "batch_size": batch_size,
      "resolution": f"{res}x{res}",
      "mode": mode,
  }
  try:
    @jax.jit
    def phys_only(d, u):
      return jax.vmap(lambda di, ui: mjx.step(mx, di.replace(ctrl=ui)))(d, u)

    @jax.jit
    def e2e_fn(d, u):
      dn = jax.vmap(lambda di, ui: mjx.step(mx, di.replace(ctrl=ui)))(d, u)
      if mode == "pallas_v2":
        gb = render_batch(spec, dn, mode="pallas_v2", pallas_interpret=pallas_interp)
      elif mode == "v2_soa":
        gb = render_batch(spec, dn, mode="v2_soa")
      else:
        gb = pallas_raycast_hierarchical(
            spec, dn, interpret=pallas_interp, block_b=8, block_r=1024, cull_mode=mode
        )
      return dn, gb.rgb, gb.depth

    hlo_info = _check_hlo_zero_copy(e2e_fn.lower(dx_batch, ctrl).as_text())
    assert hlo_info["host_callbacks"] == 0
    assert hlo_info["send_recv_ops"] == 0

    dp = dx_batch
    for _ in range(n_warmup):
      dp = phys_only(dp, ctrl)
    jax.block_until_ready(dp.qpos)
    t0 = time.perf_counter()
    for _ in range(n_steps):
      dp = phys_only(dp, ctrl)
    jax.block_until_ready(dp.qpos)
    dt_phys_ms = (time.perf_counter() - t0) * 1e3 / n_steps

    dc = dx_batch
    t_c0 = time.perf_counter()
    for _ in range(n_warmup):
      dc, rgb, depth = e2e_fn(dc, ctrl)
    jax.block_until_ready((rgb, depth))
    compile_s = time.perf_counter() - t_c0

    t1 = time.perf_counter()
    for _ in range(n_steps):
      dc, rgb, depth = e2e_fn(dc, ctrl)
    jax.block_until_ready((rgb, depth))
    dt_e2e_ms = (time.perf_counter() - t1) * 1e3 / n_steps
    dt_render_ms = max(dt_e2e_ms - dt_phys_ms, 0.0)
    fps = float(batch_size * 1e3 / max(dt_e2e_ms, 1e-9))

    rec.update({
        "status": "ok",
        "dt_physics_ms": round(dt_phys_ms, 3),
        "dt_render_ms": round(dt_render_ms, 3),
        "dt_e2e_ms": round(dt_e2e_ms, 3),
        "e2e_fps": round(fps, 1),
        "mrays_per_sec": round(batch_size * res * res / max(dt_render_ms * 1e-3, 1e-9) / 1e6, 2),
        "compile_s": round(compile_s, 2),
        "hlo_zero_copy": True,
        "hlo_ops": hlo_info,
    })
  except Exception as exc:
    rec.update({"status": "error", "error_type": type(exc).__name__, "error": str(exc)[:500]})
    traceback.print_exc()
  print(f"[W2 1C] G={rec['n_enabled_geoms']} B={batch_size} {res}x{res} {mode} -> {rec.get('status')} "
        f"e2e={rec.get('dt_e2e_ms')}ms phys={rec.get('dt_physics_ms')}ms ren={rec.get('dt_render_ms')}ms fps={rec.get('e2e_fps')}", flush=True)
  return rec


def bench_4c_case(n_prims: int, batch_size: int, res: int, mode: str, n_warmup: int = 2, n_steps: int = 12) -> Dict[str, Any]:
  devs = jax.devices()[:4]
  assert len(devs) == 4
  assert batch_size % 4 == 0
  pallas_interp = (jax.default_backend() == "cpu")
  m, mx, spec, dx_batch, ctrl = _make_warehouse_batch(n_prims, batch_size, res, res)

  mesh = jax.sharding.Mesh(np.array(devs), ("dp",))
  P = jax.sharding.PartitionSpec
  sharding = jax.sharding.NamedSharding(mesh, P("dp"))
  replicate = jax.sharding.NamedSharding(mesh, P())
  dx_batch = jax.device_put(dx_batch, sharding)
  ctrl = jax.device_put(ctrl, sharding)
  mx = jax.device_put(mx, replicate)

  sm_kw = {"check_rep": False} if "check_rep" in inspect.signature(shard_map).parameters else {"check_vma": False}

  rec: Dict[str, Any] = {
      "n_chips": 4,
      "n_target_prims": n_prims,
      "n_enabled_geoms": len(spec.enabled_geom_ids),
      "nq": int(m.nq),
      "nv": int(m.nv),
      "batch_size": batch_size,
      "resolution": f"{res}x{res}",
      "mode": mode,
  }
  try:
    _phys_sm = shard_map(
        lambda d_b, u_b, mx_l: jax.vmap(lambda d, u: mjx.step(mx_l, d.replace(ctrl=u)))(d_b, u_b),
        mesh=mesh, in_specs=(P("dp"), P("dp"), P()), out_specs=P("dp"), **sm_kw,
    )

    @jax.jit
    def phys_only(d_b, u_b):
      return _phys_sm(d_b, u_b, mx)

    def _e2e_local(d_b, u_b, mx_l):
      dn = jax.vmap(lambda d, u: mjx.step(mx_l, d.replace(ctrl=u)))(d_b, u_b)
      if mode == "pallas_v2":
        gb = render_batch(spec, dn, mode="pallas_v2", pallas_interpret=pallas_interp)
      else:
        gb = pallas_raycast_hierarchical(
            spec, dn, interpret=pallas_interp, block_b=8, block_r=1024, cull_mode=mode
        )
      return dn, gb.rgb, gb.depth

    _e2e_sm = shard_map(
        _e2e_local, mesh=mesh, in_specs=(P("dp"), P("dp"), P()), out_specs=(P("dp"), P("dp"), P("dp")), **sm_kw,
    )

    @jax.jit
    def e2e_fn(d_b, u_b):
      return _e2e_sm(d_b, u_b, mx)

    hlo_info = _check_hlo_zero_copy(e2e_fn.lower(dx_batch, ctrl).as_text())
    assert hlo_info["host_callbacks"] == 0
    assert hlo_info["send_recv_ops"] == 0

    dp = dx_batch
    for _ in range(n_warmup):
      dp = phys_only(dp, ctrl)
    jax.block_until_ready(dp.qpos)
    t0 = time.perf_counter()
    for _ in range(n_steps):
      dp = phys_only(dp, ctrl)
    jax.block_until_ready(dp.qpos)
    dt_phys_ms = (time.perf_counter() - t0) * 1e3 / n_steps

    dc = dx_batch
    for _ in range(n_warmup):
      dc, rgb, depth = e2e_fn(dc, ctrl)
    jax.block_until_ready((rgb, depth))
    t1 = time.perf_counter()
    for _ in range(n_steps):
      dc, rgb, depth = e2e_fn(dc, ctrl)
    jax.block_until_ready((rgb, depth))
    dt_e2e_ms = (time.perf_counter() - t1) * 1e3 / n_steps
    dt_render_ms = max(dt_e2e_ms - dt_phys_ms, 0.0)
    fps = float(batch_size * 1e3 / max(dt_e2e_ms, 1e-9))

    rec.update({
        "status": "ok",
        "dt_physics_ms": round(dt_phys_ms, 3),
        "dt_render_ms": round(dt_render_ms, 3),
        "dt_e2e_ms": round(dt_e2e_ms, 3),
        "e2e_fps": round(fps, 1),
        "mrays_per_sec": round(batch_size * res * res / max(dt_render_ms * 1e-3, 1e-9) / 1e6, 2),
        "hlo_zero_copy": True,
        "hlo_ops": hlo_info,
    })
  except Exception as exc:
    rec.update({"status": "error", "error_type": type(exc).__name__, "error": str(exc)[:500]})
    traceback.print_exc()
  print(f"[W2 4C] G={rec['n_enabled_geoms']} B={batch_size} {res}x{res} {mode} -> {rec.get('status')} "
        f"e2e={rec.get('dt_e2e_ms')}ms phys={rec.get('dt_physics_ms')}ms ren={rec.get('dt_render_ms')}ms fps={rec.get('e2e_fps')}", flush=True)
  return rec


def render_showcase_npz(out_npz: str) -> None:
  """Render crisp 256x256 & 512x512 cluttered warehouse frames (G=200 & G=1000) on live w1 TPU."""
  pallas_interp = (jax.default_backend() == "cpu")
  arrays = {}
  for n_prims, res in ((200, 256), (1000, 256), (1000, 512)):
    for cam_name in ("overview", "closeup"):
      if res == 512 and cam_name == "closeup":
        continue
      m, mx, spec, dx_batch, ctrl = _make_warehouse_batch(n_prims, batch_size=8, height=res, width=res, cam_name=cam_name)
      dx_batch = jax.device_put(dx_batch, jax.devices()[0])
      ctrl = jax.device_put(ctrl, jax.devices()[0])
      step_fn = jax.jit(lambda d, u: jax.vmap(lambda di, ui: mjx.step(mx, di.replace(ctrl=ui)))(d, u))
      for _ in range(10):
        dx_batch = step_fn(dx_batch, ctrl)
      jax.block_until_ready(dx_batch.geom_xpos)
      gb = jax.jit(lambda d: pallas_raycast_hierarchical(
          spec, d, interpret=pallas_interp, block_b=8, block_r=1024, cull_mode="hier2"
      ))(dx_batch)
      jax.block_until_ready((gb.rgb, gb.depth, gb.geom_id))
      key = f"g{n_prims}_{res}_{cam_name}"
      arrays[f"{key}_rgb"] = np.asarray(gb.rgb[0])
      arrays[f"{key}_depth"] = np.asarray(gb.depth[0])
      arrays[f"{key}_gid"] = np.asarray(gb.geom_id[0])
      print(f"[SHOWCASE] rendered {key} shape={arrays[f'{key}_rgb'].shape}", flush=True)
  os.makedirs(os.path.dirname(out_npz) or ".", exist_ok=True)
  np.savez_compressed(out_npz, **arrays)


def _load_or_init(path: str) -> Dict[str, Any]:
  if os.path.exists(path):
    with open(path, "r") as f:
      return json.load(f)
  return {
      "backend": jax.default_backend(),
      "device_kind": jax.devices()[0].device_kind,
      "parity_checks": [],
      "runs_1c": [],
      "runs_4c": [],
  }


def _save(path: str, data: Dict[str, Any]) -> None:
  os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
  with open(path, "w") as f:
    json.dump(data, f, indent=2)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--phase", required=True, choices=["parity", "bench_1c", "bench_4c", "showcase"])
  ap.add_argument("--out_json", required=True)
  ap.add_argument("--out_npz", default="/dev/shm/zaychik-tpu-mjx-w1/warehouse_showcase.npz")
  ap.add_argument("--prims", type=int, nargs="+", default=[50, 200, 1000])
  ap.add_argument("--res_list", type=int, nargs="+", default=[64, 128, 256])
  ap.add_argument("--modes", type=str, nargs="+", default=["flat_loop", "cluster1", "hier2"])
  args = ap.parse_args()

  data = _load_or_init(args.out_json)

  if args.phase == "parity":
    data["parity_checks"] = run_parity_wave2()
    _save(args.out_json, data)

  elif args.phase == "bench_1c":
    batch_by_res = {64: 1024, 128: 256, 256: 64}
    done = {(r["n_target_prims"], r["resolution"], r["mode"]) for r in data.get("runs_1c", []) if r.get("status") == "ok"}
    for n_p in args.prims:
      for res in args.res_list:
        B = batch_by_res[res]
        for mode in args.modes:
          if mode == "pallas_v2" and n_p > 50:
            continue
          if (n_p, f"{res}x{res}", mode) in done:
            continue
          rec = bench_1c_case(n_p, B, res, mode)
          data["runs_1c"].append(rec)
          _save(args.out_json, data)

  elif args.phase == "bench_4c":
    batch_by_res = {64: 1024, 128: 256, 256: 64}
    done = {(r["n_target_prims"], r["resolution"], r["mode"]) for r in data.get("runs_4c", []) if r.get("status") == "ok"}
    for n_p in args.prims:
      for res in args.res_list:
        B = batch_by_res[res]
        for mode in args.modes:
          if mode == "pallas_v2" and n_p > 50:
            continue
          if (n_p, f"{res}x{res}", mode) in done:
            continue
          rec = bench_4c_case(n_p, B, res, mode)
          data["runs_4c"].append(rec)
          _save(args.out_json, data)

  elif args.phase == "showcase":
    render_showcase_npz(args.out_npz)


if __name__ == "__main__":
  main()
