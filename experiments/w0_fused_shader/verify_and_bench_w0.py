"""Verification, HLO audit, and E2E benchmark for V3 Fused In-VMEM Shader on w0 TPU v5p."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import re
import time
from typing import Any, Dict, List

import jax
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
import numpy as np

from fused_v3 import pallas_raycast_fused_v3
from tpu_mjx_render.pallas_v2 import pallas_raycast_batch_v2
from tpu_mjx_render.renderer import RenderSceneSpec, render_batch
from tpu_mjx_render.roofline import inspect_hlo_ops


def _check_hlo(lowered) -> Dict[str, Any]:
  txt = lowered.as_text()
  info = inspect_hlo_ops(txt)
  info["send_recv_ops"] = len(re.findall(r"\bsend\(|\brecv\(", txt, re.I))
  assert info["host_callbacks"] == 0 and info["send_recv_ops"] == 0, info
  info["zero_copy_verified"] = True
  return info


def run_local_cpu_check(block_r: int = 512, out_dtype_str: str = "float32") -> Dict[str, Any]:
  from tests.test_v2_pallas_and_assets import _make_synthetic_all_prims_batch

  out_dtype = jnp.bfloat16 if out_dtype_str == "bfloat16" else jnp.float32
  spec, d_batch = _make_synthetic_all_prims_batch(B=16, H=64, W=64)
  fn_pv2 = jax.jit(lambda d: pallas_raycast_batch_v2(spec, d, interpret=True))
  fn_v3 = jax.jit(lambda d: pallas_raycast_fused_v3(spec, d, interpret=True, block_r=block_r, out_dtype=out_dtype))

  gb_v2 = fn_pv2(d_batch)
  f_v3 = fn_v3(d_batch)

  depth_diff = jnp.abs(gb_v2.depth - f_v3.depth.astype(jnp.float32))
  rgb_diff = jnp.abs(gb_v2.rgb - f_v3.rgb.astype(jnp.float32))
  res = {
      "backend": jax.default_backend(),
      "scene": "synthetic_all_prims",
      "B": 16,
      "res": "64x64",
      "block_r": block_r,
      "out_dtype": out_dtype_str,
      "depth_mae_m": float(jnp.mean(depth_diff)),
      "max_depth_diff_m": float(jnp.max(depth_diff)),
      "max_rgb_diff": float(jnp.max(rgb_diff)),
      "mean_rgb_diff": float(jnp.mean(rgb_diff)),
      "hlo_v3": _check_hlo(fn_v3.lower(d_batch)),
  }
  assert res["depth_mae_m"] < 1e-4 and res["max_rgb_diff"] < 5e-4, res
  return res


def run_parity_suite(block_r: int = 512, out_dtype_str: str = "float32") -> Dict[str, Any]:
  import mujoco
  from mujoco import mjx
  from tpu_mjx_render.scenes import get_scene_xml

  out_dtype = jnp.bfloat16 if out_dtype_str == "bfloat16" else jnp.float32
  interpret = (jax.default_backend() == "cpu")
  dev0 = jax.devices()[0]
  results: Dict[str, Any] = {}

  for scene_name, use_shadows in [
      ("panda_pick_cube", True),
      ("panda_pick_cube", False),
      ("all_primitives", True),
  ]:
    m = mujoco.MjModel.from_xml_string(get_scene_xml(scene_name))
    mx = mjx.put_model(m, impl="jax")
    spec = RenderSceneSpec.from_mjmodel(m, cam_id=0, height=64, width=64, use_shadows=use_shadows)

    if m.nv == 0:
      d_host = mujoco.MjData(m)
      mujoco.mj_forward(m, d_host)
      dx0 = mjx.put_data(m, d_host, impl="jax")
    else:
      dx0 = mjx.make_data(m, impl="jax")
      if m.nkey > 0:
        d_host = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d_host, 0)
        dx0 = dx0.replace(qpos=jnp.asarray(d_host.qpos, jnp.float32))
      dx0 = mjx.forward(mx, dx0)

    # Create B=16 diverse states by stepping with small sinusoidal controls
    B = 16
    dx_b = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x[None, ...], (B, *x.shape)).copy(), dx0)
    if m.nu > 0 and m.nv > 0:
      u = jnp.linspace(-0.2, 0.2, B, dtype=jnp.float32)[:, None] * jnp.ones((1, m.nu), dtype=jnp.float32)
      dx_b = jax.jit(lambda d, ctrl: jax.vmap(lambda di, ui: mjx.step(mx, di.replace(ctrl=ui)))(d, ctrl))(dx_b, u)
    dx_b = jax.device_put(dx_b, dev0)

    fn_pv2 = jax.jit(lambda d: pallas_raycast_batch_v2(spec, d, interpret=interpret))
    fn_v3 = jax.jit(
        lambda d: pallas_raycast_fused_v3(spec, d, interpret=interpret, block_r=block_r, out_dtype=out_dtype)
    )

    gb_v2 = fn_pv2(dx_b)
    f_v3 = fn_v3(dx_b)
    jax.block_until_ready((gb_v2.rgb, gb_v2.depth, f_v3.rgb, f_v3.depth))

    d_diff = jnp.abs(gb_v2.depth - f_v3.depth.astype(jnp.float32))
    r_diff = jnp.abs(gb_v2.rgb - f_v3.rgb.astype(jnp.float32))

    hlo_v3 = _check_hlo(fn_v3.lower(dx_b))
    key = f"{scene_name}_shadows_{use_shadows}"
    rec = {
        "scene": scene_name,
        "B": B,
        "res": "64x64",
        "use_shadows": use_shadows,
        "interpret": interpret,
        "block_r": block_r,
        "out_dtype": out_dtype_str,
        "depth_mae_m": float(jnp.mean(d_diff)),
        "max_depth_diff_m": float(jnp.max(d_diff)),
        "max_rgb_diff": float(jnp.max(r_diff)),
        "mean_rgb_diff": float(jnp.mean(r_diff)),
        "hlo_v3": hlo_v3,
    }
    print(f"PARITY [{key}]:", json.dumps(rec), flush=True)
    if out_dtype_str == "float32":
      assert rec["max_rgb_diff"] < 5e-4 and rec["depth_mae_m"] < 1e-4, rec
    results[key] = rec

    # Also verify dynamic geom_rgba_batch parity on panda_pick_cube with shadows=True
    if scene_name == "panda_pick_cube" and use_shadows:
      rng = np.random.default_rng(42)
      rgba_np = np.broadcast_to(spec.geom_rgba[None, :, :], (B, m.ngeom, 4)).copy()
      rgba_np[:, 1:, :3] = np.clip(rgba_np[:, 1:, :3] * rng.uniform(0.7, 1.3, size=(B, m.ngeom - 1, 3)), 0.05, 1.0)
      rgba_dyn = jax.device_put(jnp.asarray(rgba_np, jnp.float32), dev0)
      fn_pv2_dyn = jax.jit(lambda d, rg: pallas_raycast_batch_v2(spec, d, geom_rgba_batch=rg, interpret=interpret))
      fn_v3_dyn = jax.jit(
          lambda d, rg: pallas_raycast_fused_v3(
              spec, d, geom_rgba_batch=rg, interpret=interpret, block_r=block_r, out_dtype=out_dtype
          )
      )
      gb_v2_d = fn_pv2_dyn(dx_b, rgba_dyn)
      f_v3_d = fn_v3_dyn(dx_b, rgba_dyn)
      jax.block_until_ready((gb_v2_d.rgb, f_v3_d.rgb))
      rec_dyn = {
          "scene": "panda_pick_cube_dynamic_rgba",
          "B": B,
          "res": "64x64",
          "use_shadows": True,
          "depth_mae_m": float(jnp.mean(jnp.abs(gb_v2_d.depth - f_v3_d.depth.astype(jnp.float32)))),
          "max_depth_diff_m": float(jnp.max(jnp.abs(gb_v2_d.depth - f_v3_d.depth.astype(jnp.float32)))),
          "max_rgb_diff": float(jnp.max(jnp.abs(gb_v2_d.rgb - f_v3_d.rgb.astype(jnp.float32)))),
          "mean_rgb_diff": float(jnp.mean(jnp.abs(gb_v2_d.rgb - f_v3_d.rgb.astype(jnp.float32)))),
          "hlo_v3": _check_hlo(fn_v3_dyn.lower(dx_b, rgba_dyn)),
      }
      print("PARITY [dynamic_rgba]:", json.dumps(rec_dyn), flush=True)
      if out_dtype_str == "float32":
        assert rec_dyn["max_rgb_diff"] < 5e-4 and rec_dyn["depth_mae_m"] < 1e-4, rec_dyn
      results["panda_pick_cube_dynamic_rgba"] = rec_dyn

  return results


def benchmark_e2e_v3_vs_v2(
    mode: str,
    batch_size: int,
    height: int,
    width: int,
    n_chips: int = 1,
    n_warmup: int = 3,
    n_steps: int = 20,
    use_shadows: bool = True,
    block_r: int = 512,
    out_dtype_str: str = "float32",
) -> Dict[str, Any]:
  import mujoco
  from mujoco import mjx
  from tpu_mjx_render.scenes import get_scene_xml

  out_dtype = jnp.bfloat16 if out_dtype_str == "bfloat16" else jnp.float32
  m = mujoco.MjModel.from_xml_string(get_scene_xml("panda_pick_cube"))
  mx = mjx.put_model(m, impl="jax")
  spec = RenderSceneSpec.from_mjmodel(m, cam_id=0, height=height, width=width, use_shadows=use_shadows)

  devs = jax.devices()[:n_chips]
  if n_chips > 1:
    assert batch_size % n_chips == 0
    mesh = jax.sharding.Mesh(np.array(devs), ("dp",))
    P = jax.sharding.PartitionSpec
    sharding = jax.sharding.NamedSharding(mesh, P("dp"))
    replicate = jax.sharding.NamedSharding(mesh, P())
  else:
    mesh = None
    P = None
    sharding = None
    replicate = None

  dx0 = mjx.make_data(m, impl="jax")
  if m.nkey > 0:
    d_host = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d_host, 0)
    dx0 = dx0.replace(qpos=jnp.asarray(d_host.qpos, jnp.float32))
  dx0 = mjx.forward(mx, dx0)

  dx_batch = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x[None, ...], (batch_size, *x.shape)).copy(), dx0)
  ctrl = jnp.zeros((batch_size, m.nu), dtype=jnp.float32)

  if n_chips > 1:
    dx_batch = jax.device_put(dx_batch, sharding)
    ctrl = jax.device_put(ctrl, sharding)
    mx = jax.device_put(mx, replicate)
  else:
    dx_batch = jax.device_put(dx_batch, devs[0])
    ctrl = jax.device_put(ctrl, devs[0])

  pallas_interp = (jax.default_backend() == "cpu")

  def _render_fn(d_next):
    if mode == "pallas_v2":
      gb = render_batch(spec, d_next, mode="pallas_v2", pallas_interpret=pallas_interp)
      return gb.rgb, gb.depth
    elif mode == "pallas_v3_fused":
      fr = pallas_raycast_fused_v3(
          spec, d_next, interpret=pallas_interp, block_r=block_r, out_dtype=out_dtype
      )
      return fr.rgb, fr.depth
    else:
      raise ValueError(f"Unknown mode: {mode}")

  if n_chips > 1:
    sm_kw = {"check_rep": False} if "check_rep" in inspect.signature(shard_map).parameters else {"check_vma": False}

    def _phys_local(d_b, u_b, mx_l):
      return jax.vmap(lambda d, u: mjx.step(mx_l, d.replace(ctrl=u)))(d_b, u_b)

    _phys_sm = shard_map(_phys_local, mesh=mesh, in_specs=(P("dp"), P("dp"), P()), out_specs=P("dp"), **sm_kw)

    @jax.jit
    def physics_only(d_b, u_b):
      return _phys_sm(d_b, u_b, mx)

    def _e2e_local(d_b, u_b, mx_l):
      d_next = jax.vmap(lambda d, u: mjx.step(mx_l, d.replace(ctrl=u)))(d_b, u_b)
      rgb, depth = _render_fn(d_next)
      return d_next, rgb, depth

    _e2e_sm = shard_map(
        _e2e_local,
        mesh=mesh,
        in_specs=(P("dp"), P("dp"), P()),
        out_specs=(P("dp"), P("dp"), P("dp")),
        **sm_kw,
    )

    @jax.jit
    def e2e_step_render(d_b, u_b):
      return _e2e_sm(d_b, u_b, mx)
  else:
    @jax.jit
    def physics_only(d_b, u_b):
      return jax.vmap(lambda d, u: mjx.step(mx, d.replace(ctrl=u)))(d_b, u_b)

    @jax.jit
    def e2e_step_render(d_b, u_b):
      d_next = jax.vmap(lambda d, u: mjx.step(mx, d.replace(ctrl=u)))(d_b, u_b)
      rgb, depth = _render_fn(d_next)
      return d_next, rgb, depth

  hlo_info = _check_hlo(e2e_step_render.lower(dx_batch, ctrl))

  d_cur = dx_batch
  for _ in range(n_warmup):
    d_cur, rgb, depth = e2e_step_render(d_cur, ctrl)
  jax.block_until_ready((rgb, depth))

  d_p = dx_batch
  for _ in range(n_warmup):
    d_p = physics_only(d_p, ctrl)
  jax.block_until_ready(d_p.qpos)

  t0 = time.perf_counter()
  for _ in range(n_steps):
    d_p = physics_only(d_p, ctrl)
  jax.block_until_ready(d_p.qpos)
  dt_phys = (time.perf_counter() - t0) / n_steps

  t1 = time.perf_counter()
  for _ in range(n_steps):
    d_cur, rgb, depth = e2e_step_render(d_cur, ctrl)
  jax.block_until_ready((rgb, depth))
  dt_e2e = (time.perf_counter() - t1) / n_steps

  dt_render = max(dt_e2e - dt_phys, 0.0)
  fps_e2e = float(batch_size / max(dt_e2e, 1e-9))

  rec = {
      "scene": "panda_pick_cube",
      "backend": jax.default_backend(),
      "device_kind": devs[0].device_kind,
      "n_chips": n_chips,
      "batch_size": batch_size,
      "resolution": f"{height}x{width}",
      "mode": mode,
      "block_r": block_r if mode == "pallas_v3_fused" else 512,
      "out_dtype": out_dtype_str if mode == "pallas_v3_fused" else "float32",
      "use_shadows": use_shadows,
      "dt_physics_ms": dt_phys * 1e3,
      "dt_render_ms": dt_render * 1e3,
      "dt_e2e_ms": dt_e2e * 1e3,
      "e2e_fps": fps_e2e,
      "hlo_zero_copy": hlo_info["host_callbacks"] == 0 and hlo_info["send_recv_ops"] == 0,
      "hlo_ops": hlo_info,
  }
  print("BENCH:", json.dumps(rec), flush=True)
  return rec


def _save_payload(path: str, payload: Dict[str, Any]) -> None:
  os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
  tmp = path + ".tmp"
  with open(tmp, "w") as f:
    json.dump(payload, f, indent=2)
  os.replace(tmp, path)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument(
      "--phase",
      required=True,
      choices=["local_cpu", "parity", "bench_1c_64", "bench_1c_128", "bench_4c", "ablation_1c"],
  )
  ap.add_argument("--block_r", type=int, default=512)
  ap.add_argument("--out_dtype", default="float32", choices=["float32", "bfloat16"])
  ap.add_argument("--batches", type=int, nargs="+", default=None)
  ap.add_argument("--out_json", required=True)
  args = ap.parse_args()

  if os.path.exists(args.out_json):
    with open(args.out_json, "r") as f:
      payload = json.load(f)
  else:
    payload = {}

  if args.phase == "local_cpu":
    payload["local_cpu_check"] = run_local_cpu_check(block_r=args.block_r, out_dtype_str=args.out_dtype)
    _save_payload(args.out_json, payload)
  elif args.phase == "parity":
    payload["hardware"] = {
        "backend": jax.default_backend(),
        "device_count": jax.device_count(),
        "device_kind": jax.devices()[0].device_kind,
    }
    payload["parity"] = run_parity_suite(block_r=args.block_r, out_dtype_str=args.out_dtype)
    _save_payload(args.out_json, payload)
  elif args.phase == "bench_1c_64":
    runs_1c = [r for r in payload.get("bench_1c", []) if r.get("resolution") != "64x64"]
    for B in [1024, 2048, 4096]:
      for mode in ["pallas_v2", "pallas_v3_fused"]:
        runs_1c.append(
            benchmark_e2e_v3_vs_v2(
                mode=mode, batch_size=B, height=64, width=64, n_chips=1,
                block_r=args.block_r, out_dtype_str=args.out_dtype,
            )
        )
        payload["bench_1c"] = runs_1c
        _save_payload(args.out_json, payload)
  elif args.phase == "bench_1c_128":
    runs_1c = [r for r in payload.get("bench_1c", []) if r.get("resolution") != "128x128"]
    for B in [1024, 2048]:
      for mode in ["pallas_v2", "pallas_v3_fused"]:
        runs_1c.append(
            benchmark_e2e_v3_vs_v2(
                mode=mode, batch_size=B, height=128, width=128, n_chips=1,
                block_r=args.block_r, out_dtype_str=args.out_dtype,
            )
        )
        payload["bench_1c"] = runs_1c
        _save_payload(args.out_json, payload)
  elif args.phase == "bench_4c":
    batches = args.batches if args.batches is not None else [1024, 2048, 4096]
    runs_4c = [r for r in payload.get("bench_4c", []) if r.get("batch_size") not in batches]
    for B in batches:
      for mode in ["pallas_v2", "pallas_v3_fused"]:
        runs_4c.append(
            benchmark_e2e_v3_vs_v2(
                mode=mode, batch_size=B, height=64, width=64, n_chips=4,
                block_r=args.block_r, out_dtype_str=args.out_dtype,
            )
        )
        payload["bench_4c"] = sorted(runs_4c, key=lambda r: (r["batch_size"], r["mode"]))
        _save_payload(args.out_json, payload)
  elif args.phase == "ablation_1c":
    abl_key = f"ablation_r{args.block_r}_{args.out_dtype}"
    runs = []
    for B in [1024, 2048, 4096]:
      runs.append(
          benchmark_e2e_v3_vs_v2(
              mode="pallas_v3_fused", batch_size=B, height=64, width=64, n_chips=1,
              block_r=args.block_r, out_dtype_str=args.out_dtype,
          )
      )
      payload[abl_key] = runs
      _save_payload(args.out_json, payload)

  print(f"Saved {args.out_json}", flush=True)


if __name__ == "__main__":
  main()
