"""V3 Tile Geometry, Grid Order, Culling & Packing Ablation Suite on Live TPU v5p (w1)."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import re
import time
import traceback
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import jax
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np

from tpu_mjx_render.pallas_v2 import _compute_geom_bound_sq, build_local_camera_rays
from tpu_mjx_render.primitives import GEOM_PLANE
from tpu_mjx_render.renderer import RenderSceneSpec, render_batch
from tpu_mjx_render.roofline import inspect_hlo_ops
from tpu_mjx_render.scenes import get_scene_xml

from v3_pallas import pallas_raycast_batch_v3


def _make_mjx_batch(scene_name: str, batch_size: int, height: int = 64, width: int = 64):
  m = mujoco.MjModel.from_xml_string(get_scene_xml(scene_name))
  mx = mjx.put_model(m, impl="jax")
  spec = RenderSceneSpec.from_mjmodel(m, cam_id=0, height=height, width=width, use_shadows=True)
  dx0 = mjx.make_data(m, impl="jax")
  if m.nkey > 0:
    d_host = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d_host, 0)
    dx0 = dx0.replace(qpos=jnp.asarray(d_host.qpos, jnp.float32))
  dx0 = mjx.forward(mx, dx0)
  dx_batch = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x[None, ...], (batch_size, *x.shape)).copy(), dx0)
  ctrl = jnp.zeros((batch_size, m.nu), dtype=jnp.float32)
  return m, mx, spec, dx_batch, ctrl


def _check_hlo_zero_copy(hlo_text: str) -> Dict[str, int]:
  info = inspect_hlo_ops(hlo_text)
  info["send_recv_ops"] = len(re.findall(r"\bsend\b|\brecv\b", hlo_text))
  return info


def benchmark_config_1c(
    spec: RenderSceneSpec,
    mx: Any,
    dx_batch: Any,
    ctrl: Any,
    batch_size: int,
    block_b: int,
    block_r: int,
    grid_order: str,
    cull_mode: str = "none",
    geom_pack: str = "v2_128",
    fast_math: str = "none",
    use_v2_baseline: bool = False,
    n_warmup: int = 3,
    n_steps: int = 20,
    dt_phys_ms: Optional[float] = None,
) -> Dict[str, Any]:
  pallas_interp = (jax.default_backend() == "cpu")
  rec: Dict[str, Any] = {
      "label": "v2_2_baseline" if use_v2_baseline else f"b{block_b}_r{block_r}_{grid_order}_{cull_mode}_{geom_pack}_{fast_math}",
      "batch_size": batch_size,
      "n_chips": 1,
      "block_b": block_b,
      "block_r": block_r,
      "grid_order": grid_order,
      "cull_mode": cull_mode,
      "geom_pack": geom_pack,
      "fast_math": fast_math,
      "use_v2_baseline": use_v2_baseline,
  }
  try:
    @jax.jit
    def physics_only(d_b, u_b):
      return jax.vmap(lambda d, u: mjx.step(mx, d.replace(ctrl=u)))(d_b, u_b)

    @jax.jit
    def e2e_step_render(d_b, u_b):
      d_next = jax.vmap(lambda d, u: mjx.step(mx, d.replace(ctrl=u)))(d_b, u_b)
      if use_v2_baseline:
        gb = render_batch(spec, d_next, mode="pallas_v2", pallas_interpret=pallas_interp)
      else:
        gb = pallas_raycast_batch_v3(
            spec,
            d_next,
            interpret=pallas_interp,
            block_b=block_b,
            block_r=block_r,
            grid_order=grid_order,
            cull_mode=cull_mode,
            geom_pack=geom_pack,
            fast_math=fast_math,
        )
      return d_next, gb.rgb, gb.depth

    hlo_text = e2e_step_render.lower(dx_batch, ctrl).as_text()
    hlo_info = _check_hlo_zero_copy(hlo_text)
    assert hlo_info["host_callbacks"] == 0, f"HLO host_callbacks={hlo_info['host_callbacks']}"
    assert hlo_info["send_recv_ops"] == 0, f"HLO send_recv_ops={hlo_info['send_recv_ops']}"

    if dt_phys_ms is None:
      dp = dx_batch
      for _ in range(n_warmup):
        dp = physics_only(dp, ctrl)
      jax.block_until_ready(dp.qpos)
      t0 = time.perf_counter()
      for _ in range(n_steps):
        dp = physics_only(dp, ctrl)
      jax.block_until_ready(dp.qpos)
      dt_phys_ms = (time.perf_counter() - t0) * 1e3 / n_steps

    dc = dx_batch
    t_comp0 = time.perf_counter()
    for _ in range(n_warmup):
      dc, rgb, depth = e2e_step_render(dc, ctrl)
    jax.block_until_ready((rgb, depth))
    warmup_s = time.perf_counter() - t_comp0

    t1 = time.perf_counter()
    for _ in range(n_steps):
      dc, rgb, depth = e2e_step_render(dc, ctrl)
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
        "warmup_compile_s": round(warmup_s, 2),
        "hlo_zero_copy": True,
        "hlo_ops": hlo_info,
    })
  except Exception as exc:
    rec.update({
        "status": "error",
        "error_type": type(exc).__name__,
        "error": str(exc)[:600],
    })
    traceback.print_exc()
  print(f"[1C] B={batch_size} {rec['label']} -> {rec.get('status')} "
        f"e2e={rec.get('dt_e2e_ms')}ms render={rec.get('dt_render_ms')}ms fps={rec.get('e2e_fps')}", flush=True)
  return rec


def benchmark_config_4c(
    batch_size: int,
    block_b: int,
    block_r: int,
    grid_order: str,
    cull_mode: str,
    geom_pack: str,
    fast_math: str,
    use_v2_baseline: bool = False,
    n_warmup: int = 3,
    n_steps: int = 20,
) -> Dict[str, Any]:
  devs = jax.devices()[:4]
  assert len(devs) == 4, f"Expected 4 TPU chips, got {len(devs)}: {devs}"
  assert batch_size % 4 == 0
  m, mx, spec, dx_batch, ctrl = _make_mjx_batch("panda_pick_cube", batch_size, 64, 64)

  mesh = jax.sharding.Mesh(np.array(devs), ("dp",))
  P = jax.sharding.PartitionSpec
  sharding = jax.sharding.NamedSharding(mesh, P("dp"))
  replicate = jax.sharding.NamedSharding(mesh, P())
  dx_batch = jax.device_put(dx_batch, sharding)
  ctrl = jax.device_put(ctrl, sharding)
  mx = jax.device_put(mx, replicate)

  pallas_interp = (jax.default_backend() == "cpu")
  sm_kw = {"check_rep": False} if "check_rep" in inspect.signature(shard_map).parameters else {"check_vma": False}

  def _phys_local(d_b, u_b, mx_l):
    return jax.vmap(lambda d, u: mjx.step(mx_l, d.replace(ctrl=u)))(d_b, u_b)

  _phys_sm = shard_map(_phys_local, mesh=mesh, in_specs=(P("dp"), P("dp"), P()), out_specs=P("dp"), **sm_kw)

  @jax.jit
  def physics_only(d_b, u_b):
    return _phys_sm(d_b, u_b, mx)

  def _e2e_local(d_b, u_b, mx_l):
    d_next = jax.vmap(lambda d, u: mjx.step(mx_l, d.replace(ctrl=u)))(d_b, u_b)
    if use_v2_baseline:
      gb = render_batch(spec, d_next, mode="pallas_v2", pallas_interpret=pallas_interp)
    else:
      gb = pallas_raycast_batch_v3(
          spec,
          d_next,
          interpret=pallas_interp,
          block_b=block_b,
          block_r=block_r,
          grid_order=grid_order,
          cull_mode=cull_mode,
          geom_pack=geom_pack,
          fast_math=fast_math,
      )
    return d_next, gb.rgb, gb.depth

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

  hlo_text = e2e_step_render.lower(dx_batch, ctrl).as_text()
  hlo_info = _check_hlo_zero_copy(hlo_text)
  assert hlo_info["host_callbacks"] == 0, f"HLO host_callbacks={hlo_info['host_callbacks']}"
  assert hlo_info["send_recv_ops"] == 0, f"HLO send_recv_ops={hlo_info['send_recv_ops']}"

  dp = dx_batch
  for _ in range(n_warmup):
    dp = physics_only(dp, ctrl)
  jax.block_until_ready(dp.qpos)
  t0 = time.perf_counter()
  for _ in range(n_steps):
    dp = physics_only(dp, ctrl)
  jax.block_until_ready(dp.qpos)
  dt_phys_ms = (time.perf_counter() - t0) * 1e3 / n_steps

  dc = dx_batch
  for _ in range(n_warmup):
    dc, rgb, depth = e2e_step_render(dc, ctrl)
  jax.block_until_ready((rgb, depth))
  t1 = time.perf_counter()
  for _ in range(n_steps):
    dc, rgb, depth = e2e_step_render(dc, ctrl)
  jax.block_until_ready((rgb, depth))
  dt_e2e_ms = (time.perf_counter() - t1) * 1e3 / n_steps
  dt_render_ms = max(dt_e2e_ms - dt_phys_ms, 0.0)
  fps = float(batch_size * 1e3 / max(dt_e2e_ms, 1e-9))

  rec = {
      "label": "v2_2_baseline" if use_v2_baseline else f"b{block_b}_r{block_r}_{grid_order}_{cull_mode}_{geom_pack}_{fast_math}",
      "status": "ok",
      "batch_size": batch_size,
      "n_chips": 4,
      "block_b": block_b,
      "block_r": block_r,
      "grid_order": grid_order,
      "cull_mode": cull_mode,
      "geom_pack": geom_pack,
      "fast_math": fast_math,
      "use_v2_baseline": use_v2_baseline,
      "dt_physics_ms": round(dt_phys_ms, 3),
      "dt_render_ms": round(dt_render_ms, 3),
      "dt_e2e_ms": round(dt_e2e_ms, 3),
      "e2e_fps": round(fps, 1),
      "hlo_zero_copy": True,
      "hlo_ops": hlo_info,
  }
  print(f"[4C] B={batch_size} {rec['label']} -> e2e={rec['dt_e2e_ms']}ms render={rec['dt_render_ms']}ms fps={rec['e2e_fps']}", flush=True)
  return rec


def measure_tile_cull_rates(block_sizes: List[Tuple[int, int]], B: int = 32) -> List[Dict[str, Any]]:
  """Compute exact empirical fraction of (tile, non-plane geom) pairs skipped by sphere culling on PandaPickCube."""
  m, mx, spec, dx_batch, _ = _make_mjx_batch("panda_pick_cube", B, 64, 64)
  # Perturb joint angles slightly across B so worlds have realistic pose diversity
  key = jax.random.PRNGKey(0)
  noise = 0.15 * jax.random.normal(key, (B, 7), dtype=jnp.float32)
  qpos = dx_batch.qpos.at[:, :7].add(noise)
  dx0_single = jax.tree_util.tree_map(lambda x: x[0], dx_batch)
  dx_batch = jax.vmap(lambda q: mjx.forward(mx, dx0_single.replace(qpos=q)))(qpos)

  cam_pos = np.asarray(dx_batch.cam_xpos[:, int(spec.cam_id), :])
  cam_mat = np.asarray(dx_batch.cam_xmat[:, int(spec.cam_id), :, :])
  gids = np.asarray(spec.enabled_geom_ids, dtype=np.int32)
  gpos = np.asarray(dx_batch.geom_xpos[:, gids, :])
  rb_sq = _compute_geom_bound_sq(spec, gids)

  ld_np, lo_np, neg_z_np = build_local_camera_rays(spec)
  # Compute world rays (B, R, 3)
  ray_dir = np.einsum("bij,rj->bri", cam_mat, ld_np)
  ray_dir /= np.maximum(np.linalg.norm(ray_dir, axis=-1, keepdims=True), 1e-12)
  ray_off = np.einsum("bij,rj->bri", cam_mat, lo_np)
  inv_cos = (float(spec.znear) / neg_z_np)[None, :, None]
  ray_orig = cam_pos[:, None, :] + ray_off + inv_cos * ray_dir

  non_plane = [k for k, gid in enumerate(gids) if int(spec.geom_type[gid]) != GEOM_PLANE]
  out = []
  for bb, br in block_sizes:
    total_pairs = 0
    culled_line = 0
    culled_front = 0
    for bi in range(0, B, bb):
      ro_b = ray_orig[bi : bi + bb]
      rd_b = ray_dir[bi : bi + bb]
      gp_b = gpos[bi : bi + bb]
      for ri in range(0, 4096, br):
        ro_t = ro_b[:, ri : ri + br, :]
        rd_t = rd_b[:, ri : ri + br, :]
        for k in non_plane:
          d = ro_t - gp_b[:, k : k + 1, :]
          proj = np.sum(d * rd_t, axis=-1)
          d2 = np.sum(d * d, axis=-1)
          d2_perp = d2 - proj * proj
          hit_line = d2_perp <= rb_sq[k]
          hit_front = hit_line & ((proj < 0.0) | (d2 <= rb_sq[k]))
          total_pairs += 1
          if not np.any(hit_line):
            culled_line += 1
          if not np.any(hit_front):
            culled_front += 1
    out.append({
        "block_b": bb,
        "block_r": br,
        "total_nonplane_tile_geom_pairs": total_pairs,
        "culled_line_frac": round(culled_line / max(total_pairs, 1), 4),
        "culled_front_frac": round(culled_front / max(total_pairs, 1), 4),
    })
  return out


class _MockBatch(NamedTuple):
  cam_xpos: jax.Array
  cam_xmat: jax.Array
  geom_xpos: jax.Array
  geom_xmat: jax.Array
  xpos: jax.Array
  xmat: jax.Array


def verify_parity_configs(configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
  """Verify exact numerical parity vs V2.2 pallas_v2 on PandaPickCube (B=16) and all_primitives (B=16)."""
  pallas_interp = (jax.default_backend() == "cpu")
  results = []

  for scene in ("panda_pick_cube", "all_primitives"):
    if scene == "panda_pick_cube":
      m, mx, spec, dx_batch, _ = _make_mjx_batch(scene, 16, 64, 64)
      noise = 0.1 * jax.random.normal(jax.random.PRNGKey(42), (16, 7), dtype=jnp.float32)
      dx0_single = jax.tree_util.tree_map(lambda x: x[0], dx_batch)
      dx_batch = jax.vmap(lambda q: mjx.forward(mx, dx0_single.replace(qpos=q)))(dx_batch.qpos.at[:, :7].add(noise))
    else:
      m = mujoco.MjModel.from_xml_string(get_scene_xml(scene))
      spec = RenderSceneSpec.from_mjmodel(m, cam_id=0, height=64, width=64, use_shadows=True)
      d_host = mujoco.MjData(m)
      mujoco.mj_forward(m, d_host)
      dx_batch = _MockBatch(
          cam_xpos=jnp.broadcast_to(jnp.asarray(d_host.cam_xpos, jnp.float32)[None], (16, m.ncam, 3)),
          cam_xmat=jnp.broadcast_to(jnp.asarray(d_host.cam_xmat.reshape(m.ncam, 3, 3), jnp.float32)[None], (16, m.ncam, 3, 3)),
          geom_xpos=jnp.broadcast_to(jnp.asarray(d_host.geom_xpos, jnp.float32)[None], (16, m.ngeom, 3)),
          geom_xmat=jnp.broadcast_to(jnp.asarray(d_host.geom_xmat.reshape(m.ngeom, 3, 3), jnp.float32)[None], (16, m.ngeom, 3, 3)),
          xpos=jnp.broadcast_to(jnp.asarray(d_host.xpos, jnp.float32)[None], (16, m.nbody, 3)),
          xmat=jnp.broadcast_to(jnp.asarray(d_host.xmat.reshape(m.nbody, 3, 3), jnp.float32)[None], (16, m.nbody, 3, 3)),
      )
    dx_batch = jax.device_put(dx_batch, jax.devices()[0])
    ref_gb = jax.jit(lambda d: render_batch(spec, d, mode="pallas_v2", pallas_interpret=pallas_interp))(dx_batch)
    jax.block_until_ready((ref_gb.rgb, ref_gb.depth, ref_gb.geom_id))

    for cfg in configs:
      fn = jax.jit(lambda d: pallas_raycast_batch_v3(
          spec, d, interpret=pallas_interp,
          block_b=cfg["block_b"], block_r=cfg["block_r"],
          grid_order=cfg["grid_order"], cull_mode=cfg["cull_mode"],
          geom_pack=cfg["geom_pack"], fast_math=cfg["fast_math"],
      ))
      hlo_info = _check_hlo_zero_copy(fn.lower(dx_batch).as_text())
      gb = fn(dx_batch)
      jax.block_until_ready((gb.rgb, gb.depth, gb.geom_id))

      geom_match = float(jnp.mean(gb.geom_id == ref_gb.geom_id))
      depth_mae = float(jnp.mean(jnp.abs(gb.depth - ref_gb.depth)))
      max_depth_diff = float(jnp.max(jnp.abs(gb.depth - ref_gb.depth)))
      max_rgb_diff = float(jnp.max(jnp.abs(gb.rgb - ref_gb.rgb)))
      passed = (
          geom_match == 1.0
          and depth_mae < 1e-5
          and max_rgb_diff < 5e-4
          and hlo_info["host_callbacks"] == 0
          and hlo_info["send_recv_ops"] == 0
      )
      rec = {
          "scene": scene,
          "config": cfg,
          "geom_match": geom_match,
          "depth_mae_m": depth_mae,
          "max_depth_diff_m": max_depth_diff,
          "max_rgb_diff": max_rgb_diff,
          "host_callbacks": hlo_info["host_callbacks"],
          "send_recv_ops": hlo_info["send_recv_ops"],
          "passed": passed,
      }
      print(f"[PARITY] {scene} {cfg} -> pass={passed} geom={geom_match} depth_mae={depth_mae:.3e} rgb_max={max_rgb_diff:.3e}", flush=True)
      results.append(rec)
  return results


def _load_or_init(path: str) -> Dict[str, Any]:
  if os.path.exists(path):
    with open(path, "r") as f:
      return json.load(f)
  return {
      "backend": jax.default_backend(),
      "device_kind": jax.devices()[0].device_kind,
      "tile_grid_sweep_1c": [],
      "cull_ablation_1c": [],
      "cull_rates": [],
      "pack_math_ablation_1c": [],
      "parity_checks": [],
      "best_4c_runs": [],
  }


def _save(path: str, data: Dict[str, Any]):
  os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
  with open(path, "w") as f:
    json.dump(data, f, indent=2)


def _merge_runs(existing: List[Dict[str, Any]], new_runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
  by_key = {(r.get("label"), r.get("batch_size"), r.get("n_chips", 1)): r for r in existing}
  for r in new_runs:
    by_key[(r.get("label"), r.get("batch_size"), r.get("n_chips", 1))] = r
  return list(by_key.values())


def _existing_keys(runs: List[Dict[str, Any]], retry_errors: bool = False) -> set:
  statuses = ("ok",) if retry_errors else ("ok", "error")
  return {(r.get("label"), r.get("batch_size"), r.get("n_chips", 1)) for r in runs if r.get("status") in statuses}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--phase", required=True, choices=["tile_grid", "cull", "pack_math", "parity", "bench_4c"])
  ap.add_argument("--out_json", required=True)
  ap.add_argument("--batches", type=int, nargs="+", default=[1024, 2048, 4096])
  ap.add_argument("--grid_orders", type=str, nargs="+", default=["b_r", "r_b"])
  ap.add_argument("--max_new_runs", type=int, default=999)
  ap.add_argument("--retry_errors", action="store_true")
  ap.add_argument("--best_cfg_json", default=None, help="JSON string overriding best config for parity/4c")
  args = ap.parse_args()

  data = _load_or_init(args.out_json)
  new_count = 0

  if args.phase == "tile_grid":
    tiles = [(8, 256), (8, 512), (8, 1024), (16, 256), (16, 512), (32, 256)]
    done = _existing_keys(data.get("tile_grid_sweep_1c", []), args.retry_errors)
    for B in args.batches:
      if new_count >= args.max_new_runs:
        break
      m, mx, spec, dx_batch, ctrl = _make_mjx_batch("panda_pick_cube", B, 64, 64)
      dx_batch = jax.device_put(dx_batch, jax.devices()[0])
      ctrl = jax.device_put(ctrl, jax.devices()[0])

      dt_phys = None
      for r in data.get("tile_grid_sweep_1c", []):
        if r.get("batch_size") == B and r.get("status") == "ok" and r.get("dt_physics_ms") is not None:
          dt_phys = r["dt_physics_ms"]
          break

      if ("v2_2_baseline", B, 1) not in done and new_count < args.max_new_runs:
        base_rec = benchmark_config_1c(spec, mx, dx_batch, ctrl, B, 8, 512, "b_r", use_v2_baseline=True, dt_phys_ms=dt_phys)
        dt_phys = base_rec.get("dt_physics_ms", dt_phys)
        data["tile_grid_sweep_1c"] = _merge_runs(data.get("tile_grid_sweep_1c", []), [base_rec])
        _save(args.out_json, data)
        done.add(("v2_2_baseline", B, 1))
        new_count += 1

      for bb, br in tiles:
        for go in args.grid_orders:
          if new_count >= args.max_new_runs:
            break
          lbl = f"b{bb}_r{br}_{go}_none_v2_128_none"
          if (lbl, B, 1) in done:
            continue
          rec = benchmark_config_1c(
              spec, mx, dx_batch, ctrl, B, bb, br, go,
              cull_mode="none", geom_pack="v2_128", fast_math="none",
              dt_phys_ms=dt_phys,
          )
          dt_phys = rec.get("dt_physics_ms", dt_phys)
          data["tile_grid_sweep_1c"] = _merge_runs(data.get("tile_grid_sweep_1c", []), [rec])
          _save(args.out_json, data)
          done.add((lbl, B, 1))
          new_count += 1

  elif args.phase == "cull":
    candidates = [(8, 1024, "b_r"), (8, 512, "b_r"), (8, 256, "b_r"), (16, 256, "b_r")]
    if args.batches != [1024]:
      # For B=2048/4096, focus on the winning/key cull tile sizes (8,1024) and (8,512)
      candidates = [(8, 1024, "b_r"), (8, 512, "b_r")]

    if not data.get("cull_rates"):
      data["cull_rates"] = measure_tile_cull_rates([(bb, br) for bb, br, _ in candidates], B=32)
      print("CULL RATES:", data["cull_rates"], flush=True)
      _save(args.out_json, data)

    done = _existing_keys(data.get("cull_ablation_1c", []), args.retry_errors)
    for B in args.batches:
      if new_count >= args.max_new_runs:
        break
      m, mx, spec, dx_batch, ctrl = _make_mjx_batch("panda_pick_cube", B, 64, 64)
      dx_batch = jax.device_put(dx_batch, jax.devices()[0])
      ctrl = jax.device_put(ctrl, jax.devices()[0])
      dt_phys = None
      for r in data.get("tile_grid_sweep_1c", []) + data.get("cull_ablation_1c", []):
        if r.get("batch_size") == B and r.get("status") == "ok" and r.get("dt_physics_ms") is not None:
          dt_phys = r["dt_physics_ms"]
          break
      for bb, br, go in candidates:
        modes = ("none", "sphere_primary", "sphere_both", "frustum_tight") if B == 1024 else ("sphere_primary", "sphere_both")
        for cm in modes:
          if new_count >= args.max_new_runs:
            break
          lbl = f"b{bb}_r{br}_{go}_{cm}_v2_128_none"
          if (lbl, B, 1) in done:
            continue
          rec = benchmark_config_1c(
              spec, mx, dx_batch, ctrl, B, bb, br, go,
              cull_mode=cm, geom_pack="v2_128", fast_math="none",
              dt_phys_ms=dt_phys,
          )
          dt_phys = rec.get("dt_physics_ms", dt_phys)
          data["cull_ablation_1c"] = _merge_runs(data.get("cull_ablation_1c", []), [rec])
          _save(args.out_json, data)
          done.add((lbl, B, 1))
          new_count += 1

  elif args.phase == "pack_math":
    test_cfgs = [
        (8, 1024, "b_r", "sphere_both"),
        (8, 512, "b_r", "none"),
        (32, 256, "b_r", "none"),
        (32, 512, "b_r", "none"),
    ]

    done = _existing_keys(data.get("pack_math_ablation_1c", []), args.retry_errors)
    for B in args.batches:
      if new_count >= args.max_new_runs:
        break
      m, mx, spec, dx_batch, ctrl = _make_mjx_batch("panda_pick_cube", B, 64, 64)
      dx_batch = jax.device_put(dx_batch, jax.devices()[0])
      ctrl = jax.device_put(ctrl, jax.devices()[0])
      dt_phys = None
      for r in data.get("tile_grid_sweep_1c", []) + data.get("pack_math_ablation_1c", []):
        if r.get("batch_size") == B and r.get("status") == "ok" and r.get("dt_physics_ms") is not None:
          dt_phys = r["dt_physics_ms"]
          break
      for bb, br, go, cm in test_cfgs:
        combos = [("compact_row", "none"), ("v2_128", "rsqrt_no_raynorm"), ("compact_row", "rsqrt_no_raynorm")]
        if (bb, br) == (32, 512):
          combos.insert(0, ("v2_128", "none"))
        for gp, fm in combos:
          if new_count >= args.max_new_runs:
            break
          lbl = f"b{bb}_r{br}_{go}_{cm}_{gp}_{fm}"
          if (lbl, B, 1) in done:
            continue
          rec = benchmark_config_1c(
              spec, mx, dx_batch, ctrl, B, bb, br, go,
              cull_mode=cm, geom_pack=gp, fast_math=fm,
              dt_phys_ms=dt_phys,
          )
          dt_phys = rec.get("dt_physics_ms", dt_phys)
          data["pack_math_ablation_1c"] = _merge_runs(data.get("pack_math_ablation_1c", []), [rec])
          _save(args.out_json, data)
          done.add((lbl, B, 1))
          new_count += 1

  elif args.phase == "parity":
    if args.best_cfg_json:
      cfgs = json.loads(args.best_cfg_json)
    else:
      all_ok = [
          r for r in (
              data.get("tile_grid_sweep_1c", [])
              + data.get("cull_ablation_1c", [])
              + data.get("pack_math_ablation_1c", [])
          )
          if r.get("status") == "ok" and not r.get("use_v2_baseline")
      ]
      # Take top distinct configs by B=4096 or B=1024 FPS
      by_label = {}
      for r in sorted(all_ok, key=lambda x: x["dt_e2e_ms"]):
        k = (r["block_b"], r["block_r"], r["grid_order"], r["cull_mode"], r["geom_pack"], r["fast_math"])
        if k not in by_label:
          by_label[k] = {
              "block_b": r["block_b"],
              "block_r": r["block_r"],
              "grid_order": r["grid_order"],
              "cull_mode": r["cull_mode"],
              "geom_pack": r["geom_pack"],
              "fast_math": r["fast_math"],
          }
      cfgs = list(by_label.values())[:4]
    data["parity_checks"] = verify_parity_configs(cfgs)
    _save(args.out_json, data)

  elif args.phase == "bench_4c":
    if args.best_cfg_json:
      best = json.loads(args.best_cfg_json)
    else:
      # Pick fastest config that passed ALL parity checks
      passed_keys = set()
      by_cfg: Dict[tuple, List[bool]] = {}
      for p in data.get("parity_checks", []):
        c = p["config"]
        k = (c["block_b"], c["block_r"], c["grid_order"], c["cull_mode"], c["geom_pack"], c["fast_math"])
        by_cfg.setdefault(k, []).append(bool(p["passed"]))
      for k, flags in by_cfg.items():
        if all(flags):
          passed_keys.add(k)

      all_ok = [
          r for r in (
              data.get("tile_grid_sweep_1c", [])
              + data.get("cull_ablation_1c", [])
              + data.get("pack_math_ablation_1c", [])
          )
          if r.get("status") == "ok" and not r.get("use_v2_baseline")
          and (r["block_b"], r["block_r"], r["grid_order"], r["cull_mode"], r["geom_pack"], r["fast_math"]) in passed_keys
      ]
      best_r = min(all_ok, key=lambda r: r["dt_e2e_ms"] / r["batch_size"])
      best = {
          "block_b": best_r["block_b"],
          "block_r": best_r["block_r"],
          "grid_order": best_r["grid_order"],
          "cull_mode": best_r["cull_mode"],
          "geom_pack": best_r["geom_pack"],
          "fast_math": best_r["fast_math"],
      }
    print("4C BENCH BEST CONFIG:", best, flush=True)
    runs = []
    for B in (1024, 2048, 4096):
      runs.append(benchmark_config_4c(B, 8, 512, "b_r", "none", "v2_128", "none", use_v2_baseline=True))
      runs.append(benchmark_config_4c(
          B,
          best["block_b"],
          best["block_r"],
          best["grid_order"],
          best["cull_mode"],
          best["geom_pack"],
          best["fast_math"],
          use_v2_baseline=False,
      ))
    data["best_config"] = best
    data["best_4c_runs"] = runs
    _save(args.out_json, data)


if __name__ == "__main__":
  main()
