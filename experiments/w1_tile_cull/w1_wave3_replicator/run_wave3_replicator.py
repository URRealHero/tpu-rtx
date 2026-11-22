"""Wave 3: Isaac Sim TiledCamera + Omniverse Replicator DR Benchmarks on w1 TPU v5p."""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from typing import Any, Dict, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np

from tpu_mjx_render.adapters.isaac_replicator import (
    DEFAULT_MODALITIES,
    ReplicatorCfg,
    TiledCamera,
    TiledCameraCfg,
    audit_replicator_hlo,
    randomize_qpos_batch,
    rep,
    tile_grid_mosaic,
)
from tpu_mjx_render.renderer import RenderSceneSpec, render_batch

from experiments.w1_tile_cull.w1_wave3_replicator.scenes_wave3 import (
    assign_robodojo_semantic,
    assign_robotwin_semantic,
    make_robodojo_cluttered_xml,
    make_robotwin_bimanual_xml,
)


def _make_rep_cfg(target_center: Tuple[float, float, float], pose_cfg=None) -> ReplicatorCfg:
  return ReplicatorCfg(
      light=rep.randomizer.light(
          azimuth_deg=(0.0, 360.0),
          elevation_deg=(30.0, 74.0),
          radius=(1.8, 3.0),
          target_center=target_center,
          kelvin_range=(2800.0, 8200.0),
          intensity_range=(0.95, 1.50),
          ambient_scale=(0.15, 0.28),
          exposure_range=(1.02, 1.24),
      ),
      materials=rep.randomizer.materials(
          randomize_objects=True,
          randomize_table=True,
          randomize_floor=True,
          randomize_robot=True,
      ),
      camera=rep.randomizer.camera(
          pos_jitter_xyz=(0.03, 0.03, 0.02),
          rot_jitter_deg=(2.5, 2.5, 3.5),
          fovy_jitter_deg=5.5,
      ),
      poses=pose_cfg,
  )


def _prepare_robotwin_batch(batch_size: int, n_step_warmup: int = 3):
  xml = make_robotwin_bimanual_xml()
  m = mujoco.MjModel.from_xml_string(xml)
  mx = mjx.put_model(m, impl="jax")
  d_host = mujoco.MjData(m)
  mujoco.mj_resetDataKeyframe(m, d_host, 0)
  mujoco.mj_forward(m, d_host)
  dx0 = mjx.put_data(m, d_host, impl="jax")
  dx_b = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x[None, ...], (batch_size, *x.shape)).copy(), dx0)

  t_phase = np.linspace(-0.22, 0.22, batch_size, dtype=np.float32)[:, None]
  ctrl_np = np.asarray(dx0.ctrl, np.float32)[None, :] + t_phase * np.ones((1, m.nu), np.float32)
  ctrl = jnp.asarray(ctrl_np, dtype=jnp.float32)

  # Step briefly on device 0 so cvel is non-zero for motion_vectors verification
  if n_step_warmup > 0:
    dev0 = jax.devices()[0]
    dx_b = jax.device_put(dx_b, dev0)
    ctrl_0 = jax.device_put(ctrl, dev0)
    step_fn = jax.jit(lambda d, u: jax.vmap(lambda di, ui: mjx.step(mx, di.replace(ctrl=ui)))(d, u))
    for _ in range(n_step_warmup):
      dx_b = step_fn(dx_b, ctrl_0)
    jax.block_until_ready(dx_b.qpos)
  return m, mx, dx_b, ctrl


def _prepare_robodojo_batch(n_target_prims: int, batch_size: int, n_step_warmup: int = 3):
  xml = make_robodojo_cluttered_xml(n_target_prims=n_target_prims)
  m = mujoco.MjModel.from_xml_string(xml)
  mx = mjx.put_model(m, impl="jax")
  dx0 = mjx.make_data(m, impl="jax")
  dx0 = mjx.forward(mx, dx0)
  dx_b = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x[None, ...], (batch_size, *x.shape)).copy(), dx0)

  ctrl_np = np.zeros((batch_size, m.nu), dtype=np.float32)
  t_phase = np.linspace(0.0, 2.0 * np.pi, batch_size, endpoint=False, dtype=np.float32)
  for j in range(m.nu):
    ctrl_np[:, j] = 0.35 * np.sin(t_phase + 0.5 * j)
  ctrl = jnp.asarray(ctrl_np, dtype=jnp.float32)

  if n_step_warmup > 0:
    dev0 = jax.devices()[0]
    dx_b = jax.device_put(dx_b, dev0)
    ctrl_0 = jax.device_put(ctrl, dev0)
    step_fn = jax.jit(lambda d, u: jax.vmap(lambda di, ui: mjx.step(mx, di.replace(ctrl=ui)))(d, u))
    for _ in range(n_step_warmup):
      dx_b = step_fn(dx_b, ctrl_0)
    jax.block_until_ready(dx_b.qpos)
  return m, mx, dx_b, ctrl


def run_parity_and_hlo_checks() -> List[Dict[str, Any]]:
  """Verify exact geometric parity vs v2_soa + HLO zero-copy across unrolled & hierarchical kernels."""
  pallas_interp = (jax.default_backend() == "cpu")
  dev0 = jax.devices()[0]
  results: List[Dict[str, Any]] = []

  # 1. RoboTwin Unrolled (G <= 128) parity vs v2_soa
  m_rt, mx_rt, dx_rt, ctrl_rt = _prepare_robotwin_batch(batch_size=8, n_step_warmup=4)
  rt_head_cid = mujoco.mj_name2id(m_rt, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")
  spec_rt = RenderSceneSpec.from_mjmodel(m_rt, cam_id=rt_head_cid, height=64, width=64, use_shadows=True)
  ref_rt = jax.jit(lambda d: render_batch(spec_rt, d, mode="v2_soa"))(dx_rt)
  jax.block_until_ready((ref_rt.z_depth, ref_rt.geom_id, ref_rt.normal))

  cam_rt = TiledCamera(
      m_rt,
      TiledCameraCfg(
          cam_names="head_camera",
          height=64,
          width=64,
          data_types=DEFAULT_MODALITIES,
          shader="pbr_ggx",
          tonemap="agx",
          out_dtype=jnp.float32,
          interpret=pallas_interp,
          semantic_fn=assign_robotwin_semantic,
      ),
  )
  fn_rt_static = jax.jit(lambda d: cam_rt.render_local(d, rep_batch=None))
  hlo_rt_static = audit_replicator_hlo(fn_rt_static.lower(dx_rt).as_text())
  obs_rt = fn_rt_static(dx_rt)
  jax.block_until_ready(tuple(obs_rt.values()))

  geom_match = float(jnp.mean(obs_rt["instance_id"] == ref_rt.geom_id))
  depth_mae = float(jnp.mean(jnp.abs(obs_rt["distance_to_image_plane"] - ref_rt.z_depth)))
  normal_mae = float(jnp.mean(jnp.abs(obs_rt["normals"] - ref_rt.normal)))
  passed_rt = (
      geom_match == 1.0
      and depth_mae < 1e-5
      and normal_mae < 1e-4
      and hlo_rt_static["zero_copy_verified"]
  )
  rec_rt = {
      "check": "robotwin_unrolled_vs_v2_soa",
      "n_enabled_geoms": cam_rt.n_enabled_geoms,
      "geom_match": geom_match,
      "depth_mae_m": depth_mae,
      "normal_mae": normal_mae,
      "hlo_zero_copy": hlo_rt_static["zero_copy_verified"],
      "hlo_ops": hlo_rt_static,
      "passed": passed_rt,
  }
  print(f"[PARITY 1] RoboTwin G={cam_rt.n_enabled_geoms} -> pass={passed_rt} "
        f"geom={geom_match} z_mae={depth_mae:.2e} n_mae={normal_mae:.2e} hlo={hlo_rt_static}", flush=True)
  results.append(rec_rt)

  # 2. RoboDojo Hierarchical (G > 128, G=204) parity vs v2_soa
  m_rd, mx_rd, dx_rd, ctrl_rd = _prepare_robodojo_batch(n_target_prims=200, batch_size=8, n_step_warmup=4)
  rd_ov_cid = mujoco.mj_name2id(m_rd, mujoco.mjtObj.mjOBJ_CAMERA, "overview")
  spec_rd = RenderSceneSpec.from_mjmodel(m_rd, cam_id=rd_ov_cid, height=64, width=64, use_shadows=True)
  ref_rd = jax.jit(lambda d: render_batch(spec_rd, d, mode="v2_soa"))(dx_rd)
  jax.block_until_ready((ref_rd.z_depth, ref_rd.geom_id, ref_rd.normal))

  cam_rd = TiledCamera(
      m_rd,
      TiledCameraCfg(
          cam_names="overview",
          height=64,
          width=64,
          data_types=DEFAULT_MODALITIES,
          shader="pbr_ggx",
          tonemap="agx",
          out_dtype=jnp.float32,
          cull_mode="cluster1",
          interpret=pallas_interp,
          semantic_fn=assign_robodojo_semantic,
      ),
  )
  fn_rd_static = jax.jit(lambda d: cam_rd.render_local(d, rep_batch=None))
  hlo_rd_static = audit_replicator_hlo(fn_rd_static.lower(dx_rd).as_text())
  obs_rd = fn_rd_static(dx_rd)
  jax.block_until_ready(tuple(obs_rd.values()))

  geom_match_rd = float(jnp.mean(obs_rd["instance_id"] == ref_rd.geom_id))
  depth_mae_rd = float(jnp.mean(jnp.abs(obs_rd["distance_to_image_plane"] - ref_rd.z_depth)))
  normal_mae_rd = float(jnp.mean(jnp.abs(obs_rd["normals"] - ref_rd.normal)))
  passed_rd = (
      geom_match_rd == 1.0
      and depth_mae_rd < 1e-5
      and normal_mae_rd < 1e-4
      and hlo_rd_static["zero_copy_verified"]
  )
  rec_rd = {
      "check": "robodojo_hier_cluster1_vs_v2_soa",
      "n_enabled_geoms": cam_rd.n_enabled_geoms,
      "geom_match": geom_match_rd,
      "depth_mae_m": depth_mae_rd,
      "normal_mae": normal_mae_rd,
      "hlo_zero_copy": hlo_rd_static["zero_copy_verified"],
      "hlo_ops": hlo_rd_static,
      "passed": passed_rd,
  }
  print(f"[PARITY 2] RoboDojo G={cam_rd.n_enabled_geoms} -> pass={passed_rd} "
        f"geom={geom_match_rd} z_mae={depth_mae_rd:.2e} n_mae={normal_mae_rd:.2e} hlo={hlo_rd_static}", flush=True)
  results.append(rec_rd)

  # 3. Live DR + TiledGrid Mosaic + Multi-Cam Zero-Copy HLO Check
  cam_rt3 = TiledCamera(
      m_rt,
      TiledCameraCfg(
          cam_names=("head_camera", "left_wrist_camera", "right_wrist_camera"),
          height=64,
          width=64,
          data_types=DEFAULT_MODALITIES,
          shader="pbr_ggx",
          tonemap="agx",
          out_dtype=jnp.bfloat16,
          interpret=pallas_interp,
          semantic_fn=assign_robotwin_semantic,
      ),
  )
  rep_cfg_rt = _make_rep_cfg(target_center=(0.60, 0.0, 0.14))
  step_fn_dr = cam_rt3.build_step_and_render_fn(mx_rt, batch_size=8, rep_cfg=rep_cfg_rt, with_dr=True, with_physics=True)
  keys_8 = jax.device_put(jax.random.split(jax.random.PRNGKey(42), 8), dev0)
  hlo_rt3_dr = audit_replicator_hlo(step_fn_dr.lower(dx_rt, ctrl_rt, keys_8).as_text())
  _, obs_rt3 = step_fn_dr(dx_rt, ctrl_rt, keys_8)
  jax.block_until_ready(tuple(obs_rt3.values()))
  # Verify TiledGrid zero-copy canvas stitching
  mosaic = jax.jit(lambda x: tile_grid_mosaic(x[:, 0], (2, 4)))(obs_rt3["rgb"])
  jax.block_until_ready(mosaic)
  assert mosaic.shape == (128, 256, 3), mosaic.shape
  flow_std = float(jnp.std(obs_rt3["motion_vectors"].astype(jnp.float32)))
  rgb_env_diff = float(jnp.mean(jnp.abs(obs_rt3["rgb"][0].astype(jnp.float32) - obs_rt3["rgb"][1].astype(jnp.float32))))
  passed_dr = hlo_rt3_dr["zero_copy_verified"] and (rgb_env_diff > 0.01) and (flow_std > 1e-4)
  rec_dr = {
      "check": "robotwin_3cam_live_dr_and_tiled_mosaic",
      "n_enabled_geoms": cam_rt3.n_enabled_geoms,
      "rgb_shape": list(obs_rt3["rgb"].shape),
      "mosaic_shape": list(mosaic.shape),
      "rgb_env_diversity_l1": rgb_env_diff,
      "motion_vectors_std_px": flow_std,
      "hlo_zero_copy": hlo_rt3_dr["zero_copy_verified"],
      "hlo_ops": hlo_rt3_dr,
      "passed": passed_dr,
  }
  print(f"[PARITY 3] RoboTwin 3-Cam DR -> pass={passed_dr} shape={obs_rt3['rgb'].shape} "
        f"env_div={rgb_env_diff:.3f} flow_std={flow_std:.4f} hlo={hlo_rt3_dr}", flush=True)
  results.append(rec_dr)
  return results


def _bench_case(
    name: str,
    m: Any,
    mx: Any,
    dx_base: Any,
    ctrl_base: Any,
    cam_cfg: TiledCameraCfg,
    rep_cfg: ReplicatorCfg,
    batch_size: int,
    n_chips: int,
    with_dr: bool,
    with_physics: bool = True,
    n_warmup: int = 2,
    n_steps: int = 10,
) -> Dict[str, Any]:
  pallas_interp = (jax.default_backend() == "cpu")
  cfg = dataclasses.replace(cam_cfg, n_chips=n_chips, interpret=pallas_interp)
  cam = TiledCamera(m, cfg)

  # Replicate/slice dx_base & ctrl_base to exact batch_size if needed
  cur_b = int(dx_base.qpos.shape[0])
  if cur_b != batch_size:
    raise ValueError(f"Expected dx_base batch {batch_size}, got {cur_b}")

  dx_s = cam.shard_tree(dx_base)
  ctrl_s = cam.shard_tree(ctrl_base)
  keys_s = cam.shard_tree(jax.random.split(jax.random.PRNGKey(2026 + batch_size), batch_size))

  rec: Dict[str, Any] = {
      "name": name,
      "n_chips": n_chips,
      "n_cameras": len(cam.cam_ids),
      "n_modalities": len(cfg.data_types),
      "data_types": list(cfg.data_types),
      "shader": cfg.shader,
      "n_enabled_geoms": cam.n_enabled_geoms,
      "batch_size": batch_size,
      "resolution": f"{cfg.height}x{cfg.width}",
      "with_dr": with_dr,
      "with_physics": with_physics,
  }
  try:
    step_fn = cam.build_step_and_render_fn(
        mx, batch_size=batch_size, rep_cfg=rep_cfg, with_dr=with_dr, with_physics=with_physics, substeps=1
    )
    hlo_info = audit_replicator_hlo(step_fn.lower(dx_s, ctrl_s, keys_s).as_text())
    assert hlo_info["zero_copy_verified"], hlo_info

    dc = dx_s
    t_c0 = time.perf_counter()
    for i in range(n_warmup):
      dc, obs = step_fn(dc, ctrl_s, keys_s)
    jax.block_until_ready(tuple(obs.values()))
    compile_s = time.perf_counter() - t_c0

    t0 = time.perf_counter()
    for i in range(n_steps):
      dc, obs = step_fn(dc, ctrl_s, keys_s)
    jax.block_until_ready(tuple(obs.values()))
    dt_ms = (time.perf_counter() - t0) * 1e3 / n_steps

    fps = float(batch_size * 1e3 / max(dt_ms, 1e-9))
    cam_fps = fps * len(cam.cam_ids)
    rec.update({
        "status": "ok",
        "dt_e2e_ms": round(dt_ms, 3),
        "e2e_fps": round(fps, 1),
        "cam_frames_per_s": round(cam_fps, 1),
        "compile_s": round(compile_s, 2),
        "hlo_zero_copy": True,
        "hlo_ops": hlo_info,
    })
  except Exception as exc:
    rec.update({"status": "error", "error_type": type(exc).__name__, "error": str(exc)[:400]})
    traceback.print_exc()

  print(f"[BENCH {n_chips}c] {name} B={batch_size} {cfg.height}x{cfg.width} dr={with_dr} {cfg.shader} "
        f"-> {rec.get('status')} dt={rec.get('dt_e2e_ms')}ms fps={rec.get('e2e_fps')} cam_fps={rec.get('cam_frames_per_s')}",
        flush=True)
  return rec


import dataclasses


def _save_json_atomic(path: str, data: Dict[str, Any]) -> None:
  os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
  tmp = path + ".tmp"
  with open(tmp, "w") as f:
    json.dump(data, f, indent=2)
  os.replace(tmp, path)


def _find_ok(runs: List[Dict[str, Any]], name: str, n_chips: int, batch_size: int, res_str: str, shader: str, with_dr: bool):
  for r in runs:
    if (
        r.get("status") == "ok"
        and r.get("name") == name
        and r.get("n_chips") == n_chips
        and r.get("batch_size") == batch_size
        and r.get("resolution") == res_str
        and r.get("shader") == shader
        and bool(r.get("with_dr")) == bool(with_dr)
    ):
      return r
  return None


def run_benchmarks(out_json: str, export_npz: str, mode: str = "all") -> Dict[str, Any]:
  pallas_interp = (jax.default_backend() == "cpu")
  n_devs = len(jax.devices())
  chip_configs = [1, 4] if n_devs >= 4 else [1]

  if os.path.exists(out_json):
    with open(out_json, "r") as f:
      summary = json.load(f)
  else:
    summary = {
        "backend": jax.default_backend(),
        "device_kind": str(jax.devices()[0].device_kind),
        "n_devices_visible": n_devs,
    }

  if mode in ("all", "parity"):
    parity = run_parity_and_hlo_checks()
    assert all(p["passed"] for p in parity), f"Parity check failed: {parity}"
    summary["parity_checks"] = parity
    _save_json_atomic(out_json, summary)

  if mode in ("all", "robotwin", "robotwin_64_fast", "robotwin_64_ggx", "robotwin_128", "robotwin_128_ggx"):
    robotwin_runs = [r for r in summary.get("robotwin_3cam_runs", []) if r.get("status") == "ok"]
    rep_cfg_rt = _make_rep_cfg(target_center=(0.60, 0.0, 0.14))
    rt_modalities_5 = ("rgb", "distance_to_image_plane", "normals", "semantic_segmentation", "instance_id")
    rt_modalities_6 = DEFAULT_MODALITIES

    rt_cases = []
    if mode in ("all", "robotwin", "robotwin_64_fast"):
      rt_cases.extend([
          (64, 1024, "fast_dr", rt_modalities_5),
          (64, 2048, "fast_dr", rt_modalities_5),
          (64, 1024, "fast_dr", rt_modalities_6),
      ])
    if mode in ("all", "robotwin", "robotwin_64_ggx"):
      rt_cases.extend([
          (64, 1024, "pbr_ggx", rt_modalities_6),
      ])
    if mode in ("all", "robotwin", "robotwin_128"):
      rt_cases.extend([
          (128, 512, "fast_dr", rt_modalities_6),
      ])
    if mode in ("all", "robotwin", "robotwin_128_ggx"):
      rt_cases.extend([
          (128, 512, "pbr_ggx", rt_modalities_6),
      ])
    rt_cache: Dict[int, Any] = {}
    for res, B, shader, mods in rt_cases:
      res_str = f"{res}x{res}"
      rname = f"robotwin_3cam_{len(mods)}mod"
      for nc in chip_configs:
        if B % nc != 0:
          continue
        have_s = _find_ok(robotwin_runs, rname, nc, B, res_str, shader, False)
        have_d = _find_ok(robotwin_runs, rname, nc, B, res_str, shader, True)
        if have_s is not None and have_d is not None:
          print(f"[SKIP OK] {rname} {nc}c B={B} {res_str} {shader}", flush=True)
          continue
        if B not in rt_cache:
          rt_cache[B] = _prepare_robotwin_batch(batch_size=B, n_step_warmup=2)
        m_rt, mx_rt, dx_rt, ctrl_rt = rt_cache[B]
        cam_cfg = TiledCameraCfg(
            cam_names=("head_camera", "left_wrist_camera", "right_wrist_camera"),
            height=res,
            width=res,
            data_types=mods,
            shader=shader,
            tonemap="agx",
            out_dtype=jnp.bfloat16,
            block_b=8,
            block_r=512,
            semantic_fn=assign_robotwin_semantic,
        )
        r_static = have_s or _bench_case(
            rname, m_rt, mx_rt, dx_rt, ctrl_rt, cam_cfg, rep_cfg_rt, B, nc, with_dr=False,
        )
        r_dr = have_d or _bench_case(
            rname, m_rt, mx_rt, dx_rt, ctrl_rt, cam_cfg, rep_cfg_rt, B, nc, with_dr=True,
        )
        if r_static["status"] == "ok" and r_dr["status"] == "ok":
          ovh = (r_dr["dt_e2e_ms"] - r_static["dt_e2e_ms"]) / max(r_static["dt_e2e_ms"], 1e-9) * 100.0
          r_dr["dr_overhead_pct"] = round(ovh, 2)
          r_dr["static_baseline_ms"] = r_static["dt_e2e_ms"]
          r_dr["static_baseline_fps"] = r_static["e2e_fps"]
        if have_s is None:
          robotwin_runs.append(r_static)
        if have_d is None:
          robotwin_runs.append(r_dr)
        summary["robotwin_3cam_runs"] = robotwin_runs
        _save_json_atomic(out_json, summary)

  if mode in ("all", "robodojo", "robodojo_50", "robodojo_200", "robodojo_1000"):
    robodojo_runs = [r for r in summary.get("robodojo_cluttered_runs", []) if r.get("status") == "ok"]
    rep_cfg_rd = _make_rep_cfg(target_center=(0.0, 0.0, 0.45))
    rd_cases = []
    if mode in ("all", "robodojo", "robodojo_50"):
      rd_cases.extend([
          (50, 64, 1024, "fast_dr", DEFAULT_MODALITIES),
          (50, 64, 4096, "fast_dr", DEFAULT_MODALITIES),
      ])
    if mode in ("all", "robodojo", "robodojo_200"):
      rd_cases.extend([
          (200, 64, 1024, "pbr_ggx", DEFAULT_MODALITIES),
      ])
    if mode in ("all", "robodojo", "robodojo_1000"):
      rd_cases.extend([
          (1000, 64, 1024, "pbr_ggx", DEFAULT_MODALITIES),
      ])
    rd_cache: Dict[Tuple[int, int], Any] = {}
    for n_prims, res, B, shader, mods in rd_cases:
      res_str = f"{res}x{res}"
      rname = f"robodojo_G{n_prims}_{len(mods)}mod"
      for nc in chip_configs:
        if B % nc != 0:
          continue
        if B >= 4096 and nc == 1:
          continue
        have_s = _find_ok(robodojo_runs, rname, nc, B, res_str, shader, False)
        have_d = _find_ok(robodojo_runs, rname, nc, B, res_str, shader, True)
        if have_s is not None and have_d is not None:
          print(f"[SKIP OK] {rname} {nc}c B={B} {res_str} {shader}", flush=True)
          continue
        if (n_prims, B) not in rd_cache:
          rd_cache[(n_prims, B)] = _prepare_robodojo_batch(n_target_prims=n_prims, batch_size=B, n_step_warmup=2)
        m_rd, mx_rd, dx_rd, ctrl_rd = rd_cache[(n_prims, B)]
        cam_cfg = TiledCameraCfg(
            cam_names="overview",
            height=res,
            width=res,
            data_types=mods,
            shader=shader,
            tonemap="agx",
            out_dtype=jnp.bfloat16,
            block_b=8,
            block_r=512,
            cull_mode="cluster1",
            semantic_fn=assign_robodojo_semantic,
        )
        r_static = have_s or _bench_case(
            rname, m_rd, mx_rd, dx_rd, ctrl_rd, cam_cfg, rep_cfg_rd, B, nc, with_dr=False,
        )
        r_dr = have_d or _bench_case(
            rname, m_rd, mx_rd, dx_rd, ctrl_rd, cam_cfg, rep_cfg_rd, B, nc, with_dr=True,
        )
        if r_static["status"] == "ok" and r_dr["status"] == "ok":
          ovh = (r_dr["dt_e2e_ms"] - r_static["dt_e2e_ms"]) / max(r_static["dt_e2e_ms"], 1e-9) * 100.0
          r_dr["dr_overhead_pct"] = round(ovh, 2)
          r_dr["static_baseline_ms"] = r_static["dt_e2e_ms"]
          r_dr["static_baseline_fps"] = r_static["e2e_fps"]
        if have_s is None:
          robodojo_runs.append(r_static)
        if have_d is None:
          robodojo_runs.append(r_dr)
        summary["robodojo_cluttered_runs"] = robodojo_runs
        _save_json_atomic(out_json, summary)

  if mode in ("all", "showcase"):
    # =========================================================================
    # Suite 3: High-Res 256x256 Showcase Renders (8+ DR Envs + 3-Cam + 6 Modalities)
    # =========================================================================
    dev0 = jax.devices()[0]
    free_rt_qpos = [18, 25, 32]
    joint_rt_qpos = [0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14]
    m_rt, mx_rt, dx_rt16, ctrl_rt16 = _prepare_robotwin_batch(batch_size=16, n_step_warmup=0)
    dx_rt16 = randomize_qpos_batch(
        jax.random.PRNGKey(9001), mx_rt, dx_rt16,
        free_qpos_indices=free_rt_qpos,
        joint_qpos_indices=joint_rt_qpos,
        cfg=rep.randomizer.poses(joint_jitter_rad=0.14, free_xy_jitter_m=0.05),
    )
    step_rt = jax.jit(lambda d, u: jax.vmap(lambda di, ui: mjx.step(mx_rt, di.replace(ctrl=ui)))(d, u))
    for _ in range(4):
      dx_rt16 = step_rt(dx_rt16, ctrl_rt16)
    jax.block_until_ready(dx_rt16.qpos)

    cam_rt_vis = TiledCamera(
        m_rt,
        TiledCameraCfg(
            cam_names=("diag_overview", "head_camera", "left_wrist_camera", "right_wrist_camera"),
            height=256,
            width=256,
            data_types=DEFAULT_MODALITIES,
            shader="pbr_ggx",
            tonemap="agx",
            out_dtype=jnp.float32,
            block_b=8,
            block_r=512,
            interpret=pallas_interp,
            semantic_fn=assign_robotwin_semantic,
        ),
    )
    rep_vis_rt = _make_rep_cfg(target_center=(0.60, 0.0, 0.14))
    keys_rt16 = jax.device_put(jax.random.split(jax.random.PRNGKey(777), 16), dev0)
    obs_rt_vis = jax.jit(lambda d, k: cam_rt_vis.render_local(d, rng_or_keys=k, rep_cfg=rep_vis_rt))(dx_rt16, keys_rt16)
    jax.block_until_ready(tuple(obs_rt_vis.values()))

    m_rd1k, mx_rd1k, dx_rd1k, ctrl_rd1k = _prepare_robodojo_batch(n_target_prims=1000, batch_size=8, n_step_warmup=4)
    cam_rd_vis = TiledCamera(
        m_rd1k,
        TiledCameraCfg(
            cam_names=("overview", "closeup"),
            height=256,
            width=256,
            data_types=DEFAULT_MODALITIES,
            shader="pbr_ggx",
            tonemap="agx",
            out_dtype=jnp.float32,
            block_b=8,
            block_r=512,
            cull_mode="cluster1",
            interpret=pallas_interp,
            semantic_fn=assign_robodojo_semantic,
        ),
    )
    rep_vis_rd = _make_rep_cfg(target_center=(0.0, 0.0, 0.45))
    keys_rd8 = jax.device_put(jax.random.split(jax.random.PRNGKey(888), 8), dev0)
    obs_rd_vis = jax.jit(lambda d, k: cam_rd_vis.render_local(d, rng_or_keys=k, rep_cfg=rep_vis_rd))(dx_rd1k, keys_rd8)
    jax.block_until_ready(tuple(obs_rd_vis.values()))

    os.makedirs(os.path.dirname(export_npz) or ".", exist_ok=True)
    np.savez(
        export_npz,
        rt_diag_rgb=np.asarray(obs_rt_vis["rgb"][:8, 0], np.float32),
        rt_head_rgb=np.asarray(obs_rt_vis["rgb"][:4, 1], np.float32),
        rt_lw_rgb=np.asarray(obs_rt_vis["rgb"][:4, 2], np.float32),
        rt_rw_rgb=np.asarray(obs_rt_vis["rgb"][:4, 3], np.float32),
        rt_head_depth=np.asarray(obs_rt_vis["distance_to_image_plane"][0, 1], np.float32),
        rt_head_normals=np.asarray(obs_rt_vis["normals"][0, 1], np.float32),
        rt_head_sem=np.asarray(obs_rt_vis["semantic_segmentation"][0, 1], np.int32),
        rt_head_inst=np.asarray(obs_rt_vis["instance_id"][0, 1], np.int32),
        rt_head_flow=np.asarray(obs_rt_vis["motion_vectors"][0, 1], np.float32),
        rd_over_rgb=np.asarray(obs_rd_vis["rgb"][:8, 0], np.float32),
        rd_close_rgb=np.asarray(obs_rd_vis["rgb"][:8, 1], np.float32),
        rd_close_depth=np.asarray(obs_rd_vis["distance_to_image_plane"][0, 1], np.float32),
        rd_close_normals=np.asarray(obs_rd_vis["normals"][0, 1], np.float32),
        rd_close_sem=np.asarray(obs_rd_vis["semantic_segmentation"][0, 1], np.int32),
        rd_close_flow=np.asarray(obs_rd_vis["motion_vectors"][0, 1], np.float32),
    )
    print("Saved showcase npz:", export_npz, flush=True)
    summary["export_npz"] = export_npz

  os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
  with open(out_json, "w") as f:
    json.dump(summary, f, indent=2)
  print("Saved summary JSON:", out_json, flush=True)
  return summary


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--out_json", default="/dev/shm/zaychik-tpu-mjx-w1/out/v3_w1_wave3_replicator.json")
  ap.add_argument("--export_npz", default="/dev/shm/zaychik-tpu-mjx-w1/out/replicator_showcase.npz")
  ap.add_argument(
      "--mode",
      default="all",
      choices=[
          "all", "parity", "robotwin", "robotwin_64_fast", "robotwin_64_ggx",
          "robotwin_128", "robotwin_128_ggx", "robodojo", "robodojo_50",
          "robodojo_200", "robodojo_1000", "showcase",
      ],
  )
  args = ap.parse_args()
  run_benchmarks(args.out_json, args.export_npz, mode=args.mode)


if __name__ == "__main__":
  main()
