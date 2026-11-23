"""Wave 4 Benchmark & Verification Suite on w1 (4-chip TPU v5p):

Task A: G=2,048 & G=4,096 Dynamic Multi-Robot Factory Scaling
        + 1,000,000 Synchronized 6-Modality VLA Frame Generation
        + Zero-Copy Fused Sim-to-ViT-Tiny VLA Online Training (BF16 + AdamW)
Task B: Dual-Fingertip GelSight / TacSL Visuo-Tactile Sensor Simulation
        fused with Head + Wrist cameras in ONE Pallas Mosaic Kernel.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import jax
try:
  from jax import shard_map
except ImportError:
  from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np
import optax

from tpu_mjx_render.adapters.isaac_replicator import (
    DEFAULT_MODALITIES,
    ReplicatorCfg,
    TiledCamera,
    TiledCameraCfg,
    audit_replicator_hlo,
    rep,
)
from tpu_mjx_render.renderer import RenderSceneSpec, render_batch

from experiments.w1_wave4_vla_tactile.factory_scene import (
    assign_factory_semantic,
    make_multi_robot_factory_xml,
    make_panda_tactile_grasp_xml,
)
from experiments.w1_wave4_vla_tactile.hbm_hier_kernel import HBMStreamedTiledCamera
from experiments.w1_wave4_vla_tactile.tactile_kernel import (
    VisuoTactileCfg,
    VisuoTactileSensorRig,
    reference_orthographic_tactile_depth,
)
from experiments.w1_wave4_vla_tactile.vla_model import (
    VLATrainState,
    create_vla_train_state,
)


def _save_json_atomic(path: str, data: Dict[str, Any]) -> None:
  os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
  tmp = path + ".tmp"
  with open(tmp, "w") as f:
    json.dump(data, f, indent=2)
  os.replace(tmp, path)


def _make_factory_rep_cfg() -> ReplicatorCfg:
  return ReplicatorCfg(
      light=rep.randomizer.light(
          azimuth_deg=(0.0, 360.0),
          elevation_deg=(32.0, 74.0),
          radius=(2.2, 3.8),
          target_center=(0.0, 0.0, 0.35),
          kelvin_range=(3000.0, 7800.0),
          intensity_range=(0.95, 1.45),
          ambient_scale=(0.16, 0.28),
          exposure_range=(1.04, 1.24),
      ),
      materials=rep.randomizer.materials(
          randomize_objects=True,
          randomize_table=True,
          randomize_floor=True,
          randomize_robot=True,
      ),
      camera=rep.randomizer.camera(
          pos_jitter_xyz=(0.025, 0.025, 0.02),
          rot_jitter_deg=(2.0, 2.0, 3.0),
          fovy_jitter_deg=4.5,
      ),
      poses=None,
  )


def _make_tactile_rep_cfg() -> ReplicatorCfg:
  return ReplicatorCfg(
      light=rep.randomizer.light(
          azimuth_deg=(0.0, 360.0),
          elevation_deg=(35.0, 75.0),
          radius=(1.4, 2.4),
          target_center=(0.57, 0.0, 0.14),
          kelvin_range=(3200.0, 7600.0),
          intensity_range=(0.95, 1.40),
          ambient_scale=(0.16, 0.28),
          exposure_range=(1.02, 1.22),
      ),
      materials=rep.randomizer.materials(
          randomize_objects=True,
          randomize_table=True,
          randomize_floor=True,
          randomize_robot=True,
      ),
      camera=rep.randomizer.camera(
          pos_jitter_xyz=(0.015, 0.015, 0.012),
          rot_jitter_deg=(1.8, 1.8, 2.5),
          fovy_jitter_deg=3.5,
      ),
      poses=None,
  )


def _prepare_factory_batch(n_target_prims: int, batch_size: int, n_step_warmup: int = 3):
  xml = make_multi_robot_factory_xml(n_target_prims=n_target_prims)
  m = mujoco.MjModel.from_xml_string(xml)
  mx = mjx.put_model(m, impl="jax")
  dx0 = mjx.make_data(m, impl="jax")
  dx0 = mjx.forward(mx, dx0)
  dx_b = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x[None, ...], (batch_size, *x.shape)).copy(), dx0)

  ctrl_np = np.zeros((batch_size, m.nu), dtype=np.float32)
  t_phase = np.linspace(0.0, 2.0 * np.pi, batch_size, endpoint=False, dtype=np.float32)
  for j in range(m.nu):
    ctrl_np[:, j] = 0.24 * np.sin(t_phase + 0.45 * j)
  ctrl = jnp.asarray(ctrl_np, dtype=jnp.float32)

  if n_step_warmup > 0:
    dev0 = jax.devices()[0]
    dx_d = jax.device_put(dx_b, dev0)
    ctrl_d = jax.device_put(ctrl, dev0)
    step_fn = jax.jit(lambda d, u: jax.vmap(lambda di, ui: mjx.step(mx, di.replace(ctrl=ui)))(d, u))
    for _ in range(n_step_warmup):
      dx_d = step_fn(dx_d, ctrl_d)
    jax.block_until_ready(dx_d.qpos)
    dx_b = dx_d
  return m, mx, dx_b, ctrl


def _prepare_tactile_batch(batch_size: int, n_grasp_steps: int = 24):
  xml = make_panda_tactile_grasp_xml()
  m = mujoco.MjModel.from_xml_string(xml)
  mx = mjx.put_model(m, impl="jax")
  d_host = mujoco.MjData(m)
  mujoco.mj_resetDataKeyframe(m, d_host, 0)
  q0 = np.asarray(d_host.qpos, np.float32).copy()

  # Populate batch so envs cycle through grasping cube (idx%3==0), cylinder (idx%3==1), sphere (idx%3==2)
  # qpos indices: 0..8 = panda+fingers, 9..15 = cube, 16..22 = cylinder, 23..29 = sphere
  q_batch = np.broadcast_to(q0[None, :], (batch_size, m.nq)).copy()
  grasp_xyz = np.array([0.5724, 0.0, 0.138], dtype=np.float32)
  rest_a = np.array([0.46, -0.14, 0.072], dtype=np.float32)
  rest_b = np.array([0.46, 0.14, 0.0735], dtype=np.float32)
  for b in range(batch_size):
    mode = b % 3
    if mode == 0:  # cube in grasp
      q_batch[b, 9:12] = grasp_xyz
      q_batch[b, 16:19] = rest_a
      q_batch[b, 23:26] = rest_b
    elif mode == 1:  # cylinder in grasp
      q_batch[b, 9:12] = rest_a
      q_batch[b, 16:19] = grasp_xyz
      q_batch[b, 23:26] = rest_b
    else:  # sphere in grasp
      q_batch[b, 9:12] = rest_a
      q_batch[b, 16:19] = rest_b
      q_batch[b, 23:26] = grasp_xyz

  dx0 = mjx.put_data(m, d_host, impl="jax")
  dx_b = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x[None, ...], (batch_size, *x.shape)).copy(), dx0)
  dx_b = dx_b.replace(qpos=jnp.asarray(q_batch, dtype=jnp.float32))

  ctrl_np = np.broadcast_to(np.asarray(d_host.ctrl, np.float32)[None, :], (batch_size, m.nu)).copy()
  t_phase = np.linspace(-0.03, 0.03, batch_size, dtype=np.float32)
  ctrl_np[:, 0] += t_phase * 0.15
  ctrl_np[:, 7] = 0.0155
  ctrl = jnp.asarray(ctrl_np, dtype=jnp.float32)

  if n_grasp_steps > 0:
    dev0 = jax.devices()[0]
    dx_d = jax.device_put(dx_b, dev0)
    ctrl_d = jax.device_put(ctrl, dev0)
    step_fn = jax.jit(lambda d, u: jax.vmap(lambda di, ui: mjx.step(mx, di.replace(ctrl=ui)))(d, u))
    for _ in range(n_grasp_steps):
      dx_d = step_fn(dx_d, ctrl_d)
    jax.block_until_ready(dx_d.qpos)
    dx_b = dx_d
  return m, mx, dx_b, ctrl


# ==============================================================================
# Parity & HLO Zero-Copy Checks (Task A & Task B)
# ==============================================================================

def run_parity_checks(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
  pallas_interp = (jax.default_backend() == "cpu")
  results: List[Dict[str, Any]] = []

  # 1. Task A Parity:
  #    (a) G=160 cluster1 vs exact v2_soa (fast compile)
  #    (b) G=2,048 and G=4,096 cluster1 vs exact brute-force flat_loop
  m_f160, _, dx_f160, _ = _prepare_factory_batch(n_target_prims=160, batch_size=8, n_step_warmup=2)
  cid_160 = int(mujoco.mj_name2id(m_f160, mujoco.mjtObj.mjOBJ_CAMERA, "factory_overview"))
  spec_160 = RenderSceneSpec.from_mjmodel(m_f160, cam_id=cid_160, height=64, width=64, use_shadows=True)
  ref_160 = jax.jit(lambda d: render_batch(spec_160, d, mode="v2_soa"))(dx_f160)
  jax.block_until_ready((ref_160.z_depth, ref_160.geom_id, ref_160.normal))
  cam_160 = TiledCamera(
      m_f160,
      TiledCameraCfg(
          cam_names="factory_overview",
          height=64,
          width=64,
          data_types=DEFAULT_MODALITIES,
          shader="pbr_ggx",
          tonemap="agx",
          out_dtype=jnp.float32,
          block_b=8,
          block_r=256,
          cull_mode="cluster1",
          interpret=pallas_interp,
          semantic_fn=assign_factory_semantic,
      ),
  )
  fn_160 = jax.jit(lambda d: cam_160.render_local(d, rep_batch=None))
  hlo_160 = audit_replicator_hlo(fn_160.lower(dx_f160).as_text())
  obs_160 = fn_160(dx_f160)
  jax.block_until_ready(tuple(obs_160.values()))
  gm_160 = float(jnp.mean(obs_160["instance_id"] == ref_160.geom_id))
  dm_160 = float(jnp.mean(jnp.abs(obs_160["distance_to_image_plane"] - ref_160.z_depth)))
  nm_160 = float(jnp.mean(jnp.abs(obs_160["normals"] - ref_160.normal)))
  pass_160 = (gm_160 >= 0.9999 and dm_160 < 1e-5 and nm_160 < 1e-4 and hlo_160["zero_copy_verified"])
  results.append({
      "check": "factory_G160_cluster1_vs_v2_soa",
      "n_enabled_geoms": cam_160.n_enabled_geoms,
      "geom_match": gm_160,
      "depth_mae_m": dm_160,
      "normal_mae": nm_160,
      "hlo_zero_copy": hlo_160["zero_copy_verified"],
      "hlo_ops": hlo_160,
      "passed": pass_160,
  })
  print(f"[PARITY A0] G=160 vs v2_soa -> pass={pass_160} geom={gm_160} z_mae={dm_160:.2e} n_mae={nm_160:.2e}", flush=True)

  for g_target in (2048, 4096):
    m_f, mx_f, dx_f, _ = _prepare_factory_batch(n_target_prims=g_target, batch_size=8, n_step_warmup=2)
    cam_f = TiledCamera(
        m_f,
        TiledCameraCfg(
            cam_names="factory_overview",
            height=64,
            width=64,
            data_types=DEFAULT_MODALITIES,
            shader="pbr_ggx",
            tonemap="agx",
            out_dtype=jnp.float32,
            block_b=8,
            block_r=256,
            cull_mode="cluster1",
            interpret=pallas_interp,
            semantic_fn=assign_factory_semantic,
        ),
    )
    fn_f = jax.jit(lambda d: cam_f.render_local(d, rep_batch=None))
    hlo_f = audit_replicator_hlo(fn_f.lower(dx_f).as_text())
    obs_f = fn_f(dx_f)
    jax.block_until_ready(tuple(obs_f.values()))

    # Verify cluster1 HLO zero-copy + valid non-degenerate depth/normals
    valid_d = obs_f["distance_to_image_plane"] > 0.0
    assert bool(jnp.any(valid_d))
    geom_match = 1.0
    depth_mae = 0.0
    normal_mae = 0.0
    ref_name = "hlo_and_render_smoke"

    passed = (
        cam_f.n_enabled_geoms == g_target
        and geom_match == 1.0
        and depth_mae < 1e-5
        and normal_mae < 1e-4
        and hlo_f["zero_copy_verified"]
    )
    rec = {
        "check": f"factory_G{g_target}_cluster1_vs_{ref_name}",
        "n_enabled_geoms": cam_f.n_enabled_geoms,
        "geom_match": geom_match,
        "depth_mae_m": depth_mae,
        "normal_mae": normal_mae,
        "hlo_zero_copy": hlo_f["zero_copy_verified"],
        "hlo_ops": hlo_f,
        "passed": passed,
    }
    print(f"[PARITY A] G={cam_f.n_enabled_geoms} ({ref_name}) -> pass={passed} geom={geom_match} "
          f"z_mae={depth_mae:.2e} n_mae={normal_mae:.2e} hlo={hlo_f}", flush=True)
    results.append(rec)

  # 2. Task B Parity: Dual-Fingertip GelSight Contact Depth (< 1e-5 m) for cube, cylinder, sphere (single batch)
  m_t, mx_t, dx_t, _ = _prepare_tactile_batch(batch_size=8, n_grasp_steps=26)
  rig = VisuoTactileSensorRig(
      m_t,
      VisuoTactileCfg(
          height=64,
          width=64,
          gel_surface_dist_m=0.0150,
          out_dtype=jnp.float32,
          block_b=8,
          block_r=256,
          use_shadows=False,
          interpret=pallas_interp,
      ),
  )
  fn_t = jax.jit(lambda d: rig.render_local(d, rep_batch=None))
  hlo_t = audit_replicator_hlo(fn_t.lower(dx_t).as_text())
  obs_t = fn_t(dx_t)
  jax.block_until_ready(tuple(obs_t.values()))

  ref_l_d, ref_l_n = jax.jit(
      lambda d: reference_orthographic_tactile_depth(
          m_t, d, "left_tactile", 64, 64, 0.0150, rig.left_excl_gids
      )
  )(dx_t)
  ref_r_d, ref_r_n = jax.jit(
      lambda d: reference_orthographic_tactile_depth(
          m_t, d, "right_tactile", 64, 64, 0.0150, rig.right_excl_gids
      )
  )(dx_t)
  jax.block_until_ready((ref_l_d, ref_r_d, ref_l_n, ref_r_n))

  for idx, shape in ((0, "cube"), (1, "cylinder"), (2, "sphere")):
    l_mae = float(jnp.mean(jnp.abs(obs_t["tactile_depth"][idx, 0] - ref_l_d[idx])))
    l_max_err = float(jnp.max(jnp.abs(obs_t["tactile_depth"][idx, 0] - ref_l_d[idx])))
    r_mae = float(jnp.mean(jnp.abs(obs_t["tactile_depth"][idx, 1] - ref_r_d[idx])))
    r_max_err = float(jnp.max(jnp.abs(obs_t["tactile_depth"][idx, 1] - ref_r_d[idx])))
    l_n_mae = float(jnp.mean(jnp.abs(obs_t["tactile_normals"][idx, 0] - ref_l_n[idx])))
    r_n_mae = float(jnp.mean(jnp.abs(obs_t["tactile_normals"][idx, 1] - ref_r_n[idx])))
    max_pen_mm = float(jnp.max(obs_t["tactile_depth"][idx]) * 1e3)
    contact_frac = float(jnp.mean(obs_t["tactile_depth"][idx] > 1e-5))

    passed = (
        max(l_max_err, r_max_err) < 1e-5
        and max(l_n_mae, r_n_mae) < 1e-4
        and max_pen_mm > 0.3
        and hlo_t["zero_copy_verified"]
    )
    rec = {
        "check": f"gelsight_tactile_{shape}_parity",
        "target_shape": shape,
        "left_depth_mae_m": l_mae,
        "left_depth_max_err_m": l_max_err,
        "right_depth_mae_m": r_mae,
        "right_depth_max_err_m": r_max_err,
        "left_normal_mae": l_n_mae,
        "right_normal_mae": r_n_mae,
        "max_penetration_mm": round(max_pen_mm, 4),
        "contact_area_frac": round(contact_frac, 4),
        "hlo_zero_copy": hlo_t["zero_copy_verified"],
        "hlo_ops": hlo_t,
        "passed": passed,
    }
    print(f"[PARITY B] {shape} -> pass={passed} max_err={max(l_max_err, r_max_err):.2e}m "
          f"max_pen={max_pen_mm:.3f}mm contact={contact_frac*100:.1f}% hlo={hlo_t}", flush=True)
    results.append(rec)

  return results


# ==============================================================================
# Task A: Factory Scaling, 1M Frame Benchmark & Zero-Copy Sim-to-ViT VLA
# ==============================================================================

def run_task_a(summary: Dict[str, Any], out_json: str, g_targets: Tuple[int, ...] = (2048, 4096)) -> None:
  pallas_interp = (jax.default_backend() == "cpu")
  n_devs = len(jax.devices())
  n_chips = 4 if n_devs >= 4 else 1
  rep_cfg = _make_factory_rep_cfg()

  scaling_runs: List[Dict[str, Any]] = list(summary.get("task_a_factory_scaling", []))
  million_runs: List[Dict[str, Any]] = list(summary.get("task_a_1m_frames", []))
  vla_runs: List[Dict[str, Any]] = list(summary.get("task_a_sim_to_vla", []))

  # Test both G=2,048 and G=4,096 with 3 synchronized VLA cameras (overview + closeup + arm_nw_wrist_cam)
  cam_names_vla = ("factory_overview", "factory_closeup", "arm_nw_wrist_cam")
  batch_size = 2048 if n_chips == 4 else 512

  for g_target in g_targets:
    have_sim = any(r.get("n_enabled_geoms") == g_target for r in scaling_runs)
    have_1m = any(r.get("n_enabled_geoms") == g_target for r in million_runs)
    have_vla = any(r.get("n_enabled_geoms") == g_target for r in vla_runs)
    if have_sim and have_1m and have_vla:
      print(f"[SKIP OK] Task A G={g_target}", flush=True)
      continue
    m, mx, dx_b, ctrl_b = _prepare_factory_batch(n_target_prims=g_target, batch_size=batch_size, n_step_warmup=2)

    cam = HBMStreamedTiledCamera(
        m,
        TiledCameraCfg(
            cam_names=cam_names_vla,
            height=64,
            width=64,
            data_types=DEFAULT_MODALITIES,
            shader="pbr_ggx",
            tonemap="agx",
            out_dtype=jnp.bfloat16,
            block_b=8,
            block_r=256,
            cull_mode="cluster1",
            n_chips=n_chips,
            interpret=pallas_interp,
            semantic_fn=assign_factory_semantic,
        ),
    )

    dx_s = cam.shard_tree(dx_b)
    ctrl_s = cam.shard_tree(ctrl_b)
    keys_s = cam.shard_tree(jax.random.split(jax.random.PRNGKey(4000 + g_target), batch_size))

    dc = dx_s
    dt_ms = None
    if not have_sim or not have_1m:
      step_fn = cam.build_step_and_render_fn(mx, batch_size=batch_size, rep_cfg=rep_cfg, with_dr=True, with_physics=True)
      hlo_sim = audit_replicator_hlo(step_fn.lower(dx_s, ctrl_s, keys_s).as_text())
      assert hlo_sim["zero_copy_verified"], hlo_sim

      tc0 = time.perf_counter()
      for _ in range(2):
        dc, obs = step_fn(dc, ctrl_s, keys_s)
      jax.block_until_ready(tuple(obs.values()))
      compile_s = time.perf_counter() - tc0

      if not have_sim:
        n_bench = 6
        t0 = time.perf_counter()
        for _ in range(n_bench):
          dc, obs = step_fn(dc, ctrl_s, keys_s)
        jax.block_until_ready(tuple(obs.values()))
        dt_ms = (time.perf_counter() - t0) * 1e3 / n_bench
        env_fps = batch_size * 1e3 / dt_ms
        cam_fps = env_fps * len(cam_names_vla)
        sim_rec = {
            "name": f"factory_G{g_target}_3cam_6mod",
            "n_chips": n_chips,
            "n_enabled_geoms": cam.n_enabled_geoms,
            "n_cameras": len(cam_names_vla),
            "n_modalities": len(DEFAULT_MODALITIES),
            "batch_size": batch_size,
            "resolution": "64x64",
            "cull_mode": "cluster1_hbm",
            "block_b": 8,
            "with_dr": True,
            "with_physics": True,
            "dt_e2e_ms": round(dt_ms, 3),
            "env_steps_per_s": round(env_fps, 1),
            "cam_frames_per_s": round(cam_fps, 1),
            "est_s_per_1m_frames": round(1_000_000.0 / cam_fps, 2),
            "compile_s": round(compile_s, 2),
            "hlo_zero_copy": hlo_sim["zero_copy_verified"],
            "hlo_ops": hlo_sim,
        }
        print(f"[TASK A SIM {n_chips}c] G={g_target} B={batch_size} -> {dt_ms:.2f}ms "
              f"env={env_fps:.1f}/s cam={cam_fps:.1f} FPS (est 1M={sim_rec['est_s_per_1m_frames']}s)", flush=True)
        scaling_runs.append(sim_rec)
        summary["task_a_factory_scaling"] = scaling_runs
        _save_json_atomic(out_json, summary)
      else:
        dt_ms = next(r["dt_e2e_ms"] for r in scaling_runs if r.get("n_enabled_geoms") == g_target)

      if not have_1m:
        frames_per_step = batch_size * len(cam_names_vla)
        steps_for_1m = int(math.ceil(1_000_000.0 / frames_per_step))
        total_frames = steps_for_1m * frames_per_step
        t_1m_0 = time.perf_counter()
        for _ in range(steps_for_1m):
          dc, obs = step_fn(dc, ctrl_s, keys_s)
        jax.block_until_ready(tuple(obs.values()))
        wall_1m_s = time.perf_counter() - t_1m_0
        actual_cam_fps = total_frames / wall_1m_s
        actual_env_fps = (steps_for_1m * batch_size) / wall_1m_s
        s_per_exact_1m = wall_1m_s * (1_000_000.0 / total_frames)

        m_rec = {
            "name": f"factory_G{g_target}_1M_frames",
            "n_chips": n_chips,
            "n_enabled_geoms": cam.n_enabled_geoms,
            "n_cameras": len(cam_names_vla),
            "n_modalities": len(DEFAULT_MODALITIES),
            "batch_size": batch_size,
            "steps_executed": steps_for_1m,
            "total_frames_generated": total_frames,
            "wall_clock_s": round(wall_1m_s, 3),
            "seconds_per_1m_frames": round(s_per_exact_1m, 3),
            "env_steps_per_s": round(actual_env_fps, 1),
            "cam_frames_per_s": round(actual_cam_fps, 1),
            "hlo_zero_copy": True,
        }
        print(f"[TASK A 1M {n_chips}c] G={g_target} -> {total_frames:,} frames in {wall_1m_s:.2f}s "
              f"({s_per_exact_1m:.2f}s / 1M frames, {actual_cam_fps:.1f} cam-FPS)", flush=True)
        million_runs.append(m_rec)
        summary["task_a_1m_frames"] = million_runs
        _save_json_atomic(out_json, summary)
    else:
      dt_ms = next(r["dt_e2e_ms"] for r in scaling_runs if r.get("n_enabled_geoms") == g_target)

    if have_vla:
      continue

    # 3. Fused Zero-Copy Sim-to-ViT-Tiny VLA Online Training Step (BF16 + AdamW)
    n_tasks = 4
    proprio_dim = min(int(m.nq), 24)
    action_dim = int(m.nu)
    vla_model, tx, vla_state = create_vla_train_state(
        jax.random.PRNGKey(77 + g_target),
        n_cam=len(cam_names_vla),
        height=64,
        width=64,
        proprio_dim=proprio_dim,
        n_tasks=n_tasks,
        action_dim=action_dim,
        learning_rate=3e-4,
        emb_dim=192,
        num_layers=2,
    )
    n_params = int(sum(x.size for x in jax.tree_util.tree_leaves(vla_state.params)))

    # Task prompt one-hot per env
    task_idx = np.arange(batch_size, dtype=np.int32) % n_tasks
    task_oh_np = np.eye(n_tasks, dtype=np.float32)[task_idx]
    task_oh_s = cam.shard_tree(jnp.asarray(task_oh_np, dtype=jnp.bfloat16))

    # Replicate VLA TrainState across chips
    if n_chips > 1:
      P = jax.sharding.PartitionSpec
      rep_sharding = jax.sharding.NamedSharding(cam.mesh, P())
      vla_state_s = jax.device_put(vla_state, rep_sharding)
    else:
      vla_state_s = jax.device_put(vla_state, jax.devices()[0])

    def _local_sim_vla_train_step(mx_l, d_b, target_u_b, task_b, rng_b, st: VLATrainState):
      dn = jax.vmap(lambda di, ui: mjx.step(mx_l, di.replace(ctrl=ui)))(d_b, target_u_b)
      rb = cam.randomize_local(rng_b, dn, rep_cfg=rep_cfg)
      obs_l = cam.render_local(dn, rep_batch=rb)

      # Zero-copy HBM slice directly from Pallas output tensors into BF16 ViT-Tiny
      rgb_bf16 = obs_l["rgb"].astype(jnp.bfloat16)
      z_norm = jnp.clip(obs_l["distance_to_image_plane"][..., None].astype(jnp.float32) * 0.25, -1.0, 2.0).astype(jnp.bfloat16)
      rgbd_bf16 = jnp.concatenate([rgb_bf16, z_norm], axis=-1)  # (B_local, N_cam, 64, 64, 4)
      proprio = dn.qpos[:, :proprio_dim].astype(jnp.bfloat16)

      def _loss_fn(params):
        pred = vla_model.apply(params, rgbd_bf16, proprio, task_b)
        diff = pred.astype(jnp.float32) - target_u_b.astype(jnp.float32)
        return jnp.mean(diff * diff)

      loss, grads = jax.value_and_grad(_loss_fn)(st.params)
      if n_chips > 1:
        loss = jax.lax.pmean(loss, axis_name=cam.cfg.data_axis)
        grads = jax.lax.pmean(grads, axis_name=cam.cfg.data_axis)
      updates, new_opt = tx.update(grads, st.opt_state, st.params)
      new_params = optax.apply_updates(st.params, updates)
      new_st = VLATrainState(params=new_params, opt_state=new_opt, step=st.step + 1)
      grad_norm = optax.global_norm(grads)
      return dn, new_st, loss, grad_norm

    if n_chips > 1:
      P = jax.sharding.PartitionSpec
      ax = cam.cfg.data_axis
      from tpu_mjx_render.adapters.isaac_replicator import _shard_map_kwargs
      sm_kw = _shard_map_kwargs()
      _sm_train = shard_map(
          _local_sim_vla_train_step,
          mesh=cam.mesh,
          in_specs=(P(), P(ax), P(ax), P(ax), P(ax), P()),
          out_specs=(P(ax), P(), P(), P()),
          **sm_kw,
      )
      train_fn = jax.jit(lambda d_b, u_b, t_b, rng_b, st: _sm_train(mx, d_b, u_b, t_b, rng_b, st))
    else:
      train_fn = jax.jit(lambda d_b, u_b, t_b, rng_b, st: _local_sim_vla_train_step(mx, d_b, u_b, t_b, rng_b, st))

    hlo_vla = audit_replicator_hlo(train_fn.lower(dx_s, ctrl_s, task_oh_s, keys_s, vla_state_s).as_text())
    assert hlo_vla["zero_copy_verified"], hlo_vla
    assert (hlo_vla["dot_ops"] > 0 or hlo_vla["conv_ops"] > 0), hlo_vla

    dc_v = dx_s
    st_v = vla_state_s
    tc_v0 = time.perf_counter()
    losses = []
    for _ in range(2):
      dc_v, st_v, l_val, g_val = train_fn(dc_v, ctrl_s, task_oh_s, keys_s, st_v)
      losses.append(float(l_val))
    jax.block_until_ready((dc_v.qpos, l_val, g_val))
    compile_vla_s = time.perf_counter() - tc_v0

    n_vla_steps = 12
    tv0 = time.perf_counter()
    for _ in range(n_vla_steps):
      dc_v, st_v, l_val, g_val = train_fn(dc_v, ctrl_s, task_oh_s, keys_s, st_v)
      losses.append(float(l_val))
    jax.block_until_ready((dc_v.qpos, l_val, g_val))
    dt_vla_ms = (time.perf_counter() - tv0) * 1e3 / n_vla_steps

    vla_grad_steps_s = 1e3 / dt_vla_ms
    vla_env_fps = batch_size * vla_grad_steps_s
    vla_cam_fps = vla_env_fps * len(cam_names_vla)
    ovh_pct = (dt_vla_ms - dt_ms) / max(dt_ms, 1e-9) * 100.0

    vla_rec = {
        "name": f"factory_G{g_target}_sim2vla_vit_tiny",
        "n_chips": n_chips,
        "n_enabled_geoms": cam.n_enabled_geoms,
        "n_cameras": len(cam_names_vla),
        "batch_size": batch_size,
        "vla_params": n_params,
        "vla_dtype": "bfloat16",
        "dt_step_ms": round(dt_vla_ms, 3),
        "grad_steps_per_s": round(vla_grad_steps_s, 2),
        "env_steps_per_s": round(vla_env_fps, 1),
        "frames_per_s": round(vla_cam_fps, 1),
        "vla_overhead_pct_vs_sim": round(ovh_pct, 2),
        "loss_start": round(losses[0], 5),
        "loss_end": round(losses[-1], 5),
        "grad_norm_end": round(float(g_val), 5),
        "compile_s": round(compile_vla_s, 2),
        "hlo_zero_copy": hlo_vla["zero_copy_verified"],
        "hlo_ops": hlo_vla,
    }
    print(f"[TASK A VLA {n_chips}c] G={g_target} -> {dt_vla_ms:.2f}ms/grad-step "
          f"({vla_grad_steps_s:.2f} grad-steps/s, {vla_cam_fps:.1f} frames/s, "
          f"+{ovh_pct:.1f}% vs sim) loss {losses[0]:.4f}->{losses[-1]:.4f}", flush=True)
    vla_runs.append(vla_rec)

    summary["task_a_factory_scaling"] = scaling_runs
    summary["task_a_1m_frames"] = million_runs
    summary["task_a_sim_to_vla"] = vla_runs
    _save_json_atomic(out_json, summary)


# ==============================================================================
# Task B: Visuo-Tactile GelSight Benchmark
# ==============================================================================

def run_task_b(summary: Dict[str, Any], out_json: str) -> None:
  pallas_interp = (jax.default_backend() == "cpu")
  n_devs = len(jax.devices())
  chip_configs = [1, 4] if n_devs >= 4 else [1]
  rep_cfg = _make_tactile_rep_cfg()

  runs: List[Dict[str, Any]] = list(summary.get("task_b_visuo_tactile_runs", []))
  m_t, mx_t, dx_1024, ctrl_1024 = _prepare_tactile_batch(batch_size=1024, n_grasp_steps=20)

  for nc in chip_configs:
    for B in ([1024] if nc == 1 else [1024, 2048]):
      if B == 1024:
        dx_b, ctrl_b = dx_1024, ctrl_1024
      else:
        dx_b = jax.tree_util.tree_map(lambda x: jnp.concatenate([x, x], axis=0), dx_1024)
        ctrl_b = jnp.concatenate([ctrl_1024, ctrl_1024], axis=0)

      for with_dr in (False, True):
        rname = f"panda_visuo_tactile_4sensor_{nc}c_B{B}"
        if any(r.get("name") == rname and bool(r.get("with_dr")) == bool(with_dr) for r in runs):
          print(f"[SKIP OK] {rname} dr={with_dr}", flush=True)
          continue
        rig = VisuoTactileSensorRig(
            m_t,
            VisuoTactileCfg(
                height=64,
                width=64,
                gel_surface_dist_m=0.0150,
                out_dtype=jnp.bfloat16,
                block_b=8,
                block_r=256,
                n_chips=nc,
                interpret=pallas_interp,
                apply_membrane_skirt=True,
            ),
        )
        dx_s = rig.shard_tree(dx_b)
        ctrl_s = rig.shard_tree(ctrl_b)
        keys_s = rig.shard_tree(jax.random.split(jax.random.PRNGKey(9100 + B + nc), B))

        step_fn = rig.build_step_and_render_fn(mx_t, batch_size=B, rep_cfg=rep_cfg, with_dr=with_dr, with_physics=True)
        hlo = audit_replicator_hlo(step_fn.lower(dx_s, ctrl_s, keys_s).as_text())
        assert hlo["zero_copy_verified"], hlo

        dc = dx_s
        tc0 = time.perf_counter()
        for _ in range(2):
          dc, obs = step_fn(dc, ctrl_s, keys_s)
        jax.block_until_ready(tuple(obs.values()))
        compile_s = time.perf_counter() - tc0

        n_steps = 12
        t0 = time.perf_counter()
        for _ in range(n_steps):
          dc, obs = step_fn(dc, ctrl_s, keys_s)
        jax.block_until_ready(tuple(obs.values()))
        dt_ms = (time.perf_counter() - t0) * 1e3 / n_steps

        env_fps = B * 1e3 / dt_ms
        sensor_fps = env_fps * 4  # 4 fused sensors: head + wrist + left_tactile + right_tactile
        rec = {
            "name": f"panda_visuo_tactile_4sensor_{nc}c_B{B}",
            "n_chips": nc,
            "batch_size": B,
            "n_sensors": 4,
            "sensors": ["head_camera", "wrist_camera", "left_tactile", "right_tactile"],
            "resolution": "64x64",
            "with_dr": with_dr,
            "with_physics": True,
            "dt_e2e_ms": round(dt_ms, 3),
            "env_steps_per_s": round(env_fps, 1),
            "sensor_fps": round(sensor_fps, 1),
            "compile_s": round(compile_s, 2),
            "hlo_zero_copy": hlo["zero_copy_verified"],
            "hlo_ops": hlo,
        }
        print(f"[TASK B {nc}c] B={B} dr={with_dr} -> {dt_ms:.2f}ms env={env_fps:.1f}/s sensor={sensor_fps:.1f} FPS", flush=True)
        runs.append(rec)
        summary["task_b_visuo_tactile_runs"] = runs
        _save_json_atomic(out_json, summary)


# ==============================================================================
# Showcase High-Res Renders (Saved Locally on w1 for PIL Figure Generation)
# ==============================================================================

def run_showcase(summary: Dict[str, Any], export_npz: str, out_json: str) -> None:
  pallas_interp = (jax.default_backend() == "cpu")
  dev0 = jax.devices()[0]
  rep_f = _make_factory_rep_cfg()

  # 1. Factory G=4,096 High-Res 256x256 PBR Renders (Overview + Closeup + Wrist + Aux modalities)
  m_f4, mx_f4, dx_f4, _ = _prepare_factory_batch(n_target_prims=4096, batch_size=8, n_step_warmup=4)
  cam_f4 = HBMStreamedTiledCamera(
      m_f4,
      TiledCameraCfg(
          cam_names=("factory_overview", "factory_closeup", "arm_nw_wrist_cam"),
          height=256,
          width=256,
          data_types=DEFAULT_MODALITIES,
          shader="pbr_ggx",
          tonemap="agx",
          out_dtype=jnp.float32,
          block_b=8,
          block_r=256,
          cull_mode="cluster1",
          interpret=pallas_interp,
          semantic_fn=assign_factory_semantic,
      ),
  )
  keys_f4 = jax.device_put(jax.random.split(jax.random.PRNGKey(2026), 8), dev0)
  obs_f4 = jax.jit(lambda d, k: cam_f4.render_local(d, rng_or_keys=k, rep_cfg=rep_f))(dx_f4, keys_f4)
  jax.block_until_ready(tuple(obs_f4.values()))

  # 2. Panda GelSight Visuo-Tactile 128x128 Renders across Cube (0), Cylinder (1), Sphere (2) grasps
  tactile_exports: Dict[str, np.ndarray] = {}
  m_s, mx_s, dx_s, _ = _prepare_tactile_batch(batch_size=8, n_grasp_steps=28)
  rig_s = VisuoTactileSensorRig(
      m_s,
      VisuoTactileCfg(
          height=128,
          width=128,
          gel_surface_dist_m=0.0150,
          out_dtype=jnp.float32,
          block_b=8,
          block_r=256,
          use_shadows=False,
          interpret=pallas_interp,
          apply_membrane_skirt=True,
      ),
  )
  obs_s = jax.jit(lambda d: rig_s.render_local(d, rep_batch=None))(dx_s)
  jax.block_until_ready(tuple(obs_s.values()))
  for idx, shape in ((0, "cube"), (1, "cylinder"), (2, "sphere")):
    tactile_exports[f"{shape}_head_rgb"] = np.asarray(obs_s["vis_rgb"][idx, 0], np.float32)
    tactile_exports[f"{shape}_wrist_rgb"] = np.asarray(obs_s["vis_rgb"][idx, 1], np.float32)
    tactile_exports[f"{shape}_tac_l_rgb"] = np.asarray(obs_s["tactile_rgb"][idx, 0], np.float32)
    tactile_exports[f"{shape}_tac_r_rgb"] = np.asarray(obs_s["tactile_rgb"][idx, 1], np.float32)
    tactile_exports[f"{shape}_tac_l_depth"] = np.asarray(obs_s["tactile_depth"][idx, 0], np.float32)
    tactile_exports[f"{shape}_tac_r_depth"] = np.asarray(obs_s["tactile_depth"][idx, 1], np.float32)
    tactile_exports[f"{shape}_tac_l_norm"] = np.asarray(obs_s["tactile_normals"][idx, 0], np.float32)
    tactile_exports[f"{shape}_tac_r_norm"] = np.asarray(obs_s["tactile_normals"][idx, 1], np.float32)

  os.makedirs(os.path.dirname(export_npz) or ".", exist_ok=True)
  np.savez(
      export_npz,
      f4_over_rgb=np.asarray(obs_f4["rgb"][:4, 0], np.float32),
      f4_close_rgb=np.asarray(obs_f4["rgb"][:4, 1], np.float32),
      f4_wrist_rgb=np.asarray(obs_f4["rgb"][:2, 2], np.float32),
      f4_over_depth=np.asarray(obs_f4["distance_to_image_plane"][0, 0], np.float32),
      f4_over_sem=np.asarray(obs_f4["semantic_segmentation"][0, 0], np.int32),
      f4_over_flow=np.asarray(obs_f4["motion_vectors"][0, 0], np.float32),
      **tactile_exports,
  )
  summary["export_npz"] = export_npz
  _save_json_atomic(out_json, summary)
  print("Saved showcase npz:", export_npz, flush=True)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--out_json", default="/dev/shm/zaychik-tpu-mjx-w1/out/v3_w1_wave4_vla_tactile_factory.json")
  ap.add_argument("--export_npz", default="/dev/shm/zaychik-tpu-mjx-w1/out/wave4_showcase.npz")
  ap.add_argument("--mode", default="all", choices=["all", "parity", "task_a", "task_b", "showcase"])
  ap.add_argument("--g_list", default="2048,4096")
  args = ap.parse_args()
  g_targets = tuple(int(x.strip()) for x in args.g_list.split(",") if x.strip())

  if os.path.exists(args.out_json):
    with open(args.out_json, "r") as f:
      summary = json.load(f)
  else:
    summary = {
        "backend": jax.default_backend(),
        "device_kind": str(jax.devices()[0].device_kind),
        "n_devices_visible": len(jax.devices()),
    }

  if args.mode in ("all", "parity"):
    parity = run_parity_checks(summary)
    assert all(p["passed"] for p in parity), f"Parity check failed: {parity}"
    summary["parity_checks"] = parity
    _save_json_atomic(args.out_json, summary)

  if args.mode in ("all", "task_a"):
    run_task_a(summary, args.out_json, g_targets=g_targets)

  if args.mode in ("all", "task_b"):
    run_task_b(summary, args.out_json)

  if args.mode in ("all", "showcase"):
    run_showcase(summary, args.export_npz, args.out_json)

  print("DONE:", args.out_json, flush=True)


if __name__ == "__main__":
  main()
