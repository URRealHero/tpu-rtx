"""Worker w1: 1024x1024 Deformable Cloth BVH Refit + RoboTwin Bimanual Tabletop (120 frames each)."""

from __future__ import annotations

import argparse
import gc
import math
import os
import sys
import time

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
  sys.path.insert(0, REPO_ROOT)

sys.path.insert(0, os.path.join(REPO_ROOT, "experiments", "w2_mesh_1m_bvh"))

import tpu_rtx
from experiments.showcase_videos.video_utils import look_at_mjx_cmat, save_frames_tar
from experiments.w0_wave4_rl_gallery.multi_embodiment_gallery import (
    _dual_arm_studio_xml,
    _humanoid_studio_xml,
    _pose_embodiment,
    _quadruped_studio_xml,
    studio_agx_post_tpu,
)
from experiments.w2_mesh_1m_bvh.wave4b_deformable_cloth import (
    _build_scene_spec_for_cloth,
    _make_table_mesh,
    _make_uv_sphere_mesh,
    make_cloth_initial_state,
    pbd_cloth_step,
    refit_cloth_wide_bvh,
    shade_cloth_pbr_agx,
)
from wave3_500m_10b_8k import build_compact_bvh_mesh_item, render_compact_bvh_batch


def render_cloth_video(out_dir: str, n_frames: int = 120, res: int = 2048, clip_name: str = "cloth_bvh_refit_2k") -> None:
  sph = build_compact_bvh_mesh_item(1, _make_uv_sphere_mesh(), "spatial_l1_morton")
  tbl = build_compact_bvh_mesh_item(2, _make_table_mesh(), "spatial_l1_morton")
  static_items = [sph, tbl]
  static_devs = tuple(
      (jnp.asarray(it.l2_roots_aabb), jnp.asarray(it.l2_aabbs), jnp.asarray(it.l0_aabbs), jnp.asarray(it.leaf_bank))
      for it in static_items
  )

  oxp = jnp.zeros((1, 3, 3), dtype=jnp.float32)
  oxm = jnp.broadcast_to(jnp.eye(3, dtype=jnp.float32)[None, None, :, :], (1, 3, 3, 3))

  nx, ny = 128, 128
  pos, prev, dx, dy = make_cloth_initial_state(nx, ny)
  spec = _build_scene_spec_for_cloth(nx, ny, res, res, static_items)

  @jax.jit
  def _step_refit_render(p, pp, cxp, cxm):
    pn, ppn = pbd_cloth_step(p, pp, dx, dy)
    c_dev = refit_cloth_wide_bvh(pn)
    gb = render_compact_bvh_batch(spec, (c_dev,) + static_devs, oxp, oxm, cxp, cxm, interpret=False)
    rgb = shade_cloth_pbr_agx(gb, spec)
    return pn, ppn, rgb[0]

  target = np.array([0.04, -0.02, 0.56], dtype=np.float64)
  r_cam = 2.65
  frames_u8 = []

  def _cam_pair(i: int):
    ang = -0.65 + 2.0 * math.pi * (i / float(n_frames))
    eye = target + np.array([r_cam * math.cos(ang), r_cam * math.sin(ang), 0.72 + 0.14 * math.sin(ang)], dtype=np.float64)
    cmat = look_at_mjx_cmat(eye, target)
    cxp = jnp.asarray(eye.astype(np.float32)[None, :], dtype=jnp.float32)
    cxm = jnp.asarray(cmat[None, :, :], dtype=jnp.float32)
    return cxp, cxm

  t0 = time.perf_counter()
  cxp0, cxm0 = _cam_pair(0)
  p_w, pp_w, rgb_w = _step_refit_render(pos, prev, cxp0, cxm0)
  jax.block_until_ready(rgb_w)
  print(f"[w1:cloth] Compiled 128x128 (32k tris) + WideBVH refit @ {res}x{res} in {time.perf_counter() - t0:.1f}s", flush=True)

  p_cur, pp_cur = pos, prev
  t1 = time.perf_counter()
  for i in range(n_frames):
    cxp, cxm = _cam_pair(i)
    # 2 PBD substeps per frame for lush silky drape speed at 30 fps
    p_cur, pp_cur, _ = _step_refit_render(p_cur, pp_cur, cxp, cxm)
    p_cur, pp_cur, rgb = _step_refit_render(p_cur, pp_cur, cxp, cxm)
    jax.block_until_ready(rgb)
    frames_u8.append(np.clip(np.round(np.asarray(rgb) * 255.0), 0, 255).astype(np.uint8))

  print(f"[w1:cloth] Rendered {n_frames} frames in {time.perf_counter() - t1:.1f}s", flush=True)
  save_frames_tar(frames_u8, out_dir, "cloth_bvh_refit")


def render_robotwin_video(out_dir: str, n_frames: int = 120, res: int = 1024) -> None:
  xml = _dual_arm_studio_xml()
  m = mujoco.MjModel.from_xml_string(xml)
  mx = mjx.put_model(m, impl="jax")
  key_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "bimanual_ready")
  d_host = mujoco.MjData(m)
  if key_id >= 0:
    mujoco.mj_resetDataKeyframe(m, d_host, key_id)
  mujoco.mj_forward(m, d_host)

  qpos_base = np.asarray(d_host.qpos, dtype=np.float32).copy()
  floor_gid = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor"))
  sc = tpu_rtx.compile(
      m,
      shader="fused_ao_ggx",
      res=(res, res),
      sharding=1,
      cam_id="diag_overview",
      dtype="float32",
      outputs=("rgb", "normal", "seg_id"),
      block_b=8,
      block_r=256,
      n_ao_samples=4,
      n_shadow_samples=4,
      ao_radius=0.35,
      ao_strength=0.82,
      shadow_softness=0.05,
  )

  dx_base = mjx.make_data(m, impl="jax")

  @jax.jit
  def _kinematic_render(qpos_vec, cpos, cmat):
    dx = mjx.forward(mx, dx_base.replace(qpos=qpos_vec))
    dx = dx.replace(
        cam_xpos=dx.cam_xpos.at[int(sc.cam_ids[0])].set(cpos),
        cam_xmat=dx.cam_xmat.at[int(sc.cam_ids[0])].set(cmat),
    )
    out = tpu_rtx.render(dx, sc)
    return studio_agx_post_tpu(out["rgb"], out["normal"], out["seg_id"], floor_gid)

  # Smooth coordinated dual-arm tabletop reach + grasp dance + subtle turntable arc
  target = np.array([0.52, 0.0, 0.22], dtype=np.float64)
  frames_u8 = []

  def _inputs_for(i: int):
    phase = 2.0 * math.pi * (i / float(n_frames))
    q = qpos_base.copy()
    # Left & Right 7-DOF arms live in first 14 DOFs (shoulder/elbow/wrist oscillation)
    q[0] += 0.18 * math.sin(phase)
    q[1] += 0.16 * math.sin(phase - 0.6)
    q[3] += 0.22 * math.sin(phase * 1.0)
    q[5] += 0.24 * math.cos(phase)
    if len(q) >= 14:
      q[7] -= 0.18 * math.sin(phase + 0.3)
      q[8] += 0.16 * math.cos(phase)
      q[10] -= 0.20 * math.sin(phase)
      q[12] += 0.22 * math.sin(phase)

    ang = -0.68 + 0.52 * math.sin(phase)
    r_orb = 1.38 + 0.10 * math.cos(phase)
    eye = target + np.array([r_orb * math.cos(ang), r_orb * math.sin(ang), 0.78 + 0.08 * math.sin(phase)], dtype=np.float64)
    cmat = look_at_mjx_cmat(eye, target)
    return jnp.asarray(q, jnp.float32), jnp.asarray(eye, jnp.float32), jnp.asarray(cmat, jnp.float32)

  t0 = time.perf_counter()
  q0, p0, m0 = _inputs_for(0)
  rgb0 = _kinematic_render(q0, p0, m0)
  jax.block_until_ready(rgb0)
  print(f"[w1:robotwin] Compiled RoboTwin {res}x{res} in {time.perf_counter() - t0:.1f}s", flush=True)

  t1 = time.perf_counter()
  for i in range(n_frames):
    qi, pi, mi = _inputs_for(i)
    rgb = _kinematic_render(qi, pi, mi)
    jax.block_until_ready(rgb)
    frames_u8.append(np.clip(np.round(np.asarray(rgb) * 255.0), 0, 255).astype(np.uint8))

  print(f"[w1:robotwin] Rendered {n_frames} frames in {time.perf_counter() - t1:.1f}s", flush=True)
  save_frames_tar(frames_u8, out_dir, "robotwin_bimanual")


def _render_studio_embodiment_orbit(
    name: str,
    xml: str,
    clip_name: str,
    out_dir: str,
    base_eye: tuple,
    target: tuple,
    n_frames: int = 90,
    res: int = 1024,
) -> None:
  m = mujoco.MjModel.from_xml_string(xml)
  mx = mjx.put_model(m, impl="jax")
  d_host = _pose_embodiment(name, m)
  qpos_base = np.asarray(d_host.qpos, dtype=np.float32).copy()
  floor_gid = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor"))

  sc = tpu_rtx.compile(
      m, shader="fused_ao_ggx", res=(res, res), sharding=1, cam_id="hero", dtype="float32",
      outputs=("rgb", "normal", "seg_id"), block_b=8, block_r=512,
      n_ao_samples=8, n_shadow_samples=8, ao_radius=0.35, ao_strength=0.82, shadow_softness=0.05,
  )
  dx_base = mjx.make_data(m, impl="jax")

  @jax.jit
  def _render_pose(qpos_vec, cpos, cmat):
    dx = mjx.forward(mx, dx_base.replace(qpos=qpos_vec))
    dx = dx.replace(
        cam_xpos=dx.cam_xpos.at[int(sc.cam_ids[0])].set(cpos),
        cam_xmat=dx.cam_xmat.at[int(sc.cam_ids[0])].set(cmat),
    )
    out = tpu_rtx.render(dx, sc)
    return studio_agx_post_tpu(out["rgb"], out["normal"], out["seg_id"], floor_gid)

  e0 = np.asarray(base_eye, dtype=np.float64)
  tgt = np.asarray(target, dtype=np.float64)
  rel = e0 - tgt
  r_xy = float(np.linalg.norm(rel[:2]))
  ang0 = math.atan2(rel[1], rel[0])

  def _inputs_for(i: int):
    phase = 2.0 * math.pi * (i / float(n_frames))
    ang = ang0 + phase
    eye = np.array([
        tgt[0] + r_xy * math.cos(ang),
        tgt[1] + r_xy * math.sin(ang),
        float(e0[2]) + 0.08 * math.sin(phase),
    ], dtype=np.float64)
    cmat = look_at_mjx_cmat(eye, tgt)
    q = qpos_base.copy()
    if name == "humanoid_running" and len(q) >= 21:
      s = math.sin(phase * 2.0)
      c = math.cos(phase * 2.0)
      q[7] += 0.14 * s
      q[10] += 0.28 * s
      q[13] -= 0.28 * s
      q[15] += 0.35 * c
      q[18] -= 0.35 * c
    elif name == "quadruped_trot" and len(q) >= 19:
      s = math.sin(phase * 2.0)
      q[8] += 0.16 * s
      q[11] -= 0.16 * s
      q[14] -= 0.16 * s
      q[17] += 0.16 * s
    return jnp.asarray(q, jnp.float32), jnp.asarray(eye, jnp.float32), jnp.asarray(cmat, jnp.float32)

  t0 = time.perf_counter()
  q0, p0, m0 = _inputs_for(0)
  jax.block_until_ready(_render_pose(q0, p0, m0))
  print(f"[w1:{clip_name}] Compiled {res}x{res} in {time.perf_counter() - t0:.1f}s", flush=True)

  frames_u8 = []
  t1 = time.perf_counter()
  for i in range(n_frames):
    qi, pi, mi = _inputs_for(i)
    rgb = _render_pose(qi, pi, mi)
    jax.block_until_ready(rgb)
    frames_u8.append(np.clip(np.round(np.asarray(rgb) * 255.0), 0, 255).astype(np.uint8))
  print(f"[w1:{clip_name}] Rendered {n_frames} frames in {time.perf_counter() - t1:.1f}s", flush=True)
  save_frames_tar(frames_u8, out_dir, clip_name)


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--out_dir", default="/dev/shm/zaychik-tpu-rtx/out_videos")
  ap.add_argument("--n_frames", type=int, default=120)
  args = ap.parse_args()
  os.makedirs(args.out_dir, exist_ok=True)
  render_robotwin_video(args.out_dir, n_frames=args.n_frames)
  jax.clear_caches()
  gc.collect()
  _render_studio_embodiment_orbit(
      "humanoid_running", _humanoid_studio_xml(), "humanoid_running_orbit",
      args.out_dir, (1.95, -1.65, 1.18), (0.0, 0.0, 0.85), n_frames=90,
  )
  jax.clear_caches()
  gc.collect()
  _render_studio_embodiment_orbit(
      "quadruped_trot", _quadruped_studio_xml(), "quadruped_trot_orbit",
      args.out_dir, (1.12, -1.02, 0.60), (0.0, 0.0, 0.30), n_frames=90,
  )
  jax.clear_caches()
  gc.collect()
  render_cloth_video(args.out_dir, n_frames=120, res=2048, clip_name="cloth_bvh_refit_2k")


if __name__ == "__main__":
  main()
