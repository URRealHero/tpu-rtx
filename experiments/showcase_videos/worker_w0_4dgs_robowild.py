"""Worker w0: 4-chip TPU RoboWild Dynamic 4DGS x MJX Panda Contact Push (1080p, 105 frames)."""

from __future__ import annotations

import argparse
import gc
import inspect
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

from experiments.showcase_videos.video_utils import save_frames_tar
from experiments.showcase_videos.worker_w2_mipnerf_garden import render_scene_orbit
from experiments.w3_4dgs_robowild.fetch_mipnerf360 import fetch_scenes
from experiments.w3_4dgs_robowild.run_robowild_4dgs import (
    _quat_mul_jnp,
    _rotmat_to_quat_jnp,
    _rotmat_to_quat_np,
    _scaled_K,
    _shard_scene_with_mask,
    build_robowild_4dgs_xml,
    fit_full_table_plane,
    plan_smooth_panda_push_trajectory,
    segment_lego_bulldozer,
)
from tpu_mjx_render.adapters.tpu_gsplat import (
    composite_mjx_and_3dgs,
    load_cameras_json,
    load_ply,
    make_sharded_rasterizer,
)
from tpu_mjx_render.renderer import RenderSceneSpec

sys.path.insert(0, os.path.join(REPO_ROOT, "experiments", "w3_realworld_3dgs"))
from robowild_kitchen import compute_opencv_rays, pallas_raycast_mjx_table_strip


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--data_dir", default="/dev/shm/zaychik-tpu-rtx/mipnerf360")
  ap.add_argument("--out_dir", default="/dev/shm/zaychik-tpu-rtx/out_videos")
  ap.add_argument("--n_frames", type=int, default=90)
  args = ap.parse_args()

  os.makedirs(args.data_dir, exist_ok=True)
  os.makedirs(args.out_dir, exist_ok=True)
  fetch_scenes(args.data_dir, ("kitchen",))

  devs = jax.devices()[:4]
  mesh = jax.sharding.Mesh(np.array(devs), ("dp",))
  rep = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
  P = jax.sharding.PartitionSpec
  sm_kw = {"check_vma": False} if "check_vma" in inspect.signature(jax.shard_map).parameters else {"check_rep": False}

  sp = load_ply(os.path.join(args.data_dir, "kitchen/point_cloud/iteration_30000/point_cloud.ply"))
  cams = load_cameras_json(os.path.join(args.data_dir, "kitchen/cameras.json"))
  c0 = cams[0]

  W_fit, H_fit = 1920, 1080
  K_fit = _scaled_K(c0, W_fit, H_fit)
  fn_init = make_sharded_rasterizer(
      mesh, W_fit, H_fit, sh_degree=3, macro_size=32, macro_rep=5,
      k_macro=10240, k_large_per_chip=1536, k_tile=512,
  )
  m0, q0, s0, o0, sh0, _, _ = _shard_scene_with_mask(sp, np.zeros((sp["n_gaussians"],), dtype=bool), mesh)
  v_dev = jax.device_put(jnp.asarray(c0["viewmat"]), rep)
  k_fit_dev = jax.device_put(jnp.asarray(K_fit), rep)
  _, _, dep_init = fn_init(m0, q0, s0, o0, sh0, v_dev, k_fit_dev)
  jax.block_until_ready(dep_init)

  tbl = fit_full_table_plane(
      np.asarray(dep_init[..., 0]),
      c0["rotation"], c0["position"],
      float(K_fit[0, 0]), float(K_fit[1, 1]), 0.5 * W_fit, 0.5 * H_fit,
      place_xy_1080=(450.0, 645.0),
      target_xy_1080=(900.0, 635.0),
  )
  del fn_init, m0, q0, s0, o0, sh0, dep_init
  gc.collect()
  fg_mask, seg_info = segment_lego_bulldozer(sp, c0, tbl)
  m_orig, q_orig, s_orig, o_orig, sh_orig, is_dyn, _ = _shard_scene_with_mask(sp, fg_mask, mesh)

  W, H = 3840, 2160
  K_4k = _scaled_K(c0, W, H)
  k_dev = jax.device_put(jnp.asarray(K_4k), rep)
  fn_4k = make_sharded_rasterizer(
      mesh, W, H, sh_degree=3, macro_size=64, macro_rep=4,
      k_macro=8192, k_large_per_chip=2048, k_tile=512,
  )

  robot_scale = 1.48
  table_z_mjx = float(seg_info["z_floor"] / robot_scale)
  e_long = np.asarray(seg_info["e_long"], dtype=np.float32)
  e_lat = np.asarray(seg_info["e_lat"], dtype=np.float32)
  hx_mjx, hy_mjx, hz_mjx = [v / robot_scale for v in seg_info["half_extents_local"]]
  xy_ctr_mjx = np.asarray(seg_info["xy_box_ctr"], dtype=np.float32) / robot_scale
  dozer_pos_0 = np.array([xy_ctr_mjx[0], xy_ctr_mjx[1], table_z_mjx + hz_mjx], dtype=np.float32)
  R_dozer_0 = np.column_stack([
      np.array([e_long[0], e_long[1], 0.0], dtype=np.float32),
      np.array([e_lat[0], e_lat[1], 0.0], dtype=np.float32),
      np.array([0.0, 0.0, 1.0], dtype=np.float32),
  ])
  dozer_quat_0 = _rotmat_to_quat_np(R_dozer_0)

  xml_str = build_robowild_4dgs_xml(
      tuple(float(v) for v in dozer_pos_0),
      tuple(float(v) for v in dozer_quat_0),
      (float(hx_mjx), float(hy_mjx), float(hz_mjx)),
      table_z_mjx,
  )
  mj_model = mujoco.MjModel.from_xml_string(xml_str)
  mjx_model = mjx.put_model(mj_model, impl="jax")
  dozer_bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "lego_dozer")

  n_frames = args.n_frames
  n_substeps = 1
  q_traj_np = plan_smooth_panda_push_trajectory(
      mj_model, dozer_pos_0, e_long, e_lat, hx_mjx, hy_mjx, table_z_mjx, n_frames=n_frames
  )

  d_init = mjx.make_data(mj_model, impl="jax")
  qpos0 = mj_model.qpos0.copy()
  qpos0[:7] = q_traj_np[0]
  d_init = d_init.replace(qpos=jnp.asarray(qpos0, jnp.float32), ctrl=jnp.asarray(q_traj_np[0], jnp.float32))
  d_init = mjx.forward(mjx_model, d_init)

  R_tbl = jnp.asarray(tbl["R_table"], dtype=jnp.float32)
  P_org = jnp.asarray(tbl["P_origin"], dtype=jnp.float32)
  t_nrm = jnp.asarray(tbl["table_normal"], dtype=jnp.float32)
  t_d = jnp.asarray(tbl["table_d"], dtype=jnp.float32)
  sun_local = jnp.array([-0.52, 0.52, 0.74], dtype=jnp.float32)
  sun_local = sun_local / jnp.linalg.norm(sun_local)
  sun_dir_w = R_tbl @ sun_local
  P_0_w = (jnp.asarray(dozer_pos_0) * robot_scale) @ R_tbl.T + P_org
  R_0_w = R_tbl @ jnp.asarray(R_dozer_0, dtype=jnp.float32)

  spec_strip = RenderSceneSpec.from_mjmodel(mj_model, cam_id=0, height=H // 4, width=W, use_shadows=True)
  _sm_mjx = jax.shard_map(
      lambda gxp, gxm, gsz, grg, sd, tn, td, ro, rd, cz: pallas_raycast_mjx_table_strip(
          spec_strip, gxp, gxm, gsz, grg, sd, tn, td, ro, rd, cz
      ),
      mesh=mesh,
      in_specs=(P(), P(), P(), P(), P(), P(), P(), P("dp"), P("dp"), P("dp")),
      out_specs=(P("dp"), P("dp"), P("dp"), P("dp"), P("dp"), P("dp"), P("dp")),
      **sm_kw,
  )
  _sm_update_4dgs = jax.shard_map(
      lambda mo, qo, dyn, pt, rd, qd: (
          jnp.where(dyn[:, None], pt[None, :] + (mo - P_0_w[None, :]) @ rd.T, mo),
          jnp.where(dyn[:, None], _quat_mul_jnp(qd[None, :], qo), qo),
      ),
      mesh=mesh,
      in_specs=(P("dp"), P("dp"), P("dp"), P(), P(), P()),
      out_specs=(P("dp"), P("dp")),
      **sm_kw,
  )

  @jax.jit
  def _step_and_render(d_in, ctrl_t, mo, qo, so, oo, sho, dyn, vm, Km, cp, rc2w):
    d_c = d_in.replace(ctrl=ctrl_t)
    d_out = jax.lax.fori_loop(0, n_substeps, lambda _, d_s: mjx.step(mjx_model, d_s), d_c)

    pos_t_mjx = d_out.xpos[dozer_bid]
    mat_t_mjx = d_out.xmat[dozer_bid]
    P_t_w = (pos_t_mjx * robot_scale) @ R_tbl.T + P_org
    R_t_w = R_tbl @ mat_t_mjx
    R_delta = R_t_w @ R_0_w.T
    q_delta = _rotmat_to_quat_jnp(R_delta)
    m_live, q_live = _sm_update_4dgs(mo, qo, dyn, P_t_w, R_delta, q_delta)

    g_xpos = (d_out.geom_xpos * robot_scale) @ R_tbl.T + P_org[None, :]
    g_xmat = jnp.einsum("ij,gjk->gik", R_tbl, d_out.geom_xmat, precision=jax.lax.Precision.HIGHEST)
    g_size = jnp.asarray(spec_strip.geom_size, dtype=jnp.float32) * robot_scale
    g_rgba = jnp.asarray(spec_strip.geom_rgba, dtype=jnp.float32)

    ro, rd, cz = compute_opencv_rays(cp, rc2w, Km, H, W)
    alb_s, nrm_s, zd_s, rsh_s, tsh_s, tz_s, vd_s = _sm_mjx(
        g_xpos, g_xmat, g_size, g_rgba, sun_dir_w, t_nrm, t_d, ro, rd, cz
    )
    alb = alb_s.reshape(H, W, 3)
    p_nrm = nrm_s.reshape(H, W, 3)
    p_vd = vd_s.reshape(H, W, 3)
    r_sh = rsh_s.reshape(H, W)
    mjx_z = zd_s.reshape(H, W)
    tbl_sh = tsh_s.reshape(H, W)
    tbl_z = tz_s.reshape(H, W)

    ndotl = jnp.maximum(0.0, jnp.sum(p_nrm * sun_dir_w[None, None, :], axis=-1))
    hemi = 0.5 + 0.5 * jnp.sum(p_nrm * t_nrm[None, None, :], axis=-1)
    hvec = sun_dir_w[None, None, :] + p_vd
    hvec = hvec / jnp.maximum(jnp.linalg.norm(hvec, axis=-1, keepdims=True), 1e-12)
    spec_term = (jnp.maximum(0.0, jnp.sum(p_nrm * hvec, axis=-1)) ** 32.0) * 0.25 * r_sh
    lit = (0.36 + 0.22 * hemi + 0.56 * ndotl * r_sh)[..., None] * alb + spec_term[..., None]
    mjx_rgb = jnp.clip(lit, 0.0, 1.0)

    g_rgb, g_alp, g_dep = fn_4k(m_live, q_live, so, oo, sho, vm, Km)
    g_alp2 = g_alp[..., 0]
    g_dep2 = g_dep[..., 0]
    on_tbl = (tbl_sh > 0.5) & (g_dep2 > 0.0) & (jnp.abs(g_dep2 - tbl_z) < 0.25)
    g_rgb_sh = jnp.where(on_tbl[..., None], g_rgb * jnp.array([0.44, 0.45, 0.48], jnp.float32), g_rgb)
    comp_rgb, _ = composite_mjx_and_3dgs(mjx_rgb, mjx_z, g_rgb_sh, g_alp2, g_dep2)
    return d_out, comp_rgb

  d_cur = jax.device_put(d_init, rep)
  q_traj_dev = jax.device_put(jnp.asarray(q_traj_np), rep)
  cp_dev = jax.device_put(jnp.asarray(c0["position"]), rep)
  rc2w_dev = jax.device_put(jnp.asarray(c0["rotation"]), rep)

  # Compile once
  t0 = time.perf_counter()
  _, warmup_rgb = _step_and_render(d_cur, q_traj_dev[0], m_orig, q_orig, s_orig, o_orig, sh_orig, is_dyn, v_dev, k_dev, cp_dev, rc2w_dev)
  jax.block_until_ready(warmup_rgb)
  print(f"[w0] Compiled Native 4K 4DGSxMJX in {time.perf_counter() - t0:.1f}s", flush=True)

  frames_u8 = []
  t1 = time.perf_counter()
  for fi in range(n_frames):
    d_cur, rgb = _step_and_render(d_cur, q_traj_dev[fi], m_orig, q_orig, s_orig, o_orig, sh_orig, is_dyn, v_dev, k_dev, cp_dev, rc2w_dev)
    jax.block_until_ready(rgb)
    frames_u8.append(np.clip(np.round(np.asarray(rgb) * 255.0), 0, 255).astype(np.uint8))
  print(f"[w0] Rendered {n_frames} frames @ Native 4K in {time.perf_counter() - t1:.1f}s", flush=True)
  save_frames_tar(frames_u8, args.out_dir, "robowild_4dgs_push_4k")

  del frames_u8, m_orig, q_orig, s_orig, o_orig, sh_orig, is_dyn, fn_4k, sp
  jax.clear_caches()
  gc.collect()

  render_scene_orbit(args.data_dir, "counter", "mipnerf_counter_orbit", args.out_dir, n_frames=120)
  jax.clear_caches()
  gc.collect()
  render_scene_orbit(args.data_dir, "room", "mipnerf_room_orbit", args.out_dir, n_frames=120)


if __name__ == "__main__":
  main()
