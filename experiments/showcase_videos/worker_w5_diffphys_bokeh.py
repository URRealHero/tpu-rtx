"""Worker w5: Differentiable Physics Servoing (768x768) + 4-Chip 720p f/1.4 Thin-Lens Focus Pull."""

from __future__ import annotations

import argparse
import gc
import inspect
import math
import os
import sys
import time

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
  sys.path.insert(0, REPO_ROOT)

import tpu_rtx
from experiments.showcase_videos.video_utils import save_frames_tar, smoothstep
from experiments.w0_wave5_menagerie.run_diff_phys_render import (
    _gaussian_blur_2d,
    _panda_torque_arm_xml,
    render_diff_pbr,
)
from experiments.w0_wave5_menagerie.run_menagerie_showcase import studio_agx_post_tpu
from experiments.w6_wave5_surround_bokeh.thinlens_bokeh_4k import (
    _aces_tonemap_srgb,
    _build_thinlens_strip_arrays,
    _shade_thinlens_hdr,
    make_bokeh_hero_xml,
)
from tpu_mjx_render.pallas_v2 import (
    BLOCK_B,
    BLOCK_R,
    _compute_geom_bound_sq,
    _extract_light_batch,
    _ray_one_geom_v2,
)
from tpu_mjx_render.primitives import MJ_MAXVAL
from tpu_mjx_render.renderer import RenderSceneSpec


def render_diffphys_servoing_video(out_dir: str, n_frames: int = 120, res: int = 768) -> None:
  xml = _panda_torque_arm_xml()
  m = mujoco.MjModel.from_xml_string(xml)
  mx = mjx.put_model(m, impl="jax")
  spec_grad = RenderSceneSpec.from_mjmodel(m, cam_id=0, height=64, width=64, enabled_geom_groups=(0, 1, 2, 3))

  q_init = np.array([-0.40, 0.20, 0.0, -1.65, 0.0, 1.55, 0.78], dtype=np.float32)
  d0 = mjx.make_data(m, impl="jax").replace(qpos=jnp.asarray(q_init))
  d0 = mjx.forward(mx, d0)

  horizon = 12
  n_iters = 240
  t_ax = np.linspace(0.0, 1.0, horizon, dtype=np.float32)[:, None]
  ctrl_gt = np.hstack([
      1.15 * np.ones_like(t_ax),
      -0.35 * np.ones_like(t_ax),
      0.15 * np.ones_like(t_ax),
      0.75 * np.ones_like(t_ax),
      0.30 * np.sin(math.pi * t_ax),
      -0.25 * np.ones_like(t_ax),
      0.15 * np.ones_like(t_ax),
  ]).astype(np.float32)

  def _rollout_render(ctrl_seq, sigma=0.008):
    def _step(d, u):
      d_next = mjx.step(mx, d.replace(ctrl=jnp.clip(u, -1.5, 1.5)))
      return d_next, None
    d_final, _ = jax.lax.scan(_step, d0, ctrl_seq)
    rgb = render_diff_pbr(spec_grad, d_final, sigma=sigma, tau_z=0.015)
    return rgb, d_final

  target_rgb, _ = jax.jit(lambda c: _rollout_render(c, sigma=0.004))(jnp.asarray(ctrl_gt))
  jax.block_until_ready(target_rgb)
  target_b4 = jax.jit(lambda x: _gaussian_blur_2d(x, 2.5, ksize=9))(target_rgb)
  target_b12 = jax.jit(lambda x: _gaussian_blur_2d(x, 6.0, ksize=15))(target_rgb)
  target_b24 = jax.jit(lambda x: _gaussian_blur_2d(x, 12.0, ksize=25))(target_rgb)

  def _loss_fn(ctrl_seq, sigma_cur, w_fine, w_coarse):
    rgb, _ = _rollout_render(ctrl_seq, sigma=sigma_cur)
    l_fine = jnp.mean((rgb - target_rgb) ** 2)
    l_mid = jnp.mean((_gaussian_blur_2d(rgb, 2.5, ksize=9) - target_b4) ** 2)
    l_coarse = jnp.mean((_gaussian_blur_2d(rgb, 6.0, ksize=15) - target_b12) ** 2)
    l_global = jnp.mean((_gaussian_blur_2d(rgb, 12.0, ksize=25) - target_b24) ** 2)
    dctrl = ctrl_seq[1:] - ctrl_seq[:-1]
    reg = 2e-6 * jnp.mean(ctrl_seq ** 2) + 2e-5 * jnp.mean(dctrl ** 2)
    return w_fine * l_fine + w_coarse * (1.5 * l_mid + 3.0 * l_coarse + 6.0 * l_global) + reg

  @jax.jit
  def _train_step(ctrl_seq, opt_m, opt_v, step_idx):
    t_frac = step_idx.astype(jnp.float32) / float(max(n_iters - 1, 1))
    sigma_cur = 0.018 * (1.0 - t_frac) + 0.004 * t_frac
    w_fine = jnp.clip((t_frac - 0.15) / 0.35, 0.05, 1.0)
    w_coarse = jnp.maximum(0.05, 1.0 - 0.95 * t_frac)
    loss_val, grad = jax.value_and_grad(_loss_fn)(ctrl_seq, sigma_cur, w_fine, w_coarse)
    cur_lr = 0.04 * (0.05 + 0.95 * 0.5 * (1.0 + jnp.cos(math.pi * t_frac)))
    b1, b2, eps = 0.9, 0.999, 5e-5
    opt_m = b1 * opt_m + (1.0 - b1) * grad
    opt_v = b2 * opt_v + (1.0 - b2) * (grad * grad)
    m_hat = opt_m / (1.0 - b1 ** (step_idx + 1))
    v_hat = opt_v / (1.0 - b2 ** (step_idx + 1))
    ctrl_next = jnp.clip(ctrl_seq - cur_lr * m_hat / (jnp.sqrt(v_hat) + eps), -1.5, 1.5)
    return ctrl_next, opt_m, opt_v, loss_val

  ctrl_cur = jnp.zeros((horizon, m.nu), dtype=jnp.float32)
  opt_m = jnp.zeros_like(ctrl_cur)
  opt_v = jnp.zeros_like(ctrl_cur)
  ctrl_snapshots = []
  stride = max(1, n_iters // n_frames)

  t0 = time.perf_counter()
  ctrl_cur, opt_m, opt_v, loss0 = _train_step(ctrl_cur, opt_m, opt_v, jnp.int32(0))
  jax.block_until_ready(loss0)
  ctrl_cur = jnp.zeros((horizon, m.nu), dtype=jnp.float32)
  opt_m = jnp.zeros_like(ctrl_cur)
  opt_v = jnp.zeros_like(ctrl_cur)
  for it in range(n_iters):
    ctrl_cur, opt_m, opt_v, _ = _train_step(ctrl_cur, opt_m, opt_v, jnp.int32(it))
    if it % stride == 0 and len(ctrl_snapshots) < n_frames:
      ctrl_snapshots.append(np.asarray(ctrl_cur))
  jax.block_until_ready(ctrl_cur)
  while len(ctrl_snapshots) < n_frames:
    ctrl_snapshots.append(np.asarray(ctrl_cur))
  print(f"[w5:diffphys] Solved 240 value_and_grad iterations in {time.perf_counter() - t0:.1f}s", flush=True)

  # High-res 768x768 studio PBR render of each optimization step's rolled-out end pose
  sc_hi = tpu_rtx.compile(
      m, shader="fused_ao_ggx", res=(res, res), sharding=1, cam_id=0, dtype="float32",
      outputs=("rgb", "normal", "seg_id"), block_b=8, block_r=512,
      n_ao_samples=8, n_shadow_samples=8, ao_radius=0.30, ao_strength=0.80, shadow_softness=0.05,
  )
  floor_gid = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor"))

  @jax.jit
  def _render_ctrl_hi(c_seq):
    def _step(d, u):
      return mjx.step(mx, d.replace(ctrl=jnp.clip(u, -1.5, 1.5))), None
    d_f, _ = jax.lax.scan(_step, d0, c_seq)
    out = tpu_rtx.render(d_f, sc_hi)
    return studio_agx_post_tpu(out["rgb"], out["normal"], out["seg_id"], floor_gid=floor_gid)

  jax.block_until_ready(_render_ctrl_hi(jnp.asarray(ctrl_snapshots[0])))
  frames_u8 = []
  for c_np in ctrl_snapshots:
    rgb = _render_ctrl_hi(jnp.asarray(c_np))
    jax.block_until_ready(rgb)
    frames_u8.append(np.clip(np.round(np.asarray(rgb) * 255.0), 0, 255).astype(np.uint8))
  save_frames_tar(frames_u8, out_dir, "diffphys_servoing")


def _pallas_thinlens_raycast_dyn_focus(
    spec: RenderSceneSpec,
    cam_pos: jax.Array,
    cam_mat: jax.Array,
    geom_xpos: jax.Array,
    geom_xmat: jax.Array,
    light_xpos: jax.Array,
    light_xdir: jax.Array,
    lray_strip: jax.Array,
    uv_8: jax.Array,
    lens_radius: float,
    focus_dist_dyn: jax.Array,  # scalar () float32
    interpret: bool = False,
):
  """Dynamic focus_dist variant of `_pallas_thinlens_raycast_8samples` (zero recompilation across focus pull)."""
  R_strip = lray_strip.shape[1]
  gids = np.asarray(spec.enabled_geom_ids, dtype=np.int32)
  G = len(gids)
  loc_block_r = 256 if (G > 36 or (R_strip % BLOCK_R != 0)) else BLOCK_R
  assert R_strip % loc_block_r == 0, (R_strip, loc_block_r)

  B_pad = BLOCK_B
  G_pad = ((G + 127) // 128) * 128
  n_r_blk = R_strip // loc_block_r

  cam_row = jnp.concatenate([cam_pos, cam_mat.reshape(9)], axis=0)
  cam_b = jnp.broadcast_to(cam_row[None, :], (B_pad, 12))
  focus_col = jnp.full((B_pad, 1), focus_dist_dyn, dtype=jnp.float32)
  cam_128 = jnp.pad(jnp.concatenate([cam_b, uv_8, focus_col], axis=-1), ((0, 0), (0, 128 - 15)))

  geom_size = jnp.asarray(spec.geom_size, dtype=jnp.float32)
  gpos = jnp.broadcast_to(geom_xpos[gids][None, :, :], (B_pad, G, 3))
  gmat = jnp.broadcast_to(geom_xmat[gids].reshape(G, 9)[None, :, :], (B_pad, G, 9))
  gsz_all = jnp.broadcast_to(geom_size[gids][None, :, :], (B_pad, G, 3))
  rb_sq = jnp.broadcast_to(jnp.asarray(_compute_geom_bound_sq(spec, gids))[None, :, None], (B_pad, G, 1))
  g_16 = jnp.pad(jnp.concatenate([gpos, gmat, gsz_all, rb_sq], axis=-1), ((0, 0), (0, G_pad - G), (0, 0)))
  geom_fields = g_16.transpose(2, 0, 1).reshape(16 * BLOCK_B, G_pad)

  shadow_l_idx = -1
  for li in range(int(spec.nlight)):
    if spec.use_shadows and bool(spec.light_active[li]) and bool(spec.light_castshadow[li]):
      shadow_l_idx = li
      break

  light_128 = jnp.zeros((B_pad, 128), dtype=jnp.float32)
  if shadow_l_idx >= 0:
    is_dir = 1.0 if int(spec.light_type[shadow_l_idx]) == 1 else 0.0
    lp = light_xpos[shadow_l_idx]
    ld_v = light_xdir[shadow_l_idx]
    ld_v = ld_v / jnp.maximum(jnp.linalg.norm(ld_v), 1e-12)
    l_row = jnp.broadcast_to(jnp.concatenate([jnp.array([is_dir], jnp.float32), lp, ld_v])[None, :], (B_pad, 7))
    light_128 = jnp.pad(l_row, ((0, 0), (0, 128 - 7)))

  g_types_static = tuple(int(spec.geom_type[g]) for g in gids)
  gids_static = tuple(int(g) for g in gids)
  plane_fx = tuple(float(spec.geom_size[g, 0]) > 0.0 for g in gids)
  plane_fy = tuple(float(spec.geom_size[g, 1]) > 0.0 for g in gids)
  znear_f = float(spec.znear)
  zfar_f = float(spec.zfar)
  lens_r_f = float(lens_radius)
  do_in_vmem_shadow = (shadow_l_idx >= 0) and (G <= 56)

  def _thinlens_kernel(
      cam_ref, lray_ref, gf_ref, light_ref,
      out_d_ref, out_nx_ref, out_ny_ref, out_nz_ref, out_gid_ref, out_sh_ref,
      out_rox_ref, out_roy_ref, out_roz_ref, out_rdx_ref, out_rdy_ref, out_rdz_ref,
  ):
    cam = cam_ref[...]
    lr = lray_ref[...]
    gf = gf_ref[...]
    linfo = light_ref[...]
    ux, uy, h1, h2 = lr[0:1, :], lr[1:2, :], lr[2:3, :], lr[3:4, :]

    cpx, cpy, cpz = cam[:, 0:1], cam[:, 1:2], cam[:, 2:3]
    c00, c01, c02 = cam[:, 3:4], cam[:, 4:5], cam[:, 5:6]
    c10, c11, c12 = cam[:, 6:7], cam[:, 7:8], cam[:, 8:9]
    c20, c21, c22 = cam[:, 9:10], cam[:, 10:11], cam[:, 11:12]
    u1_s, u2_s = cam[:, 12:13], cam[:, 13:14]
    focus_d = cam[:, 14:15]

    if lens_r_f > 0.0:
      u1 = (u1_s + h1) - jnp.floor(u1_s + h1)
      u2 = (u2_s + h2) - jnp.floor(u2_s + h2)
      a = 2.0 * u1 - 1.0
      b = 2.0 * u2 - 1.0
      use_a = jnp.abs(a) > jnp.abs(b)
      safe_a = jnp.where(jnp.abs(a) > 1e-12, a, 1.0)
      safe_b = jnp.where(jnp.abs(b) > 1e-12, b, 1.0)
      r_disk = jnp.where(use_a, a, b)
      phi_disk = jnp.where(use_a, (math.pi / 4.0) * (b / safe_a), (math.pi / 2.0) - (math.pi / 4.0) * (a / safe_b))
      zero_ab = (jnp.abs(a) <= 1e-12) & (jnp.abs(b) <= 1e-12)
      r_disk = jnp.where(zero_ab, 0.0, r_disk)
      lx = lens_r_f * r_disk * jnp.cos(phi_disk)
      ly = lens_r_f * r_disk * jnp.sin(phi_disk)
    else:
      lx = jnp.zeros((BLOCK_B, loc_block_r), dtype=jnp.float32)
      ly = jnp.zeros((BLOCK_B, loc_block_r), dtype=jnp.float32)

    dcx_raw = focus_d * ux - lx
    dcy_raw = focus_d * uy - ly
    dcz_raw = -jnp.broadcast_to(focus_d, (BLOCK_B, loc_block_r))
    inv_dc = 1.0 / jnp.maximum(jnp.sqrt(dcx_raw * dcx_raw + dcy_raw * dcy_raw + dcz_raw * dcz_raw), 1e-12)
    dcx, dcy, dcz = dcx_raw * inv_dc, dcy_raw * inv_dc, dcz_raw * inv_dc

    rdx_raw = c00 * dcx + c01 * dcy + c02 * dcz
    rdy_raw = c10 * dcx + c11 * dcy + c12 * dcz
    rdz_raw = c20 * dcx + c21 * dcy + c22 * dcz
    inv_rn = 1.0 / jnp.maximum(jnp.sqrt(rdx_raw * rdx_raw + rdy_raw * rdy_raw + rdz_raw * rdz_raw), 1e-12)
    rdx, rdy, rdz = rdx_raw * inv_rn, rdy_raw * inv_rn, rdz_raw * inv_rn

    neg_z = jnp.maximum(-dcz, 1e-6)
    inv_cos = znear_f / neg_z
    rox = cpx + (c00 * lx + c01 * ly) + inv_cos * rdx
    roy = cpy + (c10 * lx + c11 * ly) + inv_cos * rdy
    roz = cpz + (c20 * lx + c21 * ly) + inv_cos * rdz

    max_d = (zfar_f - znear_f) / neg_z if zfar_f > 0.0 else jnp.full((BLOCK_B, loc_block_r), MJ_MAXVAL, dtype=jnp.float32)
    best_d = max_d
    best_nx = jnp.zeros((BLOCK_B, loc_block_r), dtype=jnp.float32)
    best_ny = jnp.zeros((BLOCK_B, loc_block_r), dtype=jnp.float32)
    best_nz = jnp.zeros((BLOCK_B, loc_block_r), dtype=jnp.float32)
    best_gid = jnp.full((BLOCK_B, loc_block_r), -1, dtype=jnp.int32)

    for k, (gt, gid_val, pfx, pfy) in enumerate(zip(g_types_static, gids_static, plane_fx, plane_fy)):
      px, py, pz = gf[0 * 8 : 1 * 8, k : k + 1], gf[1 * 8 : 2 * 8, k : k + 1], gf[2 * 8 : 3 * 8, k : k + 1]
      m00, m01, m02 = gf[3 * 8 : 4 * 8, k : k + 1], gf[4 * 8 : 5 * 8, k : k + 1], gf[5 * 8 : 6 * 8, k : k + 1]
      m10, m11, m12 = gf[6 * 8 : 7 * 8, k : k + 1], gf[7 * 8 : 8 * 8, k : k + 1], gf[8 * 8 : 9 * 8, k : k + 1]
      m20, m21, m22 = gf[9 * 8 : 10 * 8, k : k + 1], gf[10 * 8 : 11 * 8, k : k + 1], gf[11 * 8 : 12 * 8, k : k + 1]
      sx, sy, sz = gf[12 * 8 : 13 * 8, k : k + 1], gf[13 * 8 : 14 * 8, k : k + 1], gf[14 * 8 : 15 * 8, k : k + 1]
      rb_sq = gf[15 * 8 : 16 * 8, k : k + 1]
      cd, cnx, cny, cnz = _ray_one_geom_v2(
          gt, px, py, pz, m00, m01, m02, m10, m11, m12, m20, m21, m22,
          sx, sy, sz, rb_sq, rox, roy, roz, rdx, rdy, rdz,
          plane_finite_x=pfx, plane_finite_y=pfy, compute_normal=True,
      )
      imp = cd < best_d
      best_d = jnp.where(imp, cd, best_d)
      best_gid = jnp.where(imp, jnp.int32(gid_val), best_gid)
      best_nx = jnp.where(imp, cnx, best_nx)
      best_ny = jnp.where(imp, cny, best_ny)
      best_nz = jnp.where(imp, cnz, best_nz)

    hit = best_gid >= 0
    best_d_out = jnp.where(hit, best_d, -1.0)
    out_nx_v = jnp.where(hit, best_nx, 0.0)
    out_ny_v = jnp.where(hit, best_ny, 0.0)
    out_nz_v = jnp.where(hit, best_nz, 0.0)

    sh_occ = jnp.zeros((BLOCK_B, loc_block_r), dtype=jnp.float32)
    if do_in_vmem_shadow:
      d_safe = jnp.where(hit, best_d, 0.0)
      hx, hy, hz = rox + d_safe * rdx + out_nx_v * 1e-4, roy + d_safe * rdy + out_ny_v * 1e-4, roz + d_safe * rdz + out_nz_v * 1e-4
      is_dir = linfo[:, 0:1] > 0.5
      dlx, dly, dlz = linfo[:, 1:2] - hx, linfo[:, 2:3] - hy, linfo[:, 3:4] - hz
      d_pt = jnp.sqrt(jnp.maximum(dlx * dlx + dly * dly + dlz * dlz, 1e-12))
      lx_v = jnp.where(is_dir, -linfo[:, 4:5], dlx / d_pt)
      ly_v = jnp.where(is_dir, -linfo[:, 5:6], dly / d_pt)
      lz_v = jnp.where(is_dir, -linfo[:, 6:7], dlz / d_pt)
      ndotl = out_nx_v * lx_v + out_ny_v * ly_v + out_nz_v * lz_v
      sh_max = jnp.where(hit & (ndotl > 0.0), jnp.where(is_dir, 1.0e8, jnp.maximum(d_pt - 1.0e-3, 0.0)), 0.0)
      occ_bool = jnp.zeros((BLOCK_B, loc_block_r), dtype=jnp.bool_)
      for k, (gt, pfx, pfy) in enumerate(zip(g_types_static, plane_fx, plane_fy)):
        px, py, pz = gf[0 * 8 : 1 * 8, k : k + 1], gf[1 * 8 : 2 * 8, k : k + 1], gf[2 * 8 : 3 * 8, k : k + 1]
        m00, m01, m02 = gf[3 * 8 : 4 * 8, k : k + 1], gf[4 * 8 : 5 * 8, k : k + 1], gf[5 * 8 : 6 * 8, k : k + 1]
        m10, m11, m12 = gf[6 * 8 : 7 * 8, k : k + 1], gf[7 * 8 : 8 * 8, k : k + 1], gf[8 * 8 : 9 * 8, k : k + 1]
        m20, m21, m22 = gf[9 * 8 : 10 * 8, k : k + 1], gf[10 * 8 : 11 * 8, k : k + 1], gf[11 * 8 : 12 * 8, k : k + 1]
        sx, sy, sz = gf[12 * 8 : 13 * 8, k : k + 1], gf[13 * 8 : 14 * 8, k : k + 1], gf[14 * 8 : 15 * 8, k : k + 1]
        rb_sq = gf[15 * 8 : 16 * 8, k : k + 1]
        scd, _, _, _ = _ray_one_geom_v2(
            gt, px, py, pz, m00, m01, m02, m10, m11, m12, m20, m21, m22,
            sx, sy, sz, rb_sq, hx, hy, hz, lx_v, ly_v, lz_v,
            plane_finite_x=pfx, plane_finite_y=pfy, compute_normal=False,
        )
        occ_bool = occ_bool | (scd < sh_max)
      sh_occ = occ_bool.astype(jnp.float32)

    out_d_ref[...] = best_d_out
    out_nx_ref[...], out_ny_ref[...], out_nz_ref[...] = out_nx_v, out_ny_v, out_nz_v
    out_gid_ref[...], out_sh_ref[...] = best_gid, sh_occ
    out_rox_ref[...], out_roy_ref[...], out_roz_ref[...] = rox, roy, roz
    out_rdx_ref[...], out_rdy_ref[...], out_rdz_ref[...] = rdx, rdy, rdz

  outs = pl.pallas_call(
      _thinlens_kernel,
      out_shape=[jax.ShapeDtypeStruct((B_pad, R_strip), jnp.float32) for _ in range(4)]
      + [jax.ShapeDtypeStruct((B_pad, R_strip), jnp.int32)]
      + [jax.ShapeDtypeStruct((B_pad, R_strip), jnp.float32) for _ in range(7)],
      grid=(1, n_r_blk),
      in_specs=[
          pl.BlockSpec((BLOCK_B, 128), lambda bi, ri: (0, 0)),
          pl.BlockSpec((8, loc_block_r), lambda bi, ri: (0, ri)),
          pl.BlockSpec((16 * BLOCK_B, G_pad), lambda bi, ri: (0, 0)),
          pl.BlockSpec((BLOCK_B, 128), lambda bi, ri: (0, 0)),
      ],
      out_specs=[pl.BlockSpec((BLOCK_B, loc_block_r), lambda bi, ri: (0, ri)) for _ in range(12)],
      compiler_params=None if interpret else pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
      interpret=interpret,
  )(cam_128, lray_strip, geom_fields, light_128)
  dist, nx, ny, nz, geom_id, sh_out, rox, roy, roz, rdx, rdy, rdz = outs
  return dist, jnp.stack([nx, ny, nz], axis=-1), geom_id, (sh_out > 0.5) if do_in_vmem_shadow else None, jnp.stack([rox, roy, roz], axis=-1), jnp.stack([rdx, rdy, rdz], axis=-1)


def render_bokeh_pull_video(
    out_dir: str,
    n_frames: int = 90,
    W: int = 3840,
    H: int = 2160,
    clip_name: str = "thinlens_bokeh_pull_4k",
    dolly_parallax: bool = False,
) -> None:
  devs = jax.devices()[:4]
  spp = 32
  W_STRIP = W // 4
  xml = make_bokeh_hero_xml()
  m = mujoco.MjModel.from_xml_string(xml)
  cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "bokeh_hero")
  spec = RenderSceneSpec.from_mjmodel(m, cam_id=cam_id, height=H, width=W, use_shadows=True)

  d_host = mujoco.MjData(m)
  if m.nkey > 0:
    mujoco.mj_resetDataKeyframe(m, d_host, 0)
  mujoco.mj_forward(m, d_host)
  dx_1 = jax.tree_util.tree_map(lambda x: x[None, ...], mjx.put_data(m, d_host, impl="jax"))

  mesh = jax.sharding.Mesh(np.array(devs), ("sp",))
  P = jax.sharding.PartitionSpec
  sh_sp = jax.sharding.NamedSharding(mesh, P("sp", None, None))
  rep_sp = jax.sharding.NamedSharding(mesh, P())
  sm_kw = {"check_rep": False} if "check_rep" in inspect.signature(shard_map).parameters else {"check_vma": False}

  lens_r = 0.055 / (2.0 * 1.4)
  lray_strips = []
  uv_base = None
  for c in range(4):
    lr_c, uv_b = _build_thinlens_strip_arrays(spec.cam_fovy, H, W, c * W_STRIP, W_STRIP, spp)
    lray_strips.append(lr_c)
    uv_base = uv_b
  lray_sh = jax.device_put(jnp.asarray(np.stack(lray_strips, axis=0), jnp.float32), sh_sp)
  uv_rep = jax.device_put(jnp.asarray(uv_base.reshape(spp // 8, 8, 2), jnp.float32), rep_sp)
  dx_rep = jax.device_put(dx_1, rep_sp)

  def _sp_thinlens(d, lr_s, uv_c, cpos, cmat, fdist):
    gxp, gxm = d.geom_xpos[0], d.geom_xmat[0]
    lxp, lxd = _extract_light_batch(spec, d)
    n_c = uv_c.shape[0]

    def _body(i, acc):
      dist, normal, geom_id, fused_occ, rorig, rdir = _pallas_thinlens_raycast_dyn_focus(
          spec, cpos, cmat, gxp, gxm, lxp[0], lxd[0],
          lr_s[0], uv_c[i], lens_radius=lens_r, focus_dist_dyn=fdist, interpret=False,
      )
      rgb8 = _shade_thinlens_hdr(spec, gxp, gxm, lxp[0], lxd[0], rorig, rdir, dist, normal, geom_id, fused_occ, hdr_mode=True)
      return acc + jnp.mean(rgb8, axis=0)

    lin = jax.lax.fori_loop(0, n_c, _body, jnp.zeros((lr_s.shape[2], 3), jnp.float32)) / float(n_c)
    return _aces_tonemap_srgb(lin).reshape(H, W_STRIP, 3)

  _sm = shard_map(
      _sp_thinlens, mesh=mesh,
      in_specs=(P(), P("sp", None, None), P(), P(), P(), P()),
      out_specs=P(None, "sp", None), **sm_kw,
  )
  fn = jax.jit(lambda d, lr, uv, cp, cm, fd: _sm(d, lr, uv, cp, cm, fd))

  from experiments.showcase_videos.video_utils import look_at_mjx_cmat
  base_eye = np.array([1.46, -0.05, 0.36], dtype=np.float64)
  tgt_pt = np.array([0.45, 0.0, 0.19], dtype=np.float64)
  cpos0 = jnp.asarray(d_host.cam_xpos[cam_id], dtype=jnp.float32)
  cmat0 = jnp.asarray(d_host.cam_xmat[cam_id].reshape(3, 3), dtype=jnp.float32)

  t0 = time.perf_counter()
  warm = fn(dx_rep, lray_sh, uv_rep, jax.device_put(cpos0, rep_sp), jax.device_put(cmat0, rep_sp), jax.device_put(jnp.float32(0.80), rep_sp))
  jax.block_until_ready(warm)
  print(f"[w5:{clip_name}] Compiled 4-chip {W}x{H} f/1.4 in {time.perf_counter() - t0:.1f}s", flush=True)

  frames_u8 = []
  t1 = time.perf_counter()
  for i in range(n_frames):
    u = i / float(n_frames)
    s = smoothstep(2.0 * u if u < 0.5 else 2.0 * (1.0 - u))
    if dolly_parallax:
      phase = 2.0 * math.pi * u
      eye = base_eye + np.array([0.08 * math.cos(phase) - 0.08, 0.24 * math.sin(phase), 0.05 * math.sin(phase)], dtype=np.float64)
      cp_np = eye.astype(np.float32)
      cm_np = look_at_mjx_cmat(eye, tgt_pt)
      fdist = 0.78 + 1.30 * (0.5 - 0.5 * math.cos(phase))
    else:
      cp_np = np.asarray(cpos0, dtype=np.float32)
      cm_np = np.asarray(cmat0, dtype=np.float32)
      fdist = 0.78 * (1.0 - s) + 2.25 * s
    img = fn(dx_rep, lray_sh, uv_rep, jax.device_put(jnp.asarray(cp_np), rep_sp), jax.device_put(jnp.asarray(cm_np), rep_sp), jax.device_put(jnp.float32(fdist), rep_sp))
    jax.block_until_ready(img)
    frames_u8.append(np.clip(np.round(np.asarray(img) * 255.0), 0, 255).astype(np.uint8))

  print(f"[w5:{clip_name}] Rendered {n_frames} {W}x{H} frames in {time.perf_counter() - t1:.1f}s", flush=True)
  save_frames_tar(frames_u8, out_dir, clip_name)


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--out_dir", default="/dev/shm/zaychik-tpu-rtx/out_videos")
  args = ap.parse_args()
  os.makedirs(args.out_dir, exist_ok=True)
  render_bokeh_pull_video(args.out_dir, n_frames=90, W=3840, H=2160, clip_name="thinlens_bokeh_pull_4k", dolly_parallax=False)
  render_bokeh_pull_video(args.out_dir, n_frames=90, W=3840, H=2160, clip_name="thinlens_bokeh_dolly_4k", dolly_parallax=True)


if __name__ == "__main__":
  main()
