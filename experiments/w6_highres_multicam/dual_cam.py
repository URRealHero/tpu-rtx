"""Fused Dual-Camera (Overhead + Moving Wrist Cam) TPU Mosaic Raycaster (V3).

Packs two cameras' rays into (B, 2*R) within a single `pl.pallas_call` over
(BLOCK_B=8, BLOCK_R=512) tiles so the (B, G, 16) `geom_fields` and light state
are DMA-loaded ONCE into VMEM for both cameras, followed by a single fused
`_shade_batch_soa` pass.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Optional, Tuple

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
import numpy as np

from tpu_mjx_render.pallas_v2 import (
    BLOCK_B,
    BLOCK_R,
    _compute_batch_camera_rays_soa,
    _compute_geom_bound_sq,
    _extract_light_batch,
    _ray_one_geom_v2,
    _shade_batch_soa,
    build_local_camera_rays,
)
from tpu_mjx_render.primitives import MJ_MAXVAL
from tpu_mjx_render.renderer import GBuffer
from tpu_mjx_render.scenes import PANDA_PICK_CUBE_XML


def make_dual_cam_panda_xml() -> str:
  """Build 2-camera PandaPickCube MJCF with static 'front' (id 0) and hand-mounted 'wrist' (id 1)."""
  xml = PANDA_PICK_CUBE_XML
  side_line = '    <camera fovy="45" name="side" pos="0.60 -0.85 0.55" xyaxes="1 0 0 0 0.45 0.89"/>\n'
  assert side_line in xml, "Expected side camera line in PANDA_PICK_CUBE_XML"
  xml = xml.replace(side_line, "")

  anchor = '                      <site name="gripper" pos="0 0 0.10"/>\n'
  wrist_cam = (
      '                      <camera name="wrist" fovy="55" pos="-0.06 0.0 0.04" '
      'xyaxes="0 1 0 0.8575 0 -0.5145"/>\n'
  )
  assert anchor in xml, "Expected gripper site in PANDA_PICK_CUBE_XML"
  xml = xml.replace(anchor, anchor + wrist_cam)
  return xml


def _split_gbuffer(gb_packed: GBuffer, H: int) -> Tuple[GBuffer, GBuffer]:
  """Split a (B, 2*H, W, ...) packed GBuffer into (cam0_gb, cam1_gb) each (B, H, W, ...)."""
  gb0 = GBuffer(*(f[:, :H, ...] for f in gb_packed))
  gb1 = GBuffer(*(f[:, H:, ...] for f in gb_packed))
  return gb0, gb1


def pallas_raycast_dual_cam_v2(
    spec0: Any,
    spec1: Any,
    mjx_data_batch: Any,
    geom_rgba_batch: Optional[jax.Array] = None,
    interpret: bool = False,
) -> Tuple[GBuffer, GBuffer]:
  """Fused dual-camera TPU Mosaic raycaster sharing geom_fields VMEM DMA across both cameras."""
  H, W = int(spec0.height), int(spec0.width)
  assert int(spec1.height) == H and int(spec1.width) == W, "Dual-cam requires matching HxW"
  assert spec0.enabled_geom_ids == spec1.enabled_geom_ids
  R = H * W
  R_total = 2 * R

  cam0_pos = mjx_data_batch.cam_xpos[:, int(spec0.cam_id), :]
  cam0_mat = mjx_data_batch.cam_xmat[:, int(spec0.cam_id), :, :]
  cam1_pos = mjx_data_batch.cam_xpos[:, int(spec1.cam_id), :]
  cam1_mat = mjx_data_batch.cam_xmat[:, int(spec1.cam_id), :, :]

  geom_xpos = mjx_data_batch.geom_xpos
  geom_xmat = mjx_data_batch.geom_xmat
  light_xpos, light_xdir = _extract_light_batch(spec0, mjx_data_batch)
  geom_size = jnp.asarray(spec0.geom_size, dtype=jnp.float32)
  geom_rgba = jnp.asarray(spec0.geom_rgba, dtype=jnp.float32) if geom_rgba_batch is None else geom_rgba_batch

  B = cam0_pos.shape[0]
  gids = np.asarray(spec0.enabled_geom_ids, dtype=np.int32)
  G = len(gids)
  if G == 0:
    ro0, rd0, nz0 = _compute_batch_camera_rays_soa(spec0, cam0_pos, cam0_mat)
    ro1, rd1, nz1 = _compute_batch_camera_rays_soa(spec1, cam1_pos, cam1_mat)
    ro_d = jnp.concatenate([ro0, ro1], axis=1)
    rd_d = jnp.concatenate([rd0, rd1], axis=1)
    nz_d = jnp.concatenate([nz0, nz1], axis=0)
    spec_dual = dataclasses.replace(spec0, height=2 * H, width=W)
    gb_d = _shade_batch_soa(
        spec_dual, geom_xpos, geom_xmat, geom_size, geom_rgba, light_xpos, light_xdir,
        ro_d, rd_d, nz_d,
        jnp.full((B, R_total), -1.0, jnp.float32),
        jnp.zeros((B, R_total, 3), jnp.float32),
        jnp.full((B, R_total), -1, jnp.int32),
    )
    return _split_gbuffer(gb_d, H)

  B_pad = ((B + BLOCK_B - 1) // BLOCK_B) * BLOCK_B
  R_total_pad = ((R_total + BLOCK_R - 1) // BLOCK_R) * BLOCK_R
  G_pad = ((G + 127) // 128) * 128
  tile_aligned = (R % BLOCK_R == 0)
  n_r_blk_cam0 = R // BLOCK_R

  # Pack both camera poses into (B_pad, 128): [0:12]=cam0, [12:24]=cam1
  cam_pack = jnp.concatenate(
      [cam0_pos, cam0_mat.reshape(B, 9), cam1_pos, cam1_mat.reshape(B, 9)],
      axis=-1,
  )  # (B, 24)
  cam_128 = jnp.pad(cam_pack, ((0, B_pad - B), (0, 128 - 24)))

  ld0, lo0, nz0_np = build_local_camera_rays(spec0)
  ld1, lo1, nz1_np = build_local_camera_rays(spec1)
  md0 = ((float(spec0.zfar) - float(spec0.znear)) / nz0_np) if spec0.zfar > 0.0 else np.full((R,), MJ_MAXVAL, dtype=np.float32)
  md1 = ((float(spec1.zfar) - float(spec1.znear)) / nz1_np) if spec1.zfar > 0.0 else np.full((R,), MJ_MAXVAL, dtype=np.float32)

  lray_rows = 8 if tile_aligned else 16
  lray_raw = np.zeros((lray_rows, R_total_pad), dtype=np.float32)
  lray_raw[0:3, :R] = ld0.T
  lray_raw[3:6, :R] = lo0.T
  lray_raw[6, :R] = nz0_np
  lray_raw[7, :R] = md0
  lray_raw[0:3, R:R_total] = ld1.T
  lray_raw[3:6, R:R_total] = lo1.T
  lray_raw[6, R:R_total] = nz1_np
  lray_raw[7, R:R_total] = md1
  if not tile_aligned:
    lray_raw[8, :R] = 0.0
    lray_raw[8, R:R_total] = 1.0
  if R_total_pad > R_total:
    lray_raw[2, R_total:] = -1.0
    lray_raw[6, R_total:] = 1.0
    lray_raw[7, R_total:] = 1.0
  lray_dev = jnp.asarray(lray_raw, dtype=jnp.float32)

  # Pack enabled geoms ONCE for both cameras: (n_b_blk * 16 * BLOCK_B, G_pad)
  gpos = geom_xpos[:, gids, :]
  gmat = geom_xmat[:, gids, :, :].reshape(B, G, 9)
  gsz_all = jnp.broadcast_to(geom_size[gids][None, :, :], (B, G, 3))
  rb_sq = jnp.broadcast_to(jnp.asarray(_compute_geom_bound_sq(spec0, gids))[None, :, None], (B, G, 1))
  g_16 = jnp.concatenate([gpos, gmat, gsz_all, rb_sq], axis=-1)
  g_16 = jnp.pad(g_16, ((0, B_pad - B), (0, G_pad - G), (0, 0)))

  n_b_blk = B_pad // BLOCK_B
  n_r_blk = R_total_pad // BLOCK_R
  geom_fields = (
      g_16.reshape(n_b_blk, BLOCK_B, G_pad, 16)
      .transpose(0, 3, 1, 2)
      .reshape(n_b_blk * 16 * BLOCK_B, G_pad)
  )

  shadow_l_idx = -1
  for li in range(int(spec0.nlight)):
    if spec0.use_shadows and bool(spec0.light_active[li]) and bool(spec0.light_castshadow[li]):
      shadow_l_idx = li
      break

  light_128 = jnp.zeros((B_pad, 128), dtype=jnp.float32)
  if shadow_l_idx >= 0:
    is_dir = 1.0 if int(spec0.light_type[shadow_l_idx]) == 1 else 0.0
    lp = light_xpos[:, shadow_l_idx, :]
    ld_vec = light_xdir[:, shadow_l_idx, :]
    ld_vec = ld_vec / jnp.maximum(jnp.linalg.norm(ld_vec, axis=-1, keepdims=True), 1e-12)
    l_row = jnp.concatenate([jnp.full((B, 1), is_dir, jnp.float32), lp, ld_vec], axis=-1)
    light_128 = jnp.pad(l_row, ((0, B_pad - B), (0, 128 - 7)))

  g_types_static = tuple(int(spec0.geom_type[g]) for g in gids)
  gids_static = tuple(int(g) for g in gids)
  plane_fx = tuple(float(spec0.geom_size[g, 0]) > 0.0 for g in gids)
  plane_fy = tuple(float(spec0.geom_size[g, 1]) > 0.0 for g in gids)
  znear_f = float(spec0.znear)
  do_in_vmem_shadow = shadow_l_idx >= 0

  def _v2_dual_kernel(
      cam_ref, lray_ref, gf_ref, light_ref,
      out_d_ref, out_nx_ref, out_ny_ref, out_nz_ref, out_gid_ref, out_sh_ref,
  ):
    cam = cam_ref[...]       # (8, 128)
    lr = lray_ref[...]       # (8 or 16, 512)
    gf = gf_ref[...]         # (128, G_pad)
    linfo = light_ref[...]   # (8, 128)

    ldx, ldy, ldz = lr[0:1, :], lr[1:2, :], lr[2:3, :]
    lox, loy, loz = lr[3:4, :], lr[4:5, :], lr[5:6, :]
    inv_cos = znear_f / lr[6:7, :]
    max_d = jnp.broadcast_to(lr[7:8, :], (BLOCK_B, BLOCK_R))

    if tile_aligned:
      # Block-level selection on (8, 1) camera pose scalars before broadcasting to (8, 512)
      is_c1 = pl.program_id(1) >= n_r_blk_cam0
      cpx = jnp.where(is_c1, cam[:, 12:13], cam[:, 0:1])
      cpy = jnp.where(is_c1, cam[:, 13:14], cam[:, 1:2])
      cpz = jnp.where(is_c1, cam[:, 14:15], cam[:, 2:3])
      c00 = jnp.where(is_c1, cam[:, 15:16], cam[:, 3:4])
      c01 = jnp.where(is_c1, cam[:, 16:17], cam[:, 4:5])
      c02 = jnp.where(is_c1, cam[:, 17:18], cam[:, 5:6])
      c10 = jnp.where(is_c1, cam[:, 18:19], cam[:, 6:7])
      c11 = jnp.where(is_c1, cam[:, 19:20], cam[:, 7:8])
      c12 = jnp.where(is_c1, cam[:, 20:21], cam[:, 8:9])
      c20 = jnp.where(is_c1, cam[:, 21:22], cam[:, 9:10])
      c21 = jnp.where(is_c1, cam[:, 22:23], cam[:, 10:11])
      c22 = jnp.where(is_c1, cam[:, 23:24], cam[:, 11:12])
    else:
      is_c1 = jnp.broadcast_to(lr[8:9, :] > 0.5, (BLOCK_B, BLOCK_R))

      def _sel(c0_col, c1_col):
        return jnp.where(
            is_c1,
            jnp.broadcast_to(cam[:, c1_col : c1_col + 1], (BLOCK_B, BLOCK_R)),
            jnp.broadcast_to(cam[:, c0_col : c0_col + 1], (BLOCK_B, BLOCK_R)),
        )

      cpx, cpy, cpz = _sel(0, 12), _sel(1, 13), _sel(2, 14)
      c00, c01, c02 = _sel(3, 15), _sel(4, 16), _sel(5, 17)
      c10, c11, c12 = _sel(6, 18), _sel(7, 19), _sel(8, 20)
      c20, c21, c22 = _sel(9, 21), _sel(10, 22), _sel(11, 23)

    rdx_raw = c00 * ldx + c01 * ldy + c02 * ldz
    rdy_raw = c10 * ldx + c11 * ldy + c12 * ldz
    rdz_raw = c20 * ldx + c21 * ldy + c22 * ldz
    inv_rn = 1.0 / jnp.maximum(jnp.sqrt(rdx_raw * rdx_raw + rdy_raw * rdy_raw + rdz_raw * rdz_raw), 1e-12)
    rdx, rdy, rdz = rdx_raw * inv_rn, rdy_raw * inv_rn, rdz_raw * inv_rn

    rox = cpx + (c00 * lox + c01 * loy + c02 * loz) + inv_cos * rdx
    roy = cpy + (c10 * lox + c11 * loy + c12 * loz) + inv_cos * rdy
    roz = cpz + (c20 * lox + c21 * loy + c22 * loz) + inv_cos * rdz

    best_d = max_d
    best_nx = jnp.zeros((BLOCK_B, BLOCK_R), dtype=jnp.float32)
    best_ny = jnp.zeros((BLOCK_B, BLOCK_R), dtype=jnp.float32)
    best_nz = jnp.zeros((BLOCK_B, BLOCK_R), dtype=jnp.float32)
    best_gid = jnp.full((BLOCK_B, BLOCK_R), -1, dtype=jnp.int32)

    for k, (gt, gid_val, pfx, pfy) in enumerate(zip(g_types_static, gids_static, plane_fx, plane_fy)):
      px = gf[0 * 8 : 1 * 8, k : k + 1]
      py = gf[1 * 8 : 2 * 8, k : k + 1]
      pz = gf[2 * 8 : 3 * 8, k : k + 1]
      m00, m01, m02 = gf[3 * 8 : 4 * 8, k : k + 1], gf[4 * 8 : 5 * 8, k : k + 1], gf[5 * 8 : 6 * 8, k : k + 1]
      m10, m11, m12 = gf[6 * 8 : 7 * 8, k : k + 1], gf[7 * 8 : 8 * 8, k : k + 1], gf[8 * 8 : 9 * 8, k : k + 1]
      m20, m21, m22 = gf[9 * 8 : 10 * 8, k : k + 1], gf[10 * 8 : 11 * 8, k : k + 1], gf[11 * 8 : 12 * 8, k : k + 1]
      sx, sy, sz = gf[12 * 8 : 13 * 8, k : k + 1], gf[13 * 8 : 14 * 8, k : k + 1], gf[14 * 8 : 15 * 8, k : k + 1]
      rb_sq = gf[15 * 8 : 16 * 8, k : k + 1]

      cd, cnx, cny, cnz = _ray_one_geom_v2(
          gt,
          px, py, pz,
          m00, m01, m02,
          m10, m11, m12,
          m20, m21, m22,
          sx, sy, sz,
          rb_sq,
          rox, roy, roz, rdx, rdy, rdz,
          plane_finite_x=pfx,
          plane_finite_y=pfy,
          compute_normal=True,
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

    sh_occ = jnp.zeros((BLOCK_B, BLOCK_R), dtype=jnp.float32)
    if do_in_vmem_shadow:
      d_safe = jnp.where(hit, best_d, 0.0)
      hx = rox + d_safe * rdx + out_nx_v * 1.0e-4
      hy = roy + d_safe * rdy + out_ny_v * 1.0e-4
      hz = roz + d_safe * rdz + out_nz_v * 1.0e-4

      is_dir = linfo[:, 0:1] > 0.5
      dlx = linfo[:, 1:2] - hx
      dly = linfo[:, 2:3] - hy
      dlz = linfo[:, 3:4] - hz
      d_pt = jnp.sqrt(jnp.maximum(dlx * dlx + dly * dly + dlz * dlz, 1e-12))
      lx_v = jnp.where(is_dir, -linfo[:, 4:5], dlx / d_pt)
      ly_v = jnp.where(is_dir, -linfo[:, 5:6], dly / d_pt)
      lz_v = jnp.where(is_dir, -linfo[:, 6:7], dlz / d_pt)
      ndotl = out_nx_v * lx_v + out_ny_v * ly_v + out_nz_v * lz_v
      dist_lim = jnp.where(is_dir, 1.0e8, jnp.maximum(d_pt - 1.0e-3, 0.0))
      sh_max = jnp.where(hit & (ndotl > 0.0), dist_lim, 0.0)

      occ_bool = jnp.zeros((BLOCK_B, BLOCK_R), dtype=jnp.bool_)
      for k, (gt, pfx, pfy) in enumerate(zip(g_types_static, plane_fx, plane_fy)):
        px = gf[0 * 8 : 1 * 8, k : k + 1]
        py = gf[1 * 8 : 2 * 8, k : k + 1]
        pz = gf[2 * 8 : 3 * 8, k : k + 1]
        m00, m01, m02 = gf[3 * 8 : 4 * 8, k : k + 1], gf[4 * 8 : 5 * 8, k : k + 1], gf[5 * 8 : 6 * 8, k : k + 1]
        m10, m11, m12 = gf[6 * 8 : 7 * 8, k : k + 1], gf[7 * 8 : 8 * 8, k : k + 1], gf[8 * 8 : 9 * 8, k : k + 1]
        m20, m21, m22 = gf[9 * 8 : 10 * 8, k : k + 1], gf[10 * 8 : 11 * 8, k : k + 1], gf[11 * 8 : 12 * 8, k : k + 1]
        sx, sy, sz = gf[12 * 8 : 13 * 8, k : k + 1], gf[13 * 8 : 14 * 8, k : k + 1], gf[14 * 8 : 15 * 8, k : k + 1]
        rb_sq = gf[15 * 8 : 16 * 8, k : k + 1]
        scd, _, _, _ = _ray_one_geom_v2(
            gt,
            px, py, pz,
            m00, m01, m02,
            m10, m11, m12,
            m20, m21, m22,
            sx, sy, sz,
            rb_sq,
            hx, hy, hz, lx_v, ly_v, lz_v,
            plane_finite_x=pfx,
            plane_finite_y=pfy,
            compute_normal=False,
        )
        occ_bool = occ_bool | (scd < sh_max)
      sh_occ = occ_bool.astype(jnp.float32)

    out_d_ref[...] = best_d_out
    out_nx_ref[...] = out_nx_v
    out_ny_ref[...] = out_ny_v
    out_nz_ref[...] = out_nz_v
    out_gid_ref[...] = best_gid
    out_sh_ref[...] = sh_occ

  d_out, nx_out, ny_out, nz_out, gid_out, sh_out = pl.pallas_call(
      _v2_dual_kernel,
      out_shape=[
          jax.ShapeDtypeStruct((B_pad, R_total_pad), jnp.float32),
          jax.ShapeDtypeStruct((B_pad, R_total_pad), jnp.float32),
          jax.ShapeDtypeStruct((B_pad, R_total_pad), jnp.float32),
          jax.ShapeDtypeStruct((B_pad, R_total_pad), jnp.float32),
          jax.ShapeDtypeStruct((B_pad, R_total_pad), jnp.int32),
          jax.ShapeDtypeStruct((B_pad, R_total_pad), jnp.float32),
      ],
      grid=(n_b_blk, n_r_blk),
      in_specs=[
          pl.BlockSpec((BLOCK_B, 128), lambda bi, ri: (bi, 0)),
          pl.BlockSpec((lray_rows, BLOCK_R), lambda bi, ri: (0, ri)),
          pl.BlockSpec((16 * BLOCK_B, G_pad), lambda bi, ri: (bi, 0)),
          pl.BlockSpec((BLOCK_B, 128), lambda bi, ri: (bi, 0)),
      ],
      out_specs=[
          pl.BlockSpec((BLOCK_B, BLOCK_R), lambda bi, ri: (bi, ri)),
          pl.BlockSpec((BLOCK_B, BLOCK_R), lambda bi, ri: (bi, ri)),
          pl.BlockSpec((BLOCK_B, BLOCK_R), lambda bi, ri: (bi, ri)),
          pl.BlockSpec((BLOCK_B, BLOCK_R), lambda bi, ri: (bi, ri)),
          pl.BlockSpec((BLOCK_B, BLOCK_R), lambda bi, ri: (bi, ri)),
          pl.BlockSpec((BLOCK_B, BLOCK_R), lambda bi, ri: (bi, ri)),
      ],
      compiler_params=None if interpret else pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
      interpret=interpret,
  )(cam_128, lray_dev, geom_fields, light_128)

  dist = d_out[:B, :R_total]
  normal = jnp.stack([nx_out[:B, :R_total], ny_out[:B, :R_total], nz_out[:B, :R_total]], axis=-1)
  geom_id = gid_out[:B, :R_total]
  fused_occ = (sh_out[:B, :R_total] > 0.5) if do_in_vmem_shadow else None

  ro0, rd0, nz0 = _compute_batch_camera_rays_soa(spec0, cam0_pos, cam0_mat)
  ro1, rd1, nz1 = _compute_batch_camera_rays_soa(spec1, cam1_pos, cam1_mat)
  ro_dual = jnp.concatenate([ro0, ro1], axis=1)
  rd_dual = jnp.concatenate([rd0, rd1], axis=1)
  nz_dual = jnp.concatenate([nz0, nz1], axis=0)

  spec_dual = dataclasses.replace(spec0, height=2 * H, width=W)
  gb_dual = _shade_batch_soa(
      spec_dual,
      geom_xpos,
      geom_xmat,
      geom_size,
      geom_rgba,
      light_xpos,
      light_xdir,
      ro_dual,
      rd_dual,
      nz_dual,
      dist,
      normal,
      geom_id,
      fused_shadow0_occluded=fused_occ,
  )
  return _split_gbuffer(gb_dual, H)
