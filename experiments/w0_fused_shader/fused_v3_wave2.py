"""Wave 2 V3 Fused Shaders:
1. `pallas_raycast_fused_v3_opt`: Material-deduplicated + configurable tile geometry (BLOCK_B, BLOCK_R)
   and BF16/FP32 output paths for pushing 100k-350k+ FPS on 64x64 and 128x128.
2. `pallas_raycast_pbr_ao_v3`: In-VMEM Fused GGX Microfacet PBR + Multi-Sample Area Soft-Shadow
   + Multi-Sample Stratified Hemisphere Ambient Occlusion (AO) inside a single pallas_call.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
import numpy as np

from fused_v3 import FusedFrame
from tpu_mjx_render.pallas_v2 import (
    _compute_geom_bound_sq,
    _extract_light_batch,
    _ray_one_geom_v2,
    build_local_camera_rays,
)
from tpu_mjx_render.primitives import MJ_MAXVAL


def _branchless_onb(nx, ny, nz):
  """Duff et al. (2017) branchless orthonormal basis from unit normal (nx, ny, nz)."""
  sign_z = jnp.where(nz >= 0.0, 1.0, -1.0)
  a = -1.0 / (sign_z + nz)
  b = nx * ny * a
  tx = 1.0 + sign_z * nx * nx * a
  ty = sign_z * b
  tz = -sign_z * nx
  bx = b
  by = sign_z + ny * ny * a
  bz = -ny
  return (tx, ty, tz), (bx, by, bz)


def _r2_pixel_rotation(H: int, W: int, R_pad: int) -> tuple[np.ndarray, np.ndarray]:
  """2D R2 quasirandom per-pixel tangent rotation (cos_rot, sin_rot) for stratified AO/soft-shadow."""
  px = np.arange(W, dtype=np.float64)
  py = np.arange(H, dtype=np.float64)
  gx, gy = np.meshgrid(px, py, indexing="xy")
  u = (gx * 0.7548776662466927 + gy * 0.5698402909980532) % 1.0
  theta = (2.0 * math.pi * u).reshape(-1)
  cos_r = np.ones((R_pad,), dtype=np.float32)
  sin_r = np.zeros((R_pad,), dtype=np.float32)
  R = H * W
  cos_r[:R] = np.cos(theta).astype(np.float32)
  sin_r[:R] = np.sin(theta).astype(np.float32)
  return cos_r, sin_r


def pallas_raycast_fused_v3_opt(
    spec: Any,
    mjx_data_batch: Any,
    geom_rgba_batch: Optional[jax.Array] = None,
    interpret: bool = True,
    block_b: int = 8,
    block_r: int = 512,
    out_dtype: Any = jnp.float32,
    dedup_materials: bool = True,
) -> FusedFrame:
  """Optimized V3 fused shader with material-group deduplication and flexible tile geometry."""
  cam_pos = mjx_data_batch.cam_xpos[:, int(spec.cam_id), :]
  cam_mat = mjx_data_batch.cam_xmat[:, int(spec.cam_id), :, :]
  geom_xpos = mjx_data_batch.geom_xpos
  geom_xmat = mjx_data_batch.geom_xmat
  light_xpos, light_xdir = _extract_light_batch(spec, mjx_data_batch)

  B = cam_pos.shape[0]
  H, W = int(spec.height), int(spec.width)
  R = H * W

  gids = np.asarray(spec.enabled_geom_ids, dtype=np.int32)
  G = len(gids)
  if G == 0:
    ld_np, _, _ = build_local_camera_rays(spec)
    ld = jnp.asarray(ld_np, dtype=jnp.float32)
    ray_dir = jnp.einsum("bij,rj->bri", cam_mat, ld, precision=jax.lax.Precision.HIGHEST)
    ray_dir = ray_dir / jnp.maximum(jnp.linalg.norm(ray_dir, axis=-1, keepdims=True), 1e-12)
    sky_t = jnp.clip(0.5 * (ray_dir[..., 2] + 1.0), 0.0, 1.0)[..., None]
    sky_rgb = (
        sky_t * jnp.asarray(spec.sky_rgb1, dtype=jnp.float32)[None, None, :]
        + (1.0 - sky_t) * jnp.asarray(spec.sky_rgb2, dtype=jnp.float32)[None, None, :]
    )
    return FusedFrame(
        rgb=sky_rgb.astype(out_dtype).reshape(B, H, W, 3),
        depth=jnp.full((B, H, W), -1.0, dtype=out_dtype),
    )

  B_pad = ((B + block_b - 1) // block_b) * block_b
  R_pad = ((R + block_r - 1) // block_r) * block_r
  G_pad = ((G + 127) // 128) * 128
  n_b_blk = B_pad // block_b
  n_r_blk = R_pad // block_r

  cam_pack = jnp.concatenate([cam_pos, cam_mat.reshape(B, 9)], axis=-1)
  cam_128 = jnp.pad(cam_pack, ((0, B_pad - B), (0, 128 - 12)))

  ld_np, lo_np, neg_z_np = build_local_camera_rays(spec)
  md_np = ((float(spec.zfar) - float(spec.znear)) / neg_z_np) if spec.zfar > 0.0 else np.full((R,), MJ_MAXVAL, dtype=np.float32)
  lray_raw = np.zeros((8, R_pad), dtype=np.float32)
  lray_raw[0:3, :R] = ld_np.T
  lray_raw[3:6, :R] = lo_np.T
  lray_raw[6, :R] = neg_z_np
  lray_raw[7, :R] = md_np
  lray_raw[2, R:] = -1.0
  lray_raw[6, R:] = 1.0
  lray_raw[7, R:] = 1.0
  lray_dev = jnp.asarray(lray_raw, dtype=jnp.float32)

  geom_size = jnp.asarray(spec.geom_size, dtype=jnp.float32)
  gpos = geom_xpos[:, gids, :]
  gmat = geom_xmat[:, gids, :, :].reshape(B, G, 9)
  gsz_all = jnp.broadcast_to(geom_size[gids][None, :, :], (B, G, 3))
  rb_sq = jnp.broadcast_to(jnp.asarray(_compute_geom_bound_sq(spec, gids))[None, :, None], (B, G, 1))
  g_16 = jnp.concatenate([gpos, gmat, gsz_all, rb_sq], axis=-1)
  g_16 = jnp.pad(g_16, ((0, B_pad - B), (0, G_pad - G), (0, 0)))
  geom_fields = (
      g_16.reshape(n_b_blk, block_b, G_pad, 16)
      .transpose(0, 3, 1, 2)
      .reshape(n_b_blk * 16 * block_b, G_pad)
  )

  has_dynamic_rgba = geom_rgba_batch is not None
  if has_dynamic_rgba:
    g_rgba_4 = jnp.pad(geom_rgba_batch[:, gids, :], ((0, B_pad - B), (0, G_pad - G), (0, 0)))
    rgba_fields = (
        g_rgba_4.reshape(n_b_blk, block_b, G_pad, 4)
        .transpose(0, 3, 1, 2)
        .reshape(n_b_blk * 4 * block_b, G_pad)
    )
  else:
    rgba_fields = None

  active_lights = []
  light_chunks = []
  for li in range(int(spec.nlight)):
    if not bool(spec.light_active[li]):
      continue
    lp = light_xpos[:, li, :]
    ld_v = light_xdir[:, li, :]
    ld_v = ld_v / jnp.maximum(jnp.linalg.norm(ld_v, axis=-1, keepdims=True), 1e-12)
    light_chunks.append(lp)
    light_chunks.append(ld_v)
    active_lights.append((
        li,
        int(spec.light_type[li]),
        tuple(float(x) for x in spec.light_ambient[li]),
        tuple(float(x) for x in spec.light_diffuse[li]),
        tuple(float(x) for x in spec.light_specular[li]),
        tuple(float(x) for x in spec.light_attenuation[li]),
        float(np.cos(spec.light_cutoff_rad[li])),
        float(spec.light_exponent[li]),
        bool(spec.use_shadows and bool(spec.light_castshadow[li])),
    ))

  if light_chunks:
    l_cat = jnp.concatenate(light_chunks, axis=-1)
    light_128 = jnp.pad(l_cat, ((0, B_pad - B), (0, 128 - l_cat.shape[1])))
  else:
    light_128 = jnp.zeros((B_pad, 128), dtype=jnp.float32)

  g_types_static = tuple(int(spec.geom_type[g]) for g in gids)
  gids_static = tuple(int(g) for g in gids)
  plane_fx = tuple(float(spec.geom_size[g, 0]) > 0.0 for g in gids)
  plane_fy = tuple(float(spec.geom_size[g, 1]) > 0.0 for g in gids)
  g_rgba_static = tuple(tuple(float(c) for c in spec.geom_rgba[g, :3]) for g in gids)
  g_spec_static = tuple(float(spec.geom_spec[g]) for g in gids)
  g_shin_static = tuple(float(spec.geom_shin[g]) * 128.0 for g in gids)
  g_emit_static = tuple(float(spec.geom_emit[g]) for g in gids)
  chk_indices = tuple((k, int(g)) for k, g in enumerate(gids) if bool(spec.geom_is_checker[g]))

  spec_uniform = len(set(g_spec_static)) == 1
  shin_uniform = len(set(g_shin_static)) == 1
  emit_uniform = len(set(g_emit_static)) == 1

  # Material-group deduplication when rgba is static (reduces ~34x6 selects to ~6x3)
  mat_groups: dict[tuple[float, ...], list[int]] = {}
  if dedup_materials and not has_dynamic_rgba:
    for k, gid_val in enumerate(gids_static):
      if bool(spec.geom_is_checker[gid_val]):
        continue  # overwritten by checker below anyway
      mkey = (
          *g_rgba_static[k],
          g_spec_static[k],
          g_shin_static[k],
          g_emit_static[k],
      )
      mat_groups.setdefault(mkey, []).append(gid_val)

  znear_f = float(spec.znear)
  rep_x, rep_y = float(spec.checker_repeat[0]), float(spec.checker_repeat[1])
  chk1_r, chk1_g, chk1_b = [float(x) for x in spec.checker_rgb1]
  chk2_r, chk2_g, chk2_b = [float(x) for x in spec.checker_rgb2]
  sky1_r, sky1_g, sky1_b = [float(x) for x in spec.sky_rgb1]
  sky2_r, sky2_g, sky2_b = [float(x) for x in spec.sky_rgb2]
  shadow_frac = float(spec.shadow_light_fraction)
  enable_specular = bool(spec.enable_specular)
  headlight_active = bool(spec.headlight_active)
  hl_amb = tuple(float(x) for x in spec.headlight_ambient)
  hl_diff = tuple(float(x) for x in spec.headlight_diffuse)
  hl_spec = tuple(float(x) for x in spec.headlight_specular)

  def _v3_opt_kernel(*refs):
    if has_dynamic_rgba:
      cam_ref, lray_ref, gf_ref, light_ref, rgba_ref, out_r_ref, out_g_ref, out_b_ref, out_d_ref = refs
      rgba_blk = rgba_ref[...]
    else:
      cam_ref, lray_ref, gf_ref, light_ref, out_r_ref, out_g_ref, out_b_ref, out_d_ref = refs
      rgba_blk = None

    cam = cam_ref[...]
    lr = lray_ref[...]
    gf = gf_ref[...]
    linfo = light_ref[...]

    ldx, ldy, ldz = lr[0:1, :], lr[1:2, :], lr[2:3, :]
    lox, loy, loz = lr[3:4, :], lr[4:5, :], lr[5:6, :]
    inv_cos = znear_f / lr[6:7, :]
    max_d = jnp.broadcast_to(lr[7:8, :], (block_b, block_r))

    cpx, cpy, cpz = cam[:, 0:1], cam[:, 1:2], cam[:, 2:3]
    c00, c01, c02 = cam[:, 3:4], cam[:, 4:5], cam[:, 5:6]
    c10, c11, c12 = cam[:, 6:7], cam[:, 7:8], cam[:, 8:9]
    c20, c21, c22 = cam[:, 9:10], cam[:, 10:11], cam[:, 11:12]

    rdx_raw = c00 * ldx + c01 * ldy + c02 * ldz
    rdy_raw = c10 * ldx + c11 * ldy + c12 * ldz
    rdz_raw = c20 * ldx + c21 * ldy + c22 * ldz
    inv_rn = 1.0 / jnp.maximum(jnp.sqrt(rdx_raw * rdx_raw + rdy_raw * rdy_raw + rdz_raw * rdz_raw), 1e-12)
    rdx, rdy, rdz = rdx_raw * inv_rn, rdy_raw * inv_rn, rdz_raw * inv_rn

    rox = cpx + (c00 * lox + c01 * loy + c02 * loz) + inv_cos * rdx
    roy = cpy + (c10 * lox + c11 * loy + c12 * loz) + inv_cos * rdy
    roz = cpz + (c20 * lox + c21 * loy + c22 * loz) + inv_cos * rdz

    best_d = max_d
    best_nx = jnp.zeros((block_b, block_r), dtype=jnp.float32)
    best_ny = jnp.zeros((block_b, block_r), dtype=jnp.float32)
    best_nz = jnp.zeros((block_b, block_r), dtype=jnp.float32)
    best_gid = jnp.full((block_b, block_r), -1, dtype=jnp.int32)

    for k, (gt, gid_val, pfx, pfy) in enumerate(zip(g_types_static, gids_static, plane_fx, plane_fy)):
      px = gf[0 * block_b : 1 * block_b, k : k + 1]
      py = gf[1 * block_b : 2 * block_b, k : k + 1]
      pz = gf[2 * block_b : 3 * block_b, k : k + 1]
      m00, m01, m02 = gf[3 * block_b : 4 * block_b, k : k + 1], gf[4 * block_b : 5 * block_b, k : k + 1], gf[5 * block_b : 6 * block_b, k : k + 1]
      m10, m11, m12 = gf[6 * block_b : 7 * block_b, k : k + 1], gf[7 * block_b : 8 * block_b, k : k + 1], gf[8 * block_b : 9 * block_b, k : k + 1]
      m20, m21, m22 = gf[9 * block_b : 10 * block_b, k : k + 1], gf[10 * block_b : 11 * block_b, k : k + 1], gf[11 * block_b : 12 * block_b, k : k + 1]
      sx, sy, sz = gf[12 * block_b : 13 * block_b, k : k + 1], gf[13 * block_b : 14 * block_b, k : k + 1], gf[14 * block_b : 15 * block_b, k : k + 1]
      rb_sq = gf[15 * block_b : 16 * block_b, k : k + 1]

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
    d_safe = jnp.where(hit, best_d, 0.0)
    nx_v = jnp.where(hit, best_nx, 0.0)
    ny_v = jnp.where(hit, best_ny, 0.0)
    nz_v = jnp.where(hit, best_nz, 0.0)

    hx = rox + d_safe * rdx
    hy = roy + d_safe * rdy
    hz = roz + d_safe * rdz

    alb_r = jnp.zeros((block_b, block_r), dtype=jnp.float32)
    alb_g = jnp.zeros((block_b, block_r), dtype=jnp.float32)
    alb_b = jnp.zeros((block_b, block_r), dtype=jnp.float32)
    spec_coef = jnp.full((block_b, block_r), g_spec_static[0], dtype=jnp.float32) if spec_uniform else jnp.zeros((block_b, block_r), dtype=jnp.float32)
    shin_exp = jnp.full((block_b, block_r), g_shin_static[0], dtype=jnp.float32) if shin_uniform else jnp.zeros((block_b, block_r), dtype=jnp.float32)
    emit_coef = jnp.full((block_b, block_r), g_emit_static[0], dtype=jnp.float32) if emit_uniform else jnp.zeros((block_b, block_r), dtype=jnp.float32)

    if dedup_materials and rgba_blk is None:
      for mkey, gid_list in mat_groups.items():
        rk, gk, bk, sk, shk, ek = mkey
        is_m = best_gid == gid_list[0]
        for g_other in gid_list[1:]:
          is_m = is_m | (best_gid == g_other)
        alb_r = jnp.where(is_m, rk, alb_r)
        alb_g = jnp.where(is_m, gk, alb_g)
        alb_b = jnp.where(is_m, bk, alb_b)
        if not spec_uniform:
          spec_coef = jnp.where(is_m, sk, spec_coef)
        if not shin_uniform:
          shin_exp = jnp.where(is_m, shk, shin_exp)
        if not emit_uniform:
          emit_coef = jnp.where(is_m, ek, emit_coef)
      # Checker geom spec/shin/emit if non-uniform
      if not (spec_uniform and shin_uniform and emit_uniform):
        for k, cg in chk_indices:
          is_cg = best_gid == cg
          if not spec_uniform:
            spec_coef = jnp.where(is_cg, g_spec_static[k], spec_coef)
          if not shin_uniform:
            shin_exp = jnp.where(is_cg, g_shin_static[k], shin_exp)
          if not emit_uniform:
            emit_coef = jnp.where(is_cg, g_emit_static[k], emit_coef)
    else:
      for k, gid_val in enumerate(gids_static):
        is_k = best_gid == gid_val
        if rgba_blk is not None:
          rk = rgba_blk[0 * block_b : 1 * block_b, k : k + 1]
          gk = rgba_blk[1 * block_b : 2 * block_b, k : k + 1]
          bk = rgba_blk[2 * block_b : 3 * block_b, k : k + 1]
        else:
          rk, gk, bk = g_rgba_static[k]
        alb_r = jnp.where(is_k, rk, alb_r)
        alb_g = jnp.where(is_k, gk, alb_g)
        alb_b = jnp.where(is_k, bk, alb_b)
        if not spec_uniform:
          spec_coef = jnp.where(is_k, g_spec_static[k], spec_coef)
        if not shin_uniform:
          shin_exp = jnp.where(is_k, g_shin_static[k], shin_exp)
        if not emit_uniform:
          emit_coef = jnp.where(is_k, g_emit_static[k], emit_coef)

    for k, cg in chk_indices:
      px = gf[0 * block_b : 1 * block_b, k : k + 1]
      py = gf[1 * block_b : 2 * block_b, k : k + 1]
      pz = gf[2 * block_b : 3 * block_b, k : k + 1]
      m00, m01 = gf[3 * block_b : 4 * block_b, k : k + 1], gf[4 * block_b : 5 * block_b, k : k + 1]
      m10, m11 = gf[6 * block_b : 7 * block_b, k : k + 1], gf[7 * block_b : 8 * block_b, k : k + 1]
      m20, m21 = gf[9 * block_b : 10 * block_b, k : k + 1], gf[10 * block_b : 11 * block_b, k : k + 1]
      dcx, dcy, dcz = hx - px, hy - py, hz - pz
      lx_cg = m00 * dcx + m10 * dcy + m20 * dcz
      ly_cg = m01 * dcx + m11 * dcy + m21 * dcz
      cx = jnp.floor(lx_cg * rep_x).astype(jnp.int32)
      cy = jnp.floor(ly_cg * rep_y).astype(jnp.int32)
      even = ((cx + cy) & 1) == 0
      is_cg = best_gid == cg
      alb_r = jnp.where(is_cg, jnp.where(even, chk1_r, chk2_r), alb_r)
      alb_g = jnp.where(is_cg, jnp.where(even, chk1_g, chk2_g), alb_g)
      alb_b = jnp.where(is_cg, jnp.where(even, chk1_b, chk2_b), alb_b)

    diff_r = jnp.zeros((block_b, block_r), dtype=jnp.float32)
    diff_g = jnp.zeros((block_b, block_r), dtype=jnp.float32)
    diff_b = jnp.zeros((block_b, block_r), dtype=jnp.float32)
    spec_r = jnp.zeros((block_b, block_r), dtype=jnp.float32)
    spec_g = jnp.zeros((block_b, block_r), dtype=jnp.float32)
    spec_b = jnp.zeros((block_b, block_r), dtype=jnp.float32)

    for slot, (_, l_type, l_amb, l_diff, l_spec, l_att, cos_cut, exp_val, do_sh) in enumerate(active_lights):
      off = slot * 6
      lpx_l = linfo[:, off + 0 : off + 1]
      lpy_l = linfo[:, off + 1 : off + 2]
      lpz_l = linfo[:, off + 2 : off + 3]
      ldx_l = linfo[:, off + 3 : off + 4]
      ldy_l = linfo[:, off + 4 : off + 5]
      ldz_l = linfo[:, off + 5 : off + 6]

      diff_r = diff_r + l_amb[0]
      diff_g = diff_g + l_amb[1]
      diff_b = diff_b + l_amb[2]

      if l_type == 1:
        lx_v = jnp.broadcast_to(-ldx_l, (block_b, block_r))
        ly_v = jnp.broadcast_to(-ldy_l, (block_b, block_r))
        lz_v = jnp.broadcast_to(-ldz_l, (block_b, block_r))
        dist_lim = 1.0e8
        attenuation = jnp.ones((block_b, block_r), dtype=jnp.float32)
      else:
        dlx, dly, dlz = lpx_l - hx, lpy_l - hy, lpz_l - hz
        d_sq = dlx * dlx + dly * dly + dlz * dlz
        d_raw = jnp.sqrt(d_sq)
        inv_dl = 1.0 / jnp.maximum(d_raw, 1e-12)
        lx_v, ly_v, lz_v = dlx * inv_dl, dly * inv_dl, dlz * inv_dl
        dist_lim = jnp.maximum(d_raw - 1.0e-3, 0.0)
        att_denom = l_att[0] + l_att[1] * d_raw + l_att[2] * d_sq
        attenuation = jnp.where(att_denom > 1e-12, 1.0 / jnp.maximum(att_denom, 1e-12), 1.0)
        if l_type == 0:
          cos_theta = -(lx_v * ldx_l + ly_v * ldy_l + lz_v * ldz_l)
          in_cone = cos_theta >= cos_cut
          safe_cos = jnp.where(cos_theta > 0.0, cos_theta, 1.0)
          spot_pow = jnp.where(cos_theta > 0.0, safe_cos ** exp_val, 1.0 if exp_val == 0.0 else 0.0)
          attenuation = jnp.where(in_cone, attenuation * spot_pow, 0.0)

      ndotl = jnp.maximum(0.0, nx_v * lx_v + ny_v * ly_v + nz_v * lz_v)

      if do_sh:
        sh_ox = hx + nx_v * 1.0e-4
        sh_oy = hy + ny_v * 1.0e-4
        sh_oz = hz + nz_v * 1.0e-4
        sh_max = jnp.where(hit & (ndotl > 0.0), dist_lim, 0.0)
        occ_bool = jnp.zeros((block_b, block_r), dtype=jnp.bool_)
        for k, (gt, pfx, pfy) in enumerate(zip(g_types_static, plane_fx, plane_fy)):
          px = gf[0 * block_b : 1 * block_b, k : k + 1]
          py = gf[1 * block_b : 2 * block_b, k : k + 1]
          pz = gf[2 * block_b : 3 * block_b, k : k + 1]
          m00, m01, m02 = gf[3 * block_b : 4 * block_b, k : k + 1], gf[4 * block_b : 5 * block_b, k : k + 1], gf[5 * block_b : 6 * block_b, k : k + 1]
          m10, m11, m12 = gf[6 * block_b : 7 * block_b, k : k + 1], gf[7 * block_b : 8 * block_b, k : k + 1], gf[8 * block_b : 9 * block_b, k : k + 1]
          m20, m21, m22 = gf[9 * block_b : 10 * block_b, k : k + 1], gf[10 * block_b : 11 * block_b, k : k + 1], gf[11 * block_b : 12 * block_b, k : k + 1]
          sx, sy, sz = gf[12 * block_b : 13 * block_b, k : k + 1], gf[13 * block_b : 14 * block_b, k : k + 1], gf[14 * block_b : 15 * block_b, k : k + 1]
          rb_sq = gf[15 * block_b : 16 * block_b, k : k + 1]
          scd, _, _, _ = _ray_one_geom_v2(
              gt, px, py, pz, m00, m01, m02, m10, m11, m12, m20, m21, m22,
              sx, sy, sz, rb_sq, sh_ox, sh_oy, sh_oz, lx_v, ly_v, lz_v,
              plane_finite_x=pfx, plane_finite_y=pfy, compute_normal=False,
          )
          occ_bool = occ_bool | (scd < sh_max)
        vis = jnp.where(hit & (ndotl > 0.0) & occ_bool, shadow_frac, 1.0)
      else:
        vis = 1.0

      ndotl_w = ndotl * attenuation * vis
      diff_r = diff_r + l_diff[0] * ndotl_w
      diff_g = diff_g + l_diff[1] * ndotl_w
      diff_b = diff_b + l_diff[2] * ndotl_w

      if enable_specular:
        hx_v, hy_v, hz_v = lx_v - rdx, ly_v - rdy, lz_v - rdz
        inv_hn = 1.0 / jnp.maximum(jnp.sqrt(hx_v * hx_v + hy_v * hy_v + hz_v * hz_v), 1e-12)
        ndoth = jnp.maximum(0.0, (nx_v * hx_v + ny_v * hy_v + nz_v * hz_v) * inv_hn)
        safe_ndoth = jnp.where(ndoth > 0.0, ndoth, 1.0)
        spec_w = jnp.where(
            (spec_coef > 0.0) & (shin_exp > 0.0) & (ndoth > 0.0),
            spec_coef * (safe_ndoth ** shin_exp) * attenuation * vis,
            0.0,
        )
        spec_r = spec_r + l_spec[0] * spec_w
        spec_g = spec_g + l_spec[1] * spec_w
        spec_b = spec_b + l_spec[2] * spec_w

    if headlight_active:
      ndotl_hl = jnp.maximum(0.0, -(nx_v * rdx + ny_v * rdy + nz_v * rdz))
      diff_r = diff_r + hl_amb[0] + hl_diff[0] * ndotl_hl
      diff_g = diff_g + hl_amb[1] + hl_diff[1] * ndotl_hl
      diff_b = diff_b + hl_amb[2] + hl_diff[2] * ndotl_hl
      if enable_specular:
        safe_ndoth_hl = jnp.where(ndotl_hl > 0.0, ndotl_hl, 1.0)
        spec_w_hl = jnp.where(
            (spec_coef > 0.0) & (shin_exp > 0.0) & (ndotl_hl > 0.0),
            spec_coef * (safe_ndoth_hl ** shin_exp),
            0.0,
        )
        spec_r = spec_r + hl_spec[0] * spec_w_hl
        spec_g = spec_g + hl_spec[1] * spec_w_hl
        spec_b = spec_b + hl_spec[2] * spec_w_hl

    direct_r = jnp.clip(alb_r * diff_r + spec_r + alb_r * emit_coef, 0.0, 1.0)
    direct_g = jnp.clip(alb_g * diff_g + spec_g + alb_g * emit_coef, 0.0, 1.0)
    direct_b = jnp.clip(alb_b * diff_b + spec_b + alb_b * emit_coef, 0.0, 1.0)

    sky_t = jnp.clip(0.5 * (rdz + 1.0), 0.0, 1.0)
    sky_r = sky_t * sky1_r + (1.0 - sky_t) * sky2_r
    sky_g = sky_t * sky1_g + (1.0 - sky_t) * sky2_g
    sky_b = sky_t * sky1_b + (1.0 - sky_t) * sky2_b

    out_r_ref[...] = jnp.where(hit, direct_r, sky_r).astype(out_dtype)
    out_g_ref[...] = jnp.where(hit, direct_g, sky_g).astype(out_dtype)
    out_b_ref[...] = jnp.where(hit, direct_b, sky_b).astype(out_dtype)
    out_d_ref[...] = jnp.where(hit, best_d + inv_cos, -1.0).astype(out_dtype)

  in_specs = [
      pl.BlockSpec((block_b, 128), lambda bi, ri: (bi, 0)),
      pl.BlockSpec((8, block_r), lambda bi, ri: (0, ri)),
      pl.BlockSpec((16 * block_b, G_pad), lambda bi, ri: (bi, 0)),
      pl.BlockSpec((block_b, 128), lambda bi, ri: (bi, 0)),
  ]
  call_args = [cam_128, lray_dev, geom_fields, light_128]
  if has_dynamic_rgba:
    in_specs.append(pl.BlockSpec((4 * block_b, G_pad), lambda bi, ri: (bi, 0)))
    call_args.append(rgba_fields)

  r_out, g_out, b_out, d_out = pl.pallas_call(
      _v3_opt_kernel,
      out_shape=[
          jax.ShapeDtypeStruct((B_pad, R_pad), out_dtype),
          jax.ShapeDtypeStruct((B_pad, R_pad), out_dtype),
          jax.ShapeDtypeStruct((B_pad, R_pad), out_dtype),
          jax.ShapeDtypeStruct((B_pad, R_pad), out_dtype),
      ],
      grid=(n_b_blk, n_r_blk),
      in_specs=in_specs,
      out_specs=[
          pl.BlockSpec((block_b, block_r), lambda bi, ri: (bi, ri)),
          pl.BlockSpec((block_b, block_r), lambda bi, ri: (bi, ri)),
          pl.BlockSpec((block_b, block_r), lambda bi, ri: (bi, ri)),
          pl.BlockSpec((block_b, block_r), lambda bi, ri: (bi, ri)),
      ],
      compiler_params=None if interpret else pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
      interpret=interpret,
  )(*call_args)

  rgb = jnp.stack([r_out[:B, :R], g_out[:B, :R], b_out[:B, :R]], axis=-1).reshape(B, H, W, 3)
  depth = d_out[:B, :R].reshape(B, H, W)
  return FusedFrame(rgb=rgb, depth=depth)


def pallas_raycast_pbr_ao_v3(
    spec: Any,
    mjx_data_batch: Any,
    interpret: bool = True,
    block_b: int = 8,
    block_r: int = 512,
    out_dtype: Any = jnp.float32,
    n_ao_samples: int = 4,
    n_shadow_samples: int = 4,
    ao_radius: float = 0.35,
    ao_strength: float = 0.75,
    shadow_softness: float = 0.06,
) -> FusedFrame:
  """In-VMEM Fused GGX Microfacet PBR + Multi-Sample Soft Shadow + Hemisphere AO shader."""
  cam_pos = mjx_data_batch.cam_xpos[:, int(spec.cam_id), :]
  cam_mat = mjx_data_batch.cam_xmat[:, int(spec.cam_id), :, :]
  geom_xpos = mjx_data_batch.geom_xpos
  geom_xmat = mjx_data_batch.geom_xmat
  light_xpos, light_xdir = _extract_light_batch(spec, mjx_data_batch)

  B = cam_pos.shape[0]
  H, W = int(spec.height), int(spec.width)
  R = H * W

  gids = np.asarray(spec.enabled_geom_ids, dtype=np.int32)
  G = len(gids)
  B_pad = ((B + block_b - 1) // block_b) * block_b
  R_pad = ((R + block_r - 1) // block_r) * block_r
  G_pad = ((G + 127) // 128) * 128
  n_b_blk = B_pad // block_b
  n_r_blk = R_pad // block_r

  cam_pack = jnp.concatenate([cam_pos, cam_mat.reshape(B, 9)], axis=-1)
  cam_128 = jnp.pad(cam_pack, ((0, B_pad - B), (0, 128 - 12)))

  # Pack 16 rows per R_pad: 0..7=camera rays, 8=cos_rot, 9=sin_rot
  ld_np, lo_np, neg_z_np = build_local_camera_rays(spec)
  md_np = ((float(spec.zfar) - float(spec.znear)) / neg_z_np) if spec.zfar > 0.0 else np.full((R,), MJ_MAXVAL, dtype=np.float32)
  cos_rot_np, sin_rot_np = _r2_pixel_rotation(H, W, R_pad)
  lray_raw = np.zeros((16, R_pad), dtype=np.float32)
  lray_raw[0:3, :R] = ld_np.T
  lray_raw[3:6, :R] = lo_np.T
  lray_raw[6, :R] = neg_z_np
  lray_raw[7, :R] = md_np
  lray_raw[2, R:] = -1.0
  lray_raw[6, R:] = 1.0
  lray_raw[7, R:] = 1.0
  lray_raw[8, :] = cos_rot_np
  lray_raw[9, :] = sin_rot_np
  lray_dev = jnp.asarray(lray_raw, dtype=jnp.float32)

  geom_size = jnp.asarray(spec.geom_size, dtype=jnp.float32)
  gpos = geom_xpos[:, gids, :]
  gmat = geom_xmat[:, gids, :, :].reshape(B, G, 9)
  gsz_all = jnp.broadcast_to(geom_size[gids][None, :, :], (B, G, 3))
  rb_sq = jnp.broadcast_to(jnp.asarray(_compute_geom_bound_sq(spec, gids))[None, :, None], (B, G, 1))
  g_16 = jnp.concatenate([gpos, gmat, gsz_all, rb_sq], axis=-1)
  g_16 = jnp.pad(g_16, ((0, B_pad - B), (0, G_pad - G), (0, 0)))
  geom_fields = (
      g_16.reshape(n_b_blk, block_b, G_pad, 16)
      .transpose(0, 3, 1, 2)
      .reshape(n_b_blk * 16 * block_b, G_pad)
  )

  active_lights = []
  light_chunks = []
  for li in range(int(spec.nlight)):
    if not bool(spec.light_active[li]):
      continue
    lp = light_xpos[:, li, :]
    ld_v = light_xdir[:, li, :]
    ld_v = ld_v / jnp.maximum(jnp.linalg.norm(ld_v, axis=-1, keepdims=True), 1e-12)
    light_chunks.append(lp)
    light_chunks.append(ld_v)
    active_lights.append((
        li,
        int(spec.light_type[li]),
        tuple(float(x) for x in spec.light_ambient[li]),
        tuple(float(x) for x in spec.light_diffuse[li]),
        tuple(float(x) for x in spec.light_specular[li]),
        tuple(float(x) for x in spec.light_attenuation[li]),
        float(np.cos(spec.light_cutoff_rad[li])),
        float(spec.light_exponent[li]),
        bool(spec.use_shadows and bool(spec.light_castshadow[li])),
    ))

  if light_chunks:
    l_cat = jnp.concatenate(light_chunks, axis=-1)
    light_128 = jnp.pad(l_cat, ((0, B_pad - B), (0, 128 - l_cat.shape[1])))
  else:
    light_128 = jnp.zeros((B_pad, 128), dtype=jnp.float32)

  g_types_static = tuple(int(spec.geom_type[g]) for g in gids)
  gids_static = tuple(int(g) for g in gids)
  plane_fx = tuple(float(spec.geom_size[g, 0]) > 0.0 for g in gids)
  plane_fy = tuple(float(spec.geom_size[g, 1]) > 0.0 for g in gids)
  g_rgba_static = tuple(tuple(float(c) for c in spec.geom_rgba[g, :3]) for g in gids)
  g_spec_static = tuple(float(spec.geom_spec[g]) for g in gids)
  g_shin_static = tuple(float(spec.geom_shin[g]) for g in gids)
  g_emit_static = tuple(float(spec.geom_emit[g]) for g in gids)
  chk_indices = tuple((k, int(g)) for k, g in enumerate(gids) if bool(spec.geom_is_checker[g]))

  spec_uniform = len(set(g_spec_static)) == 1
  shin_uniform = len(set(g_shin_static)) == 1
  emit_uniform = len(set(g_emit_static)) == 1

  mat_groups: dict[tuple[float, ...], list[int]] = {}
  for k, gid_val in enumerate(gids_static):
    if bool(spec.geom_is_checker[gid_val]):
      continue
    mkey = (*g_rgba_static[k], g_spec_static[k], g_shin_static[k], g_emit_static[k])
    mat_groups.setdefault(mkey, []).append(gid_val)

  # Precompute stratified cosine-weighted AO hemisphere samples & disk shadow samples
  ao_samples = []
  for s in range(max(n_ao_samples, 0)):
    r_s = math.sqrt((s + 0.5) / max(n_ao_samples, 1))
    phi_s = 2.0 * math.pi * ((s + 0.5) * 0.61803398875)
    uz = math.sqrt(max(1.0 - r_s * r_s, 1e-4))
    ao_samples.append((r_s * math.cos(phi_s), r_s * math.sin(phi_s), uz))

  sh_samples = []
  for s in range(max(n_shadow_samples, 1)):
    if n_shadow_samples <= 1:
      sh_samples.append((0.0, 0.0))
    else:
      r_s = math.sqrt((s + 0.5) / n_shadow_samples) * shadow_softness
      phi_s = 2.0 * math.pi * ((s + 0.5) * 0.61803398875)
      sh_samples.append((r_s * math.cos(phi_s), r_s * math.sin(phi_s)))

  znear_f = float(spec.znear)
  rep_x, rep_y = float(spec.checker_repeat[0]), float(spec.checker_repeat[1])
  chk1_r, chk1_g, chk1_b = [float(x) for x in spec.checker_rgb1]
  chk2_r, chk2_g, chk2_b = [float(x) for x in spec.checker_rgb2]
  sky1_r, sky1_g, sky1_b = [float(x) for x in spec.sky_rgb1]
  sky2_r, sky2_g, sky2_b = [float(x) for x in spec.sky_rgb2]
  shadow_frac = float(spec.shadow_light_fraction)
  headlight_active = bool(spec.headlight_active)
  hl_amb = tuple(float(x) for x in spec.headlight_ambient)
  hl_diff = tuple(float(x) for x in spec.headlight_diffuse)
  hl_spec = tuple(float(x) for x in spec.headlight_specular)

  def _pbr_ao_kernel(cam_ref, lray_ref, gf_ref, light_ref, out_r_ref, out_g_ref, out_b_ref, out_d_ref):
    cam = cam_ref[...]
    lr = lray_ref[...]
    gf = gf_ref[...]
    linfo = light_ref[...]

    ldx, ldy, ldz = lr[0:1, :], lr[1:2, :], lr[2:3, :]
    lox, loy, loz = lr[3:4, :], lr[4:5, :], lr[5:6, :]
    inv_cos = znear_f / lr[6:7, :]
    max_d = jnp.broadcast_to(lr[7:8, :], (block_b, block_r))
    cos_rot = lr[8:9, :]
    sin_rot = lr[9:10, :]

    cpx, cpy, cpz = cam[:, 0:1], cam[:, 1:2], cam[:, 2:3]
    c00, c01, c02 = cam[:, 3:4], cam[:, 4:5], cam[:, 5:6]
    c10, c11, c12 = cam[:, 6:7], cam[:, 7:8], cam[:, 8:9]
    c20, c21, c22 = cam[:, 9:10], cam[:, 10:11], cam[:, 11:12]

    rdx_raw = c00 * ldx + c01 * ldy + c02 * ldz
    rdy_raw = c10 * ldx + c11 * ldy + c12 * ldz
    rdz_raw = c20 * ldx + c21 * ldy + c22 * ldz
    inv_rn = 1.0 / jnp.maximum(jnp.sqrt(rdx_raw * rdx_raw + rdy_raw * rdy_raw + rdz_raw * rdz_raw), 1e-12)
    rdx, rdy, rdz = rdx_raw * inv_rn, rdy_raw * inv_rn, rdz_raw * inv_rn

    rox = cpx + (c00 * lox + c01 * loy + c02 * loz) + inv_cos * rdx
    roy = cpy + (c10 * lox + c11 * loy + c12 * loz) + inv_cos * rdy
    roz = cpz + (c20 * lox + c21 * loy + c22 * loz) + inv_cos * rdz

    best_d = max_d
    best_nx = jnp.zeros((block_b, block_r), dtype=jnp.float32)
    best_ny = jnp.zeros((block_b, block_r), dtype=jnp.float32)
    best_nz = jnp.zeros((block_b, block_r), dtype=jnp.float32)
    best_gid = jnp.full((block_b, block_r), -1, dtype=jnp.int32)

    for k, (gt, gid_val, pfx, pfy) in enumerate(zip(g_types_static, gids_static, plane_fx, plane_fy)):
      px = gf[0 * block_b : 1 * block_b, k : k + 1]
      py = gf[1 * block_b : 2 * block_b, k : k + 1]
      pz = gf[2 * block_b : 3 * block_b, k : k + 1]
      m00, m01, m02 = gf[3 * block_b : 4 * block_b, k : k + 1], gf[4 * block_b : 5 * block_b, k : k + 1], gf[5 * block_b : 6 * block_b, k : k + 1]
      m10, m11, m12 = gf[6 * block_b : 7 * block_b, k : k + 1], gf[7 * block_b : 8 * block_b, k : k + 1], gf[8 * block_b : 9 * block_b, k : k + 1]
      m20, m21, m22 = gf[9 * block_b : 10 * block_b, k : k + 1], gf[10 * block_b : 11 * block_b, k : k + 1], gf[11 * block_b : 12 * block_b, k : k + 1]
      sx, sy, sz = gf[12 * block_b : 13 * block_b, k : k + 1], gf[13 * block_b : 14 * block_b, k : k + 1], gf[14 * block_b : 15 * block_b, k : k + 1]
      rb_sq = gf[15 * block_b : 16 * block_b, k : k + 1]
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
    d_safe = jnp.where(hit, best_d, 0.0)
    nx_v = jnp.where(hit, best_nx, 0.0)
    ny_v = jnp.where(hit, best_ny, 0.0)
    nz_v = jnp.where(hit, best_nz, 1.0)  # safe unit fallback for ONB on miss

    hx = rox + d_safe * rdx
    hy = roy + d_safe * rdy
    hz = roz + d_safe * rdz

    alb_r = jnp.zeros((block_b, block_r), dtype=jnp.float32)
    alb_g = jnp.zeros((block_b, block_r), dtype=jnp.float32)
    alb_b = jnp.zeros((block_b, block_r), dtype=jnp.float32)
    spec_coef = jnp.full((block_b, block_r), g_spec_static[0], dtype=jnp.float32) if spec_uniform else jnp.zeros((block_b, block_r), dtype=jnp.float32)
    shin_val = jnp.full((block_b, block_r), g_shin_static[0], dtype=jnp.float32) if shin_uniform else jnp.zeros((block_b, block_r), dtype=jnp.float32)
    emit_coef = jnp.full((block_b, block_r), g_emit_static[0], dtype=jnp.float32) if emit_uniform else jnp.zeros((block_b, block_r), dtype=jnp.float32)

    for mkey, gid_list in mat_groups.items():
      rk, gk, bk, sk, shk, ek = mkey
      is_m = best_gid == gid_list[0]
      for g_other in gid_list[1:]:
        is_m = is_m | (best_gid == g_other)
      alb_r = jnp.where(is_m, rk, alb_r)
      alb_g = jnp.where(is_m, gk, alb_g)
      alb_b = jnp.where(is_m, bk, alb_b)
      if not spec_uniform:
        spec_coef = jnp.where(is_m, sk, spec_coef)
      if not shin_uniform:
        shin_val = jnp.where(is_m, shk, shin_val)
      if not emit_uniform:
        emit_coef = jnp.where(is_m, ek, emit_coef)

    for k, cg in chk_indices:
      px = gf[0 * block_b : 1 * block_b, k : k + 1]
      py = gf[1 * block_b : 2 * block_b, k : k + 1]
      pz = gf[2 * block_b : 3 * block_b, k : k + 1]
      m00, m01 = gf[3 * block_b : 4 * block_b, k : k + 1], gf[4 * block_b : 5 * block_b, k : k + 1]
      m10, m11 = gf[6 * block_b : 7 * block_b, k : k + 1], gf[7 * block_b : 8 * block_b, k : k + 1]
      m20, m21 = gf[9 * block_b : 10 * block_b, k : k + 1], gf[10 * block_b : 11 * block_b, k : k + 1]
      dcx, dcy, dcz = hx - px, hy - py, hz - pz
      lx_cg = m00 * dcx + m10 * dcy + m20 * dcz
      ly_cg = m01 * dcx + m11 * dcy + m21 * dcz
      cx = jnp.floor(lx_cg * rep_x).astype(jnp.int32)
      cy = jnp.floor(ly_cg * rep_y).astype(jnp.int32)
      even = ((cx + cy) & 1) == 0
      is_cg = best_gid == cg
      alb_r = jnp.where(is_cg, jnp.where(even, chk1_r, chk2_r), alb_r)
      alb_g = jnp.where(is_cg, jnp.where(even, chk1_g, chk2_g), alb_g)
      alb_b = jnp.where(is_cg, jnp.where(even, chk1_b, chk2_b), alb_b)

    # In-VMEM Multi-Sample Hemisphere Ambient Occlusion
    sh_ox = hx + nx_v * 2.0e-4
    sh_oy = hy + ny_v * 2.0e-4
    sh_oz = hz + nz_v * 2.0e-4

    if n_ao_samples > 0:
      (tx, ty, tz), (bx, by, bz) = _branchless_onb(nx_v, ny_v, nz_v)
      ao_occ_sum = jnp.zeros((block_b, block_r), dtype=jnp.float32)
      ao_max_d = jnp.where(hit, ao_radius, 0.0)
      for ux0, uy0, uz0 in ao_samples:
        ux = ux0 * cos_rot - uy0 * sin_rot
        uy = ux0 * sin_rot + uy0 * cos_rot
        aodx = ux * tx + uy * bx + uz0 * nx_v
        aody = ux * ty + uy * by + uz0 * ny_v
        aodz = ux * tz + uy * bz + uz0 * nz_v
        min_ao = ao_max_d
        for k, (gt, pfx, pfy) in enumerate(zip(g_types_static, plane_fx, plane_fy)):
          px = gf[0 * block_b : 1 * block_b, k : k + 1]
          py = gf[1 * block_b : 2 * block_b, k : k + 1]
          pz = gf[2 * block_b : 3 * block_b, k : k + 1]
          m00, m01, m02 = gf[3 * block_b : 4 * block_b, k : k + 1], gf[4 * block_b : 5 * block_b, k : k + 1], gf[5 * block_b : 6 * block_b, k : k + 1]
          m10, m11, m12 = gf[6 * block_b : 7 * block_b, k : k + 1], gf[7 * block_b : 8 * block_b, k : k + 1], gf[8 * block_b : 9 * block_b, k : k + 1]
          m20, m21, m22 = gf[9 * block_b : 10 * block_b, k : k + 1], gf[10 * block_b : 11 * block_b, k : k + 1], gf[11 * block_b : 12 * block_b, k : k + 1]
          sx, sy, sz = gf[12 * block_b : 13 * block_b, k : k + 1], gf[13 * block_b : 14 * block_b, k : k + 1], gf[14 * block_b : 15 * block_b, k : k + 1]
          rb_sq = gf[15 * block_b : 16 * block_b, k : k + 1]
          acd, _, _, _ = _ray_one_geom_v2(
              gt, px, py, pz, m00, m01, m02, m10, m11, m12, m20, m21, m22,
              sx, sy, sz, rb_sq, sh_ox, sh_oy, sh_oz, aodx, aody, aodz,
              plane_finite_x=pfx, plane_finite_y=pfy, compute_normal=False,
          )
          min_ao = jnp.minimum(min_ao, acd)
        ao_occ_sum = ao_occ_sum + jnp.where(hit & (min_ao < ao_radius), 1.0 - 0.5 * (min_ao / ao_radius), 0.0)
      ao_vis = jnp.clip(1.0 - ao_strength * (ao_occ_sum / float(n_ao_samples)), 0.15, 1.0)
    else:
      ao_vis = 1.0

    # GGX Microfacet PBR parameters
    roughness = jnp.clip(1.0 - shin_val * 1.4, 0.14, 0.90)
    alpha = roughness * roughness
    a2 = alpha * alpha
    k_g = ((roughness + 1.0) * (roughness + 1.0)) * 0.125
    f0 = 0.04 + 0.08 * spec_coef
    ndotv = jnp.maximum(1e-4, -(nx_v * rdx + ny_v * rdy + nz_v * rdz))
    g1_v = ndotv / (ndotv * (1.0 - k_g) + k_g + 1e-12)

    amb_r = jnp.zeros((block_b, block_r), dtype=jnp.float32)
    amb_g = jnp.zeros((block_b, block_r), dtype=jnp.float32)
    amb_b = jnp.zeros((block_b, block_r), dtype=jnp.float32)
    diff_r = jnp.zeros((block_b, block_r), dtype=jnp.float32)
    diff_g = jnp.zeros((block_b, block_r), dtype=jnp.float32)
    diff_b = jnp.zeros((block_b, block_r), dtype=jnp.float32)
    spec_r = jnp.zeros((block_b, block_r), dtype=jnp.float32)
    spec_g = jnp.zeros((block_b, block_r), dtype=jnp.float32)
    spec_b = jnp.zeros((block_b, block_r), dtype=jnp.float32)

    for slot, (_, l_type, l_amb, l_diff, l_spec, l_att, cos_cut, exp_val, do_sh) in enumerate(active_lights):
      off = slot * 6
      lpx_l = linfo[:, off + 0 : off + 1]
      lpy_l = linfo[:, off + 1 : off + 2]
      lpz_l = linfo[:, off + 2 : off + 3]
      ldx_l = linfo[:, off + 3 : off + 4]
      ldy_l = linfo[:, off + 4 : off + 5]
      ldz_l = linfo[:, off + 5 : off + 6]

      amb_r = amb_r + l_amb[0]
      amb_g = amb_g + l_amb[1]
      amb_b = amb_b + l_amb[2]

      if l_type == 1:
        lx_v = jnp.broadcast_to(-ldx_l, (block_b, block_r))
        ly_v = jnp.broadcast_to(-ldy_l, (block_b, block_r))
        lz_v = jnp.broadcast_to(-ldz_l, (block_b, block_r))
        dist_lim = 1.0e8
        attenuation = jnp.ones((block_b, block_r), dtype=jnp.float32)
      else:
        dlx, dly, dlz = lpx_l - hx, lpy_l - hy, lpz_l - hz
        d_sq = dlx * dlx + dly * dly + dlz * dlz
        d_raw = jnp.sqrt(d_sq)
        inv_dl = 1.0 / jnp.maximum(d_raw, 1e-12)
        lx_v, ly_v, lz_v = dlx * inv_dl, dly * inv_dl, dlz * inv_dl
        dist_lim = jnp.maximum(d_raw - 1.0e-3, 0.0)
        att_denom = l_att[0] + l_att[1] * d_raw + l_att[2] * d_sq
        attenuation = jnp.where(att_denom > 1e-12, 1.0 / jnp.maximum(att_denom, 1e-12), 1.0)
        if l_type == 0:
          cos_theta = -(lx_v * ldx_l + ly_v * ldy_l + lz_v * ldz_l)
          in_cone = cos_theta >= cos_cut
          safe_cos = jnp.where(cos_theta > 0.0, cos_theta, 1.0)
          spot_pow = jnp.where(cos_theta > 0.0, safe_cos ** exp_val, 1.0 if exp_val == 0.0 else 0.0)
          attenuation = jnp.where(in_cone, attenuation * spot_pow, 0.0)

      ndotl = jnp.maximum(0.0, nx_v * lx_v + ny_v * ly_v + nz_v * lz_v)

      if do_sh:
        sh_max = jnp.where(hit & (ndotl > 0.0), dist_lim, 0.0)
        (ltx, lty, ltz), (lbx, lby, lbz) = _branchless_onb(lx_v, ly_v, lz_v)
        occ_sum = jnp.zeros((block_b, block_r), dtype=jnp.float32)
        for dx_d, dy_d in sh_samples:
          if dx_d == 0.0 and dy_d == 0.0:
            sdx, sdy, sdz = lx_v, ly_v, lz_v
          else:
            rx_d = dx_d * cos_rot - dy_d * sin_rot
            ry_d = dx_d * sin_rot + dy_d * cos_rot
            raw_x = lx_v + rx_d * ltx + ry_d * lbx
            raw_y = ly_v + rx_d * lty + ry_d * lby
            raw_z = lz_v + rx_d * ltz + ry_d * lbz
            inv_sn = 1.0 / jnp.maximum(jnp.sqrt(raw_x * raw_x + raw_y * raw_y + raw_z * raw_z), 1e-12)
            sdx, sdy, sdz = raw_x * inv_sn, raw_y * inv_sn, raw_z * inv_sn
          occ_s = jnp.zeros((block_b, block_r), dtype=jnp.bool_)
          for k, (gt, pfx, pfy) in enumerate(zip(g_types_static, plane_fx, plane_fy)):
            px = gf[0 * block_b : 1 * block_b, k : k + 1]
            py = gf[1 * block_b : 2 * block_b, k : k + 1]
            pz = gf[2 * block_b : 3 * block_b, k : k + 1]
            m00, m01, m02 = gf[3 * block_b : 4 * block_b, k : k + 1], gf[4 * block_b : 5 * block_b, k : k + 1], gf[5 * block_b : 6 * block_b, k : k + 1]
            m10, m11, m12 = gf[6 * block_b : 7 * block_b, k : k + 1], gf[7 * block_b : 8 * block_b, k : k + 1], gf[8 * block_b : 9 * block_b, k : k + 1]
            m20, m21, m22 = gf[9 * block_b : 10 * block_b, k : k + 1], gf[10 * block_b : 11 * block_b, k : k + 1], gf[11 * block_b : 12 * block_b, k : k + 1]
            sx, sy, sz = gf[12 * block_b : 13 * block_b, k : k + 1], gf[13 * block_b : 14 * block_b, k : k + 1], gf[14 * block_b : 15 * block_b, k : k + 1]
            rb_sq = gf[15 * block_b : 16 * block_b, k : k + 1]
            scd, _, _, _ = _ray_one_geom_v2(
                gt, px, py, pz, m00, m01, m02, m10, m11, m12, m20, m21, m22,
                sx, sy, sz, rb_sq, sh_ox, sh_oy, sh_oz, sdx, sdy, sdz,
                plane_finite_x=pfx, plane_finite_y=pfy, compute_normal=False,
            )
            occ_s = occ_s | (scd < sh_max)
          occ_sum = occ_sum + (hit & (ndotl > 0.0) & occ_s).astype(jnp.float32)
        vis = 1.0 - (1.0 - shadow_frac) * (occ_sum / float(len(sh_samples)))
      else:
        vis = 1.0

      # GGX Microfacet Cook-Torrance BRDF
      hx_v, hy_v, hz_v = lx_v - rdx, ly_v - rdy, lz_v - rdz
      inv_hn = 1.0 / jnp.maximum(jnp.sqrt(hx_v * hx_v + hy_v * hy_v + hz_v * hz_v), 1e-12)
      hx_n, hy_n, hz_n = hx_v * inv_hn, hy_v * inv_hn, hz_v * inv_hn
      ndoth = jnp.maximum(0.0, nx_v * hx_n + ny_v * hy_n + nz_v * hz_n)
      vdoth = jnp.maximum(0.0, -(rdx * hx_n + rdy * hy_n + rdz * hz_n))

      denom_d = ndoth * ndoth * (a2 - 1.0) + 1.0
      d_ggx = a2 / (math.pi * denom_d * denom_d + 1e-12)
      omv = 1.0 - vdoth
      omv2 = omv * omv
      f_schlick = f0 + (1.0 - f0) * (omv2 * omv2 * omv)
      g1_l = ndotl / (ndotl * (1.0 - k_g) + k_g + 1e-12)
      spec_brdf = (d_ggx * f_schlick * g1_l * g1_v) / (4.0 * ndotl * ndotv + 1e-4)

      weight = attenuation * vis * ndotl
      kd = 1.0 - f_schlick
      diff_r = diff_r + l_diff[0] * weight * kd
      diff_g = diff_g + l_diff[1] * weight * kd
      diff_b = diff_b + l_diff[2] * weight * kd

      spec_fac = jnp.where((spec_coef > 0.0) & (ndotl > 0.0), spec_brdf * weight * math.pi, 0.0)
      spec_r = spec_r + l_spec[0] * spec_fac
      spec_g = spec_g + l_spec[1] * spec_fac
      spec_b = spec_b + l_spec[2] * spec_fac

    if headlight_active:
      amb_r = amb_r + hl_amb[0]
      amb_g = amb_g + hl_amb[1]
      amb_b = amb_b + hl_amb[2]
      ndotl_hl = ndotv
      diff_r = diff_r + hl_diff[0] * ndotl_hl
      diff_g = diff_g + hl_diff[1] * ndotl_hl
      diff_b = diff_b + hl_diff[2] * ndotl_hl
      # Subtle GGX headlight highlight (H == V == -ray_dir => ndoth=ndotv, vdoth=1)
      denom_hl = ndotv * ndotv * (a2 - 1.0) + 1.0
      d_hl = a2 / (math.pi * denom_hl * denom_hl + 1e-12)
      spec_hl = jnp.where(spec_coef > 0.0, d_hl * f0 * g1_v * g1_v / (4.0 * ndotv + 1e-4) * math.pi, 0.0)
      spec_r = spec_r + hl_spec[0] * spec_hl
      spec_g = spec_g + hl_spec[1] * spec_hl
      spec_b = spec_b + hl_spec[2] * spec_hl

    # Combine AO-modulated ambient + direct GGX diffuse/specular + emission
    tot_r = alb_r * (amb_r * ao_vis + diff_r + emit_coef) + spec_r * (0.5 + 0.5 * ao_vis)
    tot_g = alb_g * (amb_g * ao_vis + diff_g + emit_coef) + spec_g * (0.5 + 0.5 * ao_vis)
    tot_b = alb_b * (amb_b * ao_vis + diff_b + emit_coef) + spec_b * (0.5 + 0.5 * ao_vis)

    direct_r = jnp.clip(tot_r, 0.0, 1.0)
    direct_g = jnp.clip(tot_g, 0.0, 1.0)
    direct_b = jnp.clip(tot_b, 0.0, 1.0)

    sky_t = jnp.clip(0.5 * (rdz + 1.0), 0.0, 1.0)
    sky_r = sky_t * sky1_r + (1.0 - sky_t) * sky2_r
    sky_g = sky_t * sky1_g + (1.0 - sky_t) * sky2_g
    sky_b = sky_t * sky1_b + (1.0 - sky_t) * sky2_b

    out_r_ref[...] = jnp.where(hit, direct_r, sky_r).astype(out_dtype)
    out_g_ref[...] = jnp.where(hit, direct_g, sky_g).astype(out_dtype)
    out_b_ref[...] = jnp.where(hit, direct_b, sky_b).astype(out_dtype)
    out_d_ref[...] = jnp.where(hit, best_d + inv_cos, -1.0).astype(out_dtype)

  r_out, g_out, b_out, d_out = pl.pallas_call(
      _pbr_ao_kernel,
      out_shape=[
          jax.ShapeDtypeStruct((B_pad, R_pad), out_dtype),
          jax.ShapeDtypeStruct((B_pad, R_pad), out_dtype),
          jax.ShapeDtypeStruct((B_pad, R_pad), out_dtype),
          jax.ShapeDtypeStruct((B_pad, R_pad), out_dtype),
      ],
      grid=(n_b_blk, n_r_blk),
      in_specs=[
          pl.BlockSpec((block_b, 128), lambda bi, ri: (bi, 0)),
          pl.BlockSpec((16, block_r), lambda bi, ri: (0, ri)),
          pl.BlockSpec((16 * block_b, G_pad), lambda bi, ri: (bi, 0)),
          pl.BlockSpec((block_b, 128), lambda bi, ri: (bi, 0)),
      ],
      out_specs=[
          pl.BlockSpec((block_b, block_r), lambda bi, ri: (bi, ri)),
          pl.BlockSpec((block_b, block_r), lambda bi, ri: (bi, ri)),
          pl.BlockSpec((block_b, block_r), lambda bi, ri: (bi, ri)),
          pl.BlockSpec((block_b, block_r), lambda bi, ri: (bi, ri)),
      ],
      compiler_params=None if interpret else pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
      interpret=interpret,
  )(cam_128, lray_dev, geom_fields, light_128)

  rgb = jnp.stack([r_out[:B, :R], g_out[:B, :R], b_out[:B, :R]], axis=-1).reshape(B, H, W, 3)
  depth = d_out[:B, :R].reshape(B, H, W)
  return FusedFrame(rgb=rgb, depth=depth)
