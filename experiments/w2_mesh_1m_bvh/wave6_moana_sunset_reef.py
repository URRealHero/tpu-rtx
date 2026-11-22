"""Wave 6: Motunui Golden-Hour Sunset + Avatar Way-of-Water Underwater Coral Reef @ Native 4K.

Reuses the Disney Moana OpenUSD v2.1 (22.30B instanced tris) Wide-TLAS-64 + Compact BLAS-64
TPU Mosaic Pallas pipeline from Wave 5 with two cinema shaders:
  1. Panel 1 (Motunui Golden-Hour Sunset, shotCam @ 3840x2160):
     - Low-angle 10 deg sunset sun over the lagoon channel
     - Physically-inspired Rayleigh + Mie crimson/amber horizon glow blended with islandsun.exr clouds
     - Long dramatic palm shadows + golden foliage SSS + molten-gold wave specular glitter path
     - AgX filmic tonemapping
  2. Panel 2 (Way of Water Underwater Coral Reef, reefCam @ 3840x2160):
     - Camera submerged inside the Moana lagoon looking across real isCoral & isLavaRocks formations
     - Snell's Window total internal reflection / sky refraction on the underside of osOcean
     - Beer-Lambert cyan-emerald underwater absorption
     - Analytical optical-Jacobian wave-focused caustic ripples projected onto corals & sand
     - 16-step HG-phase volumetric underwater sunbeams (God Rays) with 4-tap real BVH shadow shafts

All TPU kernels verified interpret=False, host_callbacks == 0, gather_ops == 0, send_recv_ops == 0.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tpu_mjx_render.roofline import inspect_hlo_ops
from wave3_500m_10b_8k import _pallas_compact_bvh_raycast
from wide_bvh_v3 import TILE_H
from wave5_moana_4k import TAG_TO_ID, _load_compiled_cache, agx_filmic_tonemap, pallas_sample_hdri_env
from wave5_moana_8k_mjx import (
    _compute_camera_rays_subband,
    _pallas_tlas_precomp_raycast,
    _precompute_tlas_group,
    extract_all_moana_cameras,
    load_islandsun_hdri_polished,
)


def make_lookat_camera(
    name: str,
    pos: Tuple[float, float, float],
    target: Tuple[float, float, float],
    fovy_deg: float = 54.0,
) -> Dict[str, Any]:
  """Build Z-up camera dictionary matching OpenUSD camera convention ([right, up, -fwd])."""
  p = np.asarray(pos, dtype=np.float32)
  t = np.asarray(target, dtype=np.float32)
  fwd = t - p
  fwd /= max(float(np.linalg.norm(fwd)), 1e-8)
  world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
  right = np.cross(fwd, world_up)
  if float(np.linalg.norm(right)) < 1e-6:
    right = np.cross(fwd, np.array([0.0, 1.0, 0.0], dtype=np.float32))
  right /= max(float(np.linalg.norm(right)), 1e-8)
  up = np.cross(right, fwd)
  up /= max(float(np.linalg.norm(up)), 1e-8)
  R_cam = np.stack([right, up, -fwd], axis=1).astype(np.float32)
  tan_half_fovy = math.tan(0.5 * math.radians(fovy_deg))
  return {
      "path": f"/island/cam/{name}",
      "cam_xpos": p[None, :],
      "cam_xmat": R_cam[None, :, :],
      "tan_half_fovy": np.array([tan_half_fovy], dtype=np.float32),
      "fovy_deg": float(fovy_deg),
  }


def _ocean_wave_slope_and_jacobian(
    wx: jax.Array,
    wy: jax.Array,
    focus_scale: jax.Array,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
  """Pure-ALU multi-octave lagoon wave slopes (hx, hy) & optical caustic Jacobian determinant.

  Zero texture lookups / zero gather ops; evaluates 6 Gerstner-like sinusoidal harmonics
  and their analytical Hessian for physically-consistent water normals & focused caustics.
  """
  # (kx, ky, amp, phase)
  waves = (
      (1.55, 0.92, 0.0150, 0.30),
      (-1.18, 1.95, 0.0115, 1.70),
      (3.40, -2.35, 0.0068, 2.90),
      (-2.75, -4.05, 0.0056, 0.80),
      (5.65, 3.25, 0.0033, 4.10),
      (-4.85, 6.10, 0.0027, 2.20),
  )
  hx = jnp.zeros_like(wx)
  hy = jnp.zeros_like(wx)
  hxx = jnp.zeros_like(wx)
  hyy = jnp.zeros_like(wx)
  hxy = jnp.zeros_like(wx)
  for kx, ky, amp, ph in waves:
    theta = kx * wx + ky * wy + ph
    c = jnp.cos(theta)
    s = jnp.sin(theta)
    hx = hx + (amp * kx) * c
    hy = hy + (amp * ky) * c
    hxx = hxx - (amp * kx * kx) * s
    hyy = hyy - (amp * ky * ky) * s
    hxy = hxy - (amp * kx * ky) * s

  def _caustic_i(sc: jax.Array) -> jax.Array:
    s_eff = sc * 2.35
    det_j = (1.0 + s_eff * hxx) * (1.0 + s_eff * hyy) - (s_eff * hxy) * (s_eff * hxy)
    raw = jnp.clip(0.32 / jnp.maximum(jnp.abs(det_j), 0.065) - 0.36, 0.0, 5.5)
    return jnp.power(raw, 1.35)

  # Subtle chromatic dispersion across R/G/B refracted focal planes
  c_r = _caustic_i(focus_scale * 0.94)
  c_g = _caustic_i(focus_scale * 1.00)
  c_b = _caustic_i(focus_scale * 1.07)
  caustic_rgb = jnp.stack([c_r, c_g, c_b], axis=-1)
  return hx, hy, caustic_rgb


def _sunset_sky_radiance(
    dirs: jax.Array,
    hdri_rgb: jax.Array,
    sun_dir: jax.Array,
) -> jax.Array:
  """Golden-hour Rayleigh+Mie sunset atmosphere modulated by islandsun.exr cloud silhouettes."""
  dz = jnp.clip(dirs[..., 2], -0.2, 1.0)
  horizon_w = jnp.exp(-jnp.maximum(dz, 0.0) * 6.2)
  mid_w = jnp.exp(-jnp.maximum(dz, 0.0) * 2.0)

  # Solar alignment & azimuthal glow
  mu = jnp.maximum(0.0, jnp.sum(dirs * sun_dir, axis=-1))
  d_xy = dirs[..., :2] / jnp.maximum(jnp.linalg.norm(dirs[..., :2], axis=-1, keepdims=True), 1e-6)
  s_xy = sun_dir[..., :2] / jnp.maximum(jnp.linalg.norm(sun_dir[..., :2], axis=-1, keepdims=True), 1e-6)
  az_align = jnp.maximum(0.0, jnp.sum(d_xy * s_xy, axis=-1))

  # Rich saturated sunset palette: deep twilight indigo zenith -> crimson-rose mid -> fiery amber-crimson horizon
  zenith_col = jnp.array([0.028, 0.055, 0.20], dtype=jnp.float32)[None, None, :]
  mid_col = jnp.array([0.38, 0.11, 0.22], dtype=jnp.float32)[None, None, :]
  hor_base = jnp.array([0.95, 0.18, 0.06], dtype=jnp.float32)[None, None, :]
  hor_sun = jnp.array([1.95, 0.56, 0.08], dtype=jnp.float32)[None, None, :]

  sky_grad = (
      zenith_col * (1.0 - mid_w[..., None])
      + mid_col * (mid_w - horizon_w)[..., None]
      + (hor_base + hor_sun * (az_align ** 2.4)[..., None]) * horizon_w[..., None]
  )

  # Extract bright cloud density from real islandsun.exr HDRI & rim-light with golden-crimson sunset
  hdri_lum = 0.2126 * hdri_rgb[..., 0] + 0.7152 * hdri_rgb[..., 1] + 0.0722 * hdri_rgb[..., 2]
  cloud_mask = jnp.clip((hdri_lum - 0.56) * 1.55, 0.0, 1.0) * jnp.clip((dz + 0.02) * 8.0, 0.0, 1.0)
  cloud_lit = (
      jnp.array([0.28, 0.12, 0.22], dtype=jnp.float32)[None, None, :]
      + jnp.array([1.75, 0.55, 0.12], dtype=jnp.float32)[None, None, :] * (0.20 + 0.80 * (az_align ** 1.6) * (0.30 + 0.70 * horizon_w))[..., None]
  )
  sky_with_clouds = sky_grad * (1.0 - 0.72 * cloud_mask[..., None]) + cloud_lit * (0.82 * cloud_mask[..., None])

  # Mie forward-scattering aureole + blazing sunset solar disk
  mie_broad = (mu ** 8.0)[..., None] * jnp.array([1.45, 0.42, 0.09], dtype=jnp.float32)[None, None, :]
  mie_tight = (mu ** 52.0)[..., None] * jnp.array([4.20, 1.65, 0.35], dtype=jnp.float32)[None, None, :]
  sun_disk = (mu ** 1100.0)[..., None] * jnp.array([26.0, 12.5, 3.2], dtype=jnp.float32)[None, None, :]
  return jnp.maximum(sky_with_clouds + mie_broad + mie_tight + sun_disk, 0.0)


def _moana_albedo_and_roughness(
    hit_pos: jax.Array,
    best_hit: jax.Array,
    best_nrm: jax.Array,
    best_alb: jax.Array,
    best_rough: jax.Array,
    best_tag: jax.Array,
    water_hit: jax.Array,
    water_d: jax.Array,
    best_d: jax.Array,
    underwater_mode: bool = False,
) -> Tuple[jax.Array, jax.Array]:
  """Procedural world-space PBR albedo & roughness across Moana sand, foliage, bark, rock, and coral."""
  px, py, pz = hit_pos[..., 0], hit_pos[..., 1], hit_pos[..., 2]
  n_wave1 = jnp.sin(px * 0.37 + py * 0.83 + pz * 0.29) * jnp.sin(py * 0.41 - px * 0.71 + pz * 0.53)
  n_wave2 = jnp.sin(px * 1.93 - py * 2.71 + pz * 1.37) * jnp.cos(py * 2.19 + px * 1.57 - pz * 2.41)

  sand_rip = jnp.sin(px * 8.5 + py * 4.2 + 0.7 * n_wave1)
  sand_scale = (0.94 + 0.07 * sand_rip + 0.04 * n_wave2)[..., None]
  if underwater_mode:
    # Bright carbonate coral sand floor reflects vivid cyan-turquoise caustic patterns
    sand_alb = best_alb * jnp.array([1.18, 1.16, 1.05], dtype=jnp.float32)[None, None, :] * sand_scale
  else:
    sand_alb = best_alb * jnp.array([1.08, 0.97, 0.80], dtype=jnp.float32)[None, None, :] * sand_scale

  fol_var = 0.52 + 0.22 * n_wave1 + 0.13 * n_wave2
  fol_alb = best_alb * jnp.stack([fol_var * 0.72, fol_var * 0.96, fol_var * 0.45], axis=-1)
  bark_alb = best_alb * (0.82 + 0.18 * n_wave1 + 0.10 * n_wave2)[..., None]

  rock_var = 0.72 + 0.28 * n_wave1 + 0.15 * n_wave2
  wet_mask = jnp.clip(1.0 - (pz + 0.08) / 0.75, 0.0, 1.0)
  rock_alb = best_alb * rock_var[..., None] * (1.0 - 0.38 * wet_mask[..., None])
  eff_rough = jnp.where(best_tag == 4, best_rough * (1.0 - 0.48 * wet_mask), best_rough)

  if underwater_mode:
    # Rich tropical reef biodiversity: warm ochre/amber staghorn, chartreuse/emerald cabbage coral, rose-coral tips
    patch_a = 0.5 + 0.5 * jnp.sin(px * 1.45 - py * 1.85 + pz * 2.1)
    patch_b = 0.5 + 0.5 * jnp.cos(px * 2.60 + py * 1.90 - pz * 1.7)
    coral_col_a = jnp.array([0.78, 0.56, 0.22], dtype=jnp.float32)[None, None, :]   # golden staghorn ochre
    coral_col_b = jnp.array([0.46, 0.74, 0.24], dtype=jnp.float32)[None, None, :]   # sunlit olive-lime cabbage coral
    coral_col_c = jnp.array([0.82, 0.40, 0.38], dtype=jnp.float32)[None, None, :]   # warm rose-terracotta coral accent
    coral_mix = coral_col_a * (1.0 - patch_a[..., None]) + coral_col_b * patch_a[..., None]
    coral_mix = coral_mix * (1.0 - 0.35 * patch_b[..., None]) + coral_col_c * (0.35 * patch_b[..., None])
    coral_alb = (0.55 * best_alb * 1.8 + 0.45 * coral_mix) * (0.85 + 0.25 * n_wave2)[..., None]
  else:
    coral_alb = best_alb * jnp.stack([1.35 + 0.42 * n_wave1, 1.65 + 0.45 * n_wave2, 0.38 + 0.18 * n_wave1], axis=-1)

  albedo = jnp.where((best_tag == 1)[..., None], sand_alb, best_alb)
  albedo = jnp.where((best_tag == 2)[..., None], fol_alb, albedo)
  albedo = jnp.where((best_tag == 3)[..., None], bark_alb, albedo)
  albedo = jnp.where((best_tag == 4)[..., None], rock_alb, albedo)
  albedo = jnp.where((best_tag == 5)[..., None], coral_alb, albedo)

  if underwater_mode:
    reef_moss = jnp.clip(best_nrm[..., 2] * (0.55 + 0.35 * n_wave1), 0.0, 1.0) * ((best_tag == 4) | (best_tag == 0)).astype(jnp.float32)
    reef_tint = jnp.array([0.32, 0.50, 0.16], dtype=jnp.float32)[None, None, :] * (0.75 + 0.45 * n_wave2)[..., None]
    albedo = albedo * (1.0 - 0.50 * reef_moss[..., None]) + reef_tint * (0.50 * reef_moss[..., None])
  else:
    is_uw = water_hit & (water_d < best_d) & best_hit & ((best_tag == 4) | (best_tag == 5) | (best_tag == 0))
    reef_moss = jnp.clip(best_nrm[..., 2] * (0.55 + 0.35 * n_wave1), 0.0, 1.0)
    reef_tint = jnp.array([0.28, 0.44, 0.08], dtype=jnp.float32)[None, None, :] * (0.75 + 0.45 * n_wave2)[..., None]
    albedo = jnp.where(is_uw[..., None], albedo * (1.0 - 0.65 * reef_moss[..., None]) + reef_tint * (0.65 * reef_moss[..., None]), albedo)

  return jnp.clip(albedo, 0.01, 0.98), eff_rough


def make_moana_wave6_renderer(
    compiled: Dict[str, Any],
    env_bf16: jax.Array,
    devices: List[jax.Device],
    H: int,
    W: int,
    n_subbands: int,
    mode: str,
    sun_dir_np: np.ndarray,
    use_shadows: bool = True,
    vol_steps: int = 16,
    vol_shadow_taps: int = 4,
):
  """Build 4-chip spatial-sharded Moana renderer for 'sunset' or 'underwater_reef' mode."""
  assert mode in ("sunset", "underwater_reef"), mode
  n_chips = len(devices)
  assert H % n_chips == 0, (H, n_chips)
  band_h = H // n_chips
  assert band_h % n_subbands == 0, (band_h, n_subbands)
  sub_h = band_h // n_subbands
  sub_h_pad = ((sub_h + TILE_H - 1) // TILE_H) * TILE_H

  mesh = Mesh(np.array(devices), ("spatial",))
  rep = NamedSharding(mesh, P())

  u_spec = compiled["unique_spec"]
  u_devs = compiled["unique_dev_arrays"]
  u_pbr = compiled["unique_pbr"]
  w_spec = compiled["water_spec"]
  w_devs = compiled["water_dev_arrays"]
  i_spec = compiled["inst_spec"]
  i_devs = compiled["inst_dev_arrays"]
  i_pbr = compiled["inst_pbr"]
  base_spec = u_spec if u_spec is not None else i_spec
  znear_f = float(base_spec.znear)
  zfar_f = float(base_spec.zfar)

  inst_inv_xmats = []
  for xm in compiled["inst_xmat"]:
    xm_np = np.asarray(xm)
    inv_np = np.zeros_like(xm_np)
    flat_m = xm_np.reshape(-1, 3, 3)
    flat_inv = inv_np.reshape(-1, 3, 3)
    for idx in range(flat_m.shape[0]):
      try:
        flat_inv[idx] = np.linalg.inv(flat_m[idx])
      except np.linalg.LinAlgError:
        flat_inv[idx] = np.linalg.pinv(flat_m[idx])
    inst_inv_xmats.append(jax.device_put(inv_np.astype(np.float32), rep))
  inst_inv_xmats_t = tuple(inst_inv_xmats)

  sun_unit = np.asarray(sun_dir_np, dtype=np.float32)
  sun_unit /= max(float(np.linalg.norm(sun_unit)), 1e-8)

  # Snell-refracted sun direction underwater (n_air=1.0 -> n_water=1.333)
  eta = 1.0 / 1.333
  s_xy = sun_unit[:2] * eta
  s_z = math.sqrt(max(1.0 - float(np.dot(s_xy, s_xy)), 0.04))
  sun_refr_unit = np.array([s_xy[0], s_xy[1], s_z], dtype=np.float32)
  sun_refr_unit /= max(float(np.linalg.norm(sun_refr_unit)), 1e-8)

  water_z_mean = -0.266

  def _trace_shadow_occ(tlas_precomps, u_da, u_oxp, u_oxm, i_da, sh_o, sh_dir, sh_lim):
    """Exact BVH+TLAS any-hit shadow occlusion test (zero host callbacks, zero gather)."""
    occ = jnp.zeros(sh_lim.shape, dtype=jnp.bool_)
    if u_spec is not None:
      for m_idx, _it in enumerate(u_spec.meshes):
        l2_r, l2_a, l0_a, leaf_b = u_da[m_idx]
        p = u_oxp[:, m_idx, :]
        rmat = u_oxm[:, m_idx, :, :]
        ro_l = jnp.einsum("bji,brj->bri", rmat, sh_o - p[:, None, :], precision=jax.lax.Precision.HIGHEST)
        rd_l = jnp.einsum("bji,brj->bri", rmat, sh_dir, precision=jax.lax.Precision.HIGHEST)
        sh_lim_rem = jnp.where(occ, 0.0, sh_lim)
        d_sh, _, _ = _pallas_compact_bvh_raycast(l2_r, l2_a, l0_a, leaf_b, ro_l, rd_l, sh_lim_rem, sub_h_pad, W, any_hit=True, interpret=False)
        occ = occ | (d_sh < sh_lim_rem)
    if i_spec is not None:
      for m_idx, _it in enumerate(i_spec.meshes):
        l2_r, l2_a, l0_a, leaf_b = i_da[m_idx]
        sh_lim_rem = jnp.where(occ, 0.0, sh_lim)
        d_sh, _ = _pallas_tlas_precomp_raycast(
            tlas_precomps[m_idx], l2_r, l2_a, l0_a, leaf_b,
            sh_o, sh_dir, sh_lim_rem, sub_h_pad, W,
            tlas_group_size=64, tlas_top_groups=4, max_active_per_tile=16, any_hit=True,
        )
        occ = occ | (d_sh < sh_lim_rem)
    return occ

  def _shard_fn(u_da, u_oxp, u_oxm, w_da, i_da, i_xp, i_xm, i_inv, cxp, cxm, tan_hf, env_b):
    bidx = jax.lax.axis_index("spatial")
    tlas_precomps = []
    if i_spec is not None:
      for m_idx, it in enumerate(i_spec.meshes):
        tlas_precomps.append(_precompute_tlas_group(it, i_xp[m_idx], i_xm[m_idx], i_inv[m_idx], tlas_group_size=64))

    def _render_subband(_, sb_idx):
      ray_orig, ray_dir, neg_z = _compute_camera_rays_subband(
          cxp, cxm, tan_hf, znear_f, H, W, bidx, band_h, sb_idx, sub_h, sub_h_pad
      )
      B, R, _ = ray_orig.shape
      max_dist = jnp.broadcast_to(((zfar_f - znear_f) / neg_z)[None, :], (B, R))

      best_d = max_dist
      best_hit = jnp.zeros((B, R), dtype=jnp.bool_)
      best_nrm = jnp.zeros((B, R, 3), dtype=jnp.float32)
      best_alb = jnp.zeros((B, R, 3), dtype=jnp.float32)
      best_rough = jnp.full((B, R), 0.72, dtype=jnp.float32)
      best_tag = jnp.zeros((B, R), dtype=jnp.int32)

      if u_spec is not None:
        for m_idx, it in enumerate(u_spec.meshes):
          l2_r, l2_a, l0_a, leaf_b = u_da[m_idx]
          p = u_oxp[:, m_idx, :]
          rmat = u_oxm[:, m_idx, :, :]
          ro_loc = jnp.einsum("bji,brj->bri", rmat, ray_orig - p[:, None, :], precision=jax.lax.Precision.HIGHEST)
          rd_loc = jnp.einsum("bji,brj->bri", rmat, ray_dir, precision=jax.lax.Precision.HIGHEST)
          d_m, n_loc_m, _ = _pallas_compact_bvh_raycast(l2_r, l2_a, l0_a, leaf_b, ro_loc, rd_loc, best_d, sub_h_pad, W, any_hit=False, interpret=False)
          imp = d_m < best_d
          n_w_m = jnp.einsum("bij,brj->bri", rmat, n_loc_m, precision=jax.lax.Precision.HIGHEST)
          best_d = jnp.where(imp, d_m, best_d)
          best_hit = best_hit | imp
          best_nrm = jnp.where(imp[..., None], n_w_m, best_nrm)
          best_alb = jnp.where(imp[..., None], jnp.asarray(it.rgba[:3], dtype=jnp.float32)[None, None, :], best_alb)
          r_m, _, tag_m = u_pbr[m_idx]
          best_rough = jnp.where(imp, float(r_m), best_rough)
          best_tag = jnp.where(imp, jnp.int32(TAG_TO_ID.get(tag_m, 0)), best_tag)

      if i_spec is not None:
        for m_idx, it in enumerate(i_spec.meshes):
          l2_r, l2_a, l0_a, leaf_b = i_da[m_idx]
          d_m, n_w_m = _pallas_tlas_precomp_raycast(
              tlas_precomps[m_idx], l2_r, l2_a, l0_a, leaf_b,
              ray_orig, ray_dir, best_d, sub_h_pad, W,
              tlas_group_size=64, tlas_top_groups=6, max_active_per_tile=24, any_hit=False,
          )
          imp = d_m < best_d
          best_d = jnp.where(imp, d_m, best_d)
          best_hit = best_hit | imp
          best_nrm = jnp.where(imp[..., None], n_w_m, best_nrm)
          best_alb = jnp.where(imp[..., None], jnp.asarray(it.rgba[:3], dtype=jnp.float32)[None, None, :], best_alb)
          r_m, _, tag_m = i_pbr[m_idx]
          best_rough = jnp.where(imp, float(r_m), best_rough)
          best_tag = jnp.where(imp, jnp.int32(TAG_TO_ID.get(tag_m, 0)), best_tag)

      water_d = max_dist
      water_hit = jnp.zeros((B, R), dtype=jnp.bool_)
      water_nrm = jnp.zeros((B, R, 3), dtype=jnp.float32)
      if w_spec is not None and len(w_spec.meshes) > 0:
        for m_idx, _it in enumerate(w_spec.meshes):
          l2_r, l2_a, l0_a, leaf_b = w_da[m_idx]
          d_w, n_w, _ = _pallas_compact_bvh_raycast(l2_r, l2_a, l0_a, leaf_b, ray_orig, ray_dir, water_d, sub_h_pad, W, any_hit=False, interpret=False)
          imp_w = d_w < water_d
          water_d = jnp.where(imp_w, d_w, water_d)
          water_hit = water_hit | imp_w
          water_nrm = jnp.where(imp_w[..., None], n_w, water_nrm)

      vdir = -ray_dir
      best_nrm = jnp.where(best_hit[..., None], best_nrm, 0.0)
      flip = jnp.sum(best_nrm * vdir, axis=-1, keepdims=True) < 0.0
      best_nrm = jnp.where(flip, -best_nrm, best_nrm)
      hit_pos = ray_orig + jnp.where(best_hit, best_d, 0.0)[..., None] * ray_dir

      albedo, eff_rough = _moana_albedo_and_roughness(
          hit_pos, best_hit, best_nrm, best_alb, best_rough, best_tag,
          water_hit, water_d, best_d, underwater_mode=(mode == "underwater_reef"),
      )

      L_sun = jnp.broadcast_to(jnp.asarray(sun_unit, dtype=jnp.float32)[None, None, :], ray_dir.shape)
      L_refr = jnp.broadcast_to(jnp.asarray(sun_refr_unit, dtype=jnp.float32)[None, None, :], ray_dir.shape)
      L_shade = L_refr if mode == "underwater_reef" else L_sun
      ndotl = jnp.maximum(0.0, jnp.sum(best_nrm * L_shade, axis=-1))

      vis = jnp.ones((B, R), dtype=jnp.float32)
      if use_shadows:
        sh_bias = jnp.maximum(0.03, best_d * 0.0012)[..., None]
        sh_o = hit_pos + best_nrm * sh_bias
        sh_lim = jnp.where(best_hit & (ndotl > 0.02), 350.0, 0.0)
        occ = _trace_shadow_occ(tlas_precomps, u_da, u_oxp, u_oxm, i_da, sh_o, L_shade, sh_lim)
        vis = jnp.where(occ, 0.04 if mode == "sunset" else 0.08, 1.0)

      if mode == "sunset":
        # -------------------------------------------------------------------
        # Mode 1: Motunui Golden-Hour Sunset (10 deg sun, crimson/amber glow,
        #         long dramatic palm shadows, golden specular ocean glitter)
        # -------------------------------------------------------------------
        sun_col = jnp.array([5.40, 2.35, 0.72], dtype=jnp.float32)[None, None, :]
        H_vec = (L_sun + vdir) / jnp.maximum(jnp.linalg.norm(L_sun + vdir, axis=-1, keepdims=True), 1e-12)
        ndoth = jnp.maximum(0.0, jnp.sum(best_nrm * H_vec, axis=-1))
        ndotv = jnp.maximum(1e-3, jnp.sum(best_nrm * vdir, axis=-1))
        alpha = jnp.clip(eff_rough * eff_rough, 0.03, 0.98)
        a2 = alpha * alpha
        denom = ndoth * ndoth * (a2 - 1.0) + 1.0
        D_ggx = a2 / (np.pi * denom * denom + 1e-6)
        F_schlick = 0.04 + 0.96 * ((1.0 - jnp.maximum(0.0, jnp.sum(H_vec * vdir, axis=-1))) ** 5.0)
        spec_term = jnp.minimum((D_ggx * F_schlick) / (4.0 * ndotv * jnp.maximum(ndotl, 0.05) + 0.14), 3.2) * vis

        # Warm golden backlit palm & tropical canopy translucency
        sss_dir = jnp.maximum(0.0, jnp.sum(vdir * (-L_sun + 0.35 * best_nrm), axis=-1)) ** 2.1
        sss_mask = ((best_tag == 2) | (best_tag == 6)).astype(jnp.float32)
        sss_rgb = albedo * sun_col * (0.62 * sss_dir * sss_mask * (0.20 + 0.80 * vis))[..., None]

        # Dusk indigo-violet zenith shadow fill + warm amber horizon bounce
        sky_up = 0.5 + 0.5 * best_nrm[..., 2]
        sky_fill = (
            sky_up[..., None] * jnp.array([0.11, 0.10, 0.24], dtype=jnp.float32)[None, None, :]
            + (1.0 - sky_up[..., None]) * jnp.array([0.14, 0.06, 0.04], dtype=jnp.float32)[None, None, :]
        )
        ao = jnp.where(best_tag == 2, 0.30 + 0.70 * (sky_up ** 1.35), 0.45 + 0.55 * sky_up)
        solid_hdr = (
            albedo * (sun_col * (ndotl * vis)[..., None] * (1.0 / np.pi) + sky_fill * ao[..., None] * 0.22)
            + sun_col * (spec_term * ndotl)[..., None]
            + sss_rgb
        )

        # Multi-scale lagoon waves for sparkling golden glitter path
        is_water_front = water_hit & (water_d < best_d)
        water_pos = ray_orig + jnp.where(water_hit, water_d, 0.0)[..., None] * ray_dir
        wx, wy = water_pos[..., 0], water_pos[..., 1]
        hx_w, hy_w, _ = _ocean_wave_slope_and_jacobian(wx, wy, jnp.ones_like(wx))
        w_nrm = jnp.where(water_nrm[..., 2:3] >= 0.0, water_nrm, -water_nrm) + jnp.stack([hx_w * 0.72, hy_w * 0.72, jnp.zeros_like(hx_w)], axis=-1)
        w_nrm = w_nrm / jnp.maximum(jnp.linalg.norm(w_nrm, axis=-1, keepdims=True), 1e-8)

        refl_w = ray_dir - 2.0 * jnp.sum(ray_dir * w_nrm, axis=-1, keepdims=True) * w_nrm
        refl_w = jnp.where(refl_w[..., 2:3] < 0.015, jnp.concatenate([refl_w[..., :2], jnp.full_like(refl_w[..., 2:3], 0.015)], axis=-1), refl_w)
        refl_w = refl_w / jnp.maximum(jnp.linalg.norm(refl_w, axis=-1, keepdims=True), 1e-8)

        env_lookup_dir = jnp.where(is_water_front[..., None], refl_w, ray_dir)
        hdri_raw = pallas_sample_hdri_env(env_lookup_dir, env_b, sub_h_pad, W, H_env=512, W_env=1024)
        sunset_env = _sunset_sky_radiance(env_lookup_dir, hdri_raw, L_sun)

        # Beer-Lambert shallow coral lagoon under sunset Fresnel reflections
        t_water = jnp.where(best_hit, jnp.maximum(0.0, best_d - water_d), 45.0)
        sigma_t = jnp.array([0.48, 0.16, 0.18], dtype=jnp.float32)[None, None, :]
        transmittance = jnp.exp(-sigma_t * t_water[..., None])
        in_scatter = jnp.array([0.025, 0.065, 0.085], dtype=jnp.float32)[None, None, :] * (1.0 - jnp.exp(-0.20 * t_water[..., None]))
        underwater_rgb = jnp.where(best_hit[..., None], solid_hdr * 1.15 * transmittance + in_scatter, in_scatter * 0.45)

        cos_w = jnp.maximum(0.0, jnp.sum(w_nrm * vdir, axis=-1))
        fresnel_w = jnp.minimum(0.74, 0.04 + 0.76 * ((1.0 - cos_w) ** 3.6))
        w_ndoth = jnp.maximum(0.0, jnp.sum(w_nrm * H_vec, axis=-1))
        # Molten-gold sun glitter path across lagoon ripples
        glitter_core = (w_ndoth ** 650.0) * 7.50
        glitter_sheen = (w_ndoth ** 110.0) * 0.45
        glitter_tail = (w_ndoth ** 28.0) * 0.035
        w_sun_spec = jnp.array([6.80, 2.45, 0.45], dtype=jnp.float32)[None, None, :] * (glitter_core + glitter_sheen + glitter_tail)[..., None]
        water_rgb = underwater_rgb * (1.0 - fresnel_w[..., None]) + sunset_env * 0.68 * fresnel_w[..., None] + w_sun_spec

        surf_rgb = jnp.where(is_water_front[..., None], water_rgb, solid_hdr)
        eff_d = jnp.where(is_water_front, water_d, best_d)
        any_surface = best_hit | is_water_front

        # Warm golden-crimson sunset aerial haze across distant volcanic cliffs
        fog_t = jnp.clip(1.0 - jnp.exp(-eff_d * 0.00018), 0.0, 0.22)[..., None]
        sunset_mist = jnp.array([0.78, 0.26, 0.14], dtype=jnp.float32)[None, None, :]
        surf_fog = surf_rgb * (1.0 - fog_t) + sunset_mist * fog_t

        hdr_final = jnp.where(any_surface[..., None], surf_fog, sunset_env)
        ldr_rgb = agx_filmic_tonemap(hdr_final, exposure=1.08)
        return None, ldr_rgb.reshape(B, sub_h_pad, W, 3)[:, :sub_h, :, :]

      # ---------------------------------------------------------------------
      # Mode 2: Way of Water Underwater Coral Reef Shot (reefCam)
      #   - Beer-Lambert cyan-emerald underwater absorption
      #   - Analytical optical-Jacobian caustic ripples on corals & sand
      #   - 16-step HG-phase volumetric sunbeams (God Rays) w/ 4-tap BVH shadows
      #   - Snell's Window looking up at underside of osOcean
      # ---------------------------------------------------------------------
      sun_uw = jnp.array([2.60, 2.95, 2.85], dtype=jnp.float32)[None, None, :]
      sigma_uw = jnp.array([0.46, 0.095, 0.115], dtype=jnp.float32)[None, None, :]

      # Refracted water-surface caustic ripples projected onto coral & sand surfaces
      sub_depth = jnp.clip(water_z_mean - hit_pos[..., 2], 0.04, 8.0)
      cx_proj = hit_pos[..., 0] + sub_depth * (L_refr[..., 0] / jnp.maximum(L_refr[..., 2], 0.20))
      cy_proj = hit_pos[..., 1] + sub_depth * (L_refr[..., 1] / jnp.maximum(L_refr[..., 2], 0.20))
      _, _, caustic_rgb = _ocean_wave_slope_and_jacobian(cx_proj, cy_proj, jnp.clip(sub_depth * 0.62 + 0.28, 0.28, 1.85))
      # High-contrast caustic webs modulated by depth & BVH coral/rock shadows
      caustic_atten = jnp.exp(-0.18 * sub_depth)[..., None] * vis[..., None] * jnp.clip(ndotl * 1.45 + 0.12, 0.0, 1.0)[..., None]
      caustic_light = caustic_rgb * jnp.array([1.25, 2.40, 2.35], dtype=jnp.float32)[None, None, :] * caustic_atten

      # Underwater coral & rock PBR shading + intense caustic illumination
      H_uw = (L_refr + vdir) / jnp.maximum(jnp.linalg.norm(L_refr + vdir, axis=-1, keepdims=True), 1e-12)
      ndoth_uw = jnp.maximum(0.0, jnp.sum(best_nrm * H_uw, axis=-1))
      ndotv_uw = jnp.maximum(1e-3, jnp.sum(best_nrm * vdir, axis=-1))
      alpha_uw = jnp.clip(eff_rough * eff_rough, 0.06, 0.98)
      a2_uw = alpha_uw * alpha_uw
      denom_uw = ndoth_uw * ndoth_uw * (a2_uw - 1.0) + 1.0
      D_uw = a2_uw / (np.pi * denom_uw * denom_uw + 1e-6)
      spec_uw = jnp.minimum((D_uw * 0.04) / (4.0 * ndotv_uw * jnp.maximum(ndotl, 0.05) + 0.16), 1.8) * vis

      # Deep Way-of-Water cyan-emerald shadow & ambient fill (preserves dramatic contrast)
      uw_up = 0.5 + 0.5 * best_nrm[..., 2]
      uw_amb = (
          uw_up[..., None] * jnp.array([0.04, 0.34, 0.42], dtype=jnp.float32)[None, None, :]
          + (1.0 - uw_up[..., None]) * jnp.array([0.01, 0.08, 0.12], dtype=jnp.float32)[None, None, :]
      )
      sun_surf_trans = jnp.exp(-sigma_uw * (sub_depth / jnp.maximum(L_refr[..., 2], 0.25))[..., None])
      coral_bounce = jnp.where((best_tag == 5)[..., None], 1.45, 0.82)
      solid_uw_hdr = (
          albedo * coral_bounce * (
              sun_uw * sun_surf_trans * ((ndotl * vis)[..., None] * (0.62 / np.pi))
              + caustic_light * 0.95
              + uw_amb * 0.36
          )
          + sun_uw * sun_surf_trans * (spec_uw * ndotl)[..., None]
      )

      # Underside of osOcean (Snell's Window + Total Internal Reflection!)
      is_water_ceiling = water_hit & (water_d < best_d)
      water_pos = ray_orig + jnp.where(water_hit, water_d, 0.0)[..., None] * ray_dir
      wx, wy = water_pos[..., 0], water_pos[..., 1]
      hx_w, hy_w, _ = _ocean_wave_slope_and_jacobian(wx, wy, jnp.ones_like(wx))
      w_nrm_up = jnp.where(water_nrm[..., 2:3] >= 0.0, water_nrm, -water_nrm) + jnp.stack([hx_w * 1.15, hy_w * 1.15, jnp.zeros_like(hx_w)], axis=-1)
      w_nrm_up = w_nrm_up / jnp.maximum(jnp.linalg.norm(w_nrm_up, axis=-1, keepdims=True), 1e-8)

      cos_i = jnp.maximum(0.0, jnp.sum(ray_dir * w_nrm_up, axis=-1))
      eta_w2a = 1.333
      sin2_t = (eta_w2a * eta_w2a) * jnp.maximum(0.0, 1.0 - cos_i * cos_i)
      in_snell = sin2_t < 0.999
      cos_t = jnp.sqrt(jnp.maximum(1.0 - sin2_t, 0.0))
      refr_air_dir = eta_w2a * ray_dir + (cos_t - eta_w2a * cos_i)[..., None] * w_nrm_up
      refr_air_dir = refr_air_dir / jnp.maximum(jnp.linalg.norm(refr_air_dir, axis=-1, keepdims=True), 1e-8)

      hdri_snell = pallas_sample_hdri_env(refr_air_dir, env_b, sub_h_pad, W, H_env=512, W_env=1024)
      sun_glare = (jnp.maximum(0.0, jnp.sum(refr_air_dir * L_sun, axis=-1)) ** 24.0)[..., None] * jnp.array([12.0, 11.5, 9.8], dtype=jnp.float32)[None, None, :]
      snell_sky = hdri_snell * 1.65 + sun_glare

      # Total internal reflection outside Snell's cone: rich rippled deep teal-silver ceiling
      tir_col = (
          jnp.array([0.008, 0.14, 0.19], dtype=jnp.float32)[None, None, :]
          + jnp.array([0.05, 0.36, 0.40], dtype=jnp.float32)[None, None, :] * jnp.clip(0.5 + 4.5 * hx_w, 0.0, 1.0)[..., None]
      )
      fresnel_uw = jnp.where(in_snell, jnp.minimum(0.92, 0.03 + 0.95 * ((1.0 - cos_i) ** 4.5)), 1.0)
      water_ceiling_hdr = snell_sky * (1.0 - fresnel_uw[..., None]) + tir_col * fresnel_uw[..., None]

      # Distance of primary underwater ray segment
      t_water_ray = jnp.where(is_water_ceiling, water_d, jnp.where(best_hit, best_d, 48.0))
      t_march_max = jnp.minimum(t_water_ray, 26.0)

      # Surface radiance before underwater absorption + volumetric God Rays
      bg_abyss = jnp.array([0.004, 0.08, 0.12], dtype=jnp.float32)[None, None, :]
      endpoint_hdr = jnp.where(is_water_ceiling[..., None], water_ceiling_hdr, jnp.where(best_hit[..., None], solid_uw_hdr, bg_abyss))
      trans_cam = jnp.exp(-sigma_uw * t_water_ray[..., None])

      # Deep rich cyan-emerald underwater in-scatter (keeps shadows deep so God Rays pop!)
      ambient_in = jnp.array([0.003, 0.055, 0.075], dtype=jnp.float32)[None, None, :] * (1.0 - jnp.exp(-0.14 * t_water_ray[..., None]))

      # ---------------------------------------------------------------------
      # Volumetric Underwater Sunbeams (God Rays) via 16-step HG Phase March
      # + 4-Tap Real BVH/TLAS Volumetric Shadow Shafts (zero host callbacks!)
      # ---------------------------------------------------------------------
      px_idx = jnp.arange(W, dtype=jnp.float32)[None, :]
      py_idx = (
          bidx.astype(jnp.float32) * float(band_h)
          + sb_idx.astype(jnp.float32) * float(sub_h)
          + jnp.arange(sub_h_pad, dtype=jnp.float32)[:, None]
      )
      jitter = jnp.mod(jnp.sin(px_idx * 12.9898 + py_idx * 78.233) * 43758.5453, 1.0).reshape(1, sub_h_pad * W)

      tap_fracs = (0.14, 0.36, 0.62, 0.86)
      tap_vis_list = []
      for tf in tap_fracs:
        t_s = t_march_max * tf
        p_s = ray_orig + t_s[..., None] * ray_dir
        above_w = p_s[..., 2] >= (water_z_mean - 0.02)
        if use_shadows:
          lim_s = jnp.where((t_march_max > 0.25) & (~above_w), 120.0, 0.0)
          occ_s = _trace_shadow_occ(tlas_precomps, u_da, u_oxp, u_oxm, i_da, p_s, L_refr, lim_s)
          tap_vis_list.append(jnp.where(occ_s, 0.06, 1.0))
        else:
          tap_vis_list.append(jnp.ones((B, R), dtype=jnp.float32))
      tap_vis = jnp.stack(tap_vis_list, axis=0)

      cos_hg = jnp.sum(ray_dir * L_refr, axis=-1)
      g_hg = 0.74
      hg_phase = (1.0 - g_hg * g_hg) / (4.0 * np.pi * jnp.power(jnp.maximum(1.0 + g_hg * g_hg - 2.0 * g_hg * cos_hg, 1e-3), 1.5))
      phase_eff = 0.24 + 2.40 * hg_phase

      dt = t_march_max / float(vol_steps)

      def _vol_step(k_idx, acc_god_rays):
        frac_k = (k_idx.astype(jnp.float32) + jitter) / float(vol_steps)
        t_k = frac_k * t_march_max
        p_k = ray_orig + t_k[..., None] * ray_dir
        d_surf_k = jnp.maximum(0.02, water_z_mean - p_k[..., 2])
        valid_k = (p_k[..., 2] < water_z_mean).astype(jnp.float32)

        idx_f = jnp.clip((frac_k - 0.14) / 0.24, 0.0, 2.999)
        i0 = jnp.floor(idx_f).astype(jnp.int32)
        w1 = idx_f - i0.astype(jnp.float32)
        v_k = jnp.zeros((B, R), dtype=jnp.float32)
        for ti in range(vol_shadow_taps - 1):
          m_t = (i0 == ti).astype(jnp.float32)
          v_k = v_k + m_t * ((1.0 - w1) * tap_vis[ti] + w1 * tap_vis[ti + 1])

        # High-contrast surface wave focusing carves crisp diagonal volumetric light curtains!
        wx_k = p_k[..., 0] + d_surf_k * (L_refr[..., 0] / jnp.maximum(L_refr[..., 2], 0.20))
        wy_k = p_k[..., 1] + d_surf_k * (L_refr[..., 1] / jnp.maximum(L_refr[..., 2], 0.20))
        _, _, caus_k = _ocean_wave_slope_and_jacobian(wx_k, wy_k, jnp.clip(d_surf_k * 0.62 + 0.28, 0.28, 1.85))
        beam_mod = 0.08 + 0.92 * jnp.power(jnp.clip(caus_k, 0.0, 4.0), 1.15)

        in_att = jnp.exp(-sigma_uw * (d_surf_k / jnp.maximum(L_refr[..., 2], 0.25))[..., None])
        out_att = jnp.exp(-sigma_uw * t_k[..., None])
        scatter_col = jnp.array([0.14, 0.96, 0.92], dtype=jnp.float32)[None, None, :]
        return acc_god_rays + (scatter_col * beam_mod * in_att * out_att * (v_k * phase_eff * valid_k * dt)[..., None])

      god_rays = jax.lax.fori_loop(0, vol_steps, _vol_step, jnp.zeros((B, R, 3), dtype=jnp.float32))

      hdr_final = endpoint_hdr * trans_cam + ambient_in + god_rays * 0.78
      ldr_rgb = agx_filmic_tonemap(hdr_final, exposure=1.18)
      return None, ldr_rgb.reshape(B, sub_h_pad, W, 3)[:, :sub_h, :, :]

    _, sub_stack = jax.lax.scan(_render_subband, None, jnp.arange(n_subbands, dtype=jnp.int32))
    return sub_stack.transpose(1, 0, 2, 3, 4).reshape(1, band_h, W, 3)

  sharded = shard_map(
      _shard_fn,
      mesh=mesh,
      in_specs=(P(), P(), P(), P(), P(), P(), P(), P(), P(), P(), P(), P()),
      out_specs=P(None, "spatial", None, None),
      check_rep=False,
  )
  fn = jax.jit(sharded)
  base_args = (
      u_devs, compiled["obj_xpos"], compiled["obj_xmat"],
      w_devs,
      i_devs, compiled["inst_xpos"], compiled["inst_xmat"], inst_inv_xmats_t,
  )
  return fn, base_args, rep


def make_sunset_reef_showcase(
    sunset_png: str,
    reef_png: str,
    out_png: str,
    stats_sunset: Dict[str, Any],
    stats_reef: Dict[str, Any],
):
  """Compose razor-sharp 2-panel Native 4K showcase (3840x2160 each, 1:1 native 4K pixels)."""
  im1 = Image.open(sunset_png).convert("RGB")
  im2 = Image.open(reef_png).convert("RGB")
  w, h = im1.size
  assert im2.size == (w, h), (im1.size, im2.size)

  pad = 24
  top_h = 84
  hdr_h = 54
  cw = w + 2 * pad
  ch = top_h + 2 * (hdr_h + h) + 3 * pad
  canvas = Image.new("RGB", (cw, ch), (10, 14, 20))
  dr = ImageDraw.Draw(canvas)
  try:
    font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)
    font_sub = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    font_hdr = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
  except Exception:
    font_title = font_sub = font_hdr = ImageFont.load_default()

  dr.text(
      (pad, 14),
      "Disney Moana (OpenUSD v2.1, 22.30B Tris) on 4-Chip TPU v5p — Golden-Hour Sunset & Way-of-Water Coral Reef @ Native 4K",
      fill=(245, 248, 255),
      font=font_title,
  )
  dr.text(
      (pad, 52),
      f"Zero-Gather Pallas Mosaic  |  Sunset: {stats_sunset['ms']:.1f} ms ({stats_sunset['mrays_per_s']:.2f} Mrays/s)  |  "
      f"Underwater Reef: {stats_reef['ms']:.1f} ms ({stats_reef['mrays_per_s']:.2f} Mrays/s)  |  "
      f"host_callbacks=0, gather_ops=0, send_recv_ops=0",
      fill=(120, 235, 210),
      font=font_sub,
  )

  y1 = top_h + pad
  dr.text(
      (pad, y1 + 12),
      "1. Motunui Golden-Hour Sunset (shotCam, 3840x2160) — 10 deg Low Sun, Crimson/Amber Horizon Glow, Long Palm Shadows & Ocean Glitter Path",
      fill=(255, 215, 145),
      font=font_hdr,
  )
  canvas.paste(im1, (pad, y1 + hdr_h))

  y2 = y1 + hdr_h + h + pad
  dr.text(
      (pad, y2 + 12),
      "2. Way of Water Underwater Coral Reef (reefCam, 3840x2160) — Real isCoral Reef, Beer-Lambert Cyan-Emerald, HG God Rays & Jacobian Caustics",
      fill=(125, 240, 235),
      font=font_hdr,
  )
  canvas.paste(im2, (pad, y2 + hdr_h))

  os.makedirs(os.path.dirname(out_png), exist_ok=True)
  canvas.save(out_png)
  print(f"Saved 2-panel Native 4K showcase to {out_png} ({canvas.size})", flush=True)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--mode", choices=["preview_1k", "render_4k"], default="render_4k")
  ap.add_argument("--stage", default="/dev/shm/zaychik-moana/island/usd/island.usda")
  ap.add_argument("--exr", default="/dev/shm/zaychik-moana/island/textures/islandsun.exr")
  ap.add_argument("--cache-dir", default="/dev/shm/zaychik-tpu-mjx-w2/moana_bvh_cache")
  ap.add_argument("--out-dir", default="/dev/shm/zaychik-tpu-mjx-w2/out")
  ap.add_argument("--sunset-az-deg", type=float, default=8.0)
  ap.add_argument("--sunset-el-deg", type=float, default=10.0)
  ap.add_argument("--reef-cam", choices=["shallow", "mid", "fg", "deep"], default="fg")
  args = ap.parse_args()

  os.makedirs(args.out_dir, exist_ok=True)
  devs = jax.devices()
  assert len(devs) == 4, f"Expected 4 TPU chips, got {len(devs)}"
  mesh = Mesh(np.array(devs), ("spatial",))
  rep = NamedSharding(mesh, P())

  compiled = _load_compiled_cache(args.cache_dir, devs, 2160, 3840)
  assert compiled is not None, f"Missing Moana BVH cache at {args.cache_dir}"

  env_np = load_islandsun_hdri_polished(args.exr, H_env=512, W_env=1024)
  env_bf16 = jax.device_put(jnp.asarray(env_np, dtype=jnp.bfloat16), rep)

  cams = extract_all_moana_cameras(args.stage, target_aspect=16.0 / 9.0)
  shot_cam = cams["shotCam"]

  # Candidate submerged reef cameras inside the Moana lagoon (water surface Z = -0.266 m)
  reef_candidates = {
      "fg": make_lookat_camera("reefCam_fg", (-23.0, -33.8, -0.92), (-10.0, -26.8, -0.32), fovy_deg=60.0),
      "shallow": make_lookat_camera("reefCam_shallow", (-15.0, -31.5, -0.96), (-5.5, -26.5, -0.20), fovy_deg=60.0),
      "mid": make_lookat_camera("reefCam_mid", (-13.5, -36.5, -1.15), (-5.0, -28.0, -0.25), fovy_deg=60.0),
      "deep": make_lookat_camera("reefCam_deep", (-2.0, -45.5, -1.75), (-6.0, -31.0, -0.35), fovy_deg=60.0),
  }

  az_r = math.radians(args.sunset_az_deg)
  el_r = math.radians(args.sunset_el_deg)
  sunset_sun = np.array([math.cos(el_r) * math.cos(az_r), math.cos(el_r) * math.sin(az_r), math.sin(el_r)], dtype=np.float32)

  # High tropical sun for bright underwater caustic webs & diagonal God Rays
  uw_sun = np.array([0.42, 0.52, 0.74], dtype=np.float32)

  def _run_cam(fn, base_args, cam_dict):
    cxp = jax.device_put(cam_dict["cam_xpos"], rep)
    cxm = jax.device_put(cam_dict["cam_xmat"], rep)
    thf = jax.device_put(cam_dict["tan_half_fovy"], rep)
    return fn(*base_args, cxp, cxm, thf, env_bf16)

  def _audit_hlo(fn, base_args, cam_dict, tag: str) -> Dict[str, Any]:
    cxp = jax.device_put(cam_dict["cam_xpos"], rep)
    cxm = jax.device_put(cam_dict["cam_xmat"], rep)
    thf = jax.device_put(cam_dict["tan_half_fovy"], rep)
    htxt = fn.lower(*base_args, cxp, cxm, thf, env_bf16).as_text()
    hlo = inspect_hlo_ops(htxt)
    hlo["send_recv_ops"] = len(re.findall(r"\bsend\(|\brecv\(", htxt, re.I))
    print(f"[{tag}] HLO audit:", hlo, flush=True)
    assert hlo["host_callbacks"] == 0 and hlo["gather_ops"] == 0 and hlo["send_recv_ops"] == 0, hlo
    return hlo

  if args.mode == "preview_1k":
    H_p, W_p = 720, 1280
    fn_sun_p, ba_sun_p, _ = make_moana_wave6_renderer(
        compiled, env_bf16, devs, H_p, W_p, n_subbands=1, mode="sunset", sun_dir_np=sunset_sun, use_shadows=True,
    )
    rgb_sun_p = _run_cam(fn_sun_p, ba_sun_p, shot_cam)
    jax.block_until_ready(rgb_sun_p)
    p_sun = os.path.join(args.out_dir, "preview_sunset_1k.png")
    Image.fromarray(np.clip(np.asarray(rgb_sun_p[0]) * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(p_sun)
    print(f"Saved sunset preview -> {p_sun}", flush=True)

    fn_uw_p, ba_uw_p, _ = make_moana_wave6_renderer(
        compiled, env_bf16, devs, H_p, W_p, n_subbands=1, mode="underwater_reef", sun_dir_np=uw_sun, use_shadows=True,
    )
    for kname, rcam in reef_candidates.items():
      rgb_uw_p = _run_cam(fn_uw_p, ba_uw_p, rcam)
      jax.block_until_ready(rgb_uw_p)
      p_uw = os.path.join(args.out_dir, f"preview_reef_{kname}_1k.png")
      Image.fromarray(np.clip(np.asarray(rgb_uw_p[0]) * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(p_uw)
      print(f"Saved reef preview -> {p_uw}", flush=True)
    return

  # -------------------------------------------------------------------------
  # Full Native 4K (3840x2160) Production Benchmark & 2-Panel Showcase
  # -------------------------------------------------------------------------
  H_4k, W_4k = 2160, 3840

  # 1. Motunui Golden-Hour Sunset @ Native 4K
  fn_sun, ba_sun, _ = make_moana_wave6_renderer(
      compiled, env_bf16, devs, H_4k, W_4k, n_subbands=2, mode="sunset", sun_dir_np=sunset_sun, use_shadows=True,
  )
  hlo_sun = _audit_hlo(fn_sun, ba_sun, shot_cam, "Sunset_4K")
  t0 = time.perf_counter()
  rgb_sun = _run_cam(fn_sun, ba_sun, shot_cam)
  jax.block_until_ready(rgb_sun)
  xla_sun_s = time.perf_counter() - t0
  sunset_4k_path = os.path.join(args.out_dir, "v3_w2_moana_sunset_4k.png")
  Image.fromarray(np.clip(np.asarray(rgb_sun[0]) * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(sunset_4k_path)

  n_steps = 3
  t1 = time.perf_counter()
  for _ in range(n_steps):
    rgb_tmp = _run_cam(fn_sun, ba_sun, shot_cam)
  jax.block_until_ready(rgb_tmp)
  ms_sun = (time.perf_counter() - t1) * 1e3 / n_steps
  stats_sunset = {
      "camera": "shotCam",
      "res": f"{H_4k}x{W_4k}",
      "pixels": H_4k * W_4k,
      "n_chips": len(devs),
      "sun_az_deg": args.sunset_az_deg,
      "sun_el_deg": args.sunset_el_deg,
      "xla_compile_s": round(xla_sun_s, 2),
      "ms": round(ms_sun, 2),
      "fps": round(1e3 / ms_sun, 3),
      "mrays_per_s": round((H_4k * W_4k * 2.0) / (ms_sun * 1e-3) / 1e6, 2),
      "hlo": hlo_sun,
  }
  print(f"SUNSET 4K: {ms_sun:.2f} ms ({stats_sunset['mrays_per_s']} Mrays/s)", flush=True)

  # 2. Way-of-Water Underwater Coral Reef @ Native 4K
  reef_cam = reef_candidates[args.reef_cam]
  fn_reef, ba_reef, _ = make_moana_wave6_renderer(
      compiled, env_bf16, devs, H_4k, W_4k, n_subbands=2, mode="underwater_reef", sun_dir_np=uw_sun, use_shadows=True,
  )
  hlo_reef = _audit_hlo(fn_reef, ba_reef, reef_cam, "UnderwaterReef_4K")
  t2 = time.perf_counter()
  rgb_reef = _run_cam(fn_reef, ba_reef, reef_cam)
  jax.block_until_ready(rgb_reef)
  xla_reef_s = time.perf_counter() - t2
  reef_4k_path = os.path.join(args.out_dir, "v3_w2_moana_reef_4k.png")
  Image.fromarray(np.clip(np.asarray(rgb_reef[0]) * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(reef_4k_path)

  t3 = time.perf_counter()
  for _ in range(n_steps):
    rgb_rtmp = _run_cam(fn_reef, ba_reef, reef_cam)
  jax.block_until_ready(rgb_rtmp)
  ms_reef = (time.perf_counter() - t3) * 1e3 / n_steps
  # Primary + shadow + 4 volumetric shadow taps = 6 ray passes/px
  stats_reef = {
      "camera": f"reefCam_{args.reef_cam}",
      "cam_pos": [round(float(x), 3) for x in reef_cam["cam_xpos"][0]],
      "res": f"{H_4k}x{W_4k}",
      "pixels": H_4k * W_4k,
      "n_chips": len(devs),
      "vol_march_steps": 16,
      "vol_shadow_taps": 4,
      "xla_compile_s": round(xla_reef_s, 2),
      "ms": round(ms_reef, 2),
      "fps": round(1e3 / ms_reef, 3),
      "mrays_per_s": round((H_4k * W_4k * 6.0) / (ms_reef * 1e-3) / 1e6, 2),
      "hlo": hlo_reef,
  }
  print(f"REEF 4K: {ms_reef:.2f} ms ({stats_reef['mrays_per_s']} Mrays/s)", flush=True)

  showcase_path = os.path.join(args.out_dir, "v3_w2_moana_sunset_reef_4k.png")
  make_sunset_reef_showcase(sunset_4k_path, reef_4k_path, showcase_path, stats_sunset, stats_reef)

  metrics = {
      "name": "Wave6_Moana_GoldenHour_Sunset_And_Underwater_Coral_Reef_4K",
      "stage_path": args.stage,
      "hbm_gb": round(compiled["hbm_bytes"] / (1024 ** 3), 3),
      "unique_tris": compiled["unique_tris"],
      "inst_blas_tris": compiled["inst_blas_tris"],
      "inst_effective_tris": compiled["inst_effective_tris"],
      "sunset_4k": stats_sunset,
      "underwater_reef_4k": stats_reef,
      "artifacts": {
          "showcase_4k_png": "out/figures/v3_w2_moana_sunset_reef_4k.png",
      },
  }
  json_path = os.path.join(args.out_dir, "v3_w2_wave6_moana_sunset_reef.json")
  with open(json_path, "w") as f:
    json.dump(metrics, f, indent=2)
  print(f"Saved metrics to {json_path}", flush=True)


if __name__ == "__main__":
  main()
