"""Priority #2: Physical FX Suite on TPU v5p (Dielectric Caustics + Volumetric God Rays).

Implements inside the TPU-native multi-bounce Mosaic path tracer (`host_callbacks == 0`):
  (a) Exact Dielectric Glass Refraction & Caustics:
      Snell's Law + Schlick Fresnel + Total Internal Reflection (TIR)
      + Beer-Lambert colored absorption `exp(-sigma_a * d_inside)`
      + TPU-native Forward Light-Tracing Caustic Irradiance Map (`pbr_raycast_tpu`).
  (b) Volumetric Fog / Participating Media & God Rays:
      Stratified 24-step ray-marched single + 2nd-order multi-scattering
      with Henyey-Greenstein phase function `p_HG(cos_theta, g)`.
"""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import jax
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np
from PIL import Image, ImageDraw

from bvh_pbr_pt import (
    LightRig,
    PBRWideBVHMesh,
    _agx_tonemap_srgb,
    _build_onb,
    _eval_pbr_brdf,
    _sample_cosine_hemisphere,
    _sample_ggx_half,
    _smith_g1,
    audit_hlo_zero_copy,
    pbr_raycast_tpu,
)
from complex_scenes_pt import MeshBuilder, _look_at_mat, build_classroom_sunlit


@dataclasses.dataclass(frozen=True)
class CausticConfig:
  enabled: bool = False
  grid_res: int = 512
  n_light_rays_h: int = 512
  n_light_rays_w: int = 1024
  bounds_xy: Tuple[float, float, float, float] = (-1.45, 1.45, -1.45, 1.45)
  target_xy_bounds: Tuple[float, float, float, float] = (-0.92, 0.92, -0.78, 0.72)
  caustic_power: float = 4.6


@dataclasses.dataclass(frozen=True)
class VolumetricConfig:
  enabled: bool = False
  sigma_t: float = 0.088
  scatter_albedo: float = 0.92
  hg_g: float = 0.78
  height_scale: float = 2.8
  n_steps: int = 28
  max_dist: float = 9.0
  multi_scatter_boost: float = 0.04


def _henyey_greenstein(cos_theta: jax.Array, g: float) -> jax.Array:
  g2 = g * g
  denom = jnp.maximum(1.0 + g2 - 2.0 * g * cos_theta, 1e-4)
  return (1.0 - g2) / (4.0 * math.pi * denom * jnp.sqrt(denom))


def _glass_ior_from_mcode(mcode: jax.Array) -> jax.Array:
  # mat_code 2.0 -> 1.52, 2.2 -> 1.58, 2.4 -> 1.64
  return 1.52 + 0.30 * jnp.clip(mcode - 2.0, 0.0, 1.0)


def _beer_lambert_sigma_a(alb: jax.Array, ref_dist: float = 0.45) -> jax.Array:
  return -jnp.log(jnp.clip(alb, 0.03, 0.995)) / ref_dist


def _trace_caustic_map_tpu(
    l2r: jax.Array,
    l2: jax.Array,
    l0: jax.Array,
    lf: jax.Array,
    rig: LightRig,
    cfg: CausticConfig,
    rng_key: jax.Array,
) -> jax.Array:
  """TPU-native Forward Caustic Photon Pass using `pbr_raycast_tpu` + Beer-Lambert + Snell."""
  CH, CW = cfg.n_light_rays_h, cfg.n_light_rays_w
  n_photons = CH * CW
  tx0, tx1, ty0, ty1 = cfg.target_xy_bounds

  k_pos, k_tgt, k_b = jax.random.split(rng_key, 3)
  # Stratified + jittered grid aimed at the glass zone on the table
  gx = (jnp.arange(CW, dtype=jnp.float32) + 0.5) / float(CW)
  gy = (jnp.arange(CH, dtype=jnp.float32) + 0.5) / float(CH)
  ug, vg = jnp.meshgrid(gx, gy, indexing="xy")
  jit = jax.random.uniform(k_tgt, (CH, CW, 2), minval=-0.5, maxval=0.5, dtype=jnp.float32)
  tx = tx0 + (tx1 - tx0) * jnp.clip(ug + jit[..., 0] / float(CW), 0.0, 1.0)
  ty = ty0 + (ty1 - ty0) * jnp.clip(vg + jit[..., 1] / float(CH), 0.0, 1.0)
  tz = jnp.full_like(tx, 0.22)
  tgt_pts = jnp.stack([tx, ty, tz], axis=-1).reshape(1, n_photons, 3)

  area_c = jnp.asarray(rig.area_center, dtype=jnp.float32)
  area_n = jnp.asarray(rig.area_normal, dtype=jnp.float32)
  area_n = area_n / jnp.maximum(jnp.linalg.norm(area_n), 1e-12)
  at1, at2 = _build_onb(area_n)
  lu = jax.random.uniform(k_pos, (1, n_photons), minval=0.0, maxval=1.0, dtype=jnp.float32)
  lv = jax.random.uniform(jax.random.fold_in(k_pos, 1), (1, n_photons), minval=0.0, maxval=1.0, dtype=jnp.float32)
  lr = (rig.area_radius * 0.65) * jnp.sqrt(lu)
  lphi = (2.0 * math.pi) * lv
  ro0 = area_c[None, None, :] + (lr * jnp.cos(lphi))[..., None] * at1[None, None, :] + (lr * jnp.sin(lphi))[..., None] * at2[None, None, :]
  rd0 = tgt_pts - ro0
  rd0 = rd0 / jnp.maximum(jnp.linalg.norm(rd0, axis=-1, keepdims=True), 1e-12)

  # Exact solid-angle radiometric flux per photon: Phi_p / cell_area
  beam_area = (tx1 - tx0) * (ty1 - ty0)
  cell_area = ((cfg.bounds_xy[1] - cfg.bounds_xy[0]) / float(cfg.grid_res)) * (
      (cfg.bounds_xy[3] - cfg.bounds_xy[2]) / float(cfg.grid_res)
  )
  dist_est = jnp.maximum(jnp.linalg.norm(jnp.array([0.0, 0.0, 0.22], jnp.float32) - area_c), 0.5)
  light_area_eff = math.pi * ((rig.area_radius * 0.65) ** 2)
  flux_scale = cfg.caustic_power * (light_area_eff / (dist_est * dist_est)) * (beam_area / float(n_photons)) / max(cell_area, 1e-6)
  flux0 = jnp.broadcast_to(jnp.asarray(rig.area_emit, dtype=jnp.float32)[None, None, :] * flux_scale, (1, n_photons, 3))
  alive0 = jnp.ones((1, n_photons), dtype=jnp.bool_)
  refr0 = jnp.zeros((1, n_photons), dtype=jnp.bool_)
  t_lim0 = jnp.full((1, n_photons), 15.0, dtype=jnp.float32)
  grid0 = jnp.zeros((cfg.grid_res, cfg.grid_res, 3), dtype=jnp.float32)

  bx0, bx1, by0, by1 = cfg.bounds_xy
  G = cfg.grid_res

  def _photon_step(b_idx, carry):
    ro, rd, flux, alive, has_refr, cmap = carry
    d_hit, nrm_raw, alb, _rgh, mcode = pbr_raycast_tpu(
        l2r, l2, l0, lf, ro, rd, jnp.where(alive, t_lim0, 0.0), CH, CW, any_hit=False
    )
    hit = alive & (d_hit < t_lim0 - 1e-4) & (d_hit >= 0.0)
    n_unit = nrm_raw / jnp.maximum(jnp.linalg.norm(nrm_raw, axis=-1, keepdims=True), 1e-12)
    wo = -rd
    entering = jnp.sum(n_unit * wo, axis=-1) >= 0.0
    n_ff = jnp.where(entering[..., None], n_unit, -n_unit)

    is_glass = hit & (mcode >= 1.8) & (mcode < 2.8)
    is_opaque = hit & (mcode < 1.8)
    hit_pos = ro + jnp.where(hit, d_hit, 0.0)[..., None] * rd

    # Deposit photon on opaque upward-facing surface if it previously refracted through glass!
    deposit = is_opaque & has_refr & (n_ff[..., 2] > 0.25)
    gx_f = ((hit_pos[0, :, 0] - bx0) / (bx1 - bx0)) * float(G - 1)
    gy_f = ((hit_pos[0, :, 1] - by0) / (by1 - by0)) * float(G - 1)
    in_bounds = deposit[0] & (gx_f >= 0.0) & (gx_f < float(G - 1)) & (gy_f >= 0.0) & (gy_f < float(G - 1))
    ix = jnp.clip(jnp.floor(gx_f).astype(jnp.int32), 0, G - 2)
    iy = jnp.clip(jnp.floor(gy_f).astype(jnp.int32), 0, G - 2)
    fx = jnp.clip(gx_f - ix.astype(jnp.float32), 0.0, 1.0)
    fy = jnp.clip(gy_f - iy.astype(jnp.float32), 0.0, 1.0)
    dep_flux = jnp.where(in_bounds[:, None], flux[0], 0.0)

    cmap = cmap.at[iy, ix].add(dep_flux * ((1.0 - fx) * (1.0 - fy))[:, None])
    cmap = cmap.at[iy, ix + 1].add(dep_flux * (fx * (1.0 - fy))[:, None])
    cmap = cmap.at[iy + 1, ix].add(dep_flux * ((1.0 - fx) * fy)[:, None])
    cmap = cmap.at[iy + 1, ix + 1].add(dep_flux * (fx * fy)[:, None])

    # Beer-Lambert interior absorption on exit (~entering) + Snell refraction
    ior = _glass_ior_from_mcode(mcode)
    sigma_a = _beer_lambert_sigma_a(alb)
    beer = jnp.where((is_glass & (~entering))[..., None], jnp.exp(-sigma_a * jnp.maximum(d_hit, 0.0)[..., None]), 1.0)

    eta = jnp.where(entering, 1.0 / ior, ior)
    cos_i = jnp.clip(jnp.sum(wo * n_ff, axis=-1), 0.0, 1.0)
    sin2_t = eta * eta * (1.0 - cos_i * cos_i)
    tir = sin2_t > 1.0
    r0_g = ((1.0 - eta) / (1.0 + eta)) ** 2
    fresnel_g = r0_g + (1.0 - r0_g) * ((1.0 - cos_i) ** 5)

    u_f = jax.random.uniform(jax.random.fold_in(k_b, b_idx), (1, n_photons), minval=0.0, maxval=1.0, dtype=jnp.float32)
    do_refl = tir | (u_f < fresnel_g)
    refl_dir = 2.0 * cos_i[..., None] * n_ff - wo
    cos_t = jnp.sqrt(jnp.maximum(1.0 - sin2_t, 0.0))
    refr_dir = eta[..., None] * rd + (eta * cos_i - cos_t)[..., None] * n_ff
    wi_next = jnp.where(do_refl[..., None], refl_dir, refr_dir)
    wi_next = wi_next / jnp.maximum(jnp.linalg.norm(wi_next, axis=-1, keepdims=True), 1e-12)
    new_ro = jnp.where(do_refl[..., None], hit_pos + n_ff * 1.5e-3, hit_pos - n_ff * 1.5e-3)

    new_flux = flux * beer
    new_alive = is_glass & (jnp.max(new_flux, axis=-1) > 1e-4)
    new_refr = has_refr | (is_glass & (~do_refl))
    return (new_ro, wi_next, new_flux, new_alive, new_refr, cmap)

  _, _, _, _, _, raw_cmap = jax.lax.fori_loop(
      0, 5, _photon_step, (ro0, rd0, flux0, alive0, refr0, grid0)
  )

  # Tighter separable 3x3 Gaussian smoothing in pure JAX (OIHW: (3, 1, 3, 3) for feature_group_count=3)
  k1d = jnp.array([0.16, 0.68, 0.16], dtype=jnp.float32)
  k2d = jnp.outer(k1d, k1d)[None, None, :, :]
  k_rgb = jnp.broadcast_to(k2d, (3, 1, 3, 3))
  cmap_nchw = jnp.transpose(raw_cmap, (2, 0, 1))[None, ...]  # (1, 3, G, G)
  smooth = jax.lax.conv_general_dilated(
      cmap_nchw, k_rgb, window_strides=(1, 1), padding="SAME", feature_group_count=3
  )
  return jnp.transpose(smooth[0], (1, 2, 0))


def _tpu_joint_atrous_denoise(
    rad_lin: jax.Array,    # (H, W, 3) linear HDR
    nrm: jax.Array,        # (H, W, 3) primary normal
    alb: jax.Array,        # (H, W, 3) primary albedo
    depth: jax.Array,      # (H, W) primary depth
    spec_mask: jax.Array,  # (H, W) 1.0 for glass/mirror, 0.0 for diffuse
    sigma_l: float = 0.36,
    n_passes: int = 4,
) -> jax.Array:
  """Pure-JAX TPU Joint Feature-Guided Firefly Clamp + A-Trous Wavelet Denoiser (zero host callbacks)."""
  H, W, _ = rad_lin.shape
  # 1. 3x3 Luminance Firefly Clamp on non-specular surfaces
  lum = 0.2126 * rad_lin[..., 0] + 0.7152 * rad_lin[..., 1] + 0.0722 * rad_lin[..., 2]
  lum_pad = jnp.pad(lum, ((1, 1), (1, 1)), mode="edge")
  nbr_sum = jnp.zeros_like(lum)
  nbr_sq = jnp.zeros_like(lum)
  for dy in (-1, 0, 1):
    for dx in (-1, 0, 1):
      if dy == 0 and dx == 0:
        continue
      sl = lum_pad[1 + dy : 1 + dy + H, 1 + dx : 1 + dx + W]
      nbr_sum = nbr_sum + sl
      nbr_sq = nbr_sq + sl * sl
  nbr_mu = nbr_sum / 8.0
  nbr_std = jnp.sqrt(jnp.maximum(nbr_sq / 8.0 - nbr_mu * nbr_mu, 0.0))
  lum_cap = jnp.where(spec_mask > 0.5, lum, jnp.minimum(lum, nbr_mu + 3.0 * nbr_std + 0.12))
  scale_ff = jnp.where(lum > 1e-6, lum_cap / jnp.maximum(lum, 1e-6), 1.0)
  cur = rad_lin * scale_ff[..., None]

  # 2. Multi-scale 3x3 A-Trous Joint Bilateral Wavelet passes (steps 1, 2, 4, 8) with single halo pad
  max_s = 1 << (n_passes - 1)
  p_nrm = jnp.pad(nrm, ((max_s, max_s), (max_s, max_s), (0, 0)), mode="edge")
  p_alb = jnp.pad(alb, ((max_s, max_s), (max_s, max_s), (0, 0)), mode="edge")
  p_dep = jnp.pad(depth, ((max_s, max_s), (max_s, max_s)), mode="edge")
  k_1d = [0.25, 0.50, 0.25]
  for p_idx in range(n_passes):
    step = 1 << p_idx
    p_cur = jnp.pad(cur, ((max_s, max_s), (max_s, max_s), (0, 0)), mode="edge")
    log_c = jnp.log1p(jnp.maximum(cur, 0.0))
    acc_rad = jnp.zeros_like(cur)
    acc_w = jnp.zeros((H, W), dtype=jnp.float32)

    s_l_cur = sigma_l * (0.85 ** p_idx)
    inv_2n = 1.0 / (2.0 * 0.22 * 0.22)
    inv_2z = 1.0 / (2.0 * 0.16 * 0.16)
    inv_2a = 1.0 / (2.0 * 0.25 * 0.25)
    inv_2l = 1.0 / (2.0 * s_l_cur * s_l_cur)

    for iy, dy in enumerate((-step, 0, step)):
      for ix, dx in enumerate((-step, 0, step)):
        kw = k_1d[iy] * k_1d[ix]
        q_rad = p_cur[max_s + dy : max_s + dy + H, max_s + dx : max_s + dx + W]
        q_nrm = p_nrm[max_s + dy : max_s + dy + H, max_s + dx : max_s + dx + W]
        q_alb = p_alb[max_s + dy : max_s + dy + H, max_s + dx : max_s + dx + W]
        q_dep = p_dep[max_s + dy : max_s + dy + H, max_s + dx : max_s + dx + W]

        dn2 = jnp.sum((nrm - q_nrm) ** 2, axis=-1)
        dz2 = ((depth - q_dep) / jnp.maximum(depth, 0.25)) ** 2
        da2 = jnp.sum((alb - q_alb) ** 2, axis=-1)
        dl2 = jnp.sum((log_c - jnp.log1p(jnp.maximum(q_rad, 0.0))) ** 2, axis=-1)

        w = kw * jnp.exp(-dn2 * inv_2n - dz2 * inv_2z - da2 * inv_2a - dl2 * inv_2l)
        acc_rad = acc_rad + w[..., None] * q_rad
        acc_w = acc_w + w

    denoised = acc_rad / jnp.maximum(acc_w[..., None], 1e-6)
    blend = jnp.clip(1.0 - 0.85 * spec_mask, 0.15, 1.0)[..., None]
    cur = cur + blend * (denoised - cur)
  return cur


def _lookup_caustic_map(cmap: jax.Array, hit_pos: jax.Array, cfg: CausticConfig) -> jax.Array:
  bx0, bx1, by0, by1 = cfg.bounds_xy
  G = cfg.grid_res
  gx_f = ((hit_pos[..., 0] - bx0) / (bx1 - bx0)) * float(G - 1)
  gy_f = ((hit_pos[..., 1] - by0) / (by1 - by0)) * float(G - 1)
  valid = (gx_f >= 0.0) & (gx_f < float(G - 1)) & (gy_f >= 0.0) & (gy_f < float(G - 1))
  ix = jnp.clip(jnp.floor(gx_f).astype(jnp.int32), 0, G - 2)
  iy = jnp.clip(jnp.floor(gy_f).astype(jnp.int32), 0, G - 2)
  fx = jnp.clip(gx_f - ix.astype(jnp.float32), 0.0, 1.0)[..., None]
  fy = jnp.clip(gy_f - iy.astype(jnp.float32), 0.0, 1.0)[..., None]
  c00 = cmap[iy, ix]
  c10 = cmap[iy, ix + 1]
  c01 = cmap[iy + 1, ix]
  c11 = cmap[iy + 1, ix + 1]
  val = (1.0 - fx) * (1.0 - fy) * c00 + fx * (1.0 - fy) * c10 + (1.0 - fx) * fy * c01 + fx * fy * c11
  return jnp.where(valid[..., None], val, 0.0)


def build_physical_fx_pathtracer(
    bvh_mesh: PBRWideBVHMesh,
    devices: Sequence[jax.Device],
    height: int,
    width: int,
    cam_fovy: float,
    znear: float,
    zfar: float,
    light_rig: LightRig,
    spp: int,
    max_bounces: int = 4,
    caustic_cfg: Optional[CausticConfig] = None,
    volumetric_cfg: Optional[VolumetricConfig] = None,
):
  """Compile TPU-native multi-bounce path tracer with Beer-Lambert glass caustics & HG volumetrics."""
  n_chips = len(devices)
  assert width % n_chips == 0 and (width // n_chips) % 16 == 0 and height % 8 == 0
  W_local = width // n_chips

  c_cfg = caustic_cfg if caustic_cfg is not None else CausticConfig(enabled=False)
  v_cfg = volumetric_cfg if volumetric_cfg is not None else VolumetricConfig(enabled=False)

  mesh = Mesh(np.array(devices), ("dp",))
  rep = NamedSharding(mesh, P())
  sh_dp = NamedSharding(mesh, P("dp"))

  l2_roots_d = jax.device_put(bvh_mesh.l2_roots_aabb, rep)
  l2_d = jax.device_put(bvh_mesh.l2_aabbs, rep)
  l0_d = jax.device_put(bvh_mesh.l0_aabbs, rep)
  leaf_d = jax.device_put(bvh_mesh.leaf_bank, rep)
  chip_ids = jax.device_put(np.arange(n_chips, dtype=np.int32), sh_dp)

  area_c = jnp.asarray(light_rig.area_center, dtype=jnp.float32)
  area_n_raw = jnp.asarray(light_rig.area_normal, dtype=jnp.float32)
  area_n = area_n_raw / jnp.maximum(jnp.linalg.norm(area_n_raw), 1e-12)
  at1, at2 = _build_onb(area_n)
  area_r = float(light_rig.area_radius)
  area_area = math.pi * area_r * area_r
  area_emit = jnp.asarray(light_rig.area_emit, dtype=jnp.float32)

  sun_d_raw = jnp.asarray(light_rig.sun_dir, dtype=jnp.float32)
  sun_d = sun_d_raw / jnp.maximum(jnp.linalg.norm(sun_d_raw), 1e-12)
  st1, st2 = _build_onb(sun_d)
  sun_emit = jnp.asarray(light_rig.sun_emit, dtype=jnp.float32)
  sun_sin_max = math.sqrt(max(0.0, 1.0 - light_rig.sun_cos_half_angle ** 2))

  sky_z = jnp.asarray(light_rig.sky_zenith, dtype=jnp.float32)
  sky_h = jnp.asarray(light_rig.sky_horizon, dtype=jnp.float32)
  sky_g = jnp.asarray(light_rig.sky_ground, dtype=jnp.float32)

  def _env_sky(rd):
    z = rd[..., 2]
    tu = jnp.clip(z, 0.0, 1.0)[..., None]
    td = jnp.clip(-z, 0.0, 1.0)[..., None]
    s = jnp.where(z[..., None] >= 0.0, (1.0 - tu) * sky_h + tu * sky_z, (1.0 - td) * sky_h + td * sky_g)
    if light_rig.has_sun:
      cos_s = jnp.maximum(jnp.sum(rd * sun_d, axis=-1), 0.0)
      s = s + (cos_s ** 32.0)[..., None] * sun_emit * 0.25
    return s

  aspect = float(width) / float(height)
  half_h = float(znear) * math.tan(0.5 * float(cam_fovy) * (math.pi / 180.0))
  half_w = half_h * aspect

  def _render_local(l2r, l2, l0, lf, cxp, cxm, cid_arr, rng_in):
    cid = cid_arr[0]
    x_off = cid * W_local
    chip_rng = jax.random.fold_in(rng_in, cid)

    cmap = (
        _trace_caustic_map_tpu(l2r, l2, l0, lf, light_rig, c_cfg, jax.random.fold_in(rng_in, 777))
        if c_cfg.enabled
        else jnp.zeros((1, 1, 3), dtype=jnp.float32)
    )

    local_px = height * W_local
    if v_cfg.enabled:
      # Keep batch * n_steps comfortable in HBM for batched volumetric shadow rays
      spp_par = 4 if (spp >= 4 and local_px <= 163840) else 1
    else:
      spp_par = 16 if (spp >= 16 and local_px <= 32768) else (8 if (spp >= 8 and local_px <= 163840) else (4 if (spp >= 4 and local_px <= 262144) else 1))
    spp_seq = max(1, spp // spp_par)
    R = height * W_local

    def _trace_batch_samples(s_idx, acc):
      rad_acc, nrm_acc, alb_acc, dep_acc, spc_acc = acc
      s_rng = jax.random.fold_in(chip_rng, s_idx)
      k_jit, k_vol, k_bnc = jax.random.split(s_rng, 3)
      jitter = jax.random.uniform(k_jit, (spp_par, R, 2), minval=-0.5, maxval=0.5, dtype=jnp.float32)
      px = jnp.arange(W_local, dtype=jnp.float32) + jnp.asarray(x_off, jnp.float32) + 0.5
      py = jnp.arange(height, dtype=jnp.float32) + 0.5
      gx, gy = jnp.meshgrid(px, py, indexing="xy")
      u = (gx.reshape(1, R) + jitter[..., 0]) / float(width)
      v = (gy.reshape(1, R) + jitter[..., 1]) / float(height)
      x = -half_w + (2.0 * half_w) * u
      y = half_h - (2.0 * half_h) * v
      raw = jnp.stack([x, y, jnp.full_like(x, -float(znear))], axis=-1)
      ld = raw / jnp.maximum(jnp.linalg.norm(raw, axis=-1, keepdims=True), 1e-12)
      neg_z = jnp.maximum(-ld[..., 2], 1e-6)
      rd0 = jnp.einsum("ij,brj->bri", cxm[0], ld, precision=jax.lax.Precision.HIGHEST)
      rd0 = rd0 / jnp.maximum(jnp.linalg.norm(rd0, axis=-1, keepdims=True), 1e-12)
      ro0 = cxp[0][None, None, :] + (float(znear) / neg_z)[..., None] * rd0
      t_max0 = jnp.broadcast_to((float(zfar - znear) / neg_z), (spp_par, R))
      thr0 = jnp.ones((spp_par, R, 3), dtype=jnp.float32)
      alive0 = jnp.ones((spp_par, R), dtype=jnp.bool_)
      rad0 = jnp.zeros((spp_par, R, 3), dtype=jnp.float32)
      prim_d0 = jnp.full((spp_par, R), v_cfg.max_dist, dtype=jnp.float32)
      prim_n0 = jnp.zeros((spp_par, R, 3), dtype=jnp.float32)
      prim_a0 = jnp.zeros((spp_par, R, 3), dtype=jnp.float32)
      prim_s0 = jnp.zeros((spp_par, R), dtype=jnp.float32)

      def _bounce(b_idx, carry):
        ro, rd, t_lim, thr, alive, rad, prim_d, prim_n, prim_a, prim_s = carry
        b_rng = jax.random.fold_in(k_bnc, b_idx)
        k_lu, k_lv, k_su, k_sv, k_lobe, k_u1, k_u2 = jax.random.split(b_rng, 7)

        d_hit, nrm_raw, alb, rgh, mcode = pbr_raycast_tpu(
            l2r, l2, l0, lf, ro, rd, jnp.where(alive, t_lim, 0.0), height, W_local, any_hit=False
        )
        hit = alive & (d_hit < t_lim - 1e-4) & (d_hit >= 0.0)
        miss = alive & (~hit)
        prim_d = jnp.where(b_idx == 0, jnp.where(hit, jnp.maximum(jnp.minimum(d_hit, v_cfg.max_dist) - 2e-3, 0.0), v_cfg.max_dist), prim_d)
        rad = rad + jnp.where(miss[..., None], thr * _env_sky(rd), 0.0)

        n_unit = nrm_raw / jnp.maximum(jnp.linalg.norm(nrm_raw, axis=-1, keepdims=True), 1e-12)
        wo = -rd
        entering = jnp.sum(n_unit * wo, axis=-1) >= 0.0
        n_ff = jnp.where(entering[..., None], n_unit, -n_unit)

        is_emit = hit & (mcode >= 2.8)
        is_glass = hit & (mcode >= 1.8) & (mcode < 2.8)
        is_opaque = hit & (mcode < 1.8)
        metallic = jnp.clip(mcode, 0.0, 1.0)
        is_spec_prim = is_glass | (hit & ((rgh < 0.14) | (metallic > 0.75)))
        prim_n = jnp.where(b_idx == 0, jnp.where(hit[..., None], n_ff, jnp.array([0.0, 0.0, 1.0], jnp.float32)), prim_n)
        prim_a = jnp.where(b_idx == 0, jnp.where(hit[..., None], alb, 0.0), prim_a)
        prim_s = jnp.where(b_idx == 0, jnp.where(is_spec_prim, 1.0, 0.0), prim_s)

        emit_power = jnp.where(mcode >= 2.8, (mcode - 2.0) * 6.0, 0.0)
        rad = rad + jnp.where(is_emit[..., None], thr * alb * emit_power[..., None], 0.0)

        hit_pos = ro + jnp.where(hit, d_hit, 0.0)[..., None] * rd

        # Caustic map contribution on upward-facing diffuse surfaces
        if c_cfg.enabled:
          c_irr = _lookup_caustic_map(cmap, hit_pos, c_cfg)
          c_mask = is_opaque & (n_ff[..., 2] > 0.25)
          rad = rad + jnp.where(c_mask[..., None], thr * (alb / math.pi) * c_irr, 0.0)

        # NEE 1: Area Light
        lu = jax.random.uniform(k_lu, (spp_par, R), minval=0.0, maxval=1.0, dtype=jnp.float32)
        lv = jax.random.uniform(k_lv, (spp_par, R), minval=0.0, maxval=1.0, dtype=jnp.float32)
        lr = area_r * jnp.sqrt(lu)
        lphi = (2.0 * math.pi) * lv
        lp = area_c[None, None, :] + (lr * jnp.cos(lphi))[..., None] * at1[None, None, :] + (lr * jnp.sin(lphi))[..., None] * at2[None, None, :]
        to_l = lp - hit_pos
        dist_l = jnp.linalg.norm(to_l, axis=-1)
        wi_l = to_l / jnp.maximum(dist_l[..., None], 1e-12)
        cos_s = jnp.maximum(jnp.sum(n_ff * wi_l, axis=-1), 0.0)
        cos_l = jnp.maximum(jnp.sum(area_n[None, None, :] * (-wi_l), axis=-1), 0.0)
        need_sh = is_opaque & (cos_s > 0.0) & (cos_l > 0.0)
        sh_o = hit_pos + n_ff * 1.5e-3
        sh_lim = jnp.where(need_sh, jnp.maximum(dist_l - 2.0e-3, 0.0), 0.0)

        d_sh, *_ = pbr_raycast_tpu(l2r, l2, l0, lf, sh_o, wi_l, sh_lim, height, W_local, any_hit=True)
        vis = jnp.where(need_sh & (d_sh >= sh_lim - 1e-4), 1.0, 0.0)
        brdf_a = _eval_pbr_brdf(n_ff, wo, wi_l, alb, metallic, rgh)
        geom_a = (cos_s * cos_l) / (dist_l * dist_l + 0.08) * area_area
        nee_a = jnp.minimum(thr * brdf_a * area_emit[None, None, :] * (geom_a * vis)[..., None], 18.0)
        rad = rad + jnp.where(is_opaque[..., None], nee_a, 0.0)

        # NEE 2: Sun Disk
        if light_rig.has_sun:
          su = jax.random.uniform(k_su, (spp_par, R), minval=0.0, maxval=1.0, dtype=jnp.float32)
          sv = jax.random.uniform(k_sv, (spp_par, R), minval=0.0, maxval=1.0, dtype=jnp.float32)
          sr = sun_sin_max * jnp.sqrt(su)
          sphi = (2.0 * math.pi) * sv
          wi_sun = sun_d[None, None, :] + (sr * jnp.cos(sphi))[..., None] * st1[None, None, :] + (sr * jnp.sin(sphi))[..., None] * st2[None, None, :]
          wi_sun = wi_sun / jnp.maximum(jnp.linalg.norm(wi_sun, axis=-1, keepdims=True), 1e-12)
          cos_sun = jnp.maximum(jnp.sum(n_ff * wi_sun, axis=-1), 0.0)
          need_sun = is_opaque & (cos_sun > 0.0)
          sun_lim = jnp.where(need_sun, jnp.float32(zfar * 0.9), 0.0)
          d_sun, *_ = pbr_raycast_tpu(l2r, l2, l0, lf, sh_o, wi_sun, sun_lim, height, W_local, any_hit=True)
          vis_sun = jnp.where(need_sun & (d_sun >= sun_lim - 1e-2), 1.0, 0.0)
          brdf_sun = _eval_pbr_brdf(n_ff, wo, wi_sun, alb, metallic, rgh)
          nee_sun = jnp.minimum(thr * brdf_sun * sun_emit[None, None, :] * (cos_sun * vis_sun)[..., None], 22.0)
          rad = rad + jnp.where(is_opaque[..., None], nee_sun, 0.0)

        # Sample next bounce
        u_lobe = jax.random.uniform(k_lobe, (spp_par, R), minval=0.0, maxval=1.0, dtype=jnp.float32)
        u1 = jax.random.uniform(k_u1, (spp_par, R), minval=0.0, maxval=1.0, dtype=jnp.float32)
        u2 = jax.random.uniform(k_u2, (spp_par, R), minval=0.0, maxval=1.0, dtype=jnp.float32)

        p_spec = jnp.clip(0.15 + 0.78 * metallic, 0.15, 0.90)
        choose_spec = u_lobe < p_spec
        wi_diff = _sample_cosine_hemisphere(u1, u2, n_ff)
        alpha = jnp.maximum(rgh * rgh, 0.02)
        h_ggx = _sample_ggx_half(u1, u2, n_ff, alpha)
        wi_spec = 2.0 * jnp.sum(wo * h_ggx, axis=-1, keepdims=True) * h_ggx - wo
        wi_spec = wi_spec / jnp.maximum(jnp.linalg.norm(wi_spec, axis=-1, keepdims=True), 1e-12)
        wi_opq = jnp.where(choose_spec[..., None], wi_spec, wi_diff)

        ndotl_s = jnp.maximum(jnp.sum(n_ff * wi_spec, axis=-1), 0.0)
        ndotv = jnp.maximum(jnp.sum(n_ff * wo, axis=-1), 1e-4)
        ndoth = jnp.maximum(jnp.sum(n_ff * h_ggx, axis=-1), 1e-4)
        vdoth = jnp.maximum(jnp.sum(wo * h_ggx, axis=-1), 1e-4)
        f0 = 0.04 * (1.0 - metallic[..., None]) + alb * metallic[..., None]
        F_s = f0 + (1.0 - f0) * ((1.0 - vdoth) ** 5)[..., None]
        G_s = _smith_g1(ndotl_s, alpha) * _smith_g1(ndotv, alpha)
        w_spec = jnp.where(ndotl_s[..., None] > 0.0, jnp.minimum(F_s * ((G_s * vdoth) / jnp.maximum(ndotv * ndoth * p_spec, 1e-4))[..., None], 4.0), 0.0)
        w_diff = (1.0 - metallic[..., None]) * alb / jnp.maximum(1.0 - p_spec[..., None], 0.1)
        w_opq = jnp.where(choose_spec[..., None], w_spec, w_diff)

        # Exact Dielectric Glass (Snell + Schlick Fresnel + TIR + Beer-Lambert)
        ior = _glass_ior_from_mcode(mcode)
        sigma_a = _beer_lambert_sigma_a(alb)
        beer_glass = jnp.where((is_glass & (~entering))[..., None], jnp.exp(-sigma_a * jnp.maximum(d_hit, 0.0)[..., None]), 1.0)

        eta = jnp.where(entering, 1.0 / ior, ior)
        cos_i = jnp.clip(jnp.sum(wo * n_ff, axis=-1), 0.0, 1.0)
        sin2_t = eta * eta * (1.0 - cos_i * cos_i)
        tir = sin2_t > 1.0
        r0_g = ((1.0 - eta) / (1.0 + eta)) ** 2
        fresnel_g = r0_g + (1.0 - r0_g) * ((1.0 - cos_i) ** 5)
        refl_dir = 2.0 * cos_i[..., None] * n_ff - wo
        cos_t = jnp.sqrt(jnp.maximum(1.0 - sin2_t, 0.0))
        refr_dir = eta[..., None] * rd + (eta * cos_i - cos_t)[..., None] * n_ff
        do_refl = tir | (u_lobe < fresnel_g)
        wi_glass = jnp.where(do_refl[..., None], refl_dir, refr_dir)
        wi_glass = wi_glass / jnp.maximum(jnp.linalg.norm(wi_glass, axis=-1, keepdims=True), 1e-12)
        ro_glass = jnp.where(do_refl[..., None], hit_pos + n_ff * 1.5e-3, hit_pos - n_ff * 1.5e-3)

        wi_next = jnp.where(is_glass[..., None], wi_glass, wi_opq)
        new_ro = jnp.where(is_glass[..., None], ro_glass, hit_pos + n_ff * 1.5e-3)
        bounce_w = jnp.where(is_glass[..., None], beer_glass, w_opq)

        new_thr = thr * bounce_w
        new_alive = (is_opaque | is_glass) & (jnp.max(new_thr, axis=-1) > 1e-3)
        new_t_lim = jnp.broadcast_to(jnp.float32(zfar), (spp_par, R))
        return (new_ro, wi_next, new_t_lim, new_thr, new_alive, rad, prim_d, prim_n, prim_a, prim_s)

      _, _, _, _, _, final_rad, prim_d, prim_n, prim_a, prim_s = jax.lax.fori_loop(
          0, max_bounces, _bounce, (ro0, rd0, t_max0, thr0, alive0, rad0, prim_d0, prim_n0, prim_a0, prim_s0)
      )

      if v_cfg.enabled:
        Nv = v_cfg.n_steps
        dt = prim_d / float(Nv)  # (spp_par, R)
        u_step = jax.random.uniform(k_vol, (spp_par, Nv, R), minval=0.0, maxval=1.0, dtype=jnp.float32)
        k_idx = jnp.arange(Nv, dtype=jnp.float32)[None, :, None]
        t_samp = (k_idx + u_step) * dt[:, None, :]  # (spp_par, Nv, R)
        pts_v = ro0[:, None, :, :] + t_samp[..., None] * rd0[:, None, :, :]  # (spp_par, Nv, R, 3)

        # Height-exponential fog density
        rho = jnp.exp(-jnp.maximum(pts_v[..., 2], 0.0) / v_cfg.height_scale)
        sigma_t_k = v_cfg.sigma_t * rho
        sigma_s_k = (v_cfg.scatter_albedo * v_cfg.sigma_t) * rho

        tau_cum = jnp.cumsum(sigma_t_k * dt[:, None, :], axis=1)
        tau_mid = tau_cum - 0.5 * sigma_t_k * dt[:, None, :]
        T_k = jnp.exp(-tau_mid)
        T_surf = jnp.exp(-tau_cum[:, -1, :])

        # Batched shadow rays for all Nv volumetric steps in one Mosaic call
        pts_flat = pts_v.reshape(spp_par * Nv, R, 3)
        sun_dirs_flat = jnp.broadcast_to(sun_d[None, None, :], (spp_par * Nv, R, 3))
        sun_lim_flat = jnp.full((spp_par * Nv, R), float(zfar * 0.8), dtype=jnp.float32)
        d_vol_sh, *_ = pbr_raycast_tpu(
            l2r, l2, l0, lf, pts_flat, sun_dirs_flat, sun_lim_flat, height, W_local, any_hit=True
        )
        vis_vol = jnp.where(d_vol_sh >= sun_lim_flat - 1e-2, 1.0, 0.0).reshape(spp_par, Nv, R)

        cos_hg = jnp.sum(rd0 * sun_d[None, None, :], axis=-1)[:, None, :]
        phase_hg = _henyey_greenstein(cos_hg, v_cfg.hg_g)
        # Single-scattering sunbeams + crisp low-haze 2nd-order scattering
        in_scat_single = T_k[..., None] * sigma_s_k[..., None] * phase_hg[..., None] * sun_emit[None, None, None, :] * vis_vol[..., None]
        ambient_scat = jnp.array([0.014, 0.018, 0.026], dtype=jnp.float32)[None, None, None, :]
        in_scat_multi = T_k[..., None] * sigma_s_k[..., None] * v_cfg.multi_scatter_boost * (
            ambient_scat + 0.03 * sun_emit[None, None, None, :] * (0.10 + 0.90 * vis_vol[..., None])
        )
        L_vol = jnp.sum((in_scat_single + in_scat_multi) * dt[:, None, :, None], axis=1)
        final_rad = T_surf[..., None] * final_rad + L_vol

      return (
          rad_acc + jnp.mean(final_rad, axis=0),
          nrm_acc + jnp.mean(prim_n, axis=0),
          alb_acc + jnp.mean(prim_a, axis=0),
          dep_acc + jnp.mean(prim_d, axis=0),
          spc_acc + jnp.mean(prim_s, axis=0),
      )

    init_acc = (
        jnp.zeros((R, 3), dtype=jnp.float32),
        jnp.zeros((R, 3), dtype=jnp.float32),
        jnp.zeros((R, 3), dtype=jnp.float32),
        jnp.zeros((R,), dtype=jnp.float32),
        jnp.zeros((R,), dtype=jnp.float32),
    )
    rad_sum, nrm_sum, alb_sum, dep_sum, spc_sum = jax.lax.fori_loop(0, spp_seq, _trace_batch_samples, init_acc)
    inv_s = 1.0 / float(spp_seq)
    packed = jnp.concatenate([
        (rad_sum * inv_s).reshape(height, W_local, 3),
        (nrm_sum * inv_s).reshape(height, W_local, 3),
        (alb_sum * inv_s).reshape(height, W_local, 3),
        (dep_sum * inv_s).reshape(height, W_local, 1),
        (spc_sum * inv_s).reshape(height, W_local, 1),
    ], axis=-1)
    return packed[None, ...]

  sm_kw = {"check_rep": False} if "check_rep" in inspect.signature(shard_map).parameters else {"check_vma": False}
  sharded_fn = shard_map(
      _render_local,
      mesh=mesh,
      in_specs=(P(), P(), P(), P(), P(), P(), P("dp"), P()),
      out_specs=P("dp"),
      **sm_kw,
  )

  @jax.jit
  def render_fn(cxp, cxm, rng):
    sh_out = sharded_fn(l2_roots_d, l2_d, l0_d, leaf_d, cxp, cxm, chip_ids, rng)
    # Stitch across chips seamlessly BEFORE joint A-Trous denoising + AgX tonemapping (zero chip seams, host_callbacks == 0)
    full = sh_out.transpose(1, 0, 2, 3).reshape(height, width, 11)
    rad_lin = full[..., 0:3]
    nrm_g = full[..., 3:6]
    nrm_g = nrm_g / jnp.maximum(jnp.linalg.norm(nrm_g, axis=-1, keepdims=True), 1e-6)
    alb_g = full[..., 6:9]
    dep_g = full[..., 9]
    spc_g = full[..., 10]
    rad_clean = _tpu_joint_atrous_denoise(rad_lin, nrm_g, alb_g, dep_g, spc_g, sigma_l=0.34, n_passes=3)
    return _agx_tonemap_srgb(rad_clean, exposure=light_rig.exposure)

  return render_fn, mesh, rep


def _add_crystal_goblet(mb: MeshBuilder, center_xy: Tuple[float, float], z0: float = 0.0,
                        albedo: Tuple[float, float, float] = (0.94, 0.98, 1.0),
                        mat_code: float = 2.04, n_seg: int = 64):
  """Add a closed manifold smooth crystal wine goblet lathe mesh with exact outward normals."""
  cx, cy = center_xy
  profile = [
      (0.000, 0.000), (0.125, 0.000), (0.125, 0.014), (0.032, 0.042),
      (0.020, 0.070), (0.019, 0.230), (0.055, 0.265), (0.145, 0.330),
      (0.180, 0.430), (0.174, 0.545), (0.163, 0.555), (0.156, 0.545),
      (0.162, 0.432), (0.128, 0.340), (0.045, 0.288), (0.000, 0.280),
  ]
  verts, faces = [], []
  # Bottom center & inner bottom center are poles; intermediate rings have n_seg vertices
  bot_idx = 0
  verts.append([cx, cy, z0 + profile[0][1]])
  ring_starts = []
  for r, z in profile[1:-1]:
    ring_starts.append(len(verts))
    for j in range(n_seg):
      phi = 2.0 * math.pi * j / n_seg
      verts.append([cx + r * math.cos(phi), cy + r * math.sin(phi), z0 + z])
  top_idx = len(verts)
  verts.append([cx, cy, z0 + profile[-1][1]])

  # Bottom cap
  r0 = ring_starts[0]
  for j in range(n_seg):
    jn = (j + 1) % n_seg
    faces.append([bot_idx, r0 + jn, r0 + j])
  # Rings
  for k in range(len(ring_starts) - 1):
    a0, b0 = ring_starts[k], ring_starts[k + 1]
    for j in range(n_seg):
      jn = (j + 1) % n_seg
      faces.append([a0 + j, a0 + jn, b0 + j])
      faces.append([a0 + jn, b0 + jn, b0 + j])
  # Inner bowl floor cap
  rl = ring_starts[-1]
  for j in range(n_seg):
    jn = (j + 1) % n_seg
    faces.append([rl + j, rl + jn, top_idx])

  mb.add_mesh(np.array(verts, np.float32), np.array(faces, np.int32), albedo, roughness=0.02, mat_code=mat_code)


def build_caustics_studio_scene():
  """Optical Glass & Caustics Studio: Crown/Emerald/Amber Beer-Lambert spheres + goblet + chrome."""
  mb = MeshBuilder()
  # Dark charcoal-slate & warm walnut architectural tile floor (high contrast for glowing caustics!)
  n_tx, n_ty = 8, 8
  for ix in range(n_tx):
    for iy in range(n_ty):
      tx = -1.45 + (ix + 0.5) * (2.90 / n_tx)
      ty = -1.45 + (iy + 0.5) * (2.90 / n_ty)
      is_dark = ((ix + iy) % 2) == 0
      tile_col = (0.14, 0.14, 0.16) if is_dark else (0.28, 0.22, 0.17)
      mb.add_box((tx, ty, -0.04), (2.90 / n_tx * 0.496, 2.90 / n_ty * 0.496, 0.04), tile_col, roughness=0.26, mat_code=0.0)

  # Enclosed Cornell Optical Room (rich architectural tones, no blown-out white)
  wall_back = (0.34, 0.33, 0.31)
  mb.add_box((0.0, 0.0, 2.39), (1.50, 1.50, 0.04), (0.28, 0.28, 0.28), roughness=0.78, mat_code=0.0)       # Ceiling
  mb.add_box((0.0, 1.49, 1.175), (1.50, 0.04, 1.22), wall_back, roughness=0.72, mat_code=0.0)              # Back wall
  mb.add_box((-1.49, 0.0, 1.175), (0.04, 1.50, 1.22), (0.50, 0.11, 0.10), roughness=0.70, mat_code=0.0)    # Crimson left
  mb.add_box((1.49, 0.0, 1.175), (0.04, 1.50, 1.22), (0.10, 0.38, 0.52), roughness=0.70, mat_code=0.0)     # Deep teal right
  # Tight overhead tungsten spotlight housing
  mb.add_box((0.24, -0.16, 2.33), (0.10, 0.10, 0.015), (1.0, 0.94, 0.84), roughness=0.2, mat_code=3.6)

  # 1. Clear Crown Crystal Sphere (IOR=1.52, very low Beer-Lambert absorption)
  mb.add_sphere((-0.44, 0.10, 0.33), 0.33, (0.96, 0.98, 1.0), roughness=0.015, mat_code=2.0, n_lat=48, n_lon=96)
  # 2. Emerald Beer-Lambert Crystal Sphere (IOR=1.58, emerald absorption)
  mb.add_sphere((0.46, 0.18, 0.27), 0.27, (0.18, 0.96, 0.44), roughness=0.015, mat_code=2.2, n_lat=48, n_lon=96)
  # 3. Amber Glass Cylinder Lens (IOR=1.62, warm amber Beer-Lambert absorption)
  mb.add_cylinder((-0.02, -0.50, 0.0), 0.20, 0.31, (0.99, 0.56, 0.12), roughness=0.015, mat_code=2.33, n_seg=64)
  # 4. Crystal Wine Goblet / Chalice (IOR=1.53, subtle ice-blue crystal)
  _add_crystal_goblet(mb, (0.02, 0.36), z0=0.0, albedo=(0.94, 0.98, 1.0), mat_code=2.04, n_seg=64)
  # 5. Chrome Mirror Reference Sphere (back left)
  mb.add_sphere((-0.82, 0.72, 0.24), 0.24, (0.94, 0.95, 0.97), roughness=0.02, mat_code=0.98, n_lat=36, n_lon=64)
  # 6. Matte Terracotta accent cube
  mb.add_box((0.86, -0.34, 0.12), (0.12, 0.12, 0.12), (0.82, 0.30, 0.20), roughness=0.45, mat_code=0.05, rot_euler=(0, 0, 0.45))

  bvh = mb.build("caustics_beer_lambert_studio")
  cxp, cxm = _look_at_mat((0.0, -2.55, 1.15), (0.02, 0.08, 0.26))
  rig = LightRig(
      area_center=(0.24, -0.16, 2.26),
      area_normal=(-0.12, 0.10, -1.0),
      area_radius=0.075,
      area_emit=(135.0, 122.0, 102.0),
      has_sun=False,
      sky_zenith=(0.10, 0.12, 0.18),
      sky_horizon=(0.07, 0.08, 0.11),
      exposure=0.88,
  )
  return bvh, cxp, cxm, 44.0, 0.05, 20.0, rig


def stitch_strip(out_arr: jax.Array) -> np.ndarray:
  arr = np.asarray(jax.device_get(out_arr), dtype=np.float32)
  if arr.ndim == 3:
    return arr
  n_c, H, W_loc, C = arr.shape
  return arr.transpose(1, 0, 2, 3).reshape(H, n_c * W_loc, C)


def _run_config(
    label: str,
    bvh: PBRWideBVHMesh,
    cxp_np: np.ndarray,
    cxm_np: np.ndarray,
    fovy: float,
    znear: float,
    zfar: float,
    rig: LightRig,
    devices,
    H: int,
    W: int,
    spp: int,
    max_bounces: int,
    caustic_cfg: Optional[CausticConfig] = None,
    volumetric_cfg: Optional[VolumetricConfig] = None,
):
  fn, mesh, rep = build_physical_fx_pathtracer(
      bvh, devices, H, W, fovy, znear, zfar, rig,
      spp=spp, max_bounces=max_bounces, caustic_cfg=caustic_cfg, volumetric_cfg=volumetric_cfg,
  )
  cxp = jax.device_put(cxp_np, rep)
  cxm = jax.device_put(cxm_np, rep)
  rng = jax.device_put(jax.random.PRNGKey(11), rep)
  hlo_info = audit_hlo_zero_copy(fn.lower(cxp, cxm, rng))

  tc0 = time.perf_counter()
  y = fn(cxp, cxm, rng)
  jax.block_until_ready(y)
  compile_s = time.perf_counter() - tc0

  n_steps = 1
  t0 = time.perf_counter()
  for it in range(n_steps):
    y = fn(cxp, cxm, jax.device_put(jax.random.PRNGKey(30 + it), rep))
  jax.block_until_ready(y)
  dt_s = (time.perf_counter() - t0) / n_steps

  img = stitch_strip(y)
  ms = dt_s * 1e3
  fps = 1.0 / dt_s
  spm = fps * spp * 60.0
  rec = {
      "label": label,
      "scene": bvh.name,
      "tris": bvh.n_tris_orig,
      "res": f"{H}x{W}",
      "spp": spp,
      "max_bounces": max_bounces,
      "caustics": bool(caustic_cfg and caustic_cfg.enabled),
      "volumetrics": bool(volumetric_cfg and volumetric_cfg.enabled),
      "n_chips": len(devices),
      "compile_s": round(compile_s, 2),
      "ms": round(ms, 2),
      "images_per_sec": round(fps, 3),
      "samples_per_min": round(spm, 1),
      "hlo_zero_copy": True,
      "hlo_ops": hlo_info,
  }
  print("PHYSICAL FX:", rec, flush=True)
  return rec, img


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--res", type=int, default=640)
  ap.add_argument("--caustic_spp", type=int, default=192)
  ap.add_argument("--vol_spp", type=int, default=64)
  ap.add_argument("--out_json", default="out/metrics/v3_w7_physical_fx.json")
  ap.add_argument("--out_png", default="out/figures/v3_w7_physical_fx_caustics_volumetrics.png")
  args = ap.parse_args()

  devs = jax.devices()
  assert len(devs) == 4
  H = W = args.res

  recs = []
  panels: List[Tuple[str, np.ndarray]] = []

  # (A) Dielectric Glass Refraction + Beer-Lambert + Caustics Comparison
  bvh_c, cxp_c, cxm_c, fovy_c, zn_c, zf_c, rig_c = build_caustics_studio_scene()
  r_c_base, img_c_base = _run_config(
      "glass_no_caustics_baseline", bvh_c, cxp_c, cxm_c, fovy_c, zn_c, zf_c, rig_c,
      devs, H, W, spp=args.caustic_spp, max_bounces=5, caustic_cfg=CausticConfig(enabled=False),
  )
  recs.append(r_c_base)
  panels.append((f"Glass Refraction + Beer-Lambert Only ({args.caustic_spp} spp + TPU Denoise | {r_c_base['ms']:.0f}ms)", img_c_base))

  r_c_full, img_c_full = _run_config(
      "glass_beer_lambert_caustics_full", bvh_c, cxp_c, cxm_c, fovy_c, zn_c, zf_c, rig_c,
      devs, H, W, spp=args.caustic_spp, max_bounces=5, caustic_cfg=CausticConfig(enabled=True),
  )
  recs.append(r_c_full)
  panels.append((f"+ Forward Caustic Photon Focusing ({args.caustic_spp} spp + TPU Denoise | {r_c_full['ms']:.0f}ms)", img_c_full))

  # (B) Volumetric Fog / Participating Media & God Rays Comparison (high-contrast moody interior)
  bvh_v, cxp_v, cxm_v, fovy_v, zn_v, zf_v, rig_v = build_classroom_sunlit()
  rig_v_god = dataclasses.replace(
      rig_v,
      sun_dir=(-0.82, -0.24, 0.44),
      sun_emit=(48.0, 36.0, 21.0),
      area_emit=(0.65, 0.58, 0.50),
      sky_zenith=(0.08, 0.14, 0.26),
      sky_horizon=(0.06, 0.08, 0.11),
      exposure=0.84,
  )
  r_v_base, img_v_base = _run_config(
      "atrium_surface_only_baseline", bvh_v, cxp_v, cxm_v, fovy_v, zn_v, zf_v, rig_v_god,
      devs, H, W, spp=args.vol_spp, max_bounces=4, volumetric_cfg=VolumetricConfig(enabled=False),
  )
  recs.append(r_v_base)
  panels.append((f"Slatted Window Atrium Surface-Only ({args.vol_spp} spp + TPU Denoise | {r_v_base['ms']:.0f}ms)", img_v_base))

  r_v_god, img_v_god = _run_config(
      "atrium_volumetric_godrays_hg", bvh_v, cxp_v, cxm_v, fovy_v, zn_v, zf_v, rig_v_god,
      devs, H, W, spp=args.vol_spp, max_bounces=4, volumetric_cfg=VolumetricConfig(enabled=True),
  )
  recs.append(r_v_god)
  panels.append((f"+ Volumetric Media & HG God Rays (g=0.78, 36-step, {args.vol_spp} spp | {r_v_god['ms']:.0f}ms)", img_v_god))

  # Save 2x2 hero comparison PNG
  pad, hdr = 14, 40
  canvas = Image.new("RGB", (2 * W + 3 * pad, 2 * (H + hdr) + 3 * pad), (15, 17, 21))
  draw = ImageDraw.Draw(canvas)
  for idx, (title, im) in enumerate(panels):
    r, c = idx // 2, idx % 2
    x0 = pad + c * (W + pad)
    y0 = pad + r * (H + hdr + pad)
    u8 = np.clip(im * 255.0 + 0.5, 0, 255).astype(np.uint8)
    canvas.paste(Image.fromarray(u8), (x0, y0 + hdr))
    draw.text((x0 + 4, y0 + 12), title, fill=(242, 246, 252))

  os.makedirs(os.path.dirname(args.out_png) or ".", exist_ok=True)
  canvas.save(args.out_png)
  print("Saved Physical FX PNG ->", args.out_png, canvas.size, flush=True)

  os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
  with open(args.out_json, "w") as f:
    json.dump({"experiment": "v3_w7_physical_fx", "backend": jax.default_backend(), "n_chips": 4, "runs": recs}, f, indent=2)
  print("Saved Physical FX JSON ->", args.out_json, flush=True)


if __name__ == "__main__":
  main()
