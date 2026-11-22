"""Wave 6 Physical Optics & Subsurface Transport on 4-Chip TPU v5p:
  (1) 6-Band Hero-Wavelength Spectral Dispersion (Cauchy n(lambda) = 1.50 + 0.015 / lambda_um^2,
      CIE 1931 XYZ -> linear sRGB) through a crystal glass prism + brilliant-cut diamond gem.
  (2) Volumetric Random-Walk Subsurface Scattering (SSS) on sculpted translucent emerald jade
      trefoil knot & warm amber beeswax fluted vessel (Chiang et al. albedo inversion, HG phase).

Pure JAX + Mosaic Pallas `pbr_raycast_tpu(..., interpret=False)`, zero host callbacks,
zero send/recv ops, seamless 4-chip shard_map, joint A-Trous denoiser, and AgX tonemapping.
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
try:
  from jax import shard_map
except ImportError:
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
    build_pbr_wide_bvh,
    pbr_raycast_tpu,
)
from complex_scenes_pt import _look_at_mat
from physical_fx_pt import (
    CinemaMeshBuilder,
    _henyey_greenstein,
    _tpu_joint_atrous_denoise,
    stitch_strip,
)


# ==============================================================================
# Spectral Optics: 6-Band Hero Wavelength + Cauchy Dispersion + CIE XYZ -> sRGB
# ==============================================================================

LAMBDA_MIN_NM = 420.0
LAMBDA_MAX_NM = 680.0
N_SPECTRAL_BANDS = 6

# IEC 61966-2-1 CIE XYZ -> linear sRGB matrix
_M_XYZ_TO_SRGB = np.array([
    [3.2404542, -1.5371385, -0.4985314],
    [-0.9692660, 1.8760108, 0.0415560],
    [0.0556434, -0.2040259, 1.0572252],
], dtype=np.float32)


def _cie_xyz_analytic_np(lam_nm: np.ndarray) -> np.ndarray:
  """Wyman-Sloan-Shirley (2013) piecewise Gaussian CIE 1931 2-deg CMF."""
  t1 = (lam_nm - 442.0) * np.where(lam_nm < 442.0, 0.0624, 0.0374)
  t2 = (lam_nm - 599.8) * np.where(lam_nm < 599.8, 0.0264, 0.0323)
  t3 = (lam_nm - 501.1) * np.where(lam_nm < 501.1, 0.0490, 0.0382)
  x = 0.362 * np.exp(-0.5 * t1 * t1) + 1.056 * np.exp(-0.5 * t2 * t2) - 0.065 * np.exp(-0.5 * t3 * t3)
  u1 = (lam_nm - 568.8) * np.where(lam_nm < 568.8, 0.0213, 0.0247)
  u2 = (lam_nm - 530.9) * np.where(lam_nm < 530.9, 0.0613, 0.0322)
  y = 0.821 * np.exp(-0.5 * u1 * u1) + 0.286 * np.exp(-0.5 * u2 * u2)
  v1 = (lam_nm - 437.0) * np.where(lam_nm < 437.0, 0.0845, 0.0278)
  v2 = (lam_nm - 459.0) * np.where(lam_nm < 459.0, 0.0385, 0.0725)
  z = 1.217 * np.exp(-0.5 * v1 * v1) + 0.681 * np.exp(-0.5 * v2 * v2)
  return np.stack([x, y, z], axis=-1).astype(np.float32)


# Precompute D65 / equal-energy white-point normalization over [420, 680] nm
_LAM_GRID = np.linspace(LAMBDA_MIN_NM, LAMBDA_MAX_NM, 1024, dtype=np.float32)
_RGB_GRID = np.maximum(_cie_xyz_analytic_np(_LAM_GRID) @ _M_XYZ_TO_SRGB.T, 0.0)
_SPEC_NORM_RGB = (1.0 / np.mean(_RGB_GRID, axis=0)).astype(np.float32)


def _spectral_srgb_weight_jax(lam_nm: jax.Array) -> jax.Array:
  """CIE 1931 XYZ -> linear sRGB normalized so equal-energy [420..680nm] integrates to (1,1,1)."""
  t1 = (lam_nm - 442.0) * jnp.where(lam_nm < 442.0, 0.0624, 0.0374)
  t2 = (lam_nm - 599.8) * jnp.where(lam_nm < 599.8, 0.0264, 0.0323)
  t3 = (lam_nm - 501.1) * jnp.where(lam_nm < 501.1, 0.0490, 0.0382)
  x = 0.362 * jnp.exp(-0.5 * t1 * t1) + 1.056 * jnp.exp(-0.5 * t2 * t2) - 0.065 * jnp.exp(-0.5 * t3 * t3)
  u1 = (lam_nm - 568.8) * jnp.where(lam_nm < 568.8, 0.0213, 0.0247)
  u2 = (lam_nm - 530.9) * jnp.where(lam_nm < 530.9, 0.0613, 0.0322)
  y = 0.821 * jnp.exp(-0.5 * u1 * u1) + 0.286 * jnp.exp(-0.5 * u2 * u2)
  v1 = (lam_nm - 437.0) * jnp.where(lam_nm < 437.0, 0.0845, 0.0278)
  v2 = (lam_nm - 459.0) * jnp.where(lam_nm < 459.0, 0.0385, 0.0725)
  z = 1.217 * jnp.exp(-0.5 * v1 * v1) + 0.681 * jnp.exp(-0.5 * v2 * v2)
  xyz = jnp.stack([x, y, z], axis=-1)
  m = jnp.asarray(_M_XYZ_TO_SRGB, dtype=jnp.float32)
  norm = jnp.asarray(_SPEC_NORM_RGB, dtype=jnp.float32)
  rgb = jnp.einsum("...j,ij->...i", xyz, m)
  return jnp.maximum(rgb, 0.0) * norm


def _cauchy_ior(lam_nm: jax.Array, dispersion: bool) -> jax.Array:
  """Cauchy glass dispersion n(lambda) = 1.50 + 0.015 / lambda_um^2 (or 1.55 baseline)."""
  if not dispersion:
    return jnp.full_like(lam_nm, 1.55)
  lam_um = lam_nm * 1e-3
  return 1.50 + 0.015 / (lam_um * lam_um)


@dataclasses.dataclass(frozen=True)
class DispersionConfig:
  enabled: bool = False          # True = 6-band Cauchy dispersion; False = constant n=1.55 baseline
  caustics: bool = True
  grid_res: int = 768
  n_photons_h: int = 256
  n_photons_w: int = 512         # per band -> 6 * 131,072 = 786,432 photons total
  bounds_xy: Tuple[float, float, float, float] = (-1.50, 1.50, -1.50, 1.50)
  prism_power: float = 95.0
  gem_power: float = 36.0


@dataclasses.dataclass(frozen=True)
class SSSConfig:
  enabled: bool = False          # True = volumetric random-walk SSS; False = opaque PBR baseline
  n_walk_steps: int = 16
  g_sss: float = 0.35


# ==============================================================================
# Forward 6-Band Spectral Caustic Photon Tracer (Prism Slit Beam + Gem Spot)
# ==============================================================================

def _trace_spectral_caustic_map_tpu(
    l2r: jax.Array,
    l2: jax.Array,
    l0: jax.Array,
    lf: jax.Array,
    cfg: DispersionConfig,
    rng_key: jax.Array,
) -> jax.Array:
  """Traces 6 stratified spectral bands (420..680 nm) through crystal prism & faceted diamond."""
  CH, CW = cfg.n_photons_h, cfg.n_photons_w
  n_photons = CH * CW
  n_prism = n_photons // 2
  n_gem = n_photons - n_prism

  bx0, bx1, by0, by1 = cfg.bounds_xy
  G = cfg.grid_res
  cell_area = ((bx1 - bx0) / float(G)) * ((by1 - by0) / float(G))

  # Stratified 2D grid indices
  gx = (jnp.arange(CW, dtype=jnp.float32) + 0.5) / float(CW)
  gy = (jnp.arange(CH, dtype=jnp.float32) + 0.5) / float(CH)
  ug, vg = jnp.meshgrid(gx, gy, indexing="xy")
  u_flat = ug.reshape(-1)
  v_flat = vg.reshape(-1)

  # Beam 1 (first half of photons): collimated white slit ribbon aimed at 60-deg crystal prism
  # Pitched downward at -10.5 deg so lower edge marks incoming white beam before prism,
  # and upper ribbon refracts through prism to cast a sweeping rainbow across the dark slate.
  pitch = -11.5 * (math.pi / 180.0)
  d_prism = jnp.array([math.cos(pitch), 0.0, math.sin(pitch)], dtype=jnp.float32)
  # Beam 2 (second half): focused white optical spot onto the brilliant-cut faceted diamond gem
  gem_src = jnp.array([-0.18, 0.85, 0.92], dtype=jnp.float32)
  gem_center = jnp.array([0.46, 0.28, 0.11], dtype=jnp.float32)

  band_w = (LAMBDA_MAX_NM - LAMBDA_MIN_NM) / float(N_SPECTRAL_BANDS)

  def _band_pass(b_idx, cmap_acc):
    b_rng = jax.random.fold_in(rng_key, b_idx)
    k_lam, k_jit, k_gem, k_bnc = jax.random.split(b_rng, 4)

    # Continuous stratified wavelength inside band b_idx -> smooth continuous spectrum
    u_lam = jax.random.uniform(k_lam, (1, n_photons), minval=0.0, maxval=1.0, dtype=jnp.float32)
    lam_nm = LAMBDA_MIN_NM + (jnp.asarray(b_idx, jnp.float32) + u_lam) * band_w
    ior_band = _cauchy_ior(lam_nm, cfg.enabled)
    spec_rgb = _spectral_srgb_weight_jax(lam_nm) / float(N_SPECTRAL_BANDS)

    jit = jax.random.uniform(k_jit, (n_photons, 2), minval=-0.5, maxval=0.5, dtype=jnp.float32)
    u_j = jnp.clip(u_flat + jit[:, 0] / float(CW), 0.0, 1.0)
    v_j = jnp.clip(v_flat + jit[:, 1] / float(CH), 0.0, 1.0)

    # Prism slit origin plane at x = -0.97 (clear in front of collimator x_max = -0.988).
    # Note: [:n_prism] takes the lower half of meshgrid rows (v_j in [0, 0.5]), so scale v_j * 2.0 to span [0, 1]!
    v_prism = jnp.clip(v_j[:n_prism] * 2.0, 0.0, 1.0)
    ro_p = jnp.stack([
        jnp.full((n_prism,), -0.97, dtype=jnp.float32),
        0.06 + 0.12 * u_j[:n_prism],
        0.040 + 0.320 * v_prism,
    ], axis=-1)
    rd_p = jnp.broadcast_to(d_prism[None, :], (n_prism, 3))

    # Gem spot rays from gem_src aimed uniformly over disk around gem_center (r = 0.21)
    g_u = jax.random.uniform(k_gem, (n_gem,), minval=0.0, maxval=1.0, dtype=jnp.float32)
    g_v = jax.random.uniform(jax.random.fold_in(k_gem, 1), (n_gem,), minval=0.0, maxval=1.0, dtype=jnp.float32)
    gr = 0.21 * jnp.sqrt(g_u)
    gphi = (2.0 * math.pi) * g_v
    g_tgt = gem_center[None, :] + jnp.stack([gr * jnp.cos(gphi), gr * jnp.sin(gphi), jnp.zeros_like(gr)], axis=-1)
    ro_g = jnp.broadcast_to(gem_src[None, :], (n_gem, 3))
    rd_g = g_tgt - ro_g
    rd_g = rd_g / jnp.maximum(jnp.linalg.norm(rd_g, axis=-1, keepdims=True), 1e-12)

    ro0 = jnp.concatenate([ro_p, ro_g], axis=0)[None, :, :]
    rd0 = jnp.concatenate([rd_p, rd_g], axis=0)[None, :, :]

    is_prism_ph = jnp.concatenate([
        jnp.ones((n_prism,), dtype=jnp.bool_),
        jnp.zeros((n_gem,), dtype=jnp.bool_),
    ])[None, :]

    prism_beam_area = 0.12 * 0.320
    gem_beam_area = math.pi * (0.21 ** 2)
    flux_p = cfg.prism_power * (prism_beam_area / float(n_prism)) / max(cell_area, 1e-6)
    flux_g = cfg.gem_power * (gem_beam_area / float(n_gem)) / max(cell_area, 1e-6)
    flux_scale = jnp.where(is_prism_ph[..., None], flux_p, flux_g)
    flux0 = spec_rgb * flux_scale

    alive0 = jnp.ones((1, n_photons), dtype=jnp.bool_)
    refr0 = jnp.zeros((1, n_photons), dtype=jnp.bool_)
    t_lim0 = jnp.full((1, n_photons), 12.0, dtype=jnp.float32)

    def _photon_step(step_i, carry):
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

      # Deposit refracted caustics on tabletop; also let unrefracted prism slit beam mark the
      # incoming white beam ONLY before the prism (x < -0.35) so no stray unrefracted white patch appears!
      pre_prism_white = is_opaque & is_prism_ph & (~has_refr) & (hit_pos[..., 0] < -0.35)
      deposit = is_opaque & (n_ff[..., 2] > 0.25) & (has_refr | pre_prism_white)
      # Boost pre-prism incoming beam visibility slightly relative to dispersed rainbow
      dep_mult = jnp.where(has_refr, 1.0, 0.65)

      gx_f = ((hit_pos[0, :, 0] - bx0) / (bx1 - bx0)) * float(G - 1)
      gy_f = ((hit_pos[0, :, 1] - by0) / (by1 - by0)) * float(G - 1)
      in_bounds = deposit[0] & (gx_f >= 0.0) & (gx_f < float(G - 1)) & (gy_f >= 0.0) & (gy_f < float(G - 1))
      ix = jnp.clip(jnp.floor(gx_f).astype(jnp.int32), 0, G - 2)
      iy = jnp.clip(jnp.floor(gy_f).astype(jnp.int32), 0, G - 2)
      fx = jnp.clip(gx_f - ix.astype(jnp.float32), 0.0, 1.0)
      fy = jnp.clip(gy_f - iy.astype(jnp.float32), 0.0, 1.0)
      dep_flux = jnp.where(in_bounds[:, None], flux[0] * dep_mult[0, :, None], 0.0)

      cmap = cmap.at[iy, ix].add(dep_flux * ((1.0 - fx) * (1.0 - fy))[:, None])
      cmap = cmap.at[iy, ix + 1].add(dep_flux * (fx * (1.0 - fy))[:, None])
      cmap = cmap.at[iy + 1, ix].add(dep_flux * ((1.0 - fx) * fy)[:, None])
      cmap = cmap.at[iy + 1, ix + 1].add(dep_flux * (fx * fy)[:, None])

      # Subtle crystal absorption inside glass
      sigma_a = -jnp.log(jnp.clip(alb, 0.05, 0.998)) / 0.85
      beer = jnp.where((is_glass & (~entering))[..., None], jnp.exp(-sigma_a * jnp.maximum(d_hit, 0.0)[..., None]), 1.0)

      eta = jnp.where(entering, 1.0 / ior_band, ior_band)
      cos_i = jnp.clip(jnp.sum(wo * n_ff, axis=-1), 0.0, 1.0)
      sin2_t = eta * eta * (1.0 - cos_i * cos_i)
      tir = sin2_t > 1.0
      r0_g = ((1.0 - eta) / (1.0 + eta)) ** 2
      fresnel_g = r0_g + (1.0 - r0_g) * ((1.0 - cos_i) ** 5)

      u_f = jax.random.uniform(jax.random.fold_in(k_bnc, step_i), (1, n_photons), minval=0.0, maxval=1.0, dtype=jnp.float32)
      do_refl = tir | (u_f < fresnel_g)
      refl_dir = 2.0 * cos_i[..., None] * n_ff - wo
      cos_t = jnp.sqrt(jnp.maximum(1.0 - sin2_t, 0.0))
      refr_dir = eta[..., None] * rd + (eta * cos_i - cos_t)[..., None] * n_ff
      wi_next = jnp.where(do_refl[..., None], refl_dir, refr_dir)
      wi_next = wi_next / jnp.maximum(jnp.linalg.norm(wi_next, axis=-1, keepdims=True), 1e-12)
      new_ro = jnp.where(do_refl[..., None], hit_pos + n_ff * 1.5e-3, hit_pos - n_ff * 1.5e-3)

      new_flux = flux * beer
      new_alive = is_glass & (jnp.max(new_flux, axis=-1) > 1e-5)
      new_refr = has_refr | (is_glass & (~do_refl))
      return (new_ro, wi_next, new_flux, new_alive, new_refr, cmap)

    _, _, _, _, _, cmap_out = jax.lax.fori_loop(
        0, 6, _photon_step, (ro0, rd0, flux0, alive0, refr0, cmap_acc)
    )
    return cmap_out

  raw_cmap = jax.lax.fori_loop(
      0, N_SPECTRAL_BANDS, _band_pass, jnp.zeros((G, G, 3), dtype=jnp.float32)
  )

  # Gentle 3x3 separable anti-aliasing filter preserving razor-sharp spectral dispersion
  k1d = jnp.array([0.14, 0.72, 0.14], dtype=jnp.float32)
  k2d = jnp.outer(k1d, k1d)[None, None, :, :]
  k_rgb = jnp.broadcast_to(k2d, (3, 1, 3, 3))
  cmap_nchw = jnp.transpose(raw_cmap, (2, 0, 1))[None, ...]
  smooth = jax.lax.conv_general_dilated(
      cmap_nchw, k_rgb, window_strides=(1, 1), padding="SAME", feature_group_count=3
  )
  return jnp.transpose(smooth[0], (1, 2, 0))


def _lookup_caustic_map(cmap: jax.Array, hit_pos: jax.Array, cfg: DispersionConfig) -> jax.Array:
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


# ==============================================================================
# Volumetric Random-Walk Subsurface Scattering (SSS) Helpers
# ==============================================================================

def _chiang_single_scatter_albedo(alb: jax.Array) -> jax.Array:
  """Chiang et al. (2016) diffusion inversion mapping desired diffuse albedo -> single-scatter alpha."""
  a = jnp.clip(alb, 0.01, 0.995)
  return 1.0 - jnp.exp(-5.09406 * a + 2.61188 * a * a - 4.31805 * a * a * a)


def _sample_hg_dir(u1: jax.Array, u2: jax.Array, rd: jax.Array, g: float) -> jax.Array:
  """Sample Henyey-Greenstein phase function direction around unit vector `rd`."""
  if abs(g) < 1e-3:
    cos_t = 1.0 - 2.0 * u1
  else:
    sqr = (1.0 - g * g) / jnp.maximum(1.0 - g + 2.0 * g * u1, 1e-4)
    cos_t = (1.0 + g * g - sqr * sqr) / (2.0 * g)
  cos_t = jnp.clip(cos_t, -1.0, 1.0)
  sin_t = jnp.sqrt(jnp.maximum(1.0 - cos_t * cos_t, 0.0))
  phi = 2.0 * math.pi * u2
  t1, t2 = _build_onb(rd)
  d = t1 * (sin_t * jnp.cos(phi))[..., None] + t2 * (sin_t * jnp.sin(phi))[..., None] + rd * cos_t[..., None]
  return d / jnp.maximum(jnp.linalg.norm(d, axis=-1, keepdims=True), 1e-12)


def _sss_params_from_mcode(mcode: jax.Array) -> Tuple[jax.Array, jax.Array]:
  """mcode 1.50 -> Emerald Jade (sigma_t=24.0, ior=1.52); 1.60 -> Amber Wax (sigma_t=17.0, ior=1.46)."""
  t_w = jnp.clip((mcode - 1.50) / 0.10, 0.0, 1.0)
  sigma_t = 24.0 * (1.0 - t_w) + 17.0 * t_w
  ior = 1.52 * (1.0 - t_w) + 1.46 * t_w
  return sigma_t, ior


# ==============================================================================
# Unified 4-Chip TPU Mosaic Path Tracer (Hero Spectral Dispersion + Random-Walk SSS)
# ==============================================================================

def build_dispersion_sss_pathtracer(
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
    disp_cfg: Optional[DispersionConfig] = None,
    sss_cfg: Optional[SSSConfig] = None,
):
  n_chips = len(devices)
  assert width % n_chips == 0 and (width // n_chips) % 16 == 0 and height % 8 == 0
  W_local = width // n_chips

  d_cfg = disp_cfg if disp_cfg is not None else DispersionConfig(enabled=False, caustics=False)
  s_cfg = sss_cfg if sss_cfg is not None else SSSConfig(enabled=False)

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
      s = s + (cos_s ** 48.0)[..., None] * sun_emit * 0.15
    return s

  aspect = float(width) / float(height)
  half_h = float(znear) * math.tan(0.5 * float(cam_fovy) * (math.pi / 180.0))
  half_w = half_h * aspect
  band_w = (LAMBDA_MAX_NM - LAMBDA_MIN_NM) / float(N_SPECTRAL_BANDS)

  def _eval_area_sun_nee(l2r, l2, l0, lf, pos, n_out, wo, alb, metallic, rgh, is_lambertian_exit, k_lu, k_lv, k_su, k_sv, shape_br):
    """Evaluate NEE against analytical area light + sun disk (used for opaque surfaces & SSS exits)."""
    spp_par, R = shape_br
    lu = jax.random.uniform(k_lu, (spp_par, R), minval=0.0, maxval=1.0, dtype=jnp.float32)
    lv = jax.random.uniform(k_lv, (spp_par, R), minval=0.0, maxval=1.0, dtype=jnp.float32)
    lr = area_r * jnp.sqrt(lu)
    lphi = (2.0 * math.pi) * lv
    lp = area_c[None, None, :] + (lr * jnp.cos(lphi))[..., None] * at1[None, None, :] + (lr * jnp.sin(lphi))[..., None] * at2[None, None, :]
    to_l = lp - pos
    dist_l = jnp.linalg.norm(to_l, axis=-1)
    wi_l = to_l / jnp.maximum(dist_l[..., None], 1e-12)
    cos_s = jnp.maximum(jnp.sum(n_out * wi_l, axis=-1), 0.0)
    cos_l = jnp.maximum(jnp.sum(area_n[None, None, :] * (-wi_l), axis=-1), 0.0)
    need_sh = (cos_s > 0.0) & (cos_l > 0.0)
    sh_o = pos + n_out * 2.0e-3
    sh_lim = jnp.where(need_sh, jnp.maximum(dist_l - 2.5e-3, 0.0), 0.0)
    d_sh, *_ = pbr_raycast_tpu(l2r, l2, l0, lf, sh_o, wi_l, sh_lim, height, W_local, any_hit=True)
    vis = jnp.where(need_sh & (d_sh >= sh_lim - 1e-4), 1.0, 0.0)
    if is_lambertian_exit:
      brdf_a = jnp.full((*shape_br, 3), 1.0 / math.pi, dtype=jnp.float32)
    else:
      brdf_a = _eval_pbr_brdf(n_out, wo, wi_l, alb, metallic, rgh)
    geom_a = (cos_s * cos_l) / (dist_l * dist_l + 0.08) * area_area
    nee_tot = jnp.minimum(brdf_a * area_emit[None, None, :] * (geom_a * vis)[..., None], 18.0)

    if light_rig.has_sun:
      su = jax.random.uniform(k_su, (spp_par, R), minval=0.0, maxval=1.0, dtype=jnp.float32)
      sv = jax.random.uniform(k_sv, (spp_par, R), minval=0.0, maxval=1.0, dtype=jnp.float32)
      sr = sun_sin_max * jnp.sqrt(su)
      sphi = (2.0 * math.pi) * sv
      wi_sun = sun_d[None, None, :] + (sr * jnp.cos(sphi))[..., None] * st1[None, None, :] + (sr * jnp.sin(sphi))[..., None] * st2[None, None, :]
      wi_sun = wi_sun / jnp.maximum(jnp.linalg.norm(wi_sun, axis=-1, keepdims=True), 1e-12)
      cos_sun = jnp.maximum(jnp.sum(n_out * wi_sun, axis=-1), 0.0)
      need_sun = cos_sun > 0.0
      sun_lim = jnp.where(need_sun, jnp.float32(zfar * 0.9), 0.0)
      d_sun, *_ = pbr_raycast_tpu(l2r, l2, l0, lf, sh_o, wi_sun, sun_lim, height, W_local, any_hit=True)
      vis_sun = jnp.where(need_sun & (d_sun >= sun_lim - 1e-2), 1.0, 0.0)
      if is_lambertian_exit:
        brdf_s = jnp.full((*shape_br, 3), 1.0 / math.pi, dtype=jnp.float32)
      else:
        brdf_s = _eval_pbr_brdf(n_out, wo, wi_sun, alb, metallic, rgh)
      nee_tot = nee_tot + jnp.minimum(brdf_s * sun_emit[None, None, :] * (cos_sun * vis_sun)[..., None], 22.0)

    return nee_tot

  def _render_local(l2r, l2, l0, lf, cxp, cxm, cid_arr, rng_in):
    cid = cid_arr[0]
    x_off = cid * W_local
    chip_rng = jax.random.fold_in(rng_in, cid)

    cmap = (
        _trace_spectral_caustic_map_tpu(l2r, l2, l0, lf, d_cfg, jax.random.fold_in(rng_in, 777))
        if d_cfg.caustics
        else jnp.zeros((1, 1, 3), dtype=jnp.float32)
    )

    local_px = height * W_local
    if s_cfg.enabled:
      spp_par = 4 if (spp >= 4 and local_px <= 163840) else 1
    else:
      spp_par = 8 if (spp >= 8 and local_px <= 163840) else (4 if (spp >= 4 and local_px <= 262144) else 1)
    spp_seq = max(1, spp // spp_par)
    R = height * W_local

    def _trace_batch_samples(s_idx, acc):
      rad_acc, nrm_acc, alb_acc, dep_acc, spc_acc = acc
      s_rng = jax.random.fold_in(chip_rng, s_idx)
      k_jit, k_lam, k_bnc = jax.random.split(s_rng, 3)

      # Hero wavelength per sample stratified over 6 bands [420..680 nm]
      band_idx = (s_idx * spp_par + jnp.arange(spp_par, dtype=jnp.int32)) % N_SPECTRAL_BANDS
      u_lam = jax.random.uniform(k_lam, (spp_par, R), minval=0.0, maxval=1.0, dtype=jnp.float32)
      lam_nm = LAMBDA_MIN_NM + (band_idx[:, None].astype(jnp.float32) + u_lam) * band_w
      hero_ior = _cauchy_ior(lam_nm, d_cfg.enabled)
      hero_spec_w = _spectral_srgb_weight_jax(lam_nm) if d_cfg.enabled else jnp.ones((spp_par, R, 3), dtype=jnp.float32)

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
      disp0 = jnp.zeros((spp_par, R), dtype=jnp.bool_)
      rad0 = jnp.zeros((spp_par, R, 3), dtype=jnp.float32)
      prim_d0 = jnp.full((spp_par, R), float(zfar), dtype=jnp.float32)
      prim_n0 = jnp.zeros((spp_par, R, 3), dtype=jnp.float32)
      prim_a0 = jnp.zeros((spp_par, R, 3), dtype=jnp.float32)
      prim_s0 = jnp.zeros((spp_par, R), dtype=jnp.float32)

      def _bounce(b_idx, carry):
        ro, rd, t_lim, thr, alive, has_disp, rad, prim_d, prim_n, prim_a, prim_s = carry
        b_rng = jax.random.fold_in(k_bnc, b_idx)
        k_lu, k_lv, k_su, k_sv, k_lobe, k_u1, k_u2, k_rw = jax.random.split(b_rng, 8)

        d_hit, nrm_raw, alb, rgh, mcode = pbr_raycast_tpu(
            l2r, l2, l0, lf, ro, rd, jnp.where(alive, t_lim, 0.0), height, W_local, any_hit=False
        )
        hit = alive & (d_hit < t_lim - 1e-4) & (d_hit >= 0.0)
        miss = alive & (~hit)
        prim_d = jnp.where(b_idx == 0, jnp.where(hit, d_hit, float(zfar)), prim_d)
        rad = rad + jnp.where(miss[..., None], thr * _env_sky(rd), 0.0)

        n_unit = nrm_raw / jnp.maximum(jnp.linalg.norm(nrm_raw, axis=-1, keepdims=True), 1e-12)
        wo = -rd
        entering = jnp.sum(n_unit * wo, axis=-1) >= 0.0
        n_ff = jnp.where(entering[..., None], n_unit, -n_unit)

        is_emit = hit & (mcode >= 2.8)
        is_glass = hit & (mcode >= 1.8) & (mcode < 2.8)
        if s_cfg.enabled:
          is_sss = hit & (mcode >= 1.35) & (mcode < 1.75)
          is_opaque = hit & (mcode < 1.35)
        else:
          is_sss = jnp.zeros_like(hit)
          is_opaque = hit & (mcode < 1.8)

        metallic = jnp.where(mcode < 1.35, jnp.clip(mcode, 0.0, 1.0), 0.0)
        is_spec_prim = is_glass | (hit & (~is_sss) & ((rgh < 0.14) | (metallic > 0.75)))
        prim_n = jnp.where(b_idx == 0, jnp.where(hit[..., None], n_ff, jnp.array([0.0, 0.0, 1.0], jnp.float32)), prim_n)
        prim_a = jnp.where(b_idx == 0, jnp.where(hit[..., None], alb, 0.0), prim_a)
        prim_s = jnp.where(b_idx == 0, jnp.where(is_spec_prim, 1.0, 0.0), prim_s)

        emit_power = jnp.where(mcode >= 2.8, (mcode - 2.0) * 6.0, 0.0)
        rad = rad + jnp.where(is_emit[..., None], thr * alb * emit_power[..., None], 0.0)

        hit_pos = ro + jnp.where(hit, d_hit, 0.0)[..., None] * rd

        # Spectral caustic map lookup on upward-facing diffuse tabletop
        if d_cfg.caustics:
          c_irr = _lookup_caustic_map(cmap, hit_pos, d_cfg)
          c_mask = is_opaque & (n_ff[..., 2] > 0.25)
          rad = rad + jnp.where(c_mask[..., None], thr * (alb / math.pi) * c_irr, 0.0)

        # Direct NEE on opaque surfaces (and glossy surface specular on SSS)
        nee_opq = _eval_area_sun_nee(
            l2r, l2, l0, lf, hit_pos, n_ff, wo, alb, metallic, rgh, False, k_lu, k_lv, k_su, k_sv, (spp_par, R)
        )
        # For SSS surface, only specular Fresnel highlight on outer coat; interior comes from random walk
        cos_i_surf = jnp.clip(jnp.sum(wo * n_ff, axis=-1), 0.0, 1.0)
        _, ior_sss = _sss_params_from_mcode(mcode)
        r0_sss = ((1.0 - ior_sss) / (1.0 + ior_sss)) ** 2
        F_sss = r0_sss + (1.0 - r0_sss) * ((1.0 - cos_i_surf) ** 5)
        nee_sss_spec = _eval_area_sun_nee(
            l2r, l2, l0, lf, hit_pos, n_ff, wo, jnp.zeros_like(alb), jnp.zeros_like(metallic), rgh, False, k_lu, k_lv, k_su, k_sv, (spp_par, R)
        )
        rad = rad + jnp.where(is_opaque[..., None], thr * nee_opq, 0.0)
        rad = rad + jnp.where(is_sss[..., None], thr * nee_sss_spec, 0.0)

        # Sample next bounce for opaque surfaces
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

        # Dielectric Glass with Hero-Wavelength Cauchy Dispersion
        sigma_a = -jnp.log(jnp.clip(alb, 0.05, 0.998)) / 0.85
        beer_glass = jnp.where((is_glass & (~entering))[..., None], jnp.exp(-sigma_a * jnp.maximum(d_hit, 0.0)[..., None]), 1.0)
        eta = jnp.where(entering, 1.0 / hero_ior, hero_ior)
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

        # Apply hero wavelength spectral color once upon first refractive dispersion event
        first_disp = is_glass & (~do_refl) & (~has_disp)
        disp_mult = jnp.where(first_disp[..., None], hero_spec_w, 1.0)
        new_has_disp = has_disp | (is_glass & (~do_refl))

        # Volumetric Random-Walk Subsurface Scattering (SSS)
        if s_cfg.enabled:
          sss_refl = u_lobe < F_sss
          enter_sss = is_sss & (~sss_refl)

          eta_sss = 1.0 / ior_sss
          sin2_sss = eta_sss * eta_sss * (1.0 - cos_i_surf * cos_i_surf)
          cos_t_sss = jnp.sqrt(jnp.maximum(1.0 - sin2_sss, 0.0))
          rd_enter_sss = eta_sss[..., None] * rd + (eta_sss * cos_i_surf - cos_t_sss)[..., None] * n_ff
          rd_enter_sss = rd_enter_sss / jnp.maximum(jnp.linalg.norm(rd_enter_sss, axis=-1, keepdims=True), 1e-12)

          sigma_t_sss, _ = _sss_params_from_mcode(mcode)
          alpha_sss = _chiang_single_scatter_albedo(alb) * 0.96

          ro_rw0 = hit_pos - n_ff * 1.5e-3
          rd_rw0 = rd_enter_sss
          thr_rw0 = jnp.ones((spp_par, R, 3), dtype=jnp.float32)
          exited0 = jnp.zeros((spp_par, R), dtype=jnp.bool_)
          exit_pos0 = hit_pos
          exit_n0 = n_ff
          exit_dir0 = rd_enter_sss
          n_scat0 = jnp.zeros((spp_par, R), dtype=jnp.int32)

          def _rw_step(rw_i, rw_carry):
            ro_w, rd_w, thr_w, rw_act, ex_done, ex_p, ex_n, ex_d, n_sc = rw_carry
            step_rng = jax.random.fold_in(k_rw, rw_i)
            k_df, k_hg1, k_hg2 = jax.random.split(step_rng, 3)

            u_df = jnp.clip(jax.random.uniform(k_df, (spp_par, R), minval=1e-4, maxval=0.9999, dtype=jnp.float32), 1e-4, 0.9999)
            d_free = -jnp.log(u_df) / jnp.maximum(sigma_t_sss, 1e-3)

            d_bnd, n_bnd_raw, *_ = pbr_raycast_tpu(
                l2r, l2, l0, lf, ro_w, rd_w, jnp.where(rw_act, d_free, 0.0), height, W_local, any_hit=False
            )
            hit_bnd = rw_act & (d_bnd < d_free - 1e-4) & (d_bnd >= 0.0)
            n_bnd = n_bnd_raw / jnp.maximum(jnp.linalg.norm(n_bnd_raw, axis=-1, keepdims=True), 1e-12)
            # Outward normal at exit boundary points into the same hemisphere as rd_w
            n_out = jnp.where(jnp.sum(n_bnd * rd_w, axis=-1, keepdims=True) >= 0.0, n_bnd, -n_bnd)
            bnd_p = ro_w + jnp.where(hit_bnd, d_bnd, 0.0)[..., None] * rd_w

            # Scatter inside medium if boundary wasn't hit
            scat_now = rw_act & (~hit_bnd)
            scat_p = ro_w + d_free[..., None] * rd_w
            u_h1 = jax.random.uniform(k_hg1, (spp_par, R), minval=0.0, maxval=1.0, dtype=jnp.float32)
            u_h2 = jax.random.uniform(k_hg2, (spp_par, R), minval=0.0, maxval=1.0, dtype=jnp.float32)
            rd_scat = _sample_hg_dir(u_h1, u_h2, rd_w, s_cfg.g_sss)

            new_ro_w = jnp.where(scat_now[..., None], scat_p, bnd_p)
            new_rd_w = jnp.where(scat_now[..., None], rd_scat, rd_w)
            new_thr_w = jnp.where(scat_now[..., None], thr_w * alpha_sss, thr_w)
            new_rw_act = scat_now & (jnp.max(new_thr_w, axis=-1) > 1e-3)
            new_ex_done = ex_done | hit_bnd
            new_ex_p = jnp.where(hit_bnd[..., None], bnd_p, ex_p)
            new_ex_n = jnp.where(hit_bnd[..., None], n_out, ex_n)
            new_ex_d = jnp.where(hit_bnd[..., None], rd_w, ex_d)
            new_n_sc = n_sc + jnp.where(scat_now, 1, 0)
            return (new_ro_w, new_rd_w, new_thr_w, new_rw_act, new_ex_done, new_ex_p, new_ex_n, new_ex_d, new_n_sc)

          _, _, thr_rw_f, rw_still_in, ex_done_f, ex_p_f, ex_n_f, ex_d_f, n_sc_f = jax.lax.fori_loop(
              0, s_cfg.n_walk_steps, _rw_step,
              (ro_rw0, rd_rw0, thr_rw0, enter_sss, exited0, exit_pos0, exit_n0, exit_dir0, n_scat0)
          )

          # Evaluate exit NEE to backlight + rim light for rays that exited the SSS volume
          k_ex1, k_ex2, k_ex3, k_ex4, k_ex5, k_ex6 = jax.random.split(jax.random.fold_in(k_rw, 999), 6)
          ex_nee = _eval_area_sun_nee(
              l2r, l2, l0, lf, ex_p_f, ex_n_f, -ex_d_f, jnp.ones_like(alb), jnp.zeros_like(metallic), rgh,
              True, k_ex1, k_ex2, k_ex3, k_ex4, (spp_par, R)
          )
          sss_valid_exit = enter_sss & ex_done_f
          rad = rad + jnp.where(sss_valid_exit[..., None], thr * thr_rw_f * ex_nee, 0.0)

          # Continue exiting ray for ambient/tabletop indirect bounce (cosine hemisphere or Snell)
          u_e1 = jax.random.uniform(k_ex5, (spp_par, R), minval=0.0, maxval=1.0, dtype=jnp.float32)
          u_e2 = jax.random.uniform(k_ex6, (spp_par, R), minval=0.0, maxval=1.0, dtype=jnp.float32)
          wi_sss_exit = _sample_cosine_hemisphere(u_e1, u_e2, ex_n_f)
          # If ray reflected off glossy SSS coat, bounce off specularly; if exited SSS, continue from exit
          wi_sss = jnp.where(sss_refl[..., None], wi_spec, wi_sss_exit)
          ro_sss = jnp.where(sss_refl[..., None], hit_pos + n_ff * 1.5e-3, ex_p_f + ex_n_f * 1.5e-3)
          w_sss = jnp.where(sss_refl[..., None], jnp.ones_like(thr_rw_f), thr_rw_f)
          sss_cont = (is_sss & sss_refl) | sss_valid_exit

          wi_next = jnp.where(is_sss[..., None], wi_sss, jnp.where(is_glass[..., None], wi_glass, wi_opq))
          new_ro = jnp.where(is_sss[..., None], ro_sss, jnp.where(is_glass[..., None], ro_glass, hit_pos + n_ff * 1.5e-3))
          bounce_w = jnp.where(is_sss[..., None], w_sss, jnp.where(is_glass[..., None], beer_glass * disp_mult, w_opq))
          can_cont = jnp.where(is_sss, sss_cont, is_opaque | is_glass)
        else:
          wi_next = jnp.where(is_glass[..., None], wi_glass, wi_opq)
          new_ro = jnp.where(is_glass[..., None], ro_glass, hit_pos + n_ff * 1.5e-3)
          bounce_w = jnp.where(is_glass[..., None], beer_glass * disp_mult, w_opq)
          can_cont = is_opaque | is_glass

        new_thr = thr * bounce_w
        new_alive = can_cont & (jnp.max(new_thr, axis=-1) > 1e-3)
        new_t_lim = jnp.broadcast_to(jnp.float32(zfar), (spp_par, R))
        return (new_ro, wi_next, new_t_lim, new_thr, new_alive, new_has_disp, rad, prim_d, prim_n, prim_a, prim_s)

      _, _, _, _, _, _, final_rad, prim_d, prim_n, prim_a, prim_s = jax.lax.fori_loop(
          0, max_bounces, _bounce,
          (ro0, rd0, t_max0, thr0, alive0, disp0, rad0, prim_d0, prim_n0, prim_a0, prim_s0),
      )

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
    full = sh_out.transpose(1, 0, 2, 3).reshape(height, width, 11)
    rad_lin = full[..., 0:3]
    nrm_g = full[..., 3:6]
    nrm_g = nrm_g / jnp.maximum(jnp.linalg.norm(nrm_g, axis=-1, keepdims=True), 1e-6)
    alb_g = full[..., 6:9]
    dep_g = full[..., 9]
    spc_g = full[..., 10]
    rad_clean = _tpu_joint_atrous_denoise(rad_lin, nrm_g, alb_g, dep_g, spc_g, sigma_l=0.32, n_passes=3)
    return _agx_tonemap_srgb(rad_clean, exposure=light_rig.exposure)

  return render_fn, mesh, rep


# ==============================================================================
# Procedural High-Res Crystal Prism, Brilliant-Cut Gem, & SSS Sculptures
# ==============================================================================

def _ensure_outward_winding(v: np.ndarray, f: np.ndarray) -> np.ndarray:
  """Ensure closed manifold mesh has positive signed volume (outward normals)."""
  v0, v1, v2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
  vol6 = np.sum(v0 * np.cross(v1, v2))
  if vol6 < 0.0:
    return f[:, [0, 2, 1]].copy()
  return f


def _add_equilateral_prism(
    mb: CinemaMeshBuilder,
    center_xy: Tuple[float, float],
    radius: float,
    height: float,
    rot_z_deg: float,
    albedo: Tuple[float, float, float] = (0.985, 0.992, 1.0),
    mat_code: float = 2.0,
):
  """60-deg equilateral crystal glass prism with exact flat faces & outward normals."""
  cx, cy = center_xy
  # Place entry face outward normal at 144 deg and exit face outward normal at 24 deg:
  # That means vertex 0 is at 84 deg (top apex), vertex 1 at 204 deg, vertex 2 at 324 deg!
  angles_deg = [84.0 + rot_z_deg, 204.0 + rot_z_deg, 324.0 + rot_z_deg]
  pts = [(cx + radius * math.cos(math.radians(a)), cy + radius * math.sin(math.radians(a))) for a in angles_deg]
  v = np.array([
      [pts[0][0], pts[0][1], 0.0],
      [pts[1][0], pts[1][1], 0.0],
      [pts[2][0], pts[2][1], 0.0],
      [pts[0][0], pts[0][1], height],
      [pts[1][0], pts[1][1], height],
      [pts[2][0], pts[2][1], height],
  ], dtype=np.float32)
  f = np.array([
      [0, 2, 1],
      [3, 4, 5],
      [0, 1, 4], [0, 4, 3],
      [1, 2, 5], [1, 5, 4],
      [2, 0, 3], [2, 3, 5],
  ], dtype=np.int32)
  f = _ensure_outward_winding(v, f)
  mb.add_mesh(v, f, albedo, roughness=0.02, mat_code=mat_code, smooth=False)


def _add_brilliant_diamond(
    mb: CinemaMeshBuilder,
    center_xy: Tuple[float, float],
    radius: float,
    z0: float = 0.0,
    albedo: Tuple[float, float, float] = (0.98, 0.99, 1.0),
    mat_code: float = 2.0,
    n_facets: int = 16,
):
  """Faceted brilliant-cut gemstone with flat table, crown star/bezel facets, girdle, and steep TIR pavilion."""
  cx, cy = center_xy
  r_girdle = radius
  r_crown_mid = radius * 0.78
  r_table = radius * 0.54
  z_culet = z0 + 0.004
  z_girdle_bot = z0 + radius * 0.96
  z_girdle_top = z0 + radius * 1.04
  z_crown_mid = z0 + radius * 1.24
  z_table = z0 + radius * 1.38

  verts: List[List[float]] = []
  verts.append([cx, cy, z_culet])  # 0: culet
  idx_gb = len(verts)
  for j in range(n_facets):
    a = 2.0 * math.pi * j / n_facets
    verts.append([cx + r_girdle * math.cos(a), cy + r_girdle * math.sin(a), z_girdle_bot])
  idx_gt = len(verts)
  for j in range(n_facets):
    a = 2.0 * math.pi * j / n_facets
    verts.append([cx + r_girdle * math.cos(a), cy + r_girdle * math.sin(a), z_girdle_top])
  idx_cm = len(verts)
  for j in range(n_facets // 2):
    a = 2.0 * math.pi * (2 * j + 1) / n_facets
    verts.append([cx + r_crown_mid * math.cos(a), cy + r_crown_mid * math.sin(a), z_crown_mid])
  idx_tb = len(verts)
  for j in range(n_facets // 2):
    a = 2.0 * math.pi * (2 * j) / n_facets
    verts.append([cx + r_table * math.cos(a), cy + r_table * math.sin(a), z_table])
  idx_tc = len(verts)
  verts.append([cx, cy, z_table])  # table center

  faces: List[List[int]] = []
  n_half = n_facets // 2
  # Pavilion facets
  for j in range(n_facets):
    jn = (j + 1) % n_facets
    faces.append([0, idx_gb + jn, idx_gb + j])
  # Girdle band
  for j in range(n_facets):
    jn = (j + 1) % n_facets
    faces.append([idx_gb + j, idx_gb + jn, idx_gt + j])
    faces.append([idx_gb + jn, idx_gt + jn, idx_gt + j])
  # Crown faceted star + kite triangulation
  for j in range(n_half):
    jn = (j + 1) % n_half
    g0 = idx_gt + 2 * j
    g1 = idx_gt + 2 * j + 1
    g2 = idx_gt + ((2 * j + 2) % n_facets)
    cm_p = idx_cm + ((j - 1) % n_half)
    cm_c = idx_cm + j
    t0 = idx_tb + j
    t1 = idx_tb + jn
    faces.append([g0, g1, cm_c])
    faces.append([g1, g2, cm_c])
    faces.append([g0, cm_c, t0])
    faces.append([g0, t0, cm_p])
    faces.append([cm_c, t1, t0])
    faces.append([idx_tc, t0, t1])

  v_arr = np.array(verts, dtype=np.float32)
  f_arr = _ensure_outward_winding(v_arr, np.array(faces, dtype=np.int32))
  mb.add_mesh(v_arr, f_arr, albedo, roughness=0.02, mat_code=mat_code, smooth=False)


def _add_fluted_trefoil_knot(
    mb: CinemaMeshBuilder,
    center: Tuple[float, float, float],
    scale: float = 0.22,
    tube_r: float = 0.088,
    albedo: Tuple[float, float, float] = (0.12, 0.90, 0.44),
    roughness: float = 0.09,
    mat_code: float = 1.50,
    n_u: int = 192,
    n_v: int = 44,
):
  """High-res sculpted trefoil (2,3) torus-knot with thin fluted fins & thick lobes for dramatic SSS."""
  cx, cy, cz = center
  u_vals = np.linspace(0.0, 2.0 * math.pi, n_u, endpoint=False)
  v_vals = np.linspace(0.0, 2.0 * math.pi, n_v, endpoint=False)

  def _knot_pos(u):
    # Trefoil (2,3) curve oriented upright facing camera (-Y)
    x = np.sin(u) + 2.0 * np.sin(2.0 * u)
    z = np.cos(u) - 2.0 * np.cos(2.0 * u)
    y = -np.sin(3.0 * u) * 0.85
    return np.stack([x, y, z], axis=-1) * scale

  eps = 1e-4
  p = _knot_pos(u_vals)
  dp = _knot_pos(u_vals + eps) - _knot_pos(u_vals - eps)
  T = dp / np.maximum(np.linalg.norm(dp, axis=-1, keepdims=True), 1e-12)
  up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
  N = np.cross(T, up[None, :])
  N = N / np.maximum(np.linalg.norm(N, axis=-1, keepdims=True), 1e-12)
  B = np.cross(T, N)

  verts = []
  for iu in range(n_u):
    u = u_vals[iu]
    # Sculpted 3-fin cross-section: thick rounded core + thin translucent sculpted edges!
    fin_mod = 1.0 + 0.42 * np.cos(3.0 * (v_vals + 2.0 * u))
    lobe_mod = 0.78 + 0.24 * np.cos(3.0 * u)
    r_local = tube_r * fin_mod * lobe_mod
    for iv in range(n_v):
      v_ang = v_vals[iv]
      pt = p[iu] + r_local[iv] * (math.cos(v_ang) * N[iu] + math.sin(v_ang) * B[iu])
      verts.append([cx + pt[0], cy + pt[1], cz + pt[2]])

  faces = []
  for iu in range(n_u):
    iun = (iu + 1) % n_u
    for iv in range(n_v):
      ivn = (iv + 1) % n_v
      a = iu * n_v + iv
      b = iun * n_v + iv
      c = iun * n_v + ivn
      d = iu * n_v + ivn
      faces.append([a, b, c])
      faces.append([a, c, d])

  v_arr = np.array(verts, dtype=np.float32)
  f_arr = _ensure_outward_winding(v_arr, np.array(faces, dtype=np.int32))
  mb.add_mesh(v_arr, f_arr, albedo, roughness=roughness, mat_code=mat_code, smooth=True)


def _add_amber_wax_vessel(
    mb: CinemaMeshBuilder,
    center_xy: Tuple[float, float],
    z0: float = 0.0,
    albedo: Tuple[float, float, float] = (0.98, 0.56, 0.14),
    roughness: float = 0.22,
    mat_code: float = 1.60,
    n_seg: int = 80,
):
  """Closed manifold translucent fluted beeswax vessel with thin glowing upper rim."""
  cx, cy = center_xy
  # Outer bottom -> flared fluted belly -> thin neck & undulating rim -> inner wall -> inner floor
  profile = [
      (0.000, 0.000), (0.145, 0.000), (0.165, 0.060), (0.215, 0.190),
      (0.225, 0.330), (0.190, 0.470), (0.165, 0.580), (0.175, 0.650),
      (0.158, 0.660), (0.146, 0.642), (0.164, 0.555), (0.186, 0.435),
      (0.184, 0.285), (0.142, 0.130), (0.090, 0.055), (0.000, 0.045),
  ]
  verts = [[cx, cy, z0 + profile[0][1]]]
  ring_starts = []
  for r, z in profile[1:-1]:
    ring_starts.append(len(verts))
    for j in range(n_seg):
      phi = 2.0 * math.pi * j / n_seg
      # Gentle 8-fold spiral fluting on the outer/upper wax body
      flute = 1.0 + 0.09 * math.sin(8.0 * phi + 5.0 * z) * min(z / 0.35, 1.0)
      rf = r * flute
      verts.append([cx + rf * math.cos(phi), cy + rf * math.sin(phi), z0 + z])
  top_idx = len(verts)
  verts.append([cx, cy, z0 + profile[-1][1]])

  faces = []
  r0 = ring_starts[0]
  for j in range(n_seg):
    jn = (j + 1) % n_seg
    faces.append([0, r0 + jn, r0 + j])
  for k in range(len(ring_starts) - 1):
    a0, b0 = ring_starts[k], ring_starts[k + 1]
    for j in range(n_seg):
      jn = (j + 1) % n_seg
      faces.append([a0 + j, a0 + jn, b0 + j])
      faces.append([a0 + jn, b0 + jn, b0 + j])
  rl = ring_starts[-1]
  for j in range(n_seg):
    jn = (j + 1) % n_seg
    faces.append([rl + j, rl + jn, top_idx])

  v_arr = np.array(verts, dtype=np.float32)
  f_arr = _ensure_outward_winding(v_arr, np.array(faces, dtype=np.int32))
  mb.add_mesh(v_arr, f_arr, albedo, roughness=roughness, mat_code=mat_code, smooth=True)


# ==============================================================================
# Scene Builders
# ==============================================================================

def build_dispersion_prism_scene():
  """Dark Slate Optical Studio: Collimated white slit beam -> 60-deg Crystal Prism + Diamond Gem."""
  mb = CinemaMeshBuilder()

  # Seamless dark brushed slate studio tabletop (z <= 0)
  mb.add_box((0.0, 0.0, -0.05), (2.2, 2.2, 0.03), (0.09, 0.095, 0.105), roughness=0.36, mat_code=0.0)
  n_t = 8
  for ix in range(n_t):
    for iy in range(n_t):
      tx = -1.6 + (ix + 0.5) * (3.2 / n_t)
      ty = -1.6 + (iy + 0.5) * (3.2 / n_t)
      col = (0.095, 0.10, 0.112) if ((ix + iy) % 2 == 0) else (0.115, 0.12, 0.132)
      mb.add_box((tx, ty, -0.02), (3.2 / n_t * 0.499, 3.2 / n_t * 0.499, 0.02), col, roughness=0.34, mat_code=0.0)

  # Dark studio enclosure walls (matte charcoal so rainbow caustics glow cleanly)
  mb.add_box((0.0, 1.95, 1.2), (2.2, 0.05, 1.3), (0.06, 0.065, 0.075), roughness=0.85, mat_code=0.0)
  mb.add_box((-1.95, 0.0, 1.2), (0.05, 2.2, 1.3), (0.06, 0.065, 0.075), roughness=0.85, mat_code=0.0)
  mb.add_box((1.95, 0.0, 1.2), (0.05, 2.2, 1.3), (0.06, 0.065, 0.075), roughness=0.85, mat_code=0.0)

  # Sleek anodized black optical slit-beam collimator on the left
  mb.add_box((-1.12, 0.12, 0.20), (0.12, 0.14, 0.20), (0.08, 0.085, 0.095), roughness=0.28, mat_code=0.75)
  mb.add_box((-1.00, 0.12, 0.26), (0.012, 0.07, 0.12), (1.0, 0.99, 0.97), roughness=0.20, mat_code=3.3)

  # Hero 1: High-res 60-deg Crystal Glass Prism (center at (-0.28, 0.12))
  _add_equilateral_prism(mb, (-0.28, 0.12), radius=0.28, height=0.46, rot_z_deg=0.0, mat_code=2.0)

  # Hero 2: Brilliant-Cut Faceted Crystal Diamond Gem (center at (0.46, 0.28))
  _add_brilliant_diamond(mb, (0.46, 0.28), radius=0.20, z0=0.0, mat_code=2.0, n_facets=16)

  # Subtle brushed-brass calibration weight / mirror accent in back left
  mb.add_cylinder((-0.78, 0.84, 0.0), 0.10, 0.16, (0.86, 0.68, 0.30), roughness=0.16, mat_code=0.92, n_seg=48)

  bvh = mb.build("spectral_prism_dispersion_studio")
  cxp, cxm = _look_at_mat((-0.04, -2.02, 1.24), (0.02, -0.08, 0.08))
  rig = LightRig(
      area_center=(-0.10, 0.85, 1.95),
      area_normal=(0.08, -0.42, -0.90),
      area_radius=0.52,
      area_emit=(3.6, 3.9, 4.5),   # Low-key dark studio overhead softbox so rainbow caustics pop!
      has_sun=False,
      sky_zenith=(0.025, 0.03, 0.04),
      sky_horizon=(0.015, 0.018, 0.024),
      sky_ground=(0.008, 0.008, 0.010),
      exposure=1.22,
  )
  return bvh, cxp, cxm, 44.0, 0.05, 20.0, rig


def build_sss_sculpture_scene():
  """Backlit Dark Studio: Translucent Emerald Jade Trefoil Knot + Warm Amber Wax Vessel."""
  mb = CinemaMeshBuilder()

  # Dark matte slate studio floor & cyclorama backdrop
  mb.add_box((0.0, 0.0, -0.05), (2.5, 2.5, 0.05), (0.075, 0.08, 0.09), roughness=0.42, mat_code=0.0)
  mb.add_box((0.0, 2.2, 1.4), (2.6, 0.05, 1.5), (0.055, 0.06, 0.072), roughness=0.85, mat_code=0.0)
  mb.add_box((-2.2, 0.0, 1.4), (0.05, 2.5, 1.5), (0.055, 0.06, 0.072), roughness=0.85, mat_code=0.0)
  mb.add_box((2.2, 0.0, 1.4), (0.05, 2.5, 1.5), (0.055, 0.06, 0.072), roughness=0.85, mat_code=0.0)

  # Sleek dark basalt plinths under the sculptures
  mb.add_cylinder((-0.36, 0.02, 0.0), 0.34, 0.06, (0.10, 0.105, 0.115), roughness=0.35, mat_code=0.1, n_seg=48)
  mb.add_cylinder((0.40, 0.08, 0.0), 0.24, 0.04, (0.10, 0.105, 0.115), roughness=0.35, mat_code=0.1, n_seg=48)

  # Hero 1: Sculpted Fluted Emerald Jade Trefoil Torus-Knot (mat_code = 1.50 -> Jade SSS)
  _add_fluted_trefoil_knot(
      mb, center=(-0.36, 0.02, 0.42), scale=0.145, tube_r=0.082,
      albedo=(0.04, 0.88, 0.28), roughness=0.09, mat_code=1.50, n_u=180, n_v=40,
  )

  # Hero 2: Translucent Fluted Warm Amber Wax Vessel (mat_code = 1.60 -> Amber Wax SSS)
  _add_amber_wax_vessel(
      mb, center_xy=(0.40, 0.08), z0=0.04,
      albedo=(0.98, 0.36, 0.025), roughness=0.22, mat_code=1.60, n_seg=80,
  )

  bvh = mb.build("backlit_sss_sculptures_studio")
  cxp, cxm = _look_at_mat((0.02, -1.78, 0.68), (0.02, 0.05, 0.38))
  rig = LightRig(
      area_center=(0.02, 1.32, 0.58),
      area_normal=(0.0, -0.98, -0.18),
      area_radius=0.55,
      area_emit=(6.5, 5.8, 5.0),     # Calibrated studio backlight for saturated emerald/amber SSS
      has_sun=True,
      sun_dir=(-0.10, 0.94, 0.32),     # Secondary crisp rim backlight from behind
      sun_cos_half_angle=0.996,
      sun_emit=(4.5, 4.0, 3.4),
      sky_zenith=(0.14, 0.16, 0.21),   # Soft studio fill so opaque baseline reads cleanly
      sky_horizon=(0.09, 0.10, 0.13),
      sky_ground=(0.02, 0.02, 0.025),
      exposure=1.02,
  )
  return bvh, cxp, cxm, 40.0, 0.05, 20.0, rig


# ==============================================================================
# Runner & Benchmark
# ==============================================================================

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
    disp_cfg: Optional[DispersionConfig] = None,
    sss_cfg: Optional[SSSConfig] = None,
):
  fn, mesh, rep = build_dispersion_sss_pathtracer(
      bvh, devices, H, W, fovy, znear, zfar, rig,
      spp=spp, max_bounces=max_bounces, disp_cfg=disp_cfg, sss_cfg=sss_cfg,
  )
  cxp = jax.device_put(cxp_np, rep)
  cxm = jax.device_put(cxm_np, rep)
  rng = jax.device_put(jax.random.PRNGKey(17), rep)
  hlo_info = audit_hlo_zero_copy(fn.lower(cxp, cxm, rng))

  tc0 = time.perf_counter()
  y = fn(cxp, cxm, rng)
  jax.block_until_ready(y)
  compile_s = time.perf_counter() - tc0

  n_steps = 1
  t0 = time.perf_counter()
  for it in range(n_steps):
    y = fn(cxp, cxm, jax.device_put(jax.random.PRNGKey(40 + it), rep))
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
      "dispersion": bool(disp_cfg and disp_cfg.enabled),
      "sss_random_walk": bool(sss_cfg and sss_cfg.enabled),
      "n_chips": len(devices),
      "compile_s": round(compile_s, 2),
      "ms": round(ms, 2),
      "images_per_sec": round(fps, 3),
      "samples_per_min": round(spm, 1),
      "hlo_zero_copy": True,
      "hlo_ops": hlo_info,
  }
  print("DISPERSION_SSS:", rec, flush=True)
  return rec, img


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--res", type=int, default=640)
  ap.add_argument("--disp_spp", type=int, default=192)
  ap.add_argument("--sss_spp", type=int, default=160)
  ap.add_argument("--out_json", default="out/metrics/v3_w7_wave6_dispersion_sss.json")
  ap.add_argument("--out_png", default="out/figures/v3_w7_dispersion_sss_fx.png")
  args = ap.parse_args()

  devs = jax.devices()
  assert len(devs) == 4, f"Expected 4 TPU chips, got {len(devs)}"
  H = W = args.res

  recs = []
  panels: List[Tuple[str, np.ndarray]] = []

  # Top Row: (1) No-Dispersion Glass Baseline vs (2) 6-Band Hero-Wavelength Cauchy Dispersion
  bvh_d, cxp_d, cxm_d, fovy_d, zn_d, zf_d, rig_d = build_dispersion_prism_scene()
  r_d_base, img_d_base = _run_config(
      "glass_no_dispersion_baseline", bvh_d, cxp_d, cxm_d, fovy_d, zn_d, zf_d, rig_d,
      devs, H, W, spp=args.disp_spp, max_bounces=6,
      disp_cfg=DispersionConfig(enabled=False, caustics=True),
  )
  recs.append(r_d_base)
  panels.append((f"No-Dispersion Glass Baseline (n=1.55 | {args.disp_spp} spp | {r_d_base['ms']:.0f}ms)", img_d_base))

  r_d_spec, img_d_spec = _run_config(
      "spectral_cauchy_prism_caustics", bvh_d, cxp_d, cxm_d, fovy_d, zn_d, zf_d, rig_d,
      devs, H, W, spp=args.disp_spp, max_bounces=6,
      disp_cfg=DispersionConfig(enabled=True, caustics=True),
  )
  recs.append(r_d_spec)
  panels.append((f"6-Band Spectral Cauchy Dispersion + Rainbow Caustics ({args.disp_spp} spp | {r_d_spec['ms']:.0f}ms)", img_d_spec))

  # Bottom Row: (3) Opaque Diffuse Baseline vs (4) Backlit Volumetric Random-Walk SSS
  bvh_s, cxp_s, cxm_s, fovy_s, zn_s, zf_s, rig_s = build_sss_sculpture_scene()
  r_s_base, img_s_base = _run_config(
      "sss_opaque_diffuse_baseline", bvh_s, cxp_s, cxm_s, fovy_s, zn_s, zf_s, rig_s,
      devs, H, W, spp=args.sss_spp, max_bounces=4,
      sss_cfg=SSSConfig(enabled=False),
  )
  recs.append(r_s_base)
  panels.append((f"Opaque PBR Baseline (Backlit Jade & Amber | {args.sss_spp} spp | {r_s_base['ms']:.0f}ms)", img_s_base))

  r_s_rw, img_s_rw = _run_config(
      "sss_backlit_random_walk_volumetric", bvh_s, cxp_s, cxm_s, fovy_s, zn_s, zf_s, rig_s,
      devs, H, W, spp=args.sss_spp, max_bounces=4,
      sss_cfg=SSSConfig(enabled=True, n_walk_steps=16, g_sss=0.35),
  )
  recs.append(r_s_rw)
  panels.append((f"Backlit Volumetric Random-Walk SSS (16-step RW | {args.sss_spp} spp | {r_s_rw['ms']:.0f}ms)", img_s_rw))

  # Compose razor-sharp 2x2 dark-cinema showcase PNG
  pad, hdr = 14, 42
  canvas = Image.new("RGB", (2 * W + 3 * pad, 2 * (H + hdr) + 3 * pad), (12, 14, 18))
  draw = ImageDraw.Draw(canvas)
  for idx, (title, im) in enumerate(panels):
    r, c = idx // 2, idx % 2
    x0 = pad + c * (W + pad)
    y0 = pad + r * (H + hdr + pad)
    u8 = np.clip(im * 255.0 + 0.5, 0, 255).astype(np.uint8)
    canvas.paste(Image.fromarray(u8), (x0, y0 + hdr))
    draw.text((x0 + 4, y0 + 13), title, fill=(240, 244, 250))

  os.makedirs(os.path.dirname(args.out_png) or ".", exist_ok=True)
  canvas.save(args.out_png)
  print("Saved Showcase PNG ->", args.out_png, canvas.size, flush=True)

  os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
  with open(args.out_json, "w") as f:
    json.dump({
        "experiment": "v3_w7_wave6_dispersion_sss",
        "backend": jax.default_backend(),
        "n_chips": len(devs),
        "spectral_bands": N_SPECTRAL_BANDS,
        "wavelength_range_nm": [LAMBDA_MIN_NM, LAMBDA_MAX_NM],
        "cauchy_formula": "n(lambda) = 1.50 + 0.015 / lambda_um^2",
        "runs": recs,
    }, f, indent=2)
  print("Saved Metrics JSON ->", args.out_json, flush=True)


if __name__ == "__main__":
  main()
