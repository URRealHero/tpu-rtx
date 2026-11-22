"""Wave 7 Physical Optics & Deep Specular Transport on 4-Chip TPU v5p:
  (1) Thin-Film Spectral Interference (Airy Iridescence) + Anisotropic Brushed Metal:
      - Floating iridescent soap bubbles (d = 280..620 nm variable-thickness Airy interference)
      - Anodized iridescent TiO2-on-Ti metallic sphere (complex-IOR Airy substrate interference)
      - Radial-brushed anisotropic gold & stainless steel discs (alpha_u=0.05, alpha_v=0.45
        Heitz VNDF GGX anisotropic highlights) vs isotropic baseline (640x640, 128 spp).
  (2) Deep 12-Bounce Infinite Mirror Gallery (max_bounces=12):
      - Dark neon-lit obsidian & brushed-gold infinite mirror corridor with glowing lanterns
        & refractive crystal glass orb, comparing 2-bounce vs 12-bounce deep recursive specular
        reflection inside a single XLA while_loop (interpret=False, host_callbacks==0, send_recv_ops==0).
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
from PIL import Image, ImageDraw, ImageFont

from bvh_pbr_pt import (
    LightRig,
    PBRWideBVHMesh,
    _agx_tonemap_srgb,
    _build_onb,
    _sample_cosine_hemisphere,
    _sample_ggx_half,
    _smith_g1,
    audit_hlo_zero_copy,
    pbr_raycast_tpu,
)
from complex_scenes_pt import _look_at_mat
from physical_fx_pt import CinemaMeshBuilder, stitch_strip


# ==============================================================================
# Spectral CIE 1931 Quadrature + Airy Thin-Film Interference
# ==============================================================================

LAMBDA_MIN_NM = 420.0
LAMBDA_MAX_NM = 680.0
N_SPEC_BINS = 8

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


_SPEC_LAM_NP = np.linspace(LAMBDA_MIN_NM, LAMBDA_MAX_NM, N_SPEC_BINS + 1, dtype=np.float32)
_SPEC_LAM_MID_NP = 0.5 * (_SPEC_LAM_NP[:-1] + _SPEC_LAM_NP[1:])
_SPEC_RGB_RAW_NP = np.maximum(_cie_xyz_analytic_np(_SPEC_LAM_MID_NP) @ _M_XYZ_TO_SRGB.T, 0.0)
# White-balanced spectral quadrature weights: sum_k W_k == (1, 1, 1)
_SPEC_RGB_WEIGHTS_NP = (_SPEC_RGB_RAW_NP / np.sum(_SPEC_RGB_RAW_NP, axis=0, keepdims=True)).astype(np.float32)


def _bubble_thickness_nm(hit_pos: jax.Array, n_ff: jax.Array) -> jax.Array:
  """Smooth gravity-drained + Marangoni swirling soap-film thickness in [280, 620] nm."""
  px, py, pz = hit_pos[..., 0], hit_pos[..., 1], hit_pos[..., 2]
  swirl1 = jnp.sin(10.5 * px + 7.2 * pz + 1.4 * jnp.cos(8.5 * py))
  swirl2 = jnp.cos(9.0 * py - 11.0 * pz + 0.9 * jnp.sin(6.5 * px))
  drain = jnp.clip(0.5 - 0.5 * n_ff[..., 2], 0.0, 1.0)
  u = jnp.clip(0.20 + 0.40 * drain + 0.25 * (0.5 + 0.5 * swirl1) + 0.15 * (0.5 + 0.5 * swirl2), 0.0, 1.0)
  return 280.0 + (620.0 - 280.0) * u


def _anodized_ti_thickness_nm(hit_pos: jax.Array, n_ff: jax.Array) -> jax.Array:
  """Smooth anodized TiO2 oxide thickness across the titanium sphere in [85, 225] nm."""
  px, py, pz = hit_pos[..., 0], hit_pos[..., 1], hit_pos[..., 2]
  band = 0.5 + 0.5 * jnp.sin(6.5 * pz + 3.2 * px + 2.0 * jnp.cos(5.0 * py))
  angle_mod = 0.5 + 0.5 * n_ff[..., 2]
  u = jnp.clip(0.68 * band + 0.32 * angle_mod, 0.0, 1.0)
  return 85.0 + (225.0 - 85.0) * u


def _cdiv(ar: jax.Array, ai: jax.Array, br: jax.Array, bi: jax.Array) -> Tuple[jax.Array, jax.Array]:
  denom = jnp.maximum(br * br + bi * bi, 1e-12)
  return (ar * br + ai * bi) / denom, (ai * br - ar * bi) / denom


def _csqrt(ur: jax.Array, ui: jax.Array) -> Tuple[jax.Array, jax.Array]:
  r = jnp.sqrt(jnp.maximum(ur * ur + ui * ui, 1e-24))
  xr = jnp.sqrt(jnp.maximum(0.5 * (r + ur), 0.0))
  xi = jnp.where(ui >= 0.0, 1.0, -1.0) * jnp.sqrt(jnp.maximum(0.5 * (r - ur), 0.0))
  return xr, xi


def _airy_soap_film_rgb(cos_i: jax.Array, d_nm: jax.Array, interference: bool) -> jax.Array:
  """Airy thin-film spectral reflectance for air (1.0) -> soap (1.333) -> air (1.0)."""
  nf = 1.333
  c0 = jnp.clip(cos_i, 1e-4, 1.0)
  sin2_0 = 1.0 - c0 * c0
  cf = jnp.sqrt(jnp.maximum(1.0 - sin2_0 / (nf * nf), 1e-4))

  r01_s = (c0 - nf * cf) / jnp.maximum(c0 + nf * cf, 1e-6)
  r01_p = (nf * c0 - cf) / jnp.maximum(nf * c0 + cf, 1e-6)

  if not interference:
    # Incoherent double-interface Fresnel reflectance (achromatic baseline)
    rs2 = r01_s * r01_s
    rp2 = r01_p * r01_p
    r_incoh = 0.5 * ((2.0 * rs2) / (1.0 + rs2) + (2.0 * rp2) / (1.0 + rp2))
    return jnp.broadcast_to(r_incoh[..., None], (*cos_i.shape, 3))

  r12_s = -r01_s
  r12_p = -r01_p
  lam = jnp.asarray(_SPEC_LAM_MID_NP, dtype=jnp.float32)        # (K,)
  w_rgb = jnp.asarray(_SPEC_RGB_WEIGHTS_NP, dtype=jnp.float32)  # (K, 3)

  phi = (4.0 * math.pi * nf * d_nm[..., None] * cf[..., None]) / lam
  cphi = jnp.cos(phi)

  r01s2 = (r01_s * r01_s)[..., None]
  r12s2 = (r12_s * r12_s)[..., None]
  two_s = (2.0 * r01_s * r12_s)[..., None] * cphi
  rs = (r01s2 + r12s2 + two_s) / jnp.maximum(1.0 + r01s2 * r12s2 + two_s, 1e-6)

  r01p2 = (r01_p * r01_p)[..., None]
  r12p2 = (r12_p * r12_p)[..., None]
  two_p = (2.0 * r01_p * r12_p)[..., None] * cphi
  rp = (r01p2 + r12p2 + two_p) / jnp.maximum(1.0 + r01p2 * r12p2 + two_p, 1e-6)

  r_spec = 0.5 * (rs + rp)  # (..., K)
  return jnp.clip(jnp.einsum("...k,kc->...c", r_spec, w_rgb), 0.0, 1.0)


def _airy_anodized_ti_rgb(cos_i: jax.Array, d_nm: jax.Array) -> jax.Array:
  """Airy interference for TiO2 (nf=2.40) thin oxide film over metallic Ti (n2=2.15 + 2.95j)."""
  nf = 2.40
  n2_r, n2_i = 2.15, 2.95
  c0 = jnp.clip(cos_i, 1e-4, 1.0)
  sin2_0 = 1.0 - c0 * c0
  cf = jnp.sqrt(jnp.maximum(1.0 - sin2_0 / (nf * nf), 1e-4))

  r01_s = (c0 - nf * cf) / jnp.maximum(c0 + nf * cf, 1e-6)
  r01_p = (nf * c0 - cf) / jnp.maximum(nf * c0 + cf, 1e-6)

  # Complex Snell into Ti: (n0 / n2)^2 = 1 / (n2_r + i*n2_i)^2
  n2sq_r = n2_r * n2_r - n2_i * n2_i
  n2sq_i = 2.0 * n2_r * n2_i
  inv_r, inv_i = _cdiv(jnp.ones_like(c0), jnp.zeros_like(c0), jnp.full_like(c0, n2sq_r), jnp.full_like(c0, n2sq_i))
  c2_r, c2_i = _csqrt(1.0 - sin2_0 * inv_r, -sin2_0 * inv_i)

  # r12_s = (nf*cf - n2*c2) / (nf*cf + n2*c2)
  n2c2_r = n2_r * c2_r - n2_i * c2_i
  n2c2_i = n2_r * c2_i + n2_i * c2_r
  r12s_r, r12s_i = _cdiv(nf * cf - n2c2_r, -n2c2_i, nf * cf + n2c2_r, n2c2_i)

  # r12_p = (n2*cf - nf*c2) / (n2*cf + nf*c2)
  r12p_r, r12p_i = _cdiv(n2_r * cf - nf * c2_r, n2_i * cf - nf * c2_i, n2_r * cf + nf * c2_r, n2_i * cf + nf * c2_i)

  mag_s = jnp.sqrt(jnp.maximum(r12s_r * r12s_r + r12s_i * r12s_i, 1e-12))
  del_s = jnp.arctan2(r12s_i, r12s_r)
  mag_p = jnp.sqrt(jnp.maximum(r12p_r * r12p_r + r12p_i * r12p_i, 1e-12))
  del_p = jnp.arctan2(r12p_i, r12p_r)

  lam = jnp.asarray(_SPEC_LAM_MID_NP, dtype=jnp.float32)
  w_rgb = jnp.asarray(_SPEC_RGB_WEIGHTS_NP, dtype=jnp.float32)
  phi = (4.0 * math.pi * nf * d_nm[..., None] * cf[..., None]) / lam

  c_s = jnp.cos(phi + del_s[..., None])
  r01s2 = (r01_s * r01_s)[..., None]
  mags2 = (mag_s * mag_s)[..., None]
  twos = (2.0 * r01_s * mag_s)[..., None] * c_s
  rs = (r01s2 + mags2 + twos) / jnp.maximum(1.0 + r01s2 * mags2 + twos, 1e-6)

  c_p = jnp.cos(phi + del_p[..., None])
  r01p2 = (r01_p * r01_p)[..., None]
  magp2 = (mag_p * mag_p)[..., None]
  twop = (2.0 * r01_p * mag_p)[..., None] * c_p
  rp = (r01p2 + magp2 + twop) / jnp.maximum(1.0 + r01p2 * magp2 + twop, 1e-6)

  r_spec = 0.5 * (rs + rp)
  return jnp.clip(jnp.einsum("...k,kc->...c", r_spec, w_rgb), 0.0, 1.0)


# ==============================================================================
# Anisotropic GGX BRDF + Heitz (2018) VNDF Sampling
# ==============================================================================

def _radial_aniso_frame(hit_pos: jax.Array, n_ff: jax.Array, mcode: jax.Array) -> Tuple[jax.Array, jax.Array]:
  """Compute circular-brushed tangent frame (t_u = circumferential, t_v = radial) on discs."""
  c_gold = jnp.array([-0.48, -0.10, 0.05], dtype=jnp.float32)
  c_steel = jnp.array([0.48, -0.10, 0.05], dtype=jnp.float32)
  is_gold = (mcode >= 1.15) & (mcode < 1.25)
  disc_c = jnp.where(is_gold[..., None], c_gold, c_steel)

  r_vec = hit_pos - disc_c
  r_proj = r_vec - jnp.sum(r_vec * n_ff, axis=-1, keepdims=True) * n_ff
  r_norm = jnp.linalg.norm(r_proj, axis=-1, keepdims=True)
  fb1, _ = _build_onb(n_ff)
  t_rad = jnp.where(r_norm > 1e-4, r_proj / jnp.maximum(r_norm, 1e-6), fb1)
  t_phi = jnp.cross(n_ff, t_rad)
  t_phi = t_phi / jnp.maximum(jnp.linalg.norm(t_phi, axis=-1, keepdims=True), 1e-12)
  return t_phi, t_rad


def _aniso_lambda(w: jax.Array, n: jax.Array, t_u: jax.Array, t_v: jax.Array, au: jax.Array, av: jax.Array) -> jax.Array:
  wu = jnp.sum(w * t_u, axis=-1)
  wv = jnp.sum(w * t_v, axis=-1)
  wn = jnp.maximum(jnp.sum(w * n, axis=-1), 1e-4)
  a2_tan2 = ((au * wu) ** 2 + (av * wv) ** 2) / (wn * wn)
  return 0.5 * (-1.0 + jnp.sqrt(1.0 + a2_tan2))


def _eval_aniso_ggx_brdf(
    n: jax.Array, t_u: jax.Array, t_v: jax.Array,
    wo: jax.Array, wi: jax.Array,
    albedo: jax.Array, metallic: jax.Array,
    au: jax.Array, av: jax.Array,
    fresnel_override: Optional[jax.Array] = None,
) -> jax.Array:
  ndotl = jnp.maximum(jnp.sum(n * wi, axis=-1), 0.0)
  ndotv = jnp.maximum(jnp.sum(n * wo, axis=-1), 1e-4)
  h = wo + wi
  h = h / jnp.maximum(jnp.linalg.norm(h, axis=-1, keepdims=True), 1e-12)
  ndoth = jnp.maximum(jnp.sum(n * h, axis=-1), 0.0)
  vdoth = jnp.maximum(jnp.sum(wo * h, axis=-1), 0.0)
  hu = jnp.sum(h * t_u, axis=-1)
  hv = jnp.sum(h * t_v, axis=-1)

  denom = (hu / au) ** 2 + (hv / av) ** 2 + ndoth * ndoth
  D = 1.0 / (math.pi * au * av * denom * denom + 1e-7)
  lam_o = _aniso_lambda(wo, n, t_u, t_v, au, av)
  lam_i = _aniso_lambda(wi, n, t_u, t_v, au, av)
  G2 = 1.0 / (1.0 + lam_o + lam_i)

  if fresnel_override is not None:
    F = fresnel_override
  else:
    f0 = 0.04 * (1.0 - metallic[..., None]) + albedo * metallic[..., None]
    F = f0 + (1.0 - f0) * ((1.0 - vdoth) ** 5)[..., None]

  spec = (D * G2 / jnp.maximum(4.0 * ndotl * ndotv, 1e-4))[..., None] * F
  kd = (1.0 - metallic[..., None]) * (1.0 - F)
  return kd * albedo / math.pi + spec


def _sample_ggx_vndf_aniso(
    u1: jax.Array, u2: jax.Array,
    wo: jax.Array, n: jax.Array, t_u: jax.Array, t_v: jax.Array,
    au: jax.Array, av: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
  """Heitz (2018) Sampling the GGX Distribution of Visible Normals (returns wi, h)."""
  vx = jnp.sum(wo * t_u, axis=-1)
  vy = jnp.sum(wo * t_v, axis=-1)
  vz = jnp.maximum(jnp.sum(wo * n, axis=-1), 1e-4)

  vh_raw = jnp.stack([au * vx, av * vy, vz], axis=-1)
  vh = vh_raw / jnp.maximum(jnp.linalg.norm(vh_raw, axis=-1, keepdims=True), 1e-12)
  lensq = vh[..., 0] ** 2 + vh[..., 1] ** 2
  t1_fb = jnp.broadcast_to(jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32), vh.shape)
  t1_dir = jnp.stack([-vh[..., 1], vh[..., 0], jnp.zeros_like(lensq)], axis=-1) / jnp.sqrt(jnp.maximum(lensq, 1e-7))[..., None]
  T1 = jnp.where((lensq > 1e-7)[..., None], t1_dir, t1_fb)
  T2 = jnp.cross(vh, T1)

  r = jnp.sqrt(jnp.clip(u1, 1e-6, 1.0 - 1e-6))
  phi = 2.0 * math.pi * u2
  p1 = r * jnp.cos(phi)
  p2 = r * jnp.sin(phi)
  s = 0.5 * (1.0 + vh[..., 2])
  p2 = (1.0 - s) * jnp.sqrt(jnp.maximum(1.0 - p1 * p1, 0.0)) + s * p2
  pz = jnp.sqrt(jnp.maximum(1.0 - p1 * p1 - p2 * p2, 0.0))

  nh = p1[..., None] * T1 + p2[..., None] * T2 + pz[..., None] * vh
  ne_raw = jnp.stack([au * nh[..., 0], av * nh[..., 1], jnp.maximum(nh[..., 2], 0.0)], axis=-1)
  ne = ne_raw / jnp.maximum(jnp.linalg.norm(ne_raw, axis=-1, keepdims=True), 1e-12)

  h = ne[..., 0:1] * t_u + ne[..., 1:2] * t_v + ne[..., 2:3] * n
  h = h / jnp.maximum(jnp.linalg.norm(h, axis=-1, keepdims=True), 1e-12)
  wi = 2.0 * jnp.sum(wo * h, axis=-1, keepdims=True) * h - wo
  wi = wi / jnp.maximum(jnp.linalg.norm(wi, axis=-1, keepdims=True), 1e-12)
  return wi, h


# ==============================================================================
# Pure-JAX TPU Joint A-Trous Denoiser (Preserving Sharp Mirrors & Bubbles)
# ==============================================================================

def _tpu_joint_atrous_denoise_sharp(
    rad_lin: jax.Array,
    nrm: jax.Array,
    alb: jax.Array,
    depth: jax.Array,
    spec_mask: jax.Array,
    sigma_l: float = 0.30,
    n_passes: int = 3,
) -> jax.Array:
  H, W, _ = rad_lin.shape
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
  # Higher clamp ceiling on specular surfaces so bright 12-bounce mirror lantern pinpoints stay crisp
  cap_thresh = jnp.where(spec_mask > 0.5, nbr_mu + 4.8 * nbr_std + 1.80, nbr_mu + 3.0 * nbr_std + 0.14)
  lum_cap = jnp.minimum(lum, cap_thresh)
  scale_ff = jnp.where(lum > 1e-6, lum_cap / jnp.maximum(lum, 1e-6), 1.0)
  cur = rad_lin * scale_ff[..., None]

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
    # Keep specular mirrors/bubbles razor sharp while smoothing satin obsidian floor/pillars cleanly
    blend = jnp.clip(1.0 - 0.90 * spec_mask, 0.10, 1.0)[..., None]
    cur = cur + blend * (denoised - cur)
  return cur


# ==============================================================================
# Unified 4-Chip TPU Path Tracer (Thin-Film + Anisotropic GGX + 12-Bounce While Loop)
# ==============================================================================

@dataclasses.dataclass(frozen=True)
class OpticsConfig:
  thinfilm: bool = True
  anisotropic: bool = True
  alpha_u: float = 0.05
  alpha_v: float = 0.45
  alpha_iso: float = 0.15
  use_denoiser: bool = True


def build_wave7_pathtracer(
    bvh_mesh: PBRWideBVHMesh,
    devices: Sequence[jax.Device],
    height: int,
    width: int,
    cam_fovy: float,
    znear: float,
    zfar: float,
    light_rig: LightRig,
    spp: int,
    max_bounces: int,
    opt_cfg: OpticsConfig,
):
  n_chips = len(devices)
  assert width % n_chips == 0 and (width // n_chips) % 16 == 0 and height % 8 == 0
  W_local = width // n_chips

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
      s = s + (cos_s ** 48.0)[..., None] * sun_emit * 0.22
    return s

  aspect = float(width) / float(height)
  half_h = float(znear) * math.tan(0.5 * float(cam_fovy) * (math.pi / 180.0))
  half_w = half_h * aspect

  def _render_local(l2r, l2, l0, lf, cxp, cxm, cid_arr, rng_in):
    cid = cid_arr[0]
    x_off = cid * W_local
    chip_rng = jax.random.fold_in(rng_in, cid)

    local_px = height * W_local
    spp_par = 8 if (spp >= 8 and local_px <= 163840) else (4 if (spp >= 4 and local_px <= 262144) else 1)
    spp_seq = max(1, spp // spp_par)
    R = height * W_local

    def _trace_batch_samples(s_idx, acc):
      rad_acc, nrm_acc, alb_acc, dep_acc, spc_acc = acc
      s_rng = jax.random.fold_in(chip_rng, s_idx)
      k_jit, k_bnc = jax.random.split(s_rng, 2)

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
      prim_d0 = jnp.full((spp_par, R), float(zfar), dtype=jnp.float32)
      prim_n0 = jnp.zeros((spp_par, R, 3), dtype=jnp.float32)
      prim_a0 = jnp.zeros((spp_par, R, 3), dtype=jnp.float32)
      prim_s0 = jnp.zeros((spp_par, R), dtype=jnp.float32)

      def _bounce_cond(carry):
        b_idx, _, _, _, _, alive = carry[:6]
        return (b_idx < max_bounces) & jnp.any(alive)

      def _bounce_body(carry):
        b_idx, ro, rd, t_lim, thr, alive, rad, prim_d, prim_n, prim_a, prim_s = carry
        b_rng = jax.random.fold_in(k_bnc, b_idx)
        k_lu, k_lv, k_su, k_sv, k_lobe, k_u1, k_u2 = jax.random.split(b_rng, 7)

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

        # Material classification
        is_emit = hit & (mcode >= 2.8)
        is_bubble = hit & (mcode >= 2.45) & (mcode < 2.65)
        is_glass = hit & (mcode >= 1.8) & (mcode < 2.45)
        is_mirror = hit & (mcode >= 1.65) & (mcode < 1.75)
        is_ti = hit & (mcode >= 1.35) & (mcode < 1.48)
        is_aniso_disc = hit & (mcode >= 1.15) & (mcode < 1.35)
        is_std_opq = hit & (mcode < 1.15)
        is_opaque_shade = is_std_opq | is_aniso_disc | is_ti | is_mirror

        metallic = jnp.where(is_aniso_disc | is_ti | is_mirror, 1.0, jnp.clip(mcode, 0.0, 1.0))
        is_spec_prim = is_glass | is_bubble | is_mirror | is_ti | is_aniso_disc
        prim_n = jnp.where(b_idx == 0, jnp.where(hit[..., None], n_ff, jnp.array([0.0, 0.0, 1.0], jnp.float32)), prim_n)
        prim_a = jnp.where(b_idx == 0, jnp.where(hit[..., None], alb, 0.0), prim_a)
        prim_s = jnp.where(b_idx == 0, jnp.where(is_spec_prim, 1.0, 0.0), prim_s)

        emit_power = jnp.where(mcode >= 2.8, (mcode - 2.0) * 5.2, 0.0)
        emit_contrib = jnp.where(is_emit[..., None], thr * alb * emit_power[..., None], 0.0)
        # Softly cap high-bounce indirect emissive fireflies while keeping direct & mirror bounces crisp
        rad = rad + jnp.where(b_idx >= 1, jnp.minimum(emit_contrib, 7.5), emit_contrib)

        hit_pos = ro + jnp.where(hit, d_hit, 0.0)[..., None] * rd

        # Tangent frames & roughness (Anisotropic radial-brushed discs vs isotropic fallback)
        t_u_disc, t_v_disc = _radial_aniso_frame(hit_pos, n_ff, mcode)
        t_u_iso, t_v_iso = _build_onb(n_ff)
        use_aniso = opt_cfg.anisotropic & is_aniso_disc
        t_u = jnp.where(use_aniso[..., None], t_u_disc, t_u_iso)
        t_v = jnp.where(use_aniso[..., None], t_v_disc, t_v_iso)

        alpha_std = jnp.where(is_mirror, 1e-4, jnp.maximum(rgh * rgh, 0.008))
        alpha_disc = jnp.float32(opt_cfg.alpha_iso)
        au = jnp.where(use_aniso, jnp.float32(opt_cfg.alpha_u), jnp.where(is_aniso_disc, alpha_disc, alpha_std))
        av = jnp.where(use_aniso, jnp.float32(opt_cfg.alpha_v), jnp.where(is_aniso_disc, alpha_disc, alpha_std))

        # Airy thin-film reflectance for soap bubbles and anodized titanium sphere
        cos_i_surf = jnp.clip(jnp.sum(wo * n_ff, axis=-1), 0.0, 1.0)
        d_bub = _bubble_thickness_nm(hit_pos, n_ff)
        F_bub = _airy_soap_film_rgb(cos_i_surf, d_bub, opt_cfg.thinfilm)

        d_ti = _anodized_ti_thickness_nm(hit_pos, n_ff)
        F_ti_airy = _airy_anodized_ti_rgb(cos_i_surf, d_ti)

        # Direct NEE against area light & optional rim/sun light
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
        need_sh = is_opaque_shade & (cos_s > 0.0) & (cos_l > 0.0) & (~is_mirror)
        sh_o = hit_pos + n_ff * 2.0e-3
        sh_lim = jnp.where(need_sh, jnp.maximum(dist_l - 2.5e-3, 0.0), 0.0)
        d_sh, *_ = pbr_raycast_tpu(l2r, l2, l0, lf, sh_o, wi_l, sh_lim, height, W_local, any_hit=True)
        vis = jnp.where(need_sh & (d_sh >= sh_lim - 1e-4), 1.0, 0.0)

        h_l = wo + wi_l
        h_l = h_l / jnp.maximum(jnp.linalg.norm(h_l, axis=-1, keepdims=True), 1e-12)
        vdoth_l = jnp.maximum(jnp.sum(wo * h_l, axis=-1), 0.0)
        f0_std = 0.04 * (1.0 - metallic[..., None]) + alb * metallic[..., None]
        F_std_l = f0_std + (1.0 - f0_std) * ((1.0 - vdoth_l) ** 5)[..., None]
        F_ti_l = _airy_anodized_ti_rgb(vdoth_l, d_ti)
        F_nee_l = jnp.where((opt_cfg.thinfilm & is_ti)[..., None], F_ti_l, F_std_l)

        brdf_a = _eval_aniso_ggx_brdf(n_ff, t_u, t_v, wo, wi_l, alb, metallic, au, av, fresnel_override=F_nee_l)
        geom_a = (cos_s * cos_l) / (dist_l * dist_l + 0.08) * area_area
        nee_tot = jnp.minimum(brdf_a * area_emit[None, None, :] * (geom_a * vis)[..., None], 22.0)

        if light_rig.has_sun:
          su = jax.random.uniform(k_su, (spp_par, R), minval=0.0, maxval=1.0, dtype=jnp.float32)
          sv = jax.random.uniform(k_sv, (spp_par, R), minval=0.0, maxval=1.0, dtype=jnp.float32)
          sr = sun_sin_max * jnp.sqrt(su)
          sphi = (2.0 * math.pi) * sv
          wi_sun = sun_d[None, None, :] + (sr * jnp.cos(sphi))[..., None] * st1[None, None, :] + (sr * jnp.sin(sphi))[..., None] * st2[None, None, :]
          wi_sun = wi_sun / jnp.maximum(jnp.linalg.norm(wi_sun, axis=-1, keepdims=True), 1e-12)
          cos_sun = jnp.maximum(jnp.sum(n_ff * wi_sun, axis=-1), 0.0)
          need_sun = is_opaque_shade & (cos_sun > 0.0) & (~is_mirror)
          sun_lim = jnp.where(need_sun, jnp.float32(zfar * 0.9), 0.0)
          d_sun, *_ = pbr_raycast_tpu(l2r, l2, l0, lf, sh_o, wi_sun, sun_lim, height, W_local, any_hit=True)
          vis_sun = jnp.where(need_sun & (d_sun >= sun_lim - 1e-2), 1.0, 0.0)

          h_s = wo + wi_sun
          h_s = h_s / jnp.maximum(jnp.linalg.norm(h_s, axis=-1, keepdims=True), 1e-12)
          vdoth_s = jnp.maximum(jnp.sum(wo * h_s, axis=-1), 0.0)
          F_std_s = f0_std + (1.0 - f0_std) * ((1.0 - vdoth_s) ** 5)[..., None]
          F_ti_s = _airy_anodized_ti_rgb(vdoth_s, d_ti)
          F_nee_s = jnp.where((opt_cfg.thinfilm & is_ti)[..., None], F_ti_s, F_std_s)
          brdf_s = _eval_aniso_ggx_brdf(n_ff, t_u, t_v, wo, wi_sun, alb, metallic, au, av, fresnel_override=F_nee_s)
          nee_tot = nee_tot + jnp.minimum(brdf_s * sun_emit[None, None, :] * (cos_sun * vis_sun)[..., None], 24.0)

        rad = rad + jnp.where(is_opaque_shade[..., None], thr * nee_tot, 0.0)

        # Sample next indirect bounce
        u_lobe = jax.random.uniform(k_lobe, (spp_par, R), minval=0.0, maxval=1.0, dtype=jnp.float32)
        u1 = jax.random.uniform(k_u1, (spp_par, R), minval=0.0, maxval=1.0, dtype=jnp.float32)
        u2 = jax.random.uniform(k_u2, (spp_par, R), minval=0.0, maxval=1.0, dtype=jnp.float32)

        # Pure specular on mirrors, brushed discs, and anodized Ti (metallic=1.0)
        p_spec = jnp.where(is_mirror | is_aniso_disc | is_ti, 1.0, jnp.clip(0.15 + 0.78 * metallic, 0.15, 0.90))
        choose_spec = u_lobe < p_spec
        wi_diff = _sample_cosine_hemisphere(u1, u2, n_ff)
        wi_spec_vndf, h_vndf = _sample_ggx_vndf_aniso(u1, u2, wo, n_ff, t_u, t_v, au, av)

        refl_ideal = 2.0 * cos_i_surf[..., None] * n_ff - wo
        refl_ideal = refl_ideal / jnp.maximum(jnp.linalg.norm(refl_ideal, axis=-1, keepdims=True), 1e-12)
        wi_spec = jnp.where(is_mirror[..., None], refl_ideal, wi_spec_vndf)
        wi_opq = jnp.where(choose_spec[..., None], wi_spec, wi_diff)

        ndotl_s = jnp.maximum(jnp.sum(n_ff * wi_spec, axis=-1), 0.0)
        vdoth_v = jnp.maximum(jnp.sum(wo * h_vndf, axis=-1), 1e-4)
        F_std_v = f0_std + (1.0 - f0_std) * ((1.0 - vdoth_v) ** 5)[..., None]
        F_ti_v = _airy_anodized_ti_rgb(vdoth_v, d_ti)
        F_v = jnp.where((opt_cfg.thinfilm & is_ti)[..., None], F_ti_v, F_std_v)
        lam_o_v = _aniso_lambda(wo, n_ff, t_u, t_v, au, av)
        lam_i_v = _aniso_lambda(wi_spec, n_ff, t_u, t_v, au, av)
        g2_over_g1 = (1.0 + lam_o_v) / jnp.maximum(1.0 + lam_o_v + lam_i_v, 1e-4)
        w_spec_vndf = jnp.where(ndotl_s[..., None] > 0.0, jnp.minimum(F_v * (g2_over_g1 / jnp.maximum(p_spec, 1e-4))[..., None], 4.0), 0.0)
        # Mirror albedo throughput (0.95..0.98) for deep 12-bounce corridor
        w_mirror = alb
        w_spec = jnp.where(is_mirror[..., None], w_mirror, w_spec_vndf)
        w_diff = (1.0 - metallic[..., None]) * alb / jnp.maximum(1.0 - p_spec[..., None], 0.1)
        w_opq = jnp.where(choose_spec[..., None], w_spec, w_diff)

        # Soap bubble thin-film: stochastic reflection (Airy R) vs straight transmission (1 - Airy R)
        p_bub_refl = jnp.clip(jnp.mean(F_bub, axis=-1), 0.08, 0.92)
        do_bub_refl = u_lobe < p_bub_refl
        wi_bub = jnp.where(do_bub_refl[..., None], refl_ideal, rd)
        ro_bub = jnp.where(do_bub_refl[..., None], hit_pos + n_ff * 1.5e-3, hit_pos + rd * 1.5e-3)
        w_bub = jnp.where(
            do_bub_refl[..., None],
            F_bub / jnp.maximum(p_bub_refl[..., None], 1e-4),
            (1.0 - F_bub) / jnp.maximum(1.0 - p_bub_refl[..., None], 1e-4),
        )

        # Refractive Crystal Glass (IOR = 1.52)
        ior_g = jnp.float32(1.52)
        sigma_a = -jnp.log(jnp.clip(alb, 0.05, 0.998)) / 0.85
        beer_glass = jnp.where((is_glass & (~entering))[..., None], jnp.exp(-sigma_a * jnp.maximum(d_hit, 0.0)[..., None]), 1.0)
        eta = jnp.where(entering, 1.0 / ior_g, ior_g)
        sin2_t = eta * eta * (1.0 - cos_i_surf * cos_i_surf)
        tir = sin2_t > 1.0
        r0_g = ((1.0 - eta) / (1.0 + eta)) ** 2
        fresnel_g = r0_g + (1.0 - r0_g) * ((1.0 - cos_i_surf) ** 5)
        cos_t = jnp.sqrt(jnp.maximum(1.0 - sin2_t, 0.0))
        refr_dir = eta[..., None] * rd + (eta * cos_i_surf - cos_t)[..., None] * n_ff
        do_g_refl = tir | (u_lobe < fresnel_g)
        wi_glass = jnp.where(do_g_refl[..., None], refl_ideal, refr_dir)
        wi_glass = wi_glass / jnp.maximum(jnp.linalg.norm(wi_glass, axis=-1, keepdims=True), 1e-12)
        ro_glass = jnp.where(do_g_refl[..., None], hit_pos + n_ff * 1.5e-3, hit_pos - n_ff * 1.5e-3)

        wi_next = jnp.where(is_bubble[..., None], wi_bub, jnp.where(is_glass[..., None], wi_glass, wi_opq))
        new_ro = jnp.where(is_bubble[..., None], ro_bub, jnp.where(is_glass[..., None], ro_glass, hit_pos + n_ff * 1.5e-3))
        bounce_w = jnp.where(is_bubble[..., None], w_bub, jnp.where(is_glass[..., None], beer_glass, w_opq))

        can_cont = is_opaque_shade | is_glass | is_bubble
        new_thr = thr * bounce_w
        new_alive = can_cont & (jnp.max(new_thr, axis=-1) > 1e-3)
        new_t_lim = jnp.broadcast_to(jnp.float32(zfar), (spp_par, R))
        return (b_idx + 1, new_ro, wi_next, new_t_lim, new_thr, new_alive, rad, prim_d, prim_n, prim_a, prim_s)

      init_carry = (
          jnp.int32(0), ro0, rd0, t_max0, thr0, alive0, rad0, prim_d0, prim_n0, prim_a0, prim_s0
      )
      _, _, _, _, _, _, final_rad, prim_d, prim_n, prim_a, prim_s = jax.lax.while_loop(
          _bounce_cond, _bounce_body, init_carry
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
    if opt_cfg.use_denoiser:
      nrm_g = full[..., 3:6]
      nrm_g = nrm_g / jnp.maximum(jnp.linalg.norm(nrm_g, axis=-1, keepdims=True), 1e-6)
      alb_g = full[..., 6:9]
      dep_g = full[..., 9]
      spc_g = full[..., 10]
      rad_lin = _tpu_joint_atrous_denoise_sharp(rad_lin, nrm_g, alb_g, dep_g, spc_g, sigma_l=0.36, n_passes=4)
    return _agx_tonemap_srgb(rad_lin, exposure=light_rig.exposure)

  return render_fn, mesh, rep


# ==============================================================================
# Procedural High-Res Scenes
# ==============================================================================

def _add_chamfered_disc(
    mb: CinemaMeshBuilder,
    center: Tuple[float, float, float],
    radius: float,
    thickness: float,
    tilt_x_deg: float,
    albedo: Tuple[float, float, float],
    roughness: float,
    mat_code: float,
    n_seg: int = 96,
):
  """Closed bevelled circular disc tilted toward camera with crisp flat top face."""
  cx, cy, cz = center
  r_top = radius * 0.955
  r_bot = radius
  hz = thickness * 0.5
  ang = math.radians(tilt_x_deg)
  ca, sa = math.cos(ang), math.sin(ang)
  # Tilt around X: +Y moves up (+Z), -Y moves down, so normal (0,0,1) tilts toward -Y camera: (0, -sa, ca)
  Rx = np.array([[1.0, 0.0, 0.0], [0.0, ca, -sa], [0.0, sa, ca]], dtype=np.float32)

  # Top flat cap (unshared flat normal)
  top_verts = [[0.0, 0.0, hz]]
  for j in range(n_seg):
    p = 2.0 * math.pi * j / n_seg
    top_verts.append([r_top * math.cos(p), r_top * math.sin(p), hz])
  top_faces = [[0, 1 + j, 1 + ((j + 1) % n_seg)] for j in range(n_seg)]
  tv = (np.array(top_verts, np.float32) @ Rx.T) + np.array([cx, cy, cz], np.float32)
  mb.add_mesh(tv, np.array(top_faces, np.int32), albedo, roughness, mat_code, smooth=False)

  # Smooth chamfer + cylindrical skirt + bottom cap
  ring_bot, ring_mid_b, ring_mid_t, ring_top = [], [], [], []
  for j in range(n_seg):
    p = 2.0 * math.pi * j / n_seg
    cp, sp = math.cos(p), math.sin(p)
    ring_bot.append([r_bot * cp, r_bot * sp, -hz])
    ring_mid_b.append([r_bot * cp, r_bot * sp, hz * 0.65])
    ring_top.append([r_top * cp, r_top * sp, hz])
  verts = [[0.0, 0.0, -hz]] + ring_bot + ring_mid_b + ring_top
  faces = []
  o_b, o_m, o_t = 1, 1 + n_seg, 1 + 2 * n_seg
  for j in range(n_seg):
    jn = (j + 1) % n_seg
    faces.append([0, o_b + jn, o_b + j])
    faces.append([o_b + j, o_b + jn, o_m + j])
    faces.append([o_b + jn, o_m + jn, o_m + j])
    faces.append([o_m + j, o_m + jn, o_t + j])
    faces.append([o_m + jn, o_t + jn, o_t + j])
  sv = (np.array(verts, np.float32) @ Rx.T) + np.array([cx, cy, cz], np.float32)
  mb.add_mesh(sv, np.array(faces, np.int32), albedo, roughness, mat_code, smooth=True)


def build_thinfilm_aniso_scene():
  """Dark Cinema Studio: Floating Iridescent Soap Bubbles + Anodized Ti Sphere + Brushed Gold/Steel Discs."""
  mb = CinemaMeshBuilder()

  # Dark slate tiled studio tabletop (z <= 0)
  mb.add_box((0.0, 0.0, -0.05), (2.4, 2.4, 0.03), (0.038, 0.042, 0.050), roughness=0.42, mat_code=0.0)
  for ix in range(8):
    for iy in range(8):
      tx = -1.6 + (ix + 0.5) * (3.2 / 8)
      ty = -1.6 + (iy + 0.5) * (3.2 / 8)
      col = (0.040, 0.044, 0.052) if ((ix + iy) % 2 == 0) else (0.058, 0.062, 0.072)
      mb.add_box((tx, ty, -0.02), (0.1995, 0.1995, 0.02), col, roughness=0.38, mat_code=0.0)

  # Dark cyclorama studio enclosure
  mb.add_box((0.0, 2.0, 1.3), (2.5, 0.05, 1.4), (0.040, 0.044, 0.054), roughness=0.88, mat_code=0.0)
  mb.add_box((-2.1, 0.0, 1.3), (0.05, 2.4, 1.4), (0.040, 0.044, 0.054), roughness=0.88, mat_code=0.0)
  mb.add_box((2.1, 0.0, 1.3), (0.05, 2.4, 1.4), (0.040, 0.044, 0.054), roughness=0.88, mat_code=0.0)

  # Studio softbox light panels (positioned cleanly outside direct camera view for crisp bubble reflections)
  mb.add_box((0.0, -0.25, 1.88), (0.68, 0.20, 0.02), (1.0, 0.98, 0.94), roughness=0.2, mat_code=3.15)
  mb.add_box((-1.55, -0.15, 0.92), (0.02, 0.55, 0.45), (0.78, 0.93, 1.0), roughness=0.2, mat_code=3.05)
  mb.add_box((1.55, -0.15, 0.92), (0.02, 0.55, 0.45), (1.0, 0.90, 0.76), roughness=0.2, mat_code=3.05)
  mb.add_box((0.0, 1.72, 1.15), (0.95, 0.02, 0.12), (0.86, 0.92, 1.0), roughness=0.2, mat_code=2.92)

  # Hero 1 & 2: Radial-Brushed Anisotropic Gold & Steel Discs (tilted 12 deg toward camera)
  _add_chamfered_disc(
      mb, center=(-0.48, -0.10, 0.05), radius=0.33, thickness=0.045, tilt_x_deg=12.0,
      albedo=(0.94, 0.68, 0.22), roughness=0.14, mat_code=1.20, n_seg=96,
  )
  _add_chamfered_disc(
      mb, center=(0.48, -0.10, 0.05), radius=0.33, thickness=0.045, tilt_x_deg=12.0,
      albedo=(0.82, 0.85, 0.90), roughness=0.14, mat_code=1.30, n_seg=96,
  )

  # Hero 3: Anodized Iridescent Titanium Sphere on sleek dark obsidian ring stand
  mb.add_cylinder((0.0, 0.30, 0.0), 0.18, 0.045, (0.08, 0.082, 0.092), roughness=0.32, mat_code=0.05, n_seg=48)
  mb.add_sphere((0.0, 0.30, 0.285), 0.25, (0.64, 0.67, 0.72), roughness=0.11, mat_code=1.40, n_lat=48, n_lon=96)

  # Hero 4: Floating Iridescent Soap Bubbles (d = 280..620 nm)
  mb.add_sphere((-0.30, -0.34, 0.50), 0.21, (1.0, 1.0, 1.0), roughness=0.02, mat_code=2.50, n_lat=44, n_lon=88)
  mb.add_sphere((0.36, -0.26, 0.48), 0.16, (1.0, 1.0, 1.0), roughness=0.02, mat_code=2.50, n_lat=40, n_lon=80)
  mb.add_sphere((-0.62, -0.22, 0.42), 0.105, (1.0, 1.0, 1.0), roughness=0.02, mat_code=2.50, n_lat=32, n_lon=64)

  bvh = mb.build("thinfilm_anisotropic_studio")
  cxp, cxm = _look_at_mat((0.0, -2.02, 0.96), (0.0, -0.02, 0.28))
  rig = LightRig(
      area_center=(0.0, 0.58, 1.72),
      area_normal=(0.0, -0.34, -0.94),
      area_radius=0.36,
      area_emit=(5.8, 5.4, 5.0),
      has_sun=True,
      sun_dir=(-0.22, 0.45, 0.86),
      sun_cos_half_angle=0.996,
      sun_emit=(3.6, 3.4, 3.1),
      sky_zenith=(0.04, 0.05, 0.07),
      sky_horizon=(0.025, 0.03, 0.04),
      sky_ground=(0.01, 0.01, 0.014),
      exposure=0.96,
  )
  return bvh, cxp, cxm, 40.0, 0.05, 20.0, rig


def _add_lantern(
    mb: CinemaMeshBuilder,
    center: Tuple[float, float, float],
    radius: float,
    emit_color: Tuple[float, float, float],
    emit_mcode: float,
    ceiling_z: float = 2.02,
):
  """Hanging obsidian/gold-capped glowing neon lantern suspended from ceiling."""
  cx, cy, cz = center
  # Slender gold suspension rod
  rod_h = max(ceiling_z - (cz + radius), 0.05)
  mb.add_cylinder((cx, cy, cz + radius), 0.008, rod_h, (0.96, 0.76, 0.28), roughness=0.14, mat_code=0.95, n_seg=12)
  # Gold top & bottom caps
  mb.add_cylinder((cx, cy, cz + radius * 0.78), radius * 0.72, radius * 0.26, (0.96, 0.76, 0.28), roughness=0.14, mat_code=0.95, n_seg=20)
  mb.add_cylinder((cx, cy, cz - radius * 1.04), radius * 0.72, radius * 0.26, (0.96, 0.76, 0.28), roughness=0.14, mat_code=0.95, n_seg=20)
  # Glowing lantern core
  mb.add_sphere((cx, cy, cz), radius, emit_color, roughness=0.15, mat_code=emit_mcode, n_lat=18, n_lon=32)


def build_infinite_mirror_gallery_scene():
  """Dark Neon-Lit Obsidian & Gold Infinite Mirror Gallery (4-wall mirror chamber + glass orb + lanterns)."""
  mb = CinemaMeshBuilder()

  W_x, L_y, H_z = 1.05, 1.45, 2.02
  gold = (0.96, 0.75, 0.28)
  obsidian = (0.045, 0.048, 0.058)

  # Dark glossy obsidian floor with inlaid brushed-gold trim grid (z <= 0)
  mb.add_box((0.0, 0.0, -0.05), (W_x + 0.12, L_y + 0.12, 0.05), obsidian, roughness=0.22, mat_code=0.14)
  for gx in (-0.52, 0.0, 0.52):
    mb.add_box((gx, 0.0, 0.002), (0.012, L_y, 0.002), gold, roughness=0.16, mat_code=0.95)
  for gy in (-0.95, -0.35, 0.25, 0.85):
    mb.add_box((0.0, gy, 0.002), (W_x, 0.012, 0.002), gold, roughness=0.16, mat_code=0.95)

  # Dark coffered obsidian ceiling + brushed-gold beams
  mb.add_box((0.0, 0.0, H_z + 0.04), (W_x + 0.12, L_y + 0.12, 0.04), obsidian, roughness=0.45, mat_code=0.10)
  for gy in (-0.95, -0.35, 0.25, 0.85):
    mb.add_box((0.0, gy, H_z - 0.02), (W_x, 0.035, 0.025), gold, roughness=0.16, mat_code=0.95)

  # 4-Wall Infinite Mirror Enclosure (mcode=1.70, roughness=0.02 -> crisp recursive mirror)
  mb.add_box((-W_x - 0.03, 0.0, H_z * 0.5), (0.03, L_y + 0.08, H_z * 0.52), (0.94, 0.96, 0.98), roughness=0.02, mat_code=1.70)
  mb.add_box((W_x + 0.03, 0.0, H_z * 0.5), (0.03, L_y + 0.08, H_z * 0.52), (0.98, 0.93, 0.83), roughness=0.02, mat_code=1.70)
  mb.add_box((0.0, L_y + 0.03, H_z * 0.5), (W_x + 0.08, 0.03, H_z * 0.52), (0.95, 0.95, 0.97), roughness=0.02, mat_code=1.70)
  mb.add_box((0.0, -L_y - 0.03, H_z * 0.5), (W_x + 0.08, 0.03, H_z * 0.52), (0.95, 0.95, 0.97), roughness=0.02, mat_code=1.70)

  # Obsidian & Gold vertical corner pilasters + glowing neon vertical strips
  pilasters = [
      (-W_x + 0.03, -L_y + 0.03, (1.0, 0.65, 0.16)),
      (W_x - 0.03, -L_y + 0.03, (0.14, 0.88, 1.0)),
      (-W_x + 0.03, L_y - 0.03, (0.14, 0.88, 1.0)),
      (W_x - 0.03, L_y - 0.03, (1.0, 0.65, 0.16)),
      (-W_x + 0.02, 0.15, (1.0, 0.24, 0.72)),
      (W_x - 0.02, 0.15, (1.0, 0.24, 0.72)),
  ]
  for px, py, ncol in pilasters:
    mb.add_box((px, py, H_z * 0.5), (0.035, 0.035, H_z * 0.5), obsidian, roughness=0.22, mat_code=0.15)
    mb.add_box((px, py, 0.08), (0.048, 0.048, 0.08), gold, roughness=0.15, mat_code=0.95)
    mb.add_box((px, py, H_z - 0.08), (0.048, 0.048, 0.08), gold, roughness=0.15, mat_code=0.95)
    # Recessed neon strip facing room center
    nx = -0.025 if px > 0 else 0.025
    mb.add_box((px + nx, py, H_z * 0.5), (0.008, 0.012, H_z * 0.42), ncol, roughness=0.2, mat_code=3.35)

  # Central Obsidian & Gold Pedestal + Refractive Crystal Glass Orb
  orb_xy = (0.0, 0.22)
  mb.add_cylinder((orb_xy[0], orb_xy[1], 0.0), 0.24, 0.06, gold, roughness=0.14, mat_code=0.95, n_seg=48)
  mb.add_cylinder((orb_xy[0], orb_xy[1], 0.06), 0.18, 0.30, obsidian, roughness=0.18, mat_code=0.18, n_seg=48)
  mb.add_cylinder((orb_xy[0], orb_xy[1], 0.36), 0.20, 0.035, gold, roughness=0.14, mat_code=0.95, n_seg=48)
  mb.add_sphere((orb_xy[0], orb_xy[1], 0.625), 0.23, (0.94, 0.99, 1.0), roughness=0.015, mat_code=2.00, n_lat=40, n_lon=80)

  # Hanging Neon & Warm-Gold Lanterns at staggered heights/depths (saturated cores)
  lanterns = [
      ((-0.44, 0.55, 1.18), 0.085, (1.0, 0.48, 0.06), 3.05),
      ((0.46, 0.58, 0.96), 0.080, (0.06, 0.72, 1.0), 3.05),
      ((0.02, 0.88, 1.38), 0.095, (1.0, 0.14, 0.56), 3.05),
      ((-0.55, -0.18, 1.42), 0.075, (0.06, 0.72, 1.0), 3.05),
      ((0.56, -0.12, 1.28), 0.080, (1.0, 0.48, 0.06), 3.05),
      ((0.04, -0.45, 1.56), 0.075, (1.0, 0.54, 0.10), 3.05),
      ((-0.38, 1.05, 0.72), 0.068, (1.0, 0.14, 0.56), 2.98),
      ((0.42, 1.08, 0.64), 0.068, (1.0, 0.48, 0.06), 2.98),
  ]
  for pos, rad, col, mc in lanterns:
    _add_lantern(mb, pos, rad, col, mc, ceiling_z=H_z)

  bvh = mb.build("obsidian_gold_infinite_mirror_gallery")
  cxp, cxm = _look_at_mat((-0.46, -1.16, 0.86), (0.18, 0.68, 0.68))
  rig = LightRig(
      area_center=(0.0, 0.22, 1.88),
      area_normal=(0.0, 0.0, -1.0),
      area_radius=0.45,
      area_emit=(6.2, 5.4, 4.4),
      has_sun=False,
      sky_zenith=(0.02, 0.02, 0.03),
      sky_horizon=(0.01, 0.01, 0.02),
      sky_ground=(0.005, 0.005, 0.008),
      exposure=1.05,
  )
  return bvh, cxp, cxm, 52.0, 0.05, 25.0, rig


# ==============================================================================
# Runner & 2x2 Dark-Cinema Benchmark
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
    opt_cfg: OpticsConfig,
):
  fn, mesh, rep = build_wave7_pathtracer(
      bvh, devices, H, W, fovy, znear, zfar, rig,
      spp=spp, max_bounces=max_bounces, opt_cfg=opt_cfg,
  )
  cxp = jax.device_put(cxp_np, rep)
  cxm = jax.device_put(cxm_np, rep)
  rng = jax.device_put(jax.random.PRNGKey(21), rep)
  hlo_info = audit_hlo_zero_copy(fn.lower(cxp, cxm, rng))

  tc0 = time.perf_counter()
  y = fn(cxp, cxm, rng)
  jax.block_until_ready(y)
  compile_s = time.perf_counter() - tc0

  n_steps = 1
  t0 = time.perf_counter()
  for it in range(n_steps):
    y = fn(cxp, cxm, jax.device_put(jax.random.PRNGKey(50 + it), rep))
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
      "thinfilm": opt_cfg.thinfilm,
      "anisotropic": opt_cfg.anisotropic,
      "n_chips": len(devices),
      "compile_s": round(compile_s, 2),
      "ms": round(ms, 2),
      "images_per_sec": round(fps, 3),
      "samples_per_min": round(spm, 1),
      "hlo_zero_copy": True,
      "hlo_ops": hlo_info,
  }
  print("WAVE7_RUN:", rec, flush=True)
  return rec, img


def _load_header_font(size: int = 16):
  candidates = [
      "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
      "/usr/share/fonts/truetype/noto/NotoSansMono-Bold.ttf",
  ]
  for path in candidates:
    if os.path.exists(path):
      return ImageFont.truetype(path, size=size)
  try:
    return ImageFont.load_default(size=size)
  except TypeError:
    return ImageFont.load_default()


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--res", type=int, default=640)
  ap.add_argument("--spp", type=int, default=128)
  ap.add_argument("--out_json", default="out/metrics/v3_w7_wave7_thinfilm_mirror.json")
  ap.add_argument("--out_png", default="out/figures/v3_w7_thinfilm_mirror_fx.png")
  args = ap.parse_args()

  devs = jax.devices()
  assert len(devs) == 4, f"Expected 4 TPU chips, got {len(devs)}"
  H = W = args.res

  recs = []
  panels: List[Tuple[str, np.ndarray]] = []

  # Top Row: Thin-Film Interference & Anisotropic Brushed Metal vs Isotropic Baseline
  bvh_tf, cxp_tf, cxm_tf, fovy_tf, zn_tf, zf_tf, rig_tf = build_thinfilm_aniso_scene()
  r_tf_base, img_tf_base = _run_config(
      "isotropic_no_thinfilm_baseline", bvh_tf, cxp_tf, cxm_tf, fovy_tf, zn_tf, zf_tf, rig_tf,
      devs, H, W, spp=args.spp, max_bounces=6,
      opt_cfg=OpticsConfig(thinfilm=False, anisotropic=False, alpha_iso=0.15),
  )
  recs.append(r_tf_base)
  panels.append((f"Isotropic Baseline (No Interference | {args.spp} spp | {r_tf_base['ms']:.0f}ms)", img_tf_base))

  r_tf_full, img_tf_full = _run_config(
      "airy_thinfilm_anisotropic_ggx", bvh_tf, cxp_tf, cxm_tf, fovy_tf, zn_tf, zf_tf, rig_tf,
      devs, H, W, spp=args.spp, max_bounces=6,
      opt_cfg=OpticsConfig(thinfilm=True, anisotropic=True, alpha_u=0.05, alpha_v=0.45),
  )
  recs.append(r_tf_full)
  panels.append((f"Airy Thin-Film (280-620nm) + Aniso GGX (0.05/0.45) | {r_tf_full['ms']:.0f}ms", img_tf_full))

  # Bottom Row: Deep 12-Bounce Infinite Mirror Gallery (2-bounce vs 12-bounce)
  bvh_mg, cxp_mg, cxm_mg, fovy_mg, zn_mg, zf_mg, rig_mg = build_infinite_mirror_gallery_scene()
  r_mg_2, img_mg_2 = _run_config(
      "mirror_gallery_2bounce", bvh_mg, cxp_mg, cxm_mg, fovy_mg, zn_mg, zf_mg, rig_mg,
      devs, H, W, spp=args.spp, max_bounces=2,
      opt_cfg=OpticsConfig(thinfilm=False, anisotropic=False),
  )
  recs.append(r_mg_2)
  panels.append((f"Infinite Mirror Gallery (max_bounces=2 | {args.spp} spp | {r_mg_2['ms']:.0f}ms)", img_mg_2))

  r_mg_12, img_mg_12 = _run_config(
      "mirror_gallery_12bounce", bvh_mg, cxp_mg, cxm_mg, fovy_mg, zn_mg, zf_mg, rig_mg,
      devs, H, W, spp=args.spp, max_bounces=12,
      opt_cfg=OpticsConfig(thinfilm=False, anisotropic=False),
  )
  recs.append(r_mg_12)
  panels.append((f"Infinite Mirror Gallery (max_bounces=12 | {args.spp} spp | {r_mg_12['ms']:.0f}ms)", img_mg_12))

  # Compose crisp 2x2 dark-cinema showcase PNG with centered headers
  pad, hdr = 16, 46
  canvas = Image.new("RGB", (2 * W + 3 * pad, 2 * (H + hdr) + 3 * pad), (12, 14, 18))
  draw = ImageDraw.Draw(canvas)
  font = _load_header_font(size=15)
  for idx, (title, im) in enumerate(panels):
    r, c = idx // 2, idx % 2
    x0 = pad + c * (W + pad)
    y0 = pad + r * (H + hdr + pad)
    u8 = np.clip(im * 255.0 + 0.5, 0, 255).astype(np.uint8)
    canvas.paste(Image.fromarray(u8), (x0, y0 + hdr))
    draw.text((x0 + W // 2, y0 + hdr // 2), title, fill=(242, 246, 252), font=font, anchor="mm")

  os.makedirs(os.path.dirname(args.out_png) or ".", exist_ok=True)
  canvas.save(args.out_png)
  print("Saved Showcase PNG ->", args.out_png, canvas.size, flush=True)

  os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
  with open(args.out_json, "w") as f:
    json.dump({
        "experiment": "v3_w7_wave7_thinfilm_mirror",
        "backend": jax.default_backend(),
        "n_chips": len(devs),
        "spectral_quadrature_bins": N_SPEC_BINS,
        "wavelength_range_nm": [LAMBDA_MIN_NM, LAMBDA_MAX_NM],
        "bubble_thickness_nm": [280.0, 620.0],
        "anisotropic_roughness": {"alpha_u": 0.05, "alpha_v": 0.45},
        "runs": recs,
    }, f, indent=2)
  print("Saved Metrics JSON ->", args.out_json, flush=True)


if __name__ == "__main__":
  main()
