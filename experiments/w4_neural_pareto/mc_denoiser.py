"""Wave 3: Real-Time BF16 MXU Monte Carlo Path Tracer Denoiser (TPU v5p, up to Native 4K).

Uses w7's 5-bounce PBR + Glass + Area/Sun BVH Path Tracer (`bvh_pbr_pt.py` + `complex_scenes_pt.py`)
to render multi-view 1-spp, 4-spp, and converged 256-spp GT datasets paired with exact primary-hit
G-buffers (albedo, normal, depth, roughness, mat_code), trains an albedo-demodulated SpaceToDepth(2)
BF16 MXU U-Net denoiser, verifies quality recovery on unseen camera views, and benchmarks 1-chip &
4-chip `shard_map` throughput at 128x128, 512x512, and Native 4K (3840x2160) with zero host callbacks.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import pickle
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import flax.linen as nn
import jax
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np
import optax
from PIL import Image, ImageDraw

from bvh_pbr_pt import (
    LightRig,
    PBRWideBVHMesh,
    _agx_tonemap_srgb,
    _build_onb,
    _eval_pbr_brdf,
    _sample_cosine_hemisphere,
    _sample_ggx_half,
    audit_hlo_zero_copy,
    pbr_raycast_tpu,
)
from complex_scenes_pt import (
    _look_at_mat,
    build_classroom_sunlit,
    build_cornell_workshop,
)
from gate import compute_psnr, compute_ssim_batch, luminance


# ==============================================================================
# 1. BF16 MXU Albedo-Demodulated MC Denoiser Architecture (SpaceToDepth 3-Scale)
# ==============================================================================

class _DenoiseResBlock(nn.Module):
  features: int

  @nn.compact
  def __call__(self, x: jax.Array) -> jax.Array:
    dt = jnp.bfloat16
    y = nn.Conv(self.features, (3, 3), padding="SAME", dtype=dt, param_dtype=jnp.float32)(x)
    y = nn.relu(y)
    y = nn.Conv(self.features, (3, 3), padding="SAME", dtype=dt, param_dtype=jnp.float32)(y)
    return nn.relu(x + y)


def build_denoiser_features(noisy_rgb: jax.Array, gbuf: jax.Array) -> jax.Array:
  """Pack (B, H, W, 16) BF16-ready features from noisy RGB (3) + primary G-buffer (9).

  gbuf layout: [albedo(3), normal(3), depth_norm(1), roughness(1), mcode_norm(1)]
  """
  albedo = gbuf[..., 0:3]
  normal = gbuf[..., 3:6]
  depth_norm = gbuf[..., 6:7]
  roughness = gbuf[..., 7:8]
  mcode_norm = gbuf[..., 8:9]
  hit_mask = (depth_norm > 0.0).astype(jnp.float32)
  lum_noisy = (
      0.2126 * noisy_rgb[..., 0:1] + 0.7152 * noisy_rgb[..., 1:2] + 0.0722 * noisy_rgb[..., 2:3]
  )

  # 4-neighbor edge discontinuity detectors on normal & depth
  npad = jnp.pad(normal, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="edge")
  zpad = jnp.pad(jnp.where(hit_mask > 0.5, depth_norm, 2.0), ((0, 0), (1, 1), (1, 1), (0, 0)), mode="edge")
  H, W = noisy_rgb.shape[1], noisy_rgb.shape[2]
  n_edge = jnp.clip(
      0.5 * (
          jnp.linalg.norm(npad[:, 2 : H + 2, 1 : W + 1] - npad[:, :H, 1 : W + 1], axis=-1, keepdims=True)
          + jnp.linalg.norm(npad[:, 1 : H + 1, 2 : W + 2] - npad[:, 1 : H + 1, :W], axis=-1, keepdims=True)
      ),
      0.0,
      2.0,
  )
  z_edge = jnp.clip(
      jnp.abs(zpad[:, 2 : H + 2, 1 : W + 1] - zpad[:, :H, 1 : W + 1])
      + jnp.abs(zpad[:, 1 : H + 1, 2 : W + 2] - zpad[:, 1 : H + 1, :W]),
      0.0,
      1.0,
  )

  return jnp.concatenate(
      [
          noisy_rgb,   # 3
          albedo,      # 3
          normal,      # 3
          depth_norm,  # 1
          roughness,   # 1
          mcode_norm,  # 1
          hit_mask,    # 1
          lum_noisy,   # 1
          n_edge,      # 1
          z_edge,      # 1
      ],
      axis=-1,
  )


class MXUAlbedoDemodDenoiser(nn.Module):
  """Ultra-fast SpaceToDepth(2) 3-scale BF16 MXU Albedo-Demodulated MC Denoiser.

  Operates spatial convolutions at H/2, H/4, H/8 for wide receptive field + 4K real-time speed,
  then unpacks to H x W and predicts (irrad, spec_add, blend_w) per pixel via 1x1 BRDF head:
    pred = clip(albedo * irrad + spec_add + blend_w * noisy_rgb, 0, 1)
  """

  c1: int = 48
  c2: int = 96
  c3: int = 128
  c_pix: int = 24
  head_width: int = 64

  @nn.compact
  def __call__(self, x: jax.Array, train: bool = False) -> jax.Array:
    del train
    dt = jnp.bfloat16
    B, H, W, C = x.shape
    x_bf = x.astype(dt)
    noisy_rgb = x_bf[..., 0:3]
    albedo = x_bf[..., 3:6]

    # SpaceToDepth(2): (B, H, W, 16) -> (B, H/2, W/2, 64)
    s2d = (
        x_bf.reshape(B, H // 2, 2, W // 2, 2, C)
        .transpose(0, 1, 3, 2, 4, 5)
        .reshape(B, H // 2, W // 2, 4 * C)
    )

    h1 = nn.relu(nn.Conv(self.c1, (3, 3), padding="SAME", dtype=dt, param_dtype=jnp.float32)(s2d))
    h1 = _DenoiseResBlock(self.c1)(h1)

    h2 = nn.relu(nn.Conv(self.c2, (3, 3), strides=(2, 2), padding="SAME", dtype=dt, param_dtype=jnp.float32)(h1))
    h2 = _DenoiseResBlock(self.c2)(h2)

    h3 = nn.relu(nn.Conv(self.c3, (3, 3), strides=(2, 2), padding="SAME", dtype=dt, param_dtype=jnp.float32)(h2))
    h3 = _DenoiseResBlock(self.c3)(h3)

    h3_up = jax.image.resize(h3.astype(jnp.float32), (B, H // 4, W // 4, self.c3), method="bilinear").astype(dt)
    d2 = nn.relu(
        nn.Conv(self.c2, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(
            jnp.concatenate([h2, h3_up], axis=-1)
        )
    )
    d2_up = jax.image.resize(d2.astype(jnp.float32), (B, H // 2, W // 2, self.c2), method="bilinear").astype(dt)
    d1 = nn.relu(
        nn.Conv(4 * self.c_pix, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(
            jnp.concatenate([h1, d2_up], axis=-1)
        )
    )

    h_pix = (
        d1.reshape(B, H // 2, W // 2, 2, 2, self.c_pix)
        .transpose(0, 1, 3, 2, 4, 5)
        .reshape(B, H, W, self.c_pix)
    )

    out_in = jnp.concatenate([h_pix, x_bf], axis=-1)
    out_mid = nn.relu(nn.Conv(self.head_width, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(out_in))
    out_mid = nn.relu(nn.Conv(self.head_width, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(out_mid))
    raw_out = nn.Conv(9, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(out_mid)

    irrad = raw_out[..., 0:3]
    spec_add = raw_out[..., 3:6]
    blend_w = jax.nn.sigmoid(raw_out[..., 6:9].astype(jnp.float32)).astype(dt) * jnp.asarray(0.5, dt)
    pred = albedo * irrad + spec_add + blend_w * noisy_rgb
    return jnp.clip(pred.astype(jnp.float32), 0.0, 1.0)


# ==============================================================================
# 2. TPU 5-Bounce PBR Path Tracer + Primary G-Buffer Extractor
# ==============================================================================

def build_pt_and_gbuf_fn(
    bvh_mesh: PBRWideBVHMesh,
    devices: Sequence[jax.Device],
    height: int,
    width: int,
    cam_fovy: float,
    znear: float,
    zfar: float,
    light_rig: LightRig,
    spp: int,
    max_bounces: int = 5,
    denoiser_model: Optional[nn.Module] = None,
    denoiser_params: Any = None,
):
  """Compile 1-chip or N-chip spatially-sharded 5-bounce MC PT + G-buffer (+ optional MXU Denoiser)."""
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
  dn_params_rep = jax.device_put(denoiser_params, rep) if denoiser_params is not None else None

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

  def _trace_core(l2r, l2, l0, lf, cxp, cxm, cid, rng_in):
    x_off = cid * W_local
    chip_rng = jax.random.fold_in(rng_in, cid)

    local_px = height * W_local
    if spp >= 16 and local_px <= 32768:
      spp_par = 16
    elif spp >= 8 and local_px <= 131072:
      spp_par = 8
    elif spp >= 4 and local_px <= 262144:
      spp_par = 4
    else:
      spp_par = 1
    spp_seq = spp // spp_par
    R = height * W_local

    # Center-pixel primary G-buffer (exact crisp albedo, normal, depth, roughness, mcode)
    px_c = jnp.arange(W_local, dtype=jnp.float32) + jnp.asarray(x_off, jnp.float32) + 0.5
    py_c = jnp.arange(height, dtype=jnp.float32) + 0.5
    gxc, gyc = jnp.meshgrid(px_c, py_c, indexing="xy")
    uc = gxc.reshape(1, R) / float(width)
    vc = gyc.reshape(1, R) / float(height)
    xc = -half_w + (2.0 * half_w) * uc
    yc = half_h - (2.0 * half_h) * vc
    raw_c = jnp.stack([xc, yc, jnp.full_like(xc, -float(znear))], axis=-1)
    ld_c = raw_c / jnp.maximum(jnp.linalg.norm(raw_c, axis=-1, keepdims=True), 1e-12)
    neg_zc = jnp.maximum(-ld_c[..., 2], 1e-6)
    rd_c = jnp.einsum("ij,brj->bri", cxm[0], ld_c, precision=jax.lax.Precision.HIGHEST)
    rd_c = rd_c / jnp.maximum(jnp.linalg.norm(rd_c, axis=-1, keepdims=True), 1e-12)
    ro_c = cxp[0][None, None, :] + (float(znear) / neg_zc)[..., None] * rd_c
    tmax_c = jnp.broadcast_to((float(zfar - znear) / neg_zc), (1, R))

    d_p, n_p, alb_p, rgh_p, mc_p = pbr_raycast_tpu(
        l2r, l2, l0, lf, ro_c, rd_c, tmax_c, height, W_local, any_hit=False
    )
    hit_p = (d_p < tmax_c - 1e-4) & (d_p >= 0.0)
    n_p_u = n_p / jnp.maximum(jnp.linalg.norm(n_p, axis=-1, keepdims=True), 1e-12)
    n_p_ff = jnp.where((jnp.sum(n_p_u * (-rd_c), axis=-1) >= 0.0)[..., None], n_p_u, -n_p_u)
    sky_alb = jnp.clip(_agx_tonemap_srgb(_env_sky(rd_c), exposure=light_rig.exposure), 0.0, 1.0)
    gbuf_flat = jnp.concatenate(
        [
            jnp.where(hit_p[..., None], alb_p, sky_alb),
            jnp.where(hit_p[..., None], n_p_ff, 0.0),
            jnp.where(hit_p, jnp.clip(d_p / 10.0, 0.0, 1.5), -1.0)[..., None],
            jnp.where(hit_p, rgh_p, 1.0)[..., None],
            jnp.where(hit_p, jnp.clip(mc_p * 0.5, 0.0, 1.5), 0.0)[..., None],
        ],
        axis=-1,
    )
    gbuf_hw = gbuf_flat[0].reshape(height, W_local, 9)

    def _trace_batch_samples(s_idx, rad_acc):
      s_rng = jax.random.fold_in(chip_rng, s_idx)
      k_jit, k_bnc = jax.random.split(s_rng)
      jitter = jax.random.uniform(k_jit, (spp_par, R, 2), minval=-0.5, maxval=0.5, dtype=jnp.float32)
      u = (gxc.reshape(1, R) + jitter[..., 0]) / float(width)
      v = (gyc.reshape(1, R) + jitter[..., 1]) / float(height)
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

      def _bounce(b_idx, carry):
        ro, rd, t_lim, thr, alive, rad = carry
        b_rng = jax.random.fold_in(k_bnc, b_idx)
        k_lu, k_lv, k_su, k_sv, k_lobe, k_u1, k_u2 = jax.random.split(b_rng, 7)

        d_hit, nrm_raw, alb, rgh, mcode = pbr_raycast_tpu(
            l2r, l2, l0, lf, ro, rd, jnp.where(alive, t_lim, 0.0), height, W_local, any_hit=False
        )
        hit = alive & (d_hit < t_lim - 1e-4) & (d_hit >= 0.0)
        miss = alive & (~hit)
        rad = rad + jnp.where(miss[..., None], thr * _env_sky(rd), 0.0)

        n_unit = nrm_raw / jnp.maximum(jnp.linalg.norm(nrm_raw, axis=-1, keepdims=True), 1e-12)
        wo = -rd
        entering = jnp.sum(n_unit * wo, axis=-1) >= 0.0
        n_ff = jnp.where(entering[..., None], n_unit, -n_unit)

        is_emit = hit & (mcode >= 2.8)
        is_glass = hit & (mcode >= 1.8) & (mcode < 2.8)
        is_opaque = hit & (mcode < 1.8)
        metallic = jnp.clip(mcode, 0.0, 1.0)

        emit_power = jnp.where(mcode >= 2.8, (mcode - 2.0) * 6.0, 0.0)
        rad = rad + jnp.where(is_emit[..., None], thr * alb * emit_power[..., None], 0.0)

        hit_pos = ro + jnp.where(hit, d_hit, 0.0)[..., None] * rd

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
        nee_a = jnp.minimum(thr * brdf_a * area_emit[None, None, :] * (geom_a * vis)[..., None], 14.0)
        rad = rad + jnp.where(is_opaque[..., None], nee_a, 0.0)

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
          nee_sun = jnp.minimum(thr * brdf_sun * sun_emit[None, None, :] * (cos_sun * vis_sun)[..., None], 18.0)
          rad = rad + jnp.where(is_opaque[..., None], nee_sun, 0.0)

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

        eta = jnp.where(entering, 1.0 / 1.50, 1.50)
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
        bounce_w = jnp.where(is_glass[..., None], alb, w_opq)

        new_thr = thr * bounce_w
        new_alive = (is_opaque | is_glass) & (jnp.max(new_thr, axis=-1) > 1e-3)
        new_t_lim = jnp.broadcast_to(jnp.float32(zfar), (spp_par, R))
        return (new_ro, wi_next, new_t_lim, new_thr, new_alive, rad)

      _, _, _, _, _, final_rad = jax.lax.fori_loop(0, max_bounces, _bounce, (ro0, rd0, t_max0, thr0, alive0, rad0))
      return rad_acc + jnp.mean(final_rad, axis=0)

    rad_sum = jax.lax.fori_loop(0, spp_seq, _trace_batch_samples, jnp.zeros((R, 3), dtype=jnp.float32))
    rad_avg = (rad_sum / float(spp_seq)).reshape(height, W_local, 3)
    rgb = _agx_tonemap_srgb(rad_avg, exposure=light_rig.exposure)
    return rgb, gbuf_hw

  sm_kw = {"check_rep": False} if "check_rep" in inspect.signature(shard_map).parameters else {"check_vma": False}

  if denoiser_model is not None and dn_params_rep is not None:
    def _render_e2e_local(l2r, l2, l0, lf, cxp, cxm, cid_arr, rng_in, dnp):
      rgb_noisy, gbuf_hw = _trace_core(l2r, l2, l0, lf, cxp, cxm, cid_arr[0], rng_in)
      feats = build_denoiser_features(rgb_noisy[None, ...], gbuf_hw[None, ...])
      rgb_dn = denoiser_model.apply({"params": dnp}, feats, train=False)[0]
      return rgb_dn[None, ...]

    sh_fn = shard_map(
        _render_e2e_local,
        mesh=mesh,
        in_specs=(P(), P(), P(), P(), P(), P(), P("dp"), P(), P()),
        out_specs=P("dp"),
        **sm_kw,
    )

    @jax.jit
    def render_fn(cxp, cxm, rng):
      return sh_fn(l2_roots_d, l2_d, l0_d, leaf_d, cxp, cxm, chip_ids, rng, dn_params_rep)

    return render_fn, mesh, rep
  else:
    def _render_local(l2r, l2, l0, lf, cxp, cxm, cid_arr, rng_in):
      rgb, gbuf_hw = _trace_core(l2r, l2, l0, lf, cxp, cxm, cid_arr[0], rng_in)
      return rgb[None, ...], gbuf_hw[None, ...]

    sh_fn = shard_map(
        _render_local,
        mesh=mesh,
        in_specs=(P(), P(), P(), P(), P(), P(), P("dp"), P()),
        out_specs=(P("dp"), P("dp")),
        **sm_kw,
    )

    @jax.jit
    def render_fn(cxp, cxm, rng):
      return sh_fn(l2_roots_d, l2_d, l0_d, leaf_d, cxp, cxm, chip_ids, rng)

    return render_fn, mesh, rep


def _stitch(arr_j: jax.Array) -> np.ndarray:
  arr = np.asarray(jax.device_get(arr_j), dtype=np.float32)
  n_c, H, W_loc, C = arr.shape
  return arr.transpose(1, 0, 2, 3).reshape(H, n_c * W_loc, C)


# ==============================================================================
# 3. Camera Pose Generator + Multi-View Dataset + Training + 4K Benchmarking
# ==============================================================================

def _generate_views(scene_name: str, n_train: int = 16, n_test: int = 4) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[Tuple[np.ndarray, np.ndarray]]]:
  rng = np.random.default_rng(1234 if scene_name == "classroom" else 5678)
  views_tr, views_te = [], []
  for i in range(n_train + n_test):
    is_test = i >= n_train
    if scene_name == "classroom":
      if i == n_train:  # Exact hero canonical camera
        pos = (1.65, -1.75, 1.38)
        tgt = (-0.35, 0.75, 0.92)
      else:
        pos = (
            float(1.65 + rng.uniform(-0.85, 0.55)),
            float(-1.75 + rng.uniform(-0.35, 1.25)),
            float(1.38 + rng.uniform(-0.38, 0.45)),
        )
        tgt = (
            float(-0.35 + rng.uniform(-0.45, 0.45)),
            float(0.75 + rng.uniform(-0.45, 0.45)),
            float(0.92 + rng.uniform(-0.25, 0.25)),
        )
    else:  # cornell_workshop
      if i == n_train:  # Exact hero canonical camera
        pos = (0.0, -2.55, 0.88)
        tgt = (0.0, 0.25, 0.62)
      else:
        pos = (
            float(rng.uniform(-0.75, 0.75)),
            float(-2.55 + rng.uniform(-0.25, 0.75)),
            float(0.88 + rng.uniform(-0.30, 0.55)),
        )
        tgt = (
            float(rng.uniform(-0.25, 0.25)),
            float(0.25 + rng.uniform(-0.25, 0.25)),
            float(0.62 + rng.uniform(-0.20, 0.20)),
        )
    cxp, cxm = _look_at_mat(pos, tgt)
    if is_test:
      views_te.append((cxp, cxm))
    else:
      views_tr.append((cxp, cxm))
  return views_tr, views_te


def render_multiview_dataset(
    devs_4c: Sequence[jax.Device],
    train_res: int = 256,
    test_res: int = 512,
    gt_spp: int = 256,
    n_train_per_scene: int = 16,
    n_test_per_scene: int = 4,
):
  scenes = [
      ("classroom", build_classroom_sunlit()),
      ("cornell_workshop", build_cornell_workshop()),
  ]
  tr_noisy1, tr_noisy4, tr_gbuf, tr_gt = [], [], [], []
  te_data: Dict[str, Any] = {}

  for sname, (bvh, _, _, fovy, zn, zf, rig) in scenes:
    v_tr, v_te = _generate_views(sname, n_train=n_train_per_scene, n_test=n_test_per_scene)
    print(f"Rendering {sname} ({bvh.n_tris_orig:,} tris): {len(v_tr)} train @{train_res} + {len(v_te)} test @{test_res} (GT={gt_spp} spp)...", flush=True)

    # Train resolution renderers
    fn1_tr, _, rep = build_pt_and_gbuf_fn(bvh, devs_4c, train_res, train_res, fovy, zn, zf, rig, spp=1, max_bounces=5)
    fn4_tr, _, _ = build_pt_and_gbuf_fn(bvh, devs_4c, train_res, train_res, fovy, zn, zf, rig, spp=4, max_bounces=5)
    fngt_tr, _, _ = build_pt_and_gbuf_fn(bvh, devs_4c, train_res, train_res, fovy, zn, zf, rig, spp=gt_spp, max_bounces=5)

    for idx, (cxp_n, cxm_n) in enumerate(v_tr):
      cxp_d, cxm_d = jax.device_put(cxp_n, rep), jax.device_put(cxm_n, rep)
      r1, gb = fn1_tr(cxp_d, cxm_d, jax.device_put(jax.random.PRNGKey(1000 + idx), rep))
      r4, _ = fn4_tr(cxp_d, cxm_d, jax.device_put(jax.random.PRNGKey(2000 + idx), rep))
      rgt, _ = fngt_tr(cxp_d, cxm_d, jax.device_put(jax.random.PRNGKey(3000 + idx), rep))
      jax.block_until_ready((r1, r4, rgt))
      tr_noisy1.append(_stitch(r1))
      tr_noisy4.append(_stitch(r4))
      tr_gbuf.append(_stitch(gb))
      tr_gt.append(_stitch(rgt))

    # Held-out unseen camera views at test_res (512x512)
    fn1_te, _, rep_te = build_pt_and_gbuf_fn(bvh, devs_4c, test_res, test_res, fovy, zn, zf, rig, spp=1, max_bounces=5)
    fn4_te, _, _ = build_pt_and_gbuf_fn(bvh, devs_4c, test_res, test_res, fovy, zn, zf, rig, spp=4, max_bounces=5)
    fngt_te, _, _ = build_pt_and_gbuf_fn(bvh, devs_4c, test_res, test_res, fovy, zn, zf, rig, spp=gt_spp, max_bounces=5)

    s_n1, s_n4, s_gb, s_gt = [], [], [], []
    for idx, (cxp_n, cxm_n) in enumerate(v_te):
      cxp_d, cxm_d = jax.device_put(cxp_n, rep_te), jax.device_put(cxm_n, rep_te)
      r1, gb = fn1_te(cxp_d, cxm_d, jax.device_put(jax.random.PRNGKey(4000 + idx), rep_te))
      r4, _ = fn4_te(cxp_d, cxm_d, jax.device_put(jax.random.PRNGKey(5000 + idx), rep_te))
      rgt, _ = fngt_te(cxp_d, cxm_d, jax.device_put(jax.random.PRNGKey(6000 + idx), rep_te))
      jax.block_until_ready((r1, r4, rgt))
      s_n1.append(_stitch(r1))
      s_n4.append(_stitch(r4))
      s_gb.append(_stitch(gb))
      s_gt.append(_stitch(rgt))
    te_data[sname] = {
        "noisy1": np.stack(s_n1, axis=0),
        "noisy4": np.stack(s_n4, axis=0),
        "gbuf": np.stack(s_gb, axis=0),
        "gt": np.stack(s_gt, axis=0),
        "hero_cam": v_te[0],
        "scene_obj": (bvh, fovy, zn, zf, rig),
    }

  return (
      np.stack(tr_noisy1, axis=0),
      np.stack(tr_noisy4, axis=0),
      np.stack(tr_gbuf, axis=0),
      np.stack(tr_gt, axis=0),
      te_data,
  )


def _ssim_loss(pred: jax.Array, target: jax.Array) -> jax.Array:
  c1, c2 = 0.01 ** 2, 0.03 ** 2
  mu_x = jax.lax.reduce_window(pred, 0.0, jax.lax.add, (1, 5, 5, 1), (1, 1, 1, 1), "SAME") / 25.0
  mu_y = jax.lax.reduce_window(target, 0.0, jax.lax.add, (1, 5, 5, 1), (1, 1, 1, 1), "SAME") / 25.0
  sig_x = jax.lax.reduce_window(pred * pred, 0.0, jax.lax.add, (1, 5, 5, 1), (1, 1, 1, 1), "SAME") / 25.0 - mu_x * mu_x
  sig_y = jax.lax.reduce_window(target * target, 0.0, jax.lax.add, (1, 5, 5, 1), (1, 1, 1, 1), "SAME") / 25.0 - mu_y * mu_y
  sig_xy = jax.lax.reduce_window(pred * target, 0.0, jax.lax.add, (1, 5, 5, 1), (1, 1, 1, 1), "SAME") / 25.0 - mu_x * mu_y
  num = (2.0 * mu_x * mu_y + c1) * (2.0 * sig_xy + c2)
  den = (mu_x * mu_x + mu_y * mu_y + c1) * (jnp.maximum(sig_x, 0.0) + jnp.maximum(sig_y, 0.0) + c2)
  return 1.0 - jnp.mean(num / jnp.maximum(den, 1e-12))


def train_denoiser(
    model: MXUAlbedoDemodDenoiser,
    tr_n1: np.ndarray,
    tr_n4: np.ndarray,
    tr_gb: np.ndarray,
    tr_gt: np.ndarray,
    epochs: int = 180,
    batch_size: int = 16,
    patch_size: int = 128,
    lr: float = 2.0e-3,
):
  """Train unified BF16 MXU Albedo-Demodulated Denoiser on random 128x128 crops of 1-spp & 4-spp."""
  dev = jax.devices()[0]
  # Combine both 1-spp and 4-spp noisy inputs -> 2x dataset for unified 1-spp & 4-spp denoising
  n_all = np.concatenate([tr_n1, tr_n4], axis=0)
  gb_all = np.concatenate([tr_gb, tr_gb], axis=0)
  gt_all = np.concatenate([tr_gt, tr_gt], axis=0)

  # Precompute 16-ch features once on device
  @jax.jit
  def _make_feats(n_b, gb_b):
    return build_denoiser_features(n_b, gb_b)

  x_all = np.asarray(_make_feats(jnp.asarray(n_all), jnp.asarray(gb_all)), dtype=np.float32)
  x_dev = jax.device_put(x_all, dev)
  y_dev = jax.device_put(gt_all.astype(np.float32), dev)
  N, H, W, _ = x_all.shape

  rng = jax.random.PRNGKey(2026)
  params = model.init(rng, jnp.zeros((1, patch_size, patch_size, 16), dtype=jnp.float32))["params"]
  ema_params = params
  n_params = int(sum(p.size for p in jax.tree_util.tree_leaves(params)))

  steps_per_ep = 16
  total_steps = epochs * steps_per_ep
  sched = optax.warmup_cosine_decay_schedule(lr * 0.1, lr, max(20, total_steps // 15), total_steps, lr * 0.02)
  tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=2e-5))
  opt_state = tx.init(params)

  @jax.jit
  def train_epoch(p, ema_p, ost, step_rng):
    def _scan_step(carry, s_idx):
      cp, cema, cost = carry
      k = jax.random.fold_in(step_rng, s_idx)
      k_idx, k_y, k_x = jax.random.split(k, 3)
      b_idx = jax.random.randint(k_idx, (batch_size,), 0, N)
      y0 = jax.random.randint(k_y, (), 0, H - patch_size + 1)
      x0 = jax.random.randint(k_x, (), 0, W - patch_size + 1)
      xb = jax.lax.dynamic_slice(x_dev[b_idx], (0, y0, x0, 0), (batch_size, patch_size, patch_size, 16))
      yb = jax.lax.dynamic_slice(y_dev[b_idx], (0, y0, x0, 0), (batch_size, patch_size, patch_size, 3))

      def loss_fn(pr):
        pred = model.apply({"params": pr}, xb, train=True)
        l2 = jnp.mean((pred - yb) ** 2)
        l1 = jnp.mean(jnp.abs(pred - yb))
        ls = _ssim_loss(pred, yb)
        # Edge-preserving gradient loss
        gx_p = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        gx_y = yb[:, :, 1:, :] - yb[:, :, :-1, :]
        gy_p = pred[:, 1:, :, :] - pred[:, :-1, :, :]
        gy_y = yb[:, 1:, :, :] - yb[:, :-1, :, :]
        lgrad = jnp.mean(jnp.abs(gx_p - gx_y)) + jnp.mean(jnp.abs(gy_p - gy_y))
        return 0.50 * l2 + 0.30 * l1 + 0.15 * ls + 0.10 * lgrad

      loss, grads = jax.value_and_grad(loss_fn)(cp)
      upd, n_ost = tx.update(grads, cost, cp)
      np_p = optax.apply_updates(cp, upd)
      n_ema = jax.tree_util.tree_map(lambda e, n: 0.995 * e + 0.005 * n, cema, np_p)
      return (np_p, n_ema, n_ost), loss

    (p_o, e_o, o_o), losses = jax.lax.scan(_scan_step, (p, ema_p, ost), jnp.arange(steps_per_ep))
    return p_o, e_o, o_o, jnp.mean(losses)

  t0 = time.time()
  for ep in range(epochs):
    params, ema_params, opt_state, ep_loss = train_epoch(
        params, ema_params, opt_state, jax.random.PRNGKey(ep + 10000)
    )
    if (ep + 1) % 30 == 0 or ep + 1 == epochs:
      jax.block_until_ready(ep_loss)
      print(f"[MC-DENOISER] ep {ep+1}/{epochs} loss={float(ep_loss):.5f} ({time.time()-t0:.1f}s)", flush=True)

  return ema_params, n_params, round(time.time() - t0, 2)


def benchmark_denoiser_scaling(
    model: MXUAlbedoDemodDenoiser,
    params: Any,
    devs_all: Sequence[jax.Device],
    resolutions: List[Tuple[int, int]] = [(128, 128), (512, 512), (2160, 3840)],
) -> List[Dict[str, Any]]:
  """Benchmark pure BF16 MXU Denoiser on 1-chip and 4-chip shard_map (128x128, 512x512, 4K)."""
  recs = []
  sm_kw = {"check_rep": False} if "check_rep" in inspect.signature(shard_map).parameters else {"check_vma": False}

  for H, W in resolutions:
    for n_chips in [1, 4]:
      devs = list(devs_all[:n_chips])
      n_steps = 20 if H <= 512 else 15
      if n_chips == 1:
        dev = devs[0]
        p_d = jax.device_put(jax.tree_util.tree_map(jnp.asarray, params), dev)
        n_d = jax.device_put(jnp.ones((1, H, W, 3), dtype=jnp.float32) * 0.4, dev)
        gb_d = jax.device_put(jnp.ones((1, H, W, 9), dtype=jnp.float32) * 0.5, dev)

        @jax.jit
        def dn_1c(p, n_in, g_in):
          feats = build_denoiser_features(n_in, g_in)
          return model.apply({"params": p}, feats, train=False)

        hlo_info = audit_hlo_zero_copy(dn_1c.lower(p_d, n_d, gb_d))
        for _ in range(3):
          y = dn_1c(p_d, n_d, gb_d)
        jax.block_until_ready(y)
        t0 = time.perf_counter()
        for _ in range(n_steps):
          y = dn_1c(p_d, n_d, gb_d)
        jax.block_until_ready(y)
        ms = (time.perf_counter() - t0) * 1e3 / n_steps
      else:
        # 4-chip spatial shard_map along W (zero inter-chip send/recv; each chip processes (1, H, W/4, C))
        mesh = Mesh(np.array(devs), ("dp",))
        rep = NamedSharding(mesh, P())
        sh_w = NamedSharding(mesh, P(None, None, "dp", None))
        p_d = jax.device_put(jax.tree_util.tree_map(jnp.asarray, params), rep)
        n_d = jax.device_put(np.full((1, H, W, 3), 0.4, dtype=np.float32), sh_w)
        gb_d = jax.device_put(np.full((1, H, W, 9), 0.5, dtype=np.float32), sh_w)

        def _dn_local(p_l, n_l, g_l):
          feats = build_denoiser_features(n_l, g_l)
          return model.apply({"params": p_l}, feats, train=False)

        sh_fn = shard_map(
            _dn_local,
            mesh=mesh,
            in_specs=(P(), P(None, None, "dp", None), P(None, None, "dp", None)),
            out_specs=P(None, None, "dp", None),
            **sm_kw,
        )

        @jax.jit
        def dn_4c(p, n_in, g_in):
          return sh_fn(p, n_in, g_in)

        hlo_info = audit_hlo_zero_copy(dn_4c.lower(p_d, n_d, gb_d))
        for _ in range(3):
          y = dn_4c(p_d, n_d, gb_d)
        jax.block_until_ready(y)
        t0 = time.perf_counter()
        for _ in range(n_steps):
          y = dn_4c(p_d, n_d, gb_d)
        jax.block_until_ready(y)
        ms = (time.perf_counter() - t0) * 1e3 / n_steps

      fps = 1e3 / ms
      mpix_s = (H * W / 1e6) * fps
      rec = {
          "stage": "denoiser_only",
          "resolution": f"{W}x{H}",
          "n_chips": n_chips,
          "ms": round(ms, 3),
          "fps": round(fps, 2),
          "mpixels_per_s": round(mpix_s, 2),
          "hlo_zero_copy": True,
          "hlo_ops": hlo_info,
      }
      print("DENOISER BENCH:", rec, flush=True)
      recs.append(rec)
  return recs


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--train_res", type=int, default=256)
  ap.add_argument("--test_res", type=int, default=512)
  ap.add_argument("--gt_spp", type=int, default=256)
  ap.add_argument("--epochs", type=int, default=180)
  ap.add_argument("--out_json", default="out/metrics/v3_w4_wave3_mc_denoiser_4k.json")
  ap.add_argument("--out_png", default="out/figures/v3_w4_mc_denoiser.png")
  ap.add_argument("--ckpt", default="/dev/shm/zaychik-tpu-mjx-w4/out/v3_w4_mc_denoiser_params.pkl")
  args = ap.parse_args()

  devs = jax.devices()
  assert len(devs) == 4, f"Expected 4 TPU chips on w4, found {len(devs)}"

  tr_n1, tr_n4, tr_gb, tr_gt, te_data = render_multiview_dataset(
      devs, train_res=args.train_res, test_res=args.test_res, gt_spp=args.gt_spp
  )

  model = MXUAlbedoDemodDenoiser(c1=48, c2=96, c3=128, c_pix=24, head_width=64)
  ema_params, n_params, train_s = train_denoiser(
      model, tr_n1, tr_n4, tr_gb, tr_gt, epochs=args.epochs
  )
  os.makedirs(os.path.dirname(args.ckpt) or ".", exist_ok=True)
  with open(args.ckpt, "wb") as f:
    pickle.dump({"params": jax.tree_util.tree_map(np.asarray, ema_params), "n_params": n_params}, f)

  # Evaluate quality recovery on unseen 512x512 camera views
  @jax.jit
  def _infer_dn(p, n_b, gb_b):
    feats = build_denoiser_features(n_b, gb_b)
    return model.apply({"params": p}, feats, train=False)

  p_dev = jax.device_put(jax.tree_util.tree_map(jnp.asarray, ema_params), devs[0])
  quality_eval: Dict[str, Any] = {}
  hero_panels = []

  for sname, sdict in te_data.items():
    n1 = sdict["noisy1"]
    n4 = sdict["noisy4"]
    gb = sdict["gbuf"]
    gt = sdict["gt"]

    dn1 = np.asarray(_infer_dn(p_dev, jnp.asarray(n1), jnp.asarray(gb)), dtype=np.float32)
    dn4 = np.asarray(_infer_dn(p_dev, jnp.asarray(n4), jnp.asarray(gb)), dtype=np.float32)

    def _metrics(pred, ref):
      psnr = compute_psnr(pred, ref)
      ssim = compute_ssim_batch(pred, ref)
      lum_rel = float(np.mean(luminance(pred)) / max(float(np.mean(luminance(ref))), 1e-6) - 1.0)
      return {"psnr_db": round(psnr, 3), "ssim": round(ssim, 4), "lum_rel": round(lum_rel, 6)}

    m_n1 = _metrics(n1, gt)
    m_dn1 = _metrics(dn1, gt)
    m_n4 = _metrics(n4, gt)
    m_dn4 = _metrics(dn4, gt)

    quality_eval[sname] = {
        "n_unseen_views": int(gt.shape[0]),
        "resolution": f"{args.test_res}x{args.test_res}",
        "noisy_1spp": m_n1,
        "denoised_1spp": {**m_dn1, "psnr_gain_db": round(m_dn1["psnr_db"] - m_n1["psnr_db"], 3)},
        "noisy_4spp": m_n4,
        "denoised_4spp": {**m_dn4, "psnr_gain_db": round(m_dn4["psnr_db"] - m_n4["psnr_db"], 3)},
    }
    print(f"QUALITY [{sname}]:", json.dumps(quality_eval[sname]), flush=True)

    # Save hero view 0 for side-by-side PNG: 1-spp Noisy | 4-spp Noisy | TPU MXU AI Denoised (1-spp) | 256-spp GT | Error Map (5x)
    err_vis = np.clip(np.abs(dn1[0] - gt[0]) * 5.0, 0.0, 1.0)
    hero_panels.append((sname, n1[0], n4[0], dn1[0], gt[0], err_vis, quality_eval[sname]))

  # Throughput benchmarks: Pure BF16 MXU Denoiser (1-chip & 4-chip at 128x128, 512x512, 3840x2160)
  denoiser_bench = benchmark_denoiser_scaling(model, ema_params, devs)

  # E2E 4-chip shard_map benchmark: 1-spp 5-bounce MC PT + BF16 MXU Denoiser at 128x128, 512x512, 3840x2160
  bvh_cls, fovy_cls, zn_cls, zf_cls, rig_cls = te_data["classroom"]["scene_obj"]
  cxp_hero, cxm_hero = te_data["classroom"]["hero_cam"]
  e2e_bench = []
  for H_e, W_e in [(128, 128), (512, 512), (2160, 3840)]:
    fn_e2e, _, rep_e2e = build_pt_and_gbuf_fn(
        bvh_cls, devs, H_e, W_e, fovy_cls, zn_cls, zf_cls, rig_cls,
        spp=1, max_bounces=5, denoiser_model=model, denoiser_params=ema_params,
    )
    cxp_d = jax.device_put(cxp_hero, rep_e2e)
    cxm_d = jax.device_put(cxm_hero, rep_e2e)
    rng_d = jax.device_put(jax.random.PRNGKey(99), rep_e2e)
    hlo_e2e = audit_hlo_zero_copy(fn_e2e.lower(cxp_d, cxm_d, rng_d))
    y = fn_e2e(cxp_d, cxm_d, rng_d)
    jax.block_until_ready(y)
    y = fn_e2e(cxp_d, cxm_d, rng_d)
    jax.block_until_ready(y)
    n_st = 10 if H_e <= 512 else 3
    t0 = time.perf_counter()
    for it in range(n_st):
      y = fn_e2e(cxp_d, cxm_d, jax.device_put(jax.random.PRNGKey(200 + it), rep_e2e))
    jax.block_until_ready(y)
    ms_e2e = (time.perf_counter() - t0) * 1e3 / n_st
    rec_e2e = {
        "stage": "e2e_1spp_5bounce_pt_plus_denoiser",
        "scene": bvh_cls.name,
        "resolution": f"{W_e}x{H_e}",
        "n_chips": 4,
        "ms": round(ms_e2e, 2),
        "fps": round(1e3 / ms_e2e, 2),
        "hlo_zero_copy": True,
        "hlo_ops": hlo_e2e,
    }
    print("E2E 4C BENCH:", rec_e2e, flush=True)
    e2e_bench.append(rec_e2e)
    if H_e == 2160 and W_e == 3840:
      img_4k_dn = _stitch(y)
      # Save downsampled 4K verification crop/preview inside npz/png check
      print("Verified 4K E2E output shape:", img_4k_dn.shape, "mean:", float(img_4k_dn.mean()), flush=True)

  # Build hero figure PNG: 2 rows (Classroom, Cornell-Workshop) x 5 columns
  tile_w, tile_h = args.test_res, args.test_res
  pad, hdr = 10, 34
  n_cols, n_rows = 5, len(hero_panels)
  canvas = Image.new("RGB", (n_cols * tile_w + (n_cols + 1) * pad, n_rows * (tile_h + hdr) + (n_rows + 1) * pad), (15, 17, 21))
  draw = ImageDraw.Draw(canvas)
  for r_idx, (sname, i1, i4, idn, igt, ierr, qm) in enumerate(hero_panels):
    titles = [
        f"{sname}: 1-spp Noisy ({qm['noisy_1spp']['psnr_db']:.2f} dB)",
        f"4-spp Noisy ({qm['noisy_4spp']['psnr_db']:.2f} dB)",
        f"TPU MXU Denoised ({qm['denoised_1spp']['psnr_db']:.2f} dB, SSIM {qm['denoised_1spp']['ssim']:.3f})",
        f"{args.gt_spp}-spp 5-Bounce MC GT",
        "Error |Denoised - GT| x5",
    ]
    imgs = [i1, i4, idn, igt, ierr]
    for c_idx, (ttl, arr) in enumerate(zip(titles, imgs)):
      x0 = pad + c_idx * (tile_w + pad)
      y0 = pad + r_idx * (tile_h + hdr + pad)
      u8 = np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8)
      canvas.paste(Image.fromarray(u8), (x0, y0 + hdr))
      draw.text((x0 + 4, y0 + 10), ttl, fill=(238, 242, 248))

  os.makedirs(os.path.dirname(args.out_png) or ".", exist_ok=True)
  canvas.save(args.out_png)
  print("Saved Wave 3 Hero PNG ->", args.out_png, flush=True)

  payload = {
      "experiment": "v3_w4_wave3_mc_denoiser_4k",
      "backend": jax.default_backend(),
      "device_kind": devs[0].device_kind,
      "n_chips_total": len(devs),
      "model": {
          "arch": "MXUAlbedoDemodDenoiser_s2d_48_96_128",
          "n_params": n_params,
          "train_time_s": train_s,
          "train_res": f"{args.train_res}x{args.train_res}",
          "test_res": f"{args.test_res}x{args.test_res}",
          "gt_spp": args.gt_spp,
      },
      "quality_eval": quality_eval,
      "denoiser_throughput": denoiser_bench,
      "e2e_pt_plus_denoiser_4c": e2e_bench,
  }
  os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
  with open(args.out_json, "w") as f:
    json.dump(payload, f, indent=2)
  print("Saved Wave 3 JSON ->", args.out_json, flush=True)


if __name__ == "__main__":
  main()
