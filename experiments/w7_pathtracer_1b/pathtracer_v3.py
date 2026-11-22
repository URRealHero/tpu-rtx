"""Wave 2 TPU-Native Multi-Bounce Monte Carlo Path Tracer (Wide-BVH-64 + TLAS/BLAS).

Supports:
- Wave 2A: Multi-bounce (1-4 bounces) Monte Carlo path tracing with:
  * Cosine-weighted hemisphere diffuse bounces
  * GGX microfacet glossy/metallic importance sampling + Smith GGX visibility
  * Next-Event Estimation (NEE) toward a soft disk area light + any_hit soft shadows
  * Analytic studio sky-dome environment lighting + ACES filmic tone mapping
  * On-device Threefry PRNG, zero host callbacks (host_callbacks == 0, send_recv_ops == 0)
  * Track 2A (unique multi-mesh BVH: 1.18M / 10.64M / 108.23M tris) & Track 2B (TLAS+BLAS instanced: 106.4M / 1.02B tris)
  * 1-chip and 4-chip spatially-sharded shard_map across 256x256, 512x512, 4K (3840x2160), 8K (7680x4320).
- Wave 2B: 1,021,870,080 (1.02B) instanced triangles at Native 8K UHD (7680x4320, 259,200 8x16 tiles).
"""

from __future__ import annotations

import dataclasses
import inspect
import math
import re
from typing import Any, Dict, Optional, Sequence, Tuple

import jax
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np

from tpu_mjx_render.primitives import MJ_MAXVAL
from tpu_mjx_render.roofline import inspect_hlo_ops
from wide_bvh_v3 import (
    WideBVHSceneSpec,
    _pallas_blas_instanced_raycast,
    _pallas_wide_bvh_raycast,
)


def audit_hlo_zero_copy(lowered: Any) -> Dict[str, Any]:
  txt = lowered.as_text()
  info = inspect_hlo_ops(txt)
  info["send_recv_ops"] = len(re.findall(r"\bsend\(|\brecv\(", txt, re.I))
  assert info["host_callbacks"] == 0, f"HLO contains host callbacks: {info}"
  assert info["send_recv_ops"] == 0, f"HLO contains send/recv ops: {info}"
  return info


def _build_onb(n: jax.Array) -> Tuple[jax.Array, jax.Array]:
  """Frisvad orthonormal basis (tangent, bitangent) from unit normal n (..., 3)."""
  nz = n[..., 2:3]
  sign = jnp.where(nz >= 0.0, 1.0, -1.0)
  a = -1.0 / (sign + nz)
  b = n[..., 0:1] * n[..., 1:2] * a
  t1 = jnp.concatenate(
      [1.0 + sign * (n[..., 0:1] ** 2) * a, sign * b, -sign * n[..., 0:1]], axis=-1
  )
  t2 = jnp.concatenate([b, sign + (n[..., 1:2] ** 2) * a, -n[..., 1:2]], axis=-1)
  # Fallback near -Z singularity
  near_neg_z = nz < -0.999999
  t1_fb = jnp.broadcast_to(jnp.array([0.0, -1.0, 0.0], dtype=jnp.float32), t1.shape)
  t2_fb = jnp.broadcast_to(jnp.array([-1.0, 0.0, 0.0], dtype=jnp.float32), t2.shape)
  t1 = jnp.where(near_neg_z, t1_fb, t1)
  t2 = jnp.where(near_neg_z, t2_fb, t2)
  t1 = t1 / jnp.maximum(jnp.linalg.norm(t1, axis=-1, keepdims=True), 1e-12)
  t2 = jnp.cross(n, t1)
  return t1, t2


def _sample_cosine_hemisphere(u1: jax.Array, u2: jax.Array, n: jax.Array) -> jax.Array:
  r = jnp.sqrt(jnp.clip(u1, 0.0, 1.0))
  phi = 2.0 * math.pi * u2
  lx = r * jnp.cos(phi)
  ly = r * jnp.sin(phi)
  lz = jnp.sqrt(jnp.maximum(1.0 - u1, 0.0))
  t1, t2 = _build_onb(n)
  d = t1 * lx[..., None] + t2 * ly[..., None] + n * lz[..., None]
  return d / jnp.maximum(jnp.linalg.norm(d, axis=-1, keepdims=True), 1e-12)


def _sample_ggx_half(u1: jax.Array, u2: jax.Array, n: jax.Array, alpha: jax.Array) -> jax.Array:
  a2 = alpha * alpha
  u1_safe = jnp.clip(u1, 1e-5, 1.0 - 1e-5)
  cos_t2 = (1.0 - u1_safe) / (1.0 + (a2 - 1.0) * u1_safe)
  cos_t = jnp.sqrt(jnp.clip(cos_t2, 0.0, 1.0))
  sin_t = jnp.sqrt(jnp.maximum(1.0 - cos_t2, 0.0))
  phi = 2.0 * math.pi * u2
  lx = sin_t * jnp.cos(phi)
  ly = sin_t * jnp.sin(phi)
  t1, t2 = _build_onb(n)
  h = t1 * lx[..., None] + t2 * ly[..., None] + n * cos_t[..., None]
  return h / jnp.maximum(jnp.linalg.norm(h, axis=-1, keepdims=True), 1e-12)


def _smith_g1(ndotv: jax.Array, alpha: jax.Array) -> jax.Array:
  a2 = alpha * alpha
  ndv = jnp.maximum(ndotv, 1e-4)
  return (2.0 * ndv) / (ndv + jnp.sqrt(a2 + (1.0 - a2) * ndv * ndv))


def _eval_pbr_brdf(
    n: jax.Array,
    wo: jax.Array,
    wi: jax.Array,
    albedo: jax.Array,
    metallic: jax.Array,
    roughness: jax.Array,
) -> jax.Array:
  ndotl = jnp.maximum(jnp.sum(n * wi, axis=-1), 0.0)
  ndotv = jnp.maximum(jnp.sum(n * wo, axis=-1), 1e-4)
  h = wo + wi
  h_norm = jnp.linalg.norm(h, axis=-1, keepdims=True)
  h = h / jnp.maximum(h_norm, 1e-12)
  ndoth = jnp.maximum(jnp.sum(n * h, axis=-1), 0.0)
  vdoth = jnp.maximum(jnp.sum(wo * h, axis=-1), 0.0)

  alpha = jnp.maximum(roughness * roughness, 0.02)
  a2 = alpha * alpha
  denom_d = ndoth * ndoth * (a2 - 1.0) + 1.0
  D = a2 / (math.pi * denom_d * denom_d + 1e-6)
  G = _smith_g1(ndotl, alpha) * _smith_g1(ndotv, alpha)
  f0 = 0.04 * (1.0 - metallic[..., None]) + albedo * metallic[..., None]
  F = f0 + (1.0 - f0) * ((1.0 - vdoth) ** 5)[..., None]

  spec = (D * G / jnp.maximum(4.0 * ndotl * ndotv, 1e-4))[..., None] * F
  kd = (1.0 - metallic[..., None]) * (1.0 - F)
  diff = kd * albedo / math.pi
  return diff + spec


def _sky_dome(rd: jax.Array) -> jax.Array:
  """Studio hemisphere sky + subtle horizon + warm ground bounce."""
  z = rd[..., 2]
  t_up = jnp.clip(z, 0.0, 1.0)[..., None]
  t_dn = jnp.clip(-z, 0.0, 1.0)[..., None]
  zenith = jnp.array([0.24, 0.39, 0.64], dtype=jnp.float32)
  horizon = jnp.array([0.19, 0.21, 0.25], dtype=jnp.float32)
  ground = jnp.array([0.08, 0.075, 0.07], dtype=jnp.float32)
  up_col = (1.0 - t_up) * horizon + t_up * zenith
  sky = jnp.where(z[..., None] >= 0.0, up_col, (1.0 - t_dn) * horizon + t_dn * ground)
  rim_dir = jnp.array([-0.55, 0.65, 0.52], dtype=jnp.float32)
  rim = (jnp.maximum(jnp.sum(rd * rim_dir, axis=-1), 0.0) ** 8.0)[..., None] * jnp.array(
      [0.24, 0.28, 0.38], dtype=jnp.float32
  )
  return sky + rim


def _aces_tonemap_srgb(rgb_lin: jax.Array) -> jax.Array:
  """ACES Filmic tone map + sRGB gamma."""
  x = jnp.maximum(rgb_lin, 0.0)
  a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
  mapped = jnp.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)
  return jnp.power(mapped, 1.0 / 2.2)


def _compute_strip_camera_rays(
    cam_fovy: float,
    znear: float,
    full_H: int,
    full_W: int,
    y_offset: int,
    strip_H: int,
    x_offset: int,
    strip_W: int,
    cam_xpos: jax.Array,   # (B, 3)
    cam_xmat: jax.Array,   # (B, 3, 3)
    jitter_xy: Optional[jax.Array] = None,  # (B, strip_H * strip_W, 2) in [-0.5, 0.5]
) -> Tuple[jax.Array, jax.Array, jax.Array]:
  B = cam_xpos.shape[0]
  aspect = float(full_W) / float(full_H)
  fovy_rad = float(cam_fovy) * (math.pi / 180.0)
  half_h = float(znear) * math.tan(0.5 * fovy_rad)
  half_w = half_h * aspect

  px = jnp.arange(strip_W, dtype=jnp.float32) + jnp.asarray(x_offset, dtype=jnp.float32) + 0.5
  py = jnp.arange(strip_H, dtype=jnp.float32) + jnp.asarray(y_offset, dtype=jnp.float32) + 0.5
  gx, gy = jnp.meshgrid(px, py, indexing="xy")  # (strip_H, strip_W)
  gx = gx.reshape(1, strip_H * strip_W)
  gy = gy.reshape(1, strip_H * strip_W)
  if jitter_xy is not None:
    u = (gx + jitter_xy[..., 0]) / float(full_W)
    v = (gy + jitter_xy[..., 1]) / float(full_H)
  else:
    u = jnp.broadcast_to(gx / float(full_W), (B, strip_H * strip_W))
    v = jnp.broadcast_to(gy / float(full_H), (B, strip_H * strip_W))

  x = -half_w + (2.0 * half_w) * u
  y = half_h - (2.0 * half_h) * v
  raw = jnp.stack([x, y, jnp.full_like(x, -float(znear))], axis=-1)  # (B, R, 3)
  ld = raw / jnp.maximum(jnp.linalg.norm(raw, axis=-1, keepdims=True), 1e-12)
  neg_z = jnp.maximum(-ld[..., 2], 1e-6)
  rd = jnp.einsum("bij,brj->bri", cam_xmat, ld, precision=jax.lax.Precision.HIGHEST)
  rd = rd / jnp.maximum(jnp.linalg.norm(rd, axis=-1, keepdims=True), 1e-12)
  inv_cos = (float(znear) / neg_z)[..., None]
  ro = cam_xpos[:, None, :] + inv_cos * rd
  return ro, rd, neg_z


def _raycast_unique_world(
    spec: WideBVHSceneSpec,
    dev_arrays: Tuple[Tuple[jax.Array, ...], ...],
    obj_xpos: jax.Array,
    obj_xmat: jax.Array,
    ro: jax.Array,
    rd: jax.Array,
    t_limit: jax.Array,
    H: int,
    W: int,
    any_hit: bool = False,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
  B, R, _ = ro.shape
  best_d = t_limit
  best_gid = jnp.full((B, R), -1, dtype=jnp.int32)
  best_nrm = jnp.zeros((B, R, 3), dtype=jnp.float32)

  for m_idx, item in enumerate(spec.meshes):
    l2_roots_dev, l2_dev, l0_dev, leaf_dev = dev_arrays[m_idx]
    p = obj_xpos[:, m_idx, :]
    rmat = obj_xmat[:, m_idx, :, :]
    ro_loc = jnp.einsum("bji,brj->bri", rmat, ro - p[:, None, :], precision=jax.lax.Precision.HIGHEST)
    rd_loc = jnp.einsum("bji,brj->bri", rmat, rd, precision=jax.lax.Precision.HIGHEST)
    d_m, n_loc_m, _ = _pallas_wide_bvh_raycast(
        item,
        l2_roots_dev,
        l2_dev,
        l0_dev,
        leaf_dev,
        ro_loc,
        rd_loc,
        best_d,
        H,
        W,
        any_hit=any_hit,
        interpret=False,
        return_stats=False,
    )
    imp = d_m < best_d
    if any_hit:
      best_d = jnp.where(imp, d_m, best_d)
      best_gid = jnp.where(imp, jnp.int32(m_idx), best_gid)
    else:
      n_w_m = jnp.einsum("bij,brj->bri", rmat, n_loc_m, precision=jax.lax.Precision.HIGHEST)
      best_d = jnp.where(imp, d_m, best_d)
      best_gid = jnp.where(imp, jnp.int32(m_idx), best_gid)
      best_nrm = jnp.where(imp[..., None], n_w_m, best_nrm)

  return best_d, best_gid, best_nrm


def _raycast_instanced_world(
    spec: WideBVHSceneSpec,
    dev_arrays: Tuple[Tuple[jax.Array, ...], ...],
    inst_xpos_per_blas: Tuple[jax.Array, ...],
    inst_xmat_per_blas: Tuple[jax.Array, ...],
    ro: jax.Array,
    rd: jax.Array,
    t_limit: jax.Array,
    H: int,
    W: int,
    max_active_per_tile: int = 16,
    any_hit: bool = False,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  B, R, _ = ro.shape
  best_d = t_limit
  best_gid = jnp.full((B, R), -1, dtype=jnp.int32)
  best_inst = jnp.full((B, R), -1, dtype=jnp.int32)
  best_nrm = jnp.zeros((B, R, 3), dtype=jnp.float32)

  for m_idx, item in enumerate(spec.meshes):
    l2_roots_dev, l2_dev, l0_dev, leaf_dev = dev_arrays[m_idx]
    d_m, n_w_m, inst_f = _pallas_blas_instanced_raycast(
        item,
        l2_roots_dev,
        l2_dev,
        l0_dev,
        leaf_dev,
        ro,
        rd,
        best_d,
        inst_xpos_per_blas[m_idx],
        inst_xmat_per_blas[m_idx],
        H,
        W,
        max_active_per_tile=max_active_per_tile,
        any_hit=any_hit,
        interpret=False,
    )
    imp = d_m < best_d
    best_d = jnp.where(imp, d_m, best_d)
    best_gid = jnp.where(imp, jnp.int32(m_idx), best_gid)
    if not any_hit:
      best_inst = jnp.where(imp, jnp.rint(inst_f).astype(jnp.int32), best_inst)
      best_nrm = jnp.where(imp[..., None], n_w_m, best_nrm)

  return best_d, best_gid, best_inst, best_nrm


def _trace_strip_pathtracer(
    spec: WideBVHSceneSpec,
    dev_arrays: Tuple[Tuple[jax.Array, ...], ...],
    poses_a: Any,
    poses_b: Any,
    cam_xpos: jax.Array,
    cam_xmat: jax.Array,
    full_H: int,
    full_W: int,
    y_offset: jax.Array,
    strip_H: int,
    x_offset: jax.Array,
    strip_W: int,
    spp: int,
    max_bounces: int,
    base_rng: jax.Array,
    instanced: bool = False,
    max_active_per_tile: int = 16,
    mat_albedo_tbl: Optional[jax.Array] = None,
    mat_metal_tbl: Optional[jax.Array] = None,
    mat_rough_tbl: Optional[jax.Array] = None,
) -> jax.Array:
  """Pure-XLA/Mosaic multi-bounce MC path tracer on a (B, strip_H, strip_W) tile strip."""
  B = cam_xpos.shape[0]
  R = strip_H * strip_W

  # Default material table per mesh: Suzanne (coral satin), Torus (metallic cyan), Pedestal (warm matte)
  if mat_albedo_tbl is None:
    albs = [it.rgba[:3] for it in spec.meshes]
    mat_albedo_tbl = jnp.asarray(albs, dtype=jnp.float32)
  if mat_metal_tbl is None:
    mets = [0.05 if i == 0 else (0.88 if i == 1 else 0.0) for i in range(len(spec.meshes))]
    mat_metal_tbl = jnp.asarray(mets, dtype=jnp.float32)
  if mat_rough_tbl is None:
    rghs = [0.38 if i == 0 else (0.14 if i == 1 else 0.58) for i in range(len(spec.meshes))]
    mat_rough_tbl = jnp.asarray(rghs, dtype=jnp.float32)

  # Warm studio soft disk area light overhead
  light_c = jnp.asarray(spec.light_pos, dtype=jnp.float32)
  light_dir_target = jnp.array([0.0, 0.0, 0.15], dtype=jnp.float32) - light_c
  light_n = light_dir_target / jnp.maximum(jnp.linalg.norm(light_dir_target), 1e-12)
  lt1, lt2 = _build_onb(light_n)
  light_radius = 0.55
  light_area = math.pi * (light_radius ** 2)
  light_emit = jnp.array([9.5, 8.9, 8.1], dtype=jnp.float32)

  def _sample_one_spp(s_idx, rad_acc):
    s_rng = jax.random.fold_in(base_rng, s_idx)
    k_jit, k_bnc = jax.random.split(s_rng)
    # Independent per-batch-item subpixel jitter & bounce PRNG if B > 1
    jitter = jax.random.uniform(k_jit, (B, R, 2), minval=-0.5, maxval=0.5, dtype=jnp.float32)
    ro0, rd0, neg_z0 = _compute_strip_camera_rays(
        spec.cam_fovy,
        spec.znear,
        full_H,
        full_W,
        y_offset,
        strip_H,
        x_offset,
        strip_W,
        cam_xpos,
        cam_xmat,
        jitter_xy=jitter,
    )
    t_max0 = jnp.broadcast_to(((spec.zfar - spec.znear) / neg_z0), (B, R))
    thr0 = jnp.ones((B, R, 3), dtype=jnp.float32)
    alive0 = jnp.ones((B, R), dtype=jnp.bool_)
    rad0 = jnp.zeros((B, R, 3), dtype=jnp.float32)

    def _bounce_body(b_idx, carry):
      ro, rd, t_lim, thr, alive, rad = carry
      b_rng = jax.random.fold_in(k_bnc, b_idx)
      k_lu, k_lv, k_lobe, k_u1, k_u2 = jax.random.split(b_rng, 5)

      if instanced:
        d_hit, gid, inst_id, nrm = _raycast_instanced_world(
            spec,
            dev_arrays,
            poses_a,
            poses_b,
            ro,
            rd,
            jnp.where(alive, t_lim, 0.0),
            strip_H,
            strip_W,
            max_active_per_tile=max_active_per_tile,
            any_hit=False,
        )
      else:
        d_hit, gid, nrm = _raycast_unique_world(
            spec,
            dev_arrays,
            poses_a,
            poses_b,
            ro,
            rd,
            jnp.where(alive, t_lim, 0.0),
            strip_H,
            strip_W,
            any_hit=False,
        )
        inst_id = jnp.zeros_like(gid)

      hit = alive & (gid >= 0)
      miss = alive & (~hit)
      rad = rad + jnp.where(miss[..., None], thr * _sky_dome(rd), 0.0)

      wo = -rd
      nrm = jnp.where(hit[..., None], nrm, jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32))
      nrm = nrm / jnp.maximum(jnp.linalg.norm(nrm, axis=-1, keepdims=True), 1e-12)
      flip = jnp.sum(nrm * wo, axis=-1, keepdims=True) < 0.0
      nrm = jnp.where(flip, -nrm, nrm)

      hit_pos = ro + jnp.where(hit, d_hit, 0.0)[..., None] * rd

      safe_gid = jnp.maximum(gid, 0)
      if instanced and mat_albedo_tbl.ndim == 2 and mat_albedo_tbl.shape[0] > len(spec.meshes):
        # Per-(mesh, inst) palette lookup
        n_pal = mat_albedo_tbl.shape[0]
        p_idx = (safe_gid * 7 + jnp.maximum(inst_id, 0)) % n_pal
        alb = mat_albedo_tbl[p_idx]
        met = mat_metal_tbl[p_idx]
        rgh = mat_rough_tbl[p_idx]
      else:
        alb = mat_albedo_tbl[safe_gid]
        met = mat_metal_tbl[safe_gid]
        rgh = mat_rough_tbl[safe_gid]

      # Next-Event Estimation (NEE) toward soft area disk light
      lu = jax.random.uniform(k_lu, (B, R), minval=0.0, maxval=1.0, dtype=jnp.float32)
      lv = jax.random.uniform(k_lv, (B, R), minval=0.0, maxval=1.0, dtype=jnp.float32)
      lr = light_radius * jnp.sqrt(lu)
      lphi = (2.0 * math.pi) * lv
      lp = light_c[None, None, :] + (lr * jnp.cos(lphi))[..., None] * lt1[None, None, :] + (lr * jnp.sin(lphi))[..., None] * lt2[None, None, :]

      to_l = lp - hit_pos
      dist_l = jnp.linalg.norm(to_l, axis=-1)
      wi_l = to_l / jnp.maximum(dist_l[..., None], 1e-12)
      cos_s = jnp.maximum(jnp.sum(nrm * wi_l, axis=-1), 0.0)
      cos_l = jnp.maximum(jnp.sum(light_n[None, None, :] * (-wi_l), axis=-1), 0.0)
      need_sh = hit & (cos_s > 0.0) & (cos_l > 0.0)
      sh_o = hit_pos + nrm * 1.5e-3
      sh_lim = jnp.where(need_sh, jnp.maximum(dist_l - 1.5e-3, 0.0), 0.0)

      if instanced:
        d_sh, sh_gid, _, _ = _raycast_instanced_world(
            spec,
            dev_arrays,
            poses_a,
            poses_b,
            sh_o,
            wi_l,
            sh_lim,
            strip_H,
            strip_W,
            max_active_per_tile=max_active_per_tile,
            any_hit=True,
        )
      else:
        d_sh, sh_gid, _ = _raycast_unique_world(
            spec,
            dev_arrays,
            poses_a,
            poses_b,
            sh_o,
            wi_l,
            sh_lim,
            strip_H,
            strip_W,
            any_hit=True,
        )

      vis = jnp.where(need_sh & (d_sh >= sh_lim - 1e-4) & (sh_gid < 0), 1.0, 0.0)
      brdf_nee = _eval_pbr_brdf(nrm, wo, wi_l, alb, met, rgh)
      geom_term = (cos_s * cos_l) / (dist_l * dist_l + 0.05) * light_area
      nee_contrib = thr * brdf_nee * light_emit[None, None, :] * (geom_term * vis)[..., None]
      nee_contrib = jnp.minimum(nee_contrib, 12.0)  # Firefly clamp
      rad = rad + jnp.where(hit[..., None], nee_contrib, 0.0)

      # Sample next indirect bounce (Cosine diffuse vs GGX specular lobe)
      u_lobe = jax.random.uniform(k_lobe, (B, R), minval=0.0, maxval=1.0, dtype=jnp.float32)
      u1 = jax.random.uniform(k_u1, (B, R), minval=0.0, maxval=1.0, dtype=jnp.float32)
      u2 = jax.random.uniform(k_u2, (B, R), minval=0.0, maxval=1.0, dtype=jnp.float32)

      p_spec = jnp.clip(0.15 + 0.75 * met, 0.15, 0.88)
      choose_spec = u_lobe < p_spec

      wi_diff = _sample_cosine_hemisphere(u1, u2, nrm)
      alpha = jnp.maximum(rgh * rgh, 0.02)
      h_ggx = _sample_ggx_half(u1, u2, nrm, alpha)
      wi_spec = 2.0 * jnp.sum(wo * h_ggx, axis=-1, keepdims=True) * h_ggx - wo
      wi_spec = wi_spec / jnp.maximum(jnp.linalg.norm(wi_spec, axis=-1, keepdims=True), 1e-12)

      wi_next = jnp.where(choose_spec[..., None], wi_spec, wi_diff)

      # Throughput update for chosen lobe
      ndotl_s = jnp.maximum(jnp.sum(nrm * wi_spec, axis=-1), 0.0)
      ndotv = jnp.maximum(jnp.sum(nrm * wo, axis=-1), 1e-4)
      ndoth = jnp.maximum(jnp.sum(nrm * h_ggx, axis=-1), 1e-4)
      vdoth = jnp.maximum(jnp.sum(wo * h_ggx, axis=-1), 1e-4)
      f0 = 0.04 * (1.0 - met[..., None]) + alb * met[..., None]
      F_s = f0 + (1.0 - f0) * ((1.0 - vdoth) ** 5)[..., None]
      G_s = _smith_g1(ndotl_s, alpha) * _smith_g1(ndotv, alpha)
      spec_weight = F_s * ((G_s * vdoth) / jnp.maximum(ndotv * ndoth * p_spec, 1e-4))[..., None]
      spec_weight = jnp.where(ndotl_s[..., None] > 0.0, jnp.minimum(spec_weight, 4.0), 0.0)

      diff_weight = (1.0 - met[..., None]) * alb / jnp.maximum(1.0 - p_spec[..., None], 0.1)
      bounce_w = jnp.where(choose_spec[..., None], spec_weight, diff_weight)

      new_thr = thr * bounce_w
      new_alive = hit & (jnp.max(new_thr, axis=-1) > 1e-3)
      new_ro = hit_pos + nrm * 1.5e-3
      new_t_lim = jnp.broadcast_to(jnp.float32(spec.zfar), (B, R))
      return (new_ro, wi_next, new_t_lim, new_thr, new_alive, rad)

    _, _, _, _, _, final_rad = jax.lax.fori_loop(
        0, max_bounces, _bounce_body, (ro0, rd0, t_max0, thr0, alive0, rad0)
    )
    return rad_acc + final_rad

  rad_sum = jax.lax.fori_loop(
      0, spp, _sample_one_spp, jnp.zeros((B, R, 3), dtype=jnp.float32)
  )
  rad_avg = rad_sum / float(spp)
  return rad_avg.reshape(B, strip_H, strip_W, 3)


def build_pathtracer_fn(
    spec: WideBVHSceneSpec,
    dev_arrays: Tuple[Tuple[jax.Array, ...], ...],
    devices: Sequence[jax.Device],
    spp: int,
    max_bounces: int,
    instanced: bool = False,
    max_active_per_tile: int = 16,
    strip_height: Optional[int] = None,
    mat_albedo_tbl: Optional[np.ndarray] = None,
    mat_metal_tbl: Optional[np.ndarray] = None,
    mat_rough_tbl: Optional[np.ndarray] = None,
    tonemap: bool = True,
):
  """Compile 1-chip or N-chip spatially-sharded multi-bounce MC path tracer (zero host_callbacks)."""
  H, W = int(spec.height), int(spec.width)
  n_chips = len(devices)
  assert W % n_chips == 0, f"W={W} must divide n_chips={n_chips}"
  W_local = W // n_chips
  assert W_local % 16 == 0 and H % 8 == 0, f"Local tile grid ({H}, {W_local}) must align to 8x16"

  if strip_height is None:
    strip_H = 360 if (instanced and H >= 720 and H % 360 == 0) else H
  else:
    strip_H = int(strip_height)
  assert H % strip_H == 0 and strip_H % 8 == 0, f"strip_H={strip_H} must divide H={H} and align to 8"
  n_strips = H // strip_H

  mesh = Mesh(np.array(devices), ("dp",))
  rep = NamedSharding(mesh, P())
  sharded_dp = NamedSharding(mesh, P("dp"))

  # Replicate material tables
  if mat_albedo_tbl is None:
    mat_albedo_tbl = np.asarray([it.rgba[:3] for it in spec.meshes], dtype=np.float32)
  if mat_metal_tbl is None:
    mat_metal_tbl = np.asarray([0.05 if i == 0 else (0.88 if i == 1 else 0.0) for i in range(len(spec.meshes))], dtype=np.float32)
  if mat_rough_tbl is None:
    mat_rough_tbl = np.asarray([0.38 if i == 0 else (0.14 if i == 1 else 0.58) for i in range(len(spec.meshes))], dtype=np.float32)

  alb_j = jax.device_put(np.asarray(mat_albedo_tbl, dtype=np.float32), rep)
  met_j = jax.device_put(np.asarray(mat_metal_tbl, dtype=np.float32), rep)
  rgh_j = jax.device_put(np.asarray(mat_rough_tbl, dtype=np.float32), rep)

  chip_ids = jax.device_put(np.arange(n_chips, dtype=np.int32), sharded_dp)

  def _render_local(da, pa, pb, cxp, cxm, cid_arr, rng_in, ma, mm, mr):
    cid = cid_arr[0]
    x_off = cid * W_local
    chip_rng = jax.random.fold_in(rng_in, cid)
    B_in = cxp.shape[0]
    # Vectorize multiple SPP samples along batch dim when local tile count is small
    local_px = H * W_local
    if spp >= 16 and B_in == 1 and local_px <= 16384:
      spp_par = 16
    elif spp >= 8 and B_in == 1 and local_px <= 65536:
      spp_par = 8
    elif spp >= 4 and B_in == 1 and local_px <= 131072:
      spp_par = 4
    else:
      spp_par = 1
    spp_seq = spp // spp_par

    if spp_par > 1:
      pa_e = jax.tree_util.tree_map(lambda x: jnp.repeat(x, spp_par, axis=0), pa)
      pb_e = jax.tree_util.tree_map(lambda x: jnp.repeat(x, spp_par, axis=0), pb)
      cxp_e = jnp.repeat(cxp, spp_par, axis=0)
      cxm_e = jnp.repeat(cxm, spp_par, axis=0)
    else:
      pa_e, pb_e, cxp_e, cxm_e = pa, pb, cxp, cxm

    if n_strips == 1:
      rad_e = _trace_strip_pathtracer(
          spec,
          da,
          pa_e,
          pb_e,
          cxp_e,
          cxm_e,
          H,
          W,
          jnp.int32(0),
          strip_H,
          x_off,
          W_local,
          spp_seq,
          max_bounces,
          chip_rng,
          instanced=instanced,
          max_active_per_tile=max_active_per_tile,
          mat_albedo_tbl=ma,
          mat_metal_tbl=mm,
          mat_rough_tbl=mr,
      )
    else:
      def _scan_strip(carry, s_idx):
        del carry
        y_off = s_idx * strip_H
        s_rng = jax.random.fold_in(chip_rng, s_idx)
        strip_rad = _trace_strip_pathtracer(
            spec,
            da,
            pa_e,
            pb_e,
            cxp_e,
            cxm_e,
            H,
            W,
            y_off,
            strip_H,
            x_off,
            W_local,
            spp_seq,
            max_bounces,
            s_rng,
            instanced=instanced,
            max_active_per_tile=max_active_per_tile,
            mat_albedo_tbl=ma,
            mat_metal_tbl=mm,
            mat_rough_tbl=mr,
        )
        return None, strip_rad

      _, strips = jax.lax.scan(_scan_strip, None, jnp.arange(n_strips, dtype=jnp.int32))
      rad_e = jnp.transpose(strips, (1, 0, 2, 3, 4)).reshape(cxp_e.shape[0], H, W_local, 3)

    if spp_par > 1:
      rad = jnp.mean(rad_e.reshape(B_in, spp_par, H, W_local, 3), axis=1)
    else:
      rad = rad_e

    rgb = _aces_tonemap_srgb(rad) if tonemap else rad
    return rgb[None, ...]

  sm_kw = {"check_rep": False} if "check_rep" in inspect.signature(shard_map).parameters else {"check_vma": False}
  sharded_fn = shard_map(
      _render_local,
      mesh=mesh,
      in_specs=(P(), P(), P(), P(), P(), P("dp"), P(), P(), P(), P()),
      out_specs=P("dp"),
      **sm_kw,
  )

  @jax.jit
  def render_fn(pa, pb, cxp, cxm, rng):
    return sharded_fn(dev_arrays, pa, pb, cxp, cxm, chip_ids, rng, alb_j, met_j, rgh_j)

  return render_fn, mesh, rep
