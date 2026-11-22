"""TPU-DLSS 3.5: Real-Time Temporal Ray Reconstruction + 2x Super-Resolution on TPU v5p.

Provides a zero-host-callback (`host_callbacks == 0`, `send_recv_ops == 0`) JAX/Flax
adapter that reconstructs crisp full-resolution (`H x W`, e.g. 3840x2160 Native 4K)
radiance from half-resolution (`H/2 x W/2`, e.g. 1920x1080) 1-spp multi-bounce
Monte Carlo path-traced noisy radiance + primary G-buffers + previous high-res
history:
  1. Analytical world-space-to-previous-screen 2D motion vectors (`compute_motion_vectors`).
  2. Single-gather SpaceToDepth(2) RGB10-packed history + G-buffer warping with ICI
     `ppermute` boundary halo exchange (`warp_history_and_gbuf_s2d12` / `warp_history_and_gbuf_4x4`).
  3. Depth/normal disocclusion mask + YCoCg 3x3 neighborhood AABB variance clipping
     (`compute_disocclusion_and_ycocg_clamp`).
  4. BF16 MXU U-Net + DepthToSpace(2) + albedo demodulation (`TPUDLSS35Reconstructor`).
"""

from __future__ import annotations

import inspect
import math
import re
from typing import Any, Dict, Optional, Sequence, Tuple

import flax.linen as nn
import jax
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np

from tpu_mjx_render.roofline import inspect_hlo_ops


# ==============================================================================
# 1. HLO Audit & Color / Exposure Utilities
# ==============================================================================

def audit_hlo_dlss(lowered: Any) -> Dict[str, Any]:
  """Verify lowered XLA HLO has zero host callbacks and zero send/recv ops."""
  txt = lowered.as_text()
  info = inspect_hlo_ops(txt)
  info["send_recv_ops"] = len(re.findall(r"\bsend\(|\brecv\(", txt, re.I))
  assert info["host_callbacks"] == 0, f"HLO contains host callbacks: {info}"
  assert info["send_recv_ops"] == 0, f"HLO contains send/recv ops: {info}"
  return info


def compute_exposure_scale(noisy_rgb: jax.Array, axis_name: Optional[str] = None) -> jax.Array:
  """Global exposure scale so the reconstructor operates in unit-luminance space."""
  lum = (
      0.2126 * noisy_rgb[..., 0:1]
      + 0.7152 * noisy_rgb[..., 1:2]
      + 0.0722 * noisy_rgb[..., 2:3]
  )
  mu_l = jnp.mean(lum, axis=(1, 2, 3), keepdims=True)
  if axis_name is not None:
    mu_l = jax.lax.pmean(mu_l, axis_name=axis_name)
  return jnp.clip(mu_l / 0.42, 0.04, 2.5)


def rgb_to_ycocg(rgb: jax.Array) -> jax.Array:
  """Convert RGB (..., 3) to YCoCg (..., 3)."""
  r, g, b = rgb[..., 0:1], rgb[..., 1:2], rgb[..., 2:3]
  half = jnp.asarray(0.5, rgb.dtype)
  co = r - b
  t = b + half * co
  cg = g - t
  y = t + half * cg
  return jnp.concatenate([y, co, cg], axis=-1)


def ycocg_to_rgb(ycocg: jax.Array) -> jax.Array:
  """Convert YCoCg (..., 3) back to RGB (..., 3)."""
  y, co, cg = ycocg[..., 0:1], ycocg[..., 1:2], ycocg[..., 2:3]
  half = jnp.asarray(0.5, ycocg.dtype)
  t = y - half * cg
  g = cg + t
  b = t - half * co
  r = b + co
  return jnp.concatenate([r, g, b], axis=-1)


# ==============================================================================
# 2. Analytical Camera Reprojection & 2D Motion Vectors
# ==============================================================================

def compute_motion_vectors(
    depth_norm_lr: jax.Array,   # (B, Hl, Wl_loc, 1)
    curr_cxp: jax.Array,        # (1, 3) or (3,)
    curr_cxm: jax.Array,        # (1, 3, 3) or (3, 3)
    prev_cxp: jax.Array,        # (1, 3) or (3,)
    prev_cxm: jax.Array,        # (1, 3, 3) or (3, 3)
    cam_fovy: float,
    znear: float,
    full_width_lr: int,
    x_offset_lr: Any = 0,
    jitter_lr: Optional[jax.Array] = None,
    full_height_lr: Optional[int] = None,
    y_offset_lr: Any = 0,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Compute exact world-space-to-previous-screen 2D motion vectors in LR pixels.

  Returns:
    mv_lr: (B, Hl, Wl_loc, 2) motion vector in LR pixels (prev_canonical - curr_sample).
    prev_uv: (B, Hl, Wl_loc, 2) normalized [0, 1] coordinates in previous frame.
    expected_d_prev_norm: (B, Hl, Wl_loc, 1) expected ray distance/10.0 in previous camera.
    valid_z: (B, Hl, Wl_loc, 1) bool mask (True if point lies in front of previous camera).
  """
  B, Hl, Wl_loc, _ = depth_norm_lr.shape
  cxp_c = jnp.reshape(curr_cxp, (3,)).astype(jnp.float32)
  cxm_c = jnp.reshape(curr_cxm, (3, 3)).astype(jnp.float32)
  cxp_p = jnp.reshape(prev_cxp, (3,)).astype(jnp.float32)
  cxm_p = jnp.reshape(prev_cxm, (3, 3)).astype(jnp.float32)

  Hl_full = int(full_height_lr) if full_height_lr is not None else Hl
  aspect = float(full_width_lr) / float(Hl_full)
  half_h = jnp.asarray(znear, jnp.float32) * jnp.tan(0.5 * jnp.asarray(cam_fovy, jnp.float32) * (math.pi / 180.0))
  half_w = half_h * aspect

  jx = jitter_lr[0] if jitter_lr is not None else jnp.float32(0.0)
  jy = jitter_lr[1] if jitter_lr is not None else jnp.float32(0.0)

  px_col = jnp.arange(Wl_loc, dtype=jnp.float32) + jnp.asarray(x_offset_lr, jnp.float32) + 0.5 + jx
  py_row = jnp.arange(Hl, dtype=jnp.float32) + jnp.asarray(y_offset_lr, jnp.float32) + 0.5 + jy
  gx, gy = jnp.meshgrid(px_col, py_row, indexing="xy")

  u = gx / float(full_width_lr)
  v = gy / float(Hl_full)
  xc = -half_w + (2.0 * half_w) * u
  yc = half_h - (2.0 * half_h) * v
  raw = jnp.stack([xc, yc, jnp.full_like(xc, -float(znear))], axis=-1)
  raw_len = jnp.maximum(jnp.linalg.norm(raw, axis=-1, keepdims=True), 1e-12)
  ld = raw / raw_len
  rd = jnp.einsum("ij,hwj->hwi", cxm_c, ld, precision=jax.lax.Precision.HIGHEST)
  rd = rd / jnp.maximum(jnp.linalg.norm(rd, axis=-1, keepdims=True), 1e-12)
  ro = cxp_c[None, None, :] + raw_len * rd

  d_norm = depth_norm_lr[..., 0]
  is_hit = d_norm >= 0.0
  d_m = jnp.maximum(d_norm, 0.0) * 10.0

  p_hit = ro[None, ...] + d_m[..., None] * rd[None, ...]
  p_sky = cxp_p[None, None, None, :] + 1000.0 * rd[None, ...]
  p_world = jnp.where(is_hit[..., None], p_hit, p_sky)

  rel = p_world - cxp_p[None, None, None, :]
  p_prev = jnp.einsum("ji,bhwj->bhwi", cxm_p, rel, precision=jax.lax.Precision.HIGHEST)
  z_fwd = -p_prev[..., 2]
  valid_z = z_fwd > 1e-4
  z_safe = jnp.maximum(z_fwd, 1e-4)

  u_prev = 0.5 + 0.5 * (p_prev[..., 0] * float(znear)) / (z_safe * half_w)
  v_prev = 0.5 - 0.5 * (p_prev[..., 1] * float(znear)) / (z_safe * half_h)
  prev_uv = jnp.stack([u_prev, v_prev], axis=-1)

  xc_p = -half_w + (2.0 * half_w) * u_prev
  yc_p = half_h - (2.0 * half_h) * v_prev
  raw_p_len = jnp.sqrt(xc_p * xc_p + yc_p * yc_p + float(znear) * float(znear))
  ray_p_len = jnp.linalg.norm(rel, axis=-1)
  expected_d_prev_norm = jnp.where(is_hit, jnp.maximum(ray_p_len - raw_p_len, 0.0) / 10.0, -1.0)[..., None]

  mv_x = u_prev * float(full_width_lr) - gx[None, ...]
  mv_y = v_prev * float(Hl_full) - gy[None, ...]
  mv_lr = jnp.stack([mv_x, mv_y], axis=-1)
  return mv_lr, prev_uv, expected_d_prev_norm, valid_z[..., None]


# ==============================================================================
# 3. Single-Gather History + G-Buffer Warping (s2d12 Ultra-Fast & 4x4 Exact)
# ==============================================================================

def _pack_rgb10(rgb: jax.Array) -> jax.Array:
  c = jnp.clip(jnp.round(rgb.astype(jnp.float32) * 1023.0), 0.0, 1023.0).astype(jnp.uint32)
  return (c[..., 0] << 20) | (c[..., 1] << 10) | c[..., 2]


def _unpack_rgb10(u: jax.Array) -> jax.Array:
  r = ((u >> 20) & 1023).astype(jnp.float32) * (1.0 / 1023.0)
  g = ((u >> 10) & 1023).astype(jnp.float32) * (1.0 / 1023.0)
  b = (u & 1023).astype(jnp.float32) * (1.0 / 1023.0)
  return jnp.stack([r, g, b], axis=-1).astype(jnp.bfloat16)


def _exchange_halo_LR_and_HR(
    hist_hr: jax.Array,      # (B, H, W_loc, 3)
    geom_lr: jax.Array,      # (B, Hl, Wl_loc, 4)
    axis_name: Optional[str],
    halo_lr: int,
) -> Tuple[jax.Array, jax.Array]:
  """Exchange left/right boundary strips across TPU chips via ICI `ppermute`."""
  if axis_name is None:
    return hist_hr, geom_lr

  n_chips = jax.lax.axis_size(axis_name)
  if n_chips == 1:
    return hist_hr, geom_lr
  cid = jax.lax.axis_index(axis_name)

  halo_hr = 2 * halo_lr
  perm_right = [(i, (i + 1) % n_chips) for i in range(n_chips)]
  perm_left = [((i + 1) % n_chips, i) for i in range(n_chips)]

  h_from_left = jax.lax.ppermute(hist_hr[:, :, -halo_hr:, :], axis_name, perm_right)
  h_from_right = jax.lax.ppermute(hist_hr[:, :, :halo_hr, :], axis_name, perm_left)
  h_from_left = jnp.where(cid > 0, h_from_left, hist_hr[:, :, :1, :])
  h_from_right = jnp.where(cid < n_chips - 1, h_from_right, hist_hr[:, :, -1:, :])
  hist_pad = jnp.concatenate([h_from_left, hist_hr, h_from_right], axis=2)

  g_from_left = jax.lax.ppermute(geom_lr[:, :, -halo_lr:, :], axis_name, perm_right)
  g_from_right = jax.lax.ppermute(geom_lr[:, :, :halo_lr, :], axis_name, perm_left)
  g_from_left = jnp.where(cid > 0, g_from_left, geom_lr[:, :, :1, :])
  g_from_right = jnp.where(cid < n_chips - 1, g_from_right, geom_lr[:, :, -1:, :])
  geom_pad = jnp.concatenate([g_from_left, geom_lr, g_from_right], axis=2)
  return hist_pad, geom_pad


def warp_history_and_gbuf_s2d12(
    prev_hist_hr: jax.Array,     # (B, H, W_loc, 3)
    prev_geom_lr: jax.Array,     # (B, Hl, Wl_loc, 4) = [normal(3), depth_norm(1)]
    mv_lr: jax.Array,            # (B, Hl, Wl_loc, 2)
    axis_name: Optional[str] = None,
    halo_lr: int = 32,
    use_rgb10: bool = True,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
  """Ultra-fast 6-uint32 SpaceToDepth(2) history + G-buffer warp (>90 FPS @ 4K).

  Packs 4 high-res subpixels (RGB10 -> 4 uint32) + previous LR G-buffer (4 BF16 -> 2 uint32)
  into a single half-res gather and returns exact sub-block offset `(sub_dx, sub_dy)` so
  the MXU U-Net learns sub-pixel resampling across neighboring 2x2 blocks.
  """
  dt = jnp.bfloat16
  B, H, W_loc, _ = prev_hist_hr.shape
  Hl, Wl_loc = H // 2, W_loc // 2

  hist_pad, geom_pad = _exchange_halo_LR_and_HR(prev_hist_hr, prev_geom_lr, axis_name, halo_lr)
  eff_halo = halo_lr if axis_name is not None else 0
  Wl_p = geom_pad.shape[2]

  rx_f = jnp.arange(Wl_loc, dtype=jnp.float32)[None, None, :] + float(eff_halo) + mv_lr[..., 0]
  ry_f = jnp.arange(Hl, dtype=jnp.float32)[None, :, None] + mv_lr[..., 1]
  bx_f = jnp.round(rx_f)
  by_f = jnp.round(ry_f)
  sub_offset = jnp.stack([(rx_f - bx_f).astype(dt), (ry_f - by_f).astype(dt)], axis=-1)
  bx = jnp.clip(bx_f.astype(jnp.int32), 0, Wl_p - 1)
  by = jnp.clip(by_f.astype(jnp.int32), 0, Hl - 1)

  if use_rgb10:
    h_u32 = (
        _pack_rgb10(hist_pad)
        .reshape(B, Hl, 2, Wl_p, 2)
        .transpose(0, 1, 3, 2, 4)
        .reshape(B, Hl, Wl_p, 4)
    )
    g_u32 = jax.lax.bitcast_convert_type(geom_pad.astype(dt).reshape(B, Hl, Wl_p, 2, 2), jnp.uint32)
    packed_lr = jnp.concatenate([h_u32, g_u32], axis=-1)
    if B == 1:
      gath = packed_lr[0, by[0], bx[0]][None, ...]
    else:
      gath = packed_lr[jnp.arange(B, dtype=jnp.int32)[:, None, None], by, bx]
    hist_warped_s2d = _unpack_rgb10(gath[..., :4]).reshape(B, Hl, Wl_loc, 12)
    prev_geom_warped = jax.lax.bitcast_convert_type(gath[..., 4:6], dt).reshape(B, Hl, Wl_loc, 4).astype(jnp.float32)
  else:
    h_bf = (
        hist_pad.astype(dt)
        .reshape(B, Hl, 2, Wl_p, 2, 3)
        .transpose(0, 1, 3, 2, 4, 5)
        .reshape(B, Hl, Wl_p, 12)
    )
    packed_lr = jnp.concatenate([h_bf, geom_pad.astype(dt)], axis=-1)
    if B == 1:
      gath = packed_lr[0, by[0], bx[0]][None, ...]
    else:
      gath = packed_lr[jnp.arange(B, dtype=jnp.int32)[:, None, None], by, bx]
    hist_warped_s2d = gath[..., :12]
    prev_geom_warped = gath[..., 12:16].astype(jnp.float32)

  return hist_warped_s2d, prev_geom_warped, sub_offset


def warp_history_and_gbuf_4x4(
    prev_hist_hr: jax.Array,     # (B, H, W_loc, 3)
    prev_geom_lr: jax.Array,     # (B, Hl, Wl_loc, 4) = [normal(3), depth_norm(1)]
    mv_lr: jax.Array,            # (B, Hl, Wl_loc, 2)
    axis_name: Optional[str] = None,
    halo_lr: int = 32,
    use_rgb10: bool = True,
) -> Tuple[jax.Array, jax.Array]:
  """Exact 4x4-packed single-gather high-res bilinear warp + LR G-buffer warp."""
  B, H, W_loc, _ = prev_hist_hr.shape
  Hl, Wl_loc = H // 2, W_loc // 2

  has_halo = axis_name is not None
  hist_pad, geom_pad = _exchange_halo_LR_and_HR(prev_hist_hr, prev_geom_lr, axis_name, halo_lr)
  eff_halo_lr = halo_lr if has_halo else 0
  eff_halo_hr = 2 * eff_halo_lr
  W_p = hist_pad.shape[2]
  Wl_p = geom_pad.shape[2]

  xs_hr = jnp.clip(
      jnp.arange(Wl_loc, dtype=jnp.float32)[None, None, :] * 2.0 + float(eff_halo_hr) + 2.0 * mv_lr[..., 0],
      0.0,
      float(W_p - 3),
  )
  ys_hr = jnp.clip(
      jnp.arange(Hl, dtype=jnp.float32)[None, :, None] * 2.0 + 2.0 * mv_lr[..., 1],
      0.0,
      float(H - 3),
  )
  x0_hr = jnp.floor(xs_hr).astype(jnp.int32)
  y0_hr = jnp.floor(ys_hr).astype(jnp.int32)
  fx = (xs_hr - x0_hr.astype(jnp.float32)).astype(jnp.bfloat16)[..., None]
  fy = (ys_hr - y0_hr.astype(jnp.float32)).astype(jnp.bfloat16)[..., None]

  bx = jnp.clip(x0_hr // 2, 0, Wl_p - 1)
  by = jnp.clip(y0_hr // 2, 0, Hl - 1)
  rx = x0_hr & 1
  ry = y0_hr & 1

  if use_rgb10:
    h_u32 = _pack_rgb10(hist_pad)
    h_p2 = jnp.pad(h_u32, ((0, 0), (0, 2), (0, 2)), mode="edge")
    win_h = jnp.stack([h_p2[:, dy : H + dy : 2, dx : W_p + dx : 2] for dy in range(4) for dx in range(4)], axis=-1)
    g_u32 = jax.lax.bitcast_convert_type(geom_pad.astype(jnp.bfloat16).reshape(B, Hl, Wl_p, 2, 2), jnp.uint32)
    packed_lr = jnp.concatenate([win_h, g_u32], axis=-1)
    if B == 1:
      gathered = packed_lr[0, by[0], bx[0]][None, ...]
    else:
      gathered = packed_lr[jnp.arange(B, dtype=jnp.int32)[:, None, None], by, bx]
    w = _unpack_rgb10(gathered[..., :16]).reshape(B, Hl, Wl_loc, 4, 4, 3)
    prev_geom_warped = jax.lax.bitcast_convert_type(gathered[..., 16:18], jnp.bfloat16).reshape(B, Hl, Wl_loc, 4).astype(jnp.float32)
  else:
    h_p2 = jnp.pad(hist_pad.astype(jnp.bfloat16), ((0, 0), (0, 2), (0, 2), (0, 0)), mode="edge")
    win_h = jnp.concatenate([h_p2[:, dy : H + dy : 2, dx : W_p + dx : 2, :] for dy in range(4) for dx in range(4)], axis=-1)
    packed_lr = jnp.concatenate([win_h, geom_pad.astype(jnp.bfloat16)], axis=-1)
    if B == 1:
      gathered = packed_lr[0, by[0], bx[0]][None, ...]
    else:
      gathered = packed_lr[jnp.arange(B, dtype=jnp.int32)[:, None, None], by, bx]
    w = gathered[..., :48].reshape(B, Hl, Wl_loc, 4, 4, 3)
    prev_geom_warped = gathered[..., 48:52].astype(jnp.float32)

  def _interp(oy: int, ox: int) -> jax.Array:
    iy = ry + oy
    ix = rx + ox
    wy0 = (iy == 0).astype(jnp.bfloat16)
    wy1 = (iy == 1).astype(jnp.bfloat16)
    wy2 = (iy == 2).astype(jnp.bfloat16)
    row0 = wy0[..., None, None] * w[:, :, :, 0] + wy1[..., None, None] * w[:, :, :, 1] + wy2[..., None, None] * w[:, :, :, 2]
    row1 = wy0[..., None, None] * w[:, :, :, 1] + wy1[..., None, None] * w[:, :, :, 2] + wy2[..., None, None] * w[:, :, :, 3]
    wx0 = (ix == 0).astype(jnp.bfloat16)
    wx1 = (ix == 1).astype(jnp.bfloat16)
    wx2 = (ix == 2).astype(jnp.bfloat16)
    c00 = wx0[..., None] * row0[:, :, :, 0] + wx1[..., None] * row0[:, :, :, 1] + wx2[..., None] * row0[:, :, :, 2]
    c01 = wx0[..., None] * row0[:, :, :, 1] + wx1[..., None] * row0[:, :, :, 2] + wx2[..., None] * row0[:, :, :, 3]
    c10 = wx0[..., None] * row1[:, :, :, 0] + wx1[..., None] * row1[:, :, :, 1] + wx2[..., None] * row1[:, :, :, 2]
    c11 = wx0[..., None] * row1[:, :, :, 1] + wx1[..., None] * row1[:, :, :, 2] + wx2[..., None] * row1[:, :, :, 3]
    return (jnp.bfloat16(1.0) - fy) * ((jnp.bfloat16(1.0) - fx) * c00 + fx * c01) + fy * ((jnp.bfloat16(1.0) - fx) * c10 + fx * c11)

  hist_warped_s2d = jnp.concatenate([_interp(0, 0), _interp(0, 1), _interp(1, 0), _interp(1, 1)], axis=-1)
  return hist_warped_s2d, prev_geom_warped


# ==============================================================================
# 4. Disocclusion Mask & YCoCg Neighborhood AABB Variance Clipping
# ==============================================================================

def compute_disocclusion_and_ycocg_clamp(
    noisy_norm: jax.Array,           # (B, Hl, Wl_loc, 3) BF16 exposure-normalized
    curr_gbuf_lr: jax.Array,         # (B, Hl, Wl_loc, 9)
    prev_geom_warped: jax.Array,     # (B, Hl, Wl_loc, 4) [normal(3), depth_norm(1)]
    hist_warped_s2d: jax.Array,      # (B, Hl, Wl_loc, 12) BF16
    prev_uv: jax.Array,              # (B, Hl, Wl_loc, 2)
    expected_d_prev_norm: jax.Array, # (B, Hl, Wl_loc, 1)
    valid_z: jax.Array,              # (B, Hl, Wl_loc, 1) bool
    l_scale: jax.Array,              # (B, 1, 1, 1)
    has_prev_history: Optional[jax.Array] = None,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
  """Compute geometric disocclusion mask and YCoCg neighborhood variance-clipped history."""
  dt = jnp.bfloat16
  curr_n = curr_gbuf_lr[..., 3:6]
  curr_d = curr_gbuf_lr[..., 6:7]
  curr_hit = curr_d >= 0.0

  prev_n = prev_geom_warped[..., 0:3]
  prev_d = prev_geom_warped[..., 3:4]
  prev_hit = prev_d >= 0.0

  in_bounds = (
      (prev_uv[..., 0:1] >= 0.0)
      & (prev_uv[..., 0:1] <= 1.0)
      & (prev_uv[..., 1:2] >= 0.0)
      & (prev_uv[..., 1:2] <= 1.0)
      & valid_z
  )
  hit_match = (curr_hit == prev_hit)

  rel_d_err = jnp.abs(expected_d_prev_norm - prev_d) / jnp.maximum(expected_d_prev_norm, 0.05)
  depth_valid = jnp.where(curr_hit, jnp.clip(1.0 - (rel_d_err - 0.04) / 0.10, 0.0, 1.0), 1.0)

  n_dot = jnp.sum(curr_n * prev_n, axis=-1, keepdims=True)
  norm_valid = jnp.where(curr_hit, jnp.clip((n_dot - 0.65) / 0.25, 0.0, 1.0), 1.0)

  valid_hist = jnp.where(in_bounds & hit_match, depth_valid * norm_valid, 0.0)
  if has_prev_history is not None:
    valid_hist = valid_hist * jnp.asarray(has_prev_history, jnp.float32)
  disocc_lr = jnp.clip(1.0 - valid_hist, 0.0, 1.0)

  # Exact 3x3 neighborhood AABB variance clipping in YCoCg space
  curr_ycc = rgb_to_ycocg(noisy_norm.astype(dt))
  inv9 = jnp.asarray(1.0 / 9.0, dt)
  mu_ycc = jax.lax.reduce_window(curr_ycc, jnp.asarray(0.0, dt), jax.lax.add, (1, 3, 3, 1), (1, 1, 1, 1), "SAME") * inv9
  mu2_ycc = jax.lax.reduce_window(curr_ycc * curr_ycc, jnp.asarray(0.0, dt), jax.lax.add, (1, 3, 3, 1), (1, 1, 1, 1), "SAME") * inv9
  sig_ycc = jnp.sqrt(jnp.maximum(mu2_ycc - mu_ycc * mu_ycc, jnp.asarray(1e-6, dt)))

  gamma_lo = jnp.array([2.0, 1.6, 1.6], dtype=dt)[None, None, None, None, :]
  gamma_hi = jnp.array([3.6, 1.6, 1.6], dtype=dt)[None, None, None, None, :]
  box_min = mu_ycc[:, :, :, None, :] - gamma_lo * sig_ycc[:, :, :, None, :]
  box_max = mu_ycc[:, :, :, None, :] + gamma_hi * sig_ycc[:, :, :, None, :]

  B_h, Hl_h, Wl_h, _ = hist_warped_s2d.shape
  hist_sub_rgb = (hist_warped_s2d / l_scale.astype(dt)).reshape(B_h, Hl_h, Wl_h, 4, 3)
  hist_sub_ycc_c = jnp.clip(rgb_to_ycocg(hist_sub_rgb), box_min, box_max)
  hist_sub_rgb_c = jnp.clip(ycocg_to_rgb(hist_sub_ycc_c), jnp.asarray(0.0, dt), jnp.asarray(6.0, dt))

  clip_mag_lr = jnp.clip(
      jnp.mean(jnp.abs(hist_sub_rgb_c - hist_sub_rgb), axis=(3, 4))[..., None],
      jnp.asarray(0.0, dt),
      jnp.asarray(2.0, dt),
  )
  hist_clamped_s2d_norm = hist_sub_rgb_c.reshape(B_h, Hl_h, Wl_h, 12)
  return hist_clamped_s2d_norm, disocc_lr, clip_mag_lr


# ==============================================================================
# 5. BF16 MXU Temporal + 2x Spatial Super-Resolution Ray Reconstructor
# ==============================================================================

class _DLSSResBlock(nn.Module):
  features: int

  @nn.compact
  def __call__(self, x: jax.Array) -> jax.Array:
    dt = jnp.bfloat16
    y = nn.relu(nn.Conv(self.features, (3, 3), padding="SAME", dtype=dt, param_dtype=jnp.float32)(x))
    y = nn.Conv(self.features, (3, 3), padding="SAME", dtype=dt, param_dtype=jnp.float32)(y)
    return nn.relu(x + y)


def _upsample_nn2x_bf16(x: jax.Array) -> jax.Array:
  """Zero-FP32-cast 2x nearest upsample in BF16 (followed by 3x3 conv in decoder)."""
  B, H, W, C = x.shape
  return jnp.broadcast_to(x[:, :, None, :, None, :], (B, H, 2, W, 2, C)).reshape(B, H * 2, W * 2, C)


def _bilinear_2x_s2d_bf16(x_lr: jax.Array) -> jax.Array:
  """Compute 2x bilinear upsample packed as SpaceToDepth(2) (B, Hl, Wl, 4, C) in BF16."""
  dt = jnp.bfloat16
  B, Hl, Wl, C = x_lr.shape
  xp = jnp.pad(x_lr.astype(dt), ((0, 0), (1, 1), (1, 1), (0, 0)), mode="edge")
  c_c = xp[:, 1 : Hl + 1, 1 : Wl + 1]
  c_l = xp[:, 1 : Hl + 1, 0:Wl]
  c_r = xp[:, 1 : Hl + 1, 2 : Wl + 2]
  c_u = xp[:, 0:Hl, 1 : Wl + 1]
  c_d = xp[:, 2 : Hl + 2, 1 : Wl + 1]
  c_ul = xp[:, 0:Hl, 0:Wl]
  c_ur = xp[:, 0:Hl, 2 : Wl + 2]
  c_dl = xp[:, 2 : Hl + 2, 0:Wl]
  c_dr = xp[:, 2 : Hl + 2, 2 : Wl + 2]
  w0, w1, w2 = jnp.asarray(0.5625, dt), jnp.asarray(0.1875, dt), jnp.asarray(0.0625, dt)
  p00 = w0 * c_c + w1 * (c_l + c_u) + w2 * c_ul
  p01 = w0 * c_c + w1 * (c_r + c_u) + w2 * c_ur
  p10 = w0 * c_c + w1 * (c_l + c_d) + w2 * c_dl
  p11 = w0 * c_c + w1 * (c_r + c_d) + w2 * c_dr
  return jnp.stack([p00, p01, p10, p11], axis=3)


def build_lr_dlss_features(
    noisy_norm: jax.Array,            # (B, Hl, Wl_loc, 3) BF16
    mu3_norm: jax.Array,              # (B, Hl, Wl_loc, 3) BF16
    sig3_norm: jax.Array,             # (B, Hl, Wl_loc, 3) BF16
    gbuf_lr: jax.Array,               # (B, Hl, Wl_loc, 9)
    hist_clamped_s2d_norm: jax.Array, # (B, Hl, Wl_loc, 12) BF16
    mv_lr: jax.Array,                 # (B, Hl, Wl_loc, 2)
    sub_offset_lr: jax.Array,         # (B, Hl, Wl_loc, 2) BF16
    disocc_lr: jax.Array,             # (B, Hl, Wl_loc, 1)
    clip_mag_lr: jax.Array,           # (B, Hl, Wl_loc, 1) BF16
    jitter_lr: Optional[jax.Array] = None,
) -> jax.Array:
  """Pack (B, Hl, Wl_loc, 44) BF16 features for TPU-DLSS 3.5."""
  dt = jnp.bfloat16
  B, Hl, Wl_loc, _ = noisy_norm.shape
  albedo = gbuf_lr[..., 0:3].astype(dt)
  normal = gbuf_lr[..., 3:6].astype(dt)
  depth_norm = gbuf_lr[..., 6:7].astype(dt)
  roughness = gbuf_lr[..., 7:8].astype(dt)
  mcode_norm = gbuf_lr[..., 8:9].astype(dt)

  hit_mask = (depth_norm > jnp.asarray(0.0, dt)).astype(dt)
  is_glass = (mcode_norm >= jnp.asarray(0.85, dt)).astype(dt)
  noisy_demod = jnp.clip(
      noisy_norm / jnp.maximum(albedo, jnp.asarray(0.08, dt)),
      jnp.asarray(0.0, dt),
      jnp.asarray(6.0, dt),
  ) * jnp.asarray(0.25, dt)

  lum_noisy = (
      jnp.asarray(0.2126, dt) * noisy_norm[..., 0:1]
      + jnp.asarray(0.7152, dt) * noisy_norm[..., 1:2]
      + jnp.asarray(0.0722, dt) * noisy_norm[..., 2:3]
  )
  std_lum = (
      jnp.asarray(0.2126, dt) * sig3_norm[..., 0:1]
      + jnp.asarray(0.7152, dt) * sig3_norm[..., 1:2]
      + jnp.asarray(0.0722, dt) * sig3_norm[..., 2:3]
  )

  npad = jnp.pad(normal, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="edge")
  zpad = jnp.pad(
      jnp.where(hit_mask > jnp.asarray(0.5, dt), depth_norm, jnp.asarray(2.0, dt)),
      ((0, 0), (1, 1), (1, 1), (0, 0)),
      mode="edge",
  )
  n_edge = jnp.clip(
      jnp.sum(jnp.abs(npad[:, 2 : Hl + 2, 1 : Wl_loc + 1] - npad[:, :Hl, 1 : Wl_loc + 1]), axis=-1, keepdims=True)
      + jnp.sum(jnp.abs(npad[:, 1 : Hl + 1, 2 : Wl_loc + 2] - npad[:, 1 : Hl + 1, :Wl_loc]), axis=-1, keepdims=True),
      jnp.asarray(0.0, dt),
      jnp.asarray(2.0, dt),
  )
  z_edge = jnp.clip(
      jnp.abs(zpad[:, 2 : Hl + 2, 1 : Wl_loc + 1] - zpad[:, :Hl, 1 : Wl_loc + 1])
      + jnp.abs(zpad[:, 1 : Hl + 1, 2 : Wl_loc + 2] - zpad[:, 1 : Hl + 1, :Wl_loc]),
      jnp.asarray(0.0, dt),
      jnp.asarray(1.0, dt),
  )

  mv_scaled = jnp.clip(mv_lr * (1.0 / 32.0), -2.0, 2.0).astype(dt)
  mv_mag = jnp.clip(jnp.sqrt(jnp.sum(mv_scaled * mv_scaled, axis=-1, keepdims=True) + jnp.asarray(1e-6, dt)), jnp.asarray(0.0, dt), jnp.asarray(2.5, dt))
  dis_bf = disocc_lr.astype(dt)
  val_bf = jnp.asarray(1.0, dt) - dis_bf
  if jitter_lr is not None:
    jx_b = jnp.broadcast_to(jnp.asarray(jitter_lr[0], dt), (B, Hl, Wl_loc, 1))
    jy_b = jnp.broadcast_to(jnp.asarray(jitter_lr[1], dt), (B, Hl, Wl_loc, 1))
  else:
    jx_b = jnp.zeros((B, Hl, Wl_loc, 1), dtype=dt)
    jy_b = jnp.zeros((B, Hl, Wl_loc, 1), dtype=dt)

  return jnp.concatenate(
      [
          noisy_norm,             # 3
          mu3_norm,               # 3
          albedo,                 # 3
          noisy_demod,            # 3
          normal,                 # 3
          depth_norm,             # 1
          roughness,              # 1
          mcode_norm,             # 1
          is_glass,               # 1
          hit_mask,               # 1
          lum_noisy,              # 1
          std_lum,                # 1
          n_edge,                 # 1
          z_edge,                 # 1  -> 24 curr feats
          hist_clamped_s2d_norm,  # 12 -> 36
          sub_offset_lr,          # 2  -> 38
          mv_mag,                 # 1
          dis_bf,                 # 1
          val_bf,                 # 1
          clip_mag_lr,            # 1
          jx_b,                   # 1
          jy_b,                   # 1  -> 44 total
      ],
      axis=-1,
  )


class TPUDLSS35Reconstructor(nn.Module):
  """MXU-aligned BF16 Temporal + 2x Super-Resolution Ray Reconstructor.

  Operates convolutions at H/2, H/4, H/8 and outputs 2x2 sub-pixel predictions
  directly in SpaceToDepth(2) space for zero-overhead 1080p-domain demodulation.
  """

  c1: int = 48
  c2: int = 96
  c3: int = 128

  @nn.compact
  def __call__(self, x_lr: jax.Array, train: bool = False) -> jax.Array:
    del train
    dt = jnp.bfloat16
    B, Hl, Wl, _ = x_lr.shape
    x_bf = x_lr.astype(dt)

    h1 = nn.relu(nn.Conv(self.c1, (3, 3), padding="SAME", dtype=dt, param_dtype=jnp.float32)(x_bf))
    h2 = nn.relu(nn.Conv(self.c2, (3, 3), strides=(2, 2), padding="SAME", dtype=dt, param_dtype=jnp.float32)(h1))
    h2 = _DLSSResBlock(self.c2)(h2)
    h3 = nn.relu(nn.Conv(self.c3, (3, 3), strides=(2, 2), padding="SAME", dtype=dt, param_dtype=jnp.float32)(h2))
    h3 = _DLSSResBlock(self.c3)(h3)

    h3_up = _upsample_nn2x_bf16(h3)
    d2 = nn.relu(nn.Conv(self.c2, (3, 3), padding="SAME", dtype=dt, param_dtype=jnp.float32)(jnp.concatenate([h2, h3_up], axis=-1)))
    d2_up = _upsample_nn2x_bf16(d2)
    d1 = nn.relu(nn.Conv(self.c1, (3, 3), padding="SAME", dtype=dt, param_dtype=jnp.float32)(jnp.concatenate([h1, d2_up], axis=-1)))

    out_s2d = nn.Conv(40, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(d1)
    return out_s2d.reshape(B, Hl, Wl, 4, 10)


def apply_tpu_dlss(
    model: TPUDLSS35Reconstructor,
    params: Any,
    curr_noisy_lr: jax.Array,      # (B, Hl, Wl_loc, 3)
    curr_gbuf_lr: jax.Array,       # (B, Hl, Wl_loc, 9)
    prev_hist_hr: jax.Array,       # (B, H, W_loc, 3)
    prev_gbuf_lr: jax.Array,       # (B, Hl, Wl_loc, 9)
    curr_cxp: jax.Array,
    curr_cxm: jax.Array,
    prev_cxp: jax.Array,
    prev_cxm: jax.Array,
    cam_fovy: float,
    znear: float,
    full_width_lr: int,
    x_offset_lr: Any = 0,
    jitter_lr: Optional[jax.Array] = None,
    has_prev_history: Optional[jax.Array] = None,
    axis_name: Optional[str] = None,
    halo_lr: int = 32,
    use_rgb10: bool = True,
    warp_mode: str = "4x4",
    train: bool = False,
    return_aux: bool = False,
    full_height_lr: Optional[int] = None,
    y_offset_lr: Any = 0,
) -> Tuple[Any, ...]:
  """Run one full TPU-DLSS 3.5 temporal + 2x super-resolution reconstruction step."""
  dt = jnp.bfloat16
  B, Hl, Wl_loc, _ = curr_noisy_lr.shape
  H, W_loc = Hl * 2, Wl_loc * 2

  l_scale = compute_exposure_scale(curr_noisy_lr, axis_name=axis_name)
  noisy_norm = jnp.clip((curr_noisy_lr / l_scale).astype(dt), jnp.asarray(0.0, dt), jnp.asarray(6.0, dt))

  mv_lr, prev_uv, expected_d_prev_norm, valid_z = compute_motion_vectors(
      curr_gbuf_lr[..., 6:7],
      curr_cxp,
      curr_cxm,
      prev_cxp,
      prev_cxm,
      cam_fovy=cam_fovy,
      znear=znear,
      full_width_lr=full_width_lr,
      x_offset_lr=x_offset_lr,
      jitter_lr=jitter_lr,
      full_height_lr=full_height_lr,
      y_offset_lr=y_offset_lr,
  )

  if warp_mode == "s2d12":
    hist_warped_s2d, prev_geom_warped, sub_offset_lr = warp_history_and_gbuf_s2d12(
        prev_hist_hr, prev_gbuf_lr[..., 3:7], mv_lr, axis_name=axis_name, halo_lr=halo_lr, use_rgb10=use_rgb10
    )
    w_hist_cap = 0.55
  else:
    hist_warped_s2d, prev_geom_warped = warp_history_and_gbuf_4x4(
        prev_hist_hr, prev_gbuf_lr[..., 3:7], mv_lr, axis_name=axis_name, halo_lr=halo_lr, use_rgb10=use_rgb10
    )
    sub_offset_lr = jnp.clip(mv_lr * (1.0 / 32.0), -2.0, 2.0).astype(dt)
    w_hist_cap = 0.90

  inv9 = jnp.asarray(1.0 / 9.0, dt)
  mu3_norm = jax.lax.reduce_window(noisy_norm, jnp.asarray(0.0, dt), jax.lax.add, (1, 3, 3, 1), (1, 1, 1, 1), "SAME") * inv9
  mu3_sq = jax.lax.reduce_window(noisy_norm * noisy_norm, jnp.asarray(0.0, dt), jax.lax.add, (1, 3, 3, 1), (1, 1, 1, 1), "SAME") * inv9
  sig3_norm = jnp.sqrt(jnp.maximum(mu3_sq - mu3_norm * mu3_norm, jnp.asarray(1e-6, dt)))

  hist_clamped_s2d_norm, disocc_lr, clip_mag_lr = compute_disocclusion_and_ycocg_clamp(
      noisy_norm,
      curr_gbuf_lr,
      prev_geom_warped,
      hist_warped_s2d,
      prev_uv,
      expected_d_prev_norm,
      valid_z,
      l_scale,
      has_prev_history=has_prev_history,
  )

  feats_lr = build_lr_dlss_features(
      noisy_norm,
      mu3_norm,
      sig3_norm,
      curr_gbuf_lr,
      hist_clamped_s2d_norm,
      mv_lr,
      sub_offset_lr,
      disocc_lr,
      clip_mag_lr,
      jitter_lr=jitter_lr,
  )

  out_s2d = model.apply({"params": params}, feats_lr, train=train)
  irrad_s2d = out_s2d[..., 0:3]
  spec_s2d = out_s2d[..., 3:6]
  albedo_res_s2d = out_s2d[..., 6:9]
  w_hist_logit_s2d = out_s2d[..., 9:10]

  albedo_base_s2d = _bilinear_2x_s2d_bf16(curr_gbuf_lr[..., 0:3])
  albedo_s2d = jnp.clip(
      albedo_base_s2d + jnp.asarray(0.25, dt) * jnp.tanh(albedo_res_s2d),
      jnp.asarray(0.02, dt),
      jnp.asarray(1.0, dt),
  )
  disocc_s2d = disocc_lr.astype(dt)[:, :, :, None, :]
  hist_s2d = hist_clamped_s2d_norm.reshape(B, Hl, Wl_loc, 4, 3)

  w_hist = jax.nn.sigmoid(w_hist_logit_s2d) * (jnp.asarray(1.0, dt) - disocc_s2d) * jnp.asarray(w_hist_cap, dt)
  pred_curr_s2d = albedo_s2d * irrad_s2d + spec_s2d
  pred_s2d = (jnp.asarray(1.0, dt) - w_hist) * pred_curr_s2d + w_hist * hist_s2d
  pred_hr = jnp.clip(
      (l_scale.astype(dt)[:, :, :, None, :] * pred_s2d)
      .reshape(B, Hl, Wl_loc, 2, 2, 3)
      .transpose(0, 1, 3, 2, 4, 5)
      .reshape(B, H, W_loc, 3)
      .astype(jnp.float32),
      0.0,
      1.0,
  )
  if return_aux:
    return pred_hr, disocc_lr, mv_lr, hist_warped_s2d
  return pred_hr, disocc_lr


# ==============================================================================
# 6. Multi-Chip Sharded Runner Builders (Pure Kernel)
# ==============================================================================

def build_sharded_tpu_dlss_fn(
    model: TPUDLSS35Reconstructor,
    params: Any,
    devices: Sequence[jax.Device],
    hr_height: int,
    hr_width: int,
    cam_fovy: float = 48.0,
    znear: float = 0.05,
    halo_lr: int = 32,
    use_rgb10: bool = True,
    warp_mode: str = "4x4",
):
  """Compile 1-chip or N-chip sharded TPU-DLSS 3.5 reconstructor (`H/2xW/2 -> HxW`)."""
  n_chips = len(devices)
  lr_h, lr_w = hr_height // 2, hr_width // 2
  assert lr_w % n_chips == 0 and (lr_w // n_chips) % 4 == 0 and lr_h % 4 == 0
  Wl_loc = lr_w // n_chips
  sm_kw = {"check_rep": False} if "check_rep" in inspect.signature(shard_map).parameters else {"check_vma": False}

  if n_chips == 1:
    dev = devices[0]
    p_d = jax.device_put(jax.tree_util.tree_map(jnp.asarray, params), dev)

    @jax.jit
    def dlss_fn(p, n_lr, gb_lr, p_hist_hr, p_gb_lr, c_cxp, c_cxm, p_cxp, p_cxm, jit_lr, has_prev):
      pred_hr, _ = apply_tpu_dlss(
          model, p, n_lr, gb_lr, p_hist_hr, p_gb_lr,
          c_cxp, c_cxm, p_cxp, p_cxm,
          cam_fovy=cam_fovy, znear=znear, full_width_lr=lr_w,
          x_offset_lr=0, jitter_lr=jit_lr, has_prev_history=has_prev,
          axis_name=None, halo_lr=halo_lr, use_rgb10=use_rgb10, warp_mode=warp_mode, train=False,
      )
      return pred_hr

    return dlss_fn, p_d, None, None

  mesh = Mesh(np.array(devices), ("dp",))
  rep = NamedSharding(mesh, P())
  sh_img = NamedSharding(mesh, P(None, None, "dp", None))
  sh_dp = NamedSharding(mesh, P("dp"))
  p_d = jax.device_put(jax.tree_util.tree_map(jnp.asarray, params), rep)
  cids = jax.device_put(np.arange(n_chips, dtype=np.int32), sh_dp)

  def _local_dlss(p_l, n_l, gb_l, ph_l, pgb_l, ccxp, ccxm, pcxp, pcxm, jit_l, hp_l, cid_arr):
    x_off = cid_arr[0] * Wl_loc
    pred_hr, _ = apply_tpu_dlss(
        model, p_l, n_l, gb_l, ph_l, pgb_l,
        ccxp, ccxm, pcxp, pcxm,
        cam_fovy=cam_fovy, znear=znear, full_width_lr=lr_w,
        x_offset_lr=x_off, jitter_lr=jit_l, has_prev_history=hp_l,
        axis_name="dp", halo_lr=halo_lr, use_rgb10=use_rgb10, warp_mode=warp_mode, train=False,
    )
    return pred_hr

  sh_fn = shard_map(
      _local_dlss,
      mesh=mesh,
      in_specs=(
          P(),
          P(None, None, "dp", None),
          P(None, None, "dp", None),
          P(None, None, "dp", None),
          P(None, None, "dp", None),
          P(), P(), P(), P(), P(), P(), P("dp"),
      ),
      out_specs=P(None, None, "dp", None),
      **sm_kw,
  )

  @jax.jit
  def dlss_4c_fn(p, n_lr, gb_lr, p_hist_hr, p_gb_lr, c_cxp, c_cxm, p_cxp, p_cxm, jit_lr, has_prev):
    return sh_fn(p, n_lr, gb_lr, p_hist_hr, p_gb_lr, c_cxp, c_cxm, p_cxp, p_cxm, jit_lr, has_prev, cids)

  return dlss_4c_fn, p_d, mesh, (rep, sh_img)
