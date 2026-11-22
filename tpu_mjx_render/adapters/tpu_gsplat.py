"""TPU-Native `gsplat`-Compatible 3D Gaussian Splatting Adapter (`tpu_gsplat`).

Drop-in `gsplat`-style API on JAX / TPU v5p powered by the V3 Mosaic Pallas
kernel (chunked VMEM `take_along_axis` + zero-bloat multi-tile MXU prefix-sum
transmittance) with:
  - Zero host callbacks (`host_callbacks == 0`)
  - Zero send/recv ops (`send_recv_ops == 0`)
  - Zero HBM gather ops (`gather_ops == 0`)
  - Full `jax.grad` support via `@jax.custom_vjp` using multi-array `jax.lax.sort`
    + `jax.checkpoint` MXU chunked scan (also `gather_ops == 0`, `host_callbacks == 0`)
  - Exact Z-buffer Real-to-Sim compositing helpers for MJX G-Buffer integration.
"""

from __future__ import annotations

import inspect
import json
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
import numpy as np


def _quats_to_rotmat(quats: jax.Array) -> jax.Array:
  """Convert `(..., 4)` `(w, x, y, z)` quaternions to `(..., 3, 3)` rotation matrices."""
  q = quats / jnp.maximum(jnp.linalg.norm(quats, axis=-1, keepdims=True), 1e-12)
  w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
  r00 = 1.0 - 2.0 * (y * y + z * z)
  r01 = 2.0 * (x * y - z * w)
  r02 = 2.0 * (x * z + y * w)
  r10 = 2.0 * (x * y + z * w)
  r11 = 1.0 - 2.0 * (x * x + z * z)
  r12 = 2.0 * (y * z - x * w)
  r20 = 2.0 * (x * z - y * w)
  r21 = 2.0 * (y * z + x * w)
  r22 = 1.0 - 2.0 * (x * x + y * y)
  row0 = jnp.stack([r00, r01, r02], axis=-1)
  row1 = jnp.stack([r10, r11, r12], axis=-1)
  row2 = jnp.stack([r20, r21, r22], axis=-1)
  return jnp.stack([row0, row1, row2], axis=-2)


def _pallas_gs3d_v3_fused_rasterize(
    splats_b16n: jax.Array,  # (C, 16, N_pad)
    top_idx: jax.Array,      # (C, T, K) int32, -1 for invalid
    t_ox: np.ndarray,        # (T,) float32
    t_oy: np.ndarray,        # (T,) float32
    tile_size: int = 8,
    use_bf16_mxu: bool = False,
    t_grp: int = 16,
    interpret: bool = False,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
  """Fused VMEM chunked take_along_axis gather + zero-padding multi-tile MXU rasterizer."""
  C, _, N_pad = splats_b16n.shape
  _, T, K_in = top_idx.shape
  P_t = tile_size * tile_size  # 64 pixels per 8x8 tile
  assert P_t == 64, "tpu_gsplat Mosaic kernel requires 8x8 tiles (P_t=64)"
  assert K_in <= 128, "Single Pallas pass expects K_in <= 128"
  K = 128
  if K_in < K:
    top_idx = jnp.pad(top_idx, ((0, 0), (0, 0), (0, K - K_in)), constant_values=-1)
  assert N_pad % 128 == 0, "N_pad must be a multiple of 128"
  assert T % t_grp == 0 and t_grp in (4, 8, 16), f"T={T} must be divisible by t_grp={t_grp}"

  n_chunks = N_pad // 128
  n_tg = T // t_grp
  out_cols = t_grp * 8

  lx = np.tile(np.arange(tile_size, dtype=np.float32), tile_size) + 0.5
  ly = np.repeat(np.arange(tile_size, dtype=np.float32), tile_size) + 0.5
  loc_xy_dev = jnp.asarray(
      np.stack(
          [np.broadcast_to(lx[:, None], (P_t, K)), np.broadcast_to(ly[:, None], (P_t, K))],
          axis=0,
      ).reshape(2 * P_t, K),
      dtype=jnp.float32,
  )

  u_np = np.triu(np.ones((K, K), dtype=np.float32), k=1)
  u_dev = jnp.asarray(u_np, dtype=jnp.bfloat16 if use_bf16_mxu else jnp.float32)

  oxy_rows = ((2 * t_grp + 7) // 8) * 8
  t_oxy_np = np.zeros((n_tg, oxy_rows, 128), dtype=np.float32)
  t_oxy_np[:, :t_grp, :] = t_ox.reshape(n_tg, t_grp, 1)
  t_oxy_np[:, t_grp : 2 * t_grp, :] = t_oy.reshape(n_tg, t_grp, 1)
  t_oxy_dev = jnp.asarray(t_oxy_np.reshape(n_tg * oxy_rows, 128), dtype=jnp.float32)

  def _fused_kernel(s_ref, i_ref, oxy_ref, lxy_ref, u_ref, out_ref):
    s_all = s_ref[...]    # (16, N_pad)
    ix = i_ref[...]       # (t_grp, 128)
    oxy = oxy_ref[...]    # (oxy_rows, 128)
    lxy = lxy_ref[...]    # (128, 128)
    lx_m, ly_m = lxy[0:64, :], lxy[64:128, :]
    u_m = u_ref[...]
    ones_k = jnp.ones((1, K), dtype=jnp.float32)

    ix_tiles = [jnp.broadcast_to(ix[t : t + 1, :], (16, K)) for t in range(t_grp)]
    accs = [jnp.zeros((16, K), dtype=jnp.float32) for _ in range(t_grp)]
    for c in range(n_chunks):
      base = c * 128
      s_c = s_all[:, base : base + 128]
      for t in range(t_grp):
        ix_t = ix_tiles[t]
        mask = (ix_t >= base) & (ix_t < base + 128)
        local_ix = jnp.where(mask, ix_t - base, 0)
        vals = jnp.take_along_axis(s_c, local_ix, axis=1)
        accs[t] = accs[t] + jnp.where(mask, vals, 0.0)

    outs = []
    for t in range(t_grp):
      sf = accs[t]
      px = jnp.broadcast_to(oxy[t : t + 1, 0:1], (P_t, K)) + lx_m
      py = jnp.broadcast_to(oxy[t_grp + t : t_grp + t + 1, 0:1], (P_t, K)) + ly_m
      dx = px - jnp.broadcast_to(sf[0:1, :], (P_t, K))
      dy = py - jnp.broadcast_to(sf[1:2, :], (P_t, K))
      cxx = jnp.broadcast_to(sf[2:3, :], (P_t, K))
      cxy = jnp.broadcast_to(sf[3:4, :], (P_t, K))
      cyy = jnp.broadcast_to(sf[4:5, :], (P_t, K))
      gop = jnp.broadcast_to(sf[8:9, :], (P_t, K))

      power = -0.5 * (cxx * dx * dx + 2.0 * cxy * dx * dy + cyy * dy * dy)
      valid_p = (power <= 0.0) & (power > -16.0)
      alpha = jnp.where(valid_p, jnp.minimum(0.99, gop * jnp.exp(jnp.maximum(power, -16.0))), 0.0)
      alpha = jnp.where(alpha >= (1.0 / 255.0), alpha, 0.0)

      log_1ma = jnp.log(jnp.maximum(1.0 - alpha, 1.0e-7))
      if use_bf16_mxu:
        log_tr = jnp.matmul(log_1ma.astype(jnp.bfloat16), u_m, preferred_element_type=jnp.float32)
      else:
        log_tr = jnp.matmul(log_1ma, u_m, precision=jax.lax.Precision.HIGHEST)
      trans = jnp.exp(jnp.maximum(log_tr, -30.0))
      w = alpha * trans

      attr_t = jnp.transpose(jnp.concatenate([sf[5:8, :], sf[9:13, :], ones_k], axis=0), (1, 0))
      outs.append(jnp.matmul(w, attr_t, precision=jax.lax.Precision.HIGHEST))

    out_ref[...] = jnp.concatenate(outs, axis=1)

  out_raw = pl.pallas_call(
      _fused_kernel,
      out_shape=jax.ShapeDtypeStruct((C * n_tg * P_t, out_cols), jnp.float32),
      grid=(C, n_tg),
      in_specs=[
          pl.BlockSpec((16, N_pad), lambda c, tg: (c, 0)),
          pl.BlockSpec((t_grp, K), lambda c, tg: (c * n_tg + tg, 0)),
          pl.BlockSpec((oxy_rows, 128), lambda c, tg: (tg, 0)),
          pl.BlockSpec((2 * P_t, K), lambda c, tg: (0, 0)),
          pl.BlockSpec((K, K), lambda c, tg: (0, 0)),
      ],
      out_specs=pl.BlockSpec((P_t, out_cols), lambda c, tg: (c * n_tg + tg, 0)),
      compiler_params=None if interpret else pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
      interpret=interpret,
  )(
      splats_b16n.reshape(C * 16, N_pad),
      top_idx.reshape(C * T, K),
      t_oxy_dev,
      loc_xy_dev,
      u_dev,
  )

  unp = (
      out_raw.reshape(C, n_tg, P_t, t_grp, 8)
      .transpose(0, 1, 3, 2, 4)
      .reshape(C, T, tile_size, tile_size, 8)
  )
  return unp[..., 0:3], unp[..., 3], unp[..., 7]


def _project_to_splats(
    means: jax.Array,      # (N, 3)
    quats: jax.Array,      # (N, 4) (w, x, y, z)
    scales: jax.Array,     # (N, 3)
    opacities: jax.Array,  # (N,)
    colors: jax.Array,     # (C, N, 3)
    viewmats: jax.Array,   # (C, 4, 4)
    Ks: jax.Array,         # (C, 3, 3)
    width: int,
    height: int,
    near_plane: float,
    far_plane: float,
    eps2d: float,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
  """Project 3D Gaussians into `(C, 16, N_pad)` EWA splat attributes (zero gathers)."""
  C = viewmats.shape[0]
  N = means.shape[0]
  N_pad = ((N + 127) // 128) * 128

  R_q = _quats_to_rotmat(quats)                       # (N, 3, 3)
  M_world = R_q * jnp.maximum(scales, 1e-6)[:, None, :]  # (N, 3, 3)

  R_wc = viewmats[:, :3, :3]  # (C, 3, 3)
  t_wc = viewmats[:, :3, 3]   # (C, 3)
  p_cam = jnp.einsum("cij,nj->cni", R_wc, means, precision=jax.lax.Precision.HIGHEST) + t_wc[:, None, :]

  cx, cy, cz = p_cam[..., 0], p_cam[..., 1], p_cam[..., 2]
  z_d = cz
  in_front = (z_d > near_plane) & (z_d < far_plane)
  z_safe = jnp.maximum(z_d, near_plane)

  fx = Ks[:, 0, 0][:, None]
  fy = Ks[:, 1, 1][:, None]
  cx0 = Ks[:, 0, 2][:, None]
  cy0 = Ks[:, 1, 2][:, None]

  mean_x = fx * (cx / z_safe) + cx0
  mean_y = fy * (cy / z_safe) + cy0

  inv_z = 1.0 / z_safe
  inv_z2 = inv_z * inv_z
  tan_x = 0.5 * float(width) / jnp.maximum(fx, 1e-6)
  tan_y = 0.5 * float(height) / jnp.maximum(fy, 1e-6)
  lim_x = 1.3 * tan_x * z_safe
  lim_y = 1.3 * tan_y * z_safe
  tx = jnp.clip(cx, -lim_x, lim_x)
  ty = jnp.clip(cy, -lim_y, lim_y)

  j00 = fx * inv_z
  j02 = -fx * tx * inv_z2
  j11 = fy * inv_z
  j12 = -fy * ty * inv_z2

  w0 = j00[..., None] * R_wc[:, None, 0, :] + j02[..., None] * R_wc[:, None, 2, :]  # (C, N, 3)
  w1 = j11[..., None] * R_wc[:, None, 1, :] + j12[..., None] * R_wc[:, None, 2, :]  # (C, N, 3)

  a0 = jnp.einsum("cnj,njk->cnk", w0, M_world, precision=jax.lax.Precision.HIGHEST)
  a1 = jnp.einsum("cnj,njk->cnk", w1, M_world, precision=jax.lax.Precision.HIGHEST)

  cov2_xx = jnp.sum(a0 * a0, axis=-1) + eps2d
  cov2_xy = jnp.sum(a0 * a1, axis=-1)
  cov2_yy = jnp.sum(a1 * a1, axis=-1) + eps2d

  det = cov2_xx * cov2_yy - cov2_xy * cov2_xy
  valid_g = in_front & (det > 1e-6)
  inv_det = jnp.where(valid_g, 1.0 / jnp.maximum(det, 1e-6), 0.0)
  icxx = cov2_yy * inv_det
  icxy = -cov2_xy * inv_det
  icyy = cov2_xx * inv_det

  mid = 0.5 * (cov2_xx + cov2_yy)
  disc = jnp.sqrt(jnp.maximum(mid * mid - det, 0.0))
  radius = 3.0 * jnp.sqrt(jnp.maximum(mid + disc, 0.1))

  op_cn = jnp.broadcast_to(jnp.clip(opacities, 0.0, 1.0)[None, :], (C, N))
  op_cn = jnp.where(valid_g, op_cn, 0.0)
  z_cn = jnp.zeros((C, N), dtype=jnp.float32)

  splats_c16n = jnp.stack(
      [
          mean_x, mean_y, icxx, icxy, icyy,
          colors[..., 0], colors[..., 1], colors[..., 2],
          op_cn, z_d, z_cn, z_cn, z_cn,
          z_cn, z_cn, z_cn,
      ],
      axis=1,
  )
  if N_pad > N:
    splats_c16n = jnp.pad(splats_c16n, ((0, 0), (0, 0), (0, N_pad - N)))

  return splats_c16n, mean_x, mean_y, z_d, valid_g, radius


def _rasterize_pallas_primal(
    splats_c16n: jax.Array,  # (C, 16, N_pad)
    mean_x: jax.Array,       # (C, N)
    mean_y: jax.Array,       # (C, N)
    z_d: jax.Array,          # (C, N)
    valid_g: jax.Array,      # (C, N)
    radius: jax.Array,       # (C, N)
    t_ox: np.ndarray,        # (T,)
    t_oy: np.ndarray,        # (T,)
    tile_size: int,
    top_k: int,
    use_bf16_mxu: bool,
    use_approx_topk: bool,
    t_grp: int,
    interpret: bool,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
  """Zero-gather Mosaic Pallas rasterization (supports top_k in {128, 256, 512})."""
  C, _, N_pad = splats_c16n.shape
  N = mean_x.shape[1]
  T = len(t_ox)
  tox_dev = jnp.asarray(t_ox, dtype=jnp.float32)[None, :, None]
  toy_dev = jnp.asarray(t_oy, dtype=jnp.float32)[None, :, None]

  near_x = jnp.clip(mean_x[:, None, :], tox_dev, tox_dev + float(tile_size))
  near_y = jnp.clip(mean_y[:, None, :], toy_dev, toy_dev + float(tile_size))
  dx_t = mean_x[:, None, :] - near_x
  dy_t = mean_y[:, None, :] - near_y
  overlap = valid_g[:, None, :] & ((dx_t * dx_t + dy_t * dy_t) <= (radius * radius)[:, None, :])

  score = jnp.where(overlap, -z_d[:, None, :], -1.0e9)
  k_req = ((min(top_k, N) + 127) // 128) * 128
  k_req = max(128, k_req)
  k_use = min(k_req, N)

  if use_approx_topk and not interpret and N >= k_use:
    top_scores, top_idx = jax.lax.approx_max_k(score, k_use, recall_target=1.0)
    neg_ts, top_idx = jax.lax.sort_key_val(-top_scores, top_idx, dimension=-1)
    top_scores = -neg_ts
  else:
    top_scores, top_idx = jax.lax.top_k(score, k_use)
  top_idx = jnp.where(top_scores > -1.0e8, top_idx, -1)
  if k_use < k_req:
    top_idx = jnp.pad(top_idx, ((0, 0), (0, 0), (0, k_req - k_use)), constant_values=-1)

  if k_req == 128:
    return _pallas_gs3d_v3_fused_rasterize(
        splats_c16n, top_idx, t_ox, t_oy,
        tile_size=tile_size, use_bf16_mxu=use_bf16_mxu, t_grp=t_grp, interpret=interpret,
    )

  rgb_acc = jnp.zeros((C, T, tile_size, tile_size, 3), dtype=jnp.float32)
  dep_acc = jnp.zeros((C, T, tile_size, tile_size), dtype=jnp.float32)
  t_acc = jnp.ones((C, T, tile_size, tile_size), dtype=jnp.float32)
  for pass_i in range(k_req // 128):
    rc, dc, ac = _pallas_gs3d_v3_fused_rasterize(
        splats_c16n, top_idx[..., pass_i * 128 : (pass_i + 1) * 128],
        t_ox, t_oy, tile_size=tile_size, use_bf16_mxu=use_bf16_mxu, t_grp=t_grp, interpret=interpret,
    )
    rgb_acc = rgb_acc + t_acc[..., None] * rc
    dep_acc = dep_acc + t_acc * dc
    t_acc = t_acc * jnp.maximum(1.0 - ac, 0.0)
  return rgb_acc, dep_acc, 1.0 - t_acc


def _rasterize_diff_scan(
    splats_c16n: jax.Array,  # (C, 16, N_pad)
    z_d: jax.Array,          # (C, N)
    valid_g: jax.Array,      # (C, N)
    t_ox: np.ndarray,        # (T,)
    t_oy: np.ndarray,        # (T,)
    tile_size: int = 8,
    use_bf16_mxu: bool = False,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
  """Pure-JAX zero-gather differentiable MXU chunked-scan rasterizer for `jax.grad`."""
  C, _, N_pad = splats_c16n.shape
  N = z_d.shape[1]
  T = len(t_ox)
  P_t = tile_size * tile_size
  K_chunk = 128
  n_chunks = N_pad // K_chunk

  # Pad valid mask + z_sort to N_pad, then sort all 10 active rows simultaneously via jax.lax.sort (ZERO gather_ops!)
  z_pad = jnp.where(valid_g, z_d, 1.0e9)
  if N_pad > N:
    z_pad = jnp.pad(z_pad, ((0, 0), (0, N_pad - N)), constant_values=1.0e9)

  # Channels needed: mx(0), my(1), icxx(2), icxy(3), icyy(4), r(5), g(6), b(7), op(8), zd(9)
  cols = [z_pad] + [splats_c16n[:, i, :] for i in range(10)]
  sorted_cols = jax.lax.sort(cols, dimension=-1, num_keys=1)
  s_sorted = jnp.stack(sorted_cols[1:], axis=1)  # (C, 10, N_pad)

  # Chunk into (n_chunks, C, 10, K_chunk)
  chunks = s_sorted.reshape(C, 10, n_chunks, K_chunk).transpose(2, 0, 1, 3)

  lx = np.tile(np.arange(tile_size, dtype=np.float32), tile_size) + 0.5
  ly = np.repeat(np.arange(tile_size, dtype=np.float32), tile_size) + 0.5
  px = jnp.asarray(t_ox[:, None] + lx[None, :], dtype=jnp.float32)[None, :, :, None]  # (1, T, P_t, 1)
  py = jnp.asarray(t_oy[:, None] + ly[None, :], dtype=jnp.float32)[None, :, :, None]  # (1, T, P_t, 1)
  u_mat = jnp.asarray(np.triu(np.ones((K_chunk, K_chunk), dtype=np.float32), k=1),
                      dtype=jnp.bfloat16 if use_bf16_mxu else jnp.float32)

  @jax.checkpoint
  def _scan_step(carry, ch):
    rgb_acc, dep_acc, t_acc = carry  # (C, T, P_t, 3), (C, T, P_t), (C, T, P_t)
    mx = ch[:, 0, None, None, :]
    my = ch[:, 1, None, None, :]
    cxx = ch[:, 2, None, None, :]
    cxy = ch[:, 3, None, None, :]
    cyy = ch[:, 4, None, None, :]
    gop = ch[:, 8, None, None, :]

    dx = px - mx
    dy = py - my
    power = -0.5 * (cxx * dx * dx + 2.0 * cxy * dx * dy + cyy * dy * dy)
    # Smooth gradient flow inside Gaussian support
    valid_p = (power <= 0.0) & (power > -16.0)
    alpha = jnp.where(valid_p, jnp.minimum(0.99, gop * jnp.exp(jnp.maximum(power, -16.0))), 0.0)

    log_1ma = jnp.log(jnp.maximum(1.0 - alpha, 1.0e-7))
    if use_bf16_mxu:
      log_tr = jnp.matmul(log_1ma.astype(jnp.bfloat16), u_mat, preferred_element_type=jnp.float32)
    else:
      log_tr = jnp.matmul(log_1ma, u_mat, precision=jax.lax.Precision.HIGHEST)
    trans = jnp.exp(jnp.maximum(log_tr, -30.0))
    w = alpha * trans  # (C, T, P_t, K_chunk)

    rgb_attr = jnp.transpose(ch[:, 5:8, :], (0, 2, 1))[:, None, :, :]  # (C, 1, K_chunk, 3)
    rgb_c = jnp.matmul(w, rgb_attr, precision=jax.lax.Precision.HIGHEST)
    dep_c = jnp.einsum("ctpk,ck->ctp", w, ch[:, 9, :], precision=jax.lax.Precision.HIGHEST)

    rgb_acc = rgb_acc + t_acc[..., None] * rgb_c
    dep_acc = dep_acc + t_acc * dep_c
    chunk_log_t = jnp.sum(log_1ma, axis=-1)
    t_acc = t_acc * jnp.exp(jnp.maximum(chunk_log_t, -30.0))
    return (rgb_acc, dep_acc, t_acc), None

  init = (
      jnp.zeros((C, T, P_t, 3), dtype=jnp.float32),
      jnp.zeros((C, T, P_t), dtype=jnp.float32),
      jnp.ones((C, T, P_t), dtype=jnp.float32),
  )
  (rgb_out, dep_out, t_final), _ = jax.lax.scan(_scan_step, init, chunks)
  alp_out = 1.0 - t_final
  return (
      rgb_out.reshape(C, T, tile_size, tile_size, 3),
      dep_out.reshape(C, T, tile_size, tile_size),
      alp_out.reshape(C, T, tile_size, tile_size),
  )


def _make_custom_vjp_rasterizer(
    t_ox: np.ndarray,
    t_oy: np.ndarray,
    tile_size: int,
    top_k: int,
    use_bf16_mxu: bool,
    use_approx_topk: bool,
    t_grp: int,
    interpret: bool,
):
  @jax.custom_vjp
  def _core(splats_c16n, mean_x, mean_y, z_d, valid_g, radius):
    return _rasterize_pallas_primal(
        splats_c16n, mean_x, mean_y, z_d, valid_g, radius,
        t_ox=t_ox, t_oy=t_oy, tile_size=tile_size, top_k=top_k,
        use_bf16_mxu=use_bf16_mxu, use_approx_topk=use_approx_topk,
        t_grp=t_grp, interpret=interpret,
    )

  def _fwd(splats_c16n, mean_x, mean_y, z_d, valid_g, radius):
    del mean_x, mean_y, radius
    out = _rasterize_diff_scan(
        splats_c16n, z_d, valid_g, t_ox=t_ox, t_oy=t_oy,
        tile_size=tile_size, use_bf16_mxu=use_bf16_mxu,
    )
    return out, (splats_c16n, z_d, valid_g)

  def _bwd(res, g_out):
    splats_c16n, z_d, valid_g = res
    _, vjp_fn = jax.vjp(
        lambda s: _rasterize_diff_scan(
            s, z_d, valid_g, t_ox=t_ox, t_oy=t_oy,
            tile_size=tile_size, use_bf16_mxu=use_bf16_mxu,
        ),
        splats_c16n,
    )
    (g_splats,) = vjp_fn(g_out)
    return (g_splats, None, None, None, None, None)

  _core.defvjp(_fwd, _bwd)
  return _core


def rasterization(
    means: jax.Array,       # (N, 3)
    quats: jax.Array,       # (N, 4) (w, x, y, z)
    scales: jax.Array,      # (N, 3)
    opacities: jax.Array,   # (N,)
    colors: jax.Array,      # (N, 3) or (C, N, 3)
    viewmats: jax.Array,    # (C, 4, 4) or (4, 4) world-to-camera (OpenCV)
    Ks: jax.Array,          # (C, 3, 3) or (3, 3)
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    backgrounds: Optional[jax.Array] = None,
    packed: bool = False,
    tile_size: int = 8,
    top_k: int = 128,
    use_bf16_mxu: bool = False,
    use_approx_topk: bool = True,
    t_grp: int = 16,
    eps2d: float = 0.30,
    interpret: bool = False,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
  """gsplat-compatible zero-copy TPU rasterizer returning `(rgb, alpha, depth)`."""
  del packed  # TPU adapter is natively tile-packed in VMEM
  if viewmats.ndim == 2:
    viewmats = viewmats[None, :, :]
  if Ks.ndim == 2:
    Ks = Ks[None, :, :]
  C = viewmats.shape[0]
  N = means.shape[0]
  if colors.ndim == 2:
    colors = jnp.broadcast_to(colors[None, :, :], (C, N, 3))

  # Pad up to 32x32 macro-tile grid so T is always divisible by t_grp=16
  H_pad = ((height + 31) // 32) * 32
  W_pad = ((width + 31) // 32) * 32
  Ty = H_pad // tile_size
  Tx = W_pad // tile_size
  t_ox = np.tile(np.arange(Tx, dtype=np.float32) * tile_size, Ty)
  t_oy = np.repeat(np.arange(Ty, dtype=np.float32) * tile_size, Tx)

  splats_c16n, mean_x, mean_y, z_d, valid_g, radius = _project_to_splats(
      means, quats, scales, opacities, colors, viewmats, Ks,
      width=width, height=height, near_plane=near_plane, far_plane=far_plane, eps2d=eps2d,
  )

  core_fn = _make_custom_vjp_rasterizer(
      t_ox=t_ox, t_oy=t_oy, tile_size=tile_size, top_k=top_k,
      use_bf16_mxu=use_bf16_mxu, use_approx_topk=use_approx_topk,
      t_grp=t_grp, interpret=interpret,
  )
  rgb_t, dep_t, alp_t = core_fn(splats_c16n, mean_x, mean_y, z_d, valid_g, radius)

  rgb_img = rgb_t.reshape(C, Ty, Tx, tile_size, tile_size, 3).transpose(0, 1, 3, 2, 4, 5).reshape(C, H_pad, W_pad, 3)[:, :height, :width, :]
  dep_raw = dep_t.reshape(C, Ty, Tx, tile_size, tile_size).transpose(0, 1, 3, 2, 4).reshape(C, H_pad, W_pad)[:, :height, :width]
  alp_img = alp_t.reshape(C, Ty, Tx, tile_size, tile_size).transpose(0, 1, 3, 2, 4).reshape(C, H_pad, W_pad)[:, :height, :width]

  if backgrounds is not None:
    bg = jnp.asarray(backgrounds, dtype=jnp.float32)
    if bg.ndim == 1:
      bg = bg[None, :]
    rgb_img = rgb_img + (1.0 - alp_img[..., None]) * bg[:, None, None, :]

  depth_img = jnp.where(alp_img > 1e-4, dep_raw / jnp.maximum(alp_img, 1e-4), -1.0)
  return rgb_img, alp_img[..., None], depth_img[..., None]


def cam_pose_to_viewmat_K(
    cam_xpos: jax.Array,  # (3,)
    cam_xmat: jax.Array,  # (3, 3) OpenGL columns: (right, up, -forward)
    height: int,
    width: int,
    fovy_deg: float,
) -> Tuple[jax.Array, jax.Array]:
  """Convert MuJoCo/OpenGL camera pose `(cam_xpos, cam_xmat)` to gsplat `(viewmat, K)`."""
  flip = jnp.diag(jnp.array([1.0, -1.0, -1.0], dtype=jnp.float32))
  R_wc = flip @ cam_xmat.T
  t_wc = -(R_wc @ cam_xpos)
  viewmat = jnp.eye(4, dtype=jnp.float32).at[:3, :3].set(R_wc).at[:3, 3].set(t_wc)

  half_h = math.tan(0.5 * math.radians(float(fovy_deg)))
  fy = 0.5 * float(height) / half_h
  fx = fy
  K = jnp.array(
      [[fx, 0.0, 0.5 * float(width)], [0.0, fy, 0.5 * float(height)], [0.0, 0.0, 1.0]],
      dtype=jnp.float32,
  )
  return viewmat, K


def composite_mjx_and_3dgs(
    mjx_rgb: jax.Array,      # (..., H, W, 3)
    mjx_z_depth: jax.Array,  # (..., H, W) camera -Z depth (-1 on miss)
    gs_rgb: jax.Array,       # (..., H, W, 3)
    gs_alpha: jax.Array,     # (..., H, W)
    gs_z_depth: jax.Array,   # (..., H, W) camera +Z depth (-1 on miss)
) -> Tuple[jax.Array, jax.Array]:
  """Exact zero-copy Z-buffer compositor fusing analytic MJX robot and 3DGS environment."""
  mjx_hit = mjx_z_depth > 0.0
  gs_hit = (gs_alpha > 0.05) & (gs_z_depth > 0.0)
  mjx_front = mjx_hit & (~gs_hit | (mjx_z_depth <= gs_z_depth))

  comp_rgb = jnp.where(mjx_front[..., None], mjx_rgb, gs_rgb)
  comp_depth = jnp.where(
      mjx_front,
      mjx_z_depth,
      jnp.where(gs_hit, gs_z_depth, -1.0),
  )
  return jnp.clip(comp_rgb, 0.0, 1.0), comp_depth


# ---------------------------------------------------------------------------
# Spherical Harmonics (Degree 0..3) + Official 3DGS Binary .ply Loader
# ---------------------------------------------------------------------------

_SH_C0 = 0.28209479177387814
_SH_C1 = 0.4886025119029199
_SH_C2 = (
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
)
_SH_C3 = (
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
)


def eval_sh(deg: int, sh: jax.Array, dirs: jax.Array) -> jax.Array:
  """Evaluate Spherical Harmonics up to degree 3 for `(..., K, 3)` coeffs & `(..., 3)` unit dirs.

  Matches standard Inria 3DGS / `gsplat` convention (`dirs = normalize(means - cam_pos)`).
  Returns sRGB colors clamped to `[0, +inf)` (or `[0, 1]` after compositing).
  """
  result = _SH_C0 * sh[..., 0, :]
  if deg > 0:
    x = dirs[..., 0:1]
    y = dirs[..., 1:2]
    z = dirs[..., 2:3]
    result = result + _SH_C1 * (-y * sh[..., 1, :] + z * sh[..., 2, :] - x * sh[..., 3, :])
    if deg > 1:
      xx, yy, zz = x * x, y * y, z * z
      xy, yz, xz = x * y, y * z, x * z
      result = (
          result
          + _SH_C2[0] * xy * sh[..., 4, :]
          + _SH_C2[1] * yz * sh[..., 5, :]
          + _SH_C2[2] * (2.0 * zz - xx - yy) * sh[..., 6, :]
          + _SH_C2[3] * xz * sh[..., 7, :]
          + _SH_C2[4] * (xx - yy) * sh[..., 8, :]
      )
      if deg > 2:
        result = (
            result
            + _SH_C3[0] * y * (3.0 * xx - yy) * sh[..., 9, :]
            + _SH_C3[1] * xy * z * sh[..., 10, :]
            + _SH_C3[2] * y * (4.0 * zz - xx - yy) * sh[..., 11, :]
            + _SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh[..., 12, :]
            + _SH_C3[4] * x * (4.0 * zz - xx - yy) * sh[..., 13, :]
            + _SH_C3[5] * z * (xx - yy) * sh[..., 14, :]
            + _SH_C3[6] * x * (xx - 3.0 * yy) * sh[..., 15, :]
        )
  return jnp.maximum(result + 0.5, 0.0)


_PLY_DTYPES = {
    "char": "i1", "int8": "i1",
    "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2",
    "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4",
    "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4",
    "double": "f8", "float64": "f8",
}


def load_ply(ply_path: str, max_sh_degree: int = 3) -> Dict[str, Any]:
  """Load a standard 3D Gaussian Splatting binary `.ply` file into NumPy arrays.

  Applies standard 3DGS parameter activations:
    - `opacities = sigmoid(opacity_logit)`
    - `scales = exp(log_scales)`
    - `quats = normalize(rot_0..3)` in `(w, x, y, z)` order
    - `sh = concat([f_dc[:, None, :], f_rest.reshape(N, 3, K).transpose(0, 2, 1)], axis=1)`
  """
  with open(ply_path, "rb") as f:
    fmt = "binary_little_endian"
    n_verts = 0
    in_vertex = False
    props: List[Tuple[str, str]] = []
    while True:
      raw_line = f.readline()
      if not raw_line:
        raise ValueError(f"Unexpected EOF in PLY header: {ply_path}")
      line = raw_line.decode("ascii", errors="replace").strip()
      if line == "end_header":
        break
      parts = line.split()
      if not parts:
        continue
      if parts[0] == "format":
        fmt = parts[1]
      elif parts[0] == "element":
        in_vertex = parts[1] == "vertex"
        if in_vertex:
          n_verts = int(parts[2])
      elif parts[0] == "property" and in_vertex:
        if parts[1] == "list":
          raise ValueError(f"Unsupported list property in vertex element: {line}")
        props.append((parts[2], _PLY_DTYPES[parts[1]]))

    if fmt != "binary_little_endian":
      raise ValueError(f"Unsupported PLY format '{fmt}' (expected binary_little_endian)")

    dt = np.dtype([(name, "<" + code) for name, code in props])
    arr = np.fromfile(f, dtype=dt, count=n_verts)

  if len(arr) != n_verts:
    raise ValueError(f"Short read on {ply_path}: got {len(arr)} vertices, expected {n_verts}")

  names = set(arr.dtype.names or ())
  means = np.stack([arr["x"], arr["y"], arr["z"]], axis=-1).astype(np.float32)

  op_raw = arr["opacity"].astype(np.float32)
  opacities = (1.0 / (1.0 + np.exp(-np.clip(op_raw, -30.0, 30.0)))).astype(np.float32)

  scale_names = [n for n in ("scale_0", "scale_1", "scale_2") if n in names]
  scales = np.exp(
      np.clip(np.stack([arr[n] for n in scale_names], axis=-1).astype(np.float32), -20.0, 10.0)
  ).astype(np.float32)

  rot_names = [f"rot_{i}" for i in range(4)]
  quats = np.stack([arr[n] for n in rot_names], axis=-1).astype(np.float32)
  quats = quats / np.maximum(np.linalg.norm(quats, axis=-1, keepdims=True), 1e-12)

  f_dc = np.stack([arr["f_dc_0"], arr["f_dc_1"], arr["f_dc_2"]], axis=-1).astype(np.float32)

  rest_names = sorted(
      [n for n in names if n.startswith("f_rest_")],
      key=lambda s: int(s.split("_")[-1]),
  )
  if rest_names and max_sh_degree > 0:
    k_rest_total = len(rest_names) // 3
    f_rest_flat = np.stack([arr[n] for n in rest_names], axis=-1).astype(np.float32)
    # Inria save_ply stores features_rest.transpose(1, 2).flatten(1) -> (N, 3, K_rest)
    sh_rest = f_rest_flat.reshape(n_verts, 3, k_rest_total).transpose(0, 2, 1)
    k_max_rest = (max_sh_degree + 1) ** 2 - 1
    sh_rest = sh_rest[:, :k_max_rest, :]
    sh = np.concatenate([f_dc[:, None, :], sh_rest], axis=1).astype(np.float32)
  else:
    sh_rest = np.zeros((n_verts, 0, 3), dtype=np.float32)
    sh = f_dc[:, None, :].astype(np.float32)

  sh_degree = int(round(math.sqrt(sh.shape[1])) - 1)
  return {
      "means": np.ascontiguousarray(means, dtype=np.float32),
      "quats": np.ascontiguousarray(quats, dtype=np.float32),
      "scales": np.ascontiguousarray(scales, dtype=np.float32),
      "opacities": np.ascontiguousarray(opacities, dtype=np.float32),
      "sh_dc": np.ascontiguousarray(f_dc[:, None, :], dtype=np.float32),
      "sh_rest": np.ascontiguousarray(sh_rest, dtype=np.float32),
      "sh": np.ascontiguousarray(sh, dtype=np.float32),
      "sh_degree": sh_degree,
      "n_gaussians": int(n_verts),
  }


def load_cameras_json(cameras_json_path: str) -> List[Dict[str, Any]]:
  """Load Inria 3DGS `cameras.json` into OpenCV `(viewmat, K, position, rotation)` entries."""
  with open(cameras_json_path, "r") as f:
    raw = json.load(f)
  cams: List[Dict[str, Any]] = []
  for c in raw:
    pos = np.asarray(c["position"], dtype=np.float32)
    rot_c2w = np.asarray(c["rotation"], dtype=np.float32)
    R_wc = rot_c2w.T
    t_wc = -(R_wc @ pos)
    V = np.eye(4, dtype=np.float32)
    V[:3, :3] = R_wc
    V[:3, 3] = t_wc
    w, h = int(c["width"]), int(c["height"])
    fx, fy = float(c["fx"]), float(c["fy"])
    K = np.array(
        [[fx, 0.0, 0.5 * w], [0.0, fy, 0.5 * h], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    cams.append({
        "id": int(c.get("id", len(cams))),
        "img_name": str(c.get("img_name", f"cam_{len(cams)}")),
        "width": w,
        "height": h,
        "fx": fx,
        "fy": fy,
        "position": pos,
        "rotation": rot_c2w,
        "viewmat": V,
        "K": K,
    })
  return cams


def make_sharded_rasterizer(
    mesh: Optional[jax.sharding.Mesh],
    width: int,
    height: int,
    sh_degree: int = 3,
    near_plane: float = 0.05,
    far_plane: float = 1e10,
    eps2d: float = 0.30,
    macro_size: int = 32,
    macro_rep: int = 5,
    k_macro: int = 10240,
    k_large_per_chip: int = 1536,
    k_tile: int = 512,
    use_bf16_mxu: bool = False,
    use_approx_topk: bool = True,
    t_grp: Optional[int] = None,
    backgrounds: Optional[jax.Array] = None,
):
  """Build a zero-gather, zero-host-callback scalable 1-chip or 4-chip 3DGS rasterizer.

  Uses two-band exact EWA projection (no covariance clamping):
    - Small/medium splats (`span <= macro_rep` macro-tiles): `macro_rep^2` key-value
      `jax.lax.sort` + `jnp.bincount` + `jax.lax.fori_loop(dynamic_slice)` + `all_to_all`.
    - Large background/table splats (`span > macro_rep` macro-tiles): single-pass
      `jax.lax.sort` + `all_gather` (shared across all macro-tiles without clamping).
  Emits ZERO `gather(` ops in lowered HLO and ZERO `host_callbacks`.

  Returns a JIT-compiled callable:
    `fn(means, quats, scales, opacities, sh_or_colors, viewmat, K) -> (rgb, alpha, depth)`
  """
  n_chips = 1 if mesh is None else len(mesh.devices)
  assert k_macro % n_chips == 0, f"k_macro={k_macro} must be divisible by n_chips={n_chips}"
  assert k_tile % 128 == 0 and k_tile >= 128
  assert macro_size in (32, 64), "macro_size must be 32 or 64"
  k_local = k_macro // n_chips

  H_pad = ((height + macro_size - 1) // macro_size) * macro_size
  W_pad = ((width + macro_size - 1) // macro_size) * macro_size
  My = H_pad // macro_size
  Mx = W_pad // macro_size
  M_tiles = My * Mx
  assert M_tiles < 8191, f"M_tiles={M_tiles} must fit in 13 bits (<8191)"
  M_pad = ((M_tiles + n_chips - 1) // n_chips) * n_chips
  M_sub = M_pad // n_chips

  n_sub_side = macro_size // 8
  T_macro = n_sub_side * n_sub_side  # 16 for 32x32, 64 for 64x64
  t_ox_0 = np.tile(np.arange(n_sub_side, dtype=np.float32) * 8.0, n_sub_side)
  t_oy_0 = np.repeat(np.arange(n_sub_side, dtype=np.float32) * 8.0, n_sub_side)

  t_ox_np = np.zeros((M_pad,), dtype=np.float32)
  t_oy_np = np.zeros((M_pad,), dtype=np.float32)
  t_ox_np[:M_tiles] = np.tile(np.arange(Mx, dtype=np.float32) * float(macro_size), My)
  t_oy_np[:M_tiles] = np.repeat(np.arange(My, dtype=np.float32) * float(macro_size), Mx)

  bg_arr = (
      jnp.asarray(backgrounds, dtype=jnp.float32).reshape(1, 1, 3)
      if backgrounds is not None
      else None
  )

  def _body(means, quats, scales, opacities, sh_or_colors, viewmat, K, tox_sub, toy_sub):
    R_wc = viewmat[:3, :3]
    t_wc = viewmat[:3, 3]
    cam_pos = -(R_wc.T @ t_wc)

    # View-dependent SH (deg 0..3) or direct RGB colors
    if sh_or_colors.ndim == 3:
      dirs = means - cam_pos[None, :]
      dirs = dirs / jnp.maximum(jnp.linalg.norm(dirs, axis=-1, keepdims=True), 1e-12)
      rgb = eval_sh(sh_degree, sh_or_colors.astype(jnp.float32), dirs)
    else:
      rgb = sh_or_colors.astype(jnp.float32)

    R_q = _quats_to_rotmat(quats)
    M_world = R_q * jnp.maximum(scales, 1e-6)[:, None, :]

    p_cam = jnp.einsum("ij,nj->ni", R_wc, means, precision=jax.lax.Precision.HIGHEST) + t_wc[None, :]
    cx, cy, cz = p_cam[:, 0], p_cam[:, 1], p_cam[:, 2]
    z_d = cz
    in_front = (z_d > near_plane) & (z_d < far_plane)
    z_safe = jnp.maximum(z_d, near_plane)

    fx = K[0, 0]
    fy = K[1, 1]
    cx0 = K[0, 2]
    cy0 = K[1, 2]
    mean_x = fx * (cx / z_safe) + cx0
    mean_y = fy * (cy / z_safe) + cy0

    inv_z = 1.0 / z_safe
    inv_z2 = inv_z * inv_z
    lim_x = 1.3 * (0.5 * float(width) / jnp.maximum(fx, 1e-6)) * z_safe
    lim_y = 1.3 * (0.5 * float(height) / jnp.maximum(fy, 1e-6)) * z_safe
    tx_c = jnp.clip(cx, -lim_x, lim_x)
    ty_c = jnp.clip(cy, -lim_y, lim_y)
    j00, j02 = fx * inv_z, -fx * tx_c * inv_z2
    j11, j12 = fy * inv_z, -fy * ty_c * inv_z2

    w0 = j00[:, None] * R_wc[None, 0, :] + j02[:, None] * R_wc[None, 2, :]
    w1 = j11[:, None] * R_wc[None, 1, :] + j12[:, None] * R_wc[None, 2, :]
    a0 = jnp.einsum("nj,njk->nk", w0, M_world, precision=jax.lax.Precision.HIGHEST)
    a1 = jnp.einsum("nj,njk->nk", w1, M_world, precision=jax.lax.Precision.HIGHEST)

    cov2_xx = jnp.sum(a0 * a0, axis=-1) + eps2d
    cov2_xy = jnp.sum(a0 * a1, axis=-1)
    cov2_yy = jnp.sum(a1 * a1, axis=-1) + eps2d

    det = cov2_xx * cov2_yy - cov2_xy * cov2_xy
    valid_g = in_front & (det > 1e-6) & (opacities > (1.0 / 255.0))
    inv_det = jnp.where(valid_g, 1.0 / jnp.maximum(det, 1e-6), 0.0)
    icxx = cov2_yy * inv_det
    icxy = -cov2_xy * inv_det
    icyy = cov2_xx * inv_det

    # Exact support radius where op * exp(-0.5 * r^2) >= 1/255 (clamped to [0.5, 3.33])
    sigma_eff = jnp.minimum(3.33, jnp.sqrt(2.0 * jnp.maximum(jnp.log(255.0 * jnp.maximum(opacities, 1e-6)), 0.1)))
    hx = sigma_eff * jnp.sqrt(jnp.maximum(cov2_xx, 0.1))
    hy = sigma_eff * jnp.sqrt(jnp.maximum(cov2_yy, 0.1))

    on_screen = (
        valid_g
        & (mean_x + hx >= 0.0)
        & (mean_x - hx < float(width))
        & (mean_y + hy >= 0.0)
        & (mean_y - hy < float(height))
    )

    tx0 = jnp.clip(jnp.floor((mean_x - hx) / float(macro_size)).astype(jnp.int32), 0, Mx - 1)
    tx1 = jnp.clip(jnp.floor((mean_x + hx) / float(macro_size)).astype(jnp.int32), 0, Mx - 1)
    ty0 = jnp.clip(jnp.floor((mean_y - hy) / float(macro_size)).astype(jnp.int32), 0, My - 1)
    ty1 = jnp.clip(jnp.floor((mean_y + hy) / float(macro_size)).astype(jnp.int32), 0, My - 1)

    is_large = on_screen & (((tx1 - tx0) >= macro_rep) | ((ty1 - ty0) >= macro_rep))
    is_small = on_screen & (~is_large)

    z_u32 = jax.lax.bitcast_convert_type(jnp.maximum(z_safe, 1e-5), jnp.uint32) >> jnp.uint32(13)
    INV_MACRO = jnp.uint32(8191)

    op_clean = jnp.where(valid_g, jnp.clip(opacities, 0.0, 1.0), 0.0)
    chans = [mean_x, mean_y, icxx, icxy, icyy, rgb[:, 0], rgb[:, 1], rgb[:, 2], op_clean, z_d, hx, hy]

    # Band 1: Small/medium splats (span <= macro_rep)
    def _make_key(v_mask, tx, ty):
      m_id = jnp.where(v_mask, (ty * Mx + tx).astype(jnp.uint32), INV_MACRO)
      return (m_id << jnp.uint32(19)) | z_u32

    key_parts = []
    for dy in range(macro_rep):
      for dx in range(macro_rep):
        vx = is_small if dx == 0 else (is_small & ((tx0 + dx) <= tx1))
        vy = vx if dy == 0 else (vx & ((ty0 + dy) <= ty1))
        key_parts.append(_make_key(vy, jnp.minimum(tx0 + dx, Mx - 1), jnp.minimum(ty0 + dy, My - 1)))

    keys_all = jnp.concatenate(key_parts, axis=0)
    cols_rep = [keys_all] + [jnp.tile(c, macro_rep * macro_rep) for c in chans]

    sorted_cols = jax.lax.sort(cols_rep, dimension=0, num_keys=1)
    m_s = (sorted_cols[0] >> jnp.uint32(19)).astype(jnp.int32)
    cnts = jnp.bincount(m_s, length=8192)[:M_tiles]
    ends = jnp.cumsum(cnts)
    starts = ends - cnts

    s_mat = jnp.stack(sorted_cols[1:], axis=1)  # (rep^2 * N_loc, 12)
    s_pad = jnp.pad(s_mat, ((0, k_local), (0, 0)))

    def _slice_macro(i, acc):
      st = starts[i]
      cnt = jnp.minimum(cnts[i], k_local)
      sl = jax.lax.dynamic_slice(s_pad, (st, 0), (k_local, 12))
      sl = jnp.where((jnp.arange(k_local, dtype=jnp.int32) < cnt)[:, None], sl, 0.0)
      return acc.at[i].set(sl)

    buckets_loc = jax.lax.fori_loop(
        0, M_tiles, _slice_macro, jnp.zeros((M_pad, k_local, 12), dtype=jnp.float32)
    )

    if n_chips > 1:
      g12_send = buckets_loc.reshape(n_chips, M_sub, k_local, 12)
      g12_recv = jax.lax.all_to_all(g12_send, "dp", split_axis=0, concat_axis=0)
      g12 = g12_recv.transpose(1, 0, 2, 3).reshape(M_sub, k_macro, 12)
    else:
      g12 = buckets_loc

    # Band 2: Large splats (span > macro_rep) without covariance clamping
    if k_large_per_chip > 0:
      lg_key = jnp.where(is_large, z_u32, jnp.uint32(0xFFFFFFFF))
      op_lg = jnp.where(is_large, op_clean, 0.0)
      lg_chans = [mean_x, mean_y, icxx, icxy, icyy, rgb[:, 0], rgb[:, 1], rgb[:, 2], op_lg, z_d, hx, hy]
      lg_sorted = jax.lax.sort([lg_key] + lg_chans, dimension=0, num_keys=1)
      k_lg_take = min(k_large_per_chip, means.shape[0])
      lg_loc = jnp.stack([c[:k_lg_take] for c in lg_sorted[1:]], axis=1)  # (k_lg_take, 12)
      if n_chips > 1:
        lg_all = jax.lax.all_gather(lg_loc, "dp", axis=0, tiled=True)  # (n_chips * k_lg_take, 12)
      else:
        lg_all = lg_loc
      lg_bcast = jnp.broadcast_to(lg_all[None, :, :], (M_sub, lg_all.shape[0], 12))
      g12 = jnp.concatenate([g12, lg_bcast], axis=1)

    mx, my, zd, hx_m, hy_m = g12[..., 0], g12[..., 1], g12[..., 9], g12[..., 10], g12[..., 11]
    icxx_m, icxy_m, icyy_m, op_m = g12[..., 2], g12[..., 3], g12[..., 4], g12[..., 8]
    valid_slot = op_m > 0.0

    # Pad 12 -> 16 channels for _pallas_gs3d_v3_fused_rasterize
    g16 = jnp.pad(g12, ((0, 0), (0, 0), (0, 4)))
    splats_b16n = jnp.transpose(g16, (0, 2, 1))  # (M_sub, 16, K_tot)

    t_ox_loc = jnp.asarray(t_ox_0, dtype=jnp.float32)[None, :] + tox_sub[:, None]
    t_oy_loc = jnp.asarray(t_oy_0, dtype=jnp.float32)[None, :] + toy_sub[:, None]

    x0 = t_ox_loc[:, :, None]
    x1 = x0 + 8.0
    y0 = t_oy_loc[:, :, None]
    y1 = y0 + 8.0
    mx_b, my_b = mx[:, None, :], my[:, None, :]

    aabb_ok = (
        valid_slot[:, None, :]
        & (mx_b + hx_m[:, None, :] >= x0)
        & (mx_b - hx_m[:, None, :] <= x1)
        & (my_b + hy_m[:, None, :] >= y0)
        & (my_b - hy_m[:, None, :] <= y1)
    )

    # Exact minimum of Mahalanobis quadratic Q(x, y) over each 8x8 micro-tile box!
    icxx_b = icxx_m[:, None, :]
    icxy_b = icxy_m[:, None, :]
    icyy_b = icyy_m[:, None, :]
    inv_icyy = 1.0 / jnp.maximum(icyy_b, 1e-8)
    inv_icxx = 1.0 / jnp.maximum(icxx_b, 1e-8)

    def _edge_x(xe):
      dx = xe - mx_b
      dy = jnp.clip(my_b - (icxy_b * inv_icyy) * dx, y0, y1) - my_b
      return icxx_b * dx * dx + 2.0 * icxy_b * dx * dy + icyy_b * dy * dy

    def _edge_y(ye):
      dy = ye - my_b
      dx = jnp.clip(mx_b - (icxy_b * inv_icxx) * dy, x0, x1) - mx_b
      return icxx_b * dx * dx + 2.0 * icxy_b * dx * dy + icyy_b * dy * dy

    inside = (mx_b >= x0) & (mx_b <= x1) & (my_b >= y0) & (my_b <= y1)
    q_min = jnp.where(
        inside,
        0.0,
        jnp.minimum(jnp.minimum(_edge_x(x0), _edge_x(x1)), jnp.minimum(_edge_y(y0), _edge_y(y1))),
    )
    max_q = 2.0 * jnp.maximum(jnp.log(255.0 * jnp.maximum(op_m[:, None, :], 1e-6)), 0.0)
    overlap = aabb_ok & (q_min <= max_q)

    score = jnp.where(overlap, -zd[:, None, :], -1.0e9)
    if use_approx_topk:
      top_scores, top_idx = jax.lax.approx_max_k(score, k_tile, recall_target=1.0)
      neg_ts, top_idx = jax.lax.sort_key_val(-top_scores, top_idx, dimension=-1)
      top_scores = -neg_ts
    else:
      top_scores, top_idx = jax.lax.top_k(score, k_tile)
    top_idx = jnp.where(top_scores > -1.0e8, top_idx, -1)

    splats_local = splats_b16n.at[:, 0, :].set(mx - tox_sub[:, None])
    splats_local = splats_local.at[:, 1, :].set(my - toy_sub[:, None])

    k_tot_eff = splats_local.shape[-1]
    tg_use = t_grp if t_grp is not None else (8 if k_tot_eff > 12288 else 16)

    if k_tile == 128:
      rgb_t, dep_t, alp_t = _pallas_gs3d_v3_fused_rasterize(
          splats_local, top_idx, t_ox_0, t_oy_0,
          tile_size=8, use_bf16_mxu=use_bf16_mxu, t_grp=tg_use, interpret=False,
      )
    else:
      rgb_t = jnp.zeros((M_sub, T_macro, 8, 8, 3), dtype=jnp.float32)
      dep_t = jnp.zeros((M_sub, T_macro, 8, 8), dtype=jnp.float32)
      t_acc = jnp.ones((M_sub, T_macro, 8, 8), dtype=jnp.float32)
      for c in range(k_tile // 128):
        rc, dc, ac = _pallas_gs3d_v3_fused_rasterize(
            splats_local, top_idx[..., c * 128 : (c + 1) * 128],
            t_ox_0, t_oy_0, tile_size=8, use_bf16_mxu=use_bf16_mxu, t_grp=tg_use, interpret=False,
        )
        rgb_t = rgb_t + t_acc[..., None] * rc
        dep_t = dep_t + t_acc * dc
        t_acc = t_acc * jnp.maximum(1.0 - ac, 0.0)
      alp_t = 1.0 - t_acc

    rgb_m = rgb_t.reshape(M_sub, n_sub_side, n_sub_side, 8, 8, 3).transpose(0, 1, 3, 2, 4, 5).reshape(M_sub, macro_size, macro_size, 3)
    dep_m = dep_t.reshape(M_sub, n_sub_side, n_sub_side, 8, 8).transpose(0, 1, 3, 2, 4).reshape(M_sub, macro_size, macro_size)
    alp_m = alp_t.reshape(M_sub, n_sub_side, n_sub_side, 8, 8).transpose(0, 1, 3, 2, 4).reshape(M_sub, macro_size, macro_size)
    return rgb_m, dep_m, alp_m

  if n_chips > 1:
    P = jax.sharding.PartitionSpec
    sm_kw = {"check_vma": False} if "check_vma" in inspect.signature(jax.shard_map).parameters else {"check_rep": False}
    _sm_fn = jax.shard_map(
        _body,
        mesh=mesh,
        in_specs=(P("dp"), P("dp"), P("dp"), P("dp"), P("dp"), P(), P(), P("dp"), P("dp")),
        out_specs=(P("dp"), P("dp"), P("dp")),
        **sm_kw,
    )
    t_ox_dev = jax.device_put(jnp.asarray(t_ox_np), jax.sharding.NamedSharding(mesh, P("dp")))
    t_oy_dev = jax.device_put(jnp.asarray(t_oy_np), jax.sharding.NamedSharding(mesh, P("dp")))
  else:
    t_ox_dev = jnp.asarray(t_ox_np)
    t_oy_dev = jnp.asarray(t_oy_np)

  @jax.jit
  def _run(means, quats, scales, opacities, sh_or_colors, viewmat, K):
    if n_chips > 1:
      rgb_m, dep_m, alp_m = _sm_fn(means, quats, scales, opacities, sh_or_colors, viewmat, K, t_ox_dev, t_oy_dev)
    else:
      rgb_m, dep_m, alp_m = _body(means, quats, scales, opacities, sh_or_colors, viewmat, K, t_ox_dev, t_oy_dev)
    rgb_m = rgb_m[:M_tiles]
    dep_m = dep_m[:M_tiles]
    alp_m = alp_m[:M_tiles]
    rgb_img = rgb_m.reshape(My, Mx, macro_size, macro_size, 3).transpose(0, 2, 1, 3, 4).reshape(H_pad, W_pad, 3)[:height, :width, :]
    dep_raw = dep_m.reshape(My, Mx, macro_size, macro_size).transpose(0, 2, 1, 3).reshape(H_pad, W_pad)[:height, :width]
    alp_img = alp_m.reshape(My, Mx, macro_size, macro_size).transpose(0, 2, 1, 3).reshape(H_pad, W_pad)[:height, :width]
    if bg_arr is not None:
      rgb_img = rgb_img + (1.0 - alp_img[..., None]) * bg_arr
    depth_img = jnp.where(alp_img > 1e-4, dep_raw / jnp.maximum(alp_img, 1e-4), -1.0)
    return jnp.clip(rgb_img, 0.0, 1.0), alp_img[..., None], depth_img[..., None]

  return _run


def shard_splats_for_mesh(
    splats: Dict[str, Any],
    mesh: Optional[jax.sharding.Mesh],
    seed: int = 42,
    use_sh: bool = True,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, int]:
  """Shuffle, pad to `n_chips`, and shard loaded `.ply` arrays across `mesh`."""
  n_chips = 1 if mesh is None else len(mesh.devices)
  N = int(splats["n_gaussians"])
  N_pad = ((N + n_chips - 1) // n_chips) * n_chips
  rng = np.random.default_rng(seed)
  perm = rng.permutation(N)

  def _pad_shuf(arr: np.ndarray, fill: float = 0.0) -> np.ndarray:
    a = arr[perm]
    if N_pad > N:
      pad_w = [(0, N_pad - N)] + [(0, 0)] * (a.ndim - 1)
      a = np.pad(a, pad_w, constant_values=fill)
    return np.ascontiguousarray(a, dtype=np.float32)

  means = _pad_shuf(splats["means"])
  quats = _pad_shuf(splats["quats"])
  quats[N:, 0] = 1.0
  scales = _pad_shuf(splats["scales"], fill=1e-4)
  opacities = _pad_shuf(splats["opacities"], fill=0.0)
  col_or_sh = _pad_shuf(splats["sh"] if use_sh else np.clip(_SH_C0 * splats["sh_dc"][:, 0, :] + 0.5, 0.0, 1.0))

  if n_chips > 1 and mesh is not None:
    dp = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("dp"))
    return (
        jax.device_put(jnp.asarray(means), dp),
        jax.device_put(jnp.asarray(quats), dp),
        jax.device_put(jnp.asarray(scales), dp),
        jax.device_put(jnp.asarray(opacities), dp),
        jax.device_put(jnp.asarray(col_or_sh), dp),
        N_pad,
    )
  return (
      jnp.asarray(means),
      jnp.asarray(quats),
      jnp.asarray(scales),
      jnp.asarray(opacities),
      jnp.asarray(col_or_sh),
      N_pad,
  )


def render_ply(
    ply_or_splats: Union[str, Dict[str, Any]],
    viewmat: jax.Array,
    K: jax.Array,
    width: int,
    height: int,
    mesh: Optional[jax.sharding.Mesh] = None,
    backgrounds: Optional[jax.Array] = None,
    macro_size: int = 32,
    macro_rep: int = 5,
    k_macro: int = 10240,
    k_large_per_chip: int = 1536,
    k_tile: int = 512,
    use_bf16_mxu: bool = False,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
  """High-level one-call `.ply` 3DGS renderer supporting SH degree 0..3 and 1-chip/4-chip TPUs."""
  splats = load_ply(ply_or_splats) if isinstance(ply_or_splats, str) else ply_or_splats
  means, quats, scales, opacities, sh, _ = shard_splats_for_mesh(splats, mesh, use_sh=True)
  if mesh is not None and len(mesh.devices) > 1:
    rep = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    viewmat = jax.device_put(jnp.asarray(viewmat, dtype=jnp.float32), rep)
    K = jax.device_put(jnp.asarray(K, dtype=jnp.float32), rep)
  else:
    viewmat = jnp.asarray(viewmat, dtype=jnp.float32)
    K = jnp.asarray(K, dtype=jnp.float32)
  fn = make_sharded_rasterizer(
      mesh=mesh,
      width=width,
      height=height,
      sh_degree=int(splats["sh_degree"]),
      macro_size=macro_size,
      macro_rep=macro_rep,
      k_macro=k_macro,
      k_large_per_chip=k_large_per_chip,
      k_tile=k_tile,
      use_bf16_mxu=use_bf16_mxu,
      backgrounds=backgrounds,
  )
  return fn(means, quats, scales, opacities, sh, viewmat, K)
