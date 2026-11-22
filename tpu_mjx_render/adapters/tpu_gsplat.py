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

import math
from typing import Optional, Tuple

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
