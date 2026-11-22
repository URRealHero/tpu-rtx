"""Fast 4-Chip Data-Parallel Multi-View jax.grad 3DGS Training Benchmark (32 Real Garden Views, 50k-100k Gaussians)."""

from __future__ import annotations

import argparse
import functools
import inspect
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import check_hlo, save_png, scaled_K

from tpu_mjx_render.adapters.tpu_gsplat import (
    _quats_to_rotmat,
    eval_sh,
    load_cameras_json,
    load_ply,
)


def _project_view(
    means: jax.Array,      # (N, 3)
    quats: jax.Array,      # (N, 4)
    scales: jax.Array,     # (N, 3)
    opacities: jax.Array,  # (N,)
    colors: jax.Array,     # (N, 3)
    viewmat: jax.Array,    # (4, 4)
    K: jax.Array,          # (3, 3)
    width: int,
    height: int,
    near_plane: float = 0.05,
    far_plane: float = 1e10,
    eps2d: float = 0.30,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
  """Project N 3D Gaussians for a single camera view -> `(splats_9, z_d, valid_g, hx, hy)`."""
  R_q = _quats_to_rotmat(quats)
  M_world = R_q * jnp.maximum(scales, 1e-6)[:, None, :]

  R_wc = viewmat[:3, :3]
  t_wc = viewmat[:3, 3]
  p_cam = jnp.einsum("ij,nj->ni", R_wc, means, precision=jax.lax.Precision.HIGHEST) + t_wc[None, :]
  cx, cy, cz = p_cam[:, 0], p_cam[:, 1], p_cam[:, 2]
  z_d = cz
  in_front = (z_d > near_plane) & (z_d < far_plane)
  z_safe = jnp.maximum(z_d, near_plane)

  fx, fy = K[0, 0], K[1, 1]
  cx0, cy0 = K[0, 2], K[1, 2]
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

  sigma_eff = jnp.minimum(3.33, jnp.sqrt(2.0 * jnp.maximum(jnp.log(255.0 * jnp.maximum(opacities, 1e-6)), 0.1)))
  hx = sigma_eff * jnp.sqrt(jnp.maximum(cov2_xx, 0.1))
  hy = sigma_eff * jnp.sqrt(jnp.maximum(cov2_yy, 0.1))

  op_clean = jnp.where(valid_g, jnp.clip(opacities, 0.0, 1.0), 0.0)
  splats_9 = jnp.stack(
      [mean_x, mean_y, icxx, icxy, icyy, colors[:, 0], colors[:, 1], colors[:, 2], op_clean],
      axis=-1,
  )  # (N, 9)
  return splats_9, z_d, valid_g, hx, hy


def _make_tile_culled_diff_rasterizer(
    width: int,
    height: int,
    tile_size: int = 16,
    rep: int = 3,
    k_small: int = 192,
    k_large: int = 64,
):
  """Build a zero-gather, zero-host-callback tile-culled differentiable 3DGS rasterizer."""
  assert width % tile_size == 0 and height % tile_size == 0
  Tx = width // tile_size
  Ty = height // tile_size
  T = Tx * Ty
  assert T < 8191
  P_t = tile_size * tile_size
  K_tile = k_small + k_large
  K_chunk = 128
  assert K_tile % K_chunk == 0
  n_chunks = K_tile // K_chunk

  t_ox = np.tile(np.arange(Tx, dtype=np.float32) * float(tile_size), Ty)
  t_oy = np.repeat(np.arange(Ty, dtype=np.float32) * float(tile_size), Tx)
  lx = np.tile(np.arange(tile_size, dtype=np.float32), tile_size) + 0.5
  ly = np.repeat(np.arange(tile_size, dtype=np.float32), tile_size) + 0.5
  px_np = t_ox[:, None] + lx[None, :]  # (T, P_t)
  py_np = t_oy[:, None] + ly[None, :]  # (T, P_t)
  u_np = np.triu(np.ones((K_chunk, K_chunk), dtype=np.float32), k=1)

  def _rasterize_packed(packed_splats: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """Differentiable chunked MXU rasterizer over tile-packed `(T, K_tile, 9)` splats."""
    px = jnp.asarray(px_np, dtype=jnp.float32)[:, :, None]  # (T, P_t, 1)
    py = jnp.asarray(py_np, dtype=jnp.float32)[:, :, None]  # (T, P_t, 1)
    u_mat = jnp.asarray(u_np, dtype=jnp.float32)

    # (T, n_chunks, K_chunk, 9) -> (n_chunks, T, K_chunk, 9)
    chunks = packed_splats.reshape(T, n_chunks, K_chunk, 9).transpose(1, 0, 2, 3)

    @jax.checkpoint
    def _scan_step(carry, ch):
      rgb_acc, t_acc = carry  # (T, P_t, 3), (T, P_t)
      mx = ch[:, None, :, 0]
      my = ch[:, None, :, 1]
      cxx = ch[:, None, :, 2]
      cxy = ch[:, None, :, 3]
      cyy = ch[:, None, :, 4]
      gop = ch[:, None, :, 8]

      dx = px - mx
      dy = py - my
      power = -0.5 * (cxx * dx * dx + 2.0 * cxy * dx * dy + cyy * dy * dy)
      valid_p = (power <= 0.0) & (power > -16.0)
      alpha = jnp.where(valid_p, jnp.minimum(0.99, gop * jnp.exp(jnp.maximum(power, -16.0))), 0.0)

      log_1ma = jnp.log(jnp.maximum(1.0 - alpha, 1.0e-7))
      log_tr = jnp.matmul(log_1ma, u_mat, precision=jax.lax.Precision.HIGHEST)
      trans = jnp.exp(jnp.maximum(log_tr, -30.0))
      w = alpha * trans  # (T, P_t, K_chunk)

      rgb_c = jnp.matmul(w, ch[:, :, 5:8], precision=jax.lax.Precision.HIGHEST)  # (T, P_t, 3)
      rgb_acc = rgb_acc + t_acc[..., None] * rgb_c
      chunk_log_t = jnp.sum(log_1ma, axis=-1)
      t_acc = t_acc * jnp.exp(jnp.maximum(chunk_log_t, -30.0))
      return (rgb_acc, t_acc), None

    init = (jnp.zeros((T, P_t, 3), dtype=jnp.float32), jnp.ones((T, P_t), dtype=jnp.float32))
    (rgb_t, t_final), _ = jax.lax.scan(_scan_step, init, chunks)
    return rgb_t, 1.0 - t_final

  def _pack_view(
      splats_9: jax.Array,  # (N, 9)
      z_d: jax.Array,       # (N,)
      valid_g: jax.Array,   # (N,)
      hx: jax.Array,        # (N,)
      hy: jax.Array,        # (N,)
  ) -> Tuple[jax.Array, jax.Array]:
    """Pack `(N, 9)` splats into `(T, K_tile, 9)` and `packed_idx (T, K_tile)` with zero gathers."""
    N = splats_9.shape[0]
    mx = splats_9[:, 0]
    my = splats_9[:, 1]
    on_screen = (
        valid_g
        & (mx + hx >= 0.0)
        & (mx - hx < float(width))
        & (my + hy >= 0.0)
        & (my - hy < float(height))
    )
    tx0 = jnp.clip(jnp.floor((mx - hx) / float(tile_size)).astype(jnp.int32), 0, Tx - 1)
    tx1 = jnp.clip(jnp.floor((mx + hx) / float(tile_size)).astype(jnp.int32), 0, Tx - 1)
    ty0 = jnp.clip(jnp.floor((my - hy) / float(tile_size)).astype(jnp.int32), 0, Ty - 1)
    ty1 = jnp.clip(jnp.floor((my + hy) / float(tile_size)).astype(jnp.int32), 0, Ty - 1)

    is_large = on_screen & (((tx1 - tx0) >= rep) | ((ty1 - ty0) >= rep))
    is_small = on_screen & (~is_large)

    z_safe = jnp.maximum(z_d, 1e-5)
    z_u32 = jax.lax.bitcast_convert_type(z_safe, jnp.uint32) >> jnp.uint32(13)
    INV_TILE = jnp.uint32(8191)

    def _make_key(v_mask, tx, ty):
      tid = jnp.where(v_mask, (ty * Tx + tx).astype(jnp.uint32), INV_TILE)
      return (tid << jnp.uint32(19)) | z_u32

    key_parts = []
    for dy in range(rep):
      for dx in range(rep):
        vx = is_small if dx == 0 else (is_small & ((tx0 + dx) <= tx1))
        vy = vx if dy == 0 else (vx & ((ty0 + dy) <= ty1))
        key_parts.append(_make_key(vy, jnp.minimum(tx0 + dx, Tx - 1), jnp.minimum(ty0 + dy, Ty - 1)))

    keys_all = jnp.concatenate(key_parts, axis=0)
    g_idx = jnp.arange(N, dtype=jnp.int32)
    idx_rep = jnp.tile(g_idx, rep * rep)
    z_rep = jnp.tile(z_d, rep * rep)
    cols = [keys_all, idx_rep, z_rep] + [jnp.tile(splats_9[:, c], rep * rep) for c in range(9)]

    sorted_cols = jax.lax.sort(cols, dimension=0, num_keys=1)
    t_s = (sorted_cols[0] >> jnp.uint32(19)).astype(jnp.int32)
    cnts = jnp.bincount(t_s, length=8192)[:T]
    ends = jnp.cumsum(cnts)
    starts = ends - cnts

    s_idx = sorted_cols[1]
    s_zd = sorted_cols[2]
    s_spl = jnp.stack(sorted_cols[3:], axis=1)  # (rep^2 * N, 9)

    s_idx_pad = jnp.pad(s_idx, (0, k_small), constant_values=-1)
    s_zd_pad = jnp.pad(s_zd, (0, k_small), constant_values=1.0e9)
    s_spl_pad = jnp.pad(s_spl, ((0, k_small), (0, 0)))

    def _slice_tile(i, carry):
      acc_spl, acc_idx, acc_zd = carry
      st = starts[i]
      cnt = jnp.minimum(cnts[i], k_small)
      valid_slot = jnp.arange(k_small, dtype=jnp.int32) < cnt

      sl_spl = jax.lax.dynamic_slice(s_spl_pad, (st, 0), (k_small, 9))
      sl_idx = jax.lax.dynamic_slice(s_idx_pad, (st,), (k_small,))
      sl_zd = jax.lax.dynamic_slice(s_zd_pad, (st,), (k_small,))

      acc_spl = acc_spl.at[i].set(jnp.where(valid_slot[:, None], sl_spl, 0.0))
      acc_idx = acc_idx.at[i].set(jnp.where(valid_slot, sl_idx, -1))
      acc_zd = acc_zd.at[i].set(jnp.where(valid_slot, sl_zd, 1.0e9))
      return acc_spl, acc_idx, acc_zd

    sm_spl, sm_idx, sm_zd = jax.lax.fori_loop(
        0,
        T,
        _slice_tile,
        (
            jnp.zeros((T, k_small, 9), dtype=jnp.float32),
            jnp.full((T, k_small), -1, dtype=jnp.int32),
            jnp.full((T, k_small), 1.0e9, dtype=jnp.float32),
        ),
    )

    if k_large > 0:
      lg_key = jnp.where(is_large, z_u32, jnp.uint32(0xFFFFFFFF))
      lg_idx = jnp.where(is_large, g_idx, -1)
      lg_zd = jnp.where(is_large, z_d, 1.0e9)
      lg_spl = jnp.where(is_large[:, None], splats_9, 0.0)
      lg_hx = jnp.where(is_large, hx, -1.0)
      lg_hy = jnp.where(is_large, hy, -1.0)
      lg_cols = [lg_key, lg_idx, lg_zd, lg_hx, lg_hy] + [lg_spl[:, c] for c in range(9)]
      lg_sorted = jax.lax.sort(lg_cols, dimension=0, num_keys=1)

      k_lg_take = min(k_large, N)
      li = lg_sorted[1][:k_lg_take]
      lz = lg_sorted[2][:k_lg_take]
      lhx = lg_sorted[3][:k_lg_take]
      lhy = lg_sorted[4][:k_lg_take]
      ls = jnp.stack([c[:k_lg_take] for c in lg_sorted[5:]], axis=1)  # (k_lg_take, 9)

      tox_d = jnp.asarray(t_ox, dtype=jnp.float32)[:, None]
      toy_d = jnp.asarray(t_oy, dtype=jnp.float32)[:, None]
      aabb_lg = (
          (li[None, :] >= 0)
          & (ls[None, :, 0] + lhx[None, :] >= tox_d)
          & (ls[None, :, 0] - lhx[None, :] <= tox_d + float(tile_size))
          & (ls[None, :, 1] + lhy[None, :] >= toy_d)
          & (ls[None, :, 1] - lhy[None, :] <= toy_d + float(tile_size))
      )
      lg_spl_t = jnp.where(aabb_lg[:, :, None], ls[None, :, :], 0.0)
      lg_idx_t = jnp.where(aabb_lg, li[None, :], -1)
      lg_zd_t = jnp.where(aabb_lg, lz[None, :], 1.0e9)
      if k_lg_take < k_large:
        pad_n = k_large - k_lg_take
        lg_spl_t = jnp.pad(lg_spl_t, ((0, 0), (0, pad_n), (0, 0)))
        lg_idx_t = jnp.pad(lg_idx_t, ((0, 0), (0, pad_n)), constant_values=-1)
        lg_zd_t = jnp.pad(lg_zd_t, ((0, 0), (0, pad_n)), constant_values=1.0e9)

      all_spl = jnp.concatenate([sm_spl, lg_spl_t], axis=1)  # (T, K_tile, 9)
      all_idx = jnp.concatenate([sm_idx, lg_idx_t], axis=1)  # (T, K_tile)
      all_zd = jnp.concatenate([sm_zd, lg_zd_t], axis=1)     # (T, K_tile)
      # Sort each tile front-to-back across small + large splats
      sort_in = [all_zd, all_idx] + [all_spl[..., c] for c in range(9)]
      sort_out = jax.lax.sort(sort_in, dimension=1, num_keys=1)
      packed_idx = sort_out[1]
      packed_spl = jnp.stack(sort_out[2:], axis=-1)
    else:
      packed_idx = sm_idx
      packed_spl = sm_spl

    return packed_spl, packed_idx

  @jax.custom_vjp
  def _tile_raster_core(splats_9, z_d, valid_g, hx, hy):
    packed_spl, _ = _pack_view(splats_9, z_d, valid_g, hx, hy)
    return _rasterize_packed(packed_spl)

  def _fwd(splats_9, z_d, valid_g, hx, hy):
    packed_spl, packed_idx = _pack_view(
        jax.lax.stop_gradient(splats_9),
        jax.lax.stop_gradient(z_d),
        jax.lax.stop_gradient(valid_g),
        jax.lax.stop_gradient(hx),
        jax.lax.stop_gradient(hy),
    )
    out = _rasterize_packed(packed_spl)
    return out, (packed_spl, packed_idx, splats_9.shape[0])

  def _bwd(res, g_out):
    packed_spl, packed_idx, n_splats = res
    _, vjp_fn = jax.vjp(_rasterize_packed, packed_spl)
    (g_packed,) = vjp_fn(g_out)
    safe_idx = jnp.where(packed_idx >= 0, packed_idx, n_splats)
    g_splats = (
        jnp.zeros((n_splats + 1, 9), dtype=jnp.float32)
        .at[safe_idx.reshape(-1)]
        .add(g_packed.reshape(-1, 9))[:n_splats]
    )
    return (g_splats, None, None, None, None)

  _tile_raster_core.defvjp(_fwd, _bwd)

  def _render_single_view(means, quats, scales, opacities, colors, viewmat, K, bg):
    splats_9, z_d, valid_g, hx, hy = _project_view(
        means, quats, scales, opacities, colors, viewmat, K, width=width, height=height
    )
    rgb_t, alp_t = _tile_raster_core(splats_9, z_d, valid_g, hx, hy)
    rgb_img = (
        rgb_t.reshape(Ty, Tx, tile_size, tile_size, 3)
        .transpose(0, 2, 1, 3, 4)
        .reshape(height, width, 3)
    )
    alp_img = (
        alp_t.reshape(Ty, Tx, tile_size, tile_size)
        .transpose(0, 2, 1, 3)
        .reshape(height, width)
    )
    return rgb_img + (1.0 - alp_img[..., None]) * bg[None, None, :]

  return _render_single_view


def _select_garden_subset(sp: Dict[str, Any], cams: List[Dict[str, Any]], n_gaussians: int) -> Dict[str, np.ndarray]:
  """Select N real Mip-NeRF 360 Garden Gaussians weighted around the 32-camera focus core."""
  A_foc = np.zeros((3, 3), dtype=np.float64)
  b_foc = np.zeros((3,), dtype=np.float64)
  for c in cams:
    d_ax = c["rotation"][:, 2].astype(np.float64)
    P_proj = np.eye(3) - np.outer(d_ax, d_ax)
    A_foc += P_proj
    b_foc += P_proj @ c["position"].astype(np.float64)
  focus_ctr = np.linalg.solve(A_foc, b_foc).astype(np.float32)

  dist_foc = np.linalg.norm(sp["means"] - focus_ctr[None, :], axis=-1)
  max_scale = np.max(sp["scales"], axis=-1)
  core_mask = (dist_foc < 8.5) & (sp["opacities"] > 0.05) & (max_scale < 0.35)
  core_idx = np.where(core_mask)[0]
  if len(core_idx) < n_gaussians:
    core_idx = np.argsort(dist_foc)[:n_gaussians]

  # Importance-rank by opacity / (0.6 + dist_to_focus) so centerpiece + surrounding yard are dense
  score = sp["opacities"][core_idx] / (0.6 + dist_foc[core_idx])
  top_idx = core_idx[np.argsort(-score)[:n_gaussians]]

  # Compute mean view-dependent SH3 RGB across the 32 cameras for crisp real-world GT appearance
  cam_pos_c = np.stack([c["position"] for c in cams], axis=0)
  cam_mean = cam_pos_c.mean(axis=0)
  dirs = sp["means"][top_idx] - cam_mean[None, :]
  dirs = dirs / np.maximum(np.linalg.norm(dirs, axis=-1, keepdims=True), 1e-12)
  gt_colors = np.asarray(eval_sh(3, jnp.asarray(sp["sh"][top_idx]), jnp.asarray(dirs)))
  gt_colors = np.clip(gt_colors, 0.0, 1.0).astype(np.float32)

  return {
      "focus_ctr": focus_ctr,
      "means": np.ascontiguousarray(sp["means"][top_idx], dtype=np.float32),
      "quats": np.ascontiguousarray(sp["quats"][top_idx], dtype=np.float32),
      "scales": np.ascontiguousarray(sp["scales"][top_idx], dtype=np.float32),
      "opacities": np.ascontiguousarray(sp["opacities"][top_idx], dtype=np.float32),
      "colors": np.ascontiguousarray(gt_colors, dtype=np.float32),
  }


def run_single_scale_grad_bench(
    sp: Dict[str, Any],
    cams_32: List[Dict[str, Any]],
    n_gaussians: int,
    mesh: jax.sharding.Mesh,
    W: int = 128,
    H: int = 128,
    n_steps: int = 350,
    seed: int = 2026,
) -> Tuple[Dict[str, Any], Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
  """Benchmark 4-chip data-parallel jax.grad 3DGS training across 32 real Garden views."""
  C_total = len(cams_32)
  assert C_total == 32
  n_chips = len(mesh.devices)
  assert n_chips == 4 and C_total % n_chips == 0
  C_loc = C_total // n_chips  # 8 views/chip

  sub = _select_garden_subset(sp, cams_32, n_gaussians)
  Vs = np.stack([c["viewmat"] for c in cams_32], axis=0)
  Ks = np.stack([scaled_K(c, W, H) for c in cams_32], axis=0)

  dp = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("dp"))
  rep = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
  P = jax.sharding.PartitionSpec
  sm_kw = {"check_vma": False} if "check_vma" in inspect.signature(jax.shard_map).parameters else {"check_rep": False}

  render_view_fn = _make_tile_culled_diff_rasterizer(
      width=W, height=H, tile_size=16, rep=3, k_small=192, k_large=64
  )
  bg_np = np.array([0.08, 0.10, 0.14], dtype=np.float32)

  def _render_views(means, quats, scales, opacities, colors, vms, kms, bg):
    return jax.vmap(
        lambda vm, km: render_view_fn(means, quats, scales, opacities, colors, vm, km, bg)
    )(vms, kms)

  render_sharded = jax.jit(
      jax.shard_map(
          _render_views,
          mesh=mesh,
          in_specs=(P(), P(), P(), P(), P(), P("dp"), P("dp"), P()),
          out_specs=P("dp"),
          **sm_kw,
      )
  )

  vms_dev = jax.device_put(jnp.asarray(Vs), dp)
  kms_dev = jax.device_put(jnp.asarray(Ks), dp)
  bg_dev = jax.device_put(jnp.asarray(bg_np), rep)

  gt_rgb_dev = render_sharded(
      jax.device_put(jnp.asarray(sub["means"]), rep),
      jax.device_put(jnp.asarray(sub["quats"]), rep),
      jax.device_put(jnp.asarray(sub["scales"]), rep),
      jax.device_put(jnp.asarray(sub["opacities"]), rep),
      jax.device_put(jnp.asarray(sub["colors"]), rep),
      vms_dev,
      kms_dev,
      bg_dev,
  )
  jax.block_until_ready(gt_rgb_dev)

  # Perturbed initialization around Garden geometry (noisy positions, scales, opacities, colors)
  rng = np.random.default_rng(seed)
  init_means = sub["means"] + rng.normal(0.0, 0.035, size=(n_gaussians, 3)).astype(np.float32)
  init_quats = rng.normal(0.0, 1.0, size=(n_gaussians, 4)).astype(np.float32)
  init_quats /= np.maximum(np.linalg.norm(init_quats, axis=-1, keepdims=True), 1e-8)
  init_log_scales = np.log(
      np.clip(sub["scales"] * rng.uniform(0.75, 1.35, size=(n_gaussians, 3)), 1e-4, 0.35)
  ).astype(np.float32)
  init_logit_op = np.zeros((n_gaussians,), dtype=np.float32)
  init_logit_col = rng.normal(0.0, 0.35, size=(n_gaussians, 3)).astype(np.float32)

  params = {
      "means": jax.device_put(jnp.asarray(init_means), rep),
      "quats": jax.device_put(jnp.asarray(init_quats), rep),
      "log_scales": jax.device_put(jnp.asarray(init_log_scales), rep),
      "logit_opacities": jax.device_put(jnp.asarray(init_logit_op), rep),
      "logit_colors": jax.device_put(jnp.asarray(init_logit_col), rep),
  }
  m_state = jax.tree.map(lambda x: jax.device_put(jnp.zeros_like(x), rep), params)
  v_state = jax.tree.map(lambda x: jax.device_put(jnp.zeros_like(x), rep), params)

  init_rgb_dev = render_sharded(
      params["means"],
      params["quats"],
      jnp.exp(jnp.clip(params["log_scales"], -9.5, -1.0)),
      jax.nn.sigmoid(params["logit_opacities"]),
      jax.nn.sigmoid(params["logit_colors"]),
      vms_dev,
      kms_dev,
      bg_dev,
  )
  jax.block_until_ready(init_rgb_dev)
  init_mse = float(jnp.mean((init_rgb_dev - gt_rgb_dev) ** 2))
  init_psnr = 10.0 * math.log10(1.0 / max(init_mse, 1e-12))
  print(f"[N={n_gaussians:,}] Init PSNR across 32 Garden views: {init_psnr:.2f} dB", flush=True)

  lr_base = {
      "means": 0.009,
      "quats": 0.025,
      "log_scales": 0.050,
      "logit_opacities": 0.100,
      "logit_colors": 0.160,
  }

  def _dp_grad_step(p, m_s, v_s, step_idx, vms_loc, kms_loc, gt_loc, bg):
    def _local_loss(pr):
      pred = _render_views(
          pr["means"],
          pr["quats"],
          jnp.exp(jnp.clip(pr["log_scales"], -9.5, -1.0)),
          jax.nn.sigmoid(pr["logit_opacities"]),
          jax.nn.sigmoid(pr["logit_colors"]),
          vms_loc,
          kms_loc,
          bg,
      )
      l1 = jnp.mean(jnp.abs(pred - gt_loc))
      mse = jnp.mean((pred - gt_loc) ** 2)
      return 0.8 * l1 + 0.2 * mse, mse

    (loss_val, mse_val), grads = jax.value_and_grad(_local_loss, has_aux=True)(p)
    # 4-chip data-parallel gradient & metric sync across the 32 views (8 views/chip)
    grads = jax.lax.pmean(grads, "dp")
    loss_val = jax.lax.pmean(loss_val, "dp")
    mse_val = jax.lax.pmean(mse_val, "dp")

    b1, b2, eps = 0.9, 0.999, 1e-8
    t_f = step_idx.astype(jnp.float32)
    decay = jnp.power(0.10, t_f / float(n_steps))
    new_p, new_m, new_v = {}, {}, {}
    for k in p:
      g = grads[k]
      mk = b1 * m_s[k] + (1.0 - b1) * g
      vk = b2 * v_s[k] + (1.0 - b2) * (g * g)
      m_hat = mk / (1.0 - (b1 ** t_f))
      v_hat = vk / (1.0 - (b2 ** t_f))
      new_p[k] = p[k] - (lr_base[k] * decay) * m_hat / (jnp.sqrt(v_hat) + eps)
      new_m[k] = mk
      new_v[k] = vk
    return new_p, new_m, new_v, loss_val, mse_val

  train_step_fn = jax.jit(
      jax.shard_map(
          _dp_grad_step,
          mesh=mesh,
          in_specs=(P(), P(), P(), P(), P("dp"), P("dp"), P("dp"), P()),
          out_specs=(P(), P(), P(), P(), P()),
          **sm_kw,
      )
  )

  step1_d = jax.device_put(jnp.array(1, jnp.int32), rep)
  low = train_step_fn.lower(params, m_state, v_state, step1_d, vms_dev, kms_dev, gt_rgb_dev, bg_dev)
  hlo_info = check_hlo(low, f"garden_32v_grad_{n_gaussians}")

  t_c = time.perf_counter()
  comp = low.compile()
  params, m_state, v_state, loss0, mse0 = comp(
      params, m_state, v_state, step1_d, vms_dev, kms_dev, gt_rgb_dev, bg_dev
  )
  jax.block_until_ready((params, loss0))
  compile_s = time.perf_counter() - t_c
  print(f"[N={n_gaussians:,}] Compiled 4-chip 32-view DP grad step in {compile_s:.2f}s", flush=True)

  # Warmup 2 steps before throughput timing
  for st in (2, 3):
    params, m_state, v_state, loss_v, mse_v = comp(
        params, m_state, v_state, jax.device_put(jnp.array(st, jnp.int32), rep),
        vms_dev, kms_dev, gt_rgb_dev, bg_dev,
    )
  jax.block_until_ready((params, loss_v))

  history: List[Dict[str, Any]] = [{"step": 0, "wall_s": 0.0, "psnr_db": round(init_psnr, 2), "mse": round(init_mse, 6)}]
  milestones: Dict[str, Any] = {}
  t_train_start = time.perf_counter()
  pure_compute_s = 0.0
  pure_compute_steps = 0

  # Run in blocks of 25 steps (pure uninterrupted dispatch inside each block!)
  block_size = 25
  st_cur = 4
  while st_cur <= n_steps:
    st_end = min(n_steps, st_cur + block_size - 1)
    n_blk = st_end - st_cur + 1
    t_b0 = time.perf_counter()
    for st in range(st_cur, st_end + 1):
      params, m_state, v_state, loss_v, mse_v = comp(
          params, m_state, v_state, jax.device_put(jnp.array(st, jnp.int32), rep),
          vms_dev, kms_dev, gt_rgb_dev, bg_dev,
      )
    jax.block_until_ready((loss_v, mse_v))
    t_b1 = time.perf_counter()
    pure_compute_s += (t_b1 - t_b0)
    pure_compute_steps += n_blk

    wall_s = t_b1 - t_train_start
    psnr_v = 10.0 * math.log10(1.0 / max(float(mse_v), 1e-12))
    rec_h = {
        "step": st_end,
        "wall_s": round(wall_s, 3),
        "psnr_db": round(psnr_v, 2),
        "loss": round(float(loss_v), 5),
    }
    history.append(rec_h)
    for thr in (28.0, 30.0, 32.0):
      key = f"time_to_{int(thr)}db"
      if key not in milestones and psnr_v >= thr:
        milestones[key] = {"step": st_end, "wall_s": round(wall_s, 3), "psnr_db": round(psnr_v, 2)}

    if st_end in (25, 50, 100, 150, 200, 300, 400, 500, 600) or st_end == n_steps:
      print(
          f"  [N={n_gaussians:,}] step {st_end:3d}/{n_steps} ({wall_s:5.2f}s): "
          f"loss={float(loss_v):.4f}, PSNR={psnr_v:.2f} dB",
          flush=True,
      )
    st_cur = st_end + 1

  jax.block_until_ready(params)
  total_wall_s = time.perf_counter() - t_train_start
  steps_per_s = pure_compute_steps / pure_compute_s
  ms_per_step = 1000.0 / steps_per_s

  final_rgb_dev = render_sharded(
      params["means"],
      params["quats"],
      jnp.exp(jnp.clip(params["log_scales"], -9.5, -1.0)),
      jax.nn.sigmoid(params["logit_opacities"]),
      jax.nn.sigmoid(params["logit_colors"]),
      vms_dev,
      kms_dev,
      bg_dev,
  )
  jax.block_until_ready(final_rgb_dev)
  final_mse = float(jnp.mean((final_rgb_dev - gt_rgb_dev) ** 2))
  final_psnr = 10.0 * math.log10(1.0 / max(final_mse, 1e-12))

  metrics = {
      "scene": "MipNeRF360_Garden_32Views",
      "n_views_total": C_total,
      "n_views_per_chip": C_loc,
      "n_chips": n_chips,
      "data_parallel": True,
      "res": f"{W}x{H}",
      "n_gaussians": int(n_gaussians),
      "n_steps": int(n_steps),
      "compile_s": round(compile_s, 2),
      "total_train_s": round(total_wall_s, 2),
      "steps_per_s": round(steps_per_s, 2),
      "ms_per_step": round(ms_per_step, 2),
      "view_steps_per_s": round(steps_per_s * C_total, 1),
      "init_psnr_db": round(init_psnr, 2),
      "final_psnr_db": round(final_psnr, 2),
      "psnr_gain_db": round(final_psnr - init_psnr, 2),
      "milestones": milestones,
      "history": history,
  }
  print(
      f"[N={n_gaussians:,}] DONE: {init_psnr:.2f} -> {final_psnr:.2f} dB | "
      f"{steps_per_s:.2f} steps/s ({ms_per_step:.2f} ms/step, {steps_per_s * C_total:.1f} view-steps/s) | "
      f"milestones={milestones}",
      flush=True,
  )
  return metrics, hlo_info, np.asarray(init_rgb_dev), np.asarray(final_rgb_dev), np.asarray(gt_rgb_dev)


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--data_root", default="/dev/shm/zaychik-w3-v7/data")
  ap.add_argument("--out_json", default="/dev/shm/zaychik-w3-v7/out/v3_w3_wave7_8k_and_gradtrain.json")
  ap.add_argument("--export_dir", default="/dev/shm/zaychik-w3-v7/out")
  ap.add_argument("--counts", nargs="+", type=int, default=[50000, 100000])
  ap.add_argument("--n_steps", type=int, default=350)
  ap.add_argument("--res", type=int, default=128)
  args = ap.parse_args()

  os.makedirs(args.export_dir, exist_ok=True)
  os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)

  devs = jax.devices()[:4]
  mesh = jax.sharding.Mesh(np.array(devs), ("dp",))

  ply_p = os.path.join(args.data_root, "garden/point_cloud/iteration_30000/point_cloud.ply")
  cam_p = os.path.join(args.data_root, "garden/cameras.json")
  sp = load_ply(ply_p, max_sh_degree=3)
  cams_all = load_cameras_json(cam_p)
  idx_32 = np.linspace(0, len(cams_all) - 1, 32, dtype=int)
  cams_32 = [cams_all[i] for i in idx_32]

  if os.path.exists(args.out_json):
    with open(args.out_json) as f:
      report = json.load(f)
  else:
    report = {
        "backend": jax.default_backend(),
        "device_kind": jax.devices()[0].device_kind,
        "jax_version": jax.__version__,
        "interpret": False,
        "hlo_checks": {},
    }

  runs = []
  last_vis = None
  for n_g in args.counts:
    m, hlo, init_np, fin_np, gt_np = run_single_scale_grad_bench(
        sp, cams_32, n_gaussians=n_g, mesh=mesh, W=args.res, H=args.res, n_steps=args.n_steps
    )
    runs.append(m)
    report.setdefault("hlo_checks", {})[f"garden_32v_grad_{n_g}"] = hlo
    last_vis = (n_g, init_np, fin_np, gt_np)

  report["fastgrad_32views_garden"] = runs
  if last_vis is not None:
    n_g, init_np, fin_np, gt_np = last_vis
    rows = []
    for vi in (0, 8, 16, 24):
      rows.append(np.concatenate([init_np[vi], fin_np[vi], gt_np[vi]], axis=1))
    grid = np.concatenate(rows, axis=0)
    vis_path = os.path.join(args.export_dir, f"v3_w3_garden_32v_grad_{n_g}.png")
    save_png(grid, vis_path)
    report["fastgrad_visual_path"] = vis_path

  with open(args.out_json, "w") as f:
    json.dump(report, f, indent=2)
  print(f"=== FastGrad 32-View Benchmark Done -> {args.out_json} ===", flush=True)


if __name__ == "__main__":
  main()
