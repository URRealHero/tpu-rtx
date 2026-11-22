"""OptiX-Equivalent Python/JAX Ray Tracing SDK on TPU v5p (Wide-BVH-64 + 2-Level TLAS + jax.grad).

Provides a clean OptiX-style programming model natively backed by TPU Mosaic Pallas kernels
with zero host callbacks (`host_callbacks == 0`, `send_recv_ops == 0`):
  - `OptixDeviceContext`: TPU mesh/sharding context, HLO zero-copy auditing, launch metrics.
  - `OptixBuildInputTriangleArray` / `OptixBuildInputInstanceArray`:
      `accelBuild(GAS=Wide-BVH-64, IAS=2-Level TLAS)`
  - `ShaderBindingTable(raygen, miss, closesthit, anyhit)`: pure-JAX programmable shader stages.
  - `optixTrace(...)` & `optixLaunch(...)`: sharded multi-chip execution with custom VJP for
      end-to-end `jax.grad` differentiable path tracing & inverse rendering.
"""

from __future__ import annotations

import dataclasses
from functools import partial
import inspect
import math
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np

from tpu_mjx_render.primitives import MJ_MAXVAL
from tpu_mjx_render.roofline import inspect_hlo_ops


# ==============================================================================
# Constants & HLO Zero-Copy Audit
# ==============================================================================

LEAF_SIZE = 128
BRANCH_FACTOR = 64
BLOCK_R = 128
TILE_H = 8
TILE_W = 16

OPTIX_RAY_FLAG_NONE = 0
OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT = 1


def audit_hlo_zero_copy(lowered: Any) -> Dict[str, int]:
  """Verify lowered XLA HLO has zero host callbacks and zero send/recv ops."""
  txt = lowered.as_text()
  info = inspect_hlo_ops(txt)
  info["send_recv_ops"] = len(re.findall(r"\bsend\(|\brecv\(", txt, re.I))
  assert info["host_callbacks"] == 0, f"HLO contains host callbacks: {info}"
  assert info["send_recv_ops"] == 0, f"HLO contains send/recv ops: {info}"
  return info


# ==============================================================================
# Wide-BVH-64 Builder Utilities (Self-Contained, No External Experiment Imports)
# ==============================================================================

def _compute_smooth_vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
  v0 = verts[faces[:, 0]]
  v1 = verts[faces[:, 1]]
  v2 = verts[faces[:, 2]]
  fn = np.cross(v1 - v0, v2 - v0)
  n_v = verts.shape[0]
  vn = np.empty((n_v, 3), dtype=np.float32)
  f_flat = faces.ravel()
  for c in range(3):
    vn[:, c] = np.bincount(f_flat, weights=np.repeat(fn[:, c], 3), minlength=n_v)
  norms = np.linalg.norm(vn, axis=-1, keepdims=True)
  bad = norms[:, 0] < 1e-12
  if np.any(bad):
    vn[bad] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    norms = np.linalg.norm(vn, axis=-1, keepdims=True)
  return (vn / np.maximum(norms, 1e-12)).astype(np.float32)


def _spatial_median_order(
    centroids: np.ndarray,
    leaf_size: int = LEAF_SIZE,
    branch_factor: int = BRANCH_FACTOR,
    stop_at_l1: bool = True,
) -> np.ndarray:
  F = int(centroids.shape[0])
  idx = np.arange(F, dtype=np.int32)
  if F <= leaf_size:
    return idx

  def _split_recursive(sub_idx: np.ndarray, target_chunk: int) -> None:
    n = sub_idx.shape[0]
    if n <= target_chunk:
      return
    n_blocks = (n + target_chunk - 1) // target_chunk
    left_blocks = 1 << ((n_blocks - 1).bit_length() - 1)
    if left_blocks <= 0:
      left_blocks = n_blocks // 2
    mid = min(left_blocks * target_chunk, n - 1)
    pts = centroids[sub_idx]
    span = np.max(pts, axis=0) - np.min(pts, axis=0)
    axis = int(np.argmax(span))
    part = np.argpartition(pts[:, axis], mid)
    sub_idx[:] = sub_idx[part]
    _split_recursive(sub_idx[:mid], target_chunk)
    _split_recursive(sub_idx[mid:], target_chunk)

  if stop_at_l1:
    _split_recursive(idx, leaf_size * branch_factor)
    n_l1 = (F + (leaf_size * branch_factor) - 1) // (leaf_size * branch_factor)
    for b in range(n_l1):
      s = b * leaf_size * branch_factor
      e = min((b + 1) * leaf_size * branch_factor, F)
      _split_recursive(idx[s:e], leaf_size)
  else:
    _split_recursive(idx, leaf_size)
  return idx


def _pack_rays_tiled(
    ro: jax.Array, rd: jax.Array, tl: jax.Array, H: int, W: int
) -> Tuple[jax.Array, int, bool]:
  B, R, _ = ro.shape
  can_2d_tile = (H % TILE_H == 0) and (W % TILE_W == 0) and (R == H * W)
  if can_2d_tile:
    ty, tx = H // TILE_H, W // TILE_W
    n_tiles = ty * tx

    def _to_tiles(x):
      c = x.shape[-1]
      return (
          x.reshape(B, ty, TILE_H, tx, TILE_W, c)
          .transpose(0, 1, 3, 2, 4, 5)
          .reshape(B * n_tiles, BLOCK_R, c)
      )

    ro_t = _to_tiles(ro)
    rd_t = _to_tiles(rd)
    tl_t = _to_tiles(tl[..., None])
  else:
    R_pad = ((R + BLOCK_R - 1) // BLOCK_R) * BLOCK_R
    n_tiles = R_pad // BLOCK_R
    pad_n = R_pad - R
    if pad_n > 0:
      ro = jnp.pad(ro, ((0, 0), (0, pad_n), (0, 0)))
      rd = jnp.pad(rd, ((0, 0), (0, pad_n), (0, 0)))
      tl = jnp.pad(tl, ((0, 0), (0, pad_n)))
    ro_t = ro.reshape(B * n_tiles, BLOCK_R, 3)
    rd_t = rd.reshape(B * n_tiles, BLOCK_R, 3)
    tl_t = tl.reshape(B * n_tiles, BLOCK_R, 1)

  z = jnp.zeros_like(tl_t)
  packed = jnp.concatenate([ro_t, rd_t, tl_t, z], axis=-1)  # (B*n_tiles, 128, 8)
  rays_2d = packed.transpose(0, 2, 1).reshape(B * n_tiles * 8, BLOCK_R)
  return rays_2d, n_tiles, can_2d_tile


# ==============================================================================
# Core Pallas Wide-BVH-64 Traversal + Custom VJP for jax.grad
# ==============================================================================

def _pallas_gas_raycast_raw(
    l2_roots_dev: jax.Array,
    l2_dev: jax.Array,
    l0_dev: jax.Array,
    leaf_dev: jax.Array,
    ro: jax.Array,      # (B, R, 3)
    rd: jax.Array,      # (B, R, 3)
    t_limit: jax.Array, # (B, R)
    H: int,
    W: int,
    any_hit: bool = False,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
  """Zero-copy Mosaic Wide-BVH-64 traversal returning (dist, nrm, albedo, roughness, mat_code)."""
  B, R, _ = ro.shape
  rays_2d, n_tiles, can_2d_tile = _pack_rays_tiled(ro, rd, t_limit, H, W)
  total_tiles = B * n_tiles
  leaf_rows = 16 if any_hit else 24

  def _bvh_kernel(
      ray_ref, l2_roots_ref, l2_hbm_ref, l0_hbm_ref, leaf_hbm_ref, out_ref,
      l2_vmem, l0_vmem, leaf_vmem, sem_l2, sem_l0, sem_leaf,
  ):
    cols_iota_64 = jax.lax.broadcasted_iota(jnp.int32, (1, BRANCH_FACTOR), 1)
    rt = ray_ref[...]
    l2_roots_blk = l2_roots_ref[...]

    rox = jnp.transpose(rt[0:1, :], (1, 0))
    roy = jnp.transpose(rt[1:2, :], (1, 0))
    roz = jnp.transpose(rt[2:3, :], (1, 0))
    rdx = jnp.transpose(rt[3:4, :], (1, 0))
    rdy = jnp.transpose(rt[4:5, :], (1, 0))
    rdz = jnp.transpose(rt[5:6, :], (1, 0))
    bd = jnp.transpose(rt[6:7, :], (1, 0))

    irx = jnp.where(jnp.abs(rdx) > 1e-12, 1.0 / jnp.where(jnp.abs(rdx) > 1e-12, rdx, 1.0), 1e12)
    iry = jnp.where(jnp.abs(rdy) > 1e-12, 1.0 / jnp.where(jnp.abs(rdy) > 1e-12, rdy, 1.0), 1e12)
    irz = jnp.where(jnp.abs(rdz) > 1e-12, 1.0 / jnp.where(jnp.abs(rdz) > 1e-12, rdz, 1.0), 1e12)

    l2_bx0, l2_by0, l2_bz0 = l2_roots_blk[0:1, :BRANCH_FACTOR], l2_roots_blk[1:2, :BRANCH_FACTOR], l2_roots_blk[2:3, :BRANCH_FACTOR]
    l2_bx1, l2_by1, l2_bz1 = l2_roots_blk[3:4, :BRANCH_FACTOR], l2_roots_blk[4:5, :BRANCH_FACTOR], l2_roots_blk[5:6, :BRANCH_FACTOR]
    l2_valid = l2_roots_blk[6:7, :BRANCH_FACTOR] > 0.5

    tx0_2, tx1_2 = (l2_bx0 - rox) * irx, (l2_bx1 - rox) * irx
    ty0_2, ty1_2 = (l2_by0 - roy) * iry, (l2_by1 - roy) * iry
    tz0_2, tz1_2 = (l2_bz0 - roz) * irz, (l2_bz1 - roz) * irz
    tmin2_pos = jnp.maximum(0.0, jnp.maximum(jnp.minimum(tx0_2, tx1_2), jnp.maximum(jnp.minimum(ty0_2, ty1_2), jnp.minimum(tz0_2, tz1_2))))
    tmax2 = jnp.minimum(jnp.maximum(tx0_2, tx1_2), jnp.minimum(jnp.maximum(ty0_2, ty1_2), jnp.maximum(tz0_2, tz1_2)))
    tmin2_eff = jnp.where((tmax2 >= tmin2_pos) & l2_valid, tmin2_pos, 1e20)
    l2_unvis_f = jnp.where(l2_valid, 1.0, 0.0)
    l2_tmin = jnp.min(jnp.where(tmin2_eff < bd, tmin2_eff, 1e20), axis=0, keepdims=True)
    z1 = jnp.zeros((BLOCK_R, 1), jnp.float32)

    def _l2_cond(c2):
      return jnp.min(c2[0]) < jnp.max(c2[2])

    def _l2_body(c2):
      t2, unvis2_f, bd2, nx2, ny2, nz2, ar2, ag2, ab2, pk2 = c2
      best_r = jnp.argmin(t2, axis=1)[0]
      unvis2_next_f = jnp.where(cols_iota_64 == best_r, 0.0, unvis2_f)

      cp2 = pltpu.make_async_copy(l2_hbm_ref.at[best_r, :, :], l2_vmem, sem_l2)
      cp2.start(); cp2.wait()
      l2_blk = l2_vmem[...]

      l1_bx0, l1_by0, l1_bz0 = l2_blk[0:1, :BRANCH_FACTOR], l2_blk[1:2, :BRANCH_FACTOR], l2_blk[2:3, :BRANCH_FACTOR]
      l1_bx1, l1_by1, l1_bz1 = l2_blk[3:4, :BRANCH_FACTOR], l2_blk[4:5, :BRANCH_FACTOR], l2_blk[5:6, :BRANCH_FACTOR]
      l1_valid = l2_blk[6:7, :BRANCH_FACTOR] > 0.5

      tx0_1, tx1_1 = (l1_bx0 - rox) * irx, (l1_bx1 - rox) * irx
      ty0_1, ty1_1 = (l1_by0 - roy) * iry, (l1_by1 - roy) * iry
      tz0_1, tz1_1 = (l1_bz0 - roz) * irz, (l1_bz1 - roz) * irz
      tmin1_pos = jnp.maximum(0.0, jnp.maximum(jnp.minimum(tx0_1, tx1_1), jnp.maximum(jnp.minimum(ty0_1, ty1_1), jnp.minimum(tz0_1, tz1_1))))
      tmax1 = jnp.minimum(jnp.maximum(tx0_1, tx1_1), jnp.minimum(jnp.maximum(ty0_1, ty1_1), jnp.maximum(tz0_1, tz1_1)))
      tmin1_eff = jnp.where((tmax1 >= tmin1_pos) & l1_valid, tmin1_pos, 1e20)
      l1_unvis_f = jnp.where(l1_valid, 1.0, 0.0)
      l1_tmin = jnp.min(jnp.where(tmin1_eff < bd2, tmin1_eff, 1e20), axis=0, keepdims=True)

      def _l1_cond(c1):
        return jnp.min(c1[0]) < jnp.max(c1[2])

      def _l1_body(c1):
        t1, unvis1_f, bd1, nx1, ny1, nz1, ar1, ag1, ab1, pk1 = c1
        best_c1 = jnp.argmin(t1, axis=1)[0]
        unvis1_next_f = jnp.where(cols_iota_64 == best_c1, 0.0, unvis1_f)
        l1_idx = best_r * BRANCH_FACTOR + best_c1

        cp0 = pltpu.make_async_copy(l0_hbm_ref.at[l1_idx, :, :], l0_vmem, sem_l0)
        cp0.start(); cp0.wait()
        l0_blk = l0_vmem[...]

        l0_bx0, l0_by0, l0_bz0 = l0_blk[0:1, :BRANCH_FACTOR], l0_blk[1:2, :BRANCH_FACTOR], l0_blk[2:3, :BRANCH_FACTOR]
        l0_bx1, l0_by1, l0_bz1 = l0_blk[3:4, :BRANCH_FACTOR], l0_blk[4:5, :BRANCH_FACTOR], l0_blk[5:6, :BRANCH_FACTOR]
        l0_valid = l0_blk[6:7, :BRANCH_FACTOR] > 0.5

        tx0_0, tx1_0 = (l0_bx0 - rox) * irx, (l0_bx1 - rox) * irx
        ty0_0, ty1_0 = (l0_by0 - roy) * iry, (l0_by1 - roy) * iry
        tz0_0, tz1_0 = (l0_bz0 - roz) * irz, (l0_bz1 - roz) * irz
        tmin0_pos = jnp.maximum(0.0, jnp.maximum(jnp.minimum(tx0_0, tx1_0), jnp.maximum(jnp.minimum(ty0_0, ty1_0), jnp.minimum(tz0_0, tz1_0))))
        tmax0 = jnp.minimum(jnp.maximum(tx0_0, tx1_0), jnp.minimum(jnp.maximum(ty0_0, ty1_0), jnp.maximum(tz0_0, tz1_0)))
        tmin0_eff = jnp.where((tmax0 >= tmin0_pos) & l0_valid, tmin0_pos, 1e20)
        l0_unvis_f = jnp.where(l0_valid, 1.0, 0.0)
        l0_tmin = jnp.min(jnp.where(tmin0_eff < bd1, tmin0_eff, 1e20), axis=0, keepdims=True)

        def _l0_cond(c0):
          return jnp.min(c0[0]) < jnp.max(c0[2])

        def _l0_body(c0):
          t0, unvis0_f, bd0, nx0, ny0, nz0, ar0, ag0, ab0, pk0 = c0
          best_c0 = jnp.argmin(t0, axis=1)[0]
          unvis0_next_f = jnp.where(cols_iota_64 == best_c0, 0.0, unvis0_f)
          leaf_idx = l1_idx * BRANCH_FACTOR + best_c0

          cp1 = pltpu.make_async_copy(leaf_hbm_ref.at[leaf_idx, :leaf_rows, :], leaf_vmem, sem_leaf)
          cp1.start(); cp1.wait()
          pk = leaf_vmem[...]

          v0x, v0y, v0z = pk[0:1, :], pk[1:2, :], pk[2:3, :]
          e1x, e1y, e1z = pk[3:4, :], pk[4:5, :], pk[5:6, :]
          e2x, e2y, e2z = pk[6:7, :], pk[7:8, :], pk[8:9, :]
          tv = pk[9:10, :] > 0.5

          hx = rdy * e2z - rdz * e2y
          hy = rdz * e2x - rdx * e2z
          hz = rdx * e2y - rdy * e2x
          a = e1x * hx + e1y * hy + e1z * hz
          ok_a = tv & (jnp.abs(a) > 1e-8)
          f = jnp.where(ok_a, 1.0 / jnp.where(ok_a, a, 1.0), 0.0)

          sx, sy, sz = rox - v0x, roy - v0y, roz - v0z
          u = f * (sx * hx + sy * hy + sz * hz)
          qx, qy, qz = sy * e1z - sz * e1y, sz * e1x - sx * e1z, sx * e1y - sy * e1x
          v = f * (rdx * qx + rdy * qy + rdz * qz)
          t = f * (e2x * qx + e2y * qy + e2z * qz)

          hit_t = ok_a & (u >= 0.0) & (v >= 0.0) & ((u + v) <= 1.0) & (t >= 0.0) & (t < bd0)

          if any_hit:
            any_ray_hit = jnp.any(hit_t, axis=1, keepdims=True)
            new_bd = jnp.where(any_ray_hit, 0.0, bd0)
            t0_next = jnp.min(jnp.where((tmin0_eff < new_bd) & (unvis0_next_f > 0.5), tmin0_eff, 1e20), axis=0, keepdims=True)
            return (t0_next, unvis0_next_f, new_bd, nx0, ny0, nz0, ar0, ag0, ab0, pk0)

          t_cand = jnp.where(hit_t, t, MJ_MAXVAL)
          min_t = jnp.min(t_cand, axis=1, keepdims=True)
          imp = min_t < bd0
          any_imp = jnp.any(imp)

          def _on_hit(_):
            n0x, n0y, n0z = pk[10:11, :], pk[11:12, :], pk[12:13, :]
            n1x, n1y, n1z = pk[13:14, :], pk[14:15, :], pk[15:16, :]
            n2x, n2y, n2z = pk[16:17, :], pk[17:18, :], pk[18:19, :]
            win = hit_t & (t_cand == min_t)
            bw = 1.0 - u - v
            lnx = jnp.sum(jnp.where(win, bw * n0x + u * n1x + v * n2x, 0.0), axis=1, keepdims=True)
            lny = jnp.sum(jnp.where(win, bw * n0y + u * n1y + v * n2y, 0.0), axis=1, keepdims=True)
            lnz = jnp.sum(jnp.where(win, bw * n0z + u * n1z + v * n2z, 0.0), axis=1, keepdims=True)
            inv_l = 1.0 / jnp.maximum(jnp.sqrt(lnx * lnx + lny * lny + lnz * lnz), 1e-12)
            lnx, lny, lnz = lnx * inv_l, lny * inv_l, lnz * inv_l

            lar = jnp.sum(jnp.where(win, pk[19:20, :], 0.0), axis=1, keepdims=True)
            lag = jnp.sum(jnp.where(win, pk[20:21, :], 0.0), axis=1, keepdims=True)
            lab = jnp.sum(jnp.where(win, pk[21:22, :], 0.0), axis=1, keepdims=True)
            lrgh = jnp.sum(jnp.where(win, pk[22:23, :], 0.0), axis=1, keepdims=True)
            lmat = jnp.sum(jnp.where(win, pk[23:24, :], 0.0), axis=1, keepdims=True)
            lpk = jnp.clip(lrgh, 0.02, 0.99) + 2.0 * jnp.floor(lmat * 100.0 + 0.5)

            new_bd = jnp.where(imp, min_t, bd0)
            t0_n = jnp.min(jnp.where((tmin0_eff < new_bd) & (unvis0_next_f > 0.5), tmin0_eff, 1e20), axis=0, keepdims=True)
            return (
                t0_n,
                new_bd,
                jnp.where(imp, lnx, nx0),
                jnp.where(imp, lny, ny0),
                jnp.where(imp, lnz, nz0),
                jnp.where(imp, lar, ar0),
                jnp.where(imp, lag, ag0),
                jnp.where(imp, lab, ab0),
                jnp.where(imp, lpk, pk0),
            )

          def _on_miss(_):
            return (jnp.where(unvis0_next_f > 0.5, t0, 1e20), bd0, nx0, ny0, nz0, ar0, ag0, ab0, pk0)

          t0_n, nbd, nnx, nny, nnz, nar, nag, nab, npk = jax.lax.cond(any_imp, _on_hit, _on_miss, None)
          return (t0_n, unvis0_next_f, nbd, nnx, nny, nnz, nar, nag, nab, npk)

        _, _, n_bd1, n_nx1, n_ny1, n_nz1, n_ar1, n_ag1, n_ab1, n_pk1 = jax.lax.while_loop(
            _l0_cond, _l0_body, (l0_tmin, l0_unvis_f, bd1, nx1, ny1, nz1, ar1, ag1, ab1, pk1)
        )
        t1_next = jnp.min(jnp.where((tmin1_eff < n_bd1) & (unvis1_next_f > 0.5), tmin1_eff, 1e20), axis=0, keepdims=True)
        return (t1_next, unvis1_next_f, n_bd1, n_nx1, n_ny1, n_nz1, n_ar1, n_ag1, n_ab1, n_pk1)

      _, _, n_bd2, n_nx2, n_ny2, n_nz2, n_ar2, n_ag2, n_ab2, n_pk2 = jax.lax.while_loop(
          _l1_cond, _l1_body, (l1_tmin, l1_unvis_f, bd2, nx2, ny2, nz2, ar2, ag2, ab2, pk2)
      )
      t2_next = jnp.min(jnp.where((tmin2_eff < n_bd2) & (unvis2_next_f > 0.5), tmin2_eff, 1e20), axis=0, keepdims=True)
      return (t2_next, unvis2_next_f, n_bd2, n_nx2, n_ny2, n_nz2, n_ar2, n_ag2, n_ab2, n_pk2)

    _, _, out_bd, out_nx, out_ny, out_nz, out_ar, out_ag, out_ab, out_pk = jax.lax.while_loop(
        _l2_cond, _l2_body, (l2_tmin, l2_unvis_f, bd, z1, z1, z1, z1, z1, z1, z1)
    )
    out_ref[...] = jnp.concatenate(
        [
            jnp.transpose(out_bd, (1, 0)),
            jnp.transpose(out_nx, (1, 0)),
            jnp.transpose(out_ny, (1, 0)),
            jnp.transpose(out_nz, (1, 0)),
            jnp.transpose(out_ar, (1, 0)),
            jnp.transpose(out_ag, (1, 0)),
            jnp.transpose(out_ab, (1, 0)),
            jnp.transpose(out_pk, (1, 0)),
        ],
        axis=0,
    )

  out_2d = pl.pallas_call(
      _bvh_kernel,
      out_shape=jax.ShapeDtypeStruct((total_tiles * 8, BLOCK_R), jnp.float32),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          grid=(total_tiles,),
          in_specs=[
              pl.BlockSpec((8, BLOCK_R), lambda i: (i, 0)),
              pl.BlockSpec((8, 128), lambda i: (0, 0)),
              pl.BlockSpec(memory_space=pltpu.HBM),
              pl.BlockSpec(memory_space=pltpu.HBM),
              pl.BlockSpec(memory_space=pltpu.HBM),
          ],
          out_specs=pl.BlockSpec((8, BLOCK_R), lambda i: (i, 0)),
          scratch_shapes=[
              pltpu.VMEM((8, 128), jnp.float32),
              pltpu.VMEM((8, 128), jnp.float32),
              pltpu.VMEM((leaf_rows, 128), jnp.float32),
              pltpu.SemaphoreType.DMA,
              pltpu.SemaphoreType.DMA,
              pltpu.SemaphoreType.DMA,
          ],
      ),
      compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
      interpret=False,
  )(rays_2d, l2_roots_dev, l2_dev, l0_dev, leaf_dev)

  out_t = out_2d.reshape(B, n_tiles, 8, BLOCK_R).transpose(0, 1, 3, 2)
  if can_2d_tile:
    ty, tx = H // TILE_H, W // TILE_W
    out_hw = out_t.reshape(B, ty, tx, TILE_H, TILE_W, 8).transpose(0, 1, 3, 2, 4, 5).reshape(B, R, 8)
  else:
    out_hw = out_t.reshape(B, n_tiles * BLOCK_R, 8)[:, :R, :]

  dist = out_hw[..., 0]
  nrm = out_hw[..., 1:4]
  albedo = out_hw[..., 4:7]
  pk_val = out_hw[..., 7]
  mat_code = jnp.floor(pk_val * 0.5) * 0.01
  roughness = jnp.clip(pk_val - 2.0 * jnp.floor(pk_val * 0.5), 0.02, 0.99)
  return dist, nrm, albedo, roughness, mat_code


@partial(jax.custom_vjp, nondiff_argnums=(7, 8, 9))
def _raycast_gas_diff(
    l2_roots_dev: jax.Array,
    l2_dev: jax.Array,
    l0_dev: jax.Array,
    leaf_dev: jax.Array,
    ro: jax.Array,
    rd: jax.Array,
    t_limit: jax.Array,
    H: int,
    W: int,
    any_hit: bool,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
  """Differentiable wrapper around Mosaic Pallas Wide-BVH-64 raycast via analytical ray-plane VJP."""
  return _pallas_gas_raycast_raw(l2_roots_dev, l2_dev, l0_dev, leaf_dev, ro, rd, t_limit, H, W, any_hit)


def _raycast_gas_fwd(l2r, l2, l0, lf, ro, rd, t_limit, H, W, any_hit):
  dist, nrm, alb, rgh, mcode = _pallas_gas_raycast_raw(l2r, l2, l0, lf, ro, rd, t_limit, H, W, any_hit)
  return (dist, nrm, alb, rgh, mcode), (ro, rd, t_limit, dist, nrm)


def _raycast_gas_bwd(H, W, any_hit, res, g_outputs):
  ro, rd, t_limit, dist, nrm = res
  g_dist, _g_nrm, _g_alb, _g_rgh, _g_mc = g_outputs
  if any_hit:
    z_ro = jnp.zeros_like(ro)
    z_rd = jnp.zeros_like(rd)
    z_tl = jnp.zeros_like(t_limit)
    return (None, None, None, None, z_ro, z_rd, z_tl)
  hit_mask = (dist < t_limit - 1e-4) & (dist >= 0.0)
  denom = jnp.sum(rd * nrm, axis=-1, keepdims=True)
  safe_denom = jnp.where(jnp.abs(denom) > 1e-3, denom, jnp.where(denom >= 0.0, 1e-3, -1e-3))
  scale = jnp.where(hit_mask[..., None], -g_dist[..., None] / safe_denom, 0.0)
  g_ro = scale * nrm
  g_rd = (scale * dist[..., None]) * nrm
  g_tl = jnp.zeros_like(t_limit)
  return (None, None, None, None, g_ro, g_rd, g_tl)


_raycast_gas_diff.defvjp(_raycast_gas_fwd, _raycast_gas_bwd)


# ==============================================================================
# OptiX-Equivalent Data Structures & Context
# ==============================================================================

@dataclasses.dataclass(frozen=True)
class OptixBuildInputTriangleArray:
  """OptiX Triangle Array input for GAS (Wide-BVH-64) construction."""
  name: str
  vertices: np.ndarray            # (V, 3) float32
  indices: np.ndarray             # (F, 3) int32
  vert_normals: Optional[np.ndarray] = None
  tri_albedo: Optional[np.ndarray] = None      # (F, 3) float32
  tri_roughness: Optional[np.ndarray] = None   # (F,) float32
  tri_mat_code: Optional[np.ndarray] = None    # (F,) float32 (0..1 metal, 2 glass, >=3 emit, or object tag)
  use_morton_only: bool = False


@dataclasses.dataclass(frozen=True)
class OptixGAS:
  """Geometry Acceleration Structure (Wide-BVH-64) resident on TPU HBM."""
  name: str
  n_tris: int
  n_leaves: int
  n_l1: int
  n_l2: int
  obj_aabb_min: np.ndarray
  obj_aabb_max: np.ndarray
  l2_roots_dev: jax.Array
  l2_dev: jax.Array
  l0_dev: jax.Array
  leaf_dev: jax.Array
  hbm_mb: float


@dataclasses.dataclass(frozen=True)
class OptixInstance:
  """Single instance entry inside a 2-Level TLAS (OptixIAS)."""
  gas: OptixGAS
  transform_pos: np.ndarray       # (3,) float32
  transform_mat: np.ndarray       # (3, 3) float32
  instance_id: int = 0
  sbt_offset: int = 0
  visibility_mask: int = 0xFF


@dataclasses.dataclass(frozen=True)
class OptixBuildInputInstanceArray:
  """OptiX Instance Array input for 2-Level TLAS (IAS) construction."""
  name: str
  instances: Tuple[OptixInstance, ...]
  max_active_per_tile: int = 16


@dataclasses.dataclass(frozen=True)
class OptixIAS:
  """Instance Acceleration Structure (2-Level TLAS over Wide-BVH-64 GAS prototypes)."""
  name: str
  unique_gases: Tuple[OptixGAS, ...]
  inst_xpos_per_gas: Tuple[jax.Array, ...]   # (1, K_g, 3)
  inst_xmat_per_gas: Tuple[jax.Array, ...]   # (1, K_g, 3, 3)
  inst_ids_per_gas: Tuple[jax.Array, ...]    # (K_g,)
  max_active_per_tile: int = 16
  total_effective_tris: int = 0


@dataclasses.dataclass(frozen=True)
class OptixHitPayload:
  hit: jax.Array
  dist: jax.Array
  normal: jax.Array
  albedo: jax.Array
  roughness: jax.Array
  mat_code: jax.Array
  instance_id: jax.Array
  ro: jax.Array
  rd: jax.Array


@dataclasses.dataclass(frozen=True)
class ShaderBindingTable:
  """OptiX-style Shader Binding Table holding pure-JAX shader programs."""
  raygen: Callable[..., jax.Array]
  miss: Callable[..., jax.Array]
  closesthit: Callable[..., jax.Array]
  anyhit: Optional[Callable[..., jax.Array]] = None


class OptixDeviceContext:
  """OptiX-equivalent TPU device context managing Mesh, Shardings, GAS/IAS builds, and launches."""

  def __init__(self, devices: Optional[Sequence[jax.Device]] = None):
    self.devices = list(devices) if devices is not None else list(jax.devices())
    self.n_chips = len(self.devices)
    self.backend = jax.default_backend()
    self.mesh = Mesh(np.array(self.devices), ("dp",))
    self.rep = NamedSharding(self.mesh, P())
    self.sh_dp = NamedSharding(self.mesh, P("dp"))
    self.chip_ids = jax.device_put(np.arange(self.n_chips, dtype=np.int32), self.sh_dp)

  def accelBuild(
      self,
      build_input: Union[OptixBuildInputTriangleArray, OptixBuildInputInstanceArray],
  ) -> Union[OptixGAS, OptixIAS]:
    return optixAccelBuild(self, build_input)

  def audit_hlo(self, lowered: Any) -> Dict[str, int]:
    return audit_hlo_zero_copy(lowered)


# ==============================================================================
# accelBuild: GAS (Wide-BVH-64) & IAS (2-Level TLAS)
# ==============================================================================

def _build_gas(ctx: OptixDeviceContext, inp: OptixBuildInputTriangleArray) -> OptixGAS:
  verts = np.asarray(inp.vertices, dtype=np.float32)
  faces = np.asarray(inp.indices, dtype=np.int32)
  F = int(faces.shape[0])
  tri_alb = np.full((F, 3), 0.72, dtype=np.float32) if inp.tri_albedo is None else np.asarray(inp.tri_albedo, dtype=np.float32)
  tri_rgh = np.full((F,), 0.45, dtype=np.float32) if inp.tri_roughness is None else np.clip(np.asarray(inp.tri_roughness, dtype=np.float32), 0.02, 0.99)
  tri_mc = np.zeros((F,), dtype=np.float32) if inp.tri_mat_code is None else np.asarray(inp.tri_mat_code, dtype=np.float32)
  vnorm = _compute_smooth_vertex_normals(verts, faces) if inp.vert_normals is None else np.asarray(inp.vert_normals, dtype=np.float32)

  centroids = (verts[faces[:, 0]] + verts[faces[:, 1]] + verts[faces[:, 2]]) * (1.0 / 3.0)
  order = _spatial_median_order(centroids, LEAF_SIZE, BRANCH_FACTOR, stop_at_l1=True)
  faces = faces[order]
  tri_alb = tri_alb[order]
  tri_rgh = tri_rgh[order]
  tri_mc = tri_mc[order]

  n_leaves = max(1, (F + LEAF_SIZE - 1) // LEAF_SIZE)
  F_pad = n_leaves * LEAF_SIZE
  leaf_bank = np.zeros((n_leaves, 24, LEAF_SIZE), dtype=np.float32)

  def _pack_vec3(row_start: int, arr_f3: np.ndarray) -> None:
    buf = np.empty((F_pad, 3), dtype=np.float32)
    buf[:F] = arr_f3
    if F_pad > F:
      buf[F:] = arr_f3[-1]
    leaf_bank[:, row_start : row_start + 3, :] = buf.reshape(n_leaves, LEAF_SIZE, 3).transpose(0, 2, 1)

  def _pack_scalar(row_idx: int, arr_f: np.ndarray) -> None:
    buf = np.empty((F_pad,), dtype=np.float32)
    buf[:F] = arr_f
    if F_pad > F:
      buf[F:] = arr_f[-1]
    leaf_bank[:, row_idx, :] = buf.reshape(n_leaves, LEAF_SIZE)

  tv0 = verts[faces[:, 0]]
  _pack_vec3(0, tv0)
  _pack_vec3(3, verts[faces[:, 1]] - tv0)
  _pack_vec3(6, verts[faces[:, 2]] - tv0)
  tv_buf = np.zeros((F_pad,), dtype=np.float32)
  tv_buf[:F] = 1.0
  leaf_bank[:, 9, :] = tv_buf.reshape(n_leaves, LEAF_SIZE)

  _pack_vec3(10, vnorm[faces[:, 0]])
  _pack_vec3(13, vnorm[faces[:, 1]])
  _pack_vec3(16, vnorm[faces[:, 2]])
  _pack_vec3(19, tri_alb)
  _pack_scalar(22, tri_rgh)
  _pack_scalar(23, tri_mc)

  v0_l = leaf_bank[:, 0:3, :]
  v1_l = v0_l + leaf_bank[:, 3:6, :]
  v2_l = v0_l + leaf_bank[:, 6:9, :]
  pts_min = np.minimum(v0_l, np.minimum(v1_l, v2_l))
  pts_max = np.maximum(v0_l, np.maximum(v1_l, v2_l))
  tv_mask = leaf_bank[:, 9:10, :] > 0.5
  leaf_min = np.min(np.where(tv_mask, pts_min, 1e20), axis=2) - 1e-4
  leaf_max = np.max(np.where(tv_mask, pts_max, -1e20), axis=2) + 1e-4

  n_l1 = max(1, (n_leaves + BRANCH_FACTOR - 1) // BRANCH_FACTOR)
  n_leaves_p = n_l1 * BRANCH_FACTOR
  l0_aabbs = np.zeros((n_l1, 8, 128), dtype=np.float32)
  l0_aabbs[:, 0:3, :] = 1e20
  l0_aabbs[:, 3:6, :] = -1e20
  lmin_p = np.full((n_leaves_p, 3), 1e20, dtype=np.float32)
  lmax_p = np.full((n_leaves_p, 3), -1e20, dtype=np.float32)
  lval_p = np.zeros((n_leaves_p,), dtype=np.float32)
  lmin_p[:n_leaves] = leaf_min
  lmax_p[:n_leaves] = leaf_max
  lval_p[:n_leaves] = 1.0
  lmin_g = lmin_p.reshape(n_l1, BRANCH_FACTOR, 3).transpose(0, 2, 1)
  lmax_g = lmax_p.reshape(n_l1, BRANCH_FACTOR, 3).transpose(0, 2, 1)
  l0_aabbs[:, 0:3, :BRANCH_FACTOR] = lmin_g
  l0_aabbs[:, 3:6, :BRANCH_FACTOR] = lmax_g
  l0_aabbs[:, 6, :BRANCH_FACTOR] = lval_p.reshape(n_l1, BRANCH_FACTOR)
  l1_min = np.min(lmin_g, axis=2)
  l1_max = np.max(lmax_g, axis=2)

  n_l2 = max(1, (n_l1 + BRANCH_FACTOR - 1) // BRANCH_FACTOR)
  assert n_l2 <= BRANCH_FACTOR, f"n_l2={n_l2} > 64"
  n_l1_p = n_l2 * BRANCH_FACTOR
  l2_aabbs = np.zeros((n_l2, 8, 128), dtype=np.float32)
  l2_aabbs[:, 0:3, :] = 1e20
  l2_aabbs[:, 3:6, :] = -1e20
  m1min_p = np.full((n_l1_p, 3), 1e20, dtype=np.float32)
  m1max_p = np.full((n_l1_p, 3), -1e20, dtype=np.float32)
  m1val_p = np.zeros((n_l1_p,), dtype=np.float32)
  m1min_p[:n_l1] = l1_min
  m1max_p[:n_l1] = l1_max
  m1val_p[:n_l1] = 1.0
  m1min_g = m1min_p.reshape(n_l2, BRANCH_FACTOR, 3).transpose(0, 2, 1)
  m1max_g = m1max_p.reshape(n_l2, BRANCH_FACTOR, 3).transpose(0, 2, 1)
  l2_aabbs[:, 0:3, :BRANCH_FACTOR] = m1min_g
  l2_aabbs[:, 3:6, :BRANCH_FACTOR] = m1max_g
  l2_aabbs[:, 6, :BRANCH_FACTOR] = m1val_p.reshape(n_l2, BRANCH_FACTOR)

  l2_min = np.min(m1min_g, axis=2)
  l2_max = np.max(m1max_g, axis=2)
  l2_roots_aabb = np.zeros((8, 128), dtype=np.float32)
  l2_roots_aabb[0:3, :] = 1e20
  l2_roots_aabb[3:6, :] = -1e20
  l2_roots_aabb[0:3, :n_l2] = l2_min.T
  l2_roots_aabb[3:6, :n_l2] = l2_max.T
  l2_roots_aabb[6, :n_l2] = 1.0

  obj_min = np.min(l2_min, axis=0).astype(np.float32)
  obj_max = np.max(l2_max, axis=0).astype(np.float32)
  hbm_bytes = l2_roots_aabb.nbytes + l2_aabbs.nbytes + l0_aabbs.nbytes + leaf_bank.nbytes

  return OptixGAS(
      name=inp.name,
      n_tris=F,
      n_leaves=n_leaves,
      n_l1=n_l1,
      n_l2=n_l2,
      obj_aabb_min=obj_min,
      obj_aabb_max=obj_max,
      l2_roots_dev=jax.device_put(l2_roots_aabb, ctx.rep),
      l2_dev=jax.device_put(l2_aabbs, ctx.rep),
      l0_dev=jax.device_put(l0_aabbs, ctx.rep),
      leaf_dev=jax.device_put(leaf_bank, ctx.rep),
      hbm_mb=round(hbm_bytes / (1024.0 * 1024.0), 3),
  )


def _build_ias(ctx: OptixDeviceContext, inp: OptixBuildInputInstanceArray) -> OptixIAS:
  by_gas: Dict[int, List[OptixInstance]] = {}
  gas_by_key: Dict[int, OptixGAS] = {}
  for inst in inp.instances:
    if inst.visibility_mask == 0:
      continue
    k = id(inst.gas)
    gas_by_key[k] = inst.gas
    by_gas.setdefault(k, []).append(inst)

  u_gases, xpos_list, xmat_list, ids_list = [], [], [], []
  total_tris = 0
  for k, insts in by_gas.items():
    g = gas_by_key[k]
    u_gases.append(g)
    total_tris += g.n_tris * len(insts)
    xp = np.stack([np.asarray(i.transform_pos, dtype=np.float32) for i in insts], axis=0)[None, :, :]
    xm = np.stack([np.asarray(i.transform_mat, dtype=np.float32) for i in insts], axis=0)[None, :, :, :]
    iids = np.array([int(i.instance_id) for i in insts], dtype=np.int32)
    xpos_list.append(jax.device_put(xp, ctx.rep))
    xmat_list.append(jax.device_put(xm, ctx.rep))
    ids_list.append(jax.device_put(iids, ctx.rep))

  return OptixIAS(
      name=inp.name,
      unique_gases=tuple(u_gases),
      inst_xpos_per_gas=tuple(xpos_list),
      inst_xmat_per_gas=tuple(xmat_list),
      inst_ids_per_gas=tuple(ids_list),
      max_active_per_tile=inp.max_active_per_tile,
      total_effective_tris=total_tris,
  )


def optixAccelBuild(
    ctx: OptixDeviceContext,
    build_input: Union[OptixBuildInputTriangleArray, OptixBuildInputInstanceArray],
) -> Union[OptixGAS, OptixIAS]:
  """Build a TPU-resident GAS (Wide-BVH-64) or IAS (2-Level TLAS)."""
  if isinstance(build_input, OptixBuildInputTriangleArray):
    return _build_gas(ctx, build_input)
  if isinstance(build_input, OptixBuildInputInstanceArray):
    return _build_ias(ctx, build_input)
  raise TypeError(f"Unsupported build_input: {type(build_input)}")


# ==============================================================================
# optixTrace: Unified GAS & 2-Level TLAS (IAS) Ray Tracing with jax.grad
# ==============================================================================

def _trace_ias_gas_group(
    gas: OptixGAS,
    inst_xpos: jax.Array,   # (1, K, 3)
    inst_xmat: jax.Array,   # (1, K, 3, 3)
    inst_ids: jax.Array,    # (K,)
    ro: jax.Array,          # (B, R, 3)
    rd: jax.Array,          # (B, R, 3)
    t_max: jax.Array,       # (B, R)
    H: int,
    W: int,
    any_hit: bool,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
  """2-Level TLAS culling + local-space Wide-BVH-64 traversal across K instances."""
  B, R, _ = ro.shape
  K = int(inst_xpos.shape[1])
  best_d = t_max
  best_nrm = jnp.zeros((B, R, 3), dtype=jnp.float32)
  best_alb = jnp.zeros((B, R, 3), dtype=jnp.float32)
  best_rgh = jnp.full((B, R), 0.5, dtype=jnp.float32)
  best_mc = jnp.zeros((B, R), dtype=jnp.float32)
  best_iid = jnp.full((B, R), -1, dtype=jnp.int32)

  # TLAS Arvo OBB -> World AABB fast culling
  c_loc = jnp.asarray(0.5 * (gas.obj_aabb_min + gas.obj_aabb_max), dtype=jnp.float32)
  h_loc = jnp.asarray(0.5 * (gas.obj_aabb_max - gas.obj_aabb_min), dtype=jnp.float32)

  for k_i in range(K):
    p = inst_xpos[0, k_i]      # (3,)
    rmat = inst_xmat[0, k_i]   # (3, 3)
    c_w = p + jnp.einsum("ij,j->i", rmat, c_loc)
    h_w = jnp.einsum("ij,j->i", jnp.abs(rmat), h_loc)
    w_min, w_max = c_w - h_w, c_w + h_w
    ir = jnp.where(jnp.abs(rd) > 1e-12, 1.0 / jnp.where(jnp.abs(rd) > 1e-12, rd, 1.0), 1e12)
    t0 = (w_min[None, None, :] - ro) * ir
    t1 = (w_max[None, None, :] - ro) * ir
    tmin_w = jnp.maximum(0.0, jnp.max(jnp.minimum(t0, t1), axis=-1))
    tmax_w = jnp.min(jnp.maximum(t0, t1), axis=-1)
    aabb_hit = (tmax_w >= tmin_w) & (tmin_w < best_d)

    ro_loc = jnp.einsum("ji,brj->bri", rmat, ro - p[None, None, :], precision=jax.lax.Precision.HIGHEST)
    rd_loc = jnp.einsum("ji,brj->bri", rmat, rd, precision=jax.lax.Precision.HIGHEST)
    t_lim_loc = jnp.where(aabb_hit, best_d, 0.0)

    d_k, n_loc, a_k, r_k, m_k = _raycast_gas_diff(
        gas.l2_roots_dev, gas.l2_dev, gas.l0_dev, gas.leaf_dev,
        ro_loc, rd_loc, t_lim_loc, H, W, any_hit,
    )
    imp = aabb_hit & (d_k < best_d)
    if any_hit:
      best_d = jnp.where(imp, 0.0, best_d)
    else:
      n_w = jnp.einsum("ij,brj->bri", rmat, n_loc, precision=jax.lax.Precision.HIGHEST)
      best_d = jnp.where(imp, d_k, best_d)
      best_nrm = jnp.where(imp[..., None], n_w, best_nrm)
      best_alb = jnp.where(imp[..., None], a_k, best_alb)
      best_rgh = jnp.where(imp, r_k, best_rgh)
      best_mc = jnp.where(imp, m_k, best_mc)
      best_iid = jnp.where(imp, inst_ids[k_i], best_iid)

  return best_d, best_nrm, best_alb, best_rgh, best_mc, best_iid


def optixTrace(
    accel: Union[OptixGAS, OptixIAS],
    ro: jax.Array,
    rd: jax.Array,
    t_min: float,
    t_max: jax.Array,
    H: int,
    W: int,
    ray_flags: int = OPTIX_RAY_FLAG_NONE,
    sbt: Optional[ShaderBindingTable] = None,
    params: Any = None,
) -> Union[OptixHitPayload, jax.Array]:
  """Trace rays against GAS (Wide-BVH-64) or IAS (2-Level TLAS).

  If `sbt` is provided, automatically dispatches `closesthit` / `miss` / `anyhit`
  programs and returns shaded radiance; otherwise returns `OptixHitPayload`.
  """
  any_hit = bool(ray_flags & OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT)
  ro_adj = ro + t_min * rd
  t_lim = jnp.maximum(t_max - t_min, 0.0)

  if isinstance(accel, OptixGAS):
    dist, nrm, alb, rgh, mcode = _raycast_gas_diff(
        accel.l2_roots_dev, accel.l2_dev, accel.l0_dev, accel.leaf_dev,
        ro_adj, rd, t_lim, H, W, any_hit,
    )
    iid = jnp.where(dist < t_lim - 1e-4, jnp.int32(0), jnp.int32(-1))
  elif isinstance(accel, OptixIAS):
    B, R, _ = ro.shape
    dist = t_lim
    nrm = jnp.zeros((B, R, 3), dtype=jnp.float32)
    alb = jnp.zeros((B, R, 3), dtype=jnp.float32)
    rgh = jnp.full((B, R), 0.5, dtype=jnp.float32)
    mcode = jnp.zeros((B, R), dtype=jnp.float32)
    iid = jnp.full((B, R), -1, dtype=jnp.int32)
    for g, xp, xm, iids in zip(
        accel.unique_gases, accel.inst_xpos_per_gas, accel.inst_xmat_per_gas, accel.inst_ids_per_gas
    ):
      dg, ng, ag, rg, mg, ig = _trace_ias_gas_group(g, xp, xm, iids, ro_adj, rd, dist, H, W, any_hit)
      imp = dg < dist
      dist = jnp.where(imp, dg, dist)
      nrm = jnp.where(imp[..., None], ng, nrm)
      alb = jnp.where(imp[..., None], ag, alb)
      rgh = jnp.where(imp, rg, rgh)
      mcode = jnp.where(imp, mg, mcode)
      iid = jnp.where(imp, ig, iid)
  else:
    raise TypeError(f"Unsupported accel type: {type(accel)}")

  hit = (dist < t_lim - 1e-4) & (dist >= 0.0)
  payload = OptixHitPayload(
      hit=hit,
      dist=dist + t_min,
      normal=nrm,
      albedo=alb,
      roughness=rgh,
      mat_code=mcode,
      instance_id=iid,
      ro=ro,
      rd=rd,
  )
  if sbt is None:
    return payload
  if any_hit and sbt.anyhit is not None:
    return sbt.anyhit(payload, params)
  ch_rad = sbt.closesthit(payload, params)
  ms_rad = sbt.miss(payload, params)
  return jnp.where(hit[..., None], ch_rad, ms_rad)


# ==============================================================================
# Default PBR / Differentiable Path-Tracing SBT Programs
# ==============================================================================

def _safe_unit(v: jax.Array, eps: float = 1e-10) -> jax.Array:
  """Differentiable NaN-safe vector normalization (safe at v=0 under jax.grad)."""
  inv_l = jax.lax.rsqrt(jnp.sum(v * v, axis=-1, keepdims=True) + eps)
  return v * inv_l


def _safe_norm(v: jax.Array, eps: float = 1e-10) -> jax.Array:
  """Differentiable NaN-safe Euclidean norm (safe at v=0 under jax.grad)."""
  return jnp.sqrt(jnp.sum(v * v, axis=-1) + eps)


def _build_onb(n: jax.Array) -> Tuple[jax.Array, jax.Array]:
  nz = n[..., 2:3]
  sign = jnp.where(nz >= 0.0, 1.0, -1.0)
  a = -1.0 / (sign + nz + jnp.where(jnp.abs(sign + nz) < 1e-6, 1e-6, 0.0))
  b = n[..., 0:1] * n[..., 1:2] * a
  t1 = jnp.concatenate([1.0 + sign * (n[..., 0:1] ** 2) * a, sign * b, -sign * n[..., 0:1]], axis=-1)
  near_neg_z = nz < -0.999999
  t1_fb = jnp.broadcast_to(jnp.array([0.0, -1.0, 0.0], dtype=jnp.float32), t1.shape)
  t1 = jnp.where(near_neg_z, t1_fb, t1)
  t1 = _safe_unit(t1)
  return t1, jnp.cross(n, t1)


def _smith_g1(ndotv: jax.Array, alpha: jax.Array) -> jax.Array:
  a2 = alpha * alpha
  ndv = jnp.maximum(ndotv, 1e-4)
  return (2.0 * ndv) / (ndv + jnp.sqrt(a2 + (1.0 - a2) * ndv * ndv + 1e-10))


def eval_ggx_brdf(
    n: jax.Array, wo: jax.Array, wi: jax.Array,
    albedo: jax.Array, metallic: jax.Array, roughness: jax.Array,
) -> jax.Array:
  ndotl = jnp.maximum(jnp.sum(n * wi, axis=-1), 0.0)
  ndotv = jnp.maximum(jnp.sum(n * wo, axis=-1), 1e-4)
  h = _safe_unit(wo + wi)
  ndoth = jnp.maximum(jnp.sum(n * h, axis=-1), 0.0)
  vdoth = jnp.maximum(jnp.sum(wo * h, axis=-1), 0.0)

  alpha = jnp.maximum(roughness * roughness, 0.015)
  a2 = alpha * alpha
  denom_d = ndoth * ndoth * (a2 - 1.0) + 1.0
  D = a2 / (math.pi * denom_d * denom_d + 1e-6)
  G = _smith_g1(ndotl, alpha) * _smith_g1(ndotv, alpha)
  f0 = 0.04 * (1.0 - metallic[..., None]) + albedo * metallic[..., None]
  F = f0 + (1.0 - f0) * ((1.0 - vdoth) ** 5)[..., None]

  spec = (D * G / jnp.maximum(4.0 * ndotl * ndotv, 1e-4))[..., None] * F
  kd = (1.0 - metallic[..., None]) * (1.0 - F)
  return kd * albedo / math.pi + spec


def agx_tonemap_srgb(rgb_lin: jax.Array, exposure: float = 1.0) -> jax.Array:
  x = jnp.maximum(rgb_lin * exposure, 0.0)
  m = jnp.array([
      [0.84247906, 0.07843360, 0.07922375],
      [0.04232824, 0.87846864, 0.07916613],
      [0.04237565, 0.07843360, 0.87914297],
  ], dtype=jnp.float32)
  inv_m = jnp.array([
      [1.19687900, -0.09802088, -0.09902974],
      [-0.05289685, 1.15190313, -0.09896118],
      [-0.05297163, -0.09804345, 1.15107367],
  ], dtype=jnp.float32)
  x_in = jnp.einsum("...j,ij->...i", x, m)
  a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
  curved = jnp.clip((x_in * (a * x_in + b)) / (x_in * (c * x_in + d) + e), 0.0, 1.0)
  out_lin = jnp.clip(jnp.einsum("...j,ij->...i", curved, inv_m), 0.0, 1.0)
  return jnp.power(jnp.maximum(out_lin, 1e-4), 1.0 / 2.2)


def make_default_pbr_sbt(
    cam_fovy: float,
    znear: float,
    zfar: float,
    spp: int = 16,
    max_bounces: int = 3,
    target_mat_tag: float = 0.45,
) -> ShaderBindingTable:
  """Create a default differentiable multi-bounce PBR ShaderBindingTable for `optixLaunch`."""

  def _miss(payload: OptixHitPayload, params: Dict[str, Any]) -> jax.Array:
    z = payload.rd[..., 2]
    tu = jnp.clip(z, 0.0, 1.0)[..., None]
    td = jnp.clip(-z, 0.0, 1.0)[..., None]
    sky_z = params.get("sky_zenith", jnp.array([0.22, 0.36, 0.62], jnp.float32))
    sky_h = params.get("sky_horizon", jnp.array([0.16, 0.18, 0.22], jnp.float32))
    sky_g = params.get("sky_ground", jnp.array([0.05, 0.048, 0.045], jnp.float32))
    return jnp.where(z[..., None] >= 0.0, (1.0 - tu) * sky_h + tu * sky_z, (1.0 - td) * sky_h + td * sky_g)

  def _anyhit(payload: OptixHitPayload, params: Dict[str, Any]) -> jax.Array:
    return payload.hit

  def _closesthit(payload: OptixHitPayload, params: Dict[str, Any], aux: Dict[str, Any]) -> jax.Array:
    H_l, W_l = aux["H"], aux["W"]
    accel = aux["accel"]
    rng = aux["rng"]
    n_unit = _safe_unit(payload.normal)
    wo = -payload.rd
    entering = jnp.sum(n_unit * wo, axis=-1) >= 0.0
    n_ff = jnp.where(entering[..., None], n_unit, -n_unit)

    # Differentiable material override for tagged hero surfaces (e.g. mat_code ~ target_mat_tag)
    is_hero = jnp.abs(payload.mat_code - target_mat_tag) < 0.08
    hero_alb = params.get("hero_albedo", None)
    hero_rgh = params.get("hero_roughness", None)
    alb = payload.albedo if hero_alb is None else jnp.where(is_hero[..., None], hero_alb[None, None, :], payload.albedo)
    rgh = payload.roughness if hero_rgh is None else jnp.where(is_hero, hero_rgh, payload.roughness)
    metallic = jnp.where(is_hero, params.get("hero_metallic", 0.0), jnp.clip(payload.mat_code, 0.0, 1.0))

    hit_pos = payload.ro + jnp.where(payload.hit, payload.dist, 0.0)[..., None] * payload.rd
    l_pos = params["light_pos"]
    l_emit = params["light_emit"]
    l_rad = params.get("light_radius", 0.35)
    l_norm = _safe_unit(params.get("light_normal", jnp.array([0.0, 0.0, -1.0], jnp.float32)))
    lt1, lt2 = _build_onb(l_norm)

    k_u, k_v = jax.random.split(rng)
    lu = jax.random.uniform(k_u, payload.hit.shape, minval=0.0, maxval=1.0, dtype=jnp.float32)
    lv = jax.random.uniform(k_v, payload.hit.shape, minval=0.0, maxval=1.0, dtype=jnp.float32)
    lr = l_rad * jnp.sqrt(lu + 1e-8)
    lphi = (2.0 * math.pi) * lv
    lp = l_pos[None, None, :] + (lr * jnp.cos(lphi))[..., None] * lt1[None, None, :] + (lr * jnp.sin(lphi))[..., None] * lt2[None, None, :]

    to_l = lp - hit_pos
    dist_l = _safe_norm(to_l)
    wi_l = _safe_unit(to_l)
    cos_s = jnp.maximum(jnp.sum(n_ff * wi_l, axis=-1), 0.0)
    cos_l = jnp.maximum(jnp.sum(l_norm[None, None, :] * (-wi_l), axis=-1), 0.0)

    sh_o = hit_pos + n_ff * 1.5e-3
    sh_lim = jnp.where(payload.hit & (cos_s > 0.0) & (cos_l > 0.0), jnp.maximum(dist_l - 2.0e-3, 0.0), 0.0)
    sh_occ = optixTrace(
        accel, sh_o, wi_l, 0.0, sh_lim, H_l, W_l,
        ray_flags=OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
        sbt=ShaderBindingTable(lambda *_: None, _miss, lambda *_: None, _anyhit),
        params=params,
    )
    vis = jnp.where(payload.hit & (cos_s > 0.0) & (cos_l > 0.0) & (~sh_occ), 1.0, 0.0)
    brdf = eval_ggx_brdf(n_ff, wo, wi_l, alb, metallic, rgh)
    l_area = math.pi * l_rad * l_rad
    geom = (cos_s * cos_l) / (dist_l * dist_l + 0.08) * l_area
    return jnp.minimum(brdf * l_emit[None, None, :] * (geom * vis)[..., None], 16.0)

  def _raygen(
      accel: Union[OptixGAS, OptixIAS],
      params: Dict[str, Any],
      H: int,
      W_local: int,
      W_total: int,
      x_off: jax.Array,
      rng_in: jax.Array,
  ) -> jax.Array:
    aspect = float(W_total) / float(H)
    half_h = float(znear) * math.tan(0.5 * float(cam_fovy) * (math.pi / 180.0))
    half_w = half_h * aspect
    R = H * W_local
    local_px = R
    spp_par = 16 if (spp >= 16 and local_px <= 32768) else (8 if (spp >= 8 and local_px <= 131072) else (4 if (spp >= 4 and local_px <= 262144) else 1))
    spp_seq = max(1, spp // spp_par)

    cxp = params["cam_pos"]
    cxm = params["cam_xmat"]

    def _sample_batch(s_idx, rad_acc):
      s_rng = jax.random.fold_in(rng_in, s_idx)
      k_jit, k_bnc = jax.random.split(s_rng)
      jitter = jax.random.uniform(k_jit, (spp_par, R, 2), minval=-0.5, maxval=0.5, dtype=jnp.float32)
      px = jnp.arange(W_local, dtype=jnp.float32) + jnp.asarray(x_off, jnp.float32) + 0.5
      py = jnp.arange(H, dtype=jnp.float32) + 0.5
      gx, gy = jnp.meshgrid(px, py, indexing="xy")
      u = (gx.reshape(1, R) + jitter[..., 0]) / float(W_total)
      v = (gy.reshape(1, R) + jitter[..., 1]) / float(H)
      x = -half_w + (2.0 * half_w) * u
      y = half_h - (2.0 * half_h) * v
      raw = jnp.stack([x, y, jnp.full_like(x, -float(znear))], axis=-1)
      ld = _safe_unit(raw)
      neg_z = jnp.maximum(-ld[..., 2], 1e-6)
      rd0 = jnp.einsum("ij,brj->bri", cxm, ld, precision=jax.lax.Precision.HIGHEST)
      rd0 = _safe_unit(rd0)
      ro0 = cxp[None, None, :] + (float(znear) / neg_z)[..., None] * rd0
      t_max0 = jnp.broadcast_to((float(zfar - znear) / neg_z), (spp_par, R))
      thr0 = jnp.ones((spp_par, R, 3), dtype=jnp.float32)
      alive0 = jnp.ones((spp_par, R), dtype=jnp.bool_)
      rad0 = jnp.zeros((spp_par, R, 3), dtype=jnp.float32)

      def _bounce(b_idx, carry):
        ro, rd, t_lim, thr, alive, rad = carry
        b_rng = jax.random.fold_in(k_bnc, b_idx)
        k_nee, k_u1, k_u2 = jax.random.split(b_rng, 3)

        pl_hit = optixTrace(accel, ro, rd, 0.0, jnp.where(alive, t_lim, 0.0), H, W_local)
        hit = alive & pl_hit.hit
        miss = alive & (~hit)
        rad = rad + jnp.where(miss[..., None], thr * _miss(pl_hit, params), 0.0)

        nee = _closesthit(pl_hit, params, {"H": H, "W": W_local, "accel": accel, "rng": k_nee})
        rad = rad + jnp.where(hit[..., None], thr * nee, 0.0)

        n_unit = _safe_unit(pl_hit.normal)
        wo = -rd
        entering = jnp.sum(n_unit * wo, axis=-1) >= 0.0
        n_ff = jnp.where(entering[..., None], n_unit, -n_unit)

        is_hero = jnp.abs(pl_hit.mat_code - target_mat_tag) < 0.08
        hero_alb = params.get("hero_albedo", None)
        alb = pl_hit.albedo if hero_alb is None else jnp.where(is_hero[..., None], hero_alb[None, None, :], pl_hit.albedo)
        metallic = jnp.where(is_hero, params.get("hero_metallic", 0.0), jnp.clip(pl_hit.mat_code, 0.0, 1.0))

        u1 = jax.random.uniform(k_u1, (spp_par, R), minval=0.0, maxval=1.0, dtype=jnp.float32)
        u2 = jax.random.uniform(k_u2, (spp_par, R), minval=0.0, maxval=1.0, dtype=jnp.float32)
        r_cos = jnp.sqrt(jnp.clip(u1, 1e-6, 1.0))
        phi = 2.0 * math.pi * u2
        t1, t2 = _build_onb(n_ff)
        wi_next = _safe_unit(t1 * (r_cos * jnp.cos(phi))[..., None] + t2 * (r_cos * jnp.sin(phi))[..., None] + n_ff * jnp.sqrt(jnp.maximum(1.0 - u1, 1e-6))[..., None])

        hit_pos = ro + jnp.where(hit, pl_hit.dist, 0.0)[..., None] * rd
        new_ro = hit_pos + n_ff * 1.5e-3
        new_thr = thr * alb * (1.0 - 0.35 * metallic[..., None])
        new_alive = hit & (jnp.max(new_thr, axis=-1) > 1e-3)
        new_t_lim = jnp.broadcast_to(jnp.float32(zfar), (spp_par, R))
        return (new_ro, wi_next, new_t_lim, new_thr, new_alive, rad)

      _, _, _, _, _, final_rad = jax.lax.fori_loop(0, max_bounces, _bounce, (ro0, rd0, t_max0, thr0, alive0, rad0))
      return rad_acc + jnp.mean(final_rad, axis=0)

    rad_sum = jax.lax.fori_loop(0, spp_seq, _sample_batch, jnp.zeros((R, 3), dtype=jnp.float32))
    rad_avg = (rad_sum / float(spp_seq)).reshape(H, W_local, 3)
    rgb = agx_tonemap_srgb(rad_avg, exposure=params.get("exposure", 1.0))
    return rgb[None, ...]

  return ShaderBindingTable(
      raygen=_raygen,
      miss=_miss,
      closesthit=_closesthit,
      anyhit=_anyhit,
  )


# ==============================================================================
# optixLaunch: Multi-Chip Sharded Execution (+ End-to-End jax.grad Ready)
# ==============================================================================

def build_optix_launcher(
    ctx: OptixDeviceContext,
    sbt: ShaderBindingTable,
    accel: Union[OptixGAS, OptixIAS],
    width: int,
    height: int,
):
  """Compile a reusable, zero-copy multi-chip `optixLaunch` callable (supports `jax.grad`)."""
  n_chips = ctx.n_chips
  assert width % n_chips == 0 and (width // n_chips) % 16 == 0 and height % 8 == 0
  W_local = width // n_chips

  def _local_launch(ac, prms, cid_arr, rng_in):
    cid = cid_arr[0]
    x_off = cid * W_local
    chip_rng = jax.random.fold_in(rng_in, cid)
    return sbt.raygen(ac, prms, height, W_local, width, x_off, chip_rng)

  sm_kw = {"check_rep": False} if "check_rep" in inspect.signature(shard_map).parameters else {"check_vma": False}
  sharded = shard_map(
      _local_launch,
      mesh=ctx.mesh,
      in_specs=(P(), P(), P("dp"), P()),
      out_specs=P("dp"),
      **sm_kw,
  )

  @jax.jit
  def launch_fn(params: Dict[str, Any], rng: jax.Array) -> jax.Array:
    return sharded(accel, params, ctx.chip_ids, rng)

  return launch_fn


def optixLaunch(
    ctx: OptixDeviceContext,
    sbt: ShaderBindingTable,
    params: Dict[str, Any],
    accel: Union[OptixGAS, OptixIAS],
    width: int,
    height: int,
    rng: Optional[jax.Array] = None,
) -> jax.Array:
  """One-shot or JIT-cached OptiX launch across all TPU chips in `ctx`."""
  if rng is None:
    rng = jax.random.PRNGKey(0)
  fn = build_optix_launcher(ctx, sbt, accel, width, height)
  prms_d = jax.device_put(params, ctx.rep)
  rng_d = jax.device_put(rng, ctx.rep)
  return fn(prms_d, rng_d)
