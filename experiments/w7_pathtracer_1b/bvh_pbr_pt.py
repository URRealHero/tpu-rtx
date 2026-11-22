"""Unified Single-BVH Per-Triangle PBR + Glass + Multi-Bounce Path Tracer on TPU v5p.

Extends `wide_bvh_v3` by packing per-triangle PBR materials (albedo RGB, roughness,
metallic / dielectric glass IOR 1.5 / emissive) into unused rows 19..23 of `leaf_bank`
and rows 4..7 of `out_ref` with ZERO extra HBM/VMEM DMA overhead.
"""

from __future__ import annotations

import dataclasses
import inspect
import math
import re
from typing import Any, Dict, Optional, Sequence, Tuple

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np

from tpu_mjx_render.primitives import MJ_MAXVAL
from tpu_mjx_render.roofline import inspect_hlo_ops
from wide_bvh_v3 import (
    BRANCH_FACTOR,
    LEAF_SIZE,
    BLOCK_R,
    _compute_smooth_vertex_normals,
    _morton3d_20bit,
    _pack_rays_tiled,
    _spatial_median_order,
)


def audit_hlo_zero_copy(lowered: Any) -> Dict[str, Any]:
  txt = lowered.as_text()
  info = inspect_hlo_ops(txt)
  info["send_recv_ops"] = len(re.findall(r"\bsend\(|\brecv\(", txt, re.I))
  assert info["host_callbacks"] == 0, f"HLO contains host callbacks: {info}"
  assert info["send_recv_ops"] == 0, f"HLO contains send/recv ops: {info}"
  return info


@dataclasses.dataclass(frozen=True)
class PBRWideBVHMesh:
  name: str
  n_tris_orig: int
  n_leaves: int
  n_l1: int
  n_l2: int
  l2_roots_aabb: np.ndarray  # (8, 128)
  l2_aabbs: np.ndarray       # (N_L2, 8, 128)
  l0_aabbs: np.ndarray       # (N_L1, 8, 128)
  leaf_bank: np.ndarray      # (N_leaves, 24, 128)


def build_pbr_wide_bvh(
    name: str,
    verts: np.ndarray,
    faces: np.ndarray,
    tri_albedo: np.ndarray,      # (F, 3) float32 [0, 1]
    tri_roughness: np.ndarray,   # (F,) float32 [0.02, 1.0]
    tri_mat_code: np.ndarray,    # (F,) float32: [0..1]=metallic, 2.0=glass(IOR 1.5), >=3.0=emissive_strength
    vert_normals: Optional[np.ndarray] = None,
    use_morton_only: bool = False,
) -> PBRWideBVHMesh:
  verts = np.asarray(verts, dtype=np.float32)
  faces = np.asarray(faces, dtype=np.int32)
  tri_albedo = np.asarray(tri_albedo, dtype=np.float32)
  tri_roughness = np.clip(np.asarray(tri_roughness, dtype=np.float32), 0.02, 0.99)
  tri_mat_code = np.asarray(tri_mat_code, dtype=np.float32)

  if vert_normals is None:
    vnorm = _compute_smooth_vertex_normals(verts, faces)
  else:
    vnorm = np.asarray(vert_normals, dtype=np.float32)

  centroids = (verts[faces[:, 0]] + verts[faces[:, 1]] + verts[faces[:, 2]]) * (1.0 / 3.0)
  if use_morton_only:
    order = np.argsort(_morton3d_20bit(centroids), kind="stable")
  else:
    order = _spatial_median_order(centroids, LEAF_SIZE, BRANCH_FACTOR, stop_at_l1=True)
  del centroids

  faces = faces[order]
  tri_albedo = tri_albedo[order]
  tri_roughness = tri_roughness[order]
  tri_mat_code = tri_mat_code[order]
  del order

  F = int(faces.shape[0])
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
  del tv0

  tv_buf = np.zeros((F_pad,), dtype=np.float32)
  tv_buf[:F] = 1.0
  leaf_bank[:, 9, :] = tv_buf.reshape(n_leaves, LEAF_SIZE)
  del tv_buf

  _pack_vec3(10, vnorm[faces[:, 0]])
  _pack_vec3(13, vnorm[faces[:, 1]])
  _pack_vec3(16, vnorm[faces[:, 2]])
  _pack_vec3(19, tri_albedo)
  _pack_scalar(22, tri_roughness)
  _pack_scalar(23, tri_mat_code)

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

  return PBRWideBVHMesh(
      name=name,
      n_tris_orig=F,
      n_leaves=n_leaves,
      n_l1=n_l1,
      n_l2=n_l2,
      l2_roots_aabb=l2_roots_aabb,
      l2_aabbs=l2_aabbs,
      l0_aabbs=l0_aabbs,
      leaf_bank=leaf_bank,
  )


def pbr_raycast_tpu(
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
  """Single-BVH Mosaic raycast returning (dist, normal, albedo, roughness, mat_code)."""
  B, R, _ = ro.shape
  rays_2d, n_tiles, can_2d_tile = _pack_rays_tiled(ro, rd, t_limit, H, W)
  total_tiles = B * n_tiles
  leaf_rows = 24

  def _bvh_pbr_kernel(
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
            # Skip dielectric glass & emissives (mat_code >= 1.8) so sunlight passes through window glass
            hit_opq = hit_t & (pk[23:24, :] < 1.8)
            any_ray_hit = jnp.any(hit_opq, axis=1, keepdims=True)
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
            # Pack roughness [0.02, 0.99] + 2.0 * round(mat_code * 100) into one float32 cleanly
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
      _bvh_pbr_kernel,
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
    ty, tx = H // 8, W // 16
    out_hw = out_t.reshape(B, ty, tx, 8, 16, 8).transpose(0, 1, 3, 2, 4, 5).reshape(B, R, 8)
  else:
    out_hw = out_t.reshape(B, n_tiles * BLOCK_R, 8)[:, :R, :]

  dist = out_hw[..., 0]
  nrm = out_hw[..., 1:4]
  albedo = out_hw[..., 4:7]
  pk_val = out_hw[..., 7]
  mat_code = jnp.floor(pk_val * 0.5) * 0.01
  roughness = jnp.clip(pk_val - 2.0 * jnp.floor(pk_val * 0.5), 0.02, 0.99)
  return dist, nrm, albedo, roughness, mat_code


def _build_onb(n: jax.Array) -> Tuple[jax.Array, jax.Array]:
  nz = n[..., 2:3]
  sign = jnp.where(nz >= 0.0, 1.0, -1.0)
  a = -1.0 / (sign + nz)
  b = n[..., 0:1] * n[..., 1:2] * a
  t1 = jnp.concatenate([1.0 + sign * (n[..., 0:1] ** 2) * a, sign * b, -sign * n[..., 0:1]], axis=-1)
  near_neg_z = nz < -0.999999
  t1_fb = jnp.broadcast_to(jnp.array([0.0, -1.0, 0.0], dtype=jnp.float32), t1.shape)
  t1 = jnp.where(near_neg_z, t1_fb, t1)
  t1 = t1 / jnp.maximum(jnp.linalg.norm(t1, axis=-1, keepdims=True), 1e-12)
  t2 = jnp.cross(n, t1)
  return t1, t2


def _sample_cosine_hemisphere(u1: jax.Array, u2: jax.Array, n: jax.Array) -> jax.Array:
  r = jnp.sqrt(jnp.clip(u1, 0.0, 1.0))
  phi = 2.0 * math.pi * u2
  t1, t2 = _build_onb(n)
  d = t1 * (r * jnp.cos(phi))[..., None] + t2 * (r * jnp.sin(phi))[..., None] + n * jnp.sqrt(jnp.maximum(1.0 - u1, 0.0))[..., None]
  return d / jnp.maximum(jnp.linalg.norm(d, axis=-1, keepdims=True), 1e-12)


def _sample_ggx_half(u1: jax.Array, u2: jax.Array, n: jax.Array, alpha: jax.Array) -> jax.Array:
  a2 = alpha * alpha
  u1_safe = jnp.clip(u1, 1e-5, 1.0 - 1e-5)
  cos_t2 = (1.0 - u1_safe) / (1.0 + (a2 - 1.0) * u1_safe)
  cos_t = jnp.sqrt(jnp.clip(cos_t2, 0.0, 1.0))
  sin_t = jnp.sqrt(jnp.maximum(1.0 - cos_t2, 0.0))
  phi = 2.0 * math.pi * u2
  t1, t2 = _build_onb(n)
  h = t1 * (sin_t * jnp.cos(phi))[..., None] + t2 * (sin_t * jnp.sin(phi))[..., None] + n * cos_t[..., None]
  return h / jnp.maximum(jnp.linalg.norm(h, axis=-1, keepdims=True), 1e-12)


def _smith_g1(ndotv: jax.Array, alpha: jax.Array) -> jax.Array:
  a2 = alpha * alpha
  ndv = jnp.maximum(ndotv, 1e-4)
  return (2.0 * ndv) / (ndv + jnp.sqrt(a2 + (1.0 - a2) * ndv * ndv))


def _eval_pbr_brdf(n, wo, wi, albedo, metallic, roughness):
  ndotl = jnp.maximum(jnp.sum(n * wi, axis=-1), 0.0)
  ndotv = jnp.maximum(jnp.sum(n * wo, axis=-1), 1e-4)
  h = wo + wi
  h = h / jnp.maximum(jnp.linalg.norm(h, axis=-1, keepdims=True), 1e-12)
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
  return kd * albedo / math.pi + spec


def _agx_tonemap_srgb(rgb_lin: jax.Array, exposure: float = 1.0) -> jax.Array:
  """AgX-inspired filmic curve preserving highlight saturation + sRGB gamma."""
  x = jnp.maximum(rgb_lin * exposure, 0.0)
  # Soft AgX-like shoulder + ACES chroma blend
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
  return jnp.power(out_lin, 1.0 / 2.2)


@dataclasses.dataclass(frozen=True)
class LightRig:
  area_center: Tuple[float, float, float]
  area_normal: Tuple[float, float, float]
  area_radius: float
  area_emit: Tuple[float, float, float]
  has_sun: bool = False
  sun_dir: Tuple[float, float, float] = (0.5, -0.5, 0.7)  # points TOWARD sun
  sun_cos_half_angle: float = 0.9994  # ~2 deg soft sun disk
  sun_emit: Tuple[float, float, float] = (14.0, 11.5, 8.0)
  sky_zenith: Tuple[float, float, float] = (0.25, 0.42, 0.72)
  sky_horizon: Tuple[float, float, float] = (0.18, 0.21, 0.26)
  sky_ground: Tuple[float, float, float] = (0.06, 0.055, 0.05)
  exposure: float = 1.0


def build_complex_pbr_pathtracer(
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
      # Subtle sky sun glow
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

    def _trace_batch_samples(s_idx, rad_acc):
      s_rng = jax.random.fold_in(chip_rng, s_idx)
      k_jit, k_bnc = jax.random.split(s_rng)
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

        # Direct emissive geometry hit (area light panels / bulbs)
        emit_power = jnp.where(mcode >= 2.8, (mcode - 2.0) * 6.0, 0.0)
        rad = rad + jnp.where(is_emit[..., None], thr * alb * emit_power[..., None], 0.0)

        hit_pos = ro + jnp.where(hit, d_hit, 0.0)[..., None] * rd

        # NEE 1: Soft Area Light (only for opaque surfaces)
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

        # NEE 2: Optional Sun Directional Light with soft angular penumbra
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

        # Sample next bounce (Opaque Cosine/GGX OR Dielectric Glass Snell + Schlick)
        u_lobe = jax.random.uniform(k_lobe, (spp_par, R), minval=0.0, maxval=1.0, dtype=jnp.float32)
        u1 = jax.random.uniform(k_u1, (spp_par, R), minval=0.0, maxval=1.0, dtype=jnp.float32)
        u2 = jax.random.uniform(k_u2, (spp_par, R), minval=0.0, maxval=1.0, dtype=jnp.float32)

        # Opaque bounce
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

        # Dielectric Glass bounce (IOR = 1.50)
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

      _, _, _, _, _, final_rad = jax.lax.fori_loop(
          0, max_bounces, _bounce, (ro0, rd0, t_max0, thr0, alive0, rad0)
      )
      return rad_acc + jnp.mean(final_rad, axis=0)

    rad_sum = jax.lax.fori_loop(0, spp_seq, _trace_batch_samples, jnp.zeros((R, 3), dtype=jnp.float32))
    rad_avg = (rad_sum / float(spp_seq)).reshape(height, W_local, 3)
    rgb = _agx_tonemap_srgb(rad_avg, exposure=light_rig.exposure)
    return rgb[None, ...]

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
    return sharded_fn(l2_roots_d, l2_d, l0_d, leaf_d, cxp, cxm, chip_ids, rng)

  return render_fn, mesh, rep
