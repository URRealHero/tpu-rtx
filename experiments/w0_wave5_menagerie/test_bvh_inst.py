"""Prototype & verify Single-Kernel Multi-Instance Wide-BVH-64 Pallas raycaster on TPU v5p."""

from __future__ import annotations

import time
import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np

from tpu_mjx_render.adapters.tpu_optix import (
    BRANCH_FACTOR,
    LEAF_SIZE,
    _compute_smooth_vertex_normals,
    _spatial_median_order,
)
from tpu_mjx_render.primitives import MJ_MAXVAL

BLOCK_R = 128


def build_multi_mesh_bvh_from_mjmodel(m: mujoco.MjModel, enabled_groups=(0, 1, 2)):
  """Build a single concatenated Wide-BVH-64 bank for all visual GEOM_MESH in MjModel."""
  mesh_gids = [
      g for g in range(int(m.ngeom))
      if int(m.geom_type[g]) == 7
      and int(m.geom_group[g]) in enabled_groups
      and int(m.geom_dataid[g]) >= 0
  ]
  # Deduplicate by mesh_id
  proto_by_mid = {}
  l2r_list, l2_list, l0_list, leaf_list = [], [], [], []
  l2r_off, l2_off, l0_off, leaf_off = 0, 0, 0, 0

  for g in mesh_gids:
    mid = int(m.geom_dataid[g])
    if mid in proto_by_mid:
      continue
    va, vn = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
    fa, fn = int(m.mesh_faceadr[mid]), int(m.mesh_facenum[mid])
    verts = np.asarray(m.mesh_vert[va : va + vn], dtype=np.float32)
    faces = np.asarray(m.mesh_face[fa : fa + fn], dtype=np.int32)
    vnorm = _compute_smooth_vertex_normals(verts, faces)

    centroids = (verts[faces[:, 0]] + verts[faces[:, 1]] + verts[faces[:, 2]]) * (1.0 / 3.0)
    order = _spatial_median_order(centroids, LEAF_SIZE, BRANCH_FACTOR, stop_at_l1=True)
    faces = faces[order]

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
    assert n_l2 <= BRANCH_FACTOR
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
    l2_roots_aabb = np.zeros((1, 8, 128), dtype=np.float32)
    l2_roots_aabb[0, 0:3, :] = 1e20
    l2_roots_aabb[0, 3:6, :] = -1e20
    l2_roots_aabb[0, 0:3, :n_l2] = l2_min.T
    l2_roots_aabb[0, 3:6, :n_l2] = l2_max.T
    l2_roots_aabb[0, 6, :n_l2] = 1.0

    obj_min = np.min(l2_min, axis=0).astype(np.float32)
    obj_max = np.max(l2_max, axis=0).astype(np.float32)

    proto_by_mid[mid] = (l2r_off, l2_off, l0_off, leaf_off, obj_min, obj_max, F)
    l2r_list.append(l2_roots_aabb)
    l2_list.append(l2_aabbs)
    l0_list.append(l0_aabbs)
    leaf_list.append(leaf_bank)
    l2r_off += 1
    l2_off += n_l2
    l0_off += n_l1
    leaf_off += n_leaves

  return {
      "mesh_gids": np.array(mesh_gids, dtype=np.int32),
      "proto_by_mid": proto_by_mid,
      "geom_dataid": np.asarray(m.geom_dataid, dtype=np.int32),
      "l2r_bank": np.concatenate(l2r_list, axis=0),   # (U, 8, 128)
      "l2_bank": np.concatenate(l2_list, axis=0),     # (sum_l2, 8, 128)
      "l0_bank": np.concatenate(l0_list, axis=0),     # (sum_l1, 8, 128)
      "leaf_bank": np.concatenate(leaf_list, axis=0), # (sum_leaves, 24, 128)
  }


def trace_multi_mesh_bvh(
    bank,
    obj_xpos,   # (B, M, 3)
    obj_xmat,   # (B, M, 3, 3)
    ro,         # (B, R, 3)
    rd,         # (B, R, 3)
    t_limit,    # (B, R)
    H: int,
    W: int,
    any_hit: bool = False,
):
  B, R, _ = ro.shape
  M = len(bank["mesh_gids"])
  R_pad = ((R + BLOCK_R - 1) // BLOCK_R) * BLOCK_R
  n_tiles = R_pad // BLOCK_R

  ro_p = jnp.pad(ro, ((0, 0), (0, R_pad - R), (0, 0))).reshape(B * n_tiles, BLOCK_R, 3)
  rd_p = jnp.pad(rd, ((0, 0), (0, R_pad - R), (0, 0)), constant_values=1.0 / np.sqrt(3.0)).reshape(B * n_tiles, BLOCK_R, 3)
  tl_p = jnp.pad(t_limit, ((0, 0), (0, R_pad - R)), constant_values=0.0).reshape(B * n_tiles, BLOCK_R, 1)
  rays_2d = jnp.concatenate([ro_p, rd_p, tl_p, jnp.zeros_like(tl_p)], axis=-1).transpose(0, 2, 1).reshape(B * n_tiles * 8, BLOCK_R)

  # Build (B, M, 8, 128) instance table in HBM:
  # row 0, col 0..2: xpos, col 3..5: world_aabb_min, col 6..8: world_aabb_max
  # row 1, col 0..8: xmat (row-major 9 floats)
  # row 2, col 0..3: l2r_off, l2_off, l0_off, leaf_off (as int32 bitcast or float32 exact int!)
  # Wait: floats exact up to 2^24 = 16M! Our offsets are < 100,000!
  c_loc_arr, h_loc_arr, offs_arr = [], [], []
  for g in bank["mesh_gids"]:
    mid = int(bank["geom_dataid"][g])
    l2ro, l2o, l0o, lfo, omin, omax, _ = bank["proto_by_mid"][mid]
    c_loc_arr.append(0.5 * (omin + omax))
    h_loc_arr.append(0.5 * (omax - omin))
    offs_arr.append([float(l2ro), float(l2o), float(l0o), float(lfo)])
  c_loc = jnp.asarray(np.stack(c_loc_arr, axis=0), jnp.float32)  # (M, 3)
  h_loc = jnp.asarray(np.stack(h_loc_arr, axis=0), jnp.float32)  # (M, 3)
  offs = jnp.asarray(np.array(offs_arr, dtype=np.float32), jnp.float32)  # (M, 4)

  c_w = obj_xpos + jnp.einsum("bmij,mj->bmi", obj_xmat, c_loc)
  h_w = jnp.einsum("bmij,mj->bmi", jnp.abs(obj_xmat), h_loc)
  w_min = c_w - h_w
  w_max = c_w + h_w

  # Pack into (B, M, 8, 128) by setting columns via broadcasted assignment
  # Wait: easier to build (B, M, 8, 128) with zeros and scatter/concat along col=128!
  r0 = jnp.pad(jnp.concatenate([obj_xpos, w_min, w_max], axis=-1), ((0, 0), (0, 0), (0, 128 - 9)))
  r1 = jnp.pad(obj_xmat.reshape(B, M, 9), ((0, 0), (0, 0), (0, 128 - 9)))
  r2 = jnp.pad(jnp.broadcast_to(offs[None, :, :], (B, M, 4)), ((0, 0), (0, 0), (0, 128 - 4)))
  zeros_r = jnp.zeros((B, M, 5, 128), dtype=jnp.float32)
  inst_hbm = jnp.concatenate([r0[:, :, None, :], r1[:, :, None, :], r2[:, :, None, :], zeros_r], axis=2)  # (B, M, 8, 128)
  inst_flat = inst_hbm.reshape(B * M, 8, 128)

  l2r_dev = jnp.asarray(bank["l2r_bank"], dtype=jnp.float32)
  l2_dev = jnp.asarray(bank["l2_bank"], dtype=jnp.float32)
  l0_dev = jnp.asarray(bank["l0_bank"], dtype=jnp.float32)
  leaf_dev = jnp.asarray(bank["leaf_bank"], dtype=jnp.float32)
  leaf_rows = 16 if any_hit else 24

  def _kernel(
      ray_ref, inst_hbm_ref, l2r_hbm_ref, l2_hbm_ref, l0_hbm_ref, leaf_hbm_ref, out_ref,
      inst_vmem, l2r_vmem, l2_vmem, l0_vmem, leaf_vmem,
      sem_i, sem_r, sem_2, sem_0, sem_l,
  ):
    cols_iota_64 = jax.lax.broadcasted_iota(jnp.int32, (1, BRANCH_FACTOR), 1)
    tile_idx = pl.program_id(0)
    b_idx = tile_idx // n_tiles
    rt = ray_ref[...]

    rox_w = jnp.transpose(rt[0:1, :], (1, 0))
    roy_w = jnp.transpose(rt[1:2, :], (1, 0))
    roz_w = jnp.transpose(rt[2:3, :], (1, 0))
    rdx_w = jnp.transpose(rt[3:4, :], (1, 0))
    rdy_w = jnp.transpose(rt[4:5, :], (1, 0))
    rdz_w = jnp.transpose(rt[5:6, :], (1, 0))
    bd_init = jnp.transpose(rt[6:7, :], (1, 0))

    irx_w = jnp.where(jnp.abs(rdx_w) > 1e-12, 1.0 / jnp.where(jnp.abs(rdx_w) > 1e-12, rdx_w, 1.0), 1e12)
    iry_w = jnp.where(jnp.abs(rdy_w) > 1e-12, 1.0 / jnp.where(jnp.abs(rdy_w) > 1e-12, rdy_w, 1.0), 1e12)
    irz_w = jnp.where(jnp.abs(rdz_w) > 1e-12, 1.0 / jnp.where(jnp.abs(rdz_w) > 1e-12, rdz_w, 1.0), 1e12)

    z1 = jnp.zeros((BLOCK_R, 1), jnp.float32)
    gid_init = jnp.full((BLOCK_R, 1), -1, jnp.int32)

    def _inst_body(m_i, state):
      cbd, cnx, cny, cnz, cgid = state
      cpi = pltpu.make_async_copy(inst_hbm_ref.at[b_idx * M + m_i, :, :], inst_vmem, sem_i)
      cpi.start(); cpi.wait()
      iv = inst_vmem[...]

      # Note: Mosaic indexing scalar from (8, 128) -> slice (1, 1) or broadcast via sum/gather?
      # Wait: test slicing iv[0:1, 0:1] vs broadcasting!
      # Actually iv[0:1, :] has 128 cols; let's check if iv[0, 0] or iv[0:1, 0:1] works cleanly in Mosaic!
      px = iv[0, 0]; py = iv[0, 1]; pz = iv[0, 2]
      wmnx = iv[0, 3]; wmny = iv[0, 4]; wmnz = iv[0, 5]
      wmxx = iv[0, 6]; wmxy = iv[0, 7]; wmxz = iv[0, 8]

      tx0 = (wmnx - rox_w) * irx_w; tx1 = (wmxx - rox_w) * irx_w
      ty0 = (wmny - roy_w) * iry_w; ty1 = (wmxy - roy_w) * iry_w
      tz0 = (wmnz - roz_w) * irz_w; tz1 = (wmxz - roz_w) * irz_w
      tmin_w = jnp.maximum(0.0, jnp.maximum(jnp.minimum(tx0, tx1), jnp.maximum(jnp.minimum(ty0, ty1), jnp.minimum(tz0, tz1))))
      tmax_w = jnp.minimum(jnp.maximum(tx0, tx1), jnp.minimum(jnp.maximum(ty0, ty1), jnp.maximum(tz0, tz1)))
      hit_w = jnp.any((tmax_w >= tmin_w) & (tmin_w < cbd))

      def _run_inst(_):
        m00, m01, m02 = iv[1, 0], iv[1, 1], iv[1, 2]
        m10, m11, m12 = iv[1, 3], iv[1, 4], iv[1, 5]
        m20, m21, m22 = iv[1, 6], iv[1, 7], iv[1, 8]
        l2r_o = iv[2, 0].astype(jnp.int32)
        l2_o = iv[2, 1].astype(jnp.int32)
        l0_o = iv[2, 2].astype(jnp.int32)
        lf_o = iv[2, 3].astype(jnp.int32)

        dx, dy, dz = rox_w - px, roy_w - py, roz_w - pz
        rox = m00 * dx + m10 * dy + m20 * dz
        roy = m01 * dx + m11 * dy + m21 * dz
        roz = m02 * dx + m12 * dy + m22 * dz
        rdx = m00 * rdx_w + m10 * rdy_w + m20 * rdz_w
        rdy = m01 * rdx_w + m11 * rdy_w + m21 * rdz_w
        rdz = m02 * rdx_w + m12 * rdy_w + m22 * rdz_w

        irx = jnp.where(jnp.abs(rdx) > 1e-12, 1.0 / jnp.where(jnp.abs(rdx) > 1e-12, rdx, 1.0), 1e12)
        iry = jnp.where(jnp.abs(rdy) > 1e-12, 1.0 / jnp.where(jnp.abs(rdy) > 1e-12, rdy, 1.0), 1e12)
        irz = jnp.where(jnp.abs(rdz) > 1e-12, 1.0 / jnp.where(jnp.abs(rdz) > 1e-12, rdz, 1.0), 1e12)

        cpr = pltpu.make_async_copy(l2r_hbm_ref.at[l2r_o, :, :], l2r_vmem, sem_r)
        cpr.start(); cpr.wait()
        l2_roots_blk = l2r_vmem[...]

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
        l2_tmin = jnp.min(jnp.where(tmin2_eff < cbd, tmin2_eff, 1e20), axis=0, keepdims=True)

        def _l2_cond(c2):
          return jnp.min(c2[0]) < jnp.max(c2[2])

        def _l2_body(c2):
          t2, unvis2_f, bd2, nx2, ny2, nz2 = c2
          best_r = jnp.argmin(t2, axis=1)[0]
          unvis2_next_f = jnp.where(cols_iota_64 == best_r, 0.0, unvis2_f)

          cp2 = pltpu.make_async_copy(l2_hbm_ref.at[l2_o + best_r, :, :], l2_vmem, sem_2)
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
            t1, unvis1_f, bd1, nx1, ny1, nz1 = c1
            best_c1 = jnp.argmin(t1, axis=1)[0]
            unvis1_next_f = jnp.where(cols_iota_64 == best_c1, 0.0, unvis1_f)
            l1_idx = best_r * BRANCH_FACTOR + best_c1

            cp0 = pltpu.make_async_copy(l0_hbm_ref.at[l0_o + l1_idx, :, :], l0_vmem, sem_0)
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
              t0, unvis0_f, bd0, nx0, ny0, nz0 = c0
              best_c0 = jnp.argmin(t0, axis=1)[0]
              unvis0_next_f = jnp.where(cols_iota_64 == best_c0, 0.0, unvis0_f)
              leaf_idx = l1_idx * BRANCH_FACTOR + best_c0

              cp1 = pltpu.make_async_copy(leaf_hbm_ref.at[lf_o + leaf_idx, :leaf_rows, :], leaf_vmem, sem_l)
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
                any_rh = jnp.any(hit_t, axis=1, keepdims=True)
                new_bd = jnp.where(any_rh, 0.0, bd0)
                t0_next = jnp.min(jnp.where((tmin0_eff < new_bd) & (unvis0_next_f > 0.5), tmin0_eff, 1e20), axis=0, keepdims=True)
                return (t0_next, unvis0_next_f, new_bd, nx0, ny0, nz0)

              t_cand = jnp.where(hit_t, t, MJ_MAXVAL)
              min_t = jnp.min(t_cand, axis=1, keepdims=True)
              imp = min_t < bd0

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
                new_bd = jnp.where(imp, min_t, bd0)
                t0_n = jnp.min(jnp.where((tmin0_eff < new_bd) & (unvis0_next_f > 0.5), tmin0_eff, 1e20), axis=0, keepdims=True)
                return (t0_n, new_bd, jnp.where(imp, lnx, nx0), jnp.where(imp, lny, ny0), jnp.where(imp, lnz, nz0))

              def _on_miss(_):
                return (jnp.where(unvis0_next_f > 0.5, t0, 1e20), bd0, nx0, ny0, nz0)

              t0_n, nbd, nnx, nny, nnz = jax.lax.cond(jnp.any(imp), _on_hit, _on_miss, None)
              return (t0_n, unvis0_next_f, nbd, nnx, nny, nnz)

            _, _, n_bd1, n_nx1, n_ny1, n_nz1 = jax.lax.while_loop(_l0_cond, _l0_body, (l0_tmin, l0_unvis_f, bd1, nx1, ny1, nz1))
            t1_next = jnp.min(jnp.where((tmin1_eff < n_bd1) & (unvis1_next_f > 0.5), tmin1_eff, 1e20), axis=0, keepdims=True)
            return (t1_next, unvis1_next_f, n_bd1, n_nx1, n_ny1, n_nz1)

          _, _, n_bd2, n_nx2, n_ny2, n_nz2 = jax.lax.while_loop(_l1_cond, _l1_body, (l1_tmin, l1_unvis_f, bd2, nx2, ny2, nz2))
          t2_next = jnp.min(jnp.where((tmin2_eff < n_bd2) & (unvis2_next_f > 0.5), tmin2_eff, 1e20), axis=0, keepdims=True)
          return (t2_next, unvis2_next_f, n_bd2, n_nx2, n_ny2, n_nz2)

        _, _, n_bd, lnx, lny, lnz = jax.lax.while_loop(_l2_cond, _l2_body, (l2_tmin, l2_unvis_f, cbd, z1, z1, z1))
        imp_m = n_bd < cbd
        wnx = m00 * lnx + m01 * lny + m02 * lnz
        wny = m10 * lnx + m11 * lny + m12 * lnz
        wnz = m20 * lnx + m21 * lny + m22 * lnz
        return (
            jnp.where(imp_m, n_bd, cbd),
            jnp.where(imp_m, wnx, cnx),
            jnp.where(imp_m, wny, cny),
            jnp.where(imp_m, wnz, cnz),
            jnp.where(imp_m, m_i, cgid),
        )

      return jax.lax.cond(hit_w, _run_inst, lambda _: state, None)

    out_bd, out_nx, out_ny, out_nz, out_gid = jax.lax.fori_loop(
        0, M, _inst_body, (bd_init, z1, z1, z1, gid_init)
    )
    out_ref[...] = jnp.concatenate(
        [
            jnp.transpose(out_bd, (1, 0)),
            jnp.transpose(out_nx, (1, 0)),
            jnp.transpose(out_ny, (1, 0)),
            jnp.transpose(out_nz, (1, 0)),
            jnp.transpose(out_gid.astype(jnp.float32), (1, 0)),
            jnp.zeros((3, BLOCK_R), jnp.float32),
        ],
        axis=0,
    )

  out_2d = pl.pallas_call(
      _kernel,
      out_shape=jax.ShapeDtypeStruct((B * n_tiles * 8, BLOCK_R), jnp.float32),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          grid=(B * n_tiles,),
          in_specs=[
              pl.BlockSpec((8, BLOCK_R), lambda i: (i, 0)),
              pl.BlockSpec(memory_space=pltpu.HBM),
              pl.BlockSpec(memory_space=pltpu.HBM),
              pl.BlockSpec(memory_space=pltpu.HBM),
              pl.BlockSpec(memory_space=pltpu.HBM),
              pl.BlockSpec(memory_space=pltpu.HBM),
          ],
          out_specs=pl.BlockSpec((8, BLOCK_R), lambda i: (i, 0)),
          scratch_shapes=[
              pltpu.VMEM((8, 128), jnp.float32),
              pltpu.VMEM((8, 128), jnp.float32),
              pltpu.VMEM((8, 128), jnp.float32),
              pltpu.VMEM((8, 128), jnp.float32),
              pltpu.VMEM((leaf_rows, 128), jnp.float32),
              pltpu.SemaphoreType.DMA,
              pltpu.SemaphoreType.DMA,
              pltpu.SemaphoreType.DMA,
              pltpu.SemaphoreType.DMA,
              pltpu.SemaphoreType.DMA,
          ],
      ),
      compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
      interpret=False,
  )(rays_2d, inst_flat, l2r_dev, l2_dev, l0_dev, leaf_dev)

  out_t = out_2d.reshape(B, n_tiles, 8, BLOCK_R).transpose(0, 1, 3, 2).reshape(B, R_pad, 8)[:, :R, :]
  return out_t[..., 0], out_t[..., 1:4], out_t[..., 4].astype(jnp.int32)


if __name__ == "__main__":
  path = "/dev/shm/zaychik-tpu-mjx-w0/menagerie/unitree_g1/scene_mjx.xml"
  m = mujoco.MjModel.from_xml_path(path)
  mx = mjx.put_model(m, impl="jax")
  d = mjx.make_data(m, impl="jax")
  if m.nkey > 0:
    d = d.replace(qpos=jnp.asarray(m.key_qpos[0], dtype=jnp.float32))
  d = mjx.forward(mx, d)
  bank = build_multi_mesh_bvh_from_mjmodel(m)
  print("Built G1 BVH bank: M=", len(bank["mesh_gids"]),
        "l2r=", bank["l2r_bank"].shape, "l2=", bank["l2_bank"].shape,
        "l0=", bank["l0_bank"].shape, "leaf=", bank["leaf_bank"].shape)

  H, W = 128, 128
  R = H * W
  ro = jnp.broadcast_to(jnp.array([1.8, -1.6, 1.1], jnp.float32)[None, None, :], (1, R, 3))
  # Simple pinhole rays looking toward (0,0,0.65)
  target = jnp.array([0.0, 0.0, 0.65], jnp.float32)
  fwd = target - ro[0, 0]; fwd = fwd / jnp.linalg.norm(fwd)
  right = jnp.cross(fwd, jnp.array([0.0, 0.0, 1.0])); right = right / jnp.linalg.norm(right)
  up = jnp.cross(right, fwd)
  yy, xx = jnp.meshgrid(jnp.linspace(0.45, -0.45, H), jnp.linspace(-0.45, 0.45, W), indexing="ij")
  rd1 = fwd[None, None, :] + xx[..., None] * right[None, None, :] + yy[..., None] * up[None, None, :]
  rd1 = rd1 / jnp.linalg.norm(rd1, axis=-1, keepdims=True)
  rd = rd1.reshape(1, R, 3)
  tl = jnp.full((1, R), 20.0, jnp.float32)
  gids = bank["mesh_gids"]
  xpos = d.geom_xpos[gids][None, :, :]
  xmat = d.geom_xmat[gids][None, :, :, :]

  fn = jax.jit(lambda xp, xm: trace_multi_mesh_bvh(bank, xp, xm, ro, rd, tl, H, W, False))
  dist, nrm, mid = fn(xpos, xmat)
  jax.block_until_ready(dist)
  t0 = time.perf_counter()
  for _ in range(10):
    dist, nrm, mid = fn(xpos, xmat)
  jax.block_until_ready(dist)
  dt_ms = (time.perf_counter() - t0) * 100.0
  hits = int(jnp.sum(mid >= 0))
  print(f"G1 128x128 trace OK! hits={hits}/{R} ({100*hits/R:.1f}%), dt={dt_ms:.3f} ms")
