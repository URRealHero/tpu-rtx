"""Wave 3: 250M-500M Unique Triangles (16-row BF16-Normal Compact BVH),
1B-10B Dynamic Instanced Triangles (2-Level TLAS + Compact BLAS),
Complex High-Occlusion Sponza Architectural Megascene, and Native 8K (7680x4320) 4-Chip Spatial Sharding.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Sequence, Tuple

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np

from tpu_mjx_render.primitives import MJ_MAXVAL
from tpu_mjx_render.renderer import GBuffer
from wide_bvh_v3 import (
    BRANCH_FACTOR,
    LEAF_SIZE,
    BLOCK_R,
    TILE_H,
    TILE_W,
    _compute_smooth_vertex_normals,
    _morton3d_20bit,
    _pack_rays_tiled,
    _shade_mesh_hits,
    _spatial_median_order,
    _unpack_out_tiled,
)


@dataclasses.dataclass(frozen=True)
class CompactWideBVHMeshItem:
  mesh_id: int
  name: str
  n_tris_orig: int
  n_leaves: int
  n_l1: int
  n_l2: int
  obj_aabb_min: np.ndarray     # (3,)
  obj_aabb_max: np.ndarray     # (3,)
  l2_roots_aabb: np.ndarray    # (8, 128) float32
  l2_aabbs: np.ndarray         # (N_L2, 8, 128) float32
  l0_aabbs: np.ndarray         # (N_L1, 8, 128) float32
  leaf_bank: np.ndarray        # (N_leaves, 16, 128) float32 (rows 0..9 FP32 geom, rows 10..14 packed BF16 normals)
  rgba: np.ndarray             # (4,)


@dataclasses.dataclass(frozen=True)
class CompactWideBVHSceneSpec:
  height: int
  width: int
  cam_fovy: float
  cam_projection: int
  cam_sensorsize: Tuple[float, float]
  cam_intrinsic: Tuple[float, float, float, float]
  znear: float
  zfar: float
  meshes: Tuple[CompactWideBVHMeshItem, ...]
  light_pos: np.ndarray
  light_ambient: Tuple[float, float, float]
  light_diffuse: Tuple[float, float, float]
  light_specular: Tuple[float, float, float]
  sky_rgb1: Tuple[float, float, float]
  sky_rgb2: Tuple[float, float, float]
  use_shadows: bool = True
  shadow_light_fraction: float = 0.0


def _pack_bf16_pair_to_f32(hi_f32: np.ndarray, lo_f32: np.ndarray) -> np.ndarray:
  """Pack two unit-range (|x| <= 1) FP32 arrays as rounded BF16 pairs inside one FP32 array."""
  hi_u32 = np.asarray(hi_f32, dtype=np.float32).view(np.uint32)
  lo_u32 = np.asarray(lo_f32, dtype=np.float32).view(np.uint32)
  hi_16 = ((hi_u32 + np.uint32(0x7FFF) + ((hi_u32 >> np.uint32(16)) & np.uint32(1))) >> np.uint32(16)) & np.uint32(0xFFFF)
  lo_16 = ((lo_u32 + np.uint32(0x7FFF) + ((lo_u32 >> np.uint32(16)) & np.uint32(1))) >> np.uint32(16)) & np.uint32(0xFFFF)
  return ((hi_16 << np.uint32(16)) | lo_16).view(np.float32)


def build_compact_bvh_mesh_item(
    mesh_id: int,
    md: Dict[str, Any],
    build_method: str = "spatial_l1_morton",
) -> CompactWideBVHMeshItem:
  """Build compact 16-row (64 B/tri) 3-level Wide-BVH with FP32 geometry & BF16-packed normals."""
  verts = np.asarray(md["verts"], dtype=np.float32)
  faces = np.asarray(md["faces"], dtype=np.int32)
  if "vert_normals" in md and md["vert_normals"] is not None:
    vnorm = np.asarray(md["vert_normals"], dtype=np.float32)
  else:
    vnorm = _compute_smooth_vertex_normals(verts, faces)

  rgba = np.asarray(md.get("rgba", [0.7, 0.7, 0.7, 1.0]), dtype=np.float32)
  if rgba.shape[0] == 3:
    rgba = np.concatenate([rgba, np.array([1.0], dtype=np.float32)])

  centroids = (verts[faces[:, 0]] + verts[faces[:, 1]] + verts[faces[:, 2]]) * (1.0 / 3.0)
  if build_method == "morton":
    order = np.argsort(_morton3d_20bit(centroids), kind="stable")
  elif build_method == "spatial_l1_morton":
    order = _spatial_median_order(centroids, LEAF_SIZE, BRANCH_FACTOR, stop_at_l1=True)
  else:
    order = _spatial_median_order(centroids, LEAF_SIZE, BRANCH_FACTOR, stop_at_l1=False)
  del centroids
  faces = faces[order]
  del order

  F = int(faces.shape[0])
  n_leaves = max(1, (F + LEAF_SIZE - 1) // LEAF_SIZE)
  F_pad = n_leaves * LEAF_SIZE

  leaf_bank = np.zeros((n_leaves, 16, LEAF_SIZE), dtype=np.float32)

  def _pack_vec3(row_start: int, arr_f3: np.ndarray) -> None:
    buf = np.empty((F_pad, 3), dtype=np.float32)
    buf[:F] = arr_f3
    if F_pad > F:
      buf[F:] = arr_f3[-1]
    leaf_bank[:, row_start : row_start + 3, :] = buf.reshape(n_leaves, LEAF_SIZE, 3).transpose(0, 2, 1)

  tv0 = verts[faces[:, 0]]
  _pack_vec3(0, tv0)
  tv1 = verts[faces[:, 1]]
  _pack_vec3(3, tv1 - tv0)
  del tv1
  tv2 = verts[faces[:, 2]]
  _pack_vec3(6, tv2 - tv0)
  del tv0, tv2

  tv_buf = np.zeros((F_pad,), dtype=np.float32)
  tv_buf[:F] = 1.0
  leaf_bank[:, 9, :] = tv_buf.reshape(n_leaves, LEAF_SIZE)
  del tv_buf

  # Pack 9 unit-normal floats (n0, n1, n2) as 5 rows of BF16 pairs!
  def _pad_n(arr_f3: np.ndarray) -> np.ndarray:
    buf = np.empty((F_pad, 3), dtype=np.float32)
    buf[:F] = np.clip(arr_f3, -1.0, 1.0)
    if F_pad > F:
      buf[F:] = buf[F - 1]
    return buf

  pn0 = _pad_n(vnorm[faces[:, 0]])
  pn1 = _pad_n(vnorm[faces[:, 1]])
  pn2 = _pad_n(vnorm[faces[:, 2]])
  zero_col = np.zeros((F_pad,), dtype=np.float32)

  def _write_pair(row_idx: int, a: np.ndarray, b: np.ndarray) -> None:
    packed = _pack_bf16_pair_to_f32(a, b)
    leaf_bank[:, row_idx, :] = packed.reshape(n_leaves, LEAF_SIZE)

  _write_pair(10, pn0[:, 0], pn0[:, 1])
  _write_pair(11, pn0[:, 2], pn1[:, 0])
  _write_pair(12, pn1[:, 1], pn1[:, 2])
  _write_pair(13, pn2[:, 0], pn2[:, 1])
  _write_pair(14, pn2[:, 2], zero_col)
  del pn0, pn1, pn2, zero_col

  # Tight leaf AABBs
  v0_l = leaf_bank[:, 0:3, :]
  v1_l = v0_l + leaf_bank[:, 3:6, :]
  v2_l = v0_l + leaf_bank[:, 6:9, :]
  pts_min = np.minimum(v0_l, np.minimum(v1_l, v2_l))
  pts_max = np.maximum(v0_l, np.maximum(v1_l, v2_l))
  tv_mask = leaf_bank[:, 9:10, :] > 0.5
  leaf_min = np.min(np.where(tv_mask, pts_min, 1e20), axis=2) - 1e-4
  leaf_max = np.max(np.where(tv_mask, pts_max, -1e20), axis=2) + 1e-4

  # Level 1 (BVH-64)
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

  # Level 2 (BVH-64)
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
  l2_roots_aabb = np.zeros((8, 128), dtype=np.float32)
  l2_roots_aabb[0:3, :] = 1e20
  l2_roots_aabb[3:6, :] = -1e20
  l2_roots_aabb[0:3, :n_l2] = l2_min.T
  l2_roots_aabb[3:6, :n_l2] = l2_max.T
  l2_roots_aabb[6, :n_l2] = 1.0

  return CompactWideBVHMeshItem(
      mesh_id=mesh_id,
      name=str(md.get("name", f"mesh_{mesh_id}")),
      n_tris_orig=F,
      n_leaves=n_leaves,
      n_l1=n_l1,
      n_l2=n_l2,
      obj_aabb_min=np.min(l2_min, axis=0).astype(np.float32),
      obj_aabb_max=np.max(l2_max, axis=0).astype(np.float32),
      l2_roots_aabb=l2_roots_aabb,
      l2_aabbs=l2_aabbs,
      l0_aabbs=l0_aabbs,
      leaf_bank=leaf_bank,
      rgba=rgba,
  )


def _compute_camera_rays_band(
    spec: Any,
    cam_pos: jax.Array,
    cam_mat: jax.Array,
    H_full: int,
    W_full: int,
    band_idx: jax.Array,
    band_h: int,
    band_stride_h: Optional[int] = None,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
  """Compute (B, band_h * W_full, 3) rays for horizontal spatial band `band_idx` on-device."""
  stride_h = band_h if band_stride_h is None else int(band_stride_h)
  aspect = float(W_full) / float(H_full)
  fovy_rad = float(spec.cam_fovy) * (np.pi / 180.0)
  half_h = float(spec.znear) * float(np.tan(0.5 * fovy_rad))
  half_w = half_h * aspect
  u = (jnp.arange(W_full, dtype=jnp.float32) + 0.5) / float(W_full)
  v = (band_idx.astype(jnp.float32) * float(stride_h) + jnp.arange(band_h, dtype=jnp.float32) + 0.5) / float(H_full)
  gx, gy = jnp.meshgrid(u, v, indexing="xy")
  x = -half_w + (2.0 * half_w) * gx
  y = half_h - (2.0 * half_h) * gy
  raw = jnp.stack([x, y, jnp.full_like(x, -float(spec.znear))], axis=-1).reshape(-1, 3)
  ld = raw / jnp.maximum(jnp.linalg.norm(raw, axis=-1, keepdims=True), 1e-12)
  neg_z = jnp.maximum(-ld[:, 2], 1e-6)
  ray_dir = jnp.einsum("bij,rj->bri", cam_mat, ld, precision=jax.lax.Precision.HIGHEST)
  ray_dir = ray_dir / jnp.maximum(jnp.linalg.norm(ray_dir, axis=-1, keepdims=True), 1e-12)
  inv_cos = (float(spec.znear) / neg_z)[None, :, None]
  ray_orig = cam_pos[:, None, :] + inv_cos * ray_dir
  return ray_orig, ray_dir, neg_z


def _traverse_compact_bvh_in_kernel(
    rox, roy, roz, irx, iry, irz, rdx, rdy, rdz,
    init_bd, init_nx, init_ny, init_nz,
    l2_roots_blk, l2_hbm_ref, l0_hbm_ref, leaf_hbm_ref,
    l2_vmem, l0_vmem, leaf_vmem, sem_l2, sem_l0, sem_leaf,
    cols_iota_64, any_hit: bool, track_stats: bool,
):
  """3-Level Wide-BVH traversal over compact 16-row (64 B/tri) leaves with BF16-packed normals."""
  def _unpack_bf16_pair(f_row):
    u = jax.lax.bitcast_convert_type(f_row, jnp.int32)
    hi = jax.lax.bitcast_convert_type(jax.lax.bitwise_and(u, jnp.int32(-65536)), jnp.float32)
    lo = jax.lax.bitcast_convert_type(
        jax.lax.shift_left(jax.lax.bitwise_and(u, jnp.int32(0xFFFF)), jnp.int32(16)), jnp.float32
    )
    return hi, lo

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
  l2_tmin = jnp.min(jnp.where(tmin2_eff < init_bd, tmin2_eff, 1e20), axis=0, keepdims=True)

  def _l2_cond(c2):
    return jnp.min(c2[0]) < jnp.max(c2[2])

  def _l2_body(c2):
    t2, unvis2_f, bd2, nx2, ny2, nz2, lcnt2 = c2
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
      t1, unvis1_f, bd1, nx1, ny1, nz1, lcnt1 = c1
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
        t0, unvis0_f, bd0, nx0, ny0, nz0, lcnt0 = c0
        best_c0 = jnp.argmin(t0, axis=1)[0]
        unvis0_next_f = jnp.where(cols_iota_64 == best_c0, 0.0, unvis0_f)
        leaf_idx = l1_idx * BRANCH_FACTOR + best_c0

        cp1 = pltpu.make_async_copy(leaf_hbm_ref.at[leaf_idx, :, :], leaf_vmem, sem_leaf)
        cp1.start(); cp1.wait()
        pk = leaf_vmem[...]  # (16, 128) = 8 KB!

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
        qx = sy * e1z - sz * e1y
        qy = sz * e1x - sx * e1z
        qz = sx * e1y - sy * e1x
        v = f * (rdx * qx + rdy * qy + rdz * qz)
        t = f * (e2x * qx + e2y * qy + e2z * qz)

        hit_t = ok_a & (u >= 0.0) & (v >= 0.0) & ((u + v) <= 1.0) & (t >= 0.0) & (t < bd0)
        lcnt_next = lcnt0 + 1.0 if track_stats else lcnt0

        if any_hit:
          any_ray_hit = jnp.any(hit_t, axis=1, keepdims=True)
          new_bd = jnp.where(any_ray_hit, 0.0, bd0)
          t0_next = jnp.min(jnp.where((tmin0_eff < new_bd) & (unvis0_next_f > 0.5), tmin0_eff, 1e20), axis=0, keepdims=True)
          return (t0_next, unvis0_next_f, new_bd, nx0, ny0, nz0, lcnt_next)

        t_cand = jnp.where(hit_t, t, MJ_MAXVAL)
        min_t = jnp.min(t_cand, axis=1, keepdims=True)
        imp = min_t < bd0
        any_imp = jnp.any(imp)

        def _on_hit(_):
          n0x, n0y = _unpack_bf16_pair(pk[10:11, :])
          n0z, n1x = _unpack_bf16_pair(pk[11:12, :])
          n1y, n1z = _unpack_bf16_pair(pk[12:13, :])
          n2x, n2y = _unpack_bf16_pair(pk[13:14, :])
          n2z, _   = _unpack_bf16_pair(pk[14:15, :])
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

        t0_next, new_bd, new_nx, new_ny, new_nz = jax.lax.cond(any_imp, _on_hit, _on_miss, None)
        return (t0_next, unvis0_next_f, new_bd, new_nx, new_ny, new_nz, lcnt_next)

      _, _, n_bd1, n_nx1, n_ny1, n_nz1, n_lcnt1 = jax.lax.while_loop(
          _l0_cond, _l0_body, (l0_tmin, l0_unvis_f, bd1, nx1, ny1, nz1, lcnt1)
      )
      t1_next = jnp.min(jnp.where((tmin1_eff < n_bd1) & (unvis1_next_f > 0.5), tmin1_eff, 1e20), axis=0, keepdims=True)
      return (t1_next, unvis1_next_f, n_bd1, n_nx1, n_ny1, n_nz1, n_lcnt1)

    _, _, n_bd2, n_nx2, n_ny2, n_nz2, n_lcnt2 = jax.lax.while_loop(
        _l1_cond, _l1_body, (l1_tmin, l1_unvis_f, bd2, nx2, ny2, nz2, lcnt2)
    )
    t2_next = jnp.min(jnp.where((tmin2_eff < n_bd2) & (unvis2_next_f > 0.5), tmin2_eff, 1e20), axis=0, keepdims=True)
    return (t2_next, unvis2_next_f, n_bd2, n_nx2, n_ny2, n_nz2, n_lcnt2)

  _, _, out_bd, out_nx, out_ny, out_nz, out_cnt = jax.lax.while_loop(
      _l2_cond, _l2_body, (l2_tmin, l2_unvis_f, init_bd, init_nx, init_ny, init_nz, jnp.float32(0.0))
  )
  return out_bd, out_nx, out_ny, out_nz, out_cnt


def _pallas_compact_bvh_raycast(
    l2_roots_dev, l2_dev, l0_dev, leaf_dev,
    ro_loc, rd_loc, t_limit, H: int, W: int,
    any_hit: bool = False, interpret: bool = False, return_stats: bool = False,
):
  B = ro_loc.shape[0]
  rays_2d, n_tiles, can_2d_tile = _pack_rays_tiled(ro_loc, rd_loc, t_limit, H, W)
  total_tiles = B * n_tiles

  def _kernel(ray_ref, l2_roots_ref, l2_hbm_ref, l0_hbm_ref, leaf_hbm_ref, out_ref, l2_vmem, l0_vmem, leaf_vmem, sem_l2, sem_l0, sem_leaf):
    cols_iota_64 = jax.lax.broadcasted_iota(jnp.int32, (1, BRANCH_FACTOR), 1)
    rt = ray_ref[...]
    l2_roots_blk = l2_roots_ref[...]
    rox, roy, roz = jnp.transpose(rt[0:1, :], (1, 0)), jnp.transpose(rt[1:2, :], (1, 0)), jnp.transpose(rt[2:3, :], (1, 0))
    rdx, rdy, rdz = jnp.transpose(rt[3:4, :], (1, 0)), jnp.transpose(rt[4:5, :], (1, 0)), jnp.transpose(rt[5:6, :], (1, 0))
    bd = jnp.transpose(rt[6:7, :], (1, 0))
    irx = jnp.where(jnp.abs(rdx) > 1e-12, 1.0 / jnp.where(jnp.abs(rdx) > 1e-12, rdx, 1.0), 1e12)
    iry = jnp.where(jnp.abs(rdy) > 1e-12, 1.0 / jnp.where(jnp.abs(rdy) > 1e-12, rdy, 1.0), 1e12)
    irz = jnp.where(jnp.abs(rdz) > 1e-12, 1.0 / jnp.where(jnp.abs(rdz) > 1e-12, rdz, 1.0), 1e12)
    zcol = jnp.zeros((BLOCK_R, 1), jnp.float32)

    cbd, cnx, cny, cnz, lcnt = _traverse_compact_bvh_in_kernel(
        rox, roy, roz, irx, iry, irz, rdx, rdy, rdz,
        bd, zcol, zcol, zcol,
        l2_roots_blk, l2_hbm_ref, l0_hbm_ref, leaf_hbm_ref,
        l2_vmem, l0_vmem, leaf_vmem, sem_l2, sem_l0, sem_leaf,
        cols_iota_64, any_hit, return_stats,
    )
    z = jnp.zeros((1, BLOCK_R), jnp.float32)
    cnt_row = jnp.broadcast_to(lcnt, (1, BLOCK_R)) if return_stats else z
    out_ref[...] = jnp.concatenate(
        [jnp.transpose(cbd, (1, 0)), jnp.transpose(cnx, (1, 0)), jnp.transpose(cny, (1, 0)), jnp.transpose(cnz, (1, 0)), cnt_row, z, z, z],
        axis=0,
    )

  out_2d = pl.pallas_call(
      _kernel,
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
              pltpu.VMEM((16, 128), jnp.float32),
              pltpu.SemaphoreType.DMA,
              pltpu.SemaphoreType.DMA,
              pltpu.SemaphoreType.DMA,
          ],
      ),
      compiler_params=None if interpret else pltpu.CompilerParams(dimension_semantics=("parallel",)),
      interpret=interpret,
  )(rays_2d, l2_roots_dev, l2_dev, l0_dev, leaf_dev)

  return _unpack_out_tiled(out_2d, B, H, W, n_tiles, can_2d_tile, return_stats=return_stats)


def render_compact_bvh_batch(
    spec: CompactWideBVHSceneSpec,
    dev_arrays: Tuple[Tuple[jax.Array, ...], ...],
    obj_xpos: jax.Array,
    obj_xmat: jax.Array,
    cam_xpos: jax.Array,
    cam_xmat: jax.Array,
    interpret: bool = False,
    return_stats: bool = False,
    band_idx: Optional[jax.Array] = None,
    H_full: Optional[int] = None,
    W_full: Optional[int] = None,
    band_stride_h: Optional[int] = None,
):
  """Render compact 16-row Wide-BVH batch (optionally a horizontal spatial band for 8K 4-chip sharding)."""
  H, W = spec.height, spec.width
  if band_idx is not None:
    assert H_full is not None and W_full is not None
    ray_orig, ray_dir, neg_z = _compute_camera_rays_band(spec, cam_xpos, cam_xmat, H_full, W_full, band_idx, H, band_stride_h=band_stride_h)
  else:
    ray_orig, ray_dir, neg_z = _compute_camera_rays_band(spec, cam_xpos, cam_xmat, H, W, jnp.int32(0), H)

  B, R, _ = ray_orig.shape
  max_dist = jnp.broadcast_to(((spec.zfar - spec.znear) / neg_z)[None, :], (B, R))

  best_d = max_dist
  best_gid = jnp.full((B, R), -1, dtype=jnp.int32)
  best_nrm = jnp.zeros((B, R, 3), dtype=jnp.float32)
  total_leaf_visits = jnp.zeros((B,), dtype=jnp.float32)

  for m_idx, _item in enumerate(spec.meshes):
    l2_roots_dev, l2_dev, l0_dev, leaf_dev = dev_arrays[m_idx]
    p = obj_xpos[:, m_idx, :]
    rmat = obj_xmat[:, m_idx, :, :]
    ro_loc = jnp.einsum("bji,brj->bri", rmat, ray_orig - p[:, None, :], precision=jax.lax.Precision.HIGHEST)
    rd_loc = jnp.einsum("bji,brj->bri", rmat, ray_dir, precision=jax.lax.Precision.HIGHEST)

    d_m, n_loc_m, st_m = _pallas_compact_bvh_raycast(
        l2_roots_dev, l2_dev, l0_dev, leaf_dev, ro_loc, rd_loc, best_d, H, W,
        any_hit=False, interpret=interpret, return_stats=return_stats,
    )
    if return_stats and st_m is not None:
      total_leaf_visits = total_leaf_visits + jnp.sum(st_m, axis=1)
    imp = d_m < best_d
    n_w_m = jnp.einsum("bij,brj->bri", rmat, n_loc_m, precision=jax.lax.Precision.HIGHEST)
    best_d = jnp.where(imp, d_m, best_d)
    best_gid = jnp.where(imp, jnp.int32(m_idx), best_gid)
    best_nrm = jnp.where(imp[..., None], n_w_m, best_nrm)

  hit = best_gid >= 0
  vdir = -ray_dir
  best_nrm = jnp.where(hit[..., None], best_nrm, 0.0)
  flip = jnp.sum(best_nrm * vdir, axis=-1, keepdims=True) < 0.0
  best_nrm = jnp.where(flip, -best_nrm, best_nrm)

  hit_pos = ray_orig + jnp.where(hit, best_d, 0.0)[..., None] * ray_dir
  l_pos = jnp.asarray(spec.light_pos, dtype=jnp.float32)
  delta_l = l_pos[None, None, :] - hit_pos
  dist_l = jnp.linalg.norm(delta_l, axis=-1)
  L = delta_l / jnp.maximum(dist_l[..., None], 1e-12)
  ndotl = jnp.maximum(0.0, jnp.sum(best_nrm * L, axis=-1))

  vis = jnp.ones((B, R), dtype=jnp.float32)
  if spec.use_shadows:
    sh_o = hit_pos + best_nrm * 2.0e-3
    sh_lim = jnp.where(hit & (ndotl > 0.0), jnp.maximum(dist_l - 1e-3, 0.0), 0.0)
    occ = jnp.zeros((B, R), dtype=jnp.bool_)
    for m_idx, _item in enumerate(spec.meshes):
      l2_roots_dev, l2_dev, l0_dev, leaf_dev = dev_arrays[m_idx]
      p = obj_xpos[:, m_idx, :]
      rmat = obj_xmat[:, m_idx, :, :]
      ro_l = jnp.einsum("bji,brj->bri", rmat, sh_o - p[:, None, :], precision=jax.lax.Precision.HIGHEST)
      rd_l = jnp.einsum("bji,brj->bri", rmat, L, precision=jax.lax.Precision.HIGHEST)
      sh_lim_rem = jnp.where(occ, 0.0, sh_lim)
      d_sh, _, st_sh = _pallas_compact_bvh_raycast(
          l2_roots_dev, l2_dev, l0_dev, leaf_dev, ro_l, rd_l, sh_lim_rem, H, W,
          any_hit=True, interpret=interpret, return_stats=return_stats,
      )
      if return_stats and st_sh is not None:
        total_leaf_visits = total_leaf_visits + jnp.sum(st_sh, axis=1)
      occ = occ | (d_sh < sh_lim_rem)
    vis = jnp.where(occ, float(spec.shadow_light_fraction), 1.0)

  gb = _shade_mesh_hits(spec, ray_orig, ray_dir, neg_z, best_d, best_gid, best_nrm, vis)
  if return_stats:
    return gb, total_leaf_visits
  return gb


# ==============================================================================
# Track 3B: 2-Level Hierarchical TLAS + Compact BLAS Instancer (1B -> 10B+ Tris)
# ==============================================================================

def _bundle_aabb_tmin(o_min, o_max, d_min, d_max, max_tl, b_min, b_max):
  """Exact conservative ray-bundle vs AABB slab test (zero false negatives)."""
  rhs1 = b_min - o_max
  t_lo1 = jnp.where(d_max > 1e-9, rhs1 / jnp.maximum(d_max, 1e-9), 0.0)
  t_hi1 = jnp.where(d_max < -1e-9, rhs1 / jnp.minimum(d_max, -1e-9), 1e20)
  ok1 = (d_max > 1e-9) | (d_max < -1e-9) | (rhs1 <= 0.0)

  rhs2 = b_max - o_min
  t_lo2 = jnp.where(d_min < -1e-9, rhs2 / jnp.minimum(d_min, -1e-9), 0.0)
  t_hi2 = jnp.where(d_min > 1e-9, rhs2 / jnp.maximum(d_min, 1e-9), 1e20)
  ok2 = (d_min < -1e-9) | (d_min > 1e-9) | (rhs2 >= 0.0)

  tmin = jnp.maximum(0.0, jnp.max(jnp.maximum(t_lo1, t_lo2), axis=-1))
  tmax = jnp.min(jnp.minimum(t_hi1, t_hi2), axis=-1)
  ok = jnp.all(ok1 & ok2, axis=-1) & (tmax >= tmin) & (tmin < max_tl)
  return ok, tmin


def _pallas_tlas2_compact_instanced_raycast(
    item: CompactWideBVHMeshItem,
    l2_roots_dev, l2_dev, l0_dev, leaf_dev,
    ray_o_world: jax.Array,     # (B, R, 3)
    ray_d_world: jax.Array,     # (B, R, 3)
    t_limit: jax.Array,         # (B, R)
    inst_xpos: jax.Array,       # (B, K_inst, 3)
    inst_xmat: jax.Array,       # (B, K_inst, 3, 3)
    H: int,
    W: int,
    tlas_group_size: int = 32,
    tlas_top_groups: int = 4,
    max_active_per_tile: int = 16,
    any_hit: bool = False,
    interpret: bool = False,
) -> Tuple[jax.Array, jax.Array]:
  """2-Level Hierarchical TLAS + Shared Compact BLAS raycast supporting 1000+ instances (10B+ tris)."""
  B, R, _ = ray_o_world.shape
  K_inst = int(inst_xpos.shape[1])
  rays_2d, n_tiles, can_2d_tile = _pack_rays_tiled(ray_o_world, ray_d_world, t_limit, H, W)
  total_tiles = B * n_tiles

  # 1. Compute world-space OBB->AABB for all K_inst instances
  c_loc = jnp.asarray(0.5 * (item.obj_aabb_min + item.obj_aabb_max), dtype=jnp.float32)
  h_loc = jnp.asarray(0.5 * (item.obj_aabb_max - item.obj_aabb_min), dtype=jnp.float32)
  c_w = inst_xpos + jnp.einsum("bkij,j->bki", inst_xmat, c_loc, precision=jax.lax.Precision.HIGHEST)
  h_w = jnp.einsum("bkij,j->bki", jnp.abs(inst_xmat), h_loc, precision=jax.lax.Precision.HIGHEST)
  w_min = c_w - h_w  # (B, K_inst, 3)
  w_max = c_w + h_w

  # Pad K_inst up to multiple of tlas_group_size
  G = max(1, (K_inst + tlas_group_size - 1) // tlas_group_size)
  K_pad_grp = G * tlas_group_size
  if K_pad_grp > K_inst:
    pad_n = K_pad_grp - K_inst
    w_min_p = jnp.pad(w_min, ((0, 0), (0, pad_n), (0, 0)), constant_values=1e20)
    w_max_p = jnp.pad(w_max, ((0, 0), (0, pad_n), (0, 0)), constant_values=-1e20)
  else:
    w_min_p, w_max_p = w_min, w_max

  # TLAS Level 1: G coarse group AABBs (B, G, 3)
  g_min = jnp.min(w_min_p.reshape(B, G, tlas_group_size, 3), axis=2)
  g_max = jnp.max(w_max_p.reshape(B, G, tlas_group_size, 3), axis=2)

  # Tile ray-bundles (B, n_tiles, 3) -> 128x smaller than testing 128 rays!
  ro_t = rays_2d.reshape(B, n_tiles, 8, BLOCK_R)[:, :, 0:3, :].transpose(0, 1, 3, 2)
  rd_t = rays_2d.reshape(B, n_tiles, 8, BLOCK_R)[:, :, 3:6, :].transpose(0, 1, 3, 2)
  tl_t = rays_2d.reshape(B, n_tiles, 8, BLOCK_R)[:, :, 6, :]
  ro_min, ro_max = jnp.min(ro_t, axis=2), jnp.max(ro_t, axis=2)
  rd_min, rd_max = jnp.min(rd_t, axis=2), jnp.max(rd_t, axis=2)
  max_tl = jnp.max(tl_t, axis=2)

  if G > tlas_top_groups:
    ok_g, tmin_g = _bundle_aabb_tmin(
        ro_min[:, :, None, :], ro_max[:, :, None, :],
        rd_min[:, :, None, :], rd_max[:, :, None, :],
        max_tl[:, :, None],
        g_min[:, None, :, :], g_max[:, None, :, :],
    )
    score_g = jnp.where(ok_g, tmin_g, 1e20)
    neg_g, top_g = jax.lax.top_k(-score_g, tlas_top_groups)  # (B, n_tiles, M_grp)
    # Gather-free one-hot selection of candidate instances from top_g groups:
    sel_g = (top_g[:, :, :, None] == jnp.arange(G, dtype=top_g.dtype)[None, None, None, :]).astype(jnp.float32)
    cand_w_min = jnp.einsum(
        "btmg,bgsc->btmsc",
        sel_g,
        w_min_p.reshape(B, G, tlas_group_size, 3),
        precision=jax.lax.Precision.HIGHEST,
    ).reshape(B, n_tiles, tlas_top_groups * tlas_group_size, 3)
    cand_w_max = jnp.einsum(
        "btmg,bgsc->btmsc",
        sel_g,
        w_max_p.reshape(B, G, tlas_group_size, 3),
        precision=jax.lax.Precision.HIGHEST,
    ).reshape(B, n_tiles, tlas_top_groups * tlas_group_size, 3)
    base_ids = (top_g[:, :, :, None] * tlas_group_size + jnp.arange(tlas_group_size)[None, None, None, :]).reshape(
        B, n_tiles, tlas_top_groups * tlas_group_size
    )
    cand_valid = (neg_g[:, :, :, None] > -1e19) & (base_ids.reshape(B, n_tiles, tlas_top_groups, tlas_group_size) < K_inst)
    cand_valid = cand_valid.reshape(B, n_tiles, tlas_top_groups * tlas_group_size)
  else:
    cand_w_min = w_min_p[:, None, :, :]
    cand_w_max = w_max_p[:, None, :, :]
    base_ids = jnp.broadcast_to(jnp.arange(K_pad_grp)[None, None, :], (B, n_tiles, K_pad_grp))
    cand_valid = base_ids < K_inst

  # TLAS Level 2: Test tile ray-bundle against candidate instances -> pick top K_act front-most IDs
  ok_i, tmin_i = _bundle_aabb_tmin(
      ro_min[:, :, None, :], ro_max[:, :, None, :],
      rd_min[:, :, None, :], rd_max[:, :, None, :],
      max_tl[:, :, None],
      cand_w_min, cand_w_max,
  )
  score_i = jnp.where(ok_i & cand_valid, tmin_i, 1e20)
  M_cand = score_i.shape[-1]
  K_act = min(max_active_per_tile, M_cand)
  neg_i, top_local = jax.lax.top_k(-score_i, K_act)
  sel_i = (top_local[:, :, :, None] == jnp.arange(M_cand, dtype=top_local.dtype)[None, None, None, :]).astype(jnp.float32)
  top_inst_ids = jnp.round(
      jnp.einsum("btkc,btc->btk", sel_i, base_ids.astype(jnp.float32), precision=jax.lax.Precision.HIGHEST)
  ).astype(jnp.int32)
  act_ids = jnp.where((-neg_i) < 1e19, top_inst_ids, -1).astype(jnp.float32)

  act_2d = jnp.concatenate([
      jnp.concatenate([
          act_ids.reshape(total_tiles, 1, K_act),
          jnp.full((total_tiles, 1, 128 - K_act), -1.0, dtype=jnp.float32),
      ], axis=2),
      jnp.full((total_tiles, 7, 128), -1.0, dtype=jnp.float32),
  ], axis=1).reshape(total_tiles * 8, 128)

  # Pack ALL K_inst poses into HBM as (B * n_pose_chunks, 16, 128) so K_inst is unlimited!
  n_pose_chunks = max(1, (K_inst + 127) // 128)
  K_pose_pad = n_pose_chunks * 128
  pad_k = K_pose_pad - K_inst
  pos_t = inst_xpos.transpose(0, 2, 1)  # (B, 3, K_inst)
  rot_t = inst_xmat.reshape(B, K_inst, 9).transpose(0, 2, 1)  # (B, 9, K_inst)
  if pad_k > 0:
    pos_t = jnp.concatenate([pos_t, jnp.zeros((B, 3, pad_k), dtype=jnp.float32)], axis=2)
    rot_t = jnp.concatenate([rot_t, jnp.zeros((B, 9, pad_k), dtype=jnp.float32)], axis=2)
  poses_b = jnp.concatenate([pos_t, rot_t, jnp.zeros((B, 4, K_pose_pad), dtype=jnp.float32)], axis=1)
  poses_hbm = poses_b.reshape(B, 16, n_pose_chunks, 128).transpose(0, 2, 1, 3).reshape(B * n_pose_chunks, 16, 128)

  def _inst_kernel(
      ray_ref, act_ref, poses_hbm_ref, l2_roots_ref, l2_hbm_ref, l0_hbm_ref, leaf_hbm_ref, out_ref,
      pose_vmem, l2_vmem, l0_vmem, leaf_vmem, sem_pose, sem_l2, sem_l0, sem_leaf,
  ):
    t_idx = pl.program_id(0)
    world_b = t_idx // n_tiles
    cols_iota_64 = jax.lax.broadcasted_iota(jnp.int32, (1, BRANCH_FACTOR), 1)
    cols_iota_128 = jax.lax.broadcasted_iota(jnp.int32, (1, 128), 1)

    rt = ray_ref[...]
    act_r = act_ref[0:1, :]
    l2_roots_blk = l2_roots_ref[...]

    w_rox, w_roy, w_roz = jnp.transpose(rt[0:1, :], (1, 0)), jnp.transpose(rt[1:2, :], (1, 0)), jnp.transpose(rt[2:3, :], (1, 0))
    w_rdx, w_rdy, w_rdz = jnp.transpose(rt[3:4, :], (1, 0)), jnp.transpose(rt[4:5, :], (1, 0)), jnp.transpose(rt[5:6, :], (1, 0))
    bd = jnp.transpose(rt[6:7, :], (1, 0))
    zcol = jnp.zeros((BLOCK_R, 1), jnp.float32)

    def _read_act_id(k_idx):
      return jnp.sum(jnp.where(cols_iota_128 == k_idx, act_r, 0.0))

    def _inst_cond(ci):
      k, inst_f, *_ = ci
      return (k < K_act) & (inst_f >= 0.0)

    def _inst_body(ci):
      k, inst_f, bd_i, nx_i, ny_i, nz_i = ci
      inst_id = inst_f.astype(jnp.int32)
      chunk_id = world_b * n_pose_chunks + (inst_id >> 7)
      col_id = inst_id & 127

      cp_p = pltpu.make_async_copy(poses_hbm_ref.at[chunk_id, :, :], pose_vmem, sem_pose)
      cp_p.start(); cp_p.wait()
      poses_w = pose_vmem[...]

      cmask = cols_iota_128 == col_id
      px = jnp.sum(jnp.where(cmask, poses_w[0:1, :], 0.0))
      py = jnp.sum(jnp.where(cmask, poses_w[1:2, :], 0.0))
      pz = jnp.sum(jnp.where(cmask, poses_w[2:3, :], 0.0))
      r00 = jnp.sum(jnp.where(cmask, poses_w[3:4, :], 0.0))
      r01 = jnp.sum(jnp.where(cmask, poses_w[4:5, :], 0.0))
      r02 = jnp.sum(jnp.where(cmask, poses_w[5:6, :], 0.0))
      r10 = jnp.sum(jnp.where(cmask, poses_w[6:7, :], 0.0))
      r11 = jnp.sum(jnp.where(cmask, poses_w[7:8, :], 0.0))
      r12 = jnp.sum(jnp.where(cmask, poses_w[8:9, :], 0.0))
      r20 = jnp.sum(jnp.where(cmask, poses_w[9:10, :], 0.0))
      r21 = jnp.sum(jnp.where(cmask, poses_w[10:11, :], 0.0))
      r22 = jnp.sum(jnp.where(cmask, poses_w[11:12, :], 0.0))

      dx_p, dy_p, dz_p = w_rox - px, w_roy - py, w_roz - pz
      rox = r00 * dx_p + r10 * dy_p + r20 * dz_p
      roy = r01 * dx_p + r11 * dy_p + r21 * dz_p
      roz = r02 * dx_p + r12 * dy_p + r22 * dz_p
      rdx = r00 * w_rdx + r10 * w_rdy + r20 * w_rdz
      rdy = r01 * w_rdx + r11 * w_rdy + r21 * w_rdz
      rdz = r02 * w_rdx + r12 * w_rdy + r22 * w_rdz

      irx = jnp.where(jnp.abs(rdx) > 1e-12, 1.0 / jnp.where(jnp.abs(rdx) > 1e-12, rdx, 1.0), 1e12)
      iry = jnp.where(jnp.abs(rdy) > 1e-12, 1.0 / jnp.where(jnp.abs(rdy) > 1e-12, rdy, 1.0), 1e12)
      irz = jnp.where(jnp.abs(rdz) > 1e-12, 1.0 / jnp.where(jnp.abs(rdz) > 1e-12, rdz, 1.0), 1e12)

      new_bd, lnx, lny, lnz, _ = _traverse_compact_bvh_in_kernel(
          rox, roy, roz, irx, iry, irz, rdx, rdy, rdz,
          bd_i, zcol, zcol, zcol,
          l2_roots_blk, l2_hbm_ref, l0_hbm_ref, leaf_hbm_ref,
          l2_vmem, l0_vmem, leaf_vmem, sem_l2, sem_l0, sem_leaf,
          cols_iota_64, any_hit, False,
      )
      imp = new_bd < bd_i
      k_next = k + 1
      inst_next_f = _read_act_id(k_next)
      if any_hit:
        return (k_next, inst_next_f, new_bd, nx_i, ny_i, nz_i)

      wnx = r00 * lnx + r01 * lny + r02 * lnz
      wny = r10 * lnx + r11 * lny + r12 * lnz
      wnz = r20 * lnx + r21 * lny + r22 * lnz
      return (k_next, inst_next_f, new_bd, jnp.where(imp, wnx, nx_i), jnp.where(imp, wny, ny_i), jnp.where(imp, wnz, nz_i))

    _, _, f_bd, f_nx, f_ny, f_nz = jax.lax.while_loop(
        _inst_cond, _inst_body, (0, _read_act_id(0), bd, zcol, zcol, zcol)
    )
    z = jnp.zeros((1, BLOCK_R), jnp.float32)
    out_ref[...] = jnp.concatenate(
        [jnp.transpose(f_bd, (1, 0)), jnp.transpose(f_nx, (1, 0)), jnp.transpose(f_ny, (1, 0)), jnp.transpose(f_nz, (1, 0)), z, z, z, z],
        axis=0,
    )

  out_2d = pl.pallas_call(
      _inst_kernel,
      out_shape=jax.ShapeDtypeStruct((total_tiles * 8, BLOCK_R), jnp.float32),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          grid=(total_tiles,),
          in_specs=[
              pl.BlockSpec((8, BLOCK_R), lambda i: (i, 0)),
              pl.BlockSpec((8, 128), lambda i: (i, 0)),
              pl.BlockSpec(memory_space=pltpu.HBM),
              pl.BlockSpec((8, 128), lambda i: (0, 0)),
              pl.BlockSpec(memory_space=pltpu.HBM),
              pl.BlockSpec(memory_space=pltpu.HBM),
              pl.BlockSpec(memory_space=pltpu.HBM),
          ],
          out_specs=pl.BlockSpec((8, BLOCK_R), lambda i: (i, 0)),
          scratch_shapes=[
              pltpu.VMEM((16, 128), jnp.float32),
              pltpu.VMEM((8, 128), jnp.float32),
              pltpu.VMEM((8, 128), jnp.float32),
              pltpu.VMEM((16, 128), jnp.float32),
              pltpu.SemaphoreType.DMA,
              pltpu.SemaphoreType.DMA,
              pltpu.SemaphoreType.DMA,
              pltpu.SemaphoreType.DMA,
          ],
      ),
      compiler_params=None if interpret else pltpu.CompilerParams(dimension_semantics=("parallel",)),
      interpret=interpret,
  )(rays_2d, act_2d, poses_hbm, l2_roots_dev, l2_dev, l0_dev, leaf_dev)

  d_out, n_out, _ = _unpack_out_tiled(out_2d, B, H, W, n_tiles, can_2d_tile, return_stats=False)
  return d_out, n_out


def render_tlas2_compact_instanced_batch(
    spec: CompactWideBVHSceneSpec,
    dev_arrays: Tuple[Tuple[jax.Array, ...], ...],
    inst_xpos_per_blas: Tuple[jax.Array, ...],
    inst_xmat_per_blas: Tuple[jax.Array, ...],
    cam_xpos: jax.Array,
    cam_xmat: jax.Array,
    tlas_group_size: int = 32,
    tlas_top_groups: int = 4,
    max_active_per_tile: int = 16,
    interpret: bool = False,
    band_idx: Optional[jax.Array] = None,
    H_full: Optional[int] = None,
    W_full: Optional[int] = None,
    band_stride_h: Optional[int] = None,
):
  """2-Level TLAS + Shared Compact BLAS renderer supporting 1B-10B+ dynamic instanced tris & 8K spatial bands."""
  H, W = spec.height, spec.width
  if band_idx is not None:
    assert H_full is not None and W_full is not None
    ray_orig, ray_dir, neg_z = _compute_camera_rays_band(spec, cam_xpos, cam_xmat, H_full, W_full, band_idx, H, band_stride_h=band_stride_h)
  else:
    ray_orig, ray_dir, neg_z = _compute_camera_rays_band(spec, cam_xpos, cam_xmat, H, W, jnp.int32(0), H)

  B, R, _ = ray_orig.shape
  max_dist = jnp.broadcast_to(((spec.zfar - spec.znear) / neg_z)[None, :], (B, R))

  best_d = max_dist
  best_gid = jnp.full((B, R), -1, dtype=jnp.int32)
  best_nrm = jnp.zeros((B, R, 3), dtype=jnp.float32)

  for m_idx, item in enumerate(spec.meshes):
    l2_roots_dev, l2_dev, l0_dev, leaf_dev = dev_arrays[m_idx]
    d_m, n_w_m = _pallas_tlas2_compact_instanced_raycast(
        item, l2_roots_dev, l2_dev, l0_dev, leaf_dev,
        ray_orig, ray_dir, best_d,
        inst_xpos_per_blas[m_idx], inst_xmat_per_blas[m_idx],
        H, W, tlas_group_size=tlas_group_size, tlas_top_groups=tlas_top_groups,
        max_active_per_tile=max_active_per_tile, any_hit=False, interpret=interpret,
    )
    imp = d_m < best_d
    best_d = jnp.where(imp, d_m, best_d)
    best_gid = jnp.where(imp, jnp.int32(m_idx), best_gid)
    best_nrm = jnp.where(imp[..., None], n_w_m, best_nrm)

  hit = best_gid >= 0
  vdir = -ray_dir
  best_nrm = jnp.where(hit[..., None], best_nrm, 0.0)
  flip = jnp.sum(best_nrm * vdir, axis=-1, keepdims=True) < 0.0
  best_nrm = jnp.where(flip, -best_nrm, best_nrm)

  hit_pos = ray_orig + jnp.where(hit, best_d, 0.0)[..., None] * ray_dir
  l_pos = jnp.asarray(spec.light_pos, dtype=jnp.float32)
  delta_l = l_pos[None, None, :] - hit_pos
  dist_l = jnp.linalg.norm(delta_l, axis=-1)
  L = delta_l / jnp.maximum(dist_l[..., None], 1e-12)
  ndotl = jnp.maximum(0.0, jnp.sum(best_nrm * L, axis=-1))

  vis = jnp.ones((B, R), dtype=jnp.float32)
  if spec.use_shadows:
    sh_o = hit_pos + best_nrm * 2.0e-3
    sh_lim = jnp.where(hit & (ndotl > 0.0), jnp.maximum(dist_l - 1e-3, 0.0), 0.0)
    occ = jnp.zeros((B, R), dtype=jnp.bool_)
    for m_idx, item in enumerate(spec.meshes):
      l2_roots_dev, l2_dev, l0_dev, leaf_dev = dev_arrays[m_idx]
      sh_lim_rem = jnp.where(occ, 0.0, sh_lim)
      d_sh, _ = _pallas_tlas2_compact_instanced_raycast(
          item, l2_roots_dev, l2_dev, l0_dev, leaf_dev,
          sh_o, jnp.broadcast_to(L, ray_orig.shape), sh_lim_rem,
          inst_xpos_per_blas[m_idx], inst_xmat_per_blas[m_idx],
          H, W, tlas_group_size=tlas_group_size, tlas_top_groups=tlas_top_groups,
          max_active_per_tile=max_active_per_tile, any_hit=True, interpret=interpret,
      )
      occ = occ | (d_sh < sh_lim_rem)
    vis = jnp.where(occ, float(spec.shadow_light_fraction), 1.0)

  return _shade_mesh_hits(spec, ray_orig, ray_dir, neg_z, best_d, best_gid, best_nrm, vis)


def make_spatial_8k_sharded_instanced_renderer(
    spec_full: CompactWideBVHSceneSpec,
    devices: Sequence[jax.Device],
    dev_arrays: Tuple[Tuple[jax.Array, ...], ...],
    tlas_group_size: int = 32,
    tlas_top_groups: int = 4,
    max_active_per_tile: int = 16,
    interpret: bool = False,
):
  """4-chip spatial-sharded renderer: splits H into 4 bands (e.g. 4 x 540 for 4K, 4 x 1080 for 8K)."""
  n_chips = len(devices)
  assert spec_full.height % n_chips == 0, (spec_full.height, n_chips)
  band_h = spec_full.height // n_chips
  band_h_pad = ((band_h + TILE_H - 1) // TILE_H) * TILE_H
  spec_band = dataclasses.replace(spec_full, height=band_h_pad)
  mesh = Mesh(np.array(devices), ("spatial",))

  def _shard_fn(da, ixp, ixm, cxp, cxm):
    bidx = jax.lax.axis_index("spatial")
    rgb = render_tlas2_compact_instanced_batch(
        spec_band, da, ixp, ixm, cxp, cxm,
        tlas_group_size=tlas_group_size, tlas_top_groups=tlas_top_groups,
        max_active_per_tile=max_active_per_tile, interpret=interpret,
        band_idx=bidx, H_full=spec_full.height, W_full=spec_full.width, band_stride_h=band_h,
    ).rgb
    return rgb[:, :band_h, :, :]

  # Replicate inputs, shard output along H axis (axis 1 of (B, H, W, 3)) -> ZERO cross-chip AllReduce!
  sharded = shard_map(
      _shard_fn,
      mesh=mesh,
      in_specs=(P(), P(), P(), P(), P()),
      out_specs=P(None, "spatial", None, None),
      check_rep=False,
  )
  return jax.jit(lambda ixp, ixm, cxp, cxm: sharded(dev_arrays, ixp, ixm, cxp, cxm)), mesh


def make_spatial_8k_sharded_unique_renderer(
    spec_full: CompactWideBVHSceneSpec,
    devices: Sequence[jax.Device],
    dev_arrays: Tuple[Tuple[jax.Array, ...], ...],
    interpret: bool = False,
):
  """4-chip spatial-sharded renderer for unique non-instanced meshes (e.g. Sponza Megascene 4K & 8K)."""
  n_chips = len(devices)
  assert spec_full.height % n_chips == 0, (spec_full.height, n_chips)
  band_h = spec_full.height // n_chips
  band_h_pad = ((band_h + TILE_H - 1) // TILE_H) * TILE_H
  spec_band = dataclasses.replace(spec_full, height=band_h_pad)
  mesh = Mesh(np.array(devices), ("spatial",))

  def _shard_fn(da, oxp, oxm, cxp, cxm):
    bidx = jax.lax.axis_index("spatial")
    rgb = render_compact_bvh_batch(
        spec_band, da, oxp, oxm, cxp, cxm,
        interpret=interpret, band_idx=bidx, H_full=spec_full.height, W_full=spec_full.width, band_stride_h=band_h,
    ).rgb
    return rgb[:, :band_h, :, :]

  sharded = shard_map(
      _shard_fn,
      mesh=mesh,
      in_specs=(P(), P(), P(), P(), P()),
      out_specs=P(None, "spatial", None, None),
      check_rep=False,
  )
  return jax.jit(lambda oxp, oxm, cxp, cxm: sharded(dev_arrays, oxp, oxm, cxp, cxm)), mesh


def make_geometry_sharded_unique_renderer(
    spec_template: CompactWideBVHSceneSpec,
    devices: Sequence[jax.Device],
    n_local_meshes: int,
    interpret: bool = False,
):
  """4-chip geometry-sharded renderer across 383 GiB HBM with exact pmin depth + psum shadow composite.

  Inputs passed to returned jit fn:
    dev_sharded: tuple of n_local_meshes tuples (l2_roots, l2, l0, leaf), each array having leading axis n_chips sharded P("geom")
    rgba_sharded: (n_chips, n_local_meshes, 3) float32 sharded P("geom")
    oxp_sharded: (n_chips, B, n_local_meshes, 3) float32 sharded P("geom")
    oxm_sharded: (n_chips, B, n_local_meshes, 3, 3) float32 sharded P("geom")
    cxp: (B, 3) replicated P()
    cxm: (B, 3, 3) replicated P()
  """
  n_chips = len(devices)
  mesh = Mesh(np.array(devices), ("geom",))
  H, W = spec_template.height, spec_template.width

  def _geom_fn(dev_s, rgba_s, oxp_s, oxm_s, cxp, cxm):
    rgba_loc = rgba_s[0]  # (n_local_meshes, 3)
    oxp_loc = oxp_s[0]    # (B, n_local_meshes, 3)
    oxm_loc = oxm_s[0]    # (B, n_local_meshes, 3, 3)
    ray_orig, ray_dir, neg_z = _compute_camera_rays_band(spec_template, cxp, cxm, H, W, jnp.int32(0), H)
    B, R, _ = ray_orig.shape
    max_dist = jnp.broadcast_to(((spec_template.zfar - spec_template.znear) / neg_z)[None, :], (B, R))

    best_d = max_dist
    best_hit = jnp.zeros((B, R), dtype=jnp.bool_)
    best_nrm = jnp.zeros((B, R, 3), dtype=jnp.float32)
    best_alb = jnp.zeros((B, R, 3), dtype=jnp.float32)

    for m_idx in range(n_local_meshes):
      l2_r, l2_a, l0_a, leaf_b = dev_s[m_idx]
      l2_r, l2_a, l0_a, leaf_b = l2_r[0], l2_a[0], l0_a[0], leaf_b[0]
      p = oxp_loc[:, m_idx, :]
      rmat = oxm_loc[:, m_idx, :, :]
      ro_loc = jnp.einsum("bji,brj->bri", rmat, ray_orig - p[:, None, :], precision=jax.lax.Precision.HIGHEST)
      rd_loc = jnp.einsum("bji,brj->bri", rmat, ray_dir, precision=jax.lax.Precision.HIGHEST)
      d_m, n_loc_m, _ = _pallas_compact_bvh_raycast(
          l2_r, l2_a, l0_a, leaf_b, ro_loc, rd_loc, best_d, H, W,
          any_hit=False, interpret=interpret, return_stats=False,
      )
      imp = d_m < best_d
      n_w_m = jnp.einsum("bij,brj->bri", rmat, n_loc_m, precision=jax.lax.Precision.HIGHEST)
      best_d = jnp.where(imp, d_m, best_d)
      best_hit = best_hit | imp
      best_nrm = jnp.where(imp[..., None], n_w_m, best_nrm)
      best_alb = jnp.where(imp[..., None], rgba_loc[m_idx][None, None, :], best_alb)

    # Exact multi-chip closest-hit depth composite via pmin + psum
    global_d = jax.lax.pmin(best_d, "geom")
    win = best_hit & (best_d <= global_d + 1e-6)
    win_f = win.astype(jnp.float32)
    win_sum = jnp.maximum(jax.lax.psum(win_f, "geom"), 1.0)
    w_hit = jax.lax.psum(win.astype(jnp.int32), "geom") > 0
    w_nrm = jax.lax.psum(jnp.where(win[..., None], best_nrm, 0.0), "geom") / win_sum[..., None]
    w_alb = jax.lax.psum(jnp.where(win[..., None], best_alb, 0.0), "geom") / win_sum[..., None]

    vdir = -ray_dir
    w_nrm = jnp.where(w_hit[..., None], w_nrm, 0.0)
    flip = jnp.sum(w_nrm * vdir, axis=-1, keepdims=True) < 0.0
    w_nrm = jnp.where(flip, -w_nrm, w_nrm)

    hit_pos = ray_orig + jnp.where(w_hit, global_d, 0.0)[..., None] * ray_dir
    l_pos = jnp.asarray(spec_template.light_pos, dtype=jnp.float32)
    delta_l = l_pos[None, None, :] - hit_pos
    dist_l = jnp.linalg.norm(delta_l, axis=-1)
    L = delta_l / jnp.maximum(dist_l[..., None], 1e-12)
    ndotl = jnp.maximum(0.0, jnp.sum(w_nrm * L, axis=-1))

    vis = jnp.ones((B, R), dtype=jnp.float32)
    if spec_template.use_shadows:
      sh_o = hit_pos + w_nrm * 2.0e-3
      sh_lim = jnp.where(w_hit & (ndotl > 0.0), jnp.maximum(dist_l - 1e-3, 0.0), 0.0)
      occ_loc = jnp.zeros((B, R), dtype=jnp.bool_)
      for m_idx in range(n_local_meshes):
        l2_r, l2_a, l0_a, leaf_b = dev_s[m_idx]
        l2_r, l2_a, l0_a, leaf_b = l2_r[0], l2_a[0], l0_a[0], leaf_b[0]
        p = oxp_loc[:, m_idx, :]
        rmat = oxm_loc[:, m_idx, :, :]
        ro_l = jnp.einsum("bji,brj->bri", rmat, sh_o - p[:, None, :], precision=jax.lax.Precision.HIGHEST)
        rd_l = jnp.einsum("bji,brj->bri", rmat, L, precision=jax.lax.Precision.HIGHEST)
        sh_lim_rem = jnp.where(occ_loc, 0.0, sh_lim)
        d_sh, _, _ = _pallas_compact_bvh_raycast(
            l2_r, l2_a, l0_a, leaf_b, ro_l, rd_l, sh_lim_rem, H, W,
            any_hit=True, interpret=interpret, return_stats=False,
        )
        occ_loc = occ_loc | (d_sh < sh_lim_rem)
      occ = jax.lax.psum(occ_loc.astype(jnp.int32), "geom") > 0
      vis = jnp.where(occ, float(spec_template.shadow_light_fraction), 1.0)

    l_amb = jnp.asarray(spec_template.light_ambient, dtype=jnp.float32)
    l_diff = jnp.asarray(spec_template.light_diffuse, dtype=jnp.float32)
    l_spec = jnp.asarray(spec_template.light_specular, dtype=jnp.float32)
    H_vec = (L + vdir) / jnp.maximum(jnp.linalg.norm(L + vdir, axis=-1, keepdims=True), 1e-12)
    spec_w = (jnp.maximum(0.0, jnp.sum(w_nrm * H_vec, axis=-1)) ** 32.0) * 0.25 * vis
    direct_rgb = jnp.clip(
        w_alb * (l_amb[None, None, :] + l_diff[None, None, :] * (ndotl * vis)[..., None]) + l_spec[None, None, :] * spec_w[..., None],
        0.0, 1.0,
    )
    sky_t = jnp.clip(0.5 * (ray_dir[..., 2] + 1.0), 0.0, 1.0)[..., None]
    sky_rgb = (
        sky_t * jnp.asarray(spec_template.sky_rgb1, dtype=jnp.float32)[None, None, :]
        + (1.0 - sky_t) * jnp.asarray(spec_template.sky_rgb2, dtype=jnp.float32)[None, None, :]
    )
    rgb = jnp.where(w_hit[..., None], direct_rgb, sky_rgb).reshape(B, H, W, 3)
    return rgb

  sharded = shard_map(
      _geom_fn,
      mesh=mesh,
      in_specs=(P("geom"), P("geom"), P("geom"), P("geom"), P(), P()),
      out_specs=P(),
      check_rep=False,
  )
  return jax.jit(sharded), mesh
