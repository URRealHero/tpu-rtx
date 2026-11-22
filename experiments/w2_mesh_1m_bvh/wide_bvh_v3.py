"""V3 Million-to-Ten-Million+ Triangle Static Wide-BVH & TLAS+BLAS Instanced TPU Raytracer.

Supports both:
- Track 2A (10M+ Unique Non-Instanced Triangles in HBM):
  * Pointer-free 3-level 64-ary Wide-BVH (L=128 tris/leaf, BVH-64 L1 & L2, up to 33.5M tris/mesh).
  * 3-level nested front-to-back `while_loop` traversal inside Mosaic `pl.pallas_call`
    (O(1) constant compile size regardless of triangle count or `n_l2`).
  * Dynamic HBM->VMEM `pltpu.async_copy` of intersected L2/L0 AABBs (`4 KB`) and leaves
    (`12 KB` primary / `8 KB` geometry-only shadow `any_hit`), plus dynamic Early-Z pruning.
  * Optional exact hardware-executed leaf/triangle counter (`return_stats=True`).
- Track 2B (10M+ to 50M+ Effective Triangles via Zero-Copy TLAS + BLAS Rigid Instancing):
  * Unique BLAS `WideBVHMeshItem` buffers live once in HBM (zero leaf duplication).
  * Dynamic world-space TLAS OBB->AABB culling per 8x16 tile sorts active instances
    front-to-back (`K_active`), and a single fused Pallas kernel per BLAS archetype loops
    over active instances in local coordinates with Early-Z skip.
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

from tpu_mjx_render.pallas_v2 import _compute_batch_camera_rays_soa
from tpu_mjx_render.primitives import MJ_MAXVAL
from tpu_mjx_render.renderer import GBuffer


def _compute_camera_rays_ondevice(
    spec: Any, cam_pos: jax.Array, cam_mat: jax.Array
) -> Tuple[jax.Array, jax.Array, jax.Array]:
  """Compute (B, R, 3) camera rays on-device with zero large embedded HLO constants (safe up to 4K+)."""
  H, W = int(spec.height), int(spec.width)
  if H * W <= 262144 and (int(spec.cam_projection) != 0 or float(spec.cam_sensorsize[1]) != 0.0):
    return _compute_batch_camera_rays_soa(spec, cam_pos, cam_mat)
  aspect = float(W) / float(H)
  fovy_rad = float(spec.cam_fovy) * (np.pi / 180.0)
  half_h = float(spec.znear) * float(np.tan(0.5 * fovy_rad))
  half_w = half_h * aspect
  u = (jnp.arange(W, dtype=jnp.float32) + 0.5) / float(W)
  v = (jnp.arange(H, dtype=jnp.float32) + 0.5) / float(H)
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

LEAF_SIZE = 128
BRANCH_FACTOR = 64
BLOCK_R = 128
TILE_H = 8
TILE_W = 16


@dataclasses.dataclass(frozen=True)
class WideBVHMeshItem:
  mesh_id: int
  name: str
  n_tris_orig: int
  n_leaves: int
  n_l1: int
  n_l2: int
  obj_aabb_min: np.ndarray     # (3,)
  obj_aabb_max: np.ndarray     # (3,)
  l2_roots_aabb: np.ndarray    # (8, 128) float32: cols 0:64 = L2 root AABBs, row 6 = valid
  l2_aabbs: np.ndarray         # (N_L2, 8, 128) float32: cols 0:64 = L1 child AABBs, row 6 = valid
  l0_aabbs: np.ndarray         # (N_L1, 8, 128) float32: cols 0:64 = L0 leaf AABBs, row 6 = valid
  leaf_bank: np.ndarray        # (N_leaves, 24, 128) float32
  rgba: np.ndarray             # (4,)


@dataclasses.dataclass(frozen=True)
class WideBVHSceneSpec:
  height: int
  width: int
  cam_fovy: float
  cam_projection: int
  cam_sensorsize: Tuple[float, float]
  cam_intrinsic: Tuple[float, float, float, float]
  znear: float
  zfar: float
  meshes: Tuple[WideBVHMeshItem, ...]
  light_pos: np.ndarray        # (3,)
  light_ambient: Tuple[float, float, float]
  light_diffuse: Tuple[float, float, float]
  light_specular: Tuple[float, float, float]
  sky_rgb1: Tuple[float, float, float]
  sky_rgb2: Tuple[float, float, float]
  use_shadows: bool = True
  shadow_light_fraction: float = 0.0


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
  return (vn / np.maximum(norms, 1e-12)).astype(np.float32)


def _morton3d_20bit(pts: np.ndarray) -> np.ndarray:
  """Compute 60-bit 3D Morton code for quantized [0, 2^20 - 1]^3 triangle centroids."""
  pmin = np.min(pts, axis=0, keepdims=True)
  pmax = np.max(pts, axis=0, keepdims=True)
  span = np.maximum(pmax - pmin, 1e-6)
  q = np.clip(np.floor((pts - pmin) / span * 1048575.0), 0, 1048575).astype(np.uint64)

  def _expand(v: np.ndarray) -> np.ndarray:
    v = (v | (v << np.uint64(32))) & np.uint64(0x001F00000000FFFF)
    v = (v | (v << np.uint64(16))) & np.uint64(0x001F0000FF0000FF)
    v = (v | (v << np.uint64(8))) & np.uint64(0x100F00F00F00F00F)
    v = (v | (v << np.uint64(4))) & np.uint64(0x10C30C30C30C30C3)
    v = (v | (v << np.uint64(2))) & np.uint64(0x1249249249249249)
    return v

  return (_expand(q[:, 0]) << np.uint64(2)) | (_expand(q[:, 1]) << np.uint64(1)) | _expand(q[:, 2])


def _spatial_median_order(
    centroids: np.ndarray,
    leaf_size: int = LEAF_SIZE,
    branch_factor: int = BRANCH_FACTOR,
    stop_at_l1: bool = False,
) -> np.ndarray:
  """Order triangles into spatially tight aligned leaves & 64-ary L1/L2 buckets via top-down argpartition."""
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

  l1_tris = branch_factor * leaf_size
  _split_recursive(idx, l1_tris)
  if stop_at_l1:
    # Fast vectorized Morton sort within each tight 8192-tri L1 spatial bucket (10x faster for 100M+ tris!)
    mcodes = _morton3d_20bit(centroids)
    for start in range(0, F, l1_tris):
      end = min(start + l1_tris, F)
      sub = idx[start:end]
      idx[start:end] = sub[np.argsort(mcodes[sub], kind="stable")]
  else:
    for start in range(0, F, l1_tris):
      _split_recursive(idx[start : min(start + l1_tris, F)], leaf_size)
  return idx


def build_wide_bvh_mesh_item(
    mesh_id: int,
    md: Dict[str, Any],
    build_method: str = "spatial_median",
) -> WideBVHMeshItem:
  """Build pointer-free 3-level Wide-BVH (L=128 leaves, BVH-64 L1/L2) in low-RAM streaming fashion."""
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

  leaf_bank = np.zeros((n_leaves, 24, LEAF_SIZE), dtype=np.float32)

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

  _pack_vec3(10, vnorm[faces[:, 0]])
  _pack_vec3(13, vnorm[faces[:, 1]])
  _pack_vec3(16, vnorm[faces[:, 2]])

  # Compute tight leaf AABBs vectorized
  v0_l = leaf_bank[:, 0:3, :]
  v1_l = v0_l + leaf_bank[:, 3:6, :]
  v2_l = v0_l + leaf_bank[:, 6:9, :]
  pts_min = np.minimum(v0_l, np.minimum(v1_l, v2_l))
  pts_max = np.maximum(v0_l, np.maximum(v1_l, v2_l))
  tv_mask = leaf_bank[:, 9:10, :] > 0.5
  leaf_min = np.min(np.where(tv_mask, pts_min, 1e20), axis=2) - 1e-4
  leaf_max = np.max(np.where(tv_mask, pts_max, -1e20), axis=2) + 1e-4

  # Level 1 (BVH-64 over leaves)
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
  lval_g = lval_p.reshape(n_l1, BRANCH_FACTOR)

  l0_aabbs[:, 0:3, :BRANCH_FACTOR] = lmin_g
  l0_aabbs[:, 3:6, :BRANCH_FACTOR] = lmax_g
  l0_aabbs[:, 6, :BRANCH_FACTOR] = lval_g

  l1_min = np.min(lmin_g, axis=2)
  l1_max = np.max(lmax_g, axis=2)

  # Level 2 (BVH-64 over L1 nodes, supports up to 64 L2 roots = 33.5M tris/mesh)
  n_l2 = max(1, (n_l1 + BRANCH_FACTOR - 1) // BRANCH_FACTOR)
  assert n_l2 <= BRANCH_FACTOR, f"n_l2={n_l2} exceeds 64 L2 roots (>33.5M tris in one mesh)"
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
  m1val_g = m1val_p.reshape(n_l2, BRANCH_FACTOR)

  l2_aabbs[:, 0:3, :BRANCH_FACTOR] = m1min_g
  l2_aabbs[:, 3:6, :BRANCH_FACTOR] = m1max_g
  l2_aabbs[:, 6, :BRANCH_FACTOR] = m1val_g

  l2_min = np.min(m1min_g, axis=2)  # (n_l2, 3)
  l2_max = np.max(m1max_g, axis=2)  # (n_l2, 3)

  l2_roots_aabb = np.zeros((8, 128), dtype=np.float32)
  l2_roots_aabb[0:3, :] = 1e20
  l2_roots_aabb[3:6, :] = -1e20
  l2_roots_aabb[0:3, :n_l2] = l2_min.T
  l2_roots_aabb[3:6, :n_l2] = l2_max.T
  l2_roots_aabb[6, :n_l2] = 1.0

  obj_min = np.min(l2_min, axis=0)
  obj_max = np.max(l2_max, axis=0)

  return WideBVHMeshItem(
      mesh_id=mesh_id,
      name=str(md.get("name", f"mesh_{mesh_id}")),
      n_tris_orig=F,
      n_leaves=n_leaves,
      n_l1=n_l1,
      n_l2=n_l2,
      obj_aabb_min=obj_min.astype(np.float32),
      obj_aabb_max=obj_max.astype(np.float32),
      l2_roots_aabb=l2_roots_aabb,
      l2_aabbs=l2_aabbs,
      l0_aabbs=l0_aabbs,
      leaf_bank=leaf_bank,
      rgba=rgba,
  )


def build_wide_bvh_scene_spec(
    mesh_dicts: Sequence[Dict[str, Any]],
    height: int = 64,
    width: int = 64,
    cam_fovy: float = 42.0,
    znear: float = 0.05,
    zfar: float = 50.0,
    light_pos: Tuple[float, float, float] = (0.65, 0.25, 1.85),
    light_ambient: Tuple[float, float, float] = (0.28, 0.30, 0.34),
    light_diffuse: Tuple[float, float, float] = (0.85, 0.83, 0.80),
    light_specular: Tuple[float, float, float] = (0.30, 0.30, 0.30),
    sky_rgb1: Tuple[float, float, float] = (0.28, 0.36, 0.48),
    sky_rgb2: Tuple[float, float, float] = (0.10, 0.14, 0.20),
    use_shadows: bool = True,
    build_method: str = "spatial_median",
) -> WideBVHSceneSpec:
  items = tuple(build_wide_bvh_mesh_item(mid, md, build_method=build_method) for mid, md in enumerate(mesh_dicts))
  return WideBVHSceneSpec(
      height=int(height),
      width=int(width),
      cam_fovy=float(cam_fovy),
      cam_projection=0,
      cam_sensorsize=(0.0, 0.0),
      cam_intrinsic=(0.0, 0.0, 0.0, 0.0),
      znear=float(znear),
      zfar=float(zfar),
      meshes=items,
      light_pos=np.asarray(light_pos, dtype=np.float32),
      light_ambient=light_ambient,
      light_diffuse=light_diffuse,
      light_specular=light_specular,
      sky_rgb1=sky_rgb1,
      sky_rgb2=sky_rgb2,
      use_shadows=bool(use_shadows),
  )


def to_device_arrays(spec: WideBVHSceneSpec) -> Tuple[Tuple[jax.Array, ...], ...]:
  """Pre-upload static Wide-BVH buffers onto TPU HBM once outside the render loop."""
  dev_items = []
  for item in spec.meshes:
    dev_items.append((
        jnp.asarray(item.l2_roots_aabb, dtype=jnp.float32),
        jnp.asarray(item.l2_aabbs, dtype=jnp.float32),
        jnp.asarray(item.l0_aabbs, dtype=jnp.float32),
        jnp.asarray(item.leaf_bank, dtype=jnp.float32),
    ))
  return tuple(dev_items)


def _pack_rays_tiled(
    ro: jax.Array, rd: jax.Array, tl: jax.Array, H: int, W: int
) -> Tuple[jax.Array, int, bool]:
  """Reorder (B, H*W, ...) into spatially coherent 8x16 (128-ray) screen tiles -> (B*n_tiles*8, 128)."""
  B, R, _ = ro.shape
  can_2d_tile = (H % TILE_H == 0) and (W % TILE_W == 0)
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
    pad = R_pad - R
    ro_t = jnp.pad(ro, ((0, 0), (0, pad), (0, 0))).reshape(B * n_tiles, BLOCK_R, 3)
    rd_t = jnp.pad(rd, ((0, 0), (0, pad), (0, 0)), constant_values=1.0 / np.sqrt(3.0)).reshape(B * n_tiles, BLOCK_R, 3)
    tl_t = jnp.pad(tl, ((0, 0), (0, pad)), constant_values=0.0).reshape(B * n_tiles, BLOCK_R, 1)

  rays_8 = jnp.concatenate([ro_t, rd_t, tl_t, jnp.zeros_like(tl_t)], axis=-1)
  rays_2d = rays_8.transpose(0, 2, 1).reshape(B * n_tiles * 8, BLOCK_R)
  return rays_2d, n_tiles, can_2d_tile


def _unpack_out_tiled(
    out_2d: jax.Array, B: int, H: int, W: int, n_tiles: int, can_2d_tile: bool, return_stats: bool = False
) -> Tuple[jax.Array, jax.Array, Optional[jax.Array]]:
  R = H * W
  out_t = out_2d.reshape(B, n_tiles, 8, BLOCK_R).transpose(0, 1, 3, 2)
  stats = out_t[:, :, 0, 4] if return_stats else None  # (B, n_tiles) leaf visits per tile
  if can_2d_tile:
    ty, tx = H // TILE_H, W // TILE_W
    out_hw = (
        out_t.reshape(B, ty, tx, TILE_H, TILE_W, 8)
        .transpose(0, 1, 3, 2, 4, 5)
        .reshape(B, R, 8)
    )
  else:
    out_hw = out_t.reshape(B, n_tiles * BLOCK_R, 8)[:, :R, :]
  return out_hw[..., 0], out_hw[..., 1:4], stats


def _traverse_bvh_in_kernel(
    rox, roy, roz, irx, iry, irz, rdx, rdy, rdz,
    init_bd, init_nx, init_ny, init_nz,
    l2_roots_blk, l2_hbm_ref, l0_hbm_ref, leaf_hbm_ref,
    l2_vmem, l0_vmem, leaf_vmem, sem_l2, sem_l0, sem_leaf,
    cols_iota_64, leaf_rows: int, any_hit: bool, track_stats: bool,
):
  """3-Level front-to-back Wide-BVH traversal inside Mosaic Pallas kernel (O(1) MLIR size)."""
  l2_bx0 = l2_roots_blk[0:1, :BRANCH_FACTOR]
  l2_by0 = l2_roots_blk[1:2, :BRANCH_FACTOR]
  l2_bz0 = l2_roots_blk[2:3, :BRANCH_FACTOR]
  l2_bx1 = l2_roots_blk[3:4, :BRANCH_FACTOR]
  l2_by1 = l2_roots_blk[4:5, :BRANCH_FACTOR]
  l2_bz1 = l2_roots_blk[5:6, :BRANCH_FACTOR]
  l2_valid = l2_roots_blk[6:7, :BRANCH_FACTOR] > 0.5

  tx0_2, tx1_2 = (l2_bx0 - rox) * irx, (l2_bx1 - rox) * irx
  ty0_2, ty1_2 = (l2_by0 - roy) * iry, (l2_by1 - roy) * iry
  tz0_2, tz1_2 = (l2_bz0 - roz) * irz, (l2_bz1 - roz) * irz
  tmin2_pos = jnp.maximum(
      0.0, jnp.maximum(jnp.minimum(tx0_2, tx1_2), jnp.maximum(jnp.minimum(ty0_2, ty1_2), jnp.minimum(tz0_2, tz1_2)))
  )
  tmax2 = jnp.minimum(jnp.maximum(tx0_2, tx1_2), jnp.minimum(jnp.maximum(ty0_2, ty1_2), jnp.maximum(tz0_2, tz1_2)))
  tmin2_eff = jnp.where((tmax2 >= tmin2_pos) & l2_valid, tmin2_pos, 1e20)
  l2_unvis_f = jnp.where(l2_valid, 1.0, 0.0)
  l2_tmin = jnp.min(jnp.where(tmin2_eff < init_bd, tmin2_eff, 1e20), axis=0, keepdims=True)

  def _l2_cond(c2):
    t2, _, bd2, *_ = c2
    return jnp.min(t2) < jnp.max(bd2)

  def _l2_body(c2):
    t2, unvis2_f, bd2, nx2, ny2, nz2, lcnt2 = c2
    best_r = jnp.argmin(t2, axis=1)[0]
    unvis2_next_f = jnp.where(cols_iota_64 == best_r, 0.0, unvis2_f)

    cp2 = pltpu.make_async_copy(l2_hbm_ref.at[best_r, :, :], l2_vmem, sem_l2)
    cp2.start()
    cp2.wait()
    l2_blk = l2_vmem[...]

    l1_bx0 = l2_blk[0:1, :BRANCH_FACTOR]
    l1_by0 = l2_blk[1:2, :BRANCH_FACTOR]
    l1_bz0 = l2_blk[2:3, :BRANCH_FACTOR]
    l1_bx1 = l2_blk[3:4, :BRANCH_FACTOR]
    l1_by1 = l2_blk[4:5, :BRANCH_FACTOR]
    l1_bz1 = l2_blk[5:6, :BRANCH_FACTOR]
    l1_valid = l2_blk[6:7, :BRANCH_FACTOR] > 0.5

    tx0_1, tx1_1 = (l1_bx0 - rox) * irx, (l1_bx1 - rox) * irx
    ty0_1, ty1_1 = (l1_by0 - roy) * iry, (l1_by1 - roy) * iry
    tz0_1, tz1_1 = (l1_bz0 - roz) * irz, (l1_bz1 - roz) * irz
    tmin1_pos = jnp.maximum(
        0.0, jnp.maximum(jnp.minimum(tx0_1, tx1_1), jnp.maximum(jnp.minimum(ty0_1, ty1_1), jnp.minimum(tz0_1, tz1_1)))
    )
    tmax1 = jnp.minimum(jnp.maximum(tx0_1, tx1_1), jnp.minimum(jnp.maximum(ty0_1, ty1_1), jnp.maximum(tz0_1, tz1_1)))
    tmin1_eff = jnp.where((tmax1 >= tmin1_pos) & l1_valid, tmin1_pos, 1e20)
    l1_unvis_f = jnp.where(l1_valid, 1.0, 0.0)
    l1_tmin = jnp.min(jnp.where(tmin1_eff < bd2, tmin1_eff, 1e20), axis=0, keepdims=True)

    def _l1_cond(c1):
      t1, _, bd1, *_ = c1
      return jnp.min(t1) < jnp.max(bd1)

    def _l1_body(c1):
      t1, unvis1_f, bd1, nx1, ny1, nz1, lcnt1 = c1
      best_c1 = jnp.argmin(t1, axis=1)[0]
      unvis1_next_f = jnp.where(cols_iota_64 == best_c1, 0.0, unvis1_f)
      l1_idx = best_r * BRANCH_FACTOR + best_c1

      cp0 = pltpu.make_async_copy(l0_hbm_ref.at[l1_idx, :, :], l0_vmem, sem_l0)
      cp0.start()
      cp0.wait()
      l0_blk = l0_vmem[...]

      l0_bx0 = l0_blk[0:1, :BRANCH_FACTOR]
      l0_by0 = l0_blk[1:2, :BRANCH_FACTOR]
      l0_bz0 = l0_blk[2:3, :BRANCH_FACTOR]
      l0_bx1 = l0_blk[3:4, :BRANCH_FACTOR]
      l0_by1 = l0_blk[4:5, :BRANCH_FACTOR]
      l0_bz1 = l0_blk[5:6, :BRANCH_FACTOR]
      l0_valid = l0_blk[6:7, :BRANCH_FACTOR] > 0.5

      tx0_0, tx1_0 = (l0_bx0 - rox) * irx, (l0_bx1 - rox) * irx
      ty0_0, ty1_0 = (l0_by0 - roy) * iry, (l0_by1 - roy) * iry
      tz0_0, tz1_0 = (l0_bz0 - roz) * irz, (l0_bz1 - roz) * irz
      tmin0_pos = jnp.maximum(
          0.0, jnp.maximum(jnp.minimum(tx0_0, tx1_0), jnp.maximum(jnp.minimum(ty0_0, ty1_0), jnp.minimum(tz0_0, tz1_0)))
      )
      tmax0 = jnp.minimum(jnp.maximum(tx0_0, tx1_0), jnp.minimum(jnp.maximum(ty0_0, ty1_0), jnp.maximum(tz0_0, tz1_0)))
      tmin0_eff = jnp.where((tmax0 >= tmin0_pos) & l0_valid, tmin0_pos, 1e20)
      l0_unvis_f = jnp.where(l0_valid, 1.0, 0.0)
      l0_tmin = jnp.min(jnp.where(tmin0_eff < bd1, tmin0_eff, 1e20), axis=0, keepdims=True)

      def _l0_cond(c0):
        t0, _, bd0, *_ = c0
        return jnp.min(t0) < jnp.max(bd0)

      def _l0_body(c0):
        t0, unvis0_f, bd0, nx0, ny0, nz0, lcnt0 = c0
        best_c0 = jnp.argmin(t0, axis=1)[0]
        unvis0_next_f = jnp.where(cols_iota_64 == best_c0, 0.0, unvis0_f)
        leaf_idx = l1_idx * BRANCH_FACTOR + best_c0

        cp1 = pltpu.make_async_copy(leaf_hbm_ref.at[leaf_idx, :leaf_rows, :], leaf_vmem, sem_leaf)
        cp1.start()
        cp1.wait()
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

        sx = rox - v0x
        sy = roy - v0y
        sz = roz - v0z
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
          t0_next = jnp.min(
              jnp.where((tmin0_eff < new_bd) & (unvis0_next_f > 0.5), tmin0_eff, 1e20),
              axis=0,
              keepdims=True,
          )
          return (t0_next, unvis0_next_f, new_bd, nx0, ny0, nz0, lcnt_next)

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
          new_bd = jnp.where(imp, min_t, bd0)
          t0_n = jnp.min(
              jnp.where((tmin0_eff < new_bd) & (unvis0_next_f > 0.5), tmin0_eff, 1e20),
              axis=0,
              keepdims=True,
          )
          return (
              t0_n,
              new_bd,
              jnp.where(imp, lnx, nx0),
              jnp.where(imp, lny, ny0),
              jnp.where(imp, lnz, nz0),
          )

        def _on_miss(_):
          return (jnp.where(unvis0_next_f > 0.5, t0, 1e20), bd0, nx0, ny0, nz0)

        t0_next, new_bd, new_nx, new_ny, new_nz = jax.lax.cond(any_imp, _on_hit, _on_miss, None)
        return (t0_next, unvis0_next_f, new_bd, new_nx, new_ny, new_nz, lcnt_next)

      _, _, n_bd1, n_nx1, n_ny1, n_nz1, n_lcnt1 = jax.lax.while_loop(
          _l0_cond, _l0_body, (l0_tmin, l0_unvis_f, bd1, nx1, ny1, nz1, lcnt1)
      )
      t1_next = jnp.min(
          jnp.where((tmin1_eff < n_bd1) & (unvis1_next_f > 0.5), tmin1_eff, 1e20),
          axis=0,
          keepdims=True,
      )
      return (t1_next, unvis1_next_f, n_bd1, n_nx1, n_ny1, n_nz1, n_lcnt1)

    _, _, n_bd2, n_nx2, n_ny2, n_nz2, n_lcnt2 = jax.lax.while_loop(
        _l1_cond, _l1_body, (l1_tmin, l1_unvis_f, bd2, nx2, ny2, nz2, lcnt2)
    )
    t2_next = jnp.min(
        jnp.where((tmin2_eff < n_bd2) & (unvis2_next_f > 0.5), tmin2_eff, 1e20),
        axis=0,
        keepdims=True,
    )
    return (t2_next, unvis2_next_f, n_bd2, n_nx2, n_ny2, n_nz2, n_lcnt2)

  init_cnt = jnp.float32(0.0)
  _, _, out_bd, out_nx, out_ny, out_nz, out_cnt = jax.lax.while_loop(
      _l2_cond, _l2_body, (l2_tmin, l2_unvis_f, init_bd, init_nx, init_ny, init_nz, init_cnt)
  )
  return out_bd, out_nx, out_ny, out_nz, out_cnt


def _pallas_wide_bvh_raycast(
    item: WideBVHMeshItem,
    l2_roots_dev: jax.Array,
    l2_dev: jax.Array,
    l0_dev: jax.Array,
    leaf_dev: jax.Array,
    ro_loc: jax.Array,
    rd_loc: jax.Array,
    t_limit: jax.Array,
    H: int,
    W: int,
    any_hit: bool = False,
    interpret: bool = False,
    return_stats: bool = False,
) -> Tuple[jax.Array, jax.Array, Optional[jax.Array]]:
  """3-Level Wide-BVH Mosaic Pallas kernel with dynamic HBM->VMEM DMA & Early-Z pruning."""
  B = ro_loc.shape[0]
  rays_2d, n_tiles, can_2d_tile = _pack_rays_tiled(ro_loc, rd_loc, t_limit, H, W)
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

    bnx = jnp.zeros((BLOCK_R, 1), jnp.float32)
    bny = jnp.zeros((BLOCK_R, 1), jnp.float32)
    bnz = jnp.zeros((BLOCK_R, 1), jnp.float32)

    cbd, cnx, cny, cnz, lcnt = _traverse_bvh_in_kernel(
        rox, roy, roz, irx, iry, irz, rdx, rdy, rdz,
        bd, bnx, bny, bnz,
        l2_roots_blk, l2_hbm_ref, l0_hbm_ref, leaf_hbm_ref,
        l2_vmem, l0_vmem, leaf_vmem, sem_l2, sem_l0, sem_leaf,
        cols_iota_64, leaf_rows, any_hit, return_stats,
    )

    z = jnp.zeros((1, BLOCK_R), jnp.float32)
    cnt_row = jnp.broadcast_to(lcnt, (1, BLOCK_R)) if return_stats else z
    out_ref[...] = jnp.concatenate(
        [
            jnp.transpose(cbd, (1, 0)),
            jnp.transpose(cnx, (1, 0)),
            jnp.transpose(cny, (1, 0)),
            jnp.transpose(cnz, (1, 0)),
            cnt_row, z, z, z,
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
      compiler_params=None if interpret else pltpu.CompilerParams(dimension_semantics=("parallel",)),
      interpret=interpret,
  )(rays_2d, l2_roots_dev, l2_dev, l0_dev, leaf_dev)

  return _unpack_out_tiled(out_2d, B, H, W, n_tiles, can_2d_tile, return_stats=return_stats)


def _shade_mesh_hits(spec, ray_orig, ray_dir, neg_z, best_d, best_gid, best_nrm, vis):
  B, R, _ = ray_orig.shape
  H, W = spec.height, spec.width
  hit = best_gid >= 0
  best_d = jnp.where(hit, best_d, -1.0)
  best_nrm = jnp.where(hit[..., None], best_nrm, 0.0)
  vdir = -ray_dir
  hit_pos = ray_orig + jnp.where(hit, best_d, 0.0)[..., None] * ray_dir
  rgba_tbl = jnp.asarray([it.rgba[:3] for it in spec.meshes], dtype=jnp.float32)
  albedo = rgba_tbl[jnp.maximum(best_gid, 0)]

  l_pos = jnp.asarray(spec.light_pos, dtype=jnp.float32)
  delta_l = l_pos[None, None, :] - hit_pos
  dist_l = jnp.linalg.norm(delta_l, axis=-1)
  L = delta_l / jnp.maximum(dist_l[..., None], 1e-12)
  ndotl = jnp.maximum(0.0, jnp.sum(best_nrm * L, axis=-1))

  l_amb = jnp.asarray(spec.light_ambient, dtype=jnp.float32)
  l_diff = jnp.asarray(spec.light_diffuse, dtype=jnp.float32)
  l_spec = jnp.asarray(spec.light_specular, dtype=jnp.float32)
  H_vec = (L + vdir) / jnp.maximum(jnp.linalg.norm(L + vdir, axis=-1, keepdims=True), 1e-12)
  spec_w = (jnp.maximum(0.0, jnp.sum(best_nrm * H_vec, axis=-1)) ** 32.0) * 0.25 * vis

  direct_rgb = jnp.clip(
      albedo * (l_amb[None, None, :] + l_diff[None, None, :] * (ndotl * vis)[..., None]) + l_spec[None, None, :] * spec_w[..., None],
      0.0,
      1.0,
  )
  sky_t = jnp.clip(0.5 * (ray_dir[..., 2] + 1.0), 0.0, 1.0)[..., None]
  sky_rgb = (
      sky_t * jnp.asarray(spec.sky_rgb1, dtype=jnp.float32)[None, None, :]
      + (1.0 - sky_t) * jnp.asarray(spec.sky_rgb2, dtype=jnp.float32)[None, None, :]
  )
  rgb = jnp.where(hit[..., None], direct_rgb, sky_rgb)
  inv_cos = (1.0 / neg_z)[None, :]
  euclid_d = jnp.where(hit, best_d + float(spec.znear) * inv_cos, -1.0)
  z_d = jnp.where(hit, float(spec.znear) + best_d * neg_z[None, :], -1.0)

  return GBuffer(
      rgb=rgb.reshape(B, H, W, 3),
      albedo=jnp.where(hit[..., None], albedo, sky_rgb).reshape(B, H, W, 3),
      depth=euclid_d.reshape(B, H, W),
      z_depth=z_d.reshape(B, H, W),
      normal=best_nrm.reshape(B, H, W, 3),
      view_dir=vdir.reshape(B, H, W, 3),
      world_pos=jnp.where(hit[..., None], hit_pos, 0.0).reshape(B, H, W, 3),
      geom_id=best_gid.reshape(B, H, W),
      direct_shadow=jnp.where(hit, vis, 1.0).reshape(B, H, W),
      direct_rgb=rgb.reshape(B, H, W, 3),
  )


def render_wide_bvh_batch(
    spec: WideBVHSceneSpec,
    dev_arrays: Tuple[Tuple[jax.Array, ...], ...],
    obj_xpos: jax.Array,
    obj_xmat: jax.Array,
    cam_xpos: jax.Array,
    cam_xmat: jax.Array,
    interpret: bool = False,
    return_stats: bool = False,
):
  """Zero-copy batched 3-level Wide-BVH mesh raytracer consuming dynamic rigid poses."""
  ray_orig, ray_dir, neg_z = _compute_camera_rays_ondevice(spec, cam_xpos, cam_xmat)
  B, R, _ = ray_orig.shape
  H, W = spec.height, spec.width
  max_dist = jnp.broadcast_to(((spec.zfar - spec.znear) / neg_z)[None, :], (B, R))

  best_d = max_dist
  best_gid = jnp.full((B, R), -1, dtype=jnp.int32)
  best_nrm = jnp.zeros((B, R, 3), dtype=jnp.float32)
  total_leaf_visits = jnp.zeros((B,), dtype=jnp.float32)

  for m_idx, item in enumerate(spec.meshes):
    l2_roots_dev, l2_dev, l0_dev, leaf_dev = dev_arrays[m_idx]
    p = obj_xpos[:, m_idx, :]
    rmat = obj_xmat[:, m_idx, :, :]
    ro_loc = jnp.einsum("bji,brj->bri", rmat, ray_orig - p[:, None, :], precision=jax.lax.Precision.HIGHEST)
    rd_loc = jnp.einsum("bji,brj->bri", rmat, ray_dir, precision=jax.lax.Precision.HIGHEST)

    d_m, n_loc_m, st_m = _pallas_wide_bvh_raycast(
        item, l2_roots_dev, l2_dev, l0_dev, leaf_dev, ro_loc, rd_loc, best_d, H, W,
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
    for m_idx, item in enumerate(spec.meshes):
      l2_roots_dev, l2_dev, l0_dev, leaf_dev = dev_arrays[m_idx]
      p = obj_xpos[:, m_idx, :]
      rmat = obj_xmat[:, m_idx, :, :]
      ro_l = jnp.einsum("bji,brj->bri", rmat, sh_o - p[:, None, :], precision=jax.lax.Precision.HIGHEST)
      rd_l = jnp.einsum("bji,brj->bri", rmat, L, precision=jax.lax.Precision.HIGHEST)
      sh_lim_rem = jnp.where(occ, 0.0, sh_lim)
      d_sh, _, st_sh = _pallas_wide_bvh_raycast(
          item, l2_roots_dev, l2_dev, l0_dev, leaf_dev, ro_l, rd_l, sh_lim_rem, H, W,
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


def to_replicated_device_arrays(spec: WideBVHSceneSpec, mesh: Mesh) -> Tuple[Tuple[jax.Array, ...], ...]:
  """Upload static Wide-BVH buffers directly as P() replicated arrays across a multi-chip mesh."""
  rep = NamedSharding(mesh, P())
  dev_items = []
  for item in spec.meshes:
    dev_items.append((
        jax.device_put(np.asarray(item.l2_roots_aabb, dtype=np.float32), rep),
        jax.device_put(np.asarray(item.l2_aabbs, dtype=np.float32), rep),
        jax.device_put(np.asarray(item.l0_aabbs, dtype=np.float32), rep),
        jax.device_put(np.asarray(item.leaf_bank, dtype=np.float32), rep),
    ))
  return tuple(dev_items)


def make_sharded_wide_bvh_renderer(
    spec: WideBVHSceneSpec,
    devices: Sequence[jax.Device],
    interpret: bool = False,
    dev_arrays: Optional[Tuple[Tuple[jax.Array, ...], ...]] = None,
):
  """Create N-chip shard_map render function with replicated static BVH in HBM."""
  mesh = Mesh(np.array(devices), ("data",))
  if dev_arrays is None:
    dev_arrays = to_replicated_device_arrays(spec, mesh)

  def _shard_fn(da, oxp, oxm, cxp, cxm):
    return render_wide_bvh_batch(spec, da, oxp, oxm, cxp, cxm, interpret=interpret)

  sharded = shard_map(
      _shard_fn,
      mesh=mesh,
      in_specs=(P(), P("data"), P("data"), P("data"), P("data")),
      out_specs=P("data"),
      check_rep=False,
  )
  return jax.jit(lambda oxp, oxm, cxp, cxm: sharded(dev_arrays, oxp, oxm, cxp, cxm)), mesh


# ==============================================================================
# Track 2B: Zero-Copy TLAS + BLAS Rigid Instancing (10M+ to 50M+ Effective Tris)
# ==============================================================================

def _pallas_blas_instanced_raycast(
    item: WideBVHMeshItem,
    l2_roots_dev: jax.Array,
    l2_dev: jax.Array,
    l0_dev: jax.Array,
    leaf_dev: jax.Array,
    ray_o_world: jax.Array,     # (B, R, 3)
    ray_d_world: jax.Array,     # (B, R, 3)
    t_limit: jax.Array,         # (B, R)
    inst_xpos: jax.Array,       # (B, K_inst, 3)
    inst_xmat: jax.Array,       # (B, K_inst, 3, 3)
    H: int,
    W: int,
    max_active_per_tile: int = 16,
    any_hit: bool = False,
    interpret: bool = False,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
  """TLAS + Shared-BLAS raycast across K_inst dynamic instances of `item` (<= 128)."""
  B, R, _ = ray_o_world.shape
  K_inst = int(inst_xpos.shape[1])
  assert K_inst <= 128, f"K_inst={K_inst} exceeds 128 lane limit"

  rays_2d, n_tiles, can_2d_tile = _pack_rays_tiled(ray_o_world, ray_d_world, t_limit, H, W)
  total_tiles = B * n_tiles

  # 1. TLAS World-Space OBB -> AABB Computation (Arvo)
  c_loc = jnp.asarray(0.5 * (item.obj_aabb_min + item.obj_aabb_max), dtype=jnp.float32)
  h_loc = jnp.asarray(0.5 * (item.obj_aabb_max - item.obj_aabb_min), dtype=jnp.float32)
  c_w = inst_xpos + jnp.einsum("bkij,j->bki", inst_xmat, c_loc, precision=jax.lax.Precision.HIGHEST)
  h_w = jnp.einsum("bkij,j->bki", jnp.abs(inst_xmat), h_loc, precision=jax.lax.Precision.HIGHEST)
  w_min = c_w - h_w  # (B, K_inst, 3)
  w_max = c_w + h_w

  # 2. Per-Tile TLAS Ray-vs-Instance-AABB Culling + Front-to-Back Sorting in XLA
  # Reshape world rays into (B, n_tiles, 128, 3)
  ro_t = rays_2d.reshape(B, n_tiles, 8, BLOCK_R)[:, :, 0:3, :].transpose(0, 1, 3, 2)
  rd_t = rays_2d.reshape(B, n_tiles, 8, BLOCK_R)[:, :, 3:6, :].transpose(0, 1, 3, 2)
  tl_t = rays_2d.reshape(B, n_tiles, 8, BLOCK_R)[:, :, 6, :]  # (B, n_tiles, 128)
  ir_t = jnp.where(jnp.abs(rd_t) > 1e-12, 1.0 / jnp.where(jnp.abs(rd_t) > 1e-12, rd_t, 1.0), 1e12)

  # (B, n_tiles, 128, K_inst)
  t0 = (w_min[:, None, None, :, :] - ro_t[:, :, :, None, :]) * ir_t[:, :, :, None, :]
  t1 = (w_max[:, None, None, :, :] - ro_t[:, :, :, None, :]) * ir_t[:, :, :, None, :]
  tmin_w = jnp.maximum(0.0, jnp.max(jnp.minimum(t0, t1), axis=-1))
  tmax_w = jnp.min(jnp.maximum(t0, t1), axis=-1)
  hit_w = (tmax_w >= tmin_w) & (tmin_w < tl_t[:, :, :, None])
  tile_inst_tmin = jnp.min(jnp.where(hit_w, tmin_w, 1e20), axis=2)  # (B, n_tiles, K_inst)

  K_act = min(max_active_per_tile, K_inst)
  neg_vals, top_idx = jax.lax.top_k(-tile_inst_tmin, K_act)  # (B, n_tiles, K_act)
  act_ids = jnp.where((-neg_vals) < 1e19, top_idx, -1).astype(jnp.int32)  # (B, n_tiles, K_act)

  # Pack per-world inverse transforms (pos + rmat) into (B, 16, 128) -> repeat/index per world!
  # Pack into (B * 16, 128): rows 0..2 = pos, rows 3..11 = rmat (row-major 3x3 for world->local & local->world)
  poses_b = jnp.zeros((B, 16, 128), dtype=jnp.float32)
  poses_b = poses_b.at[:, 0:3, :K_inst].set(inst_xpos.transpose(0, 2, 1))
  poses_b = poses_b.at[:, 3:12, :K_inst].set(inst_xmat.reshape(B, K_inst, 9).transpose(0, 2, 1))
  poses_2d = poses_b.reshape(B * 16, 128)

  # Pack active instance IDs into (total_tiles * 8, 128) float32 (-1.0 sentinel, zero SMEM limit at 4K!)
  act_blk = jnp.full((total_tiles, 8, 128), -1.0, dtype=jnp.float32)
  act_blk = act_blk.at[:, 0, :K_act].set(act_ids.reshape(total_tiles, K_act).astype(jnp.float32))
  act_2d = act_blk.reshape(total_tiles * 8, 128)

  leaf_rows = 16 if any_hit else 24

  def _inst_kernel(
      ray_ref, act_ref, pose_ref, l2_roots_ref, l2_hbm_ref, l0_hbm_ref, leaf_hbm_ref, out_ref,
      l2_vmem, l0_vmem, leaf_vmem, sem_l2, sem_l0, sem_leaf,
  ):
    cols_iota_64 = jax.lax.broadcasted_iota(jnp.int32, (1, BRANCH_FACTOR), 1)
    cols_iota_128 = jax.lax.broadcasted_iota(jnp.int32, (1, 128), 1)

    rt = ray_ref[...]       # (8, 128)
    act_r = act_ref[0:1, :] # (1, 128) float32
    poses_w = pose_ref[...] # (16, 128) float32
    l2_roots_blk = l2_roots_ref[...]

    w_rox = jnp.transpose(rt[0:1, :], (1, 0))
    w_roy = jnp.transpose(rt[1:2, :], (1, 0))
    w_roz = jnp.transpose(rt[2:3, :], (1, 0))
    w_rdx = jnp.transpose(rt[3:4, :], (1, 0))
    w_rdy = jnp.transpose(rt[4:5, :], (1, 0))
    w_rdz = jnp.transpose(rt[5:6, :], (1, 0))
    bd = jnp.transpose(rt[6:7, :], (1, 0))

    bnx = jnp.zeros((BLOCK_R, 1), jnp.float32)
    bny = jnp.zeros((BLOCK_R, 1), jnp.float32)
    bnz = jnp.zeros((BLOCK_R, 1), jnp.float32)
    bid = jnp.full((BLOCK_R, 1), -1.0, jnp.float32)

    def _read_act_id(k_idx):
      return jnp.sum(jnp.where(cols_iota_128 == k_idx, act_r, 0.0))

    def _inst_cond(ci):
      k, inst_f, *_ = ci
      return (k < K_act) & (inst_f >= 0.0)

    def _inst_body(ci):
      k, inst_f, bd_i, nx_i, ny_i, nz_i, id_i = ci
      inst_id = inst_f.astype(jnp.int32)
      cmask = cols_iota_128 == inst_id  # (1, 128)

      # Extract pose (pos, rmat) for inst_id via masked lane sum (1 VPU op!)
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

      # Transform world rays -> instance local frame (R^T * (o - p), R^T * d)
      dx_p = w_rox - px
      dy_p = w_roy - py
      dz_p = w_roz - pz
      rox = r00 * dx_p + r10 * dy_p + r20 * dz_p
      roy = r01 * dx_p + r11 * dy_p + r21 * dz_p
      roz = r02 * dx_p + r12 * dy_p + r22 * dz_p
      rdx = r00 * w_rdx + r10 * w_rdy + r20 * w_rdz
      rdy = r01 * w_rdx + r11 * w_rdy + r21 * w_rdz
      rdz = r02 * w_rdx + r12 * w_rdy + r22 * w_rdz

      irx = jnp.where(jnp.abs(rdx) > 1e-12, 1.0 / jnp.where(jnp.abs(rdx) > 1e-12, rdx, 1.0), 1e12)
      iry = jnp.where(jnp.abs(rdy) > 1e-12, 1.0 / jnp.where(jnp.abs(rdy) > 1e-12, rdy, 1.0), 1e12)
      irz = jnp.where(jnp.abs(rdz) > 1e-12, 1.0 / jnp.where(jnp.abs(rdz) > 1e-12, rdz, 1.0), 1e12)

      z_loc = jnp.zeros((BLOCK_R, 1), jnp.float32)
      new_bd, lnx, lny, lnz, _ = _traverse_bvh_in_kernel(
          rox, roy, roz, irx, iry, irz, rdx, rdy, rdz,
          bd_i, z_loc, z_loc, z_loc,
          l2_roots_blk, l2_hbm_ref, l0_hbm_ref, leaf_hbm_ref,
          l2_vmem, l0_vmem, leaf_vmem, sem_l2, sem_l0, sem_leaf,
          cols_iota_64, leaf_rows, any_hit, False,
      )
      imp = new_bd < bd_i
      k_next = k + 1
      inst_next_f = _read_act_id(k_next)
      if any_hit:
        return (k_next, inst_next_f, new_bd, nx_i, ny_i, nz_i, id_i)

      # Transform local normal -> world space (R * n_loc)
      wnx = r00 * lnx + r01 * lny + r02 * lnz
      wny = r10 * lnx + r11 * lny + r12 * lnz
      wnz = r20 * lnx + r21 * lny + r22 * lnz
      return (
          k_next,
          inst_next_f,
          new_bd,
          jnp.where(imp, wnx, nx_i),
          jnp.where(imp, wny, ny_i),
          jnp.where(imp, wnz, nz_i),
          jnp.where(imp, inst_f, id_i),
      )

    _, _, f_bd, f_nx, f_ny, f_nz, f_id = jax.lax.while_loop(
        _inst_cond, _inst_body, (0, _read_act_id(0), bd, bnx, bny, bnz, bid)
    )

    z = jnp.zeros((1, BLOCK_R), jnp.float32)
    out_ref[...] = jnp.concatenate(
        [
            jnp.transpose(f_bd, (1, 0)),
            jnp.transpose(f_nx, (1, 0)),
            jnp.transpose(f_ny, (1, 0)),
            jnp.transpose(f_nz, (1, 0)),
            jnp.transpose(f_id, (1, 0)),
            z, z, z,
        ],
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
              pl.BlockSpec((16, 128), lambda i: (i // n_tiles, 0)),
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
      compiler_params=None if interpret else pltpu.CompilerParams(dimension_semantics=("parallel",)),
      interpret=interpret,
  )(rays_2d, act_2d, poses_2d, l2_roots_dev, l2_dev, l0_dev, leaf_dev)

  # Unpack including row 4 (hit inst_id)
  out_t = out_2d.reshape(B, n_tiles, 8, BLOCK_R).transpose(0, 1, 3, 2)
  if can_2d_tile:
    ty, tx = H // TILE_H, W // TILE_W
    out_hw = (
        out_t.reshape(B, ty, tx, TILE_H, TILE_W, 8)
        .transpose(0, 1, 3, 2, 4, 5)
        .reshape(B, R, 8)
    )
  else:
    out_hw = out_t.reshape(B, n_tiles * BLOCK_R, 8)[:, :R, :]
  return out_hw[..., 0], out_hw[..., 1:4], out_hw[..., 4]


def render_wide_bvh_instanced_batch(
    spec: WideBVHSceneSpec,
    dev_arrays: Tuple[Tuple[jax.Array, ...], ...],
    inst_xpos_per_blas: Tuple[jax.Array, ...],   # M_blas tuples of (B, K_inst, 3)
    inst_xmat_per_blas: Tuple[jax.Array, ...],   # M_blas tuples of (B, K_inst, 3, 3)
    cam_xpos: jax.Array,
    cam_xmat: jax.Array,
    max_active_per_tile: int = 16,
    interpret: bool = False,
) -> GBuffer:
  """Zero-copy TLAS + BLAS instanced raytracer consuming dynamic poses across M_blas * K_inst instances."""
  ray_orig, ray_dir, neg_z = _compute_camera_rays_ondevice(spec, cam_xpos, cam_xmat)
  B, R, _ = ray_orig.shape
  H, W = spec.height, spec.width
  max_dist = jnp.broadcast_to(((spec.zfar - spec.znear) / neg_z)[None, :], (B, R))

  best_d = max_dist
  best_gid = jnp.full((B, R), -1, dtype=jnp.int32)
  best_nrm = jnp.zeros((B, R, 3), dtype=jnp.float32)

  for m_idx, item in enumerate(spec.meshes):
    l2_roots_dev, l2_dev, l0_dev, leaf_dev = dev_arrays[m_idx]
    d_m, n_w_m, _ = _pallas_blas_instanced_raycast(
        item, l2_roots_dev, l2_dev, l0_dev, leaf_dev,
        ray_orig, ray_dir, best_d,
        inst_xpos_per_blas[m_idx], inst_xmat_per_blas[m_idx],
        H, W, max_active_per_tile=max_active_per_tile, any_hit=False, interpret=interpret,
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
      d_sh, _, _ = _pallas_blas_instanced_raycast(
          item, l2_roots_dev, l2_dev, l0_dev, leaf_dev,
          sh_o, jnp.broadcast_to(L, ray_orig.shape), sh_lim_rem,
          inst_xpos_per_blas[m_idx], inst_xmat_per_blas[m_idx],
          H, W, max_active_per_tile=max_active_per_tile, any_hit=True, interpret=interpret,
      )
      occ = occ | (d_sh < sh_lim_rem)
    vis = jnp.where(occ, float(spec.shadow_light_fraction), 1.0)

  return _shade_mesh_hits(spec, ray_orig, ray_dir, neg_z, best_d, best_gid, best_nrm, vis)


def make_sharded_instanced_renderer(
    spec: WideBVHSceneSpec,
    devices: Sequence[jax.Device],
    max_active_per_tile: int = 16,
    interpret: bool = False,
    dev_arrays: Optional[Tuple[Tuple[jax.Array, ...], ...]] = None,
):
  mesh = Mesh(np.array(devices), ("data",))
  if dev_arrays is None:
    dev_arrays = to_replicated_device_arrays(spec, mesh)

  def _shard_fn(da, ixp, ixm, cxp, cxm):
    return render_wide_bvh_instanced_batch(
        spec, da, ixp, ixm, cxp, cxm,
        max_active_per_tile=max_active_per_tile, interpret=interpret,
    )

  sharded = shard_map(
      _shard_fn,
      mesh=mesh,
      in_specs=(P(), P("data"), P("data"), P("data"), P("data")),
      out_specs=P("data"),
      check_rep=False,
  )
  return jax.jit(lambda ixp, ixm, cxp, cxm: sharded(dev_arrays, ixp, ixm, cxp, cxm)), mesh
