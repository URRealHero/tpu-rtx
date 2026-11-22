"""TPU-Native Per-Pixel Bilinear UV-Textured 3-Level Wide-BVH-64 Mosaic Pallas Raytracer.

Supports arbitrary `.blend` scenes compiled via `blend_compiler.ScenePack`:
  - Single 3-level Wide-BVH-64 up to 33.5M triangles (`N_L2 <= 64, N_L1 <= 4096`).
  - Compact 24-row leaf bank: FP32 geometry + BF16-packed smooth vertex normals +
    FP32 barycentric UVs + per-triangle material ID + fallback albedo/roughness.
  - Ultra-fast `leaf_rows=10` (5 KB/leaf) geometry-only shadow `any_hit` kernel.
  - Per-pixel bilinear `tex_atlas` UV sampling + multi-light GGX/Diffuse/Emissive/Glass
    direct shading + scene-normalized 26-ch G-Buffer features for Stage 2 TPU-NRC.
"""

from __future__ import annotations

import dataclasses
import math
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
    _spatial_median_order,
)


def _pack_bf16_pair_to_f32(hi_f32: np.ndarray, lo_f32: np.ndarray) -> np.ndarray:
  hi_u32 = np.asarray(hi_f32, dtype=np.float32).view(np.uint32)
  lo_u32 = np.asarray(lo_f32, dtype=np.float32).view(np.uint32)
  hi_16 = ((hi_u32 + np.uint32(0x7FFF) + ((hi_u32 >> np.uint32(16)) & np.uint32(1))) >> np.uint32(16)) & np.uint32(0xFFFF)
  lo_16 = ((lo_u32 + np.uint32(0x7FFF) + ((lo_u32 >> np.uint32(16)) & np.uint32(1))) >> np.uint32(16)) & np.uint32(0xFFFF)
  return ((hi_16 << np.uint32(16)) | lo_16).view(np.float32)


@dataclasses.dataclass(frozen=True)
class WideBVH64UVMesh:
  name: str
  n_tris_orig: int
  n_leaves: int
  n_l1: int
  n_l2: int
  bmin: np.ndarray
  bmax: np.ndarray
  l2_roots_aabb: np.ndarray  # (8, 128)
  l2_aabbs: np.ndarray       # (N_L2, 8, 128)
  l0_aabbs: np.ndarray       # (N_L1, 8, 128)
  leaf_bank: np.ndarray      # (N_leaves, 24, 128)


@dataclasses.dataclass(frozen=True)
class WideBVH64UVSceneSpec:
  height: int
  width: int
  cam_fovy: float
  cam_shift: Tuple[float, float]
  znear: float
  zfar: float
  bvh: WideBVH64UVMesh
  tex_atlas: np.ndarray      # (T_safe, TH, TW, 4) float32
  mat_tex_ids: np.ndarray    # (M,) int32
  mat_codes: np.ndarray      # (M,) float32
  light_pos: np.ndarray      # (L, 3) float32 (up to 4 lights)
  light_dir: np.ndarray      # (L, 3) float32
  light_col: np.ndarray      # (L, 3) float32
  light_type: np.ndarray     # (L,) int32: 0=SUN, 1=AREA/POINT/SPOT
  sky_zenith: Tuple[float, float, float]
  sky_horizon: Tuple[float, float, float]
  sky_ground: Tuple[float, float, float]
  scene_center: np.ndarray   # (3,)
  scene_scale: np.ndarray    # (3,)
  z_scale: float
  use_shadows: bool = True
  exposure: float = 1.0


def build_wide_bvh64_uv(
    pack: Any,
    height: int = 512,
    width: int = 512,
    use_shadows: bool = True,
    build_method: str = "spatial_l1_morton",
) -> WideBVH64UVSceneSpec:
  """Pack a compiled `ScenePack` into a single 3-level Wide-BVH-64 + UV texture atlas."""
  verts = np.asarray(pack.verts, dtype=np.float32)
  faces = np.asarray(pack.faces, dtype=np.int32)
  vnorm = np.asarray(pack.vnorm, dtype=np.float32)
  tri_uvs = np.asarray(pack.tri_uvs, dtype=np.float32)
  tri_mid = np.asarray(pack.tri_mat_id, dtype=np.int32)
  tri_alb = np.asarray(pack.tri_albedo, dtype=np.float32)
  tri_rgh = np.clip(np.asarray(pack.tri_roughness, dtype=np.float32), 0.02, 0.99)
  tri_mc = np.asarray(pack.tri_mat_code, dtype=np.float32)

  centroids = (verts[faces[:, 0]] + verts[faces[:, 1]] + verts[faces[:, 2]]) * (1.0 / 3.0)
  if build_method == "morton":
    order = np.argsort(_morton3d_20bit(centroids), kind="stable")
  else:
    order = _spatial_median_order(centroids, LEAF_SIZE, BRANCH_FACTOR, stop_at_l1=True)
  del centroids

  faces = faces[order]
  tri_uvs = tri_uvs[order]
  tri_mid = tri_mid[order]
  tri_alb = tri_alb[order]
  tri_rgh = tri_rgh[order]
  tri_mc = tri_mc[order]
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

  def _pack_pair(row_idx: int, a: np.ndarray, b: np.ndarray) -> None:
    ba = np.empty((F_pad,), dtype=np.float32)
    bb = np.empty((F_pad,), dtype=np.float32)
    ba[:F] = a
    bb[:F] = b
    if F_pad > F:
      ba[F:] = a[-1]
      bb[F:] = b[-1]
    leaf_bank[:, row_idx, :] = _pack_bf16_pair_to_f32(ba, bb).reshape(n_leaves, LEAF_SIZE)

  tv0 = verts[faces[:, 0]]
  _pack_vec3(0, tv0)
  _pack_vec3(3, verts[faces[:, 1]] - tv0)
  _pack_vec3(6, verts[faces[:, 2]] - tv0)
  del tv0

  # Row 9: 2.0 = opaque shadow-casting tri, 1.0 = glass/transparent (primary hit only, shadow pass-through), 0.0 = pad
  tv_buf = np.zeros((F_pad,), dtype=np.float32)
  is_glass_tri = (tri_mc >= 1.8) & (tri_mc < 2.8)
  tv_buf[:F] = np.where(is_glass_tri, 1.0, 2.0).astype(np.float32)
  leaf_bank[:, 9, :] = tv_buf.reshape(n_leaves, LEAF_SIZE)
  del tv_buf

  # Rows 10..14: BF16-packed smooth vertex normals
  n0 = np.clip(vnorm[faces[:, 0]], -1.0, 1.0)
  n1 = np.clip(vnorm[faces[:, 1]], -1.0, 1.0)
  n2 = np.clip(vnorm[faces[:, 2]], -1.0, 1.0)
  zeros_f = np.zeros((F,), dtype=np.float32)
  _pack_pair(10, n0[:, 0], n0[:, 1])
  _pack_pair(11, n0[:, 2], n1[:, 0])
  _pack_pair(12, n1[:, 1], n1[:, 2])
  _pack_pair(13, n2[:, 0], n2[:, 1])
  _pack_pair(14, n2[:, 2], zeros_f)
  del n0, n1, n2, zeros_f

  # Rows 15..20: FP32 UV0, dUV1, dUV2
  uv0 = tri_uvs[:, 0, :]
  duv1 = tri_uvs[:, 1, :] - uv0
  duv2 = tri_uvs[:, 2, :] - uv0
  _pack_scalar(15, uv0[:, 0])
  _pack_scalar(16, uv0[:, 1])
  _pack_scalar(17, duv1[:, 0])
  _pack_scalar(18, duv1[:, 1])
  _pack_scalar(19, duv2[:, 0])
  _pack_scalar(20, duv2[:, 1])

  # Row 21: tri_mat_id (FP32 int), Rows 22..23: BF16 fallback albedo & roughness
  _pack_scalar(21, tri_mid.astype(np.float32))
  _pack_pair(22, np.clip(tri_alb[:, 0], 0.0, 1.0), np.clip(tri_alb[:, 1], 0.0, 1.0))
  _pack_pair(23, np.clip(tri_alb[:, 2], 0.0, 1.0), tri_rgh)

  # Tight leaf AABBs
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

  bmin = np.min(verts, axis=0)
  bmax = np.max(verts, axis=0)
  bvh = WideBVH64UVMesh(
      name=pack.scene_name,
      n_tris_orig=F,
      n_leaves=n_leaves,
      n_l1=n_l1,
      n_l2=n_l2,
      bmin=bmin,
      bmax=bmax,
      l2_roots_aabb=l2_roots_aabb,
      l2_aabbs=l2_aabbs,
      l0_aabbs=l0_aabbs,
      leaf_bank=leaf_bank,
  )

  # Ensure non-empty texture atlas even if scene has 0 textures
  if pack.tex_atlas.shape[0] > 0:
    tex_atlas = pack.tex_atlas.astype(np.float32)
  else:
    tex_atlas = np.full((1, 8, 8, 4), 0.7, dtype=np.float32)

  # Extract up to 4 top lights (Sun + Area/Point/Spot)
  lights_sorted = sorted(pack.lights, key=lambda l: float(l.get("energy", 1.0)) * (500.0 if l.get("type") == "SUN" else 1.0), reverse=True)
  if not lights_sorted:
    ctr = 0.5 * (bmin + bmax)
    span = float(np.max(bmax - bmin))
    lights_sorted = [{
        "type": "AREA",
        "pos": [float(ctr[0]), float(ctr[1] - 0.3 * span), float(bmax[2] + 0.5 * span)],
        "dir": [0.0, 0.2, -1.0],
        "color": [1.0, 0.96, 0.90],
        "energy": 400.0,
    }]
  lights_top = lights_sorted[:4]
  L = len(lights_top)
  l_pos = np.zeros((L, 3), dtype=np.float32)
  l_dir = np.zeros((L, 3), dtype=np.float32)
  l_col = np.zeros((L, 3), dtype=np.float32)
  l_type = np.zeros((L,), dtype=np.int32)

  # Normalize light intensities nicely for Stage 1 direct preview
  max_e = max(float(l.get("energy", 1.0)) for l in lights_top if l.get("type") != "SUN") if any(l.get("type") != "SUN" for l in lights_top) else 100.0
  max_e = max(max_e, 1.0)
  for i, l in enumerate(lights_top):
    l_pos[i] = np.asarray(l["pos"], dtype=np.float32)
    d = np.asarray(l["dir"], dtype=np.float32)
    l_dir[i] = d / max(float(np.linalg.norm(d)), 1e-12)
    col = np.asarray(l["color"], dtype=np.float32)
    is_sun = (l.get("type") == "SUN")
    l_type[i] = 0 if is_sun else 1
    if is_sun:
      gain = np.clip(float(l.get("energy", 3.0)) * 0.45, 0.8, 3.0)
    else:
      gain = np.clip(float(l.get("energy", 50.0)) / max_e * 2.2, 0.25, 2.5)
    l_col[i] = col * gain

  scene_center = 0.5 * (bmin + bmax)
  scene_scale = np.maximum(0.5 * (bmax - bmin), 0.25)
  z_scale = float(max(np.linalg.norm(bmax - bmin) * 0.65, 2.0))

  return WideBVH64UVSceneSpec(
      height=int(height),
      width=int(width),
      cam_fovy=float(pack.cam_fovy),
      cam_shift=pack.cam_shift,
      znear=float(pack.znear),
      zfar=float(pack.zfar),
      bvh=bvh,
      tex_atlas=tex_atlas,
      mat_tex_ids=np.asarray(pack.mat_tex_ids, dtype=np.int32),
      mat_codes=np.asarray(pack.mat_codes, dtype=np.float32),
      light_pos=l_pos,
      light_dir=l_dir,
      light_col=l_col,
      light_type=l_type,
      sky_zenith=pack.sky_zenith,
      sky_horizon=pack.sky_horizon,
      sky_ground=pack.sky_ground,
      scene_center=scene_center.astype(np.float32),
      scene_scale=scene_scale.astype(np.float32),
      z_scale=z_scale,
      use_shadows=bool(use_shadows),
  )


def compute_blend_camera_rays(
    spec: WideBVH64UVSceneSpec,
    cam_pos: jax.Array,
    cam_mat: jax.Array,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
  H, W = int(spec.height), int(spec.width)
  aspect = float(W) / float(H)
  fovy_rad = float(spec.cam_fovy) * (math.pi / 180.0)
  half_h = float(spec.znear) * math.tan(0.5 * fovy_rad)
  half_w = half_h * aspect
  sx, sy = float(spec.cam_shift[0]), float(spec.cam_shift[1])
  u = (jnp.arange(W, dtype=jnp.float32) + 0.5) / float(W)
  v = (jnp.arange(H, dtype=jnp.float32) + 0.5) / float(H)
  gx, gy = jnp.meshgrid(u, v, indexing="xy")
  x = (-half_w + (2.0 * half_w) * gx) + sx * (2.0 * half_w)
  y = (half_h - (2.0 * half_h) * gy) + sy * (2.0 * half_h * aspect)
  raw = jnp.stack([x, y, jnp.full_like(x, -float(spec.znear))], axis=-1).reshape(-1, 3)
  ld = raw / jnp.maximum(jnp.linalg.norm(raw, axis=-1, keepdims=True), 1e-12)
  neg_z = jnp.maximum(-ld[:, 2], 1e-6)
  ray_dir = jnp.einsum("bij,rj->bri", cam_mat, ld, precision=jax.lax.Precision.HIGHEST)
  ray_dir = ray_dir / jnp.maximum(jnp.linalg.norm(ray_dir, axis=-1, keepdims=True), 1e-12)
  inv_cos = (float(spec.znear) / neg_z)[None, :, None]
  ray_orig = cam_pos[:, None, :] + inv_cos * ray_dir
  return ray_orig, ray_dir, neg_z


def pallas_wide_bvh64_uv_raycast(
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
    interpret: bool = False,
):
  """Mosaic Pallas Wide-BVH-64 raycaster returning (dist, normal, uv, mid, alb, rgh)."""
  B, R, _ = ro.shape
  rays_2d, n_tiles, can_2d_tile = _pack_rays_tiled(ro, rd, t_limit, H, W)
  total_tiles = B * n_tiles
  leaf_rows = 10 if any_hit else 24

  def _bvh_uv_kernel(
      ray_ref, l2_roots_ref, l2_hbm_ref, l0_hbm_ref, leaf_hbm_ref, out_ref,
      l2_vmem, l0_vmem, leaf_vmem, sem_l2, sem_l0, sem_leaf,
  ):
    def _unpack_bf16_pair(f_row):
      u = jax.lax.bitcast_convert_type(f_row, jnp.int32)
      hi = jax.lax.bitcast_convert_type(jax.lax.bitwise_and(u, jnp.int32(-65536)), jnp.float32)
      lo = jax.lax.bitcast_convert_type(
          jax.lax.shift_left(jax.lax.bitwise_and(u, jnp.int32(0xFFFF)), jnp.int32(16)), jnp.float32
      )
      return hi, lo

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
    neg1 = jnp.full((BLOCK_R, 1), -1.0, jnp.float32)

    def _l2_cond(c2):
      return jnp.min(c2[0]) < jnp.max(c2[2])

    def _l2_body(c2):
      t2, unvis2_f, bd2, nx2, ny2, nz2, u2, v2, m2, pk_rg2, pk_br2 = c2
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
        t1, unvis1_f, bd1, nx1, ny1, nz1, u1, v1, m1, pk_rg1, pk_br1 = c1
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
          t0, unvis0_f, bd0, nx0, ny0, nz0, u0, v0, m0, pk_rg0, pk_br0 = c0
          best_c0 = jnp.argmin(t0, axis=1)[0]
          unvis0_next_f = jnp.where(cols_iota_64 == best_c0, 0.0, unvis0_f)
          leaf_idx = l1_idx * BRANCH_FACTOR + best_c0

          cp1 = pltpu.make_async_copy(leaf_hbm_ref.at[leaf_idx, :leaf_rows, :], leaf_vmem, sem_leaf)
          cp1.start(); cp1.wait()
          pk = leaf_vmem[...]

          v0x, v0y, v0z = pk[0:1, :], pk[1:2, :], pk[2:3, :]
          e1x, e1y, e1z = pk[3:4, :], pk[4:5, :], pk[5:6, :]
          e2x, e2y, e2z = pk[6:7, :], pk[7:8, :], pk[8:9, :]
          tv = (pk[9:10, :] > 1.5) if any_hit else (pk[9:10, :] > 0.5)

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
            return (t0_next, unvis0_next_f, new_bd, nx0, ny0, nz0, u0, v0, m0, pk_rg0, pk_br0)

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

            lu = jnp.sum(jnp.where(win, pk[15:16, :] + u * pk[17:18, :] + v * pk[19:20, :], 0.0), axis=1, keepdims=True)
            lv = jnp.sum(jnp.where(win, pk[16:17, :] + u * pk[18:19, :] + v * pk[20:21, :], 0.0), axis=1, keepdims=True)
            lmid = jnp.sum(jnp.where(win, pk[21:22, :], 0.0), axis=1, keepdims=True)
            lpk_rg = jnp.sum(jnp.where(win, pk[22:23, :], 0.0), axis=1, keepdims=True)
            lpk_br = jnp.sum(jnp.where(win, pk[23:24, :], 0.0), axis=1, keepdims=True)

            new_bd = jnp.where(imp, min_t, bd0)
            t0_n = jnp.min(jnp.where((tmin0_eff < new_bd) & (unvis0_next_f > 0.5), tmin0_eff, 1e20), axis=0, keepdims=True)
            return (
                t0_n,
                new_bd,
                jnp.where(imp, lnx, nx0),
                jnp.where(imp, lny, ny0),
                jnp.where(imp, lnz, nz0),
                jnp.where(imp, lu, u0),
                jnp.where(imp, lv, v0),
                jnp.where(imp, lmid, m0),
                jnp.where(imp, lpk_rg, pk_rg0),
                jnp.where(imp, lpk_br, pk_br0),
            )

          def _on_miss(_):
            return (jnp.where(unvis0_next_f > 0.5, t0, 1e20), bd0, nx0, ny0, nz0, u0, v0, m0, pk_rg0, pk_br0)

          t0_n, nbd, nnx, nny, nnz, nu, nv, nm, nrg, nbr = jax.lax.cond(any_imp, _on_hit, _on_miss, None)
          return (t0_n, unvis0_next_f, nbd, nnx, nny, nnz, nu, nv, nm, nrg, nbr)

        _, _, n_bd1, n_nx1, n_ny1, n_nz1, n_u1, n_v1, n_m1, n_rg1, n_br1 = jax.lax.while_loop(
            _l0_cond, _l0_body, (l0_tmin, l0_unvis_f, bd1, nx1, ny1, nz1, u1, v1, m1, pk_rg1, pk_br1)
        )
        t1_next = jnp.min(jnp.where((tmin1_eff < n_bd1) & (unvis1_next_f > 0.5), tmin1_eff, 1e20), axis=0, keepdims=True)
        return (t1_next, unvis1_next_f, n_bd1, n_nx1, n_ny1, n_nz1, n_u1, n_v1, n_m1, n_rg1, n_br1)

      _, _, n_bd2, n_nx2, n_ny2, n_nz2, n_u2, n_v2, n_m2, n_rg2, n_br2 = jax.lax.while_loop(
          _l1_cond, _l1_body, (l1_tmin, l1_unvis_f, bd2, nx2, ny2, nz2, u2, v2, m2, pk_rg2, pk_br2)
      )
      t2_next = jnp.min(jnp.where((tmin2_eff < n_bd2) & (unvis2_next_f > 0.5), tmin2_eff, 1e20), axis=0, keepdims=True)
      return (t2_next, unvis2_next_f, n_bd2, n_nx2, n_ny2, n_nz2, n_u2, n_v2, n_m2, n_rg2, n_br2)

    _, _, out_bd, out_nx, out_ny, out_nz, out_u, out_v, out_m, out_rg, out_br = jax.lax.while_loop(
        _l2_cond, _l2_body, (l2_tmin, l2_unvis_f, bd, z1, z1, z1, z1, z1, neg1, z1, z1)
    )

    if any_hit:
      out_ref[...] = jnp.concatenate([jnp.transpose(out_bd, (1, 0)), jnp.zeros((7, BLOCK_R), jnp.float32)], axis=0)
    else:
      out_ref[...] = jnp.concatenate(
          [
              jnp.transpose(out_bd, (1, 0)),
              jnp.transpose(out_nx, (1, 0)),
              jnp.transpose(out_ny, (1, 0)),
              jnp.transpose(out_nz, (1, 0)),
              jnp.transpose(out_u, (1, 0)),
              jnp.transpose(out_v, (1, 0)),
              jnp.transpose(out_m, (1, 0)),
              jnp.transpose(out_rg, (1, 0)),
          ],
          axis=0,
      )
      # Wait: we need out_br too! Pack out_m + out_rg + out_br? No, 8 rows = 0..7!
      # Wait: can we put out_br in a 16-row output OR pack alb+rgh?
      # Look: out_rg and out_br are ALREADY BF16 pairs in FP32 words! We only have 8 rows in (8, 128).
      # If we want 9 rows, out_shape can be (total_tiles * 16, 128) or pack (out_nx, out_ny, out_nz) + (out_u, out_v, out_m, out_rg, out_br) = 9 rows -> (16, 128)!
      # Even better: use 16 rows in out_shape when any_hit=False! Row 8 = out_br!

  out_rows = 8 if any_hit else 16

  def _bvh_uv_kernel_wrap(
      ray_ref, l2_roots_ref, l2_hbm_ref, l0_hbm_ref, leaf_hbm_ref, out_ref,
      l2_vmem, l0_vmem, leaf_vmem, sem_l2, sem_l0, sem_leaf,
  ):
    def _unpack_bf16_pair(f_row):
      u = jax.lax.bitcast_convert_type(f_row, jnp.int32)
      hi = jax.lax.bitcast_convert_type(jax.lax.bitwise_and(u, jnp.int32(-65536)), jnp.float32)
      lo = jax.lax.bitcast_convert_type(
          jax.lax.shift_left(jax.lax.bitwise_and(u, jnp.int32(0xFFFF)), jnp.int32(16)), jnp.float32
      )
      return hi, lo

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
    neg1 = jnp.full((BLOCK_R, 1), -1.0, jnp.float32)

    def _l2_cond(c2):
      return jnp.min(c2[0]) < jnp.max(c2[2])

    def _l2_body(c2):
      t2, unvis2_f, bd2, nx2, ny2, nz2, u2, v2, m2, pk_rg2, pk_br2 = c2
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
        t1, unvis1_f, bd1, nx1, ny1, nz1, u1, v1, m1, pk_rg1, pk_br1 = c1
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
          t0, unvis0_f, bd0, nx0, ny0, nz0, u0, v0, m0, pk_rg0, pk_br0 = c0
          best_c0 = jnp.argmin(t0, axis=1)[0]
          unvis0_next_f = jnp.where(cols_iota_64 == best_c0, 0.0, unvis0_f)
          leaf_idx = l1_idx * BRANCH_FACTOR + best_c0

          cp1 = pltpu.make_async_copy(leaf_hbm_ref.at[leaf_idx, :leaf_rows, :], leaf_vmem, sem_leaf)
          cp1.start(); cp1.wait()
          pk = leaf_vmem[...]

          v0x, v0y, v0z = pk[0:1, :], pk[1:2, :], pk[2:3, :]
          e1x, e1y, e1z = pk[3:4, :], pk[4:5, :], pk[5:6, :]
          e2x, e2y, e2z = pk[6:7, :], pk[7:8, :], pk[8:9, :]
          tv = (pk[9:10, :] > 1.5) if any_hit else (pk[9:10, :] > 0.5)

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
            return (t0_next, unvis0_next_f, new_bd, nx0, ny0, nz0, u0, v0, m0, pk_rg0, pk_br0)

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

            lu = jnp.sum(jnp.where(win, pk[15:16, :] + u * pk[17:18, :] + v * pk[19:20, :], 0.0), axis=1, keepdims=True)
            lv = jnp.sum(jnp.where(win, pk[16:17, :] + u * pk[18:19, :] + v * pk[20:21, :], 0.0), axis=1, keepdims=True)
            lmid = jnp.sum(jnp.where(win, pk[21:22, :], 0.0), axis=1, keepdims=True)
            lpk_rg = jnp.sum(jnp.where(win, pk[22:23, :], 0.0), axis=1, keepdims=True)
            lpk_br = jnp.sum(jnp.where(win, pk[23:24, :], 0.0), axis=1, keepdims=True)

            new_bd = jnp.where(imp, min_t, bd0)
            t0_n = jnp.min(jnp.where((tmin0_eff < new_bd) & (unvis0_next_f > 0.5), tmin0_eff, 1e20), axis=0, keepdims=True)
            return (
                t0_n,
                new_bd,
                jnp.where(imp, lnx, nx0),
                jnp.where(imp, lny, ny0),
                jnp.where(imp, lnz, nz0),
                jnp.where(imp, lu, u0),
                jnp.where(imp, lv, v0),
                jnp.where(imp, lmid, m0),
                jnp.where(imp, lpk_rg, pk_rg0),
                jnp.where(imp, lpk_br, pk_br0),
            )

          def _on_miss(_):
            return (jnp.where(unvis0_next_f > 0.5, t0, 1e20), bd0, nx0, ny0, nz0, u0, v0, m0, pk_rg0, pk_br0)

          t0_n, nbd, nnx, nny, nnz, nu, nv, nm, nrg, nbr = jax.lax.cond(any_imp, _on_hit, _on_miss, None)
          return (t0_n, unvis0_next_f, nbd, nnx, nny, nnz, nu, nv, nm, nrg, nbr)

        _, _, n_bd1, n_nx1, n_ny1, n_nz1, n_u1, n_v1, n_m1, n_rg1, n_br1 = jax.lax.while_loop(
            _l0_cond, _l0_body, (l0_tmin, l0_unvis_f, bd1, nx1, ny1, nz1, u1, v1, m1, pk_rg1, pk_br1)
        )
        t1_next = jnp.min(jnp.where((tmin1_eff < n_bd1) & (unvis1_next_f > 0.5), tmin1_eff, 1e20), axis=0, keepdims=True)
        return (t1_next, unvis1_next_f, n_bd1, n_nx1, n_ny1, n_nz1, n_u1, n_v1, n_m1, n_rg1, n_br1)

      _, _, n_bd2, n_nx2, n_ny2, n_nz2, n_u2, n_v2, n_m2, n_rg2, n_br2 = jax.lax.while_loop(
          _l1_cond, _l1_body, (l1_tmin, l1_unvis_f, bd2, nx2, ny2, nz2, u2, v2, m2, pk_rg2, pk_br2)
      )
      t2_next = jnp.min(jnp.where((tmin2_eff < n_bd2) & (unvis2_next_f > 0.5), tmin2_eff, 1e20), axis=0, keepdims=True)
      return (t2_next, unvis2_next_f, n_bd2, n_nx2, n_ny2, n_nz2, n_u2, n_v2, n_m2, n_rg2, n_br2)

    _, _, out_bd, out_nx, out_ny, out_nz, out_u, out_v, out_m, out_rg, out_br = jax.lax.while_loop(
        _l2_cond, _l2_body, (l2_tmin, l2_unvis_f, bd, z1, z1, z1, z1, z1, neg1, z1, z1)
    )

    if any_hit:
      out_ref[...] = jnp.concatenate([jnp.transpose(out_bd, (1, 0)), jnp.zeros((7, BLOCK_R), jnp.float32)], axis=0)
    else:
      z7 = jnp.zeros((7, BLOCK_R), jnp.float32)
      out_ref[...] = jnp.concatenate(
          [
              jnp.transpose(out_bd, (1, 0)),
              jnp.transpose(out_nx, (1, 0)),
              jnp.transpose(out_ny, (1, 0)),
              jnp.transpose(out_nz, (1, 0)),
              jnp.transpose(out_u, (1, 0)),
              jnp.transpose(out_v, (1, 0)),
              jnp.transpose(out_m, (1, 0)),
              jnp.transpose(out_rg, (1, 0)),
              jnp.transpose(out_br, (1, 0)),
              z7,
          ],
          axis=0,
      )

  out_2d = pl.pallas_call(
      _bvh_uv_kernel_wrap,
      out_shape=jax.ShapeDtypeStruct((total_tiles * out_rows, BLOCK_R), jnp.float32),
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
          out_specs=pl.BlockSpec((out_rows, BLOCK_R), lambda i: (i, 0)),
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

  def _untile(arr):
    if can_2d_tile:
      th, tw = H // TILE_H, W // TILE_W
      x = arr.reshape(B, th, tw, TILE_H, TILE_W)
      x = jnp.swapaxes(x, 2, 3)
      return x.reshape(B, H * W)
    return arr.reshape(B, n_tiles * BLOCK_R)[:, : H * W]

  rows = out_2d.reshape(total_tiles, out_rows, BLOCK_R)
  dist = _untile(rows[:, 0, :])
  if any_hit:
    return dist

  nx = _untile(rows[:, 1, :])
  ny = _untile(rows[:, 2, :])
  nz = _untile(rows[:, 3, :])
  uu = _untile(rows[:, 4, :])
  vv = _untile(rows[:, 5, :])
  mid_f = _untile(rows[:, 6, :])
  rg_f = _untile(rows[:, 7, :])
  br_f = _untile(rows[:, 8, :])

  def _unpack_bf16_jnp(f_arr):
    u = jax.lax.bitcast_convert_type(f_arr, jnp.int32)
    hi = jax.lax.bitcast_convert_type(jax.lax.bitwise_and(u, jnp.int32(-65536)), jnp.float32)
    lo = jax.lax.bitcast_convert_type(
        jax.lax.shift_left(jax.lax.bitwise_and(u, jnp.int32(0xFFFF)), jnp.int32(16)), jnp.float32
    )
    return hi, lo

  alb_r, alb_g = _unpack_bf16_jnp(rg_f)
  alb_b, rgh = _unpack_bf16_jnp(br_f)
  nrm = jnp.stack([nx, ny, nz], axis=-1)
  uv = jnp.stack([uu, vv], axis=-1)
  alb = jnp.stack([alb_r, alb_g, alb_b], axis=-1)
  mid = jnp.round(mid_f).astype(jnp.int32)
  return dist, nrm, uv, mid, alb, rgh


def _sample_tex_atlas_bilinear(
    tex_atlas: jax.Array,  # (T, TH, TW, 4)
    tex_id: jax.Array,     # (B, R) int32
    uv: jax.Array,         # (B, R, 2)
    fallback_alb: jax.Array,
    fallback_rgh: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
  T, TH, TW, _ = tex_atlas.shape
  has_tex = tex_id >= 0
  safe_tid = jnp.clip(tex_id, 0, T - 1)
  u = jnp.mod(uv[..., 0], 1.0)
  v = jnp.mod(1.0 - uv[..., 1], 1.0)
  fx = u * float(TW - 1)
  fy = v * float(TH - 1)
  x0 = jnp.clip(jnp.floor(fx).astype(jnp.int32), 0, TW - 1)
  y0 = jnp.clip(jnp.floor(fy).astype(jnp.int32), 0, TH - 1)
  x1 = jnp.minimum(x0 + 1, TW - 1)
  y1 = jnp.minimum(y0 + 1, TH - 1)
  wx = (fx - x0)[..., None]
  wy = (fy - y0)[..., None]

  c00 = tex_atlas[safe_tid, y0, x0]
  c10 = tex_atlas[safe_tid, y0, x1]
  c01 = tex_atlas[safe_tid, y1, x0]
  c11 = tex_atlas[safe_tid, y1, x1]
  tex_val = (1.0 - wy) * ((1.0 - wx) * c00 + wx * c10) + wy * ((1.0 - wx) * c01 + wx * c11)
  alb = jnp.where(has_tex[..., None], tex_val[..., :3], fallback_alb)
  rgh = jnp.where(has_tex, tex_val[..., 3], fallback_rgh)
  return alb, jnp.clip(rgh, 0.02, 0.98)


def _linear_to_srgb_jnp(x: jax.Array) -> jax.Array:
  x = jnp.maximum(x, 0.0)
  return jnp.where(x <= 0.0031308, 12.92 * x, 1.055 * (jnp.maximum(x, 1e-8) ** (1.0 / 2.4)) - 0.055)


def render_wide_bvh64_uv_stage1(
    spec: WideBVH64UVSceneSpec,
    dev_arrays: Tuple[jax.Array, ...],  # (l2_roots, l2, l0, leaf, tex_atlas, mat_tex_ids, mat_codes)
    cam_xpos: jax.Array,                # (B, 3)
    cam_xmat: jax.Array,                # (B, 3, 3)
    interpret: bool = False,
) -> Tuple[GBuffer, jax.Array, jax.Array]:
  """Render exact Stage 1 UV-textured PBR G-Buffer + (roughness, mat_code) maps."""
  l2_roots_d, l2_d, l0_d, leaf_d, tex_d, mtid_d, mcode_d = dev_arrays
  H, W = int(spec.height), int(spec.width)
  ro, rd, neg_z = compute_blend_camera_rays(spec, cam_xpos, cam_xmat)
  B, R, _ = ro.shape
  max_dist = jnp.broadcast_to(((float(spec.zfar) - float(spec.znear)) / neg_z)[None, :], (B, R))

  d_hit, nrm, uv, mid, fb_alb, fb_rgh = pallas_wide_bvh64_uv_raycast(
      l2_roots_d, l2_d, l0_d, leaf_d, ro, rd, max_dist, H, W, any_hit=False, interpret=interpret
  )

  hit = mid >= 0
  best_d = jnp.where(hit, d_hit, -1.0)
  vdir = -rd
  nrm = jnp.where(hit[..., None], nrm, 0.0)
  flip = jnp.sum(nrm * vdir, axis=-1, keepdims=True) < 0.0
  nrm = jnp.where(flip, -nrm, nrm)

  safe_mid = jnp.clip(mid, 0, len(spec.mat_tex_ids) - 1)
  tex_id = jnp.where(hit, mtid_d[safe_mid], -1)
  mcode = jnp.where(hit, mcode_d[safe_mid], 0.0)
  alb_lin, rgh = _sample_tex_atlas_bilinear(tex_d, tex_id, uv, fb_alb, fb_rgh)
  hit_pos = ro + jnp.where(hit, best_d, 0.0)[..., None] * rd

  # Multi-light GGX + Diffuse + Emissive + Glass + Sky direct shading
  metallic = jnp.clip(mcode, 0.0, 1.0)
  is_emit = mcode >= 2.8
  is_glass = (mcode >= 1.8) & (mcode < 2.8)
  emit_pow = jnp.where(is_emit, (mcode - 2.0) * 2.5, 0.0)

  # Primary light shadow raycast
  l_pos0 = jnp.asarray(spec.light_pos[0], dtype=jnp.float32)
  l_dir0 = jnp.asarray(spec.light_dir[0], dtype=jnp.float32)
  l_type0 = int(spec.light_type[0])
  if l_type0 == 0:
    L0 = jnp.broadcast_to(-l_dir0[None, None, :], ro.shape)
    dist0 = jnp.full((B, R), float(spec.zfar) * 0.9, dtype=jnp.float32)
  else:
    d_vec0 = l_pos0[None, None, :] - hit_pos
    dist0 = jnp.linalg.norm(d_vec0, axis=-1)
    L0 = d_vec0 / jnp.maximum(dist0[..., None], 1e-12)
  ndotl0 = jnp.maximum(0.0, jnp.sum(nrm * L0, axis=-1))

  vis0 = jnp.ones((B, R), dtype=jnp.float32)
  if spec.use_shadows:
    sh_o = hit_pos + nrm * 2.0e-3
    sh_lim = jnp.where(hit & (ndotl0 > 0.0), jnp.maximum(dist0 - 2.0e-3, 0.0), 0.0)
    d_sh = pallas_wide_bvh64_uv_raycast(
        l2_roots_d, l2_d, l0_d, leaf_d, sh_o, L0, sh_lim, H, W, any_hit=True, interpret=interpret
    )
    vis0 = jnp.where(d_sh < sh_lim - 1e-4, 0.15, 1.0)

  # Hemisphere ambient
  up_w = 0.5 * (nrm[..., 2] + 1.0)
  sky_z = jnp.asarray(spec.sky_zenith, dtype=jnp.float32)
  sky_h = jnp.asarray(spec.sky_horizon, dtype=jnp.float32)
  sky_g = jnp.asarray(spec.sky_ground, dtype=jnp.float32)
  amb_col = up_w[..., None] * sky_h + (1.0 - up_w[..., None]) * sky_g
  rad = alb_lin * (0.45 * amb_col + 0.06)

  ndotv = jnp.maximum(jnp.sum(nrm * vdir, axis=-1), 1e-4)
  f0 = 0.04 * (1.0 - metallic[..., None]) + alb_lin * metallic[..., None]
  alpha = jnp.maximum(rgh * rgh, 0.025)
  a2 = alpha * alpha

  for li in range(len(spec.light_type)):
    lp = jnp.asarray(spec.light_pos[li], dtype=jnp.float32)
    ld = jnp.asarray(spec.light_dir[li], dtype=jnp.float32)
    lc = jnp.asarray(spec.light_col[li], dtype=jnp.float32)
    if int(spec.light_type[li]) == 0:
      Li = jnp.broadcast_to(-ld[None, None, :], ro.shape)
      atten = 1.0
    else:
      dv = lp[None, None, :] - hit_pos
      d2 = jnp.sum(dv * dv, axis=-1)
      dl = jnp.sqrt(jnp.maximum(d2, 1e-8))
      Li = dv / dl[..., None]
      atten = 1.0 / (1.0 + 0.05 * d2)
    ndotl = jnp.maximum(0.0, jnp.sum(nrm * Li, axis=-1))
    vis_i = vis0 if li == 0 else 1.0
    H_v = (Li + vdir) / jnp.maximum(jnp.linalg.norm(Li + vdir, axis=-1, keepdims=True), 1e-12)
    ndoth = jnp.maximum(jnp.sum(nrm * H_v, axis=-1), 0.0)
    vdoth = jnp.maximum(jnp.sum(vdir * H_v, axis=-1), 0.0)
    denom = ndoth * ndoth * (a2 - 1.0) + 1.0
    D = a2 / jnp.maximum(math.pi * denom * denom, 1e-5)
    F = f0 + (1.0 - f0) * ((1.0 - vdoth) ** 5)[..., None]
    spec_term = jnp.minimum((D * 0.25)[..., None] * F, 2.5)
    diff_term = (1.0 - metallic[..., None]) * alb_lin / math.pi
    rad = rad + (diff_term + spec_term) * lc[None, None, :] * (ndotl * atten * vis_i)[..., None]

  rad = rad + jnp.where(is_emit[..., None], alb_lin * emit_pow[..., None], 0.0)
  # Brighten dielectric glass panes / lamp shades with Fresnel sky glow
  glass_fres = 0.04 + 0.96 * ((1.0 - ndotv) ** 4)
  rad = jnp.where(is_glass[..., None], alb_lin * (0.85 * jnp.maximum(sky_h[None, None, :], 0.65) + 0.85 * glass_fres[..., None]), rad)
  rad_tm = rad / (1.0 + 0.25 * rad)
  direct_srgb = jnp.clip(_linear_to_srgb_jnp(rad_tm), 0.0, 1.0)
  alb_srgb = jnp.clip(_linear_to_srgb_jnp(alb_lin), 0.0, 1.0)

  sz = rd[..., 2]
  tu = jnp.clip(sz, 0.0, 1.0)[..., None]
  td = jnp.clip(-sz, 0.0, 1.0)[..., None]
  sky_lin = jnp.where(sz[..., None] >= 0.0, (1.0 - tu) * sky_h + tu * sky_z, (1.0 - td) * sky_h + td * sky_g)
  sky_srgb = jnp.clip(_linear_to_srgb_jnp(sky_lin / (1.0 + 0.25 * sky_lin)), 0.0, 1.0)

  rgb = jnp.where(hit[..., None], direct_srgb, sky_srgb)
  inv_cos = (1.0 / neg_z)[None, :]
  euclid_d = jnp.where(hit, best_d + float(spec.znear) * inv_cos, -1.0)
  z_d = jnp.where(hit, float(spec.znear) + best_d * neg_z[None, :], -1.0)

  gb = GBuffer(
      rgb=rgb.reshape(B, H, W, 3),
      albedo=jnp.where(hit[..., None], alb_srgb, sky_srgb).reshape(B, H, W, 3),
      depth=euclid_d.reshape(B, H, W),
      z_depth=z_d.reshape(B, H, W),
      normal=nrm.reshape(B, H, W, 3),
      view_dir=vdir.reshape(B, H, W, 3),
      world_pos=jnp.where(hit[..., None], hit_pos, 0.0).reshape(B, H, W, 3),
      geom_id=jnp.where(hit, mid, -1).reshape(B, H, W),
      direct_shadow=jnp.where(hit, vis0, 1.0).reshape(B, H, W),
      direct_rgb=rgb.reshape(B, H, W, 3),
  )
  return gb, rgh.reshape(B, H, W), mcode.reshape(B, H, W)


def build_scene_gbuffer_features(
    gb: GBuffer,
    rgh: jax.Array,
    mcode: jax.Array,
    scene_center: Tuple[float, float, float],
    scene_scale: Tuple[float, float, float],
    z_scale: float,
) -> jax.Array:
  """Scene-adaptive 26-ch G-Buffer features (normalized world_pos & z_depth + roughness/mcode)."""
  from tpu_mjx_render.neural_shader import _local_edges
  hit = (gb.geom_id >= 0).astype(jnp.float32)[..., None]
  ctr = jnp.asarray(scene_center, dtype=jnp.float32)
  scl = jnp.asarray(scene_scale, dtype=jnp.float32)
  pos_norm = jnp.clip((gb.world_pos - ctr) / scl, -1.5, 1.5)
  z_norm = jnp.where(gb.z_depth > 0.0, jnp.clip(gb.z_depth / max(float(z_scale), 0.5), 0.0, 2.0), 2.0)[..., None]
  sh = gb.direct_shadow[..., None]
  g_edge, z_edge = _local_edges(gb.geom_id, gb.z_depth)
  ndotv_raw = jnp.sum(gb.view_dir * gb.normal, axis=-1, keepdims=True)
  ndotv = jnp.clip(jnp.abs(ndotv_raw), 0.0, 1.0)
  refl = gb.view_dir - 2.0 * ndotv_raw * gb.normal
  return jnp.concatenate(
      [
          gb.rgb,                     # 3
          gb.albedo,                  # 3
          gb.normal,                  # 3
          gb.view_dir,                # 3
          pos_norm,                   # 3
          z_norm,                     # 1
          sh,                         # 1
          hit,                        # 1
          g_edge,                     # 1
          z_edge,                     # 1
          refl,                       # 3
          ndotv,                      # 1
          rgh[..., None],             # 1
          jnp.clip(mcode, 0.0, 4.0)[..., None] * 0.25,  # 1
      ],
      axis=-1,
  )  # Total = 26 channels


def pack_dev_arrays(spec: WideBVH64UVSceneSpec, device_or_sharding: Any) -> Tuple[jax.Array, ...]:
  arrs = (
      spec.bvh.l2_roots_aabb,
      spec.bvh.l2_aabbs,
      spec.bvh.l0_aabbs,
      spec.bvh.leaf_bank,
      spec.tex_atlas,
      spec.mat_tex_ids,
      spec.mat_codes,
  )
  return tuple(jax.device_put(a, device_or_sharding) for a in arrs)
