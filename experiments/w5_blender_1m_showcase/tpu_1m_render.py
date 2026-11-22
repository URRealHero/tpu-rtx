"""TPU-Native 2-Level Wide-BVH (BVH-128) Mosaic Pallas Raytracer for 1M+ Triangle Textured Blender Scenes.

Features:
  1. Zero dynamic pointers & zero BVH rebuild under rigid object motion.
  2. 2-level Morton-ordered Wide-BVH (`BVH-128` Top-Nodes -> `128`-tri Leaves, up to 2M tris/mesh).
  3. 8x16 2D screen-space ray tiling for high spatial BVH coherence.
  4. Native Mosaic `pl.pallas_call` kernel (`interpret=False` on TPU) with HBM->VMEM `make_async_copy`
     streaming ONLY visited BVH nodes & leaves into VMEM scratchpads, strictly (8, 128) & (128, 128) VPU ops.
  5. Exact barycentric smooth vertex normals + bilinearly interpolated 512x512 UV albedo & roughness maps.
  6. Supports 1-chip and 4-chip `shard_map` execution with zero host callbacks (`host_callbacks = 0`).
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
import numpy as np

from tpu_mjx_render.primitives import MJ_MAXVAL
from tpu_mjx_render.renderer import GBuffer

LEAF_SIZE = 128
TOP_FAN = 128
TILE_H = 8
TILE_W = 16
BLOCK_R = TILE_H * TILE_W  # 128 rays per 2D tile


@dataclasses.dataclass(frozen=True)
class BVHMeshItem:
  mesh_id: int
  name: str
  n_tris_orig: int
  n_leaves: int
  n_top: int
  metallic: float
  obj_aabb_min: np.ndarray     # (3,)
  obj_aabb_max: np.ndarray     # (3,)
  root_box: jax.Array          # (8, 128) float32
  leaf_boxes: jax.Array        # (n_top, 8, 128) float32
  leaf_tris: jax.Array         # (n_leaves, 24, 128) float32
  # CPU copies for pure-JAX reference mode
  leaf_boxes_np: np.ndarray    # (n_top, 8, 128)
  leaf_tris_np: np.ndarray     # (n_leaves, 24, 128)


@dataclasses.dataclass(frozen=True)
class BVHSceneSpec:
  height: int
  width: int
  cam_fovy: float
  znear: float
  zfar: float
  meshes: Tuple[BVHMeshItem, ...]
  tex_pack: jax.Array          # (M, TH, TW, 4) linear RGB + roughness
  metallic_tbl: jax.Array      # (M,)
  light_pos: np.ndarray        # (3,)
  sky_rgb1: Tuple[float, float, float]
  sky_rgb2: Tuple[float, float, float]
  use_shadows: bool = True


def _morton3d_10bit(pts: np.ndarray) -> np.ndarray:
  pmin = np.min(pts, axis=0, keepdims=True)
  pmax = np.max(pts, axis=0, keepdims=True)
  span = np.maximum(pmax - pmin, 1e-6)
  q = np.clip(np.floor((pts - pmin) / span * 1023.0), 0, 1023).astype(np.uint32)

  def _expand(v: np.ndarray) -> np.ndarray:
    v = (v * 0x00010001) & 0xFF0000FF
    v = (v * 0x00000101) & 0x0F00F00F
    v = (v * 0x00000011) & 0xC30C30C3
    v = (v * 0x00000005) & 0x49249249
    return v

  return (_expand(q[:, 0]) << 2) | (_expand(q[:, 1]) << 1) | _expand(q[:, 2])


def _srgb_to_linear_np(x: np.ndarray) -> np.ndarray:
  x = np.clip(x, 0.0, 1.0).astype(np.float32)
  return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def _linear_to_srgb_jnp(x: jax.Array) -> jax.Array:
  x = jnp.maximum(x, 0.0)
  return jnp.where(x <= 0.0031308, 12.92 * x, 1.055 * (jnp.maximum(x, 1e-8) ** (1.0 / 2.4)) - 0.055)


def build_bvh_scene_spec(
    scene_npz_path: str,
    height: int = 256,
    width: int = 256,
    use_shadows: bool = True,
    put_on_device: bool = True,
) -> BVHSceneSpec:
  """Load Blender 1M+ tri NPZ and pack Morton-ordered 2-level BVH-128 structures."""
  ds = np.load(scene_npz_path)
  M = int(ds["n_meshes"])
  items: List[BVHMeshItem] = []
  tex_list = []
  met_list = []

  for mid in range(M):
    name = str(ds[f"mesh_{mid}_name"])
    verts = ds[f"mesh_{mid}_verts"].astype(np.float32)
    vnorm = ds[f"mesh_{mid}_normals"].astype(np.float32)
    vuv = ds[f"mesh_{mid}_uvs"].astype(np.float32)
    faces = ds[f"mesh_{mid}_faces"].astype(np.int32)
    alb_srgb = ds[f"mesh_{mid}_tex_alb"].astype(np.float32) / 255.0
    rgh = ds[f"mesh_{mid}_tex_rgh"].astype(np.float32) / 255.0
    met = float(ds[f"mesh_{mid}_metallic"])

    tex_lin = _srgb_to_linear_np(alb_srgb)
    tex_list.append(np.concatenate([tex_lin, rgh[..., None]], axis=-1))
    met_list.append(met)

    # 30-bit Morton sort of triangle centroids
    tv0_raw = verts[faces[:, 0]]
    tv1_raw = verts[faces[:, 1]]
    tv2_raw = verts[faces[:, 2]]
    centroids = (tv0_raw + tv1_raw + tv2_raw) / 3.0
    order = np.argsort(_morton3d_10bit(centroids), kind="stable")
    faces = faces[order]

    F = int(faces.shape[0])
    n_leaves = (F + LEAF_SIZE - 1) // LEAF_SIZE
    F_pad = n_leaves * LEAF_SIZE

    tv0 = verts[faces[:, 0]]
    tv1 = verts[faces[:, 1]]
    tv2 = verts[faces[:, 2]]
    tn0 = vnorm[faces[:, 0]]
    tn1 = vnorm[faces[:, 1]]
    tn2 = vnorm[faces[:, 2]]
    tu0 = vuv[faces[:, 0]]
    tu1 = vuv[faces[:, 1]]
    tu2 = vuv[faces[:, 2]]

    v0 = np.zeros((F_pad, 3), dtype=np.float32)
    e1 = np.zeros((F_pad, 3), dtype=np.float32)
    e2 = np.zeros((F_pad, 3), dtype=np.float32)
    n0 = np.zeros((F_pad, 3), dtype=np.float32)
    n1 = np.zeros((F_pad, 3), dtype=np.float32)
    n2 = np.zeros((F_pad, 3), dtype=np.float32)
    uv0 = np.zeros((F_pad, 2), dtype=np.float32)
    duv1 = np.zeros((F_pad, 2), dtype=np.float32)
    duv2 = np.zeros((F_pad, 2), dtype=np.float32)

    v0[:F] = tv0
    e1[:F] = tv1 - tv0
    e2[:F] = tv2 - tv0
    n0[:F] = tn0
    n1[:F] = tn1
    n2[:F] = tn2
    uv0[:F] = tu0
    duv1[:F] = tu1 - tu0
    duv2[:F] = tu2 - tu0

    # Degenerate padding: e1=e2=0 -> det=0 automatically skipped, v0 keeps AABB tight
    if F_pad > F:
      v0[F:] = tv0[-1]

    v0_l = v0.reshape(n_leaves, LEAF_SIZE, 3)
    e1_l = e1.reshape(n_leaves, LEAF_SIZE, 3)
    e2_l = e2.reshape(n_leaves, LEAF_SIZE, 3)
    pts_l = np.stack([v0_l, v0_l + e1_l, v0_l + e2_l], axis=2)
    lmin = np.min(pts_l, axis=(1, 2)) - 1e-4  # (n_leaves, 3)
    lmax = np.max(pts_l, axis=(1, 2)) + 1e-4

    # Pack leaf_tris: (n_leaves, 24, 128)
    leaf_tris_np = np.zeros((n_leaves, 24, LEAF_SIZE), dtype=np.float32)
    leaf_tris_np[:, 0:3, :] = v0_l.transpose(0, 2, 1)
    leaf_tris_np[:, 3:6, :] = e1_l.transpose(0, 2, 1)
    leaf_tris_np[:, 6:9, :] = e2_l.transpose(0, 2, 1)
    leaf_tris_np[:, 9:12, :] = n0.reshape(n_leaves, LEAF_SIZE, 3).transpose(0, 2, 1)
    leaf_tris_np[:, 12:15, :] = n1.reshape(n_leaves, LEAF_SIZE, 3).transpose(0, 2, 1)
    leaf_tris_np[:, 15:18, :] = n2.reshape(n_leaves, LEAF_SIZE, 3).transpose(0, 2, 1)
    leaf_tris_np[:, 18:20, :] = uv0.reshape(n_leaves, LEAF_SIZE, 2).transpose(0, 2, 1)
    leaf_tris_np[:, 20:22, :] = duv1.reshape(n_leaves, LEAF_SIZE, 2).transpose(0, 2, 1)
    leaf_tris_np[:, 22:24, :] = duv2.reshape(n_leaves, LEAF_SIZE, 2).transpose(0, 2, 1)

    # Top-level BVH-128 over leaf AABBs
    n_top = (n_leaves + TOP_FAN - 1) // TOP_FAN
    assert n_top <= TOP_FAN, f"Mesh {name} has n_top={n_top} > {TOP_FAN}"
    L_pad = n_top * TOP_FAN
    lmin_p = np.full((L_pad, 3), 1e6, dtype=np.float32)
    lmax_p = np.full((L_pad, 3), -1e6, dtype=np.float32)
    lval_p = np.zeros((L_pad,), dtype=np.float32)
    lmin_p[:n_leaves] = lmin
    lmax_p[:n_leaves] = lmax
    lval_p[:n_leaves] = 1.0

    leaf_boxes_np = np.zeros((n_top, 8, TOP_FAN), dtype=np.float32)
    leaf_boxes_np[:, 0:3, :] = lmin_p.reshape(n_top, TOP_FAN, 3).transpose(0, 2, 1)
    leaf_boxes_np[:, 3:6, :] = lmax_p.reshape(n_top, TOP_FAN, 3).transpose(0, 2, 1)
    leaf_boxes_np[:, 6, :] = lval_p.reshape(n_top, TOP_FAN)

    tmin = np.min(lmin_p.reshape(n_top, TOP_FAN, 3), axis=1)
    tmax = np.max(lmax_p.reshape(n_top, TOP_FAN, 3), axis=1)
    obj_min = np.min(tmin, axis=0)
    obj_max = np.max(tmax, axis=0)

    root_box_np = np.zeros((8, TOP_FAN), dtype=np.float32)
    root_box_np[0:3, :n_top] = tmin.T
    root_box_np[3:6, :n_top] = tmax.T
    root_box_np[6, :n_top] = 1.0

    items.append(
        BVHMeshItem(
            mesh_id=mid,
            name=name,
            n_tris_orig=F,
            n_leaves=n_leaves,
            n_top=n_top,
            metallic=met,
            obj_aabb_min=obj_min.astype(np.float32),
            obj_aabb_max=obj_max.astype(np.float32),
            root_box=jnp.asarray(root_box_np) if put_on_device else jnp.zeros((1,)),
            leaf_boxes=jnp.asarray(leaf_boxes_np) if put_on_device else jnp.zeros((1,)),
            leaf_tris=jnp.asarray(leaf_tris_np) if put_on_device else jnp.zeros((1,)),
            leaf_boxes_np=leaf_boxes_np,
            leaf_tris_np=leaf_tris_np,
        )
    )

  tex_arr = np.stack(tex_list, axis=0).astype(np.float32)
  met_arr = np.asarray(met_list, dtype=np.float32)
  return BVHSceneSpec(
      height=int(height),
      width=int(width),
      cam_fovy=float(ds["cam_fovy"]),
      znear=float(ds["znear"]),
      zfar=float(ds["zfar"]),
      meshes=tuple(items),
      tex_pack=jnp.asarray(tex_arr) if put_on_device else jnp.zeros((1,)),
      metallic_tbl=jnp.asarray(met_arr) if put_on_device else jnp.zeros((1,)),
      light_pos=ds["light_pos"].astype(np.float32),
      sky_rgb1=(0.28, 0.36, 0.48),
      sky_rgb2=(0.12, 0.16, 0.22),
      use_shadows=bool(use_shadows),
  )


def _compute_camera_rays_2d_tiled(
    spec: BVHSceneSpec,
    cam_xpos: jax.Array,
    cam_xmat: jax.Array,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
  """Build (B, H, W, 3) rays matching Blender vertical-FOV camera, return tiled & untiled."""
  H, W = spec.height, spec.width
  px = jnp.arange(W, dtype=jnp.float32)
  py = jnp.arange(H, dtype=jnp.float32)
  gx, gy = jnp.meshgrid(px, py, indexing="xy")
  u = (gx + 0.5) / float(W)
  v = (gy + 0.5) / float(H)
  aspect = float(W) / float(H)
  fovy_rad = float(spec.cam_fovy) * (math.pi / 180.0)
  half_h = float(spec.znear) * math.tan(0.5 * fovy_rad)
  half_w = half_h * aspect
  x = -half_w + (2.0 * half_w) * u
  y = half_h - (2.0 * half_h) * v
  raw = jnp.stack([x, y, jnp.full_like(x, -float(spec.znear))], axis=-1)
  local_dir = raw / jnp.maximum(jnp.linalg.norm(raw, axis=-1, keepdims=True), 1e-12)
  neg_z = jnp.maximum(-local_dir[..., 2], 1e-6)  # (H, W)

  # Rotate into world space: (B, H, W, 3)
  ray_dir = jnp.einsum("bij,hwj->bhwi", cam_xmat, local_dir, precision=jax.lax.Precision.HIGHEST)
  ray_orig = jnp.broadcast_to(cam_xpos[:, None, None, :], ray_dir.shape) + float(spec.znear) * (1.0 / neg_z)[None, :, :, None] * ray_dir
  return ray_orig, ray_dir, neg_z


def _tile_rays(arr: jax.Array, H: int, W: int) -> jax.Array:
  """(B, H, W, ...) -> (B, n_tiles, 128, ...) using 8x16 2D screen tiles."""
  B = arr.shape[0]
  th, tw = H // TILE_H, W // TILE_W
  tail = arr.shape[3:]
  x = arr.reshape(B, th, TILE_H, tw, TILE_W, *tail)
  x = jnp.swapaxes(x, 2, 3)  # (B, th, tw, TILE_H, TILE_W, ...)
  return x.reshape(B, th * tw, BLOCK_R, *tail)


def _untile_rays(arr: jax.Array, H: int, W: int) -> jax.Array:
  """(B, n_tiles, 128, ...) -> (B, H, W, ...)."""
  B = arr.shape[0]
  th, tw = H // TILE_H, W // TILE_W
  tail = arr.shape[3:]
  x = arr.reshape(B, th, tw, TILE_H, TILE_W, *tail)
  x = jnp.swapaxes(x, 2, 3)
  return x.reshape(B, H, W, *tail)


def _aabb_slab_128(
    rox: jax.Array, roy: jax.Array, roz: jax.Array,
    rdx: jax.Array, rdy: jax.Array, rdz: jax.Array,
    bmin_x: jax.Array, bmin_y: jax.Array, bmin_z: jax.Array,
    bmax_x: jax.Array, bmax_y: jax.Array, bmax_z: jax.Array,
    tl: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
  """Branchless 2D (128 rays x 128 boxes) slab test returning (hit_mask, tmin)."""
  def _slab(o, d, mn, mx):
    ok = jnp.abs(d) > 1e-12
    inv = jnp.where(ok, 1.0 / jnp.where(ok, d, 1.0), 1e12)
    t0 = (mn - o) * inv
    t1 = (mx - o) * inv
    return jnp.minimum(t0, t1), jnp.maximum(t0, t1)

  tx0, tx1 = _slab(rox, rdx, bmin_x, bmax_x)
  ty0, ty1 = _slab(roy, rdy, bmin_y, bmax_y)
  tz0, tz1 = _slab(roz, rdz, bmin_z, bmax_z)
  tmin = jnp.maximum(tx0, jnp.maximum(ty0, tz0))
  tmax = jnp.minimum(tx1, jnp.minimum(ty1, tz1))
  hit = (tmax >= jnp.maximum(tmin, 0.0)) & (tmin < tl)
  return hit, jnp.maximum(tmin, 0.0)


def _pallas_bvh_mesh_raycast(
    item: BVHMeshItem,
    ro_tile: jax.Array,   # (B, n_tiles, 128, 3)
    rd_tile: jax.Array,   # (B, n_tiles, 128, 3)
    tl_tile: jax.Array,   # (B, n_tiles, 128)
    any_hit: bool = False,
    interpret: bool = False,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
  """Native Mosaic Pallas 2-level Wide-BVH-128 kernel with HBM->VMEM DMA streaming."""
  B, n_tiles, _, _ = ro_tile.shape
  N_BLK = B * n_tiles

  # Pack rays into (N_BLK * 8, 128): rows 0..2=ro, 3..5=rd, 6=tl, 7=0
  rays_8 = jnp.concatenate(
      [ro_tile.reshape(N_BLK, BLOCK_R, 3), rd_tile.reshape(N_BLK, BLOCK_R, 3), tl_tile.reshape(N_BLK, BLOCK_R, 1), jnp.zeros((N_BLK, BLOCK_R, 1), jnp.float32)],
      axis=-1,
  )  # (N_BLK, 128, 8)
  rays_2d = rays_8.transpose(0, 2, 1).reshape(N_BLK * 8, BLOCK_R)

  def _bvh_kernel(ray_ref, root_ref, lbox_hbm, ltri_hbm, out_ref, lbox_scratch, ltri_scratch, sem):
    rt = ray_ref[...]      # (8, 128)
    rb = root_ref[...]     # (8, 128)

    # Ray vectors (128, 1) for native (128 rays x 128 boxes/tris) 2D VPU ops
    rox = jnp.transpose(rt[0:1, :], (1, 0))
    roy = jnp.transpose(rt[1:2, :], (1, 0))
    roz = jnp.transpose(rt[2:3, :], (1, 0))
    rdx = jnp.transpose(rt[3:4, :], (1, 0))
    rdy = jnp.transpose(rt[4:5, :], (1, 0))
    rdz = jnp.transpose(rt[5:6, :], (1, 0))
    bd0 = rt[6:7, :]  # (1, 128)

    ids_1d = jnp.arange(128, dtype=jnp.int32).astype(jnp.float32)
    ids_row = jnp.broadcast_to(ids_1d[None, :], (1, 128))
    ids_2d = jnp.broadcast_to(ids_1d[None, :], (128, 128))

    # Top-level (128 rays x 128 top-nodes) slab test in VPU registers
    t_hit_mat, t_tmin_mat = _aabb_slab_128(
        rox, roy, roz, rdx, rdy, rdz,
        rb[0:1, :], rb[1:2, :], rb[2:3, :],
        rb[3:4, :], rb[4:5, :], rb[5:6, :],
        jnp.transpose(bd0, (1, 0)),
    )
    t_hit_mat = t_hit_mat & (rb[6:7, :] > 0.5)
    top_rem0 = jnp.any(t_hit_mat, axis=0, keepdims=True).astype(jnp.float32)  # (1, 128)
    top_tmin0 = jnp.min(jnp.where(t_hit_mat, t_tmin_mat, 1e9), axis=0, keepdims=True)

    # Pack loop state into 3 (8, 128) float32 tensors using concatenate (zero scatter ops!)
    # ray_st: [bd, bnx, bny, bnz, bu, bv, 0, 0]
    ray_st0 = jnp.concatenate([bd0, jnp.zeros((7, 128), jnp.float32)], axis=0)
    # top_st: row 0 = top_rem, row 1 = top_tmin, row 2 = cur_top_idx
    top_st0 = jnp.concatenate([top_rem0, top_tmin0, jnp.zeros((6, 128), jnp.float32)], axis=0)
    # leaf_st: row 0 = leaf_rem, row 1 = leaf_tmin, rows 2..7 = leaf aabbs
    leaf_st0 = jnp.zeros((8, 128), jnp.float32)

    def _cond(carry):
      r_st, t_st, l_st = carry
      has_work = (jnp.sum(l_st[0:1, :]) > 0.5) | (jnp.sum(t_st[0:1, :]) > 0.5)
      if any_hit:
        has_work = has_work & (jnp.max(r_st[0:1, :]) > 0.0)
      return has_work

    def _body(carry):
      r_st, t_st, l_st = carry
      has_leaf = jnp.sum(l_st[0:1, :]) > 0.5

      def _step_leaf(c):
        rs, ts, ls = c
        bd_col = jnp.transpose(rs[0:1, :], (1, 0))
        l_rem = ls[0:1, :] > 0.5
        l_tmin = ls[1:2, :]
        l_rel = jnp.argmin(jnp.where(l_rem, l_tmin, 1e9))
        new_l_rem = jnp.where(ids_row == l_rel.astype(jnp.float32), 0.0, ls[0:1, :])
        ls = jnp.concatenate([new_l_rem, ls[1:8, :]], axis=0)

        # Early AABB re-test against updated bd_col before issuing HBM DMA
        sel_mask = (ids_row == l_rel.astype(jnp.float32)).astype(jnp.float32)
        b0 = jnp.sum(ls[2:3, :] * sel_mask, axis=1, keepdims=True)
        b1 = jnp.sum(ls[3:4, :] * sel_mask, axis=1, keepdims=True)
        b2 = jnp.sum(ls[4:5, :] * sel_mask, axis=1, keepdims=True)
        b3 = jnp.sum(ls[5:6, :] * sel_mask, axis=1, keepdims=True)
        b4 = jnp.sum(ls[6:7, :] * sel_mask, axis=1, keepdims=True)
        b5 = jnp.sum(ls[7:8, :] * sel_mask, axis=1, keepdims=True)
        still_hit, _ = _aabb_slab_128(rox, roy, roz, rdx, rdy, rdz, b0, b1, b2, b3, b4, b5, bd_col)
        need_dma = jnp.any(still_hit)

        def _intersect_leaf(rs_in):
          cur_top = jnp.mean(ts[2:3, :]).astype(jnp.int32)
          leaf_id = cur_top * TOP_FAN + l_rel
          cp = pltpu.make_async_copy(ltri_hbm.at[leaf_id], ltri_scratch, sem)
          cp.start()
          cp.wait()
          tri = ltri_scratch[...]  # (24, 128)

          v0x, v0y, v0z = tri[0:1, :], tri[1:2, :], tri[2:3, :]
          e1x, e1y, e1z = tri[3:4, :], tri[4:5, :], tri[5:6, :]
          e2x, e2y, e2z = tri[6:7, :], tri[7:8, :], tri[8:9, :]

          hx = rdy * e2z - rdz * e2y
          hy = rdz * e2x - rdx * e2z
          hz = rdx * e2y - rdy * e2x
          a = e1x * hx + e1y * hy + e1z * hz
          ok_a = jnp.abs(a) > 1e-8
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

          s_bd = jnp.transpose(rs_in[0:1, :], (1, 0))
          hit_t = ok_a & (u >= 0.0) & (v >= 0.0) & ((u + v) <= 1.0) & (t >= 0.0) & (t < s_bd)
          t_cand = jnp.where(hit_t, t, MJ_MAXVAL)
          min_t = jnp.min(t_cand, axis=1, keepdims=True)
          imp = min_t < s_bd

          if any_hit:
            # Set occluded rays' remaining distance to 0.0 so subsequent tests skip them
            new_bd = jnp.where(imp, 0.0, s_bd)
            return jnp.concatenate([jnp.transpose(new_bd, (1, 0)), rs_in[1:8, :]], axis=0)

          win_idx = jnp.argmin(t_cand, axis=1, keepdims=True).astype(jnp.float32)
          win = (ids_2d == win_idx) & imp

          n0x, n0y, n0z = tri[9:10, :], tri[10:11, :], tri[11:12, :]
          n1x, n1y, n1z = tri[12:13, :], tri[13:14, :], tri[14:15, :]
          n2x, n2y, n2z = tri[15:16, :], tri[16:17, :], tri[17:18, :]
          u0x, u0y = tri[18:19, :], tri[19:20, :]
          du1x, du1y = tri[20:21, :], tri[21:22, :]
          du2x, du2y = tri[22:23, :], tri[23:24, :]

          bw = 1.0 - u - v
          lnx = jnp.sum(jnp.where(win, bw * n0x + u * n1x + v * n2x, 0.0), axis=1, keepdims=True)
          lny = jnp.sum(jnp.where(win, bw * n0y + u * n1y + v * n2y, 0.0), axis=1, keepdims=True)
          lnz = jnp.sum(jnp.where(win, bw * n0z + u * n1z + v * n2z, 0.0), axis=1, keepdims=True)
          inv_l = 1.0 / jnp.maximum(jnp.sqrt(lnx * lnx + lny * lny + lnz * lnz), 1e-12)
          lnx, lny, lnz = lnx * inv_l, lny * inv_l, lnz * inv_l

          lu = jnp.sum(jnp.where(win, u0x + u * du1x + v * du2x, 0.0), axis=1, keepdims=True)
          lv = jnp.sum(jnp.where(win, u0y + u * du1y + v * du2y, 0.0), axis=1, keepdims=True)

          imp_r = jnp.transpose(imp, (1, 0))
          r0 = jnp.where(imp_r, jnp.transpose(min_t, (1, 0)), rs_in[0:1, :])
          r1 = jnp.where(imp_r, jnp.transpose(lnx, (1, 0)), rs_in[1:2, :])
          r2 = jnp.where(imp_r, jnp.transpose(lny, (1, 0)), rs_in[2:3, :])
          r3 = jnp.where(imp_r, jnp.transpose(lnz, (1, 0)), rs_in[3:4, :])
          r4 = jnp.where(imp_r, jnp.transpose(lu, (1, 0)), rs_in[4:5, :])
          r5 = jnp.where(imp_r, jnp.transpose(lv, (1, 0)), rs_in[5:6, :])
          return jnp.concatenate([r0, r1, r2, r3, r4, r5, rs_in[6:8, :]], axis=0)

        rs = jax.lax.cond(need_dma, _intersect_leaf, lambda r: r, rs)
        return (rs, ts, ls)

      def _step_top(c):
        rs, ts, ls = c
        bd_col = jnp.transpose(rs[0:1, :], (1, 0))
        t_rem = ts[0:1, :] > 0.5
        t_tmin = ts[1:2, :]
        t_rel = jnp.argmin(jnp.where(t_rem, t_tmin, 1e9))
        new_t_rem = jnp.where(ids_row == t_rel.astype(jnp.float32), 0.0, ts[0:1, :])
        t_rel_row = jnp.broadcast_to(t_rel.astype(jnp.float32)[None, None], (1, 128))
        ts = jnp.concatenate([new_t_rem, ts[1:2, :], t_rel_row, ts[3:8, :]], axis=0)

        sel_mask = (ids_row == t_rel.astype(jnp.float32)).astype(jnp.float32)
        b0 = jnp.sum(rb[0:1, :] * sel_mask, axis=1, keepdims=True)
        b1 = jnp.sum(rb[1:2, :] * sel_mask, axis=1, keepdims=True)
        b2 = jnp.sum(rb[2:3, :] * sel_mask, axis=1, keepdims=True)
        b3 = jnp.sum(rb[3:4, :] * sel_mask, axis=1, keepdims=True)
        b4 = jnp.sum(rb[4:5, :] * sel_mask, axis=1, keepdims=True)
        b5 = jnp.sum(rb[5:6, :] * sel_mask, axis=1, keepdims=True)
        still_hit, _ = _aabb_slab_128(rox, roy, roz, rdx, rdy, rdz, b0, b1, b2, b3, b4, b5, bd_col)
        need_dma = jnp.any(still_hit)

        def _expand_top(_):
          cp = pltpu.make_async_copy(lbox_hbm.at[t_rel], lbox_scratch, sem)
          cp.start()
          cp.wait()
          lb = lbox_scratch[...]  # (8, 128)
          l_hit_mat, l_tmin_mat = _aabb_slab_128(
              rox, roy, roz, rdx, rdy, rdz,
              lb[0:1, :], lb[1:2, :], lb[2:3, :],
              lb[3:4, :], lb[4:5, :], lb[5:6, :],
              bd_col,
          )
          l_hit_mat = l_hit_mat & (lb[6:7, :] > 0.5)
          l_rem0 = jnp.any(l_hit_mat, axis=0, keepdims=True).astype(jnp.float32)
          l_tm0 = jnp.min(jnp.where(l_hit_mat, l_tmin_mat, 1e9), axis=0, keepdims=True)
          return jnp.concatenate([l_rem0, l_tm0, lb[0:6, :]], axis=0)

        ls_new = jax.lax.cond(need_dma, _expand_top, lambda _: ls, None)
        return (rs, ts, ls_new)

      return jax.lax.cond(has_leaf, _step_leaf, _step_top, carry)

    fin_r, _, _ = jax.lax.while_loop(_cond, _body, (ray_st0, top_st0, leaf_st0))
    out_ref[...] = fin_r

  out_2d = pl.pallas_call(
      _bvh_kernel,
      out_shape=jax.ShapeDtypeStruct((N_BLK * 8, BLOCK_R), jnp.float32),
      grid=(N_BLK,),
      in_specs=[
          pl.BlockSpec((8, BLOCK_R), lambda i: (i, 0)),
          pl.BlockSpec((8, TOP_FAN), lambda i: (0, 0)),
          pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
          pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
      ],
      out_specs=pl.BlockSpec((8, BLOCK_R), lambda i: (i, 0)),
      scratch_shapes=[
          pltpu.VMEM((8, TOP_FAN), jnp.float32),
          pltpu.VMEM((24, LEAF_SIZE), jnp.float32),
          pltpu.SemaphoreType.DMA,
      ],
      compiler_params=None if interpret else pltpu.CompilerParams(dimension_semantics=("parallel",)),
      interpret=interpret,
  )(rays_2d, item.root_box, item.leaf_boxes, item.leaf_tris)

  out_t = out_2d.reshape(B, n_tiles, 8, BLOCK_R).transpose(0, 1, 3, 2)  # (B, n_tiles, 128, 8)
  return out_t[..., 0], out_t[..., 1:4], out_t[..., 4:6]


def _jax_bvh_mesh_raycast(
    item: BVHMeshItem,
    ro_tile: jax.Array,   # (B, n_tiles, 128, 3)
    rd_tile: jax.Array,   # (B, n_tiles, 128, 3)
    tl_tile: jax.Array,   # (B, n_tiles, 128)
    any_hit: bool = False,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
  """Pure-JAX BVH-128 reference implementation matching _pallas_bvh_mesh_raycast."""
  B, n_tiles, _, _ = ro_tile.shape
  N_BLK = B * n_tiles
  ro_f = ro_tile.reshape(N_BLK, BLOCK_R, 3)
  rd_f = rd_tile.reshape(N_BLK, BLOCK_R, 3)
  tl_f = tl_tile.reshape(N_BLK, BLOCK_R)

  rb = jnp.asarray(item.root_box, jnp.float32)
  lbox_all = jnp.asarray(item.leaf_boxes, jnp.float32)
  ltri_all = jnp.asarray(item.leaf_tris, jnp.float32)

  def _trace_one_tile(ro, rd, tl):
    rox, roy, roz = ro[:, 0:1], ro[:, 1:2], ro[:, 2:3]
    rdx, rdy, rdz = rd[:, 0:1], rd[:, 1:2], rd[:, 2:3]
    bd = tl[:, None]
    bn = jnp.zeros((BLOCK_R, 3), jnp.float32)
    buv = jnp.zeros((BLOCK_R, 2), jnp.float32)

    t_hit, t_tm = _aabb_slab_128(
        rox, roy, roz, rdx, rdy, rdz,
        rb[0:1, :], rb[1:2, :], rb[2:3, :],
        rb[3:4, :], rb[4:5, :], rb[5:6, :],
        bd,
    )
    t_hit = t_hit & (rb[6:7, :] > 0.5)
    t_rem = jnp.any(t_hit, axis=0)
    t_tmin = jnp.min(jnp.where(t_hit, t_tm, 1e9), axis=0)
    l_rem = jnp.zeros((TOP_FAN,), jnp.bool_)
    l_tmin = jnp.full((TOP_FAN,), 1e9, jnp.float32)
    lb_cur = jnp.zeros((8, TOP_FAN), jnp.float32)
    cur_top = jnp.int32(0)

    def cond(st):
      d_s, _, _, tr, _, lr, _, _, _ = st
      ok = jnp.any(lr) | jnp.any(tr)
      if any_hit:
        ok = ok & (jnp.max(d_s) > 0.0)
      return ok

    def body(st):
      d_s, n_s, uv_s, tr, tt, lr, lt, lbc, ctop = st
      has_l = jnp.any(lr)

      def _do_l(s):
        d1, n1, uv1, tr1, tt1, lr1, lt1, lbc1, ctop1 = s
        l_rel = jnp.argmin(jnp.where(lr1, lt1, 1e9))
        lr1 = lr1 & (jnp.arange(TOP_FAN) != l_rel)
        lid = ctop1 * TOP_FAN + l_rel
        tri = ltri_all[lid]
        v0x, v0y, v0z = tri[0:1, :], tri[1:2, :], tri[2:3, :]
        e1x, e1y, e1z = tri[3:4, :], tri[4:5, :], tri[5:6, :]
        e2x, e2y, e2z = tri[6:7, :], tri[7:8, :], tri[8:9, :]

        hx = rdy * e2z - rdz * e2y
        hy = rdz * e2x - rdx * e2z
        hz = rdx * e2y - rdy * e2x
        a = e1x * hx + e1y * hy + e1z * hz
        ok_a = jnp.abs(a) > 1e-8
        f = jnp.where(ok_a, 1.0 / jnp.where(ok_a, a, 1.0), 0.0)
        sx, sy, sz = rox - v0x, roy - v0y, roz - v0z
        u = f * (sx * hx + sy * hy + sz * hz)
        qx = sy * e1z - sz * e1y
        qy = sz * e1x - sx * e1z
        qz = sx * e1y - sy * e1x
        v = f * (rdx * qx + rdy * qy + rdz * qz)
        t = f * (e2x * qx + e2y * qy + e2z * qz)
        hit_t = ok_a & (u >= 0.0) & (v >= 0.0) & ((u + v) <= 1.0) & (t >= 0.0) & (t < d1)
        t_cand = jnp.where(hit_t, t, MJ_MAXVAL)
        min_t = jnp.min(t_cand, axis=1, keepdims=True)
        imp = min_t < d1
        if any_hit:
          return (jnp.where(imp, 0.0, d1), n1, uv1, tr1, tt1, lr1, lt1, lbc1, ctop1)
        win_idx = jnp.argmin(t_cand, axis=1, keepdims=True)
        win = (jnp.arange(LEAF_SIZE)[None, :] == win_idx) & imp
        bw = 1.0 - u - v
        ln = jnp.stack([
            jnp.sum(jnp.where(win, bw * tri[9:10] + u * tri[12:13] + v * tri[15:16], 0.0), axis=1),
            jnp.sum(jnp.where(win, bw * tri[10:11] + u * tri[13:14] + v * tri[16:17], 0.0), axis=1),
            jnp.sum(jnp.where(win, bw * tri[11:12] + u * tri[14:15] + v * tri[17:18], 0.0), axis=1),
        ], axis=-1)
        ln = ln / jnp.maximum(jnp.linalg.norm(ln, axis=-1, keepdims=True), 1e-12)
        luv = jnp.stack([
            jnp.sum(jnp.where(win, tri[18:19] + u * tri[20:21] + v * tri[22:23], 0.0), axis=1),
            jnp.sum(jnp.where(win, tri[19:20] + u * tri[21:22] + v * tri[23:24], 0.0), axis=1),
        ], axis=-1)
        d1 = jnp.where(imp, min_t, d1)
        n1 = jnp.where(imp, ln, n1)
        uv1 = jnp.where(imp, luv, uv1)
        return (d1, n1, uv1, tr1, tt1, lr1, lt1, lbc1, ctop1)

      def _do_t(s):
        d1, n1, uv1, tr1, tt1, lr1, lt1, lbc1, ctop1 = s
        t_rel = jnp.argmin(jnp.where(tr1, tt1, 1e9))
        tr1 = tr1 & (jnp.arange(TOP_FAN) != t_rel)
        lbc_new = lbox_all[t_rel]
        lh, ltm = _aabb_slab_128(
            rox, roy, roz, rdx, rdy, rdz,
            lbc_new[0:1, :], lbc_new[1:2, :], lbc_new[2:3, :],
            lbc_new[3:4, :], lbc_new[4:5, :], lbc_new[5:6, :],
            d1,
        )
        lh = lh & (lbc_new[6:7, :] > 0.5)
        lr_new = jnp.any(lh, axis=0)
        lt_new = jnp.min(jnp.where(lh, ltm, 1e9), axis=0)
        return (d1, n1, uv1, tr1, tt1, lr_new, lt_new, lbc_new, t_rel)

      return jax.lax.cond(has_l, _do_l, _do_t, st)

    fd, fn, fuv, *_ = jax.lax.while_loop(cond, body, (bd, bn, buv, t_rem, t_tmin, l_rem, l_tmin, lb_cur, cur_top))
    return fd[:, 0], fn, fuv

  d_out, n_out, uv_out = jax.vmap(_trace_one_tile)(ro_f, rd_f, tl_f)
  return d_out.reshape(B, n_tiles, BLOCK_R), n_out.reshape(B, n_tiles, BLOCK_R, 3), uv_out.reshape(B, n_tiles, BLOCK_R, 2)


def _sample_textures_bilinear(
    tex_pack: jax.Array,  # (M, TH, TW, 4)
    gid: jax.Array,       # (B, H, W)
    uv: jax.Array,        # (B, H, W, 2)
) -> Tuple[jax.Array, jax.Array]:
  """Bilinearly sample 512x512 RGB linear albedo & roughness maps with REPEAT wrap."""
  _, TH, TW, _ = tex_pack.shape
  u_w = (uv[..., 0] % 1.0) * float(TW) - 0.5
  v_w = (uv[..., 1] % 1.0) * float(TH) - 0.5
  x0 = jnp.floor(u_w).astype(jnp.int32)
  y0 = jnp.floor(v_w).astype(jnp.int32)
  fx = (u_w - x0)[..., None]
  fy = (v_w - y0)[..., None]

  x0m = x0 % TW
  x1m = (x0 + 1) % TW
  y0m = y0 % TH
  y1m = (y0 + 1) % TH
  g_safe = jnp.maximum(gid, 0)

  c00 = tex_pack[g_safe, y0m, x0m]
  c10 = tex_pack[g_safe, y0m, x1m]
  c01 = tex_pack[g_safe, y1m, x0m]
  c11 = tex_pack[g_safe, y1m, x1m]
  samp = (1.0 - fx) * (1.0 - fy) * c00 + fx * (1.0 - fy) * c10 + (1.0 - fx) * fy * c01 + fx * fy * c11
  return samp[..., :3], samp[..., 3]


def render_bvh_batch(
    spec: BVHSceneSpec,
    obj_xpos: jax.Array,   # (B, M, 3)
    obj_xmat: jax.Array,   # (B, M, 3, 3)
    cam_xpos: jax.Array,   # (B, 3)
    cam_xmat: jax.Array,   # (B, 3, 3)
    mode: str = "pallas_bvh",
    interpret: bool = False,
) -> GBuffer:
  """Zero-copy batched 1M+ tri UV-textured raytracer on TPU v5p."""
  H, W = spec.height, spec.width
  ray_orig, ray_dir, neg_z = _compute_camera_rays_2d_tiled(spec, cam_xpos, cam_xmat)
  B = ray_orig.shape[0]
  max_dist = jnp.broadcast_to(((spec.zfar - spec.znear) / neg_z)[None, :, :], (B, H, W))

  ro_t = _tile_rays(ray_orig, H, W)
  rd_t = _tile_rays(ray_dir, H, W)
  best_d = _tile_rays(max_dist, H, W)
  n_tiles = ro_t.shape[1]

  best_gid = jnp.full((B, n_tiles, BLOCK_R), -1, dtype=jnp.int32)
  best_nrm = jnp.zeros((B, n_tiles, BLOCK_R, 3), dtype=jnp.float32)
  best_uv = jnp.zeros((B, n_tiles, BLOCK_R, 2), dtype=jnp.float32)

  for m_idx, item in enumerate(spec.meshes):
    p = obj_xpos[:, m_idx, :]
    rmat = obj_xmat[:, m_idx, :, :]
    ro_loc = jnp.einsum("bji,btrj->btri", rmat, ro_t - p[:, None, None, :], precision=jax.lax.Precision.HIGHEST)
    rd_loc = jnp.einsum("bji,btrj->btri", rmat, rd_t, precision=jax.lax.Precision.HIGHEST)

    if mode == "pallas_bvh":
      d_m, n_loc_m, uv_m = _pallas_bvh_mesh_raycast(item, ro_loc, rd_loc, best_d, any_hit=False, interpret=interpret)
    else:
      d_m, n_loc_m, uv_m = _jax_bvh_mesh_raycast(item, ro_loc, rd_loc, best_d, any_hit=False)

    imp = d_m < best_d
    n_w_m = jnp.einsum("bij,btrj->btri", rmat, n_loc_m, precision=jax.lax.Precision.HIGHEST)
    best_d = jnp.where(imp, d_m, best_d)
    best_gid = jnp.where(imp, jnp.int32(m_idx), best_gid)
    best_nrm = jnp.where(imp[..., None], n_w_m, best_nrm)
    best_uv = jnp.where(imp[..., None], uv_m, best_uv)

  best_d_hw = _untile_rays(best_d, H, W)
  best_gid_hw = _untile_rays(best_gid, H, W)
  best_nrm_hw = _untile_rays(best_nrm, H, W)
  best_uv_hw = _untile_rays(best_uv, H, W)

  hit = best_gid_hw >= 0
  best_d_hw = jnp.where(hit, best_d_hw, -1.0)
  best_nrm_hw = jnp.where(hit[..., None], best_nrm_hw, 0.0)
  vdir = -ray_dir
  flip = jnp.sum(best_nrm_hw * vdir, axis=-1, keepdims=True) < 0.0
  best_nrm_hw = jnp.where(flip, -best_nrm_hw, best_nrm_hw)

  hit_pos = ray_orig + jnp.where(hit, best_d_hw, 0.0)[..., None] * ray_dir
  alb_lin, rough = _sample_textures_bilinear(spec.tex_pack, best_gid_hw, best_uv_hw)
  alb_srgb = jnp.clip(_linear_to_srgb_jnp(alb_lin), 0.0, 1.0)

  # Primary OverheadKey + cool SideFill + hard shadow any_hit
  l_pos = jnp.asarray(spec.light_pos, dtype=jnp.float32)
  delta_l = l_pos[None, None, None, :] - hit_pos
  dist_l = jnp.linalg.norm(delta_l, axis=-1)
  L = delta_l / jnp.maximum(dist_l[..., None], 1e-12)
  ndotl = jnp.maximum(0.0, jnp.sum(best_nrm_hw * L, axis=-1))

  vis = jnp.ones((B, H, W), dtype=jnp.float32)
  if spec.use_shadows:
    sh_o_hw = hit_pos + best_nrm_hw * 2.5e-3
    sh_lim_hw = jnp.where(hit & (ndotl > 0.0), jnp.maximum(dist_l - 1e-3, 0.0), 0.0)
    sh_o_t = _tile_rays(sh_o_hw, H, W)
    L_t = _tile_rays(L, H, W)
    sh_lim_t = _tile_rays(sh_lim_hw, H, W)
    occ_t = jnp.zeros((B, n_tiles, BLOCK_R), dtype=jnp.bool_)

    for m_idx, item in enumerate(spec.meshes):
      p = obj_xpos[:, m_idx, :]
      rmat = obj_xmat[:, m_idx, :, :]
      ro_l = jnp.einsum("bji,btrj->btri", rmat, sh_o_t - p[:, None, None, :], precision=jax.lax.Precision.HIGHEST)
      rd_l = jnp.einsum("bji,btrj->btri", rmat, L_t, precision=jax.lax.Precision.HIGHEST)
      if mode == "pallas_bvh":
        d_sh, _, _ = _pallas_bvh_mesh_raycast(item, ro_l, rd_l, sh_lim_t, any_hit=True, interpret=interpret)
      else:
        d_sh, _, _ = _jax_bvh_mesh_raycast(item, ro_l, rd_l, sh_lim_t, any_hit=True)
      occ_t = occ_t | (d_sh < sh_lim_t)
    occ_hw = _untile_rays(occ_t, H, W)
    vis = jnp.where(occ_hw, 0.0, 1.0)

  # Linear PBR-like direct shading matching Blender area lights + Reinhard + sRGB
  l_fill_dir = jnp.array([-1.20, -1.10, 1.65], dtype=jnp.float32)
  l_fill_dir = l_fill_dir / jnp.linalg.norm(l_fill_dir)
  ndotl_fill = jnp.maximum(0.0, jnp.sum(best_nrm_hw * l_fill_dir[None, None, None, :], axis=-1))

  met = spec.metallic_tbl[jnp.maximum(best_gid_hw, 0)]
  shin = jnp.clip(2.0 / jnp.maximum(rough ** 2, 0.02) - 2.0, 4.0, 96.0)
  H_vec = (L + vdir) / jnp.maximum(jnp.linalg.norm(L + vdir, axis=-1, keepdims=True), 1e-12)
  spec_w = (jnp.maximum(0.0, jnp.sum(best_nrm_hw * H_vec, axis=-1)) ** shin) * vis * 0.45
  spec_col = (1.0 - met)[..., None] * 1.0 + met[..., None] * alb_lin

  # Linear irradiance
  amb_lin = jnp.array([0.11, 0.14, 0.18], dtype=jnp.float32)
  key_lin = jnp.array([1.15, 1.08, 0.98], dtype=jnp.float32)
  fill_lin = jnp.array([0.24, 0.28, 0.34], dtype=jnp.float32)

  diff_col = alb_lin * (1.0 - 0.65 * met[..., None])
  rad_lin = (
      diff_col * (amb_lin[None, None, None, :] + key_lin[None, None, None, :] * (ndotl * vis)[..., None] + fill_lin[None, None, None, :] * ndotl_fill[..., None])
      + spec_col * (key_lin[None, None, None, :] * spec_w[..., None])
  )
  # Exact Blender post-process: Reinhard 0.15 + sRGB
  rad_tm = rad_lin / (1.0 + 0.15 * rad_lin)
  direct_srgb = jnp.clip(_linear_to_srgb_jnp(rad_tm), 0.0, 1.0)

  sky_t = jnp.clip(0.5 * (ray_dir[..., 2] + 1.0), 0.0, 1.0)[..., None]
  sky_rgb = (
      sky_t * jnp.asarray(spec.sky_rgb1, dtype=jnp.float32)[None, None, None, :]
      + (1.0 - sky_t) * jnp.asarray(spec.sky_rgb2, dtype=jnp.float32)[None, None, None, :]
  )
  # Match world background tone (~0.34, 0.42, 0.54 * 0.48 -> sRGB ~0.43, 0.48, 0.54)
  sky_bg = jnp.broadcast_to(jnp.array([0.42, 0.47, 0.54], dtype=jnp.float32)[None, None, None, :], ray_dir.shape)
  sky_rgb = 0.7 * sky_bg + 0.3 * sky_rgb

  rgb = jnp.where(hit[..., None], direct_srgb, sky_rgb)
  inv_cos = (1.0 / neg_z)[None, :, :]
  euclid_d = jnp.where(hit, best_d_hw + float(spec.znear) * inv_cos, -1.0)
  z_d = jnp.where(hit, float(spec.znear) + best_d_hw * neg_z[None, :, :], -1.0)

  return GBuffer(
      rgb=rgb,
      albedo=jnp.where(hit[..., None], alb_srgb, sky_rgb),
      depth=euclid_d,
      z_depth=z_d,
      normal=best_nrm_hw,
      view_dir=vdir,
      world_pos=jnp.where(hit[..., None], hit_pos, 0.0),
      geom_id=best_gid_hw,
      direct_shadow=jnp.where(hit, vis, 1.0),
      direct_rgb=rgb,
  )
