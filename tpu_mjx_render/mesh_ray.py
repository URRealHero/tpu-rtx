"""V2 Static Cluster-BVH Triangle Mesh Raytracer for Arbitrary Blender / CAD Assets.

Zero dynamic pointers, zero BVH rebuild when rigid bodies move:
- Offline `build_mesh_scene_spec` sorts each mesh's triangles by 3D Morton code
  and packs them into fixed-size K=16 triangle clusters with local cluster AABBs.
- Online `render_mesh_batch` transforms rays into each object's local frame from
  batched (B, M, 3) / (B, M, 3, 3) poses and performs two-level slab AABB culling
  + branchless Moller-Thumbore intersection + barycentric smooth vertex normals.
- Includes `pallas_mesh_raycast` VMEM-resident Pallas kernel (BLOCK_R=128).
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Sequence, Tuple

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
import numpy as np

from tpu_mjx_render.pallas_v2 import _compute_batch_camera_rays_soa
from tpu_mjx_render.primitives import MJ_MAXVAL
from tpu_mjx_render.renderer import GBuffer

CLUSTER_SIZE = 16
BLOCK_R = 128


@dataclasses.dataclass(frozen=True)
class MeshSpecItem:
  mesh_id: int
  name: str
  n_tris_orig: int
  n_clusters: int
  obj_aabb_min: np.ndarray     # (3,)
  obj_aabb_max: np.ndarray     # (3,)
  cluster_min: np.ndarray      # (C, 3)
  cluster_max: np.ndarray      # (C, 3)
  v0: np.ndarray               # (C, K, 3)
  e1: np.ndarray               # (C, K, 3)
  e2: np.ndarray               # (C, K, 3)
  n0: np.ndarray               # (C, K, 3)
  n1: np.ndarray               # (C, K, 3)
  n2: np.ndarray               # (C, K, 3)
  tri_valid: np.ndarray        # (C, K) bool
  rgba: np.ndarray             # (4,)


@dataclasses.dataclass(frozen=True)
class MeshAssetSpec:
  height: int
  width: int
  cam_fovy: float
  cam_projection: int
  cam_sensorsize: Tuple[float, float]
  cam_intrinsic: Tuple[float, float, float, float]
  znear: float
  zfar: float
  meshes: Tuple[MeshSpecItem, ...]
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
  fn = np.cross(v1 - v0, v2 - v0)  # area-weighted face normal
  vn = np.zeros_like(verts, dtype=np.float32)
  np.add.at(vn, faces[:, 0], fn)
  np.add.at(vn, faces[:, 1], fn)
  np.add.at(vn, faces[:, 2], fn)
  norms = np.linalg.norm(vn, axis=-1, keepdims=True)
  return (vn / np.maximum(norms, 1e-12)).astype(np.float32)


def _morton3d_10bit(pts: np.ndarray) -> np.ndarray:
  """Compute 30-bit Morton code for quantized [0, 1023]^3 triangle centroids."""
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


def build_mesh_scene_spec(
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
    cluster_size: int = CLUSTER_SIZE,
) -> MeshAssetSpec:
  """Build Morton-ordered static K=16 triangle cluster hierarchy from arbitrary meshes."""
  items: List[MeshSpecItem] = []
  for mid, md in enumerate(mesh_dicts):
    verts = np.asarray(md["verts"], dtype=np.float32)
    faces = np.asarray(md["faces"], dtype=np.int32)
    if "vert_normals" in md and md["vert_normals"] is not None:
      vnorm = np.asarray(md["vert_normals"], dtype=np.float32)
    else:
      vnorm = _compute_smooth_vertex_normals(verts, faces)

    rgba = np.asarray(md.get("rgba", [0.7, 0.7, 0.7, 1.0]), dtype=np.float32)
    if rgba.shape[0] == 3:
      rgba = np.concatenate([rgba, np.array([1.0], dtype=np.float32)])

    v0_all = verts[faces[:, 0]]
    v1_all = verts[faces[:, 1]]
    v2_all = verts[faces[:, 2]]
    centroids = (v0_all + v1_all + v2_all) / 3.0
    order = np.argsort(_morton3d_10bit(centroids), kind="stable")
    faces = faces[order]

    F = int(faces.shape[0])
    C = (F + cluster_size - 1) // cluster_size
    F_pad = C * cluster_size

    v0 = np.zeros((F_pad, 3), dtype=np.float32)
    e1 = np.zeros((F_pad, 3), dtype=np.float32)
    e2 = np.zeros((F_pad, 3), dtype=np.float32)
    n0 = np.zeros((F_pad, 3), dtype=np.float32)
    n1 = np.zeros((F_pad, 3), dtype=np.float32)
    n2 = np.zeros((F_pad, 3), dtype=np.float32)
    tri_valid = np.zeros((F_pad,), dtype=bool)

    tv0 = verts[faces[:, 0]]
    tv1 = verts[faces[:, 1]]
    tv2 = verts[faces[:, 2]]
    v0[:F] = tv0
    e1[:F] = tv1 - tv0
    e2[:F] = tv2 - tv0
    n0[:F] = vnorm[faces[:, 0]]
    n1[:F] = vnorm[faces[:, 1]]
    n2[:F] = vnorm[faces[:, 2]]
    tri_valid[:F] = True

    # Pad unused cluster slots with degenerate copy of last valid tri so AABB stays tight
    if F_pad > F:
      v0[F:] = tv0[-1]
      n0[F:] = n0[F - 1]
      n1[F:] = n1[F - 1]
      n2[F:] = n2[F - 1]

    v0_c = v0.reshape(C, cluster_size, 3)
    e1_c = e1.reshape(C, cluster_size, 3)
    e2_c = e2.reshape(C, cluster_size, 3)
    pts_c = np.stack([v0_c, v0_c + e1_c, v0_c + e2_c], axis=2)  # (C, K, 3, 3)
    cmin = np.min(pts_c, axis=(1, 2)) - 1e-4
    cmax = np.max(pts_c, axis=(1, 2)) + 1e-4

    obj_min = np.min(cmin, axis=0)
    obj_max = np.max(cmax, axis=0)

    items.append(
        MeshSpecItem(
            mesh_id=mid,
            name=str(md.get("name", f"mesh_{mid}")),
            n_tris_orig=F,
            n_clusters=C,
            obj_aabb_min=obj_min.astype(np.float32),
            obj_aabb_max=obj_max.astype(np.float32),
            cluster_min=cmin.astype(np.float32),
            cluster_max=cmax.astype(np.float32),
            v0=v0_c,
            e1=e1_c,
            e2=e2_c,
            n0=n0.reshape(C, cluster_size, 3),
            n1=n1.reshape(C, cluster_size, 3),
            n2=n2.reshape(C, cluster_size, 3),
            tri_valid=tri_valid.reshape(C, cluster_size),
            rgba=rgba,
        )
    )

  return MeshAssetSpec(
      height=int(height),
      width=int(width),
      cam_fovy=float(cam_fovy),
      cam_projection=0,
      cam_sensorsize=(0.0, 0.0),
      cam_intrinsic=(0.0, 0.0, 0.0, 0.0),
      znear=float(znear),
      zfar=float(zfar),
      meshes=tuple(items),
      light_pos=np.asarray(light_pos, dtype=np.float32),
      light_ambient=light_ambient,
      light_diffuse=light_diffuse,
      light_specular=light_specular,
      sky_rgb1=sky_rgb1,
      sky_rgb2=sky_rgb2,
      use_shadows=bool(use_shadows),
  )


def _aabb_slab_hit(ro: jax.Array, rd: jax.Array, bmin: jax.Array, bmax: jax.Array, t_limit: jax.Array) -> jax.Array:
  """Branchless slab AABB test over arbitrary broadcastable trailing (3,) axes."""
  inv_d = jnp.where(jnp.abs(rd) > 1e-12, 1.0 / jnp.where(jnp.abs(rd) > 1e-12, rd, 1.0), 1e12)
  t0 = (bmin - ro) * inv_d
  t1 = (bmax - ro) * inv_d
  tmin = jnp.max(jnp.minimum(t0, t1), axis=-1)
  tmax = jnp.min(jnp.maximum(t0, t1), axis=-1)
  return (tmax >= jnp.maximum(tmin, 0.0)) & (tmin < t_limit)


def _intersect_mesh_clusters_tiled(
    item: MeshSpecItem,
    ro_loc: jax.Array,
    rd_loc: jax.Array,
    t_limit: jax.Array,
    tile_size: int = 128,
    any_hit: bool = False,
) -> Tuple[jax.Array, jax.Array]:
  """Cast (R, 3) local rays against static K=16 Morton triangle clusters with 2-level AABB cull."""
  R = ro_loc.shape[0]
  pad = (tile_size - (R % tile_size)) % tile_size
  if pad > 0:
    ro_loc = jnp.pad(ro_loc, ((0, pad), (0, 0)))
    rd_loc = jnp.pad(rd_loc, ((0, pad), (0, 0)), constant_values=1.0 / np.sqrt(3.0))
    t_limit = jnp.pad(t_limit, ((0, pad),), constant_values=0.0)

  n_tiles = ro_loc.shape[0] // tile_size
  ro_t = ro_loc.reshape(n_tiles, tile_size, 3)
  rd_t = rd_loc.reshape(n_tiles, tile_size, 3)
  tl_t = t_limit.reshape(n_tiles, tile_size)

  omin = jnp.asarray(item.obj_aabb_min, dtype=jnp.float32)
  omax = jnp.asarray(item.obj_aabb_max, dtype=jnp.float32)
  cmin = jnp.asarray(item.cluster_min, dtype=jnp.float32)  # (C, 3)
  cmax = jnp.asarray(item.cluster_max, dtype=jnp.float32)  # (C, 3)
  v0 = jnp.asarray(item.v0, dtype=jnp.float32)            # (C, K, 3)
  e1 = jnp.asarray(item.e1, dtype=jnp.float32)
  e2 = jnp.asarray(item.e2, dtype=jnp.float32)
  n0 = jnp.asarray(item.n0, dtype=jnp.float32)
  n1 = jnp.asarray(item.n1, dtype=jnp.float32)
  n2 = jnp.asarray(item.n2, dtype=jnp.float32)
  tval = jnp.asarray(item.tri_valid, dtype=jnp.bool_)    # (C, K)

  C = item.n_clusters

  def _eval_tile_clusters(ro_b, rd_b, tl_b):
    # Test all cluster AABBs for this ray tile: (T, C)
    c_hit = _aabb_slab_hit(ro_b[:, None, :], rd_b[:, None, :], cmin[None, :, :], cmax[None, :, :], tl_b[:, None])

    def _cluster_body(carry, c_idx):
      bd, bnx, bny, bnz = carry
      cmask = c_hit[:, c_idx]

      def _run_tris(_):
        # Moller-Thumbore over (T, K, 3)
        cv0 = v0[c_idx][None, :, :]
        ce1 = e1[c_idx][None, :, :]
        ce2 = e2[c_idx][None, :, :]
        ctv = tval[c_idx][None, :]

        r_o = ro_b[:, None, :]
        r_d = rd_b[:, None, :]

        h = jnp.cross(r_d, ce2)
        a = jnp.sum(ce1 * h, axis=-1)
        ok_a = ctv & cmask[:, None] & (jnp.abs(a) > 1e-8)
        f = jnp.where(ok_a, 1.0 / jnp.where(ok_a, a, 1.0), 0.0)
        s = r_o - cv0
        u = f * jnp.sum(s * h, axis=-1)
        q = jnp.cross(s, ce1)
        v = f * jnp.sum(r_d * q, axis=-1)
        t = f * jnp.sum(ce2 * q, axis=-1)

        hit_k = ok_a & (u >= 0.0) & (v >= 0.0) & ((u + v) <= 1.0) & (t >= 0.0) & (t < bd[:, None])
        t_cand = jnp.where(hit_k, t, MJ_MAXVAL)
        best_k = jnp.argmin(t_cand, axis=1)
        min_t = jnp.take_along_axis(t_cand, best_k[:, None], axis=1)[:, 0]
        imp = min_t < bd

        new_d = jnp.where(imp, min_t, bd)
        if any_hit:
          return (new_d, bnx, bny, bnz)

        bu = jnp.take_along_axis(u, best_k[:, None], axis=1)[:, 0]
        bv = jnp.take_along_axis(v, best_k[:, None], axis=1)[:, 0]
        bw = 1.0 - bu - bv
        ln_tri = (
            bw[:, None] * n0[c_idx][best_k]
            + bu[:, None] * n1[c_idx][best_k]
            + bv[:, None] * n2[c_idx][best_k]
        )
        inv_ln = 1.0 / jnp.maximum(jnp.linalg.norm(ln_tri, axis=-1, keepdims=True), 1e-12)
        ln_tri = ln_tri * inv_ln
        return (
            new_d,
            jnp.where(imp, ln_tri[:, 0], bnx),
            jnp.where(imp, ln_tri[:, 1], bny),
            jnp.where(imp, ln_tri[:, 2], bnz),
        )

      def _skip_tris(_):
        return (bd, bnx, bny, bnz)

      nxt = jax.lax.cond(jnp.any(cmask), _run_tris, _skip_tris, operand=None)
      return nxt, None

    init_state = (tl_b, jnp.zeros((tile_size,), jnp.float32), jnp.zeros((tile_size,), jnp.float32), jnp.zeros((tile_size,), jnp.float32))
    (od, onx, ony, onz), _ = jax.lax.scan(_cluster_body, init_state, jnp.arange(C, dtype=jnp.int32))
    return od, jnp.stack([onx, ony, onz], axis=-1)

  def _tile_scan(_, xs):
    ro_b, rd_b, tl_b = xs
    obj_hit = _aabb_slab_hit(ro_b, rd_b, omin[None, :], omax[None, :], tl_b)

    def _run_obj(_):
      return _eval_tile_clusters(ro_b, rd_b, tl_b)

    def _skip_obj(_):
      return (tl_b, jnp.zeros((tile_size, 3), jnp.float32))

    res = jax.lax.cond(jnp.any(obj_hit), _run_obj, _skip_obj, operand=None)
    return None, res

  _, (d_all, n_all) = jax.lax.scan(_tile_scan, None, (ro_t, rd_t, tl_t))
  return d_all.reshape(-1)[:R], n_all.reshape(-1, 3)[:R]


def _pallas_mesh_item_raycast(
    item: MeshSpecItem,
    ro_loc: jax.Array,
    rd_loc: jax.Array,
    t_limit: jax.Array,
    interpret: bool = True,
) -> Tuple[jax.Array, jax.Array]:
  """Native Mosaic VMEM Pallas kernel streaming Morton macro-clusters of 128 tris.

  Strictly 2D (128 rays x 128 tris) VPU operations with zero `dynamic_slice`,
  zero `gather`/`take_along_axis`, and `jax.lax.cond` macro-cluster AABB culling.
  """
  B, R, _ = ro_loc.shape
  R_pad = ((R + BLOCK_R - 1) // BLOCK_R) * BLOCK_R
  n_r_blk = R_pad // BLOCK_R

  ro_flat = jnp.pad(ro_loc, ((0, 0), (0, R_pad - R), (0, 0))).reshape(B * R_pad, 3)
  rd_flat = jnp.pad(rd_loc, ((0, 0), (0, R_pad - R), (0, 0)), constant_values=1.0 / np.sqrt(3.0)).reshape(B * R_pad, 3)
  tl_flat = jnp.pad(t_limit, ((0, 0), (0, R_pad - R)), constant_values=0.0).reshape(B * R_pad, 1)

  rays_8 = jnp.concatenate([ro_flat, rd_flat, tl_flat, jnp.zeros_like(tl_flat)], axis=-1)
  rays_2d = rays_8.reshape(B * n_r_blk, BLOCK_R, 8).transpose(0, 2, 1).reshape(B * n_r_blk * 8, BLOCK_R)

  # Group K=16 Morton clusters into 128-triangle macro-clusters (8 clusters = 128 tris)
  C = item.n_clusters
  n_macro = (C + 7) // 8
  C_pad = n_macro * 8
  F_macro = n_macro * 128

  def _pad_tri_field(arr_c_k_3):
    flat = arr_c_k_3.reshape(C * CLUSTER_SIZE, 3)
    out = np.zeros((F_macro, 3), dtype=np.float32)
    out[: flat.shape[0]] = flat
    return out.reshape(n_macro, 128, 3)

  v0_m = _pad_tri_field(item.v0)
  e1_m = _pad_tri_field(item.e1)
  e2_m = _pad_tri_field(item.e2)
  n0_m = _pad_tri_field(item.n0)
  n1_m = _pad_tri_field(item.n1)
  n2_m = _pad_tri_field(item.n2)
  tv_f = np.zeros((F_macro,), dtype=np.float32)
  tv_f[: C * CLUSTER_SIZE] = item.tri_valid.reshape(-1).astype(np.float32)
  tv_m = tv_f.reshape(n_macro, 128)

  cmin_p = np.full((C_pad, 3), 1e6, dtype=np.float32)
  cmax_p = np.full((C_pad, 3), -1e6, dtype=np.float32)
  cmin_p[:C] = item.cluster_min
  cmax_p[:C] = item.cluster_max
  mmin = np.min(cmin_p.reshape(n_macro, 8, 3), axis=1)  # (n_macro, 3)
  mmax = np.max(cmax_p.reshape(n_macro, 8, 3), axis=1)

  # Pack all macro-clusters into a single 2D array (n_macro * 24, 128) aligned to (8, 128):
  # rows 0..18: v0(3), e1(3), e2(3), n0(3), n1(3), n2(3), tv(1) across 128 tris
  # row 19: macro aabb_min(0:3), macro aabb_max(3:6), obj_min(6:9), obj_max(9:12)
  pack_np = np.zeros((n_macro * 24, 128), dtype=np.float32)
  for m_i in range(n_macro):
    base = m_i * 24
    pack_np[base + 0 : base + 3] = v0_m[m_i].T
    pack_np[base + 3 : base + 6] = e1_m[m_i].T
    pack_np[base + 6 : base + 9] = e2_m[m_i].T
    pack_np[base + 9 : base + 12] = n0_m[m_i].T
    pack_np[base + 12 : base + 15] = n1_m[m_i].T
    pack_np[base + 15 : base + 18] = n2_m[m_i].T
    pack_np[base + 18] = tv_m[m_i]
    pack_np[base + 19, 0:3] = mmin[m_i]
    pack_np[base + 19, 3:6] = mmax[m_i]
    pack_np[base + 19, 6:9] = item.obj_aabb_min
    pack_np[base + 19, 9:12] = item.obj_aabb_max
  pack_dev = jnp.asarray(pack_np, dtype=jnp.float32)

  def _aabb_hit_soa(rx, ry, rz, dx, dy, dz, bx0, by0, bz0, bx1, by1, bz1, tl):
    def _slab(o, d, mn, mx):
      ok_d = jnp.abs(d) > 1e-12
      id_ = jnp.where(ok_d, 1.0 / jnp.where(ok_d, d, 1.0), 1e12)
      t0, t1 = (mn - o) * id_, (mx - o) * id_
      return jnp.minimum(t0, t1), jnp.maximum(t0, t1)
    tx0, tx1 = _slab(rx, dx, bx0, bx1)
    ty0, ty1 = _slab(ry, dy, by0, by1)
    tz0, tz1 = _slab(rz, dz, bz0, bz1)
    tmin = jnp.maximum(tx0, jnp.maximum(ty0, tz0))
    tmax = jnp.minimum(tx1, jnp.minimum(ty1, tz1))
    return (tmax >= jnp.maximum(tmin, 0.0)) & (tmin < tl)

  def _mesh_kernel(ray_ref, pk_ref, out_ref):
    rt = ray_ref[...]  # (8, 128)
    pk = pk_ref[...]   # (n_macro * 24, 128)

    # Transpose ray coords into (128, 1) column vectors so (128 rays x 128 tris) is native 2D (128, 128)!
    rox = jnp.transpose(rt[0:1, :], (1, 0))
    roy = jnp.transpose(rt[1:2, :], (1, 0))
    roz = jnp.transpose(rt[2:3, :], (1, 0))
    rdx = jnp.transpose(rt[3:4, :], (1, 0))
    rdy = jnp.transpose(rt[4:5, :], (1, 0))
    rdz = jnp.transpose(rt[5:6, :], (1, 0))
    bd = jnp.transpose(rt[6:7, :], (1, 0))  # (128, 1)
    bnx = jnp.zeros((BLOCK_R, 1), jnp.float32)
    bny = jnp.zeros((BLOCK_R, 1), jnp.float32)
    bnz = jnp.zeros((BLOCK_R, 1), jnp.float32)

    obj_hit = jnp.any(
        _aabb_hit_soa(
            rox, roy, roz, rdx, rdy, rdz,
            pk[19:20, 6:7], pk[19:20, 7:8], pk[19:20, 8:9],
            pk[19:20, 9:10], pk[19:20, 10:11], pk[19:20, 11:12],
            bd,
        )
    )

    def _run_macros(_):
      cbd, cnx, cny, cnz = bd, bnx, bny, bnz
      for m_i in range(n_macro):
        base = m_i * 24
        m_hit = jnp.any(
            _aabb_hit_soa(
                rox, roy, roz, rdx, rdy, rdz,
                pk[base + 19 : base + 20, 0:1], pk[base + 19 : base + 20, 1:2], pk[base + 19 : base + 20, 2:3],
                pk[base + 19 : base + 20, 3:4], pk[base + 19 : base + 20, 4:5], pk[base + 19 : base + 20, 5:6],
                cbd,
            )
        )

        def _do_macro(state, _b=base):
          s_bd, s_nx, s_ny, s_nz = state
          v0x, v0y, v0z = pk[_b + 0 : _b + 1, :], pk[_b + 1 : _b + 2, :], pk[_b + 2 : _b + 3, :]
          e1x, e1y, e1z = pk[_b + 3 : _b + 4, :], pk[_b + 4 : _b + 5, :], pk[_b + 5 : _b + 6, :]
          e2x, e2y, e2z = pk[_b + 6 : _b + 7, :], pk[_b + 7 : _b + 8, :], pk[_b + 8 : _b + 9, :]
          n0x, n0y, n0z = pk[_b + 9 : _b + 10, :], pk[_b + 10 : _b + 11, :], pk[_b + 11 : _b + 12, :]
          n1x, n1y, n1z = pk[_b + 12 : _b + 13, :], pk[_b + 13 : _b + 14, :], pk[_b + 14 : _b + 15, :]
          n2x, n2y, n2z = pk[_b + 15 : _b + 16, :], pk[_b + 16 : _b + 17, :], pk[_b + 17 : _b + 18, :]
          tv = pk[_b + 18 : _b + 19, :] > 0.5

          # Strictly 2D (128 rays x 128 tris) Moller-Thumbore!
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

          hit_t = ok_a & (u >= 0.0) & (v >= 0.0) & ((u + v) <= 1.0) & (t >= 0.0) & (t < s_bd)
          t_cand = jnp.where(hit_t, t, MJ_MAXVAL)
          min_t = jnp.min(t_cand, axis=1, keepdims=True)
          imp = min_t < s_bd

          # Mask out winning triangle per ray (exact, zero dynamic gather!)
          win = hit_t & (t_cand == min_t)
          # Tie-break safely via first-hit cumsum in rare exact tie, or masked sum normalized
          bw = 1.0 - u - v
          lnx = jnp.sum(jnp.where(win, bw * n0x + u * n1x + v * n2x, 0.0), axis=1, keepdims=True)
          lny = jnp.sum(jnp.where(win, bw * n0y + u * n1y + v * n2y, 0.0), axis=1, keepdims=True)
          lnz = jnp.sum(jnp.where(win, bw * n0z + u * n1z + v * n2z, 0.0), axis=1, keepdims=True)
          inv_l = 1.0 / jnp.maximum(jnp.sqrt(lnx * lnx + lny * lny + lnz * lnz), 1e-12)
          lnx, lny, lnz = lnx * inv_l, lny * inv_l, lnz * inv_l

          return (
              jnp.where(imp, min_t, s_bd),
              jnp.where(imp, lnx, s_nx),
              jnp.where(imp, lny, s_ny),
              jnp.where(imp, lnz, s_nz),
          )

        cbd, cnx, cny, cnz = jax.lax.cond(m_hit, _do_macro, lambda s: s, (cbd, cnx, cny, cnz))
      return cbd, cnx, cny, cnz

    fd, fnx, fny, fnz = jax.lax.cond(obj_hit, _run_macros, lambda _: (bd, bnx, bny, bnz), None)
    z = jnp.zeros((1, BLOCK_R), jnp.float32)
    out_ref[...] = jnp.concatenate(
        [
            jnp.transpose(fd, (1, 0)),
            jnp.transpose(fnx, (1, 0)),
            jnp.transpose(fny, (1, 0)),
            jnp.transpose(fnz, (1, 0)),
            z, z, z, z,
        ],
        axis=0,
    )

  out_2d = pl.pallas_call(
      _mesh_kernel,
      out_shape=jax.ShapeDtypeStruct((B * n_r_blk * 8, BLOCK_R), jnp.float32),
      grid=(B * n_r_blk,),
      in_specs=[
          pl.BlockSpec((8, BLOCK_R), lambda i: (i, 0)),
          pl.BlockSpec((n_macro * 24, 128), lambda i: (0, 0)),
      ],
      out_specs=pl.BlockSpec((8, BLOCK_R), lambda i: (i, 0)),
      compiler_params=None if interpret else pltpu.CompilerParams(dimension_semantics=("parallel",)),
      interpret=interpret,
  )(rays_2d, pack_dev)

  out_tiles = out_2d.reshape(B, n_r_blk, 8, BLOCK_R).transpose(0, 1, 3, 2).reshape(B, R_pad, 8)[:, :R, :]
  return out_tiles[..., 0], out_tiles[..., 1:4]


def render_mesh_batch(
    spec: MeshAssetSpec,
    obj_xpos: jax.Array,
    obj_xmat: jax.Array,
    cam_xpos: jax.Array,
    cam_xmat: jax.Array,
    mode: str = "jax_cluster",
    interpret: bool = True,
) -> GBuffer:
  """Zero-copy batched triangle mesh raytracer consuming dynamic rigid poses (B, M, 3/3x3)."""
  ray_orig, ray_dir, neg_z = _compute_batch_camera_rays_soa(spec, cam_xpos, cam_xmat)
  B, R, _ = ray_orig.shape
  H, W = spec.height, spec.width
  max_dist = jnp.broadcast_to(((spec.zfar - spec.znear) / neg_z)[None, :], (B, R))

  best_d = max_dist
  best_gid = jnp.full((B, R), -1, dtype=jnp.int32)
  best_nrm = jnp.zeros((B, R, 3), dtype=jnp.float32)

  for m_idx, item in enumerate(spec.meshes):
    p = obj_xpos[:, m_idx, :]   # (B, 3)
    rmat = obj_xmat[:, m_idx, :, :]  # (B, 3, 3)
    ro_loc = jnp.einsum("bji,brj->bri", rmat, ray_orig - p[:, None, :], precision=jax.lax.Precision.HIGHEST)
    rd_loc = jnp.einsum("bji,brj->bri", rmat, ray_dir, precision=jax.lax.Precision.HIGHEST)

    if mode == "pallas":
      d_m, n_loc_m = _pallas_mesh_item_raycast(item, ro_loc, rd_loc, best_d, interpret=interpret)
    else:
      d_m, n_loc_m = jax.vmap(
          lambda ro, rd, tl: _intersect_mesh_clusters_tiled(item, ro, rd, tl, tile_size=128, any_hit=False)
      )(ro_loc, rd_loc, best_d)

    imp = d_m < best_d
    n_w_m = jnp.einsum("bij,brj->bri", rmat, n_loc_m, precision=jax.lax.Precision.HIGHEST)
    best_d = jnp.where(imp, d_m, best_d)
    best_gid = jnp.where(imp, jnp.int32(m_idx), best_gid)
    best_nrm = jnp.where(imp[..., None], n_w_m, best_nrm)

  hit = best_gid >= 0
  best_d = jnp.where(hit, best_d, -1.0)
  best_nrm = jnp.where(hit[..., None], best_nrm, 0.0)
  # Orient normal toward viewer for double-sided CAD meshes
  vdir = -ray_dir
  flip = jnp.sum(best_nrm * vdir, axis=-1, keepdims=True) < 0.0
  best_nrm = jnp.where(flip, -best_nrm, best_nrm)

  hit_pos = ray_orig + jnp.where(hit, best_d, 0.0)[..., None] * ray_dir
  rgba_tbl = jnp.asarray([it.rgba[:3] for it in spec.meshes], dtype=jnp.float32)
  albedo = rgba_tbl[jnp.maximum(best_gid, 0)]

  # Studio directional/point shading + mesh shadow any_hit
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
      p = obj_xpos[:, m_idx, :]
      rmat = obj_xmat[:, m_idx, :, :]
      ro_l = jnp.einsum("bji,brj->bri", rmat, sh_o - p[:, None, :], precision=jax.lax.Precision.HIGHEST)
      rd_l = jnp.einsum("bji,brj->bri", rmat, L, precision=jax.lax.Precision.HIGHEST)
      if mode == "pallas":
        d_sh, _ = _pallas_mesh_item_raycast(item, ro_l, rd_l, sh_lim, interpret=interpret)
      else:
        d_sh, _ = jax.vmap(
            lambda ro, rd, tl: _intersect_mesh_clusters_tiled(item, ro, rd, tl, tile_size=128, any_hit=True)
        )(ro_l, rd_l, sh_lim)
      occ = occ | (d_sh < sh_lim)
    vis = jnp.where(occ, float(spec.shadow_light_fraction), 1.0)

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
