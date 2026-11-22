"""Wave 2: Hierarchical 2-Level Morton Cluster-BVH Dynamic Primitive Renderer in Mosaic VMEM.

Supports arbitrary numbers of moving MJX primitives (50 -> 200 -> 1,000+) with:
- 100% on-device JAX 3D Morton sorting & 2-level bounding-sphere hierarchy (zero host callbacks)
- Constant O(1) Mosaic MLIR compile size via jax.lax.fori_loop + aligned dynamic_slice
- Coarse per-tile Macro-128 & Micro-16 jax.lax.cond culling on both primary and shadow rays
- Exact numerical parity against V2.2 pallas_v2 / v2_soa
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
import numpy as np

from tpu_mjx_render.pallas_v2 import (
    _compute_batch_camera_rays_soa,
    _compute_geom_bound_sq,
    _extract_light_batch,
    _shade_batch_soa,
    build_local_camera_rays,
)
from tpu_mjx_render.primitives import (
    GEOM_BOX,
    GEOM_CAPSULE,
    GEOM_CYLINDER,
    GEOM_ELLIPSOID,
    GEOM_PLANE,
    GEOM_SPHERE,
    MJ_MAXVAL,
)

from v3_pallas import _ray_one_geom_from_delta

CLUSTER_K = 16
MACRO_M = 8  # 8 micro-clusters * 16 geoms = 128 geoms per macro-cluster


def _expand_bits_10(v: jax.Array) -> jax.Array:
  v = (v | (v << 16)) & jnp.uint32(0x030000FF)
  v = (v | (v << 8)) & jnp.uint32(0x0300F00F)
  v = (v | (v << 4)) & jnp.uint32(0x030C30C3)
  v = (v | (v << 2)) & jnp.uint32(0x09249249)
  return v


def _morton_codes_3d(pos: jax.Array) -> jax.Array:
  """Compute 30-bit 3D Morton codes for (B, G, 3) positions in parallel."""
  mn = jnp.min(pos, axis=1, keepdims=True)
  mx = jnp.max(pos, axis=1, keepdims=True)
  span = jnp.maximum(mx - mn, 1e-4)
  norm = jnp.clip((pos - mn) / span, 0.0, 0.999999)
  q = jnp.floor(norm * 1024.0).astype(jnp.uint32)
  return (_expand_bits_10(q[..., 0]) << 2) | (_expand_bits_10(q[..., 1]) << 1) | _expand_bits_10(q[..., 2])


def _build_type_hierarchy_tensors(
    pos_t: jax.Array,      # (B, G_t, 3)
    mat_t: jax.Array,      # (B, G_t, 3, 3)
    size_t: jax.Array,     # (G_t, 3)
    rb_t: jax.Array,       # (G_t,) radius (not squared)
    gid_t: jax.Array,      # (G_t,) int32
    block_b: int,
    B_pad: int,
) -> Tuple[jax.Array, jax.Array, int, int, int]:
  """On-device Morton sort + 2-level bounding sphere hierarchy packed for Mosaic VMEM."""
  B, G_t, _ = pos_t.shape
  n_b_blk = B_pad // block_b

  codes = _morton_codes_3d(pos_t)
  order = jnp.argsort(codes, axis=-1)  # (B, G_t)

  pos_s = jnp.take_along_axis(pos_t, order[..., None], axis=1)
  mat_s = jnp.take_along_axis(mat_t.reshape(B, G_t, 9), order[..., None], axis=1)
  sz_b = jnp.broadcast_to(size_t[None, :, :], (B, G_t, 3))
  sz_s = jnp.take_along_axis(sz_b, order[..., None], axis=1)
  rb_b = jnp.broadcast_to(rb_t[None, :], (B, G_t))
  rb_s = jnp.take_along_axis(rb_b, order, axis=1)
  gid_b = jnp.broadcast_to(gid_t[None, :].astype(jnp.float32), (B, G_t))
  gid_s = jnp.take_along_axis(gid_b, order, axis=1)
  valid_s = jnp.ones((B, G_t), dtype=jnp.bool_)

  n_micro_raw = (G_t + CLUSTER_K - 1) // CLUSTER_K
  n_macro = (n_micro_raw + MACRO_M - 1) // MACRO_M
  n_micro = n_macro * MACRO_M
  G_pad = n_micro * CLUSTER_K
  pad_g = G_pad - G_t

  if pad_g > 0:
    pos_s = jnp.pad(pos_s, ((0, 0), (0, pad_g), (0, 0)))
    mat_s = jnp.pad(mat_s, ((0, 0), (0, pad_g), (0, 0)))
    sz_s = jnp.pad(sz_s, ((0, 0), (0, pad_g), (0, 0)))
    rb_s = jnp.pad(rb_s, ((0, 0), (0, pad_g)), constant_values=-1.0)
    gid_s = jnp.pad(gid_s, ((0, 0), (0, pad_g)), constant_values=-1.0)
    valid_s = jnp.pad(valid_s, ((0, 0), (0, pad_g)), constant_values=False)

  # Compute micro-cluster bounding spheres: (B, n_micro, K)
  pos_c = pos_s.reshape(B, n_micro, CLUSTER_K, 3)
  rb_c = rb_s.reshape(B, n_micro, CLUSTER_K)
  val_c = valid_s.reshape(B, n_micro, CLUSTER_K)
  any_val_c = jnp.any(val_c, axis=2)  # (B, n_micro)

  c_min = jnp.min(jnp.where(val_c[..., None], pos_c, 1.0e9), axis=2)
  c_max = jnp.max(jnp.where(val_c[..., None], pos_c, -1.0e9), axis=2)
  c_center = jnp.where(any_val_c[..., None], 0.5 * (c_min + c_max), 0.0)  # (B, n_micro, 3)
  c_dists = jnp.linalg.norm(pos_c - c_center[:, :, None, :], axis=-1) + rb_c
  c_rad = jnp.max(jnp.where(val_c, c_dists, 0.0), axis=2) + 1.0e-4
  c_rad_sq = jnp.where(any_val_c, c_rad * c_rad, -1.0)  # (B, n_micro)

  # Compute macro-cluster bounding spheres: (B, n_macro, MACRO_M)
  uc_m = c_center.reshape(B, n_macro, MACRO_M, 3)
  ur_m = jnp.where(any_val_c, c_rad, 0.0).reshape(B, n_macro, MACRO_M)
  uval_m = any_val_c.reshape(B, n_macro, MACRO_M)
  any_val_m = jnp.any(uval_m, axis=2)  # (B, n_macro)

  m_min = jnp.min(jnp.where(uval_m[..., None], uc_m - ur_m[..., None], 1.0e9), axis=2)
  m_max = jnp.max(jnp.where(uval_m[..., None], uc_m + ur_m[..., None], -1.0e9), axis=2)
  m_center = jnp.where(any_val_m[..., None], 0.5 * (m_min + m_max), 0.0)  # (B, n_macro, 3)
  m_dists = jnp.linalg.norm(uc_m - m_center[:, :, None, :], axis=-1) + ur_m
  m_rad = jnp.max(jnp.where(uval_m, m_dists, 0.0), axis=2) + 1.0e-4
  m_rad_sq = jnp.where(any_val_m, m_rad * m_rad, -1.0)  # (B, n_macro)

  # Pack micro-clusters into (n_b_blk * n_micro * 18 * block_b, 128):
  # For each micro-cluster c: 18 field rows (0..16 = 17 geom fields in cols 0..15;
  # row 17 cols 0..3 = c_cx, c_cy, c_cz, c_rad_sq)
  rb_sq_s = jnp.where(valid_s, rb_s * rb_s, -1.0)
  g17 = jnp.concatenate([pos_s, mat_s, sz_s, rb_sq_s[..., None], gid_s[..., None]], axis=-1)  # (B, n_micro*K, 17)
  g17_c = g17.reshape(B, n_micro, CLUSTER_K, 17).transpose(0, 1, 3, 2)  # (B, n_micro, 17, 16)
  g17_pad = jnp.pad(g17_c, ((0, 0), (0, 0), (0, 0), (0, 128 - CLUSTER_K)))  # (B, n_micro, 17, 128)

  c_meta = jnp.zeros((B, n_micro, 1, 128), dtype=jnp.float32)
  c_meta = c_meta.at[:, :, 0, 0:3].set(c_center)
  c_meta = c_meta.at[:, :, 0, 3].set(c_rad_sq)
  micro_18 = jnp.concatenate([g17_pad, c_meta], axis=2)  # (B, n_micro, 18, 128)
  if B_pad > B:
    micro_18 = jnp.pad(micro_18, ((0, B_pad - B), (0, 0), (0, 0), (0, 0)))
  # Transpose to (n_b_blk, n_micro, 18, block_b, 128) -> (n_b_blk * n_micro * 18 * block_b, 128)
  gf_packed = (
      micro_18.reshape(n_b_blk, block_b, n_micro, 18, 128)
      .transpose(0, 2, 3, 1, 4)
      .reshape(n_b_blk * n_micro * 18 * block_b, 128)
  )

  # Pack macro-clusters into (n_b_blk * n_macro * 4 * block_b, 128)
  m_4 = jnp.stack([m_center[..., 0], m_center[..., 1], m_center[..., 2], m_rad_sq], axis=-1)  # (B, n_macro, 4)
  m_4_pad = jnp.pad(m_4, ((0, B_pad - B), (0, 0), (0, 0)))  # (B_pad, n_macro, 4)
  m_128 = jnp.pad(m_4_pad[:, :, :, None], ((0, 0), (0, 0), (0, 0), (0, 127)))  # (B_pad, n_macro, 4, 128)
  mf_packed = (
      m_128.reshape(n_b_blk, block_b, n_macro, 4, 128)
      .transpose(0, 2, 3, 1, 4)
      .reshape(n_b_blk * n_macro * 4 * block_b, 128)
  )

  return gf_packed, mf_packed, n_micro, n_macro, n_micro_raw


def pallas_raycast_hierarchical(
    spec: Any,
    mjx_data_batch: Any,
    geom_rgba_batch: Optional[jax.Array] = None,
    interpret: bool = False,
    block_b: int = 8,
    block_r: int = 1024,
    cull_mode: str = "hier2",  # "flat_loop" | "cluster1" | "hier2"
):
  """Hierarchical 2-level Morton cluster-BVH Mosaic raycaster for 50..1000+ moving primitives."""
  assert cull_mode in ("flat_loop", "cluster1", "hier2")
  cam_pos = mjx_data_batch.cam_xpos[:, int(spec.cam_id), :]
  cam_mat = mjx_data_batch.cam_xmat[:, int(spec.cam_id), :, :]
  geom_xpos = mjx_data_batch.geom_xpos
  geom_xmat = mjx_data_batch.geom_xmat
  light_xpos, light_xdir = _extract_light_batch(spec, mjx_data_batch)
  geom_size = jnp.asarray(spec.geom_size, dtype=jnp.float32)
  geom_rgba = jnp.asarray(spec.geom_rgba, dtype=jnp.float32) if geom_rgba_batch is None else geom_rgba_batch

  B = cam_pos.shape[0]
  H, W = int(spec.height), int(spec.width)
  R = H * W

  gids = np.asarray(spec.enabled_geom_ids, dtype=np.int32)
  if len(gids) == 0:
    ray_orig, ray_dir, neg_z = _compute_batch_camera_rays_soa(spec, cam_pos, cam_mat)
    return _shade_batch_soa(
        spec, geom_xpos, geom_xmat, geom_size, geom_rgba, light_xpos, light_xdir,
        ray_orig, ray_dir, neg_z,
        jnp.full((B, R), -1.0, jnp.float32),
        jnp.zeros((B, R, 3), jnp.float32),
        jnp.full((B, R), -1, jnp.int32),
    )

  B_pad = ((B + block_b - 1) // block_b) * block_b
  R_pad = ((R + block_r - 1) // block_r) * block_r
  n_b_blk = B_pad // block_b
  n_r_blk = R_pad // block_r

  cam_pack = jnp.concatenate([cam_pos, cam_mat.reshape(B, 9)], axis=-1)
  cam_128 = jnp.pad(cam_pack, ((0, B_pad - B), (0, 128 - 12)))

  ld_np, lo_np, neg_z_np = build_local_camera_rays(spec)
  md_np = ((float(spec.zfar) - float(spec.znear)) / neg_z_np) if spec.zfar > 0.0 else np.full((R,), MJ_MAXVAL, dtype=np.float32)
  lray_raw = np.zeros((8, R_pad), dtype=np.float32)
  lray_raw[0:3, :R] = ld_np.T
  lray_raw[3:6, :R] = lo_np.T
  lray_raw[6, :R] = neg_z_np
  lray_raw[7, :R] = md_np
  lray_raw[2, R:] = -1.0
  lray_raw[6, R:] = 1.0
  lray_raw[7, R:] = 1.0
  lray_dev = jnp.asarray(lray_raw, dtype=jnp.float32)

  # Separate planes vs bounded types
  plane_gids = [int(g) for g in gids if int(spec.geom_type[g]) == GEOM_PLANE]
  bounded_types = [GEOM_SPHERE, GEOM_CAPSULE, GEOM_ELLIPSOID, GEOM_CYLINDER, GEOM_BOX]
  rb_all = np.sqrt(_compute_geom_bound_sq(spec, gids))
  gid_to_rb = {int(g): float(rb_all[i]) for i, g in enumerate(gids)}

  # Pack planes into (n_b_blk * 16 * block_b, P_pad)
  P = len(plane_gids)
  P_pad = max(128, ((P + 127) // 128) * 128)
  if P > 0:
    pg = np.asarray(plane_gids, dtype=np.int32)
    p_pos = geom_xpos[:, pg, :]
    p_mat = geom_xmat[:, pg, :, :].reshape(B, P, 9)
    p_sz = jnp.broadcast_to(geom_size[pg][None, :, :], (B, P, 3))
    p_rb = jnp.broadcast_to(jnp.asarray([gid_to_rb[g] ** 2 for g in plane_gids])[None, :, None], (B, P, 1))
    p_16 = jnp.pad(jnp.concatenate([p_pos, p_mat, p_sz, p_rb], axis=-1), ((0, B_pad - B), (0, P_pad - P), (0, 0)))
    plane_fields = (
        p_16.reshape(n_b_blk, block_b, P_pad, 16)
        .transpose(0, 3, 1, 2)
        .reshape(n_b_blk * 16 * block_b, P_pad)
    )
  else:
    plane_fields = jnp.zeros((n_b_blk * 16 * block_b, 128), dtype=jnp.float32)

  # Build dynamic Morton hierarchy per active bounded geom type
  active_types: List[int] = []
  gf_list: List[jax.Array] = []
  mf_list: List[jax.Array] = []
  n_micro_list: List[int] = []
  n_macro_list: List[int] = []
  n_micro_raw_list: List[int] = []

  for gt in bounded_types:
    tgids = [int(g) for g in gids if int(spec.geom_type[g]) == gt]
    if not tgids:
      continue
    tg_np = np.asarray(tgids, dtype=np.int32)
    pos_t = geom_xpos[:, tg_np, :]
    mat_t = geom_xmat[:, tg_np, :, :]
    sz_t = geom_size[tg_np]
    rb_t = jnp.asarray([gid_to_rb[g] for g in tgids], dtype=jnp.float32)
    gid_t = jnp.asarray(tg_np, dtype=jnp.int32)
    gf_t, mf_t, n_micro_t, n_macro_t, n_micro_raw_t = _build_type_hierarchy_tensors(
        pos_t, mat_t, sz_t, rb_t, gid_t, block_b, B_pad
    )
    active_types.append(gt)
    gf_list.append(gf_t)
    mf_list.append(mf_t)
    n_micro_list.append(n_micro_t)
    n_macro_list.append(n_macro_t)
    n_micro_raw_list.append(n_micro_raw_t)

  shadow_l_idx = -1
  for li in range(int(spec.nlight)):
    if spec.use_shadows and bool(spec.light_active[li]) and bool(spec.light_castshadow[li]):
      shadow_l_idx = li
      break

  light_128 = jnp.zeros((B_pad, 128), dtype=jnp.float32)
  if shadow_l_idx >= 0:
    is_dir = 1.0 if int(spec.light_type[shadow_l_idx]) == 1 else 0.0
    lp = light_xpos[:, shadow_l_idx, :]
    ld_vec = light_xdir[:, shadow_l_idx, :]
    ld_vec = ld_vec / jnp.maximum(jnp.linalg.norm(ld_vec, axis=-1, keepdims=True), 1e-12)
    l_row = jnp.concatenate([jnp.full((B, 1), is_dir, jnp.float32), lp, ld_vec], axis=-1)
    light_128 = jnp.pad(l_row, ((0, B_pad - B), (0, 128 - 7)))

  plane_fx = tuple(float(spec.geom_size[g, 0]) > 0.0 for g in plane_gids)
  plane_fy = tuple(float(spec.geom_size[g, 1]) > 0.0 for g in plane_gids)
  plane_gids_t = tuple(plane_gids)
  znear_f = float(spec.znear)
  do_in_vmem_shadow = shadow_l_idx >= 0
  n_active_types = len(active_types)

  def _hier_kernel(cam_ref, lray_ref, pf_ref, light_ref, *rest):
    # rest contains gf_0..gf_{T-1}, mf_0..mf_{T-1}, followed by the 6 output refs
    gf_refs = rest[:n_active_types]
    mf_refs = rest[n_active_types : 2 * n_active_types]
    out_d_ref, out_nx_ref, out_ny_ref, out_nz_ref, out_gid_ref, out_sh_ref = rest[2 * n_active_types :]

    cam = cam_ref[...]
    lr = lray_ref[...]
    pf = pf_ref[...]
    linfo = light_ref[...]

    ldx, ldy, ldz = lr[0:1, :], lr[1:2, :], lr[2:3, :]
    lox, loy, loz = lr[3:4, :], lr[4:5, :], lr[5:6, :]
    inv_cos = znear_f / lr[6:7, :]
    max_d = jnp.broadcast_to(lr[7:8, :], (block_b, block_r))

    cpx, cpy, cpz = cam[:, 0:1], cam[:, 1:2], cam[:, 2:3]
    c00, c01, c02 = cam[:, 3:4], cam[:, 4:5], cam[:, 5:6]
    c10, c11, c12 = cam[:, 6:7], cam[:, 7:8], cam[:, 8:9]
    c20, c21, c22 = cam[:, 9:10], cam[:, 10:11], cam[:, 11:12]

    rdx_raw = c00 * ldx + c01 * ldy + c02 * ldz
    rdy_raw = c10 * ldx + c11 * ldy + c12 * ldz
    rdz_raw = c20 * ldx + c21 * ldy + c22 * ldz
    inv_rn = 1.0 / jnp.maximum(jnp.sqrt(rdx_raw * rdx_raw + rdy_raw * rdy_raw + rdz_raw * rdz_raw), 1e-12)
    rdx, rdy, rdz = rdx_raw * inv_rn, rdy_raw * inv_rn, rdz_raw * inv_rn

    rox = cpx + (c00 * lox + c01 * loy + c02 * loz) + inv_cos * rdx
    roy = cpy + (c10 * lox + c11 * loy + c12 * loz) + inv_cos * rdy
    roz = cpz + (c20 * lox + c21 * loy + c22 * loz) + inv_cos * rdz

    best_d = max_d
    best_nx = jnp.zeros((block_b, block_r), dtype=jnp.float32)
    best_ny = jnp.zeros((block_b, block_r), dtype=jnp.float32)
    best_nz = jnp.zeros((block_b, block_r), dtype=jnp.float32)
    best_gid = jnp.full((block_b, block_r), -1, dtype=jnp.int32)

    # 1. Plane geoms
    for pk_i, (pgid, pfx, pfy) in enumerate(zip(plane_gids_t, plane_fx, plane_fy)):
      px = pf[0 * block_b : 1 * block_b, pk_i : pk_i + 1]
      py = pf[1 * block_b : 2 * block_b, pk_i : pk_i + 1]
      pz = pf[2 * block_b : 3 * block_b, pk_i : pk_i + 1]
      m00, m01, m02 = pf[3 * block_b : 4 * block_b, pk_i : pk_i + 1], pf[4 * block_b : 5 * block_b, pk_i : pk_i + 1], pf[5 * block_b : 6 * block_b, pk_i : pk_i + 1]
      m10, m11, m12 = pf[6 * block_b : 7 * block_b, pk_i : pk_i + 1], pf[7 * block_b : 8 * block_b, pk_i : pk_i + 1], pf[8 * block_b : 9 * block_b, pk_i : pk_i + 1]
      m20, m21, m22 = pf[9 * block_b : 10 * block_b, pk_i : pk_i + 1], pf[10 * block_b : 11 * block_b, pk_i : pk_i + 1], pf[11 * block_b : 12 * block_b, pk_i : pk_i + 1]
      sx, sy, sz = pf[12 * block_b : 13 * block_b, pk_i : pk_i + 1], pf[13 * block_b : 14 * block_b, pk_i : pk_i + 1], pf[14 * block_b : 15 * block_b, pk_i : pk_i + 1]
      cd, cnx, cny, cnz = _ray_one_geom_from_delta(
          GEOM_PLANE, rox - px, roy - py, roz - pz, True,
          m00, m01, m02, m10, m11, m12, m20, m21, m22,
          sx, sy, sz, rdx, rdy, rdz,
          plane_finite_x=pfx, plane_finite_y=pfy, compute_normal=True,
      )
      imp = cd < best_d
      best_d = jnp.where(imp, cd, best_d)
      best_gid = jnp.where(imp, jnp.int32(pgid), best_gid)
      best_nx = jnp.where(imp, cnx, best_nx)
      best_ny = jnp.where(imp, cny, best_ny)
      best_nz = jnp.where(imp, cnz, best_nz)

    def _sphere_tile_hit(ox, oy, oz, dx_r, dy_r, dz_r, cx, cy, cz, r2, dist_cap=None):
      ddx = ox - cx
      ddy = oy - cy
      ddz = oz - cz
      proj = ddx * dx_r + ddy * dy_r + ddz * dz_r
      d2 = ddx * ddx + ddy * ddy + ddz * ddz
      d2_perp = d2 - proj * proj
      hit_s = (d2_perp <= r2) & ((proj < 0.0) | (d2 <= r2))
      if dist_cap is not None:
        hit_s = hit_s & (dist_cap > 0.0)
      return hit_s

    def _eval_micro_primary(gt, c_slice, state):
      bd, bgid, bnx, bny, bnz = state
      for k in range(CLUSTER_K):
        px = c_slice[0 * block_b : 1 * block_b, k : k + 1]
        py = c_slice[1 * block_b : 2 * block_b, k : k + 1]
        pz = c_slice[2 * block_b : 3 * block_b, k : k + 1]
        m00, m01, m02 = c_slice[3 * block_b : 4 * block_b, k : k + 1], c_slice[4 * block_b : 5 * block_b, k : k + 1], c_slice[5 * block_b : 6 * block_b, k : k + 1]
        m10, m11, m12 = c_slice[6 * block_b : 7 * block_b, k : k + 1], c_slice[7 * block_b : 8 * block_b, k : k + 1], c_slice[8 * block_b : 9 * block_b, k : k + 1]
        m20, m21, m22 = c_slice[9 * block_b : 10 * block_b, k : k + 1], c_slice[10 * block_b : 11 * block_b, k : k + 1], c_slice[11 * block_b : 12 * block_b, k : k + 1]
        sx, sy, sz = c_slice[12 * block_b : 13 * block_b, k : k + 1], c_slice[13 * block_b : 14 * block_b, k : k + 1], c_slice[14 * block_b : 15 * block_b, k : k + 1]
        rb_sq = c_slice[15 * block_b : 16 * block_b, k : k + 1]
        gid_f = c_slice[16 * block_b : 17 * block_b, k : k + 1]

        dx = rox - px
        dy = roy - py
        dz = roz - pz
        proj = dx * rdx + dy * rdy + dz * rdz
        d2 = dx * dx + dy * dy + dz * dz
        sh = (d2 - proj * proj <= rb_sq) & ((proj < 0.0) | (d2 <= rb_sq))
        cd, cnx, cny, cnz = _ray_one_geom_from_delta(
            gt, dx, dy, dz, sh,
            m00, m01, m02, m10, m11, m12, m20, m21, m22,
            sx, sy, sz, rdx, rdy, rdz, compute_normal=True,
        )
        imp = cd < bd
        bd = jnp.where(imp, cd, bd)
        bgid = jnp.where(imp, jnp.broadcast_to(gid_f.astype(jnp.int32), bgid.shape), bgid)
        bnx = jnp.where(imp, cnx, bnx)
        bny = jnp.where(imp, cny, bny)
        bnz = jnp.where(imp, cnz, bnz)
      return (bd, bgid, bnx, bny, bnz)

    # 2. Bounded dynamic geoms per type
    state_p = (best_d, best_gid, best_nx, best_ny, best_nz)
    for t_i, gt in enumerate(active_types):
      gf_ref_t = gf_refs[t_i]
      mf_ref_t = mf_refs[t_i]
      n_micro_raw_t = n_micro_raw_list[t_i]
      n_macro_t = n_macro_list[t_i]

      def _run_micro_by_idx(c_idx, st, _gt=gt, _gf_ref=gf_ref_t):
        base_r = c_idx * 18 * block_b
        if cull_mode == "flat_loop":
          c_slice = _gf_ref[pl.ds(base_r, 17 * block_b), :]
          return _eval_micro_primary(_gt, c_slice, st)
        c_meta = _gf_ref[pl.ds(base_r + 17 * block_b, block_b), :]
        ccx = c_meta[:, 0:1]
        ccy = c_meta[:, 1:2]
        ccz = c_meta[:, 2:3]
        cr2 = c_meta[:, 3:4]
        u_hit = jnp.any(_sphere_tile_hit(rox, roy, roz, rdx, rdy, rdz, ccx, ccy, ccz, cr2))
        return jax.lax.cond(
            u_hit,
            lambda s: _eval_micro_primary(_gt, _gf_ref[pl.ds(base_r, 17 * block_b), :], s),
            lambda s: s,
            st,
        )

      if cull_mode in ("flat_loop", "cluster1"):
        state_p = jax.lax.fori_loop(0, n_micro_raw_t, _run_micro_by_idx, state_p)
      else:  # hier2
        def _macro_step(m_idx, st, _mf_ref=mf_ref_t, _rm=_run_micro_by_idx, _nm=n_micro_raw_t):
          m_slice = _mf_ref[pl.ds(m_idx * 4 * block_b, 4 * block_b), :]
          mcx = m_slice[0 * block_b : 1 * block_b, 0:1]
          mcy = m_slice[1 * block_b : 2 * block_b, 0:1]
          mcz = m_slice[2 * block_b : 3 * block_b, 0:1]
          mr2 = m_slice[3 * block_b : 4 * block_b, 0:1]
          m_hit = jnp.any(_sphere_tile_hit(rox, roy, roz, rdx, rdy, rdz, mcx, mcy, mcz, mr2))

          def _do_macro(s):
            base_c = m_idx * MACRO_M
            end_c = jnp.minimum(base_c + MACRO_M, _nm)
            return jax.lax.fori_loop(base_c, end_c, _rm, s)

          return jax.lax.cond(m_hit, _do_macro, lambda s: s, st)

        state_p = jax.lax.fori_loop(0, n_macro_t, _macro_step, state_p)

    best_d, best_gid, best_nx, best_ny, best_nz = state_p
    hit = best_gid >= 0
    best_d_out = jnp.where(hit, best_d, -1.0)
    out_nx_v = jnp.where(hit, best_nx, 0.0)
    out_ny_v = jnp.where(hit, best_ny, 0.0)
    out_nz_v = jnp.where(hit, best_nz, 0.0)

    # 3. Fused shadow any-hit with hierarchical culling
    sh_occ = jnp.zeros((block_b, block_r), dtype=jnp.float32)
    if do_in_vmem_shadow:
      d_safe = jnp.where(hit, best_d, 0.0)
      hx = rox + d_safe * rdx + out_nx_v * 1.0e-4
      hy = roy + d_safe * rdy + out_ny_v * 1.0e-4
      hz = roz + d_safe * rdz + out_nz_v * 1.0e-4

      is_dir = linfo[:, 0:1] > 0.5
      dlx = linfo[:, 1:2] - hx
      dly = linfo[:, 2:3] - hy
      dlz = linfo[:, 3:4] - hz
      d_pt = jnp.sqrt(jnp.maximum(dlx * dlx + dly * dly + dlz * dlz, 1e-12))
      lx_v = jnp.where(is_dir, -linfo[:, 4:5], dlx / d_pt)
      ly_v = jnp.where(is_dir, -linfo[:, 5:6], dly / d_pt)
      lz_v = jnp.where(is_dir, -linfo[:, 6:7], dlz / d_pt)
      ndotl = out_nx_v * lx_v + out_ny_v * ly_v + out_nz_v * lz_v
      dist_lim = jnp.where(is_dir, 1.0e8, jnp.maximum(d_pt - 1.0e-3, 0.0))
      sh_max = jnp.where(hit & (ndotl > 0.0), dist_lim, 0.0)

      occ_i32 = jnp.zeros((block_b, block_r), dtype=jnp.int32)
      # Plane shadows
      for pk_i, (pfx, pfy) in enumerate(zip(plane_fx, plane_fy)):
        px = pf[0 * block_b : 1 * block_b, pk_i : pk_i + 1]
        py = pf[1 * block_b : 2 * block_b, pk_i : pk_i + 1]
        pz = pf[2 * block_b : 3 * block_b, pk_i : pk_i + 1]
        m00, m01, m02 = pf[3 * block_b : 4 * block_b, pk_i : pk_i + 1], pf[4 * block_b : 5 * block_b, pk_i : pk_i + 1], pf[5 * block_b : 6 * block_b, pk_i : pk_i + 1]
        m10, m11, m12 = pf[6 * block_b : 7 * block_b, pk_i : pk_i + 1], pf[7 * block_b : 8 * block_b, pk_i : pk_i + 1], pf[8 * block_b : 9 * block_b, pk_i : pk_i + 1]
        m20, m21, m22 = pf[9 * block_b : 10 * block_b, pk_i : pk_i + 1], pf[10 * block_b : 11 * block_b, pk_i : pk_i + 1], pf[11 * block_b : 12 * block_b, pk_i : pk_i + 1]
        sx, sy, sz = pf[12 * block_b : 13 * block_b, pk_i : pk_i + 1], pf[13 * block_b : 14 * block_b, pk_i : pk_i + 1], pf[14 * block_b : 15 * block_b, pk_i : pk_i + 1]
        scd, _, _, _ = _ray_one_geom_from_delta(
            GEOM_PLANE, hx - px, hy - py, hz - pz, True,
            m00, m01, m02, m10, m11, m12, m20, m21, m22,
            sx, sy, sz, lx_v, ly_v, lz_v,
            plane_finite_x=pfx, plane_finite_y=pfy, compute_normal=False,
        )
        occ_i32 = occ_i32 | (scd < sh_max).astype(jnp.int32)

      def _eval_micro_shadow(gt, c_slice, occ_in):
        occ_cur = occ_in
        for k in range(CLUSTER_K):
          px = c_slice[0 * block_b : 1 * block_b, k : k + 1]
          py = c_slice[1 * block_b : 2 * block_b, k : k + 1]
          pz = c_slice[2 * block_b : 3 * block_b, k : k + 1]
          m00, m01, m02 = c_slice[3 * block_b : 4 * block_b, k : k + 1], c_slice[4 * block_b : 5 * block_b, k : k + 1], c_slice[5 * block_b : 6 * block_b, k : k + 1]
          m10, m11, m12 = c_slice[6 * block_b : 7 * block_b, k : k + 1], c_slice[7 * block_b : 8 * block_b, k : k + 1], c_slice[8 * block_b : 9 * block_b, k : k + 1]
          m20, m21, m22 = c_slice[9 * block_b : 10 * block_b, k : k + 1], c_slice[10 * block_b : 11 * block_b, k : k + 1], c_slice[11 * block_b : 12 * block_b, k : k + 1]
          sx, sy, sz = c_slice[12 * block_b : 13 * block_b, k : k + 1], c_slice[13 * block_b : 14 * block_b, k : k + 1], c_slice[14 * block_b : 15 * block_b, k : k + 1]
          rb_sq = c_slice[15 * block_b : 16 * block_b, k : k + 1]

          dx_s = hx - px
          dy_s = hy - py
          dz_s = hz - pz
          proj_s = dx_s * lx_v + dy_s * ly_v + dz_s * lz_v
          d2_s = dx_s * dx_s + dy_s * dy_s + dz_s * dz_s
          sh_s = (d2_s - proj_s * proj_s <= rb_sq) & ((proj_s < 0.0) | (d2_s <= rb_sq)) & (sh_max > 0.0)
          scd, _, _, _ = _ray_one_geom_from_delta(
              gt, dx_s, dy_s, dz_s, sh_s,
              m00, m01, m02, m10, m11, m12, m20, m21, m22,
              sx, sy, sz, lx_v, ly_v, lz_v, compute_normal=False,
          )
          occ_cur = occ_cur | (scd < sh_max).astype(jnp.int32)
        return occ_cur

      for t_i, gt in enumerate(active_types):
        gf_ref_t = gf_refs[t_i]
        mf_ref_t = mf_refs[t_i]
        n_micro_raw_t = n_micro_raw_list[t_i]
        n_macro_t = n_macro_list[t_i]

        def _run_sh_micro(c_idx, occ_in, _gt=gt, _gf_ref=gf_ref_t):
          base_r = c_idx * 18 * block_b
          if cull_mode == "flat_loop":
            c_slice = _gf_ref[pl.ds(base_r, 17 * block_b), :]
            return _eval_micro_shadow(_gt, c_slice, occ_in)
          c_meta = _gf_ref[pl.ds(base_r + 17 * block_b, block_b), :]
          ccx = c_meta[:, 0:1]
          ccy = c_meta[:, 1:2]
          ccz = c_meta[:, 2:3]
          cr2 = c_meta[:, 3:4]
          u_hit = jnp.any(_sphere_tile_hit(hx, hy, hz, lx_v, ly_v, lz_v, ccx, ccy, ccz, cr2, dist_cap=sh_max))
          return jax.lax.cond(
              u_hit,
              lambda o: _eval_micro_shadow(_gt, _gf_ref[pl.ds(base_r, 17 * block_b), :], o),
              lambda o: o,
              occ_in,
          )

        if cull_mode in ("flat_loop", "cluster1"):
          occ_i32 = jax.lax.fori_loop(0, n_micro_raw_t, _run_sh_micro, occ_i32)
        else:
          def _sh_macro(m_idx, occ_in, _mf_ref=mf_ref_t, _rm=_run_sh_micro, _nm=n_micro_raw_t):
            m_slice = _mf_ref[pl.ds(m_idx * 4 * block_b, 4 * block_b), :]
            mcx = m_slice[0 * block_b : 1 * block_b, 0:1]
            mcy = m_slice[1 * block_b : 2 * block_b, 0:1]
            mcz = m_slice[2 * block_b : 3 * block_b, 0:1]
            mr2 = m_slice[3 * block_b : 4 * block_b, 0:1]
            m_hit = jnp.any(_sphere_tile_hit(hx, hy, hz, lx_v, ly_v, lz_v, mcx, mcy, mcz, mr2, dist_cap=sh_max))
            return jax.lax.cond(
                m_hit,
                lambda o: jax.lax.fori_loop(m_idx * MACRO_M, jnp.minimum(m_idx * MACRO_M + MACRO_M, _nm), _rm, o),
                lambda o: o,
                occ_in,
            )

          occ_i32 = jax.lax.fori_loop(0, n_macro_t, _sh_macro, occ_i32)

      sh_occ = occ_i32.astype(jnp.float32)

    out_d_ref[...] = best_d_out
    out_nx_ref[...] = out_nx_v
    out_ny_ref[...] = out_ny_v
    out_nz_ref[...] = out_nz_v
    out_gid_ref[...] = best_gid
    out_sh_ref[...] = sh_occ

  in_arrays = [cam_128, lray_dev, plane_fields, light_128] + gf_list + mf_list
  in_specs = [
      pl.BlockSpec((block_b, 128), lambda bi, ri: (bi, 0)),
      pl.BlockSpec((8, block_r), lambda bi, ri: (0, ri)),
      pl.BlockSpec((16 * block_b, P_pad), lambda bi, ri: (bi, 0)),
      pl.BlockSpec((block_b, 128), lambda bi, ri: (bi, 0)),
  ]
  for n_micro_t in n_micro_list:
    in_specs.append(pl.BlockSpec((n_micro_t * 18 * block_b, 128), lambda bi, ri: (bi, 0)))
  for n_macro_t in n_macro_list:
    in_specs.append(pl.BlockSpec((n_macro_t * 4 * block_b, 128), lambda bi, ri: (bi, 0)))

  outs = pl.pallas_call(
      _hier_kernel,
      out_shape=[
          jax.ShapeDtypeStruct((B_pad, R_pad), jnp.float32),
          jax.ShapeDtypeStruct((B_pad, R_pad), jnp.float32),
          jax.ShapeDtypeStruct((B_pad, R_pad), jnp.float32),
          jax.ShapeDtypeStruct((B_pad, R_pad), jnp.float32),
          jax.ShapeDtypeStruct((B_pad, R_pad), jnp.int32),
          jax.ShapeDtypeStruct((B_pad, R_pad), jnp.float32),
      ],
      grid=(n_b_blk, n_r_blk),
      in_specs=in_specs,
      out_specs=[pl.BlockSpec((block_b, block_r), lambda bi, ri: (bi, ri)) for _ in range(6)],
      compiler_params=None if interpret else pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
      interpret=interpret,
  )(*in_arrays)

  d_out, nx_out, ny_out, nz_out, gid_out, sh_out = outs
  dist = d_out[:B, :R]
  normal = jnp.stack([nx_out[:B, :R], ny_out[:B, :R], nz_out[:B, :R]], axis=-1)
  geom_id = gid_out[:B, :R]
  fused_occ = (sh_out[:B, :R] > 0.5) if do_in_vmem_shadow else None

  ray_orig, ray_dir, neg_z = _compute_batch_camera_rays_soa(spec, cam_pos, cam_mat)
  return _shade_batch_soa(
      spec,
      geom_xpos,
      geom_xmat,
      geom_size,
      geom_rgba,
      light_xpos,
      light_xdir,
      ray_orig,
      ray_dir,
      neg_z,
      dist,
      normal,
      geom_id,
      fused_shadow0_occluded=fused_occ,
  )
