"""Wave 4B — Physical FX #1: Real-Time Deformable Cloth + Live Bottom-Up
Wide-BVH-64 Refit (Leaf -> L0 -> L1 -> L2) inside XLA (zero CPU rebuild).

Every op — PBD cloth physics step, smooth vertex normal update, 16-row BF16
leaf packing, bottom-up L0 -> L1 -> L2 AABB refit, and Pallas Wide-BVH-64
raytracing — runs inside one fused XLA compilation with:
  host_callbacks == 0, send_recv_ops == 0, gather_ops == 0.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import re
import time
from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw

from tpu_mjx_render.roofline import inspect_hlo_ops
from wave3_500m_10b_8k import (
    CompactWideBVHMeshItem,
    CompactWideBVHSceneSpec,
    build_compact_bvh_mesh_item,
    render_compact_bvh_batch,
)
from wide_bvh_v3 import BRANCH_FACTOR, LEAF_SIZE


def _pack_bf16_pair_jax(hi_f32: jax.Array, lo_f32: jax.Array) -> jax.Array:
  """Pack two unit-range FP32 arrays as (hi_bf16, lo_bf16) inside one FP32 array."""
  hi_c = jnp.clip(hi_f32, -1.0, 1.0).astype(jnp.bfloat16)
  lo_c = jnp.clip(lo_f32, -1.0, 1.0).astype(jnp.bfloat16)
  hi_u = jax.lax.bitcast_convert_type(hi_c, jnp.uint16).astype(jnp.uint32)
  lo_u = jax.lax.bitcast_convert_type(lo_c, jnp.uint16).astype(jnp.uint32)
  return jax.lax.bitcast_convert_type((hi_u << jnp.uint32(16)) | lo_u, jnp.float32)


def make_cloth_initial_state(
    nx: int = 128,
    ny: int = 128,
    width_m: float = 1.6,
    height_m: float = 1.6,
    z0: float = 1.25,
    center_xy: Tuple[float, float] = (0.04, -0.02),
) -> Tuple[jax.Array, jax.Array, float, float]:
  """Create (Ny+1, Nx+1, 3) rest cloth grid above collider."""
  assert nx % 8 == 0 and ny % 8 == 0, "nx and ny must be multiples of 8 (8x8 quad patch = 128 tris/leaf)"
  xs = np.linspace(-0.5 * width_m, 0.5 * width_m, nx + 1, dtype=np.float32) + center_xy[0]
  ys = np.linspace(-0.5 * height_m, 0.5 * height_m, ny + 1, dtype=np.float32) + center_xy[1]
  gx, gy = np.meshgrid(xs, ys, indexing="xy")
  # Subtle initial ripple so folds break symmetry naturally
  gz = np.full_like(gx, z0) + 0.015 * np.sin(5.0 * gx) * np.cos(4.0 * gy)
  pos = np.stack([gx, gy, gz], axis=-1).astype(np.float32)
  dx = float(width_m / nx)
  dy = float(height_m / ny)
  return jnp.asarray(pos), jnp.asarray(pos), dx, dy


def _jacobi_spring_pass(pos: jax.Array, d_row: int, d_col: int, rest_len: float, stiffness: float) -> jax.Array:
  """Gather-free Jacobi distance constraint over shifted grid slices."""
  p_a = pos[:-d_row if d_row > 0 else None, :-d_col if d_col > 0 else None, :]
  p_b = pos[d_row:, d_col:, :]
  delta = p_b - p_a
  dist = jnp.sqrt(jnp.sum(delta * delta, axis=-1, keepdims=True) + 1e-12)
  corr = 0.5 * stiffness * (dist - rest_len) * (delta / dist)
  # Accumulate corrections into zero buffer using slice updates (pure XLA pad/slice, zero gather!)
  ny1, nx1, _ = pos.shape
  pad_a = ((0, d_row), (0, d_col), (0, 0))
  pad_b = ((d_row, 0), (d_col, 0), (0, 0))
  da = jnp.pad(corr, pad_a)
  db = jnp.pad(-corr, pad_b)
  assert da.shape == (ny1, nx1, 3) and db.shape == (ny1, nx1, 3)
  return pos + da + db


def pbd_cloth_step(
    pos: jax.Array,
    prev_pos: jax.Array,
    dx: float,
    dy: float,
    sphere_center: Tuple[float, float, float] = (0.0, 0.0, 0.45),
    sphere_radius: float = 0.36,
    table_z: float = 0.08,
    dt: float = 1.0 / 120.0,
    substeps: int = 4,
    iters: int = 4,
) -> Tuple[jax.Array, jax.Array]:
  """Gather-free PBD cloth step with structural, shear, bend springs + sphere/table collision."""
  sub_dt = dt / float(substeps)
  d_diag = math.sqrt(dx * dx + dy * dy)
  sc = jnp.asarray(sphere_center, dtype=jnp.float32)[None, None, :]
  r_col = float(sphere_radius + 0.015)

  def _substep(i, carry):
    p, pp = carry
    vel = (p - pp) * 0.992
    # Gentle aerodynamic wind + gravity
    acc = jnp.array([0.12, -0.05, -9.81], dtype=jnp.float32)[None, None, :]
    p_pred = p + vel + (sub_dt * sub_dt) * acc

    def _relax(_, cur):
      # Structural
      cur = _jacobi_spring_pass(cur, 0, 1, dx, 0.48)
      cur = _jacobi_spring_pass(cur, 1, 0, dy, 0.48)
      # Shear
      cur = _jacobi_spring_pass(cur, 1, 1, d_diag, 0.35)
      # Opposite shear (flip along x via slice/pad)
      pa = cur[:-1, 1:, :]
      pb = cur[1:, :-1, :]
      delta = pb - pa
      dist = jnp.sqrt(jnp.sum(delta * delta, axis=-1, keepdims=True) + 1e-12)
      corr = 0.5 * 0.35 * (dist - d_diag) * (delta / dist)
      cur = cur + jnp.pad(corr, ((0, 1), (1, 0), (0, 0))) + jnp.pad(-corr, ((1, 0), (0, 1), (0, 0)))
      # Bending
      cur = _jacobi_spring_pass(cur, 0, 2, 2.0 * dx, 0.20)
      cur = _jacobi_spring_pass(cur, 2, 0, 2.0 * dy, 0.20)

      # Sphere collision
      d_vec = cur - sc
      d_len = jnp.sqrt(jnp.sum(d_vec * d_vec, axis=-1, keepdims=True) + 1e-12)
      hit_s = d_len < r_col
      p_proj_s = sc + d_vec * (r_col / d_len)
      cur = jnp.where(hit_s, p_proj_s, cur)

      # Table plane collision (tabletop slab [-1.1, 1.1] x [-1.1, 1.1])
      in_tbl = (jnp.abs(cur[..., 0:1]) < 1.15) & (jnp.abs(cur[..., 1:2]) < 1.15)
      hit_t = in_tbl & (cur[..., 2:3] < table_z + 0.012)
      cur = jnp.where(hit_t, jnp.concatenate([cur[..., 0:2], jnp.full_like(cur[..., 2:3], table_z + 0.012)], axis=-1), cur)
      return cur

    p_new = jax.lax.fori_loop(0, iters, _relax, p_pred)
    # Friction damping on contacts
    d_vec = p_new - sc
    d_len = jnp.sqrt(jnp.sum(d_vec * d_vec, axis=-1, keepdims=True) + 1e-12)
    on_surf = (d_len <= r_col + 0.004) | ((jnp.abs(p_new[..., 0:1]) < 1.15) & (jnp.abs(p_new[..., 1:2]) < 1.15) & (p_new[..., 2:3] <= table_z + 0.016))
    pp_new = jnp.where(on_surf, p + (p_new - p) * 0.88, p)
    return (p_new, pp_new)

  return jax.lax.fori_loop(0, substeps, _substep, (pos, prev_pos))


def compute_grid_normals(pos: jax.Array) -> jax.Array:
  """Gather-free exact smooth vertex normals on (Ny+1, Nx+1, 3) structured grid."""
  v00 = pos[:-1, :-1, :]
  v10 = pos[:-1, 1:, :]
  v01 = pos[1:, :-1, :]
  v11 = pos[1:, 1:, :]
  fn_a = jnp.cross(v10 - v00, v11 - v00)  # tri (v00, v10, v11)
  fn_b = jnp.cross(v11 - v00, v01 - v00)  # tri (v00, v11, v01)
  vn = (
      jnp.pad(fn_a + fn_b, ((0, 1), (0, 1), (0, 0)))  # v00
      + jnp.pad(fn_a, ((0, 1), (1, 0), (0, 0)))       # v10
      + jnp.pad(fn_b, ((1, 0), (0, 1), (0, 0)))       # v01
      + jnp.pad(fn_a + fn_b, ((1, 0), (1, 0), (0, 0)))  # v11
  )
  norms = jnp.sqrt(jnp.sum(vn * vn, axis=-1, keepdims=True) + 1e-12)
  return vn / norms


def refit_cloth_wide_bvh(pos: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Live bottom-up Leaf -> L0 -> L1 -> L2 Wide-BVH-64 refit inside XLA (zero gather!).

  Tiles (Ny, Nx) quads into 8x8 quad patches = 128 triangles per leaf.
  """
  ny = pos.shape[0] - 1
  nx = pos.shape[1] - 1
  py = ny // 8
  px = nx // 8
  n_leaves = py * px
  vnorm = compute_grid_normals(pos)

  def _quad_corners(arr):
    # (Ny, Nx, 3) -> (py, 8, px, 8, 3) -> (n_leaves, 64, 3)
    c00 = arr[:-1, :-1, :].reshape(py, 8, px, 8, 3).transpose(0, 2, 1, 3, 4).reshape(n_leaves, 64, 3)
    c10 = arr[:-1, 1:, :].reshape(py, 8, px, 8, 3).transpose(0, 2, 1, 3, 4).reshape(n_leaves, 64, 3)
    c01 = arr[1:, :-1, :].reshape(py, 8, px, 8, 3).transpose(0, 2, 1, 3, 4).reshape(n_leaves, 64, 3)
    c11 = arr[1:, 1:, :].reshape(py, 8, px, 8, 3).transpose(0, 2, 1, 3, 4).reshape(n_leaves, 64, 3)
    return c00, c10, c01, c11

  v00, v10, v01, v11 = _quad_corners(pos)
  n00, n10, n01, n11 = _quad_corners(vnorm)

  # 128 tris/leaf: first 64 tris (v00, v10, v11), next 64 tris (v00, v11, v01)
  tv0 = jnp.concatenate([v00, v00], axis=1)        # (n_leaves, 128, 3)
  tv1 = jnp.concatenate([v10, v11], axis=1)
  tv2 = jnp.concatenate([v11, v01], axis=1)
  te1 = tv1 - tv0
  te2 = tv2 - tv0

  tn0 = jnp.concatenate([n00, n00], axis=1)
  tn1 = jnp.concatenate([n10, n11], axis=1)
  tn2 = jnp.concatenate([n11, n01], axis=1)

  # Assemble (n_leaves, 16, 128) compact leaf_bank
  z128 = jnp.zeros((n_leaves, 1, LEAF_SIZE), dtype=jnp.float32)
  o128 = jnp.ones((n_leaves, 1, LEAF_SIZE), dtype=jnp.float32)
  r10 = _pack_bf16_pair_jax(tn0[:, :, 0], tn0[:, :, 1])[:, None, :]
  r11 = _pack_bf16_pair_jax(tn0[:, :, 2], tn1[:, :, 0])[:, None, :]
  r12 = _pack_bf16_pair_jax(tn1[:, :, 1], tn1[:, :, 2])[:, None, :]
  r13 = _pack_bf16_pair_jax(tn2[:, :, 0], tn2[:, :, 1])[:, None, :]
  r14 = _pack_bf16_pair_jax(tn2[:, :, 2], jnp.zeros((n_leaves, LEAF_SIZE), dtype=jnp.float32))[:, None, :]

  leaf_bank = jnp.concatenate([
      tv0.transpose(0, 2, 1),
      te1.transpose(0, 2, 1),
      te2.transpose(0, 2, 1),
      o128,
      r10, r11, r12, r13, r14,
      z128,
  ], axis=1)

  # Tight leaf AABBs (n_leaves, 3)
  pts_min = jnp.minimum(tv0, jnp.minimum(tv1, tv2))
  pts_max = jnp.maximum(tv0, jnp.maximum(tv1, tv2))
  leaf_min = jnp.min(pts_min, axis=1) - 1e-4
  leaf_max = jnp.max(pts_max, axis=1) + 1e-4

  # Bottom-up Level 1 (BVH-64)
  n_l1 = max(1, (n_leaves + BRANCH_FACTOR - 1) // BRANCH_FACTOR)
  n_leaves_p = n_l1 * BRANCH_FACTOR
  pad_l = n_leaves_p - n_leaves
  if pad_l > 0:
    lmin_p = jnp.concatenate([leaf_min, jnp.full((pad_l, 3), 1e20, dtype=jnp.float32)], axis=0)
    lmax_p = jnp.concatenate([leaf_max, jnp.full((pad_l, 3), -1e20, dtype=jnp.float32)], axis=0)
    lval_p = jnp.concatenate([jnp.ones((n_leaves,), dtype=jnp.float32), jnp.zeros((pad_l,), dtype=jnp.float32)], axis=0)
  else:
    lmin_p, lmax_p = leaf_min, leaf_max
    lval_p = jnp.ones((n_leaves,), dtype=jnp.float32)

  lmin_g = lmin_p.reshape(n_l1, BRANCH_FACTOR, 3).transpose(0, 2, 1)
  lmax_g = lmax_p.reshape(n_l1, BRANCH_FACTOR, 3).transpose(0, 2, 1)
  lval_g = lval_p.reshape(n_l1, 1, BRANCH_FACTOR)

  l0_pad = jnp.concatenate([
      jnp.full((n_l1, 3, 64), 1e20, dtype=jnp.float32),
      jnp.full((n_l1, 3, 64), -1e20, dtype=jnp.float32),
      jnp.zeros((n_l1, 2, 64), dtype=jnp.float32),
  ], axis=1)
  l0_aabbs = jnp.concatenate([
      jnp.concatenate([lmin_g, lmax_g, lval_g, jnp.zeros((n_l1, 1, BRANCH_FACTOR), dtype=jnp.float32)], axis=1),
      l0_pad,
  ], axis=2)

  l1_min = jnp.min(lmin_g, axis=2)  # (n_l1, 3)
  l1_max = jnp.max(lmax_g, axis=2)

  # Bottom-up Level 2 (BVH-64)
  n_l2 = max(1, (n_l1 + BRANCH_FACTOR - 1) // BRANCH_FACTOR)
  assert n_l2 <= BRANCH_FACTOR
  n_l1_p = n_l2 * BRANCH_FACTOR
  pad_1 = n_l1_p - n_l1
  if pad_1 > 0:
    m1min_p = jnp.concatenate([l1_min, jnp.full((pad_1, 3), 1e20, dtype=jnp.float32)], axis=0)
    m1max_p = jnp.concatenate([l1_max, jnp.full((pad_1, 3), -1e20, dtype=jnp.float32)], axis=0)
    m1val_p = jnp.concatenate([jnp.ones((n_l1,), dtype=jnp.float32), jnp.zeros((pad_1,), dtype=jnp.float32)], axis=0)
  else:
    m1min_p, m1max_p = l1_min, l1_max
    m1val_p = jnp.ones((n_l1,), dtype=jnp.float32)

  m1min_g = m1min_p.reshape(n_l2, BRANCH_FACTOR, 3).transpose(0, 2, 1)
  m1max_g = m1max_p.reshape(n_l2, BRANCH_FACTOR, 3).transpose(0, 2, 1)
  m1val_g = m1val_p.reshape(n_l2, 1, BRANCH_FACTOR)
  l2_pad = jnp.concatenate([
      jnp.full((n_l2, 3, 64), 1e20, dtype=jnp.float32),
      jnp.full((n_l2, 3, 64), -1e20, dtype=jnp.float32),
      jnp.zeros((n_l2, 2, 64), dtype=jnp.float32),
  ], axis=1)
  l2_aabbs = jnp.concatenate([
      jnp.concatenate([m1min_g, m1max_g, m1val_g, jnp.zeros((n_l2, 1, BRANCH_FACTOR), dtype=jnp.float32)], axis=1),
      l2_pad,
  ], axis=2)

  l2_min = jnp.min(m1min_g, axis=2)  # (n_l2, 3)
  l2_max = jnp.max(m1max_g, axis=2)
  pad_2 = 128 - n_l2
  l2_roots_aabb = jnp.concatenate([
      jnp.concatenate([l2_min.T, jnp.full((3, pad_2), 1e20, dtype=jnp.float32)], axis=1),
      jnp.concatenate([l2_max.T, jnp.full((3, pad_2), -1e20, dtype=jnp.float32)], axis=1),
      jnp.concatenate([jnp.ones((1, n_l2), dtype=jnp.float32), jnp.zeros((1, pad_2), dtype=jnp.float32)], axis=1),
      jnp.zeros((1, 128), dtype=jnp.float32),
  ], axis=0)
  return l2_roots_aabb, l2_aabbs, l0_aabbs, leaf_bank


def _make_uv_sphere_mesh(center=(0.0, 0.0, 0.45), radius=0.36, segs=48, rings=32) -> Dict[str, Any]:
  verts = []
  for r in range(rings + 1):
    phi = math.pi * r / rings
    sp, cp = math.sin(phi), math.cos(phi)
    for s in range(segs):
      th = 2.0 * math.pi * s / segs
      verts.append((center[0] + radius * sp * math.cos(th), center[1] + radius * sp * math.sin(th), center[2] + radius * cp))
  faces = []
  for r in range(rings):
    for s in range(segs):
      sn = (s + 1) % segs
      a, b = r * segs + s, r * segs + sn
      c, d = (r + 1) * segs + s, (r + 1) * segs + sn
      if r > 0:
        faces.append((a, c, b))
      if r < rings - 1:
        faces.append((b, c, d))
  return {
      "name": "collider_sphere",
      "verts": np.asarray(verts, dtype=np.float32),
      "faces": np.asarray(faces, dtype=np.int32),
      "rgba": np.array([0.82, 0.66, 0.30, 1.0], dtype=np.float32),
  }


def _make_table_mesh(hx=1.15, hy=1.15, z_top=0.08, z_bot=-0.04) -> Dict[str, Any]:
  verts = np.array([
      [-hx, -hy, z_bot], [hx, -hy, z_bot], [hx, hy, z_bot], [-hx, hy, z_bot],
      [-hx, -hy, z_top], [hx, -hy, z_top], [hx, hy, z_top], [-hx, hy, z_top],
  ], dtype=np.float32)
  faces = np.array([
      [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
      [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
      [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5],
  ], dtype=np.int32)
  return {
      "name": "pedestal_table",
      "verts": verts,
      "faces": faces,
      "rgba": np.array([0.32, 0.34, 0.38, 1.0], dtype=np.float32),
  }


def _cloth_grid_to_cpu_mesh(pos_np: np.ndarray) -> Dict[str, Any]:
  """Convert (Ny+1, Nx+1, 3) grid into flat triangle mesh for exact CPU BVH rebuild parity check."""
  ny1, nx1, _ = pos_np.shape
  ny, nx = ny1 - 1, nx1 - 1
  verts = pos_np.reshape(-1, 3)
  faces = []
  for iy in range(ny):
    for ix in range(nx):
      a = iy * nx1 + ix
      b = iy * nx1 + (ix + 1)
      c = (iy + 1) * nx1 + ix
      d = (iy + 1) * nx1 + (ix + 1)
      faces.append((a, b, d))
      faces.append((a, d, c))
  return {
      "name": "cloth_cpu_ref",
      "verts": verts.astype(np.float32),
      "faces": np.asarray(faces, dtype=np.int32),
      "rgba": np.array([0.78, 0.22, 0.28, 1.0], dtype=np.float32),
  }


def _build_scene_spec_for_cloth(nx: int, ny: int, H: int, W: int, static_items: List[CompactWideBVHMeshItem]) -> CompactWideBVHSceneSpec:
  n_tris = 2 * nx * ny
  n_leaves = (nx // 8) * (ny // 8)
  n_l1 = max(1, (n_leaves + BRANCH_FACTOR - 1) // BRANCH_FACTOR)
  n_l2 = max(1, (n_l1 + BRANCH_FACTOR - 1) // BRANCH_FACTOR)
  empty = np.empty((0,), dtype=np.float32)
  cloth_meta = CompactWideBVHMeshItem(
      mesh_id=0,
      name=f"deformable_cloth_{nx}x{ny}",
      n_tris_orig=n_tris,
      n_leaves=n_leaves,
      n_l1=n_l1,
      n_l2=n_l2,
      obj_aabb_min=np.array([-2.0, -2.0, -1.0], dtype=np.float32),
      obj_aabb_max=np.array([2.0, 2.0, 2.0], dtype=np.float32),
      l2_roots_aabb=empty,
      l2_aabbs=empty,
      l0_aabbs=empty,
      leaf_bank=empty,
      rgba=np.array([0.82, 0.22, 0.28, 1.0], dtype=np.float32),
  )
  meshes = [cloth_meta] + [dataclasses.replace(it, mesh_id=i + 1, l2_roots_aabb=empty, l2_aabbs=empty, l0_aabbs=empty, leaf_bank=empty) for i, it in enumerate(static_items)]
  return CompactWideBVHSceneSpec(
      height=H,
      width=W,
      cam_fovy=45.0,
      cam_projection=0,
      cam_sensorsize=(0.0, 0.0),
      cam_intrinsic=(0.0, 0.0, 0.0, 0.0),
      znear=0.1,
      zfar=50.0,
      meshes=tuple(meshes),
      light_pos=np.array([2.5, -2.0, 3.8], dtype=np.float32),
      light_ambient=(0.28, 0.30, 0.34),
      light_diffuse=(0.88, 0.85, 0.80),
      light_specular=(0.35, 0.35, 0.35),
      sky_rgb1=(0.26, 0.34, 0.46),
      sky_rgb2=(0.10, 0.13, 0.19),
      use_shadows=True,
  )


def _camera_matrices() -> Tuple[jax.Array, jax.Array]:
  cpos = np.array([2.25, -2.05, 1.65], dtype=np.float32)
  target = np.array([0.0, 0.0, 0.48], dtype=np.float32)
  fwd = target - cpos
  fwd /= np.linalg.norm(fwd)
  right = np.cross(fwd, np.array([0.0, 0.0, 1.0], dtype=np.float32))
  right /= np.linalg.norm(right)
  up = np.cross(right, fwd)
  cmat = np.stack([right, up, -fwd], axis=1).astype(np.float32)
  return jnp.asarray(cpos[None, :]), jnp.asarray(cmat[None, :, :])


def _agx_tonemap(hdr_rgb: jax.Array, exposure: float = 1.15) -> jax.Array:
  """AgX filmic tonemapping for rich studio cloth & metallic highlights."""
  x = jnp.maximum(hdr_rgb * exposure, 0.0)
  agx_in = jnp.array([
      [0.84247906, 0.07843360, 0.07922375],
      [0.04232824, 0.87846864, 0.07916613],
      [0.04237565, 0.07843360, 0.87914297],
  ], dtype=jnp.float32)
  x = jnp.einsum("ij,...j->...i", agx_in, x)
  x = jnp.clip(jnp.log2(jnp.maximum(x, 1e-6)), -12.47393, 4.026069)
  x = (x + 12.47393) / (4.026069 + 12.47393)
  x2 = x * x
  x4 = x2 * x2
  y = (
      15.5 * x4 * x2
      - 40.14 * x4 * x
      + 31.96 * x4
      - 6.868 * x2 * x
      + 0.4298 * x2
      + 0.1191 * x
      - 0.00232
  )
  agx_out = jnp.array([
      [1.19687900, -0.09802088, -0.09902974],
      [-0.05289685, 1.15190313, -0.09896118],
      [-0.05297163, -0.09804345, 1.15107367],
  ], dtype=jnp.float32)
  out = jnp.einsum("ij,...j->...i", agx_out, y)
  out = jnp.clip(out, 0.0, 1.0)
  return jnp.clip(out * out * (3.0 - 2.0 * out) * 0.18 + out * 0.82, 0.0, 1.0)


def shade_cloth_pbr_agx(gb, spec: CompactWideBVHSceneSpec) -> jax.Array:
  """Gather-free PBR fabric weave + brushed gold sphere + studio rim + AgX shading from GBuffer."""
  hit = gb.geom_id >= 0
  nrm = gb.normal
  vdir = gb.view_dir
  pos = gb.world_pos
  vis = 0.24 + 0.76 * gb.direct_shadow

  # Material classification: 0=crimson silk cloth, 1=brushed gold sphere, 2=charcoal slate table
  is_cloth = gb.geom_id == 0
  is_sphere = gb.geom_id == 1

  # Subtle woven textile micro-pattern & velvet grazing sheen on cloth
  weave = 0.92 + 0.08 * jnp.sin(pos[..., 0] * 140.0) * jnp.sin(pos[..., 1] * 140.0)
  cloth_alb = jnp.array([0.52, 0.06, 0.10], dtype=jnp.float32)[None, None, None, :] * weave[..., None]
  sphere_alb = jnp.array([0.84, 0.62, 0.24], dtype=jnp.float32)[None, None, None, :]
  table_chk = 0.90 + 0.10 * jnp.sign(jnp.sin(pos[..., 0] * 12.0) * jnp.sin(pos[..., 1] * 12.0))
  table_alb = jnp.array([0.16, 0.17, 0.20], dtype=jnp.float32)[None, None, None, :] * table_chk[..., None]

  albedo = jnp.where(is_cloth[..., None], cloth_alb, jnp.where(is_sphere[..., None], sphere_alb, table_alb))
  rough = jnp.where(is_cloth, 0.58, jnp.where(is_sphere, 0.24, 0.52))
  metal = jnp.where(is_sphere, 0.85, 0.0)

  l_pos = jnp.asarray(spec.light_pos, dtype=jnp.float32)[None, None, None, :]
  delta_l = l_pos - pos
  L = delta_l / jnp.maximum(jnp.linalg.norm(delta_l, axis=-1, keepdims=True), 1e-12)
  ndotl = jnp.maximum(0.0, jnp.sum(nrm * L, axis=-1))
  ndotv = jnp.maximum(1e-3, jnp.sum(nrm * vdir, axis=-1))

  H_vec = (L + vdir) / jnp.maximum(jnp.linalg.norm(L + vdir, axis=-1, keepdims=True), 1e-12)
  ndoth = jnp.maximum(0.0, jnp.sum(nrm * H_vec, axis=-1))
  a2 = (rough * rough) ** 2
  denom = ndoth * ndoth * (a2 - 1.0) + 1.0
  D_ggx = a2 / (np.pi * denom * denom + 1e-6)
  f0 = 0.04 * (1.0 - metal[..., None]) + albedo * metal[..., None]
  F_s = f0 + (1.0 - f0) * ((1.0 - jnp.maximum(0.0, jnp.sum(H_vec * vdir, axis=-1)))[..., None] ** 5.0)
  spec_rgb = (D_ggx[..., None] * F_s) / (4.0 * ndotv * jnp.maximum(ndotl, 0.05) + 0.12)[..., None]
  spec_rgb = jnp.minimum(spec_rgb, 3.5) * (ndotl * vis)[..., None]

  # Silk cloth grazing Charlie sheen
  sheen_w = ((1.0 - ndotv) ** 3.2) * is_cloth.astype(jnp.float32) * (0.35 + 0.65 * vis)
  sheen_rgb = jnp.array([0.85, 0.28, 0.34], dtype=jnp.float32)[None, None, None, :] * sheen_w[..., None] * 0.42

  # Warm studio key + cool studio rim + hemisphere bounce
  key_col = jnp.array([2.45, 2.32, 2.15], dtype=jnp.float32)[None, None, None, :]
  rim_dir = jnp.array([-0.68, 0.60, 0.42], dtype=jnp.float32)
  rim_dir = rim_dir / jnp.linalg.norm(rim_dir)
  rim_ndotl = jnp.maximum(0.0, jnp.sum(nrm * rim_dir[None, None, None, :], axis=-1))
  rim_col = jnp.array([0.55, 0.72, 0.98], dtype=jnp.float32)[None, None, None, :] * rim_ndotl[..., None]
  sky_up = 0.5 + 0.5 * nrm[..., 2]
  amb = (
      sky_up[..., None] * jnp.array([0.34, 0.40, 0.52], dtype=jnp.float32)[None, None, None, :]
      + (1.0 - sky_up[..., None]) * jnp.array([0.16, 0.14, 0.15], dtype=jnp.float32)[None, None, None, :]
  )

  diff_alb = albedo * (1.0 - metal[..., None])
  hdr_surf = diff_alb * (key_col * (ndotl * vis)[..., None] + rim_col * 0.65 + amb * 0.85) + spec_rgb * key_col + sheen_rgb

  # Soft studio backdrop gradient
  sky_t = jnp.clip(0.5 * (-vdir[..., 2] + 1.0), 0.0, 1.0)[..., None]
  bg_rgb = (
      sky_t * jnp.array([0.18, 0.23, 0.32], dtype=jnp.float32)[None, None, None, :]
      + (1.0 - sky_t) * jnp.array([0.07, 0.09, 0.13], dtype=jnp.float32)[None, None, None, :]
  )
  return _agx_tonemap(jnp.where(hit[..., None], hdr_surf, bg_rgb), exposure=1.18)


def run_wave4b(out_json: str, out_png: str):
  sph = build_compact_bvh_mesh_item(1, _make_uv_sphere_mesh(), "spatial_l1_morton")
  tbl = build_compact_bvh_mesh_item(2, _make_table_mesh(), "spatial_l1_morton")
  static_items = [sph, tbl]
  static_devs = tuple(
      (jnp.asarray(it.l2_roots_aabb), jnp.asarray(it.l2_aabbs), jnp.asarray(it.l0_aabbs), jnp.asarray(it.leaf_bank))
      for it in static_items
  )
  cxp, cxm = _camera_matrices()
  oxp = jnp.zeros((1, 3, 3), dtype=jnp.float32)
  oxm = jnp.broadcast_to(jnp.eye(3, dtype=jnp.float32)[None, None, :, :], (1, 3, 3, 3))

  results = {
      "backend": jax.default_backend(),
      "jax_version": jax.__version__,
      "device_kind": jax.devices()[0].device_kind,
      "parity": {},
      "benchmarks": [],
      "frames_rendered": [],
  }

  # 1. Parity Verification at 128x128 cloth (32,768 tris) after 40 simulation steps
  nx_p, ny_p = 128, 128
  pos, prev, dx, dy = make_cloth_initial_state(nx_p, ny_p)
  step_fn = jax.jit(lambda p, pp: pbd_cloth_step(p, pp, dx, dy))
  for _ in range(40):
    pos, prev = step_fn(pos, prev)
  jax.block_until_ready(pos)

  spec_p = _build_scene_spec_for_cloth(nx_p, ny_p, 128, 128, static_items)

  def _refit_and_render(p_cur):
    c_dev = refit_cloth_wide_bvh(p_cur)
    all_devs = (c_dev,) + static_devs
    gb = render_compact_bvh_batch(spec_p, all_devs, oxp, oxm, cxp, cxm, interpret=False)
    return gb, shade_cloth_pbr_agx(gb, spec_p)

  fn_rr = jax.jit(_refit_and_render)
  gb_refit, rgb_refit = fn_rr(pos)
  jax.block_until_ready(rgb_refit)

  # Fresh CPU BVH rebuild on deformed vertex positions
  cpu_cloth_item = build_compact_bvh_mesh_item(0, _cloth_grid_to_cpu_mesh(np.asarray(pos)), "spatial_l1_morton")
  cpu_cloth_dev = (jnp.asarray(cpu_cloth_item.l2_roots_aabb), jnp.asarray(cpu_cloth_item.l2_aabbs), jnp.asarray(cpu_cloth_item.l0_aabbs), jnp.asarray(cpu_cloth_item.leaf_bank))
  gb_cpu, rgb_cpu = jax.jit(lambda: (lambda gb: (gb, shade_cloth_pbr_agx(gb, spec_p)))(render_compact_bvh_batch(spec_p, (cpu_cloth_dev,) + static_devs, oxp, oxm, cxp, cxm, interpret=False)))()
  jax.block_until_ready(rgb_cpu)

  # HLO Check on full fused step + refit + render!
  def _full_step_refit_render(p, pp):
    pn, ppn = pbd_cloth_step(p, pp, dx, dy)
    c_dev = refit_cloth_wide_bvh(pn)
    gb = render_compact_bvh_batch(spec_p, (c_dev,) + static_devs, oxp, oxm, cxp, cxm, interpret=False)
    return pn, ppn, shade_cloth_pbr_agx(gb, spec_p)

  fn_full = jax.jit(_full_step_refit_render)
  htxt = fn_full.lower(pos, prev).as_text()
  hlo = inspect_hlo_ops(htxt)
  hlo["send_recv_ops"] = len(re.findall(r"\bsend\(|\brecv\(", htxt, re.I))
  assert hlo["host_callbacks"] == 0 and hlo["send_recv_ops"] == 0 and hlo["gather_ops"] == 0, hlo

  depth_diff = float(jnp.max(jnp.abs(gb_refit.depth - gb_cpu.depth)))
  geom_match = float(jnp.mean(gb_refit.geom_id == gb_cpu.geom_id))
  rgb_mae = float(jnp.mean(jnp.abs(rgb_refit - rgb_cpu)))
  assert depth_diff < 1e-4 and geom_match == 1.0, (depth_diff, geom_match)
  results["parity"] = {
      "cloth_grid": f"{nx_p}x{ny_p}",
      "cloth_tris": 2 * nx_p * ny_p,
      "frame": 40,
      "max_depth_diff_vs_cpu_rebuild_m": depth_diff,
      "geom_match": geom_match,
      "mean_rgb_diff": rgb_mae,
      "hlo_full_step_refit_render": hlo,
  }
  print("WAVE 4B PARITY PASS:", results["parity"], flush=True)

  # 2. Performance Benchmarks: step, refit, render, and fused step+refit+render
  for nx, ny in [(64, 64), (128, 128)]:
    p0, pp0, dx_i, dy_i = make_cloth_initial_state(nx, ny)
    fn_step = jax.jit(lambda p, pp: pbd_cloth_step(p, pp, dx_i, dy_i))
    fn_refit = jax.jit(refit_cloth_wide_bvh)
    jax.block_until_ready(fn_step(p0, pp0))
    jax.block_until_ready(fn_refit(p0))

    n_b = 20
    t0 = time.perf_counter()
    p_tmp, pp_tmp = p0, pp0
    for _ in range(n_b):
      p_tmp, pp_tmp = fn_step(p_tmp, pp_tmp)
    jax.block_until_ready(p_tmp)
    step_ms = (time.perf_counter() - t0) * 1e3 / n_b

    t0 = time.perf_counter()
    for _ in range(n_b):
      bv = fn_refit(p_tmp)
    jax.block_until_ready(bv)
    refit_ms = (time.perf_counter() - t0) * 1e3 / n_b

    for H, W in [(64, 64), (512, 512), (1024, 1024)]:
      sp_r = _build_scene_spec_for_cloth(nx, ny, H, W, static_items)

      def _fused(p, pp):
        pn, ppn = pbd_cloth_step(p, pp, dx_i, dy_i)
        c_dev = refit_cloth_wide_bvh(pn)
        gb = render_compact_bvh_batch(sp_r, (c_dev,) + static_devs, oxp, oxm, cxp, cxm, interpret=False)
        return pn, ppn, shade_cloth_pbr_agx(gb, sp_r)

      fn_f = jax.jit(_fused)
      fn_ren_only = jax.jit(lambda c_d: shade_cloth_pbr_agx(render_compact_bvh_batch(sp_r, (c_d,) + static_devs, oxp, oxm, cxp, cxm, interpret=False), sp_r))
      tc0 = time.perf_counter()
      _, _, r0 = fn_f(p0, pp0)
      jax.block_until_ready(r0)
      cs = time.perf_counter() - tc0

      bv_warm = fn_refit(p0)
      jax.block_until_ready(fn_ren_only(bv_warm))
      n_it = 5 if H >= 1024 else 10
      t_ren0 = time.perf_counter()
      for _ in range(n_it):
        r_only = fn_ren_only(bv_warm)
      jax.block_until_ready(r_only)
      render_ms = (time.perf_counter() - t_ren0) * 1e3 / n_it

      if nx == 128 and H == 1024:
        ht = fn_f.lower(p0, pp0).as_text()
        hi = inspect_hlo_ops(ht)
        hi["send_recv_ops"] = len(re.findall(r"\bsend\(|\brecv\(", ht, re.I))
        assert hi["host_callbacks"] == 0 and hi["send_recv_ops"] == 0 and hi["gather_ops"] == 0, hi
        results["hlo_128x128_cloth_1024x1024"] = hi

      t0 = time.perf_counter()
      pa, ppa = p0, pp0
      for _ in range(n_it):
        pa, ppa, rgb_out = fn_f(pa, ppa)
      jax.block_until_ready(rgb_out)
      tot_ms = (time.perf_counter() - t0) * 1e3 / n_it
      rec = {
          "cloth_grid": f"{nx}x{ny}",
          "cloth_tris": 2 * nx * ny,
          "res": f"{H}x{W}",
          "step_ms": round(step_ms, 3),
          "refit_ms": round(refit_ms, 3),
          "render_ms": round(render_ms, 3),
          "total_step_refit_render_ms": round(tot_ms, 3),
          "fps": round(1e3 / tot_ms, 2),
          "compile_s": round(cs, 2),
      }
      results["benchmarks"].append(rec)
      print("WAVE 4B BENCH:", rec, flush=True)

  # 3. Render 6-frame Hero Filmstrip (frames 0, 18, 36, 54, 72, 96) at 512x512 on 128x128 cloth (32,768 tris)
  sp_film = _build_scene_spec_for_cloth(128, 128, 512, 512, static_items)
  p_f, pp_f, dx_f, dy_f = make_cloth_initial_state(128, 128)
  fn_step_f = jax.jit(lambda p, pp: pbd_cloth_step(p, pp, dx_f, dy_f))
  fn_ren_f = jax.jit(lambda p: shade_cloth_pbr_agx(render_compact_bvh_batch(sp_film, (refit_cloth_wide_bvh(p),) + static_devs, oxp, oxm, cxp, cxm, interpret=False), sp_film))

  capture_frames = [0, 18, 36, 54, 72, 96]
  tiles = []
  for f_idx in range(capture_frames[-1] + 1):
    if f_idx in capture_frames:
      rgb = np.asarray(fn_ren_f(p_f)[0])
      im = Image.fromarray(np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8))
      dr = ImageDraw.Draw(im)
      dr.rectangle([8, 8, 220, 34], fill=(15, 18, 24))
      dr.text((14, 14), f"Frame t={f_idx} (32.8K tris live refit)", fill=(245, 245, 245))
      tiles.append(im)
      results["frames_rendered"].append(f_idx)
    p_f, pp_f = fn_step_f(p_f, pp_f)

  strip = Image.new("RGB", (512 * 3, 512 * 2))
  for idx, im in enumerate(tiles):
    strip.paste(im, ((idx % 3) * 512, (idx // 3) * 512))
  os.makedirs(os.path.dirname(out_png), exist_ok=True)
  strip.save(out_png)
  with open(out_json, "w") as f:
    json.dump(results, f, indent=2)
  print(f"Saved Wave 4B JSON to {out_json} and filmstrip to {out_png}", flush=True)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", default="/dev/shm/zaychik-tpu-mjx-w2/v3_w2_deformable_cloth.json")
  ap.add_argument("--png", default="/dev/shm/zaychik-tpu-mjx-w2/v3_w2_deformable_cloth_refit.png")
  args = ap.parse_args()
  run_wave4b(args.out, args.png)


if __name__ == "__main__":
  main()
