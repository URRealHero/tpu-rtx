"""Complex Multi-Bounce Path Tracing Suite on TPU v5p (4-chip shard_map).

Includes:
1. Open Khronos glTF Sample Assets loader (`trimesh` + per-triangle UV-sampled albedo,
   metallic/roughness/transmission/emissive PBR): Sponza Atrium, ABeautifulGame, FlightHelmet/Dragon.
2. Procedural Scene A: Sunlit Slatted-Window Classroom / Architectural Atrium (venetian blinds
   casting golden sunbeams + warm ceiling GI bounce + chalkboard + desks/chairs/props).
3. Procedural Scene B: Cluttered Multi-Material Cornell-Workshop (crimson/cyan walls, chrome mirror,
   dielectric glass IOR 1.50 refraction, frosted gold torus, porcelain Suzanne, copper/emerald props).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np
from PIL import Image, ImageDraw

from bvh_pbr_pt import (
    LightRig,
    PBRWideBVHMesh,
    audit_hlo_zero_copy,
    build_complex_pbr_pathtracer,
    build_pbr_wide_bvh,
)


# ==============================================================================
# Procedural Geometry Helpers (Boxes, Spheres, Cylinders, Tori, Subdivided Meshes)
# ==============================================================================

class MeshBuilder:
  def __init__(self):
    self.verts: List[np.ndarray] = []
    self.faces: List[np.ndarray] = []
    self.albs: List[np.ndarray] = []
    self.rghs: List[np.ndarray] = []
    self.mcodes: List[np.ndarray] = []
    self.v_offset = 0

  def add_mesh(
      self,
      v: np.ndarray,
      f: np.ndarray,
      albedo: Tuple[float, float, float],
      roughness: float,
      mat_code: float,
  ):
    v = np.asarray(v, dtype=np.float32)
    f = np.asarray(f, dtype=np.int32)
    nf = f.shape[0]
    self.verts.append(v)
    self.faces.append(f + self.v_offset)
    self.albs.append(np.broadcast_to(np.asarray(albedo, dtype=np.float32), (nf, 3)).copy())
    self.rghs.append(np.full((nf,), float(roughness), dtype=np.float32))
    self.mcodes.append(np.full((nf,), float(mat_code), dtype=np.float32))
    self.v_offset += v.shape[0]

  def add_box(
      self,
      center: Tuple[float, float, float],
      half_extents: Tuple[float, float, float],
      albedo: Tuple[float, float, float],
      roughness: float = 0.6,
      mat_code: float = 0.0,
      rot_euler: Tuple[float, float, float] = (0.0, 0.0, 0.0),
  ):
    hx, hy, hz = half_extents
    v = np.array([
        [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
        [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz],
    ], dtype=np.float32)
    f = np.array([
        [0, 2, 1], [0, 3, 2],  # -Z
        [4, 5, 6], [4, 6, 7],  # +Z
        [0, 1, 5], [0, 5, 4],  # -Y
        [2, 3, 7], [2, 7, 6],  # +Y
        [0, 4, 7], [0, 7, 3],  # -X
        [1, 2, 6], [1, 6, 5],  # +X
    ], dtype=np.int32)
    rx, ry, rz = rot_euler
    if rx != 0.0 or ry != 0.0 or rz != 0.0:
      cx, sx = math.cos(rx), math.sin(rx)
      cy, sy = math.cos(ry), math.sin(ry)
      cz, sz = math.cos(rz), math.sin(rz)
      Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
      Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
      Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
      v = v @ (Rz @ Ry @ Rx).T
    v = v + np.asarray(center, dtype=np.float32)[None, :]
    self.add_mesh(v, f, albedo, roughness, mat_code)

  def add_sphere(
      self,
      center: Tuple[float, float, float],
      radius: float,
      albedo: Tuple[float, float, float],
      roughness: float = 0.1,
      mat_code: float = 0.0,
      n_lat: int = 24,
      n_lon: int = 48,
  ):
    verts, faces = [], []
    for i in range(n_lat + 1):
      theta = math.pi * i / n_lat
      st, ct = math.sin(theta), math.cos(theta)
      for j in range(n_lon):
        phi = 2.0 * math.pi * j / n_lon
        verts.append([
            center[0] + radius * st * math.cos(phi),
            center[1] + radius * st * math.sin(phi),
            center[2] + radius * ct,
        ])
    for i in range(n_lat):
      for j in range(n_lon):
        jn = (j + 1) % n_lon
        a = i * n_lon + j
        b = i * n_lon + jn
        c = (i + 1) * n_lon + j
        d = (i + 1) * n_lon + jn
        if i > 0:
          faces.append([a, c, b])
        if i < n_lat - 1:
          faces.append([b, c, d])
    self.add_mesh(np.array(verts, np.float32), np.array(faces, np.int32), albedo, roughness, mat_code)

  def add_cylinder(
      self,
      base_center: Tuple[float, float, float],
      radius: float,
      height: float,
      albedo: Tuple[float, float, float],
      roughness: float = 0.2,
      mat_code: float = 0.0,
      n_seg: int = 32,
  ):
    cx, cy, cz = base_center
    verts, faces = [], []
    for j in range(n_seg):
      phi = 2.0 * math.pi * j / n_seg
      x, y = cx + radius * math.cos(phi), cy + radius * math.sin(phi)
      verts.append([x, y, cz])
      verts.append([x, y, cz + height])
    b_idx = len(verts)
    verts.append([cx, cy, cz])
    t_idx = len(verts)
    verts.append([cx, cy, cz + height])
    for j in range(n_seg):
      jn = (j + 1) % n_seg
      b0, t0 = 2 * j, 2 * j + 1
      b1, t1 = 2 * jn, 2 * jn + 1
      faces.append([b0, b1, t1])
      faces.append([b0, t1, t0])
      faces.append([b_idx, b1, b0])
      faces.append([t_idx, t0, t1])
    self.add_mesh(np.array(verts, np.float32), np.array(faces, np.int32), albedo, roughness, mat_code)

  def build(self, name: str) -> PBRWideBVHMesh:
    v = np.concatenate(self.verts, axis=0)
    f = np.concatenate(self.faces, axis=0)
    a = np.concatenate(self.albs, axis=0)
    r = np.concatenate(self.rghs, axis=0)
    m = np.concatenate(self.mcodes, axis=0)
    return build_pbr_wide_bvh(name, v, f, a, r, m)


def _look_at_mat(pos: Tuple[float, float, float], tgt: Tuple[float, float, float]) -> Tuple[np.ndarray, np.ndarray]:
  p = np.asarray(pos, dtype=np.float32)
  t = np.asarray(tgt, dtype=np.float32)
  fwd = t - p
  fwd = fwd / np.linalg.norm(fwd)
  up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
  right = np.cross(fwd, up)
  right = right / np.linalg.norm(right)
  cam_up = np.cross(right, fwd)
  rmat = np.stack([right, cam_up, -fwd], axis=1).astype(np.float32)
  return p[None, :], rmat[None, :, :]


def _subdivide_simple(verts: np.ndarray, faces: np.ndarray, k: int):
  if k <= 1:
    return verts.astype(np.float32), faces.astype(np.int32)
  pts_uv, idx_map = [], {}
  for i in range(k + 1):
    for j in range(k + 1 - i):
      idx_map[(i, j)] = len(pts_uv)
      pts_uv.append((i / k, j / k))
  pts_uv = np.array(pts_uv, dtype=np.float32)
  P_cnt = pts_uv.shape[0]
  sub_faces = []
  for i in range(k):
    for j in range(k - i):
      a, b, c = idx_map[(i, j)], idx_map[(i + 1, j)], idx_map[(i, j + 1)]
      sub_faces.append((a, b, c))
      if i + j + 1 < k:
        sub_faces.append((b, idx_map[(i + 1, j + 1)], c))
  sub_faces = np.array(sub_faces, dtype=np.int32)
  F = faces.shape[0]
  u, v = pts_uv[None, :, 0:1], pts_uv[None, :, 1:2]
  w = 1.0 - u - v
  nv = (w * verts[faces[:, 0]][:, None] + u * verts[faces[:, 1]][:, None] + v * verts[faces[:, 2]][:, None]).reshape(F * P_cnt, 3).astype(np.float32)
  nf = (sub_faces[None] + (np.arange(F, dtype=np.int32) * P_cnt)[:, None, None]).reshape(F * k * k, 3)
  return nv, nf


# ==============================================================================
# Procedural Scene A: Sunlit Slatted-Window Classroom / Architectural Atrium
# ==============================================================================

def build_classroom_sunlit():
  mb = MeshBuilder()
  # Oak plank floor (alternating warm plank tones)
  n_px, n_py = 12, 12
  for ix in range(n_px):
    for iy in range(n_py):
      x0 = -2.6 + (ix + 0.5) * (5.2 / n_px)
      y0 = -2.4 + (iy + 0.5) * (5.2 / n_py)
      tint = 0.85 + 0.25 * (((ix * 3 + iy * 7) % 5) / 4.0)
      mb.add_box((x0, y0, -0.04), (5.2 / n_px * 0.495, 5.2 / n_py * 0.495, 0.04),
                 (0.46 * tint, 0.28 * tint, 0.14 * tint), roughness=0.38, mat_code=0.0)

  # Walls & Ceiling (enclosed room with large slatted window on x = -2.6)
  wall_col = (0.82, 0.78, 0.72)
  mb.add_box((0.0, 2.84, 1.5), (2.7, 0.04, 1.55), wall_col, roughness=0.78, mat_code=0.0)   # Front wall
  mb.add_box((0.0, -2.44, 1.5), (2.7, 0.04, 1.55), wall_col, roughness=0.78, mat_code=0.0)  # Rear wall
  mb.add_box((2.64, 0.2, 1.5), (0.04, 2.65, 1.55), wall_col, roughness=0.78, mat_code=0.0)  # Right wall
  mb.add_box((0.0, 0.2, 3.04), (2.7, 2.7, 0.04), (0.86, 0.84, 0.80), roughness=0.82, mat_code=0.0)  # Ceiling
  # Overhead soft warm ceiling lamp panel
  mb.add_box((0.1, 0.4, 2.98), (0.85, 0.45, 0.02), (1.0, 0.94, 0.84), roughness=0.2, mat_code=3.4)

  # West window wall (x = -2.6) framing opening y in [-1.8, 2.2], z in [0.75, 2.65]
  mb.add_box((-2.64, 0.2, 0.375), (0.06, 2.7, 0.375), (0.72, 0.66, 0.58), roughness=0.72, mat_code=0.0)  # Lower sill wall
  mb.add_box((-2.64, 0.2, 2.825), (0.06, 2.7, 0.175), wall_col, roughness=0.78, mat_code=0.0)            # Header
  mb.add_box((-2.64, -2.12, 1.7), (0.06, 0.32, 0.95), wall_col, roughness=0.78, mat_code=0.0)            # Rear pillar
  mb.add_box((-2.64, 2.52, 1.7), (0.06, 0.32, 0.95), wall_col, roughness=0.78, mat_code=0.0)             # Front pillar
  # Window mullions + 18 angled Venetian blind slats casting dramatic sunbeams!
  for my in [-0.5, 0.85]:
    mb.add_box((-2.60, my, 1.7), (0.05, 0.04, 0.96), (0.28, 0.20, 0.14), roughness=0.45, mat_code=0.0)
  for s_i in range(18):
    sz = 0.80 + s_i * (1.80 / 17.0)
    mb.add_box((-2.56, 0.2, sz), (0.055, 1.98, 0.008), (0.76, 0.72, 0.65),
               roughness=0.45, mat_code=0.0, rot_euler=(0.0, 0.46, 0.0))

  # Chalkboard on front wall (y = 2.77)
  mb.add_box((0.0, 2.77, 1.55), (1.45, 0.025, 0.62), (0.36, 0.22, 0.11), roughness=0.42, mat_code=0.0)   # Oak frame
  mb.add_box((0.0, 2.75, 1.55), (1.36, 0.02, 0.54), (0.10, 0.23, 0.17), roughness=0.78, mat_code=0.0)    # Slate board
  mb.add_box((0.0, 2.70, 0.94), (1.36, 0.06, 0.02), (0.36, 0.22, 0.11), roughness=0.42, mat_code=0.0)    # Chalk tray

  # Teacher desk + globe + metallic lamp
  mb.add_box((1.1, 1.75, 0.74), (0.62, 0.36, 0.025), (0.65, 0.45, 0.25), roughness=0.32, mat_code=0.0)
  for lx in [0.54, 1.66]:
    for ly in [1.44, 2.06]:
      mb.add_cylinder((lx, ly, 0.0), 0.022, 0.72, (0.22, 0.23, 0.25), roughness=0.28, mat_code=0.85, n_seg=16)
  mb.add_sphere((0.78, 1.72, 0.92), 0.15, (0.22, 0.50, 0.78), roughness=0.22, mat_code=0.1, n_lat=18, n_lon=32)
  mb.add_cylinder((1.38, 1.76, 0.765), 0.08, 0.04, (0.85, 0.68, 0.28), roughness=0.18, mat_code=0.92, n_seg=20)
  mb.add_sphere((1.38, 1.76, 0.92), 0.09, (0.95, 0.85, 0.65), roughness=0.15, mat_code=3.2, n_lat=14, n_lon=24)

  # 6 Student desks & chairs (2 rows x 3 cols) + colorful books on desks
  book_cols = [(0.78, 0.22, 0.18), (0.18, 0.48, 0.74), (0.24, 0.62, 0.36), (0.82, 0.62, 0.18), (0.64, 0.28, 0.66), (0.22, 0.64, 0.68)]
  idx = 0
  for row_y in [0.55, -0.65]:
    for col_x in [-1.25, 0.0, 1.25]:
      # Desk top + steel legs
      mb.add_box((col_x, row_y, 0.70), (0.42, 0.28, 0.02), (0.72, 0.54, 0.34), roughness=0.36, mat_code=0.0)
      for dx in [-0.36, 0.36]:
        for dy in [-0.22, 0.22]:
          mb.add_cylinder((col_x + dx, row_y + dy, 0.0), 0.018, 0.68, (0.25, 0.26, 0.28), roughness=0.25, mat_code=0.85, n_seg=14)
      # Chair behind desk
      cy = row_y - 0.44
      mb.add_box((col_x, cy, 0.43), (0.21, 0.20, 0.02), (0.58, 0.40, 0.24), roughness=0.42, mat_code=0.0)
      mb.add_box((col_x, cy - 0.19, 0.66), (0.21, 0.02, 0.21), (0.58, 0.40, 0.24), roughness=0.42, mat_code=0.0)
      for dx in [-0.17, 0.17]:
        for dy in [-0.17, 0.17]:
          mb.add_cylinder((col_x + dx, cy + dy, 0.0), 0.014, 0.41, (0.25, 0.26, 0.28), roughness=0.25, mat_code=0.85, n_seg=12)
      # Book + metallic pencil cup on desk
      mb.add_box((col_x - 0.10, row_y + 0.03, 0.735), (0.12, 0.09, 0.015), book_cols[idx % len(book_cols)], roughness=0.55, mat_code=0.0, rot_euler=(0, 0, 0.15 * idx))
      mb.add_cylinder((col_x + 0.22, row_y + 0.12, 0.72), 0.032, 0.09, (0.82, 0.78, 0.72), roughness=0.18, mat_code=0.90, n_seg=16)
      idx += 1

  bvh = mb.build("classroom_sunlit")
  cxp, cxm = _look_at_mat((2.15, -2.05, 1.52), (-0.45, 0.85, 0.85))
  rig = LightRig(
      area_center=(0.1, 0.4, 2.92),
      area_normal=(0.0, 0.0, -1.0),
      area_radius=0.55,
      area_emit=(8.0, 7.2, 6.0),
      has_sun=True,
      sun_dir=(-0.82, -0.30, 0.48),
      sun_cos_half_angle=0.9992,
      sun_emit=(26.0, 20.5, 13.5),
      sky_zenith=(0.28, 0.48, 0.82),
      sky_horizon=(0.24, 0.28, 0.36),
      exposure=0.95,
  )
  return bvh, cxp, cxm, 48.0, 0.05, 40.0, rig


# ==============================================================================
# Procedural Scene B: Cluttered Multi-Material Cornell-Workshop
# ==============================================================================

def build_cornell_workshop():
  mb = MeshBuilder()
  ivory = (0.78, 0.76, 0.72)
  # Enclosed Cornell Box: x in [-1.25, 1.25], y in [-1.35, 1.25], z in [0, 2.25]
  mb.add_box((0.0, -0.05, -0.04), (1.30, 1.35, 0.04), ivory, roughness=0.72, mat_code=0.0)                 # Floor
  mb.add_box((0.0, -0.05, 2.29), (1.30, 1.35, 0.04), ivory, roughness=0.78, mat_code=0.0)                  # Ceiling
  mb.add_box((0.0, 1.29, 1.125), (1.30, 0.04, 1.17), ivory, roughness=0.75, mat_code=0.0)                  # Back wall
  mb.add_box((-1.29, -0.05, 1.125), (0.04, 1.35, 1.17), (0.80, 0.14, 0.12), roughness=0.72, mat_code=0.0)  # Left crimson wall
  mb.add_box((1.29, -0.05, 1.125), (0.04, 1.35, 1.17), (0.12, 0.60, 0.78), roughness=0.72, mat_code=0.0)   # Right cyan wall
  # Ceiling emissive softbox panel
  mb.add_box((0.0, 0.05, 2.23), (0.48, 0.42, 0.015), (1.0, 0.94, 0.86), roughness=0.2, mat_code=3.8)

  # Wooden workbench + slate pedestal
  mb.add_box((-0.28, 0.35, 0.56), (0.68, 0.44, 0.03), (0.56, 0.36, 0.18), roughness=0.35, mat_code=0.0)
  for lx in [-0.88, 0.32]:
    for ly in [-0.03, 0.73]:
      mb.add_box((lx, ly, 0.27), (0.035, 0.035, 0.27), (0.38, 0.24, 0.12), roughness=0.45, mat_code=0.0)
  mb.add_box((0.72, 0.45, 0.34), (0.34, 0.34, 0.34), (0.28, 0.30, 0.34), roughness=0.52, mat_code=0.0, rot_euler=(0, 0, -0.22))

  # Hero 1: Polished Chrome Mirror Sphere (on left workbench)
  mb.add_sphere((-0.58, 0.38, 0.85), 0.26, (0.94, 0.95, 0.97), roughness=0.03, mat_code=0.98, n_lat=28, n_lon=52)
  # Hero 2: Dielectric Clear Glass Sphere (IOR 1.50, foreground right)
  mb.add_sphere((0.36, -0.22, 0.25), 0.25, (0.96, 0.99, 0.98), roughness=0.02, mat_code=2.0, n_lat=28, n_lon=52)
  # Hero 3: Frosted Brushed Gold Torus (built from Blender torus mesh rotated)
  ds = np.load("/dev/shm/zaychik-tpu-mjx/out/assets/blender_real_scene.npz")
  tor_v, tor_f = _subdivide_simple(ds["mesh_1_verts"], ds["mesh_1_faces"], 4)
  tor_v = tor_v * 1.45
  # Rotate torus to lean upright on slate pedestal
  ang_x = math.radians(68.0)
  Rx = np.array([[1, 0, 0], [0, math.cos(ang_x), -math.sin(ang_x)], [0, math.sin(ang_x), math.cos(ang_x)]], np.float32)
  tor_v = (tor_v @ Rx.T) + np.array([0.72, 0.42, 0.92], np.float32)
  mb.add_mesh(tor_v, tor_f, (0.95, 0.74, 0.28), roughness=0.18, mat_code=0.95)

  # Hero 4: Subdivided Porcelain Suzanne on workbench catching red/cyan GI bleed!
  suz_v, suz_f = _subdivide_simple(ds["mesh_0_verts"], ds["mesh_0_faces"], 4)
  suz_v = suz_v * 1.45
  az = math.radians(-22.0)
  Rz = np.array([[math.cos(az), -math.sin(az), 0], [math.sin(az), math.cos(az), 0], [0, 0, 1]], np.float32)
  suz_v = (suz_v @ Rz.T) + np.array([0.02, 0.36, 0.78], np.float32)
  mb.add_mesh(suz_v, suz_f, (0.90, 0.88, 0.85), roughness=0.26, mat_code=0.05)

  # Dense workshop clutter: copper cylinder, emerald ceramic sphere, glass prism block, stacked cubes
  mb.add_cylinder((-0.68, -0.32, 0.0), 0.14, 0.42, (0.88, 0.48, 0.26), roughness=0.14, mat_code=0.92, n_seg=32)
  mb.add_sphere((-0.24, -0.38, 0.16), 0.16, (0.20, 0.74, 0.46), roughness=0.22, mat_code=0.08, n_lat=22, n_lon=40)
  mb.add_box((-0.18, 0.08, 0.67), (0.08, 0.08, 0.08), (0.95, 0.98, 1.0), roughness=0.02, mat_code=2.0, rot_euler=(0, 0, 0.4))
  mb.add_box((0.20, 0.12, 0.65), (0.06, 0.06, 0.06), (0.90, 0.32, 0.22), roughness=0.35, mat_code=0.1, rot_euler=(0, 0, -0.3))
  mb.add_box((0.76, -0.36, 0.10), (0.10, 0.10, 0.10), (0.86, 0.78, 0.36), roughness=0.24, mat_code=0.85, rot_euler=(0, 0, 0.5))

  bvh = mb.build("cornell_workshop")
  cxp, cxm = _look_at_mat((0.0, -2.55, 1.08), (0.0, 0.15, 0.86))
  rig = LightRig(
      area_center=(0.0, 0.05, 2.18),
      area_normal=(0.0, 0.0, -1.0),
      area_radius=0.45,
      area_emit=(12.5, 11.6, 10.4),
      has_sun=False,
      sky_zenith=(0.14, 0.16, 0.20),
      sky_horizon=(0.10, 0.11, 0.13),
      exposure=0.95,
  )
  return bvh, cxp, cxm, 44.0, 0.05, 20.0, rig


# ==============================================================================
# Khronos glTF Sample Assets Loader (with Per-Triangle UV Texture Sampling!)
# ==============================================================================

def _sample_tex_at_tri_centroids(uv: np.ndarray, faces: np.ndarray, pil_img: Image.Image) -> np.ndarray:
  arr = np.asarray(pil_img.convert("RGB"), dtype=np.float32) / 255.0
  H_t, W_t, _ = arr.shape
  tri_uv = (uv[faces[:, 0]] + uv[faces[:, 1]] + uv[faces[:, 2]]) * (1.0 / 3.0)
  u = np.mod(tri_uv[:, 0], 1.0)
  v = np.mod(1.0 - tri_uv[:, 1], 1.0)
  px = np.clip((u * (W_t - 1)).astype(np.int32), 0, W_t - 1)
  py = np.clip((v * (H_t - 1)).astype(np.int32), 0, H_t - 1)
  return arr[py, px]


def load_gltf_to_pbr_bvh(
    gltf_path: str,
    name: str,
    cam_pos: Tuple[float, float, float],
    cam_tgt: Tuple[float, float, float],
    light_rig: LightRig,
    cam_fovy: float = 48.0,
    rotate_z_up: bool = True,
    add_studio_floor: bool = False,
) -> Tuple[PBRWideBVHMesh, np.ndarray, np.ndarray, float, float, float, LightRig]:
  import trimesh

  scene = trimesh.load(gltf_path, force="scene")
  # Rotate glTF Y-up -> Z-up if needed
  R_yup_to_zup = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32) if rotate_z_up else np.eye(3, dtype=np.float32)

  raw_verts, raw_faces, raw_albs, raw_rghs, raw_mcodes = [], [], [], [], []
  v_off = 0

  for node_name in scene.graph.nodes_geometry:
    xform, geom_name = scene.graph[node_name]
    geom = scene.geometry[geom_name]
    if not isinstance(geom, trimesh.Trimesh) or len(geom.faces) == 0:
      continue
    v_local = np.asarray(geom.vertices, dtype=np.float32)
    f_local = np.asarray(geom.faces, dtype=np.int32)
    R = xform[:3, :3].astype(np.float32)
    t = xform[:3, 3].astype(np.float32)
    v_w = (v_local @ R.T + t[None, :]) @ R_yup_to_zup.T
    # Preserve outward winding if negative determinant
    if np.linalg.det(R) < 0:
      f_local = f_local[:, [0, 2, 1]]

    nf = f_local.shape[0]
    alb = np.full((nf, 3), 0.72, dtype=np.float32)
    rgh = 0.45
    met = 0.0
    is_glass = False

    vis = getattr(geom, "visual", None)
    mat = getattr(vis, "material", None) if vis is not None else None
    if mat is not None:
      bcf = getattr(mat, "baseColorFactor", None)
      base_tint = np.array([1.0, 1.0, 1.0], dtype=np.float32)
      if bcf is not None:
        bcf_a = np.asarray(bcf, dtype=np.float32).flatten()
        if bcf_a.size >= 3:
          base_tint = bcf_a[:3] / (255.0 if bcf_a.max() > 1.5 else 1.0)
      bct = getattr(mat, "baseColorTexture", None)
      uv = getattr(vis, "uv", None)
      if bct is not None and uv is not None and len(uv) == len(v_local):
        alb = np.clip(_sample_tex_at_tri_centroids(np.asarray(uv, np.float32), f_local, bct) * base_tint[None, :], 0.02, 0.99)
      else:
        alb = np.broadcast_to(np.clip(base_tint, 0.02, 0.99), (nf, 3)).copy()
      rf = getattr(mat, "roughnessFactor", None)
      mf = getattr(mat, "metallicFactor", None)
      if rf is not None:
        rgh = float(np.clip(rf, 0.04, 0.98))
      if mf is not None:
        met = float(np.clip(mf, 0.0, 1.0))
      # Check glass/transmission by material name or extras
      mname = str(getattr(mat, "name", "")).lower()
      if "glass" in mname or "window" in mname or "attenuation" in mname:
        is_glass = True
        rgh = 0.03

    raw_verts.append(v_w)
    raw_faces.append(f_local + v_off)
    raw_albs.append(alb)
    raw_rghs.append(np.full((nf,), rgh, dtype=np.float32))
    raw_mcodes.append(np.full((nf,), 2.0 if is_glass else met, dtype=np.float32))
    v_off += v_w.shape[0]

  v_all = np.concatenate(raw_verts, axis=0)
  f_all = np.concatenate(raw_faces, axis=0)
  a_all = np.concatenate(raw_albs, axis=0)
  r_all = np.concatenate(raw_rghs, axis=0)
  m_all = np.concatenate(raw_mcodes, axis=0)

  # Normalize non-Sponza single hero models to [-0.8, 0.8] sitting on z=0
  bmin, bmax = v_all.min(axis=0), v_all.max(axis=0)
  span = float(np.max(bmax - bmin))
  if span > 1e-5:
    target_span = 18.0 if "sponza" in name.lower() else 1.5
    scale = target_span / span
    center_xy = 0.5 * (bmin[:2] + bmax[:2])
    v_all[:, :2] -= center_xy[None, :]
    v_all[:, 2] -= bmin[2]
    v_all *= scale

  if add_studio_floor:
    mb_f = MeshBuilder()
    mb_f.add_box((0.0, 0.0, -0.025), (3.5, 3.5, 0.025), (0.68, 0.66, 0.62), roughness=0.48, mat_code=0.0)
    v_f, f_f = mb_f.verts[0], mb_f.faces[0] + v_all.shape[0]
    v_all = np.concatenate([v_all, v_f], axis=0)
    f_all = np.concatenate([f_all, f_f], axis=0)
    a_all = np.concatenate([a_all, mb_f.albs[0]], axis=0)
    r_all = np.concatenate([r_all, mb_f.rghs[0]], axis=0)
    m_all = np.concatenate([m_all, mb_f.mcodes[0]], axis=0)

  bvh = build_pbr_wide_bvh(name, v_all, f_all, a_all, r_all, m_all)
  cxp, cxm = _look_at_mat(cam_pos, cam_tgt)
  return bvh, cxp, cxm, cam_fovy, 0.05, 80.0, light_rig


def stitch_strip(out_arr: jax.Array) -> np.ndarray:
  arr = np.asarray(jax.device_get(out_arr), dtype=np.float32)
  n_c, H, W_loc, C = arr.shape
  return arr.transpose(1, 0, 2, 3).reshape(H, n_c * W_loc, C)


def run_scene(
    bvh: PBRWideBVHMesh,
    cxp_np: np.ndarray,
    cxm_np: np.ndarray,
    fovy: float,
    znear: float,
    zfar: float,
    rig: LightRig,
    devices,
    H: int,
    W: int,
    spps: List[int],
    max_bounces: int,
    audit_hlo: bool = True,
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
  recs = []
  last_img = None
  for idx, spp in enumerate(spps):
    fn, mesh, rep = build_complex_pbr_pathtracer(
        bvh, devices, H, W, fovy, znear, zfar, rig, spp=spp, max_bounces=max_bounces
    )
    cxp = jax.device_put(cxp_np, rep)
    cxm = jax.device_put(cxm_np, rep)
    rng = jax.device_put(jax.random.PRNGKey(idx + 10), rep)
    hlo_info = audit_hlo_zero_copy(fn.lower(cxp, cxm, rng)) if (audit_hlo and idx == 0) else None

    tc0 = time.perf_counter()
    y = fn(cxp, cxm, rng)
    jax.block_until_ready(y)
    compile_s = time.perf_counter() - tc0

    y = fn(cxp, cxm, jax.device_put(jax.random.PRNGKey(idx + 100), rep))
    jax.block_until_ready(y)

    n_steps = 1 if spp >= 128 else 2
    t0 = time.perf_counter()
    for it in range(n_steps):
      y = fn(cxp, cxm, jax.device_put(jax.random.PRNGKey(idx + 200 + it), rep))
    jax.block_until_ready(y)
    dt_s = (time.perf_counter() - t0) / n_steps

    ms = dt_s * 1e3
    fps = 1.0 / dt_s
    samples_per_min = fps * spp * 60.0
    rays_per_sample = (3 if rig.has_sun else 2) * max_bounces
    mrays_per_s = (H * W * spp * rays_per_sample / dt_s) / 1e6

    rec = {
        "scene": bvh.name,
        "tris": bvh.n_tris_orig,
        "leaves_128": bvh.n_leaves,
        "res": f"{H}x{W}",
        "spp": spp,
        "max_bounces": max_bounces,
        "has_sun": rig.has_sun,
        "n_chips": len(devices),
        "compile_s": round(compile_s, 2),
        "ms": round(ms, 2),
        "fps": round(fps, 3),
        "samples_per_min": round(samples_per_min, 1),
        "mrays_per_s": round(mrays_per_s, 2),
    }
    if hlo_info is not None:
      rec["hlo_zero_copy"] = True
      rec["hlo_ops"] = hlo_info
    print("COMPLEX PT:", rec, flush=True)
    recs.append(rec)
    last_img = stitch_strip(y)
  return recs, last_img


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--assets_dir", default="/dev/shm/zaychik-tpu-mjx-w7/assets")
  ap.add_argument("--res", type=int, default=512)
  ap.add_argument("--out_json", default="out/metrics/v3_w7_complex_scenes_pt.json")
  ap.add_argument("--out_png", default="out/figures/v3_w7_complex_scenes_pt.png")
  args = ap.parse_args()

  devs = jax.devices()
  assert len(devs) == 4
  H = W = args.res

  all_recs = []
  hero_panels: List[Tuple[str, np.ndarray]] = []

  # 1. Procedural Sunlit Slatted-Window Classroom (5-bounce GI + Sunbeams + Area Light)
  bvh_cls, cxp_cls, cxm_cls, fovy_cls, zn_cls, zf_cls, rig_cls = build_classroom_sunlit()
  recs_cls, img_cls = run_scene(
      bvh_cls, cxp_cls, cxm_cls, fovy_cls, zn_cls, zf_cls, rig_cls,
      devs, H, W, spps=[16, 64, 256], max_bounces=5,
  )
  all_recs.extend(recs_cls)
  hero_panels.append((f"Sunlit Slatted Classroom (5-Bounce GI, 256 spp | {recs_cls[-1]['ms']:.0f}ms)", img_cls))

  # 2. Procedural Cluttered Multi-Material Cornell-Workshop (5-bounce GI + Glass IOR 1.5 + Chrome + Gold)
  bvh_cw, cxp_cw, cxm_cw, fovy_cw, zn_cw, zf_cw, rig_cw = build_cornell_workshop()
  recs_cw, img_cw = run_scene(
      bvh_cw, cxp_cw, cxm_cw, fovy_cw, zn_cw, zf_cw, rig_cw,
      devs, H, W, spps=[16, 64, 256], max_bounces=5,
  )
  all_recs.extend(recs_cw)
  hero_panels.append((f"Cornell-Workshop: Glass/Chrome/Gold (5-Bounce, 256 spp | {recs_cw[-1]['ms']:.0f}ms)", img_cw))

  # 3. Khronos glTF Assets (Sponza Atrium, ABeautifulGame Chessboard, FlightHelmet / DragonAttenuation)
  gltf_specs = [
      (
          "Khronos_Sponza",
          os.path.join(args.assets_dir, "Sponza/glTF/Sponza.gltf"),
          (-6.2, 0.0, 2.2),
          (4.0, 0.0, 2.6),
          LightRig(
              area_center=(0.0, 0.0, 8.0),
              area_normal=(0.0, 0.0, -1.0),
              area_radius=2.5,
              area_emit=(12.0, 11.2, 9.8),
              has_sun=True,
              sun_dir=(0.25, -0.35, 0.90),
              sun_cos_half_angle=0.9985,
              sun_emit=(22.0, 18.0, 13.0),
              exposure=1.0,
          ),
          52.0,
          False,
      ),
      (
          "Khronos_ABeautifulGame",
          os.path.join(args.assets_dir, "ABeautifulGame/glTF/ABeautifulGame.gltf"),
          (1.35, -1.45, 0.95),
          (0.0, 0.0, 0.22),
          LightRig(
              area_center=(0.9, -0.7, 2.2),
              area_normal=(-0.3, 0.2, -1.0),
              area_radius=0.65,
              area_emit=(14.0, 13.0, 11.8),
              has_sun=False,
              exposure=1.05,
          ),
          42.0,
          True,
      ),
      (
          "Khronos_FlightHelmet",
          os.path.join(args.assets_dir, "FlightHelmet/glTF/FlightHelmet.gltf"),
          (1.15, -1.25, 0.95),
          (0.0, 0.0, 0.65),
          LightRig(
              area_center=(0.9, -0.8, 2.3),
              area_normal=(-0.3, 0.2, -1.0),
              area_radius=0.65,
              area_emit=(14.0, 13.0, 11.8),
              has_sun=False,
              exposure=1.05,
          ),
          42.0,
          True,
      ),
  ]

  for g_name, g_path, cpos, ctgt, grig, gfovy, gfloor in gltf_specs:
    if not os.path.exists(g_path):
      print(f"WARN: Missing glTF asset {g_path}, skipping", flush=True)
      continue
    bvh_g, cxp_g, cxm_g, fovy_g, zn_g, zf_g, rig_g = load_gltf_to_pbr_bvh(
        g_path, g_name, cpos, ctgt, grig, cam_fovy=gfovy, rotate_z_up=True, add_studio_floor=gfloor
    )
    recs_g, img_g = run_scene(
        bvh_g, cxp_g, cxm_g, fovy_g, zn_g, zf_g, rig_g,
        devs, H, W, spps=[16, 64, 256], max_bounces=4,
    )
    all_recs.extend(recs_g)
    hero_panels.append((f"{g_name} ({bvh_g.n_tris_orig:,} tris, 4-Bounce, 256 spp | {recs_g[-1]['ms']:.0f}ms)", img_g))

  # Assemble 2x3 or 2x2 crisp hero showcase PNG
  n_p = len(hero_panels)
  n_cols = 3 if n_p >= 5 else 2
  n_rows = (n_p + n_cols - 1) // n_cols
  pad = 14
  hdr = 40
  canvas = Image.new("RGB", (n_cols * W + (n_cols + 1) * pad, n_rows * (H + hdr) + (n_rows + 1) * pad), (15, 17, 21))
  draw = ImageDraw.Draw(canvas)
  for idx, (title, img_f) in enumerate(hero_panels):
    r, c = idx // n_cols, idx % n_cols
    x0 = pad + c * (W + pad)
    y0 = pad + r * (H + hdr + pad)
    u8 = np.clip(img_f * 255.0 + 0.5, 0, 255).astype(np.uint8)
    canvas.paste(Image.fromarray(u8), (x0, y0 + hdr))
    draw.text((x0 + 4, y0 + 12), title, fill=(238, 242, 248))

  os.makedirs(os.path.dirname(args.out_png) or ".", exist_ok=True)
  canvas.save(args.out_png)
  print("Saved Complex Scenes Hero PNG ->", args.out_png, canvas.size, flush=True)

  os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
  with open(args.out_json, "w") as f:
    json.dump({"experiment": "v3_w7_complex_scenes_pt", "backend": jax.default_backend(), "n_chips": 4, "runs": all_recs}, f, indent=2)
  print("Saved Complex Scenes JSON ->", args.out_json, flush=True)


if __name__ == "__main__":
  main()
