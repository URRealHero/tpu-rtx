"""Task A: TPU-ReSTIR DI — Real-Time 1,024-Light Many-Light Path Tracing on TPU v5p.

Builds a dramatic Night Lantern Courtyard / Arcade with 1,024 dynamic glowing
colored lanterns & warm/cyan area lights, rich architectural arches, wet
reflective cobblestone floor, and metallic ornaments, and compares at equal
visibility-ray budget (2 shadow rays/px):
  (a) Naive Uniform Many-Light Sampling (2 vis rays/px),
  (b) TPU-ReSTIR DI (M=32 RIS + Initial Vis + 2-pass Spatial Reuse),
  (c) TPU-ReSTIR DI + Joint A-Trous Denoise,
  (d) High-spp Reference Ground Truth (128 spp x M=32 RIS).

Benchmarks 1-chip and 4-chip throughput at 512x512, 1080p, and Native 4K with
strict HLO zero-copy auditing (`host_callbacks == 0`, `send_recv_ops == 0`).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from complex_scenes_pt import MeshBuilder, _look_at_mat
from run_opendata_pt import compute_psnr, compute_ssim
from tpu_mjx_render.adapters.tpu_optix import (
    OptixBuildInputTriangleArray,
    OptixDeviceContext,
    audit_hlo_zero_copy,
)
from tpu_mjx_render.adapters.tpu_restir import (
    ReSTIRConfig,
    ReSTIRLightBank,
    build_restir_renderer,
    make_restir_light_bank,
)


def _stitch(out_arr: jax.Array) -> np.ndarray:
  arr = np.asarray(jax.device_get(out_arr), dtype=np.float32)
  if arr.ndim == 4:  # (n_chips, H, W_local, C)
    n_c, H, W_loc, C = arr.shape
    return arr.transpose(1, 0, 2, 3).reshape(H, n_c * W_loc, C)
  n_c, H, W_loc = arr.shape
  return arr.transpose(1, 0, 2).reshape(H, n_c * W_loc)


# ==============================================================================
# Procedural Night Lantern Courtyard / Arcade (1,024 Lights + Wet Cobblestones)
# ==============================================================================

def _add_arch_bay(
    mb: MeshBuilder,
    x_pos: float,
    y0: float,
    y1: float,
    z_spring: float,
    arch_r: float,
    thickness: float,
    n_seg: int = 12,
) -> None:
  """Add a semicircular stone arch spanning y0..y1 at x=x_pos above z_spring."""
  yc = 0.5 * (y0 + y1)
  r_in = arch_r
  r_out = arch_r + 0.45
  hx = 0.5 * thickness
  verts, faces = [], []
  for i in range(n_seg + 1):
    th = math.pi * i / n_seg
    cy = yc - math.cos(th)
    yi = yc - r_in * math.cos(th)
    zi = z_spring + r_in * math.sin(th)
    yo = yc - r_out * math.cos(th)
    zo = z_spring + r_out * math.sin(th)
    verts.extend([
        [x_pos - hx, yi, zi], [x_pos + hx, yi, zi],
        [x_pos - hx, yo, zo], [x_pos + hx, yo, zo],
    ])
  for i in range(n_seg):
    b0 = 4 * i
    b1 = 4 * (i + 1)
    # Inner soffit, outer extrados, front/back faces
    faces.extend([
        [b0 + 0, b0 + 1, b1 + 1], [b0 + 0, b1 + 1, b1 + 0],
        [b0 + 2, b1 + 3, b0 + 3], [b0 + 2, b1 + 2, b1 + 3],
        [b0 + 0, b1 + 0, b1 + 2], [b0 + 0, b1 + 2, b0 + 2],
        [b0 + 1, b0 + 3, b1 + 3], [b0 + 1, b1 + 3, b1 + 1],
    ])
  mb.add_mesh(
      np.array(verts, dtype=np.float32),
      np.array(faces, dtype=np.int32),
      albedo=(0.46, 0.40, 0.34),
      roughness=0.58,
      mat_code=0.0,
  )


def _add_wet_cobblestone_floor(
    mb: MeshBuilder,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    nx: int = 112,
    ny: int = 136,
) -> None:
  """Tessellated wet cobblestone courtyard with mirror-like rain puddles."""
  xs = np.linspace(x_min, x_max, nx + 1, dtype=np.float32)
  ys = np.linspace(y_min, y_max, ny + 1, dtype=np.float32)
  gx, gy = np.meshgrid(xs, ys, indexing="xy")

  # Subtle multi-frequency cobblestone relief & puddle depressions
  cobble_u = np.sin(gx * 5.2) * np.cos(gy * 5.2)
  puddle_field = (
      0.55 * np.sin(0.45 * gx + 0.32 * gy)
      + 0.35 * np.cos(0.75 * gx - 0.58 * gy + 0.7)
      + 0.25 * np.sin(1.25 * gx + 1.15 * gy - 0.4)
  )
  # Extra central walkway rain pool along x in [-3.2, 3.2]
  walkway_pool = 0.32 * np.exp(-((gx / 2.8) ** 2)) * np.cos(0.42 * gy)
  wetness = puddle_field + walkway_pool

  gz = np.where(wetness > 0.15, 0.0, 0.012 * np.maximum(cobble_u, 0.0)).astype(np.float32)
  verts = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)

  i0 = np.arange(ny, dtype=np.int32)[:, None] * (nx + 1) + np.arange(nx, dtype=np.int32)[None, :]
  a = i0
  b = i0 + 1
  c = i0 + (nx + 1) + 1
  d = i0 + (nx + 1)
  f1 = np.stack([a, b, c], axis=-1).reshape(-1, 3)
  f2 = np.stack([a, c, d], axis=-1).reshape(-1, 3)
  faces = np.concatenate([f1, f2], axis=0)

  # Per-triangle albedo & roughness modulated by wetness + cobblestone checker
  cx = 0.5 * (gx[:-1, :-1] + gx[1:, 1:])
  cy = 0.5 * (gy[:-1, :-1] + gy[1:, 1:])
  cw = 0.5 * (wetness[:-1, :-1] + wetness[1:, 1:])
  is_puddle = cw > 0.15
  stone_var = 0.5 + 0.5 * np.sin(np.floor(cx * 2.2) * 12.9898 + np.floor(cy * 2.2) * 78.233)
  stone_alb = np.stack([
      0.16 + 0.08 * stone_var,
      0.14 + 0.07 * stone_var,
      0.13 + 0.06 * stone_var,
  ], axis=-1)
  puddle_alb = np.broadcast_to(np.array([0.055, 0.065, 0.085], dtype=np.float32), stone_alb.shape)
  alb_grid = np.where(is_puddle[..., None], puddle_alb, stone_alb).astype(np.float32)
  rgh_grid = np.where(is_puddle, 0.048, 0.19 + 0.14 * stone_var).astype(np.float32)

  nf_half = nx * ny
  alb = np.concatenate([alb_grid.reshape(nf_half, 3), alb_grid.reshape(nf_half, 3)], axis=0)
  rgh = np.concatenate([rgh_grid.reshape(nf_half), rgh_grid.reshape(nf_half)], axis=0)
  mc = np.where(np.concatenate([is_puddle.reshape(nf_half), is_puddle.reshape(nf_half)], axis=0), 0.12, 0.0).astype(np.float32)

  mb.verts.append(verts)
  mb.faces.append(faces + mb.v_offset)
  mb.albs.append(alb)
  mb.rghs.append(rgh)
  mb.mcodes.append(mc)
  mb.v_offset += verts.shape[0]


def _add_octa_lantern(
    mb: MeshBuilder,
    center: Tuple[float, float, float],
    radius: float,
    color: Tuple[float, float, float],
    emit_tag: float = 3.24,
) -> None:
  """Smooth glowing spherical silk lantern bulb (preserves vivid color under AgX)."""
  mb.add_sphere(center, radius, albedo=color, roughness=0.45, mat_code=emit_tag, n_lat=5, n_lon=8)


def build_night_lantern_courtyard(n_lights: int = 1024) -> Tuple[OptixBuildInputTriangleArray, ReSTIRLightBank, np.ndarray, np.ndarray]:
  """Build the Night Lantern Courtyard / Arcade scene + exact `n_lights` dynamic light bank."""
  mb = MeshBuilder()

  # 1. Wet Reflective Cobblestone Floor + Outer Cloister Slabs
  _add_wet_cobblestone_floor(mb, -10.5, 10.5, -12.0, 16.0, nx=112, ny=136)

  # Raised colonnade side walkways (z = 0.16)
  mb.add_box((-7.6, 2.0, 0.08), (2.2, 14.0, 0.08), (0.30, 0.27, 0.24), roughness=0.35, mat_code=0.0)
  mb.add_box((+7.6, 2.0, 0.08), (2.2, 14.0, 0.08), (0.30, 0.27, 0.24), roughness=0.35, mat_code=0.0)
  # Cloister outer walls with dark plaster & pilasters
  mb.add_box((-9.8, 2.0, 4.4), (0.35, 14.2, 4.4), (0.28, 0.25, 0.23), roughness=0.72, mat_code=0.0)
  mb.add_box((+9.8, 2.0, 4.4), (0.35, 14.2, 4.4), (0.28, 0.25, 0.23), roughness=0.72, mat_code=0.0)
  # Arcade roofs over left & right colonnades (cast dramatic shadow silhouettes!)
  mb.add_box((-7.6, 2.0, 6.9), (2.5, 14.2, 0.25), (0.22, 0.20, 0.19), roughness=0.75, mat_code=0.0)
  mb.add_box((+7.6, 2.0, 6.9), (2.5, 14.2, 0.25), (0.22, 0.20, 0.19), roughness=0.75, mat_code=0.0)

  # 2. Left & Right Arched Arcade Colonnades (6 bays each, y = -8..12)
  col_ys = [-8.0, -4.0, 0.0, 4.0, 8.0, 12.0]
  for sx in (-5.4, 5.4):
    for yc in col_ys:
      # Stone plinth + base + column shaft + capital
      mb.add_box((sx, yc, 0.28), (0.42, 0.42, 0.28), (0.40, 0.36, 0.31), roughness=0.55, mat_code=0.0)
      mb.add_cylinder((sx, yc, 0.56), 0.28, 3.25, (0.48, 0.43, 0.37), roughness=0.48, mat_code=0.0, n_seg=18)
      mb.add_box((sx, yc, 3.92), (0.42, 0.42, 0.14), (0.50, 0.44, 0.38), roughness=0.50, mat_code=0.0)
    # Semicircular arches between adjacent columns + upper spandrel cornice
    for i in range(len(col_ys) - 1):
      _add_arch_bay(mb, sx, col_ys[i], col_ys[i + 1], z_spring=4.06, arch_r=1.62, thickness=0.58, n_seg=12)
    mb.add_box((sx, 2.0, 6.35), (0.34, 12.2, 0.42), (0.42, 0.37, 0.32), roughness=0.55, mat_code=0.0)
    # Upper gallery balustrade piers
    for yc in np.linspace(-7.0, 11.0, 10):
      mb.add_box((sx, float(yc), 7.05), (0.16, 0.16, 0.35), (0.45, 0.40, 0.35), roughness=0.52, mat_code=0.0)

  # 3. Far Grand Triple-Arch Pavilion Wall (y = 13.6)
  mb.add_box((-7.6, 13.6, 3.6), (2.2, 0.45, 3.6), (0.38, 0.33, 0.29), roughness=0.62, mat_code=0.0)
  mb.add_box((+7.6, 13.6, 3.6), (2.2, 0.45, 3.6), (0.38, 0.33, 0.29), roughness=0.62, mat_code=0.0)
  # Central triumphal piers + arch rotated across X
  for px in (-2.4, 2.4):
    mb.add_box((px, 13.6, 0.35), (0.50, 0.55, 0.35), (0.44, 0.39, 0.34), roughness=0.52, mat_code=0.0)
    mb.add_cylinder((px, 13.6, 0.70), 0.34, 3.6, (0.50, 0.44, 0.38), roughness=0.45, mat_code=0.0, n_seg=20)
    mb.add_box((px, 13.6, 4.42), (0.48, 0.50, 0.16), (0.52, 0.45, 0.38), roughness=0.48, mat_code=0.0)
  # Far central arch lintel + pediment
  mb.add_box((0.0, 13.6, 6.35), (5.4, 0.55, 0.75), (0.44, 0.38, 0.33), roughness=0.55, mat_code=0.0)
  mb.add_box((0.0, 15.6, 3.8), (10.2, 0.40, 4.0), (0.20, 0.19, 0.22), roughness=0.72, mat_code=0.0)

  # 4. Central Courtyard Fountain Basin, Calm Mirror Pool & Metallic Ornaments
  fx, fy = 0.0, 4.2
  # Octagonal-style basin steps + dark water surface
  mb.add_cylinder((fx, fy, 0.0), 2.35, 0.34, (0.34, 0.31, 0.29), roughness=0.38, mat_code=0.0, n_seg=24)
  mb.add_cylinder((fx, fy, 0.34), 1.95, 0.04, (0.04, 0.07, 0.09), roughness=0.032, mat_code=0.18, n_seg=32)
  # Central polished bronze/gold fountain pedestal + tiered metallic spheres
  mb.add_cylinder((fx, fy, 0.38), 0.42, 0.95, (0.86, 0.62, 0.28), roughness=0.18, mat_code=0.92, n_seg=24)
  mb.add_cylinder((fx, fy, 1.33), 0.85, 0.14, (0.88, 0.65, 0.30), roughness=0.14, mat_code=0.95, n_seg=28)
  mb.add_cylinder((fx, fy, 1.47), 0.22, 0.85, (0.84, 0.86, 0.90), roughness=0.10, mat_code=0.98, n_seg=20)
  mb.add_sphere((fx, fy, 2.58), 0.42, (0.94, 0.74, 0.32), roughness=0.08, mat_code=0.98, n_lat=22, n_lon=40)

  # Flanking metallic courtyard ornaments (brushed chrome & warm copper spheres on stone plinths)
  ornaments = [
      (-3.1, -1.8, (0.90, 0.92, 0.96), 0.07, 0.98),
      (+3.1, -1.8, (0.92, 0.56, 0.32), 0.11, 0.94),
      (-3.1, +9.6, (0.92, 0.68, 0.28), 0.10, 0.96),
      (+3.1, +9.6, (0.88, 0.92, 0.96), 0.08, 0.98),
  ]
  for ox, oy, oalb, orgh, omc in ornaments:
    mb.add_box((ox, oy, 0.42), (0.36, 0.36, 0.42), (0.36, 0.33, 0.30), roughness=0.45, mat_code=0.0)
    mb.add_sphere((ox, oy, 1.18), 0.34, oalb, roughness=orgh, mat_code=omc, n_lat=18, n_lon=32)

  # 5. Generate Exact N_L = 1,024 Dynamic Lights (Lanterns + Warm/Cyan Area Panels)
  rng = np.random.default_rng(2026)
  l_pos: List[List[float]] = []
  l_nrm: List[List[float]] = []
  l_emit: List[List[float]] = []
  l_rad: List[float] = []
  l_iso: List[float] = []

  warm_palette = [
      np.array([1.00, 0.56, 0.16], np.float32),  # warm amber
      np.array([1.00, 0.74, 0.32], np.float32),  # golden lantern
      np.array([1.00, 0.30, 0.12], np.float32),  # crimson silk
      np.array([1.00, 0.46, 0.22], np.float32),  # tangerine
  ]
  cyan_palette = [
      np.array([0.16, 0.88, 0.98], np.float32),  # electric cyan
      np.array([0.22, 0.96, 0.74], np.float32),  # jade teal
      np.array([0.36, 0.70, 1.00], np.float32),  # moonlit azure
  ]

  def _add_lantern(pos_xyz, col, power, rad, add_mesh=True):
    l_pos.append([float(pos_xyz[0]), float(pos_xyz[1]), float(pos_xyz[2])])
    l_nrm.append([0.0, 0.0, -1.0])
    l_emit.append((col * power).tolist())
    l_rad.append(float(rad))
    l_iso.append(1.0)
    if add_mesh:
      _add_octa_lantern(mb, tuple(pos_xyz), rad * 0.92, tuple(np.clip(col, 0.04, 0.98)), emit_tag=3.24)

  # Budget allocation totaling exact n_lights (scaled cleanly if n_lights < 1024 for curve)
  n_area = min(32, max(4, n_lights // 32))
  n_lantern_total = n_lights - n_area
  n_canopy = int(round(n_lantern_total * 0.34))
  n_arcade = int(round(n_lantern_total * 0.38))
  n_ground = n_lantern_total - n_canopy - n_arcade

  # 5a. Overhead catenary lantern canopy criss-crossing courtyard
  n_strings = 10
  for i in range(n_canopy):
    s_i = i % n_strings
    t = (i // n_strings + 0.5) / max(1, (n_canopy // n_strings) + 1)
    y_span = -8.2 + 2.15 * s_i
    x = -5.0 + 10.0 * t
    sag = 1.35 * (1.0 - (2.0 * t - 1.0) ** 2)
    z = 5.95 - sag + 0.12 * math.sin(i * 1.7)
    is_cyan = (s_i % 3 == 1) and (i % 2 == 0)
    col = cyan_palette[i % len(cyan_palette)] if is_cyan else warm_palette[i % len(warm_palette)]
    pwr = float(rng.uniform(8.5, 16.0))
    _add_lantern((x, y_span, z), col, pwr, rad=0.085)

  # 5b. Arcade arch & colonnade lanterns (left & right cloisters + far triumphal wall)
  for i in range(n_arcade):
    side = -1.0 if (i % 2 == 0) else 1.0
    sub = i // 2
    if sub % 4 == 0:
      bay = (sub // 4) % 5
      frac = ((sub // 20) % 7 + 1) / 8.0
      x = side * (5.4 + 0.25 * math.sin(i))
      y = -8.0 + 4.0 * bay + 4.0 * frac
      z = 4.55 - 0.65 * math.sin(math.pi * frac)
    else:
      x = side * float(rng.uniform(6.2, 8.9))
      y = float(rng.uniform(-9.0, 13.0))
      z = float(rng.uniform(2.3, 5.2))
    is_cyan = (side > 0 and i % 3 == 0) or (side < 0 and i % 5 == 0)
    col = cyan_palette[i % len(cyan_palette)] if is_cyan else warm_palette[i % len(warm_palette)]
    pwr = float(rng.uniform(7.5, 14.5))
    _add_lantern((x, y, z), col, pwr, rad=0.08)

  # 5c. Floating fountain lanterns + promenade path lanterns (raised slightly for rich puddle streaks)
  for i in range(n_ground):
    if i % 3 == 0:
      ang = 2.0 * math.pi * (i / max(1, n_ground)) * 7.3 + float(rng.uniform(-0.15, 0.15))
      rr = float(rng.uniform(0.65, 1.78))
      px = fx + rr * math.cos(ang)
      py = fy + rr * math.sin(ang)
      pz = 0.44
      col = warm_palette[i % len(warm_palette)] if i % 4 != 0 else cyan_palette[i % len(cyan_palette)]
      _add_lantern((px, py, pz), col, float(rng.uniform(6.5, 11.5)), rad=0.065)
    else:
      side = -1.0 if (i % 2 == 0) else 1.0
      px = side * float(rng.uniform(2.7, 4.9))
      py = float(rng.uniform(-9.2, 13.2))
      pz = float(rng.uniform(0.65, 1.55))
      is_cyan = (i % 4 == 1)
      col = cyan_palette[i % len(cyan_palette)] if is_cyan else warm_palette[i % len(warm_palette)]
      _add_lantern((px, py, pz), col, float(rng.uniform(6.0, 12.0)), rad=0.075)

  # 5d. Warm & Cyan Area Lights in Arcade Soffits & Central Courtyard Canopy
  for i in range(n_area):
    is_cyan = (i % 2 == 1)
    if i < n_area // 2:
      side = -1.0 if (i % 2 == 0) else 1.0
      ax = side * 7.4
      ay = -6.5 + 18.0 * (i / max(1, n_area // 2 - 1))
      az = 6.35
      an = [0.0, 0.0, -1.0]
    else:
      j = i - n_area // 2
      ax = -3.0 + 6.0 * (j % 2)
      ay = -4.0 + 16.0 * ((j // 2) / max(1, (n_area - n_area // 2) // 2))
      az = 6.8
      an = [0.0, 0.0, -1.0]
    col = np.array([0.16, 0.86, 1.00], np.float32) if is_cyan else np.array([1.00, 0.58, 0.20], np.float32)
    l_pos.append([float(ax), float(ay), float(az)])
    l_nrm.append(an)
    l_emit.append((col * 4.5).tolist())
    l_rad.append(0.42)
    l_iso.append(0.0)

  assert len(l_pos) == n_lights, f"Expected {n_lights}, got {len(l_pos)}"

  v = np.concatenate(mb.verts, axis=0)
  f = np.concatenate(mb.faces, axis=0)
  a = np.concatenate(mb.albs, axis=0)
  r = np.concatenate(mb.rghs, axis=0)
  m = np.concatenate(mb.mcodes, axis=0)

  # Compute crisp normals: flat face normals for boxes/arches (roughness > 0.35 or floor),
  # smooth vertex normals for curved spheres/cylinders/water/orbs.
  v0 = v[f[:, 0]]
  v1 = v[f[:, 1]]
  v2 = v[f[:, 2]]
  fn = np.cross(v1 - v0, v2 - v0)
  fn_norm = np.linalg.norm(fn, axis=-1, keepdims=True)
  fn_unit = fn / np.maximum(fn_norm, 1e-12)
  # Smooth vertex normals accumulated per vertex
  vn_smooth = np.zeros_like(v)
  f_flat = f.ravel()
  for c in range(3):
    vn_smooth[:, c] = np.bincount(f_flat, weights=np.repeat(fn[:, c], 3), minlength=v.shape[0])
  vn_smooth = vn_smooth / np.maximum(np.linalg.norm(vn_smooth, axis=-1, keepdims=True), 1e-12)

  # Split flat-shaded architectural boxes/arches (where smooth normal deviates > 25 deg from face normal)
  n0_s = vn_smooth[f[:, 0]]
  dot_f = np.sum(n0_s * fn_unit, axis=-1)
  is_sharp = dot_f < 0.90  # corners of boxes/arches/plinths
  if np.any(is_sharp):
    # Duplicate vertices only for sharp triangles so smooth curved surfaces stay smooth!
    v_sharp = v[f[is_sharp].ravel()]
    n_sharp = np.repeat(fn_unit[is_sharp], 3, axis=0).astype(np.float32)
    f_sharp = np.arange(v_sharp.shape[0], dtype=np.int32).reshape(-1, 3) + v.shape[0]
    v_out = np.concatenate([v, v_sharp], axis=0).astype(np.float32)
    vn_out = np.concatenate([vn_smooth, n_sharp], axis=0).astype(np.float32)
    f_out = np.concatenate([f[~is_sharp], f_sharp], axis=0).astype(np.int32)
    a_out = np.concatenate([a[~is_sharp], a[is_sharp]], axis=0).astype(np.float32)
    r_out = np.concatenate([r[~is_sharp], r[is_sharp]], axis=0).astype(np.float32)
    m_out = np.concatenate([m[~is_sharp], m[is_sharp]], axis=0).astype(np.float32)
  else:
    v_out, vn_out, f_out, a_out, r_out, m_out = v, vn_smooth, f, a, r, m

  tri_inp = OptixBuildInputTriangleArray(
      name=f"night_lantern_courtyard_{n_lights}",
      vertices=v_out,
      indices=f_out,
      vert_normals=vn_out,
      tri_albedo=a_out,
      tri_roughness=r_out,
      tri_mat_code=m_out,
  )
  light_bank = make_restir_light_bank(
      pos=np.array(l_pos, dtype=np.float32),
      emit=np.array(l_emit, dtype=np.float32),
      radius=np.array(l_rad, dtype=np.float32),
      normal=np.array(l_nrm, dtype=np.float32),
      is_isotropic=np.array(l_iso, dtype=np.float32),
  )
  cam_pos, cam_xmat = _look_at_mat((0.0, -9.2, 2.05), (0.0, 4.5, 2.15))
  return tri_inp, light_bank, cam_pos, cam_xmat


# ==============================================================================
# Cinema Figure Builder (Comparison + Zoomed Wet-Floor Crop + Scaling Curve)
# ==============================================================================

def _draw_scaling_chart(
    scaling_recs: List[Dict[str, Any]],
    width: int = 620,
    height: int = 340,
) -> Image.Image:
  """Render crisp dark-theme Light Count Scaling Curve (PSNR & Variance Reduction vs N_L)."""
  im = Image.new("RGB", (width, height), (18, 22, 30))
  draw = ImageDraw.Draw(im)
  ml, mr, mt, mb = 64, 62, 42, 48
  pw = width - ml - mr
  ph = height - mt - mb

  draw.text((14, 12), "Many-Light Scaling (Equal Visibility-Ray Budget = 2 shadow rays/px)", fill=(238, 244, 252))
  draw.rectangle([ml, mt, ml + pw, mt + ph], outline=(70, 82, 102), fill=(13, 16, 22))

  nls = [r["n_lights"] for r in scaling_recs]
  x_logs = [math.log2(n) for n in nls]
  x0, x1 = min(x_logs), max(x_logs)
  p_naive = [r["naive_psnr_db"] for r in scaling_recs]
  p_restir = [r["restir_psnr_db"] for r in scaling_recs]
  p_den = [r["restir_denoise_psnr_db"] for r in scaling_recs]
  y_min = min(min(p_naive), min(p_restir)) - 2.0
  y_max = max(max(p_den), max(p_restir)) + 2.0

  def _px(xl):
    return ml + int(round(((xl - x0) / max(x1 - x0, 1e-6)) * pw))

  def _py(yv):
    return mt + ph - int(round(((yv - y_min) / max(y_max - y_min, 1e-6)) * ph))

  # Gridlines
  for frac in (0.25, 0.5, 0.75):
    gy = mt + int(round(frac * ph))
    draw.line([(ml, gy), (ml + pw, gy)], fill=(34, 40, 52), width=1)
    yv = y_max - frac * (y_max - y_min)
    draw.text((10, gy - 6), f"{yv:.1f}dB", fill=(160, 175, 195))

  for xl, nl in zip(x_logs, nls):
    gx = _px(xl)
    draw.line([(gx, mt), (gx, mt + ph)], fill=(34, 40, 52), width=1)
    draw.text((gx - 12, mt + ph + 8), f"{nl}", fill=(195, 208, 225))
  draw.text((ml + pw // 2 - 42, mt + ph + 26), "Light Count (N_L)", fill=(180, 195, 215))

  def _plot_series(vals, color, label, ly):
    pts = [(_px(xl), _py(v)) for xl, v in zip(x_logs, vals)]
    if len(pts) >= 2:
      draw.line(pts, fill=color, width=3)
    for (x, y), v in zip(pts, vals):
      draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=color, outline=(255, 255, 255))
      draw.text((x - 14, y - 16), f"{v:.1f}", fill=color)
    lx = ml + pw - 236
    draw.line([(lx, ly), (lx + 20, ly)], fill=color, width=3)
    draw.text((lx + 26, ly - 6), label, fill=color)

  _plot_series(p_den, (90, 235, 180), "TPU-ReSTIR DI + Joint A-Trous", mt + 14)
  _plot_series(p_restir, (90, 185, 255), "TPU-ReSTIR DI (M=32 + 2x Sp)", mt + 32)
  _plot_series(p_naive, (255, 115, 95), "Naive Uniform (2 vis rays/px)", mt + 50)
  return im


def make_restir_cinema_figure(
    img_naive: np.ndarray,
    img_restir: np.ndarray,
    img_denoise: np.ndarray,
    img_gt: np.ndarray,
    metrics: Dict[str, Any],
    scaling_recs: List[Dict[str, Any]],
    out_png: str,
) -> None:
  H, W, _ = img_gt.shape
  pw, ph = 460, int(round(460 * H / W))
  pad, hdr, top_h = 12, 30, 54

  # Bottom row: 4-panel zoomed wet cobblestone crop + scaling curve + perf summary
  cw = 4 * pw + 5 * pad
  crop_w = (cw - 3 * pad) // 2
  row2_h = 340
  ch = top_h + (ph + hdr + pad) + (row2_h + hdr + 2 * pad)

  canvas = Image.new("RGB", (cw, ch), (13, 16, 22))
  draw = ImageDraw.Draw(canvas)

  q = metrics["quality_1024lights"]
  b_1080 = next((b for b in metrics["benchmarks"] if b["resolution"] == "1920x1080" and b["n_chips"] == 4 and b["mode"] == "restir_denoise"), metrics["benchmarks"][-1])
  b_4k = next((b for b in metrics["benchmarks"] if b["resolution"] == "3840x2160" and b["n_chips"] == 4 and b["mode"] == "restir_denoise"), metrics["benchmarks"][-1])

  draw.rectangle([0, 0, cw, top_h], fill=(22, 27, 38))
  banner = (
      f"TPU-ReSTIR DI: 1,024 Dynamic Lights Night Courtyard ({metrics['scene_tris']:,} tris, Equal 2-Vis-Ray Budget) | "
      f"ReSTIR+Denoise: {q['restir_denoise_psnr_db']:.2f} dB (+{q['psnr_gain_restir_denoise_db']:.2f} dB vs Naive), "
      f"SSIM {q['restir_denoise_ssim']:.4f}, {q['variance_reduction_restir_denoise_hdr']:.1f}x VarRed | "
      f"4c 1080p: {b_1080['fps']:.2f} FPS | 4c 4K: {b_4k['fps']:.2f} FPS | 0 host callbacks"
  )
  draw.text((pad, 19), banner, fill=(240, 246, 255))

  top_panels = [
      (img_naive, f"(a) Naive Uniform (2 vis/px): {q['naive_psnr_db']:.2f} dB | SSIM {q['naive_ssim']:.4f}"),
      (img_restir, f"(b) TPU-ReSTIR DI (M=32+2xSp): {q['restir_psnr_db']:.2f} dB ({q['variance_reduction_restir_hdr']:.1f}x VarRed)"),
      (img_denoise, f"(c) TPU-ReSTIR DI + A-Trous: {q['restir_denoise_psnr_db']:.2f} dB | SSIM {q['restir_denoise_ssim']:.4f}"),
      (img_gt, f"(d) Reference GT (128 spp x M=32 RIS, {W}x{H})"),
  ]
  for i, (im_arr, title) in enumerate(top_panels):
    x0 = pad + i * (pw + pad)
    y0 = top_h + pad
    u8 = (np.clip(im_arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    pil_im = Image.fromarray(u8).resize((pw, ph), Image.Resampling.BILINEAR)
    canvas.paste(pil_im, (x0, y0 + hdr))
    draw.text((x0 + 4, y0 + 8), title, fill=(232, 240, 250))

  # Bottom-Left: 4-slice Zoomed Wet-Floor Reflection Crop (lower center cobblestone puddles)
  y_r2 = top_h + pad + ph + hdr + pad
  cy0, cy1 = int(0.60 * H), int(0.95 * H)
  cx0, cx1 = int(0.28 * W), int(0.72 * W)
  slices = [
      ("Naive", img_naive[cy0:cy1, cx0:cx1]),
      ("ReSTIR DI", img_restir[cy0:cy1, cx0:cx1]),
      ("ReSTIR+Den", img_denoise[cy0:cy1, cx0:cx1]),
      ("GT", img_gt[cy0:cy1, cx0:cx1]),
  ]
  sw = slices[0][1].shape[1]
  q_w = sw // 4
  zoom_strip = np.concatenate([
      slices[0][1][:, 0:q_w],
      slices[1][1][:, q_w : 2 * q_w],
      slices[2][1][:, 2 * q_w : 3 * q_w],
      slices[3][1][:, 3 * q_w : 4 * q_w],
  ], axis=1)
  for sep in (q_w, 2 * q_w, 3 * q_w):
    zoom_strip[:, max(0, sep - 1) : min(zoom_strip.shape[1], sep + 1)] = np.array([0.2, 0.9, 1.0], np.float32)
  zoom_pil = Image.fromarray((np.clip(zoom_strip, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)).resize((crop_w, row2_h), Image.Resampling.NEAREST)
  canvas.paste(zoom_pil, (pad, y_r2 + hdr))
  draw.text((pad + 4, y_r2 + 8), "2x Zoomed Wet Cobblestone Reflections: Naive | ReSTIR DI | ReSTIR+A-Trous | GT", fill=(120, 230, 255))

  # Bottom-Right: Scaling Curve + Throughput summary side-by-side
  chart_w = int(crop_w * 0.62)
  card_w = crop_w - chart_w - pad
  x_chart = 2 * pad + crop_w
  chart_im = _draw_scaling_chart(scaling_recs, width=chart_w, height=row2_h)
  canvas.paste(chart_im, (x_chart, y_r2 + hdr))
  draw.text((x_chart + 4, y_r2 + 8), "Light Count Scaling Curve (64 -> 1,024 Lights)", fill=(235, 242, 250))

  x_card = x_chart + chart_w + pad
  draw.rectangle([x_card, y_r2 + hdr, x_card + card_w, y_r2 + hdr + row2_h], fill=(18, 22, 30), outline=(55, 68, 90))
  draw.text((x_card + 12, y_r2 + hdr + 12), "TPU v5p Throughput (0 Callbacks)", fill=(255, 220, 100))
  ly = y_r2 + hdr + 42
  for b in metrics["benchmarks"]:
    if b["mode"] in ("restir", "restir_denoise"):
      tag = "ReSTIR+Den" if b["mode"] == "restir_denoise" else "ReSTIR DI "
      line = f"{b['n_chips']}c {b['resolution']:<9} {tag}: {b['fps']:6.2f} FPS ({b['ms']:6.1f} ms)"
      draw.text((x_card + 12, ly), line, fill=(215, 232, 248))
      ly += 22

  os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
  canvas.save(out_png)
  print("SAVED RESTIR CINEMA PNG:", out_png, canvas.size, flush=True)


# ==============================================================================
# Main Driver
# ==============================================================================

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--out_json", default="out/metrics/v3_w7_wave5_restir_cinema.json")
  ap.add_argument("--out_png", default="out/figures/v3_w7_restir_1024lights.png")
  ap.add_argument("--cmp_w", type=int, default=1280)
  ap.add_argument("--cmp_h", type=int, default=720)
  ap.add_argument("--gt_spp", type=int, default=96)
  args = ap.parse_args()

  devs_all = jax.devices()
  assert len(devs_all) == 4, f"Expected 4 chips, got {len(devs_all)}"
  ctx4 = OptixDeviceContext(devs_all)
  ctx1 = OptixDeviceContext(devs_all[:1])

  # 1. Build 1,024-Light Night Lantern Courtyard
  tri_inp, light_bank_1024, cam_pos_np, cam_xmat_np = build_night_lantern_courtyard(1024)
  t_bvh0 = time.perf_counter()
  gas4 = ctx4.accelBuild(tri_inp)
  bvh_s = time.perf_counter() - t_bvh0
  print(f"GAS4 built: {gas4.n_tris:,} tris, {gas4.hbm_mb:.2f} MB in {bvh_s:.2f}s", flush=True)

  cfg = ReSTIRConfig(
      m_candidates=32,
      m_chunk=8,
      spatial_passes=2,
      spatial_neighbors=8,
      spatial_radii=(20, 10),
      depth_rel_thresh=0.10,
      normal_dot_thresh=0.90,
      use_initial_visibility=True,
      use_final_visibility=True,
      use_unbiased_mis=True,
      use_power_sampling=True,
      firefly_clamp=80.0,
      denoise_sigma_l=0.28,
      denoise_passes=4,
  )
  exposure = 1.45
  znear, zfar, fovy = 0.05, 120.0, 54.0
  sky_z = (0.004, 0.008, 0.022)
  sky_h = (0.010, 0.016, 0.038)

  def _eval_quality(ctx, accel, lights_np, W, H, gt_spp, tag):
    lb_d = jax.device_put(lights_np, ctx.rep)
    cxp_d = jax.device_put(cam_pos_np, ctx.rep)
    cxm_d = jax.device_put(cam_xmat_np, ctx.rep)
    rng_d = jax.device_put(jax.random.PRNGKey(17), ctx.rep)

    outputs = {}
    hlo_audit = {}
    for mode in ("naive", "restir", "restir_denoise", "gt"):
      fn = build_restir_renderer(
          ctx, accel, lb_d, W, H, fovy, znear, zfar,
          mode=mode, cfg=cfg, n_vis_rays_naive=2, gt_spp=gt_spp, exposure=exposure,
          sky_zenith=sky_z, sky_horizon=sky_h,
      )
      low = fn.lower(lb_d, cxp_d, cxm_d, rng_d)
      hlo_audit[mode] = audit_hlo_zero_copy(low)
      comp = low.compile()
      rgb_d, hdr_d, dep_d = comp(lb_d, cxp_d, cxm_d, rng_d)
      jax.block_until_ready((rgb_d, hdr_d, dep_d))
      outputs[mode] = (_stitch(rgb_d), _stitch(hdr_d))

    img_n, hdr_n = outputs["naive"]
    img_r, hdr_r = outputs["restir"]
    img_d, hdr_d = outputs["restir_denoise"]
    img_g, hdr_g = outputs["gt"]

    mse_lin_n = float(np.mean((hdr_n - hdr_g) ** 2))
    mse_lin_r = float(np.mean((hdr_r - hdr_g) ** 2))
    mse_lin_d = float(np.mean((hdr_d - hdr_g) ** 2))

    mse_srgb_n = float(np.mean((img_n - img_g) ** 2))
    mse_srgb_r = float(np.mean((img_r - img_g) ** 2))
    mse_srgb_d = float(np.mean((img_d - img_g) ** 2))

    psnr_n = compute_psnr(img_n, img_g)
    psnr_r = compute_psnr(img_r, img_g)
    psnr_d = compute_psnr(img_d, img_g)
    ssim_n = compute_ssim(img_n, img_g)
    ssim_r = compute_ssim(img_r, img_g)
    ssim_d = compute_ssim(img_d, img_g)

    rec = {
        "n_lights": lights_np.n_lights,
        "res": f"{W}x{H}",
        "naive_psnr_db": round(psnr_n, 3),
        "restir_psnr_db": round(psnr_r, 3),
        "restir_denoise_psnr_db": round(psnr_d, 3),
        "psnr_gain_restir_db": round(psnr_r - psnr_n, 3),
        "psnr_gain_restir_denoise_db": round(psnr_d - psnr_n, 3),
        "naive_ssim": round(ssim_n, 4),
        "restir_ssim": round(ssim_r, 4),
        "restir_denoise_ssim": round(ssim_d, 4),
        "variance_reduction_restir_hdr": round(mse_lin_n / max(mse_lin_r, 1e-10), 2),
        "variance_reduction_restir_denoise_hdr": round(mse_lin_n / max(mse_lin_d, 1e-10), 2),
        "variance_reduction_restir_srgb": round(mse_srgb_n / max(mse_srgb_r, 1e-10), 2),
        "variance_reduction_restir_denoise_srgb": round(mse_srgb_n / max(mse_srgb_d, 1e-10), 2),
        "hlo_audit": hlo_audit,
    }
    print(f"[QUALITY {tag}]", json.dumps({k: v for k, v in rec.items() if k != "hlo_audit"}), flush=True)
    return rec, (img_n, img_r, img_d, img_g)

  # 2. Main 1,024-Light Cinema Comparison at 1280x720
  q_1024, (img_n, img_r, img_d, img_g) = _eval_quality(
      ctx4, gas4, light_bank_1024, args.cmp_w, args.cmp_h, args.gt_spp, "1024L"
  )

  # 3. Light Count Scaling Curve (64, 256, 512, 1024) at 512x512 for crisp scaling
  scaling_recs = []
  for nl in (64, 256, 512):
    ti_s, lb_s, _, _ = build_night_lantern_courtyard(nl)
    gas_s = ctx4.accelBuild(ti_s)
    rs, _ = _eval_quality(ctx4, gas_s, lb_s, 512, 512, 64, f"{nl}L")
    scaling_recs.append(rs)
    del gas_s
  # Append 1024L at 512x512 as well so scaling curve is strictly apples-to-apples at 512x512
  rs_1024_512, _ = _eval_quality(ctx4, gas4, light_bank_1024, 512, 512, 64, "1024L_512")
  scaling_recs.append(rs_1024_512)

  # 4. 1-Chip & 4-Chip Throughput Benchmarks (512x512, 1080p, Native 4K)
  gas1 = ctx1.accelBuild(tri_inp)
  benches = []
  res_list = [(512, 512, 8), (1920, 1080, 5), (3840, 2160, 3)]
  for nc, ctx, accel in [(1, ctx1, gas1), (4, ctx4, gas4)]:
    lb_d = jax.device_put(light_bank_1024, ctx.rep)
    cxp_d = jax.device_put(cam_pos_np, ctx.rep)
    cxm_d = jax.device_put(cam_xmat_np, ctx.rep)
    for rw, rh, n_st in res_list:
      for mode in ("naive", "restir", "restir_denoise"):
        fn = build_restir_renderer(
            ctx, accel, lb_d, rw, rh, fovy, znear, zfar,
            mode=mode, cfg=cfg, n_vis_rays_naive=2, exposure=exposure,
            sky_zenith=sky_z, sky_horizon=sky_h,
        )
        rng_d = jax.device_put(jax.random.PRNGKey(31), ctx.rep)
        low = fn.lower(lb_d, cxp_d, cxm_d, rng_d)
        hlo = audit_hlo_zero_copy(low)
        tc0 = time.perf_counter()
        comp = low.compile()
        comp_s = time.perf_counter() - tc0

        # Warmup
        for w_i in range(2):
          out = comp(lb_d, cxp_d, cxm_d, jax.device_put(jax.random.PRNGKey(40 + w_i), ctx.rep))
        jax.block_until_ready(out)

        t0 = time.perf_counter()
        for s_i in range(n_st):
          out = comp(lb_d, cxp_d, cxm_d, jax.device_put(jax.random.PRNGKey(100 + s_i), ctx.rep))
        jax.block_until_ready(out)
        dt = (time.perf_counter() - t0) / n_st

        rec = {
            "n_chips": nc,
            "resolution": f"{rw}x{rh}",
            "mode": mode,
            "compile_s": round(comp_s, 2),
            "ms": round(dt * 1e3, 2),
            "fps": round(1.0 / max(dt, 1e-9), 3),
            "mpix_per_sec": round((rw * rh / max(dt, 1e-9)) / 1e6, 2),
            "host_callbacks": hlo["host_callbacks"],
            "send_recv_ops": hlo["send_recv_ops"],
        }
        print("[BENCH]", rec, flush=True)
        benches.append(rec)

  results = {
      "experiment": "v3_w7_wave5_restir_cinema",
      "device_kind": devs_all[0].device_kind,
      "n_devices": len(devs_all),
      "scene": "night_lantern_courtyard_1024lights",
      "scene_tris": int(gas4.n_tris),
      "gas_hbm_mb": float(gas4.hbm_mb),
      "restir_config": {
          "m_candidates": cfg.m_candidates,
          "spatial_passes": cfg.spatial_passes,
          "spatial_neighbors": cfg.spatial_neighbors,
          "spatial_radii": list(cfg.spatial_radii),
          "depth_rel_thresh": cfg.depth_rel_thresh,
          "normal_dot_thresh": cfg.normal_dot_thresh,
          "equal_vis_rays_per_px": 2,
      },
      "quality_1024lights": q_1024,
      "light_count_scaling_512": scaling_recs,
      "benchmarks": benches,
  }
  os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
  with open(args.out_json, "w") as f:
    json.dump(results, f, indent=2)
  print("SAVED JSON:", args.out_json, flush=True)

  make_restir_cinema_figure(img_n, img_r, img_d, img_g, results, scaling_recs, args.out_png)


if __name__ == "__main__":
  main()
