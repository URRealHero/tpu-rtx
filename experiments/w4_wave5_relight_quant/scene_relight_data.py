"""Dynamic Time-of-Day & Material-Conditioned 4-Bounce MC GT + G-Buffer Generator.

Builds relightable versions of `classroom` and `cornell_workshop` with:
  1. Per-triangle `hero_flag` encoded cleanly in `mat_code` (+10.0 offset) so hero
     materials (Matte Clay -> Satin -> Brushed Gold -> Chrome) can be swapped in <1 ms
     in `leaf_bank` without rebuilding the BVH.
  2. Runtime-parameterized `sun_dir`, `sun_emit`, `sky_zenith`, `sky_horizon`,
     `area_emit`, and `leaf_bank` in the 4-bounce MC path tracer (single XLA compile).
  3. Primary G-buffer extraction:
     [albedo(3), normal(3), depth_norm(1), roughness(1), metallic(1),
      world_pos(3), view_dir(3), hero_mask(1)] -> (H, W, 16).
"""

from __future__ import annotations

import dataclasses
import inspect
import math
import os
from typing import Any, Dict, List, Sequence, Tuple

import jax
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np

from bvh_pbr_pt import (
    LightRig,
    PBRWideBVHMesh,
    _agx_tonemap_srgb,
    _build_onb,
    _eval_pbr_brdf,
    _sample_cosine_hemisphere,
    _sample_ggx_half,
    _smith_g1,
    build_pbr_wide_bvh,
    pbr_raycast_tpu,
)
from complex_scenes_pt import MeshBuilder, _look_at_mat, _subdivide_simple


# ==============================================================================
# 1. Continuous Time-of-Day & Material Presets
# ==============================================================================

TOD_PRESETS: List[Dict[str, Any]] = [
    {
        "name": "Morning",
        "az_deg": 158.0,
        "el_deg": 32.0,
        "temp_k": 4300.0,
        "sun_scale": 0.90,
        "area_scale": 0.65,
        "sky_scale": 1.05,
    },
    {
        "name": "Noon",
        "az_deg": 184.0,
        "el_deg": 54.0,
        "temp_k": 6200.0,
        "sun_scale": 1.15,
        "area_scale": 0.45,
        "sky_scale": 1.25,
    },
    {
        "name": "Golden Sunset",
        "az_deg": 202.0,
        "el_deg": 15.0,
        "temp_k": 2600.0,
        "sun_scale": 1.25,
        "area_scale": 0.85,
        "sky_scale": 0.85,
    },
    {
        "name": "Blue Twilight",
        "az_deg": 212.0,
        "el_deg": 2.0,
        "temp_k": 9500.0,
        "sun_scale": 0.10,
        "area_scale": 1.85,
        "sky_scale": 0.70,
    },
]

MATERIAL_PRESETS: List[Dict[str, Any]] = [
    {
        "name": "Matte Clay",
        "roughness": 0.88,
        "metallic": 0.00,
        "albedo": (0.76, 0.44, 0.34),
    },
    {
        "name": "Satin",
        "roughness": 0.32,
        "metallic": 0.15,
        "albedo": (0.86, 0.84, 0.80),
    },
    {
        "name": "Brushed Gold",
        "roughness": 0.22,
        "metallic": 0.94,
        "albedo": (0.96, 0.74, 0.26),
    },
    {
        "name": "Chrome",
        "roughness": 0.03,
        "metallic": 1.00,
        "albedo": (0.94, 0.96, 0.98),
    },
]


def kelvin_to_rgb(temp_k: float) -> np.ndarray:
  """Approximate blackbody color temperature [1800K..12000K] -> linear-ish RGB [0..1]."""
  t = float(np.clip(temp_k, 1800.0, 12000.0)) / 100.0
  if t <= 66.0:
    r = 255.0
    g = 99.4708025861 * math.log(max(t, 1.0)) - 161.1195681661
    b = 0.0 if t <= 19.0 else 138.5177312231 * math.log(max(t - 10.0, 1.0)) - 305.0447927307
  else:
    r = 329.698727446 * ((t - 60.0) ** -0.1332047592)
    g = 288.1221695283 * ((t - 60.0) ** -0.0755148492)
    b = 255.0
  rgb = np.clip(np.array([r, g, b], dtype=np.float32) / 255.0, 0.05, 1.0)
  return rgb / max(float(np.max(rgb)), 1e-6)


def make_lighting_params(
    scene_name: str,
    az_deg: float,
    el_deg: float,
    temp_k: float,
    sun_scale: float = 1.0,
    area_scale: float = 1.0,
    sky_scale: float = 1.0,
) -> Dict[str, np.ndarray]:
  """Convert continuous (az, el, temp_k, scales) into physical light rig arrays."""
  az = math.radians(float(az_deg))
  el = math.radians(float(el_deg))
  sun_dir = np.array(
      [math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)],
      dtype=np.float32,
  )
  sun_dir = sun_dir / max(float(np.linalg.norm(sun_dir)), 1e-12)

  tint = kelvin_to_rgb(temp_k)
  temp_norm = np.float32((np.clip(temp_k, 2000.0, 10000.0) - 2000.0) / 8000.0)
  coolness = float(temp_norm)

  if scene_name == "classroom":
    sun_emit = (26.0 * sun_scale * tint).astype(np.float32)
    # Sky shifts from warm golden-violet at sunset to crisp blue at noon / deep cobalt at twilight
    warm_sky = np.array([0.42, 0.24, 0.22], dtype=np.float32)
    day_sky = np.array([0.26, 0.48, 0.88], dtype=np.float32)
    dusk_sky = np.array([0.12, 0.24, 0.68], dtype=np.float32)
    if coolness < 0.5:
      w = coolness * 2.0
      sky_z = ((1.0 - w) * warm_sky + w * day_sky) * sky_scale
    else:
      w = (coolness - 0.5) * 2.0
      sky_z = ((1.0 - w) * day_sky + w * dusk_sky) * sky_scale
    sky_h = (0.55 * sky_z + 0.45 * (0.28 * tint + 0.08)).astype(np.float32)
    area_emit = (np.array([7.5, 6.7, 5.5], dtype=np.float32) * area_scale).astype(np.float32)
    has_sun = True
  else:  # cornell_workshop
    # Overhead softbox modulated subtly by color temperature + optional front-portal key
    base_area = np.array([12.5, 11.6, 10.4], dtype=np.float32)
    area_tint = 0.45 * tint + 0.55 * np.array([1.0, 0.96, 0.90], dtype=np.float32)
    area_emit = (base_area * area_tint * area_scale).astype(np.float32)
    sun_emit = (4.5 * sun_scale * tint).astype(np.float32)
    sky_z = (np.array([0.14, 0.16, 0.22], dtype=np.float32) * sky_scale).astype(np.float32)
    sky_h = (np.array([0.10, 0.11, 0.14], dtype=np.float32) * sky_scale).astype(np.float32)
    has_sun = True

  return {
      "sun_dir": sun_dir.astype(np.float32),
      "sun_emit": sun_emit.astype(np.float32),
      "sky_zenith": sky_z.astype(np.float32),
      "sky_horizon": sky_h.astype(np.float32),
      "area_emit": area_emit.astype(np.float32),
      "az_deg": np.float32(az_deg),
      "el_deg": np.float32(el_deg),
      "temp_k": np.float32(temp_k),
      "temp_norm": temp_norm,
      "has_sun": np.bool_(has_sun),
  }


# ==============================================================================
# 2. Relightable BVH Builders (Hero Mask encoded in mat_code +10.0)
# ==============================================================================

def _build_classroom_relightable() -> Tuple[PBRWideBVHMesh, np.ndarray, np.ndarray, float, float, float, LightRig]:
  mb = MeshBuilder()
  n_px, n_py = 12, 12
  for ix in range(n_px):
    for iy in range(n_py):
      x0 = -2.6 + (ix + 0.5) * (5.2 / n_px)
      y0 = -2.4 + (iy + 0.5) * (5.2 / n_py)
      tint = 0.85 + 0.25 * (((ix * 3 + iy * 7) % 5) / 4.0)
      mb.add_box((x0, y0, -0.04), (5.2 / n_px * 0.495, 5.2 / n_py * 0.495, 0.04),
                 (0.46 * tint, 0.28 * tint, 0.14 * tint), roughness=0.38, mat_code=0.0)

  wall_col = (0.82, 0.78, 0.72)
  mb.add_box((0.0, 2.84, 1.5), (2.7, 0.04, 1.55), wall_col, roughness=0.78, mat_code=0.0)
  mb.add_box((0.0, -2.44, 1.5), (2.7, 0.04, 1.55), wall_col, roughness=0.78, mat_code=0.0)
  mb.add_box((2.64, 0.2, 1.5), (0.04, 2.65, 1.55), wall_col, roughness=0.78, mat_code=0.0)
  mb.add_box((0.0, 0.2, 3.04), (2.7, 2.7, 0.04), (0.86, 0.84, 0.80), roughness=0.82, mat_code=0.0)
  mb.add_box((0.1, 0.4, 2.98), (0.85, 0.45, 0.02), (1.0, 0.94, 0.84), roughness=0.2, mat_code=3.4)

  # West slatted window
  mb.add_box((-2.64, 0.2, 0.375), (0.06, 2.7, 0.375), (0.72, 0.66, 0.58), roughness=0.72, mat_code=0.0)
  mb.add_box((-2.64, 0.2, 2.825), (0.06, 2.7, 0.175), wall_col, roughness=0.78, mat_code=0.0)
  mb.add_box((-2.64, -2.12, 1.7), (0.06, 0.32, 0.95), wall_col, roughness=0.78, mat_code=0.0)
  mb.add_box((-2.64, 2.52, 1.7), (0.06, 0.32, 0.95), wall_col, roughness=0.78, mat_code=0.0)
  for my in [-0.5, 0.85]:
    mb.add_box((-2.60, my, 1.7), (0.05, 0.04, 0.96), (0.28, 0.20, 0.14), roughness=0.45, mat_code=0.0)
  for s_i in range(18):
    sz = 0.80 + s_i * (1.80 / 17.0)
    mb.add_box((-2.56, 0.2, sz), (0.055, 1.98, 0.008), (0.76, 0.72, 0.65),
               roughness=0.45, mat_code=0.0, rot_euler=(0.0, 0.46, 0.0))

  # Chalkboard
  mb.add_box((0.0, 2.77, 1.55), (1.45, 0.025, 0.62), (0.36, 0.22, 0.11), roughness=0.42, mat_code=0.0)
  mb.add_box((0.0, 2.75, 1.55), (1.36, 0.02, 0.54), (0.10, 0.23, 0.17), roughness=0.78, mat_code=0.0)
  mb.add_box((0.0, 2.70, 0.94), (1.36, 0.06, 0.02), (0.36, 0.22, 0.11), roughness=0.42, mat_code=0.0)

  # Teacher desk + HERO globe (tagged mat_code += 10.0) + lamp
  mb.add_box((1.1, 1.75, 0.74), (0.62, 0.36, 0.025), (0.65, 0.45, 0.25), roughness=0.32, mat_code=0.0)
  for lx in [0.54, 1.66]:
    for ly in [1.44, 2.06]:
      mb.add_cylinder((lx, ly, 0.0), 0.022, 0.72, (0.22, 0.23, 0.25), roughness=0.28, mat_code=0.85, n_seg=16)
  # Hero globe on teacher desk
  mb.add_sphere((0.78, 1.72, 0.92), 0.17, (0.22, 0.50, 0.78), roughness=0.22, mat_code=10.1, n_lat=22, n_lon=40)
  mb.add_cylinder((1.38, 1.76, 0.765), 0.08, 0.04, (0.85, 0.68, 0.28), roughness=0.18, mat_code=0.92, n_seg=20)
  mb.add_sphere((1.38, 1.76, 0.92), 0.09, (0.95, 0.85, 0.65), roughness=0.15, mat_code=3.2, n_lat=14, n_lon=24)

  # 6 Student desks & chairs + colorful books + metallic cups
  book_cols = [(0.78, 0.22, 0.18), (0.18, 0.48, 0.74), (0.24, 0.62, 0.36), (0.82, 0.62, 0.18), (0.64, 0.28, 0.66), (0.22, 0.64, 0.68)]
  idx = 0
  for row_y in [0.55, -0.65]:
    for col_x in [-1.25, 0.0, 1.25]:
      mb.add_box((col_x, row_y, 0.70), (0.42, 0.28, 0.02), (0.72, 0.54, 0.34), roughness=0.36, mat_code=0.0)
      for dx in [-0.36, 0.36]:
        for dy in [-0.22, 0.22]:
          mb.add_cylinder((col_x + dx, row_y + dy, 0.0), 0.018, 0.68, (0.25, 0.26, 0.28), roughness=0.25, mat_code=0.85, n_seg=14)
      cy = row_y - 0.44
      mb.add_box((col_x, cy, 0.43), (0.21, 0.20, 0.02), (0.58, 0.40, 0.24), roughness=0.42, mat_code=0.0)
      mb.add_box((col_x, cy - 0.19, 0.66), (0.21, 0.02, 0.21), (0.58, 0.40, 0.24), roughness=0.42, mat_code=0.0)
      for dx in [-0.17, 0.17]:
        for dy in [-0.17, 0.17]:
          mb.add_cylinder((col_x + dx, cy + dy, 0.0), 0.014, 0.41, (0.25, 0.26, 0.28), roughness=0.25, mat_code=0.85, n_seg=12)
      mb.add_box((col_x - 0.10, row_y + 0.03, 0.735), (0.12, 0.09, 0.015), book_cols[idx % len(book_cols)], roughness=0.55, mat_code=0.0, rot_euler=(0, 0, 0.15 * idx))
      # Foreground center desk hero orb for extra close-up material visibility in classroom
      if idx == 1:
        mb.add_sphere((col_x + 0.18, row_y + 0.04, 0.84), 0.12, (0.76, 0.44, 0.34), roughness=0.40, mat_code=10.1, n_lat=22, n_lon=40)
      else:
        mb.add_cylinder((col_x + 0.22, row_y + 0.12, 0.72), 0.032, 0.09, (0.82, 0.78, 0.72), roughness=0.18, mat_code=0.90, n_seg=16)
      idx += 1

  bvh = mb.build("classroom_relightable")
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


def _build_cornell_relightable() -> Tuple[PBRWideBVHMesh, np.ndarray, np.ndarray, float, float, float, LightRig]:
  mb = MeshBuilder()
  ivory = (0.78, 0.76, 0.72)
  mb.add_box((0.0, -0.05, -0.04), (1.30, 1.35, 0.04), ivory, roughness=0.72, mat_code=0.0)
  mb.add_box((0.0, -0.05, 2.29), (1.30, 1.35, 0.04), ivory, roughness=0.78, mat_code=0.0)
  mb.add_box((0.0, 1.29, 1.125), (1.30, 0.04, 1.17), ivory, roughness=0.75, mat_code=0.0)
  mb.add_box((-1.29, -0.05, 1.125), (0.04, 1.35, 1.17), (0.80, 0.14, 0.12), roughness=0.72, mat_code=0.0)
  mb.add_box((1.29, -0.05, 1.125), (0.04, 1.35, 1.17), (0.12, 0.60, 0.78), roughness=0.72, mat_code=0.0)
  mb.add_box((0.0, 0.05, 2.23), (0.48, 0.42, 0.015), (1.0, 0.94, 0.86), roughness=0.2, mat_code=3.8)

  # Wooden workbench + slate pedestal
  mb.add_box((-0.28, 0.35, 0.56), (0.68, 0.44, 0.03), (0.56, 0.36, 0.18), roughness=0.35, mat_code=0.0)
  for lx in [-0.88, 0.32]:
    for ly in [-0.03, 0.73]:
      mb.add_box((lx, ly, 0.27), (0.035, 0.035, 0.27), (0.38, 0.24, 0.12), roughness=0.45, mat_code=0.0)
  mb.add_box((0.72, 0.45, 0.34), (0.34, 0.34, 0.34), (0.28, 0.30, 0.34), roughness=0.52, mat_code=0.0, rot_euler=(0, 0, -0.22))

  # Anchor 1: Polished Chrome Mirror Sphere
  mb.add_sphere((-0.58, 0.38, 0.85), 0.26, (0.94, 0.95, 0.97), roughness=0.03, mat_code=0.98, n_lat=28, n_lon=52)
  # Anchor 2: Dielectric Clear Glass Sphere (IOR 1.50)
  mb.add_sphere((0.36, -0.22, 0.25), 0.25, (0.96, 0.99, 0.98), roughness=0.02, mat_code=2.0, n_lat=28, n_lon=52)

  _repo_asset = os.path.join(
      os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
      "out/assets/blender_real_scene.npz",
  )
  _asset_path = _repo_asset if os.path.exists(_repo_asset) else "/dev/shm/zaychik-tpu-mjx/out/assets/blender_real_scene.npz"
  ds = np.load(_asset_path)
  # Anchor 3: Gold Torus on pedestal
  tor_v, tor_f = _subdivide_simple(ds["mesh_1_verts"], ds["mesh_1_faces"], 4)
  tor_v = tor_v * 1.45
  ang_x = math.radians(68.0)
  Rx = np.array([[1, 0, 0], [0, math.cos(ang_x), -math.sin(ang_x)], [0, math.sin(ang_x), math.cos(ang_x)]], np.float32)
  tor_v = (tor_v @ Rx.T) + np.array([0.72, 0.42, 0.92], np.float32)
  mb.add_mesh(tor_v, tor_f, (0.95, 0.74, 0.28), roughness=0.18, mat_code=0.95)

  # HERO 1: Subdivided Suzanne on workbench (tagged mat_code += 10.0 for live material sweep!)
  suz_v, suz_f = _subdivide_simple(ds["mesh_0_verts"], ds["mesh_0_faces"], 4)
  suz_v = suz_v * 1.55
  az = math.radians(-22.0)
  Rz = np.array([[math.cos(az), -math.sin(az), 0], [math.sin(az), math.cos(az), 0], [0, 0, 1]], np.float32)
  suz_v = (suz_v @ Rz.T) + np.array([0.02, 0.36, 0.79], np.float32)
  mb.add_mesh(suz_v, suz_f, (0.86, 0.84, 0.80), roughness=0.26, mat_code=10.05)

  # HERO 2: Foreground sphere (also tagged mat_code += 10.0 so both sculpted & smooth curvature show material!)
  mb.add_sphere((-0.24, -0.38, 0.18), 0.18, (0.86, 0.84, 0.80), roughness=0.26, mat_code=10.05, n_lat=24, n_lon=44)

  # Static workshop clutter
  mb.add_cylinder((-0.68, -0.32, 0.0), 0.14, 0.42, (0.88, 0.48, 0.26), roughness=0.14, mat_code=0.92, n_seg=32)
  mb.add_box((-0.18, 0.08, 0.67), (0.08, 0.08, 0.08), (0.95, 0.98, 1.0), roughness=0.02, mat_code=2.0, rot_euler=(0, 0, 0.4))
  mb.add_box((0.20, 0.12, 0.65), (0.06, 0.06, 0.06), (0.90, 0.32, 0.22), roughness=0.35, mat_code=0.1, rot_euler=(0, 0, -0.3))
  mb.add_box((0.76, -0.36, 0.10), (0.10, 0.10, 0.10), (0.86, 0.78, 0.36), roughness=0.24, mat_code=0.85, rot_euler=(0, 0, 0.5))

  bvh = mb.build("cornell_relightable")
  cxp, cxm = _look_at_mat((0.0, -2.55, 1.08), (0.0, 0.15, 0.86))
  rig = LightRig(
      area_center=(0.0, 0.05, 2.18),
      area_normal=(0.0, 0.0, -1.0),
      area_radius=0.45,
      area_emit=(12.5, 11.6, 10.4),
      has_sun=True,
      sun_dir=(-0.25, -0.90, 0.36),  # Front-portal subtle studio key light
      sun_cos_half_angle=0.996,
      sun_emit=(4.5, 4.2, 3.8),
      sky_zenith=(0.14, 0.16, 0.22),
      sky_horizon=(0.10, 0.11, 0.14),
      exposure=0.95,
  )
  return bvh, cxp, cxm, 44.0, 0.05, 20.0, rig


def build_relight_scene(scene_name: str):
  if scene_name == "classroom":
    return _build_classroom_relightable()
  if scene_name == "cornell_workshop":
    return _build_cornell_relightable()
  raise ValueError(f"Unknown scene: {scene_name}")


def override_hero_leaf_bank(
    base_leaf_bank: np.ndarray,
    roughness: float,
    metallic: float,
    albedo_rgb: Tuple[float, float, float],
) -> np.ndarray:
  """Swap hero triangles' (albedo, roughness, mat_code) in <1 ms with zero BVH rebuild."""
  lb = base_leaf_bank.copy()
  hero_mask = lb[:, 23, :] >= 9.0  # (N_leaves, LEAF_SIZE)
  alb = np.asarray(albedo_rgb, dtype=np.float32)
  for c in range(3):
    lb[:, 19 + c, :] = np.where(hero_mask, alb[c], lb[:, 19 + c, :])
  lb[:, 22, :] = np.where(hero_mask, float(np.clip(roughness, 0.02, 0.99)), lb[:, 22, :])
  lb[:, 23, :] = np.where(hero_mask, 10.0 + float(np.clip(metallic, 0.0, 1.0)), lb[:, 23, :])
  return lb


# ==============================================================================
# 3. Single-Compile Dynamic 4-Bounce MC Path Tracer + 16-ch Primary G-Buffer
# ==============================================================================

def build_dynamic_pt_and_gbuf_fn(
    bvh_mesh: PBRWideBVHMesh,
    devices: Sequence[jax.Device],
    height: int,
    width: int,
    cam_fovy: float,
    znear: float,
    zfar: float,
    base_rig: LightRig,
    spp: int,
    max_bounces: int = 4,
):
  """Compile 4-chip sharded 4-bounce MC PT + 16-ch G-Buffer with runtime light & leaf_bank."""
  n_chips = len(devices)
  assert width % n_chips == 0 and (width // n_chips) % 16 == 0 and height % 8 == 0
  W_local = width // n_chips

  mesh = Mesh(np.array(devices), ("dp",))
  rep = NamedSharding(mesh, P())
  sh_dp = NamedSharding(mesh, P("dp"))

  l2_roots_d = jax.device_put(bvh_mesh.l2_roots_aabb, rep)
  l2_d = jax.device_put(bvh_mesh.l2_aabbs, rep)
  l0_d = jax.device_put(bvh_mesh.l0_aabbs, rep)
  chip_ids = jax.device_put(np.arange(n_chips, dtype=np.int32), sh_dp)

  area_c = jnp.asarray(base_rig.area_center, dtype=jnp.float32)
  area_n_raw = jnp.asarray(base_rig.area_normal, dtype=jnp.float32)
  area_n = area_n_raw / jnp.maximum(jnp.linalg.norm(area_n_raw), 1e-12)
  at1, at2 = _build_onb(area_n)
  area_r = float(base_rig.area_radius)
  area_area = math.pi * area_r * area_r
  sky_g = jnp.asarray(base_rig.sky_ground, dtype=jnp.float32)
  sun_sin_max = math.sqrt(max(0.0, 1.0 - base_rig.sun_cos_half_angle ** 2))
  exposure = float(base_rig.exposure)

  aspect = float(width) / float(height)
  half_h = float(znear) * math.tan(0.5 * float(cam_fovy) * (math.pi / 180.0))
  half_w = half_h * aspect

  def _trace_core(l2r, l2, l0, lf, cxp, cxm, sun_dir_in, sun_emit_in, sky_z_in, sky_h_in, area_emit_in, cid, rng_in):
    x_off = cid * W_local
    chip_rng = jax.random.fold_in(rng_in, cid)

    sun_d = sun_dir_in / jnp.maximum(jnp.linalg.norm(sun_dir_in), 1e-12)
    st1, st2 = _build_onb(sun_d)

    def _env_sky(rd):
      z = rd[..., 2]
      tu = jnp.clip(z, 0.0, 1.0)[..., None]
      td = jnp.clip(-z, 0.0, 1.0)[..., None]
      s = jnp.where(z[..., None] >= 0.0, (1.0 - tu) * sky_h_in + tu * sky_z_in, (1.0 - td) * sky_h_in + td * sky_g)
      cos_s = jnp.maximum(jnp.sum(rd * sun_d, axis=-1), 0.0)
      s = s + (cos_s ** 32.0)[..., None] * sun_emit_in * 0.25
      return s

    local_px = height * W_local
    if spp >= 16 and local_px <= 32768:
      spp_par = 16
    elif spp >= 8 and local_px <= 131072:
      spp_par = 8
    elif spp >= 4 and local_px <= 262144:
      spp_par = 4
    else:
      spp_par = 1
    spp_seq = spp // spp_par
    R = height * W_local

    # Center-pixel primary G-buffer (16 channels)
    px_c = jnp.arange(W_local, dtype=jnp.float32) + jnp.asarray(x_off, jnp.float32) + 0.5
    py_c = jnp.arange(height, dtype=jnp.float32) + 0.5
    gxc, gyc = jnp.meshgrid(px_c, py_c, indexing="xy")
    uc = gxc.reshape(1, R) / float(width)
    vc = gyc.reshape(1, R) / float(height)
    xc = -half_w + (2.0 * half_w) * uc
    yc = half_h - (2.0 * half_h) * vc
    raw_c = jnp.stack([xc, yc, jnp.full_like(xc, -float(znear))], axis=-1)
    ld_c = raw_c / jnp.maximum(jnp.linalg.norm(raw_c, axis=-1, keepdims=True), 1e-12)
    neg_zc = jnp.maximum(-ld_c[..., 2], 1e-6)
    rd_c = jnp.einsum("ij,brj->bri", cxm[0], ld_c, precision=jax.lax.Precision.HIGHEST)
    rd_c = rd_c / jnp.maximum(jnp.linalg.norm(rd_c, axis=-1, keepdims=True), 1e-12)
    ro_c = cxp[0][None, None, :] + (float(znear) / neg_zc)[..., None] * rd_c
    tmax_c = jnp.broadcast_to((float(zfar - znear) / neg_zc), (1, R))

    d_p, n_p, alb_p, rgh_p, mc_p_raw = pbr_raycast_tpu(
        l2r, l2, l0, lf, ro_c, rd_c, tmax_c, height, W_local, any_hit=False
    )
    hit_p = (d_p < tmax_c - 1e-4) & (d_p >= 0.0)
    is_hero_p = hit_p & (mc_p_raw >= 9.0)
    mc_p = jnp.where(mc_p_raw >= 9.0, mc_p_raw - 10.0, mc_p_raw)
    n_p_u = n_p / jnp.maximum(jnp.linalg.norm(n_p, axis=-1, keepdims=True), 1e-12)
    n_p_ff = jnp.where((jnp.sum(n_p_u * (-rd_c), axis=-1) >= 0.0)[..., None], n_p_u, -n_p_u)
    wpos_p = ro_c + jnp.where(hit_p, d_p, 0.0)[..., None] * rd_c
    sky_alb = jnp.clip(_agx_tonemap_srgb(_env_sky(rd_c), exposure=exposure), 0.0, 1.0)
    gbuf_flat = jnp.concatenate(
        [
            jnp.where(hit_p[..., None], alb_p, sky_alb),                       # 0:3 albedo
            jnp.where(hit_p[..., None], n_p_ff, 0.0),                          # 3:6 normal
            jnp.where(hit_p, jnp.clip(d_p / 10.0, 0.0, 1.5), -1.0)[..., None], # 6:7 depth_norm
            jnp.where(hit_p, rgh_p, 1.0)[..., None],                           # 7:8 roughness
            jnp.where(hit_p, jnp.clip(mc_p * 0.5, 0.0, 2.0), 0.0)[..., None],  # 8:9 mcode_norm
            jnp.where(hit_p[..., None], wpos_p, 0.0),                          # 9:12 world_pos
            -rd_c,                                                             # 12:15 view_dir (wo)
            jnp.where(is_hero_p, 1.0, 0.0)[..., None],                         # 15:16 hero_mask
        ],
        axis=-1,
    )
    gbuf_hw = gbuf_flat[0].reshape(height, W_local, 16)

    def _trace_batch_samples(s_idx, rad_acc):
      s_rng = jax.random.fold_in(chip_rng, s_idx)
      k_jit, k_bnc = jax.random.split(s_rng)
      jitter = jax.random.uniform(k_jit, (spp_par, R, 2), minval=-0.5, maxval=0.5, dtype=jnp.float32)
      u = (gxc.reshape(1, R) + jitter[..., 0]) / float(width)
      v = (gyc.reshape(1, R) + jitter[..., 1]) / float(height)
      x = -half_w + (2.0 * half_w) * u
      y = half_h - (2.0 * half_h) * v
      raw = jnp.stack([x, y, jnp.full_like(x, -float(znear))], axis=-1)
      ld = raw / jnp.maximum(jnp.linalg.norm(raw, axis=-1, keepdims=True), 1e-12)
      neg_z = jnp.maximum(-ld[..., 2], 1e-6)
      rd0 = jnp.einsum("ij,brj->bri", cxm[0], ld, precision=jax.lax.Precision.HIGHEST)
      rd0 = rd0 / jnp.maximum(jnp.linalg.norm(rd0, axis=-1, keepdims=True), 1e-12)
      ro0 = cxp[0][None, None, :] + (float(znear) / neg_z)[..., None] * rd0
      t_max0 = jnp.broadcast_to((float(zfar - znear) / neg_z), (spp_par, R))
      thr0 = jnp.ones((spp_par, R, 3), dtype=jnp.float32)
      alive0 = jnp.ones((spp_par, R), dtype=jnp.bool_)
      rad0 = jnp.zeros((spp_par, R, 3), dtype=jnp.float32)

      def _bounce(b_idx, carry):
        ro, rd, t_lim, thr, alive, rad = carry
        b_rng = jax.random.fold_in(k_bnc, b_idx)
        k_lu, k_lv, k_su, k_sv, k_lobe, k_u1, k_u2 = jax.random.split(b_rng, 7)

        d_hit, nrm_raw, alb, rgh, mcode_raw = pbr_raycast_tpu(
            l2r, l2, l0, lf, ro, rd, jnp.where(alive, t_lim, 0.0), height, W_local, any_hit=False
        )
        mcode = jnp.where(mcode_raw >= 9.0, mcode_raw - 10.0, mcode_raw)
        hit = alive & (d_hit < t_lim - 1e-4) & (d_hit >= 0.0)
        miss = alive & (~hit)
        rad = rad + jnp.where(miss[..., None], thr * _env_sky(rd), 0.0)

        n_unit = nrm_raw / jnp.maximum(jnp.linalg.norm(nrm_raw, axis=-1, keepdims=True), 1e-12)
        wo = -rd
        entering = jnp.sum(n_unit * wo, axis=-1) >= 0.0
        n_ff = jnp.where(entering[..., None], n_unit, -n_unit)

        is_emit = hit & (mcode >= 2.8)
        is_glass = hit & (mcode >= 1.8) & (mcode < 2.8)
        is_opaque = hit & (mcode < 1.8)
        metallic = jnp.clip(mcode, 0.0, 1.0)

        emit_power = jnp.where(mcode >= 2.8, (mcode - 2.0) * 6.0, 0.0)
        # Scale emissive ceiling panels gently with area_emit relative luminance
        area_rel = jnp.mean(area_emit_in) / 9.0
        rad = rad + jnp.where(is_emit[..., None], thr * alb * emit_power[..., None] * jnp.clip(area_rel, 0.5, 1.8), 0.0)

        hit_pos = ro + jnp.where(hit, d_hit, 0.0)[..., None] * rd

        lu = jax.random.uniform(k_lu, (spp_par, R), minval=0.0, maxval=1.0, dtype=jnp.float32)
        lv = jax.random.uniform(k_lv, (spp_par, R), minval=0.0, maxval=1.0, dtype=jnp.float32)
        lr = area_r * jnp.sqrt(lu)
        lphi = (2.0 * math.pi) * lv
        lp = area_c[None, None, :] + (lr * jnp.cos(lphi))[..., None] * at1[None, None, :] + (lr * jnp.sin(lphi))[..., None] * at2[None, None, :]
        to_l = lp - hit_pos
        dist_l = jnp.linalg.norm(to_l, axis=-1)
        wi_l = to_l / jnp.maximum(dist_l[..., None], 1e-12)
        cos_s = jnp.maximum(jnp.sum(n_ff * wi_l, axis=-1), 0.0)
        cos_l = jnp.maximum(jnp.sum(area_n[None, None, :] * (-wi_l), axis=-1), 0.0)
        need_sh = is_opaque & (cos_s > 0.0) & (cos_l > 0.0)
        sh_o = hit_pos + n_ff * 1.5e-3
        sh_lim = jnp.where(need_sh, jnp.maximum(dist_l - 2.0e-3, 0.0), 0.0)

        d_sh, *_ = pbr_raycast_tpu(l2r, l2, l0, lf, sh_o, wi_l, sh_lim, height, W_local, any_hit=True)
        vis = jnp.where(need_sh & (d_sh >= sh_lim - 1e-4), 1.0, 0.0)
        brdf_a = _eval_pbr_brdf(n_ff, wo, wi_l, alb, metallic, rgh)
        geom_a = (cos_s * cos_l) / (dist_l * dist_l + 0.08) * area_area
        nee_a = jnp.minimum(thr * brdf_a * area_emit_in[None, None, :] * (geom_a * vis)[..., None], 14.0)
        rad = rad + jnp.where(is_opaque[..., None], nee_a, 0.0)

        # Sun directional NEE
        su = jax.random.uniform(k_su, (spp_par, R), minval=0.0, maxval=1.0, dtype=jnp.float32)
        sv = jax.random.uniform(k_sv, (spp_par, R), minval=0.0, maxval=1.0, dtype=jnp.float32)
        sr = sun_sin_max * jnp.sqrt(su)
        sphi = (2.0 * math.pi) * sv
        wi_sun = sun_d[None, None, :] + (sr * jnp.cos(sphi))[..., None] * st1[None, None, :] + (sr * jnp.sin(sphi))[..., None] * st2[None, None, :]
        wi_sun = wi_sun / jnp.maximum(jnp.linalg.norm(wi_sun, axis=-1, keepdims=True), 1e-12)
        cos_sun = jnp.maximum(jnp.sum(n_ff * wi_sun, axis=-1), 0.0)
        need_sun = is_opaque & (cos_sun > 0.0) & (jnp.max(sun_emit_in) > 1e-4)
        sun_lim = jnp.where(need_sun, jnp.float32(zfar * 0.9), 0.0)
        d_sun, *_ = pbr_raycast_tpu(l2r, l2, l0, lf, sh_o, wi_sun, sun_lim, height, W_local, any_hit=True)
        vis_sun = jnp.where(need_sun & (d_sun >= sun_lim - 1e-2), 1.0, 0.0)
        brdf_sun = _eval_pbr_brdf(n_ff, wo, wi_sun, alb, metallic, rgh)
        nee_sun = jnp.minimum(thr * brdf_sun * sun_emit_in[None, None, :] * (cos_sun * vis_sun)[..., None], 18.0)
        rad = rad + jnp.where(is_opaque[..., None], nee_sun, 0.0)

        u_lobe = jax.random.uniform(k_lobe, (spp_par, R), minval=0.0, maxval=1.0, dtype=jnp.float32)
        u1 = jax.random.uniform(k_u1, (spp_par, R), minval=0.0, maxval=1.0, dtype=jnp.float32)
        u2 = jax.random.uniform(k_u2, (spp_par, R), minval=0.0, maxval=1.0, dtype=jnp.float32)

        p_spec = jnp.clip(0.15 + 0.78 * metallic, 0.15, 0.90)
        choose_spec = u_lobe < p_spec
        wi_diff = _sample_cosine_hemisphere(u1, u2, n_ff)
        alpha = jnp.maximum(rgh * rgh, 0.02)
        h_ggx = _sample_ggx_half(u1, u2, n_ff, alpha)
        wi_spec = 2.0 * jnp.sum(wo * h_ggx, axis=-1, keepdims=True) * h_ggx - wo
        wi_spec = wi_spec / jnp.maximum(jnp.linalg.norm(wi_spec, axis=-1, keepdims=True), 1e-12)
        wi_opq = jnp.where(choose_spec[..., None], wi_spec, wi_diff)

        ndotl_s = jnp.maximum(jnp.sum(n_ff * wi_spec, axis=-1), 0.0)
        ndotv = jnp.maximum(jnp.sum(n_ff * wo, axis=-1), 1e-4)
        ndoth = jnp.maximum(jnp.sum(n_ff * h_ggx, axis=-1), 1e-4)
        vdoth = jnp.maximum(jnp.sum(wo * h_ggx, axis=-1), 1e-4)
        f0 = 0.04 * (1.0 - metallic[..., None]) + alb * metallic[..., None]
        F_s = f0 + (1.0 - f0) * ((1.0 - vdoth) ** 5)[..., None]
        G_s = _smith_g1(ndotl_s, alpha) * _smith_g1(ndotv, alpha)
        w_spec = jnp.where(ndotl_s[..., None] > 0.0, jnp.minimum(F_s * ((G_s * vdoth) / jnp.maximum(ndotv * ndoth * p_spec, 1e-4))[..., None], 4.0), 0.0)
        w_diff = (1.0 - metallic[..., None]) * alb / jnp.maximum(1.0 - p_spec[..., None], 0.1)
        w_opq = jnp.where(choose_spec[..., None], w_spec, w_diff)

        eta = jnp.where(entering, 1.0 / 1.50, 1.50)
        cos_i = jnp.clip(jnp.sum(wo * n_ff, axis=-1), 0.0, 1.0)
        sin2_t = eta * eta * (1.0 - cos_i * cos_i)
        tir = sin2_t > 1.0
        r0_g = ((1.0 - eta) / (1.0 + eta)) ** 2
        fresnel_g = r0_g + (1.0 - r0_g) * ((1.0 - cos_i) ** 5)
        refl_dir = 2.0 * cos_i[..., None] * n_ff - wo
        cos_t = jnp.sqrt(jnp.maximum(1.0 - sin2_t, 0.0))
        refr_dir = eta[..., None] * rd + (eta * cos_i - cos_t)[..., None] * n_ff
        do_refl = tir | (u_lobe < fresnel_g)
        wi_glass = jnp.where(do_refl[..., None], refl_dir, refr_dir)
        wi_glass = wi_glass / jnp.maximum(jnp.linalg.norm(wi_glass, axis=-1, keepdims=True), 1e-12)
        ro_glass = jnp.where(do_refl[..., None], hit_pos + n_ff * 1.5e-3, hit_pos - n_ff * 1.5e-3)

        wi_next = jnp.where(is_glass[..., None], wi_glass, wi_opq)
        new_ro = jnp.where(is_glass[..., None], ro_glass, hit_pos + n_ff * 1.5e-3)
        bounce_w = jnp.where(is_glass[..., None], alb, w_opq)

        new_thr = thr * bounce_w
        new_alive = (is_opaque | is_glass) & (jnp.max(new_thr, axis=-1) > 1e-3)
        new_t_lim = jnp.broadcast_to(jnp.float32(zfar), (spp_par, R))
        return (new_ro, wi_next, new_t_lim, new_thr, new_alive, rad)

      _, _, _, _, _, final_rad = jax.lax.fori_loop(0, max_bounces, _bounce, (ro0, rd0, t_max0, thr0, alive0, rad0))
      return rad_acc + jnp.mean(final_rad, axis=0)

    rad_sum = jax.lax.fori_loop(0, spp_seq, _trace_batch_samples, jnp.zeros((R, 3), dtype=jnp.float32))
    rad_avg = (rad_sum / float(spp_seq)).reshape(height, W_local, 3)
    rgb = _agx_tonemap_srgb(rad_avg, exposure=exposure)
    return rgb, gbuf_hw

  sm_kw = {"check_rep": False} if "check_rep" in inspect.signature(shard_map).parameters else {"check_vma": False}

  def _render_local(l2r, l2, l0, lf, cxp, cxm, sd, se, sz, sh, ae, cid_arr, rng_in):
    rgb, gbuf_hw = _trace_core(l2r, l2, l0, lf, cxp, cxm, sd, se, sz, sh, ae, cid_arr[0], rng_in)
    return rgb[None, ...], gbuf_hw[None, ...]

  sh_fn = shard_map(
      _render_local,
      mesh=mesh,
      in_specs=(P(), P(), P(), P(), P(), P(), P(), P(), P(), P(), P(), P("dp"), P()),
      out_specs=(P("dp"), P("dp")),
      **sm_kw,
  )

  @jax.jit
  def render_fn(leaf_d, cxp, cxm, sun_dir, sun_emit, sky_z, sky_h, area_emit, rng):
    return sh_fn(l2_roots_d, l2_d, l0_d, leaf_d, cxp, cxm, sun_dir, sun_emit, sky_z, sky_h, area_emit, chip_ids, rng)

  return render_fn, mesh, rep


def stitch_sharded(arr_j: jax.Array) -> np.ndarray:
  arr = np.asarray(jax.device_get(arr_j), dtype=np.float32)
  n_c, H, W_loc, C = arr.shape
  return arr.transpose(1, 0, 2, 3).reshape(H, n_c * W_loc, C)


# ==============================================================================
# 4. Sampling Continuous (Time-of-Day, Material, View) States
# ==============================================================================

def _sample_view(scene_name: str, rng: np.random.Generator, hero: bool = False) -> Tuple[np.ndarray, np.ndarray]:
  if scene_name == "classroom":
    if hero:
      return _look_at_mat((2.15, -2.05, 1.52), (-0.45, 0.85, 0.85))
    pos = (
        float(2.15 + rng.uniform(-0.75, 0.25)),
        float(-2.05 + rng.uniform(-0.20, 0.70)),
        float(1.52 + rng.uniform(-0.28, 0.28)),
    )
    tgt = (
        float(-0.45 + rng.uniform(-0.35, 0.35)),
        float(0.85 + rng.uniform(-0.35, 0.35)),
        float(0.85 + rng.uniform(-0.18, 0.18)),
    )
    return _look_at_mat(pos, tgt)
  else:
    if hero:
      return _look_at_mat((0.0, -2.55, 1.08), (0.0, 0.15, 0.86))
    pos = (
        float(rng.uniform(-0.48, 0.48)),
        float(-2.55 + rng.uniform(-0.15, 0.45)),
        float(1.08 + rng.uniform(-0.20, 0.28)),
    )
    tgt = (
        float(rng.uniform(-0.18, 0.18)),
        float(0.15 + rng.uniform(-0.18, 0.18)),
        float(0.86 + rng.uniform(-0.14, 0.14)),
    )
    return _look_at_mat(pos, tgt)


def _sample_material(rng: np.random.Generator) -> Tuple[float, float, Tuple[float, float, float]]:
  mode = rng.uniform(0.0, 1.0)
  if mode < 0.25:
    # Matte / clay-like
    r = float(rng.uniform(0.55, 0.95))
    m = float(rng.uniform(0.0, 0.08))
    alb = tuple(float(x) for x in rng.uniform(0.25, 0.90, size=3))
  elif mode < 0.55:
    # Satin / ceramic
    r = float(rng.uniform(0.18, 0.48))
    m = float(rng.uniform(0.05, 0.45))
    alb = tuple(float(x) for x in rng.uniform(0.35, 0.92, size=3))
  elif mode < 0.80:
    # Brushed / anodized metal (gold, copper, bronze, tinted steel)
    r = float(rng.uniform(0.10, 0.35))
    m = float(rng.uniform(0.75, 1.0))
    alb = tuple(float(x) for x in rng.uniform(0.30, 0.98, size=3))
  else:
    # Mirror / chrome
    r = float(rng.uniform(0.02, 0.12))
    m = float(rng.uniform(0.90, 1.0))
    alb = tuple(float(x) for x in rng.uniform(0.75, 0.99, size=3))
  return r, m, alb


def pack_cond_vector(
    sun_dir: np.ndarray,
    sun_emit: np.ndarray,
    sky_z: np.ndarray,
    sky_h: np.ndarray,
    area_emit: np.ndarray,
    temp_norm: float,
    mat_rgh: float,
    mat_met: float,
    mat_alb: Tuple[float, float, float],
    scene_id: int,
) -> np.ndarray:
  """Pack normalized 24-D continuous conditioning vector for FiLM & feature builder."""
  return np.concatenate([
      np.asarray(sun_dir, dtype=np.float32).reshape(3),                     # 0:3
      np.asarray(sun_emit, dtype=np.float32).reshape(3) / 30.0,             # 3:6
      np.asarray(sky_z, dtype=np.float32).reshape(3),                       # 6:9
      np.asarray(sky_h, dtype=np.float32).reshape(3),                       # 9:12
      np.asarray(area_emit, dtype=np.float32).reshape(3) / 16.0,            # 12:15
      np.array([float(temp_norm)], dtype=np.float32),                       # 15:16
      np.array([float(mat_rgh), float(mat_met)], dtype=np.float32),         # 16:18
      np.asarray(mat_alb, dtype=np.float32).reshape(3),                     # 18:21
      np.array([1.0 if scene_id == 0 else 0.0, 1.0 if scene_id == 1 else 0.0, 0.0], dtype=np.float32),  # 21:24
  ], axis=0).astype(np.float32)


def generate_relight_dataset(
    devs_4c: Sequence[jax.Device],
    out_npz: str,
    train_res: int = 256,
    test_res: int = 512,
    gt_spp: int = 256,
    n_train_per_scene: int = 36,
    n_test_rand_per_scene: int = 4,
    max_bounces: int = 4,
) -> str:
  """Render multi-state (ToD + Material + View) 4-bounce MC GT & primary G-Buffers."""
  import os
  import time

  tr_gb_l, tr_gt_l, tr_cond_l = [], [], []
  te_records: Dict[str, Dict[str, Any]] = {}

  rng_np = np.random.default_rng(20261122)

  for scene_id, scene_name in enumerate(["classroom", "cornell_workshop"]):
    bvh, _, _, fovy, zn, zf, base_rig = build_relight_scene(scene_name)
    base_lb = bvh.leaf_bank

    # Compile 1x train_res and 1x test_res
    fn_tr, _, rep = build_dynamic_pt_and_gbuf_fn(
        bvh, devs_4c, train_res, train_res, fovy, zn, zf, base_rig, spp=gt_spp, max_bounces=max_bounces
    )
    fn_te, _, rep_te = build_dynamic_pt_and_gbuf_fn(
        bvh, devs_4c, test_res, test_res, fovy, zn, zf, base_rig, spp=gt_spp, max_bounces=max_bounces
    )

    def _run_state(fn, rep_sh, az, el, temp_k, s_sc, a_sc, k_sc, rgh, met, alb, cxp, cxm, seed):
      lp = make_lighting_params(scene_name, az, el, temp_k, s_sc, a_sc, k_sc)
      lb = override_hero_leaf_bank(base_lb, rgh, met, alb)
      rgb_j, gb_j = fn(
          jax.device_put(lb, rep_sh),
          jax.device_put(cxp, rep_sh),
          jax.device_put(cxm, rep_sh),
          jax.device_put(lp["sun_dir"], rep_sh),
          jax.device_put(lp["sun_emit"], rep_sh),
          jax.device_put(lp["sky_zenith"], rep_sh),
          jax.device_put(lp["sky_horizon"], rep_sh),
          jax.device_put(lp["area_emit"], rep_sh),
          jax.device_put(jax.random.PRNGKey(seed), rep_sh),
      )
      jax.block_until_ready((rgb_j, gb_j))
      cond = pack_cond_vector(
          lp["sun_dir"], lp["sun_emit"], lp["sky_zenith"], lp["sky_horizon"],
          lp["area_emit"], float(lp["temp_norm"]), rgh, met, alb, scene_id
      )
      return stitch_sharded(gb_j), stitch_sharded(rgb_j), cond

    t0 = time.perf_counter()
    for i in range(n_train_per_scene):
      # Mix hero camera (45%) & random views (55%) so hero camera relighting + view gen are both sharp
      cxp, cxm = _sample_view(scene_name, rng_np, hero=(i < int(n_train_per_scene * 0.45)))
      if scene_name == "classroom":
        az = float(rng_np.uniform(152.0, 215.0))
        el = float(rng_np.uniform(1.0, 60.0))
        temp_k = float(rng_np.uniform(2400.0, 9800.0))
        # Realistic twilight attenuation when el < 8 deg
        s_sc = float(rng_np.uniform(0.08, 0.30)) if el < 8.0 else float(rng_np.uniform(0.75, 1.35))
        a_sc = float(rng_np.uniform(1.3, 2.0)) if el < 8.0 else float(rng_np.uniform(0.4, 1.0))
        k_sc = float(rng_np.uniform(0.65, 1.30))
      else:
        az = float(rng_np.uniform(-118.0, -62.0))
        el = float(rng_np.uniform(15.0, 52.0))
        temp_k = float(rng_np.uniform(2600.0, 9500.0))
        s_sc = float(rng_np.uniform(0.6, 1.4))
        a_sc = float(rng_np.uniform(0.7, 1.4))
        k_sc = float(rng_np.uniform(0.7, 1.3))
      rgh, met, alb = _sample_material(rng_np)
      gb_s, gt_s, cond_s = _run_state(
          fn_tr, rep, az, el, temp_k, s_sc, a_sc, k_sc, rgh, met, alb, cxp, cxm, 10000 + scene_id * 1000 + i
      )
      tr_gb_l.append(gb_s)
      tr_gt_l.append(gt_s)
      tr_cond_l.append(cond_s)
    print(f"[DATA] {scene_name}: rendered {n_train_per_scene} train states @ {train_res} in {time.perf_counter()-t0:.1f}s", flush=True)

    # Test set: 4 canonical Time-of-Day presets + 4 canonical Material presets + random unseen states @ test_res
    t1 = time.perf_counter()
    cxp_h, cxm_h = _sample_view(scene_name, rng_np, hero=True)
    tod_gbs, tod_gts, tod_conds, tod_names = [], [], [], []
    for p_idx, tod in enumerate(TOD_PRESETS):
      def_mat = MATERIAL_PRESETS[1]  # Satin default
      az = tod["az_deg"] if scene_name == "classroom" else (-90.0 + (p_idx - 1.5) * 14.0)
      el = tod["el_deg"] if scene_name == "classroom" else (34.0 - p_idx * 5.0)
      gb_s, gt_s, cond_s = _run_state(
          fn_te, rep_te, az, el, tod["temp_k"], tod["sun_scale"], tod["area_scale"], tod["sky_scale"],
          def_mat["roughness"], def_mat["metallic"], def_mat["albedo"],
          cxp_h, cxm_h, 20000 + scene_id * 1000 + p_idx
      )
      tod_gbs.append(gb_s)
      tod_gts.append(gt_s)
      tod_conds.append(cond_s)
      tod_names.append(tod["name"])

    mat_gbs, mat_gts, mat_conds, mat_names = [], [], [], []
    for m_idx, mat in enumerate(MATERIAL_PRESETS):
      # Use Noon/clear studio lighting for material sweep
      az = 178.0 if scene_name == "classroom" else -95.0
      el = 42.0 if scene_name == "classroom" else 34.0
      gb_s, gt_s, cond_s = _run_state(
          fn_te, rep_te, az, el, 5600.0, 1.05, 1.0, 1.05,
          mat["roughness"], mat["metallic"], mat["albedo"],
          cxp_h, cxm_h, 30000 + scene_id * 1000 + m_idx
      )
      mat_gbs.append(gb_s)
      mat_gts.append(gt_s)
      mat_conds.append(cond_s)
      mat_names.append(mat["name"])

    rand_gbs, rand_gts, rand_conds = [], [], []
    for r_idx in range(n_test_rand_per_scene):
      cxp_r, cxm_r = _sample_view(scene_name, rng_np, hero=False)
      az = float(rng_np.uniform(155.0, 210.0)) if scene_name == "classroom" else float(rng_np.uniform(-115.0, -65.0))
      el = float(rng_np.uniform(6.0, 56.0)) if scene_name == "classroom" else float(rng_np.uniform(18.0, 48.0))
      temp_k = float(rng_np.uniform(2600.0, 9200.0))
      s_sc = 0.25 if el < 8.0 else float(rng_np.uniform(0.8, 1.25))
      a_sc = 1.6 if el < 8.0 else float(rng_np.uniform(0.5, 1.1))
      rgh, met, alb = _sample_material(rng_np)
      gb_s, gt_s, cond_s = _run_state(
          fn_te, rep_te, az, el, temp_k, s_sc, a_sc, 1.0,
          rgh, met, alb, cxp_r, cxm_r, 40000 + scene_id * 1000 + r_idx
      )
      rand_gbs.append(gb_s)
      rand_gts.append(gt_s)
      rand_conds.append(cond_s)

    print(f"[DATA] {scene_name}: rendered {4+4+n_test_rand_per_scene} test states @ {test_res} in {time.perf_counter()-t1:.1f}s", flush=True)
    te_records[scene_name] = {
        "tod_gb": np.stack(tod_gbs, axis=0),
        "tod_gt": np.stack(tod_gts, axis=0),
        "tod_cond": np.stack(tod_conds, axis=0),
        "tod_names": np.array(tod_names),
        "mat_gb": np.stack(mat_gbs, axis=0),
        "mat_gt": np.stack(mat_gts, axis=0),
        "mat_cond": np.stack(mat_conds, axis=0),
        "mat_names": np.array(mat_names),
        "rand_gb": np.stack(rand_gbs, axis=0),
        "rand_gt": np.stack(rand_gts, axis=0),
        "rand_cond": np.stack(rand_conds, axis=0),
    }

  os.makedirs(os.path.dirname(out_npz) or ".", exist_ok=True)
  save_dict = {
      "tr_gb": np.stack(tr_gb_l, axis=0),
      "tr_gt": np.stack(tr_gt_l, axis=0),
      "tr_cond": np.stack(tr_cond_l, axis=0),
  }
  for sname, sd in te_records.items():
    for k, v in sd.items():
      save_dict[f"{sname}_{k}"] = v
  np.savez(out_npz, **save_dict)
  print(f"[DATA] Saved complete dataset -> {out_npz}", flush=True)
  return out_npz
