"""Build & export a genuine 1,000,000+ triangle UV-textured PBR Blender 5.2 gallery diorama.

Creates a 6-asset cinematic gallery scene totaling ~1.23M triangles:
  1. suzanne_hero      (~252k tris) - Subsurf-4 Suzanne in hammered bronze + patina PBR
  2. torus_knot        (~197k tris) - (2,3) trefoil knot in glazed cobalt-teal ceramic
  3. wave_sculpture    (~261k tris) - Gyroid-rippled orb in veined terracotta marble
  4. gallery_colonnade (~265k tris) - 6 fluted rotunda columns + architrave in travertine
  5. gallery_floor     (~205k tris) - 320x320 tessellated Calacatta/Nero marble floor
  6. pedestal          (~49k tris)  - Lathe-turned dark walnut display pedestal

Generates 512x512 sRGB albedo & roughness textures shared identically between
Blender 5.2 Cycles (48/64 spp) and the TPU v5p raytracer, exports clean indexed
geometry + vertex normals + vertex UVs + 48 dynamic 256x256 frames (36 train /
4 calib / 8 test) + 4 held-out 512x512 hero showcase views + Cycles EXR Z-depth.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
import time
from typing import Dict, List, Tuple

import bmesh
import bpy
from mathutils import Matrix, Vector
import numpy as np


def _parse_args():
  argv = sys.argv
  argv = argv[argv.index("--") + 1 :] if "--" in argv else []
  ap = argparse.ArgumentParser()
  ap.add_argument("--out_dir", default="out/assets")
  ap.add_argument("--n_frames", type=int, default=48)
  ap.add_argument("--n_hero", type=int, default=4)
  ap.add_argument("--res", type=int, default=256)
  ap.add_argument("--hero_res", type=int, default=512)
  ap.add_argument("--samples", type=int, default=48)
  ap.add_argument("--hero_samples", type=int, default=64)
  ap.add_argument("--seed", type=int, default=2026)
  return ap.parse_args(argv)


def _euler_zyx(yaw: float, pitch: float, roll: float) -> np.ndarray:
  cz, sz = math.cos(yaw), math.sin(yaw)
  cy, sy = math.cos(pitch), math.sin(pitch)
  cx, sx = math.cos(roll), math.sin(roll)
  rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], np.float32)
  ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], np.float32)
  rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], np.float32)
  return (rz @ ry @ rx).astype(np.float32)


def _look_at_cam_mat(cam_pos: np.ndarray, target: np.ndarray) -> np.ndarray:
  """Return camera matrix whose columns are (right, up, -forward)."""
  fwd = target - cam_pos
  fwd = fwd / max(np.linalg.norm(fwd), 1e-12)
  up_ref = np.array([0.0, 0.0, 1.0], dtype=np.float32)
  right = np.cross(fwd, up_ref)
  right = right / max(np.linalg.norm(right), 1e-12)
  up = np.cross(right, fwd)
  return np.stack([right, up, -fwd], axis=1).astype(np.float32)


def _set_obj_pose(obj, pos: np.ndarray, mat3: np.ndarray):
  obj.matrix_world = Matrix((
      (float(mat3[0, 0]), float(mat3[0, 1]), float(mat3[0, 2]), float(pos[0])),
      (float(mat3[1, 0]), float(mat3[1, 1]), float(mat3[1, 2]), float(pos[1])),
      (float(mat3[2, 0]), float(mat3[2, 1]), float(mat3[2, 2]), float(pos[2])),
      (0.0, 0.0, 0.0, 1.0),
  ))


# ---------------------------------------------------------------------------
# Procedural 512x512 tileable PBR textures (bottom-up rows for exact Blender UV)
# ---------------------------------------------------------------------------

def _fbm_tileable(H: int, W: int, seed: int, octaves: int = 5, base_freq: int = 4) -> np.ndarray:
  """Fast periodic multi-octave trigonometric FBM in [0, 1] on a [0, 2pi) torus."""
  rng = np.random.default_rng(seed)
  y = np.linspace(0.0, 2.0 * math.pi, H, endpoint=False, dtype=np.float32)[:, None]
  x = np.linspace(0.0, 2.0 * math.pi, W, endpoint=False, dtype=np.float32)[None, :]
  acc = np.zeros((H, W), dtype=np.float32)
  amp = 0.5
  norm = 0.0
  for o in range(octaves):
    freq = base_freq * (2 ** o)
    oct_val = np.zeros((H, W), dtype=np.float32)
    for _ in range(4):
      kx = int(rng.integers(1, max(2, freq)))
      ky = int(rng.integers(1, max(2, freq)))
      sx = float(rng.choice([-1.0, 1.0]))
      ph = float(rng.uniform(0.0, 2.0 * math.pi))
      oct_val += np.sin(sx * kx * x + ky * y + ph)
    acc += amp * (oct_val * 0.25)
    norm += amp
    amp *= 0.5
  out = 0.5 + 0.5 * (acc / max(norm, 1e-6))
  return np.clip(out, 0.0, 1.0).astype(np.float32)


def make_pbr_textures(size: int = 512) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
  """Generate 512x512 (albedo_srgb, roughness) pairs for all 6 assets."""
  H = W = size
  u = np.linspace(0.0, 1.0, W, endpoint=False, dtype=np.float32)[None, :]
  v = np.linspace(0.0, 1.0, H, endpoint=False, dtype=np.float32)[:, None]

  # 1. Hammered Antique Bronze + Patina
  n1 = _fbm_tileable(H, W, seed=101, octaves=5, base_freq=4)
  n2 = _fbm_tileable(H, W, seed=102, octaves=4, base_freq=10)
  patina = np.clip((n1 - 0.58) * 3.8, 0.0, 1.0)
  bronze_base = np.array([0.80, 0.52, 0.24], np.float32)[None, None, :]
  bronze_hi = np.array([0.92, 0.68, 0.36], np.float32)[None, None, :]
  patina_col = np.array([0.22, 0.52, 0.46], np.float32)[None, None, :]
  alb_suz = (1.0 - n2[..., None]) * bronze_base + n2[..., None] * bronze_hi
  alb_suz = (1.0 - 0.55 * patina[..., None]) * alb_suz + 0.55 * patina[..., None] * patina_col
  rgh_suz = np.clip(0.20 + 0.14 * patina + 0.05 * n2, 0.14, 0.45).astype(np.float32)

  # 2. Glazed Cobalt-Teal Ceramic
  n_cer = _fbm_tileable(H, W, seed=201, octaves=5, base_freq=3)
  swirl = 0.5 + 0.5 * np.sin(2.0 * math.pi * (3.0 * u + 2.0 * v) + 5.0 * n_cer)
  c_deep = np.array([0.06, 0.36, 0.46], np.float32)[None, None, :]
  c_aqua = np.array([0.18, 0.72, 0.78], np.float32)[None, None, :]
  alb_tor = (1.0 - swirl[..., None]) * c_deep + swirl[..., None] * c_aqua
  rgh_tor = np.clip(0.12 + 0.10 * n_cer, 0.08, 0.28).astype(np.float32)

  # 3. Veined Rosso Terracotta Marble
  n_m1 = _fbm_tileable(H, W, seed=301, octaves=5, base_freq=3)
  vein = np.abs(np.sin(2.0 * math.pi * (2.0 * u - 3.0 * v) + 6.5 * n_m1))
  vein_mask = np.clip(1.0 - (vein ** 0.35), 0.0, 1.0)
  m_terra = np.array([0.78, 0.28, 0.19], np.float32)[None, None, :]
  m_dark = np.array([0.52, 0.16, 0.12], np.float32)[None, None, :]
  m_cream = np.array([0.94, 0.84, 0.74], np.float32)[None, None, :]
  alb_wav = (1.0 - n_m1[..., None]) * m_dark + n_m1[..., None] * m_terra
  alb_wav = (1.0 - 0.75 * vein_mask[..., None]) * alb_wav + 0.75 * vein_mask[..., None] * m_cream
  rgh_wav = np.clip(0.18 + 0.12 * n_m1, 0.14, 0.35).astype(np.float32)

  # 4. Warm Travertine Limestone
  n_tr = _fbm_tileable(H, W, seed=401, octaves=5, base_freq=4)
  strata = 0.5 + 0.5 * np.sin(2.0 * math.pi * 6.0 * v + 2.5 * n_tr)
  t_sand = np.array([0.78, 0.73, 0.64], np.float32)[None, None, :]
  t_warm = np.array([0.88, 0.83, 0.75], np.float32)[None, None, :]
  alb_col = (1.0 - strata[..., None]) * t_sand + strata[..., None] * t_warm
  rgh_col = np.clip(0.54 + 0.18 * n_tr, 0.45, 0.78).astype(np.float32)

  # 5. Calacatta & Nero Marquina Marble Checkerboard (4x4 tiles per UV unit)
  n_fl = _fbm_tileable(H, W, seed=501, octaves=5, base_freq=4)
  tx = (u * 4.0) % 1.0
  ty = (v * 4.0) % 1.0
  ix = np.floor(u * 4.0).astype(np.int32)
  iy = np.floor(v * 4.0).astype(np.int32)
  is_light = ((ix + iy) % 2) == 0
  grout = (tx < 0.025) | (tx > 0.975) | (ty < 0.025) | (ty > 0.975)
  vein_fl = np.clip(1.0 - (np.abs(np.sin(2.0 * math.pi * (u * 3.0 + v * 2.0) + 5.5 * n_fl)) ** 0.3), 0.0, 1.0)
  calacatta = (1.0 - 0.22 * vein_fl[..., None]) * np.array([0.88, 0.86, 0.82], np.float32) + 0.22 * vein_fl[..., None] * np.array([0.55, 0.52, 0.48], np.float32)
  nero = (1.0 - 0.35 * vein_fl[..., None]) * np.array([0.14, 0.15, 0.18], np.float32) + 0.35 * vein_fl[..., None] * np.array([0.75, 0.65, 0.45], np.float32)
  alb_fl = np.where(is_light[..., None], calacatta, nero)
  alb_fl = np.where(grout[..., None], np.array([0.55, 0.46, 0.28], np.float32), alb_fl)
  rgh_fl = np.where(grout, 0.42, np.clip(0.16 + 0.10 * n_fl, 0.12, 0.30)).astype(np.float32)

  # 6. Dark Walnut Wood Grain (concentric lathe rings along v)
  n_wd = _fbm_tileable(H, W, seed=601, octaves=4, base_freq=4)
  ring = 0.5 + 0.5 * np.sin(2.0 * math.pi * 8.0 * v + 3.5 * n_wd)
  w_dark = np.array([0.24, 0.13, 0.07], np.float32)[None, None, :]
  w_rich = np.array([0.46, 0.26, 0.14], np.float32)[None, None, :]
  alb_ped = (1.0 - ring[..., None]) * w_dark + ring[..., None] * w_rich
  rgh_ped = np.clip(0.30 + 0.14 * ring, 0.25, 0.48).astype(np.float32)

  return {
      "suzanne_hero": (np.clip(alb_suz, 0.0, 1.0), rgh_suz),
      "torus_knot": (np.clip(alb_tor, 0.0, 1.0), rgh_tor),
      "wave_sculpture": (np.clip(alb_wav, 0.0, 1.0), rgh_wav),
      "gallery_colonnade": (np.clip(alb_col, 0.0, 1.0), rgh_col),
      "gallery_floor": (np.clip(alb_fl, 0.0, 1.0), rgh_fl),
      "pedestal": (np.clip(alb_ped, 0.0, 1.0), rgh_ped),
  }


def _make_textured_material(
    name: str,
    albedo_srgb: np.ndarray,
    roughness: np.ndarray,
    metallic: float,
    uv_scale: float = 1.0,
    tex_dir: str = "/tmp",
):
  """Create Principled BSDF material driven by exact 512x512 sRGB albedo & roughness maps."""
  H, W, _ = albedo_srgb.shape

  alb_path = os.path.join(tex_dir, f"{name}_alb.png")
  img_alb = bpy.data.images.new(f"{name}_alb", width=W, height=H, alpha=True)
  rgba = np.ones((H, W, 4), dtype=np.float32)
  rgba[..., :3] = albedo_srgb
  img_alb.pixels.foreach_set(rgba.reshape(-1))
  img_alb.filepath_raw = alb_path
  img_alb.file_format = "PNG"
  img_alb.save()
  img_alb.reload()
  img_alb.colorspace_settings.name = "sRGB"

  rgh_path = os.path.join(tex_dir, f"{name}_rgh.png")
  img_rgh = bpy.data.images.new(f"{name}_rgh", width=W, height=H, alpha=True)
  r_rgba = np.ones((H, W, 4), dtype=np.float32)
  r_rgba[..., 0] = roughness
  r_rgba[..., 1] = roughness
  r_rgba[..., 2] = roughness
  img_rgh.pixels.foreach_set(r_rgba.reshape(-1))
  img_rgh.filepath_raw = rgh_path
  img_rgh.file_format = "PNG"
  img_rgh.save()
  img_rgh.reload()
  img_rgh.colorspace_settings.name = "Non-Color"

  mat = bpy.data.materials.new(name=name)
  mat.use_nodes = True
  nt = mat.node_tree
  bsdf = nt.nodes.get("Principled BSDF")
  bsdf.inputs["Metallic"].default_value = float(metallic)

  uv_node = nt.nodes.new("ShaderNodeTexCoord")
  map_node = nt.nodes.new("ShaderNodeMapping")
  map_node.inputs["Scale"].default_value = (uv_scale, uv_scale, 1.0)
  nt.links.new(uv_node.outputs["UV"], map_node.inputs["Vector"])

  tex_alb = nt.nodes.new("ShaderNodeTexImage")
  tex_alb.image = img_alb
  tex_alb.extension = "REPEAT"
  nt.links.new(map_node.outputs["Vector"], tex_alb.inputs["Vector"])
  nt.links.new(tex_alb.outputs["Color"], bsdf.inputs["Base Color"])

  tex_rgh = nt.nodes.new("ShaderNodeTexImage")
  tex_rgh.image = img_rgh
  tex_rgh.extension = "REPEAT"
  nt.links.new(map_node.outputs["Vector"], tex_rgh.inputs["Vector"])
  nt.links.new(tex_rgh.outputs["Color"], bsdf.inputs["Roughness"])
  return mat


# ---------------------------------------------------------------------------
# High-Resolution Mesh Builders (Total > 1,000,000 Triangles)
# ---------------------------------------------------------------------------

def _create_mesh_from_grid(
    name: str,
    xyz: np.ndarray,  # (U, V, 3)
    uv: np.ndarray,   # (U, V, 2)
    wrap_u: bool = False,
    wrap_v: bool = False,
) -> bpy.types.Object:
  """Build a fast UV-mapped triangulated grid mesh in Blender."""
  Nu, Nv, _ = xyz.shape
  verts = xyz.reshape(-1, 3).tolist()
  u_max = Nu if wrap_u else Nu - 1
  v_max = Nv if wrap_v else Nv - 1

  i_idx = np.arange(u_max, dtype=np.int32)[:, None]
  j_idx = np.arange(v_max, dtype=np.int32)[None, :]
  i_nxt = (i_idx + 1) % Nu
  j_nxt = (j_idx + 1) % Nv

  v00 = (i_idx * Nv + j_idx).reshape(-1)
  v10 = (i_nxt * Nv + j_idx).reshape(-1)
  v11 = (i_nxt * Nv + j_nxt).reshape(-1)
  v01 = (i_idx * Nv + j_nxt).reshape(-1)

  t1 = np.stack([v00, v10, v11], axis=-1)
  t2 = np.stack([v00, v11, v01], axis=-1)
  tris = np.concatenate([t1, t2], axis=0)

  mesh = bpy.data.meshes.new(f"{name}_mesh")
  mesh.from_pydata(verts, [], tris.tolist())
  uv_layer = mesh.uv_layers.new(name="UVMap")
  uv_flat = uv.reshape(-1, 2)
  # Wait: handle wrap seam UVs cleanly by storing per-loop UV!
  loop_u = np.concatenate([
      np.broadcast_to(i_idx, (u_max, v_max)).reshape(-1),
      np.broadcast_to(i_nxt, (u_max, v_max)).reshape(-1),
      np.broadcast_to(i_nxt, (u_max, v_max)).reshape(-1),
      np.broadcast_to(i_idx, (u_max, v_max)).reshape(-1),
      np.broadcast_to(i_nxt, (u_max, v_max)).reshape(-1),
      np.broadcast_to(i_idx, (u_max, v_max)).reshape(-1),
  ])
  # Compute exact continuous UVs across wrap boundaries
  du = 1.0 / float(Nu - (0 if wrap_u else 1))
  dv = 1.0 / float(Nv - (0 if wrap_v else 1))
  ii = np.broadcast_to(i_idx, (u_max, v_max)).reshape(-1).astype(np.float32)
  jj = np.broadcast_to(j_idx, (u_max, v_max)).reshape(-1).astype(np.float32)

  # Each triangle has 3 loops: t1=(00,10,11), t2=(00,11,01)
  u_t1 = np.stack([ii * du, (ii + 1.0) * du, (ii + 1.0) * du], axis=-1)
  v_t1 = np.stack([jj * dv, jj * dv, (jj + 1.0) * dv], axis=-1)
  u_t2 = np.stack([ii * du, (ii + 1.0) * du, ii * du], axis=-1)
  v_t2 = np.stack([jj * dv, (jj + 1.0) * dv, (jj + 1.0) * dv], axis=-1)
  loop_uvs = np.stack(
      [np.concatenate([u_t1, u_t2], axis=0), np.concatenate([v_t1, v_t2], axis=0)],
      axis=-1,
  ).reshape(-1, 2).astype(np.float32)
  uv_layer.data.foreach_set("uv", loop_uvs.reshape(-1))
  mesh.polygons.foreach_set("use_smooth", np.ones(len(mesh.polygons), dtype=bool))
  mesh.update()

  obj = bpy.data.objects.new(name, mesh)
  bpy.context.collection.objects.link(obj)
  return obj


def _build_suzanne_hero() -> bpy.types.Object:
  bpy.ops.mesh.primitive_monkey_add(size=0.34, location=(0, 0, 0))
  obj = bpy.context.active_object
  obj.name = "suzanne_hero"
  bpy.ops.object.mode_set(mode="EDIT")
  bpy.ops.mesh.select_all(action="SELECT")
  bpy.ops.uv.smart_project(angle_limit=1.15, island_margin=0.01)
  bpy.ops.object.mode_set(mode="OBJECT")
  sub = obj.modifiers.new(name="Subsurf", type="SUBSURF")
  sub.levels = 4
  sub.render_levels = 4
  bpy.ops.object.shade_smooth()
  bpy.context.view_layer.objects.active = obj
  bpy.ops.object.modifier_apply(modifier=sub.name)
  return obj


def _build_torus_knot(Nu: int = 768, Nv: int = 128) -> bpy.types.Object:
  """Parametric (2, 3) trefoil torus knot -> 196,608 triangles."""
  u = np.linspace(0.0, 2.0 * math.pi, Nu, endpoint=False, dtype=np.float64)
  v = np.linspace(0.0, 2.0 * math.pi, Nv, endpoint=False, dtype=np.float64)
  p, q = 2.0, 3.0
  R_maj, R_min, r_tube = 0.135, 0.055, 0.030

  cu, su = np.cos(p * u), np.sin(p * u)
  cq, sq = np.cos(q * u), np.sin(q * u)
  rad = R_maj + R_min * cq
  cx = rad * cu
  cy = rad * su
  cz = R_min * sq

  # Finite-diff tangent for clean Bishop/Frenet frame
  eps = 1e-4
  u2 = u + eps
  rad2 = R_maj + R_min * np.cos(q * u2)
  tx = rad2 * np.cos(p * u2) - cx
  ty = rad2 * np.sin(p * u2) - cy
  tz = R_min * np.sin(q * u2) - cz
  T = np.stack([tx, ty, tz], axis=-1)
  T /= np.maximum(np.linalg.norm(T, axis=-1, keepdims=True), 1e-12)

  # Outward radial normal reference
  up = np.stack([cu, su, np.zeros_like(u)], axis=-1)
  B = np.cross(T, up)
  B /= np.maximum(np.linalg.norm(B, axis=-1, keepdims=True), 1e-12)
  N = np.cross(B, T)

  cv = np.cos(v)[None, :, None]
  sv = np.sin(v)[None, :, None]
  center = np.stack([cx, cy, cz], axis=-1)[:, None, :]
  pts = center + r_tube * (cv * N[:, None, :] + sv * B[:, None, :])
  uv = np.zeros((Nu, Nv, 2), dtype=np.float32)
  return _create_mesh_from_grid("torus_knot", pts.astype(np.float32), uv, wrap_u=True, wrap_v=True)


def _build_wave_sculpture(Nu: int = 512, Nv: int = 256) -> bpy.types.Object:
  """High-res rippled sculptural sphere -> ~261k triangles."""
  theta = np.linspace(0.0, 2.0 * math.pi, Nu, endpoint=False, dtype=np.float64)[:, None]
  phi = np.linspace(0.02, math.pi - 0.02, Nv, endpoint=True, dtype=np.float64)[None, :]
  r = 0.135 * (
      1.0
      + 0.14 * np.sin(6.0 * theta) * np.sin(5.0 * phi)
      + 0.06 * np.cos(10.0 * theta + 3.0 * phi)
  )
  x = r * np.sin(phi) * np.cos(theta)
  y = r * np.sin(phi) * np.sin(theta)
  z = r * np.cos(phi)
  pts = np.stack([x, y, z], axis=-1).astype(np.float32)
  uv = np.zeros((Nu, Nv, 2), dtype=np.float32)
  return _create_mesh_from_grid("wave_sculpture", pts, uv, wrap_u=True, wrap_v=False)


def _build_gallery_colonnade() -> bpy.types.Object:
  """8 fluted architectural columns at R=2.15 m + ring architrave -> ~280k tris."""
  objs: List[bpy.types.Object] = []
  R_col = 2.15
  n_cols = 8
  Nu_c, Nv_c = 128, 96  # 8 * 128 * 95 * 2 = 194,560 tris

  theta = np.linspace(0.0, 2.0 * math.pi, Nu_c, endpoint=False, dtype=np.float64)[:, None]
  z_arr = np.linspace(-0.05, 1.15, Nv_c, dtype=np.float64)[None, :]
  h_norm = (z_arr + 0.05) / 1.20
  # Classical entasis + fluted shaft + stepped base & capital
  flute = 0.012 * np.maximum(0.0, np.cos(20.0 * theta)) * np.sin(math.pi * np.clip((h_norm - 0.08) / 0.84, 0.0, 1.0))
  base_flare = 0.055 * np.exp(-((h_norm / 0.06) ** 2)) + 0.050 * np.exp(-(((1.0 - h_norm) / 0.06) ** 2))
  entasis = 1.0 - 0.10 * h_norm + 0.04 * np.sin(math.pi * h_norm)
  rad = 0.140 * entasis + base_flare - flute

  for k in range(n_cols):
    ang = k * (2.0 * math.pi / n_cols) + 0.18
    cx, cy = R_col * math.cos(ang), R_col * math.sin(ang)
    pts = np.stack([cx + rad * np.cos(theta), cy + rad * np.sin(theta), np.broadcast_to(z_arr, (Nu_c, Nv_c))], axis=-1)
    o = _create_mesh_from_grid(f"col_{k}", pts.astype(np.float32), np.zeros((Nu_c, Nv_c, 2), np.float32), wrap_u=True, wrap_v=False)
    objs.append(o)

  # Ring architrave connecting all 8 columns overhead (z = 1.13 .. 1.35): 512 x 82 = ~83k tris
  Nu_r, Nv_r = 512, 82
  u_r = np.linspace(0.0, 2.0 * math.pi, Nu_r, endpoint=False, dtype=np.float64)[:, None]
  v_r = np.linspace(0.0, 2.0 * math.pi, Nv_r, endpoint=False, dtype=np.float64)[None, :]
  # Stepped rectangular-ish superellipse cross-section
  rx = 0.18 * np.sign(np.cos(v_r)) * (np.abs(np.cos(v_r)) ** 0.45)
  rz = 0.11 * np.sign(np.sin(v_r)) * (np.abs(np.sin(v_r)) ** 0.45)
  rr = R_col + rx
  pts_r = np.stack([rr * np.cos(u_r), rr * np.sin(u_r), np.broadcast_to(1.24 + rz, (Nu_r, Nv_r))], axis=-1)
  o_ring = _create_mesh_from_grid("architrave", pts_r.astype(np.float32), np.zeros((Nu_r, Nv_r, 2), np.float32), wrap_u=True, wrap_v=True)
  objs.append(o_ring)

  # Join all colonnade parts into a single clean mesh
  bpy.ops.object.select_all(action="DESELECT")
  for o in objs:
    o.select_set(True)
  bpy.context.view_layer.objects.active = objs[0]
  bpy.ops.object.join()
  joined = bpy.context.active_object
  joined.name = "gallery_colonnade"
  return joined


def _build_gallery_floor(Nu: int = 321, Nv: int = 321) -> bpy.types.Object:
  """320x320 tessellated marble floor (204,800 tris) spanning [-2.6, 2.6]^2 at z=-0.05."""
  xs = np.linspace(-2.6, 2.6, Nu, dtype=np.float32)
  ys = np.linspace(-2.6, 2.6, Nv, dtype=np.float32)
  gx, gy = np.meshgrid(xs, ys, indexing="ij")
  pts = np.stack([gx, gy, np.full_like(gx, -0.05)], axis=-1)
  return _create_mesh_from_grid("gallery_floor", pts, np.zeros((Nu, Nv, 2), np.float32), wrap_u=False, wrap_v=False)


def _build_pedestal(Nu: int = 256, Nv: int = 97) -> bpy.types.Object:
  """Turned dark walnut gallery pedestal (49,152 tris) from z=-0.05 to z=0.14."""
  theta = np.linspace(0.0, 2.0 * math.pi, Nu, endpoint=False, dtype=np.float64)[:, None]
  s = np.linspace(0.0, 1.0, Nv, dtype=np.float64)[None, :]
  # Profile from bottom outer lip -> fluted waist -> top disk -> center cap
  r_prof = np.zeros_like(s)
  z_prof = np.zeros_like(s)
  for idx, sv in enumerate(s[0]):
    if sv < 0.20:
      t = sv / 0.20
      r_prof[0, idx] = 0.44 - 0.04 * t
      z_prof[0, idx] = -0.05 + 0.04 * t
    elif sv < 0.65:
      t = (sv - 0.20) / 0.45
      r_prof[0, idx] = 0.36 - 0.03 * math.sin(math.pi * t)
      z_prof[0, idx] = -0.01 + 0.12 * t
    elif sv < 0.85:
      t = (sv - 0.65) / 0.20
      r_prof[0, idx] = 0.36 + 0.04 * math.sin(math.pi * 0.5 * t)
      z_prof[0, idx] = 0.11 + 0.03 * t
    else:
      t = (sv - 0.85) / 0.15
      r_prof[0, idx] = 0.40 * (1.0 - t)
      z_prof[0, idx] = 0.14
  pts = np.stack([r_prof * np.cos(theta), r_prof * np.sin(theta), np.broadcast_to(z_prof, (Nu, Nv))], axis=-1)
  return _create_mesh_from_grid("pedestal", pts.astype(np.float32), np.zeros((Nu, Nv, 2), np.float32), wrap_u=True, wrap_v=False)


# ---------------------------------------------------------------------------
# Fast Vectorized Mesh + UV + Smooth Normal Extraction
# ---------------------------------------------------------------------------

def _finalize_and_extract_mesh(obj: bpy.types.Object, uv_scale: float = 1.0):
  """Permanently triangulate, ensure outward smooth normals, and extract deduplicated (V, N, UV, F)."""
  bpy.context.view_layer.objects.active = obj
  bm = bmesh.new()
  bm.from_mesh(obj.data)
  bmesh.ops.triangulate(bm, faces=bm.faces[:])
  bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
  bm.to_mesh(obj.data)
  bm.free()

  mesh = obj.data
  mesh.polygons.foreach_set("use_smooth", np.ones(len(mesh.polygons), dtype=bool))
  mesh.update()
  mesh.calc_loop_triangles()

  n_tris = len(mesh.loop_triangles)
  tri_loops = np.empty(n_tris * 3, dtype=np.int32)
  mesh.loop_triangles.foreach_get("loops", tri_loops)

  loop_verts = np.empty(len(mesh.loops), dtype=np.int32)
  mesh.loops.foreach_get("vertex_index", loop_verts)

  uv_raw = np.empty(len(mesh.loops) * 2, dtype=np.float32)
  mesh.uv_layers.active.data.foreach_get("uv", uv_raw)
  uv_raw = uv_raw.reshape(-1, 2) * np.float32(uv_scale)

  v_co = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
  mesh.vertices.foreach_get("co", v_co)
  v_co = v_co.reshape(-1, 3)

  v_nrm = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
  mesh.vertices.foreach_get("normal", v_nrm)
  v_nrm = v_nrm.reshape(-1, 3)

  c_vidx = loop_verts[tri_loops]
  c_uv = uv_raw[tri_loops]
  # Quantize UV to 1e-4 for clean UV-seam splitting while preserving exact welded geometry
  uq = np.round(c_uv[:, 0] * 10000.0).astype(np.int64)
  vq = np.round(c_uv[:, 1] * 10000.0).astype(np.int64)
  keys = np.stack([c_vidx.astype(np.int64), uq, vq], axis=1)
  _, first_occ, inv = np.unique(keys, axis=0, return_index=True, return_inverse=True)

  v_out = v_co[c_vidx[first_occ]].astype(np.float32)
  n_out = v_nrm[c_vidx[first_occ]].astype(np.float32)
  n_len = np.maximum(np.linalg.norm(n_out, axis=-1, keepdims=True), 1e-12)
  n_out = (n_out / n_len).astype(np.float32)
  uv_out = c_uv[first_occ].astype(np.float32)
  f_out = inv.reshape(n_tris, 3).astype(np.int32)
  return v_out, n_out, uv_out, f_out


def main():
  args = _parse_args()
  os.makedirs(args.out_dir, exist_ok=True)
  rng = np.random.default_rng(args.seed)

  bpy.ops.wm.read_factory_settings(use_empty=True)
  scene = bpy.context.scene
  scene.render.engine = "CYCLES"
  scene.cycles.device = "CPU"
  scene.cycles.use_denoising = True
  scene.cycles.max_bounces = 6
  scene.cycles.diffuse_bounces = 4
  scene.cycles.glossy_bounces = 4
  scene.cycles.use_adaptive_sampling = True
  scene.cycles.adaptive_threshold = 0.02

  fovy = 42.0
  scene.render.resolution_percentage = 100
  scene.render.image_settings.file_format = "OPEN_EXR"
  scene.render.image_settings.color_mode = "RGB"
  scene.render.image_settings.color_depth = "32"
  scene.view_settings.view_transform = "Standard"
  scene.view_settings.look = "None"
  scene.view_settings.exposure = 0.0

  # Studio Gallery World + Overhead Warm Key + Side Cool Fill
  world = bpy.data.worlds.new("GalleryWorld")
  scene.world = world
  world.use_nodes = True
  bg = world.node_tree.nodes.get("Background")
  bg.inputs["Color"].default_value = (0.34, 0.42, 0.54, 1.0)
  bg.inputs["Strength"].default_value = 0.48

  light_pos = np.array([0.85, 0.55, 2.35], dtype=np.float32)
  area_data = bpy.data.lights.new(name="OverheadKey", type="AREA")
  area_data.energy = 85.0
  area_data.size = 1.05
  area_data.color = (1.0, 0.96, 0.90)
  area_obj = bpy.data.objects.new("OverheadKey", area_data)
  bpy.context.collection.objects.link(area_obj)
  area_obj.location = tuple(float(x) for x in light_pos)
  area_obj.rotation_euler = (Vector((0.0, 0.0, 0.25)) - Vector(light_pos)).to_track_quat("-Z", "Y").to_euler()

  fill_data = bpy.data.lights.new(name="SideFill", type="AREA")
  fill_data.energy = 38.0
  fill_data.size = 1.60
  fill_data.color = (0.82, 0.90, 1.0)
  fill_obj = bpy.data.objects.new("SideFill", fill_data)
  bpy.context.collection.objects.link(fill_obj)
  fill_obj.location = (-1.20, -1.10, 1.65)
  fill_obj.rotation_euler = (Vector((0.0, 0.0, 0.25)) - fill_obj.location).to_track_quat("-Z", "Y").to_euler()

  # Generate PBR textures and build 6 high-res meshes (>1M triangles total)
  tmpdir = tempfile.mkdtemp(prefix="blender_1m_")
  t_build = time.time()
  tex_map = make_pbr_textures(512)

  suz = _build_suzanne_hero()
  tor = _build_torus_knot()
  wav = _build_wave_sculpture()
  col = _build_gallery_colonnade()
  flr = _build_gallery_floor()
  ped = _build_pedestal()

  asset_specs = [
      ("suzanne_hero", suz, 0.35, 1.0),
      ("torus_knot", tor, 0.08, 2.0),
      ("wave_sculpture", wav, 0.05, 1.0),
      ("gallery_colonnade", col, 0.0, 2.0),
      ("gallery_floor", flr, 0.04, 3.0),
      ("pedestal", ped, 0.0, 1.5),
  ]

  meshes_np = []
  total_tris = 0
  for name, obj, metallic, uv_scale in asset_specs:
    alb, rgh = tex_map[name]
    mat = _make_textured_material(f"Mat_{name}", alb, rgh, metallic=metallic, uv_scale=uv_scale, tex_dir=tmpdir)
    obj.data.materials.append(mat)
    v, n, uv, f = _finalize_and_extract_mesh(obj, uv_scale=uv_scale)
    total_tris += int(f.shape[0])
    meshes_np.append((name, v, n, uv, f, alb, rgh, float(metallic)))
    print(f"Built '{name}': verts={v.shape[0]:,}, tris={f.shape[0]:,}", flush=True)

  print(f"TOTAL SCENE TRIANGLES: {total_tris:,} (built in {time.time()-t_build:.2f}s)", flush=True)
  assert total_tris >= 1_000_000, f"Expected >= 1,000,000 tris, got {total_tris}"

  # Camera setup aligned to compute_camera_rays
  cam_data = bpy.data.cameras.new("MainCam")
  cam_data.sensor_fit = "VERTICAL"
  cam_data.sensor_width = 36.0
  cam_data.sensor_height = 36.0
  cam_data.lens_unit = "FOV"
  cam_data.angle = math.radians(fovy)
  cam_data.clip_start = 0.05
  cam_data.clip_end = 50.0
  cam_obj = bpy.data.objects.new("MainCam", cam_data)
  bpy.context.collection.objects.link(cam_obj)
  scene.camera = cam_obj

  # Generate poses: 48 frames (256x256) + 4 held-out 512x512 hero showcase views
  N = args.n_frames
  N_hero = args.n_hero
  N_tot = N + N_hero
  M = len(meshes_np)
  obj_xpos = np.zeros((N_tot, M, 3), dtype=np.float32)
  obj_xmat = np.zeros((N_tot, M, 3, 3), dtype=np.float32)
  cam_xpos = np.zeros((N_tot, 3), dtype=np.float32)
  cam_xmat = np.zeros((N_tot, 3, 3), dtype=np.float32)

  for i in range(N_tot):
    if i < N:
      az = 2.0 * math.pi * (i / float(N)) - 0.55
      r_cam = 1.66 + 0.08 * math.sin(2.0 * az)
      cz = 0.86 + 0.10 * math.cos(3.0 * az)
      if i == 0:
        az = -0.55
        r_cam = 1.66
        cz = 0.88
      cp = np.array([r_cam * math.cos(az), r_cam * math.sin(az), cz], dtype=np.float32)
      t = az + 0.55
    else:
      # 4 distinct held-out 512x512 hero orbit angles & dynamic poses
      k = i - N
      az = 0.42 + k * (0.5 * math.pi)
      r_cam = 1.62 + 0.06 * (k % 2)
      cz = 0.84 + 0.08 * math.sin(k * 1.1)
      cp = np.array([r_cam * math.cos(az), r_cam * math.sin(az), cz], dtype=np.float32)
      t = az + 1.37

    cm = _look_at_cam_mat(cp, np.array([0.0, 0.0, 0.34], dtype=np.float32))
    cam_xpos[i], cam_xmat[i] = cp, cm

    # Dynamic Hero Suzanne on pedestal (rotates & gently bobs)
    suz_p = np.array([0.0, 0.0, 0.35 + 0.025 * math.sin(2.0 * t)], np.float32)
    suz_m = _euler_zyx(0.45 + 0.65 * math.sin(t), -0.10 + 0.12 * math.cos(t), 0.08 * math.sin(1.5 * t))

    # Dynamic Torus Knot orbiting hero
    tk_ang = t * 1.0 + 0.35
    tor_p = np.array([0.48 * math.cos(tk_ang), 0.48 * math.sin(tk_ang), 0.33 + 0.04 * math.cos(2.0 * t)], np.float32)
    tor_m = _euler_zyx(tk_ang * 1.4, 0.55 * math.sin(t), 0.45 * math.cos(t))

    # Dynamic Wave Sculpture orbiting opposite Torus Knot
    wv_ang = tk_ang + math.pi
    wav_p = np.array([0.48 * math.cos(wv_ang), 0.48 * math.sin(wv_ang), 0.32 + 0.04 * math.sin(2.0 * t)], np.float32)
    wav_m = _euler_zyx(-wv_ang * 1.2, 0.35 * math.cos(t), -0.35 * math.sin(t))

    # Static gallery colonnade, floor, pedestal
    id_m = np.eye(3, dtype=np.float32)
    z_p = np.zeros((3,), dtype=np.float32)

    poses = [(suz_p, suz_m), (tor_p, tor_m), (wav_p, wav_m), (z_p, id_m), (z_p, id_m), (z_p, id_m)]
    for m_idx, (op, om) in enumerate(poses):
      obj_xpos[i, m_idx] = op
      obj_xmat[i, m_idx] = om

  exr_path = os.path.join(tmpdir, "frame.exr")

  def _render_range(indices: List[int], res: int, samples: int, save_f0_depth: bool):
    scene.render.resolution_x = res
    scene.render.resolution_y = res
    scene.cycles.samples = samples
    out_rgb = np.zeros((len(indices), res, res, 3), dtype=np.float32)
    f0_depth = None

    if save_f0_depth:
      bpy.context.view_layer.use_pass_z = True
      ng = bpy.data.node_groups.new(f"Comp_{res}", "CompositorNodeTree")
      scene.compositing_node_group = ng
      rl = ng.nodes.new("CompositorNodeRLayers")
      fout = ng.nodes.new("CompositorNodeOutputFile")
      fout.directory = tmpdir
      fout.file_name = f"depth_{res}_"
      fout.format.media_type = "IMAGE"
      fout.format.file_format = "OPEN_EXR"
      fout.format.color_depth = "32"
      fout.file_output_items.clear()
      fout.file_output_items.new("FLOAT", "Depth")
      ng.links.new(rl.outputs["Depth"], fout.inputs["Depth"])

    t0 = time.time()
    all_objs = [suz, tor, wav, col, flr, ped]
    for k, fi in enumerate(indices):
      for m_idx, o in enumerate(all_objs):
        _set_obj_pose(o, obj_xpos[fi, m_idx], obj_xmat[fi, m_idx])
      _set_obj_pose(cam_obj, cam_xpos[fi], cam_xmat[fi])

      if k == 1 and save_f0_depth:
        scene.compositing_node_group = None

      scene.render.filepath = exr_path
      bpy.ops.render.render(write_still=True)

      img = bpy.data.images.load(exr_path, check_existing=False)
      px = np.asarray(img.pixels[:], dtype=np.float32).reshape(res, res, 4)[..., :3][::-1, :, :].copy()
      bpy.data.images.remove(img)
      px = np.maximum(px, 0.0)
      px = px / (1.0 + 0.15 * px)
      px = np.where(px <= 0.0031308, 12.92 * px, 1.055 * (np.maximum(px, 1e-8) ** (1.0 / 2.4)) - 0.055)
      out_rgb[k] = np.clip(px, 0.0, 1.0)

      if k == 0 and save_f0_depth:
        d_files = sorted(f for f in os.listdir(tmpdir) if f.startswith(f"depth_{res}_") and f.endswith(".exr"))
        if d_files:
          dimg = bpy.data.images.load(os.path.join(tmpdir, d_files[0]), check_existing=False)
          f0_depth = np.asarray(dimg.pixels[:], dtype=np.float32).reshape(res, res, 4)[..., 0][::-1, :].copy()
          bpy.data.images.remove(dimg)

      if (k + 1) % 8 == 0 or k + 1 == len(indices):
        print(f"Rendered {k+1}/{len(indices)} @ {res}x{res} ({samples} spp) in {time.time()-t0:.1f}s", flush=True)
    return out_rgb, f0_depth

  gt_256, f0_z_256 = _render_range(list(range(N)), args.res, args.samples, save_f0_depth=True)
  gt_512, f0_z_512 = _render_range(list(range(N, N + N_hero)), args.hero_res, args.hero_samples, save_f0_depth=True)

  # Interleaved 36 train / 4 calib / 8 test across the 360-degree orbit
  split = np.zeros((N,), dtype=np.int32)
  test_ids = [5, 11, 17, 23, 29, 35, 41, 47]
  calib_ids = [3, 15, 27, 39]
  for tid in test_ids:
    if tid < N:
      split[tid] = 2
  for cid in calib_ids:
    if cid < N:
      split[cid] = 1

  bundle = {
      "n_meshes": np.int32(M),
      "total_tris": np.int32(total_tris),
      "height": np.int32(args.res),
      "width": np.int32(args.res),
      "hero_height": np.int32(args.hero_res),
      "hero_width": np.int32(args.hero_res),
      "cam_fovy": np.float32(fovy),
      "znear": np.float32(0.05),
      "zfar": np.float32(50.0),
      "light_pos": light_pos,
      "obj_xpos": obj_xpos,
      "obj_xmat": obj_xmat,
      "cam_xpos": cam_xpos,
      "cam_xmat": cam_xmat,
      "split_256": split,
  }
  for idx, (name, v, n, uv, f, alb, rgh, met) in enumerate(meshes_np):
    bundle[f"mesh_{idx}_name"] = np.asarray(name)
    bundle[f"mesh_{idx}_verts"] = v
    bundle[f"mesh_{idx}_normals"] = n
    bundle[f"mesh_{idx}_uvs"] = uv
    bundle[f"mesh_{idx}_faces"] = f
    bundle[f"mesh_{idx}_tex_alb"] = (alb * 255.0 + 0.5).astype(np.uint8)
    bundle[f"mesh_{idx}_tex_rgh"] = (rgh * 255.0 + 0.5).astype(np.uint8)
    bundle[f"mesh_{idx}_metallic"] = np.float32(met)

  scene_path = os.path.join(args.out_dir, "v3_w5_blender_1m_scene.npz")
  gt256_path = os.path.join(args.out_dir, "v3_w5_blender_1m_gt_256.npz")
  gt512_path = os.path.join(args.out_dir, "v3_w5_blender_1m_gt_512.npz")
  np.savez_compressed(scene_path, **bundle)
  np.savez_compressed(gt256_path, rgb=gt_256, frame0_z_depth=f0_z_256)
  np.savez_compressed(gt512_path, rgb=gt_512, frame0_z_depth=f0_z_512)
  print(f"Saved {scene_path} ({os.path.getsize(scene_path)/1e6:.1f} MB), {gt256_path}, {gt512_path}", flush=True)


if __name__ == "__main__":
  main()
