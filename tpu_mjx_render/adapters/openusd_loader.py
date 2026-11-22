"""Native OpenUSD (.usd/.usda/.usdc/.usdz) -> TPU Wide-BVH-64 Compiler.

Auto-compiles OpenUSD stages directly into 2-Level Wide-TLAS-64 + Compact
BLAS-64 TPU buffers with zero per-scene custom export scripts:
  - UsdGeom.Mesh (vectorized fan triangulation for quads/ngons, normals, UVs)
    & analytic UsdGeom prims (Sphere, Cube, Cylinder) + optional UsdGeom.BasisCurves
    ribbon tessellation (for feathery Ironwood needles & Pandanus leaves)
  - UsdGeom.XformCache world transforms + automatic Y-up -> Z-up stage axis
    conversion
  - Full USD Native Instancing (IsInstance() / stage.GetPrototypes()) AND
    nested UsdGeom.PointInstancer -> 2-Level Wide-TLAS-64 / Compact BLAS-64
    buffers with 3D Morton spatial sorting & prototype deduplication
  - UsdShade MaterialBindingAPI -> UsdPreviewSurface / OmniPBR / UsdUVTexture
    AND Disney Animation PxrDisneyBsdf / PxrSurface / BaseMaterial /
    PtexBaseMaterial semantic PBR fallback (albedo, roughness, metallic, tag)
  - UsdLux (DistantLight, SphereLight, RectLight, DomeLight) -> lighting & HDRI
  - UsdGeom.Camera -> (cam_xpos, cam_xmat, cam_fovy, znear, zfar)

Includes a pure-Python .usda fallback parser when `pxr` (usd-core) is absent.
"""

from __future__ import annotations

import dataclasses
import math
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple
import zipfile

import numpy as np

try:
  from PIL import Image
except Exception:
  Image = None  # type: ignore

try:
  from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade
  HAS_PXR = True
except Exception:
  HAS_PXR = False


@dataclasses.dataclass
class OpenUSDMeshData:
  name: str
  verts: np.ndarray            # (V, 3) float32
  faces: np.ndarray            # (F, 3) int32
  vert_normals: Optional[np.ndarray]  # (V, 3) float32 or None
  rgba: np.ndarray             # (4,) float32
  uvs: Optional[np.ndarray] = None
  texture_rgb: Optional[np.ndarray] = None  # (H_t, W_t, 3) float32
  roughness: float = 0.55
  metallic: float = 0.0
  material_tag: str = "default"


@dataclasses.dataclass
class OpenUSDPointInstancerGroup:
  name: str
  prototype_mesh: OpenUSDMeshData
  inst_xpos: np.ndarray        # (K, 3) float32
  inst_xmat: np.ndarray        # (K, 3, 3) float32


@dataclasses.dataclass
class OpenUSDCamera:
  name: str
  cam_xpos: np.ndarray         # (3,) float32
  cam_xmat: np.ndarray         # (3, 3) float32 ([right, up, -fwd])
  cam_fovy: float
  znear: float
  zfar: float
  focal_length: float = 24.0
  horizontal_aperture: float = 36.0
  vertical_aperture: float = 20.25


@dataclasses.dataclass
class OpenUSDStageData:
  stage_path: str
  up_axis: str
  meters_per_unit: float
  unique_meshes: List[OpenUSDMeshData]
  instanced_groups: List[OpenUSDPointInstancerGroup]
  camera: OpenUSDCamera
  light_pos: np.ndarray        # (3,) float32
  light_diffuse: Tuple[float, float, float]
  light_ambient: Tuple[float, float, float]
  sky_rgb1: Tuple[float, float, float]
  sky_rgb2: Tuple[float, float, float]
  water_meshes: List[OpenUSDMeshData] = dataclasses.field(default_factory=list)
  dome_texture_path: Optional[str] = None
  dome_rotate_y_deg: float = 0.0
  n_usd_instances: int = 0
  n_point_instances: int = 0


def _y_up_to_z_up_points(pts: np.ndarray) -> np.ndarray:
  """Rotate Y-up OpenUSD coords (x, y, z) -> Z-up (x, -z, y)."""
  if pts.size == 0:
    return pts.astype(np.float32)
  return np.stack([pts[:, 0], -pts[:, 2], pts[:, 1]], axis=-1).astype(np.float32)


def _y_up_to_z_up_mat3(R: np.ndarray) -> np.ndarray:
  """Conjugate 3x3 rotation/linear matrix from Y-up to Z-up: P @ R @ P^T."""
  P = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float32)
  return (P @ R @ P.T).astype(np.float32)


def _y_up_to_z_up_mat3_batch(Rs: np.ndarray) -> np.ndarray:
  """Vectorized conjugation P @ R @ P^T for (K, 3, 3) matrices."""
  if Rs.size == 0:
    return Rs.astype(np.float32)
  P = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float32)
  return np.einsum("ij,bjk,lk->bil", P, Rs, P).astype(np.float32)


def _morton3d_20bit(pts: np.ndarray) -> np.ndarray:
  """Compute 60-bit 3D Morton code for spatial sorting of instance positions."""
  if pts.shape[0] <= 1:
    return np.zeros((pts.shape[0],), dtype=np.uint64)
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


def _compute_smooth_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
  if verts.shape[0] == 0 or faces.shape[0] == 0:
    return np.zeros_like(verts, dtype=np.float32)
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
  bad = norms[:, 0] < 1e-12
  if np.any(bad):
    vn[bad] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    norms = np.linalg.norm(vn, axis=-1, keepdims=True)
  return (vn / np.maximum(norms, 1e-12)).astype(np.float32)


def _triangulate_faces(counts: np.ndarray, indices: np.ndarray) -> np.ndarray:
  """Vectorized fan-triangulate arbitrary polygon counts into (F_tri, 3) int32."""
  if counts.size == 0 or indices.size == 0:
    return np.zeros((0, 3), dtype=np.int32)
  if np.all(counts == 3):
    return indices.reshape(-1, 3).astype(np.int32)
  if np.all(counts == 4):
    q = indices.reshape(-1, 4)
    out = np.empty((q.shape[0] * 2, 3), dtype=np.int32)
    out[0::2] = q[:, [0, 1, 2]]
    out[1::2] = q[:, [0, 2, 3]]
    return out
  # Mixed counts: fast vectorized construction
  valid = counts >= 3
  if not np.any(valid):
    return np.zeros((0, 3), dtype=np.int32)
  starts = np.cumsum(counts) - counts
  tris = []
  for c, s in zip(counts[valid], starts[valid]):
    v0 = int(indices[s])
    for k in range(1, int(c) - 1):
      tris.append((v0, int(indices[s + k]), int(indices[s + k + 1])))
  if not tris:
    return np.zeros((0, 3), dtype=np.int32)
  return np.asarray(tris, dtype=np.int32)


def _make_uv_sphere(radius: float = 1.0, segs: int = 24, rings: int = 16) -> Tuple[np.ndarray, np.ndarray]:
  verts = []
  for r in range(rings + 1):
    phi = math.pi * r / rings
    sp, cp = math.sin(phi), math.cos(phi)
    for s in range(segs):
      th = 2.0 * math.pi * s / segs
      verts.append((radius * sp * math.cos(th), radius * sp * math.sin(th), radius * cp))
  faces = []
  for r in range(rings):
    for s in range(segs):
      sn = (s + 1) % segs
      a = r * segs + s
      b = r * segs + sn
      c = (r + 1) * segs + s
      d = (r + 1) * segs + sn
      if r > 0:
        faces.append((a, c, b))
      if r < rings - 1:
        faces.append((b, c, d))
  return np.asarray(verts, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def _make_box(half_size: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
  h = float(half_size)
  verts = np.array([
      [-h, -h, -h], [h, -h, -h], [h, h, -h], [-h, h, -h],
      [-h, -h,  h], [h, -h,  h], [h, h,  h], [-h, h,  h],
  ], dtype=np.float32)
  faces = np.array([
      [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
      [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
      [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5],
  ], dtype=np.int32)
  return verts, faces


def _make_cylinder(radius: float = 1.0, height: float = 2.0, segs: int = 24) -> Tuple[np.ndarray, np.ndarray]:
  hz = 0.5 * float(height)
  verts = [(0.0, 0.0, -hz), (0.0, 0.0, hz)]
  for z in (-hz, hz):
    for s in range(segs):
      th = 2.0 * math.pi * s / segs
      verts.append((radius * math.cos(th), radius * math.sin(th), z))
  faces = []
  b0, t0 = 2, 2 + segs
  for s in range(segs):
    sn = (s + 1) % segs
    faces.append((0, b0 + sn, b0 + s))
    faces.append((1, t0 + s, t0 + sn))
    faces.append((b0 + s, b0 + sn, t0 + sn))
    faces.append((b0 + s, t0 + sn, t0 + s))
  return np.asarray(verts, dtype=np.float32), np.asarray(faces, dtype=np.int32)


_TEX_MEAN_CACHE: Dict[str, Optional[np.ndarray]] = {}


def _load_texture_rgb(stage_path: str, asset_path: str, layer_dir: str = "") -> Optional[np.ndarray]:
  if Image is None or not asset_path:
    return None
  resolved = asset_path
  if not os.path.isabs(resolved):
    base_dir = layer_dir if layer_dir else os.path.dirname(os.path.abspath(stage_path))
    resolved = os.path.normpath(os.path.join(base_dir, asset_path))
  if resolved in _TEX_MEAN_CACHE:
    return _TEX_MEAN_CACHE[resolved]
  try:
    if os.path.exists(resolved):
      with Image.open(resolved) as im:
        arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
        _TEX_MEAN_CACHE[resolved] = arr
        return arr
    if stage_path.lower().endswith(".usdz") and os.path.exists(stage_path):
      with zipfile.ZipFile(stage_path, "r") as zf:
        rel = asset_path.lstrip("./")
        for member in zf.namelist():
          if member.endswith(rel):
            with zf.open(member) as mf:
              with Image.open(mf) as im:
                arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
                _TEX_MEAN_CACHE[resolved] = arr
                return arr
  except Exception:
    _TEX_MEAN_CACHE[resolved] = None
    return None
  _TEX_MEAN_CACHE[resolved] = None
  return None


def _is_sentinel_color(c: Tuple[float, float, float]) -> bool:
  """Detect Disney USD Ptex placeholder colors like (1,0,0), (1,1,0), (1,0,1), or flat (0.5,0.5,0.5)."""
  r, g, b = c
  if (r > 0.95 or r < 0.05) and (g > 0.95 or g < 0.05) and (b > 0.95 or b < 0.05):
    # Pure primary/secondary debug sentinels (1,0,0), (1,1,0), (1,0,1), (0,1,0), (0,0,1), (0,1,1)
    if not (r > 0.95 and g > 0.95 and b > 0.95) and not (r < 0.05 and g < 0.05 and b < 0.05):
      return True
  if abs(r - 0.502) < 5e-3 and abs(g - 0.502) < 5e-3 and abs(b - 0.502) < 5e-3:
    return True
  return False


def _resolve_semantic_pbr(prim_path: str, mat_name: str, raw_rgb: Optional[Tuple[float, float, float]], raw_rough: Optional[float], raw_metal: Optional[float]) -> Tuple[Tuple[float, float, float], float, float, str]:
  """Resolve physically-plausible linear PBR albedo, roughness, metallic, and semantic tag."""
  low_p = prim_path.lower()
  low_m = mat_name.lower()
  rough = float(raw_rough) if raw_rough is not None else 0.65
  metal = float(raw_metal) if raw_metal is not None else 0.0

  # Water / ocean
  if "osocean" in low_p or "water" in low_m:
    return (0.04, 0.36, 0.35), 0.04, 0.0, "water"

  # Beach / sand / coastline
  if "sand" in low_m or "beach_geo" in low_p or ("isbeach" in low_p and "hibiscus" not in low_p and "grass" not in low_p and "debris" not in low_p and "stone" not in low_p and "pebble" not in low_p):
    return (0.68, 0.56, 0.41), 0.85, 0.0, "sand"
  if "iscoastline" in low_p and "sand" in low_m:
    return (0.64, 0.53, 0.39), 0.86, 0.0, "sand"
  if "pebble" in low_p or "pebble" in low_m:
    return (0.38, 0.34, 0.29), 0.80, 0.0, "rock"

  # Volcanic lava rocks & dark rock faces
  if "lavarock" in low_p or "volcanicrock" in low_m or "rockface" in low_m or "rocksm" in low_m or "stone" in low_m:
    return (0.09, 0.08, 0.075), 0.72, 0.0, "rock"

  # Coral reefs (underwater mossy olive, chartreuse, teal-brown)
  if "iscoral" in low_p or "cabbage" in low_m or "antler" in low_m or "staghorn" in low_m or "flute" in low_m:
    if "flute" in low_m or "flutes" in low_p:
      return (0.34, 0.46, 0.14), 0.82, 0.0, "coral"
    if "cabbage" in low_m or "cabbage" in low_p:
      return (0.28, 0.38, 0.12), 0.84, 0.0, "coral"
    if "antler" in low_m or "staghorn" in low_m:
      return (0.32, 0.28, 0.18), 0.82, 0.0, "coral"
    return (0.19, 0.24, 0.12), 0.86, 0.0, "coral"

  # Flowers (hibiscus yellow, gardenia ivory)
  if "flower" in low_m or "petal" in low_m or "stigma" in low_m or "sepal" in low_m or "anther" in low_m or "style" in low_m:
    if "gardenia" in low_p or "flowerbud" in low_m:
      return (0.86, 0.82, 0.72), 0.55, 0.0, "flower"
    return (0.88, 0.72, 0.06), 0.60, 0.0, "flower"

  # Palm trunks & dead palms
  if ("ispalm" in low_p or "coco" in low_p) and ("trunk" in low_m or "dead" in low_p or "spine" in low_m):
    return (0.29, 0.24, 0.19), 0.84, 0.0, "bark"

  # Ironwood & BayCedar bark/branches
  if "bark" in low_m or "trunk" in low_m or "branch" in low_m or "root" in low_m or "soil" in low_m or "debris" in low_m or "fiber" in low_m:
    if raw_rgb is not None and not _is_sentinel_color(raw_rgb):
      # Gently warm/normalize weathered bark
      return (max(0.05, raw_rgb[0] * 0.65), max(0.04, raw_rgb[1] * 0.62), max(0.035, raw_rgb[2] * 0.58)), rough, metal, "bark"
    if "soil" in low_m or "dunes" in low_p:
      return (0.18, 0.12, 0.08), 0.88, 0.0, "bark"
    return (0.24, 0.20, 0.16), 0.80, 0.0, "bark"

  # Foliage (palms, mountain canopy, ferns, hibiscus leaves, pandanus, gardenia, bay cedar, naupaka, kava)
  is_foliage = any(k in low_p or k in low_m for k in (
      "frond", "leaf", "leaves", "foliage", "fern", "grass", "breadfruit", "breakfruit",
      "hibiscus", "gardenia", "baycedar", "naupaka", "kava", "pandanus", "ironwood", "mountain", "needle", "pine",
  ))
  if raw_rgb is not None and not _is_sentinel_color(raw_rgb):
    if is_foliage:
      # Moana USD linear albedos for mountain/foliage look rich and lush under sunlight
      if "mountainlo" in low_m:
        return (raw_rgb[0] * 0.42, raw_rgb[1] * 0.52, raw_rgb[2] * 0.30), 0.88, 0.0, "foliage"
      return (raw_rgb[0] * 0.72, raw_rgb[1] * 0.78, raw_rgb[2] * 0.68), rough, metal, "foliage"
    return raw_rgb, rough, metal, "default"

  if is_foliage:
    if "mountain" in low_p:
      return (0.11, 0.22, 0.055), 0.90, 0.0, "foliage"
    if "fern" in low_m or "fern" in low_p:
      return (0.16, 0.32, 0.07), 0.60, 0.0, "foliage"
    if "gardenia" in low_p:
      return (0.12, 0.26, 0.06), 0.42, 0.0, "foliage"
    if "pandanus" in low_p:
      return (0.30, 0.34, 0.14), 0.65, 0.0, "foliage"
    return (0.16, 0.30, 0.07), 0.58, 0.0, "foliage"

  if "dunes" in low_p:
    return (0.28, 0.21, 0.14), 0.85, 0.0, "bark"

  return (0.48, 0.45, 0.40), rough, metal, "default"


def _extract_shader_rgba(stage_path: str, prim: "Usd.Prim") -> Tuple[np.ndarray, Optional[np.ndarray], float, float, str]:
  """Extract (rgba[4], optional texture_rgb, roughness, metallic, tag) from UsdShade MaterialBindingAPI."""
  default_rgba = np.array([0.74, 0.72, 0.68, 1.0], dtype=np.float32)
  tex_rgb: Optional[np.ndarray] = None
  pstr = str(prim.GetPath())
  try:
    mat_api = UsdShade.MaterialBindingAPI(prim)
    mat, _ = mat_api.ComputeBoundMaterial()
    if not mat:
      cur = prim
      while cur and cur.IsValid() and not cur.IsPseudoRoot():
        rel = cur.GetRelationship("material:binding")
        if rel and rel.GetTargets():
          tprim = prim.GetStage().GetPrimAtPath(rel.GetTargets()[0])
          if tprim and tprim.IsValid():
            mat = UsdShade.Material(tprim)
            break
        cur = cur.GetParent()
    if not mat:
      gprim = UsdGeom.Gprim(prim)
      dc = gprim.GetDisplayColorAttr()
      if dc and dc.HasAuthoredValueOpinion():
        vals = dc.Get()
        if vals and len(vals) > 0:
          c = vals[0]
          return np.array([float(c[0]), float(c[1]), float(c[2]), 1.0], dtype=np.float32), None, 0.55, 0.0, "default"
      rgb_s, r_s, m_s, tag_s = _resolve_semantic_pbr(pstr, "", None, None, None)
      if "moana" in stage_path.lower() or "island" in pstr.lower():
        return np.array([rgb_s[0], rgb_s[1], rgb_s[2], 1.0], dtype=np.float32), None, r_s, m_s, tag_s
      return default_rgba, None, 0.55, 0.0, "default"

    mprim = mat.GetPrim()
    mname = mprim.GetName()
    is_moana = ("moana" in stage_path.lower()) or ("/island/" in pstr.lower()) or ("/__prototype_" in pstr.lower())

    # 1. Check Material-level Disney Pxr inputs (Moana USD authoring pattern)
    mat_bc = mprim.GetAttribute("inputs:baseColor")
    mat_rough = mprim.GetAttribute("inputs:roughness")
    mat_metal = mprim.GetAttribute("inputs:metallic")
    mat_alpha = mprim.GetAttribute("inputs:alpha")
    raw_bc = None
    if mat_bc and mat_bc.Get() is not None:
      v = mat_bc.Get()
      raw_bc = (float(v[0]), float(v[1]), float(v[2]))
    raw_r = float(mat_rough.Get()) if (mat_rough and mat_rough.Get() is not None) else None
    raw_m = float(mat_metal.Get()) if (mat_metal and mat_metal.Get() is not None) else None
    alpha = float(mat_alpha.Get()) if (mat_alpha and mat_alpha.Get() is not None) else 1.0

    if is_moana:
      rgb_s, r_s, m_s, tag_s = _resolve_semantic_pbr(pstr, mname, raw_bc, raw_r, raw_m)
      return np.array([rgb_s[0], rgb_s[1], rgb_s[2], alpha], dtype=np.float32), None, r_s, m_s, tag_s

    # 2. Standard UsdPreviewSurface / OmniPBR / glslfx outputs
    surf_out = mat.GetSurfaceOutput()
    if not surf_out or not surf_out.HasConnectedSource():
      surf_out = mat.GetSurfaceOutput("mdl")
    if not surf_out or not surf_out.HasConnectedSource():
      surf_out = mat.GetSurfaceOutput("glslfx")
    if surf_out and surf_out.HasConnectedSource():
      src_info = surf_out.GetConnectedSource()
      shader_prim = src_info[0].GetPrim() if src_info else None
      if shader_prim:
        shader = UsdShade.Shader(shader_prim)
        op_in = shader.GetInput("opacity")
        if op_in and op_in.Get() is not None:
          alpha = float(op_in.Get())
        r_in = shader.GetInput("roughness") or shader.GetInput("reflection_roughness_constant")
        if r_in and r_in.Get() is not None:
          raw_r = float(r_in.Get())
        m_in = shader.GetInput("metallic") or shader.GetInput("metallic_constant")
        if m_in and m_in.Get() is not None:
          raw_m = float(m_in.Get())
        for cname in ("diffuseColor", "diffuse_color_constant", "albedo_color", "baseColor", "diffuse_tint"):
          cin = shader.GetInput(cname)
          if not cin:
            continue
          if cin.HasConnectedSource():
            # Walk up to 4 hops upstream to locate the Image_Texture / UsdUVTexture file
            queue = [cin.GetConnectedSource()[0].GetPrim()] if cin.GetConnectedSource() else []
            seen_nodes = set()
            while queue:
              tp = queue.pop(0)
              if not tp or str(tp.GetPath()) in seen_nodes:
                continue
              seen_nodes.add(str(tp.GetPath()))
              tsh = UsdShade.Shader(tp)
              fin = tsh.GetInput("file")
              if fin and fin.Get() is not None:
                asset = fin.Get()
                apath = getattr(asset, "resolvedPath", "") or getattr(asset, "path", str(asset))
                stack = fin.GetAttr().GetPropertyStack(Usd.TimeCode.Default())
                ldir = os.path.dirname(stack[0].layer.realPath) if (stack and stack[0].layer.realPath) else ""
                tex_rgb = _load_texture_rgb(stage_path, apath, layer_dir=ldir)
                if tex_rgb is not None:
                  mean_c = np.mean(tex_rgb.reshape(-1, 3), axis=0)
                  mean_lin = np.power(np.clip(mean_c, 0.0, 1.0), 2.2)
                  return (
                      np.array([float(mean_lin[0]), float(mean_lin[1]), float(mean_lin[2]), alpha], dtype=np.float32),
                      None,
                      raw_r if raw_r is not None else 0.55,
                      raw_m if raw_m is not None else 0.0,
                      "default",
                  )
              for sub_in in tsh.GetInputs():
                if sub_in.HasConnectedSource():
                  src_i = sub_in.GetConnectedSource()
                  if src_i:
                    queue.append(src_i[0].GetPrim())
          val = cin.Get()
          if val is not None:
            return (
                np.array([float(val[0]), float(val[1]), float(val[2]), alpha], dtype=np.float32),
                tex_rgb,
                raw_r if raw_r is not None else 0.55,
                raw_m if raw_m is not None else 0.0,
                "default",
            )
        for tname in ("diffuse_texture", "albedo_tex", "baseColorTexture"):
          tin = shader.GetInput(tname)
          if tin and tin.Get() is not None:
            asset = tin.Get()
            apath = getattr(asset, "resolvedPath", "") or getattr(asset, "path", str(asset))
            stack = tin.GetAttr().GetPropertyStack(Usd.TimeCode.Default())
            ldir = os.path.dirname(stack[0].layer.realPath) if (stack and stack[0].layer.realPath) else ""
            tex_rgb = _load_texture_rgb(stage_path, apath, layer_dir=ldir)
            if tex_rgb is not None:
              mean_c = np.mean(tex_rgb.reshape(-1, 3), axis=0)
              mean_lin = np.power(np.clip(mean_c, 0.0, 1.0), 2.2)
              return (
                  np.array([float(mean_lin[0]), float(mean_lin[1]), float(mean_lin[2]), alpha], dtype=np.float32),
                  None,
                  raw_r if raw_r is not None else 0.55,
                  raw_m if raw_m is not None else 0.0,
                  "default",
              )
    if raw_bc is not None:
      return np.array([raw_bc[0], raw_bc[1], raw_bc[2], alpha], dtype=np.float32), None, raw_r or 0.55, raw_m or 0.0, "default"
    # Semantic fallback for non-Moana assets when external texture PNGs were omitted
    lp = (pstr + "/" + mname).lower()
    if any(k in lp for k in ("black", "rubber", "tape", "visor", "tire")):
      return np.array([0.05, 0.05, 0.055, alpha], dtype=np.float32), None, 0.65, 0.0, "default"
    if any(k in lp for k in ("chrome", "aluminum", "metal", "steel", "key", "ring", "handle")):
      return np.array([0.62, 0.64, 0.68, alpha], dtype=np.float32), None, 0.22, 0.85, "default"
    if any(k in lp for k in ("wood", "cabinet", "drawer", "table", "desk", "door")):
      return np.array([0.32, 0.21, 0.12, alpha], dtype=np.float32), None, 0.60, 0.0, "default"
    if any(k in lp for k in ("metro", "card", "blue")):
      return np.array([0.12, 0.34, 0.58, alpha], dtype=np.float32), None, 0.40, 0.0, "default"
    if any(k in lp for k in ("bottle", "pump", "tray", "bowl")):
      return np.array([0.52, 0.54, 0.56, alpha], dtype=np.float32), None, 0.35, 0.0, "default"
  except Exception:
    pass
  return default_rgba, tex_rgb, 0.55, 0.0, "default"


def _transform_points_gf(pts: np.ndarray, mat_4x4: np.ndarray) -> np.ndarray:
  """Apply 4x4 row-vector USD matrix (p_row @ M) to (N, 3) points."""
  if pts.size == 0:
    return pts.astype(np.float32)
  M = np.asarray(mat_4x4, dtype=np.float64)
  out = pts.astype(np.float64) @ M[:3, :3] + M[3:4, :3]
  return out.astype(np.float32)


def _quats_to_mat3_batch(quats: Any) -> np.ndarray:
  """Vectorized conversion of USD Quath/Quatf/Quatd array -> (K, 3, 3) column-vector rotation matrices."""
  K = len(quats)
  if K == 0:
    return np.zeros((0, 3, 3), dtype=np.float32)
  # Convert via numpy buffer or fast comprehension
  q_np = np.array([(float(q.GetReal()), float(q.GetImaginary()[0]), float(q.GetImaginary()[1]), float(q.GetImaginary()[2])) for q in quats], dtype=np.float32)
  r, x, y, z = q_np[:, 0], q_np[:, 1], q_np[:, 2], q_np[:, 3]
  xx, yy, zz = x * x, y * y, z * z
  xy, xz, yz = x * y, x * z, y * z
  wx, wy, wz = r * x, r * y, r * z
  rots = np.empty((K, 3, 3), dtype=np.float32)
  rots[:, 0, 0] = 1.0 - 2.0 * (yy + zz)
  rots[:, 0, 1] = 2.0 * (xy - wz)
  rots[:, 0, 2] = 2.0 * (xz + wy)
  rots[:, 1, 0] = 2.0 * (xy + wz)
  rots[:, 1, 1] = 1.0 - 2.0 * (xx + zz)
  rots[:, 1, 2] = 2.0 * (yz - wx)
  rots[:, 2, 0] = 2.0 * (xz - wy)
  rots[:, 2, 1] = 2.0 * (yz + wx)
  rots[:, 2, 2] = 1.0 - 2.0 * (xx + yy)
  return rots


def _basis_curves_to_ribbon_mesh(
    prim: "Usd.Prim",
    tc: "Usd.TimeCode",
    meters_per_unit: float,
    stride: int = 3,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
  """Convert UsdGeom.BasisCurves into lightweight leaf-tapered ribbon triangles for feathery foliage."""
  bc = UsdGeom.BasisCurves(prim)
  pts = bc.GetPointsAttr().Get(tc)
  cnts = bc.GetCurveVertexCountsAttr().Get(tc)
  if pts is None or cnts is None or len(pts) == 0 or len(cnts) == 0:
    return None
  pts_np = np.asarray(pts, dtype=np.float32) * meters_per_unit
  cnts_np = np.asarray(cnts, dtype=np.int32)
  widths_attr = bc.GetWidthsAttr().Get(tc)
  w_base = float(widths_attr[0]) * meters_per_unit if (widths_attr is not None and len(widths_attr) > 0) else 0.025
  hw = max(0.016, min(0.08, 0.75 * w_base))

  # Uniform curve vertex counts fast path (common in XGen exports like Palms/Pandanus/Ironwood)
  if np.all(cnts_np == cnts_np[0]) and int(cnts_np[0]) >= 2:
    L = int(cnts_np[0])
    C = int(cnts_np.shape[0])
    if pts_np.shape[0] != C * L:
      return None
    eff_stride = stride if L >= 10 else max(1, stride - 1)
    c_pts = pts_np.reshape(C, L, 3)
    if eff_stride > 1 and L > 3:
      idx = list(range(0, L - 1, eff_stride)) + [L - 1]
      c_pts = c_pts[:, idx, :]
    Ls = c_pts.shape[1]
    tang = np.empty_like(c_pts)
    tang[:, :-1, :] = c_pts[:, 1:, :] - c_pts[:, :-1, :]
    tang[:, -1, :] = tang[:, -2, :]
    tnorm = np.linalg.norm(tang, axis=-1, keepdims=True)
    tang = tang / np.maximum(tnorm, 1e-8)
    up_ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)[None, None, :]
    side = np.cross(tang, up_ref)
    snorm = np.linalg.norm(side, axis=-1, keepdims=True)
    alt = np.cross(tang, np.array([1.0, 0.0, 0.0], dtype=np.float32)[None, None, :])
    side = np.where(snorm > 1e-5, side / np.maximum(snorm, 1e-8), alt / np.maximum(np.linalg.norm(alt, axis=-1, keepdims=True), 1e-8))
    # Leaf-blade taper: fuller middle, pointed tip
    taper = np.sin(np.linspace(0.30, 0.92 * np.pi, Ls, dtype=np.float32))[None, :, None]
    off = side * (hw * taper)
    v_left = c_pts - off
    v_right = c_pts + off
    verts = np.concatenate([v_left, v_right], axis=2).reshape(C * Ls * 2, 3)
    # Build triangle indices
    c_idx = np.arange(C, dtype=np.int32)[:, None] * (Ls * 2)
    s_idx = np.arange(Ls - 1, dtype=np.int32)[None, :] * 2
    base = (c_idx + s_idx).reshape(-1)
    t1 = np.stack([base, base + 1, base + 3], axis=-1)
    t2 = np.stack([base, base + 3, base + 2], axis=-1)
    faces = np.concatenate([t1, t2], axis=0).astype(np.int32)
    return verts.astype(np.float32), faces

  return None


def _merge_meshes_by_material(meshes: List[OpenUSDMeshData], max_buckets: int = 24) -> List[OpenUSDMeshData]:
  """Merge large counts of USD prims into <= max_buckets material buckets for fast XLA compilation."""
  if len(meshes) <= max_buckets:
    return meshes
  buckets: Dict[Tuple[str, int, int, int], List[OpenUSDMeshData]] = {}
  for md in meshes:
    key = (
        md.material_tag,
        int(round(float(md.rgba[0]) * 12)),
        int(round(float(md.rgba[1]) * 12)),
        int(round(float(md.rgba[2]) * 12)),
    )
    buckets.setdefault(key, []).append(md)
  items = sorted(buckets.items(), key=lambda kv: sum(m.faces.shape[0] for m in kv[1]), reverse=True)
  if len(items) > max_buckets:
    head = items[: max_buckets - 1]
    tail_meshes = [m for _, ms in items[max_buckets - 1 :] for m in ms]
    items = head + [(items[max_buckets - 1][0], tail_meshes)]
  merged: List[OpenUSDMeshData] = []
  for b_idx, ((tag, _, _, _), ms) in enumerate(items):
    vs, fs, ns = [], [], []
    voff = 0
    w_rgba = np.zeros((4,), dtype=np.float64)
    w_r, w_m, w_tot = 0.0, 0.0, 0
    for m in ms:
      vs.append(m.verts)
      fs.append(m.faces + voff)
      ns.append(m.vert_normals if m.vert_normals is not None else _compute_smooth_normals(m.verts, m.faces))
      voff += m.verts.shape[0]
      nf = int(m.faces.shape[0])
      w_rgba += np.asarray(m.rgba, dtype=np.float64) * nf
      w_r += float(m.roughness) * nf
      w_m += float(m.metallic) * nf
      w_tot += nf
    avg_rgba = (w_rgba / max(w_tot, 1)).astype(np.float32)
    merged.append(OpenUSDMeshData(
        name=f"usd_mat_bucket_{b_idx}_{tag}",
        verts=np.vstack(vs),
        faces=np.vstack(fs),
        vert_normals=np.vstack(ns),
        rgba=avg_rgba,
        roughness=float(w_r / max(w_tot, 1)),
        metallic=float(w_m / max(w_tot, 1)),
        material_tag=tag,
    ))
  return merged


def _load_stage_via_pxr(
    stage_path: str,
    time_code: Optional[float] = None,
    convert_y_up_to_z_up: bool = True,
    preferred_camera: Optional[str] = None,
    target_aspect: Optional[float] = None,
    include_curves: bool = False,
    max_curve_tris: int = 2_000_000,
    max_variants_per_pi: int = 4,
    min_instance_diag_m: float = 0.0,
    separate_water: bool = False,
) -> OpenUSDStageData:
  stage = Usd.Stage.Open(stage_path)
  if stage is None:
    raise RuntimeError(f"Failed to open USD stage: {stage_path}")
  tc = Usd.TimeCode.Default() if time_code is None else Usd.TimeCode(float(time_code))
  up_axis = str(UsdGeom.GetStageUpAxis(stage) or "Y")
  meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage) or 1.0)
  y_to_z = convert_y_up_to_z_up and (up_axis.upper() == "Y")
  xcache = UsdGeom.XformCache(tc)

  # Map USD Native Prototypes -> list of instance world transforms (for instanceable=true prims)
  usd_protos = stage.GetPrototypes()
  proto_users: Dict[str, List[Tuple[str, np.ndarray]]] = {str(pr.GetPath()): [] for pr in usd_protos}
  pi_proto_paths = set()
  top_instancers = []

  # First collect all PointInstancer prototype subtrees so we exclude them from USD instance collection
  pi_proto_paths = set()
  top_instancers = []
  for prim in stage.Traverse():
    if prim.IsA(UsdGeom.PointInstancer):
      top_instancers.append(prim)
      for tgt in UsdGeom.PointInstancer(prim).GetPrototypesRel().GetTargets():
        pi_proto_paths.add(str(tgt))

  for pr in usd_protos:
    for sp in Usd.PrimRange(pr):
      if sp.IsA(UsdGeom.PointInstancer):
        for tgt in UsdGeom.PointInstancer(sp).GetPrototypesRel().GetTargets():
          pi_proto_paths.add(str(tgt))

  def _is_under_pi_proto(prim_path: str) -> bool:
    return any(prim_path == p or prim_path.startswith(p + "/") for p in pi_proto_paths)

  for prim in stage.Traverse():
    if _is_under_pi_proto(str(prim.GetPath())):
      continue
    if prim.IsInstance():
      ppath = str(prim.GetPrototype().GetPath())
      M_u = np.asarray(xcache.GetLocalToWorldTransform(prim), dtype=np.float64)
      M_u[3, :3] *= meters_per_unit
      proto_users.setdefault(ppath, []).append((str(prim.GetPath()), M_u))

  unique_meshes: List[OpenUSDMeshData] = []
  water_meshes: List[OpenUSDMeshData] = []
  cameras: List[OpenUSDCamera] = []
  light_pos: Optional[np.ndarray] = None
  light_diffuse = (0.85, 0.83, 0.80)
  light_ambient = (0.28, 0.30, 0.34)
  sky_rgb1 = (0.28, 0.36, 0.48)
  sky_rgb2 = (0.10, 0.14, 0.20)
  dome_tex: Optional[str] = None
  dome_rot_y: float = 0.0
  curve_tris_used = 0

  def _extract_mesh_prim(
      prim: "Usd.Prim",
      parent_M: Optional[np.ndarray] = None,
      root_override: Optional["Usd.Prim"] = None,
      local_only: bool = False,
      name_prefix: str = "",
  ) -> Optional[OpenUSDMeshData]:
    nonlocal curve_tris_used
    rgba, tex_rgb, rough, metal, tag = _extract_shader_rgba(stage_path, prim)
    if name_prefix:
      _, _, _, tag_o = _resolve_semantic_pbr(name_prefix + "/" + prim.GetName(), prim.GetName(), (float(rgba[0]), float(rgba[1]), float(rgba[2])), rough, metal)
      if tag == "default" and tag_o != "default":
        tag = tag_o

    verts = None
    faces = None
    vnorm = None

    if prim.IsA(UsdGeom.Mesh):
      um = UsdGeom.Mesh(prim)
      pts = um.GetPointsAttr().Get(tc)
      cnt = um.GetFaceVertexCountsAttr().Get(tc)
      idx = um.GetFaceVertexIndicesAttr().Get(tc)
      if pts is None or cnt is None or idx is None or len(pts) == 0 or len(idx) == 0:
        return None
      verts = np.asarray(pts, dtype=np.float32) * meters_per_unit
      faces = _triangulate_faces(np.asarray(cnt, dtype=np.int32), np.asarray(idx, dtype=np.int32))
      n_attr = um.GetNormalsAttr()
      if n_attr and n_attr.HasAuthoredValueOpinion():
        n_vals = n_attr.Get(tc)
        if n_vals is not None and len(n_vals) == verts.shape[0]:
          vnorm = np.asarray(n_vals, dtype=np.float32)
    elif include_curves and prim.IsA(UsdGeom.BasisCurves) and curve_tris_used < max_curve_tris:
      res = _basis_curves_to_ribbon_mesh(prim, tc, meters_per_unit, stride=3)
      if res is None:
        return None
      verts, faces = res
      if curve_tris_used + faces.shape[0] > max_curve_tris:
        keep_f = max(0, max_curve_tris - curve_tris_used)
        if keep_f < 100:
          return None
        faces = faces[:keep_f]
      curve_tris_used += int(faces.shape[0])
    elif prim.IsA(UsdGeom.Sphere):
      us = UsdGeom.Sphere(prim)
      rad = float(us.GetRadiusAttr().Get(tc) or 1.0) * meters_per_unit
      verts, faces = _make_uv_sphere(rad)
    elif prim.IsA(UsdGeom.Cube):
      uc = UsdGeom.Cube(prim)
      sz = float(uc.GetSizeAttr().Get(tc) or 2.0) * 0.5 * meters_per_unit
      verts, faces = _make_box(sz)
    elif prim.IsA(UsdGeom.Cylinder):
      ucyl = UsdGeom.Cylinder(prim)
      rad = float(ucyl.GetRadiusAttr().Get(tc) or 1.0) * meters_per_unit
      ht = float(ucyl.GetHeightAttr().Get(tc) or 2.0) * meters_per_unit
      verts, faces = _make_cylinder(rad, ht)
    else:
      return None

    if faces is None or faces.shape[0] == 0:
      return None

    if not local_only:
      if root_override is not None:
        M_rel = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(tc), dtype=np.float64)
        M_rel[3, :3] *= meters_per_unit
      else:
        M_rel = np.asarray(xcache.GetLocalToWorldTransform(prim), dtype=np.float64)
        M_rel[3, :3] *= meters_per_unit
      if parent_M is not None:
        M = M_rel @ parent_M
      else:
        M = M_rel
      verts = _transform_points_gf(verts, M)
      if vnorm is not None:
        try:
          R_n = np.linalg.inv(M[:3, :3]).T
          vnorm = (vnorm @ R_n.T).astype(np.float32)
          vnorm /= np.maximum(np.linalg.norm(vnorm, axis=-1, keepdims=True), 1e-12)
        except Exception:
          vnorm = None

    if y_to_z:
      verts = _y_up_to_z_up_points(verts)
      if vnorm is not None:
        vnorm = _y_up_to_z_up_points(vnorm)

    if vnorm is None:
      vnorm = _compute_smooth_normals(verts, faces)

    return OpenUSDMeshData(
        name=name_prefix + str(prim.GetPath()),
        verts=verts,
        faces=faces,
        vert_normals=vnorm,
        rgba=rgba,
        roughness=rough,
        metallic=metal,
        material_tag=tag,
    )

  # 1. Traverse non-instanceable top-level stage (cameras, lights, non-instance meshes)
  for prim in stage.Traverse():
    pstr = str(prim.GetPath())
    if _is_under_pi_proto(pstr):
      continue
    if prim.IsA(UsdGeom.Camera):
      cam = UsdGeom.Camera(prim)
      M = np.asarray(xcache.GetLocalToWorldTransform(prim), dtype=np.float32)
      pos = M[3, :3] * meters_per_unit
      right = M[0, :3]
      up = M[1, :3]
      back = M[2, :3]
      R_cam = np.stack([right, up, back], axis=1).astype(np.float32)
      if y_to_z:
        pos = _y_up_to_z_up_points(pos[None, :])[0]
        R_cam = np.stack([
            _y_up_to_z_up_points(right[None, :])[0],
            _y_up_to_z_up_points(up[None, :])[0],
            _y_up_to_z_up_points(back[None, :])[0],
        ], axis=1).astype(np.float32)
      for col in range(3):
        R_cam[:, col] /= max(float(np.linalg.norm(R_cam[:, col])), 1e-8)
      fl = float(cam.GetFocalLengthAttr().Get(tc) or 24.0)
      va = float(cam.GetVerticalApertureAttr().Get(tc) or 20.25)
      ha = float(cam.GetHorizontalApertureAttr().Get(tc) or (va * 16.0 / 9.0))
      if target_aspect is not None and target_aspect > 0.0:
        eff_va = ha / float(target_aspect)
      else:
        eff_va = va
      fovy = 2.0 * math.degrees(math.atan(0.5 * eff_va / max(fl, 1e-4)))
      clip = cam.GetClippingRangeAttr().Get(tc)
      zn, zf = (float(clip[0]) * meters_per_unit, float(clip[1]) * meters_per_unit) if clip is not None else (0.1, 1000.0)
      cameras.append(OpenUSDCamera(
          name=pstr, cam_xpos=pos, cam_xmat=R_cam, cam_fovy=fovy,
          znear=max(0.05, zn), zfar=max(200.0, zf),
          focal_length=fl, horizontal_aperture=ha, vertical_aperture=va,
      ))
      continue

    type_name = prim.GetTypeName()
    if type_name in ("DistantLight", "SphereLight", "RectLight", "DiskLight", "CylinderLight", "DomeLight"):
      M = np.asarray(xcache.GetLocalToWorldTransform(prim), dtype=np.float32)
      col_attr = prim.GetAttribute("inputs:color") or prim.GetAttribute("color")
      cval = col_attr.Get(tc) if col_attr and col_attr.HasAuthoredValueOpinion() else None
      rgb = (float(cval[0]), float(cval[1]), float(cval[2])) if cval is not None else (0.9, 0.88, 0.82)
      if type_name == "DomeLight":
        sky_rgb1 = (0.35 * rgb[0], 0.42 * rgb[1], 0.54 * rgb[2])
        light_ambient = (0.28 * rgb[0], 0.30 * rgb[1], 0.34 * rgb[2])
        tex_attr = prim.GetAttribute("inputs:texture:file")
        if tex_attr and tex_attr.Get() is not None and dome_tex is None:
          ap = getattr(tex_attr.Get(), "resolvedPath", "") or getattr(tex_attr.Get(), "path", str(tex_attr.Get()))
          if ap and (ap.lower().endswith(".exr") or ap.lower().endswith(".png") or ap.lower().endswith(".hdr")):
            if not os.path.isabs(ap):
              ap = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(stage_path)), ap))
            if os.path.exists(ap):
              dome_tex = ap
              rot_attr = prim.GetAttribute("xformOp:rotateXYZ")
              if rot_attr and rot_attr.Get() is not None:
                dome_rot_y = float(rot_attr.Get()[1])
      else:
        pos = M[3, :3] * meters_per_unit
        if type_name == "DistantLight":
          dir_back = M[2, :3]
          pos = dir_back * 40.0
        if y_to_z:
          pos = _y_up_to_z_up_points(pos[None, :])[0]
        if light_pos is None or prim.GetName() == "sun_quad_llc":
          light_pos = pos
          light_diffuse = rgb
      continue

    mdata = _extract_mesh_prim(prim, local_only=False)
    if mdata is not None:
      if separate_water and mdata.material_tag == "water":
        water_meshes.append(mdata)
      else:
        unique_meshes.append(mdata)

  # 2. Traverse USD Native Prototypes (for instanceable=true prims like isBeach, osOcean, isMountainA/B, isBayCedarA1, ...)
  instanced_groups: List[OpenUSDPointInstancerGroup] = []
  n_usd_instances = 0
  for pr in usd_protos:
    ppath = str(pr.GetPath())
    users = proto_users.get(ppath, [])
    if not users:
      continue
    n_usd_instances += len(users)
    u0_name = users[0][0]
    static_parts: List[OpenUSDMeshData] = []
    for sp in Usd.PrimRange(pr):
      if _is_under_pi_proto(str(sp.GetPath())):
        continue
      if len(users) == 1:
        md = _extract_mesh_prim(sp, parent_M=users[0][1], root_override=pr, local_only=False, name_prefix=u0_name + "::")
        if md is not None:
          if separate_water and md.material_tag == "water":
            water_meshes.append(md)
          else:
            unique_meshes.append(md)
      else:
        # Local transform inside prototype
        md = _extract_mesh_prim(sp, local_only=False, root_override=pr, name_prefix=u0_name + "::")
        if md is not None:
          static_parts.append(md)

    if len(users) > 1 and static_parts:
      merged_parts = _merge_meshes_by_material(static_parts, max_buckets=3)
      pos_arr = np.stack([M_u[3, :3] for _, M_u in users], axis=0).astype(np.float32)
      rot_arr = np.stack([M_u[:3, :3].T for _, M_u in users], axis=0).astype(np.float32)
      if y_to_z:
        pos_arr = _y_up_to_z_up_points(pos_arr)
        rot_arr = _y_up_to_z_up_mat3_batch(rot_arr)
      mcode = _morton3d_20bit(pos_arr)
      order = np.argsort(mcode, kind="stable")
      pos_arr, rot_arr = pos_arr[order], rot_arr[order]
      for bi, pm in enumerate(merged_parts):
        instanced_groups.append(OpenUSDPointInstancerGroup(
            name=f"{u0_name}_usdinst_b{bi}",
            prototype_mesh=pm,
            inst_xpos=pos_arr,
            inst_xmat=rot_arr,
        ))

  # 3. Parse ALL PointInstancers (both top-level AND inside USD Native Prototypes)
  all_pi_jobs: List[Tuple["Usd.Prim", List[Tuple[str, np.ndarray]]]] = []
  for iprim in top_instancers:
    all_pi_jobs.append((iprim, [(str(iprim.GetPath()), np.eye(4, dtype=np.float64))]))
  for pr in usd_protos:
    users = proto_users.get(str(pr.GetPath()), [])
    if not users:
      continue
    for sp in Usd.PrimRange(pr):
      if sp.IsA(UsdGeom.PointInstancer) and not _is_under_pi_proto(str(sp.GetPath())):
        all_pi_jobs.append((sp, users))

  # Deduplicate identical PointInstancer prototypes across elements (e.g., isCoral1..5, isNaupakaA1..3)
  dedup_pi_buckets: Dict[Tuple[str, int, int], Dict[str, Any]] = {}
  n_point_instances = 0

  for iprim, users in all_pi_jobs:
    pi = UsdGeom.PointInstancer(iprim)
    if users[0][0] == str(iprim.GetPath()):
      M_pi_loc = np.asarray(xcache.GetLocalToWorldTransform(iprim), dtype=np.float64)
    else:
      M_pi_loc = np.asarray(UsdGeom.Xformable(iprim).ComputeLocalToWorldTransform(tc), dtype=np.float64)
    M_pi_loc[3, :3] *= meters_per_unit

    targets = pi.GetPrototypesRel().GetTargets()
    proto_idx_raw = pi.GetProtoIndicesAttr().Get(tc)
    pos_raw = pi.GetPositionsAttr().Get(tc)
    if pos_raw is None or len(pos_raw) == 0 or len(targets) == 0 or proto_idx_raw is None:
      continue
    proto_idx_arr = np.asarray(proto_idx_raw, dtype=np.int32)
    pos_arr = np.asarray(pos_raw, dtype=np.float32) * meters_per_unit
    K = pos_arr.shape[0]
    orients_attr = pi.GetOrientationsAttr().Get(tc)
    scales_attr = pi.GetScalesAttr().Get(tc)
    rots = np.broadcast_to(np.eye(3, dtype=np.float32)[None, :, :], (K, 3, 3)).copy()
    if orients_attr is not None and len(orients_attr) == K:
      rots = _quats_to_mat3_batch(orients_attr)
    if scales_attr is not None and len(scales_attr) == K:
      sc_np = np.asarray(scales_attr, dtype=np.float32)
      rots = rots * sc_np[:, None, :]

    # Extract each prototype mesh in local coords (merging multi-part proto subtrees cleanly)
    n_t = len(targets)
    proto_meshes: List[Optional[OpenUSDMeshData]] = []
    for p_idx, tgt in enumerate(targets):
      proto_prim = stage.GetPrimAtPath(tgt)
      p_verts, p_faces, p_norms = [], [], []
      voff = 0
      p_rgba = np.array([0.74, 0.72, 0.68, 1.0], dtype=np.float32)
      p_rough, p_metal, p_tag = 0.55, 0.0, "default"
      for sub in Usd.PrimRange(proto_prim):
        if sub.IsA(UsdGeom.PointInstancer):
          continue
        md = _extract_mesh_prim(sub, local_only=True, name_prefix=users[0][0] + "::")
        if md is not None:
          p_verts.append(md.verts)
          p_faces.append(md.faces + voff)
          p_norms.append(md.vert_normals)
          voff += md.verts.shape[0]
          p_rgba, p_rough, p_metal, p_tag = md.rgba, md.roughness, md.metallic, md.material_tag
      if not p_verts:
        proto_meshes.append(None)
      else:
        vb = np.vstack(p_verts)
        fb = np.vstack(p_faces)
        diag = float(np.linalg.norm(vb.max(axis=0) - vb.min(axis=0)))
        if min_instance_diag_m > 0.0 and diag < min_instance_diag_m:
          proto_meshes.append(None)
        else:
          proto_meshes.append(OpenUSDMeshData(
              name=f"{users[0][0]}/{iprim.GetName()}::proto_{p_idx}",
              verts=vb,
              faces=fb,
              vert_normals=np.vstack(p_norms),
              rgba=p_rgba,
              roughness=p_rough,
              metallic=p_metal,
              material_tag=p_tag,
          ))

    # Consolidate excessive sub-variants (> max_variants_per_pi) for small props to keep BLAS count compact
    valid_p = [i for i, pm in enumerate(proto_meshes) if pm is not None]
    if not valid_p:
      continue
    if max_variants_per_pi > 0 and len(valid_p) > max_variants_per_pi:
      # Keep up to max_variants_per_pi most populated prototypes (or all if >= 50k tris each)
      large_p = [i for i in valid_p if proto_meshes[i].faces.shape[0] >= 50_000]  # type: ignore
      if len(large_p) < len(valid_p):
        counts_p = sorted([(int(np.sum(proto_idx_arr == i)), i) for i in valid_p], reverse=True)
        keep_p = [i for _, i in counts_p[:max_variants_per_pi]]
        remap = {i: keep_p[idx % len(keep_p)] for idx, i in enumerate(valid_p)}
        proto_idx_arr = np.asarray([remap.get(int(x), keep_p[0]) for x in proto_idx_arr], dtype=np.int32)
        valid_p = keep_p

    # Transform instances by M_pi_loc and each USD prototype user's M_u
    for u_name, M_u in users:
      M_tot = M_pi_loc @ M_u
      pos_w = _transform_points_gf(pos_arr, M_tot)
      rots_w = np.einsum("ji,bjk->bik", M_tot[:3, :3].astype(np.float32), rots).astype(np.float32)
      if y_to_z:
        pos_w = _y_up_to_z_up_points(pos_w)
        rots_w = _y_up_to_z_up_mat3_batch(rots_w)

      for p_idx in valid_p:
        pm = proto_meshes[p_idx]
        if pm is None:
          continue
        mask = proto_idx_arr == p_idx
        if not np.any(mask):
          continue
        n_m = int(np.sum(mask))
        n_point_instances += n_m
        dkey = (int(pm.verts.shape[0]), int(pm.faces.shape[0]), pm.material_tag)
        b = dedup_pi_buckets.get(dkey)
        if b is None:
          dedup_pi_buckets[dkey] = {
              "proto": pm,
              "pos_list": [pos_w[mask]],
              "rot_list": [rots_w[mask]],
          }
        else:
          b["pos_list"].append(pos_w[mask])
          b["rot_list"].append(rots_w[mask])

  for dkey, b in dedup_pi_buckets.items():
    pos_cat = np.concatenate(b["pos_list"], axis=0)
    rot_cat = np.concatenate(b["rot_list"], axis=0)
    # 3D Morton spatial sorting for tight Wide-TLAS-64 clusters!
    order = np.argsort(_morton3d_20bit(pos_cat), kind="stable")
    instanced_groups.append(OpenUSDPointInstancerGroup(
        name=b["proto"].name,
        prototype_mesh=b["proto"],
        inst_xpos=pos_cat[order],
        inst_xmat=rot_cat[order],
    ))

  if cameras:
    camera = cameras[0]
    if preferred_camera is not None:
      for c in cameras:
        if c.name.endswith("/" + preferred_camera) or c.name == preferred_camera:
          camera = c
          break
  else:
    persp = stage.GetRootLayer().customLayerData.get("cameraSettings", {}).get("Perspective", {})
    if "position" in persp and "target" in persp:
      cpos = np.asarray(persp["position"], dtype=np.float32) * meters_per_unit
      ctgt = np.asarray(persp["target"], dtype=np.float32) * meters_per_unit
      if y_to_z:
        cpos = _y_up_to_z_up_points(cpos[None, :])[0]
        ctgt = _y_up_to_z_up_points(ctgt[None, :])[0]
      fwd = ctgt - cpos
      fwd /= max(float(np.linalg.norm(fwd)), 1e-8)
      right = np.cross(fwd, np.array([0.0, 0.0, 1.0], dtype=np.float32))
      if np.linalg.norm(right) < 1e-6:
        right = np.cross(fwd, np.array([0.0, 1.0, 0.0], dtype=np.float32))
      right /= max(float(np.linalg.norm(right)), 1e-8)
      up = np.cross(right, fwd)
      cmat = np.stack([right, up, -fwd], axis=1).astype(np.float32)
      camera = OpenUSDCamera("omniverse_persp_cam", cpos, cmat, cam_fovy=48.0, znear=0.05, zfar=200.0)
    else:
      all_pts = [m.verts for m in unique_meshes]
      if all_pts:
        pts_cat = np.concatenate(all_pts, axis=0)
        mn, mx = pts_cat.min(axis=0), pts_cat.max(axis=0)
      else:
        mn, mx = np.array([-1.0, -1.0, 0.0]), np.array([1.0, 1.0, 2.0])
      ctr = 0.5 * (mn + mx)
      diag = float(np.linalg.norm(mx - mn))
      cpos = ctr + np.array([0.65 * diag, -0.65 * diag, 0.25 * diag], dtype=np.float32)
      fwd = ctr - cpos
      fwd /= max(float(np.linalg.norm(fwd)), 1e-8)
      right = np.cross(fwd, np.array([0.0, 0.0, 1.0], dtype=np.float32))
      right /= max(float(np.linalg.norm(right)), 1e-8)
      up = np.cross(right, fwd)
      cmat = np.stack([right, up, -fwd], axis=1).astype(np.float32)
      camera = OpenUSDCamera("auto_cam", cpos, cmat, cam_fovy=50.0, znear=0.05, zfar=max(100.0, 5.0 * diag))

  if light_pos is None:
    light_pos = camera.cam_xpos + np.array([0.0, 0.0, 10.0], dtype=np.float32)

  return OpenUSDStageData(
      stage_path=stage_path,
      up_axis=up_axis,
      meters_per_unit=meters_per_unit,
      unique_meshes=unique_meshes,
      instanced_groups=instanced_groups,
      camera=camera,
      light_pos=light_pos.astype(np.float32),
      light_diffuse=light_diffuse,
      light_ambient=light_ambient,
      sky_rgb1=sky_rgb1,
      sky_rgb2=sky_rgb2,
      water_meshes=water_meshes,
      dome_texture_path=dome_tex,
      dome_rotate_y_deg=dome_rot_y,
      n_usd_instances=n_usd_instances,
      n_point_instances=n_point_instances,
  )


def _load_stage_usda_fallback(stage_path: str, convert_y_up_to_z_up: bool = True) -> OpenUSDStageData:
  """Pure-Python lightweight .usda text parser fallback when pxr is unavailable."""
  with open(stage_path, "r", encoding="utf-8") as f:
    txt = f.read()
  m_up = re.search(r'upAxis\s*=\s*"([YZ])"', txt)
  up_axis = m_up.group(1) if m_up else "Y"
  y_to_z = convert_y_up_to_z_up and (up_axis == "Y")

  meshes: List[OpenUSDMeshData] = []
  for m in re.finditer(r'def\s+Mesh\s+"([^"]+)"\s*\{(.*?)\n\s*\}', txt, re.S):
    name, body = m.group(1), m.group(2)
    m_pts = re.search(r'point3f\[\]\s+points\s*=\s*\[(.*?)\]', body, re.S)
    m_cnt = re.search(r'int\[\]\s+faceVertexCounts\s*=\s*\[(.*?)\]', body, re.S)
    m_idx = re.search(r'int\[\]\s+faceVertexIndices\s*=\s*\[(.*?)\]', body, re.S)
    if not (m_pts and m_cnt and m_idx):
      continue
    pts = np.asarray(re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', m_pts.group(1)), dtype=np.float32).reshape(-1, 3)
    cnt = np.asarray(re.findall(r'-?\d+', m_cnt.group(1)), dtype=np.int32)
    idx = np.asarray(re.findall(r'-?\d+', m_idx.group(1)), dtype=np.int32)
    faces = _triangulate_faces(cnt, idx)
    if y_to_z:
      pts = _y_up_to_z_up_points(pts)
    meshes.append(OpenUSDMeshData(
        name=name,
        verts=pts,
        faces=faces,
        vert_normals=_compute_smooth_normals(pts, faces),
        rgba=np.array([0.74, 0.72, 0.68, 1.0], dtype=np.float32),
    ))
  if not meshes:
    raise RuntimeError(f"Pure-Python .usda fallback found no Mesh prims in {stage_path}")
  pts_cat = np.concatenate([m.verts for m in meshes], axis=0)
  mn, mx = pts_cat.min(axis=0), pts_cat.max(axis=0)
  ctr = 0.5 * (mn + mx)
  diag = float(np.linalg.norm(mx - mn))
  cpos = ctr + np.array([0.65 * diag, -0.65 * diag, 0.25 * diag], dtype=np.float32)
  fwd = ctr - cpos; fwd /= max(float(np.linalg.norm(fwd)), 1e-8)
  right = np.cross(fwd, np.array([0.0, 0.0, 1.0], dtype=np.float32)); right /= max(float(np.linalg.norm(right)), 1e-8)
  up = np.cross(right, fwd)
  cmat = np.stack([right, up, -fwd], axis=1).astype(np.float32)
  return OpenUSDStageData(
      stage_path=stage_path,
      up_axis=up_axis,
      meters_per_unit=1.0,
      unique_meshes=meshes,
      instanced_groups=[],
      camera=OpenUSDCamera("fallback_cam", cpos, cmat, 50.0, 0.05, max(100.0, 5.0 * diag)),
      light_pos=(cpos + np.array([0.0, 0.0, 10.0], dtype=np.float32)),
      light_diffuse=(0.85, 0.83, 0.80),
      light_ambient=(0.28, 0.30, 0.34),
      sky_rgb1=(0.28, 0.36, 0.48),
      sky_rgb2=(0.10, 0.14, 0.20),
  )


def load_openusd_stage(
    stage_path: str,
    time_code: Optional[float] = None,
    convert_y_up_to_z_up: bool = True,
    preferred_camera: Optional[str] = None,
    target_aspect: Optional[float] = None,
    include_curves: bool = False,
    max_curve_tris: int = 2_000_000,
    max_variants_per_pi: int = 4,
    min_instance_diag_m: float = 0.0,
    separate_water: bool = False,
) -> OpenUSDStageData:
  """Load .usd / .usda / .usdc / .usdz into structured OpenUSDStageData."""
  if HAS_PXR:
    return _load_stage_via_pxr(
        stage_path,
        time_code=time_code,
        convert_y_up_to_z_up=convert_y_up_to_z_up,
        preferred_camera=preferred_camera,
        target_aspect=target_aspect,
        include_curves=include_curves,
        max_curve_tris=max_curve_tris,
        max_variants_per_pi=max_variants_per_pi,
        min_instance_diag_m=min_instance_diag_m,
        separate_water=separate_water,
    )
  if stage_path.lower().endswith(".usda"):
    return _load_stage_usda_fallback(stage_path, convert_y_up_to_z_up=convert_y_up_to_z_up)
  raise ImportError("pxr (usd-core) is required to load binary .usd/.usdc/.usdz stages.")


def compile_openusd_to_wide_bvh(
    stage_path: str,
    height: int = 2160,
    width: int = 3840,
    build_method: str = "spatial_l1_morton",
    time_code: Optional[float] = None,
    use_shadows: bool = True,
    subdivide_k: int = 1,
    devices: Optional[Sequence[Any]] = None,
    max_unique_bvh_meshes: int = 16,
    preferred_camera: Optional[str] = None,
    include_curves: bool = False,
    max_curve_tris: int = 2_000_000,
    max_variants_per_pi: int = 4,
    min_instance_diag_m: float = 0.0,
    separate_water: bool = False,
) -> Dict[str, Any]:
  """Compile any OpenUSD stage directly into CompactWideBVHSceneSpec + TPU HBM buffers."""
  import jax
  import jax.numpy as jnp
  from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
  from wave3_500m_10b_8k import CompactWideBVHMeshItem, CompactWideBVHSceneSpec, build_compact_bvh_mesh_item

  st = load_openusd_stage(
      stage_path,
      time_code=time_code,
      preferred_camera=preferred_camera,
      target_aspect=float(width) / float(height),
      include_curves=include_curves,
      max_curve_tris=max_curve_tris,
      max_variants_per_pi=max_variants_per_pi,
      min_instance_diag_m=min_instance_diag_m,
      separate_water=separate_water,
  )
  unique_meshes = _merge_meshes_by_material(st.unique_meshes, max_buckets=max_unique_bvh_meshes)
  water_meshes = _merge_meshes_by_material(st.water_meshes, max_buckets=1) if st.water_meshes else []

  def _subdivide_if_needed(md: OpenUSDMeshData) -> Dict[str, Any]:
    v, f, n = md.verts, md.faces, md.vert_normals
    if subdivide_k > 1:
      from run_wave3_suite import subdivide_mesh_barycentric
      v, f, n = subdivide_mesh_barycentric(v, f, subdivide_k)
    return {"name": md.name, "verts": v, "faces": f, "vert_normals": n, "rgba": md.rgba}

  rep = None
  if devices is not None and len(devices) > 1:
    mesh_sh = Mesh(np.array(devices), ("spatial",))
    rep = NamedSharding(mesh_sh, P())

  def _put(arr):
    if rep is not None:
      return jax.device_put(arr, rep)
    return jnp.asarray(arr, dtype=jnp.float32)

  def _strip(it: CompactWideBVHMeshItem) -> CompactWideBVHMeshItem:
    empty = np.empty((0,), dtype=np.float32)
    return dataclasses.replace(it, l2_roots_aabb=empty, l2_aabbs=empty, l0_aabbs=empty, leaf_bank=empty)

  unique_items_meta = []
  unique_dev_arrays = []
  unique_pbr = []
  hbm_bytes = 0
  for mid, md in enumerate(unique_meshes):
    it = build_compact_bvh_mesh_item(mid, _subdivide_if_needed(md), build_method=build_method)
    hbm_bytes += it.l2_roots_aabb.nbytes + it.l2_aabbs.nbytes + it.l0_aabbs.nbytes + it.leaf_bank.nbytes
    unique_dev_arrays.append((_put(it.l2_roots_aabb), _put(it.l2_aabbs), _put(it.l0_aabbs), _put(it.leaf_bank)))
    unique_items_meta.append(_strip(it))
    unique_pbr.append((float(md.roughness), float(md.metallic), str(md.material_tag)))

  water_items_meta = []
  water_dev_arrays = []
  for wid, md in enumerate(water_meshes):
    it = build_compact_bvh_mesh_item(wid, _subdivide_if_needed(md), build_method=build_method)
    hbm_bytes += it.l2_roots_aabb.nbytes + it.l2_aabbs.nbytes + it.l0_aabbs.nbytes + it.leaf_bank.nbytes
    water_dev_arrays.append((_put(it.l2_roots_aabb), _put(it.l2_aabbs), _put(it.l0_aabbs), _put(it.leaf_bank)))
    water_items_meta.append(_strip(it))

  inst_items_meta = []
  inst_dev_arrays = []
  inst_xpos_list = []
  inst_xmat_list = []
  inst_pbr = []
  for gid, grp in enumerate(st.instanced_groups):
    it = build_compact_bvh_mesh_item(gid, _subdivide_if_needed(grp.prototype_mesh), build_method=build_method)
    hbm_bytes += it.l2_roots_aabb.nbytes + it.l2_aabbs.nbytes + it.l0_aabbs.nbytes + it.leaf_bank.nbytes
    inst_dev_arrays.append((_put(it.l2_roots_aabb), _put(it.l2_aabbs), _put(it.l0_aabbs), _put(it.leaf_bank)))
    inst_items_meta.append(_strip(it))
    ixp = grp.inst_xpos[None, :, :].astype(np.float32)
    ixm = grp.inst_xmat[None, :, :, :].astype(np.float32)
    hbm_bytes += ixp.nbytes + ixm.nbytes
    inst_xpos_list.append(jax.device_put(ixp, rep) if rep is not None else jnp.asarray(ixp))
    inst_xmat_list.append(jax.device_put(ixm, rep) if rep is not None else jnp.asarray(ixm))
    inst_pbr.append((float(grp.prototype_mesh.roughness), float(grp.prototype_mesh.metallic), str(grp.prototype_mesh.material_tag)))

  def _mk_spec(meshes_tuple):
    return CompactWideBVHSceneSpec(
        height=height,
        width=width,
        cam_fovy=float(st.camera.cam_fovy),
        cam_projection=0,
        cam_sensorsize=(0.0, 0.0),
        cam_intrinsic=(0.0, 0.0, 0.0, 0.0),
        znear=float(st.camera.znear),
        zfar=float(st.camera.zfar),
        meshes=tuple(meshes_tuple),
        light_pos=st.light_pos,
        light_ambient=st.light_ambient,
        light_diffuse=st.light_diffuse,
        light_specular=(0.30, 0.30, 0.30),
        sky_rgb1=st.sky_rgb1,
        sky_rgb2=st.sky_rgb2,
        use_shadows=use_shadows,
    )

  n_u = len(unique_items_meta)
  oxp = np.zeros((1, max(n_u, 1), 3), dtype=np.float32)
  oxm = np.broadcast_to(np.eye(3, dtype=np.float32)[None, None, :, :], (1, max(n_u, 1), 3, 3)).copy()
  cxp = st.camera.cam_xpos[None, :].astype(np.float32)
  cxm = st.camera.cam_xmat[None, :, :].astype(np.float32)

  return {
      "stage": st,
      "unique_spec": _mk_spec(unique_items_meta) if unique_items_meta else None,
      "unique_dev_arrays": tuple(unique_dev_arrays),
      "unique_pbr": unique_pbr,
      "water_spec": _mk_spec(water_items_meta) if water_items_meta else None,
      "water_dev_arrays": tuple(water_dev_arrays),
      "obj_xpos": jax.device_put(oxp, rep) if rep is not None else jnp.asarray(oxp),
      "obj_xmat": jax.device_put(oxm, rep) if rep is not None else jnp.asarray(oxm),
      "inst_spec": _mk_spec(inst_items_meta) if inst_items_meta else None,
      "inst_dev_arrays": tuple(inst_dev_arrays),
      "inst_pbr": inst_pbr,
      "inst_xpos": tuple(inst_xpos_list),
      "inst_xmat": tuple(inst_xmat_list),
      "cam_xpos": jax.device_put(cxp, rep) if rep is not None else jnp.asarray(cxp),
      "cam_xmat": jax.device_put(cxm, rep) if rep is not None else jnp.asarray(cxm),
      "hbm_bytes": hbm_bytes,
      "unique_tris": sum(it.n_tris_orig for it in unique_items_meta) + sum(it.n_tris_orig for it in water_items_meta),
      "inst_blas_tris": sum(it.n_tris_orig for it in inst_items_meta),
      "inst_effective_tris": sum(it.n_tris_orig * grp.inst_xpos.shape[0] for it, grp in zip(inst_items_meta, st.instanced_groups)),
  }
