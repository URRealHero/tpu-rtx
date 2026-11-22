"""Native OpenUSD (.usd/.usda/.usdc/.usdz) -> TPU Wide-BVH-64 Compiler.

Auto-compiles OpenUSD stages directly into Wide-BVH-64 TPU buffers with zero
per-scene custom export scripts:
  - UsdGeom.Mesh (fan triangulation for quads/ngons, normals, UVs) & analytic
    UsdGeom prims (Sphere, Cube, Cylinder, Cone)
  - UsdGeom.XformCache world transforms + automatic Y-up -> Z-up stage axis
    conversion
  - UsdGeom.PointInstancer -> 2-Level TLAS/BLAS buffers (prototypes as shared
    Compact BLAS + batched world poses/rotations)
  - UsdShade MaterialBindingAPI -> UsdPreviewSurface / OmniPBR / UsdUVTexture
    (average/sampled diffuse RGBA + texture extraction, including .usdz packages)
  - UsdLux (DistantLight, SphereLight, RectLight, DomeLight) -> lighting & sky
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


def _y_up_to_z_up_points(pts: np.ndarray) -> np.ndarray:
  """Rotate Y-up OpenUSD coords (x, y, z) -> Z-up (x, -z, y)."""
  if pts.size == 0:
    return pts.astype(np.float32)
  return np.stack([pts[:, 0], -pts[:, 2], pts[:, 1]], axis=-1).astype(np.float32)


def _y_up_to_z_up_mat3(R: np.ndarray) -> np.ndarray:
  """Conjugate 3x3 rotation/linear matrix from Y-up to Z-up: P @ R @ P^T."""
  P = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float32)
  return (P @ R @ P.T).astype(np.float32)


def _compute_smooth_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
  if verts.shape[0] == 0 or faces.shape[0] == 0:
    return np.zeros_like(verts, dtype=np.float32)
  v0 = verts[faces[:, 0]]
  v1 = verts[faces[:, 1]]
  v2 = verts[faces[:, 2]]
  fn = np.cross(v1 - v0, v2 - v0)
  vn = np.zeros_like(verts, dtype=np.float32)
  np.add.at(vn, faces[:, 0], fn)
  np.add.at(vn, faces[:, 1], fn)
  np.add.at(vn, faces[:, 2], fn)
  norms = np.linalg.norm(vn, axis=-1, keepdims=True)
  bad = norms[:, 0] < 1e-12
  if np.any(bad):
    vn[bad] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    norms = np.linalg.norm(vn, axis=-1, keepdims=True)
  return (vn / np.maximum(norms, 1e-12)).astype(np.float32)


def _triangulate_faces(counts: np.ndarray, indices: np.ndarray) -> np.ndarray:
  """Fan-triangulate arbitrary polygon counts into (F_tri, 3) int32."""
  if counts.size == 0 or indices.size == 0:
    return np.zeros((0, 3), dtype=np.int32)
  if np.all(counts == 3):
    return indices.reshape(-1, 3).astype(np.int32)
  if np.all(counts == 4):
    q = indices.reshape(-1, 4)
    t1 = q[:, [0, 1, 2]]
    t2 = q[:, [0, 2, 3]]
    return np.concatenate([t1, t2], axis=0).astype(np.int32)
  tris = []
  off = 0
  for c in counts:
    c = int(c)
    if c >= 3:
      v0 = int(indices[off])
      for k in range(1, c - 1):
        tris.append((v0, int(indices[off + k]), int(indices[off + k + 1])))
    off += c
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


def _load_texture_rgb(stage_path: str, asset_path: str) -> Optional[np.ndarray]:
  if Image is None or not asset_path:
    return None
  resolved = asset_path
  if not os.path.isabs(resolved):
    resolved = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(stage_path)), asset_path))
  try:
    if os.path.exists(resolved):
      with Image.open(resolved) as im:
        return np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    if stage_path.lower().endswith(".usdz") and os.path.exists(stage_path):
      with zipfile.ZipFile(stage_path, "r") as zf:
        rel = asset_path.lstrip("./")
        for member in zf.namelist():
          if member.endswith(rel):
            with zf.open(member) as mf:
              with Image.open(mf) as im:
                return np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
  except Exception:
    return None
  return None


def _extract_shader_rgba(stage_path: str, prim: "Usd.Prim") -> Tuple[np.ndarray, Optional[np.ndarray]]:
  """Extract (rgba[4], optional texture_rgb) from UsdShade MaterialBindingAPI (UsdPreviewSurface / OmniPBR / UsdUVTexture)."""
  default_rgba = np.array([0.74, 0.72, 0.68, 1.0], dtype=np.float32)
  tex_rgb: Optional[np.ndarray] = None
  try:
    mat_api = UsdShade.MaterialBindingAPI(prim)
    mat, _ = mat_api.ComputeBoundMaterial()
    if not mat:
      # Fallback for assets (e.g., RoboWild Sketchfab conversions) that author rel material:binding without MaterialBindingAPI
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
          return np.array([float(c[0]), float(c[1]), float(c[2]), 1.0], dtype=np.float32), None
      return default_rgba, None

    surf_out = mat.GetSurfaceOutput()
    if not surf_out or not surf_out.HasConnectedSource():
      surf_out = mat.GetSurfaceOutput("mdl")
    if surf_out and surf_out.HasConnectedSource():
      src_info = surf_out.GetConnectedSource()
      shader_prim = src_info[0].GetPrim() if src_info else None
      if shader_prim:
        shader = UsdShade.Shader(shader_prim)
        alpha = 1.0
        op_in = shader.GetInput("opacity")
        if op_in and op_in.Get() is not None:
          alpha = float(op_in.Get())
        for cname in ("diffuseColor", "diffuse_color_constant", "albedo_color", "baseColor", "diffuse_tint"):
          cin = shader.GetInput(cname)
          if not cin:
            continue
          if cin.HasConnectedSource():
            tex_src = cin.GetConnectedSource()
            tex_prim = tex_src[0].GetPrim() if tex_src else None
            if tex_prim:
              tex_shader = UsdShade.Shader(tex_prim)
              fin = tex_shader.GetInput("file")
              if fin and fin.Get() is not None:
                asset = fin.Get()
                apath = getattr(asset, "resolvedPath", "") or getattr(asset, "path", str(asset))
                tex_rgb = _load_texture_rgb(stage_path, apath)
                if tex_rgb is not None:
                  mean_c = np.mean(tex_rgb.reshape(-1, 3), axis=0)
                  return np.array([float(mean_c[0]), float(mean_c[1]), float(mean_c[2]), alpha], dtype=np.float32), tex_rgb
          val = cin.Get()
          if val is not None:
            return np.array([float(val[0]), float(val[1]), float(val[2]), alpha], dtype=np.float32), tex_rgb
        # Also check direct texture asset inputs (OmniPBR diffuse_texture / albedo_tex)
        for tname in ("diffuse_texture", "albedo_tex", "baseColorTexture"):
          tin = shader.GetInput(tname)
          if tin and tin.Get() is not None:
            asset = tin.Get()
            apath = getattr(asset, "resolvedPath", "") or getattr(asset, "path", str(asset))
            tex_rgb = _load_texture_rgb(stage_path, apath)
            if tex_rgb is not None:
              mean_c = np.mean(tex_rgb.reshape(-1, 3), axis=0)
              return np.array([float(mean_c[0]), float(mean_c[1]), float(mean_c[2]), alpha], dtype=np.float32), tex_rgb
  except Exception:
    pass
  return default_rgba, tex_rgb


def _transform_points_gf(pts: np.ndarray, mat_4x4: np.ndarray) -> np.ndarray:
  """Apply 4x4 row-vector USD matrix (p_row @ M) to (N, 3) points."""
  if pts.size == 0:
    return pts.astype(np.float32)
  M = np.asarray(mat_4x4, dtype=np.float64)
  out = pts.astype(np.float64) @ M[:3, :3] + M[3:4, :3]
  return out.astype(np.float32)


def _quat_to_mat3(q: Any) -> np.ndarray:
  """Convert USD Gf.Quath/Quatf/Quatd (real, imag[3]) -> (3, 3) column-vector rotation matrix."""
  r = float(q.GetReal())
  im = q.GetImaginary()
  x, y, z = float(im[0]), float(im[1]), float(im[2])
  xx, yy, zz = x * x, y * y, z * z
  xy, xz, yz = x * y, x * z, y * z
  wx, wy, wz = r * x, r * y, r * z
  return np.array([
      [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy)],
      [2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
      [2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy)],
  ], dtype=np.float32)


def _load_stage_via_pxr(
    stage_path: str,
    time_code: Optional[float] = None,
    convert_y_up_to_z_up: bool = True,
) -> OpenUSDStageData:
  stage = Usd.Stage.Open(stage_path)
  if stage is None:
    raise RuntimeError(f"Failed to open USD stage: {stage_path}")
  tc = Usd.TimeCode.Default() if time_code is None else Usd.TimeCode(float(time_code))
  up_axis = str(UsdGeom.GetStageUpAxis(stage) or "Y")
  meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage) or 1.0)
  y_to_z = convert_y_up_to_z_up and (up_axis.upper() == "Y")
  xcache = UsdGeom.XformCache(tc)

  # Collect prototypes referenced by PointInstancers so we do not duplicate them as static meshes
  instancer_prims = []
  proto_paths = set()
  for prim in stage.Traverse():
    if prim.IsA(UsdGeom.PointInstancer):
      instancer_prims.append(prim)
      pi = UsdGeom.PointInstancer(prim)
      for tgt in pi.GetPrototypesRel().GetTargets():
        proto_paths.add(str(tgt))

  def _is_under_proto(prim_path: str) -> bool:
    return any(prim_path == p or prim_path.startswith(p + "/") for p in proto_paths)

  unique_meshes: List[OpenUSDMeshData] = []
  cameras: List[OpenUSDCamera] = []
  light_pos: Optional[np.ndarray] = None
  light_diffuse = (0.85, 0.83, 0.80)
  light_ambient = (0.28, 0.30, 0.34)
  sky_rgb1 = (0.28, 0.36, 0.48)
  sky_rgb2 = (0.10, 0.14, 0.20)

  def _extract_mesh_prim(prim: "Usd.Prim", local_only: bool = False) -> Optional[OpenUSDMeshData]:
    rgba, tex_rgb = _extract_shader_rgba(stage_path, prim)
    verts = None
    faces = None
    vnorm = None
    uvs = None

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
      M = np.asarray(xcache.GetLocalToWorldTransform(prim), dtype=np.float64)
      M_scaled = M.copy()
      M_scaled[3, :3] *= meters_per_unit
      verts = _transform_points_gf(verts, M_scaled)
      if vnorm is not None:
        R_n = np.linalg.inv(M[:3, :3]).T
        vnorm = (vnorm @ R_n.T).astype(np.float32)
        vnorm /= np.maximum(np.linalg.norm(vnorm, axis=-1, keepdims=True), 1e-12)

    if y_to_z:
      verts = _y_up_to_z_up_points(verts)
      if vnorm is not None:
        vnorm = _y_up_to_z_up_points(vnorm)

    if vnorm is None:
      vnorm = _compute_smooth_normals(verts, faces)

    return OpenUSDMeshData(
        name=str(prim.GetPath()),
        verts=verts,
        faces=faces,
        vert_normals=vnorm,
        rgba=rgba,
        uvs=uvs,
        texture_rgb=tex_rgb,
    )

  for prim in stage.Traverse():
    pstr = str(prim.GetPath())
    if _is_under_proto(pstr):
      continue
    if prim.IsA(UsdGeom.Camera):
      cam = UsdGeom.Camera(prim)
      M = np.asarray(xcache.GetLocalToWorldTransform(prim), dtype=np.float32)
      pos = M[3, :3] * meters_per_unit
      # USD Gf row-vector convention: local axes in world are rows of M[:3, :3]
      right = M[0, :3]
      up = M[1, :3]
      back = M[2, :3]  # USD camera looks down -Z, so +Z is -fwd
      R_cam = np.stack([right, up, back], axis=1).astype(np.float32)
      if y_to_z:
        pos = _y_up_to_z_up_points(pos[None, :])[0]
        R_cam = np.stack([
            _y_up_to_z_up_points(right[None, :])[0],
            _y_up_to_z_up_points(up[None, :])[0],
            _y_up_to_z_up_points(back[None, :])[0],
        ], axis=1).astype(np.float32)
      # Orthonormalize R_cam
      for col in range(3):
        R_cam[:, col] /= max(float(np.linalg.norm(R_cam[:, col])), 1e-8)
      fl = float(cam.GetFocalLengthAttr().Get(tc) or 24.0)
      va = float(cam.GetVerticalApertureAttr().Get(tc) or 20.25)
      fovy = 2.0 * math.degrees(math.atan(0.5 * va / max(fl, 1e-4)))
      clip = cam.GetClippingRangeAttr().Get(tc)
      zn, zf = (float(clip[0]), float(clip[1])) if clip is not None else (0.1, 100.0)
      cameras.append(OpenUSDCamera(name=pstr, cam_xpos=pos, cam_xmat=R_cam, cam_fovy=fovy, znear=zn, zfar=zf))
      continue

    if prim.HasAPI(UsdLux.LightAPI) if hasattr(UsdLux, "LightAPI") else False or prim.IsA(UsdLux.BoundableLightBase) if hasattr(UsdLux, "BoundableLightBase") else False:
      pass

    type_name = prim.GetTypeName()
    if type_name in ("DistantLight", "SphereLight", "RectLight", "DiskLight", "CylinderLight", "DomeLight"):
      M = np.asarray(xcache.GetLocalToWorldTransform(prim), dtype=np.float32)
      col_attr = prim.GetAttribute("inputs:color") or prim.GetAttribute("color")
      cval = col_attr.Get(tc) if col_attr and col_attr.HasAuthoredValueOpinion() else None
      rgb = (float(cval[0]), float(cval[1]), float(cval[2])) if cval is not None else (0.9, 0.88, 0.82)
      if type_name == "DomeLight":
        sky_rgb1 = (0.35 * rgb[0], 0.42 * rgb[1], 0.54 * rgb[2])
        light_ambient = (0.28 * rgb[0], 0.30 * rgb[1], 0.34 * rgb[2])
      else:
        pos = M[3, :3] * meters_per_unit
        if type_name == "DistantLight":
          # DistantLight shines along -Z in local frame -> place effective far light along +Z
          dir_back = M[2, :3]
          pos = dir_back * 40.0
        if y_to_z:
          pos = _y_up_to_z_up_points(pos[None, :])[0]
        if light_pos is None:
          light_pos = pos
          light_diffuse = rgb
      continue

    mdata = _extract_mesh_prim(prim, local_only=False)
    if mdata is not None:
      unique_meshes.append(mdata)

  # Parse PointInstancers -> 2-Level TLAS/BLAS groups
  instanced_groups: List[OpenUSDPointInstancerGroup] = []
  for iprim in instancer_prims:
    pi = UsdGeom.PointInstancer(iprim)
    M_pi = np.asarray(xcache.GetLocalToWorldTransform(iprim), dtype=np.float32)
    M_pi[3, :3] *= meters_per_unit
    targets = pi.GetPrototypesRel().GetTargets()
    proto_idx_arr = np.asarray(pi.GetProtoIndicesAttr().Get(tc) or [], dtype=np.int32)
    pos_arr = np.asarray(pi.GetPositionsAttr().Get(tc) or [], dtype=np.float32) * meters_per_unit
    if pos_arr.size == 0 or len(targets) == 0:
      continue
    orients_attr = pi.GetOrientationsAttr().Get(tc)
    scales_attr = pi.GetScalesAttr().Get(tc)
    K = pos_arr.shape[0]
    rots = np.broadcast_to(np.eye(3, dtype=np.float32)[None, :, :], (K, 3, 3)).copy()
    if orients_attr is not None and len(orients_attr) == K:
      for k_i in range(K):
        rots[k_i] = _quat_to_mat3(orients_attr[k_i])
    if scales_attr is not None and len(scales_attr) == K:
      sc_np = np.asarray(scales_attr, dtype=np.float32)
      rots = rots * sc_np[:, None, :]

    # Apply PointInstancer parent transform
    pos_w = _transform_points_gf(pos_arr, M_pi)
    rots_w = np.einsum("ji,bkj->bki", M_pi[:3, :3], rots).astype(np.float32)
    if y_to_z:
      pos_w = _y_up_to_z_up_points(pos_w)
      rots_w = np.stack([_y_up_to_z_up_mat3(rots_w[k_i]) for k_i in range(K)], axis=0)

    for p_idx, tgt in enumerate(targets):
      mask = proto_idx_arr == p_idx
      if not np.any(mask):
        continue
      proto_prim = stage.GetPrimAtPath(tgt)
      # Gather all meshes under prototype subtree into one combined BLAS mesh
      p_verts, p_faces, p_norms = [], [], []
      voff = 0
      p_rgba = np.array([0.74, 0.72, 0.68, 1.0], dtype=np.float32)
      for sub in Usd.PrimRange(proto_prim):
        md = _extract_mesh_prim(sub, local_only=True)
        if md is not None:
          p_verts.append(md.verts)
          p_faces.append(md.faces + voff)
          p_norms.append(md.vert_normals)
          voff += md.verts.shape[0]
          p_rgba = md.rgba
      if not p_verts:
        continue
      proto_mesh = OpenUSDMeshData(
          name=f"{iprim.GetPath()}::proto_{p_idx}",
          verts=np.vstack(p_verts),
          faces=np.vstack(p_faces),
          vert_normals=np.vstack(p_norms),
          rgba=p_rgba,
      )
      instanced_groups.append(OpenUSDPointInstancerGroup(
          name=f"{iprim.GetPath()}_p{p_idx}",
          prototype_mesh=proto_mesh,
          inst_xpos=pos_w[mask],
          inst_xmat=rots_w[mask],
      ))

  if cameras:
    camera = cameras[0]
  else:
    # Check Omniverse/IsaacLab customLayerData.cameraSettings.Perspective before falling back to auto-frame
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
      # Auto-frame camera from scene bounds
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
  )


def _load_stage_usda_fallback(stage_path: str, convert_y_up_to_z_up: bool = True) -> OpenUSDStageData:
  """Pure-Python lightweight .usda text parser fallback when pxr is unavailable."""
  with open(stage_path, "r", encoding="utf-8") as f:
    txt = f.read()
  m_up = re.search(r'upAxis\s*=\s*"([YZ])"', txt)
  up_axis = m_up.group(1) if m_up else "Y"
  y_to_z = convert_y_up_to_z_up and (up_axis == "Y")

  meshes: List[OpenUSDMeshData] = []
  # Parse def Mesh blocks
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
) -> OpenUSDStageData:
  """Load .usd / .usda / .usdc / .usdz into structured OpenUSDStageData."""
  if HAS_PXR:
    return _load_stage_via_pxr(stage_path, time_code=time_code, convert_y_up_to_z_up=convert_y_up_to_z_up)
  if stage_path.lower().endswith(".usda"):
    return _load_stage_usda_fallback(stage_path, convert_y_up_to_z_up=convert_y_up_to_z_up)
  raise ImportError("pxr (usd-core) is required to load binary .usd/.usdc/.usdz stages.")


def _merge_meshes_by_color(meshes: List[OpenUSDMeshData], max_buckets: int = 16) -> List[OpenUSDMeshData]:
  """Merge large counts of small USD prims into <= max_buckets material buckets for fast XLA compilation."""
  if len(meshes) <= max_buckets:
    return meshes
  buckets: Dict[Tuple[int, int, int], List[OpenUSDMeshData]] = {}
  for md in meshes:
    key = (int(round(float(md.rgba[0]) * 16)), int(round(float(md.rgba[1]) * 16)), int(round(float(md.rgba[2]) * 16)))
    buckets.setdefault(key, []).append(md)
  # If still > max_buckets, fold smallest keys together
  items = sorted(buckets.items(), key=lambda kv: sum(m.faces.shape[0] for m in kv[1]), reverse=True)
  if len(items) > max_buckets:
    head = items[: max_buckets - 1]
    tail_meshes = [m for _, ms in items[max_buckets - 1 :] for m in ms]
    items = head + [(items[max_buckets - 1][0], tail_meshes)]
  merged: List[OpenUSDMeshData] = []
  for b_idx, (_, ms) in enumerate(items):
    vs, fs, ns = [], [], []
    voff = 0
    w_rgba = np.zeros((4,), dtype=np.float64)
    w_tot = 0
    for m in ms:
      vs.append(m.verts)
      fs.append(m.faces + voff)
      ns.append(m.vert_normals if m.vert_normals is not None else _compute_smooth_normals(m.verts, m.faces))
      voff += m.verts.shape[0]
      nf = m.faces.shape[0]
      w_rgba += np.asarray(m.rgba, dtype=np.float64) * nf
      w_tot += nf
    avg_rgba = (w_rgba / max(w_tot, 1)).astype(np.float32)
    merged.append(OpenUSDMeshData(
        name=f"usd_mat_bucket_{b_idx}",
        verts=np.vstack(vs),
        faces=np.vstack(fs),
        vert_normals=np.vstack(ns),
        rgba=avg_rgba,
    ))
  return merged


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
) -> Dict[str, Any]:
  """Compile any OpenUSD stage directly into CompactWideBVHSceneSpec + TPU HBM buffers."""
  import jax
  import jax.numpy as jnp
  from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
  from wave3_500m_10b_8k import CompactWideBVHMeshItem, CompactWideBVHSceneSpec, build_compact_bvh_mesh_item

  st = load_openusd_stage(stage_path, time_code=time_code)
  unique_meshes = _merge_meshes_by_color(st.unique_meshes, max_buckets=max_unique_bvh_meshes)

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
  hbm_bytes = 0
  for mid, md in enumerate(unique_meshes):
    it = build_compact_bvh_mesh_item(mid, _subdivide_if_needed(md), build_method=build_method)
    hbm_bytes += it.l2_roots_aabb.nbytes + it.l2_aabbs.nbytes + it.l0_aabbs.nbytes + it.leaf_bank.nbytes
    unique_dev_arrays.append((_put(it.l2_roots_aabb), _put(it.l2_aabbs), _put(it.l0_aabbs), _put(it.leaf_bank)))
    unique_items_meta.append(_strip(it))

  inst_items_meta = []
  inst_dev_arrays = []
  inst_xpos_list = []
  inst_xmat_list = []
  for gid, grp in enumerate(st.instanced_groups):
    it = build_compact_bvh_mesh_item(gid, _subdivide_if_needed(grp.prototype_mesh), build_method=build_method)
    hbm_bytes += it.l2_roots_aabb.nbytes + it.l2_aabbs.nbytes + it.l0_aabbs.nbytes + it.leaf_bank.nbytes
    inst_dev_arrays.append((_put(it.l2_roots_aabb), _put(it.l2_aabbs), _put(it.l0_aabbs), _put(it.leaf_bank)))
    inst_items_meta.append(_strip(it))
    ixp = grp.inst_xpos[None, :, :].astype(np.float32)
    ixm = grp.inst_xmat[None, :, :, :].astype(np.float32)
    inst_xpos_list.append(jax.device_put(ixp, rep) if rep is not None else jnp.asarray(ixp))
    inst_xmat_list.append(jax.device_put(ixm, rep) if rep is not None else jnp.asarray(ixm))

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
      "obj_xpos": jax.device_put(oxp, rep) if rep is not None else jnp.asarray(oxp),
      "obj_xmat": jax.device_put(oxm, rep) if rep is not None else jnp.asarray(oxm),
      "inst_spec": _mk_spec(inst_items_meta) if inst_items_meta else None,
      "inst_dev_arrays": tuple(inst_dev_arrays),
      "inst_xpos": tuple(inst_xpos_list),
      "inst_xmat": tuple(inst_xmat_list),
      "cam_xpos": jax.device_put(cxp, rep) if rep is not None else jnp.asarray(cxp),
      "cam_xmat": jax.device_put(cxm, rep) if rep is not None else jnp.asarray(cxm),
      "hbm_bytes": hbm_bytes,
      "unique_tris": sum(it.n_tris_orig for it in unique_items_meta),
      "inst_blas_tris": sum(it.n_tris_orig for it in inst_items_meta),
      "inst_effective_tris": sum(it.n_tris_orig * grp.inst_xpos.shape[0] for it, grp in zip(inst_items_meta, st.instanced_groups)),
  }
