"""Pure-Python glTF 2.0 loader & vectorized dense anisotropic 3DGS sampler."""

from __future__ import annotations

import json
import math
import os
import struct
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


_COMP_DTYPE = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}
_TYPE_COUNT = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT4": 16,
}


def _quat_to_mat3(q: List[float]) -> np.ndarray:
  x, y, z, w = [float(v) for v in q]
  return np.array(
      [
          [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
          [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
          [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
      ],
      dtype=np.float64,
  )


def _node_matrix(node: Dict[str, Any]) -> np.ndarray:
  if "matrix" in node:
    return np.array(node["matrix"], dtype=np.float64).reshape(4, 4).T
  M = np.eye(4, dtype=np.float64)
  if "scale" in node:
    s = np.array(node["scale"], dtype=np.float64)
    M[:3, :3] *= s[None, :]
  if "rotation" in node:
    R = _quat_to_mat3(node["rotation"])
    M[:3, :3] = R @ M[:3, :3]
  if "translation" in node:
    M[:3, 3] = np.array(node["translation"], dtype=np.float64)
  return M


def _read_accessor(gltf: Dict[str, Any], buffers: List[bytes], acc_idx: int) -> np.ndarray:
  acc = gltf["accessors"][acc_idx]
  bv = gltf["bufferViews"][acc["bufferView"]]
  buf = buffers[bv["buffer"]]
  dtype = np.dtype(_COMP_DTYPE[acc["componentType"]])
  n_comp = _TYPE_COUNT[acc["type"]]
  count = int(acc["count"])
  offset = int(bv.get("byteOffset", 0)) + int(acc.get("byteOffset", 0))
  stride = int(bv.get("byteStride", 0))
  elem_bytes = dtype.itemsize * n_comp

  if stride == 0 or stride == elem_bytes:
    arr = np.frombuffer(buf, dtype=dtype, count=count * n_comp, offset=offset).reshape(count, n_comp)
  else:
    raw = np.frombuffer(buf, dtype=np.uint8, count=count * stride, offset=offset).reshape(count, stride)
    arr = np.frombuffer(np.ascontiguousarray(raw[:, :elem_bytes]).tobytes(), dtype=dtype).reshape(count, n_comp)

  arr = arr.copy()
  if acc.get("normalized", False):
    if dtype == np.uint8:
      arr = arr.astype(np.float32) / 255.0
    elif dtype == np.uint16:
      arr = arr.astype(np.float32) / 65535.0
    elif dtype == np.int8:
      arr = np.maximum(arr.astype(np.float32) / 127.0, -1.0)
    elif dtype == np.int16:
      arr = np.maximum(arr.astype(np.float32) / 32767.0, -1.0)
  return arr if n_comp > 1 else arr[:, 0]


def load_gltf_triangles(gltf_path: str) -> Dict[str, Any]:
  """Load all triangle geometry + material textures from a .gltf file."""
  base_dir = os.path.dirname(gltf_path)
  with open(gltf_path, "r") as f:
    gltf = json.load(f)

  buffers: List[bytes] = []
  for b in gltf.get("buffers", []):
    uri = b["uri"]
    with open(os.path.join(base_dir, uri), "rb") as bf:
      buffers.append(bf.read())

  # Load textures used as baseColorTexture
  images_cache: Dict[int, np.ndarray] = {}
  def _get_image(img_idx: int) -> np.ndarray:
    if img_idx not in images_cache:
      uri = gltf["images"][img_idx]["uri"]
      im = Image.open(os.path.join(base_dir, uri)).convert("RGBA")
      images_cache[img_idx] = np.asarray(im, dtype=np.float32) / 255.0
    return images_cache[img_idx]

  materials_info: List[Dict[str, Any]] = []
  for mat in gltf.get("materials", []):
    pbr = mat.get("pbrMetallicRoughness", {})
    factor = np.array(pbr.get("baseColorFactor", [1.0, 1.0, 1.0, 1.0]), dtype=np.float32)
    tex_arr = None
    if "baseColorTexture" in pbr:
      tex_idx = pbr["baseColorTexture"]["index"]
      src_idx = gltf["textures"][tex_idx]["source"]
      tex_arr = _get_image(src_idx)
    alpha_mode = mat.get("alphaMode", "OPAQUE")
    alpha_cutoff = float(mat.get("alphaCutoff", 0.5))
    materials_info.append({
        "factor": factor,
        "tex": tex_arr,
        "alpha_mode": alpha_mode,
        "alpha_cutoff": alpha_cutoff,
    })

  if not materials_info:
    materials_info.append({"factor": np.ones(4, np.float32), "tex": None, "alpha_mode": "OPAQUE", "alpha_cutoff": 0.5})

  v0_l, v1_l, v2_l = [], [], []
  n0_l, n1_l, n2_l = [], [], []
  uv0_l, uv1_l, uv2_l = [], [], []
  mat_l = []

  nodes = gltf.get("nodes", [])
  meshes = gltf.get("meshes", [])

  def _visit(node_idx: int, parent_M: np.ndarray) -> None:
    node = nodes[node_idx]
    M = parent_M @ _node_matrix(node)
    if "mesh" in node:
      R3 = M[:3, :3]
      nrm_M = np.linalg.inv(R3).T if abs(np.linalg.det(R3)) > 1e-12 else np.eye(3)
      t3 = M[:3, 3]
      mesh = meshes[node["mesh"]]
      for prim in mesh.get("primitives", []):
        mode = prim.get("mode", 4)
        if mode != 4:
          continue
        attrs = prim["attributes"]
        pos = _read_accessor(gltf, buffers, attrs["POSITION"]).astype(np.float64)
        pos_w = (pos @ R3.T + t3[None, :]).astype(np.float32)

        if "indices" in prim:
          idx = _read_accessor(gltf, buffers, prim["indices"]).astype(np.int32)
        else:
          idx = np.arange(len(pos_w), dtype=np.int32)
        tris = idx.reshape(-1, 3)

        p0 = pos_w[tris[:, 0]]
        p1 = pos_w[tris[:, 1]]
        p2 = pos_w[tris[:, 2]]

        if "NORMAL" in attrs:
          nrm = _read_accessor(gltf, buffers, attrs["NORMAL"]).astype(np.float64)
          nrm_w = nrm @ nrm_M.T
          nrm_w = (nrm_w / np.maximum(np.linalg.norm(nrm_w, axis=-1, keepdims=True), 1e-12)).astype(np.float32)
          na, nb, nc = nrm_w[tris[:, 0]], nrm_w[tris[:, 1]], nrm_w[tris[:, 2]]
        else:
          fn = np.cross(p1 - p0, p2 - p0)
          fn = (fn / np.maximum(np.linalg.norm(fn, axis=-1, keepdims=True), 1e-12)).astype(np.float32)
          na = nb = nc = fn

        if "TEXCOORD_0" in attrs:
          uv = _read_accessor(gltf, buffers, attrs["TEXCOORD_0"]).astype(np.float32)
          uva, uvb, uvc = uv[tris[:, 0]], uv[tris[:, 1]], uv[tris[:, 2]]
        else:
          z_uv = np.zeros((len(tris), 2), dtype=np.float32)
          uva = uvb = uvc = z_uv

        mid = int(prim.get("material", 0))
        v0_l.append(p0); v1_l.append(p1); v2_l.append(p2)
        n0_l.append(na); n1_l.append(nb); n2_l.append(nc)
        uv0_l.append(uva); uv1_l.append(uvb); uv2_l.append(uvc)
        mat_l.append(np.full((len(tris),), mid, dtype=np.int32))

    for ch in node.get("children", []):
      _visit(ch, M)

  scene_idx = int(gltf.get("scene", 0))
  for root_idx in gltf["scenes"][scene_idx]["nodes"]:
    _visit(root_idx, np.eye(4, dtype=np.float64))

  v0 = np.concatenate(v0_l, axis=0)
  v1 = np.concatenate(v1_l, axis=0)
  v2 = np.concatenate(v2_l, axis=0)
  return {
      "v0": v0, "v1": v1, "v2": v2,
      "n0": np.concatenate(n0_l, axis=0),
      "n1": np.concatenate(n1_l, axis=0),
      "n2": np.concatenate(n2_l, axis=0),
      "uv0": np.concatenate(uv0_l, axis=0),
      "uv1": np.concatenate(uv1_l, axis=0),
      "uv2": np.concatenate(uv2_l, axis=0),
      "mat_id": np.concatenate(mat_l, axis=0),
      "materials": materials_info,
      "bbox_min": np.minimum(v0.min(axis=0), np.minimum(v1.min(axis=0), v2.min(axis=0))),
      "bbox_max": np.maximum(v0.max(axis=0), np.maximum(v1.max(axis=0), v2.max(axis=0))),
  }


def sample_anisotropic_3dgs_from_gltf(
    scene_tris: Dict[str, Any],
    n_gaussians: int,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
  """Dense-sample N anisotropic tangent-aligned textured 3D Gaussians from glTF triangles."""
  rng = np.random.default_rng(seed)
  v0, v1, v2 = scene_tris["v0"], scene_tris["v1"], scene_tris["v2"]
  areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=-1).astype(np.float64)
  total_area = float(np.sum(areas))
  probs = areas / max(total_area, 1e-12)
  probs = probs / np.sum(probs)

  # Oversample slightly (1.08x) to allow dropping alpha-masked texels cleanly, then trim/pad to N
  n_draw = int(n_gaussians * 1.08)
  tri_idx = rng.choice(len(v0), size=n_draw, replace=True, p=probs)

  r1 = np.sqrt(rng.uniform(0.0, 1.0, size=(n_draw, 1))).astype(np.float32)
  r2 = rng.uniform(0.0, 1.0, size=(n_draw, 1)).astype(np.float32)
  w0, w1, w2 = 1.0 - r1, r1 * (1.0 - r2), r1 * r2

  mu = (w0 * v0[tri_idx] + w1 * v1[tri_idx] + w2 * v2[tri_idx]).astype(np.float32)
  nrm = (
      w0 * scene_tris["n0"][tri_idx]
      + w1 * scene_tris["n1"][tri_idx]
      + w2 * scene_tris["n2"][tri_idx]
  ).astype(np.float32)
  nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-12)
  uv = (
      w0 * scene_tris["uv0"][tri_idx]
      + w1 * scene_tris["uv1"][tri_idx]
      + w2 * scene_tris["uv2"][tri_idx]
  ).astype(np.float32)
  mid = scene_tris["mat_id"][tri_idx]

  rgb = np.zeros((n_draw, 3), dtype=np.float32)
  keep = np.ones((n_draw,), dtype=bool)
  materials = scene_tris["materials"]
  for m_i, mat in enumerate(materials):
    sel = np.where(mid == m_i)[0]
    if len(sel) == 0:
      continue
    fac = mat["factor"]
    tex = mat["tex"]
    if tex is not None:
      ht, wt, _ = tex.shape
      u_sel = np.mod(uv[sel, 0], 1.0)
      v_sel = np.mod(uv[sel, 1], 1.0)
      ix = np.clip((u_sel * (wt - 1)).astype(np.int32), 0, wt - 1)
      iy = np.clip((v_sel * (ht - 1)).astype(np.int32), 0, ht - 1)
      texels = tex[iy, ix]  # (cnt, 4)
      rgb[sel] = np.clip(texels[:, :3] * fac[:3][None, :], 0.0, 1.0)
      if mat["alpha_mode"] == "MASK":
        keep[sel] = (texels[:, 3] * fac[3]) >= mat["alpha_cutoff"]
    else:
      rgb[sel] = np.clip(fac[:3][None, :], 0.0, 1.0)

  kept_idx = np.where(keep)[0]
  if len(kept_idx) >= n_gaussians:
    sel_idx = kept_idx[:n_gaussians]
  else:
    pad_cnt = n_gaussians - len(kept_idx)
    sel_idx = np.concatenate([kept_idx, rng.choice(kept_idx, size=pad_cnt, replace=True)])

  mu = mu[sel_idx]
  nrm = nrm[sel_idx]
  rgb = rgb[sel_idx]

  # Tangent-aligned anisotropic surface disk covariance packed as 6 floats (xx, xy, xz, yy, yz, zz)
  ref = np.where(
      np.abs(nrm[:, 2:3]) < 0.9,
      np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
      np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
  )
  t1 = np.cross(ref, nrm)
  t1 = t1 / np.maximum(np.linalg.norm(t1, axis=-1, keepdims=True), 1e-12)
  t2 = np.cross(nrm, t1)

  area_per_g = total_area / float(n_gaussians)
  rad = float(np.sqrt(area_per_g / math.pi) * 1.65)
  thick = max(rad * 0.14, 1e-4)
  r2_val = np.float32(rad * rad)
  t2_val = np.float32(thick * thick)

  # cov = r2 * (t1 t1^T + t2 t2^T) + t2 * (nrm nrm^T) -> zero (N, 3, 3) temp!
  cxx = r2_val * (t1[:, 0] * t1[:, 0] + t2[:, 0] * t2[:, 0]) + t2_val * (nrm[:, 0] * nrm[:, 0])
  cxy = r2_val * (t1[:, 0] * t1[:, 1] + t2[:, 0] * t2[:, 1]) + t2_val * (nrm[:, 0] * nrm[:, 1])
  cxz = r2_val * (t1[:, 0] * t1[:, 2] + t2[:, 0] * t2[:, 2]) + t2_val * (nrm[:, 0] * nrm[:, 2])
  cyy = r2_val * (t1[:, 1] * t1[:, 1] + t2[:, 1] * t2[:, 1]) + t2_val * (nrm[:, 1] * nrm[:, 1])
  cyz = r2_val * (t1[:, 1] * t1[:, 2] + t2[:, 1] * t2[:, 2]) + t2_val * (nrm[:, 1] * nrm[:, 2])
  czz = r2_val * (t1[:, 2] * t1[:, 2] + t2[:, 2] * t2[:, 2]) + t2_val * (nrm[:, 2] * nrm[:, 2])
  cov6 = np.stack([cxx, cxy, cxz, cyy, cyz, czz], axis=-1).astype(np.float32)

  return {
      "mu": mu,
      "cov6": cov6,
      "nrm": nrm,
      "rgb": rgb,
      "opacity": np.full((n_gaussians,), 0.92, dtype=np.float32),
      "total_area": np.float32(total_area),
      "splat_radius": np.float32(rad),
  }


def look_at_cam_mat(cam_pos: np.ndarray, target: np.ndarray, world_up: Tuple[float, float, float] = (0.0, 1.0, 0.0)) -> np.ndarray:
  fwd = target - cam_pos
  fwd = fwd / max(float(np.linalg.norm(fwd)), 1e-12)
  up_ref = np.array(world_up, dtype=np.float32)
  right = np.cross(fwd, up_ref)
  right = right / max(float(np.linalg.norm(right)), 1e-12)
  up = np.cross(right, fwd)
  return np.stack([right, up, -fwd], axis=1).astype(np.float32)
