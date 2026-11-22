"""Universal `.blend` Cycles Shader Graph & Depsgraph Auto-Compiler -> TPU ScenePack IR.

Automatically walks ANY `.blend` file without per-scene hand scripts:
  - Evaluates Blender's `depsgraph.object_instances` (modifiers, subdivisions, curves,
    linked collections, instanced data-blocks) into unique BLAS meshes + world poses
    as well as flattened world-space geometry.
  - Recursively walks Cycles material shader graphs (`Principled BSDF`, `Glass BSDF`,
    `Glossy BSDF`, `Diffuse BSDF`, `Refraction BSDF`, `Translucent BSDF`,
    `Transparent BSDF`, `Emission`, `Mix Shader`, `Add Shader`, `ShaderNodeGroup`)
    and value/texture graphs (`Image Texture`, `Mapping`, `TexCoord`, `ColorRamp`,
    `Mix` / `MixRGB`, `Hue/Sat`, `Bright/Contrast`, `Gamma`, `Invert`, `Math`,
    `MapRange`, `Checker`, `Noise`, `Vertex Color` / `Attribute`) into per-material
    PBR properties and packed texture atlases.
  - Extracts world background (`Background`, `Sky Texture`, `Environment Texture`),
    all scene lights (`SUN`, `AREA`, `POINT`, `SPOT`) + emissive mesh emitters,
    and active/override camera intrinsics (`sensor_fit`, `lens`, `shift_x/y`, `clip`).
  - Dual-mode: runs inside Blender (`import bpy`) or from host Python via `compile_blend()`.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
  import bpy  # type: ignore
  HAS_BPY = True
except Exception:
  bpy = None  # type: ignore
  HAS_BPY = False


# ==============================================================================
# ScenePack Intermediate Representation (IR)
# ==============================================================================

@dataclasses.dataclass
class CompiledMaterial:
  mat_id: int
  name: str
  base_color: Tuple[float, float, float]   # Linear RGB [0, 1]
  roughness: float                         # [0.02, 0.99]
  metallic: float                          # [0.0, 1.0]
  transmission: float                      # [0.0, 1.0]
  ior: float                               # e.g. 1.45 / 1.50
  emissive_color: Tuple[float, float, float]
  emissive_strength: float
  alpha: float
  mat_code: float                          # [0..1]=metallic, 2.0=glass, >=3.0=emissive
  uv_scale_offset: Tuple[float, float, float, float] = (1.0, 1.0, 0.0, 0.0)
  tex_id: int = -1                         # index into ScenePack.tex_atlas (-1 if untextured)
  rgh_tex_id: int = -1                     # optional roughness texture index
  vcol_layer: Optional[str] = None


@dataclasses.dataclass
class ScenePack:
  """Portable TPU-ready IR compiled from any `.blend` scene."""

  scene_name: str
  verts: np.ndarray              # (V, 3) float32 world-space
  faces: np.ndarray              # (F, 3) int32
  vnorm: np.ndarray              # (V, 3) float32 smooth normals
  tri_uvs: np.ndarray            # (F, 3, 2) float32 per-triangle corner UVs
  tri_mat_id: np.ndarray         # (F,) int32
  tri_albedo: np.ndarray         # (F, 3) float32 linear RGB
  tri_roughness: np.ndarray      # (F,) float32
  tri_mat_code: np.ndarray       # (F,) float32
  tex_atlas: np.ndarray          # (T, TH, TW, 4) float32 linear RGB + roughness (empty (0,TH,TW,4) if none)
  mat_tex_ids: np.ndarray        # (M,) int32 mapping mat_id -> tex_atlas index (-1 if none)
  mat_base_rgba: np.ndarray      # (M, 4) float32 (base_color RGB, roughness)
  mat_codes: np.ndarray          # (M,) float32
  # Camera & Environment
  cam_pos: np.ndarray            # (3,) float32
  cam_xmat: np.ndarray           # (3, 3) float32 columns=[right, up, -forward]
  cam_fovy: float
  cam_shift: Tuple[float, float]
  znear: float
  zfar: float
  native_res: Tuple[int, int, int]  # (W, H, pct)
  lights: List[Dict[str, Any]]
  sky_zenith: Tuple[float, float, float]
  sky_horizon: Tuple[float, float, float]
  sky_ground: Tuple[float, float, float]
  materials: List[Dict[str, Any]]
  # Optional BLAS instancing summary
  n_blas_meshes: int = 0
  n_instances: int = 0
  compile_meta: Dict[str, Any] = dataclasses.field(default_factory=dict)

  @property
  def n_verts(self) -> int:
    return int(self.verts.shape[0])

  @property
  def n_tris(self) -> int:
    return int(self.faces.shape[0])

  def save(self, npz_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(npz_path)), exist_ok=True)
    np.savez(
        npz_path,
        scene_name=np.bytes_(self.scene_name),
        verts=self.verts.astype(np.float32),
        faces=self.faces.astype(np.int32),
        vnorm=self.vnorm.astype(np.float32),
        tri_uvs=self.tri_uvs.astype(np.float16),
        tri_mat_id=self.tri_mat_id.astype(np.int32),
        tri_albedo=self.tri_albedo.astype(np.float16),
        tri_roughness=self.tri_roughness.astype(np.float16),
        tri_mat_code=self.tri_mat_code.astype(np.float32),
        tex_atlas=(np.clip(self.tex_atlas, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8) if self.tex_atlas.size > 0 else np.zeros((0, 1, 1, 4), np.uint8),
        mat_tex_ids=self.mat_tex_ids.astype(np.int32),
        mat_base_rgba=self.mat_base_rgba.astype(np.float32),
        mat_codes=self.mat_codes.astype(np.float32),
        cam_pos=self.cam_pos.astype(np.float32),
        cam_xmat=self.cam_xmat.astype(np.float32),
        cam_fovy=np.float32(self.cam_fovy),
        cam_shift=np.asarray(self.cam_shift, dtype=np.float32),
        znear=np.float32(self.znear),
        zfar=np.float32(self.zfar),
        native_res=np.asarray(self.native_res, dtype=np.int32),
        lights_json=np.bytes_(json.dumps(self.lights)),
        sky_zenith=np.asarray(self.sky_zenith, dtype=np.float32),
        sky_horizon=np.asarray(self.sky_horizon, dtype=np.float32),
        sky_ground=np.asarray(self.sky_ground, dtype=np.float32),
        materials_json=np.bytes_(json.dumps(self.materials)),
        inst_counts=np.asarray([self.n_blas_meshes, self.n_instances], dtype=np.int32),
        compile_meta_json=np.bytes_(json.dumps(self.compile_meta)),
    )

  @classmethod
  def load(cls, npz_path: str) -> "ScenePack":
    ds = np.load(npz_path, allow_pickle=False)
    ta_u8 = ds["tex_atlas"]
    tex_atlas = (ta_u8.astype(np.float32) / 255.0) if ta_u8.shape[0] > 0 else np.zeros((0, 1, 1, 4), np.float32)
    ic = ds["inst_counts"] if "inst_counts" in ds else np.array([0, 0], dtype=np.int32)
    meta = json.loads(bytes(ds["compile_meta_json"]).decode("utf-8")) if "compile_meta_json" in ds else {}
    return cls(
        scene_name=bytes(ds["scene_name"]).decode("utf-8"),
        verts=ds["verts"].astype(np.float32),
        faces=ds["faces"].astype(np.int32),
        vnorm=ds["vnorm"].astype(np.float32),
        tri_uvs=ds["tri_uvs"].astype(np.float32),
        tri_mat_id=ds["tri_mat_id"].astype(np.int32),
        tri_albedo=ds["tri_albedo"].astype(np.float32),
        tri_roughness=ds["tri_roughness"].astype(np.float32),
        tri_mat_code=ds["tri_mat_code"].astype(np.float32),
        tex_atlas=tex_atlas,
        mat_tex_ids=ds["mat_tex_ids"].astype(np.int32),
        mat_base_rgba=ds["mat_base_rgba"].astype(np.float32),
        mat_codes=ds["mat_codes"].astype(np.float32),
        cam_pos=ds["cam_pos"].astype(np.float32),
        cam_xmat=ds["cam_xmat"].astype(np.float32),
        cam_fovy=float(ds["cam_fovy"]),
        cam_shift=(float(ds["cam_shift"][0]), float(ds["cam_shift"][1])),
        znear=float(ds["znear"]),
        zfar=float(ds["zfar"]),
        native_res=tuple(int(x) for x in ds["native_res"]),
        lights=json.loads(bytes(ds["lights_json"]).decode("utf-8")),
        sky_zenith=tuple(float(x) for x in ds["sky_zenith"]),
        sky_horizon=tuple(float(x) for x in ds["sky_horizon"]),
        sky_ground=tuple(float(x) for x in ds["sky_ground"]),
        materials=json.loads(bytes(ds["materials_json"]).decode("utf-8")),
        n_blas_meshes=int(ic[0]),
        n_instances=int(ic[1]),
        compile_meta=meta,
    )

  def summary(self) -> Dict[str, Any]:
    return {
        "scene_name": self.scene_name,
        "verts": self.n_verts,
        "tris": self.n_tris,
        "n_materials": len(self.materials),
        "n_textures": int(self.tex_atlas.shape[0]),
        "tex_res": f"{self.tex_atlas.shape[2]}x{self.tex_atlas.shape[1]}" if self.tex_atlas.shape[0] > 0 else "0x0",
        "n_lights": len(self.lights),
        "n_blas_meshes": self.n_blas_meshes,
        "n_instances": self.n_instances,
        "cam_fovy": round(self.cam_fovy, 3),
        "native_res": f"{self.native_res[0]}x{self.native_res[1]}@{self.native_res[2]}%",
    }


# ==============================================================================
# Blender Cycles Shader Graph Auto-Walker
# ==============================================================================

def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
  c = np.clip(c, 0.0, 1.0).astype(np.float32)
  return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
  c = np.maximum(c, 0.0).astype(np.float32)
  return np.where(c <= 0.0031308, 12.92 * c, 1.055 * (np.maximum(c, 1e-8) ** (1.0 / 2.4)) - 0.055).astype(np.float32)


def _resize_bilinear(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
  h, w = img.shape[0], img.shape[1]
  if h == out_h and w == out_w:
    return img
  ys = (np.arange(out_h, dtype=np.float32) + 0.5) * (h / float(out_h)) - 0.5
  xs = (np.arange(out_w, dtype=np.float32) + 0.5) * (w / float(out_w)) - 0.5
  ys = np.clip(ys, 0.0, max(h - 1, 0))
  xs = np.clip(xs, 0.0, max(w - 1, 0))
  y0 = np.floor(ys).astype(np.int32)
  x0 = np.floor(xs).astype(np.int32)
  y1 = np.minimum(y0 + 1, h - 1)
  x1 = np.minimum(x0 + 1, w - 1)
  wy = (ys - y0)[:, None, None]
  wx = (xs - x0)[None, :, None]
  if img.ndim == 2:
    img = img[..., None]
    squeezed = True
  else:
    squeezed = False
  top = (1.0 - wx) * img[y0[:, None], x0[None, :]] + wx * img[y0[:, None], x1[None, :]]
  bot = (1.0 - wx) * img[y1[:, None], x0[None, :]] + wx * img[y1[:, None], x1[None, :]]
  out = ((1.0 - wy) * top + wy * bot).astype(np.float32)
  return out[..., 0] if squeezed else out


def _eval_color_ramp(node, t: np.ndarray) -> np.ndarray:
  elems = list(node.color_ramp.elements)
  if not elems:
    return np.broadcast_to(np.array([0.5, 0.5, 0.5], np.float32), (*t.shape, 3)).copy()
  elems.sort(key=lambda e: float(e.position))
  pos = np.array([float(e.position) for e in elems], dtype=np.float32)
  cols = np.array([list(e.color[:3]) for e in elems], dtype=np.float32)
  out = np.empty((*t.shape, 3), dtype=np.float32)
  t_flat = np.clip(t.ravel(), 0.0, 1.0)
  for c in range(3):
    out[..., c] = np.interp(t_flat, pos, cols[:, c]).reshape(t.shape)
  return out


def _synthesize_checker(scale: float, c1: np.ndarray, c2: np.ndarray, size: int = 128) -> np.ndarray:
  u = np.linspace(0.0, max(scale, 1.0), size, endpoint=False, dtype=np.float32)[None, :]
  v = np.linspace(0.0, max(scale, 1.0), size, endpoint=False, dtype=np.float32)[:, None]
  chk = ((np.floor(u).astype(np.int32) + np.floor(v).astype(np.int32)) & 1) == 0
  return np.where(chk[..., None], c1[None, None, :], c2[None, None, :]).astype(np.float32)


def _synthesize_procedural_scalar(seed: int, scale: float, size: int = 128) -> np.ndarray:
  rng = np.random.default_rng(abs(seed) % (2**31 - 1))
  y = np.linspace(0.0, 2.0 * math.pi, size, endpoint=False, dtype=np.float32)[:, None]
  x = np.linspace(0.0, 2.0 * math.pi, size, endpoint=False, dtype=np.float32)[None, :]
  freq = max(1, min(int(round(abs(scale))), 12))
  acc = np.zeros((size, size), dtype=np.float32)
  amp, norm = 0.5, 0.0
  for o in range(4):
    f = freq * (2 ** o)
    kx = int(rng.integers(1, max(2, f)))
    ky = int(rng.integers(1, max(2, f)))
    ph = float(rng.uniform(0.0, 2.0 * math.pi))
    acc += amp * np.sin(kx * x + ky * y + ph)
    norm += amp
    amp *= 0.5
  return np.clip(0.5 + 0.5 * acc / max(norm, 1e-6), 0.0, 1.0).astype(np.float32)


class _ShaderGraphWalker:
  """Recursively walks Blender Cycles node trees (including ShaderNodeGroup & Mix)."""

  def __init__(self, tex_res: int = 256):
    self.tex_res = int(tex_res)
    self.img_cache: Dict[str, Optional[np.ndarray]] = {}

  def _load_image(self, im, as_linear: bool = True) -> Optional[np.ndarray]:
    if im is None:
      return None
    key = f"{im.name}_{int(as_linear)}"
    if key in self.img_cache:
      return self.img_cache[key]
    arr_out = None
    try:
      w, h = int(im.size[0]), int(im.size[1])
      if w <= 0 or h <= 0:
        try:
          im.reload()
          w, h = int(im.size[0]), int(im.size[1])
        except Exception:
          pass
      if w > 0 and h > 0 and len(im.pixels) == w * h * 4:
        raw_flat = np.empty(w * h * 4, dtype=np.float32)
        im.pixels.foreach_get(raw_flat)
        raw = raw_flat.reshape(h, w, 4)[..., :3]
        # Flip bottom-up Blender buffer -> top-down image
        raw = raw[::-1].copy()
        # Resize large textures to bounded atlas resolution
        if max(h, w) > self.tex_res:
          raw = _resize_bilinear(raw, self.tex_res, self.tex_res)
        elif h != self.tex_res or w != self.tex_res:
          raw = _resize_bilinear(raw, self.tex_res, self.tex_res)
        cs_name = getattr(getattr(im, "colorspace_settings", None), "name", "sRGB").lower()
        if as_linear and ("srgb" in cs_name or cs_name == ""):
          raw = _srgb_to_linear(raw)
        arr_out = np.clip(raw, 0.0, 1.0).astype(np.float32)
    except Exception:
      arr_out = None
    self.img_cache[key] = arr_out
    return arr_out

  def _trace_uv_transform(self, vec_sock, depth: int = 0) -> Tuple[Tuple[float, float, float, float], Optional[str]]:
    scale_off = (1.0, 1.0, 0.0, 0.0)
    uv_name = None
    if vec_sock is None or not vec_sock.is_linked or depth > 8:
      return scale_off, uv_name
    fn = vec_sock.links[0].from_node
    if fn.type == "MAPPING":
      try:
        loc = fn.inputs["Location"].default_value
        scl = fn.inputs["Scale"].default_value
        scale_off = (float(scl[0]), float(scl[1]), float(loc[0]), float(loc[1]))
      except Exception:
        pass
      sub_so, sub_uv = self._trace_uv_transform(fn.inputs.get("Vector", None), depth + 1)
      if sub_uv is not None:
        uv_name = sub_uv
    elif fn.type == "UVMAP":
      uv_name = getattr(fn, "uv_map", None) or None
    elif fn.type == "ATTRIBUTE":
      uv_name = getattr(fn, "attribute_name", None) or None
    return scale_off, uv_name

  def _resolve_socket_link(self, sock, group_stack: List[Any]):
    """Follow link, unwinding GroupInput back into outer node tree if inside a ShaderNodeGroup."""
    if sock is None or not sock.is_linked:
      return None, group_stack
    link = sock.links[0]
    fn = link.from_node
    fs = link.from_socket
    if fn.type == "GROUP_INPUT" and group_stack:
      caller_node, outer_stack = group_stack[-1], group_stack[:-1]
      outer_sock = caller_node.inputs.get(fs.name, None)
      if outer_sock is None:
        # Match by index
        idx = list(fn.outputs).index(fs)
        if idx < len(caller_node.inputs):
          outer_sock = caller_node.inputs[idx]
      if outer_sock is not None and outer_sock.is_linked:
        return self._resolve_socket_link(outer_sock, outer_stack)
      return outer_sock, outer_stack
    return (fn, fs), group_stack

  def _eval_scalar(self, sock, default: float, group_stack: List[Any], depth: int = 0) -> Tuple[float, Optional[np.ndarray]]:
    if sock is None or depth > 14:
      return float(default), None
    if not sock.is_linked:
      try:
        dv = sock.default_value
        if hasattr(dv, "__len__"):
          return float(np.mean(dv[:3])), None
        return float(dv), None
      except Exception:
        return float(default), None

    resolved, new_stack = self._resolve_socket_link(sock, group_stack)
    if resolved is None:
      return float(default), None
    if not isinstance(resolved, tuple):
      try:
        return float(resolved.default_value), None
      except Exception:
        return float(default), None
    fn, fs = resolved

    if fn.type == "VALUE":
      return float(fn.outputs[0].default_value), None
    if fn.type == "TEX_IMAGE":
      img = self._load_image(fn.image, as_linear=False)
      if img is not None:
        g = np.mean(img, axis=-1)
        return float(np.mean(g)), g
      return float(default), None
    if fn.type == "RGBTOBW":
      c, ctex = self._eval_color(fn.inputs[0], (default, default, default), new_stack, depth + 1)
      g = (0.2126 * ctex[..., 0] + 0.7152 * ctex[..., 1] + 0.0722 * ctex[..., 2]) if ctex is not None else None
      return float(0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]), g
    if fn.type == "VALTORGB":
      fac_v, fac_t = self._eval_scalar(fn.inputs.get("Fac"), 0.5, new_stack, depth + 1)
      if fac_t is not None:
        cols = _eval_color_ramp(fn, fac_t)
        g = np.mean(cols, axis=-1)
        return float(np.mean(g)), g
      c = _eval_color_ramp(fn, np.array([fac_v], np.float32))[0]
      return float(np.mean(c)), None
    if fn.type == "MATH":
      v0, t0 = self._eval_scalar(fn.inputs[0], default, new_stack, depth + 1)
      v1, t1 = self._eval_scalar(fn.inputs[1] if len(fn.inputs) > 1 else None, 0.5, new_stack, depth + 1)
      op = fn.operation
      def _apply_op(a, b):
        if op == "ADD": return a + b
        if op == "SUBTRACT": return a - b
        if op == "MULTIPLY": return a * b
        if op == "DIVIDE": return a / np.where(np.abs(b) > 1e-6, b, 1.0)
        if op == "POWER": return np.power(np.maximum(a, 0.0), np.clip(b, -4.0, 4.0))
        if op == "MINIMUM": return np.minimum(a, b)
        if op == "MAXIMUM": return np.maximum(a, b)
        return a
      val = float(_apply_op(v0, v1))
      tex = _apply_op(t0 if t0 is not None else v0, t1 if t1 is not None else v1) if (t0 is not None or t1 is not None) else None
      if getattr(fn, "use_clamp", False):
        val = float(np.clip(val, 0.0, 1.0))
        if tex is not None:
          tex = np.clip(tex, 0.0, 1.0)
      return val, tex
    if fn.type == "MAP_RANGE":
      v, t = self._eval_scalar(fn.inputs.get("Value"), default, new_stack, depth + 1)
      fmin, _ = self._eval_scalar(fn.inputs.get("From Min"), 0.0, new_stack, depth + 1)
      fmax, _ = self._eval_scalar(fn.inputs.get("From Max"), 1.0, new_stack, depth + 1)
      tmin, _ = self._eval_scalar(fn.inputs.get("To Min"), 0.0, new_stack, depth + 1)
      tmax, _ = self._eval_scalar(fn.inputs.get("To Max"), 1.0, new_stack, depth + 1)
      den = fmax - fmin if abs(fmax - fmin) > 1e-6 else 1.0
      def _mr(x):
        r = tmin + ((x - fmin) / den) * (tmax - tmin)
        return np.clip(r, min(tmin, tmax), max(tmin, tmax)) if getattr(fn, "clamp", True) else r
      return float(_mr(v)), (_mr(t) if t is not None else None)
    if fn.type == "INVERT":
      fac, _ = self._eval_scalar(fn.inputs.get("Fac"), 1.0, new_stack, depth + 1)
      v, t = self._eval_scalar(fn.inputs.get("Color"), default, new_stack, depth + 1)
      inv_v = (1.0 - fac) * v + fac * (1.0 - v)
      inv_t = ((1.0 - fac) * t + fac * (1.0 - t)) if t is not None else None
      return float(inv_v), inv_t
    if fn.type in ("TEX_NOISE", "TEX_VORONOI", "TEX_WAVE", "TEX_MUSGRAVE"):
      scl, _ = self._eval_scalar(fn.inputs.get("Scale"), 4.0, new_stack, depth + 1)
      pat = _synthesize_procedural_scalar(hash(fn.name), scl, size=min(128, self.tex_res))
      return float(np.mean(pat)), pat
    if fn.type in ("FRESNEL", "LAYER_WEIGHT"):
      return 0.25, None
    if fn.type == "MIX" and getattr(fn, "data_type", "FLOAT") == "FLOAT":
      fac, _ = self._eval_scalar(fn.inputs[0], 0.5, new_stack, depth + 1)
      a, ta = self._eval_scalar(fn.inputs[2], default, new_stack, depth + 1)
      b, tb = self._eval_scalar(fn.inputs[3], default, new_stack, depth + 1)
      val = (1.0 - fac) * a + fac * b
      tex = ((1.0 - fac) * (ta if ta is not None else a) + fac * (tb if tb is not None else b)) if (ta is not None or tb is not None) else None
      return float(val), tex
    if fn.type == "GROUP" and fn.node_tree is not None:
      g_out = next((n for n in fn.node_tree.nodes if n.type == "GROUP_OUTPUT" and n.is_active_output), None)
      if g_out is None:
        g_out = next((n for n in fn.node_tree.nodes if n.type == "GROUP_OUTPUT"), None)
      if g_out is not None:
        in_s = g_out.inputs.get(fs.name, g_out.inputs[0] if len(g_out.inputs) > 0 else None)
        return self._eval_scalar(in_s, default, new_stack + [fn], depth + 1)
    if fn.type in ("SEPARATE_COLOR", "SEPXYZ"):
      c, ct = self._eval_color(fn.inputs[0], (default, default, default), new_stack, depth + 1)
      ch = {"Red": 0, "X": 0, "Green": 1, "Y": 1, "Blue": 2, "Z": 2}.get(fs.name, 0)
      return float(c[ch]), (ct[..., ch] if ct is not None else None)
    return float(default), None

  def _eval_color(
      self,
      sock,
      default: Tuple[float, float, float],
      group_stack: List[Any],
      depth: int = 0,
  ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    d_arr = np.asarray(default, dtype=np.float32)
    if sock is None or depth > 14:
      return d_arr, None
    if not sock.is_linked:
      try:
        dv = sock.default_value
        if hasattr(dv, "__len__") and len(dv) >= 3:
          return np.asarray(dv[:3], dtype=np.float32), None
        v = float(dv)
        return np.array([v, v, v], dtype=np.float32), None
      except Exception:
        return d_arr, None

    resolved, new_stack = self._resolve_socket_link(sock, group_stack)
    if resolved is None:
      return d_arr, None
    if not isinstance(resolved, tuple):
      try:
        return np.asarray(resolved.default_value[:3], dtype=np.float32), None
      except Exception:
        return d_arr, None
    fn, fs = resolved

    if fn.type == "RGB":
      return np.asarray(fn.outputs[0].default_value[:3], dtype=np.float32), None
    if fn.type == "TEX_IMAGE":
      img = self._load_image(fn.image, as_linear=True)
      if img is not None:
        return np.mean(img, axis=(0, 1)).astype(np.float32), img
      return d_arr, None
    if fn.type == "TEX_CHECKER":
      c1, _ = self._eval_color(fn.inputs.get("Color1"), (0.8, 0.8, 0.8), new_stack, depth + 1)
      c2, _ = self._eval_color(fn.inputs.get("Color2"), (0.2, 0.2, 0.2), new_stack, depth + 1)
      scl, _ = self._eval_scalar(fn.inputs.get("Scale"), 5.0, new_stack, depth + 1)
      tex = _synthesize_checker(scl, c1, c2, size=min(128, self.tex_res))
      return (0.5 * (c1 + c2)).astype(np.float32), tex
    if fn.type in ("TEX_NOISE", "TEX_VORONOI", "TEX_WAVE", "TEX_MUSGRAVE"):
      scl, _ = self._eval_scalar(fn.inputs.get("Scale"), 4.0, new_stack, depth + 1)
      pat = _synthesize_procedural_scalar(hash(fn.name), scl, size=min(128, self.tex_res))
      tex = np.repeat(pat[..., None], 3, axis=-1)
      return np.mean(tex, axis=(0, 1)).astype(np.float32), tex
    if fn.type == "VALTORGB":
      fv, ft = self._eval_scalar(fn.inputs.get("Fac"), 0.5, new_stack, depth + 1)
      if ft is not None:
        tex = _eval_color_ramp(fn, ft)
        return np.mean(tex, axis=(0, 1)).astype(np.float32), tex
      c = _eval_color_ramp(fn, np.array([fv], np.float32))[0]
      return c, None
    if fn.type in ("MIX", "MIX_RGB"):
      if fn.type == "MIX":
        dt = getattr(fn, "data_type", "RGBA")
        if dt == "FLOAT":
          v, t = self._eval_scalar(sock, float(np.mean(d_arr)), group_stack, depth + 1)
          return np.array([v, v, v], np.float32), (np.repeat(t[..., None], 3, axis=-1) if t is not None else None)
        fac_sock = fn.inputs[0]
        # Blender 4+/5 RGBA inputs are typically named A / B at indices 6 / 7
        rgba_ins = [inp for inp in fn.inputs if inp.type == "RGBA"]
        a_sock = rgba_ins[0] if len(rgba_ins) >= 1 else fn.inputs[6]
        b_sock = rgba_ins[1] if len(rgba_ins) >= 2 else fn.inputs[7]
        btype = getattr(fn, "blend_type", "MIX")
      else:
        fac_sock = fn.inputs.get("Fac", fn.inputs[0])
        a_sock = fn.inputs.get("Color1", fn.inputs[1])
        b_sock = fn.inputs.get("Color2", fn.inputs[2])
        btype = getattr(fn, "blend_type", "MIX")
      fac, fac_t = self._eval_scalar(fac_sock, 0.5, new_stack, depth + 1)
      ca, ta = self._eval_color(a_sock, default, new_stack, depth + 1)
      cb, tb = self._eval_color(b_sock, default, new_stack, depth + 1)

      def _blend_rgb(a, b, f):
        if btype == "MULTIPLY":
          return (1.0 - f) * a + f * (a * b)
        if btype == "ADD":
          return np.clip(a + f * b, 0.0, 1.0)
        if btype == "SUBTRACT":
          return np.clip(a - f * b, 0.0, 1.0)
        if btype == "SCREEN":
          return (1.0 - f) * a + f * (1.0 - (1.0 - a) * (1.0 - b))
        return (1.0 - f) * a + f * b

      col = _blend_rgb(ca, cb, fac).astype(np.float32)
      tex = None
      if ta is not None or tb is not None or fac_t is not None:
        def _to_grid(t_in, c_in):
          if t_in is None:
            return np.broadcast_to(c_in[None, None, :], (self.tex_res, self.tex_res, 3))
          if t_in.shape[:2] != (self.tex_res, self.tex_res):
            return _resize_bilinear(t_in, self.tex_res, self.tex_res)
          return t_in
        ga = _to_grid(ta, ca)
        gb = _to_grid(tb, cb)
        gf = fac_t[..., None] if fac_t is not None else fac
        if isinstance(gf, np.ndarray) and gf.shape[:2] != (self.tex_res, self.tex_res):
          gf = _resize_bilinear(gf[..., 0], self.tex_res, self.tex_res)[..., None]
        tex = np.clip(_blend_rgb(ga, gb, gf), 0.0, 1.0).astype(np.float32)
      return col, tex
    if fn.type == "HUE_SAT":
      sat, _ = self._eval_scalar(fn.inputs.get("Saturation"), 1.0, new_stack, depth + 1)
      val, _ = self._eval_scalar(fn.inputs.get("Value"), 1.0, new_stack, depth + 1)
      fac, _ = self._eval_scalar(fn.inputs.get("Fac"), 1.0, new_stack, depth + 1)
      c, ct = self._eval_color(fn.inputs.get("Color"), default, new_stack, depth + 1)
      def _hsv_mod(arr):
        lum = np.sum(arr * np.array([0.2126, 0.7152, 0.0722], np.float32), axis=-1, keepdims=True)
        mod = np.clip((lum + sat * (arr - lum)) * val, 0.0, 1.0)
        return ((1.0 - fac) * arr + fac * mod).astype(np.float32)
      return _hsv_mod(c), (_hsv_mod(ct) if ct is not None else None)
    if fn.type == "BRIGHTCONTRAST":
      br, _ = self._eval_scalar(fn.inputs.get("Bright"), 0.0, new_stack, depth + 1)
      ct_v, _ = self._eval_scalar(fn.inputs.get("Contrast"), 0.0, new_stack, depth + 1)
      c, ct = self._eval_color(fn.inputs.get("Color"), default, new_stack, depth + 1)
      def _bc(arr):
        return np.clip((arr - 0.5) * (1.0 + ct_v) + 0.5 + br, 0.0, 1.0).astype(np.float32)
      return _bc(c), (_bc(ct) if ct is not None else None)
    if fn.type == "GAMMA":
      gm, _ = self._eval_scalar(fn.inputs.get("Gamma"), 1.0, new_stack, depth + 1)
      c, ct = self._eval_color(fn.inputs.get("Color"), default, new_stack, depth + 1)
      g_fn = lambda a: np.power(np.maximum(a, 1e-6), max(gm, 0.05)).astype(np.float32)
      return g_fn(c), (g_fn(ct) if ct is not None else None)
    if fn.type == "INVERT":
      fac, _ = self._eval_scalar(fn.inputs.get("Fac"), 1.0, new_stack, depth + 1)
      c, ct = self._eval_color(fn.inputs.get("Color"), default, new_stack, depth + 1)
      i_fn = lambda a: ((1.0 - fac) * a + fac * (1.0 - a)).astype(np.float32)
      return i_fn(c), (i_fn(ct) if ct is not None else None)
    if fn.type == "CURVE_RGB":
      fac, _ = self._eval_scalar(fn.inputs.get("Fac"), 1.0, new_stack, depth + 1)
      c, ct = self._eval_color(fn.inputs.get("Color"), default, new_stack, depth + 1)
      try:
        mapping = fn.mapping
        mapping.initialize()
        curves = mapping.curves
        # Curves: 0=R, 1=G, 2=B, 3=Combined(C)
        def _eval_lut(arr_in):
          lut_x = np.linspace(0.0, 1.0, 65, dtype=np.float32)
          out = np.empty_like(arr_in, dtype=np.float32)
          c_lut = np.array([mapping.evaluate(curves[3], float(x)) for x in lut_x], dtype=np.float32) if len(curves) > 3 else lut_x
          for ch in range(3):
            v_ch = np.interp(np.clip(arr_in[..., ch], 0.0, 1.0), lut_x, c_lut)
            if ch < len(curves):
              ch_lut = np.array([mapping.evaluate(curves[ch], float(x)) for x in lut_x], dtype=np.float32)
              v_ch = np.interp(np.clip(v_ch, 0.0, 1.0), lut_x, ch_lut)
            out[..., ch] = v_ch
          return np.clip((1.0 - fac) * arr_in + fac * out, 0.0, 1.0).astype(np.float32)
        return _eval_lut(c), (_eval_lut(ct) if ct is not None else None)
      except Exception:
        return c, ct
    if fn.type == "BLACKBODY":
      temp, _ = self._eval_scalar(fn.inputs.get("Temperature"), 6500.0, new_stack, depth + 1)
      t = float(np.clip(temp, 1000.0, 12000.0)) / 100.0
      r = 1.0 if t <= 66.0 else np.clip(1.292936 * ((t - 60.0) ** -0.1332047592), 0.0, 1.0)
      g = np.clip(0.39008157 * math.log(max(t, 1.0)) - 0.63184144, 0.0, 1.0) if t <= 66.0 else np.clip(1.129891 * ((t - 60.0) ** -0.0755148492), 0.0, 1.0)
      b = 1.0 if t >= 66.0 else (0.0 if t <= 19.0 else np.clip(0.54320678 * math.log(max(t - 10.0, 1.0)) - 1.19625408, 0.0, 1.0))
      return _srgb_to_linear(np.array([r, g, b], dtype=np.float32)), None
    if fn.type == "AMBIENT_OCCLUSION":
      return self._eval_color(fn.inputs.get("Color"), default, new_stack, depth + 1)
    if fn.type == "GROUP" and fn.node_tree is not None:
      g_out = next((n for n in fn.node_tree.nodes if n.type == "GROUP_OUTPUT" and n.is_active_output), None)
      if g_out is None:
        g_out = next((n for n in fn.node_tree.nodes if n.type == "GROUP_OUTPUT"), None)
      if g_out is not None:
        in_s = g_out.inputs.get(fs.name, g_out.inputs[0] if len(g_out.inputs) > 0 else None)
        return self._eval_color(in_s, default, new_stack + [fn], depth + 1)
    if fn.type == "VALUE":
      v = float(fn.outputs[0].default_value)
      return np.array([v, v, v], dtype=np.float32), None
    return d_arr, None

  def _find_first_tex_uv_meta(self, sock, depth: int = 0) -> Tuple[Tuple[float, float, float, float], Optional[str]]:
    if sock is None or not sock.is_linked or depth > 10:
      return (1.0, 1.0, 0.0, 0.0), None
    fn = sock.links[0].from_node
    if fn.type == "TEX_IMAGE":
      return self._trace_uv_transform(fn.inputs.get("Vector", None))
    for inp in fn.inputs:
      if inp.is_linked and inp.type in ("RGBA", "VALUE", "VECTOR"):
        so, uvn = self._find_first_tex_uv_meta(inp, depth + 1)
        if so != (1.0, 1.0, 0.0, 0.0) or uvn is not None:
          return so, uvn
    return (1.0, 1.0, 0.0, 0.0), None

  def _eval_shader_socket(
      self,
      sock,
      group_stack: List[Any],
      depth: int = 0,
  ) -> Dict[str, Any]:
    res = {
        "base_color": np.array([0.7, 0.7, 0.7], dtype=np.float32),
        "roughness": 0.5,
        "metallic": 0.0,
        "transmission": 0.0,
        "ior": 1.45,
        "emissive_color": np.zeros(3, dtype=np.float32),
        "emissive_strength": 0.0,
        "alpha": 1.0,
        "tex_albedo": None,
        "tex_roughness": None,
        "uv_scale_offset": (1.0, 1.0, 0.0, 0.0),
        "uv_name": None,
    }
    if sock is None or not sock.is_linked or depth > 12:
      return res

    resolved, new_stack = self._resolve_socket_link(sock, group_stack)
    if resolved is None or not isinstance(resolved, tuple):
      return res
    fn, fs = resolved

    if fn.type == "MIX_SHADER":
      fac, _ = self._eval_scalar(fn.inputs[0], 0.5, new_stack, depth + 1)
      fac = float(np.clip(fac, 0.0, 1.0))
      s1 = self._eval_shader_socket(fn.inputs[1], new_stack, depth + 1)
      s2 = self._eval_shader_socket(fn.inputs[2], new_stack, depth + 1)
      # If one branch is transparent, treat opacity carefully
      if s2["alpha"] < 0.05 and s1["alpha"] > 0.5:
        s1["alpha"] *= (1.0 - fac)
        return s1
      if s1["alpha"] < 0.05 and s2["alpha"] > 0.5:
        s2["alpha"] *= fac
        return s2
      out = dict(s1)
      out["base_color"] = (1.0 - fac) * s1["base_color"] + fac * s2["base_color"]
      out["roughness"] = (1.0 - fac) * s1["roughness"] + fac * s2["roughness"]
      out["metallic"] = (1.0 - fac) * s1["metallic"] + fac * s2["metallic"]
      out["transmission"] = (1.0 - fac) * s1["transmission"] + fac * s2["transmission"]
      out["ior"] = (1.0 - fac) * s1["ior"] + fac * s2["ior"]
      out["emissive_color"] = (1.0 - fac) * s1["emissive_color"] + fac * s2["emissive_color"]
      out["emissive_strength"] = (1.0 - fac) * s1["emissive_strength"] + fac * s2["emissive_strength"]
      out["alpha"] = (1.0 - fac) * s1["alpha"] + fac * s2["alpha"]
      if s1["tex_albedo"] is not None and s2["tex_albedo"] is not None:
        out["tex_albedo"] = (1.0 - fac) * s1["tex_albedo"] + fac * s2["tex_albedo"]
      elif s2["tex_albedo"] is not None and fac > 0.35:
        out["tex_albedo"] = (1.0 - fac) * s1["base_color"][None, None, :] + fac * s2["tex_albedo"]
        out["uv_scale_offset"] = s2["uv_scale_offset"]
        out["uv_name"] = s2["uv_name"]
      elif s1["tex_albedo"] is not None:
        out["tex_albedo"] = (1.0 - fac) * s1["tex_albedo"] + fac * s2["base_color"][None, None, :]
      return out

    if fn.type == "ADD_SHADER":
      s1 = self._eval_shader_socket(fn.inputs[0], new_stack, depth + 1)
      s2 = self._eval_shader_socket(fn.inputs[1], new_stack, depth + 1)
      out = dict(s1 if s1["emissive_strength"] <= s2["emissive_strength"] else s2)
      out["base_color"] = np.clip(s1["base_color"] + s2["base_color"], 0.0, 1.0)
      out["emissive_strength"] = s1["emissive_strength"] + s2["emissive_strength"]
      out["emissive_color"] = np.clip(s1["emissive_color"] + s2["emissive_color"], 0.0, 1.0)
      out["tex_albedo"] = s1["tex_albedo"] if s1["tex_albedo"] is not None else s2["tex_albedo"]
      return out

    if fn.type == "GROUP" and fn.node_tree is not None:
      g_out = next((n for n in fn.node_tree.nodes if n.type == "GROUP_OUTPUT" and n.is_active_output), None)
      if g_out is None:
        g_out = next((n for n in fn.node_tree.nodes if n.type == "GROUP_OUTPUT"), None)
      if g_out is not None:
        in_s = g_out.inputs.get(fs.name, g_out.inputs[0] if len(g_out.inputs) > 0 else None)
        return self._eval_shader_socket(in_s, new_stack + [fn], depth + 1)

    if fn.type == "BSDF_PRINCIPLED":
      bc_s = fn.inputs.get("Base Color")
      col, tex = self._eval_color(bc_s, (0.7, 0.7, 0.7), new_stack, depth + 1)
      so, uvn = self._find_first_tex_uv_meta(bc_s)
      rgh, rgh_t = self._eval_scalar(fn.inputs.get("Roughness"), 0.5, new_stack, depth + 1)
      met, _ = self._eval_scalar(fn.inputs.get("Metallic"), 0.0, new_stack, depth + 1)
      ior, _ = self._eval_scalar(fn.inputs.get("IOR"), 1.45, new_stack, depth + 1)
      tr_s = fn.inputs.get("Transmission Weight", fn.inputs.get("Transmission"))
      tr, _ = self._eval_scalar(tr_s, 0.0, new_stack, depth + 1)
      em_str, _ = self._eval_scalar(fn.inputs.get("Emission Strength"), 0.0, new_stack, depth + 1)
      em_col, _ = self._eval_color(fn.inputs.get("Emission Color", fn.inputs.get("Emission")), (0.0, 0.0, 0.0), new_stack, depth + 1)
      alp, _ = self._eval_scalar(fn.inputs.get("Alpha"), 1.0, new_stack, depth + 1)
      res.update({
          "base_color": np.clip(col, 0.01, 0.99),
          "roughness": float(np.clip(rgh, 0.02, 0.98)),
          "metallic": float(np.clip(met, 0.0, 1.0)),
          "transmission": float(np.clip(tr, 0.0, 1.0)),
          "ior": float(np.clip(ior, 1.0, 2.5)),
          "emissive_color": np.clip(em_col, 0.0, 1.0),
          "emissive_strength": float(max(0.0, em_str)) if float(np.max(em_col)) > 0.02 else 0.0,
          "alpha": float(np.clip(alp, 0.0, 1.0)),
          "tex_albedo": tex,
          "tex_roughness": np.clip(rgh_t, 0.02, 0.98) if rgh_t is not None else None,
          "uv_scale_offset": so,
          "uv_name": uvn,
      })
      return res

    if fn.type == "EMISSION":
      col, tex = self._eval_color(fn.inputs.get("Color"), (1.0, 1.0, 1.0), new_stack, depth + 1)
      st, _ = self._eval_scalar(fn.inputs.get("Strength"), 1.0, new_stack, depth + 1)
      res.update({
          "base_color": np.clip(col, 0.02, 1.0),
          "roughness": 0.9,
          "emissive_color": np.clip(col, 0.02, 1.0),
          "emissive_strength": float(max(0.1, st)),
          "tex_albedo": tex,
      })
      return res

    if fn.type in ("BSDF_GLASS", "BSDF_REFRACTION"):
      col, tex = self._eval_color(fn.inputs.get("Color"), (0.95, 0.98, 1.0), new_stack, depth + 1)
      rgh, _ = self._eval_scalar(fn.inputs.get("Roughness"), 0.04, new_stack, depth + 1)
      ior, _ = self._eval_scalar(fn.inputs.get("IOR"), 1.50, new_stack, depth + 1)
      res.update({
          "base_color": np.clip(col, 0.05, 0.99),
          "roughness": float(np.clip(rgh, 0.02, 0.25)),
          "transmission": 1.0,
          "ior": float(np.clip(ior, 1.0, 2.5)),
          "tex_albedo": tex,
      })
      return res

    if fn.type in ("BSDF_GLOSSY", "BSDF_ANISOTROPIC"):
      col, tex = self._eval_color(fn.inputs.get("Color"), (0.85, 0.85, 0.85), new_stack, depth + 1)
      rgh, rgh_t = self._eval_scalar(fn.inputs.get("Roughness"), 0.15, new_stack, depth + 1)
      so, uvn = self._find_first_tex_uv_meta(fn.inputs.get("Color"))
      res.update({
          "base_color": np.clip(col, 0.02, 0.99),
          "roughness": float(np.clip(rgh, 0.02, 0.70)),
          "metallic": 0.92,
          "tex_albedo": tex,
          "tex_roughness": np.clip(rgh_t, 0.02, 0.70) if rgh_t is not None else None,
          "uv_scale_offset": so,
          "uv_name": uvn,
      })
      return res

    if fn.type in ("BSDF_DIFFUSE", "BSDF_TRANSLUCENT", "BSDF_VELVET"):
      col, tex = self._eval_color(fn.inputs.get("Color"), (0.7, 0.7, 0.7), new_stack, depth + 1)
      rgh, _ = self._eval_scalar(fn.inputs.get("Roughness"), 0.75, new_stack, depth + 1)
      so, uvn = self._find_first_tex_uv_meta(fn.inputs.get("Color"))
      res.update({
          "base_color": np.clip(col, 0.01, 0.99),
          "roughness": float(np.clip(max(rgh, 0.45), 0.45, 0.98)),
          "metallic": 0.0,
          "transmission": 0.35 if fn.type == "BSDF_TRANSLUCENT" else 0.0,
          "tex_albedo": tex,
          "uv_scale_offset": so,
          "uv_name": uvn,
      })
      return res

    if fn.type == "BSDF_TRANSPARENT":
      col, _ = self._eval_color(fn.inputs.get("Color"), (1.0, 1.0, 1.0), new_stack, depth + 1)
      res.update({"base_color": col, "alpha": 0.0, "transmission": 1.0})
      return res

    return res

  def compile_material(self, mat, mat_id: int) -> Tuple[CompiledMaterial, Optional[np.ndarray], Optional[np.ndarray]]:
    if mat is None:
      cm = CompiledMaterial(
          mat_id=mat_id, name="default", base_color=(0.7, 0.7, 0.7),
          roughness=0.5, metallic=0.0, transmission=0.0, ior=1.45,
          emissive_color=(0.0, 0.0, 0.0), emissive_strength=0.0, alpha=1.0, mat_code=0.0,
      )
      return cm, None, None

    if not mat.use_nodes or mat.node_tree is None:
      dc = tuple(float(x) for x in getattr(mat, "diffuse_color", (0.7, 0.7, 0.7, 1.0))[:3])
      met = float(getattr(mat, "metallic", 0.0))
      rgh = float(getattr(mat, "roughness", 0.5))
      cm = CompiledMaterial(
          mat_id=mat_id, name=mat.name, base_color=dc, roughness=rgh,
          metallic=met, transmission=0.0, ior=1.45,
          emissive_color=(0.0, 0.0, 0.0), emissive_strength=0.0, alpha=1.0, mat_code=met,
      )
      return cm, None, None

    nodes = mat.node_tree.nodes
    out_node = next(
        (n for n in nodes if n.type == "OUTPUT_MATERIAL" and n.is_active_output and getattr(n, "target", "ALL") in ("ALL", "CYCLES")),
        None,
    )
    if out_node is None:
      out_node = next((n for n in nodes if n.type == "OUTPUT_MATERIAL"), None)

    if out_node is not None and out_node.inputs.get("Surface") is not None and out_node.inputs["Surface"].is_linked:
      props = self._eval_shader_socket(out_node.inputs["Surface"], [], depth=0)
    else:
      # Fallback: find Principled/Emission/Glass/Glossy directly
      princ = next((n for n in nodes if n.type == "BSDF_PRINCIPLED"), None)
      if princ is not None and len(princ.outputs) > 0:
        # Wrap pseudo-link evaluation
        props = {
            "base_color": self._eval_color(princ.inputs.get("Base Color"), (0.7, 0.7, 0.7), [])[0],
            "roughness": self._eval_scalar(princ.inputs.get("Roughness"), 0.5, [])[0],
            "metallic": self._eval_scalar(princ.inputs.get("Metallic"), 0.0, [])[0],
            "transmission": 0.0, "ior": 1.45,
            "emissive_color": np.zeros(3, np.float32), "emissive_strength": 0.0,
            "alpha": 1.0, "tex_albedo": None, "tex_roughness": None,
            "uv_scale_offset": (1.0, 1.0, 0.0, 0.0), "uv_name": None,
        }
      else:
        props = self._eval_shader_socket(None, [], depth=0)

    em_s = float(props["emissive_strength"])
    tr = float(props["transmission"])
    met = float(props["metallic"])
    bc = tuple(float(x) for x in props["base_color"])
    rgh = float(props["roughness"])

    if em_s > 0.25 and max(props["emissive_color"]) > 0.02:
      mcode = 3.0 + min(em_s, 20.0) * 0.1
      bc = tuple(float(np.clip(x, 0.02, 1.0)) for x in props["emissive_color"])
    elif tr > 0.45:
      mcode = 2.0
      rgh = min(rgh, 0.12)
    else:
      mcode = float(np.clip(met, 0.0, 1.0))

    cm = CompiledMaterial(
        mat_id=mat_id,
        name=mat.name,
        base_color=bc,
        roughness=float(np.clip(rgh, 0.02, 0.98)),
        metallic=met,
        transmission=tr,
        ior=float(props["ior"]),
        emissive_color=tuple(float(x) for x in props["emissive_color"]),
        emissive_strength=em_s,
        alpha=float(props["alpha"]),
        mat_code=mcode,
        uv_scale_offset=props["uv_scale_offset"],
    )
    return cm, props["tex_albedo"], props["tex_roughness"]


# ==============================================================================
# Blender Scene -> ScenePack Compiler
# ==============================================================================

def _extract_world_sky(scene) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]:
  sky_z = (0.35, 0.50, 0.72)
  sky_h = (0.55, 0.65, 0.78)
  sky_g = (0.14, 0.15, 0.17)
  world = getattr(scene, "world", None)
  if world is None:
    return sky_z, sky_h, sky_g
  if not world.use_nodes or world.node_tree is None:
    c = tuple(float(x) for x in getattr(world, "color", (0.2, 0.25, 0.3)))
    return c, c, tuple(0.3 * x for x in c)
  bg = next((n for n in world.node_tree.nodes if n.type == "BACKGROUND"), None)
  if bg is not None:
    st = float(bg.inputs["Strength"].default_value) if "Strength" in bg.inputs else 1.0
    col_s = bg.inputs.get("Color")
    c = np.array(col_s.default_value[:3], dtype=np.float32)
    if col_s is not None and col_s.is_linked:
      fn = col_s.links[0].from_node
      if fn.type == "TEX_SKY":
        c = np.array([0.45, 0.65, 0.95], dtype=np.float32)
      elif fn.type == "TEX_ENVIRONMENT" and fn.image is not None:
        try:
          w, h = fn.image.size[0], fn.image.size[1]
          if w > 0 and h > 0 and len(fn.image.pixels) == w * h * 4:
            px_flat = np.empty(w * h * 4, dtype=np.float32)
            fn.image.pixels.foreach_get(px_flat)
            px = px_flat.reshape(h, w, 4)[..., :3]
            c = np.clip(np.mean(px, axis=(0, 1)), 0.02, 2.0)
        except Exception:
          pass
    c_eff = tuple(float(np.clip(x * min(st, 3.0), 0.02, 2.0)) for x in c)
    return c_eff, tuple(min(1.5, x * 1.15) for x in c_eff), tuple(x * 0.25 for x in c_eff)
  return sky_z, sky_h, sky_g


def _extract_camera(scene) -> Tuple[np.ndarray, np.ndarray, float, Tuple[float, float], float, float, Tuple[int, int, int]]:
  cam_obj = scene.camera
  if cam_obj is None:
    cam_obj = next((o for o in bpy.data.objects if o.type == "CAMERA"), None)
  if cam_obj is None:
    raise RuntimeError("No camera found in .blend scene")
  cam_mat = np.array(cam_obj.matrix_world, dtype=np.float32)
  cam_pos = cam_mat[:3, 3].copy()
  cam_xmat = cam_mat[:3, :3].copy()
  # Orthonormalize rotation columns in case camera has scale
  for c in range(3):
    nrm = float(np.linalg.norm(cam_xmat[:, c]))
    if nrm > 1e-8:
      cam_xmat[:, c] /= nrm

  res_x = int(scene.render.resolution_x)
  res_y = int(scene.render.resolution_y)
  res_pct = int(scene.render.resolution_percentage)
  eff_w = max(1, (res_x * res_pct) // 100)
  eff_h = max(1, (res_y * res_pct) // 100)
  aspect = float(eff_w) / float(eff_h)

  cd = cam_obj.data
  fit = getattr(cd, "sensor_fit", "AUTO")
  if fit == "VERTICAL":
    fovy_rad = 2.0 * math.atan(0.5 * float(cd.sensor_height) / max(float(cd.lens), 1e-4))
  elif fit == "HORIZONTAL" or (fit == "AUTO" and aspect >= 1.0):
    fovx_rad = 2.0 * math.atan(0.5 * float(cd.sensor_width) / max(float(cd.lens), 1e-4))
    fovy_rad = 2.0 * math.atan(math.tan(0.5 * fovx_rad) / max(aspect, 1e-4))
  else:
    fovy_rad = 2.0 * math.atan(0.5 * float(cd.sensor_width) / max(float(cd.lens), 1e-4))
  cam_fovy = float(math.degrees(fovy_rad))
  shift = (float(getattr(cd, "shift_x", 0.0)), float(getattr(cd, "shift_y", 0.0)))
  znear = max(float(getattr(cd, "clip_start", 0.05)), 1e-3)
  zfar = max(float(getattr(cd, "clip_end", 100.0)), 80.0)
  return cam_pos, cam_xmat, cam_fovy, shift, znear, zfar, (res_x, res_y, res_pct)


def _extract_lights(scene) -> List[Dict[str, Any]]:
  lights: List[Dict[str, Any]] = []
  for obj in bpy.data.objects:
    if obj.type != "LIGHT" or obj.hide_render:
      continue
    ld = obj.data
    mw = np.array(obj.matrix_world, dtype=np.float32)
    l_dir = -mw[:3, 2]
    l_dir = l_dir / max(float(np.linalg.norm(l_dir)), 1e-12)
    lights.append({
        "name": obj.name,
        "type": ld.type,
        "pos": mw[:3, 3].tolist(),
        "dir": l_dir.tolist(),
        "color": [float(x) for x in ld.color[:3]],
        "energy": float(ld.energy),
        "size": float(getattr(ld, "size", 0.5)),
        "spot_size": float(getattr(ld, "spot_size", math.pi * 0.25)),
    })
  return lights


def compile_active_blend(
    scene_name: Optional[str] = None,
    max_tex_res: int = 256,
) -> ScenePack:
  """Compile currently loaded Blender scene (`bpy.context.scene`) into `ScenePack` IR."""
  assert HAS_BPY, "compile_active_blend must run inside Blender (bpy)"
  t0 = time.perf_counter()
  scene = bpy.context.scene
  depsgraph = bpy.context.evaluated_depsgraph_get()
  walker = _ShaderGraphWalker(tex_res=max_tex_res)

  mat_map: Dict[int, CompiledMaterial] = {}
  mat_list: List[CompiledMaterial] = []
  tex_list: List[np.ndarray] = []

  def _register_mat(mat) -> CompiledMaterial:
    ptr = mat.as_pointer() if mat is not None else -1
    if ptr in mat_map:
      return mat_map[ptr]
    mid = len(mat_list)
    cm, tex_alb, tex_rgh = walker.compile_material(mat, mid)
    if tex_alb is not None:
      tid = len(tex_list)
      rgh_grid = tex_rgh if tex_rgh is not None else np.full((max_tex_res, max_tex_res), cm.roughness, dtype=np.float32)
      if tex_alb.shape[:2] != (max_tex_res, max_tex_res):
        tex_alb = _resize_bilinear(tex_alb, max_tex_res, max_tex_res)
      if rgh_grid.shape[:2] != (max_tex_res, max_tex_res):
        rgh_grid = _resize_bilinear(rgh_grid, max_tex_res, max_tex_res)
      tex_list.append(np.concatenate([tex_alb, rgh_grid[..., None]], axis=-1).astype(np.float32))
      cm.tex_id = tid
    mat_map[ptr] = cm
    mat_list.append(cm)
    return cm

  all_v: List[np.ndarray] = []
  all_f: List[np.ndarray] = []
  all_n: List[np.ndarray] = []
  all_uv: List[np.ndarray] = []
  all_mid: List[np.ndarray] = []
  all_alb: List[np.ndarray] = []
  all_rgh: List[np.ndarray] = []
  all_mc: List[np.ndarray] = []

  v_off = 0
  seen_blas = set()
  n_inst = 0

  for inst in depsgraph.object_instances:
    obj = inst.object
    if obj.hide_render or (not getattr(obj, "visible_camera", True)) or obj.type not in ("MESH", "CURVE", "SURFACE", "FONT", "META"):
      continue
    me = obj.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
    if me is None or len(me.polygons) == 0:
      obj.to_mesh_clear()
      continue
    me.calc_loop_triangles()
    n_verts = len(me.vertices)
    n_tris = len(me.loop_triangles)
    if n_verts == 0 or n_tris == 0:
      obj.to_mesh_clear()
      continue

    orig_ptr = obj.original.data.as_pointer() if getattr(obj, "original", None) and obj.original.data else id(me)
    seen_blas.add(orig_ptr)
    n_inst += 1

    verts_loc = np.empty((n_verts, 3), dtype=np.float32)
    me.vertices.foreach_get("co", verts_loc.ravel())

    mat_w = np.array(inst.matrix_world, dtype=np.float32)
    R = mat_w[:3, :3]
    t = mat_w[:3, 3]
    verts_w = verts_loc @ R.T + t[None, :]
    det = float(np.linalg.det(R))
    R_n = np.linalg.inv(R).T if abs(det) > 1e-8 else R

    tri_v = np.empty((n_tris * 3,), dtype=np.int32)
    me.loop_triangles.foreach_get("vertices", tri_v)
    faces = tri_v.reshape(n_tris, 3)

    vnorm_loc = np.empty((n_verts, 3), dtype=np.float32)
    me.vertices.foreach_get("normal", vnorm_loc.ravel())
    vnorm_w = vnorm_loc @ R_n.T
    vnorm_w = vnorm_w / np.maximum(np.linalg.norm(vnorm_w, axis=-1, keepdims=True), 1e-12)

    tri_mat_idx = np.empty((n_tris,), dtype=np.int32)
    me.loop_triangles.foreach_get("material_index", tri_mat_idx)

    tri_loops = np.empty((n_tris * 3,), dtype=np.int32)
    me.loop_triangles.foreach_get("loops", tri_loops)
    tri_loops = tri_loops.reshape(n_tris, 3)

    if det < 0.0:
      faces = faces[:, [0, 2, 1]]
      tri_loops = tri_loops[:, [0, 2, 1]]

    uv_layer = me.uv_layers.active.data if (len(me.uv_layers) > 0 and me.uv_layers.active is not None) else None
    if uv_layer is not None:
      n_loops = len(me.loops)
      uvs_flat = np.empty((n_loops * 2,), dtype=np.float32)
      uv_layer.foreach_get("uv", uvs_flat)
      uvs_flat = uvs_flat.reshape(n_loops, 2)
      tri_uv = uvs_flat[tri_loops]  # (n_tris, 3, 2)
    else:
      tri_uv = np.zeros((n_tris, 3, 2), dtype=np.float32)

    # Check active vertex color attribute if present
    vcol_rgb = None
    if len(me.color_attributes) > 0:
      ca = me.color_attributes.active_color
      if ca is not None and ca.domain == "CORNER":
        c_flat = np.empty((len(me.loops) * 4,), dtype=np.float32)
        ca.data.foreach_get("color", c_flat)
        c_flat = c_flat.reshape(len(me.loops), 4)[..., :3]
        vcol_rgb = np.mean(c_flat[tri_loops], axis=1)

    mats = list(me.materials) if len(me.materials) > 0 else [None]
    mid_f = np.zeros((n_tris,), dtype=np.int32)
    alb_f = np.full((n_tris, 3), 0.70, dtype=np.float32)
    rgh_f = np.full((n_tris,), 0.50, dtype=np.float32)
    mc_f = np.full((n_tris,), 0.0, dtype=np.float32)

    for slot_i, mat in enumerate(mats):
      mask = (tri_mat_idx == slot_i) if len(mats) > 1 else np.ones((n_tris,), dtype=bool)
      if not np.any(mask):
        continue
      cm = _register_mat(mat)
      mid_f[mask] = cm.mat_id
      rgh_f[mask] = cm.roughness
      mc_f[mask] = cm.mat_code
      sx, sy, ox, oy = cm.uv_scale_offset
      if (sx, sy, ox, oy) != (1.0, 1.0, 0.0, 0.0):
        tri_uv[mask, :, 0] = tri_uv[mask, :, 0] * sx + ox
        tri_uv[mask, :, 1] = tri_uv[mask, :, 1] * sy + oy

      if cm.tex_id >= 0 and uv_layer is not None:
        tex_arr = tex_list[cm.tex_id]
        th, tw, _ = tex_arr.shape
        uv_c = np.mean(tri_uv[mask], axis=1)
        u = np.mod(uv_c[:, 0], 1.0)
        v = np.mod(1.0 - uv_c[:, 1], 1.0)
        px = np.clip((u * (tw - 1)).astype(np.int32), 0, tw - 1)
        py = np.clip((v * (th - 1)).astype(np.int32), 0, th - 1)
        alb_f[mask] = np.clip(tex_arr[py, px, :3], 0.01, 0.99)
        rgh_f[mask] = np.clip(tex_arr[py, px, 3], 0.02, 0.98)
      elif vcol_rgb is not None and cm.base_color == (0.7, 0.7, 0.7):
        alb_f[mask] = np.clip(vcol_rgb[mask], 0.01, 0.99)
      else:
        alb_f[mask] = np.asarray(cm.base_color, dtype=np.float32)[None, :]

    all_v.append(verts_w.astype(np.float32))
    all_f.append((faces + v_off).astype(np.int32))
    all_n.append(vnorm_w.astype(np.float32))
    all_uv.append(tri_uv.astype(np.float32))
    all_mid.append(mid_f)
    all_alb.append(alb_f)
    all_rgh.append(rgh_f)
    all_mc.append(mc_f)
    v_off += n_verts
    obj.to_mesh_clear()

  if not all_v:
    raise RuntimeError("No renderable mesh geometry found in .blend scene")

  verts = np.concatenate(all_v, axis=0)
  faces = np.concatenate(all_f, axis=0)
  vnorm = np.concatenate(all_n, axis=0)
  tri_uvs = np.concatenate(all_uv, axis=0)
  tri_mat_id = np.concatenate(all_mid, axis=0)
  tri_albedo = np.concatenate(all_alb, axis=0)
  tri_roughness = np.concatenate(all_rgh, axis=0)
  tri_mat_code = np.concatenate(all_mc, axis=0)

  tex_atlas = np.stack(tex_list, axis=0).astype(np.float32) if tex_list else np.zeros((0, max_tex_res, max_tex_res, 4), np.float32)
  mat_tex_ids = np.array([m.tex_id for m in mat_list], dtype=np.int32)
  mat_base_rgba = np.array([[*m.base_color, m.roughness] for m in mat_list], dtype=np.float32)
  mat_codes = np.array([m.mat_code for m in mat_list], dtype=np.float32)

  cam_pos, cam_xmat, cam_fovy, cam_shift, znear, zfar, native_res = _extract_camera(scene)
  lights = _extract_lights(scene)
  sky_z, sky_h, sky_g = _extract_world_sky(scene)
  sname = scene_name or os.path.splitext(os.path.basename(bpy.data.filepath or "untitled.blend"))[0]

  return ScenePack(
      scene_name=sname,
      verts=verts,
      faces=faces,
      vnorm=vnorm,
      tri_uvs=tri_uvs,
      tri_mat_id=tri_mat_id,
      tri_albedo=tri_albedo,
      tri_roughness=tri_roughness,
      tri_mat_code=tri_mat_code,
      tex_atlas=tex_atlas,
      mat_tex_ids=mat_tex_ids,
      mat_base_rgba=mat_base_rgba,
      mat_codes=mat_codes,
      cam_pos=cam_pos,
      cam_xmat=cam_xmat,
      cam_fovy=cam_fovy,
      cam_shift=cam_shift,
      znear=znear,
      zfar=zfar,
      native_res=native_res,
      lights=lights,
      sky_zenith=sky_z,
      sky_horizon=sky_h,
      sky_ground=sky_g,
      materials=[dataclasses.asdict(m) for m in mat_list],
      n_blas_meshes=len(seen_blas),
      n_instances=n_inst,
      compile_meta={"compile_s": round(time.perf_counter() - t0, 3)},
  )


# ==============================================================================
# Perturbed Orbit Camera Generator & Cycles 5.3.0 GT + EXR Z-Depth Exporter
# ==============================================================================

def _look_at_cam_xmat(cam_pos: np.ndarray, target: np.ndarray, up_hint: np.ndarray) -> np.ndarray:
  fwd = target - cam_pos
  fwd = fwd / max(float(np.linalg.norm(fwd)), 1e-12)
  right = np.cross(fwd, up_hint)
  if float(np.linalg.norm(right)) < 1e-6:
    right = np.cross(fwd, np.array([0.0, 1.0, 0.0], np.float32))
  right = right / max(float(np.linalg.norm(right)), 1e-12)
  up = np.cross(right, fwd)
  return np.stack([right, up, -fwd], axis=1).astype(np.float32)


def _generate_perturbed_views_in_blender(
    scene,
    cam_pos: np.ndarray,
    cam_xmat: np.ndarray,
    n_orbit: int,
    phase_offset: float = 0.0,
    max_yaw: float = 0.14,
    max_pitch: float = 0.08,
    include_hero: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
  """Generate safe perturbed orbit views around hero camera, avoiding wall occlusion via raycast."""
  from mathutils import Vector
  depsgraph = bpy.context.evaluated_depsgraph_get()
  fwd0 = -cam_xmat[:, 2]
  right0 = cam_xmat[:, 0]
  up0 = cam_xmat[:, 1]

  # Raycast hero center ray to find focal target distance
  hit, loc, *_ = scene.ray_cast(depsgraph, Vector(cam_pos), Vector(fwd0))
  dist_focus = float((loc - Vector(cam_pos)).length) if hit else 4.0
  dist_focus = float(np.clip(dist_focus, 1.5, 10.0))
  target = cam_pos + fwd0 * dist_focus

  c_pos_list = [cam_pos.copy()] if include_hero else []
  c_mat_list = [cam_xmat.copy()] if include_hero else []
  start_k = 1 if include_hero else 0
  if n_orbit <= start_k:
    return np.stack(c_pos_list, axis=0), np.stack(c_mat_list, axis=0)

  denom = float(max(n_orbit - start_k, 1))
  for k in range(start_k, n_orbit):
    idx = k - start_k
    ang = 2.0 * math.pi * (idx / denom) + phase_offset
    yaw_off = max_yaw * math.sin(ang)
    pitch_off = max_pitch * math.cos(ang)
    rad_scale = 1.0 + 0.05 * math.sin(2.0 * ang)

    offset = -fwd0 * (dist_focus * rad_scale)
    offset = offset + right0 * (dist_focus * math.sin(yaw_off)) + up0 * (dist_focus * math.sin(pitch_off))
    cand = target + offset
    # Verify line of sight from hero cam_pos -> cand isn't blocked by a wall
    delta = cand - cam_pos
    dlen = float(np.linalg.norm(delta))
    if dlen > 1e-4:
      blk, bloc, *_ = scene.ray_cast(depsgraph, Vector(cam_pos), Vector(delta / dlen), distance=dlen)
      if blk:
        safe_d = max(0.0, float((bloc - Vector(cam_pos)).length) - 0.22)
        cand = cam_pos + (delta / dlen) * safe_d

    cmat = _look_at_cam_xmat(cand.astype(np.float32), target, np.array([0.0, 0.0, 1.0], np.float32))
    c_pos_list.append(cand.astype(np.float32))
    c_mat_list.append(cmat)

  return np.stack(c_pos_list, axis=0), np.stack(c_mat_list, axis=0)


def render_cycles_gt_and_depth(
    pack: ScenePack,
    out_gt_npz: str,
    width: int,
    height: int,
    samples: int = 48,
    n_views: int = 1,
    hero_png_path: Optional[str] = None,
    phase_offset: float = 0.0,
    include_hero: bool = True,
) -> None:
  """Render Cycles 5.3.0 RGB GT (scene native color management) + EXR Z-depth inside Blender."""
  assert HAS_BPY, "render_cycles_gt_and_depth must run inside Blender"
  from mathutils import Matrix
  scene = bpy.context.scene
  cam_obj = scene.camera or next(o for o in bpy.data.objects if o.type == "CAMERA")

  cam_xpos, cam_xmat = _generate_perturbed_views_in_blender(
      scene, pack.cam_pos, pack.cam_xmat, n_views, phase_offset=phase_offset, include_hero=include_hero
  )

  scene.render.engine = "CYCLES"
  scene.cycles.device = "CPU"
  scene.cycles.samples = int(samples)
  scene.cycles.use_denoising = True
  scene.cycles.use_adaptive_sampling = True
  scene.cycles.adaptive_threshold = 0.02
  scene.render.resolution_x = int(width)
  scene.render.resolution_y = int(height)
  scene.render.resolution_percentage = 100
  scene.render.image_settings.file_format = "PNG"
  scene.render.image_settings.color_mode = "RGB"
  scene.render.image_settings.color_depth = "8"

  tmpdir = tempfile.mkdtemp(prefix="blend_gt_", dir=os.environ.get("TMPDIR", "/tmp"))
  rgb_all = np.zeros((n_views, height, width, 3), dtype=np.uint8)
  f0_depth = np.zeros((height, width), dtype=np.float32)

  # Set up compositor EXR depth output for Frame 0
  bpy.context.view_layer.use_pass_z = True
  orig_use_comp = bool(scene.render.use_compositing)
  scene.render.use_compositing = True
  orig_comp = getattr(scene, "compositing_node_group", None)
  ng = bpy.data.node_groups.new("BlendCompilerDepthComp", "CompositorNodeTree")
  scene.compositing_node_group = ng
  rl = ng.nodes.new("CompositorNodeRLayers")
  fout = ng.nodes.new("CompositorNodeOutputFile")
  fout.directory = tmpdir
  fout.file_name = "depth_"
  fout.format.media_type = "IMAGE"
  fout.format.file_format = "OPEN_EXR"
  fout.format.color_depth = "32"
  fout.file_output_items.clear()
  fout.file_output_items.new("FLOAT", "Depth")
  ng.links.new(rl.outputs["Depth"], fout.inputs["Depth"])

  # Save original camera matrix
  orig_mat = cam_obj.matrix_world.copy()
  t0 = time.perf_counter()
  for k in range(n_views):
    m = cam_xmat[k]
    p = cam_xpos[k]
    cam_obj.matrix_world = Matrix((
        (float(m[0, 0]), float(m[0, 1]), float(m[0, 2]), float(p[0])),
        (float(m[1, 0]), float(m[1, 1]), float(m[1, 2]), float(p[1])),
        (float(m[2, 0]), float(m[2, 1]), float(m[2, 2]), float(p[2])),
        (0.0, 0.0, 0.0, 1.0),
    ))
    if k == 1:
      scene.compositing_node_group = orig_comp
    png_p = os.path.join(tmpdir, f"view_{k:03d}.png")
    scene.render.filepath = png_p
    bpy.ops.render.render(write_still=True)

    im = bpy.data.images.load(png_p, check_existing=False)
    arr_flat = np.empty(width * height * 4, dtype=np.float32)
    im.pixels.foreach_get(arr_flat)
    arr = (np.clip(arr_flat.reshape(height, width, 4)[..., :3][::-1], 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    bpy.data.images.remove(im)
    rgb_all[k] = arr
    if k == 0:
      if hero_png_path:
        os.makedirs(os.path.dirname(os.path.abspath(hero_png_path)), exist_ok=True)
        shutil.copy2(png_p, hero_png_path)
      d_files = sorted(f for f in os.listdir(tmpdir) if f.startswith("depth_") and f.endswith(".exr"))
      if d_files:
        dimg = bpy.data.images.load(os.path.join(tmpdir, d_files[0]), check_existing=False)
        d_flat = np.empty(width * height * 4, dtype=np.float32)
        dimg.pixels.foreach_get(d_flat)
        f0_depth = d_flat.reshape(height, width, 4)[..., 0][::-1].copy()
        bpy.data.images.remove(dimg)

  cam_obj.matrix_world = orig_mat
  scene.compositing_node_group = orig_comp
  scene.render.use_compositing = orig_use_comp
  shutil.rmtree(tmpdir, ignore_errors=True)

  os.makedirs(os.path.dirname(os.path.abspath(out_gt_npz)), exist_ok=True)
  np.savez(
      out_gt_npz,
      rgb=rgb_all,
      frame0_z_depth=f0_depth,
      cam_xpos=cam_xpos,
      cam_xmat=cam_xmat,
      width=np.int32(width),
      height=np.int32(height),
      samples=np.int32(samples),
      render_s=np.float32(time.perf_counter() - t0),
  )
  print(f"SAVED GT {out_gt_npz}: {n_views} views @ {width}x{height} ({samples} spp) in {time.perf_counter()-t0:.2f}s", flush=True)


# ==============================================================================
# Host-Python Entrypoint: compile_blend()
# ==============================================================================

def find_blender_binary(blender_bin: Optional[str] = None) -> str:
  candidates = [
      blender_bin,
      os.environ.get("BLENDER_BIN"),
      shutil.which("blender"),
  ]
  for root in ("/dev/shm/zaychik-tpu-mjx/opendata", "/dev/shm", "/opt", "/usr/local"):
    if os.path.isdir(root):
      for dirpath, _, filenames in os.walk(root):
        if "blender" in filenames:
          p = os.path.join(dirpath, "blender")
          if os.access(p, os.X_OK):
            candidates.append(p)
  for c in candidates:
    if c and os.path.isfile(c) and os.access(c, os.X_OK):
      return c
  raise FileNotFoundError("Could not find Blender executable. Pass blender_bin= or set BLENDER_BIN.")


def compile_blend(
    blend_path: str,
    out_npz: Optional[str] = None,
    blender_bin: Optional[str] = None,
    max_tex_res: int = 256,
    render_gt_npz: Optional[str] = None,
    gt_width: int = 512,
    gt_height: int = 512,
    gt_samples: int = 64,
    n_views: int = 1,
    hero_png_path: Optional[str] = None,
) -> ScenePack:
  """Compile any `.blend` file into `ScenePack` IR (invoking headless Blender if needed)."""
  if HAS_BPY:
    bpy.ops.wm.open_mainfile(filepath=os.path.abspath(blend_path))
    pack = compile_active_blend(max_tex_res=max_tex_res)
    if out_npz:
      pack.save(out_npz)
    if render_gt_npz:
      render_cycles_gt_and_depth(
          pack, render_gt_npz, width=gt_width, height=gt_height,
          samples=gt_samples, n_views=n_views, hero_png_path=hero_png_path,
      )
    return pack

  bl = find_blender_binary(blender_bin)
  cleanup_npz = False
  if out_npz is None:
    tmp_f = tempfile.NamedTemporaryFile(suffix=".npz", delete=False, dir=os.environ.get("TMPDIR", "/tmp"))
    out_npz = tmp_f.name
    tmp_f.close()
    cleanup_npz = True

  cmd = [
      bl, "-b", os.path.abspath(blend_path),
      "--python", os.path.abspath(__file__),
      "--",
      "--out_npz", out_npz,
      "--max_tex_res", str(max_tex_res),
  ]
  if render_gt_npz:
    cmd += [
        "--gt_npz", render_gt_npz,
        "--gt_w", str(gt_width),
        "--gt_h", str(gt_height),
        "--gt_spp", str(gt_samples),
        "--n_views", str(n_views),
    ]
    if hero_png_path:
      cmd += ["--hero_png", hero_png_path]

  env = os.environ.copy()
  env.pop("PYTHONPATH", None)  # Keep Blender's embedded Python clean
  extra_lib = "/dev/shm/zaychik_libs/root/usr/lib/x86_64-linux-gnu"
  if os.path.isdir(extra_lib):
    env["LD_LIBRARY_PATH"] = f"{extra_lib}:{env.get('LD_LIBRARY_PATH', '')}"
  subprocess.run(cmd, check=True, env=env)
  pack = ScenePack.load(out_npz)
  if cleanup_npz and os.path.exists(out_npz):
    os.remove(out_npz)
  return pack


def _main_blender():
  import argparse
  argv = sys.argv
  argv = argv[argv.index("--") + 1 :] if "--" in argv else []
  ap = argparse.ArgumentParser()
  ap.add_argument("--out_npz", default=None)
  ap.add_argument("--scene_name", default=None)
  ap.add_argument("--max_tex_res", type=int, default=256)
  ap.add_argument("--gt_npz", default=None)
  ap.add_argument("--gt_w", type=int, default=512)
  ap.add_argument("--gt_h", type=int, default=512)
  ap.add_argument("--gt_spp", type=int, default=48)
  ap.add_argument("--n_views", type=int, default=1)
  ap.add_argument("--hero_png", default=None)
  ap.add_argument("--orbit_npz", default=None)
  ap.add_argument("--orbit_w", type=int, default=256)
  ap.add_argument("--orbit_h", type=int, default=256)
  ap.add_argument("--orbit_spp", type=int, default=32)
  ap.add_argument("--orbit_views", type=int, default=24)
  args = ap.parse_args(argv)

  pack = compile_active_blend(scene_name=args.scene_name, max_tex_res=args.max_tex_res)
  if args.out_npz:
    pack.save(args.out_npz)
  print("COMPILED BLEND:", json.dumps(pack.summary()), flush=True)
  if args.orbit_npz:
    render_cycles_gt_and_depth(
        pack, args.orbit_npz, width=args.orbit_w, height=args.orbit_h,
        samples=args.orbit_spp, n_views=args.orbit_views, phase_offset=0.0, include_hero=True,
    )
  if args.gt_npz:
    render_cycles_gt_and_depth(
        pack, args.gt_npz, width=args.gt_w, height=args.gt_h,
        samples=args.gt_spp, n_views=args.n_views, hero_png_path=args.hero_png,
        phase_offset=0.37, include_hero=True,
    )


if __name__ == "__main__":
  if HAS_BPY:
    _main_blender()
  else:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("blend_path")
    ap.add_argument("--out_npz", required=True)
    ap.add_argument("--max_tex_res", type=int, default=256)
    ap.add_argument("--gt_npz", default=None)
    ap.add_argument("--gt_w", type=int, default=512)
    ap.add_argument("--gt_h", type=int, default=512)
    ap.add_argument("--gt_spp", type=int, default=64)
    ap.add_argument("--n_views", type=int, default=1)
    ap.add_argument("--hero_png", default=None)
    args = ap.parse_args()
    p = compile_blend(
        args.blend_path, out_npz=args.out_npz, max_tex_res=args.max_tex_res,
        render_gt_npz=args.gt_npz, gt_width=args.gt_w, gt_height=args.gt_h,
        gt_samples=args.gt_spp, n_views=args.n_views, hero_png_path=args.hero_png,
    )
    print("COMPILED BLEND:", json.dumps(p.summary()), flush=True)
