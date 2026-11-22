"""Complex Artist Khronos PBR Scenes (Sponza, ABeautifulGame, FlightHelmet) Natively on TPU v5p.

Features:
  - Loads Khronos glTF 2.0 assets via `trimesh` into Wide-BVH-64 (`leaf_rows=24` primary / `10` shadow).
  - Full UV-mapped PBR: Bilinear Albedo (sRGB->Linear), Roughness, Metallic, and Screen-Space
    Derivative Tangent-Frame Normal Mapping (Schuler cotangent frame, zero extra BVH rows).
  - R2-stratified 8-sample Hemisphere Ambient Occlusion + 4-sample Soft Area Shadow via
    Mosaic `any_hit` (`leaf_rows=10`), Cook-Torrance GGX + Hemisphere Sky IBL + AgX Tonemapping.
  - Benchmarked on 1-chip and 4-chip `shard_map` with `host_callbacks == 0` & `send_recv_ops == 0`.
"""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import jax
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import trimesh

from blend_compiler import ScenePack, _resize_bilinear, _srgb_to_linear
from run_w5_showcase import _hlo_audit
from wide_bvh_v3 import _compute_smooth_vertex_normals
from wide_bvh64_uv import (
    WideBVH64UVSceneSpec,
    _linear_to_srgb_jnp,
    _sample_tex_atlas_bilinear,
    build_wide_bvh64_uv,
    compute_blend_camera_rays,
    pack_dev_arrays,
    pallas_wide_bvh64_uv_raycast,
)


def _agx_tonemap_jnp(lin: jax.Array, exposure: float = 1.0) -> jax.Array:
  """Blender-aligned AgX-style punchy filmic tonemap + sRGB encoding."""
  x = jnp.maximum(lin * float(exposure), 0.0)
  # AgX inset matrix (sRGB -> AgX working space)
  m_in = jnp.array([
      [0.84247906, 0.07843360, 0.07922375],
      [0.04232824, 0.87846864, 0.07916613],
      [0.04237565, 0.07843360, 0.87914297],
  ], dtype=jnp.float32)
  m_out = jnp.array([
      [1.19687900, -0.09802088, -0.09902974],
      [-0.05289685, 1.15190313, -0.09896118],
      [-0.05297163, -0.09804345, 1.15107367],
  ], dtype=jnp.float32)
  x = jnp.einsum("ij,...j->...i", m_in, x)
  # Log2 compression + sigmoid curve
  x_log = jnp.clip((jnp.log2(jnp.maximum(x, 1e-6)) + 10.0) / 16.5, 0.0, 1.0)
  x_sig = x_log * x_log * (3.0 - 2.0 * x_log)
  x_sig = jnp.power(x_sig, 1.05)
  out = jnp.einsum("ij,...j->...i", m_out, x_sig)
  return jnp.clip(out, 0.0, 1.0)


def _pil_to_np(im: Any, tex_res: int, as_linear: bool = True) -> np.ndarray:
  arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
  arr = _resize_bilinear(arr, tex_res, tex_res)
  if as_linear:
    arr = _srgb_to_linear(arr)
  return np.clip(arr, 0.0, 1.0).astype(np.float32)


def load_gltf_to_scenepack(
    gltf_path: str,
    scene_name: str,
    cam_pos: Tuple[float, float, float],
    cam_tgt: Tuple[float, float, float],
    cam_fovy: float = 45.0,
    tex_res: int = 256,
    add_floor: bool = False,
) -> Tuple[ScenePack, np.ndarray]:
  """Load Khronos glTF 2.0 asset into ScenePack + (T, TH, TW, 4) normal+metallic atlas."""
  sc = trimesh.load(gltf_path, force="scene")
  R_yz = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)

  all_v, all_f, all_n, all_uv = [], [], [], []
  all_mid, all_alb, all_rgh, all_mc = [], [], [], []
  tex_alb_rgh: List[np.ndarray] = []
  tex_nrm_met: List[np.ndarray] = []
  mat_tex_ids: List[int] = []
  mat_base_rgba: List[List[float]] = []
  mat_codes: List[float] = []
  mat_records: List[Dict[str, Any]] = []
  mat_cache: Dict[int, int] = {}

  v_off = 0
  for node_name in sc.graph.nodes_geometry:
    xform, geom_name = sc.graph[node_name]
    geom = sc.geometry[geom_name]
    if not isinstance(geom, trimesh.Trimesh) or len(geom.faces) == 0:
      continue
    v_loc = np.asarray(geom.vertices, dtype=np.float32)
    f_loc = np.asarray(geom.faces, dtype=np.int32)
    R = xform[:3, :3].astype(np.float32)
    t = xform[:3, 3].astype(np.float32)
    v_w = (v_loc @ R.T + t[None, :]) @ R_yz.T
    if np.linalg.det(R) < 0:
      f_loc = f_loc[:, [0, 2, 1]]

    # Normals
    try:
      vn_loc = np.asarray(geom.vertex_normals, dtype=np.float32)
      if vn_loc.shape != v_loc.shape:
        vn_loc = _compute_smooth_vertex_normals(v_loc, f_loc)
    except Exception:
      vn_loc = _compute_smooth_vertex_normals(v_loc, f_loc)
    R_n = np.linalg.inv(R).T if abs(np.linalg.det(R)) > 1e-8 else R
    vn_w = (vn_loc @ R_n.T) @ R_yz.T
    vn_w = vn_w / np.maximum(np.linalg.norm(vn_w, axis=-1, keepdims=True), 1e-12)

    nf = f_loc.shape[0]
    vis = getattr(geom, "visual", None)
    uv = getattr(vis, "uv", None)
    if uv is not None and len(uv) == len(v_loc):
      # glTF UV v=0 is top, Blender/our tex_atlas expects v=0 bottom or 1-v
      uv_np = np.asarray(uv, dtype=np.float32)
      tri_uv = uv_np[f_loc]
    else:
      tri_uv = np.zeros((nf, 3, 2), dtype=np.float32)

    mat = getattr(vis, "material", None) if vis is not None else None
    mkey = id(mat) if mat is not None else -1
    if mkey not in mat_cache:
      mid = len(mat_records)
      mat_cache[mkey] = mid
      base_col = np.array([0.75, 0.75, 0.75], dtype=np.float32)
      rgh_f = 0.45
      met_f = 0.0
      tid = -1
      if mat is not None:
        bcf = getattr(mat, "baseColorFactor", None)
        if bcf is not None:
          ba = np.asarray(bcf, dtype=np.float32).ravel()
          if ba.size >= 3:
            base_col = _srgb_to_linear(ba[:3] / (255.0 if ba.max() > 1.5 else 1.0))
        if getattr(mat, "roughnessFactor", None) is not None:
          rgh_f = float(np.clip(mat.roughnessFactor, 0.04, 0.98))
        if getattr(mat, "metallicFactor", None) is not None:
          met_f = float(np.clip(mat.metallicFactor, 0.0, 1.0))

        bct = getattr(mat, "baseColorTexture", None)
        mrt = getattr(mat, "metallicRoughnessTexture", None)
        nmt = getattr(mat, "normalTexture", None)
        if bct is not None or mrt is not None or nmt is not None:
          tid = len(tex_alb_rgh)
          alb_g = _pil_to_np(bct, tex_res, as_linear=True) * base_col[None, None, :] if bct is not None else np.broadcast_to(base_col[None, None, :], (tex_res, tex_res, 3)).copy()
          if mrt is not None:
            mr_g = _pil_to_np(mrt, tex_res, as_linear=False)
            rgh_g = np.clip(mr_g[..., 1] * rgh_f, 0.04, 0.98)
            met_g = np.clip(mr_g[..., 2] * max(met_f, 1.0 if mr_g[..., 2].max() > 0.1 else met_f), 0.0, 1.0)
          else:
            rgh_g = np.full((tex_res, tex_res), rgh_f, dtype=np.float32)
            met_g = np.full((tex_res, tex_res), met_f, dtype=np.float32)

          if nmt is not None:
            nm_g = _pil_to_np(nmt, tex_res, as_linear=False)
          else:
            nm_g = np.broadcast_to(np.array([0.5, 0.5, 1.0], np.float32), (tex_res, tex_res, 3)).copy()

          tex_alb_rgh.append(np.concatenate([alb_g, rgh_g[..., None]], axis=-1).astype(np.float32))
          tex_nrm_met.append(np.concatenate([nm_g, met_g[..., None]], axis=-1).astype(np.float32))

      mat_tex_ids.append(tid)
      mat_base_rgba.append([float(base_col[0]), float(base_col[1]), float(base_col[2]), rgh_f])
      mat_codes.append(met_f)
      mat_records.append({"name": str(getattr(mat, "name", f"mat_{mid}")), "tex_id": tid, "metallic": met_f, "roughness": rgh_f})
    else:
      mid = mat_cache[mkey]

    bc = np.asarray(mat_base_rgba[mid][:3], dtype=np.float32)
    rg = float(mat_base_rgba[mid][3])
    mc = float(mat_codes[mid])

    all_v.append(v_w)
    all_f.append(f_loc + v_off)
    all_n.append(vn_w.astype(np.float32))
    all_uv.append(tri_uv.astype(np.float32))
    all_mid.append(np.full((nf,), mid, dtype=np.int32))
    all_alb.append(np.broadcast_to(bc[None, :], (nf, 3)).copy())
    all_rgh.append(np.full((nf,), rg, dtype=np.float32))
    all_mc.append(np.full((nf,), mc, dtype=np.float32))
    v_off += v_w.shape[0]

  v_all = np.concatenate(all_v, axis=0)
  f_all = np.concatenate(all_f, axis=0)
  n_all = np.concatenate(all_n, axis=0)
  uv_all = np.concatenate(all_uv, axis=0)
  mid_all = np.concatenate(all_mid, axis=0)
  alb_all = np.concatenate(all_alb, axis=0)
  rgh_all = np.concatenate(all_rgh, axis=0)
  mc_all = np.concatenate(all_mc, axis=0)

  # Normalize scale to museum diorama / architectural courtyard
  bmin, bmax = v_all.min(0), v_all.max(0)
  span = float(np.max(bmax - bmin))
  target_span = 18.0 if "sponza" in scene_name.lower() else 1.6
  scl = target_span / max(span, 1e-5)
  cxy = 0.5 * (bmin[:2] + bmax[:2])
  v_all[:, :2] -= cxy[None, :]
  v_all[:, 2] -= bmin[2]
  v_all *= scl

  if add_floor:
    fl_v = np.array([[-3.2, -3.2, -0.01], [3.2, -3.2, -0.01], [3.2, 3.2, -0.01], [-3.2, 3.2, -0.01]], dtype=np.float32)
    fl_f = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32) + v_all.shape[0]
    fl_n = np.array([[0, 0, 1]] * 4, dtype=np.float32)
    fl_uv = np.array([[[0, 0], [1, 0], [1, 1]], [[0, 0], [1, 1], [0, 1]]], dtype=np.float32)
    f_mid = len(mat_records)
    mat_tex_ids.append(-1)
    mat_base_rgba.append([0.55, 0.54, 0.52, 0.35])
    mat_codes.append(0.05)
    mat_records.append({"name": "studio_floor", "tex_id": -1, "metallic": 0.05, "roughness": 0.35})
    v_all = np.concatenate([v_all, fl_v], axis=0)
    f_all = np.concatenate([f_all, fl_f], axis=0)
    n_all = np.concatenate([n_all, fl_n], axis=0)
    uv_all = np.concatenate([uv_all, fl_uv], axis=0)
    mid_all = np.concatenate([mid_all, np.full((2,), f_mid, dtype=np.int32)], axis=0)
    alb_all = np.concatenate([alb_all, np.full((2, 3), 0.55, dtype=np.float32)], axis=0)
    rgh_all = np.concatenate([rgh_all, np.full((2,), 0.35, dtype=np.float32)], axis=0)
    mc_all = np.concatenate([mc_all, np.full((2,), 0.05, dtype=np.float32)], axis=0)

  cp = np.asarray(cam_pos, dtype=np.float32)
  ct = np.asarray(cam_tgt, dtype=np.float32)
  fwd = ct - cp
  fwd /= max(float(np.linalg.norm(fwd)), 1e-12)
  right = np.cross(fwd, np.array([0.0, 0.0, 1.0], dtype=np.float32))
  right /= max(float(np.linalg.norm(right)), 1e-12)
  up = np.cross(right, fwd)
  cmat = np.stack([right, up, -fwd], axis=1).astype(np.float32)

  tex_atlas = np.stack(tex_alb_rgh, axis=0) if tex_alb_rgh else np.zeros((1, tex_res, tex_res, 4), np.float32)
  nrm_met_atlas = np.stack(tex_nrm_met, axis=0) if tex_nrm_met else np.broadcast_to(np.array([0.5, 0.5, 1.0, 0.0], np.float32), (1, tex_res, tex_res, 4)).copy()

  lights = [
      {"name": "KeySun", "type": "SUN", "pos": [4.0, -4.0, 8.0], "dir": [-0.35, 0.35, -0.87], "color": [1.0, 0.94, 0.85], "energy": 3.5},
      {"name": "FillArea", "type": "AREA", "pos": [-3.0, -2.5, 4.5], "dir": [0.4, 0.3, -0.85], "color": [0.75, 0.86, 1.0], "energy": 120.0},
  ]

  pack = ScenePack(
      scene_name=scene_name,
      verts=v_all, faces=f_all, vnorm=n_all, tri_uvs=uv_all,
      tri_mat_id=mid_all, tri_albedo=alb_all, tri_roughness=rgh_all, tri_mat_code=mc_all,
      tex_atlas=tex_atlas,
      mat_tex_ids=np.asarray(mat_tex_ids, dtype=np.int32),
      mat_base_rgba=np.asarray(mat_base_rgba, dtype=np.float32),
      mat_codes=np.asarray(mat_codes, dtype=np.float32),
      cam_pos=cp, cam_xmat=cmat, cam_fovy=float(cam_fovy), cam_shift=(0.0, 0.0),
      znear=0.05, zfar=100.0, native_res=(1024, 1024, 100),
      lights=lights,
      sky_zenith=(0.38, 0.56, 0.85), sky_horizon=(0.72, 0.80, 0.90), sky_ground=(0.18, 0.17, 0.16),
      materials=mat_records,
  )
  return pack, nrm_met_atlas


def _perturb_normal_cotangent(
    pos_hw: jax.Array,     # (B, H, W, 3)
    nrm_hw: jax.Array,     # (B, H, W, 3)
    uv_hw: jax.Array,      # (B, H, W, 2)
    nrm_map_hw: jax.Array, # (B, H, W, 3) in [0, 1]
    hit_hw: jax.Array,     # (B, H, W)
) -> jax.Array:
  """Screen-space derivative cotangent frame normal perturbation (Schuler 2013)."""
  dp_dx = jnp.pad(pos_hw[:, :, 1:, :] - pos_hw[:, :, :-1, :], ((0, 0), (0, 0), (0, 1), (0, 0)), mode="edge")
  dp_dy = jnp.pad(pos_hw[:, 1:, :, :] - pos_hw[:, :-1, :, :], ((0, 0), (0, 1), (0, 0), (0, 0)), mode="edge")
  duv_dx = jnp.pad(uv_hw[:, :, 1:, :] - uv_hw[:, :, :-1, :], ((0, 0), (0, 0), (0, 1), (0, 0)), mode="edge")
  duv_dy = jnp.pad(uv_hw[:, 1:, :, :] - uv_hw[:, :-1, :, :], ((0, 0), (0, 1), (0, 0), (0, 0)), mode="edge")

  dp2_perp = jnp.cross(dp_dy, nrm_hw)
  dp1_perp = jnp.cross(nrm_hw, dp_dx)
  T = dp2_perp * duv_dx[..., 0:1] + dp1_perp * duv_dy[..., 0:1]
  B = dp2_perp * duv_dx[..., 1:2] + dp1_perp * duv_dy[..., 1:2]
  inv_max = jax.lax.rsqrt(jnp.maximum(jnp.maximum(jnp.sum(T * T, axis=-1, keepdims=True), jnp.sum(B * B, axis=-1, keepdims=True)), 1e-8))
  valid_tb = jnp.sum(T * T, axis=-1, keepdims=True) > 1e-7
  T_n = T * inv_max
  B_n = B * inv_max

  # Tangent space normal in [-1, 1]
  nm = nrm_map_hw * 2.0 - 1.0
  n_pert = T_n * nm[..., 0:1] + B_n * nm[..., 1:2] + nrm_hw * nm[..., 2:3]
  n_pert = n_pert / jnp.maximum(jnp.linalg.norm(n_pert, axis=-1, keepdims=True), 1e-8)
  n_out = jnp.where(hit_hw[..., None] & valid_tb, n_pert, nrm_hw)
  return n_out


def render_khronos_pbr_ao_ggx(
    spec: WideBVH64UVSceneSpec,
    dev_arrays: Tuple[jax.Array, ...],
    nrm_met_atlas_d: jax.Array,
    cam_xpos: jax.Array,
    cam_xmat: jax.Array,
    ao_samples: int = 8,
    ao_radius: float = 0.8,
    exposure: float = 1.25,
    interpret: bool = False,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Render full UV Albedo + Normal-Mapped + Roughness/Metallic GGX + R2 8-sample AO + AgX."""
  l2_roots_d, l2_d, l0_d, leaf_d, tex_d, mtid_d, mcode_d = dev_arrays
  H, W = int(spec.height), int(spec.width)
  ro, rd, neg_z = compute_blend_camera_rays(spec, cam_xpos, cam_xmat)
  B, R, _ = ro.shape
  max_dist = jnp.broadcast_to(((float(spec.zfar) - float(spec.znear)) / neg_z)[None, :], (B, R))

  d_hit, nrm, uv, mid, fb_alb, fb_rgh = pallas_wide_bvh64_uv_raycast(
      l2_roots_d, l2_d, l0_d, leaf_d, ro, rd, max_dist, H, W, any_hit=False, interpret=interpret
  )

  hit = mid >= 0
  best_d = jnp.where(hit, d_hit, -1.0)
  vdir = -rd
  nrm = jnp.where(hit[..., None], nrm, 0.0)
  flip = jnp.sum(nrm * vdir, axis=-1, keepdims=True) < 0.0
  nrm = jnp.where(flip, -nrm, nrm)

  safe_mid = jnp.clip(mid, 0, len(spec.mat_tex_ids) - 1)
  tex_id = jnp.where(hit, mtid_d[safe_mid], -1)
  base_met = jnp.where(hit, jnp.clip(mcode_d[safe_mid], 0.0, 1.0), 0.0)
  alb_lin, rgh = _sample_tex_atlas_bilinear(tex_d, tex_id, uv, fb_alb, fb_rgh)
  nrm_map, met_map = _sample_tex_atlas_bilinear(
      nrm_met_atlas_d, tex_id, uv,
      jnp.broadcast_to(jnp.array([0.5, 0.5, 1.0], jnp.float32), alb_lin.shape),
      base_met,
  )
  metallic = jnp.where(tex_id >= 0, met_map, base_met)
  hit_pos = ro + jnp.where(hit, best_d, 0.0)[..., None] * rd

  # Apply tangent-space normal map on 2D grid
  nrm_pert = _perturb_normal_cotangent(
      hit_pos.reshape(B, H, W, 3),
      nrm.reshape(B, H, W, 3),
      uv.reshape(B, H, W, 2),
      nrm_map.reshape(B, H, W, 3),
      hit.reshape(B, H, W),
  ).reshape(B, R, 3)

  # R2-stratified 8-sample Hemisphere AO via fast leaf_rows=10 any_hit BVH kernel
  px = jnp.arange(W, dtype=jnp.float32)
  py = jnp.arange(H, dtype=jnp.float32)
  gx, gy = jnp.meshgrid(px, py, indexing="xy")
  r2_phi0 = (2.0 * math.pi) * jnp.mod(gx * 0.75487766 + gy * 0.56984029, 1.0).reshape(1, R)

  # Duff ONB around nrm_pert
  nz = nrm_pert[..., 2]
  sz = jnp.where(nz >= 0.0, 1.0, -1.0)
  aa = -1.0 / (sz + nz + 1e-6)
  bb = nrm_pert[..., 0] * nrm_pert[..., 1] * aa
  t1 = jnp.stack([1.0 + sz * nrm_pert[..., 0] ** 2 * aa, sz * bb, -sz * nrm_pert[..., 0]], axis=-1)
  t2 = jnp.stack([bb, sz + nrm_pert[..., 1] ** 2 * aa, -nrm_pert[..., 1]], axis=-1)

  sh_o = hit_pos + nrm_pert * 3.0e-3
  ao_acc = jnp.zeros((B, R), dtype=jnp.float32)
  for k in range(ao_samples):
    u1 = (k + 0.5) / float(ao_samples)
    phi = r2_phi0 + k * (2.39996323)  # golden angle
    r_disk = math.sqrt(u1)
    z_loc = math.sqrt(max(1.0 - u1, 0.0))
    d_ao = t1 * (r_disk * jnp.cos(phi))[..., None] + t2 * (r_disk * jnp.sin(phi))[..., None] + nrm_pert * z_loc
    d_ao = d_ao / jnp.maximum(jnp.linalg.norm(d_ao, axis=-1, keepdims=True), 1e-12)
    ao_lim = jnp.where(hit, jnp.float32(ao_radius), 0.0)
    d_sh = pallas_wide_bvh64_uv_raycast(
        l2_roots_d, l2_d, l0_d, leaf_d, sh_o, d_ao, ao_lim, H, W, any_hit=True, interpret=interpret
    )
    ao_acc = ao_acc + jnp.where(d_sh >= ao_lim - 1e-4, 1.0, 0.0)
  ao = ao_acc / float(ao_samples)

  # Primary key sun shadow
  l_dir0 = jnp.asarray(spec.light_dir[0], dtype=jnp.float32)
  L0 = jnp.broadcast_to(-l_dir0[None, None, :], ro.shape)
  ndotl0 = jnp.maximum(0.0, jnp.sum(nrm_pert * L0, axis=-1))
  sh_lim0 = jnp.where(hit & (ndotl0 > 0.0), jnp.float32(spec.zfar * 0.9), 0.0)
  d_sh0 = pallas_wide_bvh64_uv_raycast(
      l2_roots_d, l2_d, l0_d, leaf_d, sh_o, L0, sh_lim0, H, W, any_hit=True, interpret=interpret
  )
  vis0 = jnp.where(d_sh0 < sh_lim0 - 1e-4, 0.0, 1.0)

  # Cook-Torrance GGX + Hemisphere Sky IBL + Specular Reflection Approximation
  up_w = 0.5 * (nrm_pert[..., 2] + 1.0)
  sky_z = jnp.asarray(spec.sky_zenith, dtype=jnp.float32)
  sky_h = jnp.asarray(spec.sky_horizon, dtype=jnp.float32)
  sky_g = jnp.asarray(spec.sky_ground, dtype=jnp.float32)
  amb_env = up_w[..., None] * sky_h + (1.0 - up_w[..., None]) * sky_g

  ndotv = jnp.maximum(jnp.sum(nrm_pert * vdir, axis=-1), 1e-4)
  f0 = 0.04 * (1.0 - metallic[..., None]) + alb_lin * metallic[..., None]
  F_v = f0 + (1.0 - f0) * ((1.0 - ndotv) ** 5)[..., None]

  refl_d = 2.0 * jnp.sum(nrm_pert * vdir, axis=-1, keepdims=True) * nrm_pert - vdir
  r_up = jnp.clip(0.5 * (refl_d[..., 2] + 1.0), 0.0, 1.0)[..., None]
  refl_env = r_up * sky_z + (1.0 - r_up) * sky_g
  spec_ibl = refl_env * F_v * (1.0 - 0.7 * rgh)[..., None] * (0.35 + 0.65 * ao)[..., None]
  diff_ibl = (1.0 - metallic[..., None]) * alb_lin * amb_env * (0.45 * ao[..., None])
  rad = diff_ibl + spec_ibl

  alpha = jnp.maximum(rgh * rgh, 0.025)
  a2 = alpha * alpha
  for li in range(len(spec.light_type)):
    lp = jnp.asarray(spec.light_pos[li], dtype=jnp.float32)
    ld = jnp.asarray(spec.light_dir[li], dtype=jnp.float32)
    lc = jnp.asarray(spec.light_col[li], dtype=jnp.float32)
    if int(spec.light_type[li]) == 0:
      Li = jnp.broadcast_to(-ld[None, None, :], ro.shape)
      atten = 1.0
    else:
      dv = lp[None, None, :] - hit_pos
      d2 = jnp.sum(dv * dv, axis=-1)
      Li = dv / jnp.sqrt(jnp.maximum(d2, 1e-8))[..., None]
      atten = 1.0 / (1.0 + 0.04 * d2)
    ndotl = jnp.maximum(0.0, jnp.sum(nrm_pert * Li, axis=-1))
    vis_i = vis0 if li == 0 else (0.5 + 0.5 * ao)
    H_v = (Li + vdir) / jnp.maximum(jnp.linalg.norm(Li + vdir, axis=-1, keepdims=True), 1e-12)
    ndoth = jnp.maximum(jnp.sum(nrm_pert * H_v, axis=-1), 0.0)
    vdoth = jnp.maximum(jnp.sum(vdir * H_v, axis=-1), 0.0)
    denom = ndoth * ndoth * (a2 - 1.0) + 1.0
    D = a2 / jnp.maximum(math.pi * denom * denom, 1e-5)
    F = f0 + (1.0 - f0) * ((1.0 - vdoth) ** 5)[..., None]
    spec_l = jnp.minimum((D * 0.25)[..., None] * F, 3.0)
    diff_l = (1.0 - metallic[..., None]) * alb_lin / math.pi
    rad = rad + (diff_l + spec_l) * lc[None, None, :] * (ndotl * atten * vis_i)[..., None]

  srgb_pbr = _agx_tonemap_jnp(rad, exposure=exposure)
  sz_r = rd[..., 2]
  tu = jnp.clip(sz_r, 0.0, 1.0)[..., None]
  td = jnp.clip(-sz_r, 0.0, 1.0)[..., None]
  sky_lin = jnp.where(sz_r[..., None] >= 0.0, (1.0 - tu) * sky_h + tu * sky_z, (1.0 - td) * sky_h + td * sky_g)
  sky_srgb = _agx_tonemap_jnp(sky_lin, exposure=1.0)

  rgb = jnp.where(hit[..., None], srgb_pbr, sky_srgb).reshape(B, H, W, 3)
  alb_out = jnp.clip(_linear_to_srgb_jnp(alb_lin), 0.0, 1.0).reshape(B, H, W, 3)
  nrm_out = jnp.where(hit[..., None], 0.5 + 0.5 * nrm_pert, 0.0).reshape(B, H, W, 3)
  ao_out = jnp.where(hit, ao, 1.0).reshape(B, H, W)
  return rgb, alb_out, nrm_out, ao_out


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--assets_dir", default="/dev/shm/zaychik-tpu-mjx/assets")
  ap.add_argument("--out_json", default="/dev/shm/zaychik-tpu-mjx-w5/v3_w5_complex_gltf_gallery.json")
  ap.add_argument("--out_png", default="/dev/shm/zaychik-tpu-mjx-w5/v3_w5_complex_gltf_gallery.png")
  args = ap.parse_args()

  specs = [
      ("Sponza", os.path.join(args.assets_dir, "Sponza/glTF/Sponza.gltf"), (-6.2, 0.0, 2.2), (4.0, 0.0, 2.6), 52.0, False, 1.6),
      ("ABeautifulGame", os.path.join(args.assets_dir, "ABeautifulGame/glTF/ABeautifulGame.gltf"), (1.35, -1.45, 0.95), (0.0, 0.0, 0.22), 42.0, True, 0.45),
      ("FlightHelmet", os.path.join(args.assets_dir, "FlightHelmet/glTF/FlightHelmet.gltf"), (1.15, -1.25, 0.95), (0.0, 0.0, 0.65), 42.0, True, 0.45),
  ]

  devs_all = jax.devices()
  results = {"experiment": "v3_w5_complex_gltf_gallery", "device_kind": devs_all[0].device_kind, "n_devices": len(devs_all), "scenes": []}
  panels = []

  for name, path, cpos, ctgt, fovy, add_fl, ao_r in specs:
    if not os.path.exists(path):
      print(f"SKIP {name}: missing {path}", flush=True)
      continue
    pack, nrm_met_atlas = load_gltf_to_scenepack(path, name, cpos, ctgt, cam_fovy=fovy, tex_res=256, add_floor=add_fl)
    benches = []
    hero_pack = None

    for res in [512, 1024]:
      spec_r = build_wide_bvh64_uv(pack, height=res, width=res, use_shadows=True)
      for nc in ([1, 4] if len(devs_all) >= 4 else [1]):
        bs = 4 if nc == 4 else 1
        cxp_np = np.repeat(pack.cam_pos[None, :], bs, axis=0)
        cxm_np = np.repeat(pack.cam_xmat[None, :, :], bs, axis=0)
        if nc == 1:
          da = pack_dev_arrays(spec_r, devs_all[0])
          nm_d = jax.device_put(nrm_met_atlas, devs_all[0])
          cp_d = jax.device_put(cxp_np, devs_all[0])
          cm_d = jax.device_put(cxm_np, devs_all[0])

          @jax.jit
          def _run1(da_i, nm_i, cp_i, cm_i):
            return render_khronos_pbr_ao_ggx(spec_r, da_i, nm_i, cp_i, cm_i, ao_samples=8, ao_radius=ao_r, exposure=1.35)

          hlo = _hlo_audit(_run1.lower(da, nm_d, cp_d, cm_d).as_text())
          assert hlo["host_callbacks"] == 0 and hlo["send_recv_ops"] == 0, hlo
          for _ in range(2):
            out = _run1(da, nm_d, cp_d, cm_d)
          jax.block_until_ready(out)
          t0 = time.perf_counter()
          for _ in range(4):
            out = _run1(da, nm_d, cp_d, cm_d)
          jax.block_until_ready(out)
          dt = (time.perf_counter() - t0) / 4.0
        else:
          mesh = jax.sharding.Mesh(np.array(devs_all[:4]), ("dp",))
          P = jax.sharding.PartitionSpec
          sh_dp = jax.sharding.NamedSharding(mesh, P("dp"))
          rep = jax.sharding.NamedSharding(mesh, P())
          da = pack_dev_arrays(spec_r, rep)
          nm_d = jax.device_put(nrm_met_atlas, rep)
          cp_d = jax.device_put(cxp_np, sh_dp)
          cm_d = jax.device_put(cxm_np, sh_dp)
          sm_kw = {"check_rep": False} if "check_rep" in inspect.signature(shard_map).parameters else {"check_vma": False}

          def _loc(da_i, nm_i, cp_i, cm_i):
            return render_khronos_pbr_ao_ggx(spec_r, da_i, nm_i, cp_i, cm_i, ao_samples=8, ao_radius=ao_r, exposure=1.35)

          _sm = shard_map(_loc, mesh=mesh, in_specs=(P(), P(), P("dp"), P("dp")), out_specs=(P("dp"), P("dp"), P("dp"), P("dp")), **sm_kw)
          _run4 = jax.jit(lambda da_i, nm_i, cp_i, cm_i: _sm(da_i, nm_i, cp_i, cm_i))
          hlo = _hlo_audit(_run4.lower(da, nm_d, cp_d, cm_d).as_text())
          assert hlo["host_callbacks"] == 0 and hlo["send_recv_ops"] == 0, hlo
          for _ in range(2):
            out = _run4(da, nm_d, cp_d, cm_d)
          jax.block_until_ready(out)
          t0 = time.perf_counter()
          for _ in range(4):
            out = _run4(da, nm_d, cp_d, cm_d)
          jax.block_until_ready(out)
          dt = (time.perf_counter() - t0) / 4.0

        rec = {
            "resolution": f"{res}x{res}",
            "n_chips": nc,
            "batch_size": bs,
            "dt_ms": round(dt * 1e3, 3),
            "img_per_sec": round(bs / max(dt, 1e-9), 3),
            "host_callbacks": hlo["host_callbacks"],
            "send_recv_ops": hlo["send_recv_ops"],
        }
        print(f"[{name}]", rec, flush=True)
        benches.append(rec)
        if res == 1024 and nc == (4 if len(devs_all) >= 4 else 1):
          hero_pack = tuple(np.asarray(x[0], np.float32) for x in out)

    results["scenes"].append({
        "scene_meta": pack.summary(),
        "benchmarks": benches,
    })
    if hero_pack is not None:
      panels.append((name, pack.n_tris, benches[-1], *hero_pack))

  if panels:
    fig, axes = plt.subplots(len(panels), 4, figsize=(17.0, 4.2 * len(panels)), dpi=160)
    if len(panels) == 1:
      axes = axes[None, :]
    for r_i, (nm, tris, bn, rgb, alb, nrm, ao) in enumerate(panels):
      items = [
          (alb, f"{nm} ({tris:,} tris)\nUV Albedo (sRGB)", None),
          (nrm, "Screen-Space Cotangent\nNormal Map (World Space)", None),
          (ao, "R2 8-Sample Hemisphere\nAmbient Occlusion", "gray"),
          (rgb, f"TPU GGX + AO + AgX (1024²)\n{bn['img_per_sec']:.2f} img/s ({bn['n_chips']}c) | 0 callbacks", None),
      ]
      for c_i, (im, tit, cmap) in enumerate(items):
        axes[r_i, c_i].imshow(np.clip(im, 0, 1), cmap=cmap)
        axes[r_i, c_i].set_title(tit, fontsize=9.5, fontweight="bold")
        axes[r_i, c_i].axis("off")
    plt.suptitle("Khronos glTF 2.0 PBR Showcase on Live TPU v5p (UV Albedo + Normal Map + Metallic/Roughness GGX + 8x AO + AgX)", fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out_png)), exist_ok=True)
    plt.savefig(args.out_png, bbox_inches="tight")
    plt.close(fig)
    print("SAVED PNG:", args.out_png, flush=True)

  os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
  with open(args.out_json, "w") as f:
    json.dump(results, f, indent=2)
  print("SAVED JSON:", args.out_json, flush=True)


if __name__ == "__main__":
  main()
