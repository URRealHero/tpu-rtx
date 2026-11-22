"""V3.2 100M+ Unique Non-Instanced & 100M-1B+ TLAS+BLAS Instanced + Native 4K UHD Suite."""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import math
import os
import re
import time
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np
from PIL import Image

from tpu_mjx_render.mesh_ray import build_mesh_scene_spec, render_mesh_batch
from tpu_mjx_render.roofline import inspect_hlo_ops
from wide_bvh_v3 import (
    WideBVHMeshItem,
    WideBVHSceneSpec,
    _compute_smooth_vertex_normals,
    build_wide_bvh_mesh_item,
    build_wide_bvh_scene_spec,
    make_sharded_instanced_renderer,
    make_sharded_wide_bvh_renderer,
    render_wide_bvh_batch,
    render_wide_bvh_instanced_batch,
    to_device_arrays,
    to_replicated_device_arrays,
)


def subdivide_mesh_barycentric(verts: np.ndarray, faces: np.ndarray, k: int):
  if k <= 1:
    return verts.astype(np.float32), faces.astype(np.int32), _compute_smooth_vertex_normals(verts, faces)
  vnorm = _compute_smooth_vertex_normals(verts, faces)
  pts_uv = []
  idx_map = {}
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
  new_verts = (w * verts[faces[:, 0]][:, None] + u * verts[faces[:, 1]][:, None] + v * verts[faces[:, 2]][:, None]).reshape(F * P_cnt, 3).astype(np.float32)
  new_norms = (w * vnorm[faces[:, 0]][:, None] + u * vnorm[faces[:, 1]][:, None] + v * vnorm[faces[:, 2]][:, None]).reshape(F * P_cnt, 3)
  new_norms = (new_norms / np.maximum(np.linalg.norm(new_norms, axis=-1, keepdims=True), 1e-12)).astype(np.float32)
  new_faces = (sub_faces[None] + (np.arange(F, dtype=np.int32) * P_cnt)[:, None, None]).reshape(F * k * k, 3)
  return new_verts, new_faces, new_norms


def load_base_scene():
  ds = np.load("/dev/shm/zaychik-tpu-mjx/out/assets/blender_real_scene.npz")
  gt = np.load("/dev/shm/zaychik-tpu-mjx/out/assets/blender_real_scene_gt.npz")
  M = int(ds["n_meshes"])
  base_meshes = [
      {
          "name": str(ds[f"mesh_{m}_name"]),
          "verts": ds[f"mesh_{m}_verts"],
          "faces": ds[f"mesh_{m}_faces"],
          "rgba": ds[f"mesh_{m}_rgba"],
      }
      for m in range(M)
  ]
  return ds, gt, base_meshes


def _strip_host_arrays(item: WideBVHMeshItem) -> WideBVHMeshItem:
  """Drop large NumPy host arrays once uploaded to TPU HBM to keep host RAM lean."""
  empty = np.empty((0,), dtype=np.float32)
  return dataclasses.replace(item, l2_roots_aabb=empty, l2_aabbs=empty, l0_aabbs=empty, leaf_bank=empty)


def build_streaming_3mesh_to_hbm(base_meshes, k: int, kw: dict, mesh: Optional[Mesh] = None):
  """Subdivide & pack 3 meshes one by one, uploading directly to TPU HBM."""
  items_meta = []
  dev_arrays = []
  t_sub_tot, t_bvh_tot, hbm_bytes = 0.0, 0.0, 0
  rep = NamedSharding(mesh, P()) if mesh is not None else None

  for mid, m in enumerate(base_meshes):
    t0 = time.perf_counter()
    nv, nf, nn = subdivide_mesh_barycentric(m["verts"], m["faces"], k)
    t_sub_tot += time.perf_counter() - t0
    t1 = time.perf_counter()
    it = build_wide_bvh_mesh_item(mid, {"name": m["name"], "verts": nv, "faces": nf, "vert_normals": nn, "rgba": m["rgba"]}, build_method="spatial_l1_morton")
    t_bvh_tot += time.perf_counter() - t1
    del nv, nf, nn
    hbm_bytes += it.l2_roots_aabb.nbytes + it.l2_aabbs.nbytes + it.l0_aabbs.nbytes + it.leaf_bank.nbytes
    if rep is not None:
      dev_arrays.append((
          jax.device_put(it.l2_roots_aabb, rep),
          jax.device_put(it.l2_aabbs, rep),
          jax.device_put(it.l0_aabbs, rep),
          jax.device_put(it.leaf_bank, rep),
      ))
    else:
      dev_arrays.append((
          jnp.asarray(it.l2_roots_aabb, dtype=jnp.float32),
          jnp.asarray(it.l2_aabbs, dtype=jnp.float32),
          jnp.asarray(it.l0_aabbs, dtype=jnp.float32),
          jnp.asarray(it.leaf_bank, dtype=jnp.float32),
      ))
    items_meta.append(_strip_host_arrays(it))
    del it
    gc.collect()

  spec = WideBVHSceneSpec(
      height=64, width=64, cam_projection=0, cam_sensorsize=(0.0, 0.0), cam_intrinsic=(0.0, 0.0, 0.0, 0.0),
      meshes=tuple(items_meta),
      light_pos=np.asarray(kw["light_pos"], dtype=np.float32),
      light_ambient=(0.28, 0.30, 0.34), light_diffuse=(0.85, 0.83, 0.80), light_specular=(0.30, 0.30, 0.30),
      sky_rgb1=(0.28, 0.36, 0.48), sky_rgb2=(0.10, 0.14, 0.20),
      cam_fovy=kw["cam_fovy"], znear=kw["znear"], zfar=kw["zfar"], use_shadows=True,
  )
  return spec, tuple(dev_arrays), t_sub_tot, t_bvh_tot, hbm_bytes


def build_streaming_100m_unique_to_hbm(base_meshes, kw: dict, mesh: Optional[Mesh] = None):
  """Build 108.23M UNIQUE non-instanced tris (8 uniquely displaced Suzannes + Torus + Pedestal, k=58)."""
  k = 58
  items_meta = []
  dev_arrays = []
  t_sub_tot, t_bvh_tot, hbm_bytes = 0.0, 0.0, 0
  rep = NamedSharding(mesh, P()) if mesh is not None else None

  # 8 unique sculpted Suzanne variants with distinct 3D multi-octave displacement & albedo
  suz = base_meshes[0]
  palette = [
      [0.86, 0.48, 0.28, 1.0],
      [0.24, 0.72, 0.76, 1.0],
      [0.88, 0.74, 0.28, 1.0],
      [0.78, 0.34, 0.46, 1.0],
      [0.36, 0.54, 0.88, 1.0],
      [0.42, 0.78, 0.46, 1.0],
      [0.76, 0.78, 0.82, 1.0],
      [0.68, 0.46, 0.84, 1.0],
  ]

  t0 = time.perf_counter()
  base_nv, base_nf, base_nn = subdivide_mesh_barycentric(suz["verts"], suz["faces"], k)
  t_sub_base = time.perf_counter() - t0
  t_sub_tot += t_sub_base

  mid = 0
  for s_i in range(8):
    t_d0 = time.perf_counter()
    fx = 18.0 + 4.5 * s_i
    fy = 21.0 + 3.7 * s_i
    fz = 25.0 + 5.1 * s_i
    ph = 0.75 * s_i
    disp = 0.007 * (
        np.sin(fx * base_nv[:, 0] + ph) * np.cos(fy * base_nv[:, 1] - ph)
        + 0.45 * np.sin(fz * base_nv[:, 2] + 1.3 * ph)
    )
    nv_u = (base_nv + disp[:, None] * base_nn).astype(np.float32)
    nn_u = base_nn
    t_sub_tot += time.perf_counter() - t_d0

    t1 = time.perf_counter()
    it = build_wide_bvh_mesh_item(
        mid,
        {"name": f"suzanne_unique_{s_i}", "verts": nv_u, "faces": base_nf, "vert_normals": nn_u, "rgba": palette[s_i]},
        build_method="spatial_l1_morton",
    )
    t_bvh_tot += time.perf_counter() - t1
    del nv_u, nn_u, disp
    hbm_bytes += it.l2_roots_aabb.nbytes + it.l2_aabbs.nbytes + it.l0_aabbs.nbytes + it.leaf_bank.nbytes
    if rep is not None:
      dev_arrays.append((
          jax.device_put(it.l2_roots_aabb, rep),
          jax.device_put(it.l2_aabbs, rep),
          jax.device_put(it.l0_aabbs, rep),
          jax.device_put(it.leaf_bank, rep),
      ))
    else:
      dev_arrays.append((
          jnp.asarray(it.l2_roots_aabb, dtype=jnp.float32),
          jnp.asarray(it.l2_aabbs, dtype=jnp.float32),
          jnp.asarray(it.l0_aabbs, dtype=jnp.float32),
          jnp.asarray(it.leaf_bank, dtype=jnp.float32),
      ))
    items_meta.append(_strip_host_arrays(it))
    del it
    gc.collect()
    mid += 1

  del base_nv, base_nf, base_nn
  gc.collect()

  # Plus high-poly Torus & Pedestal (k=58)
  for m_raw in base_meshes[1:]:
    t0 = time.perf_counter()
    nv, nf, nn = subdivide_mesh_barycentric(m_raw["verts"], m_raw["faces"], k)
    t_sub_tot += time.perf_counter() - t0
    t1 = time.perf_counter()
    it = build_wide_bvh_mesh_item(
        mid,
        {"name": f"{m_raw['name']}_unique", "verts": nv, "faces": nf, "vert_normals": nn, "rgba": m_raw["rgba"]},
        build_method="spatial_l1_morton",
    )
    t_bvh_tot += time.perf_counter() - t1
    del nv, nf, nn
    hbm_bytes += it.l2_roots_aabb.nbytes + it.l2_aabbs.nbytes + it.l0_aabbs.nbytes + it.leaf_bank.nbytes
    if rep is not None:
      dev_arrays.append((
          jax.device_put(it.l2_roots_aabb, rep),
          jax.device_put(it.l2_aabbs, rep),
          jax.device_put(it.l0_aabbs, rep),
          jax.device_put(it.leaf_bank, rep),
      ))
    else:
      dev_arrays.append((
          jnp.asarray(it.l2_roots_aabb, dtype=jnp.float32),
          jnp.asarray(it.l2_aabbs, dtype=jnp.float32),
          jnp.asarray(it.l0_aabbs, dtype=jnp.float32),
          jnp.asarray(it.leaf_bank, dtype=jnp.float32),
      ))
    items_meta.append(_strip_host_arrays(it))
    del it
    gc.collect()
    mid += 1

  spec = WideBVHSceneSpec(
      height=64, width=64, cam_projection=0, cam_sensorsize=(0.0, 0.0), cam_intrinsic=(0.0, 0.0, 0.0, 0.0),
      meshes=tuple(items_meta),
      light_pos=np.asarray(kw["light_pos"], dtype=np.float32),
      light_ambient=(0.28, 0.30, 0.34), light_diffuse=(0.85, 0.83, 0.80), light_specular=(0.30, 0.30, 0.30),
      sky_rgb1=(0.28, 0.36, 0.48), sky_rgb2=(0.10, 0.14, 0.20),
      cam_fovy=kw["cam_fovy"], znear=kw["znear"], zfar=kw["zfar"], use_shadows=True,
  )
  return spec, tuple(dev_arrays), t_sub_tot, t_bvh_tot, hbm_bytes


def make_100m_gallery_poses(ds, B: int):
  """Generate dynamic rigid poses (B, 10, 3) & (B, 10, 3, 3) for the 10-mesh 108.23M scene."""
  # Base positions in 2 rows visible from cam_xpos=[0.82, -0.58, 0.62]
  suz_pos = np.array([
      [-0.45,  0.32, 0.38],
      [-0.20,  0.38, 0.38],
      [ 0.08,  0.35, 0.38],
      [ 0.32,  0.22, 0.38],
      [-0.38, -0.02, 0.22],
      [-0.14,  0.04, 0.23],
      [ 0.10,  0.08, 0.22],
      [ 0.30, -0.18, 0.22],
  ], dtype=np.float32)
  tor_pos = np.array([0.19, -0.04, 0.19], dtype=np.float32)
  ped_pos = np.array([0.00,  0.00, 0.00], dtype=np.float32)
  base_p = np.vstack([suz_pos, tor_pos[None], ped_pos[None]])  # (10, 3)

  oxp = np.broadcast_to(base_p[None, :, :], (B, 10, 3)).copy()
  oxm = np.broadcast_to(np.eye(3, dtype=np.float32)[None, None, :, :], (B, 10, 3, 3)).copy()
  for b in range(B):
    for m in range(9):  # animate sculptures + torus dynamically per world
      ang = 0.05 * b + 0.35 * m
      c, s = math.cos(ang), math.sin(ang)
      oxm[b, m] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
      oxp[b, m, 2] += 0.01 * math.sin(0.2 * b + m)

  def _tile(arr):
    rep = (B + arr.shape[0] - 1) // arr.shape[0]
    return np.tile(arr, (rep,) + (1,) * (arr.ndim - 1))[:B]

  return jnp.asarray(oxp), jnp.asarray(oxm), jnp.asarray(_tile(ds["cam_xpos"])), jnp.asarray(_tile(ds["cam_xmat"]))


def make_instanced_poses(ds, B: int, K_inst: int):
  """Generate dynamic rigid poses for K_inst instances of the 3-mesh hero BLAS."""
  # Spread K_inst instances in a 3D frustum gallery visible from [0.82, -0.58, 0.62]
  rng = np.random.default_rng(42)
  offsets = np.zeros((K_inst, 3), dtype=np.float32)
  grid_n = int(math.ceil(math.sqrt(K_inst)))
  for k_i in range(K_inst):
    gx = (k_i % grid_n) / max(grid_n - 1, 1) - 0.5
    gy = (k_i // grid_n) / max(grid_n - 1, 1) - 0.5
    offsets[k_i] = np.array([gx * 1.15 - 0.15, gy * 1.15 + 0.10, 0.06 * (k_i % 3)], dtype=np.float32)

  def _tile(arr):
    rep = (B + arr.shape[0] - 1) // arr.shape[0]
    return np.tile(arr, (rep,) + (1,) * (arr.ndim - 1))[:B]

  base_oxp = _tile(ds["obj_xpos"])  # (B, 3, 3)
  base_oxm = _tile(ds["obj_xmat"])  # (B, 3, 3, 3)
  cxp = jnp.asarray(_tile(ds["cam_xpos"]))
  cxm = jnp.asarray(_tile(ds["cam_xmat"]))

  ixp_list, ixm_list = [], []
  for m in range(3):
    pos_m = base_oxp[:, m : m + 1, :] + offsets[None, :, :]  # (B, K_inst, 3)
    # Apply small per-instance Z rotation
    rot_m = np.broadcast_to(base_oxm[:, m : m + 1, :, :], (B, K_inst, 3, 3)).copy()
    for k_i in range(K_inst):
      ang = 0.4 * k_i
      c, s = math.cos(ang), math.sin(ang)
      rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
      rot_m[:, k_i] = rot_m[:, k_i] @ rz
    ixp_list.append(jnp.asarray(pos_m))
    ixm_list.append(jnp.asarray(rot_m))
  return tuple(ixp_list), tuple(ixm_list), cxp, cxm


def tile_poses(ds, B: int):
  def _tile(arr):
    rep = (B + arr.shape[0] - 1) // arr.shape[0]
    return jnp.asarray(np.tile(arr, (rep,) + (1,) * (arr.ndim - 1))[:B])
  return _tile(ds["obj_xpos"]), _tile(ds["obj_xmat"]), _tile(ds["cam_xpos"]), _tile(ds["cam_xmat"])


def spec_summary(spec: WideBVHSceneSpec, hbm_bytes: int, t_sub: float, t_bvh: float):
  return {
      "total_tris": int(sum(it.n_tris_orig for it in spec.meshes)),
      "n_meshes": len(spec.meshes),
      "total_leaves_128": int(sum(it.n_leaves for it in spec.meshes)),
      "total_l1_64": int(sum(it.n_l1 for it in spec.meshes)),
      "total_l2_64": int(sum(it.n_l2 for it in spec.meshes)),
      "bvh_hbm_mb": round(hbm_bytes / (1024 ** 2), 2),
      "bvh_hbm_gb": round(hbm_bytes / (1024 ** 3), 4),
      "subdivide_s": round(t_sub, 2),
      "bvh_build_s": round(t_bvh, 2),
  }


def _load_or_init_json(path: str):
  if os.path.exists(path):
    with open(path, "r") as f:
      return json.load(f)
  return {
      "backend": jax.default_backend(),
      "jax_version": jax.__version__,
      "device_kind": jax.devices()[0].device_kind,
      "hbm_gib_per_chip": 95.74,
      "parity": {},
      "track_2a_unique_1c": [],
      "track_2a_unique_4c": [],
      "track_2b_instanced_1c": [],
      "track_2b_instanced_4c": [],
  }


def _save_json(path: str, data: dict):
  with open(path, "w") as f:
    json.dump(data, f, indent=2)


def bench_one_unique(spec_r, dev_arr, oxp, oxm, cxp, cxm, n_chips: int, stats_enabled: bool = False):
  B = int(oxp.shape[0])
  H, W = spec_r.height, spec_r.width
  if n_chips == 1:
    fn = jax.jit(lambda da, a, b, c, d: render_wide_bvh_batch(spec_r, da, a, b, c, d, interpret=False).rgb)
    t_c0 = time.perf_counter()
    y = fn(dev_arr, oxp, oxm, cxp, cxm)
    jax.block_until_ready(y)
    compile_s = time.perf_counter() - t_c0
    jax.block_until_ready(fn(dev_arr, oxp, oxm, cxp, cxm))

    n_steps = 2 if (H >= 2160) else (5 if H >= 512 else 10)
    t0 = time.perf_counter()
    for _ in range(n_steps):
      y = fn(dev_arr, oxp, oxm, cxp, cxm)
    jax.block_until_ready(y)
    ms = (time.perf_counter() - t0) * 1e3 / n_steps
  else:
    fn_4c, _ = make_sharded_wide_bvh_renderer(spec_r, jax.devices(), interpret=False, dev_arrays=dev_arr)
    t_c0 = time.perf_counter()
    y = fn_4c(oxp, oxm, cxp, cxm).rgb
    jax.block_until_ready(y)
    compile_s = time.perf_counter() - t_c0
    jax.block_until_ready(fn_4c(oxp, oxm, cxp, cxm).rgb)

    n_steps = 2 if (H >= 2160) else (5 if H >= 512 else 10)
    t0 = time.perf_counter()
    for _ in range(n_steps):
      y = fn_4c(oxp, oxm, cxp, cxm).rgb
    jax.block_until_ready(y)
    ms = (time.perf_counter() - t0) * 1e3 / n_steps

  fps = B * 1e3 / ms
  mrays_per_s = (B * H * W * 2.0) / (ms * 1e-3) / 1e6
  res_dict = {
      "n_chips": n_chips,
      "res": f"{H}x{W}",
      "B": B,
      "compile_s": round(compile_s, 2),
      "ms": round(ms, 2),
      "fps": round(fps, 2),
      "mrays_per_s": round(mrays_per_s, 2),
  }

  if stats_enabled and n_chips == 1:
    # Measure hardware leaf visits on B=1 (or oxp[:1])
    fn_st = jax.jit(lambda da, a, b, c, d: render_wide_bvh_batch(spec_r, da, a, b, c, d, interpret=False, return_stats=True))
    _, lv = fn_st(dev_arr, oxp[:1], oxm[:1], cxp[:1], cxm[:1])
    jax.block_until_ready(lv)
    n_tiles = (H // 8) * (W // 16)
    avg_leaves_tile = float(lv[0]) / float(n_tiles)
    # 1 leaf = 128 tris tested across 128 primary + 128 shadow rays in tile -> 128 tris per ray-tile step
    avg_tris_per_ray = avg_leaves_tile * 128.0
    tot_tris = sum(it.n_tris_orig for it in spec_r.meshes)
    cull_factor = tot_tris / max(avg_tris_per_ray, 1.0)
    res_dict["avg_leaves_per_tile"] = round(avg_leaves_tile, 2)
    res_dict["avg_tris_tested_per_ray"] = round(avg_tris_per_ray, 1)
    res_dict["culling_factor_x"] = round(cull_factor, 1)
    res_dict["pct_tris_culled"] = round((1.0 - 1.0 / max(cull_factor, 1.0)) * 100.0, 4)

  return res_dict


def run_2a_1c_ladder(out_path: str):
  ds, gt, base_meshes = load_base_scene()
  data = _load_or_init_json(out_path)
  kw = dict(cam_fovy=float(ds["cam_fovy"]), znear=float(ds["znear"]), zfar=float(ds["zfar"]), light_pos=tuple(float(x) for x in ds["light_pos"]))

  # 1. Parity check on 4,620 tris (spatial_l1_morton vs jax_cluster + Cycles EXR)
  spec_1, dev_1, _, _, _ = build_streaming_3mesh_to_hbm(base_meshes, 1, kw)
  mspec_v2 = build_mesh_scene_spec(base_meshes, height=64, width=64, **kw)
  idx_eval = np.array([0] + list(range(len(ds["split"]) - 7, len(ds["split"]))), dtype=np.int32)
  oxp_e, oxm_e = jnp.asarray(ds["obj_xpos"][idx_eval]), jnp.asarray(ds["obj_xmat"][idx_eval])
  cxp_e, cxm_e = jnp.asarray(ds["cam_xpos"][idx_eval]), jnp.asarray(ds["cam_xmat"][idx_eval])
  fn_v2 = jax.jit(lambda p, m, cp, cm: render_mesh_batch(mspec_v2, p, m, cp, cm, mode="jax_cluster"))
  fn_v3 = jax.jit(lambda da, p, m, cp, cm: render_wide_bvh_batch(spec_1, da, p, m, cp, cm, interpret=False))
  gb2 = fn_v2(oxp_e, oxm_e, cxp_e, cxm_e)
  gb3 = fn_v3(dev_1, oxp_e, oxm_e, cxp_e, cxm_e)
  jax.block_until_ready(gb3.rgb)
  f0_cy_z = np.asarray(gt["frame0_z_depth"], np.float32)
  f0_v3_z = np.asarray(gb3.z_depth[0], np.float32)
  both_fg = (f0_cy_z < 20.0) & (f0_v3_z > 0.0)
  hlo_txt = fn_v3.lower(dev_1, oxp_e, oxm_e, cxp_e, cxm_e).as_text()
  hlo = inspect_hlo_ops(hlo_txt)
  hlo["send_recv_ops"] = len(re.findall(r"\bsend\(|\brecv\(", hlo_txt, re.I))
  assert hlo["host_callbacks"] == 0 and hlo["send_recv_ops"] == 0
  data["parity"]["spatial_l1_morton_4620"] = {
      "max_depth_diff_m": float(jnp.max(jnp.abs(gb2.depth - gb3.depth))),
      "max_rgb_diff": float(jnp.max(jnp.abs(gb2.rgb - gb3.rgb))),
      "geom_match": float(jnp.mean(gb2.geom_id == gb3.geom_id)),
      "cycles_z_mae_m": float(np.mean(np.abs(f0_cy_z[both_fg] - f0_v3_z[both_fg]))),
      "hlo": hlo,
  }
  assert data["parity"]["spatial_l1_morton_4620"]["max_depth_diff_m"] < 1e-4
  print("PARITY 4620 PASS:", data["parity"]["spatial_l1_morton_4620"], flush=True)
  _save_json(out_path, data)
  del spec_1, dev_1; gc.collect()

  # 2. Scaling Ladder: 4.6k -> 115k -> 1.18M -> 10.64M (plus native 4K at 10.64M)
  for k in [1, 5, 16, 48]:
    spec_b, dev_b, t_sub, t_bvh, hbm_b = build_streaming_3mesh_to_hbm(base_meshes, k, kw)
    st = spec_summary(spec_b, hbm_b, t_sub, t_bvh)
    print(f"BUILT k={k}: {st}", flush=True)
    cfgs = [(64, 64, [64, 256]), (256, 256, [16, 64]), (512, 512, [8, 16])]
    if k == 48:
      cfgs.append((2160, 3840, [1, 2, 4]))
    for H, W, bs in cfgs:
      spec_r = dataclasses.replace(spec_b, height=H, width=W)
      for idx_b, B in enumerate(bs):
        oxp, oxm, cxp, cxm = tile_poses(ds, B)
        if k == 48 and H == 2160 and idx_b == 0:
          fn_h = jax.jit(lambda da, a, b, c, d: render_wide_bvh_batch(spec_r, da, a, b, c, d, interpret=False).rgb)
          htxt = fn_h.lower(dev_b, oxp, oxm, cxp, cxm).as_text()
          hi = inspect_hlo_ops(htxt)
          hi["send_recv_ops"] = len(re.findall(r"\bsend\(|\brecv\(", htxt, re.I))
          assert hi["host_callbacks"] == 0 and hi["send_recv_ops"] == 0
          data["parity"]["hlo_10m_4k_1c"] = hi
        rec = {**st, **bench_one_unique(spec_r, dev_b, oxp, oxm, cxp, cxm, n_chips=1, stats_enabled=(idx_b == 0))}
        data["track_2a_unique_1c"].append(rec)
        _save_json(out_path, data)
        print("2A 1C:", rec, flush=True)
    del dev_b, spec_b; gc.collect()


def run_2a_1c_100m(out_path: str, png_path: str):
  ds, _, base_meshes = load_base_scene()
  data = _load_or_init_json(out_path)
  kw = dict(cam_fovy=float(ds["cam_fovy"]), znear=float(ds["znear"]), zfar=float(ds["zfar"]), light_pos=tuple(float(x) for x in ds["light_pos"]))

  spec_100m, dev_100m, t_sub, t_bvh, hbm_b = build_streaming_100m_unique_to_hbm(base_meshes, kw)
  st100 = spec_summary(spec_100m, hbm_b, t_sub, t_bvh)
  print(f"BUILT 108M UNIQUE: {st100}", flush=True)
  done = {(r["total_tris"], r["res"], r["B"]) for r in data["track_2a_unique_1c"]}
  for H, W, bs in [(2160, 3840, [1, 2, 4]), (64, 64, [64, 256]), (256, 256, [16, 64]), (512, 512, [8, 16])]:
    spec_r = dataclasses.replace(spec_100m, height=H, width=W)
    for idx_b, B in enumerate(bs):
      if (st100["total_tris"], f"{H}x{W}", B) in done:
        print(f"SKIP existing 108M 1C {H}x{W} B={B}", flush=True)
        continue
      oxp, oxm, cxp, cxm = make_100m_gallery_poses(ds, B)
      if H == 2160 and idx_b == 0:
        fn_img = jax.jit(lambda da, a, b, c, d: render_wide_bvh_batch(spec_r, da, a, b, c, d, interpret=False).rgb)
        htxt = fn_img.lower(dev_100m, oxp, oxm, cxp, cxm).as_text()
        hi = inspect_hlo_ops(htxt)
        hi["send_recv_ops"] = len(re.findall(r"\bsend\(|\brecv\(", htxt, re.I))
        assert hi["host_callbacks"] == 0 and hi["send_recv_ops"] == 0
        data["parity"]["hlo_108m_4k_1c"] = hi
        rgb_4k = np.asarray(fn_img(dev_100m, oxp, oxm, cxp, cxm)[0])
        img = Image.fromarray(np.clip(rgb_4k * 255.0 + 0.5, 0, 255).astype(np.uint8))
        img.save(png_path)
        print(f"Saved native 4K 108M+ PNG to {png_path} ({img.size})", flush=True)
      rec = {**st100, **bench_one_unique(spec_r, dev_100m, oxp, oxm, cxp, cxm, n_chips=1, stats_enabled=(idx_b == 0))}
      data["track_2a_unique_1c"].append(rec)
      _save_json(out_path, data)
      print("2A 108M 1C:", rec, flush=True)


def run_2a_4c_ladder(out_path: str):
  ds, _, base_meshes = load_base_scene()
  devs = jax.devices()
  assert len(devs) == 4
  mesh = Mesh(np.array(devs), ("data",))
  data = _load_or_init_json(out_path)
  kw = dict(cam_fovy=float(ds["cam_fovy"]), znear=float(ds["znear"]), zfar=float(ds["zfar"]), light_pos=tuple(float(x) for x in ds["light_pos"]))

  for k in [1, 5, 16, 48]:
    spec_b, dev_b, t_sub, t_bvh, hbm_b = build_streaming_3mesh_to_hbm(base_meshes, k, kw, mesh=mesh)
    st = spec_summary(spec_b, hbm_b, t_sub, t_bvh)
    cfgs = [(64, 64, [256, 1024]), (256, 256, [64, 256]), (512, 512, [16, 64])]
    if k == 48:
      cfgs.append((2160, 3840, [4, 8]))
    for H, W, bs in cfgs:
      spec_r = dataclasses.replace(spec_b, height=H, width=W)
      for idx_b, B in enumerate(bs):
        oxp, oxm, cxp, cxm = tile_poses(ds, B)
        rec = {**st, **bench_one_unique(spec_r, dev_b, oxp, oxm, cxp, cxm, n_chips=4, stats_enabled=False)}
        data["track_2a_unique_4c"].append(rec)
        _save_json(out_path, data)
        print("2A 4C:", rec, flush=True)
    del dev_b, spec_b; gc.collect()


def run_2a_4c_100m(out_path: str):
  ds, _, base_meshes = load_base_scene()
  devs = jax.devices()
  assert len(devs) == 4
  mesh = Mesh(np.array(devs), ("data",))
  data = _load_or_init_json(out_path)
  kw = dict(cam_fovy=float(ds["cam_fovy"]), znear=float(ds["znear"]), zfar=float(ds["zfar"]), light_pos=tuple(float(x) for x in ds["light_pos"]))

  spec_100m, dev_100m, t_sub, t_bvh, hbm_b = build_streaming_100m_unique_to_hbm(base_meshes, kw, mesh=mesh)
  st100 = spec_summary(spec_100m, hbm_b, t_sub, t_bvh)
  for H, W, bs in [(2160, 3840, [4, 8]), (64, 64, [256, 1024]), (256, 256, [64, 256]), (512, 512, [16, 64])]:
    spec_r = dataclasses.replace(spec_100m, height=H, width=W)
    for idx_b, B in enumerate(bs):
      oxp, oxm, cxp, cxm = make_100m_gallery_poses(ds, B)
      if H == 2160 and idx_b == 0:
        fn_4c, _ = make_sharded_wide_bvh_renderer(spec_r, devs, interpret=False, dev_arrays=dev_100m)
        htxt = fn_4c.lower(oxp, oxm, cxp, cxm).as_text()
        hi = inspect_hlo_ops(htxt)
        hi["send_recv_ops"] = len(re.findall(r"\bsend\(|\brecv\(", htxt, re.I))
        assert hi["host_callbacks"] == 0 and hi["send_recv_ops"] == 0
        data["parity"]["hlo_108m_4k_4c"] = hi
      rec = {**st100, **bench_one_unique(spec_r, dev_100m, oxp, oxm, cxp, cxm, n_chips=4, stats_enabled=False)}
      data["track_2a_unique_4c"].append(rec)
      _save_json(out_path, data)
      print("2A 108M 4C:", rec, flush=True)


def run_2b(out_path: str, n_chips: int):
  ds, _, base_meshes = load_base_scene()
  devs = jax.devices()
  mesh = Mesh(np.array(devs), ("data",)) if n_chips > 1 else None
  data = _load_or_init_json(out_path)
  kw = dict(cam_fovy=float(ds["cam_fovy"]), znear=float(ds["znear"]), zfar=float(ds["zfar"]), light_pos=tuple(float(x) for x in ds["light_pos"]))

  if n_chips == 1:
    # Parity check K_inst=1 on k=1
    spec_1, dev_1, _, _, _ = build_streaming_3mesh_to_hbm(base_meshes, 1, kw)
    oxp, oxm, cxp, cxm = tile_poses(ds, 4)
    ixp1 = tuple(oxp[:, m : m + 1, :] for m in range(3))
    ixm1 = tuple(oxm[:, m : m + 1, :, :] for m in range(3))
    gb_a = jax.jit(lambda: render_wide_bvh_batch(spec_1, dev_1, oxp, oxm, cxp, cxm, interpret=False))()
    gb_b = jax.jit(lambda: render_wide_bvh_instanced_batch(spec_1, dev_1, ixp1, ixm1, cxp, cxm, max_active_per_tile=4, interpret=False))()
    jax.block_until_ready(gb_b.rgb)
    data["parity"]["track_2b_k1_vs_2a"] = {
        "max_depth_diff_m": float(jnp.max(jnp.abs(gb_a.depth - gb_b.depth))),
        "max_rgb_diff": float(jnp.max(jnp.abs(gb_a.rgb - gb_b.rgb))),
        "geom_match": float(jnp.mean(gb_a.geom_id == gb_b.geom_id)),
    }
    assert data["parity"]["track_2b_k1_vs_2a"]["max_depth_diff_m"] < 1e-4
    print("TRACK 2B PARITY PASS:", data["parity"]["track_2b_k1_vs_2a"], flush=True)
    del spec_1, dev_1; gc.collect()

  # Build 10.64M-tri (k=48) shared BLAS once in HBM (979.7 MB)
  spec_blas, dev_blas, t_sub, t_bvh, hbm_b = build_streaming_3mesh_to_hbm(base_meshes, 48, kw, mesh=mesh)
  blas_tris = sum(it.n_tris_orig for it in spec_blas.meshes)
  hbm_mb = round(hbm_b / (1024 ** 2), 2)
  hbm_gb = round(hbm_b / (1024 ** 3), 4)

  inst_configs = [(10, 16), (50, 16), (96, 16)]  # 106.4M, 532.2M, 1.02B effective tris!
  out_key = "track_2b_instanced_1c" if n_chips == 1 else "track_2b_instanced_4c"

  for K_inst, max_act in inst_configs:
    eff_tris = blas_tris * K_inst
    if n_chips == 1:
      cfgs = [(64, 64, [64, 256]), (512, 512, [8, 16])]
      if K_inst <= 50:
        cfgs.append((2160, 3840, [1, 2]))
      else:
        cfgs.append((2160, 3840, [1]))
    else:
      cfgs = [(64, 64, [256, 1024]), (512, 512, [16, 64]), (2160, 3840, [4, 8] if K_inst <= 50 else [4])]

    for H, W, bs in cfgs:
      spec_r = dataclasses.replace(spec_blas, height=H, width=W)
      for idx_b, B in enumerate(bs):
        ixp, ixm, cxp, cxm = make_instanced_poses(ds, B, K_inst)
        if n_chips == 1:
          fn = jax.jit(lambda da, ip, im, cp, cm: render_wide_bvh_instanced_batch(spec_r, da, ip, im, cp, cm, max_active_per_tile=max_act, interpret=False).rgb)
          if K_inst == 96 and H == 512 and idx_b == 0:
            htxt = fn.lower(dev_blas, ixp, ixm, cxp, cxm).as_text()
            hi = inspect_hlo_ops(htxt)
            hi["send_recv_ops"] = len(re.findall(r"\bsend\(|\brecv\(", htxt, re.I))
            assert hi["host_callbacks"] == 0 and hi["send_recv_ops"] == 0
            data["parity"]["hlo_2b_1b_1c"] = hi
          t_c0 = time.perf_counter()
          y = fn(dev_blas, ixp, ixm, cxp, cxm)
          jax.block_until_ready(y)
          compile_s = time.perf_counter() - t_c0
          jax.block_until_ready(fn(dev_blas, ixp, ixm, cxp, cxm))
          n_steps = 2 if H >= 2160 else (5 if H >= 512 else 10)
          t0 = time.perf_counter()
          for _ in range(n_steps):
            y = fn(dev_blas, ixp, ixm, cxp, cxm)
          jax.block_until_ready(y)
          ms = (time.perf_counter() - t0) * 1e3 / n_steps
        else:
          fn_4c, _ = make_sharded_instanced_renderer(spec_r, devs, max_active_per_tile=max_act, interpret=False, dev_arrays=dev_blas)
          if K_inst == 96 and H == 512 and idx_b == 0:
            htxt = fn_4c.lower(ixp, ixm, cxp, cxm).as_text()
            hi = inspect_hlo_ops(htxt)
            hi["send_recv_ops"] = len(re.findall(r"\bsend\(|\brecv\(", htxt, re.I))
            assert hi["host_callbacks"] == 0 and hi["send_recv_ops"] == 0
            data["parity"]["hlo_2b_1b_4c"] = hi
          t_c0 = time.perf_counter()
          y = fn_4c(ixp, ixm, cxp, cxm).rgb
          jax.block_until_ready(y)
          compile_s = time.perf_counter() - t_c0
          jax.block_until_ready(fn_4c(ixp, ixm, cxp, cxm).rgb)
          n_steps = 2 if H >= 2160 else (5 if H >= 512 else 10)
          t0 = time.perf_counter()
          for _ in range(n_steps):
            y = fn_4c(ixp, ixm, cxp, cxm).rgb
          jax.block_until_ready(y)
          ms = (time.perf_counter() - t0) * 1e3 / n_steps

        fps = B * 1e3 / ms
        mrays = (B * H * W * 2.0) / (ms * 1e-3) / 1e6
        rec = {
            "blas_unique_tris": blas_tris,
            "k_instances": K_inst,
            "effective_tris": eff_tris,
            "bvh_hbm_mb": hbm_mb,
            "bvh_hbm_gb": hbm_gb,
            "n_chips": n_chips,
            "res": f"{H}x{W}",
            "B": B,
            "compile_s": round(compile_s, 2),
            "ms": round(ms, 2),
            "fps": round(fps, 2),
            "mrays_per_s": round(mrays, 2),
        }
        data[out_key].append(rec)
        _save_json(out_path, data)
        print(f"2B {n_chips}C:", rec, flush=True)

  _save_json(out_path, data)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--phase", choices=["2a_1c_ladder", "2a_1c_100m", "2a_4c_ladder", "2a_4c_100m", "2b_1c", "2b_4c"], required=True)
  ap.add_argument("--out", required=True)
  ap.add_argument("--png", default="/dev/shm/zaychik-tpu-mjx-w2/v3_w2_mesh_100m_4k.png")
  args = ap.parse_args()
  if args.phase == "2a_1c_ladder":
    run_2a_1c_ladder(args.out)
  elif args.phase == "2a_1c_100m":
    run_2a_1c_100m(args.out, args.png)
  elif args.phase == "2a_4c_ladder":
    run_2a_4c_ladder(args.out)
  elif args.phase == "2a_4c_100m":
    run_2a_4c_100m(args.out)
  elif args.phase == "2b_1c":
    run_2b(args.out, n_chips=1)
  elif args.phase == "2b_4c":
    run_2b(args.out, n_chips=4)


if __name__ == "__main__":
  main()
