"""Wave 3 Runner: 250M-500M Unique Non-Instanced (16-row Compact BF16),
1B-10B Dynamic Instanced (2-Level TLAS + Compact BLAS), Native 8K (7680x4320),
and High-Occlusion Sponza Architectural Megascene.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
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
import trimesh

from typing import Optional
from tpu_mjx_render.roofline import inspect_hlo_ops
from wave3_500m_10b_8k import (
    CompactWideBVHMeshItem,
    CompactWideBVHSceneSpec,
    build_compact_bvh_mesh_item,
    make_geometry_sharded_unique_renderer,
    make_spatial_8k_sharded_instanced_renderer,
    make_spatial_8k_sharded_unique_renderer,
    render_compact_bvh_batch,
    render_tlas2_compact_instanced_batch,
)
from wide_bvh_v3 import (
    _compute_smooth_vertex_normals,
    build_wide_bvh_scene_spec,
    render_wide_bvh_batch,
    to_device_arrays,
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
  return ds, base_meshes


def _strip(it: CompactWideBVHMeshItem) -> CompactWideBVHMeshItem:
  empty = np.empty((0,), dtype=np.float32)
  return dataclasses.replace(it, l2_roots_aabb=empty, l2_aabbs=empty, l0_aabbs=empty, leaf_bank=empty)


def _load_or_init(path: str):
  if os.path.exists(path):
    with open(path, "r") as f:
      return json.load(f)
  return {
      "backend": jax.default_backend(),
      "jax_version": jax.__version__,
      "device_kind": jax.devices()[0].device_kind,
      "hbm_gib_per_chip": 95.74,
      "parity": {},
      "track_3a_compact_unique": [],
      "track_3b_tlas2_billion_8k": [],
      "track_3c_sponza_megascene": [],
  }


def _save(path: str, d: dict):
  with open(path, "w") as f:
    json.dump(d, f, indent=2)


def _make_spec(items_meta, kw, H=64, W=64, use_shadows=True):
  return CompactWideBVHSceneSpec(
      height=H, width=W, cam_projection=0, cam_sensorsize=(0.0, 0.0), cam_intrinsic=(0.0, 0.0, 0.0, 0.0),
      meshes=tuple(items_meta),
      light_pos=np.asarray(kw["light_pos"], dtype=np.float32),
      light_ambient=(0.28, 0.30, 0.34), light_diffuse=(0.85, 0.83, 0.80), light_specular=(0.30, 0.30, 0.30),
      sky_rgb1=(0.28, 0.36, 0.48), sky_rgb2=(0.10, 0.14, 0.20),
      cam_fovy=kw["cam_fovy"], znear=kw["znear"], zfar=kw["zfar"], use_shadows=use_shadows,
  )


def build_n_unique_suzannes_compact(suz_dict, n_meshes: int, k: int, kw: dict, mesh: Optional[Mesh] = None):
  """Build n_meshes uniquely displaced high-poly Suzannes in compact 16-row (64 B/tri) HBM layout."""
  t0 = time.perf_counter()
  base_nv, base_nf, base_nn = subdivide_mesh_barycentric(suz_dict["verts"], suz_dict["faces"], k)
  t_sub = time.perf_counter() - t0
  palette = [
      [0.86, 0.48, 0.28, 1.0], [0.24, 0.72, 0.76, 1.0], [0.88, 0.74, 0.28, 1.0],
      [0.78, 0.34, 0.46, 1.0], [0.36, 0.54, 0.88, 1.0], [0.42, 0.78, 0.46, 1.0],
      [0.76, 0.78, 0.82, 1.0], [0.68, 0.46, 0.84, 1.0],
  ]
  rep = NamedSharding(mesh, P()) if mesh is not None else None
  items_meta, dev_arrays = [], []
  hbm_bytes = 0
  t_bvh = 0.0

  def _build_one(s_i: int):
    fx, fy, fz, ph = 18.0 + 3.1 * s_i, 21.0 + 2.7 * s_i, 25.0 + 4.3 * s_i, 0.61 * s_i
    disp = 0.007 * (np.sin(fx * base_nv[:, 0] + ph) * np.cos(fy * base_nv[:, 1] - ph) + 0.45 * np.sin(fz * base_nv[:, 2] + ph))
    nv_u = (base_nv + disp[:, None] * base_nn).astype(np.float32)
    it = build_compact_bvh_mesh_item(
        s_i,
        {"name": f"suz_u_{s_i}", "verts": nv_u, "faces": base_nf, "vert_normals": base_nn, "rgba": palette[s_i % len(palette)]},
        build_method="spatial_l1_morton",
    )
    del nv_u, disp
    return it

  # Build in batches of 16 with ThreadPoolExecutor (w2 has 208 cores / 440 GB RAM)
  t_b0 = time.perf_counter()
  for b_start in range(0, n_meshes, 16):
    batch_ids = list(range(b_start, min(b_start + 16, n_meshes)))
    with ThreadPoolExecutor(max_workers=len(batch_ids)) as ex:
      built = list(ex.map(_build_one, batch_ids))
    for it in built:
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
      items_meta.append(_strip(it))
    del built
    gc.collect()
  t_bvh = time.perf_counter() - t_b0
  del base_nv, base_nf, base_nn
  gc.collect()
  return _make_spec(items_meta, kw), tuple(dev_arrays), t_sub, t_bvh, hbm_bytes


def make_gallery_n_poses(ds, B: int, n_meshes: int):
  oxp = np.zeros((B, n_meshes, 3), dtype=np.float32)
  oxm = np.broadcast_to(np.eye(3, dtype=np.float32)[None, None, :, :], (B, n_meshes, 3, 3)).copy()
  cols = 5
  for m in range(n_meshes):
    r, c = m // cols, m % cols
    oxp[:, m, 0] = -0.45 + 0.22 * c
    oxp[:, m, 1] = -0.15 + 0.25 * r
    oxp[:, m, 2] = 0.16 + 0.12 * (r % 2)
    for b in range(B):
      ang = 0.05 * b + 0.3 * m
      ca, sa = math.cos(ang), math.sin(ang)
      oxm[b, m] = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)

  def _tile(arr):
    rep = (B + arr.shape[0] - 1) // arr.shape[0]
    return np.tile(arr, (rep,) + (1,) * (arr.ndim - 1))[:B]

  return jnp.asarray(oxp), jnp.asarray(oxm), jnp.asarray(_tile(ds["cam_xpos"])), jnp.asarray(_tile(ds["cam_xmat"]))


def run_w3_parity_and_250m(out_path: str):
  ds, base_meshes = load_base_scene()
  data = _load_or_init(out_path)
  kw = dict(cam_fovy=float(ds["cam_fovy"]), znear=float(ds["znear"]), zfar=float(ds["zfar"]), light_pos=tuple(float(x) for x in ds["light_pos"]))

  # 1. Compact 16-row BF16-Normal Parity vs FP32 24-row Reference on 4,620 tris
  spec_fp32 = build_wide_bvh_scene_spec(base_meshes, height=64, width=64, **kw)
  dev_fp32 = to_device_arrays(spec_fp32)
  items_c = [build_compact_bvh_mesh_item(i, m, "spatial_l1_morton") for i, m in enumerate(base_meshes)]
  spec_c = _make_spec([_strip(x) for x in items_c], kw, 64, 64)
  dev_c = tuple((jnp.asarray(x.l2_roots_aabb), jnp.asarray(x.l2_aabbs), jnp.asarray(x.l0_aabbs), jnp.asarray(x.leaf_bank)) for x in items_c)

  oxp_e, oxm_e = jnp.asarray(ds["obj_xpos"][:8]), jnp.asarray(ds["obj_xmat"][:8])
  cxp_e, cxm_e = jnp.asarray(ds["cam_xpos"][:8]), jnp.asarray(ds["cam_xmat"][:8])

  gb_fp32 = jax.jit(lambda: render_wide_bvh_batch(spec_fp32, dev_fp32, oxp_e, oxm_e, cxp_e, cxm_e, interpret=False))()
  fn_c = jax.jit(lambda da, a, b, c, d: render_compact_bvh_batch(spec_c, da, a, b, c, d, interpret=False))
  gb_c = fn_c(dev_c, oxp_e, oxm_e, cxp_e, cxm_e)
  jax.block_until_ready(gb_c.rgb)

  htxt = fn_c.lower(dev_c, oxp_e, oxm_e, cxp_e, cxm_e).as_text()
  hlo = inspect_hlo_ops(htxt)
  hlo["send_recv_ops"] = len(re.findall(r"\bsend\(|\brecv\(", htxt, re.I))
  assert hlo["host_callbacks"] == 0 and hlo["send_recv_ops"] == 0

  p_cmp = {
      "max_depth_diff_vs_fp32_m": float(jnp.max(jnp.abs(gb_fp32.depth - gb_c.depth))),
      "geom_match_vs_fp32": float(jnp.mean(gb_fp32.geom_id == gb_c.geom_id)),
      "max_rgb_diff_vs_fp32": float(jnp.max(jnp.abs(gb_fp32.rgb - gb_c.rgb))),
      "mean_rgb_diff_vs_fp32": float(jnp.mean(jnp.abs(gb_fp32.rgb - gb_c.rgb))),
      "hlo": hlo,
  }
  assert p_cmp["max_depth_diff_vs_fp32_m"] < 1e-4 and p_cmp["geom_match_vs_fp32"] == 1.0
  data["parity"]["compact_bf16_16row_vs_fp32"] = p_cmp
  print("COMPACT 16-ROW PARITY PASS:", p_cmp, flush=True)

  # TLAS2 parity K_inst=1
  ixp1 = tuple(oxp_e[:, m : m + 1, :] for m in range(3))
  ixm1 = tuple(oxm_e[:, m : m + 1, :, :] for m in range(3))
  gb_t2 = jax.jit(lambda: render_tlas2_compact_instanced_batch(spec_c, dev_c, ixp1, ixm1, cxp_e, cxm_e, tlas_group_size=1, tlas_top_groups=1, max_active_per_tile=1, interpret=False))()
  jax.block_until_ready(gb_t2.rgb)
  p_t2 = {
      "max_depth_diff_m": float(jnp.max(jnp.abs(gb_c.depth - gb_t2.depth))),
      "geom_match": float(jnp.mean(gb_c.geom_id == gb_t2.geom_id)),
  }
  assert p_t2["max_depth_diff_m"] < 1e-4 and p_t2["geom_match"] == 1.0
  data["parity"]["tlas2_k1_vs_compact"] = p_t2
  _save(out_path, data)
  del spec_fp32, dev_fp32, items_c, spec_c, dev_c; gc.collect()

  # 2. 264.66M Unique Non-Instanced Tris (10 uniquely displaced Suzannes, k=82, ~16 GB HBM)
  spec_250, dev_250, t_sub, t_bvh, hbm_b = build_n_unique_suzannes_compact(base_meshes[0], n_meshes=10, k=82, kw=kw)
  tot_tris = sum(it.n_tris_orig for it in spec_250.meshes)
  print(f"BUILT 264M UNIQUE: tris={tot_tris:,} sub={t_sub:.2f}s bvh={t_bvh:.2f}s hbm={hbm_b/(1024**3):.3f} GB", flush=True)

  for H, W, bs in [(64, 64, [64, 256]), (512, 512, [8]), (2160, 3840, [1, 2])]:
    spec_r = dataclasses.replace(spec_250, height=H, width=W)
    for idx_b, B in enumerate(bs):
      oxp, oxm, cxp, cxm = make_gallery_n_poses(ds, B, 10)
      fn = jax.jit(lambda da, a, b, c, d: render_compact_bvh_batch(spec_r, da, a, b, c, d, interpret=False).rgb)
      if H == 2160 and idx_b == 0:
        htxt = fn.lower(dev_250, oxp, oxm, cxp, cxm).as_text()
        hi = inspect_hlo_ops(htxt)
        hi["send_recv_ops"] = len(re.findall(r"\bsend\(|\brecv\(", htxt, re.I))
        assert hi["host_callbacks"] == 0 and hi["send_recv_ops"] == 0
        data["parity"]["hlo_264m_4k_1c"] = hi
      tc0 = time.perf_counter()
      jax.block_until_ready(fn(dev_250, oxp, oxm, cxp, cxm))
      cs = time.perf_counter() - tc0
      n_steps = 2 if H >= 2160 else (5 if H >= 512 else 10)
      t0 = time.perf_counter()
      for _ in range(n_steps):
        y = fn(dev_250, oxp, oxm, cxp, cxm)
      jax.block_until_ready(y)
      ms = (time.perf_counter() - t0) * 1e3 / n_steps
      rec = {
          "total_tris": tot_tris, "n_meshes": 10,
          "bvh_hbm_gb": round(hbm_b / (1024 ** 3), 4),
          "bvh_build_s": round(t_bvh, 2), "n_chips": 1, "res": f"{H}x{W}", "B": B,
          "compile_s": round(cs, 2), "ms": round(ms, 2), "fps": round(B * 1e3 / ms, 2),
          "mrays_per_s": round((B * H * W * 2.0) / (ms * 1e-3) / 1e6, 2),
      }
      if idx_b == 0:
        _, lv = jax.jit(lambda da, a, b, c, d: render_compact_bvh_batch(spec_r, da, a, b, c, d, interpret=False, return_stats=True))(dev_250, oxp[:1], oxm[:1], cxp[:1], cxm[:1])
        jax.block_until_ready(lv)
        avg_l = float(lv[0]) / float((H // 8) * (W // 16))
        rec["avg_leaves_per_tile"] = round(avg_l, 2)
        rec["culling_factor_x"] = round(tot_tris / max(avg_l * 128.0, 1.0), 1)
      data["track_3a_compact_unique"].append(rec)
      _save(out_path, data)
      print("3A 264M:", rec, flush=True)


def run_w3_500m_1c(out_path: str):
  ds, base_meshes = load_base_scene()
  data = _load_or_init(out_path)
  kw = dict(cam_fovy=float(ds["cam_fovy"]), znear=float(ds["znear"]), zfar=float(ds["zfar"]), light_pos=tuple(float(x) for x in ds["light_pos"]))

  # 502.85M Unique Non-Instanced Tris (19 uniquely displaced Suzannes, k=82, ~30.5 GB HBM)
  spec_500, dev_500, t_sub, t_bvh, hbm_b = build_n_unique_suzannes_compact(base_meshes[0], n_meshes=19, k=82, kw=kw)
  tot_tris = sum(it.n_tris_orig for it in spec_500.meshes)
  print(f"BUILT 502M UNIQUE: tris={tot_tris:,} sub={t_sub:.2f}s bvh={t_bvh:.2f}s hbm={hbm_b/(1024**3):.3f} GB", flush=True)

  for H, W, bs in [(2160, 3840, [1]), (64, 64, [64]), (512, 512, [8])]:
    spec_r = dataclasses.replace(spec_500, height=H, width=W)
    for idx_b, B in enumerate(bs):
      oxp, oxm, cxp, cxm = make_gallery_n_poses(ds, B, 19)
      fn = jax.jit(lambda da, a, b, c, d: render_compact_bvh_batch(spec_r, da, a, b, c, d, interpret=False).rgb)
      if H == 2160 and idx_b == 0:
        htxt = fn.lower(dev_500, oxp, oxm, cxp, cxm).as_text()
        hi = inspect_hlo_ops(htxt)
        hi["send_recv_ops"] = len(re.findall(r"\bsend\(|\brecv\(", htxt, re.I))
        assert hi["host_callbacks"] == 0 and hi["send_recv_ops"] == 0 and hi["gather_ops"] == 0
        data["parity"]["hlo_502m_4k_1c"] = hi
      tc0 = time.perf_counter()
      jax.block_until_ready(fn(dev_500, oxp, oxm, cxp, cxm))
      cs = time.perf_counter() - tc0
      n_steps = 2 if H >= 2160 else (3 if H >= 512 else 5)
      t0 = time.perf_counter()
      for _ in range(n_steps):
        y = fn(dev_500, oxp, oxm, cxp, cxm)
      jax.block_until_ready(y)
      ms = (time.perf_counter() - t0) * 1e3 / n_steps
      rec = {
          "total_tris": tot_tris, "n_meshes": 19,
          "bvh_hbm_gb": round(hbm_b / (1024 ** 3), 4),
          "bvh_build_s": round(t_bvh, 2), "n_chips": 1, "res": f"{H}x{W}", "B": B,
          "compile_s": round(cs, 2), "ms": round(ms, 2), "fps": round(B * 1e3 / ms, 2),
          "mrays_per_s": round((B * H * W * 2.0) / (ms * 1e-3) / 1e6, 2),
      }
      data["track_3a_compact_unique"].append(rec)
      _save(out_path, data)
      print("3A 502M:", rec, flush=True)


def make_stratified_3d_poses(ds, B: int, K_inst: int):
  """Create K_inst poses arranged in spatially coherent groups of 32 for 2-level TLAS."""
  # Arrange in a 3D lattice in front of camera [0.82, -0.58, 0.62] looking toward [-0.5, 0.5, 0.0]
  nx = max(1, int(round(K_inst ** (1/3))))
  ny = max(1, int(round(math.sqrt(K_inst / nx))))
  nz = max(1, int(math.ceil(K_inst / (nx * ny))))
  pts = []
  for ix in range(nx):
    for iy in range(ny):
      for iz in range(nz):
        if len(pts) >= K_inst:
          break
        pts.append([
            -0.9 + 1.8 * (ix / max(nx - 1, 1)),
            -0.4 + 1.8 * (iy / max(ny - 1, 1)),
            -0.1 + 0.8 * (iz / max(nz - 1, 1)),
        ])
  pts = np.array(pts[:K_inst], dtype=np.float32)
  # Morton-sort instance positions so consecutive groups of 32 are spatially coherent TLAS nodes!
  from wide_bvh_v3 import _morton3d_20bit
  pts = pts[np.argsort(_morton3d_20bit(pts), kind="stable")]

  def _tile(arr):
    rep = (B + arr.shape[0] - 1) // arr.shape[0]
    return np.tile(arr, (rep,) + (1,) * (arr.ndim - 1))[:B]

  base_oxp = _tile(ds["obj_xpos"])
  base_oxm = _tile(ds["obj_xmat"])
  cxp = jnp.asarray(_tile(ds["cam_xpos"]))
  cxm = jnp.asarray(_tile(ds["cam_xmat"]))

  ixp_l, ixm_l = [], []
  for m in range(3):
    pos_m = base_oxp[:, m : m + 1, :] + pts[None, :, :]
    rot_m = np.broadcast_to(base_oxm[:, m : m + 1, :, :], (B, K_inst, 3, 3)).copy()
    ang = (np.arange(K_inst, dtype=np.float32) * 0.37)[:, None, None]
    ca, sa = np.cos(ang), np.sin(ang)
    rz = np.zeros((K_inst, 3, 3), dtype=np.float32)
    rz[:, 0, 0] = ca[:, 0, 0]; rz[:, 0, 1] = -sa[:, 0, 0]
    rz[:, 1, 0] = sa[:, 0, 0]; rz[:, 1, 1] = ca[:, 0, 0]
    rz[:, 2, 2] = 1.0
    rot_m = rot_m @ rz[None, :, :, :]
    ixp_l.append(jnp.asarray(pos_m))
    ixm_l.append(jnp.asarray(rot_m))
  return tuple(ixp_l), tuple(ixm_l), cxp, cxm


def run_w3_tlas_1b_10b_4c(out_path: str, png_8k_path: str, png_crop_path: str, configs=None):
  ds, base_meshes = load_base_scene()
  devs = jax.devices()
  assert len(devs) == 4
  mesh = Mesh(np.array(devs), ("spatial",))
  rep = NamedSharding(mesh, P())
  data = _load_or_init(out_path)
  kw = dict(cam_fovy=float(ds["cam_fovy"]), znear=float(ds["znear"]), zfar=float(ds["zfar"]), light_pos=tuple(float(x) for x in ds["light_pos"]))

  # Build 10.64M Compact BLAS (k=48, 654 MB HBM!) replicated on 4 chips
  items_c = []
  dev_c = []
  hbm_b = 0
  for mid, m in enumerate(base_meshes):
    nv, nf, nn = subdivide_mesh_barycentric(m["verts"], m["faces"], 48)
    it = build_compact_bvh_mesh_item(mid, {"name": m["name"], "verts": nv, "faces": nf, "vert_normals": nn, "rgba": m["rgba"]}, "spatial_l1_morton")
    del nv, nf, nn
    hbm_b += it.l2_roots_aabb.nbytes + it.l2_aabbs.nbytes + it.l0_aabbs.nbytes + it.leaf_bank.nbytes
    dev_c.append((jax.device_put(it.l2_roots_aabb, rep), jax.device_put(it.l2_aabbs, rep), jax.device_put(it.l0_aabbs, rep), jax.device_put(it.leaf_bank, rep)))
    items_c.append(_strip(it))
    gc.collect()
  dev_c = tuple(dev_c)
  blas_tris = sum(it.n_tris_orig for it in items_c)
  print(f"BUILT 10.64M COMPACT BLAS: tris={blas_tris:,} hbm={hbm_b/(1024**2):.1f} MB", flush=True)

  if configs is None:
    configs = [(96, 16, 4), (960, 30, 4), (9600, 64, 4)]
  # Test K_inst = 96 (1.02B), 960 (10.22B), and 9600 (102.18B effective tris) at 4K & Spatial-Sharded 8K!
  for K_inst, grp_sz, top_g in configs:
    eff_tris = blas_tris * K_inst
    ixp, ixm, cxp, cxm = make_stratified_3d_poses(ds, 1, K_inst)
    ixp_r = tuple(jax.device_put(np.asarray(x), rep) for x in ixp)
    ixm_r = tuple(jax.device_put(np.asarray(x), rep) for x in ixm)
    cxp_r = jax.device_put(np.asarray(cxp), rep)
    cxm_r = jax.device_put(np.asarray(cxm), rep)

    for H, W in [(2160, 3840), (4320, 7680)]:
      spec_full = _make_spec(items_c, kw, H, W)
      fn_8k, _ = make_spatial_8k_sharded_instanced_renderer(
          spec_full, devs, dev_c, tlas_group_size=grp_sz, tlas_top_groups=top_g, max_active_per_tile=16, interpret=False,
      )
      if K_inst in (960, 9600) and H == 4320:
        htxt = fn_8k.lower(ixp_r, ixm_r, cxp_r, cxm_r).as_text()
        hi = inspect_hlo_ops(htxt)
        hi["send_recv_ops"] = len(re.findall(r"\bsend\(|\brecv\(", htxt, re.I))
        assert hi["host_callbacks"] == 0 and hi["send_recv_ops"] == 0 and hi["gather_ops"] == 0, hi
        key_h = "hlo_10b_8k_4c_spatial" if K_inst == 960 else "hlo_100b_8k_4c_spatial"
        data["parity"][key_h] = hi

      tc0 = time.perf_counter()
      rgb = fn_8k(ixp_r, ixm_r, cxp_r, cxm_r)
      jax.block_until_ready(rgb)
      cs = time.perf_counter() - tc0

      if K_inst == 9600 and H == 4320:
        rgb_np = np.asarray(rgb[0])
        img = Image.fromarray(np.clip(rgb_np * 255.0 + 0.5, 0, 255).astype(np.uint8))
        img.save(png_8k_path)
        cx, cy = 7680 // 2, 4320 // 2
        crop = img.crop((cx - 960, cy - 540, cx + 960, cy + 540))
        crop.save(png_crop_path)
        print(f"Saved Native 8K 100B+ Hero ({img.size}) + 100% crop ({crop.size}) to {png_8k_path}", flush=True)

      n_steps = 2
      t0 = time.perf_counter()
      for _ in range(n_steps):
        rgb = fn_8k(ixp_r, ixm_r, cxp_r, cxm_r)
      jax.block_until_ready(rgb)
      ms = (time.perf_counter() - t0) * 1e3 / n_steps
      rec = {
          "blas_unique_tris": blas_tris,
          "k_instances": K_inst,
          "effective_tris": eff_tris,
          "bvh_hbm_mb": round(hbm_b / (1024 ** 2), 2),
          "n_chips": 4,
          "sharding": "spatial_4band",
          "res": f"{H}x{W}",
          "B": 1,
          "compile_s": round(cs, 2),
          "ms": round(ms, 2),
          "fps": round(1e3 / ms, 3),
          "mrays_per_s": round((H * W * 2.0) / (ms * 1e-3) / 1e6, 2),
      }
      data["track_3b_tlas2_billion_8k"].append(rec)
      _save(out_path, data)
      print("3B TLAS2 4C:", rec, flush=True)


def run_w3_sponza_megascene(out_path: str, png_4k_path: str):
  """Complex High-Occlusion Architectural Megascene: Sponza Colonnade subdivided to 105M+ tris."""
  ds, base_meshes = load_base_scene()
  devs = jax.devices()
  assert len(devs) == 4
  mesh = Mesh(np.array(devs), ("spatial",))
  rep = NamedSharding(mesh, P())
  data = _load_or_init(out_path)

  sc = trimesh.load("/dev/shm/zaychik-tpu-mjx/assets/sponza/Sponza.gltf", force="scene", process=False)
  with open("/dev/shm/zaychik-tpu-mjx/assets/sponza/Sponza.gltf", "r") as f_gltf:
    gltf_meta = json.load(f_gltf)
  prim_mats = [p.get("material", 0) for p in gltf_meta["meshes"][0]["primitives"]]
  # Group the 103 Sponza primitives by material ID into 8 architectural color buckets and rotate Y-up -> Z-up
  bucket_verts = [[] for _ in range(8)]
  bucket_faces = [[] for _ in range(8)]
  bucket_voff = [0] * 8
  arch_palette = [
      [0.78, 0.74, 0.66, 1.0],  # 0: limestone arches & walls
      [0.64, 0.60, 0.54, 1.0],  # 1: stone columns & capitals
      [0.46, 0.43, 0.40, 1.0],  # 2: marble floor / base
      [0.72, 0.36, 0.24, 1.0],  # 3: terracotta roof / brickwork
      [0.36, 0.28, 0.20, 1.0],  # 4: dark timber / chains / foliage bases
      [0.70, 0.16, 0.16, 1.0],  # 5: crimson banners & drapes
      [0.20, 0.38, 0.58, 1.0],  # 6: cobalt / emerald curtains
      [0.78, 0.64, 0.32, 1.0],  # 7: gilded moldings & lions
  ]
  mat_to_bucket = {
      0: 4, 1: 4, 2: 7, 3: 4, 4: 0, 5: 0, 6: 0, 7: 1, 8: 1, 9: 2, 10: 0,
      11: 1, 12: 0, 13: 1, 14: 5, 15: 6, 16: 5, 17: 6, 18: 5, 19: 6,
      20: 4, 21: 4, 22: 2, 23: 7, 24: 3,
  }
  for g_idx, node_name in enumerate(sc.graph.nodes_geometry):
    M, g_name = sc.graph.get(node_name)
    geom = sc.geometry[g_name]
    if geom.vertices.size == 0 or geom.faces.size == 0:
      continue
    v = trimesh.transform_points(np.asarray(geom.vertices, dtype=np.float32), M).astype(np.float32)
    f = np.asarray(geom.faces, dtype=np.int32)
    # Convert glTF Y-up -> Z-up: (x, y, z) -> (x, -z, y)
    vz = np.stack([v[:, 0], -v[:, 2], v[:, 1]], axis=-1)
    mat_id = prim_mats[g_idx] if g_idx < len(prim_mats) else (g_idx % 25)
    b = mat_to_bucket.get(mat_id, g_idx % 8)
    bucket_verts[b].append(vz)
    bucket_faces[b].append(f + bucket_voff[b])
    bucket_voff[b] += vz.shape[0]

  # Subdivide Sponza architectural buckets by k=20 (262,267 * 400 = 104,906,800 unique tris!)
  items_meta, dev_c = [], []
  hbm_b = 0
  t_b0 = time.perf_counter()
  mid = 0
  for b in range(8):
    if not bucket_verts[b]:
      continue
    vb = np.vstack(bucket_verts[b])
    fb = np.vstack(bucket_faces[b])
    nv, nf, nn = subdivide_mesh_barycentric(vb, fb, 20)
    del vb, fb
    it = build_compact_bvh_mesh_item(mid, {"name": f"sponza_b{b}", "verts": nv, "faces": nf, "vert_normals": nn, "rgba": arch_palette[b]}, "spatial_l1_morton")
    del nv, nf, nn
    hbm_b += it.l2_roots_aabb.nbytes + it.l2_aabbs.nbytes + it.l0_aabbs.nbytes + it.leaf_bank.nbytes
    dev_c.append((jax.device_put(it.l2_roots_aabb, rep), jax.device_put(it.l2_aabbs, rep), jax.device_put(it.l0_aabbs, rep), jax.device_put(it.leaf_bank, rep)))
    items_meta.append(_strip(it))
    mid += 1
    gc.collect()

  # Plus high-poly interior hero statue inside the colonnade atrium (k=32 -> ~3.8M tris)
  suz_nv, suz_nf, suz_nn = subdivide_mesh_barycentric(base_meshes[0]["verts"] * 8.0, base_meshes[0]["faces"], 32)
  it_s = build_compact_bvh_mesh_item(mid, {"name": "atrium_suzanne", "verts": suz_nv, "faces": suz_nf, "vert_normals": suz_nn, "rgba": [0.85, 0.72, 0.35, 1.0]}, "spatial_l1_morton")
  del suz_nv, suz_nf, suz_nn
  hbm_b += it_s.l2_roots_aabb.nbytes + it_s.l2_aabbs.nbytes + it_s.l0_aabbs.nbytes + it_s.leaf_bank.nbytes
  dev_c.append((jax.device_put(it_s.l2_roots_aabb, rep), jax.device_put(it_s.l2_aabbs, rep), jax.device_put(it_s.l0_aabbs, rep), jax.device_put(it_s.leaf_bank, rep)))
  items_meta.append(_strip(it_s))
  mid += 1
  t_bvh = time.perf_counter() - t_b0
  dev_c = tuple(dev_c)
  tot_tris = sum(it.n_tris_orig for it in items_meta)
  print(f"BUILT SPONZA MEGASCENE: tris={tot_tris:,} meshes={len(items_meta)} bvh={t_bvh:.2f}s hbm={hbm_b/(1024**3):.3f} GB", flush=True)

  # Camera inside Sponza colonnade looking down long atrium (deep multi-layer architectural occlusion)
  cam_pos = np.array([[9.0, 0.25, 2.2]], dtype=np.float32)
  target = np.array([-8.0, 0.25, 2.8], dtype=np.float32)
  fwd = target - cam_pos[0]; fwd /= np.linalg.norm(fwd)
  right = np.cross(fwd, np.array([0.0, 0.0, 1.0], np.float32)); right /= np.linalg.norm(right)
  up = np.cross(right, fwd)
  cam_mat = np.stack([right, up, -fwd], axis=1)[None, :, :].astype(np.float32)

  oxp = np.zeros((1, len(items_meta), 3), dtype=np.float32)
  oxp[0, -1] = np.array([1.5, 0.25, 1.1], dtype=np.float32)  # Hero statue in atrium center
  oxm = np.broadcast_to(np.eye(3, dtype=np.float32)[None, None, :, :], (1, len(items_meta), 3, 3)).copy()

  oxp_r, oxm_r = jax.device_put(oxp, rep), jax.device_put(oxm, rep)
  cxp_r, cxm_r = jax.device_put(cam_pos, rep), jax.device_put(cam_mat, rep)
  kw = dict(cam_fovy=55.0, znear=0.1, zfar=80.0, light_pos=(2.0, 0.25, 14.0))

  for H, W in [(2160, 3840), (4320, 7680)]:
    spec_full = _make_spec(items_meta, kw, H, W, use_shadows=True)
    fn_sp, _ = make_spatial_8k_sharded_unique_renderer(spec_full, devs, dev_c, interpret=False)
    if H == 2160:
      htxt = fn_sp.lower(oxp_r, oxm_r, cxp_r, cxm_r).as_text()
      hi = inspect_hlo_ops(htxt)
      hi["send_recv_ops"] = len(re.findall(r"\bsend\(|\brecv\(", htxt, re.I))
      assert hi["host_callbacks"] == 0 and hi["send_recv_ops"] == 0 and hi["gather_ops"] == 0
      data["parity"]["hlo_sponza_109m_4k_4c"] = hi

    tc0 = time.perf_counter()
    rgb = fn_sp(oxp_r, oxm_r, cxp_r, cxm_r)
    jax.block_until_ready(rgb)
    cs = time.perf_counter() - tc0

    if H == 2160:
      rgb_np = np.asarray(rgb[0])
      img = Image.fromarray(np.clip(rgb_np * 255.0 + 0.5, 0, 255).astype(np.uint8))
      img.save(png_4k_path)
      print(f"Saved Sponza Megascene 4K PNG to {png_4k_path} ({img.size})", flush=True)

    n_steps = 2
    t0 = time.perf_counter()
    for _ in range(n_steps):
      rgb = fn_sp(oxp_r, oxm_r, cxp_r, cxm_r)
    jax.block_until_ready(rgb)
    ms = (time.perf_counter() - t0) * 1e3 / n_steps
    rec = {
        "scene": "Sponza_HighOcclusion_Colonnade_Plus_Sculpture",
        "total_tris": tot_tris,
        "n_meshes": len(items_meta),
        "bvh_hbm_gb": round(hbm_b / (1024 ** 3), 4),
        "bvh_build_s": round(t_bvh, 2),
        "n_chips": 4,
        "sharding": "spatial_4band",
        "res": f"{H}x{W}",
        "B": 1,
        "compile_s": round(cs, 2),
        "ms": round(ms, 2),
        "fps": round(1e3 / ms, 3),
        "mrays_per_s": round((H * W * 2.0) / (ms * 1e-3) / 1e6, 2),
    }
    data["track_3c_sponza_megascene"].append(rec)
    _save(out_path, data)
    print("3C SPONZA 4C:", rec, flush=True)


def run_w3_1b_unique_4c(out_path: str):
  """4-Chip Geometry-Sharded 1.029B (10.29 亿) 100% Unique Non-Instanced Tris via shard_map + pmin."""
  ds, base_meshes = load_base_scene()
  devs = jax.devices()
  assert len(devs) == 4
  mesh = Mesh(np.array(devs), ("geom",))
  sh_geom = NamedSharding(mesh, P("geom"))
  rep = NamedSharding(mesh, P())
  data = _load_or_init(out_path)
  kw = dict(cam_fovy=float(ds["cam_fovy"]), znear=float(ds["znear"]), zfar=float(ds["zfar"]), light_pos=tuple(float(x) for x in ds["light_pos"]))

  base_nv, base_nf, base_nn = subdivide_mesh_barycentric(base_meshes[0]["verts"], base_meshes[0]["faces"], 90)
  palette = [
      [0.86, 0.48, 0.28], [0.24, 0.72, 0.76], [0.88, 0.74, 0.28], [0.78, 0.34, 0.46],
      [0.36, 0.54, 0.88], [0.42, 0.78, 0.46], [0.76, 0.78, 0.82], [0.68, 0.46, 0.84],
  ]
  n_per_chip = 8
  n_total = 4 * n_per_chip  # 32 uniquely displaced Suzannes (k=90) = 1,028,505,600 unique tris!

  def _build_one(s_i: int):
    fx, fy, fz, ph = 18.0 + 3.1 * s_i, 21.0 + 2.7 * s_i, 25.0 + 4.3 * s_i, 0.61 * s_i
    disp = 0.007 * (np.sin(fx * base_nv[:, 0] + ph) * np.cos(fy * base_nv[:, 1] - ph) + 0.45 * np.sin(fz * base_nv[:, 2] + ph))
    nv_u = (base_nv + disp[:, None] * base_nn).astype(np.float32)
    it = build_compact_bvh_mesh_item(
        s_i,
        {"name": f"suz_1b_{s_i}", "verts": nv_u, "faces": base_nf, "vert_normals": base_nn, "rgba": palette[s_i % len(palette)]},
        build_method="spatial_l1_morton",
    )
    del nv_u, disp
    return it

  t_b0 = time.perf_counter()
  hbm_bytes = 0
  tot_tris = 0
  # Build in 2 batches of 16 across 208 CPU cores, then stack per slot across 4 chips
  built_all = []
  for b_start in range(0, n_total, 16):
    ids = list(range(b_start, min(b_start + 16, n_total)))
    with ThreadPoolExecutor(max_workers=len(ids)) as ex:
      built_all.extend(list(ex.map(_build_one, ids)))
  del base_nv, base_nf, base_nn
  gc.collect()

  per_slot_sharded = []
  rgba_host = np.zeros((4, n_per_chip, 3), dtype=np.float32)
  for m_slot in range(n_per_chip):
    s_r, s_l2, s_l0, s_lf = [], [], [], []
    for c in range(4):
      it = built_all[c * n_per_chip + m_slot]
      hbm_bytes += it.l2_roots_aabb.nbytes + it.l2_aabbs.nbytes + it.l0_aabbs.nbytes + it.leaf_bank.nbytes
      tot_tris += it.n_tris_orig
      rgba_host[c, m_slot] = it.rgba[:3]
      s_r.append(jax.device_put(it.l2_roots_aabb[None], devs[c]))
      s_l2.append(jax.device_put(it.l2_aabbs[None], devs[c]))
      s_l0.append(jax.device_put(it.l0_aabbs[None], devs[c]))
      s_lf.append(jax.device_put(it.leaf_bank[None], devs[c]))
      built_all[c * n_per_chip + m_slot] = None
    per_slot_sharded.append((
        jax.make_array_from_single_device_arrays((4, *s_r[0].shape[1:]), sh_geom, s_r),
        jax.make_array_from_single_device_arrays((4, *s_l2[0].shape[1:]), sh_geom, s_l2),
        jax.make_array_from_single_device_arrays((4, *s_l0[0].shape[1:]), sh_geom, s_l0),
        jax.make_array_from_single_device_arrays((4, *s_lf[0].shape[1:]), sh_geom, s_lf),
    ))
    gc.collect()
  t_bvh = time.perf_counter() - t_b0
  dev_sharded = tuple(per_slot_sharded)
  rgba_sharded = jax.device_put(rgba_host, sh_geom)
  print(f"BUILT 1.03B UNIQUE 4-CHIP: tris={tot_tris:,} meshes={n_total} bvh={t_bvh:.2f}s hbm={hbm_bytes/(1024**3):.3f} GB", flush=True)

  for H, W, B in [(2160, 3840, 1), (64, 64, 64), (512, 512, 8)]:
    spec_t = _make_spec([], kw, H, W, use_shadows=True)
    fn_4c, _ = make_geometry_sharded_unique_renderer(spec_t, devs, n_per_chip, interpret=False)
    oxp, oxm, cxp, cxm = make_gallery_n_poses(ds, B, n_total)
    oxp_s = jax.device_put(np.asarray(oxp).reshape(B, 4, n_per_chip, 3).transpose(1, 0, 2, 3), sh_geom)
    oxm_s = jax.device_put(np.asarray(oxm).reshape(B, 4, n_per_chip, 3, 3).transpose(1, 0, 2, 3, 4), sh_geom)
    cxp_r = jax.device_put(np.asarray(cxp), rep)
    cxm_r = jax.device_put(np.asarray(cxm), rep)

    if H == 2160:
      htxt = fn_4c.lower(dev_sharded, rgba_sharded, oxp_s, oxm_s, cxp_r, cxm_r).as_text()
      hi = inspect_hlo_ops(htxt)
      hi["send_recv_ops"] = len(re.findall(r"\bsend\(|\brecv\(", htxt, re.I))
      assert hi["host_callbacks"] == 0 and hi["send_recv_ops"] == 0 and hi["gather_ops"] == 0, hi
      data["parity"]["hlo_1b_unique_4k_4c"] = hi

    tc0 = time.perf_counter()
    rgb = fn_4c(dev_sharded, rgba_sharded, oxp_s, oxm_s, cxp_r, cxm_r)
    jax.block_until_ready(rgb)
    cs = time.perf_counter() - tc0

    n_steps = 2 if H >= 2160 else 3
    t0 = time.perf_counter()
    for _ in range(n_steps):
      rgb = fn_4c(dev_sharded, rgba_sharded, oxp_s, oxm_s, cxp_r, cxm_r)
    jax.block_until_ready(rgb)
    ms = (time.perf_counter() - t0) * 1e3 / n_steps
    rec = {
        "total_tris": tot_tris,
        "n_meshes": n_total,
        "bvh_hbm_gb": round(hbm_bytes / (1024 ** 3), 4),
        "bvh_build_s": round(t_bvh, 2),
        "n_chips": 4,
        "sharding": "geometry_4chip_pmin",
        "res": f"{H}x{W}",
        "B": B,
        "compile_s": round(cs, 2),
        "ms": round(ms, 2),
        "fps": round(B * 1e3 / ms, 2),
        "mrays_per_s": round((B * H * W * 2.0) / (ms * 1e-3) / 1e6, 2),
    }
    data["track_3a_compact_unique"].append(rec)
    _save(out_path, data)
    print("3A 1.03B 4C:", rec, flush=True)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--phase", choices=["w3_parity_250m", "w3_500m_1c", "w3_1b_unique_4c", "w3_tlas_1b_10b_4c", "w3_tlas_100b_4c", "w3_sponza_megascene"], required=True)
  ap.add_argument("--out", required=True)
  ap.add_argument("--png8k", default="/dev/shm/zaychik-tpu-mjx-w2/v3_w2_100b_8k_hero.png")
  ap.add_argument("--pngcrop", default="/dev/shm/zaychik-tpu-mjx-w2/v3_w2_100b_8k_crop.png")
  ap.add_argument("--pngsponza", default="/dev/shm/zaychik-tpu-mjx-w2/v3_w2_complex_megascene_4k.png")
  args = ap.parse_args()
  if args.phase == "w3_parity_250m":
    run_w3_parity_and_250m(args.out)
  elif args.phase == "w3_500m_1c":
    run_w3_500m_1c(args.out)
  elif args.phase == "w3_1b_unique_4c":
    run_w3_1b_unique_4c(args.out)
  elif args.phase == "w3_tlas_1b_10b_4c":
    run_w3_tlas_1b_10b_4c(args.out, args.png8k, args.pngcrop)
  elif args.phase == "w3_tlas_100b_4c":
    run_w3_tlas_1b_10b_4c(args.out, args.png8k, args.pngcrop, configs=[(9600, 64, 4)])
  elif args.phase == "w3_sponza_megascene":
    run_w3_sponza_megascene(args.out, args.pngsponza)


if __name__ == "__main__":
  main()
