"""Wave 2 Runner: Multi-Bounce TPU Path Tracing (256/512/4K) + 1B-Tri Native 8K UHD."""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import math
import os
import time
from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np
from PIL import Image, ImageDraw

from pathtracer_v3 import audit_hlo_zero_copy, build_pathtracer_fn
from wide_bvh_v3 import (
    WideBVHMeshItem,
    WideBVHSceneSpec,
    _compute_smooth_vertex_normals,
    build_wide_bvh_mesh_item,
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
  new_verts = (
      w * verts[faces[:, 0]][:, None] + u * verts[faces[:, 1]][:, None] + v * verts[faces[:, 2]][:, None]
  ).reshape(F * P_cnt, 3).astype(np.float32)
  new_norms = (
      w * vnorm[faces[:, 0]][:, None] + u * vnorm[faces[:, 1]][:, None] + v * vnorm[faces[:, 2]][:, None]
  ).reshape(F * P_cnt, 3)
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


def _strip_host(item: WideBVHMeshItem) -> WideBVHMeshItem:
  empty = np.empty((0,), dtype=np.float32)
  return dataclasses.replace(item, l2_roots_aabb=empty, l2_aabbs=empty, l0_aabbs=empty, leaf_bank=empty)


def build_3mesh_to_hbm(base_meshes, k: int, ds, mesh: Mesh):
  rep = NamedSharding(mesh, P())
  items_meta, dev_arrays = [], []
  t0 = time.perf_counter()
  hbm_bytes = 0
  for mid, m in enumerate(base_meshes):
    nv, nf, nn = subdivide_mesh_barycentric(m["verts"], m["faces"], k)
    it = build_wide_bvh_mesh_item(
        mid,
        {"name": m["name"], "verts": nv, "faces": nf, "vert_normals": nn, "rgba": m["rgba"]},
        build_method="spatial_l1_morton",
    )
    del nv, nf, nn
    hbm_bytes += it.l2_roots_aabb.nbytes + it.l2_aabbs.nbytes + it.l0_aabbs.nbytes + it.leaf_bank.nbytes
    dev_arrays.append(
        (
            jax.device_put(it.l2_roots_aabb, rep),
            jax.device_put(it.l2_aabbs, rep),
            jax.device_put(it.l0_aabbs, rep),
            jax.device_put(it.leaf_bank, rep),
        )
    )
    items_meta.append(_strip_host(it))
    del it
    gc.collect()
  build_s = time.perf_counter() - t0
  spec = WideBVHSceneSpec(
      height=256,
      width=256,
      cam_fovy=float(ds["cam_fovy"]),
      cam_projection=0,
      cam_sensorsize=(0.0, 0.0),
      cam_intrinsic=(0.0, 0.0, 0.0, 0.0),
      znear=float(ds["znear"]),
      zfar=float(ds["zfar"]),
      meshes=tuple(items_meta),
      light_pos=np.asarray(ds["light_pos"], dtype=np.float32),
      light_ambient=(0.28, 0.30, 0.34),
      light_diffuse=(0.85, 0.83, 0.80),
      light_specular=(0.30, 0.30, 0.30),
      sky_rgb1=(0.28, 0.36, 0.48),
      sky_rgb2=(0.10, 0.14, 0.20),
      use_shadows=True,
  )
  return spec, tuple(dev_arrays), hbm_bytes, build_s


def make_unique_poses(ds, B: int, mesh: Mesh):
  rep = NamedSharding(mesh, P())
  def _tile(arr):
    rep_n = (B + arr.shape[0] - 1) // arr.shape[0]
    return np.tile(arr, (rep_n,) + (1,) * (arr.ndim - 1))[:B]
  return (
      jax.device_put(np.asarray(_tile(ds["obj_xpos"]), np.float32), rep),
      jax.device_put(np.asarray(_tile(ds["obj_xmat"]), np.float32), rep),
      jax.device_put(np.asarray(_tile(ds["cam_xpos"]), np.float32), rep),
      jax.device_put(np.asarray(_tile(ds["cam_xmat"]), np.float32), rep),
  )


def make_instanced_poses(ds, B: int, K_inst: int, mesh: Mesh, pull_back: bool = False):
  rep = NamedSharding(mesh, P())
  offsets = np.zeros((K_inst, 3), dtype=np.float32)
  grid_n = int(math.ceil(math.sqrt(K_inst)))
  spread = 1.45 if pull_back else 1.15
  for k_i in range(K_inst):
    gx = (k_i % grid_n) / max(grid_n - 1, 1) - 0.5
    gy = (k_i // grid_n) / max(grid_n - 1, 1) - 0.5
    offsets[k_i] = np.array([gx * spread - 0.10, gy * spread + 0.08, 0.05 * (k_i % 3)], dtype=np.float32)

  def _tile(arr):
    rep_n = (B + arr.shape[0] - 1) // arr.shape[0]
    return np.tile(arr, (rep_n,) + (1,) * (arr.ndim - 1))[:B]

  base_oxp = _tile(ds["obj_xpos"])
  base_oxm = _tile(ds["obj_xmat"])
  cxp_np = np.asarray(_tile(ds["cam_xpos"]), np.float32)
  cxm_np = np.asarray(_tile(ds["cam_xmat"]), np.float32)
  if pull_back:
    pos = np.array([1.22, -0.98, 0.92], dtype=np.float32)
    tgt = np.array([-0.10, 0.08, 0.14], dtype=np.float32)
    fwd = tgt - pos
    fwd = fwd / np.linalg.norm(fwd)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    right = np.cross(fwd, up)
    right = right / np.linalg.norm(right)
    cam_up = np.cross(right, fwd)
    rmat = np.stack([right, cam_up, -fwd], axis=1).astype(np.float32)
    cxp_np[:] = pos[None, :]
    cxm_np[:] = rmat[None, :, :]

  ixp_list, ixm_list = [], []
  for m in range(3):
    pos_m = base_oxp[:, m : m + 1, :] + offsets[None, :, :]
    rot_m = np.broadcast_to(base_oxm[:, m : m + 1, :, :], (B, K_inst, 3, 3)).copy()
    for k_i in range(K_inst):
      ang = 0.4 * k_i
      c, s = math.cos(ang), math.sin(ang)
      rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
      rot_m[:, k_i] = rot_m[:, k_i] @ rz
    ixp_list.append(jax.device_put(np.asarray(pos_m, np.float32), rep))
    ixm_list.append(jax.device_put(np.asarray(rot_m, np.float32), rep))
  return (
      tuple(ixp_list),
      tuple(ixm_list),
      jax.device_put(cxp_np, rep),
      jax.device_put(cxm_np, rep),
  )


def stitch_strips(out_arr: jax.Array) -> np.ndarray:
  arr = np.asarray(jax.device_get(out_arr), dtype=np.float32)
  n_c, B, H, W_loc, C = arr.shape
  return arr.transpose(1, 2, 0, 3, 4).reshape(B, H, n_c * W_loc, C)


def bench_pt_cfg(
    label: str,
    spec: WideBVHSceneSpec,
    dev_arrays,
    poses_a,
    poses_b,
    cxp,
    cxm,
    devices,
    H: int,
    W: int,
    B: int,
    spp: int,
    max_bounces: int,
    instanced: bool = False,
    k_inst: int = 1,
    max_active_per_tile: int = 16,
    mat_palette=None,
    n_steps: int = 3,
    audit_hlo: bool = False,
    return_img: bool = False,
):
  spec_r = dataclasses.replace(spec, height=H, width=W)
  alb_t, met_t, rgh_t = mat_palette if mat_palette is not None else (None, None, None)
  fn, mesh, rep = build_pathtracer_fn(
      spec_r,
      dev_arrays,
      devices,
      spp=spp,
      max_bounces=max_bounces,
      instanced=instanced,
      max_active_per_tile=max_active_per_tile,
      mat_albedo_tbl=alb_t,
      mat_metal_tbl=met_t,
      mat_rough_tbl=rgh_t,
      tonemap=True,
  )
  rng = jax.device_put(jax.random.PRNGKey(0), rep)
  hlo_info = None
  if audit_hlo:
    hlo_info = audit_hlo_zero_copy(fn.lower(poses_a, poses_b, cxp, cxm, rng))

  t_c0 = time.perf_counter()
  y = fn(poses_a, poses_b, cxp, cxm, rng)
  jax.block_until_ready(y)
  compile_s = time.perf_counter() - t_c0

  rng = jax.device_put(jax.random.PRNGKey(1), rep)
  y = fn(poses_a, poses_b, cxp, cxm, rng)
  jax.block_until_ready(y)

  t0 = time.perf_counter()
  for it in range(n_steps):
    rng_it = jax.device_put(jax.random.PRNGKey(it + 2), rep)
    y = fn(poses_a, poses_b, cxp, cxm, rng_it)
  jax.block_until_ready(y)
  dt_s = (time.perf_counter() - t0) / n_steps

  ms = dt_s * 1e3
  fps = B / dt_s
  samples_per_min = fps * spp * 60.0
  pixel_msamples_per_min = (B * H * W * spp / dt_s) * 60.0 / 1e6
  rays_per_sample = 2 * max_bounces
  mrays_per_s = (B * H * W * spp * rays_per_sample / dt_s) / 1e6

  blas_tris = int(sum(it.n_tris_orig for it in spec.meshes))
  eff_tris = blas_tris * k_inst
  rec = {
      "label": label,
      "n_chips": len(devices),
      "instanced": instanced,
      "k_instances": k_inst,
      "blas_tris": blas_tris,
      "effective_tris": eff_tris,
      "res": f"{H}x{W}",
      "B": B,
      "spp": spp,
      "max_bounces": max_bounces,
      "compile_s": round(compile_s, 2),
      "ms": round(ms, 2),
      "fps": round(fps, 3),
      "samples_per_min": round(samples_per_min, 1),
      "pixel_msamples_per_min": round(pixel_msamples_per_min, 2),
      "mrays_per_s": round(mrays_per_s, 2),
  }
  if hlo_info is not None:
    rec["hlo_zero_copy"] = True
    rec["hlo_ops"] = hlo_info
  print("PT BENCH:", rec, flush=True)
  img_np = stitch_strips(y)[0] if return_img else None
  return rec, img_np


def save_gi_comparison_png(panels: List[Tuple[str, np.ndarray]], out_png: str):
  n_cols = 4
  n_rows = (len(panels) + n_cols - 1) // n_cols
  h, w, _ = panels[0][1].shape
  pad = 14
  bar = 40
  canvas = Image.new("RGB", (n_cols * w + (n_cols + 1) * pad, n_rows * (h + bar) + (n_rows + 1) * pad), (18, 20, 24))
  draw = ImageDraw.Draw(canvas)
  for idx, (title, img_f) in enumerate(panels):
    r, c = idx // n_cols, idx % n_cols
    x0 = pad + c * (w + pad)
    y0 = pad + r * (h + bar + pad)
    u8 = np.clip(img_f * 255.0 + 0.5, 0, 255).astype(np.uint8)
    canvas.paste(Image.fromarray(u8), (x0, y0 + bar))
    draw.text((x0 + 4, y0 + 12), title, fill=(235, 240, 245))
  os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
  canvas.save(out_png)
  print("Saved GI comparison PNG ->", out_png, canvas.size, flush=True)


def save_8k_overview_and_crop_png(img_8k: np.ndarray, out_png: str, meta_line: str):
  H, W, _ = img_8k.shape
  assert (H, W) == (4320, 7680), f"Expected 8K (4320, 7680), got {(H, W)}"
  u8 = np.clip(img_8k * 255.0 + 0.5, 0, 255).astype(np.uint8)
  full_im = Image.fromarray(u8)
  ov = full_im.resize((1920, 1080), Image.Resampling.LANCZOS)
  cx, cy = W // 2, H // 2
  crop = full_im.crop((cx - 540, cy - 540, cx + 540, cy + 540))
  draw_ov = ImageDraw.Draw(ov)
  rx0, ry0 = (cx - 540) // 4, (cy - 540) // 4
  rx1, ry1 = (cx + 540) // 4, (cy + 540) // 4
  draw_ov.rectangle([rx0, ry0, rx1, ry1], outline=(255, 220, 40), width=4)

  pad = 16
  hdr = 44
  canvas = Image.new("RGB", (1920 + 1080 + 3 * pad, 1080 + hdr + 2 * pad), (15, 17, 21))
  canvas.paste(ov, (pad, hdr + pad))
  canvas.paste(crop, (1920 + 2 * pad, hdr + pad))
  draw = ImageDraw.Draw(canvas)
  draw.text((pad, 14), f"Native 8K UHD (7680x4320, 33.18M px) — {meta_line}", fill=(240, 244, 250))
  draw.text((1920 + 2 * pad, 14), "Native 1:1 8K Center Crop (1080x1080 pixels)", fill=(255, 220, 60))
  os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
  canvas.save(out_png)
  print("Saved 8K overview+crop PNG ->", out_png, canvas.size, flush=True)


def run_wave2a_gi(out_json: str, out_gi_png: str):
  ds, base_meshes = load_base_scene()
  devs_all = jax.devices()
  assert len(devs_all) == 4
  mesh_4c = Mesh(np.array(devs_all), ("dp",))

  results = []
  spec_1m, dev_1m_4c, hbm_1m, build_s = build_3mesh_to_hbm(base_meshes, 16, ds, mesh_4c)
  oxp_1, oxm_1, cxp_1, cxm_1 = make_unique_poses(ds, 1, mesh_4c)
  gi_panels = []
  for bnc in [1, 3]:
    for spp in [1, 4, 16, 64]:
      rec, img = bench_pt_cfg(
          f"1.18M_4c_512_b{bnc}_spp{spp}",
          spec_1m, dev_1m_4c, oxp_1, oxm_1, cxp_1, cxm_1, devs_all,
          512, 512, 1, spp, bnc, instanced=False, n_steps=(1 if spp == 64 else 3),
          audit_hlo=(bnc == 3 and spp == 4), return_img=True,
      )
      rec["bvh_hbm_mb"] = round(hbm_1m / (1024 ** 2), 2)
      rec["bvh_build_s"] = round(build_s, 2)
      results.append(rec)
      gi_panels.append((f"{bnc}-Bounce GI | {spp} spp | {rec['ms']:.0f}ms ({rec['samples_per_min']:,.0f}/min)", img))
  save_gi_comparison_png(gi_panels, out_gi_png)

  # Also test 2-bounce & 4-bounce at 512x512 (4 spp) + 4K (1 & 4 spp) on 4 chips
  for bnc, spp in [(2, 4), (4, 4)]:
    rec, _ = bench_pt_cfg(
        f"1.18M_4c_512_b{bnc}_spp{spp}",
        spec_1m, dev_1m_4c, oxp_1, oxm_1, cxp_1, cxm_1, devs_all,
        512, 512, 1, spp, bnc, instanced=False, n_steps=3,
    )
    results.append(rec)

  for spp in [1, 4]:
    rec, _ = bench_pt_cfg(
        f"1.18M_4c_4K_b3_spp{spp}",
        spec_1m, dev_1m_4c, oxp_1, oxm_1, cxp_1, cxm_1, devs_all,
        2160, 3840, 1, spp, 3, instanced=False, n_steps=(1 if spp == 4 else 2),
        audit_hlo=(spp == 1),
    )
    results.append(rec)

  os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
  with open(out_json, "w") as f:
    json.dump({"wave": "2A_gi", "backend": jax.default_backend(), "runs": results}, f, indent=2)
  print("Saved Wave 2A GI ->", out_json, flush=True)


def run_wave2a_scale(out_json: str):
  ds, base_meshes = load_base_scene()
  devs_all = jax.devices()
  assert len(devs_all) == 4
  mesh_1c = Mesh(np.array(devs_all[:1]), ("dp",))
  mesh_4c = Mesh(np.array(devs_all), ("dp",))

  results = []
  # 1-chip vs 4-chip scaling on 1.18M tris (256x256, 512x512, 4K) + Batched B=16 @ 256x256
  spec_1m_1c, dev_1m_1c, hbm_1m, _ = build_3mesh_to_hbm(base_meshes, 16, ds, mesh_1c)
  oxp_u1, oxm_u1, cxp_u1, cxm_u1 = make_unique_poses(ds, 1, mesh_1c)
  for H, W, spps in [(256, 256, [1, 16]), (512, 512, [1, 16]), (2160, 3840, [1])]:
    for spp in spps:
      rec, _ = bench_pt_cfg(
          f"1.18M_1c_{H}_b3_spp{spp}",
          spec_1m_1c, dev_1m_1c, oxp_u1, oxm_u1, cxp_u1, cxm_u1, devs_all[:1],
          H, W, 1, spp, 3, instanced=False, n_steps=(1 if H >= 2160 or spp >= 16 else 3),
          audit_hlo=(H == 2160 and spp == 1),
      )
      rec["bvh_hbm_mb"] = round(hbm_1m / (1024 ** 2), 2)
      results.append(rec)
  del dev_1m_1c, oxp_u1, oxm_u1, cxp_u1, cxm_u1; gc.collect()

  spec_1m_4c, dev_1m_4c, _, _ = build_3mesh_to_hbm(base_meshes, 16, ds, mesh_4c)
  oxp_u4, oxm_u4, cxp_u4, cxm_u4 = make_unique_poses(ds, 1, mesh_4c)
  for H, W, spps in [(256, 256, [1, 16, 64])]:
    for spp in spps:
      rec, _ = bench_pt_cfg(
          f"1.18M_4c_{H}_b3_spp{spp}",
          spec_1m_4c, dev_1m_4c, oxp_u4, oxm_u4, cxp_u4, cxm_u4, devs_all,
          H, W, 1, spp, 3, instanced=False, n_steps=2,
      )
      results.append(rec)
  # Batched multi-frame throughput B=16 @ 256x256 (1.18M tris, 2 & 3 bounces)
  oxp_b16, oxm_b16, cxp_b16, cxm_b16 = make_unique_poses(ds, 16, mesh_4c)
  for bnc in [2, 3]:
    rec, _ = bench_pt_cfg(
        f"1.18M_4c_256_B16_b{bnc}_spp1",
        spec_1m_4c, dev_1m_4c, oxp_b16, oxm_b16, cxp_b16, cxm_b16, devs_all,
        256, 256, 16, 1, bnc, instanced=False, n_steps=2,
    )
    results.append(rec)
  del dev_1m_4c, oxp_u4, oxm_u4, cxp_u4, cxm_u4, oxp_b16, oxm_b16, cxp_b16, cxm_b16; gc.collect()

  # 106.4M instanced (k=48, K_inst=10) @ 256, 512, 4K (4-chip)
  spec_10m, dev_10m_4c, hbm_10m, _ = build_3mesh_to_hbm(base_meshes, 48, ds, mesh_4c)
  ixp10, ixm10, cxp10, cxm10 = make_instanced_poses(ds, 1, 10, mesh_4c, pull_back=False)
  for H, W, bnc, spp in [(256, 256, 3, 4), (512, 512, 3, 4), (2160, 3840, 2, 1)]:
    rec, _ = bench_pt_cfg(
        f"106M_inst_4c_{H}_b{bnc}_spp{spp}",
        spec_10m, dev_10m_4c, ixp10, ixm10, cxp10, cxm10, devs_all,
        H, W, 1, spp, bnc, instanced=True, k_inst=10, max_active_per_tile=16,
        n_steps=(1 if H >= 2160 else 2), audit_hlo=(H == 2160),
    )
    rec["bvh_hbm_mb"] = round(hbm_10m / (1024 ** 2), 2)
    results.append(rec)

  os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
  with open(out_json, "w") as f:
    json.dump({"wave": "2A_scale", "backend": jax.default_backend(), "runs": results}, f, indent=2)
  print("Saved Wave 2A Scale ->", out_json, flush=True)


def run_wave2b(out_json: str, out_8k_png: str):
  ds, base_meshes = load_base_scene()
  devs_all = jax.devices()
  assert len(devs_all) == 4
  mesh_4c = Mesh(np.array(devs_all), ("dp",))

  # 10.64M BLAS x 96 instances = 1,021,870,080 (1.02B) effective triangles!
  spec_10m, dev_10m_4c, hbm_10m, build_s = build_3mesh_to_hbm(base_meshes, 48, ds, mesh_4c)
  ixp96, ixm96, cxp96, cxm96 = make_instanced_poses(ds, 1, 96, mesh_4c, pull_back=True)

  mat_alb = np.array([
      [0.92, 0.72, 0.30], [0.85, 0.30, 0.20], [0.22, 0.68, 0.80], [0.82, 0.84, 0.88],
      [0.25, 0.72, 0.45], [0.28, 0.44, 0.86], [0.88, 0.50, 0.28], [0.76, 0.74, 0.70],
      [0.72, 0.32, 0.68], [0.90, 0.80, 0.42], [0.62, 0.64, 0.68], [0.38, 0.78, 0.72],
  ], dtype=np.float32)
  mat_met = np.array([0.92, 0.05, 0.85, 0.95, 0.10, 0.80, 0.90, 0.0, 0.15, 0.88, 0.0, 0.75], dtype=np.float32)
  mat_rgh = np.array([0.14, 0.36, 0.16, 0.10, 0.32, 0.18, 0.16, 0.55, 0.35, 0.15, 0.60, 0.20], dtype=np.float32)
  palette = (mat_alb, mat_met, mat_rgh)

  results = []
  for H, W, bnc, spp in [(512, 512, 3, 1), (512, 512, 3, 4), (2160, 3840, 2, 1)]:
    rec, _ = bench_pt_cfg(
        f"1.02B_inst_4c_{H}_b{bnc}_spp{spp}",
        spec_10m, dev_10m_4c, ixp96, ixm96, cxp96, cxm96, devs_all,
        H, W, 1, spp, bnc, instanced=True, k_inst=96, max_active_per_tile=16,
        mat_palette=palette, n_steps=(1 if H >= 2160 else 3), audit_hlo=(H == 2160),
    )
    rec["bvh_hbm_mb"] = round(hbm_10m / (1024 ** 2), 2)
    rec["bvh_build_s"] = round(build_s, 2)
    results.append(rec)

  # Native 8K UHD (7680x4320 = 33.18M px, 259,200 8x16 tiles = 64,800/chip, zero padding) on 1.02B tris!
  rec_8k, img_8k = bench_pt_cfg(
      "1.02B_inst_4c_8K_b2_spp1",
      spec_10m, dev_10m_4c, ixp96, ixm96, cxp96, cxm96, devs_all,
      4320, 7680, 1, 1, 2, instanced=True, k_inst=96, max_active_per_tile=16,
      mat_palette=palette, n_steps=1, audit_hlo=True, return_img=True,
  )
  rec_8k["bvh_hbm_mb"] = round(hbm_10m / (1024 ** 2), 2)
  rec_8k["total_tiles_8x16"] = (4320 // 8) * (7680 // 16)
  rec_8k["tiles_per_chip"] = rec_8k["total_tiles_8x16"] // 4
  results.append(rec_8k)

  meta_line = (
      f"1.022B Instanced Tris | 4x TPU v5p shard_map | 2-Bounce MC PT (1 spp) | "
      f"{rec_8k['ms']:.1f} ms ({rec_8k['mrays_per_s']:.1f} Mrays/s)"
  )
  save_8k_overview_and_crop_png(img_8k, out_8k_png, meta_line)

  os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
  with open(out_json, "w") as f:
    json.dump({"wave": "2B", "backend": jax.default_backend(), "runs": results}, f, indent=2)
  print("Saved Wave 2B ->", out_json, flush=True)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--mode", required=True, choices=["smoke_gi", "wave2a_gi", "wave2a_scale", "wave2b"])
  ap.add_argument("--out_json", default="out/metrics/v3_w7_wave2.json")
  ap.add_argument("--out_png", default="out/figures/v3_w7_out.png")
  args = ap.parse_args()

  if args.mode == "smoke_gi":
    ds, base_meshes = load_base_scene()
    devs_all = jax.devices()
    mesh_4c = Mesh(np.array(devs_all), ("dp",))
    spec_1m, dev_1m_4c, _, _ = build_3mesh_to_hbm(base_meshes, 16, ds, mesh_4c)
    oxp_1, oxm_1, cxp_1, cxm_1 = make_unique_poses(ds, 1, mesh_4c)
    panels, recs = [], []
    for bnc in [1, 3]:
      for spp in [1, 4, 16, 64]:
        rec, img = bench_pt_cfg(
            f"smoke_1.18M_4c_256_b{bnc}_spp{spp}",
            spec_1m, dev_1m_4c, oxp_1, oxm_1, cxp_1, cxm_1, devs_all,
            256, 256, 1, spp, bnc, instanced=False, n_steps=3,
            audit_hlo=(bnc == 3 and spp == 4), return_img=True,
        )
        recs.append(rec)
        panels.append((f"{bnc}-Bounce | {spp} spp | {rec['samples_per_min']:,.0f}/m", img))
    save_gi_comparison_png(panels, args.out_png)
    with open(args.out_json, "w") as f:
      json.dump({"smoke": recs}, f, indent=2)
  elif args.mode == "wave2a_gi":
    run_wave2a_gi(args.out_json, args.out_png)
  elif args.mode == "wave2a_scale":
    run_wave2a_scale(args.out_json)
  elif args.mode == "wave2b":
    run_wave2b(args.out_json, args.out_png)


if __name__ == "__main__":
  main()
