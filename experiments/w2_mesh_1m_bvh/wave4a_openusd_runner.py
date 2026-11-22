"""Wave 4A Runner — Native OpenUSD (.usd/.usda/.usdc/.usdz) -> TPU Wide-BVH-64 Compiler.

1. Fetches real robotics benchmark OpenUSD assets from xlangai/RoboWild (HuggingFace):
   - RoboWild/scenes/scene01/scene.usda (+ recursive referenced objects/ & textures)
   - RoboWild/robots/franka/panda_robotiq_2f_85/franka_robotiq_2f_85_flattened.usd
   Compiles directly via openusd_loader.compile_openusd_to_wide_bvh() and renders at
   Native 4K (3840x2160) on TPU v5p -> v3_w2_openusd_robowild_4k.png.

2. Exports a real binary .usdc OpenUSD stage (Sponza + UsdGeom.PointInstancer +
   UsdPreviewSurface + UsdLux + UsdGeom.Camera) via pxr, auto-compiles directly
   into Wide-BVH-64 + 2-Level TLAS/BLAS TPU buffers, and renders at Native 4K
   (3840x2160) -> v3_w2_openusd_4k.png + v3_w2_wave4_openusd_tpu.json.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import dataclasses
import json
import os
import posixpath
import re
import time
from typing import Any, Dict, List, Set, Tuple
import urllib.request

import jax
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np
from PIL import Image
import trimesh

from tpu_mjx_render.adapters.openusd_loader import compile_openusd_to_wide_bvh
from tpu_mjx_render.roofline import inspect_hlo_ops
from wave3_500m_10b_8k import (
    _compute_camera_rays_band,
    _pallas_compact_bvh_raycast,
    _pallas_tlas2_compact_instanced_raycast,
)
from wide_bvh_v3 import TILE_H


HF_BASE = "https://huggingface.co/datasets/xlangai/RoboWild/resolve/main/"


def _download_hf_file(rel_path: str, local_root: str) -> str:
  rel_path = rel_path.lstrip("/")
  dst = os.path.join(local_root, rel_path)
  if os.path.exists(dst) and os.path.getsize(dst) > 0:
    return dst
  os.makedirs(os.path.dirname(dst), exist_ok=True)
  url = HF_BASE + rel_path
  req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
  with urllib.request.urlopen(req, timeout=60) as r:
    data = r.read()
  with open(dst, "wb") as f:
    f.write(data)
  return dst


def download_robowild_sparse(asset_root: str) -> Tuple[str, str]:
  """Download RoboWild franka_robotiq_2f_85_flattened.usd and scenes/scene01 (+ recursive USD refs)."""
  franka_rel = "robots/franka/panda_robotiq_2f_85/franka_robotiq_2f_85_flattened.usd"
  franka_local = _download_hf_file(franka_rel, asset_root)
  print(f"Downloaded RoboWild Franka+Robotiq USD: {franka_local} ({os.path.getsize(franka_local)/1e6:.2f} MB)", flush=True)

  # Recursively fetch scene01/scene.usda and all referenced .usd/.usda/.usdc + textures
  visited: Set[str] = set()
  queue: List[str] = ["scenes/scene01/scene.usda"]
  max_files = 160
  while queue and len(visited) < max_files:
    batch = []
    while queue and len(batch) < 16 and len(visited) + len(batch) < max_files:
      rp = posixpath.normpath(queue.pop(0))
      if rp in visited or rp.startswith(".."):
        continue
      visited.add(rp)
      batch.append(rp)
    if not batch:
      break

    def _fetch(rp: str):
      try:
        lp = _download_hf_file(rp, asset_root)
        return rp, lp
      except Exception as e:
        return rp, None

    with ThreadPoolExecutor(max_workers=min(8, len(batch))) as ex:
      res = list(ex.map(_fetch, batch))

    for rp, lp in res:
      if lp is None:
        continue
      if lp.lower().endswith((".usd", ".usda", ".usdc")):
        refs = []
        try:
          from pxr import Sdf
          lyr = Sdf.Layer.FindOrOpen(lp)
          if lyr is not None:
            refs.extend(list(lyr.GetExternalReferences()))
        except Exception:
          pass
        if lp.lower().endswith(".usda"):
          with open(lp, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read()
          refs.extend(re.findall(r"@([^@]+)@", txt))
        parent = posixpath.dirname(rp)
        for ref in refs:
          if not ref or ref.startswith("http") or ref.startswith("/") or ref.startswith("anon:"):
            continue
          cand = posixpath.normpath(posixpath.join(parent, ref))
          if cand not in visited and any(cand.lower().endswith(ext) for ext in (".usd", ".usda", ".usdc", ".png", ".jpg", ".jpeg")):
            queue.append(cand)
  scene01_local = os.path.join(asset_root, "scenes/scene01/scene.usda")
  print(f"Downloaded RoboWild scene01 sparse tree: {len(visited)} files -> {scene01_local}", flush=True)
  return scene01_local, franka_local


def export_sponza_instanced_usdc(usdc_path: str) -> str:
  """Export Sponza + 128 PointInstancer sculptures + UsdPreviewSurface + UsdLux + Camera into binary .usdc."""
  from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt

  os.makedirs(os.path.dirname(usdc_path), exist_ok=True)
  if os.path.exists(usdc_path):
    os.remove(usdc_path)
  stage = Usd.Stage.CreateNew(usdc_path)
  UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
  UsdGeom.SetStageMetersPerUnit(stage, 1.0)
  root = UsdGeom.Xform.Define(stage, "/World")
  stage.SetDefaultPrim(root.GetPrim())

  sc = trimesh.load("/dev/shm/zaychik-tpu-mjx/assets/sponza/Sponza.gltf", force="scene", process=False)
  with open("/dev/shm/zaychik-tpu-mjx/assets/sponza/Sponza.gltf", "r") as f_gltf:
    gltf_meta = json.load(f_gltf)
  prim_mats = [p.get("material", 0) for p in gltf_meta["meshes"][0]["primitives"]]
  arch_palette = [
      (0.78, 0.74, 0.66), (0.64, 0.60, 0.54), (0.46, 0.43, 0.40), (0.72, 0.36, 0.24),
      (0.36, 0.28, 0.20), (0.70, 0.16, 0.16), (0.20, 0.38, 0.58), (0.78, 0.64, 0.32),
  ]
  mat_to_bucket = {
      0: 4, 1: 4, 2: 7, 3: 4, 4: 0, 5: 0, 6: 0, 7: 1, 8: 1, 9: 2, 10: 0,
      11: 1, 12: 0, 13: 1, 14: 5, 15: 6, 16: 5, 17: 6, 18: 5, 19: 6,
      20: 4, 21: 4, 22: 2, 23: 7, 24: 3,
  }

  # Author 8 UsdShade UsdPreviewSurface materials
  mat_scope = UsdGeom.Scope.Define(stage, "/World/Materials")
  usd_mats = []
  for b_idx, rgb in enumerate(arch_palette):
    mat = UsdShade.Material.Define(stage, f"/World/Materials/Mat_{b_idx}")
    sh = UsdShade.Shader.Define(stage, f"/World/Materials/Mat_{b_idx}/PreviewSurface")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.45)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    usd_mats.append(mat)

  bucket_v, bucket_f, bucket_voff = [[] for _ in range(8)], [[] for _ in range(8)], [0] * 8
  for g_idx, node_name in enumerate(sc.graph.nodes_geometry):
    M, g_name = sc.graph.get(node_name)
    geom = sc.geometry[g_name]
    if geom.vertices.size == 0 or geom.faces.size == 0:
      continue
    v = trimesh.transform_points(np.asarray(geom.vertices, dtype=np.float32), M).astype(np.float32)
    vz = np.stack([v[:, 0], -v[:, 2], v[:, 1]], axis=-1)
    f = np.asarray(geom.faces, dtype=np.int32)
    b = mat_to_bucket.get(prim_mats[g_idx] if g_idx < len(prim_mats) else (g_idx % 25), g_idx % 8)
    bucket_v[b].append(vz)
    bucket_f[b].append(f + bucket_voff[b])
    bucket_voff[b] += vz.shape[0]

  for b in range(8):
    if not bucket_v[b]:
      continue
    vb = np.vstack(bucket_v[b])
    fb = np.vstack(bucket_f[b])
    um = UsdGeom.Mesh.Define(stage, f"/World/Sponza_Bucket_{b}")
    um.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vb))
    um.CreateFaceVertexCountsAttr(Vt.IntArray([3] * fb.shape[0]))
    um.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(fb.reshape(-1)))
    UsdShade.MaterialBindingAPI.Apply(um.GetPrim()).Bind(usd_mats[b])

  # Add UsdGeom.PointInstancer placing 128 gilded sculptures along the Sponza colonnade & balconies!
  ds = np.load("/dev/shm/zaychik-tpu-mjx/out/assets/blender_real_scene.npz")
  sv = ds["mesh_0_verts"].astype(np.float32) * 3.0
  sf = ds["mesh_0_faces"].astype(np.int32)
  proto_root = UsdGeom.Xform.Define(stage, "/World/Prototypes")
  proto_mesh = UsdGeom.Mesh.Define(stage, "/World/Prototypes/HeroSculpture")
  proto_mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(sv))
  proto_mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * sf.shape[0]))
  proto_mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(sf.reshape(-1)))
  UsdShade.MaterialBindingAPI.Apply(proto_mesh.GetPrim()).Bind(usd_mats[7])

  pi = UsdGeom.PointInstancer.Define(stage, "/World/ColonnadeSculptures")
  pi.CreatePrototypesRel().AddTarget(proto_mesh.GetPath())
  positions = []
  orients = []
  proto_ids = []
  # Two rows on ground floor + two rows on upper gallery
  for ix in range(16):
    x = -8.5 + 1.15 * ix
    for iy, y in enumerate((-1.2, 1.7)):
      for iz, z in enumerate((0.42, 4.42)):
        for j in range(2):
          dx = 0.28 * (j - 0.5)
          positions.append(Gf.Vec3f(float(x + dx), float(y), float(z)))
          ang = 0.35 * ix + 1.57 * iy
          orients.append(Gf.Quath(float(np.cos(0.5 * ang)), 0.0, 0.0, float(np.sin(0.5 * ang))))
          proto_ids.append(0)
  pi.CreateProtoIndicesAttr(Vt.IntArray(proto_ids))
  pi.CreatePositionsAttr(Vt.Vec3fArray(positions))
  pi.CreateOrientationsAttr(Vt.QuathArray(orients))

  # Author UsdLux lights + UsdGeom.Camera
  dl = UsdLux.DistantLight.Define(stage, "/World/SunLight")
  dl.CreateColorAttr(Gf.Vec3f(0.92, 0.88, 0.82))
  dl.AddRotateXYZOp().Set(Gf.Vec3f(25.0, -15.0, 0.0))
  dome = UsdLux.DomeLight.Define(stage, "/World/SkyDome")
  dome.CreateColorAttr(Gf.Vec3f(0.75, 0.82, 0.95))

  cam = UsdGeom.Camera.Define(stage, "/World/HeroCam")
  cam.CreateFocalLengthAttr(21.0)
  cam.CreateVerticalApertureAttr(20.25)
  cam.CreateClippingRangeAttr(Gf.Vec2f(0.1, 100.0))
  # Position at (9.0, 0.25, 2.4) looking toward (-8.0, 0.25, 2.8)
  cpos = np.array([9.0, 0.25, 2.4], dtype=np.float64)
  ctgt = np.array([-8.0, 0.25, 2.8], dtype=np.float64)
  fwd = ctgt - cpos; fwd /= np.linalg.norm(fwd)
  right = np.cross(fwd, np.array([0.0, 0.0, 1.0])); right /= np.linalg.norm(right)
  up = np.cross(right, fwd)
  m4 = Gf.Matrix4d(
      right[0], right[1], right[2], 0.0,
      up[0], up[1], up[2], 0.0,
      -fwd[0], -fwd[1], -fwd[2], 0.0,
      cpos[0], cpos[1], cpos[2], 1.0,
  )
  cam.AddTransformOp().Set(m4)

  stage.GetRootLayer().Save()
  print(f"Exported binary .usdc OpenUSD stage: {usdc_path} ({os.path.getsize(usdc_path)/1e6:.2f} MB)", flush=True)
  return usdc_path


def make_openusd_4chip_renderer(compiled: Dict[str, Any], devices: List[jax.Device], H: int, W: int):
  """4-chip spatial-sharded OpenUSD renderer supporting BOTH unique meshes AND PointInstancer TLAS/BLAS!"""
  n_chips = len(devices)
  assert H % (n_chips * TILE_H) == 0
  band_h = H // n_chips
  mesh = Mesh(np.array(devices), ("spatial",))

  u_spec = compiled["unique_spec"]
  u_devs = compiled["unique_dev_arrays"]
  i_spec = compiled["inst_spec"]
  i_devs = compiled["inst_dev_arrays"]
  base_spec = u_spec if u_spec is not None else i_spec
  spec_band = dataclasses.replace(base_spec, height=band_h, width=W)

  def _shard_fn(u_da, u_oxp, u_oxm, i_da, i_xp, i_xm, cxp, cxm):
    bidx = jax.lax.axis_index("spatial")
    ray_orig, ray_dir, neg_z = _compute_camera_rays_band(spec_band, cxp, cxm, H, W, bidx, band_h)
    B, R, _ = ray_orig.shape
    max_dist = jnp.broadcast_to(((spec_band.zfar - spec_band.znear) / neg_z)[None, :], (B, R))

    best_d = max_dist
    best_hit = jnp.zeros((B, R), dtype=jnp.bool_)
    best_nrm = jnp.zeros((B, R, 3), dtype=jnp.float32)
    best_alb = jnp.zeros((B, R, 3), dtype=jnp.float32)

    if u_spec is not None:
      for m_idx, it in enumerate(u_spec.meshes):
        l2_r, l2_a, l0_a, leaf_b = u_da[m_idx]
        p = u_oxp[:, m_idx, :]
        rmat = u_oxm[:, m_idx, :, :]
        ro_loc = jnp.einsum("bji,brj->bri", rmat, ray_orig - p[:, None, :], precision=jax.lax.Precision.HIGHEST)
        rd_loc = jnp.einsum("bji,brj->bri", rmat, ray_dir, precision=jax.lax.Precision.HIGHEST)
        d_m, n_loc_m, _ = _pallas_compact_bvh_raycast(l2_r, l2_a, l0_a, leaf_b, ro_loc, rd_loc, best_d, band_h, W, any_hit=False, interpret=False)
        imp = d_m < best_d
        n_w_m = jnp.einsum("bij,brj->bri", rmat, n_loc_m, precision=jax.lax.Precision.HIGHEST)
        best_d = jnp.where(imp, d_m, best_d)
        best_hit = best_hit | imp
        best_nrm = jnp.where(imp[..., None], n_w_m, best_nrm)
        alb_m = jnp.asarray(it.rgba[:3], dtype=jnp.float32)[None, None, :]
        best_alb = jnp.where(imp[..., None], alb_m, best_alb)

    if i_spec is not None:
      for m_idx, it in enumerate(i_spec.meshes):
        l2_r, l2_a, l0_a, leaf_b = i_da[m_idx]
        d_m, n_w_m = _pallas_tlas2_compact_instanced_raycast(
            it, l2_r, l2_a, l0_a, leaf_b,
            ray_orig, ray_dir, best_d, i_xp[m_idx], i_xm[m_idx],
            band_h, W, tlas_group_size=16, tlas_top_groups=4, max_active_per_tile=16, any_hit=False, interpret=False,
        )
        imp = d_m < best_d
        best_d = jnp.where(imp, d_m, best_d)
        best_hit = best_hit | imp
        best_nrm = jnp.where(imp[..., None], n_w_m, best_nrm)
        alb_m = jnp.asarray(it.rgba[:3], dtype=jnp.float32)[None, None, :]
        best_alb = jnp.where(imp[..., None], alb_m, best_alb)

    vdir = -ray_dir
    best_nrm = jnp.where(best_hit[..., None], best_nrm, 0.0)
    flip = jnp.sum(best_nrm * vdir, axis=-1, keepdims=True) < 0.0
    best_nrm = jnp.where(flip, -best_nrm, best_nrm)

    hit_pos = ray_orig + jnp.where(best_hit, best_d, 0.0)[..., None] * ray_dir
    l_pos = jnp.asarray(spec_band.light_pos, dtype=jnp.float32)
    delta_l = l_pos[None, None, :] - hit_pos
    dist_l = jnp.linalg.norm(delta_l, axis=-1)
    L = delta_l / jnp.maximum(dist_l[..., None], 1e-12)
    ndotl = jnp.maximum(0.0, jnp.sum(best_nrm * L, axis=-1))

    vis = jnp.ones((B, R), dtype=jnp.float32)
    if spec_band.use_shadows:
      sh_o = hit_pos + best_nrm * 2.0e-3
      sh_lim = jnp.where(best_hit & (ndotl > 0.0), jnp.maximum(dist_l - 1e-3, 0.0), 0.0)
      occ = jnp.zeros((B, R), dtype=jnp.bool_)
      if u_spec is not None:
        for m_idx, _it in enumerate(u_spec.meshes):
          l2_r, l2_a, l0_a, leaf_b = u_da[m_idx]
          p = u_oxp[:, m_idx, :]
          rmat = u_oxm[:, m_idx, :, :]
          ro_l = jnp.einsum("bji,brj->bri", rmat, sh_o - p[:, None, :], precision=jax.lax.Precision.HIGHEST)
          rd_l = jnp.einsum("bji,brj->bri", rmat, L, precision=jax.lax.Precision.HIGHEST)
          sh_lim_rem = jnp.where(occ, 0.0, sh_lim)
          d_sh, _, _ = _pallas_compact_bvh_raycast(l2_r, l2_a, l0_a, leaf_b, ro_l, rd_l, sh_lim_rem, band_h, W, any_hit=True, interpret=False)
          occ = occ | (d_sh < sh_lim_rem)
      if i_spec is not None:
        for m_idx, it in enumerate(i_spec.meshes):
          l2_r, l2_a, l0_a, leaf_b = i_da[m_idx]
          sh_lim_rem = jnp.where(occ, 0.0, sh_lim)
          d_sh, _ = _pallas_tlas2_compact_instanced_raycast(
              it, l2_r, l2_a, l0_a, leaf_b,
              sh_o, jnp.broadcast_to(L, ray_orig.shape), sh_lim_rem,
              i_xp[m_idx], i_xm[m_idx],
              band_h, W, tlas_group_size=16, tlas_top_groups=4, max_active_per_tile=16, any_hit=True, interpret=False,
          )
          occ = occ | (d_sh < sh_lim_rem)
      vis = jnp.where(occ, float(spec_band.shadow_light_fraction), 1.0)

    l_amb = jnp.asarray(spec_band.light_ambient, dtype=jnp.float32)
    l_diff = jnp.asarray(spec_band.light_diffuse, dtype=jnp.float32)
    l_spec = jnp.asarray(spec_band.light_specular, dtype=jnp.float32)
    H_vec = (L + vdir) / jnp.maximum(jnp.linalg.norm(L + vdir, axis=-1, keepdims=True), 1e-12)
    spec_w = (jnp.maximum(0.0, jnp.sum(best_nrm * H_vec, axis=-1)) ** 32.0) * 0.25 * vis
    direct_rgb = jnp.clip(
        best_alb * (l_amb[None, None, :] + l_diff[None, None, :] * (ndotl * vis)[..., None]) + l_spec[None, None, :] * spec_w[..., None],
        0.0, 1.0,
    )
    sky_t = jnp.clip(0.5 * (ray_dir[..., 2] + 1.0), 0.0, 1.0)[..., None]
    sky_rgb = (
        sky_t * jnp.asarray(spec_band.sky_rgb1, dtype=jnp.float32)[None, None, :]
        + (1.0 - sky_t) * jnp.asarray(spec_band.sky_rgb2, dtype=jnp.float32)[None, None, :]
    )
    return jnp.where(best_hit[..., None], direct_rgb, sky_rgb).reshape(B, band_h, W, 3)

  sharded = shard_map(
      _shard_fn,
      mesh=mesh,
      in_specs=(P(), P(), P(), P(), P(), P(), P(), P()),
      out_specs=P(None, "spatial", None, None),
      check_rep=False,
  )
  fn = jax.jit(lambda: sharded(
      u_devs, compiled["obj_xpos"], compiled["obj_xmat"],
      i_devs, compiled["inst_xpos"], compiled["inst_xmat"],
      compiled["cam_xpos"], compiled["cam_xmat"],
  ))
  return fn


def _bench_stage(name: str, stage_path: str, devs: List[jax.Device], png_4k: str, subdivide_k: int = 1) -> Dict[str, Any]:
  t0 = time.perf_counter()
  compiled = compile_openusd_to_wide_bvh(stage_path, height=2160, width=3840, subdivide_k=subdivide_k, devices=devs)
  compile_host_s = time.perf_counter() - t0
  st = compiled["stage"]
  print(f"Compiled OpenUSD {name}: unique_meshes={len(st.unique_meshes)} ({compiled['unique_tris']:,} tris), inst_groups={len(st.instanced_groups)} ({compiled['inst_effective_tris']:,} eff tris), HBM={compiled['hbm_bytes']/1e6:.2f} MB in {compile_host_s:.2f}s", flush=True)

  fn_4k = make_openusd_4chip_renderer(compiled, devs, 2160, 3840)
  htxt = fn_4k.lower().as_text()
  hlo = inspect_hlo_ops(htxt)
  hlo["send_recv_ops"] = len(re.findall(r"\bsend\(|\brecv\(", htxt, re.I))
  assert hlo["host_callbacks"] == 0 and hlo["send_recv_ops"] == 0 and hlo["gather_ops"] == 0, hlo

  tc0 = time.perf_counter()
  rgb = fn_4k()
  jax.block_until_ready(rgb)
  xla_compile_s = time.perf_counter() - tc0

  img = Image.fromarray(np.clip(np.asarray(rgb[0]) * 255.0 + 0.5, 0, 255).astype(np.uint8))
  img.save(png_4k)
  print(f"Saved Native 4K OpenUSD render ({img.size}) to {png_4k}", flush=True)

  n_steps = 3
  t1 = time.perf_counter()
  for _ in range(n_steps):
    rgb = fn_4k()
  jax.block_until_ready(rgb)
  ms = (time.perf_counter() - t1) * 1e3 / n_steps

  return {
      "name": name,
      "stage_path": stage_path,
      "up_axis": st.up_axis,
      "meters_per_unit": st.meters_per_unit,
      "unique_meshes": len(st.unique_meshes),
      "unique_tris": compiled["unique_tris"],
      "inst_groups": len(st.instanced_groups),
      "inst_blas_tris": compiled["inst_blas_tris"],
      "inst_effective_tris": compiled["inst_effective_tris"],
      "hbm_mb": round(compiled["hbm_bytes"] / (1024 ** 2), 2),
      "usd_to_bvh_host_s": round(compile_host_s, 2),
      "xla_compile_s": round(xla_compile_s, 2),
      "res": "2160x3840",
      "n_chips": len(devs),
      "ms": round(ms, 2),
      "fps": round(1e3 / ms, 2),
      "mrays_per_s": round((2160 * 3840 * 2.0) / (ms * 1e-3) / 1e6, 2),
      "hlo": hlo,
      "png_4k": png_4k,
  }


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", default="/dev/shm/zaychik-tpu-mjx-w2/v3_w2_wave4_openusd_tpu.json")
  ap.add_argument("--png-usdc", default="/dev/shm/zaychik-tpu-mjx-w2/v3_w2_openusd_4k.png")
  ap.add_argument("--png-robowild", default="/dev/shm/zaychik-tpu-mjx-w2/v3_w2_openusd_robowild_4k.png")
  ap.add_argument("--png-franka", default="/dev/shm/zaychik-tpu-mjx-w2/v3_w2_openusd_franka_4k.png")
  args = ap.parse_args()

  devs = jax.devices()
  assert len(devs) == 4
  asset_root = "/dev/shm/zaychik-tpu-mjx/assets/robowild"
  scene01_path, franka_path = download_robowild_sparse(asset_root)
  usdc_path = "/dev/shm/zaychik-tpu-mjx/assets/sponza/sponza_instanced.usdc"
  export_sponza_instanced_usdc(usdc_path)

  stages = []
  stages.append(_bench_stage("Sponza_PointInstancer_Binary_USDC", usdc_path, devs, args.png_usdc, subdivide_k=2))
  stages.append(_bench_stage("RoboWild_Franka_Robotiq_2F85_USD", franka_path, devs, args.png_franka, subdivide_k=1))
  stages.append(_bench_stage("RoboWild_Scene01_IsaacLab_USDA", scene01_path, devs, args.png_robowild, subdivide_k=1))

  out = {
      "backend": jax.default_backend(),
      "jax_version": jax.__version__,
      "device_kind": devs[0].device_kind,
      "n_chips": len(devs),
      "stages": stages,
  }
  with open(args.out, "w") as f:
    json.dump(out, f, indent=2)
  print(f"Saved Wave 4A results to {args.out}", flush=True)


if __name__ == "__main__":
  main()
