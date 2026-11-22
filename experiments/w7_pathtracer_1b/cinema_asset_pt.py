"""Priority #3: Cinema Blender Film Asset Upgrade on TPU v5p (Pixel-Exact UV Textures + Multi-Light NEE + TPU Denoiser).

Supports any `.blend` file (e.g. Blender `lone-monk_cycles_and_exposure-node_demo.blend` or OpenData `classroom`) via:
  1. Blender 5.3.0 Depsgraph + Cycles Shader Graph Exporter (`use_smooth`-aware normals,
     512x512 packed `tex_atlas` linear RGB + roughness, per-triangle UVs, multi-light rig,
     and clean denoised Cycles 5.3.0 GT reference).
  2. TPU Mosaic Wide-BVH-64 kernel (`uv_bvh_raycast_tpu`) returning exact `(dist, nrm, bary_u, bary_v, tri_idx, mat_code)`
     with pure-JAX bilinear `tex_atlas` sampling, multi-bounce GGX + dielectric glass + multi-light NEE,
     TPU Joint A-Trous Wavelet Denoiser, and AgX Filmic Tonemapping (`host_callbacks == 0`).
"""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
import math
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ==============================================================================
# Part 1: Blender 5.3.0 Headless Cinema Exporter (runs inside `blender -b`)
# ==============================================================================

def _blender_export_main():
  import bpy
  sys.path.insert(0, "/dev/shm/zaychik-tpu-mjx-w7/tpu_mjx_render/adapters")
  from blend_compiler import (
      _ShaderGraphWalker,
      _extract_camera,
      _extract_lights,
      _extract_world_sky,
      _resize_bilinear,
  )

  argv = sys.argv
  argv = argv[argv.index("--") + 1 :]
  ap = argparse.ArgumentParser()
  ap.add_argument("--out_npz", required=True)
  ap.add_argument("--gt_png", required=True)
  ap.add_argument("--gt_w", type=int, default=960)
  ap.add_argument("--gt_h", type=int, default=540)
  ap.add_argument("--gt_spp", type=int, default=96)
  ap.add_argument("--tex_res", type=int, default=512)
  ap.add_argument("--skip_gt", action="store_true")
  args = ap.parse_args(argv)

  scene = bpy.context.scene
  depsgraph = bpy.context.evaluated_depsgraph_get()

  class CinemaShaderWalker(_ShaderGraphWalker):
    def _resolve_socket_link(self, sock, group_stack):
      cur_sock = sock
      cur_stack = list(group_stack)
      for _ in range(24):
        res, cur_stack = super()._resolve_socket_link(cur_sock, cur_stack)
        if res is None or not isinstance(res, tuple):
          return res, cur_stack
        fn, fs = res
        if fn.type == "REROUTE" and len(fn.inputs) > 0 and fn.inputs[0].is_linked:
          cur_sock = fn.inputs[0]
          continue
        return (fn, fs), cur_stack
      return None, cur_stack

    def _trace_uv_transform(self, vec_sock, depth=0):
      cur = vec_sock
      for _ in range(16):
        if cur is None or not cur.is_linked:
          return (1.0, 1.0, 0.0, 0.0), None
        fn = cur.links[0].from_node
        if fn.type == "REROUTE" and len(fn.inputs) > 0 and fn.inputs[0].is_linked:
          cur = fn.inputs[0]
          continue
        return super()._trace_uv_transform(cur, depth)
      return (1.0, 1.0, 0.0, 0.0), None

    def _find_first_tex_uv_meta(self, sock, depth=0):
      if sock is None or not sock.is_linked or depth > 14:
        return (1.0, 1.0, 0.0, 0.0), None
      fn = sock.links[0].from_node
      if fn.type == "REROUTE" and len(fn.inputs) > 0:
        return self._find_first_tex_uv_meta(fn.inputs[0], depth + 1)
      return super()._find_first_tex_uv_meta(sock, depth)

    def compile_material(self, mat, mat_id):
      cm, ta, tr = super().compile_material(mat, mat_id)
      if mat is not None and "glass" in mat.name.lower():
        cm.mat_code = 2.0
        cm.base_color = (0.94, 0.98, 1.0)
        cm.roughness = 0.03
      return cm, ta, tr

  walker = CinemaShaderWalker(tex_res=args.tex_res)

  mat_map: Dict[int, Any] = {}
  mat_list: List[Any] = []
  tex_list: List[np.ndarray] = []

  def _register_mat(mat):
    ptr = mat.as_pointer() if mat is not None else -1
    if ptr in mat_map:
      return mat_map[ptr]
    mid = len(mat_list)
    cm, tex_alb, tex_rgh = walker.compile_material(mat, mid)
    if tex_alb is not None:
      tid = len(tex_list)
      rgh_grid = tex_rgh if tex_rgh is not None else np.full((args.tex_res, args.tex_res), cm.roughness, dtype=np.float32)
      if tex_alb.shape[:2] != (args.tex_res, args.tex_res):
        tex_alb = _resize_bilinear(tex_alb, args.tex_res, args.tex_res)
      if rgh_grid.shape[:2] != (args.tex_res, args.tex_res):
        rgh_grid = _resize_bilinear(rgh_grid, args.tex_res, args.tex_res)
      tex_list.append(np.concatenate([tex_alb, rgh_grid[..., None]], axis=-1).astype(np.float32))
      cm.tex_id = tid
    mat_map[ptr] = cm
    mat_list.append(cm)
    return cm

  all_v, all_f, all_n = [], [], []
  all_uv, all_tid, all_alb, all_rgh, all_mc = [], [], [], [], []
  v_off = 0

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

    verts_loc = np.empty((n_verts, 3), dtype=np.float32)
    me.vertices.foreach_get("co", verts_loc.ravel())
    mat_w = np.array(inst.matrix_world, dtype=np.float32)
    R = mat_w[:3, :3]
    t = mat_w[:3, 3]
    verts_w_orig = verts_loc @ R.T + t[None, :]
    det = float(np.linalg.det(R))
    R_n = np.linalg.inv(R).T if abs(det) > 1e-8 else R

    tri_v = np.empty((n_tris * 3,), dtype=np.int32)
    me.loop_triangles.foreach_get("vertices", tri_v)
    faces_orig = tri_v.reshape(n_tris, 3)

    vnorm_loc = np.empty((n_verts, 3), dtype=np.float32)
    me.vertices.foreach_get("normal", vnorm_loc.ravel())
    vnorm_w_orig = vnorm_loc @ R_n.T
    vnorm_w_orig = vnorm_w_orig / np.maximum(np.linalg.norm(vnorm_w_orig, axis=-1, keepdims=True), 1e-12)

    tri_mat_idx = np.empty((n_tris,), dtype=np.int32)
    me.loop_triangles.foreach_get("material_index", tri_mat_idx)

    tri_loops = np.empty((n_tris * 3,), dtype=np.int32)
    me.loop_triangles.foreach_get("loops", tri_loops)
    tri_loops = tri_loops.reshape(n_tris, 3)

    tri_smooth = np.empty((n_tris,), dtype=bool)
    me.loop_triangles.foreach_get("use_smooth", tri_smooth)

    if det < 0.0:
      faces_orig = faces_orig[:, [0, 2, 1]]
      tri_loops = tri_loops[:, [0, 2, 1]]

    uv_layer = me.uv_layers.active.data if (len(me.uv_layers) > 0 and me.uv_layers.active is not None) else None
    if uv_layer is not None:
      n_loops = len(me.loops)
      uvs_flat = np.empty((n_loops * 2,), dtype=np.float32)
      uv_layer.foreach_get("uv", uvs_flat)
      uvs_flat = uvs_flat.reshape(n_loops, 2)
      tri_uv = uvs_flat[tri_loops]
    else:
      tri_uv = np.zeros((n_tris, 3, 2), dtype=np.float32)

    # Split flat vs smooth triangles accurately so flat surfaces never pillow-shade!
    if np.all(tri_smooth):
      v_use = verts_w_orig
      n_use = vnorm_w_orig
      f_use = faces_orig
      nv_use = n_verts
    elif not np.any(tri_smooth):
      v_use = verts_w_orig[faces_orig.ravel()]
      e1 = v_use[1::3] - v_use[0::3]
      e2 = v_use[2::3] - v_use[0::3]
      fn = np.cross(e1, e2)
      fn = fn / np.maximum(np.linalg.norm(fn, axis=-1, keepdims=True), 1e-12)
      n_use = np.repeat(fn, 3, axis=0).astype(np.float32)
      nv_use = n_tris * 3
      f_use = np.arange(nv_use, dtype=np.int32).reshape(n_tris, 3)
    else:
      # Mixed smooth & flat mesh: duplicate per triangle to mix exact flat face normals & smooth normals
      v_use = verts_w_orig[faces_orig.ravel()]
      n_smooth_corners = vnorm_w_orig[faces_orig.ravel()]
      e1 = v_use[1::3] - v_use[0::3]
      e2 = v_use[2::3] - v_use[0::3]
      fn = np.cross(e1, e2)
      fn = fn / np.maximum(np.linalg.norm(fn, axis=-1, keepdims=True), 1e-12)
      n_flat_corners = np.repeat(fn, 3, axis=0).astype(np.float32)
      smooth_corner_mask = np.repeat(tri_smooth, 3)[:, None]
      n_use = np.where(smooth_corner_mask, n_smooth_corners, n_flat_corners).astype(np.float32)
      nv_use = n_tris * 3
      f_use = np.arange(nv_use, dtype=np.int32).reshape(n_tris, 3)

    vcol_rgb = None
    if len(me.color_attributes) > 0:
      ca = me.color_attributes.active_color
      if ca is not None and ca.domain == "CORNER":
        c_flat = np.empty((len(me.loops) * 4,), dtype=np.float32)
        ca.data.foreach_get("color", c_flat)
        c_flat = c_flat.reshape(len(me.loops), 4)[..., :3]
        vcol_rgb = np.mean(c_flat[tri_loops], axis=1)

    mats = list(me.materials) if len(me.materials) > 0 else [None]
    tid_f = np.full((n_tris,), -1, dtype=np.int32)
    alb_f = np.full((n_tris, 3), 0.70, dtype=np.float32)
    rgh_f = np.full((n_tris,), 0.50, dtype=np.float32)
    mc_f = np.full((n_tris,), 0.0, dtype=np.float32)

    for slot_i, mat in enumerate(mats):
      mask = (tri_mat_idx == slot_i) if len(mats) > 1 else np.ones((n_tris,), dtype=bool)
      if not np.any(mask):
        continue
      cm = _register_mat(mat)
      rgh_f[mask] = cm.roughness
      mc_f[mask] = cm.mat_code
      sx, sy, ox, oy = cm.uv_scale_offset
      if (sx, sy, ox, oy) != (1.0, 1.0, 0.0, 0.0):
        tri_uv[mask, :, 0] = tri_uv[mask, :, 0] * sx + ox
        tri_uv[mask, :, 1] = tri_uv[mask, :, 1] * sy + oy

      if cm.tex_id >= 0 and uv_layer is not None:
        tid_f[mask] = cm.tex_id
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

    all_v.append(v_use.astype(np.float32))
    all_f.append((f_use + v_off).astype(np.int32))
    all_n.append(n_use.astype(np.float32))
    all_uv.append(tri_uv.astype(np.float32))
    all_tid.append(tid_f)
    all_alb.append(alb_f)
    all_rgh.append(rgh_f)
    all_mc.append(mc_f)
    v_off += nv_use
    obj.to_mesh_clear()

  verts = np.concatenate(all_v, axis=0)
  faces = np.concatenate(all_f, axis=0)
  vnorm = np.concatenate(all_n, axis=0)
  tri_uvs = np.concatenate(all_uv, axis=0)
  tri_tex_id = np.concatenate(all_tid, axis=0)
  tri_albedo = np.concatenate(all_alb, axis=0)
  tri_roughness = np.concatenate(all_rgh, axis=0)
  tri_mat_code = np.concatenate(all_mc, axis=0)

  tex_atlas = np.stack(tex_list, axis=0).astype(np.float16) if tex_list else np.zeros((0, args.tex_res, args.tex_res, 4), np.float16)
  cam_pos, cam_xmat, cam_fovy, cam_shift, znear, zfar, native_res = _extract_camera(scene)
  lights = _extract_lights(scene)
  sky_z, sky_h, sky_g = _extract_world_sky(scene)
  view_exp = float(getattr(scene.view_settings, "exposure", 0.0))

  # Extract Nishita / Sky Texture sun if no explicit SUN light exists
  if not any(l["type"] == "SUN" for l in lights) and getattr(scene, "world", None) and scene.world.use_nodes:
    for nd in scene.world.node_tree.nodes:
      if nd.type == "TEX_SKY":
        elev = float(getattr(nd, "sun_elevation", 0.7))
        rot = float(getattr(nd, "sun_rotation", 0.0))
        s_dir = np.array([
            math.sin(rot) * math.cos(elev),
            math.cos(rot) * math.cos(elev),
            math.sin(elev),
        ], dtype=np.float32)
        s_dir = s_dir / max(float(np.linalg.norm(s_dir)), 1e-12)
        lights.append({
            "name": "Nishita_Sky_Sun",
            "type": "SUN",
            "pos": [0.0, 0.0, 10.0],
            "dir": (-s_dir).tolist(),
            "color": [1.0, 0.92, 0.80],
            "energy": 10.5 * float(getattr(nd, "sun_intensity", 1.0)),
            "size": 0.5,
        })
        sky_z = (0.42, 0.68, 1.12)
        sky_h = (0.76, 0.86, 1.04)
        sky_g = (0.16, 0.14, 0.12)
        break
      if nd.type == "TEX_ENVIRONMENT" and nd.image is not None:
        try:
          iw, ih = nd.image.size[0], nd.image.size[1]
          if iw > 0 and ih > 0 and len(nd.image.pixels) == iw * ih * 4:
            px_f = np.empty(iw * ih * 4, dtype=np.float32)
            nd.image.pixels.foreach_get(px_f)
            px = px_f.reshape(ih, iw, 4)[..., :3]
            lum = 0.2126 * px[..., 0] + 0.7152 * px[..., 1] + 0.0722 * px[..., 2]
            py, px_x = np.unravel_index(int(np.argmax(lum)), lum.shape)
            phi = (float(px_x) / float(iw) - 0.5) * 2.0 * math.pi
            theta = (1.0 - float(py) / float(ih)) * math.pi
            s_dir = np.array([math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta)], np.float32)
            peak_c = np.clip(px[py, px_x], 0.5, 40.0)
            lights.append({
                "name": "HDRI_Peak_Sun",
                "type": "SUN",
                "pos": [0.0, 0.0, 10.0],
                "dir": (-s_dir).tolist(),
                "color": (peak_c / max(float(np.max(peak_c)), 1e-4)).tolist(),
                "energy": float(min(max(np.max(peak_c), 2.0), 16.0)),
                "size": 0.5,
            })
            break
        except Exception:
          pass

  np.savez(
      args.out_npz,
      verts=verts,
      faces=faces,
      vnorm=vnorm,
      tri_uvs=tri_uvs.astype(np.float32),
      tri_tex_id=tri_tex_id,
      tri_albedo=tri_albedo,
      tri_roughness=tri_roughness,
      tri_mat_code=tri_mat_code,
      tex_atlas=tex_atlas,
      cam_pos=cam_pos,
      cam_xmat=cam_xmat,
      cam_fovy=np.float32(cam_fovy),
      cam_shift=np.array(cam_shift, dtype=np.float32),
      cam_znear=np.float32(znear),
      cam_zfar=np.float32(max(zfar, 120.0)),
      native_res=np.array(native_res, dtype=np.int32),
      lights_json=np.bytes_(json.dumps(lights)),
      sky_zenith=np.array(sky_z, dtype=np.float32),
      sky_horizon=np.array(sky_h, dtype=np.float32),
      sky_ground=np.array(sky_g, dtype=np.float32),
      blender_exposure=np.float32(view_exp),
  )
  print(
      f"CINEMA EXPORT {args.out_npz}: tris={len(faces):,} tex_atlas={tex_atlas.shape} lights={len(lights)} fovy={cam_fovy:.2f} exp={view_exp:.2f}",
      flush=True,
  )

  # Render clean denoised Cycles 5.3.0 GT reference (skip if --skip_gt and file exists)
  if not (args.skip_gt and os.path.exists(args.gt_png)):
    scene.render.use_compositing = False
    if hasattr(scene.render, "compositor_device"):
      scene.render.compositor_device = "CPU"
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = args.gt_spp
    scene.cycles.use_denoising = True
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.015
    scene.render.resolution_x = args.gt_w
    scene.render.resolution_y = args.gt_h
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = args.gt_png
    t0 = time.perf_counter()
    bpy.ops.render.render(write_still=True)
    print(f"CYCLES GT {args.gt_png} ({args.gt_w}x{args.gt_h} @ {args.gt_spp} spp) in {time.perf_counter()-t0:.2f}s", flush=True)


# ==============================================================================
# Part 2: TPU Wide-BVH-64 UV-Exact Raycast + Bilinear Texture Sampling + Multi-Light PT
# ==============================================================================

def _tpu_main():
  import jax
  import jax.experimental.pallas as pl
  import jax.experimental.pallas.tpu as pltpu
  from jax.experimental.shard_map import shard_map
  import jax.numpy as jnp
  from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
  from PIL import Image, ImageDraw

  from bvh_pbr_pt import (
      _agx_tonemap_srgb,
      _build_onb,
      _eval_pbr_brdf,
      _sample_cosine_hemisphere,
      _sample_ggx_half,
      _smith_g1,
      audit_hlo_zero_copy,
  )
  from physical_fx_pt import _tpu_joint_atrous_denoise
  from run_opendata_pt import compute_psnr, compute_ssim
  from tpu_mjx_render.primitives import MJ_MAXVAL
  from wide_bvh_v3 import (
      BLOCK_R,
      BRANCH_FACTOR,
      LEAF_SIZE,
      _compute_smooth_vertex_normals,
      _pack_rays_tiled,
      _spatial_median_order,
  )

  ap = argparse.ArgumentParser()
  ap.add_argument("--npz", required=True)
  ap.add_argument("--gt_png", required=True)
  ap.add_argument("--scene_label", default="cinema_asset")
  ap.add_argument("--spp", type=int, default=256)
  ap.add_argument("--max_bounces", type=int, default=4)
  ap.add_argument("--exposure", type=float, default=None)
  ap.add_argument("--bvh_cache", default=None)
  ap.add_argument("--out_json", default="out/metrics/v3_w7_cinema_pt.json")
  ap.add_argument("--out_png", default="out/figures/v3_w7_cinema_pt_showcase.png")
  args = ap.parse_args()

  ds = np.load(args.npz, allow_pickle=False)
  verts = ds["verts"].astype(np.float32)
  faces = ds["faces"].astype(np.int32)
  vnorm = ds["vnorm"].astype(np.float32)
  tri_uvs = ds["tri_uvs"].astype(np.float32)
  tri_tex_id = ds["tri_tex_id"].astype(np.int32)
  tri_alb = ds["tri_albedo"].astype(np.float32)
  tri_rgh = np.clip(ds["tri_roughness"].astype(np.float32), 0.02, 0.99)
  tri_mc = ds["tri_mat_code"].astype(np.float32)
  tex_atlas_np = ds["tex_atlas"].astype(np.float32)
  if tex_atlas_np.shape[0] == 0:
    tex_atlas_np = np.ones((1, 4, 4, 4), dtype=np.float32)

  cam_pos = ds["cam_pos"][None, :].astype(np.float32)
  cam_xmat = ds["cam_xmat"][None, :, :].astype(np.float32)
  cam_fovy = float(ds["cam_fovy"])
  cam_znear = float(ds["cam_znear"]) if "cam_znear" in ds else 0.05
  cam_zfar = float(ds["cam_zfar"]) if "cam_zfar" in ds else 120.0
  lights_list = json.loads(bytes(ds["lights_json"]).decode("utf-8"))
  sky_z = np.asarray(ds["sky_zenith"], dtype=np.float32)
  sky_h = np.asarray(ds["sky_horizon"], dtype=np.float32)
  sky_g = np.asarray(ds["sky_ground"], dtype=np.float32)
  bl_exp = float(ds["blender_exposure"])

  # Build or load cached BVH ordering & leaf_bank
  if args.bvh_cache and os.path.exists(args.bvh_cache):
    bc = np.load(args.bvh_cache, allow_pickle=False)
    order = bc["order"]
    faces = faces[order]
    tri_uvs = tri_uvs[order]
    tri_tex_id = tri_tex_id[order]
    tri_alb = tri_alb[order]
    tri_rgh = tri_rgh[order]
    tri_mc = tri_mc[order]
    F = int(faces.shape[0])
    l2_roots_aabb = bc["l2_roots_aabb"]
    l2_aabbs = bc["l2_aabbs"]
    l0_aabbs = bc["l0_aabbs"]
    leaf_bank = bc["leaf_bank"]
  else:
    centroids = (verts[faces[:, 0]] + verts[faces[:, 1]] + verts[faces[:, 2]]) * (1.0 / 3.0)
    order = _spatial_median_order(centroids, LEAF_SIZE, BRANCH_FACTOR, stop_at_l1=True)
    del centroids

    faces = faces[order]
    tri_uvs = tri_uvs[order]
    tri_tex_id = tri_tex_id[order]
    tri_alb = tri_alb[order]
    tri_rgh = tri_rgh[order]
    tri_mc = tri_mc[order]

    F = int(faces.shape[0])
    n_leaves = max(1, (F + LEAF_SIZE - 1) // LEAF_SIZE)
    F_pad = n_leaves * LEAF_SIZE
    leaf_bank = np.zeros((n_leaves, 24, LEAF_SIZE), dtype=np.float32)

    def _pack_vec3(row_start: int, arr_f3: np.ndarray) -> None:
      buf = np.empty((F_pad, 3), dtype=np.float32)
      buf[:F] = arr_f3
      if F_pad > F:
        buf[F:] = arr_f3[-1]
      leaf_bank[:, row_start : row_start + 3, :] = buf.reshape(n_leaves, LEAF_SIZE, 3).transpose(0, 2, 1)

    def _pack_scalar(row_idx: int, arr_f: np.ndarray) -> None:
      buf = np.empty((F_pad,), dtype=np.float32)
      buf[:F] = arr_f
      if F_pad > F:
        buf[F:] = arr_f[-1]
      leaf_bank[:, row_idx, :] = buf.reshape(n_leaves, LEAF_SIZE)

    tv0 = verts[faces[:, 0]]
    _pack_vec3(0, tv0)
    _pack_vec3(3, verts[faces[:, 1]] - tv0)
    _pack_vec3(6, verts[faces[:, 2]] - tv0)
    del tv0

    tv_buf = np.zeros((F_pad,), dtype=np.float32)
    tv_buf[:F] = 1.0
    leaf_bank[:, 9, :] = tv_buf.reshape(n_leaves, LEAF_SIZE)
    _pack_vec3(10, vnorm[faces[:, 0]])
    _pack_vec3(13, vnorm[faces[:, 1]])
    _pack_vec3(16, vnorm[faces[:, 2]])
    _pack_vec3(19, tri_alb)
    _pack_scalar(22, np.arange(F, dtype=np.float32))
    _pack_scalar(23, tri_mc)

    v0_l = leaf_bank[:, 0:3, :]
    v1_l = v0_l + leaf_bank[:, 3:6, :]
    v2_l = v0_l + leaf_bank[:, 6:9, :]
    pts_min = np.minimum(v0_l, np.minimum(v1_l, v2_l))
    pts_max = np.maximum(v0_l, np.maximum(v1_l, v2_l))
    tv_mask = leaf_bank[:, 9:10, :] > 0.5
    leaf_min = np.min(np.where(tv_mask, pts_min, 1e20), axis=2) - 1e-4
    leaf_max = np.max(np.where(tv_mask, pts_max, -1e20), axis=2) + 1e-4

    n_l1 = max(1, (n_leaves + BRANCH_FACTOR - 1) // BRANCH_FACTOR)
    n_leaves_p = n_l1 * BRANCH_FACTOR
    l0_aabbs = np.zeros((n_l1, 8, 128), dtype=np.float32)
    l0_aabbs[:, 0:3, :] = 1e20
    l0_aabbs[:, 3:6, :] = -1e20
    lmin_p = np.full((n_leaves_p, 3), 1e20, dtype=np.float32)
    lmax_p = np.full((n_leaves_p, 3), -1e20, dtype=np.float32)
    lval_p = np.zeros((n_leaves_p,), dtype=np.float32)
    lmin_p[:n_leaves] = leaf_min
    lmax_p[:n_leaves] = leaf_max
    lval_p[:n_leaves] = 1.0
    lmin_g = lmin_p.reshape(n_l1, BRANCH_FACTOR, 3).transpose(0, 2, 1)
    lmax_g = lmax_p.reshape(n_l1, BRANCH_FACTOR, 3).transpose(0, 2, 1)
    l0_aabbs[:, 0:3, :BRANCH_FACTOR] = lmin_g
    l0_aabbs[:, 3:6, :BRANCH_FACTOR] = lmax_g
    l0_aabbs[:, 6, :BRANCH_FACTOR] = lval_p.reshape(n_l1, BRANCH_FACTOR)
    l1_min = np.min(lmin_g, axis=2)
    l1_max = np.max(lmax_g, axis=2)

    n_l2 = max(1, (n_l1 + BRANCH_FACTOR - 1) // BRANCH_FACTOR)
    assert n_l2 <= BRANCH_FACTOR, f"n_l2={n_l2} > 64"
    n_l1_p = n_l2 * BRANCH_FACTOR
    l2_aabbs = np.zeros((n_l2, 8, 128), dtype=np.float32)
    l2_aabbs[:, 0:3, :] = 1e20
    l2_aabbs[:, 3:6, :] = -1e20
    m1min_p = np.full((n_l1_p, 3), 1e20, dtype=np.float32)
    m1max_p = np.full((n_l1_p, 3), -1e20, dtype=np.float32)
    m1val_p = np.zeros((n_l1_p,), dtype=np.float32)
    m1min_p[:n_l1] = l1_min
    m1max_p[:n_l1] = l1_max
    m1val_p[:n_l1] = 1.0
    m1min_g = m1min_p.reshape(n_l2, BRANCH_FACTOR, 3).transpose(0, 2, 1)
    m1max_g = m1max_p.reshape(n_l2, BRANCH_FACTOR, 3).transpose(0, 2, 1)
    l2_aabbs[:, 0:3, :BRANCH_FACTOR] = m1min_g
    l2_aabbs[:, 3:6, :BRANCH_FACTOR] = m1max_g
    l2_aabbs[:, 6, :BRANCH_FACTOR] = m1val_p.reshape(n_l2, BRANCH_FACTOR)

    l2_min = np.min(m1min_g, axis=2)
    l2_max = np.max(m1max_g, axis=2)
    l2_roots_aabb = np.zeros((8, 128), dtype=np.float32)
    l2_roots_aabb[0:3, :] = 1e20
    l2_roots_aabb[3:6, :] = -1e20
    l2_roots_aabb[0:3, :n_l2] = l2_min.T
    l2_roots_aabb[3:6, :n_l2] = l2_max.T
    l2_roots_aabb[6, :n_l2] = 1.0
    if args.bvh_cache:
      np.savez(args.bvh_cache, order=order, l2_roots_aabb=l2_roots_aabb, l2_aabbs=l2_aabbs, l0_aabbs=l0_aabbs, leaf_bank=leaf_bank)

  # Build multi-light array (up to 4 local area/point/spot lights + optional sun)
  bmin, bmax = verts.min(0), verts.max(0)
  center = 0.5 * (bmin + bmax)
  span = float(np.max(bmax - bmin))
  sun_l = next((l for l in lights_list if l["type"] == "SUN"), None)
  local_ls = [
      l for l in lights_list
      if l["type"] in ("AREA", "POINT", "SPOT")
      and float(l["energy"]) > 0.1
      and float(np.linalg.norm(np.asarray(l["pos"], np.float32) - center)) < 35.0
  ]
  if not local_ls:
    local_ls = [l for l in lights_list if l["type"] in ("AREA", "POINT", "SPOT") and float(l["energy"]) > 0.1]
  local_ls.sort(
      key=lambda l: float(l["energy"]) / (float(np.linalg.norm(np.asarray(l["pos"], np.float32) - center)) ** 2 + 1.0),
      reverse=True,
  )
  local_ls = local_ls[:4]

  l_pos_list, l_nrm_list, l_emit_list, l_rad_list = [], [], [], []
  for l in local_ls:
    lp = np.asarray(l["pos"], dtype=np.float32)
    ld = np.asarray(l["dir"], dtype=np.float32)
    if np.linalg.norm(ld) < 1e-4 or l["type"] == "POINT":
      ld = center - lp
    ld = ld / max(float(np.linalg.norm(ld)), 1e-12)
    lr = max(0.35, min(float(l.get("size", 0.6)) * 0.75, span * 0.22))
    col = np.asarray(l["color"], dtype=np.float32)
    # Physical Watts / (pi * r^2) scaled gently so area lights match Cycles indoor exposure
    area = math.pi * lr * lr
    rad_e = (float(l["energy"]) / max(area * math.pi, 0.25)) * col
    rad_e = np.clip(rad_e, 0.0, 45.0)
    l_pos_list.append(lp)
    l_nrm_list.append(ld)
    l_emit_list.append(rad_e)
    l_rad_list.append(lr)

  if not l_pos_list:
    l_pos_list.append(np.array([center[0], center[1], bmax[2] - 0.05 * span], np.float32))
    l_nrm_list.append(np.array([0.0, 0.0, -1.0], np.float32))
    l_emit_list.append(np.array([12.0, 11.0, 9.5], np.float32))
    l_rad_list.append(max(0.5, span * 0.12))

  n_local_lights = len(l_pos_list)
  l_pos_arr = jnp.asarray(np.stack(l_pos_list, axis=0), dtype=jnp.float32)
  l_nrm_arr = jnp.asarray(np.stack(l_nrm_list, axis=0), dtype=jnp.float32)
  l_emit_arr = jnp.asarray(np.stack(l_emit_list, axis=0), dtype=jnp.float32)
  l_rad_arr = jnp.asarray(np.array(l_rad_list, dtype=np.float32), dtype=jnp.float32)

  has_sun = sun_l is not None
  if has_sun:
    s_toward = -np.asarray(sun_l["dir"], dtype=np.float32)
    s_toward = s_toward / max(float(np.linalg.norm(s_toward)), 1e-12)
    scol = np.asarray(sun_l["color"], dtype=np.float32)
    s_emit = scol * min(max(float(sun_l["energy"]) * 2.5, 8.0), 32.0)
  else:
    s_toward = np.array([0.4, -0.4, 0.8], np.float32)
    s_emit = np.array([10.0, 9.0, 7.5], np.float32)

  sun_d = jnp.asarray(s_toward, dtype=jnp.float32)
  st1, st2 = _build_onb(sun_d)
  sun_emit = jnp.asarray(s_emit, dtype=jnp.float32)
  sun_sin_max = math.sqrt(max(0.0, 1.0 - 0.9988 ** 2))

  gt_im = Image.open(args.gt_png).convert("RGB")
  gt_w, gt_h = gt_im.size
  gt_np = np.asarray(gt_im, dtype=np.float32) / 255.0

  devs = jax.devices()
  n_chips = len(devs)
  assert n_chips == 4
  w_mult = 16 * n_chips
  W = ((gt_w + w_mult - 1) // w_mult) * w_mult
  H = ((gt_h + 7) // 8) * 8
  W_local = W // n_chips

  exposure = args.exposure if args.exposure is not None else (1.15 * (2.0 ** bl_exp) if "classroom" not in args.scene_label else 0.72)

  mesh = Mesh(np.array(devs), ("dp",))
  rep = NamedSharding(mesh, P())
  sh_dp = NamedSharding(mesh, P("dp"))

  l2r_d = jax.device_put(l2_roots_aabb, rep)
  l2_d = jax.device_put(l2_aabbs, rep)
  l0_d = jax.device_put(l0_aabbs, rep)
  leaf_d = jax.device_put(leaf_bank, rep)
  tri_uv_d = jax.device_put(tri_uvs, rep)
  tri_tid_d = jax.device_put(tri_tex_id, rep)
  tri_alb_d = jax.device_put(tri_alb, rep)
  tri_rgh_d = jax.device_put(tri_rgh, rep)
  tri_mc_d = jax.device_put(tri_mc, rep)
  tex_atlas_d = jax.device_put(tex_atlas_np, rep)
  chip_ids = jax.device_put(np.arange(n_chips, dtype=np.int32), sh_dp)

  def _uv_raycast(l2r, l2, l0, lf, ro, rd, t_limit, H_l, W_l, any_hit=False):
    B, R, _ = ro.shape
    rays_2d, n_tiles, can_2d_tile = _pack_rays_tiled(ro, rd, t_limit, H_l, W_l)
    total_tiles = B * n_tiles
    leaf_rows = 24

    def _kern(ray_ref, l2r_ref, l2_ref, l0_ref, lf_ref, out_ref, l2_v, l0_v, lf_v, s2, s0, sl):
      cols64 = jax.lax.broadcasted_iota(jnp.int32, (1, BRANCH_FACTOR), 1)
      rt = ray_ref[...]
      l2rb = l2r_ref[...]
      rox = jnp.transpose(rt[0:1, :], (1, 0))
      roy = jnp.transpose(rt[1:2, :], (1, 0))
      roz = jnp.transpose(rt[2:3, :], (1, 0))
      rdx = jnp.transpose(rt[3:4, :], (1, 0))
      rdy = jnp.transpose(rt[4:5, :], (1, 0))
      rdz = jnp.transpose(rt[5:6, :], (1, 0))
      bd = jnp.transpose(rt[6:7, :], (1, 0))

      irx = jnp.where(jnp.abs(rdx) > 1e-12, 1.0 / jnp.where(jnp.abs(rdx) > 1e-12, rdx, 1.0), 1e12)
      iry = jnp.where(jnp.abs(rdy) > 1e-12, 1.0 / jnp.where(jnp.abs(rdy) > 1e-12, rdy, 1.0), 1e12)
      irz = jnp.where(jnp.abs(rdz) > 1e-12, 1.0 / jnp.where(jnp.abs(rdz) > 1e-12, rdz, 1.0), 1e12)

      l2_bx0, l2_by0, l2_bz0 = l2rb[0:1, :BRANCH_FACTOR], l2rb[1:2, :BRANCH_FACTOR], l2rb[2:3, :BRANCH_FACTOR]
      l2_bx1, l2_by1, l2_bz1 = l2rb[3:4, :BRANCH_FACTOR], l2rb[4:5, :BRANCH_FACTOR], l2rb[5:6, :BRANCH_FACTOR]
      l2_val = l2rb[6:7, :BRANCH_FACTOR] > 0.5
      tx0, tx1 = (l2_bx0 - rox) * irx, (l2_bx1 - rox) * irx
      ty0, ty1 = (l2_by0 - roy) * iry, (l2_by1 - roy) * iry
      tz0, tz1 = (l2_bz0 - roz) * irz, (l2_bz1 - roz) * irz
      tpos = jnp.maximum(0.0, jnp.maximum(jnp.minimum(tx0, tx1), jnp.maximum(jnp.minimum(ty0, ty1), jnp.minimum(tz0, tz1))))
      tmax = jnp.minimum(jnp.maximum(tx0, tx1), jnp.minimum(jnp.maximum(ty0, ty1), jnp.maximum(tz0, tz1)))
      t2_eff = jnp.where((tmax >= tpos) & l2_val, tpos, 1e20)
      u2_f = jnp.where(l2_val, 1.0, 0.0)
      l2_tm = jnp.min(jnp.where(t2_eff < bd, t2_eff, 1e20), axis=0, keepdims=True)
      z1 = jnp.zeros((BLOCK_R, 1), jnp.float32)

      def _l2_c(c): return jnp.min(c[0]) < jnp.max(c[2])
      def _l2_b(c):
        t2, un2, bd2, nx2, ny2, nz2, bu2, bv2, ti2, mc2 = c
        br = jnp.argmin(t2, axis=1)[0]
        un2n = jnp.where(cols64 == br, 0.0, un2)
        cp2 = pltpu.make_async_copy(l2_ref.at[br, :, :], l2_v, s2); cp2.start(); cp2.wait()
        lb2 = l2_v[...]
        bx0, by0, bz0 = lb2[0:1, :BRANCH_FACTOR], lb2[1:2, :BRANCH_FACTOR], lb2[2:3, :BRANCH_FACTOR]
        bx1, by1, bz1 = lb2[3:4, :BRANCH_FACTOR], lb2[4:5, :BRANCH_FACTOR], lb2[5:6, :BRANCH_FACTOR]
        l1v = lb2[6:7, :BRANCH_FACTOR] > 0.5
        tx0, tx1 = (bx0 - rox) * irx, (bx1 - rox) * irx
        ty0, ty1 = (by0 - roy) * iry, (by1 - roy) * iry
        tz0, tz1 = (bz0 - roz) * irz, (bz1 - roz) * irz
        tp1 = jnp.maximum(0.0, jnp.maximum(jnp.minimum(tx0, tx1), jnp.maximum(jnp.minimum(ty0, ty1), jnp.minimum(tz0, tz1))))
        tm1 = jnp.minimum(jnp.maximum(tx0, tx1), jnp.minimum(jnp.maximum(ty0, ty1), jnp.maximum(tz0, tz1)))
        t1_eff = jnp.where((tm1 >= tp1) & l1v, tp1, 1e20)
        u1_f = jnp.where(l1v, 1.0, 0.0)
        l1_tm = jnp.min(jnp.where(t1_eff < bd2, t1_eff, 1e20), axis=0, keepdims=True)

        def _l1_c(c1): return jnp.min(c1[0]) < jnp.max(c1[2])
        def _l1_b(c1):
          t1, un1, bd1, nx1, ny1, nz1, bu1, bv1, ti1, mc1 = c1
          bc1 = jnp.argmin(t1, axis=1)[0]
          un1n = jnp.where(cols64 == bc1, 0.0, un1)
          l1i = br * BRANCH_FACTOR + bc1
          cp0 = pltpu.make_async_copy(l0_ref.at[l1i, :, :], l0_v, s0); cp0.start(); cp0.wait()
          lb0 = l0_v[...]
          bx0, by0, bz0 = lb0[0:1, :BRANCH_FACTOR], lb0[1:2, :BRANCH_FACTOR], lb0[2:3, :BRANCH_FACTOR]
          bx1, by1, bz1 = lb0[3:4, :BRANCH_FACTOR], lb0[4:5, :BRANCH_FACTOR], lb0[5:6, :BRANCH_FACTOR]
          l0v = lb0[6:7, :BRANCH_FACTOR] > 0.5
          tx0, tx1 = (bx0 - rox) * irx, (bx1 - rox) * irx
          ty0, ty1 = (by0 - roy) * iry, (by1 - roy) * iry
          tz0, tz1 = (bz0 - roz) * irz, (bz1 - roz) * irz
          tp0 = jnp.maximum(0.0, jnp.maximum(jnp.minimum(tx0, tx1), jnp.maximum(jnp.minimum(ty0, ty1), jnp.minimum(tz0, tz1))))
          tm0 = jnp.minimum(jnp.maximum(tx0, tx1), jnp.minimum(jnp.maximum(ty0, ty1), jnp.maximum(tz0, tz1)))
          t0_eff = jnp.where((tm0 >= tp0) & l0v, tp0, 1e20)
          u0_f = jnp.where(l0v, 1.0, 0.0)
          l0_tm = jnp.min(jnp.where(t0_eff < bd1, t0_eff, 1e20), axis=0, keepdims=True)

          def _l0_c(c0): return jnp.min(c0[0]) < jnp.max(c0[2])
          def _l0_b(c0):
            t0, un0, bd0, nx0, ny0, nz0, bu0, bv0, ti0, mc0 = c0
            bc0 = jnp.argmin(t0, axis=1)[0]
            un0n = jnp.where(cols64 == bc0, 0.0, un0)
            li = l1i * BRANCH_FACTOR + bc0
            cp1 = pltpu.make_async_copy(lf_ref.at[li, :leaf_rows, :], lf_v, sl); cp1.start(); cp1.wait()
            pk = lf_v[...]
            v0x, v0y, v0z = pk[0:1, :], pk[1:2, :], pk[2:3, :]
            e1x, e1y, e1z = pk[3:4, :], pk[4:5, :], pk[5:6, :]
            e2x, e2y, e2z = pk[6:7, :], pk[7:8, :], pk[8:9, :]
            tv = pk[9:10, :] > 0.5
            hx, hy, hz = rdy * e2z - rdz * e2y, rdz * e2x - rdx * e2z, rdx * e2y - rdy * e2x
            a = e1x * hx + e1y * hy + e1z * hz
            ok_a = tv & (jnp.abs(a) > 1e-8)
            f = jnp.where(ok_a, 1.0 / jnp.where(ok_a, a, 1.0), 0.0)
            sx, sy, sz = rox - v0x, roy - v0y, roz - v0z
            u = f * (sx * hx + sy * hy + sz * hz)
            qx, qy, qz = sy * e1z - sz * e1y, sz * e1x - sx * e1z, sx * e1y - sy * e1x
            v = f * (rdx * qx + rdy * qy + rdz * qz)
            t = f * (e2x * qx + e2y * qy + e2z * qz)
            hit_t = ok_a & (u >= 0.0) & (v >= 0.0) & ((u + v) <= 1.0) & (t >= 0.0) & (t < bd0)

            if any_hit:
              hit_opq = hit_t & (pk[23:24, :] < 1.8)
              new_bd = jnp.where(jnp.any(hit_opq, axis=1, keepdims=True), 0.0, bd0)
              t0n = jnp.min(jnp.where((t0_eff < new_bd) & (un0n > 0.5), t0_eff, 1e20), axis=0, keepdims=True)
              return (t0n, un0n, new_bd, nx0, ny0, nz0, bu0, bv0, ti0, mc0)

            t_cand = jnp.where(hit_t, t, MJ_MAXVAL)
            min_t = jnp.min(t_cand, axis=1, keepdims=True)
            imp = min_t < bd0

            def _on_hit(_):
              n0x, n0y, n0z = pk[10:11, :], pk[11:12, :], pk[12:13, :]
              n1x, n1y, n1z = pk[13:14, :], pk[14:15, :], pk[15:16, :]
              n2x, n2y, n2z = pk[16:17, :], pk[17:18, :], pk[18:19, :]
              win = hit_t & (t_cand == min_t)
              bw = 1.0 - u - v
              lnx = jnp.sum(jnp.where(win, bw * n0x + u * n1x + v * n2x, 0.0), axis=1, keepdims=True)
              lny = jnp.sum(jnp.where(win, bw * n0y + u * n1y + v * n2y, 0.0), axis=1, keepdims=True)
              lnz = jnp.sum(jnp.where(win, bw * n0z + u * n1z + v * n2z, 0.0), axis=1, keepdims=True)
              inv_l = 1.0 / jnp.maximum(jnp.sqrt(lnx * lnx + lny * lny + lnz * lnz), 1e-12)
              lbu = jnp.sum(jnp.where(win, u, 0.0), axis=1, keepdims=True)
              lbv = jnp.sum(jnp.where(win, v, 0.0), axis=1, keepdims=True)
              lti = jnp.sum(jnp.where(win, pk[22:23, :], 0.0), axis=1, keepdims=True)
              lmc = jnp.sum(jnp.where(win, pk[23:24, :], 0.0), axis=1, keepdims=True)
              new_bd = jnp.where(imp, min_t, bd0)
              t0n = jnp.min(jnp.where((t0_eff < new_bd) & (un0n > 0.5), t0_eff, 1e20), axis=0, keepdims=True)
              return (
                  t0n, new_bd,
                  jnp.where(imp, lnx * inv_l, nx0), jnp.where(imp, lny * inv_l, ny0), jnp.where(imp, lnz * inv_l, nz0),
                  jnp.where(imp, lbu, bu0), jnp.where(imp, lbv, bv0),
                  jnp.where(imp, lti, ti0), jnp.where(imp, lmc, mc0),
              )

            def _on_miss(_):
              return (jnp.where(un0n > 0.5, t0, 1e20), bd0, nx0, ny0, nz0, bu0, bv0, ti0, mc0)

            t0n, nbd, nnx, nny, nnz, nbu, nbv, nti, nmc = jax.lax.cond(jnp.any(imp), _on_hit, _on_miss, None)
            return (t0n, un0n, nbd, nnx, nny, nnz, nbu, nbv, nti, nmc)

          _, _, nbd1, nnx1, nny1, nnz1, nbu1, nbv1, nti1, nmc1 = jax.lax.while_loop(
              _l0_c, _l0_b, (l0_tm, u0_f, bd1, nx1, ny1, nz1, bu1, bv1, ti1, mc1)
          )
          t1n = jnp.min(jnp.where((t1_eff < nbd1) & (un1n > 0.5), t1_eff, 1e20), axis=0, keepdims=True)
          return (t1n, un1n, nbd1, nnx1, nny1, nnz1, nbu1, nbv1, nti1, nmc1)

        _, _, nbd2, nnx2, nny2, nnz2, nbu2, nbv2, nti2, nmc2 = jax.lax.while_loop(
            _l1_c, _l1_b, (l1_tm, u1_f, bd2, nx2, ny2, nz2, bu2, bv2, ti2, mc2)
        )
        t2n = jnp.min(jnp.where((t2_eff < nbd2) & (un2n > 0.5), t2_eff, 1e20), axis=0, keepdims=True)
        return (t2n, un2n, nbd2, nnx2, nny2, nnz2, nbu2, nbv2, nti2, nmc2)

      _, _, obd, onx, ony, onz, obu, obv, oti, omc = jax.lax.while_loop(
          _l2_c, _l2_b, (l2_tm, u2_f, bd, z1, z1, z1, z1, z1, z1, z1)
      )
      out_ref[...] = jnp.concatenate([
          jnp.transpose(obd, (1, 0)), jnp.transpose(onx, (1, 0)), jnp.transpose(ony, (1, 0)), jnp.transpose(onz, (1, 0)),
          jnp.transpose(obu, (1, 0)), jnp.transpose(obv, (1, 0)), jnp.transpose(oti, (1, 0)), jnp.transpose(omc, (1, 0)),
      ], axis=0)

    out_2d = pl.pallas_call(
        _kern,
        out_shape=jax.ShapeDtypeStruct((total_tiles * 8, BLOCK_R), jnp.float32),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0, grid=(total_tiles,),
            in_specs=[
                pl.BlockSpec((8, BLOCK_R), lambda i: (i, 0)),
                pl.BlockSpec((8, 128), lambda i: (0, 0)),
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=pl.BlockSpec((8, BLOCK_R), lambda i: (i, 0)),
            scratch_shapes=[
                pltpu.VMEM((8, 128), jnp.float32), pltpu.VMEM((8, 128), jnp.float32),
                pltpu.VMEM((leaf_rows, 128), jnp.float32),
                pltpu.SemaphoreType.DMA, pltpu.SemaphoreType.DMA, pltpu.SemaphoreType.DMA,
            ],
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=False,
    )(rays_2d, l2r, l2, l0, lf)

    out_t = out_2d.reshape(B, n_tiles, 8, BLOCK_R).transpose(0, 1, 3, 2)
    if can_2d_tile:
      ty, tx = H_l // 8, W_l // 16
      out_hw = out_t.reshape(B, ty, tx, 8, 16, 8).transpose(0, 1, 3, 2, 4, 5).reshape(B, R, 8)
    else:
      out_hw = out_t.reshape(B, n_tiles * BLOCK_R, 8)[:, :R, :]
    return out_hw[..., 0], out_hw[..., 1:4], out_hw[..., 4], out_hw[..., 5], out_hw[..., 6], out_hw[..., 7]

  aspect = float(W) / float(H)
  half_h = cam_znear * math.tan(0.5 * cam_fovy * (math.pi / 180.0))
  half_w = half_h * aspect
  sky_z_j, sky_h_j, sky_g_j = jnp.asarray(sky_z), jnp.asarray(sky_h), jnp.asarray(sky_g)

  def _env_sky(rd):
    z = rd[..., 2]
    tu = jnp.clip(z, 0.0, 1.0)[..., None]
    td = jnp.clip(-z, 0.0, 1.0)[..., None]
    s = jnp.where(z[..., None] >= 0.0, (1.0 - tu) * sky_h_j + tu * sky_z_j, (1.0 - td) * sky_h_j + td * sky_g_j)
    if has_sun:
      cos_s = jnp.maximum(jnp.sum(rd * sun_d, axis=-1), 0.0)
      s = s + (cos_s ** 32.0)[..., None] * sun_emit * 0.25
    return s

  def _render_local(l2r, l2, l0, lf, tuv, ttid, talb, trgh, tmc, tex_at, cxp, cxm, cid_arr, rng_in):
    cid = cid_arr[0]
    x_off = cid * W_local
    chip_rng = jax.random.fold_in(rng_in, cid)
    R = H * W_local
    spp_par = 16 if (args.spp >= 16 and R <= 32768) else (8 if (args.spp >= 8 and R <= 131072) else (4 if (args.spp >= 4 and R <= 262144) else 1))
    spp_seq = max(1, args.spp // spp_par)
    _, TH, TW, _ = tex_at.shape

    def _sample_tex_uv(tri_i, bu, bv):
      uvs = tuv[tri_i]  # (B, R, 3, 2)
      bw = 1.0 - bu - bv
      uv = bw[..., None] * uvs[..., 0, :] + bu[..., None] * uvs[..., 1, :] + bv[..., None] * uvs[..., 2, :]
      u_w = jnp.mod(uv[..., 0], 1.0) * float(TW - 1)
      v_w = jnp.mod(1.0 - uv[..., 1], 1.0) * float(TH - 1)
      x0 = jnp.clip(jnp.floor(u_w).astype(jnp.int32), 0, TW - 1)
      y0 = jnp.clip(jnp.floor(v_w).astype(jnp.int32), 0, TH - 1)
      x1 = jnp.minimum(x0 + 1, TW - 1)
      y1 = jnp.minimum(y0 + 1, TH - 1)
      fx = (u_w - x0.astype(jnp.float32))[..., None]
      fy = (v_w - y0.astype(jnp.float32))[..., None]
      tid = ttid[tri_i]
      has_t = tid >= 0
      tid_s = jnp.maximum(tid, 0)
      c00 = tex_at[tid_s, y0, x0]
      c10 = tex_at[tid_s, y0, x1]
      c01 = tex_at[tid_s, y1, x0]
      c11 = tex_at[tid_s, y1, x1]
      tex_rgba = (1.0 - fx) * (1.0 - fy) * c00 + fx * (1.0 - fy) * c10 + (1.0 - fx) * fy * c01 + fx * fy * c11
      alb = jnp.where(has_t[..., None], jnp.clip(tex_rgba[..., :3], 0.01, 0.99), talb[tri_i])
      rgh = jnp.where(has_t, jnp.clip(tex_rgba[..., 3], 0.02, 0.98), trgh[tri_i])
      return alb, rgh

    def _trace_batch(s_idx, acc):
      rad_acc, nrm_acc, alb_acc, dep_acc, spc_acc = acc
      s_rng = jax.random.fold_in(chip_rng, s_idx)
      k_jit, k_bnc = jax.random.split(s_rng)
      jitter = jax.random.uniform(k_jit, (spp_par, R, 2), minval=-0.5, maxval=0.5, dtype=jnp.float32)
      px = jnp.arange(W_local, dtype=jnp.float32) + jnp.asarray(x_off, jnp.float32) + 0.5
      py = jnp.arange(H, dtype=jnp.float32) + 0.5
      gx, gy = jnp.meshgrid(px, py, indexing="xy")
      u = (gx.reshape(1, R) + jitter[..., 0]) / float(W)
      v = (gy.reshape(1, R) + jitter[..., 1]) / float(H)
      x = -half_w + (2.0 * half_w) * u
      y = half_h - (2.0 * half_h) * v
      raw = jnp.stack([x, y, jnp.full_like(x, -cam_znear)], axis=-1)
      ld = raw / jnp.maximum(jnp.linalg.norm(raw, axis=-1, keepdims=True), 1e-12)
      neg_z = jnp.maximum(-ld[..., 2], 1e-6)
      rd0 = jnp.einsum("ij,brj->bri", cxm[0], ld, precision=jax.lax.Precision.HIGHEST)
      rd0 = rd0 / jnp.maximum(jnp.linalg.norm(rd0, axis=-1, keepdims=True), 1e-12)
      ro0 = cxp[0][None, None, :] + (cam_znear / neg_z)[..., None] * rd0
      t_max0 = jnp.broadcast_to(((cam_zfar - cam_znear) / neg_z), (spp_par, R))
      thr0 = jnp.ones((spp_par, R, 3), dtype=jnp.float32)
      alive0 = jnp.ones((spp_par, R), dtype=jnp.bool_)
      rad0 = jnp.zeros((spp_par, R, 3), dtype=jnp.float32)
      pn0 = jnp.zeros((spp_par, R, 3), dtype=jnp.float32)
      pa0 = jnp.zeros((spp_par, R, 3), dtype=jnp.float32)
      pd0 = jnp.full((spp_par, R), 120.0, dtype=jnp.float32)
      ps0 = jnp.zeros((spp_par, R), dtype=jnp.float32)

      def _bounce(b_idx, carry):
        ro, rd, t_lim, thr, alive, rad, pn, pa, pd, ps = carry
        b_rng = jax.random.fold_in(k_bnc, b_idx)
        k_lsel, k_lu, k_lv, k_su, k_sv, k_lobe, k_u1, k_u2 = jax.random.split(b_rng, 8)

        d_hit, nrm_raw, bu, bv, ti_f, _mc_f = _uv_raycast(
            l2r, l2, l0, lf, ro, rd, jnp.where(alive, t_lim, 0.0), H, W_local, any_hit=False
        )
        hit = alive & (d_hit < t_lim - 1e-4) & (d_hit >= 0.0)
        miss = alive & (~hit)
        rad = rad + jnp.where(miss[..., None], thr * _env_sky(rd), 0.0)

        tri_i = jnp.clip(jnp.floor(ti_f + 0.5).astype(jnp.int32), 0, F - 1)
        alb, rgh = _sample_tex_uv(tri_i, bu, bv)
        mcode = tmc[tri_i]

        n_unit = nrm_raw / jnp.maximum(jnp.linalg.norm(nrm_raw, axis=-1, keepdims=True), 1e-12)
        wo = -rd
        entering = jnp.sum(n_unit * wo, axis=-1) >= 0.0
        n_ff = jnp.where(entering[..., None], n_unit, -n_unit)

        is_emit = hit & (mcode >= 2.8)
        is_glass = hit & (mcode >= 1.8) & (mcode < 2.8)
        is_opaque = hit & (mcode < 1.8)
        metallic = jnp.clip(mcode, 0.0, 1.0)
        is_spec_prim = is_glass | (hit & ((rgh < 0.14) | (metallic > 0.75)))

        pn = jnp.where(b_idx == 0, jnp.where(hit[..., None], n_ff, jnp.array([0.0, 0.0, 1.0], jnp.float32)), pn)
        pa = jnp.where(b_idx == 0, jnp.where(hit[..., None], alb, 0.0), pa)
        pd = jnp.where(b_idx == 0, jnp.where(hit, d_hit, 120.0), pd)
        ps = jnp.where(b_idx == 0, jnp.where(is_spec_prim, 1.0, 0.0), ps)

        emit_power = jnp.where(mcode >= 2.8, (mcode - 2.0) * 6.0, 0.0)
        rad = rad + jnp.where(is_emit[..., None], thr * alb * emit_power[..., None], 0.0)
        hit_pos = ro + jnp.where(hit, d_hit, 0.0)[..., None] * rd

        # Multi-light stochastic NEE (uniformly pick 1 of n_local_lights and scale by n_local_lights)
        l_idx = jax.random.randint(k_lsel, (spp_par, R), 0, n_local_lights)
        ac = l_pos_arr[l_idx]
        an = l_nrm_arr[l_idx]
        ae = l_emit_arr[l_idx]
        ar = l_rad_arr[l_idx]
        at1, at2 = _build_onb(an)
        lu = jax.random.uniform(k_lu, (spp_par, R), minval=0.0, maxval=1.0, dtype=jnp.float32)
        lv = jax.random.uniform(k_lv, (spp_par, R), minval=0.0, maxval=1.0, dtype=jnp.float32)
        lr = ar * jnp.sqrt(lu)
        lphi = (2.0 * math.pi) * lv
        lp = ac + (lr * jnp.cos(lphi))[..., None] * at1 + (lr * jnp.sin(lphi))[..., None] * at2
        to_l = lp - hit_pos
        dist_l = jnp.linalg.norm(to_l, axis=-1)
        wi_l = to_l / jnp.maximum(dist_l[..., None], 1e-12)
        cos_s = jnp.maximum(jnp.sum(n_ff * wi_l, axis=-1), 0.0)
        cos_l = jnp.maximum(jnp.sum(an * (-wi_l), axis=-1), 0.0)
        need_sh = is_opaque & (cos_s > 0.0) & (cos_l > 0.0)
        sh_o = hit_pos + n_ff * 1.5e-3
        sh_lim = jnp.where(need_sh, jnp.maximum(dist_l - 2.0e-3, 0.0), 0.0)

        d_sh, *_ = _uv_raycast(l2r, l2, l0, lf, sh_o, wi_l, sh_lim, H, W_local, any_hit=True)
        vis = jnp.where(need_sh & (d_sh >= sh_lim - 1e-4), 1.0, 0.0)
        brdf_a = _eval_pbr_brdf(n_ff, wo, wi_l, alb, metallic, rgh)
        geom_a = (cos_s * cos_l) / (dist_l * dist_l + 0.08) * (math.pi * ar * ar) * float(n_local_lights)
        nee_a = jnp.minimum(thr * brdf_a * ae * (geom_a * vis)[..., None], 18.0)
        rad = rad + jnp.where(is_opaque[..., None], nee_a, 0.0)

        if has_sun:
          su = jax.random.uniform(k_su, (spp_par, R), minval=0.0, maxval=1.0, dtype=jnp.float32)
          sv = jax.random.uniform(k_sv, (spp_par, R), minval=0.0, maxval=1.0, dtype=jnp.float32)
          sr = sun_sin_max * jnp.sqrt(su)
          sphi = (2.0 * math.pi) * sv
          wi_sun = sun_d[None, None, :] + (sr * jnp.cos(sphi))[..., None] * st1[None, None, :] + (sr * jnp.sin(sphi))[..., None] * st2[None, None, :]
          wi_sun = wi_sun / jnp.maximum(jnp.linalg.norm(wi_sun, axis=-1, keepdims=True), 1e-12)
          cos_sun = jnp.maximum(jnp.sum(n_ff * wi_sun, axis=-1), 0.0)
          need_sun = is_opaque & (cos_sun > 0.0)
          sun_lim = jnp.where(need_sun, jnp.float32(108.0), 0.0)
          d_sun, *_ = _uv_raycast(l2r, l2, l0, lf, sh_o, wi_sun, sun_lim, H, W_local, any_hit=True)
          vis_sun = jnp.where(need_sun & (d_sun >= sun_lim - 1e-2), 1.0, 0.0)
          brdf_sun = _eval_pbr_brdf(n_ff, wo, wi_sun, alb, metallic, rgh)
          nee_sun = jnp.minimum(thr * brdf_sun * sun_emit[None, None, :] * (cos_sun * vis_sun)[..., None], 20.0)
          rad = rad + jnp.where(is_opaque[..., None], nee_sun, 0.0)

        u_lobe = jax.random.uniform(k_lobe, (spp_par, R), minval=0.0, maxval=1.0, dtype=jnp.float32)
        u1 = jax.random.uniform(k_u1, (spp_par, R), minval=0.0, maxval=1.0, dtype=jnp.float32)
        u2 = jax.random.uniform(k_u2, (spp_par, R), minval=0.0, maxval=1.0, dtype=jnp.float32)

        p_spec = jnp.clip(0.15 + 0.78 * metallic, 0.15, 0.90)
        choose_spec = u_lobe < p_spec
        wi_diff = _sample_cosine_hemisphere(u1, u2, n_ff)
        alpha = jnp.maximum(rgh * rgh, 0.02)
        h_ggx = _sample_ggx_half(u1, u2, n_ff, alpha)
        wi_spec = 2.0 * jnp.sum(wo * h_ggx, axis=-1, keepdims=True) * h_ggx - wo
        wi_spec = wi_spec / jnp.maximum(jnp.linalg.norm(wi_spec, axis=-1, keepdims=True), 1e-12)
        wi_opq = jnp.where(choose_spec[..., None], wi_spec, wi_diff)

        ndotl_s = jnp.maximum(jnp.sum(n_ff * wi_spec, axis=-1), 0.0)
        ndotv = jnp.maximum(jnp.sum(n_ff * wo, axis=-1), 1e-4)
        ndoth = jnp.maximum(jnp.sum(n_ff * h_ggx, axis=-1), 1e-4)
        vdoth = jnp.maximum(jnp.sum(wo * h_ggx, axis=-1), 1e-4)
        f0 = 0.04 * (1.0 - metallic[..., None]) + alb * metallic[..., None]
        F_s = f0 + (1.0 - f0) * ((1.0 - vdoth) ** 5)[..., None]
        G_s = _smith_g1(ndotl_s, alpha) * _smith_g1(ndotv, alpha)
        w_spec = jnp.where(ndotl_s[..., None] > 0.0, jnp.minimum(F_s * ((G_s * vdoth) / jnp.maximum(ndotv * ndoth * p_spec, 1e-4))[..., None], 4.0), 0.0)
        w_diff = (1.0 - metallic[..., None]) * alb / jnp.maximum(1.0 - p_spec[..., None], 0.1)
        w_opq = jnp.where(choose_spec[..., None], w_spec, w_diff)

        eta = jnp.where(entering, 1.0 / 1.50, 1.50)
        cos_i = jnp.clip(jnp.sum(wo * n_ff, axis=-1), 0.0, 1.0)
        sin2_t = eta * eta * (1.0 - cos_i * cos_i)
        tir = sin2_t > 1.0
        r0_g = ((1.0 - eta) / (1.0 + eta)) ** 2
        fresnel_g = r0_g + (1.0 - r0_g) * ((1.0 - cos_i) ** 5)
        refl_dir = 2.0 * cos_i[..., None] * n_ff - wo
        cos_t = jnp.sqrt(jnp.maximum(1.0 - sin2_t, 0.0))
        refr_dir = eta[..., None] * rd + (eta * cos_i - cos_t)[..., None] * n_ff
        do_refl = tir | (u_lobe < fresnel_g)
        wi_glass = jnp.where(do_refl[..., None], refl_dir, refr_dir)
        wi_glass = wi_glass / jnp.maximum(jnp.linalg.norm(wi_glass, axis=-1, keepdims=True), 1e-12)
        ro_glass = jnp.where(do_refl[..., None], hit_pos + n_ff * 1.5e-3, hit_pos - n_ff * 1.5e-3)

        wi_next = jnp.where(is_glass[..., None], wi_glass, wi_opq)
        new_ro = jnp.where(is_glass[..., None], ro_glass, hit_pos + n_ff * 1.5e-3)
        bounce_w = jnp.where(is_glass[..., None], alb, w_opq)
        new_thr = thr * bounce_w
        new_alive = (is_opaque | is_glass) & (jnp.max(new_thr, axis=-1) > 1e-3)
        new_t_lim = jnp.broadcast_to(jnp.float32(120.0), (spp_par, R))
        return (new_ro, wi_next, new_t_lim, new_thr, new_alive, rad, pn, pa, pd, ps)

      _, _, _, _, _, final_rad, pn, pa, pd, ps = jax.lax.fori_loop(
          0, args.max_bounces, _bounce, (ro0, rd0, t_max0, thr0, alive0, rad0, pn0, pa0, pd0, ps0)
      )
      return (
          rad_acc + jnp.mean(final_rad, axis=0),
          nrm_acc + jnp.mean(pn, axis=0),
          alb_acc + jnp.mean(pa, axis=0),
          dep_acc + jnp.mean(pd, axis=0),
          spc_acc + jnp.mean(ps, axis=0),
      )

    init_acc = (
        jnp.zeros((R, 3), dtype=jnp.float32),
        jnp.zeros((R, 3), dtype=jnp.float32),
        jnp.zeros((R, 3), dtype=jnp.float32),
        jnp.zeros((R,), dtype=jnp.float32),
        jnp.zeros((R,), dtype=jnp.float32),
    )
    rs, ns, as_, ds_, ss = jax.lax.fori_loop(0, spp_seq, _trace_batch, init_acc)
    inv_s = 1.0 / float(spp_seq)
    packed = jnp.concatenate([
        (rs * inv_s).reshape(H, W_local, 3),
        (ns * inv_s).reshape(H, W_local, 3),
        (as_ * inv_s).reshape(H, W_local, 3),
        (ds_ * inv_s).reshape(H, W_local, 1),
        (ss * inv_s).reshape(H, W_local, 1),
    ], axis=-1)
    return packed[None, ...]

  sm_kw = {"check_rep": False} if "check_rep" in inspect.signature(shard_map).parameters else {"check_vma": False}
  sharded_fn = shard_map(
      _render_local,
      mesh=mesh,
      in_specs=(P(), P(), P(), P(), P(), P(), P(), P(), P(), P(), P(), P(), P("dp"), P()),
      out_specs=P("dp"),
      **sm_kw,
  )

  @jax.jit
  def render_fn(cxp, cxm, rng):
    sh_out = sharded_fn(
        l2r_d, l2_d, l0_d, leaf_d, tri_uv_d, tri_tid_d, tri_alb_d, tri_rgh_d, tri_mc_d, tex_atlas_d,
        cxp, cxm, chip_ids, rng,
    )
    full = sh_out.transpose(1, 0, 2, 3).reshape(H, W, 11)
    rad_lin = full[..., 0:3]
    nrm_g = full[..., 3:6]
    nrm_g = nrm_g / jnp.maximum(jnp.linalg.norm(nrm_g, axis=-1, keepdims=True), 1e-6)
    alb_g = full[..., 6:9]
    dep_g = full[..., 9]
    spc_g = full[..., 10]
    rad_clean = _tpu_joint_atrous_denoise(rad_lin, nrm_g, alb_g, dep_g, spc_g, sigma_l=0.32, n_passes=4)
    return _agx_tonemap_srgb(rad_clean, exposure=exposure)

  cxp_d = jax.device_put(cam_pos, rep)
  cxm_d = jax.device_put(cam_xmat, rep)
  rng_d = jax.device_put(jax.random.PRNGKey(7), rep)
  lowered = render_fn.lower(cxp_d, cxm_d, rng_d)
  hlo_info = audit_hlo_zero_copy(lowered)

  tc0 = time.perf_counter()
  compiled = lowered.compile()
  compile_s = time.perf_counter() - tc0

  y = compiled(cxp_d, cxm_d, jax.device_put(jax.random.PRNGKey(11), rep))
  jax.block_until_ready(y)

  t0 = time.perf_counter()
  y = compiled(cxp_d, cxm_d, jax.device_put(jax.random.PRNGKey(21), rep))
  jax.block_until_ready(y)
  dt_s = time.perf_counter() - t0
  img_np = np.asarray(jax.device_get(y), dtype=np.float32)
  if (W, H) != (gt_w, gt_h):
    img_cmp = np.asarray(
        Image.fromarray(np.clip(img_np * 255.0 + 0.5, 0, 255).astype(np.uint8)).resize((gt_w, gt_h), Image.Resampling.BILINEAR),
        dtype=np.float32,
    ) / 255.0
  else:
    img_cmp = img_np

  psnr_gt = compute_psnr(img_cmp, gt_np)
  ssim_gt = compute_ssim(img_cmp, gt_np)
  ms = dt_s * 1e3
  fps = 1.0 / dt_s
  spm = fps * args.spp * 60.0

  rec = {
      "experiment": "v3_w7_cinema_pt_showcase",
      "scene": args.scene_label,
      "tris": F,
      "tex_atlas_shape": list(tex_atlas_np.shape),
      "n_local_lights": n_local_lights,
      "has_sun": has_sun,
      "exposure": round(exposure, 4),
      "res": f"{H}x{W}",
      "gt_res": f"{gt_h}x{gt_w}",
      "n_chips": n_chips,
      "spp": args.spp,
      "max_bounces": args.max_bounces,
      "compile_s": round(compile_s, 2),
      "ms": round(ms, 2),
      "images_per_sec": round(fps, 3),
      "samples_per_min": round(spm, 1),
      "psnr_vs_cycles_5_3_db": round(psnr_gt, 2),
      "ssim_vs_cycles_5_3": round(ssim_gt, 4),
      "hlo_zero_copy": True,
      "hlo_ops": hlo_info,
  }
  os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
  with open(args.out_json, "w") as f:
    json.dump(rec, f, indent=2)
  print("CINEMA PT:", rec, flush=True)

  # Save side-by-side cinema comparison PNG
  pad, hdr = 14, 42
  canvas = Image.new("RGB", (2 * gt_w + 3 * pad, gt_h + hdr + 2 * pad), (15, 17, 21))
  draw = ImageDraw.Draw(canvas)
  tpu_pil = Image.fromarray(np.clip(img_cmp * 255.0 + 0.5, 0, 255).astype(np.uint8))
  canvas.paste(tpu_pil, (pad, pad + hdr))
  canvas.paste(gt_im, (2 * pad + gt_w, pad + hdr))
  draw.text(
      (pad, pad + 12),
      f"TPU v5p 4c UV-Textured + Denoised ({args.scene_label}, {F:,} tris, {args.spp} spp) | {spm:,.0f} spm | {ms:.0f}ms",
      fill=(238, 244, 250),
  )
  draw.text(
      (2 * pad + gt_w, pad + 12),
      f"Blender 5.3.0 Cycles Reference GT | PSNR {psnr_gt:.2f} dB | SSIM {ssim_gt:.4f}",
      fill=(255, 220, 80),
  )
  os.makedirs(os.path.dirname(args.out_png) or ".", exist_ok=True)
  canvas.save(args.out_png)
  print("Saved Cinema PT PNG ->", args.out_png, canvas.size, flush=True)


if __name__ == "__main__":
  try:
    import bpy  # type: ignore
    _HAS_BPY = True
  except Exception:
    _HAS_BPY = False
  if _HAS_BPY:
    _blender_export_main()
  else:
    _tpu_main()
