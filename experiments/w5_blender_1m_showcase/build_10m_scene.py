"""Build & export a genuine 10,000,000+ (11.40M) effective triangle BLAS-instanced Blender 5.2 AgX Gallery.

Uses linked Blender data-blocks (6 unique high-res UV-textured BLAS totaling ~1.10M unique tris,
instanced 65 times into an 11-alcove, 16-column Museum Sculpture Rotunda = 11,399,168 rendered tris!):
  - 12x suzanne_hero      (251,904 tris each = 3,022,848 tris) - Hammered Bronze + Patina
  - 12x torus_knot        (196,608 tris each = 2,359,296 tris) - Glazed Cobalt-Teal Ceramic
  - 12x wave_sculpture    (261,120 tris each = 3,133,440 tris) - Veined Terracotta Marble
  - 16x fluted_column     (130,560 tris each = 2,088,960 tris) - Warm Travertine Limestone
  - 12x pedestal          ( 49,152 tris each =   589,824 tris) - Dark Walnut Wood Lathe
  -  1x gallery_floor     (204,800 tris      =   204,800 tris) - Calacatta/Nero Marble Floor
  Total = 11,399,168 rendered triangles!

Renders film-grade Blender 5.2 Cycles GT with AgX view transform + OpenImageDenoise
at 256x256 (48 frames: 36 train / 4 calib / 8 test) and 512x512 (4 held-out showcase views)
plus Frame-0 32-bit EXR Z-depth for exact metric verification.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
import time
from typing import Dict, List, Tuple

import bmesh
import bpy
from mathutils import Matrix, Vector
import numpy as np

# Reuse exact texture & mesh helpers from build_1m_scene.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_1m_scene import (
    _build_pedestal,
    _build_suzanne_hero,
    _build_torus_knot,
    _build_wave_sculpture,
    _create_mesh_from_grid,
    _euler_zyx,
    _finalize_and_extract_mesh,
    _look_at_cam_mat,
    _make_textured_material,
    _set_obj_pose,
    make_pbr_textures,
)


def _parse_args():
  argv = sys.argv
  argv = argv[argv.index("--") + 1 :] if "--" in argv else []
  ap = argparse.ArgumentParser()
  ap.add_argument("--out_dir", default="out/metrics")
  ap.add_argument("--n_frames", type=int, default=48)
  ap.add_argument("--n_hero", type=int, default=4)
  ap.add_argument("--res", type=int, default=256)
  ap.add_argument("--hero_res", type=int, default=512)
  ap.add_argument("--samples", type=int, default=48)
  ap.add_argument("--hero_samples", type=int, default=64)
  ap.add_argument("--seed", type=int, default=2027)
  return ap.parse_args(argv)


def _build_single_fluted_column(Nu: int = 256, Nv: int = 256) -> bpy.types.Object:
  """Single high-res classical fluted column (~130,560 tris per BLAS instance)."""
  theta = np.linspace(0.0, 2.0 * math.pi, Nu, endpoint=False, dtype=np.float64)[:, None]
  z_arr = np.linspace(-0.05, 1.35, Nv, dtype=np.float64)[None, :]
  h_norm = (z_arr + 0.05) / 1.40
  flute = 0.013 * np.maximum(0.0, np.cos(24.0 * theta)) * np.sin(math.pi * np.clip((h_norm - 0.08) / 0.84, 0.0, 1.0))
  base_flare = 0.060 * np.exp(-((h_norm / 0.055) ** 2)) + 0.055 * np.exp(-(((1.0 - h_norm) / 0.055) ** 2))
  entasis = 1.0 - 0.10 * h_norm + 0.04 * np.sin(math.pi * h_norm)
  rad = 0.145 * entasis + base_flare - flute
  pts = np.stack([rad * np.cos(theta), rad * np.sin(theta), np.broadcast_to(z_arr, (Nu, Nv))], axis=-1)
  return _create_mesh_from_grid("fluted_column", pts.astype(np.float32), np.zeros((Nu, Nv, 2), np.float32), wrap_u=True, wrap_v=False)


def _build_large_gallery_floor(Nu: int = 321, Nv: int = 321) -> bpy.types.Object:
  """320x320 tessellated marble rotunda floor (204,800 tris) spanning [-3.8, 3.8]^2 at z=-0.05."""
  xs = np.linspace(-3.8, 3.8, Nu, dtype=np.float32)
  ys = np.linspace(-3.8, 3.8, Nv, dtype=np.float32)
  gx, gy = np.meshgrid(xs, ys, indexing="ij")
  pts = np.stack([gx, gy, np.full_like(gx, -0.05)], axis=-1)
  return _create_mesh_from_grid("gallery_floor", pts, np.zeros((Nu, Nv, 2), np.float32), wrap_u=False, wrap_v=False)


def main():
  args = _parse_args()
  os.makedirs(args.out_dir, exist_ok=True)

  bpy.ops.wm.read_factory_settings(use_empty=True)
  scene = bpy.context.scene
  scene.render.engine = "CYCLES"
  scene.cycles.device = "CPU"
  scene.cycles.use_denoising = True
  scene.cycles.max_bounces = 6
  scene.cycles.diffuse_bounces = 4
  scene.cycles.glossy_bounces = 4
  scene.cycles.use_adaptive_sampling = True
  scene.cycles.adaptive_threshold = 0.02

  fovy = 44.0
  scene.render.resolution_percentage = 100
  scene.render.image_settings.file_format = "PNG"
  scene.render.image_settings.color_mode = "RGB"
  scene.render.image_settings.color_depth = "16"
  scene.view_settings.view_transform = "AgX"
  scene.view_settings.look = "None"
  scene.view_settings.exposure = 0.0

  # Warm Gallery Rotunda Lighting
  world = bpy.data.worlds.new("RotundaWorld")
  scene.world = world
  world.use_nodes = True
  bg = world.node_tree.nodes.get("Background")
  bg.inputs["Color"].default_value = (0.34, 0.42, 0.54, 1.0)
  bg.inputs["Strength"].default_value = 0.50

  light_pos = np.array([0.85, 0.55, 2.55], dtype=np.float32)
  area_data = bpy.data.lights.new(name="OverheadKey", type="AREA")
  area_data.energy = 110.0
  area_data.size = 1.15
  area_data.color = (1.0, 0.96, 0.90)
  area_obj = bpy.data.objects.new("OverheadKey", area_data)
  bpy.context.collection.objects.link(area_obj)
  area_obj.location = tuple(float(x) for x in light_pos)
  area_obj.rotation_euler = (Vector((0.0, 0.0, 0.28)) - Vector(light_pos)).to_track_quat("-Z", "Y").to_euler()

  fill_data = bpy.data.lights.new(name="SideFill", type="AREA")
  fill_data.energy = 48.0
  fill_data.size = 1.80
  fill_data.color = (0.80, 0.89, 1.0)
  fill_obj = bpy.data.objects.new("SideFill", fill_data)
  bpy.context.collection.objects.link(fill_obj)
  fill_obj.location = (-1.35, -1.25, 1.75)
  fill_obj.rotation_euler = (Vector((0.0, 0.0, 0.25)) - fill_obj.location).to_track_quat("-Z", "Y").to_euler()

  tmpdir = tempfile.mkdtemp(prefix="blender_10m_")
  t_build = time.time()
  tex_map = make_pbr_textures(512)

  # Build 6 unique BLAS meshes
  suz = _build_suzanne_hero()
  tor = _build_torus_knot()
  wav = _build_wave_sculpture()
  col = _build_single_fluted_column()
  ped = _build_pedestal()
  flr = _build_large_gallery_floor()

  # Map texture keys (reuse travertine for fluted_column)
  tex_map["fluted_column"] = tex_map["gallery_colonnade"]
  blas_specs = [
      ("suzanne_hero", suz, 0.35, 1.0),
      ("torus_knot", tor, 0.08, 2.0),
      ("wave_sculpture", wav, 0.05, 1.0),
      ("fluted_column", col, 0.0, 2.0),
      ("pedestal", ped, 0.0, 1.5),
      ("gallery_floor", flr, 0.04, 4.5),
  ]

  blas_np = []
  unique_tris = 0
  base_objs = []
  for name, obj, metallic, uv_scale in blas_specs:
    alb, rgh = tex_map[name]
    mat = _make_textured_material(f"Mat10M_{name}", alb, rgh, metallic=metallic, uv_scale=uv_scale, tex_dir=tmpdir)
    obj.data.materials.append(mat)
    v, n, uv, f = _finalize_and_extract_mesh(obj, uv_scale=uv_scale)
    unique_tris += int(f.shape[0])
    blas_np.append((name, v, n, uv, f, alb, rgh, float(metallic)))
    base_objs.append(obj)
    print(f"Unique BLAS '{name}': verts={v.shape[0]:,}, tris={f.shape[0]:,}", flush=True)

  # Create linked data-block instances totaling 11,399,168 rendered triangles!
  # Instances grouped by mesh_id:
  # 0: 12 suzannes, 1: 12 torus_knots, 2: 12 wave_sculptures, 3: 16 fluted_columns, 4: 12 pedestals, 5: 1 gallery_floor
  counts = [12, 12, 12, 16, 12, 1]
  inst_objs: List[bpy.types.Object] = []
  inst_mesh_ids: List[int] = []
  eff_tris = 0
  for m_id, (base_obj, cnt) in enumerate(zip(base_objs, counts)):
    eff_tris += cnt * int(blas_np[m_id][4].shape[0])
    inst_objs.append(base_obj)
    inst_mesh_ids.append(m_id)
    for k in range(1, cnt):
      dup = bpy.data.objects.new(f"{base_obj.name}_inst_{k}", base_obj.data)
      bpy.context.collection.objects.link(dup)
      inst_objs.append(dup)
      inst_mesh_ids.append(m_id)

  n_inst = len(inst_objs)
  print(
      f"TOTAL UNIQUE BLAS TRIS: {unique_tris:,} | INSTANCES: {n_inst} | EFFECTIVE RENDERED TRIS: {eff_tris:,} (built in {time.time()-t_build:.2f}s)",
      flush=True,
  )
  assert eff_tris >= 10_000_000, f"Expected >= 10,000,000 tris, got {eff_tris}"

  # Camera setup
  cam_data = bpy.data.cameras.new("MainCam")
  cam_data.sensor_fit = "VERTICAL"
  cam_data.sensor_width = 36.0
  cam_data.sensor_height = 36.0
  cam_data.lens_unit = "FOV"
  cam_data.angle = math.radians(fovy)
  cam_data.clip_start = 0.05
  cam_data.clip_end = 50.0
  cam_obj = bpy.data.objects.new("MainCam", cam_data)
  bpy.context.collection.objects.link(cam_obj)
  scene.camera = cam_obj

  N = args.n_frames
  N_hero = args.n_hero
  N_tot = N + N_hero

  inst_xpos = np.zeros((N_tot, n_inst, 3), dtype=np.float32)
  inst_xmat = np.zeros((N_tot, n_inst, 3, 3), dtype=np.float32)
  cam_xpos = np.zeros((N_tot, 3), dtype=np.float32)
  cam_xmat = np.zeros((N_tot, 3, 3), dtype=np.float32)

  # Precompute instance index slices
  # 0..11: suzanne, 12..23: torus, 24..35: wave, 36..51: col, 52..63: ped, 64: floor
  id_m = np.eye(3, dtype=np.float32)
  n_alcoves = 11
  R_alc = 2.38
  R_col = 2.95

  for i in range(N_tot):
    if i < N:
      az = 2.0 * math.pi * (i / float(N)) - 0.55
      r_cam = 1.66 + 0.08 * math.sin(2.0 * az)
      cz = 0.86 + 0.10 * math.cos(3.0 * az)
      if i == 0:
        az = -0.55
        r_cam = 1.66
        cz = 0.88
      cp = np.array([r_cam * math.cos(az), r_cam * math.sin(az), cz], dtype=np.float32)
      t = az + 0.55
    else:
      k = i - N
      az = 0.42 + k * (0.5 * math.pi)
      r_cam = 1.62 + 0.06 * (k % 2)
      cz = 0.84 + 0.08 * math.sin(k * 1.1)
      cp = np.array([r_cam * math.cos(az), r_cam * math.sin(az), cz], dtype=np.float32)
      t = az + 1.37

    cm = _look_at_cam_mat(cp, np.array([0.0, 0.0, 0.34], dtype=np.float32))
    cam_xpos[i], cam_xmat[i] = cp, cm

    # Central hero trio + central pedestal
    inst_xpos[i, 0] = np.array([0.0, 0.0, 0.35 + 0.025 * math.sin(2.0 * t)], np.float32)
    inst_xmat[i, 0] = _euler_zyx(0.45 + 0.65 * math.sin(t), -0.10 + 0.12 * math.cos(t), 0.08 * math.sin(1.5 * t))

    tk_ang = t + 0.35
    inst_xpos[i, 12] = np.array([0.48 * math.cos(tk_ang), 0.48 * math.sin(tk_ang), 0.33 + 0.04 * math.cos(2.0 * t)], np.float32)
    inst_xmat[i, 12] = _euler_zyx(tk_ang * 1.4, 0.55 * math.sin(t), 0.45 * math.cos(t))

    wv_ang = tk_ang + math.pi
    inst_xpos[i, 24] = np.array([0.48 * math.cos(wv_ang), 0.48 * math.sin(wv_ang), 0.32 + 0.04 * math.sin(2.0 * t)], np.float32)
    inst_xmat[i, 24] = _euler_zyx(-wv_ang * 1.2, 0.35 * math.cos(t), -0.35 * math.sin(t))

    inst_xpos[i, 52] = np.zeros((3,), np.float32)
    inst_xmat[i, 52] = id_m

    # 11 Museum Gallery Alcoves (Suzanne + Pedestal + elevated orbiting Torus Knot & Wave Sculpture)
    for k in range(n_alcoves):
      th = 2.0 * math.pi * k / float(n_alcoves) + 0.15
      ax, ay = R_alc * math.cos(th), R_alc * math.sin(th)
      # Satellite Pedestal
      inst_xpos[i, 53 + k] = np.array([ax, ay, 0.0], np.float32)
      inst_xmat[i, 53 + k] = id_m
      # Satellite Suzanne facing inward toward center + gentle motion
      inst_xpos[i, 1 + k] = np.array([ax, ay, 0.35 + 0.02 * math.sin(2.0 * t + k)], np.float32)
      inst_xmat[i, 1 + k] = _euler_zyx(th + math.pi + 0.3 * math.sin(t + k), 0.08 * math.cos(t + k), 0.0)
      # Elevated flanking Torus Knot & Wave Sculpture
      inst_xpos[i, 13 + k] = np.array([2.48 * math.cos(th + 0.13), 2.48 * math.sin(th + 0.13), 0.68 + 0.05 * math.cos(t + k)], np.float32)
      inst_xmat[i, 13 + k] = _euler_zyx(t + k, 0.6 * math.sin(t + k), 0.4)
      inst_xpos[i, 25 + k] = np.array([2.48 * math.cos(th - 0.13), 2.48 * math.sin(th - 0.13), 0.66 + 0.05 * math.sin(t + k)], np.float32)
      inst_xmat[i, 25 + k] = _euler_zyx(-t + k, 0.4 * math.cos(t + k), -0.3)

    # 16 Classical Fluted Columns around outer rotunda
    for k in range(16):
      th = 2.0 * math.pi * k / 16.0 + 0.08
      inst_xpos[i, 36 + k] = np.array([R_col * math.cos(th), R_col * math.sin(th), 0.0], np.float32)
      inst_xmat[i, 36 + k] = _euler_zyx(th, 0.0, 0.0)

    # Floor
    inst_xpos[i, 64] = np.zeros((3,), np.float32)
    inst_xmat[i, 64] = id_m

  png_path = os.path.join(tmpdir, "frame.png")

  def _render_range(indices: List[int], res: int, samples: int, save_f0_depth: bool):
    scene.render.resolution_x = res
    scene.render.resolution_y = res
    scene.cycles.samples = samples
    out_rgb = np.zeros((len(indices), res, res, 3), dtype=np.float32)
    f0_depth = None

    if save_f0_depth:
      bpy.context.view_layer.use_pass_z = True
      ng = bpy.data.node_groups.new(f"Comp10M_{res}", "CompositorNodeTree")
      scene.compositing_node_group = ng
      rl = ng.nodes.new("CompositorNodeRLayers")
      fout = ng.nodes.new("CompositorNodeOutputFile")
      fout.directory = tmpdir
      fout.file_name = f"depth10m_{res}_"
      fout.format.media_type = "IMAGE"
      fout.format.file_format = "OPEN_EXR"
      fout.format.color_depth = "32"
      fout.file_output_items.clear()
      fout.file_output_items.new("FLOAT", "Depth")
      ng.links.new(rl.outputs["Depth"], fout.inputs["Depth"])

    t0 = time.time()
    for k, fi in enumerate(indices):
      for idx_o, obj in enumerate(inst_objs):
        _set_obj_pose(obj, inst_xpos[fi, idx_o], inst_xmat[fi, idx_o])
      _set_obj_pose(cam_obj, cam_xpos[fi], cam_xmat[fi])

      if k == 1 and save_f0_depth:
        scene.compositing_node_group = None

      scene.render.filepath = png_path
      bpy.ops.render.render(write_still=True)

      img = bpy.data.images.load(png_path, check_existing=False)
      px = np.asarray(img.pixels[:], dtype=np.float32).reshape(res, res, 4)[..., :3][::-1, :, :].copy()
      bpy.data.images.remove(img)
      out_rgb[k] = np.clip(px, 0.0, 1.0)

      if k == 0 and save_f0_depth:
        d_files = sorted(f for f in os.listdir(tmpdir) if f.startswith(f"depth10m_{res}_") and f.endswith(".exr"))
        if d_files:
          dimg = bpy.data.images.load(os.path.join(tmpdir, d_files[0]), check_existing=False)
          f0_depth = np.asarray(dimg.pixels[:], dtype=np.float32).reshape(res, res, 4)[..., 0][::-1, :].copy()
          bpy.data.images.remove(dimg)

      if (k + 1) % 8 == 0 or k + 1 == len(indices):
        print(f"Rendered {k+1}/{len(indices)} @ {res}x{res} AgX ({samples} spp) in {time.time()-t0:.1f}s", flush=True)
    return out_rgb, f0_depth

  gt_256, f0_z_256 = _render_range(list(range(N)), args.res, args.samples, save_f0_depth=True)
  gt_512, f0_z_512 = _render_range(list(range(N, N + N_hero)), args.hero_res, args.hero_samples, save_f0_depth=True)

  split = np.zeros((N,), dtype=np.int32)
  for tid in [5, 11, 17, 23, 29, 35, 41, 47]:
    if tid < N:
      split[tid] = 2
  for cid in [3, 15, 27, 39]:
    if cid < N:
      split[cid] = 1

  bundle = {
      "n_meshes": np.int32(len(blas_np)),
      "n_instances": np.int32(n_inst),
      "unique_tris": np.int32(unique_tris),
      "effective_tris": np.int32(eff_tris),
      "inst_mesh_id": np.asarray(inst_mesh_ids, dtype=np.int32),
      "height": np.int32(args.res),
      "width": np.int32(args.res),
      "hero_height": np.int32(args.hero_res),
      "hero_width": np.int32(args.hero_res),
      "cam_fovy": np.float32(fovy),
      "znear": np.float32(0.05),
      "zfar": np.float32(50.0),
      "light_pos": light_pos,
      "inst_xpos": inst_xpos,
      "inst_xmat": inst_xmat,
      "cam_xpos": cam_xpos,
      "cam_xmat": cam_xmat,
      "split_256": split,
  }
  for idx, (name, v, n, uv, f, alb, rgh, met) in enumerate(blas_np):
    bundle[f"mesh_{idx}_name"] = np.asarray(name)
    bundle[f"mesh_{idx}_verts"] = v
    bundle[f"mesh_{idx}_normals"] = n
    bundle[f"mesh_{idx}_uvs"] = uv
    bundle[f"mesh_{idx}_faces"] = f
    bundle[f"mesh_{idx}_tex_alb"] = (alb * 255.0 + 0.5).astype(np.uint8)
    bundle[f"mesh_{idx}_tex_rgh"] = (rgh * 255.0 + 0.5).astype(np.uint8)
    bundle[f"mesh_{idx}_metallic"] = np.float32(met)

  scene_path = os.path.join(args.out_dir, "v3_w5_blender_10m_scene.npz")
  gt256_path = os.path.join(args.out_dir, "v3_w5_blender_10m_gt_256.npz")
  gt512_path = os.path.join(args.out_dir, "v3_w5_blender_10m_gt_512.npz")
  np.savez_compressed(scene_path, **bundle)
  np.savez_compressed(gt256_path, rgb=gt_256, frame0_z_depth=f0_z_256)
  np.savez_compressed(gt512_path, rgb=gt_512, frame0_z_depth=f0_z_512)
  print(f"Saved {scene_path} ({os.path.getsize(scene_path)/1e6:.1f} MB), {gt256_path}, {gt512_path}", flush=True)


if __name__ == "__main__":
  main()
