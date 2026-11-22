"""Build & export a genuine 100,000,000+ (101.63M) effective triangle Blender 5.2 AgX Gallery + 4K Hero GT.

Uses linked Blender data-blocks (6 unique high-poly UV+PBR BLAS totaling 4.42M unique tris,
instanced 100 times into a Grand Museum Colonnade Rotunda = 101,629,952 rendered tris!):
  - 32x suzanne_hero      (Subsurf-5: 1,007,616 tris each = 32,243,712 tris)
  - 22x torus_knot        (1024x512: 1,048,576 tris each = 23,068,672 tris)
  - 22x wave_sculpture    (1024x512: 1,046,528 tris each = 23,023,616 tris)
  - 22x fluted_column     (1024x512: 1,046,528 tris each = 23,023,616 tris)
  -  1x pedestal          (256x129:     65,536 tris      =     65,536 tris)
  -  1x gallery_floor     (321x321:    204,800 tris      =    204,800 tris)
  Total = 100 instances = 101,629,952 rendered triangles!

Renders Blender 5.2 Cycles GT with AgX view transform + OpenImageDenoise:
  - 48 dynamic frames @ 256x256 (48 spp, 36 train / 4 calib / 8 test) + Frame-0 EXR Z-depth
  -  4 held-out hero showcase views @ 512x512 (64 spp) + Frame-0 EXR Z-depth
  -  1 held-out Native 4K UHD (3840x2160) hero view (32 spp) + 4K EXR Z-depth
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
import time
from typing import Dict, List

import bmesh
import bpy
from mathutils import Vector
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_1m_scene import (
    _build_pedestal,
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
  ap.add_argument("--uhd_samples", type=int, default=32)
  ap.add_argument("--skip_4k_gt", action="store_true")
  return ap.parse_args(argv)


def _build_suzanne_subsurf5() -> bpy.types.Object:
  """Subsurf-5 Suzanne -> 1,007,616 triangles."""
  bpy.ops.mesh.primitive_monkey_add(size=0.34, location=(0, 0, 0))
  obj = bpy.context.active_object
  obj.name = "suzanne_hero"
  bpy.ops.object.mode_set(mode="EDIT")
  bpy.ops.mesh.select_all(action="SELECT")
  bpy.ops.uv.smart_project(angle_limit=1.15, island_margin=0.01)
  bpy.ops.object.mode_set(mode="OBJECT")
  sub = obj.modifiers.new(name="Subsurf", type="SUBSURF")
  sub.levels = 5
  sub.render_levels = 5
  bpy.ops.object.shade_smooth()
  bpy.context.view_layer.objects.active = obj
  bpy.ops.object.modifier_apply(modifier=sub.name)
  return obj


def _build_fluted_column_1m(Nu: int = 1024, Nv: int = 512) -> bpy.types.Object:
  """Ultra-high-poly classical fluted column -> 1,046,528 triangles."""
  theta = np.linspace(0.0, 2.0 * math.pi, Nu, endpoint=False, dtype=np.float64)[:, None]
  z_arr = np.linspace(-0.05, 1.40, Nv, dtype=np.float64)[None, :]
  h_norm = (z_arr + 0.05) / 1.45
  flute = 0.013 * np.maximum(0.0, np.cos(24.0 * theta)) * np.sin(math.pi * np.clip((h_norm - 0.08) / 0.84, 0.0, 1.0))
  base_flare = 0.060 * np.exp(-((h_norm / 0.055) ** 2)) + 0.055 * np.exp(-(((1.0 - h_norm) / 0.055) ** 2))
  entasis = 1.0 - 0.10 * h_norm + 0.04 * np.sin(math.pi * h_norm)
  rad = 0.145 * entasis + base_flare - flute
  pts = np.stack([rad * np.cos(theta), rad * np.sin(theta), np.broadcast_to(z_arr, (Nu, Nv))], axis=-1)
  return _create_mesh_from_grid("fluted_column", pts.astype(np.float32), np.zeros((Nu, Nv, 2), np.float32), wrap_u=True, wrap_v=False)


def _build_large_floor(Nu: int = 321, Nv: int = 321) -> bpy.types.Object:
  xs = np.linspace(-4.2, 4.2, Nu, dtype=np.float32)
  ys = np.linspace(-4.2, 4.2, Nv, dtype=np.float32)
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

  world = bpy.data.worlds.new("Rotunda100MWorld")
  scene.world = world
  world.use_nodes = True
  bg = world.node_tree.nodes.get("Background")
  bg.inputs["Color"].default_value = (0.34, 0.42, 0.54, 1.0)
  bg.inputs["Strength"].default_value = 0.50

  light_pos = np.array([0.85, 0.55, 2.55], dtype=np.float32)
  area_data = bpy.data.lights.new(name="OverheadKey", type="AREA")
  area_data.energy = 115.0
  area_data.size = 1.20
  area_data.color = (1.0, 0.96, 0.90)
  area_obj = bpy.data.objects.new("OverheadKey", area_data)
  bpy.context.collection.objects.link(area_obj)
  area_obj.location = tuple(float(x) for x in light_pos)
  area_obj.rotation_euler = (Vector((0.0, 0.0, 0.28)) - Vector(light_pos)).to_track_quat("-Z", "Y").to_euler()

  fill_data = bpy.data.lights.new(name="SideFill", type="AREA")
  fill_data.energy = 50.0
  fill_data.size = 1.85
  fill_data.color = (0.80, 0.89, 1.0)
  fill_obj = bpy.data.objects.new("SideFill", fill_data)
  bpy.context.collection.objects.link(fill_obj)
  fill_obj.location = (-1.35, -1.25, 1.75)
  fill_obj.rotation_euler = (Vector((0.0, 0.0, 0.25)) - fill_obj.location).to_track_quat("-Z", "Y").to_euler()

  tmpdir = tempfile.mkdtemp(prefix="blender_100m_")
  t_build = time.time()
  tex_map = make_pbr_textures(512)
  tex_map["fluted_column"] = tex_map["gallery_colonnade"]

  suz = _build_suzanne_subsurf5()                 # 1,007,616 tris
  tor = _build_torus_knot(Nu=1024, Nv=512)        # 1,048,576 tris
  wav = _build_wave_sculpture(Nu=1024, Nv=512)    # 1,046,528 tris
  col = _build_fluted_column_1m(Nu=1024, Nv=512)  # 1,046,528 tris
  ped = _build_pedestal(Nu=256, Nv=129)           #    65,536 tris
  flr = _build_large_floor(Nu=321, Nv=321)        #   204,800 tris

  blas_specs = [
      ("suzanne_hero", suz, 0.35, 1.0),
      ("torus_knot", tor, 0.08, 2.0),
      ("wave_sculpture", wav, 0.05, 1.0),
      ("fluted_column", col, 0.0, 2.0),
      ("pedestal", ped, 0.0, 1.5),
      ("gallery_floor", flr, 0.04, 5.0),
  ]

  blas_np = []
  unique_tris = 0
  base_objs = []
  for name, obj, metallic, uv_scale in blas_specs:
    alb, rgh = tex_map[name]
    mat = _make_textured_material(f"Mat100M_{name}", alb, rgh, metallic=metallic, uv_scale=uv_scale, tex_dir=tmpdir)
    obj.data.materials.append(mat)
    v, n, uv, f = _finalize_and_extract_mesh(obj, uv_scale=uv_scale)
    unique_tris += int(f.shape[0])
    blas_np.append((name, v, n, uv, f, alb, rgh, float(metallic)))
    base_objs.append(obj)
    print(f"Unique BLAS '{name}': verts={v.shape[0]:,}, tris={f.shape[0]:,}", flush=True)

  # 100 Linked Data-Block Instances -> 101,629,952 Effective Rendered Triangles!
  # 32 suzannes, 22 torus_knots, 22 wave_sculptures, 22 fluted_columns, 1 pedestal, 1 gallery_floor
  counts = [32, 22, 22, 22, 1, 1]
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
      f"UNIQUE BLAS TRIS: {unique_tris:,} | INSTANCES: {n_inst} | EFFECTIVE RENDERED TRIS: {eff_tris:,} (built in {time.time()-t_build:.2f}s)",
      flush=True,
  )
  assert eff_tris >= 100_000_000, f"Expected >= 100,000,000 tris, got {eff_tris}"

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
  N_uhd = 1
  N_tot = N + N_hero + N_uhd

  inst_xpos = np.zeros((N_tot, n_inst, 3), dtype=np.float32)
  inst_xmat = np.zeros((N_tot, n_inst, 3, 3), dtype=np.float32)
  cam_xpos = np.zeros((N_tot, 3), dtype=np.float32)
  cam_xmat = np.zeros((N_tot, 3, 3), dtype=np.float32)

  # Index map:
  # 0..31: suzanne (0=center hero, 1..22=column-top busts, 23..31=inner plinth busts)
  # 32..53: torus_knot (32=center orbit, 33..53=21 gallery ring sculptures)
  # 54..75: wave_sculpture (54=center orbit, 55..75=21 gallery ring sculptures)
  # 76..97: fluted_column (22 rotunda columns at R=3.05 m)
  # 98: pedestal (center)
  # 99: gallery_floor
  id_m = np.eye(3, dtype=np.float32)
  R_col = 3.05
  R_ring = 2.42

  for i in range(N_tot):
    if i < N:
      az = 2.0 * math.pi * (i / float(N)) - 0.55
      r_cam = 1.68 + 0.08 * math.sin(2.0 * az)
      cz = 0.86 + 0.10 * math.cos(3.0 * az)
      if i == 0:
        az, r_cam, cz = -0.55, 1.68, 0.88
      cp = np.array([r_cam * math.cos(az), r_cam * math.sin(az), cz], dtype=np.float32)
      t = az + 0.55
    elif i < N + N_hero:
      k = i - N
      az = 0.42 + k * (0.5 * math.pi)
      r_cam = 1.62 + 0.06 * (k % 2)
      cz = 0.84 + 0.08 * math.sin(k * 1.1)
      cp = np.array([r_cam * math.cos(az), r_cam * math.sin(az), cz], dtype=np.float32)
      t = az + 1.37
    else:
      # Native 4K UHD cinematic hero angle
      az = -0.25
      r_cam = 1.72
      cz = 0.86
      cp = np.array([r_cam * math.cos(az), r_cam * math.sin(az), cz], dtype=np.float32)
      t = 0.85

    cm = _look_at_cam_mat(cp, np.array([0.0, 0.0, 0.34], dtype=np.float32))
    cam_xpos[i], cam_xmat[i] = cp, cm

    # Central hero trio + central pedestal
    inst_xpos[i, 0] = np.array([0.0, 0.0, 0.35 + 0.025 * math.sin(2.0 * t)], np.float32)
    inst_xmat[i, 0] = _euler_zyx(0.45 + 0.65 * math.sin(t), -0.10 + 0.12 * math.cos(t), 0.08 * math.sin(1.5 * t))

    tk_ang = t + 0.35
    inst_xpos[i, 32] = np.array([0.48 * math.cos(tk_ang), 0.48 * math.sin(tk_ang), 0.33 + 0.04 * math.cos(2.0 * t)], np.float32)
    inst_xmat[i, 32] = _euler_zyx(tk_ang * 1.4, 0.55 * math.sin(t), 0.45 * math.cos(t))

    wv_ang = tk_ang + math.pi
    inst_xpos[i, 54] = np.array([0.48 * math.cos(wv_ang), 0.48 * math.sin(wv_ang), 0.32 + 0.04 * math.sin(2.0 * t)], np.float32)
    inst_xmat[i, 54] = _euler_zyx(-wv_ang * 1.2, 0.35 * math.cos(t), -0.35 * math.sin(t))

    inst_xpos[i, 98] = np.zeros((3,), np.float32)
    inst_xmat[i, 98] = id_m
    inst_xpos[i, 99] = np.zeros((3,), np.float32)
    inst_xmat[i, 99] = id_m

    # 22 Fluted Columns + 22 Bronze Suzanne Capital Busts atop columns
    for k in range(22):
      th = 2.0 * math.pi * k / 22.0 + 0.08
      cx, cy = R_col * math.cos(th), R_col * math.sin(th)
      inst_xpos[i, 76 + k] = np.array([cx, cy, 0.0], np.float32)
      inst_xmat[i, 76 + k] = _euler_zyx(th, 0.0, 0.0)
      inst_xpos[i, 1 + k] = np.array([cx, cy, 1.56], np.float32)
      inst_xmat[i, 1 + k] = _euler_zyx(th + math.pi + 0.2 * math.sin(t + k), 0.0, 0.0)

    # 9 Inner Ring Suzannes (R=2.20 m, low gallery plinth height)
    for k in range(9):
      th = 2.0 * math.pi * k / 9.0 + 0.20
      inst_xpos[i, 23 + k] = np.array([2.20 * math.cos(th), 2.20 * math.sin(th), 0.24 + 0.02 * math.sin(2.0 * t + k)], np.float32)
      inst_xmat[i, 23 + k] = _euler_zyx(th + math.pi + 0.3 * math.sin(t + k), 0.06 * math.cos(t + k), 0.0)

    # 21 Suspended Torus Knots & 21 Wave Sculptures in alternating gallery ring (R=2.42 m)
    for k in range(21):
      th = 2.0 * math.pi * k / 21.0 + 0.12
      inst_xpos[i, 33 + k] = np.array([R_ring * math.cos(th + 0.06), R_ring * math.sin(th + 0.06), 0.72 + 0.05 * math.cos(t + k)], np.float32)
      inst_xmat[i, 33 + k] = _euler_zyx(t + k, 0.6 * math.sin(t + k), 0.4)
      inst_xpos[i, 55 + k] = np.array([R_ring * math.cos(th - 0.06), R_ring * math.sin(th - 0.06), 0.68 + 0.05 * math.sin(t + k)], np.float32)
      inst_xmat[i, 55 + k] = _euler_zyx(-t + k, 0.4 * math.cos(t + k), -0.3)

  png_path = os.path.join(tmpdir, "frame.png")

  def _render_range(indices: List[int], res_w: int, res_h: int, samples: int, save_f0_depth: bool, tag: str):
    scene.render.resolution_x = res_w
    scene.render.resolution_y = res_h
    scene.cycles.samples = samples
    out_rgb = np.zeros((len(indices), res_h, res_w, 3), dtype=np.float32)
    f0_depth = None

    if save_f0_depth:
      bpy.context.view_layer.use_pass_z = True
      ng = bpy.data.node_groups.new(f"Comp100M_{tag}", "CompositorNodeTree")
      scene.compositing_node_group = ng
      rl = ng.nodes.new("CompositorNodeRLayers")
      fout = ng.nodes.new("CompositorNodeOutputFile")
      fout.directory = tmpdir
      fout.file_name = f"depth100m_{tag}_"
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
      px = np.asarray(img.pixels[:], dtype=np.float32).reshape(res_h, res_w, 4)[..., :3][::-1, :, :].copy()
      bpy.data.images.remove(img)
      out_rgb[k] = np.clip(px, 0.0, 1.0)

      if k == 0 and save_f0_depth:
        d_files = sorted(f for f in os.listdir(tmpdir) if f.startswith(f"depth100m_{tag}_") and f.endswith(".exr"))
        if d_files:
          dimg = bpy.data.images.load(os.path.join(tmpdir, d_files[0]), check_existing=False)
          f0_depth = np.asarray(dimg.pixels[:], dtype=np.float32).reshape(res_h, res_w, 4)[..., 0][::-1, :].copy()
          bpy.data.images.remove(dimg)

      if (k + 1) % 8 == 0 or k + 1 == len(indices):
        print(f"Rendered {k+1}/{len(indices)} @ {res_w}x{res_h} AgX ({samples} spp) in {time.time()-t0:.1f}s", flush=True)
    return out_rgb, f0_depth

  gt_256, f0_z_256 = _render_range(list(range(N)), args.res, args.res, args.samples, save_f0_depth=True, tag="256")
  gt_512, f0_z_512 = _render_range(list(range(N, N + N_hero)), args.hero_res, args.hero_res, args.hero_samples, save_f0_depth=True, tag="512")

  if not args.skip_4k_gt:
    gt_4k, f0_z_4k = _render_range([N + N_hero], 3840, 2160, args.uhd_samples, save_f0_depth=True, tag="4k")
  else:
    gt_4k, f0_z_4k = None, None

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

  scene_path = os.path.join(args.out_dir, "v3_w5_blender_100m_scene.npz")
  gt256_path = os.path.join(args.out_dir, "v3_w5_blender_100m_gt_256.npz")
  gt512_path = os.path.join(args.out_dir, "v3_w5_blender_100m_gt_512.npz")
  np.savez_compressed(scene_path, **bundle)
  np.savez_compressed(gt256_path, rgb=gt_256, frame0_z_depth=f0_z_256)
  np.savez_compressed(gt512_path, rgb=gt_512, frame0_z_depth=f0_z_512)
  print(f"Saved {scene_path} ({os.path.getsize(scene_path)/1e6:.1f} MB), {gt256_path}, {gt512_path}", flush=True)

  if gt_4k is not None:
    gt4k_path = os.path.join(args.out_dir, "v3_w5_blender_100m_gt_4k.npz")
    np.savez_compressed(gt4k_path, rgb=(gt_4k[0] * 255.0 + 0.5).astype(np.uint8), frame0_z_depth=f0_z_4k)
    print(f"Saved {gt4k_path} ({os.path.getsize(gt4k_path)/1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
  main()
