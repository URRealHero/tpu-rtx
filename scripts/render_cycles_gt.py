"""Render high-fidelity Blender 5.2 Cycles path-traced GT RGB for all 384 PandaPickCube poses.

Runs locally inside:
  /opt/homebrew/bin/blender --background --factory-startup --python scripts/render_cycles_gt.py -- ...
Verifies sub-pixel silhouette/depth alignment with Stage 1 G-buffer on frame 0 before rendering.
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
import time

import bpy
import bmesh
from mathutils import Matrix, Vector
import numpy as np


def _parse_args():
  argv = sys.argv
  if "--" in argv:
    argv = argv[argv.index("--") + 1 :]
  else:
    argv = []
  import argparse
  ap = argparse.ArgumentParser()
  ap.add_argument("--dataset", default="out/data/panda_cycles_dataset.npz")
  ap.add_argument("--out", default="out/data/panda_cycles_gt_rgb.npy")
  ap.add_argument("--samples", type=int, default=48)
  ap.add_argument("--max_frames", type=int, default=0)
  return ap.parse_args(argv)


def make_material(name: str, rgba: tuple, roughness: float = 0.35, metallic: float = 0.0):
  mat = bpy.data.materials.new(name=name)
  mat.use_nodes = True
  nt = mat.node_tree
  bsdf = nt.nodes.get("Principled BSDF")
  bsdf.inputs["Base Color"].default_value = (rgba[0], rgba[1], rgba[2], 1.0)
  bsdf.inputs["Roughness"].default_value = roughness
  bsdf.inputs["Metallic"].default_value = metallic
  return mat


def make_checker_material(name: str, rgb1, rgb2, repeat=(5.0, 5.0)):
  mat = bpy.data.materials.new(name=name)
  mat.use_nodes = True
  nt = mat.node_tree
  nodes = nt.nodes
  links = nt.links
  bsdf = nodes.get("Principled BSDF")
  bsdf.inputs["Roughness"].default_value = 0.55

  tex_coord = nodes.new("ShaderNodeTexCoord")
  mapping = nodes.new("ShaderNodeMapping")
  # MuJoCo plane checker uses local XY * texrepeat
  mapping.inputs["Scale"].default_value = (repeat[0], repeat[1], 1.0)
  checker = nodes.new("ShaderNodeTexChecker")
  checker.inputs["Color1"].default_value = (rgb1[0], rgb1[1], rgb1[2], 1.0)
  checker.inputs["Color2"].default_value = (rgb2[0], rgb2[1], rgb2[2], 1.0)
  checker.inputs["Scale"].default_value = 1.0

  links.new(tex_coord.outputs["Object"], mapping.inputs["Vector"])
  links.new(mapping.outputs["Vector"], checker.inputs["Vector"])
  links.new(checker.outputs["Color"], bsdf.inputs["Base Color"])
  return mat


def create_capsule_mesh(name: str, radius: float, half_len: float, u_seg: int = 20, v_seg: int = 12):
  mesh = bpy.data.meshes.new(name)
  bm = bmesh.new()
  # Build capsule aligned along local Z from -(half_len+radius) to +(half_len+radius)
  # Use UV sphere stretched/split into two hemispheres + cylinder
  bmesh.ops.create_uvsphere(bm, u_segments=u_seg, v_segments=v_seg, radius=radius)
  for v in bm.verts:
    if v.co.z > 0.0:
      v.co.z += half_len
    elif v.co.z < 0.0:
      v.co.z -= half_len
  bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
  bm.to_mesh(mesh)
  bm.free()
  for p in mesh.polygons:
    p.use_smooth = True
  return mesh


def build_geom_object(gid: int, gtype: int, size: np.ndarray, rgba: np.ndarray, is_checker: bool, ds):
  name = f"geom_{gid}"
  if gtype == 0:  # plane
    sx = float(size[0]) if size[0] > 0 else 8.0
    sy = float(size[1]) if size[1] > 0 else 8.0
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=1.0)
    for v in bm.verts:
      v.co.x *= sx
      v.co.y *= sy
    bm.to_mesh(mesh)
    bm.free()
  elif gtype == 2:  # sphere
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=24, v_segments=14, radius=float(size[0]))
    bm.to_mesh(mesh)
    bm.free()
    for p in mesh.polygons:
      p.use_smooth = True
  elif gtype == 3:  # capsule
    mesh = create_capsule_mesh(name, float(size[0]), float(size[1]))
  elif gtype == 4:  # ellipsoid
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=24, v_segments=14, radius=1.0)
    for v in bm.verts:
      v.co.x *= float(size[0])
      v.co.y *= float(size[1])
      v.co.z *= float(size[2])
    bm.to_mesh(mesh)
    bm.free()
    for p in mesh.polygons:
      p.use_smooth = True
  elif gtype == 5:  # cylinder
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=28,
                          radius1=float(size[0]), radius2=float(size[0]), depth=2.0 * float(size[1]))
    bm.to_mesh(mesh)
    bm.free()
    for p in mesh.polygons:
      if abs(p.normal.z) < 0.5:
        p.use_smooth = True
  elif gtype == 6:  # box
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=2.0)
    for v in bm.verts:
      v.co.x *= float(size[0])
      v.co.y *= float(size[1])
      v.co.z *= float(size[2])
    bm.to_mesh(mesh)
    bm.free()
  else:
    return None

  obj = bpy.data.objects.new(name, mesh)
  bpy.context.collection.objects.link(obj)
  if is_checker:
    mat = make_checker_material(f"mat_{gid}", ds["checker_rgb1"], ds["checker_rgb2"], repeat=(2.0, 2.0))
  else:
    # Slightly glossy ceramic/plastic robot links, matte floor/strip, satin cube
    rough = 0.28 if gtype in (2, 3, 5) else 0.42
    obj.data.materials.append(make_material(f"mat_{gid}", tuple(rgba), roughness=rough))
    return obj
  obj.data.materials.append(mat)
  return obj


def set_obj_pose(obj, pos: np.ndarray, mat3: np.ndarray):
  m4 = Matrix((
      (float(mat3[0, 0]), float(mat3[0, 1]), float(mat3[0, 2]), float(pos[0])),
      (float(mat3[1, 0]), float(mat3[1, 1]), float(mat3[1, 2]), float(pos[1])),
      (float(mat3[2, 0]), float(mat3[2, 1]), float(mat3[2, 2]), float(pos[2])),
      (0.0, 0.0, 0.0, 1.0),
  ))
  obj.matrix_world = m4


def setup_studio_lighting_and_world(ds):
  scene = bpy.context.scene
  # Studio world sky gradient
  world = bpy.data.worlds.new("StudioWorld")
  scene.world = world
  world.use_nodes = True
  wn = world.node_tree.nodes
  wl = world.node_tree.links
  bg = wn.get("Background")
  bg.inputs["Color"].default_value = (0.28, 0.36, 0.48, 1.0)
  bg.inputs["Strength"].default_value = 0.22

  # Main overhead soft Area Light (produces soft shadows + contact AO + indirect bounce)
  area_data = bpy.data.lights.new(name="OverheadArea", type="AREA")
  area_data.energy = 26.0
  area_data.size = 0.85
  area_data.color = (1.0, 0.97, 0.92)
  area_obj = bpy.data.objects.new("OverheadArea", area_data)
  bpy.context.collection.objects.link(area_obj)
  area_obj.location = (0.65, 0.25, 1.85)
  direction = Vector((0.55, 0.0, 0.10)) - area_obj.location
  area_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()

  # Warm bounce rim/fill light for crisp 3D shading and color bleeding
  fill_data = bpy.data.lights.new(name="SideFill", type="AREA")
  fill_data.energy = 10.0
  fill_data.size = 1.2
  fill_data.color = (0.78, 0.88, 1.0)
  fill_obj = bpy.data.objects.new("SideFill", fill_data)
  bpy.context.collection.objects.link(fill_obj)
  fill_obj.location = (0.95, -1.1, 0.95)
  direction2 = Vector((0.55, 0.0, 0.15)) - fill_obj.location
  fill_obj.rotation_euler = direction2.to_track_quat("-Z", "Y").to_euler()


def main():
  args = _parse_args()
  ds = np.load(args.dataset)
  N = ds["geom_xpos"].shape[0]
  if args.max_frames > 0:
    N = min(N, args.max_frames)
  H = int(ds["height"])
  W = int(ds["width"])
  fovy = float(ds["cam_fovy"])

  bpy.ops.wm.read_factory_settings(use_empty=True)
  scene = bpy.context.scene
  scene.render.engine = "CYCLES"
  scene.cycles.device = "CPU"
  scene.cycles.samples = args.samples
  scene.cycles.use_denoising = True
  scene.cycles.max_bounces = 6
  scene.cycles.diffuse_bounces = 4
  scene.cycles.glossy_bounces = 4
  scene.cycles.use_adaptive_sampling = True
  scene.cycles.adaptive_threshold = 0.02

  scene.render.resolution_x = W
  scene.render.resolution_y = H
  scene.render.resolution_percentage = 100
  scene.render.image_settings.file_format = "OPEN_EXR"
  scene.render.image_settings.color_mode = "RGB"
  scene.render.image_settings.color_depth = "32"
  scene.view_settings.view_transform = "Standard"
  scene.view_settings.look = "None"
  scene.view_settings.exposure = 0.0

  # Camera aligned to MuJoCo fovy and cam_xmat
  cam_data = bpy.data.cameras.new("MainCam")
  cam_data.sensor_fit = "VERTICAL"
  cam_data.sensor_width = 36.0
  cam_data.sensor_height = 36.0
  cam_data.lens_unit = "FOV"
  cam_data.angle = math.radians(fovy)
  cam_data.clip_start = 0.01
  cam_data.clip_end = 100.0
  cam_obj = bpy.data.objects.new("MainCam", cam_data)
  bpy.context.collection.objects.link(cam_obj)
  scene.camera = cam_obj

  setup_studio_lighting_and_world(ds)

  # Build enabled analytic geom objects
  geom_objs = {}
  enabled = ds["enabled_geom_ids"]
  for gid in enabled:
    gid = int(gid)
    gtype = int(ds["geom_type"][gid])
    obj = build_geom_object(
        gid,
        gtype,
        ds["geom_size"][gid],
        ds["geom_rgba"][gid],
        bool(ds["geom_is_checker"][gid]),
        ds,
    )
    if obj is not None:
      geom_objs[gid] = obj

  # Enable Z depth pass for frame-0 alignment check
  vl = bpy.context.view_layer
  vl.use_pass_z = True

  out_rgb = np.zeros((N, H, W, 3), dtype=np.float32)
  tmpdir = tempfile.mkdtemp(prefix="cycles_gt_")
  exr_path = os.path.join(tmpdir, "frame.exr")

  # Compositor setup to inspect frame 0 Z-depth too
  ng = bpy.data.node_groups.new("Comp", "CompositorNodeTree")
  scene.compositing_node_group = ng
  rl = ng.nodes.new("CompositorNodeRLayers")
  fo = ng.nodes.new("CompositorNodeOutputFile")
  fo.directory = tmpdir
  fo.file_name = "depth_"
  fo.format.media_type = "IMAGE"
  fo.format.file_format = "OPEN_EXR"
  fo.format.color_depth = "32"
  fo.file_output_items.clear()
  fo.file_output_items.new("FLOAT", "Depth")
  ng.links.new(rl.outputs["Depth"], fo.inputs["Depth"])

  t0 = time.time()
  for i in range(N):
    for gid, obj in geom_objs.items():
      set_obj_pose(obj, ds["geom_xpos"][i, gid], ds["geom_xmat"][i, gid])
    set_obj_pose(cam_obj, ds["cam_xpos"][i], ds["cam_xmat"][i])

    # Disable compositor depth dump after frame 0 for speed
    if i == 1:
      scene.compositing_node_group = None

    scene.render.filepath = exr_path
    bpy.ops.render.render(write_still=True)

    img = bpy.data.images.load(exr_path, check_existing=False)
    px = np.asarray(img.pixels[:], dtype=np.float32).reshape(H, W, 4)[..., :3]
    # Blender image buffer is bottom-to-top; flip vertically to top-to-bottom
    px = px[::-1, :, :].copy()
    bpy.data.images.remove(img)
    # Tone-map linear radiance cleanly and encode to sRGB [0, 1]
    px = np.maximum(px, 0.0)
    px = px / (1.0 + 0.15 * px)
    px = np.where(px <= 0.0031308, 12.92 * px, 1.055 * (np.maximum(px, 1e-8) ** (1.0 / 2.4)) - 0.055)
    out_rgb[i] = np.clip(px, 0.0, 1.0)

    if i == 0:
      # Sub-pixel Z-depth alignment check vs Stage 1 z_depth
      d_files = [f for f in os.listdir(tmpdir) if f.startswith("depth_") and f.endswith(".exr")]
      if d_files:
        dimg = bpy.data.images.load(os.path.join(tmpdir, d_files[0]), check_existing=False)
        dpx = np.asarray(dimg.pixels[:], dtype=np.float32).reshape(H, W, 4)[..., 0][::-1, :]
        bpy.data.images.remove(dimg)
        s1_zd = ds["s1_z_depth"][0]
        valid = (ds["s1_geom_id"][0] >= 0) & (dpx < 1e4)
        z_mae = float(np.mean(np.abs(dpx[valid] - s1_zd[valid])))
        fg_iou = float(np.mean((dpx < 10.0) == (s1_zd > 0.0)))
        print(f"FRAME 0 ALIGNMENT: z_depth_mae={z_mae:.5f} m, fg_mask_match={fg_iou:.4f}", flush=True)
        assert z_mae < 0.02, f"Misaligned Blender vs Stage 1 camera! z_mae={z_mae}"

    if (i + 1) % 32 == 0 or i + 1 == N:
      dt = time.time() - t0
      print(f"Rendered {i+1}/{N} Cycles GT frames ({dt:.1f}s, {dt/(i+1):.2f}s/frame)", flush=True)

  os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
  np.save(args.out, out_rgb)
  print(f"Saved Cycles GT RGB {out_rgb.shape} to {args.out}")


if __name__ == "__main__":
  main()
