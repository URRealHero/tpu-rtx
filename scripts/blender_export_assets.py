"""Export real curved Blender 5.2 multi-mesh scene + render 48-spp Cycles GT & Z-depth.

Creates a genuine Blender multi-mesh studio scene:
- Subdivided Suzanne monkey head (~2k tris, smooth shaded, metallic-terracotta)
- Smooth Torus ring (~576 tris, teal)
- Beveled tabletop pedestal (dark slate)
Animates rigid poses across N_frames (default 48: 36 train / 4 calib / 8 test),
exports local triangulated vertices/faces/normals + per-frame poses to
`out/assets/blender_real_scene.npz`, renders 48-spp Cycles GT RGB + Frame-0
EXR Z-depth to `out/assets/blender_real_scene_gt.npz`.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
import time

import bmesh
import bpy
from mathutils import Matrix, Vector
import numpy as np


def _parse_args():
  argv = sys.argv
  argv = argv[argv.index("--") + 1 :] if "--" in argv else []
  ap = argparse.ArgumentParser()
  ap.add_argument("--out_dir", default="out/assets")
  ap.add_argument("--n_frames", type=int, default=48)
  ap.add_argument("--res", type=int, default=64)
  ap.add_argument("--samples", type=int, default=48)
  ap.add_argument("--seed", type=int, default=2026)
  return ap.parse_args(argv)


def _euler_zyx(yaw: float, pitch: float, roll: float) -> np.ndarray:
  cz, sz = math.cos(yaw), math.sin(yaw)
  cy, sy = math.cos(pitch), math.sin(pitch)
  cx, sx = math.cos(roll), math.sin(roll)
  rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], np.float32)
  ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], np.float32)
  rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], np.float32)
  return (rz @ ry @ rx).astype(np.float32)


def _look_at_cam_mat(cam_pos: np.ndarray, target: np.ndarray) -> np.ndarray:
  """Return MuJoCo/Blender camera matrix whose columns are (right, up, -forward)."""
  fwd = target - cam_pos
  fwd = fwd / max(np.linalg.norm(fwd), 1e-12)
  up_ref = np.array([0.0, 0.0, 1.0], dtype=np.float32)
  right = np.cross(fwd, up_ref)
  right = right / max(np.linalg.norm(right), 1e-12)
  up = np.cross(right, fwd)
  return np.stack([right, up, -fwd], axis=1).astype(np.float32)


def _set_obj_pose(obj, pos: np.ndarray, mat3: np.ndarray):
  m4 = Matrix((
      (float(mat3[0, 0]), float(mat3[0, 1]), float(mat3[0, 2]), float(pos[0])),
      (float(mat3[1, 0]), float(mat3[1, 1]), float(mat3[1, 2]), float(pos[1])),
      (float(mat3[2, 0]), float(mat3[2, 1]), float(mat3[2, 2]), float(pos[2])),
      (0.0, 0.0, 0.0, 1.0),
  ))
  obj.matrix_world = m4


def _make_material(name: str, rgba: tuple, roughness: float = 0.35, metallic: float = 0.05):
  mat = bpy.data.materials.new(name=name)
  mat.use_nodes = True
  bsdf = mat.node_tree.nodes.get("Principled BSDF")
  bsdf.inputs["Base Color"].default_value = (rgba[0], rgba[1], rgba[2], 1.0)
  bsdf.inputs["Roughness"].default_value = roughness
  bsdf.inputs["Metallic"].default_value = metallic
  return mat


def _extract_tri_mesh(obj, depsgraph) -> Tuple[np.ndarray, np.ndarray]:
  eval_obj = obj.evaluated_get(depsgraph)
  mesh = eval_obj.to_mesh()
  bm = bmesh.new()
  bm.from_mesh(mesh)
  bmesh.ops.triangulate(bm, faces=bm.faces[:])
  bm.verts.ensure_lookup_table()
  bm.faces.ensure_lookup_table()
  verts = np.array([[v.co.x, v.co.y, v.co.z] for v in bm.verts], dtype=np.float32)
  faces = np.array([[f.verts[0].index, f.verts[1].index, f.verts[2].index] for f in bm.faces], dtype=np.int32)
  bm.free()
  eval_obj.to_mesh_clear()
  return verts, faces


def main():
  args = _parse_args()
  os.makedirs(args.out_dir, exist_ok=True)
  rng = np.random.default_rng(args.seed)

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

  H = W = args.res
  fovy = 42.0
  scene.render.resolution_x = W
  scene.render.resolution_y = H
  scene.render.resolution_percentage = 100
  scene.render.image_settings.file_format = "OPEN_EXR"
  scene.render.image_settings.color_mode = "RGB"
  scene.render.image_settings.color_depth = "32"
  scene.view_settings.view_transform = "Standard"
  scene.view_settings.look = "None"
  scene.view_settings.exposure = 0.0

  # Studio world & area lights matching shader lighting conventions
  world = bpy.data.worlds.new("StudioWorld")
  scene.world = world
  world.use_nodes = True
  bg = world.node_tree.nodes.get("Background")
  bg.inputs["Color"].default_value = (0.28, 0.36, 0.48, 1.0)
  bg.inputs["Strength"].default_value = 0.28

  area_data = bpy.data.lights.new(name="OverheadArea", type="AREA")
  area_data.energy = 28.0
  area_data.size = 0.80
  area_data.color = (1.0, 0.97, 0.92)
  area_obj = bpy.data.objects.new("OverheadArea", area_data)
  bpy.context.collection.objects.link(area_obj)
  light_pos = np.array([0.65, 0.25, 1.85], dtype=np.float32)
  area_obj.location = tuple(float(x) for x in light_pos)
  area_obj.rotation_euler = (Vector((0.0, 0.0, 0.18)) - Vector(light_pos)).to_track_quat("-Z", "Y").to_euler()

  fill_data = bpy.data.lights.new(name="SideFill", type="AREA")
  fill_data.energy = 12.0
  fill_data.size = 1.2
  fill_data.color = (0.78, 0.88, 1.0)
  fill_obj = bpy.data.objects.new("SideFill", fill_data)
  bpy.context.collection.objects.link(fill_obj)
  fill_obj.location = (0.95, -1.1, 0.95)
  fill_obj.rotation_euler = (Vector((0.0, 0.0, 0.15)) - fill_obj.location).to_track_quat("-Z", "Y").to_euler()

  # Build 3 real curved Blender mesh assets: Suzanne, Torus, Pedestal
  bpy.ops.mesh.primitive_monkey_add(size=0.28, location=(0, 0, 0))
  suzanne = bpy.context.active_object
  suzanne.name = "suzanne"
  sub = suzanne.modifiers.new(name="Subsurf", type="SUBSURF")
  sub.levels = 1
  sub.render_levels = 1
  bpy.ops.object.shade_smooth()
  rgba_suz = np.array([0.85, 0.32, 0.22, 1.0], dtype=np.float32)
  suzanne.data.materials.append(_make_material("MatSuzanne", rgba_suz, roughness=0.32, metallic=0.08))

  bpy.ops.mesh.primitive_torus_add(
      major_radius=0.14, minor_radius=0.048, major_segments=24, minor_segments=12, location=(0, 0, 0)
  )
  torus = bpy.context.active_object
  torus.name = "torus"
  bpy.ops.object.shade_smooth()
  rgba_tor = np.array([0.20, 0.65, 0.78, 1.0], dtype=np.float32)
  torus.data.materials.append(_make_material("MatTorus", rgba_tor, roughness=0.28, metallic=0.10))

  bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0))
  table = bpy.context.active_object
  table.name = "pedestal"
  for v in table.data.vertices:
    v.co.x *= 0.95
    v.co.y *= 0.95
    v.co.z *= 0.08
  bev = table.modifiers.new(name="Bevel", type="BEVEL")
  bev.width = 0.015
  bev.segments = 2
  rgba_tab = np.array([0.62, 0.64, 0.68, 1.0], dtype=np.float32)
  table.data.materials.append(_make_material("MatPedestal", rgba_tab, roughness=0.55, metallic=0.0))

  depsgraph = bpy.context.evaluated_depsgraph_get()
  specs_raw = [
      ("suzanne", suzanne, rgba_suz),
      ("torus", torus, rgba_tor),
      ("pedestal", table, rgba_tab),
  ]
  meshes_np = []
  for name, obj, col in specs_raw:
    v, f = _extract_tri_mesh(obj, depsgraph)
    # Apply modifiers permanently so Cycles and exported mesh use exact identical faces
    bpy.context.view_layer.objects.active = obj
    for mod in list(obj.modifiers):
      bpy.ops.object.modifier_apply(modifier=mod.name)
    # Replace object mesh with exact triangulated mesh so face winding & vertices are 1:1
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    v = np.array([[vt.co.x, vt.co.y, vt.co.z] for vt in bm.verts], dtype=np.float32)
    f = np.array([[fc.verts[0].index, fc.verts[1].index, fc.verts[2].index] for fc in bm.faces], dtype=np.int32)
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    meshes_np.append((name, v, f, col))
    print(f"Exported Blender mesh '{name}': verts={v.shape[0]}, tris={f.shape[0]}", flush=True)

  # Camera aligned cleanly to compute_camera_rays
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

  # Compositor EXR Z-depth output for Frame 0 parity verification (Blender 5.2 API)
  bpy.context.view_layer.use_pass_z = True
  tmpdir = tempfile.mkdtemp(prefix="blender_real_")
  ng = bpy.data.node_groups.new("Comp", "CompositorNodeTree")
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

  N = args.n_frames
  M = 3
  obj_xpos = np.zeros((N, M, 3), dtype=np.float32)
  obj_xmat = np.zeros((N, M, 3, 3), dtype=np.float32)
  cam_xpos = np.zeros((N, 3), dtype=np.float32)
  cam_xmat = np.zeros((N, 3, 3), dtype=np.float32)
  gt_rgb = np.zeros((N, H, W, 3), dtype=np.float32)
  frame0_depth = None

  exr_path = os.path.join(tmpdir, "frame.exr")
  t0 = time.time()
  for i in range(N):
    # Frame 0 deterministic hero pose; frames 1..N-1 randomized rigid poses + slight camera jitter
    if i == 0:
      suz_p = np.array([-0.14, 0.04, 0.23], np.float32)
      suz_m = _euler_zyx(0.45, -0.15, 0.10)
      tor_p = np.array([0.19, -0.04, 0.19], np.float32)
      tor_m = _euler_zyx(-0.35, 0.55, 0.30)
      tab_p = np.array([0.0, 0.0, 0.0], np.float32)
      tab_m = np.eye(3, dtype=np.float32)
      cp = np.array([0.82, -0.58, 0.62], np.float32)
    else:
      suz_p = np.array([rng.uniform(-0.24, 0.14), rng.uniform(-0.18, 0.22), rng.uniform(0.18, 0.34)], np.float32)
      suz_m = _euler_zyx(rng.uniform(-1.0, 1.0), rng.uniform(-0.35, 0.35), rng.uniform(-0.25, 0.25))
      tor_p = np.array([rng.uniform(0.05, 0.28), rng.uniform(-0.22, 0.16), rng.uniform(0.15, 0.30)], np.float32)
      tor_m = _euler_zyx(rng.uniform(-1.2, 1.2), rng.uniform(-0.9, 0.9), rng.uniform(-0.9, 0.9))
      tab_p = np.array([0.0, 0.0, 0.0], np.float32)
      tab_m = _euler_zyx(rng.uniform(-0.15, 0.15), 0.0, 0.0)
      cp = np.array([rng.uniform(0.76, 0.92), rng.uniform(-0.66, -0.46), rng.uniform(0.55, 0.72)], np.float32)

    cm = _look_at_cam_mat(cp, np.array([0.02, 0.0, 0.16], np.float32))
    obj_xpos[i, 0], obj_xmat[i, 0] = suz_p, suz_m
    obj_xpos[i, 1], obj_xmat[i, 1] = tor_p, tor_m
    obj_xpos[i, 2], obj_xmat[i, 2] = tab_p, tab_m
    cam_xpos[i], cam_xmat[i] = cp, cm

    _set_obj_pose(suzanne, suz_p, suz_m)
    _set_obj_pose(torus, tor_p, tor_m)
    _set_obj_pose(table, tab_p, tab_m)
    _set_obj_pose(cam_obj, cp, cm)

    if i == 1:
      scene.compositing_node_group = None

    scene.render.filepath = exr_path
    bpy.ops.render.render(write_still=True)

    img = bpy.data.images.load(exr_path, check_existing=False)
    px = np.asarray(img.pixels[:], dtype=np.float32).reshape(H, W, 4)[..., :3][::-1, :, :].copy()
    bpy.data.images.remove(img)
    px = np.maximum(px, 0.0)
    px = px / (1.0 + 0.15 * px)
    px = np.where(px <= 0.0031308, 12.92 * px, 1.055 * (np.maximum(px, 1e-8) ** (1.0 / 2.4)) - 0.055)
    gt_rgb[i] = np.clip(px, 0.0, 1.0)

    if i == 0:
      d_files = [f for f in os.listdir(tmpdir) if f.startswith("depth_") and f.endswith(".exr")]
      if d_files:
        dimg = bpy.data.images.load(os.path.join(tmpdir, d_files[0]), check_existing=False)
        frame0_depth = np.asarray(dimg.pixels[:], dtype=np.float32).reshape(H, W, 4)[..., 0][::-1, :].copy()
        bpy.data.images.remove(dimg)

    if (i + 1) % 16 == 0 or i + 1 == N:
      print(f"Rendered {i+1}/{N} real Blender Cycles frames ({time.time()-t0:.1f}s)", flush=True)

  split = np.zeros((N,), dtype=np.int32)
  n_tr = max(1, (N * 3) // 4)
  n_cal = max(1, N // 12)
  split[n_tr : n_tr + n_cal] = 1
  split[n_tr + n_cal :] = 2

  bundle = {
      "n_meshes": np.int32(M),
      "height": np.int32(H),
      "width": np.int32(W),
      "cam_fovy": np.float32(fovy),
      "znear": np.float32(0.05),
      "zfar": np.float32(50.0),
      "light_pos": light_pos,
      "obj_xpos": obj_xpos,
      "obj_xmat": obj_xmat,
      "cam_xpos": cam_xpos,
      "cam_xmat": cam_xmat,
      "split": split,
  }
  for idx, (name, v, f, col) in enumerate(meshes_np):
    bundle[f"mesh_{idx}_name"] = np.asarray(name)
    bundle[f"mesh_{idx}_verts"] = v
    bundle[f"mesh_{idx}_faces"] = f
    bundle[f"mesh_{idx}_rgba"] = col

  scene_path = os.path.join(args.out_dir, "blender_real_scene.npz")
  gt_path = os.path.join(args.out_dir, "blender_real_scene_gt.npz")
  np.savez_compressed(scene_path, **bundle)
  np.savez_compressed(
      gt_path,
      rgb=gt_rgb,
      frame0_z_depth=frame0_depth if frame0_depth is not None else np.zeros((H, W), np.float32),
  )
  print(f"Saved {scene_path} and {gt_path}")


if __name__ == "__main__":
  main()
