"""Blender 5.3.0 Cycles Reference Renderer for Priority #3 (`tpu_optix` jax.grad Inverse Rendering)."""

import json
import math
import os
import sys
import time

import bpy
import mathutils
import numpy as np


def clear_scene():
  bpy.ops.wm.read_factory_settings(use_empty=True)


def make_bsdf_mat(name, base_color, roughness, metallic=0.0, emit_rgb=None):
  mat = bpy.data.materials.new(name)
  mat.use_nodes = True
  nt = mat.node_tree
  nt.nodes.clear()
  out = nt.nodes.new("ShaderNodeOutputMaterial")
  if emit_rgb is not None:
    em = nt.nodes.new("ShaderNodeEmission")
    max_c = max(max(emit_rgb), 1e-4)
    em.inputs["Color"].default_value = (emit_rgb[0] / max_c, emit_rgb[1] / max_c, emit_rgb[2] / max_c, 1.0)
    em.inputs["Strength"].default_value = float(max_c)
    nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
    return mat

  bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
  bsdf.inputs["Base Color"].default_value = (*base_color, 1.0)
  bsdf.inputs["Roughness"].default_value = float(roughness)
  bsdf.inputs["Metallic"].default_value = float(metallic)
  if "IOR" in bsdf.inputs:
    bsdf.inputs["IOR"].default_value = 1.5
  nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
  return mat


def add_box(name, center, half_ext, mat):
  bpy.ops.mesh.primitive_cube_add(size=2.0, location=center)
  obj = bpy.context.active_object
  obj.name = name
  obj.scale = half_ext
  bpy.ops.object.transform_apply(scale=True)
  obj.data.materials.append(mat)
  return obj


def main():
  argv = sys.argv
  idx = argv.index("--") + 1
  out_npz = argv[idx]
  out_png = argv[idx + 1]
  res = int(argv[idx + 2])
  spp = int(argv[idx + 3])

  clear_scene()
  scene = bpy.context.scene

  # Ground-truth physical parameters to recover via tpu_optix jax.grad
  light_pos_gt = [0.38, -0.22, 2.08]
  light_emit_gt = [15.5, 13.8, 11.2]
  light_radius = 0.34
  hero_albedo_gt = [0.88, 0.56, 0.16]
  hero_roughness_gt = 0.14

  # Cornell enclosure (x,y in [-1.25, 1.25], z in [0, 2.25])
  m_ivory = make_bsdf_mat("Ivory", (0.76, 0.74, 0.70), 0.75)
  m_red = make_bsdf_mat("RedWall", (0.80, 0.14, 0.12), 0.75)
  m_cyan = make_bsdf_mat("CyanWall", (0.12, 0.60, 0.78), 0.75)
  m_ped = make_bsdf_mat("Pedestal", (0.28, 0.30, 0.34), 0.45)
  m_hero = make_bsdf_mat("HeroSphere", hero_albedo_gt, hero_roughness_gt, metallic=0.0)
  m_emit = make_bsdf_mat("CeilingDisk", (1, 1, 1), 0.5, emit_rgb=light_emit_gt)

  add_box("Floor", (0.0, 0.0, -0.04), (1.30, 1.30, 0.04), m_ivory)
  add_box("Ceiling", (0.0, 0.0, 2.29), (1.30, 1.30, 0.04), m_ivory)
  add_box("BackWall", (0.0, 1.29, 1.125), (1.30, 0.04, 1.17), m_ivory)
  add_box("LeftWall", (-1.29, 0.0, 1.125), (0.04, 1.30, 1.17), m_red)
  add_box("RightWall", (1.29, 0.0, 1.125), (0.04, 1.30, 1.17), m_cyan)
  add_box("Pedestal", (0.0, 0.18, 0.22), (0.42, 0.38, 0.22), m_ped)

  # Hero glossy sphere sitting on pedestal (z_base = 0.44, radius = 0.38 -> center z = 0.82)
  bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=28, radius=0.38, location=(0.0, 0.18, 0.82))
  hero = bpy.context.active_object
  hero.name = "HeroSphere"
  bpy.ops.object.shade_smooth()
  hero.data.materials.append(m_hero)

  # Physical disk emitter matching Area Light in tpu_optix
  bpy.ops.mesh.primitive_circle_add(vertices=48, radius=light_radius, fill_type="NGON", location=tuple(light_pos_gt))
  disk = bpy.context.active_object
  disk.name = "LightDisk"
  disk.rotation_euler = (math.pi, 0.0, 0.0)  # Normal pointing -Z
  disk.data.materials.append(m_emit)

  # Dark subtle world background
  world = bpy.data.worlds.new("World")
  scene.world = world
  world.use_nodes = True
  bg = world.node_tree.nodes.get("Background")
  bg.inputs["Color"].default_value = (0.08, 0.10, 0.14, 1.0)
  bg.inputs["Strength"].default_value = 0.4

  # Camera at (0, -2.45, 1.05) looking at (0, 0.15, 0.72)
  bpy.ops.object.camera_add(location=(0.0, -2.45, 1.05))
  cam = bpy.context.active_object
  direction = mathutils.Vector((0.0, 0.15, 0.72)) - cam.location
  cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
  cam.data.lens = 35.0
  cam.data.sensor_width = 36.0
  scene.camera = cam

  # Render Cycles GT (AgX/Standard sRGB clean)
  scene.render.engine = "CYCLES"
  scene.cycles.device = "CPU"
  scene.cycles.samples = spp
  scene.cycles.max_bounces = 4
  scene.cycles.use_denoising = True
  scene.render.resolution_x = res
  scene.render.resolution_y = res
  scene.render.resolution_percentage = 100
  scene.view_settings.view_transform = "Standard"
  scene.view_settings.exposure = 0.0
  scene.render.image_settings.file_format = "PNG"
  scene.render.filepath = out_png

  t0 = time.perf_counter()
  bpy.ops.render.render(write_still=True)
  print(f"CYCLES INV REF {out_png} ({res}x{res} @ {spp} spp) in {time.perf_counter()-t0:.2f}s", flush=True)

  # Save exact triangle mesh + GT metadata for tpu_optix
  depsgraph = bpy.context.evaluated_depsgraph_get()
  all_v, all_f, all_n, all_alb, all_rgh, all_mc = [], [], [], [], [], []
  v_off = 0
  for inst in depsgraph.object_instances:
    obj = inst.object
    if obj.type != "MESH" or obj.name == "LightDisk":
      continue
    me = obj.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
    me.calc_loop_triangles()
    nt_cnt = len(me.loop_triangles)
    nv_orig = len(me.vertices)
    v_loc = np.empty((nv_orig, 3), np.float32)
    me.vertices.foreach_get("co", v_loc.ravel())
    mw = np.array(inst.matrix_world, np.float32)
    v_w_orig = v_loc @ mw[:3, :3].T + mw[:3, 3][None, :]
    n_loc = np.empty((nv_orig, 3), np.float32)
    me.vertices.foreach_get("normal", n_loc.ravel())
    n_w_orig = n_loc @ np.linalg.inv(mw[:3, :3]).T
    n_w_orig = n_w_orig / np.maximum(np.linalg.norm(n_w_orig, axis=-1, keepdims=True), 1e-12)
    f_arr_orig = np.empty((nt_cnt * 3,), np.int32)
    me.loop_triangles.foreach_get("vertices", f_arr_orig)
    f_arr_orig = f_arr_orig.reshape(nt_cnt, 3)

    tri_smooth = np.empty((nt_cnt,), dtype=bool)
    me.loop_triangles.foreach_get("use_smooth", tri_smooth)

    if np.all(tri_smooth):
      v_w = v_w_orig
      n_w = n_w_orig
      f_arr = f_arr_orig
      nv = nv_orig
    else:
      # Unshare flat-shaded triangle vertices so flat boxes never pillow-shade across 90-deg edges!
      v_w = v_w_orig[f_arr_orig.ravel()]
      e1 = v_w[1::3] - v_w[0::3]
      e2 = v_w[2::3] - v_w[0::3]
      fn = np.cross(e1, e2)
      fn = fn / np.maximum(np.linalg.norm(fn, axis=-1, keepdims=True), 1e-12)
      n_w = np.repeat(fn, 3, axis=0).astype(np.float32)
      nv = nt_cnt * 3
      f_arr = np.arange(nv, dtype=np.int32).reshape(nt_cnt, 3)

    if obj.name == "HeroSphere":
      alb = np.broadcast_to(np.array(hero_albedo_gt, np.float32), (nt_cnt, 3)).copy()
      rgh = np.full((nt_cnt,), hero_roughness_gt, np.float32)
      mc = np.full((nt_cnt,), 0.45, np.float32)
    else:
      mat = obj.data.materials[0]
      bsdf = mat.node_tree.nodes.get("Principled BSDF")
      bc = np.array(bsdf.inputs["Base Color"].default_value[:3], np.float32)
      rg = float(bsdf.inputs["Roughness"].default_value)
      alb = np.broadcast_to(bc, (nt_cnt, 3)).copy()
      rgh = np.full((nt_cnt,), rg, np.float32)
      mc = np.zeros((nt_cnt,), np.float32)

    all_v.append(v_w)
    all_f.append(f_arr + v_off)
    all_n.append(n_w.astype(np.float32))
    all_alb.append(alb)
    all_rgh.append(rgh)
    all_mc.append(mc)
    v_off += nv
    obj.to_mesh_clear()

  cam_mat = np.array(cam.matrix_world, np.float32)
  fovy_deg = math.degrees(2.0 * math.atan(0.5 * cam.data.sensor_width / cam.data.lens))
  np.savez(
      out_npz,
      verts=np.concatenate(all_v, axis=0),
      faces=np.concatenate(all_f, axis=0),
      vnorm=np.concatenate(all_n, axis=0),
      tri_albedo=np.concatenate(all_alb, axis=0),
      tri_roughness=np.concatenate(all_rgh, axis=0),
      tri_mat_code=np.concatenate(all_mc, axis=0),
      cam_pos=cam_mat[:3, 3],
      cam_xmat=cam_mat[:3, :3],
      cam_fovy=np.float32(fovy_deg),
      gt_meta=np.bytes_(json.dumps({
          "light_pos_gt": light_pos_gt,
          "light_emit_gt": light_emit_gt,
          "light_radius": light_radius,
          "hero_albedo_gt": hero_albedo_gt,
          "hero_roughness_gt": hero_roughness_gt,
      })),
  )
  print("Saved inverse scene ->", out_npz, flush=True)


if __name__ == "__main__":
  main()
