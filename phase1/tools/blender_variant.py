"""Render a Cycles variant of an Open Data scene for the image-gap breakdown (run inside Blender).

Modes:
  original    the .blend as shipped
  tpu_lights  the .blend with its lights and world replaced by TPU-RTX's rig (rig.json)
  tpu_scene   TPU-RTX's exported scene (.npz: per-triangle colour/roughness/material code,
              smooth vertex normals) with TPU-RTX's rig, built from scratch

blender -b [scene.blend] -t 64 --python blender_variant.py -- --mode M --out PREFIX
    --res W H --spp N [--rig rig.json] [--npz scene.npz] [--pinhole] [--max-bounces N]
    [--box-filter] [--no-clamp] [--seed S]

Writes PREFIX.exr (multilayer, linear, with passes), PREFIX.png (scene view transform)
and PREFIX.json (settings).
"""

import argparse
import json
import math
import os
import sys
import time

import bpy
import numpy as np
from mathutils import Matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blender_tpu_rig as tr  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--mode", required=True, choices=["original", "tpu_lights", "tpu_scene"])
ap.add_argument("--out", required=True)
ap.add_argument("--res", type=int, nargs=2, default=[640, 360])
ap.add_argument("--spp", type=int, default=1024)
ap.add_argument("--rig")
ap.add_argument("--npz")
ap.add_argument("--pinhole", action="store_true")
ap.add_argument("--max-bounces", type=int)
ap.add_argument("--box-filter", action="store_true")
ap.add_argument("--no-clamp", action="store_true")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--view", help="view transform to use (tpu_scene defaults to Standard)")
ap.add_argument("--look", default="None")
ap.add_argument("--exposure-stops", type=float, default=0.0)
args = ap.parse_args(sys.argv[sys.argv.index("--") + 1:])
rig = json.load(open(args.rig)) if args.rig else None
info = {"args": vars(args)}


def build_tpu_scene(npz_path):
  ds = np.load(npz_path, allow_pickle=False)
  verts = ds["verts"].astype(np.float32)
  faces = ds["faces"].astype(np.int32)
  vnorm = ds["vnorm"].astype(np.float32)
  alb = ds["tri_albedo"].astype(np.float32)
  rgh = np.clip(ds["tri_roughness"].astype(np.float32), 0.02, 0.99)
  mc = np.floor(ds["tri_mat_code"].astype(np.float32) * 100.0 + 0.5) * 0.01
  F = len(faces)
  bpy.ops.wm.read_factory_settings(use_empty=True)
  me = bpy.data.meshes.new("tpu_scene")
  me.vertices.add(len(verts))
  me.vertices.foreach_set("co", verts.ravel())
  me.loops.add(3 * F)
  me.loops.foreach_set("vertex_index", faces.ravel())
  me.polygons.add(F)
  me.polygons.foreach_set("loop_start", np.arange(0, 3 * F, 3, dtype=np.int32))
  me.update()
  mat_idx = np.where(mc >= 2.8, 2, np.where(mc >= 1.8, 1, 0)).astype(np.int32)
  me.polygons.foreach_set("material_index", mat_idx)
  me.polygons.foreach_set("use_smooth", np.ones(F, dtype=bool))
  a = me.attributes.new("alb", "FLOAT_COLOR", "FACE")
  a.data.foreach_set("color", np.concatenate([alb, np.ones((F, 1), np.float32)], 1).ravel())
  for name, val in (("rgh", rgh), ("met", np.clip(mc, 0.0, 1.0)), ("emit_s", np.where(mc >= 2.8, (mc - 2.0) * 6.0, 0.0))):
    at = me.attributes.new(name, "FLOAT", "FACE")
    at.data.foreach_set("value", val.astype(np.float32))
  me.normals_split_custom_set_from_vertices(vnorm.tolist())
  obj = bpy.data.objects.new("tpu_scene", me)
  bpy.context.scene.collection.objects.link(obj)

  def attr(nt, name, out="Color"):
    n = nt.nodes.new("ShaderNodeAttribute")
    n.attribute_type = "GEOMETRY"
    n.attribute_name = name
    return n.outputs[out]

  # 0: opaque, TPU's GGX + Lambert with F0 = 0.04 (Principled at default specular).
  m0 = bpy.data.materials.new("tpu_opaque")
  m0.use_nodes = True
  nt = m0.node_tree
  p = nt.nodes["Principled BSDF"]
  nt.links.new(attr(nt, "alb"), p.inputs["Base Color"])
  nt.links.new(attr(nt, "rgh", "Fac"), p.inputs["Roughness"])
  nt.links.new(attr(nt, "met", "Fac"), p.inputs["Metallic"])
  # 1: glass, IOR 1.5, smooth, tinted by albedo.
  m1 = bpy.data.materials.new("tpu_glass")
  m1.use_nodes = True
  nt = m1.node_tree
  nt.nodes.remove(nt.nodes["Principled BSDF"])
  g = nt.nodes.new("ShaderNodeBsdfGlass")
  g.inputs["IOR"].default_value = 1.5
  g.inputs["Roughness"].default_value = 0.0
  nt.links.new(attr(nt, "alb"), g.inputs["Color"])
  nt.links.new(g.outputs[0], nt.nodes["Material Output"].inputs["Surface"])
  # 2: emitter, radiance = albedo * (mat_code - 2) * 6, as bvh_pbr_pt.py:679.
  m2 = bpy.data.materials.new("tpu_emit")
  m2.use_nodes = True
  nt = m2.node_tree
  nt.nodes.remove(nt.nodes["Principled BSDF"])
  e = nt.nodes.new("ShaderNodeEmission")
  nt.links.new(attr(nt, "alb"), e.inputs["Color"])
  nt.links.new(attr(nt, "emit_s", "Fac"), e.inputs["Strength"])
  nt.links.new(e.outputs[0], nt.nodes["Material Output"].inputs["Surface"])
  for m in (m0, m1, m2):
    me.materials.append(m)

  cd = bpy.data.cameras.new("tpu_cam")
  cd.sensor_fit = "VERTICAL"
  cd.angle_y = math.radians(float(ds["cam_fovy"]))
  cd.clip_start = 0.05
  cd.clip_end = 120.0
  co = bpy.data.objects.new("tpu_cam", cd)
  bpy.context.scene.collection.objects.link(co)
  mw = Matrix.Identity(4)
  xm = ds["cam_xmat"]
  for i in range(3):
    for j in range(3):
      mw[i][j] = float(xm[i, j])
    mw[i][3] = float(ds["cam_pos"][i])
  co.matrix_world = mw
  bpy.context.scene.camera = co
  info["tpu_scene"] = {"tris": int(F), "opaque": int((mat_idx == 0).sum()), "glass": int((mat_idx == 1).sum()), "emissive": int((mat_idx == 2).sum())}


if args.mode == "tpu_scene":
  build_tpu_scene(args.npz)
sc = bpy.context.scene
if args.mode in ("tpu_lights", "tpu_scene"):
  info["hidden_scene_lights"] = tr.hide_scene_lights()
  tr.set_tpu_sky(rig)
  tr.add_area_light(rig)
  tr.add_sun(rig)
if args.view or args.mode == "tpu_scene":
  sc.view_settings.view_transform = args.view or "Standard"
  sc.view_settings.look = args.look
  sc.view_settings.exposure = args.exposure_stops

sc.render.engine = "CYCLES"
cy = sc.cycles
cy.device = "CPU"
cy.samples = args.spp
cy.use_adaptive_sampling = False
cy.use_denoising = False
cy.seed = args.seed
sc.render.resolution_x, sc.render.resolution_y = args.res
sc.render.resolution_percentage = 100
sc.render.use_motion_blur = False
if args.pinhole and sc.camera is not None:
  sc.camera.data.dof.use_dof = False
if args.max_bounces is not None:
  for k in ("max_bounces", "diffuse_bounces", "glossy_bounces", "transmission_bounces", "volume_bounces"):
    setattr(cy, k, args.max_bounces)
if args.box_filter:
  cy.pixel_filter_type = "BOX"
  cy.filter_width = 1.0
if args.no_clamp:
  cy.sample_clamp_direct = 0.0
  cy.sample_clamp_indirect = 0.0
vl = sc.view_layers[0]
for p in ("use_pass_z", "use_pass_normal", "use_pass_diffuse_color", "use_pass_glossy_color", "use_pass_transmission_color",
          "use_pass_diffuse_direct", "use_pass_diffuse_indirect", "use_pass_glossy_direct", "use_pass_glossy_indirect",
          "use_pass_transmission_direct", "use_pass_transmission_indirect", "use_pass_emit", "use_pass_environment"):
  try:
    setattr(vl, p, True)
  except AttributeError:
    pass
info["settings"] = {k: getattr(cy, k) for k in ("samples", "max_bounces", "diffuse_bounces", "glossy_bounces", "transmission_bounces",
                                                 "sample_clamp_direct", "sample_clamp_indirect", "pixel_filter_type", "filter_width")}
info["settings"].update(view_transform=sc.view_settings.view_transform, look=sc.view_settings.look, exposure=sc.view_settings.exposure,
                        dof=bool(sc.camera.data.dof.use_dof), res=list(args.res))

t0 = time.perf_counter()
bpy.ops.render.render(write_still=False)
info["render_s"] = round(time.perf_counter() - t0, 1)
rr = bpy.data.images["Render Result"]
ims = sc.render.image_settings
if hasattr(ims, "media_type"):  # Blender 5.x: multilayer EXR is a media type
  ims.media_type = "MULTI_LAYER_IMAGE"
ims.file_format = "OPEN_EXR_MULTILAYER"
ims.color_depth = "32"
rr.save_render(filepath=args.out + ".exr", scene=sc)
if hasattr(ims, "media_type"):
  ims.media_type = "IMAGE"
ims.file_format = "PNG"
ims.color_depth = "8"
ims.color_mode = "RGB"
rr.save_render(filepath=args.out + ".png", scene=sc)
with open(args.out + ".json", "w") as f:
  json.dump(info, f, indent=1, default=str)
print("VARIANT_OK", json.dumps(info, default=str))
