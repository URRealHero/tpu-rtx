"""Calibrate Cycles light units against TPU-RTX's conventions (run inside Blender).

blender -b --python blender_calib.py -- out.json

Each case renders a tiny image of a white diffuse plane (or of the sky) in linear EXR and
compares the pixel value to what TPU-RTX's estimator gives analytically.
"""

import json
import math
import os
import sys
import tempfile

import bpy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blender_tpu_rig as tr  # noqa: E402

out_path = sys.argv[sys.argv.index("--") + 1]


def reset():
  bpy.ops.wm.read_factory_settings(use_empty=True)
  sc = bpy.context.scene
  sc.render.engine = "CYCLES"
  sc.cycles.device = "CPU"
  sc.cycles.samples = 512
  sc.cycles.use_denoising = False
  sc.cycles.use_adaptive_sampling = False
  sc.view_settings.view_transform = "Standard"
  sc.render.resolution_x = 8
  sc.render.resolution_y = 8
  sc.render.image_settings.file_format = "OPEN_EXR"
  sc.render.image_settings.color_depth = "32"
  w = bpy.data.worlds.new("black")
  w.use_nodes = True
  w.node_tree.nodes["Background"].inputs["Color"].default_value = (0, 0, 0, 1)
  sc.world = w
  return sc


def white_plane():
  bpy.ops.mesh.primitive_plane_add(size=200.0, location=(0, 0, 0))
  m = bpy.data.materials.new("white")
  m.use_nodes = True
  nt = m.node_tree
  nt.nodes.clear()
  o = nt.nodes.new("ShaderNodeOutputMaterial")
  d = nt.nodes.new("ShaderNodeBsdfDiffuse")
  d.inputs["Color"].default_value = (1, 1, 1, 1)
  nt.links.new(d.outputs[0], o.inputs[0])
  bpy.context.active_object.data.materials.append(m)


def ortho_down_camera(z=20.0):
  cd = bpy.data.cameras.new("cam")
  cd.type = "ORTHO"
  cd.ortho_scale = 0.05
  co = bpy.data.objects.new("cam", cd)
  bpy.context.scene.collection.objects.link(co)
  co.location = (0, 0, z)
  co.rotation_euler = (0, 0, 0)
  bpy.context.scene.camera = co


def render_px():
  sc = bpy.context.scene
  fd, p = tempfile.mkstemp(suffix=".exr")
  os.close(fd)
  sc.render.filepath = p
  bpy.ops.render.render(write_still=True)
  img = bpy.data.images.load(p)
  a = np.array(img.pixels[:], dtype=np.float32).reshape(img.size[1], img.size[0], 4)[..., :3]
  bpy.data.images.remove(img)
  os.remove(p)
  return a


res = {}

# 1. Area disk light, radius r at height h above the plane, facing down. TPU radiance L.
#    Irradiance at the centre below a Lambertian disk: E = pi L r^2/(r^2+h^2); pixel = E/pi.
reset(); white_plane(); ortho_down_camera()
r, h, L = 1.0, 1.0, 3.0
tr.add_area_light({"area_radius": r, "area_emit": [L, L, L], "area_center": [0, 0, h], "area_normal": [0, 0, -1]})
px = float(render_px().mean())
expect = L * r * r / (r * r + h * h)
res["area_disk"] = {"pixel": px, "tpu_expect": expect, "ratio": px / expect}

# 2. Sun straight down with TPU irradiance S: pixel = S/pi.
reset(); white_plane(); ortho_down_camera()
S = 2.0
tr.add_sun({"has_sun": True, "sun_emit": [S, S, S], "sun_cos_half_angle": 0.9985, "sun_dir": [0, 0, 1]})
px = float(render_px().mean())
res["sun"] = {"pixel": px, "tpu_expect": S / math.pi, "ratio": px / (S / math.pi)}

# 3. Uniform world radiance 0.5: pixel = 0.5.
reset(); white_plane(); ortho_down_camera()
bpy.context.scene.world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.5, 0.5, 0.5, 1)
px = float(render_px().mean())
res["uniform_world"] = {"pixel": px, "tpu_expect": 0.5, "ratio": px / 0.5}

# 4. Emission shader strength 2 seen directly: TPU adds alb * power as radiance.
reset(); ortho_down_camera()
bpy.ops.mesh.primitive_plane_add(size=10.0)
m = bpy.data.materials.new("em")
m.use_nodes = True
nt = m.node_tree
nt.nodes.clear()
o = nt.nodes.new("ShaderNodeOutputMaterial")
e = nt.nodes.new("ShaderNodeEmission")
e.inputs["Strength"].default_value = 2.0
nt.links.new(e.outputs[0], o.inputs[0])
bpy.context.active_object.data.materials.append(m)
px = float(render_px().mean())
res["emission_direct"] = {"pixel": px, "tpu_expect": 2.0, "ratio": px / 2.0}

# 5. TPU sky gradient seen from inside: equirect camera, compare rows to the formula.
reset()
sc = bpy.context.scene
sc.render.resolution_x = 64
sc.render.resolution_y = 32
sc.cycles.samples = 64
rig = {"sky_zenith": [0.35, 0.52, 0.8], "sky_horizon": [0.28, 0.31, 0.36], "sky_ground": [0.12, 0.11, 0.1],
       "has_sun": True, "sun_dir": [0.6, 0.0, 0.8], "sun_emit": [10, 10, 10]}
tr.set_tpu_sky(rig)
cd = bpy.data.cameras.new("pano")
cd.type = "PANO"
cd.panorama_type = "EQUIRECTANGULAR"
co = bpy.data.objects.new("pano", cd)
sc.collection.objects.link(co)
co.rotation_euler = (math.pi / 2, 0, -math.pi / 2)  # look along +X, Z up
sc.camera = co
img = render_px()
Hh, Ww = img.shape[:2]
errs = []
for yy in range(Hh):
  for xx in range(Ww):
    lat = math.pi * ((yy + 0.5) / Hh - 0.5)  # Blender image rows are bottom-to-top
    lon = 2 * math.pi * ((xx + 0.5) / Ww - 0.5)
    d = np.array([math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)])
    zc = d[2]
    hh, ze, gg = (np.array(rig[k]) for k in ("sky_horizon", "sky_zenith", "sky_ground"))
    s = (1 - zc) * hh + zc * ze if zc >= 0 else (1 + zc) * hh - zc * gg
    sd = np.array(rig["sun_dir"]) / np.linalg.norm(rig["sun_dir"])
    s = s + max(float(d @ sd), 0.0) ** 32 * np.array(rig["sun_emit"]) * 0.25
    errs.append((float(np.abs(img[yy, xx] - s).max()), float(np.abs(img[yy, xx][::-1] - s).max())))
errs = np.array(errs)
res["sky"] = {"max_abs_err_assumed_orientation": float(np.quantile(errs[:, 0], 0.95)),
              "note": "95th-percentile per-pixel max abs error vs formula (pixel centres; sun glow is sharp)"}

with open(out_path, "w") as f:
  json.dump(res, f, indent=1)
print("CALIB", json.dumps(res))
