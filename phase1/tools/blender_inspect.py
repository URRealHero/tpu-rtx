"""Print a .blend scene's render settings, lights and world as JSON (run inside Blender).

blender -b scene.blend --python blender_inspect.py -- out.json
"""

import json
import sys

import bpy

out = sys.argv[sys.argv.index("--") + 1]
sc = bpy.context.scene
cy = sc.cycles
vs = sc.view_settings
info = {
    "file": bpy.data.filepath,
    "resolution": [sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage],
    "samples": getattr(cy, "samples", None),
    "adaptive_sampling": getattr(cy, "use_adaptive_sampling", None),
    "adaptive_threshold": getattr(cy, "adaptive_threshold", None),
    "denoising": getattr(cy, "use_denoising", None),
    "max_bounces": getattr(cy, "max_bounces", None),
    "diffuse_bounces": getattr(cy, "diffuse_bounces", None),
    "glossy_bounces": getattr(cy, "glossy_bounces", None),
    "transmission_bounces": getattr(cy, "transmission_bounces", None),
    "transparent_max_bounces": getattr(cy, "transparent_max_bounces", None),
    "sample_clamp_direct": getattr(cy, "sample_clamp_direct", None),
    "sample_clamp_indirect": getattr(cy, "sample_clamp_indirect", None),
    "light_threshold": getattr(cy, "light_sampling_threshold", None),
    "caustics_reflective": getattr(cy, "caustics_reflective", None),
    "caustics_refractive": getattr(cy, "caustics_refractive", None),
    "filter_glossy": getattr(cy, "blur_glossy", None),
    "view_transform": vs.view_transform,
    "look": vs.look,
    "exposure": vs.exposure,
    "gamma": vs.gamma,
    "display_device": sc.display_settings.display_device,
    "camera": sc.camera.name if sc.camera else None,
    "camera_lens_mm": sc.camera.data.lens if sc.camera else None,
    "camera_sensor_fit": sc.camera.data.sensor_fit if sc.camera else None,
    "camera_dof": sc.camera.data.dof.use_dof if sc.camera else None,
    "motion_blur": sc.render.use_motion_blur,
    "film_transparent": sc.render.film_transparent,
    "pixel_filter_width": getattr(cy, "filter_width", None),
}
lights = []
for o in bpy.data.objects:
  if o.type == "LIGHT":
    d = o.data
    lights.append({
        "name": o.name, "type": d.type, "energy": d.energy, "color": list(d.color),
        "hide_render": o.hide_render, "size": getattr(d, "size", None),
        "shape": getattr(d, "shape", None), "location": list(o.matrix_world.translation),
    })
info["lights"] = lights
info["n_lights_rendered"] = sum(1 for l in lights if not l["hide_render"])
w = sc.world
world = {"name": w.name if w else None, "nodes": []}
if w and w.use_nodes:
  for n in w.node_tree.nodes:
    world["nodes"].append({"type": n.type, "name": n.name, "image": getattr(getattr(n, "image", None), "name", None)})
info["world"] = world
emissive = 0
textured = 0
mats = 0
node_types = {}
for m in bpy.data.materials:
  if not m.use_nodes or m.users == 0:
    continue
  mats += 1
  for n in m.node_tree.nodes:
    node_types[n.type] = node_types.get(n.type, 0) + 1
info["materials_in_use"] = mats
info["material_node_type_counts"] = dict(sorted(node_types.items(), key=lambda kv: -kv[1]))
with open(out, "w") as f:
  json.dump(info, f, indent=1, default=str)
print("INSPECT_OK", out)
