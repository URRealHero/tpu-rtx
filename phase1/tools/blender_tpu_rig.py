"""Blender-side helpers: build TPU-RTX's light rig (one disk area light, a sun, a gradient sky)
inside a Cycles scene, so Cycles can render with exactly the lights TPU-RTX uses.

Unit conversions are checked by blender_calib.py:
  area light: TPU 'area_emit' is radiance; Cycles area power P = L * A * pi (verified by calibration)
  sun:        TPU 'sun_emit' is irradiance; Cycles sun strength is irradiance
  sky:        TPU sky value is radiance; Cycles background color * strength is radiance
"""

import math

import bpy
from mathutils import Vector


def _look_minus_z(obj, direction):
  """Orient obj so its local -Z axis points along `direction` (Blender lights emit along -Z)."""
  d = Vector(direction).normalized()
  obj.rotation_euler = (-d).to_track_quat("Z", "Y").to_euler()


def add_area_light(rig, name="TPU_area", area_power_factor=1.0):
  r = float(rig["area_radius"])
  emit = [float(x) for x in rig["area_emit"]]
  peak = max(emit)
  ld = bpy.data.lights.new(name, type="AREA")
  ld.shape = "DISK"
  ld.size = 2.0 * r
  ld.color = [e / peak for e in emit]
  ld.energy = peak * math.pi * r * r * math.pi * area_power_factor
  obj = bpy.data.objects.new(name, ld)
  bpy.context.scene.collection.objects.link(obj)
  obj.location = Vector(rig["area_center"])
  _look_minus_z(obj, rig["area_normal"])
  obj.visible_camera = False
  return obj


def add_sun(rig, name="TPU_sun"):
  if not rig.get("has_sun"):
    return None
  emit = [float(x) for x in rig["sun_emit"]]
  peak = max(emit)
  ld = bpy.data.lights.new(name, type="SUN")
  ld.color = [e / peak for e in emit]
  ld.energy = peak
  ld.angle = 2.0 * math.acos(float(rig["sun_cos_half_angle"]))
  obj = bpy.data.objects.new(name, ld)
  bpy.context.scene.collection.objects.link(obj)
  _look_minus_z(obj, [-x for x in rig["sun_dir"]])
  obj.visible_camera = False
  return obj


def set_tpu_sky(rig, world=None):
  """World = TPU-RTX's _env_sky: piecewise-linear gradient in direction z plus a cos^32 sun glow."""
  sc = bpy.context.scene
  if world is None:
    world = bpy.data.worlds.new("TPU_sky")
  sc.world = world
  world.use_nodes = True
  nt = world.node_tree
  nt.nodes.clear()
  out = nt.nodes.new("ShaderNodeOutputWorld")
  bg = nt.nodes.new("ShaderNodeBackground")
  bg.inputs["Strength"].default_value = 1.0
  tc = nt.nodes.new("ShaderNodeTexCoord")
  dirn = nt.nodes.new("ShaderNodeVectorMath")
  dirn.operation = "NORMALIZE"
  nt.links.new(tc.outputs["Generated"], dirn.inputs[0])
  sep = nt.nodes.new("ShaderNodeSeparateXYZ")
  nt.links.new(dirn.outputs["Vector"], sep.inputs[0])

  def math_node(op, a=None, b=None):
    n = nt.nodes.new("ShaderNodeMath")
    n.operation = op
    for i, x in enumerate((a, b)):
      if x is None:
        continue
      if isinstance(x, (int, float)):
        n.inputs[i].default_value = float(x)
      else:
        nt.links.new(x, n.inputs[i])
    return n.outputs[0]

  def vec_const(v):
    n = nt.nodes.new("ShaderNodeCombineXYZ")
    for i, c in enumerate("XYZ"):
      n.inputs[c].default_value = float(v[i])
    return n.outputs[0]

  def vscale(vec, s):
    n = nt.nodes.new("ShaderNodeVectorMath")
    n.operation = "SCALE"
    nt.links.new(vec, n.inputs[0])
    nt.links.new(s, n.inputs["Scale"])
    return n.outputs[0]

  def vadd(a, b):
    n = nt.nodes.new("ShaderNodeVectorMath")
    n.operation = "ADD"
    nt.links.new(a, n.inputs[0])
    nt.links.new(b, n.inputs[1])
    return n.outputs[0]

  z = sep.outputs["Z"]
  tu = math_node("MAXIMUM", z, 0.0)
  td = math_node("MAXIMUM", math_node("MULTIPLY", z, -1.0), 0.0)
  h = [float(x) for x in rig["sky_horizon"]]
  zen = [float(x) for x in rig["sky_zenith"]]
  g = [float(x) for x in rig["sky_ground"]]
  col = vadd(vadd(vec_const(h), vscale(vec_const([zen[i] - h[i] for i in range(3)]), tu)),
             vscale(vec_const([g[i] - h[i] for i in range(3)]), td))
  if rig.get("has_sun"):
    sd = Vector(rig["sun_dir"]).normalized()
    dot = nt.nodes.new("ShaderNodeVectorMath")
    dot.operation = "DOT_PRODUCT"
    nt.links.new(dirn.outputs["Vector"], dot.inputs[0])
    dot.inputs[1].default_value = sd
    glow = math_node("POWER", math_node("MAXIMUM", dot.outputs["Value"], 0.0), 32.0)
    se = [0.25 * float(x) for x in rig["sun_emit"]]
    col = vadd(col, vscale(vec_const(se), glow))
  nt.links.new(col, bg.inputs["Color"])
  nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
  return world


def hide_scene_lights():
  n = 0
  for o in bpy.data.objects:
    if o.type == "LIGHT" and not o.hide_render:
      o.hide_render = True
      n += 1
  return n
