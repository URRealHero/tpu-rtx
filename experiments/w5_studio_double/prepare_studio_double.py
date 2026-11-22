"""Prepare Blender Cycles Official Demo Double-Feature (`fishy_cat` + `monster_sss`).

Runs inside Blender 5.3.0 (`--python`) to:
  1. Force CPU Cycles & CPU compositor across all scenes (no headless EGL crash).
  2. Filter out `visible_camera=False`, `BOUNDS`/`WIRE` proxies (e.g. `camera_filter`,
     `sparkles_*`), holdouts, and volume-only / pure-transparent proxy meshes.
  3. Compile `ScenePack` IR via `blend_compiler.compile_active_blend`.
  4. Render Cycles 5.3.0 orbit & hero GT + 32-bit EXR Z-depth (plus pinhole depth
     for DOF scenes so geometric parity can be measured cleanly).
"""

from __future__ import annotations

import argparse
import os
import sys

import bpy
import numpy as np

sys.path.insert(0, "/dev/shm/zaychik-tpu-mjx")
sys.path.insert(0, "/dev/shm/zaychik-tpu-mjx/tpu_mjx_render/adapters")
import blend_compiler as bc  # noqa: E402


def _is_proxy_or_volume(obj) -> bool:
  if obj.hide_render or not getattr(obj, "visible_camera", True):
    return True
  if obj.type not in ("MESH", "CURVE", "SURFACE", "FONT", "META"):
    return True
  if getattr(obj, "is_holdout", False):
    return True
  if getattr(obj, "display_type", "TEXTURED") in ("BOUNDS", "WIRE"):
    return True
  name_l = obj.name.lower()
  if "camera_filter" in name_l or "sparkle" in name_l:
    return True
  mats = list(getattr(obj.data, "materials", [])) if obj.data else []
  if mats:
    all_non_surf = True
    for m in mats:
      if m is None:
        all_non_surf = False
        continue
      if getattr(m, "use_nodes", False) and m.node_tree:
        out = next(
            (n for n in m.node_tree.nodes if n.type == "OUTPUT_MATERIAL" and n.is_active_output),
            None,
        )
        if out is None:
          out = next((n for n in m.node_tree.nodes if n.type == "OUTPUT_MATERIAL"), None)
        if out is not None:
          if not out.inputs["Surface"].is_linked:
            continue
          fn = out.inputs["Surface"].links[0].from_node
          if fn.type == "BSDF_TRANSPARENT":
            continue
      all_non_surf = False
      break
    if all_non_surf:
      return True
  return False


def prepare_scene(
    blend_path: str,
    scene_key: str,
    out_dir: str,
    orbit_w: int,
    orbit_h: int,
    orbit_views: int,
    orbit_spp: int,
    hero_w: int,
    hero_h: int,
    hero_views: int,
    hero_spp: int,
    max_yaw: float,
    max_pitch: float,
    max_tex_res: int = 512,
) -> None:
  bpy.ops.wm.open_mainfile(filepath=blend_path)
  sc = bpy.context.scene
  for s in bpy.data.scenes:
    if hasattr(s.render, "compositor_device"):
      s.render.compositor_device = "CPU"
    s.render.engine = "CYCLES"
    s.cycles.device = "CPU"
  sc.render.use_sequencer = False
  if hasattr(sc.render.image_settings, "media_type"):
    sc.render.image_settings.media_type = "IMAGE"
  sc.render.image_settings.file_format = "PNG"

  # Set target aspect resolution before compiling camera intrinsics
  sc.render.resolution_x = int(hero_w)
  sc.render.resolution_y = int(hero_h)
  sc.render.resolution_percentage = 100

  n_hidden = 0
  for o in bpy.data.objects:
    if not o.hide_render and o.type in ("MESH", "CURVE", "SURFACE", "FONT", "META"):
      if _is_proxy_or_volume(o):
        o.hide_render = True
        n_hidden += 1

  pack = bc.compile_active_blend(scene_name=scene_key, max_tex_res=max_tex_res)
  pack_path = os.path.join(out_dir, f"{scene_key}_pack.npz")
  pack.save(pack_path)
  print(f"[{scene_key}] Compiled ScenePack (filtered {n_hidden} proxy/bounds objs): {pack.summary()}", flush=True)

  orb_path = os.path.join(out_dir, f"{scene_key}_gt_orbit.npz")
  bc.render_cycles_gt_and_depth(
      pack, orb_path, width=orbit_w, height=orbit_h,
      samples=orbit_spp, n_views=orbit_views, phase_offset=0.0,
      include_hero=True, max_yaw=max_yaw, max_pitch=max_pitch,
  )

  hero_path = os.path.join(out_dir, f"{scene_key}_gt_hero.npz")
  hero_png = os.path.join(out_dir, f"{scene_key}_hero_gt.png")
  bc.render_cycles_gt_and_depth(
      pack, hero_path, width=hero_w, height=hero_h,
      samples=hero_spp, n_views=hero_views, hero_png_path=hero_png,
      phase_offset=0.37, include_hero=True, max_yaw=max_yaw, max_pitch=max_pitch,
  )

  # If camera has shallow DOF, also export 1-frame pinhole EXR depth for exact geometric audit
  cam_obj = sc.camera or next(o for o in bpy.data.objects if o.type == "CAMERA")
  if getattr(cam_obj.data, "dof", None) and cam_obj.data.dof.use_dof:
    cam_obj.data.dof.use_dof = False
    pin_path = os.path.join(out_dir, f"{scene_key}_gt_hero_pinhole_z.npz")
    bc.render_cycles_gt_and_depth(
        pack, pin_path, width=hero_w, height=hero_h,
        samples=4, n_views=1, phase_offset=0.0, include_hero=True,
    )
    hd = dict(np.load(hero_path, allow_pickle=False))
    pd = np.load(pin_path, allow_pickle=False)
    hd["frame0_z_depth_pinhole"] = pd["frame0_z_depth"]
    np.savez(hero_path, **hd)
    os.remove(pin_path)
    print(f"[{scene_key}] Attached frame0_z_depth_pinhole to {hero_path}", flush=True)


def main() -> None:
  argv = sys.argv
  argv = argv[argv.index("--") + 1 :] if "--" in argv else []
  ap = argparse.ArgumentParser()
  ap.add_argument("--out_dir", default="/dev/shm/zaychik-tpu-mjx-w5")
  ap.add_argument("--scenes", nargs="+", default=["fishy_cat", "monster_sss"])
  args = ap.parse_args(argv)

  configs = {
      "fishy_cat": dict(
          blend_path="/dev/shm/zaychik-tpu-mjx-w5/scenes/fishy_cat/blender_splash_fishy_cat/fishy_cat.blend",
          orbit_w=560, orbit_h=256, orbit_views=36, orbit_spp=24,
          hero_w=1120, hero_h=512, hero_views=5, hero_spp=40,
          max_yaw=0.10, max_pitch=0.06,
      ),
      "monster_sss": dict(
          blend_path="/dev/shm/zaychik-tpu-mjx-w5/scenes/monster_sss.blend",
          orbit_w=512, orbit_h=512, orbit_views=40, orbit_spp=24,
          hero_w=1024, hero_h=1024, hero_views=5, hero_spp=48,
          max_yaw=0.14, max_pitch=0.08,
      ),
  }
  for sk in args.scenes:
    prepare_scene(scene_key=sk, out_dir=args.out_dir, **configs[sk])


if __name__ == "__main__":
  main()
