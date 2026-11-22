"""Wave 3 Cinema-Grade 4K Showcase & Hybrid Real2Sim Runner on 4-Chip TPU v5p.

Generates:
1. `v3_w3_cinema_3dgs_4k.png`: Native 4K (3840x2160) 10M-Gaussian Amazon Lumberyard Bistro
   with TPU Voxel Cone-Traced Golden-Hour Sun Soft Shadows + 12-Ray Hemisphere AO
   + Cool Sky Fill + Warm Streetlamp Glow + Dynamic GGX Specular/Rim + Nishita
   Golden-Hour Sky Dome + Depth Atmospheric Haze + Subtle Bloom + Blender AgX.
2. `v3_w3_hybrid_mjx_3dgs_4k.png`: Native 4K (3840x2160) Hybrid MJX Panda x 10M 3DGS Sponza
   with exact per-pixel Z-buffer occlusion and unified Cinema Golden-Hour HDR lighting,
   Nishita Sky Dome, Depth Haze, Subtle Bloom, and Blender AgX Tonemapping.
3. Updates `v3_w3_wave3_realtosim_hybrid.json` with verified benchmarks & HLO checks.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np
from PIL import Image

from tpu_mjx_render.camera import compute_camera_rays
from tpu_mjx_render.renderer import RenderSceneSpec, _cast_rays_tiled, _shade_scene
from tpu_mjx_render.roofline import inspect_hlo_ops
from tpu_mjx_render.scenes import get_scene_xml

from cinema_3dgs import (
    bake_cinema_lighting_tpu,
    composite_cinema_hdr_and_tonemap,
    load_gltf_scene_cinema,
    make_cinema_gs3d_4c_fn,
    sample_cinema_3dgs,
)
from gltf_to_3dgs import look_at_cam_mat


BISTRO_GLTF = "/dev/shm/zaychik-tpu-mjx-w3/assets/niagara_bistro/bistro.gltf"
SPONZA_GLTF = "/dev/shm/zaychik-tpu-mjx-w3/assets/gltf-assets/Models/Sponza/glTF/Sponza.gltf"


def _check_hlo_zero_host_comm(lowered) -> Dict[str, Any]:
  txt = lowered.as_text()
  info = inspect_hlo_ops(txt)
  info["send_recv_ops"] = len(re.findall(r"\bsend\(|\brecv\(", txt, re.I))
  assert info["host_callbacks"] == 0, f"Found host_callbacks: {info}"
  assert info["send_recv_ops"] == 0, f"Found send_recv_ops: {info}"
  info["zero_host_comm_verified"] = True
  return info


def _bench_fn(fn, args, n_warmup: int = 1, n_steps: int = 3):
  t0 = time.perf_counter()
  out = fn(*args)
  jax.block_until_ready(out)
  compile_s = time.perf_counter() - t0
  for _ in range(n_warmup):
    out = fn(*args)
  jax.block_until_ready(out)
  t1 = time.perf_counter()
  for _ in range(n_steps):
    out = fn(*args)
  jax.block_until_ready(out)
  ms = (time.perf_counter() - t1) * 1e3 / n_steps
  return ms, compile_s, out


def _save_png(rgb_hwc: np.ndarray, path: str) -> None:
  arr = np.clip(np.round(rgb_hwc * 255.0), 0, 255).astype(np.uint8)
  Image.fromarray(arr).save(path)


def run_cinema_bistro(
    data: Dict[str, Any],
    export_dir: str,
    n_gaussians: int = 10_000_000,
    preview_only: bool = False,
) -> None:
  print(f"=== Loading & Sampling Amazon Lumberyard Bistro ({n_gaussians:,} Gaussians) ===", flush=True)
  t_load = time.perf_counter()
  sc = load_gltf_scene_cinema(BISTRO_GLTF, load_normal_maps=True)

  # Use Bistro's cinematic hero street camera (or slight golden-hour composition refinement)
  if sc["cameras"]:
    cam0 = sc["cameras"][0]
    cam_pos = cam0["pos"]
    cam_mat = cam0["mat"]
    cam_fwd = -cam_mat[:, 2]
    fovy = max(min(cam0["yfov_deg"], 48.0), 36.0)
  else:
    cam_pos = np.array([-26.43, 3.16, 11.17], dtype=np.float32)
    target = np.array([-8.0, 2.5, 16.5], dtype=np.float32)
    cam_mat = look_at_cam_mat(cam_pos, target, world_up=(0.0, 1.0, 0.0))
    cam_fwd = -cam_mat[:, 2]
    fovy = 38.0

  print(f"Bistro camera pos={cam_pos.round(2)}, fwd={cam_fwd.round(3)}, fovy={fovy:.1f} deg", flush=True)

  gs = sample_cinema_3dgs(
      sc, n_gaussians=n_gaussians, seed=42, splat_scale=1.35, thick_ratio=0.22,
      cam_pos=cam_pos, cam_fwd=cam_fwd, focus_radius=26.0,
  )

  # Warm up pure-white streetlamps into golden tungsten while preserving Bistro's colored cafe/string lights
  warm_lights = []
  for l in sc["point_lights"]:
    c = l["color"].copy()
    if float(np.min(c)) > 0.75:
      c = np.array([1.0, 0.68, 0.34], dtype=np.float32)
    warm_lights.append({**l, "color": c})

  # Golden-Hour sun slanting down Rue de Paris from ahead-left (catches sky & casts dramatic awning/lamp shadows)
  sun_dir = np.array([0.72, 0.34, 0.60], dtype=np.float32)
  sun_dir /= np.linalg.norm(sun_dir)
  sun_color = np.array([4.2, 2.35, 1.05], dtype=np.float32)
  sky_color = np.array([0.36, 0.56, 1.05], dtype=np.float32)
  ground_color = np.array([0.32, 0.22, 0.14], dtype=np.float32)

  # Focused lighting volume around street corridor
  bmin = np.array([-42.0, -3.5, -24.0], dtype=np.float32)
  bmax = np.array([34.0, 24.0, 38.0], dtype=np.float32)
  baked = bake_cinema_lighting_tpu(
      gs["mu"], gs["nrm"], gs["albedo_lin"], gs["emissive_hdr"],
      sun_dir=sun_dir, sun_color=sun_color, sky_color=sky_color, ground_color=ground_color,
      point_lights=warm_lights, voxel_bounds=(bmin, bmax),
      grid_shape=(384, 144, 320), shadow_dist=32.0, ao_dist=4.5, point_light_boost=1.45,
  )
  load_s = time.perf_counter() - t_load
  print(f"Bistro Cinema asset prep + TPU bake total: {load_s:.2f}s", flush=True)

  devs = jax.devices()[:4]
  mesh = jax.sharding.Mesh(np.array(devs), ("dp",))
  rep = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

  args_4c = (
      jax.device_put(jnp.asarray(gs["mu"]), rep),
      jax.device_put(jnp.asarray(gs["cov6"]), rep),
      jax.device_put(jnp.asarray(baked["diffuse_hdr"]), rep),
      jax.device_put(jnp.asarray(gs["opacity"]), rep),
      jax.device_put(jnp.asarray(gs["nrm"]), rep),
      jax.device_put(jnp.asarray(gs["spec_col"]), rep),
      jax.device_put(jnp.asarray(gs["roughness"]), rep),
      jax.device_put(jnp.asarray(baked["sun_vis"]), rep),
      jax.device_put(jnp.asarray(baked["sky_vis"]), rep),
      jax.device_put(jnp.asarray(cam_pos), rep),
      jax.device_put(jnp.asarray(cam_mat), rep),
      jax.device_put(jnp.asarray(sun_dir), rep),
      jax.device_put(jnp.asarray(sun_color), rep),
      jax.device_put(jnp.asarray(sky_color), rep),
  )

  res_list = [("1080p", 1920, 1080)] if preview_only else [("1080p", 1920, 1080), ("4K", 3840, 2160)]
  cinema_runs = []
  for res_label, W, H in res_list:
    gs_4c_fn = make_cinema_gs3d_4c_fn(
        mesh, height=H, width=W, cam_fovy=fovy, znear=0.12,
        k_macro=4096, k_tile=512, macro_rep=4, use_bf16_mxu=False,
    )

    @jax.jit
    def _render_cinema_frame(mu, cov6, diff_hdr, op, nrm, spec_c, rough, sun_v, sky_v, cxp, cxm, sd, sc_c, sk_c):
      gs_rgb_prem, gs_z, gs_alp = gs_4c_fn(mu, cov6, diff_hdr, op, nrm, spec_c, rough, sun_v, sky_v, cxp, cxm, sd, sc_c, sk_c)
      _, ray_dir, _ = compute_camera_rays(cxp, cxm, fovy, H, W, znear=0.12)
      srgb, comp_depth = composite_cinema_hdr_and_tonemap(
          gs_rgb_prem, gs_alp, gs_z, ray_dir.reshape(H, W, 3), sd,
          fog_density=0.012, fog_color=(1.15, 0.64, 0.32),
          exposure=1.08, bloom_strength=0.13,
      )
      return srgb, comp_depth

    if res_label == "4K" or preview_only:
      hlo_info = _check_hlo_zero_host_comm(_render_cinema_frame.lower(*args_4c))
      data.setdefault("hlo_checks", {})[f"cinema_bistro_{n_gaussians}_{res_label}"] = hlo_info
      print(f"Cinema Bistro {res_label} HLO verified:", hlo_info, flush=True)

    ms, comp_s, out_t = _bench_fn(_render_cinema_frame, args_4c, n_warmup=1, n_steps=3)
    srgb_img, _ = out_t
    rec = {
        "scene": "Amazon_Lumberyard_Bistro",
        "n_chips": 4,
        "res_label": res_label,
        "res": f"{W}x{H}",
        "n_gaussians": n_gaussians,
        "k_tile": 512,
        "macro_rep": 4,
        "bake_s": round(float(baked["bake_s"]), 2),
        "ms": round(ms, 2),
        "fps": round(1000.0 / ms, 2),
        "compile_s": round(comp_s, 2),
        "features": [
            "voxel_cone_traced_soft_shadows",
            "12_ray_hemisphere_ao",
            "96_khr_punctual_streetlamps",
            "ggx_specular_and_fresnel_rim",
            "nishita_golden_hour_sky_dome",
            "exponential_depth_haze",
            "hdr_bloom_and_blender_agx",
        ],
    }
    cinema_runs.append(rec)
    print(f"CINEMA BISTRO {res_label}:", rec, flush=True)

    fname = "v3_w3_cinema_3dgs_preview.png" if preview_only else (
        "v3_w3_cinema_3dgs_4k.png" if res_label == "4K" else "v3_w3_cinema_3dgs_1080p.png"
    )
    out_png = os.path.join(export_dir, fname)
    _save_png(np.asarray(srgb_img), out_png)
    print(f"Saved Cinema Bistro render: {out_png}", flush=True)

  data["cinema_bistro_4k"] = cinema_runs


def run_cinema_hybrid_4k(
    data: Dict[str, Any],
    export_dir: str,
    n_gaussians: int = 10_000_000,
    preview_only: bool = False,
) -> None:
  print(f"=== Running Cinema-Grade Hybrid MJX Panda x 10M 3DGS Sponza ===", flush=True)
  mj_model = mujoco.MjModel.from_xml_string(get_scene_xml("panda_pick_cube"))
  for gname in ("floor", "init_space"):
    gid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, gname)
    if gid >= 0:
      mj_model.geom_rgba[gid, 3] = 0.0
      mj_model.geom_matid[gid] = -1

  mjx_model = mjx.put_model(mj_model, impl="jax")
  key_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, "picked")
  d0 = mjx.make_data(mj_model, impl="jax").replace(
      qpos=jnp.asarray(mj_model.key_qpos[key_id], dtype=jnp.float32),
      ctrl=jnp.asarray(mj_model.key_ctrl[key_id], dtype=jnp.float32),
  )

  cam_pos = np.array([-6.9, 1.55, -0.75], dtype=np.float32)
  target = np.array([1.2, 1.85, 0.45], dtype=np.float32)
  cam_mat = look_at_cam_mat(cam_pos, target, world_up=(0.0, 1.0, 0.0))
  cam_fwd = -cam_mat[:, 2]
  fovy, znear = 50.0, 0.10

  sc = load_gltf_scene_cinema(SPONZA_GLTF, load_normal_maps=True)
  gs = sample_cinema_3dgs(
      sc, n_gaussians=n_gaussians, seed=42, splat_scale=1.30, thick_ratio=0.22,
      cam_pos=cam_pos, cam_fwd=cam_fwd, focus_radius=14.0,
  )

  # Golden-Hour sun slanting down into Sponza courtyard + warm amber colonnade torch lights
  sun_dir = np.array([0.36, 0.80, 0.48], dtype=np.float32)
  sun_dir /= np.linalg.norm(sun_dir)
  sun_color = np.array([4.4, 2.55, 1.25], dtype=np.float32)
  sky_color = np.array([0.42, 0.62, 1.10], dtype=np.float32)
  ground_color = np.array([0.36, 0.25, 0.16], dtype=np.float32)

  # Warm brazier / colonnade arch accent lights along Sponza lower arcade
  sponza_lights = []
  for x_l in (-9.0, -4.5, 0.0, 4.5, 9.0):
    for z_l in (-4.4, 4.4):
      sponza_lights.append({
          "name": f"arcade_lantern_{x_l}_{z_l}",
          "pos": np.array([x_l, 2.1, z_l], dtype=np.float32),
          "color": np.array([1.0, 0.56, 0.22], dtype=np.float32),
          "intensity": 135.0,
          "range": 6.5,
      })

  bmin = np.array([-18.5, -1.0, -11.5], dtype=np.float32)
  bmax = np.array([18.5, 15.5, 11.5], dtype=np.float32)
  baked = bake_cinema_lighting_tpu(
      gs["mu"], gs["nrm"], gs["albedo_lin"], gs["emissive_hdr"],
      sun_dir=sun_dir, sun_color=sun_color, sky_color=sky_color, ground_color=ground_color,
      point_lights=sponza_lights, voxel_bounds=(bmin, bmax),
      grid_shape=(320, 144, 224), shadow_dist=22.0, ao_dist=3.6, point_light_boost=1.0,
  )

  # Coordinate transform: MJX Z-up -> Sponza Y-up, Panda beside foreground pillar
  panda_scale = 1.45
  R_ms = jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=jnp.float32)
  ang_y = -0.45
  R_rot_y = jnp.array([
      [math.cos(ang_y), 0.0, math.sin(ang_y)],
      [0.0, 1.0, 0.0],
      [-math.sin(ang_y), 0.0, math.cos(ang_y)],
  ], dtype=jnp.float32)
  R_total = R_rot_y @ R_ms
  t_offset = jnp.array([-4.4, 0.0, 0.85], dtype=jnp.float32)

  devs = jax.devices()[:4]
  mesh = jax.sharding.Mesh(np.array(devs), ("dp",))
  rep = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

  def _make_hybrid_cinema_fn(W: int, H: int):
    spec = RenderSceneSpec.from_mjmodel(mj_model, cam_id=0, height=H, width=W, use_shadows=True)
    gs_4c_fn = make_cinema_gs3d_4c_fn(
        mesh, height=H, width=W, cam_fovy=fovy, znear=znear,
        k_macro=4096, k_tile=512, macro_rep=4, use_bf16_mxu=False,
    )

    @jax.jit
    def _unified_cinema_hybrid(d_in, mu, cov6, diff_hdr, op, nrm, spec_c, rough, sun_v, sky_v, cxp, cxm, sd, sc_c, sk_c):
      # A. Live MJX kinematics + physics step inside XLA
      d_step = mjx.step(mjx_model, d_in)
      g_xpos = (d_step.geom_xpos * panda_scale) @ R_total.T + t_offset[None, :]
      g_xmat = jnp.einsum("ij,gjk->gik", R_total, d_step.geom_xmat, precision=jax.lax.Precision.HIGHEST)
      g_size = jnp.asarray(spec.geom_size, dtype=jnp.float32) * panda_scale
      g_rgba = jnp.asarray(spec.geom_rgba, dtype=jnp.float32)

      l_pos = jnp.broadcast_to((sd * 20.0), (max(spec.nlight, 1), 3))
      l_dir = jnp.broadcast_to(-sd, (max(spec.nlight, 1), 3))

      ray_orig, ray_dir, neg_z = compute_camera_rays(cxp, cxm, fovy, H, W, znear=znear)
      dist, normal, geom_id = _cast_rays_tiled(
          spec, g_xpos, g_xmat, g_size, ray_orig, ray_dir,
          jnp.full((H * W,), spec.zfar, dtype=jnp.float32), tile_size=32768,
      )
      gb = _shade_scene(
          spec, g_xpos, g_xmat, g_size, g_rgba, cxp, cxm, l_pos, l_dir,
          ray_orig, ray_dir, neg_z, dist, normal, geom_id, mode="tiled",
      )
      mjx_z = jnp.where(gb.geom_id >= 0, gb.z_depth, -1.0)

      # Re-shade MJX Panda in matching Golden-Hour Linear HDR using G-Buffer albedo, normal, view_dir, shadow!
      p_alb_lin = jnp.where(gb.albedo <= 0.04045, gb.albedo / 12.92, jnp.power(jnp.maximum((gb.albedo + 0.055) / 1.055, 1e-6), 2.4))
      p_nrm = gb.normal
      p_vdir = gb.view_dir
      p_ndotl = jnp.maximum(0.0, jnp.sum(p_nrm * sd[None, None, :], axis=-1))
      p_ndotv = jnp.maximum(1e-3, jnp.sum(p_nrm * p_vdir, axis=-1))
      p_hvec = sd[None, None, :] + p_vdir
      p_hvec = p_hvec / jnp.maximum(jnp.linalg.norm(p_hvec, axis=-1, keepdims=True), 1e-12)
      p_ndoth = jnp.maximum(0.0, jnp.sum(p_nrm * p_hvec, axis=-1))
      p_hemi = 0.5 + 0.5 * p_nrm[..., 1]

      # Panda stays inside shaded/sunlit courtyard with GGX glossy white/orange-gold robot finish
      p_sun = sc_c[None, None, :] * (p_ndotl * gb.direct_shadow)[..., None] * 0.85
      p_sky = sk_c[None, None, :] * (0.35 + 0.65 * p_hemi)[..., None]
      p_gnd = jnp.array([0.38, 0.26, 0.16], dtype=jnp.float32)[None, None, :] * (1.0 - 0.5 * p_hemi)[..., None]
      p_spec = sc_c[None, None, :] * (0.28 * (p_ndoth ** 48.0) * gb.direct_shadow)[..., None]
      p_rim = sk_c[None, None, :] * (0.25 * ((1.0 - p_ndotv) ** 4.0))[..., None]
      mjx_rgb_hdr = p_alb_lin * (p_sun + p_sky + p_gnd) + p_spec + p_rim

      # B. 10M 3DGS Sponza HDR
      gs_rgb_prem, gs_z, gs_alp = gs_4c_fn(mu, cov6, diff_hdr, op, nrm, spec_c, rough, sun_v, sky_v, cxp, cxm, sd, sc_c, sk_c)

      # C. Exact Z-buffer HDR Compositor + Nishita Sky + Haze + Bloom + Blender AgX
      srgb, comp_depth = composite_cinema_hdr_and_tonemap(
          gs_rgb_prem, gs_alp, gs_z, ray_dir.reshape(H, W, 3), sd,
          fog_density=0.015, fog_color=(1.05, 0.60, 0.30),
          exposure=1.02, bloom_strength=0.11,
          mjx_rgb_hdr=mjx_rgb_hdr, mjx_depth=mjx_z,
      )
      return srgb, comp_depth, mjx_z, gs_z

    return _unified_cinema_hybrid

  args_4c = (
      jax.device_put(d0, rep),
      jax.device_put(jnp.asarray(gs["mu"]), rep),
      jax.device_put(jnp.asarray(gs["cov6"]), rep),
      jax.device_put(jnp.asarray(baked["diffuse_hdr"]), rep),
      jax.device_put(jnp.asarray(gs["opacity"]), rep),
      jax.device_put(jnp.asarray(gs["nrm"]), rep),
      jax.device_put(jnp.asarray(gs["spec_col"]), rep),
      jax.device_put(jnp.asarray(gs["roughness"]), rep),
      jax.device_put(jnp.asarray(baked["sun_vis"]), rep),
      jax.device_put(jnp.asarray(baked["sky_vis"]), rep),
      jax.device_put(jnp.asarray(cam_pos), rep),
      jax.device_put(jnp.asarray(cam_mat), rep),
      jax.device_put(jnp.asarray(sun_dir), rep),
      jax.device_put(jnp.asarray(sun_color), rep),
      jax.device_put(jnp.asarray(sky_color), rep),
  )

  res_list = [("1080p", 1920, 1080)] if preview_only else [("1080p", 1920, 1080), ("4K", 3840, 2160)]
  hybrid_runs = []
  for res_label, W, H in res_list:
    fn_h = _make_hybrid_cinema_fn(W, H)
    if res_label == "4K" or preview_only:
      hlo_hyb = _check_hlo_zero_host_comm(fn_h.lower(*args_4c))
      data.setdefault("hlo_checks", {})[f"hybrid_mjx_10m_3dgs_{res_label}"] = hlo_hyb
      print(f"Hybrid {res_label} HLO verified:", hlo_hyb, flush=True)

    ms, comp_s, out_t = _bench_fn(fn_h, args_4c, n_warmup=1, n_steps=3)
    srgb_img, _, mjx_z, gs_z = out_t
    mjx_pixels = int(jnp.sum(mjx_z > 0))
    occluded_by_3dgs = int(jnp.sum((mjx_z > 0) & (gs_z > 0) & (gs_z < mjx_z)))
    rec = {
        "n_chips": 4, "res_label": res_label, "res": f"{W}x{H}",
        "n_gaussians": n_gaussians, "k_tile": 512, "macro_rep": 4,
        "bake_s": round(float(baked["bake_s"]), 2),
        "ms": round(ms, 2), "fps": round(1000.0 / ms, 2), "compile_s": round(comp_s, 2),
        "mjx_robot_pixels": mjx_pixels,
        "mjx_pixels_occluded_by_3dgs_fg": occluded_by_3dgs,
        "cinema_overhaul": True,
    }
    hybrid_runs.append(rec)
    print(f"HYBRID CINEMA {res_label}:", rec, flush=True)

    if res_label == "4K" or preview_only:
      fname = "v3_w3_hybrid_mjx_3dgs_preview.png" if preview_only else "v3_w3_hybrid_mjx_3dgs_4k.png"
      out_png = os.path.join(export_dir, fname)
      _save_png(np.asarray(srgb_img), out_png)
      print(f"Saved Hybrid Cinema render: {out_png}", flush=True)

  data["hybrid_realtosim_4k"] = hybrid_runs


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--part", default="all", choices=["all", "bistro", "hybrid"])
  ap.add_argument("--preview", action="store_true")
  ap.add_argument("--n_gaussians", type=int, default=10_000_000)
  ap.add_argument("--out_json", default="/dev/shm/zaychik-tpu-mjx-w3/out/v3_w3_wave3_realtosim_hybrid.json")
  ap.add_argument("--export_dir", default="/dev/shm/zaychik-tpu-mjx-w3/out")
  args = ap.parse_args()

  os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
  os.makedirs(args.export_dir, exist_ok=True)
  if os.path.exists(args.out_json):
    with open(args.out_json) as f:
      data = json.load(f)
  else:
    data = {
        "backend": jax.default_backend(),
        "device_kind": jax.devices()[0].device_kind,
        "jax_version": jax.__version__,
    }

  if args.part in ("all", "bistro"):
    run_cinema_bistro(data, args.export_dir, n_gaussians=args.n_gaussians, preview_only=args.preview)
    with open(args.out_json, "w") as f:
      json.dump(data, f, indent=2)

  if args.part in ("all", "hybrid"):
    run_cinema_hybrid_4k(data, args.export_dir, n_gaussians=args.n_gaussians, preview_only=args.preview)
    with open(args.out_json, "w") as f:
      json.dump(data, f, indent=2)


if __name__ == "__main__":
  main()
