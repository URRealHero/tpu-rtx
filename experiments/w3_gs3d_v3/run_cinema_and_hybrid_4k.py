"""Wave 3 Cinema-Grade 4K Showcase & Hybrid Real2Sim Runner on 4-Chip TPU v5p.

Generates:
1. `v3_w3_cinema_3dgs_4k.png`: Native 4K (3840x2160) 10M-Gaussian Amazon Lumberyard Bistro
   with TPU Voxel Cone-Traced Golden-Hour Sun Soft Shadows + 12-Ray Hemisphere AO
   + Cool Sky Fill + Warm Tungsten Streetlamps (#ffd48a) + Dynamic GGX Specular/Rim
   + Nishita Golden-Hour Sky Dome + Subtle Far Depth Haze + Bloom + Blender AgX.
2. `v3_w3_hybrid_mjx_3dgs_4k.png`: Native 4K (3840x2160) Hybrid MJX Panda x 10M 3DGS Sponza
   with grounded dark matte manipulation table, red & blue tabletop cubes, contact shadows,
   4-chip sharded exact raycasting, exact per-pixel Z-buffer occlusion against Sponza
   foreground columns/urns, and unified Cinema Golden-Hour HDR + Blender AgX Tonemapping.
3. Updates `v3_w3_wave3_realtosim_hybrid.json` with verified benchmarks & HLO checks.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import re
import time
from typing import Any, Dict

import jax
from jax.experimental.shard_map import shard_map
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


def _build_hybrid_panda_xml() -> str:
  """Enhance panda_pick_cube XML with dark matte manipulation table, pedestal, orange collars & blue cube."""
  xml = get_scene_xml("panda_pick_cube")
  # Add industrial matte table + orange accent materials
  extra_mats = """    <material name="table_matte" rgba="0.09 0.10 0.12 1" specular="0.12" shininess="0.15"/>
    <material name="table_pedestal" rgba="0.14 0.15 0.17 1" specular="0.15" shininess="0.2"/>
    <material name="accent_orange" rgba="0.95 0.42 0.08 1" specular="0.45" shininess="0.4"/>
    <material name="cube_blue" rgba="0.10 0.42 0.90 1" specular="0.35" shininess="0.35"/>
"""
  xml = xml.replace('    <material name="strip_white"', extra_mats + '    <material name="strip_white"')

  # Replace flat init_space with grounded dark matte manipulation table + pedestal down to stone floor
  old_init = '<geom name="init_space" size="0.22 0.24 0.001" type="box" pos="0.631 0 0.001" material="strip_white" contype="0" conaffinity="0"/>'
  new_table = (
      '<geom name="init_space" size="0.54 0.40 0.024" type="box" pos="0.34 0.0 -0.024" material="table_matte" contype="0" conaffinity="0"/>\n'
      '    <geom name="table_pedestal" size="0.13 0.13 0.22" type="box" pos="0.34 0.0 -0.25" material="table_pedestal" contype="0" conaffinity="0"/>\n'
      '    <geom name="cube_blue_prop" size="0.028 0.028 0.028" type="box" pos="0.56 -0.17 0.028" material="cube_blue" contype="0" conaffinity="0"/>\n'
      '    <geom name="cube_red_prop2" size="0.024 0.024 0.024" type="box" pos="0.24 -0.22 0.024" material="cube_red" contype="0" conaffinity="0"/>'
  )
  assert old_init in xml
  xml = xml.replace(old_init, new_table)

  # Give joint3 & link5 collars subtle robotics orange accents for crisp white/grey/orange readability
  xml = xml.replace(
      '<geom name="link3_joint" type="sphere" pos="0.0825 0 0" size="0.056" material="black" class="visual_prim"/>',
      '<geom name="link3_joint" type="sphere" pos="0.0825 0 0" size="0.056" material="accent_orange" class="visual_prim"/>',
  )
  xml = xml.replace(
      '<geom name="link5_collar" type="cylinder" pos="0 0 0" size="0.046 0.025" material="black" class="visual_prim"/>',
      '<geom name="link5_collar" type="cylinder" pos="0 0 0" size="0.046 0.025" material="accent_orange" class="visual_prim"/>',
  )
  return xml


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

  # Unobstructed film-still camera positioned past streetlight_01c (x=-20.6, z=14.14),
  # framing the Bistro cafe facade, blue Vespa, warm tungsten bulbs, cobblestone street & open Golden-Hour Sky!
  cam_pos = np.array([-19.4, 1.88, 12.60], dtype=np.float32)
  target = np.array([-2.0, 4.45, 17.20], dtype=np.float32)
  cam_mat = look_at_cam_mat(cam_pos, target, world_up=(0.0, 1.0, 0.0))
  cam_fwd = -cam_mat[:, 2]
  fovy = 50.0

  print(f"Bistro camera pos={cam_pos.round(2)}, fwd={cam_fwd.round(3)}, fovy={fovy:.1f} deg", flush=True)

  # 0.80x tighter Gaussian scale for sharp 4K architectural edges
  gs = sample_cinema_3dgs(
      sc, n_gaussians=n_gaussians, seed=42, splat_scale=0.80, thick_ratio=0.18,
      cam_pos=cam_pos, cam_fwd=cam_fwd, focus_radius=24.0,
  )

  # Warm tungsten #ffd48a ([1.0, 0.83, 0.54] sRGB -> [1.0, 0.65, 0.25] linear) for streetlamps & cafe bulbs!
  warm_tungsten = np.array([1.0, 0.65, 0.25], dtype=np.float32)
  warm_lights = []
  for l in sc["point_lights"]:
    c = l["color"].copy()
    # Replace pure white or harsh neon green/magenta/cyan street point lights with warm tungsten cafe glow
    if float(np.min(c)) > 0.70 or "pointlight" in l["name"].lower():
      c = warm_tungsten
    warm_lights.append({**l, "color": c})

  # Crisp warm golden key sun vs cool blue sky fill
  sun_dir = np.array([0.52, 0.36, 0.77], dtype=np.float32)
  sun_dir /= np.linalg.norm(sun_dir)
  sun_color = np.array([4.3, 2.40, 1.05], dtype=np.float32)
  sky_color = np.array([0.19, 0.33, 0.66], dtype=np.float32)
  ground_color = np.array([0.14, 0.09, 0.06], dtype=np.float32)

  bmin = np.array([-42.0, -3.5, -24.0], dtype=np.float32)
  bmax = np.array([34.0, 24.0, 38.0], dtype=np.float32)
  baked = bake_cinema_lighting_tpu(
      gs["mu"], gs["nrm"], gs["albedo_lin"], gs["emissive_hdr"],
      sun_dir=sun_dir, sun_color=sun_color, sky_color=sky_color, ground_color=ground_color,
      point_lights=warm_lights, voxel_bounds=(bmin, bmax),
      grid_shape=(384, 144, 320), shadow_dist=32.0, ao_dist=4.5, point_light_boost=1.35,
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
          fog_density=0.0045, fog_start=18.0, max_fog=0.15,
          fog_color=(1.05, 0.58, 0.28),
          exposure=1.15, bloom_strength=0.055, bloom_threshold=1.45,
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
        "splat_scale": 0.80,
        "bake_s": round(float(baked["bake_s"]), 2),
        "ms": round(ms, 2),
        "fps": round(1000.0 / ms, 2),
        "compile_s": round(comp_s, 2),
        "features": [
            "voxel_cone_traced_soft_shadows",
            "12_ray_hemisphere_ao",
            "96_warm_tungsten_streetlamps",
            "ggx_specular_and_fresnel_rim",
            "nishita_golden_hour_sky_dome",
            "far_only_aerial_haze_max15pct",
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
  mj_model = mujoco.MjModel.from_xml_string(_build_hybrid_panda_xml())
  gid_floor = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
  if gid_floor >= 0:
    mj_model.geom_rgba[gid_floor, 3] = 0.0
    mj_model.geom_matid[gid_floor] = -1

  mjx_model = mjx.put_model(mj_model, impl="jax")
  key_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, "picked")
  d0 = mjx.make_data(mj_model, impl="jax").replace(
      qpos=jnp.asarray(mj_model.key_qpos[key_id], dtype=jnp.float32),
      ctrl=jnp.asarray(mj_model.key_ctrl[key_id], dtype=jnp.float32),
  )

  # Place camera & Panda table prominently at ~2.4m in Sponza courtyard foreground,
  # with foreground Sponza column/brazier framing & occluding the left edge of the table!
  cam_pos = np.array([-7.35, 1.40, -1.36], dtype=np.float32)
  target = np.array([2.5, 2.25, 0.25], dtype=np.float32)
  cam_mat = look_at_cam_mat(cam_pos, target, world_up=(0.0, 1.0, 0.0))
  cam_fwd = -cam_mat[:, 2]
  fovy, znear = 54.0, 0.10

  sc = load_gltf_scene_cinema(SPONZA_GLTF, load_normal_maps=True)
  gs = sample_cinema_3dgs(
      sc, n_gaussians=n_gaussians, seed=42, splat_scale=1.05, thick_ratio=0.18,
      cam_pos=cam_pos, cam_fwd=cam_fwd, focus_radius=14.0,
  )

  # Golden-Hour sun + cool blue sky fill + warm colonnade lantern glow
  sun_dir = np.array([0.36, 0.80, 0.48], dtype=np.float32)
  sun_dir /= np.linalg.norm(sun_dir)
  sun_color = np.array([4.6, 2.65, 1.25], dtype=np.float32)
  sky_color = np.array([0.22, 0.38, 0.74], dtype=np.float32)
  ground_color = np.array([0.18, 0.12, 0.08], dtype=np.float32)

  sponza_lights = []
  for x_l in (-9.0, -4.5, 0.0, 4.5, 9.0):
    for z_l in (-4.4, 4.4):
      sponza_lights.append({
          "name": f"arcade_lantern_{x_l}_{z_l}",
          "pos": np.array([x_l, 2.1, z_l], dtype=np.float32),
          "color": np.array([1.0, 0.62, 0.25], dtype=np.float32),
          "intensity": 190.0,
          "range": 6.8,
      })

  bmin = np.array([-18.5, -1.0, -11.5], dtype=np.float32)
  bmax = np.array([18.5, 15.5, 11.5], dtype=np.float32)
  baked = bake_cinema_lighting_tpu(
      gs["mu"], gs["nrm"], gs["albedo_lin"], gs["emissive_hdr"],
      sun_dir=sun_dir, sun_color=sun_color, sky_color=sky_color, ground_color=ground_color,
      point_lights=sponza_lights, voxel_bounds=(bmin, bmax),
      grid_shape=(320, 144, 224), shadow_dist=22.0, ao_dist=3.6, point_light_boost=1.0,
  )

  # MJX Z-up -> Sponza Y-up (right-handed: x->x, z->+y UP, y->-z), tabletop at y=0.52 (pedestal rests on Sponza floor y=0)!
  panda_scale = 1.25
  R_ms = jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=jnp.float32)
  ang_y = -0.65
  R_rot_y = jnp.array([
      [math.cos(ang_y), 0.0, math.sin(ang_y)],
      [0.0, 1.0, 0.0],
      [-math.sin(ang_y), 0.0, math.cos(ang_y)],
  ], dtype=jnp.float32)
  R_total = R_rot_y @ R_ms
  t_offset = jnp.array([-4.45, 0.52, -1.78], dtype=jnp.float32)

  devs = jax.devices()[:4]
  mesh = jax.sharding.Mesh(np.array(devs), ("dp",))
  rep = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
  P = jax.sharding.PartitionSpec
  sm_kw = {"check_rep": False} if "check_rep" in inspect.signature(shard_map).parameters else {"check_vma": False}

  def _make_hybrid_cinema_fn(W: int, H: int):
    assert H % 4 == 0
    spec_strip = RenderSceneSpec.from_mjmodel(mj_model, cam_id=0, height=H // 4, width=W, use_shadows=True)
    gs_4c_fn = make_cinema_gs3d_4c_fn(
        mesh, height=H, width=W, cam_fovy=fovy, znear=znear,
        k_macro=4096, k_tile=512, macro_rep=4, use_bf16_mxu=False,
    )

    def _mjx_strip(g_xpos, g_xmat, g_sz, g_rgba, cxp, cxm, l_pos, l_dir, ro_s, rd_s, nz_s):
      dist, normal, geom_id = _cast_rays_tiled(
          spec_strip, g_xpos, g_xmat, g_sz, ro_s, rd_s,
          jnp.full((ro_s.shape[0],), spec_strip.zfar, dtype=jnp.float32), tile_size=4096,
      )
      gb = _shade_scene(
          spec_strip, g_xpos, g_xmat, g_sz, g_rgba, cxp, cxm, l_pos, l_dir,
          ro_s, rd_s, nz_s, dist, normal, geom_id, mode="tiled",
      )
      return gb.albedo, gb.normal, gb.view_dir, gb.direct_shadow, gb.z_depth, gb.geom_id

    _sm_mjx = shard_map(
        _mjx_strip, mesh=mesh,
        in_specs=(P(), P(), P(), P(), P(), P(), P(), P(), P("dp"), P("dp"), P("dp")),
        out_specs=(P("dp"), P("dp"), P("dp"), P("dp"), P("dp"), P("dp")),
        **sm_kw,
    )

    @jax.jit
    def _unified_cinema_hybrid(d_in, mu, cov6, diff_hdr, op, nrm, spec_c, rough, sun_v, sky_v, cxp, cxm, sd, sc_c, sk_c):
      # A. Live MJX kinematics + physics step + 4-chip sharded analytic raycaster + shadow rays!
      d_step = mjx.step(mjx_model, d_in)
      g_xpos = (d_step.geom_xpos * panda_scale) @ R_total.T + t_offset[None, :]
      g_xmat = jnp.einsum("ij,gjk->gik", R_total, d_step.geom_xmat, precision=jax.lax.Precision.HIGHEST)
      g_size = jnp.asarray(spec_strip.geom_size, dtype=jnp.float32) * panda_scale
      g_rgba = jnp.asarray(spec_strip.geom_rgba, dtype=jnp.float32)

      l_pos = jnp.broadcast_to((sd * 20.0), (max(spec_strip.nlight, 1), 3))
      l_dir = jnp.broadcast_to(-sd, (max(spec_strip.nlight, 1), 3))

      ray_orig, ray_dir, neg_z = compute_camera_rays(cxp, cxm, fovy, H, W, znear=znear)
      alb_s, nrm_s, vd_s, sh_s, zd_s, gid_s = _sm_mjx(
          g_xpos, g_xmat, g_size, g_rgba, cxp, cxm, l_pos, l_dir, ray_orig, ray_dir, neg_z
      )
      alb = alb_s.reshape(H, W, 3)
      p_nrm = nrm_s.reshape(H, W, 3)
      p_vdir = vd_s.reshape(H, W, 3)
      p_sh = sh_s.reshape(H, W)
      mjx_z = jnp.where(gid_s.reshape(H, W) >= 0, zd_s.reshape(H, W), -1.0)

      # Re-shade MJX Panda + dark matte table in matching Golden-Hour Linear HDR
      p_alb_lin = jnp.where(alb <= 0.04045, alb / 12.92, jnp.power(jnp.maximum((alb + 0.055) / 1.055, 1e-6), 2.4))
      p_ndotl = jnp.maximum(0.0, jnp.sum(p_nrm * sd[None, None, :], axis=-1))
      p_ndotv = jnp.maximum(1e-3, jnp.sum(p_nrm * p_vdir, axis=-1))
      p_hvec = sd[None, None, :] + p_vdir
      p_hvec = p_hvec / jnp.maximum(jnp.linalg.norm(p_hvec, axis=-1, keepdims=True), 1e-12)
      p_ndoth = jnp.maximum(0.0, jnp.sum(p_nrm * p_hvec, axis=-1))
      p_hemi = 0.5 + 0.5 * p_nrm[..., 1]

      # Crisp direct golden-hour key + contact shadow + cool sky hemisphere fill + GGX specular
      p_sun = sc_c[None, None, :] * (p_ndotl * p_sh)[..., None] * 0.90
      p_sky = sk_c[None, None, :] * (0.28 + 0.72 * p_hemi)[..., None]
      p_gnd = jnp.array([0.16, 0.11, 0.07], dtype=jnp.float32)[None, None, :] * (1.0 - 0.5 * p_hemi)[..., None]
      p_spec = sc_c[None, None, :] * (0.22 * (p_ndoth ** 42.0) * p_sh)[..., None]
      p_rim = sk_c[None, None, :] * (0.18 * ((1.0 - p_ndotv) ** 4.0))[..., None]
      mjx_rgb_hdr = p_alb_lin * (p_sun + p_sky + p_gnd) + p_spec + p_rim

      # B. 10M 3DGS Sponza HDR
      gs_rgb_prem, gs_z, gs_alp = gs_4c_fn(mu, cov6, diff_hdr, op, nrm, spec_c, rough, sun_v, sky_v, cxp, cxm, sd, sc_c, sk_c)

      # C. Exact Z-buffer HDR Compositor + Nishita Sky + Subtle Far Haze + Bloom + Blender AgX
      srgb, comp_depth = composite_cinema_hdr_and_tonemap(
          gs_rgb_prem, gs_alp, gs_z, ray_dir.reshape(H, W, 3), sd,
          fog_density=0.005, fog_start=18.0, max_fog=0.14,
          fog_color=(1.02, 0.58, 0.28),
          exposure=1.12, bloom_strength=0.055, bloom_threshold=1.45,
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
