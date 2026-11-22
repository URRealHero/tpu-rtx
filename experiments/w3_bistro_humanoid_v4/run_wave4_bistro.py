"""Wave 4 Runner: Parisian Bistro 10M 3DGS x MJX (Humanoid-21DOF + Mobile Panda) @ 1080p/4K."""

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
try:
  from jax import shard_map
except ImportError:
  from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np
from PIL import Image

from tpu_mjx_render.camera import compute_camera_rays
from tpu_mjx_render.renderer import RenderSceneSpec
from tpu_mjx_render.roofline import inspect_hlo_ops

from cinema_3dgs import (
    bake_cinema_lighting_tpu,
    load_gltf_scene_cinema,
    sample_cinema_3dgs,
)
from gltf_to_3dgs import look_at_cam_mat
from bistro_hybrid_v4 import (
    build_bistro_embodiments_xml,
    composite_cinema_hdr_and_tonemap_4c,
    make_cinema_gs3d_4c_sharded_fn,
    pallas_raycast_mjx_4k_strip,
    set_bistro_pose,
)

BISTRO_GLTF = "/dev/shm/zaychik-tpu-mjx-w3/assets/niagara_bistro/bistro.gltf"


def _check_hlo_zero(lowered, name: str) -> Dict[str, Any]:
  txt = lowered.as_text()
  info = inspect_hlo_ops(txt)
  info["send_recv_ops"] = len(re.findall(r"\bsend\(|\brecv\(", txt, re.I))
  assert info["host_callbacks"] == 0, f"{name} host_callbacks != 0: {info}"
  assert info["send_recv_ops"] == 0, f"{name} send_recv_ops != 0: {info}"
  assert info["gather_ops"] == 0, f"{name} gather_ops != 0: {info}"
  info["zero_copy_verified"] = True
  return info


def _bench_fn(fn, args, n_warmup: int = 1, n_steps: int = 4):
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


def run_bistro_hybrid(
    n_gaussians: int = 10_000_000,
    preview_only: bool = False,
    k_tile: int = 512,
    k_macro: int = 4096,
    export_dir: str = "/dev/shm/zaychik-tpu-mjx-w3/out",
    out_json: str = "/dev/shm/zaychik-tpu-mjx-w3/out/v3_w3_wave4_bistro_humanoid.json",
) -> None:
  os.makedirs(export_dir, exist_ok=True)
  os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)

  devs = jax.devices()[:4]
  mesh = jax.sharding.Mesh(np.array(devs), ("dp",))
  rep = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
  dp_sh = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("dp"))
  P = jax.sharding.PartitionSpec
  sm_kw = {"check_rep": False} if "check_rep" in inspect.signature(shard_map).parameters else {"check_vma": False}

  print(f"=== Loading Bistro ({n_gaussians:,} Gaussians) & Building MJX Humanoid+MobilePanda ===", flush=True)
  sc = load_gltf_scene_cinema(BISTRO_GLTF, load_normal_maps=True)

  cam_pos = np.array([-19.4, 1.88, 12.60], dtype=np.float32)
  target = np.array([-2.5, 3.55, 16.95], dtype=np.float32)
  cam_mat = look_at_cam_mat(cam_pos, target, world_up=(0.0, 1.0, 0.0))
  cam_fwd = -cam_mat[:, 2]
  fovy, znear = 50.0, 0.12

  gs = sample_cinema_3dgs(
      sc, n_gaussians=n_gaussians, seed=42, splat_scale=0.80, thick_ratio=0.18,
      cam_pos=cam_pos, cam_fwd=cam_fwd, focus_radius=24.0,
  )

  # Shuffle Gaussians once so each of the 4 chips owns a uniform 25% spatial sample
  rng_shuf = np.random.default_rng(123)
  perm = rng_shuf.permutation(n_gaussians)
  for k in ("mu", "cov6", "albedo_lin", "emissive_hdr", "opacity", "nrm", "spec_col", "roughness"):
    gs[k] = np.ascontiguousarray(gs[k][perm])

  warm_tungsten = np.array([1.0, 0.65, 0.25], dtype=np.float32)
  warm_lights = []
  for l in sc["point_lights"]:
    c = l["color"].copy()
    if float(np.min(c)) > 0.70 or "pointlight" in l["name"].lower():
      c = warm_tungsten
    warm_lights.append({**l, "color": c})

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

  # Build MJX Humanoid-21DOF + Mobile Panda
  mj_model = mujoco.MjModel.from_xml_string(build_bistro_embodiments_xml())
  mjx_model = mjx.put_model(mj_model, impl="jax")
  qpos0 = set_bistro_pose(mj_model)
  d0 = mjx.make_data(mj_model, impl="jax").replace(qpos=jnp.asarray(qpos0, dtype=jnp.float32))
  d0 = mjx.forward(mjx_model, d0)

  # Map MJX Z-up -> Bistro Y-up, standing flush on cobblestone street y=0.22
  robot_scale = 1.05
  R_ms = jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=jnp.float32)
  ang_y = -0.05
  R_rot_y = jnp.array([
      [math.cos(ang_y), 0.0, math.sin(ang_y)],
      [0.0, 1.0, 0.0],
      [-math.sin(ang_y), 0.0, math.cos(ang_y)],
  ], dtype=jnp.float32)
  R_total = R_rot_y @ R_ms
  t_offset = jnp.array([-14.25, 0.22, 13.85], dtype=jnp.float32)

  # Shard 10M Gaussians across the 4 chips
  args_gs = (
      jax.device_put(jnp.asarray(gs["mu"]), dp_sh),
      jax.device_put(jnp.asarray(gs["cov6"]), dp_sh),
      jax.device_put(jnp.asarray(baked["diffuse_hdr"]), dp_sh),
      jax.device_put(jnp.asarray(gs["opacity"]), dp_sh),
      jax.device_put(jnp.asarray(gs["nrm"]), dp_sh),
      jax.device_put(jnp.asarray(gs["spec_col"]), dp_sh),
      jax.device_put(jnp.asarray(gs["roughness"]), dp_sh),
      jax.device_put(jnp.asarray(baked["sun_vis"]), dp_sh),
      jax.device_put(jnp.asarray(baked["sky_vis"]), dp_sh),
  )
  args_common = (
      jax.device_put(d0, rep),
      *args_gs,
      jax.device_put(jnp.asarray(cam_pos), rep),
      jax.device_put(jnp.asarray(cam_mat), rep),
      jax.device_put(jnp.asarray(sun_dir), rep),
      jax.device_put(jnp.asarray(sun_color), rep),
      jax.device_put(jnp.asarray(sky_color), rep),
  )

  if os.path.exists(out_json):
    with open(out_json) as f:
      data = json.load(f)
  else:
    data = {
        "backend": jax.default_backend(),
        "device_kind": jax.devices()[0].device_kind,
        "jax_version": jax.__version__,
    }

  res_list = [("1080p", 1920, 1080)] if preview_only else [("1080p", 1920, 1080), ("4K", 3840, 2160)]
  runs = []

  for res_label, W, H in res_list:
    assert H % 4 == 0
    spec_strip = RenderSceneSpec.from_mjmodel(mj_model, cam_id=0, height=H // 4, width=W, use_shadows=True)
    gs_4c_fn = make_cinema_gs3d_4c_sharded_fn(
        mesh, height=H, width=W, cam_fovy=fovy, znear=znear,
        k_macro=k_macro, k_tile=k_tile, macro_rep=4, use_bf16_mxu=False,
    )

    def _mjx_pallas_strip(g_xpos, g_xmat, g_sz, g_rgba, cxp, cxm, sd, ro_s, rd_s, nz_s):
      return pallas_raycast_mjx_4k_strip(
          spec_strip, g_xpos, g_xmat, g_sz, g_rgba, cxp, cxm, sd, ro_s, rd_s, nz_s, ground_y=0.22
      )

    _sm_mjx = shard_map(
        _mjx_pallas_strip, mesh=mesh,
        in_specs=(P(), P(), P(), P(), P(), P(), P(), P("dp"), P("dp"), P("dp")),
        out_specs=(P("dp"), P("dp"), P("dp"), P("dp"), P("dp"), P("dp"), P("dp")),
        **sm_kw,
    )

    @jax.jit
    def _render_mjx_only(d_in, cxp, cxm, sd):
      d_step = mjx.step(mjx_model, d_in)
      g_xpos = (d_step.geom_xpos * robot_scale) @ R_total.T + t_offset[None, :]
      g_xmat = jnp.einsum("ij,gjk->gik", R_total, d_step.geom_xmat, precision=jax.lax.Precision.HIGHEST)
      g_size = jnp.asarray(spec_strip.geom_size, dtype=jnp.float32) * robot_scale
      g_rgba = jnp.asarray(spec_strip.geom_rgba, dtype=jnp.float32)
      ray_orig, ray_dir, neg_z = compute_camera_rays(cxp, cxm, fovy, H, W, znear=znear)
      return _sm_mjx(g_xpos, g_xmat, g_size, g_rgba, cxp, cxm, sd, ray_orig, ray_dir, neg_z)

    mjx_ms, mjx_comp_s, _ = _bench_fn(
        _render_mjx_only,
        (args_common[0], args_common[-5], args_common[-4], args_common[-3]),
        n_warmup=1, n_steps=4,
    )
    print(f"MJX Pallas 4c {res_label} standalone: {mjx_ms:.2f} ms (compile {mjx_comp_s:.2f}s)", flush=True)

    @jax.jit
    def _unified_bistro_hybrid(
        d_in, mu, cov6, diff_hdr, op, nrm, spec_c, rough, sun_v, sky_v, cxp, cxm, sd, sc_c, sk_c
    ):
      d_step = mjx.step(mjx_model, d_in)
      g_xpos = (d_step.geom_xpos * robot_scale) @ R_total.T + t_offset[None, :]
      g_xmat = jnp.einsum("ij,gjk->gik", R_total, d_step.geom_xmat, precision=jax.lax.Precision.HIGHEST)
      g_size = jnp.asarray(spec_strip.geom_size, dtype=jnp.float32) * robot_scale
      g_rgba = jnp.asarray(spec_strip.geom_rgba, dtype=jnp.float32)

      ray_orig, ray_dir, neg_z = compute_camera_rays(cxp, cxm, fovy, H, W, znear=znear)
      alb_s, nrm_s, zd_s, rsh_s, gsh_s, gz_s, vd_s = _sm_mjx(
          g_xpos, g_xmat, g_size, g_rgba, cxp, cxm, sd, ray_orig, ray_dir, neg_z
      )
      alb = alb_s.reshape(H, W, 3)
      p_nrm = nrm_s.reshape(H, W, 3)
      p_vdir = vd_s.reshape(H, W, 3)
      p_sh = rsh_s.reshape(H, W)
      mjx_z = zd_s.reshape(H, W)
      gnd_sh = gsh_s.reshape(H, W)
      gnd_z = gz_s.reshape(H, W)

      # Golden-Hour Linear HDR PBR shading matching Bistro 3DGS
      p_alb_lin = jnp.where(alb <= 0.04045, alb / 12.92, jnp.power(jnp.maximum((alb + 0.055) / 1.055, 1e-6), 2.4))
      p_ndotl = jnp.maximum(0.0, jnp.sum(p_nrm * sd[None, None, :], axis=-1))
      p_ndotv = jnp.maximum(1e-3, jnp.sum(p_nrm * p_vdir, axis=-1))
      p_hvec = sd[None, None, :] + p_vdir
      p_hvec = p_hvec / jnp.maximum(jnp.linalg.norm(p_hvec, axis=-1, keepdims=True), 1e-12)
      p_ndoth = jnp.maximum(0.0, jnp.sum(p_nrm * p_hvec, axis=-1))
      p_hemi = 0.5 + 0.5 * p_nrm[..., 1]

      # Warm tungsten bounce from Bistro cafe lamps on left side
      p_cafe_bounce = jnp.array([0.45, 0.28, 0.11], dtype=jnp.float32)[None, None, :] * jnp.clip(0.5 - 0.5 * p_nrm[..., 2], 0.15, 1.0)[..., None]
      p_sun = sc_c[None, None, :] * (p_ndotl * p_sh)[..., None] * 0.92
      p_sky = sk_c[None, None, :] * (0.28 + 0.72 * p_hemi)[..., None]
      p_gnd = jnp.array([0.16, 0.11, 0.07], dtype=jnp.float32)[None, None, :] * (1.0 - 0.5 * p_hemi)[..., None]
      p_spec = sc_c[None, None, :] * (0.26 * (p_ndoth ** 46.0) * p_sh)[..., None]
      p_rim = sk_c[None, None, :] * (0.20 * ((1.0 - p_ndotv) ** 4.0))[..., None]
      mjx_rgb_hdr = p_alb_lin * (p_sun + p_sky + p_gnd + p_cafe_bounce) + p_spec + p_rim

      gs_rgb_prem, gs_z, gs_alp = gs_4c_fn(
          mu, cov6, diff_hdr, op, nrm, spec_c, rough, sun_v, sky_v, cxp, cxm, sd, sc_c, sk_c
      )

      srgb, comp_depth = composite_cinema_hdr_and_tonemap_4c(
          gs_rgb_prem, gs_alp, gs_z, ray_dir.reshape(H, W, 3), sd,
          mjx_rgb_hdr=mjx_rgb_hdr, mjx_depth=mjx_z, gnd_sh_mask=gnd_sh, gnd_z=gnd_z,
          fog_density=0.0045, fog_start=18.0, max_fog=0.15,
          fog_color=(1.05, 0.58, 0.28), exposure=1.15, bloom_strength=0.055, bloom_threshold=1.45,
      )
      return srgb, comp_depth, mjx_z, gs_z, gnd_sh

    hlo_info = _check_hlo_zero(_unified_bistro_hybrid.lower(*args_common), f"bistro_hybrid_{res_label}")
    data.setdefault("hlo_checks", {})[f"bistro_hybrid_10m_{res_label}_k{k_tile}"] = hlo_info
    print(f"Bistro Hybrid {res_label} HLO verified:", hlo_info, flush=True)

    ms, comp_s, out_t = _bench_fn(_unified_bistro_hybrid, args_common, n_warmup=1, n_steps=4)
    srgb_img, _, mjx_z, gs_z, gnd_sh = out_t
    mjx_pixels = int(jnp.sum(mjx_z > 0))
    occluded = int(jnp.sum((mjx_z > 0) & (gs_z > 0) & (gs_z < mjx_z)))
    shadow_pixels = int(jnp.sum((gnd_sh > 0.5) & (gs_z > 0)))
    rec = {
        "scene": "Amazon_Lumberyard_Bistro_Humanoid_MobilePanda",
        "n_chips": 4,
        "res_label": res_label,
        "res": f"{W}x{H}",
        "n_gaussians": n_gaussians,
        "k_macro": k_macro,
        "k_tile": k_tile,
        "bake_s": round(float(baked["bake_s"]), 2),
        "mjx_pallas_ms": round(mjx_ms, 2),
        "total_hybrid_ms": round(ms, 2),
        "fps": round(1000.0 / ms, 2),
        "speedup_vs_wave3_4k": round(2299.64 / ms, 2) if res_label == "4K" else None,
        "compile_s": round(comp_s, 2),
        "mjx_robot_pixels": mjx_pixels,
        "mjx_pixels_occluded_by_3dgs_fg": occluded,
        "cast_shadow_pixels_on_street": shadow_pixels,
    }
    runs.append(rec)
    print(f"BISTRO HYBRID {res_label}:", rec, flush=True)

    fname = "v3_w3_bistro_humanoid_preview.png" if preview_only else (
        f"v3_w3_bistro_humanoid_{res_label.lower()}.png"
    )
    out_png = os.path.join(export_dir, fname)
    _save_png(np.asarray(srgb_img), out_png)
    print(f"Saved render: {out_png}", flush=True)

  data["bistro_humanoid_hybrid"] = runs
  with open(out_json, "w") as f:
    json.dump(data, f, indent=2)
  print(f"Updated metrics: {out_json}", flush=True)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--preview", action="store_true")
  ap.add_argument("--n_gaussians", type=int, default=10_000_000)
  ap.add_argument("--k_tile", type=int, default=512)
  ap.add_argument("--k_macro", type=int, default=4096)
  args = ap.parse_args()
  run_bistro_hybrid(
      n_gaussians=args.n_gaussians,
      preview_only=args.preview,
      k_tile=args.k_tile,
      k_macro=args.k_macro,
  )


if __name__ == "__main__":
  main()
