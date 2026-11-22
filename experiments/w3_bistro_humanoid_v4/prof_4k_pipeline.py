"""Profile 4K Bistro 3DGS + Hybrid MJX bottlenecks on 4-chip TPU v5p."""

from __future__ import annotations

import inspect
import os
import time

import jax
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
import numpy as np

from tpu_mjx_render.camera import compute_camera_rays
from cinema_3dgs import (
    bake_cinema_lighting_tpu,
    composite_cinema_hdr_and_tonemap,
    load_gltf_scene_cinema,
    make_cinema_gs3d_4c_fn,
    project_and_bucket_cinema,
    render_macro_tiles_hdr_subset,
    sample_cinema_3dgs,
)
from gltf_to_3dgs import look_at_cam_mat

BISTRO_GLTF = "/dev/shm/zaychik-tpu-mjx-w3/assets/niagara_bistro/bistro.gltf"


def main():
  devs = jax.devices()[:4]
  mesh = jax.sharding.Mesh(np.array(devs), ("dp",))
  rep = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
  P = jax.sharding.PartitionSpec
  sm_kw = {"check_rep": False} if "check_rep" in inspect.signature(shard_map).parameters else {"check_vma": False}

  sc = load_gltf_scene_cinema(BISTRO_GLTF, load_normal_maps=True)
  cam_pos = np.array([-19.4, 1.88, 12.60], dtype=np.float32)
  target = np.array([-2.0, 4.45, 17.20], dtype=np.float32)
  cam_mat = look_at_cam_mat(cam_pos, target, world_up=(0.0, 1.0, 0.0))
  cam_fwd = -cam_mat[:, 2]
  fovy = 50.0

  gs = sample_cinema_3dgs(
      sc, n_gaussians=10_000_000, seed=42, splat_scale=0.80, thick_ratio=0.18,
      cam_pos=cam_pos, cam_fwd=cam_fwd, focus_radius=24.0,
  )
  sun_dir = np.array([0.52, 0.36, 0.77], dtype=np.float32)
  sun_dir /= np.linalg.norm(sun_dir)
  sun_color = np.array([4.3, 2.40, 1.05], dtype=np.float32)
  sky_color = np.array([0.19, 0.33, 0.66], dtype=np.float32)
  ground_color = np.array([0.14, 0.09, 0.06], dtype=np.float32)

  # Fast dummy bake just to profile 4K render breakdown
  diff_hdr = jax.device_put(jnp.asarray(gs["albedo_lin"]), rep)
  sun_v = jax.device_put(jnp.ones((10_000_000,), dtype=jnp.float32), rep)
  sky_v = jax.device_put(jnp.ones((10_000_000,), dtype=jnp.float32), rep)

  mu = jax.device_put(jnp.asarray(gs["mu"]), rep)
  cov6 = jax.device_put(jnp.asarray(gs["cov6"]), rep)
  op = jax.device_put(jnp.asarray(gs["opacity"]), rep)
  nrm = jax.device_put(jnp.asarray(gs["nrm"]), rep)
  spec_c = jax.device_put(jnp.asarray(gs["spec_col"]), rep)
  rough = jax.device_put(jnp.asarray(gs["roughness"]), rep)
  cxp = jax.device_put(jnp.asarray(cam_pos), rep)
  cxm = jax.device_put(jnp.asarray(cam_mat), rep)
  sd = jax.device_put(jnp.asarray(sun_dir), rep)
  sc_c = jax.device_put(jnp.asarray(sun_color), rep)
  sk_c = jax.device_put(jnp.asarray(sky_color), rep)

  H, W = 2160, 3840
  M_SIZE = 32
  H_pad = ((H + M_SIZE - 1) // M_SIZE) * M_SIZE
  W_pad = ((W + M_SIZE - 1) // M_SIZE) * M_SIZE
  My, Mx = H_pad // M_SIZE, W_pad // M_SIZE
  M_tiles = My * Mx

  @jax.jit
  def _stage1_proj():
    return project_and_bucket_cinema(
        mu, cov6, diff_hdr, op, nrm, spec_c, rough, sun_v, sky_v,
        cxp, cxm, sd, sc_c, sk_c, height=H, width=W, cam_fovy=fovy, znear=0.12,
        k_macro=4096, macro_rep=4,
    )

  splats_n16, macro_gidx, _, _, _, _, _ = _stage1_proj()
  jax.block_until_ready(splats_n16)
  t0 = time.perf_counter()
  for _ in range(3):
    out1 = _stage1_proj()
  jax.block_until_ready(out1)
  print(f"Stage 1 (project_and_bucket_cinema 10M, 1 chip): {(time.perf_counter()-t0)*1e3/3:.2f} ms", flush=True)

  t_ox_np = np.tile(np.arange(Mx, dtype=np.float32) * 32.0, My)
  t_oy_np = np.repeat(np.arange(My, dtype=np.float32) * 32.0, Mx)
  t_ox_dev = jax.device_put(jnp.asarray(t_ox_np), jax.sharding.NamedSharding(mesh, P("dp")))
  t_oy_dev = jax.device_put(jnp.asarray(t_oy_np), jax.sharding.NamedSharding(mesh, P("dp")))

  for bf16 in (False, True):
    for kt in (512, 256):
      def _r_sub(s16, mg, tox, toy):
        return render_macro_tiles_hdr_subset(s16, mg, tox, toy, use_bf16_mxu=bf16, k_tile=kt)
      sm_r = shard_map(_r_sub, mesh=mesh, in_specs=(P(), P("dp"), P("dp"), P("dp")), out_specs=(P("dp"), P("dp"), P("dp")), **sm_kw)
      j_r = jax.jit(sm_r)
      o2 = j_r(splats_n16, macro_gidx, t_ox_dev, t_oy_dev)
      jax.block_until_ready(o2)
      t0 = time.perf_counter()
      for _ in range(3):
        o2 = j_r(splats_n16, macro_gidx, t_ox_dev, t_oy_dev)
      jax.block_until_ready(o2)
      print(f"Stage 2 (4c raster bf16={bf16}, k_tile={kt}): {(time.perf_counter()-t0)*1e3/3:.2f} ms", flush=True)

  rgb_m, z_m, alp_m = o2
  img_pad = rgb_m.reshape(My, Mx, 32, 32, 3).transpose(0, 2, 1, 3, 4).reshape(H_pad, W_pad, 3)[:H, :W]
  dep_pad = z_m.reshape(My, Mx, 32, 32).transpose(0, 2, 1, 3).reshape(H_pad, W_pad)[:H, :W]
  alp_pad = alp_m.reshape(My, Mx, 32, 32).transpose(0, 2, 1, 3).reshape(H_pad, W_pad)[:H, :W]

  @jax.jit
  def _stage3_comp(r, a, d):
    _, ray_dir, _ = compute_camera_rays(cxp, cxm, fovy, H, W, znear=0.12)
    return composite_cinema_hdr_and_tonemap(
        r, a, d, ray_dir.reshape(H, W, 3), sd,
        fog_density=0.0045, fog_start=18.0, max_fog=0.15,
        fog_color=(1.05, 0.58, 0.28), exposure=1.15, bloom_strength=0.055, bloom_threshold=1.45,
    )

  o3 = _stage3_comp(img_pad, alp_pad, dep_pad)
  jax.block_until_ready(o3)
  t0 = time.perf_counter()
  for _ in range(3):
    o3 = _stage3_comp(img_pad, alp_pad, dep_pad)
  jax.block_until_ready(o3)
  print(f"Stage 3 (composite + AgX 4K, 1 chip): {(time.perf_counter()-t0)*1e3/3:.2f} ms", flush=True)


if __name__ == "__main__":
  main()
