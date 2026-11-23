"""Worker w2: 4-chip TPU 5.83M SH3 Mip-NeRF 360 Garden 360-deg Orbit (1080p, 120 frames)."""

from __future__ import annotations

import argparse
import gc
import math
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
  sys.path.insert(0, REPO_ROOT)

from experiments.showcase_videos.video_utils import fit_mipnerf_orbit_basis, look_at_opencv, save_frames_tar
from experiments.w3_4dgs_robowild.fetch_mipnerf360 import fetch_scenes
from experiments.w3_4dgs_robowild.run_robowild_4dgs import _scaled_K, _shard_scene_with_mask
from tpu_mjx_render.adapters.tpu_gsplat import load_cameras_json, load_ply, make_sharded_rasterizer


def render_scene_orbit(data_dir: str, scene_name: str, clip_name: str, out_dir: str, n_frames: int = 120, W: int = 1920, H: int = 1080) -> None:
  fetch_scenes(data_dir, (scene_name,))
  devs = jax.devices()[:4]
  mesh = jax.sharding.Mesh(np.array(devs), ("dp",))
  rep = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

  sp = load_ply(os.path.join(data_dir, f"{scene_name}/point_cloud/iteration_30000/point_cloud.ply"))
  cams = load_cameras_json(os.path.join(data_dir, f"{scene_name}/cameras.json"))
  orbit = fit_mipnerf_orbit_basis(cams)
  c0 = orbit["c0"]
  K = _scaled_K(c0, W, H)

  if W >= 3840:
    fn = make_sharded_rasterizer(
        mesh, W, H, sh_degree=3, macro_size=64, macro_rep=4,
        k_macro=8192, k_large_per_chip=2048, k_tile=512,
    )
  else:
    fn = make_sharded_rasterizer(
        mesh, W, H, sh_degree=3, macro_size=32, macro_rep=5,
        k_macro=10240, k_large_per_chip=1536, k_tile=512,
    )
  m, q, s, o, sh, _, _ = _shard_scene_with_mask(sp, np.zeros((sp["n_gaussians"],), dtype=bool), mesh)
  k_dev = jax.device_put(jnp.asarray(K), rep)

  v0 = jax.device_put(jnp.asarray(c0["viewmat"]), rep)
  t0 = time.perf_counter()
  rgb0, _, _ = fn(m, q, s, o, sh, v0, k_dev)
  jax.block_until_ready(rgb0)
  print(f"[{scene_name}] Compiled {W}x{H} ({sp['n_gaussians']:,} splats) in {time.perf_counter() - t0:.1f}s", flush=True)

  frames_u8 = []
  t1 = time.perf_counter()
  target = orbit["target"]
  up_w = orbit["up_world"]
  u_ax = orbit["u_axis"]
  v_ax = orbit["v_axis"]
  rad = orbit["radius"] * 0.96
  h_base = orbit["height"]

  for i in range(n_frames):
    ang = 2.0 * math.pi * (i / float(n_frames))
    eye = target + rad * (math.cos(ang) * u_ax + math.sin(ang) * v_ax) + (h_base + 0.08 * rad * math.sin(2.0 * ang)) * up_w
    vm, _ = look_at_opencv(eye, target, up_w)
    vm_dev = jax.device_put(jnp.asarray(vm), rep)
    rgb, _, _ = fn(m, q, s, o, sh, vm_dev, k_dev)
    jax.block_until_ready(rgb)
    frames_u8.append(np.clip(np.round(np.asarray(rgb) * 255.0), 0, 255).astype(np.uint8))

  print(f"[{scene_name}] Rendered {n_frames} orbiting frames in {time.perf_counter() - t1:.1f}s", flush=True)
  save_frames_tar(frames_u8, out_dir, clip_name)


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--data_dir", default="/dev/shm/zaychik-tpu-rtx/mipnerf360")
  ap.add_argument("--out_dir", default="/dev/shm/zaychik-tpu-rtx/out_videos")
  ap.add_argument("--n_frames", type=int, default=90)
  args = ap.parse_args()
  os.makedirs(args.data_dir, exist_ok=True)
  render_scene_orbit(args.data_dir, "garden", "mipnerf_garden_4k", args.out_dir, n_frames=args.n_frames, W=3840, H=2160)


if __name__ == "__main__":
  main()
