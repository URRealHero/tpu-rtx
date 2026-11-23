"""Worker w3: 4-chip TPU Mip-NeRF 360 Bicycle & Bonsai 360-deg Orbits (1080p, 120 frames each)."""

from __future__ import annotations

import argparse
import gc
import os
import sys

import jax

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
  sys.path.insert(0, REPO_ROOT)

from experiments.showcase_videos.worker_w2_mipnerf_garden import render_scene_orbit


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--data_dir", default="/dev/shm/zaychik-tpu-rtx/mipnerf360")
  ap.add_argument("--out_dir", default="/dev/shm/zaychik-tpu-rtx/out_videos")
  ap.add_argument("--n_frames", type=int, default=90)
  args = ap.parse_args()
  os.makedirs(args.data_dir, exist_ok=True)

  render_scene_orbit(args.data_dir, "bonsai", "mipnerf_bonsai_4k", args.out_dir, n_frames=args.n_frames, W=3840, H=2160)
  jax.clear_caches()
  gc.collect()
  render_scene_orbit(args.data_dir, "bicycle", "mipnerf_bicycle_4k", args.out_dir, n_frames=args.n_frames, W=3840, H=2160)


if __name__ == "__main__":
  main()
