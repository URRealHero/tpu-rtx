"""Worker w7: 4-chip TPU Physical Wave Optics & Infinite Mirror Path-Traced Cinema (640x640, 64 spp)."""

from __future__ import annotations

import argparse
import gc
import math
import os
import sys
import time

import jax
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for sub in ("", "experiments/w2_mesh_1m_bvh", "experiments/w7_pathtracer_1b", "experiments/w7_thinfilm_mirror", "experiments/w7_dispersion_sss"):
  p = os.path.join(REPO_ROOT, sub) if sub else REPO_ROOT
  if p not in sys.path:
    sys.path.insert(0, p)

from complex_scenes_pt import _look_at_mat
from experiments.showcase_videos.video_utils import save_frames_tar, smoothstep


def _stitch_any(out_arr) -> np.ndarray:
  arr = np.asarray(jax.device_get(out_arr), dtype=np.float32)
  if arr.ndim == 3:
    return arr
  return arr.transpose(1, 0, 2, 3).reshape(arr.shape[1], arr.shape[0] * arr.shape[2], arr.shape[3])
from run_dispersion_sss import (
    DispersionConfig,
    SSSConfig,
    build_dispersion_prism_scene,
    build_dispersion_sss_pathtracer,
    build_sss_sculpture_scene,
)
from run_thinfilm_mirror import (
    OpticsConfig,
    build_infinite_mirror_gallery_scene,
    build_thinfilm_aniso_scene,
    build_wave7_pathtracer,
)


def _render_orbit_sequence(fn, rep, devs, start_eye, target, az_sweep_deg: float, n_frames: int, clip_name: str, out_dir: str, z_bob: float = 0.05) -> None:
  e0 = np.asarray(start_eye, dtype=np.float64)
  t = np.asarray(target, dtype=np.float64)
  rel = e0 - t
  r_xy = float(np.linalg.norm(rel[:2]))
  ang0 = math.atan2(rel[1], rel[0])
  z0 = float(e0[2])

  cxp0, cxm0 = _look_at_mat(tuple(float(v) for v in e0), tuple(float(v) for v in t))
  rng_fixed = jax.device_put(jax.random.PRNGKey(42), rep)

  tc0 = time.perf_counter()
  y0 = fn(jax.device_put(cxp0, rep), jax.device_put(cxm0, rep), rng_fixed)
  jax.block_until_ready(y0)
  print(f"[w7:{clip_name}] Compiled in {time.perf_counter() - tc0:.1f}s", flush=True)

  frames_u8 = []
  t0 = time.perf_counter()
  for i in range(n_frames):
    u = i / float(n_frames)
    # Ping-pong smooth sinusoidal sway around hero center so motion loops cleanly
    sway = math.sin(2.0 * math.pi * u) * math.radians(az_sweep_deg)
    ang = ang0 + sway
    eye = (
        float(t[0] + r_xy * math.cos(ang)),
        float(t[1] + r_xy * math.sin(ang)),
        float(z0 + z_bob * math.cos(2.0 * math.pi * u)),
    )
    cxp, cxm = _look_at_mat(eye, tuple(float(v) for v in t))
    y = fn(jax.device_put(cxp, rep), jax.device_put(cxm, rep), rng_fixed)
    jax.block_until_ready(y)
    img = _stitch_any(y)
    frames_u8.append(np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8))

  print(f"[w7:{clip_name}] Rendered {n_frames} frames in {time.perf_counter() - t0:.1f}s", flush=True)
  save_frames_tar(frames_u8, out_dir, clip_name)


def _render_glide_sequence(fn, rep, n_frames: int, clip_name: str, out_dir: str) -> None:
  rng_fixed = jax.device_put(jax.random.PRNGKey(42), rep)
  cxp0, cxm0 = _look_at_mat((-0.46, -1.16, 0.86), (0.18, 0.68, 0.68))

  tc0 = time.perf_counter()
  y0 = fn(jax.device_put(cxp0, rep), jax.device_put(cxm0, rep), rng_fixed)
  jax.block_until_ready(y0)
  print(f"[w7:{clip_name}] Compiled in {time.perf_counter() - tc0:.1f}s", flush=True)

  frames_u8 = []
  t0 = time.perf_counter()
  for i in range(n_frames):
    u = i / float(n_frames)
    s = smoothstep(2.0 * u if u < 0.5 else 2.0 * (1.0 - u))
    eye = (
        float(-0.46 + 0.56 * s),
        float(-1.16 + 0.42 * s),
        float(0.86 - 0.14 * s),
    )
    tgt = (float(0.18 - 0.12 * s), 0.68, float(0.68 - 0.04 * s))
    cxp, cxm = _look_at_mat(eye, tgt)
    y = fn(jax.device_put(cxp, rep), jax.device_put(cxm, rep), rng_fixed)
    jax.block_until_ready(y)
    img = _stitch_any(y)
    frames_u8.append(np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8))

  print(f"[w7:{clip_name}] Rendered {n_frames} frames in {time.perf_counter() - t0:.1f}s", flush=True)
  save_frames_tar(frames_u8, out_dir, clip_name)


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--out_dir", default="/dev/shm/zaychik-tpu-rtx/out_videos")
  ap.add_argument("--res", type=int, default=768)
  ap.add_argument("--spp", type=int, default=96)
  ap.add_argument("--n_frames", type=int, default=90)
  args = ap.parse_args()

  os.makedirs(args.out_dir, exist_ok=True)
  devs = jax.devices()[:4]
  H = W = args.res

  # 1. Airy Thin-Film Soap Bubbles & Brushed Anisotropic Titanium
  bvh_tf, _, _, fovy_tf, zn_tf, zf_tf, rig_tf = build_thinfilm_aniso_scene()
  fn_tf, _, rep_tf = build_wave7_pathtracer(
      bvh_tf, devs, H, W, fovy_tf, zn_tf, zf_tf, rig_tf,
      spp=args.spp, max_bounces=6, opt_cfg=OpticsConfig(thinfilm=True, anisotropic=True, alpha_u=0.05, alpha_v=0.45),
  )
  _render_orbit_sequence(fn_tf, rep_tf, devs, (0.0, -2.02, 0.96), (0.0, -0.02, 0.28), 32.0, args.n_frames, "optics_thinfilm_orbit", args.out_dir, z_bob=0.08)
  del fn_tf, bvh_tf
  jax.clear_caches()
  gc.collect()

  # 2. Backlit Volumetric Random-Walk Subsurface Scattering (Jade Trefoil + Beeswax Vessel)
  bvh_s, _, _, fovy_s, zn_s, zf_s, rig_s = build_sss_sculpture_scene()
  fn_s, _, rep_s = build_dispersion_sss_pathtracer(
      bvh_s, devs, H, W, fovy_s, zn_s, zf_s, rig_s,
      spp=args.spp, max_bounces=4, disp_cfg=None, sss_cfg=SSSConfig(enabled=True, n_walk_steps=16, g_sss=0.35),
  )
  _render_orbit_sequence(fn_s, rep_s, devs, (0.02, -1.78, 0.68), (0.02, 0.05, 0.38), 30.0, args.n_frames, "optics_sss_jade_orbit", args.out_dir, z_bob=0.06)
  del fn_s, bvh_s
  jax.clear_caches()
  gc.collect()

  # 3. 6-Band Cauchy Spectral Prism Dispersion + Rainbow Caustics
  bvh_d, _, _, fovy_d, zn_d, zf_d, rig_d = build_dispersion_prism_scene()
  fn_d, _, rep_d = build_dispersion_sss_pathtracer(
      bvh_d, devs, H, W, fovy_d, zn_d, zf_d, rig_d,
      spp=args.spp, max_bounces=6, disp_cfg=DispersionConfig(enabled=True, caustics=True), sss_cfg=None,
  )
  _render_orbit_sequence(fn_d, rep_d, devs, (-0.04, -2.02, 1.24), (0.02, -0.08, 0.08), 28.0, args.n_frames, "optics_prism_orbit", args.out_dir, z_bob=0.06)
  del fn_d, bvh_d
  jax.clear_caches()
  gc.collect()

  # 4. 12-Bounce Infinite Mirror Corridor Glide
  bvh_mg, _, _, fovy_mg, zn_mg, zf_mg, rig_mg = build_infinite_mirror_gallery_scene()
  fn_mg, _, rep_mg = build_wave7_pathtracer(
      bvh_mg, devs, H, W, fovy_mg, zn_mg, zf_mg, rig_mg,
      spp=args.spp, max_bounces=12, opt_cfg=OpticsConfig(thinfilm=False, anisotropic=False),
  )
  _render_glide_sequence(fn_mg, rep_mg, args.n_frames, "optics_mirror_glide", args.out_dir)


if __name__ == "__main__":
  main()
