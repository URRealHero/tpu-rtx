"""Worker w4: Zero-Raytrace Interactive Neural Relighting Sweeps (512x512 -> 1024, 120 frames x2)."""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for sub in ("", "experiments/w2_mesh_1m_bvh", "experiments/w4_wave5_relight_quant", "experiments/w7_pathtracer_1b", "experiments/showcase_videos"):
  p = os.path.join(REPO_ROOT, sub) if sub else REPO_ROOT
  if p not in sys.path:
    sys.path.insert(0, p)

from bvh_pbr_pt import build_complex_pbr_pathtracer
from complex_scenes_pt import build_classroom_sunlit, build_cornell_workshop
from experiments.showcase_videos.video_utils import save_frames_tar, smoothstep
from experiments.showcase_videos.worker_w7_wave_optics_cinema import _render_orbit_sequence
from relight_models import relight_forward_float_or_qat
from run_wave5_relight_quant import train_relighter
from scene_relight_data import (
    MATERIAL_PRESETS,
    TOD_PRESETS,
    generate_relight_dataset,
    make_lighting_params,
    pack_cond_vector,
)


def _interp_color(c0, c1, t: float):
  a = np.asarray(c0, dtype=np.float32)
  b = np.asarray(c1, dtype=np.float32)
  return tuple(float(v) for v in ((1.0 - t) * a + t * b))


def _build_tod_cond_sequence(n_frames: int) -> np.ndarray:
  """Smooth cyclic loop across Morning -> Noon -> Golden Sunset -> Blue Twilight -> Morning."""
  n_p = len(TOD_PRESETS)
  mat_def = MATERIAL_PRESETS[1]
  conds = []
  for i in range(n_frames):
    u = (i / float(n_frames)) * n_p
    idx0 = int(math_floor := int(u)) % n_p
    idx1 = (idx0 + 1) % n_p
    frac = smoothstep(u - math_floor)
    p0, p1 = TOD_PRESETS[idx0], TOD_PRESETS[idx1]
    az = (1.0 - frac) * p0["az_deg"] + frac * p1["az_deg"]
    el = (1.0 - frac) * p0["el_deg"] + frac * p1["el_deg"]
    temp_k = (1.0 - frac) * p0["temp_k"] + frac * p1["temp_k"]
    s_sc = (1.0 - frac) * p0["sun_scale"] + frac * p1["sun_scale"]
    a_sc = (1.0 - frac) * p0["area_scale"] + frac * p1["area_scale"]
    k_sc = (1.0 - frac) * p0["sky_scale"] + frac * p1["sky_scale"]
    lp = make_lighting_params("classroom", az, el, temp_k, s_sc, a_sc, k_sc)
    conds.append(
        pack_cond_vector(
            lp["sun_dir"], lp["sun_emit"], lp["sky_zenith"], lp["sky_horizon"],
            lp["area_emit"], float(lp["temp_norm"]),
            mat_def["roughness"], mat_def["metallic"], mat_def["albedo"], scene_id=0,
        )
    )
  return np.stack(conds, axis=0).astype(np.float32)


def _build_material_cond_sequence(n_frames: int) -> np.ndarray:
  """Smooth cyclic loop across Matte Clay -> Satin -> Brushed Gold -> Chrome -> Matte Clay."""
  n_m = len(MATERIAL_PRESETS)
  lp = make_lighting_params("cornell_workshop", -95.0, 34.0, 5600.0, 1.05, 1.0, 1.05)
  conds = []
  for i in range(n_frames):
    u = (i / float(n_frames)) * n_m
    idx0 = int(u) % n_m
    idx1 = (idx0 + 1) % n_m
    frac = smoothstep(u - int(u))
    m0, m1 = MATERIAL_PRESETS[idx0], MATERIAL_PRESETS[idx1]
    rgh = (1.0 - frac) * m0["roughness"] + frac * m1["roughness"]
    met = (1.0 - frac) * m0["metallic"] + frac * m1["metallic"]
    alb = _interp_color(m0["albedo"], m1["albedo"], frac)
    conds.append(
        pack_cond_vector(
            lp["sun_dir"], lp["sun_emit"], lp["sky_zenith"], lp["sky_horizon"],
            lp["area_emit"], float(lp["temp_norm"]), rgh, met, alb, scene_id=1,
        )
    )
  return np.stack(conds, axis=0).astype(np.float32)


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--work_dir", default="/dev/shm/zaychik-tpu-rtx/relight")
  ap.add_argument("--out_dir", default="/dev/shm/zaychik-tpu-rtx/out_videos")
  ap.add_argument("--n_frames", type=int, default=150)
  args = ap.parse_args()

  os.makedirs(args.work_dir, exist_ok=True)
  os.makedirs(args.out_dir, exist_ok=True)
  devs = jax.devices()[:4]

  ds_npz = os.path.join(args.work_dir, "relight_ds.npz")
  generate_relight_dataset(
      devs, ds_npz, train_res=256, test_res=512, gt_spp=64,
      n_train_per_scene=24, n_test_rand_per_scene=2, max_bounces=4,
  )
  ds = np.load(ds_npz, allow_pickle=False)

  bf16_params, n_params, train_s = train_relighter(
      ds["tr_gb"], ds["tr_gt"], ds["tr_cond"],
      epochs=240, batch_size=16, patch_size=160, lr=1.8e-3, qat=False,
  )
  print(f"[w4] Trained BF16 Relighter ({n_params:,} params) in {train_s:.1f}s", flush=True)

  dev0 = devs[0]
  p_d = jax.device_put(jax.tree_util.tree_map(jnp.asarray, bf16_params), dev0)

  @jax.jit
  def _infer_chunk(p, gb_static_1, cond_b):
    gb_b = jnp.broadcast_to(gb_static_1, (cond_b.shape[0],) + gb_static_1.shape[1:])
    return relight_forward_float_or_qat(p, gb_b, cond_b, precision_mode="bf16", qat=False)

  def _render_sweep(gb_1_np: np.ndarray, conds_np: np.ndarray, clip_name: str) -> None:
    gb_d = jax.device_put(gb_1_np[:1].astype(np.float32), dev0)
    frames_u8 = []
    chunk = 8
    t0 = time.perf_counter()
    for start in range(0, len(conds_np), chunk):
      c_b = jax.device_put(conds_np[start : start + chunk], dev0)
      pred = _infer_chunk(p_d, gb_d, c_b)
      jax.block_until_ready(pred)
      arr = np.clip(np.round(np.asarray(pred) * 255.0), 0, 255).astype(np.uint8)
      for bi in range(arr.shape[0]):
        frames_u8.append(arr[bi])
    print(f"[w4] Zero-raytrace inference {clip_name}: {len(frames_u8)} frames in {time.perf_counter() - t0:.2f}s", flush=True)
    save_frames_tar(frames_u8, args.out_dir, clip_name)

  tod_conds = _build_tod_cond_sequence(args.n_frames)
  mat_conds = _build_material_cond_sequence(args.n_frames)
  _render_sweep(ds["classroom_tod_gb"], tod_conds, "relight_tod_sweep")
  _render_sweep(ds["cornell_workshop_mat_gb"], mat_conds, "relight_material_sweep")

  del ds, bf16_params, p_d
  jax.clear_caches()
  gc.collect()

  # Render 4-chip 768x768 4-bounce MC Path-Traced classroom & cornell workshop orbits
  bvh_cls, _, _, fovy_cls, zn_cls, zf_cls, rig_cls = build_classroom_sunlit()
  fn_cls, _, rep_cls = build_complex_pbr_pathtracer(
      bvh_cls, devs, 768, 768, fovy_cls, zn_cls, zf_cls, rig_cls, spp=64, max_bounces=4,
  )
  _render_orbit_sequence(fn_cls, rep_cls, devs, (-1.1, -2.0, 1.55), (0.35, 1.0, 0.95), 22.0, 75, "pt_classroom_orbit", args.out_dir, z_bob=0.06)
  del fn_cls, bvh_cls
  jax.clear_caches()
  gc.collect()

  bvh_crn, _, _, fovy_crn, zn_crn, zf_crn, rig_crn = build_cornell_workshop()
  fn_crn, _, rep_crn = build_complex_pbr_pathtracer(
      bvh_crn, devs, 768, 768, fovy_crn, zn_crn, zf_crn, rig_crn, spp=64, max_bounces=4,
  )
  _render_orbit_sequence(fn_crn, rep_crn, devs, (0.0, -2.55, 1.08), (0.0, 0.15, 0.86), 24.0, 75, "pt_cornell_orbit", args.out_dir, z_bob=0.05)


if __name__ == "__main__":
  main()
