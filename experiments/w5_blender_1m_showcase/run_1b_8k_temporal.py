"""Task C: 1B+ Instanced Cinema Gallery @ Native 8K (7680x4320) + 60-Frame Temporal Consistency on 4 TPU v5p Chips."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import time
from typing import Any, Dict, List

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np
from PIL import Image, ImageDraw

from gate import compute_psnr, compute_ssim_batch
from run_wave3_suite import (
    _make_spec,
    _strip,
    load_base_scene,
    make_stratified_3d_poses,
    subdivide_mesh_barycentric,
)
from tpu_mjx_render.roofline import inspect_hlo_ops
from wave3_500m_10b_8k import (
    build_compact_bvh_mesh_item,
    make_spatial_8k_sharded_instanced_renderer,
)


def make_smooth_orbit_path(
    n_frames: int = 60,
    radius_x: float = 2.6,
    radius_y: float = 2.5,
    z_base: float = 1.75,
    z_amp: float = 0.28,
) -> tuple[np.ndarray, np.ndarray]:
  """Generate a smooth 60-frame cinema camera orbit around the 1B-triangle gallery."""
  target = np.array([0.0, 0.45, 0.25], dtype=np.float32)
  up_w = np.array([0.0, 0.0, 1.0], dtype=np.float32)
  cpos_l, cmat_l = [], []
  for t in range(n_frames):
    ang = -0.82 + 0.52 * (t / max(n_frames - 1, 1))
    px = radius_x * math.sin(ang)
    py = -radius_y * math.cos(ang) + 0.45
    pz = z_base + z_amp * math.sin(2.0 * math.pi * (t / max(n_frames - 1, 1)))
    cp = np.array([px, py, pz], dtype=np.float32)
    fwd = target - cp
    fwd /= max(float(np.linalg.norm(fwd)), 1e-12)
    right = np.cross(fwd, up_w)
    right /= max(float(np.linalg.norm(right)), 1e-12)
    up = np.cross(right, fwd)
    cmat = np.stack([right, up, -fwd], axis=1).astype(np.float32)
    cpos_l.append(cp)
    cmat_l.append(cmat)
  return np.stack(cpos_l, axis=0), np.stack(cmat_l, axis=0)


def make_hero_figure(
    hero_8k: np.ndarray,
    frames_1080p: List[np.ndarray],
    results: Dict[str, Any],
    out_png: str,
) -> None:
  H8, W8, _ = hero_8k.shape
  # Layout: top banner (54px), main 8K hero (1440x810), right column (520px wide): 2x 1:1 8K native crops + temporal filmstrip
  main_w, main_h = 1440, 810
  crop_w, crop_h = 520, 398
  pad, hdr, banner_h = 14, 30, 56
  film_h = 150
  cw = main_w + crop_w + 3 * pad
  ch = banner_h + pad + main_h + hdr + pad + film_h + hdr + pad
  canvas = Image.new("RGB", (cw, ch), (13, 15, 20))
  draw = ImageDraw.Draw(canvas)

  b8k = results["bench_8k"]
  tmp = results["temporal_60f"]
  banner = (
      f"1.022B Instanced Cinema Gallery ({results['effective_tris']:,} tris = {results['blas_unique_tris']:,} BLAS x {results['k_instances']} TLAS) | "
      f"Native 8K (7680x4320) 4-Chip Spatial Sharding: {b8k['ms']:.1f} ms ({b8k['fps']:.3f} fps, {b8k['mrays_per_s']:.1f} MRays/s) | "
      f"60f Temporal L1: {tmp['adj_l1_mean']:.5f}, SSIM: {tmp['adj_ssim_mean']:.4f}, 2nd-Order Flicker: {tmp['temporal_accel_l1_mean']:.5f} | 0 callbacks"
  )
  draw.rectangle([0, 0, cw, banner_h], fill=(22, 26, 36))
  draw.text((pad, 20), banner, fill=(242, 247, 255))

  # Main 8K hero panel
  hero_pil = Image.fromarray((np.clip(hero_8k, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8))
  hero_disp = hero_pil.resize((main_w, main_h), Image.Resampling.LANCZOS)
  canvas.paste(hero_disp, (pad, banner_h + pad + hdr))
  draw.text((pad + 4, banner_h + pad + 8), f"Native 8K (7680x4320) 1.022B-Tri Cinema Gallery (4-Chip Spatial Sharding, {b8k['ms']:.1f} ms/frame)", fill=(235, 242, 250))

  # Two 1:1 native-pixel 8K crops on right column
  c1 = hero_8k[int(0.32 * H8) : int(0.32 * H8) + crop_h, int(0.36 * W8) : int(0.36 * W8) + crop_w]
  c2 = hero_8k[int(0.46 * H8) : int(0.46 * H8) + crop_h, int(0.52 * W8) : int(0.52 * W8) + crop_w]
  for ci, (crop_arr, title) in enumerate([(c1, "Native 8K 1:1 Pixel Crop A (Center Sculpture Cluster)"), (c2, "Native 8K 1:1 Pixel Crop B (Specular & Shadow Edges)")]):
    cx = pad + main_w + pad
    cy = banner_h + pad + ci * (crop_h + hdr + 14)
    u8 = (np.clip(crop_arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    canvas.paste(Image.fromarray(u8).resize((crop_w, crop_h), Image.Resampling.NEAREST), (cx, cy + hdr))
    draw.text((cx + 4, cy + 8), title, fill=(235, 242, 250))

  # Bottom strip: 6 keyframes from 60-frame smooth camera orbit
  fy = banner_h + pad + main_h + hdr + pad
  draw.text((pad + 4, fy + 6), f"60-Frame Smooth Camera Orbit Filmstrip (1920x1080, {results['bench_1080p']['fps']:.2f} fps): Adj L1={tmp['adj_l1_mean']:.5f}, Adj SSIM={tmp['adj_ssim_mean']:.4f}, Accel Flicker={tmp['temporal_accel_l1_mean']:.5f}", fill=(235, 242, 250))
  n_strip = 6
  fw = (cw - (n_strip + 1) * pad) // n_strip
  idxs = [0, 11, 23, 35, 47, 59]
  for si, fi in enumerate(idxs):
    fx = pad + si * (fw + pad)
    f_u8 = (np.clip(frames_1080p[fi], 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    f_im = Image.fromarray(f_u8).resize((fw, film_h), Image.Resampling.BILINEAR)
    canvas.paste(f_im, (fx, fy + hdr))
    draw.text((fx + 6, fy + hdr + 6), f"f{fi:02d}", fill=(255, 240, 180))

  os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
  canvas.save(out_png)
  print("SAVED 1B 8K HERO PNG:", out_png, canvas.size, flush=True)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--k_inst", type=int, default=96)
  ap.add_argument("--subdiv_k", type=int, default=48)
  ap.add_argument("--n_frames", type=int, default=60)
  ap.add_argument("--out_json", default="/dev/shm/zaychik-tpu-mjx-w5/v3_w5_wave3_1b_8k_temporal.json")
  ap.add_argument("--out_png", default="/dev/shm/zaychik-tpu-mjx-w5/v3_w5_1b_8k_hero.png")
  args = ap.parse_args()

  ds, base_meshes = load_base_scene()
  devs = jax.devices()
  assert len(devs) == 4, f"Need 4 chips, got {len(devs)}"
  mesh = Mesh(np.array(devs), ("spatial",))
  rep = NamedSharding(mesh, P())
  kw = dict(cam_fovy=float(ds["cam_fovy"]), znear=float(ds["znear"]), zfar=float(ds["zfar"]), light_pos=tuple(float(x) for x in ds["light_pos"]))

  t_b0 = time.perf_counter()
  items_c, dev_c = [], []
  hbm_b = 0
  for mid, m in enumerate(base_meshes):
    nv, nf, nn = subdivide_mesh_barycentric(m["verts"], m["faces"], args.subdiv_k)
    it = build_compact_bvh_mesh_item(mid, {"name": m["name"], "verts": nv, "faces": nf, "vert_normals": nn, "rgba": m["rgba"]}, "spatial_l1_morton")
    del nv, nf, nn
    hbm_b += it.l2_roots_aabb.nbytes + it.l2_aabbs.nbytes + it.l0_aabbs.nbytes + it.leaf_bank.nbytes
    dev_c.append((jax.device_put(it.l2_roots_aabb, rep), jax.device_put(it.l2_aabbs, rep), jax.device_put(it.l0_aabbs, rep), jax.device_put(it.leaf_bank, rep)))
    items_c.append(_strip(it))
    gc.collect()
  dev_c = tuple(dev_c)
  bvh_build_s = time.perf_counter() - t_b0
  blas_tris = sum(it.n_tris_orig for it in items_c)
  eff_tris = blas_tris * args.k_inst
  assert eff_tris >= 1_000_000_000, f"Effective tris {eff_tris} < 1B!"
  print(f"BUILT COMPACT BLAS: unique={blas_tris:,} x K={args.k_inst} -> effective={eff_tris:,} tris, HBM={hbm_b/(1024**2):.1f} MB in {bvh_build_s:.2f}s", flush=True)

  ixp, ixm, cxp0, cxm0 = make_stratified_3d_poses(ds, 1, args.k_inst)
  ixp_r = tuple(jax.device_put(np.asarray(x), rep) for x in ixp)
  ixm_r = tuple(jax.device_put(np.asarray(x), rep) for x in ixm)

  # Smooth 60-frame orbit trajectory around the 1B cinema gallery
  cpos_path, cmat_path = make_smooth_orbit_path(args.n_frames)
  cxp_hero = jax.device_put(cpos_path[:1], rep)
  cxm_hero = jax.device_put(cmat_path[:1], rep)

  def _bench_res(H: int, W: int, n_steps: int = 3):
    spec_full = _make_spec(items_c, kw, H, W)
    fn_sp, _ = make_spatial_8k_sharded_instanced_renderer(
        spec_full, devs, dev_c, tlas_group_size=16, tlas_top_groups=4, max_active_per_tile=16, interpret=False,
    )
    htxt = fn_sp.lower(ixp_r, ixm_r, cxp_hero, cxm_hero).as_text()
    hi = inspect_hlo_ops(htxt)
    hi["send_recv_ops"] = len(re.findall(r"\b(?:send|recv|send-done|recv-done)\(", htxt))
    assert hi["host_callbacks"] == 0 and hi["send_recv_ops"] == 0 and hi["gather_ops"] == 0, hi

    tc0 = time.perf_counter()
    rgb = fn_sp(ixp_r, ixm_r, cxp_hero, cxm_hero)
    jax.block_until_ready(rgb)
    compile_s = time.perf_counter() - tc0

    t0 = time.perf_counter()
    for _ in range(n_steps):
      rgb = fn_sp(ixp_r, ixm_r, cxp_hero, cxm_hero)
    jax.block_until_ready(rgb)
    ms = (time.perf_counter() - t0) * 1e3 / n_steps
    rec = {
        "res": f"{W}x{H}",
        "n_chips": 4,
        "sharding": "spatial_4band",
        "compile_s": round(compile_s, 2),
        "ms": round(ms, 2),
        "fps": round(1e3 / ms, 3),
        "mrays_per_s": round((H * W * 2.0) / (ms * 1e-3) / 1e6, 2),
        "hlo": hi,
    }
    return rec, fn_sp, np.asarray(rgb[0], np.float32)

  # 1. Native 4K (3840x2160) + Native 8K (7680x4320)
  bench_4k, _, _ = _bench_res(2160, 3840, n_steps=3)
  print("4K BENCH:", bench_4k, flush=True)
  bench_8k, fn_8k, hero_8k_np = _bench_res(4320, 7680, n_steps=3)
  print("8K BENCH:", bench_8k, flush=True)

  # Check 8K deterministic zero-flicker on identical pose + 2-frame delta
  rgb_8k_repeat = np.asarray(fn_8k(ixp_r, ixm_r, cxp_hero, cxm_hero)[0], np.float32)
  static_8k_l1 = float(np.mean(np.abs(hero_8k_np - rgb_8k_repeat)))
  del rgb_8k_repeat
  gc.collect()

  # 2. 60-frame smooth camera-path temporal consistency @ 1920x1080 on 4 chips
  bench_1080p, fn_1080p, _ = _bench_res(1080, 1920, n_steps=5)
  print("1080P BENCH:", bench_1080p, flush=True)

  frames = []
  t_seq0 = time.perf_counter()
  for fi in range(args.n_frames):
    cp_d = jax.device_put(cpos_path[fi : fi + 1], rep)
    cm_d = jax.device_put(cmat_path[fi : fi + 1], rep)
    fr = fn_1080p(ixp_r, ixm_r, cp_d, cm_d)
    jax.block_until_ready(fr)
    frames.append(np.asarray(fr[0], np.float32))
  seq_s = time.perf_counter() - t_seq0

  adj_l1, adj_psnr, adj_ssim = [], [], []
  for fi in range(len(frames) - 1):
    a, b = frames[fi], frames[fi + 1]
    adj_l1.append(float(np.mean(np.abs(b - a))))
    adj_psnr.append(compute_psnr(b[None], a[None]))
    adj_ssim.append(compute_ssim_batch(b[None], a[None]))

  accel_l1 = []
  for fi in range(1, len(frames) - 1):
    # 2nd-order finite difference measures temporal jitter/flicker beyond linear motion
    d2 = frames[fi + 1] - 2.0 * frames[fi] + frames[fi - 1]
    accel_l1.append(float(np.mean(np.abs(d2))))

  temporal = {
      "n_frames": args.n_frames,
      "res": "1920x1080",
      "total_render_s": round(seq_s, 2),
      "effective_fps": round(args.n_frames / max(seq_s, 1e-6), 3),
      "static_8k_determinism_l1": static_8k_l1,
      "adj_l1_mean": round(float(np.mean(adj_l1)), 6),
      "adj_l1_max": round(float(np.max(adj_l1)), 6),
      "adj_psnr_mean_db": round(float(np.mean(adj_psnr)), 3),
      "adj_ssim_mean": round(float(np.mean(adj_ssim)), 5),
      "adj_ssim_min": round(float(np.min(adj_ssim)), 5),
      "temporal_accel_l1_mean": round(float(np.mean(accel_l1)), 6),
      "temporal_accel_l1_max": round(float(np.max(accel_l1)), 6),
  }
  print("TEMPORAL 60F:", temporal, flush=True)

  results = {
      "experiment": "v3_w5_wave3_1b_8k_temporal",
      "device_kind": jax.devices()[0].device_kind,
      "n_devices": len(devs),
      "blas_unique_tris": int(blas_tris),
      "k_instances": int(args.k_inst),
      "effective_tris": int(eff_tris),
      "bvh_hbm_mb": round(hbm_b / (1024 ** 2), 2),
      "bvh_build_s": round(bvh_build_s, 2),
      "bench_4k": bench_4k,
      "bench_8k": bench_8k,
      "bench_1080p": bench_1080p,
      "temporal_60f": temporal,
  }
  os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
  with open(args.out_json, "w") as f:
    json.dump(results, f, indent=2)
  print("SAVED 1B 8K TEMPORAL JSON:", args.out_json, flush=True)

  make_hero_figure(hero_8k_np, frames, results, args.out_png)


if __name__ == "__main__":
  main()
