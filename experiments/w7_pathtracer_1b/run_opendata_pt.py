"""Run Official Blender Open Data 5.3.0 Benchmark (classroom, monster, junkshop) on w7 TPU."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Any, Dict, List, Tuple

import jax
import numpy as np
from PIL import Image, ImageDraw

from bvh_pbr_pt import (
    LightRig,
    audit_hlo_zero_copy,
    build_complex_pbr_pathtracer,
    build_pbr_wide_bvh,
)


def _align_res(w: int, h: int, n_chips: int) -> Tuple[int, int]:
  w_mult = 16 * n_chips
  w_al = ((w + w_mult - 1) // w_mult) * w_mult
  h_al = ((h + 7) // 8) * 8
  return w_al, h_al


def _build_rig_from_lights(lights_list: List[Dict[str, Any]], scene_name: str, bmin: np.ndarray, bmax: np.ndarray) -> LightRig:
  center = 0.5 * (bmin + bmax)
  span = float(np.max(bmax - bmin))
  sun_l = next((l for l in lights_list if l["type"] == "SUN"), None)
  area_ls = [l for l in lights_list if l["type"] in ("AREA", "POINT", "SPOT") and float(l["energy"]) > 0.1]
  # Filter out ultra-distant backdrop lights (> 25m from center) if local scene lights exist
  local_ls = [l for l in area_ls if float(np.linalg.norm(np.asarray(l["pos"], np.float32) - center)) < 25.0]
  cand_ls = local_ls if local_ls else area_ls
  if cand_ls:
    def _irrad_score(l):
      d = float(np.linalg.norm(np.asarray(l["pos"], np.float32) - center))
      return float(l["energy"]) / (d * d + 1.0)
    best_a = max(cand_ls, key=_irrad_score)
    a_pos = tuple(float(x) for x in best_a["pos"])
    a_dir = np.asarray(best_a["dir"], dtype=np.float32)
    if np.linalg.norm(a_dir) < 1e-4 or best_a["type"] == "POINT":
      a_dir = center - np.asarray(a_pos, dtype=np.float32)
    a_dir = a_dir / max(float(np.linalg.norm(a_dir)), 1e-12)
    a_r = max(0.55, min(float(best_a.get("size", 0.8)) * 0.8, span * 0.25))
    col = np.asarray(best_a["color"], dtype=np.float32)
    dist_c = max(float(np.linalg.norm(np.asarray(a_pos, np.float32) - center)), 1.0)
    # Scale radiance so incident irradiance matches scene scale cleanly without clipping
    rad_scale = max(10.0, min(22.0, 5.0 * dist_c / max(a_r, 0.5)))
    a_emit = tuple(float(x) for x in (col * rad_scale))
  else:
    a_pos = (float(center[0]), float(center[1]), float(bmax[2] - 0.05 * span))
    a_dir = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    a_r = max(0.6, span * 0.15)
    a_emit = (12.0, 11.2, 10.2)

  has_sun = sun_l is not None
  if sun_l is not None:
    s_toward = -np.asarray(sun_l["dir"], dtype=np.float32)
    s_toward = s_toward / max(float(np.linalg.norm(s_toward)), 1e-12)
    scol = np.asarray(sun_l["color"], dtype=np.float32)
    s_emit = tuple(float(x) for x in (scol * 10.0))
  else:
    s_toward = (0.4, -0.4, 0.8)
    s_emit = (12.0, 10.5, 8.5)

  exp = 0.55 if "classroom" in scene_name else 0.90
  return LightRig(
      area_center=a_pos,
      area_normal=tuple(float(x) for x in a_dir),
      area_radius=a_r,
      area_emit=a_emit,
      has_sun=has_sun,
      sun_dir=tuple(float(x) for x in s_toward),
      sun_cos_half_angle=0.9985,
      sun_emit=s_emit,
      sky_zenith=(0.35, 0.52, 0.80),
      sky_horizon=(0.28, 0.31, 0.36),
      sky_ground=(0.12, 0.11, 0.10),
      exposure=exp,
  )


def stitch_strip(out_arr: jax.Array) -> np.ndarray:
  arr = np.asarray(jax.device_get(out_arr), dtype=np.float32)
  n_c, H, W_loc, C = arr.shape
  return arr.transpose(1, 0, 2, 3).reshape(H, n_c * W_loc, C)


def compute_psnr(img_a: np.ndarray, img_b: np.ndarray) -> float:
  mse = float(np.mean((img_a.astype(np.float32) - img_b.astype(np.float32)) ** 2))
  return float(10.0 * math.log10(1.0 / max(mse, 1e-10)))


def compute_ssim(img_a: np.ndarray, img_b: np.ndarray, win: int = 7) -> float:
  """Pure-NumPy structural similarity (SSIM) over 7x7 sliding window, averaged across RGB."""
  a = img_a.astype(np.float32)
  b = img_b.astype(np.float32)
  c1 = 0.01 ** 2
  c2 = 0.03 ** 2

  def _box_mean(x: np.ndarray) -> np.ndarray:
    pad = win // 2
    xp = np.pad(x, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    cs = np.cumsum(xp, axis=0, dtype=np.float64)
    cs = np.pad(cs, ((1, 0), (0, 0), (0, 0)), mode="constant")
    vy = cs[win:, :, :] - cs[:-win, :, :]
    csx = np.cumsum(vy, axis=1, dtype=np.float64)
    csx = np.pad(csx, ((0, 0), (1, 0), (0, 0)), mode="constant")
    return ((csx[:, win:, :] - csx[:, :-win, :]) / float(win * win)).astype(np.float32)

  mu_a = _box_mean(a)
  mu_b = _box_mean(b)
  mu_a2 = mu_a * mu_a
  mu_b2 = mu_b * mu_b
  mu_ab = mu_a * mu_b
  sig_a2 = np.maximum(_box_mean(a * a) - mu_a2, 0.0)
  sig_b2 = np.maximum(_box_mean(b * b) - mu_b2, 0.0)
  sig_ab = _box_mean(a * b) - mu_ab
  num = (2.0 * mu_ab + c1) * (2.0 * sig_ab + c2)
  den = (mu_a2 + mu_b2 + c1) * (sig_a2 + sig_b2 + c2)
  return float(np.mean(num / np.maximum(den, 1e-12)))


def bench_opendata_scene(
    scene_name: str,
    npz_path: str,
    gt_png_path: str,
    devices_all,
    spps: List[int] = [16, 64, 256],
    max_bounces: int = 4,
):
  ds = np.load(npz_path, allow_pickle=False)
  verts = ds["verts"]
  faces = ds["faces"]
  vnorm = ds["vnorm"]
  tri_alb = ds["tri_albedo"]
  tri_rgh = ds["tri_roughness"]
  tri_mc = ds["tri_mat_code"]
  cam_pos = ds["cam_pos"][None, :]
  cam_xmat = ds["cam_xmat"][None, :, :]
  cam_fovy = float(ds["cam_fovy"])
  native_res = ds["native_res"]
  lights_list = json.loads(bytes(ds["lights_json"]).decode("utf-8"))

  bmin, bmax = verts.min(0), verts.max(0)
  rig = _build_rig_from_lights(lights_list, scene_name, bmin, bmax)
  bvh = build_pbr_wide_bvh(scene_name, verts, faces, tri_alb, tri_rgh, tri_mc, vert_normals=vnorm)

  gt_im = Image.open(gt_png_path).convert("RGB")
  gt_w, gt_h = gt_im.size
  gt_np = np.asarray(gt_im, dtype=np.float32) / 255.0

  # Use exact GT resolution aligned to 64x8 (16*4 x 8)
  W_tpu, H_tpu = _align_res(gt_w, gt_h, 4)

  recs = []
  best_img = None

  for n_chips in [1, 4]:
    devs = devices_all[:n_chips]
    # Run 16 spp on 1-chip for scaling comparison, and full [16, 64, 256] on 4-chip
    run_spps = [16] if n_chips == 1 else spps
    for idx, spp in enumerate(run_spps):
      fn, mesh, rep = build_complex_pbr_pathtracer(
          bvh, devs, H_tpu, W_tpu, cam_fovy, 0.05, 120.0, rig, spp=spp, max_bounces=max_bounces
      )
      cxp = jax.device_put(cam_pos, rep)
      cxm = jax.device_put(cam_xmat, rep)
      rng = jax.device_put(jax.random.PRNGKey(idx + 1), rep)
      lowered = fn.lower(cxp, cxm, rng)
      hlo_info = audit_hlo_zero_copy(lowered) if idx == 0 else None

      tc0 = time.perf_counter()
      compiled = lowered.compile()
      compile_s = time.perf_counter() - tc0

      if spp < 128:
        y = compiled(cxp, cxm, jax.device_put(jax.random.PRNGKey(idx + 50), rep))
        jax.block_until_ready(y)

      t0 = time.perf_counter()
      y = compiled(cxp, cxm, jax.device_put(jax.random.PRNGKey(idx + 100), rep))
      jax.block_until_ready(y)
      dt_s = time.perf_counter() - t0

      img_np = stitch_strip(y)
      if (W_tpu, H_tpu) != (gt_w, gt_h):
        img_cmp = np.asarray(
            Image.fromarray(np.clip(img_np * 255.0 + 0.5, 0, 255).astype(np.uint8)).resize((gt_w, gt_h), Image.Resampling.BILINEAR),
            dtype=np.float32,
        ) / 255.0
      else:
        img_cmp = img_np

      psnr_gt = compute_psnr(img_cmp, gt_np)
      ssim_gt = compute_ssim(img_cmp, gt_np)
      ms = dt_s * 1e3
      fps = 1.0 / dt_s
      spm = fps * spp * 60.0
      rays_per_s = (3 if rig.has_sun else 2) * max_bounces
      mrays = (H_tpu * W_tpu * spp * rays_per_s / dt_s) / 1e6

      rec = {
          "scene": scene_name,
          "tris": bvh.n_tris_orig,
          "native_blend_res": f"{int(native_res[0])}x{int(native_res[1])}@{int(native_res[2])}%",
          "res": f"{H_tpu}x{W_tpu}",
          "n_chips": n_chips,
          "spp": spp,
          "max_bounces": max_bounces,
          "compile_s": round(compile_s, 2),
          "ms": round(ms, 2),
          "images_per_sec": round(fps, 3),
          "samples_per_min": round(spm, 1),
          "mrays_per_s": round(mrays, 2),
          "psnr_vs_cycles_5_3_db": round(psnr_gt, 2),
          "ssim_vs_cycles_5_3": round(ssim_gt, 4),
      }
      if hlo_info is not None:
        rec["hlo_zero_copy"] = True
        rec["hlo_ops"] = hlo_info
      print("OPENDATA PT:", rec, flush=True)
      recs.append(rec)
      if n_chips == 4:
        best_img = img_cmp
  return recs, best_img, gt_np


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--data_dir", default="/dev/shm/zaychik-tpu-mjx-w7/opendata/exported")
  ap.add_argument("--scenes", nargs="+", default=["classroom", "monster", "junkshop"])
  ap.add_argument("--out_json", default="out/metrics/v3_w7_opendata_pathtracer.json")
  ap.add_argument("--out_png", default="out/figures/v3_w7_opendata_pathtracer.png")
  args = ap.parse_args()

  devs_all = jax.devices()
  assert len(devs_all) == 4

  all_recs = []
  pairs = []
  def _save_outputs():
    if pairs:
      # Build side-by-side Cycles 5.3.0 GT vs TPU Path Tracer hero comparison PNG (preserving each scene's aspect!)
      max_w = max(p[2].shape[1] for p in pairs)
      pad, hdr = 14, 40
      row_ys = []
      cur_y = pad
      for _sc, _rb, tpu_i, _gt_i in pairs:
        row_ys.append(cur_y)
        cur_y += tpu_i.shape[0] + hdr + pad
      canvas = Image.new("RGB", (2 * max_w + 3 * pad, cur_y), (15, 17, 21))
      draw = ImageDraw.Draw(canvas)
      for r_i, (sc, r_b, tpu_i, gt_i) in enumerate(pairs):
        y0 = row_ys[r_i]
        pw, ph = tpu_i.shape[1], tpu_i.shape[0]
        x_off = (max_w - pw) // 2
        tpu_u8 = np.clip(tpu_i * 255.0 + 0.5, 0, 255).astype(np.uint8)
        gt_u8 = np.clip(gt_i * 255.0 + 0.5, 0, 255).astype(np.uint8)
        tpu_pil = Image.fromarray(tpu_u8).resize((pw, ph), Image.Resampling.BILINEAR)
        gt_pil = Image.fromarray(gt_u8).resize((pw, ph), Image.Resampling.BILINEAR)
        canvas.paste(tpu_pil, (pad + x_off, y0 + hdr))
        canvas.paste(gt_pil, (2 * pad + max_w + x_off, y0 + hdr))
        draw.text(
            (pad, y0 + 12),
            f"TPU v5p 4c ({sc}, {r_b['tris']:,} tris, {r_b['spp']} spp) | {r_b['samples_per_min']:,.0f} spm | {r_b['ms']:.0f}ms",
            fill=(238, 244, 250),
        )
        draw.text(
            (2 * pad + max_w, y0 + 12),
            f"Blender 5.3.0 Cycles Reference GT ({sc}) | PSNR {r_b['psnr_vs_cycles_5_3_db']:.2f} dB | SSIM {r_b['ssim_vs_cycles_5_3']:.4f}",
            fill=(255, 220, 80),
        )
      os.makedirs(os.path.dirname(args.out_png) or ".", exist_ok=True)
      canvas.save(args.out_png)
      print("Saved OpenData Comparison PNG ->", args.out_png, canvas.size, flush=True)

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
      json.dump(
          {
              "experiment": "v3_w7_opendata_pathtracer",
              "backend": jax.default_backend(),
              "baselines": {
                  "stock_blender_5_3_0_h100_spm": 7041,
                  "exact_h100_spm": 14106,
                  "rtx_5090_optix_spm": 19202,
                  "nrc_scatter_h100_spm": 55148,
              },
              "runs": all_recs,
          },
          f,
          indent=2,
      )
    print("Saved OpenData JSON ->", args.out_json, flush=True)

  for sc in args.scenes:
    npz_p = os.path.join(args.data_dir, f"{sc}.npz")
    gt_p = os.path.join(args.data_dir, f"{sc}_cycles_gt.png")
    if not os.path.exists(npz_p) or not os.path.exists(gt_p):
      print(f"WARN: missing {npz_p} or {gt_p}, skipping", flush=True)
      continue
    recs, tpu_img, gt_img = bench_opendata_scene(sc, npz_p, gt_p, devs_all, spps=[16, 64], max_bounces=3)
    all_recs.extend(recs)
    r_best = recs[-1]
    pairs.append((sc, r_best, tpu_img, gt_img))
    _save_outputs()


if __name__ == "__main__":
  main()
