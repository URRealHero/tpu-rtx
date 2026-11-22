"""Generate V3 101.63M-Triangle Blender 5.2 AgX Showcase, Native 4K UHD Hero, and 4K 100%-Crop Figures."""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def make_100m_comparison():
  sm = json.load(open("out/metrics/v3_w5_blender_100m_showcase.json"))
  p512 = np.load("out/metrics/v3_w5_100m_preds_512.npz")
  nr = sm["neural_relighting"]
  par = sm["parity"]
  eff_tris = sm["scene_meta"]["effective_tris"]
  uniq_tris = sm["scene_meta"]["unique_tris"]

  s1 = np.clip(p512["s1_rgb"], 0, 1)
  nn = np.clip(p512["pred_rgb"], 0, 1)
  gt = np.clip(p512["gt_rgb"], 0, 1)
  zd = p512["s1_z_depth"]

  view_indices = [0, 2] if len(s1) >= 3 else [0]
  n_rows = len(view_indices)
  fig, axes = plt.subplots(n_rows, 5, figsize=(18.5, 3.8 * n_rows), dpi=180)
  if n_rows == 1:
    axes = axes[None, :]

  for row_i, k in enumerate(view_indices):
    err5 = np.clip(np.abs(nn[k] - gt[k]) * 5.0, 0.0, 1.0)
    err_mag = np.mean(err5, axis=-1)
    zd_vis = np.where(zd[k] > 0, 1.0 - np.clip((zd[k] - 0.8) / 3.8, 0, 1), 0.0)

    panels = [
        (s1[k], f"Stage 1 TPU BLAS+BVH-128\n({eff_tris/1e6:.2f}M tris, {nr['hero_512']['stage1_psnr_db']:.2f} dB)", None),
        (nn[k], f"TPU Stage 2 {nr['arch']}\n({nr['hero_512']['neural_psnr_db']:.2f} dB, SSIM {nr['hero_512']['neural_ssim']:.4f})", None),
        (gt[k], "Blender 5.2 Cycles AgX GT\n(64 spp Area GI + OIDN)", None),
        (err_mag, f"Error Heatmap (x5)\n(MS-SSIM {nr['hero_512']['neural_msssim']:.4f}, LumRel {nr['hero_512']['lum_rel']*100:+.2f}%)", "inferno"),
        (zd_vis, f"Exact Mesh Z-Depth\n({par['pallas_vs_cycles_512_z_mae_m']*1e6:.1f} um MAE vs EXR)", "viridis"),
    ]
    for col_i, (img, title, cmap) in enumerate(panels):
      ax = axes[row_i, col_i]
      ax.imshow(img, cmap=cmap, vmin=0 if cmap else None, vmax=1 if cmap else None)
      if row_i == 0:
        ax.set_title(title, fontsize=9.5, fontweight="bold")
      ax.axis("off")

  plt.suptitle(
      f"V3 Live TPU v5p: {eff_tris:,} ({eff_tris/1e6:.2f}M) Effective Triangles ({uniq_tris:,} Unique BLAS x 100 Instances) @ 512x512 AgX",
      fontsize=12,
      fontweight="bold",
      y=1.01,
  )
  plt.tight_layout()
  os.makedirs("out/figures", exist_ok=True)
  out_path = "out/figures/v3_w5_blender_100m_comparison.png"
  plt.savefig(out_path, bbox_inches="tight")
  plt.close(fig)
  print("Saved", out_path)


def make_100m_4k_figures():
  sm = json.load(open("out/metrics/v3_w5_blender_100m_showcase.json"))
  p4k = np.load("out/metrics/v3_w5_100m_4k_preds.npz")
  uhd = sm.get("uhd_4k_quality", {})
  eff_tris = sm["scene_meta"]["effective_tris"]

  s1 = p4k["s1_u8"]
  pred = p4k["pred_u8"]
  gt = p4k["gt_u8"] if "gt_u8" in p4k else None

  # Save full-res 3840x2160 native 4K TPU Stage 2 hero PNG
  hero_path = "out/figures/v3_w5_blender_100m_4k_hero.png"
  Image.fromarray(pred).save(hero_path)
  print("Saved", hero_path, pred.shape)

  # 100% Native Pixel Crops (512x512 crops from 3840x2160 frame!)
  # Crop 1: Central Bronze Suzanne Subsurf-5 (center face), Crop 2: Glazed Torus Knot, Crop 3: Background Fluted Column + Ring
  H, W, _ = pred.shape
  crop_boxes = [
      ("Center Suzanne (Subsurf-5, 1.01M tris)", (W // 2 - 256, int(H * 0.48) - 256)),
      ("Glazed Teal Torus Knot (1.05M tris)", (int(W * 0.68) - 256, int(H * 0.50) - 256)),
      ("Rotunda Colonnade & Ring (22+21 Inst.)", (int(W * 0.28) - 256, int(H * 0.16) - 256)),
  ]
  cs = 512
  cols = 3 if gt is not None else 2
  fig, axes = plt.subplots(len(crop_boxes), cols, figsize=(4.5 * cols, 4.3 * len(crop_boxes)), dpi=160)
  for r_i, (label, (x0, y0)) in enumerate(crop_boxes):
    x0 = max(0, min(W - cs, x0))
    y0 = max(0, min(H - cs, y0))
    slices = [
        (s1[y0 : y0 + cs, x0 : x0 + cs], f"Stage 1 TPU Exact\n({uhd.get('stage1_psnr_db', 0):.2f} dB)"),
        (pred[y0 : y0 + cs, x0 : x0 + cs], f"TPU Stage 2 Neural\n({uhd.get('neural_psnr_db', 0):.2f} dB, SSIM {uhd.get('neural_ssim', 0):.4f})"),
    ]
    if gt is not None:
      slices.append((gt[y0 : y0 + cs, x0 : x0 + cs], "Blender 5.2 Cycles 4K AgX GT\n(32 spp + OIDN)"))
    for c_i, (crop_img, title) in enumerate(slices):
      ax = axes[r_i, c_i]
      ax.imshow(crop_img)
      if r_i == 0:
        ax.set_title(title, fontsize=10, fontweight="bold")
      if c_i == 0:
        ax.set_ylabel(label, fontsize=9.5, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
      else:
        ax.axis("off")

  plt.suptitle(
      f"Native 4K UHD (3840x2160) 100%-Pixel Crops — {eff_tris:,} Triangles on Live TPU v5p",
      fontsize=12,
      fontweight="bold",
      y=1.01,
  )
  plt.tight_layout()
  crops_path = "out/figures/v3_w5_blender_100m_4k_crops.png"
  plt.savefig(crops_path, bbox_inches="tight")
  plt.close(fig)
  print("Saved", crops_path)


if __name__ == "__main__":
  make_100m_comparison()
  make_100m_4k_figures()
