"""Generate crisp V3 1.24M-Triangle Blender Showcase and 360-degree Orbit comparison figures."""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def make_showcase_512():
  sm = json.load(open("out/metrics/v3_w5_blender_1m_showcase.json"))
  p512 = np.load("out/metrics/v3_w5_showcase_preds_512.npz")
  best = sm["neural_relighting"]["best"]
  par = sm["parity"]
  tot_tris = sm["scene_meta"]["total_tris"]

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
    zd_vis = np.where(zd[k] > 0, 1.0 - np.clip((zd[k] - 0.8) / 2.2, 0, 1), 0.0)

    panels = [
        (s1[k], f"Stage 1 TPU Mosaic BVH-128\n({tot_tris:,} tris + UV, {best['hero_512']['stage1_psnr_db']:.2f} dB)", None),
        (nn[k], f"TPU Stage 2 {best['arch']}\n({best['hero_512']['neural_psnr_db']:.2f} dB, SSIM {best['hero_512']['neural_ssim']:.4f})", None),
        (gt[k], "Blender 5.2 Cycles GT\n(64 spp Area GI + Soft Shadow)", None),
        (err_mag, "Error Heatmap (x5)\n(MS-SSIM {:.4f}, LumRel {:+.2f}%)".format(best["hero_512"]["neural_msssim"], best["hero_512"]["lum_rel"] * 100.0), "inferno"),
        (zd_vis, f"Exact Mesh Z-Depth\n({par['pallas_vs_cycles_512_z_mae_m']*1e6:.1f} um MAE vs EXR)", "viridis"),
    ]
    for col_i, (img, title, cmap) in enumerate(panels):
      ax = axes[row_i, col_i]
      ax.imshow(img, cmap=cmap, vmin=0 if cmap else None, vmax=1 if cmap else None)
      if row_i == 0:
        ax.set_title(title, fontsize=9.5, fontweight="bold")
      ax.axis("off")

  plt.suptitle(
      f"V3 Live TPU v5p Showcase: {tot_tris:,}-Triangle Textured Blender 5.2 Scene @ 512x512 (Held-Out Views & Dynamic Poses, host_callbacks=0)",
      fontsize=12,
      fontweight="bold",
      y=1.01,
  )
  plt.tight_layout()
  os.makedirs("out/figures", exist_ok=True)
  out_path = "out/figures/v3_w5_1m_showcase.png"
  plt.savefig(out_path, bbox_inches="tight")
  plt.close(fig)
  print("Saved", out_path)


def make_orbit_256():
  sm = json.load(open("out/metrics/v3_w5_blender_1m_showcase.json"))
  p256 = np.load("out/metrics/v3_w5_orbit_preds_256.npz")
  best = sm["neural_relighting"]["best"]
  te_idx = p256["test_idx"]
  sel = te_idx[:6] if len(te_idx) >= 6 else te_idx

  s1 = np.clip(p256["s1_rgb"][sel], 0, 1)
  nn = np.clip(p256["pred_rgb"][sel], 0, 1)
  gt = np.clip(p256["gt_rgb"][sel], 0, 1)

  n_cols = len(sel)
  fig, axes = plt.subplots(3, n_cols, figsize=(2.8 * n_cols, 8.6), dpi=160)
  row_labels = [
      f"Stage 1 TPU Exact ({best['test_256']['stage1_psnr_db']:.2f} dB)",
      f"Stage 2 {best['arch']} ({best['test_256']['neural_psnr_db']:.2f} dB)",
      "Blender 5.2 Cycles GT (48 spp)",
  ]
  for c, fi in enumerate(sel):
    for r, arr in enumerate([s1, nn, gt]):
      ax = axes[r, c]
      ax.imshow(arr[c])
      if r == 0:
        deg = int(round(360.0 * fi / 48.0))
        ax.set_title(f"Held-out #{fi} ({deg} deg)", fontsize=9, fontweight="bold")
      if c == 0:
        ax.set_ylabel(row_labels[r], fontsize=9, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
      else:
        ax.axis("off")

  plt.suptitle(
      f"Dynamic 360-Degree Orbit + Moving Sculptures (256x256 Held-Out Test Views): {best['test_256']['neural_psnr_db']:.2f} dB PSNR, SSIM {best['test_256']['neural_ssim']:.4f}",
      fontsize=11.5,
      fontweight="bold",
      y=1.01,
  )
  plt.tight_layout()
  out_path = "out/figures/v3_w5_1m_orbit.png"
  plt.savefig(out_path, bbox_inches="tight")
  plt.close(fig)
  print("Saved", out_path)


if __name__ == "__main__":
  make_showcase_512()
  make_orbit_256()
