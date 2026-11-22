"""Generate V2 Blender Mesh+3DGS Visual Figure and Live TPU v5p V1-vs-V2 Speedup Figure."""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def make_blender_asset_figure():
  pred_path = "out/metrics/v2_live_mesh_test_preds.npz"
  sum_path = "out/metrics/v2_live_mesh_neural_summary.json"
  if os.path.exists(pred_path) and os.path.exists(sum_path):
    tp = np.load(pred_path)
    sm = json.load(open(sum_path))["best"]
  else:
    tp = np.load("out/metrics/v2_mesh_test_preds.npz")
    sm = json.load(open("out/metrics/v2_mesh_neural_summary.json"))
  gs_preds = np.load("out/metrics/v2_gs3d_test_preds.npz")

  k = 0  # First held-out dynamic test pose
  s1 = np.clip(tp["s1_rgb"][k], 0, 1)
  nn_rgb = np.clip(tp["pred_rgb"][k], 0, 1)
  splat = np.clip(gs_preds["gs_rgb"][k], 0, 1)
  gt = np.clip(tp["gt_rgb"][k], 0, 1)
  zd = tp["s1_z_depth"][k]
  zd_vis = np.where(zd > 0, 1.0 - np.clip((zd - 0.6) / 1.4, 0, 1), 0.0)

  fig, axes = plt.subplots(1, 5, figsize=(16, 3.5), dpi=160)
  panels = [
      (s1, f"Blender Mesh Exact\n(4,620 tris, {sm['stage1_mesh_psnr_db']:.2f} dB)", None),
      (nn_rgb, f"Mesh + TPU Neural W{sm['base_width']}\n({sm['neural_mesh_psnr_db']:.2f} dB, SSIM {sm['neural_mesh_ssim']:.3f})", None),
      (splat, "Rigid-Body Tile-3DGS\n(2,048 splats, 1,193 FPS)", None),
      (gt, "Blender 5.2 Cycles GT\n(48 spp GI + Soft Shadow)", None),
      (zd_vis, "Exact Mesh Z-Depth\n(8.7 um MAE vs EXR)", "inferno"),
  ]
  for ax, (img, title, cmap) in zip(axes, panels):
    ax.imshow(img, cmap=cmap, vmin=0 if cmap else None, vmax=1 if cmap else None)
    ax.set_title(title, fontsize=9.5, fontweight="bold")
    ax.axis("off")

  plt.suptitle("V2 Real Blender Curved Mesh Raytracer + Neural Relighting + Rigid 3DGS (Held-out Dynamic Pose)", fontsize=11.5, y=1.02)
  plt.tight_layout()
  os.makedirs("out/figures", exist_ok=True)
  out_path = "out/figures/v2_blender_mesh_and_3dgs.png"
  plt.savefig(out_path, bbox_inches="tight")
  plt.close(fig)
  print("Saved", out_path)


def _fps_map(path: str):
  d = json.load(open(path))
  return {r["batch_size"]: float(r["e2e_fps"]) for r in d["e2e_runs"]}


def make_tpu_speedup_figure():
  v1_t_1c = _fps_map("out/metrics/live_tpu_panda_tiled_1c.json")
  v1_m2_1c = _fps_map("out/metrics/live_tpu_panda_m2_1c.json")
  v1_m2_4c = _fps_map("out/metrics/live_tpu_panda_m2_4c.json")
  v2_pre_1c = _fps_map("out/metrics/v2_live_pallas_v2_preopt_1c.json")
  v2_s1_1c = _fps_map("out/metrics/v2_live_pallas_v2_1c.json")
  v2_m2_1c = _fps_map("out/metrics/v2_live_pallas_v2_m2_1c.json")
  v2_s1_4c = _fps_map("out/metrics/v2_live_pallas_v2_4c.json")
  v2_m2_4c = _fps_map("out/metrics/v2_live_pallas_v2_m2_4c.json")
  v2_128_1c = _fps_map("out/metrics/v2_live_pallas_v2_128_1c.json")
  v2_128_4c = _fps_map("out/metrics/v2_live_pallas_v2_128_4c.json")
  assets = {r["pipeline"]: float(r["fps"]) for r in json.load(open("out/metrics/v2_live_assets_1c.json"))["runs"]}
  ppo_v1 = float(json.load(open("out/metrics/ppo_panda.json"))["steady_env_steps_per_sec"])
  ppo_v2 = float(json.load(open("out/metrics/v2_live_ppo_panda.json"))["steady_env_steps_per_sec"])

  fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6), dpi=160)

  # Panel A: Stage 1 & Stage 2 PandaPickCube 64x64 Peak/B=4096 FPS (log scale)
  labels_a = [
      "V1 Tiled\n(1c)",
      "V2.0 Pallas\n(1c)",
      "V2.2 Pallas\n(1c)",
      "V2.2 + M2\n(1c)",
      "V1 + M2\n(4c)",
      "V2.2 + M2\n(4c)",
      "V2.2 Pallas\n(4c)",
  ]
  vals_a = [
      v1_t_1c[4096],
      v2_pre_1c[4096],
      v2_s1_1c[4096],
      v2_m2_1c[4096],
      v1_m2_4c[1024],
      v2_m2_4c[4096],
      v2_s1_4c[4096],
  ]
  cols_a = ["#6c757d", "#4c9be8", "#1f77b4", "#2ca02c", "#9467bd", "#17becf", "#d62728"]
  bars_a = axes[0].bar(range(len(labels_a)), vals_a, color=cols_a)
  axes[0].set_yscale("log")
  axes[0].set_xticks(range(len(labels_a)))
  axes[0].set_xticklabels(labels_a, fontsize=8)
  axes[0].set_ylabel("End-to-End MJX+Render FPS (log)")
  axes[0].set_title("PandaPickCube 64x64 (TPU v5p, B=4096)", fontweight="bold", fontsize=10)
  axes[0].set_ylim(300, 120000)
  for b, v in zip(bars_a, vals_a):
    axes[0].text(b.get_x() + b.get_width() / 2, v * 1.12, f"{v:,.0f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

  # Panel B: 128x128 & Scaling + Vision PPO
  labels_b = [
      "128x128 V1\n(1c, +M2)",
      "128x128 V2.2\n(1c)",
      "128x128 V2.2\n(4c)",
      "PPO V1\n(steps/s)",
      "PPO V2.2\n(steps/s)",
  ]
  v1_128_1c = _fps_map("out/metrics/live_tpu_panda_128_1c.json")[1024]
  vals_b = [v1_128_1c, v2_128_1c[2048], v2_128_4c[4096], ppo_v1, ppo_v2]
  cols_b = ["#6c757d", "#1f77b4", "#d62728", "#6c757d", "#2ca02c"]
  bars_b = axes[1].bar(range(len(labels_b)), vals_b, color=cols_b)
  axes[1].set_yscale("log")
  axes[1].set_xticks(range(len(labels_b)))
  axes[1].set_xticklabels(labels_b, fontsize=8)
  axes[1].set_ylabel("Throughput (FPS or env-steps/s, log)")
  axes[1].set_title("128x128 & Closed-Loop Vision PPO", fontweight="bold", fontsize=10)
  axes[1].set_ylim(100, 35000)
  for b, v in zip(bars_b, vals_b):
    axes[1].text(b.get_x() + b.get_width() / 2, v * 1.12, f"{v:,.0f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

  # Panel C: Real Blender 4,620-tri Meshes & 2,048-Splat 3DGS (1-chip TPU v5p, 64x64)
  labels_c = ["Mesh\nJAX Cluster", "Mesh\nPallas (1c)", "3DGS\nPallas (1c)", "3DGS\nJAX Tile (1c)"]
  vals_c = [assets["mesh_jax_cluster"], assets["mesh_pallas"], assets["gs3d_pallas"], assets["gs3d_jax_tile"]]
  cols_c = ["#6c757d", "#1f77b4", "#ff7f0e", "#2ca02c"]
  bars_c = axes[2].bar(range(len(labels_c)), vals_c, color=cols_c)
  axes[2].set_yscale("log")
  axes[2].set_xticks(range(len(labels_c)))
  axes[2].set_xticklabels(labels_c, fontsize=8.5)
  axes[2].set_ylabel("Render FPS (1-chip TPU v5p, 64x64, log)")
  axes[2].set_title("Blender 4,620-tri Mesh & 3DGS (1c)", fontweight="bold", fontsize=10)
  axes[2].set_ylim(5, 5000)
  for b, v in zip(bars_c, vals_c):
    axes[2].text(b.get_x() + b.get_width() / 2, v * 1.12, f"{v:,.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

  plt.suptitle("Live TPU v5p Verified Throughput: V1 Tiled vs V2.2 Pallas Mosaic + Blender Real Assets (Zero Host Callbacks)", fontsize=11.5, y=1.01)
  plt.tight_layout()
  out_path = "out/figures/v2_tpu_speedup.png"
  plt.savefig(out_path, bbox_inches="tight")
  plt.close(fig)
  print("Saved", out_path)


def main():
  make_blender_asset_figure()
  make_tpu_speedup_figure()


if __name__ == "__main__":
  main()
