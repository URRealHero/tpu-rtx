"""Plot Dark-Cinema Scaling Hero Figure for the 32-Chip TPU v5p-64 Pod Capstone.

Reads `out/metrics/v3_pod_v5p64_capstone.json` (or `--in_json`) and generates
`out/figures/v3_pod_v5p64_scaling.png`.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _fmt_num(val: float) -> str:
  if val >= 1e9:
    return f"{val/1e9:.2f}B"
  if val >= 1e6:
    return f"{val/1e6:.1f}M"
  if val >= 1e3:
    return f"{val/1e3:.1f}k"
  return f"{val:.1f}"


def plot_pod_scaling(in_json: str, out_png: str) -> None:
  with open(in_json, "r") as f:
    data = json.load(f)

  n_chips = int(data["n_chips"])
  hbm_tb = float(data["cluster_hbm_tb"])
  b1 = data["bench1_tpu_rtx"]
  b2_batch = data["bench2_tpu_gsplat"]["batch_runs"]
  b2_dist = data["bench2_tpu_gsplat"]["distributed_1b_plus"]
  b3 = data["bench3_batch_lidar"]
  b4 = data["bench4_geom_sharded_unique_tris"]
  b5 = data["bench5_spatial_8k_pathtracer"]

  plt.rcParams.update({
      "figure.facecolor": "#0B0F17",
      "axes.facecolor": "#121824",
      "axes.edgecolor": "#2A3447",
      "axes.labelcolor": "#E2E8F0",
      "text.color": "#F8FAFC",
      "xtick.color": "#94A3B8",
      "ytick.color": "#94A3B8",
      "grid.color": "#1E293B",
      "grid.linestyle": "--",
      "grid.alpha": 0.7,
      "font.family": "DejaVu Sans",
  })

  fig, axes = plt.subplots(2, 2, figsize=(18, 10.5), dpi=180)
  fig.subplots_adjust(left=0.065, right=0.965, top=0.87, bottom=0.08, wspace=0.22, hspace=0.30)

  # Super-title
  mode_lbl = "4-Chip Dry-Run" if data.get("dry_run_4c") else f"{n_chips}-Chip TPU v5p Pod ({hbm_tb:.2f} TB HBM)"
  fig.suptitle(
      f"GPU-Free Physics + Raytracing + 3DGS Capstone — {mode_lbl}",
      fontsize=20, fontweight="bold", color="#F8FAFC", y=0.955,
  )
  fig.text(
      0.5, 0.915,
      "Zero Host Callbacks (host_callbacks = 0)  |  Zero Send/Recv  |  Pure Mosaic Pallas + XLA shard_map over ICI",
      ha="center", fontsize=12, color="#38BDF8",
  )

  # Panel 1: Throughput Scaling (tpu_rtx FPS, 3DGS Batch FPS, 128x2048 LiDAR env-FPS)
  ax = axes[0, 0]
  labels_p1 = [
      f"tpu_rtx 64\nB={b1[0]['global_batch']:,}",
      f"tpu_rtx 64\nB={b1[-1]['global_batch']:,}",
      f"gsplat 64\nB={b2_batch[0]['global_batch']:,}",
      f"gsplat 128\nB={b2_batch[1]['global_batch']:,}",
      f"LiDAR\nB={b3[0]['global_batch']:,}",
      f"LiDAR\nB={b3[-1]['global_batch']:,}",
  ]
  vals_p1 = [
      b1[0]["e2e_fps"],
      b1[-1]["e2e_fps"],
      b2_batch[0]["fps"],
      b2_batch[1]["fps"],
      b3[0]["e2e_fps"],
      b3[-1]["e2e_fps"],
  ]
  colors_p1 = ["#38BDF8", "#0EA5E9", "#34D399", "#10B981", "#F59E0B", "#F97316"]
  bars = ax.bar(range(len(labels_p1)), vals_p1, color=colors_p1, width=0.62, edgecolor="#0B0F17", linewidth=1.2)
  ax.set_yscale("log")
  ax.set_xticks(range(len(labels_p1)))
  ax.set_xticklabels(labels_p1, fontsize=9)
  ax.set_ylabel("Cluster Throughput (FPS, log scale)", fontsize=10.5)
  ax.set_title("1. Cluster-Wide MJX + Render + 3DGS + LiDAR FPS", fontsize=12.5, fontweight="bold", pad=10)
  ax.grid(True, axis="y")
  ax.set_ylim(top=max(vals_p1) * 3.2)
  for bar, v in zip(bars, vals_p1):
    ax.text(bar.get_x() + bar.get_width() / 2.0, v * 1.18, f"{v:,.0f}\nFPS", ha="center", va="bottom", fontsize=8.8, fontweight="bold", color="#F8FAFC")

  # Panel 2: Cluster Memory & Distributed Scale (1B+ Gaussians + 8B-16B Unique Triangles)
  ax = axes[0, 1]
  b4_8b = [r for r in b4 if r["k_obj_per_chip"] == 256]
  b4_16b = [r for r in b4 if r["k_obj_per_chip"] == 512]
  p2_labels = [
      f"Dist. 3DGS\n{b2_dist['total_gaussians']/1e9:.2f}B Splats",
      f"Unique BVH (Low)\n{b4_8b[0]['total_unique_tris']/1e9:.2f}B Tris",
      f"Unique BVH (High)\n{b4_16b[0]['total_unique_tris']/1e9:.2f}B Tris",
  ]
  p2_hbm_gb = [
      b2_dist["hbm_splats_tb_cluster"] * 1000.0,
      b4_8b[0]["bvh_hbm_tb_cluster"] * 1000.0,
      b4_16b[0]["bvh_hbm_tb_cluster"] * 1000.0,
  ]
  p2_cols = ["#A855F7", "#22D3EE", "#EC4899"]
  bars2 = ax.bar(range(len(p2_labels)), p2_hbm_gb, color=p2_cols, width=0.52)
  ax.set_xticks(range(len(p2_labels)))
  ax.set_xticklabels(p2_labels, fontsize=9.5)
  ax.set_ylabel("Active Cluster HBM Resident (GB)", fontsize=10.5)
  ax.set_title(f"2. Distributed HBM Scale Across {n_chips} Chips ({hbm_tb:.2f} TB Cap)", fontsize=12.5, fontweight="bold", pad=10)
  ax.grid(True, axis="y")
  ax.set_ylim(0, max(p2_hbm_gb) * 1.28)
  for bar, v in zip(bars2, p2_hbm_gb):
    ax.text(bar.get_x() + bar.get_width() / 2.0, v + max(p2_hbm_gb) * 0.03, f"{v:,.1f} GB", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

  # Panel 3: Geometry-Sharded 8B-16B Unique Triangles @ Native 4K & 8K (ICI pmin/psum)
  ax = axes[1, 0]
  g_labels = [f"{r['total_unique_tris']/1e9:.2f}B\n{r['resolution'].split('x')[0]}px" for r in b4]
  g_fps = [r["fps"] for r in b4]
  g_ms = [r["dt_ms"] for r in b4]
  g_cols = ["#38BDF8", "#0284C7", "#F472B6", "#DB2777"][: len(b4)]
  bars3 = ax.bar(range(len(g_labels)), g_fps, color=g_cols, width=0.55)
  ax.set_xticks(range(len(g_labels)))
  ax.set_xticklabels(g_labels, fontsize=9.5)
  ax.set_ylabel("Frame Rate (FPS)", fontsize=10.5)
  ax.set_title("3. Geometry-Sharded Unique Triangles (pmin ICI @ 4K & 8K)", fontsize=12.5, fontweight="bold", pad=10)
  ax.grid(True, axis="y")
  ax.set_ylim(0, max(g_fps) * 1.28)
  for bar, f_v, m_v in zip(bars3, g_fps, g_ms):
    ax.text(bar.get_x() + bar.get_width() / 2.0, f_v + max(g_fps) * 0.03, f"{f_v:.2f} FPS\n({m_v:.1f} ms)", ha="center", va="bottom", fontsize=8.8, fontweight="bold")

  # Panel 4: Spatial-Sharded Native 8K Path Tracing + LiDAR Ray Rate Summary (dual axis)
  ax = axes[1, 1]
  pt_labels = [f"8K PT b{r['max_bounces']}\nspp={r['spp']}" for r in b5]
  pt_vals = [r["samples_per_min"] for r in b5]
  bars4 = ax.bar(range(len(pt_labels)), pt_vals, color=["#FBBF24", "#F59E0B"], width=0.45)
  ax.set_ylabel("8K Path Tracing (samples/min)", fontsize=10.5, color="#FBBF24")
  ax.set_ylim(0, max(pt_vals) * 1.32)
  ax.grid(True, axis="y")
  for bar, v in zip(bars4, pt_vals):
    ax.text(bar.get_x() + bar.get_width() / 2.0, v + max(pt_vals) * 0.03, f"{v:,.1f}\nsamples/min", ha="center", va="bottom", fontsize=9.2, fontweight="bold", color="#FBBF24")

  ax2 = ax.twinx()
  lidar_x = [len(pt_labels)]
  lidar_v = [b3[-1]["brays_per_s"]]
  bar_l = ax2.bar(lidar_x, lidar_v, color="#34D399", width=0.45)
  ax2.set_ylabel("Peak LiDAR Ray Rate (B rays/s)", fontsize=10.5, color="#34D399")
  ax2.set_ylim(0, max(lidar_v) * 1.32)
  ax2.tick_params(axis="y", colors="#34D399")
  ax2.spines["right"].set_color("#34D399")
  ax2.text(lidar_x[0], lidar_v[0] + max(lidar_v) * 0.03, f"{lidar_v[0]:.2f}\nB rays/s", ha="center", va="bottom", fontsize=9.2, fontweight="bold", color="#34D399")

  ax.set_xticks(range(len(pt_labels) + 1))
  ax.set_xticklabels(pt_labels + [f"LiDAR B={b3[-1]['global_batch']:,}\n(Peak Rate)"], fontsize=9.5)
  ax.set_title("4. Native 8K (7680x4320) 3-Bounce PT & Peak LiDAR Rate", fontsize=12.5, fontweight="bold", pad=10)

  os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
  fig.savefig(out_png, facecolor=fig.get_facecolor())
  plt.close(fig)
  print(f"Saved pod scaling figure: {out_png}", flush=True)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--in_json", default="out/metrics/v3_pod_v5p64_capstone.json")
  ap.add_argument("--out_png", default="out/figures/v3_pod_v5p64_scaling.png")
  args = ap.parse_args()
  plot_pod_scaling(args.in_json, args.out_png)


if __name__ == "__main__":
  main()
