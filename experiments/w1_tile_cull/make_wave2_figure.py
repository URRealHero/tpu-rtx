"""Generate Wave 2 Cluttered Robotics Warehouse Showcase + Scaling Figure (pure matplotlib/numpy, zero JAX)."""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--json_path", default="out/metrics/v3_w1_wave2_complex_mjx.json")
  ap.add_argument("--npz_path", default="experiments/w1_tile_cull/warehouse_showcase.npz")
  ap.add_argument("--out_png", default="out/figures/v3_w1_complex_mjx_warehouse.png")
  args = ap.parse_args()

  data = json.load(open(args.json_path))
  ds = np.load(args.npz_path)

  fig = plt.figure(figsize=(18, 9.2), dpi=160)
  gs = fig.add_gridspec(2, 4, height_ratios=[1.1, 1.0], hspace=0.28, wspace=0.20)

  # Top row: 4 visual panels
  ax1 = fig.add_subplot(gs[0, 0])
  ax1.imshow(np.clip(ds["g200_256_overview_rgb"], 0, 1))
  ax1.set_title("Cluttered Warehouse (G=204)\n256x256 Overview (TPU Mosaic)", fontsize=10, fontweight="bold")
  ax1.axis("off")

  ax2 = fig.add_subplot(gs[0, 1])
  ax2.imshow(np.clip(ds["g1000_256_overview_rgb"], 0, 1))
  ax2.set_title("Dense Warehouse (G=1,004)\n256x256 Overview (TPU Mosaic)", fontsize=10, fontweight="bold")
  ax2.axis("off")

  ax3 = fig.add_subplot(gs[0, 2])
  ax3.imshow(np.clip(ds["g1000_256_closeup_rgb"], 0, 1))
  ax3.set_title("Dual-Arm & Workbench (G=1,004)\n256x256 Closeup (Exact Shadows)", fontsize=10, fontweight="bold")
  ax3.axis("off")

  ax4 = fig.add_subplot(gs[0, 3])
  d_img = ds["g1000_256_closeup_depth"]
  d_vis = np.where(d_img > 0, 1.0 - np.clip((d_img - 0.5) / 3.5, 0.0, 1.0), 0.0)
  ax4.imshow(d_vis, cmap="inferno", vmin=0.0, vmax=1.0)
  ax4.set_title("Exact Metric Depth (G=1,004)\n0.0 m MAE vs Flat Mosaic", fontsize=10, fontweight="bold")
  ax4.axis("off")

  # Bottom-left (cols 0..1): 1-Chip Render Time & Culling Speedup across G = 55 -> 204 -> 1004
  ax5 = fig.add_subplot(gs[1, 0:2])
  r1c = {(r["n_target_prims"], r["resolution"], r["mode"]): r for r in data.get("runs_1c", []) if r.get("status") == "ok"}

  prims_list = [50, 200, 1000]
  x = np.arange(len(prims_list))
  w = 0.25
  res_colors = [("64x64", "#1f77b4"), ("128x128", "#ff7f0e"), ("256x256", "#2ca02c")]
  for i, (res_str, col) in enumerate(res_colors):
    flat_ms = [r1c.get((p, res_str, "flat_loop"), {}).get("dt_render_ms", np.nan) for p in prims_list]
    cull_ms = [r1c.get((p, res_str, "cluster1"), {}).get("dt_render_ms", np.nan) for p in prims_list]
    ax5.bar(x + (i - 1) * w - w * 0.2, flat_ms, width=w * 0.42, color=col, alpha=0.38, edgecolor=col, hatch="//", label=f"{res_str} Flat (No Cull)" if i == 0 else None)
    bars = ax5.bar(x + (i - 1) * w + w * 0.2, cull_ms, width=w * 0.42, color=col, alpha=0.95, label=f"{res_str} Culled (Morton)")
    for b, fv, cv in zip(bars, flat_ms, cull_ms):
      if not np.isnan(cv):
        sp = fv / cv if (not np.isnan(fv) and cv > 0) else 1.0
        ax5.text(b.get_x() + b.get_width() / 2, cv + 3, f"{cv:.0f}ms\n({sp:.2f}x)", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

  ax5.set_xticks(x)
  ax5.set_xticklabels(["G=55\n(pallas_v2 OK)", "G=204\n(pallas_v2 VMEM OOM!)", "G=1,004\n(1,004 moving prims)"], fontsize=9.5, fontweight="bold")
  ax5.set_ylabel("1-Chip Render Time (ms)", fontsize=10, fontweight="bold")
  ax5.set_title("Dynamic Primitive Scaling on Live TPU v5p (1 Chip, MJX+Render)\nHatched = Flat Loop | Solid = Morton Cluster Culling (Zero Host Callbacks)", fontsize=10.5, fontweight="bold")
  ax5.grid(axis="y", alpha=0.3)
  ax5.legend(fontsize=8.5, loc="upper left")

  # Bottom-right (cols 2..3): 1-Chip vs 4-Chip shard_map Throughput (FPS)
  ax6 = fig.add_subplot(gs[1, 2:4])
  r4c = {(r["n_target_prims"], r["resolution"], r["mode"]): r for r in data.get("runs_4c", []) if r.get("status") == "ok"}

  labels_4c = []
  fps_1c = []
  fps_4c = []
  for p, g_lbl in [(50, "G=55"), (200, "G=204"), (1000, "G=1004")]:
    for res_str, b_lbl in [("64x64", "64²"), ("128x128", "128²"), ("256x256", "256²")]:
      m4 = r4c.get((p, res_str, "cluster1")) or r4c.get((p, res_str, "hier2"))
      m1 = r1c.get((p, res_str, "cluster1")) or r1c.get((p, res_str, "hier2"))
      if m4 and m1:
        labels_4c.append(f"{g_lbl}\n{b_lbl}")
        fps_1c.append(m1["e2e_fps"])
        fps_4c.append(m4["e2e_fps"])

  x2 = np.arange(len(labels_4c))
  w2 = 0.36
  ax6.bar(x2 - w2 / 2, fps_1c, width=w2, color="#4c72b0", label="1 Chip TPU v5p")
  b4 = ax6.bar(x2 + w2 / 2, fps_4c, width=w2, color="#dd8452", label="4 Chips shard_map TPU v5p")
  for b, v1, v4 in zip(b4, fps_1c, fps_4c):
    ax6.text(b.get_x() + b.get_width() / 2, v4 * 1.06, f"{v4:,.0f}\n({v4/max(v1,1e-9):.2f}x)", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

  ax6.set_yscale("log")
  ax6.set_ylim(100, max(fps_4c + [10000]) * 2.8)
  ax6.set_xticks(x2)
  ax6.set_xticklabels(labels_4c, fontsize=8.5, fontweight="bold")
  ax6.set_ylabel("E2E MJX + Render Throughput (FPS, log)", fontsize=10, fontweight="bold")
  ax6.set_title("Multi-Chip shard_map E2E Scaling on Complex Cluttered Warehouse\n(B=1024@64², B=256@128², B=64@256², host_callbacks=0)", fontsize=10.5, fontweight="bold")
  ax6.grid(axis="y", alpha=0.3, which="both")
  ax6.legend(fontsize=8.5, loc="upper right")

  plt.suptitle("V3 Wave 2: Cluttered Dual-Arm Warehouse (55 -> 1,004 Moving MJX Primitives) on Live Google Cloud TPU v5p", fontsize=12.5, fontweight="bold", y=0.98)
  os.makedirs(os.path.dirname(args.out_png) or ".", exist_ok=True)
  plt.savefig(args.out_png, bbox_inches="tight")
  plt.close(fig)
  print("Saved figure:", args.out_png)


if __name__ == "__main__":
  main()
