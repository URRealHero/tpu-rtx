"""Generate publication-quality visual comparison, Pareto, and PPO training figures."""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def make_stage1_gbuffer_figure(dataset_path: str, out_path: str, frame_idx: int = 337):
  ds = np.load(dataset_path)
  if frame_idx >= len(ds["s1_rgb"]):
    frame_idx = 0
  rgb = np.clip(ds["s1_rgb"][frame_idx], 0, 1)
  alb = np.clip(ds["s1_albedo"][frame_idx], 0, 1)
  nrm = np.clip(0.5 * ds["s1_normal"][frame_idx] + 0.5, 0, 1)
  dep = ds["s1_depth"][frame_idx]
  dep_vis = np.where(dep > 0, np.clip((dep - 0.5) / 1.5, 0, 1), 1.0)
  sh = ds["s1_shadow"][frame_idx]

  fig, axes = plt.subplots(1, 5, figsize=(15, 3.2), dpi=160)
  panels = [
      (rgb, "Stage 1 Direct RGB", None),
      (alb, "Albedo", None),
      (nrm, "World Normal", None),
      (1.0 - dep_vis, "Inverse Depth", "inferno"),
      (sh, "Hard Shadow Mask", "gray"),
  ]
  for ax, (img, title, cmap) in zip(axes, panels):
    ax.imshow(img, cmap=cmap, vmin=0 if cmap else None, vmax=1 if cmap else None)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.axis("off")
  plt.suptitle("Stage 1 Exact Analytic Primitives G-Buffer (Zero-Copy from MJX Data)", fontsize=12, y=1.02)
  plt.tight_layout()
  plt.savefig(out_path, bbox_inches="tight")
  plt.close(fig)


def make_stage2_comparison_figure(preds_path: str, summary_path: str, out_path: str, frame_idx: int = 17):
  d = np.load(preds_path)
  summary = json.load(open(summary_path))
  if frame_idx >= len(d["s1_rgb"]):
    frame_idx = 0

  s1 = np.clip(d["s1_rgb"][frame_idx], 0, 1)
  m1 = np.clip(d["pred_M1_pixel_128"][frame_idx], 0, 1) if "pred_M1_pixel_128" in d else s1
  m2 = np.clip(d["pred_M2_spatial_128"][frame_idx], 0, 1) if "pred_M2_spatial_128" in d else s1
  gt = np.clip(d["gt_rgb"][frame_idx], 0, 1)
  err = np.clip(np.abs(m2 - gt) * 5.0, 0, 1)

  p_m1 = summary.get("M1_pixel_128", {}).get("gate_cal", {}).get("psnr_db", 0.0)
  p_m2 = summary.get("M2_spatial_128", {}).get("gate_cal", {}).get("psnr_db", 0.0)
  s_m2 = summary.get("M2_spatial_128", {}).get("gate_cal", {}).get("ssim", 0.0)
  p_s1 = summary.get("M2_spatial_128", {}).get("gate_cal", {}).get("stage1_psnr_db", 0.0)

  fig, axes = plt.subplots(1, 5, figsize=(16, 3.4), dpi=160)
  titles = [
      f"Stage 1 Exact\n({p_s1:.2f} dB)",
      f"M1 PixelMLP-128\n({p_m1:.2f} dB)",
      f"M2 SpatialRelighter-128\n({p_m2:.2f} dB, SSIM {s_m2:.3f})",
      "Blender Cycles GT\n(48 spp GI + Soft Shadow)",
      "M2 Error Map (x5)",
  ]
  for ax, img, t in zip(axes, [s1, m1, m2, gt, err], titles):
    ax.imshow(img)
    ax.set_title(t, fontsize=10, fontweight="bold")
    ax.axis("off")
  plt.suptitle("Stage 2 MXU Neural Relighting vs Blender Cycles Ground Truth (Held-out Test Pose)", fontsize=12, y=1.02)
  plt.tight_layout()
  plt.savefig(out_path, bbox_inches="tight")
  plt.close(fig)


def make_pareto_and_ppo_figure(
    metrics_dir: str,
    out_path: str,
):
  s2 = json.load(open(os.path.join(metrics_dir, "stage2_summary.json")))
  fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.6), dpi=160)

  # Left: Quality vs REAL Measured TPU v5p End-to-End Throughput (MJX + Render, 64x64)
  ax = axes[0]
  t1_path = os.path.join(metrics_dir, "live_tpu_panda_tiled_1c.json")
  m2_1c_path = os.path.join(metrics_dir, "live_tpu_panda_m2_1c.json")
  m2_4c_path = os.path.join(metrics_dir, "live_tpu_panda_m2_4c.json")

  pts = []
  p_s1 = s2["M2_spatial_128"]["gate_cal"]["stage1_psnr_db"]
  p_m2 = s2["M2_spatial_128"]["gate_cal"]["psnr_db"]
  if os.path.exists(t1_path):
    d1 = json.load(open(t1_path))["e2e_runs"]
    for r in d1:
      if r["batch_size"] == 2048 and r["resolution"] == "64x64":
        pts.append((f"Stage 1 Exact\n(1x v5p, B=2048)", r["e2e_fps"] / 1e3, p_s1, "#4C72B0"))
  if os.path.exists(m2_1c_path):
    d2 = json.load(open(m2_1c_path))["e2e_runs"]
    for r in d2:
      if r["batch_size"] == 2048 and r["resolution"] == "64x64":
        pts.append((f"Stage 1 + M2-128\n(1x v5p, B=2048)", r["e2e_fps"] / 1e3, p_m2, "#55A868"))
  if os.path.exists(m2_4c_path):
    d4 = json.load(open(m2_4c_path))["e2e_runs"]
    for r in d4:
      if r["batch_size"] == 4096 and r["resolution"] == "64x64":
        pts.append((f"Stage 1 + M2-128\n(4x v5p, B=4096)", r["e2e_fps"] / 1e3, p_m2, "#C44E52"))

  for label, kfps, psnr, col in pts:
    ax.scatter(kfps, psnr, s=130, color=col, zorder=4)
    dx_off = -115 if kfps > 2.0 else 8
    dy_off = -18 if psnr > 35.0 else 4
    ax.annotate(label, (kfps, psnr), textcoords="offset points", xytext=(dx_off, dy_off), fontsize=8.5)
  ax.set_xlim(0.45, 3.20)
  ax.set_ylim(12.0, 41.5)
  ax.axhline(30.0, color="red", linestyle="--", linewidth=1.2, label="Locked Quality Gate (30.0 dB)")
  ax.axhline(s2["M1_pixel_128"]["gate_cal"]["psnr_db"], color="#DD8452", linestyle=":", linewidth=1.2, label=f"M1 PixelMLP-128 ({s2['M1_pixel_128']['gate_cal']['psnr_db']:.2f} dB, Failed)")
  ax.set_xlabel("Measured TPU v5p E2E Throughput (kFPS = MJX Step + Render)")
  ax.set_ylabel("Held-Out PSNR vs Cycles GT (dB)")
  ax.set_title("Measured TPU v5p Quality vs E2E Throughput")
  ax.grid(True, alpha=0.3)
  ax.legend(loc="lower right", fontsize=8.0)

  # Right: End-to-End Zero-Copy Vision PPO Training Curves on TPU v5p (cycle-mean over reset period)
  ax2 = axes[1]
  ax2_r = ax2.twinx()
  lines, labels = [], []
  p_cp = os.path.join(metrics_dir, "ppo_cartpole.json")
  if os.path.exists(p_cp):
    d = json.load(open(p_cp))
    xs = np.array([h["iter"] for h in d["history"]])
    ys = np.array([h["mean_reward"] for h in d["history"]])
    ax2.plot(xs, ys, linewidth=1.0, alpha=0.25, color="#4C72B0")
    xc = 0.5 * (xs[0::2] + xs[1::2])
    yc = 0.5 * (ys[0::2] + ys[1::2])
    l1, = ax2.plot(xc, yc, marker="o", linewidth=2.2, markersize=4.0, color="#4C72B0",
                   label=f"Cartpole Stage-1 Vision PPO ({d['steady_env_steps_per_sec']:.0f} steps/s)")
    lines.append(l1); labels.append(l1.get_label())
  p_pd = os.path.join(metrics_dir, "ppo_panda.json")
  if os.path.exists(p_pd):
    d = json.load(open(p_pd))
    xs = np.array([h["iter"] for h in d["history"]])
    ys = np.array([h["mean_reward"] for h in d["history"]])
    ax2_r.plot(xs, ys, linewidth=1.0, alpha=0.25, color="#55A868")
    xc = 0.5 * (xs[0::2] + xs[1::2])
    yc = 0.5 * (ys[0::2] + ys[1::2])
    l2, = ax2_r.plot(xc, yc, marker="s", linewidth=2.2, markersize=4.0, color="#55A868",
                     label=f"PandaPickCube + M2 Vision PPO ({d['steady_env_steps_per_sec']:.0f} steps/s)")
    lines.append(l2); labels.append(l2.get_label())

  ax2.set_ylim(0.84, 1.00)
  ax2_r.set_ylim(2.22, 2.45)
  ax2.set_xlabel("PPO Iteration (Bold = 2-Iter Episode Cycle Mean)")
  ax2.set_ylabel("Cartpole Mean Step Reward", color="#4C72B0")
  ax2_r.set_ylabel("PandaPickCube Mean Step Reward", color="#55A868")
  ax2.tick_params(axis="y", labelcolor="#4C72B0")
  ax2_r.tick_params(axis="y", labelcolor="#55A868")
  ax2.set_title("Zero-Copy MJX + TPU Render + Vision PPO Closed Loop")
  ax2.grid(True, alpha=0.3)
  if lines:
    ax2.legend(lines, labels, loc="lower right", fontsize=8.0)

  plt.tight_layout()
  plt.savefig(out_path, bbox_inches="tight")
  plt.close(fig)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--out_dir", default="out/figures")
  args = ap.parse_args()
  os.makedirs(args.out_dir, exist_ok=True)
  if os.path.exists("out/data/panda_cycles_dataset.npz"):
    make_stage1_gbuffer_figure("out/data/panda_cycles_dataset.npz", os.path.join(args.out_dir, "stage1_gbuffer.png"))
  if os.path.exists("out/metrics/stage2_test_preds.npz") and os.path.exists("out/metrics/stage2_summary.json"):
    make_stage2_comparison_figure(
        "out/metrics/stage2_test_preds.npz",
        "out/metrics/stage2_summary.json",
        os.path.join(args.out_dir, "stage2_cycles_comparison.png"),
    )
  if os.path.exists("out/metrics/stage2_summary.json"):
    make_pareto_and_ppo_figure("out/metrics", os.path.join(args.out_dir, "pareto_and_ppo.png"))
  print("Figures generated in", args.out_dir)


if __name__ == "__main__":
  main()
