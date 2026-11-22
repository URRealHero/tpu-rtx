"""Pure-NumPy/Matplotlib figure generator for Wave 5 (safe for Mac, zero JAX/MuJoCo imports)."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _apply_dark_theme() -> None:
  plt.rcParams.update({
      "figure.facecolor": "#0b0f17",
      "axes.facecolor": "#111824",
      "axes.edgecolor": "#2c384a",
      "axes.labelcolor": "#dce4ef",
      "xtick.color": "#9fb0c7",
      "ytick.color": "#9fb0c7",
      "text.color": "#edf2f8",
      "grid.color": "#243042",
  })


def make_menagerie_pbr_figure(
    renders_npz: str,
    task_a_json: str,
    out_png: str,
) -> None:
  _apply_dark_theme()
  smp = np.load(renders_npz)
  with open(task_a_json, "r") as f:
    meta = json.load(f)

  fig = plt.figure(figsize=(22.0, 11.8), dpi=180, facecolor="#0b0f17")
  gs = fig.add_gridspec(2, 3, width_ratios=[1.12, 1.12, 1.05], wspace=0.20, hspace=0.26)

  scenes_meta = meta.get("gallery_pbr", {}).get("scenes", {})
  panels = [
      (0, 0, "unitree_g1", "Unitree G1 Humanoid (35-DOF)"),
      (0, 1, "unitree_go2", "Unitree Go2 Quadruped (18-DOF)"),
      (1, 0, "franka_emika_panda", "Franka Emika Panda (9-DOF Arm)"),
      (1, 1, "tetheria_aero_hand", "Tetheria Aero Hand Open (16-DOF)"),
  ]

  for r, c, key, title in panels:
    ax = fig.add_subplot(gs[r, c])
    raw_img = smp[f"{key}_rgb"]
    img = (raw_img.astype(np.float32) / 255.0) if raw_img.dtype == np.uint8 else np.clip(raw_img, 0.0, 1.0)
    ax.imshow(img)
    sm = scenes_meta.get(key, {})
    ntris = int(sm.get("total_tris", 0))
    nmesh = sm.get("n_mesh_geoms", "?")
    ao_n = sm.get("n_ao_samples", 8)
    sh_n = sm.get("n_shadow_samples", 8)
    dt_ms = float(sm.get("dt_render_ms", 0.0))
    sub = f"{nmesh} Meshes ({ntris:,} tris) | 1024x1024 GGX + {ao_n}-AO + {sh_n}-Sh + AgX ({dt_ms:.1f} ms)"
    ax.set_title(f"{title}\n{sub}", fontsize=9.2, fontweight="bold", pad=6, color="#f0f4fa")
    ax.axis("off")

  bench_rows = meta.get("benchmark_4chip", [])
  robot_order = ["unitree_g1", "unitree_go2", "franka_emika_panda", "tetheria_aero_hand"]
  nice = {
      "unitree_g1": "Unitree G1\n(393k tris)",
      "unitree_go2": "Unitree Go2\n(398k tris)",
      "franka_emika_panda": "Franka Panda\n(135k tris)",
      "tetheria_aero_hand": "Aero Hand\n(85k tris)",
  }

  # Top-right: 4-chip Coupled MJX + TPU-RTX Throughput (64x64 vs 128x128)
  ax_coupled = fig.add_subplot(gs[0, 2])
  c_64, c_128 = [], []
  r_64, r_128 = [], []
  for k in robot_order:
    row64 = next((x for x in bench_rows if x["robot"] == k and x["res_int"] == 64), None)
    row128 = next((x for x in bench_rows if x["robot"] == k and x["res_int"] == 128), None)
    c_64.append(row64["coupled_fps"] / 1e3 if row64 else 0.0)
    c_128.append(row128["coupled_fps"] / 1e3 if row128 else 0.0)
    r_64.append(row64["render_only_fps"] / 1e3 if row64 else 0.0)
    r_128.append(row128["render_only_fps"] / 1e3 if row128 else 0.0)

  y = np.arange(len(robot_order))
  h = 0.34
  ax_coupled.barh(y - h / 2, c_64, height=h, color="#38bdf8", label="64x64 Coupled MJX+RTX")
  ax_coupled.barh(y + h / 2, c_128, height=h, color="#f97316", label="128x128 Coupled MJX+RTX")
  ax_coupled.set_yticks(y)
  ax_coupled.set_yticklabels([nice[k] for k in robot_order], fontsize=8.8, fontweight="bold")
  ax_coupled.invert_yaxis()
  vmax_c = max(max(c_64 + c_128), 1e-3)
  for yi in range(len(robot_order)):
    lbl64 = f"{c_64[yi]:.2f}k/s" if c_64[yi] >= 1.0 else f"{c_64[yi]*1e3:.0f} FPS"
    lbl128 = f"{c_128[yi]:.2f}k/s" if c_128[yi] >= 1.0 else f"{c_128[yi]*1e3:.0f} FPS"
    ax_coupled.text(c_64[yi] + vmax_c * 0.02, y[yi] - h / 2, lbl64, va="center", ha="left", fontsize=8.2, color="#bae6fd", fontweight="bold")
    ax_coupled.text(c_128[yi] + vmax_c * 0.02, y[yi] + h / 2, lbl128, va="center", ha="left", fontsize=8.2, color="#fed7aa", fontweight="bold")
  ax_coupled.set_xlim(0, vmax_c * 1.28)
  ax_coupled.set_xlabel("Coupled MJX Physics + TPU-RTX (k env-steps/s)", fontsize=9.2, fontweight="bold")
  b_val = bench_rows[0]["batch_size"] if bench_rows else 128
  ax_coupled.set_title(f"4-Chip shard_map Coupled MJX + tpu_rtx Throughput\n(Official Menagerie MJCFs, B={b_val}, host_callbacks=0)", fontsize=10.2, fontweight="bold")
  ax_coupled.grid(axis="x", alpha=0.35)
  ax_coupled.legend(fontsize=8.2, loc="lower right", facecolor="#0f1724", edgecolor="#334155")

  # Bottom-right: Render-only Wide-BVH-64 TPU-RTX Throughput (64x64 vs 128x128)
  ax_ren = fig.add_subplot(gs[1, 2])
  ax_ren.barh(y - h / 2, r_64, height=h, color="#34d399", label="64x64 tpu_rtx.render")
  ax_ren.barh(y + h / 2, r_128, height=h, color="#a78bfa", label="128x128 tpu_rtx.render")
  ax_ren.set_yticks(y)
  ax_ren.set_yticklabels([nice[k] for k in robot_order], fontsize=8.8, fontweight="bold")
  ax_ren.invert_yaxis()
  vmax_r = max(max(r_64 + r_128), 1e-3)
  for yi in range(len(robot_order)):
    ax_ren.text(r_64[yi] + vmax_r * 0.02, y[yi] - h / 2, f"{r_64[yi]:.2f}k FPS", va="center", ha="left", fontsize=8.2, color="#a7f3d0", fontweight="bold")
    ax_ren.text(r_128[yi] + vmax_r * 0.02, y[yi] + h / 2, f"{r_128[yi]:.2f}k FPS", va="center", ha="left", fontsize=8.2, color="#ddd6fe", fontweight="bold")
  ax_ren.set_xlim(0, vmax_r * 1.28)
  ax_ren.set_xlabel("Zero-Copy TPU-RTX Wide-BVH-64 Render (k FPS)", fontsize=9.2, fontweight="bold")
  ax_ren.set_title("4-Chip Wide-BVH-64 Triangle Mesh Render Throughput\n(Multi-Instance Mosaic Kernel, gather_ops=0)", fontsize=10.2, fontweight="bold")
  ax_ren.grid(axis="x", alpha=0.35)
  ax_ren.legend(fontsize=8.2, loc="lower right", facecolor="#0f1724", edgecolor="#334155")

  fig.suptitle(
      "TPU-RTX Wave 5: Official google-deepmind/mujoco_menagerie 1024x1024 GGX PBR+AO+SoftShadow Gallery & 4-Chip MJX Benchmarks",
      fontsize=13.5,
      fontweight="bold",
      y=0.982,
      color="#f8fafc",
  )
  os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
  fig.savefig(out_png, bbox_inches="tight", facecolor="#0b0f17")
  plt.close(fig)
  print("Saved Task A figure ->", out_png, flush=True)


def make_diff_phys_figure(
    frames_npz: str,
    task_b_json: str,
    out_png: str,
) -> None:
  _apply_dark_theme()
  smp = np.load(frames_npz)
  with open(task_b_json, "r") as f:
    meta = json.load(f)

  fig = plt.figure(figsize=(20.0, 10.6), dpi=180, facecolor="#0b0f17")
  gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1.0], wspace=0.22, hspace=0.32)

  frame_panels = [
      (0, "rgb_init", f"Initial Trajectory (Iter 0)\nPSNR: {meta['init_psnr_db']:.2f} dB | EE Err: {meta['init_ee_err_cm']:.1f} cm"),
      (1, "rgb_mid", f"Mid Optimization (Iter {meta['mid_iter']})\nPSNR: {meta['mid_psnr_db']:.2f} dB | EE Err: {meta['mid_ee_err_cm']:.1f} cm"),
      (2, "rgb_final", f"Converged (Iter {meta['n_iters']})\nPSNR: {meta['final_psnr_db']:.2f} dB | EE Err: {meta['final_ee_err_cm']:.2f} cm"),
      (3, "rgb_target", f"Target Pose (Pixel Reference)\nZero State Supervision (RGB L2 Only)"),
  ]
  for col, k, title in frame_panels:
    ax = fig.add_subplot(gs[0, col])
    raw = smp[k]
    img = (raw.astype(np.float32) / 255.0) if raw.dtype == np.uint8 else np.clip(raw, 0.0, 1.0)
    ax.imshow(img)
    ax.set_title(title, fontsize=9.8, fontweight="bold", pad=7, color="#38bdf8" if k == "rgb_target" else "#f0f4fa")
    for sp in ax.spines.values():
      sp.set_edgecolor("#38bdf8" if k == "rgb_target" else "#334155")
      sp.set_linewidth(2.2 if k == "rgb_target" else 1.2)
    ax.set_xticks([])
    ax.set_yticks([])

  hist = meta["history"]
  iters = np.array([h["iter"] for h in hist], dtype=np.float32)
  psnrs = np.array([h["psnr_db"] for h in hist], dtype=np.float32)
  mses = np.array([h["pixel_mse"] for h in hist], dtype=np.float32)
  ee_cm = np.array([h["ee_err_cm"] for h in hist], dtype=np.float32)
  q_deg = np.array([h["q_rmse_deg"] for h in hist], dtype=np.float32)

  # Bottom row: 3 wide plots spanning the 4 columns cleanly via subgridspec
  gs_bot = gs[1, :].subgridspec(1, 3, wspace=0.40)

  ax_psnr = fig.add_subplot(gs_bot[0, 0])
  ax_psnr.plot(iters, psnrs, color="#34d399", lw=2.5, marker="o", ms=3.5)
  ax_psnr.set_xlabel("jax.value_and_grad Iteration", fontsize=9.5, fontweight="bold")
  ax_psnr.set_ylabel("Rendered RGB PSNR (dB)", fontsize=9.5, fontweight="bold")
  ax_psnr.set_title(f"RGB Pixel Convergence (PSNR)\n{meta['init_psnr_db']:.1f} dB -> {meta['final_psnr_db']:.2f} dB", fontsize=10.2, fontweight="bold")
  ax_psnr.grid(True, alpha=0.35)

  ax_ee = fig.add_subplot(gs_bot[0, 1])
  l1, = ax_ee.plot(iters, ee_cm, color="#f97316", lw=2.5, marker="s", ms=3.5, label="End-Effector Error (cm)")
  ax_ee.set_xlabel("jax.value_and_grad Iteration", fontsize=9.5, fontweight="bold")
  ax_ee.set_ylabel("End-Effector 3D Error (cm)", fontsize=9.5, fontweight="bold", color="#f97316")
  ax_ee2 = ax_ee.twinx()
  l2, = ax_ee2.plot(iters, q_deg, color="#38bdf8", lw=2.2, ls="--", label="Joint RMSE (deg)")
  ax_ee2.set_ylabel("Joint Angle RMSE (deg)", fontsize=9.5, fontweight="bold", color="#38bdf8")
  ax_ee.set_title(f"Pose Recovery (Zero State Loss)\nEE: {meta['init_ee_err_cm']:.1f} cm -> {meta['final_ee_err_cm']:.2f} cm | Joint: {meta['final_q_rmse_deg']:.2f} deg", fontsize=10.0, fontweight="bold")
  ax_ee.grid(True, alpha=0.35)
  ax_ee.legend([l1, l2], ["EE 3D Error (cm)", "Joint RMSE (deg)"], fontsize=8.2, loc="upper right", facecolor="#0f1724", edgecolor="#334155")

  ax_mse = fig.add_subplot(gs_bot[0, 2])
  ax_mse.semilogy(iters, np.maximum(mses, 1e-8), color="#a78bfa", lw=2.5, label="Pixel MSE (Log)")
  ax_mse.set_xlabel("jax.value_and_grad Iteration", fontsize=9.5, fontweight="bold")
  ax_mse.set_ylabel("Rendered RGB Pixel MSE", fontsize=9.5, fontweight="bold")
  ax_mse.set_title(
      f"Coupled XLA Throughput & Pixel MSE\n"
      f"{meta['grad_steps_per_sec']:.1f} grad-steps/s | {meta['coupled_phys_steps_per_sec']:.0f} MJX-steps/s (hc=0)",
      fontsize=10.0, fontweight="bold",
  )
  ax_mse.grid(True, which="both", alpha=0.35)
  ax_mse.legend(fontsize=8.5, loc="upper right", facecolor="#0f1724", edgecolor="#334155")

  fig.suptitle(
      f"TPU-RTX Wave 5: End-to-End Differentiable MJX Physics + Smooth-Visibility PBR Rendering ({meta['n_dof']}-DOF Pixel-to-Torque Servoing)",
      fontsize=13.2,
      fontweight="bold",
      y=0.985,
      color="#f8fafc",
  )
  os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
  fig.savefig(out_png, bbox_inches="tight", facecolor="#0b0f17")
  plt.close(fig)
  print("Saved Task B figure ->", out_png, flush=True)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--mode", choices=("taskA", "taskB", "both"), default="both")
  ap.add_argument("--taskA_npz", default="out/metrics/v3_w0_menagerie_gallery.npz")
  ap.add_argument("--taskA_json", default="out/metrics/v3_w0_taskA.json")
  ap.add_argument("--taskA_png", default="out/figures/v3_w0_mujoco_menagerie_pbr.png")
  ap.add_argument("--taskB_npz", default="out/metrics/v3_w0_diff_phys_frames.npz")
  ap.add_argument("--taskB_json", default="out/metrics/v3_w0_taskB.json")
  ap.add_argument("--taskB_png", default="out/figures/v3_w0_diff_phys_render.png")
  args = ap.parse_args()

  if args.mode in ("taskA", "both"):
    make_menagerie_pbr_figure(args.taskA_npz, args.taskA_json, args.taskA_png)
  if args.mode in ("taskB", "both"):
    make_diff_phys_figure(args.taskB_npz, args.taskB_json, args.taskB_png)


if __name__ == "__main__":
  main()
