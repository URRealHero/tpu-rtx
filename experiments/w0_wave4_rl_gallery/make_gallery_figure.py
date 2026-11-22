"""Pure-NumPy/Matplotlib figure generator for Wave 4 (safe for Mac, zero JAX/MuJoCo imports)."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def make_wave4_gallery_figure(
    renders_npz: str,
    metrics_json: str,
    out_png: str,
) -> None:
  smp = np.load(renders_npz)
  with open(metrics_json, "r") as f:
    meta = json.load(f)

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

  fig = plt.figure(figsize=(21.5, 11.4), dpi=180, facecolor="#0b0f17")
  gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1.15, 1.05], wspace=0.18, hspace=0.28)

  scenes_meta = meta.get("gallery_pbr", {}).get("scenes", {})
  panels = [
      (0, 0, "humanoid_running", "Humanoid-21DOF (Running Stride)"),
      (0, 1, "panda_pick_cube", "PandaPickCube (7-DOF Precision Grasp)"),
      (1, 0, "dual_arm_tabletop", "Dual-Arm Tabletop (RoboTwin Bimanual)"),
      (1, 1, "quadruped_trot", "Quadruped-12DOF (Diagonal Trot)"),
  ]

  for r, c, key, title in panels:
    ax = fig.add_subplot(gs[r, c])
    raw_img = smp[f"{key}_rgb"]
    img = (raw_img.astype(np.float32) / 255.0) if raw_img.dtype == np.uint8 else np.clip(raw_img, 0.0, 1.0)
    ax.imshow(img)
    sm = scenes_meta.get(key, {})
    ao_n = sm.get("n_ao_samples", 12)
    sh_n = sm.get("n_shadow_samples", 12)
    dt_ms = sm.get("dt_render_ms", 0.0)
    ngeom = sm.get("ngeom", "?")
    sub = f"{ngeom} Geoms | 1024x1024 GGX + {ao_n}-AO + {sh_n}-Sh + AgX ({dt_ms:.1f} ms)"
    ax.set_title(f"{title}\n{sub}", fontsize=9.2, fontweight="bold", pad=6, color="#f0f4fa")
    ax.axis("off")

  ax_sps = fig.add_subplot(gs[0, 2])
  ppo_runs = meta.get("ppo_runs", [])
  if ppo_runs:
    lbls, vals, cols = [], [], []
    cmap = {"cartpole": "#38bdf8", "panda_pick_cube": "#f97316", "humanoid_21dof": "#34d399"}
    for run in ppo_runs:
      lbls.append(f"{run['env']} B={run['num_envs']} (ar={run['action_repeat']})")
      vals.append(run["steady_env_steps_per_sec"] / 1e3)
      cols.append(cmap.get(run["env"], "#a78bfa"))
    y = np.arange(len(lbls))
    ax_sps.barh(y, vals, color=cols, height=0.58)
    ax_sps.set_yticks([])
    ax_sps.invert_yaxis()
    vmax = max(vals)
    for yi, (lbl, v) in enumerate(zip(lbls, vals)):
      if v > vmax * 0.52:
        ax_sps.text(v * 0.97, yi, f"{lbl}: {v:,.1f}k/s", va="center", ha="right", color="#090d14", fontsize=8.3, fontweight="bold")
      else:
        ax_sps.text(v + vmax * 0.02, yi, f"{lbl}: {v:,.1f}k/s", va="center", ha="left", color="#edf2f8", fontsize=8.3, fontweight="bold")
    ax_sps.set_xlim(0, vmax * 1.18)
    ax_sps.set_xlabel("Closed-Loop Vision PPO (k env-steps/s)", fontsize=9.5, fontweight="bold")
    ax_sps.set_title("4-Chip shard_map Vision PPO Throughput\n(HBM Zero-Copy tpu_rtx, host_callbacks=0)", fontsize=10.5, fontweight="bold")
    ax_sps.grid(axis="x", alpha=0.35)

  ax_rew = fig.add_subplot(gs[1, 2])
  if ppo_runs:
    best_by_env: Dict[str, Dict[str, Any]] = {}
    for run in ppo_runs:
      en = run["env"]
      ev_i = max(1e-6, float(run.get("eval_init", {}).get("ep_return", 1.0)))
      ev_f = float(run.get("eval_final", {}).get("ep_return", 0.0))
      gain = ev_f / ev_i
      prev_gain = (
          float(best_by_env[en].get("eval_final", {}).get("ep_return", 0.0))
          / max(1e-6, float(best_by_env[en].get("eval_init", {}).get("ep_return", 1.0)))
          if en in best_by_env
          else -1.0
      )
      if gain > prev_gain:
        best_by_env[en] = run
    cmap = {"cartpole": "#38bdf8", "panda_pick_cube": "#f97316", "humanoid_21dof": "#34d399"}
    nice = {"cartpole": "Cartpole (2-DOF)", "panda_pick_cube": "PandaPickCube (15-DOF)", "humanoid_21dof": "Humanoid-21DOF (27-DOF)"}
    for en in ("cartpole", "panda_pick_cube", "humanoid_21dof"):
      if en not in best_by_env:
        continue
      run = best_by_env[en]
      hist = run["history"]
      xs = np.array([h["iter"] for h in hist], dtype=np.float32)
      ys_raw = np.array([h["mean_reward"] for h in hist], dtype=np.float32)
      r0 = float(np.min(ys_raw[:min(10, len(ys_raw))]))
      ys_pct = 100.0 * (ys_raw - r0) / max(abs(r0), 1e-6)
      ev0 = float(run["eval_init"]["ep_return"])
      ev1 = float(run["eval_final"]["ep_return"])
      ax_rew.plot(
          xs,
          ys_pct,
          lw=2.3,
          color=cmap.get(en, "#fff"),
          label=f"{nice[en]} (Eval {ev0:.1f} -> {ev1:.1f})",
      )
    ax_rew.axhline(0.0, color="#475569", lw=1.0, ls="--", alpha=0.6)
    ax_rew.set_xlabel("PPO Iteration (4 TPU v5p Chips)", fontsize=9.5, fontweight="bold")
    ax_rew.set_ylabel("Step Reward Gain vs Initial Floor (%)", fontsize=9.5, fontweight="bold")
    ax_rew.set_title("Closed-Loop Vision PPO Convergence\n(2-Frame Stack + CoordConv + Critic EMA Norm)", fontsize=10.5, fontweight="bold")
    ax_rew.grid(True, alpha=0.35)
    ax_rew.legend(fontsize=8.0, loc="lower right", facecolor="#0f1724", edgecolor="#334155")

  fig.suptitle(
      "TPU-RTX Wave 4: Multi-Embodiment 1024x1024 GGX PBR+AO+SoftShadow Gallery & Zero-Copy Vision PPO",
      fontsize=14.0,
      fontweight="bold",
      y=0.985,
      color="#f8fafc",
  )
  os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
  fig.savefig(out_png, bbox_inches="tight", facecolor="#0b0f17")
  plt.close(fig)
  print("Saved figure ->", out_png, flush=True)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--npz", required=True)
  ap.add_argument("--json", required=True)
  ap.add_argument("--out_png", required=True)
  args = ap.parse_args()
  make_wave4_gallery_figure(args.npz, args.json, args.out_png)


if __name__ == "__main__":
  main()
