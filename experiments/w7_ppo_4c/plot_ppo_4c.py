"""Generate V3 4-chip Vision PPO convergence & throughput scaling figure."""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
  root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
  v3_path = os.path.join(root, "out/metrics/v3_w7_ppo_4c.json")
  v2_path = os.path.join(root, "out/metrics/v2_live_ppo_panda.json")
  out_png = os.path.join(root, "out/figures/v3_w7_ppo_curve.png")

  v3 = json.load(open(v3_path))
  runs = {r["label"]: r for r in v3["runs"]}
  v2 = json.load(open(v2_path)) if os.path.exists(v2_path) else None

  plt.rcParams.update({"font.size": 9.5, "axes.grid": True, "grid.alpha": 0.25})
  fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6), dpi=180)

  # Panel 1: PandaPickCube Reward & Success Rate (V2 vs V3 4-Chip)
  ax = axes[0]
  ax2 = ax.twinx()
  ax2.grid(False)
  panda_styles = [
      ("panda_stage1_b1024_4c", "V3 S1 B=1024 (4c)", "#1f77b4", "-"),
      ("panda_stage2_m2_b256_4c", "V3 S2+M2 B=256 (4c)", "#d62728", "-"),
      ("panda_stage2_m2_b512_4c", "V3 S2+M2 B=512 (4c)", "#ff7f0e", "--"),
  ]
  for lbl, name, col, ls in panda_styles:
    h = runs[lbl]["history"]
    xs = [r["iter"] for r in h]
    rew = [r["mean_reward"] for r in h]
    succ = [r["success_rate"] * 100.0 for r in h]
    ax.plot(xs, rew, color=col, linestyle=ls, linewidth=2.0, label=f"{name} Rew")
    ax2.plot(xs, succ, color=col, linestyle=":", linewidth=1.6, alpha=0.85, label=f"{name} Succ%")
  if v2 is not None:
    v2_h = v2["history"]
    ax.plot([r["iter"] for r in v2_h], [r["mean_reward"] for r in v2_h],
            color="#555555", linestyle="-.", linewidth=1.8, label="V2 S2+M2 B=64 (1c, Unstable)")
  ax.set_title("PandaPickCube: Reward & Success Rate", fontweight="bold")
  ax.set_xlabel("PPO Iteration")
  ax.set_ylabel("Mean Step Reward")
  ax2.set_ylabel("Success Rate (%)")
  ax.set_ylim(1.6, 3.0)
  ax2.set_ylim(-1, 32)
  h1, l1 = ax.get_legend_handles_labels()
  h2, l2 = ax2.get_legend_handles_labels()
  ax.legend(h1[:4], l1[:4], loc="lower right", fontsize=7.5, framealpha=0.9)

  # Panel 2: Critic Loss Stability (V2 Oscillation vs V3 Clipped Critic) + Cartpole
  ax = axes[1]
  ax_cp = ax.twinx()
  ax_cp.grid(False)
  if v2 is not None:
    v2_h = v2["history"]
    ax.semilogy([r["iter"] for r in v2_h], [max(r["mean_loss"], 1e-4) for r in v2_h],
                color="#7f7f7f", linestyle="-.", linewidth=1.8, label="V2 Loss (Unclipped)")
  for lbl, name, col in [
      ("panda_stage2_m2_b512_4c", "V3 Panda S2+M2 VLoss", "#ff7f0e"),
      ("panda_stage1_b1024_4c", "V3 Panda S1 VLoss", "#1f77b4"),
  ]:
    h = runs[lbl]["history"]
    ax.semilogy([r["iter"] for r in h], [max(r["v_loss"], 1e-4) for r in h],
                color=col, linewidth=1.8, label=name)
  for lbl, name, col in [
      ("cartpole_stage1_b512_4c", "Cartpole S1 PoleCos", "#2ca02c"),
      ("cartpole_stage2_m2_b512_4c", "Cartpole S2+M2 PoleCos", "#9467bd"),
  ]:
    h = runs[lbl]["history"]
    ax_cp.plot([r["iter"] for r in h], [r["pole_cos"] for r in h],
               color=col, linestyle="--", linewidth=1.8, label=name)
  ax.set_title("Critic Stabilization & Cartpole Balance", fontweight="bold")
  ax.set_xlabel("PPO Iteration")
  ax.set_ylabel("Value / Total Loss (log)")
  ax_cp.set_ylabel("Cartpole cos(theta)")
  ax_cp.set_ylim(0.80, 1.01)
  h1, l1 = ax.get_legend_handles_labels()
  h2, l2 = ax_cp.get_legend_handles_labels()
  ax.legend(h1 + h2, l1 + l2, loc="center right", fontsize=7.5, framealpha=0.9)

  # Panel 3: End-to-End Closed-Loop RL Throughput (env-steps/s)
  ax = axes[2]
  bars = [
      ("V1 1c\nPanda+M2", 377.3, "#9e9e9e"),
      ("V2 1c\nPanda+M2", v2["steady_env_steps_per_sec"] if v2 else 3445.4, "#607d8b"),
      ("V3 4c\nPanda S2\nB=256", runs["panda_stage2_m2_b256_4c"]["steady_env_steps_per_sec"], "#d62728"),
      ("V3 4c\nPanda S2\nB=512", runs["panda_stage2_m2_b512_4c"]["steady_env_steps_per_sec"], "#ff7f0e"),
      ("V3 4c\nPanda S1\nB=512", runs["panda_stage1_b512_4c"]["steady_env_steps_per_sec"], "#1f77b4"),
      ("V3 4c\nPanda S1\nB=1024", runs["panda_stage1_b1024_4c"]["steady_env_steps_per_sec"], "#17becf"),
      ("V3 4c\nCart S2\nB=512", runs["cartpole_stage2_m2_b512_4c"]["steady_env_steps_per_sec"], "#9467bd"),
      ("V3 4c\nCart S1\nB=512", runs["cartpole_stage1_b512_4c"]["steady_env_steps_per_sec"], "#2ca02c"),
  ]
  x_pos = np.arange(len(bars))
  vals = [b[1] for b in bars]
  cols = [b[2] for b in bars]
  rects = ax.bar(x_pos, vals, color=cols, width=0.68, edgecolor="black", linewidth=0.5)
  ax.axhline(10000.0, color="#b71c1c", linestyle="--", linewidth=1.4, label="10k steps/s target")
  for rect, val in zip(rects, vals):
    ax.text(
        rect.get_x() + rect.get_width() / 2.0,
        val + 1100,
        f"{val:,.0f}",
        ha="center",
        va="bottom",
        fontsize=7.3,
        fontweight="bold",
    )
  ax.set_xticks(x_pos)
  ax.set_xticklabels([b[0] for b in bars], fontsize=7.2)
  ax.set_ylim(0, max(vals) * 1.15)
  ax.set_title("Closed-Loop Vision PPO Throughput (TPU v5p)", fontweight="bold")
  ax.set_ylabel("Steady-State Env-Steps / s")
  ax.legend(loc="upper left", fontsize=8.0, framealpha=0.9)

  plt.suptitle("V3 4-Chip shard_map Zero-Copy Vision PPO: Convergence Fix & >10k–60k Env-Steps/s Scaling (TPU v5p)",
               fontsize=11.5, fontweight="bold", y=1.01)
  plt.tight_layout()
  os.makedirs(os.path.dirname(out_png), exist_ok=True)
  plt.savefig(out_png, bbox_inches="tight")
  plt.close(fig)
  print("Saved figure ->", out_png)


if __name__ == "__main__":
  main()
