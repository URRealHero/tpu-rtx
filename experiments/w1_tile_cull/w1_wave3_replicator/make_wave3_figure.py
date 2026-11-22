"""Generate Wave 3 Showcase Figure `out/figures/v3_w1_isaac_replicator_dr.png` (pure matplotlib/numpy)."""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
from matplotlib.colors import hsv_to_rgb
import matplotlib.pyplot as plt
import numpy as np


def _normalize_depth(d: np.ndarray) -> np.ndarray:
  valid = d >= 0.0
  if not np.any(valid):
    return np.zeros_like(d)
  lo = np.percentile(d[valid], 2)
  hi = np.percentile(d[valid], 95)
  out = np.clip((d - lo) / max(hi - lo, 1e-4), 0.0, 1.0)
  return np.where(valid, 1.0 - out, 0.0)


def _colorize_ids(ids: np.ndarray, is_sem: bool = True) -> np.ndarray:
  h, w = ids.shape
  rgb = np.full((h, w, 3), 0.05, dtype=np.float32)
  if is_sem:
    palette = np.array([
        [0.05, 0.06, 0.08],  # 0: sky
        [0.24, 0.29, 0.36],  # 1: floor / workbench
        [0.16, 0.60, 0.95],  # 2: left arm
        [0.96, 0.44, 0.14],  # 3: right arm
        [0.94, 0.20, 0.36],  # 4: grasp targets / props
        [0.22, 0.78, 0.46],  # 5: warehouse racks
        [0.72, 0.42, 0.90],  # 6: tools
    ], dtype=np.float32)
    m = ids >= 0
    idx = np.clip(ids, 0, len(palette) - 1)
    rgb[m] = palette[idx[m]]
  else:
    m = ids >= 0
    hv = ((ids.astype(np.float32) * 0.61803398875) % 1.0)
    sv = np.full_like(hv, 0.78)
    vv = np.where((ids % 2) == 0, 0.92, 0.72)
    col = hsv_to_rgb(np.stack([hv, sv, vv], axis=-1))
    rgb[m] = col[m]
  return rgb


def _colorize_flow(flow: np.ndarray) -> np.ndarray:
  fx, fy = flow[..., 0], flow[..., 1]
  mag = np.sqrt(fx * fx + fy * fy)
  ang = (np.arctan2(fy, fx) + np.pi) / (2.0 * np.pi)
  moving = mag > 1e-4
  ref = float(np.percentile(mag[moving], 96)) if np.any(moving) else 1e-3
  scale = np.clip(mag / max(ref, 1e-4), 0.0, 1.0)
  # Dark slate background for static regions, vivid saturated HSV for moving bodies/arms/trays
  hsv = np.stack([ang, np.where(moving, 0.92, 0.0), np.where(moving, 0.35 + 0.65 * scale, 0.08)], axis=-1)
  return hsv_to_rgb(hsv)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--json", default="out/metrics/v3_w1_wave3_replicator.json")
  ap.add_argument("--npz", default="experiments/w1_tile_cull/w1_wave3_replicator/replicator_showcase.npz")
  ap.add_argument("--out_png", default="out/figures/v3_w1_isaac_replicator_dr.png")
  args = ap.parse_args()

  with open(args.json, "r") as f:
    data = json.load(f)
  ds = np.load(args.npz)

  fig = plt.figure(figsize=(21.5, 14.2), dpi=180, facecolor="#ffffff")
  outer = fig.add_gridspec(4, 1, height_ratios=[1.0, 1.0, 1.05, 1.18], hspace=0.30, top=0.925, bottom=0.055, left=0.045, right=0.975)

  # Row 0: 4 RoboTwin Bimanual DR Envs + Env #0 L-Wrist & R-Wrist (3-Cam Rig)
  gs0 = outer[0].subgridspec(1, 6, wspace=0.12)
  rt_titles = [
      "RoboTwin DR Env #0\n(Diag Overview, 256²)",
      "RoboTwin DR Env #1\n(Light + Wood/Grid + FOV)",
      "RoboTwin DR Env #2\n(Warm Key + Pose Jitter)",
      "RoboTwin DR Env #3\n(Cool Key + Checker Bench)",
  ]
  for c in range(4):
    ax = fig.add_subplot(gs0[0, c])
    ax.imshow(np.clip(ds["rt_diag_rgb"][c], 0.0, 1.0))
    ax.set_title(rt_titles[c], fontsize=9.5, fontweight="bold", pad=5)
    ax.axis("off")

  ax_lw = fig.add_subplot(gs0[0, 4])
  ax_lw.imshow(np.clip(ds["rt_lw_rgb"][0], 0.0, 1.0))
  ax_lw.set_title("RoboTwin Env #0\nLeft Wrist Cam (256²)", fontsize=9.5, fontweight="bold", pad=5)
  ax_lw.axis("off")

  ax_rw = fig.add_subplot(gs0[0, 5])
  ax_rw.imshow(np.clip(ds["rt_rw_rgb"][0], 0.0, 1.0))
  ax_rw.set_title("RoboTwin Env #0\nRight Wrist Cam (256²)", fontsize=9.5, fontweight="bold", pad=5)
  ax_rw.axis("off")

  # Row 1: 4 RoboDojo G=1,004 Moving Prims DR Envs + 2 Closeups (10 total DR renders shown!)
  gs1 = outer[1].subgridspec(1, 6, wspace=0.12)
  rd_panels = [
      (ds["rd_over_rgb"][0], "RoboDojo G=1,004 DR #0\n(Morton BVH + AgX, 256²)"),
      (ds["rd_over_rgb"][1], "RoboDojo G=1,004 DR #1\n(Light Az/El + Tex DR)"),
      (ds["rd_over_rgb"][2], "RoboDojo G=1,004 DR #2\n(Warm Tungsten + Albedo)"),
      (ds["rd_over_rgb"][3], "RoboDojo G=1,004 DR #3\n(Cool Daylight + FOV DR)"),
      (ds["rd_close_rgb"][0], "RoboDojo G=1,004 Closeup #0\n(Workbench Clutter, 256²)"),
      (ds["rd_close_rgb"][4], "RoboDojo G=1,004 Closeup #4\n(Distinct Seed + Shadows)"),
  ]
  for c, (img, ttl) in enumerate(rd_panels):
    ax = fig.add_subplot(gs1[0, c])
    ax.imshow(np.clip(img, 0.0, 1.0))
    ax.set_title(ttl, fontsize=9.5, fontweight="bold", pad=5)
    ax.axis("off")

  # Row 2: Synchronized 5-Modality TiledCamera Passes (RoboDojo G=1,004 Closeup, Single XLA Graph)
  gs2 = outer[2].subgridspec(1, 5, wspace=0.14)
  mod_panels = [
      (np.clip(ds["rd_close_rgb"][0], 0.0, 1.0), "1. RGB (GGX PBR + AgX)\nPer-Env DR Lights & Tex", None),
      (_normalize_depth(ds["rd_close_depth"]), "2. Distance to Image Plane\nExact Metric Z-Depth (m)", "inferno"),
      (np.clip(0.5 * (ds["rd_close_normals"] + 1.0), 0.0, 1.0), "3. World Normals\nExact Analytic Surface", None),
      (_colorize_ids(ds["rd_close_sem"], is_sem=True), "4. Semantic Segmentation\n5-Class In-VMEM Mask", None),
      (_colorize_flow(ds["rd_close_flow"]), "5. Motion Vectors\nZero-Gather 2D Optical Flow", None),
  ]
  for c, (img, ttl, cmap) in enumerate(mod_panels):
    ax = fig.add_subplot(gs2[0, c])
    if cmap is not None:
      ax.imshow(img, cmap=cmap, vmin=0.0, vmax=1.0)
    else:
      ax.imshow(img)
    ax.set_title(ttl, fontsize=9.8, fontweight="bold", pad=5)
    ax.axis("off")

  # Row 3: Two wide engineering charts with generous spacing
  gs3 = outer[3].subgridspec(1, 2, wspace=0.20)
  ax_fps = fig.add_subplot(gs3[0, 0])
  ax_ovh = fig.add_subplot(gs3[0, 1])

  rt_runs = [r for r in data.get("robotwin_3cam_runs", []) if r.get("status") == "ok" and r.get("with_dr")]
  rd_runs = [r for r in data.get("robodojo_cluttered_runs", []) if r.get("status") == "ok" and r.get("with_dr")]

  labels, fps_1c, fps_4c = [], [], []

  def _find(runs, name, res, shader, nc, batch_size):
    for r in runs:
      if (
          r["name"] == name
          and r["resolution"] == res
          and r["shader"] == shader
          and r["n_chips"] == nc
          and r["batch_size"] == batch_size
      ):
        return r
    return None

  pairs = [
      (rt_runs, "robotwin_3cam_5mod", "64x64", "fast_dr", 1024, "RoboTwin 3-Cam\n5-Mod B=1024"),
      (rt_runs, "robotwin_3cam_5mod", "64x64", "fast_dr", 2048, "RoboTwin 3-Cam\n5-Mod B=2048"),
      (rt_runs, "robotwin_3cam_6mod", "64x64", "fast_dr", 1024, "RoboTwin 3-Cam\n6-Mod (+Flow)"),
      (rd_runs, "robodojo_G50_6mod", "64x64", "fast_dr", 1024, "RoboDojo G=55\n6-Mod FastDR"),
      (rd_runs, "robodojo_G1000_6mod", "64x64", "pbr_ggx", 1024, "RoboDojo G=1,004\n6-Mod Morton DR"),
  ]
  for runs, nm, res, sh, bs, lbl in pairs:
    r1 = _find(runs, nm, res, sh, 1, bs)
    r4 = _find(runs, nm, res, sh, 4, bs)
    if r1 and r4:
      labels.append(lbl)
      fps_1c.append(r1["e2e_fps"])
      fps_4c.append(r4["e2e_fps"])

  x = np.arange(len(labels))
  w = 0.35
  ax_fps.bar(x - w / 2, fps_1c, width=w, color="#3b75af", label="1 Chip TPU v5p (MJX+DR+TiledCam)")
  b4 = ax_fps.bar(x + w / 2, fps_4c, width=w, color="#e67e22", label="4 Chips shard_map TPU v5p")
  for b, v1, v4 in zip(b4, fps_1c, fps_4c):
    ax_fps.text(
        b.get_x() + b.get_width() / 2, v4 * 1.08,
        f"{v4:,.0f}\n({v4/max(v1, 1e-9):.2f}x)",
        ha="center", va="bottom", fontsize=8.0, fontweight="bold",
    )
  ax_fps.set_yscale("log")
  ax_fps.set_ylim(500, max(fps_4c + [10000]) * 6.0)
  ax_fps.set_xticks(x)
  ax_fps.set_xticklabels(labels, fontsize=8.5, fontweight="bold")
  ax_fps.set_ylabel("E2E Env-Steps / s (FPS, log)", fontsize=9.8, fontweight="bold")
  ax_fps.set_title("Live TPU v5p Multi-Modal TiledCamera + Omniverse Replicator DR Throughput\n(All Modalities in One Zero-Copy XLA Graph: host_callbacks=0, gather_ops=0)", fontsize=10.5, fontweight="bold")
  ax_fps.grid(axis="y", which="both", alpha=0.3)
  ax_fps.legend(fontsize=8.5, loc="upper left", ncol=2, frameon=True)

  # Right chart: Static vs Live DR Overhead
  ovh_labels, ms_static, ms_dr, ovh_pcts = [], [], [], []
  for runs, nm, res, sh, bs, lbl in pairs:
    r4_dr = _find(runs, nm, res, sh, 4, bs)
    if r4_dr and "static_baseline_ms" in r4_dr:
      ovh_labels.append(lbl)
      ms_static.append(r4_dr["static_baseline_ms"])
      ms_dr.append(r4_dr["dt_e2e_ms"])
      ovh_pcts.append(r4_dr["dr_overhead_pct"])

  x2 = np.arange(len(ovh_labels))
  ax_ovh.bar(x2 - w / 2, ms_static, width=w, color="#6c757d", alpha=0.85, label="Static Lighting & Materials (4c)")
  b_dr = ax_ovh.bar(x2 + w / 2, ms_dr, width=w, color="#2ca02c", alpha=0.92, label="Full Omniverse Replicator DR (4c)")
  for b, s_ms, d_ms, pct in zip(b_dr, ms_static, ms_dr, ovh_pcts):
    sign = "+" if pct >= 0 else ""
    ax_ovh.text(
        b.get_x() + b.get_width() / 2, d_ms + max(ms_dr + [1.0]) * 0.025,
        f"{d_ms:.1f}ms\n({sign}{pct:.1f}%)",
        ha="center", va="bottom", fontsize=8.0, fontweight="bold",
    )
  ax_ovh.set_ylim(0, max(ms_dr + [10.0]) * 1.28)
  ax_ovh.set_xticks(x2)
  ax_ovh.set_xticklabels(ovh_labels, fontsize=8.5, fontweight="bold")
  ax_ovh.set_ylabel("4-Chip Batch Step + Render Latency (ms)", fontsize=9.8, fontweight="bold")
  ax_ovh.set_title("Omniverse Replicator DR Overhead vs Static Rendering (4-Chip TPU v5p)\n(Per-Env Light Az/El/Kelvin + Procedural Tex + Albedo/Roughness + FOV/Pose Jitter)", fontsize=10.5, fontweight="bold")
  ax_ovh.grid(axis="y", alpha=0.3)
  ax_ovh.legend(fontsize=8.5, loc="upper left", frameon=True)

  fig.suptitle(
      "V3 Wave 3 (Pillar 1B): Isaac Sim TiledCamera + Omniverse Replicator DR on Live Google Cloud TPU v5p",
      fontsize=14.2, fontweight="bold", y=0.975,
  )
  os.makedirs(os.path.dirname(args.out_png) or ".", exist_ok=True)
  plt.savefig(args.out_png, bbox_inches="tight")
  plt.close(fig)
  print("Saved figure:", args.out_png)


if __name__ == "__main__":
  main()
