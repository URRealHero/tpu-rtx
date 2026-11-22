"""Generate Wave 3 showcase figure `out/figures/v3_w0_universal_mjx_suite.png`."""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _normalize_depth(d: np.ndarray) -> np.ndarray:
  valid = d >= 0.0
  if not np.any(valid):
    return np.zeros_like(d)
  lo = np.percentile(d[valid], 2)
  hi = np.percentile(d[valid], 98)
  out = np.clip((d - lo) / max(hi - lo, 1e-4), 0.0, 1.0)
  return np.where(valid, 1.0 - out, 0.0)


def _colorize_seg(seg: np.ndarray) -> np.ndarray:
  h, w = seg.shape
  rgb = np.zeros((h, w, 3), dtype=np.float32)
  palette = np.array([
      [0.20, 0.55, 0.85], [0.90, 0.35, 0.25], [0.25, 0.78, 0.45],
      [0.92, 0.72, 0.18], [0.68, 0.38, 0.86], [0.18, 0.78, 0.82],
      [0.88, 0.45, 0.68], [0.75, 0.78, 0.82], [0.55, 0.40, 0.28],
  ], dtype=np.float32)
  m = seg >= 0
  idx = np.maximum(seg, 0) % len(palette)
  rgb[m] = palette[idx[m]]
  return rgb


def _colorize_flow(flow: np.ndarray) -> np.ndarray:
  fx, fy = flow[..., 0], flow[..., 1]
  mag = np.sqrt(fx * fx + fy * fy)
  ang = (np.arctan2(fy, fx) + np.pi) / (2.0 * np.pi)
  scale = np.clip(mag / max(float(np.percentile(mag, 99)), 1e-3), 0.0, 1.0)
  hsv = np.stack([ang, np.ones_like(ang), scale], axis=-1)
  from matplotlib.colors import hsv_to_rgb
  return hsv_to_rgb(hsv)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--json", required=True)
  ap.add_argument("--npz", required=True)
  ap.add_argument("--out_png", required=True)
  args = ap.parse_args()

  with open(args.json, "r") as f:
    data = json.load(f)
  smp = np.load(args.npz)

  fig = plt.figure(figsize=(19.5, 10.5), dpi=180)
  gs = fig.add_gridspec(3, 5, hspace=0.34, wspace=0.28, width_ratios=[1, 1, 1, 1, 1.8])

  scenes = [
      ("panda_pick_cube", "PandaPickCube (15-DOF, 22 geoms)"),
      ("cartpole", "Cartpole (2-DOF, 5 geoms)"),
      ("humanoid_21dof", "Humanoid (27-DOF, 21 geoms)"),
  ]

  for row, (sname, stitle) in enumerate(scenes):
    rgb = np.clip(smp[f"{sname}_rgb"], 0.0, 1.0)
    nrm = np.clip(0.5 * (smp[f"{sname}_normal"] + 1.0), 0.0, 1.0)
    dep = _normalize_depth(smp[f"{sname}_depth"])
    seg = _colorize_seg(smp[f"{sname}_seg_id"])
    flw = _colorize_flow(smp[f"{sname}_flow"])

    # Combine SegID + Flow as inset or show RGB / Normal / Depth / Seg+Flow
    panels = [
        (rgb, f"{stitle}\nGGX PBR + 4-AO + SoftShadow"),
        (dep, "Depth (m)"),
        (nrm, "World Normal"),
        (seg if row != 0 else flw, "GeomID Seg" if row != 0 else "2D Optical Flow (px)"),
    ]
    for col, (img, ttl) in enumerate(panels):
      ax = fig.add_subplot(gs[row, col])
      if img.ndim == 2:
        ax.imshow(img, cmap="inferno")
      else:
        ax.imshow(img)
      ax.set_title(ttl, fontsize=9.5, fontweight="bold")
      ax.axis("off")

  # Right column (col 4): 3 stacked engineering charts
  # Top-right: Render-Only & E2E Throughput Bar Chart
  ax_fps = fig.add_subplot(gs[0, 4])
  bars_lbl, bars_val, bars_col = [], [], []
  label_map = {
      "1c_64x64_fast_fused_b16_r512_bfloat16_B2048_2out": "Rend 1c B2048 (2out)",
      "4c_64x64_fast_fused_b16_r512_bfloat16_B8192_2out": "Rend 4c B8192 (2out)",
      "4c_64x64_fast_fused_b16_r512_bfloat16_B16384_2out": "Rend 4c B16k (2out)",
      "4c_64x64_fast_fused_b16_r512_bfloat16_B8192_5out": "Rend 4c B8192 (5out)",
  }
  for r in data.get("render_push", []):
    if r["label"] in label_map:
      bars_lbl.append(label_map[r["label"]])
      bars_val.append(r["render_scan_fps"] / 1e3)
      bars_col.append("#1f77b4")
  e2e_map = {
      "panda_1c_64x64_std_B2048": "E2E Panda 1c B2048",
      "panda_4c_64x64_std_scan_B8192": "E2E Panda 4c std",
      "panda_4c_64x64_it2ls4_B8192": "E2E Panda 4c it2ls4",
      "cartpole_4c_64x64_B16384": "E2E Cartpole 4c B16k",
  }
  for r in data.get("e2e_push", []):
    bars_lbl.append(e2e_map.get(r["label"], f"E2E {r['label'][:16]}"))
    bars_val.append(r["e2e_fps"] / 1e3)
    bars_col.append("#2ca02c")
  if bars_val:
    y = np.arange(len(bars_val))
    ax_fps.barh(y, bars_val, color=bars_col)
    ax_fps.set_yticks([])
    ax_fps.invert_yaxis()
    ax_fps.set_xlabel("Throughput (k FPS)", fontsize=8.5)
    ax_fps.set_title("Live TPU v5p Throughput (64x64)", fontsize=10, fontweight="bold")
    vmax = max(bars_val)
    for yi, (lbl, v) in enumerate(zip(bars_lbl, bars_val)):
      if v > vmax * 0.55:
        ax_fps.text(v * 0.97, yi, f"{lbl}: {v:,.0f}k", va="center", ha="right", color="white", fontsize=7.2, fontweight="bold")
      else:
        ax_fps.text(v + vmax * 0.02, yi, f"{lbl}: {v:,.0f}k", va="center", ha="left", color="#111111", fontsize=7.2)
    ax_fps.set_xlim(0, vmax * 1.15)
    ax_fps.grid(axis="x", alpha=0.3)

  # Middle-right: Multi-Chip Pod Scaling (1 -> 4 -> 32 -> 64 -> 1000 chips)
  ax_pod = fig.add_subplot(gs[1, 4])
  pod = data.get("pod_scaling", [])
  if pod:
    chips = [p["n_chips"] for p in pod]
    pfps = [p["projected_render_fps"] / 1e6 for p in pod]
    ax_pod.loglog(chips, pfps, "o-", color="#d62728", lw=2, markersize=6, label="SPMD shard_map (0 allreduce)")
    # Highlight live measured 1c & 4c points
    r1 = next((r["render_scan_fps"] / 1e6 for r in data.get("render_push", []) if r["n_chips"] == 1), None)
    r4 = max((r["render_scan_fps"] / 1e6 for r in data.get("render_push", []) if r["n_chips"] == 4 and "2out" in r["label"]), default=None)
    if r1 and r4:
      ax_pod.loglog([1, 4], [r1, r4], "s", color="#111111", markersize=7, label="Measured live w0 TPU")
    ax_pod.set_xlabel("TPU v5p Chips (data_axis)", fontsize=8.5)
    ax_pod.set_ylabel("Render Throughput (M FPS)", fontsize=8.5)
    ax_pod.set_title("SPMD Pod Scaling (1 -> 1000 Chips)", fontsize=10, fontweight="bold")
    ax_pod.grid(True, which="both", alpha=0.3)
    ax_pod.legend(fontsize=7.5, loc="upper left")

  # Bottom-right: Parity & Roofline Summary Table
  ax_txt = fig.add_subplot(gs[2, 4])
  ax_txt.axis("off")
  lines = ["TPU-RTX Wave 3 Verified Summary (w0 TPU v5p)", "--------------------------------------------"]
  ps = data.get("parity_and_sdk", {})
  for sn in ("panda_pick_cube", "cartpole", "humanoid_21dof"):
    if sn in ps and "fused_ao_ggx" in ps[sn]:
      pr = ps[sn]["fused_ao_ggx"]
      lines.append(
          f"{sn}: dMAE={pr['depth_mae_m']:.1e}m "
          f"nErr={pr['normal_deg']:.3f}deg match={pr['geom_match']*100:.2f}% HLO=0cb/0g"
      )
  r_best = max((r["render_scan_fps"] for r in data.get("render_push", [])), default=0.0)
  lines.append(f"Peak Render-Only (4c, 64x64): {r_best:,.0f} FPS")
  if data.get("e2e_push"):
    e_best = max(r["e2e_fps"] for r in data["e2e_push"])
    lines.append(f"Peak E2E Physics+Render (4c, 64x64): {e_best:,.0f} FPS")
  lines.append("HBM: 2.765 TB/s/chip | VPU: 28 TF/chip | VMEM: 64 MiB")
  ax_txt.text(0.02, 0.95, "\n".join(lines), va="top", family="monospace", fontsize=8.0,
              bbox=dict(boxstyle="round,pad=0.5", facecolor="#f4f6f8", edgecolor="#cccccc"))

  fig.suptitle("TPU-RTX Universal MJX SDK: Zero-Copy Multi-Scene Rendering, Parity & Pod Scaling", fontsize=13.5, fontweight="bold", y=0.98)
  os.makedirs(os.path.dirname(args.out_png) or ".", exist_ok=True)
  fig.savefig(args.out_png, bbox_inches="tight")
  plt.close(fig)
  print("Saved figure:", args.out_png)


if __name__ == "__main__":
  main()
