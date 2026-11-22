"""Locked, immutable quality gate for Stage 1 Exact and Stage 2 Neural tracks.

Thresholds locked in Phase 0 (2026-11-22) and NEVER relaxed thereafter:
  Stage 1 Exact (vs official MuJoCo C mj_ray & reference):
    - Depth MAE: <= 1e-4 m (cartpole) / <= 2e-4 m (panda_pick_cube)
    - Surface normal mean angular error: <= 0.5 deg
    - GeomID segmentation match rate: >= 99.8%
    - Basic direct RGB PSNR >= 38.0 dB, |mean_lum_ratio - 1| <= 0.5%
    - Contact ROI edge error <= 1.0 px

  Stage 2 Neural (64 held-out unseen poses/seeds vs Cycles GT):
    - Full-image PSNR >= 30.0 dB AND >= +4.0 dB over Stage 1 non-GI baseline
    - Full-image SSIM >= 0.94, MS-SSIM >= 0.96
    - Global mean luminance shift: |mean(Y_pred) / mean(Y_ref) - 1| <= 0.5%
    - Contact ROI (finger pads + box dilated 4 px):
        * ROI PSNR >= 28.0 dB
        * Edge localization error <= 1.0 px
        * Foreground/geom occlusion mismatch <= 1.0%
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, Optional

import numpy as np

# Locked Phase 0 thresholds
STAGE1_THRESHOLDS = {
    "cartpole": {
        "depth_mae_max": 1.0e-4,
        "normal_deg_max": 0.5,
        "geom_match_min": 0.998,
        "rgb_psnr_min": 38.0,
        "lum_rel_max": 0.005,
        "roi_edge_px_max": 1.0,
    },
    "panda_pick_cube": {
        "depth_mae_max": 2.0e-4,
        "normal_deg_max": 0.5,
        "geom_match_min": 0.998,
        "rgb_psnr_min": 38.0,
        "lum_rel_max": 0.005,
        "roi_edge_px_max": 1.0,
    },
}

STAGE2_THRESHOLDS = {
    "psnr_min": 30.0,
    "psnr_gain_min": 4.0,
    "ssim_min": 0.94,
    "msssim_min": 0.96,
    "lum_rel_max": 0.005,
    "roi_psnr_min": 28.0,
    "roi_edge_px_max": 1.0,
    "roi_occ_mismatch_max": 0.01,
}

REC709_W = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def luminance(rgb: np.ndarray) -> np.ndarray:
  return rgb @ REC709_W


def compute_psnr(pred: np.ndarray, ref: np.ndarray) -> float:
  mse = float(np.mean((pred.astype(np.float64) - ref.astype(np.float64)) ** 2))
  return float(10.0 * np.log10(1.0 / max(mse, 1.0e-12)))


def _gaussian_window_1d(size: int = 7, sigma: float = 1.5) -> np.ndarray:
  x = np.arange(size, dtype=np.float64) - size // 2
  w = np.exp(-0.5 * (x / sigma) ** 2)
  return w / w.sum()


def _box_blur2d(img: np.ndarray, win: np.ndarray) -> np.ndarray:
  # img: (..., H, W)
  pad = len(win) // 2
  out = np.zeros_like(img, dtype=np.float64)
  padded = np.pad(img.astype(np.float64), [(0, 0)] * (img.ndim - 2) + [(pad, pad), (pad, pad)], mode="reflect")
  for i, wi in enumerate(win):
    for j, wj in enumerate(win):
      out += (wi * wj) * padded[..., i : i + img.shape[-2], j : j + img.shape[-1]]
  return out


def compute_ssim_batch(pred: np.ndarray, ref: np.ndarray, win_size: int = 7) -> float:
  """Compute mean SSIM across (N, H, W, 3) images in [0, 1]."""
  c1 = 0.01 ** 2
  c2 = 0.03 ** 2
  win = _gaussian_window_1d(win_size, 1.5)
  vals = []
  for ch in range(3):
    x = pred[..., ch]
    y = ref[..., ch]
    mu_x = _box_blur2d(x, win)
    mu_y = _box_blur2d(y, win)
    sig_x = _box_blur2d(x * x, win) - mu_x * mu_x
    sig_y = _box_blur2d(y * y, win) - mu_y * mu_y
    sig_xy = _box_blur2d(x * y, win) - mu_x * mu_y
    num = (2.0 * mu_x * mu_y + c1) * (2.0 * sig_xy + c2)
    den = (mu_x * mu_x + mu_y * mu_y + c1) * (np.maximum(sig_x, 0.0) + np.maximum(sig_y, 0.0) + c2)
    vals.append(np.mean(num / np.maximum(den, 1e-12)))
  return float(np.mean(vals))


def compute_msssim_batch(pred: np.ndarray, ref: np.ndarray) -> float:
  """3-scale MS-SSIM suited for 64x64 and 128x128 robotics images."""
  weights = [0.4, 0.35, 0.25]
  x = pred.astype(np.float64)
  y = ref.astype(np.float64)
  scores = []
  for w in weights:
    win = 5 if x.shape[1] >= 16 else 3
    scores.append(max(compute_ssim_batch(x, y, win_size=win), 1e-6) ** w)
    if x.shape[1] >= 32 and x.shape[2] >= 32:
      x = 0.25 * (x[:, 0::2, 0::2] + x[:, 0::2, 1::2] + x[:, 1::2, 0::2] + x[:, 1::2, 1::2])
      y = 0.25 * (y[:, 0::2, 0::2] + y[:, 0::2, 1::2] + y[:, 1::2, 0::2] + y[:, 1::2, 1::2])
  return float(np.prod(scores))


def dilate_mask(mask: np.ndarray, radius: int = 4) -> np.ndarray:
  """Binary morphological dilation on (N, H, W) mask."""
  out = mask.copy()
  N, H, W = mask.shape
  padded = np.pad(mask, ((0, 0), (radius, radius), (radius, radius)), mode="constant", constant_values=False)
  for dy in range(-radius, radius + 1):
    for dx in range(-radius, radius + 1):
      if dy * dy + dx * dx <= radius * radius:
        out |= padded[:, radius + dy : radius + dy + H, radius + dx : radius + dx + W]
  return out


def sobel_edges(img_lum: np.ndarray) -> np.ndarray:
  """Gradient magnitude on (N, H, W)."""
  gx = np.zeros_like(img_lum)
  gy = np.zeros_like(img_lum)
  gx[:, :, 1:-1] = 0.5 * (img_lum[:, :, 2:] - img_lum[:, :, :-2])
  gy[:, 1:-1, :] = 0.5 * (img_lum[:, 2:, :] - img_lum[:, :-2, :])
  return np.sqrt(gx * gx + gy * gy)


def edge_localization_error_px(pred_rgb: np.ndarray, ref_rgb: np.ndarray, roi_mask: np.ndarray) -> float:
  """Mean nearest-edge distance (in pixels) between strong Sobel edges inside ROI."""
  ep = sobel_edges(luminance(pred_rgb))
  er = sobel_edges(luminance(ref_rgb))
  dists = []
  for i in range(pred_rgb.shape[0]):
    m = roi_mask[i]
    if not np.any(m):
      continue
    thr_r = max(float(np.percentile(er[i][m], 80)), 0.04)
    thr_p = max(float(np.percentile(ep[i][m], 80)), 0.04)
    pr, pc = np.where(m & (er[i] >= thr_r))
    pp, pq = np.where(m & (ep[i] >= thr_p))
    if len(pr) == 0 or len(pp) == 0:
      continue
    pts_r = np.stack([pr, pc], axis=-1)[:, None, :].astype(np.float32)
    pts_p = np.stack([pp, pq], axis=-1)[None, :, :].astype(np.float32)
    dmat = np.sqrt(np.sum((pts_r - pts_p) ** 2, axis=-1))
    dists.append(float(np.mean(np.min(dmat, axis=1))))
  return float(np.mean(dists)) if dists else 0.0


def evaluate_stage1(
    scene_name: str,
    depth_pred: np.ndarray,
    depth_ref: np.ndarray,
    normal_pred: np.ndarray,
    normal_ref: np.ndarray,
    geom_pred: np.ndarray,
    geom_ref: np.ndarray,
    rgb_pred: Optional[np.ndarray] = None,
    rgb_ref: Optional[np.ndarray] = None,
    roi_mask: Optional[np.ndarray] = None,
) -> Dict[str, object]:
  """Evaluate Stage 1 Exact metrics against locked thresholds."""
  th = STAGE1_THRESHOLDS[scene_name]
  valid = (geom_ref >= 0) & (geom_pred >= 0)
  depth_mae = float(np.mean(np.abs(depth_pred[valid] - depth_ref[valid]))) if np.any(valid) else 999.0

  n_dot = np.sum(normal_pred[valid] * normal_ref[valid], axis=-1) if np.any(valid) else np.array([1.0])
  n_dot = np.clip(n_dot, -1.0, 1.0)
  normal_deg = float(np.mean(np.degrees(np.arccos(n_dot))))
  geom_match = float(np.mean(geom_pred == geom_ref))

  passed = (
      depth_mae <= th["depth_mae_max"]
      and normal_deg <= th["normal_deg_max"]
      and geom_match >= th["geom_match_min"]
  )
  out: Dict[str, object] = {
      "scene": scene_name,
      "depth_mae_m": depth_mae,
      "normal_deg": normal_deg,
      "geom_match": geom_match,
  }
  if rgb_pred is not None and rgb_ref is not None:
    psnr = compute_psnr(rgb_pred, rgb_ref)
    lum_rel = float(np.mean(luminance(rgb_pred)) / max(float(np.mean(luminance(rgb_ref))), 1e-6) - 1.0)
    passed = passed and (psnr >= th["rgb_psnr_min"]) and (abs(lum_rel) <= th["lum_rel_max"])
    out["rgb_psnr_db"] = psnr
    out["lum_rel"] = lum_rel
    if roi_mask is not None:
      edge_px = edge_localization_error_px(rgb_pred, rgb_ref, roi_mask)
      passed = passed and (edge_px <= th["roi_edge_px_max"])
      out["roi_edge_px"] = edge_px

  out["passed"] = bool(passed)
  out["thresholds"] = th
  return out


def evaluate_stage2_neural(
    rgb_pred: np.ndarray,
    rgb_ref: np.ndarray,
    rgb_stage1_baseline: np.ndarray,
    roi_mask: np.ndarray,
    geom_id: np.ndarray,
    box_geom_id: int,
    finger_geom_ids: tuple[int, ...],
) -> Dict[str, object]:
  """Evaluate Stage 2 Neural shader on 64 held-out frames against locked gate."""
  th = STAGE2_THRESHOLDS
  psnr_pred = compute_psnr(rgb_pred, rgb_ref)
  psnr_base = compute_psnr(rgb_stage1_baseline, rgb_ref)
  psnr_gain = psnr_pred - psnr_base
  ssim = compute_ssim_batch(rgb_pred, rgb_ref)
  msssim = compute_msssim_batch(rgb_pred, rgb_ref)

  y_pred = float(np.mean(luminance(rgb_pred)))
  y_ref = float(np.mean(luminance(rgb_ref)))
  lum_rel = y_pred / max(y_ref, 1.0e-6) - 1.0

  dilated = dilate_mask(roi_mask, radius=4)
  roi_psnr = compute_psnr(rgb_pred[dilated], rgb_ref[dilated]) if np.any(dilated) else 0.0
  roi_edge_px = edge_localization_error_px(rgb_pred, rgb_ref, dilated)

  # Occlusion mismatch check: compare predicted vs reference luminance discontinuity
  # across box vs finger boundaries (where exact Stage 1 GeomID separates box & fingers).
  box_mask = geom_id == box_geom_id
  finger_mask = np.isin(geom_id, list(finger_geom_ids))
  boundary_mask = dilate_mask(box_mask, radius=1) & dilate_mask(finger_mask, radius=1)
  if np.any(boundary_mask):
    err_b = np.abs(luminance(rgb_pred)[boundary_mask] - luminance(rgb_ref)[boundary_mask])
    occ_mismatch = float(np.mean(err_b > 0.12))
  else:
    occ_mismatch = 0.0

  passed = (
      psnr_pred >= th["psnr_min"]
      and psnr_gain >= th["psnr_gain_min"]
      and ssim >= th["ssim_min"]
      and msssim >= th["msssim_min"]
      and abs(lum_rel) <= th["lum_rel_max"]
      and roi_psnr >= th["roi_psnr_min"]
      and roi_edge_px <= th["roi_edge_px_max"]
      and occ_mismatch <= th["roi_occ_mismatch_max"]
  )

  return {
      "psnr_db": psnr_pred,
      "stage1_psnr_db": psnr_base,
      "psnr_gain_db": psnr_gain,
      "ssim": ssim,
      "msssim": msssim,
      "lum_rel": lum_rel,
      "roi_psnr_db": roi_psnr,
      "roi_edge_px": roi_edge_px,
      "roi_occ_mismatch": occ_mismatch,
      "passed": bool(passed),
      "thresholds": th,
  }


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--stage", required=True, choices=["exact", "neural"])
  ap.add_argument("--npz", required=True, help="Path to saved evaluation arrays")
  ap.add_argument("--scene", default="panda_pick_cube")
  ap.add_argument("--out_json", default=None)
  args = ap.parse_args()

  d = np.load(args.npz)
  if args.stage == "exact":
    res = evaluate_stage1(
        scene_name=args.scene,
        depth_pred=d["depth_pred"],
        depth_ref=d["depth_ref"],
        normal_pred=d["normal_pred"],
        normal_ref=d["normal_ref"],
        geom_pred=d["geom_pred"],
        geom_ref=d["geom_ref"],
        rgb_pred=d["rgb_pred"] if "rgb_pred" in d else None,
        rgb_ref=d["rgb_ref"] if "rgb_ref" in d else None,
        roi_mask=d["roi_mask"] if "roi_mask" in d else None,
    )
  else:
    res = evaluate_stage2_neural(
        rgb_pred=d["rgb_pred"],
        rgb_ref=d["rgb_ref"],
        rgb_stage1_baseline=d["rgb_stage1"],
        roi_mask=d["roi_mask"],
        geom_id=d["geom_id"],
        box_geom_id=int(d["box_geom_id"]),
        finger_geom_ids=tuple(int(x) for x in d["finger_geom_ids"]),
    )

  print(json.dumps(res, indent=2))
  if args.out_json:
    with open(args.out_json, "w") as f:
      json.dump(res, f, indent=2)
  sys.exit(0 if res["passed"] else 1)


if __name__ == "__main__":
  main()
