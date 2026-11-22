"""Wave 5: Interactive Time-of-Day/Material Neural Relighting & TPU v5p MXU Quantization Pareto.

End-to-end pipeline on w4 (4x TPU v5p chips):
  1. Render multi-state 4-bounce MC GT + 16-ch primary G-buffer on `classroom` & `cornell_workshop`.
  2. Train FiLM-conditioned SpaceToDepth(2) MXU Neural Relighter in BF16.
  3. Calibrate INT8 W8A8 PTQ scales & run QAT fine-tuning (Straight-Through Estimator).
  4. Evaluate true zero-raytrace interactive Time-of-Day (`Morning -> Noon -> Golden Sunset -> Blue Twilight`)
     and Material (`Matte Clay -> Satin -> Brushed Gold -> Chrome`) sweeps against 4-bounce MC GT.
  5. Measure precision Pareto (PSNR / SSIM / MS-SSIM) AND 1-chip / 4-chip 1080p & Native 4K FPS
     across `FP32` (Precision.HIGHEST), `BF16`, and `INT8 W8A8` (`conv_general_dilated` + `dot_general`)
     with strict zero-copy HLO audits (`host_callbacks == 0`, `gather_ops == 0`, `send_recv_ops == 0`).
  6. Export `v3_w4_wave5_relight_quant.json` and dark-cinema `v3_w4_interactive_relight_quant.png`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import time
from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from PIL import Image, ImageDraw, ImageFont

from gate import compute_msssim_batch, compute_psnr, compute_ssim_batch, luminance
from relight_models import (
    benchmark_precision_pareto,
    calibrate_act_scales,
    init_relighter_params,
    quantize_params_static_int8,
    relight_forward_float_or_qat,
    relight_forward_int8,
)
from scene_relight_data import (
    MATERIAL_PRESETS,
    TOD_PRESETS,
    generate_relight_dataset,
)


# ==============================================================================
# 1. Loss, BF16 Training & INT8 QAT Fine-Tuning
# ==============================================================================

def _ssim_loss(pred: jax.Array, target: jax.Array) -> jax.Array:
  c1, c2 = 0.01 ** 2, 0.03 ** 2
  mu_x = jax.lax.reduce_window(pred, 0.0, jax.lax.add, (1, 5, 5, 1), (1, 1, 1, 1), "SAME") / 25.0
  mu_y = jax.lax.reduce_window(target, 0.0, jax.lax.add, (1, 5, 5, 1), (1, 1, 1, 1), "SAME") / 25.0
  sig_x = jax.lax.reduce_window(pred * pred, 0.0, jax.lax.add, (1, 5, 5, 1), (1, 1, 1, 1), "SAME") / 25.0 - mu_x * mu_x
  sig_y = jax.lax.reduce_window(target * target, 0.0, jax.lax.add, (1, 5, 5, 1), (1, 1, 1, 1), "SAME") / 25.0 - mu_y * mu_y
  sig_xy = jax.lax.reduce_window(pred * target, 0.0, jax.lax.add, (1, 5, 5, 1), (1, 1, 1, 1), "SAME") / 25.0 - mu_x * mu_y
  num = (2.0 * mu_x * mu_y + c1) * (2.0 * sig_xy + c2)
  den = (mu_x * mu_x + mu_y * mu_y + c1) * (jnp.maximum(sig_x, 0.0) + jnp.maximum(sig_y, 0.0) + c2)
  return 1.0 - jnp.mean(num / jnp.maximum(den, 1e-12))


def _relight_loss(pred: jax.Array, yb: jax.Array) -> jax.Array:
  l2 = jnp.mean((pred - yb) ** 2)
  l1 = jnp.mean(jnp.abs(pred - yb))
  ls = _ssim_loss(pred, yb)
  gx_p = pred[:, :, 1:, :] - pred[:, :, :-1, :]
  gx_y = yb[:, :, 1:, :] - yb[:, :, :-1, :]
  gy_p = pred[:, 1:, :, :] - pred[:, :-1, :, :]
  gy_y = yb[:, 1:, :, :] - yb[:, :-1, :, :]
  lgrad = jnp.mean(jnp.abs(gx_p - gx_y)) + jnp.mean(jnp.abs(gy_p - gy_y))
  lum_p = 0.2126 * pred[..., 0] + 0.7152 * pred[..., 1] + 0.0722 * pred[..., 2]
  lum_y = 0.2126 * yb[..., 0] + 0.7152 * yb[..., 1] + 0.0722 * yb[..., 2]
  llum = jnp.abs(jnp.mean(lum_p) - jnp.mean(lum_y))
  return 0.45 * l2 + 0.30 * l1 + 0.15 * ls + 0.10 * lgrad + 0.04 * llum


def train_relighter(
    tr_gb: np.ndarray,
    tr_gt: np.ndarray,
    tr_cond: np.ndarray,
    epochs: int = 320,
    batch_size: int = 16,
    patch_size: int = 160,
    lr: float = 2.0e-3,
    init_params: Any = None,
    qat: bool = False,
    seed_offset: int = 0,
) -> Tuple[Any, int, float]:
  dev = jax.devices()[0]
  gb_dev = jax.device_put(tr_gb.astype(np.float32), dev)
  y_dev = jax.device_put(tr_gt.astype(np.float32), dev)
  c_dev = jax.device_put(tr_cond.astype(np.float32), dev)
  N, H, W, _ = tr_gb.shape

  params = init_relighter_params(seed=2026) if init_params is None else jax.tree_util.tree_map(jnp.asarray, init_params)
  params = jax.device_put(params, dev)
  ema_params = params
  n_params = int(sum(p.size for p in jax.tree_util.tree_leaves(params)))

  steps_per_ep = 16
  total_steps = epochs * steps_per_ep
  sched = optax.warmup_cosine_decay_schedule(lr * 0.1, lr, max(16, total_steps // 14), total_steps, lr * 0.02)
  tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=2e-5))
  opt_state = tx.init(params)

  @jax.jit
  def train_ep(p, ema_p, ost, ep_rng):
    def _step(carry, s_idx):
      cp, cema, cost = carry
      k = jax.random.fold_in(ep_rng, s_idx)
      k_idx, k_y, k_x = jax.random.split(k, 3)
      b_idx = jax.random.randint(k_idx, (batch_size,), 0, N)
      y0 = jax.random.randint(k_y, (), 0, H - patch_size + 1)
      x0 = jax.random.randint(k_x, (), 0, W - patch_size + 1)
      gbb = jax.lax.dynamic_slice(gb_dev[b_idx], (0, y0, x0, 0), (batch_size, patch_size, patch_size, 16))
      yb = jax.lax.dynamic_slice(y_dev[b_idx], (0, y0, x0, 0), (batch_size, patch_size, patch_size, 3))
      cb = c_dev[b_idx]

      def loss_fn(pr):
        pm = "fp32" if qat else "bf16"
        pred = relight_forward_float_or_qat(pr, gbb, cb, precision_mode=pm, qat=qat)
        return _relight_loss(pred, yb)

      loss, grads = jax.value_and_grad(loss_fn)(cp)
      # Keep calibrated act_scales stable during both BF16 and QAT
      grads["act_scales"] = jax.tree_util.tree_map(jnp.zeros_like, grads["act_scales"])
      upd, n_ost = tx.update(grads, cost, cp)
      np_p = optax.apply_updates(cp, upd)
      n_ema = jax.tree_util.tree_map(lambda e, n: 0.994 * e + 0.006 * n, cema, np_p)
      return (np_p, n_ema, n_ost), loss

    (p_o, e_o, o_o), losses = jax.lax.scan(_step, (p, ema_p, ost), jnp.arange(steps_per_ep))
    return p_o, e_o, o_o, jnp.mean(losses)

  tag = "QAT-INT8" if qat else "BF16-RELIGHT"
  t0 = time.time()
  for ep in range(epochs):
    params, ema_params, opt_state, ep_loss = train_ep(params, ema_params, opt_state, jax.random.PRNGKey(ep + 5000 + seed_offset))
    if (ep + 1) % max(20, epochs // 5) == 0 or ep + 1 == epochs:
      jax.block_until_ready(ep_loss)
      print(f"[{tag}] ep {ep+1}/{epochs} loss={float(ep_loss):.5f} ({time.time()-t0:.1f}s)", flush=True)
  return ema_params, n_params, round(time.time() - t0, 2)


# ==============================================================================
# 2. Evaluation Helpers & Visualization
# ==============================================================================

def _compute_metrics(pred: np.ndarray, ref: np.ndarray) -> Dict[str, float]:
  psnr = compute_psnr(pred, ref)
  ssim = compute_ssim_batch(pred, ref)
  msssim = compute_msssim_batch(pred, ref)
  lum_rel = float(np.mean(luminance(pred)) / max(float(np.mean(luminance(ref))), 1e-6) - 1.0)
  return {
      "psnr_db": round(psnr, 3),
      "ssim": round(ssim, 4),
      "msssim": round(msssim, 4),
      "lum_bias_pct": round(abs(lum_rel) * 100.0, 4),
  }


def _load_font(size: int, bold: bool = False):
  candidates = [
      "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
      "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
  ]
  for p in candidates:
    if os.path.exists(p):
      try:
        return ImageFont.truetype(p, size)
      except Exception:
        pass
  return ImageFont.load_default()


def make_showcase_figure(
    cls_tod_preds: np.ndarray,   # (4, H, W, 3)
    cls_tod_gts: np.ndarray,     # (4, H, W, 3)
    cls_tod_metrics: List[Dict[str, Any]],
    crn_mat_preds: np.ndarray,   # (4, H, W, 3)
    crn_mat_gts: np.ndarray,     # (4, H, W, 3)
    crn_mat_metrics: List[Dict[str, Any]],
    pareto_summary: Dict[str, Any],
    bench_recs: List[Dict[str, Any]],
    out_png: str,
) -> None:
  """Render crisp 3-row dark-cinema showcase: ToD Sweep + Material Sweep + Precision Pareto."""
  tile = 460
  pad = 16
  hdr_top = 74
  row_lbl_h = 34
  sub_h = 46
  chart_h = 360

  canvas_w = 4 * tile + 5 * pad
  canvas_h = hdr_top + (row_lbl_h + tile + sub_h + pad) * 2 + (row_lbl_h + chart_h + pad) + 12

  bg_col = (13, 16, 22)
  card_col = (20, 24, 34)
  border_col = (46, 54, 72)
  txt_main = (240, 244, 250)
  txt_muted = (165, 176, 196)
  accent_cyan = (88, 204, 245)
  accent_gold = (248, 196, 86)

  canvas = Image.new("RGB", (canvas_w, canvas_h), bg_col)
  draw = ImageDraw.Draw(canvas)

  f_title = _load_font(24, bold=True)
  f_sub = _load_font(15, bold=False)
  f_sec = _load_font(18, bold=True)
  f_cap = _load_font(14, bold=True)
  f_cap_s = _load_font(13, bold=False)

  # Top cinema header
  draw.rectangle([0, 0, canvas_w, hdr_top - 6], fill=(18, 22, 31))
  draw.line([0, hdr_top - 6, canvas_w, hdr_top - 6], fill=border_col, width=2)
  draw.text(
      (pad, 12),
      "TPU v5p Zero-Copy Neural Relighting & MXU Quantization Pareto (FP32 vs BF16 vs INT8 W8A8)",
      fill=txt_main,
      font=f_title,
  )
  bf16_4k = next(r for r in bench_recs if r["precision"] == "bf16" and r["resolution"] == "3840x2160" and r["n_chips"] == 4)
  int8_4k = next(r for r in bench_recs if r["precision"] == "int8" and r["resolution"] == "3840x2160" and r["n_chips"] == 4)
  fp32_4k = next(r for r in bench_recs if r["precision"] == "fp32" and r["resolution"] == "3840x2160" and r["n_chips"] == 4)
  draw.text(
      (pad, 43),
      f"Single Static G-Buffer Interactive Relighting (0 Raytrace Ops/Slider)  |  Native 4K 4-Chip: "
      f"FP32 {fp32_4k['fps']:.1f} FPS -> BF16 {bf16_4k['fps']:.1f} FPS -> INT8 {int8_4k['fps']:.1f} FPS  |  host_callbacks=0, gather_ops=0",
      fill=accent_cyan,
      font=f_sub,
  )

  # Helper to paste 4-panel image row with corner GT inset
  def _draw_panel_row(y_start: int, sec_title: str, sec_color, preds: np.ndarray, gts: np.ndarray, infos: List[Dict[str, Any]], mode: str):
    draw.text((pad, y_start + 4), sec_title, fill=sec_color, font=f_sec)
    y_img = y_start + row_lbl_h
    for i in range(4):
      x0 = pad + i * (tile + pad)
      draw.rectangle([x0 - 1, y_img - 1, x0 + tile, y_img + tile + sub_h], fill=card_col, outline=border_col, width=1)
      im_p = Image.fromarray(np.clip(preds[i] * 255.0 + 0.5, 0, 255).astype(np.uint8)).resize((tile, tile), Image.Resampling.LANCZOS)
      # Small crisp 116x116 4-bounce MC GT inset in top-right corner
      inset_sz = 116
      im_g = Image.fromarray(np.clip(gts[i] * 255.0 + 0.5, 0, 255).astype(np.uint8)).resize((inset_sz, inset_sz), Image.Resampling.LANCZOS)
      im_p.paste(im_g, (tile - inset_sz - 8, 8))
      idraw = ImageDraw.Draw(im_p)
      idraw.rectangle([tile - inset_sz - 9, 7, tile - 7, 8 + inset_sz], outline=(220, 230, 245), width=2)
      idraw.rectangle([tile - inset_sz - 8, 8, tile - inset_sz + 52, 24], fill=(15, 18, 25))
      idraw.text((tile - inset_sz - 5, 9), "MC GT", fill=accent_gold, font=f_cap_s)
      canvas.paste(im_p, (x0, y_img))

      m = infos[i]
      if mode == "tod":
        l1 = f"{m['name']}  (az={m['az_deg']:.0f}°, el={m['el_deg']:.0f}°, {int(m['temp_k'])}K)"
      else:
        l1 = f"{m['name']}  (rough={m['roughness']:.2f}, metal={m['metallic']:.2f})"
      l2 = f"PSNR {m['psnr_db']:.2f} dB  |  SSIM {m['ssim']:.3f}  |  MS-SSIM {m['msssim']:.3f}"
      draw.text((x0 + 10, y_img + tile + 5), l1, fill=txt_main, font=f_cap)
      draw.text((x0 + 10, y_img + tile + 24), l2, fill=txt_muted, font=f_cap_s)

  y_r1 = hdr_top
  _draw_panel_row(
      y_r1,
      "1. Interactive Zero-Copy Time-of-Day Relighting — Classroom (Single Static G-Buffer, 4-Bounce MC GT Inset)",
      accent_cyan,
      cls_tod_preds,
      cls_tod_gts,
      cls_tod_metrics,
      "tod",
  )

  y_r2 = y_r1 + row_lbl_h + tile + sub_h + pad
  _draw_panel_row(
      y_r2,
      "2. Interactive Zero-Copy Material Relighting — Cornell Workshop Hero (Matte Clay -> Satin -> Brushed Gold -> Chrome)",
      accent_gold,
      crn_mat_preds,
      crn_mat_gts,
      crn_mat_metrics,
      "mat",
  )

  # Row 3: 2x-supersampled PIL dark-cinema Pareto + Scaling charts
  y_r3 = y_r2 + row_lbl_h + tile + sub_h + pad
  draw.text(
      (pad, y_r3 + 4),
      "3. TPU v5p MXU Precision Pareto (FP32 vs BF16 vs INT8 W8A8): Native 4K FPS vs Accuracy Retention & 1-Chip / 4-Chip Scaling",
      fill=(160, 232, 175),
      font=f_sec,
  )

  chart_w = canvas_w - 2 * pad
  ss = 2
  cw2, ch2 = chart_w * ss, chart_h * ss
  c_im = Image.new("RGB", (cw2, ch2), (20, 24, 34))
  cd = ImageDraw.Draw(c_im)
  fc_t = _load_font(15 * ss, bold=True)
  fc_l = _load_font(12 * ss, bold=False)
  fc_b = _load_font(12 * ss, bold=True)
  fc_s = _load_font(10 * ss, bold=False)

  # Subpanel geometry (left: Pareto scatter, right: log-scale grouped bar chart)
  gap_c = 44 * ss
  pw = (cw2 - 3 * gap_c) // 2
  p_top, p_bot = 52 * ss, ch2 - 58 * ss

  def _draw_panel_box(x_l, x_r, title_s):
    cd.rectangle([x_l, p_top, x_r, p_bot], fill=(24, 29, 42), outline=(51, 65, 85), width=2)
    cd.text((x_l + 8 * ss, 16 * ss), title_s, fill=(241, 245, 249), font=fc_t)

  # ---- Left Panel: 4K 4-Chip FPS vs Mean PSNR Pareto ----
  lx0, lx1 = gap_c + 12 * ss, gap_c + pw
  _draw_panel_box(lx0, lx1, "Native 4K (3840x2160, 4-Chip) FPS vs Relighting PSNR")

  pts = [
      ("FP32 (HIGHEST)", fp32_4k["fps"], pareto_summary["fp32"]["mean_psnr_db"], (248, 113, 113), (16, -18)),
      ("BF16 (Native MXU)", bf16_4k["fps"], pareto_summary["bf16"]["mean_psnr_db"], (56, 189, 248), (-168, -78)),
      ("INT8 W8A8 (PTQ)", int8_4k["fps"], pareto_summary["int8_ptq"]["mean_psnr_db"], (251, 191, 36), (-168, 32)),
      ("INT8 W8A8 (QAT)", int8_4k["fps"], pareto_summary["int8_qat"]["mean_psnr_db"], (74, 222, 128), (-190, -22)),
  ]
  xs_p = [p[1] for p in pts]
  ys_p = [p[2] for p in pts]
  x_max = max(xs_p) * 1.16
  y_min = min(ys_p) - 0.55
  y_max = max(ys_p) + 0.55

  def _map_l(xv, yv):
    px = lx0 + int((xv / max(x_max, 1e-6)) * (lx1 - lx0 - 24 * ss)) + 12 * ss
    py = p_bot - int(((yv - y_min) / max(y_max - y_min, 1e-6)) * (p_bot - p_top - 24 * ss)) - 12 * ss
    return px, py

  # Grid lines
  for gi in range(1, 5):
    yv = y_min + gi * (y_max - y_min) / 5.0
    _, gy = _map_l(0.0, yv)
    cd.line([lx0, gy, lx1, gy], fill=(42, 52, 70), width=1)
    cd.text((lx0 - 44 * ss, gy - 7 * ss), f"{yv:.1f}", fill=(148, 163, 184), font=fc_s)
  for gi in range(0, 5):
    xv = gi * x_max / 4.0
    gx, _ = _map_l(xv, y_min)
    cd.line([gx, p_top, gx, p_bot], fill=(42, 52, 70), width=1)
    cd.text((gx - 12 * ss, p_bot + 8 * ss), f"{xv:.0f}", fill=(148, 163, 184), font=fc_s)

  cd.text((lx0 + pw // 2 - 125 * ss, p_bot + 28 * ss), "Throughput (FPS at Native 4K, 4 TPU v5p Chips)", fill=(203, 213, 225), font=fc_l)
  cd.text((lx0 - 46 * ss, p_top - 22 * ss), "PSNR (dB)", fill=(203, 213, 225), font=fc_s)

  # Pareto frontier line FP32 -> BF16 -> INT8 QAT
  p0 = _map_l(pts[0][1], pts[0][2])
  p1 = _map_l(pts[1][1], pts[1][2])
  p3 = _map_l(pts[3][1], pts[3][2])
  cd.line([p0, p1, p3], fill=(148, 163, 184), width=3 * ss)

  for lbl, xv, yv, col, (ox, oy) in pts:
    px, py = _map_l(xv, yv)
    r = 8 * ss
    bx, by = px + ox * ss, py + oy * ss
    bx = max(lx0 + 6 * ss, min(bx, lx1 - 154 * ss))
    by = max(p_top + 6 * ss, min(by, p_bot - 42 * ss))
    cd.line([px, py, bx + 74 * ss, by + 18 * ss], fill=col, width=2)
    cd.ellipse([px - r, py - r, px + r, py + r], fill=col, outline=(15, 23, 42), width=2 * ss)
    cd.rounded_rectangle([bx, by, bx + 148 * ss, by + 36 * ss], radius=6 * ss, fill=(15, 23, 42), outline=col, width=2)
    cd.text((bx + 7 * ss, by + 4 * ss), lbl, fill=col, font=fc_s)
    cd.text((bx + 7 * ss, by + 18 * ss), f"{xv:.1f} FPS | {yv:.2f} dB", fill=(241, 245, 249), font=fc_s)

  # ---- Right Panel: 1-Chip vs 4-Chip Grouped Bar Chart (log scale) ----
  rx0, rx1 = 2 * gap_c + pw, 2 * gap_c + 2 * pw
  _draw_panel_box(rx0, rx1, "1-Chip vs 4-Chip Throughput Scaling (1080p & Native 4K, Log FPS)")

  configs = [("1080p 1-Chip", "1920x1080", 1), ("1080p 4-Chip", "1920x1080", 4), ("4K 1-Chip", "3840x2160", 1), ("4K 4-Chip", "3840x2160", 4)]
  log_min = math.log10(2.0)
  log_max = math.log10(max(r["fps"] for r in bench_recs) * 2.4)

  def _map_ry(v):
    lv = math.log10(max(v, 2.0))
    frac = (lv - log_min) / max(log_max - log_min, 1e-6)
    return p_bot - int(frac * (p_bot - p_top - 16 * ss))

  for tick in [3, 10, 30, 100, 300]:
    if 10 ** log_min <= tick <= 10 ** log_max:
      ty = _map_ry(float(tick))
      cd.line([rx0, ty, rx1, ty], fill=(42, 52, 70), width=1)
      cd.text((rx0 - 34 * ss, ty - 7 * ss), str(tick), fill=(148, 163, 184), font=fc_s)
  cd.text((rx0 - 8 * ss, p_top - 20 * ss), "FPS (log)", fill=(203, 213, 225), font=fc_s)

  precs_bar = [("fp32", "FP32", (248, 113, 113)), ("bf16", "BF16", (56, 189, 248)), ("int8", "INT8 W8A8", (74, 222, 128))]
  grp_w = (rx1 - rx0 - 40 * ss) // len(configs)
  bw = int(grp_w * 0.24)
  for c_idx, (c_lbl, res_s, nc) in enumerate(configs):
    gx_center = rx0 + 20 * ss + c_idx * grp_w + grp_w // 2
    cd.text((gx_center - 38 * ss, p_bot + 12 * ss), c_lbl, fill=(226, 232, 240), font=fc_b)
    for p_idx, (prec, _, col) in enumerate(precs_bar):
      rec = next(r for r in bench_recs if r["precision"] == prec and r["resolution"] == res_s and r["n_chips"] == nc)
      v = rec["fps"]
      bx0 = gx_center + int((p_idx - 1.5) * (bw + 4 * ss))
      bx1 = bx0 + bw
      by0 = _map_ry(v)
      cd.rectangle([bx0, by0, bx1, p_bot], fill=col, outline=(15, 23, 42), width=2)
      v_str = f"{v:.0f}" if v >= 20 else f"{v:.1f}"
      cd.text((bx0 + 2 * ss, by0 - 16 * ss), v_str, fill=col, font=fc_s)

  # Legend in top-right of right panel
  leg_x, leg_y = rx1 - 245 * ss, p_top + 10 * ss
  cd.rounded_rectangle([leg_x, leg_y, rx1 - 10 * ss, leg_y + 28 * ss], radius=6 * ss, fill=(15, 23, 42), outline=(51, 65, 85), width=2)
  for p_idx, (_, p_lbl, col) in enumerate(precs_bar):
    lx_i = leg_x + 10 * ss + p_idx * 74 * ss
    cd.rectangle([lx_i, leg_y + 8 * ss, lx_i + 12 * ss, leg_y + 20 * ss], fill=col)
    cd.text((lx_i + 16 * ss, leg_y + 7 * ss), p_lbl, fill=(226, 232, 240), font=fc_s)

  im_chart = c_im.resize((chart_w, chart_h), Image.Resampling.LANCZOS)
  canvas.paste(im_chart, (pad, y_r3 + row_lbl_h))
  draw.rectangle([pad - 1, y_r3 + row_lbl_h - 1, pad + chart_w, y_r3 + row_lbl_h + chart_h], outline=border_col, width=1)

  os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
  canvas.save(out_png)
  print("Saved showcase PNG ->", out_png, canvas.size, flush=True)


# ==============================================================================
# 3. Main Pipeline
# ==============================================================================

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--data_npz", default="/dev/shm/zaychik-tpu-mjx-w4/out/wave5_relight_ds.npz")
  ap.add_argument("--ckpt_pkl", default="/dev/shm/zaychik-tpu-mjx-w4/out/wave5_relight_ckpts.pkl")
  ap.add_argument("--out_json", default="/dev/shm/zaychik-tpu-mjx-w4/out/v3_w4_wave5_relight_quant.json")
  ap.add_argument("--out_png", default="/dev/shm/zaychik-tpu-mjx-w4/out/v3_w4_interactive_relight_quant.png")
  ap.add_argument("--n_train_per_scene", type=int, default=36)
  ap.add_argument("--gt_spp", type=int, default=256)
  ap.add_argument("--epochs_bf16", type=int, default=400)
  ap.add_argument("--epochs_qat", type=int, default=60)
  ap.add_argument("--skip_data", action="store_true")
  ap.add_argument("--skip_train", action="store_true")
  args = ap.parse_args()

  devs = jax.devices()
  assert len(devs) == 4, f"Expected 4 TPU chips on w4, found {len(devs)}"

  if not (args.skip_data and os.path.exists(args.data_npz)):
    generate_relight_dataset(
        devs,
        args.data_npz,
        train_res=256,
        test_res=512,
        gt_spp=args.gt_spp,
        n_train_per_scene=args.n_train_per_scene,
        n_test_rand_per_scene=4,
        max_bounces=4,
    )

  ds = np.load(args.data_npz, allow_pickle=False)
  tr_gb, tr_gt, tr_cond = ds["tr_gb"], ds["tr_gt"], ds["tr_cond"]
  print(f"[MAIN] Loaded dataset: train={tr_gt.shape}, test_res={ds['classroom_tod_gt'].shape[1]}x{ds['classroom_tod_gt'].shape[2]}", flush=True)

  if args.skip_train and os.path.exists(args.ckpt_pkl):
    with open(args.ckpt_pkl, "rb") as f:
      ck = pickle.load(f)
    bf16_params = ck["bf16_params"]
    ptq_int8_params = ck["ptq_int8_params"]
    qat_int8_params = ck["qat_int8_params"]
    n_params = int(ck["n_params"])
    train_bf16_s = float(ck["train_bf16_s"])
    train_qat_s = float(ck["train_qat_s"])
  else:
    bf16_params, n_params, train_bf16_s = train_relighter(
        tr_gb, tr_gt, tr_cond, epochs=args.epochs_bf16, batch_size=16, patch_size=160, lr=2.0e-3, qat=False
    )
    # Calibrate activation clip scales across BOTH classroom & cornell_workshop (strided!)
    dev0 = devs[0]
    cal_scales = calibrate_act_scales(
        jax.device_put(bf16_params, dev0),
        jax.device_put(tr_gb[::2], dev0),
        jax.device_put(tr_cond[::2], dev0),
        percentile=99.99,
    )
    bf16_with_cal = dict(bf16_params)
    bf16_with_cal["act_scales"] = cal_scales
    ptq_int8_params = quantize_params_static_int8(bf16_with_cal)

    qat_float_params, _, train_qat_s = train_relighter(
        tr_gb, tr_gt, tr_cond, epochs=args.epochs_qat, batch_size=16, patch_size=160, lr=2.5e-5,
        init_params=bf16_with_cal, qat=True, seed_offset=20000
    )
    qat_int8_params = quantize_params_static_int8(qat_float_params)

    os.makedirs(os.path.dirname(args.ckpt_pkl) or ".", exist_ok=True)
    with open(args.ckpt_pkl, "wb") as f:
      pickle.dump(
          {
              "bf16_params": jax.tree_util.tree_map(np.asarray, bf16_with_cal),
              "ptq_int8_params": jax.tree_util.tree_map(np.asarray, ptq_int8_params),
              "qat_int8_params": jax.tree_util.tree_map(np.asarray, qat_int8_params),
              "n_params": n_params,
              "train_bf16_s": train_bf16_s,
              "train_qat_s": train_qat_s,
          },
          f,
      )
    bf16_params = bf16_with_cal

  dev0 = devs[0]
  p_fp32_d = jax.device_put(jax.tree_util.tree_map(jnp.asarray, bf16_params), dev0)
  p_ptq_d = jax.device_put(jax.tree_util.tree_map(jnp.asarray, ptq_int8_params), dev0)
  p_qat_d = jax.device_put(jax.tree_util.tree_map(jnp.asarray, qat_int8_params), dev0)

  @jax.jit
  def _infer_fp32(p, gb, cond):
    return relight_forward_float_or_qat(p, gb, cond, precision_mode="fp32", qat=False)

  @jax.jit
  def _infer_bf16(p, gb, cond):
    return relight_forward_float_or_qat(p, gb, cond, precision_mode="bf16", qat=False)

  @jax.jit
  def _infer_int8(p, gb, cond):
    return relight_forward_int8(p, gb, cond)

  # Evaluate all splits across all 4 precision modes using ZERO-RAYTRACE single static G-buffer for ToD/Material sweeps!
  def _eval_split(gb_np: np.ndarray, gt_np: np.ndarray, cond_np: np.ndarray, static_single_gbuf: bool = False):
    gb_in = np.broadcast_to(gb_np[:1], gb_np.shape).copy() if static_single_gbuf else gb_np
    gb_d = jax.device_put(gb_in.astype(np.float32), dev0)
    c_d = jax.device_put(cond_np.astype(np.float32), dev0)
    pr_fp32 = np.asarray(_infer_fp32(p_fp32_d, gb_d, c_d), dtype=np.float32)
    pr_bf16 = np.asarray(_infer_bf16(p_fp32_d, gb_d, c_d), dtype=np.float32)
    pr_ptq = np.asarray(_infer_int8(p_ptq_d, gb_d, c_d), dtype=np.float32)
    pr_qat = np.asarray(_infer_int8(p_qat_d, gb_d, c_d), dtype=np.float32)
    return {
        "fp32": (_compute_metrics(pr_fp32, gt_np), pr_fp32),
        "bf16": (_compute_metrics(pr_bf16, gt_np), pr_bf16),
        "int8_ptq": (_compute_metrics(pr_ptq, gt_np), pr_ptq),
        "int8_qat": (_compute_metrics(pr_qat, gt_np), pr_qat),
    }

  quality_results: Dict[str, Any] = {}
  cls_tod_preds, crn_mat_preds = None, None
  cls_tod_metrics, crn_mat_metrics = [], []

  for sname in ["classroom", "cornell_workshop"]:
    tod_res = _eval_split(ds[f"{sname}_tod_gb"], ds[f"{sname}_tod_gt"], ds[f"{sname}_tod_cond"], static_single_gbuf=True)
    mat_res = _eval_split(ds[f"{sname}_mat_gb"], ds[f"{sname}_mat_gt"], ds[f"{sname}_mat_cond"], static_single_gbuf=True)
    rnd_res = _eval_split(ds[f"{sname}_rand_gb"], ds[f"{sname}_rand_gt"], ds[f"{sname}_rand_cond"], static_single_gbuf=False)

    # Per-preset breakdown for BF16 & INT8 QAT
    tod_per_preset = []
    for i, tod in enumerate(TOD_PRESETS):
      m_bf = _compute_metrics(tod_res["bf16"][1][i:i+1], ds[f"{sname}_tod_gt"][i:i+1])
      m_q = _compute_metrics(tod_res["int8_qat"][1][i:i+1], ds[f"{sname}_tod_gt"][i:i+1])
      tod_per_preset.append({**tod, **m_bf, "int8_qat_psnr_db": m_q["psnr_db"], "int8_qat_ssim": m_q["ssim"]})

    mat_per_preset = []
    for i, mat in enumerate(MATERIAL_PRESETS):
      m_bf = _compute_metrics(mat_res["bf16"][1][i:i+1], ds[f"{sname}_mat_gt"][i:i+1])
      m_q = _compute_metrics(mat_res["int8_qat"][1][i:i+1], ds[f"{sname}_mat_gt"][i:i+1])
      mat_per_preset.append({**mat, **m_bf, "int8_qat_psnr_db": m_q["psnr_db"], "int8_qat_ssim": m_q["ssim"]})

    if sname == "classroom":
      cls_tod_preds = tod_res["bf16"][1]
      cls_tod_metrics = tod_per_preset
    if sname == "cornell_workshop":
      crn_mat_preds = mat_res["bf16"][1]
      crn_mat_metrics = mat_per_preset

    quality_results[sname] = {
        "tod_sweep_static_gbuf": {k: v[0] for k, v in tod_res.items()},
        "tod_per_preset": tod_per_preset,
        "mat_sweep_static_gbuf": {k: v[0] for k, v in mat_res.items()},
        "mat_per_preset": mat_per_preset,
        "unseen_rand_views": {k: v[0] for k, v in rnd_res.items()},
    }
    print(f"[EVAL {sname}] ToD BF16={tod_res['bf16'][0]['psnr_db']:.2f}dB / INT8-QAT={tod_res['int8_qat'][0]['psnr_db']:.2f}dB | "
          f"Mat BF16={mat_res['bf16'][0]['psnr_db']:.2f}dB / INT8-QAT={mat_res['int8_qat'][0]['psnr_db']:.2f}dB | "
          f"RandView BF16={rnd_res['bf16'][0]['psnr_db']:.2f}dB", flush=True)

  # Aggregate mean PSNR / SSIM across all held-out evaluation frames (ToD + Mat + Rand on both scenes)
  all_gt = np.concatenate([
      ds["classroom_tod_gt"], ds["classroom_mat_gt"], ds["classroom_rand_gt"],
      ds["cornell_workshop_tod_gt"], ds["cornell_workshop_mat_gt"], ds["cornell_workshop_rand_gt"],
  ], axis=0)
  all_gb = np.concatenate([
      np.broadcast_to(ds["classroom_tod_gb"][:1], ds["classroom_tod_gb"].shape),
      np.broadcast_to(ds["classroom_mat_gb"][:1], ds["classroom_mat_gb"].shape),
      ds["classroom_rand_gb"],
      np.broadcast_to(ds["cornell_workshop_tod_gb"][:1], ds["cornell_workshop_tod_gb"].shape),
      np.broadcast_to(ds["cornell_workshop_mat_gb"][:1], ds["cornell_workshop_mat_gb"].shape),
      ds["cornell_workshop_rand_gb"],
  ], axis=0)
  all_cond = np.concatenate([
      ds["classroom_tod_cond"], ds["classroom_mat_cond"], ds["classroom_rand_cond"],
      ds["cornell_workshop_tod_cond"], ds["cornell_workshop_mat_cond"], ds["cornell_workshop_rand_cond"],
  ], axis=0)

  agg_res = _eval_split(all_gb, all_gt, all_cond, static_single_gbuf=False)
  pareto_summary = {
      k: {
          "mean_psnr_db": v[0]["psnr_db"],
          "mean_ssim": v[0]["ssim"],
          "mean_msssim": v[0]["msssim"],
          "lum_bias_pct": v[0]["lum_bias_pct"],
      }
      for k, v in agg_res.items()
  }
  print("[PARETO ACCURACY]", json.dumps(pareto_summary, indent=2), flush=True)

  # Throughput Pareto Benchmark: FP32 vs BF16 vs INT8 W8A8 on 1-chip & 4-chip at 1080p & Native 4K
  bench_recs = benchmark_precision_pareto(
      bf16_params,
      qat_int8_params,
      devs,
      resolutions=((1080, 1920), (2160, 3840)),
      chip_counts=(1, 4),
      precisions=("fp32", "bf16", "int8"),
  )

  # Render dark-cinema showcase PNG
  make_showcase_figure(
      cls_tod_preds,
      ds["classroom_tod_gt"],
      cls_tod_metrics,
      crn_mat_preds,
      ds["cornell_workshop_mat_gt"],
      crn_mat_metrics,
      pareto_summary,
      bench_recs,
      args.out_png,
  )

  payload = {
      "experiment": "v3_w4_wave5_relight_quant",
      "backend": jax.default_backend(),
      "device_kind": devs[0].device_kind,
      "n_chips_total": len(devs),
      "model": {
          "arch": "CondFiLMMXURelighter_s2d_64_128_128_pix32_hw64",
          "n_params": n_params,
          "train_bf16_s": train_bf16_s,
          "train_qat_s": train_qat_s,
          "n_train_states": int(tr_gt.shape[0]),
          "gt_spp": int(args.gt_spp),
          "max_bounces": 4,
      },
      "precision_pareto_accuracy": pareto_summary,
      "quality_by_scene": quality_results,
      "throughput_benchmarks": bench_recs,
  }
  os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
  with open(args.out_json, "w") as f:
    json.dump(payload, f, indent=2)
  print("Saved Wave 5 JSON ->", args.out_json, flush=True)


if __name__ == "__main__":
  main()
