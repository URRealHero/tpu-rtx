"""Generate Dark-Cinema Wave 4 Showcase Figure `out/figures/v3_w1_wave4_vla_tactile_factory.png` using PIL + NumPy."""

from __future__ import annotations

import argparse
import json
import os
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_REG = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"


def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
  try:
    return ImageFont.truetype(path, size)
  except Exception:
    return ImageFont.load_default()


def _inferno_cmap(v: np.ndarray) -> np.ndarray:
  """Piecewise-linear inferno-like RGB colormap on [0, 1]."""
  x = np.clip(v, 0.0, 1.0)
  pts = np.array([0.0, 0.25, 0.50, 0.75, 1.0], dtype=np.float32)
  r = np.interp(x, pts, [0.02, 0.34, 0.73, 0.97, 0.99])
  g = np.interp(x, pts, [0.02, 0.06, 0.21, 0.55, 0.99])
  b = np.interp(x, pts, [0.08, 0.43, 0.33, 0.12, 0.64])
  return np.stack([r, g, b], axis=-1)


def _turbo_depth_cmap(v: np.ndarray) -> np.ndarray:
  """Crisp high-contrast cyan-amber-magenta tactile penetration colormap on [0, 1]."""
  x = np.clip(v, 0.0, 1.0)
  pts = np.array([0.0, 0.20, 0.50, 0.80, 1.0], dtype=np.float32)
  r = np.interp(x, pts, [0.07, 0.08, 0.22, 0.96, 1.00])
  g = np.interp(x, pts, [0.09, 0.48, 0.85, 0.66, 0.96])
  b = np.interp(x, pts, [0.14, 0.78, 0.55, 0.14, 0.70])
  return np.stack([r, g, b], axis=-1)


def _norm_scene_depth(d: np.ndarray) -> np.ndarray:
  valid = d > 0.0
  if not np.any(valid):
    return np.zeros((*d.shape, 3), dtype=np.uint8)
  lo = float(np.percentile(d[valid], 2))
  hi = float(np.percentile(d[valid], 95))
  n = np.clip(1.0 - (d - lo) / max(hi - lo, 1e-4), 0.0, 1.0)
  n = np.where(valid, n, 0.0)
  return (np.clip(_inferno_cmap(n), 0.0, 1.0) * 255.0).astype(np.uint8)


def _colorize_flow(flow: np.ndarray) -> np.ndarray:
  fx, fy = flow[..., 0], flow[..., 1]
  mag = np.sqrt(fx * fx + fy * fy)
  ang = (np.arctan2(fy, fx) + np.pi) / (2.0 * np.pi)
  moving = mag > 1e-4
  ref = float(np.percentile(mag[moving], 96)) if np.any(moving) else 1e-3
  scale = np.clip(mag / max(ref, 1e-4), 0.0, 1.0)
  # HSV -> RGB
  h6 = (ang % 1.0) * 6.0
  sat = np.where(moving, 0.92, 0.0)
  val = np.where(moving, 0.35 + 0.65 * scale, 0.09)
  c = val * sat
  x = c * (1.0 - np.abs((h6 % 2.0) - 1.0))
  m = val - c
  hi = np.floor(h6).astype(np.int32) % 6
  r = np.where(hi == 0, c, np.where(hi == 1, x, np.where(hi == 4, x, np.where(hi == 5, c, 0.0)))) + m
  g = np.where(hi == 0, x, np.where(hi == 1, c, np.where(hi == 2, c, np.where(hi == 3, x, 0.0)))) + m
  b = np.where(hi == 2, x, np.where(hi == 3, c, np.where(hi == 4, c, np.where(hi == 5, x, 0.0)))) + m
  return (np.clip(np.stack([r, g, b], axis=-1), 0.0, 1.0) * 255.0).astype(np.uint8)


def _to_u8(img: np.ndarray) -> np.ndarray:
  return (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)


def _paste_panel(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    arr_u8: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    title: str,
    subtitle: str = "",
    border_col: Tuple[int, int, int] = (46, 58, 80),
) -> None:
  f_t = _font(FONT_BOLD, 17)
  f_s = _font(FONT_REG, 13)
  # Card box
  draw.rounded_rectangle([x - 6, y - 48, x + w + 6, y + h + 8], radius=10, fill=(20, 26, 36), outline=border_col, width=2)
  draw.text((x + 2, y - 42), title, font=f_t, fill=(240, 244, 248))
  if subtitle:
    draw.text((x + 2, y - 21), subtitle, font=f_s, fill=(148, 163, 184))
  im = Image.fromarray(arr_u8).resize((w, h), Image.Resampling.BILINEAR)
  canvas.paste(im, (x, y))
  draw.rectangle([x, y, x + w - 1, y + h - 1], outline=(56, 70, 96), width=1)


def _draw_bar_chart(
    w: int,
    h: int,
    title: str,
    subtitle: str,
    categories: list[str],
    series: list[tuple[str, list[float], Tuple[int, int, int]]],
    ylabel: str,
    val_fmt: str = "{:.1f}",
    annotations: list[str] | None = None,
) -> Image.Image:
  """Render a 2x supersampled crisp dark-cinema bar chart."""
  S = 2
  W, H = w * S, h * S
  im = Image.new("RGB", (W, H), (20, 26, 36))
  dr = ImageDraw.Draw(im)
  dr.rounded_rectangle([2, 2, W - 3, H - 3], radius=18, fill=(20, 26, 36), outline=(46, 58, 80), width=3)

  f_ttl = _font(FONT_BOLD, 18 * S)
  f_sub = _font(FONT_REG, 13 * S)
  f_ax = _font(FONT_REG, 13 * S)
  f_lbl = _font(FONT_BOLD, 13 * S)
  f_val = _font(FONT_MONO, 12 * S)

  dr.text((20 * S, 14 * S), title, font=f_ttl, fill=(240, 244, 248))
  dr.text((20 * S, 38 * S), subtitle, font=f_sub, fill=(148, 163, 184))

  # Legend top-right
  lx = W - 20 * S
  for sname, _, scol in reversed(series):
    tw = dr.textlength(sname, font=f_sub)
    lx -= (tw + 26 * S)
    dr.rounded_rectangle([lx, 38 * S, lx + 14 * S, 50 * S], radius=3 * S, fill=scol)
    dr.text((lx + 18 * S, 36 * S), sname, font=f_sub, fill=(220, 228, 238))
    lx -= 12 * S

  pad_l, pad_r, pad_t, pad_b = 76 * S, 24 * S, 76 * S, 56 * S
  px0, py0, px1, py1 = pad_l, pad_t, W - pad_r, H - pad_b
  pw, ph = px1 - px0, py1 - py0

  all_vals = [v for _, vals, _ in series for v in vals]
  max_v = max(all_vals) * 1.24 if all_vals else 1.0

  # Gridlines
  for i in range(5):
    frac = i / 4.0
    yy = int(py1 - frac * ph)
    dr.line([(px0, yy), (px1, yy)], fill=(40, 50, 68), width=2)
    yv = max_v * frac
    ytxt = f"{yv/1000:.0f}k" if max_v >= 4000 else f"{yv:.1f}"
    tw = dr.textlength(ytxt, font=f_ax)
    dr.text((px0 - tw - 10 * S, yy - 8 * S), ytxt, font=f_ax, fill=(148, 163, 184))

  dr.line([(px0, py0), (px0, py1)], fill=(90, 108, 136), width=2)
  dr.line([(px0, py1), (px1, py1)], fill=(90, 108, 136), width=2)
  dr.text((14 * S, py0 - 20 * S), ylabel, font=f_sub, fill=(148, 163, 184))

  n_cat = len(categories)
  n_ser = len(series)
  grp_w = pw / max(n_cat, 1)
  bar_w = min(int(grp_w * 0.66 / max(n_ser, 1)), 64 * S)
  gap = 8 * S

  for ci, cat in enumerate(categories):
    cx = px0 + (ci + 0.5) * grp_w
    total_w = n_ser * bar_w + (n_ser - 1) * gap
    bx_start = cx - 0.5 * total_w
    for si, (_, vals, scol) in enumerate(series):
      v = vals[ci]
      bh = max(int((v / max(max_v, 1e-9)) * ph), 4)
      bx0 = int(bx_start + si * (bar_w + gap))
      bx1 = bx0 + bar_w
      by0 = py1 - bh
      dr.rounded_rectangle([bx0, by0, bx1, py1], radius=5 * S, fill=scol)
      v_str = val_fmt.format(v)
      vw = dr.textlength(v_str, font=f_val)
      dr.text((bx0 + 0.5 * bar_w - 0.5 * vw, by0 - 18 * S), v_str, font=f_val, fill=(245, 248, 252))

    # Category label
    cw = dr.textlength(cat, font=f_lbl)
    dr.text((cx - 0.5 * cw, py1 + 10 * S), cat, font=f_lbl, fill=(226, 232, 240))
    if annotations and ci < len(annotations):
      aw = dr.textlength(annotations[ci], font=f_sub)
      dr.text((cx - 0.5 * aw, py1 + 30 * S), annotations[ci], font=f_sub, fill=(125, 211, 252))

  return im.resize((w, h), Image.Resampling.LANCZOS)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--json", default="/dev/shm/zaychik-tpu-mjx-w1/out/v3_w1_wave4_vla_tactile_factory.json")
  ap.add_argument("--npz", default="/dev/shm/zaychik-tpu-mjx-w1/out/wave4_showcase.npz")
  ap.add_argument("--out_png", default="/dev/shm/zaychik-tpu-mjx-w1/out/v3_w1_wave4_vla_tactile_factory.png")
  args = ap.parse_args()

  with open(args.json, "r") as f:
    data = json.load(f)
  ds = np.load(args.npz)

  W, H = 2400, 1520
  canvas = Image.new("RGB", (W, H), (11, 14, 20))
  draw = ImageDraw.Draw(canvas)

  # Top Dark-Cinema Banner
  f_h1 = _font(FONT_BOLD, 30)
  f_h2 = _font(FONT_REG, 18)
  f_sec = _font(FONT_BOLD, 20)
  draw.text((40, 22), "TPU-MJX Wave 4: G=4,096 Multi-Robot Factory VLA & GelSight Visuo-Tactile", font=f_h1, fill=(248, 250, 252))
  draw.text(
      (40, 62),
      "4-Chip TPU v5p | Morton Cluster-BVH + Isaac Replicator DR | Zero-Copy Flax BF16 ViT-Tiny + AdamW | Single-Kernel Fused GelSight",
      font=f_h2,
      fill=(56, 189, 248),
  )
  draw.line([(40, 94), (W - 40, 94)], fill=(38, 48, 66), width=2)

  # Section 1: G=4,096 Multi-Robot Factory PBR & Modalities (6 panels across top)
  draw.text((40, 108), "A. G=4,096 Dynamic Multi-Robot Assembly Factory (4 Articulated Arms + Dual Conveyors + Parcels, 256²)", font=f_sec, fill=(226, 232, 240))
  pw1, ph1 = 360, 280
  gap1 = 32
  y1 = 192
  row1_panels = [
      (_to_u8(ds["f4_over_rgb"][0]), "Factory G=4,096 DR #0", "Overview (GGX PBR + AgX)"),
      (_to_u8(ds["f4_over_rgb"][1]), "Factory G=4,096 DR #1", "Warm Tungsten + Tex DR"),
      (_to_u8(ds["f4_close_rgb"][0]), "Factory G=4,096 Closeup", "4 Arms + Moving Conveyors"),
      (_to_u8(ds["f4_wrist_rgb"][0]), "Ego Wrist Cam (arm_nw)", "Live Articulated VLA View"),
      (_norm_scene_depth(ds["f4_over_depth"]), "Metric Planar Z-Depth", "Exact Analytic Distance (m)"),
      (_colorize_flow(ds["f4_over_flow"]), "2D Optical Flow", "Zero-Gather Motion Vectors"),
  ]
  for i, (arr, ttl, sub) in enumerate(row1_panels):
    x = 40 + i * (pw1 + gap1)
    _paste_panel(canvas, draw, arr, x, y1, pw1, ph1, ttl, sub)

  # Section 2: Dual-Fingertip GelSight / TacSL Visuo-Tactile Maps (Cube, Cylinder, Sphere)
  draw.text((40, 500), "B. Single-Kernel Fused Visuo-Tactile Simulation (Head + Wrist + L/R GelSight Tricolor Photometric Stereo, Depth & Normals)", font=f_sec, fill=(226, 232, 240))

  # Compute global max penetration across all 3 shapes for consistent depth colormap scale
  max_pen_m = max(
      float(np.max(ds[f"{s}_tac_{side}_depth"]))
      for s in ("cube", "cylinder", "sphere")
      for side in ("l", "r")
  )
  max_pen_m = max(max_pen_m, 1e-4)

  card_w = 750
  card_gap = 35
  y2 = 540
  card_h = 445
  shape_meta = [
      ("cube", "1. Tilted Cube Grasp", (244, 63, 94)),
      ("cylinder", "2. Cylinder Grasp", (56, 189, 248)),
      ("sphere", "3. Sphere Grasp", (245, 158, 11)),
  ]
  f_ct = _font(FONT_BOLD, 18)
  f_cs = _font(FONT_REG, 13)
  for ci, (shp, stitle, accent) in enumerate(shape_meta):
    cx = 40 + ci * (card_w + card_gap)
    draw.rounded_rectangle([cx, y2, cx + card_w, y2 + card_h], radius=12, fill=(18, 23, 33), outline=accent, width=2)
    pen_mm = max(float(np.max(ds[f"{shp}_tac_l_depth"])), float(np.max(ds[f"{shp}_tac_r_depth"]))) * 1e3
    draw.text((cx + 16, y2 + 12), f"{stitle}   (Max Indentation: {pen_mm:.2f} mm)", font=f_ct, fill=(248, 250, 252))

    # 4 sub-panels per shape: [Head+Wrist RGB] | [L+R GelSight RGB] | [L+R Depth] | [L+R Normals]
    # Top row inside card: Head | Wrist | L-GelSight RGB | R-GelSight RGB (4 x 165x165)
    # Bottom row inside card: L-Depth | R-Depth | L-Normal | R-Normal (4 x 165x165)
    sub_sz = 165
    sub_gap = 16
    sx0 = cx + 22
    sy_top = y2 + 66
    sy_bot = y2 + 266

    top_items = [
        (_to_u8(ds[f"{shp}_head_rgb"]), "Head Cam (128²)"),
        (_to_u8(ds[f"{shp}_wrist_rgb"]), "Wrist Cam (128²)"),
        (_to_u8(ds[f"{shp}_tac_l_rgb"]), "L-GelSight R/G/B"),
        (_to_u8(ds[f"{shp}_tac_r_rgb"]), "R-GelSight R/G/B"),
    ]
    d_l_rgb = _to_u8(_turbo_depth_cmap(ds[f"{shp}_tac_l_depth"] / max_pen_m))
    d_r_rgb = _to_u8(_turbo_depth_cmap(ds[f"{shp}_tac_r_depth"] / max_pen_m))
    n_l_rgb = _to_u8(0.5 * (ds[f"{shp}_tac_l_norm"] + 1.0))
    n_r_rgb = _to_u8(0.5 * (ds[f"{shp}_tac_r_norm"] + 1.0))
    bot_items = [
        (d_l_rgb, "L-Depth (mm)"),
        (d_r_rgb, "R-Depth (mm)"),
        (n_l_rgb, "L-Contact Normal"),
        (n_r_rgb, "R-Contact Normal"),
    ]
    for k, (arr, lbl) in enumerate(top_items):
      px = sx0 + k * (sub_sz + sub_gap)
      draw.text((px, sy_top - 20), lbl, font=f_cs, fill=(203, 213, 225))
      im = Image.fromarray(arr).resize((sub_sz, sub_sz), Image.Resampling.BILINEAR)
      canvas.paste(im, (px, sy_top))
      draw.rectangle([px, sy_top, px + sub_sz - 1, sy_top + sub_sz - 1], outline=(56, 70, 96), width=1)

    for k, (arr, lbl) in enumerate(bot_items):
      px = sx0 + k * (sub_sz + sub_gap)
      draw.text((px, sy_bot - 20), lbl, font=f_cs, fill=(203, 213, 225))
      im = Image.fromarray(arr).resize((sub_sz, sub_sz), Image.Resampling.BILINEAR)
      canvas.paste(im, (px, sy_bot))
      draw.rectangle([px, sy_bot, px + sub_sz - 1, sy_bot + sub_sz - 1], outline=(56, 70, 96), width=1)

  # Section 3: Quantitative Benchmark Charts (Bottom Row, 3 wide charts)
  draw.text((40, 1008), "C. 4-Chip TPU v5p Quantitative Scaling: 1M-Frame Wall-Clock, Zero-Copy Sim-to-VLA & Visuo-Tactile Throughput", font=f_sec, fill=(226, 232, 240))

  y3 = 1044
  ch_w, ch_h, ch_gap = 750, 445, 35

  # Chart 1: 1M-frame Wall-Clock Time (seconds) + cam-FPS
  m_runs = data.get("task_a_1m_frames", [])
  m_2k = next((r for r in m_runs if r["n_enabled_geoms"] == 2048), None)
  m_4k = next((r for r in m_runs if r["n_enabled_geoms"] == 4096), None)
  s_1m_vals = [
      m_2k["seconds_per_1m_frames"] if m_2k else 0.0,
      m_4k["seconds_per_1m_frames"] if m_4k else 0.0,
  ]
  fps_1m_vals = [
      m_2k["cam_frames_per_s"] if m_2k else 0.0,
      m_4k["cam_frames_per_s"] if m_4k else 0.0,
  ]
  ann_1 = [
      f"{fps_1m_vals[0]:,.0f} cam-FPS | {m_2k['env_steps_per_s']:,.0f} env/s" if m_2k else "",
      f"{fps_1m_vals[1]:,.0f} cam-FPS | {m_4k['env_steps_per_s']:,.0f} env/s" if m_4k else "",
  ]
  ch1 = _draw_bar_chart(
      ch_w, ch_h,
      "1,000,000-Frame Wall-Clock (4-Chip TPU v5p)",
      "6-Modality 2-Cam Factory + MJX Physics + DR (Lower is Faster)",
      ["G=2,048 Prims", "G=4,096 Prims"],
      [("Seconds per 1M Frames", s_1m_vals, (56, 189, 248))],
      "Seconds",
      val_fmt="{:.2f}s",
      annotations=ann_1,
  )
  canvas.paste(ch1, (40, y3))

  # Chart 2: Zero-Copy Sim-to-VLA Online Training vs Sim-Only Throughput
  s_runs = data.get("task_a_factory_scaling", [])
  v_runs = data.get("task_a_sim_to_vla", [])
  s2k = next((r for r in s_runs if r["n_enabled_geoms"] == 2048), None)
  s4k = next((r for r in s_runs if r["n_enabled_geoms"] == 4096), None)
  v2k = next((r for r in v_runs if r["n_enabled_geoms"] == 2048), None)
  v4k = next((r for r in v_runs if r["n_enabled_geoms"] == 4096), None)

  ch2 = _draw_bar_chart(
      ch_w, ch_h,
      "Zero-Copy Sim-to-ViT-Tiny VLA Training",
      "Fused MJX + Pallas + BF16 ViT-Tiny + AdamW (Same XLA Graph)",
      ["G=2,048 Prims", "G=4,096 Prims"],
      [
          ("Sim+Render Only", [s2k["cam_frames_per_s"] if s2k else 0.0, s4k["cam_frames_per_s"] if s4k else 0.0], (34, 197, 94)),
          ("Sim+ViT-Tiny+AdamW", [v2k["frames_per_s"] if v2k else 0.0, v4k["frames_per_s"] if v4k else 0.0], (245, 158, 11)),
      ],
      "Frames / sec",
      val_fmt="{:,.0f}",
      annotations=[
          f"{v2k['grad_steps_per_s']:.1f} grad-step/s (+{v2k['vla_overhead_pct_vs_sim']:.1f}%)" if v2k else "",
          f"{v4k['grad_steps_per_s']:.1f} grad-step/s (+{v4k['vla_overhead_pct_vs_sim']:.1f}%)" if v4k else "",
      ],
  )
  canvas.paste(ch2, (40 + ch_w + ch_gap, y3))

  # Chart 3: Visuo-Tactile 4-Sensor Throughput (1c vs 4c)
  t_runs = data.get("task_b_visuo_tactile_runs", [])
  def _find_t(nc: int, b: int, dr: bool):
    return next((r for r in t_runs if r["n_chips"] == nc and r["batch_size"] == b and r["with_dr"] == dr), None)

  t_1c_s = _find_t(1, 1024, False)
  t_1c_d = _find_t(1, 1024, True)
  t_4c1_s = _find_t(4, 1024, False)
  t_4c1_d = _find_t(4, 1024, True)
  t_4c2_s = _find_t(4, 2048, False)
  t_4c2_d = _find_t(4, 2048, True)

  ch3 = _draw_bar_chart(
      ch_w, ch_h,
      "4-Sensor Visuo-Tactile Scaling (Head+Wrist+L/R GelSight)",
      "Single Pallas Kernel + MJX Grasp Physics (Parity < 1e-5 m)",
      ["1c (B=1024)", "4c (B=1024)", "4c (B=2048)"],
      [
          ("Static Material", [
              t_1c_s["sensor_fps"] if t_1c_s else 0.0,
              t_4c1_s["sensor_fps"] if t_4c1_s else 0.0,
              t_4c2_s["sensor_fps"] if t_4c2_s else 0.0,
          ], (99, 102, 241)),
          ("Live Replicator DR", [
              t_1c_d["sensor_fps"] if t_1c_d else 0.0,
              t_4c1_d["sensor_fps"] if t_4c1_d else 0.0,
              t_4c2_d["sensor_fps"] if t_4c2_d else 0.0,
          ], (236, 72, 153)),
      ],
      "Sensor-FPS",
      val_fmt="{:,.0f}",
      annotations=[
          f"{t_1c_d['env_steps_per_s']:,.0f} env/s" if t_1c_d else "",
          f"{t_4c1_d['env_steps_per_s']:,.0f} env/s" if t_4c1_d else "",
          f"{t_4c2_d['env_steps_per_s']:,.0f} env/s" if t_4c2_d else "",
      ],
  )
  canvas.paste(ch3, (40 + 2 * (ch_w + ch_gap), y3))

  os.makedirs(os.path.dirname(args.out_png) or ".", exist_ok=True)
  canvas.save(args.out_png, optimize=True)
  print("Saved dark-cinema showcase PNG:", args.out_png, flush=True)


if __name__ == "__main__":
  main()
