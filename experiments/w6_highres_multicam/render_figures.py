"""Compose crisp PNG figures from TPU-rendered 4K/8K/RoboTwin/LiDAR arrays using NumPy + PIL."""

from __future__ import annotations

import argparse
import os
from typing import Dict, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Semantic palettes (RGB 0..255)
SEM_PALETTE_LIDAR = np.array([
    [14, 18, 28],     # 0: sky / miss
    [76, 86, 106],    # 1: floor / workbench
    [235, 239, 245],  # 2: robot panda
    [244, 67, 54],    # 3: target objects
    [33, 150, 243],   # 4: sorting bins
    [255, 179, 0],    # 5: tools / bottles / rack
], dtype=np.uint8)

SEM_PALETTE_BIMANUAL = np.array([
    [14, 18, 28],     # 0: sky / miss
    [72, 82, 100],    # 1: floor / workbench / mast
    [0, 188, 212],    # 2: left Panda arm
    [255, 112, 67],   # 3: right Panda arm
    [244, 67, 54],    # 4: target objects / props
    [56, 142, 245],   # 5: sorting bins
    [255, 193, 7],    # 6: tools / bottles / rack
], dtype=np.uint8)


def _to_u8(rgb_f: np.ndarray) -> np.ndarray:
  return np.clip(rgb_f * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)


def _turbo_colormap(x: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
  """Compact analytic Turbo-like colormap for depth/range in [0, 1]."""
  t = np.clip(x, 0.0, 1.0)
  r = np.clip(0.1357 + t * (4.6153 + t * (-42.660 + t * (132.13 + t * (-152.95 + t * 59.286)))), 0.0, 1.0)
  g = np.clip(0.0914 + t * (2.1942 + t * (4.8430 + t * (-14.185 + t * (4.2772 + t * 2.8296)))), 0.0, 1.0)
  b = np.clip(0.1067 + t * (12.642 + t * (-60.582 + t * (110.36 + t * (-89.903 + t * 27.348)))), 0.0, 1.0)
  out = np.stack([r, g, b], axis=-1)
  out = np.where(valid_mask[..., None], out, 0.06)
  return _to_u8(out)


def _depth_vis(depth: np.ndarray, d_min: float = 0.15, d_max: float = 2.4) -> np.ndarray:
  valid = depth > 0.0
  norm = np.where(valid, 1.0 - np.clip((depth - d_min) / max(d_max - d_min, 1e-6), 0.0, 1.0), 0.0)
  return _turbo_colormap(norm, valid)


def _draw_label(img: Image.Image, text: str, xy: Tuple[int, int] = (12, 10), bg=(15, 18, 26, 215), fg=(240, 246, 255)):
  draw = ImageDraw.Draw(img, "RGBA")
  try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
  except Exception:
    font = ImageFont.load_default()
  bbox = draw.textbbox(xy, text, font=font)
  pad = 8
  draw.rectangle([bbox[0] - pad, bbox[1] - pad + 2, bbox[2] + pad, bbox[3] + pad], fill=bg)
  draw.text(xy, text, fill=fg, font=font)


def _resize_u8(arr: np.ndarray, w: int, h: int, resample=Image.BILINEAR) -> np.ndarray:
  return np.asarray(Image.fromarray(arr).resize((w, h), resample=resample))


def compose_4k_panda(vis_dir: str, out_dir: str):
  s1_p = os.path.join(vis_dir, "frame_4k_stage1_rgb.npy")
  s12_p = os.path.join(vis_dir, "frame_4k_stage12_rgb.npy")
  if os.path.exists(s1_p):
    s1 = np.load(s1_p)
    Image.fromarray(_to_u8(s1)).save(os.path.join(out_dir, "v3_w6_panda_4k_stage1.png"))
  if os.path.exists(s12_p):
    s12 = np.load(s12_p)
    Image.fromarray(_to_u8(s12)).save(os.path.join(out_dir, "v3_w6_panda_4k_stage12.png"))
  if os.path.exists(s1_p) and os.path.exists(s12_p):
    s1 = np.load(s1_p)
    s12 = np.load(s12_p)
    # 100% crop around gripper+cube (center region of 3840x2160)
    y0, y1, x0, x1 = 620, 1580, 1280, 2560
    c1 = _to_u8(s1[y0:y1, x0:x1])
    c2 = _to_u8(s12[y0:y1, x0:x1])
    diff = np.clip(np.abs(s12[y0:y1, x0:x1] - s1[y0:y1, x0:x1]) * 4.0, 0.0, 1.0)
    c3 = _to_u8(diff)
    canvas = Image.new("RGB", (1280 * 3, 960 + 60), (14, 18, 26))
    canvas.paste(Image.fromarray(c1), (0, 60))
    canvas.paste(Image.fromarray(c2), (1280, 60))
    canvas.paste(Image.fromarray(c3), (2560, 60))
    _draw_label(canvas, "Native 4K (3840x2160) 100% Crop: Stage 1 Analytic (Left) vs Stage 1 + M2_spatial_128 BF16 (Mid) vs 4x |Delta| (Right)", (20, 16))
    canvas.save(os.path.join(out_dir, "v3_w6_panda_4k_compare_crop.png"))
    print("Saved 4K Panda PNGs", flush=True)


def compose_complex_multicam_4k(vis_dir: str, out_dir: str):
  rt_p = os.path.join(vis_dir, "robotwin_bimanual_4k_data.npz")
  if not os.path.exists(rt_p):
    return
  ds = np.load(rt_p)

  # Top row: 4 x 960x1080 RGB tiles -> 3840x1080
  top_specs = [
      ("head_camera (RGB, RoboTwin Bimanual)", _to_u8(ds["head_rgb"])),
      ("diag_overview (RGB, Cluttered Tabletop)", _to_u8(ds["diag_rgb"])),
      ("left_wrist_camera (Ego RGB)", _to_u8(ds["lw_rgb"])),
      ("right_wrist_camera (Ego RGB)", _to_u8(ds["rw_rgb"])),
  ]

  # Bottom row: 4 x 960x1080 stacked pairs (each 960x540)
  def _stack_pair(top_arr, top_lbl, bot_arr, bot_lbl):
    t_im = Image.fromarray(_resize_u8(top_arr, 960, 540))
    _draw_label(t_im, top_lbl, (14, 12))
    b_im = Image.fromarray(_resize_u8(bot_arr, 960, 540))
    _draw_label(b_im, bot_lbl, (14, 12))
    col = Image.new("RGB", (960, 1080))
    col.paste(t_im, (0, 0))
    col.paste(b_im, (0, 540))
    return col

  bot_cols = [
      _stack_pair(
          _depth_vis(ds["head_depth"], 0.25, 2.2), "head_camera Depth (Turbo)",
          SEM_PALETTE_BIMANUAL[np.clip(ds["head_sem"], 0, 6)], "head_camera Semantic (L/R Arm, Obj, Bins, Tools)",
      ),
      _stack_pair(
          _to_u8(ds["front_rgb"]), "front_overview (RGB)",
          _to_u8(ds["top_rgb"]), "top_down (RGB)",
      ),
      _stack_pair(
          _depth_vis(ds["lw_depth"], 0.12, 1.5), "left_wrist Depth",
          SEM_PALETTE_BIMANUAL[np.clip(ds["lw_sem"], 0, 6)], "left_wrist Semantic",
      ),
      _stack_pair(
          _depth_vis(ds["rw_depth"], 0.12, 1.5), "right_wrist Depth",
          SEM_PALETTE_BIMANUAL[np.clip(ds["rw_sem"], 0, 6)], "right_wrist Semantic",
      ),
  ]

  canvas = Image.new("RGB", (3840, 2160), (12, 15, 22))
  for ci, (lbl, arr) in enumerate(top_specs):
    im = Image.fromarray(arr)
    _draw_label(im, lbl, (14, 14))
    canvas.paste(im, (ci * 960, 0))
  for ci, col_im in enumerate(bot_cols):
    canvas.paste(col_im, (ci * 960, 1080))

  # Thin grid lines
  draw = ImageDraw.Draw(canvas)
  for x in (960, 1920, 2880):
    draw.line([(x, 0), (x, 2160)], fill=(30, 36, 48), width=3)
  draw.line([(0, 1080), (3840, 1080)], fill=(30, 36, 48), width=3)

  out_p = os.path.join(out_dir, "v3_w6_complex_multicam_4k.png")
  canvas.save(out_p)
  print(f"Saved {out_p}", flush=True)


def compose_8k_hero(vis_dir: str, out_dir: str):
  d4 = os.path.join(vis_dir, "cluttered_8k_down4.npy")
  c4 = os.path.join(vis_dir, "cluttered_8k_crop_1280x960.npy")
  if not (os.path.exists(d4) and os.path.exists(c4)):
    d4 = os.path.join(vis_dir, "panda_8k_down4.npy")
    c4 = os.path.join(vis_dir, "panda_8k_crop_1280x960.npy")
  if not (os.path.exists(d4) and os.path.exists(c4)):
    return
  ov = _to_u8(np.load(d4))      # (1080, 1920, 3)
  crop = _to_u8(np.load(c4))    # (960, 1280, 3)
  crop_rs = _resize_u8(crop, 1440, 1080, resample=Image.NEAREST)
  canvas = Image.new("RGB", (1920 + 1440, 1080 + 70), (14, 18, 26))
  im_ov = Image.fromarray(ov)
  _draw_label(im_ov, "Full 8K UHD (7680x4320 = 33.18M px, 4x4 downscaled to 1920x1080)", (18, 16))
  im_cr = Image.fromarray(crop_rs)
  _draw_label(im_cr, "100% Native 8K Pixel Crop (1280x960 ROI)", (18, 16))
  canvas.paste(im_ov, (0, 70))
  canvas.paste(im_cr, (1920, 70))
  _draw_label(canvas, "TPU v5p 4-Chip Spatial-Sharded 8K UHD Stage 1 + M2_spatial_128 BF16 (Zero Host Callbacks)", (20, 16))
  out_p = os.path.join(out_dir, "v3_w6_8k_hero.png")
  canvas.save(out_p)
  print(f"Saved {out_p}", flush=True)


def _render_lidar_pointcloud_3d(xyz: np.ndarray, sem: np.ndarray, w: int = 1920, h: int = 1080) -> np.ndarray:
  """Software z-buffered 3D isometric point cloud projection for 262,144 LiDAR points."""
  pts = xyz.reshape(-1, 3)
  sids = sem.reshape(-1)
  valid = sids > 0
  pts = pts[valid]
  sids = sids[valid]

  # Filter within tabletop ROI radius <= 2.4m for crisp visualization
  r_xy = np.linalg.norm(pts[:, :2] - np.array([0.55, 0.0], dtype=np.float32), axis=-1)
  keep = (r_xy < 2.2) & (pts[:, 2] > -0.05) & (pts[:, 2] < 1.35)
  pts = pts[keep]
  sids = sids[keep]

  # Camera looking from (1.85, -1.55, 1.35) toward (0.55, -0.02, 0.22)
  cam_pos = np.array([1.85, -1.55, 1.35], dtype=np.float32)
  target = np.array([0.55, -0.02, 0.22], dtype=np.float32)
  fwd = target - cam_pos
  fwd /= np.linalg.norm(fwd)
  right = np.cross(fwd, np.array([0.0, 0.0, 1.0], dtype=np.float32))
  right /= np.linalg.norm(right)
  up = np.cross(right, fwd)

  rel = pts - cam_pos[None, :]
  zc = rel @ fwd
  xc = rel @ right
  yc = rel @ up
  mask = zc > 0.2
  xc, yc, zc, sids = xc[mask], yc[mask], zc[mask], sids[mask]

  f = 1.45 * h
  u = np.round(0.5 * w + f * (xc / zc)).astype(np.int32)
  v = np.round(0.5 * h - f * (yc / zc)).astype(np.int32)
  in_img = (u >= 1) & (u < w - 1) & (v >= 1) & (v < h - 1)
  u, v, zc, sids = u[in_img], v[in_img], zc[in_img], sids[in_img]

  # Sort far-to-near so nearer points overwrite farther points cleanly (2x2 splat)
  order = np.argsort(-zc)
  u, v, sids = u[order], v[order], sids[order]
  colors = SEM_PALETTE_LIDAR[np.clip(sids, 0, 5)]

  img = np.full((h, w, 3), 12, dtype=np.uint8)
  for du in (0, 1):
    for dv in (0, 1):
      img[v + dv, u + du] = colors
  return img


def compose_lidar_multicam_4k(vis_dir: str, out_dir: str):
  npz_p = os.path.join(vis_dir, "wave4_vis_data.npz")
  if not os.path.exists(npz_p):
    return
  ds = np.load(npz_p)

  canvas = Image.new("RGB", (3840, 2160), (12, 15, 22))

  # Top-left (1920x1080): Cluttered Tabletop Overview
  im_ov = Image.fromarray(_to_u8(ds["ov_rgb"]))
  _draw_label(im_ov, "Isaac Tabletop Scene + 360-deg LiDAR Rig (1920x1080 Pallas TPU)", (18, 16))
  canvas.paste(im_ov, (0, 0))

  # Top-right (1920x1080): 2x2 Stereo L, Stereo R, Depth, Semantic (960x540 each)
  st_l = Image.fromarray(_to_u8(ds["st_l_rgb"]))
  _draw_label(st_l, "Stereo Left (RGB)", (14, 12))
  st_r = Image.fromarray(_to_u8(ds["st_r_rgb"]))
  _draw_label(st_r, "Stereo Right (RGB)", (14, 12))
  st_d = Image.fromarray(_depth_vis(ds["st_l_depth"], 0.25, 2.2))
  _draw_label(st_d, "Stereo Depth (Turbo)", (14, 12))
  st_s = Image.fromarray(SEM_PALETTE_LIDAR[np.clip(ds["st_l_sem"], 0, 5)])
  _draw_label(st_s, "Stereo Semantic (0 Gather Ops)", (14, 12))

  canvas.paste(st_l, (1920, 0))
  canvas.paste(st_r, (2880, 0))
  canvas.paste(st_d, (1920, 540))
  canvas.paste(st_s, (2880, 540))

  # Bottom-left (1920x1080): 128x2048 Equirectangular Range + Semantic Panorama
  li_rng = ds["lidar_range"]  # (128, 2048)
  li_sem = ds["lidar_sem"]    # (128, 2048)
  valid_r = li_rng > 0.0
  norm_r = np.where(valid_r, 1.0 - np.clip((li_rng - 0.15) / 2.6, 0.0, 1.0), 0.0)
  pano_rng = _resize_u8(_turbo_colormap(norm_r, valid_r)[::-1], 1920, 540, resample=Image.NEAREST)
  pano_sem = _resize_u8(SEM_PALETTE_LIDAR[np.clip(li_sem, 0, 5)][::-1], 1920, 540, resample=Image.NEAREST)

  im_pr = Image.fromarray(pano_rng)
  _draw_label(im_pr, "128-Beam x 2048-Azimuth (262,144 rays) 360-deg LiDAR Range Panorama", (18, 14))
  im_ps = Image.fromarray(pano_sem)
  _draw_label(im_ps, "360-deg LiDAR Semantic Panorama (Floor | Robot | Targets | Bins | Tools)", (18, 14))
  canvas.paste(im_pr, (0, 1080))
  canvas.paste(im_ps, (0, 1620))

  # Bottom-right (1920x1080): 3D Projected LiDAR Point Cloud
  pc_img = _render_lidar_pointcloud_3d(ds["lidar_xyz"], li_sem, w=1920, h=1080)
  im_pc = Image.fromarray(pc_img)
  _draw_label(im_pc, "3D Projected LiDAR Point Cloud (262,144 pts/env, Semantic Colored)", (18, 16))
  canvas.paste(im_pc, (1920, 1080))

  draw = ImageDraw.Draw(canvas)
  draw.line([(1920, 0), (1920, 2160)], fill=(36, 44, 58), width=4)
  draw.line([(0, 1080), (3840, 1080)], fill=(36, 44, 58), width=4)

  out_p = os.path.join(out_dir, "v3_w6_lidar_multicam_4k.png")
  canvas.save(out_p)
  print(f"Saved {out_p}", flush=True)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--vis_dir", default="/dev/shm/zaychik-tpu-mjx-w6/out/vis")
  ap.add_argument("--out_dir", default="/dev/shm/zaychik-tpu-mjx-w6/out/figures")
  ap.add_argument("--which", nargs="+", default=["4k_panda", "multicam_4k", "8k_hero", "lidar_4k"])
  args = ap.parse_args()
  os.makedirs(args.out_dir, exist_ok=True)
  if "4k_panda" in args.which:
    compose_4k_panda(args.vis_dir, args.out_dir)
  if "multicam_4k" in args.which:
    compose_complex_multicam_4k(args.vis_dir, args.out_dir)
  if "8k_hero" in args.which:
    compose_8k_hero(args.vis_dir, args.out_dir)
  if "lidar_4k" in args.which:
    compose_lidar_multicam_4k(args.vis_dir, args.out_dir)


if __name__ == "__main__":
  main()
