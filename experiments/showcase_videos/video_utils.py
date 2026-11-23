"""Shared camera, geometry, and frame-export helpers for 8-host TPU showcase videos."""

from __future__ import annotations

import math
import os
import subprocess
from typing import Iterable, Tuple

import numpy as np
from PIL import Image


def smoothstep(t: float) -> float:
  x = max(0.0, min(1.0, float(t)))
  return x * x * (3.0 - 2.0 * x)


def look_at_mjx_cmat(eye: Iterable[float], target: Iterable[float], world_up=(0.0, 0.0, 1.0)) -> np.ndarray:
  """Return MuJoCo 3x3 cam_xmat with columns [right, up, -forward]."""
  e = np.asarray(eye, dtype=np.float64)
  t = np.asarray(target, dtype=np.float64)
  up_w = np.asarray(world_up, dtype=np.float64)
  fwd = t - e
  fwd /= max(np.linalg.norm(fwd), 1e-12)
  right = np.cross(fwd, up_w)
  if np.linalg.norm(right) < 1e-6:
    right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
  right /= max(np.linalg.norm(right), 1e-12)
  up = np.cross(right, fwd)
  up /= max(np.linalg.norm(up), 1e-12)
  # MuJoCo camera looks along -Z, +X right, +Y up
  return np.column_stack([right, up, -fwd]).astype(np.float32)


def look_at_opencv(eye: Iterable[float], target: Iterable[float], world_up=(0.0, 1.0, 0.0)) -> Tuple[np.ndarray, np.ndarray]:
  """Return OpenCV 4x4 viewmat (world-to-cam) and 3x3 c2w [right, down, forward]."""
  e = np.asarray(eye, dtype=np.float64)
  t = np.asarray(target, dtype=np.float64)
  up_w = np.asarray(world_up, dtype=np.float64)
  fwd = t - e
  fwd /= max(np.linalg.norm(fwd), 1e-12)
  # Right = fwd x world_up; down = fwd x right (so -up)
  right = np.cross(fwd, up_w)
  if np.linalg.norm(right) < 1e-6:
    right = np.cross(fwd, np.array([1.0, 0.0, 0.0], dtype=np.float64))
  right /= max(np.linalg.norm(right), 1e-12)
  up = np.cross(right, fwd)
  up /= max(np.linalg.norm(up), 1e-12)
  down = -up
  c2w = np.column_stack([right, down, fwd]).astype(np.float32)
  R_wc = c2w.T
  t_wc = -(R_wc @ e.astype(np.float32))
  V = np.eye(4, dtype=np.float32)
  V[:3, :3] = R_wc
  V[:3, 3] = t_wc
  return V, c2w


def fit_mipnerf_orbit_basis(cams: list) -> dict:
  """Estimate right-side-up orbital ring (target, up, u_axis, v_axis, radius, height, c0_angle) from MipNeRF 360 cameras."""
  positions = np.stack([np.asarray(c["position"], dtype=np.float64) for c in cams], axis=0)
  rots = np.stack([np.asarray(c["rotation"], dtype=np.float64) for c in cams], axis=0)

  # OpenCV camera columns are [right, down, forward]; world_up ~ mean(-down)
  ups = -rots[:, :, 1]
  up_world = np.mean(ups, axis=0)
  up_world /= max(np.linalg.norm(up_world), 1e-12)

  # Intersect viewing rays via least squares: sum(I - d d^T) p = sum(I - d d^T) o
  fwds = rots[:, :, 2]
  A = np.zeros((3, 3), dtype=np.float64)
  b = np.zeros((3,), dtype=np.float64)
  I3 = np.eye(3, dtype=np.float64)
  for o, d in zip(positions, fwds):
    d = d / max(np.linalg.norm(d), 1e-12)
    M = I3 - np.outer(d, d)
    A += M
    b += M @ o
  target = np.linalg.solve(A + 1e-4 * I3, b)

  rel = positions - target[None, :]
  heights = rel @ up_world
  rel_plane = rel - heights[:, None] * up_world[None, :]
  radii = np.linalg.norm(rel_plane, axis=-1)

  # Anchor azimuth=0 cleanly at cam[0] so video starts from a verified clean training viewpoint!
  r0_vec = rel_plane[0]
  if np.linalg.norm(r0_vec) < 1e-4:
    r0_vec = np.cross(up_world, np.array([1.0, 0.0, 0.0], dtype=np.float64))
  u_axis = r0_vec / max(np.linalg.norm(r0_vec), 1e-12)
  v_axis = np.cross(up_world, u_axis)
  v_axis /= max(np.linalg.norm(v_axis), 1e-12)

  # Guard against outlier cameras with extreme radius/height
  r_med = float(np.median(radii))
  h_med = float(heights[0]) * 0.45 + float(np.median(heights)) * 0.55
  return {
      "target": target,
      "up_world": up_world,
      "u_axis": u_axis,
      "v_axis": v_axis,
      "radius": r_med,
      "height": h_med,
      "c0": cams[0],
  }


def save_frames_tar(frames_u8: list[np.ndarray], out_dir: str, clip_name: str) -> str:
  """Write PNG (or high-quality JPEG for >=1920px) frames and bundle as `<out_dir>/<clip_name>.tar`."""
  os.makedirs(out_dir, exist_ok=True)
  frame_dir = os.path.join(out_dir, f"_frames_{clip_name}")
  os.makedirs(frame_dir, exist_ok=True)
  use_jpg = bool(frames_u8 and frames_u8[0].shape[1] >= 1920)
  for idx, arr in enumerate(frames_u8):
    if arr.dtype != np.uint8:
      arr = np.clip(np.round(arr * 255.0), 0, 255).astype(np.uint8)
    im = Image.fromarray(arr)
    if use_jpg:
      im.save(os.path.join(frame_dir, f"f_{idx:04d}.jpg"), quality=96, subsampling=0)
    else:
      im.save(os.path.join(frame_dir, f"f_{idx:04d}.png"), compress_level=1)
  tar_path = os.path.join(out_dir, f"{clip_name}.tar")
  subprocess.run(["tar", "-cf", tar_path, "-C", out_dir, f"_frames_{clip_name}"], check=True)
  subprocess.run(["rm", "-rf", frame_dir], check=True)
  fmt = "JPG-Q96" if use_jpg else "PNG"
  print(f"[Export] Saved {len(frames_u8)} {fmt} frames -> {tar_path}", flush=True)
  return tar_path
