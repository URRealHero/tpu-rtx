"""Shared helpers for Wave 7 Native 8K Real-World 3DGS & Fast 32-View jax.grad Benchmarks."""

from __future__ import annotations

import re
from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tpu_mjx_render.roofline import inspect_hlo_ops


def check_hlo(lowered: Any, name: str) -> Dict[str, Any]:
  """Verify lowered XLA HLO emits zero host callbacks, zero send/recv ops, and zero gathers."""
  txt = lowered.as_text()
  info = inspect_hlo_ops(txt)
  info["send_recv_ops"] = len(re.findall(r"\bsend\(|\brecv\(", txt, re.I))
  assert info["host_callbacks"] == 0, f"{name} host_callbacks != 0: {info}"
  assert info["send_recv_ops"] == 0, f"{name} send_recv_ops != 0: {info}"
  assert info["gather_ops"] == 0, f"{name} gather_ops != 0: {info}"
  info["zero_copy_verified"] = True
  return info


def scaled_K(cam: Dict[str, Any], width: int, height: int) -> np.ndarray:
  """Scale camera intrinsics from native capture resolution to `(width, height)`."""
  sf = float(height) / float(cam["height"])
  return np.array(
      [
          [float(cam["fx"]) * sf, 0.0, 0.5 * width],
          [0.0, float(cam["fy"]) * sf, 0.5 * height],
          [0.0, 0.0, 1.0],
      ],
      dtype=np.float32,
  )


def save_png(rgb_hwc: np.ndarray, path: str) -> None:
  """Save a float32 `[0, 1]` HWC image to uint8 PNG."""
  arr = np.clip(np.round(rgb_hwc * 255.0), 0, 255).astype(np.uint8)
  Image.fromarray(arr).save(path)


def add_label(
    img_pil: Image.Image,
    title: str,
    subtitle: str,
    accent: Tuple[int, int, int, int] = (255, 165, 60, 220),
) -> Image.Image:
  """Draw a high-contrast HUD badge onto a PIL image."""
  out = img_pil.copy()
  draw = ImageDraw.Draw(out, "RGBA")
  try:
    font_b = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
    font_s = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 21)
  except Exception:
    font_b = ImageFont.load_default()
    font_s = font_b
  pad_x, pad_y = 22, 18
  tw = max(draw.textlength(title, font=font_b), draw.textlength(subtitle, font=font_s))
  box = [pad_x, pad_y, pad_x + int(tw) + 34, pad_y + 80]
  draw.rounded_rectangle(box, radius=12, fill=(10, 14, 20, 205), outline=accent, width=2)
  draw.text((pad_x + 16, pad_y + 10), title, fill=(255, 255, 255, 255), font=font_b)
  draw.text((pad_x + 16, pad_y + 45), subtitle, fill=(215, 225, 240, 255), font=font_s)
  return out
