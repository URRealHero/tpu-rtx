"""Native 8K (7680x4320, 33.18M px) Real-World Mip-NeRF 360 Garden 3DGS Benchmark on 4-Chip TPU v5p."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from typing import Any, Dict, List

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(__file__))
from common import add_label, check_hlo, save_png, scaled_K

from tpu_mjx_render.adapters.tpu_gsplat import (
    load_cameras_json,
    load_ply,
    make_sharded_rasterizer,
    shard_splats_for_mesh,
)


def _measure_compiled(comp, args, n_warmup: int = 1, n_steps: int = 4):
  for _ in range(n_warmup):
    out = comp(*args)
    jax.block_until_ready(out)
  t0 = time.perf_counter()
  for _ in range(n_steps):
    out = comp(*args)
  jax.block_until_ready(out)
  ms = (time.perf_counter() - t0) * 1e3 / n_steps
  return ms, out


def _build_8k_crop_showcase(
    rgb_8k_uint8: np.ndarray,
    out_path: str,
    ms_8k: float,
    n_gaussians: int,
    crop_center_xy: tuple[int, int] = (3840, 1520),
    crop_wh: tuple[int, int] = (1920, 1080),
) -> Dict[str, int]:
  """Build a 3840x1080 side-by-side: 1080p full-frame overview + 1:1 native 8K pixel crop."""
  H, W, _ = rgb_8k_uint8.shape
  cw, ch = crop_wh
  cx, cy = crop_center_xy
  x0 = max(0, min(W - cw, cx - cw // 2))
  y0 = max(0, min(H - ch, cy - ch // 2))
  x1, y1 = x0 + cw, y0 + ch

  full_pil = Image.fromarray(rgb_8k_uint8)
  crop_pil = full_pil.crop((x0, y0, x1, y1))

  # Left pane: full 8K image cleanly downsampled 4x to 1920x1080 with exact crop box overlay
  overview = full_pil.resize((1920, 1080), Image.Resampling.LANCZOS)
  ov_draw = ImageDraw.Draw(overview, "RGBA")
  sx, sy = 1920.0 / float(W), 1080.0 / float(H)
  rx0, ry0 = int(round(x0 * sx)), int(round(y0 * sy))
  rx1, ry1 = int(round(x1 * sx)), int(round(y1 * sy))
  ov_draw.rectangle([rx0, ry0, rx1, ry1], outline=(0, 235, 255, 255), width=4)
  ov_draw.rectangle([rx0 + 4, ry0 + 4, rx1 - 4, ry1 - 4], outline=(255, 255, 255, 180), width=1)
  # Connector callout lines to right edge
  ov_draw.line([(rx1, ry0), (1919, 0)], fill=(0, 235, 255, 180), width=2)
  ov_draw.line([(rx1, ry1), (1919, 1079)], fill=(0, 235, 255, 180), width=2)

  mpix_s = (W * H / 1e6) / (ms_8k / 1e3)
  left_pane = add_label(
      overview,
      f"Native 8K Full Frame ({W}x{H}, 33.18M px)",
      f"Mip-NeRF 360 Garden | {n_gaussians:,} Gaussians (SH3) | 4-Chip TPU v5p: {ms_8k:.1f} ms ({mpix_s:.1f} MPix/s)",
      accent=(255, 165, 60, 230),
  )
  right_pane = add_label(
      crop_pil,
      "1:1 Native 8K Pixel Crop (Zero Upscaling)",
      f"ROI [{x0}:{x1}, {y0}:{y1}] ({cw}x{ch}) | Mosaic Pallas V3 (host_callbacks=0, gather_ops=0)",
      accent=(0, 235, 255, 230),
  )

  canvas = Image.new("RGB", (3840, 1080))
  canvas.paste(left_pane, (0, 0))
  canvas.paste(right_pane, (1920, 0))
  # Crisp vertical divider
  div_draw = ImageDraw.Draw(canvas)
  div_draw.line([(1920, 0), (1920, 1079)], fill=(0, 235, 255), width=3)
  os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
  canvas.save(out_path)
  return {"x0": x0, "y0": y0, "x1": x1, "y1": y1}


def run_8k_benchmark(
    data_root: str,
    out_json: str,
    export_dir: str,
    save_full_8k: bool = False,
) -> Dict[str, Any]:
  os.makedirs(export_dir, exist_ok=True)
  os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)

  devs = jax.devices()[:4]
  mesh = jax.sharding.Mesh(np.array(devs), ("dp",))
  rep = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

  ply_p = os.path.join(data_root, "garden/point_cloud/iteration_30000/point_cloud.ply")
  cam_p = os.path.join(data_root, "garden/cameras.json")
  print(f"Loading {ply_p} ...", flush=True)
  t_load = time.perf_counter()
  sp = load_ply(ply_p, max_sh_degree=3)
  cams = load_cameras_json(cam_p)
  c0 = cams[0]
  print(f"Loaded {sp['n_gaussians']:,} Gaussians (SH{sp['sh_degree']}) in {time.perf_counter() - t_load:.2f}s", flush=True)

  m, q, s, o, sh, n_pad = shard_splats_for_mesh(sp, mesh, seed=42, use_sh=True)
  v_dev = jax.device_put(jnp.asarray(c0["viewmat"]), rep)

  if os.path.exists(out_json):
    with open(out_json) as f:
      report = json.load(f)
  else:
    report = {
        "backend": jax.default_backend(),
        "device_kind": jax.devices()[0].device_kind,
        "jax_version": jax.__version__,
        "interpret": False,
        "hlo_checks": {},
    }

  # Resolution sweep on real Mip-NeRF 360 Garden (5.83M Gaussians, SH3, 4 chips):
  # Note: macro_size=64 gives M_tiles=8160 < 8191 at Native 8K (7680x4320)!
  configs = [
      ("4K", 3840, 2160, 64, 4, 8192, 2048, 512),
      ("8K", 7680, 4320, 64, 5, 8192, 2048, 512),
  ]
  res_records: List[Dict[str, Any]] = []
  crop_info = None
  crop_png_path = os.path.join(export_dir, "v3_w3_realworld_8k_crop.png")

  for res_label, W, H, m_sz, m_rep, k_mac, k_lg, k_tile in configs:
    K_mat = scaled_K(c0, W, H)
    k_dev = jax.device_put(jnp.asarray(K_mat), rep)
    fn = make_sharded_rasterizer(
        mesh, W, H, sh_degree=3, macro_size=m_sz, macro_rep=m_rep,
        k_macro=k_mac, k_large_per_chip=k_lg, k_tile=k_tile,
    )
    low = fn.lower(m, q, s, o, sh, v_dev, k_dev)
    hlo = check_hlo(low, f"garden_{res_label}")
    report["hlo_checks"][f"garden_{res_label.lower()}"] = hlo

    t_c = time.perf_counter()
    comp = low.compile()
    mem = comp.memory_analysis()
    arg_gb = float(mem.argument_size_in_bytes) / 1e9
    temp_gb = float(mem.temp_size_in_bytes) / 1e9
    out_gb = float(mem.output_size_in_bytes) / 1e9
    alias_gb = float(getattr(mem, "alias_size_in_bytes", 0)) / 1e9
    chip_hbm_gb = arg_gb + temp_gb + out_gb - alias_gb

    out0 = comp(m, q, s, o, sh, v_dev, k_dev)
    jax.block_until_ready(out0)
    compile_s = time.perf_counter() - t_c

    ms, out = _measure_compiled(comp, (m, q, s, o, sh, v_dev, k_dev), n_warmup=1, n_steps=4)
    rgb, alp, _ = out
    rgb_np = np.asarray(rgb)
    alp_np = np.asarray(alp[..., 0])
    mpix = (W * H) / 1e6
    mpix_s = mpix / (ms / 1e3)
    fps = 1000.0 / ms

    rec = {
        "scene": "garden",
        "res_label": res_label,
        "res": f"{W}x{H}",
        "pixels_m": round(mpix, 2),
        "n_gaussians": int(sp["n_gaussians"]),
        "n_pad": int(n_pad),
        "sh_degree": int(sp["sh_degree"]),
        "n_chips": 4,
        "macro_size": m_sz,
        "macro_rep": m_rep,
        "k_macro": k_mac,
        "k_large_per_chip": k_lg,
        "k_tile": k_tile,
        "ms": round(ms, 2),
        "fps": round(fps, 3),
        "mpix_per_s": round(mpix_s, 2),
        "compile_s": round(compile_s, 2),
        "hbm_per_chip_gb": round(chip_hbm_gb, 2),
        "hbm_temp_per_chip_gb": round(temp_gb, 2),
        "mean_alpha": round(float(alp_np.mean()), 4),
        "low_alp_frac": round(float(np.mean(alp_np < 0.5)), 5),
    }
    res_records.append(rec)
    print(
        f"GARDEN {res_label} ({W}x{H}, {mpix:.2f}M px): {ms:.2f} ms ({fps:.2f} FPS, {mpix_s:.2f} MPix/s) "
        f"| HBM={chip_hbm_gb:.2f} GB/chip | mean_alp={rec['mean_alpha']:.4f} | low_alp={rec['low_alp_frac']:.5f}",
        flush=True,
    )

    if res_label == "8K":
      rgb_u8 = np.clip(np.round(rgb_np * 255.0), 0, 255).astype(np.uint8)
      crop_info = _build_8k_crop_showcase(
          rgb_u8, crop_png_path, ms_8k=ms, n_gaussians=int(sp["n_gaussians"])
      )
      print(f"Saved 8K side-by-side crop showcase -> {crop_png_path} (crop={crop_info})", flush=True)
      if save_full_8k:
        full_8k_path = os.path.join(export_dir, "v3_w3_realworld_garden_8k.png")
        Image.fromarray(rgb_u8).save(full_8k_path)
        print(f"Saved full 8K PNG -> {full_8k_path}", flush=True)
      del rgb_u8

    del rgb_np, alp_np, out, out0, comp, low
    gc.collect()

  report["native_8k_garden"] = {
      "records": res_records,
      "crop_showcase_path": crop_png_path,
      "crop_roi_8k": crop_info,
  }
  with open(out_json, "w") as f:
    json.dump(report, f, indent=2)
  print(f"Updated metrics JSON -> {out_json}", flush=True)
  return report


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--data_root", default="/dev/shm/zaychik-w3-v7/data")
  ap.add_argument("--out_json", default="/dev/shm/zaychik-w3-v7/out/v3_w3_wave7_8k_and_gradtrain.json")
  ap.add_argument("--export_dir", default="/dev/shm/zaychik-w3-v7/out")
  ap.add_argument("--save_full_8k", action="store_true")
  args = ap.parse_args()
  run_8k_benchmark(args.data_root, args.out_json, args.export_dir, save_full_8k=args.save_full_8k)


if __name__ == "__main__":
  main()
