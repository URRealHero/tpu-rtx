"""Wave 2 Benchmark & 4K Render Export: 100k -> 1M -> 10M Gaussians on TPU v5p (1c & 4c)."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from typing import Any, Dict, List

import jax
import jax.numpy as jnp
import numpy as np

from tpu_mjx_render.roofline import inspect_hlo_ops
from gs3d_wave2_10m import (
    build_massive_scene_arrays,
    make_massive_scene_4c_fn,
    render_massive_scene_1c,
)


def _check_hlo(lowered) -> Dict[str, Any]:
  txt = lowered.as_text()
  info = inspect_hlo_ops(txt)
  info["send_recv_ops"] = len(re.findall(r"\bsend\(|\brecv\(", txt, re.I))
  assert info["host_callbacks"] == 0 and info["send_recv_ops"] == 0, info
  info["zero_copy_verified"] = True
  return info


def _bench_fn(fn, args, n_warmup: int = 2, n_steps: int = 5):
  t0 = time.perf_counter()
  out = fn(*args)
  jax.block_until_ready(out)
  compile_s = time.perf_counter() - t0
  for _ in range(n_warmup):
    out = fn(*args)
  jax.block_until_ready(out)
  t1 = time.perf_counter()
  for _ in range(n_steps):
    out = fn(*args)
  jax.block_until_ready(out)
  ms = (time.perf_counter() - t1) * 1e3 / n_steps
  return ms, compile_s, out


def _save_png(rgb_hwc: np.ndarray, path: str) -> None:
  try:
    from PIL import Image
    arr = np.clip(np.round(rgb_hwc * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)
  except ImportError:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.imsave(path, np.clip(rgb_hwc, 0.0, 1.0))


def run_wave2_1c(out_json: str, data: Dict[str, Any], export_dir: str, n_list: List[int]) -> None:
  configs_res = [
      ("256x256", 256, 256),
      ("512x512", 512, 512),
      ("1080p", 1920, 1080),
      ("4K", 3840, 2160),
  ]
  for n_g in n_list:
    sc = build_massive_scene_arrays(n_g, seed=42)
    mu_d = jnp.asarray(sc["mu"])
    s2_d = jnp.asarray(sc["scale2"])
    rgb_d = jnp.asarray(sc["rgb"])
    op_d = jnp.asarray(sc["opacity"])
    cxp_d = jnp.asarray(sc["cam_xpos"])
    cxm_d = jnp.asarray(sc["cam_xmat"])
    fovy = float(sc["cam_fovy"])
    znear = float(sc["znear"])
    args = (mu_d, s2_d, rgb_d, op_d, cxp_d, cxm_d)

    for res_name, W, H in configs_res:
      for use_bf16 in (False, True):
        mode_tag = "v3_macro_bf16_mxu" if use_bf16 else "v3_macro_fp32"
        fn = jax.jit(
            lambda a, b, c, d, e, f: render_massive_scene_1c(
                a, b, c, d, e, f, height=H, width=W, cam_fovy=fovy, znear=znear,
                k_macro=2048, use_bf16_mxu=use_bf16, interpret=False,
            )
        )
        if n_g == 10_000_000 and res_name == "4K" and not use_bf16:
          data.setdefault("hlo_checks", {})["1c_10m_4k_fp32"] = _check_hlo(fn.lower(*args))

        n_steps = 5 if res_name in ("256x256", "512x512") else 3
        ms, comp_s, img = _bench_fn(fn, args, n_warmup=2, n_steps=n_steps)
        fps = 1000.0 / ms
        rec = {
            "n_chips": 1,
            "n_gaussians": n_g,
            "res_label": res_name,
            "res": f"{W}x{H}",
            "k_macro": 2048,
            "pipeline": mode_tag,
            "ms": round(ms, 2),
            "fps": round(fps, 1),
            "mega_gaussians_per_s": round((n_g * fps) / 1e6, 2),
            "compile_s": round(comp_s, 2),
            "mean_rgb": round(float(jnp.mean(img)), 4),
        }
        key = (rec["n_chips"], rec["n_gaussians"], rec["res"], rec["pipeline"])
        data["runs"] = [
            r for r in data.get("runs", [])
            if (r.get("n_chips"), r.get("n_gaussians"), r.get("res"), r.get("pipeline")) != key
        ] + [rec]
        with open(out_json, "w") as f:
          json.dump(data, f, indent=2)
        print("WAVE2 1C:", rec, flush=True)

        # Save Native 4K 10M & 100k renders for visual figure
        if res_name == "4K" and not use_bf16 and n_g in (100_000, 10_000_000):
          png_path = os.path.join(export_dir, f"render_4k_{n_g}.png")
          _save_png(np.asarray(img), png_path)
          print(f"Saved 4K render: {png_path}", flush=True)

    # Free device buffers cleanly before next N
    del mu_d, s2_d, rgb_d, op_d


def run_wave2_4c(out_json: str, data: Dict[str, Any], n_list: List[int]) -> None:
  devs = jax.devices()[:4]
  assert len(devs) == 4, f"Expected 4 chips, got {devs}"
  mesh = jax.sharding.Mesh(np.array(devs), ("dp",))
  rep = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

  configs_res = [
      ("256x256", 256, 256),
      ("512x512", 512, 512),
      ("1080p", 1920, 1080),
      ("4K", 3840, 2160),
  ]
  for n_g in n_list:
    sc = build_massive_scene_arrays(n_g, seed=42)
    args = tuple(jax.device_put(jnp.asarray(sc[k]), rep) for k in ("mu", "scale2", "rgb", "opacity", "cam_xpos", "cam_xmat"))
    fovy = float(sc["cam_fovy"])
    znear = float(sc["znear"])

    for res_name, W, H in configs_res:
      for use_bf16 in (False, True):
        mode_tag = "v3_macro_bf16_mxu" if use_bf16 else "v3_macro_fp32"
        fn = make_massive_scene_4c_fn(
            mesh, height=H, width=W, cam_fovy=fovy, znear=znear,
            k_macro=2048, use_bf16_mxu=use_bf16,
        )
        if n_g == 10_000_000 and res_name == "4K" and not use_bf16:
          data.setdefault("hlo_checks", {})["4c_10m_4k_fp32"] = _check_hlo(fn.lower(*args))

        n_steps = 5 if res_name in ("256x256", "512x512") else 3
        ms, comp_s, img = _bench_fn(fn, args, n_warmup=2, n_steps=n_steps)
        fps = 1000.0 / ms
        rec = {
            "n_chips": 4,
            "n_gaussians": n_g,
            "res_label": res_name,
            "res": f"{W}x{H}",
            "k_macro": 2048,
            "pipeline": mode_tag,
            "ms": round(ms, 2),
            "fps": round(fps, 1),
            "mega_gaussians_per_s": round((n_g * fps) / 1e6, 2),
            "compile_s": round(comp_s, 2),
            "mean_rgb": round(float(jnp.mean(img)), 4),
        }
        key = (rec["n_chips"], rec["n_gaussians"], rec["res"], rec["pipeline"])
        data["runs"] = [
            r for r in data.get("runs", [])
            if (r.get("n_chips"), r.get("n_gaussians"), r.get("res"), r.get("pipeline")) != key
        ] + [rec]
        with open(out_json, "w") as f:
          json.dump(data, f, indent=2)
        print("WAVE2 4C:", rec, flush=True)

    del args


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--phase", required=True, choices=["1c", "4c"])
  ap.add_argument("--n_list", type=int, nargs="+", default=[100_000, 1_000_000, 10_000_000])
  ap.add_argument("--out_json", default="/dev/shm/zaychik-tpu-mjx-w3/out/v3_w3_wave2_10m_gs3d_4k.json")
  ap.add_argument("--export_dir", default="/dev/shm/zaychik-tpu-mjx-w3/out")
  args = ap.parse_args()

  os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
  os.makedirs(args.export_dir, exist_ok=True)
  if os.path.exists(args.out_json):
    with open(args.out_json) as f:
      data = json.load(f)
  else:
    data = {
        "backend": jax.default_backend(),
        "device_kind": jax.devices()[0].device_kind,
        "jax_version": jax.__version__,
        "runs": [],
    }

  if args.phase == "1c":
    run_wave2_1c(args.out_json, data, args.export_dir, args.n_list)
  elif args.phase == "4c":
    run_wave2_4c(args.out_json, data, args.n_list)


if __name__ == "__main__":
  main()
