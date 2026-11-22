"""Wave 2 Final Benchmark & Native 4K Sponza Showcase: 100k -> 1M -> 10M Anisotropic 3DGS on TPU v5p (1c & 4c)."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from typing import Any, Dict, List

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from tpu_mjx_render.roofline import inspect_hlo_ops
from gltf_to_3dgs import load_gltf_triangles, look_at_cam_mat, sample_anisotropic_3dgs_from_gltf
from gs3d_wave2_10m import make_massive_scene_4c_fn, render_massive_scene_1c


SPONZA_GLTF = "/dev/shm/zaychik-tpu-mjx-w3/assets/gltf-assets/Models/Sponza/glTF/Sponza.gltf"


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
  arr = np.clip(np.round(rgb_hwc * 255.0), 0, 255).astype(np.uint8)
  Image.fromarray(arr).save(path)


def run_phase(
    phase: str,
    n_list: List[int],
    out_json: str,
    export_dir: str,
    k_macro: int,
) -> None:
  os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
  os.makedirs(export_dir, exist_ok=True)
  if os.path.exists(out_json):
    with open(out_json) as f:
      data = json.load(f)
  else:
    data = {
        "scene": "KhronosGroup/glTF-Sample-Assets/Models/Sponza",
        "backend": jax.default_backend(),
        "device_kind": jax.devices()[0].device_kind,
        "jax_version": jax.__version__,
        "runs": [],
        "hlo_checks": {},
    }

  sc = load_gltf_triangles(SPONZA_GLTF)
  cam_pos = np.array([-7.2, 2.3, -1.1], dtype=np.float32)
  target = np.array([7.5, 3.8, 0.5], dtype=np.float32)
  cam_mat = look_at_cam_mat(cam_pos, target, world_up=(0.0, 1.0, 0.0))
  fovy, znear = 52.0, 0.25

  data["scene_stats"] = {
      "n_triangles": int(len(sc["v0"])),
      "n_materials": int(len(sc["materials"])),
      "bbox_min": [round(float(x), 4) for x in sc["bbox_min"]],
      "bbox_max": [round(float(x), 4) for x in sc["bbox_max"]],
  }

  configs_res = [
      ("256x256", 256, 256),
      ("512x512", 512, 512),
      ("1080p", 1920, 1080),
      ("4K", 3840, 2160),
  ]

  if phase == "4c":
    devs = jax.devices()[:4]
    assert len(devs) == 4
    mesh = jax.sharding.Mesh(np.array(devs), ("dp",))
    rep = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    n_chips = 4
  else:
    mesh = None
    rep = None
    n_chips = 1

  for n_g in n_list:
    gs = sample_anisotropic_3dgs_from_gltf(sc, n_g, seed=42)
    if n_chips == 4:
      args = tuple(
          jax.device_put(jnp.asarray(x), rep)
          for x in (gs["mu"], gs["cov6"], gs["rgb"], gs["opacity"], cam_pos, cam_mat, gs["nrm"])
      )
    else:
      args = tuple(
          jnp.asarray(x)
          for x in (gs["mu"], gs["cov6"], gs["rgb"], gs["opacity"], cam_pos, cam_mat, gs["nrm"])
      )

    for res_name, W, H in configs_res:
      for use_bf16 in (False, True):
        mode_tag = "v3_macro_bf16_mxu" if use_bf16 else "v3_macro_fp32"
        if n_chips == 4:
          fn = make_massive_scene_4c_fn(
              mesh, height=H, width=W, cam_fovy=fovy, znear=znear,
              k_macro=k_macro, use_bf16_mxu=use_bf16, has_nrm=True,
          )
        else:
          fn = jax.jit(
              lambda a, b, c, d, e, f, g: render_massive_scene_1c(
                  a, b, c, d, e, f, height=H, width=W, nrm=g, cam_fovy=fovy, znear=znear,
                  k_macro=k_macro, use_bf16_mxu=use_bf16, interpret=False,
              )
          )

        if n_g == 10_000_000 and res_name == "4K" and not use_bf16:
          data["hlo_checks"][f"{n_chips}c_10m_4k_fp32_k{k_macro}"] = _check_hlo(fn.lower(*args))

        n_steps = 5 if res_name in ("256x256", "512x512") else 3
        ms, comp_s, img = _bench_fn(fn, args, n_warmup=2, n_steps=n_steps)
        fps = 1000.0 / ms
        rec = {
            "n_chips": n_chips,
            "scene": "Sponza",
            "n_gaussians": n_g,
            "splat_radius_m": round(float(gs["splat_radius"]), 5),
            "res_label": res_name,
            "res": f"{W}x{H}",
            "k_macro": k_macro,
            "pipeline": mode_tag,
            "ms": round(ms, 2),
            "fps": round(fps, 1),
            "mega_gaussians_per_s": round((n_g * fps) / 1e6, 2),
            "compile_s": round(comp_s, 2),
            "mean_rgb": round(float(jnp.mean(img)), 4),
        }
        key = (rec["n_chips"], rec["n_gaussians"], rec["res"], rec["k_macro"], rec["pipeline"])
        data["runs"] = [
            r for r in data.get("runs", [])
            if (r.get("n_chips"), r.get("n_gaussians"), r.get("res"), r.get("k_macro"), r.get("pipeline")) != key
        ] + [rec]
        with open(out_json, "w") as f:
          json.dump(data, f, indent=2)
        print(f"WAVE2 {n_chips}C:", rec, flush=True)

        if n_chips == 1 and res_name == "4K" and not use_bf16 and n_g == 10_000_000:
          png_path = os.path.join(export_dir, "v3_w3_gs3d_4k.png")
          _save_png(np.asarray(img), png_path)
          print(f"Saved Native 4K Sponza 10M render: {png_path}", flush=True)

    del args


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--phase", required=True, choices=["1c", "4c"])
  ap.add_argument("--n_list", type=int, nargs="+", default=[100_000, 1_000_000, 10_000_000])
  ap.add_argument("--k_macro", type=int, default=4096)
  ap.add_argument("--out_json", default="/dev/shm/zaychik-tpu-mjx-w3/out/v3_w3_wave2_10m_gs3d_4k.json")
  ap.add_argument("--export_dir", default="/dev/shm/zaychik-tpu-mjx-w3/out")
  args = ap.parse_args()
  run_phase(args.phase, args.n_list, args.out_json, args.export_dir, args.k_macro)


if __name__ == "__main__":
  main()
