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
    k_tile: int = 128,
    macro_rep: int = 3,
    splat_scale: float = 1.0,
    thick_ratio: float = 0.14,
    export_hq_4k: bool = False,
    res_filter: List[str] | None = None,
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
    gs = sample_anisotropic_3dgs_from_gltf(sc, n_g, seed=42, splat_scale=splat_scale, thick_ratio=thick_ratio)
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
      if res_filter and res_name not in res_filter:
        continue
      for use_bf16 in (False, True):
        mode_tag = "v3_macro_bf16_mxu" if use_bf16 else "v3_macro_fp32"
        if k_tile > 128:
          mode_tag = f"{mode_tag}_k{k_tile}"
        if n_chips == 4:
          fn = make_massive_scene_4c_fn(
              mesh, height=H, width=W, cam_fovy=fovy, znear=znear,
              k_macro=k_macro, use_bf16_mxu=use_bf16, has_nrm=True,
              k_tile=k_tile, macro_rep=macro_rep,
          )
        else:
          fn = jax.jit(
              lambda a, b, c, d, e, f, g: render_massive_scene_1c(
                  a, b, c, d, e, f, height=H, width=W, nrm=g, cam_fovy=fovy, znear=znear,
                  k_macro=k_macro, use_bf16_mxu=use_bf16, k_tile=k_tile, macro_rep=macro_rep, interpret=False,
              )
          )

        if n_g == 10_000_000 and res_name == "4K" and not use_bf16:
          data.setdefault("hlo_checks", {})[f"{n_chips}c_10m_4k_fp32_km{k_macro}_kt{k_tile}"] = _check_hlo(fn.lower(*args))

        n_steps = 5 if res_name in ("256x256", "512x512") else 3
        ms, comp_s, img = _bench_fn(fn, args, n_warmup=2, n_steps=n_steps)
        fps = 1000.0 / ms
        rec = {
            "n_chips": n_chips,
            "scene": "Sponza",
            "n_gaussians": n_g,
            "splat_radius_m": round(float(gs["splat_radius"]), 5),
            "splat_scale": splat_scale,
            "res_label": res_name,
            "res": f"{W}x{H}",
            "k_macro": k_macro,
            "k_tile": k_tile,
            "pipeline": mode_tag,
            "ms": round(ms, 2),
            "fps": round(fps, 1),
            "mega_gaussians_per_s": round((n_g * fps) / 1e6, 2),
            "compile_s": round(comp_s, 2),
            "mean_rgb": round(float(jnp.mean(img)), 4),
        }
        key = (rec["n_chips"], rec["n_gaussians"], rec["res"], rec["k_macro"], rec["k_tile"], rec["pipeline"])
        data["runs"] = [
            r for r in data.get("runs", [])
            if (r.get("n_chips"), r.get("n_gaussians"), r.get("res"), r.get("k_macro"), r.get("k_tile", 128), r.get("pipeline")) != key
        ] + [rec]
        with open(out_json, "w") as f:
          json.dump(data, f, indent=2)
        print(f"WAVE2 {n_chips}C:", rec, flush=True)

        if n_chips == 1 and res_name == "4K" and not use_bf16 and n_g == 10_000_000 and not export_hq_4k:
          png_path = os.path.join(export_dir, "v3_w3_gs3d_4k.png")
          _save_png(np.asarray(img), png_path)
          print(f"Saved Native 4K Sponza 10M render: {png_path}", flush=True)

    if export_hq_4k and n_g == 10_000_000:
      gs_hq = sample_anisotropic_3dgs_from_gltf(sc, n_g, seed=42, splat_scale=1.25, thick_ratio=0.22)
      if n_chips == 4:
        args_hq = tuple(
            jax.device_put(jnp.asarray(x), rep)
            for x in (gs_hq["mu"], gs_hq["cov6"], gs_hq["rgb"], gs_hq["opacity"], cam_pos, cam_mat, gs_hq["nrm"])
        )
        fn_hq = make_massive_scene_4c_fn(
            mesh, height=2160, width=3840, cam_fovy=fovy, znear=znear,
            k_macro=k_macro, use_bf16_mxu=False, has_nrm=True, k_tile=512, macro_rep=4,
        )
      else:
        args_hq = tuple(
            jnp.asarray(x)
            for x in (gs_hq["mu"], gs_hq["cov6"], gs_hq["rgb"], gs_hq["opacity"], cam_pos, cam_mat, gs_hq["nrm"])
        )
        fn_hq = jax.jit(
            lambda a, b, c, d, e, f, g: render_massive_scene_1c(
                a, b, c, d, e, f, height=2160, width=3840, nrm=g, cam_fovy=fovy, znear=znear,
                k_macro=k_macro, use_bf16_mxu=False, k_tile=512, macro_rep=4, interpret=False,
            )
        )
      ms_hq, comp_hq, img_hq = _bench_fn(fn_hq, args_hq, n_warmup=1, n_steps=3)
      fps_hq = 1000.0 / ms_hq
      rec_hq = {
          "n_chips": n_chips, "scene": "Sponza", "n_gaussians": n_g,
          "splat_radius_m": round(float(gs_hq["splat_radius"]), 5), "splat_scale": 1.25,
          "res_label": "4K", "res": "3840x2160", "k_macro": k_macro, "k_tile": 512,
          "pipeline": "v3_macro_fp32_k512_rep4", "ms": round(ms_hq, 2), "fps": round(fps_hq, 1),
          "mega_gaussians_per_s": round((n_g * fps_hq) / 1e6, 2),
          "compile_s": round(comp_hq, 2), "mean_rgb": round(float(jnp.mean(img_hq)), 4),
      }
      data["runs"].append(rec_hq)
      with open(out_json, "w") as f:
        json.dump(data, f, indent=2)
      png_path = os.path.join(export_dir, "v3_w3_gs3d_4k.png")
      _save_png(np.asarray(img_hq), png_path)
      print(f"Saved HQ Watertight Native 4K Sponza 10M render: {png_path} ({ms_hq:.2f} ms, {fps_hq:.1f} FPS)", flush=True)
      del args_hq

    del args


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--phase", required=True, choices=["1c", "4c"])
  ap.add_argument("--n_list", type=int, nargs="+", default=[100_000, 1_000_000, 10_000_000])
  ap.add_argument("--k_macro", type=int, default=4096)
  ap.add_argument("--k_tile", type=int, default=128)
  ap.add_argument("--macro_rep", type=int, default=3)
  ap.add_argument("--splat_scale", type=float, default=1.0)
  ap.add_argument("--thick_ratio", type=float, default=0.14)
  ap.add_argument("--export_hq_4k", action="store_true")
  ap.add_argument("--res_filter", type=str, nargs="*", default=None)
  ap.add_argument("--out_json", default="/dev/shm/zaychik-tpu-mjx-w3/out/v3_w3_wave2_10m_gs3d_4k.json")
  ap.add_argument("--export_dir", default="/dev/shm/zaychik-tpu-mjx-w3/out")
  args = ap.parse_args()
  run_phase(
      args.phase, args.n_list, args.out_json, args.export_dir, args.k_macro,
      k_tile=args.k_tile, macro_rep=args.macro_rep,
      splat_scale=args.splat_scale, thick_ratio=args.thick_ratio,
      export_hq_4k=args.export_hq_4k, res_filter=args.res_filter,
  )


if __name__ == "__main__":
  main()
