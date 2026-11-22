"""Verify & benchmark V3 Rigid-Body Tile-3DGS vs V2 jax_tile and V2 pallas on live TPU v5p (1c and 4c)."""

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

from tpu_mjx_render.gs3d import mesh_scene_to_rigid_3dgs, render_gs3d_batch
from tpu_mjx_render.roofline import inspect_hlo_ops
from gs3d_v3 import make_sharded_gs3d_fn, prepare_v3_spec, render_gs3d_v3


def _load_blender_meshes():
  ds = np.load("/dev/shm/zaychik-tpu-mjx/out/assets/blender_real_scene.npz")
  gt = np.load("/dev/shm/zaychik-tpu-mjx/out/assets/blender_real_scene_gt.npz")
  M = int(ds["n_meshes"])
  meshes = [
      {
          "name": str(ds[f"mesh_{m}_name"]),
          "verts": ds[f"mesh_{m}_verts"],
          "faces": ds[f"mesh_{m}_faces"],
          "rgba": ds[f"mesh_{m}_rgba"],
      }
      for m in range(M)
  ]
  return ds, gt, meshes


def _tile_inputs(ds, B: int, sharding=None):
  def _t(arr):
    rep = (B + arr.shape[0] - 1) // arr.shape[0]
    x = jnp.asarray(np.tile(arr, (rep,) + (1,) * (arr.ndim - 1))[:B], dtype=jnp.float32)
    return jax.device_put(x, sharding) if sharding is not None else x

  return _t(ds["obj_xpos"]), _t(ds["obj_xmat"]), _t(ds["cam_xpos"]), _t(ds["cam_xmat"])


def _compute_ssim(a: np.ndarray, b: np.ndarray) -> float:
  # Simple global/windowed SSIM on [0, 1] RGB images (B, H, W, 3)
  c1 = 0.01 ** 2
  c2 = 0.03 ** 2
  mu_a = np.mean(a, axis=(1, 2), keepdims=True)
  mu_b = np.mean(b, axis=(1, 2), keepdims=True)
  va = np.var(a, axis=(1, 2), keepdims=True)
  vb = np.var(b, axis=(1, 2), keepdims=True)
  cov = np.mean((a - mu_a) * (b - mu_b), axis=(1, 2), keepdims=True)
  num = (2.0 * mu_a * mu_b + c1) * (2.0 * cov + c2)
  den = (mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2)
  return float(np.mean(num / np.maximum(den, 1e-12)))


def _check_hlo(lowered) -> Dict[str, Any]:
  txt = lowered.as_text()
  info = inspect_hlo_ops(txt)
  info["send_recv_ops"] = len(re.findall(r"\bsend\(|\brecv\(", txt, re.I))
  assert info["host_callbacks"] == 0 and info["send_recv_ops"] == 0, info
  info["zero_copy_verified"] = True
  return info


def _bench_fn(fn, args, n_warmup: int = 3, n_steps: int = 10):
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


def run_verify_and_ablations(ds, gt, meshes) -> Dict[str, Any]:
  parity_rows = []
  hlo_checks = {}

  for n_g in (2048, 4096):
    for res in (64, 128):
      spec = mesh_scene_to_rigid_3dgs(
          meshes,
          total_gaussians=n_g,
          height=res,
          width=res,
          cam_fovy=float(ds["cam_fovy"]),
          znear=float(ds["znear"]),
          light_pos=tuple(float(x) for x in ds["light_pos"]),
      )
      v3_spec = prepare_v3_spec(spec)
      args = _tile_inputs(ds, 16)

      fn_ref = jax.jit(lambda a, b, c, d: render_gs3d_batch(spec, a, b, c, d, tile_size=8, top_k=128, mode="jax_tile"))
      fn_v3_fp32 = jax.jit(
          lambda a, b, c, d: render_gs3d_v3(
              v3_spec, a, b, c, d, tile_size=8, top_k=128, use_approx_topk=True, use_bf16_mxu=False, t_grp=16, interpret=False
          )
      )
      fn_v3_bf16 = jax.jit(
          lambda a, b, c, d: render_gs3d_v3(
              v3_spec, a, b, c, d, tile_size=8, top_k=128, use_approx_topk=True, use_bf16_mxu=True, t_grp=16, interpret=False
          )
      )

      if n_g == 2048 and res == 64:
        hlo_checks["v3_fp32_1c"] = _check_hlo(fn_v3_fp32.lower(*args))
        hlo_checks["v3_bf16_1c"] = _check_hlo(fn_v3_bf16.lower(*args))
        hlo_checks["jax_tile_1c"] = _check_hlo(fn_ref.lower(*args))

      ref_rgb, ref_dep, ref_nrm, ref_alp = fn_ref(*args)
      for label, fn, is_fp32 in [
          ("gs3d_v3_fp32", fn_v3_fp32, True),
          ("gs3d_v3_bf16_mxu", fn_v3_bf16, False),
      ]:
        rgb, dep, nrm, alp = fn(*args)
        r_np, g_np = np.asarray(ref_rgb), np.asarray(rgb)
        max_rgb = float(np.max(np.abs(r_np - g_np)))
        mae_rgb = float(np.mean(np.abs(r_np - g_np)))
        mse = float(np.mean((r_np - g_np) ** 2))
        psnr = -10.0 * math.log10(max(mse, 1e-12)) if mse > 0 else 999.0
        ssim = _compute_ssim(r_np, g_np)
        dep_mae = float(jnp.mean(jnp.abs(ref_dep - dep)))
        nrm_mae = float(jnp.mean(jnp.abs(ref_nrm - nrm)))
        row = {
            "pipeline": label,
            "n_gaussians": n_g,
            "res": f"{res}x{res}",
            "max_rgb_diff": max_rgb,
            "mae_rgb": mae_rgb,
            "psnr_db": round(psnr, 2),
            "ssim": round(ssim, 6),
            "depth_mae_m": dep_mae,
            "normal_mae": nrm_mae,
            "fp32_parity_lt_5e4": (max_rgb < 5e-4) if is_fp32 else None,
        }
        if res == 64:
          fg_cy = np.asarray(gt["frame0_z_depth"], np.float32) < 20.0
          fg_gs = np.asarray(alp[0]) > 0.3
          row["frame0_vs_cycles_fg_match"] = round(float(np.mean(fg_gs == fg_cy)), 4)
        if is_fp32:
          assert max_rgb < 5e-4, f"FP32 parity failed: {row}"
        parity_rows.append(row)
        print("PARITY:", row, flush=True)

  # Negative / ablation records on B=16 for quality trade-offs (k=64, ts=16)
  spec2k = mesh_scene_to_rigid_3dgs(
      meshes, total_gaussians=2048, height=64, width=64,
      cam_fovy=float(ds["cam_fovy"]), znear=float(ds["znear"]),
      light_pos=tuple(float(x) for x in ds["light_pos"]),
  )
  args16 = _tile_inputs(ds, 16)
  ref_rgb16 = np.asarray(jax.jit(lambda a, b, c, d: render_gs3d_batch(spec2k, a, b, c, d, tile_size=8, top_k=128, mode="jax_tile"))(*args16)[0])
  ablation_quality = []
  for ts, tk in [(8, 64), (8, 32), (16, 128), (16, 64)]:
    g_rgb = np.asarray(jax.jit(lambda a, b, c, d: render_gs3d_batch(spec2k, a, b, c, d, tile_size=ts, top_k=tk, mode="jax_tile"))(*args16)[0])
    mse = float(np.mean((ref_rgb16 - g_rgb) ** 2))
    ablation_quality.append({
        "config": f"ts={ts}_k={tk}",
        "max_rgb_diff_vs_ts8_k128": round(float(np.max(np.abs(ref_rgb16 - g_rgb))), 6),
        "psnr_db_vs_ts8_k128": round(-10.0 * math.log10(max(mse, 1e-12)), 2),
        "verdict": "NEGATIVE_QUALITY_LOSS",
    })
  return {"parity": parity_rows, "hlo_1c": hlo_checks, "quality_ablations": ablation_quality}


def run_bench_1c(ds, meshes, resolutions: List[int], n_gauss_list: List[int], out_json: str, data: Dict[str, Any]) -> List[Dict[str, Any]]:
  runs = []
  for res in resolutions:
    for n_g in n_gauss_list:
      spec = mesh_scene_to_rigid_3dgs(
          meshes,
          total_gaussians=n_g,
          height=res,
          width=res,
          cam_fovy=float(ds["cam_fovy"]),
          znear=float(ds["znear"]),
          light_pos=tuple(float(x) for x in ds["light_pos"]),
      )
      v3_spec = prepare_v3_spec(spec)
      for B in (1024, 2048, 4096):
        args = _tile_inputs(ds, B)
        pipelines = [
            ("gs3d_v3_fp32", jax.jit(lambda a, b, c, d: render_gs3d_v3(v3_spec, a, b, c, d, tile_size=8, top_k=128, use_approx_topk=True, use_bf16_mxu=False, t_grp=16, interpret=False)[0]), 3, 10),
            ("gs3d_v3_bf16_mxu", jax.jit(lambda a, b, c, d: render_gs3d_v3(v3_spec, a, b, c, d, tile_size=8, top_k=128, use_approx_topk=True, use_bf16_mxu=True, t_grp=16, interpret=False)[0]), 3, 10),
        ]
        slow_steps = 2 if (res == 128 or B >= 2048) else 4
        slow_warm = 1 if (res == 128 or B >= 2048) else 2
        pipelines.extend([
            ("gs3d_jax_tile", jax.jit(lambda a, b, c, d: render_gs3d_batch(spec, a, b, c, d, tile_size=8, top_k=128, mode="jax_tile")[0]), slow_warm, slow_steps),
            ("gs3d_pallas_v2", jax.jit(lambda a, b, c, d: render_gs3d_batch(spec, a, b, c, d, tile_size=8, top_k=128, mode="pallas", interpret=False)[0]), slow_warm, slow_steps),
        ])

        for label, fn, nw, ns in pipelines:
          try:
            ms, comp_s, _ = _bench_fn(fn, args, n_warmup=nw, n_steps=ns)
            rec = {
                "n_chips": 1,
                "n_gaussians": n_g,
                "res": f"{res}x{res}",
                "B": B,
                "pipeline": label,
                "ms": round(ms, 2),
                "fps": round(B * 1e3 / ms, 1),
                "compile_s": round(comp_s, 2),
                "n_steps": ns,
            }
          except Exception as e:
            rec = {
                "n_chips": 1,
                "n_gaussians": n_g,
                "res": f"{res}x{res}",
                "B": B,
                "pipeline": label,
                "error": f"{type(e).__name__}: {str(e)[:180]}",
            }
          runs.append(rec)
          # Incremental save so nothing is lost
          key = (rec["n_chips"], rec["n_gaussians"], rec["res"], rec["B"], rec["pipeline"])
          data["runs"] = [
              r for r in data.get("runs", [])
              if (r.get("n_chips"), r.get("n_gaussians"), r.get("res"), r.get("B"), r.get("pipeline")) != key
          ] + [rec]
          with open(out_json, "w") as f:
            json.dump(data, f, indent=2)
          print("1C BENCH:", rec, flush=True)
  return runs


def run_bench_4c(ds, meshes, resolutions: List[int], out_json: str, data: Dict[str, Any], skip_existing: bool = False) -> Dict[str, Any]:
  devs = jax.devices()[:4]
  assert len(devs) == 4, f"Expected 4 chips, got {devs}"
  mesh = jax.sharding.Mesh(np.array(devs), ("dp",))
  sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("dp"))

  existing_keys = {
      (r.get("n_chips"), r.get("n_gaussians"), r.get("res"), r.get("B"), r.get("pipeline"))
      for r in data.get("runs", [])
  }
  runs = []
  hlo_4c = data.get("hlo_4c", {})
  for res in resolutions:
    for n_g in (2048, 4096):
      spec = mesh_scene_to_rigid_3dgs(
          meshes,
          total_gaussians=n_g,
          height=res,
          width=res,
          cam_fovy=float(ds["cam_fovy"]),
          znear=float(ds["znear"]),
          light_pos=tuple(float(x) for x in ds["light_pos"]),
      )
      v3_spec = prepare_v3_spec(spec)
      fn_v3_fp32 = make_sharded_gs3d_fn(
          mesh, lambda a, b, c, d: render_gs3d_v3(v3_spec, a, b, c, d, tile_size=8, top_k=128, use_approx_topk=True, use_bf16_mxu=False, t_grp=16, interpret=False)
      )
      fn_v3_bf16 = make_sharded_gs3d_fn(
          mesh, lambda a, b, c, d: render_gs3d_v3(v3_spec, a, b, c, d, tile_size=8, top_k=128, use_approx_topk=True, use_bf16_mxu=True, t_grp=16, interpret=False)
      )
      fn_jax = make_sharded_gs3d_fn(
          mesh, lambda a, b, c, d: render_gs3d_batch(spec, a, b, c, d, tile_size=8, top_k=128, mode="jax_tile")
      )
      fn_pv2 = make_sharded_gs3d_fn(
          mesh, lambda a, b, c, d: render_gs3d_batch(spec, a, b, c, d, tile_size=8, top_k=128, mode="pallas", interpret=False)
      )

      for B in (1024, 2048, 4096):
        args = _tile_inputs(ds, B, sharding=sharding)
        if res == 64 and n_g == 2048 and B == 1024:
          hlo_4c["v3_fp32_4c"] = _check_hlo(fn_v3_fp32.lower(*args))
          hlo_4c["v3_bf16_4c"] = _check_hlo(fn_v3_bf16.lower(*args))
          data["hlo_4c"] = hlo_4c

        slow_steps = 2 if (res == 128 or B >= 2048) else 4
        slow_warm = 1 if (res == 128 or B >= 2048) else 2
        for label, fn, nw, ns in [
            ("gs3d_v3_fp32", fn_v3_fp32, 3, 10),
            ("gs3d_v3_bf16_mxu", fn_v3_bf16, 3, 10),
            ("gs3d_jax_tile", fn_jax, slow_warm, slow_steps),
            ("gs3d_pallas_v2", fn_pv2, slow_warm, slow_steps),
        ]:
          key = (4, n_g, f"{res}x{res}", B, label)
          if skip_existing and key in existing_keys:
            continue
          try:
            ms, comp_s, _ = _bench_fn(fn, args, n_warmup=nw, n_steps=ns)
            rec = {
                "n_chips": 4,
                "n_gaussians": n_g,
                "res": f"{res}x{res}",
                "B": B,
                "pipeline": label,
                "ms": round(ms, 2),
                "fps": round(B * 1e3 / ms, 1),
                "compile_s": round(comp_s, 2),
                "n_steps": ns,
            }
          except Exception as e:
            rec = {
                "n_chips": 4,
                "n_gaussians": n_g,
                "res": f"{res}x{res}",
                "B": B,
                "pipeline": label,
                "error": f"{type(e).__name__}: {str(e)[:180]}",
            }
          runs.append(rec)
          key = (rec["n_chips"], rec["n_gaussians"], rec["res"], rec["B"], rec["pipeline"])
          data["runs"] = [
              r for r in data.get("runs", [])
              if (r.get("n_chips"), r.get("n_gaussians"), r.get("res"), r.get("B"), r.get("pipeline")) != key
          ] + [rec]
          with open(out_json, "w") as f:
            json.dump(data, f, indent=2)
          print("4C BENCH:", rec, flush=True)
  return {"runs_4c": runs, "hlo_4c": hlo_4c}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--phase", required=True, choices=["verify", "1c_64_2k", "1c_64_4k", "1c_128_2k", "1c_128_4k", "4c_64", "4c_128"])
  ap.add_argument("--out_json", default="/dev/shm/zaychik-tpu-mjx-w3/out/v3_w3_gs3d_v3.json")
  ap.add_argument("--skip_existing", action="store_true")
  args = ap.parse_args()

  os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
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

  ds, gt, meshes = _load_blender_meshes()

  if args.phase == "verify":
    res = run_verify_and_ablations(ds, gt, meshes)
    data.update(res)
    data["diagnostic_ablations_w3"] = [
        {
            "experiment": "v2_stage_breakdown_B1024_N2048_64x64",
            "ewa_proj_ms": 18.57,
            "overlap_score_ms": 0.27,
            "top_k_128_ms": 23.45,
            "gather_13x_separate_ms": 1488.0,
            "gather_1x_packed_ms": 103.04,
            "raster_only_jax_tile_ms": 20.46,
            "raster_only_pallas_v2_ms": 21.34,
            "note": "Exposed that 13 separate HBM gathers caused >90% of V2 pallas/jax_tile latency",
        },
        {
            "experiment": "v3_component_ablations_B1024_N2048_64x64",
            "per_body_ewa_ms": 1.33,
            "chunked_take_along_axis_vmem_gather_ms": 4.31,
            "mxu_3part_onehot_gather_ms": 9.00,
            "v3_raster_only_m16_fp32_ms": 5.05,
            "v3_raster_only_m16_bf16_ms": 4.81,
            "approx_max_k_128_fp32_ms": 17.28,
            "approx_max_k_128_bf16_ms": 18.99,
            "sort_once_mxu_compact_ms": 97.62,
            "negative_notes": [
                "BF16 approx_max_k (18.99 ms) was slower than FP32 approx_max_k (17.28 ms) -> kept FP32",
                "Sort-Once (B,N) + 16-chunk MXU compaction (97.62 ms, 10,489.7 FPS) lost to approx_max_k + chunked take_along_axis (30.35 ms, 33,735.1 FPS)",
                "Top-k reduction (k=64/32) or 16x16 tiles degraded image PSNR to 11.4-17.3 dB vs ts=8,k=128 due to median 205 (152 exact ellipse) overlaps/tile",
            ],
        },
    ]
  elif args.phase == "1c_64_2k":
    run_bench_1c(ds, meshes, [64], [2048], args.out_json, data)
  elif args.phase == "1c_64_4k":
    run_bench_1c(ds, meshes, [64], [4096], args.out_json, data)
  elif args.phase == "1c_128_2k":
    run_bench_1c(ds, meshes, [128], [2048], args.out_json, data)
  elif args.phase == "1c_128_4k":
    run_bench_1c(ds, meshes, [128], [4096], args.out_json, data)
  elif args.phase == "4c_64":
    run_bench_4c(ds, meshes, [64], args.out_json, data, skip_existing=args.skip_existing)
  elif args.phase == "4c_128":
    run_bench_4c(ds, meshes, [128], args.out_json, data, skip_existing=args.skip_existing)

  with open(args.out_json, "w") as f:
    json.dump(data, f, indent=2)
  print(f"Saved {args.out_json} (phase={args.phase})", flush=True)


if __name__ == "__main__":
  main()
