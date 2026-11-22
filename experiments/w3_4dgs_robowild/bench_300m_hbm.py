"""Task B: 200M and 250M-300M Single-Host 4-Chip 3DGS HBM Ceiling Benchmark."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import time
from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from tpu_mjx_render.adapters.tpu_gsplat import (
    _SH_C0,
    load_cameras_json,
    load_ply,
    make_sharded_rasterizer,
)
from tpu_mjx_render.roofline import inspect_hlo_ops


def _check_hlo(lowered, name: str) -> Dict[str, Any]:
  txt = lowered.as_text()
  info = inspect_hlo_ops(txt)
  info["send_recv_ops"] = len(re.findall(r"\bsend\(|\brecv\(", txt, re.I))
  assert info["host_callbacks"] == 0, f"{name} host_callbacks != 0: {info}"
  assert info["send_recv_ops"] == 0, f"{name} send_recv_ops != 0: {info}"
  assert info["gather_ops"] == 0, f"{name} gather_ops != 0: {info}"
  info["zero_copy_verified"] = True
  return info


def _scaled_K(cam: Dict[str, Any], width: int, height: int) -> np.ndarray:
  sf = float(height) / float(cam["height"])
  return np.array(
      [[float(cam["fx"]) * sf, 0.0, 0.5 * width],
       [0.0, float(cam["fy"]) * sf, 0.5 * height],
       [0.0, 0.0, 1.0]],
      dtype=np.float32,
  )


def _load_base_garden(data_root: str) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
  garden_dir = os.path.join(data_root, "garden")
  ply_p = os.path.join(garden_dir, "point_cloud/iteration_30000/point_cloud.ply")
  cam_p = os.path.join(garden_dir, "cameras.json")
  if not os.path.exists(ply_p):
    # Fallback to kitchen if garden wasn't fetched
    garden_dir = os.path.join(data_root, "kitchen")
    ply_p = os.path.join(garden_dir, "point_cloud/iteration_30000/point_cloud.ply")
    cam_p = os.path.join(garden_dir, "cameras.json")
  sp = load_ply(ply_p, max_sh_degree=0)
  c0 = load_cameras_json(cam_p)[0]
  rad = np.linalg.norm(sp["means"], axis=-1)
  keep = (rad < 12.0) & (sp["opacities"] > 0.03)
  base = {
      "means": np.ascontiguousarray(sp["means"][keep], dtype=np.float32),
      "quats": np.ascontiguousarray(sp["quats"][keep], dtype=np.float32),
      "scales": np.ascontiguousarray(sp["scales"][keep], dtype=np.float32),
      "opacities": np.ascontiguousarray(sp["opacities"][keep], dtype=np.float32),
      "colors": np.ascontiguousarray(
          np.clip(_SH_C0 * sp["sh_dc"][keep, 0, :] + 0.5, 0.0, 1.0), dtype=np.float32
      ),
  }
  return base, c0


def _build_sharded_megascale_on_devices(
    base: Dict[str, np.ndarray],
    target_n: int,
    devs: List[jax.Device],
    mesh: jax.sharding.Mesh,
    seed: int = 101,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
  """Build target_n Gaussians directly shard-by-shard onto 4 TPU chips to minimize host RAM."""
  n_chips = len(devs)
  assert target_n % n_chips == 0
  n_per_chip = target_n // n_chips
  n_base = len(base["means"])
  n_base_chip = n_base // n_chips
  n_reps = (n_per_chip + n_base_chip - 1) // n_base_chip

  offsets = [np.array([0.0, 0.0, 0.0], np.float32)]
  for ring_i in range(1, n_reps):
    ang = ring_i * 2.39996
    rad = 3.5 + 1.8 * math.sqrt(ring_i)
    offsets.append(np.array([rad * math.cos(ang), 0.0, rad * math.sin(ang)], np.float32))

  m_bufs, q_bufs, s_bufs, o_bufs, c_bufs = [], [], [], [], []
  for dev_i, dev in enumerate(devs):
    rng = np.random.default_rng(seed + dev_i * 17)
    sl_src = slice(dev_i, n_base_chip * n_chips, n_chips)
    bm = base["means"][sl_src]
    bq = base["quats"][sl_src]
    bs = base["scales"][sl_src]
    bo = base["opacities"][sl_src]
    bc = base["colors"][sl_src]

    m_loc = np.empty((n_per_chip, 3), dtype=np.float32)
    q_loc = np.empty((n_per_chip, 4), dtype=np.float32)
    s_loc = np.empty((n_per_chip, 3), dtype=np.float32)
    o_loc = np.empty((n_per_chip,), dtype=np.float32)
    c_loc = np.empty((n_per_chip, 3), dtype=np.float32)

    cursor = 0
    for r_idx, off in enumerate(offsets):
      take = min(n_base_chip, n_per_chip - cursor)
      if take <= 0:
        break
      sl = slice(cursor, cursor + take)
      if r_idx == 0:
        m_loc[sl] = bm[:take]
      else:
        jitter = rng.normal(0.0, 0.008, size=(take, 3)).astype(np.float32)
        m_loc[sl] = bm[:take] + off[None, :] + jitter
      q_loc[sl] = bq[:take]
      s_loc[sl] = bs[:take] * (1.0 if r_idx == 0 else 0.85)
      o_loc[sl] = bo[:take]
      c_loc[sl] = bc[:take]
      cursor += take

    perm = rng.permutation(n_per_chip)
    m_bufs.append(jax.device_put(np.ascontiguousarray(m_loc[perm]), dev))
    q_bufs.append(jax.device_put(np.ascontiguousarray(q_loc[perm]), dev))
    s_bufs.append(jax.device_put(np.ascontiguousarray(s_loc[perm]), dev))
    o_bufs.append(jax.device_put(np.ascontiguousarray(o_loc[perm]), dev))
    c_bufs.append(jax.device_put(np.ascontiguousarray(c_loc[perm]), dev))
    del m_loc, q_loc, s_loc, o_loc, c_loc, perm
    gc.collect()

  dp = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("dp"))
  m_dev = jax.make_array_from_single_device_arrays((target_n, 3), dp, m_bufs)
  q_dev = jax.make_array_from_single_device_arrays((target_n, 4), dp, q_bufs)
  s_dev = jax.make_array_from_single_device_arrays((target_n, 3), dp, s_bufs)
  o_dev = jax.make_array_from_single_device_arrays((target_n,), dp, o_bufs)
  c_dev = jax.make_array_from_single_device_arrays((target_n, 3), dp, c_bufs)
  return m_dev, q_dev, s_dev, o_dev, c_dev


def run_benchmark(
    data_root: str,
    counts: List[int],
    out_json: str,
    export_dir: str,
) -> Dict[str, Any]:
  os.makedirs(export_dir, exist_ok=True)
  os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)

  devs = jax.devices()[:4]
  mesh = jax.sharding.Mesh(np.array(devs), ("dp",))
  rep = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

  base, c0 = _load_base_garden(data_root)
  print(f"Loaded base scene: {len(base['means']):,} core Gaussians", flush=True)

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

  results: List[Dict[str, Any]] = []
  max_clean_n = 0

  for target_n in counts:
    t0 = time.perf_counter()
    print(f"\n--- Building & sharding {target_n:,} Gaussians across 4 chips ---", flush=True)
    m_dev, q_dev, s_dev, o_dev, c_dev = _build_sharded_megascale_on_devices(
        base, target_n, devs, mesh, seed=101 + target_n // 1_000_000
    )
    print(f"Sharded {target_n:,} Gaussians to 4 chips in {time.perf_counter() - t0:.1f}s", flush=True)

    n_ok_res = 0
    for res_label, W, H in (("1080p", 1920, 1080), ("4K", 3840, 2160)):
      K = _scaled_K(c0, W, H)
      v_dev = jax.device_put(jnp.asarray(c0["viewmat"]), rep)
      k_dev = jax.device_put(jnp.asarray(K), rep)

      fn = make_sharded_rasterizer(
          mesh, W, H, sh_degree=0, macro_size=64, macro_rep=2,
          k_macro=4096, k_large_per_chip=256, k_tile=256,
      )
      try:
        low = fn.lower(m_dev, q_dev, s_dev, o_dev, c_dev, v_dev, k_dev)
        hlo = _check_hlo(low, f"mega_{target_n}_{res_label}")
        report["hlo_checks"][f"mega_{target_n // 1_000_000}m_{res_label}"] = hlo

        t_c = time.perf_counter()
        comp = low.compile()
        mem = comp.memory_analysis()
        temp_gb = float(mem.temp_size_in_bytes) / 1e9
        arg_gb = float(mem.argument_size_in_bytes) / 1e9
        out_gb = float(mem.output_size_in_bytes) / 1e9
        alias_gb = float(getattr(mem, "alias_size_in_bytes", 0)) / 1e9
        chip_hbm_gb = arg_gb + temp_gb + out_gb - alias_gb
        total_4c_hbm_gb = 4.0 * chip_hbm_gb

        out = comp(m_dev, q_dev, s_dev, o_dev, c_dev, v_dev, k_dev)
        jax.block_until_ready(out)
        compile_s = time.perf_counter() - t_c

        # Warmup + timed loop
        out = comp(m_dev, q_dev, s_dev, o_dev, c_dev, v_dev, k_dev)
        jax.block_until_ready(out)

        t1 = time.perf_counter()
        for _ in range(3):
          out = comp(m_dev, q_dev, s_dev, o_dev, c_dev, v_dev, k_dev)
        jax.block_until_ready(out)
        ms = (time.perf_counter() - t1) * 1e3 / 3.0
        rgb, alp, _ = out

        n_micro_tiles = ((W + 63) // 64) * ((H + 63) // 64) * 64
        rec = {
            "status": "ok",
            "n_gaussians": int(target_n),
            "n_chips": 4,
            "res_label": res_label,
            "res": f"{W}x{H}",
            "ms": round(ms, 2),
            "fps": round(1000.0 / ms, 2),
            "compile_s": round(compile_s, 2),
            "hbm_per_chip_gb": round(chip_hbm_gb, 2),
            "hbm_4chip_total_gb": round(total_4c_hbm_gb, 2),
            "hbm_arg_per_chip_gb": round(arg_gb, 2),
            "hbm_temp_per_chip_gb": round(temp_gb, 2),
            "hbm_utilization_pct_of_380gb": round(100.0 * total_4c_hbm_gb / 380.0, 2),
            "macro_size": 64,
            "macro_rep": 2,
            "k_macro": 4096,
            "k_tile": 256,
            "micro_tiles": n_micro_tiles,
            "cull_ratio_per_tile": round(float(target_n) / 256.0, 1),
            "mean_alpha": round(float(jnp.mean(alp)), 4),
        }
        results.append(rec)
        n_ok_res += 1
        print(
            f"MEGA {target_n // 1_000_000}M {res_label}: {ms:.2f} ms ({1000/ms:.2f} FPS) | "
            f"HBM={total_4c_hbm_gb:.2f} GB / 380 GB ({rec['hbm_utilization_pct_of_380gb']:.1f}%) | "
            f"cull={rec['cull_ratio_per_tile']:,.1f}:1",
            flush=True,
        )
        if res_label == "4K":
          arr = np.clip(np.round(np.asarray(rgb) * 255.0), 0, 255).astype(np.uint8)
          Image.fromarray(arr).save(os.path.join(export_dir, f"v3_w3_{target_n // 1_000_000}m_4k.png"))
        del out, rgb, alp, comp, low
      except Exception as exc:
        err_msg = str(exc)[:240]
        print(f"MEGA {target_n // 1_000_000}M {res_label} FAILED: {err_msg}", flush=True)
        results.append({
            "status": "oom_or_error",
            "n_gaussians": int(target_n),
            "res_label": res_label,
            "res": f"{W}x{H}",
            "error": err_msg,
        })
      gc.collect()

    if n_ok_res == 2:
      max_clean_n = max(max_clean_n, target_n)

    report["megascale_hbm_ceiling"] = {
        "max_clean_gaussians": max_clean_n,
        "runs": results,
    }
    with open(out_json, "w") as f:
      json.dump(report, f, indent=2)

    del m_dev, q_dev, s_dev, o_dev, c_dev
    gc.collect()

  print(f"\nSaved Task B metrics -> {out_json}", flush=True)
  return report


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--data_root", default="/dev/shm/zaychik-w3-v6/data")
  ap.add_argument("--counts", nargs="+", type=int, default=[200_000_000, 250_000_000, 300_000_000])
  ap.add_argument("--out_json", default="/dev/shm/zaychik-w3-v6/out/v3_w3_wave6_4dgs_300m.json")
  ap.add_argument("--export_dir", default="/dev/shm/zaychik-w3-v6/out")
  args = ap.parse_args()
  run_benchmark(args.data_root, args.counts, args.out_json, args.export_dir)


if __name__ == "__main__":
  main()
