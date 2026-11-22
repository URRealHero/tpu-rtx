"""Inspect kitchen 3DGS table plane and LEGO bulldozer segmentation bounds."""

from __future__ import annotations

import json
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from tpu_mjx_render.adapters.tpu_gsplat import (
    _SH_C0,
    load_cameras_json,
    load_ply,
    make_sharded_rasterizer,
    shard_splats_for_mesh,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "w3_realworld_3dgs"))
from robowild_kitchen import fit_table_plane_from_depth


def scaled_K(cam, width, height):
  sf = float(height) / float(cam["height"])
  return np.array(
      [[float(cam["fx"]) * sf, 0.0, 0.5 * width],
       [0.0, float(cam["fy"]) * sf, 0.5 * height],
       [0.0, 0.0, 1.0]],
      dtype=np.float32,
  )


def main():
  data_dir = "/dev/shm/zaychik-w3-v6/data/kitchen"
  out_dir = "/dev/shm/zaychik-w3-v6/out"
  os.makedirs(out_dir, exist_ok=True)

  sp = load_ply(os.path.join(data_dir, "point_cloud/iteration_30000/point_cloud.ply"))
  cams = load_cameras_json(os.path.join(data_dir, "cameras.json"))
  c0 = cams[0]
  W, H = 1920, 1080
  K = scaled_K(c0, W, H)

  devs = jax.devices()[:4]
  mesh = jax.sharding.Mesh(np.array(devs), ("dp",))
  rep = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

  fn = make_sharded_rasterizer(
      mesh, W, H, sh_degree=3, macro_size=32, macro_rep=5,
      k_macro=10240, k_large_per_chip=1536, k_tile=512,
  )
  m, q, s, o, sh, _ = shard_splats_for_mesh(sp, mesh, seed=42, use_sh=True)
  v_dev = jax.device_put(jnp.asarray(c0["viewmat"]), rep)
  k_dev = jax.device_put(jnp.asarray(K), rep)

  rgb, alp, dep = fn(m, q, s, o, sh, v_dev, k_dev)
  jax.block_until_ready((rgb, dep))
  dep_np = np.asarray(dep[..., 0])

  # Note: scale Wave 5 4K ROI (1040,1840,240,1400) -> 1080p (half res)
  tbl = fit_table_plane_from_depth(
      dep_np, c0["rotation"], c0["position"],
      float(K[0, 0]), float(K[1, 1]), 0.5 * W, 0.5 * H,
      roi=(520, 920, 120, 700),
      place_xy=(440.0, 670.0),
      target_xy=(780.0, 740.0),
  )
  print("Table fit median_res:", tbl["median_res"])
  print("P_origin:", tbl["P_origin"], "table_normal:", tbl["table_normal"])
  print("R_table:\n", tbl["R_table"])

  means = sp["means"]
  ops = sp["opacities"]
  scales = sp["scales"]
  rgb_dc = np.clip(_SH_C0 * sp["sh_dc"][:, 0, :] + 0.5, 0.0, 1.0)

  # Transform to table-local coordinates
  local = (means - tbl["P_origin"][None, :]) @ tbl["R_table"]
  lx, ly, lz = local[:, 0], local[:, 1], local[:, 2]

  # Find yellow LEGO seed points above the table
  is_yellow = (
      (ops > 0.15)
      & (lz > 0.015) & (lz < 0.85)
      & (rgb_dc[:, 0] > 0.45)
      & (rgb_dc[:, 1] > 0.32)
      & (rgb_dc[:, 2] < 0.25)
      & ((rgb_dc[:, 0] - rgb_dc[:, 2]) > 0.25)
  )
  y_pts = local[is_yellow]
  print(f"Yellow seed count: {len(y_pts)}")
  for name, p_lo, p_hi in [("1..99%", 1, 99), ("0.5..99.5%", 0.5, 99.5), ("min..max", 0, 100)]:
    lo = np.percentile(y_pts, p_lo, axis=0)
    hi = np.percentile(y_pts, p_hi, axis=0)
    print(f"Yellow {name}: x=[{lo[0]:.4f}, {hi[0]:.4f}], y=[{lo[1]:.4f}, {hi[1]:.4f}], z=[{lo[2]:.4f}, {hi[2]:.4f}]")

  # Also check 2D projection of yellow points in 1080p image
  p_cam = (means - c0["position"][None, :]) @ c0["rotation"]
  px = K[0, 0] * (p_cam[:, 0] / np.maximum(p_cam[:, 2], 1e-4)) + K[0, 2]
  py = K[1, 1] * (p_cam[:, 1] / np.maximum(p_cam[:, 2], 1e-4)) + K[1, 2]
  yp_px = px[is_yellow]
  yp_py = py[is_yellow]
  print(f"Yellow 2D pixel box (1080p): u=[{np.percentile(yp_px,1):.1f}, {np.percentile(yp_px,99):.1f}], v=[{np.percentile(yp_py,1):.1f}, {np.percentile(yp_py,99):.1f}]")

  # Check core percentiles of yellow points
  lo = np.percentile(y_pts, 1.0, axis=0)
  hi = np.percentile(y_pts, 99.0, axis=0)
  # Expand slightly for tracks/wheels/red beacon, keeping strict z floor > 0.012
  fg_mask = (
      (lx >= lo[0] - 0.14) & (lx <= hi[0] + 0.12)
      & (ly >= lo[1] - 0.12) & (ly <= hi[1] + 0.12)
      & (lz >= 0.012) & (lz <= hi[2] + 0.08)
  )
  print(f"Candidate fg_mask count: {np.sum(fg_mask)} / {len(means)}")

  # Save two quick renders: foreground only, and background only
  for tag, keep_mask in [("fg", fg_mask), ("bg", ~fg_mask)]:
    sub = {
        "means": sp["means"][keep_mask],
        "quats": sp["quats"][keep_mask],
        "scales": sp["scales"][keep_mask],
        "opacities": sp["opacities"][keep_mask],
        "sh_dc": sp["sh_dc"][keep_mask],
        "sh": sp["sh"][keep_mask],
        "sh_degree": 3,
        "n_gaussians": int(np.sum(keep_mask)),
    }
    ms, qs, ss, os_, shs, _ = shard_splats_for_mesh(sub, mesh, seed=42, use_sh=True)
    r_sub, _, _ = fn(ms, qs, ss, os_, shs, v_dev, k_dev)
    jax.block_until_ready(r_sub)
    arr = np.clip(np.round(np.asarray(r_sub) * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(os.path.join(out_dir, f"debug_seg_{tag}.png"))

  with open(os.path.join(out_dir, "dozer_stats.json"), "w") as f:
    json.dump({
        "median_res": float(tbl["median_res"]),
        "P_origin": tbl["P_origin"].tolist(),
        "R_table": tbl["R_table"].tolist(),
        "yellow_count": int(len(y_pts)),
        "yellow_p1": lo.tolist(),
        "yellow_p99": hi.tolist(),
        "fg_count": int(np.sum(fg_mask)),
    }, f, indent=2)
  print("Inspection complete!", flush=True)


if __name__ == "__main__":
  main()
