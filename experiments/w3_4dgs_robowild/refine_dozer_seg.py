"""Verify clean surgical segmentation (v3) of LEGO bulldozer using full-table plane."""

from __future__ import annotations

import json
import os

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
  with open(os.path.join(out_dir, "full_table_meta.json")) as f:
    meta = json.load(f)

  P_origin = np.asarray(meta["P_origin"], dtype=np.float32)
  R_table = np.asarray(meta["R_table"], dtype=np.float32)

  sp = load_ply(os.path.join(data_dir, "point_cloud/iteration_30000/point_cloud.ply"))
  cams = load_cameras_json(os.path.join(data_dir, "cameras.json"))
  c0 = cams[0]
  W, H = 1920, 1080
  K = scaled_K(c0, W, H)

  means = sp["means"]
  ops = sp["opacities"]
  scales = sp["scales"]
  max_scale = np.max(scales, axis=-1)
  rgb = np.clip(_SH_C0 * sp["sh_dc"][:, 0, :] + 0.5, 0.0, 1.0)

  local = (means - P_origin[None, :]) @ R_table
  lx, ly, lz = local[:, 0], local[:, 1], local[:, 2]

  p_cam = (means - c0["position"][None, :]) @ c0["rotation"]
  in_front = p_cam[:, 2] > 0.2
  cam_dist = np.linalg.norm(p_cam, axis=-1)
  px = K[0, 0] * (p_cam[:, 0] / np.maximum(p_cam[:, 2], 1e-4)) + K[0, 2]
  py = K[1, 1] * (p_cam[:, 1] / np.maximum(p_cam[:, 2], 1e-4)) + K[1, 2]

  seed_y = (
      in_front & (cam_dist < 4.5)
      & (px >= 720) & (px <= 1600)
      & (py >= 110) & (py <= 780)
      & (ops > 0.15) & (max_scale < 0.08)
      & (lz > -0.02) & (lz < 1.00)
      & (rgb[:, 0] > 0.48) & (rgb[:, 1] > 0.34) & (rgb[:, 2] < 0.22)
      & ((rgb[:, 0] - rgb[:, 2]) > 0.28)
  )
  y_pts = local[seed_y]
  med_xy = np.median(y_pts[:, :2], axis=0)
  y_pts = y_pts[np.linalg.norm(y_pts[:, :2] - med_xy[None, :], axis=-1) < 0.95]

  xy_ctr = np.mean(y_pts[:, :2], axis=0)
  _, eigvecs = np.linalg.eigh(np.cov((y_pts[:, :2] - xy_ctr).T))
  e_long = eigvecs[:, 1]
  if e_long[0] < 0:
    e_long = -e_long
  e_lat = np.array([-e_long[1], e_long[0]], dtype=np.float32)

  uv = (local[:, :2] - xy_ctr[None, :]) @ np.stack([e_long, e_lat], axis=1)
  u_d, v_d = uv[:, 0], uv[:, 1]
  y_uv = (y_pts[:, :2] - xy_ctr[None, :]) @ np.stack([e_long, e_lat], axis=1)

  u_p1, u_p99 = np.percentile(y_uv[:, 0], [0.5, 99.5])
  v_p1, v_p99 = np.percentile(y_uv[:, 1], [0.5, 99.5])
  z_p99 = float(np.percentile(y_pts[:, 2], 99.8))
  print(f"Yellow p0.5..99.5: u=[{u_p1:.4f}, {u_p99:.4f}], v=[{v_p1:.4f}, {v_p99:.4f}], z_hi={z_p99:.4f}")

  # Tight OBB around bulldozer (excluding pouch behind +v, including rear wheel at -u and blade at +u)
  in_uv_body = (
      (u_d >= u_p1 - 0.14) & (u_d <= u_p99 + 0.08)
      & (v_d >= v_p1 - 0.08) & (v_d <= v_p99 + 0.04)
  )
  # Separate upper body (lz >= 0.015) from floor-contact layer (lz in [-0.015, 0.015])
  is_yellow_col = (rgb[:, 0] > 0.44) & (rgb[:, 1] > 0.28) & (rgb[:, 2] < 0.25) & ((rgb[:, 0] - rgb[:, 2]) > 0.22)
  is_dark_col = (np.max(rgb, axis=-1) < 0.26)
  is_grey_col = (np.max(rgb, axis=-1) - np.min(rgb, axis=-1) < 0.12) & (np.mean(rgb, axis=-1) < 0.60)

  fg_mask = (
      in_uv_body
      & (max_scale < 0.08)
      & (lz <= z_p99 + 0.08)
      & (
          (lz >= 0.008)
          | ((lz >= -0.016) & (lz < 0.008) & (is_yellow_col | is_dark_col | is_grey_col))
      )
  )
  print(f"Surgical fg_mask count: {np.sum(fg_mask)} / {len(means)}")

  fg_local = local[fg_mask]
  fg_min = np.percentile(fg_local, 0.5, axis=0)
  fg_max = np.percentile(fg_local, 99.5, axis=0)
  fg_ctr = np.array([
      0.5 * (fg_min[0] + fg_max[0]),
      0.5 * (fg_min[1] + fg_max[1]),
      0.5 * (fg_min[2] + fg_max[2]),
  ], dtype=np.float32)
  print(f"Dozer local bbox min={fg_min.round(4)}, max={fg_max.round(4)}, ctr={fg_ctr.round(4)}")

  meta.update({
      "xy_ctr": xy_ctr.tolist(),
      "e_long": e_long.tolist(),
      "e_lat": e_lat.tolist(),
      "u_p1": float(u_p1), "u_p99": float(u_p99),
      "v_p1": float(v_p1), "v_p99": float(v_p99),
      "z_p99": float(z_p99),
      "fg_count": int(np.sum(fg_mask)),
      "fg_min_local": fg_min.tolist(),
      "fg_max_local": fg_max.tolist(),
      "fg_ctr_local": fg_ctr.tolist(),
  })
  with open(os.path.join(out_dir, "full_table_meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

  devs = jax.devices()[:4]
  mesh = jax.sharding.Mesh(np.array(devs), ("dp",))
  rep = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
  fn = make_sharded_rasterizer(
      mesh, W, H, sh_degree=3, macro_size=32, macro_rep=5,
      k_macro=10240, k_large_per_chip=1536, k_tile=512,
  )
  v_dev = jax.device_put(jnp.asarray(c0["viewmat"]), rep)
  k_dev = jax.device_put(jnp.asarray(K), rep)

  for tag, keep_mask in [("fg_v3", fg_mask), ("bg_v3", ~fg_mask)]:
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
  print("Saved debug_seg_fg_v3.png and debug_seg_bg_v3.png", flush=True)


if __name__ == "__main__":
  main()
