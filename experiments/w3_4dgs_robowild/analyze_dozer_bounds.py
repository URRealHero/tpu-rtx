"""Fit exact full-table plane (zero tilt) and measure LEGO bulldozer OBB."""

from __future__ import annotations

import json
import os

import jax
import jax.numpy as jnp
import numpy as np

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


def fit_full_table_plane(dep_hw, R_c2w, cam_pos, fx, fy, cx0, cy0, place_xy, target_xy):
  H, W = dep_hw.shape
  # Sample full lower half of table (y in [450, 1060], x in [80, 1840]), excluding bulldozer bbox
  ys, xs = np.mgrid[450:1060:6, 80:1840:6]
  # Exclude bulldozer pixels u in [680, 1620], v in [100, 780]
  not_dozer = ~((xs >= 680) & (xs <= 1620) & (ys >= 100) & (ys <= 780))
  zs = dep_hw[ys, xs]
  ok = not_dozer & (zs > 0.2) & (zs < 6.0)
  xs_f, ys_f, zs_f = xs[ok] + 0.5, ys[ok] + 0.5, zs[ok]
  pts_cam = np.stack([(xs_f - cx0) / fx * zs_f, (ys_f - cy0) / fy * zs_f, zs_f], axis=-1)
  pts_w = pts_cam @ R_c2w.T + cam_pos[None, :]

  mask = np.ones(len(pts_w), dtype=bool)
  for _ in range(5):
    sub = pts_w[mask]
    ctr = sub.mean(axis=0)
    _, _, vt = np.linalg.svd(sub - ctr, full_matrices=False)
    nrm = vt[-1]
    res = np.abs((pts_w - ctr) @ nrm)
    mask = res < np.percentile(res, 65)

  sub = pts_w[mask]
  ctr = sub.mean(axis=0)
  _, _, vt = np.linalg.svd(sub - ctr, full_matrices=False)
  nrm = vt[-1]
  if np.dot(nrm, cam_pos - ctr) < 0.0:
    nrm = -nrm
  nrm = nrm / np.linalg.norm(nrm)
  med_res = float(np.median(np.abs((sub - ctr) @ nrm)))

  def _ray(px, py):
    rc = np.array([(px - cx0) / fx, (py - cy0) / fy, 1.0], dtype=np.float32)
    rw = R_c2w @ rc
    t = float(np.dot(ctr - cam_pos, nrm) / np.dot(rw, nrm))
    return (cam_pos + t * rw).astype(np.float32)

  p_origin = _ray(*place_xy)
  p_target = _ray(*target_xy)
  e_x = p_target - p_origin
  e_x = e_x - np.dot(e_x, nrm) * nrm
  e_x = e_x / np.linalg.norm(e_x)
  e_z = nrm.astype(np.float32)
  e_y = np.cross(e_z, e_x)
  e_y = e_y / np.linalg.norm(e_y)
  R_table = np.stack([e_x, e_y, e_z], axis=1).astype(np.float32)
  return {"P_origin": p_origin, "R_table": R_table, "table_normal": e_z, "median_res": med_res}


def main():
  data_dir = "/dev/shm/zaychik-w3-v6/data/kitchen"
  out_dir = "/dev/shm/zaychik-w3-v6/out"
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
  _, _, dep = fn(m, q, s, o, sh, v_dev, k_dev)
  jax.block_until_ready(dep)
  dep_np = np.asarray(dep[..., 0])

  # Place Panda near (470, 630) reaching toward (920, 620) (right beside the bulldozer!)
  tbl = fit_full_table_plane(
      dep_np, c0["rotation"], c0["position"],
      float(K[0, 0]), float(K[1, 1]), 0.5 * W, 0.5 * H,
      place_xy=(470.0, 630.0), target_xy=(920.0, 620.0),
  )
  print("Full-table fit median_res:", tbl["median_res"])

  means = sp["means"]
  ops = sp["opacities"]
  scales = sp["scales"]
  max_scale = np.max(scales, axis=-1)
  rgb = np.clip(_SH_C0 * sp["sh_dc"][:, 0, :] + 0.5, 0.0, 1.0)

  local = (means - tbl["P_origin"][None, :]) @ tbl["R_table"]
  lx, ly, lz = local[:, 0], local[:, 1], local[:, 2]

  # Project to c0 1080p
  p_cam = (means - c0["position"][None, :]) @ c0["rotation"]
  in_front = p_cam[:, 2] > 0.2
  px = K[0, 0] * (p_cam[:, 0] / np.maximum(p_cam[:, 2], 1e-4)) + K[0, 2]
  py = K[1, 1] * (p_cam[:, 1] / np.maximum(p_cam[:, 2], 1e-4)) + K[1, 2]

  # Coarse cluster in 2D + distance < 5.0 from camera
  cam_dist = np.linalg.norm(p_cam, axis=-1)
  seed_y = (
      in_front & (cam_dist < 4.5)
      & (px >= 720) & (px <= 1600)
      & (py >= 110) & (py <= 780)
      & (ops > 0.15) & (max_scale < 0.08)
      & (lz > -0.10) & (lz < 1.00)
      & (rgb[:, 0] > 0.48) & (rgb[:, 1] > 0.34) & (rgb[:, 2] < 0.22)
      & ((rgb[:, 0] - rgb[:, 2]) > 0.28)
  )
  y_pts = local[seed_y]
  med_xy = np.median(y_pts[:, :2], axis=0)
  d_med = np.linalg.norm(y_pts[:, :2] - med_xy[None, :], axis=-1)
  y_pts = y_pts[d_med < 1.0]
  print(f"Tight yellow core count: {len(y_pts)}")
  print(f"  lx=[{y_pts[:,0].min():.4f}, {y_pts[:,0].max():.4f}]")
  print(f"  ly=[{y_pts[:,1].min():.4f}, {y_pts[:,1].max():.4f}]")
  print(f"  lz=[{y_pts[:,2].min():.4f}, {y_pts[:,2].max():.4f}] (p1={np.percentile(y_pts[:,2],1):.4f})")

  # PCA of bulldozer XY
  xy_ctr = np.mean(y_pts[:, :2], axis=0)
  _, eigvecs = np.linalg.eigh(np.cov((y_pts[:, :2] - xy_ctr).T))
  e_long = eigvecs[:, 1]
  if e_long[0] < 0:
    e_long = -e_long
  e_lat = np.array([-e_long[1], e_long[0]], dtype=np.float32)

  uv = (local[:, :2] - xy_ctr[None, :]) @ np.stack([e_long, e_lat], axis=1)
  u_d, v_d = uv[:, 0], uv[:, 1]
  y_uv = (y_pts[:, :2] - xy_ctr[None, :]) @ np.stack([e_long, e_lat], axis=1)
  u_lo, u_hi = y_uv[:, 0].min(), y_uv[:, 0].max()
  v_lo, v_hi = y_uv[:, 1].min(), y_uv[:, 1].max()
  print(f"  xy_ctr={xy_ctr}, e_long={e_long}")
  print(f"  u=[{u_lo:.4f}, {u_hi:.4f}], v=[{v_lo:.4f}, {v_hi:.4f}]")

  # Check z histogram inside tight OBB
  in_obb = (u_d >= u_lo - 0.12) & (u_d <= u_hi + 0.08) & (v_d >= v_lo - 0.08) & (v_d <= v_hi + 0.08)
  for z0 in np.arange(-0.04, 0.06, 0.01):
    m_z = in_obb & (lz >= z0) & (lz < z0 + 0.01) & (max_scale < 0.08)
    is_yel = (rgb[:, 0] > 0.45) & (rgb[:, 1] > 0.30) & (rgb[:, 2] < 0.22)
    is_drk = (np.max(rgb, axis=-1) < 0.25)
    print(f"  lz [{z0:+.2f}, {z0+0.01:+.2f}): n={np.sum(m_z):5d}, yel={np.sum(m_z & is_yel):4d}, drk={np.sum(m_z & is_drk):4d}")

  with open(os.path.join(out_dir, "full_table_meta.json"), "w") as f:
    json.dump({
        "P_origin": tbl["P_origin"].tolist(),
        "R_table": tbl["R_table"].tolist(),
        "table_normal": tbl["table_normal"].tolist(),
        "median_res": tbl["median_res"],
        "xy_ctr": xy_ctr.tolist(),
        "e_long": e_long.tolist(),
        "e_lat": e_lat.tolist(),
        "u_lo": float(u_lo), "u_hi": float(u_hi),
        "v_lo": float(v_lo), "v_hi": float(v_hi),
        "z_lo": float(y_pts[:, 2].min()), "z_hi": float(y_pts[:, 2].max()),
    }, f, indent=2)


if __name__ == "__main__":
  main()
