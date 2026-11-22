"""Camera ray generation aligned 1-to-1 with mujoco_warp/_src/render_util.py."""

from __future__ import annotations

import math
from typing import Tuple

import jax
import jax.numpy as jnp

PROJ_PERSPECTIVE = 0
PROJ_ORTHOGRAPHIC = 1


def compute_camera_rays(
    cam_pos: jax.Array,
    cam_mat: jax.Array,
    fovy_deg: float | jax.Array,
    height: int,
    width: int,
    projection: int = PROJ_PERSPECTIVE,
    sensorsize: Tuple[float, float] = (0.0, 0.0),
    intrinsic: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    znear: float = 0.01,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
  """Generate world-space ray origins, unit directions, and local -z cosines.

  Matches mujoco_warp/_src/render_util.py:compute_ray and render.py ray transform.

  Args:
    cam_pos: (3,) camera world position from mjx.Data.cam_xpos[cam_id].
    cam_mat: (3, 3) camera rotation matrix from mjx.Data.cam_xmat[cam_id].
    fovy_deg: vertical FOV in degrees (perspective) or vertical extent (ortho).
    height: image height H.
    width: image width W.
    projection: 0=perspective, 1=orthographic.
    sensorsize: MuJoCo cam_sensorsize (w, h).
    intrinsic: MuJoCo cam_intrinsic (fx, fy, cx, cy).
    znear: near plane distance.

  Returns:
    ray_orig: (H*W, 3) world-space ray origins (at near plane, matching render.py).
    ray_dir:  (H*W, 3) unit world-space ray directions.
    neg_z:    (H*W,) local -ray_dir_cam[2] (> 0) for z-depth conversion.
  """
  px = jnp.arange(width, dtype=jnp.float32)
  py = jnp.arange(height, dtype=jnp.float32)
  gx, gy = jnp.meshgrid(px, py, indexing="xy")  # (H, W)
  u = (gx + 0.5) / float(width)
  v = (gy + 0.5) / float(height)
  aspect = float(width) / float(height)

  if projection == PROJ_ORTHOGRAPHIC:
    sensor_h = jnp.asarray(fovy_deg, dtype=jnp.float32)
    sensor_w = sensor_h * aspect
    x = -0.5 * sensor_w + sensor_w * u
    y = 0.5 * sensor_h - sensor_h * v
    local_dir = jnp.broadcast_to(jnp.array([0.0, 0.0, -1.0], dtype=jnp.float32), (height, width, 3))
    local_off = jnp.stack([x, y, jnp.zeros_like(x)], axis=-1)
  else:
    sensor_w_val, sensor_h_val = float(sensorsize[0]), float(sensorsize[1])
    if sensor_h_val != 0.0:
      fx, fy, cx, cy = [float(val) for val in intrinsic]
      sw, sh = sensor_w_val, sensor_h_val
      if aspect > sw / sh:
        sh = sw / aspect
      elif aspect < sw / sh:
        sw = sh * aspect
      inv_fx_z = znear / fx
      inv_fy_z = znear / fy
      left = -inv_fx_z * (0.5 * sw - cx)
      right = inv_fx_z * (0.5 * sw + cx)
      top = inv_fy_z * (0.5 * sh - cy)
      bottom = -inv_fy_z * (0.5 * sh + cy)
      x = left + (right - left) * u
      y = top + (bottom - top) * v
    else:
      fovy_rad = jnp.asarray(fovy_deg, dtype=jnp.float32) * (math.pi / 180.0)
      half_h = znear * jnp.tan(0.5 * fovy_rad)
      half_w = half_h * aspect
      x = -half_w + (2.0 * half_w) * u
      y = half_h - (2.0 * half_h) * v

    raw = jnp.stack([x, y, jnp.full_like(x, -znear)], axis=-1)
    norm = jnp.linalg.norm(raw, axis=-1, keepdims=True)
    local_dir = raw / jnp.maximum(norm, 1e-12)
    local_off = jnp.zeros_like(local_dir)

  local_dir_flat = local_dir.reshape(-1, 3)  # (N, 3)
  local_off_flat = local_off.reshape(-1, 3)

  # cam_mat columns are camera axes in world coordinates: dir_world = cam_mat @ dir_cam
  ray_dir = jnp.einsum("ij,nj->ni", cam_mat, local_dir_flat, precision=jax.lax.Precision.HIGHEST)
  ray_dir = ray_dir / jnp.maximum(jnp.linalg.norm(ray_dir, axis=-1, keepdims=True), 1e-12)

  neg_z = jnp.maximum(-local_dir_flat[:, 2], 1e-6)
  inv_cos = 1.0 / neg_z
  ray_orig = cam_pos[None, :] + jnp.einsum("ij,nj->ni", cam_mat, local_off_flat, precision=jax.lax.Precision.HIGHEST)
  ray_orig = ray_orig + (znear * inv_cos)[:, None] * ray_dir
  return ray_orig, ray_dir, neg_z
