"""Branchless JAX analytic ray-primitive intersections matching mujoco_warp/_src/ray.py."""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp

# Official MuJoCo mjtGeom constants
GEOM_PLANE = 0
GEOM_SPHERE = 2
GEOM_CAPSULE = 3
GEOM_ELLIPSOID = 4
GEOM_CYLINDER = 5
GEOM_BOX = 6

MJ_MINVAL = 1.0e-15
MJ_MAXVAL = 1.0e10


def _safe_div(num: jax.Array, den: jax.Array) -> jax.Array:
  return jnp.where(jnp.abs(den) > MJ_MINVAL, num / jnp.where(jnp.abs(den) > MJ_MINVAL, den, 1.0), 0.0)


def _ray_map(
    pos: jax.Array, mat: jax.Array, pnt: jax.Array, vec: jax.Array
) -> Tuple[jax.Array, jax.Array]:
  """Map (N, 3) world rays into local geom frame defined by pos (..., 3), mat (..., 3, 3).

  Supports broadcasting over both (R, 3) rays and (G, ...) geoms.
  If pos/mat have shape (..., 3) / (..., 3, 3) and pnt/vec have (..., 3):
  we compute mat^T @ (pnt - pos) and mat^T @ vec.
  """
  diff = pnt - pos
  lpnt = jnp.einsum("...ji,...j->...i", mat, diff, precision=jax.lax.Precision.HIGHEST)
  lvec = jnp.einsum("...ji,...j->...i", mat, vec, precision=jax.lax.Precision.HIGHEST)
  return lpnt, lvec


def _ray_quad(a: jax.Array, b: jax.Array, c: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
  """Solve a * x^2 + 2 * b * x + c = 0. Returns (min_nonneg_x, x0, x1)."""
  det = b * b - a * c
  valid_det = (det >= MJ_MINVAL) & (jnp.abs(a) > MJ_MINVAL)
  sqrt_det = jnp.sqrt(jnp.maximum(det, 0.0))
  inv_a = _safe_div(jnp.ones_like(a), a)
  x0 = jnp.where(valid_det, (-b - sqrt_det) * inv_a, -1.0)
  x1 = jnp.where(valid_det, (-b + sqrt_det) * inv_a, -1.0)
  sol = jnp.where(x0 >= 0.0, x0, jnp.where(x1 >= 0.0, x1, -1.0))
  sol = jnp.where(valid_det, sol, -1.0)
  x0 = jnp.where(valid_det, x0, -1.0)
  x1 = jnp.where(valid_det, x1, -1.0)
  return sol, x0, x1


def _normalize3(v: jax.Array) -> jax.Array:
  n = jnp.linalg.norm(v, axis=-1, keepdims=True)
  return v / jnp.maximum(n, 1e-12)


def ray_plane(
    pos: jax.Array, mat: jax.Array, size: jax.Array, pnt: jax.Array, vec: jax.Array
) -> Tuple[jax.Array, jax.Array]:
  """Ray-plane intersection matching mujoco_warp ray_plane."""
  lpnt, lvec = _ray_map(pos, mat, pnt, vec)
  front = lvec[..., 2] <= -MJ_MINVAL
  denom = jnp.where(front, lvec[..., 2], -1.0)
  x = -lpnt[..., 2] / denom
  px = lpnt[..., 0] + x * lvec[..., 0]
  py = lpnt[..., 1] + x * lvec[..., 1]
  in_x = (size[..., 0] <= 0.0) | (jnp.abs(px) <= size[..., 0])
  in_y = (size[..., 1] <= 0.0) | (jnp.abs(py) <= size[..., 1])
  hit = front & (x >= 0.0) & in_x & in_y
  dist = jnp.where(hit, x, -1.0)
  nrm = jnp.broadcast_to(mat[..., :, 2], pnt.shape)
  nrm = jnp.where(hit[..., None], nrm, 0.0)
  return dist, nrm


def ray_sphere(
    pos: jax.Array, mat: jax.Array, size: jax.Array, pnt: jax.Array, vec: jax.Array
) -> Tuple[jax.Array, jax.Array]:
  """Ray-sphere intersection matching mujoco_warp ray_sphere."""
  del mat
  dif = pnt - pos
  a = jnp.sum(vec * vec, axis=-1)
  b = jnp.sum(vec * dif, axis=-1)
  c = jnp.sum(dif * dif, axis=-1) - size[..., 0] * size[..., 0]
  sol, _, _ = _ray_quad(a, b, c)
  hit = sol >= 0.0
  hit_p = pnt + vec * sol[..., None]
  nrm = _normalize3(hit_p - pos)
  nrm = jnp.where(hit[..., None], nrm, 0.0)
  return jnp.where(hit, sol, -1.0), nrm


def ray_ellipsoid(
    pos: jax.Array, mat: jax.Array, size: jax.Array, pnt: jax.Array, vec: jax.Array
) -> Tuple[jax.Array, jax.Array]:
  """Ray-ellipsoid intersection matching mujoco_warp ray_ellipsoid."""
  lpnt, lvec = _ray_map(pos, mat, pnt, vec)
  s = _safe_div(jnp.ones_like(size), size * size)
  slvec = s * lvec
  a = jnp.sum(slvec * lvec, axis=-1)
  b = jnp.sum(slvec * lpnt, axis=-1)
  c = jnp.sum((s * lpnt) * lpnt, axis=-1) - 1.0
  sol, _, _ = _ray_quad(a, b, c)
  hit = sol >= 0.0
  l_hit = lpnt + lvec * sol[..., None]
  lnrm = _normalize3(s * l_hit)
  nrm = jnp.einsum("...ij,...j->...i", mat, lnrm, precision=jax.lax.Precision.HIGHEST)
  nrm = jnp.where(hit[..., None], nrm, 0.0)
  return jnp.where(hit, sol, -1.0), nrm


def ray_capsule(
    pos: jax.Array, mat: jax.Array, size: jax.Array, pnt: jax.Array, vec: jax.Array
) -> Tuple[jax.Array, jax.Array]:
  """Ray-capsule intersection matching mujoco_warp ray_capsule."""
  r = size[..., 0]
  half_h = size[..., 1]
  sq_r = r * r

  # Bounding sphere reject (optional in vectorized mode; keeps math exact)
  lpnt, lvec = _ray_map(pos, mat, pnt, vec)

  # Cylinder body
  a_cyl = lvec[..., 0] * lvec[..., 0] + lvec[..., 1] * lvec[..., 1]
  b_cyl = lvec[..., 0] * lpnt[..., 0] + lvec[..., 1] * lpnt[..., 1]
  c_cyl = lpnt[..., 0] * lpnt[..., 0] + lpnt[..., 1] * lpnt[..., 1] - sq_r
  sol_cyl, _, _ = _ray_quad(a_cyl, b_cyl, c_cyl)
  z_cyl = lpnt[..., 2] + sol_cyl * lvec[..., 2]
  valid_cyl = (sol_cyl >= 0.0) & (jnp.abs(z_cyl) <= half_h)

  # Top cap (center +half_h)
  a_full = a_cyl + lvec[..., 2] * lvec[..., 2]
  dz_top = lpnt[..., 2] - half_h
  b_top = b_cyl + lvec[..., 2] * dz_top
  c_top = c_cyl + dz_top * dz_top
  _, t0_top, t1_top = _ray_quad(a_full, b_top, c_top)
  v0_top = (t0_top >= 0.0) & ((lpnt[..., 2] + t0_top * lvec[..., 2]) >= half_h)
  v1_top = (t1_top >= 0.0) & ((lpnt[..., 2] + t1_top * lvec[..., 2]) >= half_h)
  sol_top = jnp.where(v0_top & v1_top, jnp.minimum(t0_top, t1_top), jnp.where(v0_top, t0_top, jnp.where(v1_top, t1_top, -1.0)))
  valid_top = sol_top >= 0.0

  # Bottom cap (center -half_h)
  dz_bot = lpnt[..., 2] + half_h
  b_bot = b_cyl + lvec[..., 2] * dz_bot
  c_bot = c_cyl + dz_bot * dz_bot
  _, t0_bot, t1_bot = _ray_quad(a_full, b_bot, c_bot)
  v0_bot = (t0_bot >= 0.0) & ((lpnt[..., 2] + t0_bot * lvec[..., 2]) <= -half_h)
  v1_bot = (t1_bot >= 0.0) & ((lpnt[..., 2] + t1_bot * lvec[..., 2]) <= -half_h)
  sol_bot = jnp.where(v0_bot & v1_bot, jnp.minimum(t0_bot, t1_bot), jnp.where(v0_bot, t0_bot, jnp.where(v1_bot, t1_bot, -1.0)))
  valid_bot = sol_bot >= 0.0

  inf = jnp.full_like(sol_cyl, MJ_MAXVAL)
  d_cyl = jnp.where(valid_cyl, sol_cyl, inf)
  d_top = jnp.where(valid_top, sol_top, inf)
  d_bot = jnp.where(valid_bot, sol_bot, inf)

  best_d = jnp.minimum(d_cyl, jnp.minimum(d_top, d_bot))
  hit = best_d < (MJ_MAXVAL * 0.5)

  is_top = valid_top & (d_top <= d_cyl) & (d_top <= d_bot)
  is_bot = valid_bot & (d_bot < d_cyl) & (d_bot < d_top)
  z_shift = jnp.where(is_top, -half_h, jnp.where(is_bot, half_h, 0.0))

  hit_local = lpnt + lvec * jnp.where(hit, best_d, 0.0)[..., None]
  lnrm_z = jnp.where(is_top | is_bot, hit_local[..., 2] + z_shift, 0.0)
  lnrm = _normalize3(jnp.stack([hit_local[..., 0], hit_local[..., 1], lnrm_z], axis=-1))
  nrm = jnp.einsum("...ij,...j->...i", mat, lnrm, precision=jax.lax.Precision.HIGHEST)
  nrm = jnp.where(hit[..., None], nrm, 0.0)
  return jnp.where(hit, best_d, -1.0), nrm


def ray_cylinder(
    pos: jax.Array, mat: jax.Array, size: jax.Array, pnt: jax.Array, vec: jax.Array
) -> Tuple[jax.Array, jax.Array]:
  """Ray-cylinder intersection matching mujoco_warp ray_cylinder."""
  r = size[..., 0]
  half_h = size[..., 1]
  sq_r = r * r
  lpnt, lvec = _ray_map(pos, mat, pnt, vec)

  # Round side
  a = lvec[..., 0] * lvec[..., 0] + lvec[..., 1] * lvec[..., 1]
  b = lvec[..., 0] * lpnt[..., 0] + lvec[..., 1] * lpnt[..., 1]
  c = lpnt[..., 0] * lpnt[..., 0] + lpnt[..., 1] * lpnt[..., 1] - sq_r
  sol_side, _, _ = _ray_quad(a, b, c)
  valid_side = (sol_side >= 0.0) & (jnp.abs(lpnt[..., 2] + sol_side * lvec[..., 2]) <= half_h)

  # Flat caps (z = +half_h and -half_h)
  ok_z = jnp.abs(lvec[..., 2]) > MJ_MINVAL
  inv_lz = jnp.where(ok_z, 1.0 / jnp.where(ok_z, lvec[..., 2], 1.0), 0.0)

  t_top = (half_h - lpnt[..., 2]) * inv_lz
  p_top_x = lpnt[..., 0] + t_top * lvec[..., 0]
  p_top_y = lpnt[..., 1] + t_top * lvec[..., 1]
  valid_top = ok_z & (t_top >= 0.0) & ((p_top_x * p_top_x + p_top_y * p_top_y) <= sq_r)

  t_bot = (-half_h - lpnt[..., 2]) * inv_lz
  p_bot_x = lpnt[..., 0] + t_bot * lvec[..., 0]
  p_bot_y = lpnt[..., 1] + t_bot * lvec[..., 1]
  valid_bot = ok_z & (t_bot >= 0.0) & ((p_bot_x * p_bot_x + p_bot_y * p_bot_y) <= sq_r)

  inf = jnp.full_like(sol_side, MJ_MAXVAL)
  d_side = jnp.where(valid_side, sol_side, inf)
  d_top = jnp.where(valid_top, t_top, inf)
  d_bot = jnp.where(valid_bot, t_bot, inf)
  best_d = jnp.minimum(d_side, jnp.minimum(d_top, d_bot))
  hit = best_d < (MJ_MAXVAL * 0.5)

  is_top = valid_top & (d_top <= d_side) & (d_top <= d_bot)
  is_bot = valid_bot & (d_bot < d_side) & (d_bot < d_top)

  hit_local = lpnt + lvec * jnp.where(hit, best_d, 0.0)[..., None]
  n_side = _normalize3(jnp.stack([hit_local[..., 0], hit_local[..., 1], jnp.zeros_like(hit_local[..., 2])], axis=-1))
  lnrm = jnp.where(
      is_top[..., None],
      jnp.array([0.0, 0.0, 1.0], dtype=lpnt.dtype),
      jnp.where(is_bot[..., None], jnp.array([0.0, 0.0, -1.0], dtype=lpnt.dtype), n_side),
  )
  nrm = jnp.einsum("...ij,...j->...i", mat, lnrm, precision=jax.lax.Precision.HIGHEST)
  nrm = jnp.where(hit[..., None], nrm, 0.0)
  return jnp.where(hit, best_d, -1.0), nrm


def ray_box(
    pos: jax.Array, mat: jax.Array, size: jax.Array, pnt: jax.Array, vec: jax.Array
) -> Tuple[jax.Array, jax.Array]:
  """Branchless exact ray-box intersection matching mujoco_warp ray_box."""
  lpnt, lvec = _ray_map(pos, mat, pnt, vec)

  # Evaluate all 6 faces (axis 0,1,2 x side -1,+1) uniformly without scalar loops
  ok = jnp.abs(lvec) > MJ_MINVAL  # (..., 3)
  inv_lvec = jnp.where(ok, 1.0 / jnp.where(ok, lvec, 1.0), 0.0)

  # Positive and negative faces: shape (..., 3, 2)
  sides = jnp.array([-1.0, 1.0], dtype=lpnt.dtype)
  # target: (..., 3, 2)
  target = sides * size[..., None]
  sol = (target - lpnt[..., None]) * inv_lvec[..., None]  # (..., 3, 2)

  # Compute hit coordinates along all 3 axes for each of the 6 candidate planes
  # lpnt: (..., 3) -> (..., 3, 1, 1) wait: face (i, s) hits at p_k = lpnt_k + sol_{i,s} * lvec_k
  # For axis i, check the other two axes j != i: |p_j| <= size_j
  p_hit = lpnt[..., None, None, :] + sol[..., :, :, None] * lvec[..., None, None, :]  # (..., 3, 2, 3)
  in_bounds = jnp.abs(p_hit) <= size[..., None, None, :]  # (..., 3, 2, 3)
  # Ignore self-axis bound (always 0 error up to fp rounding) via mask
  not_self = 1.0 - jnp.eye(3, dtype=lpnt.dtype)  # (3, 3)
  in_other = jnp.all(in_bounds | (not_self[:, None, :] == 0.0), axis=-1)  # (..., 3, 2)

  valid = ok[..., None] & (sol >= 0.0) & in_other  # (..., 3, 2)
  d_faces = jnp.where(valid, sol, MJ_MAXVAL)  # (..., 3, 2)
  d_flat = d_faces.reshape(*lpnt.shape[:-1], 6)
  best_idx = jnp.argmin(d_flat, axis=-1)  # (...,)
  best_d = jnp.take_along_axis(d_flat, best_idx[..., None], axis=-1)[..., 0]
  hit = best_d < (MJ_MAXVAL * 0.5)

  # Map best_idx (0..5) -> local normal
  face_normals = jnp.array(
      [
          [-1.0, 0.0, 0.0],
          [1.0, 0.0, 0.0],
          [0.0, -1.0, 0.0],
          [0.0, 1.0, 0.0],
          [0.0, 0.0, -1.0],
          [0.0, 0.0, 1.0],
      ],
      dtype=lpnt.dtype,
  )
  lnrm = face_normals[best_idx]
  nrm = jnp.einsum("...ij,...j->...i", mat, lnrm, precision=jax.lax.Precision.HIGHEST)
  nrm = jnp.where(hit[..., None], nrm, 0.0)
  return jnp.where(hit, best_d, -1.0), nrm


RAY_FN_BY_TYPE = {
    GEOM_PLANE: ray_plane,
    GEOM_SPHERE: ray_sphere,
    GEOM_CAPSULE: ray_capsule,
    GEOM_ELLIPSOID: ray_ellipsoid,
    GEOM_CYLINDER: ray_cylinder,
    GEOM_BOX: ray_box,
}
