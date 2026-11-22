"""V2 Extreme-Throughput Batched SoA & Native (BLOCK_B=8, BLOCK_R=128) Pallas TPU Raytracer.

Key V2 optimizations over V1:
1. Eliminate `jax.vmap(render_exact_gbuffer)` per-world scalar overhead via native
   (B, R) Structure-of-Arrays (SoA) batching across all worlds simultaneously.
2. In-kernel camera ray generation from compact (8, 12) camera poses + static
   (R, 4) camera-space ray templates (eliminates (B, R, 3) ray HBM write/read).
3. Native TPU Mosaic (BLOCK_B=8, BLOCK_R=128) VPU register tiling with strict
   2D (8, 128) / (128, 128) BlockSpecs and zero sublane padding waste.
4. Coarse bounding-sphere reject + in-VMEM fused shadow `any_hit` raycasting
   (skips normals and sorting on shadow rays).
"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
import numpy as np

from tpu_mjx_render.camera import PROJ_ORTHOGRAPHIC
from tpu_mjx_render.primitives import (
    GEOM_BOX,
    GEOM_CAPSULE,
    GEOM_CYLINDER,
    GEOM_ELLIPSOID,
    GEOM_PLANE,
    GEOM_SPHERE,
    MJ_MAXVAL,
    MJ_MINVAL,
    RAY_FN_BY_TYPE,
)

BLOCK_B = 8
BLOCK_R = 512


def _pad_axis(arr: jax.Array, multiple: int, axis: int = 0, const_val: float = 0.0) -> jax.Array:
  n = arr.shape[axis]
  rem = n % multiple
  if rem == 0:
    return arr
  pw = [(0, 0)] * arr.ndim
  pw[axis] = (0, multiple - rem)
  return jnp.pad(arr, pw, constant_values=const_val)


def build_local_camera_rays(spec: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Precompute static camera-space ray directions, origins (ortho), and neg_z."""
  H, W = int(spec.height), int(spec.width)
  px = np.arange(W, dtype=np.float32)
  py = np.arange(H, dtype=np.float32)
  gx, gy = np.meshgrid(px, py, indexing="xy")
  u = (gx + 0.5) / float(W)
  v = (gy + 0.5) / float(H)
  aspect = float(W) / float(H)

  if int(spec.cam_projection) == PROJ_ORTHOGRAPHIC:
    sensor_h = float(spec.cam_fovy)
    sensor_w = sensor_h * aspect
    x = -0.5 * sensor_w + sensor_w * u
    y = 0.5 * sensor_h - sensor_h * v
    local_dir = np.broadcast_to(np.array([0.0, 0.0, -1.0], dtype=np.float32), (H, W, 3)).copy()
    local_off = np.stack([x, y, np.zeros_like(x)], axis=-1).astype(np.float32)
  else:
    sw_val, sh_val = float(spec.cam_sensorsize[0]), float(spec.cam_sensorsize[1])
    if sh_val != 0.0:
      fx, fy, cx, cy = [float(val) for val in spec.cam_intrinsic]
      sw, sh = sw_val, sh_val
      if aspect > sw / sh:
        sh = sw / aspect
      elif aspect < sw / sh:
        sw = sh * aspect
      inv_fx_z = float(spec.znear) / fx
      inv_fy_z = float(spec.znear) / fy
      left = -inv_fx_z * (0.5 * sw - cx)
      right = inv_fx_z * (0.5 * sw + cx)
      top = inv_fy_z * (0.5 * sh - cy)
      bottom = -inv_fy_z * (0.5 * sh + cy)
      x = left + (right - left) * u
      y = top + (bottom - top) * v
    else:
      fovy_rad = float(spec.cam_fovy) * (math.pi / 180.0)
      half_h = float(spec.znear) * math.tan(0.5 * fovy_rad)
      half_w = half_h * aspect
      x = -half_w + (2.0 * half_w) * u
      y = half_h - (2.0 * half_h) * v
    raw = np.stack([x, y, np.full_like(x, -float(spec.znear))], axis=-1).astype(np.float32)
    norm = np.maximum(np.linalg.norm(raw, axis=-1, keepdims=True), 1e-12)
    local_dir = (raw / norm).astype(np.float32)
    local_off = np.zeros_like(local_dir)

  ld = local_dir.reshape(-1, 3)
  lo = local_off.reshape(-1, 3)
  neg_z = np.maximum(-ld[:, 2], 1e-6).astype(np.float32)
  return ld, lo, neg_z


def _compute_geom_bound_sq(spec: Any, gids: np.ndarray) -> np.ndarray:
  """Tight per-geom bounding radius^2 for coarse sphere reject (inf for planes)."""
  out = np.zeros((len(gids),), dtype=np.float32)
  for k, gid in enumerate(gids):
    gt = int(spec.geom_type[gid])
    sx, sy, sz = [float(x) for x in spec.geom_size[gid]]
    if gt == GEOM_PLANE:
      out[k] = 1.0e20
    elif gt == GEOM_SPHERE:
      out[k] = (sx + 1e-4) ** 2
    elif gt == GEOM_CAPSULE:
      out[k] = (sx + sy + 1e-4) ** 2
    elif gt == GEOM_CYLINDER:
      out[k] = sx * sx + sy * sy + 1e-3
    elif gt == GEOM_ELLIPSOID:
      out[k] = (max(sx, sy, sz) + 1e-4) ** 2
    else:  # box
      out[k] = sx * sx + sy * sy + sz * sz + 1e-3
  return out


def _compute_batch_camera_rays_soa(
    spec: Any, cam_pos: jax.Array, cam_mat: jax.Array
) -> Tuple[jax.Array, jax.Array, jax.Array]:
  """Fused (B, R, 3) ray generation directly from batched camera poses."""
  ld_np, lo_np, neg_z_np = build_local_camera_rays(spec)
  ld = jnp.asarray(ld_np, dtype=jnp.float32)      # (R, 3)
  lo = jnp.asarray(lo_np, dtype=jnp.float32)      # (R, 3)
  neg_z = jnp.asarray(neg_z_np, dtype=jnp.float32)  # (R,)

  # ray_dir[b, r, i] = sum_j cam_mat[b, i, j] * ld[r, j]
  ray_dir = jnp.einsum("bij,rj->bri", cam_mat, ld, precision=jax.lax.Precision.HIGHEST)
  ray_dir = ray_dir / jnp.maximum(jnp.linalg.norm(ray_dir, axis=-1, keepdims=True), 1e-12)
  ray_off = jnp.einsum("bij,rj->bri", cam_mat, lo, precision=jax.lax.Precision.HIGHEST)
  inv_cos = (float(spec.znear) / neg_z)[None, :, None]
  ray_orig = cam_pos[:, None, :] + ray_off + inv_cos * ray_dir
  return ray_orig, ray_dir, neg_z


def _ray_one_geom_v2(
    gt: int,
    px, py, pz,
    m00, m01, m02,
    m10, m11, m12,
    m20, m21, m22,
    sx, sy, sz,
    rb_sq,
    rox, roy, roz,
    rdx, rdy, rdz,
    plane_finite_x: bool = True,
    plane_finite_y: bool = True,
    compute_normal: bool = True,
):
  """Branchless SoA primitive intersection on arbitrary broadcastable shapes (e.g. (8, 128))."""
  dx = rox - px
  dy = roy - py
  dz = roz - pz

  # Coarse bounding-sphere reject (exact upper bound outside plane)
  if gt != GEOM_PLANE:
    proj = dx * rdx + dy * rdy + dz * rdz
    d2_perp = (dx * dx + dy * dy + dz * dz) - proj * proj
    sphere_hit = d2_perp <= rb_sq
  else:
    sphere_hit = True

  lx = m00 * dx + m10 * dy + m20 * dz
  ly = m01 * dx + m11 * dy + m21 * dz
  lz = m02 * dx + m12 * dy + m22 * dz

  vx = m00 * rdx + m10 * rdy + m20 * rdz
  vy = m01 * rdx + m11 * rdy + m21 * rdz
  vz = m02 * rdx + m12 * rdy + m22 * rdz

  if gt == GEOM_PLANE:
    front = vz <= -MJ_MINVAL
    den = jnp.where(front, vz, -1.0)
    t = -lz / den
    hx = lx + t * vx
    hy = ly + t * vy
    ok = front & (t >= 0.0)
    if plane_finite_x:
      ok = ok & (jnp.abs(hx) <= sx)
    if plane_finite_y:
      ok = ok & (jnp.abs(hy) <= sy)
    cand_d = jnp.where(ok, t, MJ_MAXVAL)
    if compute_normal:
      return cand_d, m02, m12, m22
    return cand_d, 0.0, 0.0, 0.0

  if gt == GEOM_SPHERE:
    a = rdx * rdx + rdy * rdy + rdz * rdz
    b = rdx * dx + rdy * dy + rdz * dz
    c = dx * dx + dy * dy + dz * dz - sx * sx
    det = b * b - a * c
    valid = sphere_hit & (det >= MJ_MINVAL) & (jnp.abs(a) > MJ_MINVAL)
    sd = jnp.sqrt(jnp.maximum(det, 0.0))
    inv_a = jnp.where(valid, 1.0 / jnp.where(valid, a, 1.0), 0.0)
    t0 = (-b - sd) * inv_a
    t1 = (-b + sd) * inv_a
    t = jnp.where(t0 >= 0.0, t0, jnp.where(t1 >= 0.0, t1, -1.0))
    ok = valid & (t >= 0.0)
    cand_d = jnp.where(ok, t, MJ_MAXVAL)
    if not compute_normal:
      return cand_d, 0.0, 0.0, 0.0
    ts = jnp.where(ok, t, 0.0)
    nx_w = dx + rdx * ts
    ny_w = dy + rdy * ts
    nz_w = dz + rdz * ts
    inv_n = 1.0 / jnp.maximum(jnp.sqrt(nx_w * nx_w + ny_w * ny_w + nz_w * nz_w), 1e-12)
    return cand_d, nx_w * inv_n, ny_w * inv_n, nz_w * inv_n

  if gt == GEOM_ELLIPSOID:
    ix2 = 1.0 / jnp.maximum(sx * sx, MJ_MINVAL)
    iy2 = 1.0 / jnp.maximum(sy * sy, MJ_MINVAL)
    iz2 = 1.0 / jnp.maximum(sz * sz, MJ_MINVAL)
    a = ix2 * vx * vx + iy2 * vy * vy + iz2 * vz * vz
    b = ix2 * vx * lx + iy2 * vy * ly + iz2 * vz * lz
    c = ix2 * lx * lx + iy2 * ly * ly + iz2 * lz * lz - 1.0
    det = b * b - a * c
    valid = sphere_hit & (det >= MJ_MINVAL) & (jnp.abs(a) > MJ_MINVAL)
    sd = jnp.sqrt(jnp.maximum(det, 0.0))
    inv_a = jnp.where(valid, 1.0 / jnp.where(valid, a, 1.0), 0.0)
    t0 = (-b - sd) * inv_a
    t1 = (-b + sd) * inv_a
    t = jnp.where(t0 >= 0.0, t0, jnp.where(t1 >= 0.0, t1, -1.0))
    ok = valid & (t >= 0.0)
    cand_d = jnp.where(ok, t, MJ_MAXVAL)
    if not compute_normal:
      return cand_d, 0.0, 0.0, 0.0
    ts = jnp.where(ok, t, 0.0)
    lnx = ix2 * (lx + vx * ts)
    lny = iy2 * (ly + vy * ts)
    lnz = iz2 * (lz + vz * ts)
    inv_n = 1.0 / jnp.maximum(jnp.sqrt(lnx * lnx + lny * lny + lnz * lnz), 1e-12)
    lnx, lny, lnz = lnx * inv_n, lny * inv_n, lnz * inv_n
    return (
        cand_d,
        m00 * lnx + m01 * lny + m02 * lnz,
        m10 * lnx + m11 * lny + m12 * lnz,
        m20 * lnx + m21 * lny + m22 * lnz,
    )

  if gt == GEOM_CAPSULE:
    sq_r = sx * sx
    ac = vx * vx + vy * vy
    bc = vx * lx + vy * ly
    cc = lx * lx + ly * ly - sq_r
    detc = bc * bc - ac * cc
    vc = sphere_hit & (detc >= MJ_MINVAL) & (jnp.abs(ac) > MJ_MINVAL)
    sdc = jnp.sqrt(jnp.maximum(detc, 0.0))
    iac = jnp.where(vc, 1.0 / jnp.where(vc, ac, 1.0), 0.0)
    tc0 = (-bc - sdc) * iac
    tc1 = (-bc + sdc) * iac
    tc = jnp.where(tc0 >= 0.0, tc0, jnp.where(tc1 >= 0.0, tc1, -1.0))
    ok_cyl = vc & (tc >= 0.0) & (jnp.abs(lz + tc * vz) <= sy)
    dc = jnp.where(ok_cyl, tc, MJ_MAXVAL)

    af = ac + vz * vz
    dzt = lz - sy
    bt = bc + vz * dzt
    ct = cc + dzt * dzt
    dett = bt * bt - af * ct
    vt = sphere_hit & (dett >= MJ_MINVAL) & (jnp.abs(af) > MJ_MINVAL)
    sdt = jnp.sqrt(jnp.maximum(dett, 0.0))
    iaf = jnp.where(vt, 1.0 / jnp.where(vt, af, 1.0), 0.0)
    tt0 = (-bt - sdt) * iaf
    tt1 = (-bt + sdt) * iaf
    vt0 = vt & (tt0 >= 0.0) & ((lz + tt0 * vz) >= sy)
    vt1 = vt & (tt1 >= 0.0) & ((lz + tt1 * vz) >= sy)
    tt = jnp.where(vt0 & vt1, jnp.minimum(tt0, tt1), jnp.where(vt0, tt0, jnp.where(vt1, tt1, -1.0)))
    ok_top = tt >= 0.0
    dt = jnp.where(ok_top, tt, MJ_MAXVAL)

    dzb = lz + sy
    bb = bc + vz * dzb
    cb = cc + dzb * dzb
    detb = bb * bb - af * cb
    vb = sphere_hit & (detb >= MJ_MINVAL) & (jnp.abs(af) > MJ_MINVAL)
    sdb = jnp.sqrt(jnp.maximum(detb, 0.0))
    iab = jnp.where(vb, 1.0 / jnp.where(vb, af, 1.0), 0.0)
    tb0 = (-bb - sdb) * iab
    tb1 = (-bb + sdb) * iab
    vb0 = vb & (tb0 >= 0.0) & ((lz + tb0 * vz) <= -sy)
    vb1 = vb & (tb1 >= 0.0) & ((lz + tb1 * vz) <= -sy)
    tb = jnp.where(vb0 & vb1, jnp.minimum(tb0, tb1), jnp.where(vb0, tb0, jnp.where(vb1, tb1, -1.0)))
    ok_bot = tb >= 0.0
    db = jnp.where(ok_bot, tb, MJ_MAXVAL)

    cand_d = jnp.minimum(dc, jnp.minimum(dt, db))
    if not compute_normal:
      return cand_d, 0.0, 0.0, 0.0
    ok = cand_d < (MJ_MAXVAL * 0.5)
    is_top = ok_top & (dt <= dc) & (dt <= db)
    is_bot = ok_bot & (db < dc) & (db < dt)
    ts = jnp.where(ok, cand_d, 0.0)
    lnx = lx + vx * ts
    lny = ly + vy * ts
    lnz = jnp.where(is_top | is_bot, lz + vz * ts + jnp.where(is_top, -sy, sy), 0.0)
    inv_n = 1.0 / jnp.maximum(jnp.sqrt(lnx * lnx + lny * lny + lnz * lnz), 1e-12)
    lnx, lny, lnz = lnx * inv_n, lny * inv_n, lnz * inv_n
    return (
        cand_d,
        m00 * lnx + m01 * lny + m02 * lnz,
        m10 * lnx + m11 * lny + m12 * lnz,
        m20 * lnx + m21 * lny + m22 * lnz,
    )

  if gt == GEOM_CYLINDER:
    sq_r = sx * sx
    ac = vx * vx + vy * vy
    bc = vx * lx + vy * ly
    cc = lx * lx + ly * ly - sq_r
    detc = bc * bc - ac * cc
    vc = sphere_hit & (detc >= MJ_MINVAL) & (jnp.abs(ac) > MJ_MINVAL)
    sdc = jnp.sqrt(jnp.maximum(detc, 0.0))
    iac = jnp.where(vc, 1.0 / jnp.where(vc, ac, 1.0), 0.0)
    tc0 = (-bc - sdc) * iac
    tc1 = (-bc + sdc) * iac
    tc = jnp.where(tc0 >= 0.0, tc0, jnp.where(tc1 >= 0.0, tc1, -1.0))
    ok_s = vc & (tc >= 0.0) & (jnp.abs(lz + tc * vz) <= sy)
    ds = jnp.where(ok_s, tc, MJ_MAXVAL)

    ok_z = sphere_hit & (jnp.abs(vz) > MJ_MINVAL)
    ivz = jnp.where(ok_z, 1.0 / jnp.where(ok_z, vz, 1.0), 0.0)
    tt = (sy - lz) * ivz
    ok_t = ok_z & (tt >= 0.0) & (((lx + tt * vx) ** 2 + (ly + tt * vy) ** 2) <= sq_r)
    dt = jnp.where(ok_t, tt, MJ_MAXVAL)
    tb = (-sy - lz) * ivz
    ok_b = ok_z & (tb >= 0.0) & (((lx + tb * vx) ** 2 + (ly + tb * vy) ** 2) <= sq_r)
    db = jnp.where(ok_b, tb, MJ_MAXVAL)

    cand_d = jnp.minimum(ds, jnp.minimum(dt, db))
    if not compute_normal:
      return cand_d, 0.0, 0.0, 0.0
    ok = cand_d < (MJ_MAXVAL * 0.5)
    is_top = ok_t & (dt <= ds) & (dt <= db)
    is_bot = ok_b & (db < ds) & (db < dt)
    ts = jnp.where(ok, cand_d, 0.0)
    hlx = lx + vx * ts
    hly = ly + vy * ts
    inv_n = 1.0 / jnp.maximum(jnp.sqrt(hlx * hlx + hly * hly), 1e-12)
    lnx = jnp.where(is_top | is_bot, 0.0, hlx * inv_n)
    lny = jnp.where(is_top | is_bot, 0.0, hly * inv_n)
    lnz = jnp.where(is_top, 1.0, jnp.where(is_bot, -1.0, 0.0))
    return (
        cand_d,
        m00 * lnx + m01 * lny + m02 * lnz,
        m10 * lnx + m11 * lny + m12 * lnz,
        m20 * lnx + m21 * lny + m22 * lnz,
    )

  # GEOM_BOX
  okx = sphere_hit & (jnp.abs(vx) > MJ_MINVAL)
  ivx = jnp.where(okx, 1.0 / jnp.where(okx, vx, 1.0), 0.0)
  tx0 = (-sx - lx) * ivx
  tx1 = (sx - lx) * ivx
  v_x0 = okx & (tx0 >= 0.0) & (jnp.abs(ly + tx0 * vy) <= sy) & (jnp.abs(lz + tx0 * vz) <= sz)
  v_x1 = okx & (tx1 >= 0.0) & (jnp.abs(ly + tx1 * vy) <= sy) & (jnp.abs(lz + tx1 * vz) <= sz)
  dx0 = jnp.where(v_x0, tx0, MJ_MAXVAL)
  dx1 = jnp.where(v_x1, tx1, MJ_MAXVAL)

  oky = sphere_hit & (jnp.abs(vy) > MJ_MINVAL)
  ivy = jnp.where(oky, 1.0 / jnp.where(oky, vy, 1.0), 0.0)
  ty0 = (-sy - ly) * ivy
  ty1 = (sy - ly) * ivy
  v_y0 = oky & (ty0 >= 0.0) & (jnp.abs(lx + ty0 * vx) <= sx) & (jnp.abs(lz + ty0 * vz) <= sz)
  v_y1 = oky & (ty1 >= 0.0) & (jnp.abs(lx + ty1 * vx) <= sx) & (jnp.abs(lz + ty1 * vz) <= sz)
  dy0 = jnp.where(v_y0, ty0, MJ_MAXVAL)
  dy1 = jnp.where(v_y1, ty1, MJ_MAXVAL)

  okz = sphere_hit & (jnp.abs(vz) > MJ_MINVAL)
  ivz = jnp.where(okz, 1.0 / jnp.where(okz, vz, 1.0), 0.0)
  tz0 = (-sz - lz) * ivz
  tz1 = (sz - lz) * ivz
  v_z0 = okz & (tz0 >= 0.0) & (jnp.abs(lx + tz0 * vx) <= sx) & (jnp.abs(ly + tz0 * vy) <= sy)
  v_z1 = okz & (tz1 >= 0.0) & (jnp.abs(lx + tz1 * vx) <= sx) & (jnp.abs(ly + tz1 * vy) <= sy)
  dz0 = jnp.where(v_z0, tz0, MJ_MAXVAL)
  dz1 = jnp.where(v_z1, tz1, MJ_MAXVAL)

  cand_d = jnp.minimum(jnp.minimum(dx0, dx1), jnp.minimum(jnp.minimum(dy0, dy1), jnp.minimum(dz0, dz1)))
  if not compute_normal:
    return cand_d, 0.0, 0.0, 0.0
  lnx = jnp.where(cand_d == dx0, -1.0, jnp.where(cand_d == dx1, 1.0, 0.0))
  lny = jnp.where(
      (cand_d != dx0) & (cand_d != dx1) & (cand_d == dy0),
      -1.0,
      jnp.where((cand_d != dx0) & (cand_d != dx1) & (cand_d == dy1), 1.0, 0.0),
  )
  lnz = jnp.where(
      (cand_d != dx0) & (cand_d != dx1) & (cand_d != dy0) & (cand_d != dy1) & (cand_d == dz0),
      -1.0,
      jnp.where((cand_d != dx0) & (cand_d != dx1) & (cand_d != dy0) & (cand_d != dy1) & (cand_d == dz1), 1.0, 0.0),
  )
  return (
      cand_d,
      m00 * lnx + m01 * lny + m02 * lnz,
      m10 * lnx + m11 * lny + m12 * lnz,
      m20 * lnx + m21 * lny + m22 * lnz,
  )


def _anyhit_shadow_soa(
    spec: Any,
    geom_xpos: jax.Array,
    geom_xmat: jax.Array,
    geom_size: jax.Array,
    sh_orig: jax.Array,
    sh_dir: jax.Array,
    sh_max: jax.Array,
) -> jax.Array:
  """Vectorized (B, R) shadow any-hit test skipping all normal computations."""
  B, R, _ = sh_orig.shape
  ox, oy, oz = sh_orig[..., 0], sh_orig[..., 1], sh_orig[..., 2]
  dx, dy, dz = sh_dir[..., 0], sh_dir[..., 1], sh_dir[..., 2]
  gids = np.asarray(spec.enabled_geom_ids, dtype=np.int32)
  rb_sq_all = _compute_geom_bound_sq(spec, gids)

  occluded = jnp.zeros((B, R), dtype=jnp.bool_)
  for k, gid_val in enumerate(gids):
    gt = int(spec.geom_type[gid_val])
    px = geom_xpos[:, gid_val, 0:1]
    py = geom_xpos[:, gid_val, 1:2]
    pz = geom_xpos[:, gid_val, 2:3]
    mat = geom_xmat[:, gid_val, :, :]
    sz = spec.geom_size[gid_val]
    cand_d, _, _, _ = _ray_one_geom_v2(
        gt,
        px, py, pz,
        mat[:, 0, 0:1], mat[:, 0, 1:2], mat[:, 0, 2:3],
        mat[:, 1, 0:1], mat[:, 1, 1:2], mat[:, 1, 2:3],
        mat[:, 2, 0:1], mat[:, 2, 1:2], mat[:, 2, 2:3],
        float(sz[0]), float(sz[1]), float(sz[2]),
        float(rb_sq_all[k]),
        ox, oy, oz, dx, dy, dz,
        plane_finite_x=float(spec.geom_size[gid_val, 0]) > 0.0,
        plane_finite_y=float(spec.geom_size[gid_val, 1]) > 0.0,
        compute_normal=False,
    )
    occluded = occluded | (cand_d < sh_max)
  return occluded


def _shade_batch_soa(
    spec: Any,
    geom_xpos: jax.Array,
    geom_xmat: jax.Array,
    geom_size: jax.Array,
    geom_rgba: jax.Array,
    light_xpos: jax.Array,
    light_xdir: jax.Array,
    ray_orig: jax.Array,
    ray_dir: jax.Array,
    neg_z: jax.Array,
    dist: jax.Array,
    normal: jax.Array,
    geom_id: jax.Array,
    fused_shadow0_occluded: Optional[jax.Array] = None,
):
  """Compute exact direct RGB, checker albedo, and pack GBuffer for (B, R) rays."""
  from tpu_mjx_render.renderer import GBuffer

  B, R, _ = ray_orig.shape
  H, W = int(spec.height), int(spec.width)
  hit = geom_id >= 0
  safe_gid = jnp.maximum(geom_id, 0)

  hit_pos = ray_orig + jnp.where(hit, dist, 0.0)[..., None] * ray_dir
  view_dir = -ray_dir

  # Albedo + checkerboard plane
  # geom_rgba: (B, ngeom, 4) -> gather per (b, r)
  base_rgba = jnp.take_along_axis(
      geom_rgba[:, :, :3][:, None, :, :] if False else geom_rgba,
      safe_gid[..., None],
      axis=1,
  ) if geom_rgba.ndim == 3 else geom_rgba[safe_gid, :3]
  albedo = base_rgba[..., :3]
  chk_gids = [i for i, c in enumerate(spec.geom_is_checker) if bool(c)]
  if chk_gids:
    rep_x, rep_y = spec.checker_repeat
    rgb1 = jnp.asarray(spec.checker_rgb1, dtype=jnp.float32)
    rgb2 = jnp.asarray(spec.checker_rgb2, dtype=jnp.float32)
    for cg in chk_gids:
      gp = geom_xpos[:, cg, :][:, None, :]
      gm = geom_xmat[:, cg, :, :]
      dcg = hit_pos - gp
      lx_cg = gm[:, 0:1, 0] * dcg[..., 0] + gm[:, 1:2, 0] * dcg[..., 1] + gm[:, 2:3, 0] * dcg[..., 2]
      ly_cg = gm[:, 0:1, 1] * dcg[..., 0] + gm[:, 1:2, 1] * dcg[..., 1] + gm[:, 2:3, 1] * dcg[..., 2]
      cx = jnp.floor(lx_cg * rep_x).astype(jnp.int32)
      cy = jnp.floor(ly_cg * rep_y).astype(jnp.int32)
      even = ((cx + cy) & 1) == 0
      chk_rgb = jnp.where(even[..., None], rgb1[None, None, :], rgb2[None, None, :])
      albedo = jnp.where((geom_id == cg)[..., None], chk_rgb, albedo)

  spec_coef = jnp.asarray(spec.geom_spec, dtype=jnp.float32)[safe_gid]
  shin_exp = jnp.asarray(spec.geom_shin, dtype=jnp.float32)[safe_gid] * 128.0
  emit_coef = jnp.asarray(spec.geom_emit, dtype=jnp.float32)[safe_gid]

  diff_accum = jnp.zeros((B, R, 3), dtype=jnp.float32)
  spec_accum = jnp.zeros((B, R, 3), dtype=jnp.float32)
  shadow_vis_accum = jnp.ones((B, R), dtype=jnp.float32)

  shadow_slot_used = False
  for l_idx in range(int(spec.nlight)):
    if not bool(spec.light_active[l_idx]):
      continue
    l_type = int(spec.light_type[l_idx])
    l_pos = light_xpos[:, l_idx, :]   # (B, 3)
    l_dir = light_xdir[:, l_idx, :]   # (B, 3)
    l_dir = l_dir / jnp.maximum(jnp.linalg.norm(l_dir, axis=-1, keepdims=True), 1e-12)
    l_amb = jnp.asarray(spec.light_ambient[l_idx], dtype=jnp.float32)
    l_diff = jnp.asarray(spec.light_diffuse[l_idx], dtype=jnp.float32)
    l_spec = jnp.asarray(spec.light_specular[l_idx], dtype=jnp.float32)
    l_att = jnp.asarray(spec.light_attenuation[l_idx], dtype=jnp.float32)

    diff_accum = diff_accum + l_amb[None, None, :]
    if l_type == 1:
      L = jnp.broadcast_to(-l_dir[:, None, :], (B, R, 3))
      dist_to_light = jnp.full((B, R), 1.0e8, dtype=jnp.float32)
      attenuation = jnp.ones((B, R), dtype=jnp.float32)
    else:
      delta = l_pos[:, None, :] - hit_pos
      dist_to_light = jnp.linalg.norm(delta, axis=-1)
      L = delta / jnp.maximum(dist_to_light[..., None], 1e-12)
      att_denom = l_att[0] + l_att[1] * dist_to_light + l_att[2] * dist_to_light * dist_to_light
      attenuation = jnp.where(att_denom > 1e-12, 1.0 / jnp.maximum(att_denom, 1e-12), 1.0)
      if l_type == 0:
        cos_theta = jnp.sum(-L * l_dir[:, None, :], axis=-1)
        cos_cut = float(np.cos(spec.light_cutoff_rad[l_idx]))
        exp_val = float(spec.light_exponent[l_idx])
        in_cone = cos_theta >= cos_cut
        attenuation = jnp.where(in_cone, attenuation * (jnp.maximum(cos_theta, 0.0) ** exp_val), 0.0)

    ndotl = jnp.maximum(0.0, jnp.sum(normal * L, axis=-1))
    vis = jnp.ones((B, R), dtype=jnp.float32)
    if spec.use_shadows and bool(spec.light_castshadow[l_idx]):
      if (fused_shadow0_occluded is not None) and (not shadow_slot_used):
        occluded = hit & (ndotl > 0.0) & fused_shadow0_occluded
        shadow_slot_used = True
      else:
        sh_orig = hit_pos + normal * 1.0e-4
        sh_max = jnp.where(hit & (ndotl > 0.0), jnp.maximum(dist_to_light - 1.0e-3, 0.0), 0.0)
        occluded = hit & (ndotl > 0.0) & _anyhit_shadow_soa(
            spec, geom_xpos, geom_xmat, geom_size, sh_orig, L, sh_max
        )
      vis = jnp.where(occluded, float(spec.shadow_light_fraction), 1.0)
      if l_idx == 0:
        shadow_vis_accum = vis

    weight = attenuation * vis
    diff_accum = diff_accum + l_diff[None, None, :] * (ndotl * weight)[..., None]
    if spec.enable_specular:
      H_vec = L + view_dir
      H_vec = H_vec / jnp.maximum(jnp.linalg.norm(H_vec, axis=-1, keepdims=True), 1e-12)
      ndoth = jnp.maximum(0.0, jnp.sum(normal * H_vec, axis=-1))
      spec_w = jnp.where((spec_coef > 0.0) & (shin_exp > 0.0), spec_coef * (ndoth ** shin_exp) * weight, 0.0)
      spec_accum = spec_accum + l_spec[None, None, :] * spec_w[..., None]

  if spec.headlight_active:
    hl_amb = jnp.asarray(spec.headlight_ambient, dtype=jnp.float32)
    hl_diff = jnp.asarray(spec.headlight_diffuse, dtype=jnp.float32)
    hl_spec = jnp.asarray(spec.headlight_specular, dtype=jnp.float32)
    L_hl = -ray_dir
    ndotl_hl = jnp.maximum(0.0, jnp.sum(normal * L_hl, axis=-1))
    diff_accum = diff_accum + hl_amb[None, None, :] + hl_diff[None, None, :] * ndotl_hl[..., None]
    if spec.enable_specular:
      spec_w_hl = jnp.where((spec_coef > 0.0) & (shin_exp > 0.0), spec_coef * (ndotl_hl ** shin_exp), 0.0)
      spec_accum = spec_accum + hl_spec[None, None, :] * spec_w_hl[..., None]

  direct_rgb = jnp.clip(albedo * diff_accum + spec_accum + albedo * emit_coef[..., None], 0.0, 1.0)
  sky_t = jnp.clip(0.5 * (ray_dir[..., 2] + 1.0), 0.0, 1.0)[..., None]
  sky_rgb = (
      sky_t * jnp.asarray(spec.sky_rgb1, dtype=jnp.float32)[None, None, :]
      + (1.0 - sky_t) * jnp.asarray(spec.sky_rgb2, dtype=jnp.float32)[None, None, :]
  )
  rgb = jnp.where(hit[..., None], direct_rgb, sky_rgb)

  inv_cos = (1.0 / neg_z)[None, :]
  euclid_depth = jnp.where(hit, dist + float(spec.znear) * inv_cos, -1.0)
  z_depth = jnp.where(hit, float(spec.znear) + dist * neg_z[None, :], -1.0)

  return GBuffer(
      rgb=rgb.reshape(B, H, W, 3),
      albedo=jnp.where(hit[..., None], albedo, sky_rgb).reshape(B, H, W, 3),
      depth=euclid_depth.reshape(B, H, W),
      z_depth=z_depth.reshape(B, H, W),
      normal=jnp.where(hit[..., None], normal, 0.0).reshape(B, H, W, 3),
      view_dir=view_dir.reshape(B, H, W, 3),
      world_pos=jnp.where(hit[..., None], hit_pos, 0.0).reshape(B, H, W, 3),
      geom_id=geom_id.reshape(B, H, W),
      direct_shadow=jnp.where(hit, shadow_vis_accum, 1.0).reshape(B, H, W),
      direct_rgb=jnp.where(hit[..., None], direct_rgb, sky_rgb).reshape(B, H, W, 3),
  )


def _extract_light_batch(spec: Any, mjx_data_batch: Any) -> Tuple[jax.Array, jax.Array]:
  B = mjx_data_batch.cam_xpos.shape[0]
  if int(spec.nlight) > 0:
    l_bid = np.asarray(spec.light_bodyid, dtype=np.int32)
    l_pos0 = jnp.asarray(spec.light_pos, dtype=jnp.float32)
    l_dir0 = jnp.asarray(spec.light_dir, dtype=jnp.float32)
    xpos_l = mjx_data_batch.xpos[:, l_bid, :]    # (B, L, 3)
    xmat_l = mjx_data_batch.xmat[:, l_bid, :, :]  # (B, L, 3, 3)
    light_xpos = xpos_l + jnp.einsum("blij,lj->bli", xmat_l, l_pos0, precision=jax.lax.Precision.HIGHEST)
    light_xdir = jnp.einsum("blij,lj->bli", xmat_l, l_dir0, precision=jax.lax.Precision.HIGHEST)
    return light_xpos, light_xdir
  return jnp.zeros((B, 0, 3), dtype=jnp.float32), jnp.zeros((B, 0, 3), dtype=jnp.float32)


def raycast_batch_soa(
    spec: Any,
    mjx_data_batch: Any,
    geom_rgba_batch: Optional[jax.Array] = None,
):
  """Batched (B, R) SoA renderer eliminating per-world vmap scalar overhead."""
  cam_pos = mjx_data_batch.cam_xpos[:, int(spec.cam_id), :]
  cam_mat = mjx_data_batch.cam_xmat[:, int(spec.cam_id), :, :]
  geom_xpos = mjx_data_batch.geom_xpos
  geom_xmat = mjx_data_batch.geom_xmat
  light_xpos, light_xdir = _extract_light_batch(spec, mjx_data_batch)
  geom_size = jnp.asarray(spec.geom_size, dtype=jnp.float32)
  geom_rgba = jnp.asarray(spec.geom_rgba, dtype=jnp.float32) if geom_rgba_batch is None else geom_rgba_batch

  ray_orig, ray_dir, neg_z = _compute_batch_camera_rays_soa(spec, cam_pos, cam_mat)
  B, R, _ = ray_orig.shape
  max_dist = ((float(spec.zfar) - float(spec.znear)) / neg_z)[None, :] if spec.zfar > 0.0 else jnp.full((1, R), MJ_MAXVAL, dtype=jnp.float32)
  max_dist = jnp.broadcast_to(max_dist, (B, R))

  gids = np.asarray(spec.enabled_geom_ids, dtype=np.int32)
  rb_sq_all = _compute_geom_bound_sq(spec, gids)

  rox, roy, roz = ray_orig[..., 0], ray_orig[..., 1], ray_orig[..., 2]
  rdx, rdy, rdz = ray_dir[..., 0], ray_dir[..., 1], ray_dir[..., 2]

  best_d = max_dist
  best_gid = jnp.full((B, R), -1, dtype=jnp.int32)
  best_nx = jnp.zeros((B, R), dtype=jnp.float32)
  best_ny = jnp.zeros((B, R), dtype=jnp.float32)
  best_nz = jnp.zeros((B, R), dtype=jnp.float32)

  for k, gid_val in enumerate(gids):
    gt = int(spec.geom_type[gid_val])
    px = geom_xpos[:, gid_val, 0:1]
    py = geom_xpos[:, gid_val, 1:2]
    pz = geom_xpos[:, gid_val, 2:3]
    mat = geom_xmat[:, gid_val, :, :]
    sz = spec.geom_size[gid_val]
    cand_d, cnx, cny, cnz = _ray_one_geom_v2(
        gt,
        px, py, pz,
        mat[:, 0, 0:1], mat[:, 0, 1:2], mat[:, 0, 2:3],
        mat[:, 1, 0:1], mat[:, 1, 1:2], mat[:, 1, 2:3],
        mat[:, 2, 0:1], mat[:, 2, 1:2], mat[:, 2, 2:3],
        float(sz[0]), float(sz[1]), float(sz[2]),
        float(rb_sq_all[k]),
        rox, roy, roz, rdx, rdy, rdz,
        plane_finite_x=float(spec.geom_size[gid_val, 0]) > 0.0,
        plane_finite_y=float(spec.geom_size[gid_val, 1]) > 0.0,
        compute_normal=True,
    )
    imp = cand_d < best_d
    best_d = jnp.where(imp, cand_d, best_d)
    best_gid = jnp.where(imp, jnp.int32(gid_val), best_gid)
    best_nx = jnp.where(imp, cnx, best_nx)
    best_ny = jnp.where(imp, cny, best_ny)
    best_nz = jnp.where(imp, cnz, best_nz)

  hit = best_gid >= 0
  best_d = jnp.where(hit, best_d, -1.0)
  normal = jnp.where(hit[..., None], jnp.stack([best_nx, best_ny, best_nz], axis=-1), 0.0)

  return _shade_batch_soa(
      spec,
      geom_xpos,
      geom_xmat,
      geom_size,
      geom_rgba,
      light_xpos,
      light_xdir,
      ray_orig,
      ray_dir,
      neg_z,
      best_d,
      normal,
      best_gid,
  )


def pallas_raycast_batch_v2(
    spec: Any,
    mjx_data_batch: Any,
    geom_rgba_batch: Optional[jax.Array] = None,
    interpret: bool = True,
):
  """Native TPU (BLOCK_B=8, BLOCK_R=128) SoA Pallas kernel with fused raygen & in-VMEM shadow any_hit."""
  cam_pos = mjx_data_batch.cam_xpos[:, int(spec.cam_id), :]
  cam_mat = mjx_data_batch.cam_xmat[:, int(spec.cam_id), :, :]
  geom_xpos = mjx_data_batch.geom_xpos
  geom_xmat = mjx_data_batch.geom_xmat
  light_xpos, light_xdir = _extract_light_batch(spec, mjx_data_batch)
  geom_size = jnp.asarray(spec.geom_size, dtype=jnp.float32)
  geom_rgba = jnp.asarray(spec.geom_rgba, dtype=jnp.float32) if geom_rgba_batch is None else geom_rgba_batch

  B = cam_pos.shape[0]
  H, W = int(spec.height), int(spec.width)
  R = H * W

  gids = np.asarray(spec.enabled_geom_ids, dtype=np.int32)
  G = len(gids)
  if G == 0:
    ray_orig, ray_dir, neg_z = _compute_batch_camera_rays_soa(spec, cam_pos, cam_mat)
    return _shade_batch_soa(
        spec, geom_xpos, geom_xmat, geom_size, geom_rgba, light_xpos, light_xdir,
        ray_orig, ray_dir, neg_z,
        jnp.full((B, R), -1.0, jnp.float32),
        jnp.zeros((B, R, 3), jnp.float32),
        jnp.full((B, R), -1, jnp.int32),
    )

  B_pad = ((B + BLOCK_B - 1) // BLOCK_B) * BLOCK_B
  R_pad = ((R + BLOCK_R - 1) // BLOCK_R) * BLOCK_R
  G_pad = ((G + 127) // 128) * 128

  # Pack camera pose into (B_pad, 128) 2D block
  cam_pack = jnp.concatenate([cam_pos, cam_mat.reshape(B, 9)], axis=-1)  # (B, 12)
  cam_128 = jnp.pad(cam_pack, ((0, B_pad - B), (0, 128 - 12)))

  # Pack local camera ray templates into (8, R_pad) 2D block: rows 0..2=ld, 3..5=lo, 6=neg_z, 7=max_dist
  ld_np, lo_np, neg_z_np = build_local_camera_rays(spec)
  md_np = ((float(spec.zfar) - float(spec.znear)) / neg_z_np) if spec.zfar > 0.0 else np.full((R,), MJ_MAXVAL, dtype=np.float32)
  lray_raw = np.zeros((8, R_pad), dtype=np.float32)
  lray_raw[0:3, :R] = ld_np.T
  lray_raw[3:6, :R] = lo_np.T
  lray_raw[6, :R] = neg_z_np
  lray_raw[7, :R] = md_np
  lray_raw[2, R:] = -1.0
  lray_raw[6, R:] = 1.0
  lray_raw[7, R:] = 1.0
  lray_dev = jnp.asarray(lray_raw, dtype=jnp.float32)

  # Pack enabled geoms across B_pad into (B_pad * 16, G_pad) -> every bi loads (128, G_pad)
  gpos = geom_xpos[:, gids, :]               # (B, G, 3)
  gmat = geom_xmat[:, gids, :, :].reshape(B, G, 9)
  gsz_all = jnp.broadcast_to(geom_size[gids][None, :, :], (B, G, 3))
  rb_sq = jnp.broadcast_to(jnp.asarray(_compute_geom_bound_sq(spec, gids))[None, :, None], (B, G, 1))
  g_16 = jnp.concatenate([gpos, gmat, gsz_all, rb_sq], axis=-1)  # (B, G, 16)
  g_16 = jnp.pad(g_16, ((0, B_pad - B), (0, G_pad - G), (0, 0)))
  # Reshape as (n_b_blocks, 8, G_pad, 16) -> transpose to (n_b_blocks, 16, 8, G_pad) -> (B_pad*16, G_pad)
  n_b_blk = B_pad // BLOCK_B
  n_r_blk = R_pad // BLOCK_R
  geom_fields = (
      g_16.reshape(n_b_blk, BLOCK_B, G_pad, 16)
      .transpose(0, 3, 1, 2)
      .reshape(n_b_blk * 16 * BLOCK_B, G_pad)
  )

  # Identify primary shadow light (first active castshadow light) for in-VMEM any_hit
  shadow_l_idx = -1
  for li in range(int(spec.nlight)):
    if spec.use_shadows and bool(spec.light_active[li]) and bool(spec.light_castshadow[li]):
      shadow_l_idx = li
      break

  light_128 = jnp.zeros((B_pad, 128), dtype=jnp.float32)
  if shadow_l_idx >= 0:
    is_dir = 1.0 if int(spec.light_type[shadow_l_idx]) == 1 else 0.0
    lp = light_xpos[:, shadow_l_idx, :]
    ld_vec = light_xdir[:, shadow_l_idx, :]
    ld_vec = ld_vec / jnp.maximum(jnp.linalg.norm(ld_vec, axis=-1, keepdims=True), 1e-12)
    l_row = jnp.concatenate([jnp.full((B, 1), is_dir, jnp.float32), lp, ld_vec], axis=-1)
    light_128 = jnp.pad(l_row, ((0, B_pad - B), (0, 128 - 7)))

  g_types_static = tuple(int(spec.geom_type[g]) for g in gids)
  gids_static = tuple(int(g) for g in gids)
  plane_fx = tuple(float(spec.geom_size[g, 0]) > 0.0 for g in gids)
  plane_fy = tuple(float(spec.geom_size[g, 1]) > 0.0 for g in gids)
  znear_f = float(spec.znear)
  do_in_vmem_shadow = shadow_l_idx >= 0

  def _v2_kernel(cam_ref, lray_ref, gf_ref, light_ref, out_d_ref, out_nx_ref, out_ny_ref, out_nz_ref, out_gid_ref, out_sh_ref):
    cam = cam_ref[...]       # (8, 128)
    lr = lray_ref[...]       # (8, 128)
    gf = gf_ref[...]         # (128, G_pad)
    linfo = light_ref[...]   # (8, 128)

    # In-VMEM fused camera raygen: (8, 12) x (3, 128) -> (8, 128)
    ldx, ldy, ldz = lr[0:1, :], lr[1:2, :], lr[2:3, :]
    lox, loy, loz = lr[3:4, :], lr[4:5, :], lr[5:6, :]
    inv_cos = znear_f / lr[6:7, :]
    max_d = jnp.broadcast_to(lr[7:8, :], (BLOCK_B, BLOCK_R))

    cpx, cpy, cpz = cam[:, 0:1], cam[:, 1:2], cam[:, 2:3]
    c00, c01, c02 = cam[:, 3:4], cam[:, 4:5], cam[:, 5:6]
    c10, c11, c12 = cam[:, 6:7], cam[:, 7:8], cam[:, 8:9]
    c20, c21, c22 = cam[:, 9:10], cam[:, 10:11], cam[:, 11:12]

    rdx_raw = c00 * ldx + c01 * ldy + c02 * ldz
    rdy_raw = c10 * ldx + c11 * ldy + c12 * ldz
    rdz_raw = c20 * ldx + c21 * ldy + c22 * ldz
    inv_rn = 1.0 / jnp.maximum(jnp.sqrt(rdx_raw * rdx_raw + rdy_raw * rdy_raw + rdz_raw * rdz_raw), 1e-12)
    rdx, rdy, rdz = rdx_raw * inv_rn, rdy_raw * inv_rn, rdz_raw * inv_rn

    rox = cpx + (c00 * lox + c01 * loy + c02 * loz) + inv_cos * rdx
    roy = cpy + (c10 * lox + c11 * loy + c12 * loz) + inv_cos * rdy
    roz = cpz + (c20 * lox + c21 * loy + c22 * loz) + inv_cos * rdz

    best_d = max_d
    best_nx = jnp.zeros((BLOCK_B, BLOCK_R), dtype=jnp.float32)
    best_ny = jnp.zeros((BLOCK_B, BLOCK_R), dtype=jnp.float32)
    best_nz = jnp.zeros((BLOCK_B, BLOCK_R), dtype=jnp.float32)
    best_gid = jnp.full((BLOCK_B, BLOCK_R), -1, dtype=jnp.int32)

    for k, (gt, gid_val, pfx, pfy) in enumerate(zip(g_types_static, gids_static, plane_fx, plane_fy)):
      # Slice (8, 1) from (128, G_pad) where row block f occupies [f*8 : (f+1)*8]
      px = gf[0 * 8 : 1 * 8, k : k + 1]
      py = gf[1 * 8 : 2 * 8, k : k + 1]
      pz = gf[2 * 8 : 3 * 8, k : k + 1]
      m00, m01, m02 = gf[3 * 8 : 4 * 8, k : k + 1], gf[4 * 8 : 5 * 8, k : k + 1], gf[5 * 8 : 6 * 8, k : k + 1]
      m10, m11, m12 = gf[6 * 8 : 7 * 8, k : k + 1], gf[7 * 8 : 8 * 8, k : k + 1], gf[8 * 8 : 9 * 8, k : k + 1]
      m20, m21, m22 = gf[9 * 8 : 10 * 8, k : k + 1], gf[10 * 8 : 11 * 8, k : k + 1], gf[11 * 8 : 12 * 8, k : k + 1]
      sx, sy, sz = gf[12 * 8 : 13 * 8, k : k + 1], gf[13 * 8 : 14 * 8, k : k + 1], gf[14 * 8 : 15 * 8, k : k + 1]
      rb_sq = gf[15 * 8 : 16 * 8, k : k + 1]

      cd, cnx, cny, cnz = _ray_one_geom_v2(
          gt,
          px, py, pz,
          m00, m01, m02,
          m10, m11, m12,
          m20, m21, m22,
          sx, sy, sz,
          rb_sq,
          rox, roy, roz, rdx, rdy, rdz,
          plane_finite_x=pfx,
          plane_finite_y=pfy,
          compute_normal=True,
      )
      imp = cd < best_d
      best_d = jnp.where(imp, cd, best_d)
      best_gid = jnp.where(imp, jnp.int32(gid_val), best_gid)
      best_nx = jnp.where(imp, cnx, best_nx)
      best_ny = jnp.where(imp, cny, best_ny)
      best_nz = jnp.where(imp, cnz, best_nz)

    hit = best_gid >= 0
    best_d_out = jnp.where(hit, best_d, -1.0)
    out_nx_v = jnp.where(hit, best_nx, 0.0)
    out_ny_v = jnp.where(hit, best_ny, 0.0)
    out_nz_v = jnp.where(hit, best_nz, 0.0)

    # In-VMEM fused shadow any_hit (primary hit never touches HBM before shooting shadow!)
    sh_occ = jnp.zeros((BLOCK_B, BLOCK_R), dtype=jnp.float32)
    if do_in_vmem_shadow:
      d_safe = jnp.where(hit, best_d, 0.0)
      hx = rox + d_safe * rdx + out_nx_v * 1.0e-4
      hy = roy + d_safe * rdy + out_ny_v * 1.0e-4
      hz = roz + d_safe * rdz + out_nz_v * 1.0e-4

      is_dir = linfo[:, 0:1] > 0.5
      dlx = linfo[:, 1:2] - hx
      dly = linfo[:, 2:3] - hy
      dlz = linfo[:, 3:4] - hz
      d_pt = jnp.sqrt(jnp.maximum(dlx * dlx + dly * dly + dlz * dlz, 1e-12))
      lx_v = jnp.where(is_dir, -linfo[:, 4:5], dlx / d_pt)
      ly_v = jnp.where(is_dir, -linfo[:, 5:6], dly / d_pt)
      lz_v = jnp.where(is_dir, -linfo[:, 6:7], dlz / d_pt)
      ndotl = out_nx_v * lx_v + out_ny_v * ly_v + out_nz_v * lz_v
      dist_lim = jnp.where(is_dir, 1.0e8, jnp.maximum(d_pt - 1.0e-3, 0.0))
      sh_max = jnp.where(hit & (ndotl > 0.0), dist_lim, 0.0)

      occ_bool = jnp.zeros((BLOCK_B, BLOCK_R), dtype=jnp.bool_)
      for k, (gt, pfx, pfy) in enumerate(zip(g_types_static, plane_fx, plane_fy)):
        px = gf[0 * 8 : 1 * 8, k : k + 1]
        py = gf[1 * 8 : 2 * 8, k : k + 1]
        pz = gf[2 * 8 : 3 * 8, k : k + 1]
        m00, m01, m02 = gf[3 * 8 : 4 * 8, k : k + 1], gf[4 * 8 : 5 * 8, k : k + 1], gf[5 * 8 : 6 * 8, k : k + 1]
        m10, m11, m12 = gf[6 * 8 : 7 * 8, k : k + 1], gf[7 * 8 : 8 * 8, k : k + 1], gf[8 * 8 : 9 * 8, k : k + 1]
        m20, m21, m22 = gf[9 * 8 : 10 * 8, k : k + 1], gf[10 * 8 : 11 * 8, k : k + 1], gf[11 * 8 : 12 * 8, k : k + 1]
        sx, sy, sz = gf[12 * 8 : 13 * 8, k : k + 1], gf[13 * 8 : 14 * 8, k : k + 1], gf[14 * 8 : 15 * 8, k : k + 1]
        rb_sq = gf[15 * 8 : 16 * 8, k : k + 1]
        scd, _, _, _ = _ray_one_geom_v2(
            gt,
            px, py, pz,
            m00, m01, m02,
            m10, m11, m12,
            m20, m21, m22,
            sx, sy, sz,
            rb_sq,
            hx, hy, hz, lx_v, ly_v, lz_v,
            plane_finite_x=pfx,
            plane_finite_y=pfy,
            compute_normal=False,
        )
        occ_bool = occ_bool | (scd < sh_max)
      sh_occ = occ_bool.astype(jnp.float32)

    out_d_ref[...] = best_d_out
    out_nx_ref[...] = out_nx_v
    out_ny_ref[...] = out_ny_v
    out_nz_ref[...] = out_nz_v
    out_gid_ref[...] = best_gid
    out_sh_ref[...] = sh_occ

  d_out, nx_out, ny_out, nz_out, gid_out, sh_out = pl.pallas_call(
      _v2_kernel,
      out_shape=[
          jax.ShapeDtypeStruct((B_pad, R_pad), jnp.float32),
          jax.ShapeDtypeStruct((B_pad, R_pad), jnp.float32),
          jax.ShapeDtypeStruct((B_pad, R_pad), jnp.float32),
          jax.ShapeDtypeStruct((B_pad, R_pad), jnp.float32),
          jax.ShapeDtypeStruct((B_pad, R_pad), jnp.int32),
          jax.ShapeDtypeStruct((B_pad, R_pad), jnp.float32),
      ],
      grid=(n_b_blk, n_r_blk),
      in_specs=[
          pl.BlockSpec((BLOCK_B, 128), lambda bi, ri: (bi, 0)),
          pl.BlockSpec((8, BLOCK_R), lambda bi, ri: (0, ri)),
          pl.BlockSpec((16 * BLOCK_B, G_pad), lambda bi, ri: (bi, 0)),
          pl.BlockSpec((BLOCK_B, 128), lambda bi, ri: (bi, 0)),
      ],
      out_specs=[
          pl.BlockSpec((BLOCK_B, BLOCK_R), lambda bi, ri: (bi, ri)),
          pl.BlockSpec((BLOCK_B, BLOCK_R), lambda bi, ri: (bi, ri)),
          pl.BlockSpec((BLOCK_B, BLOCK_R), lambda bi, ri: (bi, ri)),
          pl.BlockSpec((BLOCK_B, BLOCK_R), lambda bi, ri: (bi, ri)),
          pl.BlockSpec((BLOCK_B, BLOCK_R), lambda bi, ri: (bi, ri)),
          pl.BlockSpec((BLOCK_B, BLOCK_R), lambda bi, ri: (bi, ri)),
      ],
      compiler_params=None if interpret else pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
      interpret=interpret,
  )(cam_128, lray_dev, geom_fields, light_128)

  dist = d_out[:B, :R]
  normal = jnp.stack([nx_out[:B, :R], ny_out[:B, :R], nz_out[:B, :R]], axis=-1)
  geom_id = gid_out[:B, :R]
  fused_occ = (sh_out[:B, :R] > 0.5) if do_in_vmem_shadow else None

  ray_orig, ray_dir, neg_z = _compute_batch_camera_rays_soa(spec, cam_pos, cam_mat)
  return _shade_batch_soa(
      spec,
      geom_xpos,
      geom_xmat,
      geom_size,
      geom_rgba,
      light_xpos,
      light_xdir,
      ray_orig,
      ray_dir,
      neg_z,
      dist,
      normal,
      geom_id,
      fused_shadow0_occluded=fused_occ,
  )
