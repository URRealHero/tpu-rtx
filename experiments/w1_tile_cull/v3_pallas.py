"""V3 Parameterized Pallas TPU Raytracer for Tile Geometry, Grid Order & Culling Ablations.

Never modifies core tpu_mjx_render/*.py files; lives exclusively in experiments/w1_tile_cull/.
"""

from __future__ import annotations

from typing import Any, Optional

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
import numpy as np

from tpu_mjx_render.pallas_v2 import (
    _compute_batch_camera_rays_soa,
    _compute_geom_bound_sq,
    _extract_light_batch,
    _shade_batch_soa,
    build_local_camera_rays,
)
from tpu_mjx_render.primitives import (
    GEOM_BOX,
    GEOM_CAPSULE,
    GEOM_CYLINDER,
    GEOM_ELLIPSOID,
    GEOM_PLANE,
    GEOM_SPHERE,
    MJ_MAXVAL,
    MJ_MINVAL,
)


def _inv_sqrt(x, use_rsqrt: bool):
  if use_rsqrt:
    return jax.lax.rsqrt(jnp.maximum(x, 1e-24))
  return 1.0 / jnp.maximum(jnp.sqrt(x), 1e-12)


def _ray_one_geom_from_delta(
    gt: int,
    dx, dy, dz,
    sphere_hit,
    m00, m01, m02,
    m10, m11, m12,
    m20, m21, m22,
    sx, sy, sz,
    rdx, rdy, rdz,
    plane_finite_x: bool = True,
    plane_finite_y: bool = True,
    compute_normal: bool = True,
    use_rsqrt: bool = False,
):
  """Branchless SoA primitive intersection from precomputed ray-origin delta and sphere_hit."""
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
    inv_n = _inv_sqrt(nx_w * nx_w + ny_w * ny_w + nz_w * nz_w, use_rsqrt)
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
    inv_n = _inv_sqrt(lnx * lnx + lny * lny + lnz * lnz, use_rsqrt)
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
    inv_n = _inv_sqrt(lnx * lnx + lny * lny + lnz * lnz, use_rsqrt)
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
    inv_n = _inv_sqrt(hlx * hlx + hly * hly, use_rsqrt)
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


def pallas_raycast_batch_v3(
    spec: Any,
    mjx_data_batch: Any,
    geom_rgba_batch: Optional[jax.Array] = None,
    interpret: bool = False,
    block_b: int = 8,
    block_r: int = 512,
    grid_order: str = "b_r",
    cull_mode: str = "none",
    geom_pack: str = "v2_128",
    fast_math: str = "none",
):
  """Parameterized V3 Pallas raycaster supporting tile sweeps, grid order, lax.cond culling, and geom packing."""
  assert block_b % 8 == 0, f"block_b must be a multiple of 8, got {block_b}"
  assert block_r % 128 == 0, f"block_r must be a multiple of 128, got {block_r}"
  assert grid_order in ("b_r", "r_b")
  assert cull_mode in ("none", "sphere_primary", "sphere_both", "frustum_tight")
  assert geom_pack in ("v2_128", "compact_row")
  assert fast_math in ("none", "rsqrt_no_raynorm")

  use_rsqrt = (fast_math == "rsqrt_no_raynorm")
  skip_ray_norm = (fast_math == "rsqrt_no_raynorm")

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

  B_pad = ((B + block_b - 1) // block_b) * block_b
  R_pad = ((R + block_r - 1) // block_r) * block_r
  n_b_blk = B_pad // block_b
  n_r_blk = R_pad // block_r

  cam_pack = jnp.concatenate([cam_pos, cam_mat.reshape(B, 9)], axis=-1)
  cam_128 = jnp.pad(cam_pack, ((0, B_pad - B), (0, 128 - 12)))

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

  gpos = geom_xpos[:, gids, :]
  gmat = geom_xmat[:, gids, :, :].reshape(B, G, 9)
  gsz_all = jnp.broadcast_to(geom_size[gids][None, :, :], (B, G, 3))
  rb_sq = jnp.broadcast_to(jnp.asarray(_compute_geom_bound_sq(spec, gids))[None, :, None], (B, G, 1))
  g_16 = jnp.concatenate([gpos, gmat, gsz_all, rb_sq], axis=-1)  # (B, G, 16)

  if geom_pack == "v2_128":
    G_pad = ((G + 127) // 128) * 128
    g_16_pad = jnp.pad(g_16, ((0, B_pad - B), (0, G_pad - G), (0, 0)))
    geom_fields = (
        g_16_pad.reshape(n_b_blk, block_b, G_pad, 16)
        .transpose(0, 3, 1, 2)
        .reshape(n_b_blk * 16 * block_b, G_pad)
    )
    gf_block_shape = (16 * block_b, G_pad)
    G_stride = G_pad
  else:  # compact_row: (B_pad, ceil128(16 * G_pad8))
    G_pad8 = ((G + 7) // 8) * 8
    col_pad = ((16 * G_pad8 + 127) // 128) * 128
    g_16_pad = jnp.pad(g_16, ((0, B_pad - B), (0, G_pad8 - G), (0, 0)))  # (B_pad, G_pad8, 16)
    g_flat = g_16_pad.transpose(0, 2, 1).reshape(B_pad, 16 * G_pad8)
    geom_fields = jnp.pad(g_flat, ((0, 0), (0, col_pad - 16 * G_pad8)))
    gf_block_shape = (block_b, col_pad)
    G_stride = G_pad8

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

  def _extract_geom_fields(gf, k):
    if geom_pack == "v2_128":
      return (
          gf[0 * block_b : 1 * block_b, k : k + 1],
          gf[1 * block_b : 2 * block_b, k : k + 1],
          gf[2 * block_b : 3 * block_b, k : k + 1],
          gf[3 * block_b : 4 * block_b, k : k + 1],
          gf[4 * block_b : 5 * block_b, k : k + 1],
          gf[5 * block_b : 6 * block_b, k : k + 1],
          gf[6 * block_b : 7 * block_b, k : k + 1],
          gf[7 * block_b : 8 * block_b, k : k + 1],
          gf[8 * block_b : 9 * block_b, k : k + 1],
          gf[9 * block_b : 10 * block_b, k : k + 1],
          gf[10 * block_b : 11 * block_b, k : k + 1],
          gf[11 * block_b : 12 * block_b, k : k + 1],
          gf[12 * block_b : 13 * block_b, k : k + 1],
          gf[13 * block_b : 14 * block_b, k : k + 1],
          gf[14 * block_b : 15 * block_b, k : k + 1],
          gf[15 * block_b : 16 * block_b, k : k + 1],
      )
    return tuple(gf[:, f * G_stride + k : f * G_stride + k + 1] for f in range(16))

  def _v3_kernel(cam_ref, lray_ref, gf_ref, light_ref, out_d_ref, out_nx_ref, out_ny_ref, out_nz_ref, out_gid_ref, out_sh_ref):
    cam = cam_ref[...]
    lr = lray_ref[...]
    gf = gf_ref[...]
    linfo = light_ref[...]

    ldx, ldy, ldz = lr[0:1, :], lr[1:2, :], lr[2:3, :]
    lox, loy, loz = lr[3:4, :], lr[4:5, :], lr[5:6, :]
    inv_cos = znear_f / lr[6:7, :]
    max_d = jnp.broadcast_to(lr[7:8, :], (block_b, block_r))

    cpx, cpy, cpz = cam[:, 0:1], cam[:, 1:2], cam[:, 2:3]
    c00, c01, c02 = cam[:, 3:4], cam[:, 4:5], cam[:, 5:6]
    c10, c11, c12 = cam[:, 6:7], cam[:, 7:8], cam[:, 8:9]
    c20, c21, c22 = cam[:, 9:10], cam[:, 10:11], cam[:, 11:12]

    rdx_raw = c00 * ldx + c01 * ldy + c02 * ldz
    rdy_raw = c10 * ldx + c11 * ldy + c12 * ldz
    rdz_raw = c20 * ldx + c21 * ldy + c22 * ldz
    if skip_ray_norm:
      rdx, rdy, rdz = rdx_raw, rdy_raw, rdz_raw
    else:
      inv_rn = _inv_sqrt(rdx_raw * rdx_raw + rdy_raw * rdy_raw + rdz_raw * rdz_raw, use_rsqrt)
      rdx, rdy, rdz = rdx_raw * inv_rn, rdy_raw * inv_rn, rdz_raw * inv_rn

    rox = cpx + (c00 * lox + c01 * loy + c02 * loz) + inv_cos * rdx
    roy = cpy + (c10 * lox + c11 * loy + c12 * loz) + inv_cos * rdy
    roz = cpz + (c20 * lox + c21 * loy + c22 * loz) + inv_cos * rdz

    best_d = max_d
    best_nx = jnp.zeros((block_b, block_r), dtype=jnp.float32)
    best_ny = jnp.zeros((block_b, block_r), dtype=jnp.float32)
    best_nz = jnp.zeros((block_b, block_r), dtype=jnp.float32)
    best_gid = jnp.full((block_b, block_r), -1, dtype=jnp.int32)

    cull_primary = cull_mode in ("sphere_primary", "sphere_both", "frustum_tight")
    cull_shadow = cull_mode in ("sphere_both", "frustum_tight")
    tight_front = (cull_mode == "frustum_tight")

    for k, (gt, gid_val, pfx, pfy) in enumerate(zip(g_types_static, gids_static, plane_fx, plane_fy)):
      (px, py, pz, m00, m01, m02, m10, m11, m12, m20, m21, m22, sx, sy, sz, rb_sq) = _extract_geom_fields(gf, k)
      dx = rox - px
      dy = roy - py
      dz = roz - pz

      if gt != GEOM_PLANE:
        proj = dx * rdx + dy * rdy + dz * rdz
        d2 = dx * dx + dy * dy + dz * dz
        d2_perp = d2 - proj * proj
        sphere_hit = d2_perp <= rb_sq
        if tight_front:
          sphere_hit = sphere_hit & ((proj < 0.0) | (d2 <= rb_sq))
      else:
        sphere_hit = True

      if cull_primary and (gt != GEOM_PLANE):
        any_hit = jnp.any(sphere_hit)

        def _do_primary(carry):
          b_d, b_gid, b_nx, b_ny, b_nz = carry
          cd, cnx, cny, cnz = _ray_one_geom_from_delta(
              gt, dx, dy, dz, sphere_hit,
              m00, m01, m02, m10, m11, m12, m20, m21, m22,
              sx, sy, sz, rdx, rdy, rdz,
              plane_finite_x=pfx, plane_finite_y=pfy, compute_normal=True, use_rsqrt=use_rsqrt,
          )
          imp = cd < b_d
          return (
              jnp.where(imp, cd, b_d),
              jnp.where(imp, jnp.int32(gid_val), b_gid),
              jnp.where(imp, cnx, b_nx),
              jnp.where(imp, cny, b_ny),
              jnp.where(imp, cnz, b_nz),
          )

        best_d, best_gid, best_nx, best_ny, best_nz = jax.lax.cond(
            any_hit,
            _do_primary,
            lambda carry: carry,
            (best_d, best_gid, best_nx, best_ny, best_nz),
        )
      else:
        cd, cnx, cny, cnz = _ray_one_geom_from_delta(
            gt, dx, dy, dz, sphere_hit,
            m00, m01, m02, m10, m11, m12, m20, m21, m22,
            sx, sy, sz, rdx, rdy, rdz,
            plane_finite_x=pfx, plane_finite_y=pfy, compute_normal=True, use_rsqrt=use_rsqrt,
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

    sh_occ = jnp.zeros((block_b, block_r), dtype=jnp.float32)
    if do_in_vmem_shadow:
      d_safe = jnp.where(hit, best_d, 0.0)
      hx = rox + d_safe * rdx + out_nx_v * 1.0e-4
      hy = roy + d_safe * rdy + out_ny_v * 1.0e-4
      hz = roz + d_safe * rdz + out_nz_v * 1.0e-4

      is_dir = linfo[:, 0:1] > 0.5
      dlx = linfo[:, 1:2] - hx
      dly = linfo[:, 2:3] - hy
      dlz = linfo[:, 3:4] - hz
      d2_pt = jnp.maximum(dlx * dlx + dly * dly + dlz * dlz, 1e-12)
      d_pt = jnp.sqrt(d2_pt)
      inv_dpt = 1.0 / d_pt
      lx_v = jnp.where(is_dir, -linfo[:, 4:5], dlx * inv_dpt)
      ly_v = jnp.where(is_dir, -linfo[:, 5:6], dly * inv_dpt)
      lz_v = jnp.where(is_dir, -linfo[:, 6:7], dlz * inv_dpt)
      ndotl = out_nx_v * lx_v + out_ny_v * ly_v + out_nz_v * lz_v
      dist_lim = jnp.where(is_dir, 1.0e8, jnp.maximum(d_pt - 1.0e-3, 0.0))
      sh_max = jnp.where(hit & (ndotl > 0.0), dist_lim, 0.0)

      def _run_shadow_loop():
        occ_i32 = jnp.zeros((block_b, block_r), dtype=jnp.int32)
        for k, (gt, pfx, pfy) in enumerate(zip(g_types_static, plane_fx, plane_fy)):
          (px, py, pz, m00, m01, m02, m10, m11, m12, m20, m21, m22, sx, sy, sz, rb_sq) = _extract_geom_fields(gf, k)
          dx_s = hx - px
          dy_s = hy - py
          dz_s = hz - pz
          if gt != GEOM_PLANE:
            proj_s = dx_s * lx_v + dy_s * ly_v + dz_s * lz_v
            d2_s = dx_s * dx_s + dy_s * dy_s + dz_s * dz_s
            d2_perp_s = d2_s - proj_s * proj_s
            sh_sphere_hit = d2_perp_s <= rb_sq
            if tight_front:
              sh_sphere_hit = sh_sphere_hit & ((proj_s < 0.0) | (d2_s <= rb_sq)) & (sh_max > 0.0)
          else:
            sh_sphere_hit = True

          if cull_shadow and (gt != GEOM_PLANE):
            sh_any = jnp.any(sh_sphere_hit & (sh_max > 0.0))

            def _do_sh(occ_in):
              scd, _, _, _ = _ray_one_geom_from_delta(
                  gt, dx_s, dy_s, dz_s, sh_sphere_hit,
                  m00, m01, m02, m10, m11, m12, m20, m21, m22,
                  sx, sy, sz, lx_v, ly_v, lz_v,
                  plane_finite_x=pfx, plane_finite_y=pfy, compute_normal=False, use_rsqrt=use_rsqrt,
              )
              return occ_in | (scd < sh_max).astype(jnp.int32)

            occ_i32 = jax.lax.cond(sh_any, _do_sh, lambda o: o, occ_i32)
          else:
            scd, _, _, _ = _ray_one_geom_from_delta(
                gt, dx_s, dy_s, dz_s, sh_sphere_hit,
                m00, m01, m02, m10, m11, m12, m20, m21, m22,
                sx, sy, sz, lx_v, ly_v, lz_v,
                plane_finite_x=pfx, plane_finite_y=pfy, compute_normal=False, use_rsqrt=use_rsqrt,
            )
            occ_i32 = occ_i32 | (scd < sh_max).astype(jnp.int32)
        return occ_i32.astype(jnp.float32)

      if tight_front:
        sh_occ = jax.lax.cond(
            jnp.any(sh_max > 0.0),
            _run_shadow_loop,
            lambda: jnp.zeros((block_b, block_r), dtype=jnp.float32),
        )
      else:
        sh_occ = _run_shadow_loop()

    out_d_ref[...] = best_d_out
    out_nx_ref[...] = out_nx_v
    out_ny_ref[...] = out_ny_v
    out_nz_ref[...] = out_nz_v
    out_gid_ref[...] = best_gid
    out_sh_ref[...] = sh_occ

  if grid_order == "b_r":
    grid = (n_b_blk, n_r_blk)
    idx_b0 = lambda bi, ri: (bi, 0)
    idx_0r = lambda bi, ri: (0, ri)
    idx_br = lambda bi, ri: (bi, ri)
  else:
    grid = (n_r_blk, n_b_blk)
    idx_b0 = lambda ri, bi: (bi, 0)
    idx_0r = lambda ri, bi: (0, ri)
    idx_br = lambda ri, bi: (bi, ri)

  d_out, nx_out, ny_out, nz_out, gid_out, sh_out = pl.pallas_call(
      _v3_kernel,
      out_shape=[
          jax.ShapeDtypeStruct((B_pad, R_pad), jnp.float32),
          jax.ShapeDtypeStruct((B_pad, R_pad), jnp.float32),
          jax.ShapeDtypeStruct((B_pad, R_pad), jnp.float32),
          jax.ShapeDtypeStruct((B_pad, R_pad), jnp.float32),
          jax.ShapeDtypeStruct((B_pad, R_pad), jnp.int32),
          jax.ShapeDtypeStruct((B_pad, R_pad), jnp.float32),
      ],
      grid=grid,
      in_specs=[
          pl.BlockSpec((block_b, 128), idx_b0),
          pl.BlockSpec((8, block_r), idx_0r),
          pl.BlockSpec(gf_block_shape, idx_b0),
          pl.BlockSpec((block_b, 128), idx_b0),
      ],
      out_specs=[
          pl.BlockSpec((block_b, block_r), idx_br),
          pl.BlockSpec((block_b, block_r), idx_br),
          pl.BlockSpec((block_b, block_r), idx_br),
          pl.BlockSpec((block_b, block_r), idx_br),
          pl.BlockSpec((block_b, block_r), idx_br),
          pl.BlockSpec((block_b, block_r), idx_br),
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
