"""Genuine JAX Pallas TPU (Mosaic) VMEM-resident raycasting and MXU shader kernels."""

from __future__ import annotations

from typing import Tuple

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
import numpy as np

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

BLOCK_P = 128  # Matches TPU VPU 128-lane vector width & v5p 128x128 MXU tile


def _pad_to_block(arr: np.ndarray | jax.Array, block: int = BLOCK_P, axis: int = 0) -> jax.Array:
  n = arr.shape[axis]
  rem = n % block
  if rem == 0:
    return jnp.asarray(arr)
  pad_w = [(0, 0)] * arr.ndim
  pad_w[axis] = (0, block - rem)
  return jnp.pad(arr, pad_w)


def pallas_raycast_scene(
    spec,
    geom_xpos: jax.Array,
    geom_xmat: jax.Array,
    geom_size: jax.Array,
    ray_orig: jax.Array,
    ray_dir: jax.Array,
    max_dist: jax.Array,
    interpret: bool = True,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
  """Pallas TPU kernel keeping [BLOCK_P=128, 3] ray tiles & all scene primitives in VMEM.

  Eliminates HBM materialization of [R, N_geoms] intersection buffers.
  Falls back cleanly via `interpret=True` on CPU or `interpret=False` on Mosaic TPU.
  """
  R = ray_orig.shape[0]
  orig_pad = _pad_to_block(ray_orig, BLOCK_P, axis=0)
  dir_pad = _pad_to_block(ray_dir, BLOCK_P, axis=0)
  md_pad = _pad_to_block(max_dist, BLOCK_P, axis=0)
  R_pad = orig_pad.shape[0]
  n_tiles = R_pad // BLOCK_P

  # Pack enabled primitives into a compact VMEM-resident static buffer [G_pad, 16]
  # Layout per geom: [pos(3), mat(9), size(3), type_f(1)] -> 16 floats
  gids = np.asarray(spec.enabled_geom_ids, dtype=np.int32)
  G = len(gids)
  if G == 0:
    return jnp.full((R,), -1.0, jnp.float32), jnp.zeros((R, 3), jnp.float32), jnp.full((R,), -1, jnp.int32)

  g_pos = geom_xpos[gids]                  # (G, 3)
  g_mat = geom_xmat[gids].reshape(G, 9)    # (G, 9)
  g_sz = geom_size[gids]                   # (G, 3)
  g_tp = jnp.asarray(spec.geom_type[gids], dtype=jnp.float32)[:, None]
  packed_geoms = jnp.concatenate([g_pos, g_mat, g_sz, g_tp], axis=-1)  # (G, 16)
  # Pad geom count to multiple of 8 for TPU sublane alignment (8x128)
  g_pad = ((G + 7) // 8) * 8
  if g_pad > G:
    packed_geoms = jnp.pad(packed_geoms, ((0, g_pad - G), (0, 0)))

  g_types_static = tuple(int(spec.geom_type[g]) for g in gids)
  gids_static = tuple(int(g) for g in gids)

  def _kernel(ro_ref, rd_ref, md_ref, geoms_ref, out_d_ref, out_n_ref, out_g_ref):
    ro = ro_ref[...]  # (BLOCK_P, 3) in VMEM
    rd = rd_ref[...]  # (BLOCK_P, 3) in VMEM
    md = md_ref[...]  # (BLOCK_P, 1) in VMEM
    geoms = geoms_ref[...]  # (g_pad, 16) in VMEM

    best_d = md[:, 0]
    best_nx = jnp.zeros((BLOCK_P,), dtype=jnp.float32)
    best_ny = jnp.zeros((BLOCK_P,), dtype=jnp.float32)
    best_nz = jnp.zeros((BLOCK_P,), dtype=jnp.float32)
    best_gid = jnp.full((BLOCK_P,), -1, dtype=jnp.int32)

    # Static-unrolled primitive loop in VMEM (zero HBM spills, zero dynamic branch)
    for local_i, (gt, gid_val) in enumerate(zip(g_types_static, gids_static)):
      row = geoms[local_i]
      px, py, pz = row[0], row[1], row[2]
      m00, m01, m02 = row[3], row[4], row[5]
      m10, m11, m12 = row[6], row[7], row[8]
      m20, m21, m22 = row[9], row[10], row[11]
      sx, sy, sz = row[12], row[13], row[14]

      dx = ro[:, 0] - px
      dy = ro[:, 1] - py
      dz = ro[:, 2] - pz

      # mat^T @ (ro - pos) and mat^T @ rd
      lx = m00 * dx + m10 * dy + m20 * dz
      ly = m01 * dx + m11 * dy + m21 * dz
      lz = m02 * dx + m12 * dy + m22 * dz

      vx = m00 * rd[:, 0] + m10 * rd[:, 1] + m20 * rd[:, 2]
      vy = m01 * rd[:, 0] + m11 * rd[:, 1] + m21 * rd[:, 2]
      vz = m02 * rd[:, 0] + m12 * rd[:, 1] + m22 * rd[:, 2]

      if gt == GEOM_PLANE:
        front = vz <= -MJ_MINVAL
        den = jnp.where(front, vz, -1.0)
        t = -lz / den
        hx = lx + t * vx
        hy = ly + t * vy
        ok = front & (t >= 0.0)
        if float(spec.geom_size[gid_val, 0]) > 0.0:
          ok = ok & (jnp.abs(hx) <= sx)
        if float(spec.geom_size[gid_val, 1]) > 0.0:
          ok = ok & (jnp.abs(hy) <= sy)
        cand_d = jnp.where(ok, t, MJ_MAXVAL)
        cand_nx = jnp.full((BLOCK_P,), m02, dtype=jnp.float32)
        cand_ny = jnp.full((BLOCK_P,), m12, dtype=jnp.float32)
        cand_nz = jnp.full((BLOCK_P,), m22, dtype=jnp.float32)

      elif gt == GEOM_SPHERE:
        a = rd[:, 0] * rd[:, 0] + rd[:, 1] * rd[:, 1] + rd[:, 2] * rd[:, 2]
        b = rd[:, 0] * dx + rd[:, 1] * dy + rd[:, 2] * dz
        c = dx * dx + dy * dy + dz * dz - sx * sx
        det = b * b - a * c
        valid = (det >= MJ_MINVAL) & (jnp.abs(a) > MJ_MINVAL)
        sd = jnp.sqrt(jnp.maximum(det, 0.0))
        inv_a = jnp.where(valid, 1.0 / jnp.where(valid, a, 1.0), 0.0)
        t0 = (-b - sd) * inv_a
        t1 = (-b + sd) * inv_a
        t = jnp.where(t0 >= 0.0, t0, jnp.where(t1 >= 0.0, t1, -1.0))
        ok = valid & (t >= 0.0)
        cand_d = jnp.where(ok, t, MJ_MAXVAL)
        nx_w = dx + rd[:, 0] * jnp.where(ok, t, 0.0)
        ny_w = dy + rd[:, 1] * jnp.where(ok, t, 0.0)
        nz_w = dz + rd[:, 2] * jnp.where(ok, t, 0.0)
        inv_n = 1.0 / jnp.maximum(jnp.sqrt(nx_w * nx_w + ny_w * ny_w + nz_w * nz_w), 1e-12)
        cand_nx, cand_ny, cand_nz = nx_w * inv_n, ny_w * inv_n, nz_w * inv_n

      elif gt == GEOM_ELLIPSOID:
        ix2 = 1.0 / jnp.maximum(sx * sx, MJ_MINVAL)
        iy2 = 1.0 / jnp.maximum(sy * sy, MJ_MINVAL)
        iz2 = 1.0 / jnp.maximum(sz * sz, MJ_MINVAL)
        a = ix2 * vx * vx + iy2 * vy * vy + iz2 * vz * vz
        b = ix2 * vx * lx + iy2 * vy * ly + iz2 * vz * lz
        c = ix2 * lx * lx + iy2 * ly * ly + iz2 * lz * lz - 1.0
        det = b * b - a * c
        valid = (det >= MJ_MINVAL) & (jnp.abs(a) > MJ_MINVAL)
        sd = jnp.sqrt(jnp.maximum(det, 0.0))
        inv_a = jnp.where(valid, 1.0 / jnp.where(valid, a, 1.0), 0.0)
        t0 = (-b - sd) * inv_a
        t1 = (-b + sd) * inv_a
        t = jnp.where(t0 >= 0.0, t0, jnp.where(t1 >= 0.0, t1, -1.0))
        ok = valid & (t >= 0.0)
        cand_d = jnp.where(ok, t, MJ_MAXVAL)
        t_safe = jnp.where(ok, t, 0.0)
        lnx = ix2 * (lx + vx * t_safe)
        lny = iy2 * (ly + vy * t_safe)
        lnz = iz2 * (lz + vz * t_safe)
        inv_n = 1.0 / jnp.maximum(jnp.sqrt(lnx * lnx + lny * lny + lnz * lnz), 1e-12)
        lnx, lny, lnz = lnx * inv_n, lny * inv_n, lnz * inv_n
        cand_nx = m00 * lnx + m01 * lny + m02 * lnz
        cand_ny = m10 * lnx + m11 * lny + m12 * lnz
        cand_nz = m20 * lnx + m21 * lny + m22 * lnz

      elif gt == GEOM_CAPSULE:
        sq_r = sx * sx
        ac = vx * vx + vy * vy
        bc = vx * lx + vy * ly
        cc = lx * lx + ly * ly - sq_r
        detc = bc * bc - ac * cc
        vc = (detc >= MJ_MINVAL) & (jnp.abs(ac) > MJ_MINVAL)
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
        vt = (dett >= MJ_MINVAL) & (jnp.abs(af) > MJ_MINVAL)
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
        vb = (detb >= MJ_MINVAL) & (jnp.abs(af) > MJ_MINVAL)
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
        ok = cand_d < (MJ_MAXVAL * 0.5)
        is_top = ok_top & (dt <= dc) & (dt <= db)
        is_bot = ok_bot & (db < dc) & (db < dt)
        t_safe = jnp.where(ok, cand_d, 0.0)
        lnx = lx + vx * t_safe
        lny = ly + vy * t_safe
        lnz = jnp.where(is_top | is_bot, lz + vz * t_safe + jnp.where(is_top, -sy, sy), 0.0)
        inv_n = 1.0 / jnp.maximum(jnp.sqrt(lnx * lnx + lny * lny + lnz * lnz), 1e-12)
        lnx, lny, lnz = lnx * inv_n, lny * inv_n, lnz * inv_n
        cand_nx = m00 * lnx + m01 * lny + m02 * lnz
        cand_ny = m10 * lnx + m11 * lny + m12 * lnz
        cand_nz = m20 * lnx + m21 * lny + m22 * lnz

      elif gt == GEOM_CYLINDER:
        sq_r = sx * sx
        ac = vx * vx + vy * vy
        bc = vx * lx + vy * ly
        cc = lx * lx + ly * ly - sq_r
        detc = bc * bc - ac * cc
        vc = (detc >= MJ_MINVAL) & (jnp.abs(ac) > MJ_MINVAL)
        sdc = jnp.sqrt(jnp.maximum(detc, 0.0))
        iac = jnp.where(vc, 1.0 / jnp.where(vc, ac, 1.0), 0.0)
        tc0 = (-bc - sdc) * iac
        tc1 = (-bc + sdc) * iac
        tc = jnp.where(tc0 >= 0.0, tc0, jnp.where(tc1 >= 0.0, tc1, -1.0))
        ok_s = vc & (tc >= 0.0) & (jnp.abs(lz + tc * vz) <= sy)
        ds = jnp.where(ok_s, tc, MJ_MAXVAL)

        ok_z = jnp.abs(vz) > MJ_MINVAL
        ivz = jnp.where(ok_z, 1.0 / jnp.where(ok_z, vz, 1.0), 0.0)
        tt = (sy - lz) * ivz
        ok_t = ok_z & (tt >= 0.0) & (((lx + tt * vx) ** 2 + (ly + tt * vy) ** 2) <= sq_r)
        dt = jnp.where(ok_t, tt, MJ_MAXVAL)
        tb = (-sy - lz) * ivz
        ok_b = ok_z & (tb >= 0.0) & (((lx + tb * vx) ** 2 + (ly + tb * vy) ** 2) <= sq_r)
        db = jnp.where(ok_b, tb, MJ_MAXVAL)

        cand_d = jnp.minimum(ds, jnp.minimum(dt, db))
        ok = cand_d < (MJ_MAXVAL * 0.5)
        is_top = ok_t & (dt <= ds) & (dt <= db)
        is_bot = ok_b & (db < ds) & (db < dt)
        t_safe = jnp.where(ok, cand_d, 0.0)
        hlx = lx + vx * t_safe
        hly = ly + vy * t_safe
        inv_n = 1.0 / jnp.maximum(jnp.sqrt(hlx * hlx + hly * hly), 1e-12)
        lnx = jnp.where(is_top | is_bot, 0.0, hlx * inv_n)
        lny = jnp.where(is_top | is_bot, 0.0, hly * inv_n)
        lnz = jnp.where(is_top, 1.0, jnp.where(is_bot, -1.0, 0.0))
        cand_nx = m00 * lnx + m01 * lny + m02 * lnz
        cand_ny = m10 * lnx + m11 * lny + m12 * lnz
        cand_nz = m20 * lnx + m21 * lny + m22 * lnz

      else:  # GEOM_BOX
        # 6-face exact slab test in local box frame
        okx = jnp.abs(vx) > MJ_MINVAL
        ivx = jnp.where(okx, 1.0 / jnp.where(okx, vx, 1.0), 0.0)
        tx0 = (-sx - lx) * ivx
        tx1 = (sx - lx) * ivx
        v_x0 = okx & (tx0 >= 0.0) & (jnp.abs(ly + tx0 * vy) <= sy) & (jnp.abs(lz + tx0 * vz) <= sz)
        v_x1 = okx & (tx1 >= 0.0) & (jnp.abs(ly + tx1 * vy) <= sy) & (jnp.abs(lz + tx1 * vz) <= sz)
        dx0 = jnp.where(v_x0, tx0, MJ_MAXVAL)
        dx1 = jnp.where(v_x1, tx1, MJ_MAXVAL)

        oky = jnp.abs(vy) > MJ_MINVAL
        ivy = jnp.where(oky, 1.0 / jnp.where(oky, vy, 1.0), 0.0)
        ty0 = (-sy - ly) * ivy
        ty1 = (sy - ly) * ivy
        v_y0 = oky & (ty0 >= 0.0) & (jnp.abs(lx + ty0 * vx) <= sx) & (jnp.abs(lz + ty0 * vz) <= sz)
        v_y1 = oky & (ty1 >= 0.0) & (jnp.abs(lx + ty1 * vx) <= sx) & (jnp.abs(lz + ty1 * vz) <= sz)
        dy0 = jnp.where(v_y0, ty0, MJ_MAXVAL)
        dy1 = jnp.where(v_y1, ty1, MJ_MAXVAL)

        okz = jnp.abs(vz) > MJ_MINVAL
        ivz = jnp.where(okz, 1.0 / jnp.where(okz, vz, 1.0), 0.0)
        tz0 = (-sz - lz) * ivz
        tz1 = (sz - lz) * ivz
        v_z0 = okz & (tz0 >= 0.0) & (jnp.abs(lx + tz0 * vx) <= sx) & (jnp.abs(ly + tz0 * vy) <= sy)
        v_z1 = okz & (tz1 >= 0.0) & (jnp.abs(lx + tz1 * vx) <= sx) & (jnp.abs(ly + tz1 * vy) <= sy)
        dz0 = jnp.where(v_z0, tz0, MJ_MAXVAL)
        dz1 = jnp.where(v_z1, tz1, MJ_MAXVAL)

        cand_d = jnp.minimum(jnp.minimum(dx0, dx1), jnp.minimum(jnp.minimum(dy0, dy1), jnp.minimum(dz0, dz1)))
        lnx = jnp.where(cand_d == dx0, -1.0, jnp.where(cand_d == dx1, 1.0, 0.0))
        lny = jnp.where((cand_d != dx0) & (cand_d != dx1) & (cand_d == dy0), -1.0,
                        jnp.where((cand_d != dx0) & (cand_d != dx1) & (cand_d == dy1), 1.0, 0.0))
        lnz = jnp.where((cand_d != dx0) & (cand_d != dx1) & (cand_d != dy0) & (cand_d != dy1) & (cand_d == dz0), -1.0,
                        jnp.where((cand_d != dx0) & (cand_d != dx1) & (cand_d != dy0) & (cand_d != dy1) & (cand_d == dz1), 1.0, 0.0))
        cand_nx = m00 * lnx + m01 * lny + m02 * lnz
        cand_ny = m10 * lnx + m11 * lny + m12 * lnz
        cand_nz = m20 * lnx + m21 * lny + m22 * lnz

      improved = cand_d < best_d
      best_d = jnp.where(improved, cand_d, best_d)
      best_gid = jnp.where(improved, jnp.int32(gid_val), best_gid)
      best_nx = jnp.where(improved, cand_nx, best_nx)
      best_ny = jnp.where(improved, cand_ny, best_ny)
      best_nz = jnp.where(improved, cand_nz, best_nz)

    hit = best_gid >= 0
    out_d_ref[...] = jnp.where(hit, best_d, -1.0)[:, None]
    out_n_ref[...] = jnp.where(hit[:, None], jnp.stack([best_nx, best_ny, best_nz], axis=-1), 0.0)
    out_g_ref[...] = best_gid[:, None]

  d_out, n_out, g_out = pl.pallas_call(
      _kernel,
      out_shape=[
          jax.ShapeDtypeStruct((R_pad, 1), jnp.float32),
          jax.ShapeDtypeStruct((R_pad, 3), jnp.float32),
          jax.ShapeDtypeStruct((R_pad, 1), jnp.int32),
      ],
      grid=(n_tiles,),
      in_specs=[
          pl.BlockSpec((BLOCK_P, 3), lambda i: (i, 0)),
          pl.BlockSpec((BLOCK_P, 3), lambda i: (i, 0)),
          pl.BlockSpec((BLOCK_P, 1), lambda i: (i, 0)),
          pl.BlockSpec((g_pad, 16), lambda i: (0, 0)),
      ],
      out_specs=[
          pl.BlockSpec((BLOCK_P, 1), lambda i: (i, 0)),
          pl.BlockSpec((BLOCK_P, 3), lambda i: (i, 0)),
          pl.BlockSpec((BLOCK_P, 1), lambda i: (i, 0)),
      ],
      interpret=interpret,
  )(orig_pad, dir_pad, md_pad[:, None], packed_geoms)

  return d_out[:R, 0], n_out[:R], g_out[:R, 0]


def pallas_mxu_pixel_mlp(
    x_in: jax.Array,
    w0: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    w3: jax.Array,
    interpret: bool = True,
) -> jax.Array:
  """VMEM-resident 4-layer BF16 MXU MLP kernel for Stage 2 PixelResMLP-128.

  Pins all weights (128x128 BF16 matrices, < 100 KB total) in VMEM and streams
  [BLOCK_P=128, 128] pixel feature tiles through the MXU with zero inter-layer HBM traffic.
  """
  N, C = x_in.shape
  assert C == 128, f"Expected MXU-aligned width 128, got {C}"
  x_pad = _pad_to_block(x_in, BLOCK_P, axis=0).astype(jnp.bfloat16)
  n_tiles = x_pad.shape[0] // BLOCK_P

  def _mlp_kernel(x_ref, w0_ref, w1_ref, w2_ref, w3_ref, out_ref):
    h = x_ref[...].astype(jnp.bfloat16)
    h = jax.nn.relu(jnp.dot(h, w0_ref[...], preferred_element_type=jnp.float32)).astype(jnp.bfloat16)
    h = jax.nn.relu(jnp.dot(h, w1_ref[...], preferred_element_type=jnp.float32)).astype(jnp.bfloat16)
    h = jax.nn.relu(jnp.dot(h, w2_ref[...], preferred_element_type=jnp.float32)).astype(jnp.bfloat16)
    out = jnp.dot(h, w3_ref[...], preferred_element_type=jnp.float32)  # (BLOCK_P, 128)
    out_ref[...] = out

  out_pad = pl.pallas_call(
      _mlp_kernel,
      out_shape=jax.ShapeDtypeStruct((x_pad.shape[0], 128), jnp.float32),
      grid=(n_tiles,),
      in_specs=[
          pl.BlockSpec((BLOCK_P, 128), lambda i: (i, 0)),
          pl.BlockSpec((128, 128), lambda i: (0, 0)),
          pl.BlockSpec((128, 128), lambda i: (0, 0)),
          pl.BlockSpec((128, 128), lambda i: (0, 0)),
          pl.BlockSpec((128, 128), lambda i: (0, 0)),
      ],
      out_specs=pl.BlockSpec((BLOCK_P, 128), lambda i: (i, 0)),
      interpret=interpret,
  )(x_pad, w0.astype(jnp.bfloat16), w1.astype(jnp.bfloat16), w2.astype(jnp.bfloat16), w3.astype(jnp.bfloat16))
  return out_pad[:N, :3]
