"""Test complete Fused Pallas V3 Rigid-Body 3DGS E2E on live TPU v5p."""

from __future__ import annotations

import math
import re
import time
import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
import numpy as np

from profile_v2_gs3d import load_spec, _tile_inputs, bench_fn
from tpu_mjx_render.gs3d import render_gs3d_batch
from tpu_mjx_render.roofline import inspect_hlo_ops


def ewa_per_body_packed16(spec, obj_xpos, obj_xmat, cam_xpos, cam_xmat, body_slices):
  """Compute per-Gaussian attributes directly per rigid body (zero (B, N, 3, 3) tensors).

  Returns:
    splats_b16n: (B, 16, N_pad) float32 where rows 0..12 are:
      0:mx, 1:my, 2:icxx, 3:icxy, 4:icyy, 5:sr, 6:sg, 7:sb, 8:op, 9:zd, 10:nx, 11:ny, 12:nz
    mean_x: (B, N)
    mean_y: (B, N)
    z_d:    (B, N)
    valid_g:(B, N) bool
    radius: (B, N)
  """
  B = obj_xpos.shape[0]
  H, W = spec.height, spec.width
  N = spec.n_gaussians
  N_pad = ((N + 127) // 128) * 128

  l_pos = jnp.asarray(spec.light_pos, jnp.float32)
  l_amb = jnp.asarray(spec.light_ambient, jnp.float32)
  l_diff = jnp.asarray(spec.light_diffuse, jnp.float32)
  half_h = math.tan(0.5 * math.radians(spec.cam_fovy))
  fy = 0.5 * float(H) / half_h
  fx = fy

  R_co = jnp.einsum("bji,bmjk->bmik", cam_xmat, obj_xmat, precision=jax.lax.Precision.HIGHEST)
  t_co = jnp.einsum("bji,bmj->bmi", cam_xmat, obj_xpos - cam_xpos[:, None, :], precision=jax.lax.Precision.HIGHEST)

  chunks = []
  mx_l, my_l, zd_l, vg_l, rad_l = [], [], [], [], []
  for m, mu_l, cov_l, nrm_l, rgb_l, op_l in body_slices:
    Rm = obj_xmat[:, m]
    pm = obj_xpos[:, m]
    Rcm = R_co[:, m]
    tcm = t_co[:, m]

    l_loc = jnp.einsum("bji,bj->bi", Rm, l_pos[None, :] - pm, precision=jax.lax.Precision.HIGHEST)
    c_loc = jnp.einsum("bji,bj->bi", Rm, cam_xpos - pm, precision=jax.lax.Precision.HIGHEST)
    ldir_l = l_loc[:, None, :] - mu_l[None, :, :]
    ldir_l = ldir_l / jnp.maximum(jnp.linalg.norm(ldir_l, axis=-1, keepdims=True), 1e-12)
    vdir_l = c_loc[:, None, :] - mu_l[None, :, :]
    vdir_l = vdir_l / jnp.maximum(jnp.linalg.norm(vdir_l, axis=-1, keepdims=True), 1e-12)
    nrm_vis_l = jnp.where(jnp.sum(nrm_l[None, :, :] * vdir_l, axis=-1, keepdims=True) < 0.0, -nrm_l[None, :, :], nrm_l[None, :, :])
    ndotl = jnp.maximum(0.0, jnp.sum(nrm_vis_l * ldir_l, axis=-1))
    rgb_lit = jnp.clip(rgb_l[None, :, :] * (l_amb[None, None, :] + l_diff[None, None, :] * ndotl[..., None]), 0.0, 1.0)
    nrm_vis = jnp.einsum("bij,bnj->bni", Rm, nrm_vis_l, precision=jax.lax.Precision.HIGHEST)

    p_cam = tcm[:, None, :] + jnp.einsum("bij,nj->bni", Rcm, mu_l, precision=jax.lax.Precision.HIGHEST)
    rc = jnp.einsum("bij,njk->bnik", Rcm, cov_l, precision=jax.lax.Precision.HIGHEST)
    cov_cam = jnp.einsum("bnik,blk->bnil", rc, Rcm, precision=jax.lax.Precision.HIGHEST)

    cx, cy, cz = p_cam[..., 0], p_cam[..., 1], p_cam[..., 2]
    z_d = -cz
    in_front = z_d > spec.znear
    z_safe = jnp.maximum(z_d, spec.znear)
    mean_x = 0.5 * float(W) + fx * (cx / z_safe)
    mean_y = 0.5 * float(H) - fy * (cy / z_safe)

    inv_z = 1.0 / z_safe
    inv_z2 = inv_z * inv_z
    lim_x = 1.3 * (0.5 * float(W) / fx) * z_safe
    lim_y = 1.3 * (0.5 * float(H) / fy) * z_safe
    cx_c = jnp.clip(cx, -lim_x, lim_x)
    cy_c = jnp.clip(cy, -lim_y, lim_y)
    j00, j02 = fx * inv_z, fx * cx_c * inv_z2
    j11, j12 = -fy * inv_z, -fy * cy_c * inv_z2

    s00 = j00 * cov_cam[..., 0, 0] + j02 * cov_cam[..., 2, 0]
    s01 = j00 * cov_cam[..., 0, 1] + j02 * cov_cam[..., 2, 1]
    s02 = j00 * cov_cam[..., 0, 2] + j02 * cov_cam[..., 2, 2]
    s11 = j11 * cov_cam[..., 1, 1] + j12 * cov_cam[..., 2, 1]
    s12 = j11 * cov_cam[..., 1, 2] + j12 * cov_cam[..., 2, 2]
    cov2_xx = s00 * j00 + s02 * j02 + 0.30
    cov2_xy = s01 * j11 + s02 * j12
    cov2_yy = s11 * j11 + s12 * j12 + 0.30
    det = cov2_xx * cov2_yy - cov2_xy * cov2_xy
    valid_g = in_front & (det > 1e-6)
    inv_det = jnp.where(valid_g, 1.0 / jnp.maximum(det, 1e-6), 0.0)
    icxx, icxy, icyy = cov2_yy * inv_det, -cov2_xy * inv_det, cov2_xx * inv_det
    mid = 0.5 * (cov2_xx + cov2_yy)
    disc = jnp.sqrt(jnp.maximum(mid * mid - det, 0.0))
    radius = 3.0 * jnp.sqrt(jnp.maximum(mid + disc, 0.1))

    Nm = mu_l.shape[0]
    op_bn = jnp.broadcast_to(op_l[None, :], (B, Nm))
    z_bn = jnp.zeros((B, Nm), jnp.float32)
    # Stack (B, 16, Nm)
    c16 = jnp.stack(
        [
            mean_x, mean_y, icxx, icxy, icyy,
            rgb_lit[..., 0], rgb_lit[..., 1], rgb_lit[..., 2],
            op_bn, z_d, nrm_vis[..., 0], nrm_vis[..., 1], nrm_vis[..., 2],
            z_bn, z_bn, z_bn,
        ],
        axis=1,
    )
    chunks.append(c16)
    mx_l.append(mean_x); my_l.append(mean_y); zd_l.append(z_d); vg_l.append(valid_g); rad_l.append(radius)

  splats_b16n = jnp.concatenate(chunks, axis=2)
  if N_pad > N:
    splats_b16n = jnp.pad(splats_b16n, ((0, 0), (0, 0), (0, N_pad - N)))
  return (
      splats_b16n,
      jnp.concatenate(mx_l, axis=1),
      jnp.concatenate(my_l, axis=1),
      jnp.concatenate(zd_l, axis=1),
      jnp.concatenate(vg_l, axis=1),
      jnp.concatenate(rad_l, axis=1),
  )


def pallas_v3_fused_rasterize(
    splats_b16n: jax.Array,  # (B, 16, N_pad)
    top_idx: jax.Array,      # (B, T, K) with -1 for invalid
    t_ox: np.ndarray,        # (T,)
    t_oy: np.ndarray,        # (T,)
    tile_size: int = 8,
    use_bf16_mxu: bool = False,
    t_grp: int = 8,
):
  B, _, N_pad = splats_b16n.shape
  _, T, K = top_idx.shape
  P_t = tile_size * tile_size  # 64
  assert P_t == 64 and K == 128 and N_pad % 128 == 0 and T % t_grp == 0
  n_chunks = N_pad // 128
  n_tg = T // t_grp

  lx = np.tile(np.arange(tile_size, dtype=np.float32), tile_size) + 0.5
  ly = np.repeat(np.arange(tile_size, dtype=np.float32), tile_size) + 0.5
  loc_xy_dev = jnp.asarray(
      np.stack([np.broadcast_to(lx[:, None], (P_t, K)), np.broadcast_to(ly[:, None], (P_t, K))], axis=0).reshape(2 * P_t, K),
      jnp.float32,
  )
  u_np = np.triu(np.ones((K, K), dtype=np.float32), k=1)
  u_dev = jnp.asarray(u_np, jnp.bfloat16 if use_bf16_mxu else jnp.float32)

  # Tile origins: (n_tg * oxy_rows, 128) where row 0..t_grp-1 is ox, row t_grp..2*t_grp-1 is oy
  oxy_rows = ((2 * t_grp + 7) // 8) * 8
  t_oxy_np = np.zeros((n_tg, oxy_rows, 128), dtype=np.float32)
  t_oxy_np[:, :t_grp, :] = t_ox.reshape(n_tg, t_grp, 1)
  t_oxy_np[:, t_grp : 2 * t_grp, :] = t_oy.reshape(n_tg, t_grp, 1)
  t_oxy_dev = jnp.asarray(t_oxy_np.reshape(n_tg * oxy_rows, 128), jnp.float32)

  out_cols = t_grp * 8  # 64 for t_grp=8, 128 for t_grp=16

  def _fused_kernel(s_ref, i_ref, oxy_ref, lxy_ref, u_ref, out_ref):
    s_all = s_ref[...]    # (16, N_pad)
    ix = i_ref[...]       # (t_grp, 128)
    oxy = oxy_ref[...]    # (oxy_rows, 128)
    lxy = lxy_ref[...]    # (128, 128)
    lx_m, ly_m = lxy[0:64, :], lxy[64:128, :]
    u_m = u_ref[...]
    ones_k = jnp.ones((1, K), jnp.float32)

    # 1. Chunked VMEM take_along_axis gather across N_pad (zero HBM intermediate!)
    ix_tiles = [jnp.broadcast_to(ix[t : t + 1, :], (16, K)) for t in range(t_grp)]
    accs = [jnp.zeros((16, K), jnp.float32) for _ in range(t_grp)]
    for c in range(n_chunks):
      base = c * 128
      s_c = s_all[:, base : base + 128]
      for t in range(t_grp):
        ix_t = ix_tiles[t]
        mask = (ix_t >= base) & (ix_t < base + 128)
        local_ix = jnp.where(mask, ix_t - base, 0)
        vals = jnp.take_along_axis(s_c, local_ix, axis=1)
        accs[t] = accs[t] + jnp.where(mask, vals, 0.0)

    # 2. V3 multi-tile rasterizer directly from gathered VMEM registers
    outs = []
    for t in range(t_grp):
      sf = accs[t]  # (16, 128)
      px = jnp.broadcast_to(oxy[t : t + 1, 0:1], (P_t, K)) + lx_m
      py = jnp.broadcast_to(oxy[t_grp + t : t_grp + t + 1, 0:1], (P_t, K)) + ly_m
      dx = px - jnp.broadcast_to(sf[0:1, :], (P_t, K))
      dy = py - jnp.broadcast_to(sf[1:2, :], (P_t, K))
      cxx = jnp.broadcast_to(sf[2:3, :], (P_t, K))
      cxy = jnp.broadcast_to(sf[3:4, :], (P_t, K))
      cyy = jnp.broadcast_to(sf[4:5, :], (P_t, K))
      gop = jnp.broadcast_to(sf[8:9, :], (P_t, K))

      power = -0.5 * (cxx * dx * dx + 2.0 * cxy * dx * dy + cyy * dy * dy)
      valid_p = (power <= 0.0) & (power > -16.0)
      alpha = jnp.where(valid_p, jnp.minimum(0.99, gop * jnp.exp(jnp.maximum(power, -16.0))), 0.0)
      alpha = jnp.where(alpha >= (1.0 / 255.0), alpha, 0.0)

      log_1ma = jnp.log(jnp.maximum(1.0 - alpha, 1.0e-7))
      if use_bf16_mxu:
        log_tr = jnp.matmul(log_1ma.astype(jnp.bfloat16), u_m, preferred_element_type=jnp.float32)
      else:
        log_tr = jnp.matmul(log_1ma, u_m, precision=jax.lax.Precision.HIGHEST)
      trans = jnp.exp(jnp.maximum(log_tr, -30.0))
      w = alpha * trans

      # Pad attr to (128, 128)? Or matmul (64, 128) @ (128, 8)
      attr_8 = jnp.concatenate([sf[5:8, :], sf[9:13, :], ones_k], axis=0)  # (8, 128)
      # Use VPU sum or matmul? Wait: let's test (64, 128) @ (128, 8) vs sum!
      attr_t = jnp.transpose(attr_8, (1, 0))  # (128, 8)
      outs.append(jnp.matmul(w, attr_t, precision=jax.lax.Precision.HIGHEST))

    out_ref[...] = jnp.concatenate(outs, axis=1)  # (64, t_grp * 8)

  out_raw = pl.pallas_call(
      _fused_kernel,
      out_shape=jax.ShapeDtypeStruct((B * n_tg * P_t, out_cols), jnp.float32),
      grid=(B, n_tg),
      in_specs=[
          pl.BlockSpec((16, N_pad), lambda b, tg: (b, 0)),
          pl.BlockSpec((t_grp, K), lambda b, tg: (b * n_tg + tg, 0)),
          pl.BlockSpec((oxy_rows, 128), lambda b, tg: (tg, 0)),
          pl.BlockSpec((2 * P_t, K), lambda b, tg: (0, 0)),
          pl.BlockSpec((K, K), lambda b, tg: (0, 0)),
      ],
      out_specs=pl.BlockSpec((P_t, out_cols), lambda b, tg: (b * n_tg + tg, 0)),
      compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
      interpret=False,
  )(
      splats_b16n.reshape(B * 16, N_pad),
      top_idx.reshape(B * T, K),
      t_oxy_dev,
      loc_xy_dev,
      u_dev,
  )
  # out_raw: (B, n_tg, P_t, t_grp, 8) -> transpose to (B, n_tg, t_grp, P_t, 8) -> (B, T, 8, 8, 8)
  unp = out_raw.reshape(B, n_tg, P_t, t_grp, 8).transpose(0, 1, 3, 2, 4).reshape(B, T, tile_size, tile_size, 8)
  return unp[..., 0:3], unp[..., 3], unp[..., 4:7], unp[..., 7]


def render_gs3d_v3(spec, obj_xpos, obj_xmat, cam_xpos, cam_xmat, body_slices, use_bf16_mxu=False, t_grp=8):
  B = obj_xpos.shape[0]
  H, W = spec.height, spec.width
  tile_size = 8
  top_k = 128
  Ty, Tx = H // tile_size, W // tile_size
  T = Ty * Tx

  splats_b16n, mean_x, mean_y, z_d, valid_g, radius = ewa_per_body_packed16(
      spec, obj_xpos, obj_xmat, cam_xpos, cam_xmat, body_slices
  )

  t_ox = np.tile(np.arange(Tx, dtype=np.float32) * tile_size, Ty)
  t_oy = np.repeat(np.arange(Ty, dtype=np.float32) * tile_size, Tx)
  tox_dev = jnp.asarray(t_ox, jnp.float32)[None, :, None]
  toy_dev = jnp.asarray(t_oy, jnp.float32)[None, :, None]

  near_x = jnp.clip(mean_x[:, None, :], tox_dev, tox_dev + float(tile_size))
  near_y = jnp.clip(mean_y[:, None, :], toy_dev, toy_dev + float(tile_size))
  dx_t = mean_x[:, None, :] - near_x
  dy_t = mean_y[:, None, :] - near_y
  overlap = valid_g[:, None, :] & ((dx_t * dx_t + dy_t * dy_t) <= (radius * radius)[:, None, :])

  score = jnp.where(overlap, -z_d[:, None, :], -1.0e9)
  top_scores, top_idx = jax.lax.top_k(score, top_k)
  top_idx = jnp.where(top_scores > -1.0e8, top_idx, -1)

  rgb_t, dep_t, nrm_t, alp_t = pallas_v3_fused_rasterize(
      splats_b16n, top_idx, t_ox, t_oy, tile_size=tile_size, use_bf16_mxu=use_bf16_mxu, t_grp=t_grp
  )

  rgb_img = rgb_t.reshape(B, Ty, Tx, tile_size, tile_size, 3).transpose(0, 1, 3, 2, 4, 5).reshape(B, H, W, 3)
  dep_raw = dep_t.reshape(B, Ty, Tx, tile_size, tile_size).transpose(0, 1, 3, 2, 4).reshape(B, H, W)
  nrm_img = nrm_t.reshape(B, Ty, Tx, tile_size, tile_size, 3).transpose(0, 1, 3, 2, 4, 5).reshape(B, H, W, 3)
  alp_img = alp_t.reshape(B, Ty, Tx, tile_size, tile_size).transpose(0, 1, 3, 2, 4).reshape(B, H, W)

  sky_v = jnp.linspace(1.0, 0.0, H, dtype=jnp.float32)[None, :, None, None]
  sky = sky_v * jnp.asarray(spec.sky_rgb1, jnp.float32)[None, None, None, :] + (1.0 - sky_v) * jnp.asarray(spec.sky_rgb2, jnp.float32)[None, None, None, :]
  rgb_final = jnp.clip(rgb_img + (1.0 - alp_img[..., None]) * sky, 0.0, 1.0)
  depth_final = jnp.where(alp_img > 0.1, dep_raw / jnp.maximum(alp_img, 1e-4), -1.0)
  return rgb_final, depth_final, nrm_img, alp_img


def main():
  ds, spec = load_spec(2048, 64, 64)
  # Check that body_ids are contiguous as constructed by mesh_scene_to_rigid_3dgs
  body_slices = []
  offset = 0
  for m in range(spec.n_bodies):
    idx = np.where(spec.body_id == m)[0]
    assert np.array_equal(idx, np.arange(offset, offset + len(idx))), "body_id must be contiguous"
    offset += len(idx)
    body_slices.append((
        m,
        jnp.asarray(spec.mu_local[idx], jnp.float32),
        jnp.asarray(spec.cov_local[idx], jnp.float32),
        jnp.asarray(spec.nrm_local[idx], jnp.float32),
        jnp.asarray(spec.rgb_base[idx], jnp.float32),
        jnp.asarray(spec.opacity[idx], jnp.float32),
    ))

  B = 1024
  args = _tile_inputs(ds, B)

  # Reference V2 jax_tile
  fn_ref = jax.jit(lambda a, b, c, d: render_gs3d_batch(spec, a, b, c, d, tile_size=8, top_k=128, mode="jax_tile"))
  ms_ref, _, (ref_rgb, ref_dep, ref_nrm, ref_alp) = bench_fn(fn_ref, args, n_warmup=2, n_steps=5)
  print(f"REF jax_tile: {ms_ref:.2f} ms ({B*1e3/ms_ref:.1f} FPS)", flush=True)

  for use_bf16, t_grp in [(False, 8), (True, 8)]:
    fn_v3 = jax.jit(lambda a, b, c, d: render_gs3d_v3(spec, a, b, c, d, body_slices, use_bf16_mxu=use_bf16, t_grp=t_grp))
    txt = fn_v3.lower(*args).as_text()
    hlo = inspect_hlo_ops(txt)
    sr = len(re.findall(r"\bsend\(|\brecv\(", txt, re.I))
    assert hlo["host_callbacks"] == 0 and sr == 0, hlo

    ms_v3, comp_s, (v3_rgb, v3_dep, v3_nrm, v3_alp) = bench_fn(fn_v3, args, n_warmup=3, n_steps=10)
    max_rgb = float(jnp.max(jnp.abs(ref_rgb[:32] - v3_rgb[:32])))
    mae_dep = float(jnp.mean(jnp.abs(ref_dep[:32] - v3_dep[:32])))
    mse = float(jnp.mean((ref_rgb[:32] - v3_rgb[:32]) ** 2))
    psnr = -10.0 * math.log10(max(mse, 1e-12)) if mse > 0 else 999.0
    print(
        f"V3 FUSED (bf16_mxu={use_bf16}, t_grp={t_grp}): {ms_v3:.2f} ms ({B*1e3/ms_v3:.1f} FPS, {ms_ref/ms_v3:.2f}x vs jax_tile), "
        f"compile={comp_s:.2f}s | max_rgb_diff={max_rgb:.2e}, depth_mae={mae_dep:.2e}, psnr={psnr:.2f} dB, hlo_cb={hlo['host_callbacks']}",
        flush=True,
    )


if __name__ == "__main__":
  main()
