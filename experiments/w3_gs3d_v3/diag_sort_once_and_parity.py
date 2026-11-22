"""Test exact FP32 parity fix in EWA, approx_max_k parity, and Sort-Once VMEM Compaction on w3."""

from __future__ import annotations

import math
import time
import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
import numpy as np

from profile_v2_gs3d import load_spec, _tile_inputs, bench_fn
from test_fused_v3_e2e import pallas_v3_fused_rasterize
from tpu_mjx_render.gs3d import render_gs3d_batch


def ewa_per_body_exact(spec, obj_xpos, obj_xmat, cam_xpos, cam_xmat, body_slices):
  """Per-body EWA matching V2 floating-point order for p_cam while avoiding (B, N, 3, 3) tensors."""
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

  chunks = []
  mx_l, my_l, zd_l, vg_l, rad_l = [], [], [], [], []
  for m, mu_l, cov_l, nrm_l, rgb_l, op_l in body_slices:
    Rm = obj_xmat[:, m]
    pm = obj_xpos[:, m]
    Rcm = R_co[:, m]

    # Exact world-space mu_w and nrm_w per body (zero (B, N, 3, 3) gather!)
    mu_w = pm[:, None, :] + jnp.einsum("bij,nj->bni", Rm, mu_l, precision=jax.lax.Precision.HIGHEST)
    nrm_w = jnp.einsum("bij,nj->bni", Rm, nrm_l, precision=jax.lax.Precision.HIGHEST)
    nrm_w = nrm_w / jnp.maximum(jnp.linalg.norm(nrm_w, axis=-1, keepdims=True), 1e-12)

    ldir = l_pos[None, None, :] - mu_w
    ldir = ldir / jnp.maximum(jnp.linalg.norm(ldir, axis=-1, keepdims=True), 1e-12)
    vdir = cam_xpos[:, None, :] - mu_w
    vdir = vdir / jnp.maximum(jnp.linalg.norm(vdir, axis=-1, keepdims=True), 1e-12)
    nrm_vis = jnp.where(jnp.sum(nrm_w * vdir, axis=-1, keepdims=True) < 0.0, -nrm_w, nrm_w)
    ndotl = jnp.maximum(0.0, jnp.sum(nrm_vis * ldir, axis=-1))
    rgb_lit = jnp.clip(rgb_l[None, :, :] * (l_amb[None, None, :] + l_diff[None, None, :] * ndotl[..., None]), 0.0, 1.0)

    p_cam = jnp.einsum("bji,bnj->bni", cam_xmat, mu_w - cam_xpos[:, None, :], precision=jax.lax.Precision.HIGHEST)
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


def render_v3_with_options(spec, obj_xpos, obj_xmat, cam_xpos, cam_xmat, body_slices, use_approx=False, use_bf16_mxu=False, t_grp=16):
  B = obj_xpos.shape[0]
  H, W = spec.height, spec.width
  tile_size = 8
  top_k = 128
  Ty, Tx = H // tile_size, W // tile_size
  T = Ty * Tx

  splats_b16n, mean_x, mean_y, z_d, valid_g, radius = ewa_per_body_exact(
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
  if use_approx:
    top_scores, top_idx = jax.lax.approx_max_k(score, top_k, recall_target=1.0)
    # approx_max_k does NOT guarantee sorted order! Sort the K=128 elements front-to-back!
    neg_ts, perm = jax.lax.sort_key_val(-top_scores, top_idx, dimension=-1)
    top_scores = -neg_ts
    top_idx = perm
  else:
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


def test_sort_once_compaction(spec, args, body_slices):
  """Test Sort-Once (B, N) + Pallas VMEM overlap + MXU-cumsum compaction + rasterization!"""
  B = args[0].shape[0]
  H, W = spec.height, spec.width
  N = spec.n_gaussians
  N_pad = ((N + 127) // 128) * 128
  n_chunks = N_pad // 128
  tile_size = 8
  Ty, Tx = H // tile_size, W // tile_size
  T = Ty * Tx
  K = 128
  P_t = 64
  t_grp = 8  # 8 tiles per block -> overlap is (8, 128) = 1 VREG!
  n_tg = T // t_grp

  t_ox = np.tile(np.arange(Tx, dtype=np.float32) * tile_size, Ty)
  t_oy = np.repeat(np.arange(Ty, dtype=np.float32) * tile_size, Tx)
  t_oxy_np = np.zeros((n_tg, 16, 128), dtype=np.float32)
  t_oxy_np[:, :t_grp, :] = t_ox.reshape(n_tg, t_grp, 1)
  t_oxy_np[:, t_grp : 2 * t_grp, :] = t_oy.reshape(n_tg, t_grp, 1)
  t_oxy_dev = jnp.asarray(t_oxy_np.reshape(n_tg * 16, 128), jnp.float32)

  lx = np.tile(np.arange(tile_size, dtype=np.float32), tile_size) + 0.5
  ly = np.repeat(np.arange(tile_size, dtype=np.float32), tile_size) + 0.5
  loc_xy_dev = jnp.asarray(
      np.stack([np.broadcast_to(lx[:, None], (P_t, K)), np.broadcast_to(ly[:, None], (P_t, K))], axis=0).reshape(2 * P_t, K),
      jnp.float32,
  )
  u_dev = jnp.asarray(np.triu(np.ones((K, K), dtype=np.float32), k=1), jnp.float32)
  l_128_bf16 = jnp.asarray(np.triu(np.ones((128, 128), dtype=np.float32), k=0), jnp.bfloat16)  # L[i, j] = 1 iff i <= j -> x @ L = inclusive cumsum!

  def _sort_once_pipeline(obj_xpos, obj_xmat, cam_xpos, cam_xmat):
    splats_b16n, mean_x, mean_y, z_d, valid_g, radius = ewa_per_body_exact(
        spec, obj_xpos, obj_xmat, cam_xpos, cam_xmat, body_slices
    )
    # Sort once per world (B, N): put invalid_g at +1e9
    sort_key = jnp.where(valid_g, z_d, 1.0e9)
    iota_n = jnp.broadcast_to(jnp.arange(N_pad, dtype=jnp.int32)[None, :], (B, N_pad))
    _, perm = jax.lax.sort_key_val(sort_key, iota_n, dimension=1)
    # Put valid_g and radius in rows 13, 14 of splats_b16n so ONE gather sorts all 16 rows!
    splats_b16n = splats_b16n.at[:, 13, :N].set(jnp.where(valid_g, radius, -1.0))
    # Gather (B, 16, N) -> (B, 16, N) via take_along_axis
    splats_sorted = jnp.take_along_axis(splats_b16n, perm[:, None, :], axis=2)

    def _k_compact(s_ref, oxy_ref, lxy_ref, u_ref, lcum_ref, out_ref):
      s_all = s_ref[...]     # (16, N_pad) sorted front-to-back!
      oxy = oxy_ref[...]     # (16, 128)
      lxy = lxy_ref[...]     # (128, 128)
      lx_m, ly_m = lxy[0:64, :], lxy[64:128, :]
      u_m = u_ref[...]
      lcum = lcum_ref[...]   # (128, 128) bf16
      ones_k = jnp.ones((1, K), jnp.float32)

      ox_col = oxy[0:8, 0:1]   # (8, 1)
      oy_col = oxy[8:16, 0:1]  # (8, 1)
      ox_b = jnp.broadcast_to(ox_col, (8, 128))
      oy_b = jnp.broadcast_to(oy_col, (8, 128))

      iota_128 = jnp.broadcast_to(jnp.arange(128, dtype=jnp.int32)[None, :], (128, 128))
      # Accumulate compacted splats for 8 tiles:
      accs = [jnp.zeros((16, 128), jnp.float32) for _ in range(t_grp)]
      counts = jnp.zeros((8, 1), jnp.float32)

      for c in range(n_chunks):
        base = c * 128
        sc = s_all[:, base : base + 128]  # (16, 128)
        mx_c = jnp.broadcast_to(sc[0:1, :], (8, 128))
        my_c = jnp.broadcast_to(sc[1:2, :], (8, 128))
        rad_c = jnp.broadcast_to(sc[13:14, :], (8, 128))

        nx_c = jnp.clip(mx_c, ox_b, ox_b + 8.0)
        ny_c = jnp.clip(my_c, oy_b, oy_b + 8.0)
        dx_c = mx_c - nx_c
        dy_c = my_c - ny_c
        ov_8 = (rad_c > 0.0) & ((dx_c * dx_c + dy_c * dy_c) <= (rad_c * rad_c))  # (8, 128) bool

        # Inclusive cumsum via MXU matmul for all 8 tiles at once!
        csum_8 = jnp.matmul(ov_8.astype(jnp.bfloat16), lcum, preferred_element_type=jnp.float32)  # (8, 128)
        rank_8 = counts + csum_8 - 1.0  # 0-based slot in [0, 128)
        counts = counts + csum_8[:, 127:128]

        # Split sc into hi+mid+lo BF16 for exact FP32 scatter
        s_hi = sc.astype(jnp.bfloat16)
        r1 = sc - s_hi.astype(jnp.float32)
        s_mid = r1.astype(jnp.bfloat16)
        s_lo = (r1 - s_mid.astype(jnp.float32)).astype(jnp.bfloat16)

        for t in range(t_grp):
          r_t = jnp.broadcast_to(jnp.transpose(rank_8[t : t + 1, :], (1, 0)), (128, 128))  # wait: transpose (1,128)->(128,1)?
          # Actually row n of rank_8[t] goes to col k iff rank_8[t, n] == k!
          # So rank_t_col is (128, 1) broadcast to (128, 128), ov_t_col is (128, 1)!
          ov_t = jnp.broadcast_to(jnp.transpose(ov_8[t : t + 1, :].astype(jnp.float32), (1, 0)), (128, 128)) > 0.5
          oh = (ov_t & (r_t.astype(jnp.int32) == iota_128)).astype(jnp.bfloat16)
          accs[t] = (
              accs[t]
              + jnp.matmul(s_hi, oh, preferred_element_type=jnp.float32)
              + jnp.matmul(s_mid, oh, preferred_element_type=jnp.float32)
              + jnp.matmul(s_lo, oh, preferred_element_type=jnp.float32)
          )

      outs = []
      for t in range(t_grp):
        sf = accs[t]
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
        log_tr = jnp.matmul(log_1ma, u_m, precision=jax.lax.Precision.HIGHEST)
        trans = jnp.exp(jnp.maximum(log_tr, -30.0))
        w = alpha * trans
        attr_8 = jnp.concatenate([sf[5:8, :], sf[9:13, :], ones_k], axis=0)
        outs.append(jnp.matmul(w, jnp.transpose(attr_8, (1, 0)), precision=jax.lax.Precision.HIGHEST))
      out_ref[...] = jnp.concatenate(outs, axis=1)

    out_raw = pl.pallas_call(
        _k_compact,
        out_shape=jax.ShapeDtypeStruct((B * n_tg * P_t, t_grp * 8), jnp.float32),
        grid=(B, n_tg),
        in_specs=[
            pl.BlockSpec((16, N_pad), lambda b, tg: (b, 0)),
            pl.BlockSpec((16, 128), lambda b, tg: (tg, 0)),
            pl.BlockSpec((2 * P_t, K), lambda b, tg: (0, 0)),
            pl.BlockSpec((K, K), lambda b, tg: (0, 0)),
            pl.BlockSpec((128, 128), lambda b, tg: (0, 0)),
        ],
        out_specs=pl.BlockSpec((P_t, t_grp * 8), lambda b, tg: (b * n_tg + tg, 0)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
        interpret=False,
    )(splats_sorted.reshape(B * 16, N_pad), t_oxy_dev, loc_xy_dev, u_dev, l_128_bf16)
    unp = out_raw.reshape(B, n_tg, P_t, t_grp, 8).transpose(0, 1, 3, 2, 4).reshape(B, T, tile_size, tile_size, 8)
    rgb_img = unp[..., 0:3].reshape(B, Ty, Tx, tile_size, tile_size, 3).transpose(0, 1, 3, 2, 4, 5).reshape(B, H, W, 3)
    alp_img = unp[..., 7].reshape(B, Ty, Tx, tile_size, tile_size).transpose(0, 1, 3, 2, 4).reshape(B, H, W)
    sky_v = jnp.linspace(1.0, 0.0, H, dtype=jnp.float32)[None, :, None, None]
    sky = sky_v * jnp.asarray(spec.sky_rgb1, jnp.float32)[None, None, None, :] + (1.0 - sky_v) * jnp.asarray(spec.sky_rgb2, jnp.float32)[None, None, None, :]
    return jnp.clip(rgb_img + (1.0 - alp_img[..., None]) * sky, 0.0, 1.0)

  fn_so = jax.jit(_sort_once_pipeline)
  ms_so, c_so, rgb_so = bench_fn(fn_so, args, n_warmup=2, n_steps=5)
  print(f"SORT_ONCE COMPACT V3 (B={B}): {ms_so:.2f} ms ({B*1e3/ms_so:.1f} FPS), compile={c_so:.2f}s", flush=True)
  return rgb_so


def main():
  ds, spec = load_spec(2048, 64, 64)
  body_slices = []
  for m in range(spec.n_bodies):
    idx = np.where(spec.body_id == m)[0]
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
  ref_rgb, ref_dep, _, _ = jax.jit(lambda a, b, c, d: render_gs3d_batch(spec, a, b, c, d, tile_size=8, top_k=128, mode="jax_tile"))(*[x[:32] for x in args])

  for use_approx, use_bf16 in [(False, False), (True, False), (True, True)]:
    fn = jax.jit(lambda a, b, c, d: render_v3_with_options(spec, a, b, c, d, body_slices, use_approx=use_approx, use_bf16_mxu=use_bf16, t_grp=16))
    ms, comp, (rgb, dep, _, _) = bench_fn(fn, args, n_warmup=3, n_steps=10)
    max_rgb = float(jnp.max(jnp.abs(ref_rgb - rgb[:32])))
    mae_dep = float(jnp.mean(jnp.abs(ref_dep - dep[:32])))
    mse = float(jnp.mean((ref_rgb - rgb[:32]) ** 2))
    psnr = -10.0 * math.log10(max(mse, 1e-12))
    print(f"V3 EXACT_EWA (approx={use_approx}, bf16={use_bf16}, t_grp=16): {ms:.2f} ms ({B*1e3/ms:.1f} FPS), comp={comp:.2f}s | max_rgb={max_rgb:.2e}, dep_mae={mae_dep:.2e}, psnr={psnr:.2f} dB", flush=True)

  try:
    rgb_so = test_sort_once_compaction(spec, args, body_slices)
    max_so = float(jnp.max(jnp.abs(ref_rgb - rgb_so[:32])))
    mse_so = float(jnp.mean((ref_rgb - rgb_so[:32]) ** 2))
    psnr_so = -10.0 * math.log10(max(mse_so, 1e-12))
    print(f"SORT_ONCE PARITY: max_rgb={max_so:.2e}, psnr={psnr_so:.2f} dB", flush=True)
  except Exception as e:
    print(f"SORT_ONCE FAILED: {type(e).__name__}: {str(e)[:400]}", flush=True)


if __name__ == "__main__":
  main()
