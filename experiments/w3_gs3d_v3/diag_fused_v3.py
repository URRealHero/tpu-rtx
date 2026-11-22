"""Test fused Pallas V3 (MXU gather + multi-tile zero-bloat raster), per-body EWA, and BF16 top_k on w3."""

from __future__ import annotations

import math
import time
import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
import numpy as np

from profile_v2_gs3d import load_spec, _tile_inputs, bench_fn, stage_frontend_and_binning
from tpu_mjx_render.gs3d import render_gs3d_batch


def test_take_128_chunks_vs_mxu_3part():
  B, T, K, N = 1024, 64, 128, 2048
  key = jax.random.PRNGKey(1)
  splats_b16n = jax.random.uniform(key, (B, 16, N), jnp.float32) * 100.0
  idx_btk = jax.random.randint(key, (B, T, K), 0, N, jnp.int32)
  ref = jnp.transpose(splats_b16n, (0, 2, 1))[jnp.arange(B)[:, None, None], idx_btk]

  N_CHUNKS = N // 128
  T_GRP = 8
  n_tg = T // T_GRP

  # Test A: take_along_axis on (16, 128) chunks (1 VREG along axis 1!)
  try:
    def _k_take_chunked(s_ref, i_ref, o_ref):
      s_all = s_ref[...]  # (16, 2048)
      ix = i_ref[...]     # (8, 128)
      ix_tiles = [jnp.broadcast_to(ix[t : t + 1, :], (16, 128)) for t in range(T_GRP)]
      accs = [jnp.zeros((16, 128), jnp.float32) for _ in range(T_GRP)]
      for c in range(N_CHUNKS):
        base = c * 128
        s_c = s_all[:, base : base + 128]
        for t in range(T_GRP):
          ix_t = ix_tiles[t]
          mask = (ix_t >= base) & (ix_t < base + 128)
          local_ix = jnp.where(mask, ix_t - base, 0)
          vals = jnp.take_along_axis(s_c, local_ix, axis=1)
          accs[t] = accs[t] + jnp.where(mask, vals, 0.0)
      o_ref[...] = jnp.concatenate(accs, axis=0)

    fn_tc = jax.jit(lambda s, ix: pl.pallas_call(
        _k_take_chunked,
        out_shape=jax.ShapeDtypeStruct((B * T * 16, K), jnp.float32),
        grid=(B, n_tg),
        in_specs=[
            pl.BlockSpec((16, N), lambda b, tg: (b, 0)),
            pl.BlockSpec((T_GRP, K), lambda b, tg: (b * n_tg + tg, 0)),
        ],
        out_specs=pl.BlockSpec((T_GRP * 16, K), lambda b, tg: (b * n_tg + tg, 0)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
        interpret=False,
    )(s.reshape(B * 16, N), ix.reshape(B * T, K)))
    ms, comp, out_tc = bench_fn(fn_tc, (splats_b16n, idx_btk), n_warmup=1, n_steps=5)
    got = jnp.transpose(out_tc.reshape(B, T, 16, K), (0, 1, 3, 2))
    print(f"take_along_axis (16,128) chunked: {ms:.2f} ms (compile {comp:.2f}s), max_diff={float(jnp.max(jnp.abs(got[:4]-ref[:4]))):.2e}", flush=True)
  except Exception as e:
    print(f"take_along_axis chunked FAILED: {type(e).__name__}: {str(e)[:300]}", flush=True)

  # Test B: MXU 3-part (hi + mid + lo) -> exact FP32!
  def _k_mxu_3part(s_ref, i_ref, o_ref):
    s_all = s_ref[...]
    ix = i_ref[...]
    ix_cols = [jnp.broadcast_to(ix[t : t + 1, :], (128, 128)) for t in range(T_GRP)]
    accs = [jnp.zeros((16, 128), jnp.float32) for _ in range(T_GRP)]
    iota_col = jnp.broadcast_to(jnp.transpose(jnp.arange(128, dtype=jnp.int32)[None, :], (1, 0)), (128, 128))
    for c in range(N_CHUNKS):
      base = c * 128
      sc = s_all[:, base : base + 128]
      s_hi = sc.astype(jnp.bfloat16)
      rem1 = sc - s_hi.astype(jnp.float32)
      s_mid = rem1.astype(jnp.bfloat16)
      s_lo = (rem1 - s_mid.astype(jnp.float32)).astype(jnp.bfloat16)
      iota_b = iota_col + base
      for t in range(T_GRP):
        oh = (ix_cols[t] == iota_b).astype(jnp.bfloat16)
        accs[t] = (
            accs[t]
            + jnp.matmul(s_hi, oh, preferred_element_type=jnp.float32)
            + jnp.matmul(s_mid, oh, preferred_element_type=jnp.float32)
            + jnp.matmul(s_lo, oh, preferred_element_type=jnp.float32)
        )
    o_ref[...] = jnp.concatenate(accs, axis=0)

  fn_m3 = jax.jit(lambda s, ix: pl.pallas_call(
      _k_mxu_3part,
      out_shape=jax.ShapeDtypeStruct((B * T * 16, K), jnp.float32),
      grid=(B, n_tg),
      in_specs=[
          pl.BlockSpec((16, N), lambda b, tg: (b, 0)),
          pl.BlockSpec((T_GRP, K), lambda b, tg: (b * n_tg + tg, 0)),
      ],
      out_specs=pl.BlockSpec((T_GRP * 16, K), lambda b, tg: (b * n_tg + tg, 0)),
      compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
      interpret=False,
  )(s.reshape(B * 16, N), ix.reshape(B * T, K)))
  ms, comp, out_m3 = bench_fn(fn_m3, (splats_b16n, idx_btk), n_warmup=1, n_steps=5)
  got3 = jnp.transpose(out_m3.reshape(B, T, 16, K), (0, 1, 3, 2))
  print(f"MXU 3-part exact FP32 gather: {ms:.2f} ms (compile {comp:.2f}s), max_diff={float(jnp.max(jnp.abs(got3[:4]-ref[:4]))):.2e}", flush=True)


def test_raster_compile_and_speed():
  """Test M_TILES=8/16 with matmul(w, attr) instead of 8x for-loop for fast compile + speed."""
  B, T, K = 1024, 64, 128
  BT = B * T
  P_t = 64
  key = jax.random.PRNGKey(2)
  splats = jax.random.uniform(key, (BT, 16, K), jnp.float32)
  splats = splats.at[:, 2, :].set(0.5).at[:, 3, :].set(0.0).at[:, 4, :].set(0.5).at[:, 8, :].set(0.2)

  lx = np.tile(np.arange(8, dtype=np.float32), 8) + 0.5
  ly = np.repeat(np.arange(8, dtype=np.float32), 8) + 0.5
  loc_xy_dev = jnp.asarray(
      np.stack([np.broadcast_to(lx[:, None], (P_t, K)), np.broadcast_to(ly[:, None], (P_t, K))], axis=0).reshape(2 * P_t, K),
      jnp.float32,
  )
  u_128_bf16 = jnp.asarray(np.triu(np.ones((K, K), dtype=np.float32), k=1), jnp.bfloat16)

  for M_TILES in [4, 8, 16]:
    n_macro = BT // M_TILES
    splats_2d = splats.reshape(n_macro * M_TILES * 16, K)
    # Output packs M_TILES * 8 cols if <= 128 (4->32, 8->64, 16->128)!
    out_cols = M_TILES * 8

    def _k_mm(spl_ref, lxy_ref, u_ref, out_ref):
      sf_all = spl_ref[...]
      lxy = lxy_ref[...]
      lx_m, ly_m = lxy[0:64, :], lxy[64:128, :]
      u_m = u_ref[...]
      ones_k = jnp.ones((1, K), jnp.float32)
      outs = []
      for m in range(M_TILES):
        b0 = m * 16
        sf = sf_all[b0 : b0 + 16, :]
        px = jnp.broadcast_to(sf[13:14, 0:1], (P_t, K)) + lx_m
        py = jnp.broadcast_to(sf[14:15, 0:1], (P_t, K)) + ly_m
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
        log_tr = jnp.matmul(log_1ma.astype(jnp.bfloat16), u_m, preferred_element_type=jnp.float32)
        trans = jnp.exp(jnp.maximum(log_tr, -30.0))
        w = alpha * trans  # (64, 128)
        # Pack 8 attrs (8, 128) -> transpose (128, 8) -> w @ attr_t = (64, 8)
        attr_t = jnp.transpose(jnp.concatenate([sf[5:8, :], sf[9:13, :], ones_k], axis=0), (1, 0))
        # Pad attr_t to (128, 128) or use (64, 128) @ (128, 8) directly? Let's test (128, 8) directly!
        outs.append(jnp.matmul(w, attr_t, precision=jax.lax.Precision.HIGHEST))
      out_ref[...] = jnp.concatenate(outs, axis=1)

    fn = jax.jit(lambda: pl.pallas_call(
        _k_mm,
        out_shape=jax.ShapeDtypeStruct((n_macro * P_t, out_cols), jnp.float32),
        grid=(n_macro,),
        in_specs=[
            pl.BlockSpec((M_TILES * 16, K), lambda i: (i, 0)),
            pl.BlockSpec((2 * P_t, K), lambda i: (0, 0)),
            pl.BlockSpec((K, K), lambda i: (0, 0)),
        ],
        out_specs=pl.BlockSpec((P_t, out_cols), lambda i: (i, 0)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=False,
    )(splats_2d, loc_xy_dev, u_128_bf16))
    ms, comp, _ = bench_fn(fn, (), n_warmup=2, n_steps=10)
    print(f"V3 RASTER matmul(w, attr_8) M_TILES={M_TILES}: {ms:.2f} ms (compile {comp:.2f}s)", flush=True)


def test_frontend_and_topk(ds, spec):
  B = 1024
  args = _tile_inputs(ds, B)
  oxp, oxm, cxp, cxm = args
  H, W = spec.height, spec.width
  N = spec.n_gaussians

  # Per-body EWA projection (no (B, N, 3, 3) tensors!)
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
        idx,
    ))

  def ewa_per_body(obj_xpos, obj_xmat, cam_xpos, cam_xmat):
    l_pos = jnp.asarray(spec.light_pos, jnp.float32)
    l_amb = jnp.asarray(spec.light_ambient, jnp.float32)
    l_diff = jnp.asarray(spec.light_diffuse, jnp.float32)
    half_h = math.tan(0.5 * math.radians(spec.cam_fovy))
    fy = 0.5 * float(H) / half_h
    fx = fy

    # Compute R_co = cam_xmat^T @ obj_xmat (B, M, 3, 3) and t_co = cam_xmat^T @ (obj_xpos - cam_xpos) (B, M, 3)
    R_co = jnp.einsum("bji,bmjk->bmik", cam_xmat, obj_xmat, precision=jax.lax.Precision.HIGHEST)
    t_co = jnp.einsum("bji,bmj->bmi", cam_xmat, obj_xpos - cam_xpos[:, None, :], precision=jax.lax.Precision.HIGHEST)

    # Pre-order or concat outputs per body (note: body_id is contiguous in mesh_scene_to_rigid_3dgs!)
    out_mx, out_my, out_icxx, out_icxy, out_icyy = [], [], [], [], []
    out_rgb, out_zd, out_nrm, out_vg, out_rad, out_op = [], [], [], [], [], []
    for m, mu_l, cov_l, nrm_l, rgb_l, op_l, _ in body_slices:
      Rm = obj_xmat[:, m]  # (B, 3, 3)
      pm = obj_xpos[:, m]  # (B, 3)
      Rcm = R_co[:, m]     # (B, 3, 3)
      tcm = t_co[:, m]     # (B, 3)

      # Local-space light & camera positions (B, 3) -> avoids rotating mu_w and nrm_w per Gaussian for shading!
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
      # Direct cov_cam via Rcm (B, 3, 3) and cov_l (N_m, 3, 3)
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

      out_mx.append(mean_x); out_my.append(mean_y)
      out_icxx.append(icxx); out_icxy.append(icxy); out_icyy.append(icyy)
      out_rgb.append(rgb_lit); out_zd.append(z_d); out_nrm.append(nrm_vis)
      out_vg.append(valid_g); out_rad.append(radius)

    return (
        jnp.concatenate(out_mx, axis=1), jnp.concatenate(out_my, axis=1),
        jnp.concatenate(out_icxx, axis=1), jnp.concatenate(out_icxy, axis=1), jnp.concatenate(out_icyy, axis=1),
        jnp.concatenate(out_rgb, axis=1), jnp.concatenate(out_zd, axis=1), jnp.concatenate(out_nrm, axis=1),
        jnp.concatenate(out_vg, axis=1), jnp.concatenate(out_rad, axis=1),
    )

  fn_ewa_pb = jax.jit(ewa_per_body)
  ms_pb, _, pb_out = bench_fn(fn_ewa_pb, args, n_warmup=2, n_steps=10)
  ref_out = jax.jit(lambda a, b, c, d: stage_frontend_and_binning(spec, a, b, c, d))(*args)
  diff_mx = float(jnp.max(jnp.abs(pb_out[0][:8] - ref_out[0][:8])))
  print(f"EWA per-body (B={B}, N={N}): {ms_pb:.2f} ms (vs V2 18.57 ms), max_mx_diff={diff_mx:.2e}", flush=True)

  # Test BF16 vs FP32 top_k
  mean_x, mean_y, _, _, _, _, z_d, _, valid_g, radius = pb_out
  Ty = Tx = 8
  t_ox = np.tile(np.arange(Tx, dtype=np.float32) * 8, Ty)
  t_oy = np.repeat(np.arange(Ty, dtype=np.float32) * 8, Tx)
  tox_d = jnp.asarray(t_ox, jnp.float32)[None, :, None]
  toy_d = jnp.asarray(t_oy, jnp.float32)[None, :, None]

  def topk_bf16(mx, my, zd, vg, rad):
    nx = jnp.clip(mx[:, None, :], tox_d, tox_d + 8.0)
    ny = jnp.clip(my[:, None, :], toy_d, toy_d + 8.0)
    dx, dy = mx[:, None, :] - nx, my[:, None, :] - ny
    ov = vg[:, None, :] & ((dx * dx + dy * dy) <= (rad * rad)[:, None, :])
    sc = jnp.where(ov, (-zd[:, None, :]).astype(jnp.bfloat16), jnp.bfloat16(-1.0e9))
    return jax.lax.top_k(sc, 128)

  ms_tk_bf16, _, _ = bench_fn(jax.jit(topk_bf16), (mean_x, mean_y, z_d, valid_g, radius), n_warmup=2, n_steps=10)
  print(f"TOPK_ONLY BF16 (ts=8, k=128): {ms_tk_bf16:.2f} ms (vs FP32 23.69 ms)", flush=True)


def main():
  ds, spec = load_spec(2048, 64, 64)
  test_take_128_chunks_vs_mxu_3part()
  test_raster_compile_and_speed()
  test_frontend_and_topk(ds, spec)


if __name__ == "__main__":
  main()
