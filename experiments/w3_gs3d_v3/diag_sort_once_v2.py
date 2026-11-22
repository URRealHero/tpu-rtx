"""Quick test: Sort-Once with (8, 128)->(128, 8) transpose + N=4096 + 128x128 + approx_max_k bf16."""

from __future__ import annotations

import math
import time
import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
import numpy as np

from profile_v2_gs3d import load_spec, _tile_inputs, bench_fn
from diag_sort_once_and_parity import ewa_per_body_exact, render_v3_with_options
from tpu_mjx_render.gs3d import render_gs3d_batch


def test_sort_once_fixed(spec, args, body_slices):
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
  t_grp = 8
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
  l_128_bf16 = jnp.asarray(np.triu(np.ones((128, 128), dtype=np.float32), k=0), jnp.bfloat16)

  def _sort_once_pipeline(obj_xpos, obj_xmat, cam_xpos, cam_xmat):
    splats_b16n, mean_x, mean_y, z_d, valid_g, radius = ewa_per_body_exact(
        spec, obj_xpos, obj_xmat, cam_xpos, cam_xmat, body_slices
    )
    sort_key = jnp.where(valid_g, z_d, 1.0e9)
    iota_n = jnp.broadcast_to(jnp.arange(N_pad, dtype=jnp.int32)[None, :], (B, N_pad))
    _, perm = jax.lax.sort_key_val(sort_key, iota_n, dimension=1)
    splats_b16n = splats_b16n.at[:, 13, :N].set(jnp.where(valid_g, radius, -1.0))
    splats_sorted = jnp.take_along_axis(splats_b16n, perm[:, None, :], axis=2)

    def _k_compact(s_ref, oxy_ref, lxy_ref, u_ref, lcum_ref, out_ref):
      s_all = s_ref[...]     # (16, N_pad)
      oxy = oxy_ref[...]     # (16, 128)
      lxy = lxy_ref[...]     # (128, 128)
      lx_m, ly_m = lxy[0:64, :], lxy[64:128, :]
      u_m = u_ref[...]
      lcum = lcum_ref[...]   # (128, 128) bf16
      ones_k = jnp.ones((1, K), jnp.float32)

      ox_b = jnp.broadcast_to(oxy[0:8, 0:1], (8, 128))
      oy_b = jnp.broadcast_to(oxy[8:16, 0:1], (8, 128))

      iota_128 = jnp.broadcast_to(jnp.arange(128, dtype=jnp.int32)[None, :], (128, 128))
      accs = [jnp.zeros((16, 128), jnp.float32) for _ in range(t_grp)]
      counts = jnp.zeros((8, 1), jnp.float32)

      for c in range(n_chunks):
        base = c * 128
        sc = s_all[:, base : base + 128]
        mx_c = jnp.broadcast_to(sc[0:1, :], (8, 128))
        my_c = jnp.broadcast_to(sc[1:2, :], (8, 128))
        rad_c = jnp.broadcast_to(sc[13:14, :], (8, 128))

        nx_c = jnp.clip(mx_c, ox_b, ox_b + 8.0)
        ny_c = jnp.clip(my_c, oy_b, oy_b + 8.0)
        dx_c = mx_c - nx_c
        dy_c = my_c - ny_c
        ov_8 = (rad_c > 0.0) & ((dx_c * dx_c + dy_c * dy_c) <= (rad_c * rad_c))  # (8, 128)

        csum_8 = jnp.matmul(ov_8.astype(jnp.bfloat16), lcum, preferred_element_type=jnp.float32)  # (8, 128)
        rank_8 = counts + csum_8 - 1.0
        counts = counts + csum_8[:, 127:128]

        # Transpose (8, 128) -> (128, 8) (both dims non-trivial!)
        rank_128_8 = jnp.transpose(rank_8, (1, 0))
        ov_128_8 = jnp.transpose(ov_8.astype(jnp.float32), (1, 0))

        s_hi = sc.astype(jnp.bfloat16)
        r1 = sc - s_hi.astype(jnp.float32)
        s_mid = r1.astype(jnp.bfloat16)
        s_lo = (r1 - s_mid.astype(jnp.float32)).astype(jnp.bfloat16)

        for t in range(t_grp):
          r_t = jnp.broadcast_to(rank_128_8[:, t : t + 1], (128, 128))
          ov_t = jnp.broadcast_to(ov_128_8[:, t : t + 1], (128, 128)) > 0.5
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
  ref_rgb, _, _, _ = jax.jit(lambda a, b, c, d: render_gs3d_batch(spec, a, b, c, d, tile_size=8, top_k=128, mode="jax_tile"))(*[x[:32] for x in args])

  try:
    rgb_so = test_sort_once_fixed(spec, args, body_slices)
    max_so = float(jnp.max(jnp.abs(ref_rgb - rgb_so[:32])))
    mse_so = float(jnp.mean((ref_rgb - rgb_so[:32]) ** 2))
    psnr_so = -10.0 * math.log10(max(mse_so, 1e-12))
    print(f"SORT_ONCE PARITY: max_rgb={max_so:.2e}, psnr={psnr_so:.2f} dB", flush=True)
  except Exception as e:
    print(f"SORT_ONCE FAILED: {type(e).__name__}: {str(e)[:400]}", flush=True)

  # Also check N=4096 at 64x64 and N=2048 at 128x128!
  for N_g, res in [(4096, 64), (2048, 128)]:
    _, spec_cfg = load_spec(N_g, res, res)
    bs_cfg = []
    for m in range(spec_cfg.n_bodies):
      idx = np.where(spec_cfg.body_id == m)[0]
      bs_cfg.append((
          m,
          jnp.asarray(spec_cfg.mu_local[idx], jnp.float32),
          jnp.asarray(spec_cfg.cov_local[idx], jnp.float32),
          jnp.asarray(spec_cfg.nrm_local[idx], jnp.float32),
          jnp.asarray(spec_cfg.rgb_base[idx], jnp.float32),
          jnp.asarray(spec_cfg.opacity[idx], jnp.float32),
      ))
    fn_r = jax.jit(lambda a, b, c, d: render_gs3d_batch(spec_cfg, a, b, c, d, tile_size=8, top_k=128, mode="jax_tile"))
    ms_r, _, (rr, _, _, _) = bench_fn(fn_r, args, n_warmup=1, n_steps=3)
    print(f"CFG N={N_g}, {res}x{res} | jax_tile: {ms_r:.2f} ms ({B*1e3/ms_r:.1f} FPS)", flush=True)

    for use_approx, use_bf16 in [(True, False), (True, True)]:
      fn_v = jax.jit(lambda a, b, c, d: render_v3_with_options(spec_cfg, a, b, c, d, bs_cfg, use_approx=use_approx, use_bf16_mxu=use_bf16, t_grp=16))
      ms_v, comp_v, (vr, _, _, _) = bench_fn(fn_v, args, n_warmup=2, n_steps=5)
      mx_d = float(jnp.max(jnp.abs(rr[:16] - vr[:16])))
      mse_v = float(jnp.mean((rr[:16] - vr[:16]) ** 2))
      psnr_v = -10.0 * math.log10(max(mse_v, 1e-12)) if mse_v > 0 else 999.0
      print(f"CFG N={N_g}, {res}x{res} | V3 (approx={use_approx}, bf16={use_bf16}): {ms_v:.2f} ms ({B*1e3/ms_v:.1f} FPS, {ms_r/ms_v:.2f}x), comp={comp_v:.2f}s | max_rgb={mx_d:.2e}, psnr={psnr_v:.2f} dB", flush=True)


if __name__ == "__main__":
  main()
