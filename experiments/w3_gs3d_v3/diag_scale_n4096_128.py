"""Benchmark V3 Fused on N=4096, 128x128, and test BF16 approx_max_k + fori_loop over N chunks."""

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


def build_body_slices(spec):
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
  return body_slices


def main():
  B = 1024
  # 1. Test BF16 approx_max_k vs FP32 approx_max_k at N=2048, 64x64
  ds64, spec2k_64 = load_spec(2048, 64, 64)
  bs2k = build_body_slices(spec2k_64)
  args = _tile_inputs(ds64, B)

  _, mean_x, mean_y, z_d, valid_g, radius = jax.jit(
      lambda a, b, c, d: ewa_per_body_exact(spec2k_64, a, b, c, d, bs2k)
  )(*args)
  t_ox = np.tile(np.arange(8, dtype=np.float32) * 8, 8)
  t_oy = np.repeat(np.arange(8, dtype=np.float32) * 8, 8)
  tox_d = jnp.asarray(t_ox, jnp.float32)[None, :, None]
  toy_d = jnp.asarray(t_oy, jnp.float32)[None, :, None]

  def ak_bf16(mx, my, zd, vg, rad):
    nx = jnp.clip(mx[:, None, :], tox_d, tox_d + 8.0)
    ny = jnp.clip(my[:, None, :], toy_d, toy_d + 8.0)
    dx, dy = mx[:, None, :] - nx, my[:, None, :] - ny
    ov = vg[:, None, :] & ((dx * dx + dy * dy) <= (rad * rad)[:, None, :])
    sc = jnp.where(ov, (-zd[:, None, :]).astype(jnp.bfloat16), jnp.bfloat16(-1.0e9))
    return jax.lax.approx_max_k(sc, 128, recall_target=1.0)

  ms_ak_bf16, _, _ = bench_fn(jax.jit(ak_bf16), (mean_x, mean_y, z_d, valid_g, radius), n_warmup=2, n_steps=10)
  print(f"APPROX_MAX_K_128 BF16 (64x64, N=2048): {ms_ak_bf16:.2f} ms (vs FP32 17.28 ms)", flush=True)

  # 2. Test N=4096 @ 64x64, N=2048 @ 128x128, N=4096 @ 128x128
  for N_g, res in [(4096, 64), (2048, 128), (4096, 128)]:
    _, spec_cfg = load_spec(N_g, res, res)
    bs_cfg = build_body_slices(spec_cfg)
    fn_r = jax.jit(lambda a, b, c, d: render_gs3d_batch(spec_cfg, a, b, c, d, tile_size=8, top_k=128, mode="jax_tile"))
    ms_r, _, (rr, rd, _, _) = bench_fn(fn_r, args, n_warmup=1, n_steps=3)
    print(f"CFG N={N_g}, {res}x{res} | jax_tile: {ms_r:.2f} ms ({B*1e3/ms_r:.1f} FPS)", flush=True)

    for use_bf16 in [False, True]:
      fn_v = jax.jit(lambda a, b, c, d: render_v3_with_options(spec_cfg, a, b, c, d, bs_cfg, use_approx=True, use_bf16_mxu=use_bf16, t_grp=16))
      ms_v, comp_v, (vr, vd, _, _) = bench_fn(fn_v, args, n_warmup=2, n_steps=5)
      mx_d = float(jnp.max(jnp.abs(rr[:16] - vr[:16])))
      dep_mae = float(jnp.mean(jnp.abs(rd[:16] - vd[:16])))
      mse_v = float(jnp.mean((rr[:16] - vr[:16]) ** 2))
      psnr_v = -10.0 * math.log10(max(mse_v, 1e-12)) if mse_v > 0 else 999.0
      print(
          f"CFG N={N_g}, {res}x{res} | V3 (bf16={use_bf16}, t_grp=16): {ms_v:.2f} ms ({B*1e3/ms_v:.1f} FPS, {ms_r/ms_v:.2f}x), "
          f"comp={comp_v:.2f}s | max_rgb={mx_d:.2e}, dep_mae={dep_mae:.2e}, psnr={psnr_v:.2f} dB",
          flush=True,
      )


if __name__ == "__main__":
  main()
