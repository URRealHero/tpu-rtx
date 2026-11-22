"""Isolate overlap vs top_k vs approx_max_k, t_grp=16, and FP32 parity source on w3."""

from __future__ import annotations

import math
import time
import jax
import jax.numpy as jnp
import numpy as np

from profile_v2_gs3d import load_spec, _tile_inputs, bench_fn, stage_frontend_and_binning
from test_fused_v3_e2e import ewa_per_body_packed16, pallas_v3_fused_rasterize, render_gs3d_v3
from tpu_mjx_render.gs3d import render_gs3d_batch


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
  H, W = 64, 64
  tile_size = 8
  Ty = Tx = 8
  T = 64
  t_ox = np.tile(np.arange(Tx, dtype=np.float32) * tile_size, Ty)
  t_oy = np.repeat(np.arange(Ty, dtype=np.float32) * tile_size, Tx)
  tox_dev = jnp.asarray(t_ox, jnp.float32)[None, :, None]
  toy_dev = jnp.asarray(t_oy, jnp.float32)[None, :, None]

  splats_b16n, mean_x, mean_y, z_d, valid_g, radius = jax.jit(
      lambda a, b, c, d: ewa_per_body_packed16(spec, a, b, c, d, body_slices)
  )(*args)

  # 1. Isolate overlap only vs top_k only vs approx_max_k
  def overlap_only(mx, my, zd, vg, rad):
    nx = jnp.clip(mx[:, None, :], tox_dev, tox_dev + 8.0)
    ny = jnp.clip(my[:, None, :], toy_dev, toy_dev + 8.0)
    dx, dy = mx[:, None, :] - nx, my[:, None, :] - ny
    ov = vg[:, None, :] & ((dx * dx + dy * dy) <= (rad * rad)[:, None, :])
    return jnp.where(ov, -zd[:, None, :], -1.0e9)

  fn_ov = jax.jit(overlap_only)
  ms_ov, _, score_btn = bench_fn(fn_ov, (mean_x, mean_y, z_d, valid_g, radius), n_warmup=2, n_steps=10)
  print(f"OVERLAP_SCORE_ONLY (B=1024, T=64, N=2048): {ms_ov:.2f} ms", flush=True)

  ms_tk, _, _ = bench_fn(jax.jit(lambda sc: jax.lax.top_k(sc, 128)), (score_btn,), n_warmup=2, n_steps=10)
  print(f"TOPK_128_ONLY (from score): {ms_tk:.2f} ms", flush=True)

  for rec in [1.0, 0.99, 0.95]:
    try:
      ms_ak, _, (ak_s, ak_i) = bench_fn(
          jax.jit(lambda sc: jax.lax.approx_max_k(sc, 128, recall_target=rec)),
          (score_btn,),
          n_warmup=2,
          n_steps=10,
      )
      print(f"APPROX_MAX_K_128 (recall={rec}): {ms_ak:.2f} ms", flush=True)
    except Exception as e:
      print(f"APPROX_MAX_K failed: {e}", flush=True)

  # Test sort_key_val once on (B, N)
  ms_sort_bn, _, _ = bench_fn(
      jax.jit(lambda z: jax.lax.sort_key_val(z, jnp.broadcast_to(jnp.arange(2048, dtype=jnp.int32)[None, :], (B, 2048)), dimension=1)),
      (z_d,),
      n_warmup=2,
      n_steps=10,
  )
  print(f"SORT_KEY_VAL (B=1024, N=2048): {ms_sort_bn:.2f} ms", flush=True)

  # Check overlap using center-distance shortcut: |mx - cx| <= 4 + rad AND |my - cy| <= 4 + rad (faster than clip!)
  t_cx = jnp.asarray(t_ox + 4.0, jnp.float32)[None, :, None]
  t_cy = jnp.asarray(t_oy + 4.0, jnp.float32)[None, :, None]
  def overlap_fast_box(mx, my, zd, vg, rad):
    r_eff = rad + 4.0
    ov = vg[:, None, :] & (jnp.abs(mx[:, None, :] - t_cx) <= r_eff[:, None, :]) & (jnp.abs(my[:, None, :] - t_cy) <= r_eff[:, None, :])
    return jnp.where(ov, -zd[:, None, :], -1.0e9)

  ms_ovf, _, _ = bench_fn(jax.jit(overlap_fast_box), (mean_x, mean_y, z_d, valid_g, radius), n_warmup=2, n_steps=10)
  print(f"OVERLAP_FAST_BOX_ONLY: {ms_ovf:.2f} ms", flush=True)

  # 2. Check t_grp=16 E2E
  fn_v3_16 = jax.jit(lambda a, b, c, d: render_gs3d_v3(spec, a, b, c, d, body_slices, use_bf16_mxu=False, t_grp=16))
  ms_16, c16, _ = bench_fn(fn_v3_16, args, n_warmup=3, n_steps=10)
  print(f"V3 FUSED (bf16=False, t_grp=16): {ms_16:.2f} ms ({B*1e3/ms_16:.1f} FPS), compile={c16:.2f}s", flush=True)

  fn_v3_16b = jax.jit(lambda a, b, c, d: render_gs3d_v3(spec, a, b, c, d, body_slices, use_bf16_mxu=True, t_grp=16))
  ms_16b, c16b, _ = bench_fn(fn_v3_16b, args, n_warmup=3, n_steps=10)
  print(f"V3 FUSED (bf16=True,  t_grp=16): {ms_16b:.2f} ms ({B*1e3/ms_16b:.1f} FPS), compile={c16b:.2f}s", flush=True)

  # 3. Isolate parity difference: ewa_per_body vs V2 EWA, and pallas_v3 rasterizer vs jax_tile rasterizer
  ref_rgb, ref_dep, _, _ = jax.jit(lambda a, b, c, d: render_gs3d_batch(spec, a, b, c, d, tile_size=8, top_k=128, mode="jax_tile"))(*[x[:32] for x in args])
  v3_rgb, v3_dep, _, _ = jax.jit(lambda a, b, c, d: render_gs3d_v3(spec, a, b, c, d, body_slices, use_bf16_mxu=False, t_grp=8))(*[x[:32] for x in args])
  diff = np.asarray(jnp.abs(ref_rgb - v3_rgb))
  print(f"PARITY B=32: max_diff={diff.max():.6f}, p99.9={np.percentile(diff, 99.9):.6e}, frac>5e-4={(diff>5e-4).mean():.6f}", flush=True)

  # Check if using V2 exact EWA with V3 fused rasterizer gives max_diff < 5e-4!
  mean_x2, mean_y2, icxx2, icxy2, icyy2, rgb_lit2, op_b2, z_d2, nrm_vis2, valid_g2, radius2 = stage_frontend_and_binning(
      spec, *[x[:32] for x in args]
  )
  op_bn2 = jnp.broadcast_to(op_b2[None, :], (32, spec.n_gaussians))
  z32 = jnp.zeros((32, spec.n_gaussians), jnp.float32)
  splats_v2 = jnp.stack([
      mean_x2, mean_y2, icxx2, icxy2, icyy2,
      rgb_lit2[..., 0], rgb_lit2[..., 1], rgb_lit2[..., 2],
      op_bn2, z_d2, nrm_vis2[..., 0], nrm_vis2[..., 1], nrm_vis2[..., 2],
      z32, z32, z32,
  ], axis=1)
  nx2 = jnp.clip(mean_x2[:, None, :], tox_dev, tox_dev + 8.0)
  ny2 = jnp.clip(mean_y2[:, None, :], toy_dev, toy_dev + 8.0)
  dx2, dy2 = mean_x2[:, None, :] - nx2, mean_y2[:, None, :] - ny2
  ov2 = valid_g2[:, None, :] & ((dx2 * dx2 + dy2 * dy2) <= (radius2 * radius2)[:, None, :])
  sc2 = jnp.where(ov2, -z_d2[:, None, :], -1.0e9)
  ts2, ti2 = jax.lax.top_k(sc2, 128)
  ti2 = jnp.where(ts2 > -1.0e8, ti2, -1)
  rgb_t2, dep_t2, nrm_t2, alp_t2 = pallas_v3_fused_rasterize(splats_v2, ti2, t_ox, t_oy, tile_size=8, use_bf16_mxu=False, t_grp=8)
  rgb_img2 = rgb_t2.reshape(32, Ty, Tx, 8, 8, 3).transpose(0, 1, 3, 2, 4, 5).reshape(32, H, W, 3)
  alp_img2 = alp_t2.reshape(32, Ty, Tx, 8, 8).transpose(0, 1, 3, 2, 4).reshape(32, H, W)
  sky_v = jnp.linspace(1.0, 0.0, H, dtype=jnp.float32)[None, :, None, None]
  sky = sky_v * jnp.asarray(spec.sky_rgb1, jnp.float32)[None, None, None, :] + (1.0 - sky_v) * jnp.asarray(spec.sky_rgb2, jnp.float32)[None, None, None, :]
  rgb_final2 = jnp.clip(rgb_img2 + (1.0 - alp_img2[..., None]) * sky, 0.0, 1.0)
  diff2 = float(jnp.max(jnp.abs(ref_rgb - rgb_final2)))
  print(f"PARITY V2_EWA + V3_FUSED_RASTER: max_rgb_diff={diff2:.2e}", flush=True)


if __name__ == "__main__":
  main()
