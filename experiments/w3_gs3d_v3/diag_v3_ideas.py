"""Test V3 optimizations on live TPU v5p:
1. Packed gather + jax_tile vs pallas_v2 E2E
2. Tighter tile-Gaussian overlap culling (AABB + ellipse) & top_k parity
3. Pallas V3 multi-tile packed rasterizer (M_tiles packing, zero output bloat, BF16 MXU U)
4. Pallas VMEM gather options (indirect indexing vs MXU hi+lo one-hot chunked gather)
"""

from __future__ import annotations

import math
import time
import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
import numpy as np

from tpu_mjx_render.gs3d import _pallas_gs3d_rasterize, mesh_scene_to_rigid_3dgs, render_gs3d_batch
from profile_v2_gs3d import load_spec, _tile_inputs, bench_fn, stage_frontend_and_binning


def test_tight_culling_and_packed_gather(ds, spec):
  B = 1024
  args = _tile_inputs(ds, B)
  ref_rgb, ref_dep, _, ref_alp = jax.jit(
      lambda a, b, c, d: render_gs3d_batch(spec, a, b, c, d, tile_size=8, top_k=128, mode="jax_tile")
  )(*args)

  # Check tighter bounding box vs circle overlap on B=32
  oxp, oxm, cxp, cxm = [x[:32] for x in args]
  mean_x, mean_y, icxx, icxy, icyy, rgb_lit, op_b, z_d, nrm_vis, valid_g, radius = stage_frontend_and_binning(
      spec, oxp, oxm, cxp, cxm
  )
  # Compute cov2_xx and cov2_yy from icxx, icxy, icyy:
  # det_inv = icxx * icyy - icxy^2, cov2_xx = icyy / det_inv, cov2_yy = icxx / det_inv
  det_i = icxx * icyy - icxy * icxy
  safe_di = jnp.where(valid_g & (det_i > 1e-12), det_i, 1.0)
  cov2_xx = jnp.where(valid_g, icyy / safe_di, 0.0)
  cov2_yy = jnp.where(valid_g, icxx / safe_di, 0.0)
  hx = 3.0 * jnp.sqrt(jnp.maximum(cov2_xx, 0.1))
  hy = 3.0 * jnp.sqrt(jnp.maximum(cov2_yy, 0.1))

  Ty = Tx = 8
  t_ox = np.tile(np.arange(Tx, dtype=np.float32) * 8, Ty)
  t_oy = np.repeat(np.arange(Ty, dtype=np.float32) * 8, Tx)
  tox_d = jnp.asarray(t_ox, jnp.float32)[None, :, None]
  toy_d = jnp.asarray(t_oy, jnp.float32)[None, :, None]

  # Method 1: Axis-aligned 3-sigma box overlap with tile
  aabb_ov = (
      valid_g[:, None, :]
      & (mean_x[:, None, :] + hx[:, None, :] >= tox_d)
      & (mean_x[:, None, :] - hx[:, None, :] <= tox_d + 8.0)
      & (mean_y[:, None, :] + hy[:, None, :] >= toy_d)
      & (mean_y[:, None, :] - hy[:, None, :] <= toy_d + 8.0)
  )
  cnt_aabb = jnp.sum(aabb_ov, axis=-1)

  # Method 2: AABB + clamped-point Mahalanobis <= 16.0 (exact power > -8.0, or <= 32.0 for power > -16.0)
  near_x = jnp.clip(mean_x[:, None, :], tox_d, tox_d + 8.0)
  near_y = jnp.clip(mean_y[:, None, :], toy_d, toy_d + 8.0)
  dx_n = near_x - mean_x[:, None, :]
  dy_n = near_y - mean_y[:, None, :]
  q_near = icxx[:, None, :] * dx_n * dx_n + 2.0 * icxy[:, None, :] * dx_n * dy_n + icyy[:, None, :] * dy_n * dy_n
  # Note: for rotated ellipse, min Q over tile can be < q_near, so check q_near <= 32.0 as conservative or test exact
  # Exact minimum of Q(x,y) over box [x0,x1]x[y0,y1]: either at clamped x/y along unconstrained y/x or corners!
  # Along x = x0 or x1: optimal y is my - (icxy/icyy)*(x - mx) clamped to [y0, y1]
  inv_icyy = jnp.where(icyy > 1e-12, 1.0 / jnp.maximum(icyy, 1e-12), 0.0)
  inv_icxx = jnp.where(icxx > 1e-12, 1.0 / jnp.maximum(icxx, 1e-12), 0.0)
  def _q(px, py):
    dx = px - mean_x[:, None, :]
    dy = py - mean_y[:, None, :]
    return icxx[:, None, :] * dx * dx + 2.0 * icxy[:, None, :] * dx * dy + icyy[:, None, :] * dy * dy

  # Check if center is inside tile, else min over 4 edges
  in_x = (mean_x[:, None, :] >= tox_d) & (mean_x[:, None, :] <= tox_d + 8.0)
  in_y = (mean_y[:, None, :] >= toy_d) & (mean_y[:, None, :] <= toy_d + 8.0)
  y_at_x0 = jnp.clip(mean_y[:, None, :] - (icxy * inv_icyy)[:, None, :] * (tox_d - mean_x[:, None, :]), toy_d, toy_d + 8.0)
  y_at_x1 = jnp.clip(mean_y[:, None, :] - (icxy * inv_icyy)[:, None, :] * (tox_d + 8.0 - mean_x[:, None, :]), toy_d, toy_d + 8.0)
  x_at_y0 = jnp.clip(mean_x[:, None, :] - (icxy * inv_icxx)[:, None, :] * (toy_d - mean_y[:, None, :]), tox_d, tox_d + 8.0)
  x_at_y1 = jnp.clip(mean_x[:, None, :] - (icxy * inv_icxx)[:, None, :] * (toy_d + 8.0 - mean_y[:, None, :]), tox_d, tox_d + 8.0)
  q_min = jnp.where(
      in_x & in_y,
      0.0,
      jnp.minimum(
          jnp.minimum(_q(tox_d, y_at_x0), _q(tox_d + 8.0, y_at_x1)),
          jnp.minimum(_q(x_at_y0, toy_d), _q(x_at_y1, toy_d + 8.0)),
      ),
  )
  # alpha >= 1/255 iff -0.5 * q >= -16 and op*exp(-0.5*q) >= 1/255 -> q <= 2 * ln(255 * 0.92) = 10.92!
  exact_ov = aabb_ov & (q_min <= 11.2)
  cnt_exact = jnp.sum(exact_ov, axis=-1)
  print(f"AABB overlap/tile: min={int(jnp.min(cnt_aabb))}, med={float(jnp.median(cnt_aabb)):.1f}, p95={float(jnp.percentile(cnt_aabb, 95)):.1f}, max={int(jnp.max(cnt_aabb))}", flush=True)
  print(f"EXACT_ELLIPSE overlap/tile: min={int(jnp.min(cnt_exact))}, med={float(jnp.median(cnt_exact)):.1f}, p95={float(jnp.percentile(cnt_exact, 95)):.1f}, max={int(jnp.max(cnt_exact))}", flush=True)


def test_pallas_v3_rasterizer():
  """Test M_tiles=16 packed Pallas rasterizer with zero output padding bloat + BF16 MXU."""
  B, T, K = 1024, 64, 128
  BT = B * T
  P_t = 64
  M_TILES = 16  # 16 tiles packed -> output is (64, 128) = 16 tiles * 8 channels = ZERO PADDING BLOAT!
  assert BT % M_TILES == 0
  n_macro = BT // M_TILES

  key = jax.random.PRNGKey(0)
  splats = jax.random.uniform(key, (BT, 16, K), jnp.float32)
  # Make icxx, icyy, op sane
  splats = splats.at[:, 2, :].set(0.5)
  splats = splats.at[:, 3, :].set(0.0)
  splats = splats.at[:, 4, :].set(0.5)
  splats = splats.at[:, 8, :].set(0.2)
  # Reshape splats to (n_macro * M_TILES * 16, K) -> each grid step loads (M_TILES * 16, K) = (256, 128)
  splats_2d = splats.reshape(n_macro * M_TILES * 16, K)

  lx = np.tile(np.arange(8, dtype=np.float32), 8) + 0.5
  ly = np.repeat(np.arange(8, dtype=np.float32), 8) + 0.5
  loc_xy_dev = jnp.asarray(
      np.stack([np.broadcast_to(lx[:, None], (P_t, K)), np.broadcast_to(ly[:, None], (P_t, K))], axis=0).reshape(2 * P_t, K),
      jnp.float32,
  )
  u_128_bf16 = jnp.asarray(np.triu(np.ones((K, K), dtype=np.float32), k=1), jnp.bfloat16)
  u_128_fp32 = jnp.asarray(np.triu(np.ones((K, K), dtype=np.float32), k=1), jnp.float32)

  def run_v3_raster(use_bf16_mxu=True):
    u_dev = u_128_bf16 if use_bf16_mxu else u_128_fp32
    prec = jax.lax.Precision.DEFAULT if use_bf16_mxu else jax.lax.Precision.HIGHEST

    def _k(spl_ref, lxy_ref, u_ref, out_ref):
      sf_all = spl_ref[...]   # (M_TILES * 16, 128)
      lxy = lxy_ref[...]      # (128, 128)
      lx_m = lxy[0:64, :]
      ly_m = lxy[64:128, :]
      u_m = u_ref[...]        # (128, 128)
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
        if use_bf16_mxu:
          log_tr = jnp.matmul(log_1ma.astype(jnp.bfloat16), u_m, preferred_element_type=jnp.float32)
        else:
          log_tr = jnp.matmul(log_1ma, u_m, precision=prec)
        trans = jnp.exp(jnp.maximum(log_tr, -30.0))
        w = alpha * trans  # (64, 128)

        # Notice: attr is (8, 128) -> transpose to (128, 8) and accumulate w @ attr!
        # Wait: instead of (128, 128) matmul for 8 channels, can we just sum(w * attr) or matmul?
        # Let's accumulate 8 channels directly via jnp.sum or matmul!
        attr_8 = jnp.concatenate([sf[5:8, :], sf[9:13, :], ones_k], axis=0)  # (8, 128)
        # w is (64, 128), attr_8^T is (128, 8): 8 VPU reductions OR 1 packed matmul!
        out_8 = jnp.stack([jnp.sum(w * jnp.broadcast_to(attr_8[c : c + 1, :], (P_t, K)), axis=1) for c in range(8)], axis=1)
        outs.append(out_8)

      out_ref[...] = jnp.concatenate(outs, axis=1)  # (64, 128) -> ZERO PADDING!

    return pl.pallas_call(
        _k,
        out_shape=jax.ShapeDtypeStruct((n_macro * P_t, 128), jnp.float32),
        grid=(n_macro,),
        in_specs=[
            pl.BlockSpec((M_TILES * 16, K), lambda i: (i, 0)),
            pl.BlockSpec((2 * P_t, K), lambda i: (0, 0)),
            pl.BlockSpec((K, K), lambda i: (0, 0)),
        ],
        out_specs=pl.BlockSpec((P_t, 128), lambda i: (i, 0)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=False,
    )(splats_2d, loc_xy_dev, u_dev)

  for use_bf16 in [False, True]:
    fn = jax.jit(lambda: run_v3_raster(use_bf16))
    ms, comp_s, _ = bench_fn(fn, (), n_warmup=2, n_steps=10)
    print(f"V3 RASTER_ONLY (M_TILES=16, use_bf16_mxu={use_bf16}): {ms:.2f} ms (compile {comp_s:.2f}s)", flush=True)


def test_pallas_gather_options():
  """Test whether Pallas VMEM supports indirect gather or MXU one-hot chunked gather."""
  B, T, K, N = 1024, 64, 128, 2048
  key = jax.random.PRNGKey(1)
  splats_bn = jax.random.uniform(key, (B, 16, N), jnp.float32)  # (B, 16, N)
  idx_btk = jax.random.randint(key, (B, T, K), 0, N, jnp.int32)

  # Test A: Pallas indirect VMEM gather via jnp.take_along_axis or indexing?
  try:
    def _k_take(s_ref, i_ref, o_ref):
      s = s_ref[...]  # (16, N)
      ix = i_ref[...]  # (T, K) -> (64, 128)
      # s[:, ix] -> (16, 64, 128)
      o_ref[...] = s[:, ix].reshape(16 * T, K)

    fn_take = jax.jit(lambda s, ix: pl.pallas_call(
        _k_take,
        out_shape=jax.ShapeDtypeStruct((B * 16 * T, K), jnp.float32),
        grid=(B,),
        in_specs=[pl.BlockSpec((16, N), lambda b: (b, 0)), pl.BlockSpec((T, K), lambda b: (b, 0))],
        out_specs=pl.BlockSpec((16 * T, K), lambda b: (b, 0)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=False,
    )(s.reshape(B * 16, N), ix.reshape(B * T, K)))
    ms, comp, _ = bench_fn(fn_take, (splats_bn, idx_btk), n_warmup=1, n_steps=5)
    print(f"PALLAS VMEM s[:, ix] gather: {ms:.2f} ms (compile {comp:.2f}s)", flush=True)
  except Exception as e:
    print(f"PALLAS VMEM s[:, ix] FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)

  # Test B: Pallas MXU hi+lo one-hot chunked gather!
  # Reshape splats_bn (B, 16, N) -> (B, N, 16) -> split into 16 chunks of (128, 16)
  splats_bn16 = jnp.transpose(splats_bn, (0, 2, 1)).reshape(B * N, 16)
  # Pad 16 cols to 128 or keep 16 cols? Note: TPU BlockSpec minor dim should be 128-aligned for best DMA,
  # OR (N, 128)? Wait: for (B*N, 16), let's test (B*N, 128) or (B*16, N) transposed inside kernel!
  # Actually (B, 16, N) has minor dim N=2048 (128-aligned!!)! Inside kernel we transpose (16, 128) -> (128, 16)!
  try:
    N_CHUNKS = N // 128  # 16
    iota_128 = jnp.arange(128, dtype=jnp.int32)[None, :]  # (1, 128)
    # Wait: (T*K, 128) = (8192, 128) per world -> split T=64 into 8 groups of 8 tiles -> (8*128, 128) = (1024, 128)!
    # Even simpler: grid=(B, T_GROUPS) where T_GROUPS=8, each step processes 8 tiles = (1024, 128) indices!
    T_GRP = 8
    n_tg = T // T_GRP  # 8
    def _k_mxu_gather(s_ref, i_ref, o_ref):
      s_all = s_ref[...]  # (16, 2048)
      ix = i_ref[...]     # (1024, 128) -> 8 tiles * 128 splats, wait: we want (1024, 16) output!
      # Actually ix is (T_GRP * K, 128)? No, ix is (T_GRP, K) = (8, 128)!
      # Flatten ix to (1024, 1)!!
      ix_col = ix.reshape(T_GRP * K, 1)  # (1024, 1)
      # Wait: BlockSpec for out_ref of (1024, 128) (16 cols used or 8 tiles x 16 = 128?!)
      # Let's accumulate acc_hi, acc_lo of shape (1024, 128) -> wait we only need 16 cols -> (1024, 16) -> pad to 128?
      # Wait: (T_GRP*K, 16) = (1024, 16) -> reshape or pack 8 tiles!
      iota = jnp.broadcast_to(jnp.arange(128, dtype=jnp.int32)[None, :], (T_GRP * K, 128))
      acc_hi = jnp.zeros((T_GRP * K, 128), jnp.float32)
      acc_lo = jnp.zeros((T_GRP * K, 128), jnp.float32)
      for c in range(N_CHUNKS):
        base = c * 128
        mask = (ix_col >= base) & (ix_col < base + 128)
        # one_hot: (1024, 128)
        oh = (ix_col == (iota + base)).astype(jnp.bfloat16)
        chunk_t = jnp.transpose(s_all[:, base : base + 128], (1, 0))  # (128, 16)
        chunk_128 = jnp.pad(chunk_t, ((0, 0), (0, 112)))  # (128, 128)
        c_hi = chunk_128.astype(jnp.bfloat16)
        c_lo = (chunk_128 - c_hi.astype(jnp.float32)).astype(jnp.bfloat16)
        acc_hi = acc_hi + jnp.matmul(oh, c_hi, preferred_element_type=jnp.float32)
        acc_lo = acc_lo + jnp.matmul(oh, c_lo, preferred_element_type=jnp.float32)
      o_ref[...] = acc_hi + acc_lo  # (1024, 128)

    fn_mxu = jax.jit(lambda s, ix: pl.pallas_call(
        _k_mxu_gather,
        out_shape=jax.ShapeDtypeStruct((B * T, 128), jnp.float32), # Wait: T_GRP*K = 1024 rows per step, total B*n_tg*1024 = B*T*K rows!
        grid=(B, n_tg),
        in_specs=[
            pl.BlockSpec((16, N), lambda b, tg: (b, 0)),
            pl.BlockSpec((T_GRP, K), lambda b, tg: (b * n_tg + tg, 0)),
        ],
        out_specs=pl.BlockSpec((T_GRP * K, 128), lambda b, tg: (b * n_tg + tg, 0)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
        interpret=False,
    )(splats_bn.reshape(B * 16, N), idx_btk.reshape(B * T, K)))
    # Fix out_shape to (B * T * K, 128):
    fn_mxu = jax.jit(lambda s, ix: pl.pallas_call(
        _k_mxu_gather,
        out_shape=jax.ShapeDtypeStruct((B * T * K, 128), jnp.float32),
        grid=(B, n_tg),
        in_specs=[
            pl.BlockSpec((16, N), lambda b, tg: (b, 0)),
            pl.BlockSpec((T_GRP, K), lambda b, tg: (b * n_tg + tg, 0)),
        ],
        out_specs=pl.BlockSpec((T_GRP * K, 128), lambda b, tg: (b * n_tg + tg, 0)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
        interpret=False,
    )(splats_bn.reshape(B * 16, N), idx_btk.reshape(B * T, K)))
    ms, comp, out_mxu = bench_fn(fn_mxu, (splats_bn, idx_btk), n_warmup=1, n_steps=5)
    # Verify exact parity vs XLA gather!
    ref = jnp.transpose(splats_bn, (0, 2, 1))[jnp.arange(B)[:, None, None], idx_btk]
    got = out_mxu.reshape(B, T, K, 128)[..., :16]
    diff = float(jnp.max(jnp.abs(ref[:4] - got[:4])))
    print(f"PALLAS MXU hi+lo chunked gather: {ms:.2f} ms (compile {comp:.2f}s), max_diff={diff:.2e}", flush=True)
  except Exception as e:
    print(f"PALLAS MXU gather FAILED: {type(e).__name__}: {str(e)[:300]}", flush=True)


def main():
  ds, spec = load_spec(2048, 64, 64)
  test_tight_culling_and_packed_gather(ds, spec)
  test_pallas_v3_rasterizer()
  test_pallas_gather_options()


if __name__ == "__main__":
  main()
