"""Test fast gather options in Pallas VMEM + XLA on live TPU v5p."""

from __future__ import annotations

import time
import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
import numpy as np


def bench_fn(fn, args, n_warmup=2, n_steps=5):
  t_c0 = time.perf_counter()
  out = fn(*args)
  jax.block_until_ready(out)
  compile_s = time.perf_counter() - t_c0
  for _ in range(n_warmup):
    out = fn(*args)
  jax.block_until_ready(out)
  t0 = time.perf_counter()
  for _ in range(n_steps):
    out = fn(*args)
  jax.block_until_ready(out)
  ms = (time.perf_counter() - t0) * 1e3 / n_steps
  return ms, compile_s, out


def main():
  B, T, K, N = 1024, 64, 128, 2048
  key = jax.random.PRNGKey(1)
  splats_b16n = jax.random.uniform(key, (B, 16, N), jnp.float32)  # (B, 16, N)
  idx_btk = jax.random.randint(key, (B, T, K), 0, N, jnp.int32)

  # Reference output: (B, T, K, 16)
  ref = jnp.transpose(splats_b16n, (0, 2, 1))[jnp.arange(B)[:, None, None], idx_btk]

  # Test 1: Pallas take_along_axis in VMEM!
  # s: (16, N) or (8, N)? Note: Mosaic requires sublane/lane >= (8, 128) for take_along_axis? Let's test!
  try:
    def _k_take_along(s_ref, i_ref, o_ref):
      s = s_ref[...]  # (16, N)
      ix = i_ref[...]  # (T, K) = (64, 128)
      # Gather tile-by-tile or all T*K?
      # Wait: take_along_axis needs same ndim: (16, N) and (16, K)
      outs = []
      for t in range(8):
        # Test 8 tiles first or loop? Let's test single tile or 8 tiles in fori_loop
        pass
      ix0 = jnp.broadcast_to(ix[0:1, :], (16, K))
      g0 = jnp.take_along_axis(s, ix0, axis=1)  # (16, 128)
      o_ref[...] = g0

    fn1 = jax.jit(lambda s, ix: pl.pallas_call(
        _k_take_along,
        out_shape=jax.ShapeDtypeStruct((B * 16, K), jnp.float32),
        grid=(B,),
        in_specs=[pl.BlockSpec((16, N), lambda b: (b, 0)), pl.BlockSpec((T, K), lambda b: (b, 0))],
        out_specs=pl.BlockSpec((16, K), lambda b: (b, 0)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=False,
    )(s.reshape(B * 16, N), ix.reshape(B * T, K)))
    ms, comp, out1 = bench_fn(fn1, (splats_b16n, idx_btk), n_warmup=1, n_steps=3)
    diff = float(jnp.max(jnp.abs(jnp.transpose(out1.reshape(B, 16, K), (0, 2, 1)) - ref[:, 0])))
    print(f"TEST1 take_along_axis (1 tile): {ms:.2f} ms, diff={diff:.2e}", flush=True)
  except Exception as e:
    print(f"TEST1 FAILED: {type(e).__name__}: {str(e)[:400]}", flush=True)

  # Test 2: Pallas take_along_axis over all T=64 tiles with lax.fori_loop
  try:
    def _k_take_loop(s_ref, i_ref, o_ref):
      s = s_ref[...]  # (16, N)
      ix = i_ref[...]  # (64, 128)

      def _body(t, _):
        row = jax.lax.dynamic_slice(ix, (t, 0), (1, K))
        ix16 = jnp.broadcast_to(row, (16, K))
        g = jnp.take_along_axis(s, ix16, axis=1)  # (16, 128)
        o_ref[t * 16 : (t + 1) * 16, :] = g
        return None
      # Wait: o_ref dynamic slice in Pallas requires pl.ds!
    print("Skipping dynamic o_ref without pl.ds; testing full (16*T) or take_along_axis on (N, 128)", flush=True)
  except Exception as e:
    pass

  # Test 2b: Pallas take_along_axis on s=(N, 128) padded or s=(16, N) with static loop of T_GRP=8 tiles
  try:
    T_GRP = 8
    n_tg = T // T_GRP
    def _k_take_8(s_ref, i_ref, o_ref):
      s = s_ref[...]  # (16, N)
      ix = i_ref[...]  # (8, 128)
      outs = []
      for t in range(T_GRP):
        ix16 = jnp.broadcast_to(ix[t : t + 1, :], (16, K))
        outs.append(jnp.take_along_axis(s, ix16, axis=1))
      o_ref[...] = jnp.concatenate(outs, axis=0)  # (128, 128)

    fn2 = jax.jit(lambda s, ix: pl.pallas_call(
        _k_take_8,
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
    ms, comp, out2 = bench_fn(fn2, (splats_b16n, idx_btk), n_warmup=1, n_steps=5)
    got2 = jnp.transpose(out2.reshape(B, T, 16, K), (0, 1, 3, 2))
    diff2 = float(jnp.max(jnp.abs(got2[:4] - ref[:4])))
    print(f"TEST2 take_along_axis T_GRP=8: {ms:.2f} ms (compile {comp:.2f}s), diff={diff2:.2e}", flush=True)
  except Exception as e:
    print(f"TEST2 FAILED: {type(e).__name__}: {str(e)[:400]}", flush=True)

  # Test 3: Pallas MXU hi+lo chunked gather (transpose (1, 128) -> (128, 1) per tile, NO 8x128->1024x1 reshape!)
  try:
    N_CHUNKS = N // 128
    def _k_mxu_tile(s_ref, i_ref, o_ref):
      s_all = s_ref[...]  # (16, 2048)
      ix = i_ref[...]     # (8, 128) -> 8 tiles, 128 splats each
      iota = jnp.broadcast_to(jnp.arange(128, dtype=jnp.int32)[None, :], (128, 128))
      # For each tile t in 0..7: ix[t:t+1, :].T -> (128, 1) -> broadcast (128, 128)
      outs = []
      for t in range(8):
        ix_col = jnp.broadcast_to(jnp.transpose(ix[t : t + 1, :], (1, 0)), (128, 128))
        acc_hi = jnp.zeros((128, 16), jnp.float32) # wait: 16 cols? Let's pad to 128 or use 16 cols?
        # Let's accumulate in (128, 128) with 8 tiles packed into 128 cols!! Wait: 8 tiles * 16 = 128 cols!
        pass
      # Even simpler: process 1 tile -> (128, 128) matmul, or pack 8 tiles into columns?!
      # Wait: for 1 tile t, oh_t is (128, 128). If we stack 8 tiles vertically: oh_all is (8*128, 128)? Still 128 cols!
      # Wait: we can't concat vertically without same loop, OR just loop over N_CHUNKS and inside loop over 8 tiles!
      # Wait: better yet, do 1 tile per grid step? No, 8 tiles:
      acc_list_hi = [jnp.zeros((128, 128), jnp.float32) for _ in range(1)]
      # Wait: transpose s_chunk (16, 128) -> (128, 16) -> pad to (128, 128) -> matmul with oh (128, 128)!
      # Wait: (128, 128) @ (128, 128) only uses 16 cols. If we pack 8 chunks of 16 cols into (128, 128)? No, same K!
      # What if we keep (16, 128) on the LEFT: (16, 128) @ (128, 128) -> (16, 128)!!
      # WAIT A SECOND! (16, 128) @ (128, 128) gives (16, 128) -> ZERO WASTED COLUMNS!!
      # And we don't even need to transpose s_all[:, base:base+128]!!
      ix_cols = [jnp.broadcast_to(ix[t : t + 1, :], (128, 128)) for t in range(8)]
      # Wait: if s_chunk is (16, 128) and oh_t has 1-hot along axis 0: oh_t[n, k] = 1 iff n == ix[t, k]!
      # Then (16, 128) @ (128, 128) = (16, 128)!! ZERO transpose of s_chunk AND ZERO column padding!!
      accs = [jnp.zeros((16, 128), jnp.float32) for _ in range(8)]
      iota_col = jnp.broadcast_to(jnp.transpose(jnp.arange(128, dtype=jnp.int32)[None, :], (1, 0)), (128, 128))
      for c in range(N_CHUNKS):
        base = c * 128
        s_chunk = s_all[:, base : base + 128]  # (16, 128)
        s_hi = s_chunk.astype(jnp.bfloat16)
        s_lo = (s_chunk - s_hi.astype(jnp.float32)).astype(jnp.bfloat16)
        iota_b = iota_col + base
        for t in range(8):
          oh = (ix_cols[t] == iota_b).astype(jnp.bfloat16)  # (128, 128)
          accs[t] = accs[t] + jnp.matmul(s_hi, oh, preferred_element_type=jnp.float32) + jnp.matmul(s_lo, oh, preferred_element_type=jnp.float32)
      o_ref[...] = jnp.concatenate(accs, axis=0)  # (128, 128)

    T_GRP = 8
    n_tg = T // T_GRP
    fn3 = jax.jit(lambda s, ix: pl.pallas_call(
        _k_mxu_tile,
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
    ms, comp, out3 = bench_fn(fn3, (splats_b16n, idx_btk), n_warmup=1, n_steps=5)
    got3 = jnp.transpose(out3.reshape(B, T, 16, K), (0, 1, 3, 2))
    diff3 = float(jnp.max(jnp.abs(got3[:4] - ref[:4])))
    print(f"TEST3 MXU hi+lo (16,128)@(128,128): {ms:.2f} ms (compile {comp:.2f}s), diff={diff3:.2e}", flush=True)
  except Exception as e:
    print(f"TEST3 FAILED: {type(e).__name__}: {str(e)[:400]}", flush=True)


if __name__ == "__main__":
  main()
