"""Probe Mosaic Pallas dynamic HBM->VMEM DMA & scalar/while lowering on TPU v5p."""

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
import numpy as np


def test_smem_prefetch_dma():
  """Test 1: PrefetchScalarGridSpec + dynamic HBM->VMEM async_copy from SMEM scalar."""
  N_LEAVES = 16
  N_TILES = 4
  K = 3
  bank = jnp.arange(N_LEAVES * 24 * 128, dtype=jnp.float32).reshape(N_LEAVES, 24, 128)
  cand_ids = jnp.array([[0, 5, -1], [2, 3, 7], [-1, -1, -1], [15, 1, 4]], dtype=jnp.int32)
  rays = jnp.ones((N_TILES * 8, 128), dtype=jnp.float32)

  def kernel(cand_ref, ray_ref, bank_hbm_ref, out_ref, scratch_vmem, sem):
    t_idx = pl.program_id(0)
    acc = jnp.zeros((8, 128), dtype=jnp.float32)

    def cond_fn(carry):
      k, _ = carry
      return (k < K) & (cand_ref[t_idx, k] >= 0)

    def body_fn(carry):
      k, s_acc = carry
      lid = cand_ref[t_idx, k]
      cp = pltpu.make_async_copy(bank_hbm_ref.at[lid, :, :], scratch_vmem, sem)
      cp.start()
      cp.wait()
      s_acc = s_acc + scratch_vmem[0:8, :]
      return (k + 1, s_acc)

    _, acc = jax.lax.while_loop(cond_fn, body_fn, (0, acc))
    out_ref[...] = acc + ray_ref[...]

  @jax.jit
  def run(cids, r, b):
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((N_TILES * 8, 128), jnp.float32),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(N_TILES,),
            in_specs=[
                pl.BlockSpec((8, 128), lambda i, _: (i, 0)),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=pl.BlockSpec((8, 128), lambda i, _: (i, 0)),
            scratch_shapes=[
                pltpu.VMEM((24, 128), jnp.float32),
                pltpu.SemaphoreType.DMA,
            ],
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
    )(cids, r, b)

  out = run(cand_ids, rays, bank)
  jax.block_until_ready(out)
  # Verify tile 0: leaves 0 + 5, tile 2: 0
  ref0 = bank[0, 0:8] + bank[5, 0:8] + 1.0
  ref2 = jnp.ones((8, 128), dtype=jnp.float32)
  assert jnp.allclose(out[0:8], ref0), (out[0:8, 0], ref0[:, 0])
  assert jnp.allclose(out[16:24], ref2), out[16:24, 0]
  print("TEST 1 (SMEM Prefetch + while_loop + HBM->VMEM async_copy): PASS!")


def test_vreg_argmin_to_dma():
  """Test 2: Can we compute argmin inside kernel and DMA from HBM?"""
  N_LEAVES = 64
  N_TILES = 4
  bank = jnp.arange(N_LEAVES * 8 * 128, dtype=jnp.float32).reshape(N_LEAVES, 8, 128)
  scores = jnp.full((N_TILES * 8, 128), 100.0, dtype=jnp.float32)
  scores = scores.at[0:1, 5].set(1.0)
  scores = scores.at[8:9, 12].set(2.0)

  def kernel(score_ref, bank_hbm_ref, out_ref, scratch_vmem, sem):
    s = score_ref[0:1, 0:64]  # (1, 64)
    idx = jnp.argmin(s, axis=1)[0]  # scalar int32?
    cp = pltpu.make_async_copy(bank_hbm_ref.at[idx, :, :], scratch_vmem, sem)
    cp.start()
    cp.wait()
    out_ref[...] = scratch_vmem[...]

  @jax.jit
  def run(sc, b):
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((N_TILES * 8, 128), jnp.float32),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(N_TILES,),
            in_specs=[
                pl.BlockSpec((8, 128), lambda i: (i, 0)),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=pl.BlockSpec((8, 128), lambda i: (i, 0)),
            scratch_shapes=[
                pltpu.VMEM((8, 128), jnp.float32),
                pltpu.SemaphoreType.DMA,
            ],
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
    )(sc, b)

  out = run(scores, bank)
  jax.block_until_ready(out)
  assert jnp.allclose(out[0:8], bank[5])
  assert jnp.allclose(out[8:16], bank[12])
  print("TEST 2 (VREG argmin -> HBM->VMEM async_copy): PASS!")


if __name__ == "__main__":
  print("Backend:", jax.default_backend())
  test_smem_prefetch_dma()
  try:
    test_vreg_argmin_to_dma()
  except Exception as e:
    print("TEST 2 FAILED (expected if VREG->SMEM disallowed):", type(e).__name__, e)
