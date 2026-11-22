"""Diagnose exact gather vs rasterize cost and Mosaic gather/packing options on live TPU v5p."""

from __future__ import annotations

import time
import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
import numpy as np

from tpu_mjx_render.gs3d import _pallas_gs3d_rasterize, mesh_scene_to_rigid_3dgs


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


def _jax_tile_rasterize_only(mx, my, icxx, icxy, icyy, sr, sg, sb, sop, sz, snx, sny, snz, tox_dev, toy_dev, tile_size=8):
  lx = jnp.arange(tile_size, dtype=jnp.float32) + 0.5
  ly = jnp.arange(tile_size, dtype=jnp.float32) + 0.5
  px = tox_dev[..., None] + lx[None, None, None, :]
  py = toy_dev[..., None] + ly[None, None, :, None]
  dx = px[..., None] - mx[:, :, None, None, :]
  dy = py[..., None] - my[:, :, None, None, :]
  power = -0.5 * (
      icxx[:, :, None, None, :] * dx * dx
      + 2.0 * icxy[:, :, None, None, :] * dx * dy
      + icyy[:, :, None, None, :] * dy * dy
  )
  valid_p = (power <= 0.0) & (power > -16.0)
  alpha_k = jnp.where(valid_p, jnp.minimum(0.99, sop[:, :, None, None, :] * jnp.exp(jnp.maximum(power, -16.0))), 0.0)
  alpha_k = jnp.where(alpha_k >= (1.0 / 255.0), alpha_k, 0.0)
  one_m_a = 1.0 - alpha_k
  trans_excl = jnp.cumprod(jnp.concatenate([jnp.ones_like(one_m_a[..., :1]), one_m_a[..., :-1]], axis=-1), axis=-1)
  w_k = alpha_k * trans_excl
  alp_t = 1.0 - jnp.prod(one_m_a, axis=-1)
  rgb_t = jnp.stack([
      jnp.sum(w_k * sr[:, :, None, None, :], axis=-1),
      jnp.sum(w_k * sg[:, :, None, None, :], axis=-1),
      jnp.sum(w_k * sb[:, :, None, None, :], axis=-1),
  ], axis=-1)
  dep_t = jnp.sum(w_k * sz[:, :, None, None, :], axis=-1)
  nrm_t = jnp.stack([
      jnp.sum(w_k * snx[:, :, None, None, :], axis=-1),
      jnp.sum(w_k * sny[:, :, None, None, :], axis=-1),
      jnp.sum(w_k * snz[:, :, None, None, :], axis=-1),
  ], axis=-1)
  return rgb_t, dep_t, nrm_t, alp_t


def main():
  B, T, K = 1024, 64, 128
  N = 2048
  key = jax.random.PRNGKey(0)
  # Create synthetic pre-gathered tensors in HBM
  arr13 = [jax.random.uniform(jax.random.PRNGKey(i), (B, T, K), jnp.float32) for i in range(13)]
  arr13[2] = jnp.ones((B, T, K), jnp.float32) * 0.5
  arr13[3] = jnp.zeros((B, T, K), jnp.float32)
  arr13[4] = jnp.ones((B, T, K), jnp.float32) * 0.5
  arr13[8] = jnp.full((B, T, K), 0.1, jnp.float32)
  tox = jnp.zeros((B, T), jnp.float32)
  toy = jnp.zeros((B, T), jnp.float32)
  tox_dev = jnp.zeros((1, T, 1), jnp.float32)
  toy_dev = jnp.zeros((1, T, 1), jnp.float32)

  # 1. Pure rasterize timing (no gather!)
  fn_jr = jax.jit(lambda *xs: _jax_tile_rasterize_only(*xs, tox_dev, toy_dev, tile_size=8))
  ms_jr, c_jr, _ = bench_fn(fn_jr, arr13)
  print(f"RASTER_ONLY jax_tile (B={B}, T={T}, K={K}): {ms_jr:.2f} ms (compile {c_jr:.2f}s)", flush=True)

  fn_pr = jax.jit(lambda *xs: _pallas_gs3d_rasterize(*xs, tox, toy, tile_size=8, interpret=False))
  ms_pr, c_pr, _ = bench_fn(fn_pr, arr13)
  print(f"RASTER_ONLY pallas_v2 (B={B}, T={T}, K={K}): {ms_pr:.2f} ms (compile {c_pr:.2f}s)", flush=True)

  # 2. Compare gather strategies for (B, N, 16) -> (B, T, K, 16)
  packed_bn = jax.random.uniform(key, (B, N, 16), jnp.float32)
  top_idx = jax.random.randint(key, (B, T, K), 0, N, jnp.int32)

  def g_1packed(p, idx):
    b_ix = jnp.arange(B, dtype=jnp.int32)[:, None, None]
    return p[b_ix, idx]  # (B, T, K, 16)

  def g_1packed_vmap(p, idx):
    return jax.vmap(lambda pb, ib: pb[ib])(p, idx)

  def g_1packed_take_along(p, idx):
    # p: (B, N, 16), idx: (B, T*K) -> (B, T*K, 16)
    return jnp.take_along_axis(p, idx.reshape(B, T * K, 1), axis=1).reshape(B, T, K, 16)

  def g_1packed_16first(p, idx):
    # p: (B, 16, N) -> vmap over B: p_b[:, idx_b]
    p16 = jnp.transpose(p, (0, 2, 1))
    return jax.vmap(lambda pb, ib: pb[:, ib])(p16, idx)  # (B, 16, T, K)

  for name, fn in [
      ("gather_1packed_bix", jax.jit(g_1packed)),
      ("gather_1packed_vmap", jax.jit(g_1packed_vmap)),
      ("gather_1packed_take_along", jax.jit(g_1packed_take_along)),
      ("gather_1packed_16first", jax.jit(g_1packed_16first)),
  ]:
    ms_g, c_g, _ = bench_fn(fn, (packed_bn, top_idx))
    print(f"{name}: {ms_g:.2f} ms (compile {c_g:.2f}s)", flush=True)

  # 3. Check how many Gaussians actually overlap per tile in blender_real_scene
  ds = np.load("/dev/shm/zaychik-tpu-mjx/out/assets/blender_real_scene.npz")
  M = int(ds["n_meshes"])
  meshes = [{"name": str(ds[f"mesh_{m}_name"]), "verts": ds[f"mesh_{m}_verts"], "faces": ds[f"mesh_{m}_faces"], "rgba": ds[f"mesh_{m}_rgba"]} for m in range(M)]
  spec = mesh_scene_to_rigid_3dgs(meshes, total_gaussians=2048, height=64, width=64, cam_fovy=float(ds["cam_fovy"]), znear=float(ds["znear"]), light_pos=tuple(float(x) for x in ds["light_pos"]))
  from profile_v2_gs3d import stage_frontend_and_binning
  oxp = jnp.asarray(ds["obj_xpos"][:32])
  oxm = jnp.asarray(ds["obj_xmat"][:32])
  cxp = jnp.asarray(ds["cam_xpos"][:32])
  cxm = jnp.asarray(ds["cam_xmat"][:32])
  mean_x, mean_y, icxx, icxy, icyy, rgb_lit, op_b, z_d, nrm_vis, valid_g, radius = stage_frontend_and_binning(spec, oxp, oxm, cxp, cxm)
  Ty = Tx = 8
  t_ox = np.tile(np.arange(Tx, dtype=np.float32) * 8, Ty)
  t_oy = np.repeat(np.arange(Ty, dtype=np.float32) * 8, Tx)
  tox_d = jnp.asarray(t_ox, jnp.float32)[None, :, None]
  toy_d = jnp.asarray(t_oy, jnp.float32)[None, :, None]
  near_x = jnp.clip(mean_x[:, None, :], tox_d, tox_d + 8.0)
  near_y = jnp.clip(mean_y[:, None, :], toy_d, toy_d + 8.0)
  dx_t = mean_x[:, None, :] - near_x
  dy_t = mean_y[:, None, :] - near_y
  overlap = valid_g[:, None, :] & ((dx_t * dx_t + dy_t * dy_t) <= (radius * radius)[:, None, :])
  cnt = jnp.sum(overlap, axis=-1)
  print(f"Overlap count per 8x8 tile: min={int(jnp.min(cnt))}, median={float(jnp.median(cnt)):.1f}, p95={float(jnp.percentile(cnt, 95)):.1f}, max={int(jnp.max(cnt))}", flush=True)


if __name__ == "__main__":
  main()
