"""Check counted_kernel.pbr_raycast_counted against the original kernel and Embree.

Also times generic interpret mode (interpret=True) against TPU interpret mode.
"""

import time

import common
import jax
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
import numpy as np

from bvh_pbr_pt import build_pbr_wide_bvh, pbr_raycast_tpu
from counted_kernel import pbr_raycast_counted
from smoke_interpret import camera_rays, make_mesh


def main():
  verts, faces = make_mesh(150)
  F = len(faces)
  bvh = build_pbr_wide_bvh("t", verts, faces, np.full((F, 3), 0.5, np.float32), np.full((F,), 0.5, np.float32), np.zeros((F,), np.float32))
  H, W = 64, 128
  ro, rd = camera_rays(H, W, None)
  # Mix in incoherent rays: random origins above the surface, random directions.
  rng = np.random.default_rng(0)
  k = 2048
  ro[:k] = rng.uniform([-0.9, -0.9, 0.3], [0.9, 0.9, 0.9], (k, 3))
  d = rng.standard_normal((k, 3)).astype(np.float32)
  rd[:k] = d / np.linalg.norm(d, axis=1, keepdims=True)
  tl = np.full((len(ro),), 100.0, np.float32)
  bv = [jnp.asarray(x) for x in (bvh.l2_roots_aabb, bvh.l2_aabbs, bvh.l0_aabbs, bvh.leaf_bank)]
  args = (jnp.asarray(ro[None]), jnp.asarray(rd[None]), jnp.asarray(tl[None]))
  with pltpu.force_tpu_interpret_mode():
    ref = np.asarray(jax.jit(lambda *a: pbr_raycast_tpu(*a, H, W, any_hit=False)[0])(*bv, *args))[0]
  for interp in (True, "tpu"):
    f = jax.jit(lambda *a: pbr_raycast_counted(*a, H, W, interpret=True))
    t0 = time.perf_counter()
    if interp == "tpu":
      with pltpu.force_tpu_interpret_mode():
        dist, cnt = f(*bv, *args)
        jax.block_until_ready(dist)
        t1 = time.perf_counter()
        dist, cnt = f(*bv, *args)
        jax.block_until_ready(dist)
    else:
      dist, cnt = f(*bv, *args)
      jax.block_until_ready(dist)
      t1 = time.perf_counter()
      dist, cnt = f(*bv, *args)
      jax.block_until_ready(dist)
    t2 = time.perf_counter()
    dist = np.asarray(dist)[0]
    cnt = np.asarray(cnt)
    same = np.array_equal(dist, ref)
    print(f"mode={interp} first={t1 - t0:.2f}s second={t2 - t1:.2f}s identical_to_original={same} max_abs_diff={np.max(np.abs(dist - ref)):.3e}")
    print("  per-tile counts (L2, L1, leaf, needed, live) sum:", cnt.sum(0).tolist(), "tiles:", cnt.shape[0])
    print("  coherent tiles (last 32) mean leaf fetches:", cnt[-32:, 2].mean(), " incoherent tiles (first 16):", cnt[:16, 2].mean())
  t_e, _ = common.embree_closest_hit(verts, faces, ro, rd)
  hit = dist < 100.0 - 1e-4
  print("hit agreement with Embree:", np.mean(hit == np.isfinite(t_e)))


if __name__ == "__main__":
  main()
