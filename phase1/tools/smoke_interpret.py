"""Smoke test: run TPU-RTX's Wide-BVH-64 kernel (pbr_raycast_tpu) on CPU in Pallas
TPU interpret mode, check its hits against Embree, and time it.

Usage: python smoke_interpret.py [grid_n] [H] [W]
"""

import sys
import time

import common  # noqa: F401  (sets JAX_PLATFORMS=cpu and sys.path)

import jax
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
import numpy as np

from bvh_pbr_pt import build_pbr_wide_bvh, pbr_raycast_tpu


def make_mesh(n, seed=0):
  """Bumpy heightfield of 2*n*n triangles over [-1,1]^2 plus a box of random triangles."""
  rng = np.random.default_rng(seed)
  xs = np.linspace(-1.0, 1.0, n + 1, dtype=np.float32)
  gx, gy = np.meshgrid(xs, xs, indexing="xy")
  gz = 0.15 * np.sin(4.0 * gx) * np.cos(3.0 * gy) + 0.02 * rng.standard_normal(gx.shape).astype(np.float32)
  verts = np.stack([gx, gy, gz], -1).reshape(-1, 3)
  idx = np.arange((n + 1) * (n + 1)).reshape(n + 1, n + 1)
  a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
  faces = np.concatenate([np.stack([a, b, d], -1).reshape(-1, 3), np.stack([a, d, c], -1).reshape(-1, 3)])
  k = 2000
  cent = rng.uniform([-0.8, -0.8, 0.2], [0.8, 0.8, 0.8], size=(k, 3)).astype(np.float32)
  tri = cent[:, None, :] + 0.05 * rng.standard_normal((k, 3, 3)).astype(np.float32)
  v2 = tri.reshape(-1, 3)
  f2 = np.arange(3 * k).reshape(k, 3) + len(verts)
  return np.concatenate([verts, v2]).astype(np.float32), np.concatenate([faces, f2]).astype(np.int32)


def camera_rays(H, W, rng):
  cam = np.array([0.0, -2.2, 1.6], np.float32)
  tgt = np.array([0.0, 0.0, 0.0], np.float32)
  f = tgt - cam
  f /= np.linalg.norm(f)
  r = np.cross(f, [0, 0, 1]).astype(np.float32)
  r /= np.linalg.norm(r)
  u = np.cross(r, f)
  ys, xs = np.meshgrid((np.arange(H) + 0.5) / H, (np.arange(W) + 0.5) / W, indexing="ij")
  dx = (xs * 2 - 1) * 0.6 * W / H
  dy = (1 - ys * 2) * 0.6
  d = f[None, None] + dx[..., None] * r + dy[..., None] * u
  d /= np.linalg.norm(d, axis=-1, keepdims=True)
  ro = np.broadcast_to(cam, d.shape).reshape(-1, 3)
  return ro.astype(np.float32), d.reshape(-1, 3).astype(np.float32)


def run(grid_n=150, H=64, W=128):
  verts, faces = make_mesh(grid_n)
  F = len(faces)
  bvh = build_pbr_wide_bvh(
      "smoke", verts, faces,
      np.full((F, 3), 0.5, np.float32), np.full((F,), 0.5, np.float32), np.zeros((F,), np.float32),
  )
  print(f"mesh tris={F} leaves={bvh.n_leaves} l1={bvh.n_l1} l2={bvh.n_l2}", flush=True)
  ro, rd = camera_rays(H, W, None)
  tl = np.full((len(ro),), 100.0, np.float32)
  args = [jnp.asarray(x) for x in (bvh.l2_roots_aabb, bvh.l2_aabbs, bvh.l0_aabbs, bvh.leaf_bank)]
  fn = jax.jit(lambda a, b, c, d, o, r, t: pbr_raycast_tpu(a, b, c, d, o, r, t, H, W, any_hit=False))
  with pltpu.force_tpu_interpret_mode():
    t0 = time.perf_counter()
    out = fn(*args, jnp.asarray(ro[None]), jnp.asarray(rd[None]), jnp.asarray(tl[None]))
    jax.block_until_ready(out)
    t1 = time.perf_counter()
    out = fn(*args, jnp.asarray(ro[None]), jnp.asarray(rd[None]), jnp.asarray(tl[None]))
    jax.block_until_ready(out)
    t2 = time.perf_counter()
  dist = np.asarray(out[0])[0]
  tpu_hit = dist < 100.0 - 1e-4
  t_e, _ = common.embree_closest_hit(verts, faces, ro, rd)
  e_hit = np.isfinite(t_e)
  agree_hit = np.mean(tpu_hit == e_hit)
  both = tpu_hit & e_hit
  rel = np.abs(dist[both] - t_e[both]) / np.maximum(t_e[both], 1e-6)
  print(f"rays={len(ro)} first_call_s={t1 - t0:.2f} second_call_s={t2 - t1:.2f}", flush=True)
  print(f"hit/miss agreement={agree_hit:.6f} hits={both.sum()} rel_dist_err p50={np.median(rel):.2e} p99={np.quantile(rel, 0.99):.2e} max={rel.max():.2e}", flush=True)


if __name__ == "__main__":
  a = [int(x) for x in sys.argv[1:]]
  run(*a)
