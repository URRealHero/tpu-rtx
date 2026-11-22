"""Measure 1c breakdown (XLA transforms/shade vs Pallas BVH) and 4-chip shard_map scaling."""

import time
import jax
import jax.numpy as jnp
import numpy as np

from bench_v3_throughput import load_scaled_meshes
from wide_bvh_v3 import (
    _pallas_wide_bvh_raycast,
    build_wide_bvh_scene_spec,
    make_sharded_wide_bvh_renderer,
    render_wide_bvh_batch,
    to_device_arrays,
)


def main():
  devs = jax.devices()
  print("Visible TPU devices:", len(devs), devs)

  ds = np.load("/dev/shm/zaychik-tpu-mjx/out/assets/blender_real_scene.npz")
  M = int(ds["n_meshes"])
  base = [
      {
          "name": str(ds[f"mesh_{m}_name"]),
          "verts": ds[f"mesh_{m}_verts"],
          "faces": ds[f"mesh_{m}_faces"],
          "rgba": ds[f"mesh_{m}_rgba"],
      }
      for m in range(M)
  ]

  def _tile(arr, B):
    rep = (B + arr.shape[0] - 1) // arr.shape[0]
    return jnp.asarray(np.tile(arr, (rep,) + (1,) * (arr.ndim - 1))[:B])

  for k in [1, 16]:
    meshes = load_scaled_meshes(base, k)
    total_tris = sum(m["faces"].shape[0] for m in meshes)
    for res in [64, 128]:
      spec = build_wide_bvh_scene_spec(
          meshes, height=res, width=res,
          cam_fovy=float(ds["cam_fovy"]), znear=float(ds["znear"]), zfar=float(ds["zfar"]),
          light_pos=tuple(float(x) for x in ds["light_pos"]),
      )
      B = 256 if res == 128 else 1024
      oxp, oxm = _tile(ds["obj_xpos"], B), _tile(ds["obj_xmat"], B)
      cxp, cxm = _tile(ds["cam_xpos"], B), _tile(ds["cam_xmat"], B)

      fn_4c, mesh = make_sharded_wide_bvh_renderer(spec, devs, interpret=False)
      y = fn_4c(oxp, oxm, cxp, cxm).rgb
      jax.block_until_ready(y)

      t0 = time.perf_counter()
      for _ in range(10):
        y = fn_4c(oxp, oxm, cxp, cxm).rgb
      jax.block_until_ready(y)
      ms = (time.perf_counter() - t0) * 1e3 / 10
      print(f"[4-CHIP] tris={total_tris:>9d} res={res}x{res} B={B:>4d} | {ms:7.2f} ms | {B*1e3/ms:9.1f} FPS", flush=True)


if __name__ == "__main__":
  main()
