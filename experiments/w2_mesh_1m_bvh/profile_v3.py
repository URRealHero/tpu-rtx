"""Profile V3 Wide-BVH breakdown (primary vs shadow vs raygen/shade) & leaf-hit counts on w2."""

import time
import jax
import jax.numpy as jnp
import numpy as np

from bench_v3_throughput import load_scaled_meshes
from wide_bvh_v3 import (
    _pallas_wide_bvh_raycast,
    build_wide_bvh_scene_spec,
    render_wide_bvh_batch,
    to_device_arrays,
)
from tpu_mjx_render.pallas_v2 import _compute_batch_camera_rays_soa


def main():
  ds = np.load("/dev/shm/zaychik-tpu-mjx/out/assets/blender_real_scene.npz")
  M = int(ds["n_meshes"])
  base_meshes = [
      {
          "name": str(ds[f"mesh_{m}_name"]),
          "verts": ds[f"mesh_{m}_verts"],
          "faces": ds[f"mesh_{m}_faces"],
          "rgba": ds[f"mesh_{m}_rgba"],
      }
      for m in range(M)
  ]

  B = 256
  H = W = 64
  def _tile(arr):
    rep = (B + arr.shape[0] - 1) // arr.shape[0]
    return jnp.asarray(np.tile(arr, (rep,) + (1,) * (arr.ndim - 1))[:B])

  oxp, oxm = _tile(ds["obj_xpos"]), _tile(ds["obj_xmat"])
  cxp, cxm = _tile(ds["cam_xpos"]), _tile(ds["cam_xmat"])

  for k in [1, 16]:
    meshes = load_scaled_meshes(base_meshes, k)
    total_tris = sum(m["faces"].shape[0] for m in meshes)
    for use_shadows in [False, True]:
      spec = build_wide_bvh_scene_spec(
          meshes, height=H, width=W,
          cam_fovy=float(ds["cam_fovy"]), znear=float(ds["znear"]), zfar=float(ds["zfar"]),
          light_pos=tuple(float(x) for x in ds["light_pos"]),
          use_shadows=use_shadows,
      )
      dev_arr = to_device_arrays(spec)
      fn = jax.jit(lambda da, a, b, c, d: render_wide_bvh_batch(spec, da, a, b, c, d, interpret=False).rgb)
      y = fn(dev_arr, oxp, oxm, cxp, cxm)
      jax.block_until_ready(y)
      t0 = time.perf_counter()
      for _ in range(10):
        y = fn(dev_arr, oxp, oxm, cxp, cxm)
      jax.block_until_ready(y)
      ms = (time.perf_counter() - t0) * 1e3 / 10
      print(f"tris={total_tris:>9d} shadows={str(use_shadows):5s} | {ms:7.2f} ms | {B*1e3/ms:8.1f} FPS", flush=True)


if __name__ == "__main__":
  main()
