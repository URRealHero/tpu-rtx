"""Smoke & parity test for upgraded V3.2 (3-level while_loop, stats, TLAS+BLAS instancing, 4K aspect)."""

import jax
import jax.numpy as jnp
import numpy as np

from wide_bvh_v3 import (
    build_wide_bvh_scene_spec,
    render_wide_bvh_batch,
    render_wide_bvh_instanced_batch,
    to_device_arrays,
)


def main():
  ds = np.load("/dev/shm/zaychik-tpu-mjx/out/assets/blender_real_scene.npz")
  M = int(ds["n_meshes"])
  meshes = [
      {
          "name": str(ds[f"mesh_{m}_name"]),
          "verts": ds[f"mesh_{m}_verts"],
          "faces": ds[f"mesh_{m}_faces"],
          "rgba": ds[f"mesh_{m}_rgba"],
      }
      for m in range(M)
  ]
  kw = dict(
      cam_fovy=float(ds["cam_fovy"]),
      znear=float(ds["znear"]),
      zfar=float(ds["zfar"]),
      light_pos=tuple(float(x) for x in ds["light_pos"]),
  )
  spec = build_wide_bvh_scene_spec(meshes, height=64, width=64, **kw)
  dev = to_device_arrays(spec)

  oxp = jnp.asarray(ds["obj_xpos"][:4])
  oxm = jnp.asarray(ds["obj_xmat"][:4])
  cxp = jnp.asarray(ds["cam_xpos"][:4])
  cxm = jnp.asarray(ds["cam_xmat"][:4])

  # 1. Test 3-level while_loop + return_stats
  fn_base = jax.jit(lambda d, a, b, c, e: render_wide_bvh_batch(spec, d, a, b, c, e, interpret=False, return_stats=True))
  gb_base, leaf_visits = fn_base(dev, oxp, oxm, cxp, cxm)
  jax.block_until_ready(gb_base.rgb)
  print("Track 2A 3-level while_loop OK, avg leaf visits/frame:", float(jnp.mean(leaf_visits)))

  # 2. Test Track 2B Instanced parity (K_inst=1)
  ixp = tuple(oxp[:, m : m + 1, :] for m in range(M))
  ixm = tuple(oxm[:, m : m + 1, :, :] for m in range(M))
  fn_inst = jax.jit(lambda d, ip, im, cp, cm: render_wide_bvh_instanced_batch(spec, d, ip, im, cp, cm, max_active_per_tile=4, interpret=False))
  gb_inst = fn_inst(dev, ixp, ixm, cxp, cxm)
  jax.block_until_ready(gb_inst.rgb)

  d_diff = float(jnp.max(jnp.abs(gb_base.depth - gb_inst.depth)))
  r_diff = float(jnp.max(jnp.abs(gb_base.rgb - gb_inst.rgb)))
  g_match = float(jnp.mean(gb_base.geom_id == gb_inst.geom_id))
  print(f"Track 2B K_inst=1 parity vs 2A: max_depth_diff={d_diff}, max_rgb_diff={r_diff}, geom_match={g_match}")
  assert d_diff < 1e-4 and r_diff < 1e-3 and g_match == 1.0

  # 3. Test Track 2B K_inst=10 multi-instance
  offsets = np.linspace(-1.2, 1.2, 10, dtype=np.float32)
  ixp10 = tuple(
      jnp.concatenate([oxp[:, m : m + 1, :] + jnp.array([off, 0.0, 0.0], jnp.float32) for off in offsets], axis=1)
      for m in range(M)
  )
  ixm10 = tuple(jnp.repeat(oxm[:, m : m + 1, :, :], 10, axis=1) for m in range(M))
  fn_inst10 = jax.jit(lambda d, ip, im, cp, cm: render_wide_bvh_instanced_batch(spec, d, ip, im, cp, cm, max_active_per_tile=16, interpret=False))
  gb_inst10 = fn_inst10(dev, ixp10, ixm10, cxp, cxm)
  jax.block_until_ready(gb_inst10.rgb)
  print("Track 2B K_inst=10 OK, fg_frac:", float(jnp.mean(gb_inst10.geom_id >= 0)))


if __name__ == "__main__":
  main()
