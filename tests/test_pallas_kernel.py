"""Phase 1 Pallas TPU VMEM Kernel Verification (interpret=True on CPU, native on TPU)."""

from __future__ import annotations

import json
import os

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np

from tpu_mjx_render.pallas_render import pallas_mxu_pixel_mlp
from tpu_mjx_render.renderer import RenderSceneSpec, render_exact_gbuffer
from tpu_mjx_render.scenes import get_scene_xml


def main():
  os.makedirs("out/metrics", exist_ok=True)

  # 1. Verify pallas_raycast_scene matches JAX tiled/naive exact raycaster
  results = {}
  for sc in ["cartpole", "panda_pick_cube"]:
    m = mujoco.MjModel.from_xml_string(get_scene_xml(sc))
    mx = mjx.put_model(m, impl="jax")
    d = mujoco.MjData(m)
    if m.nkey > 0:
      mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    dx = mjx.make_data(m, impl="jax").replace(qpos=jnp.asarray(d.qpos, jnp.float32))
    dx = mjx.forward(mx, dx)

    spec = RenderSceneSpec.from_mjmodel(m, cam_id=0, height=64, width=64)
    gb_tiled = jax.jit(lambda x: render_exact_gbuffer(spec, x, mode="tiled"))(dx)
    gb_naive = jax.jit(lambda x: render_exact_gbuffer(spec, x, mode="naive"))(dx)
    gb_pallas = jax.jit(lambda x: render_exact_gbuffer(spec, x, mode="pallas", pallas_interpret=True))(dx)

    valid = np.asarray(gb_tiled.geom_id) >= 0
    diff_naive_depth = float(np.max(np.abs(np.asarray(gb_tiled.depth)[valid] - np.asarray(gb_naive.depth)[valid])))
    diff_pallas_depth = float(np.max(np.abs(np.asarray(gb_tiled.depth)[valid] - np.asarray(gb_pallas.depth)[valid])))
    diff_pallas_rgb = float(np.max(np.abs(np.asarray(gb_tiled.rgb) - np.asarray(gb_pallas.rgb))))
    match_pallas_gid = float(np.mean(np.asarray(gb_tiled.geom_id) == np.asarray(gb_pallas.geom_id)))

    results[sc] = {
        "naive_vs_tiled_max_depth_diff": diff_naive_depth,
        "pallas_vs_tiled_max_depth_diff": diff_pallas_depth,
        "pallas_vs_tiled_max_rgb_diff": diff_pallas_rgb,
        "pallas_vs_tiled_geom_match": match_pallas_gid,
    }
    assert diff_naive_depth < 1e-5, results[sc]
    assert diff_pallas_depth < 1e-4, results[sc]
    assert match_pallas_gid >= 0.9999, results[sc]

  # 2. Verify pallas_mxu_pixel_mlp matches pure JAX BF16 MLP
  key = jax.random.PRNGKey(0)
  k0, k1, k2, k3, kx = jax.random.split(key, 5)
  x = jax.random.normal(kx, (4096, 128), dtype=jnp.float32) * 0.5
  w0 = jax.random.normal(k0, (128, 128), dtype=jnp.float32) * 0.1
  w1 = jax.random.normal(k1, (128, 128), dtype=jnp.float32) * 0.1
  w2 = jax.random.normal(k2, (128, 128), dtype=jnp.float32) * 0.1
  w3 = jax.random.normal(k3, (128, 128), dtype=jnp.float32) * 0.1

  def ref_mlp(xi, a, b, c, d):
    h = xi.astype(jnp.bfloat16)
    h = jax.nn.relu(jnp.dot(h, a.astype(jnp.bfloat16), preferred_element_type=jnp.float32)).astype(jnp.bfloat16)
    h = jax.nn.relu(jnp.dot(h, b.astype(jnp.bfloat16), preferred_element_type=jnp.float32)).astype(jnp.bfloat16)
    h = jax.nn.relu(jnp.dot(h, c.astype(jnp.bfloat16), preferred_element_type=jnp.float32)).astype(jnp.bfloat16)
    return jnp.dot(h, d.astype(jnp.bfloat16), preferred_element_type=jnp.float32)[:, :3]

  y_ref = jax.jit(ref_mlp)(x, w0, w1, w2, w3)
  y_pal = jax.jit(lambda xi, a, b, c, d: pallas_mxu_pixel_mlp(xi, a, b, c, d, interpret=True))(x, w0, w1, w2, w3)
  mlp_max_diff = float(np.max(np.abs(np.asarray(y_ref) - np.asarray(y_pal))))
  results["pallas_mxu_pixel_mlp_max_diff"] = mlp_max_diff
  assert mlp_max_diff < 1e-4, mlp_max_diff

  with open("out/metrics/pallas_kernel_parity.json", "w") as f:
    json.dump(results, f, indent=2)
  print(json.dumps(results, indent=2))
  print("ALL PALLAS KERNEL TESTS PASSED")


if __name__ == "__main__":
  main()
