"""Diagnostic breakdown of V2.2 pallas_v2 on live TPU v5p (w1)."""

import time
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np

from tpu_mjx_render.renderer import RenderSceneSpec, render_batch
from tpu_mjx_render.pallas_v2 import (
    _compute_batch_camera_rays_soa,
    _extract_light_batch,
    _shade_batch_soa,
)
from tpu_mjx_render.scenes import get_scene_xml


def main():
  print("Backend:", jax.default_backend(), "Devices:", jax.devices())
  m = mujoco.MjModel.from_xml_string(get_scene_xml("panda_pick_cube"))
  mx = mjx.put_model(m, impl="jax")
  spec = RenderSceneSpec.from_mjmodel(m, cam_id=0, height=64, width=64, use_shadows=True)
  print("PandaPickCube enabled geoms G =", len(spec.enabled_geom_ids),
        "types =", [int(spec.geom_type[g]) for g in spec.enabled_geom_ids],
        "nlight =", spec.nlight, "castshadow =", spec.light_castshadow.tolist())

  B = 1024
  dx0 = mjx.make_data(m, impl="jax")
  d_host = mujoco.MjData(m)
  mujoco.mj_resetDataKeyframe(m, d_host, 0)
  dx0 = dx0.replace(qpos=jnp.asarray(d_host.qpos, jnp.float32))
  dx0 = mjx.forward(mx, dx0)
  dx_batch = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x[None, ...], (B, *x.shape)).copy(), dx0)
  ctrl = jnp.zeros((B, m.nu), dtype=jnp.float32)
  dx_batch = jax.device_put(dx_batch, jax.devices()[0])
  ctrl = jax.device_put(ctrl, jax.devices()[0])

  pallas_interp = (jax.default_backend() == "cpu")

  @jax.jit
  def fn_phys(d, u):
    return jax.vmap(lambda di, ui: mjx.step(mx, di.replace(ctrl=ui)))(d, u)

  @jax.jit
  def fn_render_only(d):
    gb = render_batch(spec, d, mode="pallas_v2", pallas_interpret=pallas_interp)
    return gb.rgb, gb.depth

  @jax.jit
  def fn_e2e(d, u):
    dn = jax.vmap(lambda di, ui: mjx.step(mx, di.replace(ctrl=ui)))(d, u)
    gb = render_batch(spec, dn, mode="pallas_v2", pallas_interpret=pallas_interp)
    return dn, gb.rgb, gb.depth

  @jax.jit
  def fn_shade_only(d):
    cam_pos = d.cam_xpos[:, int(spec.cam_id), :]
    cam_mat = d.cam_xmat[:, int(spec.cam_id), :, :]
    ray_orig, ray_dir, neg_z = _compute_batch_camera_rays_soa(spec, cam_pos, cam_mat)
    light_xpos, light_xdir = _extract_light_batch(spec, d)
    geom_size = jnp.asarray(spec.geom_size, dtype=jnp.float32)
    geom_rgba = jnp.asarray(spec.geom_rgba, dtype=jnp.float32)
    dist = jnp.ones((B, 4096), jnp.float32)
    normal = jnp.broadcast_to(jnp.array([0.0, 0.0, 1.0], jnp.float32), (B, 4096, 3))
    geom_id = jnp.zeros((B, 4096), jnp.int32)
    occ = jnp.zeros((B, 4096), jnp.bool_)
    gb = _shade_batch_soa(
        spec, d.geom_xpos, d.geom_xmat, geom_size, geom_rgba,
        light_xpos, light_xdir, ray_orig, ray_dir, neg_z,
        dist, normal, geom_id, fused_shadow0_occluded=occ,
    )
    return gb.rgb, gb.depth

  def bench(name, fn, *args, n_warmup=3, n_steps=20):
    for _ in range(n_warmup):
      out = fn(*args)
    jax.block_until_ready(out)
    t0 = time.perf_counter()
    for _ in range(n_steps):
      out = fn(*args)
    jax.block_until_ready(out)
    ms = (time.perf_counter() - t0) * 1e3 / n_steps
    print(f"{name}: {ms:.3f} ms ({B * 1e3 / ms:.1f} FPS)")
    return ms

  bench("phys_only", fn_phys, dx_batch, ctrl)
  bench("shade_plus_raygen_only", fn_shade_only, dx_batch)
  bench("render_only (pallas_v2)", fn_render_only, dx_batch)
  bench("e2e (phys + pallas_v2)", fn_e2e, dx_batch, ctrl)


if __name__ == "__main__":
  main()
