"""Zero-copy XLA HLO check for fused MJX step + Stage 1 Raycast + Stage 2 MXU Neural Shader."""

from __future__ import annotations

import json
import os
import pickle

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from tpu_mjx_render.neural_shader import apply_neural_shader, make_neural_shader_model
from tpu_mjx_render.renderer import RenderSceneSpec, render_batch
from tpu_mjx_render.roofline import inspect_hlo_ops
from tpu_mjx_render.scenes import get_scene_xml


def main():
  m = mujoco.MjModel.from_xml_string(get_scene_xml("panda_pick_cube"))
  mx = mjx.put_model(m, impl="jax")
  spec = RenderSceneSpec.from_mjmodel(m, cam_id=0, height=64, width=64)

  ck = pickle.load(open("out/metrics/M2_spatial_128_params.pkl", "rb"))
  model = make_neural_shader_model(ck["arch"])
  params = jax.tree_util.tree_map(jnp.asarray, ck["params"])
  cal_scale = float(ck["cal_scale"])

  B = 1024
  dx0 = mjx.make_data(m, impl="jax")
  dx_batch = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x[None, ...], (B, *x.shape)).copy(), dx0)
  ctrl = jnp.zeros((B, m.nu), dtype=jnp.float32)

  @jax.jit
  def fused_mjx_s1_s2(d_b, u_b):
    d_next = jax.vmap(lambda d, u: mjx.step(mx, d.replace(ctrl=u)))(d_b, u_b)
    gb = render_batch(spec, d_next, mode="tiled")
    rgb_neural = apply_neural_shader(model, params, gb, cal_scale=cal_scale)
    return d_next, rgb_neural, gb.depth

  hlo_text = fused_mjx_s1_s2.lower(dx_batch, ctrl).as_text()
  info = inspect_hlo_ops(hlo_text)
  info["zero_copy_verified"] = info["host_callbacks"] == 0
  assert info["zero_copy_verified"], info

  os.makedirs("out/hlo", exist_ok=True)
  with open("out/hlo/panda_fused_s1_s2_B1024.hlo.txt", "w") as f:
    f.write(hlo_text)
  with open("out/metrics/zero_copy_hlo_stage2.json", "w") as f:
    json.dump(info, f, indent=2)
  print("FUSED STAGE1+STAGE2 HLO:", json.dumps(info, indent=2))


if __name__ == "__main__":
  main()
