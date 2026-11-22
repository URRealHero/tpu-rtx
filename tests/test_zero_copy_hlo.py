"""Zero-copy XLA HLO verification: fused mjx.step + TPU render must have 0 host callbacks."""

from __future__ import annotations

import json
import os
import re

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from tpu_mjx_render.renderer import RenderSceneSpec, render_batch
from tpu_mjx_render.scenes import get_scene_xml


def check_hlo_zero_copy(hlo_text: str) -> dict:
  host_cb = len(re.findall(r"host_callback|HostCallback|python_callback|custom-call.*callback", hlo_text, re.I))
  send_recv = len(re.findall(r"\b(?:send|recv)\(", hlo_text))
  return {
      "hlo_bytes": len(hlo_text),
      "hlo_lines": hlo_text.count("\n"),
      "host_callback_count": host_cb,
      "send_recv_count": send_recv,
      "zero_copy_verified": (host_cb == 0),
  }


def main():
  os.makedirs("out/hlo", exist_ok=True)
  os.makedirs("out/metrics", exist_ok=True)
  summary = {}

  for sc in ["cartpole", "panda_pick_cube"]:
    m = mujoco.MjModel.from_xml_string(get_scene_xml(sc))
    mx = mjx.put_model(m, impl="jax")
    spec = RenderSceneSpec.from_mjmodel(m, cam_id=0, height=64, width=64)

    B = 1024
    dx0 = mjx.make_data(m, impl="jax")
    dx_batch = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x[None, ...], (B, *x.shape)).copy(), dx0)
    ctrl = jnp.zeros((B, m.nu), dtype=jnp.float32)

    @jax.jit
    def fused_step_and_render(d_b, u_b):
      def step_one(d, u):
        return mjx.step(mx, d.replace(ctrl=u))
      d_next = jax.vmap(step_one)(d_b, u_b)
      gb = render_batch(spec, d_next, mode="tiled")
      return d_next, gb.rgb, gb.depth

    lowered = fused_step_and_render.lower(dx_batch, ctrl)
    hlo_text = lowered.as_text()
    hlo_path = f"out/hlo/{sc}_fused_step_render_B1024.hlo.txt"
    with open(hlo_path, "w") as f:
      f.write(hlo_text)

    info = check_hlo_zero_copy(hlo_text)
    info["hlo_path"] = hlo_path
    summary[sc] = info
    assert info["zero_copy_verified"], f"Found host callback in {sc}: {info}"

  with open("out/metrics/zero_copy_hlo.json", "w") as f:
    json.dump(summary, f, indent=2)
  print(json.dumps(summary, indent=2))
  print("ZERO-COPY HLO VERIFICATION PASSED")


if __name__ == "__main__":
  main()
