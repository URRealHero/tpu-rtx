"""Immutable end-to-end throughput & zero-copy benchmark (MJX step + TPU Render).

Strictly measures JIT-warmed end-to-end wall-clock FPS on the active JAX backend,
verifies zero host callbacks in the compiled/lowered XLA HLO, and outputs
TPU v5p / v6e Roofline projections.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import pickle
import time
from typing import Dict, List, Optional

import jax
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np

from tpu_mjx_render.neural_shader import apply_neural_shader, make_neural_shader_model
from tpu_mjx_render.renderer import RenderSceneSpec, render_batch
from tpu_mjx_render.roofline import (
    TPU_SPECS,
    estimate_stage1_roofline,
    estimate_stage2_roofline,
    inspect_hlo_ops,
)
from tpu_mjx_render.scenes import get_scene_xml


def benchmark_e2e(
    scene_name: str,
    batch_size: int,
    height: int,
    width: int,
    mode: str = "tiled",
    n_warmup: int = 3,
    n_steps: int = 20,
    use_shadows: bool = True,
    neural_ckpt: Optional[str] = None,
    n_chips: int = 1,
) -> Dict[str, object]:
  m = mujoco.MjModel.from_xml_string(get_scene_xml(scene_name))
  mx = mjx.put_model(m, impl="jax")
  spec = RenderSceneSpec.from_mjmodel(m, cam_id=0, height=height, width=width, use_shadows=use_shadows)

  neural_model, neural_params, cal_scale, neural_arch = None, None, 1.0, None
  if neural_ckpt and os.path.exists(neural_ckpt):
    ck = pickle.load(open(neural_ckpt, "rb"))
    neural_arch = ck["arch"]
    neural_model = make_neural_shader_model(neural_arch)
    neural_params = jax.tree_util.tree_map(jnp.asarray, ck["params"])
    cal_scale = float(ck["cal_scale"])

  devs = jax.devices()[:n_chips]
  if n_chips > 1:
    assert batch_size % n_chips == 0
    mesh = jax.sharding.Mesh(np.array(devs), ("dp",))
    P = jax.sharding.PartitionSpec
    sharding = jax.sharding.NamedSharding(mesh, P("dp"))
    replicate = jax.sharding.NamedSharding(mesh, P())
  else:
    mesh = None
    P = None
    sharding = None
    replicate = None

  dx0 = mjx.make_data(m, impl="jax")
  if m.nkey > 0:
    d_host = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d_host, 0)
    dx0 = dx0.replace(qpos=jnp.asarray(d_host.qpos, jnp.float32))
  dx0 = mjx.forward(mx, dx0)

  dx_batch = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x[None, ...], (batch_size, *x.shape)).copy(), dx0)
  ctrl = jnp.zeros((batch_size, m.nu), dtype=jnp.float32)

  if n_chips > 1:
    dx_batch = jax.device_put(dx_batch, sharding)
    ctrl = jax.device_put(ctrl, sharding)
    mx = jax.device_put(mx, replicate)
    if neural_params is not None:
      neural_params = jax.device_put(neural_params, replicate)
  else:
    dx_batch = jax.device_put(dx_batch, devs[0])
    ctrl = jax.device_put(ctrl, devs[0])

  pallas_interp = (jax.default_backend() == "cpu")

  if n_chips > 1:
    sm_kw = {"check_rep": False} if "check_rep" in inspect.signature(shard_map).parameters else {"check_vma": False}

    def _phys_local(d_b, u_b, mx_l):
      return jax.vmap(lambda d, u: mjx.step(mx_l, d.replace(ctrl=u)))(d_b, u_b)

    _phys_sm = shard_map(_phys_local, mesh=mesh, in_specs=(P("dp"), P("dp"), P()), out_specs=P("dp"), **sm_kw)

    @jax.jit
    def physics_only(d_b, u_b):
      return _phys_sm(d_b, u_b, mx)

    if neural_model is not None and neural_params is not None:
      def _e2e_local_np(d_b, u_b, mx_l, np_l):
        d_next = jax.vmap(lambda d, u: mjx.step(mx_l, d.replace(ctrl=u)))(d_b, u_b)
        gb = render_batch(spec, d_next, mode=mode, pallas_interpret=pallas_interp)
        rgb_out = apply_neural_shader(neural_model, np_l, gb, cal_scale=cal_scale)
        return d_next, rgb_out, gb.depth

      _e2e_sm_np = shard_map(
          _e2e_local_np,
          mesh=mesh,
          in_specs=(P("dp"), P("dp"), P(), P()),
          out_specs=(P("dp"), P("dp"), P("dp")),
          **sm_kw,
      )

      @jax.jit
      def e2e_step_render(d_b, u_b):
        return _e2e_sm_np(d_b, u_b, mx, neural_params)
    else:
      def _e2e_local(d_b, u_b, mx_l):
        d_next = jax.vmap(lambda d, u: mjx.step(mx_l, d.replace(ctrl=u)))(d_b, u_b)
        gb = render_batch(spec, d_next, mode=mode, pallas_interpret=pallas_interp)
        return d_next, gb.rgb, gb.depth

      _e2e_sm = shard_map(
          _e2e_local,
          mesh=mesh,
          in_specs=(P("dp"), P("dp"), P()),
          out_specs=(P("dp"), P("dp"), P("dp")),
          **sm_kw,
      )

      @jax.jit
      def e2e_step_render(d_b, u_b):
        return _e2e_sm(d_b, u_b, mx)
  else:
    @jax.jit
    def physics_only(d_b, u_b):
      return jax.vmap(lambda d, u: mjx.step(mx, d.replace(ctrl=u)))(d_b, u_b)

    @jax.jit
    def e2e_step_render(d_b, u_b):
      d_next = jax.vmap(lambda d, u: mjx.step(mx, d.replace(ctrl=u)))(d_b, u_b)
      gb = render_batch(spec, d_next, mode=mode, pallas_interpret=pallas_interp)
      if neural_model is not None and neural_params is not None:
        rgb_out = apply_neural_shader(neural_model, neural_params, gb, cal_scale=cal_scale)
      else:
        rgb_out = gb.rgb
      return d_next, rgb_out, gb.depth

  hlo_text = e2e_step_render.lower(dx_batch, ctrl).as_text()
  hlo_info = inspect_hlo_ops(hlo_text)
  assert hlo_info["host_callbacks"] == 0, f"HLO contains host_callback: {hlo_info}"

  # Warmup (includes XLA JIT compile)
  d_cur = dx_batch
  for _ in range(n_warmup):
    d_cur, rgb, depth = e2e_step_render(d_cur, ctrl)
  jax.block_until_ready((rgb, depth))

  d_p = dx_batch
  for _ in range(n_warmup):
    d_p = physics_only(d_p, ctrl)
  jax.block_until_ready(d_p.qpos)
  t0 = time.perf_counter()
  for _ in range(n_steps):
    d_p = physics_only(d_p, ctrl)
  jax.block_until_ready(d_p.qpos)
  dt_phys = (time.perf_counter() - t0) / n_steps

  t1 = time.perf_counter()
  for _ in range(n_steps):
    d_cur, rgb, depth = e2e_step_render(d_cur, ctrl)
  jax.block_until_ready((rgb, depth))
  dt_e2e = (time.perf_counter() - t1) / n_steps

  dt_render = max(dt_e2e - dt_phys, 0.0)
  fps_e2e = float(batch_size / max(dt_e2e, 1e-9))

  return {
      "scene": scene_name,
      "backend": jax.default_backend(),
      "device_kind": devs[0].device_kind,
      "n_chips": n_chips,
      "batch_size": batch_size,
      "resolution": f"{height}x{width}",
      "mode": mode,
      "neural_arch": neural_arch,
      "use_shadows": use_shadows,
      "dt_physics_ms": dt_phys * 1e3,
      "dt_render_ms": dt_render * 1e3,
      "dt_e2e_ms": dt_e2e * 1e3,
      "e2e_fps": fps_e2e,
      "hlo_zero_copy": hlo_info["host_callbacks"] == 0,
      "hlo_ops": hlo_info,
  }


def build_roofline_table(batches: List[int], resolutions: List[tuple[int, int]], n_geoms: int = 34):
  rows = []
  for chip_key, hw in TPU_SPECS.items():
    for B in batches:
      for H, W in resolutions:
        s1_naive = estimate_stage1_roofline(hw, B, H, W, n_geoms=n_geoms, pallas_vmem=False)
        s1_pallas = estimate_stage1_roofline(hw, B, H, W, n_geoms=n_geoms, pallas_vmem=True)
        s2_m1 = estimate_stage2_roofline(hw, B, H, W, arch="M1_pixel_mlp_128")
        s2_m2 = estimate_stage2_roofline(hw, B, H, W, arch="M2_spatial_128")
        t_stage1_ms = s1_pallas["t_stage1_ms"]
        t_m2_ms = s2_m2["t_stage2_ms"]
        rows.append({
            "chip": chip_key,
            "batch": B,
            "res": f"{H}x{W}",
            "s1_naive_ms": round(s1_naive["t_stage1_ms"], 3),
            "s1_pallas_ms": round(t_stage1_ms, 3),
            "s1_bound": s1_pallas["bound"],
            "s2_m1_ms": round(s2_m1["t_stage2_ms"], 3),
            "s2_m2_ms": round(t_m2_ms, 3),
            "s2_bound": s2_m2["bound"],
            "render_only_fps_s1_plus_m2": round(B / ((t_stage1_ms + t_m2_ms) * 1e-3), 0),
        })
  return rows


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--scene", default="panda_pick_cube", choices=["cartpole", "panda_pick_cube"])
  ap.add_argument("--batches", type=int, nargs="+", default=[1024, 2048, 4096])
  ap.add_argument("--res", type=int, nargs="+", default=[64])
  ap.add_argument("--mode", default="tiled", choices=["naive", "tiled", "pallas", "v2_soa", "pallas_v2"])
  ap.add_argument("--neural_ckpt", default=None)
  ap.add_argument("--n_chips", type=int, default=1)
  ap.add_argument("--steps", type=int, default=20)
  ap.add_argument("--out_json", default="out/metrics/bench.json")
  args = ap.parse_args()

  os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
  e2e_runs = []
  for B in args.batches:
    for r in args.res:
      res = benchmark_e2e(
          args.scene,
          B,
          r,
          r,
          mode=args.mode,
          n_steps=args.steps,
          neural_ckpt=args.neural_ckpt,
          n_chips=args.n_chips,
      )
      print(json.dumps(res), flush=True)
      e2e_runs.append(res)

  roofline = build_roofline_table([1024, 2048, 4096], [(64, 64), (128, 128)], n_geoms=34)
  out = {"e2e_runs": e2e_runs, "tpu_roofline_projections": roofline}
  with open(args.out_json, "w") as f:
    json.dump(out, f, indent=2)


if __name__ == "__main__":
  main()
