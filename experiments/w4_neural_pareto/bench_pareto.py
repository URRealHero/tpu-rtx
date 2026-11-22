"""End-to-End MJX + pallas_v2 + Stage 2 Neural Candidate Benchmark (1-chip & 4-chip shard_map)."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import pickle
import re
import time
from typing import Any, Dict, List, Optional

import jax
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np

from models import make_candidate_model
from tpu_mjx_render.neural_shader import apply_neural_shader
from tpu_mjx_render.renderer import RenderSceneSpec, render_batch
from tpu_mjx_render.roofline import inspect_hlo_ops
from tpu_mjx_render.scenes import get_scene_xml


def benchmark_e2e_candidate(
    scene_name: str,
    batch_size: int,
    height: int,
    width: int,
    mode: str = "pallas_v2",
    n_warmup: int = 3,
    n_steps: int = 20,
    use_shadows: bool = True,
    neural_ckpt: Optional[str] = None,
    n_chips: int = 1,
) -> Dict[str, Any]:
  m = mujoco.MjModel.from_xml_string(get_scene_xml(scene_name))
  mx = mjx.put_model(m, impl="jax")
  spec = RenderSceneSpec.from_mjmodel(m, cam_id=0, height=height, width=width, use_shadows=use_shadows)

  neural_model, neural_params, cal_scale, neural_arch = None, None, 1.0, None
  if neural_ckpt and os.path.exists(neural_ckpt):
    with open(neural_ckpt, "rb") as f:
      ck = pickle.load(f)
    neural_arch = ck["arch"]
    neural_model = make_candidate_model(neural_arch)
    neural_params = jax.tree_util.tree_map(jnp.asarray, ck["params"])
    cal_scale = float(ck["cal_scale"])

  devs = jax.devices()[:n_chips]
  assert len(devs) == n_chips, f"Requested {n_chips} chips, found {len(jax.devices())}"
  if n_chips > 1:
    assert batch_size % n_chips == 0
    mesh = jax.sharding.Mesh(np.array(devs), ("dp",))
    P = jax.sharding.PartitionSpec
    sharding = jax.sharding.NamedSharding(mesh, P("dp"))
    replicate = jax.sharding.NamedSharding(mesh, P())
  else:
    mesh, P, sharding, replicate = None, None, None, None

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
    if neural_params is not None:
      neural_params = jax.device_put(neural_params, devs[0])

  pallas_interp = (jax.default_backend() == "cpu")

  if n_chips > 1:
    sm_kw = {"check_rep": False} if "check_rep" in inspect.signature(shard_map).parameters else {"check_vma": False}

    def _phys_local(d_b, u_b, mx_l):
      return jax.vmap(lambda d, u: mjx.step(mx_l, d.replace(ctrl=u)))(d_b, u_b)

    _phys_sm = shard_map(_phys_local, mesh=mesh, in_specs=(P("dp"), P("dp"), P()), out_specs=P("dp"), **sm_kw)

    @jax.jit
    def physics_only(d_b, u_b):
      return _phys_sm(d_b, u_b, mx)

    def _s1_local(d_b, u_b, mx_l):
      d_next = jax.vmap(lambda d, u: mjx.step(mx_l, d.replace(ctrl=u)))(d_b, u_b)
      gb = render_batch(spec, d_next, mode=mode, pallas_interpret=pallas_interp)
      return d_next, gb.rgb, gb.depth

    _s1_sm = shard_map(
        _s1_local,
        mesh=mesh,
        in_specs=(P("dp"), P("dp"), P()),
        out_specs=(P("dp"), P("dp"), P("dp")),
        **sm_kw,
    )

    @jax.jit
    def s1_only(d_b, u_b):
      return _s1_sm(d_b, u_b, mx)

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
      e2e_step_render = s1_only
  else:
    @jax.jit
    def physics_only(d_b, u_b):
      return jax.vmap(lambda d, u: mjx.step(mx, d.replace(ctrl=u)))(d_b, u_b)

    @jax.jit
    def s1_only(d_b, u_b):
      d_next = jax.vmap(lambda d, u: mjx.step(mx, d.replace(ctrl=u)))(d_b, u_b)
      gb = render_batch(spec, d_next, mode=mode, pallas_interpret=pallas_interp)
      return d_next, gb.rgb, gb.depth

    if neural_model is not None and neural_params is not None:
      @jax.jit
      def e2e_step_render(d_b, u_b):
        d_next = jax.vmap(lambda d, u: mjx.step(mx, d.replace(ctrl=u)))(d_b, u_b)
        gb = render_batch(spec, d_next, mode=mode, pallas_interpret=pallas_interp)
        rgb_out = apply_neural_shader(neural_model, neural_params, gb, cal_scale=cal_scale)
        return d_next, rgb_out, gb.depth
    else:
      e2e_step_render = s1_only

  hlo_text = e2e_step_render.lower(dx_batch, ctrl).as_text()
  hlo_info = inspect_hlo_ops(hlo_text)
  hlo_info["send_recv_ops"] = len(re.findall(r"\b(send|recv|send-done|recv-done)\(", hlo_text))
  assert hlo_info["host_callbacks"] == 0, f"HLO contains host_callback: {hlo_info}"
  assert hlo_info["send_recv_ops"] == 0, f"HLO contains send/recv: {hlo_info}"

  # Warmup & time physics_only
  d_p = dx_batch
  for _ in range(n_warmup):
    d_p = physics_only(d_p, ctrl)
  jax.block_until_ready(d_p.qpos)
  t0 = time.perf_counter()
  for _ in range(n_steps):
    d_p = physics_only(d_p, ctrl)
  jax.block_until_ready(d_p.qpos)
  dt_phys = (time.perf_counter() - t0) / n_steps

  # Warmup & time s1_only (physics + Stage 1)
  d_s1 = dx_batch
  for _ in range(n_warmup):
    d_s1, rgb1, dep1 = s1_only(d_s1, ctrl)
  jax.block_until_ready((rgb1, dep1))
  t_s1 = time.perf_counter()
  for _ in range(n_steps):
    d_s1, rgb1, dep1 = s1_only(d_s1, ctrl)
  jax.block_until_ready((rgb1, dep1))
  dt_phys_s1 = (time.perf_counter() - t_s1) / n_steps

  # Warmup & time full E2E (physics + Stage 1 + Stage 2)
  d_cur = dx_batch
  for _ in range(n_warmup):
    d_cur, rgb, depth = e2e_step_render(d_cur, ctrl)
  jax.block_until_ready((rgb, depth))
  t1 = time.perf_counter()
  for _ in range(n_steps):
    d_cur, rgb, depth = e2e_step_render(d_cur, ctrl)
  jax.block_until_ready((rgb, depth))
  dt_e2e = (time.perf_counter() - t1) / n_steps

  dt_s1 = max(dt_phys_s1 - dt_phys, 0.0)
  dt_s2 = max(dt_e2e - dt_phys_s1, 0.0) if neural_model is not None else 0.0
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
      "neural_arch": neural_arch or "stage1_only",
      "use_shadows": use_shadows,
      "dt_physics_ms": round(dt_phys * 1e3, 3),
      "dt_stage1_ms": round(dt_s1 * 1e3, 3),
      "dt_stage2_ms": round(dt_s2 * 1e3, 3),
      "dt_render_ms": round(dt_render * 1e3, 3),
      "dt_e2e_ms": round(dt_e2e * 1e3, 3),
      "e2e_fps": round(fps_e2e, 2),
      "hlo_zero_copy": (hlo_info["host_callbacks"] == 0 and hlo_info["send_recv_ops"] == 0),
      "hlo_ops": hlo_info,
  }


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--ckpts", nargs="*", default=[])
  ap.add_argument("--include_stage1_only", action="store_true")
  ap.add_argument("--batches", type=int, nargs="+", default=[1024, 2048, 4096])
  ap.add_argument("--res", type=int, default=64)
  ap.add_argument("--n_chips", type=int, default=1)
  ap.add_argument("--steps", type=int, default=20)
  ap.add_argument("--out_json", required=True)
  args = ap.parse_args()

  os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
  runs: List[Dict[str, Any]] = []
  targets: List[Optional[str]] = []
  if args.include_stage1_only:
    targets.append(None)
  targets.extend(args.ckpts)

  for ckpt in targets:
    for B in args.batches:
      rec = benchmark_e2e_candidate(
          scene_name="panda_pick_cube",
          batch_size=B,
          height=args.res,
          width=args.res,
          mode="pallas_v2",
          n_steps=args.steps,
          neural_ckpt=ckpt,
          n_chips=args.n_chips,
      )
      print(json.dumps(rec), flush=True)
      runs.append(rec)

  with open(args.out_json, "w") as f:
    json.dump({"n_chips": args.n_chips, "runs": runs}, f, indent=2)


if __name__ == "__main__":
  main()
