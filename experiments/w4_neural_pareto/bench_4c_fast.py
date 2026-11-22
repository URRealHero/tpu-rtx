"""Fast 4-chip shard_map Stage 2 & E2E benchmark (compiles physics & Stage 1 only once)."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import pickle
import re
import time
from typing import Any, Dict, List

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


def _hlo_audit(lowered) -> Dict[str, Any]:
  txt = lowered.as_text()
  info = inspect_hlo_ops(txt)
  info["send_recv_ops"] = len(re.findall(r"\b(send|recv|send-done|recv-done)\(", txt))
  assert info["host_callbacks"] == 0, info
  assert info["send_recv_ops"] == 0, info
  return info


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--ckpts", nargs="+", required=True)
  ap.add_argument("--batch_size", type=int, default=4096)
  ap.add_argument("--res", type=int, default=64)
  ap.add_argument("--steps", type=int, default=20)
  ap.add_argument("--out_json", required=True)
  args = ap.parse_args()

  devs = jax.devices()
  n_chips = len(devs)
  assert n_chips == 4, f"Expected 4 chips, got {n_chips}"
  B = args.batch_size
  H = W = args.res
  assert B % n_chips == 0

  mesh = jax.sharding.Mesh(np.array(devs), ("dp",))
  P = jax.sharding.PartitionSpec
  sharding = jax.sharding.NamedSharding(mesh, P("dp"))
  replicate = jax.sharding.NamedSharding(mesh, P())
  sm_kw = {"check_rep": False} if "check_rep" in inspect.signature(shard_map).parameters else {"check_vma": False}

  # 1. Pure Stage 2 4-chip shard_map throughput (B=4096)
  x_dummy = jax.device_put(np.zeros((B, H, W, 24), dtype=np.float32), sharding)
  b_dummy = jax.device_put(np.zeros((B, H, W, 3), dtype=np.float32), sharding)

  stage2_4c: List[Dict[str, Any]] = []
  loaded: List[tuple] = []
  for ck_path in args.ckpts:
    with open(ck_path, "rb") as f:
      ck = pickle.load(f)
    arch = ck["arch"]
    model = make_candidate_model(arch)
    params_rep = jax.device_put(jax.tree_util.tree_map(jnp.asarray, ck["params"]), replicate)
    cal_scale = float(ck["cal_scale"])
    loaded.append((arch, model, params_rep, cal_scale))

    def _s2_loc(p_l, x_l, b_l, _m=model, _cs=cal_scale):
      return jnp.clip((b_l + _m.apply({"params": p_l}, x_l, train=False)) * _cs, 0.0, 1.0)

    _s2_sm = shard_map(_s2_loc, mesh=mesh, in_specs=(P(), P("dp"), P("dp")), out_specs=P("dp"), **sm_kw)

    @jax.jit
    def fwd_4c(p, x_in, b_in):
      return _s2_sm(p, x_in, b_in)

    hlo = _hlo_audit(fwd_4c.lower(params_rep, x_dummy, b_dummy))
    for _ in range(3):
      y = fwd_4c(params_rep, x_dummy, b_dummy)
    jax.block_until_ready(y)
    t0 = time.perf_counter()
    for _ in range(30):
      y = fwd_4c(params_rep, x_dummy, b_dummy)
    jax.block_until_ready(y)
    dt_ms = (time.perf_counter() - t0) * 1e3 / 30.0
    rec = {
        "arch": arch,
        "n_chips": n_chips,
        "batch_size": B,
        "stage2_ms": round(dt_ms, 3),
        "stage2_only_fps": round(B * 1e3 / dt_ms, 1),
        "hlo_zero_copy": True,
        "hlo_ops": hlo,
    }
    print("S2_4C:", json.dumps(rec), flush=True)
    stage2_4c.append(rec)

  # 2. E2E (MJX step + pallas_v2 + Stage 2) 4-chip shard_map (physics & S1 compiled once)
  m = mujoco.MjModel.from_xml_string(get_scene_xml("panda_pick_cube"))
  mx = jax.device_put(mjx.put_model(m, impl="jax"), replicate)
  spec = RenderSceneSpec.from_mjmodel(m, cam_id=0, height=H, width=W, use_shadows=True)

  dx0 = mjx.make_data(m, impl="jax")
  d_host = mujoco.MjData(m)
  mujoco.mj_resetDataKeyframe(m, d_host, 0)
  dx0 = dx0.replace(qpos=jnp.asarray(d_host.qpos, jnp.float32))
  dx0 = mjx.forward(mjx.put_model(m, impl="jax"), dx0)
  dx_batch = jax.device_put(
      jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x[None, ...], (B, *x.shape)).copy(), dx0),
      sharding,
  )
  ctrl = jax.device_put(jnp.zeros((B, m.nu), dtype=jnp.float32), sharding)

  def _phys_loc(d_b, u_b, mx_l):
    return jax.vmap(lambda d, u: mjx.step(mx_l, d.replace(ctrl=u)))(d_b, u_b)

  _phys_sm = shard_map(_phys_loc, mesh=mesh, in_specs=(P("dp"), P("dp"), P()), out_specs=P("dp"), **sm_kw)

  @jax.jit
  def physics_only(d_b, u_b):
    return _phys_sm(d_b, u_b, mx)

  def _s1_loc(d_b, u_b, mx_l):
    d_next = jax.vmap(lambda d, u: mjx.step(mx_l, d.replace(ctrl=u)))(d_b, u_b)
    gb = render_batch(spec, d_next, mode="pallas_v2", pallas_interpret=False)
    return d_next, gb.rgb, gb.depth

  _s1_sm = shard_map(
      _s1_loc, mesh=mesh, in_specs=(P("dp"), P("dp"), P()), out_specs=(P("dp"), P("dp"), P("dp")), **sm_kw
  )

  @jax.jit
  def s1_only(d_b, u_b):
    return _s1_sm(d_b, u_b, mx)

  s1_hlo = _hlo_audit(s1_only.lower(dx_batch, ctrl))

  d_p = dx_batch
  for _ in range(3):
    d_p = physics_only(d_p, ctrl)
  jax.block_until_ready(d_p.qpos)
  t0 = time.perf_counter()
  for _ in range(args.steps):
    d_p = physics_only(d_p, ctrl)
  jax.block_until_ready(d_p.qpos)
  dt_phys = (time.perf_counter() - t0) / args.steps

  d_s1 = dx_batch
  for _ in range(3):
    d_s1, r1, d1 = s1_only(d_s1, ctrl)
  jax.block_until_ready((r1, d1))
  t1 = time.perf_counter()
  for _ in range(args.steps):
    d_s1, r1, d1 = s1_only(d_s1, ctrl)
  jax.block_until_ready((r1, d1))
  dt_phys_s1 = (time.perf_counter() - t1) / args.steps
  dt_s1 = max(dt_phys_s1 - dt_phys, 0.0)

  e2e_4c: List[Dict[str, Any]] = [{
      "neural_arch": "stage1_only",
      "n_chips": n_chips,
      "batch_size": B,
      "resolution": f"{H}x{W}",
      "dt_physics_ms": round(dt_phys * 1e3, 3),
      "dt_stage1_ms": round(dt_s1 * 1e3, 3),
      "dt_stage2_ms": 0.0,
      "dt_e2e_ms": round(dt_phys_s1 * 1e3, 3),
      "e2e_fps": round(B / dt_phys_s1, 2),
      "hlo_zero_copy": True,
      "hlo_ops": s1_hlo,
  }]
  print("E2E_4C:", json.dumps(e2e_4c[0]), flush=True)

  # Run E2E on first 4 checkpoints (teacher + top 3 Pareto models)
  for arch, model, params_rep, cal_scale in loaded[:4]:
    def _e2e_loc(d_b, u_b, mx_l, p_l, _m=model, _cs=cal_scale):
      d_next = jax.vmap(lambda d, u: mjx.step(mx_l, d.replace(ctrl=u)))(d_b, u_b)
      gb = render_batch(spec, d_next, mode="pallas_v2", pallas_interpret=False)
      rgb_out = apply_neural_shader(_m, p_l, gb, cal_scale=_cs)
      return d_next, rgb_out, gb.depth

    _e2e_sm = shard_map(
        _e2e_loc, mesh=mesh, in_specs=(P("dp"), P("dp"), P(), P()), out_specs=(P("dp"), P("dp"), P("dp")), **sm_kw
    )

    @jax.jit
    def e2e_fn(d_b, u_b, p):
      return _e2e_sm(d_b, u_b, mx, p)

    hlo = _hlo_audit(e2e_fn.lower(dx_batch, ctrl, params_rep))
    d_c = dx_batch
    for _ in range(3):
      d_c, rc, dc = e2e_fn(d_c, ctrl, params_rep)
    jax.block_until_ready((rc, dc))
    t2 = time.perf_counter()
    for _ in range(args.steps):
      d_c, rc, dc = e2e_fn(d_c, ctrl, params_rep)
    jax.block_until_ready((rc, dc))
    dt_e2e = (time.perf_counter() - t2) / args.steps
    rec = {
        "neural_arch": arch,
        "n_chips": n_chips,
        "batch_size": B,
        "resolution": f"{H}x{W}",
        "dt_physics_ms": round(dt_phys * 1e3, 3),
        "dt_stage1_ms": round(dt_s1 * 1e3, 3),
        "dt_stage2_ms": round(max(dt_e2e - dt_phys_s1, 0.0) * 1e3, 3),
        "dt_e2e_ms": round(dt_e2e * 1e3, 3),
        "e2e_fps": round(B / dt_e2e, 2),
        "hlo_zero_copy": True,
        "hlo_ops": hlo,
    }
    print("E2E_4C:", json.dumps(rec), flush=True)
    e2e_4c.append(rec)

  os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
  with open(args.out_json, "w") as f:
    json.dump({"n_chips": n_chips, "stage2_4c": stage2_4c, "e2e_4c": e2e_4c}, f, indent=2)
  print("Saved 4-chip benchmark ->", args.out_json, flush=True)


if __name__ == "__main__":
  main()
