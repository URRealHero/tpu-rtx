"""V3 High-Resolution (128/256/512) & Fused Multi-Camera TPU v5p Benchmark.

Runs on live TPU v5p (interpret=False) with zero host callbacks (host_callbacks == 0,
send_recv_ops == 0) across 1-chip and 4-chip shard_map configurations.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import pickle
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import jax
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np

from experiments.w6_highres_multicam.dual_cam import (
    make_dual_cam_panda_xml,
    pallas_raycast_dual_cam_v2,
)
from tpu_mjx_render.neural_shader import apply_neural_shader, make_neural_shader_model
from tpu_mjx_render.renderer import RenderSceneSpec, render_batch
from tpu_mjx_render.roofline import inspect_hlo_ops
from tpu_mjx_render.scenes import get_scene_xml


def _audit_hlo(lowered: Any) -> Dict[str, Any]:
  txt = lowered.as_text()
  info = inspect_hlo_ops(txt)
  info["send_recv_ops"] = len(re.findall(r"\bsend\(|\brecv\(", txt, re.I))
  assert info["host_callbacks"] == 0, f"HLO contains host_callback: {info}"
  assert info["send_recv_ops"] == 0, f"HLO contains send/recv: {info}"
  info["zero_copy_verified"] = True
  return info


def _load_neural_ckpt(ckpt_path: Optional[str]):
  if not ckpt_path or not os.path.exists(ckpt_path):
    return None, None, 1.0, None
  with open(ckpt_path, "rb") as f:
    ck = pickle.load(f)
  arch = ck["arch"]
  model = make_neural_shader_model(arch)
  params = jax.tree_util.tree_map(jnp.asarray, ck["params"])
  cal_scale = float(ck["cal_scale"])
  return model, params, cal_scale, arch


def _make_initial_batch(m: mujoco.MjModel, mx: Any, batch_size: int, vary_ctrl: bool = False):
  del mx
  d_host = mujoco.MjData(m)
  if m.nkey > 0:
    mujoco.mj_resetDataKeyframe(m, d_host, 0)
  mujoco.mj_forward(m, d_host)
  dx0 = mjx.put_data(m, d_host, impl="jax")
  dx_batch = jax.tree_util.tree_map(
      lambda x: jnp.broadcast_to(x[None, ...], (batch_size, *x.shape)).copy(), dx0
  )
  if vary_ctrl and m.nu > 0:
    phases = jnp.linspace(-0.35, 0.35, batch_size, dtype=jnp.float32)[:, None]
    base_ctrl = jnp.asarray(dx0.ctrl, jnp.float32)[None, :]
    ctrl = base_ctrl + phases * jnp.linspace(1.0, 0.3, m.nu, dtype=jnp.float32)[None, :]
    if m.actuator_ctrllimited is not None and np.any(m.actuator_ctrllimited):
      lo = jnp.asarray(m.actuator_ctrlrange[:, 0], jnp.float32)[None, :]
      hi = jnp.asarray(m.actuator_ctrlrange[:, 1], jnp.float32)[None, :]
      lim = jnp.asarray(m.actuator_ctrllimited, jnp.bool_)[None, :]
      ctrl = jnp.where(lim, jnp.clip(ctrl, lo, hi), ctrl)
  else:
    ctrl = jnp.zeros((batch_size, m.nu), dtype=jnp.float32)
  return dx_batch, ctrl


def _shard_setup(n_chips: int, batch_size: int):
  devs = jax.devices()[:n_chips]
  if len(devs) < n_chips:
    raise RuntimeError(f"Requested {n_chips} chips, only {len(devs)} visible")
  if n_chips > 1:
    assert batch_size % n_chips == 0
    mesh = jax.sharding.Mesh(np.array(devs), ("dp",))
    P = jax.sharding.PartitionSpec
    sharding = jax.sharding.NamedSharding(mesh, P("dp"))
    replicate = jax.sharding.NamedSharding(mesh, P())
    sm_kw = {"check_rep": False} if "check_rep" in inspect.signature(shard_map).parameters else {"check_vma": False}
    return devs, mesh, P, sharding, replicate, sm_kw
  return devs, None, None, None, None, None


class _PhysicsCache:
  """Measure MJX step once per (scene_key, batch_size, n_chips) to avoid redundant XLA compiles."""

  def __init__(self):
    self._cache: Dict[Tuple[str, int, int], float] = {}

  def get_dt_phys(
      self, key: Tuple[str, int, int], mx: Any, dx_batch: Any, ctrl: Any, n_chips: int,
      mesh=None, P=None, sm_kw=None, n_warmup: int = 2, n_steps: int = 10,
  ) -> float:
    if key in self._cache:
      return self._cache[key]
    if n_chips > 1:
      def _phys_local(d_b, u_b, mx_l):
        return jax.vmap(lambda d, u: mjx.step(mx_l, d.replace(ctrl=u)))(d_b, u_b)
      _phys_sm = shard_map(_phys_local, mesh=mesh, in_specs=(P("dp"), P("dp"), P()), out_specs=P("dp"), **sm_kw)

      @jax.jit
      def physics_only(d_b, u_b):
        return _phys_sm(d_b, u_b, mx)
    else:
      @jax.jit
      def physics_only(d_b, u_b):
        return jax.vmap(lambda d, u: mjx.step(mx, d.replace(ctrl=u)))(d_b, u_b)

    d_p = dx_batch
    for _ in range(n_warmup):
      d_p = physics_only(d_p, ctrl)
    jax.block_until_ready(d_p.qpos)
    t0 = time.perf_counter()
    for _ in range(n_steps):
      d_p = physics_only(d_p, ctrl)
    jax.block_until_ready(d_p.qpos)
    dt_phys = (time.perf_counter() - t0) / n_steps
    self._cache[key] = dt_phys
    return dt_phys


def run_parity_suite() -> Dict[str, Any]:
  """Verify dual-cam wrist motion, exact parity vs 2x pallas_v2 & v2_soa, and HLO zero-copy."""
  pallas_interp = (jax.default_backend() == "cpu")
  xml_dual = make_dual_cam_panda_xml()
  m = mujoco.MjModel.from_xml_string(xml_dual)
  mx = mjx.put_model(m, impl="jax")
  cam0_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "front")
  cam1_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "wrist")
  assert cam0_id == 0 and cam1_id == 1, (cam0_id, cam1_id)

  B = 16
  dx_batch, ctrl = _make_initial_batch(m, mx, B, vary_ctrl=True)
  devs = jax.devices()[:1]
  dx_batch = jax.device_put(dx_batch, devs[0])
  ctrl = jax.device_put(ctrl, devs[0])

  @jax.jit
  def step_n(d_b, u_b):
    def body(_, d):
      return jax.vmap(lambda x, u: mjx.step(mx, x.replace(ctrl=u)))(d, u_b)
    return jax.lax.fori_loop(0, 12, body, d_b)

  dx_moved = step_n(dx_batch, ctrl)
  jax.block_until_ready(dx_moved.cam_xpos)

  cam0_shift = float(jnp.max(jnp.abs(dx_moved.cam_xpos[:, cam0_id] - dx_batch.cam_xpos[:, cam0_id])))
  cam1_shift = float(jnp.max(jnp.abs(dx_moved.cam_xpos[:, cam1_id] - dx_batch.cam_xpos[:, cam1_id])))
  cam1_batch_spread = float(jnp.max(jnp.std(dx_moved.cam_xpos[:, cam1_id], axis=0)))
  assert cam1_shift > 1e-3 and cam1_batch_spread > 1e-4, (
      f"Wrist cam did not move dynamically: shift={cam1_shift}, spread={cam1_batch_spread}"
  )

  parity_res: Dict[str, Any] = {
      "backend": jax.default_backend(),
      "device_kind": devs[0].device_kind,
      "interpret": pallas_interp,
      "cam0_name": "front",
      "cam1_name": "wrist",
      "cam0_id": cam0_id,
      "cam1_id": cam1_id,
      "cam0_pos_shift_m": cam0_shift,
      "cam1_wrist_pos_shift_m": cam1_shift,
      "cam1_wrist_batch_spread_m": cam1_batch_spread,
      "resolutions": {},
  }

  for res in (64, 128):
    spec0 = RenderSceneSpec.from_mjmodel(m, cam_id=cam0_id, height=res, width=res, use_shadows=True)
    spec1 = RenderSceneSpec.from_mjmodel(m, cam_id=cam1_id, height=res, width=res, use_shadows=True)

    fn_sep0 = jax.jit(lambda d: render_batch(spec0, d, mode="pallas_v2", pallas_interpret=pallas_interp))
    fn_sep1 = jax.jit(lambda d: render_batch(spec1, d, mode="pallas_v2", pallas_interpret=pallas_interp))
    fn_soa0 = jax.jit(lambda d: render_batch(spec0, d, mode="v2_soa"))
    fn_soa1 = jax.jit(lambda d: render_batch(spec1, d, mode="v2_soa"))
    fn_fused = jax.jit(lambda d: pallas_raycast_dual_cam_v2(spec0, spec1, d, interpret=pallas_interp))

    hlo_fused = _audit_hlo(fn_fused.lower(dx_moved))

    ref0 = fn_sep0(dx_moved)
    ref1 = fn_sep1(dx_moved)
    soa0 = fn_soa0(dx_moved)
    soa1 = fn_soa1(dx_moved)
    fus0, fus1 = fn_fused(dx_moved)
    jax.block_until_ready((fus0.rgb, fus1.rgb))

    def _cmp(a, b):
      return {
          "depth_mae_m": float(jnp.mean(jnp.abs(a.depth - b.depth))),
          "max_depth_diff_m": float(jnp.max(jnp.abs(a.depth - b.depth))),
          "max_rgb_diff": float(jnp.max(jnp.abs(a.rgb - b.rgb))),
          "geom_match": float(jnp.mean(a.geom_id == b.geom_id)),
      }

    m0_vs_pallas = _cmp(fus0, ref0)
    m1_vs_pallas = _cmp(fus1, ref1)
    m0_vs_soa = _cmp(fus0, soa0)
    m1_vs_soa = _cmp(fus1, soa1)

    assert m0_vs_pallas["depth_mae_m"] < 1e-5 and m0_vs_pallas["geom_match"] == 1.0, m0_vs_pallas
    assert m1_vs_pallas["depth_mae_m"] < 1e-5 and m1_vs_pallas["geom_match"] == 1.0, m1_vs_pallas
    assert m0_vs_soa["depth_mae_m"] < 1e-5 and m0_vs_soa["geom_match"] == 1.0, m0_vs_soa
    assert m1_vs_soa["depth_mae_m"] < 1e-5 and m1_vs_soa["geom_match"] == 1.0, m1_vs_soa

    parity_res["resolutions"][f"{res}x{res}"] = {
        "cam0_vs_pallas_v2": m0_vs_pallas,
        "cam1_vs_pallas_v2": m1_vs_pallas,
        "cam0_vs_v2_soa": m0_vs_soa,
        "cam1_vs_v2_soa": m1_vs_soa,
        "cam0_fg_frac": float(jnp.mean(fus0.geom_id >= 0)),
        "cam1_wrist_fg_frac": float(jnp.mean(fus1.geom_id >= 0)),
        "hlo_fused": hlo_fused,
        "passed": True,
    }
    print(f"PARITY {res}x{res} PASS: cam0={m0_vs_pallas}, cam1={m1_vs_pallas}", flush=True)

  spec256 = RenderSceneSpec.from_mjmodel(m, cam_id=cam0_id, height=256, width=256, use_shadows=True)
  fn_256_soa = jax.jit(lambda d: render_batch(spec256, d, mode="v2_soa"))
  fn_256_pal = jax.jit(lambda d: render_batch(spec256, d, mode="pallas_v2", pallas_interpret=pallas_interp))
  g256_s = fn_256_soa(dx_moved)
  g256_p = fn_256_pal(dx_moved)
  jax.block_until_ready(g256_p.rgb)
  p256 = {
      "depth_mae_m": float(jnp.mean(jnp.abs(g256_p.depth - g256_s.depth))),
      "max_depth_diff_m": float(jnp.max(jnp.abs(g256_p.depth - g256_s.depth))),
      "max_rgb_diff": float(jnp.max(jnp.abs(g256_p.rgb - g256_s.rgb))),
      "geom_match": float(jnp.mean(g256_p.geom_id == g256_s.geom_id)),
      "hlo_pallas_v2": _audit_hlo(fn_256_pal.lower(dx_moved)),
  }
  assert p256["depth_mae_m"] < 1e-5 and p256["geom_match"] == 1.0, p256
  parity_res["resolutions"]["256x256_pallas_v2_vs_soa"] = p256
  print("PARITY 256x256 PASS:", p256, flush=True)
  return parity_res


def _bench_one_highres(
    phys_cache: _PhysicsCache,
    scene_name: str,
    batch_size: int,
    res: int,
    n_chips: int,
    neural_ckpt: Optional[str] = None,
    n_warmup: int = 1,
    n_steps: int = 6,
) -> Dict[str, Any]:
  pallas_interp = (jax.default_backend() == "cpu")
  m = mujoco.MjModel.from_xml_string(get_scene_xml(scene_name))
  mx = mjx.put_model(m, impl="jax")
  spec = RenderSceneSpec.from_mjmodel(m, cam_id=0, height=res, width=res, use_shadows=True)
  neural_model, neural_params, cal_scale, neural_arch = _load_neural_ckpt(neural_ckpt)

  devs, mesh, P, sharding, replicate, sm_kw = _shard_setup(n_chips, batch_size)
  dx_batch, ctrl = _make_initial_batch(m, mx, batch_size, vary_ctrl=False)

  if n_chips > 1:
    dx_batch = jax.device_put(dx_batch, sharding)
    ctrl = jax.device_put(ctrl, sharding)
    mx = jax.device_put(mx, replicate)
    if neural_params is not None:
      neural_params = jax.device_put(neural_params, replicate)

    if neural_model is not None and neural_params is not None:
      def _e2e_local_np(d_b, u_b, mx_l, np_l):
        d_next = jax.vmap(lambda d, u: mjx.step(mx_l, d.replace(ctrl=u)))(d_b, u_b)
        gb = render_batch(spec, d_next, mode="pallas_v2", pallas_interpret=pallas_interp)
        rgb_out = apply_neural_shader(neural_model, np_l, gb, cal_scale=cal_scale)
        return d_next, rgb_out, gb.depth

      _e2e_sm = shard_map(
          _e2e_local_np, mesh=mesh, in_specs=(P("dp"), P("dp"), P(), P()), out_specs=(P("dp"), P("dp"), P("dp")), **sm_kw
      )

      @jax.jit
      def e2e_step_render(d_b, u_b):
        return _e2e_sm(d_b, u_b, mx, neural_params)
    else:
      def _e2e_local(d_b, u_b, mx_l):
        d_next = jax.vmap(lambda d, u: mjx.step(mx_l, d.replace(ctrl=u)))(d_b, u_b)
        gb = render_batch(spec, d_next, mode="pallas_v2", pallas_interpret=pallas_interp)
        return d_next, gb.rgb, gb.depth

      _e2e_sm = shard_map(
          _e2e_local, mesh=mesh, in_specs=(P("dp"), P("dp"), P()), out_specs=(P("dp"), P("dp"), P("dp")), **sm_kw
      )

      @jax.jit
      def e2e_step_render(d_b, u_b):
        return _e2e_sm(d_b, u_b, mx)
  else:
    dx_batch = jax.device_put(dx_batch, devs[0])
    ctrl = jax.device_put(ctrl, devs[0])

    @jax.jit
    def e2e_step_render(d_b, u_b):
      d_next = jax.vmap(lambda d, u: mjx.step(mx, d.replace(ctrl=u)))(d_b, u_b)
      gb = render_batch(spec, d_next, mode="pallas_v2", pallas_interpret=pallas_interp)
      rgb_out = apply_neural_shader(neural_model, neural_params, gb, cal_scale=cal_scale) if neural_model is not None else gb.rgb
      return d_next, rgb_out, gb.depth

  hlo_info = _audit_hlo(e2e_step_render.lower(dx_batch, ctrl))

  dt_phys = phys_cache.get_dt_phys(
      (scene_name, batch_size, n_chips), mx, dx_batch, ctrl, n_chips,
      mesh=mesh, P=P, sm_kw=sm_kw, n_warmup=2, n_steps=10,
  )

  tc0 = time.perf_counter()
  d_cur = dx_batch
  for _ in range(n_warmup):
    d_cur, rgb, depth = e2e_step_render(d_cur, ctrl)
  jax.block_until_ready((rgb, depth))
  warmup_s = time.perf_counter() - tc0

  t1 = time.perf_counter()
  for _ in range(n_steps):
    d_cur, rgb, depth = e2e_step_render(d_cur, ctrl)
  jax.block_until_ready((rgb, depth))
  dt_e2e = (time.perf_counter() - t1) / n_steps

  dt_render = max(dt_e2e - dt_phys, 0.0)
  return {
      "status": "ok",
      "scene": scene_name,
      "backend": jax.default_backend(),
      "device_kind": devs[0].device_kind,
      "n_chips": n_chips,
      "batch_size": batch_size,
      "resolution": f"{res}x{res}",
      "mode": "pallas_v2",
      "neural_arch": neural_arch,
      "dt_physics_ms": dt_phys * 1e3,
      "dt_render_ms": dt_render * 1e3,
      "dt_e2e_ms": dt_e2e * 1e3,
      "e2e_fps": float(batch_size / max(dt_e2e, 1e-9)),
      "warmup_s": round(warmup_s, 3),
      "host_callbacks": hlo_info["host_callbacks"],
      "send_recv_ops": hlo_info["send_recv_ops"],
      "hlo_zero_copy": True,
      "hlo_ops": hlo_info,
  }


def run_highres_suite(
    n_chips: int,
    neural_ckpt: Optional[str],
    scenes: List[str],
    resolutions: List[int],
    batches: List[int],
    out_json: str,
    mode_filter: str = "all",
) -> List[Dict[str, Any]]:
  """Benchmark 128 (+M2), 256 Stage 1 (+M2 where fits), and 512 Stage 1 on PandaPickCube & Cartpole."""
  configs: List[Tuple[str, int, Optional[str], int, int]] = []
  for scene in scenes:
    if 128 in resolutions:
      if mode_filter in ("all", "s1"):
        configs.append((scene, 128, None, 2, 10))
      if mode_filter in ("all", "m2") and neural_ckpt and os.path.exists(neural_ckpt):
        configs.append((scene, 128, neural_ckpt, 2, 10))
    if 256 in resolutions:
      if mode_filter in ("all", "s1"):
        configs.append((scene, 256, None, 2, 5))
      if mode_filter in ("all", "m2") and scene == "panda_pick_cube" and neural_ckpt and os.path.exists(neural_ckpt):
        configs.append((scene, 256, neural_ckpt, 2, 4))
    if 512 in resolutions:
      if mode_filter in ("all", "s1"):
        configs.append((scene, 512, None, 2, 3))

  phys_cache = _PhysicsCache()
  rows: List[Dict[str, Any]] = []
  for scene, res, ckpt, nw, ns in configs:
    for B in batches:
      ns_use = 2 if (res == 512 and B >= 1024 and n_chips == 1) else ns
      tag = f"{scene} {res}x{res} B={B} {n_chips}c {'M2' if ckpt else 'S1'}"
      try:
        rec = _bench_one_highres(phys_cache, scene, B, res, n_chips, neural_ckpt=ckpt, n_warmup=nw, n_steps=ns_use)
        print(
            f"HIGHRES {tag}: phys={rec['dt_physics_ms']:.2f}ms "
            f"rend={rec['dt_render_ms']:.2f}ms e2e={rec['dt_e2e_ms']:.2f}ms "
            f"fps={rec['e2e_fps']:.1f} (warmup={rec['warmup_s']}s)",
            flush=True,
        )
      except Exception as exc:
        rec = {
            "status": "error",
            "scene": scene,
            "n_chips": n_chips,
            "batch_size": B,
            "resolution": f"{res}x{res}",
            "neural_arch": "M2_spatial_128" if ckpt else None,
            "error": f"{type(exc).__name__}: {str(exc)[:300]}",
        }
        print(f"HIGHRES {tag} ERROR: {rec['error']}", flush=True)
      rows.append(rec)
      with open(out_json, "w") as f:
        json.dump({"n_chips": n_chips, "highres_runs": rows}, f, indent=2)
  return rows


def _bench_one_multicam(
    phys_cache: _PhysicsCache,
    pipeline: str,
    batch_size: int,
    res: int,
    n_chips: int,
    n_warmup: int = 1,
    n_steps: int = 8,
) -> Dict[str, Any]:
  pallas_interp = (jax.default_backend() == "cpu")
  xml_dual = make_dual_cam_panda_xml()
  m = mujoco.MjModel.from_xml_string(xml_dual)
  mx = mjx.put_model(m, impl="jax")
  cam0_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "front")
  cam1_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "wrist")
  spec0 = RenderSceneSpec.from_mjmodel(m, cam_id=cam0_id, height=res, width=res, use_shadows=True)
  spec1 = RenderSceneSpec.from_mjmodel(m, cam_id=cam1_id, height=res, width=res, use_shadows=True)

  devs, mesh, P, sharding, replicate, sm_kw = _shard_setup(n_chips, batch_size)
  dx_batch, ctrl = _make_initial_batch(m, mx, batch_size, vary_ctrl=True)

  def _render_pair(d_next):
    if pipeline == "single_cam":
      gb0 = render_batch(spec0, d_next, mode="pallas_v2", pallas_interpret=pallas_interp)
      return (gb0.rgb, gb0.depth)
    if pipeline == "dual_cam_seq":
      gb0 = render_batch(spec0, d_next, mode="pallas_v2", pallas_interpret=pallas_interp)
      gb1 = render_batch(spec1, d_next, mode="pallas_v2", pallas_interpret=pallas_interp)
      return (gb0.rgb, gb0.depth, gb1.rgb, gb1.depth)
    if pipeline == "dual_cam_fused":
      gb0, gb1 = pallas_raycast_dual_cam_v2(spec0, spec1, d_next, interpret=pallas_interp)
      return (gb0.rgb, gb0.depth, gb1.rgb, gb1.depth)
    raise ValueError(f"Unknown pipeline: {pipeline}")

  n_cams = 1 if pipeline == "single_cam" else 2

  if n_chips > 1:
    dx_batch = jax.device_put(dx_batch, sharding)
    ctrl = jax.device_put(ctrl, sharding)
    mx = jax.device_put(mx, replicate)

    def _e2e_local(d_b, u_b, mx_l):
      d_next = jax.vmap(lambda d, u: mjx.step(mx_l, d.replace(ctrl=u)))(d_b, u_b)
      outs = _render_pair(d_next)
      return (d_next, *outs)

    out_ps = (P("dp"),) * (1 + 2 * n_cams)
    _e2e_sm = shard_map(_e2e_local, mesh=mesh, in_specs=(P("dp"), P("dp"), P()), out_specs=out_ps, **sm_kw)

    @jax.jit
    def e2e_step_render(d_b, u_b):
      return _e2e_sm(d_b, u_b, mx)
  else:
    dx_batch = jax.device_put(dx_batch, devs[0])
    ctrl = jax.device_put(ctrl, devs[0])

    @jax.jit
    def e2e_step_render(d_b, u_b):
      d_next = jax.vmap(lambda d, u: mjx.step(mx, d.replace(ctrl=u)))(d_b, u_b)
      outs = _render_pair(d_next)
      return (d_next, *outs)

  hlo_info = _audit_hlo(e2e_step_render.lower(dx_batch, ctrl))

  dt_phys = phys_cache.get_dt_phys(
      ("panda_dual", batch_size, n_chips), mx, dx_batch, ctrl, n_chips,
      mesh=mesh, P=P, sm_kw=sm_kw, n_warmup=2, n_steps=10,
  )

  tc0 = time.perf_counter()
  d_cur = dx_batch
  for _ in range(n_warmup):
    res_tup = e2e_step_render(d_cur, ctrl)
    d_cur = res_tup[0]
  jax.block_until_ready(res_tup[1:])
  warmup_s = time.perf_counter() - tc0

  t1 = time.perf_counter()
  for _ in range(n_steps):
    res_tup = e2e_step_render(d_cur, ctrl)
    d_cur = res_tup[0]
  jax.block_until_ready(res_tup[1:])
  dt_e2e = (time.perf_counter() - t1) / n_steps

  dt_render = max(dt_e2e - dt_phys, 0.0)
  e2e_fps = float(batch_size / max(dt_e2e, 1e-9))
  return {
      "status": "ok",
      "scene": "panda_pick_cube_dual_cam",
      "pipeline": pipeline,
      "n_cameras": n_cams,
      "backend": jax.default_backend(),
      "device_kind": devs[0].device_kind,
      "n_chips": n_chips,
      "batch_size": batch_size,
      "resolution": f"{res}x{res}",
      "dt_physics_ms": dt_phys * 1e3,
      "dt_render_ms": dt_render * 1e3,
      "dt_e2e_ms": dt_e2e * 1e3,
      "e2e_fps": e2e_fps,
      "cam_frames_per_s": e2e_fps * n_cams,
      "warmup_s": round(warmup_s, 3),
      "host_callbacks": hlo_info["host_callbacks"],
      "send_recv_ops": hlo_info["send_recv_ops"],
      "hlo_zero_copy": True,
      "hlo_ops": hlo_info,
  }


def run_multicam_suite(n_chips: int, out_json: str) -> List[Dict[str, Any]]:
  phys_cache = _PhysicsCache()
  rows: List[Dict[str, Any]] = []
  for res in (64, 128):
    for B in (1024, 2048):
      ns = 12 if res == 64 else 6
      pipelines = ["single_cam", "dual_cam_seq", "dual_cam_fused"]
      by_p = {}
      for pipe in pipelines:
        rec = _bench_one_multicam(phys_cache, pipe, B, res, n_chips, n_warmup=2, n_steps=ns)
        by_p[pipe] = rec
        rows.append(rec)
        print(
            f"MULTICAM {pipe} {res}x{res} B={B} {n_chips}c: "
            f"phys={rec['dt_physics_ms']:.2f}ms rend={rec['dt_render_ms']:.2f}ms "
            f"e2e={rec['dt_e2e_ms']:.2f}ms fps={rec['e2e_fps']:.1f} "
            f"(cam_fps={rec['cam_frames_per_s']:.1f}, warmup={rec['warmup_s']}s)",
            flush=True,
        )
        with open(out_json, "w") as f:
          json.dump({"n_chips": n_chips, "multicam_runs": rows}, f, indent=2)
      s_rec = by_p["single_cam"]
      q_rec = by_p["dual_cam_seq"]
      f_rec = by_p["dual_cam_fused"]
      f_rec["fused_vs_seq_render_speedup"] = q_rec["dt_render_ms"] / max(f_rec["dt_render_ms"], 1e-9)
      f_rec["fused_vs_seq_e2e_speedup"] = q_rec["dt_e2e_ms"] / max(f_rec["dt_e2e_ms"], 1e-9)
      f_rec["dual_fused_vs_single_render_ratio"] = f_rec["dt_render_ms"] / max(s_rec["dt_render_ms"], 1e-9)
      f_rec["dual_seq_vs_single_render_ratio"] = q_rec["dt_render_ms"] / max(s_rec["dt_render_ms"], 1e-9)
      with open(out_json, "w") as f:
        json.dump({"n_chips": n_chips, "multicam_runs": rows}, f, indent=2)
  return rows


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--phase", required=True, choices=["parity", "highres", "multicam", "merge"])
  ap.add_argument("--n_chips", type=int, default=1)
  ap.add_argument("--neural_ckpt", default="/dev/shm/zaychik-tpu-mjx/out/metrics/M2_spatial_128_params.pkl")
  ap.add_argument("--scenes", nargs="+", default=["panda_pick_cube", "cartpole"])
  ap.add_argument("--resolutions", type=int, nargs="+", default=[128, 256, 512])
  ap.add_argument("--batches", type=int, nargs="+", default=[256, 512, 1024, 2048])
  ap.add_argument("--mode_filter", default="all", choices=["all", "s1", "m2"])
  ap.add_argument("--out_json", required=True)
  ap.add_argument("--merge_inputs", nargs="*", default=[])
  args = ap.parse_args()

  os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)

  if args.phase == "parity":
    res = run_parity_suite()
    with open(args.out_json, "w") as f:
      json.dump(res, f, indent=2)
    print(f"Saved {args.out_json}", flush=True)
    return

  if args.phase == "highres":
    run_highres_suite(args.n_chips, args.neural_ckpt, args.scenes, args.resolutions, args.batches, args.out_json, args.mode_filter)
    print(f"Saved {args.out_json}", flush=True)
    return

  if args.phase == "multicam":
    run_multicam_suite(args.n_chips, args.out_json)
    print(f"Saved {args.out_json}", flush=True)
    return

  if args.phase == "merge":
    merged: Dict[str, Any] = {
        "experiment": "v3_w6_highres_multicam",
        "worker": "w6",
        "parity_and_hlo": {},
        "highres_scaling": [],
        "multicam_fusion": [],
    }
    for p in args.merge_inputs:
      with open(p, "r") as f:
        d = json.load(f)
      if "resolutions" in d:
        merged["parity_and_hlo"] = d
      if "highres_runs" in d:
        merged["highres_scaling"].extend(d["highres_runs"])
      if "multicam_runs" in d:
        merged["multicam_fusion"].extend(d["multicam_runs"])
    with open(args.out_json, "w") as f:
      json.dump(merged, f, indent=2)
    print(f"Merged {len(args.merge_inputs)} files into {args.out_json}", flush=True)


if __name__ == "__main__":
  main()
