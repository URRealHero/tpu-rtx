"""Wave 2 runner on w0 TPU v5p:
- Tile geometry + BF16 + material-dedup sweep (1-chip 64x64 & 128x128)
- 4-chip shard_map push to 300,000+ FPS (64x64) and 120,000-150,000+ FPS (128x128)
- In-VMEM Fused GGX PBR + Soft-Shadow + Hemisphere AO shader benchmark + visual samples
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import re
import time
from typing import Any, Dict, List

import jax
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
import numpy as np

from fused_v3_wave2 import pallas_raycast_fused_v3_opt, pallas_raycast_pbr_ao_v3
from tpu_mjx_render.pallas_v2 import pallas_raycast_batch_v2
from tpu_mjx_render.renderer import RenderSceneSpec
from tpu_mjx_render.roofline import inspect_hlo_ops


def _check_hlo(lowered) -> Dict[str, Any]:
  txt = lowered.as_text()
  info = inspect_hlo_ops(txt)
  info["send_recv_ops"] = len(re.findall(r"\bsend\(|\brecv\(", txt, re.I))
  assert info["host_callbacks"] == 0 and info["send_recv_ops"] == 0, info
  info["zero_copy_verified"] = True
  return info


def _save_payload(path: str, payload: Dict[str, Any]) -> None:
  os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
  tmp = path + ".tmp"
  with open(tmp, "w") as f:
    json.dump(payload, f, indent=2)
  os.replace(tmp, path)


def _setup_panda(batch_size: int, height: int, width: int, n_chips: int):
  import mujoco
  from mujoco import mjx
  from tpu_mjx_render.scenes import get_scene_xml

  m = mujoco.MjModel.from_xml_string(get_scene_xml("panda_pick_cube"))
  mx = mjx.put_model(m, impl="jax")
  spec = RenderSceneSpec.from_mjmodel(m, cam_id=0, height=height, width=width, use_shadows=True)

  devs = jax.devices()[:n_chips]
  if n_chips > 1:
    assert batch_size % n_chips == 0
    mesh = jax.sharding.Mesh(np.array(devs), ("dp",))
    P = jax.sharding.PartitionSpec
    sharding = jax.sharding.NamedSharding(mesh, P("dp"))
    replicate = jax.sharding.NamedSharding(mesh, P())
  else:
    mesh = P = sharding = replicate = None

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
  else:
    dx_batch = jax.device_put(dx_batch, devs[0])
    ctrl = jax.device_put(ctrl, devs[0])

  return m, mx, spec, devs, mesh, P, dx_batch, ctrl


def _bench_fn(
    label: str,
    render_fn,
    batch_size: int,
    height: int,
    width: int,
    n_chips: int,
    n_warmup: int = 3,
    n_steps: int = 20,
    extra_meta: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
  _, mx, spec, devs, mesh, P, dx_batch, ctrl = _setup_panda(batch_size, height, width, n_chips)

  if n_chips > 1:
    sm_kw = {"check_rep": False} if "check_rep" in inspect.signature(shard_map).parameters else {"check_vma": False}

    def _phys_local(d_b, u_b, mx_l):
      return jax.vmap(lambda d, u: mjx.step(mx_l, d.replace(ctrl=u)))(d_b, u_b)

    from mujoco import mjx
    _phys_sm = shard_map(_phys_local, mesh=mesh, in_specs=(P("dp"), P("dp"), P()), out_specs=P("dp"), **sm_kw)

    @jax.jit
    def physics_only(d_b, u_b):
      return _phys_sm(d_b, u_b, mx)

    def _e2e_local(d_b, u_b, mx_l):
      d_next = jax.vmap(lambda d, u: mjx.step(mx_l, d.replace(ctrl=u)))(d_b, u_b)
      rgb, depth = render_fn(spec, d_next)
      return d_next, rgb, depth

    _e2e_sm = shard_map(
        _e2e_local, mesh=mesh, in_specs=(P("dp"), P("dp"), P()), out_specs=(P("dp"), P("dp"), P("dp")), **sm_kw
    )

    @jax.jit
    def e2e_step_render(d_b, u_b):
      return _e2e_sm(d_b, u_b, mx)
  else:
    from mujoco import mjx

    @jax.jit
    def physics_only(d_b, u_b):
      return jax.vmap(lambda d, u: mjx.step(mx, d.replace(ctrl=u)))(d_b, u_b)

    @jax.jit
    def e2e_step_render(d_b, u_b):
      d_next = jax.vmap(lambda d, u: mjx.step(mx, d.replace(ctrl=u)))(d_b, u_b)
      rgb, depth = render_fn(spec, d_next)
      return d_next, rgb, depth

  hlo_info = _check_hlo(e2e_step_render.lower(dx_batch, ctrl))

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
  rec = {
      "label": label,
      "scene": "panda_pick_cube",
      "backend": jax.default_backend(),
      "device_kind": devs[0].device_kind,
      "n_chips": n_chips,
      "batch_size": batch_size,
      "resolution": f"{height}x{width}",
      "dt_physics_ms": dt_phys * 1e3,
      "dt_render_ms": dt_render * 1e3,
      "dt_e2e_ms": dt_e2e * 1e3,
      "e2e_fps": fps_e2e,
      "render_only_equiv_fps": float(batch_size / max(dt_render, 1e-9)),
      "hlo_zero_copy": hlo_info["host_callbacks"] == 0 and hlo_info["send_recv_ops"] == 0,
      "hlo_ops": hlo_info,
  }
  if extra_meta:
    rec.update(extra_meta)
  print(f"WAVE2 [{label}]:", json.dumps(rec), flush=True)
  return rec


def run_sweep_1c(out_json: str, payload: Dict[str, Any], resolutions: List[int]) -> None:
  import mujoco
  from mujoco import mjx
  from tpu_mjx_render.scenes import get_scene_xml

  # 1. Verify parity for dedup FP32 and dedup BF16 vs pallas_v2
  m = mujoco.MjModel.from_xml_string(get_scene_xml("panda_pick_cube"))
  mx = mjx.put_model(m, impl="jax")
  spec = RenderSceneSpec.from_mjmodel(m, cam_id=0, height=64, width=64, use_shadows=True)
  dx0 = mjx.make_data(m, impl="jax")
  d_host = mujoco.MjData(m)
  mujoco.mj_resetDataKeyframe(m, d_host, 0)
  dx0 = dx0.replace(qpos=jnp.asarray(d_host.qpos, jnp.float32))
  dx0 = mjx.forward(mx, dx0)
  B = 16
  dx_b = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x[None, ...], (B, *x.shape)).copy(), dx0)
  u = jnp.linspace(-0.2, 0.2, B, dtype=jnp.float32)[:, None] * jnp.ones((1, m.nu), dtype=jnp.float32)
  dx_b = jax.jit(lambda d, ctrl: jax.vmap(lambda di, ui: mjx.step(mx, di.replace(ctrl=ui)))(d, ctrl))(dx_b, u)
  dx_b = jax.device_put(dx_b, jax.devices()[0])

  if "wave2_parity" not in payload:
    gb_v2 = jax.jit(lambda d: pallas_raycast_batch_v2(spec, d, interpret=False))(dx_b)
    parity_w2 = {}
    for tag, bb, br, dt_str in [
        ("opt_b8_r512_f32", 8, 512, "float32"),
        ("opt_b8_r512_bf16", 8, 512, "bfloat16"),
        ("opt_b16_r512_bf16", 16, 512, "bfloat16"),
        ("opt_b8_r1024_bf16", 8, 1024, "bfloat16"),
    ]:
      dt = jnp.bfloat16 if dt_str == "bfloat16" else jnp.float32
      fn = jax.jit(lambda d, _bb=bb, _br=br, _dt=dt: pallas_raycast_fused_v3_opt(spec, d, interpret=False, block_b=_bb, block_r=_br, out_dtype=_dt))
      fr = fn(dx_b)
      jax.block_until_ready((fr.rgb, fr.depth))
      dd = jnp.abs(gb_v2.depth - fr.depth.astype(jnp.float32))
      rd = jnp.abs(gb_v2.rgb - fr.rgb.astype(jnp.float32))
      parity_w2[tag] = {
          "depth_mae_m": float(jnp.mean(dd)),
          "max_depth_diff_m": float(jnp.max(dd)),
          "max_rgb_diff": float(jnp.max(rd)),
          "mean_rgb_diff": float(jnp.mean(rd)),
          "hlo": _check_hlo(fn.lower(dx_b)),
      }
      print(f"W2 PARITY [{tag}]:", json.dumps(parity_w2[tag]), flush=True)
    payload["wave2_parity"] = parity_w2
    _save_payload(out_json, payload)

  # 2. Tile & dtype sweep at B=2048 on 1-chip
  sweep_runs = [r for r in payload.get("sweep_1c", []) if r.get("res_int") not in resolutions]
  configs = [
      ("b8_r512_f32_dedup", 8, 512, "float32"),
      ("b16_r512_f32_dedup", 16, 512, "float32"),
      ("b16_r512_bf16_dedup", 16, 512, "bfloat16"),
      ("b8_r1024_bf16_dedup", 8, 1024, "bfloat16"),
  ]
  for res in resolutions:
    for tag, bb, br, dt_str in configs:
      dt = jnp.bfloat16 if dt_str == "bfloat16" else jnp.float32
      try:
        rec = _bench_fn(
            label=f"1c_{res}x{res}_{tag}",
            render_fn=lambda s, d, _bb=bb, _br=br, _dt=dt: pallas_raycast_fused_v3_opt(
                s, d, interpret=False, block_b=_bb, block_r=_br, out_dtype=_dt
            ),
            batch_size=2048,
            height=res,
            width=res,
            n_chips=1,
            extra_meta={"res_int": res, "block_b": bb, "block_r": br, "out_dtype": dt_str},
        )
        sweep_runs.append(rec)
      except Exception as exc:
        err_rec = {"res_int": res, "label": f"1c_{res}x{res}_{tag}", "block_b": bb, "block_r": br, "out_dtype": dt_str, "error": str(exc)[:300]}
        print("SWEEP ERR:", json.dumps(err_rec), flush=True)
        sweep_runs.append(err_rec)
      payload["sweep_1c"] = sweep_runs
      _save_payload(out_json, payload)


def run_pbr_ao_1c(out_json: str, payload: Dict[str, Any]) -> None:
  import mujoco
  from mujoco import mjx
  from tpu_mjx_render.scenes import get_scene_xml

  # 1. Render visual comparison samples at 128x128 (B=8) from picked & low_home poses
  m = mujoco.MjModel.from_xml_string(get_scene_xml("panda_pick_cube"))
  mx = mjx.put_model(m, impl="jax")
  spec128 = RenderSceneSpec.from_mjmodel(m, cam_id=0, height=128, width=128, use_shadows=True)
  dx0 = mjx.make_data(m, impl="jax")
  d_host = mujoco.MjData(m)
  mujoco.mj_resetDataKeyframe(m, d_host, 0)
  dx0 = dx0.replace(qpos=jnp.asarray(d_host.qpos, jnp.float32))
  dx0 = mjx.forward(mx, dx0)
  B_vis = 8
  dx_vis = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x[None, ...], (B_vis, *x.shape)).copy(), dx0)
  u_vis = jnp.linspace(-0.4, 0.4, B_vis, dtype=jnp.float32)[:, None] * jnp.ones((1, m.nu), dtype=jnp.float32)
  step_fn = jax.jit(lambda d, u: jax.vmap(lambda di, ui: mjx.step(mx, di.replace(ctrl=ui)))(d, u))
  for _ in range(15):
    dx_vis = step_fn(dx_vis, u_vis)
  dx_vis = jax.device_put(dx_vis, jax.devices()[0])

  fr_bp = jax.jit(lambda d: pallas_raycast_fused_v3_opt(spec128, d, interpret=False, block_b=8, block_r=512))(dx_vis)
  fr_pbr_11 = jax.jit(
      lambda d: pallas_raycast_pbr_ao_v3(spec128, d, interpret=False, n_ao_samples=4, n_shadow_samples=1)
  )(dx_vis)
  fr_pbr_44 = jax.jit(
      lambda d: pallas_raycast_pbr_ao_v3(spec128, d, interpret=False, n_ao_samples=4, n_shadow_samples=4)
  )(dx_vis)
  jax.block_until_ready((fr_bp.rgb, fr_pbr_11.rgb, fr_pbr_44.rgb))

  np.savez_compressed(
      "/tmp/zaychik-tpu-mjx-w0/v3_w0_pbr_samples.npz",
      rgb_blinn_phong=np.asarray(fr_bp.rgb),
      rgb_ggx_ao4_sh1=np.asarray(fr_pbr_11.rgb),
      rgb_ggx_ao4_sh4=np.asarray(fr_pbr_44.rgb),
      depth=np.asarray(fr_bp.depth),
  )
  print("Saved /tmp/zaychik-tpu-mjx-w0/v3_w0_pbr_samples.npz", flush=True)

  # 2. Benchmark 1-chip 64x64 PBR+AO+SoftShadow
  pbr_runs = []
  for n_ao, n_sh in [(4, 1), (4, 4)]:
    for B in [1024, 2048]:
      rec = _bench_fn(
          label=f"1c_64x64_pbr_ggx_ao{n_ao}_sh{n_sh}",
          render_fn=lambda s, d, _ao=n_ao, _sh=n_sh: pallas_raycast_pbr_ao_v3(
              s, d, interpret=False, block_b=8, block_r=512, out_dtype=jnp.bfloat16, n_ao_samples=_ao, n_shadow_samples=_sh
          ),
          batch_size=B,
          height=64,
          width=64,
          n_chips=1,
          extra_meta={"shader": "ggx_pbr_ao_softshadow", "n_ao_samples": n_ao, "n_shadow_samples": n_sh, "out_dtype": "bfloat16"},
      )
      pbr_runs.append(rec)
      payload["pbr_ao_1c"] = pbr_runs
      _save_payload(out_json, payload)


def run_push_4c(out_json: str, payload: Dict[str, Any], res: int, batches: List[int], bb: int, br: int, dt_str: str, include_pbr: bool) -> None:
  dt = jnp.bfloat16 if dt_str == "bfloat16" else jnp.float32
  key = f"push_4c_{res}"
  existing = payload.get(key, [])
  for B in batches:
    lbl = f"4c_{res}x{res}_b{bb}_r{br}_{dt_str}_B{B}"
    existing = [r for r in existing if r.get("label") != lbl]
    rec = _bench_fn(
        label=lbl,
        render_fn=lambda s, d, _bb=bb, _br=br, _dt=dt: pallas_raycast_fused_v3_opt(
            s, d, interpret=False, block_b=_bb, block_r=_br, out_dtype=_dt
        ),
        batch_size=B,
        height=res,
        width=res,
        n_chips=4,
        extra_meta={"block_b": bb, "block_r": br, "out_dtype": dt_str, "shader": "blinn_phong_fused_opt"},
    )
    existing.append(rec)
    payload[key] = existing
    _save_payload(out_json, payload)

    if include_pbr:
      lbl_pbr = f"4c_{res}x{res}_pbr_ggx_ao4_sh4_bf16_B{B}"
      existing = [r for r in existing if r.get("label") != lbl_pbr]
      rec_pbr = _bench_fn(
          label=lbl_pbr,
          render_fn=lambda s, d: pallas_raycast_pbr_ao_v3(
              s, d, interpret=False, block_b=8, block_r=512, out_dtype=jnp.bfloat16, n_ao_samples=4, n_shadow_samples=4
          ),
          batch_size=B,
          height=res,
          width=res,
          n_chips=4,
          extra_meta={"block_b": 8, "block_r": 512, "out_dtype": "bfloat16", "shader": "ggx_pbr_ao4_sh4"},
      )
      existing.append(rec_pbr)
      payload[key] = existing
      _save_payload(out_json, payload)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--step", required=True, choices=["sweep_1c", "pbr_ao_1c", "push_4c_64", "push_4c_128"])
  ap.add_argument("--batches", type=int, nargs="+", default=None)
  ap.add_argument("--res", type=int, nargs="+", default=[64])
  ap.add_argument("--block_b", type=int, default=16)
  ap.add_argument("--block_r", type=int, default=512)
  ap.add_argument("--out_dtype", default="bfloat16", choices=["float32", "bfloat16"])
  ap.add_argument("--include_pbr", action="store_true")
  ap.add_argument("--out_json", required=True)
  args = ap.parse_args()

  if os.path.exists(args.out_json):
    with open(args.out_json, "r") as f:
      payload = json.load(f)
  else:
    payload = {}

  if args.step == "sweep_1c":
    run_sweep_1c(args.out_json, payload, args.res)
  elif args.step == "pbr_ao_1c":
    run_pbr_ao_1c(args.out_json, payload)
  elif args.step == "push_4c_64":
    batches = args.batches if args.batches is not None else [4096, 8192, 16384]
    run_push_4c(args.out_json, payload, 64, batches, args.block_b, args.block_r, args.out_dtype, args.include_pbr)
  elif args.step == "push_4c_128":
    batches = args.batches if args.batches is not None else [1024, 2048, 4096, 8192]
    run_push_4c(args.out_json, payload, 128, batches, args.block_b, args.block_r, args.out_dtype, args.include_pbr)


if __name__ == "__main__":
  main()
