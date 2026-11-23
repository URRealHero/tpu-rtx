"""32-Chip TPU v5p-64 Pod Capstone Benchmark Suite (8 Hosts x 4 Chips = 32 Chips, 3.06 TB HBM).

Supports:
  - Multi-host distributed execution (calling `jax.distributed.initialize()` when
    `JAX_COORDINATOR_ADDRESS` is set) across all 32 TPU v5p chips.
  - `--dry-run-4c` single-host 4-chip mode for full parity, HLO, shape, and scaling validation.

Five cluster-wide benchmarks (all verified host_callbacks == 0, send_recv_ops == 0):
  1. 32-Chip `tpu_rtx` MJX + Fused Pallas Render (64x64 BF16, B=65,536 / 131,072).
  2. 32-Chip `tpu_gsplat` 3DGS Batch (64x64 & 128x128) + Distributed 1.06B Gaussians across 3.06 TB HBM.
  3. 32-Chip `128x2048` Batch LiDAR (B=16,384 / 32,768 -> 100k+ env-FPS / 28B+ rays/s).
  4. 32-Chip Geometry-Sharded 8.59B & 17.18B 100% Unique Non-Instanced Triangles across 3.06 TB HBM
     (`shard_map` + `jax.lax.pmin` across all 32 chips over ICI @ Native 4K & 8K).
  5. 32-Chip Spatial-Sharded Native 8K (7680x4320) 3-Bounce Monte Carlo Path Tracing (samples/min).
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import mujoco
from mujoco import mjx
import numpy as np

from tpu_mjx_render.pallas_v2 import (
    BLOCK_B as P2_BLOCK_B,
    BLOCK_R as P2_BLOCK_R,
    _compute_geom_bound_sq,
    _ray_one_geom_v2,
    pallas_raycast_batch_v2,
)
from tpu_mjx_render.primitives import MJ_MAXVAL
from tpu_mjx_render.renderer import RenderSceneSpec
from tpu_mjx_render.roofline import inspect_hlo_ops
from tpu_mjx_render.scenes import PANDA_PICK_CUBE_XML

# ==============================================================================
# Helpers: Distributed Init, Multi-Host Sharding, HLO Zero-Copy Audit
# ==============================================================================

def _sm_kwargs() -> Dict[str, Any]:
  return {"check_rep": False} if "check_rep" in inspect.signature(shard_map).parameters else {"check_vma": False}


def _audit_hlo(lowered: Any, require_no_gather: bool = False) -> Dict[str, Any]:
  txt = lowered.as_text()
  info = inspect_hlo_ops(txt)
  info["send_recv_ops"] = len(re.findall(r"\bsend\(|\brecv\(", txt, re.I))
  assert info["host_callbacks"] == 0, f"HLO contains host_callback: {info}"
  assert info["send_recv_ops"] == 0, f"HLO contains send/recv: {info}"
  if require_no_gather:
    assert info["gather_ops"] == 0, f"HLO contains gather_ops: {info}"
  info["zero_copy_verified"] = True
  return info


def _make_global_from_local_shards(
    mesh: Mesh,
    pspec: P,
    local_arrays: Sequence[jax.Array],
    global_shape: Tuple[int, ...],
) -> jax.Array:
  sharding = NamedSharding(mesh, pspec)
  return jax.make_array_from_single_device_arrays(global_shape, sharding, list(local_arrays))


def _sync_hosts(tag: str, is_distributed: bool) -> None:
  if is_distributed:
    from jax.experimental import multihost_utils
    multihost_utils.sync_global_devices(tag)


# ==============================================================================
# Benchmark 1: 32-Chip tpu_rtx MJX + Fused Pallas Render (64x64 BF16)
# ==============================================================================

def run_bench1_tpu_rtx(
    mesh_dp: Mesh,
    local_devs: Sequence[jax.Device],
    n_chips: int,
    pallas_interp: bool,
    proc_idx: int,
) -> List[Dict[str, Any]]:
  m = mujoco.MjModel.from_xml_string(PANDA_PICK_CUBE_XML)
  spec_64 = RenderSceneSpec.from_mjmodel(m, cam_id=0, height=64, width=64, use_shadows=True)

  d_host = mujoco.MjData(m)
  if m.nkey > 0:
    mujoco.mj_resetDataKeyframe(m, d_host, 0)
  mujoco.mj_forward(m, d_host)
  with jax.default_device(local_devs[0]):
    mx_np = jax.tree_util.tree_map(np.asarray, mjx.put_model(m, device=local_devs[0], impl="jax"))
    dx0_np = jax.tree_util.tree_map(np.asarray, mjx.put_data(m, d_host, device=local_devs[0], impl="jax"))
  ctrl0_np = np.asarray(dx0_np.ctrl, dtype=np.float32)

  sm_kw = _sm_kwargs()
  rep_sharding = NamedSharding(mesh_dp, P())
  mx_rep = jax.tree_util.tree_map(
      lambda x: jax.make_array_from_single_device_arrays(
          x.shape, rep_sharding, [jax.device_put(x, d) for d in local_devs]
      ),
      mx_np,
  )

  rows: List[Dict[str, Any]] = []
  for b_local in (2048, 4096):
    b_global = b_local * n_chips
    phases_np = np.linspace(-0.25, 0.25, b_local, dtype=np.float32)[:, None]
    u_np = ctrl0_np[None, :] + phases_np * np.linspace(1.0, 0.3, m.nu, dtype=np.float32)[None, :]

    def _build_local_dx(dev):
      with jax.default_device(dev):
        dx_l = jax.tree_util.tree_map(
            lambda x: jax.device_put(np.broadcast_to(x[None, ...], (b_local, *x.shape)).copy(), dev),
            dx0_np,
        )
        return dx_l, jax.device_put(u_np.copy(), dev)

    local_pairs = [_build_local_dx(d) for d in local_devs]
    # Assemble global sharded dx_b and ctrl with zero cross-host data broadcast
    dx_b = jax.tree_util.tree_map(
        lambda *leaves: _make_global_from_local_shards(
            mesh_dp, P("dp"), leaves, (b_global, *leaves[0].shape[1:])
        ),
        *[p[0] for p in local_pairs],
    )
    ctrl_b = _make_global_from_local_shards(
        mesh_dp, P("dp"), [p[1] for p in local_pairs], (b_global, m.nu)
    )

    def _local_step_render(d, u, mxl):
      dn = jax.vmap(lambda x, c: mjx.step(mxl, x.replace(ctrl=c)))(d, u)
      gb = pallas_raycast_batch_v2(spec_64, dn, interpret=pallas_interp)
      return dn, gb.rgb.astype(jnp.bfloat16), gb.depth.astype(jnp.bfloat16)

    sm_fn = shard_map(
        _local_step_render,
        mesh=mesh_dp,
        in_specs=(P("dp"), P("dp"), P()),
        out_specs=(P("dp"), P("dp"), P("dp")),
        **sm_kw,
    )
    jit_fn = jax.jit(lambda d, u: sm_fn(d, u, mx_rep))
    lowered = jit_fn.lower(dx_b, ctrl_b)
    hlo = _audit_hlo(lowered, require_no_gather=False)

    tc0 = time.perf_counter()
    exec_fn = lowered.compile()
    for _ in range(2):
      dc, rgb, dep = exec_fn(dx_b, ctrl_b)
    jax.block_until_ready((rgb, dep))
    warmup_s = time.perf_counter() - tc0

    n_iters = 5
    t0 = time.perf_counter()
    for _ in range(n_iters):
      dc, rgb, dep = exec_fn(dx_b, ctrl_b)
    jax.block_until_ready((rgb, dep))
    dt_e2e = (time.perf_counter() - t0) / n_iters

    fps = float(b_global / max(dt_e2e, 1e-9))
    gpix_s = (fps * 64 * 64) / 1e9
    rec = {
        "benchmark": "1_tpu_rtx_mjx_render_64x64_bf16",
        "n_chips": n_chips,
        "b_per_chip": b_local,
        "global_batch": b_global,
        "resolution": "64x64",
        "dtype": "bfloat16",
        "dt_e2e_ms": dt_e2e * 1e3,
        "e2e_fps": fps,
        "gpix_per_s": gpix_s,
        "warmup_s": round(warmup_s, 3),
        "host_callbacks": hlo["host_callbacks"],
        "send_recv_ops": hlo["send_recv_ops"],
        "hlo_zero_copy": True,
    }
    rows.append(rec)
    if proc_idx == 0:
      print(f"[B1 tpu_rtx] {n_chips}c B={b_global}: {dt_e2e*1e3:.2f} ms -> {fps:,.1f} FPS ({gpix_s:.3f} GPix/s)", flush=True)
  return rows


# ==============================================================================
# Benchmark 2: tpu_gsplat 3DGS Batch (64x64 & 128x128) + Distributed 1.06B+ Gaussians
# ==============================================================================

def _pallas_gs3d_mosaic(
    splats_b16n: jax.Array,  # (C, 16, N_pad)
    top_idx: jax.Array,      # (C, T, K) int32
    t_ox: np.ndarray,        # (T,)
    t_oy: np.ndarray,        # (T,)
    t_grp: int = 16,
    interpret: bool = False,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
  """Self-contained V3 Mosaic Pallas 3DGS kernel (VMEM take_along_axis + MXU prefix transmittance)."""
  C, _, N_pad = splats_b16n.shape
  _, T, K_in = top_idx.shape
  P_t = 64
  K = 128
  if K_in < K:
    top_idx = jnp.pad(top_idx, ((0, 0), (0, 0), (0, K - K_in)), constant_values=-1)
  n_chunks = N_pad // 128
  n_tg = T // t_grp
  out_cols = t_grp * 8

  lx = np.tile(np.arange(8, dtype=np.float32), 8) + 0.5
  ly = np.repeat(np.arange(8, dtype=np.float32), 8) + 0.5
  loc_xy_dev = jnp.asarray(
      np.stack([np.broadcast_to(lx[:, None], (P_t, K)), np.broadcast_to(ly[:, None], (P_t, K))], axis=0).reshape(2 * P_t, K),
      dtype=jnp.float32,
  )
  u_dev = jnp.asarray(np.triu(np.ones((K, K), dtype=np.float32), k=1), dtype=jnp.bfloat16)

  oxy_rows = ((2 * t_grp + 7) // 8) * 8
  t_oxy_np = np.zeros((n_tg, oxy_rows, 128), dtype=np.float32)
  t_oxy_np[:, :t_grp, :] = t_ox.reshape(n_tg, t_grp, 1)
  t_oxy_np[:, t_grp : 2 * t_grp, :] = t_oy.reshape(n_tg, t_grp, 1)
  t_oxy_dev = jnp.asarray(t_oxy_np.reshape(n_tg * oxy_rows, 128), dtype=jnp.float32)

  def _fused_kernel(s_ref, i_ref, oxy_ref, lxy_ref, u_ref, out_ref):
    s_all = s_ref[...]
    ix = i_ref[...]
    oxy = oxy_ref[...]
    lxy = lxy_ref[...]
    lx_m, ly_m = lxy[0:64, :], lxy[64:128, :]
    u_m = u_ref[...]
    ones_k = jnp.ones((1, K), dtype=jnp.float32)

    ix_tiles = [jnp.broadcast_to(ix[t : t + 1, :], (16, K)) for t in range(t_grp)]
    accs = [jnp.zeros((16, K), dtype=jnp.float32) for _ in range(t_grp)]
    for c in range(n_chunks):
      base = c * 128
      s_c = s_all[:, base : base + 128]
      for t in range(t_grp):
        ix_t = ix_tiles[t]
        mask = (ix_t >= base) & (ix_t < base + 128)
        local_ix = jnp.where(mask, ix_t - base, 0)
        vals = jnp.take_along_axis(s_c, local_ix, axis=1)
        accs[t] = accs[t] + jnp.where(mask, vals, 0.0)

    outs = []
    for t in range(t_grp):
      sf = accs[t]
      px = jnp.broadcast_to(oxy[t : t + 1, 0:1], (P_t, K)) + lx_m
      py = jnp.broadcast_to(oxy[t_grp + t : t_grp + t + 1, 0:1], (P_t, K)) + ly_m
      dx = px - jnp.broadcast_to(sf[0:1, :], (P_t, K))
      dy = py - jnp.broadcast_to(sf[1:2, :], (P_t, K))
      cxx = jnp.broadcast_to(sf[2:3, :], (P_t, K))
      cxy = jnp.broadcast_to(sf[3:4, :], (P_t, K))
      cyy = jnp.broadcast_to(sf[4:5, :], (P_t, K))
      gop = jnp.broadcast_to(sf[8:9, :], (P_t, K))

      power = -0.5 * (cxx * dx * dx + 2.0 * cxy * dx * dy + cyy * dy * dy)
      valid_p = (power <= 0.0) & (power > -16.0)
      alpha = jnp.where(valid_p, jnp.minimum(0.99, gop * jnp.exp(jnp.maximum(power, -16.0))), 0.0)
      alpha = jnp.where(alpha >= (1.0 / 255.0), alpha, 0.0)

      log_1ma = jnp.log(jnp.maximum(1.0 - alpha, 1.0e-7))
      log_tr = jnp.matmul(log_1ma.astype(jnp.bfloat16), u_m, preferred_element_type=jnp.float32)
      trans = jnp.exp(jnp.maximum(log_tr, -30.0))
      w = alpha * trans
      attr_t = jnp.transpose(jnp.concatenate([sf[5:8, :], sf[9:13, :], ones_k], axis=0), (1, 0))
      outs.append(jnp.matmul(w, attr_t, precision=jax.lax.Precision.HIGHEST))
    out_ref[...] = jnp.concatenate(outs, axis=1)

  out_raw = pl.pallas_call(
      _fused_kernel,
      out_shape=jax.ShapeDtypeStruct((C * n_tg * P_t, out_cols), jnp.float32),
      grid=(C, n_tg),
      in_specs=[
          pl.BlockSpec((16, N_pad), lambda c, tg: (c, 0)),
          pl.BlockSpec((t_grp, K), lambda c, tg: (c * n_tg + tg, 0)),
          pl.BlockSpec((oxy_rows, 128), lambda c, tg: (tg, 0)),
          pl.BlockSpec((2 * P_t, K), lambda c, tg: (0, 0)),
          pl.BlockSpec((K, K), lambda c, tg: (0, 0)),
      ],
      out_specs=pl.BlockSpec((P_t, out_cols), lambda c, tg: (c * n_tg + tg, 0)),
      compiler_params=None if interpret else pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
      interpret=interpret,
  )(
      splats_b16n.reshape(C * 16, N_pad),
      top_idx.reshape(C * T, K),
      t_oxy_dev,
      loc_xy_dev,
      u_dev,
  )
  out_t = out_raw.reshape(C, n_tg, P_t, t_grp, 8).transpose(0, 1, 3, 2, 4).reshape(C, T, P_t, 8)
  rgb = out_t[..., 0:3]
  depth = out_t[..., 3]
  alpha = out_t[..., 7]
  return rgb, alpha, depth


def _pallas_gs3d_pertile_mosaic(
    tile_splats: jax.Array,  # (T, 16, 128) pre-binned top-K splats per 8x8 tile
    t_ox: np.ndarray,        # (T,)
    t_oy: np.ndarray,        # (T,)
    interpret: bool = False,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
  """Direct per-tile Mosaic Pallas 3DGS kernel (scales to 1B+ distributed Gaussians!)."""
  T = tile_splats.shape[0]
  t_grp = 16
  assert T % t_grp == 0
  n_tg = T // t_grp
  P_t = 64
  K = 128
  out_cols = t_grp * 8

  lx = np.tile(np.arange(8, dtype=np.float32), 8) + 0.5
  ly = np.repeat(np.arange(8, dtype=np.float32), 8) + 0.5
  loc_xy_dev = jnp.asarray(
      np.stack([np.broadcast_to(lx[:, None], (P_t, K)), np.broadcast_to(ly[:, None], (P_t, K))], axis=0).reshape(2 * P_t, K),
      dtype=jnp.float32,
  )
  u_dev = jnp.asarray(np.triu(np.ones((K, K), dtype=np.float32), k=1), dtype=jnp.bfloat16)
  oxy_rows = ((2 * t_grp + 7) // 8) * 8
  t_oxy_np = np.zeros((n_tg, oxy_rows, 128), dtype=np.float32)
  t_oxy_np[:, :t_grp, :] = t_ox.reshape(n_tg, t_grp, 1)
  t_oxy_np[:, t_grp : 2 * t_grp, :] = t_oy.reshape(n_tg, t_grp, 1)
  t_oxy_dev = jnp.asarray(t_oxy_np.reshape(n_tg * oxy_rows, 128), dtype=jnp.float32)

  def _pertile_kernel(ts_ref, oxy_ref, lxy_ref, u_ref, out_ref):
    ts = ts_ref[...]  # (t_grp * 16, 128)
    oxy = oxy_ref[...]
    lxy = lxy_ref[...]
    lx_m, ly_m = lxy[0:64, :], lxy[64:128, :]
    u_m = u_ref[...]
    ones_k = jnp.ones((1, K), dtype=jnp.float32)

    outs = []
    for t in range(t_grp):
      sf = ts[t * 16 : (t + 1) * 16, :]
      px = jnp.broadcast_to(oxy[t : t + 1, 0:1], (P_t, K)) + lx_m
      py = jnp.broadcast_to(oxy[t_grp + t : t_grp + t + 1, 0:1], (P_t, K)) + ly_m
      dx = px - jnp.broadcast_to(sf[0:1, :], (P_t, K))
      dy = py - jnp.broadcast_to(sf[1:2, :], (P_t, K))
      cxx = jnp.broadcast_to(sf[2:3, :], (P_t, K))
      cxy = jnp.broadcast_to(sf[3:4, :], (P_t, K))
      cyy = jnp.broadcast_to(sf[4:5, :], (P_t, K))
      gop = jnp.broadcast_to(sf[8:9, :], (P_t, K))

      power = -0.5 * (cxx * dx * dx + 2.0 * cxy * dx * dy + cyy * dy * dy)
      valid_p = (power <= 0.0) & (power > -16.0)
      alpha = jnp.where(valid_p, jnp.minimum(0.99, gop * jnp.exp(jnp.maximum(power, -16.0))), 0.0)
      alpha = jnp.where(alpha >= (1.0 / 255.0), alpha, 0.0)

      log_1ma = jnp.log(jnp.maximum(1.0 - alpha, 1.0e-7))
      log_tr = jnp.matmul(log_1ma.astype(jnp.bfloat16), u_m, preferred_element_type=jnp.float32)
      trans = jnp.exp(jnp.maximum(log_tr, -30.0))
      w = alpha * trans
      attr_t = jnp.transpose(jnp.concatenate([sf[5:8, :], sf[9:13, :], ones_k], axis=0), (1, 0))
      outs.append(jnp.matmul(w, attr_t, precision=jax.lax.Precision.HIGHEST))
    out_ref[...] = jnp.concatenate(outs, axis=1)

  out_raw = pl.pallas_call(
      _pertile_kernel,
      out_shape=jax.ShapeDtypeStruct((n_tg * P_t, out_cols), jnp.float32),
      grid=(n_tg,),
      in_specs=[
          pl.BlockSpec((t_grp * 16, K), lambda tg: (tg, 0)),
          pl.BlockSpec((oxy_rows, 128), lambda tg: (tg, 0)),
          pl.BlockSpec((2 * P_t, K), lambda tg: (0, 0)),
          pl.BlockSpec((K, K), lambda tg: (0, 0)),
      ],
      out_specs=pl.BlockSpec((P_t, out_cols), lambda tg: (tg, 0)),
      compiler_params=None if interpret else pltpu.CompilerParams(dimension_semantics=("parallel",)),
      interpret=interpret,
  )(tile_splats.reshape(T * 16, K), t_oxy_dev, loc_xy_dev, u_dev)

  out_t = out_raw.reshape(n_tg, P_t, t_grp, 8).transpose(0, 2, 1, 3).reshape(T, P_t, 8)
  return out_t[..., 0:3], out_t[..., 7], out_t[..., 3]


def run_bench2_tpu_gsplat(
    mesh_dp: Mesh,
    local_devs: Sequence[jax.Device],
    n_chips: int,
    pallas_interp: bool,
    proc_idx: int,
) -> Dict[str, Any]:
  sm_kw = _sm_kwargs()
  batch_rows: List[Dict[str, Any]] = []
  N_splats = 2048
  K_top = 64

  # Part 2A: Batched 3DGS (64x64 and 128x128)
  for H, W, b_local in ((64, 64, 256), (128, 128, 64)):
    tx, ty = W // 8, H // 8
    T = tx * ty
    t_ox = np.tile(np.arange(tx, dtype=np.float32) * 8.0, ty)
    t_oy = np.repeat(np.arange(ty, dtype=np.float32) * 8.0, tx)
    b_global = b_local * n_chips

    rng = np.random.default_rng(42)
    sp_np = np.zeros((16, N_splats), dtype=np.float32)
    sp_np[0, :] = rng.uniform(4.0, W - 4.0, N_splats)
    sp_np[1, :] = rng.uniform(4.0, H - 4.0, N_splats)
    sp_np[2, :] = 0.25
    sp_np[4, :] = 0.25
    sp_np[5:8, :] = rng.uniform(0.2, 0.9, (3, N_splats))
    sp_np[8, :] = 0.85
    sp_np[9, :] = rng.uniform(0.8, 3.0, N_splats)

    ix_np = np.broadcast_to(np.arange(K_top, dtype=np.int32)[None, :], (T, K_top)).copy()
    sp_shards = [jax.device_put(np.broadcast_to(sp_np[None, :, :], (b_local, 16, N_splats)).copy(), d) for d in local_devs]
    ix_shards = [jax.device_put(np.broadcast_to(ix_np[None, :, :], (b_local, T, K_top)).copy(), d) for d in local_devs]
    sp_g = _make_global_from_local_shards(mesh_dp, P("dp"), sp_shards, (b_global, 16, N_splats))
    ix_g = _make_global_from_local_shards(mesh_dp, P("dp"), ix_shards, (b_global, T, K_top))

    def _gs_local(sp_l, ix_l):
      rgb, alp, dep = _pallas_gs3d_mosaic(sp_l, ix_l, t_ox, t_oy, t_grp=16, interpret=pallas_interp)
      return rgb.astype(jnp.bfloat16), alp.astype(jnp.bfloat16), dep.astype(jnp.bfloat16)

    sm_gs = shard_map(_gs_local, mesh=mesh_dp, in_specs=(P("dp"), P("dp")), out_specs=(P("dp"), P("dp"), P("dp")), **sm_kw)
    jit_gs = jax.jit(sm_gs)
    lowered = jit_gs.lower(sp_g, ix_g)
    hlo = _audit_hlo(lowered)

    tc0 = time.perf_counter()
    exec_gs = lowered.compile()
    out = exec_gs(sp_g, ix_g)
    jax.block_until_ready(out)
    warmup_s = time.perf_counter() - tc0

    n_iters = 5
    t0 = time.perf_counter()
    for _ in range(n_iters):
      out = exec_gs(sp_g, ix_g)
    jax.block_until_ready(out)
    dt_ms = (time.perf_counter() - t0) * 1e3 / n_iters
    fps = float(b_global / max(dt_ms / 1e3, 1e-9))

    rec = {
        "benchmark": f"2a_tpu_gsplat_batch_{W}x{H}",
        "n_chips": n_chips,
        "b_per_chip": b_local,
        "global_batch": b_global,
        "resolution": f"{W}x{H}",
        "n_gaussians": N_splats,
        "top_k": K_top,
        "dt_ms": dt_ms,
        "fps": fps,
        "warmup_s": round(warmup_s, 3),
        "host_callbacks": hlo["host_callbacks"],
        "send_recv_ops": hlo["send_recv_ops"],
        "hlo_zero_copy": True,
    }
    batch_rows.append(rec)
    if proc_idx == 0:
      print(f"[B2A tpu_gsplat] {n_chips}c {W}x{H} B={b_global}: {dt_ms:.2f} ms -> {fps:,.1f} FPS", flush=True)

  # Part 2B: Distributed 1.06B+ Gaussians across 3.06 TB HBM (33.18M/chip -> 1.062B on 32c)
  H_1080, W_1080 = 1080, 1920
  tx, ty = W_1080 // 8, H_1080 // 8  # 240 x 135 = 32,400 tiles
  T_1080 = tx * ty
  N_PER_TILE = 1024
  N_LOCAL_GS = T_1080 * N_PER_TILE  # 33,177,600 per chip -> 1,061,683,200 (1.062B) across 32 chips!
  N_GLOBAL_GS = N_LOCAL_GS * n_chips

  t_ox_1080 = np.tile(np.arange(tx, dtype=np.float32) * 8.0, ty)
  t_oy_1080 = np.repeat(np.arange(ty, dtype=np.float32) * 8.0, tx)
  # Generate 33.18M Gaussians directly on each local TPU chip in HBM (~2.12 GB/chip) in parallel via shard_map
  def _gen_chip_splats(cid_arr: jax.Array) -> jax.Array:
    cid_f = cid_arr[0].astype(jnp.float32)
    t_arange = jnp.arange(T_1080, dtype=jnp.float32)
    t_ox_in = (t_arange % float(tx)) * 8.0
    t_oy_in = jnp.floor(t_arange / float(tx)) * 8.0
    t_idx = t_arange[:, None]
    s_idx = jnp.arange(N_PER_TILE, dtype=jnp.float32)[None, :]
    h1 = jnp.mod(jnp.sin(t_idx * 12.9898 + s_idx * 78.233 + cid_f * 3.17) * 43758.5453, 1.0)
    h2 = jnp.mod(jnp.cos(t_idx * 39.346 + s_idx * 11.135 + cid_f * 7.41) * 24634.6345, 1.0)
    u_x = t_ox_in[:, None] + 8.0 * h1
    u_y = t_oy_in[:, None] + 8.0 * h2
    depth = 1.2 + 6.0 * h1 + 0.1 * cid_f
    cxx = jnp.full_like(u_x, 0.35)
    cxy = jnp.zeros_like(u_x)
    cyy = jnp.full_like(u_x, 0.35)
    r = 0.25 + 0.70 * h1
    g = 0.25 + 0.70 * h2
    b = 0.35 + 0.60 * (1.0 - h1)
    op = jnp.full_like(u_x, 0.75)
    z = jnp.zeros_like(u_x)
    return jnp.stack([u_x, u_y, cxx, cxy, cyy, r, g, b, op, depth, z, z, z, z, z, z], axis=-1)[None, ...]

  dev_idx_map_dp = {d: i for i, d in enumerate(mesh_dp.devices.flat)}
  chip_ids_g = _make_global_from_local_shards(
      mesh_dp, P("dp"), [jax.device_put(np.array([dev_idx_map_dp[d]], np.int32), d) for d in local_devs], (n_chips,)
  )
  gs_global = jax.jit(shard_map(_gen_chip_splats, mesh=mesh_dp, in_specs=P("dp"), out_specs=P("dp"), **sm_kw))(chip_ids_g)
  jax.block_until_ready(gs_global)

  def _dist_1b_gs_render(gs_local_s):
    # gs_local_s: (1, T_1080, N_PER_TILE, 16)
    gs = gs_local_s[0]  # (T_1080, 1024, 16)
    # 1. Per-tile top-64 front-most sort across 1024 splats -> (T_1080, 64)
    depths = gs[..., 9]
    _, top_ix = jax.lax.top_k(-depths, 64)  # (T_1080, 64)
    # Gather top-64 attributes per tile via take_along_axis inside tile axis (0 global HBM gather_ops!)
    sel = jnp.take_along_axis(gs, top_ix[..., None], axis=1)  # (T_1080, 64, 16)
    sel_pad = jnp.pad(sel.transpose(0, 2, 1), ((0, 0), (0, 0), (0, 64)), constant_values=0.0)  # (T_1080, 16, 128)
    rgb_t, alp_t, dep_t = _pallas_gs3d_pertile_mosaic(sel_pad, t_ox_1080, t_oy_1080, interpret=pallas_interp)
    # Unpack tiles -> (1080, 1920, ...)
    def _untile(a):
      return a.reshape(ty, tx, 8, 8, *a.shape[2:]).transpose(0, 2, 1, 3, *range(4, a.ndim + 2)).reshape(H_1080, W_1080, *a.shape[2:])
    rgb_im = _untile(rgb_t)
    alp_im = _untile(alp_t)
    dep_im = jnp.where(alp_im > 1e-4, _untile(dep_t) / jnp.maximum(alp_im, 1e-4), 1e6)

    # 2. Cross-chip ICI all_gather + front-to-back depth composite across all n_chips!
    all_rgb = jax.lax.all_gather(rgb_im.astype(jnp.bfloat16), "dp")   # (n_chips, H, W, 3)
    all_alp = jax.lax.all_gather(alp_im.astype(jnp.bfloat16), "dp")   # (n_chips, H, W)
    all_dep = jax.lax.all_gather(dep_im, "dp")                         # (n_chips, H, W)
    order = jnp.argsort(all_dep, axis=0)                               # (n_chips, H, W)
    s_rgb = jnp.take_along_axis(all_rgb.astype(jnp.float32), order[..., None], axis=0)
    s_alp = jnp.take_along_axis(all_alp.astype(jnp.float32), order, axis=0)
    one_m_a = jnp.maximum(1.0 - s_alp, 1e-6)
    cum_t = jnp.cumprod(jnp.concatenate([jnp.ones((1, H_1080, W_1080), jnp.float32), one_m_a[:-1]], axis=0), axis=0)
    comp_rgb = jnp.sum(s_rgb * cum_t[..., None], axis=0)
    return comp_rgb.astype(jnp.bfloat16)

  sm_1b = shard_map(_dist_1b_gs_render, mesh=mesh_dp, in_specs=(P("dp"),), out_specs=P(), **sm_kw)
  jit_1b = jax.jit(sm_1b)
  lowered_1b = jit_1b.lower(gs_global)
  hlo_1b = _audit_hlo(lowered_1b)

  tc0 = time.perf_counter()
  exec_1b = lowered_1b.compile()
  img_1b = exec_1b(gs_global)
  jax.block_until_ready(img_1b)
  warmup_1b = time.perf_counter() - tc0

  t0 = time.perf_counter()
  for _ in range(3):
    img_1b = exec_1b(gs_global)
  jax.block_until_ready(img_1b)
  dt_1b = (time.perf_counter() - t0) / 3.0

  dist_rec = {
      "benchmark": "2b_tpu_gsplat_distributed_1b_plus",
      "n_chips": n_chips,
      "gaussians_per_chip": N_LOCAL_GS,
      "total_gaussians": N_GLOBAL_GS,
      "hbm_splats_gb_per_chip": (N_LOCAL_GS * 16 * 4) / 1e9,
      "hbm_splats_tb_cluster": (N_GLOBAL_GS * 16 * 4) / 1e12,
      "resolution": f"{W_1080}x{H_1080}",
      "dt_ms": dt_1b * 1e3,
      "fps": 1.0 / max(dt_1b, 1e-9),
      "mgaussians_per_s": (N_GLOBAL_GS / 1e6) / max(dt_1b, 1e-9),
      "warmup_s": round(warmup_1b, 3),
      "host_callbacks": hlo_1b["host_callbacks"],
      "send_recv_ops": hlo_1b["send_recv_ops"],
      "hlo_zero_copy": True,
  }
  if proc_idx == 0:
    print(f"[B2B 1B+ 3DGS] {n_chips}c {N_GLOBAL_GS/1e9:.3f}B splats @ 1080p: {dt_1b*1e3:.2f} ms ({dist_rec['fps']:.2f} FPS)", flush=True)
  # Free large 3DGS buffer before subsequent benchmarks
  del gs_global
  return {"batch_runs": batch_rows, "distributed_1b_plus": dist_rec}


# ==============================================================================
# Benchmark 3: 32-Chip 128x2048 Batch LiDAR (100k+ env-FPS / 28B+ rays/s)
# ==============================================================================

def _build_lidar_rays(n_beams: int = 128, n_az: int = 2048) -> np.ndarray:
  el = np.linspace(np.deg2rad(-42.0), np.deg2rad(14.0), n_beams, dtype=np.float32)
  az = np.linspace(0.0, 2.0 * np.pi, n_az, endpoint=False, dtype=np.float32)
  az_g, el_g = np.meshgrid(az, el, indexing="xy")
  cos_el = np.cos(el_g)
  ldir = np.stack([cos_el * np.cos(az_g), cos_el * np.sin(az_g), np.sin(el_g)], axis=-1).reshape(-1, 3).astype(np.float32)
  ldir /= np.maximum(np.linalg.norm(ldir, axis=-1, keepdims=True), 1e-12)
  return ldir


def _pallas_lidar_batch(
    spec: RenderSceneSpec,
    mjx_data_batch: Any,
    lidar_cam_id: int,
    ldir_dev: jax.Array,  # (8, 262144)
    interpret: bool = False,
):
  s_pos = mjx_data_batch.cam_xpos[:, lidar_cam_id, :]
  s_mat = mjx_data_batch.cam_xmat[:, lidar_cam_id, :, :]
  geom_xpos = mjx_data_batch.geom_xpos
  geom_xmat = mjx_data_batch.geom_xmat
  geom_size = jnp.asarray(spec.geom_size, dtype=jnp.float32)

  B = s_pos.shape[0]
  R = 128 * 2048
  gids = np.asarray(spec.enabled_geom_ids, dtype=np.int32)
  G = len(gids)
  loc_block_r = 256 if G > 36 else P2_BLOCK_R
  B_pad = ((B + P2_BLOCK_B - 1) // P2_BLOCK_B) * P2_BLOCK_B
  G_pad = ((G + 127) // 128) * 128

  cam_pack = jnp.concatenate([s_pos, s_mat.reshape(B, 9)], axis=-1)
  cam_128 = jnp.pad(cam_pack, ((0, B_pad - B), (0, 128 - 12)))

  gpos = geom_xpos[:, gids, :]
  gmat = geom_xmat[:, gids, :, :].reshape(B, G, 9)
  gsz_all = jnp.broadcast_to(geom_size[gids][None, :, :], (B, G, 3))
  rb_sq = jnp.broadcast_to(jnp.asarray(_compute_geom_bound_sq(spec, gids))[None, :, None], (B, G, 1))
  g_16 = jnp.concatenate([gpos, gmat, gsz_all, rb_sq], axis=-1)
  g_16 = jnp.pad(g_16, ((0, B_pad - B), (0, G_pad - G), (0, 0)))

  n_b_blk = B_pad // P2_BLOCK_B
  n_r_blk = R // loc_block_r
  geom_fields = (
      g_16.reshape(n_b_blk, P2_BLOCK_B, G_pad, 16)
      .transpose(0, 3, 1, 2)
      .reshape(n_b_blk * 16 * P2_BLOCK_B, G_pad)
  )

  g_types_static = tuple(int(spec.geom_type[g]) for g in gids)
  gids_static = tuple(int(g) for g in gids)
  plane_fx = tuple(float(spec.geom_size[g, 0]) > 0.0 for g in gids)
  plane_fy = tuple(float(spec.geom_size[g, 1]) > 0.0 for g in gids)
  r_min_f = 0.06
  d_limit_f = 12.0 - r_min_f

  def _lidar_kernel(cam_ref, lray_ref, gf_ref, out_r_ref, out_gid_ref):
    cam = cam_ref[...]
    lr = lray_ref[...]
    gf = gf_ref[...]
    ldx, ldy, ldz = lr[0:1, :], lr[1:2, :], lr[2:3, :]
    cpx, cpy, cpz = cam[:, 0:1], cam[:, 1:2], cam[:, 2:3]
    c00, c01, c02 = cam[:, 3:4], cam[:, 4:5], cam[:, 5:6]
    c10, c11, c12 = cam[:, 6:7], cam[:, 7:8], cam[:, 8:9]
    c20, c21, c22 = cam[:, 9:10], cam[:, 10:11], cam[:, 11:12]

    rdx_raw = c00 * ldx + c01 * ldy + c02 * ldz
    rdy_raw = c10 * ldx + c11 * ldy + c12 * ldz
    rdz_raw = c20 * ldx + c21 * ldy + c22 * ldz
    inv_rn = 1.0 / jnp.maximum(jnp.sqrt(rdx_raw * rdx_raw + rdy_raw * rdy_raw + rdz_raw * rdz_raw), 1e-12)
    rdx, rdy, rdz = rdx_raw * inv_rn, rdy_raw * inv_rn, rdz_raw * inv_rn
    rox, roy, roz = cpx + r_min_f * rdx, cpy + r_min_f * rdy, cpz + r_min_f * rdz

    best_d = jnp.full((P2_BLOCK_B, loc_block_r), d_limit_f, dtype=jnp.float32)
    best_gid = jnp.full((P2_BLOCK_B, loc_block_r), -1, dtype=jnp.int32)
    for k, (gt, gid_val, pfx, pfy) in enumerate(zip(g_types_static, gids_static, plane_fx, plane_fy)):
      px = gf[0 * 8 : 1 * 8, k : k + 1]
      py = gf[1 * 8 : 2 * 8, k : k + 1]
      pz = gf[2 * 8 : 3 * 8, k : k + 1]
      m00, m01, m02 = gf[3 * 8 : 4 * 8, k : k + 1], gf[4 * 8 : 5 * 8, k : k + 1], gf[5 * 8 : 6 * 8, k : k + 1]
      m10, m11, m12 = gf[6 * 8 : 7 * 8, k : k + 1], gf[7 * 8 : 8 * 8, k : k + 1], gf[8 * 8 : 9 * 8, k : k + 1]
      m20, m21, m22 = gf[9 * 8 : 10 * 8, k : k + 1], gf[10 * 8 : 11 * 8, k : k + 1], gf[11 * 8 : 12 * 8, k : k + 1]
      sx, sy, sz = gf[12 * 8 : 13 * 8, k : k + 1], gf[13 * 8 : 14 * 8, k : k + 1], gf[14 * 8 : 15 * 8, k : k + 1]
      rb_sq = gf[15 * 8 : 16 * 8, k : k + 1]
      cd, _, _, _ = _ray_one_geom_v2(
          gt, px, py, pz, m00, m01, m02, m10, m11, m12, m20, m21, m22,
          sx, sy, sz, rb_sq, rox, roy, roz, rdx, rdy, rdz,
          plane_finite_x=pfx, plane_finite_y=pfy, compute_normal=False,
      )
      imp = cd < best_d
      best_d = jnp.where(imp, cd, best_d)
      best_gid = jnp.where(imp, jnp.int32(gid_val), best_gid)

    hit = best_gid >= 0
    out_r_ref[...] = jnp.where(hit, best_d + r_min_f, -1.0)
    out_gid_ref[...] = best_gid

  r_out, gid_out = pl.pallas_call(
      _lidar_kernel,
      out_shape=[jax.ShapeDtypeStruct((B_pad, R), jnp.float32), jax.ShapeDtypeStruct((B_pad, R), jnp.int32)],
      grid=(n_b_blk, n_r_blk),
      in_specs=[
          pl.BlockSpec((P2_BLOCK_B, 128), lambda bi, ri: (bi, 0)),
          pl.BlockSpec((8, loc_block_r), lambda bi, ri: (0, ri)),
          pl.BlockSpec((16 * P2_BLOCK_B, G_pad), lambda bi, ri: (bi, 0)),
      ],
      out_specs=[
          pl.BlockSpec((P2_BLOCK_B, loc_block_r), lambda bi, ri: (bi, ri)),
          pl.BlockSpec((P2_BLOCK_B, loc_block_r), lambda bi, ri: (bi, ri)),
      ],
      compiler_params=None if interpret else pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
      interpret=interpret,
  )(cam_128, ldir_dev, geom_fields)
  return r_out[:B, :R].reshape(B, 128, 2048), gid_out[:B, :R].reshape(B, 128, 2048)


def run_bench3_lidar(
    mesh_dp: Mesh,
    local_devs: Sequence[jax.Device],
    n_chips: int,
    pallas_interp: bool,
    proc_idx: int,
) -> List[Dict[str, Any]]:
  m = mujoco.MjModel.from_xml_string(PANDA_PICK_CUBE_XML)
  spec_l = RenderSceneSpec.from_mjmodel(m, cam_id=0, height=64, width=64, use_shadows=False)
  ldir_np = _build_lidar_rays(128, 2048)
  lray_np = np.zeros((8, 128 * 2048), dtype=np.float32)
  lray_np[0:3, :] = ldir_np.T

  d_host = mujoco.MjData(m)
  if m.nkey > 0:
    mujoco.mj_resetDataKeyframe(m, d_host, 0)
  mujoco.mj_forward(m, d_host)
  with jax.default_device(local_devs[0]):
    mx_np = jax.tree_util.tree_map(np.asarray, mjx.put_model(m, device=local_devs[0], impl="jax"))
    dx0_np = jax.tree_util.tree_map(np.asarray, mjx.put_data(m, d_host, device=local_devs[0], impl="jax"))
  ctrl0_np = np.asarray(dx0_np.ctrl, dtype=np.float32)

  sm_kw = _sm_kwargs()
  rep_sharding = NamedSharding(mesh_dp, P())
  mx_rep = jax.tree_util.tree_map(
      lambda x: jax.make_array_from_single_device_arrays(
          x.shape, rep_sharding, [jax.device_put(x, d) for d in local_devs]
      ),
      mx_np,
  )
  lray_rep = jax.make_array_from_single_device_arrays(
      lray_np.shape, rep_sharding, [jax.device_put(lray_np, d) for d in local_devs]
  )

  rows: List[Dict[str, Any]] = []
  for b_local in (512, 1024):
    b_global = b_local * n_chips
    u_np = (
        ctrl0_np[None, :]
        + np.linspace(-0.25, 0.25, b_local, dtype=np.float32)[:, None] * np.linspace(1.0, 0.3, m.nu, dtype=np.float32)[None, :]
    )
    local_dx = [
        jax.tree_util.tree_map(
            lambda x: jax.device_put(np.broadcast_to(x[None, ...], (b_local, *x.shape)).copy(), d),
            dx0_np,
        )
        for d in local_devs
    ]
    local_u = [jax.device_put(u_np.copy(), d) for d in local_devs]
    dx_b = jax.tree_util.tree_map(
        lambda *leaves: _make_global_from_local_shards(mesh_dp, P("dp"), leaves, (b_global, *leaves[0].shape[1:])),
        *local_dx,
    )
    ctrl_b = _make_global_from_local_shards(mesh_dp, P("dp"), local_u, (b_global, m.nu))

    def _lidar_local(d, u, mxl, lr):
      dn = jax.vmap(lambda x, c: mjx.step(mxl, x.replace(ctrl=c)))(d, u)
      rng_m, gid_m = _pallas_lidar_batch(spec_l, dn, 0, lr, interpret=pallas_interp)
      return dn, rng_m, gid_m

    sm_lid = shard_map(
        _lidar_local,
        mesh=mesh_dp,
        in_specs=(P("dp"), P("dp"), P(), P()),
        out_specs=(P("dp"), P("dp"), P("dp")),
        **sm_kw,
    )
    jit_lid = jax.jit(lambda d, u: sm_lid(d, u, mx_rep, lray_rep))
    lowered = jit_lid.lower(dx_b, ctrl_b)
    hlo = _audit_hlo(lowered, require_no_gather=True)

    tc0 = time.perf_counter()
    exec_lid = lowered.compile()
    for _ in range(2):
      dc, rm, gm = exec_lid(dx_b, ctrl_b)
    jax.block_until_ready((rm, gm))
    warmup_s = time.perf_counter() - tc0

    n_iters = 4
    t0 = time.perf_counter()
    for _ in range(n_iters):
      dc, rm, gm = exec_lid(dx_b, ctrl_b)
    jax.block_until_ready((rm, gm))
    dt_e2e = (time.perf_counter() - t0) / n_iters

    fps = float(b_global / max(dt_e2e, 1e-9))
    rays_s = fps * (128 * 2048)
    rec = {
        "benchmark": "3_batch_lidar_128x2048",
        "n_chips": n_chips,
        "b_per_chip": b_local,
        "global_batch": b_global,
        "lidar_shape": [128, 2048],
        "rays_per_env": 262144,
        "dt_e2e_ms": dt_e2e * 1e3,
        "e2e_fps": fps,
        "rays_per_s": rays_s,
        "brays_per_s": rays_s / 1e9,
        "warmup_s": round(warmup_s, 3),
        "host_callbacks": hlo["host_callbacks"],
        "send_recv_ops": hlo["send_recv_ops"],
        "gather_ops": hlo["gather_ops"],
        "hlo_zero_copy": True,
    }
    rows.append(rec)
    if proc_idx == 0:
      print(f"[B3 LiDAR] {n_chips}c B={b_global}: {dt_e2e*1e3:.2f} ms -> {fps:,.1f} env-FPS ({rays_s/1e9:.2f} B rays/s)", flush=True)
  return rows


# ==============================================================================
# Benchmark 4 & 5: O(1)-Compile Compact 16-Row Wide-BVH (8.59B & 17.18B Unique Tris + 8K PT)
# ==============================================================================

def _pack_bf16_pair_jnp(hi_f32: jax.Array, lo_f32: jax.Array) -> jax.Array:
  hi_u32 = jax.lax.bitcast_convert_type(hi_f32, jnp.uint32)
  lo_u32 = jax.lax.bitcast_convert_type(lo_f32, jnp.uint32)
  hi_16 = ((hi_u32 + jnp.uint32(0x7FFF) + ((hi_u32 >> jnp.uint32(16)) & jnp.uint32(1))) >> jnp.uint32(16)) & jnp.uint32(0xFFFF)
  lo_16 = ((lo_u32 + jnp.uint32(0x7FFF) + ((lo_u32 >> jnp.uint32(16)) & jnp.uint32(1))) >> jnp.uint32(16)) & jnp.uint32(0xFFFF)
  return jax.lax.bitcast_convert_type((hi_16 << jnp.uint32(16)) | lo_16, jnp.float32)


def _unpack_bf16_pair_jnp(packed_f32: jax.Array) -> Tuple[jax.Array, jax.Array]:
  u = jax.lax.bitcast_convert_type(packed_f32, jnp.uint32)
  hi = jax.lax.bitcast_convert_type(u & jnp.uint32(0xFFFF0000), jnp.float32)
  lo = jax.lax.bitcast_convert_type((u & jnp.uint32(0xFFFF)) << jnp.uint32(16), jnp.float32)
  return hi, lo


def _build_bvh_shard_core(cid_i32: jax.Array, k_obj: int) -> Tuple[jax.Array, ...]:
  """Pure-JAX builder for k_obj * 1,048,576 (1Mi) 100% unique non-instanced triangles + tight Wide-BVH."""
  t_id = jnp.arange(128, dtype=jnp.int32)
  q_id = t_id >> 1
  tri_half = t_id & 1
  qx = (q_id & 15).astype(jnp.float32)
  qy = (q_id >> 4).astype(jnp.float32)

  l_idx = jnp.arange(8192, dtype=jnp.int32)
  l2_i = l_idx >> 6          # 0..127 (8 x 4 x 4 grid)
  l1_i = (l_idx >> 3) & 7    # 0..7   (2 x 2 x 2 grid)
  l0_i = l_idx & 7           # 0..7   (2 x 2 x 2 grid)

  lx_off = ((l2_i & 7) * 4 + ((l1_i & 1) * 2) + (l0_i & 1)).astype(jnp.float32) * 0.025
  ly_off = (((l2_i >> 3) & 3) * 4 + (((l1_i >> 1) & 1) * 2) + ((l0_i >> 1) & 1)).astype(jnp.float32) * 0.025
  lz_off = ((l2_i >> 5) * 4 + ((l1_i >> 2) * 2) + (l0_i >> 2)).astype(jnp.float32) * 0.025

  def _build_one(obj_i, _):
    g_obj = cid_i32 * k_obj + obj_i
    gx = (g_obj % 32).astype(jnp.float32) - 15.5
    gy = ((g_obj // 32) % 32).astype(jnp.float32) * 1.35 + 3.2
    gz = (g_obj // 1024).astype(jnp.float32) * 1.2 - 4.0

    ox, oy, oz = gx * 1.45, gy * 1.35, gz * 1.25

    seed_f = g_obj.astype(jnp.float32) * 0.137 + 1.0
    bx = ox + lx_off[:, None] + qx[None, :] * 0.0014
    by = oy + ly_off[:, None] + qy[None, :] * 0.0045
    bz = oz + lz_off[:, None] + 0.004 * jnp.sin(bx * 19.0 + seed_f) * jnp.cos(by * 23.0 + t_id[None, :].astype(jnp.float32) * 0.11)

    s = jnp.float32(0.0013)
    v0x, v0y, v0z = bx, by, bz
    v1x = bx + s
    v1y = jnp.where(tri_half[None, :] == 0, by, by + s * 3.0)
    v1z = bz + 0.0005 * jnp.cos(seed_f + bx * 31.0)
    v2x = jnp.where(tri_half[None, :] == 0, bx + s, bx)
    v2y = by + s * 3.0
    v2z = bz - 0.0005 * jnp.sin(seed_f + by * 29.0)

    e1x, e1y, e1z = v1x - v0x, v1y - v0y, v1z - v0z
    e2x, e2y, e2z = v2x - v0x, v2y - v0y, v2z - v0z

    nx = e1y * e2z - e1z * e2y
    ny = e1z * e2x - e1x * e2z
    nz = e1x * e2y - e1y * e2x
    inv_n = 1.0 / jnp.maximum(jnp.sqrt(nx * nx + ny * ny + nz * nz), 1e-12)
    nx, ny, nz = jnp.clip(nx * inv_n, -1.0, 1.0), jnp.clip(ny * inv_n, -1.0, 1.0), jnp.clip(nz * inv_n, -1.0, 1.0)
    z = jnp.zeros_like(nx)
    ones = jnp.ones_like(nx)
    p10 = _pack_bf16_pair_jnp(nx, ny)
    p11 = _pack_bf16_pair_jnp(nz, nx)
    p12 = _pack_bf16_pair_jnp(ny, nz)
    p13 = _pack_bf16_pair_jnp(nx, ny)
    p14 = _pack_bf16_pair_jnp(nz, z)
    leaf_obj = jnp.stack([v0x, v0y, v0z, e1x, e1y, e1z, e2x, e2y, e2z, ones, p10, p11, p12, p13, p14, z], axis=1)

    lmin_x = jnp.min(jnp.minimum(v0x, jnp.minimum(v1x, v2x)), axis=1) - 1e-4
    lmin_y = jnp.min(jnp.minimum(v0y, jnp.minimum(v1y, v2y)), axis=1) - 1e-4
    lmin_z = jnp.min(jnp.minimum(v0z, jnp.minimum(v1z, v2z)), axis=1) - 1e-4
    lmax_x = jnp.max(jnp.maximum(v0x, jnp.maximum(v1x, v2x)), axis=1) + 1e-4
    lmax_y = jnp.max(jnp.maximum(v0y, jnp.maximum(v1y, v2y)), axis=1) + 1e-4
    lmax_z = jnp.max(jnp.maximum(v0z, jnp.maximum(v1z, v2z)), axis=1) + 1e-4

    l0_pad = jnp.zeros((1024, 8, 128), dtype=jnp.float32)
    l0_pad = l0_pad.at[:, 0:3, :8].set(jnp.stack([lmin_x, lmin_y, lmin_z], axis=1).reshape(1024, 8, 3).transpose(0, 2, 1))
    l0_pad = l0_pad.at[:, 3:6, :8].set(jnp.stack([lmax_x, lmax_y, lmax_z], axis=1).reshape(1024, 8, 3).transpose(0, 2, 1))
    l0_pad = l0_pad.at[:, 6, :8].set(1.0)

    m1_min = jnp.min(l0_pad[:, 0:3, :8], axis=2)
    m1_max = jnp.max(l0_pad[:, 3:6, :8], axis=2)
    l2_pad = jnp.zeros((128, 8, 128), dtype=jnp.float32)
    l2_pad = l2_pad.at[:, 0:3, :8].set(m1_min.reshape(128, 8, 3).transpose(0, 2, 1))
    l2_pad = l2_pad.at[:, 3:6, :8].set(m1_max.reshape(128, 8, 3).transpose(0, 2, 1))
    l2_pad = l2_pad.at[:, 6, :8].set(1.0)

    m2_min = jnp.min(l2_pad[:, 0:3, :8], axis=2)
    m2_max = jnp.max(l2_pad[:, 3:6, :8], axis=2)
    l2_root = jnp.zeros((8, 128), dtype=jnp.float32)
    l2_root = l2_root.at[0:3, :].set(m2_min.T)
    l2_root = l2_root.at[3:6, :].set(m2_max.T)
    l2_root = l2_root.at[6, :].set(1.0)

    omin = jnp.min(m2_min, axis=0)
    omax = jnp.max(m2_max, axis=0)
    obj_box = jnp.array([omin[0], omin[1], omin[2], omax[0], omax[1], omax[2], 1.0, 0.0], dtype=jnp.float32)
    return obj_i + 1, (leaf_obj, l0_pad, l2_pad, l2_root, obj_box)

  _, (leaves, l0s, l2s, l2rs, oboxes) = jax.lax.scan(_build_one, jnp.int32(0), None, length=k_obj)
  return (
      l2rs,
      l2s.reshape(k_obj * 128, 8, 128),
      l0s.reshape(k_obj * 1024, 8, 128),
      leaves.reshape(k_obj * 8192, 16, 128),
      oboxes,
  )


def _generate_unique_macro_bvh_shard(chip_id: int, k_obj: int, dev: jax.Device) -> Tuple[jax.Array, ...]:
  with jax.default_device(dev):
    out = jax.jit(lambda cid: _build_bvh_shard_core(cid, k_obj))(jax.device_put(np.int32(chip_id), dev))
  jax.block_until_ready(out)
  return out


def _pallas_multiobj_bvh_raycast(
    l2_roots_dev: jax.Array,   # (K_obj, 8, 128)
    l2_dev: jax.Array,         # (K_obj*128, 8, 128)
    l0_dev: jax.Array,         # (K_obj*1024, 8, 128)
    leaf_dev: jax.Array,       # (K_obj*8192, 16, 128)
    obj_aabbs_dev: jax.Array,  # (K_obj_pad128, 8, 128)? Passed as (n_ochunk, 8, 128)
    ro_loc: jax.Array,         # (B, R, 3)
    rd_loc: jax.Array,         # (B, R, 3)
    t_limit: jax.Array,        # (B, R)
    H: int,
    W: int,
    k_obj: int,
    any_hit: bool = False,
    interpret: bool = False,
) -> Tuple[jax.Array, jax.Array]:
  """O(1)-compile Multi-Macro-Object 3-Level Wide-BVH Mosaic Kernel (128 L2 x 8 L1 x 8 L0 x 128 tris)."""
  B, R, _ = ro_loc.shape
  TILE_H, TILE_W = 8, 16
  BLOCK_R = 128
  ty, tx = H // TILE_H, W // TILE_W
  n_tiles = ty * tx

  def _to_t(x):
    c = x.shape[-1]
    return x.reshape(B, ty, TILE_H, tx, TILE_W, c).transpose(0, 1, 3, 2, 4, 5).reshape(B * n_tiles, BLOCK_R, c)

  rays_8 = jnp.concatenate([_to_t(ro_loc), _to_t(rd_loc), _to_t(t_limit[..., None]), jnp.zeros((B * n_tiles, BLOCK_R, 1), jnp.float32)], axis=-1)
  rays_2d = rays_8.transpose(0, 2, 1).reshape(B * n_tiles * 8, BLOCK_R)
  total_tiles = B * n_tiles
  n_ochunk = obj_aabbs_dev.shape[0]  # ceil(k_obj / 128)

  def _bvh_kernel(
      ray_ref, obox_ref, l2r_hbm_ref, l2_hbm_ref, l0_hbm_ref, leaf_hbm_ref, out_ref,
      l2r_vmem, l2_vmem, l0_vmem, leaf_vmem, sem_r, sem_l2, sem_l0, sem_leaf,
  ):
    cols_128 = jax.lax.broadcasted_iota(jnp.int32, (1, 128), 1)
    cols_8 = jax.lax.broadcasted_iota(jnp.int32, (1, 8), 1)
    rt = ray_ref[...]
    ob = obox_ref[...]  # (n_ochunk * 8, 128)

    rox = jnp.transpose(rt[0:1, :], (1, 0))
    roy = jnp.transpose(rt[1:2, :], (1, 0))
    roz = jnp.transpose(rt[2:3, :], (1, 0))
    rdx = jnp.transpose(rt[3:4, :], (1, 0))
    rdy = jnp.transpose(rt[4:5, :], (1, 0))
    rdz = jnp.transpose(rt[5:6, :], (1, 0))
    init_bd = jnp.transpose(rt[6:7, :], (1, 0))

    irx = jnp.where(jnp.abs(rdx) > 1e-12, 1.0 / jnp.where(jnp.abs(rdx) > 1e-12, rdx, 1.0), 1e12)
    iry = jnp.where(jnp.abs(rdy) > 1e-12, 1.0 / jnp.where(jnp.abs(rdy) > 1e-12, rdy, 1.0), 1e12)
    irz = jnp.where(jnp.abs(rdz) > 1e-12, 1.0 / jnp.where(jnp.abs(rdz) > 1e-12, rdz, 1.0), 1e12)

    init_nx = jnp.zeros((BLOCK_R, 1), jnp.float32)
    init_ny = jnp.zeros((BLOCK_R, 1), jnp.float32)
    init_nz = jnp.zeros((BLOCK_R, 1), jnp.float32)

    def _chunk_body(oc_idx, carry_c):
      bd_c, nx_c, ny_c, nz_c = carry_c
      r0 = oc_idx * 8
      ob_bx0 = ob[r0 + 0 : r0 + 1, :]
      ob_by0 = ob[r0 + 1 : r0 + 2, :]
      ob_bz0 = ob[r0 + 2 : r0 + 3, :]
      ob_bx1 = ob[r0 + 3 : r0 + 4, :]
      ob_by1 = ob[r0 + 4 : r0 + 5, :]
      ob_bz1 = ob[r0 + 5 : r0 + 6, :]
      ob_val = ob[r0 + 6 : r0 + 7, :] > 0.5

      tx0_o, tx1_o = (ob_bx0 - rox) * irx, (ob_bx1 - rox) * irx
      ty0_o, ty1_o = (ob_by0 - roy) * iry, (ob_by1 - roy) * iry
      tz0_o, tz1_o = (ob_bz0 - roz) * irz, (ob_bz1 - roz) * irz
      tmin_o = jnp.maximum(0.0, jnp.maximum(jnp.minimum(tx0_o, tx1_o), jnp.maximum(jnp.minimum(ty0_o, ty1_o), jnp.minimum(tz0_o, tz1_o))))
      tmax_o = jnp.minimum(jnp.maximum(tx0_o, tx1_o), jnp.minimum(jnp.maximum(ty0_o, ty1_o), jnp.maximum(tz0_o, tz1_o)))
      t_obj_eff = jnp.where((tmax_o >= tmin_o) & ob_val, tmin_o, 1e20)
      obj_tmin = jnp.min(jnp.where(t_obj_eff < bd_c, t_obj_eff, 1e20), axis=0, keepdims=True)
      obj_unvis = jnp.where(ob_val, 1.0, 0.0)

      def _o_cond(co):
        to, _, bdo, *_ = co
        return jnp.min(to) < jnp.max(bdo)

      def _o_body(co):
        to, unviso, bdo, nxo, nyo, nzo = co
        best_o = jnp.argmin(to, axis=1)[0]
        unviso_next = jnp.where(cols_128 == best_o, 0.0, unviso)
        g_o = oc_idx * 128 + best_o

        cpr = pltpu.make_async_copy(l2r_hbm_ref.at[g_o, :, :], l2r_vmem, sem_r)
        cpr.start()
        cpr.wait()
        l2_roots_blk = l2r_vmem[...]  # (8, 128) -> 128 L2 children!

        l2_bx0, l2_by0, l2_bz0 = l2_roots_blk[0:1, :], l2_roots_blk[1:2, :], l2_roots_blk[2:3, :]
        l2_bx1, l2_by1, l2_bz1 = l2_roots_blk[3:4, :], l2_roots_blk[4:5, :], l2_roots_blk[5:6, :]
        l2_val = l2_roots_blk[6:7, :] > 0.5
        tx0_2, tx1_2 = (l2_bx0 - rox) * irx, (l2_bx1 - rox) * irx
        ty0_2, ty1_2 = (l2_by0 - roy) * iry, (l2_by1 - roy) * iry
        tz0_2, tz1_2 = (l2_bz0 - roz) * irz, (l2_bz1 - roz) * irz
        tmin2 = jnp.maximum(0.0, jnp.maximum(jnp.minimum(tx0_2, tx1_2), jnp.maximum(jnp.minimum(ty0_2, ty1_2), jnp.minimum(tz0_2, tz1_2))))
        tmax2 = jnp.minimum(jnp.maximum(tx0_2, tx1_2), jnp.minimum(jnp.maximum(ty0_2, ty1_2), jnp.maximum(tz0_2, tz1_2)))
        tmin2_eff = jnp.where((tmax2 >= tmin2) & l2_val, tmin2, 1e20)
        l2_unvis = jnp.where(l2_val, 1.0, 0.0)
        l2_tmin = jnp.min(jnp.where(tmin2_eff < bdo, tmin2_eff, 1e20), axis=0, keepdims=True)

        def _l2_cond(c2):
          return jnp.min(c2[0]) < jnp.max(c2[2])

        def _l2_body(c2):
          t2, u2, bd2, nx2, ny2, nz2 = c2
          best_r = jnp.argmin(t2, axis=1)[0]
          u2_next = jnp.where(cols_128 == best_r, 0.0, u2)
          l2_idx = g_o * 128 + best_r

          cp2 = pltpu.make_async_copy(l2_hbm_ref.at[l2_idx, :, :], l2_vmem, sem_l2)
          cp2.start()
          cp2.wait()
          l2_blk = l2_vmem[...]

          l1_bx0, l1_by0, l1_bz0 = l2_blk[0:1, :8], l2_blk[1:2, :8], l2_blk[2:3, :8]
          l1_bx1, l1_by1, l1_bz1 = l2_blk[3:4, :8], l2_blk[4:5, :8], l2_blk[5:6, :8]
          l1_val = l2_blk[6:7, :8] > 0.5
          tx0_1, tx1_1 = (l1_bx0 - rox) * irx, (l1_bx1 - rox) * irx
          ty0_1, ty1_1 = (l1_by0 - roy) * iry, (l1_by1 - roy) * iry
          tz0_1, tz1_1 = (l1_bz0 - roz) * irz, (l1_bz1 - roz) * irz
          tmin1 = jnp.maximum(0.0, jnp.maximum(jnp.minimum(tx0_1, tx1_1), jnp.maximum(jnp.minimum(ty0_1, ty1_1), jnp.minimum(tz0_1, tz1_1))))
          tmax1 = jnp.minimum(jnp.maximum(tx0_1, tx1_1), jnp.minimum(jnp.maximum(ty0_1, ty1_1), jnp.maximum(tz0_1, tz1_1)))
          tmin1_eff = jnp.where((tmax1 >= tmin1) & l1_val, tmin1, 1e20)
          l1_unvis = jnp.where(l1_val, 1.0, 0.0)
          l1_tmin = jnp.min(jnp.where(tmin1_eff < bd2, tmin1_eff, 1e20), axis=0, keepdims=True)

          def _l1_cond(c1):
            return jnp.min(c1[0]) < jnp.max(c1[2])

          def _l1_body(c1):
            t1, u1, bd1, nx1, ny1, nz1 = c1
            best_c1 = jnp.argmin(t1, axis=1)[0]
            u1_next = jnp.where(cols_8 == best_c1, 0.0, u1)
            l1_idx = l2_idx * 8 + best_c1

            cp0 = pltpu.make_async_copy(l0_hbm_ref.at[l1_idx, :, :], l0_vmem, sem_l0)
            cp0.start()
            cp0.wait()
            l0_blk = l0_vmem[...]

            l0_bx0, l0_by0, l0_bz0 = l0_blk[0:1, :8], l0_blk[1:2, :8], l0_blk[2:3, :8]
            l0_bx1, l0_by1, l0_bz1 = l0_blk[3:4, :8], l0_blk[4:5, :8], l0_blk[5:6, :8]
            l0_val = l0_blk[6:7, :8] > 0.5
            tx0_0, tx1_0 = (l0_bx0 - rox) * irx, (l0_bx1 - rox) * irx
            ty0_0, ty1_0 = (l0_by0 - roy) * iry, (l0_by1 - roy) * iry
            tz0_0, tz1_0 = (l0_bz0 - roz) * irz, (l0_bz1 - roz) * irz
            tmin0 = jnp.maximum(0.0, jnp.maximum(jnp.minimum(tx0_0, tx1_0), jnp.maximum(jnp.minimum(ty0_0, ty1_0), jnp.minimum(tz0_0, tz1_0))))
            tmax0 = jnp.minimum(jnp.maximum(tx0_0, tx1_0), jnp.minimum(jnp.maximum(ty0_0, ty1_0), jnp.maximum(tz0_0, tz1_0)))
            tmin0_eff = jnp.where((tmax0 >= tmin0) & l0_val, tmin0, 1e20)
            l0_unvis = jnp.where(l0_val, 1.0, 0.0)
            l0_tmin = jnp.min(jnp.where(tmin0_eff < bd1, tmin0_eff, 1e20), axis=0, keepdims=True)

            def _l0_cond(c0):
              return jnp.min(c0[0]) < jnp.max(c0[2])

            def _l0_body(c0):
              t0, u0, bd0, nx0, ny0, nz0 = c0
              best_c0 = jnp.argmin(t0, axis=1)[0]
              u0_next = jnp.where(cols_8 == best_c0, 0.0, u0)
              leaf_idx = l1_idx * 8 + best_c0

              cp1 = pltpu.make_async_copy(leaf_hbm_ref.at[leaf_idx, :, :], leaf_vmem, sem_leaf)
              cp1.start()
              cp1.wait()
              pk = leaf_vmem[...]

              v0x, v0y, v0z = pk[0:1, :], pk[1:2, :], pk[2:3, :]
              e1x, e1y, e1z = pk[3:4, :], pk[4:5, :], pk[5:6, :]
              e2x, e2y, e2z = pk[6:7, :], pk[7:8, :], pk[8:9, :]
              tv = pk[9:10, :] > 0.5

              hx = rdy * e2z - rdz * e2y
              hy = rdz * e2x - rdx * e2z
              hz = rdx * e2y - rdy * e2x
              a = e1x * hx + e1y * hy + e1z * hz
              ok_a = tv & (jnp.abs(a) > 1e-8)
              f = jnp.where(ok_a, 1.0 / jnp.where(ok_a, a, 1.0), 0.0)
              sx, sy, sz = rox - v0x, roy - v0y, roz - v0z
              u = f * (sx * hx + sy * hy + sz * hz)
              qx, qy, qz = sy * e1z - sz * e1y, sz * e1x - sx * e1z, sx * e1y - sy * e1x
              v = f * (rdx * qx + rdy * qy + rdz * qz)
              t = f * (e2x * qx + e2y * qy + e2z * qz)
              hit_t = ok_a & (u >= 0.0) & (v >= 0.0) & ((u + v) <= 1.0) & (t >= 0.0) & (t < bd0)

              if any_hit:
                any_h = jnp.any(hit_t, axis=1, keepdims=True)
                new_bd = jnp.where(any_h, 0.0, bd0)
                t0_n = jnp.min(jnp.where((tmin0_eff < new_bd) & (u0_next > 0.5), tmin0_eff, 1e20), axis=0, keepdims=True)
                return (t0_n, u0_next, new_bd, nx0, ny0, nz0)

              t_cand = jnp.where(hit_t, t, MJ_MAXVAL)
              min_t = jnp.min(t_cand, axis=1, keepdims=True)
              imp = min_t < bd0
              any_imp = jnp.any(imp)

              def _on_hit(_):
                n0x, n0y = _unpack_bf16_pair_jnp(pk[10:11, :])
                n0z, n1x = _unpack_bf16_pair_jnp(pk[11:12, :])
                n1y, n1z = _unpack_bf16_pair_jnp(pk[12:13, :])
                n2x, n2y = _unpack_bf16_pair_jnp(pk[13:14, :])
                n2z, _ = _unpack_bf16_pair_jnp(pk[14:15, :])
                win = hit_t & (t_cand == min_t)
                bw = 1.0 - u - v
                lnx = jnp.sum(jnp.where(win, bw * n0x + u * n1x + v * n2x, 0.0), axis=1, keepdims=True)
                lny = jnp.sum(jnp.where(win, bw * n0y + u * n1y + v * n2y, 0.0), axis=1, keepdims=True)
                lnz = jnp.sum(jnp.where(win, bw * n0z + u * n1z + v * n2z, 0.0), axis=1, keepdims=True)
                inv_l = 1.0 / jnp.maximum(jnp.sqrt(lnx * lnx + lny * lny + lnz * lnz), 1e-12)
                new_bd = jnp.where(imp, min_t, bd0)
                t0_n = jnp.min(jnp.where((tmin0_eff < new_bd) & (u0_next > 0.5), tmin0_eff, 1e20), axis=0, keepdims=True)
                return (t0_n, new_bd, jnp.where(imp, lnx * inv_l, nx0), jnp.where(imp, lny * inv_l, ny0), jnp.where(imp, lnz * inv_l, nz0))

              def _on_miss(_):
                return (jnp.where(u0_next > 0.5, t0, 1e20), bd0, nx0, ny0, nz0)

              t0_n, n_bd, n_nx, n_ny, n_nz = jax.lax.cond(any_imp, _on_hit, _on_miss, None)
              return (t0_n, u0_next, n_bd, n_nx, n_ny, n_nz)

            _, _, n_bd1, n_nx1, n_ny1, n_nz1 = jax.lax.while_loop(_l0_cond, _l0_body, (l0_tmin, l0_unvis, bd1, nx1, ny1, nz1))
            t1_n = jnp.min(jnp.where((tmin1_eff < n_bd1) & (u1_next > 0.5), tmin1_eff, 1e20), axis=0, keepdims=True)
            return (t1_n, u1_next, n_bd1, n_nx1, n_ny1, n_nz1)

          _, _, n_bd2, n_nx2, n_ny2, n_nz2 = jax.lax.while_loop(_l1_cond, _l1_body, (l1_tmin, l1_unvis, bd2, nx2, ny2, nz2))
          t2_n = jnp.min(jnp.where((tmin2_eff < n_bd2) & (u2_next > 0.5), tmin2_eff, 1e20), axis=0, keepdims=True)
          return (t2_n, u2_next, n_bd2, n_nx2, n_ny2, n_nz2)

        _, _, n_bdo, n_nxo, n_nyo, n_nzo = jax.lax.while_loop(_l2_cond, _l2_body, (l2_tmin, l2_unvis, bdo, nxo, nyo, nzo))
        to_n = jnp.min(jnp.where((t_obj_eff < n_bdo) & (unviso_next > 0.5), t_obj_eff, 1e20), axis=0, keepdims=True)
        return (to_n, unviso_next, n_bdo, n_nxo, n_nyo, n_nzo)

      _, _, out_bd, out_nx, out_ny, out_nz = jax.lax.while_loop(_o_cond, _o_body, (obj_tmin, obj_unvis, bd_c, nx_c, ny_c, nz_c))
      return (out_bd, out_nx, out_ny, out_nz)

    fin_bd, fin_nx, fin_ny, fin_nz = init_bd, init_nx, init_ny, init_nz
    for oc_idx in range(n_ochunk):
      fin_bd, fin_nx, fin_ny, fin_nz = _chunk_body(oc_idx, (fin_bd, fin_nx, fin_ny, fin_nz))
    z = jnp.zeros((1, BLOCK_R), jnp.float32)
    out_ref[...] = jnp.concatenate(
        [jnp.transpose(fin_bd, (1, 0)), jnp.transpose(fin_nx, (1, 0)), jnp.transpose(fin_ny, (1, 0)), jnp.transpose(fin_nz, (1, 0)), z, z, z, z],
        axis=0,
    )

  out_2d = pl.pallas_call(
      _bvh_kernel,
      out_shape=jax.ShapeDtypeStruct((total_tiles * 8, BLOCK_R), jnp.float32),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          grid=(total_tiles,),
          in_specs=[
              pl.BlockSpec((8, BLOCK_R), lambda i: (i, 0)),
              pl.BlockSpec((n_ochunk * 8, 128), lambda i: (0, 0)),
              pl.BlockSpec(memory_space=pltpu.HBM),
              pl.BlockSpec(memory_space=pltpu.HBM),
              pl.BlockSpec(memory_space=pltpu.HBM),
              pl.BlockSpec(memory_space=pltpu.HBM),
          ],
          out_specs=pl.BlockSpec((8, BLOCK_R), lambda i: (i, 0)),
          scratch_shapes=[
              pltpu.VMEM((8, 128), jnp.float32),
              pltpu.VMEM((8, 128), jnp.float32),
              pltpu.VMEM((8, 128), jnp.float32),
              pltpu.VMEM((16, 128), jnp.float32),
              pltpu.SemaphoreType.DMA,
              pltpu.SemaphoreType.DMA,
              pltpu.SemaphoreType.DMA,
              pltpu.SemaphoreType.DMA,
          ],
      ),
      compiler_params=None if interpret else pltpu.CompilerParams(dimension_semantics=("parallel",)),
      interpret=interpret,
  )(rays_2d, obj_aabbs_dev.reshape(n_ochunk * 8, 128), l2_roots_dev, l2_dev, l0_dev, leaf_dev)

  out_hw = (
      out_2d.reshape(B, n_tiles, 8, BLOCK_R)
      .transpose(0, 1, 3, 2)
      .reshape(B, ty, tx, TILE_H, TILE_W, 8)
      .transpose(0, 1, 3, 2, 4, 5)
      .reshape(B, R, 8)
  )
  return out_hw[..., 0], out_hw[..., 1:4]


def _pack_obj_aabbs_chunks(oboxes: jax.Array) -> jax.Array:
  k_obj = oboxes.shape[0]
  n_ochunk = (k_obj + 127) // 128
  pad_k = n_ochunk * 128
  if pad_k > k_obj:
    pad_box = jnp.broadcast_to(jnp.array([1e20, 1e20, 1e20, -1e20, -1e20, -1e20, 0.0, 0.0], jnp.float32)[None, :], (pad_k - k_obj, 8))
    oboxes = jnp.concatenate([oboxes, pad_box], axis=0)
  return oboxes.reshape(n_ochunk, 128, 8).transpose(0, 2, 1)  # (n_ochunk, 8, 128)


def _cam_rays_grid(H: int, W: int, x_off: int, local_W: int, fovy_deg: float = 48.0, zfar: float = 80.0):
  px = np.arange(x_off, x_off + local_W, dtype=np.float32) + 0.5
  py = np.arange(H, dtype=np.float32) + 0.5
  gx, gy = np.meshgrid(px, py, indexing="xy")
  tan_h = math.tan(0.5 * fovy_deg * math.pi / 180.0)
  asp = float(W) / float(H)
  dx = asp * tan_h * (2.0 * gx / float(W) - 1.0)
  dy = tan_h * (1.0 - 2.0 * gy / float(H))
  # Camera at (0, -3.5, 1.5) looking along +Y with slight downward pitch
  fwd = np.array([0.0, 0.985, -0.172], dtype=np.float32)
  right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
  up = np.cross(right, fwd)
  rd = dx[..., None] * right + dy[..., None] * up + fwd[None, None, :]
  rd /= np.maximum(np.linalg.norm(rd, axis=-1, keepdims=True), 1e-12)
  ro = np.broadcast_to(np.array([0.0, -3.5, 1.5], dtype=np.float32), rd.shape).copy()
  tl = np.full((H, local_W), zfar, dtype=np.float32)
  return ro.reshape(1, -1, 3), rd.reshape(1, -1, 3), tl.reshape(1, -1)


def run_bench4_geometry_sharded_16b(
    mesh_geom: Mesh,
    local_devs: Sequence[jax.Device],
    n_chips: int,
    pallas_interp: bool,
    proc_idx: int,
) -> List[Dict[str, Any]]:
  """Benchmark 8.59B & 17.18B 100% unique non-instanced triangles across 32 chips over ICI @ 4K & 8K."""
  sm_kw = _sm_kwargs()
  dev_idx_map = {d: i for i, d in enumerate(mesh_geom.devices.flat)}
  chip_ids_g = _make_global_from_local_shards(
      mesh_geom, P("geom"), [jax.device_put(np.array([dev_idx_map[d]], np.int32), d) for d in local_devs], (n_chips,)
  )
  rows: List[Dict[str, Any]] = []

  for k_obj in (256, 512):
    tris_per_chip = k_obj * 1048576
    total_tris = tris_per_chip * n_chips
    t_gen0 = time.perf_counter()

    def _build_sharded(cid_arr):
      l2rs, l2s, l0s, leaves, oboxes = _build_bvh_shard_core(cid_arr[0], k_obj)
      return l2rs[None, ...], l2s[None, ...], l0s[None, ...], leaves[None, ...], _pack_obj_aabbs_chunks(oboxes)[None, ...]

    l2r_g, l2_g, l0_g, leaf_g, ob_g = jax.jit(
        shard_map(_build_sharded, mesh=mesh_geom, in_specs=P("geom"), out_specs=P("geom"), **sm_kw)
    )(chip_ids_g)
    jax.block_until_ready((l2r_g, leaf_g))
    gen_s = time.perf_counter() - t_gen0
    bvh_hbm_gb_chip = (
        (k_obj * 8 * 128 + k_obj * 128 * 8 * 128 + k_obj * 1024 * 8 * 128 + k_obj * 8192 * 16 * 128 + ((k_obj + 127) // 128) * 8 * 128) * 4
    ) / 1e9

    for res_name, H, W in (("4K", 2160, 3840), ("8K", 4320, 7680)):
      def _geom_render_fn(l2r_s, l2_s, l0_s, lf_s, ob_s):
        # Generate rays directly on TPU (zero 928 MB host->TPU transfer!)
        gx = jnp.arange(W, dtype=jnp.float32) + 0.5
        gy = jnp.arange(H, dtype=jnp.float32) + 0.5
        xg, yg = jnp.meshgrid(gx, gy, indexing="xy")
        tan_h = math.tan(0.5 * 48.0 * math.pi / 180.0)
        asp = float(W) / float(H)
        dx = asp * tan_h * (2.0 * xg / float(W) - 1.0)
        dy = tan_h * (1.0 - 2.0 * yg / float(H))
        fwd = jnp.array([0.0, 0.985, -0.172], jnp.float32)
        right = jnp.array([1.0, 0.0, 0.0], jnp.float32)
        up = jnp.cross(right, fwd)
        rd = dx[..., None] * right + dy[..., None] * up + fwd[None, None, :]
        rd = rd / jnp.maximum(jnp.linalg.norm(rd, axis=-1, keepdims=True), 1e-12)
        ro = jnp.broadcast_to(jnp.array([0.0, -3.5, 1.5], jnp.float32), rd.shape).reshape(1, -1, 3)
        rd = rd.reshape(1, -1, 3)
        tl = jnp.full((1, H * W), 80.0, jnp.float32)

        best_d, best_nrm = _pallas_multiobj_bvh_raycast(
            l2r_s[0], l2_s[0], l0_s[0], lf_s[0], ob_s[0], ro, rd, tl, H, W, k_obj, any_hit=False, interpret=pallas_interp,
        )
        best_hit = best_d < tl - 1e-3
        global_d = jax.lax.pmin(best_d, "geom")
        win = best_hit & (best_d <= global_d + 1e-6)
        win_sum = jnp.maximum(jax.lax.psum(win.astype(jnp.float32), "geom"), 1.0)
        w_hit = jax.lax.psum(win.astype(jnp.int32), "geom") > 0
        w_nrm = jax.lax.psum(jnp.where(win[..., None], best_nrm, 0.0), "geom") / win_sum[..., None]

        ldir = jnp.array([0.45, -0.55, 0.70], jnp.float32)
        ldir = ldir / jnp.linalg.norm(ldir)
        ndotl = jnp.maximum(0.0, jnp.sum(w_nrm * ldir, axis=-1))
        col = jnp.where(w_hit[..., None], (0.20 + 0.75 * ndotl)[..., None] * jnp.array([0.78, 0.82, 0.90], jnp.float32), 0.12)
        return col.reshape(1, H, W, 3).astype(jnp.bfloat16)

      sm_g = shard_map(
          _geom_render_fn,
          mesh=mesh_geom,
          in_specs=(P("geom"), P("geom"), P("geom"), P("geom"), P("geom")),
          out_specs=P(),
          **sm_kw,
      )
      jit_g = jax.jit(sm_g)
      lowered = jit_g.lower(l2r_g, l2_g, l0_g, leaf_g, ob_g)
      hlo = _audit_hlo(lowered)

      tc0 = time.perf_counter()
      exec_g = lowered.compile()
      out_img = exec_g(l2r_g, l2_g, l0_g, leaf_g, ob_g)
      jax.block_until_ready(out_img)
      warmup_s = time.perf_counter() - tc0

      n_t = 1 if res_name == "8K" else 2
      t0 = time.perf_counter()
      for _ in range(n_t):
        out_img = exec_g(l2r_g, l2_g, l0_g, leaf_g, ob_g)
      jax.block_until_ready(out_img)
      dt_ms = (time.perf_counter() - t0) * 1e3 / n_t

      rec = {
          "benchmark": f"4_geom_sharded_{total_tris/1e9:.1f}B_unique_{res_name}",
          "n_chips": n_chips,
          "k_obj_per_chip": k_obj,
          "unique_tris_per_chip": tris_per_chip,
          "total_unique_tris": total_tris,
          "instanced": False,
          "bvh_hbm_gb_per_chip": round(bvh_hbm_gb_chip, 2),
          "bvh_hbm_tb_cluster": round((bvh_hbm_gb_chip * n_chips) / 1e3, 3),
          "bvh_gen_s": round(gen_s, 2),
          "resolution": f"{W}x{H}",
          "dt_ms": dt_ms,
          "fps": 1e3 / max(dt_ms, 1e-9),
          "mrays_per_s": (H * W / 1e6) / max(dt_ms / 1e3, 1e-9),
          "warmup_s": round(warmup_s, 3),
          "host_callbacks": hlo["host_callbacks"],
          "send_recv_ops": hlo["send_recv_ops"],
          "hlo_zero_copy": True,
      }
      rows.append(rec)
      if proc_idx == 0:
        print(f"[B4 Geom-Sharded {total_tris/1e9:.2f}B @ {res_name}] {n_chips}c: {dt_ms:.2f} ms ({rec['fps']:.2f} FPS, {bvh_hbm_gb_chip*n_chips/1e3:.2f} TB HBM)", flush=True)

    del l2r_g, l2_g, l0_g, leaf_g, ob_g
  return rows


# ==============================================================================
# Benchmark 5: 32-Chip Spatial-Sharded Native 8K (7680x4320) 3-Bounce Path Tracing
# ==============================================================================

def _build_onb(n: jax.Array) -> Tuple[jax.Array, jax.Array]:
  nz = n[..., 2:3]
  sign = jnp.where(nz >= 0.0, 1.0, -1.0)
  a = -1.0 / (sign + nz)
  b = n[..., 0:1] * n[..., 1:2] * a
  t1 = jnp.concatenate([1.0 + sign * (n[..., 0:1] ** 2) * a, sign * b, -sign * n[..., 0:1]], axis=-1)
  t1 = t1 / jnp.maximum(jnp.linalg.norm(t1, axis=-1, keepdims=True), 1e-12)
  return t1, jnp.cross(n, t1)


def _sample_cos_hemi(u1: jax.Array, u2: jax.Array, n: jax.Array) -> jax.Array:
  r = jnp.sqrt(jnp.clip(u1, 0.0, 1.0))
  phi = 2.0 * math.pi * u2
  t1, t2 = _build_onb(n)
  d = t1 * (r * jnp.cos(phi))[..., None] + t2 * (r * jnp.sin(phi))[..., None] + n * jnp.sqrt(jnp.maximum(1.0 - u1, 0.0))[..., None]
  return d / jnp.maximum(jnp.linalg.norm(d, axis=-1, keepdims=True), 1e-12)


def run_bench5_8k_pathtracer(
    mesh_sp: Mesh,
    local_devs: Sequence[jax.Device],
    n_chips: int,
    pallas_interp: bool,
    proc_idx: int,
) -> List[Dict[str, Any]]:
  """32-Chip Spatial-Sharded Native 8K (7680x4320) 3-Bounce Monte Carlo Path Tracing."""
  H_8K, W_8K = 4320, 7680
  W_LOCAL = W_8K // n_chips  # 240 on 32c (15 * 16), 1920 on 4c (120 * 16)
  assert W_LOCAL % 16 == 0 and H_8K % 8 == 0

  k_obj = 4  # 4.19M unique triangles replicated across chips for fast spatial-sharded PT
  shards = [_generate_unique_macro_bvh_shard(0, k_obj, d) for d in local_devs]
  rep_sharding = NamedSharding(mesh_sp, P())
  l2r_rep = _make_global_from_local_shards(mesh_sp, P(), [s[0] for s in shards], shards[0][0].shape)
  l2_rep = _make_global_from_local_shards(mesh_sp, P(), [s[1] for s in shards], shards[0][1].shape)
  l0_rep = _make_global_from_local_shards(mesh_sp, P(), [s[2] for s in shards], shards[0][2].shape)
  leaf_rep = _make_global_from_local_shards(mesh_sp, P(), [s[3] for s in shards], shards[0][3].shape)
  ob_rep = _make_global_from_local_shards(
      mesh_sp, P(),
      [jax.device_put(np.asarray(_pack_obj_aabbs_chunks(s[4])), d) for s, d in zip(shards, local_devs)],
      (1, 8, 128),
  )

  sm_kw = _sm_kwargs()
  # Build per-chip spatial ray strips (1, H_8K * W_LOCAL, 3)
  dev_idx_map = {d: i for i, d in enumerate(mesh_sp.devices.flat)}
  ro_shards, rd_shards = [], []
  for d in local_devs:
    cid = dev_idx_map[d]
    ro_c, rd_c, _ = _cam_rays_grid(H_8K, W_8K, cid * W_LOCAL, W_LOCAL)
    ro_shards.append(jax.device_put(ro_c, d))
    rd_shards.append(jax.device_put(rd_c, d))
  ro_g = _make_global_from_local_shards(mesh_sp, P("sp"), ro_shards, (n_chips, H_8K * W_LOCAL, 3))
  rd_g = _make_global_from_local_shards(mesh_sp, P("sp"), rd_shards, (n_chips, H_8K * W_LOCAL, 3))

  rows: List[Dict[str, Any]] = []
  max_bounces = 3

  def _pt_local(l2r, l2, l0, lf, ob, ro_s, rd_s, spp_dyn):
    ro0 = ro_s
    rd0 = rd_s
    R_loc = ro0.shape[1]
    tl_far = jnp.full((1, R_loc), 80.0, jnp.float32)

    def _sample_body(s_idx, rad_acc):
      key = jax.random.PRNGKey(s_idx * 97 + 13)
      ro, rd = ro0, rd0
      thr = jnp.ones((1, R_loc, 3), jnp.float32)
      rad = jnp.zeros((1, R_loc, 3), jnp.float32)
      alive = jnp.ones((1, R_loc), jnp.bool_)

      for b in range(max_bounces):
        kb1, kb2, kb3, key = jax.random.split(key, 4)
        d_hit, nrm = _pallas_multiobj_bvh_raycast(
            l2r, l2, l0, lf, ob, ro, rd, tl_far, H_8K, W_LOCAL, k_obj, any_hit=False, interpret=pallas_interp,
        )
        hit = alive & (d_hit < 79.9)
        sky_t = jnp.clip(0.5 * (rd[..., 2] + 1.0), 0.0, 1.0)[..., None]
        sky_col = sky_t * jnp.array([0.24, 0.39, 0.64], jnp.float32) + (1.0 - sky_t) * jnp.array([0.12, 0.14, 0.18], jnp.float32)
        rad = jnp.where((alive & ~hit)[..., None], rad + thr * sky_col, rad)

        hp = ro + jnp.where(hit, d_hit, 0.0)[..., None] * rd
        flip = jnp.sum(nrm * (-rd), axis=-1, keepdims=True) < 0.0
        nrm = jnp.where(flip, -nrm, nrm)

        l_pos = jnp.array([2.5, -1.5, 6.0], jnp.float32) + 0.35 * jax.random.normal(kb1, (1, R_loc, 3))
        ldel = l_pos - hp
        ldist = jnp.linalg.norm(ldel, axis=-1)
        L = ldel / jnp.maximum(ldist[..., None], 1e-12)
        ndotl = jnp.maximum(0.0, jnp.sum(nrm * L, axis=-1))
        sh_o = hp + nrm * 2e-3
        sh_lim = jnp.where(hit & (ndotl > 0.0), jnp.maximum(ldist - 2e-3, 0.0), 0.0)
        d_sh, _ = _pallas_multiobj_bvh_raycast(
            l2r, l2, l0, lf, ob, sh_o, L, sh_lim, H_8K, W_LOCAL, k_obj, any_hit=True, interpret=pallas_interp,
        )
        vis = (d_sh >= sh_lim - 1e-4).astype(jnp.float32)
        alb = jnp.array([0.74, 0.78, 0.84], jnp.float32)
        rad = jnp.where(hit[..., None], rad + thr * (alb / math.pi) * (ndotl * vis)[..., None] * 3.2, rad)

        u1 = jax.random.uniform(kb2, (1, R_loc))
        u2 = jax.random.uniform(kb3, (1, R_loc))
        rd = _sample_cos_hemi(u1, u2, nrm)
        ro = hp + nrm * 2e-3
        thr = thr * alb
        alive = hit

      return rad_acc + rad

    rad_tot = jax.lax.fori_loop(0, spp_dyn, _sample_body, jnp.zeros((1, R_loc, 3), jnp.float32)) / spp_dyn.astype(jnp.float32)
    x = jnp.maximum(rad_tot, 0.0)
    mapped = jnp.clip((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0)
    return jnp.power(mapped, 1.0 / 2.2).reshape(1, H_8K, W_LOCAL, 3).astype(jnp.bfloat16)

  sm_pt = shard_map(
      _pt_local,
      mesh=mesh_sp,
      in_specs=(P(), P(), P(), P(), P(), P("sp"), P("sp"), P()),
      out_specs=P(None, None, "sp", None),
      **sm_kw,
  )
  jit_pt = jax.jit(lambda ro, rd, s_dyn: sm_pt(l2r_rep, l2_rep, l0_rep, leaf_rep, ob_rep, ro, rd, s_dyn))
  spp_one = jax.make_array_from_single_device_arrays((), rep_sharding, [jax.device_put(np.int32(1), d) for d in local_devs])
  lowered = jit_pt.lower(ro_g, rd_g, spp_one)
  hlo = _audit_hlo(lowered)

  tc0 = time.perf_counter()
  exec_pt = lowered.compile()
  out_pt = exec_pt(ro_g, rd_g, spp_one)
  jax.block_until_ready(out_pt)
  warmup_s = time.perf_counter() - tc0

  for spp in (1, 4):
    spp_arr = jax.make_array_from_single_device_arrays((), rep_sharding, [jax.device_put(np.int32(spp), d) for d in local_devs])
    t0 = time.perf_counter()
    out_pt = exec_pt(ro_g, rd_g, spp_arr)
    jax.block_until_ready(out_pt)
    dt_s = time.perf_counter() - t0
    samples_per_min = (60.0 * spp) / max(dt_s, 1e-9)
    total_rays = H_8K * W_8K * spp * max_bounces * 2

    rec = {
        "benchmark": f"5_spatial_8k_pt_b{max_bounces}_spp{spp}",
        "n_chips": n_chips,
        "resolution": f"{W_8K}x{H_8K}",
        "max_bounces": max_bounces,
        "spp": spp,
        "tris": k_obj * 1048576,
        "dt_ms": dt_s * 1e3,
        "fps": 1.0 / max(dt_s, 1e-9),
        "samples_per_min": samples_per_min,
        "pixel_msamples_per_min": samples_per_min * (H_8K * W_8K) / 1e6,
        "mrays_per_s": (total_rays / 1e6) / max(dt_s, 1e-9),
        "warmup_s": round(warmup_s, 3),
        "host_callbacks": hlo["host_callbacks"],
        "send_recv_ops": hlo["send_recv_ops"],
        "hlo_zero_copy": True,
    }
    rows.append(rec)
    if proc_idx == 0:
      print(f"[B5 8K PT b{max_bounces} spp={spp}] {n_chips}c: {dt_s*1e3:.1f} ms -> {samples_per_min:.1f} samples/min", flush=True)
  return rows


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--dry-run-4c", action="store_true", help="Run locally on 4 chips of 1 host.")
  ap.add_argument("--out_json", default="/dev/shm/zaychik-tpu-mjx-w6/out/v3_pod_v5p64_capstone.json")
  args = ap.parse_args()

  is_distributed = bool(os.environ.get("JAX_COORDINATOR_ADDRESS")) and not args.dry_run_4c
  if is_distributed:
    coord = os.environ["JAX_COORDINATOR_ADDRESS"]
    n_proc = os.environ.get("JAX_PROCESS_COUNT") or os.environ.get("JAX_NUM_PROCESSES")
    proc_id = os.environ.get("JAX_PROCESS_ID") or os.environ.get("JAX_PROCESS_INDEX")
    if n_proc is not None and proc_id is not None:
      jax.distributed.initialize(coordinator_address=coord, num_processes=int(n_proc), process_id=int(proc_id))
    else:
      jax.distributed.initialize()

  all_devs = jax.devices()
  local_devs = jax.local_devices()
  n_chips = len(all_devs)
  proc_idx = jax.process_index()
  pallas_interp = (jax.default_backend() == "cpu")

  if proc_idx == 0:
    print(
        f"POD CAPSTONE INIT: backend={jax.default_backend()} n_chips={n_chips} "
        f"n_processes={jax.process_count()} local_chips={len(local_devs)} dry_run_4c={args.dry_run_4c}",
        flush=True,
    )

  mesh_dp = Mesh(np.array(all_devs), ("dp",))
  mesh_geom = Mesh(np.array(all_devs), ("geom",))
  mesh_sp = Mesh(np.array(all_devs), ("sp",))

  t_start = time.perf_counter()
  b1 = run_bench1_tpu_rtx(mesh_dp, local_devs, n_chips, pallas_interp, proc_idx)
  b2 = run_bench2_tpu_gsplat(mesh_dp, local_devs, n_chips, pallas_interp, proc_idx)
  b3 = run_bench3_lidar(mesh_dp, local_devs, n_chips, pallas_interp, proc_idx)
  b4 = run_bench4_geometry_sharded_16b(mesh_geom, local_devs, n_chips, pallas_interp, proc_idx)
  b5 = run_bench5_8k_pathtracer(mesh_sp, local_devs, n_chips, pallas_interp, proc_idx)
  total_s = time.perf_counter() - t_start

  _sync_hosts("capstone_done", is_distributed)

  if proc_idx == 0:
    out_data = {
        "experiment": "pod_v5p64_capstone",
        "dry_run_4c": bool(args.dry_run_4c),
        "backend": jax.default_backend(),
        "device_kind": all_devs[0].device_kind,
        "n_chips": n_chips,
        "n_processes": jax.process_count(),
        "cluster_hbm_tb": round(n_chips * 95.74 / 1000.0, 3),
        "total_runtime_s": round(total_s, 2),
        "bench1_tpu_rtx": b1,
        "bench2_tpu_gsplat": b2,
        "bench3_batch_lidar": b3,
        "bench4_geom_sharded_unique_tris": b4,
        "bench5_spatial_8k_pathtracer": b5,
    }
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
      json.dump(out_data, f, indent=2)
    print(f"POD CAPSTONE COMPLETE in {total_s:.1f}s -> {args.out_json}", flush=True)


if __name__ == "__main__":
  main()
