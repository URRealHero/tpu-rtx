"""Native 4K UHD (3840x2160, 8,294,400 px) TPU v5p Rendering Benchmark & Visual Export.

Benchmarks on live TPU v5p (interpret=False):
1. Stage 1 (`pallas_v2`) 3840x2160:
   - 1-chip: B=1, 4, 8
   - 4-chip `shard_map` (DP): B=4, 8, 16, 32
   - 4-chip single-frame spatial split (`W=3840 -> 4x960`, 4,050 blocks of 512 per chip): B=1
2. Stage 1 + Stage 2 (`M2_spatial_128`) 3840x2160 in BF16 and FP32:
   - 1-chip: B=1 (strictly 1 frame/chip HBM guard)
   - 4-chip `shard_map`: B=4 (1 frame/chip)
3. Exports native 4K Stage 1+2 PNG (`v3_w6_panda_4k_stage12.png`) and 100%-crop NPZ/PNG data.
"""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
import os
import pickle
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import flax.linen as nn
import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np

from tpu_mjx_render.neural_shader import (
    apply_neural_shader,
    build_gbuffer_features,
    make_neural_shader_model,
)
from tpu_mjx_render.pallas_v2 import (
    BLOCK_B,
    BLOCK_R,
    _compute_geom_bound_sq,
    _extract_light_batch,
    _ray_one_geom_v2,
    _shade_batch_soa,
    build_local_camera_rays,
)
from tpu_mjx_render.primitives import MJ_MAXVAL
from tpu_mjx_render.renderer import RenderSceneSpec, render_batch
from tpu_mjx_render.roofline import inspect_hlo_ops
from tpu_mjx_render.scenes import get_scene_xml

W_4K = 3840
H_4K = 2160
R_4K = W_4K * H_4K  # 8,294,400 = 16,200 * 512


class _ResBlock(nn.Module):
  features: int
  compute_dtype: Any = jnp.bfloat16

  @nn.compact
  def __call__(self, x: jax.Array) -> jax.Array:
    dt = self.compute_dtype
    y = nn.Conv(self.features, (3, 3), padding="SAME", dtype=dt, param_dtype=jnp.float32)(x)
    y = nn.relu(y)
    y = nn.Conv(self.features, (3, 3), padding="SAME", dtype=dt, param_dtype=jnp.float32)(y)
    return nn.relu(x + y)


class SpatialRelighterDtype(nn.Module):
  """M2_spatial_128 with configurable compute_dtype (BF16 or FP32) sharing exact same params."""

  base_width: int = 64
  compute_dtype: Any = jnp.bfloat16

  @nn.compact
  def __call__(self, x: jax.Array, train: bool = False) -> jax.Array:
    del train
    dt = self.compute_dtype
    w1 = self.base_width
    w2 = self.base_width * 2

    h1 = nn.relu(nn.Conv(w1, (3, 3), padding="SAME", dtype=dt, param_dtype=jnp.float32)(x.astype(dt)))
    h1 = _ResBlock(w1, dt)(h1)
    h1 = _ResBlock(w1, dt)(h1)

    h2 = nn.relu(nn.Conv(w2, (3, 3), strides=(2, 2), padding="SAME", dtype=dt, param_dtype=jnp.float32)(h1))
    h2 = _ResBlock(w2, dt)(h2)
    h2 = _ResBlock(w2, dt)(h2)

    B, H, W, _ = x.shape
    h2_up = jax.image.resize(h2.astype(jnp.float32), (B, H, W, w2), method="bilinear").astype(dt)
    cat = jnp.concatenate([h1, h2_up], axis=-1)
    h_out = nn.relu(nn.Conv(w1, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(cat))
    h_out = _ResBlock(w1, dt)(h_out)
    out_in = jnp.concatenate([h_out, x.astype(dt)], axis=-1)
    out_mid = nn.relu(nn.Conv(w1, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(out_in))
    out_mid = nn.relu(nn.Conv(w1, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(out_mid))
    delta = nn.Conv(3, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(out_mid).astype(jnp.float32)
    return delta


def _audit_hlo(lowered: Any) -> Dict[str, Any]:
  txt = lowered.as_text()
  info = inspect_hlo_ops(txt)
  info["send_recv_ops"] = len(re.findall(r"\bsend\(|\brecv\(", txt, re.I))
  assert info["host_callbacks"] == 0, f"HLO contains host_callback: {info}"
  assert info["send_recv_ops"] == 0, f"HLO contains send/recv: {info}"
  info["zero_copy_verified"] = True
  return info


def _load_m2_ckpt(ckpt_path: str):
  with open(ckpt_path, "rb") as f:
    ck = pickle.load(f)
  params = jax.tree_util.tree_map(jnp.asarray, ck["params"])
  cal_scale = float(ck["cal_scale"])
  return params, cal_scale, ck["arch"]


def _make_batch(m: mujoco.MjModel, mx: Any, batch_size: int):
  del mx
  d_host = mujoco.MjData(m)
  if m.nkey > 0:
    mujoco.mj_resetDataKeyframe(m, d_host, 0)
  mujoco.mj_forward(m, d_host)
  dx0 = mjx.put_data(m, d_host, impl="jax")
  dx_batch = jax.tree_util.tree_map(
      lambda x: jnp.broadcast_to(x[None, ...], (batch_size, *x.shape)).copy(), dx0
  )
  ctrl = jnp.zeros((batch_size, m.nu), dtype=jnp.float32)
  return dx_batch, ctrl


def _pallas_raycast_strip(
    spec_full: RenderSceneSpec,
    mjx_data_batch: Any,
    lray_strip: jax.Array,  # (8, R_strip)
    ld_strip: jax.Array,    # (R_strip, 3)
    lo_strip: jax.Array,    # (R_strip, 3)
    nz_strip: jax.Array,    # (R_strip,)
    strip_h: int,
    strip_w: int,
    interpret: bool = False,
):
  """Render a (strip_h x strip_w) spatial strip of a 4K frame in Pallas VMEM."""
  cam_pos = mjx_data_batch.cam_xpos[:, int(spec_full.cam_id), :]
  cam_mat = mjx_data_batch.cam_xmat[:, int(spec_full.cam_id), :, :]
  geom_xpos = mjx_data_batch.geom_xpos
  geom_xmat = mjx_data_batch.geom_xmat
  light_xpos, light_xdir = _extract_light_batch(spec_full, mjx_data_batch)
  geom_size = jnp.asarray(spec_full.geom_size, dtype=jnp.float32)
  geom_rgba = jnp.asarray(spec_full.geom_rgba, dtype=jnp.float32)

  B = cam_pos.shape[0]
  R_strip = strip_h * strip_w
  assert R_strip % BLOCK_R == 0

  gids = np.asarray(spec_full.enabled_geom_ids, dtype=np.int32)
  G = len(gids)
  B_pad = ((B + BLOCK_B - 1) // BLOCK_B) * BLOCK_B
  G_pad = ((G + 127) // 128) * 128

  cam_pack = jnp.concatenate([cam_pos, cam_mat.reshape(B, 9)], axis=-1)
  cam_128 = jnp.pad(cam_pack, ((0, B_pad - B), (0, 128 - 12)))

  gpos = geom_xpos[:, gids, :]
  gmat = geom_xmat[:, gids, :, :].reshape(B, G, 9)
  gsz_all = jnp.broadcast_to(geom_size[gids][None, :, :], (B, G, 3))
  rb_sq = jnp.broadcast_to(jnp.asarray(_compute_geom_bound_sq(spec_full, gids))[None, :, None], (B, G, 1))
  g_16 = jnp.concatenate([gpos, gmat, gsz_all, rb_sq], axis=-1)
  g_16 = jnp.pad(g_16, ((0, B_pad - B), (0, G_pad - G), (0, 0)))

  n_b_blk = B_pad // BLOCK_B
  n_r_blk = R_strip // BLOCK_R
  geom_fields = (
      g_16.reshape(n_b_blk, BLOCK_B, G_pad, 16)
      .transpose(0, 3, 1, 2)
      .reshape(n_b_blk * 16 * BLOCK_B, G_pad)
  )

  shadow_l_idx = -1
  for li in range(int(spec_full.nlight)):
    if spec_full.use_shadows and bool(spec_full.light_active[li]) and bool(spec_full.light_castshadow[li]):
      shadow_l_idx = li
      break

  light_128 = jnp.zeros((B_pad, 128), dtype=jnp.float32)
  if shadow_l_idx >= 0:
    is_dir = 1.0 if int(spec_full.light_type[shadow_l_idx]) == 1 else 0.0
    lp = light_xpos[:, shadow_l_idx, :]
    ld_vec = light_xdir[:, shadow_l_idx, :]
    ld_vec = ld_vec / jnp.maximum(jnp.linalg.norm(ld_vec, axis=-1, keepdims=True), 1e-12)
    l_row = jnp.concatenate([jnp.full((B, 1), is_dir, jnp.float32), lp, ld_vec], axis=-1)
    light_128 = jnp.pad(l_row, ((0, B_pad - B), (0, 128 - 7)))

  g_types_static = tuple(int(spec_full.geom_type[g]) for g in gids)
  gids_static = tuple(int(g) for g in gids)
  plane_fx = tuple(float(spec_full.geom_size[g, 0]) > 0.0 for g in gids)
  plane_fy = tuple(float(spec_full.geom_size[g, 1]) > 0.0 for g in gids)
  znear_f = float(spec_full.znear)
  do_in_vmem_shadow = (shadow_l_idx >= 0) and (G <= 36)

  def _strip_kernel(
      cam_ref, lray_ref, gf_ref, light_ref,
      out_d_ref, out_nx_ref, out_ny_ref, out_nz_ref, out_gid_ref, out_sh_ref,
  ):
    cam = cam_ref[...]
    lr = lray_ref[...]
    gf = gf_ref[...]
    linfo = light_ref[...]

    ldx, ldy, ldz = lr[0:1, :], lr[1:2, :], lr[2:3, :]
    lox, loy, loz = lr[3:4, :], lr[4:5, :], lr[5:6, :]
    inv_cos = znear_f / lr[6:7, :]
    max_d = jnp.broadcast_to(lr[7:8, :], (BLOCK_B, BLOCK_R))

    cpx, cpy, cpz = cam[:, 0:1], cam[:, 1:2], cam[:, 2:3]
    c00, c01, c02 = cam[:, 3:4], cam[:, 4:5], cam[:, 5:6]
    c10, c11, c12 = cam[:, 6:7], cam[:, 7:8], cam[:, 8:9]
    c20, c21, c22 = cam[:, 9:10], cam[:, 10:11], cam[:, 11:12]

    rdx_raw = c00 * ldx + c01 * ldy + c02 * ldz
    rdy_raw = c10 * ldx + c11 * ldy + c12 * ldz
    rdz_raw = c20 * ldx + c21 * ldy + c22 * ldz
    inv_rn = 1.0 / jnp.maximum(jnp.sqrt(rdx_raw * rdx_raw + rdy_raw * rdy_raw + rdz_raw * rdz_raw), 1e-12)
    rdx, rdy, rdz = rdx_raw * inv_rn, rdy_raw * inv_rn, rdz_raw * inv_rn

    rox = cpx + (c00 * lox + c01 * loy + c02 * loz) + inv_cos * rdx
    roy = cpy + (c10 * lox + c11 * loy + c12 * loz) + inv_cos * rdy
    roz = cpz + (c20 * lox + c21 * loy + c22 * loz) + inv_cos * rdz

    best_d = max_d
    best_nx = jnp.zeros((BLOCK_B, BLOCK_R), dtype=jnp.float32)
    best_ny = jnp.zeros((BLOCK_B, BLOCK_R), dtype=jnp.float32)
    best_nz = jnp.zeros((BLOCK_B, BLOCK_R), dtype=jnp.float32)
    best_gid = jnp.full((BLOCK_B, BLOCK_R), -1, dtype=jnp.int32)

    for k, (gt, gid_val, pfx, pfy) in enumerate(zip(g_types_static, gids_static, plane_fx, plane_fy)):
      px = gf[0 * 8 : 1 * 8, k : k + 1]
      py = gf[1 * 8 : 2 * 8, k : k + 1]
      pz = gf[2 * 8 : 3 * 8, k : k + 1]
      m00, m01, m02 = gf[3 * 8 : 4 * 8, k : k + 1], gf[4 * 8 : 5 * 8, k : k + 1], gf[5 * 8 : 6 * 8, k : k + 1]
      m10, m11, m12 = gf[6 * 8 : 7 * 8, k : k + 1], gf[7 * 8 : 8 * 8, k : k + 1], gf[8 * 8 : 9 * 8, k : k + 1]
      m20, m21, m22 = gf[9 * 8 : 10 * 8, k : k + 1], gf[10 * 8 : 11 * 8, k : k + 1], gf[11 * 8 : 12 * 8, k : k + 1]
      sx, sy, sz = gf[12 * 8 : 13 * 8, k : k + 1], gf[13 * 8 : 14 * 8, k : k + 1], gf[14 * 8 : 15 * 8, k : k + 1]
      rb_sq = gf[15 * 8 : 16 * 8, k : k + 1]

      cd, cnx, cny, cnz = _ray_one_geom_v2(
          gt, px, py, pz, m00, m01, m02, m10, m11, m12, m20, m21, m22,
          sx, sy, sz, rb_sq, rox, roy, roz, rdx, rdy, rdz,
          plane_finite_x=pfx, plane_finite_y=pfy, compute_normal=True,
      )
      imp = cd < best_d
      best_d = jnp.where(imp, cd, best_d)
      best_gid = jnp.where(imp, jnp.int32(gid_val), best_gid)
      best_nx = jnp.where(imp, cnx, best_nx)
      best_ny = jnp.where(imp, cny, best_ny)
      best_nz = jnp.where(imp, cnz, best_nz)

    hit = best_gid >= 0
    best_d_out = jnp.where(hit, best_d, -1.0)
    out_nx_v = jnp.where(hit, best_nx, 0.0)
    out_ny_v = jnp.where(hit, best_ny, 0.0)
    out_nz_v = jnp.where(hit, best_nz, 0.0)

    sh_occ = jnp.zeros((BLOCK_B, BLOCK_R), dtype=jnp.float32)
    if do_in_vmem_shadow:
      d_safe = jnp.where(hit, best_d, 0.0)
      hx = rox + d_safe * rdx + out_nx_v * 1.0e-4
      hy = roy + d_safe * rdy + out_ny_v * 1.0e-4
      hz = roz + d_safe * rdz + out_nz_v * 1.0e-4

      is_dir = linfo[:, 0:1] > 0.5
      dlx = linfo[:, 1:2] - hx
      dly = linfo[:, 2:3] - hy
      dlz = linfo[:, 3:4] - hz
      d_pt = jnp.sqrt(jnp.maximum(dlx * dlx + dly * dly + dlz * dlz, 1e-12))
      lx_v = jnp.where(is_dir, -linfo[:, 4:5], dlx / d_pt)
      ly_v = jnp.where(is_dir, -linfo[:, 5:6], dly / d_pt)
      lz_v = jnp.where(is_dir, -linfo[:, 6:7], dlz / d_pt)
      ndotl = out_nx_v * lx_v + out_ny_v * ly_v + out_nz_v * lz_v
      dist_lim = jnp.where(is_dir, 1.0e8, jnp.maximum(d_pt - 1.0e-3, 0.0))
      sh_max = jnp.where(hit & (ndotl > 0.0), dist_lim, 0.0)

      occ_bool = jnp.zeros((BLOCK_B, BLOCK_R), dtype=jnp.bool_)
      for k, (gt, pfx, pfy) in enumerate(zip(g_types_static, plane_fx, plane_fy)):
        px = gf[0 * 8 : 1 * 8, k : k + 1]
        py = gf[1 * 8 : 2 * 8, k : k + 1]
        pz = gf[2 * 8 : 3 * 8, k : k + 1]
        m00, m01, m02 = gf[3 * 8 : 4 * 8, k : k + 1], gf[4 * 8 : 5 * 8, k : k + 1], gf[5 * 8 : 6 * 8, k : k + 1]
        m10, m11, m12 = gf[6 * 8 : 7 * 8, k : k + 1], gf[7 * 8 : 8 * 8, k : k + 1], gf[8 * 8 : 9 * 8, k : k + 1]
        m20, m21, m22 = gf[9 * 8 : 10 * 8, k : k + 1], gf[10 * 8 : 11 * 8, k : k + 1], gf[11 * 8 : 12 * 8, k : k + 1]
        sx, sy, sz = gf[12 * 8 : 13 * 8, k : k + 1], gf[13 * 8 : 14 * 8, k : k + 1], gf[14 * 8 : 15 * 8, k : k + 1]
        rb_sq = gf[15 * 8 : 16 * 8, k : k + 1]
        scd, _, _, _ = _ray_one_geom_v2(
            gt, px, py, pz, m00, m01, m02, m10, m11, m12, m20, m21, m22,
            sx, sy, sz, rb_sq, hx, hy, hz, lx_v, ly_v, lz_v,
            plane_finite_x=pfx, plane_finite_y=pfy, compute_normal=False,
        )
        occ_bool = occ_bool | (scd < sh_max)
      sh_occ = occ_bool.astype(jnp.float32)

    out_d_ref[...] = best_d_out
    out_nx_ref[...] = out_nx_v
    out_ny_ref[...] = out_ny_v
    out_nz_ref[...] = out_nz_v
    out_gid_ref[...] = best_gid
    out_sh_ref[...] = sh_occ

  d_out, nx_out, ny_out, nz_out, gid_out, sh_out = pl.pallas_call(
      _strip_kernel,
      out_shape=[
          jax.ShapeDtypeStruct((B_pad, R_strip), jnp.float32),
          jax.ShapeDtypeStruct((B_pad, R_strip), jnp.float32),
          jax.ShapeDtypeStruct((B_pad, R_strip), jnp.float32),
          jax.ShapeDtypeStruct((B_pad, R_strip), jnp.float32),
          jax.ShapeDtypeStruct((B_pad, R_strip), jnp.int32),
          jax.ShapeDtypeStruct((B_pad, R_strip), jnp.float32),
      ],
      grid=(n_b_blk, n_r_blk),
      in_specs=[
          pl.BlockSpec((BLOCK_B, 128), lambda bi, ri: (bi, 0)),
          pl.BlockSpec((8, BLOCK_R), lambda bi, ri: (0, ri)),
          pl.BlockSpec((16 * BLOCK_B, G_pad), lambda bi, ri: (bi, 0)),
          pl.BlockSpec((BLOCK_B, 128), lambda bi, ri: (bi, 0)),
      ],
      out_specs=[
          pl.BlockSpec((BLOCK_B, BLOCK_R), lambda bi, ri: (bi, ri)),
          pl.BlockSpec((BLOCK_B, BLOCK_R), lambda bi, ri: (bi, ri)),
          pl.BlockSpec((BLOCK_B, BLOCK_R), lambda bi, ri: (bi, ri)),
          pl.BlockSpec((BLOCK_B, BLOCK_R), lambda bi, ri: (bi, ri)),
          pl.BlockSpec((BLOCK_B, BLOCK_R), lambda bi, ri: (bi, ri)),
          pl.BlockSpec((BLOCK_B, BLOCK_R), lambda bi, ri: (bi, ri)),
      ],
      compiler_params=None if interpret else pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
      interpret=interpret,
  )(cam_128, lray_strip, geom_fields, light_128)

  dist = d_out[:B, :R_strip]
  normal = jnp.stack([nx_out[:B, :R_strip], ny_out[:B, :R_strip], nz_out[:B, :R_strip]], axis=-1)
  geom_id = gid_out[:B, :R_strip]
  fused_occ = (sh_out[:B, :R_strip] > 0.5) if do_in_vmem_shadow else None

  ray_dir = jnp.einsum("bij,rj->bri", cam_mat, ld_strip, precision=jax.lax.Precision.HIGHEST)
  ray_dir = ray_dir / jnp.maximum(jnp.linalg.norm(ray_dir, axis=-1, keepdims=True), 1e-12)
  ray_off = jnp.einsum("bij,rj->bri", cam_mat, lo_strip, precision=jax.lax.Precision.HIGHEST)
  inv_cos = (float(spec_full.znear) / nz_strip)[None, :, None]
  ray_orig = cam_pos[:, None, :] + ray_off + inv_cos * ray_dir

  spec_strip = dataclasses.replace(spec_full, height=strip_h, width=strip_w)
  return _shade_batch_soa(
      spec_strip, geom_xpos, geom_xmat, geom_size, geom_rgba,
      light_xpos, light_xdir, ray_orig, ray_dir, nz_strip,
      dist, normal, geom_id, fused_shadow0_occluded=fused_occ,
  )


def run_4k_suite(
    n_chips: int,
    neural_ckpt: str,
    out_json: str,
    export_dir: Optional[str] = None,
    phase: str = "all",
    s1_batches_override: Optional[List[int]] = None,
) -> Dict[str, Any]:
  pallas_interp = (jax.default_backend() == "cpu")
  devs = jax.devices()[:n_chips]
  assert len(devs) == n_chips, f"Expected {n_chips} chips, got {len(devs)}"

  m = mujoco.MjModel.from_xml_string(get_scene_xml("panda_pick_cube"))
  mx = mjx.put_model(m, impl="jax")
  spec_4k = RenderSceneSpec.from_mjmodel(m, cam_id=0, height=H_4K, width=W_4K, use_shadows=True)
  params_m2, cal_scale, _ = _load_m2_ckpt(neural_ckpt)

  results: Dict[str, Any] = {
      "resolution": f"{W_4K}x{H_4K}",
      "pixels_per_frame": R_4K,
      "pallas_blocks_per_frame": R_4K // BLOCK_R,
      "n_chips": n_chips,
      "device_kind": devs[0].device_kind,
      "stage1_runs": [],
      "stage12_runs": [],
      "spatial_split_4c": None,
  }
  os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
  def _save():
    with open(out_json, "w") as f:
      json.dump(results, f, indent=2)

  sm_kw = {"check_rep": False} if "check_rep" in inspect.signature(shard_map).parameters else {"check_vma": False}
  if n_chips > 1:
    mesh_dp = jax.sharding.Mesh(np.array(devs), ("dp",))
    P = jax.sharding.PartitionSpec
    sharding_dp = jax.sharding.NamedSharding(mesh_dp, P("dp"))
    rep_dp = jax.sharding.NamedSharding(mesh_dp, P())

  # --- 1. Stage 1 Native 4K ---
  if phase in ("all", "s1"):
    s1_batches = s1_batches_override if s1_batches_override else ([1, 4, 8] if n_chips == 1 else [4, 8, 16, 32])
    for B in s1_batches:
      dx_b, ctrl = _make_batch(m, mx, B)
      if n_chips > 1:
        dx_b = jax.device_put(dx_b, sharding_dp)
        ctrl = jax.device_put(ctrl, sharding_dp)
        mx_d = jax.device_put(mx, rep_dp)

        _phys_sm = shard_map(
            lambda d, u, mxl: jax.vmap(lambda x, c: mjx.step(mxl, x.replace(ctrl=c)))(d, u),
            mesh=mesh_dp, in_specs=(P("dp"), P("dp"), P()), out_specs=P("dp"), **sm_kw,
        )
        _e2e_sm = shard_map(
            lambda d, u, mxl: (
                (dn := jax.vmap(lambda x, c: mjx.step(mxl, x.replace(ctrl=c)))(d, u)),
                (gb := render_batch(spec_4k, dn, mode="pallas_v2", pallas_interpret=pallas_interp)).rgb,
                gb.depth,
            ),
            mesh=mesh_dp, in_specs=(P("dp"), P("dp"), P()), out_specs=(P("dp"), P("dp"), P("dp")), **sm_kw,
        )
        phys_fn = jax.jit(lambda d, u: _phys_sm(d, u, mx_d))
        e2e_fn = jax.jit(lambda d, u: _e2e_sm(d, u, mx_d))
      else:
        dx_b = jax.device_put(dx_b, devs[0])
        ctrl = jax.device_put(ctrl, devs[0])
        phys_fn = jax.jit(lambda d, u: jax.vmap(lambda x, c: mjx.step(mx, x.replace(ctrl=c)))(d, u))

        @jax.jit
        def e2e_fn(d, u):
          dn = jax.vmap(lambda x, c: mjx.step(mx, x.replace(ctrl=c)))(d, u)
          gb = render_batch(spec_4k, dn, mode="pallas_v2", pallas_interpret=pallas_interp)
          return dn, gb.rgb, gb.depth

      try:
        hlo = _audit_hlo(e2e_fn.lower(dx_b, ctrl))
        d_p = phys_fn(dx_b, ctrl)
        jax.block_until_ready(d_p.qpos)
        t0 = time.perf_counter()
        for _ in range(5):
          d_p = phys_fn(d_p, ctrl)
        jax.block_until_ready(d_p.qpos)
        dt_phys = (time.perf_counter() - t0) / 5

        tc0 = time.perf_counter()
        dc = dx_b
        for _ in range(2):
          dc, rgb, dep = e2e_fn(dc, ctrl)
        jax.block_until_ready((rgb, dep))
        warmup_s = time.perf_counter() - tc0

        n_st = 4 if B <= 8 else 3
        t1 = time.perf_counter()
        for _ in range(n_st):
          dc, rgb, dep = e2e_fn(dc, ctrl)
        jax.block_until_ready((rgb, dep))
        dt_e2e = (time.perf_counter() - t1) / n_st
        dt_rend = max(dt_e2e - dt_phys, 0.0)
        fps = float(B / max(dt_e2e, 1e-9))
        rec = {
            "status": "ok",
            "stage": "stage1_pallas_v2",
            "n_chips": n_chips,
            "batch_size": B,
            "resolution": f"{W_4K}x{H_4K}",
            "dt_physics_ms": dt_phys * 1e3,
            "dt_render_ms": dt_rend * 1e3,
            "dt_e2e_ms": dt_e2e * 1e3,
            "e2e_fps": fps,
            "megapixels_per_s": fps * R_4K / 1e6,
            "warmup_s": round(warmup_s, 3),
            "host_callbacks": hlo["host_callbacks"],
            "send_recv_ops": hlo["send_recv_ops"],
            "hlo_zero_copy": True,
        }
        print(f"4K S1 {n_chips}c B={B}: e2e={rec['dt_e2e_ms']:.2f}ms fps={fps:.2f} ({rec['megapixels_per_s']:.1f} MPix/s)", flush=True)
        if export_dir and n_chips == 1 and B == 1:
          os.makedirs(export_dir, exist_ok=True)
          np.save(os.path.join(export_dir, "frame_4k_stage1_rgb.npy"), np.asarray(rgb[0], np.float32))
      except Exception as exc:
        rec = {
            "status": "error",
            "stage": "stage1_pallas_v2",
            "n_chips": n_chips,
            "batch_size": B,
            "resolution": f"{W_4K}x{H_4K}",
            "error": f"{type(exc).__name__}: {str(exc)[:300]}",
        }
        print(f"4K S1 {n_chips}c B={B} ERROR: {rec['error']}", flush=True)
      results["stage1_runs"].append(rec)
      _save()

  # --- 2. 4-Chip Single-Frame Spatial Split (W=3840 -> 4x960, 4,050 blocks/chip) ---
  if n_chips == 4 and phase in ("all", "spatial"):
    mesh_sp = jax.sharding.Mesh(np.array(devs), ("sp",))
    P = jax.sharding.PartitionSpec
    sh_sp = jax.sharding.NamedSharding(mesh_sp, P("sp", None, None))
    sh_sp_r = jax.sharding.NamedSharding(mesh_sp, P("sp", None))
    rep_sp = jax.sharding.NamedSharding(mesh_sp, P())

    ld_np, lo_np, nz_np = build_local_camera_rays(spec_4k)
    md_np = ((float(spec_4k.zfar) - float(spec_4k.znear)) / nz_np) if spec_4k.zfar > 0.0 else np.full((R_4K,), MJ_MAXVAL, dtype=np.float32)

    # Reshape (H_4K, W_4K, ...) -> 4 vertical strips of (H_4K, 960, ...) so W=3840 -> 4x960
    W_STRIP = W_4K // 4  # 960 = 120 * 8
    R_STRIP = H_4K * W_STRIP  # 2,073,600 = 4,050 * 512

    def _split_w(arr):
      a4 = arr.reshape(H_4K, 4, W_STRIP, *arr.shape[1:]).transpose(1, 0, 2, *range(3, arr.ndim + 2))
      return a4.reshape(4, R_STRIP, *arr.shape[1:])

    ld_s = jax.device_put(jnp.asarray(_split_w(ld_np), jnp.float32), sh_sp)
    lo_s = jax.device_put(jnp.asarray(_split_w(lo_np), jnp.float32), sh_sp)
    nz_s = jax.device_put(jnp.asarray(_split_w(nz_np), jnp.float32), sh_sp_r)
    md_s = _split_w(md_np)

    lray_s_np = np.zeros((4, 8, R_STRIP), dtype=np.float32)
    lray_s_np[:, 0:3, :] = _split_w(ld_np).transpose(0, 2, 1)
    lray_s_np[:, 3:6, :] = _split_w(lo_np).transpose(0, 2, 1)
    lray_s_np[:, 6, :] = _split_w(nz_np)
    lray_s_np[:, 7, :] = md_s
    lray_s = jax.device_put(jnp.asarray(lray_s_np, jnp.float32), sh_sp)

    dx1, ctrl1 = _make_batch(m, mx, 1)
    dx1_rep = jax.device_put(dx1, rep_sp)
    ctrl1_rep = jax.device_put(ctrl1, rep_sp)
    mx_rep = jax.device_put(mx, rep_sp)

    def _sp_local(d_1, u_1, mx_l, lr_s, lds, los, nzs):
      dn = jax.vmap(lambda x, c: mjx.step(mx_l, x.replace(ctrl=c)))(d_1, u_1)
      gb = _pallas_raycast_strip(
          spec_4k, dn, lr_s[0], lds[0], los[0], nzs[0], H_4K, W_STRIP, interpret=pallas_interp,
      )
      return gb.rgb, gb.depth

    _sp_sm = shard_map(
        _sp_local,
        mesh=mesh_sp,
        in_specs=(P(), P(), P(), P("sp", None, None), P("sp", None, None), P("sp", None, None), P("sp", None)),
        out_specs=(P(None, None, "sp", None), P(None, None, "sp")),
        **sm_kw,
    )

    @jax.jit
    def sp_e2e(d_1, u_1):
      return _sp_sm(d_1, u_1, mx_rep, lray_s, ld_s, lo_s, nz_s)

    hlo_sp = _audit_hlo(sp_e2e.lower(dx1_rep, ctrl1_rep))
    tc0 = time.perf_counter()
    rgb_sp, dep_sp = sp_e2e(dx1_rep, ctrl1_rep)
    jax.block_until_ready((rgb_sp, dep_sp))
    warmup_sp = time.perf_counter() - tc0

    t1 = time.perf_counter()
    for _ in range(5):
      rgb_sp, dep_sp = sp_e2e(dx1_rep, ctrl1_rep)
    jax.block_until_ready((rgb_sp, dep_sp))
    dt_sp_e2e = (time.perf_counter() - t1) / 5

    # Verify parity against 1-chip B=1 on devs[0] (single-device placement avoids auto-SPMD on Mosaic)
    dx1_d0 = jax.device_put(dx1, devs[0])
    ctrl1_d0 = jax.device_put(ctrl1, devs[0])
    mx_d0 = jax.device_put(mx, devs[0])
    gb_ref_1c = jax.jit(lambda d: render_batch(spec_4k, d, mode="pallas_v2", pallas_interpret=pallas_interp))(
        jax.jit(lambda d, u: jax.vmap(lambda x, c: mjx.step(mx_d0, x.replace(ctrl=c)))(d, u))(dx1_d0, ctrl1_d0)
    )
    jax.block_until_ready(gb_ref_1c.rgb)
    sp_depth_mae = float(np.mean(np.abs(np.asarray(dep_sp) - np.asarray(gb_ref_1c.depth))))
    sp_rgb_max = float(np.max(np.abs(np.asarray(rgb_sp) - np.asarray(gb_ref_1c.rgb))))
    assert sp_depth_mae < 1e-5, sp_depth_mae

    results["spatial_split_4c"] = {
        "strategy": "W=3840 -> 4x960 (4050 blocks of 512 per chip)",
        "n_chips": 4,
        "batch_size": 1,
        "resolution": f"{W_4K}x{H_4K}",
        "dt_e2e_ms": dt_sp_e2e * 1e3,
        "e2e_fps": float(1.0 / max(dt_sp_e2e, 1e-9)),
        "megapixels_per_s": float(R_4K / max(dt_sp_e2e, 1e-9) / 1e6),
        "warmup_s": round(warmup_sp, 3),
        "depth_mae_vs_1c_m": sp_depth_mae,
        "max_rgb_diff_vs_1c": sp_rgb_max,
        "host_callbacks": hlo_sp["host_callbacks"],
        "send_recv_ops": hlo_sp["send_recv_ops"],
        "hlo_zero_copy": True,
    }
    print(
        f"4K SPATIAL SPLIT 4c B=1: e2e={dt_sp_e2e*1e3:.2f}ms fps={1.0/dt_sp_e2e:.2f} "
        f"depth_mae={sp_depth_mae:.2e} rgb_diff={sp_rgb_max:.2e}",
        flush=True,
    )

  # --- 3. Stage 1 + Stage 2 (M2_spatial_128) Native 4K in BF16 and FP32 (STRICTLY 1 frame/chip!) ---
  s2_list = []
  if phase in ("all", "s12_bf16"):
    s2_list.append(("bf16", jnp.bfloat16))
  if phase in ("all", "s12_fp32"):
    s2_list.append(("fp32", jnp.float32))
  B_s2 = 1 if n_chips == 1 else 4
  for dtype_name, dt_val in s2_list:
    model_dt = SpatialRelighterDtype(base_width=64, compute_dtype=dt_val)
    dx_s2, ctrl_s2 = _make_batch(m, mx, B_s2)

    if n_chips > 1:
      dx_s2 = jax.device_put(dx_s2, sharding_dp)
      ctrl_s2 = jax.device_put(ctrl_s2, sharding_dp)
      mx_d = jax.device_put(mx, rep_dp)
      params_d = jax.device_put(params_m2, rep_dp)

      _phys_s2 = shard_map(
          lambda d, u, mxl: jax.vmap(lambda x, c: mjx.step(mxl, x.replace(ctrl=c)))(d, u),
          mesh=mesh_dp, in_specs=(P("dp"), P("dp"), P()), out_specs=P("dp"), **sm_kw,
      )

      def _e2e_s2_local(d, u, mxl, prm):
        dn = jax.vmap(lambda x, c: mjx.step(mxl, x.replace(ctrl=c)))(d, u)
        gb = render_batch(spec_4k, dn, mode="pallas_v2", pallas_interpret=pallas_interp)
        rgb_s2 = apply_neural_shader(model_dt, prm, gb, cal_scale=cal_scale)
        return dn, rgb_s2, gb.depth

      _e2e_s2_sm = shard_map(
          _e2e_s2_local,
          mesh=mesh_dp, in_specs=(P("dp"), P("dp"), P(), P()), out_specs=(P("dp"), P("dp"), P("dp")), **sm_kw,
      )
      phys_s2_fn = jax.jit(lambda d, u: _phys_s2(d, u, mx_d))
      e2e_s2_fn = jax.jit(lambda d, u: _e2e_s2_sm(d, u, mx_d, params_d))
    else:
      dx_s2 = jax.device_put(dx_s2, devs[0])
      ctrl_s2 = jax.device_put(ctrl_s2, devs[0])
      params_d = jax.device_put(params_m2, devs[0])
      phys_s2_fn = jax.jit(lambda d, u: jax.vmap(lambda x, c: mjx.step(mx, x.replace(ctrl=c)))(d, u))

      @jax.jit
      def e2e_s2_fn(d, u):
        dn = jax.vmap(lambda x, c: mjx.step(mx, x.replace(ctrl=c)))(d, u)
        gb = render_batch(spec_4k, dn, mode="pallas_v2", pallas_interpret=pallas_interp)
        rgb_s2 = apply_neural_shader(model_dt, params_d, gb, cal_scale=cal_scale)
        return dn, rgb_s2, gb.depth

    try:
      hlo_s2 = _audit_hlo(e2e_s2_fn.lower(dx_s2, ctrl_s2))
      dp = phys_s2_fn(dx_s2, ctrl_s2)
      jax.block_until_ready(dp.qpos)
      t0 = time.perf_counter()
      for _ in range(5):
        dp = phys_s2_fn(dp, ctrl_s2)
      jax.block_until_ready(dp.qpos)
      dt_phys_s2 = (time.perf_counter() - t0) / 5

      tc0 = time.perf_counter()
      dc = dx_s2
      for _ in range(2):
        dc, rgb_s2, dep_s2 = e2e_s2_fn(dc, ctrl_s2)
      jax.block_until_ready((rgb_s2, dep_s2))
      warmup_s2 = time.perf_counter() - tc0

      t1 = time.perf_counter()
      for _ in range(4):
        dc, rgb_s2, dep_s2 = e2e_s2_fn(dc, ctrl_s2)
      jax.block_until_ready((rgb_s2, dep_s2))
      dt_e2e_s2 = (time.perf_counter() - t1) / 4
      dt_rend_s2 = max(dt_e2e_s2 - dt_phys_s2, 0.0)
      fps_s2 = float(B_s2 / max(dt_e2e_s2, 1e-9))

      rec_s2 = {
          "status": "ok",
          "stage": "stage1_plus_M2_spatial_128",
          "dtype": dtype_name,
          "n_chips": n_chips,
          "batch_size": B_s2,
          "resolution": f"{W_4K}x{H_4K}",
          "dt_physics_ms": dt_phys_s2 * 1e3,
          "dt_render_ms": dt_rend_s2 * 1e3,
          "dt_e2e_ms": dt_e2e_s2 * 1e3,
          "e2e_fps": fps_s2,
          "megapixels_per_s": fps_s2 * R_4K / 1e6,
          "warmup_s": round(warmup_s2, 3),
          "host_callbacks": hlo_s2["host_callbacks"],
          "send_recv_ops": hlo_s2["send_recv_ops"],
          "conv_ops": hlo_s2["conv_ops"],
          "hlo_zero_copy": True,
      }
      print(
          f"4K S1+M2 ({dtype_name}) {n_chips}c B={B_s2}: e2e={rec_s2['dt_e2e_ms']:.2f}ms "
          f"fps={fps_s2:.2f} ({rec_s2['megapixels_per_s']:.1f} MPix/s)",
          flush=True,
      )

      if export_dir and n_chips == 1 and dtype_name == "bf16":
        os.makedirs(export_dir, exist_ok=True)
        np.save(os.path.join(export_dir, "frame_4k_stage12_rgb.npy"), np.asarray(rgb_s2[0], np.float32))
    except Exception as exc:
      rec_s2 = {
          "status": "error",
          "stage": "stage1_plus_M2_spatial_128",
          "dtype": dtype_name,
          "n_chips": n_chips,
          "batch_size": B_s2,
          "resolution": f"{W_4K}x{H_4K}",
          "error": f"{type(exc).__name__}: {str(exc)[:300]}",
      }
      print(f"4K S1+M2 ({dtype_name}) {n_chips}c ERROR: {rec_s2['error']}", flush=True)
    results["stage12_runs"].append(rec_s2)
    _save()

  _save()
  print(f"Saved {out_json}", flush=True)
  return results


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--n_chips", type=int, default=1)
  ap.add_argument("--neural_ckpt", default="/dev/shm/zaychik-tpu-mjx/out/metrics/M2_spatial_128_params.pkl")
  ap.add_argument("--out_json", required=True)
  ap.add_argument("--export_dir", default=None)
  ap.add_argument("--phase", default="all", choices=["all", "s1", "spatial", "s12_bf16", "s12_fp32", "merge"])
  ap.add_argument("--s1_batches", type=int, nargs="*", default=None)
  ap.add_argument("--merge_inputs", nargs="*", default=[])
  args = ap.parse_args()
  if args.phase == "merge":
    merged: Dict[str, Any] = {
        "experiment": "v3_w6_4k_bench",
        "worker": "w6",
        "resolution": f"{W_4K}x{H_4K}",
        "pixels_per_frame": R_4K,
        "pallas_blocks_per_frame": R_4K // BLOCK_R,
        "stage1_runs": [],
        "stage12_runs": [],
        "spatial_split_4c": None,
    }
    for p in args.merge_inputs:
      with open(p, "r") as f:
        d = json.load(f)
      merged["stage1_runs"].extend(d.get("stage1_runs", []))
      merged["stage12_runs"].extend(d.get("stage12_runs", []))
      if d.get("spatial_split_4c") is not None:
        merged["spatial_split_4c"] = d["spatial_split_4c"]
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
      json.dump(merged, f, indent=2)
    print(f"Merged into {args.out_json}", flush=True)
    return
  run_4k_suite(args.n_chips, args.neural_ckpt, args.out_json, args.export_dir, phase=args.phase, s1_batches_override=args.s1_batches)


if __name__ == "__main__":
  main()
