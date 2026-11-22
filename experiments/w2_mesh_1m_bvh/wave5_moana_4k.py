"""Wave 5 (Crown Jewel): Disney Animation Moana Island Scene (OpenUSD v2.1)
at Native 4K (3840x2160) on 4-Chip TPU v5p.

Features:
  - Native OpenUSD ingestion (USD Native Instances + nested PointInstancers +
    BasisCurves ribbons) into 2-Level/3-Level Wide-TLAS-64 + Compact BLAS-64
    buffers (zero flat CPU mesh explosion).
  - Exact affine M_inv TLAS instance transform (supports arbitrary non-unit scale
    and rotation with zero extra pose HBM rows).
  - Zero-gather Pallas MXU bilinear HDRI sampler loading Disney's real
    `textures/islandsun.exr` for equirectangular sky & Fresnel ocean reflections.
  - Beer-Lambert translucent lagoon ocean shader revealing underwater `isCoral`
    reefs, GGX PBR + foliage subsurface translucency + directional sun shadows +
    aerial perspective + AgX filmic tonemapping.
  - Verified interpret=False, host_callbacks == 0, gather_ops == 0, send_recv_ops == 0.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pickle
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np
from PIL import Image, ImageDraw

from tpu_mjx_render.adapters.openusd_loader import compile_openusd_to_wide_bvh
from tpu_mjx_render.roofline import inspect_hlo_ops
from wave3_500m_10b_8k import (
    CompactWideBVHMeshItem,
    CompactWideBVHSceneSpec,
    _bundle_aabb_tmin,
    _compute_camera_rays_band,
    _pallas_compact_bvh_raycast,
    _traverse_compact_bvh_in_kernel,
)
from wide_bvh_v3 import BLOCK_R, BRANCH_FACTOR, TILE_H, _pack_rays_tiled, _unpack_out_tiled

TAG_TO_ID = {
    "default": 0,
    "sand": 1,
    "foliage": 2,
    "bark": 3,
    "rock": 4,
    "coral": 5,
    "flower": 6,
    "water": 7,
}


def load_islandsun_hdri_bf16(exr_path: str, H_env: int = 256, W_env: int = 512) -> np.ndarray:
  """Load Disney's islandsun.exr into (H_env, 3*W_env) bfloat16/float32 packed matrix for MXU lookup."""
  import OpenEXR
  import Imath

  exr = OpenEXR.InputFile(exr_path)
  dw = exr.header()["dataWindow"]
  W_full = dw.max.x - dw.min.x + 1
  H_full = dw.max.y - dw.min.y + 1
  pt = Imath.PixelType(Imath.PixelType.HALF)
  r = np.frombuffer(exr.channel("R", pt), dtype=np.float16).reshape(H_full, W_full)
  g = np.frombuffer(exr.channel("G", pt), dtype=np.float16).reshape(H_full, W_full)
  b = np.frombuffer(exr.channel("B", pt), dtype=np.float16).reshape(H_full, W_full)

  # Box-filter downsample cleanly to (H_env, W_env)
  sy = max(1, H_full // H_env)
  sx = max(1, W_full // W_env)
  hc = (H_full // sy) * sy
  wc = (W_full // sx) * sx
  rgb_small = np.stack([
      r[:hc, :wc].astype(np.float32).reshape(H_full // sy, sy, W_full // sx, sx).mean(axis=(1, 3)),
      g[:hc, :wc].astype(np.float32).reshape(H_full // sy, sy, W_full // sx, sx).mean(axis=(1, 3)),
      b[:hc, :wc].astype(np.float32).reshape(H_full // sy, sy, W_full // sx, sx).mean(axis=(1, 3)),
  ], axis=-1)

  # Resize cleanly via PIL if dimensions don't match exact target
  if rgb_small.shape[0] != H_env or rgb_small.shape[1] != W_env:
    im = Image.fromarray(np.clip(rgb_small * 255.0 + 0.5, 0, 255).astype(np.uint8))
    im = im.resize((W_env, H_env), Image.Resampling.BILINEAR)
    rgb_small = np.asarray(im, dtype=np.float32) / 255.0

  # Crisp azure sky blue & bright white cumulus clouds from islandsun.exr
  rgb_lin = np.power(np.clip(rgb_small, 0.0, 1.0), 1.35) * 1.55
  lum = 0.2126 * rgb_lin[..., 0] + 0.7152 * rgb_lin[..., 1] + 0.0722 * rgb_lin[..., 2]
  rgb_lin = rgb_lin * (1.0 + 0.75 * np.maximum(lum - 0.65, 0.0))[..., None]
  return np.concatenate([rgb_lin[..., 0], rgb_lin[..., 1], rgb_lin[..., 2]], axis=1).astype(np.float32)


def pallas_sample_hdri_env(
    dirs: jax.Array,         # (B, R, 3) Z-up unit directions
    env_packed_bf16: jax.Array,  # (H_env, 3*W_env) bfloat16
    H: int,
    W: int,
    dome_rot_z_rad: float = 0.0,
    H_env: int = 256,
    W_env: int = 512,
) -> jax.Array:
  """Zero-gather Pallas MXU bilinear equirectangular HDRI lookup -> (B, R, 3) float32."""
  B, R, _ = dirs.shape
  # Convert Z-up (x, y, z) -> Y-up (x, z, -y) for standard equirectangular HDRI
  dx = dirs[..., 0]
  dy_y = dirs[..., 2]
  dz_y = -dirs[..., 1]
  phi = jnp.arctan2(dz_y, dx)
  u = (phi / (2.0 * np.pi) + 0.36) % 1.0
  v = jnp.arccos(jnp.clip(dy_y, -1.0, 1.0)) / np.pi

  # Pack (u, v) into tiled (total_tiles*8, 128)
  dummy = jnp.zeros_like(u)
  uv_3 = jnp.stack([u, v, dummy], axis=-1)
  uv_2d, n_tiles, can_2d = _pack_rays_tiled(uv_3, uv_3, dummy, H, W)
  total_tiles = B * n_tiles

  def _env_kernel(uv_ref, env_ref, out_ref):
    rt = uv_ref[...]
    u_t, v_t = rt[0, :], rt[1, :]
    fx_tot = u_t * float(W_env) - 0.5
    fy_tot = v_t * float(H_env) - 0.5
    ix0 = jnp.floor(fx_tot).astype(jnp.int32) % W_env
    ix1 = (ix0 + 1) % W_env
    iy0 = jnp.clip(jnp.floor(fy_tot).astype(jnp.int32), 0, H_env - 1)
    iy1 = jnp.minimum(iy0 + 1, H_env - 1)
    wx1 = fx_tot - jnp.floor(fx_tot)
    wx0 = 1.0 - wx1
    wy1 = fy_tot - jnp.floor(fy_tot)
    wy0 = 1.0 - wy1

    iota_y = jax.lax.broadcasted_iota(jnp.int32, (BLOCK_R, H_env), 1)
    wy_mat = (jnp.where(iota_y == iy0[:, None], wy0[:, None], 0.0)
              + jnp.where(iota_y == iy1[:, None], wy1[:, None], 0.0)).astype(jnp.bfloat16)

    row_mix = jnp.matmul(wy_mat, env_ref[...], preferred_element_type=jnp.float32)

    iota_x = jax.lax.broadcasted_iota(jnp.int32, (BLOCK_R, W_env), 1)
    wx_mat = (jnp.where(iota_x == ix0[:, None], wx0[:, None], 0.0)
              + jnp.where(iota_x == ix1[:, None], wx1[:, None], 0.0))

    r_out = jnp.sum(row_mix[:, :W_env] * wx_mat, axis=1)
    g_out = jnp.sum(row_mix[:, W_env:2 * W_env] * wx_mat, axis=1)
    b_out = jnp.sum(row_mix[:, 2 * W_env:] * wx_mat, axis=1)
    z = jnp.zeros((1, BLOCK_R), jnp.float32)
    out_ref[...] = jnp.concatenate([z, r_out[None, :], g_out[None, :], b_out[None, :], z, z, z, z], axis=0)

  out_2d = pl.pallas_call(
      _env_kernel,
      out_shape=jax.ShapeDtypeStruct((total_tiles * 8, BLOCK_R), jnp.float32),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          grid=(total_tiles,),
          in_specs=[
              pl.BlockSpec((8, BLOCK_R), lambda i: (i, 0)),
              pl.BlockSpec((H_env, W_env * 3), lambda i: (0, 0)),
          ],
          out_specs=pl.BlockSpec((8, BLOCK_R), lambda i: (i, 0)),
      ),
      compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
      interpret=False,
  )(uv_2d, env_packed_bf16)

  _, rgb_out, _ = _unpack_out_tiled(out_2d, B, H, W, n_tiles, can_2d, return_stats=False)
  return rgb_out


def pallas_tlas_affine_instanced_raycast(
    item: CompactWideBVHMeshItem,
    l2_roots_dev: jax.Array,
    l2_dev: jax.Array,
    l0_dev: jax.Array,
    leaf_dev: jax.Array,
    ray_o_world: jax.Array,
    ray_d_world: jax.Array,
    t_limit: jax.Array,
    inst_xpos: jax.Array,       # (B, K_inst, 3)
    inst_xmat: jax.Array,       # (B, K_inst, 3, 3)
    inst_inv_xmat: jax.Array,   # (B, K_inst, 3, 3) exact inverse for arbitrary scales!
    H: int,
    W: int,
    tlas_group_size: int = 64,
    tlas_top_groups: int = 4,
    max_active_per_tile: int = 16,
    any_hit: bool = False,
) -> Tuple[jax.Array, jax.Array]:
  """Hierarchical 3-Level Wide-TLAS-64 + Affine M_inv Compact BLAS raycast."""
  B, R, _ = ray_o_world.shape
  K_inst = int(inst_xpos.shape[1])
  rays_2d, n_tiles, can_2d_tile = _pack_rays_tiled(ray_o_world, ray_d_world, t_limit, H, W)
  total_tiles = B * n_tiles

  # 1. Exact world OBB -> AABB for all K_inst instances
  c_loc = jnp.asarray(0.5 * (item.obj_aabb_min + item.obj_aabb_max), dtype=jnp.float32)
  h_loc = jnp.asarray(0.5 * (item.obj_aabb_max - item.obj_aabb_min), dtype=jnp.float32)
  c_w = inst_xpos + jnp.einsum("bkij,j->bki", inst_xmat, c_loc, precision=jax.lax.Precision.HIGHEST)
  h_w = jnp.einsum("bkij,j->bki", jnp.abs(inst_xmat), h_loc, precision=jax.lax.Precision.HIGHEST)
  w_min = c_w - h_w - 1e-3
  w_max = c_w + h_w + 1e-3

  # Tile ray-bundles (B, n_tiles, 3)
  ro_t = rays_2d.reshape(B, n_tiles, 8, BLOCK_R)[:, :, 0:3, :].transpose(0, 1, 3, 2)
  rd_t = rays_2d.reshape(B, n_tiles, 8, BLOCK_R)[:, :, 3:6, :].transpose(0, 1, 3, 2)
  tl_t = rays_2d.reshape(B, n_tiles, 8, BLOCK_R)[:, :, 6, :]
  ro_min, ro_max = jnp.min(ro_t, axis=2), jnp.max(ro_t, axis=2)
  rd_min, rd_max = jnp.min(rd_t, axis=2), jnp.max(rd_t, axis=2)
  max_tl = jnp.max(tl_t, axis=2)

  # Adaptive 1/2/3-level Wide-TLAS-64 culling (supports K_inst up to 20M+ with <25MB scratch!)
  if K_inst <= 64:
    cand_w_min, cand_w_max = w_min, w_max
    base_ids = jnp.broadcast_to(jnp.arange(K_inst, dtype=jnp.int32)[None, None, :], (B, n_tiles, K_inst))
    cand_valid = jnp.ones((B, n_tiles, K_inst), dtype=jnp.bool_)
  else:
    G = max(1, (K_inst + tlas_group_size - 1) // tlas_group_size)
    K_pad_g = G * tlas_group_size
    pad_g = K_pad_g - K_inst
    w_min_p = jnp.pad(w_min, ((0, 0), (0, pad_g), (0, 0)), constant_values=1e20) if pad_g > 0 else w_min
    w_max_p = jnp.pad(w_max, ((0, 0), (0, pad_g), (0, 0)), constant_values=-1e20) if pad_g > 0 else w_max
    g_min = jnp.min(w_min_p.reshape(B, G, tlas_group_size, 3), axis=2)
    g_max = jnp.max(w_max_p.reshape(B, G, tlas_group_size, 3), axis=2)

    if G > 256:
      # 3-Level TLAS-64: Super-Groups of 64 groups (= 4,096 instances each)
      SG = max(1, (G + 63) // 64)
      G_pad = SG * 64
      pad_sg = G_pad - G
      g_min_p = jnp.pad(g_min, ((0, 0), (0, pad_sg), (0, 0)), constant_values=1e20) if pad_sg > 0 else g_min
      g_max_p = jnp.pad(g_max, ((0, 0), (0, pad_sg), (0, 0)), constant_values=-1e20) if pad_sg > 0 else g_max
      sg_min = jnp.min(g_min_p.reshape(B, SG, 64, 3), axis=2)
      sg_max = jnp.max(g_max_p.reshape(B, SG, 64, 3), axis=2)

      top_sg_k = min(6, SG)
      ok_sg, tmin_sg = _bundle_aabb_tmin(
          ro_min[:, :, None, :], ro_max[:, :, None, :],
          rd_min[:, :, None, :], rd_max[:, :, None, :],
          max_tl[:, :, None], sg_min[:, None, :, :], sg_max[:, None, :, :],
      )
      score_sg = jnp.where(ok_sg, tmin_sg, 1e20)
      neg_sg, top_sg = jax.lax.top_k(-score_sg, top_sg_k)
      sel_sg = (top_sg[:, :, :, None] == jnp.arange(SG, dtype=top_sg.dtype)[None, None, None, :]).astype(jnp.float32)
      cg_min = jnp.einsum("btmg,bgsc->btmsc", sel_sg, g_min_p.reshape(B, SG, 64, 3), precision=jax.lax.Precision.HIGHEST).reshape(B, n_tiles, top_sg_k * 64, 3)
      cg_max = jnp.einsum("btmg,bgsc->btmsc", sel_sg, g_max_p.reshape(B, SG, 64, 3), precision=jax.lax.Precision.HIGHEST).reshape(B, n_tiles, top_sg_k * 64, 3)
      g_ids = (top_sg[:, :, :, None] * 64 + jnp.arange(64)[None, None, None, :]).reshape(B, n_tiles, top_sg_k * 64)
      cg_valid = (neg_sg[:, :, :, None] > -1e19).repeat(64, axis=-1).reshape(B, n_tiles, top_sg_k * 64) & (g_ids < G)

      ok_g, tmin_g = _bundle_aabb_tmin(
          ro_min[:, :, None, :], ro_max[:, :, None, :],
          rd_min[:, :, None, :], rd_max[:, :, None, :],
          max_tl[:, :, None], cg_min, cg_max,
      )
      score_g = jnp.where(ok_g & cg_valid, tmin_g, 1e20)
      top_g_k = min(tlas_top_groups, cg_min.shape[2])
      neg_g, top_gl = jax.lax.top_k(-score_g, top_g_k)
      sel_gl = (top_gl[:, :, :, None] == jnp.arange(cg_min.shape[2], dtype=top_gl.dtype)[None, None, None, :]).astype(jnp.float32)
      top_g = jnp.round(jnp.einsum("btkc,btc->btk", sel_gl, g_ids.astype(jnp.float32), precision=jax.lax.Precision.HIGHEST)).astype(jnp.int32)
      sel_g = (top_g[:, :, :, None] == jnp.arange(G, dtype=top_g.dtype)[None, None, None, :]).astype(jnp.float32)
    else:
      top_g_k = min(tlas_top_groups, G)
      ok_g, tmin_g = _bundle_aabb_tmin(
          ro_min[:, :, None, :], ro_max[:, :, None, :],
          rd_min[:, :, None, :], rd_max[:, :, None, :],
          max_tl[:, :, None], g_min[:, None, :, :], g_max[:, None, :, :],
      )
      score_g = jnp.where(ok_g, tmin_g, 1e20)
      neg_g, top_g = jax.lax.top_k(-score_g, top_g_k)
      sel_g = (top_g[:, :, :, None] == jnp.arange(G, dtype=top_g.dtype)[None, None, None, :]).astype(jnp.float32)

    cand_w_min = jnp.einsum("btmg,bgsc->btmsc", sel_g, w_min_p.reshape(B, G, tlas_group_size, 3), precision=jax.lax.Precision.HIGHEST).reshape(B, n_tiles, top_g_k * tlas_group_size, 3)
    cand_w_max = jnp.einsum("btmg,bgsc->btmsc", sel_g, w_max_p.reshape(B, G, tlas_group_size, 3), precision=jax.lax.Precision.HIGHEST).reshape(B, n_tiles, top_g_k * tlas_group_size, 3)
    base_ids = (top_g[:, :, :, None] * tlas_group_size + jnp.arange(tlas_group_size)[None, None, None, :]).reshape(B, n_tiles, top_g_k * tlas_group_size)
    cand_valid = (neg_g[:, :, :, None] > -1e19).repeat(tlas_group_size, axis=-1).reshape(B, n_tiles, top_g_k * tlas_group_size) & (base_ids < K_inst)

  # Final Instance Level: pick top K_act front-most instances per tile
  ok_i, tmin_i = _bundle_aabb_tmin(
      ro_min[:, :, None, :], ro_max[:, :, None, :],
      rd_min[:, :, None, :], rd_max[:, :, None, :],
      max_tl[:, :, None], cand_w_min, cand_w_max,
  )
  score_i = jnp.where(ok_i & cand_valid, tmin_i, 1e20)
  M_cand = score_i.shape[-1]
  K_act = min(max_active_per_tile, M_cand)
  neg_i, top_local = jax.lax.top_k(-score_i, K_act)
  sel_i = (top_local[:, :, :, None] == jnp.arange(M_cand, dtype=top_local.dtype)[None, None, None, :]).astype(jnp.float32)
  top_inst_ids = jnp.round(jnp.einsum("btkc,btc->btk", sel_i, base_ids.astype(jnp.float32), precision=jax.lax.Precision.HIGHEST)).astype(jnp.int32)
  act_ids = jnp.where((-neg_i) < 1e19, top_inst_ids, -1).astype(jnp.float32)

  act_2d = jnp.concatenate([
      jnp.concatenate([
          act_ids.reshape(total_tiles, 1, K_act),
          jnp.full((total_tiles, 1, 128 - K_act), -1.0, dtype=jnp.float32),
      ], axis=2),
      jnp.full((total_tiles, 7, 128), -1.0, dtype=jnp.float32),
  ], axis=1).reshape(total_tiles * 8, 128)

  # Pack (pos, M_inv) into HBM as (B * n_pose_chunks, 16, 128)
  n_pose_chunks = max(1, (K_inst + 127) // 128)
  K_pose_pad = n_pose_chunks * 128
  pad_k = K_pose_pad - K_inst
  pos_t = inst_xpos.transpose(0, 2, 1)
  inv_t = inst_inv_xmat.reshape(B, K_inst, 9).transpose(0, 2, 1)
  if pad_k > 0:
    pos_t = jnp.concatenate([pos_t, jnp.zeros((B, 3, pad_k), dtype=jnp.float32)], axis=2)
    inv_t = jnp.concatenate([inv_t, jnp.zeros((B, 9, pad_k), dtype=jnp.float32)], axis=2)
  poses_b = jnp.concatenate([pos_t, inv_t, jnp.zeros((B, 4, K_pose_pad), dtype=jnp.float32)], axis=1)
  poses_hbm = poses_b.reshape(B, 16, n_pose_chunks, 128).transpose(0, 2, 1, 3).reshape(B * n_pose_chunks, 16, 128)

  def _inst_kernel(
      ray_ref, act_ref, poses_hbm_ref, l2_roots_ref, l2_hbm_ref, l0_hbm_ref, leaf_hbm_ref, out_ref,
      pose_vmem, l2_vmem, l0_vmem, leaf_vmem, sem_pose, sem_l2, sem_l0, sem_leaf,
  ):
    t_idx = pl.program_id(0)
    world_b = t_idx // n_tiles
    cols_iota_64 = jax.lax.broadcasted_iota(jnp.int32, (1, BRANCH_FACTOR), 1)
    cols_iota_128 = jax.lax.broadcasted_iota(jnp.int32, (1, 128), 1)

    rt = ray_ref[...]
    act_r = act_ref[0:1, :]
    l2_roots_blk = l2_roots_ref[...]

    w_rox, w_roy, w_roz = jnp.transpose(rt[0:1, :], (1, 0)), jnp.transpose(rt[1:2, :], (1, 0)), jnp.transpose(rt[2:3, :], (1, 0))
    w_rdx, w_rdy, w_rdz = jnp.transpose(rt[3:4, :], (1, 0)), jnp.transpose(rt[4:5, :], (1, 0)), jnp.transpose(rt[5:6, :], (1, 0))
    bd = jnp.transpose(rt[6:7, :], (1, 0))
    zcol = jnp.zeros((BLOCK_R, 1), jnp.float32)

    def _read_act_id(k_idx):
      return jnp.sum(jnp.where(cols_iota_128 == k_idx, act_r, 0.0))

    def _inst_cond(ci):
      k, inst_f, *_ = ci
      return (k < K_act) & (inst_f >= 0.0)

    def _inst_body(ci):
      k, inst_f, bd_i, nx_i, ny_i, nz_i = ci
      inst_id = inst_f.astype(jnp.int32)
      chunk_id = world_b * n_pose_chunks + (inst_id >> 7)
      col_id = inst_id & 127

      cp_p = pltpu.make_async_copy(poses_hbm_ref.at[chunk_id, :, :], pose_vmem, sem_pose)
      cp_p.start(); cp_p.wait()
      poses_w = pose_vmem[...]

      cmask = cols_iota_128 == col_id
      px = jnp.sum(jnp.where(cmask, poses_w[0:1, :], 0.0))
      py = jnp.sum(jnp.where(cmask, poses_w[1:2, :], 0.0))
      pz = jnp.sum(jnp.where(cmask, poses_w[2:3, :], 0.0))
      # Exact affine M_inv rows:
      m00 = jnp.sum(jnp.where(cmask, poses_w[3:4, :], 0.0))
      m01 = jnp.sum(jnp.where(cmask, poses_w[4:5, :], 0.0))
      m02 = jnp.sum(jnp.where(cmask, poses_w[5:6, :], 0.0))
      m10 = jnp.sum(jnp.where(cmask, poses_w[6:7, :], 0.0))
      m11 = jnp.sum(jnp.where(cmask, poses_w[7:8, :], 0.0))
      m12 = jnp.sum(jnp.where(cmask, poses_w[8:9, :], 0.0))
      m20 = jnp.sum(jnp.where(cmask, poses_w[9:10, :], 0.0))
      m21 = jnp.sum(jnp.where(cmask, poses_w[10:11, :], 0.0))
      m22 = jnp.sum(jnp.where(cmask, poses_w[11:12, :], 0.0))

      dx_p, dy_p, dz_p = w_rox - px, w_roy - py, w_roz - pz
      rox = m00 * dx_p + m01 * dy_p + m02 * dz_p
      roy = m10 * dx_p + m11 * dy_p + m12 * dz_p
      roz = m20 * dx_p + m21 * dy_p + m22 * dz_p
      # Unnormalized local ray direction -> local t equals world t!
      rdx = m00 * w_rdx + m01 * w_rdy + m02 * w_rdz
      rdy = m10 * w_rdx + m11 * w_rdy + m12 * w_rdz
      rdz = m20 * w_rdx + m21 * w_rdy + m22 * w_rdz

      irx = jnp.where(jnp.abs(rdx) > 1e-12, 1.0 / jnp.where(jnp.abs(rdx) > 1e-12, rdx, 1.0), 1e12)
      iry = jnp.where(jnp.abs(rdy) > 1e-12, 1.0 / jnp.where(jnp.abs(rdy) > 1e-12, rdy, 1.0), 1e12)
      irz = jnp.where(jnp.abs(rdz) > 1e-12, 1.0 / jnp.where(jnp.abs(rdz) > 1e-12, rdz, 1.0), 1e12)

      new_bd, lnx, lny, lnz, _ = _traverse_compact_bvh_in_kernel(
          rox, roy, roz, irx, iry, irz, rdx, rdy, rdz,
          bd_i, zcol, zcol, zcol,
          l2_roots_blk, l2_hbm_ref, l0_hbm_ref, leaf_hbm_ref,
          l2_vmem, l0_vmem, leaf_vmem, sem_l2, sem_l0, sem_leaf,
          cols_iota_64, any_hit, False,
      )
      imp = new_bd < bd_i
      k_next = k + 1
      inst_next_f = _read_act_id(k_next)
      if any_hit:
        return (k_next, inst_next_f, new_bd, nx_i, ny_i, nz_i)

      # Exact normal transform: normalize(M_inv^T @ n_local)
      wnx = m00 * lnx + m10 * lny + m20 * lnz
      wny = m01 * lnx + m11 * lny + m21 * lnz
      wnz = m02 * lnx + m12 * lny + m22 * lnz
      inv_nl = 1.0 / jnp.maximum(jnp.sqrt(wnx * wnx + wny * wny + wnz * wnz), 1e-12)
      wnx, wny, wnz = wnx * inv_nl, wny * inv_nl, wnz * inv_nl
      return (k_next, inst_next_f, new_bd, jnp.where(imp, wnx, nx_i), jnp.where(imp, wny, ny_i), jnp.where(imp, wnz, nz_i))

    _, _, f_bd, f_nx, f_ny, f_nz = jax.lax.while_loop(
        _inst_cond, _inst_body, (0, _read_act_id(0), bd, zcol, zcol, zcol)
    )
    z = jnp.zeros((1, BLOCK_R), jnp.float32)
    out_ref[...] = jnp.concatenate(
        [jnp.transpose(f_bd, (1, 0)), jnp.transpose(f_nx, (1, 0)), jnp.transpose(f_ny, (1, 0)), jnp.transpose(f_nz, (1, 0)), z, z, z, z],
        axis=0,
    )

  out_2d = pl.pallas_call(
      _inst_kernel,
      out_shape=jax.ShapeDtypeStruct((total_tiles * 8, BLOCK_R), jnp.float32),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          grid=(total_tiles,),
          in_specs=[
              pl.BlockSpec((8, BLOCK_R), lambda i: (i, 0)),
              pl.BlockSpec((8, 128), lambda i: (i, 0)),
              pl.BlockSpec(memory_space=pltpu.HBM),
              pl.BlockSpec((8, 128), lambda i: (0, 0)),
              pl.BlockSpec(memory_space=pltpu.HBM),
              pl.BlockSpec(memory_space=pltpu.HBM),
              pl.BlockSpec(memory_space=pltpu.HBM),
          ],
          out_specs=pl.BlockSpec((8, BLOCK_R), lambda i: (i, 0)),
          scratch_shapes=[
              pltpu.VMEM((16, 128), jnp.float32),
              pltpu.VMEM((8, 128), jnp.float32),
              pltpu.VMEM((8, 128), jnp.float32),
              pltpu.VMEM((16, 128), jnp.float32),
              pltpu.SemaphoreType.DMA,
              pltpu.SemaphoreType.DMA,
              pltpu.SemaphoreType.DMA,
              pltpu.SemaphoreType.DMA,
          ],
      ),
      compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
      interpret=False,
  )(rays_2d, act_2d, poses_hbm, l2_roots_dev, l2_dev, l0_dev, leaf_dev)

  d_out, n_out, _ = _unpack_out_tiled(out_2d, B, H, W, n_tiles, can_2d_tile, return_stats=False)
  return d_out, n_out


def agx_filmic_tonemap(hdr_rgb: jax.Array, exposure: float = 1.15) -> jax.Array:
  """AgX filmic tonemapping + sRGB conversion (rich highlights & tropical saturation)."""
  x = jnp.maximum(hdr_rgb * exposure, 0.0)
  agx_in = jnp.array([
      [0.84247906, 0.07843360, 0.07922375],
      [0.04232824, 0.87846864, 0.07916613],
      [0.04237565, 0.07843360, 0.87914297],
  ], dtype=jnp.float32)
  x = jnp.einsum("ij,...j->...i", agx_in, x)
  x = jnp.clip(jnp.log2(jnp.maximum(x, 1e-6)), -12.47393, 4.026069)
  x = (x + 12.47393) / (4.026069 + 12.47393)
  x2 = x * x
  x4 = x2 * x2
  y = (
      15.5 * x4 * x2
      - 40.14 * x4 * x
      + 31.96 * x4
      - 6.868 * x2 * x
      + 0.4298 * x2
      + 0.1191 * x
      - 0.00232
  )
  agx_out = jnp.array([
      [1.19687900, -0.09802088, -0.09902974],
      [-0.05289685, 1.15190313, -0.09896118],
      [-0.05297163, -0.09804345, 1.15107367],
  ], dtype=jnp.float32)
  out = jnp.einsum("ij,...j->...i", agx_out, y)
  out = jnp.clip(out, 0.0, 1.0)
  # Gentle punchy contrast curve
  out = out * out * (3.0 - 2.0 * out) * 0.22 + out * 0.78
  return jnp.clip(out, 0.0, 1.0)


def make_moana_4chip_renderer(
    compiled: Dict[str, Any],
    env_bf16: jax.Array,
    devices: List[jax.Device],
    H: int,
    W: int,
    dome_rot_z_deg: float = 25.0,
    use_shadows: bool = True,
):
  """4-chip spatial-sharded Moana renderer with PBR + Beer-Lambert Ocean + HDRI + AgX."""
  n_chips = len(devices)
  assert H % n_chips == 0, (H, n_chips)
  band_h = H // n_chips
  band_h_pad = ((band_h + TILE_H - 1) // TILE_H) * TILE_H
  mesh = Mesh(np.array(devices), ("spatial",))

  u_spec = compiled["unique_spec"]
  u_devs = compiled["unique_dev_arrays"]
  u_pbr = compiled["unique_pbr"]
  w_spec = compiled["water_spec"]
  w_devs = compiled["water_dev_arrays"]
  i_spec = compiled["inst_spec"]
  i_devs = compiled["inst_dev_arrays"]
  i_pbr = compiled["inst_pbr"]

  base_spec = u_spec if u_spec is not None else i_spec
  spec_band = dataclasses.replace(base_spec, height=band_h_pad, width=W)

  # Precompute exact inverse 3x3 affine matrices for all instanced groups on TPU
  rep = NamedSharding(mesh, P())
  inst_inv_xmats = []
  for xm in compiled["inst_xmat"]:
    xm_np = np.asarray(xm)
    # Robust inverse even if near-singular scale
    inv_np = np.zeros_like(xm_np)
    flat_m = xm_np.reshape(-1, 3, 3)
    flat_inv = inv_np.reshape(-1, 3, 3)
    for idx in range(flat_m.shape[0]):
      try:
        flat_inv[idx] = np.linalg.inv(flat_m[idx])
      except np.linalg.LinAlgError:
        flat_inv[idx] = np.linalg.pinv(flat_m[idx])
    inst_inv_xmats.append(jax.device_put(inv_np.astype(np.float32), rep))
  inst_inv_xmats_t = tuple(inst_inv_xmats)

  # Sun direction from Disney's sun_quad_llc (Y-up [0.324, 0.643, 0.694] -> Z-up [0.324, -0.694, 0.643])
  sun_dir_np = np.array([0.324, -0.694, 0.643], dtype=np.float32)
  sun_dir_np /= np.linalg.norm(sun_dir_np)
  dome_rot_rad = float(np.deg2rad(dome_rot_z_deg))

  def _shard_fn(u_da, u_oxp, u_oxm, w_da, i_da, i_xp, i_xm, i_inv, cxp, cxm, env_b):
    bidx = jax.lax.axis_index("spatial")
    ray_orig, ray_dir, neg_z = _compute_camera_rays_band(spec_band, cxp, cxm, H, W, bidx, band_h_pad, band_stride_h=band_h)
    B, R, _ = ray_orig.shape
    max_dist = jnp.broadcast_to(((spec_band.zfar - spec_band.znear) / neg_z)[None, :], (B, R))

    best_d = max_dist
    best_hit = jnp.zeros((B, R), dtype=jnp.bool_)
    best_nrm = jnp.zeros((B, R, 3), dtype=jnp.float32)
    best_alb = jnp.zeros((B, R, 3), dtype=jnp.float32)
    best_rough = jnp.full((B, R), 0.70, dtype=jnp.float32)
    best_tag = jnp.zeros((B, R), dtype=jnp.int32)

    if u_spec is not None:
      for m_idx, it in enumerate(u_spec.meshes):
        l2_r, l2_a, l0_a, leaf_b = u_da[m_idx]
        p = u_oxp[:, m_idx, :]
        rmat = u_oxm[:, m_idx, :, :]
        ro_loc = jnp.einsum("bji,brj->bri", rmat, ray_orig - p[:, None, :], precision=jax.lax.Precision.HIGHEST)
        rd_loc = jnp.einsum("bji,brj->bri", rmat, ray_dir, precision=jax.lax.Precision.HIGHEST)
        d_m, n_loc_m, _ = _pallas_compact_bvh_raycast(l2_r, l2_a, l0_a, leaf_b, ro_loc, rd_loc, best_d, band_h_pad, W, any_hit=False, interpret=False)
        imp = d_m < best_d
        n_w_m = jnp.einsum("bij,brj->bri", rmat, n_loc_m, precision=jax.lax.Precision.HIGHEST)
        best_d = jnp.where(imp, d_m, best_d)
        best_hit = best_hit | imp
        best_nrm = jnp.where(imp[..., None], n_w_m, best_nrm)
        best_alb = jnp.where(imp[..., None], jnp.asarray(it.rgba[:3], dtype=jnp.float32)[None, None, :], best_alb)
        r_m, _, tag_m = u_pbr[m_idx]
        best_rough = jnp.where(imp, float(r_m), best_rough)
        best_tag = jnp.where(imp, jnp.int32(TAG_TO_ID.get(tag_m, 0)), best_tag)

    if i_spec is not None:
      for m_idx, it in enumerate(i_spec.meshes):
        l2_r, l2_a, l0_a, leaf_b = i_da[m_idx]
        d_m, n_w_m = pallas_tlas_affine_instanced_raycast(
            it, l2_r, l2_a, l0_a, leaf_b,
            ray_orig, ray_dir, best_d, i_xp[m_idx], i_xm[m_idx], i_inv[m_idx],
            band_h_pad, W, tlas_group_size=64, tlas_top_groups=4, max_active_per_tile=16, any_hit=False,
        )
        imp = d_m < best_d
        best_d = jnp.where(imp, d_m, best_d)
        best_hit = best_hit | imp
        best_nrm = jnp.where(imp[..., None], n_w_m, best_nrm)
        best_alb = jnp.where(imp[..., None], jnp.asarray(it.rgba[:3], dtype=jnp.float32)[None, None, :], best_alb)
        r_m, _, tag_m = i_pbr[m_idx]
        best_rough = jnp.where(imp, float(r_m), best_rough)
        best_tag = jnp.where(imp, jnp.int32(TAG_TO_ID.get(tag_m, 0)), best_tag)

    # Trace Ocean Water Mesh separately for Beer-Lambert transmittance + Fresnel sky reflection!
    water_d = max_dist
    water_hit = jnp.zeros((B, R), dtype=jnp.bool_)
    water_nrm = jnp.zeros((B, R, 3), dtype=jnp.float32)
    if w_spec is not None and len(w_spec.meshes) > 0:
      for m_idx, it in enumerate(w_spec.meshes):
        l2_r, l2_a, l0_a, leaf_b = w_da[m_idx]
        d_w, n_w, _ = _pallas_compact_bvh_raycast(l2_r, l2_a, l0_a, leaf_b, ray_orig, ray_dir, water_d, band_h_pad, W, any_hit=False, interpret=False)
        imp_w = d_w < water_d
        water_d = jnp.where(imp_w, d_w, water_d)
        water_hit = water_hit | imp_w
        water_nrm = jnp.where(imp_w[..., None], n_w, water_nrm)

    vdir = -ray_dir
    best_nrm = jnp.where(best_hit[..., None], best_nrm, 0.0)
    flip = jnp.sum(best_nrm * vdir, axis=-1, keepdims=True) < 0.0
    best_nrm = jnp.where(flip, -best_nrm, best_nrm)
    hit_pos = ray_orig + jnp.where(best_hit, best_d, 0.0)[..., None] * ray_dir

    # Procedural world-space PBR micro-variation (zero flat plastic clay, zero regular grid!)
    px, py, pz = hit_pos[..., 0], hit_pos[..., 1], hit_pos[..., 2]
    n_wave1 = jnp.sin(px * 0.37 + py * 0.83 + pz * 0.29) * jnp.sin(py * 0.41 - px * 0.71 + pz * 0.53)
    n_wave2 = jnp.sin(px * 1.93 - py * 2.71 + pz * 1.37) * jnp.cos(py * 2.19 + px * 1.57 - pz * 2.41)
    # Sand ripples + golden warm tint
    sand_rip = jnp.sin(px * 7.5 + py * 3.8 + 0.6 * n_wave1)
    sand_alb = best_alb * (0.94 + 0.08 * sand_rip + 0.04 * n_wave2)[..., None]
    # Foliage lush multi-tone deep tropical emerald/olive variation
    fol_var = 0.60 + 0.22 * n_wave1 + 0.12 * n_wave2
    fol_alb = best_alb * jnp.stack([fol_var * 0.88, fol_var * 1.04, fol_var * 0.72], axis=-1)
    # Volcanic basalt rock strata + wet shoreline darkening
    rock_var = 0.75 + 0.28 * n_wave1 + 0.15 * n_wave2
    wet_mask = jnp.clip(1.0 - (pz + 0.1) / 0.8, 0.0, 1.0)
    rock_alb = best_alb * rock_var[..., None] * (1.0 - 0.38 * wet_mask[..., None])
    rock_rough = jnp.where(best_tag == 4, best_rough * (1.0 - 0.45 * wet_mask), best_rough)
    # Coral mossy chartreuse/olive/ochre patches
    coral_alb = best_alb * jnp.stack([1.05 + 0.35 * n_wave1, 1.15 + 0.35 * n_wave2, 0.65 + 0.20 * n_wave1], axis=-1)

    albedo = jnp.where((best_tag == 1)[..., None], sand_alb, best_alb)
    albedo = jnp.where((best_tag == 2)[..., None], fol_alb, albedo)
    albedo = jnp.where((best_tag == 4)[..., None], rock_alb, albedo)
    albedo = jnp.where((best_tag == 5)[..., None], coral_alb, albedo)
    albedo = jnp.clip(albedo, 0.01, 0.98)

    L = jnp.broadcast_to(jnp.asarray(sun_dir_np, dtype=jnp.float32)[None, None, :], ray_dir.shape)
    ndotl = jnp.maximum(0.0, jnp.sum(best_nrm * L, axis=-1))

    # Directional sun shadow raycast (any_hit=True) with distance-adaptive bias (zero distant shadow acne)
    vis = jnp.ones((B, R), dtype=jnp.float32)
    if use_shadows:
      sh_bias = jnp.maximum(0.03, best_d * 0.0012)[..., None]
      sh_o = hit_pos + best_nrm * sh_bias
      sh_lim = jnp.where(best_hit & (ndotl > 0.02), 350.0, 0.0)
      occ = jnp.zeros((B, R), dtype=jnp.bool_)
      if u_spec is not None:
        for m_idx, _it in enumerate(u_spec.meshes):
          l2_r, l2_a, l0_a, leaf_b = u_da[m_idx]
          p = u_oxp[:, m_idx, :]
          rmat = u_oxm[:, m_idx, :, :]
          ro_l = jnp.einsum("bji,brj->bri", rmat, sh_o - p[:, None, :], precision=jax.lax.Precision.HIGHEST)
          rd_l = jnp.einsum("bji,brj->bri", rmat, L, precision=jax.lax.Precision.HIGHEST)
          sh_lim_rem = jnp.where(occ, 0.0, sh_lim)
          d_sh, _, _ = _pallas_compact_bvh_raycast(l2_r, l2_a, l0_a, leaf_b, ro_l, rd_l, sh_lim_rem, band_h_pad, W, any_hit=True, interpret=False)
          occ = occ | (d_sh < sh_lim_rem)
      if i_spec is not None:
        for m_idx, it in enumerate(i_spec.meshes):
          l2_r, l2_a, l0_a, leaf_b = i_da[m_idx]
          sh_lim_rem = jnp.where(occ, 0.0, sh_lim)
          d_sh, _ = pallas_tlas_affine_instanced_raycast(
              it, l2_r, l2_a, l0_a, leaf_b,
              sh_o, L, sh_lim_rem, i_xp[m_idx], i_xm[m_idx], i_inv[m_idx],
              band_h_pad, W, tlas_group_size=64, tlas_top_groups=4, max_active_per_tile=16, any_hit=True,
          )
          occ = occ | (d_sh < sh_lim_rem)
      vis = jnp.where(occ, 0.07, 1.0)

    # GGX Specular + Subsurface Foliage Translucency + Sky Hemisphere Ambient
    sun_col = jnp.array([3.35, 2.95, 2.40], dtype=jnp.float32)[None, None, :]
    H_vec = (L + vdir) / jnp.maximum(jnp.linalg.norm(L + vdir, axis=-1, keepdims=True), 1e-12)
    ndoth = jnp.maximum(0.0, jnp.sum(best_nrm * H_vec, axis=-1))
    ndotv = jnp.maximum(1e-3, jnp.sum(best_nrm * vdir, axis=-1))
    alpha = jnp.clip(rock_rough * rock_rough, 0.04, 0.98)
    a2 = alpha * alpha
    denom = ndoth * ndoth * (a2 - 1.0) + 1.0
    D_ggx = a2 / (np.pi * denom * denom + 1e-6)
    F_schlick = 0.04 + 0.96 * ((1.0 - jnp.maximum(0.0, jnp.sum(H_vec * vdir, axis=-1))) ** 5.0)
    spec_term = (D_ggx * F_schlick) / (4.0 * ndotv * jnp.maximum(ndotl, 0.05) + 0.15)
    spec_term = jnp.minimum(spec_term, 2.5) * vis

    # Foliage subsurface sunlit translucency
    sss_dir = jnp.maximum(0.0, jnp.sum(vdir * (-L + 0.35 * best_nrm), axis=-1)) ** 2.5
    sss_mask = ((best_tag == 2) | (best_tag == 6)).astype(jnp.float32)
    sss_rgb = albedo * sun_col * (0.48 * sss_dir * sss_mask * (0.25 + 0.75 * vis))[..., None]

    # Deep sculpted hemisphere sky irradiance + crevice AO
    sky_up = 0.5 + 0.5 * best_nrm[..., 2]
    sky_fill = (
        sky_up[..., None] * jnp.array([0.38, 0.58, 0.85], dtype=jnp.float32)[None, None, :]
        + (1.0 - sky_up[..., None]) * jnp.array([0.14, 0.22, 0.18], dtype=jnp.float32)[None, None, :]
    )
    ao = jnp.where(best_tag == 2, 0.45 + 0.55 * sky_up, 0.65 + 0.35 * sky_up)
    solid_hdr = (
        albedo * (sun_col * (ndotl * vis)[..., None] * (1.0 / np.pi) + sky_fill * ao[..., None] * 0.32)
        + sun_col * (spec_term * ndotl)[..., None]
        + sss_rgb
    )

    # Sample Disney's real islandsun.exr for sky background & ocean Fresnel reflections!
    is_water_front = water_hit & (water_d < best_d)
    water_pos = ray_orig + jnp.where(water_hit, water_d, 0.0)[..., None] * ray_dir
    wx, wy = water_pos[..., 0], water_pos[..., 1]
    # Gentle realistic tropical lagoon capillary ripples
    w_nx = 0.016 * jnp.cos(wx * 2.1 + wy * 1.4) + 0.009 * jnp.sin(wx * 6.3 - wy * 5.1) + 0.004 * jnp.cos(wx * 14.0 + wy * 11.0)
    w_ny = 0.016 * jnp.sin(wy * 2.3 - wx * 1.2) + 0.009 * jnp.cos(wy * 5.8 + wx * 4.7) + 0.004 * jnp.sin(wy * 13.5 - wx * 9.5)
    w_nrm = jnp.where(water_nrm[..., 2:3] >= 0.0, water_nrm, -water_nrm) + jnp.stack([w_nx, w_ny, jnp.zeros_like(w_nx)], axis=-1)
    w_nrm = w_nrm / jnp.maximum(jnp.linalg.norm(w_nrm, axis=-1, keepdims=True), 1e-8)
    refl_dir = ray_dir - 2.0 * jnp.sum(ray_dir * w_nrm, axis=-1, keepdims=True) * w_nrm
    refl_dir = jnp.where(refl_dir[..., 2:3] < 0.02, jnp.concatenate([refl_dir[..., :2], jnp.full_like(refl_dir[..., 2:3], 0.02)], axis=-1), refl_dir)
    refl_dir = refl_dir / jnp.maximum(jnp.linalg.norm(refl_dir, axis=-1, keepdims=True), 1e-8)

    env_lookup_dir = jnp.where(is_water_front[..., None], refl_dir, ray_dir)
    env_rgb = pallas_sample_hdri_env(env_lookup_dir, env_bf16, band_h_pad, W, dome_rot_z_rad=dome_rot_rad, H_env=512, W_env=1024)

    # Beer-Lambert shallow turquoise lagoon refraction over underwater coral/sand + Fresnel reflection!
    t_water = jnp.where(best_hit, jnp.maximum(0.0, best_d - water_d), 45.0)
    sigma_t = jnp.array([0.22, 0.045, 0.12], dtype=jnp.float32)[None, None, :]
    transmittance = jnp.exp(-sigma_t * t_water[..., None])
    in_scatter = jnp.array([0.006, 0.22, 0.25], dtype=jnp.float32)[None, None, :] * (1.0 - jnp.exp(-0.20 * t_water[..., None]))
    underwater_rgb = jnp.where(best_hit[..., None], solid_hdr * 2.45 * transmittance + in_scatter, in_scatter * 0.55)

    cos_w = jnp.maximum(0.0, jnp.sum(w_nrm * vdir, axis=-1))
    fresnel_w = jnp.minimum(0.45, 0.03 + 0.60 * ((1.0 - cos_w) ** 4.0))
    w_ndoth = jnp.maximum(0.0, jnp.sum(w_nrm * H_vec, axis=-1))
    w_sun_spec = sun_col * ((w_ndoth ** 420.0) * 1.4 + (w_ndoth ** 64.0) * 0.03)[..., None]
    water_rgb = underwater_rgb * (1.0 - fresnel_w[..., None]) + env_rgb * 0.62 * fresnel_w[..., None] + w_sun_spec

    surf_rgb = jnp.where(is_water_front[..., None], water_rgb, solid_hdr)
    eff_d = jnp.where(is_water_front, water_d, best_d)
    any_surface = best_hit | is_water_front

    # Subtle tropical horizon aerial perspective (keeps ridges rich emerald green)
    fog_t = jnp.clip(1.0 - jnp.exp(-eff_d * 0.00011), 0.0, 0.18)[..., None]
    horizon_mist = jnp.array([0.42, 0.68, 0.90], dtype=jnp.float32)[None, None, :]
    surf_fog = surf_rgb * (1.0 - fog_t) + horizon_mist * fog_t

    hdr_final = jnp.where(any_surface[..., None], surf_fog, env_rgb)
    ldr_rgb = agx_filmic_tonemap(hdr_final, exposure=1.18)
    return ldr_rgb.reshape(B, band_h_pad, W, 3)[:, :band_h, :, :]

  sharded = shard_map(
      _shard_fn,
      mesh=mesh,
      in_specs=(P(), P(), P(), P(), P(), P(), P(), P(), P(), P(), P()),
      out_specs=P(None, "spatial", None, None),
      check_rep=False,
  )
  fn = jax.jit(sharded)
  runtime_args = (
      u_devs, compiled["obj_xpos"], compiled["obj_xmat"],
      w_devs,
      i_devs, compiled["inst_xpos"], compiled["inst_xmat"], inst_inv_xmats_t,
      compiled["cam_xpos"], compiled["cam_xmat"],
      env_bf16,
  )
  return fn, runtime_args


def make_comparison_figure(tpu_4k_path: str, hyperion_ref_path: str, out_cmp_path: str, stats: Dict[str, Any]):
  """Create crisp 2-panel TPU v5p Native 4K vs Disney Hyperion Reference comparison figure."""
  tpu_im = Image.open(tpu_4k_path).convert("RGB")
  hyp_im = Image.open(hyperion_ref_path).convert("RGB")
  # Center-crop 16:9 TPU render to match Hyperion's 2.387:1 aspect for exact visual framing comparison
  tw, th = tpu_im.size
  hw, hh = hyp_im.size
  crop_h = int(round(tw * (hh / hw)))
  y0 = (th - crop_h) // 2
  tpu_crop = tpu_im.crop((0, y0, tw, y0 + crop_h)).resize((1600, int(round(1600 * hh / hw))), Image.Resampling.LANCZOS)
  hyp_rs = hyp_im.resize(tpu_crop.size, Image.Resampling.LANCZOS)

  pw, ph = tpu_crop.size
  hdr_h = 84
  canvas = Image.new("RGB", (pw * 2 + 24, ph + hdr_h + 16), (14, 18, 26))
  canvas.paste(tpu_crop, (8, hdr_h))
  canvas.paste(hyp_rs, (pw + 16, hdr_h))
  dr = ImageDraw.Draw(canvas)
  dr.text((16, 14), f"TPU v5p (4-Chip) Native 4K OpenUSD Renderer  |  {stats['ms']} ms ({stats['fps']} FPS)  |  host_callbacks=0, gather_ops=0", fill=(240, 245, 255))
  dr.text((16, 42), f"BLAS Tris: {stats['unique_tris'] + stats['inst_blas_tris']:,}  |  Effective Instanced Tris: {stats['inst_effective_tris']:,}  |  HBM: {stats['hbm_gb']:.2f} GiB", fill=(130, 220, 200))
  dr.text((pw + 24, 14), "Disney Animation Official Hyperion Path-Traced Reference (shotCam_hyperion.png)", fill=(240, 245, 255))
  dr.text((pw + 24, 42), f"OpenUSD v2.1 Motunui Island  |  Real islandsun.exr HDRI + Beer-Lambert Coral Lagoon + AgX", fill=(180, 200, 225))
  canvas.save(out_cmp_path)
  print(f"Saved comparison figure to {out_cmp_path} ({canvas.size})", flush=True)


def _save_compiled_cache(cache_dir: str, compiled: Dict[str, Any], compile_host_s: float):
  os.makedirs(cache_dir, exist_ok=True)
  st = compiled["stage"]
  meta = {
      "compile_host_s": compile_host_s,
      "camera_name": st.camera.name,
      "up_axis": st.up_axis,
      "meters_per_unit": st.meters_per_unit,
      "dome_rotate_y_deg": st.dome_rotate_y_deg,
      "unique_meshes": len(st.unique_meshes),
      "water_meshes": len(st.water_meshes),
      "inst_groups": len(st.instanced_groups),
      "n_usd_instances": st.n_usd_instances,
      "n_point_instances": st.n_point_instances,
      "unique_tris": compiled["unique_tris"],
      "inst_blas_tris": compiled["inst_blas_tris"],
      "inst_effective_tris": compiled["inst_effective_tris"],
      "hbm_bytes": compiled["hbm_bytes"],
      "unique_spec": compiled["unique_spec"],
      "unique_pbr": compiled["unique_pbr"],
      "water_spec": compiled["water_spec"],
      "inst_spec": compiled["inst_spec"],
      "inst_pbr": compiled["inst_pbr"],
      "obj_xpos": np.asarray(compiled["obj_xpos"]),
      "obj_xmat": np.asarray(compiled["obj_xmat"]),
      "cam_xpos": np.asarray(compiled["cam_xpos"]),
      "cam_xmat": np.asarray(compiled["cam_xmat"]),
      "unique_dev_arrays": [tuple(np.asarray(x) for x in t) for t in compiled["unique_dev_arrays"]],
      "water_dev_arrays": [tuple(np.asarray(x) for x in t) for t in compiled["water_dev_arrays"]],
      "inst_dev_arrays": [tuple(np.asarray(x) for x in t) for t in compiled["inst_dev_arrays"]],
      "inst_xpos": [np.asarray(x) for x in compiled["inst_xpos"]],
      "inst_xmat": [np.asarray(x) for x in compiled["inst_xmat"]],
  }
  with open(os.path.join(cache_dir, "moana_compiled.pkl"), "wb") as f:
    pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)
  print(f"Saved Moana BVH cache to {cache_dir}/moana_compiled.pkl", flush=True)


def _load_compiled_cache(cache_dir: str, devs: List[jax.Device], height: int, width: int) -> Optional[Dict[str, Any]]:
  p = os.path.join(cache_dir, "moana_compiled.pkl")
  if not os.path.exists(p):
    return None
  t0 = time.perf_counter()
  with open(p, "rb") as f:
    meta = pickle.load(f)
  mesh = Mesh(np.array(devs), ("spatial",))
  rep = NamedSharding(mesh, P())
  put = lambda a: jax.device_put(a, rep)

  @dataclasses.dataclass
  class _CamStub:
    name: str
  @dataclasses.dataclass
  class _StageStub:
    camera: Any
    up_axis: str
    meters_per_unit: float
    dome_rotate_y_deg: float
    unique_meshes: List[int]
    water_meshes: List[int]
    instanced_groups: List[int]
    n_usd_instances: int
    n_point_instances: int

  st_stub = _StageStub(
      camera=_CamStub(meta["camera_name"]),
      up_axis=meta["up_axis"],
      meters_per_unit=meta["meters_per_unit"],
      dome_rotate_y_deg=meta["dome_rotate_y_deg"],
      unique_meshes=[0] * meta["unique_meshes"],
      water_meshes=[0] * meta["water_meshes"],
      instanced_groups=[0] * meta["inst_groups"],
      n_usd_instances=meta["n_usd_instances"],
      n_point_instances=meta["n_point_instances"],
  )
  u_spec = dataclasses.replace(meta["unique_spec"], height=height, width=width) if meta["unique_spec"] is not None else None
  w_spec = dataclasses.replace(meta["water_spec"], height=height, width=width) if meta["water_spec"] is not None else None
  i_spec = dataclasses.replace(meta["inst_spec"], height=height, width=width) if meta["inst_spec"] is not None else None
  out = {
      "stage": st_stub,
      "compile_host_s": meta["compile_host_s"],
      "unique_spec": u_spec,
      "unique_dev_arrays": tuple(tuple(put(x) for x in t) for t in meta["unique_dev_arrays"]),
      "unique_pbr": meta["unique_pbr"],
      "water_spec": w_spec,
      "water_dev_arrays": tuple(tuple(put(x) for x in t) for t in meta["water_dev_arrays"]),
      "inst_spec": i_spec,
      "inst_dev_arrays": tuple(tuple(put(x) for x in t) for t in meta["inst_dev_arrays"]),
      "inst_pbr": meta["inst_pbr"],
      "inst_xpos": tuple(put(x) for x in meta["inst_xpos"]),
      "inst_xmat": tuple(put(x) for x in meta["inst_xmat"]),
      "obj_xpos": put(meta["obj_xpos"]),
      "obj_xmat": put(meta["obj_xmat"]),
      "cam_xpos": put(meta["cam_xpos"]),
      "cam_xmat": put(meta["cam_xmat"]),
      "hbm_bytes": meta["hbm_bytes"],
      "unique_tris": meta["unique_tris"],
      "inst_blas_tris": meta["inst_blas_tris"],
      "inst_effective_tris": meta["inst_effective_tris"],
  }
  print(f"Loaded Moana BVH cache in {time.perf_counter()-t0:.2f}s!", flush=True)
  return out


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--stage", default="/dev/shm/zaychik-moana/island/usd/island.usda")
  ap.add_argument("--exr", default="/dev/shm/zaychik-moana/island/textures/islandsun.exr")
  ap.add_argument("--hyperion-ref", default="/dev/shm/zaychik-moana/island/ref/hyperionRenders/shotCam_hyperion.png")
  ap.add_argument("--height", type=int, default=2160)
  ap.add_argument("--width", type=int, default=3840)
  ap.add_argument("--out-png", default="/dev/shm/zaychik-tpu-mjx-w2/v3_w2_moana_usd_4k.png")
  ap.add_argument("--out-cmp", default="/dev/shm/zaychik-tpu-mjx-w2/v3_w2_moana_vs_hyperion.png")
  ap.add_argument("--out-json", default="/dev/shm/zaychik-tpu-mjx-w2/v3_w2_moana_metrics.json")
  ap.add_argument("--cache-dir", default="/dev/shm/zaychik-tpu-mjx-w2/moana_bvh_cache")
  ap.add_argument("--no-shadows", action="store_true")
  ap.add_argument("--min-diag", type=float, default=0.14)
  ap.add_argument("--rebuild-cache", action="store_true")
  args = ap.parse_args()

  devs = jax.devices()
  assert len(devs) == 4, f"Expected 4 TPU chips, got {len(devs)}"

  compiled = None if args.rebuild_cache else _load_compiled_cache(args.cache_dir, devs, args.height, args.width)
  if compiled is None:
    t0 = time.perf_counter()
    compiled = compile_openusd_to_wide_bvh(
        args.stage,
        height=args.height,
        width=args.width,
        build_method="spatial_l1_morton",
        use_shadows=not args.no_shadows,
        devices=devs,
        max_unique_bvh_meshes=18,
        preferred_camera="shotCam",
        include_curves=True,
        max_curve_tris=8_000_000,
        max_variants_per_pi=3,
        min_instance_diag_m=args.min_diag,
        separate_water=True,
    )
    compile_host_s = time.perf_counter() - t0
    _save_compiled_cache(args.cache_dir, compiled, compile_host_s)
  else:
    compile_host_s = compiled["compile_host_s"]
  st = compiled["stage"]
  print(
      f"Compiled Moana OpenUSD in {compile_host_s:.2f}s | unique_tris={compiled['unique_tris']:,} "
      f"| inst_groups={len(st.instanced_groups)} (blas_tris={compiled['inst_blas_tris']:,}, "
      f"eff_tris={compiled['inst_effective_tris']:,}) | HBM={compiled['hbm_bytes']/(1024**3):.2f} GiB",
      flush=True,
  )

  mesh = Mesh(np.array(devs), ("spatial",))
  rep = NamedSharding(mesh, P())
  env_np = load_islandsun_hdri_bf16(args.exr, H_env=512, W_env=1024)
  env_bf16 = jax.device_put(jnp.asarray(env_np, dtype=jnp.bfloat16), rep)

  fn_render, r_args = make_moana_4chip_renderer(
      compiled, env_bf16, devs, args.height, args.width,
      dome_rot_z_deg=st.dome_rotate_y_deg, use_shadows=not args.no_shadows,
  )

  htxt = fn_render.lower(*r_args).as_text()
  hlo = inspect_hlo_ops(htxt)
  hlo["send_recv_ops"] = len(re.findall(r"\bsend\(|\brecv\(", htxt, re.I))
  print("HLO check:", hlo, flush=True)
  assert hlo["host_callbacks"] == 0 and hlo["send_recv_ops"] == 0 and hlo["gather_ops"] == 0, hlo

  tc0 = time.perf_counter()
  rgb = fn_render(*r_args)
  jax.block_until_ready(rgb)
  xla_compile_s = time.perf_counter() - tc0

  img = Image.fromarray(np.clip(np.asarray(rgb[0]) * 255.0 + 0.5, 0, 255).astype(np.uint8))
  os.makedirs(os.path.dirname(args.out_png), exist_ok=True)
  img.save(args.out_png)
  print(f"Saved Moana render ({img.size}) to {args.out_png}", flush=True)

  n_steps = 3
  t1 = time.perf_counter()
  for _ in range(n_steps):
    rgb = fn_render(*r_args)
  jax.block_until_ready(rgb)
  ms = (time.perf_counter() - t1) * 1e3 / n_steps

  stats = {
      "name": "Disney_Moana_Island_OpenUSD_v2.1",
      "stage_path": args.stage,
      "camera": st.camera.name,
      "up_axis": st.up_axis,
      "meters_per_unit": st.meters_per_unit,
      "unique_meshes": len(st.unique_meshes),
      "water_meshes": len(st.water_meshes),
      "unique_tris": compiled["unique_tris"],
      "inst_groups": len(st.instanced_groups),
      "inst_blas_tris": compiled["inst_blas_tris"],
      "inst_effective_tris": compiled["inst_effective_tris"],
      "n_usd_instances": st.n_usd_instances,
      "n_point_instances": st.n_point_instances,
      "hbm_mb": round(compiled["hbm_bytes"] / (1024 ** 2), 2),
      "hbm_gb": round(compiled["hbm_bytes"] / (1024 ** 3), 3),
      "usd_to_bvh_host_s": round(compile_host_s, 2),
      "xla_compile_s": round(xla_compile_s, 2),
      "res": f"{args.height}x{args.width}",
      "n_chips": len(devs),
      "ms": round(ms, 2),
      "fps": round(1e3 / ms, 2),
      "mrays_per_s": round((args.height * args.width * 2.0) / (ms * 1e-3) / 1e6, 2),
      "hlo": hlo,
      "png_4k": args.out_png,
  }
  with open(args.out_json, "w") as f:
    json.dump(stats, f, indent=2)

  if os.path.exists(args.hyperion_ref) and args.out_cmp:
    make_comparison_figure(args.out_png, args.hyperion_ref, args.out_cmp, stats)

  print(f"DONE Moana {args.height}x{args.width}: {ms:.2f} ms ({stats['fps']} FPS), HBM={stats['hbm_gb']:.2f} GiB", flush=True)


if __name__ == "__main__":
  main()
