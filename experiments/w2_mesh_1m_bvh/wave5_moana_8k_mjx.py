"""Wave 5 Polish + Native 8K + Physical AI Crossover:
1. Cinema Lighting & Contrast Polish matching Disney Hyperion on Moana OpenUSD v2.1 (22.30B instanced tris).
2. 4-Chip Native 8K (7680x4320, 33.18M px) Benchmark & 4-Angle Moana Cinema Postcard Grid:
   - shotCam polished vs Hyperion
   - beachCam palm canopy & sunlit shore
   - dunesACam / palmsCam volcanic ridge
   - 1:1 Native 8K palm-frond & coral detail crop
3. Physical AI Crossover: Articulated MJX Quadruped-12DOF + Humanoid-21DOF on Motunui Beach @ Native 4K
   with exact Z-depth occlusion and two-way tropical sunlight cast shadows.

Verified interpret=False, host_callbacks == 0, gather_ops == 0, send_recv_ops == 0.
"""

from __future__ import annotations

import argparse
import dataclasses
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
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tpu_mjx_render.primitives import (
    GEOM_BOX,
    GEOM_CAPSULE,
    GEOM_CYLINDER,
    GEOM_SPHERE,
    MJ_MAXVAL,
    MJ_MINVAL,
)
from tpu_mjx_render.roofline import inspect_hlo_ops
from wave3_500m_10b_8k import (
    CompactWideBVHMeshItem,
    _bundle_aabb_tmin,
    _pallas_compact_bvh_raycast,
    _traverse_compact_bvh_in_kernel,
)
from wide_bvh_v3 import BLOCK_R, BRANCH_FACTOR, TILE_H, TILE_W, _pack_rays_tiled, _unpack_out_tiled
from wave5_moana_4k import (
    TAG_TO_ID,
    _load_compiled_cache,
    _save_compiled_cache,
    agx_filmic_tonemap,
    pallas_sample_hdri_env,
)


def load_islandsun_hdri_polished(exr_or_png_path: str, H_env: int = 512, W_env: int = 1024) -> np.ndarray:
  """Load Disney islandsunVIS.png (or islandsun.exr if OpenEXR available) with crisp azure sky & white clouds."""
  png_fallback = os.path.join(os.path.dirname(exr_or_png_path), "islandsunVIS.png")
  rgb_small = None
  try:
    import OpenEXR
    import Imath
    exr = OpenEXR.InputFile(exr_or_png_path)
    dw = exr.header()["dataWindow"]
    W_full = dw.max.x - dw.min.x + 1
    H_full = dw.max.y - dw.min.y + 1
    pt = Imath.PixelType(Imath.PixelType.HALF)
    r = np.frombuffer(exr.channel("R", pt), dtype=np.float16).reshape(H_full, W_full)
    g = np.frombuffer(exr.channel("G", pt), dtype=np.float16).reshape(H_full, W_full)
    b = np.frombuffer(exr.channel("B", pt), dtype=np.float16).reshape(H_full, W_full)
    sy = max(1, H_full // H_env)
    sx = max(1, W_full // W_env)
    hc = (H_full // sy) * sy
    wc = (W_full // sx) * sx
    rgb_small = np.stack([
        r[:hc, :wc].astype(np.float32).reshape(H_full // sy, sy, W_full // sx, sx).mean(axis=(1, 3)),
        g[:hc, :wc].astype(np.float32).reshape(H_full // sy, sy, W_full // sx, sx).mean(axis=(1, 3)),
        b[:hc, :wc].astype(np.float32).reshape(H_full // sy, sy, W_full // sx, sx).mean(axis=(1, 3)),
    ], axis=-1)
    if rgb_small.shape[0] != H_env or rgb_small.shape[1] != W_env:
      im = Image.fromarray(np.clip(rgb_small * 255.0 + 0.5, 0, 255).astype(np.uint8))
      im = im.resize((W_env, H_env), Image.Resampling.BILINEAR)
      rgb_small = np.asarray(im, dtype=np.float32) / 255.0
    rgb_lin = np.power(np.clip(rgb_small, 0.0, 1.0), 1.38) * 1.48
  except Exception:
    src = png_fallback if os.path.exists(png_fallback) else exr_or_png_path
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(src) as im:
      im_rs = im.convert("RGB").resize((W_env, H_env), Image.Resampling.BILINEAR)
      rgb_srgb = np.asarray(im_rs, dtype=np.float32) / 255.0
    rgb_lin = np.power(np.clip(rgb_srgb, 0.0, 1.0), 2.1) * 1.62

  # Crisp tropical azure sky & clean bright white clouds
  rgb_lin[..., 0] *= 0.90
  rgb_lin[..., 1] *= 1.00
  rgb_lin[..., 2] *= 1.14
  lum = 0.2126 * rgb_lin[..., 0] + 0.7152 * rgb_lin[..., 1] + 0.0722 * rgb_lin[..., 2]
  cloud_boost = 1.0 + 0.85 * np.maximum(lum - 0.62, 0.0)
  rgb_lin = rgb_lin * cloud_boost[..., None]
  return np.concatenate([rgb_lin[..., 0], rgb_lin[..., 1], rgb_lin[..., 2]], axis=1).astype(np.float32)


def extract_all_moana_cameras(stage_path: str, target_aspect: float) -> Dict[str, Dict[str, Any]]:
  """Extract all UsdGeom.Camera poses & intrinsics from Moana USD stage in Z-up meters."""
  from pxr import Usd, UsdGeom

  stage = Usd.Stage.Open(stage_path)
  tc = Usd.TimeCode.Default()
  up_axis = str(UsdGeom.GetStageUpAxis(stage) or "Y")
  mpu = float(UsdGeom.GetStageMetersPerUnit(stage) or 1.0)
  y_to_z = up_axis.upper() == "Y"
  xcache = UsdGeom.XformCache(tc)

  def _y2z(v: np.ndarray) -> np.ndarray:
    if not y_to_z:
      return v.astype(np.float32)
    return np.array([v[0], -v[2], v[1]], dtype=np.float32)

  cams: Dict[str, Dict[str, Any]] = {}
  for prim in stage.Traverse():
    if not prim.IsA(UsdGeom.Camera):
      continue
    cam = UsdGeom.Camera(prim)
    M = np.asarray(xcache.GetLocalToWorldTransform(prim), dtype=np.float32)
    pos = _y2z(M[3, :3] * mpu)
    right = _y2z(M[0, :3])
    up = _y2z(M[1, :3])
    back = _y2z(M[2, :3])
    R_cam = np.stack([right, up, back], axis=1).astype(np.float32)
    for c in range(3):
      R_cam[:, c] /= max(float(np.linalg.norm(R_cam[:, c])), 1e-8)
    fl = float(cam.GetFocalLengthAttr().Get(tc) or 35.0)
    va = float(cam.GetVerticalApertureAttr().Get(tc) or 20.25)
    ha = float(cam.GetHorizontalApertureAttr().Get(tc) or (va * target_aspect))
    eff_va = ha / float(target_aspect)
    tan_half_fovy = float(0.5 * eff_va / max(fl, 1e-4))
    fovy_deg = 2.0 * math.degrees(math.atan(tan_half_fovy))
    short_name = prim.GetName()
    cams[short_name] = {
        "path": str(prim.GetPath()),
        "cam_xpos": pos[None, :].astype(np.float32),
        "cam_xmat": R_cam[None, :, :].astype(np.float32),
        "tan_half_fovy": np.array([tan_half_fovy], dtype=np.float32),
        "fovy_deg": fovy_deg,
    }
  return cams


def _compute_camera_rays_subband(
    cam_pos: jax.Array,       # (B, 3)
    cam_mat: jax.Array,       # (B, 3, 3)
    tan_half_fovy: jax.Array, # (1,)
    znear: float,
    H_full: int,
    W_full: int,
    band_idx: jax.Array,
    band_h: int,
    sb_idx: jax.Array,
    sub_h: int,
    sub_h_pad: int,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
  """Compute (B, sub_h_pad * W_full, 3) camera rays for spatial chip + scan subband."""
  aspect = float(W_full) / float(H_full)
  half_h = jnp.asarray(znear, dtype=jnp.float32) * tan_half_fovy[0]
  half_w = half_h * aspect
  u = (jnp.arange(W_full, dtype=jnp.float32) + 0.5) / float(W_full)
  y_px = (
      band_idx.astype(jnp.float32) * float(band_h)
      + sb_idx.astype(jnp.float32) * float(sub_h)
      + jnp.arange(sub_h_pad, dtype=jnp.float32)
      + 0.5
  )
  v = jnp.minimum(y_px / float(H_full), 1.0)
  gx, gy = jnp.meshgrid(u, v, indexing="xy")
  x = -half_w + (2.0 * half_w) * gx
  y = half_h - (2.0 * half_h) * gy
  raw = jnp.stack([x, y, jnp.full_like(x, -float(znear))], axis=-1).reshape(-1, 3)
  ld = raw / jnp.maximum(jnp.linalg.norm(raw, axis=-1, keepdims=True), 1e-12)
  neg_z = jnp.maximum(-ld[:, 2], 1e-6)
  ray_dir = jnp.einsum("bij,rj->bri", cam_mat, ld, precision=jax.lax.Precision.HIGHEST)
  ray_dir = ray_dir / jnp.maximum(jnp.linalg.norm(ray_dir, axis=-1, keepdims=True), 1e-12)
  inv_cos = (float(znear) / neg_z)[None, :, None]
  ray_orig = cam_pos[:, None, :] + inv_cos * ray_dir
  return ray_orig, ray_dir, neg_z


def _precompute_tlas_group(
    item: CompactWideBVHMeshItem,
    inst_xpos: jax.Array,
    inst_xmat: jax.Array,
    inst_inv_xmat: jax.Array,
    tlas_group_size: int = 64,
):
  """Precompute 3-Level Wide-TLAS-64 AABBs & packed affine M_inv poses once per frame."""
  B, K_inst, _ = inst_xpos.shape
  K_inst = int(K_inst)
  c_loc = jnp.asarray(0.5 * (item.obj_aabb_min + item.obj_aabb_max), dtype=jnp.float32)
  h_loc = jnp.asarray(0.5 * (item.obj_aabb_max - item.obj_aabb_min), dtype=jnp.float32)
  c_w = inst_xpos + jnp.einsum("bkij,j->bki", inst_xmat, c_loc, precision=jax.lax.Precision.HIGHEST)
  h_w = jnp.einsum("bkij,j->bki", jnp.abs(inst_xmat), h_loc, precision=jax.lax.Precision.HIGHEST)
  w_min = c_w - h_w - 1e-3
  w_max = c_w + h_w + 1e-3

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

  if K_inst <= 64:
    return (w_min, w_max, None, None, None, None, None, None, poses_hbm, n_pose_chunks, K_inst)

  G = max(1, (K_inst + tlas_group_size - 1) // tlas_group_size)
  K_pad_g = G * tlas_group_size
  pad_g = K_pad_g - K_inst
  w_min_p = jnp.pad(w_min, ((0, 0), (0, pad_g), (0, 0)), constant_values=1e20) if pad_g > 0 else w_min
  w_max_p = jnp.pad(w_max, ((0, 0), (0, pad_g), (0, 0)), constant_values=-1e20) if pad_g > 0 else w_max
  g_min = jnp.min(w_min_p.reshape(B, G, tlas_group_size, 3), axis=2)
  g_max = jnp.max(w_max_p.reshape(B, G, tlas_group_size, 3), axis=2)

  if G > 256:
    SG = max(1, (G + 63) // 64)
    G_pad = SG * 64
    pad_sg = G_pad - G
    g_min_p = jnp.pad(g_min, ((0, 0), (0, pad_sg), (0, 0)), constant_values=1e20) if pad_sg > 0 else g_min
    g_max_p = jnp.pad(g_max, ((0, 0), (0, pad_sg), (0, 0)), constant_values=-1e20) if pad_sg > 0 else g_max
    if pad_sg > 0:
      w_min_p = jnp.pad(w_min_p, ((0, 0), (0, pad_sg * tlas_group_size), (0, 0)), constant_values=1e20)
      w_max_p = jnp.pad(w_max_p, ((0, 0), (0, pad_sg * tlas_group_size), (0, 0)), constant_values=-1e20)
    sg_min = jnp.min(g_min_p.reshape(B, SG, 64, 3), axis=2)
    sg_max = jnp.max(g_max_p.reshape(B, SG, 64, 3), axis=2)
    return (w_min_p, w_max_p, g_min_p, g_max_p, sg_min, sg_max, G, SG, poses_hbm, n_pose_chunks, K_inst)
  return (w_min_p, w_max_p, g_min, g_max, None, None, G, 0, poses_hbm, n_pose_chunks, K_inst)


def _pallas_tlas_precomp_raycast(
    tlas_precomp: Tuple[Any, ...],
    l2_roots_dev: jax.Array,
    l2_dev: jax.Array,
    l0_dev: jax.Array,
    leaf_dev: jax.Array,
    ray_o_world: jax.Array,
    ray_d_world: jax.Array,
    t_limit: jax.Array,
    H_sub: int,
    W: int,
    tlas_group_size: int = 64,
    tlas_top_groups: int = 4,
    max_active_per_tile: int = 16,
    any_hit: bool = False,
) -> Tuple[jax.Array, jax.Array]:
  """Trace subband rays using precomputed TLAS AABBs & packed M_inv poses."""
  w_min_p, w_max_p, g_min_p, g_max_p, sg_min, sg_max, G, SG, poses_hbm, n_pose_chunks, K_inst = tlas_precomp
  B, _, _ = ray_o_world.shape
  rays_2d, n_tiles, can_2d_tile = _pack_rays_tiled(ray_o_world, ray_d_world, t_limit, H_sub, W)
  total_tiles = B * n_tiles

  ro_t = rays_2d.reshape(B, n_tiles, 8, BLOCK_R)[:, :, 0:3, :].transpose(0, 1, 3, 2)
  rd_t = rays_2d.reshape(B, n_tiles, 8, BLOCK_R)[:, :, 3:6, :].transpose(0, 1, 3, 2)
  tl_t = rays_2d.reshape(B, n_tiles, 8, BLOCK_R)[:, :, 6, :]
  ro_min, ro_max = jnp.min(ro_t, axis=2), jnp.max(ro_t, axis=2)
  rd_min, rd_max = jnp.min(rd_t, axis=2), jnp.max(rd_t, axis=2)
  max_tl = jnp.max(tl_t, axis=2)

  if K_inst <= 64:
    cand_w_min, cand_w_max = w_min_p, w_max_p
    base_ids = jnp.broadcast_to(jnp.arange(K_inst, dtype=jnp.int32)[None, None, :], (B, n_tiles, K_inst))
    cand_valid = jnp.ones((B, n_tiles, K_inst), dtype=jnp.bool_)
  else:
    if SG and SG > 0:
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
      # Two-step SG -> group -> leaf-instance lookup (eliminates O(n_tiles * G) 41GB tensor at 20M instances!)
      w_min_sg = jnp.einsum("btmg,bgjkc->btmjkc", sel_sg, w_min_p.reshape(B, SG, 64, tlas_group_size, 3), precision=jax.lax.Precision.HIGHEST).reshape(B, n_tiles, top_sg_k * 64, tlas_group_size, 3)
      w_max_sg = jnp.einsum("btmg,bgjkc->btmjkc", sel_sg, w_max_p.reshape(B, SG, 64, tlas_group_size, 3), precision=jax.lax.Precision.HIGHEST).reshape(B, n_tiles, top_sg_k * 64, tlas_group_size, 3)
      cand_w_min = jnp.einsum("btkc,btcsd->btksd", sel_gl, w_min_sg, precision=jax.lax.Precision.HIGHEST).reshape(B, n_tiles, top_g_k * tlas_group_size, 3)
      cand_w_max = jnp.einsum("btkc,btcsd->btksd", sel_gl, w_max_sg, precision=jax.lax.Precision.HIGHEST).reshape(B, n_tiles, top_g_k * tlas_group_size, 3)
    else:
      top_g_k = min(tlas_top_groups, G)
      ok_g, tmin_g = _bundle_aabb_tmin(
          ro_min[:, :, None, :], ro_max[:, :, None, :],
          rd_min[:, :, None, :], rd_max[:, :, None, :],
          max_tl[:, :, None], g_min_p[:, None, :, :], g_max_p[:, None, :, :],
      )
      score_g = jnp.where(ok_g, tmin_g, 1e20)
      neg_g, top_g = jax.lax.top_k(-score_g, top_g_k)
      sel_g = (top_g[:, :, :, None] == jnp.arange(G, dtype=top_g.dtype)[None, None, None, :]).astype(jnp.float32)
      cand_w_min = jnp.einsum("btmg,bgsc->btmsc", sel_g, w_min_p.reshape(B, G, tlas_group_size, 3), precision=jax.lax.Precision.HIGHEST).reshape(B, n_tiles, top_g_k * tlas_group_size, 3)
      cand_w_max = jnp.einsum("btmg,bgsc->btmsc", sel_g, w_max_p.reshape(B, G, tlas_group_size, 3), precision=jax.lax.Precision.HIGHEST).reshape(B, n_tiles, top_g_k * tlas_group_size, 3)

    base_ids = (top_g[:, :, :, None] * tlas_group_size + jnp.arange(tlas_group_size)[None, None, None, :]).reshape(B, n_tiles, top_g_k * tlas_group_size)
    cand_valid = (neg_g[:, :, :, None] > -1e19).repeat(tlas_group_size, axis=-1).reshape(B, n_tiles, top_g_k * tlas_group_size) & (base_ids < K_inst)

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

  d_out, n_out, _ = _unpack_out_tiled(out_2d, B, H_sub, W, n_tiles, can_2d_tile, return_stats=False)
  return d_out, n_out


# ---------------------------------------------------------------------------
# MJX Dual-Embodiment (Quadruped-12DOF + Humanoid-21DOF) + Pallas Raycaster
# ---------------------------------------------------------------------------

def _quad_leg_xml(name: str, x: float, y: float) -> str:
  sy = 1.0 if y > 0 else -1.0
  r45 = math.radians(45.0)
  r90 = math.radians(90.0)
  r120 = math.radians(120.0)
  r150 = math.radians(150.0)
  r20 = math.radians(20.0)
  return f"""
    <body name="{name}_hip" pos="{x:.3f} {y:.3f} 0">
      <joint name="{name}_hip_x" type="hinge" axis="1 0 0" range="-{r45:.4f} {r45:.4f}"/>
      <geom name="{name}_hip_pod" type="cylinder" pos="0 {sy*0.03:.3f} 0" quat="0.7071 0.7071 0 0" size="0.044 0.045" material="quad_dark" mass="0.6"/>
      <body name="{name}_thigh" pos="0 {sy*0.085:.3f} 0">
        <joint name="{name}_thigh_y" type="hinge" axis="0 1 0" range="-{r90:.4f} {r120:.4f}"/>
        <geom name="{name}_thigh_g" type="capsule" fromto="0 0 0 0 0 -0.21" size="0.032" material="quad_titan" mass="1.0"/>
        <geom name="{name}_knee_pod" type="sphere" pos="0 0 -0.21" size="0.034" material="quad_gold" mass="0.2"/>
        <body name="{name}_calf" pos="0 0 -0.21">
          <joint name="{name}_calf_y" type="hinge" axis="0 1 0" range="-{r150:.4f} -{r20:.4f}"/>
          <geom name="{name}_calf_g" type="capsule" fromto="0 0 0 0 0 -0.21" size="0.022" material="quad_dark" mass="0.5"/>
          <geom name="{name}_foot" type="sphere" pos="0 0 -0.215" size="0.028" material="quad_rubber" mass="0.2"/>
        </body>
      </body>
    </body>"""


def build_motunui_mjx_xml() -> str:
  """Combined MJCF with Humanoid-21DOF + Quadruped-12DOF in radian units."""
  legs = (
      _quad_leg_xml("FL", 0.21, 0.095)
      + _quad_leg_xml("FR", 0.21, -0.095)
      + _quad_leg_xml("RL", -0.21, 0.095)
      + _quad_leg_xml("RR", -0.21, -0.095)
  )
  return f"""<mujoco model="motunui_quadruped_humanoid">
  <compiler angle="radian" autolimits="true"/>
  <asset>
    <material name="hum_titan" rgba="0.80 0.84 0.89 1" specular="0.65" shininess="0.55"/>
    <material name="hum_orange" rgba="0.94 0.39 0.08 1" specular="0.50" shininess="0.45"/>
    <material name="hum_dark" rgba="0.12 0.14 0.18 1" specular="0.55" shininess="0.48"/>
    <material name="hum_visor" rgba="0.06 0.64 0.92 1" specular="0.85" shininess="0.80"/>
    <material name="quad_gold" rgba="0.92 0.56 0.10 1" specular="0.60" shininess="0.55"/>
    <material name="quad_titan" rgba="0.82 0.86 0.90 1" specular="0.65" shininess="0.55"/>
    <material name="quad_dark" rgba="0.12 0.14 0.18 1" specular="0.55" shininess="0.48"/>
    <material name="quad_rubber" rgba="0.07 0.08 0.10 1" specular="0.20" shininess="0.20"/>
    <material name="quad_visor" rgba="0.10 0.75 0.96 1" specular="0.85" shininess="0.80"/>
  </asset>
  <default>
    <joint armature="0.02" damping="2.0" limited="true"/>
    <geom contype="0" conaffinity="0"/>
  </default>
  <worldbody>
    <!-- Humanoid-21DOF -->
    <body name="hum_torso" pos="0.0 0.0 1.20">
      <freejoint name="hum_root"/>
      <geom name="hum_torso_g" type="capsule" fromto="0 -0.085 0 0 0.085 0" size="0.105" material="hum_orange" mass="5.0"/>
      <geom name="hum_chest_plate" type="box" pos="0.04 0 0.02" size="0.075 0.095 0.08" material="hum_titan" mass="1.0"/>
      <geom name="hum_head" type="sphere" pos="0.02 0 0.22" size="0.11" material="hum_titan" mass="1.5"/>
      <geom name="hum_visor" type="box" pos="0.105 0 0.225" size="0.034 0.072 0.032" material="hum_visor" mass="0.1"/>
      <geom name="hum_uwaist" type="capsule" fromto="-0.01 -0.065 -0.12 -0.01 0.065 -0.12" size="0.078" material="hum_dark" mass="2.0"/>
      <body name="hum_lwaist" pos="-0.01 0 -0.25">
        <joint name="hum_abdomen_z" type="hinge" axis="0 0 1" range="-0.8 0.8"/>
        <joint name="hum_abdomen_y" type="hinge" axis="0 1 0" range="-1.2 0.5"/>
        <geom name="hum_lwaist_g" type="capsule" fromto="0 -0.065 0 0 0.065 0" size="0.074" material="hum_orange" mass="2.0"/>
        <body name="hum_pelvis" pos="0 0 -0.165">
          <joint name="hum_abdomen_x" type="hinge" axis="1 0 0" range="-0.6 0.6"/>
          <geom name="hum_butt" type="capsule" fromto="-0.02 -0.085 0 -0.02 0.085 0" size="0.082" material="hum_dark" mass="3.0"/>
          <body name="hum_rthigh" pos="0 -0.105 -0.04">
            <joint name="hum_rhip_x" type="hinge" axis="1 0 0" range="-0.5 0.2"/>
            <joint name="hum_rhip_z" type="hinge" axis="0 0 1" range="-1.0 0.6"/>
            <joint name="hum_rhip_y" type="hinge" axis="0 1 0" range="-1.8 0.8"/>
            <geom name="hum_rthigh_g" type="capsule" fromto="0 0 0 0 0.01 -0.35" size="0.064" material="hum_titan" mass="3.0"/>
            <body name="hum_rshin" pos="0 0.01 -0.40">
              <joint name="hum_rknee" type="hinge" axis="0 -1 0" range="-2.6 0.0"/>
              <geom name="hum_rshin_g" type="capsule" fromto="0 0 0 0 0 -0.32" size="0.052" material="hum_orange" mass="2.0"/>
              <body name="hum_rfoot" pos="0 0 -0.36">
                <joint name="hum_rankle_x" type="hinge" axis="1 0 0" range="-0.8 0.8"/>
                <joint name="hum_rankle_y" type="hinge" axis="0 1 0" range="-0.9 0.9"/>
                <geom name="hum_rfoot_g" type="box" pos="0.045 0 -0.022" size="0.105 0.050 0.026" material="hum_dark" mass="1.0"/>
              </body>
            </body>
          </body>
          <body name="hum_lthigh" pos="0 0.105 -0.04">
            <joint name="hum_lhip_x" type="hinge" axis="-1 0 0" range="-0.5 0.2"/>
            <joint name="hum_lhip_z" type="hinge" axis="0 0 -1" range="-1.0 0.6"/>
            <joint name="hum_lhip_y" type="hinge" axis="0 1 0" range="-1.8 0.8"/>
            <geom name="hum_lthigh_g" type="capsule" fromto="0 0 0 0 -0.01 -0.35" size="0.064" material="hum_titan" mass="3.0"/>
            <body name="hum_lshin" pos="0 -0.01 -0.40">
              <joint name="hum_lknee" type="hinge" axis="0 -1 0" range="-2.6 0.0"/>
              <geom name="hum_lshin_g" type="capsule" fromto="0 0 0 0 0 -0.32" size="0.052" material="hum_orange" mass="2.0"/>
              <body name="hum_lfoot" pos="0 0 -0.36">
                <joint name="hum_lankle_x" type="hinge" axis="1 0 0" range="-0.8 0.8"/>
                <joint name="hum_lankle_y" type="hinge" axis="0 1 0" range="-0.9 0.9"/>
                <geom name="hum_lfoot_g" type="box" pos="0.045 0 -0.022" size="0.105 0.050 0.026" material="hum_dark" mass="1.0"/>
              </body>
            </body>
          </body>
        </body>
      </body>
      <body name="hum_ruarm" pos="0 -0.18 0.06">
        <joint name="hum_rsh1" type="hinge" axis="0 1 0" range="-2.0 1.5"/>
        <joint name="hum_rsh2" type="hinge" axis="1 0 0" range="-1.5 1.5"/>
        <geom name="hum_ruarm_g" type="capsule" fromto="0 0 0 0.02 -0.04 -0.26" size="0.046" material="hum_titan" mass="1.2"/>
        <body name="hum_rlarm" pos="0.02 -0.04 -0.28">
          <joint name="hum_relbow" type="hinge" axis="0 1 0" range="-2.2 0.2"/>
          <geom name="hum_rlarm_g" type="capsule" fromto="0 0 0 0.22 0.02 -0.04" size="0.038" material="hum_orange" mass="0.8"/>
          <geom name="hum_rhand" type="sphere" pos="0.24 0.02 -0.04" size="0.046" material="hum_dark" mass="0.3"/>
        </body>
      </body>
      <body name="hum_luarm" pos="0 0.18 0.06">
        <joint name="hum_lsh1" type="hinge" axis="0 1 0" range="-2.0 1.5"/>
        <joint name="hum_lsh2" type="hinge" axis="-1 0 0" range="-1.5 1.5"/>
        <geom name="hum_luarm_g" type="capsule" fromto="0 0 0 0.02 0.04 -0.26" size="0.046" material="hum_titan" mass="1.2"/>
        <body name="hum_llarm" pos="0.02 0.04 -0.28">
          <joint name="hum_lelbow" type="hinge" axis="0 1 0" range="-2.2 0.2"/>
          <geom name="hum_llarm_g" type="capsule" fromto="0 0 0 0.22 -0.02 -0.04" size="0.038" material="hum_orange" mass="0.8"/>
          <geom name="hum_lhand" type="sphere" pos="0.24 -0.02 -0.04" size="0.046" material="hum_dark" mass="0.3"/>
        </body>
      </body>
    </body>

    <!-- Quadruped-12DOF -->
    <body name="quad_trunk" pos="1.0 0.0 0.36">
      <freejoint name="quad_root"/>
      <geom name="chassis_main" type="box" pos="0 0 0" size="0.24 0.105 0.055" material="quad_titan" mass="6.0"/>
      <geom name="chassis_spine" type="box" pos="0 0 0.052" size="0.18 0.068 0.016" material="quad_gold" mass="1.0"/>
      <geom name="head_pod" type="box" pos="0.24 0 0.012" size="0.055 0.078 0.042" material="quad_dark" mass="0.8"/>
      <geom name="lidar_dome" type="cylinder" pos="0.21 0 0.068" size="0.036 0.018" material="quad_dark" mass="0.2"/>
      <geom name="visor_led" type="box" pos="0.295 0 0.012" size="0.008 0.055 0.016" material="quad_visor" mass="0.05"/>
      <geom name="tail_bumper" type="capsule" fromto="-0.24 -0.07 0 -0.24 0.07 0" size="0.035" material="quad_dark" mass="0.3"/>
      {legs}
    </body>
  </worldbody>
</mujoco>"""


def query_beach_sand_height_and_normal(
    stage_path: str,
    xy_points: Sequence[Tuple[float, float]],
) -> List[Tuple[float, np.ndarray]]:
  """Ray-query exact Z height & surface normal of Moana isBeach USD mesh at given (x, y)."""
  from pxr import Usd, UsdGeom
  from tpu_mjx_render.adapters.openusd_loader import _triangulate_faces, _y_up_to_z_up_points, _transform_points_gf

  beach_usd = os.path.join(os.path.dirname(os.path.abspath(stage_path)), "elements", "isBeach", "element.usda")
  st_b = Usd.Stage.Open(beach_usd)
  tc = Usd.TimeCode.Default()
  mpu = float(UsdGeom.GetStageMetersPerUnit(st_b) or 0.0254)
  xcache = UsdGeom.XformCache(tc)
  y_to_z = str(UsdGeom.GetStageUpAxis(st_b) or "Y").upper() == "Y"

  sand_tris: List[Tuple[np.ndarray, np.ndarray]] = []
  # Collect meshes from prototype or direct children
  roots = [st_b.GetPseudoRoot()] + list(st_b.GetPrototypes())
  for root in roots:
    for prim in Usd.PrimRange(root):
      if not prim.IsA(UsdGeom.Mesh):
        continue
      um = UsdGeom.Mesh(prim)
      pts = um.GetPointsAttr().Get(tc)
      cnt = um.GetFaceVertexCountsAttr().Get(tc)
      idx = um.GetFaceVertexIndicesAttr().Get(tc)
      if pts is None or cnt is None or idx is None or len(pts) == 0:
        continue
      v = np.asarray(pts, dtype=np.float32) * mpu
      M = np.asarray(xcache.GetLocalToWorldTransform(prim), dtype=np.float64)
      M[3, :3] *= mpu
      v = _transform_points_gf(v, M)
      if y_to_z:
        v = _y_up_to_z_up_points(v)
      f = _triangulate_faces(np.asarray(cnt, dtype=np.int32), np.asarray(idx, dtype=np.int32))
      if f.shape[0] > 0:
        sand_tris.append((v, f))

  results = []
  for xq, yq in xy_points:
    best_z = None
    best_n = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    for v, f in sand_tris:
      v0, v1, v2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
      min_x = np.minimum(v0[:, 0], np.minimum(v1[:, 0], v2[:, 0]))
      max_x = np.maximum(v0[:, 0], np.maximum(v1[:, 0], v2[:, 0]))
      min_y = np.minimum(v0[:, 1], np.minimum(v1[:, 1], v2[:, 1]))
      max_y = np.maximum(v0[:, 1], np.maximum(v1[:, 1], v2[:, 1]))
      cand = np.where((xq >= min_x - 1e-4) & (xq <= max_x + 1e-4) & (yq >= min_y - 1e-4) & (yq <= max_y + 1e-4))[0]
      for fi in cand:
        a, b, c = v0[fi], v1[fi], v2[fi]
        dx_ac, dy_ac = c[0] - a[0], c[1] - a[1]
        dx_ab, dy_ab = b[0] - a[0], b[1] - a[1]
        dx_ap, dy_ap = xq - a[0], yq - a[1]
        denom = dx_ac * dy_ab - dx_ab * dy_ac
        if abs(denom) < 1e-12:
          continue
        u = (dx_ap * dy_ab - dx_ab * dy_ap) / denom
        w = (dx_ac * dy_ap - dx_ap * dy_ac) / denom
        if u >= -1e-4 and w >= -1e-4 and (u + w) <= 1.0 + 1e-4:
          z_hit = float(a[2] + u * (c[2] - a[2]) + w * (b[2] - a[2]))
          if best_z is None or z_hit > best_z:
            best_z = z_hit
            fn = np.cross(b - a, c - a)
            if fn[2] < 0:
              fn = -fn
            fnorm = np.linalg.norm(fn)
            if fnorm > 1e-8:
              best_n = (fn / fnorm).astype(np.float32)
    results.append((best_z if best_z is not None else 0.35, best_n))
  return results


def build_grounded_mjx_geom_buffers(
    stage_path: str,
    hum_xy: Tuple[float, float],
    hum_yaw: float,
    quad_xy: Tuple[float, float],
    quad_yaw: float,
) -> Dict[str, Any]:
  """Pose Humanoid-21DOF + Quadruped-12DOF via MuJoCo FK and ground feet cleanly on Moana beach sand."""
  xml = build_motunui_mjx_xml()
  mj_model = mujoco.MjModel.from_xml_string(xml)
  mj_data = mujoco.MjData(mj_model)

  def _set_j(jname: str, val: float):
    jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, jname)
    if jid >= 0:
      mj_data.qpos[mj_model.jnt_qposadr[jid]] = val

  def _quat_yaw_pitch(yaw: float, pitch: float = 0.0) -> np.ndarray:
    cy, sy = math.cos(0.5 * yaw), math.sin(0.5 * yaw)
    cp, sp = math.cos(0.5 * pitch), math.sin(0.5 * pitch)
    return np.array([cy * cp, -sy * sp, cy * sp, sy * cp], dtype=np.float64)

  # Humanoid confident walking / standing pose on sand
  mj_data.qpos[0:3] = [hum_xy[0], hum_xy[1], 1.5]
  mj_data.qpos[3:7] = _quat_yaw_pitch(hum_yaw, 0.04)
  _set_j("hum_abdomen_z", 0.12)
  _set_j("hum_abdomen_y", 0.05)
  _set_j("hum_rhip_y", -0.28)
  _set_j("hum_rknee", -0.30)
  _set_j("hum_rankle_y", 0.18)
  _set_j("hum_lhip_y", 0.22)
  _set_j("hum_lknee", -0.26)
  _set_j("hum_lankle_y", 0.10)
  _set_j("hum_rsh1", -0.52)
  _set_j("hum_rsh2", 0.22)
  _set_j("hum_relbow", -0.68)
  _set_j("hum_lsh1", 0.30)
  _set_j("hum_lsh2", -0.18)
  _set_j("hum_lelbow", -0.50)

  # Quadruped alert athletic stance pose on sand
  q_jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, "quad_root")
  q_adr = mj_model.jnt_qposadr[q_jid]
  mj_data.qpos[q_adr : q_adr + 3] = [quad_xy[0], quad_xy[1], 1.5]
  mj_data.qpos[q_adr + 3 : q_adr + 7] = _quat_yaw_pitch(quad_yaw, -0.04)
  for leg, th, ca in [("FL", 0.55, -1.18), ("FR", 0.72, -1.32), ("RL", 0.70, -1.28), ("RR", 0.52, -1.14)]:
    _set_j(f"{leg}_hip_x", 0.05 if "L" in leg else -0.05)
    _set_j(f"{leg}_thigh_y", th)
    _set_j(f"{leg}_calf_y", ca)

  mujoco.mj_forward(mj_model, mj_data)

  # Ground each robot accurately to the Moana isBeach triangle surface!
  (hz, _), (qz, _) = query_beach_sand_height_and_normal(stage_path, [hum_xy, quad_xy])

  hum_foot_gids = [mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("hum_rfoot_g", "hum_lfoot_g")]
  hum_min_z = min(float(mj_data.geom_xpos[g, 2] - mj_model.geom_size[g, 2]) for g in hum_foot_gids)
  mj_data.qpos[2] += (hz - hum_min_z) - 0.004

  quad_foot_gids = [mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, f"{l}_foot") for l in ("FL", "FR", "RL", "RR")]
  quad_min_z = min(float(mj_data.geom_xpos[g, 2] - mj_model.geom_size[g, 0]) for g in quad_foot_gids)
  mj_data.qpos[q_adr + 2] += (qz - quad_min_z) - 0.003

  mujoco.mj_forward(mj_model, mj_data)

  G = int(mj_model.ngeom)
  gtypes = [int(mj_model.geom_type[g]) for g in range(G)]
  gp = mj_data.geom_xpos.astype(np.float32)
  gm = mj_data.geom_xmat.reshape(G, 3, 3).astype(np.float32)
  gsz = mj_model.geom_size.astype(np.float32)
  grgba = np.zeros((G, 4), dtype=np.float32)
  grough = np.full((G,), 0.35, dtype=np.float32)
  gmetal = np.zeros((G,), dtype=np.float32)
  for g in range(G):
    mid = int(mj_model.geom_matid[g])
    if mid >= 0:
      grgba[g] = mj_model.mat_rgba[mid]
      mname = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_MATERIAL, mid) or ""
      if "titan" in mname or "gold" in mname:
        grough[g] = 0.24
        gmetal[g] = 0.85
      elif "visor" in mname:
        grough[g] = 0.08
        gmetal[g] = 0.20
      elif "rubber" in mname:
        grough[g] = 0.80
    else:
      grgba[g] = mj_model.geom_rgba[g]

  rb_sq = np.zeros((G,), dtype=np.float32)
  for g in range(G):
    gt = gtypes[g]
    sx, sy, sz = [float(x) for x in gsz[g]]
    if gt == GEOM_SPHERE:
      rb_sq[g] = (sx + 1e-4) ** 2
    elif gt == GEOM_CAPSULE:
      rb_sq[g] = (sx + sy + 1e-4) ** 2
    elif gt == GEOM_CYLINDER:
      rb_sq[g] = sx * sx + sy * sy + 1e-3
    else:
      rb_sq[g] = sx * sx + sy * sy + sz * sz + 1e-3

  G_pad = ((G + 127) // 128) * 128
  gf_raw = np.stack([
      gp[:, 0], gp[:, 1], gp[:, 2],
      gm[:, 0, 0], gm[:, 0, 1], gm[:, 0, 2],
      gm[:, 1, 0], gm[:, 1, 1], gm[:, 1, 2],
      gm[:, 2, 0], gm[:, 2, 1], gm[:, 2, 2],
      gsz[:, 0], gsz[:, 1], gsz[:, 2],
      rb_sq,
  ], axis=0).astype(np.float32)
  if G_pad > G:
    gf_pad = np.pad(gf_raw, ((0, 0), (0, G_pad - G)))
  else:
    gf_pad = gf_raw

  print(f"Grounded MJX robots on Moana beach: hum_z={hz:.3f}m quad_z={qz:.3f}m (ngeom={G})", flush=True)
  return {
      "ngeom": G,
      "gtypes": tuple(gtypes),
      "gf_pad": gf_pad,
      "grgba": grgba,
      "grough": grough,
      "gmetal": gmetal,
      "hum_z": hz,
      "quad_z": qz,
  }


def _ray_one_geom_pallas(
    gt: int,
    px, py, pz,
    m00, m01, m02, m10, m11, m12, m20, m21, m22,
    sx, sy, sz, rb_sq,
    rox, roy, roz, rdx, rdy, rdz,
    compute_normal: bool = True,
):
  """Branchless SoA primitive intersection inside Mosaic Pallas VMEM."""
  dx = rox - px
  dy = roy - py
  dz = roz - pz
  proj = dx * rdx + dy * rdy + dz * rdz
  d2_perp = (dx * dx + dy * dy + dz * dz) - proj * proj
  sphere_hit = d2_perp <= rb_sq

  lx = m00 * dx + m10 * dy + m20 * dz
  ly = m01 * dx + m11 * dy + m21 * dz
  lz = m02 * dx + m12 * dy + m22 * dz
  vx = m00 * rdx + m10 * rdy + m20 * rdz
  vy = m01 * rdx + m11 * rdy + m21 * rdz
  vz = m02 * rdx + m12 * rdy + m22 * rdz

  if gt == GEOM_SPHERE:
    a = rdx * rdx + rdy * rdy + rdz * rdz
    b = rdx * dx + rdy * dy + rdz * dz
    c = dx * dx + dy * dy + dz * dz - sx * sx
    det = b * b - a * c
    valid = sphere_hit & (det >= MJ_MINVAL) & (jnp.abs(a) > MJ_MINVAL)
    sd = jnp.sqrt(jnp.maximum(det, 0.0))
    inv_a = jnp.where(valid, 1.0 / jnp.where(valid, a, 1.0), 0.0)
    t1 = (-b - sd) * inv_a
    t2 = (-b + sd) * inv_a
    t = jnp.where(t1 >= 0.0, t1, t2)
    ok = valid & (t >= 0.0)
    cand_d = jnp.where(ok, t, MJ_MAXVAL)
    if not compute_normal:
      return cand_d, 0.0, 0.0, 0.0
    inv_s = 1.0 / jnp.maximum(sx, 1e-12)
    return cand_d, (dx + t * rdx) * inv_s, (dy + t * rdy) * inv_s, (dz + t * rdz) * inv_s

  if gt == GEOM_CAPSULE:
    a = vx * vx + vy * vy
    b = lx * vx + ly * vy
    c = lx * lx + ly * ly - sx * sx
    det = b * b - a * c
    has_a = jnp.abs(a) > MJ_MINVAL
    sd = jnp.sqrt(jnp.maximum(det, 0.0))
    inv_a = jnp.where(has_a, 1.0 / jnp.where(has_a, a, 1.0), 0.0)
    tc1, tc2 = (-b - sd) * inv_a, (-b + sd) * inv_a
    zc1, zc2 = lz + tc1 * vz, lz + tc2 * vz
    ok_c1 = sphere_hit & has_a & (det >= MJ_MINVAL) & (tc1 >= 0.0) & (jnp.abs(zc1) <= sy)
    ok_c2 = sphere_hit & has_a & (det >= MJ_MINVAL) & (tc2 >= 0.0) & (jnp.abs(zc2) <= sy)
    dc1, dc2 = jnp.where(ok_c1, tc1, MJ_MAXVAL), jnp.where(ok_c2, tc2, MJ_MAXVAL)
    d_cyl = jnp.minimum(dc1, dc2)

    dzp, dzm = lz - sy, lz + sy
    aa = vx * vx + vy * vy + vz * vz
    inv_aa = jnp.where(aa > MJ_MINVAL, 1.0 / jnp.where(aa > MJ_MINVAL, aa, 1.0), 0.0)
    bp, cp = vx * lx + vy * ly + vz * dzp, lx * lx + ly * ly + dzp * dzp - sx * sx
    detp = bp * bp - aa * cp
    sdp = jnp.sqrt(jnp.maximum(detp, 0.0))
    tp = jnp.where((-bp - sdp) * inv_aa >= 0.0, (-bp - sdp) * inv_aa, (-bp + sdp) * inv_aa)
    dp = jnp.where(sphere_hit & (aa > MJ_MINVAL) & (detp >= MJ_MINVAL) & (tp >= 0.0), tp, MJ_MAXVAL)

    bm, cm = vx * lx + vy * ly + vz * dzm, lx * lx + ly * ly + dzm * dzm - sx * sx
    detm = bm * bm - aa * cm
    sdm = jnp.sqrt(jnp.maximum(detm, 0.0))
    tm = jnp.where((-bm - sdm) * inv_aa >= 0.0, (-bm - sdm) * inv_aa, (-bm + sdm) * inv_aa)
    dm = jnp.where(sphere_hit & (aa > MJ_MINVAL) & (detm >= MJ_MINVAL) & (tm >= 0.0), tm, MJ_MAXVAL)

    best_d = jnp.minimum(d_cyl, jnp.minimum(dp, dm))
    if not compute_normal:
      return best_d, 0.0, 0.0, 0.0
    inv_r = 1.0 / jnp.maximum(sx, 1e-12)
    lnx = (lx + d_cyl * vx) * inv_r
    lny = (ly + d_cyl * vy) * inv_r
    lnz = jnp.zeros_like(lnx)
    use_p = dp < d_cyl
    lnx = jnp.where(use_p, (lx + dp * vx) * inv_r, lnx)
    lny = jnp.where(use_p, (ly + dp * vy) * inv_r, lny)
    lnz = jnp.where(use_p, (dzp + dp * vz) * inv_r, lnz)
    use_m = dm < jnp.minimum(d_cyl, dp)
    lnx = jnp.where(use_m, (lx + dm * vx) * inv_r, lnx)
    lny = jnp.where(use_m, (ly + dm * vy) * inv_r, lny)
    lnz = jnp.where(use_m, (dzm + dm * vz) * inv_r, lnz)
    return best_d, m00 * lnx + m01 * lny + m02 * lnz, m10 * lnx + m11 * lny + m12 * lnz, m20 * lnx + m21 * lny + m22 * lnz

  if gt == GEOM_CYLINDER:
    a = vx * vx + vy * vy
    b = lx * vx + ly * vy
    c = lx * lx + ly * ly - sx * sx
    det = b * b - a * c
    has_a = jnp.abs(a) > MJ_MINVAL
    sd = jnp.sqrt(jnp.maximum(det, 0.0))
    inv_a = jnp.where(has_a, 1.0 / jnp.where(has_a, a, 1.0), 0.0)
    t1, t2 = (-b - sd) * inv_a, (-b + sd) * inv_a
    z1, z2 = lz + t1 * vz, lz + t2 * vz
    d1 = jnp.where(sphere_hit & has_a & (det >= MJ_MINVAL) & (t1 >= 0.0) & (jnp.abs(z1) <= sy), t1, MJ_MAXVAL)
    d2 = jnp.where(sphere_hit & has_a & (det >= MJ_MINVAL) & (t2 >= 0.0) & (jnp.abs(z2) <= sy), t2, MJ_MAXVAL)
    d_side = jnp.minimum(d1, d2)

    has_vz = jnp.abs(vz) > MJ_MINVAL
    inv_vz = jnp.where(has_vz, 1.0 / jnp.where(has_vz, vz, 1.0), 0.0)
    tp, tm = (sy - lz) * inv_vz, (-sy - lz) * inv_vz
    xp, yp = lx + tp * vx, ly + tp * vy
    xm, ym = lx + tm * vx, ly + tm * vy
    dp = jnp.where(sphere_hit & has_vz & (tp >= 0.0) & (xp * xp + yp * yp <= sx * sx), tp, MJ_MAXVAL)
    dm = jnp.where(sphere_hit & has_vz & (tm >= 0.0) & (xm * xm + ym * ym <= sx * sx), tm, MJ_MAXVAL)
    best_d = jnp.minimum(d_side, jnp.minimum(dp, dm))
    if not compute_normal:
      return best_d, 0.0, 0.0, 0.0
    inv_r = 1.0 / jnp.maximum(sx, 1e-12)
    lsx, lsy = (lx + d_side * vx) * inv_r, (ly + d_side * vy) * inv_r
    use_p = dp < d_side
    lnx, lny, lnz = jnp.where(use_p, 0.0, lsx), jnp.where(use_p, 0.0, lsy), jnp.where(use_p, 1.0, 0.0)
    use_m = dm < jnp.minimum(d_side, dp)
    lnx, lny, lnz = jnp.where(use_m, 0.0, lnx), jnp.where(use_m, 0.0, lny), jnp.where(use_m, -1.0, lnz)
    return best_d, m00 * lnx + m01 * lny + m02 * lnz, m10 * lnx + m11 * lny + m12 * lnz, m20 * lnx + m21 * lny + m22 * lnz

  inv_vx = jnp.where(jnp.abs(vx) > MJ_MINVAL, 1.0 / jnp.where(jnp.abs(vx) > MJ_MINVAL, vx, 1.0), 1.0e12)
  inv_vy = jnp.where(jnp.abs(vy) > MJ_MINVAL, 1.0 / jnp.where(jnp.abs(vy) > MJ_MINVAL, vy, 1.0), 1.0e12)
  inv_vz = jnp.where(jnp.abs(vz) > MJ_MINVAL, 1.0 / jnp.where(jnp.abs(vz) > MJ_MINVAL, vz, 1.0), 1.0e12)
  tx1, tx2 = (-sx - lx) * inv_vx, (sx - lx) * inv_vx
  ty1, ty2 = (-sy - ly) * inv_vy, (sy - ly) * inv_vy
  tz1, tz2 = (-sz - lz) * inv_vz, (sz - lz) * inv_vz
  tx_lo, tx_hi = jnp.minimum(tx1, tx2), jnp.maximum(tx1, tx2)
  ty_lo, ty_hi = jnp.minimum(ty1, ty2), jnp.maximum(ty1, ty2)
  tz_lo, tz_hi = jnp.minimum(tz1, tz2), jnp.maximum(tz1, tz2)
  tmin = jnp.maximum(tx_lo, jnp.maximum(ty_lo, tz_lo))
  tmax = jnp.minimum(tx_hi, jnp.minimum(ty_hi, tz_hi))
  t = jnp.where(tmin >= 0.0, tmin, tmax)
  ok = sphere_hit & (tmax >= tmin) & (tmax >= 0.0) & (t >= 0.0)
  cand_d = jnp.where(ok, t, MJ_MAXVAL)
  if not compute_normal:
    return cand_d, 0.0, 0.0, 0.0
  hit_x = (tx_lo >= ty_lo) & (tx_lo >= tz_lo)
  hit_y = (~hit_x) & (ty_lo >= tz_lo)
  lnx = jnp.where(hit_x, jnp.where(vx >= 0.0, -1.0, 1.0), 0.0)
  lny = jnp.where(hit_y, jnp.where(vy >= 0.0, -1.0, 1.0), 0.0)
  lnz = jnp.where((~hit_x) & (~hit_y), jnp.where(vz >= 0.0, -1.0, 1.0), 0.0)
  return cand_d, m00 * lnx + m01 * lny + m02 * lnz, m10 * lnx + m11 * lny + m12 * lnz, m20 * lnx + m21 * lny + m22 * lnz


def pallas_raycast_mjx_tiled(
    gtypes_static: Tuple[int, ...],
    gf_pad: jax.Array,       # (16, G_pad)
    ray_orig: jax.Array,     # (B, R, 3)
    ray_dir: jax.Array,      # (B, R, 3)
    t_limit: jax.Array,      # (B, R)
    H_sub: int,
    W: int,
    any_hit: bool = False,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
  """Zero-gather Mosaic Pallas (8, 128) MJX raycaster matching Moana tile layout -> (d, nrm, hit_gid_f)."""
  B, _, _ = ray_orig.shape
  rays_2d, n_tiles, can_2d = _pack_rays_tiled(ray_orig, ray_dir, t_limit, H_sub, W)
  total_tiles = B * n_tiles

  def _mjx_kernel(ray_ref, gf_ref, out_ref):
    rt = ray_ref[...]  # (8, 128)
    gf = gf_ref[...]   # (16, G_pad)
    rox, roy, roz = rt[0:1, :], rt[1:2, :], rt[2:3, :]
    rdx, rdy, rdz = rt[3:4, :], rt[4:5, :], rt[5:6, :]
    bd = rt[6:7, :]
    bnx = jnp.zeros((1, BLOCK_R), jnp.float32)
    bny = jnp.zeros((1, BLOCK_R), jnp.float32)
    bnz = jnp.zeros((1, BLOCK_R), jnp.float32)
    bgid = jnp.full((1, BLOCK_R), -1.0, jnp.float32)

    for k, gt in enumerate(gtypes_static):
      cd, cnx, cny, cnz = _ray_one_geom_pallas(
          gt,
          gf[0:1, k:k+1], gf[1:2, k:k+1], gf[2:3, k:k+1],
          gf[3:4, k:k+1], gf[4:5, k:k+1], gf[5:6, k:k+1],
          gf[6:7, k:k+1], gf[7:8, k:k+1], gf[8:9, k:k+1],
          gf[9:10, k:k+1], gf[10:11, k:k+1], gf[11:12, k:k+1],
          gf[12:13, k:k+1], gf[13:14, k:k+1], gf[14:15, k:k+1],
          gf[15:16, k:k+1],
          rox, roy, roz, rdx, rdy, rdz,
          compute_normal=not any_hit,
      )
      imp = cd < bd
      bd = jnp.where(imp, cd, bd)
      if not any_hit:
        bnx = jnp.where(imp, cnx, bnx)
        bny = jnp.where(imp, cny, bny)
        bnz = jnp.where(imp, cnz, bnz)
        bgid = jnp.where(imp, float(k), bgid)

    z = jnp.zeros((1, BLOCK_R), jnp.float32)
    out_ref[...] = jnp.concatenate([bd, bnx, bny, bnz, bgid, z, z, z], axis=0)

  out_2d = pl.pallas_call(
      _mjx_kernel,
      out_shape=jax.ShapeDtypeStruct((total_tiles * 8, BLOCK_R), jnp.float32),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          grid=(total_tiles,),
          in_specs=[
              pl.BlockSpec((8, BLOCK_R), lambda i: (i, 0)),
              pl.BlockSpec((16, int(gf_pad.shape[1])), lambda i: (0, 0)),
          ],
          out_specs=pl.BlockSpec((8, BLOCK_R), lambda i: (i, 0)),
      ),
      compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
      interpret=False,
  )(rays_2d, gf_pad)

  # Unpack d, nrm, and gid row
  d_out, n_out, _ = _unpack_out_tiled(out_2d, B, H_sub, W, n_tiles, can_2d, return_stats=False)
  # Extract row 4 (bgid) using identical untiling without gather
  gid_tiles = out_2d.reshape(B, n_tiles, 8, BLOCK_R)[:, :, 4, :]
  if can_2d:
    ty, tx = H_sub // TILE_H, W // TILE_W
    gid_grid = gid_tiles.reshape(B, ty, tx, TILE_H, TILE_W).transpose(0, 1, 3, 2, 4).reshape(B, H_sub * W)
  else:
    gid_grid = gid_tiles.reshape(B, n_tiles * BLOCK_R)[:, : H_sub * W]
  return d_out, n_out, jnp.round(gid_grid).astype(jnp.int32)


def make_moana_polished_renderer(
    compiled: Dict[str, Any],
    env_bf16: jax.Array,
    devices: List[jax.Device],
    H: int,
    W: int,
    n_subbands: int,
    dome_rot_z_deg: float = 25.0,
    use_shadows: bool = True,
    mjx_info: Optional[Dict[str, Any]] = None,
):
  """Polished 4-chip Moana renderer supporting subband scan (8K safe) & optional MJX robots."""
  n_chips = len(devices)
  assert H % n_chips == 0, (H, n_chips)
  band_h = H // n_chips
  assert band_h % n_subbands == 0, (band_h, n_subbands)
  sub_h = band_h // n_subbands
  sub_h_pad = ((sub_h + TILE_H - 1) // TILE_H) * TILE_H

  mesh = Mesh(np.array(devices), ("spatial",))
  rep = NamedSharding(mesh, P())

  u_spec = compiled["unique_spec"]
  u_devs = compiled["unique_dev_arrays"]
  u_pbr = compiled["unique_pbr"]
  w_spec = compiled["water_spec"]
  w_devs = compiled["water_dev_arrays"]
  i_spec = compiled["inst_spec"]
  i_devs = compiled["inst_dev_arrays"]
  i_pbr = compiled["inst_pbr"]
  base_spec = u_spec if u_spec is not None else i_spec
  znear_f = float(base_spec.znear)
  zfar_f = float(base_spec.zfar)

  inst_inv_xmats = []
  for xm in compiled["inst_xmat"]:
    xm_np = np.asarray(xm)
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

  # Sun direction from Disney sun_quad_llc
  sun_dir_np = np.array([0.324, -0.694, 0.643], dtype=np.float32)
  sun_dir_np /= np.linalg.norm(sun_dir_np)

  has_mjx = mjx_info is not None
  if has_mjx:
    mjx_gtypes = mjx_info["gtypes"]
    mjx_gf_dev = jax.device_put(mjx_info["gf_pad"], rep)
    mjx_rgba_dev = jax.device_put(mjx_info["grgba"][:, :3], rep)
    mjx_rough_dev = jax.device_put(mjx_info["grough"], rep)
    mjx_metal_dev = jax.device_put(mjx_info["gmetal"], rep)
  else:
    mjx_gtypes = ()
    mjx_gf_dev = jax.device_put(np.zeros((16, 128), dtype=np.float32), rep)
    mjx_rgba_dev = jax.device_put(np.zeros((1, 3), dtype=np.float32), rep)
    mjx_rough_dev = jax.device_put(np.zeros((1,), dtype=np.float32), rep)
    mjx_metal_dev = jax.device_put(np.zeros((1,), dtype=np.float32), rep)

  def _shard_fn(u_da, u_oxp, u_oxm, w_da, i_da, i_xp, i_xm, i_inv, cxp, cxm, tan_hf, env_b, mjx_gf, mjx_rgba, mjx_rg, mjx_mt):
    bidx = jax.lax.axis_index("spatial")

    # Precompute 3-level TLAS AABBs & packed M_inv poses once per frame!
    tlas_precomps = []
    if i_spec is not None:
      for m_idx, it in enumerate(i_spec.meshes):
        tlas_precomps.append(_precompute_tlas_group(it, i_xp[m_idx], i_xm[m_idx], i_inv[m_idx], tlas_group_size=64))

    def _render_subband(_, sb_idx):
      ray_orig, ray_dir, neg_z = _compute_camera_rays_subband(
          cxp, cxm, tan_hf, znear_f, H, W, bidx, band_h, sb_idx, sub_h, sub_h_pad
      )
      B, R, _ = ray_orig.shape
      max_dist = jnp.broadcast_to(((zfar_f - znear_f) / neg_z)[None, :], (B, R))

      best_d = max_dist
      best_hit = jnp.zeros((B, R), dtype=jnp.bool_)
      best_nrm = jnp.zeros((B, R, 3), dtype=jnp.float32)
      best_alb = jnp.zeros((B, R, 3), dtype=jnp.float32)
      best_rough = jnp.full((B, R), 0.72, dtype=jnp.float32)
      best_tag = jnp.zeros((B, R), dtype=jnp.int32)

      if u_spec is not None:
        for m_idx, it in enumerate(u_spec.meshes):
          l2_r, l2_a, l0_a, leaf_b = u_da[m_idx]
          p = u_oxp[:, m_idx, :]
          rmat = u_oxm[:, m_idx, :, :]
          ro_loc = jnp.einsum("bji,brj->bri", rmat, ray_orig - p[:, None, :], precision=jax.lax.Precision.HIGHEST)
          rd_loc = jnp.einsum("bji,brj->bri", rmat, ray_dir, precision=jax.lax.Precision.HIGHEST)
          d_m, n_loc_m, _ = _pallas_compact_bvh_raycast(l2_r, l2_a, l0_a, leaf_b, ro_loc, rd_loc, best_d, sub_h_pad, W, any_hit=False, interpret=False)
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
          d_m, n_w_m = _pallas_tlas_precomp_raycast(
              tlas_precomps[m_idx], l2_r, l2_a, l0_a, leaf_b,
              ray_orig, ray_dir, best_d, sub_h_pad, W,
              tlas_group_size=64, tlas_top_groups=6, max_active_per_tile=24, any_hit=False,
          )
          imp = d_m < best_d
          best_d = jnp.where(imp, d_m, best_d)
          best_hit = best_hit | imp
          best_nrm = jnp.where(imp[..., None], n_w_m, best_nrm)
          best_alb = jnp.where(imp[..., None], jnp.asarray(it.rgba[:3], dtype=jnp.float32)[None, None, :], best_alb)
          r_m, _, tag_m = i_pbr[m_idx]
          best_rough = jnp.where(imp, float(r_m), best_rough)
          best_tag = jnp.where(imp, jnp.int32(TAG_TO_ID.get(tag_m, 0)), best_tag)

      water_d = max_dist
      water_hit = jnp.zeros((B, R), dtype=jnp.bool_)
      water_nrm = jnp.zeros((B, R, 3), dtype=jnp.float32)
      if w_spec is not None and len(w_spec.meshes) > 0:
        for m_idx, _it in enumerate(w_spec.meshes):
          l2_r, l2_a, l0_a, leaf_b = w_da[m_idx]
          d_w, n_w, _ = _pallas_compact_bvh_raycast(l2_r, l2_a, l0_a, leaf_b, ray_orig, ray_dir, water_d, sub_h_pad, W, any_hit=False, interpret=False)
          imp_w = d_w < water_d
          water_d = jnp.where(imp_w, d_w, water_d)
          water_hit = water_hit | imp_w
          water_nrm = jnp.where(imp_w[..., None], n_w, water_nrm)

      # Optional Articulated MJX Robot Raycast + Exact Z-Depth Occlusion!
      if has_mjx:
        mjx_d, mjx_nrm, mjx_gid = pallas_raycast_mjx_tiled(
            mjx_gtypes, mjx_gf, ray_orig, ray_dir, max_dist, sub_h_pad, W, any_hit=False
        )
        mjx_hit = (mjx_gid >= 0) & (mjx_d < max_dist - 1e-3)
        eff_water_d = jnp.where(water_hit, water_d, max_dist)
        is_mjx_front = mjx_hit & (mjx_d < best_d) & (mjx_d < eff_water_d)
        # Zero-gather one-hot lookup for robot material properties (G <= 64)
        n_g = int(mjx_rgba.shape[0])
        g_oh = (mjx_gid[:, :, None] == jnp.arange(n_g, dtype=jnp.int32)[None, None, :]).astype(jnp.float32)
        mjx_alb = jnp.einsum("brg,gc->brc", g_oh, mjx_rgba, precision=jax.lax.Precision.HIGHEST)
        mjx_r_val = jnp.einsum("brg,g->br", g_oh, mjx_rg, precision=jax.lax.Precision.HIGHEST)
        mjx_m_val = jnp.einsum("brg,g->br", g_oh, mjx_mt, precision=jax.lax.Precision.HIGHEST)
      else:
        mjx_d = max_dist
        mjx_nrm = jnp.zeros((B, R, 3), dtype=jnp.float32)
        is_mjx_front = jnp.zeros((B, R), dtype=jnp.bool_)
        mjx_alb = jnp.zeros((B, R, 3), dtype=jnp.float32)
        mjx_r_val = jnp.full((B, R), 0.35, dtype=jnp.float32)
        mjx_m_val = jnp.zeros((B, R), dtype=jnp.float32)

      vdir = -ray_dir
      best_nrm = jnp.where(best_hit[..., None], best_nrm, 0.0)
      flip = jnp.sum(best_nrm * vdir, axis=-1, keepdims=True) < 0.0
      best_nrm = jnp.where(flip, -best_nrm, best_nrm)
      hit_pos = ray_orig + jnp.where(best_hit, best_d, 0.0)[..., None] * ray_dir

      # Polished Moana procedural PBR micro-variation
      px, py, pz = hit_pos[..., 0], hit_pos[..., 1], hit_pos[..., 2]
      n_wave1 = jnp.sin(px * 0.37 + py * 0.83 + pz * 0.29) * jnp.sin(py * 0.41 - px * 0.71 + pz * 0.53)
      n_wave2 = jnp.sin(px * 1.93 - py * 2.71 + pz * 1.37) * jnp.cos(py * 2.19 + px * 1.57 - pz * 2.41)

      # Rich warm golden beach sand with subtle wind ripples
      sand_rip = jnp.sin(px * 8.5 + py * 4.2 + 0.7 * n_wave1)
      sand_alb = best_alb * jnp.stack([1.08, 0.97, 0.80], axis=-1) * (0.94 + 0.07 * sand_rip + 0.04 * n_wave2)[..., None]

      # Deep lush tropical emerald foliage & mountain canopy (avoids pastel sage washout)
      fol_var = 0.52 + 0.22 * n_wave1 + 0.13 * n_wave2
      fol_alb = best_alb * jnp.stack([fol_var * 0.72, fol_var * 0.96, fol_var * 0.45], axis=-1)

      # Organic weathered palm & ironwood bark striation
      bark_var = 0.82 + 0.18 * n_wave1 + 0.10 * n_wave2
      bark_alb = best_alb * bark_var[..., None]

      # Volcanic basalt rock strata + wet shoreline sheen
      rock_var = 0.72 + 0.28 * n_wave1 + 0.15 * n_wave2
      wet_mask = jnp.clip(1.0 - (pz + 0.08) / 0.75, 0.0, 1.0)
      rock_alb = best_alb * rock_var[..., None] * (1.0 - 0.42 * wet_mask[..., None])
      eff_rough = jnp.where(best_tag == 4, best_rough * (1.0 - 0.48 * wet_mask), best_rough)

      # Vibrant sunlit chartreuse / olive-gold underwater coral heads & sunlit reef moss
      coral_alb = best_alb * jnp.stack([1.35 + 0.42 * n_wave1, 1.65 + 0.45 * n_wave2, 0.38 + 0.18 * n_wave1], axis=-1)
      is_underwater_reef = water_hit & (water_d < best_d) & best_hit & ((best_tag == 4) | (best_tag == 5) | (best_tag == 0))
      reef_moss = jnp.clip(best_nrm[..., 2] * (0.55 + 0.35 * n_wave1), 0.0, 1.0)
      reef_tint = jnp.array([0.28, 0.44, 0.08], dtype=jnp.float32)[None, None, :] * (0.75 + 0.45 * n_wave2)[..., None]

      albedo = jnp.where((best_tag == 1)[..., None], sand_alb, best_alb)
      albedo = jnp.where((best_tag == 2)[..., None], fol_alb, albedo)
      albedo = jnp.where((best_tag == 3)[..., None], bark_alb, albedo)
      albedo = jnp.where((best_tag == 4)[..., None], rock_alb, albedo)
      albedo = jnp.where((best_tag == 5)[..., None], coral_alb, albedo)
      albedo = jnp.where(is_underwater_reef[..., None], albedo * (1.0 - 0.65 * reef_moss[..., None]) + reef_tint * (0.65 * reef_moss[..., None]), albedo)
      albedo = jnp.clip(albedo, 0.01, 0.98)

      L = jnp.broadcast_to(jnp.asarray(sun_dir_np, dtype=jnp.float32)[None, None, :], ray_dir.shape)

      # Unified normal / hit position / roughness / metallic for Moana + MJX foreground
      if has_mjx:
        mjx_hit_pos = ray_orig + jnp.where(is_mjx_front, mjx_d, 0.0)[..., None] * ray_dir
        mjx_nrm_f = jnp.where(jnp.sum(mjx_nrm * vdir, axis=-1, keepdims=True) < 0.0, -mjx_nrm, mjx_nrm)
        surf_nrm = jnp.where(is_mjx_front[..., None], mjx_nrm_f, best_nrm)
        surf_pos = jnp.where(is_mjx_front[..., None], mjx_hit_pos, hit_pos)
        surf_d = jnp.where(is_mjx_front, mjx_d, best_d)
        surf_alb = jnp.where(is_mjx_front[..., None], mjx_alb, albedo)
        surf_rough = jnp.where(is_mjx_front, mjx_r_val, eff_rough)
        surf_metal = jnp.where(is_mjx_front, mjx_m_val, 0.0)
        surf_hit = best_hit | is_mjx_front
      else:
        surf_nrm = best_nrm
        surf_pos = hit_pos
        surf_d = best_d
        surf_alb = albedo
        surf_rough = eff_rough
        surf_metal = jnp.zeros((B, R), dtype=jnp.float32)
        surf_hit = best_hit

      ndotl = jnp.maximum(0.0, jnp.sum(surf_nrm * L, axis=-1))

      # Unified Two-Way Sun Shadow Raycast
      vis = jnp.ones((B, R), dtype=jnp.float32)
      if use_shadows:
        sh_bias = jnp.where(is_mjx_front, 0.008, jnp.maximum(0.03, surf_d * 0.0012))[..., None]
        sh_o = surf_pos + surf_nrm * sh_bias
        sh_lim = jnp.where(surf_hit & (ndotl > 0.02), 350.0, 0.0)
        occ = jnp.zeros((B, R), dtype=jnp.bool_)
        if u_spec is not None:
          for m_idx, _it in enumerate(u_spec.meshes):
            l2_r, l2_a, l0_a, leaf_b = u_da[m_idx]
            p = u_oxp[:, m_idx, :]
            rmat = u_oxm[:, m_idx, :, :]
            ro_l = jnp.einsum("bji,brj->bri", rmat, sh_o - p[:, None, :], precision=jax.lax.Precision.HIGHEST)
            rd_l = jnp.einsum("bji,brj->bri", rmat, L, precision=jax.lax.Precision.HIGHEST)
            sh_lim_rem = jnp.where(occ, 0.0, sh_lim)
            d_sh, _, _ = _pallas_compact_bvh_raycast(l2_r, l2_a, l0_a, leaf_b, ro_l, rd_l, sh_lim_rem, sub_h_pad, W, any_hit=True, interpret=False)
            occ = occ | (d_sh < sh_lim_rem)
        if i_spec is not None:
          for m_idx, _it in enumerate(i_spec.meshes):
            l2_r, l2_a, l0_a, leaf_b = i_da[m_idx]
            sh_lim_rem = jnp.where(occ, 0.0, sh_lim)
            d_sh, _ = _pallas_tlas_precomp_raycast(
                tlas_precomps[m_idx], l2_r, l2_a, l0_a, leaf_b,
                sh_o, L, sh_lim_rem, sub_h_pad, W,
                tlas_group_size=64, tlas_top_groups=4, max_active_per_tile=16, any_hit=True,
            )
            occ = occ | (d_sh < sh_lim_rem)
        if has_mjx:
          # Robot self-shadows + crisp articulated shadows cast onto beach sand!
          sh_lim_mjx = jnp.where(occ, 0.0, jnp.minimum(sh_lim, 25.0))
          d_sh_mjx, _, _ = pallas_raycast_mjx_tiled(
              mjx_gtypes, mjx_gf, sh_o, L, sh_lim_mjx, sub_h_pad, W, any_hit=True
          )
          occ = occ | (d_sh_mjx < sh_lim_mjx)
        vis = jnp.where(occ, 0.025, 1.0)

      # Rich warm golden tropical sunlight + deep sculpted jungle shadows
      sun_col = jnp.array([4.10, 3.10, 2.05], dtype=jnp.float32)[None, None, :]
      H_vec = (L + vdir) / jnp.maximum(jnp.linalg.norm(L + vdir, axis=-1, keepdims=True), 1e-12)
      ndoth = jnp.maximum(0.0, jnp.sum(surf_nrm * H_vec, axis=-1))
      ndotv = jnp.maximum(1e-3, jnp.sum(surf_nrm * vdir, axis=-1))
      alpha = jnp.clip(surf_rough * surf_rough, 0.02, 0.98)
      a2 = alpha * alpha
      denom = ndoth * ndoth * (a2 - 1.0) + 1.0
      D_ggx = a2 / (np.pi * denom * denom + 1e-6)
      f0 = jnp.where(is_mjx_front[..., None], 0.04 * (1.0 - surf_metal[..., None]) + surf_alb * surf_metal[..., None], 0.04)
      F_schlick = f0 + (1.0 - f0) * ((1.0 - jnp.maximum(0.0, jnp.sum(H_vec * vdir, axis=-1))) ** 5.0)[..., None]
      spec_term = (D_ggx[..., None] * F_schlick) / (4.0 * ndotv * jnp.maximum(ndotl, 0.05) + 0.12)[..., None]
      spec_term = jnp.minimum(spec_term, 3.5) * vis[..., None]

      # Foliage subsurface sunlit translucency (warm golden-green backlit palms)
      sss_dir = jnp.maximum(0.0, jnp.sum(vdir * (-L + 0.35 * surf_nrm), axis=-1)) ** 2.4
      sss_mask = ((~is_mjx_front) & ((best_tag == 2) | (best_tag == 6))).astype(jnp.float32)
      sss_rgb = surf_alb * sun_col * (0.58 * sss_dir * sss_mask * (0.18 + 0.82 * vis))[..., None]

      # Deep shadow sky hemisphere + warm sand bounce for robots
      sky_up = 0.5 + 0.5 * surf_nrm[..., 2]
      sky_fill = (
          sky_up[..., None] * jnp.array([0.16, 0.28, 0.44], dtype=jnp.float32)[None, None, :]
          + (1.0 - sky_up[..., None]) * jnp.where(
              is_mjx_front[..., None],
              jnp.array([0.42, 0.32, 0.20], dtype=jnp.float32)[None, None, :],
              jnp.array([0.05, 0.09, 0.06], dtype=jnp.float32)[None, None, :],
          )
      )
      ao = jnp.where(is_mjx_front, 0.65 + 0.35 * sky_up, jnp.where(best_tag == 2, 0.28 + 0.72 * (sky_up ** 1.35), 0.45 + 0.55 * sky_up))
      diff_weight = 1.0 - 0.75 * surf_metal[..., None]
      solid_hdr = (
          surf_alb * diff_weight * (sun_col * (ndotl * vis)[..., None] * (1.0 / np.pi) + sky_fill * ao[..., None] * 0.21)
          + sun_col * spec_term * ndotl[..., None]
          + sss_rgb
      )

      # Ocean surface + Sky / Robot HDRI reflections
      is_water_front = (~is_mjx_front) & water_hit & (water_d < best_d)
      water_pos = ray_orig + jnp.where(water_hit, water_d, 0.0)[..., None] * ray_dir
      wx, wy = water_pos[..., 0], water_pos[..., 1]
      w_nx = 0.018 * jnp.cos(wx * 2.3 + wy * 1.5) + 0.010 * jnp.sin(wx * 6.7 - wy * 5.3) + 0.005 * jnp.cos(wx * 14.5 + wy * 11.2)
      w_ny = 0.018 * jnp.sin(wy * 2.5 - wx * 1.3) + 0.010 * jnp.cos(wy * 6.0 + wx * 4.9) + 0.005 * jnp.sin(wy * 13.8 - wx * 9.8)
      w_nrm = jnp.where(water_nrm[..., 2:3] >= 0.0, water_nrm, -water_nrm) + jnp.stack([w_nx, w_ny, jnp.zeros_like(w_nx)], axis=-1)
      w_nrm = w_nrm / jnp.maximum(jnp.linalg.norm(w_nrm, axis=-1, keepdims=True), 1e-8)
      refl_w = ray_dir - 2.0 * jnp.sum(ray_dir * w_nrm, axis=-1, keepdims=True) * w_nrm
      refl_w = jnp.where(refl_w[..., 2:3] < 0.02, jnp.concatenate([refl_w[..., :2], jnp.full_like(refl_w[..., 2:3], 0.02)], axis=-1), refl_w)
      refl_w = refl_w / jnp.maximum(jnp.linalg.norm(refl_w, axis=-1, keepdims=True), 1e-8)

      refl_m = ray_dir - 2.0 * jnp.sum(ray_dir * surf_nrm, axis=-1, keepdims=True) * surf_nrm
      env_lookup_dir = jnp.where(is_water_front[..., None], refl_w, jnp.where(is_mjx_front[..., None], refl_m, ray_dir))
      env_rgb = pallas_sample_hdri_env(env_lookup_dir, env_b, sub_h_pad, W, H_env=512, W_env=1024)

      if has_mjx:
        # Metallic / glossy HDRI environment reflection on MJX robot titanium & visor
        fres_m = (0.04 + 0.70 * surf_metal + 0.25 * ((1.0 - ndotv) ** 4.0)) * (1.0 - 0.70 * surf_rough)
        solid_hdr = jnp.where(
            is_mjx_front[..., None],
            solid_hdr + env_rgb * (fres_m * 0.42)[..., None] * jnp.where(surf_metal[..., None] > 0.5, surf_alb, 1.0),
            solid_hdr,
        )

      # Vibrant crystal-clear Beer-Lambert coral lagoon (avoid over-blowing shallow beach edge)
      t_water = jnp.where(best_hit, jnp.maximum(0.0, best_d - water_d), 45.0)
      sigma_t = jnp.array([0.34, 0.068, 0.14], dtype=jnp.float32)[None, None, :]
      transmittance = jnp.exp(-sigma_t * t_water[..., None])
      in_scatter = jnp.array([0.004, 0.115, 0.135], dtype=jnp.float32)[None, None, :] * (1.0 - jnp.exp(-0.18 * t_water[..., None]))
      coral_boost = jnp.where((best_tag == 5)[..., None], 2.45, 1.25)
      underwater_rgb = jnp.where(best_hit[..., None], solid_hdr * coral_boost * transmittance + in_scatter, in_scatter * 0.45)

      cos_w = jnp.maximum(0.0, jnp.sum(w_nrm * vdir, axis=-1))
      fresnel_w = jnp.minimum(0.48, 0.025 + 0.62 * ((1.0 - cos_w) ** 4.2))
      w_ndoth = jnp.maximum(0.0, jnp.sum(w_nrm * H_vec, axis=-1))
      w_sun_spec = sun_col * ((w_ndoth ** 460.0) * 1.55 + (w_ndoth ** 64.0) * 0.03)[..., None]
      water_rgb = underwater_rgb * (1.0 - fresnel_w[..., None]) + env_rgb * 0.72 * fresnel_w[..., None] + w_sun_spec

      surf_rgb = jnp.where(is_water_front[..., None], water_rgb, solid_hdr)
      eff_d = jnp.where(is_water_front, water_d, surf_d)
      any_surface = surf_hit | is_water_front

      fog_t = jnp.clip(1.0 - jnp.exp(-eff_d * 0.000045), 0.0, 0.08)[..., None]
      horizon_mist = jnp.array([0.38, 0.66, 0.90], dtype=jnp.float32)[None, None, :]
      surf_fog = surf_rgb * (1.0 - fog_t) + horizon_mist * fog_t

      hdr_final = jnp.where(any_surface[..., None], surf_fog, env_rgb)
      ldr_rgb = agx_filmic_tonemap(hdr_final, exposure=1.22)
      return None, ldr_rgb.reshape(B, sub_h_pad, W, 3)[:, :sub_h, :, :]

    _, sub_stack = jax.lax.scan(_render_subband, None, jnp.arange(n_subbands, dtype=jnp.int32))
    # (n_subbands, B, sub_h, W, 3) -> (B, band_h, W, 3)
    return sub_stack.transpose(1, 0, 2, 3, 4).reshape(1, band_h, W, 3)

  sharded = shard_map(
      _shard_fn,
      mesh=mesh,
      in_specs=(P(), P(), P(), P(), P(), P(), P(), P(), P(), P(), P(), P(), P(), P(), P(), P()),
      out_specs=P(None, "spatial", None, None),
      check_rep=False,
  )
  fn = jax.jit(sharded)
  base_args = (
      u_devs, compiled["obj_xpos"], compiled["obj_xmat"],
      w_devs,
      i_devs, compiled["inst_xpos"], compiled["inst_xmat"], inst_inv_xmats_t,
  )
  mjx_args = (mjx_gf_dev, mjx_rgba_dev, mjx_rough_dev, mjx_metal_dev)
  return fn, base_args, mjx_args, rep


def make_multiangle_grid(
    shot_8k_path: str,
    hyperion_ref_path: str,
    beach_8k_path: str,
    dunes_8k_path: str,
    out_grid_path: str,
    stats_8k: Dict[str, Any],
):
  """Compose crisp 4-panel Moana 8K Cinema Postcard Grid + 1:1 Native 8K detail crop."""
  im_shot = Image.open(shot_8k_path).convert("RGB")
  im_hyp = Image.open(hyperion_ref_path).convert("RGB")
  im_beach = Image.open(beach_8k_path).convert("RGB")
  im_dunes = Image.open(dunes_8k_path).convert("RGB")

  pw, ph = 1440, 810
  # Panel 1: shotCam Polished TPU (left half) vs Disney Hyperion Reference (right half) split comparison
  # Crop 16:9 shot_8k to match Hyperion 2.387:1 inside a clean top/bottom or left/right layout:
  # Let's do crisp top-half TPU 8K Polished vs bottom-half Hyperion Reference OR diagonal/split-wipe!
  # Left-half TPU vs Right-half Hyperion split-screen at matched 2.387:1 center-crop is super readable!
  tw, th = im_shot.size
  hw, hh = im_hyp.size
  crop_h = int(round(tw * (hh / hw)))
  y0 = (th - crop_h) // 2
  shot_w = im_shot.crop((0, y0, tw, y0 + crop_h)).resize((pw, int(round(pw * hh / hw))), Image.Resampling.LANCZOS)
  hyp_w = im_hyp.resize(shot_w.size, Image.Resampling.LANCZOS)
  p1 = Image.new("RGB", (pw, ph), (14, 18, 26))
  # Top = TPU 8K Polished, Bottom = Hyperion Reference? Wait: split left/right at full height!
  sh_y = (ph - shot_w.size[1]) // 2
  split_x = pw // 2
  p1.paste(shot_w.crop((0, 0, split_x, shot_w.size[1])), (0, sh_y))
  p1.paste(hyp_w.crop((split_x, 0, pw, hyp_w.size[1])), (split_x, sh_y))
  dr1 = ImageDraw.Draw(p1)
  dr1.line([(split_x, sh_y), (split_x, sh_y + shot_w.size[1])], fill=(255, 220, 120), width=3)

  # Panel 2: beachCam palm canopy & sunlit shore
  p2 = im_beach.resize((pw, ph), Image.Resampling.LANCZOS)
  # Panel 3: dunesACam / palmsCam volcanic ridge & foliage
  p3 = im_dunes.resize((pw, ph), Image.Resampling.LANCZOS)
  # Panel 4: True 1:1 Native 8K pixel crop (1440x810 from 7680x4320) showing palm fronds + coral reef detail!
  # Pick region around palm canopy + shoreline + coral in shotCam 8K (center-left x~1600..3040, y~2050..2860)
  cx0 = int(tw * 0.20)
  cy0 = int(th * 0.44)
  p4 = im_shot.crop((cx0, cy0, cx0 + pw, cy0 + ph))

  pad = 24
  hdr = 56
  top_bar = 76
  cw = pad * 3 + pw * 2
  ch = top_bar + pad * 3 + (ph + hdr) * 2
  canvas = Image.new("RGB", (cw, ch), (12, 16, 22))
  dr = ImageDraw.Draw(canvas)
  try:
    font_lg = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)
    font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 19)
  except Exception:
    font_lg = ImageFont.load_default()
    font_sm = ImageFont.load_default()

  dr.text((pad, 16), f"Disney Moana Island (OpenUSD v2.1, 22.30B Instanced Tris) — Native 8K (7680x4320) on 4-Chip TPU v5p", fill=(245, 248, 255), font=font_lg)
  dr.text((pad, 48), f"8K Latency: {stats_8k['ms']:.1f} ms ({stats_8k['mrays_per_s']:.2f} Mrays/s) | HBM: {stats_8k['hbm_gb']:.2f} GiB | host_callbacks=0, gather_ops=0, send_recv_ops=0", fill=(110, 235, 200), font=font_sm)

  panels = [
      (0, 0, p1, "1. shotCam: TPU v5p 8K Polished (Left) vs Disney Hyperion Path-Traced Ref (Right)"),
      (1, 0, p2, "2. beachCam (Native 8K): Sunlit Motunui Beach Sand, Hibiscus & Coconut Palm Canopy"),
      (0, 1, p3, "3. palmsCam (Native 8K): Volcanic Mountain Ridge Shadows & Tropical Canopy"),
      (1, 1, p4, "4. 1:1 Native 8K Pixel Crop (1440x810 from 7680x4320): Palm Fronds & Coral Lagoon"),
  ]
  for gx, gy, pim, title in panels:
    x = pad + gx * (pw + pad)
    y = top_bar + pad + gy * (ph + hdr + pad)
    dr.text((x, y + 12), title, fill=(230, 238, 250), font=font_sm)
    canvas.paste(pim, (x, y + hdr))

  os.makedirs(os.path.dirname(out_grid_path), exist_ok=True)
  canvas.save(out_grid_path)
  print(f"Saved 8K multi-angle grid to {out_grid_path} ({canvas.size})", flush=True)


def annotate_mjx_beach_4k(raw_4k_path: str, out_4k_path: str, stats_mjx: Dict[str, Any]):
  """Save crisp 4K Moana x MJX figure with subtle bottom cinema metadata strip."""
  im = Image.open(raw_4k_path).convert("RGB")
  # Save clean annotated version without cropping the 3840x2160 render
  w, h = im.size
  bar_h = 56
  canvas = Image.new("RGB", (w, h + bar_h), (12, 16, 22))
  canvas.paste(im, (0, 0))
  dr = ImageDraw.Draw(canvas)
  try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
  except Exception:
    font = ImageFont.load_default()
  dr.text(
      (28, h + 14),
      f"Physical AI x Disney Moana (22.30B Tris) @ Native 4K ({w}x{h})  |  MJX Quadruped-12DOF + Humanoid-21DOF  |  "
      f"{stats_mjx['ms']:.1f} ms ({stats_mjx['fps']:.2f} FPS)  |  Exact Z-Occlusion + 2-Way Shadows  |  host_callbacks=0, gather_ops=0",
      fill=(235, 245, 255),
      font=font,
  )
  os.makedirs(os.path.dirname(out_4k_path), exist_ok=True)
  canvas.save(out_4k_path)
  print(f"Saved Moana x MJX 4K figure to {out_4k_path} ({canvas.size})", flush=True)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--mode", choices=["preview_1k", "full_8k_and_mjx"], default="full_8k_and_mjx")
  ap.add_argument("--stage", default="/dev/shm/zaychik-moana/island/usd/island.usda")
  ap.add_argument("--exr", default="/dev/shm/zaychik-moana/island/textures/islandsun.exr")
  ap.add_argument("--hyperion-ref", default="/dev/shm/zaychik-moana/island/ref/hyperionRenders/shotCam_hyperion.png")
  ap.add_argument("--cache-dir", default="/dev/shm/zaychik-tpu-mjx-w2/moana_bvh_cache")
  ap.add_argument("--out-dir", default="/dev/shm/zaychik-tpu-mjx-w2/out")
  args = ap.parse_args()

  os.makedirs(args.out_dir, exist_ok=True)
  devs = jax.devices()
  assert len(devs) == 4, f"Expected 4 TPU chips, got {len(devs)}"
  mesh = Mesh(np.array(devs), ("spatial",))
  rep = NamedSharding(mesh, P())

  compiled = _load_compiled_cache(args.cache_dir, devs, 2160, 3840)
  assert compiled is not None, f"Missing Moana BVH cache at {args.cache_dir}"

  env_np = load_islandsun_hdri_polished(args.exr, H_env=512, W_env=1024)
  env_bf16 = jax.device_put(jnp.asarray(env_np, dtype=jnp.bfloat16), rep)

  cams = extract_all_moana_cameras(args.stage, target_aspect=16.0 / 9.0)
  print("Extracted Moana cameras:", list(cams.keys()), flush=True)

  # Robot placement on sunlit Motunui beach in front of beachCam!
  # beachCam pos = [-12.967, -4.617, 2.218], fwd = [0.931, 0.357, -0.079], right = [0.358, -0.934, 0.0]
  bc = cams["beachCam"]
  bc_pos = bc["cam_xpos"][0]
  bc_fwd = -bc["cam_xmat"][0, :, 2]
  bc_right = bc["cam_xmat"][0, :, 0]
  # Place Humanoid & Quadruped side-by-side 6.5m-6.8m ahead of beachCam on sunlit sand
  hum_pos = bc_pos + 6.8 * bc_fwd - 0.72 * bc_right
  quad_pos = bc_pos + 6.5 * bc_fwd + 0.55 * bc_right
  mjx_info = build_grounded_mjx_geom_buffers(
      args.stage,
      hum_xy=(float(hum_pos[0]), float(hum_pos[1])),
      hum_yaw=2.75,
      quad_xy=(float(quad_pos[0]), float(quad_pos[1])),
      quad_yaw=2.35,
  )

  def _run_cam(fn, base_args, mjx_args, cam_dict):
    cxp = jax.device_put(cam_dict["cam_xpos"], rep)
    cxm = jax.device_put(cam_dict["cam_xmat"], rep)
    thf = jax.device_put(cam_dict["tan_half_fovy"], rep)
    return fn(*base_args, cxp, cxm, thf, env_bf16, *mjx_args)

  if args.mode == "preview_1k":
    H_p, W_p = 720, 1280
    fn_mjx_p, ba_mp, ma_mp, _ = make_moana_polished_renderer(
        compiled, env_bf16, devs, H_p, W_p, n_subbands=1, use_shadows=True, mjx_info=mjx_info
    )
    rgb_m = _run_cam(fn_mjx_p, ba_mp, ma_mp, cams["beachCam"])
    jax.block_until_ready(rgb_m)
    pm_out = os.path.join(args.out_dir, "preview_mjx_beach_1k.png")
    Image.fromarray(np.clip(np.asarray(rgb_m[0]) * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(pm_out)
    print(f"Saved MJX beach preview {pm_out}", flush=True)
    return

  # -----------------------------------------------------------------------
  # 1. Native 8K (7680x4320) Moana Polished Benchmark + 3-Angle 8K Renders
  # -----------------------------------------------------------------------
  H_8k, W_8k = 4320, 7680
  fn_8k, ba_8k, ma_8k, _ = make_moana_polished_renderer(
      compiled, env_bf16, devs, H_8k, W_8k, n_subbands=8, use_shadows=True, mjx_info=None
  )
  c_shot = cams["shotCam"]
  lower_args_8k = (
      *ba_8k,
      jax.device_put(c_shot["cam_xpos"], rep),
      jax.device_put(c_shot["cam_xmat"], rep),
      jax.device_put(c_shot["tan_half_fovy"], rep),
      env_bf16,
      *ma_8k,
  )
  htxt_8k = fn_8k.lower(*lower_args_8k).as_text()
  hlo_8k = inspect_hlo_ops(htxt_8k)
  hlo_8k["send_recv_ops"] = len(re.findall(r"\bsend\(|\brecv\(", htxt_8k, re.I))
  print("8K HLO audit:", hlo_8k, flush=True)
  assert hlo_8k["host_callbacks"] == 0 and hlo_8k["gather_ops"] == 0 and hlo_8k["send_recv_ops"] == 0, hlo_8k

  tc0 = time.perf_counter()
  rgb_shot_8k = _run_cam(fn_8k, ba_8k, ma_8k, cams["shotCam"])
  jax.block_until_ready(rgb_shot_8k)
  xla_8k_s = time.perf_counter() - tc0

  shot_8k_path = os.path.join(args.out_dir, "moana_shotCam_8k.png")
  Image.fromarray(np.clip(np.asarray(rgb_shot_8k[0]) * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(shot_8k_path)
  print(f"Saved shotCam 8K ({xla_8k_s:.2f}s compile+1st)", flush=True)

  n_steps_8k = 3
  t1 = time.perf_counter()
  for _ in range(n_steps_8k):
    rgb_tmp = _run_cam(fn_8k, ba_8k, ma_8k, cams["shotCam"])
  jax.block_until_ready(rgb_tmp)
  ms_8k = (time.perf_counter() - t1) * 1e3 / n_steps_8k

  rgb_beach_8k = _run_cam(fn_8k, ba_8k, ma_8k, cams["beachCam"])
  jax.block_until_ready(rgb_beach_8k)
  beach_8k_path = os.path.join(args.out_dir, "moana_beachCam_8k.png")
  Image.fromarray(np.clip(np.asarray(rgb_beach_8k[0]) * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(beach_8k_path)

  # Render palmsCam (mountain ridge + palm canopy) at Native 8K using exact same compiled fn!
  ridge_cam_name = "palmsCam" if "palmsCam" in cams else "dunesACam"
  rgb_ridge_8k = _run_cam(fn_8k, ba_8k, ma_8k, cams[ridge_cam_name])
  jax.block_until_ready(rgb_ridge_8k)
  ridge_8k_path = os.path.join(args.out_dir, f"moana_{ridge_cam_name}_8k.png")
  Image.fromarray(np.clip(np.asarray(rgb_ridge_8k[0]) * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(ridge_8k_path)

  stats_8k = {
      "res": "4320x7680",
      "pixels": 4320 * 7680,
      "n_chips": len(devs),
      "xla_compile_s": round(xla_8k_s, 2),
      "ms": round(ms_8k, 2),
      "fps": round(1e3 / ms_8k, 3),
      "mrays_per_s": round((H_8k * W_8k * 2.0) / (ms_8k * 1e-3) / 1e6, 2),
      "hbm_gb": round(compiled["hbm_bytes"] / (1024 ** 3), 3),
      "unique_tris": compiled["unique_tris"],
      "inst_blas_tris": compiled["inst_blas_tris"],
      "inst_effective_tris": compiled["inst_effective_tris"],
      "hlo": hlo_8k,
  }
  print(f"8K BENCH: {ms_8k:.2f} ms ({stats_8k['mrays_per_s']} Mrays/s)", flush=True)

  grid_out_path = os.path.join(args.out_dir, "v3_w2_moana_8k_multiangle.png")
  make_multiangle_grid(shot_8k_path, args.hyperion_ref, beach_8k_path, ridge_8k_path, grid_out_path, stats_8k)

  # -----------------------------------------------------------------------
  # 2. Physical AI Crossover: MJX Quadruped + Humanoid on Motunui Beach @ 4K
  # -----------------------------------------------------------------------
  H_4k, W_4k = 2160, 3840
  fn_mjx, ba_m, ma_m, _ = make_moana_polished_renderer(
      compiled, env_bf16, devs, H_4k, W_4k, n_subbands=2, use_shadows=True, mjx_info=mjx_info
  )
  c_beach = cams["beachCam"]
  lower_args_mjx = (
      *ba_m,
      jax.device_put(c_beach["cam_xpos"], rep),
      jax.device_put(c_beach["cam_xmat"], rep),
      jax.device_put(c_beach["tan_half_fovy"], rep),
      env_bf16,
      *ma_m,
  )
  htxt_mjx = fn_mjx.lower(*lower_args_mjx).as_text()
  hlo_mjx = inspect_hlo_ops(htxt_mjx)
  hlo_mjx["send_recv_ops"] = len(re.findall(r"\bsend\(|\brecv\(", htxt_mjx, re.I))
  print("4K Moana x MJX HLO audit:", hlo_mjx, flush=True)
  assert hlo_mjx["host_callbacks"] == 0 and hlo_mjx["gather_ops"] == 0 and hlo_mjx["send_recv_ops"] == 0, hlo_mjx

  tc1 = time.perf_counter()
  rgb_mjx_4k = _run_cam(fn_mjx, ba_m, ma_m, cams["beachCam"])
  jax.block_until_ready(rgb_mjx_4k)
  xla_mjx_s = time.perf_counter() - tc1

  raw_mjx_path = os.path.join(args.out_dir, "moana_mjx_beach_4k_raw.png")
  Image.fromarray(np.clip(np.asarray(rgb_mjx_4k[0]) * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(raw_mjx_path)

  n_steps_4k = 3
  t2 = time.perf_counter()
  for _ in range(n_steps_4k):
    rgb_m_tmp = _run_cam(fn_mjx, ba_m, ma_m, cams["beachCam"])
  jax.block_until_ready(rgb_m_tmp)
  ms_mjx = (time.perf_counter() - t2) * 1e3 / n_steps_4k

  stats_mjx = {
      "res": "2160x3840",
      "pixels": 2160 * 3840,
      "n_chips": len(devs),
      "mjx_ngeom": mjx_info["ngeom"],
      "hum_xy": [round(float(hum_pos[0]), 3), round(float(hum_pos[1]), 3)],
      "hum_z_ground": round(mjx_info["hum_z"], 4),
      "quad_xy": [round(float(quad_pos[0]), 3), round(float(quad_pos[1]), 3)],
      "quad_z_ground": round(mjx_info["quad_z"], 4),
      "xla_compile_s": round(xla_mjx_s, 2),
      "ms": round(ms_mjx, 2),
      "fps": round(1e3 / ms_mjx, 3),
      "mrays_per_s": round((H_4k * W_4k * 2.0) / (ms_mjx * 1e-3) / 1e6, 2),
      "hlo": hlo_mjx,
  }
  print(f"4K MOANA x MJX BENCH: {ms_mjx:.2f} ms ({stats_mjx['mrays_per_s']} Mrays/s)", flush=True)

  mjx_out_path = os.path.join(args.out_dir, "v3_w2_moana_mjx_beach_4k.png")
  annotate_mjx_beach_4k(raw_mjx_path, mjx_out_path, stats_mjx)

  full_metrics = {
      "name": "Wave5_Moana_8K_Polish_And_MJX_Beach_4K",
      "stage_path": args.stage,
      "moana_8k": stats_8k,
      "moana_mjx_beach_4k": stats_mjx,
      "artifacts": {
          "multiangle_8k_png": "out/figures/v3_w2_moana_8k_multiangle.png",
          "mjx_beach_4k_png": "out/figures/v3_w2_moana_mjx_beach_4k.png",
      },
  }
  json_out_path = os.path.join(args.out_dir, "v3_w2_wave5_moana_8k_mjx.json")
  with open(json_out_path, "w") as f:
    json.dump(full_metrics, f, indent=2)
  print(f"Saved metrics to {json_out_path}", flush=True)


if __name__ == "__main__":
  main()
