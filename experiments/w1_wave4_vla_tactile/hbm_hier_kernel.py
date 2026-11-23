"""HBM-Streamed Morton Cluster-BVH Mosaic Kernel for G=2,048..4,096+ Moving Primitives.

Eliminates the 16 MB TPU v5p scoped VMEM bottleneck by keeping per-type micro-cluster
geometry payloads in HBM (`pltpu.HBM`) and asynchronously DMA-copying only ray-hit
32-primitive micro-clusters into a single reusable 92 KB VMEM scratch buffer via
`pltpu.make_async_copy` inside `jax.lax.cond(u_hit, ...)`.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
try:
  from jax import shard_map
except ImportError:
  from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
import numpy as np

from tpu_mjx_render.adapters.isaac_replicator import (
    CLUSTER_K,
    MACRO_M,
    ReplicatorBatch,
    ReplicatorCfg,
    TEX_CHECKER,
    TEX_SOLID,
    _agx_tonemap_rgb_components,
    _apply_replicator_local,
    _build_unit_tangent_rays,
    _compute_motion_vectors_zero_gather,
    _default_category_masks,
    _default_semantic_classes,
    _eval_procedural_tex_vmem,
    _morton_codes_3d,
    _pack_camera_and_light_dr,
    _resolve_cam_ids,
    _shard_map_kwargs,
    _sphere_tile_hit,
)
from tpu_mjx_render.pallas_v2 import _compute_geom_bound_sq, _ray_one_geom_v2
from tpu_mjx_render.primitives import (
    GEOM_BOX,
    GEOM_CAPSULE,
    GEOM_CYLINDER,
    GEOM_ELLIPSOID,
    GEOM_PLANE,
    GEOM_SPHERE,
)
from tpu_mjx_render.renderer import RenderSceneSpec


def _build_type_hierarchy_hbm(
    pos_t: jax.Array,      # (B, G_t, 3)
    mat_t: jax.Array,      # (B, G_t, 3, 3)
    size_t: jax.Array,     # (G_t, 3)
    rb_t: jax.Array,       # (G_t,)
    gid_t: jax.Array,      # (G_t,) int32
    pid_t: jax.Array,      # (G_t,) int32
    rgb_t: jax.Array,      # (B, G_t, 3)
    rough_t: jax.Array,    # (B, G_t)
    spec_t: jax.Array,     # (B, G_t)
    block_b: int,
    B_pad: int,
) -> Tuple[jax.Array, jax.Array, int, int]:
  """Build HBM-streamable 3D micro-cluster payload (n_b_blk*n_micro, 23*block_b, 128) + VMEM c_meta."""
  B, G_t, _ = pos_t.shape
  n_b_blk = B_pad // block_b

  codes = _morton_codes_3d(pos_t)
  order = jnp.argsort(codes, axis=-1)

  pos_s = jnp.take_along_axis(pos_t, order[..., None], axis=1)
  mat_s = jnp.take_along_axis(mat_t.reshape(B, G_t, 9), order[..., None], axis=1)
  sz_s = jnp.take_along_axis(jnp.broadcast_to(size_t[None, :, :], (B, G_t, 3)), order[..., None], axis=1)
  rb_s = jnp.take_along_axis(jnp.broadcast_to(rb_t[None, :], (B, G_t)), order, axis=1)
  gid_s = jnp.take_along_axis(jnp.broadcast_to(gid_t[None, :].astype(jnp.float32), (B, G_t)), order, axis=1)
  pid_s = jnp.take_along_axis(jnp.broadcast_to(pid_t[None, :].astype(jnp.float32), (B, G_t)), order, axis=1)
  rgb_s = jnp.take_along_axis(rgb_t, order[..., None], axis=1)
  rough_s = jnp.take_along_axis(rough_t, order, axis=1)
  spec_s = jnp.take_along_axis(spec_t, order, axis=1)
  valid_s = jnp.ones((B, G_t), dtype=jnp.bool_)

  n_micro_raw = (G_t + CLUSTER_K - 1) // CLUSTER_K
  n_micro = n_micro_raw
  G_pad = n_micro * CLUSTER_K
  pad_g = G_pad - G_t

  if pad_g > 0:
    pos_s = jnp.pad(pos_s, ((0, 0), (0, pad_g), (0, 0)))
    mat_s = jnp.pad(mat_s, ((0, 0), (0, pad_g), (0, 0)))
    sz_s = jnp.pad(sz_s, ((0, 0), (0, pad_g), (0, 0)))
    rb_s = jnp.pad(rb_s, ((0, 0), (0, pad_g)), constant_values=-1.0)
    gid_s = jnp.pad(gid_s, ((0, 0), (0, pad_g)), constant_values=-1.0)
    pid_s = jnp.pad(pid_s, ((0, 0), (0, pad_g)), constant_values=0.0)
    rgb_s = jnp.pad(rgb_s, ((0, 0), (0, pad_g), (0, 0)))
    rough_s = jnp.pad(rough_s, ((0, 0), (0, pad_g)), constant_values=0.5)
    spec_s = jnp.pad(spec_s, ((0, 0), (0, pad_g)), constant_values=0.2)
    valid_s = jnp.pad(valid_s, ((0, 0), (0, pad_g)), constant_values=False)

  pos_c = pos_s.reshape(B, n_micro, CLUSTER_K, 3)
  rb_c = rb_s.reshape(B, n_micro, CLUSTER_K)
  val_c = valid_s.reshape(B, n_micro, CLUSTER_K)
  any_val_c = jnp.any(val_c, axis=2)

  c_min = jnp.min(jnp.where(val_c[..., None], pos_c, 1.0e9), axis=2)
  c_max = jnp.max(jnp.where(val_c[..., None], pos_c, -1.0e9), axis=2)
  c_center = jnp.where(any_val_c[..., None], 0.5 * (c_min + c_max), 0.0)
  c_dists = jnp.linalg.norm(pos_c - c_center[:, :, None, :], axis=-1) + rb_c
  c_rad = jnp.max(jnp.where(val_c, c_dists, 0.0), axis=2) + 1.0e-4
  c_rad_sq = jnp.where(any_val_c, c_rad * c_rad, -1.0)

  rb_sq_s = jnp.where(valid_s, rb_s * rb_s, -1.0)
  g23 = jnp.concatenate([
      pos_s, mat_s, sz_s, rb_sq_s[..., None], gid_s[..., None],
      rgb_s, rough_s[..., None], spec_s[..., None], pid_s[..., None],
  ], axis=-1)  # (B, n_micro*K, 23)
  g23_c = g23.reshape(B, n_micro, CLUSTER_K, 23).transpose(0, 1, 3, 2)  # (B, n_micro, 23, 32)
  g23_pad = jnp.pad(g23_c, ((0, 0), (0, 0), (0, 0), (0, 128 - CLUSTER_K)))  # (B, n_micro, 23, 128)
  if B_pad > B:
    g23_pad = jnp.pad(g23_pad, ((0, B_pad - B), (0, 0), (0, 0), (0, 0)))

  # (n_b_blk, block_b, n_micro, 23, 128) -> (n_b_blk * n_micro, 23 * block_b, 128)
  gf_hbm = (
      g23_pad.reshape(n_b_blk, block_b, n_micro, 23, 128)
      .transpose(0, 2, 3, 1, 4)
      .reshape(n_b_blk * n_micro, 23 * block_b, 128)
  )

  c_meta = jnp.zeros((B, n_micro, 128), dtype=jnp.float32)
  c_meta = c_meta.at[:, :, 0:3].set(c_center)
  c_meta = c_meta.at[:, :, 3].set(c_rad_sq)
  if B_pad > B:
    c_meta = jnp.pad(c_meta, ((0, B_pad - B), (0, 0), (0, 0)))
  # (n_b_blk * n_micro * block_b, 128)
  cmeta_vmem = (
      c_meta.reshape(n_b_blk, block_b, n_micro, 128)
      .transpose(0, 2, 1, 3)
      .reshape(n_b_blk * n_micro * block_b, 128)
  )
  return gf_hbm, cmeta_vmem, n_micro_raw, n_micro


def _pallas_hbm_hier_kernel(
    spec: RenderSceneSpec,
    mjx_data_batch: Any,
    rep_batch: Optional[ReplicatorBatch],
    cam_slot: int,
    geom_sem_ids: Tuple[int, ...],
    geom_body_ids: Tuple[int, ...],
    floor_gids: Tuple[int, ...],
    table_gids: Tuple[int, ...],
    tonemap: str = "agx",
    out_dtype: Any = jnp.bfloat16,
    block_b: int = 8,
    block_r: int = 256,
    emit_aux: bool = True,
    interpret: bool = False,
    cam_fovy_override: Optional[jax.Array] = None,
) -> Dict[str, jax.Array]:
  """HBM-streamed Hierarchical Morton Cluster-BVH Mosaic Kernel (fits G=4,096+ in < 2.5 MB VMEM)."""
  B = int(mjx_data_batch.cam_xpos.shape[0])
  H, W = int(spec.height), int(spec.width)
  R = H * W
  gids = np.asarray(spec.enabled_geom_ids, dtype=np.int32)

  B_pad = ((B + block_b - 1) // block_b) * block_b
  R_pad = ((R + block_r - 1) // block_r) * block_r
  n_b_blk = B_pad // block_b
  n_r_blk = R_pad // block_r

  cam_128, light_128 = _pack_camera_and_light_dr(
      spec, rep_batch, mjx_data_batch, cam_slot, block_b, B, B_pad, cam_fovy_override=cam_fovy_override
  )
  lray_dev = _build_unit_tangent_rays(H, W, R_pad)

  geom_size_np = np.asarray(spec.geom_size, dtype=np.float32)
  rb_all = np.sqrt(np.maximum(_compute_geom_bound_sq(spec, gids), 0.0)).astype(np.float32)

  if rep_batch is not None:
    all_rgb = rep_batch.geom_rgba[..., :3]
    all_rough = rep_batch.geom_roughness
    all_spec = rep_batch.geom_specular
  else:
    all_rgb = jnp.broadcast_to(jnp.asarray(spec.geom_rgba[:, :3], jnp.float32)[None, :, :], (B, spec.ngeom, 3))
    base_rough = jnp.clip(1.0 - jnp.asarray(spec.geom_shin, jnp.float32) * 1.4, 0.14, 0.90)
    all_rough = jnp.broadcast_to(base_rough[None, :], (B, spec.ngeom))
    all_spec = jnp.broadcast_to(jnp.asarray(spec.geom_spec, jnp.float32)[None, :], (B, spec.ngeom))

  sem_by_gid = {int(g): int(geom_sem_ids[k]) for k, g in enumerate(gids)}
  bid_by_gid = {int(g): int(geom_body_ids[k]) for k, g in enumerate(gids)}

  plane_gids = [int(g) for g in gids if int(spec.geom_type[g]) == GEOM_PLANE]
  P_cnt = len(plane_gids)
  P_pad = ((max(P_cnt, 1) + 127) // 128) * 128
  if P_cnt > 0:
    p_idx = np.asarray(plane_gids, dtype=np.int32)
    p_pos = mjx_data_batch.geom_xpos[:, p_idx, :]
    p_mat = mjx_data_batch.geom_xmat[:, p_idx, :, :].reshape(B, P_cnt, 9)
    p_sz = jnp.broadcast_to(jnp.asarray(geom_size_np[p_idx])[None, :, :], (B, P_cnt, 3))
    p_gid = jnp.broadcast_to(jnp.asarray(p_idx, jnp.float32)[None, :, None], (B, P_cnt, 1))
    p_pid = jnp.broadcast_to(
        jnp.asarray([bid_by_gid[g] * 64 + sem_by_gid[g] for g in plane_gids], jnp.float32)[None, :, None],
        (B, P_cnt, 1),
    )
    p_rgb = all_rgb[:, p_idx, :]
    p_rough = all_rough[:, p_idx][..., None]
    p_spec = all_spec[:, p_idx][..., None]
    p_22 = jnp.concatenate([p_pos, p_mat, p_sz, p_gid, p_rgb, p_rough, p_spec, p_pid], axis=-1)
    if B_pad > B or P_pad > P_cnt:
      p_22 = jnp.pad(p_22, ((0, B_pad - B), (0, P_pad - P_cnt), (0, 0)))
    p_24 = jnp.pad(p_22, ((0, 0), (0, 0), (0, 2)))
    plane_fields = (
        p_24.reshape(n_b_blk, block_b, P_pad, 24)
        .transpose(0, 3, 1, 2)
        .reshape(n_b_blk * 24 * block_b, P_pad)
    )
  else:
    plane_fields = jnp.zeros((n_b_blk * 24 * block_b, P_pad), dtype=jnp.float32)

  plane_fx = tuple(float(geom_size_np[g, 0]) > 0.0 for g in plane_gids)
  plane_fy = tuple(float(geom_size_np[g, 1]) > 0.0 for g in plane_gids)

  active_types: List[int] = []
  gf_hbm_list: List[jax.Array] = []
  cmeta_list: List[jax.Array] = []
  n_micro_list: List[int] = []

  for gt in (GEOM_BOX, GEOM_SPHERE, GEOM_CAPSULE, GEOM_CYLINDER, GEOM_ELLIPSOID):
    idxs = [k for k, g in enumerate(gids) if int(spec.geom_type[g]) == gt]
    if not idxs:
      continue
    g_sub = gids[idxs]
    pid_sub = np.asarray([bid_by_gid[int(g)] * 64 + sem_by_gid[int(g)] for g in g_sub], dtype=np.int32)
    gf_hbm_t, cmeta_t, _, n_micro_t = _build_type_hierarchy_hbm(
        mjx_data_batch.geom_xpos[:, g_sub, :],
        mjx_data_batch.geom_xmat[:, g_sub, :, :],
        jnp.asarray(geom_size_np[g_sub], dtype=jnp.float32),
        jnp.asarray(rb_all[idxs], dtype=jnp.float32),
        jnp.asarray(g_sub, dtype=jnp.int32),
        jnp.asarray(pid_sub, dtype=jnp.int32),
        all_rgb[:, g_sub, :],
        all_rough[:, g_sub],
        all_spec[:, g_sub],
        block_b, B_pad,
    )
    active_types.append(gt)
    gf_hbm_list.append(gf_hbm_t)
    cmeta_list.append(cmeta_t)
    n_micro_list.append(n_micro_t)

  n_types = len(active_types)
  znear_f = float(spec.znear)
  zfar_f = float(spec.zfar) if spec.zfar > 0.0 else 50.0
  l_type_0 = int(spec.light_type[0]) if spec.nlight > 0 else 0
  l_att_0 = tuple(float(x) for x in spec.light_attenuation[0]) if spec.nlight > 0 else (1.0, 0.0, 0.0)
  do_sh_0 = bool(spec.use_shadows) and (bool(spec.light_castshadow[0]) if spec.nlight > 0 else True)
  shadow_frac = float(spec.shadow_light_fraction)
  headlight_active = bool(spec.headlight_active)
  hl_amb = tuple(float(x) for x in spec.headlight_ambient)
  hl_diff = tuple(float(x) for x in spec.headlight_diffuse)
  use_agx = (tonemap == "agx")
  floor_gid_0 = int(floor_gids[0]) if floor_gids else -999
  table_gid_0 = int(table_gids[0]) if table_gids else -999

  n_outs = 9 if emit_aux else 4

  def _kernel(*refs):
    cam = refs[0][...]
    lr = refs[1][...]
    pf = refs[2][...]
    linfo = refs[3][...]
    cmeta_refs = refs[4 : 4 + n_types]
    gf_hbm_refs = refs[4 + n_types : 4 + 2 * n_types]
    out_start = 4 + 2 * n_types
    out_r_ref, out_g_ref, out_b_ref, out_d_ref = refs[out_start : out_start + 4]
    if emit_aux:
      out_nx_ref, out_ny_ref, out_nz_ref, out_gid_ref, out_pid_ref = refs[out_start + 4 : out_start + 9]
    c_vmem_ref, dma_sem = refs[out_start + n_outs], refs[out_start + n_outs + 1]

    bi = pl.program_id(0)

    tx_r = lr[0:1, :]
    ty_r = lr[1:2, :]
    tan_hf = cam[:, 12:13]
    ldx_raw = tx_r * tan_hf
    ldy_raw = ty_r * tan_hf
    inv_ln = 1.0 / jnp.sqrt(ldx_raw * ldx_raw + ldy_raw * ldy_raw + 1.0)
    neg_z = inv_ln
    ldx, ldy, ldz = ldx_raw * inv_ln, ldy_raw * inv_ln, -inv_ln
    inv_cos = znear_f / neg_z
    max_d = (zfar_f - znear_f) / neg_z

    cpx, cpy, cpz = cam[:, 0:1], cam[:, 1:2], cam[:, 2:3]
    c00, c01, c02 = cam[:, 3:4], cam[:, 4:5], cam[:, 5:6]
    c10, c11, c12 = cam[:, 6:7], cam[:, 7:8], cam[:, 8:9]
    c20, c21, c22 = cam[:, 9:10], cam[:, 10:11], cam[:, 11:12]

    rdx_raw = c00 * ldx + c01 * ldy + c02 * ldz
    rdy_raw = c10 * ldx + c11 * ldy + c12 * ldz
    rdz_raw = c20 * ldx + c21 * ldy + c22 * ldz
    inv_rn = 1.0 / jnp.maximum(jnp.sqrt(rdx_raw * rdx_raw + rdy_raw * rdy_raw + rdz_raw * rdz_raw), 1e-12)
    rdx, rdy, rdz = rdx_raw * inv_rn, rdy_raw * inv_rn, rdz_raw * inv_rn

    rox = cpx + inv_cos * rdx
    roy = cpy + inv_cos * rdy
    roz = cpz + inv_cos * rdz

    best_d = max_d
    best_gid = jnp.full((block_b, block_r), -1, dtype=jnp.int32)
    best_pid = jnp.zeros((block_b, block_r), dtype=jnp.int32)
    best_nx = jnp.zeros((block_b, block_r), dtype=jnp.float32)
    best_ny = jnp.zeros((block_b, block_r), dtype=jnp.float32)
    best_nz = jnp.zeros((block_b, block_r), dtype=jnp.float32)
    alb_r = jnp.zeros((block_b, block_r), dtype=jnp.float32)
    alb_g = jnp.zeros((block_b, block_r), dtype=jnp.float32)
    alb_b = jnp.zeros((block_b, block_r), dtype=jnp.float32)
    roughness = jnp.full((block_b, block_r), 0.45, dtype=jnp.float32)
    spec_coef = jnp.zeros((block_b, block_r), dtype=jnp.float32)

    for pk_i, (pfx, pfy) in enumerate(zip(plane_fx, plane_fy)):
      px = pf[0 * block_b : 1 * block_b, pk_i : pk_i + 1]
      py = pf[1 * block_b : 2 * block_b, pk_i : pk_i + 1]
      pz = pf[2 * block_b : 3 * block_b, pk_i : pk_i + 1]
      m00, m01, m02 = pf[3 * block_b : 4 * block_b, pk_i : pk_i + 1], pf[4 * block_b : 5 * block_b, pk_i : pk_i + 1], pf[5 * block_b : 6 * block_b, pk_i : pk_i + 1]
      m10, m11, m12 = pf[6 * block_b : 7 * block_b, pk_i : pk_i + 1], pf[7 * block_b : 8 * block_b, pk_i : pk_i + 1], pf[8 * block_b : 9 * block_b, pk_i : pk_i + 1]
      m20, m21, m22 = pf[9 * block_b : 10 * block_b, pk_i : pk_i + 1], pf[10 * block_b : 11 * block_b, pk_i : pk_i + 1], pf[11 * block_b : 12 * block_b, pk_i : pk_i + 1]
      sx, sy, sz = pf[12 * block_b : 13 * block_b, pk_i : pk_i + 1], pf[13 * block_b : 14 * block_b, pk_i : pk_i + 1], pf[14 * block_b : 15 * block_b, pk_i : pk_i + 1]
      gid_f = pf[15 * block_b : 16 * block_b, pk_i : pk_i + 1]
      cd, cnx, cny, cnz = _ray_one_geom_v2(
          GEOM_PLANE, px, py, pz, m00, m01, m02, m10, m11, m12, m20, m21, m22,
          sx, sy, sz, 1.0e20, rox, roy, roz, rdx, rdy, rdz,
          plane_finite_x=pfx, plane_finite_y=pfy, compute_normal=True,
      )
      imp = cd < best_d
      best_d = jnp.where(imp, cd, best_d)
      best_gid = jnp.where(imp, gid_f.astype(jnp.int32), best_gid)
      best_nx = jnp.where(imp, cnx, best_nx)
      best_ny = jnp.where(imp, cny, best_ny)
      best_nz = jnp.where(imp, cnz, best_nz)
      alb_r = jnp.where(imp, pf[16 * block_b : 17 * block_b, pk_i : pk_i + 1], alb_r)
      alb_g = jnp.where(imp, pf[17 * block_b : 18 * block_b, pk_i : pk_i + 1], alb_g)
      alb_b = jnp.where(imp, pf[18 * block_b : 19 * block_b, pk_i : pk_i + 1], alb_b)
      roughness = jnp.where(imp, pf[19 * block_b : 20 * block_b, pk_i : pk_i + 1], roughness)
      spec_coef = jnp.where(imp, pf[20 * block_b : 21 * block_b, pk_i : pk_i + 1], spec_coef)
      best_pid = jnp.where(imp, pf[21 * block_b : 22 * block_b, pk_i : pk_i + 1].astype(jnp.int32), best_pid)

    def _eval_micro_primary(gt, c_slice, state):
      bd, bgid, bpid, bnx, bny, bnz, ar, ag, ab, rgh, spc = state
      for k in range(CLUSTER_K):
        px = c_slice[0 * block_b : 1 * block_b, k : k + 1]
        py = c_slice[1 * block_b : 2 * block_b, k : k + 1]
        pz = c_slice[2 * block_b : 3 * block_b, k : k + 1]
        m00, m01, m02 = c_slice[3 * block_b : 4 * block_b, k : k + 1], c_slice[4 * block_b : 5 * block_b, k : k + 1], c_slice[5 * block_b : 6 * block_b, k : k + 1]
        m10, m11, m12 = c_slice[6 * block_b : 7 * block_b, k : k + 1], c_slice[7 * block_b : 8 * block_b, k : k + 1], c_slice[8 * block_b : 9 * block_b, k : k + 1]
        m20, m21, m22 = c_slice[9 * block_b : 10 * block_b, k : k + 1], c_slice[10 * block_b : 11 * block_b, k : k + 1], c_slice[11 * block_b : 12 * block_b, k : k + 1]
        sx, sy, sz = c_slice[12 * block_b : 13 * block_b, k : k + 1], c_slice[13 * block_b : 14 * block_b, k : k + 1], c_slice[14 * block_b : 15 * block_b, k : k + 1]
        rb_sq = c_slice[15 * block_b : 16 * block_b, k : k + 1]
        gid_f = c_slice[16 * block_b : 17 * block_b, k : k + 1]
        cd, cnx, cny, cnz = _ray_one_geom_v2(
            gt, px, py, pz, m00, m01, m02, m10, m11, m12, m20, m21, m22,
            sx, sy, sz, rb_sq, rox, roy, roz, rdx, rdy, rdz, compute_normal=True,
        )
        imp = (rb_sq >= 0.0) & (cd < bd)
        bd = jnp.where(imp, cd, bd)
        bgid = jnp.where(imp, gid_f.astype(jnp.int32), bgid)
        bnx = jnp.where(imp, cnx, bnx)
        bny = jnp.where(imp, cny, bny)
        bnz = jnp.where(imp, cnz, bnz)
        ar = jnp.where(imp, c_slice[17 * block_b : 18 * block_b, k : k + 1], ar)
        ag = jnp.where(imp, c_slice[18 * block_b : 19 * block_b, k : k + 1], ag)
        ab = jnp.where(imp, c_slice[19 * block_b : 20 * block_b, k : k + 1], ab)
        rgh = jnp.where(imp, c_slice[20 * block_b : 21 * block_b, k : k + 1], rgh)
        spc = jnp.where(imp, c_slice[21 * block_b : 22 * block_b, k : k + 1], spc)
        bpid = jnp.where(imp, c_slice[22 * block_b : 23 * block_b, k : k + 1].astype(jnp.int32), bpid)
      return (bd, bgid, bpid, bnx, bny, bnz, ar, ag, ab, rgh, spc)

    state_p = (
        best_d, best_gid, best_pid, best_nx, best_ny, best_nz,
        alb_r, alb_g, alb_b, roughness, spec_coef,
    )
    for t_i, gt in enumerate(active_types):
      cm_ref_t = cmeta_refs[t_i]
      gf_hbm_t = gf_hbm_refs[t_i]
      n_micro_t = n_micro_list[t_i]

      def _run_micro_hbm(c_idx, st, _gt=gt, _cm=cm_ref_t, _gf_hbm=gf_hbm_t, _nm=n_micro_t):
        c_meta = _cm[pl.ds(c_idx * block_b, block_b), :]
        u_hit = jnp.any(_sphere_tile_hit(
            rox, roy, roz, rdx, rdy, rdz,
            c_meta[:, 0:1], c_meta[:, 1:2], c_meta[:, 2:3], c_meta[:, 3:4],
            dist_cap=st[0],
        ))

        def _do_dma_and_eval(s):
          cp = pltpu.make_async_copy(_gf_hbm.at[bi * _nm + c_idx, :, :], c_vmem_ref, dma_sem)
          cp.start()
          cp.wait()
          return _eval_micro_primary(_gt, c_vmem_ref[...], s)

        return jax.lax.cond(u_hit, _do_dma_and_eval, lambda s: s, st)

      state_p = jax.lax.fori_loop(0, n_micro_t, _run_micro_hbm, state_p)

    best_d, best_gid, best_pid, best_nx, best_ny, best_nz, alb_r, alb_g, alb_b, roughness, spec_coef = state_p
    hit = best_gid >= 0
    d_safe = jnp.where(hit, best_d, 0.0)
    nx_out = jnp.where(hit, best_nx, 0.0)
    ny_out = jnp.where(hit, best_ny, 0.0)
    nz_out = jnp.where(hit, best_nz, 0.0)
    nz_onb = jnp.where(hit, best_nz, 1.0)

    hx = rox + d_safe * rdx
    hy = roy + d_safe * rdy
    hz = roz + d_safe * rdz

    if floor_gid_0 >= 0:
      tr, tg_c, tb = _eval_procedural_tex_vmem(
          hx, hy, linfo[:, 21:22], linfo[:, 22:23],
          linfo[:, 23:24], linfo[:, 24:25], linfo[:, 25:26],
          linfo[:, 26:27], linfo[:, 27:28], linfo[:, 28:29],
      )
      is_fg = best_gid == floor_gid_0
      alb_r = jnp.where(is_fg, tr, alb_r)
      alb_g = jnp.where(is_fg, tg_c, alb_g)
      alb_b = jnp.where(is_fg, tb, alb_b)

    if table_gid_0 >= 0:
      tr, tg_c, tb = _eval_procedural_tex_vmem(
          hx, hy, linfo[:, 29:30], linfo[:, 30:31],
          linfo[:, 31:32], linfo[:, 32:33], linfo[:, 33:34],
          linfo[:, 34:35], linfo[:, 35:36], linfo[:, 36:37],
      )
      is_tg = best_gid == table_gid_0
      alb_r = jnp.where(is_tg, tr, alb_r)
      alb_g = jnp.where(is_tg, tg_c, alb_g)
      alb_b = jnp.where(is_tg, tb, alb_b)

    lpx_l, lpy_l, lpz_l = linfo[:, 0:1], linfo[:, 1:2], linfo[:, 2:3]
    ldx_l, ldy_l, ldz_l = linfo[:, 3:4], linfo[:, 4:5], linfo[:, 5:6]
    ldiff_r, ldiff_g, ldiff_b = linfo[:, 6:7], linfo[:, 7:8], linfo[:, 8:9]
    lspec_r, lspec_g, lspec_b = linfo[:, 9:10], linfo[:, 10:11], linfo[:, 11:12]
    lamb_r, lamb_g, lamb_b = linfo[:, 12:13], linfo[:, 13:14], linfo[:, 14:15]
    sky1_r, sky1_g, sky1_b = linfo[:, 15:16], linfo[:, 16:17], linfo[:, 17:18]
    sky2_r, sky2_g, sky2_b = linfo[:, 18:19], linfo[:, 19:20], linfo[:, 20:21]
    exposure_v = linfo[:, 37:38]

    sh_ox = hx + nx_out * 2.0e-4
    sh_oy = hy + ny_out * 2.0e-4
    sh_oz = hz + nz_onb * 2.0e-4

    if l_type_0 == 1:
      lx_v = jnp.broadcast_to(-ldx_l, (block_b, block_r))
      ly_v = jnp.broadcast_to(-ldy_l, (block_b, block_r))
      lz_v = jnp.broadcast_to(-ldz_l, (block_b, block_r))
      dist_lim = 1.0e8
      attenuation = jnp.ones((block_b, block_r), dtype=jnp.float32)
    else:
      dlx, dly, dlz = lpx_l - sh_ox, lpy_l - sh_oy, lpz_l - sh_oz
      d_sq = dlx * dlx + dly * dly + dlz * dlz
      d_raw = jnp.sqrt(d_sq)
      inv_dl = 1.0 / jnp.maximum(d_raw, 1e-12)
      lx_v, ly_v, lz_v = dlx * inv_dl, dly * inv_dl, dlz * inv_dl
      dist_lim = jnp.maximum(d_raw - 1.0e-3, 0.0)
      att_denom = l_att_0[0] + l_att_0[1] * d_raw + l_att_0[2] * d_sq
      attenuation = jnp.where(att_denom > 1e-12, 1.0 / jnp.maximum(att_denom, 1e-12), 1.0)

    ndotl = jnp.maximum(0.0, nx_out * lx_v + ny_out * ly_v + nz_onb * lz_v)

    if do_sh_0:
      sh_max = jnp.where(hit & (ndotl > 0.0), dist_lim, 0.0)
      occ_i32 = jnp.zeros((block_b, block_r), dtype=jnp.int32)
      for pk_i, (pfx, pfy) in enumerate(zip(plane_fx, plane_fy)):
        px = pf[0 * block_b : 1 * block_b, pk_i : pk_i + 1]
        py = pf[1 * block_b : 2 * block_b, pk_i : pk_i + 1]
        pz = pf[2 * block_b : 3 * block_b, pk_i : pk_i + 1]
        m00, m01, m02 = pf[3 * block_b : 4 * block_b, pk_i : pk_i + 1], pf[4 * block_b : 5 * block_b, pk_i : pk_i + 1], pf[5 * block_b : 6 * block_b, pk_i : pk_i + 1]
        m10, m11, m12 = pf[6 * block_b : 7 * block_b, pk_i : pk_i + 1], pf[7 * block_b : 8 * block_b, pk_i : pk_i + 1], pf[8 * block_b : 9 * block_b, pk_i : pk_i + 1]
        m20, m21, m22 = pf[9 * block_b : 10 * block_b, pk_i : pk_i + 1], pf[10 * block_b : 11 * block_b, pk_i : pk_i + 1], pf[11 * block_b : 12 * block_b, pk_i : pk_i + 1]
        sx, sy, sz = pf[12 * block_b : 13 * block_b, pk_i : pk_i + 1], pf[13 * block_b : 14 * block_b, pk_i : pk_i + 1], pf[14 * block_b : 15 * block_b, pk_i : pk_i + 1]
        scd, _, _, _ = _ray_one_geom_v2(
            GEOM_PLANE, px, py, pz, m00, m01, m02, m10, m11, m12, m20, m21, m22,
            sx, sy, sz, 1.0e20, sh_ox, sh_oy, sh_oz, lx_v, ly_v, lz_v,
            plane_finite_x=pfx, plane_finite_y=pfy, compute_normal=False,
        )
        occ_i32 = occ_i32 | (scd < sh_max).astype(jnp.int32)

      def _eval_micro_shadow(gt, c_slice, occ_in):
        occ_cur = occ_in
        for k in range(CLUSTER_K):
          px = c_slice[0 * block_b : 1 * block_b, k : k + 1]
          py = c_slice[1 * block_b : 2 * block_b, k : k + 1]
          pz = c_slice[2 * block_b : 3 * block_b, k : k + 1]
          m00, m01, m02 = c_slice[3 * block_b : 4 * block_b, k : k + 1], c_slice[4 * block_b : 5 * block_b, k : k + 1], c_slice[5 * block_b : 6 * block_b, k : k + 1]
          m10, m11, m12 = c_slice[6 * block_b : 7 * block_b, k : k + 1], c_slice[7 * block_b : 8 * block_b, k : k + 1], c_slice[8 * block_b : 9 * block_b, k : k + 1]
          m20, m21, m22 = c_slice[9 * block_b : 10 * block_b, k : k + 1], c_slice[10 * block_b : 11 * block_b, k : k + 1], c_slice[11 * block_b : 12 * block_b, k : k + 1]
          sx, sy, sz = c_slice[12 * block_b : 13 * block_b, k : k + 1], c_slice[13 * block_b : 14 * block_b, k : k + 1], c_slice[14 * block_b : 15 * block_b, k : k + 1]
          rb_sq = c_slice[15 * block_b : 16 * block_b, k : k + 1]
          scd, _, _, _ = _ray_one_geom_v2(
              gt, px, py, pz, m00, m01, m02, m10, m11, m12, m20, m21, m22,
              sx, sy, sz, rb_sq, sh_ox, sh_oy, sh_oz, lx_v, ly_v, lz_v, compute_normal=False,
          )
          occ_cur = occ_cur | ((rb_sq >= 0.0) & (scd < sh_max)).astype(jnp.int32)
        return occ_cur

      for t_i, gt in enumerate(active_types):
        cm_ref_t = cmeta_refs[t_i]
        gf_hbm_t = gf_hbm_refs[t_i]
        n_micro_t = n_micro_list[t_i]

        def _run_sh_hbm(c_idx, occ_in, _gt=gt, _cm=cm_ref_t, _gf_hbm=gf_hbm_t, _nm=n_micro_t):
          c_meta = _cm[pl.ds(c_idx * block_b, block_b), :]
          u_hit = jnp.any(_sphere_tile_hit(
              sh_ox, sh_oy, sh_oz, lx_v, ly_v, lz_v,
              c_meta[:, 0:1], c_meta[:, 1:2], c_meta[:, 2:3], c_meta[:, 3:4],
              dist_cap=sh_max,
          ))

          def _do_dma_sh(o):
            cp = pltpu.make_async_copy(_gf_hbm.at[bi * _nm + c_idx, :, :], c_vmem_ref, dma_sem)
            cp.start()
            cp.wait()
            return _eval_micro_shadow(_gt, c_vmem_ref[...], o)

          return jax.lax.cond(u_hit, _do_dma_sh, lambda o: o, occ_in)

        occ_i32 = jax.lax.fori_loop(0, n_micro_t, _run_sh_hbm, occ_i32)

      vis = jnp.where((occ_i32 > 0) & hit & (ndotl > 0.0), shadow_frac, 1.0)
    else:
      vis = 1.0

    alpha = roughness * roughness
    a2 = alpha * alpha
    k_g = ((roughness + 1.0) * (roughness + 1.0)) * 0.125
    f0 = 0.04 + 0.08 * spec_coef
    ndotv = jnp.maximum(1e-4, -(nx_out * rdx + ny_out * rdy + nz_onb * rdz))
    g1_v = ndotv / (ndotv * (1.0 - k_g) + k_g + 1e-12)
    hx_v, hy_v, hz_v = lx_v - rdx, ly_v - rdy, lz_v - rdz
    inv_hn = 1.0 / jnp.maximum(jnp.sqrt(hx_v * hx_v + hy_v * hy_v + hz_v * hz_v), 1e-12)
    hx_n, hy_n, hz_n = hx_v * inv_hn, hy_v * inv_hn, hz_v * inv_hn
    ndoth = jnp.maximum(0.0, nx_out * hx_n + ny_out * hy_n + nz_onb * hz_n)
    vdoth = jnp.maximum(0.0, -(rdx * hx_n + rdy * hy_n + rdz * hz_n))
    denom_d = ndoth * ndoth * (a2 - 1.0) + 1.0
    d_ggx = a2 / (math.pi * denom_d * denom_d + 1e-12)
    omv = 1.0 - vdoth
    omv2 = omv * omv
    f_schlick = f0 + (1.0 - f0) * (omv2 * omv2 * omv)
    g1_l = ndotl / (ndotl * (1.0 - k_g) + k_g + 1e-12)
    spec_brdf = (d_ggx * f_schlick * g1_l * g1_v) / (4.0 * ndotl * ndotv + 1e-4)

    weight = attenuation * vis * ndotl
    kd = 1.0 - f_schlick
    diff_r, diff_g, diff_b = ldiff_r * weight * kd, ldiff_g * weight * kd, ldiff_b * weight * kd
    spec_fac = jnp.where((spec_coef > 0.0) & (ndotl > 0.0), spec_brdf * weight * math.pi, 0.0)
    spec_r, spec_g, spec_b = lspec_r * spec_fac, lspec_g * spec_fac, lspec_b * spec_fac

    hemi_t = 0.5 + 0.5 * nz_onb
    amb_r = lamb_r + 0.16 * (hemi_t * sky1_r + (1.0 - hemi_t) * sky2_r)
    amb_g = lamb_g + 0.16 * (hemi_t * sky1_g + (1.0 - hemi_t) * sky2_g)
    amb_b = lamb_b + 0.16 * (hemi_t * sky1_b + (1.0 - hemi_t) * sky2_b)
    if headlight_active:
      amb_r = amb_r + 0.35 * hl_amb[0]
      amb_g = amb_g + 0.35 * hl_amb[1]
      amb_b = amb_b + 0.35 * hl_amb[2]
      diff_r = diff_r + 0.25 * hl_diff[0] * ndotv
      diff_g = diff_g + 0.25 * hl_diff[1] * ndotv
      diff_b = diff_b + 0.25 * hl_diff[2] * ndotv

    tot_r = alb_r * (amb_r + diff_r) + spec_r
    tot_g = alb_g * (amb_g + diff_g) + spec_g
    tot_b = alb_b * (amb_b + diff_b) + spec_b
    sky_t = jnp.clip(0.5 * (rdz + 1.0), 0.0, 1.0)
    lin_r = jnp.where(hit, tot_r, sky_t * sky1_r + (1.0 - sky_t) * sky2_r)
    lin_g = jnp.where(hit, tot_g, sky_t * sky1_g + (1.0 - sky_t) * sky2_g)
    lin_b = jnp.where(hit, tot_b, sky_t * sky1_b + (1.0 - sky_t) * sky2_b)

    if use_agx:
      fin_r, fin_g, fin_b = _agx_tonemap_rgb_components(lin_r, lin_g, lin_b, exposure_v)
    else:
      fin_r, fin_g, fin_b = jnp.clip(lin_r, 0.0, 1.0), jnp.clip(lin_g, 0.0, 1.0), jnp.clip(lin_b, 0.0, 1.0)

    out_r_ref[...] = fin_r.astype(out_dtype)
    out_g_ref[...] = fin_g.astype(out_dtype)
    out_b_ref[...] = fin_b.astype(out_dtype)
    out_d_ref[...] = jnp.where(hit, znear_f + best_d * neg_z, -1.0).astype(out_dtype)
    if emit_aux:
      out_nx_ref[...] = nx_out.astype(out_dtype)
      out_ny_ref[...] = ny_out.astype(out_dtype)
      out_nz_ref[...] = nz_out.astype(out_dtype)
      out_gid_ref[...] = best_gid
      out_pid_ref[...] = jnp.where(hit, best_pid, jnp.int32(0))

  in_arrays = [cam_128, lray_dev, plane_fields, light_128] + cmeta_list + gf_hbm_list
  in_specs = [
      pl.BlockSpec((block_b, 128), lambda bi, ri: (bi, 0)),
      pl.BlockSpec((16, block_r), lambda bi, ri: (0, ri)),
      pl.BlockSpec((24 * block_b, P_pad), lambda bi, ri: (bi, 0)),
      pl.BlockSpec((block_b, 128), lambda bi, ri: (bi, 0)),
  ]
  for n_micro_t in n_micro_list:
    in_specs.append(pl.BlockSpec((n_micro_t * block_b, 128), lambda bi, ri: (bi, 0)))
  for _ in n_micro_list:
    in_specs.append(pl.BlockSpec(memory_space=pltpu.HBM))

  out_shape = [jax.ShapeDtypeStruct((B_pad, R_pad), out_dtype) for _ in range(4)]
  if emit_aux:
    out_shape.extend([
        jax.ShapeDtypeStruct((B_pad, R_pad), out_dtype),
        jax.ShapeDtypeStruct((B_pad, R_pad), out_dtype),
        jax.ShapeDtypeStruct((B_pad, R_pad), out_dtype),
        jax.ShapeDtypeStruct((B_pad, R_pad), jnp.int32),
        jax.ShapeDtypeStruct((B_pad, R_pad), jnp.int32),
    ])

  outs = pl.pallas_call(
      _kernel,
      out_shape=out_shape,
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          grid=(n_b_blk, n_r_blk),
          in_specs=in_specs,
          out_specs=[pl.BlockSpec((block_b, block_r), lambda bi, ri: (bi, ri)) for _ in range(n_outs)],
          scratch_shapes=[
              pltpu.VMEM((23 * block_b, 128), jnp.float32),
              pltpu.SemaphoreType.DMA,
          ],
      ),
      compiler_params=None if interpret else pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
      interpret=interpret,
  )(*in_arrays)

  r_out, g_out, b_out, d_out = outs[0], outs[1], outs[2], outs[3]
  res: Dict[str, jax.Array] = {
      "rgb": jnp.stack([r_out[:B, :R], g_out[:B, :R], b_out[:B, :R]], axis=-1).reshape(B, H, W, 3),
      "distance_to_image_plane": d_out[:B, :R].reshape(B, H, W),
  }
  if emit_aux:
    nx_out, ny_out, nz_out, gid_out, pid_out = outs[4], outs[5], outs[6], outs[7], outs[8]
    gid_hw = gid_out[:B, :R].reshape(B, H, W)
    pid_hw = pid_out[:B, :R].reshape(B, H, W)
    res["normals"] = jnp.stack([nx_out[:B, :R], ny_out[:B, :R], nz_out[:B, :R]], axis=-1).reshape(B, H, W, 3)
    res["instance_id"] = gid_hw
    res["semantic_segmentation"] = jnp.where(gid_hw >= 0, pid_hw % 64, jnp.int32(0))
    res["_body_id"] = jnp.where(gid_hw >= 0, pid_hw // 64, jnp.int32(-1))
  return res


class HBMStreamedTiledCamera:
  """Drop-in TiledCamera subclass/wrapper using HBM-streamed Morton Cluster-BVH for G=2,048..4,096+."""

  def __init__(self, mjmodel: Any, cfg: Any):
    self.cfg = cfg
    self.cam_ids = _resolve_cam_ids(mjmodel, cfg.cam_names)
    self.multi_cam = not isinstance(cfg.cam_names, (int, str))
    self.specs = tuple(
        RenderSceneSpec.from_mjmodel(
            mjmodel,
            cam_id=cid,
            height=int(cfg.height),
            width=int(cfg.width),
            use_shadows=bool(cfg.use_shadows),
            shadow_light_fraction=float(cfg.shadow_light_fraction),
        )
        for cid in self.cam_ids
    )
    self.base_fovys = tuple(float(sp.cam_fovy) for sp in self.specs)
    self.category_masks = _default_category_masks(mjmodel)
    gids = self.specs[0].enabled_geom_ids
    if cfg.semantic_fn is not None:
      self.geom_sem_ids = tuple(int(x) for x in cfg.semantic_fn(mjmodel, gids))
    else:
      self.geom_sem_ids = _default_semantic_classes(mjmodel, gids, self.category_masks)

    g_bid = np.asarray(mjmodel.geom_bodyid, dtype=np.int32)
    b_rid = np.asarray(mjmodel.body_rootid, dtype=np.int32)
    self.geom_body_ids = tuple(int(g_bid[g]) for g in gids)
    self.body_rootid = tuple(int(x) for x in b_rid)
    self.active_moving_bids = tuple(sorted({bid for bid in self.geom_body_ids if bid > 0}))
    self.floor_gids = tuple(int(g) for g in gids if self.category_masks["floor"][g])
    self.table_gids = tuple(int(g) for g in gids if self.category_masks["table"][g])

    n_devs = len(jax.devices())
    self.n_chips = min(n_devs, int(cfg.n_chips)) if cfg.n_chips is not None else 1
    if self.n_chips > 1:
      self.mesh = jax.sharding.Mesh(np.array(jax.devices()[: self.n_chips]), (cfg.data_axis,))
    else:
      self.mesh = None

  @property
  def n_enabled_geoms(self) -> int:
    return len(self.specs[0].enabled_geom_ids)

  def shard_tree(self, tree: Any) -> Any:
    if self.mesh is None or self.n_chips <= 1:
      return jax.device_put(tree, jax.devices()[0])
    P = jax.sharding.PartitionSpec
    return jax.device_put(tree, jax.sharding.NamedSharding(self.mesh, P(self.cfg.data_axis)))

  def randomize_local(self, rng_or_keys: jax.Array, mjx_data_batch: Any, rep_cfg: Optional[ReplicatorCfg] = None) -> ReplicatorBatch:
    return _apply_replicator_local(
        rng_or_keys=rng_or_keys,
        mjx_data_batch=mjx_data_batch,
        base_spec=self.specs[0],
        cam_ids=self.cam_ids,
        base_fovys=self.base_fovys,
        category_masks=self.category_masks,
        cfg=rep_cfg or ReplicatorCfg(),
    )

  def render_local(
      self,
      mjx_data_batch: Any,
      rep_batch: Optional[ReplicatorBatch] = None,
      rng_or_keys: Optional[jax.Array] = None,
      rep_cfg: Optional[ReplicatorCfg] = None,
  ) -> Dict[str, jax.Array]:
    if rep_batch is None and rng_or_keys is not None:
      rep_batch = self.randomize_local(rng_or_keys, mjx_data_batch, rep_cfg=rep_cfg)

    interp = (jax.default_backend() == "cpu") if self.cfg.interpret is None else bool(self.cfg.interpret)
    need_aux = any(k in self.cfg.data_types for k in ("normals", "semantic_segmentation", "instance_id", "motion_vectors"))

    if not self.multi_cam:
      raw = _pallas_hbm_hier_kernel(
          spec=self.specs[0], mjx_data_batch=mjx_data_batch, rep_batch=rep_batch, cam_slot=0,
          geom_sem_ids=self.geom_sem_ids, geom_body_ids=self.geom_body_ids,
          floor_gids=self.floor_gids, table_gids=self.table_gids,
          tonemap=self.cfg.tonemap, out_dtype=self.cfg.out_dtype,
          block_b=self.cfg.block_b, block_r=self.cfg.block_r,
          emit_aux=need_aux, interpret=interp,
      )
      if "motion_vectors" in self.cfg.data_types:
        raw["motion_vectors"] = _compute_motion_vectors_zero_gather(
            spec=self.specs[0], mjx_data_batch=mjx_data_batch, rep_batch=rep_batch, cam_slot=0,
            z_depth=raw["distance_to_image_plane"], instance_id=raw["instance_id"],
            body_id_map=raw["_body_id"], active_moving_bids=self.active_moving_bids,
            body_rootid=self.body_rootid, dt=self.cfg.dt, out_dtype=self.cfg.out_dtype,
        )
      return {k: raw[k] for k in self.cfg.data_types}

    B = int(mjx_data_batch.cam_xpos.shape[0])
    N_cam = len(self.cam_ids)
    H, W = int(self.cfg.height), int(self.cfg.width)

    def _tile_cam(x: jax.Array) -> jax.Array:
      return jnp.broadcast_to(x[:, None, ...], (B, N_cam, *x.shape[1:])).reshape(B * N_cam, *x.shape[1:])

    cpos_stack = jnp.stack([mjx_data_batch.cam_xpos[:, int(cid), :] for cid in self.cam_ids], axis=1)
    cmat_stack = jnp.stack([mjx_data_batch.cam_xmat[:, int(cid), :, :] for cid in self.cam_ids], axis=1)

    @dataclasses.dataclass
    class _FlatDataProxy:
      geom_xpos: jax.Array
      geom_xmat: jax.Array
      cam_xpos: jax.Array
      cam_xmat: jax.Array
      light_xpos: jax.Array
      light_xmat: jax.Array
      xpos: jax.Array
      xmat: jax.Array

    lpos_src = getattr(mjx_data_batch, "light_xpos", jnp.zeros((B, 0, 3), dtype=jnp.float32))
    lmat_src = getattr(mjx_data_batch, "light_xmat", jnp.zeros((B, 0, 3, 3), dtype=jnp.float32))
    flat_data = _FlatDataProxy(
        geom_xpos=_tile_cam(mjx_data_batch.geom_xpos),
        geom_xmat=_tile_cam(mjx_data_batch.geom_xmat),
        cam_xpos=cpos_stack.reshape(B * N_cam, 1, 3),
        cam_xmat=cmat_stack.reshape(B * N_cam, 1, 3, 3),
        light_xpos=_tile_cam(lpos_src),
        light_xmat=_tile_cam(lmat_src),
        xpos=_tile_cam(mjx_data_batch.xpos),
        xmat=_tile_cam(mjx_data_batch.xmat),
    )

    if rep_batch is not None:
      flat_rep = ReplicatorBatch(
          cam_xpos=rep_batch.cam_xpos.reshape(B * N_cam, 1, 3),
          cam_xmat=rep_batch.cam_xmat.reshape(B * N_cam, 1, 3, 3),
          cam_fovy_deg=rep_batch.cam_fovy_deg.reshape(B * N_cam, 1),
          light_xpos=_tile_cam(rep_batch.light_xpos),
          light_xdir=_tile_cam(rep_batch.light_xdir),
          light_diffuse=_tile_cam(rep_batch.light_diffuse),
          light_specular=_tile_cam(rep_batch.light_specular),
          light_ambient=_tile_cam(rep_batch.light_ambient),
          sky_rgb=_tile_cam(rep_batch.sky_rgb),
          exposure=_tile_cam(rep_batch.exposure),
          geom_rgba=_tile_cam(rep_batch.geom_rgba),
          geom_roughness=_tile_cam(rep_batch.geom_roughness),
          geom_specular=_tile_cam(rep_batch.geom_specular),
          floor_tex=_tile_cam(rep_batch.floor_tex),
          table_tex=_tile_cam(rep_batch.table_tex),
      )
      fovy_override = None
    else:
      flat_rep = None
      fovy_override = jnp.broadcast_to(
          jnp.asarray(self.base_fovys, dtype=jnp.float32)[None, :], (B, N_cam)
      ).reshape(B * N_cam)

    sp0 = dataclasses.replace(self.specs[0], cam_id=0)
    raw_flat = _pallas_hbm_hier_kernel(
        spec=sp0, mjx_data_batch=flat_data, rep_batch=flat_rep, cam_slot=0,
        geom_sem_ids=self.geom_sem_ids, geom_body_ids=self.geom_body_ids,
        floor_gids=self.floor_gids, table_gids=self.table_gids,
        tonemap=self.cfg.tonemap, out_dtype=self.cfg.out_dtype,
        block_b=self.cfg.block_b, block_r=self.cfg.block_r,
        emit_aux=need_aux, interpret=interp, cam_fovy_override=fovy_override,
    )
    out = {k: v.reshape(B, N_cam, H, W, *v.shape[3:]) for k, v in raw_flat.items()}
    if "motion_vectors" in self.cfg.data_types:
      flows = [
          _compute_motion_vectors_zero_gather(
              spec=self.specs[ci], mjx_data_batch=mjx_data_batch, rep_batch=rep_batch, cam_slot=ci,
              z_depth=out["distance_to_image_plane"][:, ci], instance_id=out["instance_id"][:, ci],
              body_id_map=out["_body_id"][:, ci], active_moving_bids=self.active_moving_bids,
              body_rootid=self.body_rootid, dt=self.cfg.dt, out_dtype=self.cfg.out_dtype,
          )
          for ci in range(N_cam)
      ]
      out["motion_vectors"] = jnp.stack(flows, axis=1)
    return {k: out[k] for k in self.cfg.data_types}

  def build_step_and_render_fn(self, mjx_model: Any, batch_size: int, rep_cfg: Optional[ReplicatorCfg] = None, with_dr: bool = True, with_physics: bool = True):
    from mujoco import mjx

    use_spmd = self.mesh is not None and self.n_chips > 1 and batch_size >= self.n_chips and (batch_size % self.n_chips == 0)

    def _local_fn(mx_l, d_b, u_b, rng_b):
      dn = jax.vmap(lambda di, ui: mjx.step(mx_l, di.replace(ctrl=ui)))(d_b, u_b) if with_physics else d_b
      rb = self.randomize_local(rng_b, dn, rep_cfg=rep_cfg) if with_dr else None
      obs = self.render_local(dn, rep_batch=rb)
      return dn, obs

    if not use_spmd:
      return jax.jit(lambda d_b, u_b, rng_b: _local_fn(mjx_model, d_b, u_b, rng_b))

    P = jax.sharding.PartitionSpec
    ax = self.cfg.data_axis
    sm_kw = _shard_map_kwargs()
    _sm = shard_map(
        _local_fn,
        mesh=self.mesh,
        in_specs=(P(), P(ax), P(ax), P(ax)),
        out_specs=(P(ax), {k: P(ax) for k in self.cfg.data_types}),
        **sm_kw,
    )
    return jax.jit(lambda d_b, u_b, rng_b: _sm(mjx_model, d_b, u_b, rng_b))
