"""Wave 4: Isaac Sim Multi-Sensor Robotics Suite on TPU v5p (GPU-Free, Zero-Copy).

Implements:
1. Native TPU Mosaic (BLOCK_B=8, BLOCK_R=512) 128-Beam x 2048-Azimuth
   (262,144 rays/env) 360-degree Batch LiDAR raycaster with direct VMEM
   range_m, semantic_id, geom_id, and fused world 3D point cloud (xyz).
2. Fused Multi-Sensor Robotics Step:
   mjx.step(mx, dx) + 128x2048 360-deg LiDAR + Stereo 128x128 RGB+Depth+Semantic
   in a single JIT/shard_map graph with host_callbacks == 0, send_recv_ops == 0,
   and gather_ops == 0.
3. High-res data export for 4K multi-sensor + 3D LiDAR point cloud showcase.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import re
import time
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np

from experiments.w6_highres_multicam.wave3_8k_multicam8 import (
    make_cluttered_tabletop_8cam_xml,
    pallas_raycast_ncam_v2,
)
from tpu_mjx_render.pallas_v2 import (
    BLOCK_B,
    BLOCK_R,
    _compute_geom_bound_sq,
    _ray_one_geom_v2,
)
from tpu_mjx_render.primitives import MJ_MAXVAL
from tpu_mjx_render.renderer import RenderSceneSpec
from tpu_mjx_render.roofline import inspect_hlo_ops

LIDAR_BEAMS = 128
LIDAR_AZIMUTH = 2048
LIDAR_RAYS = LIDAR_BEAMS * LIDAR_AZIMUTH  # 262,144 = 512 * 512

SEMANTIC_NAMES = {
    0: "sky_miss",
    1: "floor_workbench",
    2: "robot_panda",
    3: "target_objects",
    4: "sorting_bins",
    5: "tools_bottles_rack",
}


class LidarScan(NamedTuple):
  """Batched 128x2048 360-degree LiDAR output (all in TPU HBM)."""

  range_m: jax.Array       # (B, 128, 2048) float32 (-1.0 on miss)
  semantic_id: jax.Array   # (B, 128, 2048) int32 (0 on miss, 1..5 on hit)
  geom_id: jax.Array       # (B, 128, 2048) int32 (-1 on miss)
  xyz: Optional[jax.Array] # (B, 128, 2048, 3) float32 (0 on miss)


def _audit_hlo(lowered: Any) -> Dict[str, Any]:
  txt = lowered.as_text()
  info = inspect_hlo_ops(txt)
  info["send_recv_ops"] = len(re.findall(r"\bsend\(|\brecv\(", txt, re.I))
  assert info["host_callbacks"] == 0, f"HLO contains host_callback: {info}"
  assert info["send_recv_ops"] == 0, f"HLO contains send/recv: {info}"
  assert info["gather_ops"] == 0, f"HLO contains gather_ops: {info}"
  info["zero_copy_verified"] = True
  return info


def make_isaac_tabletop_lidar_xml() -> str:
  """Cluttered Panda tabletop scene + 360-degree LiDAR mast camera rig."""
  xml = make_cluttered_tabletop_8cam_xml()
  lidar_cam = (
      '    <camera name="lidar_rig" fovy="60" pos="0.30 -0.34 0.42" '
      'xyaxes="0 -1 0 1 0 0"/>\n'
      '    <geom name="lidar_mast" type="cylinder" pos="0.30 -0.34 0.20" '
      'size="0.014 0.20" rgba="0.18 0.20 0.24 1" contype="0" conaffinity="0"/>\n'
      '    <geom name="lidar_puck" type="cylinder" pos="0.30 -0.34 0.42" '
      'size="0.042 0.025" rgba="0.12 0.55 0.92 1" contype="0" conaffinity="0"/>\n'
  )
  anchor = '    <camera name="cam1_overhead"'
  assert anchor in xml
  xml = xml.replace(anchor, lidar_cam + anchor)
  return xml


def assign_semantic_classes(m: mujoco.MjModel, gids: Sequence[int]) -> Tuple[int, ...]:
  """Map each enabled MuJoCo geom ID to semantic class 1..5."""
  sem_ids: List[int] = []
  for gid in gids:
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(gid)) or ""
    if name in ("floor", "workbench_top", "workbench_trim"):
      sem = 1
    elif name.startswith("bin_"):
      sem = 4
    elif (
        name.startswith("bottle_")
        or name == "spray_can"
        or name.startswith("wrench_")
        or name.startswith("mallet_")
        or name.startswith("rack_")
        or name.startswith("lidar_")
    ):
      sem = 5
    elif (
        name in ("box", "mug_ellipsoid")
        or name.startswith("target_cube_")
        or name.startswith("prop_")
    ):
      sem = 3
    else:
      sem = 2  # Panda links, hand, fingers, base
    sem_ids.append(sem)
  return tuple(sem_ids)


def build_lidar_ray_templates(
    n_beams: int = LIDAR_BEAMS,
    n_azimuth: int = LIDAR_AZIMUTH,
    elev_deg: Tuple[float, float] = (-42.0, 14.0),
) -> np.ndarray:
  """Return local unit ray directions of shape (n_beams * n_azimuth, 3) float32."""
  el = np.linspace(np.deg2rad(elev_deg[0]), np.deg2rad(elev_deg[1]), n_beams, dtype=np.float32)
  az = np.linspace(0.0, 2.0 * np.pi, n_azimuth, endpoint=False, dtype=np.float32)
  az_g, el_g = np.meshgrid(az, el, indexing="xy")  # (n_beams, n_azimuth)
  cos_el = np.cos(el_g)
  dx = cos_el * np.cos(az_g)
  dy = cos_el * np.sin(az_g)
  dz = np.sin(el_g)
  ldir = np.stack([dx, dy, dz], axis=-1).reshape(-1, 3).astype(np.float32)
  # Normalize strictly for numerical parity
  ldir /= np.maximum(np.linalg.norm(ldir, axis=-1, keepdims=True), 1e-12)
  return ldir


def jax_lidar_ref(
    spec: RenderSceneSpec,
    sem_ids_static: Tuple[int, ...],
    mjx_data_batch: Any,
    lidar_cam_id: int,
    ldir_np: np.ndarray,
    r_min: float = 0.06,
    max_range: float = 12.0,
    compute_xyz: bool = True,
) -> LidarScan:
  """Exact branchless JAX SoA reference for 128x2048 360-deg LiDAR."""
  s_pos = mjx_data_batch.cam_xpos[:, lidar_cam_id, :]
  s_mat = mjx_data_batch.cam_xmat[:, lidar_cam_id, :, :]
  geom_xpos = mjx_data_batch.geom_xpos
  geom_xmat = mjx_data_batch.geom_xmat

  ld = jnp.asarray(ldir_np, dtype=jnp.float32)
  rdir = jnp.einsum("bij,rj->bri", s_mat, ld, precision=jax.lax.Precision.HIGHEST)
  rdir = rdir / jnp.maximum(jnp.linalg.norm(rdir, axis=-1, keepdims=True), 1e-12)
  rorig = s_pos[:, None, :] + float(r_min) * rdir

  B, R, _ = rorig.shape
  rox, roy, roz = rorig[..., 0], rorig[..., 1], rorig[..., 2]
  rdx, rdy, rdz = rdir[..., 0], rdir[..., 1], rdir[..., 2]

  gids = np.asarray(spec.enabled_geom_ids, dtype=np.int32)
  rb_sq_all = _compute_geom_bound_sq(spec, gids)

  d_limit = float(max_range - r_min)
  best_d = jnp.full((B, R), d_limit, dtype=jnp.float32)
  best_gid = jnp.full((B, R), -1, dtype=jnp.int32)
  best_sem = jnp.zeros((B, R), dtype=jnp.int32)

  for k, (gid_val, sem_val) in enumerate(zip(gids, sem_ids_static)):
    gt = int(spec.geom_type[gid_val])
    px = geom_xpos[:, gid_val, 0:1]
    py = geom_xpos[:, gid_val, 1:2]
    pz = geom_xpos[:, gid_val, 2:3]
    mat = geom_xmat[:, gid_val, :, :]
    sz = spec.geom_size[gid_val]
    cd, _, _, _ = _ray_one_geom_v2(
        gt,
        px, py, pz,
        mat[:, 0, 0:1], mat[:, 0, 1:2], mat[:, 0, 2:3],
        mat[:, 1, 0:1], mat[:, 1, 1:2], mat[:, 1, 2:3],
        mat[:, 2, 0:1], mat[:, 2, 1:2], mat[:, 2, 2:3],
        float(sz[0]), float(sz[1]), float(sz[2]),
        float(rb_sq_all[k]),
        rox, roy, roz, rdx, rdy, rdz,
        plane_finite_x=float(spec.geom_size[gid_val, 0]) > 0.0,
        plane_finite_y=float(spec.geom_size[gid_val, 1]) > 0.0,
        compute_normal=False,
    )
    imp = cd < best_d
    best_d = jnp.where(imp, cd, best_d)
    best_gid = jnp.where(imp, jnp.int32(gid_val), best_gid)
    best_sem = jnp.where(imp, jnp.int32(sem_val), best_sem)

  hit = best_gid >= 0
  range_flat = jnp.where(hit, best_d + float(r_min), -1.0)
  xyz = None
  if compute_xyz:
    pts = s_pos[:, None, :] + jnp.where(hit, range_flat, 0.0)[..., None] * rdir
    xyz = jnp.where(hit[..., None], pts, 0.0).reshape(B, LIDAR_BEAMS, LIDAR_AZIMUTH, 3)

  return LidarScan(
      range_m=range_flat.reshape(B, LIDAR_BEAMS, LIDAR_AZIMUTH),
      semantic_id=best_sem.reshape(B, LIDAR_BEAMS, LIDAR_AZIMUTH),
      geom_id=best_gid.reshape(B, LIDAR_BEAMS, LIDAR_AZIMUTH),
      xyz=xyz,
  )


def pallas_lidar_128x2048(
    spec: RenderSceneSpec,
    sem_ids_static: Tuple[int, ...],
    mjx_data_batch: Any,
    lidar_cam_id: int,
    ldir_np: np.ndarray,
    r_min: float = 0.06,
    max_range: float = 12.0,
    compute_xyz: bool = False,
    interpret: bool = False,
) -> LidarScan:
  """Native TPU Mosaic (8, 512) 128-Beam x 2048-Azimuth Batch LiDAR raycaster."""
  s_pos = mjx_data_batch.cam_xpos[:, lidar_cam_id, :]
  s_mat = mjx_data_batch.cam_xmat[:, lidar_cam_id, :, :]
  geom_xpos = mjx_data_batch.geom_xpos
  geom_xmat = mjx_data_batch.geom_xmat
  geom_size = jnp.asarray(spec.geom_size, dtype=jnp.float32)

  B = s_pos.shape[0]
  R = ldir_np.shape[0]
  assert R == LIDAR_RAYS and R % BLOCK_R == 0

  gids = np.asarray(spec.enabled_geom_ids, dtype=np.int32)
  G = len(gids)
  B_pad = ((B + BLOCK_B - 1) // BLOCK_B) * BLOCK_B
  G_pad = ((G + 127) // 128) * 128

  cam_pack = jnp.concatenate([s_pos, s_mat.reshape(B, 9)], axis=-1)
  cam_128 = jnp.pad(cam_pack, ((0, B_pad - B), (0, 128 - 12)))

  lray_raw = np.zeros((8, R), dtype=np.float32)
  lray_raw[0:3, :] = ldir_np.T
  lray_dev = jnp.asarray(lray_raw, dtype=jnp.float32)

  gpos = geom_xpos[:, gids, :]
  gmat = geom_xmat[:, gids, :, :].reshape(B, G, 9)
  gsz_all = jnp.broadcast_to(geom_size[gids][None, :, :], (B, G, 3))
  rb_sq = jnp.broadcast_to(jnp.asarray(_compute_geom_bound_sq(spec, gids))[None, :, None], (B, G, 1))
  g_16 = jnp.concatenate([gpos, gmat, gsz_all, rb_sq], axis=-1)
  g_16 = jnp.pad(g_16, ((0, B_pad - B), (0, G_pad - G), (0, 0)))

  n_b_blk = B_pad // BLOCK_B
  n_r_blk = R // BLOCK_R
  geom_fields = (
      g_16.reshape(n_b_blk, BLOCK_B, G_pad, 16)
      .transpose(0, 3, 1, 2)
      .reshape(n_b_blk * 16 * BLOCK_B, G_pad)
  )

  g_types_static = tuple(int(spec.geom_type[g]) for g in gids)
  gids_static = tuple(int(g) for g in gids)
  plane_fx = tuple(float(spec.geom_size[g, 0]) > 0.0 for g in gids)
  plane_fy = tuple(float(spec.geom_size[g, 1]) > 0.0 for g in gids)
  r_min_f = float(r_min)
  d_limit_f = float(max_range - r_min)

  def _lidar_kernel(cam_ref, lray_ref, gf_ref, out_r_ref, out_sem_ref, out_gid_ref):
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

    rox = cpx + r_min_f * rdx
    roy = cpy + r_min_f * rdy
    roz = cpz + r_min_f * rdz

    best_d = jnp.full((BLOCK_B, BLOCK_R), d_limit_f, dtype=jnp.float32)
    best_gid = jnp.full((BLOCK_B, BLOCK_R), -1, dtype=jnp.int32)
    best_sem = jnp.zeros((BLOCK_B, BLOCK_R), dtype=jnp.int32)

    for k, (gt, gid_val, sem_val, pfx, pfy) in enumerate(
        zip(g_types_static, gids_static, sem_ids_static, plane_fx, plane_fy)
    ):
      px = gf[0 * 8 : 1 * 8, k : k + 1]
      py = gf[1 * 8 : 2 * 8, k : k + 1]
      pz = gf[2 * 8 : 3 * 8, k : k + 1]
      m00, m01, m02 = gf[3 * 8 : 4 * 8, k : k + 1], gf[4 * 8 : 5 * 8, k : k + 1], gf[5 * 8 : 6 * 8, k : k + 1]
      m10, m11, m12 = gf[6 * 8 : 7 * 8, k : k + 1], gf[7 * 8 : 8 * 8, k : k + 1], gf[8 * 8 : 9 * 8, k : k + 1]
      m20, m21, m22 = gf[9 * 8 : 10 * 8, k : k + 1], gf[10 * 8 : 11 * 8, k : k + 1], gf[11 * 8 : 12 * 8, k : k + 1]
      sx, sy, sz = gf[12 * 8 : 13 * 8, k : k + 1], gf[13 * 8 : 14 * 8, k : k + 1], gf[14 * 8 : 15 * 8, k : k + 1]
      rb_sq = gf[15 * 8 : 16 * 8, k : k + 1]

      cd, _, _, _ = _ray_one_geom_v2(
          gt,
          px, py, pz,
          m00, m01, m02,
          m10, m11, m12,
          m20, m21, m22,
          sx, sy, sz,
          rb_sq,
          rox, roy, roz, rdx, rdy, rdz,
          plane_finite_x=pfx,
          plane_finite_y=pfy,
          compute_normal=False,
      )
      imp = cd < best_d
      best_d = jnp.where(imp, cd, best_d)
      best_gid = jnp.where(imp, jnp.int32(gid_val), best_gid)
      best_sem = jnp.where(imp, jnp.int32(sem_val), best_sem)

    hit = best_gid >= 0
    out_r_ref[...] = jnp.where(hit, best_d + r_min_f, -1.0)
    out_sem_ref[...] = best_sem
    out_gid_ref[...] = best_gid

  r_out, sem_out, gid_out = pl.pallas_call(
      _lidar_kernel,
      out_shape=[
          jax.ShapeDtypeStruct((B_pad, R), jnp.float32),
          jax.ShapeDtypeStruct((B_pad, R), jnp.int32),
          jax.ShapeDtypeStruct((B_pad, R), jnp.int32),
      ],
      grid=(n_b_blk, n_r_blk),
      in_specs=[
          pl.BlockSpec((BLOCK_B, 128), lambda bi, ri: (bi, 0)),
          pl.BlockSpec((8, BLOCK_R), lambda bi, ri: (0, ri)),
          pl.BlockSpec((16 * BLOCK_B, G_pad), lambda bi, ri: (bi, 0)),
      ],
      out_specs=[
          pl.BlockSpec((BLOCK_B, BLOCK_R), lambda bi, ri: (bi, ri)),
          pl.BlockSpec((BLOCK_B, BLOCK_R), lambda bi, ri: (bi, ri)),
          pl.BlockSpec((BLOCK_B, BLOCK_R), lambda bi, ri: (bi, ri)),
      ],
      compiler_params=None if interpret else pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
      interpret=interpret,
  )(cam_128, lray_dev, geom_fields)

  range_flat = r_out[:B, :R]
  sem_flat = sem_out[:B, :R]
  gid_flat = gid_out[:B, :R]

  xyz = None
  if compute_xyz:
    ld = jnp.asarray(ldir_np, dtype=jnp.float32)
    rdir = jnp.einsum("bij,rj->bri", s_mat, ld, precision=jax.lax.Precision.HIGHEST)
    rdir = rdir / jnp.maximum(jnp.linalg.norm(rdir, axis=-1, keepdims=True), 1e-12)
    hit = gid_flat >= 0
    pts = s_pos[:, None, :] + jnp.where(hit, range_flat, 0.0)[..., None] * rdir
    xyz = jnp.where(hit[..., None], pts, 0.0).reshape(B, LIDAR_BEAMS, LIDAR_AZIMUTH, 3)

  return LidarScan(
      range_m=range_flat.reshape(B, LIDAR_BEAMS, LIDAR_AZIMUTH),
      semantic_id=sem_flat.reshape(B, LIDAR_BEAMS, LIDAR_AZIMUTH),
      geom_id=gid_flat.reshape(B, LIDAR_BEAMS, LIDAR_AZIMUTH),
      xyz=xyz,
  )


def geom_id_to_semantic_nogather(
    geom_id: jax.Array,
    gids_static: Sequence[int],
    sem_ids_static: Sequence[int],
) -> jax.Array:
  """Convert geom_id tensor to semantic_id 0..5 using pure compares (0 gather ops)."""
  sem = jnp.zeros_like(geom_id, dtype=jnp.int32)
  for cls_id in (1, 2, 3, 4, 5):
    cls_gids = [g for g, s in zip(gids_static, sem_ids_static) if s == cls_id]
    if not cls_gids:
      continue
    mask = jnp.zeros(geom_id.shape, dtype=jnp.bool_)
    for g in cls_gids:
      mask = mask | (geom_id == int(g))
    sem = jnp.where(mask, jnp.int32(cls_id), sem)
  return sem


def _make_batch(m: mujoco.MjModel, batch_size: int, vary_ctrl: bool = True):
  d_host = mujoco.MjData(m)
  if m.nkey > 0:
    mujoco.mj_resetDataKeyframe(m, d_host, 0)
  mujoco.mj_forward(m, d_host)
  dx0 = mjx.put_data(m, d_host, impl="jax")
  dx_batch = jax.tree_util.tree_map(
      lambda x: jnp.broadcast_to(x[None, ...], (batch_size, *x.shape)).copy(), dx0
  )
  if vary_ctrl and m.nu > 0:
    phases = jnp.linspace(-0.25, 0.25, batch_size, dtype=jnp.float32)[:, None]
    ctrl = jnp.asarray(dx0.ctrl, jnp.float32)[None, :] + phases * jnp.linspace(1.0, 0.3, m.nu, dtype=jnp.float32)[None, :]
  else:
    ctrl = jnp.zeros((batch_size, m.nu), dtype=jnp.float32)
  return dx_batch, ctrl


def run_parity_suite() -> Dict[str, Any]:
  pallas_interp = (jax.default_backend() == "cpu")
  xml = make_isaac_tabletop_lidar_xml()
  m = mujoco.MjModel.from_xml_string(xml)
  mx = mjx.put_model(m, impl="jax")
  lidar_cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "lidar_rig")
  cam_l_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "cam2_stereo_l")
  cam_r_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "cam3_stereo_r")

  spec_lidar = RenderSceneSpec.from_mjmodel(m, cam_id=lidar_cam_id, height=64, width=64, use_shadows=False)
  sem_ids = assign_semantic_classes(m, spec_lidar.enabled_geom_ids)
  ldir_np = build_lidar_ray_templates()

  spec_l = RenderSceneSpec.from_mjmodel(m, cam_id=cam_l_id, height=128, width=128, use_shadows=True)
  spec_r = RenderSceneSpec.from_mjmodel(m, cam_id=cam_r_id, height=128, width=128, use_shadows=True)

  B = 8
  dx_b, ctrl = _make_batch(m, B, vary_ctrl=True)
  dev0 = jax.devices()[0]
  dx_b = jax.device_put(dx_b, dev0)
  ctrl = jax.device_put(ctrl, dev0)

  @jax.jit
  def step_n(d, u):
    return jax.lax.fori_loop(0, 8, lambda _, x: jax.vmap(lambda a, c: mjx.step(mx, a.replace(ctrl=c)))(x, u), d)

  dx_moved = step_n(dx_b, ctrl)
  jax.block_until_ready(dx_moved.qpos)

  fn_ref = jax.jit(lambda d: jax_lidar_ref(spec_lidar, sem_ids, d, lidar_cam_id, ldir_np, compute_xyz=True))
  fn_pal = jax.jit(lambda d: pallas_lidar_128x2048(spec_lidar, sem_ids, d, lidar_cam_id, ldir_np, compute_xyz=True, interpret=pallas_interp))

  hlo_lidar = _audit_hlo(fn_pal.lower(dx_moved))

  ref = fn_ref(dx_moved)
  pal = fn_pal(dx_moved)
  jax.block_until_ready((pal.range_m, pal.xyz))

  range_mae = float(jnp.mean(jnp.abs(pal.range_m - ref.range_m)))
  max_range_diff = float(jnp.max(jnp.abs(pal.range_m - ref.range_m)))
  xyz_mae = float(jnp.mean(jnp.abs(pal.xyz - ref.xyz)))
  geom_match = float(jnp.mean(pal.geom_id == ref.geom_id))
  sem_match = float(jnp.mean(pal.semantic_id == ref.semantic_id))
  hit_frac = float(jnp.mean(pal.geom_id >= 0))

  assert range_mae < 1e-5 and geom_match == 1.0 and sem_match == 1.0, (
      range_mae, max_range_diff, geom_match, sem_match
  )

  # Also audit full multi-sensor (MJX + 128x2048 LiDAR + Stereo 128x128 RGB+Depth+Semantic)
  @jax.jit
  def fused_multisensor_fn(d, u):
    dn = jax.vmap(lambda x, c: mjx.step(mx, x.replace(ctrl=c)))(d, u)
    lidar = pallas_lidar_128x2048(spec_lidar, sem_ids, dn, lidar_cam_id, ldir_np, compute_xyz=True, interpret=pallas_interp)
    gb_l, gb_r = pallas_raycast_ncam_v2([spec_l, spec_r], dn, interpret=pallas_interp)
    sem_l = geom_id_to_semantic_nogather(gb_l.geom_id, spec_l.enabled_geom_ids, sem_ids)
    sem_r = geom_id_to_semantic_nogather(gb_r.geom_id, spec_r.enabled_geom_ids, sem_ids)
    return dn, lidar.range_m, lidar.semantic_id, lidar.xyz, gb_l.rgb, gb_l.depth, sem_l, gb_r.rgb, gb_r.depth, sem_r

  hlo_multi = _audit_hlo(fused_multisensor_fn.lower(dx_moved, ctrl))
  ms_out = fused_multisensor_fn(dx_moved, ctrl)
  jax.block_until_ready(ms_out[1:])

  res = {
      "backend": jax.default_backend(),
      "device_kind": dev0.device_kind,
      "interpret": pallas_interp,
      "lidar_shape": [LIDAR_BEAMS, LIDAR_AZIMUTH],
      "rays_per_env": LIDAR_RAYS,
      "hit_frac": hit_frac,
      "range_mae_m": range_mae,
      "max_range_diff_m": max_range_diff,
      "xyz_mae_m": xyz_mae,
      "geom_match": geom_match,
      "semantic_match": sem_match,
      "hlo_lidar": hlo_lidar,
      "hlo_multisensor": hlo_multi,
      "passed": True,
  }
  print("WAVE 4 PARITY PASS:", json.dumps(res, indent=2), flush=True)
  return res


def run_bench_suite(
    n_chips: int,
    batches_override: Optional[List[int]] = None,
    pipelines_override: Optional[List[str]] = None,
    out_json: Optional[str] = None,
) -> List[Dict[str, Any]]:
  pallas_interp = (jax.default_backend() == "cpu")
  devs = jax.devices()[:n_chips]
  assert len(devs) == n_chips

  xml = make_isaac_tabletop_lidar_xml()
  m = mujoco.MjModel.from_xml_string(xml)
  mx = mjx.put_model(m, impl="jax")
  lidar_cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "lidar_rig")
  cam_l_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "cam2_stereo_l")
  cam_r_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "cam3_stereo_r")

  spec_lidar = RenderSceneSpec.from_mjmodel(m, cam_id=lidar_cam_id, height=64, width=64, use_shadows=False)
  sem_ids = assign_semantic_classes(m, spec_lidar.enabled_geom_ids)
  ldir_np = build_lidar_ray_templates()

  spec_l = RenderSceneSpec.from_mjmodel(m, cam_id=cam_l_id, height=128, width=128, use_shadows=True)
  spec_r = RenderSceneSpec.from_mjmodel(m, cam_id=cam_r_id, height=128, width=128, use_shadows=True)

  sm_kw = {"check_rep": False} if "check_rep" in inspect.signature(shard_map).parameters else {"check_vma": False}
  if n_chips > 1:
    mesh = jax.sharding.Mesh(np.array(devs), ("dp",))
    P = jax.sharding.PartitionSpec
    sh = jax.sharding.NamedSharding(mesh, P("dp"))
    rep = jax.sharding.NamedSharding(mesh, P())

  batches = batches_override if batches_override else ([256, 512, 1024, 2048] if n_chips == 1 else [1024, 2048, 4096])
  pipelines = pipelines_override if pipelines_override else ["lidar_only_128x2048", "lidar_xyz_plus_stereo_128"]
  rows: List[Dict[str, Any]] = []

  for B in batches:
    dx_b, ctrl = _make_batch(m, B, vary_ctrl=True)
    if n_chips > 1:
      dx_b = jax.device_put(dx_b, sh)
      ctrl = jax.device_put(ctrl, sh)
      mx_d = jax.device_put(mx, rep)

      _phys_sm = shard_map(
          lambda d, u, mxl: jax.vmap(lambda x, c: mjx.step(mxl, x.replace(ctrl=c)))(d, u),
          mesh=mesh, in_specs=(P("dp"), P("dp"), P()), out_specs=P("dp"), **sm_kw,
      )
      phys_fn = jax.jit(lambda d, u: _phys_sm(d, u, mx_d))
    else:
      dx_b = jax.device_put(dx_b, devs[0])
      ctrl = jax.device_put(ctrl, devs[0])
      phys_fn = jax.jit(lambda d, u: jax.vmap(lambda x, c: mjx.step(mx, x.replace(ctrl=c)))(d, u))

    dp = phys_fn(dx_b, ctrl)
    jax.block_until_ready(dp.qpos)
    t0 = time.perf_counter()
    for _ in range(5):
      dp = phys_fn(dp, ctrl)
    jax.block_until_ready(dp.qpos)
    dt_phys = (time.perf_counter() - t0) / 5.0

    for pipeline in pipelines:
      with_stereo = (pipeline == "lidar_xyz_plus_stereo_128")
      with_xyz = with_stereo

      if n_chips > 1:
        def _local_fn(d, u, mxl):
          dn = jax.vmap(lambda x, c: mjx.step(mxl, x.replace(ctrl=c)))(d, u)
          lidar = pallas_lidar_128x2048(
              spec_lidar, sem_ids, dn, lidar_cam_id, ldir_np,
              compute_xyz=with_xyz, interpret=pallas_interp,
          )
          if not with_stereo:
            return (dn, lidar.range_m, lidar.semantic_id)
          gb_l, gb_r = pallas_raycast_ncam_v2([spec_l, spec_r], dn, interpret=pallas_interp)
          sem_l = geom_id_to_semantic_nogather(gb_l.geom_id, spec_l.enabled_geom_ids, sem_ids)
          sem_r = geom_id_to_semantic_nogather(gb_r.geom_id, spec_r.enabled_geom_ids, sem_ids)
          return (dn, lidar.range_m, lidar.semantic_id, lidar.xyz, gb_l.rgb, gb_l.depth, sem_l, gb_r.rgb, gb_r.depth, sem_r)

        n_outs = 10 if with_stereo else 3
        _sm = shard_map(
            _local_fn,
            mesh=mesh, in_specs=(P("dp"), P("dp"), P()), out_specs=(P("dp"),) * n_outs, **sm_kw,
        )
        e2e_fn = jax.jit(lambda d, u: _sm(d, u, mx_d))
      else:
        @jax.jit
        def e2e_fn(d, u):
          dn = jax.vmap(lambda x, c: mjx.step(mx, x.replace(ctrl=c)))(d, u)
          lidar = pallas_lidar_128x2048(
              spec_lidar, sem_ids, dn, lidar_cam_id, ldir_np,
              compute_xyz=with_xyz, interpret=pallas_interp,
          )
          if not with_stereo:
            return (dn, lidar.range_m, lidar.semantic_id)
          gb_l, gb_r = pallas_raycast_ncam_v2([spec_l, spec_r], dn, interpret=pallas_interp)
          sem_l = geom_id_to_semantic_nogather(gb_l.geom_id, spec_l.enabled_geom_ids, sem_ids)
          sem_r = geom_id_to_semantic_nogather(gb_r.geom_id, spec_r.enabled_geom_ids, sem_ids)
          return (dn, lidar.range_m, lidar.semantic_id, lidar.xyz, gb_l.rgb, gb_l.depth, sem_l, gb_r.rgb, gb_r.depth, sem_r)

      try:
        hlo = _audit_hlo(e2e_fn.lower(dx_b, ctrl))
        tc0 = time.perf_counter()
        dc = dx_b
        for _ in range(2):
          out = e2e_fn(dc, ctrl)
          dc = out[0]
        jax.block_until_ready(out[1:])
        warmup_s = time.perf_counter() - tc0

        n_st = 4 if B <= 1024 else 3
        t1 = time.perf_counter()
        for _ in range(n_st):
          out = e2e_fn(dc, ctrl)
          dc = out[0]
        jax.block_until_ready(out[1:])
        dt_e2e = (time.perf_counter() - t1) / n_st
        dt_sens = max(dt_e2e - dt_phys, 0.0)
        fps = float(B / max(dt_e2e, 1e-9))
        rays_per_env = LIDAR_RAYS + (2 * 128 * 128 if with_stereo else 0)
        mrays_s = fps * rays_per_env / 1e6
        lidar_mpts_s = fps * LIDAR_RAYS / 1e6

        rec = {
            "status": "ok",
            "pipeline": pipeline,
            "n_chips": n_chips,
            "batch_size": B,
            "lidar_beams": LIDAR_BEAMS,
            "lidar_azimuth": LIDAR_AZIMUTH,
            "lidar_rays_per_env": LIDAR_RAYS,
            "stereo_res": "2x128x128" if with_stereo else None,
            "dt_physics_ms": dt_phys * 1e3,
            "dt_sensor_ms": dt_sens * 1e3,
            "dt_e2e_ms": dt_e2e * 1e3,
            "e2e_fps": fps,
            "mrays_per_s": mrays_s,
            "lidar_mpoints_per_s": lidar_mpts_s,
            "warmup_s": round(warmup_s, 3),
            "host_callbacks": hlo["host_callbacks"],
            "send_recv_ops": hlo["send_recv_ops"],
            "gather_ops": hlo["gather_ops"],
            "hlo_zero_copy": True,
        }
        print(
            f"WAVE4 {pipeline} {n_chips}c B={B}: e2e={rec['dt_e2e_ms']:.2f}ms "
            f"fps={fps:.1f} ({mrays_s:.1f} Mrays/s, {lidar_mpts_s:.1f} MPts/s)",
            flush=True,
        )
      except Exception as exc:
        rec = {
            "status": "error",
            "pipeline": pipeline,
            "n_chips": n_chips,
            "batch_size": B,
            "error": f"{type(exc).__name__}: {str(exc)[:300]}",
        }
        print(f"WAVE4 {pipeline} {n_chips}c B={B} ERROR: {rec['error']}", flush=True)
      rows.append(rec)
      if out_json:
        with open(out_json, "w") as f:
          json.dump({"n_chips": n_chips, "runs": rows}, f, indent=2)
  return rows


def export_vis_data(export_dir: str) -> Dict[str, Any]:
  """Render high-res cluttered tabletop overview + stereo L/R + 128x2048 LiDAR scan for 4K figure."""
  pallas_interp = (jax.default_backend() == "cpu")
  xml = make_isaac_tabletop_lidar_xml()
  m = mujoco.MjModel.from_xml_string(xml)
  mx = mjx.put_model(m, impl="jax")
  lidar_cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "lidar_rig")
  cam_ov_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "cam6_diag")
  cam_l_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "cam2_stereo_l")
  cam_r_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "cam3_stereo_r")

  dx1, _ = _make_batch(m, 1, vary_ctrl=False)
  dev0 = jax.devices()[0]
  dx1 = jax.device_put(dx1, dev0)

  # 1920x1080 (block-aligned: 4,050 * 512) overview & 960x544 (1,020 * 512) stereo pair
  spec_ov = RenderSceneSpec.from_mjmodel(m, cam_id=cam_ov_id, height=1080, width=1920, use_shadows=True)
  spec_l = RenderSceneSpec.from_mjmodel(m, cam_id=cam_l_id, height=544, width=960, use_shadows=True)
  spec_r = RenderSceneSpec.from_mjmodel(m, cam_id=cam_r_id, height=544, width=960, use_shadows=True)
  spec_lidar = RenderSceneSpec.from_mjmodel(m, cam_id=lidar_cam_id, height=64, width=64, use_shadows=False)
  sem_ids = assign_semantic_classes(m, spec_lidar.enabled_geom_ids)
  ldir_np = build_lidar_ray_templates()

  fn_ov = jax.jit(lambda d: pallas_raycast_ncam_v2([spec_ov], d, interpret=pallas_interp)[0])
  fn_st = jax.jit(lambda d: pallas_raycast_ncam_v2([spec_l, spec_r], d, interpret=pallas_interp))
  fn_li = jax.jit(lambda d: pallas_lidar_128x2048(spec_lidar, sem_ids, d, lidar_cam_id, ldir_np, compute_xyz=True, interpret=pallas_interp))

  gb_ov = fn_ov(dx1)
  gb_l, gb_r = fn_st(dx1)
  sem_l = geom_id_to_semantic_nogather(gb_l.geom_id, spec_l.enabled_geom_ids, sem_ids)
  li = fn_li(dx1)
  jax.block_until_ready((gb_ov.rgb, gb_l.rgb, gb_r.rgb, li.xyz))

  os.makedirs(export_dir, exist_ok=True)
  np.savez(
      os.path.join(export_dir, "wave4_vis_data.npz"),
      ov_rgb=np.asarray(gb_ov.rgb[0], np.float32),
      st_l_rgb=np.asarray(gb_l.rgb[0, :540, :, :], np.float32),
      st_r_rgb=np.asarray(gb_r.rgb[0, :540, :, :], np.float32),
      st_l_depth=np.asarray(gb_l.depth[0, :540, :], np.float32),
      st_l_sem=np.asarray(sem_l[0, :540, :], np.int32),
      lidar_range=np.asarray(li.range_m[0], np.float32),
      lidar_sem=np.asarray(li.semantic_id[0], np.int32),
      lidar_xyz=np.asarray(li.xyz[0], np.float32),
  )
  print(f"Exported wave4_vis_data.npz to {export_dir}", flush=True)
  return {"status": "ok", "export_path": os.path.join(export_dir, "wave4_vis_data.npz")}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--phase", required=True, choices=["parity", "bench", "vis", "merge"])
  ap.add_argument("--n_chips", type=int, default=1)
  ap.add_argument("--batches", type=int, nargs="*", default=None)
  ap.add_argument("--pipelines", nargs="*", default=None)
  ap.add_argument("--out_json", required=True)
  ap.add_argument("--export_dir", default="/dev/shm/zaychik-tpu-mjx-w6/out/vis")
  ap.add_argument("--merge_inputs", nargs="*", default=[])
  args = ap.parse_args()

  os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
  if args.phase == "parity":
    res = run_parity_suite()
  elif args.phase == "bench":
    res = {"n_chips": args.n_chips, "runs": run_bench_suite(args.n_chips, args.batches, args.pipelines, args.out_json)}
  elif args.phase == "vis":
    res = export_vis_data(args.export_dir)
  else:
    merged: Dict[str, Any] = {"experiment": "v3_w6_wave4_isaac_sensors", "worker": "w6", "parity": {}, "runs": []}
    for p in args.merge_inputs:
      with open(p, "r") as f:
        d = json.load(f)
      if "range_mae_m" in d:
        merged["parity"] = d
      if "runs" in d:
        merged["runs"].extend(d["runs"])
    res = merged

  with open(args.out_json, "w") as f:
    json.dump(res, f, indent=2)
  print(f"Saved {args.out_json}", flush=True)


if __name__ == "__main__":
  main()
