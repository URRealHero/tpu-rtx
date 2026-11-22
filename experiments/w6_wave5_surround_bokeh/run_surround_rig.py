"""Wave 5 Task A1: 4-Chip 8-Cam (256/512/1080p) + Autonomous Surround Sensor Rig.

Benchmarks on live 4-chip TPU v5p (interpret=False, zero-copy HLO):
1. 4-Chip 8-Camera Fused vs Sequential throughput across 256x256, 512x512, and 1080p (1920x1080).
2. Autonomous Surround Sensor Rig:
   mjx.step(mx, dx) + 8 surround 1080p RGB+Depth+Semantic cameras + 2 x 128x2048 360-deg LiDARs
   (524,288 LiDAR rays/env + 16,588,800 camera rays/env = 17,113,088 rays/env)
   fused inside ONE XLA/shard_map graph with host_callbacks == 0, send_recv_ops == 0, gather_ops == 0.
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

from experiments.w6_highres_multicam.isaac_lidar_sensors import (
    LIDAR_AZIMUTH,
    LIDAR_BEAMS,
    LIDAR_RAYS,
    assign_semantic_classes,
    build_lidar_ray_templates,
    geom_id_to_semantic_nogather,
    jax_lidar_ref,
)
from experiments.w6_highres_multicam.wave3_8k_multicam8 import (
    make_8cam_panda_xml,
    pallas_raycast_ncam_v2,
    run_ncam_suite,
)
from tpu_mjx_render.pallas_v2 import (
    BLOCK_B,
    BLOCK_R,
    _compute_geom_bound_sq,
    _ray_one_geom_v2,
)
from tpu_mjx_render.renderer import RenderSceneSpec
from tpu_mjx_render.roofline import inspect_hlo_ops
from tpu_mjx_render.scenes import PANDA_PICK_CUBE_XML


class DualLidarScan(NamedTuple):
  """Fused dual 128x2048 360-degree LiDAR outputs (all in TPU HBM)."""

  range_m: jax.Array       # (B, 2, 128, 2048) float32 (-1.0 on miss)
  semantic_id: jax.Array   # (B, 2, 128, 2048) int32
  geom_id: jax.Array       # (B, 2, 128, 2048) int32 (-1 on miss)
  xyz: Optional[jax.Array] # (B, 2, 128, 2048, 3) float32


def _audit_hlo(lowered: Any) -> Dict[str, Any]:
  txt = lowered.as_text()
  info = inspect_hlo_ops(txt)
  info["send_recv_ops"] = len(re.findall(r"\bsend\(|\brecv\(", txt, re.I))
  assert info["host_callbacks"] == 0, f"HLO contains host_callback: {info}"
  assert info["send_recv_ops"] == 0, f"HLO contains send/recv: {info}"
  assert info["gather_ops"] == 0, f"HLO contains gather_ops: {info}"
  info["zero_copy_verified"] = True
  return info


def make_autonomous_surround_rig_xml() -> str:
  """360-deg Autonomous Surround Rig: 8 surround cameras + 2 LiDARs + rich surround obstacles."""
  xml = PANDA_PICK_CUBE_XML
  side_line = '    <camera fovy="45" name="side" pos="0.60 -0.85 0.55" xyaxes="1 0 0 0 0.45 0.89"/>\n'
  xml = xml.replace(side_line, "")

  # 8 surround 360-deg cameras around the mobile manipulation platform + 2 LiDAR rigs
  rig_cams = (
      '    <camera name="cam0_front" fovy="52" pos="0.22 0.00 0.78" xyaxes="0 -1 0 0.35 0 0.94"/>\n'
      '    <camera name="cam1_front_left" fovy="55" pos="0.18 0.24 0.72" xyaxes="-0.64 -0.77 0 0.24 -0.20 0.95"/>\n'
      '    <camera name="cam2_front_right" fovy="55" pos="0.18 -0.24 0.72" xyaxes="0.64 -0.77 0 0.24 0.20 0.95"/>\n'
      '    <camera name="cam3_left" fovy="58" pos="0.00 0.34 0.68" xyaxes="-1 0 0 0 -0.28 0.96"/>\n'
      '    <camera name="cam4_right" fovy="58" pos="0.00 -0.34 0.68" xyaxes="1 0 0 0 0.28 0.96"/>\n'
      '    <camera name="cam5_rear_left" fovy="58" pos="-0.22 0.24 0.70" xyaxes="-0.70 0.71 0 -0.22 -0.22 0.95"/>\n'
      '    <camera name="cam6_rear_right" fovy="58" pos="-0.22 -0.24 0.70" xyaxes="0.70 0.71 0 -0.22 0.22 0.95"/>\n'
      '    <camera name="cam7_overhead_mast" fovy="62" pos="0.45 0.00 1.48" xyaxes="0 1 0 -1 0 0"/>\n'
      '    <camera name="lidar_roof_main" fovy="60" pos="0.15 0.00 0.92" xyaxes="0 -1 0 1 0 0"/>\n'
      '    <camera name="lidar_bumper_low" fovy="60" pos="0.34 0.00 0.28" xyaxes="0 -1 0 1 0 0"/>\n'
  )
  front_anchor = '    <camera fovy="42" name="front" pos="1.176 0.001 0.483" xyaxes="0.006 1.000 -0.000 -0.720 0.004 0.694"/>\n'
  assert front_anchor in xml
  xml = xml.replace(front_anchor, rig_cams)

  # Add compact 360-deg surround obstacles (pillars, bins, bollards, vehicles/crates) keeping total G compact
  surround_geoms = """
    <geom name="workbench_top" type="box" pos="0.65 0.0 0.012" size="0.38 0.48 0.012" rgba="0.52 0.38 0.24 1" contype="0" conaffinity="0"/>
    <geom name="bin_blue" type="box" pos="0.56 -0.32 0.055" size="0.09 0.09 0.032" rgba="0.12 0.45 0.88 1" contype="0" conaffinity="0"/>
    <geom name="bin_green" type="box" pos="0.56 0.32 0.055" size="0.09 0.09 0.032" rgba="0.16 0.70 0.36 1" contype="0" conaffinity="0"/>
    <geom name="pillar_fl" type="cylinder" pos="1.45 0.95 0.45" size="0.12 0.45" rgba="0.88 0.34 0.14 1" contype="0" conaffinity="0"/>
    <geom name="pillar_fr" type="cylinder" pos="1.45 -0.95 0.45" size="0.12 0.45" rgba="0.88 0.34 0.14 1" contype="0" conaffinity="0"/>
    <geom name="crate_left" type="box" pos="0.10 1.25 0.32" euler="0 0 0.2" size="0.22 0.22 0.32" rgba="0.24 0.56 0.84 1" contype="0" conaffinity="0"/>
    <geom name="crate_right" type="box" pos="0.10 -1.25 0.28" euler="0 0 -0.3" size="0.20 0.26 0.28" rgba="0.85 0.65 0.16 1" contype="0" conaffinity="0"/>
    <geom name="barrier_rear" type="box" pos="-1.35 0.0 0.38" size="0.18 0.95 0.38" rgba="0.30 0.34 0.40 1" contype="0" conaffinity="0"/>
    <geom name="bollard_rl" type="capsule" pos="-0.95 0.85 0.35" size="0.09 0.26" rgba="0.92 0.22 0.22 1" contype="0" conaffinity="0"/>
    <geom name="bollard_rr" type="capsule" pos="-0.95 -0.85 0.35" size="0.09 0.26" rgba="0.92 0.22 0.22 1" contype="0" conaffinity="0"/>
    <geom name="target_sphere" type="sphere" pos="0.72 0.14 0.07" size="0.045" rgba="0.95 0.78 0.14 1" contype="0" conaffinity="0"/>
    <geom name="lidar_puck_roof" type="cylinder" pos="0.15 0.00 0.92" size="0.05 0.03" rgba="0.12 0.55 0.95 1" contype="0" conaffinity="0"/>
  """
  anchor = '    <body mocap="true" name="mocap_target" pos="0.65 0 0.25">'
  assert anchor in xml
  xml = xml.replace(anchor, surround_geoms + "\n" + anchor)
  xml = xml.replace('<body name="box" pos="0.65 0 0.03">', '<body name="box" pos="0.64 -0.02 0.054">')
  return xml


def pallas_dual_lidar_128x2048(
    spec: RenderSceneSpec,
    sem_ids_static: Tuple[int, ...],
    mjx_data_batch: Any,
    lidar_cam_ids: Tuple[int, int],
    ldir_np: np.ndarray,
    r_min: float = 0.08,
    max_range: float = 18.0,
    compute_xyz: bool = True,
    interpret: bool = False,
) -> DualLidarScan:
  """Fused single-kernel-launch dual 128x2048 360-deg LiDAR (524,288 rays/env) sharing geom_fields DMA."""
  assert len(lidar_cam_ids) == 2
  R = ldir_np.shape[0]
  assert R == LIDAR_RAYS

  gids = np.asarray(spec.enabled_geom_ids, dtype=np.int32)
  G = len(gids)
  loc_block_r = 256 if G > 36 else BLOCK_R
  assert R % loc_block_r == 0
  n_r_blk_per_lidar = R // loc_block_r
  R_total = 2 * R
  n_r_blk = 2 * n_r_blk_per_lidar

  s_pos0 = mjx_data_batch.cam_xpos[:, int(lidar_cam_ids[0]), :]
  s_mat0 = mjx_data_batch.cam_xmat[:, int(lidar_cam_ids[0]), :, :]
  s_pos1 = mjx_data_batch.cam_xpos[:, int(lidar_cam_ids[1]), :]
  s_mat1 = mjx_data_batch.cam_xmat[:, int(lidar_cam_ids[1]), :, :]

  geom_xpos = mjx_data_batch.geom_xpos
  geom_xmat = mjx_data_batch.geom_xmat
  geom_size = jnp.asarray(spec.geom_size, dtype=jnp.float32)

  B = s_pos0.shape[0]
  B_pad = ((B + BLOCK_B - 1) // BLOCK_B) * BLOCK_B
  G_pad = ((G + 127) // 128) * 128

  cam_pack = jnp.concatenate(
      [s_pos0, s_mat0.reshape(B, 9), s_pos1, s_mat1.reshape(B, 9)], axis=-1
  )  # (B, 24)
  cam_128 = jnp.pad(cam_pack, ((0, B_pad - B), (0, 128 - 24)))

  lray_raw = np.zeros((8, R_total), dtype=np.float32)
  lray_raw[0:3, :R] = ldir_np.T
  lray_raw[0:3, R:] = ldir_np.T
  lray_dev = jnp.asarray(lray_raw, dtype=jnp.float32)

  gpos = geom_xpos[:, gids, :]
  gmat = geom_xmat[:, gids, :, :].reshape(B, G, 9)
  gsz_all = jnp.broadcast_to(geom_size[gids][None, :, :], (B, G, 3))
  rb_sq = jnp.broadcast_to(jnp.asarray(_compute_geom_bound_sq(spec, gids))[None, :, None], (B, G, 1))
  g_16 = jnp.concatenate([gpos, gmat, gsz_all, rb_sq], axis=-1)
  g_16 = jnp.pad(g_16, ((0, B_pad - B), (0, G_pad - G), (0, 0)))

  n_b_blk = B_pad // BLOCK_B
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

  def _dual_lidar_kernel(cam_ref, lray_ref, gf_ref, out_r_ref, out_sem_ref, out_gid_ref):
    cam = cam_ref[...]
    lr = lray_ref[...]
    gf = gf_ref[...]

    lid_idx = pl.program_id(1) // n_r_blk_per_lidar
    is_l1 = lid_idx == 1
    c_cols = [jnp.where(is_l1, cam[:, 12 + i : 13 + i], cam[:, i : i + 1]) for i in range(12)]
    cpx, cpy, cpz, c00, c01, c02, c10, c11, c12, c20, c21, c22 = c_cols

    ldx, ldy, ldz = lr[0:1, :], lr[1:2, :], lr[2:3, :]
    rdx_raw = c00 * ldx + c01 * ldy + c02 * ldz
    rdy_raw = c10 * ldx + c11 * ldy + c12 * ldz
    rdz_raw = c20 * ldx + c21 * ldy + c22 * ldz
    inv_rn = 1.0 / jnp.maximum(jnp.sqrt(rdx_raw * rdx_raw + rdy_raw * rdy_raw + rdz_raw * rdz_raw), 1e-12)
    rdx, rdy, rdz = rdx_raw * inv_rn, rdy_raw * inv_rn, rdz_raw * inv_rn

    rox = cpx + r_min_f * rdx
    roy = cpy + r_min_f * rdy
    roz = cpz + r_min_f * rdz

    best_d = jnp.full((BLOCK_B, loc_block_r), d_limit_f, dtype=jnp.float32)
    best_gid = jnp.full((BLOCK_B, loc_block_r), -1, dtype=jnp.int32)
    best_sem = jnp.zeros((BLOCK_B, loc_block_r), dtype=jnp.int32)

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
          gt, px, py, pz,
          m00, m01, m02, m10, m11, m12, m20, m21, m22,
          sx, sy, sz, rb_sq,
          rox, roy, roz, rdx, rdy, rdz,
          plane_finite_x=pfx, plane_finite_y=pfy, compute_normal=False,
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
      _dual_lidar_kernel,
      out_shape=[
          jax.ShapeDtypeStruct((B_pad, R_total), jnp.float32),
          jax.ShapeDtypeStruct((B_pad, R_total), jnp.int32),
          jax.ShapeDtypeStruct((B_pad, R_total), jnp.int32),
      ],
      grid=(n_b_blk, n_r_blk),
      in_specs=[
          pl.BlockSpec((BLOCK_B, 128), lambda bi, ri: (bi, 0)),
          pl.BlockSpec((8, loc_block_r), lambda bi, ri: (0, ri)),
          pl.BlockSpec((16 * BLOCK_B, G_pad), lambda bi, ri: (bi, 0)),
      ],
      out_specs=[
          pl.BlockSpec((BLOCK_B, loc_block_r), lambda bi, ri: (bi, ri)),
          pl.BlockSpec((BLOCK_B, loc_block_r), lambda bi, ri: (bi, ri)),
          pl.BlockSpec((BLOCK_B, loc_block_r), lambda bi, ri: (bi, ri)),
      ],
      compiler_params=None if interpret else pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
      interpret=interpret,
  )(cam_128, lray_dev, geom_fields)

  range_2 = r_out[:B, :R_total].reshape(B, 2, LIDAR_BEAMS, LIDAR_AZIMUTH)
  sem_2 = sem_out[:B, :R_total].reshape(B, 2, LIDAR_BEAMS, LIDAR_AZIMUTH)
  gid_2 = gid_out[:B, :R_total].reshape(B, 2, LIDAR_BEAMS, LIDAR_AZIMUTH)

  xyz_2 = None
  if compute_xyz:
    ld = jnp.asarray(ldir_np, dtype=jnp.float32)
    xyz_list = []
    for sp, sm, rf, gf_i in ((s_pos0, s_mat0, range_2[:, 0], gid_2[:, 0]), (s_pos1, s_mat1, range_2[:, 1], gid_2[:, 1])):
      rdir = jnp.einsum("bij,rj->bri", sm, ld, precision=jax.lax.Precision.HIGHEST)
      rdir = rdir / jnp.maximum(jnp.linalg.norm(rdir, axis=-1, keepdims=True), 1e-12)
      hit = gf_i.reshape(B, R) >= 0
      pts = sp[:, None, :] + jnp.where(hit, rf.reshape(B, R), 0.0)[..., None] * rdir
      xyz_list.append(jnp.where(hit[..., None], pts, 0.0).reshape(B, LIDAR_BEAMS, LIDAR_AZIMUTH, 3))
    xyz_2 = jnp.stack(xyz_list, axis=1)

  return DualLidarScan(range_m=range_2, semantic_id=sem_2, geom_id=gid_2, xyz=xyz_2)


def _make_batch(m: mujoco.MjModel, batch_size: int):
  d_host = mujoco.MjData(m)
  if m.nkey > 0:
    mujoco.mj_resetDataKeyframe(m, d_host, 0)
  mujoco.mj_forward(m, d_host)
  dx0 = mjx.put_data(m, d_host, impl="jax")
  dx_batch = jax.tree_util.tree_map(
      lambda x: jnp.broadcast_to(x[None, ...], (batch_size, *x.shape)).copy(), dx0
  )
  phases = jnp.linspace(-0.25, 0.25, batch_size, dtype=jnp.float32)[:, None]
  ctrl = jnp.asarray(dx0.ctrl, jnp.float32)[None, :] + phases * jnp.linspace(1.0, 0.3, m.nu, dtype=jnp.float32)[None, :]
  return dx_batch, ctrl


def run_surround_rig_suite(n_chips: int = 4, batches: Optional[List[int]] = None) -> Dict[str, Any]:
  """Parity check + 4-chip benchmark for Autonomous Surround Sensor Rig (8x1080p + 2x128x2048 LiDAR)."""
  pallas_interp = (jax.default_backend() == "cpu")
  devs = jax.devices()[:n_chips]
  assert len(devs) == n_chips

  xml = make_autonomous_surround_rig_xml()
  m = mujoco.MjModel.from_xml_string(xml)
  mx = mjx.put_model(m, impl="jax")

  cam_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, f"cam{i}_{s}") for i, s in enumerate(
      ["front", "front_left", "front_right", "left", "right", "rear_left", "rear_right", "overhead_mast"]
  )]
  lid0 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "lidar_roof_main")
  lid1 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "lidar_bumper_low")
  assert all(cid >= 0 for cid in cam_ids) and lid0 >= 0 and lid1 >= 0

  specs_8 = [RenderSceneSpec.from_mjmodel(m, cam_id=cid, height=1080, width=1920, use_shadows=True) for cid in cam_ids]
  spec_lidar = RenderSceneSpec.from_mjmodel(m, cam_id=lid0, height=64, width=64, use_shadows=False)
  sem_ids = assign_semantic_classes(m, spec_lidar.enabled_geom_ids)
  ldir_np = build_lidar_ray_templates()
  n_enabled_geoms = len(spec_lidar.enabled_geom_ids)

  # 1) Dual-LiDAR exact parity verification on device 0
  dx_p, ctrl_p = _make_batch(m, 8)
  dx_p = jax.device_put(dx_p, devs[0])
  ctrl_p = jax.device_put(ctrl_p, devs[0])
  dx_p = jax.jit(lambda d, u: jax.lax.fori_loop(0, 6, lambda _, x: jax.vmap(lambda a, c: mjx.step(mx, a.replace(ctrl=c)))(x, u), d))(dx_p, ctrl_p)
  jax.block_until_ready(dx_p.qpos)

  ref0 = jax.jit(lambda d: jax_lidar_ref(spec_lidar, sem_ids, d, lid0, ldir_np, r_min=0.08, max_range=18.0, compute_xyz=True))(dx_p)
  ref1 = jax.jit(lambda d: jax_lidar_ref(spec_lidar, sem_ids, d, lid1, ldir_np, r_min=0.08, max_range=18.0, compute_xyz=True))(dx_p)
  fn_dual = jax.jit(lambda d: pallas_dual_lidar_128x2048(spec_lidar, sem_ids, d, (lid0, lid1), ldir_np, r_min=0.08, max_range=18.0, compute_xyz=True, interpret=pallas_interp))
  hlo_dual_lidar = _audit_hlo(fn_dual.lower(dx_p))
  pal_dual = fn_dual(dx_p)
  jax.block_until_ready((pal_dual.range_m, pal_dual.xyz))

  mae0 = float(jnp.mean(jnp.abs(pal_dual.range_m[:, 0] - ref0.range_m)))
  mae1 = float(jnp.mean(jnp.abs(pal_dual.range_m[:, 1] - ref1.range_m)))
  gm0 = float(jnp.mean(pal_dual.geom_id[:, 0] == ref0.geom_id))
  gm1 = float(jnp.mean(pal_dual.geom_id[:, 1] == ref1.geom_id))
  xyz_mae = float(0.5 * (jnp.mean(jnp.abs(pal_dual.xyz[:, 0] - ref0.xyz)) + jnp.mean(jnp.abs(pal_dual.xyz[:, 1] - ref1.xyz))))
  assert mae0 < 1e-5 and mae1 < 1e-5 and gm0 == 1.0 and gm1 == 1.0, (mae0, mae1, gm0, gm1)

  parity = {
      "n_enabled_geoms": n_enabled_geoms,
      "lidar0_range_mae_m": mae0,
      "lidar1_range_mae_m": mae1,
      "lidar0_geom_match": gm0,
      "lidar1_geom_match": gm1,
      "dual_xyz_mae_m": xyz_mae,
      "hlo_dual_lidar": hlo_dual_lidar,
      "passed": True,
  }
  print("DUAL LIDAR PARITY PASS:", json.dumps(parity, indent=2), flush=True)

  # 2) Full Fused Autonomous Surround Sensor Rig benchmark on 4 chips
  mesh = jax.sharding.Mesh(np.array(devs), ("dp",))
  P = jax.sharding.PartitionSpec
  sh = jax.sharding.NamedSharding(mesh, P("dp"))
  rep = jax.sharding.NamedSharding(mesh, P())
  sm_kw = {"check_rep": False} if "check_rep" in inspect.signature(shard_map).parameters else {"check_vma": False}
  mx_d = jax.device_put(mx, rep)

  cam_rays_per_env = 8 * 1920 * 1080      # 16,588,800
  lidar_rays_per_env = 2 * LIDAR_RAYS     # 524,288
  total_rays_per_env = cam_rays_per_env + lidar_rays_per_env  # 17,113,088

  batch_list = batches if batches else [4, 8, 16]
  runs: List[Dict[str, Any]] = []
  for B in batch_list:
    dx_b, ctrl = _make_batch(m, B)
    dx_b = jax.device_put(dx_b, sh)
    ctrl = jax.device_put(ctrl, sh)

    _phys_sm = shard_map(
        lambda d, u, mxl: jax.vmap(lambda x, c: mjx.step(mxl, x.replace(ctrl=c)))(d, u),
        mesh=mesh, in_specs=(P("dp"), P("dp"), P()), out_specs=P("dp"), **sm_kw,
    )
    phys_fn = jax.jit(lambda d, u: _phys_sm(d, u, mx_d))
    dp = phys_fn(dx_b, ctrl)
    jax.block_until_ready(dp.qpos)
    t0 = time.perf_counter()
    for _ in range(4):
      dp = phys_fn(dp, ctrl)
    jax.block_until_ready(dp.qpos)
    dt_phys = (time.perf_counter() - t0) / 4.0

    def _surround_local(d, u, mxl):
      dn = jax.vmap(lambda x, c: mjx.step(mxl, x.replace(ctrl=c)))(d, u)
      gbs = pallas_raycast_ncam_v2(specs_8, dn, interpret=pallas_interp)
      sems = tuple(geom_id_to_semantic_nogather(g.geom_id, specs_8[0].enabled_geom_ids, sem_ids) for g in gbs)
      lid = pallas_dual_lidar_128x2048(
          spec_lidar, sem_ids, dn, (lid0, lid1), ldir_np,
          r_min=0.08, max_range=18.0, compute_xyz=True, interpret=pallas_interp,
      )
      return (dn, lid.range_m, lid.semantic_id, lid.xyz) + tuple(g.rgb for g in gbs) + tuple(g.depth for g in gbs) + sems

    n_outs = 4 + 3 * 8
    _sm = shard_map(
        _surround_local,
        mesh=mesh, in_specs=(P("dp"), P("dp"), P()), out_specs=(P("dp"),) * n_outs, **sm_kw,
    )
    e2e_fn = jax.jit(lambda d, u: _sm(d, u, mx_d))
    lowered = e2e_fn.lower(dx_b, ctrl)
    hlo = _audit_hlo(lowered)

    tc0 = time.perf_counter()
    e2e_exec = lowered.compile()
    for _ in range(2):
      out = e2e_exec(dx_b, ctrl)
    jax.block_until_ready(out[1:6])
    warmup_s = time.perf_counter() - tc0

    n_iters = 3
    t1 = time.perf_counter()
    for _ in range(n_iters):
      out = e2e_exec(dx_b, ctrl)
    jax.block_until_ready(out[1:6])
    dt_e2e = (time.perf_counter() - t1) / n_iters

    fps = float(B / max(dt_e2e, 1e-9))
    rec = {
        "status": "ok",
        "pipeline": "surround_8x1080p_plus_2x128x2048_lidar",
        "n_chips": n_chips,
        "batch_size": B,
        "n_cameras": 8,
        "camera_res": "1920x1080",
        "n_lidars": 2,
        "lidar_res": "128x2048",
        "n_enabled_geoms": n_enabled_geoms,
        "rays_per_env": total_rays_per_env,
        "dt_physics_ms": dt_phys * 1e3,
        "dt_sensor_ms": max(0.0, dt_e2e - dt_phys) * 1e3,
        "dt_e2e_ms": dt_e2e * 1e3,
        "e2e_fps": fps,
        "cam_frames_per_s": fps * 8.0,
        "lidar_scans_per_s": fps * 2.0,
        "gpix_cam_per_s": (fps * cam_rays_per_env) / 1e9,
        "mrays_lidar_per_s": (fps * lidar_rays_per_env) / 1e6,
        "mrays_total_per_s": (fps * total_rays_per_env) / 1e6,
        "warmup_s": round(warmup_s, 3),
        "host_callbacks": hlo["host_callbacks"],
        "send_recv_ops": hlo["send_recv_ops"],
        "gather_ops": hlo["gather_ops"],
        "hlo_zero_copy": True,
    }
    runs.append(rec)
    print(
        f"SURROUND RIG 4c B={B}: e2e={rec['dt_e2e_ms']:.2f}ms fps={fps:.2f} "
        f"(cam_fps={rec['cam_frames_per_s']:.1f}, total={rec['mrays_total_per_s']:.1f} MRays/s, gather={hlo['gather_ops']})",
        flush=True,
    )
  return {"parity": parity, "runs": runs}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--phase", default="all", choices=["all", "ncam_4c", "surround_rig"])
  ap.add_argument("--out_json", default="/dev/shm/zaychik-tpu-mjx-w6/out/v3_w6_surround_rig.json")
  args = ap.parse_args()

  os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
  result: Dict[str, Any] = {
      "experiment": "v3_w6_surround_rig",
      "worker": "w6",
      "backend": jax.default_backend(),
      "device_kind": jax.devices()[0].device_kind,
      "n_chips": len(jax.devices()),
  }

  if args.phase in ("all", "ncam_4c"):
    ncam_rows = run_ncam_suite(
        n_chips=4,
        res_filter=[256, 512, 1080],
        ncams_filter=[8],
        out_json=args.out_json + ".ncam_tmp.json",
    )
    result["ncam8_4c_runs"] = ncam_rows
  elif os.path.exists(args.out_json + ".ncam_tmp.json"):
    with open(args.out_json + ".ncam_tmp.json", "r") as nf:
      result["ncam8_4c_runs"] = json.load(nf).get("ncam_runs", [])

  if args.phase in ("all", "surround_rig"):
    result["surround_rig"] = run_surround_rig_suite(n_chips=4, batches=[4, 8, 16])

  with open(args.out_json, "w") as f:
    json.dump(result, f, indent=2)
  print(f"Saved {args.out_json}", flush=True)


if __name__ == "__main__":
  main()
