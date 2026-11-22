"""Wave 3: Native 8K UHD (7680x4320) 4-Chip Spatial & Fused 4/8-Camera Surround-View Pallas Kernel.

Also includes the richly cluttered multi-object robotic tabletop scene and
native 4K (3840x2160) 8-camera grid showcase renderer.
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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np

from experiments.w6_highres_multicam.bench_4k import (
    SpatialRelighterDtype,
    _pallas_raycast_strip,
)
from tpu_mjx_render.neural_shader import apply_neural_shader
from tpu_mjx_render.pallas_v2 import (
    BLOCK_B,
    BLOCK_R,
    _compute_batch_camera_rays_soa,
    _compute_geom_bound_sq,
    _extract_light_batch,
    _ray_one_geom_v2,
    _shade_batch_soa,
    build_local_camera_rays,
)
from tpu_mjx_render.primitives import MJ_MAXVAL
from tpu_mjx_render.renderer import GBuffer, RenderSceneSpec, render_batch
from tpu_mjx_render.scenes import PANDA_PICK_CUBE_XML, get_scene_xml

W_8K = 7680
H_8K = 4320
R_8K = W_8K * H_8K  # 33,177,600 = 64,800 * 512


def _audit_hlo(lowered: Any) -> Dict[str, Any]:
  from tpu_mjx_render.roofline import inspect_hlo_ops
  txt = lowered.as_text()
  info = inspect_hlo_ops(txt)
  info["send_recv_ops"] = len(re.findall(r"\bsend\(|\brecv\(", txt, re.I))
  assert info["host_callbacks"] == 0, f"HLO contains host_callback: {info}"
  assert info["send_recv_ops"] == 0, f"HLO contains send/recv: {info}"
  info["zero_copy_verified"] = True
  return info


def _make_batch(m: mujoco.MjModel, batch_size: int, vary_ctrl: bool = False):
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


def make_8cam_panda_xml() -> str:
  """PandaPickCube MJCF with 8 surround/wrist/stereo cameras."""
  xml = PANDA_PICK_CUBE_XML
  side_line = '    <camera fovy="45" name="side" pos="0.60 -0.85 0.55" xyaxes="1 0 0 0 0.45 0.89"/>\n'
  xml = xml.replace(side_line, "")

  world_cams = (
      '    <camera name="cam1_overhead" fovy="45" pos="0.55 0.00 1.45" xyaxes="0 1 0 -1 0 0"/>\n'
      '    <camera name="cam2_stereo_l" fovy="42" pos="1.15 -0.06 0.50" xyaxes="0.006 1.000 0.000 -0.720 0.004 0.694"/>\n'
      '    <camera name="cam3_stereo_r" fovy="42" pos="1.15 0.06 0.50" xyaxes="0.006 1.000 0.000 -0.720 0.004 0.694"/>\n'
      '    <camera name="cam4_left" fovy="45" pos="0.60 -0.85 0.55" xyaxes="1 0 0 0 0.45 0.89"/>\n'
      '    <camera name="cam5_right" fovy="45" pos="0.60 0.85 0.55" xyaxes="-1 0 0 0 -0.45 0.89"/>\n'
      '    <camera name="cam6_diag" fovy="45" pos="0.95 -0.65 0.72" xyaxes="0.56 0.83 0 -0.46 0.31 0.83"/>\n'
  )
  front_anchor = '    <camera fovy="42" name="front" pos="1.176 0.001 0.483" xyaxes="0.006 1.000 -0.000 -0.720 0.004 0.694"/>\n'
  assert front_anchor in xml
  xml = xml.replace(front_anchor, front_anchor + world_cams)

  wrist_anchor = '                      <site name="gripper" pos="0 0 0.10"/>\n'
  wrist_cam = '                      <camera name="cam7_wrist" fovy="55" pos="-0.06 0.0 0.04" xyaxes="0 1 0 0.8575 0 -0.5145"/>\n'
  assert wrist_anchor in xml
  xml = xml.replace(wrist_anchor, wrist_anchor + wrist_cam)
  return xml


def make_cluttered_tabletop_8cam_xml() -> str:
  """Richly populated cluttered tabletop manipulation scene with 8 cameras, bins, bottles, tools, rack."""
  xml = make_8cam_panda_xml()
  # Replace simple init_space + single box with a rich cluttered workbench + bins + tools + bottles + rack
  clutter = """
    <!-- Richly populated tabletop workbench, bins, tool rack, bottles, and manipulation props -->
    <geom name="workbench_top" type="box" pos="0.62 0.0 0.012" size="0.42 0.52 0.012" rgba="0.55 0.39 0.24 1" contype="1" conaffinity="1"/>
    <geom name="workbench_trim" type="box" pos="0.62 0.0 0.005" size="0.44 0.54 0.005" rgba="0.22 0.24 0.27 1" contype="0" conaffinity="0"/>

    <!-- Colorful sorting bins (front-left & front-right) -->
    <geom name="bin_blue_base" type="box" pos="0.50 -0.32 0.028" size="0.09 0.09 0.006" rgba="0.12 0.42 0.85 1" contype="0" conaffinity="0"/>
    <geom name="bin_blue_w1" type="box" pos="0.41 -0.32 0.062" size="0.006 0.09 0.032" rgba="0.12 0.42 0.85 1" contype="0" conaffinity="0"/>
    <geom name="bin_blue_w2" type="box" pos="0.59 -0.32 0.062" size="0.006 0.09 0.032" rgba="0.12 0.42 0.85 1" contype="0" conaffinity="0"/>
    <geom name="bin_blue_w3" type="box" pos="0.50 -0.41 0.062" size="0.09 0.006 0.032" rgba="0.12 0.42 0.85 1" contype="0" conaffinity="0"/>
    <geom name="bin_blue_w4" type="box" pos="0.50 -0.23 0.062" size="0.09 0.006 0.032" rgba="0.12 0.42 0.85 1" contype="0" conaffinity="0"/>

    <geom name="bin_green_base" type="box" pos="0.50 0.32 0.028" size="0.09 0.09 0.006" rgba="0.16 0.68 0.34 1" contype="0" conaffinity="0"/>
    <geom name="bin_green_w1" type="box" pos="0.41 0.32 0.062" size="0.006 0.09 0.032" rgba="0.16 0.68 0.34 1" contype="0" conaffinity="0"/>
    <geom name="bin_green_w2" type="box" pos="0.59 0.32 0.062" size="0.006 0.09 0.032" rgba="0.16 0.68 0.34 1" contype="0" conaffinity="0"/>
    <geom name="bin_green_w3" type="box" pos="0.50 0.23 0.062" size="0.09 0.006 0.032" rgba="0.16 0.68 0.34 1" contype="0" conaffinity="0"/>
    <geom name="bin_green_w4" type="box" pos="0.50 0.41 0.062" size="0.09 0.006 0.032" rgba="0.16 0.68 0.34 1" contype="0" conaffinity="0"/>

    <!-- Tool rack & pegs on back edge -->
    <geom name="rack_post_l" type="cylinder" pos="0.88 -0.24 0.14" size="0.015 0.12" rgba="0.25 0.26 0.28 1" contype="0" conaffinity="0"/>
    <geom name="rack_post_r" type="cylinder" pos="0.88 0.24 0.14" size="0.015 0.12" rgba="0.25 0.26 0.28 1" contype="0" conaffinity="0"/>
    <geom name="rack_bar" type="capsule" fromto="0.88 -0.25 0.23 0.88 0.25 0.23" size="0.014" rgba="0.82 0.56 0.16 1" contype="0" conaffinity="0"/>
    <geom name="rack_shelf" type="box" pos="0.88 0.0 0.12" size="0.045 0.24 0.008" rgba="0.32 0.34 0.38 1" contype="0" conaffinity="0"/>

    <!-- Bottles, canisters, spheres & tools scattered on tabletop -->
    <geom name="bottle_teal" type="cylinder" pos="0.72 -0.20 0.085" size="0.028 0.060" rgba="0.10 0.72 0.78 1" contype="0" conaffinity="0"/>
    <geom name="bottle_teal_cap" type="cylinder" pos="0.72 -0.20 0.155" size="0.014 0.012" rgba="0.95 0.95 0.95 1" contype="0" conaffinity="0"/>
    <geom name="bottle_amber" type="cylinder" pos="0.76 0.18 0.095" size="0.030 0.070" rgba="0.92 0.52 0.12 1" contype="0" conaffinity="0"/>
    <geom name="bottle_amber_cap" type="cylinder" pos="0.76 0.18 0.175" size="0.013 0.012" rgba="0.20 0.20 0.22 1" contype="0" conaffinity="0"/>
    <geom name="spray_can" type="capsule" pos="0.44 0.14 0.065" size="0.022 0.040" rgba="0.88 0.22 0.52 1" contype="0" conaffinity="0"/>

    <!-- Tools: wrench, mallet, screwdriver -->
    <geom name="wrench_handle" type="box" pos="0.56 -0.11 0.032" euler="0 0 0.45" size="0.075 0.012 0.008" rgba="0.78 0.80 0.84 1" contype="0" conaffinity="0"/>
    <geom name="wrench_head" type="cylinder" pos="0.63 -0.075 0.032" size="0.022 0.009" rgba="0.72 0.75 0.79 1" contype="0" conaffinity="0"/>
    <geom name="mallet_handle" type="capsule" fromto="0.38 0.04 0.036 0.52 0.12 0.036" size="0.011" rgba="0.74 0.50 0.26 1" contype="0" conaffinity="0"/>
    <geom name="mallet_head" type="cylinder" pos="0.53 0.125 0.045" euler="1.57 0.45 0" size="0.024 0.036" rgba="0.18 0.18 0.20 1" contype="0" conaffinity="0"/>

    <!-- Props inside bins and near target cube -->
    <geom name="prop_sphere_gold" type="sphere" pos="0.49 -0.31 0.058" size="0.024" rgba="0.96 0.76 0.14 1" contype="0" conaffinity="0"/>
    <geom name="prop_block_cyan" type="box" pos="0.52 -0.35 0.052" euler="0 0 0.3" size="0.020 0.020 0.018" rgba="0.18 0.80 0.90 1" contype="0" conaffinity="0"/>
    <geom name="prop_block_yellow" type="box" pos="0.48 0.31 0.054" euler="0 0 -0.4" size="0.022 0.022 0.020" rgba="0.95 0.85 0.18 1" contype="0" conaffinity="0"/>
    <geom name="prop_cyl_purple" type="cylinder" pos="0.53 0.34 0.060" size="0.018 0.026" rgba="0.64 0.24 0.84 1" contype="0" conaffinity="0"/>
    <geom name="target_cube_blue" type="box" pos="0.69 0.06 0.050" euler="0 0 0.5" size="0.024 0.024 0.026" rgba="0.18 0.48 0.94 1" contype="0" conaffinity="0"/>
    <geom name="target_cube_green" type="box" pos="0.58 0.09 0.044" euler="0 0 -0.2" size="0.020 0.020 0.020" rgba="0.22 0.82 0.32 1" contype="0" conaffinity="0"/>
    <geom name="mug_ellipsoid" type="ellipsoid" pos="0.73 -0.05 0.056" size="0.028 0.028 0.032" rgba="0.94 0.92 0.86 1" contype="0" conaffinity="0"/>
"""
  anchor = '    <body mocap="true" name="mocap_target" pos="0.65 0 0.25">'
  assert anchor in xml
  xml = xml.replace(anchor, clutter + "\n" + anchor)
  # Elevate free cube onto workbench top (z=0.054)
  xml = xml.replace('<body name="box" pos="0.65 0 0.03">', '<body name="box" pos="0.64 -0.02 0.054">')
  # Also update camera 5/6 slightly for clear clutter framing
  xml = xml.replace(
      'pos="0.95 -0.65 0.72" xyaxes="0.56 0.83 0 -0.46 0.31 0.83"',
      'pos="0.25 -0.58 0.42" xyaxes="0.82 0.57 0 -0.32 0.46 0.83"',
  )
  return xml


def pallas_raycast_ncam_v2(
    specs: Sequence[RenderSceneSpec],
    mjx_data_batch: Any,
    geom_rgba_batch: Optional[jax.Array] = None,
    interpret: bool = False,
) -> List[GBuffer]:
  """Fused N-camera (4/8/etc.) single-kernel-launch Pallas TPU raycaster sharing geom_fields DMA."""
  n_cams = len(specs)
  assert 1 <= n_cams <= 10, f"Single 128-wide cam_pack supports up to 10 cameras, got {n_cams}"
  s0 = specs[0]
  H, W = int(s0.height), int(s0.width)
  for s in specs[1:]:
    assert int(s.height) == H and int(s.width) == W
    assert s.enabled_geom_ids == s0.enabled_geom_ids

  gids = np.asarray(s0.enabled_geom_ids, dtype=np.int32)
  G = len(gids)
  R = H * W
  loc_block_r = 256 if G > 36 else BLOCK_R
  assert R % loc_block_r == 0, f"Requires R % {loc_block_r} == 0 for tile-aligned multi-cam, got R={R}"
  n_r_blk_cam = R // loc_block_r
  R_total = n_cams * R
  n_r_blk = n_cams * n_r_blk_cam

  cam_pos_list = [mjx_data_batch.cam_xpos[:, int(s.cam_id), :] for s in specs]
  cam_mat_list = [mjx_data_batch.cam_xmat[:, int(s.cam_id), :, :] for s in specs]
  B = cam_pos_list[0].shape[0]

  geom_xpos = mjx_data_batch.geom_xpos
  geom_xmat = mjx_data_batch.geom_xmat
  light_xpos, light_xdir = _extract_light_batch(s0, mjx_data_batch)
  geom_size = jnp.asarray(s0.geom_size, dtype=jnp.float32)
  geom_rgba = jnp.asarray(s0.geom_rgba, dtype=jnp.float32) if geom_rgba_batch is None else geom_rgba_batch

  B_pad = ((B + BLOCK_B - 1) // BLOCK_B) * BLOCK_B
  G_pad = ((G + 127) // 128) * 128

  cam_packs = []
  for cp, cm in zip(cam_pos_list, cam_mat_list):
    cam_packs.append(cp)
    cam_packs.append(cm.reshape(B, 9))
  cam_all = jnp.concatenate(cam_packs, axis=-1)  # (B, 12*n_cams)
  cam_128 = jnp.pad(cam_all, ((0, B_pad - B), (0, 128 - 12 * n_cams)))

  lray_raw = np.zeros((8, R_total), dtype=np.float32)
  for c_idx, s in enumerate(specs):
    ld_c, lo_c, nz_c = build_local_camera_rays(s)
    md_c = ((float(s.zfar) - float(s.znear)) / nz_c) if s.zfar > 0.0 else np.full((R,), MJ_MAXVAL, dtype=np.float32)
    sl = slice(c_idx * R, (c_idx + 1) * R)
    lray_raw[0:3, sl] = ld_c.T
    lray_raw[3:6, sl] = lo_c.T
    lray_raw[6, sl] = nz_c
    lray_raw[7, sl] = md_c
  lray_dev = jnp.asarray(lray_raw, dtype=jnp.float32)

  gpos = geom_xpos[:, gids, :]
  gmat = geom_xmat[:, gids, :, :].reshape(B, G, 9)
  gsz_all = jnp.broadcast_to(geom_size[gids][None, :, :], (B, G, 3))
  rb_sq = jnp.broadcast_to(jnp.asarray(_compute_geom_bound_sq(s0, gids))[None, :, None], (B, G, 1))
  g_16 = jnp.concatenate([gpos, gmat, gsz_all, rb_sq], axis=-1)
  g_16 = jnp.pad(g_16, ((0, B_pad - B), (0, G_pad - G), (0, 0)))

  n_b_blk = B_pad // BLOCK_B
  geom_fields = (
      g_16.reshape(n_b_blk, BLOCK_B, G_pad, 16)
      .transpose(0, 3, 1, 2)
      .reshape(n_b_blk * 16 * BLOCK_B, G_pad)
  )

  shadow_l_idx = -1
  for li in range(int(s0.nlight)):
    if s0.use_shadows and bool(s0.light_active[li]) and bool(s0.light_castshadow[li]):
      shadow_l_idx = li
      break

  light_128 = jnp.zeros((B_pad, 128), dtype=jnp.float32)
  if shadow_l_idx >= 0:
    is_dir = 1.0 if int(s0.light_type[shadow_l_idx]) == 1 else 0.0
    lp = light_xpos[:, shadow_l_idx, :]
    ld_vec = light_xdir[:, shadow_l_idx, :]
    ld_vec = ld_vec / jnp.maximum(jnp.linalg.norm(ld_vec, axis=-1, keepdims=True), 1e-12)
    l_row = jnp.concatenate([jnp.full((B, 1), is_dir, jnp.float32), lp, ld_vec], axis=-1)
    light_128 = jnp.pad(l_row, ((0, B_pad - B), (0, 128 - 7)))

  g_types_static = tuple(int(s0.geom_type[g]) for g in gids)
  gids_static = tuple(int(g) for g in gids)
  plane_fx = tuple(float(s0.geom_size[g, 0]) > 0.0 for g in gids)
  plane_fy = tuple(float(s0.geom_size[g, 1]) > 0.0 for g in gids)
  znear_f = float(s0.znear)
  do_in_vmem_shadow = (shadow_l_idx >= 0) and (G <= 56)

  def _ncam_kernel(
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
    max_d = jnp.broadcast_to(lr[7:8, :], (BLOCK_B, loc_block_r))

    cam_idx = pl.program_id(1) // n_r_blk_cam
    c_cols = [cam[:, i : i + 1] for i in range(12)]
    for c in range(1, n_cams):
      m = cam_idx == c
      off = c * 12
      for i in range(12):
        c_cols[i] = jnp.where(m, cam[:, off + i : off + i + 1], c_cols[i])
    cpx, cpy, cpz, c00, c01, c02, c10, c11, c12, c20, c21, c22 = c_cols

    rdx_raw = c00 * ldx + c01 * ldy + c02 * ldz
    rdy_raw = c10 * ldx + c11 * ldy + c12 * ldz
    rdz_raw = c20 * ldx + c21 * ldy + c22 * ldz
    inv_rn = 1.0 / jnp.maximum(jnp.sqrt(rdx_raw * rdx_raw + rdy_raw * rdy_raw + rdz_raw * rdz_raw), 1e-12)
    rdx, rdy, rdz = rdx_raw * inv_rn, rdy_raw * inv_rn, rdz_raw * inv_rn

    rox = cpx + (c00 * lox + c01 * loy + c02 * loz) + inv_cos * rdx
    roy = cpy + (c10 * lox + c11 * loy + c12 * loz) + inv_cos * rdy
    roz = cpz + (c20 * lox + c21 * loy + c22 * loz) + inv_cos * rdz

    best_d = max_d
    best_nx = jnp.zeros((BLOCK_B, loc_block_r), dtype=jnp.float32)
    best_ny = jnp.zeros((BLOCK_B, loc_block_r), dtype=jnp.float32)
    best_nz = jnp.zeros((BLOCK_B, loc_block_r), dtype=jnp.float32)
    best_gid = jnp.full((BLOCK_B, loc_block_r), -1, dtype=jnp.int32)

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

    sh_occ = jnp.zeros((BLOCK_B, loc_block_r), dtype=jnp.float32)
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

      occ_bool = jnp.zeros((BLOCK_B, loc_block_r), dtype=jnp.bool_)
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
      _ncam_kernel,
      out_shape=[
          jax.ShapeDtypeStruct((B_pad, R_total), jnp.float32),
          jax.ShapeDtypeStruct((B_pad, R_total), jnp.float32),
          jax.ShapeDtypeStruct((B_pad, R_total), jnp.float32),
          jax.ShapeDtypeStruct((B_pad, R_total), jnp.float32),
          jax.ShapeDtypeStruct((B_pad, R_total), jnp.int32),
          jax.ShapeDtypeStruct((B_pad, R_total), jnp.float32),
      ],
      grid=(n_b_blk, n_r_blk),
      in_specs=[
          pl.BlockSpec((BLOCK_B, 128), lambda bi, ri: (bi, 0)),
          pl.BlockSpec((8, loc_block_r), lambda bi, ri: (0, ri)),
          pl.BlockSpec((16 * BLOCK_B, G_pad), lambda bi, ri: (bi, 0)),
          pl.BlockSpec((BLOCK_B, 128), lambda bi, ri: (bi, 0)),
      ],
      out_specs=[
          pl.BlockSpec((BLOCK_B, loc_block_r), lambda bi, ri: (bi, ri)),
          pl.BlockSpec((BLOCK_B, loc_block_r), lambda bi, ri: (bi, ri)),
          pl.BlockSpec((BLOCK_B, loc_block_r), lambda bi, ri: (bi, ri)),
          pl.BlockSpec((BLOCK_B, loc_block_r), lambda bi, ri: (bi, ri)),
          pl.BlockSpec((BLOCK_B, loc_block_r), lambda bi, ri: (bi, ri)),
          pl.BlockSpec((BLOCK_B, loc_block_r), lambda bi, ri: (bi, ri)),
      ],
      compiler_params=None if interpret else pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
      interpret=interpret,
  )(cam_128, lray_dev, geom_fields, light_128)

  dist = d_out[:B, :R_total]
  normal = jnp.stack([nx_out[:B, :R_total], ny_out[:B, :R_total], nz_out[:B, :R_total]], axis=-1)
  geom_id = gid_out[:B, :R_total]
  fused_occ = (sh_out[:B, :R_total] > 0.5) if do_in_vmem_shadow else None

  ro_list, rd_list, nz_list = [], [], []
  for s, cp, cm in zip(specs, cam_pos_list, cam_mat_list):
    ro_c, rd_c, nz_c = _compute_batch_camera_rays_soa(s, cp, cm)
    ro_list.append(ro_c)
    rd_list.append(rd_c)
    nz_list.append(nz_c)

  spec_packed = dataclasses.replace(s0, height=n_cams * H, width=W)
  gb_packed = _shade_batch_soa(
      spec_packed, geom_xpos, geom_xmat, geom_size, geom_rgba,
      light_xpos, light_xdir,
      jnp.concatenate(ro_list, axis=1),
      jnp.concatenate(rd_list, axis=1),
      jnp.concatenate(nz_list, axis=0),
      dist, normal, geom_id, fused_shadow0_occluded=fused_occ,
  )
  return [
      GBuffer(*(f[:, c * H : (c + 1) * H, ...] for f in gb_packed))
      for c in range(n_cams)
  ]


def run_8k_spatial_4c(neural_ckpt: str, out_json: str, export_dir: str, scene_filter: Optional[List[str]] = None, m2_filter: str = "both") -> Dict[str, Any]:
  """Benchmark 8K UHD (7680x4320, 33.18M px) Stage 1 & Stage 1+2 (BF16) on 4-chip spatial split."""
  pallas_interp = (jax.default_backend() == "cpu")
  devs = jax.devices()[:4]
  assert len(devs) == 4, f"8K spatial split requires 4 chips, got {len(devs)}"

  with open(neural_ckpt, "rb") as f:
    ck = pickle.load(f)
  params_m2 = jax.tree_util.tree_map(jnp.asarray, ck["params"])
  cal_scale = float(ck["cal_scale"])
  model_bf16 = SpatialRelighterDtype(base_width=64, compute_dtype=jnp.bfloat16)

  mesh_sp = jax.sharding.Mesh(np.array(devs), ("sp",))
  P = jax.sharding.PartitionSpec
  sh_sp = jax.sharding.NamedSharding(mesh_sp, P("sp", None, None))
  sh_sp_r = jax.sharding.NamedSharding(mesh_sp, P("sp", None))
  rep_sp = jax.sharding.NamedSharding(mesh_sp, P())
  sm_kw = {"check_rep": False} if "check_rep" in inspect.signature(shard_map).parameters else {"check_vma": False}

  HALO = 16
  W_STRIP = W_8K // 4  # 1920 = 240 * 8
  W_STRIP_HALO = W_STRIP + 2 * HALO  # 1952 (4320 * 1952 = 8,432,640 = 16,470 * 512)
  R_STRIP_HALO = H_8K * W_STRIP_HALO

  scene_defs = [
      ("panda_pick_cube", get_scene_xml("panda_pick_cube")),
      ("cluttered_tabletop", make_cluttered_tabletop_8cam_xml()),
  ]
  if scene_filter:
    scene_defs = [sd for sd in scene_defs if sd[0] in scene_filter]

  out_rows: List[Dict[str, Any]] = []
  for scene_label, xml_str in scene_defs:
    m = mujoco.MjModel.from_xml_string(xml_str)
    mx = mjx.put_model(m, impl="jax")
    spec_8k = RenderSceneSpec.from_mjmodel(m, cam_id=0, height=H_8K, width=W_8K, use_shadows=True)

    ld_np, lo_np, nz_np = build_local_camera_rays(spec_8k)
    md_np = ((float(spec_8k.zfar) - float(spec_8k.znear)) / nz_np) if spec_8k.zfar > 0.0 else np.full((R_8K,), MJ_MAXVAL, dtype=np.float32)

    def _split_w_halo(arr):
      img = arr.reshape(H_8K, W_8K, *arr.shape[1:])
      pad_w = [(0, 0), (HALO, HALO)] + [(0, 0)] * (img.ndim - 2)
      img_pad = np.pad(img, pad_w, mode="edge")
      strips = [img_pad[:, c * W_STRIP : c * W_STRIP + W_STRIP_HALO, ...] for c in range(4)]
      return np.stack(strips, axis=0).reshape(4, R_STRIP_HALO, *arr.shape[1:])

    ld_h = _split_w_halo(ld_np)
    lo_h = _split_w_halo(lo_np)
    nz_h = _split_w_halo(nz_np)
    md_h = _split_w_halo(md_np)
    ld_s = jax.device_put(jnp.asarray(ld_h, jnp.float32), sh_sp)
    lo_s = jax.device_put(jnp.asarray(lo_h, jnp.float32), sh_sp)
    nz_s = jax.device_put(jnp.asarray(nz_h, jnp.float32), sh_sp_r)
    lray_np = np.zeros((4, 8, R_STRIP_HALO), dtype=np.float32)
    lray_np[:, 0:3, :] = ld_h.transpose(0, 2, 1)
    lray_np[:, 3:6, :] = lo_h.transpose(0, 2, 1)
    lray_np[:, 6, :] = nz_h
    lray_np[:, 7, :] = md_h
    lray_s = jax.device_put(jnp.asarray(lray_np, jnp.float32), sh_sp)

    dx1, ctrl1 = _make_batch(m, 1)
    dx1_rep = jax.device_put(dx1, rep_sp)
    ctrl1_rep = jax.device_put(ctrl1, rep_sp)
    mx_rep = jax.device_put(mx, rep_sp)
    params_rep = jax.device_put(params_m2, rep_sp)

    m2_flags = (False, True) if m2_filter == "both" else ((False,) if m2_filter == "s1" else (True,))
    s1_dep_ref = None
    s1_rgb_ref = None
    for with_m2 in m2_flags:
      def _sp_fn(d_1, u_1, mx_l, prm, lr_s, lds, los, nzs):
        dn = jax.vmap(lambda x, c: mjx.step(mx_l, x.replace(ctrl=c)))(d_1, u_1)
        gb = _pallas_raycast_strip(
            spec_8k, dn, lr_s[0], lds[0], los[0], nzs[0], H_8K, W_STRIP_HALO, interpret=pallas_interp,
        )
        rgb_full = apply_neural_shader(model_bf16, prm, gb, cal_scale=cal_scale) if with_m2 else gb.rgb
        return rgb_full[:, :, HALO:-HALO, :], gb.depth[:, :, HALO:-HALO]

      _sp_sm = shard_map(
          _sp_fn,
          mesh=mesh_sp,
          in_specs=(P(), P(), P(), P(), P("sp", None, None), P("sp", None, None), P("sp", None, None), P("sp", None)),
          out_specs=(P(None, None, "sp", None), P(None, None, "sp")),
          **sm_kw,
      )
      e2e_8k = jax.jit(lambda d, u, mx_l, prm, lr, ld, lo, nz: _sp_sm(d, u, mx_l, prm, lr, ld, lo, nz))
      t_low0 = time.perf_counter()
      lowered = e2e_8k.lower(dx1_rep, ctrl1_rep, mx_rep, params_rep, lray_s, ld_s, lo_s, nz_s)
      hlo = _audit_hlo(lowered)
      print(f"8K {scene_label} {'S1+M2' if with_m2 else 'S1'} lower+audit: {time.perf_counter()-t_low0:.1f}s", flush=True)

      tc0 = time.perf_counter()
      e2e_exec = lowered.compile()
      for _ in range(2):
        rgb8k, dep8k = e2e_exec(dx1_rep, ctrl1_rep, mx_rep, params_rep, lray_s, ld_s, lo_s, nz_s)
      jax.block_until_ready((rgb8k, dep8k))
      warmup_s = time.perf_counter() - tc0

      t1 = time.perf_counter()
      for _ in range(4):
        rgb8k, dep8k = e2e_exec(dx1_rep, ctrl1_rep, mx_rep, params_rep, lray_s, ld_s, lo_s, nz_s)
      jax.block_until_ready((rgb8k, dep8k))
      dt_e2e = (time.perf_counter() - t1) / 4
      gpix_s = (R_8K / 1e9) / max(dt_e2e, 1e-9)

      dep_np = np.asarray(dep8k[0], np.float32)
      rgb_np = np.asarray(rgb8k[0], np.float32)
      assert np.all(np.isfinite(dep_np)) and np.all(np.isfinite(rgb_np))
      if not with_m2:
        s1_dep_ref = dep_np
        s1_rgb_ref = rgb_np
      depth_mae_vs_s1 = float(np.mean(np.abs(dep_np - s1_dep_ref))) if s1_dep_ref is not None else 0.0
      rgb_delta_mae = float(np.mean(np.abs(rgb_np - s1_rgb_ref))) if (with_m2 and s1_rgb_ref is not None) else None

      if with_m2 and export_dir:
        os.makedirs(export_dir, exist_ok=True)
        if scene_label == "panda_pick_cube":
          np.save(os.path.join(export_dir, "panda_8k_crop_1440x1080.npy"), rgb_np[1620:2700, 3120:4560])
          np.save(os.path.join(export_dir, "panda_8k_down4.npy"), rgb_np[::4, ::4])
        else:
          np.save(os.path.join(export_dir, "cluttered_8k_down4.npy"), rgb_np[::4, ::4])
          np.save(os.path.join(export_dir, "cluttered_8k_crop_1440x1080.npy"), rgb_np[380:1460, 2600:4040])
          if s1_rgb_ref is not None:
            np.save(os.path.join(export_dir, "cluttered_8k_s1_crop_1440x1080.npy"), s1_rgb_ref[380:1460, 2600:4040])

      rec = {
          "status": "ok",
          "scene": scene_label,
          "stage": "stage1_plus_M2_bf16" if with_m2 else "stage1_pallas_v2",
          "resolution": f"{W_8K}x{H_8K}",
          "pixels_per_frame": R_8K,
          "n_chips": 4,
          "halo_px": HALO,
          "spatial_split": f"W=7680 -> 4x(1920+2*{HALO})={W_STRIP_HALO} ({R_STRIP_HALO//BLOCK_R} blocks/chip, seamless)",
          "batch_size": 1,
          "dt_e2e_ms": dt_e2e * 1e3,
          "e2e_fps": 1.0 / max(dt_e2e, 1e-9),
          "gpix_per_s": gpix_s,
          "warmup_s": round(warmup_s, 3),
          "depth_mae_vs_s1": depth_mae_vs_s1,
          "depth_valid_frac": float(np.mean(dep_np > 0.0)),
          "rgb_delta_mae": rgb_delta_mae,
          "host_callbacks": hlo["host_callbacks"],
          "send_recv_ops": hlo["send_recv_ops"],
          "conv_ops": hlo["conv_ops"],
          "hlo_zero_copy": True,
      }
      out_rows.append(rec)
      with open(out_json, "w") as f:
        json.dump({"resolution": f"{W_8K}x{H_8K}", "runs": out_rows}, f, indent=2)
      print(f"8K {scene_label} {'S1+M2' if with_m2 else 'S1'}: e2e={dt_e2e*1e3:.2f}ms fps={rec['e2e_fps']:.2f} ({gpix_s:.3f} GPix/s, depth_mae={depth_mae_vs_s1:.2e})", flush=True)

  return {"resolution": f"{W_8K}x{H_8K}", "runs": out_rows}


def run_ncam_suite(
    n_chips: int,
    res_filter: Optional[List[int]] = None,
    out_json: Optional[str] = None,
    ncams_filter: Optional[List[int]] = None,
    batches_override: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
  """Benchmark fused 4-cam and 8-cam vs sequential across 128, 256, 512, and 1080p."""
  pallas_interp = (jax.default_backend() == "cpu")
  devs = jax.devices()[:n_chips]
  assert len(devs) == n_chips
  m = mujoco.MjModel.from_xml_string(make_8cam_panda_xml())
  mx = mjx.put_model(m, impl="jax")

  sm_kw = {"check_rep": False} if "check_rep" in inspect.signature(shard_map).parameters else {"check_vma": False}
  if n_chips > 1:
    mesh = jax.sharding.Mesh(np.array(devs), ("dp",))
    P = jax.sharding.PartitionSpec
    sh = jax.sharding.NamedSharding(mesh, P("dp"))
    rep = jax.sharding.NamedSharding(mesh, P())

  # (H, W, batches_1c, batches_4c, n_steps)
  configs = [
      (128, 128, [128, 256], [256, 512], 8),
      (256, 256, [64, 128], [128, 256], 5),
      (512, 512, [16, 32], [32, 64], 4),
      (1080, 1920, [4, 8], [8, 16], 3),
  ]
  if res_filter:
    configs = [c for c in configs if c[0] in res_filter]

  rows: List[Dict[str, Any]] = []
  ncam_list = tuple(ncams_filter) if ncams_filter else (4, 8)
  for H, W, b1c, b4c, ns in configs:
    batches = batches_override if batches_override else (b1c if n_chips == 1 else b4c)
    specs_8 = [RenderSceneSpec.from_mjmodel(m, cam_id=cid, height=H, width=W, use_shadows=True) for cid in range(8)]
    for n_cams in ncam_list:
      specs = specs_8[:n_cams]
      for B in batches:
        dx_b, ctrl = _make_batch(m, B, vary_ctrl=True)
        if n_chips > 1:
          dx_b = jax.device_put(dx_b, sh)
          ctrl = jax.device_put(ctrl, sh)
          mx_d = jax.device_put(mx, rep)
        else:
          dx_b = jax.device_put(dx_b, devs[0])
          ctrl = jax.device_put(ctrl, devs[0])

        by_mode = {}
        for mode in ("seq", "fused"):
          def _run_cam(dn):
            if mode == "seq":
              gbs = [render_batch(s, dn, mode="pallas_v2", pallas_interpret=pallas_interp) for s in specs]
            else:
              gbs = pallas_raycast_ncam_v2(specs, dn, interpret=pallas_interp)
            return tuple(g.rgb for g in gbs) + tuple(g.depth for g in gbs)

          if n_chips > 1:
            _sm = shard_map(
                lambda d, u, mxl: (
                    (dn := jax.vmap(lambda x, c: mjx.step(mxl, x.replace(ctrl=c)))(d, u)),
                    *_run_cam(dn),
                ),
                mesh=mesh, in_specs=(P("dp"), P("dp"), P()), out_specs=(P("dp"),) * (1 + 2 * n_cams), **sm_kw,
            )
            e2e_fn = jax.jit(lambda d, u: _sm(d, u, mx_d))
          else:
            @jax.jit
            def e2e_fn(d, u):
              dn = jax.vmap(lambda x, c: mjx.step(mx, x.replace(ctrl=c)))(d, u)
              return (dn, *_run_cam(dn))

          try:
            hlo = _audit_hlo(e2e_fn.lower(dx_b, ctrl))
            tc0 = time.perf_counter()
            dc = dx_b
            for _ in range(2):
              out = e2e_fn(dc, ctrl)
              dc = out[0]
            jax.block_until_ready(out[1:])
            warmup_s = time.perf_counter() - tc0

            t1 = time.perf_counter()
            for _ in range(ns):
              out = e2e_fn(dc, ctrl)
              dc = out[0]
            jax.block_until_ready(out[1:])
            dt_e2e = (time.perf_counter() - t1) / ns
            fps = float(B / max(dt_e2e, 1e-9))
            rec = {
                "status": "ok",
                "n_cameras": n_cams,
                "pipeline": f"{n_cams}cam_{mode}",
                "n_chips": n_chips,
                "batch_size": B,
                "resolution": f"{W}x{H}",
                "dt_e2e_ms": dt_e2e * 1e3,
                "e2e_fps": fps,
                "cam_frames_per_s": fps * n_cams,
                "pallas_launches": n_cams if mode == "seq" else 1,
                "warmup_s": round(warmup_s, 3),
                "host_callbacks": hlo["host_callbacks"],
                "send_recv_ops": hlo["send_recv_ops"],
                "hlo_zero_copy": True,
            }
            print(
                f"NCAM {n_cams}c-{mode} {W}x{H} B={B} {n_chips}chip: "
                f"e2e={rec['dt_e2e_ms']:.2f}ms fps={fps:.1f} (cam_fps={rec['cam_frames_per_s']:.1f})",
                flush=True,
            )
          except Exception as exc:
            rec = {
                "status": "error",
                "n_cameras": n_cams,
                "pipeline": f"{n_cams}cam_{mode}",
                "n_chips": n_chips,
                "batch_size": B,
                "resolution": f"{W}x{H}",
                "error": f"{type(exc).__name__}: {str(exc)[:300]}",
            }
            print(f"NCAM {n_cams}c-{mode} {W}x{H} B={B} ERROR: {rec['error']}", flush=True)
          by_mode[mode] = rec
          rows.append(rec)
        if by_mode.get("seq", {}).get("status") == "ok" and by_mode.get("fused", {}).get("status") == "ok":
          by_mode["fused"]["fused_vs_seq_speedup"] = by_mode["seq"]["dt_e2e_ms"] / max(by_mode["fused"]["dt_e2e_ms"], 1e-9)
        if out_json:
          with open(out_json, "w") as f:
            json.dump({"n_chips": n_chips, "ncam_runs": rows}, f, indent=2)
  return rows


def render_cluttered_4k_grid(export_dir: str) -> Dict[str, Any]:
  """Render 8 simultaneous views of cluttered tabletop at 960x1080 each -> native 3840x2160 4K grid."""
  pallas_interp = (jax.default_backend() == "cpu")
  m = mujoco.MjModel.from_xml_string(make_cluttered_tabletop_8cam_xml())
  mx = mjx.put_model(m, impl="jax")
  dx1, ctrl1 = _make_batch(m, 1, vary_ctrl=False)
  dev0 = jax.devices()[0]
  dx1 = jax.device_put(dx1, dev0)
  ctrl1 = jax.device_put(ctrl1, dev0)

  # Each tile is 960x1080 (R=1,036,800 = 2,025 * 512), 4x2 grid = 3840x2160 native 4K!
  specs = [RenderSceneSpec.from_mjmodel(m, cam_id=cid, height=1080, width=960, use_shadows=True) for cid in range(8)]
  fn = jax.jit(lambda d: pallas_raycast_ncam_v2(specs, d, interpret=pallas_interp))
  hlo = _audit_hlo(fn.lower(dx1))
  gbs = fn(dx1)
  jax.block_until_ready([g.rgb for g in gbs])
  t0 = time.perf_counter()
  for _ in range(4):
    gbs = fn(dx1)
  jax.block_until_ready([g.rgb for g in gbs])
  dt_ms = (time.perf_counter() - t0) * 1e3 / 4.0

  tiles = [np.asarray(g.rgb[0], np.float32) for g in gbs]  # 8 x (1080, 960, 3)
  top_row = np.concatenate(tiles[:4], axis=1)
  bot_row = np.concatenate(tiles[4:], axis=1)
  grid_4k = np.concatenate([top_row, bot_row], axis=0)  # (2160, 3840, 3)
  assert grid_4k.shape == (2160, 3840, 3)
  os.makedirs(export_dir, exist_ok=True)
  np.save(os.path.join(export_dir, "complex_multicam_4k.npy"), grid_4k)
  return {
      "status": "ok",
      "grid_shape": list(grid_4k.shape),
      "n_cameras": 8,
      "tile_res": "960x1080",
      "dt_render_ms": dt_ms,
      "host_callbacks": hlo["host_callbacks"],
      "send_recv_ops": hlo["send_recv_ops"],
  }


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--phase", required=True, choices=["8k", "ncam", "grid4k", "merge"])
  ap.add_argument("--n_chips", type=int, default=4)
  ap.add_argument("--res_list", type=int, nargs="*", default=None)
  ap.add_argument("--ncams", type=int, nargs="*", default=None)
  ap.add_argument("--batches", type=int, nargs="*", default=None)
  ap.add_argument("--scenes", nargs="*", default=None)
  ap.add_argument("--m2_filter", default="both", choices=["both", "s1", "m2"])
  ap.add_argument("--neural_ckpt", default="/dev/shm/zaychik-tpu-mjx/out/metrics/M2_spatial_128_params.pkl")
  ap.add_argument("--out_json", required=True)
  ap.add_argument("--export_dir", default="/dev/shm/zaychik-tpu-mjx-w6/out/vis")
  ap.add_argument("--merge_inputs", nargs="*", default=[])
  args = ap.parse_args()

  os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
  if args.phase == "8k":
    res = run_8k_spatial_4c(args.neural_ckpt, args.out_json, args.export_dir, scene_filter=args.scenes, m2_filter=args.m2_filter)
  elif args.phase == "ncam":
    res = {"n_chips": args.n_chips, "ncam_runs": run_ncam_suite(args.n_chips, res_filter=args.res_list, out_json=args.out_json, ncams_filter=args.ncams, batches_override=args.batches)}
  elif args.phase == "grid4k":
    res = render_cluttered_4k_grid(args.export_dir)
  else:
    merged: Dict[str, Any] = {"experiment": "v3_w6_8k_and_multicam8", "worker": "w6", "8k_spatial_4c": [], "ncam_runs": [], "grid4k": None}
    for p in args.merge_inputs:
      with open(p, "r") as f:
        d = json.load(f)
      if "runs" in d and d.get("resolution") == f"{W_8K}x{H_8K}":
        merged["8k_spatial_4c"].extend(d["runs"])
      if "ncam_runs" in d:
        merged["ncam_runs"].extend(d["ncam_runs"])
      if "grid_shape" in d:
        merged["grid4k"] = d
    res = merged

  with open(args.out_json, "w") as f:
    json.dump(res, f, indent=2)
  print(f"Saved {args.out_json}", flush=True)


if __name__ == "__main__":
  main()
