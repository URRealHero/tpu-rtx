"""Wave 5 Task A2: True Stochastic Thin-Lens Camera Aperture (f/1.4 Creamy Bokeh) at Native 4K.

Implements:
1. Native TPU Mosaic (BLOCK_B=8, BLOCK_R=512) Pallas Thin-Lens DOF Kernel with:
   - Shirley-Chiu concentric disk aperture sampling
   - Per-pixel Cranley-Patterson rotated 2D R2 low-discrepancy sequence (zero grain/banding)
   - Linear-HDR specular + emissive bokeh accumulation across 16-32 spp -> ACES filmic tonemap
   - Exact zero-aperture pinhole parity check vs `pallas_v2`
   - Zero-copy HLO audit (host_callbacks == 0, send_recv_ops == 0, gather_ops == 0)
2. 4-Chip Spatial-Sharded Native 4K (3840x2160) Rendering & Benchmark (16 spp & 32 spp).
3. Side-by-side Native 4K comparison figure:
   f/16 Deep-Focus vs f/1.4 Focus-Near vs f/1.4 Focus-Far -> v3_w6_thinlens_bokeh_4k.png.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tpu_mjx_render.pallas_v2 import (
    BLOCK_B,
    BLOCK_R,
    _compute_geom_bound_sq,
    _extract_light_batch,
    _ray_one_geom_v2,
    pallas_raycast_batch_v2,
)
from tpu_mjx_render.primitives import MJ_MAXVAL
from tpu_mjx_render.renderer import RenderSceneSpec
from tpu_mjx_render.roofline import inspect_hlo_ops
from tpu_mjx_render.scenes import PANDA_PICK_CUBE_XML

W_4K = 3840
H_4K = 2160
R_4K = W_4K * H_4K  # 8,294,400


def _audit_hlo(lowered: Any) -> Dict[str, Any]:
  txt = lowered.as_text()
  info = inspect_hlo_ops(txt)
  info["send_recv_ops"] = len(re.findall(r"\bsend\(|\brecv\(", txt, re.I))
  assert info["host_callbacks"] == 0, f"HLO contains host_callback: {info}"
  assert info["send_recv_ops"] == 0, f"HLO contains send/recv: {info}"
  assert info["gather_ops"] == 0, f"HLO contains gather_ops: {info}"
  info["zero_copy_verified"] = True
  return info


def make_bokeh_hero_xml() -> str:
  """Multi-depth hero robotic tabletop + distant luminous orbs for f/1.4 DOF showcase."""
  xml = PANDA_PICK_CUBE_XML
  # Replace default front/side cameras with 50mm-equivalent (fovy=27 deg) hero framing
  old_cams = (
      '    <camera fovy="42" name="front" pos="1.176 0.001 0.483" xyaxes="0.006 1.000 -0.000 -0.720 0.004 0.694"/>\n'
      '    <camera fovy="45" name="side" pos="0.60 -0.85 0.55" xyaxes="1 0 0 0 0.45 0.89"/>\n'
  )
  # Camera at (1.46, -0.05, 0.36) looking at (0.45, 0.0, 0.19) with fovy=34 (40mm prime lens):
  # Foreground props & gripper ~ 0.72-0.85m, Panda torso ~ 1.30m, background orbs/pillars ~ 2.05-2.30m.
  new_cam = (
      '    <camera fovy="34" name="bokeh_hero" pos="1.46 -0.05 0.36" '
      'xyaxes="0.0494 0.9988 0.0 -0.1658 0.0082 0.9861"/>\n'
  )
  assert old_cams in xml
  xml = xml.replace(old_cams, new_cam)

  # Inject glossy/emissive materials into <asset> for crisp HDR bokeh disks
  asset_anchor = '    <material name="strip_white" rgba="0.92 0.92 0.90 1"/>\n'
  bokeh_mats = (
      '    <material name="mat_wood" rgba="0.38 0.28 0.20 1" specular="0.3" shininess="0.4"/>\n'
      '    <material name="mat_gold" rgba="0.96 0.74 0.16 1" specular="1.0" shininess="0.95"/>\n'
      '    <material name="mat_cyan" rgba="0.10 0.78 0.86 1" specular="0.9" shininess="0.90"/>\n'
      '    <material name="mat_pillar" rgba="0.26 0.30 0.36 1" specular="0.5" shininess="0.6"/>\n'
      '    <material name="mat_mid" rgba="0.22 0.52 0.90 1" specular="0.7" shininess="0.8"/>\n'
      '    <material name="mat_orb_amber" rgba="1.0 0.56 0.14 1" emission="4.2" specular="1.0" shininess="0.95"/>\n'
      '    <material name="mat_orb_cyan" rgba="0.16 0.86 1.0 1" emission="4.2" specular="1.0" shininess="0.95"/>\n'
      '    <material name="mat_orb_magenta" rgba="1.0 0.26 0.64 1" emission="3.8" specular="1.0" shininess="0.95"/>\n'
      '    <material name="mat_orb_gold" rgba="1.0 0.86 0.22 1" emission="3.6" specular="1.0" shininess="0.95"/>\n'
      '    <material name="mat_orb_emerald" rgba="0.20 0.96 0.48 1" emission="3.6" specular="1.0" shininess="0.95"/>\n'
      '    <material name="mat_fg_amber" rgba="1.0 0.62 0.10 1" emission="1.4" specular="1.0" shininess="0.95"/>\n'
  )
  assert asset_anchor in xml
  xml = xml.replace(asset_anchor, asset_anchor + bokeh_mats)

  # Keep total geoms <= 36 for fast in-VMEM shadow + BLOCK_R=512
  extra_geoms = """
    <!-- Foreground hero zone (~0.66 - 0.80 m from camera) -->
    <geom name="fg_workbench" type="box" pos="0.62 0.0 0.012" size="0.46 0.56 0.012" material="mat_wood" contype="0" conaffinity="0"/>
    <geom name="fg_sphere_gold" type="sphere" pos="0.72 -0.16 0.088" size="0.062" material="mat_gold" contype="0" conaffinity="0"/>
    <geom name="fg_cyl_cyan" type="cylinder" pos="0.75 0.15 0.085" size="0.042 0.060" material="mat_cyan" contype="0" conaffinity="0"/>
    <geom name="fg_bead_amber" type="sphere" pos="0.88 0.20 0.185" size="0.032" material="mat_fg_amber" contype="0" conaffinity="0"/>

    <!-- Midground accent (~1.25 m) -->
    <geom name="mid_pillar_l" type="cylinder" pos="0.15 0.36 0.22" size="0.048 0.22" material="mat_mid" contype="0" conaffinity="0"/>

    <!-- Background architectural + luminous specular orbs (~2.05 - 2.35 m from camera) -->
    <geom name="bg_pillar_l" type="box" pos="-0.68 0.36 0.28" size="0.075 0.075 0.28" material="mat_pillar" contype="0" conaffinity="0"/>
    <geom name="bg_pillar_r" type="box" pos="-0.68 -0.32 0.28" size="0.075 0.075 0.28" material="mat_pillar" contype="0" conaffinity="0"/>
    <geom name="bg_orb_warm1" type="sphere" pos="-0.65 -0.28 0.34" size="0.042" material="mat_orb_amber" contype="0" conaffinity="0"/>
    <geom name="bg_orb_cyan1" type="sphere" pos="-0.72 0.26 0.38" size="0.042" material="mat_orb_cyan" contype="0" conaffinity="0"/>
    <geom name="bg_orb_magenta" type="sphere" pos="-0.82 -0.14 0.44" size="0.036" material="mat_orb_magenta" contype="0" conaffinity="0"/>
    <geom name="bg_orb_gold" type="sphere" pos="-0.58 -0.44 0.28" size="0.034" material="mat_orb_gold" contype="0" conaffinity="0"/>
    <geom name="bg_orb_emerald" type="sphere" pos="-0.62 -0.38 0.20" size="0.032" material="mat_orb_emerald" contype="0" conaffinity="0"/>
    <geom name="bg_pedestal" type="box" pos="-0.72 0.0 0.08" size="0.12 0.45 0.08" material="mat_pillar" contype="0" conaffinity="0"/>
  """
  anchor = '    <body mocap="true" name="mocap_target" pos="0.65 0 0.25">'
  assert anchor in xml
  xml = xml.replace(anchor, extra_geoms + "\n" + anchor)
  xml = xml.replace('<body name="box" pos="0.65 0 0.03">', '<body name="box" pos="0.66 -0.03 0.056">')
  # Boost light intensity slightly for crisp 4K speculars
  xml = xml.replace('diffuse="0.8 0.8 0.8"', 'diffuse="0.92 0.90 0.86"')
  return xml


def _build_thinlens_strip_arrays(
    fovy_deg: float,
    full_H: int,
    full_W: int,
    x_start: int,
    strip_W: int,
    spp: int,
) -> Tuple[np.ndarray, np.ndarray]:
  """Build (8, full_H * strip_W) pixel ray+hash array and (spp, 2) R2 quasi-random samples."""
  px = np.arange(x_start, x_start + strip_W, dtype=np.float32)
  py = np.arange(0, full_H, dtype=np.float32)
  gx, gy = np.meshgrid(px, py, indexing="xy")  # (H, strip_W)
  u = (gx + 0.5) / float(full_W)
  v = (gy + 0.5) / float(full_H)
  aspect = float(full_W) / float(full_H)
  tan_half = math.tan(0.5 * float(fovy_deg) * math.pi / 180.0)
  ux = aspect * tan_half * (2.0 * u - 1.0)
  uy = tan_half * (1.0 - 2.0 * v)

  # Deterministic 2D Cranley-Patterson pixel hash in [0, 1) using full Murmur3 mix
  ix = gx.astype(np.uint32)
  iy = gy.astype(np.uint32)
  h = (ix * np.uint32(0x9E3779B1)) ^ (iy * np.uint32(0x85EBCA77)) ^ np.uint32(0xC2B2AE3D)
  h = ((h ^ (h >> np.uint32(16))) * np.uint32(0x85EBCA6B))
  h = ((h ^ (h >> np.uint32(13))) * np.uint32(0xC2B2AE35))
  h = h ^ (h >> np.uint32(16))
  h1 = (h & np.uint32(0xFFFF)).astype(np.float32) / 65536.0
  h2 = ((h >> np.uint32(16)) & np.uint32(0xFFFF)).astype(np.float32) / 65536.0

  R_strip = full_H * strip_W
  lray = np.zeros((8, R_strip), dtype=np.float32)
  lray[0, :] = ux.reshape(-1)
  lray[1, :] = uy.reshape(-1)
  lray[2, :] = h1.reshape(-1)
  lray[3, :] = h2.reshape(-1)

  # 2D R2 low-discrepancy sequence (plastic constant alpha1, alpha2)
  g = 1.324717957244746
  a1 = 1.0 / g
  a2 = 1.0 / (g * g)
  idx = np.arange(spp, dtype=np.float64) + 0.5
  uv_base = np.stack([(idx * a1) % 1.0, (idx * a2) % 1.0], axis=-1).astype(np.float32)
  return lray, uv_base


def _pallas_thinlens_raycast_8samples(
    spec: RenderSceneSpec,
    cam_pos: jax.Array,       # (3,)
    cam_mat: jax.Array,       # (3, 3)
    geom_xpos: jax.Array,     # (ngeom, 3)
    geom_xmat: jax.Array,     # (ngeom, 3, 3)
    light_xpos: jax.Array,    # (nlight, 3)
    light_xdir: jax.Array,    # (nlight, 3)
    lray_strip: jax.Array,    # (8, R_strip)
    uv_8: jax.Array,          # (8, 2)
    lens_radius: float,
    focus_dist: float,
    interpret: bool = False,
):
  """Trace 8 thin-lens concentric-disk samples simultaneously in Mosaic VMEM (BLOCK_B=8, BLOCK_R=512)."""
  R_strip = lray_strip.shape[1]
  gids = np.asarray(spec.enabled_geom_ids, dtype=np.int32)
  G = len(gids)
  loc_block_r = 256 if G > 36 else BLOCK_R
  assert R_strip % loc_block_r == 0, (R_strip, loc_block_r)

  B_pad = BLOCK_B  # 8 samples
  G_pad = ((G + 127) // 128) * 128
  n_r_blk = R_strip // loc_block_r

  cam_row = jnp.concatenate([cam_pos, cam_mat.reshape(9)], axis=0)  # (12,)
  cam_b = jnp.broadcast_to(cam_row[None, :], (B_pad, 12))
  cam_128 = jnp.pad(jnp.concatenate([cam_b, uv_8], axis=-1), ((0, 0), (0, 128 - 14)))  # (8, 128)

  geom_size = jnp.asarray(spec.geom_size, dtype=jnp.float32)
  gpos = jnp.broadcast_to(geom_xpos[gids][None, :, :], (B_pad, G, 3))
  gmat = jnp.broadcast_to(geom_xmat[gids].reshape(G, 9)[None, :, :], (B_pad, G, 9))
  gsz_all = jnp.broadcast_to(geom_size[gids][None, :, :], (B_pad, G, 3))
  rb_sq = jnp.broadcast_to(jnp.asarray(_compute_geom_bound_sq(spec, gids))[None, :, None], (B_pad, G, 1))
  g_16 = jnp.concatenate([gpos, gmat, gsz_all, rb_sq], axis=-1)
  g_16 = jnp.pad(g_16, ((0, 0), (0, G_pad - G), (0, 0)))
  geom_fields = g_16.transpose(2, 0, 1).reshape(16 * BLOCK_B, G_pad)

  shadow_l_idx = -1
  for li in range(int(spec.nlight)):
    if spec.use_shadows and bool(spec.light_active[li]) and bool(spec.light_castshadow[li]):
      shadow_l_idx = li
      break

  light_128 = jnp.zeros((B_pad, 128), dtype=jnp.float32)
  if shadow_l_idx >= 0:
    is_dir = 1.0 if int(spec.light_type[shadow_l_idx]) == 1 else 0.0
    lp = light_xpos[shadow_l_idx]
    ld_v = light_xdir[shadow_l_idx]
    ld_v = ld_v / jnp.maximum(jnp.linalg.norm(ld_v), 1e-12)
    l_row = jnp.broadcast_to(jnp.concatenate([jnp.array([is_dir], jnp.float32), lp, ld_v])[None, :], (B_pad, 7))
    light_128 = jnp.pad(l_row, ((0, 0), (0, 128 - 7)))

  g_types_static = tuple(int(spec.geom_type[g]) for g in gids)
  gids_static = tuple(int(g) for g in gids)
  plane_fx = tuple(float(spec.geom_size[g, 0]) > 0.0 for g in gids)
  plane_fy = tuple(float(spec.geom_size[g, 1]) > 0.0 for g in gids)
  znear_f = float(spec.znear)
  zfar_f = float(spec.zfar)
  lens_r_f = float(lens_radius)
  focus_d_f = float(focus_dist)
  do_in_vmem_shadow = (shadow_l_idx >= 0) and (G <= 56)

  def _thinlens_kernel(
      cam_ref, lray_ref, gf_ref, light_ref,
      out_d_ref, out_nx_ref, out_ny_ref, out_nz_ref, out_gid_ref, out_sh_ref,
      out_rox_ref, out_roy_ref, out_roz_ref, out_rdx_ref, out_rdy_ref, out_rdz_ref,
  ):
    cam = cam_ref[...]
    lr = lray_ref[...]
    gf = gf_ref[...]
    linfo = light_ref[...]

    ux = lr[0:1, :]
    uy = lr[1:2, :]
    h1 = lr[2:3, :]
    h2 = lr[3:4, :]

    cpx, cpy, cpz = cam[:, 0:1], cam[:, 1:2], cam[:, 2:3]
    c00, c01, c02 = cam[:, 3:4], cam[:, 4:5], cam[:, 5:6]
    c10, c11, c12 = cam[:, 6:7], cam[:, 7:8], cam[:, 8:9]
    c20, c21, c22 = cam[:, 9:10], cam[:, 10:11], cam[:, 11:12]
    u1_s, u2_s = cam[:, 12:13], cam[:, 13:14]

    # Cranley-Patterson rotation + Shirley-Chiu concentric disk mapping
    if lens_r_f > 0.0:
      u1 = (u1_s + h1) - jnp.floor(u1_s + h1)
      u2 = (u2_s + h2) - jnp.floor(u2_s + h2)
      a = 2.0 * u1 - 1.0
      b = 2.0 * u2 - 1.0
      use_a = jnp.abs(a) > jnp.abs(b)
      safe_a = jnp.where(jnp.abs(a) > 1e-12, a, 1.0)
      safe_b = jnp.where(jnp.abs(b) > 1e-12, b, 1.0)
      r_disk = jnp.where(use_a, a, b)
      phi_disk = jnp.where(
          use_a,
          (math.pi / 4.0) * (b / safe_a),
          (math.pi / 2.0) - (math.pi / 4.0) * (a / safe_b),
      )
      zero_ab = (jnp.abs(a) <= 1e-12) & (jnp.abs(b) <= 1e-12)
      r_disk = jnp.where(zero_ab, 0.0, r_disk)
      lx = lens_r_f * r_disk * jnp.cos(phi_disk)
      ly = lens_r_f * r_disk * jnp.sin(phi_disk)
    else:
      lx = jnp.zeros((BLOCK_B, loc_block_r), dtype=jnp.float32)
      ly = jnp.zeros((BLOCK_B, loc_block_r), dtype=jnp.float32)

    # Thin-lens ray in camera space -> focal plane z = -focus_d_f
    dcx_raw = focus_d_f * ux - lx
    dcy_raw = focus_d_f * uy - ly
    dcz_raw = jnp.full((BLOCK_B, loc_block_r), -focus_d_f, dtype=jnp.float32)
    inv_dc = 1.0 / jnp.maximum(jnp.sqrt(dcx_raw * dcx_raw + dcy_raw * dcy_raw + dcz_raw * dcz_raw), 1e-12)
    dcx, dcy, dcz = dcx_raw * inv_dc, dcy_raw * inv_dc, dcz_raw * inv_dc

    rdx_raw = c00 * dcx + c01 * dcy + c02 * dcz
    rdy_raw = c10 * dcx + c11 * dcy + c12 * dcz
    rdz_raw = c20 * dcx + c21 * dcy + c22 * dcz
    inv_rn = 1.0 / jnp.maximum(jnp.sqrt(rdx_raw * rdx_raw + rdy_raw * rdy_raw + rdz_raw * rdz_raw), 1e-12)
    rdx, rdy, rdz = rdx_raw * inv_rn, rdy_raw * inv_rn, rdz_raw * inv_rn

    neg_z = jnp.maximum(-dcz, 1e-6)
    inv_cos = znear_f / neg_z
    rox = cpx + (c00 * lx + c01 * ly) + inv_cos * rdx
    roy = cpy + (c10 * lx + c11 * ly) + inv_cos * rdy
    roz = cpz + (c20 * lx + c21 * ly) + inv_cos * rdz

    if zfar_f > 0.0:
      max_d = (zfar_f - znear_f) / neg_z
    else:
      max_d = jnp.full((BLOCK_B, loc_block_r), MJ_MAXVAL, dtype=jnp.float32)

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
    out_rox_ref[...] = rox
    out_roy_ref[...] = roy
    out_roz_ref[...] = roz
    out_rdx_ref[...] = rdx
    out_rdy_ref[...] = rdy
    out_rdz_ref[...] = rdz

  outs = pl.pallas_call(
      _thinlens_kernel,
      out_shape=[jax.ShapeDtypeStruct((B_pad, R_strip), jnp.float32) for _ in range(4)]
      + [jax.ShapeDtypeStruct((B_pad, R_strip), jnp.int32)]
      + [jax.ShapeDtypeStruct((B_pad, R_strip), jnp.float32) for _ in range(7)],
      grid=(1, n_r_blk),
      in_specs=[
          pl.BlockSpec((BLOCK_B, 128), lambda bi, ri: (0, 0)),
          pl.BlockSpec((8, loc_block_r), lambda bi, ri: (0, ri)),
          pl.BlockSpec((16 * BLOCK_B, G_pad), lambda bi, ri: (0, 0)),
          pl.BlockSpec((BLOCK_B, 128), lambda bi, ri: (0, 0)),
      ],
      out_specs=[pl.BlockSpec((BLOCK_B, loc_block_r), lambda bi, ri: (0, ri)) for _ in range(12)],
      compiler_params=None if interpret else pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
      interpret=interpret,
  )(cam_128, lray_strip, geom_fields, light_128)

  dist, nx, ny, nz, geom_id, sh_out, rox, roy, roz, rdx, rdy, rdz = outs
  normal = jnp.stack([nx, ny, nz], axis=-1)
  rorig = jnp.stack([rox, roy, roz], axis=-1)
  rdir = jnp.stack([rdx, rdy, rdz], axis=-1)
  fused_occ = (sh_out > 0.5) if do_in_vmem_shadow else None
  return dist, normal, geom_id, fused_occ, rorig, rdir


def _shade_thinlens_hdr(
    spec: RenderSceneSpec,
    geom_xpos: jax.Array,      # (ngeom, 3)
    geom_xmat: jax.Array,      # (ngeom, 3, 3)
    light_xpos: jax.Array,     # (nlight, 3)
    light_xdir: jax.Array,     # (nlight, 3)
    rorig: jax.Array,          # (B, R, 3)
    rdir: jax.Array,           # (B, R, 3)
    dist: jax.Array,           # (B, R)
    normal: jax.Array,         # (B, R, 3)
    geom_id: jax.Array,        # (B, R)
    fused_occ: Optional[jax.Array],
    hdr_mode: bool = True,
) -> jax.Array:
  """Shade (B=8, R) rays in linear HDR with zero gather ops (pure static geom comparison)."""
  B, R, _ = rorig.shape
  hit = geom_id >= 0
  hit_pos = rorig + jnp.where(hit, dist, 0.0)[..., None] * rdir
  view_dir = -rdir

  # Zero-gather static material attribute lookup over enabled_geom_ids
  albedo = jnp.zeros((B, R, 3), dtype=jnp.float32)
  spec_coef = jnp.zeros((B, R), dtype=jnp.float32)
  shin_exp = jnp.zeros((B, R), dtype=jnp.float32)
  emit_coef = jnp.zeros((B, R), dtype=jnp.float32)
  for gid_val in spec.enabled_geom_ids:
    m = geom_id == int(gid_val)
    albedo = jnp.where(m[..., None], jnp.asarray(spec.geom_rgba[gid_val, :3], jnp.float32)[None, None, :], albedo)
    spec_coef = jnp.where(m, jnp.float32(spec.geom_spec[gid_val]), spec_coef)
    shin_exp = jnp.where(m, jnp.float32(spec.geom_shin[gid_val] * 128.0), shin_exp)
    emit_coef = jnp.where(m, jnp.float32(spec.geom_emit[gid_val]), emit_coef)

  chk_gids = [i for i, c in enumerate(spec.geom_is_checker) if bool(c)]
  if chk_gids:
    rep_x, rep_y = spec.checker_repeat
    rgb1 = jnp.asarray(spec.checker_rgb1, dtype=jnp.float32)
    rgb2 = jnp.asarray(spec.checker_rgb2, dtype=jnp.float32)
    for cg in chk_gids:
      gp = geom_xpos[cg][None, None, :]
      gm = geom_xmat[cg]
      dcg = hit_pos - gp
      lx_cg = gm[0, 0] * dcg[..., 0] + gm[1, 0] * dcg[..., 1] + gm[2, 0] * dcg[..., 2]
      ly_cg = gm[0, 1] * dcg[..., 0] + gm[1, 1] * dcg[..., 1] + gm[2, 1] * dcg[..., 2]
      cx = jnp.floor(lx_cg * rep_x).astype(jnp.int32)
      cy = jnp.floor(ly_cg * rep_y).astype(jnp.int32)
      even = ((cx + cy) & 1) == 0
      chk_rgb = jnp.where(even[..., None], rgb1[None, None, :], rgb2[None, None, :])
      albedo = jnp.where((geom_id == cg)[..., None], chk_rgb, albedo)

  diff_accum = jnp.zeros((B, R, 3), dtype=jnp.float32)
  spec_accum = jnp.zeros((B, R, 3), dtype=jnp.float32)
  shadow_slot_used = False

  for l_idx in range(int(spec.nlight)):
    if not bool(spec.light_active[l_idx]):
      continue
    l_type = int(spec.light_type[l_idx])
    l_pos = light_xpos[l_idx]
    l_dir = light_xdir[l_idx]
    l_dir = l_dir / jnp.maximum(jnp.linalg.norm(l_dir), 1e-12)
    l_amb = jnp.asarray(spec.light_ambient[l_idx], dtype=jnp.float32)
    l_diff = jnp.asarray(spec.light_diffuse[l_idx], dtype=jnp.float32)
    l_spec = jnp.asarray(spec.light_specular[l_idx], dtype=jnp.float32)
    l_att = jnp.asarray(spec.light_attenuation[l_idx], dtype=jnp.float32)

    diff_accum = diff_accum + l_amb[None, None, :]
    if l_type == 1:
      L = jnp.broadcast_to(-l_dir[None, None, :], (B, R, 3))
      attenuation = jnp.ones((B, R), dtype=jnp.float32)
    else:
      delta = l_pos[None, None, :] - hit_pos
      dist_to_light = jnp.linalg.norm(delta, axis=-1)
      L = delta / jnp.maximum(dist_to_light[..., None], 1e-12)
      att_denom = l_att[0] + l_att[1] * dist_to_light + l_att[2] * dist_to_light * dist_to_light
      attenuation = jnp.where(att_denom > 1e-12, 1.0 / jnp.maximum(att_denom, 1e-12), 1.0)
      if l_type == 0:
        cos_theta = jnp.sum(-L * l_dir[None, None, :], axis=-1)
        cos_cut = float(np.cos(spec.light_cutoff_rad[l_idx]))
        exp_val = float(spec.light_exponent[l_idx])
        in_cone = cos_theta >= cos_cut
        attenuation = jnp.where(in_cone, attenuation * (jnp.maximum(cos_theta, 0.0) ** exp_val), 0.0)

    ndotl = jnp.maximum(0.0, jnp.sum(normal * L, axis=-1))
    vis = jnp.ones((B, R), dtype=jnp.float32)
    if spec.use_shadows and bool(spec.light_castshadow[l_idx]) and (fused_occ is not None) and (not shadow_slot_used):
      occluded = hit & (ndotl > 0.0) & fused_occ
      shadow_slot_used = True
      vis = jnp.where(occluded, float(spec.shadow_light_fraction), 1.0)

    weight = attenuation * vis
    diff_accum = diff_accum + l_diff[None, None, :] * (ndotl * weight)[..., None]
    if spec.enable_specular:
      H_vec = L + view_dir
      H_vec = H_vec / jnp.maximum(jnp.linalg.norm(H_vec, axis=-1, keepdims=True), 1e-12)
      ndoth = jnp.maximum(0.0, jnp.sum(normal * H_vec, axis=-1))
      spec_w = jnp.where((spec_coef > 0.0) & (shin_exp > 0.0), spec_coef * (ndoth ** shin_exp) * weight, 0.0)
      if hdr_mode:
        spec_w = spec_w * 2.2
      spec_accum = spec_accum + l_spec[None, None, :] * spec_w[..., None]

  if spec.headlight_active:
    hl_amb = jnp.asarray(spec.headlight_ambient, dtype=jnp.float32)
    hl_diff = jnp.asarray(spec.headlight_diffuse, dtype=jnp.float32)
    hl_spec = jnp.asarray(spec.headlight_specular, dtype=jnp.float32)
    L_hl = -rdir
    ndotl_hl = jnp.maximum(0.0, jnp.sum(normal * L_hl, axis=-1))
    diff_accum = diff_accum + hl_amb[None, None, :] + hl_diff[None, None, :] * ndotl_hl[..., None]
    if spec.enable_specular:
      spec_w_hl = jnp.where((spec_coef > 0.0) & (shin_exp > 0.0), spec_coef * (ndotl_hl ** shin_exp), 0.0)
      spec_accum = spec_accum + hl_spec[None, None, :] * spec_w_hl[..., None]

  raw_lit = albedo * diff_accum + spec_accum + albedo * emit_coef[..., None]
  direct_rgb = jnp.maximum(raw_lit, 0.0) if hdr_mode else jnp.clip(raw_lit, 0.0, 1.0)
  sky_t = jnp.clip(0.5 * (rdir[..., 2] + 1.0), 0.0, 1.0)[..., None]
  sky_rgb = (
      sky_t * jnp.asarray(spec.sky_rgb1, dtype=jnp.float32)[None, None, :]
      + (1.0 - sky_t) * jnp.asarray(spec.sky_rgb2, dtype=jnp.float32)[None, None, :]
  )
  return jnp.where(hit[..., None], direct_rgb, sky_rgb)


def _aces_tonemap_srgb(rgb_lin: jax.Array) -> jax.Array:
  x = jnp.maximum(rgb_lin * 1.15, 0.0)
  a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
  mapped = jnp.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)
  return jnp.power(mapped, 1.0 / 2.2)


def render_thinlens_strip(
    spec: RenderSceneSpec,
    dx_1: Any,
    lray_strip: jax.Array,
    uv_chunks: jax.Array,  # (n_chunks, 8, 2)
    lens_radius: float,
    focus_dist: float,
    hdr_mode: bool = True,
    interpret: bool = False,
) -> jax.Array:
  """Render (H, strip_W, 3) thin-lens DOF image by scanning chunks of 8 concentric-disk samples."""
  cam_pos = dx_1.cam_xpos[0, int(spec.cam_id)]
  cam_mat = dx_1.cam_xmat[0, int(spec.cam_id)]
  geom_xpos = dx_1.geom_xpos[0]
  geom_xmat = dx_1.geom_xmat[0]
  light_xpos, light_xdir = _extract_light_batch(spec, dx_1)
  lxp = light_xpos[0]
  lxd = light_xdir[0]

  n_chunks = uv_chunks.shape[0]
  R_strip = lray_strip.shape[1]

  def _body(i, acc):
    uv_8 = uv_chunks[i]
    dist, normal, geom_id, fused_occ, rorig, rdir = _pallas_thinlens_raycast_8samples(
        spec, cam_pos, cam_mat, geom_xpos, geom_xmat, lxp, lxd,
        lray_strip, uv_8, lens_radius=lens_radius, focus_dist=focus_dist, interpret=interpret,
    )
    rgb8 = _shade_thinlens_hdr(
        spec, geom_xpos, geom_xmat, lxp, lxd,
        rorig, rdir, dist, normal, geom_id, fused_occ, hdr_mode=hdr_mode,
    )
    return acc + jnp.mean(rgb8, axis=0)

  acc0 = jnp.zeros((R_strip, 3), dtype=jnp.float32)
  lin_avg = jax.lax.fori_loop(0, n_chunks, _body, acc0) / float(n_chunks)
  return _aces_tonemap_srgb(lin_avg) if hdr_mode else jnp.clip(lin_avg, 0.0, 1.0)


def run_pinhole_parity_check(m: mujoco.MjModel, spec: RenderSceneSpec, dx_1: Any, interpret: bool) -> Dict[str, Any]:
  """Verify thinlens kernel at lens_radius=0.0 matches exact pallas_v2 pinhole renderer."""
  spec_lo = RenderSceneSpec.from_mjmodel(m, cam_id=spec.cam_id, height=128, width=128, use_shadows=True)
  dev0 = jax.devices()[0]
  dx_d = jax.device_put(dx_1, dev0)

  lray_np, uv_base = _build_thinlens_strip_arrays(spec_lo.cam_fovy, 128, 128, 0, 128, 8)
  lray_d = jax.device_put(jnp.asarray(lray_np, jnp.float32), dev0)
  uv_d = jax.device_put(jnp.asarray(uv_base, jnp.float32), dev0)

  @jax.jit
  def thinlens_pinhole(d):
    lx, ld = _extract_light_batch(spec_lo, d)
    dist, normal, gid, occ, ro, rd = _pallas_thinlens_raycast_8samples(
        spec_lo, d.cam_xpos[0, int(spec_lo.cam_id)], d.cam_xmat[0, int(spec_lo.cam_id)],
        d.geom_xpos[0], d.geom_xmat[0], lx[0], ld[0],
        lray_d, uv_d, lens_radius=0.0, focus_dist=1.0, interpret=interpret,
    )
    rgb = _shade_thinlens_hdr(
        spec_lo, d.geom_xpos[0], d.geom_xmat[0], lx[0], ld[0],
        ro, rd, dist, normal, gid, occ, hdr_mode=False,
    )
    return rgb[0].reshape(128, 128, 3), dist[0].reshape(128, 128), gid[0].reshape(128, 128)

  hlo = _audit_hlo(thinlens_pinhole.lower(dx_d))
  rgb_tl, dist_tl, gid_tl = thinlens_pinhole(dx_d)
  gb_ref = jax.jit(lambda d: pallas_raycast_batch_v2(spec_lo, d, interpret=interpret))(dx_d)
  jax.block_until_ready((rgb_tl, gb_ref.rgb))

  rgb_mae = float(jnp.mean(jnp.abs(rgb_tl - gb_ref.rgb[0])))
  max_rgb_diff = float(jnp.max(jnp.abs(rgb_tl - gb_ref.rgb[0])))
  geom_match = float(jnp.mean(gid_tl == gb_ref.geom_id[0]))
  hit = gid_tl >= 0
  # pallas_v2 depth = dist + znear/neg_z, while thinlens_kernel returns raw dist from znear plane; compare hit mask & rgb!
  assert geom_match == 1.0 and rgb_mae < 1e-4, (geom_match, rgb_mae, max_rgb_diff)

  res = {
      "resolution": "128x128",
      "lens_radius_m": 0.0,
      "geom_match": geom_match,
      "rgb_mae": rgb_mae,
      "max_rgb_diff": max_rgb_diff,
      "hlo_thinlens": hlo,
      "passed": True,
  }
  print("THINLENS PINHOLE PARITY PASS:", json.dumps(res, indent=2), flush=True)
  return res


def compose_4k_comparison_figure(
    img_f16: np.ndarray,
    img_near: np.ndarray,
    img_far: np.ndarray,
    out_png: str,
    stats: Dict[str, Any],
):
  """Compose a crisp Native 4K (3840x2160) side-by-side + 1:1 native crop comparison figure."""
  os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
  to_u8 = lambda a: np.clip(a * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
  u_f16 = to_u8(img_f16)
  u_near = to_u8(img_near)
  u_far = to_u8(img_far)

  canvas = Image.new("RGB", (W_4K, H_4K), (12, 15, 22))
  draw = ImageDraw.Draw(canvas, "RGBA")
  try:
    f_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 34)
    f_sub = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    f_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 19)
  except Exception:
    f_title = f_sub = f_small = ImageFont.load_default()

  # Top banner (y=0..88)
  draw.rectangle([0, 0, W_4K, 88], fill=(16, 20, 30, 255))
  title = (
      f"TPU v5p Native 4K (3840x2160) Stochastic Thin-Lens Bokeh DOF  |  "
      f"55mm Lens, Concentric-Disk 32 spp  |  4-Chip Mosaic Pallas ({stats['spp32_ms']:.0f} ms/frame, 0 Host Callbacks)"
  )
  draw.text((28, 24), title, fill=(242, 246, 255), font=f_title)

  # Top row: 3 full-FOV views scaled cleanly to 1248x702 (3*1248 + 4*24 = 3840)
  pw, ph = 1248, 702
  y_top = 104
  panels = [
      ("f/16 Deep-Focus (d_f = 1.25 m, R = 1.72 mm)", u_f16, (100, 180, 255)),
      ("f/1.4 Focus-Near (d_f = 0.82 m, R = 19.64 mm)", u_near, (80, 235, 160)),
      ("f/1.4 Focus-Far (d_f = 2.15 m, R = 19.64 mm)", u_far, (255, 175, 70)),
  ]
  for idx, (lbl, arr, accent) in enumerate(panels):
    x0 = 24 + idx * (pw + 24)
    thumb = Image.fromarray(arr).resize((pw, ph), Image.LANCZOS)
    canvas.paste(thumb, (x0, y_top))
    draw.rectangle([x0, y_top, x0 + pw, y_top + 42], fill=(12, 16, 24, 210))
    draw.rectangle([x0, y_top, x0 + 6, y_top + 42], fill=accent + (255,))
    draw.text((x0 + 18, y_top + 9), lbl, fill=(245, 248, 255), font=f_sub)

  # Bottom section: 100% Native 4K (1:1 pixel) crops!
  # Upper crop band (y=830..1470, h=620): Foreground Hero Crop (Panda gripper + red cube + gold sphere)
  # Lower crop band (y=1490..2130, h=620): Background Bokeh Orbs & Pillars Crop
  # Let's pick exact 1248x610 regions from (2160, 3840) that frame near hero vs bg orbs cleanly!
  # In camera framing: center/lower-center is foreground gripper/cube/gold sphere; upper-mid is background orbs!
  fg_y0, fg_x0 = 1320, 1280
  bg_y0, bg_x0 = 380, 620
  ch = 610

  y_fg = 832
  y_bg = 1492
  for idx, (_, arr, accent) in enumerate(panels):
    x0 = 24 + idx * (pw + 24)
    fg_crop = Image.fromarray(arr[fg_y0 : fg_y0 + ch, fg_x0 : fg_x0 + pw])
    canvas.paste(fg_crop, (x0, y_fg))
    draw.rectangle([x0, y_fg, x0 + pw, y_fg + 34], fill=(12, 16, 24, 210))
    draw.text((x0 + 14, y_fg + 6), "1:1 Native 4K Crop — Foreground Hero (d ~ 0.82 m)", fill=(230, 238, 250), font=f_small)

    bg_crop = Image.fromarray(arr[bg_y0 : bg_y0 + ch, bg_x0 : bg_x0 + pw])
    canvas.paste(bg_crop, (x0, y_bg))
    draw.rectangle([x0, y_bg, x0 + pw, y_bg + 34], fill=(12, 16, 24, 210))
    draw.text((x0 + 14, y_bg + 6), "1:1 Native 4K Crop — Background Bokeh Orbs (d ~ 2.15 m)", fill=(230, 238, 250), font=f_small)

    # Subtle border around crops
    draw.rectangle([x0, y_fg, x0 + pw - 1, y_fg + ch - 1], outline=accent + (180,), width=2)
    draw.rectangle([x0, y_bg, x0 + pw - 1, y_bg + ch - 1], outline=accent + (180,), width=2)

  canvas.save(out_png)
  print(f"Saved Native 4K Thin-Lens Bokeh Comparison: {out_png}", flush=True)


def run_thinlens_4k_suite(out_json: str, out_png: str) -> Dict[str, Any]:
  pallas_interp = (jax.default_backend() == "cpu")
  devs = jax.devices()[:4]
  assert len(devs) == 4, f"Expected 4 chips on w6, got {len(devs)}"

  xml = make_bokeh_hero_xml()
  m = mujoco.MjModel.from_xml_string(xml)
  mx = mjx.put_model(m, impl="jax")
  cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "bokeh_hero")
  assert cam_id >= 0
  spec_4k = RenderSceneSpec.from_mjmodel(m, cam_id=cam_id, height=H_4K, width=W_4K, use_shadows=True)

  d_host = mujoco.MjData(m)
  if m.nkey > 0:
    mujoco.mj_resetDataKeyframe(m, d_host, 0)
  mujoco.mj_forward(m, d_host)
  dx_1 = jax.tree_util.tree_map(lambda x: x[None, ...], mjx.put_data(m, d_host, impl="jax"))

  # 1) Pinhole parity check
  parity = run_pinhole_parity_check(m, spec_4k, dx_1, pallas_interp)

  # 2) Setup 4-chip spatial sharding across W=3840 -> 4 x 960 strips
  mesh = jax.sharding.Mesh(np.array(devs), ("sp",))
  P = jax.sharding.PartitionSpec
  sh_sp = jax.sharding.NamedSharding(mesh, P("sp", None, None))
  rep_sp = jax.sharding.NamedSharding(mesh, P())
  sm_kw = {"check_rep": False} if "check_rep" in inspect.signature(shard_map).parameters else {"check_vma": False}

  W_STRIP = W_4K // 4  # 960 (2160*960 = 2,073,600 = 4050*512)
  dx_rep = jax.device_put(dx_1, rep_sp)

  focal_len_m = 0.055  # 55mm cinema prime lens (large circular f/1.4 bokeh)
  configs = [
      ("f16_deep_focus_16spp", 16.0, 1.25, 16),
      ("f1_4_focus_near_16spp", 1.4, 0.82, 16),
      ("f1_4_focus_near_32spp", 1.4, 0.82, 32),
      ("f1_4_focus_far_32spp", 1.4, 2.15, 32),
  ]

  runs: List[Dict[str, Any]] = []
  rendered_4k: Dict[str, np.ndarray] = {}

  for label, f_stop, focus_dist, spp in configs:
    assert spp % 8 == 0
    lens_r = focal_len_m / (2.0 * f_stop)
    lray_strips = []
    uv_base = None
    for c in range(4):
      lr_c, uv_b = _build_thinlens_strip_arrays(spec_4k.cam_fovy, H_4K, W_4K, c * W_STRIP, W_STRIP, spp)
      lray_strips.append(lr_c)
      uv_base = uv_b
    lray_sh = jax.device_put(jnp.asarray(np.stack(lray_strips, axis=0), jnp.float32), sh_sp)
    uv_rep = jax.device_put(jnp.asarray(uv_base.reshape(spp // 8, 8, 2), jnp.float32), rep_sp)

    def _sp_thinlens(d, lr_s, uv_c):
      rgb_strip = render_thinlens_strip(
          spec_4k, d, lr_s[0], uv_c,
          lens_radius=lens_r, focus_dist=focus_dist, hdr_mode=True, interpret=pallas_interp,
      )
      return rgb_strip.reshape(H_4K, W_STRIP, 3)

    _sm = shard_map(
        _sp_thinlens,
        mesh=mesh,
        in_specs=(P(), P("sp", None, None), P()),
        out_specs=P(None, "sp", None),
        **sm_kw,
    )
    fn_4k = jax.jit(lambda d, lr, uv: _sm(d, lr, uv))
    lowered = fn_4k.lower(dx_rep, lray_sh, uv_rep)
    hlo = _audit_hlo(lowered)

    tc0 = time.perf_counter()
    exec_4k = lowered.compile()
    out_img = exec_4k(dx_rep, lray_sh, uv_rep)
    jax.block_until_ready(out_img)
    warmup_s = time.perf_counter() - tc0

    n_iters = 2 if spp == 32 else 3
    t1 = time.perf_counter()
    for _ in range(n_iters):
      out_img = exec_4k(dx_rep, lray_sh, uv_rep)
    jax.block_until_ready(out_img)
    dt_ms = (time.perf_counter() - t1) * 1e3 / n_iters

    img_np = np.asarray(out_img, np.float32)
    assert img_np.shape == (H_4K, W_4K, 3) and np.all(np.isfinite(img_np))
    rendered_4k[label] = img_np

    total_rays = R_4K * spp
    mrays_s = (total_rays / 1e6) / max(dt_ms / 1e3, 1e-9)
    rec = {
        "label": label,
        "resolution": f"{W_4K}x{H_4K}",
        "n_chips": 4,
        "focal_length_mm": focal_len_m * 1e3,
        "f_stop": f_stop,
        "aperture_radius_mm": lens_r * 1e3,
        "focus_dist_m": focus_dist,
        "spp": spp,
        "total_rays_per_frame": total_rays,
        "dt_ms": dt_ms,
        "fps": 1e3 / max(dt_ms, 1e-9),
        "samples_per_min": (60.0 * spp) / max(dt_ms / 1e3, 1e-9),
        "mrays_per_s": mrays_s,
        "warmup_s": round(warmup_s, 3),
        "host_callbacks": hlo["host_callbacks"],
        "send_recv_ops": hlo["send_recv_ops"],
        "gather_ops": hlo["gather_ops"],
        "hlo_zero_copy": True,
    }
    runs.append(rec)
    print(
        f"4K THINLENS {label}: {dt_ms:.1f} ms ({rec['fps']:.2f} FPS, {mrays_s:.1f} MRays/s, gather={hlo['gather_ops']})",
        flush=True,
    )

  # Compose visual comparison PNG
  compose_4k_comparison_figure(
      rendered_4k["f16_deep_focus_16spp"],
      rendered_4k["f1_4_focus_near_32spp"],
      rendered_4k["f1_4_focus_far_32spp"],
      out_png,
      stats={"spp32_ms": next(r["dt_ms"] for r in runs if r["label"] == "f1_4_focus_near_32spp")},
  )

  # Compute quantitative DOF difference metrics between f/16 and f/1.4 Focus-Near / Focus-Far
  fg_mae_near_vs_f16 = float(np.mean(np.abs(
      rendered_4k["f1_4_focus_near_32spp"][1320:1930, 1280:2528]
      - rendered_4k["f16_deep_focus_16spp"][1320:1930, 1280:2528]
  )))
  bg_mae_near_vs_f16 = float(np.mean(np.abs(
      rendered_4k["f1_4_focus_near_32spp"][380:990, 620:1868]
      - rendered_4k["f16_deep_focus_16spp"][380:990, 620:1868]
  )))
  fg_mae_far_vs_f16 = float(np.mean(np.abs(
      rendered_4k["f1_4_focus_far_32spp"][1320:1930, 1280:2528]
      - rendered_4k["f16_deep_focus_16spp"][1320:1930, 1280:2528]
  )))
  bg_mae_far_vs_f16 = float(np.mean(np.abs(
      rendered_4k["f1_4_focus_far_32spp"][380:990, 620:1868]
      - rendered_4k["f16_deep_focus_16spp"][380:990, 620:1868]
  )))

  summary = {
      "experiment": "v3_w6_thinlens_bokeh_4k",
      "worker": "w6",
      "parity": parity,
      "runs": runs,
      "dof_focus_contrast": {
          "fg_mae_near_vs_f16": fg_mae_near_vs_f16,
          "bg_mae_near_vs_f16": bg_mae_near_vs_f16,
          "fg_mae_far_vs_f16": fg_mae_far_vs_f16,
          "bg_mae_far_vs_f16": bg_mae_far_vs_f16,
          "note": "Focus-Near keeps fg sharp (low fg delta) and blurs bg (high bg delta); Focus-Far inverts this.",
      },
      "figure_path": out_png,
  }
  os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
  with open(out_json, "w") as f:
    json.dump(summary, f, indent=2)
  print(f"Saved {out_json}", flush=True)
  return summary


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--out_json", default="/dev/shm/zaychik-tpu-mjx-w6/out/v3_w6_thinlens_bokeh_4k.json")
  ap.add_argument("--out_png", default="/dev/shm/zaychik-tpu-mjx-w6/out/v3_w6_thinlens_bokeh_4k.png")
  args = ap.parse_args()
  run_thinlens_4k_suite(args.out_json, args.out_png)


if __name__ == "__main__":
  main()
