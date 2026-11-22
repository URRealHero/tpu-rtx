"""Task A: Physics-Coupled Dynamic 4DGS x MJX RoboWild Manipulation on 4-Chip TPU v5p."""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import math
import os
import re
import sys
import time
from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tpu_mjx_render.adapters.tpu_gsplat import (
    _SH_C0,
    composite_mjx_and_3dgs,
    load_cameras_json,
    load_ply,
    make_sharded_rasterizer,
)
from tpu_mjx_render.renderer import RenderSceneSpec
from tpu_mjx_render.roofline import inspect_hlo_ops

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "w3_realworld_3dgs"))
from robowild_kitchen import compute_opencv_rays, pallas_raycast_mjx_table_strip


# ---------------------------------------------------------------------------
# Helpers: HLO Check, Quaternions, Table Plane, and Surgical Segmentation
# ---------------------------------------------------------------------------

def _check_hlo(lowered, name: str) -> Dict[str, Any]:
  txt = lowered.as_text()
  info = inspect_hlo_ops(txt)
  info["send_recv_ops"] = len(re.findall(r"\bsend\(|\brecv\(", txt, re.I))
  assert info["host_callbacks"] == 0, f"{name} host_callbacks != 0: {info}"
  assert info["send_recv_ops"] == 0, f"{name} send_recv_ops != 0: {info}"
  assert info["gather_ops"] == 0, f"{name} gather_ops != 0: {info}"
  info["zero_copy_verified"] = True
  return info


def _scaled_K(cam: Dict[str, Any], width: int, height: int) -> np.ndarray:
  sf = float(height) / float(cam["height"])
  return np.array(
      [[float(cam["fx"]) * sf, 0.0, 0.5 * width],
       [0.0, float(cam["fy"]) * sf, 0.5 * height],
       [0.0, 0.0, 1.0]],
      dtype=np.float32,
  )


def _quat_mul_jnp(q1: jax.Array, q2: jax.Array) -> jax.Array:
  """Multiply (..., 4) (w, x, y, z) quaternions."""
  w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
  w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
  return jnp.stack([
      w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
      w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
      w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
      w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
  ], axis=-1)


def _rotmat_to_quat_jnp(R: jax.Array) -> jax.Array:
  """Convert (3, 3) rotation matrix to (4,) (w, x, y, z) quaternion (branchless Shepperd)."""
  tr = R[0, 0] + R[1, 1] + R[2, 2]
  q0 = jnp.array([
      jnp.sqrt(jnp.maximum(1.0 + tr, 1e-8)),
      (R[2, 1] - R[1, 2]) / jnp.sqrt(jnp.maximum(1.0 + tr, 1e-8)),
      (R[0, 2] - R[2, 0]) / jnp.sqrt(jnp.maximum(1.0 + tr, 1e-8)),
      (R[1, 0] - R[0, 1]) / jnp.sqrt(jnp.maximum(1.0 + tr, 1e-8)),
  ], dtype=jnp.float32) * 0.5
  q1 = jnp.array([
      (R[2, 1] - R[1, 2]) / jnp.sqrt(jnp.maximum(1.0 + R[0, 0] - R[1, 1] - R[2, 2], 1e-8)),
      jnp.sqrt(jnp.maximum(1.0 + R[0, 0] - R[1, 1] - R[2, 2], 1e-8)),
      (R[0, 1] + R[1, 0]) / jnp.sqrt(jnp.maximum(1.0 + R[0, 0] - R[1, 1] - R[2, 2], 1e-8)),
      (R[0, 2] + R[2, 0]) / jnp.sqrt(jnp.maximum(1.0 + R[0, 0] - R[1, 1] - R[2, 2], 1e-8)),
  ], dtype=jnp.float32) * 0.5
  q2 = jnp.array([
      (R[0, 2] - R[2, 0]) / jnp.sqrt(jnp.maximum(1.0 + R[1, 1] - R[0, 0] - R[2, 2], 1e-8)),
      (R[0, 1] + R[1, 0]) / jnp.sqrt(jnp.maximum(1.0 + R[1, 1] - R[0, 0] - R[2, 2], 1e-8)),
      jnp.sqrt(jnp.maximum(1.0 + R[1, 1] - R[0, 0] - R[2, 2], 1e-8)),
      (R[1, 2] + R[2, 1]) / jnp.sqrt(jnp.maximum(1.0 + R[1, 1] - R[0, 0] - R[2, 2], 1e-8)),
  ], dtype=jnp.float32) * 0.5
  q3 = jnp.array([
      (R[1, 0] - R[0, 1]) / jnp.sqrt(jnp.maximum(1.0 + R[2, 2] - R[0, 0] - R[1, 1], 1e-8)),
      (R[0, 2] + R[2, 0]) / jnp.sqrt(jnp.maximum(1.0 + R[2, 2] - R[0, 0] - R[1, 1], 1e-8)),
      (R[1, 2] + R[2, 1]) / jnp.sqrt(jnp.maximum(1.0 + R[2, 2] - R[0, 0] - R[1, 1], 1e-8)),
      jnp.sqrt(jnp.maximum(1.0 + R[2, 2] - R[0, 0] - R[1, 1], 1e-8)),
  ], dtype=jnp.float32) * 0.5

  c0 = tr > 0.0
  c1 = (R[0, 0] >= R[1, 1]) & (R[0, 0] >= R[2, 2])
  c2 = R[1, 1] >= R[2, 2]
  q = jnp.where(c0, q0, jnp.where(c1, q1, jnp.where(c2, q2, q3)))
  return q / jnp.maximum(jnp.linalg.norm(q), 1e-12)


def _rotmat_to_quat_np(R: np.ndarray) -> np.ndarray:
  q = np.zeros(4, dtype=np.float64)
  mujoco.mju_mat2Quat(q, R.astype(np.float64).ravel())
  return q.astype(np.float32)


def fit_full_table_plane(
    dep_hw: np.ndarray,
    R_c2w: np.ndarray,
    cam_pos: np.ndarray,
    fx: float,
    fy: float,
    cx0: float,
    cy0: float,
    place_xy_1080: Tuple[float, float],
    target_xy_1080: Tuple[float, float],
) -> Dict[str, np.ndarray]:
  """Zero-tilt full-table plane fit over the entire wooden kitchen table."""
  H, W = dep_hw.shape
  sx_s, sy_s = W / 1920.0, H / 1080.0
  ys, xs = np.mgrid[int(450 * sy_s):int(1060 * sy_s):6, int(80 * sx_s):int(1840 * sx_s):6]
  not_dozer = ~(
      (xs >= 680 * sx_s) & (xs <= 1620 * sx_s)
      & (ys >= 100 * sy_s) & (ys <= 780 * sy_s)
  )
  zs = dep_hw[ys, xs]
  ok = not_dozer & (zs > 0.2) & (zs < 6.0)
  xs_f, ys_f, zs_f = xs[ok] + 0.5, ys[ok] + 0.5, zs[ok]
  pts_cam = np.stack([(xs_f - cx0) / fx * zs_f, (ys_f - cy0) / fy * zs_f, zs_f], axis=-1)
  pts_w = pts_cam @ R_c2w.T + cam_pos[None, :]

  mask = np.ones(len(pts_w), dtype=bool)
  for _ in range(5):
    sub = pts_w[mask]
    ctr = sub.mean(axis=0)
    _, _, vt = np.linalg.svd(sub - ctr, full_matrices=False)
    nrm = vt[-1]
    res = np.abs((pts_w - ctr) @ nrm)
    mask = res < np.percentile(res, 65)

  sub = pts_w[mask]
  ctr = sub.mean(axis=0)
  _, _, vt = np.linalg.svd(sub - ctr, full_matrices=False)
  nrm = vt[-1]
  if np.dot(nrm, cam_pos - ctr) < 0.0:
    nrm = -nrm
  nrm = nrm / np.linalg.norm(nrm)
  med_res = float(np.median(np.abs((sub - ctr) @ nrm)))

  def _ray(px: float, py: float) -> np.ndarray:
    rc = np.array([(px * sx_s - cx0) / fx, (py * sy_s - cy0) / fy, 1.0], dtype=np.float32)
    rw = R_c2w @ rc
    t = float(np.dot(ctr - cam_pos, nrm) / np.dot(rw, nrm))
    return (cam_pos + t * rw).astype(np.float32)

  p_origin = _ray(*place_xy_1080)
  p_target = _ray(*target_xy_1080)
  e_x = p_target - p_origin
  e_x = e_x - np.dot(e_x, nrm) * nrm
  e_x = e_x / max(float(np.linalg.norm(e_x)), 1e-8)
  e_z = nrm.astype(np.float32)
  e_y = np.cross(e_z, e_x)
  e_y = e_y / max(float(np.linalg.norm(e_y)), 1e-8)
  R_table = np.stack([e_x, e_y, e_z], axis=1).astype(np.float32)

  return {
      "P_origin": p_origin,
      "R_table": R_table,
      "table_normal": e_z,
      "table_d": np.float32(np.dot(p_origin, e_z)),
      "median_res": med_res,
  }


def segment_lego_bulldozer(
    sp: Dict[str, Any],
    c0: Dict[str, Any],
    tbl: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, Dict[str, Any]]:
  """Segment the foreground LEGO bulldozer Gaussians cleanly from the static kitchen."""
  means = sp["means"]
  ops = sp["opacities"]
  scales = sp["scales"]
  max_scale = np.max(scales, axis=-1)
  rgb = np.clip(_SH_C0 * sp["sh_dc"][:, 0, :] + 0.5, 0.0, 1.0)

  local = (means - tbl["P_origin"][None, :]) @ tbl["R_table"]
  lx, ly, lz = local[:, 0], local[:, 1], local[:, 2]

  # Project into camera 0 at 1080p for initial seed isolation
  K = _scaled_K(c0, 1920, 1080)
  p_cam = (means - c0["position"][None, :]) @ c0["rotation"]
  in_front = p_cam[:, 2] > 0.2
  cam_dist = np.linalg.norm(p_cam, axis=-1)
  px = K[0, 0] * (p_cam[:, 0] / np.maximum(p_cam[:, 2], 1e-4)) + K[0, 2]
  py = K[1, 1] * (p_cam[:, 1] / np.maximum(p_cam[:, 2], 1e-4)) + K[1, 2]

  seed_y = (
      in_front & (cam_dist < 4.5)
      & (px >= 720) & (px <= 1600)
      & (py >= 110) & (py <= 780)
      & (ops > 0.15) & (max_scale < 0.08)
      & (lz > -0.02) & (lz < 1.00)
      & (rgb[:, 0] > 0.48) & (rgb[:, 1] > 0.34) & (rgb[:, 2] < 0.22)
      & ((rgb[:, 0] - rgb[:, 2]) > 0.28)
  )
  y_pts = local[seed_y]
  med_xy = np.median(y_pts[:, :2], axis=0)
  y_pts = y_pts[np.linalg.norm(y_pts[:, :2] - med_xy[None, :], axis=-1) < 0.95]

  xy_ctr = np.mean(y_pts[:, :2], axis=0)
  _, eigvecs = np.linalg.eigh(np.cov((y_pts[:, :2] - xy_ctr).T))
  e_long = eigvecs[:, 1]
  if e_long[0] < 0:
    e_long = -e_long
  e_lat = np.array([-e_long[1], e_long[0]], dtype=np.float32)

  uv = (local[:, :2] - xy_ctr[None, :]) @ np.stack([e_long, e_lat], axis=1)
  u_d, v_d = uv[:, 0], uv[:, 1]
  y_uv = (y_pts[:, :2] - xy_ctr[None, :]) @ np.stack([e_long, e_lat], axis=1)

  u_p1, u_p99 = np.percentile(y_uv[:, 0], [0.5, 99.5])
  v_p1, v_p99 = np.percentile(y_uv[:, 1], [0.5, 99.5])
  z_p99 = float(np.percentile(y_pts[:, 2], 99.8))

  in_uv_body = (
      (u_d >= u_p1 - 0.14) & (u_d <= u_p99 + 0.18)
      & (v_d >= v_p1 - 0.09) & (v_d <= v_p99 + 0.07)
  )
  is_yellow_col = (rgb[:, 0] > 0.44) & (rgb[:, 1] > 0.28) & (rgb[:, 2] < 0.25) & ((rgb[:, 0] - rgb[:, 2]) > 0.22)
  is_dark_col = (np.max(rgb, axis=-1) < 0.26)
  is_grey_col = (np.max(rgb, axis=-1) - np.min(rgb, axis=-1) < 0.12) & (np.mean(rgb, axis=-1) < 0.60)

  fg_mask = (
      in_uv_body
      & (max_scale < 0.22)
      & (lz <= z_p99 + 0.08)
      & (
          (lz >= 0.008)
          | ((lz >= -0.036) & (lz < 0.008) & (is_yellow_col | is_dark_col | is_grey_col))
      )
  )

  fg_uv = uv[fg_mask]
  fg_lz = lz[fg_mask]
  u_min, u_max = np.percentile(fg_uv[:, 0], [1.0, 99.0])
  v_min, v_max = np.percentile(fg_uv[:, 1], [1.0, 99.0])
  z_min, z_max = float(np.percentile(fg_lz, 1.0)), float(np.percentile(fg_lz, 99.0))

  # Center of the bulldozer in table-local frame (resting on table floor z_floor = -0.014)
  u_mid = 0.5 * (u_min + u_max)
  v_mid = 0.5 * (v_min + v_max)
  xy_box_ctr = xy_ctr + u_mid * e_long + v_mid * e_lat
  z_floor = -0.014
  half_l = 0.5 * (u_max - u_min)
  half_w = 0.5 * (v_max - v_min)
  half_h = 0.5 * (z_max - z_floor)

  info = {
      "fg_count": int(np.sum(fg_mask)),
      "total_count": int(len(means)),
      "xy_box_ctr": xy_box_ctr.tolist(),
      "e_long": e_long.tolist(),
      "e_lat": e_lat.tolist(),
      "z_floor": float(z_floor),
      "half_extents_local": [float(half_l), float(half_w), float(half_h)],
  }
  return fg_mask, info


# ---------------------------------------------------------------------------
# MJX Scene: Clean 7-DOF Franka Panda + Invisible LEGO Bulldozer Rigid Body
# ---------------------------------------------------------------------------

def build_robowild_4dgs_xml(
    dozer_pos_mjx: Tuple[float, float, float],
    dozer_quat_mjx: Tuple[float, float, float, float],
    dozer_half_mjx: Tuple[float, float, float],
    table_z_mjx: float,
) -> str:
  """Build clean MJX 7-DOF Panda + table plane + freejoint LEGO bulldozer rigid body.

  No extra floating colored base cubes! Table and dozer collider have alpha=0
  so only the real 3DGS kitchen + dynamic 4DGS LEGO bulldozer + Panda + cast shadows render.
  """
  px, py, pz = dozer_pos_mjx
  qw, qx, qy, qz = dozer_quat_mjx
  hx, hy, hz = dozer_half_mjx
  return f"""<mujoco model="robowild_4dgs_panda">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.006" iterations="4" ls_iterations="6" integrator="implicitfast">
    <flag eulerdamp="disable"/>
  </option>

  <asset>
    <material name="panda_white" rgba="0.93 0.94 0.96 1" specular="0.50" shininess="0.45"/>
    <material name="panda_off" rgba="0.78 0.81 0.85 1" specular="0.45" shininess="0.40"/>
    <material name="panda_dark" rgba="0.13 0.15 0.19 1" specular="0.55" shininess="0.50"/>
    <material name="panda_orange" rgba="0.93 0.42 0.10 1" specular="0.55" shininess="0.48"/>
  </asset>

  <default>
    <joint armature="0.05" damping="4.0" limited="true"/>
    <geom contype="0" conaffinity="0"/>
  </default>

  <worldbody>
    <light name="window_sun" pos="-2 -2 4" dir="0.45 0.55 -0.70" directional="true" castshadow="true"
      diffuse="1.0 0.98 0.94" specular="0.5 0.5 0.5"/>
    <camera name="front" pos="1.5 0 0.8" xyaxes="0 1 0 0 0 1" fovy="45"/>

    <!-- Invisible table plane collider flush with real wooden table -->
    <geom name="table_collider" type="plane" pos="0 0 {table_z_mjx:.5f}" size="5 5 0.1"
      rgba="0 0 0 0" contype="1" conaffinity="1" friction="0.42 0.015 0.01"
      solref="0.008 1" solimp="0.9 0.95 0.001"/>

    <!-- Mounting base plate flush on table -->
    <body name="panda_mount" pos="0 0 {table_z_mjx:.5f}">
      <geom name="mount_plate" type="cylinder" pos="0 0 0.015" size="0.135 0.015" material="panda_dark"/>
      <geom name="mount_ring" type="cylinder" pos="0 0 0.036" size="0.108 0.008" material="panda_orange"/>
      <geom name="p_base" type="cylinder" pos="-0.02 0 0.088" size="0.082 0.048" material="panda_off"/>
      <geom name="p_ped" type="cylinder" pos="0 0 0.21" size="0.060 0.075" material="panda_white"/>
      <body name="panda_link1" pos="0 0 0.333">
        <joint name="p_j1" axis="0 0 1" range="-2.9 2.9"/>
        <geom name="p_l1" type="capsule" fromto="0 0 -0.10 0 0 0.04" size="0.056" material="panda_white"/>
        <body name="panda_link2" quat="1 -1 0 0">
          <joint name="p_j2" axis="0 0 1" range="-1.76 1.76"/>
          <geom name="p_l2_col" type="cylinder" pos="0 0 0" size="0.058 0.040" material="panda_dark"/>
          <geom name="p_l2_arm" type="capsule" fromto="0 0 0 0 -0.22 0" size="0.048" material="panda_white"/>
          <body name="panda_link3" pos="0 -0.316 0" quat="1 1 0 0">
            <joint name="p_j3" axis="0 0 1" range="-2.9 2.9"/>
            <geom name="p_l3_seg" type="capsule" fromto="0 0 -0.11 0.045 0 0" size="0.046" material="panda_white"/>
            <geom name="p_l3_jnt" type="sphere" pos="0.0825 0 0" size="0.050" material="panda_orange"/>
            <body name="panda_link4" pos="0.0825 0 0" quat="1 1 0 0">
              <joint name="p_j4" axis="0 0 1" range="-3.07 -0.07"/>
              <geom name="p_l4_arm" type="capsule" fromto="0 0 0 -0.06 0.20 0" size="0.044" material="panda_white"/>
              <body name="panda_link5" pos="-0.0825 0.384 0" quat="1 -1 0 0">
                <joint name="p_j5" axis="0 0 1" range="-2.9 2.9"/>
                <geom name="p_l5_farm" type="capsule" fromto="0 0 -0.22 0 0 -0.02" size="0.038" material="panda_off"/>
                <geom name="p_l5_col" type="cylinder" pos="0 0 0" size="0.040 0.022" material="panda_orange"/>
                <body name="panda_link6" quat="1 1 0 0">
                  <joint name="p_j6" axis="0 0 1" range="-0.02 3.75"/>
                  <geom name="p_l6_wr" type="capsule" fromto="0 0 0 0.088 0 0" size="0.035" material="panda_white"/>
                  <body name="panda_link7" pos="0.088 0 0" quat="1 1 0 0">
                    <joint name="p_j7" axis="0 0 1" range="-2.9 2.9"/>
                    <geom name="p_l7_fl" type="cylinder" pos="0 0 0.06" size="0.034 0.040" material="panda_dark"/>
                    <body name="panda_hand" pos="0 0 0.107" quat="0.9238795 0 0 -0.3826834">
                      <geom name="p_hand" type="box" pos="0 0 0.028" size="0.022 0.092 0.022" material="panda_off"/>
                      <geom name="p_lf" type="box" pos="0 0.036 0.072" size="0.011 0.008 0.032" material="panda_dark"/>
                      <geom name="p_rf" type="box" pos="0 -0.036 0.072" size="0.011 0.008 0.032" material="panda_dark"/>
                      <geom name="p_pusher_sphere" type="sphere" pos="0 0 0.088" size="0.032"
                        material="panda_orange" contype="1" conaffinity="1" friction="0.65 0.01 0.01"
                        solref="0.006 1" solimp="0.92 0.97 0.001"/>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>

    <!-- Dynamic LEGO Bulldozer rigid body (alpha=0 collider; visual rendered by live 4DGS!) -->
    <body name="lego_dozer" pos="{px:.5f} {py:.5f} {pz:.5f}" quat="{qw:.6f} {qx:.6f} {qy:.6f} {qz:.6f}">
      <freejoint name="dozer_free"/>
      <inertial pos="0 0 0" mass="1.5" diaginertia="0.06 0.12 0.14"/>
      <geom name="dozer_col" type="box" size="{hx:.5f} {hy:.5f} {hz:.5f}"
        rgba="0 0 0 0" contype="1" conaffinity="1" friction="0.42 0.015 0.01"
        solref="0.008 1" solimp="0.9 0.95 0.001"/>
    </body>
  </worldbody>

  <actuator>
    <position name="act_j1" joint="p_j1" kp="360" kv="36"/>
    <position name="act_j2" joint="p_j2" kp="360" kv="36"/>
    <position name="act_j3" joint="p_j3" kp="320" kv="32"/>
    <position name="act_j4" joint="p_j4" kp="320" kv="32"/>
    <position name="act_j5" joint="p_j5" kp="240" kv="24"/>
    <position name="act_j6" joint="p_j6" kp="240" kv="24"/>
    <position name="act_j7" joint="p_j7" kp="180" kv="18"/>
  </actuator>
</mujoco>
"""


def plan_smooth_panda_push_trajectory(
    mj_model: mujoco.MjModel,
    dozer_pos_0: np.ndarray,
    e_long: np.ndarray,
    e_lat: np.ndarray,
    hx_mjx: float,
    hy_mjx: float,
    table_z_mjx: float,
    n_frames: int = 60,
) -> np.ndarray:
  """Solve smooth 60-step 7-DOF joint reference trajectory via damped-least-squares IK."""
  mj_data = mujoco.MjData(mj_model)
  q_init = np.array([0.28, -0.32, 0.08, -2.05, 0.05, 1.85, 0.78], dtype=np.float64)
  for j in range(7):
    mj_data.qpos[j] = q_init[j]
  mujoco.mj_forward(mj_model, mj_data)

  gid_push = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "p_pusher_sphere")
  u3 = np.array([e_long[0], e_long[1], 0.0], dtype=np.float64)
  v3 = np.array([e_lat[0], e_lat[1], 0.0], dtype=np.float64)

  # Rear-left flank of the LEGO bulldozer (visible side facing Panda & camera)
  p_flank = dozer_pos_0 + (-0.52 * hx_mjx) * u3 + (-hy_mjx - 0.032) * v3
  p_flank[2] = table_z_mjx + 0.17

  # Smooth waypoints: ready -> approach -> contact flank -> push+rotate -> follow-through -> lift
  w_ready = p_flank + (-0.16) * u3 + (-0.20) * v3 + np.array([0.0, 0.0, 0.22])
  w_pre = p_flank + (-0.05) * u3 + (-0.08) * v3 + np.array([0.0, 0.0, 0.02])
  w_touch = p_flank + 0.02 * v3
  w_push1 = p_flank + 0.14 * u3 + 0.20 * v3
  w_push2 = p_flank + 0.22 * u3 + 0.36 * v3
  w_lift = w_push2 + (-0.04) * v3 + np.array([0.0, 0.0, 0.10])

  # Piecewise smooth cosine interpolation across 60 steps (contact begins early at t~10)
  def _interp(t_norm: float) -> np.ndarray:
    if t_norm < 0.10:
      u = 0.5 - 0.5 * math.cos(math.pi * (t_norm / 0.10))
      return (1.0 - u) * w_ready + u * w_pre
    elif t_norm < 0.18:
      u = 0.5 - 0.5 * math.cos(math.pi * ((t_norm - 0.10) / 0.08))
      return (1.0 - u) * w_pre + u * w_touch
    elif t_norm < 0.55:
      u = (t_norm - 0.18) / 0.37
      return (1.0 - u) * w_touch + u * w_push1
    elif t_norm < 0.88:
      u = (t_norm - 0.55) / 0.33
      return (1.0 - u) * w_push1 + u * w_push2
    else:
      u = 0.5 - 0.5 * math.cos(math.pi * ((t_norm - 0.88) / 0.12))
      return (1.0 - u) * w_push2 + u * w_lift

  q_traj = np.zeros((n_frames, 7), dtype=np.float32)
  jacp = np.zeros((3, mj_model.nv), dtype=np.float64)
  for fi in range(n_frames):
    target_p = _interp(fi / float(n_frames - 1))
    for _ in range(25):
      mujoco.mj_forward(mj_model, mj_data)
      cur_p = mj_data.geom_xpos[gid_push].copy()
      err = target_p - cur_p
      if np.linalg.norm(err) < 1e-4:
        break
      mujoco.mj_jacGeom(mj_model, mj_data, jacp, None, gid_push)
      J = jacp[:, :7]
      dq = J.T @ np.linalg.solve(J @ J.T + 0.015 * np.eye(3), err)
      # Mild posture regularization toward natural elbow-up config
      dq_post = 0.04 * (q_init - mj_data.qpos[:7])
      mj_data.qpos[:7] = np.clip(
          mj_data.qpos[:7] + 0.65 * dq + (np.eye(7) - np.linalg.pinv(J) @ J) @ dq_post,
          mj_model.jnt_range[:7, 0] + 0.02,
          mj_model.jnt_range[:7, 1] - 0.02,
      )
    q_traj[fi] = mj_data.qpos[:7].astype(np.float32)
  return q_traj


def _shard_scene_with_mask(
    sp: Dict[str, Any],
    fg_mask: np.ndarray,
    mesh: jax.sharding.Mesh,
    seed: int = 42,
):
  """Shuffle, pad, and shard kitchen Gaussians + dynamic foreground mask across 4 chips."""
  n_chips = len(mesh.devices)
  N = int(sp["n_gaussians"])
  N_pad = ((N + n_chips - 1) // n_chips) * n_chips
  rng = np.random.default_rng(seed)
  perm = rng.permutation(N)

  def _pad(arr, fill=0.0, dtype=np.float32):
    a = arr[perm]
    if N_pad > N:
      pad_w = [(0, N_pad - N)] + [(0, 0)] * (a.ndim - 1)
      a = np.pad(a, pad_w, constant_values=fill)
    return np.ascontiguousarray(a, dtype=dtype)

  means = _pad(sp["means"])
  quats = _pad(sp["quats"])
  quats[N:, 0] = 1.0
  scales = _pad(sp["scales"], fill=1e-4)
  opacities = _pad(sp["opacities"], fill=0.0)
  sh = _pad(sp["sh"])
  is_dyn = _pad(fg_mask.astype(np.bool_), fill=False, dtype=np.bool_)

  dp = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("dp"))
  return (
      jax.device_put(jnp.asarray(means), dp),
      jax.device_put(jnp.asarray(quats), dp),
      jax.device_put(jnp.asarray(scales), dp),
      jax.device_put(jnp.asarray(opacities), dp),
      jax.device_put(jnp.asarray(sh), dp),
      jax.device_put(jnp.asarray(is_dyn), dp),
      N_pad,
  )


# ---------------------------------------------------------------------------
# Filmstrip Composer
# ---------------------------------------------------------------------------

def build_6frame_filmstrip(
    frames_4k: List[np.ndarray],
    frame_meta: List[Dict[str, Any]],
    fps_1080: float,
    fps_4k: float,
    fg_count: int,
    total_count: int,
    out_path: str,
) -> None:
  """Compose a razor-sharp 3x2 (3840x1560) filmstrip supersampled from Native 4K renders."""
  pw, ph = 1280, 720
  header_h = 120
  gap = 8
  W_tot = 3 * pw + 2 * gap
  H_tot = header_h + 2 * ph + gap

  canvas = Image.new("RGB", (W_tot, H_tot), (12, 16, 24))
  draw = ImageDraw.Draw(canvas, "RGBA")
  try:
    font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 34)
    font_sub = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    font_pill_b = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    font_pill_s = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
  except Exception:
    font_title = font_sub = font_pill_b = font_pill_s = ImageFont.load_default()

  title = "Physics-Coupled Dynamic 4DGS x MJX RoboWild Manipulation (4-Chip TPU v5p)"
  subtitle = (
      f"Mip-NeRF 360 Kitchen ({total_count:,} 3DGS, {fg_count:,} Dynamic LEGO Dozer) | "
      f"MJX 7-DOF Panda Contact Push | {fps_1080:.2f} FPS @ 1080p, {fps_4k:.2f} FPS @ 4K"
  )
  draw.text((28, 20), title, fill=(248, 250, 255), font=font_title)
  draw.text((28, 68), subtitle, fill=(185, 205, 230), font=font_sub)

  for idx, (arr_4k, fm) in enumerate(zip(frames_4k, frame_meta)):
    r = idx // 3
    c = idx % 3
    x0 = c * (pw + gap)
    y0 = header_h + r * (ph + gap)
    im = Image.fromarray(np.clip(np.round(arr_4k * 255.0), 0, 255).astype(np.uint8))
    im_s = im.resize((pw, ph), Image.Resampling.LANCZOS)
    canvas.paste(im_s, (x0, y0))

    p_title = f"t = {fm['frame']:02d} / 59  |  {fm['phase']}"
    p_sub = f"Dozer Δpos: {fm['disp_cm']:+.1f} cm   |   Δyaw: {fm['yaw_deg']:+.1f}°"
    tw = max(draw.textlength(p_title, font=font_pill_b), draw.textlength(p_sub, font=font_pill_s))
    bx0, by0 = x0 + 18, y0 + 16
    draw.rounded_rectangle(
        [bx0, by0, bx0 + int(tw) + 28, by0 + 62],
        radius=10, fill=(10, 14, 22, 198), outline=(255, 170, 55, 230), width=2,
    )
    draw.text((bx0 + 14, by0 + 8), p_title, fill=(255, 255, 255, 255), font=font_pill_b)
    draw.text((bx0 + 14, by0 + 35), p_sub, fill=(255, 215, 130, 255), font=font_pill_s)

  canvas.save(out_path)
  print(f"Saved 6-frame filmstrip -> {out_path}", flush=True)


# ---------------------------------------------------------------------------
# Main Runner
# ---------------------------------------------------------------------------

def run_task_a(
    data_root: str = "/dev/shm/zaychik-w3-v6/data",
    out_json: str = "/dev/shm/zaychik-w3-v6/out/v3_w3_wave6_4dgs_300m.json",
    export_dir: str = "/dev/shm/zaychik-w3-v6/out",
) -> None:
  os.makedirs(export_dir, exist_ok=True)
  os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)

  devs = jax.devices()[:4]
  mesh = jax.sharding.Mesh(np.array(devs), ("dp",))
  rep = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
  P = jax.sharding.PartitionSpec
  sm_kw = {"check_vma": False} if "check_vma" in inspect.signature(jax.shard_map).parameters else {"check_rep": False}

  sp = load_ply(os.path.join(data_root, "kitchen/point_cloud/iteration_30000/point_cloud.ply"))
  cams = load_cameras_json(os.path.join(data_root, "kitchen/cameras.json"))
  c0 = cams[0]

  # 1. Fit full-table plane at 1080p
  W_fit, H_fit = 1920, 1080
  K_1080 = _scaled_K(c0, W_fit, H_fit)
  fn_1080_init = make_sharded_rasterizer(
      mesh, W_fit, H_fit, sh_degree=3, macro_size=32, macro_rep=5,
      k_macro=10240, k_large_per_chip=1536, k_tile=512,
  )
  m0, q0, s0, o0, sh0, dyn0, _ = _shard_scene_with_mask(
      sp, np.zeros((sp["n_gaussians"],), dtype=bool), mesh
  )
  v_dev = jax.device_put(jnp.asarray(c0["viewmat"]), rep)
  k1080_dev = jax.device_put(jnp.asarray(K_1080), rep)
  _, _, dep_init = fn_1080_init(m0, q0, s0, o0, sh0, v_dev, k1080_dev)
  jax.block_until_ready(dep_init)

  tbl = fit_full_table_plane(
      np.asarray(dep_init[..., 0]),
      c0["rotation"], c0["position"],
      float(K_1080[0, 0]), float(K_1080[1, 1]), 0.5 * W_fit, 0.5 * H_fit,
      place_xy_1080=(450.0, 645.0),
      target_xy_1080=(900.0, 635.0),
  )
  del m0, q0, s0, o0, sh0, dyn0, dep_init
  gc.collect()

  # 2. Segment LEGO bulldozer
  fg_mask, seg_info = segment_lego_bulldozer(sp, c0, tbl)
  print(f"Segmented LEGO bulldozer: {seg_info['fg_count']:,} / {seg_info['total_count']:,} Gaussians", flush=True)
  print(f"Table fit median_res={tbl['median_res']:.5f}, dozer_ctr={seg_info['xy_box_ctr']}", flush=True)

  # Shard kitchen scene + boolean mask onto 4 chips
  m_orig, q_orig, s_orig, o_orig, sh_orig, is_dyn, _ = _shard_scene_with_mask(sp, fg_mask, mesh)

  # 3. Build MJX model & plan 60-step push/rotate trajectory
  robot_scale = 1.48
  table_z_mjx = float(seg_info["z_floor"] / robot_scale)
  e_long = np.asarray(seg_info["e_long"], dtype=np.float32)
  e_lat = np.asarray(seg_info["e_lat"], dtype=np.float32)
  hx_mjx, hy_mjx, hz_mjx = [v / robot_scale for v in seg_info["half_extents_local"]]
  xy_ctr_mjx = np.asarray(seg_info["xy_box_ctr"], dtype=np.float32) / robot_scale
  dozer_pos_0 = np.array([xy_ctr_mjx[0], xy_ctr_mjx[1], table_z_mjx + hz_mjx], dtype=np.float32)
  R_dozer_0 = np.column_stack([
      np.array([e_long[0], e_long[1], 0.0], dtype=np.float32),
      np.array([e_lat[0], e_lat[1], 0.0], dtype=np.float32),
      np.array([0.0, 0.0, 1.0], dtype=np.float32),
  ])
  dozer_quat_0 = _rotmat_to_quat_np(R_dozer_0)

  xml_str = build_robowild_4dgs_xml(
      tuple(float(v) for v in dozer_pos_0),
      tuple(float(v) for v in dozer_quat_0),
      (float(hx_mjx), float(hy_mjx), float(hz_mjx)),
      table_z_mjx,
  )
  mj_model = mujoco.MjModel.from_xml_string(xml_str)
  mjx_model = mjx.put_model(mj_model, impl="jax")
  dozer_bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "lego_dozer")

  n_frames = 60
  n_substeps = 2
  q_traj_np = plan_smooth_panda_push_trajectory(
      mj_model, dozer_pos_0, e_long, e_lat, hx_mjx, hy_mjx, table_z_mjx, n_frames=n_frames
  )

  d_init = mjx.make_data(mj_model, impl="jax")
  qpos0 = mj_model.qpos0.copy()
  qpos0[:7] = q_traj_np[0]
  d_init = d_init.replace(qpos=jnp.asarray(qpos0, jnp.float32), ctrl=jnp.asarray(q_traj_np[0], jnp.float32))
  d_init = mjx.forward(mjx_model, d_init)

  R_tbl = jnp.asarray(tbl["R_table"], dtype=jnp.float32)
  P_org = jnp.asarray(tbl["P_origin"], dtype=jnp.float32)
  t_nrm = jnp.asarray(tbl["table_normal"], dtype=jnp.float32)
  t_d = jnp.asarray(tbl["table_d"], dtype=jnp.float32)

  sun_local = jnp.array([-0.52, 0.52, 0.74], dtype=jnp.float32)
  sun_local = sun_local / jnp.linalg.norm(sun_local)
  sun_dir_w = R_tbl @ sun_local

  P_0_w = (jnp.asarray(dozer_pos_0) * robot_scale) @ R_tbl.T + P_org
  R_0_w = R_tbl @ jnp.asarray(R_dozer_0, dtype=jnp.float32)

  # Compile pure physics step to verify 60-step trajectory & displacements
  @jax.jit
  def _step_physics(d_in, ctrl_t):
    d_c = d_in.replace(ctrl=ctrl_t)
    def _sub(_, d_s):
      return mjx.step(mjx_model, d_s)
    return jax.lax.fori_loop(0, n_substeps, _sub, d_c)

  # Helper to build dynamic 4DGS + MJX coupled renderer at any resolution (1080p or 4K)
  def _make_coupled_renderer(W: int, H: int, m_sz: int, m_rep: int, k_mac: int, k_lg: int, k_tile: int):
    spec_strip = RenderSceneSpec.from_mjmodel(mj_model, cam_id=0, height=H // 4, width=W, use_shadows=True)
    gs_fn = make_sharded_rasterizer(
        mesh, W, H, sh_degree=3, macro_size=m_sz, macro_rep=m_rep,
        k_macro=k_mac, k_large_per_chip=k_lg, k_tile=k_tile,
    )
    _sm_mjx = jax.shard_map(
        lambda gxp, gxm, gsz, grg, sd, tn, td, ro, rd, cz: pallas_raycast_mjx_table_strip(
            spec_strip, gxp, gxm, gsz, grg, sd, tn, td, ro, rd, cz
        ),
        mesh=mesh,
        in_specs=(P(), P(), P(), P(), P(), P(), P(), P("dp"), P("dp"), P("dp")),
        out_specs=(P("dp"), P("dp"), P("dp"), P("dp"), P("dp"), P("dp"), P("dp")),
        **sm_kw,
    )
    _sm_update_4dgs = jax.shard_map(
        lambda mo, qo, dyn, pt, rd, qd: (
            jnp.where(dyn[:, None], pt[None, :] + (mo - P_0_w[None, :]) @ rd.T, mo),
            jnp.where(dyn[:, None], _quat_mul_jnp(qd[None, :], qo), qo),
        ),
        mesh=mesh,
        in_specs=(P("dp"), P("dp"), P("dp"), P(), P(), P()),
        out_specs=(P("dp"), P("dp")),
        **sm_kw,
    )

    @jax.jit
    def _coupled_step_and_render(d_in, ctrl_t, mo, qo, so, oo, sho, dyn, vm, Km, cp, rc2w):
      d_out = _step_physics(d_in, ctrl_t)

      # Live 4DGS rigid pose coupling from MJX body pose
      pos_t_mjx = d_out.xpos[dozer_bid]
      mat_t_mjx = d_out.xmat[dozer_bid]
      P_t_w = (pos_t_mjx * robot_scale) @ R_tbl.T + P_org
      R_t_w = R_tbl @ mat_t_mjx
      R_delta = R_t_w @ R_0_w.T
      q_delta = _rotmat_to_quat_jnp(R_delta)
      m_live, q_live = _sm_update_4dgs(mo, qo, dyn, P_t_w, R_delta, q_delta)

      # MJX Panda raytracing + cast table shadows
      g_xpos = (d_out.geom_xpos * robot_scale) @ R_tbl.T + P_org[None, :]
      g_xmat = jnp.einsum("ij,gjk->gik", R_tbl, d_out.geom_xmat, precision=jax.lax.Precision.HIGHEST)
      g_size = jnp.asarray(spec_strip.geom_size, dtype=jnp.float32) * robot_scale
      g_rgba = jnp.asarray(spec_strip.geom_rgba, dtype=jnp.float32)

      ro, rd, cz = compute_opencv_rays(cp, rc2w, Km, H, W)
      alb_s, nrm_s, zd_s, rsh_s, tsh_s, tz_s, vd_s = _sm_mjx(
          g_xpos, g_xmat, g_size, g_rgba, sun_dir_w, t_nrm, t_d, ro, rd, cz
      )
      alb = alb_s.reshape(H, W, 3)
      p_nrm = nrm_s.reshape(H, W, 3)
      p_vd = vd_s.reshape(H, W, 3)
      r_sh = rsh_s.reshape(H, W)
      mjx_z = zd_s.reshape(H, W)
      tbl_sh = tsh_s.reshape(H, W)
      tbl_z = tz_s.reshape(H, W)

      ndotl = jnp.maximum(0.0, jnp.sum(p_nrm * sun_dir_w[None, None, :], axis=-1))
      hemi = 0.5 + 0.5 * jnp.sum(p_nrm * t_nrm[None, None, :], axis=-1)
      hvec = sun_dir_w[None, None, :] + p_vd
      hvec = hvec / jnp.maximum(jnp.linalg.norm(hvec, axis=-1, keepdims=True), 1e-12)
      spec_term = (jnp.maximum(0.0, jnp.sum(p_nrm * hvec, axis=-1)) ** 32.0) * 0.25 * r_sh
      lit = (0.36 + 0.22 * hemi + 0.56 * ndotl * r_sh)[..., None] * alb + spec_term[..., None]
      mjx_rgb = jnp.clip(lit, 0.0, 1.0)

      g_rgb, g_alp, g_dep = gs_fn(m_live, q_live, so, oo, sho, vm, Km)
      g_alp2 = g_alp[..., 0]
      g_dep2 = g_dep[..., 0]

      on_tbl = (tbl_sh > 0.5) & (g_dep2 > 0.0) & (jnp.abs(g_dep2 - tbl_z) < 0.25)
      g_rgb_sh = jnp.where(on_tbl[..., None], g_rgb * jnp.array([0.44, 0.45, 0.48], jnp.float32), g_rgb)
      comp_rgb, _ = composite_mjx_and_3dgs(mjx_rgb, mjx_z, g_rgb_sh, g_alp2, g_dep2)
      return d_out, comp_rgb, mjx_z, g_dep2, on_tbl

    return _coupled_step_and_render

  # Run 60-step MJX physics rollout first to verify exact displacements & yaw!
  d_cur = jax.device_put(d_init, rep)
  q_traj_dev = jax.device_put(jnp.asarray(q_traj_np), rep)
  traj_states: List[Any] = []
  traj_stats: List[Dict[str, float]] = []
  yaw0 = math.atan2(R_dozer_0[1, 0], R_dozer_0[0, 0])

  for fi in range(n_frames):
    if fi > 0:
      d_cur = _step_physics(d_cur, q_traj_dev[fi])
    traj_states.append(d_cur)
  jax.block_until_ready(d_cur)

  for fi in range(n_frames):
    st = traj_states[fi]
    pos_mjx = np.asarray(st.xpos[dozer_bid])
    mat_mjx = np.asarray(st.xmat[dozer_bid])
    disp_m = float(np.linalg.norm(pos_mjx[:2] - dozer_pos_0[:2])) * robot_scale
    yaw = math.atan2(mat_mjx[1, 0], mat_mjx[0, 0])
    dyaw = math.degrees((yaw - yaw0 + math.pi) % (2 * math.pi) - math.pi)
    traj_stats.append({
        "frame": fi,
        "disp_units": round(disp_m, 4),
        "disp_cm": round(disp_m * 35.0, 2),  # ~35 cm/unit real LEGO Technic scale
        "yaw_deg": round(dyaw, 2),
        "z_mjx": round(float(pos_mjx[2]), 4),
    })

  print("Physics rollout summary (key frames):")
  for fi in (0, 12, 24, 36, 48, 59):
    print(" ", traj_stats[fi], flush=True)

  # Load or initialize metrics JSON
  if os.path.exists(out_json):
    with open(out_json) as f:
      report = json.load(f)
  else:
    report = {
        "backend": jax.default_backend(),
        "device_kind": jax.devices()[0].device_kind,
        "jax_version": jax.__version__,
        "interpret": False,
        "hlo_checks": {},
    }

  cp_dev = jax.device_put(jnp.asarray(c0["position"]), rep)
  rc2w_dev = jax.device_put(jnp.asarray(c0["rotation"]), rep)

  # 4. Benchmark 1080p coupled 4DGS + MJX
  print("\n--- Benchmarking Dynamic 4DGS + MJX @ 1080p ---", flush=True)
  fn_1080 = _make_coupled_renderer(1920, 1080, 32, 5, 10240, 1536, 512)
  args_1080 = (
      traj_states[0], q_traj_dev[1],
      m_orig, q_orig, s_orig, o_orig, sh_orig, is_dyn,
      v_dev, k1080_dev, cp_dev, rc2w_dev,
  )
  report["hlo_checks"]["dynamic_4dgs_robowild_1080p"] = _check_hlo(
      fn_1080.lower(*args_1080), "dynamic_4dgs_robowild_1080p"
  )
  t0 = time.perf_counter()
  out0 = fn_1080(*args_1080)
  jax.block_until_ready(out0)
  comp_1080_s = time.perf_counter() - t0
  # Warmup + timed 5 steps
  jax.block_until_ready(fn_1080(*args_1080))
  t1 = time.perf_counter()
  for k_i in range(5):
    out_b = fn_1080(
        traj_states[k_i * 10], q_traj_dev[k_i * 10 + 1],
        m_orig, q_orig, s_orig, o_orig, sh_orig, is_dyn,
        v_dev, k1080_dev, cp_dev, rc2w_dev,
    )
  jax.block_until_ready(out_b)
  ms_1080 = (time.perf_counter() - t1) * 1e3 / 5.0
  fps_1080 = 1000.0 / ms_1080
  print(f"Dynamic 4DGS+MJX 1080p: {ms_1080:.2f} ms ({fps_1080:.2f} FPS) | compile={comp_1080_s:.1f}s", flush=True)
  del fn_1080, out0, out_b
  gc.collect()

  # 5. Benchmark Native 4K + render 6-frame filmstrip
  print("\n--- Benchmarking Dynamic 4DGS + MJX @ Native 4K & rendering 6-frame filmstrip ---", flush=True)
  K_4k = _scaled_K(c0, 3840, 2160)
  k4k_dev = jax.device_put(jnp.asarray(K_4k), rep)
  fn_4k = _make_coupled_renderer(3840, 2160, 64, 4, 8192, 2048, 512)
  args_4k = (
      traj_states[0], q_traj_dev[1],
      m_orig, q_orig, s_orig, o_orig, sh_orig, is_dyn,
      v_dev, k4k_dev, cp_dev, rc2w_dev,
  )
  report["hlo_checks"]["dynamic_4dgs_robowild_4k"] = _check_hlo(
      fn_4k.lower(*args_4k), "dynamic_4dgs_robowild_4k"
  )
  t0 = time.perf_counter()
  out4_0 = fn_4k(*args_4k)
  jax.block_until_ready(out4_0)
  comp_4k_s = time.perf_counter() - t0
  jax.block_until_ready(fn_4k(*args_4k))

  key_indices = [0, 12, 24, 36, 48, 59]
  phase_names = [
      "Ready / Approach",
      "Reaching Flank",
      "Contact & Push",
      "Pushing & Rotating",
      "Pivoting Dozer",
      "Final Pushed Pose",
  ]
  frames_4k: List[np.ndarray] = []
  frame_meta: List[Dict[str, Any]] = []

  t1 = time.perf_counter()
  for idx, (fi, ph) in enumerate(zip(key_indices, phase_names)):
    st_in = traj_states[fi]
    ctrl_in = q_traj_dev[fi]
    _, rgb_4k, mjz, gsz, ont = fn_4k(
        st_in, ctrl_in, m_orig, q_orig, s_orig, o_orig, sh_orig, is_dyn,
        v_dev, k4k_dev, cp_dev, rc2w_dev,
    )
    jax.block_until_ready(rgb_4k)
    frames_4k.append(np.asarray(rgb_4k))
    st_m = traj_stats[fi]
    frame_meta.append({
        **st_m,
        "phase": ph,
        "mjx_robot_pixels_4k": int(jnp.sum(mjz > 0)),
        "cast_shadow_pixels_4k": int(jnp.sum(ont)),
    })
  ms_4k = (time.perf_counter() - t1) * 1e3 / len(key_indices)
  fps_4k = 1000.0 / ms_4k
  print(f"Dynamic 4DGS+MJX Native 4K: {ms_4k:.2f} ms ({fps_4k:.2f} FPS) | compile={comp_4k_s:.1f}s", flush=True)

  filmstrip_path = os.path.join(export_dir, "v3_w3_4dgs_robowild_manipulation.png")
  build_6frame_filmstrip(
      frames_4k, frame_meta, fps_1080, fps_4k,
      seg_info["fg_count"], seg_info["total_count"], filmstrip_path,
  )
  # Also save final 4K hero frame separately for inspection if needed
  Image.fromarray(np.clip(np.round(frames_4k[-1] * 255.0), 0, 255).astype(np.uint8)).save(
      os.path.join(export_dir, "v3_w3_4dgs_robowild_final_4k.png")
  )

  report["dynamic_4dgs_robowild"] = {
      "scene": "MipNeRF360_Kitchen_Dynamic_4DGS_MJX_Panda",
      "n_chips": 4,
      "n_gaussians_total": seg_info["total_count"],
      "n_gaussians_dynamic_dozer": seg_info["fg_count"],
      "n_gaussians_static_bg": seg_info["total_count"] - seg_info["fg_count"],
      "table_fit_median_res": round(tbl["median_res"], 5),
      "robot_scale": robot_scale,
      "n_trajectory_frames": n_frames,
      "substeps_per_frame": n_substeps,
      "perf_1080p": {
          "res": "1920x1080",
          "ms": round(ms_1080, 2),
          "fps": round(fps_1080, 2),
          "compile_s": round(comp_1080_s, 2),
      },
      "perf_4k": {
          "res": "3840x2160",
          "ms": round(ms_4k, 2),
          "fps": round(fps_4k, 2),
          "compile_s": round(comp_4k_s, 2),
      },
      "key_frames": frame_meta,
      "final_displacement_units": traj_stats[-1]["disp_units"],
      "final_displacement_cm": traj_stats[-1]["disp_cm"],
      "final_yaw_deg": traj_stats[-1]["yaw_deg"],
      "filmstrip_path": "out/figures/v3_w3_4dgs_robowild_manipulation.png",
  }
  with open(out_json, "w") as f:
    json.dump(report, f, indent=2)
  print(f"Saved Task A metrics -> {out_json}", flush=True)


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--data_root", default="/dev/shm/zaychik-w3-v6/data")
  ap.add_argument("--out_json", default="/dev/shm/zaychik-w3-v6/out/v3_w3_wave6_4dgs_300m.json")
  ap.add_argument("--export_dir", default="/dev/shm/zaychik-w3-v6/out")
  args = ap.parse_args()
  run_task_a(args.data_root, args.out_json, args.export_dir)


if __name__ == "__main__":
  main()
