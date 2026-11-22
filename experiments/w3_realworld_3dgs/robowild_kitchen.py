"""RoboWild Real-to-Sim: MJX 7-DOF Franka Panda + Cast Shadows on Real Mip-NeRF 360 Kitchen Table."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
import mujoco
import numpy as np

from tpu_mjx_render.primitives import (
    GEOM_BOX,
    GEOM_CAPSULE,
    GEOM_CYLINDER,
    GEOM_ELLIPSOID,
    GEOM_PLANE,
    GEOM_SPHERE,
    MJ_MAXVAL,
    MJ_MINVAL,
)
from tpu_mjx_render.renderer import RenderSceneSpec


def build_robowild_panda_xml() -> str:
  """Build MJX 7-DOF Franka Panda manipulator + pedestal + grasp cubes sitting flush on z=0."""
  return """<mujoco model="robowild_panda">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.004" iterations="3" ls_iterations="6" integrator="implicitfast">
    <flag eulerdamp="disable"/>
  </option>

  <asset>
    <material name="panda_white" rgba="0.93 0.94 0.96 1" specular="0.50" shininess="0.45"/>
    <material name="panda_off" rgba="0.78 0.81 0.85 1" specular="0.45" shininess="0.40"/>
    <material name="panda_dark" rgba="0.13 0.15 0.19 1" specular="0.55" shininess="0.50"/>
    <material name="panda_orange" rgba="0.93 0.42 0.10 1" specular="0.55" shininess="0.48"/>
    <material name="cube_red" rgba="0.88 0.14 0.12 1" specular="0.40" shininess="0.35"/>
    <material name="cube_blue" rgba="0.12 0.45 0.90 1" specular="0.40" shininess="0.35"/>
    <material name="cube_green" rgba="0.15 0.72 0.32 1" specular="0.40" shininess="0.35"/>
  </asset>

  <default>
    <joint armature="0.03" damping="3.0" limited="true"/>
    <geom contype="0" conaffinity="0"/>
  </default>

  <worldbody>
    <light name="window_sun" pos="-2 -2 4" dir="0.45 0.55 -0.70" directional="true" castshadow="true"
      diffuse="1.0 0.98 0.94" specular="0.5 0.5 0.5"/>
    <camera name="front" pos="1.5 0 0.8" xyaxes="0 1 0 0 0 1" fovy="45"/>

    <!-- Mounting base plate flush at z=0 -->
    <body name="panda_mount" pos="0 0 0">
      <geom name="mount_plate" type="cylinder" pos="0 0 0.015" size="0.145 0.015" material="panda_dark"/>
      <geom name="mount_ring" type="cylinder" pos="0 0 0.036" size="0.115 0.008" material="panda_orange"/>
      <geom name="p_base" type="cylinder" pos="-0.02 0 0.088" size="0.085 0.048" material="panda_off"/>
      <geom name="p_ped" type="cylinder" pos="0 0 0.21" size="0.062 0.075" material="panda_white"/>
      <body name="panda_link1" pos="0 0 0.333">
        <joint name="p_j1" axis="0 0 1" range="-2.9 2.9"/>
        <geom name="p_l1" type="capsule" fromto="0 0 -0.10 0 0 0.04" size="0.058" material="panda_white"/>
        <body name="panda_link2" quat="1 -1 0 0">
          <joint name="p_j2" axis="0 0 1" range="-1.76 1.76"/>
          <geom name="p_l2_col" type="cylinder" pos="0 0 0" size="0.060 0.042" material="panda_dark"/>
          <geom name="p_l2_arm" type="capsule" fromto="0 0 0 0 -0.22 0" size="0.050" material="panda_white"/>
          <body name="panda_link3" pos="0 -0.316 0" quat="1 1 0 0">
            <joint name="p_j3" axis="0 0 1" range="-2.9 2.9"/>
            <geom name="p_l3_seg" type="capsule" fromto="0 0 -0.11 0.045 0 0" size="0.048" material="panda_white"/>
            <geom name="p_l3_jnt" type="sphere" pos="0.0825 0 0" size="0.052" material="panda_orange"/>
            <body name="panda_link4" pos="0.0825 0 0" quat="1 1 0 0">
              <joint name="p_j4" axis="0 0 1" range="-3.07 -0.07"/>
              <geom name="p_l4_arm" type="capsule" fromto="0 0 0 -0.06 0.20 0" size="0.046" material="panda_white"/>
              <body name="panda_link5" pos="-0.0825 0.384 0" quat="1 -1 0 0">
                <joint name="p_j5" axis="0 0 1" range="-2.9 2.9"/>
                <geom name="p_l5_farm" type="capsule" fromto="0 0 -0.22 0 0 -0.02" size="0.040" material="panda_off"/>
                <geom name="p_l5_col" type="cylinder" pos="0 0 0" size="0.042 0.022" material="panda_orange"/>
                <body name="panda_link6" quat="1 1 0 0">
                  <joint name="p_j6" axis="0 0 1" range="-0.02 3.75"/>
                  <geom name="p_l6_wr" type="capsule" fromto="0 0 0 0.088 0 0" size="0.036" material="panda_white"/>
                  <body name="panda_link7" pos="0.088 0 0" quat="1 1 0 0">
                    <joint name="p_j7" axis="0 0 1" range="-2.9 2.9"/>
                    <geom name="p_l7_fl" type="cylinder" pos="0 0 0.06" size="0.034 0.040" material="panda_dark"/>
                    <body name="panda_hand" pos="0 0 0.107" quat="0.9238795 0 0 -0.3826834">
                      <geom name="p_hand" type="box" pos="0 0 0.028" size="0.022 0.092 0.022" material="panda_off"/>
                      <geom name="p_lf" type="box" pos="0 0.042 0.070" size="0.010 0.008 0.028" material="panda_dark"/>
                      <geom name="p_rf" type="box" pos="0 -0.042 0.070" size="0.010 0.008 0.028" material="panda_dark"/>
                      <geom name="p_held_cube" type="box" pos="0 0 0.078" size="0.024 0.024 0.024" material="cube_red"/>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>

    <!-- Tabletop manipulation cubes sitting cleanly on table surface -->
    <body name="table_cubes" pos="0.34 -0.14 0.02">
      <geom name="tbl_cube_blue" type="box" pos="0.0 0.0 0.026" size="0.026 0.026 0.026" euler="0 0 0.35" material="cube_blue"/>
      <geom name="tbl_cube_green" type="box" pos="-0.07 0.09 0.023" size="0.023 0.023 0.023" euler="0 0 -0.25" material="cube_green"/>
    </body>
  </worldbody>
</mujoco>
"""


def set_robowild_panda_pose(mj_model: mujoco.MjModel) -> np.ndarray:
  """Set graceful reaching pose for the RoboWild Panda arm."""
  qpos = mj_model.qpos0.copy()
  joints = {
      "p_j1": 0.18,
      "p_j2": 0.36,
      "p_j3": 0.12,
      "p_j4": -1.58,
      "p_j5": -0.15,
      "p_j6": 1.98,
      "p_j7": 0.78,
  }
  for name, val in joints.items():
    jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid >= 0:
      qpos[int(mj_model.jnt_qposadr[jid])] = val
  return qpos.astype(np.float32)


def fit_table_plane_from_depth(
    dep_hw: np.ndarray,
    R_c2w: np.ndarray,
    cam_pos: np.ndarray,
    fx: float,
    fy: float,
    cx0: float,
    cy0: float,
    roi: Tuple[int, int, int, int],
    place_xy: Tuple[float, float],
    target_xy: Tuple[float, float],
) -> Dict[str, np.ndarray]:
  """Robustly fit a 3D table plane and construct orthonormal `(R_table, P_origin)` for MJX Z-up."""
  y0, y1, x0, x1 = roi
  ys, xs = np.mgrid[y0:y1:6, x0:x1:6]
  zs = dep_hw[ys, xs]
  ok = zs > 0.1
  xs_f, ys_f, zs_f = xs[ok] + 0.5, ys[ok] + 0.5, zs[ok]
  pts_cam = np.stack([(xs_f - cx0) / fx * zs_f, (ys_f - cy0) / fy * zs_f, zs_f], axis=-1)
  pts_w = pts_cam @ R_c2w.T + cam_pos[None, :]

  mask = np.ones(len(pts_w), dtype=bool)
  for _ in range(4):
    sub = pts_w[mask]
    ctr = sub.mean(axis=0)
    _, _, vt = np.linalg.svd(sub - ctr, full_matrices=False)
    nrm = vt[-1]
    res = np.abs((pts_w - ctr) @ nrm)
    mask = res < np.percentile(res, 55)

  sub = pts_w[mask]
  ctr = sub.mean(axis=0)
  _, _, vt = np.linalg.svd(sub - ctr, full_matrices=False)
  nrm = vt[-1]
  if np.dot(nrm, cam_pos - ctr) < 0.0:
    nrm = -nrm
  nrm = nrm / np.linalg.norm(nrm)
  median_res = float(np.median(np.abs((sub - ctr) @ nrm)))

  def _ray_to_plane(px: float, py: float) -> np.ndarray:
    rc = np.array([(px - cx0) / fx, (py - cy0) / fy, 1.0], dtype=np.float32)
    rw = R_c2w @ rc
    t = float(np.dot(ctr - cam_pos, nrm) / np.dot(rw, nrm))
    return (cam_pos + t * rw).astype(np.float32)

  p_origin = _ray_to_plane(*place_xy)
  p_target = _ray_to_plane(*target_xy)
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
      "median_res": median_res,
  }


def compute_opencv_rays(
    cam_pos: jax.Array,   # (3,)
    R_c2w: jax.Array,     # (3, 3) OpenCV columns: (right, down, forward)
    K: jax.Array,         # (3, 3)
    height: int,
    width: int,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
  """Generate `(H*W, 3)` world rays matching exact OpenCV `(viewmat, K)` of 3DGS."""
  xs = jnp.arange(width, dtype=jnp.float32) + 0.5
  ys = jnp.arange(height, dtype=jnp.float32) + 0.5
  gx, gy = jnp.meshgrid(xs, ys, indexing="xy")
  fx, fy, cx0, cy0 = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
  dx = (gx - cx0) / fx
  dy = (gy - cy0) / fy
  dz = jnp.ones_like(dx)
  dirs_cam = jnp.stack([dx, dy, dz], axis=-1).reshape(-1, 3)
  dirs_w = dirs_cam @ R_c2w.T
  norms = jnp.maximum(jnp.linalg.norm(dirs_w, axis=-1, keepdims=True), 1e-12)
  ray_dir = dirs_w / norms
  # Camera +Z cosine (`cos_z = 1 / ||dirs_cam||`) so t_hit * cos_z == OpenCV Z depth!
  cos_z = (1.0 / norms[:, 0]).astype(jnp.float32)
  ray_orig = jnp.broadcast_to(cam_pos[None, :], ray_dir.shape)
  return ray_orig, ray_dir, cos_z


def _ray_one_geom_pallas(
    gt: int,
    px, py, pz,
    m00, m01, m02,
    m10, m11, m12,
    m20, m21, m22,
    sx, sy, sz,
    rb_sq,
    rox, roy, roz,
    rdx, rdy, rdz,
    compute_normal: bool = True,
):
  """Branchless SoA primitive intersection inside Mosaic Pallas (8, 512) VMEM."""
  dx = rox - px
  dy = roy - py
  dz = roz - pz

  if gt != GEOM_PLANE:
    proj = dx * rdx + dy * rdy + dz * rdz
    d2_perp = (dx * dx + dy * dy + dz * dz) - proj * proj
    sphere_hit = d2_perp <= rb_sq
  else:
    sphere_hit = True

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
    tc1 = (-b - sd) * inv_a
    tc2 = (-b + sd) * inv_a
    zc1 = lz + tc1 * vz
    zc2 = lz + tc2 * vz
    ok_c1 = sphere_hit & has_a & (det >= MJ_MINVAL) & (tc1 >= 0.0) & (jnp.abs(zc1) <= sy)
    ok_c2 = sphere_hit & has_a & (det >= MJ_MINVAL) & (tc2 >= 0.0) & (jnp.abs(zc2) <= sy)
    dc1 = jnp.where(ok_c1, tc1, MJ_MAXVAL)
    dc2 = jnp.where(ok_c2, tc2, MJ_MAXVAL)
    use_c1 = dc1 <= dc2
    d_cyl = jnp.where(use_c1, dc1, dc2)
    zc = jnp.where(use_c1, zc1, zc2)

    dzp = lz - sy
    dzm = lz + sy
    aa = vx * vx + vy * vy + vz * vz
    inv_aa = jnp.where(aa > MJ_MINVAL, 1.0 / jnp.where(aa > MJ_MINVAL, aa, 1.0), 0.0)
    bp = vx * lx + vy * ly + vz * dzp
    cp = lx * lx + ly * ly + dzp * dzp - sx * sx
    detp = bp * bp - aa * cp
    sdp = jnp.sqrt(jnp.maximum(detp, 0.0))
    tp = jnp.where((-bp - sdp) * inv_aa >= 0.0, (-bp - sdp) * inv_aa, (-bp + sdp) * inv_aa)
    dp = jnp.where(sphere_hit & (aa > MJ_MINVAL) & (detp >= MJ_MINVAL) & (tp >= 0.0), tp, MJ_MAXVAL)

    bm = vx * lx + vy * ly + vz * dzm
    cm = lx * lx + ly * ly + dzm * dzm - sx * sx
    detm = bm * bm - aa * cm
    sdm = jnp.sqrt(jnp.maximum(detm, 0.0))
    tm = jnp.where((-bm - sdm) * inv_aa >= 0.0, (-bm - sdm) * inv_aa, (-bm + sdm) * inv_aa)
    dm = jnp.where(sphere_hit & (aa > MJ_MINVAL) & (detm >= MJ_MINVAL) & (tm >= 0.0), tm, MJ_MAXVAL)

    d_best = jnp.minimum(d_cyl, jnp.minimum(dp, dm))
    if not compute_normal:
      return d_best, 0.0, 0.0, 0.0
    is_cyl = d_cyl <= jnp.minimum(dp, dm)
    is_p = (~is_cyl) & (dp <= dm)
    hx = lx + d_best * vx
    hy = ly + d_best * vy
    hz = lz + d_best * vz
    lnx = hx
    lny = hy
    lnz = jnp.where(is_cyl, 0.0, jnp.where(is_p, hz - sy, hz + sy))
    nl = jnp.sqrt(jnp.maximum(lnx * lnx + lny * lny + lnz * lnz, 1e-12))
    lnx, lny, lnz = lnx / nl, lny / nl, lnz / nl
    return (
        d_best,
        m00 * lnx + m01 * lny + m02 * lnz,
        m10 * lnx + m11 * lny + m12 * lnz,
        m20 * lnx + m21 * lny + m22 * lnz,
    )

  if gt == GEOM_CYLINDER:
    a = vx * vx + vy * vy
    b = lx * vx + ly * vy
    c = lx * lx + ly * ly - sx * sx
    det = b * b - a * c
    has_a = jnp.abs(a) > MJ_MINVAL
    sd = jnp.sqrt(jnp.maximum(det, 0.0))
    inv_a = jnp.where(has_a, 1.0 / jnp.where(has_a, a, 1.0), 0.0)
    tc1 = (-b - sd) * inv_a
    tc2 = (-b + sd) * inv_a
    zc1 = lz + tc1 * vz
    zc2 = lz + tc2 * vz
    dc1 = jnp.where(sphere_hit & has_a & (det >= MJ_MINVAL) & (tc1 >= 0.0) & (jnp.abs(zc1) <= sy), tc1, MJ_MAXVAL)
    dc2 = jnp.where(sphere_hit & has_a & (det >= MJ_MINVAL) & (tc2 >= 0.0) & (jnp.abs(zc2) <= sy), tc2, MJ_MAXVAL)
    d_cyl = jnp.minimum(dc1, dc2)

    has_vz = jnp.abs(vz) > MJ_MINVAL
    inv_vz = jnp.where(has_vz, 1.0 / jnp.where(has_vz, vz, 1.0), 0.0)
    tp = (sy - lz) * inv_vz
    tm = (-sy - lz) * inv_vz
    xp, yp = lx + tp * vx, ly + tp * vy
    xm, ym = lx + tm * vx, ly + tm * vy
    dp = jnp.where(sphere_hit & has_vz & (tp >= 0.0) & (xp * xp + yp * yp <= sx * sx), tp, MJ_MAXVAL)
    dm = jnp.where(sphere_hit & has_vz & (tm >= 0.0) & (xm * xm + ym * ym <= sx * sx), tm, MJ_MAXVAL)

    d_best = jnp.minimum(d_cyl, jnp.minimum(dp, dm))
    if not compute_normal:
      return d_best, 0.0, 0.0, 0.0
    is_cyl = d_cyl <= jnp.minimum(dp, dm)
    is_p = (~is_cyl) & (dp <= dm)
    inv_sx = 1.0 / jnp.maximum(sx, 1e-12)
    lnx = jnp.where(is_cyl, (lx + d_best * vx) * inv_sx, 0.0)
    lny = jnp.where(is_cyl, (ly + d_best * vy) * inv_sx, 0.0)
    lnz = jnp.where(is_cyl, 0.0, jnp.where(is_p, 1.0, -1.0))
    return (
        d_best,
        m00 * lnx + m01 * lny + m02 * lnz,
        m10 * lnx + m11 * lny + m12 * lnz,
        m20 * lnx + m21 * lny + m22 * lnz,
    )

  if gt == GEOM_BOX:
    inv_vx = jnp.where(jnp.abs(vx) > MJ_MINVAL, 1.0 / jnp.where(jnp.abs(vx) > MJ_MINVAL, vx, 1.0), 1e12)
    inv_vy = jnp.where(jnp.abs(vy) > MJ_MINVAL, 1.0 / jnp.where(jnp.abs(vy) > MJ_MINVAL, vy, 1.0), 1e12)
    inv_vz = jnp.where(jnp.abs(vz) > MJ_MINVAL, 1.0 / jnp.where(jnp.abs(vz) > MJ_MINVAL, vz, 1e12) , 1e12)
    tx1, tx2 = (-sx - lx) * inv_vx, (sx - lx) * inv_vx
    ty1, ty2 = (-sy - ly) * inv_vy, (sy - ly) * inv_vy
    tz1, tz2 = (-sz - lz) * inv_vz, (sz - lz) * inv_vz
    tmin = jnp.maximum(jnp.minimum(tx1, tx2), jnp.maximum(jnp.minimum(ty1, ty2), jnp.minimum(tz1, tz2)))
    tmax = jnp.minimum(jnp.maximum(tx1, tx2), jnp.minimum(jnp.maximum(ty1, ty2), jnp.maximum(tz1, tz2)))
    t_hit = jnp.where(tmin >= 0.0, tmin, tmax)
    ok = sphere_hit & (tmax >= jnp.maximum(tmin, 0.0)) & (t_hit >= 0.0)
    cand_d = jnp.where(ok, t_hit, MJ_MAXVAL)
    if not compute_normal:
      return cand_d, 0.0, 0.0, 0.0
    hx = jnp.abs(lx + t_hit * vx) / jnp.maximum(sx, 1e-12)
    hy = jnp.abs(ly + t_hit * vy) / jnp.maximum(sy, 1e-12)
    hz = jnp.abs(lz + t_hit * vz) / jnp.maximum(sz, 1e-12)
    use_x = (hx >= hy) & (hx >= hz)
    use_y = (~use_x) & (hy >= hz)
    lnx = jnp.where(use_x, jnp.where(lx + t_hit * vx >= 0.0, 1.0, -1.0), 0.0)
    lny = jnp.where(use_y, jnp.where(ly + t_hit * vy >= 0.0, 1.0, -1.0), 0.0)
    lnz = jnp.where((~use_x) & (~use_y), jnp.where(lz + t_hit * vz >= 0.0, 1.0, -1.0), 0.0)
    return (
        cand_d,
        m00 * lnx + m01 * lny + m02 * lnz,
        m10 * lnx + m11 * lny + m12 * lnz,
        m20 * lnx + m21 * lny + m22 * lnz,
    )

  return jnp.full_like(rox, MJ_MAXVAL), 0.0, 0.0, 0.0


def pallas_raycast_mjx_table_strip(
    spec: RenderSceneSpec,
    geom_xpos: jax.Array,
    geom_xmat: jax.Array,
    geom_size: jax.Array,
    geom_rgba: jax.Array,
    sun_dir: jax.Array,       # (3,) toward light
    table_normal: jax.Array,  # (3,) unit upward table normal
    table_d: jax.Array,       # scalar dot(p_origin, table_normal)
    ray_orig: jax.Array,      # (R_sub, 3)
    ray_dir: jax.Array,       # (R_sub, 3)
    cos_z: jax.Array,         # (R_sub,)
):
  """4-chip Mosaic Pallas exact MJX raycaster + arbitrary-plane table shadow-catcher."""
  del geom_rgba
  gids = [g for g in range(spec.ngeom) if spec.geom_rgba[g, 3] > 0.01]
  G = len(gids)
  G_pad = ((G + 127) // 128) * 128

  R_sub = ray_orig.shape[0]
  BLOCK_B, BLOCK_R = 8, 512
  BLK = BLOCK_B * BLOCK_R
  n_r_blk = (R_sub + BLK - 1) // BLK
  R_pad = n_r_blk * BLK
  R_per_row = n_r_blk * BLOCK_R

  def _pack_1d(arr, val=0.0):
    if R_pad > R_sub:
      arr = jnp.pad(arr, (0, R_pad - R_sub), constant_values=val)
    return arr.reshape(BLOCK_B, R_per_row)

  ray_blk_in = jnp.concatenate([
      _pack_1d(ray_orig[:, 0]),
      _pack_1d(ray_orig[:, 1]),
      _pack_1d(ray_orig[:, 2]),
      _pack_1d(ray_dir[:, 0]),
      _pack_1d(ray_dir[:, 1]),
      _pack_1d(ray_dir[:, 2], 1.0),
      _pack_1d(cos_z, 1.0),
  ], axis=0)  # (56, R_per_row)

  idx_arr = np.asarray(gids, dtype=np.int32)
  gp = geom_xpos[idx_arr]
  gm = geom_xmat[idx_arr]
  gsz = geom_size[idx_arr]
  rb_sq_l = []
  for g in gids:
    gt = int(spec.geom_type[g])
    sx, sy, sz = [float(x) for x in spec.geom_size[g]]
    if gt == GEOM_SPHERE:
      rb_sq_l.append((sx + 1e-4) ** 2)
    elif gt == GEOM_CAPSULE:
      rb_sq_l.append((sx + sy + 1e-4) ** 2)
    elif gt == GEOM_CYLINDER:
      rb_sq_l.append(sx * sx + sy * sy + 1e-3)
    else:
      rb_sq_l.append(sx * sx + sy * sy + sz * sz + 1e-3)
  rb_arr = jnp.asarray(rb_sq_l, dtype=jnp.float32)

  gf_raw = jnp.stack([
      gp[:, 0], gp[:, 1], gp[:, 2],
      gm[:, 0, 0], gm[:, 0, 1], gm[:, 0, 2],
      gm[:, 1, 0], gm[:, 1, 1], gm[:, 1, 2],
      gm[:, 2, 0], gm[:, 2, 1], gm[:, 2, 2],
      gsz[:, 0], gsz[:, 1], gsz[:, 2],
      rb_arr,
  ], axis=0)
  gf_dev = jnp.pad(gf_raw, ((0, 0), (0, G_pad - G))) if G_pad > G else gf_raw

  s_blk = jnp.zeros((8, 128), dtype=jnp.float32)
  s_blk = s_blk.at[0, :].set(sun_dir[0])
  s_blk = s_blk.at[1, :].set(sun_dir[1])
  s_blk = s_blk.at[2, :].set(sun_dir[2])
  s_blk = s_blk.at[3, :].set(table_normal[0])
  s_blk = s_blk.at[4, :].set(table_normal[1])
  s_blk = s_blk.at[5, :].set(table_normal[2])
  s_blk = s_blk.at[6, :].set(table_d)

  g_types_static = tuple(int(spec.geom_type[g]) for g in gids)
  gids_static = tuple(gids)
  g_rgba_static = tuple(tuple(float(c) for c in spec.geom_rgba[g, :3]) for g in gids)
  mat_groups: Dict[Tuple[float, float, float], list[int]] = {}
  for gid_val, col in zip(gids_static, g_rgba_static):
    mat_groups.setdefault(col, []).append(gid_val)

  zfar_f = float(spec.zfar)

  def _kernel(r_ref, gf_ref, s_ref, out_ref):
    rb = r_ref[...]
    gf = gf_ref[...]
    sb = s_ref[...]

    rox, roy, roz = rb[0:8, :], rb[8:16, :], rb[16:24, :]
    rdx, rdy, rdz = rb[24:32, :], rb[32:40, :], rb[40:48, :]
    cz_b = rb[48:56, :]
    sdx, sdy, sdz = sb[0:1, 0:1], sb[1:2, 0:1], sb[2:3, 0:1]
    tnx, tny, tnz = sb[3:4, 0:1], sb[4:5, 0:1], sb[5:6, 0:1]
    td = sb[6:7, 0:1]

    best_d = jnp.full((8, 512), zfar_f, dtype=jnp.float32)
    best_nx = jnp.zeros((8, 512), dtype=jnp.float32)
    best_ny = jnp.zeros((8, 512), dtype=jnp.float32)
    best_nz = jnp.zeros((8, 512), dtype=jnp.float32)
    best_gid = jnp.full((8, 512), -1, dtype=jnp.int32)

    for k, (gt, gid_val) in enumerate(zip(g_types_static, gids_static)):
      cd, cnx, cny, cnz = _ray_one_geom_pallas(
          gt,
          gf[0:1, k:k+1], gf[1:2, k:k+1], gf[2:3, k:k+1],
          gf[3:4, k:k+1], gf[4:5, k:k+1], gf[5:6, k:k+1],
          gf[6:7, k:k+1], gf[7:8, k:k+1], gf[8:9, k:k+1],
          gf[9:10, k:k+1], gf[10:11, k:k+1], gf[11:12, k:k+1],
          gf[12:13, k:k+1], gf[13:14, k:k+1], gf[14:15, k:k+1],
          gf[15:16, k:k+1],
          rox, roy, roz, rdx, rdy, rdz,
          compute_normal=True,
      )
      imp = cd < best_d
      best_d = jnp.where(imp, cd, best_d)
      best_gid = jnp.where(imp, jnp.int32(gid_val), best_gid)
      best_nx = jnp.where(imp, cnx, best_nx)
      best_ny = jnp.where(imp, cny, best_ny)
      best_nz = jnp.where(imp, cnz, best_nz)

    hit = best_gid >= 0
    d_safe = jnp.where(hit, best_d, 0.0)
    hx = rox + d_safe * rdx
    hy = roy + d_safe * rdy
    hz = roz + d_safe * rdz

    alb_r = jnp.zeros((8, 512), dtype=jnp.float32)
    alb_g = jnp.zeros((8, 512), dtype=jnp.float32)
    alb_b = jnp.zeros((8, 512), dtype=jnp.float32)
    for (rk, gk, bk), glist in mat_groups.items():
      is_m = best_gid == glist[0]
      for go in glist[1:]:
        is_m = is_m | (best_gid == go)
      alb_r = jnp.where(is_m, rk, alb_r)
      alb_g = jnp.where(is_m, gk, alb_g)
      alb_b = jnp.where(is_m, bk, alb_b)

    # Arbitrary-plane table intersection for real-world cast shadows
    denom_t = rdx * tnx + rdy * tny + rdz * tnz
    front_tbl = denom_t < -1e-4
    t_tbl = jnp.where(
        front_tbl,
        (td - (rox * tnx + roy * tny + roz * tnz)) / jnp.where(front_tbl, denom_t, -1.0),
        -1.0,
    )
    tbl_ok = (~hit) & front_tbl & (t_tbl > 0.0) & (t_tbl < 25.0)
    tx = rox + jnp.where(tbl_ok, t_tbl, 0.0) * rdx + 2e-3 * tnx
    ty = roy + jnp.where(tbl_ok, t_tbl, 0.0) * rdy + 2e-3 * tny
    tz = roz + jnp.where(tbl_ok, t_tbl, 0.0) * rdz + 2e-3 * tnz

    sh_ox = jnp.where(hit, hx + best_nx * 2e-3, tx)
    sh_oy = jnp.where(hit, hy + best_ny * 2e-3, ty)
    sh_oz = jnp.where(hit, hz + best_nz * 2e-3, tz)

    occ_bool = jnp.zeros((8, 512), dtype=jnp.bool_)
    for k, gt in enumerate(g_types_static):
      scd, _, _, _ = _ray_one_geom_pallas(
          gt,
          gf[0:1, k:k+1], gf[1:2, k:k+1], gf[2:3, k:k+1],
          gf[3:4, k:k+1], gf[4:5, k:k+1], gf[5:6, k:k+1],
          gf[6:7, k:k+1], gf[7:8, k:k+1], gf[8:9, k:k+1],
          gf[9:10, k:k+1], gf[10:11, k:k+1], gf[11:12, k:k+1],
          gf[12:13, k:k+1], gf[13:14, k:k+1], gf[14:15, k:k+1],
          gf[15:16, k:k+1],
          sh_ox, sh_oy, sh_oz, sdx, sdy, sdz,
          compute_normal=False,
      )
      occ_bool = occ_bool | (scd < 20.0)

    robot_sh = jnp.where(hit & occ_bool, 0.0, 1.0)
    tbl_sh_mask = jnp.where(tbl_ok & occ_bool, 1.0, 0.0)
    z_depth = jnp.where(hit, best_d * cz_b, -1.0)
    tbl_z = jnp.where(tbl_ok, t_tbl * cz_b, -1.0)

    out_ref[...] = jnp.concatenate([
        alb_r, alb_g, alb_b,
        jnp.where(hit, best_nx, 0.0),
        jnp.where(hit, best_ny, 0.0),
        jnp.where(hit, best_nz, 0.0),
        z_depth,
        robot_sh,
        tbl_sh_mask,
        tbl_z,
    ], axis=0)

  out_raw = pl.pallas_call(
      _kernel,
      out_shape=jax.ShapeDtypeStruct((80, R_per_row), jnp.float32),
      grid=(n_r_blk,),
      in_specs=[
          pl.BlockSpec((56, BLOCK_R), lambda ri: (0, ri)),
          pl.BlockSpec((16, G_pad), lambda ri: (0, 0)),
          pl.BlockSpec((8, 128), lambda ri: (0, 0)),
      ],
      out_specs=pl.BlockSpec((80, BLOCK_R), lambda ri: (0, ri)),
      compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
  )(ray_blk_in, gf_dev, s_blk)

  def _unpack(row_idx):
    return out_raw[row_idx * 8 : (row_idx + 1) * 8, :].reshape(-1)[:R_sub]

  alb = jnp.stack([_unpack(0), _unpack(1), _unpack(2)], axis=-1)
  nrm = jnp.stack([_unpack(3), _unpack(4), _unpack(5)], axis=-1)
  z_depth = _unpack(6)
  robot_sh = _unpack(7)
  tbl_sh_mask = _unpack(8)
  tbl_z = _unpack(9)
  return alb, nrm, z_depth, robot_sh, tbl_sh_mask, tbl_z, -ray_dir
