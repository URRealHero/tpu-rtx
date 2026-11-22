"""Task B: End-to-End Differentiable Physics + Differentiable PBR Rendering (Pixel-to-Torque Visual Servoing).

Couples `mjx.step` (7-DOF torque-controlled robotic arm, iterations=1) with a
zero-host-callback, smooth-visibility differentiable GGX PBR + soft-shadow renderer
inside ONE `jax.value_and_grad` XLA graph on TPU v5p (`host_callbacks == 0`).

Optimizes multi-step joint control torques `ctrl[0..T-1]` purely from rendered
RGB pixel L2 loss against a target pose image (zero state supervision!).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np

from tpu_mjx_render.camera import compute_camera_rays
from tpu_mjx_render.primitives import GEOM_BOX, GEOM_CAPSULE, GEOM_CYLINDER, GEOM_PLANE, GEOM_SPHERE
from tpu_mjx_render.renderer import RenderSceneSpec
from tpu_mjx_render.roofline import inspect_hlo_ops


def _panda_torque_arm_xml() -> str:
  """7-DOF Franka-kinematics torque-controlled studio arm with rich analytic primitives."""
  return """<mujoco model="panda_torque_diff">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.012" iterations="1" ls_iterations="4" integrator="implicitfast" gravity="0 0 -9.81"/>
  <visual>
    <headlight ambient="0.06 0.065 0.075" diffuse="0.14 0.14 0.15" specular="0.06 0.06 0.06"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.15 0.21 0.30" rgb2="0.04 0.055 0.09" width="128" height="128"/>
    <texture name="studio_grid" type="2d" builtin="checker" mark="edge"
      rgb1="0.14 0.17 0.22" rgb2="0.095 0.115 0.15" markrgb="0.28 0.34 0.44" width="128" height="128"/>
    <material name="studio_grid" texture="studio_grid" texuniform="true" texrepeat="4 4" specular="0.25" shininess="0.35"/>
    <material name="arm_white" rgba="0.90 0.92 0.95 1" specular="0.55" shininess="0.50"/>
    <material name="arm_titanium" rgba="0.72 0.76 0.82 1" specular="0.62" shininess="0.55"/>
    <material name="arm_dark" rgba="0.15 0.17 0.21 1" specular="0.55" shininess="0.48"/>
    <material name="arm_orange" rgba="0.92 0.38 0.10 1" specular="0.58" shininess="0.52"/>
    <material name="ee_cyan" rgba="0.10 0.74 0.94 1" specular="0.65" shininess="0.60"/>
  </asset>
  <default>
    <joint armature="0.20" damping="1.8" axis="0 0 1" limited="false"/>
    <geom contype="0" conaffinity="0"/>
  </default>
  <worldbody>
    <light name="key" pos="1.9 -1.8 2.8" dir="-0.45 0.45 -1.0" directional="false" castshadow="true"
      diffuse="0.98 0.91 0.82" specular="0.52 0.48 0.40" ambient="0.05 0.05 0.06"/>
    <light name="rim" pos="-1.8 1.6 1.8" dir="0.5 -0.5 -0.45" directional="false" castshadow="false"
      diffuse="0.32 0.56 0.92" specular="0.48 0.68 0.96" ambient="0.0 0.0 0.0"/>
    <light name="fill" pos="-1.5 -1.4 1.6" dir="0.4 0.4 -0.6" directional="false" castshadow="false"
      diffuse="0.34 0.36 0.42" specular="0.12 0.12 0.14" ambient="0.0 0.0 0.0"/>
    <camera name="servo_cam" pos="1.15 -1.05 0.72" xyaxes="0.675 0.738 0 -0.221 0.202 0.954" fovy="38"/>
    <geom name="floor" type="plane" pos="0 0 0" size="5 5 0.05" material="studio_grid"/>
    <geom name="pedestal" type="cylinder" pos="0 0 0.02" size="0.13 0.02" material="arm_dark"/>
    <body name="link0" pos="0 0 0.04">
      <geom name="base_col" type="cylinder" pos="0 0 0.06" size="0.075 0.06" material="arm_white"/>
      <body name="link1" pos="0 0 0.16">
        <inertial mass="3.0" pos="0 0 -0.05" diaginertia="0.02 0.02 0.02"/>
        <joint name="joint1"/>
        <geom name="j1_cap" type="sphere" pos="0 0 0" size="0.068" material="arm_dark"/>
        <geom name="l1_seg" type="capsule" fromto="0 0 0 0 0 0.16" size="0.056" material="arm_white"/>
        <body name="link2" pos="0 0 0.17" quat="1 -1 0 0">
          <inertial mass="3.0" pos="0 -0.15 0" diaginertia="0.02 0.02 0.02"/>
          <joint name="joint2"/>
          <geom name="j2_cap" type="sphere" pos="0 0 0" size="0.062" material="arm_dark"/>
          <geom name="l2_seg" type="capsule" fromto="0 0 0 0 -0.28 0" size="0.052" material="arm_white"/>
          <body name="link3" pos="0 -0.316 0" quat="1 1 0 0">
            <inertial mass="2.2" pos="0 0 -0.05" diaginertia="0.015 0.015 0.015"/>
            <joint name="joint3"/>
            <geom name="j3_cap" type="sphere" pos="0 0 0" size="0.056" material="arm_orange"/>
            <geom name="l3_seg" type="capsule" fromto="0 0 -0.04 0.06 0 0" size="0.048" material="arm_titanium"/>
            <body name="link4" pos="0.0825 0 0" quat="1 1 0 0">
              <inertial mass="2.0" pos="-0.04 0.14 0" diaginertia="0.012 0.012 0.012"/>
              <joint name="joint4"/>
              <geom name="j4_cap" type="sphere" pos="0 0 0" size="0.054" material="arm_dark"/>
              <geom name="l4_seg" type="capsule" fromto="0 0 0 -0.07 0.28 0" size="0.046" material="arm_orange"/>
              <body name="link5" pos="-0.0825 0.34 0" quat="1 -1 0 0">
                <inertial mass="1.5" pos="0 0 -0.08" diaginertia="0.008 0.008 0.008"/>
                <joint name="joint5"/>
                <geom name="j5_cap" type="cylinder" pos="0 0 0" size="0.046 0.028" material="arm_dark"/>
                <geom name="l5_seg" type="capsule" fromto="0 0 -0.18 0 0 0.02" size="0.040" material="arm_white"/>
                <body name="link6" quat="1 1 0 0">
                  <inertial mass="1.0" pos="0.04 0 0" diaginertia="0.005 0.005 0.005"/>
                  <joint name="joint6"/>
                  <geom name="j6_cap" type="sphere" pos="0 0 0" size="0.044" material="arm_orange"/>
                  <geom name="l6_seg" type="capsule" fromto="0 0 0 0.085 0 0" size="0.036" material="arm_titanium"/>
                  <body name="link7" pos="0.088 0 0" quat="1 1 0 0">
                    <inertial mass="0.8" pos="0 0 0.05" diaginertia="0.003 0.003 0.003"/>
                    <joint name="joint7"/>
                    <geom name="flange" type="cylinder" pos="0 0 0.04" size="0.038 0.032" material="arm_dark"/>
                    <geom name="hand_body" type="box" pos="0 0 0.095" size="0.024 0.088 0.024" material="arm_titanium"/>
                    <geom name="finger_l" type="box" pos="0 -0.042 0.150" size="0.014 0.014 0.038" material="ee_cyan"/>
                    <geom name="finger_r" type="box" pos="0 0.042 0.150" size="0.014 0.014 0.038" material="ee_cyan"/>
                    <geom name="ee_beacon" type="sphere" pos="0 0 0.135" size="0.024" material="ee_cyan"/>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="act1" joint="joint1" gear="45" ctrlrange="-1.5 1.5"/>
    <motor name="act2" joint="joint2" gear="45" ctrlrange="-1.5 1.5"/>
    <motor name="act3" joint="joint3" gear="35" ctrlrange="-1.5 1.5"/>
    <motor name="act4" joint="joint4" gear="35" ctrlrange="-1.5 1.5"/>
    <motor name="act5" joint="joint5" gear="20" ctrlrange="-1.5 1.5"/>
    <motor name="act6" joint="joint6" gear="20" ctrlrange="-1.5 1.5"/>
    <motor name="act7" joint="joint7" gear="14" ctrlrange="-1.5 1.5"/>
  </actuator>
</mujoco>
"""


def _safe_norm(v: jax.Array, eps: float = 1e-6) -> jax.Array:
  return jnp.sqrt(jnp.maximum(jnp.sum(v * v, axis=-1, keepdims=True), eps))


def _safe_unit(v: jax.Array, eps: float = 1e-6) -> jax.Array:
  return v / _safe_norm(v, eps=eps)


def _diff_prim_ray(
    gt: int,
    pos: jax.Array,   # (3,)
    mat: jax.Array,   # (3, 3)
    size: jax.Array,  # (3,)
    ro: jax.Array,    # (R, 3)
    rd: jax.Array,    # (R, 3)
) -> Tuple[jax.Array, jax.Array, jax.Array]:
  """Differentiable ray-primitive signed clearance `phi`, effective depth `t_eff`, and world normal `nrm`.

  Guarantees smooth, NaN-free reverse-mode gradients across BOTH interior surfaces
  and grazing silhouette boundaries (`phi <= 0` inside, `phi > 0` outside).
  """
  lp = jnp.einsum("ji,rj->ri", mat, ro - pos[None, :], precision=jax.lax.Precision.HIGHEST)
  ld = jnp.einsum("ji,rj->ri", mat, rd, precision=jax.lax.Precision.HIGHEST)

  if gt == GEOM_SPHERE:
    r = size[0]
    tc = jnp.maximum(0.0, -jnp.sum(lp * ld, axis=-1))
    pc = lp + tc[:, None] * ld
    dist_c = jnp.sqrt(jnp.sum(pc * pc, axis=-1) + 1e-6)
    phi = dist_c - r
    dz = jnp.sqrt(jnp.maximum(r * r - dist_c * dist_c, 1e-5))
    t_eff = jnp.maximum(0.0, tc - dz)
    hp = lp + t_eff[:, None] * ld
    w_in = jax.nn.sigmoid((r - dist_c) / 0.004)[:, None]
    lnrm = _safe_unit(w_in * hp + (1.0 - w_in) * pc)
  elif gt == GEOM_CAPSULE:
    r = size[0]
    hh = size[1]
    dot_p = jnp.sum(lp * ld, axis=-1)
    denom = jnp.maximum(1.0 - ld[:, 2] * ld[:, 2], 1e-4)
    s0 = jnp.clip((ld[:, 2] * dot_p - lp[:, 2]) / denom, -hh, hh)
    tc = jnp.maximum(0.0, -(dot_p - ld[:, 2] * s0))
    sc = jnp.clip(lp[:, 2] + tc * ld[:, 2], -hh, hh)
    tc = jnp.maximum(0.0, -(dot_p - ld[:, 2] * sc))
    pc = lp + tc[:, None] * ld
    axis_pt = jnp.stack([jnp.zeros_like(sc), jnp.zeros_like(sc), sc], axis=-1)
    diff = pc - axis_pt
    dist_ax = jnp.sqrt(jnp.sum(diff * diff, axis=-1) + 1e-6)
    phi = dist_ax - r
    dz = jnp.sqrt(jnp.maximum(r * r - dist_ax * dist_ax, 1e-5))
    t_eff = jnp.maximum(0.0, tc - dz)
    hp = lp + t_eff[:, None] * ld
    sh = jnp.clip(hp[:, 2], -hh, hh)
    h_axis = jnp.stack([jnp.zeros_like(sh), jnp.zeros_like(sh), sh], axis=-1)
    w_in = jax.nn.sigmoid((r - dist_ax) / 0.004)[:, None]
    lnrm = _safe_unit(w_in * (hp - h_axis) + (1.0 - w_in) * diff)
  elif gt == GEOM_CYLINDER:
    r = size[0]
    hh = size[1]
    dot_p = jnp.sum(lp * ld, axis=-1)
    denom = jnp.maximum(1.0 - ld[:, 2] * ld[:, 2], 1e-4)
    s0 = jnp.clip((ld[:, 2] * dot_p - lp[:, 2]) / denom, -hh, hh)
    tc = jnp.maximum(0.0, -(dot_p - ld[:, 2] * s0))
    pc = lp + tc[:, None] * ld
    q_r = jnp.sqrt(pc[:, 0] * pc[:, 0] + pc[:, 1] * pc[:, 1] + 1e-6) - r
    q_z = jnp.abs(pc[:, 2]) - hh
    out_d = jnp.sqrt(jnp.maximum(q_r, 0.0) ** 2 + jnp.maximum(q_z, 0.0) ** 2 + 1e-6)
    in_d = jnp.minimum(jnp.maximum(q_r, q_z), 0.0)
    phi = out_d + in_d
    # Approximate front surface depth along ray
    t_eff = jnp.maximum(0.0, tc - jnp.maximum(-phi, 0.0))
    hp = lp + t_eff[:, None] * ld
    w_cap = jax.nn.sigmoid((jnp.abs(hp[:, 2]) / jnp.maximum(hh, 1e-4) - jnp.sqrt(hp[:, 0] ** 2 + hp[:, 1] ** 2 + 1e-6) / jnp.maximum(r, 1e-4)) * 12.0)
    n_side = _safe_unit(jnp.stack([hp[:, 0], hp[:, 1], jnp.zeros_like(hp[:, 2])], axis=-1))
    n_cap = jnp.stack([jnp.zeros_like(hp[:, 2]), jnp.zeros_like(hp[:, 2]), jnp.sign(hp[:, 2])], axis=-1)
    lnrm = _safe_unit(w_cap[:, None] * n_cap + (1.0 - w_cap[:, None]) * n_side)
  else:  # GEOM_BOX
    # Closest point on ray to box center & exact slab hit depth
    ok_d = jnp.abs(ld) > 1e-6
    inv_d = jnp.where(ok_d, 1.0 / jnp.where(ok_d, ld, 1.0), 1e6)
    t0 = (-size[None, :] - lp) * inv_d
    t1 = (size[None, :] - lp) * inv_d
    tmin = jnp.max(jnp.minimum(t0, t1), axis=-1)
    tmax = jnp.min(jnp.maximum(t0, t1), axis=-1)
    t_mid = jnp.maximum(0.0, 0.5 * (tmin + tmax))
    t_c = jnp.maximum(0.0, -jnp.sum(lp * ld, axis=-1))
    hit_box = tmax >= jnp.maximum(tmin, 0.0)
    t_sample = jnp.where(hit_box, t_mid, t_c)
    pm = lp + t_sample[:, None] * ld
    q = jnp.abs(pm) - size[None, :]
    out_d = jnp.sqrt(jnp.sum(jnp.maximum(q, 0.0) ** 2, axis=-1) + 1e-6)
    in_d = jnp.minimum(jnp.max(q, axis=-1), 0.0)
    phi = out_d + in_d
    t_eff = jnp.where(hit_box, jnp.maximum(tmin, 0.0), t_c)
    hp = lp + t_eff[:, None] * ld
    # Smooth box normal via softmax over normalized face coordinates
    ratio = jnp.abs(hp) / jnp.maximum(size[None, :], 1e-4)
    fw = jax.nn.softmax(ratio * 24.0, axis=-1)
    lnrm = _safe_unit(fw * jnp.tanh(hp * 80.0))

  wnrm = _safe_unit(jnp.einsum("ij,rj->ri", mat, lnrm, precision=jax.lax.Precision.HIGHEST))
  return phi, t_eff, wnrm


def render_diff_pbr(
    spec: RenderSceneSpec,
    mjx_data: Any,
    sigma: float = 0.010,
    tau_z: float = 0.015,
) -> jax.Array:
  """Pure-JAX smooth-visibility differentiable GGX PBR + soft shadow renderer (zero host callbacks)."""
  H, W = int(spec.height), int(spec.width)
  cam_pos = mjx_data.cam_xpos[int(spec.cam_id)]
  cam_mat = mjx_data.cam_xmat[int(spec.cam_id)]
  ro, rd, _ = compute_camera_rays(
      cam_pos=cam_pos,
      cam_mat=cam_mat,
      fovy_deg=spec.cam_fovy,
      height=H,
      width=W,
      projection=spec.cam_projection,
      sensorsize=spec.cam_sensorsize,
      intrinsic=spec.cam_intrinsic,
      znear=spec.znear,
  )
  R = H * W
  vdir = -rd

  # Extract lights dynamically from mjx_data
  l_bid = np.asarray(spec.light_bodyid, dtype=np.int32)
  light_xpos = mjx_data.xpos[l_bid] + jnp.einsum("nij,nj->ni", mjx_data.xmat[l_bid], jnp.asarray(spec.light_pos, jnp.float32))
  light_xdir = jnp.einsum("nij,nj->ni", mjx_data.xmat[l_bid], jnp.asarray(spec.light_dir, jnp.float32))

  fg_gids = [g for g in spec.enabled_geom_ids if int(spec.geom_type[g]) != GEOM_PLANE]
  geom_xpos = mjx_data.geom_xpos
  geom_xmat = mjx_data.geom_xmat

  # 1. Background: Floor plane (z=0) + Skybox + Differentiable Soft Floor Shadow
  sky_t = jnp.clip(0.5 * (rd[:, 2] + 1.0), 0.0, 1.0)[:, None]
  sky_rgb = sky_t * jnp.asarray(spec.sky_rgb1, jnp.float32)[None, :] + (1.0 - sky_t) * jnp.asarray(spec.sky_rgb2, jnp.float32)[None, :]

  fl_mask = jax.nn.sigmoid((-rd[:, 2] - 0.002) * 120.0)
  t_fl = jnp.clip(-ro[:, 2] / jnp.minimum(rd[:, 2], -1e-4), 0.0, 30.0)
  p_fl = ro + t_fl[:, None] * rd
  fl_nrm = jnp.broadcast_to(jnp.array([[0.0, 0.0, 1.0]], jnp.float32), (R, 3))
  # Smooth anti-aliased checker grid on floor
  rep_x, rep_y = spec.checker_repeat
  sx = jnp.sin(math.pi * p_fl[:, 0] * rep_x)
  sy = jnp.sin(math.pi * p_fl[:, 1] * rep_y)
  chk_w = 0.5 + 0.5 * jnp.tanh(20.0 * sx * sy)
  fl_alb = chk_w[:, None] * jnp.asarray(spec.checker_rgb1, jnp.float32)[None, :] + (1.0 - chk_w[:, None]) * jnp.asarray(spec.checker_rgb2, jnp.float32)[None, :]

  # Soft shadow cast by foreground geoms onto the floor from key light 0
  lp0 = light_xpos[0]
  L_sh0 = _safe_unit(lp0[None, :] - p_fl)
  sh_ro = p_fl + fl_nrm * 2e-3
  no_occ = jnp.ones((R,), dtype=jnp.float32)
  for g in fg_gids:
    gt = int(spec.geom_type[g])
    sz = jnp.asarray(spec.geom_size[g], jnp.float32)
    phi_s, _, _ = _diff_prim_ray(gt, geom_xpos[g], geom_xmat[g], sz, sh_ro, L_sh0)
    a_s = jax.nn.sigmoid(-phi_s / 0.025)
    no_occ = no_occ * (1.0 - a_s)
  fl_vis = 0.14 + 0.86 * no_occ

  def _shade_ggx(pts, nrm, alb, spec_c, shin_c, vis_key):
    rough = jnp.clip(1.0 - 0.72 * shin_c, 0.16, 0.88)
    a = rough * rough
    a2 = a * a
    k_g = ((rough + 1.0) ** 2) / 8.0
    f0 = 0.04 + 0.24 * spec_c
    ndotv = jnp.maximum(1e-3, jnp.sum(nrm * vdir, axis=-1))
    g1_v = ndotv / (ndotv * (1.0 - k_g) + k_g + 1e-6)

    amb_tot = jnp.zeros((R, 3), jnp.float32)
    diff_tot = jnp.zeros((R, 3), jnp.float32)
    spec_tot = jnp.zeros((R, 3), jnp.float32)
    for li in range(int(spec.nlight)):
      if not bool(spec.light_active[li]):
        continue
      l_amb = jnp.asarray(spec.light_ambient[li], jnp.float32)[None, :]
      l_diff = jnp.asarray(spec.light_diffuse[li], jnp.float32)[None, :]
      l_spec = jnp.asarray(spec.light_specular[li], jnp.float32)[None, :]
      amb_tot = amb_tot + l_amb
      if int(spec.light_type[li]) == 1:
        L = jnp.broadcast_to(_safe_unit(-light_xdir[li])[None, :], (R, 3))
      else:
        L = _safe_unit(light_xpos[li][None, :] - pts)
      ndotl = jnp.maximum(0.0, jnp.sum(nrm * L, axis=-1))
      vis = vis_key if li == 0 else 1.0
      H_v = _safe_unit(L + vdir)
      ndoth = jnp.maximum(0.0, jnp.sum(nrm * H_v, axis=-1))
      vdoth = jnp.maximum(0.0, jnp.sum(vdir * H_v, axis=-1))
      denom_d = ndoth * ndoth * (a2 - 1.0) + 1.0
      d_ggx = a2 / (math.pi * denom_d * denom_d + 1e-6)
      f_sch = f0 + (1.0 - f0) * ((1.0 - vdoth) ** 5)
      g1_l = ndotl / (ndotl * (1.0 - k_g) + k_g + 1e-6)
      sp_brdf = (d_ggx * f_sch * g1_l * g1_v) / (4.0 * ndotl * ndotv + 1e-3)
      w = vis * ndotl
      diff_tot = diff_tot + l_diff * (w * (1.0 - f_sch))[:, None]
      spec_tot = spec_tot + l_spec * (spec_c * sp_brdf * w * math.pi)[:, None]

    if spec.headlight_active:
      amb_tot = amb_tot + jnp.asarray(spec.headlight_ambient, jnp.float32)[None, :]
      diff_tot = diff_tot + jnp.asarray(spec.headlight_diffuse, jnp.float32)[None, :] * ndotv[:, None]
    return jnp.clip(alb * (amb_tot + diff_tot) + spec_tot, 0.0, 1.0)

  fl_rgb = _shade_ggx(p_fl, fl_nrm, fl_alb, 0.25, 0.35, fl_vis)
  rgb_bg = fl_mask[:, None] * fl_rgb + (1.0 - fl_mask[:, None]) * sky_rgb

  # 2. Foreground robot primitives: smooth-visibility alpha + soft-Z front-to-back composite
  alphas, depths, rgbs = [], [], []
  for g in fg_gids:
    gt = int(spec.geom_type[g])
    sz = jnp.asarray(spec.geom_size[g], jnp.float32)
    alb = jnp.broadcast_to(jnp.asarray(spec.geom_rgba[g, :3], jnp.float32)[None, :], (R, 3))
    sc_v = float(spec.geom_spec[g])
    sh_v = float(spec.geom_shin[g])

    phi_g, t_g, nrm_g = _diff_prim_ray(gt, geom_xpos[g], geom_xmat[g], sz, ro, rd)
    alpha_g = jax.nn.sigmoid(-phi_g / jnp.asarray(sigma, jnp.float32))
    pts_g = ro + t_g[:, None] * rd
    rgb_g = _shade_ggx(pts_g, nrm_g, alb, sc_v, sh_v, 1.0)
    alphas.append(alpha_g)
    depths.append(t_g)
    rgbs.append(rgb_g)

  alpha_all = jnp.stack(alphas, axis=-1)   # (R, G)
  depth_all = jnp.stack(depths, axis=-1)   # (R, G)
  rgb_all = jnp.stack(rgbs, axis=-2)       # (R, G, 3)

  occ_fg = 1.0 - jnp.prod(1.0 - alpha_all, axis=-1)
  t_front = jnp.min(jnp.where(alpha_all > 0.01, depth_all, 1e3), axis=-1, keepdims=True)
  w_g = alpha_all * jnp.exp(-jnp.clip(depth_all - t_front, 0.0, 2.0) / jnp.asarray(tau_z, jnp.float32))
  w_g = w_g / jnp.maximum(jnp.sum(w_g, axis=-1, keepdims=True), 1e-6)
  rgb_fg = jnp.sum(w_g[..., None] * rgb_all, axis=-2)

  rgb = occ_fg[:, None] * rgb_fg + (1.0 - occ_fg[:, None]) * rgb_bg
  return rgb.reshape(H, W, 3)


def _gaussian_blur_2d(img: jax.Array, sigma_px: float, ksize: int = 9) -> jax.Array:
  """Pure-JAX depthwise 2D Gaussian blur for multi-scale visual servoing attraction."""
  ax = jnp.arange(-(ksize // 2), ksize // 2 + 1, dtype=jnp.float32)
  k1d = jnp.exp(-0.5 * (ax / sigma_px) ** 2)
  k1d = k1d / jnp.sum(k1d)
  k2d = k1d[:, None] * k1d[None, :]
  kernel = jnp.broadcast_to(k2d[:, :, None, None], (ksize, ksize, 1, 3))
  x = img[None, ...].transpose(0, 3, 1, 2)  # (1, 3, H, W)
  # Wait: depthwise conv in NCHW with feature_group_count=3
  k_nchw = jnp.broadcast_to(k2d[None, None, :, :], (3, 1, ksize, ksize))
  out = jax.lax.conv_general_dilated(
      x, k_nchw, window_strides=(1, 1), padding="SAME",
      dimension_numbers=("NCHW", "OIHW", "NCHW"), feature_group_count=3,
  )
  return out[0].transpose(1, 2, 0)


def run_visual_servoing(
    res: int = 64,
    horizon: int = 12,
    n_iters: int = 240,
    lr: float = 0.04,
    out_npz: str = "/dev/shm/zaychik-tpu-mjx-w0/diff_phys_frames.npz",
    out_json: str = "/dev/shm/zaychik-tpu-mjx-w0/taskB_diffphys.json",
) -> Dict[str, Any]:
  xml = _panda_torque_arm_xml()
  m = mujoco.MjModel.from_xml_string(xml)
  mx = mjx.put_model(m, impl="jax")
  spec = RenderSceneSpec.from_mjmodel(m, cam_id=0, height=res, width=res, enabled_geom_groups=(0, 1, 2, 3))
  ee_gid = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "ee_beacon"))

  # Initial arm pose: upright, broadside to camera, zero self-occlusion
  q_init = np.array([-0.40, 0.20, 0.0, -1.65, 0.0, 1.55, 0.78], dtype=np.float32)
  d0 = mjx.make_data(m, impl="jax").replace(qpos=jnp.asarray(q_init))
  d0 = mjx.forward(mx, d0)

  # Ground-truth torque trajectory: smooth 20-26 cm broadside reach across the image
  t_ax = np.linspace(0.0, 1.0, horizon, dtype=np.float32)[:, None]
  ctrl_gt = np.hstack([
      1.15 * np.ones_like(t_ax),     # joint1 sweep left -> right across camera view
      -0.35 * np.ones_like(t_ax),    # joint2 lift
      0.15 * np.ones_like(t_ax),     # joint3 roll
      0.75 * np.ones_like(t_ax),     # joint4 extend elbow
      0.30 * np.sin(math.pi * t_ax), # joint5 wrist
      -0.25 * np.ones_like(t_ax),    # joint6 wrist
      0.15 * np.ones_like(t_ax),     # joint7 flange
  ]).astype(np.float32)

  def _rollout_render(ctrl_seq, sigma=0.008):
    def _step(d, u):
      d_next = mjx.step(mx, d.replace(ctrl=jnp.clip(u, -1.5, 1.5)))
      return d_next, None
    d_final, _ = jax.lax.scan(_step, d0, ctrl_seq)
    rgb = render_diff_pbr(spec, d_final, sigma=sigma, tau_z=0.015)
    return rgb, d_final

  # Render crisp target image from ground-truth torques
  target_rgb, d_target = jax.jit(lambda c: _rollout_render(c, sigma=0.004))(jnp.asarray(ctrl_gt))
  jax.block_until_ready(target_rgb)
  ee_init = np.asarray(d0.geom_xpos[ee_gid], dtype=np.float32)
  ee_target = np.asarray(d_target.geom_xpos[ee_gid], dtype=np.float32)
  q_target = np.asarray(d_target.qpos, dtype=np.float32)
  init_disp_cm = float(np.linalg.norm(ee_init - ee_target) * 100.0)
  print(f"[Task B] Target EE displacement from init: {init_disp_cm:.2f} cm", flush=True)

  target_b4 = jax.jit(lambda x: _gaussian_blur_2d(x, 2.5, ksize=9))(target_rgb)
  target_b12 = jax.jit(lambda x: _gaussian_blur_2d(x, 6.0, ksize=15))(target_rgb)
  target_b24 = jax.jit(lambda x: _gaussian_blur_2d(x, 12.0, ksize=25))(target_rgb)

  def _loss_fn(ctrl_seq, sigma_cur, w_fine, w_coarse):
    rgb, _ = _rollout_render(ctrl_seq, sigma=sigma_cur)
    l_fine = jnp.mean((rgb - target_rgb) ** 2)
    l_mid = jnp.mean((_gaussian_blur_2d(rgb, 2.5, ksize=9) - target_b4) ** 2)
    l_coarse = jnp.mean((_gaussian_blur_2d(rgb, 6.0, ksize=15) - target_b12) ** 2)
    l_global = jnp.mean((_gaussian_blur_2d(rgb, 12.0, ksize=25) - target_b24) ** 2)
    dctrl = ctrl_seq[1:] - ctrl_seq[:-1]
    reg = 2e-6 * jnp.mean(ctrl_seq ** 2) + 2e-5 * jnp.mean(dctrl ** 2)
    return w_fine * l_fine + w_coarse * (1.5 * l_mid + 3.0 * l_coarse + 6.0 * l_global) + reg, rgb

  @jax.jit
  def _train_step(ctrl_seq, opt_m, opt_v, step_idx):
    t_frac = step_idx.astype(jnp.float32) / float(max(n_iters - 1, 1))
    sigma_cur = 0.018 * (1.0 - t_frac) + 0.004 * t_frac
    # Stage-1 global silhouette pull (w_fine ramps up, w_coarse anneals down)
    w_fine = jnp.clip((t_frac - 0.15) / 0.35, 0.05, 1.0)
    w_coarse = jnp.maximum(0.05, 1.0 - 0.95 * t_frac)
    (loss_val, rgb_soft), grad = jax.value_and_grad(_loss_fn, has_aux=True)(ctrl_seq, sigma_cur, w_fine, w_coarse)
    cur_lr = float(lr) * (0.05 + 0.95 * 0.5 * (1.0 + jnp.cos(math.pi * t_frac)))
    b1, b2, eps = 0.9, 0.999, 5e-5
    opt_m = b1 * opt_m + (1.0 - b1) * grad
    opt_v = b2 * opt_v + (1.0 - b2) * (grad * grad)
    m_hat = opt_m / (1.0 - b1 ** (step_idx + 1))
    v_hat = opt_v / (1.0 - b2 ** (step_idx + 1))
    ctrl_next = jnp.clip(ctrl_seq - cur_lr * m_hat / (jnp.sqrt(v_hat) + eps), -1.5, 1.5)
    return ctrl_next, opt_m, opt_v, loss_val, jnp.linalg.norm(grad), rgb_soft

  @jax.jit
  def _eval_crisp(ctrl_seq):
    rgb_c, d_f = _rollout_render(ctrl_seq, sigma=0.004)
    mse = jnp.mean((rgb_c - target_rgb) ** 2)
    psnr = 10.0 * jnp.log10(1.0 / jnp.maximum(mse, 1e-10))
    return rgb_c, mse, psnr, d_f.geom_xpos[ee_gid], d_f.qpos

  # HLO verification on the coupled value_and_grad XLA graph!
  ctrl0 = jnp.zeros((horizon, m.nu), dtype=jnp.float32)
  m0 = jnp.zeros_like(ctrl0)
  v0 = jnp.zeros_like(ctrl0)
  lowered = _train_step.lower(ctrl0, m0, v0, jnp.int32(0))
  hlo_txt = lowered.as_text()
  hlo_info = inspect_hlo_ops(hlo_txt)
  hlo_info["send_recv_ops"] = int(len(__import__("re").findall(r"\bsend\(|\brecv\(", hlo_txt, __import__("re").I)))
  hlo_info["zero_copy_verified"] = (
      hlo_info["host_callbacks"] == 0 and hlo_info["send_recv_ops"] == 0 and hlo_info["gather_ops"] == 0
  )
  print("[Task B HLO Audit]", hlo_info, flush=True)
  assert hlo_info["host_callbacks"] == 0, f"Host callbacks found in Task B: {hlo_info}"

  # Warmup JIT
  ctrl_cur = ctrl0
  opt_m, opt_v = m0, v0
  c_w, m_w, v_w, l_w, gn_w, _ = _train_step(ctrl_cur, opt_m, opt_v, jnp.int32(0))
  jax.block_until_ready(l_w)

  # Evaluate initial state (step 0)
  rgb_init, mse0, psnr0, ee0, q0 = _eval_crisp(ctrl0)
  jax.block_until_ready(rgb_init)
  ee_err0_cm = float(np.linalg.norm(np.asarray(ee0) - ee_target) * 100.0)

  history: List[Dict[str, float]] = []
  mid_iter = n_iters // 3
  rgb_mid = None
  ee_err_mid_cm = None
  psnr_mid = None

  t_start = time.perf_counter()
  for it in range(n_iters):
    ctrl_cur, opt_m, opt_v, loss_val, gnorm, _ = _train_step(ctrl_cur, opt_m, opt_v, jnp.int32(it))
    if it == 0 or (it + 1) % 10 == 0 or it == mid_iter or it == n_iters - 1:
      rgb_c, mse_c, psnr_c, ee_c, q_c = _eval_crisp(ctrl_cur)
      jax.block_until_ready(mse_c)
      ee_cm = float(np.linalg.norm(np.asarray(ee_c) - ee_target) * 100.0)
      q_deg = float(np.sqrt(np.mean((np.asarray(q_c) - q_target) ** 2)) * 180.0 / math.pi)
      rec = {
          "iter": int(it + 1),
          "loss": float(loss_val),
          "grad_norm": float(gnorm),
          "pixel_mse": float(mse_c),
          "psnr_db": float(psnr_c),
          "ee_err_cm": round(ee_cm, 3),
          "q_rmse_deg": round(q_deg, 3),
      }
      history.append(rec)
      if it == mid_iter:
        rgb_mid = np.asarray(rgb_c)
        ee_err_mid_cm = ee_cm
        psnr_mid = float(psnr_c)
      if it == 0 or (it + 1) % 30 == 0 or it == n_iters - 1:
        print(
            f"[Servo iter {it+1:3d}/{n_iters}] loss={float(loss_val):.5f} "
            f"MSE={float(mse_c):.6f} PSNR={float(psnr_c):.2f} dB "
            f"EE_err={ee_cm:.2f} cm | grad={float(gnorm):.4f}",
            flush=True,
        )

  jax.block_until_ready(ctrl_cur)
  dt_total_s = time.perf_counter() - t_start
  grad_steps_per_sec = float(n_iters / dt_total_s)
  phys_steps_per_sec = float((n_iters * horizon) / dt_total_s)

  rgb_final, mse_f, psnr_f, ee_f, q_f = _eval_crisp(ctrl_cur)
  jax.block_until_ready(rgb_final)
  ee_err_f_cm = float(np.linalg.norm(np.asarray(ee_f) - ee_target) * 100.0)
  q_rmse_f_deg = float(np.sqrt(np.mean((np.asarray(q_f) - q_target) ** 2)) * 180.0 / math.pi)

  os.makedirs(os.path.dirname(out_npz) or ".", exist_ok=True)
  np.savez_compressed(
      out_npz,
      rgb_init=np.clip(np.asarray(rgb_init) * 255.0 + 0.5, 0, 255).astype(np.uint8),
      rgb_mid=np.clip(rgb_mid * 255.0 + 0.5, 0, 255).astype(np.uint8),
      rgb_final=np.clip(np.asarray(rgb_final) * 255.0 + 0.5, 0, 255).astype(np.uint8),
      rgb_target=np.clip(np.asarray(target_rgb) * 255.0 + 0.5, 0, 255).astype(np.uint8),
      ctrl_init=np.zeros_like(ctrl_gt),
      ctrl_final=np.asarray(ctrl_cur),
      ctrl_gt=ctrl_gt,
  )

  res_dict = {
      "task": "B_diff_phys_render_visual_servoing",
      "res": f"{res}x{res}",
      "horizon_steps": horizon,
      "n_iters": n_iters,
      "n_dof": int(m.nu),
      "init_ee_displacement_cm": round(init_disp_cm, 3),
      "init_pixel_mse": float(mse0),
      "init_psnr_db": round(float(psnr0), 2),
      "init_ee_err_cm": round(ee_err0_cm, 3),
      "mid_iter": int(mid_iter + 1),
      "mid_psnr_db": round(float(psnr_mid), 2),
      "mid_ee_err_cm": round(float(ee_err_mid_cm), 3),
      "final_pixel_mse": float(mse_f),
      "final_psnr_db": round(float(psnr_f), 2),
      "final_ee_err_cm": round(ee_err_f_cm, 3),
      "final_q_rmse_deg": round(q_rmse_f_deg, 3),
      "total_opt_sec": round(dt_total_s, 3),
      "grad_steps_per_sec": round(grad_steps_per_sec, 2),
      "coupled_phys_steps_per_sec": round(phys_steps_per_sec, 1),
      "hlo_audit": hlo_info,
      "history": history,
  }
  os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
  with open(out_json, "w") as f:
    json.dump(res_dict, f, indent=2)
  print("Saved Task B JSON ->", out_json, flush=True)
  return res_dict


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--res", type=int, default=64)
  ap.add_argument("--horizon", type=int, default=12)
  ap.add_argument("--n_iters", type=int, default=240)
  ap.add_argument("--lr", type=float, default=0.04)
  ap.add_argument("--out_npz", default="/dev/shm/zaychik-tpu-mjx-w0/diff_phys_frames.npz")
  ap.add_argument("--out_json", default="/dev/shm/zaychik-tpu-mjx-w0/taskB_diffphys.json")
  args = ap.parse_args()
  run_visual_servoing(
      res=args.res,
      horizon=args.horizon,
      n_iters=args.n_iters,
      lr=args.lr,
      out_npz=args.out_npz,
      out_json=args.out_json,
  )


if __name__ == "__main__":
  main()
