"""RoboTwin-style Bimanual Dual-Panda 3-Camera Cluttered Tabletop Suite on TPU.

Builds a self-contained Dual-Panda bimanual tabletop manipulation MJCF with:
- Left & Right 7-DOF Franka Panda arms + parallel grippers
- RoboTwin 3-camera rig: `head_camera` + `left_wrist_camera` + `right_wrist_camera`
- Additional surround cameras (`diag_overview`, `front_overview`, `top_down`, `side_overview`)
- Rich cluttered workbench (sorting bins, tool rack, bottles, wrench, mallet, target cubes)
- Synchronized RGB + Depth + Semantic Segmentation via single-kernel `pallas_raycast_ncam_v2`
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from typing import Any, Dict, List, Sequence, Tuple

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np

from experiments.w6_highres_multicam.wave3_8k_multicam8 import pallas_raycast_ncam_v2
from tpu_mjx_render.renderer import RenderSceneSpec
from tpu_mjx_render.roofline import inspect_hlo_ops


def _panda_arm_xml(prefix: str, base_pos: str, yaw_rad: float = 0.0, wrist_cam_name: str = "wrist") -> str:
  """Generate one prefixed Panda arm subtree with a wrist camera."""
  # Rotate base quat around Z by yaw_rad: [cos(yaw/2), 0, 0, sin(yaw/2)]
  c = float(np.cos(0.5 * yaw_rad))
  s = float(np.sin(0.5 * yaw_rad))
  return f"""
    <body name="{prefix}link0" pos="{base_pos}" quat="{c:.6f} 0 0 {s:.6f}" childclass="panda">
      <geom name="{prefix}link0_base" type="cylinder" pos="-0.03 0 0.07" size="0.095 0.07" material="off_white" class="visual_prim"/>
      <geom name="{prefix}link0_pedestal" type="cylinder" pos="0 0 0.21" size="0.068 0.08" material="white" class="visual_prim"/>
      <body name="{prefix}link1" pos="0 0 0.333">
        <inertial mass="2.0" pos="0 0 -0.05" diaginertia="0.01 0.01 0.01"/>
        <joint name="{prefix}joint1" damping="40"/>
        <geom name="{prefix}link1_body" type="capsule" fromto="0 0 -0.10 0 0 0.04" size="0.062" material="white" class="visual_prim"/>
        <body name="{prefix}link2" quat="1 -1 0 0">
          <inertial mass="2.0" pos="0 -0.15 0" diaginertia="0.01 0.01 0.01"/>
          <joint name="{prefix}joint2" range="-1.7628 1.7628" damping="40"/>
          <geom name="{prefix}link2_collar" type="cylinder" pos="0 0 0" size="0.065 0.045" material="black" class="visual_prim"/>
          <geom name="{prefix}link2_arm" type="capsule" fromto="0 0 0 0 -0.22 0" size="0.055" material="white" class="visual_prim"/>
          <body name="{prefix}link3" pos="0 -0.316 0" quat="1 1 0 0">
            <inertial mass="2.0" pos="0 0 -0.05" diaginertia="0.01 0.01 0.01"/>
            <joint name="{prefix}joint3" damping="40"/>
            <geom name="{prefix}link3_seg" type="capsule" fromto="0 0 -0.11 0.045 0 0" size="0.052" material="white" class="visual_prim"/>
            <geom name="{prefix}link3_joint" type="sphere" pos="0.0825 0 0" size="0.056" material="black" class="visual_prim"/>
            <body name="{prefix}link4" pos="0.0825 0 0" quat="1 1 0 0">
              <inertial mass="2.0" pos="-0.04 0.15 0" diaginertia="0.01 0.01 0.01"/>
              <joint name="{prefix}joint4" range="-3.0718 -0.0698" damping="40"/>
              <geom name="{prefix}link4_arm" type="capsule" fromto="0 0 0 -0.06 0.20 0" size="0.050" material="white" class="visual_prim"/>
              <body name="{prefix}link5" pos="-0.0825 0.384 0" quat="1 -1 0 0">
                <inertial mass="1.5" pos="0 0 -0.1" diaginertia="0.008 0.008 0.008"/>
                <joint name="{prefix}joint5" damping="2"/>
                <geom name="{prefix}link5_forearm" type="capsule" fromto="0 0 -0.22 0 0 -0.02" size="0.044" material="off_white" class="visual_prim"/>
                <geom name="{prefix}link5_collar" type="cylinder" pos="0 0 0" size="0.046 0.025" material="black" class="visual_prim"/>
                <body name="{prefix}link6" quat="1 1 0 0">
                  <inertial mass="1.0" pos="0.04 0 0" diaginertia="0.005 0.005 0.005"/>
                  <joint name="{prefix}joint6" range="-0.0175 3.7525" damping="2"/>
                  <geom name="{prefix}link6_wrist" type="capsule" fromto="0 0 0 0.088 0 0" size="0.040" material="white" class="visual_prim"/>
                  <body name="{prefix}link7" pos="0.088 0 0" quat="1 1 0 0">
                    <inertial mass="0.7" pos="0 0 0.05" diaginertia="0.003 0.003 0.003"/>
                    <joint name="{prefix}joint7" damping="2"/>
                    <geom name="{prefix}link7_flange" type="cylinder" pos="0 0 0.06" size="0.038 0.045" material="black" class="visual_prim"/>
                    <body name="{prefix}hand" pos="0 0 0.107" quat="0.9238795 0 0 -0.3826834">
                      <inertial mass="0.73" pos="0 0 0.03" diaginertia="0.001 0.0025 0.0017"/>
                      <geom name="{prefix}hand_box" type="box" pos="0 0 0.030" size="0.024 0.102 0.026" material="off_white" class="visual_prim"/>
                      <geom name="{prefix}hand_capsule" type="capsule" size="0.036 0.055" quat="1 1 0 0" pos="0 0 0.03" material="black" contype="1" conaffinity="1" group="3"/>
                      <site name="{prefix}gripper" pos="0 0 0.10"/>
                      <camera name="{wrist_cam_name}" fovy="58" pos="-0.065 0.0 0.035" xyaxes="0 1 0 0.8575 0 -0.5145"/>
                      <body name="{prefix}left_finger" pos="0 0 0.0584">
                        <inertial mass="0.03" pos="0 0.01 0.025" diaginertia="1e-5 1e-5 1e-5"/>
                        <joint name="{prefix}finger_joint1" class="finger" damping="10"/>
                        <geom name="{prefix}left_finger_vis" type="box" pos="0 0.010 0.026" size="0.010 0.008 0.024" material="black" class="visual_prim"/>
                        <geom name="{prefix}left_finger_pad" type="box" size="0.009 0.0075 0.016" pos="0 0.008 0.038" material="off_white" class="pad_col"/>
                      </body>
                      <body name="{prefix}right_finger" pos="0 0 0.0584" quat="0 0 0 1">
                        <inertial mass="0.03" pos="0 0.01 0.025" diaginertia="1e-5 1e-5 1e-5"/>
                        <joint name="{prefix}finger_joint2" class="finger" damping="10"/>
                        <geom name="{prefix}right_finger_vis" type="box" pos="0 0.010 0.026" size="0.010 0.008 0.024" material="black" class="visual_prim"/>
                        <geom name="{prefix}right_finger_pad" type="box" size="0.009 0.0075 0.016" pos="0 0.008 0.038" material="off_white" class="pad_col"/>
                      </body>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>"""


def _panda_actuators_xml(prefix: str) -> str:
  return f"""
    <position class="panda" name="{prefix}act1" joint="{prefix}joint1" kp="1000" kv="20" ctrlrange="-2.8973 2.8973"/>
    <position class="panda" name="{prefix}act2" joint="{prefix}joint2" kp="1000" kv="20" ctrlrange="-1.7628 1.7628"/>
    <position class="panda" name="{prefix}act3" joint="{prefix}joint3" kp="750" kv="4" ctrlrange="-2.8973 2.8973"/>
    <position class="panda" name="{prefix}act4" joint="{prefix}joint4" kp="750" kv="4" ctrlrange="-3.0718 -0.0698"/>
    <position class="panda" name="{prefix}act5" joint="{prefix}joint5" kp="300" kv="2" forcerange="-12 12" ctrlrange="-2.8973 2.8973"/>
    <position class="panda" name="{prefix}act6" joint="{prefix}joint6" kp="300" kv="2" forcerange="-12 12" ctrlrange="-0.0175 3.7525"/>
    <position class="panda" name="{prefix}act7" joint="{prefix}joint7" kp="300" kv="2" forcerange="-12 12"/>
    <general class="panda" name="{prefix}act8" joint="{prefix}finger_joint1" ctrlrange="0 0.04" gainprm="350 0 0"
      biasprm="0 -350 -10" forcerange="-100 100"/>"""


def make_robotwin_bimanual_xml() -> str:
  """RoboTwin-style Dual-Panda Bimanual Cluttered Tabletop MJCF with 6 cameras."""
  left_arm = _panda_arm_xml("L_", "0.02 0.28 0.01", yaw_rad=-0.18, wrist_cam_name="left_wrist_camera")
  right_arm = _panda_arm_xml("R_", "0.02 -0.28 0.01", yaw_rad=0.18, wrist_cam_name="right_wrist_camera")

  l_qpos = "0.04 0.45 -0.05 -1.82 -0.02 2.32 0.78 0.035 0.035"
  r_qpos = "-0.04 0.52 0.08 -1.78 0.04 2.28 0.78 0.028 0.028"
  cube_qpos = "0.62 -0.03 0.054 1.0 0.0 0.0 0.0"
  l_ctrl = "0.04 0.45 -0.05 -1.82 -0.02 2.32 0.78 0.035"
  r_ctrl = "-0.04 0.52 0.08 -1.78 0.04 2.28 0.78 0.028"

  return f"""<mujoco model="robotwin_bimanual_panda_tabletop">
  <compiler angle="radian" autolimits="true"/>
  <statistic center="0.5 0 0.3" extent="1.2"/>

  <option timestep="0.005" iterations="5" ls_iterations="8" integrator="implicitfast">
    <flag eulerdamp="disable"/>
  </option>

  <visual>
    <headlight diffuse="0.45 0.45 0.45" ambient="0.25 0.25 0.25" specular="0.15 0.15 0.15"/>
    <global azimuth="120" elevation="-20"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.30 0.48 0.68" rgb2="0.06 0.08 0.12" width="128" height="128"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.26 0.34 0.42" rgb2="0.16 0.22 0.30"
      markrgb="0.75 0.78 0.82" width="128" height="128"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
    <material name="white" rgba="0.95 0.96 0.97 1"/>
    <material name="off_white" rgba="0.88 0.90 0.92 1"/>
    <material name="black" rgba="0.22 0.24 0.26 1"/>
    <material name="cube_red" rgba="0.88 0.14 0.12 1"/>
  </asset>

  <default>
    <default class="panda">
      <material specular="0.4" shininess="0.3"/>
      <joint armature="0.1" damping="1" axis="0 0 1" range="-2.8973 2.8973"/>
      <general dyntype="none" biastype="affine" ctrlrange="-2.8973 2.8973" forcerange="-87 87"/>
      <position forcerange="-100 100"/>
      <default class="finger">
        <joint axis="0 1 0" type="slide" range="0 0.04"/>
      </default>
      <default class="visual_prim">
        <geom contype="0" conaffinity="0" group="2"/>
      </default>
      <default class="pad_col">
        <geom group="3" type="box" contype="1" conaffinity="1" condim="3" friction="1 0.005 0.0001"/>
      </default>
    </default>
  </default>

  <worldbody>
    <light name="sun_overhead" pos="0.6 0.3 1.9" dir="-0.15 -0.15 -1" directional="false" castshadow="true"
      diffuse="0.88 0.85 0.80" specular="0.3 0.3 0.3"/>

    <!-- RoboTwin Head Camera (overlooking bimanual arms + tabletop, clear of head_bar x=0.055..0.145) -->
    <camera name="head_camera" fovy="52" pos="0.18 0.0 0.92" xyaxes="0 -1 0 0.850 0 0.526"/>
    <!-- Surround showcase cameras -->
    <camera name="diag_overview" fovy="45" pos="1.25 -0.78 0.72" xyaxes="0.54 0.84 0 -0.46 0.30 0.84"/>
    <camera name="front_overview" fovy="44" pos="1.32 0.00 0.56" xyaxes="0 1 0 -0.62 0 0.78"/>
    <camera name="top_down" fovy="48" pos="0.58 0.00 1.48" xyaxes="0 1 0 -1 0 0"/>

    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane" contype="1" conaffinity="1"/>

    <!-- Head camera mast behind arms -->
    <geom name="head_mast" type="cylinder" pos="0.06 0.0 0.45" size="0.022 0.45" rgba="0.22 0.24 0.28 1" contype="0" conaffinity="0"/>
    <geom name="head_bar" type="box" pos="0.10 0.0 0.92" size="0.045 0.09 0.028" rgba="0.16 0.18 0.22 1" contype="0" conaffinity="0"/>

    <!-- Workbench + Clutter -->
    <geom name="workbench_top" type="box" pos="0.64 0.0 0.012" size="0.44 0.58 0.012" rgba="0.56 0.40 0.25 1" contype="1" conaffinity="1"/>
    <geom name="workbench_trim" type="box" pos="0.64 0.0 0.005" size="0.46 0.60 0.005" rgba="0.22 0.24 0.27 1" contype="0" conaffinity="0"/>

    <geom name="bin_blue_base" type="box" pos="0.52 -0.36 0.028" size="0.09 0.09 0.006" rgba="0.12 0.42 0.85 1" contype="0" conaffinity="0"/>
    <geom name="bin_blue_w1" type="box" pos="0.43 -0.36 0.062" size="0.006 0.09 0.032" rgba="0.12 0.42 0.85 1" contype="0" conaffinity="0"/>
    <geom name="bin_blue_w2" type="box" pos="0.61 -0.36 0.062" size="0.006 0.09 0.032" rgba="0.12 0.42 0.85 1" contype="0" conaffinity="0"/>
    <geom name="bin_blue_w3" type="box" pos="0.52 -0.45 0.062" size="0.09 0.006 0.032" rgba="0.12 0.42 0.85 1" contype="0" conaffinity="0"/>
    <geom name="bin_blue_w4" type="box" pos="0.52 -0.27 0.062" size="0.09 0.006 0.032" rgba="0.12 0.42 0.85 1" contype="0" conaffinity="0"/>

    <geom name="bin_green_base" type="box" pos="0.52 0.36 0.028" size="0.09 0.09 0.006" rgba="0.16 0.68 0.34 1" contype="0" conaffinity="0"/>
    <geom name="bin_green_w1" type="box" pos="0.43 0.36 0.062" size="0.006 0.09 0.032" rgba="0.16 0.68 0.34 1" contype="0" conaffinity="0"/>
    <geom name="bin_green_w2" type="box" pos="0.61 0.36 0.062" size="0.006 0.09 0.032" rgba="0.16 0.68 0.34 1" contype="0" conaffinity="0"/>
    <geom name="bin_green_w3" type="box" pos="0.52 0.27 0.062" size="0.09 0.006 0.032" rgba="0.16 0.68 0.34 1" contype="0" conaffinity="0"/>
    <geom name="bin_green_w4" type="box" pos="0.52 0.45 0.062" size="0.09 0.006 0.032" rgba="0.16 0.68 0.34 1" contype="0" conaffinity="0"/>

    <geom name="rack_post_l" type="cylinder" pos="0.90 -0.26 0.14" size="0.015 0.12" rgba="0.25 0.26 0.28 1" contype="0" conaffinity="0"/>
    <geom name="rack_post_r" type="cylinder" pos="0.90 0.26 0.14" size="0.015 0.12" rgba="0.25 0.26 0.28 1" contype="0" conaffinity="0"/>
    <geom name="rack_bar" type="capsule" fromto="0.90 -0.27 0.23 0.90 0.27 0.23" size="0.014" rgba="0.82 0.56 0.16 1" contype="0" conaffinity="0"/>
    <geom name="rack_shelf" type="box" pos="0.90 0.0 0.12" size="0.045 0.26 0.008" rgba="0.32 0.34 0.38 1" contype="0" conaffinity="0"/>

    <geom name="bottle_teal" type="cylinder" pos="0.74 -0.22 0.085" size="0.028 0.060" rgba="0.10 0.72 0.78 1" contype="0" conaffinity="0"/>
    <geom name="bottle_teal_cap" type="cylinder" pos="0.74 -0.22 0.155" size="0.014 0.012" rgba="0.95 0.95 0.95 1" contype="0" conaffinity="0"/>
    <geom name="bottle_amber" type="cylinder" pos="0.76 0.22 0.095" size="0.030 0.070" rgba="0.92 0.52 0.12 1" contype="0" conaffinity="0"/>
    <geom name="bottle_amber_cap" type="cylinder" pos="0.76 0.22 0.175" size="0.013 0.012" rgba="0.20 0.20 0.22 1" contype="0" conaffinity="0"/>
    <geom name="spray_can" type="capsule" pos="0.46 0.16 0.065" size="0.022 0.040" rgba="0.88 0.22 0.52 1" contype="0" conaffinity="0"/>

    <geom name="wrench_handle" type="box" pos="0.56 -0.14 0.032" euler="0 0 0.45" size="0.075 0.012 0.008" rgba="0.78 0.80 0.84 1" contype="0" conaffinity="0"/>
    <geom name="wrench_head" type="cylinder" pos="0.63 -0.105 0.032" size="0.022 0.009" rgba="0.72 0.75 0.79 1" contype="0" conaffinity="0"/>
    <geom name="mallet_handle" type="capsule" fromto="0.40 0.05 0.036 0.53 0.13 0.036" size="0.011" rgba="0.74 0.50 0.26 1" contype="0" conaffinity="0"/>
    <geom name="mallet_head" type="cylinder" pos="0.54 0.135 0.045" euler="1.57 0.45 0" size="0.024 0.036" rgba="0.18 0.18 0.20 1" contype="0" conaffinity="0"/>

    <geom name="prop_sphere_gold" type="sphere" pos="0.51 -0.35 0.058" size="0.024" rgba="0.96 0.76 0.14 1" contype="0" conaffinity="0"/>
    <geom name="prop_block_cyan" type="box" pos="0.54 -0.39 0.052" euler="0 0 0.3" size="0.020 0.020 0.018" rgba="0.18 0.80 0.90 1" contype="0" conaffinity="0"/>
    <geom name="prop_block_yellow" type="box" pos="0.50 0.35 0.054" euler="0 0 -0.4" size="0.022 0.022 0.020" rgba="0.95 0.85 0.18 1" contype="0" conaffinity="0"/>
    <geom name="prop_cyl_purple" type="cylinder" pos="0.55 0.38 0.060" size="0.018 0.026" rgba="0.64 0.24 0.84 1" contype="0" conaffinity="0"/>
    <geom name="target_cube_blue" type="box" pos="0.70 0.08 0.050" euler="0 0 0.5" size="0.024 0.024 0.026" rgba="0.18 0.48 0.94 1" contype="0" conaffinity="0"/>
    <geom name="target_cube_green" type="box" pos="0.59 0.11 0.044" euler="0 0 -0.2" size="0.020 0.020 0.020" rgba="0.22 0.82 0.32 1" contype="0" conaffinity="0"/>
    <geom name="mug_ellipsoid" type="ellipsoid" pos="0.74 -0.06 0.056" size="0.028 0.028 0.032" rgba="0.94 0.92 0.86 1" contype="0" conaffinity="0"/>

    {left_arm}
    {right_arm}

    <body name="box" pos="0.62 -0.03 0.054">
      <freejoint/>
      <inertial mass="0.1" pos="0 0 0" diaginertia="1e-4 1e-4 1e-4"/>
      <geom type="box" name="box" size="0.022 0.022 0.030" condim="3"
       friction="1 0.03 0.003" material="cube_red" contype="1" conaffinity="1" solref="0.01 1"/>
    </body>
  </worldbody>

  <equality>
    <joint joint1="L_finger_joint1" joint2="L_finger_joint2" solimp="0.95 0.99 0.001" solref="0.005 1"/>
    <joint joint1="R_finger_joint1" joint2="R_finger_joint2" solimp="0.95 0.99 0.001" solref="0.005 1"/>
  </equality>

  <actuator>
    {_panda_actuators_xml("L_")}
    {_panda_actuators_xml("R_")}
  </actuator>

  <keyframe>
    <key name="bimanual_ready" qpos="{l_qpos} {r_qpos} {cube_qpos}" ctrl="{l_ctrl} {r_ctrl}"/>
  </keyframe>
</mujoco>
"""


def assign_bimanual_semantic_classes(m: mujoco.MjModel, gids: Sequence[int]) -> Tuple[int, ...]:
  """Semantic classes: 1=floor/bench, 2=left_arm, 3=right_arm, 4=targets/props, 5=bins, 6=tools/rack."""
  out: List[int] = []
  for gid in gids:
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(gid)) or ""
    if name in ("floor", "workbench_top", "workbench_trim", "head_mast", "head_bar"):
      sem = 1
    elif name.startswith("L_"):
      sem = 2
    elif name.startswith("R_"):
      sem = 3
    elif name in ("box", "mug_ellipsoid") or name.startswith("target_cube_") or name.startswith("prop_"):
      sem = 4
    elif name.startswith("bin_"):
      sem = 5
    else:
      sem = 6
    out.append(sem)
  return tuple(out)


def geom_id_to_semantic_bimanual(geom_id: jax.Array, gids_static: Sequence[int], sem_ids_static: Sequence[int]) -> jax.Array:
  sem = jnp.zeros_like(geom_id, dtype=jnp.int32)
  for cls_id in (1, 2, 3, 4, 5, 6):
    cls_gids = [g for g, s in zip(gids_static, sem_ids_static) if s == cls_id]
    if not cls_gids:
      continue
    mask = jnp.zeros(geom_id.shape, dtype=jnp.bool_)
    for g in cls_gids:
      mask = mask | (geom_id == int(g))
    sem = jnp.where(mask, jnp.int32(cls_id), sem)
  return sem


def run_robotwin_suite(export_dir: str, out_json: str, batches: Tuple[int, ...] = (256,)) -> Dict[str, Any]:
  """Benchmark RoboTwin 3-cam (head + left_wrist + right_wrist) RGB+Depth+Seg & export 4K data."""
  pallas_interp = (jax.default_backend() == "cpu")
  xml = make_robotwin_bimanual_xml()
  m = mujoco.MjModel.from_xml_string(xml)
  mx = mjx.put_model(m, impl="jax")

  d_host = mujoco.MjData(m)
  mujoco.mj_resetDataKeyframe(m, d_host, 0)
  mujoco.mj_forward(m, d_host)
  dx0 = mjx.put_data(m, d_host, impl="jax")

  cam_names = [
      "head_camera",
      "diag_overview",
      "left_wrist_camera",
      "right_wrist_camera",
      "front_overview",
      "top_down",
  ]
  cam_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, nm) for nm in cam_names]
  assert all(cid >= 0 for cid in cam_ids), list(zip(cam_names, cam_ids))

  dev0 = jax.devices()[0]

  # 1. Render Native 4K (3840x2160) showcase tiles FIRST (fast / cached): 6 cams at 960x1080
  dx1 = jax.device_put(jax.tree_util.tree_map(lambda x: x[None, ...], dx0), dev0)
  specs_6 = [RenderSceneSpec.from_mjmodel(m, cam_id=cid, height=1080, width=960, use_shadows=True) for cid in cam_ids]
  sem_ids_static = assign_bimanual_semantic_classes(m, specs_6[0].enabled_geom_ids)
  fn_6 = jax.jit(lambda d: pallas_raycast_ncam_v2(specs_6, d, interpret=pallas_interp))
  gbs = fn_6(dx1)
  jax.block_until_ready([g.rgb for g in gbs])
  t0 = time.perf_counter()
  for _ in range(3):
    gbs = fn_6(dx1)
  jax.block_until_ready([g.rgb for g in gbs])
  dt_4k_ms = (time.perf_counter() - t0) * 1e3 / 3.0

  os.makedirs(export_dir, exist_ok=True)
  sems = [np.asarray(geom_id_to_semantic_bimanual(gbs[i].geom_id, specs_6[0].enabled_geom_ids, sem_ids_static)[0], np.int32) for i in (0, 2, 3)]
  np.savez(
      os.path.join(export_dir, "robotwin_bimanual_4k_data.npz"),
      head_rgb=np.asarray(gbs[0].rgb[0], np.float32),
      head_depth=np.asarray(gbs[0].depth[0], np.float32),
      head_sem=sems[0],
      diag_rgb=np.asarray(gbs[1].rgb[0], np.float32),
      lw_rgb=np.asarray(gbs[2].rgb[0], np.float32),
      lw_depth=np.asarray(gbs[2].depth[0], np.float32),
      lw_sem=sems[1],
      rw_rgb=np.asarray(gbs[3].rgb[0], np.float32),
      rw_depth=np.asarray(gbs[3].depth[0], np.float32),
      rw_sem=sems[2],
      front_rgb=np.asarray(gbs[4].rgb[0], np.float32),
      top_rgb=np.asarray(gbs[5].rgb[0], np.float32),
  )
  print(f"ROBOTWIN 6-CAM 960x1080 rendered in {dt_4k_ms:.2f}ms", flush=True)

  # 2. Benchmark RoboTwin 3-Camera (head + L_wrist + R_wrist) RGB+Depth+Semantic at 128x128
  specs_3cam = [RenderSceneSpec.from_mjmodel(m, cam_id=cid, height=128, width=128, use_shadows=True) for cid in [cam_ids[0], cam_ids[2], cam_ids[3]]]

  bench_rows = []
  for B in batches:
    dx_b = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x[None, ...], (B, *x.shape)).copy(), dx0)
    phases = jnp.linspace(-0.2, 0.2, B, dtype=jnp.float32)[:, None]
    ctrl = jnp.asarray(dx0.ctrl, jnp.float32)[None, :] + phases * jnp.ones((1, m.nu), jnp.float32)
    dx_b = jax.device_put(dx_b, dev0)
    ctrl = jax.device_put(ctrl, dev0)

    @jax.jit
    def e2e_3cam(d, u):
      dn = jax.vmap(lambda x, c: mjx.step(mx, x.replace(ctrl=c)))(d, u)
      gbs_3 = pallas_raycast_ncam_v2(specs_3cam, dn, interpret=pallas_interp)
      sems_3 = tuple(geom_id_to_semantic_bimanual(g.geom_id, specs_3cam[0].enabled_geom_ids, sem_ids_static) for g in gbs_3)
      return (dn, *(g.rgb for g in gbs_3), *(g.depth for g in gbs_3), *sems_3)

    txt = e2e_3cam.lower(dx_b, ctrl).as_text()
    hlo = inspect_hlo_ops(txt)
    hlo["send_recv_ops"] = len(re.findall(r"\bsend\(|\brecv\(", txt, re.I))
    assert hlo["host_callbacks"] == 0 and hlo["send_recv_ops"] == 0 and hlo["gather_ops"] == 0, hlo

    dc = dx_b
    for _ in range(2):
      out = e2e_3cam(dc, ctrl)
      dc = out[0]
    jax.block_until_ready(out[1:])

    t0 = time.perf_counter()
    for _ in range(6):
      out = e2e_3cam(dc, ctrl)
      dc = out[0]
    jax.block_until_ready(out[1:])
    dt_ms = (time.perf_counter() - t0) * 1e3 / 6.0
    fps = B * 1e3 / dt_ms
    rec = {
        "pipeline": "robotwin_3cam_rgb_depth_seg",
        "n_arms": 2,
        "n_cameras": 3,
        "res": "128x128",
        "batch_size": B,
        "dt_e2e_ms": dt_ms,
        "e2e_fps": fps,
        "cam_frames_per_s": fps * 3,
        "host_callbacks": hlo["host_callbacks"],
        "send_recv_ops": hlo["send_recv_ops"],
        "gather_ops": hlo["gather_ops"],
    }
    bench_rows.append(rec)
    print(f"ROBOTWIN 3-CAM B={B}: e2e={dt_ms:.2f}ms fps={fps:.1f} (cam_fps={fps*3:.1f})", flush=True)

  res = {
      "status": "ok",
      "bimanual_3cam_bench": bench_rows,
      "showcase_6cam_960x1080_ms": dt_4k_ms,
      "export_npz": os.path.join(export_dir, "robotwin_bimanual_4k_data.npz"),
  }
  with open(out_json, "w") as f:
    json.dump(res, f, indent=2)
  print(f"Saved {out_json}", flush=True)
  return res


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--out_json", required=True)
  ap.add_argument("--export_dir", default="/dev/shm/zaychik-tpu-mjx-w6/out/vis")
  ap.add_argument("--batches", type=int, nargs="+", default=[256])
  args = ap.parse_args()
  run_robotwin_suite(args.export_dir, args.out_json, tuple(args.batches))


if __name__ == "__main__":
  main()
