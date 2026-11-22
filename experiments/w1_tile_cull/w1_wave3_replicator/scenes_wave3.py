"""Wave 3 Benchmark Scenes: RoboTwin Bimanual 3-Cam & RoboDojo Multi-Task Cluttered Bench."""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import mujoco
import numpy as np


def _panda_arm_xml(prefix: str, base_pos: str, yaw_rad: float = 0.0, wrist_cam_name: str = "wrist") -> str:
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
  """RoboTwin-style Dual-Panda Bimanual Cluttered Tabletop MJCF (head + L/R wrist + overview)."""
  left_arm = _panda_arm_xml("L_", "0.02 0.28 0.01", yaw_rad=-0.18, wrist_cam_name="left_wrist_camera")
  right_arm = _panda_arm_xml("R_", "0.02 -0.28 0.01", yaw_rad=0.18, wrist_cam_name="right_wrist_camera")

  l_qpos = "-0.12 0.45 -0.05 -1.82 -0.02 2.32 0.78 0.035 0.035"
  r_qpos = "0.14 0.52 0.08 -1.78 0.04 2.28 0.78 0.028 0.028"
  cube_qpos = "0.62 -0.03 0.054 1.0 0.0 0.0 0.0"
  cube2_qpos = "0.68 0.09 0.052 1.0 0.0 0.0 0.0"
  can_qpos = "0.56 0.14 0.062 1.0 0.0 0.0 0.0"
  l_ctrl = "-0.12 0.45 -0.05 -1.82 -0.02 2.32 0.78 0.035"
  r_ctrl = "0.14 0.52 0.08 -1.78 0.04 2.28 0.78 0.028"

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
    <material name="cube_blue" rgba="0.14 0.46 0.92 1"/>
    <material name="can_yellow" rgba="0.94 0.74 0.14 1"/>
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

    <camera name="head_camera" fovy="48" pos="0.16 0.0 0.92" xyaxes="0 -1 0 0.850 0 0.526"/>
    <camera name="diag_overview" fovy="46" pos="1.25 -0.78 0.72" xyaxes="0.54 0.84 0 -0.46 0.30 0.84"/>

    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane" contype="1" conaffinity="1"/>
    <geom name="head_mast" type="cylinder" pos="0.06 0.0 0.45" size="0.022 0.45" rgba="0.22 0.24 0.28 1" contype="0" conaffinity="0"/>
    <geom name="head_bar" type="box" pos="0.10 0.0 0.92" size="0.045 0.09 0.028" rgba="0.16 0.18 0.22 1" contype="0" conaffinity="0"/>

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
    <geom name="mug_ellipsoid" type="ellipsoid" pos="0.74 -0.06 0.056" size="0.028 0.028 0.032" rgba="0.94 0.92 0.86 1" contype="0" conaffinity="0"/>

    {left_arm}
    {right_arm}

    <body name="box" pos="0.62 -0.03 0.054">
      <freejoint/>
      <inertial mass="0.1" pos="0 0 0" diaginertia="1e-4 1e-4 1e-4"/>
      <geom type="box" name="box" size="0.024 0.024 0.030" condim="3"
       friction="1 0.03 0.003" material="cube_red" contype="1" conaffinity="1" solref="0.01 1"/>
    </body>
    <body name="target_cube_blue" pos="0.68 0.09 0.052">
      <freejoint/>
      <inertial mass="0.1" pos="0 0 0" diaginertia="1e-4 1e-4 1e-4"/>
      <geom type="box" name="target_cube_blue" size="0.024 0.024 0.026" condim="3"
       friction="1 0.03 0.003" material="cube_blue" contype="1" conaffinity="1" solref="0.01 1"/>
    </body>
    <body name="target_can_yellow" pos="0.56 0.14 0.062">
      <freejoint/>
      <inertial mass="0.1" pos="0 0 0" diaginertia="1e-4 1e-4 1e-4"/>
      <geom type="capsule" name="target_can_yellow" size="0.022 0.024" condim="3"
       friction="1 0.03 0.003" material="can_yellow" contype="1" conaffinity="1" solref="0.01 1"/>
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
    <key name="bimanual_ready" qpos="{l_qpos} {r_qpos} {cube_qpos} {cube2_qpos} {can_qpos}" ctrl="{l_ctrl} {r_ctrl}"/>
  </keyframe>
</mujoco>
"""


def assign_robotwin_semantic(m: mujoco.MjModel, gids: Sequence[int]) -> Tuple[int, ...]:
  """1=floor/bench, 2=left_arm, 3=right_arm, 4=targets/props, 5=bins, 6=tools/rack."""
  out: List[int] = []
  for gid in gids:
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(gid)) or ""
    if name in ("floor", "workbench_top", "workbench_trim", "head_mast", "head_bar"):
      sem = 1
    elif name.startswith("L_"):
      sem = 2
    elif name.startswith("R_"):
      sem = 3
    elif (
        name in ("box", "mug_ellipsoid", "bottle_teal", "bottle_teal_cap", "bottle_amber", "bottle_amber_cap", "spray_can")
        or name.startswith(("target_", "prop_"))
    ):
      sem = 4
    elif name.startswith("bin_"):
      sem = 5
    else:
      sem = 6
    out.append(sem)
  return tuple(out)


def _add_dojo_arm_xml(lines: List[str], prefix: str, base_x: float, base_y: float, yaw: float) -> None:
  lines.append(f'<body name="{prefix}_base" pos="{base_x:.3f} {base_y:.3f} 0.0" euler="0 0 {yaw:.4f}">')
  lines.append(f'  <geom name="{prefix}_collar0" type="cylinder" pos="0 0 0.08" size="0.095 0.08" rgba="0.22 0.24 0.27 1" contype="0" conaffinity="0"/>')
  lines.append(f'  <geom name="{prefix}_ped" type="cylinder" pos="0 0 0.21" size="0.068 0.07" rgba="0.93 0.94 0.96 1" contype="0" conaffinity="0"/>')
  lines.append(f'  <body name="{prefix}_j1" pos="0 0 0.30">')
  lines.append(f'    <inertial mass="2.0" pos="0 0 0.05" diaginertia="0.01 0.01 0.01"/>')
  lines.append(f'    <joint name="{prefix}_j1" type="hinge" axis="0 0 1" damping="20"/>')
  lines.append(f'    <geom name="{prefix}_link1" type="capsule" fromto="0 0 0 0 0 0.14" size="0.058" rgba="0.95 0.96 0.97 1" contype="0" conaffinity="0"/>')
  lines.append(f'    <geom name="{prefix}_joint1_col" type="sphere" pos="0 0 0.14" size="0.062" rgba="0.92 0.38 0.12 1" contype="0" conaffinity="0"/>')
  lines.append(f'    <body name="{prefix}_j2" pos="0 0 0.14">')
  lines.append(f'      <inertial mass="1.8" pos="0.14 0 0" diaginertia="0.01 0.01 0.01"/>')
  lines.append(f'      <joint name="{prefix}_j2" type="hinge" axis="0 1 0" damping="20"/>')
  lines.append(f'      <geom name="{prefix}_link2" type="capsule" fromto="0 0 0 0.28 0 0.08" size="0.050" rgba="0.95 0.96 0.97 1" contype="0" conaffinity="0"/>')
  lines.append(f'      <geom name="{prefix}_collar2" type="cylinder" pos="0.28 0 0.08" euler="1.5708 0 0" size="0.052 0.04" rgba="0.20 0.22 0.25 1" contype="0" conaffinity="0"/>')
  lines.append(f'      <body name="{prefix}_j3" pos="0.28 0 0.08">')
  lines.append(f'        <inertial mass="1.4" pos="0.12 0 -0.05" diaginertia="0.008 0.008 0.008"/>')
  lines.append(f'        <joint name="{prefix}_j3" type="hinge" axis="0 1 0" damping="15"/>')
  lines.append(f'        <geom name="{prefix}_link3" type="capsule" fromto="0 0 0 0.24 0 -0.10" size="0.042" rgba="0.88 0.90 0.93 1" contype="0" conaffinity="0"/>')
  lines.append(f'        <geom name="{prefix}_joint3_col" type="sphere" pos="0.24 0 -0.10" size="0.044" rgba="0.14 0.50 0.80 1" contype="0" conaffinity="0"/>')
  lines.append(f'        <body name="{prefix}_wrist" pos="0.24 0 -0.10">')
  lines.append(f'          <inertial mass="0.8" pos="0.05 0 -0.04" diaginertia="0.003 0.003 0.003"/>')
  lines.append(f'          <joint name="{prefix}_j4" type="hinge" axis="0 0 1" damping="5"/>')
  lines.append(f'          <geom name="{prefix}_palm_collar" type="box" pos="0.04 0 -0.03" size="0.028 0.085 0.024" rgba="0.24 0.26 0.29 1" contype="0" conaffinity="0"/>')
  lines.append(f'          <geom name="{prefix}_finger_l" type="box" pos="0.04 -0.055 -0.085" size="0.012 0.010 0.035" rgba="0.85 0.88 0.92 1" contype="0" conaffinity="0"/>')
  lines.append(f'          <geom name="{prefix}_finger_r" type="box" pos="0.04 0.055 -0.085" size="0.012 0.010 0.035" rgba="0.85 0.88 0.92 1" contype="0" conaffinity="0"/>')
  lines.append('        </body>')
  lines.append('      </body>')
  lines.append('    </body>')
  lines.append('  </body>')
  lines.append('</body>')


def make_robodojo_cluttered_xml(n_target_prims: int = 200) -> str:
  """RoboDojo Multi-Task Cluttered Manipulation & Warehouse Bench (G=55..1,004 moving primitives)."""
  palette = [
      "0.86 0.18 0.16 1", "0.15 0.52 0.82 1", "0.94 0.72 0.14 1",
      "0.18 0.68 0.38 1", "0.92 0.44 0.14 1", "0.62 0.30 0.74 1",
      "0.14 0.70 0.72 1", "0.90 0.88 0.84 1", "0.74 0.56 0.36 1",
  ]
  lines: List[str] = [
      '<mujoco model="robodojo_cluttered_bench">',
      '  <compiler angle="radian" autolimits="true"/>',
      '  <statistic center="0.0 0.0 0.5" extent="2.5"/>',
      '  <option timestep="0.005" iterations="3" ls_iterations="4" integrator="implicitfast">',
      '    <flag eulerdamp="disable"/>',
      '  </option>',
      '  <visual>',
      '    <headlight diffuse="0.42 0.42 0.45" ambient="0.22 0.22 0.24" specular="0.15 0.15 0.15"/>',
      '  </visual>',
      '  <asset>',
      '    <texture type="skybox" builtin="gradient" rgb1="0.34 0.52 0.72" rgb2="0.08 0.10 0.14" width="128" height="128"/>',
      '    <texture type="2d" name="floor_tex" builtin="checker" mark="edge" rgb1="0.28 0.32 0.36" rgb2="0.20 0.24 0.28" markrgb="0.55 0.60 0.65" width="128" height="128"/>',
      '    <material name="floor_mat" texture="floor_tex" texuniform="true" texrepeat="6 6" reflectance="0.15"/>',
      '  </asset>',
      '  <worldbody>',
      '    <light name="dojo_key" pos="1.4 -1.6 3.6" dir="-0.32 0.38 -0.88" directional="false" castshadow="true" diffuse="0.92 0.90 0.85" specular="0.30 0.30 0.30"/>',
      '    <camera name="overview" pos="1.75 -2.15 1.48" xyaxes="0.77 0.64 0.0 -0.28 0.34 0.90" fovy="48"/>',
      '    <camera name="closeup" pos="0.92 -1.10 0.92" xyaxes="0.77 0.64 0.0 -0.28 0.34 0.90" fovy="44"/>',
      '    <geom name="floor" type="plane" size="0 0 0.05" material="floor_mat" contype="1" conaffinity="1"/>',
      '    <geom name="workbench_top" type="box" pos="0.0 0.0 0.38" size="0.72 0.44 0.025" rgba="0.78 0.68 0.54 1" contype="1" conaffinity="1"/>',
      '    <geom name="bench_leg1" type="box" pos="-0.64 -0.36 0.18" size="0.035 0.035 0.18" rgba="0.24 0.26 0.28 1" contype="0" conaffinity="0"/>',
      '    <geom name="bench_leg2" type="box" pos="0.64 -0.36 0.18" size="0.035 0.035 0.18" rgba="0.24 0.26 0.28 1" contype="0" conaffinity="0"/>',
      '    <geom name="bench_leg3" type="box" pos="-0.64 0.36 0.18" size="0.035 0.035 0.18" rgba="0.24 0.26 0.28 1" contype="0" conaffinity="0"/>',
      '    <geom name="bench_leg4" type="box" pos="0.64 0.36 0.18" size="0.035 0.035 0.18" rgba="0.24 0.26 0.28 1" contype="0" conaffinity="0"/>',
  ]
  prim_count = 6

  _add_dojo_arm_xml(lines, "left_arm", -0.58, -0.55, 0.95)
  _add_dojo_arm_xml(lines, "right_arm", 0.58, -0.55, 2.20)
  prim_count += 18

  rack_specs = [("rack_back", 0.0, 0.98, 0.0), ("rack_left", -1.25, 0.25, 1.35), ("rack_right", 1.25, 0.25, -1.35)]
  for rname, rx, ry, ryaw in rack_specs:
    lines.append(f'    <body name="{rname}" pos="{rx:.3f} {ry:.3f} 0.0" euler="0 0 {ryaw:.4f}">')
    for sx in (-0.65, 0.65):
      for sy in (-0.22, 0.22):
        lines.append(f'      <geom name="{rname}_post_{sx:.1f}_{sy:.1f}" type="cylinder" pos="{sx:.2f} {sy:.2f} 0.62" size="0.022 0.62" rgba="0.28 0.30 0.34 1" contype="0" conaffinity="0"/>')
        prim_count += 1
    for sz in (0.18, 0.58, 0.98):
      lines.append(f'      <geom name="{rname}_shelf_{sz:.2f}" type="box" pos="0 0 {sz:.2f}" size="0.68 0.24 0.018" rgba="0.45 0.48 0.52 1" contype="0" conaffinity="0"/>')
      prim_count += 1
    lines.append('    </body>')

  active_objs = [
      ("cube_red", "box", "0.032 0.032 0.032", "-0.18 -0.08 0.44", palette[0]),
      ("cube_blue", "box", "0.030 0.030 0.035", "0.14 -0.10 0.44", palette[1]),
      ("can_yellow", "capsule", "0.026 0.028 0", "-0.02 0.06 0.46", palette[2]),
      ("bottle_green", "capsule", "0.024 0.050 0", "0.28 0.05 0.48", palette[3]),
      ("orange_ball", "sphere", "0.032 0 0", "-0.30 0.08 0.44", palette[4]),
      ("crate_plum", "box", "0.040 0.035 0.024", "0.04 -0.16 0.43", palette[5]),
  ]
  for oname, otype, osize, opos, orgba in active_objs:
    lines.append(f'    <body name="{oname}" pos="{opos}">')
    lines.append('      <freejoint/>')
    lines.append('      <inertial mass="0.12" pos="0 0 0" diaginertia="1e-4 1e-4 1e-4"/>')
    lines.append(f'      <geom name="{oname}_geom" type="{otype}" size="{osize}" rgba="{orgba}" contype="1" conaffinity="1"/>')
    lines.append('    </body>')
    prim_count += 1

  n_remaining = max(0, n_target_prims - prim_count)
  zones = [
      ("zone_bench", "slide", "1 0 0", 0.0, 0.05, 0.405),
      ("zone_back", "slide", "1 0 0", 0.0, 0.98, 0.0),
      ("zone_left", "hinge", "0 0 1", -1.25, 0.25, 0.0),
      ("zone_right", "hinge", "0 0 1", 1.25, 0.25, 0.0),
  ]
  per_zone = [n_remaining // 4] * 4
  for i in range(n_remaining % 4):
    per_zone[i] += 1

  idx_global = 0
  for z_idx, (zname, jtype, jaxis, zx, zy, zz) in enumerate(zones):
    cnt = per_zone[z_idx]
    lines.append(f'    <body name="{zname}" pos="{zx:.3f} {zy:.3f} {zz:.3f}">')
    lines.append('      <inertial mass="2.0" pos="0 0 0" diaginertia="0.1 0.1 0.1"/>')
    lines.append(f'      <joint name="{zname}_j" type="{jtype}" axis="{jaxis}" damping="10"/>')
    for k in range(cnt):
      rgba = palette[idx_global % len(palette)]
      t_sel = idx_global % 5
      if z_idx == 0:
        cols = 20
        gx = -0.58 + 1.16 * ((k % cols) + 0.5) / cols
        gy = -0.32 + 0.58 * (((k // cols) % 10) + 0.5) / 10.0
        gz = 0.030 + 0.045 * (k // (cols * 10))
        yaw = 0.37 * k
        sc = 0.65 if n_target_prims >= 500 else 1.0
      else:
        shelf_lvl = k % 3
        along = (k // 3)
        per_shelf = max(1, (cnt + 2) // 3)
        gx = -0.56 + 1.12 * ((along + 0.5) / per_shelf)
        gy = -0.10 + 0.20 * ((idx_global % 3) * 0.5)
        gz = [0.23, 0.63, 1.03][shelf_lvl]
        yaw = 0.51 * k
        sc = 0.85
      if t_sel == 0:
        gtype, gsize = "box", f"{0.024*sc:.4f} {0.020*sc:.4f} {0.022*sc:.4f}"
      elif t_sel == 1:
        gtype, gsize = "cylinder", f"{0.018*sc:.4f} {0.028*sc:.4f}"
      elif t_sel == 2:
        gtype, gsize = "capsule", f"{0.015*sc:.4f} {0.022*sc:.4f}"
      elif t_sel == 3:
        gtype, gsize = "sphere", f"{0.021*sc:.4f}"
      else:
        gtype, gsize = "ellipsoid", f"{0.022*sc:.4f} {0.018*sc:.4f} {0.026*sc:.4f}"
      lines.append(f'      <geom name="prop_{idx_global}" type="{gtype}" pos="{gx:.4f} {gy:.4f} {gz:.4f}" euler="0 0 {yaw:.3f}" size="{gsize}" rgba="{rgba}" contype="0" conaffinity="0"/>')
      idx_global += 1
    lines.append('    </body>')

  lines.append('  </worldbody>')
  lines.append('  <actuator>')
  for prefix in ("left_arm", "right_arm"):
    for j in ("j1", "j2", "j3", "j4"):
      lines.append(f'    <position name="{prefix}_{j}" joint="{prefix}_{j}" kp="80" ctrlrange="-1.5 1.5"/>')
  for zname, jtype, _, _, _, _ in zones:
    cr = "-0.12 0.12" if jtype == "slide" else "-0.25 0.25"
    lines.append(f'    <position name="{zname}_act" joint="{zname}_j" kp="100" ctrlrange="{cr}"/>')
  lines.append('  </actuator>')
  lines.append('</mujoco>')
  return "\n".join(lines)


def assign_robodojo_semantic(m: mujoco.MjModel, gids: Sequence[int]) -> Tuple[int, ...]:
  """1=floor/bench, 2=left_arm, 3=right_arm, 4=grasp_props, 5=racks."""
  out: List[int] = []
  for gid in gids:
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(gid)) or ""
    if name.startswith("left_arm"):
      sem = 2
    elif name.startswith("right_arm"):
      sem = 3
    elif name.startswith(("cube_", "can_", "bottle_", "orange_", "crate_", "prop_")):
      sem = 4
    elif name.startswith("rack_"):
      sem = 5
    else:
      sem = 1
    out.append(sem)
  return tuple(out)
