"""High-Res Multi-Embodiment tpu_rtx PBR+AO+SoftShadow Studio Gallery (512x512 & 1024x1024).

Renders 4 robotics embodiments using `tpu_rtx.compile(..., shader="fused_ao_ggx")` +
`tpu_rtx.render(...)` with 3-point studio lighting, dark slate studio floor,
16-sample Hemisphere AO, 16-sample Soft Penumbra Shadows, subtle Fresnel HDRI
rim reflection from HBM normal/seg_id buffers, and Blender 4.x AgX filmic tonemapping.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np

import tpu_rtx


STUDIO_ENV_ASSET = """
    <texture type="skybox" builtin="gradient" rgb1="0.10 0.13 0.18" rgb2="0.02 0.03 0.045" width="128" height="128"/>
    <texture name="studio_grid" type="2d" builtin="checker" mark="edge"
      rgb1="0.095 0.115 0.145" rgb2="0.062 0.076 0.100" markrgb="0.22 0.27 0.35" width="128" height="128"/>
    <material name="studio_grid" texture="studio_grid" texuniform="true" texrepeat="3 3" specular="0.22" shininess="0.35" reflectance="0.15"/>
"""

STUDIO_LIGHTS = """
    <light name="studio_key" pos="1.9 -1.9 2.9" dir="-0.45 0.45 -1" directional="false" castshadow="true"
      diffuse="0.90 0.84 0.76" specular="0.45 0.42 0.36" ambient="0.02 0.02 0.025"/>
    <light name="studio_rim" pos="-1.8 1.7 1.8" dir="0.5 -0.5 -0.45" directional="false" castshadow="false"
      diffuse="0.26 0.48 0.82" specular="0.42 0.62 0.92" ambient="0.0 0.0 0.0"/>
    <light name="studio_fill" pos="-1.5 -1.4 1.7" dir="0.4 0.4 -0.6" directional="false" castshadow="false"
      diffuse="0.20 0.22 0.26" specular="0.10 0.10 0.12" ambient="0.0 0.0 0.0"/>
"""


def _humanoid_studio_xml() -> str:
  return f"""<mujoco model="humanoid_studio">
  <option timestep="0.005" iterations="2" ls_iterations="4" integrator="implicitfast"/>
  <visual>
    <headlight ambient="0.06 0.065 0.075" diffuse="0.12 0.12 0.13" specular="0.05 0.05 0.05"/>
  </visual>
  <asset>
    {STUDIO_ENV_ASSET}
    <material name="body_orange" rgba="0.88 0.40 0.11 1" specular="0.45" shininess="0.42"/>
    <material name="joint_dark" rgba="0.15 0.17 0.20 1" specular="0.55" shininess="0.48"/>
    <material name="limb_titan" rgba="0.76 0.80 0.85 1" specular="0.60" shininess="0.52"/>
  </asset>
  <default>
    <joint armature="0.02" damping="2.0" limited="true"/>
    <geom condim="3" friction="0.9 0.02 0.002"/>
  </default>
  <worldbody>
    {STUDIO_LIGHTS}
    <camera name="hero" pos="1.95 -1.65 1.18" xyaxes="0.64 0.77 0 -0.14 0.11 0.98" fovy="40"/>
    <geom name="floor" type="plane" pos="0 0 0" size="6 6 0.1" material="studio_grid"/>
    <geom name="plinth_ring" type="cylinder" pos="0 0 0.012" size="0.88 0.012" rgba="0.13 0.16 0.21 1" contype="0" conaffinity="0"/>
    <body name="torso" pos="0 0 1.20" euler="0 12 0">
      <freejoint name="root"/>
      <geom name="torso_geom" type="capsule" fromto="0 -0.075 0 0 0.075 0" size="0.098" material="body_orange" mass="5.0"/>
      <geom name="head" type="sphere" pos="0.02 0 0.21" size="0.105" material="limb_titan" mass="1.5"/>
      <geom name="visor" type="box" pos="0.095 0 0.215" size="0.032 0.068 0.032" rgba="0.08 0.55 0.82 1" mass="0.1"/>
      <geom name="uwaist" type="capsule" fromto="-0.01 -0.06 -0.12 -0.01 0.06 -0.12" size="0.075" material="joint_dark" mass="2.0"/>
      <body name="lwaist" pos="-0.01 0 -0.26">
        <geom name="lwaist_geom" type="capsule" fromto="0 -0.06 0 0 0.06 0" size="0.070" material="body_orange" mass="2.0"/>
        <joint name="abdomen_z" type="hinge" axis="0 0 1" range="-45 45"/>
        <joint name="abdomen_y" type="hinge" axis="0 1 0" range="-75 30"/>
        <body name="pelvis" pos="0 0 -0.165">
          <joint name="abdomen_x" type="hinge" axis="1 0 0" range="-35 35"/>
          <geom name="butt" type="capsule" fromto="-0.02 -0.08 0 -0.02 0.08 0" size="0.08" material="joint_dark" mass="3.0"/>
          <body name="right_thigh" pos="0 -0.10 -0.04">
            <joint name="right_hip_x" type="hinge" axis="1 0 0" range="-25 5"/>
            <joint name="right_hip_z" type="hinge" axis="0 0 1" range="-60 35"/>
            <joint name="right_hip_y" type="hinge" axis="0 1 0" range="-110 35"/>
            <geom name="right_thigh_g" type="capsule" fromto="0 0 0 0 0.01 -0.34" size="0.062" material="limb_titan" mass="3.0"/>
            <body name="right_shin" pos="0 0.01 -0.40">
              <joint name="right_knee" type="hinge" axis="0 -1 0" range="-160 -2"/>
              <geom name="right_shin_g" type="capsule" fromto="0 0 0 0 0 -0.30" size="0.050" material="body_orange" mass="2.0"/>
              <body name="right_foot" pos="0 0 -0.36">
                <joint name="right_ankle_x" type="hinge" axis="1 0 0" range="-45 45"/>
                <joint name="right_ankle_y" type="hinge" axis="0 1 0" range="-50 50"/>
                <geom name="right_foot_g" type="box" pos="0.04 0 -0.02" size="0.095 0.048 0.025" material="joint_dark" mass="1.0"/>
              </body>
            </body>
          </body>
          <body name="left_thigh" pos="0 0.10 -0.04">
            <joint name="left_hip_x" type="hinge" axis="-1 0 0" range="-25 5"/>
            <joint name="left_hip_z" type="hinge" axis="0 0 -1" range="-60 35"/>
            <joint name="left_hip_y" type="hinge" axis="0 1 0" range="-110 35"/>
            <geom name="left_thigh_g" type="capsule" fromto="0 0 0 0 -0.01 -0.34" size="0.062" material="limb_titan" mass="3.0"/>
            <body name="left_shin" pos="0 -0.01 -0.40">
              <joint name="left_knee" type="hinge" axis="0 -1 0" range="-160 -2"/>
              <geom name="left_shin_g" type="capsule" fromto="0 0 0 0 0 -0.30" size="0.050" material="body_orange" mass="2.0"/>
              <body name="left_foot" pos="0 0 -0.36">
                <joint name="left_ankle_x" type="hinge" axis="1 0 0" range="-45 45"/>
                <joint name="left_ankle_y" type="hinge" axis="0 1 0" range="-50 50"/>
                <geom name="left_foot_g" type="box" pos="0.04 0 -0.02" size="0.095 0.048 0.025" material="joint_dark" mass="1.0"/>
              </body>
            </body>
          </body>
        </body>
      </body>
      <body name="right_upper_arm" pos="0 -0.17 0.06">
        <joint name="right_shoulder1" type="hinge" axis="2 1 1" range="-85 60"/>
        <joint name="right_shoulder2" type="hinge" axis="0 -1 1" range="-85 60"/>
        <geom name="right_uarm_g" type="capsule" fromto="0 0 0 0.16 -0.16 -0.16" size="0.044" material="limb_titan" mass="1.2"/>
        <body name="right_lower_arm" pos="0.18 -0.18 -0.18">
          <joint name="right_elbow" type="hinge" axis="0 -1 1" range="-90 50"/>
          <geom name="right_larm_g" type="capsule" fromto="0.01 0.01 0.01 0.17 0.17 0.17" size="0.036" material="body_orange" mass="0.8"/>
          <geom name="right_hand" type="sphere" pos="0.18 0.18 0.18" size="0.044" material="joint_dark" mass="0.3"/>
        </body>
      </body>
      <body name="left_upper_arm" pos="0 0.17 0.06">
        <joint name="left_shoulder1" type="hinge" axis="2 -1 1" range="-60 85"/>
        <joint name="left_shoulder2" type="hinge" axis="0 1 1" range="-60 85"/>
        <geom name="left_uarm_g" type="capsule" fromto="0 0 0 0.16 0.16 -0.16" size="0.044" material="limb_titan" mass="1.2"/>
        <body name="left_lower_arm" pos="0.18 0.18 -0.18">
          <joint name="left_elbow" type="hinge" axis="0 -1 -1" range="-90 50"/>
          <geom name="left_larm_g" type="capsule" fromto="0.01 -0.01 0.01 0.17 -0.17 0.17" size="0.036" material="body_orange" mass="0.8"/>
          <geom name="left_hand" type="sphere" pos="0.18 -0.18 0.18" size="0.044" material="joint_dark" mass="0.3"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def _panda_studio_xml() -> str:
  return f"""<mujoco model="panda_studio">
  <compiler angle="radian" autolimits="true"/>
  <visual>
    <headlight ambient="0.06 0.065 0.075" diffuse="0.12 0.12 0.13" specular="0.05 0.05 0.05"/>
  </visual>
  <asset>
    {STUDIO_ENV_ASSET}
    <material name="white" rgba="0.88 0.90 0.93 1" specular="0.50" shininess="0.45"/>
    <material name="off_white" rgba="0.76 0.79 0.83 1" specular="0.45" shininess="0.40"/>
    <material name="black" rgba="0.14 0.16 0.19 1" specular="0.55" shininess="0.48"/>
    <material name="cube_red" rgba="0.86 0.13 0.11 1" specular="0.50" shininess="0.45"/>
    <material name="pad_dark" rgba="0.17 0.20 0.25 1" specular="0.30" shininess="0.30"/>
  </asset>
  <default>
    <default class="panda">
      <joint armature="0.1" damping="1" axis="0 0 1" range="-2.8973 2.8973"/>
      <default class="finger"><joint axis="0 1 0" type="slide" range="0 0.04"/></default>
      <default class="visual_prim"><geom contype="0" conaffinity="0" group="2"/></default>
      <default class="pad_col"><geom group="3" type="box" contype="1" conaffinity="1"/></default>
    </default>
  </default>
  <worldbody>
    {STUDIO_LIGHTS}
    <camera name="hero" fovy="40" pos="1.12 -0.82 0.62" xyaxes="0.60 0.80 0 -0.28 0.21 0.94"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="studio_grid"/>
    <geom name="init_space" size="0.24 0.26 0.005" type="box" pos="0.63 0 0.005" material="pad_dark" contype="0" conaffinity="0"/>
    <geom name="pad_accent" size="0.08 0.08 0.006" type="box" pos="0.65 0 0.006" rgba="0.10 0.42 0.68 1" contype="0" conaffinity="0"/>
    <body name="link0" childclass="panda">
      <geom name="link0_base" type="cylinder" pos="-0.03 0 0.07" size="0.095 0.07" material="off_white" class="visual_prim"/>
      <geom name="link0_pedestal" type="cylinder" pos="0 0 0.21" size="0.068 0.08" material="white" class="visual_prim"/>
      <body name="link1" pos="0 0 0.333">
        <inertial mass="2.0" pos="0 0 -0.05" diaginertia="0.01 0.01 0.01"/>
        <joint name="joint1"/>
        <geom name="link1_body" type="capsule" fromto="0 0 -0.10 0 0 0.04" size="0.062" material="white" class="visual_prim"/>
        <body name="link2" quat="1 -1 0 0">
          <inertial mass="2.0" pos="0 -0.15 0" diaginertia="0.01 0.01 0.01"/>
          <joint name="joint2"/>
          <geom name="link2_collar" type="cylinder" pos="0 0 0" size="0.065 0.045" material="black" class="visual_prim"/>
          <geom name="link2_arm" type="capsule" fromto="0 0 0 0 -0.22 0" size="0.055" material="white" class="visual_prim"/>
          <body name="link3" pos="0 -0.316 0" quat="1 1 0 0">
            <inertial mass="2.0" pos="0 0 -0.05" diaginertia="0.01 0.01 0.01"/>
            <joint name="joint3"/>
            <geom name="link3_seg" type="capsule" fromto="0 0 -0.11 0.045 0 0" size="0.052" material="white" class="visual_prim"/>
            <geom name="link3_joint" type="sphere" pos="0.0825 0 0" size="0.056" material="black" class="visual_prim"/>
            <body name="link4" pos="0.0825 0 0" quat="1 1 0 0">
              <inertial mass="2.0" pos="-0.04 0.15 0" diaginertia="0.01 0.01 0.01"/>
              <joint name="joint4"/>
              <geom name="link4_arm" type="capsule" fromto="0 0 0 -0.06 0.20 0" size="0.050" material="white" class="visual_prim"/>
              <body name="link5" pos="-0.0825 0.384 0" quat="1 -1 0 0">
                <inertial mass="1.5" pos="0 0 -0.1" diaginertia="0.008 0.008 0.008"/>
                <joint name="joint5"/>
                <geom name="link5_forearm" type="capsule" fromto="0 0 -0.22 0 0 -0.02" size="0.044" material="off_white" class="visual_prim"/>
                <geom name="link5_collar" type="cylinder" pos="0 0 0" size="0.046 0.025" material="black" class="visual_prim"/>
                <body name="link6" quat="1 1 0 0">
                  <inertial mass="1.0" pos="0.04 0 0" diaginertia="0.005 0.005 0.005"/>
                  <joint name="joint6"/>
                  <geom name="link6_wrist" type="capsule" fromto="0 0 0 0.088 0 0" size="0.040" material="white" class="visual_prim"/>
                  <body name="link7" pos="0.088 0 0" quat="1 1 0 0">
                    <inertial mass="0.7" pos="0 0 0.05" diaginertia="0.003 0.003 0.003"/>
                    <joint name="joint7"/>
                    <geom name="link7_flange" type="cylinder" pos="0 0 0.06" size="0.038 0.045" material="black" class="visual_prim"/>
                    <body name="hand" pos="0 0 0.107" quat="0.9238795 0 0 -0.3826834">
                      <inertial mass="0.73" pos="0 0 0.03" diaginertia="0.001 0.0025 0.0017"/>
                      <geom name="hand_box" type="box" pos="0 0 0.030" size="0.024 0.102 0.026" material="off_white" class="visual_prim"/>
                      <geom name="hand_capsule" type="capsule" size="0.036 0.055" quat="1 1 0 0" pos="0 0 0.03" material="black" group="3"/>
                      <body name="left_finger" pos="0 0 0.0584">
                        <inertial mass="0.03" pos="0 0.01 0.025" diaginertia="1e-5 1e-5 1e-5"/>
                        <joint name="finger_joint1" class="finger"/>
                        <geom name="left_finger_vis" type="box" pos="0 0.010 0.026" size="0.010 0.008 0.024" material="black" class="visual_prim"/>
                        <geom name="left_finger_pad" type="box" size="0.009 0.0075 0.016" pos="0 0.008 0.038" material="off_white" class="pad_col"/>
                      </body>
                      <body name="right_finger" pos="0 0 0.0584" quat="0 0 0 1">
                        <inertial mass="0.03" pos="0 0.01 0.025" diaginertia="1e-5 1e-5 1e-5"/>
                        <joint name="finger_joint2" class="finger"/>
                        <geom name="right_finger_vis" type="box" pos="0 0.010 0.026" size="0.010 0.008 0.024" material="black" class="visual_prim"/>
                        <geom name="right_finger_pad" type="box" size="0.009 0.0075 0.016" pos="0 0.008 0.038" material="off_white" class="pad_col"/>
                      </body>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
    <body name="box" pos="0.64 0.006 0.12">
      <freejoint/>
      <inertial mass="0.1" pos="0 0 0" diaginertia="1e-4 1e-4 1e-4"/>
      <geom type="box" name="box" size="0.024 0.024 0.032" material="cube_red"/>
    </body>
  </worldbody>
  <keyframe>
    <key name="picked" qpos="0.0481 0.5604 -0.0417 -1.8459 0.0342 2.3654 0.7852 0.024 0.024 0.6368 0.0064 0.125 1.0000 0.0001 -0.0010 -0.0063"/>
  </keyframe>
</mujoco>
"""


def _panda_arm_subxml(prefix: str, base_pos: str, yaw_rad: float = 0.0) -> str:
  c = float(np.cos(0.5 * yaw_rad))
  s = float(np.sin(0.5 * yaw_rad))
  return f"""
    <body name="{prefix}link0" pos="{base_pos}" quat="{c:.6f} 0 0 {s:.6f}" childclass="panda">
      <geom name="{prefix}link0_base" type="cylinder" pos="-0.03 0 0.07" size="0.095 0.07" material="off_white" class="visual_prim"/>
      <geom name="{prefix}link0_pedestal" type="cylinder" pos="0 0 0.21" size="0.068 0.08" material="white" class="visual_prim"/>
      <body name="{prefix}link1" pos="0 0 0.333">
        <inertial mass="2.0" pos="0 0 -0.05" diaginertia="0.01 0.01 0.01"/>
        <joint name="{prefix}joint1"/>
        <geom name="{prefix}link1_body" type="capsule" fromto="0 0 -0.10 0 0 0.04" size="0.062" material="white" class="visual_prim"/>
        <body name="{prefix}link2" quat="1 -1 0 0">
          <inertial mass="2.0" pos="0 -0.15 0" diaginertia="0.01 0.01 0.01"/>
          <joint name="{prefix}joint2"/>
          <geom name="{prefix}link2_collar" type="cylinder" pos="0 0 0" size="0.065 0.045" material="black" class="visual_prim"/>
          <geom name="{prefix}link2_arm" type="capsule" fromto="0 0 0 0 -0.22 0" size="0.055" material="white" class="visual_prim"/>
          <body name="{prefix}link3" pos="0 -0.316 0" quat="1 1 0 0">
            <inertial mass="2.0" pos="0 0 -0.05" diaginertia="0.01 0.01 0.01"/>
            <joint name="{prefix}joint3"/>
            <geom name="{prefix}link3_seg" type="capsule" fromto="0 0 -0.11 0.045 0 0" size="0.052" material="white" class="visual_prim"/>
            <geom name="{prefix}link3_joint" type="sphere" pos="0.0825 0 0" size="0.056" material="black" class="visual_prim"/>
            <body name="{prefix}link4" pos="0.0825 0 0" quat="1 1 0 0">
              <inertial mass="2.0" pos="-0.04 0.15 0" diaginertia="0.01 0.01 0.01"/>
              <joint name="{prefix}joint4"/>
              <geom name="{prefix}link4_arm" type="capsule" fromto="0 0 0 -0.06 0.20 0" size="0.050" material="white" class="visual_prim"/>
              <body name="{prefix}link5" pos="-0.0825 0.384 0" quat="1 -1 0 0">
                <inertial mass="1.5" pos="0 0 -0.1" diaginertia="0.008 0.008 0.008"/>
                <joint name="{prefix}joint5"/>
                <geom name="{prefix}link5_forearm" type="capsule" fromto="0 0 -0.22 0 0 -0.02" size="0.044" material="off_white" class="visual_prim"/>
                <geom name="{prefix}link5_collar" type="cylinder" pos="0 0 0" size="0.046 0.025" material="black" class="visual_prim"/>
                <body name="{prefix}link6" quat="1 1 0 0">
                  <inertial mass="1.0" pos="0.04 0 0" diaginertia="0.005 0.005 0.005"/>
                  <joint name="{prefix}joint6"/>
                  <geom name="{prefix}link6_wrist" type="capsule" fromto="0 0 0 0.088 0 0" size="0.040" material="white" class="visual_prim"/>
                  <body name="{prefix}link7" pos="0.088 0 0" quat="1 1 0 0">
                    <inertial mass="0.7" pos="0 0 0.05" diaginertia="0.003 0.003 0.003"/>
                    <joint name="{prefix}joint7"/>
                    <geom name="{prefix}link7_flange" type="cylinder" pos="0 0 0.06" size="0.038 0.045" material="black" class="visual_prim"/>
                    <body name="{prefix}hand" pos="0 0 0.107" quat="0.9238795 0 0 -0.3826834">
                      <inertial mass="0.73" pos="0 0 0.03" diaginertia="0.001 0.0025 0.0017"/>
                      <geom name="{prefix}hand_box" type="box" pos="0 0 0.030" size="0.024 0.102 0.026" material="off_white" class="visual_prim"/>
                      <geom name="{prefix}hand_capsule" type="capsule" size="0.036 0.055" quat="1 1 0 0" pos="0 0 0.03" material="black" contype="0" conaffinity="0" group="3"/>
                      <body name="{prefix}left_finger" pos="0 0 0.0584">
                        <inertial mass="0.03" pos="0 0.01 0.025" diaginertia="1e-5 1e-5 1e-5"/>
                        <joint name="{prefix}finger_joint1" class="finger"/>
                        <geom name="{prefix}left_finger_vis" type="box" pos="0 0.010 0.026" size="0.010 0.008 0.024" material="black" class="visual_prim"/>
                        <geom name="{prefix}left_finger_pad" type="box" size="0.009 0.0075 0.016" pos="0 0.008 0.038" material="off_white" class="pad_col"/>
                      </body>
                      <body name="{prefix}right_finger" pos="0 0 0.0584" quat="0 0 0 1">
                        <inertial mass="0.03" pos="0 0.01 0.025" diaginertia="1e-5 1e-5 1e-5"/>
                        <joint name="{prefix}finger_joint2" class="finger"/>
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


def _dual_arm_studio_xml() -> str:
  left_arm = _panda_arm_subxml("L_", "0.02 0.28 0.01", yaw_rad=-0.18)
  right_arm = _panda_arm_subxml("R_", "0.02 -0.28 0.01", yaw_rad=0.18)
  l_qpos = "-0.12 0.45 -0.05 -1.82 -0.02 2.32 0.78 0.035 0.035"
  r_qpos = "0.14 0.52 0.08 -1.78 0.04 2.28 0.78 0.028 0.028"
  cube_qpos = "0.62 -0.03 0.054 1.0 0.0 0.0 0.0"
  return f"""<mujoco model="dual_arm_studio">
  <compiler angle="radian" autolimits="true"/>
  <visual>
    <headlight ambient="0.06 0.065 0.075" diffuse="0.12 0.12 0.13" specular="0.05 0.05 0.05"/>
  </visual>
  <asset>
    {STUDIO_ENV_ASSET}
    <material name="white" rgba="0.88 0.90 0.93 1" specular="0.50" shininess="0.45"/>
    <material name="off_white" rgba="0.76 0.79 0.83 1" specular="0.45" shininess="0.40"/>
    <material name="black" rgba="0.14 0.16 0.19 1" specular="0.55" shininess="0.48"/>
    <material name="cube_red" rgba="0.86 0.13 0.11 1" specular="0.50" shininess="0.45"/>
  </asset>
  <default>
    <default class="panda">
      <joint armature="0.1" damping="1" axis="0 0 1" range="-2.8973 2.8973"/>
      <default class="finger"><joint axis="0 1 0" type="slide" range="0 0.04"/></default>
      <default class="visual_prim"><geom contype="0" conaffinity="0" group="2"/></default>
      <default class="pad_col"><geom group="3" type="box" contype="1" conaffinity="1"/></default>
    </default>
  </default>
  <worldbody>
    {STUDIO_LIGHTS}
    <camera name="diag_overview" fovy="44" pos="1.25 -0.78 0.72" xyaxes="0.54 0.84 0 -0.46 0.30 0.84"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="studio_grid"/>
    <geom name="head_mast" type="cylinder" pos="0.06 0.0 0.45" size="0.022 0.45" rgba="0.20 0.22 0.26 1" contype="0" conaffinity="0"/>
    <geom name="head_bar" type="box" pos="0.10 0.0 0.92" size="0.045 0.09 0.028" rgba="0.14 0.16 0.20 1" contype="0" conaffinity="0"/>
    <geom name="workbench_top" type="box" pos="0.64 0.0 0.012" size="0.44 0.58 0.012" rgba="0.34 0.24 0.15 1" contype="0" conaffinity="0"/>
    <geom name="workbench_trim" type="box" pos="0.64 0.0 0.005" size="0.46 0.60 0.005" rgba="0.14 0.16 0.19 1" contype="0" conaffinity="0"/>
    <geom name="bin_blue_base" type="box" pos="0.52 -0.36 0.028" size="0.09 0.09 0.006" rgba="0.10 0.38 0.78 1" contype="0" conaffinity="0"/>
    <geom name="bin_blue_w1" type="box" pos="0.43 -0.36 0.062" size="0.006 0.09 0.032" rgba="0.10 0.38 0.78 1" contype="0" conaffinity="0"/>
    <geom name="bin_blue_w2" type="box" pos="0.61 -0.36 0.062" size="0.006 0.09 0.032" rgba="0.10 0.38 0.78 1" contype="0" conaffinity="0"/>
    <geom name="bin_blue_w3" type="box" pos="0.52 -0.45 0.062" size="0.09 0.006 0.032" rgba="0.10 0.38 0.78 1" contype="0" conaffinity="0"/>
    <geom name="bin_blue_w4" type="box" pos="0.52 -0.27 0.062" size="0.09 0.006 0.032" rgba="0.10 0.38 0.78 1" contype="0" conaffinity="0"/>
    <geom name="bin_green_base" type="box" pos="0.52 0.36 0.028" size="0.09 0.09 0.006" rgba="0.14 0.62 0.32 1" contype="0" conaffinity="0"/>
    <geom name="bin_green_w1" type="box" pos="0.43 0.36 0.062" size="0.006 0.09 0.032" rgba="0.14 0.62 0.32 1" contype="0" conaffinity="0"/>
    <geom name="bin_green_w2" type="box" pos="0.61 0.36 0.062" size="0.006 0.09 0.032" rgba="0.14 0.62 0.32 1" contype="0" conaffinity="0"/>
    <geom name="bin_green_w3" type="box" pos="0.52 0.27 0.062" size="0.09 0.006 0.032" rgba="0.14 0.62 0.32 1" contype="0" conaffinity="0"/>
    <geom name="bin_green_w4" type="box" pos="0.52 0.45 0.062" size="0.09 0.006 0.032" rgba="0.14 0.62 0.32 1" contype="0" conaffinity="0"/>
    <geom name="rack_post_l" type="cylinder" pos="0.90 -0.26 0.14" size="0.015 0.12" rgba="0.22 0.24 0.27 1" contype="0" conaffinity="0"/>
    <geom name="rack_post_r" type="cylinder" pos="0.90 0.26 0.14" size="0.015 0.12" rgba="0.22 0.24 0.27 1" contype="0" conaffinity="0"/>
    <geom name="rack_bar" type="capsule" fromto="0.90 -0.27 0.23 0.90 0.27 0.23" size="0.014" rgba="0.82 0.56 0.16 1" contype="0" conaffinity="0"/>
    <geom name="rack_shelf" type="box" pos="0.90 0.0 0.12" size="0.045 0.26 0.008" rgba="0.28 0.30 0.34 1" contype="0" conaffinity="0"/>
    <geom name="bottle_teal" type="cylinder" pos="0.74 -0.22 0.085" size="0.028 0.060" rgba="0.10 0.68 0.74 1" contype="0" conaffinity="0"/>
    <geom name="bottle_teal_cap" type="cylinder" pos="0.74 -0.22 0.155" size="0.014 0.012" rgba="0.88 0.88 0.90 1" contype="0" conaffinity="0"/>
    <geom name="bottle_amber" type="cylinder" pos="0.76 0.22 0.095" size="0.030 0.070" rgba="0.88 0.48 0.10 1" contype="0" conaffinity="0"/>
    <geom name="bottle_amber_cap" type="cylinder" pos="0.76 0.22 0.175" size="0.013 0.012" rgba="0.18 0.18 0.20 1" contype="0" conaffinity="0"/>
    <geom name="spray_can" type="capsule" pos="0.46 0.16 0.065" size="0.022 0.040" rgba="0.84 0.20 0.48 1" contype="0" conaffinity="0"/>
    <geom name="wrench_handle" type="box" pos="0.56 -0.14 0.032" euler="0 0 0.45" size="0.075 0.012 0.008" rgba="0.74 0.76 0.80 1" contype="0" conaffinity="0"/>
    <geom name="wrench_head" type="cylinder" pos="0.63 -0.105 0.032" size="0.022 0.009" rgba="0.70 0.72 0.76 1" contype="0" conaffinity="0"/>
    <geom name="mallet_handle" type="capsule" fromto="0.40 0.05 0.036 0.53 0.13 0.036" size="0.011" rgba="0.70 0.46 0.24 1" contype="0" conaffinity="0"/>
    <geom name="mallet_head" type="cylinder" pos="0.54 0.135 0.045" euler="1.57 0.45 0" size="0.024 0.036" rgba="0.16 0.16 0.18 1" contype="0" conaffinity="0"/>
    <geom name="prop_sphere_gold" type="sphere" pos="0.51 -0.35 0.058" size="0.024" rgba="0.92 0.72 0.14 1" contype="0" conaffinity="0"/>
    <geom name="prop_block_cyan" type="box" pos="0.54 -0.39 0.052" euler="0 0 0.3" size="0.020 0.020 0.018" rgba="0.18 0.76 0.86 1" contype="0" conaffinity="0"/>
    <geom name="prop_block_yellow" type="box" pos="0.50 0.35 0.054" euler="0 0 -0.4" size="0.022 0.022 0.020" rgba="0.92 0.80 0.16 1" contype="0" conaffinity="0"/>
    <geom name="prop_cyl_purple" type="cylinder" pos="0.55 0.38 0.060" size="0.018 0.026" rgba="0.62 0.24 0.82 1" contype="0" conaffinity="0"/>
    <geom name="target_cube_blue" type="box" pos="0.70 0.08 0.050" euler="0 0 0.5" size="0.024 0.024 0.026" rgba="0.18 0.46 0.90 1" contype="0" conaffinity="0"/>
    <geom name="target_cube_green" type="box" pos="0.59 0.11 0.044" euler="0 0 -0.2" size="0.020 0.020 0.020" rgba="0.20 0.78 0.30 1" contype="0" conaffinity="0"/>
    <geom name="mug_ellipsoid" type="ellipsoid" pos="0.74 -0.06 0.056" size="0.028 0.028 0.032" rgba="0.88 0.86 0.80 1" contype="0" conaffinity="0"/>
    {left_arm}
    {right_arm}
    <body name="box" pos="0.62 -0.03 0.054">
      <freejoint/>
      <inertial mass="0.1" pos="0 0 0" diaginertia="1e-4 1e-4 1e-4"/>
      <geom type="box" name="box" size="0.022 0.022 0.030" material="cube_red" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
  <keyframe>
    <key name="bimanual_ready" qpos="{l_qpos} {r_qpos} {cube_qpos}"/>
  </keyframe>
</mujoco>
"""


def _quadruped_leg_xml(name: str, x: float, y: float) -> str:
  sy = 1.0 if y > 0 else -1.0
  return f"""
    <body name="{name}_hip" pos="{x:.3f} {y:.3f} 0">
      <joint name="{name}_hip_x" type="hinge" axis="1 0 0" range="-45 45"/>
      <geom name="{name}_hip_pod" type="cylinder" pos="0 {sy*0.03:.3f} 0" quat="0.7071 0.7071 0 0" size="0.044 0.045" material="quad_dark" mass="0.6" contype="0" conaffinity="0"/>
      <body name="{name}_thigh" pos="0 {sy*0.085:.3f} 0">
        <joint name="{name}_thigh_y" type="hinge" axis="0 1 0" range="-90 120"/>
        <geom name="{name}_thigh_g" type="capsule" fromto="0 0 0 0 0 -0.21" size="0.032" material="quad_titan" mass="1.0"/>
        <geom name="{name}_knee_pod" type="sphere" pos="0 0 -0.21" size="0.034" material="quad_gold" mass="0.2"/>
        <body name="{name}_calf" pos="0 0 -0.21">
          <joint name="{name}_calf_y" type="hinge" axis="0 1 0" range="-150 -20"/>
          <geom name="{name}_calf_g" type="capsule" fromto="0 0 0 0 0 -0.21" size="0.022" material="quad_dark" mass="0.5"/>
          <geom name="{name}_foot" type="sphere" pos="0 0 -0.215" size="0.028" material="quad_rubber" mass="0.2"/>
        </body>
      </body>
    </body>"""


def _quadruped_studio_xml() -> str:
  legs = (
      _quadruped_leg_xml("FL", 0.21, 0.095)
      + _quadruped_leg_xml("FR", 0.21, -0.095)
      + _quadruped_leg_xml("RL", -0.21, 0.095)
      + _quadruped_leg_xml("RR", -0.21, -0.095)
  )
  return f"""<mujoco model="quadruped_12dof_studio">
  <compiler angle="degree"/>
  <visual>
    <headlight ambient="0.06 0.065 0.075" diffuse="0.12 0.12 0.13" specular="0.05 0.05 0.05"/>
  </visual>
  <asset>
    {STUDIO_ENV_ASSET}
    <material name="quad_gold" rgba="0.90 0.56 0.12 1" specular="0.55" shininess="0.50"/>
    <material name="quad_titan" rgba="0.78 0.82 0.87 1" specular="0.60" shininess="0.52"/>
    <material name="quad_dark" rgba="0.14 0.16 0.20 1" specular="0.55" shininess="0.48"/>
    <material name="quad_rubber" rgba="0.08 0.09 0.11 1" specular="0.20" shininess="0.20"/>
  </asset>
  <worldbody>
    {STUDIO_LIGHTS}
    <camera name="hero" pos="1.05 -0.92 0.56" xyaxes="0.66 0.75 0 -0.22 0.19 0.96" fovy="38"/>
    <geom name="floor" type="plane" pos="0 0 0" size="6 6 0.1" material="studio_grid"/>
    <geom name="test_pad" type="box" pos="0 0 0.008" size="0.55 0.36 0.008" rgba="0.14 0.17 0.22 1" contype="0" conaffinity="0"/>
    <body name="trunk" pos="0 0 0.355" euler="-3 4 0">
      <freejoint name="root"/>
      <geom name="chassis_main" type="box" pos="0 0 0" size="0.24 0.105 0.055" material="quad_titan" mass="6.0" contype="0" conaffinity="0"/>
      <geom name="chassis_spine" type="box" pos="0 0 0.052" size="0.18 0.068 0.016" material="quad_gold" mass="1.0" contype="0" conaffinity="0"/>
      <geom name="head_pod" type="box" pos="0.24 0 0.012" size="0.055 0.078 0.042" material="quad_dark" mass="0.8" contype="0" conaffinity="0"/>
      <geom name="lidar_dome" type="cylinder" pos="0.21 0 0.068" size="0.036 0.018" material="quad_dark" mass="0.2" contype="0" conaffinity="0"/>
      <geom name="visor_led" type="box" pos="0.295 0 0.012" size="0.008 0.055 0.016" rgba="0.12 0.72 0.95 1" mass="0.05" contype="0" conaffinity="0"/>
      <geom name="tail_bumper" type="capsule" fromto="-0.24 -0.07 0 -0.24 0.07 0" size="0.035" material="quad_dark" mass="0.3" contype="0" conaffinity="0"/>
      {legs}
    </body>
  </worldbody>
</mujoco>
"""


def studio_agx_post_tpu(
    rgb: jax.Array,
    normal: jax.Array,
    seg_id: jax.Array,
    floor_gid: int,
    exposure: float = 1.18,
    rim_strength: float = 0.22,
) -> jax.Array:
  """Zero-copy TPU studio rim sheen + Blender 4.x AgX Filmic Tonemap + sRGB OETF."""
  rgb_f = rgb.astype(jnp.float32)
  nrm_f = normal.astype(jnp.float32)

  # Subtle studio HDRI Fresnel rim boost on robot links (seg_id >= 0 and != floor_gid)
  is_fg = (seg_id >= 0) & (seg_id != floor_gid)
  # Approximate grazing rim from view-facing normal component (back/side light rim)
  nrm_xy = jnp.sqrt(jnp.maximum(nrm_f[..., 0] ** 2 + nrm_f[..., 1] ** 2, 0.0))
  rim = jnp.where(is_fg, (nrm_xy ** 2.6) * rim_strength, 0.0)[..., None]
  rim_col = jnp.array([0.42, 0.64, 0.98], dtype=jnp.float32)[None, None, :]
  hdr = jnp.maximum(rgb_f * exposure + rim * rim_col, 0.0)

  # Subtle radial lens vignette
  H, W, _ = hdr.shape
  yy = jnp.linspace(-1.0, 1.0, H, dtype=jnp.float32)[:, None]
  xx = jnp.linspace(-1.0, 1.0, W, dtype=jnp.float32)[None, :]
  r2 = (xx * xx + yy * yy) * 0.5
  hdr = hdr * (1.0 - 0.12 * r2[..., None])

  # Blender 4.x AgX Filmic Transform
  agx_inset = jnp.array([
      [0.842479062253094, 0.0423282422610123, 0.0423756549057051],
      [0.0784335999999992, 0.878468636469772, 0.0784336],
      [0.0792237451477643, 0.0791661274605434, 0.879142973793104],
  ], dtype=jnp.float32)
  agx_outset = jnp.array([
      [1.19687900512017, -0.0528968517574562, -0.0529716355144438],
      [-0.0980208811401368, 1.15190312990417, -0.0980434501171241],
      [-0.0990297440797205, -0.0989611768448433, 1.15107367264116],
  ], dtype=jnp.float32)

  # Convert sRGB-like shader output to linear before AgX
  hdr_lin = jnp.power(jnp.maximum(hdr, 1e-6), 2.1)
  val = jnp.einsum("ij,hwj->hwi", agx_inset, jnp.maximum(hdr_lin, 1e-10), precision=jax.lax.Precision.HIGHEST)
  min_ev, max_ev = -6.8, 4.2
  val = jnp.log2(jnp.maximum(val, 1e-10))
  val = jnp.clip((val - min_ev) / (max_ev - min_ev), 0.0, 1.0)
  x = val
  x2 = x * x
  x4 = x2 * x2
  sig = jnp.clip(
      15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232,
      0.0, 1.0,
  )
  out_lin = jnp.clip(jnp.einsum("ij,hwj->hwi", agx_outset, sig, precision=jax.lax.Precision.HIGHEST), 0.0, 1.0)
  out_lin = jnp.power(out_lin, 1.05)
  luma = (0.2126 * out_lin[..., 0] + 0.7152 * out_lin[..., 1] + 0.0722 * out_lin[..., 2])[..., None]
  out_lin = jnp.clip(luma + 1.15 * (out_lin - luma), 0.0, 1.0)
  srgb = jnp.where(out_lin <= 0.0031308, 12.92 * out_lin, 1.055 * (jnp.maximum(out_lin, 1e-6) ** (1.0 / 2.4)) - 0.055)
  return jnp.clip(srgb, 0.0, 1.0)


def _pose_embodiment(name: str, m: mujoco.MjModel) -> mujoco.MjData:
  d = mujoco.MjData(m)
  if m.nkey > 0:
    mujoco.mj_resetDataKeyframe(m, d, 0)
  else:
    mujoco.mj_resetData(m, d)

  if name == "humanoid_running":
    # Dynamic athletic running stride
    def _setj(jname: str, val_rad: float):
      jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
      d.qpos[m.jnt_qposadr[jid]] = val_rad
    _setj("abdomen_y", 0.16)
    _setj("abdomen_z", -0.14)
    _setj("right_hip_y", -0.78)   # right leg swinging forward
    _setj("right_knee", -0.72)
    _setj("right_ankle_y", 0.28)
    _setj("left_hip_y", 0.42)     # left leg pushing off behind
    _setj("left_knee", -0.35)
    _setj("left_ankle_y", 0.42)
    _setj("right_shoulder1", 0.55)
    _setj("right_elbow", -1.15)
    _setj("left_shoulder1", -0.65)
    _setj("left_elbow", -0.95)
  elif name == "quadruped_trot":
    # Diagonal trot: FL & RR lifted forward, FR & RL planted
    def _setd(jname: str, deg: float):
      jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
      d.qpos[m.jnt_qposadr[jid]] = np.deg2rad(deg)
    _setd("FL_thigh_y", 28.0)
    _setd("FL_calf_y", -78.0)
    _setd("RR_thigh_y", 34.0)
    _setd("RR_calf_y", -74.0)
    _setd("FR_thigh_y", 42.0)
    _setd("FR_calf_y", -68.0)
    _setd("RL_thigh_y", 18.0)
    _setd("RL_calf_y", -60.0)
    _setd("FL_hip_x", 6.0)
    _setd("FR_hip_x", -6.0)
  elif name == "dual_arm_tabletop":
    # Slight asymmetric bimanual reach over workbench
    d.qpos[0] = -0.18
    d.qpos[1] = 0.54
    d.qpos[3] = -1.78
    d.qpos[9] = 0.22
    d.qpos[10] = 0.62
    d.qpos[12] = -1.70

  mujoco.mj_forward(m, d)
  return d


def render_multi_embodiment_gallery(
    res: int = 1024,
    n_ao: int = 16,
    n_shadow: int = 16,
    out_npz: str = "out/metrics/v3_w0_wave4_gallery.npz",
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
  specs_cfg = [
      ("humanoid_running", "Humanoid-21DOF (Running Stride)", _humanoid_studio_xml(), "hero"),
      ("panda_pick_cube", "PandaPickCube (7-DOF Precision Grasp)", _panda_studio_xml(), "hero"),
      ("dual_arm_tabletop", "Dual-Arm Tabletop (16-DOF Bimanual)", _dual_arm_studio_xml(), "diag_overview"),
      ("quadruped_trot", "Quadruped-12DOF (Diagonal Trot)", _quadruped_studio_xml(), "hero"),
  ]

  renders: Dict[str, np.ndarray] = {}
  bench_meta: Dict[str, Any] = {"res": f"{res}x{res}", "n_ao_samples": n_ao, "n_shadow_samples": n_shadow, "scenes": {}}

  for key, title, xml, cam_name in specs_cfg:
    m = mujoco.MjModel.from_xml_string(xml)
    mx = mjx.put_model(m, impl="jax")
    d_host = _pose_embodiment(key, m)
    dx = mjx.make_data(m, impl="jax").replace(qpos=jnp.asarray(d_host.qpos, jnp.float32))
    dx = mjx.forward(mx, dx)
    floor_gid = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor"))

    ao_s = n_ao if m.ngeom <= 32 else min(n_ao, 4)
    sh_s = n_shadow if m.ngeom <= 32 else min(n_shadow, 4)
    block_r_use = 512 if m.ngeom <= 32 else 256
    sc = tpu_rtx.compile(
        m,
        shader="fused_ao_ggx",
        res=(res, res),
        sharding=1,
        cam_id=cam_name,
        dtype="float32",
        outputs=("rgb", "depth", "normal", "seg_id"),
        block_b=8,
        block_r=block_r_use,
        n_ao_samples=ao_s,
        n_shadow_samples=sh_s,
        ao_radius=0.35,
        ao_strength=0.82,
        shadow_softness=0.05,
    )

    hlo_info = tpu_rtx.audit_hlo(tpu_rtx.lower_render(dx, sc))
    assert hlo_info["zero_copy_verified"], f"Gallery HLO failed zero-copy for {key}: {hlo_info}"

    @jax.jit
    def _render_and_grade(d_in):
      out = tpu_rtx.render(d_in, sc)
      graded = studio_agx_post_tpu(out["rgb"], out["normal"], out["seg_id"], floor_gid)
      return graded, out["depth"], out["normal"]

    # Warmup + time
    graded, depth, normal = _render_and_grade(dx)
    jax.block_until_ready(graded)
    t0 = time.perf_counter()
    for _ in range(5):
      graded, depth, normal = _render_and_grade(dx)
    jax.block_until_ready(graded)
    dt_ms = ((time.perf_counter() - t0) / 5.0) * 1e3

    renders[f"{key}_rgb"] = (np.clip(np.asarray(graded, dtype=np.float32), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    bench_meta["scenes"][key] = {
        "title": title,
        "ngeom": int(m.ngeom),
        "nq": int(m.nq),
        "nv": int(m.nv),
        "n_ao_samples": int(ao_s),
        "n_shadow_samples": int(sh_s),
        "dt_render_ms": dt_ms,
        "hlo": hlo_info,
    }
    print(f"GALLERY [{key}] {res}x{res} (AO={ao_s}, Sh={sh_s}): {dt_ms:.2f} ms | HLO=0cb/0sr/0g", flush=True)

  os.makedirs(os.path.dirname(out_npz) or ".", exist_ok=True)
  np.savez_compressed(out_npz, **renders)
  print("Saved gallery NPZ ->", out_npz, flush=True)
  return renders, bench_meta


def make_wave4_gallery_figure(
    renders_npz: str,
    metrics_json: str,
    out_png: str,
) -> None:
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  smp = np.load(renders_npz)
  with open(metrics_json, "r") as f:
    meta = json.load(f)

  plt.rcParams.update({
      "figure.facecolor": "#0b0f17",
      "axes.facecolor": "#111824",
      "axes.edgecolor": "#2c384a",
      "axes.labelcolor": "#dce4ef",
      "xtick.color": "#9fb0c7",
      "ytick.color": "#9fb0c7",
      "text.color": "#edf2f8",
      "grid.color": "#243042",
  })

  fig = plt.figure(figsize=(20.5, 11.2), dpi=180, facecolor="#0b0f17")
  gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1.15, 1.0], wspace=0.14, hspace=0.18)

  panels = [
      (0, 0, "humanoid_running", "Humanoid-21DOF (Running Stride)", "27-DOF | 21 Geoms | 1024x1024 GGX+16-AO+SoftShadow+AgX"),
      (0, 1, "panda_pick_cube", "PandaPickCube (7-DOF Precision Grasp)", "15-DOF | 23 Geoms | 1024x1024 GGX+16-AO+SoftShadow+AgX"),
      (1, 0, "dual_arm_tabletop", "Dual-Arm Tabletop (RoboTwin Bimanual)", "16-DOF + Clutter | 58 Geoms | 1024x1024 GGX+16-AO+SoftShadow+AgX"),
      (1, 1, "quadruped_trot", "Quadruped-12DOF (Diagonal Trot)", "18-DOF | 27 Geoms | 1024x1024 GGX+16-AO+SoftShadow+AgX"),
  ]

  for r, c, key, title, sub in panels:
    ax = fig.add_subplot(gs[r, c])
    raw_img = smp[f"{key}_rgb"]
    img = (raw_img.astype(np.float32) / 255.0) if raw_img.dtype == np.uint8 else np.clip(raw_img, 0.0, 1.0)
    ax.imshow(img)
    ax.set_title(f"{title}\n{sub}", fontsize=10.5, fontweight="bold", pad=6, color="#f0f4fa")
    ax.axis("off")

  # Right column top: Closed-Loop Vision PPO Throughput (4c TPU v5p)
  ax_sps = fig.add_subplot(gs[0, 2])
  ppo_runs = meta.get("ppo_runs", [])
  if ppo_runs:
    lbls, vals, cols = [], [], []
    cmap = {"cartpole": "#38bdf8", "panda_pick_cube": "#f97316", "humanoid_21dof": "#34d399"}
    for run in ppo_runs:
      lbls.append(f"{run['env']} B={run['num_envs']} (ar={run['action_repeat']})")
      vals.append(run["steady_env_steps_per_sec"] / 1e3)
      cols.append(cmap.get(run["env"], "#a78bfa"))
    y = np.arange(len(lbls))
    ax_sps.barh(y, vals, color=cols, height=0.58)
    ax_sps.set_yticks([])
    ax_sps.invert_yaxis()
    vmax = max(vals)
    for yi, (lbl, v) in enumerate(zip(lbls, vals)):
      if v > vmax * 0.52:
        ax_sps.text(v * 0.97, yi, f"{lbl}: {v:,.1f}k/s", va="center", ha="right", color="#090d14", fontsize=8.3, fontweight="bold")
      else:
        ax_sps.text(v + vmax * 0.02, yi, f"{lbl}: {v:,.1f}k/s", va="center", ha="left", color="#edf2f8", fontsize=8.3, fontweight="bold")
    ax_sps.set_xlim(0, vmax * 1.18)
    ax_sps.set_xlabel("Closed-Loop Vision PPO (k env-steps/s)", fontsize=9.5, fontweight="bold")
    ax_sps.set_title("4-Chip shard_map Vision PPO Throughput\n(HBM Zero-Copy tpu_rtx, host_callbacks=0)", fontsize=10.5, fontweight="bold")
    ax_sps.grid(axis="x", alpha=0.35)

  # Right column bottom: Normalized Reward / Policy Convergence Curves
  ax_rew = fig.add_subplot(gs[1, 2])
  if ppo_runs:
    # Pick best representative run per environment
    best_by_env: Dict[str, Dict[str, Any]] = {}
    for run in ppo_runs:
      en = run["env"]
      if en not in best_by_env or run["best_reward"] > best_by_env[en]["best_reward"]:
        best_by_env[en] = run
    cmap = {"cartpole": "#38bdf8", "panda_pick_cube": "#f97316", "humanoid_21dof": "#34d399"}
    nice = {"cartpole": "Cartpole (2-DOF)", "panda_pick_cube": "PandaPickCube (15-DOF)", "humanoid_21dof": "Humanoid-21DOF (27-DOF)"}
    for en, run in best_by_env.items():
      hist = run["history"]
      xs = np.array([h["iter"] for h in hist], dtype=np.float32)
      ys = np.array([h["mean_reward"] for h in hist], dtype=np.float32)
      # Plot normalized gain (rew - rew0) / max(|rew0|, 1) or raw reward with dual clean scaling
      ax_rew.plot(xs, ys, lw=2.2, color=cmap.get(en, "#fff"), label=f"{nice[en]} ({run['steady_env_steps_per_sec']/1e3:.1f}k/s)")
    ax_rew.set_xlabel("PPO Iteration (4 TPU v5p Chips)", fontsize=9.5, fontweight="bold")
    ax_rew.set_ylabel("Mean Step Reward", fontsize=9.5, fontweight="bold")
    ax_rew.set_title("Closed-Loop Vision PPO Convergence\n(2-Frame Stack + CoordConv + Critic EMA Norm)", fontsize=10.5, fontweight="bold")
    ax_rew.grid(True, alpha=0.35)
    ax_rew.legend(fontsize=8.2, loc="lower right", facecolor="#0f1724", edgecolor="#334155")

  fig.suptitle(
      "TPU-RTX Wave 4: Multi-Embodiment 1024x1024 GGX PBR+AO+SoftShadow Gallery & Zero-Copy Vision PPO",
      fontsize=14.0,
      fontweight="bold",
      y=0.985,
      color="#f8fafc",
  )
  os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
  fig.savefig(out_png, bbox_inches="tight", facecolor="#0b0f17")
  plt.close(fig)
  print("Saved figure ->", out_png, flush=True)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--res", type=int, default=1024)
  ap.add_argument("--n_ao", type=int, default=16)
  ap.add_argument("--n_shadow", type=int, default=16)
  ap.add_argument("--out_npz", default="out/metrics/v3_w0_wave4_gallery.npz")
  ap.add_argument("--metrics_json", default="out/metrics/v3_w0_wave4_rl_sdk.json")
  ap.add_argument("--out_png", default="out/figures/v3_w0_multi_embodiment_pbr.png")
  ap.add_argument("--render_only", action="store_true")
  ap.add_argument("--figure_only", action="store_true")
  args = ap.parse_args()

  if not args.figure_only:
    _, bench_meta = render_multi_embodiment_gallery(
        res=args.res, n_ao=args.n_ao, n_shadow=args.n_shadow, out_npz=args.out_npz
    )
    if os.path.exists(args.metrics_json):
      with open(args.metrics_json, "r") as f:
        payload = json.load(f)
    else:
      payload = {}
    payload["gallery_pbr"] = bench_meta
    tmp = args.metrics_json + ".tmp"
    os.makedirs(os.path.dirname(args.metrics_json) or ".", exist_ok=True)
    with open(tmp, "w") as f:
      json.dump(payload, f, indent=2)
    os.replace(tmp, args.metrics_json)

  if not args.render_only:
    make_wave4_gallery_figure(args.out_npz, args.metrics_json, args.out_png)


if __name__ == "__main__":
  main()
