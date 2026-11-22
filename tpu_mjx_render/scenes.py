"""Self-contained MuJoCo MJCF scenes aligned with google-deepmind/mujoco_playground."""

from __future__ import annotations

CARTPOLE_XML = """<mujoco model="cart-pole">
  <option timestep="0.01" iterations="1" ls_iterations="4">
    <flag contact="disable" eulerdamp="disable"/>
  </option>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0.1 0.1 0.1"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0.05 0.08 0.12" width="128" height="128"/>
    <texture type="2d" name="grid" builtin="checker" mark="edge" rgb1="0.18 0.24 0.32" rgb2="0.10 0.14 0.20"
      markrgb="0.7 0.75 0.8" width="128" height="128"/>
    <material name="grid" texture="grid" texuniform="true" texrepeat="1 1" reflectance="0.2"/>
    <material name="self" rgba="0.75 0.55 0.30 1"/>
    <material name="decoration" rgba="0.35 0.50 0.65 1"/>
  </asset>

  <default>
    <default class="pole">
      <joint type="hinge" axis="0 1 0" damping="2e-6"/>
      <geom type="capsule" fromto="0 0 0 0 0 1" size="0.045" material="self" mass="0.1"/>
    </default>
  </default>

  <worldbody>
    <light name="light" pos="0.5 -1.5 5" dir="-0.1 0.25 -1" directional="false" castshadow="true" diffuse="0.8 0.8 0.8" specular="0.2 0.2 0.2"/>
    <camera name="fixed" pos="0 -4 1" zaxis="0 -1 0" fovy="45"/>
    <camera name="lookatcart" mode="targetbody" target="cart" pos="0 -2 2" fovy="45"/>
    <geom name="floor" pos="0 0 -0.05" size="4 4 0.2" type="plane" material="grid"/>
    <geom name="rail1" type="capsule" pos="0  0.07 1" zaxis="1 0 0" size="0.02 2" material="decoration"/>
    <geom name="rail2" type="capsule" pos="0 -0.07 1" zaxis="1 0 0" size="0.02 2" material="decoration"/>
    <body name="cart" pos="0 0 1">
      <joint name="slider" type="slide" limited="true" axis="1 0 0" range="-1.8 1.8" solreflimit="0.08 1" damping="5e-4"/>
      <geom name="cart" type="box" size="0.2 0.15 0.1" material="self" mass="1"/>
      <body name="pole_1" childclass="pole">
        <joint name="hinge_1"/>
        <geom name="pole_1"/>
      </body>
    </body>
  </worldbody>

  <actuator>
    <motor name="slide" joint="slider" gear="10" ctrllimited="true" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""


# PandaPickCubeCartesian-aligned model (exact DH kinematics, keyframes, D435 camera,
# floor/init_space/box/mocap_target, using compact analytic primitives for all links so
# Stage 1 runs natively in TPU VMEM without BVH divergence).
PANDA_PICK_CUBE_XML = """<mujoco model="panda single cube camera">
  <compiler angle="radian" autolimits="true"/>
  <statistic center="0.3 0 0.4" extent="1"/>

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
    <material name="accent_blue" rgba="0.10 0.50 0.78 1"/>
    <material name="cube_red" rgba="0.88 0.14 0.12 1"/>
    <material name="target_ghost" rgba="0.90 0.25 0.20 0.0"/>
    <material name="strip_white" rgba="0.92 0.92 0.90 1"/>
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
    <light name="sun_overhead" pos="0.6 0.3 1.8" dir="-0.15 -0.15 -1" directional="false" castshadow="true"
      diffuse="0.85 0.83 0.78" specular="0.3 0.3 0.3"/>
    <!-- Matching Realsense D435 from mujoco_playground mjx_single_cube_camera.xml -->
    <camera fovy="42" name="front" pos="1.176 0.001 0.483" xyaxes="0.006 1.000 -0.000 -0.720 0.004 0.694"/>
    <camera fovy="45" name="side" pos="0.60 -0.85 0.55" xyaxes="1 0 0 0 0.45 0.89"/>

    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane" contype="1" conaffinity="1"/>
    <geom name="init_space" size="0.22 0.24 0.001" type="box" pos="0.631 0 0.001" material="strip_white" contype="0" conaffinity="0"/>

    <body name="link0" childclass="panda">
      <geom name="link0_base" type="cylinder" pos="-0.03 0 0.07" size="0.095 0.07" material="off_white" class="visual_prim"/>
      <geom name="link0_pedestal" type="cylinder" pos="0 0 0.21" size="0.068 0.08" material="white" class="visual_prim"/>
      <body name="link1" pos="0 0 0.333">
        <inertial mass="2.0" pos="0 0 -0.05" diaginertia="0.01 0.01 0.01"/>
        <joint name="joint1" damping="40"/>
        <geom name="link1_body" type="capsule" fromto="0 0 -0.10 0 0 0.04" size="0.062" material="white" class="visual_prim"/>
        <body name="link2" quat="1 -1 0 0">
          <inertial mass="2.0" pos="0 -0.15 0" diaginertia="0.01 0.01 0.01"/>
          <joint name="joint2" range="-1.7628 1.7628" damping="40"/>
          <geom name="link2_collar" type="cylinder" pos="0 0 0" size="0.065 0.045" material="black" class="visual_prim"/>
          <geom name="link2_arm" type="capsule" fromto="0 0 0 0 -0.22 0" size="0.055" material="white" class="visual_prim"/>
          <body name="link3" pos="0 -0.316 0" quat="1 1 0 0">
            <inertial mass="2.0" pos="0 0 -0.05" diaginertia="0.01 0.01 0.01"/>
            <joint name="joint3" damping="40"/>
            <geom name="link3_seg" type="capsule" fromto="0 0 -0.11 0.045 0 0" size="0.052" material="white" class="visual_prim"/>
            <geom name="link3_joint" type="sphere" pos="0.0825 0 0" size="0.056" material="black" class="visual_prim"/>
            <body name="link4" pos="0.0825 0 0" quat="1 1 0 0">
              <inertial mass="2.0" pos="-0.04 0.15 0" diaginertia="0.01 0.01 0.01"/>
              <joint name="joint4" range="-3.0718 -0.0698" damping="40"/>
              <geom name="link4_arm" type="capsule" fromto="0 0 0 -0.06 0.20 0" size="0.050" material="white" class="visual_prim"/>
              <body name="link5" pos="-0.0825 0.384 0" quat="1 -1 0 0">
                <inertial mass="1.5" pos="0 0 -0.1" diaginertia="0.008 0.008 0.008"/>
                <joint name="joint5" damping="2"/>
                <geom name="link5_forearm" type="capsule" fromto="0 0 -0.22 0 0 -0.02" size="0.044" material="off_white" class="visual_prim"/>
                <geom name="link5_collar" type="cylinder" pos="0 0 0" size="0.046 0.025" material="black" class="visual_prim"/>
                <body name="link6" quat="1 1 0 0">
                  <inertial mass="1.0" pos="0.04 0 0" diaginertia="0.005 0.005 0.005"/>
                  <joint name="joint6" range="-0.0175 3.7525" damping="2"/>
                  <geom name="link6_wrist" type="capsule" fromto="0 0 0 0.088 0 0" size="0.040" material="white" class="visual_prim"/>
                  <body name="link7" pos="0.088 0 0" quat="1 1 0 0">
                    <inertial mass="0.7" pos="0 0 0.05" diaginertia="0.003 0.003 0.003"/>
                    <joint name="joint7" damping="2"/>
                    <geom name="link7_flange" type="cylinder" pos="0 0 0.06" size="0.038 0.045" material="black" class="visual_prim"/>
                    <body name="hand" pos="0 0 0.107" quat="0.9238795 0 0 -0.3826834">
                      <inertial mass="0.73" pos="0 0 0.03" diaginertia="0.001 0.0025 0.0017"/>
                      <geom name="hand_box" type="box" pos="0 0 0.030" size="0.024 0.102 0.026" material="off_white" class="visual_prim"/>
                      <geom name="hand_capsule" type="capsule" size="0.036 0.055" quat="1 1 0 0" pos="0 0 0.03" material="black" contype="1" conaffinity="1" group="3"/>
                      <site name="gripper" pos="0 0 0.10"/>
                      <body name="left_finger" pos="0 0 0.0584">
                        <inertial mass="0.03" pos="0 0.01 0.025" diaginertia="1e-5 1e-5 1e-5"/>
                        <joint name="finger_joint1" class="finger" damping="10"/>
                        <geom name="left_finger_vis" type="box" pos="0 0.010 0.026" size="0.010 0.008 0.024" material="black" class="visual_prim"/>
                        <geom name="left_finger_pad" type="box" size="0.009 0.0075 0.016" pos="0 0.008 0.038" material="off_white" class="pad_col"/>
                      </body>
                      <body name="right_finger" pos="0 0 0.0584" quat="0 0 0 1">
                        <inertial mass="0.03" pos="0 0.01 0.025" diaginertia="1e-5 1e-5 1e-5"/>
                        <joint name="finger_joint2" class="finger" damping="10"/>
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

    <body name="box" pos="0.65 0 0.03">
      <freejoint/>
      <inertial mass="0.1" pos="0 0 0" diaginertia="1e-4 1e-4 1e-4"/>
      <geom type="box" name="box" size="0.02 0.02 0.03" condim="3"
       friction="1 0.03 0.003" material="cube_red" contype="1" conaffinity="1" solref="0.01 1"/>
    </body>

    <body mocap="true" name="mocap_target" pos="0.65 0 0.25">
      <geom type="box" size="0.02 0.02 0.03" material="target_ghost" contype="0" conaffinity="0"/>
    </body>
  </worldbody>

  <equality>
    <joint joint1="finger_joint1" joint2="finger_joint2" solimp="0.95 0.99 0.001" solref="0.005 1"/>
  </equality>

  <actuator>
    <position class="panda" name="actuator1" joint="joint1" kp="1000" kv="20" ctrlrange="-2.8973 2.8973"/>
    <position class="panda" name="actuator2" joint="joint2" kp="1000" kv="20" ctrlrange="-1.7628 1.7628"/>
    <position class="panda" name="actuator3" joint="joint3" kp="750" kv="4" ctrlrange="-2.8973 2.8973"/>
    <position class="panda" name="actuator4" joint="joint4" kp="750" kv="4" ctrlrange="-3.0718 -0.0698"/>
    <position class="panda" name="actuator5" joint="joint5" kp="300" kv="2" forcerange="-12 12" ctrlrange="-2.8973 2.8973"/>
    <position class="panda" name="actuator6" joint="joint6" kp="300" kv="2" forcerange="-12 12" ctrlrange="-0.0175 3.7525"/>
    <position class="panda" name="actuator7" joint="joint7" kp="300" kv="2" forcerange="-12 12"/>
    <general class="panda" name="actuator8" joint="finger_joint1" ctrlrange="0 0.04" gainprm="350 0 0"
      biasprm="0 -350 -10" forcerange="-100 100"/>
  </actuator>

  <keyframe>
    <key name="low_home"
      qpos="-0.00002 0.47804 -0.00055 -1.81309 -0.00161 2.34597 0.78501 0.04000 0.04000 0.65000 0.00000 0.03000 1.00000 0.00000 0.00000 0.00000"
      ctrl="-0.00002 0.47804 -0.00055 -1.81309 -0.00161 2.34597 0.78501 0.04000"/>
    <key name="picked"
      qpos="0.0481 0.5604 -0.0417 -1.8459 0.0342 2.3654 0.7852 0.0198 0.0200 0.6368 0.0064 0.1179 1.0000 0.0001 -0.0010 -0.0063"
      ctrl="0.0481 0.5604 -0.0417 -1.8459 0.0342 2.3654 0.7852 0.0000"/>
  </keyframe>
</mujoco>
"""


ALL_PRIMITIVES_XML = """<mujoco model="all_primitives_test">
  <worldbody>
    <light name="key" pos="1.5 -1.5 3.0" dir="-0.3 0.3 -1" directional="false" castshadow="true" diffuse="0.9 0.9 0.9"/>
    <camera name="main" pos="0 -3.2 1.2" xyaxes="1 0 0 0 0.35 0.937" fovy="45"/>
    <geom name="p_plane" type="plane" pos="0 0 0" size="3 3 0.1" rgba="0.5 0.5 0.5 1"/>
    <geom name="p_sphere" type="sphere" pos="-1.1 0 0.4" size="0.32 0 0" rgba="0.9 0.2 0.2 1"/>
    <geom name="p_capsule" type="capsule" pos="-0.4 0 0.5" euler="25 35 10" size="0.16 0.30 0" rgba="0.2 0.8 0.3 1"/>
    <geom name="p_ellipsoid" type="ellipsoid" pos="0.3 0.1 0.45" euler="15 -25 40" size="0.28 0.16 0.36" rgba="0.2 0.4 0.9 1"/>
    <geom name="p_cylinder" type="cylinder" pos="1.0 -0.1 0.42" euler="-20 15 30" size="0.20 0.32 0" rgba="0.9 0.7 0.2 1"/>
    <geom name="p_box" type="box" pos="0.0 0.8 0.35" euler="10 20 45" size="0.25 0.18 0.28" rgba="0.8 0.3 0.8 1"/>
  </worldbody>
</mujoco>
"""


def get_scene_xml(name: str) -> str:
  if name == "cartpole":
    return CARTPOLE_XML
  if name == "panda_pick_cube":
    return PANDA_PICK_CUBE_XML
  if name == "all_primitives":
    return ALL_PRIMITIVES_XML
  raise ValueError(f"Unknown scene: {name}")
