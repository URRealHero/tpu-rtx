"""Wave 4 Cinema-Grade 4K Hybrid Real-to-Sim: Humanoid-21DOF + Mobile Panda in 10M 3DGS Bistro.

Optimizations over Wave 3 (2299.64 ms @ 4K):
1. 4-Chip Sharded Stage 1 (`project_and_bucket_cinema_4c` + `jax.lax.all_to_all`):
   - Shards the 10M Gaussians (2.5M/chip) across all 4 TPU v5p chips for EWA projection,
     dynamic GGX/Fresnel shading, and 4x smaller `sort_key_val`, followed by ICI `all_to_all`.
2. 4-Chip `(8, 512)` Mosaic Pallas Exact Analytic MJX Raycaster + Ground Shadow-Catcher:
   - Replaces Wave 3's XLA `jax.lax.scan` raycaster (~1582 ms) with a fused VMEM Mosaic
     Pallas kernel (~10-20 ms) computing exact primary hits, self-shadows, and real-time
     contact shadows cast onto the 3DGS Bistro cobblestone street.
3. 4-Chip Sharded AgX Filmic Tonemap & HDR Compositor (`composite_cinema_hdr_and_tonemap_4c`).
"""

from __future__ import annotations

import inspect
import math
import time
from typing import Any, Dict, Optional, Tuple

import jax
try:
  from jax import shard_map
except ImportError:
  from jax.experimental.shard_map import shard_map
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np

from tpu_mjx_render.camera import compute_camera_rays
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

from cinema_3dgs import (
    blender_agx_tonemap_tpu,
    nishita_golden_hour_sky_tpu,
    render_macro_tiles_hdr_subset,
)
from gs3d_v3 import pallas_gs3d_v3_fused_rasterize


# ---------------------------------------------------------------------------
# 1. Dual-Embodiment MJCF Builder (Humanoid-21DOF + Mobile Panda Manipulator)
# ---------------------------------------------------------------------------

def build_bistro_embodiments_xml() -> str:
  """Build combined MJX model: dynamic Humanoid-21DOF + Wheeled Mobile Panda on Bistro street."""
  return """<mujoco model="bistro_humanoid_mobile_panda">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.004" iterations="3" ls_iterations="6" integrator="implicitfast">
    <flag eulerdamp="disable"/>
  </option>

  <asset>
    <material name="hum_titan" rgba="0.82 0.85 0.89 1" specular="0.60" shininess="0.52"/>
    <material name="hum_orange" rgba="0.93 0.41 0.09 1" specular="0.50" shininess="0.45"/>
    <material name="hum_dark" rgba="0.14 0.16 0.20 1" specular="0.55" shininess="0.48"/>
    <material name="hum_visor" rgba="0.08 0.62 0.90 1" specular="0.80" shininess="0.75"/>

    <material name="mob_chassis" rgba="0.16 0.18 0.22 1" specular="0.45" shininess="0.40"/>
    <material name="mob_deck" rgba="0.09 0.10 0.12 1" specular="0.20" shininess="0.20"/>
    <material name="mob_tire" rgba="0.08 0.08 0.09 1" specular="0.15" shininess="0.15"/>
    <material name="panda_white" rgba="0.94 0.95 0.97 1" specular="0.48" shininess="0.42"/>
    <material name="panda_off" rgba="0.84 0.87 0.90 1" specular="0.42" shininess="0.38"/>
    <material name="cube_red" rgba="0.90 0.14 0.11 1" specular="0.40" shininess="0.35"/>
    <material name="cube_blue" rgba="0.11 0.44 0.92 1" specular="0.40" shininess="0.35"/>
  </asset>

  <default>
    <joint armature="0.03" damping="3.0" limited="true"/>
    <geom condim="3" friction="0.9 0.02 0.002"/>
    <default class="vis_only">
      <geom contype="0" conaffinity="0"/>
    </default>
  </default>

  <worldbody>
    <light name="sun" pos="10 10 15" dir="-0.5 -0.7 -0.4" directional="true" castshadow="true"
      diffuse="1.0 0.7 0.4" specular="0.5 0.4 0.3"/>
    <camera name="front" pos="-5 0 1.5" xyaxes="0 1 0 0 0 1" fovy="50"/>

    <!-- Invisible physics contact plane at z=0 (mapped to Bistro street y=0.22) -->
    <geom name="floor" type="plane" pos="0 0 0" size="25 25 0.1" rgba="0 0 0 0" contype="1" conaffinity="1"/>

    <!-- ==================== EMBODIMENT 1: HUMANOID-21DOF ==================== -->
    <!-- Positioned in cobblestone street near sidewalk cafe tables -->
    <body name="hum_torso" pos="0.0 0.0 1.24" euler="0 0.10 0.45">
      <freejoint name="hum_root"/>
      <geom name="hum_torso_g" type="capsule" fromto="0 -0.085 0 0 0.085 0" size="0.105" material="hum_orange" mass="5.0" class="vis_only"/>
      <geom name="hum_chest_plate" type="box" pos="0.04 0 0.02" size="0.075 0.095 0.08" material="hum_titan" mass="1.0" class="vis_only"/>
      <geom name="hum_head" type="sphere" pos="0.02 0 0.22" size="0.11" material="hum_titan" mass="1.5" class="vis_only"/>
      <geom name="hum_visor" type="box" pos="0.105 0 0.225" size="0.034 0.072 0.032" material="hum_visor" mass="0.1" class="vis_only"/>
      <geom name="hum_uwaist" type="capsule" fromto="-0.01 -0.065 -0.12 -0.01 0.065 -0.12" size="0.078" material="hum_dark" mass="2.0" class="vis_only"/>

      <body name="hum_lwaist" pos="-0.01 0 -0.25">
        <joint name="hum_abdomen_z" type="hinge" axis="0 0 1" range="-0.8 0.8"/>
        <joint name="hum_abdomen_y" type="hinge" axis="0 1 0" range="-1.2 0.5"/>
        <geom name="hum_lwaist_g" type="capsule" fromto="0 -0.065 0 0 0.065 0" size="0.074" material="hum_orange" mass="2.0" class="vis_only"/>
        <body name="hum_pelvis" pos="0 0 -0.165">
          <joint name="hum_abdomen_x" type="hinge" axis="1 0 0" range="-0.6 0.6"/>
          <geom name="hum_butt" type="capsule" fromto="-0.02 -0.085 0 -0.02 0.085 0" size="0.082" material="hum_dark" mass="3.0" class="vis_only"/>

          <body name="hum_rthigh" pos="0 -0.105 -0.04">
            <joint name="hum_rhip_x" type="hinge" axis="1 0 0" range="-0.5 0.2"/>
            <joint name="hum_rhip_z" type="hinge" axis="0 0 1" range="-1.0 0.6"/>
            <joint name="hum_rhip_y" type="hinge" axis="0 1 0" range="-1.8 0.8"/>
            <geom name="hum_rthigh_g" type="capsule" fromto="0 0 0 0 0.01 -0.35" size="0.064" material="hum_titan" mass="3.0" class="vis_only"/>
            <body name="hum_rshin" pos="0 0.01 -0.40">
              <joint name="hum_rknee" type="hinge" axis="0 -1 0" range="-2.6 0.0"/>
              <geom name="hum_rshin_g" type="capsule" fromto="0 0 0 0 0 -0.32" size="0.052" material="hum_orange" mass="2.0" class="vis_only"/>
              <body name="hum_rfoot" pos="0 0 -0.36">
                <joint name="hum_rankle_x" type="hinge" axis="1 0 0" range="-0.8 0.8"/>
                <joint name="hum_rankle_y" type="hinge" axis="0 1 0" range="-0.9 0.9"/>
                <geom name="hum_rfoot_g" type="box" pos="0.045 0 -0.022" size="0.105 0.050 0.026" material="hum_dark" mass="1.0"/>
              </body>
            </body>
          </body>

          <body name="hum_lthigh" pos="0 0.105 -0.04">
            <joint name="hum_lhip_x" type="hinge" axis="-1 0 0" range="-0.5 0.2"/>
            <joint name="hum_lhip_z" type="hinge" axis="0 0 -1" range="-1.0 0.6"/>
            <joint name="hum_lhip_y" type="hinge" axis="0 1 0" range="-1.8 0.8"/>
            <geom name="hum_lthigh_g" type="capsule" fromto="0 0 0 0 -0.01 -0.35" size="0.064" material="hum_titan" mass="3.0" class="vis_only"/>
            <body name="hum_lshin" pos="0 -0.01 -0.40">
              <joint name="hum_lknee" type="hinge" axis="0 -1 0" range="-2.6 0.0"/>
              <geom name="hum_lshin_g" type="capsule" fromto="0 0 0 0 0 -0.32" size="0.052" material="hum_orange" mass="2.0" class="vis_only"/>
              <body name="hum_lfoot" pos="0 0 -0.36">
                <joint name="hum_lankle_x" type="hinge" axis="1 0 0" range="-0.8 0.8"/>
                <joint name="hum_lankle_y" type="hinge" axis="0 1 0" range="-0.9 0.9"/>
                <geom name="hum_lfoot_g" type="box" pos="0.045 0 -0.022" size="0.105 0.050 0.026" material="hum_dark" mass="1.0"/>
              </body>
            </body>
          </body>
        </body>
      </body>

      <body name="hum_ruarm" pos="0 -0.18 0.06">
        <joint name="hum_rsh1" type="hinge" axis="0 1 0" range="-2.0 1.5"/>
        <joint name="hum_rsh2" type="hinge" axis="1 0 0" range="-1.5 1.5"/>
        <geom name="hum_ruarm_g" type="capsule" fromto="0 0 0 0.02 -0.04 -0.26" size="0.046" material="hum_titan" mass="1.2" class="vis_only"/>
        <body name="hum_rlarm" pos="0.02 -0.04 -0.28">
          <joint name="hum_relbow" type="hinge" axis="0 1 0" range="-2.2 0.2"/>
          <geom name="hum_rlarm_g" type="capsule" fromto="0 0 0 0.22 0.02 -0.04" size="0.038" material="hum_orange" mass="0.8" class="vis_only"/>
          <geom name="hum_rhand" type="sphere" pos="0.24 0.02 -0.04" size="0.046" material="hum_dark" mass="0.3" class="vis_only"/>
        </body>
      </body>

      <body name="hum_luarm" pos="0 0.18 0.06">
        <joint name="hum_lsh1" type="hinge" axis="0 1 0" range="-2.0 1.5"/>
        <joint name="hum_lsh2" type="hinge" axis="1 0 0" range="-1.5 1.5"/>
        <geom name="hum_luarm_g" type="capsule" fromto="0 0 0 0.02 0.04 -0.26" size="0.046" material="hum_titan" mass="1.2" class="vis_only"/>
        <body name="hum_llarm" pos="0.02 0.04 -0.28">
          <joint name="hum_lelbow" type="hinge" axis="0 1 0" range="-2.2 0.2"/>
          <geom name="hum_llarm_g" type="capsule" fromto="0 0 0 0.22 -0.02 -0.04" size="0.038" material="hum_orange" mass="0.8" class="vis_only"/>
          <geom name="hum_lhand" type="sphere" pos="0.24 -0.02 -0.04" size="0.046" material="hum_dark" mass="0.3" class="vis_only"/>
        </body>
      </body>
    </body>

    <!-- ==================== EMBODIMENT 2: MOBILE PANDA MANIPULATOR ==================== -->
    <body name="mob_base" pos="1.55 -3.75 0.0" euler="0 0 2.30">
      <geom name="mob_chassis" type="box" pos="0 0 0.22" size="0.34 0.24 0.095" material="mob_chassis" mass="18.0" class="vis_only"/>
      <geom name="mob_deck" type="box" pos="0.04 0 0.33" size="0.30 0.22 0.018" material="mob_deck" mass="2.0" class="vis_only"/>
      <geom name="mob_bumper" type="box" pos="0.35 0 0.17" size="0.02 0.23 0.035" material="hum_orange" mass="0.5" class="vis_only"/>
      <!-- 4 wheeled tires (radius=0.125 -> center z=0.125 flush on z=0) -->
      <geom name="mob_w_fl" type="cylinder" pos="0.22 0.26 0.125" euler="1.5708 0 0" size="0.125 0.042" material="mob_tire" class="vis_only"/>
      <geom name="mob_w_fr" type="cylinder" pos="0.22 -0.26 0.125" euler="1.5708 0 0" size="0.125 0.042" material="mob_tire" class="vis_only"/>
      <geom name="mob_w_rl" type="cylinder" pos="-0.22 0.26 0.125" euler="1.5708 0 0" size="0.125 0.042" material="mob_tire" class="vis_only"/>
      <geom name="mob_w_rr" type="cylinder" pos="-0.22 -0.26 0.125" euler="1.5708 0 0" size="0.125 0.042" material="mob_tire" class="vis_only"/>
      <!-- Orange wheel hubs -->
      <geom name="mob_h_fl" type="cylinder" pos="0.22 0.29 0.125" euler="1.5708 0 0" size="0.055 0.018" material="hum_orange" class="vis_only"/>
      <geom name="mob_h_fr" type="cylinder" pos="0.22 -0.29 0.125" euler="1.5708 0 0" size="0.055 0.018" material="hum_orange" class="vis_only"/>
      <geom name="mob_h_rl" type="cylinder" pos="-0.22 0.29 0.125" euler="1.5708 0 0" size="0.055 0.018" material="hum_orange" class="vis_only"/>
      <geom name="mob_h_rr" type="cylinder" pos="-0.22 -0.29 0.125" euler="1.5708 0 0" size="0.055 0.018" material="hum_orange" class="vis_only"/>
      <!-- Cubes on deck -->
      <geom name="mob_cube_blue" type="box" pos="0.24 -0.10 0.378" size="0.030 0.030 0.030" material="cube_blue" class="vis_only"/>
      <geom name="mob_cube_red" type="box" pos="0.21 0.09 0.378" size="0.028 0.028 0.028" material="cube_red" class="vis_only"/>

      <!-- 7-DOF Panda Arm mounted on deck -->
      <body name="panda_link0" pos="-0.08 0 0.348">
        <geom name="p_base" type="cylinder" pos="-0.02 0 0.065" size="0.090 0.065" material="panda_off" class="vis_only"/>
        <geom name="p_ped" type="cylinder" pos="0 0 0.20" size="0.065 0.075" material="panda_white" class="vis_only"/>
        <body name="panda_link1" pos="0 0 0.333">
          <joint name="p_j1" axis="0 0 1" range="-2.9 2.9"/>
          <geom name="p_l1" type="capsule" fromto="0 0 -0.10 0 0 0.04" size="0.060" material="panda_white" mass="1.5" class="vis_only"/>
          <body name="panda_link2" quat="1 -1 0 0">
            <joint name="p_j2" axis="0 0 1" range="-1.76 1.76"/>
            <geom name="p_l2_col" type="cylinder" pos="0 0 0" size="0.062 0.042" material="hum_dark" mass="1.0" class="vis_only"/>
            <geom name="p_l2_arm" type="capsule" fromto="0 0 0 0 -0.22 0" size="0.052" material="panda_white" mass="1.5" class="vis_only"/>
            <body name="panda_link3" pos="0 -0.316 0" quat="1 1 0 0">
              <joint name="p_j3" axis="0 0 1" range="-2.9 2.9"/>
              <geom name="p_l3_seg" type="capsule" fromto="0 0 -0.11 0.045 0 0" size="0.050" material="panda_white" mass="1.2" class="vis_only"/>
              <geom name="p_l3_jnt" type="sphere" pos="0.0825 0 0" size="0.054" material="hum_orange" mass="0.5" class="vis_only"/>
              <body name="panda_link4" pos="0.0825 0 0" quat="1 1 0 0">
                <joint name="p_j4" axis="0 0 1" range="-3.07 -0.07"/>
                <geom name="p_l4_arm" type="capsule" fromto="0 0 0 -0.06 0.20 0" size="0.048" material="panda_white" mass="1.2" class="vis_only"/>
                <body name="panda_link5" pos="-0.0825 0.384 0" quat="1 -1 0 0">
                  <joint name="p_j5" axis="0 0 1" range="-2.9 2.9"/>
                  <geom name="p_l5_farm" type="capsule" fromto="0 0 -0.22 0 0 -0.02" size="0.042" material="panda_off" mass="1.0" class="vis_only"/>
                  <geom name="p_l5_col" type="cylinder" pos="0 0 0" size="0.044 0.024" material="hum_orange" mass="0.3" class="vis_only"/>
                  <body name="panda_link6" quat="1 1 0 0">
                    <joint name="p_j6" axis="0 0 1" range="-0.02 3.75"/>
                    <geom name="p_l6_wr" type="capsule" fromto="0 0 0 0.088 0 0" size="0.038" material="panda_white" mass="0.8" class="vis_only"/>
                    <body name="panda_link7" pos="0.088 0 0" quat="1 1 0 0">
                      <joint name="p_j7" axis="0 0 1" range="-2.9 2.9"/>
                      <geom name="p_l7_fl" type="cylinder" pos="0 0 0.06" size="0.036 0.042" material="hum_dark" mass="0.5" class="vis_only"/>
                      <body name="panda_hand" pos="0 0 0.107" quat="0.9238795 0 0 -0.3826834">
                        <geom name="p_hand" type="box" pos="0 0 0.030" size="0.024 0.098 0.024" material="panda_off" mass="0.5" class="vis_only"/>
                        <geom name="p_lf" type="box" pos="0 0.045 0.072" size="0.011 0.009 0.028" material="hum_dark" mass="0.05" class="vis_only"/>
                        <geom name="p_rf" type="box" pos="0 -0.045 0.072" size="0.011 0.009 0.028" material="hum_dark" mass="0.05" class="vis_only"/>
                        <geom name="p_held_cube" type="box" pos="0 0 0.082" size="0.024 0.024 0.024" material="cube_red" mass="0.05" class="vis_only"/>
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
  </worldbody>
</mujoco>
"""


def set_bistro_pose(mj_model: mujoco.MjModel) -> np.ndarray:
  """Set dynamic walking/reaching pose for Humanoid-21DOF + reaching pose for Mobile Panda."""
  qpos = mj_model.qpos0.copy()

  def _set_j(name: str, val: float) -> None:
    jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid >= 0:
      qpos[int(mj_model.jnt_qposadr[jid])] = val

  # Humanoid root: z=1.15 so feet sit flush at z~0.02 on cobblestone street (y=0.22)
  qpos[0:3] = np.array([0.0, 0.0, 1.15], dtype=np.float64)
  # 3/4 front-facing yaw walking diagonally toward camera-right / cafe & Mobile Panda
  yaw = -2.15
  pitch = 0.08
  cy, sy = math.cos(0.5 * yaw), math.sin(0.5 * yaw)
  cp, sp = math.cos(0.5 * pitch), math.sin(0.5 * pitch)
  qpos[3:7] = np.array([cy * cp, -sy * sp, cy * sp, sy * cp], dtype=np.float64)

  # Dynamic mid-stride walking + right arm reaching toward cafe/robot
  _set_j("hum_abdomen_z", 0.14)
  _set_j("hum_abdomen_y", 0.08)
  _set_j("hum_rhip_y", -0.45)   # right leg stepping forward
  _set_j("hum_rknee", -0.35)
  _set_j("hum_rankle_y", 0.22)
  _set_j("hum_lhip_y", 0.35)    # left leg pushing off back
  _set_j("hum_lknee", -0.45)
  _set_j("hum_lankle_y", 0.15)
  _set_j("hum_rsh1", -0.75)     # right arm reaching forward
  _set_j("hum_rsh2", 0.18)
  _set_j("hum_relbow", -0.65)
  _set_j("hum_lsh1", 0.42)      # left arm swinging back
  _set_j("hum_lsh2", -0.15)
  _set_j("hum_lelbow", -0.55)

  # Mobile Panda arm reaching gracefully above its deck holding red cube
  _set_j("p_j1", 0.25)
  _set_j("p_j2", 0.42)
  _set_j("p_j3", -0.18)
  _set_j("p_j4", -1.78)
  _set_j("p_j5", 0.15)
  _set_j("p_j6", 2.15)
  _set_j("p_j7", 0.78)
  return qpos.astype(np.float32)


# ---------------------------------------------------------------------------
# 2. 4-Chip Sharded 10M-Gaussian Stage 1 + ICI all_to_all
# ---------------------------------------------------------------------------

def make_cinema_gs3d_4c_sharded_fn(
    mesh: jax.sharding.Mesh,
    height: int,
    width: int,
    cam_fovy: float = 50.0,
    znear: float = 0.12,
    k_macro: int = 4096,
    k_tile: int = 512,
    macro_rep: int = 4,
    use_bf16_mxu: bool = False,
):
  """4-chip sharded 3DGS renderer: shards N across 4 chips for 4x faster projection+sort + all_to_all."""
  n_chips = len(mesh.devices)
  assert n_chips == 4
  assert k_macro % n_chips == 0
  k_local = k_macro // n_chips

  M_SIZE = 32
  H_pad = ((height + M_SIZE - 1) // M_SIZE) * M_SIZE
  W_pad = ((width + M_SIZE - 1) // M_SIZE) * M_SIZE
  My = H_pad // M_SIZE
  Mx = W_pad // M_SIZE
  M_tiles = My * Mx
  M_pad = ((M_tiles + n_chips - 1) // n_chips) * n_chips
  M_sub = M_pad // n_chips
  assert M_tiles < 8191

  t_ox_np = np.zeros((M_pad,), dtype=np.float32)
  t_oy_np = np.zeros((M_pad,), dtype=np.float32)
  t_ox_np[:M_tiles] = np.tile(np.arange(Mx, dtype=np.float32) * 32.0, My)
  t_oy_np[:M_tiles] = np.repeat(np.arange(My, dtype=np.float32) * 32.0, Mx)

  P = jax.sharding.PartitionSpec
  sm_kw = {"check_rep": False} if "check_rep" in inspect.signature(shard_map).parameters else {"check_vma": False}

  t_ox_0 = np.tile(np.arange(4, dtype=np.float32) * 8.0, 4)
  t_oy_0 = np.repeat(np.arange(4, dtype=np.float32) * 8.0, 4)

  def _sharded_gs_body(
      mu, cov6, diff_hdr, op, nrm, spec_col, rough, sun_vis, sky_vis,
      cxp, cxm, sun_d, sun_c, sky_c, tox_sub, toy_sub
  ):
    # Each chip processes N_local = N // 4 Gaussians!
    N_loc = mu.shape[0]
    p_cam = (mu - cxp[None, :]) @ cxm
    cx, cy, cz = p_cam[:, 0], p_cam[:, 1], p_cam[:, 2]
    z_d = -cz
    in_front = z_d > znear
    z_safe = jnp.maximum(z_d, znear)

    half_h = math.tan(0.5 * math.radians(float(cam_fovy)))
    fy = 0.5 * float(height) / half_h
    fx = fy
    mean_x = 0.5 * float(width) + fx * (cx / z_safe)
    mean_y = 0.5 * float(height) - fy * (cy / z_safe)

    inv_z = 1.0 / z_safe
    inv_z2 = inv_z * inv_z
    lim_x = 1.3 * (0.5 * float(width) / fx) * z_safe
    lim_y = 1.3 * (0.5 * float(height) / fy) * z_safe
    cx_c = jnp.clip(cx, -lim_x, lim_x)
    cy_c = jnp.clip(cy, -lim_y, lim_y)
    j00, j02 = fx * inv_z, fx * cx_c * inv_z2
    j11, j12 = -fy * inv_z, -fy * cy_c * inv_z2

    w0 = j00[:, None] * cxm[None, :, 0] + j02[:, None] * cxm[None, :, 2]
    w1 = j11[:, None] * cxm[None, :, 1] + j12[:, None] * cxm[None, :, 2]
    s00, s01, s02, s11, s12, s22 = [cov6[:, i] for i in range(6)]
    u0, u1, u2 = w0[:, 0], w0[:, 1], w0[:, 2]
    v0, v1, v2 = w1[:, 0], w1[:, 1], w1[:, 2]
    sw0_0 = s00 * u0 + s01 * u1 + s02 * u2
    sw0_1 = s01 * u0 + s11 * u1 + s12 * u2
    sw0_2 = s02 * u0 + s12 * u1 + s22 * u2
    sw1_0 = s00 * v0 + s01 * v1 + s02 * v2
    sw1_1 = s01 * v0 + s11 * v1 + s12 * v2
    sw1_2 = s02 * v0 + s12 * v1 + s22 * v2
    cov2_xx = u0 * sw0_0 + u1 * sw0_1 + u2 * sw0_2 + 0.30
    cov2_xy = v0 * sw0_0 + v1 * sw0_1 + v2 * sw0_2
    cov2_yy = v0 * sw1_0 + v1 * sw1_1 + v2 * sw1_2 + 0.30

    max_hx = float(macro_rep * M_SIZE) * 0.44
    max_var = (max_hx / 2.6) ** 2
    scale_clamp = jnp.minimum(1.0, max_var / jnp.maximum(jnp.maximum(cov2_xx, cov2_yy), 1e-4))
    cov2_xx = cov2_xx * scale_clamp
    cov2_xy = cov2_xy * scale_clamp
    cov2_yy = cov2_yy * scale_clamp

    det = cov2_xx * cov2_yy - cov2_xy * cov2_xy
    valid_g = in_front & (det > 1e-6)
    inv_det = jnp.where(valid_g, 1.0 / jnp.maximum(det, 1e-6), 0.0)
    icxx = cov2_yy * inv_det
    icxy = -cov2_xy * inv_det
    icyy = cov2_xx * inv_det

    hx = jnp.minimum(2.6 * jnp.sqrt(jnp.maximum(cov2_xx, 0.1)), max_hx)
    hy = jnp.minimum(2.6 * jnp.sqrt(jnp.maximum(cov2_yy, 0.1)), max_hx)

    on_screen = (
        valid_g
        & (mean_x + hx >= 0.0)
        & (mean_x - hx < float(width))
        & (mean_y + hy >= 0.0)
        & (mean_y - hy < float(height))
    )

    tx0 = jnp.clip(jnp.floor((mean_x - hx) / float(M_SIZE)).astype(jnp.int32), 0, Mx - 1)
    tx1 = jnp.clip(jnp.floor((mean_x + hx) / float(M_SIZE)).astype(jnp.int32), 0, Mx - 1)
    ty0 = jnp.clip(jnp.floor((mean_y - hy) / float(M_SIZE)).astype(jnp.int32), 0, My - 1)
    ty1 = jnp.clip(jnp.floor((mean_y + hy) / float(M_SIZE)).astype(jnp.int32), 0, My - 1)

    z_u32 = jax.lax.bitcast_convert_type(jnp.maximum(z_safe, 1e-5), jnp.uint32) >> jnp.uint32(13)
    INV_MACRO = jnp.uint32(8191)

    def _make_key(v_mask, tx, ty):
      m_id = jnp.where(v_mask, (ty * Mx + tx).astype(jnp.uint32), INV_MACRO)
      return (m_id << jnp.uint32(19)) | z_u32

    key_parts = []
    for dy in range(macro_rep):
      for dx in range(macro_rep):
        vx = on_screen if dx == 0 else (on_screen & ((tx0 + dx) <= tx1))
        vy = vx if dy == 0 else (vx & ((ty0 + dy) <= ty1))
        key_parts.append(_make_key(vy, jnp.minimum(tx0 + dx, Mx - 1), jnp.minimum(ty0 + dy, My - 1)))

    keys = jnp.concatenate(key_parts, axis=0)
    idx_n = jnp.arange(N_loc, dtype=jnp.int32)
    g_ids = jnp.tile(idx_n, macro_rep * macro_rep)

    keys_s, gids_s = jax.lax.sort_key_val(keys, g_ids, dimension=0)
    macro_s = keys_s >> jnp.uint32(19)

    m_queries = jnp.arange(M_tiles, dtype=jnp.uint32)
    starts = jnp.searchsorted(macro_s, m_queries, side="left")
    ends = jnp.searchsorted(macro_s, m_queries, side="right")
    counts = jnp.minimum(ends - starts, k_local)

    gids_pad = jnp.pad(gids_s, (0, k_local), constant_values=0)

    def _slice_macro(st, cnt):
      sl = jax.lax.dynamic_slice(gids_pad, (st,), (k_local,))
      return jnp.where(jnp.arange(k_local, dtype=jnp.int32) < cnt, sl, -1)

    macro_gidx_loc = jax.vmap(_slice_macro)(starts, counts)
    if M_pad > M_tiles:
      macro_gidx_loc = jnp.pad(macro_gidx_loc, ((0, M_pad - M_tiles), (0, 0)), constant_values=-1)

    # Dynamic GGX + Fresnel rim on N_loc
    vdir = cxp[None, :] - mu
    vdir = vdir / jnp.maximum(jnp.linalg.norm(vdir, axis=-1, keepdims=True), 1e-12)
    nrm_vis = jnp.where(jnp.sum(nrm * vdir, axis=-1, keepdims=True) < 0.0, -nrm, nrm)

    ldir = sun_d / jnp.maximum(jnp.linalg.norm(sun_d), 1e-12)
    hvec = ldir[None, :] + vdir
    hvec = hvec / jnp.maximum(jnp.linalg.norm(hvec, axis=-1, keepdims=True), 1e-12)
    ndotl = jnp.maximum(0.0, jnp.sum(nrm_vis * ldir[None, :], axis=-1))
    ndotv = jnp.maximum(1e-3, jnp.sum(nrm_vis * vdir, axis=-1))
    ndoth = jnp.maximum(0.0, jnp.sum(nrm_vis * hvec, axis=-1))
    vdoth = jnp.maximum(0.0, jnp.sum(vdir * hvec, axis=-1))

    alpha_r = jnp.maximum(rough * rough, 0.025)
    a2 = alpha_r * alpha_r
    denom_d = ndoth * ndoth * (a2 - 1.0) + 1.0
    D = a2 / jnp.maximum(math.pi * denom_d * denom_d, 1e-5)
    k_g = (rough + 1.0) ** 2 * 0.125
    G = (ndotv / (ndotv * (1.0 - k_g) + k_g)) * (ndotl / jnp.maximum(ndotl * (1.0 - k_g) + k_g, 1e-4))
    F = spec_col + (1.0 - spec_col) * ((1.0 - vdoth) ** 5)[:, None]
    ggx_spec = (D * G / jnp.maximum(4.0 * ndotv * ndotl, 1e-3))[:, None] * F * (ndotl * sun_vis)[:, None] * sun_c[None, :]
    ggx_spec = jnp.minimum(ggx_spec, 4.0)

    fresnel_rim = ((1.0 - ndotv) ** 4.0) * (1.0 - 0.6 * rough)
    rim_col = (sky_c[None, :] * sky_vis[:, None] * 0.45 + sun_c[None, :] * sun_vis[:, None] * 0.18) * fresnel_rim[:, None]
    rgb_hdr = jnp.maximum(diff_hdr + ggx_spec + rim_col, 0.0)

    # Local 16-ch packed array (N_loc, 16) -> gather locally via valid mask, then ICI all_to_all!
    z_n = jnp.zeros((N_loc,), dtype=jnp.float32)
    splats_loc = jnp.stack(
        [
            mean_x, mean_y, icxx, icxy, icyy,
            rgb_hdr[:, 0], rgb_hdr[:, 1], rgb_hdr[:, 2],
            op, z_d, nrm_vis[:, 0], nrm_vis[:, 1], nrm_vis[:, 2],
            hx, hy, z_n,
        ],
        axis=1,
    )
    safe_ix = jnp.maximum(macro_gidx_loc, 0)
    valid_loc = (macro_gidx_loc >= 0)[..., None]
    # Note: zero out invalid splats so z_d / opacity / overlap stays cleanly invalid
    g16_loc = jnp.where(valid_loc, splats_loc[safe_ix], 0.0)
    # Mark invalid z_d = 1e9 and op = -1 via channel 8 = 0, channel 15 = valid_flag
    g16_loc = g16_loc.at[..., 15].set(jnp.where(macro_gidx_loc >= 0, 1.0, 0.0))

    # ICI all_to_all across 4 chips: (4, M_sub, k_local, 16) -> (4, M_sub, k_local, 16)
    g16_send = g16_loc.reshape(n_chips, M_sub, k_local, 16)
    g16_recv = jax.lax.all_to_all(g16_send, "dp", split_axis=0, concat_axis=0)
    # Reshape to (M_sub, k_macro, 16)
    g16 = g16_recv.transpose(1, 0, 2, 3).reshape(M_sub, k_macro, 16)

    # Now rasterize M_sub macro-tiles owned by this chip!
    valid_slot = g16[..., 15] > 0.5
    mx, my, zd, hx_m, hy_m = g16[..., 0], g16[..., 1], g16[..., 9], g16[..., 13], g16[..., 14]
    splats_b16n = jnp.transpose(g16, (0, 2, 1))  # (M_sub, 16, k_macro)

    t_ox_loc = jnp.asarray(np.tile(np.arange(4, dtype=np.float32) * 8.0, 4), dtype=jnp.float32)[None, :] + tox_sub[:, None]
    t_oy_loc = jnp.asarray(np.repeat(np.arange(4, dtype=np.float32) * 8.0, 4), dtype=jnp.float32)[None, :] + toy_sub[:, None]

    overlap = (
        valid_slot[:, None, :]
        & (mx[:, None, :] + hx_m[:, None, :] >= t_ox_loc[:, :, None])
        & (mx[:, None, :] - hx_m[:, None, :] <= t_ox_loc[:, :, None] + 8.0)
        & (my[:, None, :] + hy_m[:, None, :] >= t_oy_loc[:, :, None])
        & (my[:, None, :] - hy_m[:, None, :] <= t_oy_loc[:, :, None] + 8.0)
    )

    score = jnp.where(overlap, -zd[:, None, :], -1.0e9)
    top_scores, top_idx = jax.lax.approx_max_k(score, k_tile, recall_target=1.0)
    neg_ts, top_idx = jax.lax.sort_key_val(-top_scores, top_idx, dimension=-1)
    top_scores = -neg_ts
    top_idx = jnp.where(top_scores > -1.0e8, top_idx, -1)

    splats_local = splats_b16n.at[:, 0, :].set(mx - tox_sub[:, None])
    splats_local = splats_local.at[:, 1, :].set(my - toy_sub[:, None])

    rgb_t = jnp.zeros((M_sub, 16, 8, 8, 3), dtype=jnp.float32)
    dep_t = jnp.zeros((M_sub, 16, 8, 8), dtype=jnp.float32)
    t_acc = jnp.ones((M_sub, 16, 8, 8), dtype=jnp.float32)
    for c in range(k_tile // 128):
      rc, dc, _, ac = pallas_gs3d_v3_fused_rasterize(
          splats_local,
          top_idx[..., c * 128 : (c + 1) * 128],
          t_ox_0, t_oy_0, tile_size=8, use_bf16_mxu=use_bf16_mxu, t_grp=16, interpret=False,
      )
      rgb_t = rgb_t + t_acc[..., None] * rc
      dep_t = dep_t + t_acc * dc
      t_acc = t_acc * jnp.maximum(1.0 - ac, 0.0)
    alp_t = 1.0 - t_acc

    rgb_m = rgb_t.reshape(M_sub, 4, 4, 8, 8, 3).transpose(0, 1, 3, 2, 4, 5).reshape(M_sub, 32, 32, 3)
    alp_m = alp_t.reshape(M_sub, 4, 4, 8, 8).transpose(0, 1, 3, 2, 4).reshape(M_sub, 32, 32)
    dep_m = dep_t.reshape(M_sub, 4, 4, 8, 8).transpose(0, 1, 3, 2, 4).reshape(M_sub, 32, 32)
    z_m = jnp.where(alp_m > 0.05, dep_m / jnp.maximum(alp_m, 1e-4), -1.0)
    return rgb_m, z_m, alp_m

  _sm_gs = shard_map(
      _sharded_gs_body, mesh=mesh,
      in_specs=(
          P("dp"), P("dp"), P("dp"), P("dp"), P("dp"), P("dp"), P("dp"), P("dp"), P("dp"),
          P(), P(), P(), P(), P(), P("dp"), P("dp"),
      ),
      out_specs=(P("dp"), P("dp"), P("dp")),
      **sm_kw,
  )

  t_ox_dev = jax.device_put(jnp.asarray(t_ox_np), jax.sharding.NamedSharding(mesh, P("dp")))
  t_oy_dev = jax.device_put(jnp.asarray(t_oy_np), jax.sharding.NamedSharding(mesh, P("dp")))

  @jax.jit
  def _run_4c(mu, cov6, diff_hdr, op, nrm, spec_col, rough, sun_vis, sky_vis, cxp, cxm, sun_d, sun_c, sky_c):
    rgb_m, z_m, alp_m = _sm_gs(
        mu, cov6, diff_hdr, op, nrm, spec_col, rough, sun_vis, sky_vis,
        cxp, cxm, sun_d, sun_c, sky_c, t_ox_dev, t_oy_dev,
    )
    rgb_m = rgb_m[:M_tiles]
    z_m = z_m[:M_tiles]
    alp_m = alp_m[:M_tiles]
    img_pad = rgb_m.reshape(My, Mx, 32, 32, 3).transpose(0, 2, 1, 3, 4).reshape(H_pad, W_pad, 3)
    dep_pad = z_m.reshape(My, Mx, 32, 32).transpose(0, 2, 1, 3).reshape(H_pad, W_pad)
    alp_pad = alp_m.reshape(My, Mx, 32, 32).transpose(0, 2, 1, 3).reshape(H_pad, W_pad)
    return img_pad[:height, :width, :], dep_pad[:height, :width], alp_pad[:height, :width]

  return _run_4c


# ---------------------------------------------------------------------------
# 3. 4-Chip (8, 512) Mosaic Pallas Exact MJX Raycaster + Ground Shadow-Catcher
# ---------------------------------------------------------------------------

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
    nx = (dx + t * rdx) * inv_s
    ny = (dy + t * rdy) * inv_s
    nz = (dz + t * rdz) * inv_s
    return cand_d, nx, ny, nz

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
    ok_p = sphere_hit & (aa > MJ_MINVAL) & (detp >= MJ_MINVAL) & (tp >= 0.0)
    dp = jnp.where(ok_p, tp, MJ_MAXVAL)

    bm = vx * lx + vy * ly + vz * dzm
    cm = lx * lx + ly * ly + dzm * dzm - sx * sx
    detm = bm * bm - aa * cm
    sdm = jnp.sqrt(jnp.maximum(detm, 0.0))
    tm = jnp.where((-bm - sdm) * inv_aa >= 0.0, (-bm - sdm) * inv_aa, (-bm + sdm) * inv_aa)
    ok_m = sphere_hit & (aa > MJ_MINVAL) & (detm >= MJ_MINVAL) & (tm >= 0.0)
    dm = jnp.where(ok_m, tm, MJ_MAXVAL)

    best_d = jnp.minimum(d_cyl, jnp.minimum(dp, dm))
    if not compute_normal:
      return best_d, 0.0, 0.0, 0.0

    inv_r = 1.0 / jnp.maximum(sx, 1e-12)
    n_cyl_x = (lx + d_cyl * vx) * inv_r
    n_cyl_y = (ly + d_cyl * vy) * inv_r
    n_cyl_z = 0.0
    n_p_x = (lx + dp * vx) * inv_r
    n_p_y = (ly + dp * vy) * inv_r
    n_p_z = (dzp + dp * vz) * inv_r
    n_m_x = (lx + dm * vx) * inv_r
    n_m_y = (ly + dm * vy) * inv_r
    n_m_z = (dzm + dm * vz) * inv_r

    use_p = dp < d_cyl
    lnx = jnp.where(use_p, n_p_x, n_cyl_x)
    lny = jnp.where(use_p, n_p_y, n_cyl_y)
    lnz = jnp.where(use_p, n_p_z, n_cyl_z)
    use_m = dm < jnp.minimum(d_cyl, dp)
    lnx = jnp.where(use_m, n_m_x, lnx)
    lny = jnp.where(use_m, n_m_y, lny)
    lnz = jnp.where(use_m, n_m_z, lnz)
    nx = m00 * lnx + m01 * lny + m02 * lnz
    ny = m10 * lnx + m11 * lny + m12 * lnz
    nz = m20 * lnx + m21 * lny + m22 * lnz
    return best_d, nx, ny, nz

  if gt == GEOM_CYLINDER:
    a = vx * vx + vy * vy
    b = lx * vx + ly * vy
    c = lx * lx + ly * ly - sx * sx
    det = b * b - a * c
    has_a = jnp.abs(a) > MJ_MINVAL
    sd = jnp.sqrt(jnp.maximum(det, 0.0))
    inv_a = jnp.where(has_a, 1.0 / jnp.where(has_a, a, 1.0), 0.0)
    t1 = (-b - sd) * inv_a
    t2 = (-b + sd) * inv_a
    z1 = lz + t1 * vz
    z2 = lz + t2 * vz
    ok1 = sphere_hit & has_a & (det >= MJ_MINVAL) & (t1 >= 0.0) & (jnp.abs(z1) <= sy)
    ok2 = sphere_hit & has_a & (det >= MJ_MINVAL) & (t2 >= 0.0) & (jnp.abs(z2) <= sy)
    d1 = jnp.where(ok1, t1, MJ_MAXVAL)
    d2 = jnp.where(ok2, t2, MJ_MAXVAL)
    use1 = d1 <= d2
    d_side = jnp.where(use1, d1, d2)

    has_vz = jnp.abs(vz) > MJ_MINVAL
    inv_vz = jnp.where(has_vz, 1.0 / jnp.where(has_vz, vz, 1.0), 0.0)
    tp = (sy - lz) * inv_vz
    tm = (-sy - lz) * inv_vz
    xp, yp = lx + tp * vx, ly + tp * vy
    xm, ym = lx + tm * vx, ly + tm * vy
    ok_p = sphere_hit & has_vz & (tp >= 0.0) & (xp * xp + yp * yp <= sx * sx)
    ok_m = sphere_hit & has_vz & (tm >= 0.0) & (xm * xm + ym * ym <= sx * sx)
    dp = jnp.where(ok_p, tp, MJ_MAXVAL)
    dm = jnp.where(ok_m, tm, MJ_MAXVAL)

    best_d = jnp.minimum(d_side, jnp.minimum(dp, dm))
    if not compute_normal:
      return best_d, 0.0, 0.0, 0.0

    inv_r = 1.0 / jnp.maximum(sx, 1e-12)
    lsx = (lx + d_side * vx) * inv_r
    lsy = (ly + d_side * vy) * inv_r
    use_p = dp < d_side
    lnx = jnp.where(use_p, 0.0, lsx)
    lny = jnp.where(use_p, 0.0, lsy)
    lnz = jnp.where(use_p, 1.0, 0.0)
    use_m = dm < jnp.minimum(d_side, dp)
    lnx = jnp.where(use_m, 0.0, lnx)
    lny = jnp.where(use_m, 0.0, lny)
    lnz = jnp.where(use_m, -1.0, lnz)
    nx = m00 * lnx + m01 * lny + m02 * lnz
    ny = m10 * lnx + m11 * lny + m12 * lnz
    nz = m20 * lnx + m21 * lny + m22 * lnz
    return best_d, nx, ny, nz

  # GEOM_BOX (default)
  inv_vx = jnp.where(jnp.abs(vx) > MJ_MINVAL, 1.0 / jnp.where(jnp.abs(vx) > MJ_MINVAL, vx, 1.0), 1.0e12)
  inv_vy = jnp.where(jnp.abs(vy) > MJ_MINVAL, 1.0 / jnp.where(jnp.abs(vy) > MJ_MINVAL, vy, 1.0), 1.0e12)
  inv_vz = jnp.where(jnp.abs(vz) > MJ_MINVAL, 1.0 / jnp.where(jnp.abs(vz) > MJ_MINVAL, vz, 1.0), 1.0e12)

  tx1, tx2 = (-sx - lx) * inv_vx, (sx - lx) * inv_vx
  ty1, ty2 = (-sy - ly) * inv_vy, (sy - ly) * inv_vy
  tz1, tz2 = (-sz - lz) * inv_vz, (sz - lz) * inv_vz

  tx_lo, tx_hi = jnp.minimum(tx1, tx2), jnp.maximum(tx1, tx2)
  ty_lo, ty_hi = jnp.minimum(ty1, ty2), jnp.maximum(ty1, ty2)
  tz_lo, tz_hi = jnp.minimum(tz1, tz2), jnp.maximum(tz1, tz2)

  tmin = jnp.maximum(tx_lo, jnp.maximum(ty_lo, tz_lo))
  tmax = jnp.minimum(tx_hi, jnp.minimum(ty_hi, tz_hi))
  t = jnp.where(tmin >= 0.0, tmin, tmax)
  ok = sphere_hit & (tmax >= tmin) & (tmax >= 0.0) & (t >= 0.0)
  cand_d = jnp.where(ok, t, MJ_MAXVAL)
  if not compute_normal:
    return cand_d, 0.0, 0.0, 0.0

  hit_x = (tx_lo >= ty_lo) & (tx_lo >= tz_lo)
  hit_y = (~hit_x) & (ty_lo >= tz_lo)
  sgn_x = jnp.where(vx >= 0.0, -1.0, 1.0)
  sgn_y = jnp.where(vy >= 0.0, -1.0, 1.0)
  sgn_z = jnp.where(vz >= 0.0, -1.0, 1.0)
  lnx = jnp.where(hit_x, sgn_x, 0.0)
  lny = jnp.where(hit_y, sgn_y, 0.0)
  lnz = jnp.where((~hit_x) & (~hit_y), sgn_z, 0.0)
  nx = m00 * lnx + m01 * lny + m02 * lnz
  ny = m10 * lnx + m11 * lny + m12 * lnz
  nz = m20 * lnx + m21 * lny + m22 * lnz
  return cand_d, nx, ny, nz


def pallas_raycast_mjx_4k_strip(
    spec: RenderSceneSpec,
    geom_xpos: jax.Array,   # (G, 3)
    geom_xmat: jax.Array,   # (G, 3, 3)
    geom_size: jax.Array,   # (G, 3)
    geom_rgba: jax.Array,   # (G, 4)
    cam_xpos: jax.Array,    # (3,)
    cam_xmat: jax.Array,    # (3, 3)
    sun_dir: jax.Array,     # (3,) unit vector toward sun
    ray_orig: jax.Array,    # (R_sub, 3)
    ray_dir: jax.Array,     # (R_sub, 3)
    neg_z: jax.Array,       # (R_sub,)
    ground_y: float = 0.22,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
  """Mosaic Pallas (8, 512) MJX raycaster + self-shadow + cobblestone shadow-catcher."""
  gids = [int(g) for g in spec.enabled_geom_ids if int(spec.geom_type[g]) != GEOM_PLANE]
  G = len(gids)
  G_pad = ((max(G, 1) + 127) // 128) * 128
  R_sub = ray_orig.shape[0]
  BLOCK_B = 8
  BLOCK_R = 512
  R_per_row = ((R_sub + BLOCK_B * BLOCK_R - 1) // (BLOCK_B * BLOCK_R)) * BLOCK_R
  R_pad_tot = BLOCK_B * R_per_row
  n_r_blk = R_per_row // BLOCK_R

  # Pack rays into (6 * 8 + 8 = 56 -> pad to 56? Wait: 56 is divisible by 8!)
  # Channels: rox, roy, roz, rdx, rdy, rdz, neg_z -> 7 * 8 = 56 rows!
  pad_tot = R_pad_tot - R_sub
  def _pack_1d(arr, val=0.0):
    if pad_tot > 0:
      arr = jnp.pad(arr, (0, pad_tot), constant_values=val)
    return arr.reshape(BLOCK_B, R_per_row)

  ray_blk_in = jnp.concatenate([
      _pack_1d(ray_orig[:, 0]),
      _pack_1d(ray_orig[:, 1]),
      _pack_1d(ray_orig[:, 2]),
      _pack_1d(ray_dir[:, 0]),
      _pack_1d(ray_dir[:, 1]),
      _pack_1d(ray_dir[:, 2], -1.0),
      _pack_1d(neg_z, 1.0),
  ], axis=0)  # (56, R_per_row)

  # Pack geoms into (16, G_pad)
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
  ], axis=0)  # (16, G)
  if G_pad > G:
    gf_dev = jnp.pad(gf_raw, ((0, 0), (0, G_pad - G)))
  else:
    gf_dev = gf_raw

  # Pack sun_dir + znear into (8, 128)
  s_blk = jnp.zeros((8, 128), dtype=jnp.float32)
  s_blk = s_blk.at[0, :].set(sun_dir[0])
  s_blk = s_blk.at[1, :].set(sun_dir[1])
  s_blk = s_blk.at[2, :].set(sun_dir[2])

  g_types_static = tuple(int(spec.geom_type[g]) for g in gids)
  gids_static = tuple(gids)
  g_rgba_static = tuple(tuple(float(c) for c in spec.geom_rgba[g, :3]) for g in gids)
  znear_f = float(spec.znear)
  zfar_f = float(spec.zfar)

  # Group geoms by color for fast VMEM material lookup
  mat_groups: Dict[Tuple[float, float, float], list[int]] = {}
  for k, gid_val in enumerate(gids_static):
    mat_groups.setdefault(g_rgba_static[k], []).append(gid_val)

  def _kernel(r_ref, gf_ref, s_ref, out_ref):
    rb = r_ref[...]    # (56, 512)
    gf = gf_ref[...]   # (16, G_pad)
    sb = s_ref[...]    # (8, 128)

    rox, roy, roz = rb[0:8, :], rb[8:16, :], rb[16:24, :]
    rdx, rdy, rdz = rb[24:32, :], rb[32:40, :], rb[40:48, :]
    nz_b = rb[48:56, :]
    sdx, sdy, sdz = sb[0:1, 0:1], sb[1:2, 0:1], sb[2:3, 0:1]

    best_d = (zfar_f - znear_f) / jnp.maximum(nz_b, 1e-6)
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

    # Albedo lookup (material-deduped)
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

    # Self-shadow ray for robot hits + Ground-plane shadow catcher for misses!
    # Ground plane y = ground_y
    front_gnd = rdy < -1e-4
    t_gnd = jnp.where(front_gnd, (ground_y - roy) / jnp.where(front_gnd, rdy, -1.0), -1.0)
    gnd_ok = (~hit) & front_gnd & (t_gnd > 0.0) & (t_gnd < 35.0)
    gx = rox + jnp.where(gnd_ok, t_gnd, 0.0) * rdx
    gy = ground_y + 2e-3
    gz = roz + jnp.where(gnd_ok, t_gnd, 0.0) * rdz

    sh_ox = jnp.where(hit, hx + best_nx * 2e-3, gx)
    sh_oy = jnp.where(hit, hy + best_ny * 2e-3, gy)
    sh_oz = jnp.where(hit, hz + best_nz * 2e-3, gz)
    do_sh = hit | gnd_ok

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
      occ_bool = occ_bool | (scd < 25.0)

    robot_sh = jnp.where(hit & occ_bool, 0.0, 1.0)
    gnd_sh_mask = jnp.where(gnd_ok & occ_bool, 1.0, 0.0)
    z_depth = jnp.where(hit, znear_f + best_d * nz_b, -1.0)
    gnd_z = jnp.where(gnd_ok, znear_f + t_gnd * nz_b, -1.0)

    # Pack 10 * 8 = 80 rows
    out_ref[...] = jnp.concatenate([
        alb_r, alb_g, alb_b,
        jnp.where(hit, best_nx, 0.0),
        jnp.where(hit, best_ny, 0.0),
        jnp.where(hit, best_nz, 0.0),
        z_depth,
        robot_sh,
        gnd_sh_mask,
        gnd_z,
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
  gnd_sh_mask = _unpack(8)
  gnd_z = _unpack(9)
  return alb, nrm, z_depth, robot_sh, gnd_sh_mask, gnd_z, -ray_dir


# ---------------------------------------------------------------------------
# 4. 4-Chip Unified Cinema Hybrid Compositor + AgX Tonemapper
# ---------------------------------------------------------------------------

def composite_cinema_hdr_and_tonemap_4c(
    gs_rgb_prem: jax.Array,   # (H, W, 3)
    gs_alpha: jax.Array,      # (H, W)
    gs_depth: jax.Array,      # (H, W)
    ray_dir_hw3: jax.Array,   # (H, W, 3)
    sun_dir: jax.Array,       # (3,)
    mjx_rgb_hdr: jax.Array,   # (H, W, 3)
    mjx_depth: jax.Array,     # (H, W)
    gnd_sh_mask: jax.Array,   # (H, W)
    gnd_z: jax.Array,         # (H, W)
    fog_density: float = 0.0045,
    fog_start: float = 18.0,
    max_fog: float = 0.15,
    fog_color: Tuple[float, float, float] = (1.05, 0.58, 0.28),
    exposure: float = 1.15,
    bloom_strength: float = 0.055,
    bloom_threshold: float = 1.45,
) -> Tuple[jax.Array, jax.Array]:
  """Composite HDR 3DGS + MJX robots + cast street shadows + Nishita Sky + AgX."""
  sky_hdr = nishita_golden_hour_sky_tpu(ray_dir_hw3, sun_dir)
  gs_comp_hdr = gs_rgb_prem + (1.0 - gs_alpha[..., None]) * sky_hdr

  # Modulate 3DGS cobblestone street where MJX robots cast golden-hour shadow!
  on_street = (gnd_sh_mask > 0.5) & (gs_depth > 0.0) & (jnp.abs(gs_depth - gnd_z) < 0.45)
  gs_comp_hdr = jnp.where(on_street[..., None], gs_comp_hdr * 0.42, gs_comp_hdr)

  mjx_hit = mjx_depth > 0.0
  gs_hit = (gs_alpha > 0.05) & (gs_depth > 0.0)
  mjx_front = mjx_hit & (~gs_hit | (mjx_depth <= gs_depth))
  hdr = jnp.where(mjx_front[..., None], mjx_rgb_hdr, gs_comp_hdr)
  comp_depth = jnp.where(mjx_front, mjx_depth, jnp.where(gs_hit, gs_depth, -1.0))

  has_hit = comp_depth > 0.0
  dist_fog = jnp.maximum(comp_depth - fog_start, 0.0)
  fog_raw = jnp.where(has_hit, 1.0 - jnp.exp(-fog_density * dist_fog), 0.0)
  fog_amt = jnp.minimum(fog_raw, max_fog)[..., None]
  sun_d = sun_dir / jnp.maximum(jnp.linalg.norm(sun_dir), 1e-12)
  cos_s = jnp.clip(jnp.sum(ray_dir_hw3 * sun_d[None, None, :], axis=-1), 0.0, 1.0)[..., None]
  fog_inscatter = jnp.asarray(fog_color, dtype=jnp.float32)[None, None, :] * (0.55 + 0.65 * (cos_s ** 3))
  hdr = hdr * (1.0 - fog_amt) + fog_inscatter * fog_amt

  srgb = blender_agx_tonemap_tpu(hdr, exposure=exposure, bloom_strength=bloom_strength, bloom_threshold=bloom_threshold)
  return srgb, comp_depth
