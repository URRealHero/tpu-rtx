"""Closed-Loop Zero-Copy tpu_rtx Vision PPO (4-Chip shard_map) on TPU v5p.

Integrates `tpu_rtx.compile` + `tpu_rtx.render` directly inside a 4-chip `shard_map`
Vision PPO training loop across Cartpole, PandaPickCube, and Humanoid-21DOF with:
- Zero host callbacks, zero send/recv, zero XLA gather ops (`host_callbacks=0`).
- 2-frame stacked RGB observations in HBM (`(B, H, W, 6)`) for full velocity observability
  at zero extra rendering cost.
- `mjx.kinematics` + `mjx.com_pos` lightweight reset path inside rollout scan (avoiding
  redundant constraint solves on non-terminal steps while keeping `gather_ops == 0`).
- Critic-stabilized PPO (`w7_ppo_4c` patterns): running EMA obs/return normalization
  synced via `pmean`, CoordConv spatial channels, clipped value loss, proper timeout
  truncation bootstrapping, global `pmean` advantage & gradient synchronization.
"""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
import os
import time
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import flax.linen as nn
import jax
try:
  from jax import shard_map
except ImportError:
  from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np
import optax

import tpu_rtx
from tpu_mjx_render.envs import EnvState
from tpu_mjx_render.scenes import get_scene_xml


HUMANOID_PPO_XML = """<mujoco model="humanoid_21dof_ppo">
  <option timestep="0.005" iterations="2" ls_iterations="4" integrator="implicitfast" solver="CG">
    <flag eulerdamp="disable"/>
  </option>
  <visual>
    <headlight ambient="0.25 0.25 0.25" diffuse="0.45 0.45 0.45" specular="0.15 0.15 0.15"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.24 0.40 0.62" rgb2="0.05 0.08 0.12" width="128" height="128"/>
    <texture name="grid" type="2d" builtin="checker" mark="edge" rgb1="0.22 0.30 0.38" rgb2="0.14 0.19 0.26" markrgb="0.75 0.78 0.82" width="128" height="128"/>
    <material name="grid" texture="grid" texuniform="true" texrepeat="4 4" reflectance="0.2"/>
    <material name="body_orange" rgba="0.88 0.45 0.18 1" specular="0.4" shininess="0.35"/>
    <material name="joint_dark" rgba="0.20 0.22 0.26 1" specular="0.5" shininess="0.4"/>
    <material name="limb_silver" rgba="0.78 0.82 0.86 1" specular="0.45" shininess="0.35"/>
  </asset>
  <default>
    <joint armature="0.03" damping="2.5" limited="true"/>
    <geom condim="3" friction="0.9 0.02 0.002" solref="0.02 1" contype="0" conaffinity="1"/>
    <motor ctrllimited="true" ctrlrange="-1 1"/>
  </default>
  <worldbody>
    <light name="sun" pos="1.5 -1.5 3.5" dir="-0.3 0.3 -1" directional="false" castshadow="true" diffuse="0.9 0.88 0.84" specular="0.3 0.3 0.3"/>
    <camera name="front" pos="2.4 -1.8 1.35" xyaxes="0.60 0.80 0 -0.20 0.15 0.97" fovy="50"/>
    <geom name="floor" type="plane" pos="0 0 0" size="8 8 0.1" material="grid" contype="1" conaffinity="0"/>
    <geom name="pedestal_block" type="box" pos="0.95 0.55 0.15" size="0.18 0.18 0.15" rgba="0.22 0.62 0.85 1" contype="0" conaffinity="0"/>
    <geom name="arena_sphere" type="sphere" pos="-0.85 0.65 0.22" size="0.22" rgba="0.86 0.24 0.22 1" contype="0" conaffinity="0"/>
    <geom name="arena_cyl" type="cylinder" pos="0.55 -0.85 0.20" size="0.14 0.20" rgba="0.24 0.78 0.44 1" contype="0" conaffinity="0"/>
    <body name="torso" pos="0 0 1.28">
      <freejoint name="root"/>
      <geom name="torso_geom" type="capsule" fromto="0 -0.07 0 0 0.07 0" size="0.095" material="body_orange" mass="5.0"/>
      <geom name="head" type="sphere" pos="0 0 0.21" size="0.105" material="limb_silver" mass="1.5"/>
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
            <joint name="right_hip_y" type="hinge" axis="0 1 0" range="-110 20"/>
            <geom name="right_thigh_g" type="capsule" fromto="0 0 0 0 0.01 -0.34" size="0.06" material="limb_silver" mass="3.0"/>
            <body name="right_shin" pos="0 0.01 -0.40">
              <joint name="right_knee" type="hinge" axis="0 -1 0" range="-160 -2"/>
              <geom name="right_shin_g" type="capsule" fromto="0 0 0 0 0 -0.30" size="0.049" material="body_orange" mass="2.0"/>
              <body name="right_foot" pos="0 0 -0.36">
                <joint name="right_ankle_x" type="hinge" axis="1 0 0" range="-45 45"/>
                <joint name="right_ankle_y" type="hinge" axis="0 1 0" range="-50 50"/>
                <geom name="right_foot_g" type="box" pos="0.04 0 -0.02" size="0.09 0.045 0.025" material="joint_dark" mass="1.0"/>
              </body>
            </body>
          </body>
          <body name="left_thigh" pos="0 0.10 -0.04">
            <joint name="left_hip_x" type="hinge" axis="-1 0 0" range="-25 5"/>
            <joint name="left_hip_z" type="hinge" axis="0 0 -1" range="-60 35"/>
            <joint name="left_hip_y" type="hinge" axis="0 1 0" range="-110 20"/>
            <geom name="left_thigh_g" type="capsule" fromto="0 0 0 0 -0.01 -0.34" size="0.06" material="limb_silver" mass="3.0"/>
            <body name="left_shin" pos="0 -0.01 -0.40">
              <joint name="left_knee" type="hinge" axis="0 -1 0" range="-160 -2"/>
              <geom name="left_shin_g" type="capsule" fromto="0 0 0 0 0 -0.30" size="0.049" material="body_orange" mass="2.0"/>
              <body name="left_foot" pos="0 0 -0.36">
                <joint name="left_ankle_x" type="hinge" axis="1 0 0" range="-45 45"/>
                <joint name="left_ankle_y" type="hinge" axis="0 1 0" range="-50 50"/>
                <geom name="left_foot_g" type="box" pos="0.04 0 -0.02" size="0.09 0.045 0.025" material="joint_dark" mass="1.0"/>
              </body>
            </body>
          </body>
        </body>
      </body>
      <body name="right_upper_arm" pos="0 -0.17 0.06">
        <joint name="right_shoulder1" type="hinge" axis="2 1 1" range="-85 60"/>
        <joint name="right_shoulder2" type="hinge" axis="0 -1 1" range="-85 60"/>
        <geom name="right_uarm_g" type="capsule" fromto="0 0 0 0.16 -0.16 -0.16" size="0.042" material="limb_silver" mass="1.2"/>
        <body name="right_lower_arm" pos="0.18 -0.18 -0.18">
          <joint name="right_elbow" type="hinge" axis="0 -1 1" range="-90 50"/>
          <geom name="right_larm_g" type="capsule" fromto="0.01 0.01 0.01 0.17 0.17 0.17" size="0.034" material="body_orange" mass="0.8"/>
          <geom name="right_hand" type="sphere" pos="0.18 0.18 0.18" size="0.042" material="joint_dark" mass="0.3"/>
        </body>
      </body>
      <body name="left_upper_arm" pos="0 0.17 0.06">
        <joint name="left_shoulder1" type="hinge" axis="2 -1 1" range="-60 85"/>
        <joint name="left_shoulder2" type="hinge" axis="0 1 1" range="-60 85"/>
        <geom name="left_uarm_g" type="capsule" fromto="0 0 0 0.16 0.16 -0.16" size="0.042" material="limb_silver" mass="1.2"/>
        <body name="left_lower_arm" pos="0.18 0.18 -0.18">
          <joint name="left_elbow" type="hinge" axis="0 -1 -1" range="-90 50"/>
          <geom name="left_larm_g" type="capsule" fromto="0.01 -0.01 0.01 0.17 -0.17 0.17" size="0.034" material="body_orange" mass="0.8"/>
          <geom name="left_hand" type="sphere" pos="0.18 -0.18 0.18" size="0.042" material="joint_dark" mass="0.3"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="abdomen_y" gear="100" joint="abdomen_y"/>
    <motor name="abdomen_z" gear="100" joint="abdomen_z"/>
    <motor name="abdomen_x" gear="100" joint="abdomen_x"/>
    <motor name="right_hip_x" gear="100" joint="right_hip_x"/>
    <motor name="right_hip_z" gear="100" joint="right_hip_z"/>
    <motor name="right_hip_y" gear="200" joint="right_hip_y"/>
    <motor name="right_knee" gear="150" joint="right_knee"/>
    <motor name="right_ankle_x" gear="50" joint="right_ankle_x"/>
    <motor name="right_ankle_y" gear="50" joint="right_ankle_y"/>
    <motor name="left_hip_x" gear="100" joint="left_hip_x"/>
    <motor name="left_hip_z" gear="100" joint="left_hip_z"/>
    <motor name="left_hip_y" gear="200" joint="left_hip_y"/>
    <motor name="left_knee" gear="150" joint="left_knee"/>
    <motor name="left_ankle_x" gear="50" joint="left_ankle_x"/>
    <motor name="left_ankle_y" gear="50" joint="left_ankle_y"/>
    <motor name="right_shoulder1" gear="35" joint="right_shoulder1"/>
    <motor name="right_shoulder2" gear="35" joint="right_shoulder2"/>
    <motor name="right_elbow" gear="35" joint="right_elbow"/>
    <motor name="left_shoulder1" gear="35" joint="left_shoulder1"/>
    <motor name="left_shoulder2" gear="35" joint="left_shoulder2"/>
    <motor name="left_elbow" gear="35" joint="left_elbow"/>
  </actuator>
</mujoco>
"""


class NormState(NamedTuple):
  obs_mean: jax.Array   # (1, 1, 1, 3)
  obs_var: jax.Array    # (1, 1, 1, 3)
  ret_var: jax.Array    # scalar


class VisionActorCritic(nn.Module):
  """Fast BF16 CoordConv CNN + LayerNorm + MLP Actor-Critic (2-frame stacked RGB)."""

  action_dim: int

  @nn.compact
  def __call__(self, obs_norm_2f: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
    # obs_norm_2f: (B, H, W, 6) = concat([curr_rgb_norm, prev_rgb_norm], -1)
    b, h_dim, w_dim, _ = obs_norm_2f.shape
    ys = jnp.linspace(-1.0, 1.0, h_dim, dtype=jnp.float32)
    xs = jnp.linspace(-1.0, 1.0, w_dim, dtype=jnp.float32)
    yy, xx = jnp.meshgrid(ys, xs, indexing="ij")
    coords = jnp.broadcast_to(jnp.stack([xx, yy], axis=-1)[None, ...], (b, h_dim, w_dim, 2))
    x = jnp.concatenate([obs_norm_2f, coords], axis=-1).astype(jnp.bfloat16)

    for ch in (32, 64, 64, 64):
      x = nn.relu(
          nn.Conv(
              ch,
              (3, 3),
              strides=(2, 2),
              dtype=jnp.bfloat16,
              param_dtype=jnp.float32,
              kernel_init=nn.initializers.orthogonal(np.sqrt(2.0)),
          )(x)
      )
    x = x.reshape((x.shape[0], -1))
    h = nn.relu(
        nn.Dense(
            256,
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
            kernel_init=nn.initializers.orthogonal(np.sqrt(2.0)),
        )(x)
    ).astype(jnp.float32)
    h = nn.LayerNorm()(h)
    mean = nn.Dense(self.action_dim, kernel_init=nn.initializers.orthogonal(0.01))(h)
    log_std = self.param("log_std", nn.initializers.constant(-0.6), (self.action_dim,))
    value = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(h)[..., 0]
    return mean, jnp.clip(log_std, -1.6, 0.3), value


class RTXCartpoleEnv:
  """CartpoleBalance Vision RL environment powered by `tpu_rtx`."""

  def __init__(
      self,
      res: int = 64,
      episode_length: int = 100,
      action_repeat: int = 2,
      n_chips: int = 4,
      shader: str = "fast_fused",
  ):
    self.mj_model = mujoco.MjModel.from_xml_string(get_scene_xml("cartpole"))
    self.mj_model.opt.iterations = 1
    self.mj_model.opt.ls_iterations = 4
    self.mjx_model = mjx.put_model(self.mj_model, impl="jax")
    self.res = res
    self.episode_length = episode_length
    self.action_repeat = action_repeat
    self.action_size = int(self.mj_model.nu)
    self.scene = tpu_rtx.compile(
        self.mj_model,
        shader=shader,
        res=(res, res),
        sharding=n_chips,
        dtype="bfloat16",
        outputs=("rgb",),
        block_b=16,
        block_r=512,
    )
    self.scene_local = dataclasses.replace(self.scene, mesh=None, n_chips=1)
    self._dx_template = mjx.make_data(self.mj_model, impl="jax")

  def reset_physics(self, mx_l: Any, rng: jax.Array, stagger_steps: bool = False) -> EnvState:
    k1, k2, k3, k_st = jax.random.split(rng, 4)
    qpos = jnp.array(
        [
            jax.random.uniform(k1, (), minval=-0.40, maxval=0.40),
            jax.random.uniform(k2, (), minval=-0.42, maxval=0.42),
        ],
        dtype=jnp.float32,
    )
    qvel = jnp.array(
        [
            jax.random.uniform(k3, (), minval=-0.35, maxval=0.35),
            jax.random.uniform(k3, (), minval=-0.45, maxval=0.45),
        ],
        dtype=jnp.float32,
    )
    dx = self._dx_template.replace(qpos=qpos, qvel=qvel)
    dx = mjx.kinematics(mx_l, dx)
    dx = mjx.com_pos(mx_l, dx)
    dx = mjx.camlight(mx_l, dx)
    step_c = (
        jax.random.randint(k_st, (), 0, max(1, self.episode_length), dtype=jnp.int32)
        if stagger_steps
        else jnp.array(0, dtype=jnp.int32)
    )
    pole_cos = jnp.cos(dx.qpos[1])
    balanced = (jnp.abs(dx.qpos[1]) < 0.25).astype(jnp.float32)
    zero_img = jnp.zeros((self.res, self.res, 3), dtype=jnp.bfloat16)
    return EnvState(
        data=dx,
        obs={"rgb": zero_img, "prev_rgb": zero_img},
        reward=jnp.array(0.0, dtype=jnp.float32),
        done=jnp.array(0.0, dtype=jnp.float32),
        step_count=step_c,
        info={
            "pole_cos": pole_cos,
            "balanced": balanced,
            "trunc": jnp.array(0.0, dtype=jnp.float32),
        },
    )

  def step_physics(self, mx_l: Any, state: EnvState, action: jax.Array) -> EnvState:
    u = jnp.clip(action, -1.0, 1.0)
    if self.action_repeat == 1:
      dx = mjx.step(mx_l, state.data.replace(ctrl=u))
    else:
      def _sub(_, d):
        return mjx.step(mx_l, d.replace(ctrl=u))
      dx = jax.lax.fori_loop(0, self.action_repeat, _sub, state.data)

    cart_x = dx.qpos[0]
    pole_cos = jnp.cos(dx.qpos[1])
    reward = 0.5 + 0.5 * pole_cos - 0.05 * (cart_x ** 2) - 0.01 * jnp.sum(dx.qvel ** 2)
    step_c = state.step_count + 1
    out = (jnp.abs(dx.qpos[1]) > (jnp.pi / 2.0)) | (jnp.abs(cart_x) > 1.8)
    timeout = step_c >= self.episode_length
    done = (out | timeout).astype(jnp.float32)
    trunc = (timeout & (~out)).astype(jnp.float32)
    balanced = ((jnp.abs(dx.qpos[1]) < 0.25) & (~out)).astype(jnp.float32)
    return EnvState(
        data=dx,
        obs=state.obs,
        reward=reward,
        done=done,
        step_count=step_c,
        info={
            "pole_cos": pole_cos,
            "balanced": balanced,
            "trunc": trunc,
        },
    )

  def render_local(self, data_batch: Any) -> jax.Array:
    return tpu_rtx.render(data_batch, self.scene_local)["rgb"]


class RTXPandaPickCubeEnv:
  """PandaPickCube Vision RL environment powered by `tpu_rtx` (it2_ls4 verified parity)."""

  def __init__(
      self,
      res: int = 64,
      episode_length: int = 80,
      action_repeat: int = 1,
      iterations: int = 2,
      ls_iterations: int = 4,
      n_chips: int = 4,
      shader: str = "fast_fused",
  ):
    self.mj_model = mujoco.MjModel.from_xml_string(get_scene_xml("panda_pick_cube"))
    self.mj_model.opt.iterations = int(iterations)
    self.mj_model.opt.ls_iterations = int(ls_iterations)
    self.mjx_model = mjx.put_model(self.mj_model, impl="jax")
    self.res = res
    self.episode_length = episode_length
    self.action_repeat = action_repeat
    self.action_size = int(self.mj_model.nu)
    self._box_body = self.mj_model.body("box").id
    self._gripper_site = self.mj_model.site("gripper").id
    self._init_q = jnp.asarray(self.mj_model.keyframe("low_home").qpos, dtype=jnp.float32)
    self._init_ctrl = jnp.asarray(self.mj_model.keyframe("low_home").ctrl, dtype=jnp.float32)
    self._ctrl_lo = jnp.asarray(self.mj_model.actuator_ctrlrange[:, 0], dtype=jnp.float32)
    self._ctrl_hi = jnp.asarray(self.mj_model.actuator_ctrlrange[:, 1], dtype=jnp.float32)
    self._dx_template = mjx.make_data(self.mj_model, impl="jax")
    self.scene = tpu_rtx.compile(
        self.mj_model,
        shader=shader,
        res=(res, res),
        sharding=n_chips,
        dtype="bfloat16",
        outputs=("rgb",),
        block_b=16,
        block_r=512,
    )
    self.scene_local = dataclasses.replace(self.scene, mesh=None, n_chips=1)

  def reset_physics(self, mx_l: Any, rng: jax.Array, stagger_steps: bool = False) -> EnvState:
    k_box, k_arm, k_tgt, k_st = jax.random.split(rng, 4)
    qpos = self._init_q
    box_dx = jax.random.uniform(k_box, (2,), minval=-0.06, maxval=0.06)
    qpos = qpos.at[9:11].add(box_dx)
    qpos = qpos.at[:7].add(jax.random.uniform(k_arm, (7,), minval=-0.08, maxval=0.08))
    dx = self._dx_template.replace(qpos=qpos, ctrl=self._init_ctrl)
    dx = mjx.kinematics(mx_l, dx)
    dx = mjx.com_pos(mx_l, dx)
    dx = mjx.camlight(mx_l, dx)
    target_pos = jnp.array([0.65, 0.0, 0.14], dtype=jnp.float32) + jax.random.uniform(
        k_tgt,
        (3,),
        minval=jnp.array([-0.03, -0.03, -0.02]),
        maxval=jnp.array([0.03, 0.03, 0.04]),
    )
    grip_pos = dx.site_xpos[self._gripper_site]
    box_pos = dx.xpos[self._box_body]
    d_grip_box = jnp.linalg.norm(grip_pos - box_pos)
    step_c = (
        jax.random.randint(k_st, (), 0, max(1, self.episode_length), dtype=jnp.int32)
        if stagger_steps
        else jnp.array(0, dtype=jnp.int32)
    )
    zero_img = jnp.zeros((self.res, self.res, 3), dtype=jnp.bfloat16)
    return EnvState(
        data=dx,
        obs={"rgb": zero_img, "prev_rgb": zero_img},
        reward=jnp.array(0.0, dtype=jnp.float32),
        done=jnp.array(0.0, dtype=jnp.float32),
        step_count=step_c,
        info={
            "target_pos": target_pos,
            "success": jnp.array(0.0, dtype=jnp.float32),
            "d_grip_box": d_grip_box,
            "trunc": jnp.array(0.0, dtype=jnp.float32),
        },
    )

  def step_physics(self, mx_l: Any, state: EnvState, action: jax.Array) -> EnvState:
    a = jnp.clip(action, -1.0, 1.0)
    delta_ctrl = a * jnp.array([0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.025], dtype=jnp.float32)
    new_ctrl = jnp.clip(state.data.ctrl + delta_ctrl, self._ctrl_lo, self._ctrl_hi)

    if self.action_repeat == 1:
      dx = mjx.step(mx_l, state.data.replace(ctrl=new_ctrl))
    else:
      def _sub(_, d):
        return mjx.step(mx_l, d.replace(ctrl=new_ctrl))
      dx = jax.lax.fori_loop(0, self.action_repeat, _sub, state.data)

    grip_pos = dx.site_xpos[self._gripper_site]
    box_pos = dx.xpos[self._box_body]
    target_pos = state.info["target_pos"]

    d_grip_box = jnp.linalg.norm(grip_pos - box_pos)
    d_box_tgt = jnp.linalg.norm(box_pos - target_pos)
    r_reach = 1.0 - jnp.tanh(5.0 * d_grip_box)
    r_lift = jnp.clip((box_pos[2] - 0.03) * 10.0, 0.0, 1.0)
    r_tgt = 1.0 - jnp.tanh(4.0 * d_box_tgt)
    success = (d_grip_box < 0.05) | (d_box_tgt < 0.06)
    reward = (
        2.0 * r_reach
        + 1.5 * r_lift
        + 3.0 * r_tgt
        + 0.5 * success.astype(jnp.float32)
        - 0.005 * jnp.sum(a ** 2)
    )
    step_c = state.step_count + 1
    done = (step_c >= self.episode_length).astype(jnp.float32)
    return EnvState(
        data=dx,
        obs=state.obs,
        reward=reward,
        done=done,
        step_count=step_c,
        info={
            **state.info,
            "success": success.astype(jnp.float32),
            "d_grip_box": d_grip_box,
            "trunc": done,
        },
    )

  def render_local(self, data_batch: Any) -> jax.Array:
    return tpu_rtx.render(data_batch, self.scene_local)["rgb"]


class RTXHumanoidEnv:
  """Humanoid-21DOF (27-DOF) Vision RL environment powered by `tpu_rtx`."""

  def __init__(
      self,
      res: int = 64,
      episode_length: int = 100,
      action_repeat: int = 1,
      iterations: int = 2,
      ls_iterations: int = 4,
      n_chips: int = 4,
      shader: str = "fast_fused",
  ):
    self.mj_model = mujoco.MjModel.from_xml_string(HUMANOID_PPO_XML)
    self.mj_model.opt.iterations = int(iterations)
    self.mj_model.opt.ls_iterations = int(ls_iterations)
    self.mjx_model = mjx.put_model(self.mj_model, impl="jax")
    self.res = res
    self.episode_length = episode_length
    self.action_repeat = action_repeat
    self.action_size = int(self.mj_model.nu)
    self._torso_id = self.mj_model.body("torso").id
    self._dx_template = mjx.make_data(self.mj_model, impl="jax")
    d0 = mujoco.MjData(self.mj_model)
    mujoco.mj_resetData(self.mj_model, d0)
    self._init_q = jnp.asarray(d0.qpos, dtype=jnp.float32)
    self.scene = tpu_rtx.compile(
        self.mj_model,
        shader=shader,
        res=(res, res),
        sharding=n_chips,
        dtype="bfloat16",
        outputs=("rgb",),
        block_b=16,
        block_r=512,
    )
    self.scene_local = dataclasses.replace(self.scene, mesh=None, n_chips=1)

  def reset_physics(self, mx_l: Any, rng: jax.Array, stagger_steps: bool = False) -> EnvState:
    k_q, k_v, k_st = jax.random.split(rng, 3)
    qpos = self._init_q
    qpos = qpos.at[7:].add(jax.random.uniform(k_q, (self.mj_model.nq - 7,), minval=-0.10, maxval=0.10))
    qvel = jax.random.uniform(k_v, (self.mj_model.nv,), minval=-0.10, maxval=0.10)
    dx = self._dx_template.replace(qpos=qpos, qvel=qvel, ctrl=jnp.zeros(self.action_size, dtype=jnp.float32))
    dx = mjx.kinematics(mx_l, dx)
    dx = mjx.com_pos(mx_l, dx)
    dx = mjx.camlight(mx_l, dx)
    step_c = (
        jax.random.randint(k_st, (), 0, max(1, self.episode_length), dtype=jnp.int32)
        if stagger_steps
        else jnp.array(0, dtype=jnp.int32)
    )
    torso_z = dx.xpos[self._torso_id, 2]
    zero_img = jnp.zeros((self.res, self.res, 3), dtype=jnp.bfloat16)
    return EnvState(
        data=dx,
        obs={"rgb": zero_img, "prev_rgb": zero_img},
        reward=jnp.array(0.0, dtype=jnp.float32),
        done=jnp.array(0.0, dtype=jnp.float32),
        step_count=step_c,
        info={
            "torso_z": torso_z,
            "alive": jnp.array(1.0, dtype=jnp.float32),
            "trunc": jnp.array(0.0, dtype=jnp.float32),
        },
    )

  def step_physics(self, mx_l: Any, state: EnvState, action: jax.Array) -> EnvState:
    u = jnp.clip(action, -1.0, 1.0) * 0.65
    if self.action_repeat == 1:
      dx = mjx.step(mx_l, state.data.replace(ctrl=u))
    else:
      def _sub(_, d):
        return mjx.step(mx_l, d.replace(ctrl=u))
      dx = jax.lax.fori_loop(0, self.action_repeat, _sub, state.data)

    torso_z = dx.xpos[self._torso_id, 2]
    upright_zz = dx.xmat[self._torso_id, 2, 2]
    healthy = (torso_z > 0.78) & (torso_z < 1.85)
    alive_f = healthy.astype(jnp.float32)

    height_term = jnp.exp(-4.0 * (torso_z - 1.24) ** 2)
    fwd_vel = jnp.clip(dx.qvel[0], -2.0, 3.0)
    ctrl_cost = 0.02 * jnp.sum(u ** 2)
    reward = alive_f * (1.0 + 0.8 * jnp.maximum(upright_zz, 0.0) + 0.6 * height_term + 0.4 * fwd_vel) - ctrl_cost

    step_c = state.step_count + 1
    timeout = step_c >= self.episode_length
    fell = ~healthy
    done = (fell | timeout).astype(jnp.float32)
    trunc = (timeout & (~fell)).astype(jnp.float32)
    return EnvState(
        data=dx,
        obs=state.obs,
        reward=reward,
        done=done,
        step_count=step_c,
        info={
            "torso_z": torso_z,
            "alive": alive_f,
            "trunc": trunc,
        },
    )

  def render_local(self, data_batch: Any) -> jax.Array:
    return tpu_rtx.render(data_batch, self.scene_local)["rgb"]


def _make_env(env_name: str, res: int, n_chips: int, action_repeat: int):
  if env_name == "cartpole":
    return RTXCartpoleEnv(res=res, episode_length=100, action_repeat=action_repeat, n_chips=n_chips), 100.0
  if env_name == "panda_pick_cube":
    return RTXPandaPickCubeEnv(res=res, episode_length=80, action_repeat=action_repeat, iterations=2, ls_iterations=4, n_chips=n_chips), 400.0
  if env_name == "humanoid_21dof":
    return RTXHumanoidEnv(res=res, episode_length=100, action_repeat=action_repeat, iterations=2, ls_iterations=4, n_chips=n_chips), 200.0
  raise ValueError(f"Unknown env: {env_name}")


def _normalize_obs_2f(rgb: jax.Array, prev_rgb: jax.Array, norm_st: NormState) -> jax.Array:
  std = jnp.maximum(jnp.sqrt(norm_st.obs_var), 0.05)
  c_n = jnp.clip((rgb.astype(jnp.float32) - norm_st.obs_mean) / std, -5.0, 5.0)
  p_n = jnp.clip((prev_rgb.astype(jnp.float32) - norm_st.obs_mean) / std, -5.0, 5.0)
  return jnp.concatenate([c_n, p_n], axis=-1)


def _gaussian_logp(act: jax.Array, mean: jax.Array, log_std: jax.Array) -> jax.Array:
  std = jnp.exp(log_std)
  return -0.5 * jnp.sum(((act - mean) / std) ** 2 + 2.0 * log_std + jnp.log(2.0 * jnp.pi), axis=-1)


def run_probe() -> Dict[str, Any]:
  """Quick diagnostic probe for kinematics reset parity, trackcom, and tpu_rtx inside shard_map."""
  out: Dict[str, Any] = {}
  for ename in ("cartpole", "panda_pick_cube", "humanoid_21dof"):
    env, _ = _make_env(ename, res=64, n_chips=4, action_repeat=1)
    rng = jax.random.PRNGKey(0)
    st_k = env.reset_physics(env.mjx_model, rng, stagger_steps=False)
    dx_f = mjx.forward(env.mjx_model, st_k.data.replace(qvel=st_k.data.qvel))
    geom_diff = float(jnp.max(jnp.abs(st_k.data.geom_xpos - dx_f.geom_xpos)))
    cam_diff = float(jnp.max(jnp.abs(st_k.data.cam_xpos - dx_f.cam_xpos)))
    # Also check one step from kinematics vs forward
    act = jnp.zeros((env.action_size,), dtype=jnp.float32)
    step_k = env.step_physics(env.mjx_model, st_k, act)
    step_f = env.step_physics(env.mjx_model, st_k._replace(data=dx_f), act)
    step_q_diff = float(jnp.max(jnp.abs(step_k.data.qpos - step_f.data.qpos)))

    # Lower tpu_rtx.render standalone HLO
    d_b = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x[None, ...], (16, *x.shape)).copy(), st_k.data)
    d_b_s = tpu_rtx.shard_data(d_b, env.scene)
    hlo_r = tpu_rtx.audit_hlo(tpu_rtx.lower_render(d_b_s, env.scene))
    out[ename] = {
        "geom_xpos_kin_vs_fwd_max": geom_diff,
        "cam_xpos_kin_vs_fwd_max": cam_diff,
        "step_qpos_kin_vs_fwd_max": step_q_diff,
        "render_hlo": hlo_r,
    }
    print(f"PROBE [{ename}]:", json.dumps(out[ename]), flush=True)
  return out


def run_vision_ppo_rtx(
    env_name: str,
    num_envs: int = 2048,
    rollout_len: int = 20,
    num_iters: int = 100,
    num_epochs: int = 2,
    num_minibatches: int = 4,
    res: int = 64,
    action_repeat: int = 1,
    n_chips: int = 4,
    lr_peak: float = 4e-4,
    lr_end: float = 4e-5,
    vf_coef: float = 0.25,
    ent_coef: float = 0.002,
    clip_eps: float = 0.2,
    v_clip_eps: float = 0.2,
    max_grad_norm: float = 0.5,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    seed: int = 0,
    split_rollout_update: Optional[bool] = None,
) -> Dict[str, Any]:
  devs = jax.devices()[:n_chips]
  assert len(devs) == n_chips, f"Need {n_chips} chips, found {len(devs)}"
  assert num_envs % n_chips == 0
  local_envs = num_envs // n_chips
  assert (rollout_len * local_envs) % num_minibatches == 0
  mb_size = (rollout_len * local_envs) // num_minibatches

  env, init_ret_var = _make_env(env_name, res=res, n_chips=n_chips, action_repeat=action_repeat)
  mesh = jax.sharding.Mesh(np.array(devs), ("dp",))
  P = jax.sharding.PartitionSpec
  sharding = jax.sharding.NamedSharding(mesh, P("dp"))
  replicate = jax.sharding.NamedSharding(mesh, P())
  sm_kw = {"check_rep": False} if "check_rep" in inspect.signature(shard_map).parameters else {"check_vma": False}

  model = VisionActorCritic(action_dim=env.action_size)
  rng = jax.random.PRNGKey(seed)
  rng, init_k, env_k = jax.random.split(rng, 3)

  def _init_envs_local(keys_local, mx_l):
    st_p = jax.vmap(lambda k: env.reset_physics(mx_l, k, stagger_steps=True))(keys_local)
    rgb0 = env.render_local(st_p.data)
    return st_p._replace(obs={"rgb": rgb0, "prev_rgb": rgb0})

  _init_sm = shard_map(_init_envs_local, mesh=mesh, in_specs=(P("dp"), P()), out_specs=P("dp"), **sm_kw)
  mx_rep = jax.device_put(env.mjx_model, replicate)
  env_keys = jax.device_put(jax.random.split(env_k, num_envs), sharding)
  state0 = jax.jit(_init_sm)(env_keys, mx_rep)

  # Standalone tpu_rtx HLO audit
  hlo_render = tpu_rtx.audit_hlo(tpu_rtx.lower_render(state0.data, env.scene))
  assert hlo_render["zero_copy_verified"], f"Render HLO failed zero-copy: {hlo_render}"

  obs0_np = np.asarray(jax.device_get(state0.obs["rgb"]), dtype=np.float32)
  init_obs_mean = jnp.asarray(np.mean(obs0_np, axis=(0, 1, 2), keepdims=True), dtype=jnp.float32)
  init_obs_var = jnp.asarray(np.maximum(np.var(obs0_np, axis=(0, 1, 2), keepdims=True), 1e-2), dtype=jnp.float32)

  dummy_obs = jnp.zeros((1, res, res, 6), dtype=jnp.float32)
  params = jax.device_put(model.init(init_k, dummy_obs)["params"], replicate)

  total_updates = num_iters * num_epochs * num_minibatches
  warmup_updates = max(4, int(0.05 * total_updates))
  lr_schedule = optax.warmup_cosine_decay_schedule(
      init_value=lr_end,
      peak_value=lr_peak,
      warmup_steps=warmup_updates,
      decay_steps=max(warmup_updates + 1, total_updates),
      end_value=lr_end,
  )
  tx = optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.adam(learning_rate=lr_schedule, eps=1e-5))
  opt_state = jax.device_put(tx.init(params), replicate)

  norm_state = jax.device_put(
      NormState(
          obs_mean=init_obs_mean,
          obs_var=init_obs_var,
          ret_var=jnp.array(init_ret_var, dtype=jnp.float32),
      ),
      replicate,
  )
  ret_acc0 = jax.device_put(jnp.zeros((num_envs,), dtype=jnp.float32), sharding)

  def _dp_mean(x):
    return jax.lax.pmean(x, "dp") if n_chips > 1 else x

  if split_rollout_update is None:
    split_rollout_update = False

  def _one_step_local(p, n_st, st, acc, step_rng_in, mx_l):
    step_rng = step_rng_in[0] if step_rng_in.ndim >= 2 or (step_rng_in.ndim == 1 and step_rng_in.shape[0] == 1) else step_rng_in
    rgb_cur = st.obs["rgb"]
    rgb_prv = st.obs["prev_rgb"]
    obs_n = _normalize_obs_2f(rgb_cur, rgb_prv, n_st)
    mean, log_std, val = model.apply({"params": p}, obs_n)
    std = jnp.exp(log_std)
    k_act, k_rst = jax.random.split(step_rng)
    act = mean + std * jax.random.normal(k_act, mean.shape)
    logp = _gaussian_logp(act, mean, log_std)

    next_st = jax.vmap(lambda s, a: env.step_physics(mx_l, s, a))(st, act)
    r_keys = jax.random.split(k_rst, local_envs)
    fresh = jax.vmap(lambda k: env.reset_physics(mx_l, k, stagger_steps=False))(r_keys)
    d_flag = next_st.done > 0.5

    def _where_leaf(f_leaf, c_leaf):
      mask = d_flag.reshape((local_envs,) + (1,) * (c_leaf.ndim - 1))
      return jnp.where(mask, f_leaf, c_leaf)

    reset_st = jax.tree_util.tree_map(_where_leaf, fresh, next_st)
    new_acc = acc * gamma * (1.0 - next_st.done) + next_st.reward
    if env_name == "panda_pick_cube":
      aux_m = (next_st.info["success"], next_st.info["d_grip_box"])
    elif env_name == "humanoid_21dof":
      aux_m = (next_st.info["alive"], next_st.info["torso_z"])
    else:
      aux_m = (next_st.info["balanced"], next_st.info["pole_cos"])

    step_rec = (
        rgb_cur,
        rgb_prv,
        act,
        logp,
        next_st.reward,
        next_st.done,
        next_st.info["trunc"],
        val,
        new_acc,
        aux_m,
    )
    return reset_st, new_acc, d_flag, rgb_cur, step_rec

  def _render_step_local(reset_st, d_flag, rgb_cur):
    new_rgb = env.render_local(reset_st.data)
    new_prev = jnp.where(d_flag[:, None, None, None], new_rgb, rgb_cur)
    return reset_st._replace(obs={"rgb": new_rgb, "prev_rgb": new_prev})

  def _gae_and_update_local(p, opt_st, n_st, env_st_out, traj_list_or_tuple, chip_rng_in):
    chip_rng = chip_rng_in[0] if chip_rng_in.ndim >= 2 or (chip_rng_in.ndim == 1 and chip_rng_in.shape[0] == 1) else chip_rng_in
    _, epoch_rng = jax.random.split(chip_rng)
    rgb_t, prev_t, act_t, logp_t, rew_t, done_t, trunc_t, val_t, acc_t, aux_t = traj_list_or_tuple

    rgb_f32 = rgb_t.astype(jnp.float32)
    b_obs_mean = _dp_mean(jnp.mean(rgb_f32, axis=(0, 1, 2, 3), keepdims=True)[0])
    b_obs_var = _dp_mean(jnp.mean((rgb_f32 - b_obs_mean[None, ...]) ** 2, axis=(0, 1, 2, 3), keepdims=True)[0])
    b_ret_mean = _dp_mean(jnp.mean(acc_t))
    b_ret_var = _dp_mean(jnp.mean((acc_t - b_ret_mean) ** 2))
    mom_obs = 0.02
    mom_ret = 0.05
    new_n_st = NormState(
        obs_mean=(1.0 - mom_obs) * n_st.obs_mean + mom_obs * b_obs_mean,
        obs_var=(1.0 - mom_obs) * n_st.obs_var + mom_obs * b_obs_var,
        ret_var=(1.0 - mom_ret) * n_st.ret_var + mom_ret * jnp.maximum(b_ret_var, 1.0),
    )

    last_obs_n = _normalize_obs_2f(env_st_out.obs["rgb"], env_st_out.obs["prev_rgb"], new_n_st)
    _, _, last_val = model.apply({"params": p}, last_obs_n)

    ret_scale = jnp.maximum(jnp.sqrt(new_n_st.ret_var), 1.0)
    rew_scaled = rew_t / ret_scale + gamma * trunc_t * jax.lax.stop_gradient(val_t)

    def _gae(adv_next, xs):
      r, d, v, v_next = xs
      nonterm = 1.0 - d
      delta = r + gamma * v_next * nonterm - v
      adv = delta + gamma * gae_lambda * nonterm * adv_next
      return adv, adv

    v_next_all = jnp.concatenate([val_t[1:], last_val[None, :]], axis=0)
    _, adv_t = jax.lax.scan(
        _gae,
        jnp.zeros((local_envs,), dtype=jnp.float32),
        (rew_scaled, done_t, val_t, v_next_all),
        reverse=True,
    )
    ret_t = adv_t + val_t

    adv_mean = _dp_mean(jnp.mean(adv_t))
    adv_var = _dp_mean(jnp.mean((adv_t - adv_mean) ** 2))
    adv_norm = (adv_t - adv_mean) / (jnp.sqrt(adv_var) + 1e-6)

    n_local = rollout_len * local_envs
    rgb_f = rgb_t.reshape((n_local, res, res, 3))
    prev_f = prev_t.reshape((n_local, res, res, 3))
    act_f = act_t.reshape((n_local, env.action_size))
    old_lp_f = logp_t.reshape((n_local,))
    old_val_f = val_t.reshape((n_local,))
    adv_f = adv_norm.reshape((n_local,))
    ret_f = ret_t.reshape((n_local,))

    roll_metrics = {
        "mean_reward": _dp_mean(jnp.mean(rew_t)),
        "done_frac": _dp_mean(jnp.mean(done_t)),
        "ret_scale": ret_scale,
    }
    if env_name == "panda_pick_cube":
      succ_t, dgb_t = aux_t
      roll_metrics["success_rate"] = _dp_mean(jnp.mean(succ_t))
      roll_metrics["d_grip_box"] = _dp_mean(jnp.mean(dgb_t))
    elif env_name == "humanoid_21dof":
      alv_t, tz_t = aux_t
      roll_metrics["success_rate"] = _dp_mean(jnp.mean(alv_t))
      roll_metrics["torso_z"] = _dp_mean(jnp.mean(tz_t))
    else:
      bal_t, pcos_t = aux_t
      roll_metrics["success_rate"] = _dp_mean(jnp.mean(bal_t))
      roll_metrics["pole_cos"] = _dp_mean(jnp.mean(pcos_t))

    batch = (rgb_f, prev_f, act_f, old_lp_f, old_val_f, adv_f, ret_f)
    p_new, opt_new, up_metrics = _ppo_update_local(p, opt_st, new_n_st, batch, epoch_rng)
    return p_new, opt_new, new_n_st, {**roll_metrics, **up_metrics}

  def _rollout_and_gae_local(p, n_st, env_st, r_acc, chip_rng_in, mx_l):
    chip_rng = chip_rng_in[0] if chip_rng_in.ndim >= 2 or (chip_rng_in.ndim == 1 and chip_rng_in.shape[0] == 1) else chip_rng_in
    roll_rng, epoch_rng = jax.random.split(chip_rng)

    def _rollout_step(carry, step_rng):
      st, acc = carry
      rgb_cur = st.obs["rgb"]
      rgb_prv = st.obs["prev_rgb"]
      obs_n = _normalize_obs_2f(rgb_cur, rgb_prv, n_st)
      mean, log_std, val = model.apply({"params": p}, obs_n)
      std = jnp.exp(log_std)
      k_act, k_rst = jax.random.split(step_rng)
      act = mean + std * jax.random.normal(k_act, mean.shape)
      logp = _gaussian_logp(act, mean, log_std)

      next_st = jax.vmap(lambda s, a: env.step_physics(mx_l, s, a))(st, act)
      r_keys = jax.random.split(k_rst, local_envs)
      fresh = jax.vmap(lambda k: env.reset_physics(mx_l, k, stagger_steps=False))(r_keys)
      d_flag = next_st.done > 0.5

      def _where_leaf(f_leaf, c_leaf):
        mask = d_flag.reshape((local_envs,) + (1,) * (c_leaf.ndim - 1))
        return jnp.where(mask, f_leaf, c_leaf)

      reset_st = jax.tree_util.tree_map(_where_leaf, fresh, next_st)
      new_rgb = env.render_local(reset_st.data)
      new_prev = jnp.where(d_flag[:, None, None, None], new_rgb, rgb_cur)
      reset_st = reset_st._replace(obs={"rgb": new_rgb, "prev_rgb": new_prev})
      new_acc = acc * gamma * (1.0 - next_st.done) + next_st.reward

      if env_name == "panda_pick_cube":
        aux_m = (next_st.info["success"], next_st.info["d_grip_box"])
      elif env_name == "humanoid_21dof":
        aux_m = (next_st.info["alive"], next_st.info["torso_z"])
      else:
        aux_m = (next_st.info["balanced"], next_st.info["pole_cos"])

      return (reset_st, new_acc), (
          rgb_cur,
          rgb_prv,
          act,
          logp,
          next_st.reward,
          next_st.done,
          next_st.info["trunc"],
          val,
          new_acc,
          aux_m,
      )

    step_keys = jax.random.split(roll_rng, rollout_len)
    (env_st_out, r_acc_out), traj = jax.lax.scan(_rollout_step, (env_st, r_acc), step_keys)
    rgb_t, prev_t, act_t, logp_t, rew_t, done_t, trunc_t, val_t, acc_t, aux_t = traj

    rgb_f32 = rgb_t.astype(jnp.float32)
    b_obs_mean = _dp_mean(jnp.mean(rgb_f32, axis=(0, 1, 2, 3), keepdims=True)[0])
    b_obs_var = _dp_mean(jnp.mean((rgb_f32 - b_obs_mean[None, ...]) ** 2, axis=(0, 1, 2, 3), keepdims=True)[0])
    b_ret_mean = _dp_mean(jnp.mean(acc_t))
    b_ret_var = _dp_mean(jnp.mean((acc_t - b_ret_mean) ** 2))
    mom_obs = 0.02
    mom_ret = 0.05
    new_n_st = NormState(
        obs_mean=(1.0 - mom_obs) * n_st.obs_mean + mom_obs * b_obs_mean,
        obs_var=(1.0 - mom_obs) * n_st.obs_var + mom_obs * b_obs_var,
        ret_var=(1.0 - mom_ret) * n_st.ret_var + mom_ret * jnp.maximum(b_ret_var, 1.0),
    )

    last_obs_n = _normalize_obs_2f(env_st_out.obs["rgb"], env_st_out.obs["prev_rgb"], new_n_st)
    _, _, last_val = model.apply({"params": p}, last_obs_n)

    ret_scale = jnp.maximum(jnp.sqrt(new_n_st.ret_var), 1.0)
    rew_scaled = rew_t / ret_scale + gamma * trunc_t * jax.lax.stop_gradient(val_t)

    def _gae(adv_next, xs):
      r, d, v, v_next = xs
      nonterm = 1.0 - d
      delta = r + gamma * v_next * nonterm - v
      adv = delta + gamma * gae_lambda * nonterm * adv_next
      return adv, adv

    v_next_all = jnp.concatenate([val_t[1:], last_val[None, :]], axis=0)
    _, adv_t = jax.lax.scan(
        _gae,
        jnp.zeros((local_envs,), dtype=jnp.float32),
        (rew_scaled, done_t, val_t, v_next_all),
        reverse=True,
    )
    ret_t = adv_t + val_t

    adv_mean = _dp_mean(jnp.mean(adv_t))
    adv_var = _dp_mean(jnp.mean((adv_t - adv_mean) ** 2))
    adv_norm = (adv_t - adv_mean) / (jnp.sqrt(adv_var) + 1e-6)

    n_local = rollout_len * local_envs
    rgb_f = rgb_t.reshape((n_local, res, res, 3))
    prev_f = prev_t.reshape((n_local, res, res, 3))
    act_f = act_t.reshape((n_local, env.action_size))
    old_lp_f = logp_t.reshape((n_local,))
    old_val_f = val_t.reshape((n_local,))
    adv_f = adv_norm.reshape((n_local,))
    ret_f = ret_t.reshape((n_local,))

    roll_metrics = {
        "mean_reward": _dp_mean(jnp.mean(rew_t)),
        "done_frac": _dp_mean(jnp.mean(done_t)),
        "ret_scale": ret_scale,
    }
    if env_name == "panda_pick_cube":
      succ_t, dgb_t = aux_t
      roll_metrics["success_rate"] = _dp_mean(jnp.mean(succ_t))
      roll_metrics["d_grip_box"] = _dp_mean(jnp.mean(dgb_t))
    elif env_name == "humanoid_21dof":
      alv_t, tz_t = aux_t
      roll_metrics["success_rate"] = _dp_mean(jnp.mean(alv_t))
      roll_metrics["torso_z"] = _dp_mean(jnp.mean(tz_t))
    else:
      bal_t, pcos_t = aux_t
      roll_metrics["success_rate"] = _dp_mean(jnp.mean(bal_t))
      roll_metrics["pole_cos"] = _dp_mean(jnp.mean(pcos_t))

    batch = (rgb_f, prev_f, act_f, old_lp_f, old_val_f, adv_f, ret_f)
    return new_n_st, env_st_out, r_acc_out, batch, roll_metrics, epoch_rng

  def _ppo_update_local(p, opt_st, new_n_st, batch, epoch_rng):
    rgb_f, prev_f, act_f, old_lp_f, old_val_f, adv_f, ret_f = batch
    n_local = rollout_len * local_envs

    def _ppo_loss(pr, mb):
      mb_rgb, mb_prv, mb_act, mb_old_lp, mb_old_val, mb_adv, mb_ret = mb
      mb_obs = _normalize_obs_2f(mb_rgb, mb_prv, new_n_st)
      mean, log_std, val = model.apply({"params": pr}, mb_obs)
      new_lp = _gaussian_logp(mb_act, mean, log_std)
      log_ratio = new_lp - mb_old_lp
      ratio = jnp.exp(log_ratio)
      surr1 = ratio * mb_adv
      surr2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_adv
      pi_loss = -jnp.mean(jnp.minimum(surr1, surr2))

      v_clipped = mb_old_val + jnp.clip(val - mb_old_val, -v_clip_eps, v_clip_eps)
      v_loss1 = 0.5 * (val - mb_ret) ** 2
      v_loss2 = 0.5 * (v_clipped - mb_ret) ** 2
      v_loss = jnp.mean(jnp.maximum(v_loss1, v_loss2))

      ent = jnp.mean(jnp.sum(log_std + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e), axis=-1))
      approx_kl = jnp.mean((ratio - 1.0) - log_ratio)
      total_loss = pi_loss + vf_coef * v_loss - ent_coef * ent
      return total_loss, (pi_loss, v_loss, ent, approx_kl)

    def _mb_step(carry, mb):
      pr, ost = carry
      (loss, aux), grads = jax.value_and_grad(_ppo_loss, has_aux=True)(pr, mb)
      grads = _dp_mean(grads)
      loss = _dp_mean(loss)
      aux = _dp_mean(aux)
      ups, ost = tx.update(grads, ost, pr)
      pr = optax.apply_updates(pr, ups)
      return (pr, ost), (loss, *aux)

    def _epoch_step(carry, ep_rng):
      pr, ost = carry
      perm = jax.random.permutation(ep_rng, n_local)
      mb_data = (
          rgb_f[perm].reshape((num_minibatches, mb_size, res, res, 3)),
          prev_f[perm].reshape((num_minibatches, mb_size, res, res, 3)),
          act_f[perm].reshape((num_minibatches, mb_size, env.action_size)),
          old_lp_f[perm].reshape((num_minibatches, mb_size)),
          old_val_f[perm].reshape((num_minibatches, mb_size)),
          adv_f[perm].reshape((num_minibatches, mb_size)),
          ret_f[perm].reshape((num_minibatches, mb_size)),
      )
      (pr_out, ost_out), mb_stats = jax.lax.scan(_mb_step, (pr, ost), mb_data)
      return (pr_out, ost_out), jax.tree_util.tree_map(jnp.mean, mb_stats)

    ep_keys = jax.random.split(epoch_rng, num_epochs)
    (p_new, opt_new), ep_stats = jax.lax.scan(_epoch_step, (p, opt_st), ep_keys)
    mean_loss, mean_pi_loss, mean_v_loss, mean_ent, mean_kl = jax.tree_util.tree_map(jnp.mean, ep_stats)
    up_metrics = {
        "mean_loss": mean_loss,
        "pi_loss": mean_pi_loss,
        "v_loss": mean_v_loss,
        "entropy": mean_ent,
        "approx_kl": mean_kl,
    }
    return p_new, opt_new, up_metrics

  def _ppo_iter_local(p, opt_st, n_st, env_st, r_acc, chip_rng_in, mx_l):
    new_n_st, env_st_out, r_acc_out, batch, roll_metrics, epoch_rng = _rollout_and_gae_local(
        p, n_st, env_st, r_acc, chip_rng_in, mx_l
    )
    p_new, opt_new, up_metrics = _ppo_update_local(p, opt_st, new_n_st, batch, epoch_rng)
    metrics = {**roll_metrics, **up_metrics}
    return p_new, opt_new, new_n_st, env_st_out, r_acc_out, metrics

  if split_rollout_update:
    _step_sm = shard_map(
        _one_step_local,
        mesh=mesh,
        in_specs=(P(), P(), P("dp"), P("dp"), P("dp"), P()),
        out_specs=(P("dp"), P("dp"), P("dp"), P("dp"), P("dp")),
        **sm_kw,
    )
    _rend_sm = shard_map(
        _render_step_local,
        mesh=mesh,
        in_specs=(P("dp"), P("dp"), P("dp")),
        out_specs=P("dp"),
        **sm_kw,
    )
    _gae_upd_sm = shard_map(
        _gae_and_update_local,
        mesh=mesh,
        in_specs=(P(), P(), P(), P("dp"), P(None, "dp"), P("dp")),
        out_specs=(P(), P(), P(), P()),
        **sm_kw,
    )
    step_jit = jax.jit(_step_sm)
    rend_jit = jax.jit(_rend_sm)
    gae_upd_jit = jax.jit(_gae_upd_sm)
    stack_jit = jax.jit(lambda recs: jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *recs))
  else:
    _ppo_sm = shard_map(
        _ppo_iter_local,
        mesh=mesh,
        in_specs=(P(), P(), P(), P("dp"), P("dp"), P("dp"), P()),
        out_specs=(P(), P(), P(), P("dp"), P("dp"), P()),
        **sm_kw,
    )

    @jax.jit
    def ppo_iter(p, ost, nst, est, racc, crng):
      return _ppo_sm(p, ost, nst, est, racc, crng, mx_rep)

  # Deterministic policy evaluation helper (fresh non-staggered episodes, mean action)
  def _eval_local(p, n_st, keys_local, mx_l):
    st0 = jax.vmap(lambda k: env.reset_physics(mx_l, k, stagger_steps=False))(keys_local)
    rgb0 = env.render_local(st0.data)
    st0 = st0._replace(obs={"rgb": rgb0, "prev_rgb": rgb0})

    def _eval_step(carry, _):
      st, ep_ret, ep_alive = carry
      obs_n = _normalize_obs_2f(st.obs["rgb"], st.obs["prev_rgb"], n_st)
      mean, _, _ = model.apply({"params": p}, obs_n)
      nxt = jax.vmap(lambda s, a: env.step_physics(mx_l, s, a))(st, mean)
      new_ret = ep_ret + nxt.reward * ep_alive
      new_alive = ep_alive * (1.0 - nxt.done)
      rgb_nxt = env.render_local(nxt.data)
      nxt = nxt._replace(obs={"rgb": rgb_nxt, "prev_rgb": st.obs["rgb"]})
      if env_name == "panda_pick_cube":
        aux = nxt.info["success"]
      elif env_name == "humanoid_21dof":
        aux = nxt.info["alive"]
      else:
        aux = nxt.info["balanced"]
      return (nxt, new_ret, new_alive), (nxt.reward, aux)

    eval_len = 60 if env_name == "panda_pick_cube" else 80
    (_, final_ret, _), (rew_seq, succ_seq) = jax.lax.scan(
        _eval_step,
        (st0, jnp.zeros((local_envs,), dtype=jnp.float32), jnp.ones((local_envs,), dtype=jnp.float32)),
        None,
        length=eval_len,
    )
    return (
        _dp_mean(jnp.mean(final_ret)),
        _dp_mean(jnp.mean(rew_seq)),
        _dp_mean(jnp.mean(succ_seq[-1])),
    )

  eval_fn = jax.jit(shard_map(
      _eval_local,
      mesh=mesh,
      in_specs=(P(), P(), P("dp"), P()),
      out_specs=(P(), P(), P()),
      **sm_kw,
  ))

  rng, k0, k_ev = jax.random.split(rng, 3)
  chip_rngs0 = jax.device_put(jax.random.split(k0, n_chips), sharding)
  if split_rollout_update:
    hlo_step = tpu_rtx.audit_hlo(step_jit.lower(params, norm_state, state0, ret_acc0, chip_rngs0, mx_rep))
    step_out_s = jax.eval_shape(
        lambda p, ns, st, ra, cr, mx: step_jit(p, ns, st, ra, cr, mx),
        params, norm_state, state0, ret_acc0, chip_rngs0, mx_rep,
    )
    hlo_rend = tpu_rtx.audit_hlo(rend_jit.lower(step_out_s[0], step_out_s[2], step_out_s[3]))
    sharding_traj = jax.sharding.NamedSharding(mesh, P(None, "dp"))
    traj_s = jax.tree_util.tree_map(
        lambda x: jax.ShapeDtypeStruct((rollout_len, *x.shape), x.dtype, sharding=sharding_traj),
        step_out_s[4],
    )
    hlo_upd = tpu_rtx.audit_hlo(gae_upd_jit.lower(params, opt_state, norm_state, state0, traj_s, chip_rngs0))
    hlo_ppo = {
        "host_callbacks": hlo_step["host_callbacks"] + hlo_rend["host_callbacks"] + hlo_upd["host_callbacks"],
        "send_recv_ops": hlo_step["send_recv_ops"] + hlo_rend["send_recv_ops"] + hlo_upd["send_recv_ops"],
        "gather_ops": hlo_step["gather_ops"] + hlo_rend["gather_ops"] + hlo_upd["gather_ops"],
        "zero_copy_verified": bool(hlo_step["zero_copy_verified"] and hlo_rend["zero_copy_verified"] and hlo_upd["zero_copy_verified"]),
        "step_hlo": hlo_step,
        "rend_hlo": hlo_rend,
        "upd_hlo": hlo_upd,
    }
  else:
    hlo_ppo = tpu_rtx.audit_hlo(ppo_iter.lower(params, opt_state, norm_state, state0, ret_acc0, chip_rngs0))
  assert hlo_ppo["host_callbacks"] == 0, f"Host callbacks detected in PPO HLO: {hlo_ppo}"
  assert hlo_ppo["send_recv_ops"] == 0, f"Send/Recv detected in PPO HLO: {hlo_ppo}"
  assert hlo_ppo["gather_ops"] == 0, f"Gather ops detected in PPO HLO: {hlo_ppo}"

  eval_keys = jax.device_put(jax.random.split(k_ev, num_envs), sharding)
  ev0_ret, ev0_rew, ev0_succ = eval_fn(params, norm_state, eval_keys, mx_rep)
  jax.block_until_ready(ev0_ret)
  eval_init = {"ep_return": float(ev0_ret), "step_reward": float(ev0_rew), "final_success": float(ev0_succ)}

  history: List[Dict[str, Any]] = []
  cur_state = state0
  cur_acc = ret_acc0
  t_start = time.time()
  t_compile_end = None
  for it in range(num_iters):
    rng, k = jax.random.split(rng)
    chip_rngs = jax.device_put(jax.random.split(k, n_chips), sharding)
    if split_rollout_update:
      step_recs = []
      for _t in range(rollout_len):
        rng, ks = jax.random.split(rng)
        s_rngs = jax.device_put(jax.random.split(ks, n_chips), sharding)
        rst, cur_acc, d_flag, rgb_cur, s_rec = step_jit(params, norm_state, cur_state, cur_acc, s_rngs, mx_rep)
        cur_state = rend_jit(rst, d_flag, rgb_cur)
        step_recs.append(s_rec)
      traj_tb = stack_jit(step_recs)
      params, opt_state, norm_state, m_out = gae_upd_jit(params, opt_state, norm_state, cur_state, traj_tb, chip_rngs)
    else:
      params, opt_state, norm_state, cur_state, cur_acc, m_out = ppo_iter(
          params, opt_state, norm_state, cur_state, cur_acc, chip_rngs
      )
    jax.block_until_ready(m_out["mean_reward"])
    if it == 0:
      t_compile_end = time.time()
    rec = {"iter": it + 1, **{k_: float(v_) for k_, v_ in m_out.items()}}
    history.append(rec)
    if it == 0 or (it + 1) % 10 == 0 or it == num_iters - 1:
      if "d_grip_box" in rec:
        extra = f"dgb={rec['d_grip_box']:.3f}"
      elif "torso_z" in rec:
        extra = f"tz={rec['torso_z']:.3f}"
      else:
        extra = f"pcos={rec['pole_cos']:.3f}"
      print(
          f"[{env_name}][tpu_rtx][4c B={num_envs} ar={action_repeat}][{it+1}/{num_iters}] "
          f"rew={rec['mean_reward']:.3f} succ={rec['success_rate']:.3f} {extra} "
          f"vloss={rec['v_loss']:.4f} kl={rec['approx_kl']:.4f} ent={rec['entropy']:.2f}",
          flush=True,
      )

  t_total = time.time() - t_start
  t_steady = time.time() - (t_compile_end or t_start)
  steady_steps = max(1, (num_iters - 1) * num_envs * rollout_len)
  compile_s = (t_compile_end - t_start) if t_compile_end else 0.0
  steady_sps = steady_steps / max(t_steady, 1e-6)

  evf_ret, evf_rew, evf_succ = eval_fn(params, norm_state, eval_keys, mx_rep)
  jax.block_until_ready(evf_ret)
  eval_final = {"ep_return": float(evf_ret), "step_reward": float(evf_rew), "final_success": float(evf_succ)}
  print(f"EVAL [{env_name}] B={num_envs}: init={eval_init} -> final={eval_final} | steady={steady_sps:,.0f} env-steps/s", flush=True)

  return {
      "env": env_name,
      "backend": jax.default_backend(),
      "device_kind": devs[0].device_kind,
      "n_chips": n_chips,
      "num_envs": num_envs,
      "rollout_len": rollout_len,
      "num_iters": num_iters,
      "num_epochs": num_epochs,
      "num_minibatches": num_minibatches,
      "action_repeat": action_repeat,
      "res": res,
      "render_hlo": hlo_render,
      "ppo_hlo": hlo_ppo,
      "hlo_zero_copy": bool(hlo_ppo["zero_copy_verified"] and hlo_render["zero_copy_verified"]),
      "compile_s": compile_s,
      "steady_wall_s": t_steady,
      "total_wall_s": t_total,
      "steady_env_steps_per_sec": steady_sps,
      "eval_init": eval_init,
      "eval_final": eval_final,
      "initial_metrics": history[0] if history else {},
      "final_metrics": history[-1] if history else {},
      "best_reward": max(r["mean_reward"] for r in history) if history else 0.0,
      "best_success_rate": max(r["success_rate"] for r in history) if history else 0.0,
      "history": history,
  }


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--step", default="suite", choices=["probe", "single", "suite"])
  ap.add_argument("--labels", nargs="*", default=None)
  ap.add_argument("--env", default="cartpole", choices=["cartpole", "panda_pick_cube", "humanoid_21dof"])
  ap.add_argument("--num_envs", type=int, default=2048)
  ap.add_argument("--rollout_len", type=int, default=20)
  ap.add_argument("--num_iters", type=int, default=100)
  ap.add_argument("--num_epochs", type=int, default=2)
  ap.add_argument("--num_minibatches", type=int, default=4)
  ap.add_argument("--action_repeat", type=int, default=1)
  ap.add_argument("--res", type=int, default=64)
  ap.add_argument("--out_json", default="out/metrics/v3_w0_wave4_rl_sdk.json")
  args = ap.parse_args()

  os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
  if args.step == "probe":
    res = run_probe()
    return

  if args.step == "single":
    run = run_vision_ppo_rtx(
        env_name=args.env,
        num_envs=args.num_envs,
        rollout_len=args.rollout_len,
        num_iters=args.num_iters,
        num_epochs=args.num_epochs,
        num_minibatches=args.num_minibatches,
        action_repeat=args.action_repeat,
        res=args.res,
    )
    payload = {"ppo_runs": [run]}
    with open(args.out_json, "w") as f:
      json.dump(payload, f, indent=2)
    print("Saved ->", args.out_json, flush=True)
    return

  # Full Wave 4 Vision PPO scaling suite
  configs = [
      dict(label="cartpole_rtx_b2048_4c", env_name="cartpole", num_envs=2048, rollout_len=20, num_iters=120, num_epochs=2, num_minibatches=4, action_repeat=2, lr_peak=5e-4, lr_end=5e-5),
      dict(label="cartpole_rtx_b4096_4c", env_name="cartpole", num_envs=4096, rollout_len=20, num_iters=120, num_epochs=2, num_minibatches=4, action_repeat=1, lr_peak=5e-4, lr_end=5e-5),
      dict(label="panda_rtx_b1024_4c", env_name="panda_pick_cube", num_envs=1024, rollout_len=20, num_iters=140, num_epochs=2, num_minibatches=4, action_repeat=1, lr_peak=4e-4, lr_end=4e-5),
      dict(label="panda_rtx_b2048_4c", env_name="panda_pick_cube", num_envs=2048, rollout_len=20, num_iters=120, num_epochs=2, num_minibatches=4, action_repeat=1, lr_peak=4e-4, lr_end=4e-5),
      dict(label="humanoid_rtx_b1024_4c", env_name="humanoid_21dof", num_envs=1024, rollout_len=20, num_iters=140, num_epochs=2, num_minibatches=4, action_repeat=1, lr_peak=3e-4, lr_end=3e-5),
      dict(label="humanoid_rtx_b2048_4c", env_name="humanoid_21dof", num_envs=2048, rollout_len=20, num_iters=120, num_epochs=2, num_minibatches=4, action_repeat=1, lr_peak=3e-4, lr_end=3e-5),
  ]
  if args.labels:
    want = set(args.labels)
    configs = [c for c in configs if c["label"] in want]

  if os.path.exists(args.out_json):
    with open(args.out_json, "r") as f:
      payload = json.load(f)
  else:
    payload = {}
  runs = list(payload.get("ppo_runs", []))
  done = {r["label"] for r in runs}

  for cfg in configs:
    lbl = cfg.pop("label")
    if lbl in done:
      print(f"SKIP DONE {lbl}", flush=True)
      continue
    print(f"\n=== Running {lbl} ===", flush=True)
    r = run_vision_ppo_rtx(**cfg)
    r["label"] = lbl
    runs.append(r)
    payload["ppo_runs"] = runs
    tmp = args.out_json + ".tmp"
    with open(tmp, "w") as f:
      json.dump(payload, f, indent=2)
    os.replace(tmp, args.out_json)


if __name__ == "__main__":
  main()
