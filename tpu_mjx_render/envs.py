"""Drop-in mujoco_playground-compatible zero-copy Vision RL environments on JAX/TPU."""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np

from tpu_mjx_render.neural_shader import apply_neural_shader, make_neural_shader_model
from tpu_mjx_render.renderer import RenderSceneSpec, render_batch, render_exact_gbuffer
from tpu_mjx_render.scenes import get_scene_xml


class EnvState(NamedTuple):
  data: Any
  obs: Dict[str, jax.Array]
  reward: jax.Array
  done: jax.Array
  step_count: jax.Array
  info: Dict[str, jax.Array]


class CartpoleBalanceVisionEnv:
  """Pure-primitive CartpoleBalance vision environment (zero-copy MJX + TPU Render)."""

  def __init__(
      self,
      height: int = 64,
      width: int = 64,
      episode_length: int = 100,
      action_repeat: int = 2,
      render_mode: str = "tiled",
  ):
    self.mj_model = mujoco.MjModel.from_xml_string(get_scene_xml("cartpole"))
    self.mjx_model = mjx.put_model(self.mj_model, impl="jax")
    self.spec = RenderSceneSpec.from_mjmodel(self.mj_model, cam_id=0, height=height, width=width, use_shadows=True)
    self.height = height
    self.width = width
    self.episode_length = episode_length
    self.action_repeat = action_repeat
    self.render_mode = render_mode
    self.action_size = self.mj_model.nu

  def _render_obs(self, data: Any) -> Dict[str, jax.Array]:
    gb = render_exact_gbuffer(self.spec, data, mode=self.render_mode)
    return {"pixels/view_0": gb.rgb}

  def reset(self, rng: jax.Array) -> EnvState:
    k1, k2 = jax.random.split(rng)
    qpos = jnp.array([
        jax.random.uniform(k1, (), minval=-0.25, maxval=0.25),
        jax.random.uniform(k2, (), minval=-0.12, maxval=0.12),
    ], dtype=jnp.float32)
    dx = mjx.make_data(self.mj_model, impl="jax").replace(qpos=qpos, qvel=jnp.zeros(2, dtype=jnp.float32))
    dx = mjx.forward(self.mjx_model, dx)
    return EnvState(
        data=dx,
        obs=self._render_obs(dx),
        reward=jnp.array(0.0, dtype=jnp.float32),
        done=jnp.array(0.0, dtype=jnp.float32),
        step_count=jnp.array(0, dtype=jnp.int32),
        info={"rng": rng},
    )

  def step(self, state: EnvState, action: jax.Array) -> EnvState:
    u = jnp.clip(action, -1.0, 1.0)
    def _substep(d, _):
      return mjx.step(self.mjx_model, d.replace(ctrl=u)), None
    dx, _ = jax.lax.scan(_substep, state.data, None, length=self.action_repeat)

    cart_x = dx.qpos[0]
    pole_cos = jnp.cos(dx.qpos[1])
    reward = 0.5 + 0.5 * pole_cos - 0.05 * (cart_x ** 2) - 0.01 * jnp.sum(dx.qvel ** 2)
    step_c = state.step_count + 1
    out = (jnp.abs(dx.qpos[1]) > (jnp.pi / 2.0)) | (jnp.abs(cart_x) > 1.8)
    done = (out | (step_c >= self.episode_length)).astype(jnp.float32)
    return EnvState(
        data=dx,
        obs=self._render_obs(dx),
        reward=reward,
        done=done,
        step_count=step_c,
        info=state.info,
    )


class PandaPickCubeVisionEnv:
  """PandaPickCubeCartesian-aligned Vision RL env (supports Stage 1 & Stage 2 Neural)."""

  def __init__(
      self,
      height: int = 64,
      width: int = 64,
      episode_length: int = 80,
      action_repeat: int = 2,
      render_mode: str = "tiled",
      neural_arch: Optional[str] = None,
      neural_params: Optional[Any] = None,
      cal_scale: float = 1.0,
      brightness_range: Tuple[float, float] = (0.9, 1.1),
  ):
    self.mj_model = mujoco.MjModel.from_xml_string(get_scene_xml("panda_pick_cube"))
    self.mjx_model = mjx.put_model(self.mj_model, impl="jax")
    self.spec = RenderSceneSpec.from_mjmodel(self.mj_model, cam_id=0, height=height, width=width, use_shadows=True)
    self.height = height
    self.width = width
    self.episode_length = episode_length
    self.action_repeat = action_repeat
    self.render_mode = render_mode
    self.neural_arch = neural_arch
    self.neural_model = make_neural_shader_model(neural_arch) if neural_arch else None
    self.neural_params = neural_params
    self.cal_scale = float(cal_scale)
    self.brightness_range = brightness_range

    self._box_body = self.mj_model.body("box").id
    self._gripper_site = self.mj_model.site("gripper").id
    self._init_q = jnp.asarray(self.mj_model.keyframe("low_home").qpos, dtype=jnp.float32)
    self._init_ctrl = jnp.asarray(self.mj_model.keyframe("low_home").ctrl, dtype=jnp.float32)
    self._ctrl_lo = jnp.asarray(self.mj_model.actuator_ctrlrange[:, 0], dtype=jnp.float32)
    self._ctrl_hi = jnp.asarray(self.mj_model.actuator_ctrlrange[:, 1], dtype=jnp.float32)
    self.action_size = self.mj_model.nu  # 8

  def _render_obs(self, data: Any, brightness: jax.Array) -> Dict[str, jax.Array]:
    gb = render_exact_gbuffer(self.spec, data, mode=self.render_mode)
    if self.neural_model is not None and self.neural_params is not None:
      rgb = apply_neural_shader(self.neural_model, self.neural_params, gb, cal_scale=self.cal_scale)
    else:
      rgb = gb.rgb
    rgb = jnp.clip(rgb * brightness, 0.0, 1.0)
    return {"pixels/view_0": rgb}

  def render_obs_batch(self, data_batch: Any, brightness_batch: jax.Array, pallas_interpret: bool = False) -> Dict[str, jax.Array]:
    gb = render_batch(self.spec, data_batch, mode=self.render_mode, pallas_interpret=pallas_interpret)
    if self.neural_model is not None and self.neural_params is not None:
      rgb = apply_neural_shader(self.neural_model, self.neural_params, gb, cal_scale=self.cal_scale)
    else:
      rgb = gb.rgb
    rgb = jnp.clip(rgb * brightness_batch, 0.0, 1.0)
    return {"pixels/view_0": rgb}

  def reset_physics(self, rng: jax.Array) -> EnvState:
    k_box, k_arm, k_br, k_tgt = jax.random.split(rng, 4)
    qpos = self._init_q
    # Randomize cube xy on init_space and slight arm variation
    box_dx = jax.random.uniform(k_box, (2,), minval=-0.06, maxval=0.06)
    qpos = qpos.at[9:11].add(box_dx)
    qpos = qpos.at[:7].add(jax.random.uniform(k_arm, (7,), minval=-0.08, maxval=0.08))
    dx = mjx.make_data(self.mj_model, impl="jax").replace(qpos=qpos, ctrl=self._init_ctrl)
    dx = mjx.forward(self.mjx_model, dx)

    target_pos = jnp.array([0.65, 0.0, 0.14], dtype=jnp.float32) + jax.random.uniform(
        k_tgt, (3,), minval=jnp.array([-0.03, -0.03, -0.02]), maxval=jnp.array([0.03, 0.03, 0.04])
    )
    br = jax.random.uniform(k_br, (1, 1, 1), minval=self.brightness_range[0], maxval=self.brightness_range[1])
    grip_pos = dx.site_xpos[self._gripper_site]
    box_pos = dx.xpos[self._box_body]
    d_grip_box = jnp.linalg.norm(grip_pos - box_pos)
    return EnvState(
        data=dx,
        obs={"pixels/view_0": jnp.zeros((self.height, self.width, 3), dtype=jnp.float32)},
        reward=jnp.array(0.0, dtype=jnp.float32),
        done=jnp.array(0.0, dtype=jnp.float32),
        step_count=jnp.array(0, dtype=jnp.int32),
        info={"target_pos": target_pos, "brightness": br, "success": jnp.array(0.0, dtype=jnp.float32), "d_grip_box": d_grip_box},
    )

  def step_physics(self, state: EnvState, action: jax.Array) -> EnvState:
    a = jnp.clip(action, -1.0, 1.0)
    delta_ctrl = a * jnp.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.02], dtype=jnp.float32)
    new_ctrl = jnp.clip(state.data.ctrl + delta_ctrl, self._ctrl_lo, self._ctrl_hi)

    def _substep(d, _):
      return mjx.step(self.mjx_model, d.replace(ctrl=new_ctrl)), None
    dx, _ = jax.lax.scan(_substep, state.data, None, length=self.action_repeat)

    grip_pos = dx.site_xpos[self._gripper_site]
    box_pos = dx.xpos[self._box_body]
    target_pos = state.info["target_pos"]

    d_grip_box = jnp.linalg.norm(grip_pos - box_pos)
    d_box_tgt = jnp.linalg.norm(box_pos - target_pos)
    r_reach = 1.0 - jnp.tanh(5.0 * d_grip_box)
    r_lift = jnp.clip((box_pos[2] - 0.03) * 10.0, 0.0, 1.0)
    r_tgt = 1.0 - jnp.tanh(4.0 * d_box_tgt)
    success = (d_grip_box < 0.05) | (d_box_tgt < 0.06)
    reward = 2.0 * r_reach + 1.5 * r_lift + 3.0 * r_tgt + 0.5 * success.astype(jnp.float32) - 0.005 * jnp.sum(a ** 2)

    step_c = state.step_count + 1
    done = (step_c >= self.episode_length).astype(jnp.float32)
    return EnvState(
        data=dx,
        obs=state.obs,
        reward=reward,
        done=done,
        step_count=step_c,
        info={**state.info, "success": success.astype(jnp.float32), "d_grip_box": d_grip_box},
    )

  def reset(self, rng: jax.Array) -> EnvState:
    st = self.reset_physics(rng)
    return st._replace(obs=self._render_obs(st.data, st.info["brightness"]))

  def step(self, state: EnvState, action: jax.Array) -> EnvState:
    st = self.step_physics(state, action)
    return st._replace(obs=self._render_obs(st.data, state.info["brightness"]))
