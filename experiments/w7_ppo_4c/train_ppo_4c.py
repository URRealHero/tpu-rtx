"""V3 4-Chip Multi-Device shard_map Vision PPO on Live TPU v5p.

Fixes V2 Vision PPO instability by adding:
1. Running per-channel observation normalization (initialized from batch 0, EMA synced via pmean).
2. CoordConv spatial channels inside VisionActorCritic for precise 3D visual localization.
3. Running discounted-return scaling (EMA synced via pmean) + clipped value loss.
4. Proper time-limit truncation bootstrapping (separating true termination vs episode timeout).
5. Global batch advantage normalization across all 4 chips (pmean).
6. Epoch + minibatch shuffling cleanly inside jax.jit / shard_map.
7. Warmup-cosine LR schedule (3e-4 -> 3e-5), vf_coef=0.25, max_grad_norm=0.5, log_std floor -1.5.
8. Batched single-render-per-step pallas_v2 wrapper for both CartpoleBalance & PandaPickCube.
9. Full ppo_iter wrapped in jax.experimental.shard_map.shard_map over 4 TPU v5p chips
   with jax.lax.pmean(grads, "dp") and verified host_callbacks == 0.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import pickle
import time
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import flax.linen as nn
import jax
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
from mujoco import mjx
import numpy as np
import optax

from tpu_mjx_render.envs import CartpoleBalanceVisionEnv, EnvState, PandaPickCubeVisionEnv
from tpu_mjx_render.neural_shader import apply_neural_shader, make_neural_shader_model
from tpu_mjx_render.renderer import render_batch
from tpu_mjx_render.roofline import inspect_hlo_ops


class NormState(NamedTuple):
  obs_mean: jax.Array   # (1, 1, 1, 3)
  obs_var: jax.Array    # (1, 1, 1, 3)
  ret_var: jax.Array    # scalar


class VisionActorCritic(nn.Module):
  """Flax CoordConv CNN + LayerNorm + MLP Actor-Critic with bounded log_std."""

  action_dim: int

  @nn.compact
  def __call__(self, obs_norm: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
    b, h_dim, w_dim, _ = obs_norm.shape
    ys = jnp.linspace(-1.0, 1.0, h_dim, dtype=jnp.float32)
    xs = jnp.linspace(-1.0, 1.0, w_dim, dtype=jnp.float32)
    yy, xx = jnp.meshgrid(ys, xs, indexing="ij")
    coords = jnp.broadcast_to(jnp.stack([xx, yy], axis=-1)[None, ...], (b, h_dim, w_dim, 2))
    x = jnp.concatenate([obs_norm, coords], axis=-1).astype(jnp.bfloat16)

    x = nn.relu(
        nn.Conv(
            32,
            (3, 3),
            strides=(2, 2),
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
            kernel_init=nn.initializers.orthogonal(np.sqrt(2.0)),
        )(x)
    )
    x = nn.relu(
        nn.Conv(
            64,
            (3, 3),
            strides=(2, 2),
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
            kernel_init=nn.initializers.orthogonal(np.sqrt(2.0)),
        )(x)
    )
    x = nn.relu(
        nn.Conv(
            64,
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
    log_std = self.param("log_std", nn.initializers.constant(-0.5), (self.action_dim,))
    value = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(h)[..., 0]
    return mean, jnp.clip(log_std, -1.5, 0.5), value


class BatchedCartpoleAdapter:
  """Batched pallas_v2 + optional Stage 2 Neural Shader adapter for CartpoleBalance."""

  def __init__(
      self,
      height: int = 64,
      width: int = 64,
      episode_length: int = 100,
      action_repeat: int = 2,
      render_mode: str = "pallas_v2",
      neural_arch: Optional[str] = None,
      neural_params: Optional[Any] = None,
      cal_scale: float = 1.0,
  ):
    self._base = CartpoleBalanceVisionEnv(
        height=height,
        width=width,
        episode_length=episode_length,
        action_repeat=action_repeat,
        render_mode=render_mode,
    )
    self.mj_model = self._base.mj_model
    self.mjx_model = self._base.mjx_model
    self.spec = self._base.spec
    self.height = height
    self.width = width
    self.episode_length = episode_length
    self.action_repeat = action_repeat
    self.render_mode = render_mode
    self.action_size = self._base.action_size
    self.neural_model = make_neural_shader_model(neural_arch) if neural_arch else None
    self.neural_params = neural_params
    self.cal_scale = float(cal_scale)

  def reset_physics(self, rng: jax.Array, stagger_steps: bool = False) -> EnvState:
    k1, k2, k_st = jax.random.split(rng, 3)
    qpos = jnp.array(
        [
            jax.random.uniform(k1, (), minval=-0.25, maxval=0.25),
            jax.random.uniform(k2, (), minval=-0.12, maxval=0.12),
        ],
        dtype=jnp.float32,
    )
    dx = mjx.make_data(self.mj_model, impl="jax").replace(
        qpos=qpos, qvel=jnp.zeros(2, dtype=jnp.float32)
    )
    dx = mjx.forward(self.mjx_model, dx)
    step_c = (
        jax.random.randint(k_st, (), 0, max(1, self.episode_length), dtype=jnp.int32)
        if stagger_steps
        else jnp.array(0, dtype=jnp.int32)
    )
    pole_cos = jnp.cos(dx.qpos[1])
    balanced = (jnp.abs(dx.qpos[1]) < 0.25).astype(jnp.float32)
    return EnvState(
        data=dx,
        obs={"pixels/view_0": jnp.zeros((self.height, self.width, 3), dtype=jnp.float32)},
        reward=jnp.array(0.0, dtype=jnp.float32),
        done=jnp.array(0.0, dtype=jnp.float32),
        step_count=step_c,
        info={
            "brightness": jnp.ones((1, 1, 1), dtype=jnp.float32),
            "pole_cos": pole_cos,
            "balanced": balanced,
            "trunc": jnp.array(0.0, dtype=jnp.float32),
        },
    )

  def step_physics(self, mx_local: Any, state: EnvState, action: jax.Array) -> EnvState:
    u = jnp.clip(action, -1.0, 1.0)

    def _substep(d, _):
      return mjx.step(mx_local, d.replace(ctrl=u)), None

    dx, _ = jax.lax.scan(_substep, state.data, None, length=self.action_repeat)
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
            "brightness": state.info["brightness"],
            "pole_cos": pole_cos,
            "balanced": balanced,
            "trunc": trunc,
        },
    )

  def render_obs_batch(
      self,
      data_batch: Any,
      brightness_batch: jax.Array,
      neural_params: Optional[Any],
      pallas_interpret: bool = False,
  ) -> Dict[str, jax.Array]:
    del brightness_batch
    gb = render_batch(self.spec, data_batch, mode=self.render_mode, pallas_interpret=pallas_interpret)
    if self.neural_model is not None and neural_params is not None:
      rgb = apply_neural_shader(self.neural_model, neural_params, gb, cal_scale=self.cal_scale)
    else:
      rgb = gb.rgb
    return {"pixels/view_0": rgb}


class BatchedPandaAdapter:
  """Shard-safe adapter around PandaPickCubeVisionEnv using explicit mx_local & neural_params."""

  def __init__(
      self,
      height: int = 64,
      width: int = 64,
      episode_length: int = 80,
      action_repeat: int = 2,
      render_mode: str = "pallas_v2",
      neural_arch: Optional[str] = None,
      neural_params: Optional[Any] = None,
      cal_scale: float = 1.0,
  ):
    self._base = PandaPickCubeVisionEnv(
        height=height,
        width=width,
        episode_length=episode_length,
        action_repeat=action_repeat,
        render_mode=render_mode,
        neural_arch=neural_arch,
        neural_params=neural_params,
        cal_scale=cal_scale,
    )
    self.mj_model = self._base.mj_model
    self.mjx_model = self._base.mjx_model
    self.spec = self._base.spec
    self.height = height
    self.width = width
    self.episode_length = episode_length
    self.action_repeat = action_repeat
    self.render_mode = render_mode
    self.action_size = self._base.action_size
    self.neural_model = self._base.neural_model
    self.neural_params = neural_params
    self.cal_scale = float(cal_scale)

  def reset_physics(self, rng: jax.Array, stagger_steps: bool = False) -> EnvState:
    k_box, k_arm, k_br, k_tgt, k_st = jax.random.split(rng, 5)
    qpos = self._base._init_q
    box_dx = jax.random.uniform(k_box, (2,), minval=-0.06, maxval=0.06)
    qpos = qpos.at[9:11].add(box_dx)
    qpos = qpos.at[:7].add(jax.random.uniform(k_arm, (7,), minval=-0.08, maxval=0.08))
    dx = mjx.make_data(self.mj_model, impl="jax").replace(qpos=qpos, ctrl=self._base._init_ctrl)
    dx = mjx.forward(self.mjx_model, dx)
    target_pos = jnp.array([0.65, 0.0, 0.14], dtype=jnp.float32) + jax.random.uniform(
        k_tgt,
        (3,),
        minval=jnp.array([-0.03, -0.03, -0.02]),
        maxval=jnp.array([0.03, 0.03, 0.04]),
    )
    br = jax.random.uniform(
        k_br, (1, 1, 1), minval=self._base.brightness_range[0], maxval=self._base.brightness_range[1]
    )
    grip_pos = dx.site_xpos[self._base._gripper_site]
    box_pos = dx.xpos[self._base._box_body]
    d_grip_box = jnp.linalg.norm(grip_pos - box_pos)
    step_c = (
        jax.random.randint(k_st, (), 0, max(1, self.episode_length), dtype=jnp.int32)
        if stagger_steps
        else jnp.array(0, dtype=jnp.int32)
    )
    return EnvState(
        data=dx,
        obs={"pixels/view_0": jnp.zeros((self.height, self.width, 3), dtype=jnp.float32)},
        reward=jnp.array(0.0, dtype=jnp.float32),
        done=jnp.array(0.0, dtype=jnp.float32),
        step_count=step_c,
        info={
            "target_pos": target_pos,
            "brightness": br,
            "success": jnp.array(0.0, dtype=jnp.float32),
            "d_grip_box": d_grip_box,
            "trunc": jnp.array(0.0, dtype=jnp.float32),
        },
    )

  def step_physics(self, mx_local: Any, state: EnvState, action: jax.Array) -> EnvState:
    a = jnp.clip(action, -1.0, 1.0)
    delta_ctrl = a * jnp.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.02], dtype=jnp.float32)
    new_ctrl = jnp.clip(state.data.ctrl + delta_ctrl, self._base._ctrl_lo, self._base._ctrl_hi)

    def _substep(d, _):
      return mjx.step(mx_local, d.replace(ctrl=new_ctrl)), None

    dx, _ = jax.lax.scan(_substep, state.data, None, length=self.action_repeat)

    grip_pos = dx.site_xpos[self._base._gripper_site]
    box_pos = dx.xpos[self._base._box_body]
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

  def render_obs_batch(
      self,
      data_batch: Any,
      brightness_batch: jax.Array,
      neural_params: Optional[Any],
      pallas_interpret: bool = False,
  ) -> Dict[str, jax.Array]:
    gb = render_batch(self.spec, data_batch, mode=self.render_mode, pallas_interpret=pallas_interpret)
    if self.neural_model is not None and neural_params is not None:
      rgb = apply_neural_shader(self.neural_model, neural_params, gb, cal_scale=self.cal_scale)
    else:
      rgb = gb.rgb
    rgb = jnp.clip(rgb * brightness_batch, 0.0, 1.0)
    return {"pixels/view_0": rgb}


def _normalize_obs(obs: jax.Array, norm_st: NormState) -> jax.Array:
  std = jnp.maximum(jnp.sqrt(norm_st.obs_var), 0.05)
  return jnp.clip((obs - norm_st.obs_mean) / std, -5.0, 5.0)


def _gaussian_logp(act: jax.Array, mean: jax.Array, log_std: jax.Array) -> jax.Array:
  std = jnp.exp(log_std)
  return -0.5 * jnp.sum(((act - mean) / std) ** 2 + 2.0 * log_std + jnp.log(2.0 * jnp.pi), axis=-1)


def run_ppo_4c(
    env_name: str,
    num_envs: int = 512,
    rollout_len: int = 20,
    num_iters: int = 120,
    num_epochs: int = 4,
    num_minibatches: int = 8,
    res: int = 64,
    episode_length: Optional[int] = None,
    neural_ckpt: Optional[str] = None,
    render_mode: str = "pallas_v2",
    n_chips: int = 4,
    lr_peak: float = 3e-4,
    lr_end: float = 3e-5,
    vf_coef: float = 0.25,
    ent_coef: float = 0.001,
    clip_eps: float = 0.2,
    v_clip_eps: float = 0.2,
    max_grad_norm: float = 0.5,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    seed: int = 0,
) -> Dict[str, Any]:
  devs = jax.devices()[:n_chips]
  assert len(devs) == n_chips, f"Requested {n_chips} devices, found {len(devs)}: {devs}"
  assert num_envs % n_chips == 0, f"num_envs ({num_envs}) must divide n_chips ({n_chips})"
  local_envs = num_envs // n_chips
  assert (rollout_len * local_envs) % num_minibatches == 0, (
      f"T*local_envs={rollout_len * local_envs} must divide num_minibatches={num_minibatches}"
  )
  mb_size = (rollout_len * local_envs) // num_minibatches

  neural_arch, neural_params, cal_scale = None, None, 1.0
  if neural_ckpt and os.path.exists(neural_ckpt):
    with open(neural_ckpt, "rb") as f:
      ck = pickle.load(f)
    neural_arch = ck["arch"]
    neural_params = jax.tree_util.tree_map(jnp.asarray, ck["params"])
    cal_scale = float(ck["cal_scale"])

  if env_name == "cartpole":
    ep_len = episode_length or 100
    init_ret_var = 100.0
    env = BatchedCartpoleAdapter(
        height=res,
        width=res,
        episode_length=ep_len,
        render_mode=render_mode,
        neural_arch=neural_arch,
        neural_params=neural_params,
        cal_scale=cal_scale,
    )
  else:
    ep_len = episode_length or 80
    init_ret_var = 400.0
    env = BatchedPandaAdapter(
        height=res,
        width=res,
        episode_length=ep_len,
        render_mode=render_mode,
        neural_arch=neural_arch,
        neural_params=neural_params,
        cal_scale=cal_scale,
    )

  pallas_interp = jax.default_backend() == "cpu"
  use_dp = n_chips > 1

  mesh = jax.sharding.Mesh(np.array(devs), ("dp",))
  P = jax.sharding.PartitionSpec
  sharding = jax.sharding.NamedSharding(mesh, P("dp"))
  replicate = jax.sharding.NamedSharding(mesh, P())

  model = VisionActorCritic(action_dim=env.action_size)
  rng = jax.random.PRNGKey(seed)
  rng, init_k, env_k = jax.random.split(rng, 3)

  def _init_envs_local(keys_local, mx_l, np_l):
    st_p = jax.vmap(lambda k: env.reset_physics(k, stagger_steps=True))(keys_local)
    obs = env.render_obs_batch(st_p.data, st_p.info["brightness"], np_l, pallas_interpret=pallas_interp)
    return st_p._replace(obs=obs)

  sm_kw = {"check_rep": False} if "check_rep" in inspect.signature(shard_map).parameters else {"check_vma": False}
  if neural_params is not None:
    _init_sm = shard_map(
        _init_envs_local,
        mesh=mesh,
        in_specs=(P("dp"), P(), P()),
        out_specs=P("dp"),
        **sm_kw,
    )
    mx_rep = jax.device_put(env.mjx_model, replicate)
    np_rep = jax.device_put(neural_params, replicate)
    env_keys = jax.device_put(jax.random.split(env_k, num_envs), sharding)
    state0 = jax.jit(_init_sm)(env_keys, mx_rep, np_rep)
  else:
    _init_sm = shard_map(
        lambda kl, mx_l: _init_envs_local(kl, mx_l, None),
        mesh=mesh,
        in_specs=(P("dp"), P()),
        out_specs=P("dp"),
        **sm_kw,
    )
    mx_rep = jax.device_put(env.mjx_model, replicate)
    np_rep = None
    env_keys = jax.device_put(jax.random.split(env_k, num_envs), sharding)
    state0 = jax.jit(_init_sm)(env_keys, mx_rep)

  obs0_np = np.asarray(jax.device_get(state0.obs["pixels/view_0"]))
  init_obs_mean = jnp.asarray(np.mean(obs0_np, axis=(0, 1, 2), keepdims=True), dtype=jnp.float32)
  init_obs_var = jnp.asarray(np.maximum(np.var(obs0_np, axis=(0, 1, 2), keepdims=True), 1e-2), dtype=jnp.float32)

  dummy_obs = jnp.zeros((1, res, res, 3), dtype=jnp.float32)
  params = model.init(init_k, dummy_obs)["params"]
  params = jax.device_put(params, replicate)

  total_updates = num_iters * num_epochs * num_minibatches
  warmup_updates = max(8, int(0.05 * total_updates))
  lr_schedule = optax.warmup_cosine_decay_schedule(
      init_value=lr_end,
      peak_value=lr_peak,
      warmup_steps=warmup_updates,
      decay_steps=max(warmup_updates + 1, total_updates),
      end_value=lr_end,
  )
  tx = optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.adam(learning_rate=lr_schedule, eps=1e-5))
  opt_state = jax.device_put(tx.init(params), replicate)

  norm_state = NormState(
      obs_mean=init_obs_mean,
      obs_var=init_obs_var,
      ret_var=jnp.array(init_ret_var, dtype=jnp.float32),
  )
  norm_state = jax.device_put(norm_state, replicate)
  ret_acc0 = jax.device_put(jnp.zeros((num_envs,), dtype=jnp.float32), sharding)

  def _dp_mean(x):
    return jax.lax.pmean(x, "dp") if use_dp else x

  def _ppo_iter_local(p, opt_st, n_st, env_st, r_acc, chip_rng_in, mx_l, np_l):
    chip_rng = chip_rng_in[0] if chip_rng_in.ndim >= 2 or (chip_rng_in.ndim == 1 and chip_rng_in.shape[0] == 1) else chip_rng_in
    roll_rng, epoch_rng = jax.random.split(chip_rng)

    def _rollout_step(carry, step_rng):
      st, acc = carry
      obs_raw = st.obs["pixels/view_0"]
      obs_n = _normalize_obs(obs_raw, n_st)
      mean, log_std, val = model.apply({"params": p}, obs_n)
      std = jnp.exp(log_std)
      k_act, k_rst = jax.random.split(step_rng)
      eps = jax.random.normal(k_act, mean.shape)
      act = mean + std * eps
      logp = _gaussian_logp(act, mean, log_std)

      next_st = jax.vmap(lambda s, a: env.step_physics(mx_l, s, a))(st, act)
      r_keys = jax.random.split(k_rst, local_envs)
      fresh = jax.vmap(lambda k: env.reset_physics(k, stagger_steps=False))(r_keys)
      d_flag = next_st.done > 0.5

      def _where_leaf(f_leaf, c_leaf):
        mask = d_flag.reshape((local_envs,) + (1,) * (c_leaf.ndim - 1))
        return jnp.where(mask, f_leaf, c_leaf)

      reset_st = jax.tree_util.tree_map(_where_leaf, fresh, next_st)
      reset_st = reset_st._replace(
          obs=env.render_obs_batch(reset_st.data, reset_st.info["brightness"], np_l, pallas_interpret=pallas_interp)
      )
      new_acc = acc * gamma * (1.0 - next_st.done) + next_st.reward
      if env_name == "panda_pick_cube":
        aux_m = (next_st.info["success"], next_st.info["d_grip_box"])
      else:
        aux_m = (next_st.info["balanced"], next_st.info["pole_cos"])
      return (reset_st, new_acc), (
          obs_raw,
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
    obs_t, act_t, logp_t, rew_t, done_t, trunc_t, val_t, acc_t, aux_t = traj

    b_obs_mean = _dp_mean(jnp.mean(obs_t, axis=(0, 1, 2, 3), keepdims=True)[0])
    b_obs_var = _dp_mean(jnp.mean((obs_t - b_obs_mean[None, ...]) ** 2, axis=(0, 1, 2, 3), keepdims=True)[0])
    b_ret_mean = _dp_mean(jnp.mean(acc_t))
    b_ret_var = _dp_mean(jnp.mean((acc_t - b_ret_mean) ** 2))
    mom_obs = 0.02
    mom_ret = 0.05
    new_n_st = NormState(
        obs_mean=(1.0 - mom_obs) * n_st.obs_mean + mom_obs * b_obs_mean,
        obs_var=(1.0 - mom_obs) * n_st.obs_var + mom_obs * b_obs_var,
        ret_var=(1.0 - mom_ret) * n_st.ret_var + mom_ret * jnp.maximum(b_ret_var, 1.0),
    )

    last_obs_n = _normalize_obs(env_st_out.obs["pixels/view_0"], new_n_st)
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

    obs_norm_t = _normalize_obs(obs_t, new_n_st)
    n_local = rollout_len * local_envs
    obs_f = obs_norm_t.reshape((n_local, res, res, 3))
    act_f = act_t.reshape((n_local, env.action_size))
    old_lp_f = logp_t.reshape((n_local,))
    old_val_f = val_t.reshape((n_local,))
    adv_f = adv_norm.reshape((n_local,))
    ret_f = ret_t.reshape((n_local,))

    def _ppo_loss(pr, mb):
      mb_obs, mb_act, mb_old_lp, mb_old_val, mb_adv, mb_ret = mb
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
          obs_f[perm].reshape((num_minibatches, mb_size, res, res, 3)),
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

    metrics = {
        "mean_reward": _dp_mean(jnp.mean(rew_t)),
        "mean_loss": mean_loss,
        "pi_loss": mean_pi_loss,
        "v_loss": mean_v_loss,
        "entropy": mean_ent,
        "approx_kl": mean_kl,
        "done_frac": _dp_mean(jnp.mean(done_t)),
        "ret_scale": ret_scale,
    }
    if env_name == "panda_pick_cube":
      succ_t, dgb_t = aux_t
      metrics["success_rate"] = _dp_mean(jnp.mean(succ_t))
      metrics["d_grip_box"] = _dp_mean(jnp.mean(dgb_t))
    else:
      bal_t, pcos_t = aux_t
      metrics["success_rate"] = _dp_mean(jnp.mean(bal_t))
      metrics["pole_cos"] = _dp_mean(jnp.mean(pcos_t))

    return p_new, opt_new, new_n_st, env_st_out, r_acc_out, metrics

  if neural_params is not None:
    _ppo_sm = shard_map(
        _ppo_iter_local,
        mesh=mesh,
        in_specs=(P(), P(), P(), P("dp"), P("dp"), P("dp"), P(), P()),
        out_specs=(P(), P(), P(), P("dp"), P("dp"), P()),
        **sm_kw,
    )

    @jax.jit
    def ppo_iter(p, ost, nst, est, racc, crng):
      return _ppo_sm(p, ost, nst, est, racc, crng, mx_rep, np_rep)
  else:
    _ppo_sm = shard_map(
        lambda p, ost, nst, est, racc, crng, mx_l: _ppo_iter_local(p, ost, nst, est, racc, crng, mx_l, None),
        mesh=mesh,
        in_specs=(P(), P(), P(), P("dp"), P("dp"), P("dp"), P()),
        out_specs=(P(), P(), P(), P("dp"), P("dp"), P()),
        **sm_kw,
    )

    @jax.jit
    def ppo_iter(p, ost, nst, est, racc, crng):
      return _ppo_sm(p, ost, nst, est, racc, crng, mx_rep)

  rng, k0 = jax.random.split(rng)
  chip_rngs0 = jax.device_put(jax.random.split(k0, n_chips), sharding)
  hlo_text = ppo_iter.lower(params, opt_state, norm_state, state0, ret_acc0, chip_rngs0).as_text()
  hlo_info = inspect_hlo_ops(hlo_text)
  assert hlo_info["host_callbacks"] == 0, f"Host callbacks detected in HLO: {hlo_info}"

  history = []
  cur_state = state0
  cur_acc = ret_acc0
  t_start = time.time()
  t_compile_end = None
  for it in range(num_iters):
    rng, k = jax.random.split(rng)
    chip_rngs = jax.device_put(jax.random.split(k, n_chips), sharding)
    params, opt_state, norm_state, cur_state, cur_acc, m_out = ppo_iter(
        params, opt_state, norm_state, cur_state, cur_acc, chip_rngs
    )
    jax.block_until_ready(m_out["mean_reward"])
    if it == 0:
      t_compile_end = time.time()
    rec = {"iter": it + 1, **{k_: float(v_) for k_, v_ in m_out.items()}}
    history.append(rec)
    if it == 0 or (it + 1) % 10 == 0 or it == num_iters - 1:
      extra = f"dgb={rec['d_grip_box']:.3f}" if "d_grip_box" in rec else f"pcos={rec['pole_cos']:.3f}"
      print(
          f"[{env_name}][{render_mode}][{neural_arch or 'stage1'}][n={n_chips}c B={num_envs}][{it+1}/{num_iters}] "
          f"rew={rec['mean_reward']:.3f} succ={rec['success_rate']:.3f} {extra} "
          f"loss={rec['mean_loss']:.4f} vloss={rec['v_loss']:.4f} kl={rec['approx_kl']:.4f} ent={rec['entropy']:.2f}",
          flush=True,
      )

  t_total = time.time() - t_start
  t_steady = time.time() - (t_compile_end or t_start)
  steady_steps = max(1, (num_iters - 1) * num_envs * rollout_len)
  compile_s = (t_compile_end - t_start) if t_compile_end else 0.0
  steady_sps = steady_steps / max(t_steady, 1e-6)

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
      "episode_length": ep_len,
      "res": res,
      "neural_arch": neural_arch,
      "render_mode": render_mode,
      "hlo_zero_copy": hlo_info["host_callbacks"] == 0,
      "hlo_ops": hlo_info,
      "compile_s": compile_s,
      "steady_wall_s": t_steady,
      "total_wall_s": t_total,
      "steady_env_steps_per_sec": steady_sps,
      "initial_metrics": history[0] if history else {},
      "final_metrics": history[-1] if history else {},
      "best_reward": max(r["mean_reward"] for r in history) if history else 0.0,
      "best_success_rate": max(r["success_rate"] for r in history) if history else 0.0,
      "history": history,
  }


def get_suite_configs(m2_ckpt: str) -> List[Dict[str, Any]]:
  return [
      {
          "label": "cartpole_stage1_b512_4c",
          "env_name": "cartpole",
          "num_envs": 512,
          "rollout_len": 20,
          "num_iters": 120,
          "neural_ckpt": None,
          "render_mode": "pallas_v2",
          "n_chips": 4,
      },
      {
          "label": "cartpole_stage2_m2_b512_4c",
          "env_name": "cartpole",
          "num_envs": 512,
          "rollout_len": 20,
          "num_iters": 120,
          "neural_ckpt": m2_ckpt,
          "render_mode": "pallas_v2",
          "n_chips": 4,
      },
      {
          "label": "panda_stage1_b512_4c",
          "env_name": "panda_pick_cube",
          "num_envs": 512,
          "rollout_len": 20,
          "num_iters": 150,
          "neural_ckpt": None,
          "render_mode": "pallas_v2",
          "n_chips": 4,
      },
      {
          "label": "panda_stage1_b1024_4c",
          "env_name": "panda_pick_cube",
          "num_envs": 1024,
          "rollout_len": 20,
          "num_iters": 120,
          "neural_ckpt": None,
          "render_mode": "pallas_v2",
          "n_chips": 4,
      },
      {
          "label": "panda_stage2_m2_b256_4c",
          "env_name": "panda_pick_cube",
          "num_envs": 256,
          "rollout_len": 20,
          "num_iters": 150,
          "neural_ckpt": m2_ckpt,
          "render_mode": "pallas_v2",
          "n_chips": 4,
      },
      {
          "label": "panda_stage2_m2_b512_4c",
          "env_name": "panda_pick_cube",
          "num_envs": 512,
          "rollout_len": 20,
          "num_iters": 150,
          "neural_ckpt": m2_ckpt,
          "render_mode": "pallas_v2",
          "n_chips": 4,
      },
  ]


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--suite", action="store_true", help="Run V3 4-chip convergence + scaling suite")
  ap.add_argument("--labels", nargs="*", default=None, help="Optional subset of suite labels to run")
  ap.add_argument("--env", default="panda_pick_cube", choices=["cartpole", "panda_pick_cube"])
  ap.add_argument("--num_envs", type=int, default=512)
  ap.add_argument("--rollout_len", type=int, default=20)
  ap.add_argument("--num_iters", type=int, default=120)
  ap.add_argument("--num_epochs", type=int, default=4)
  ap.add_argument("--num_minibatches", type=int, default=8)
  ap.add_argument("--res", type=int, default=64)
  ap.add_argument("--n_chips", type=int, default=4)
  ap.add_argument("--neural_ckpt", default=None)
  ap.add_argument("--render_mode", default="pallas_v2", choices=["tiled", "v2_soa", "pallas_v2"])
  ap.add_argument("--out_json", default="out/metrics/v3_w7_ppo_4c.json")
  args = ap.parse_args()

  os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
  if not args.suite:
    res = run_ppo_4c(
        env_name=args.env,
        num_envs=args.num_envs,
        rollout_len=args.rollout_len,
        num_iters=args.num_iters,
        num_epochs=args.num_epochs,
        num_minibatches=args.num_minibatches,
        res=args.res,
        neural_ckpt=args.neural_ckpt,
        render_mode=args.render_mode,
        n_chips=args.n_chips,
    )
    with open(args.out_json, "w") as f:
      json.dump(res, f, indent=2)
    print(f"Saved -> {args.out_json} (steady={res['steady_env_steps_per_sec']:.1f} steps/s)", flush=True)
    return

  m2_ckpt = args.neural_ckpt or "/dev/shm/zaychik-tpu-mjx/out/metrics/M2_spatial_128_params.pkl"
  if not os.path.exists(m2_ckpt):
    m2_ckpt = "out/metrics/M2_spatial_128_params.pkl"

  configs = get_suite_configs(m2_ckpt)
  if args.labels:
    want = set(args.labels)
    configs = [c for c in configs if c["label"] in want]

  runs = []
  for cfg in configs:
    label = cfg.pop("label")
    print(f"\n=== Running {label} ===", flush=True)
    res = run_ppo_4c(**cfg)
    res["label"] = label
    runs.append(res)
    print(
        f"=== Done {label}: {res['steady_env_steps_per_sec']:.1f} env-steps/s | "
        f"rew {res['initial_metrics']['mean_reward']:.3f} -> {res['final_metrics']['mean_reward']:.3f} | "
        f"succ {res['initial_metrics']['success_rate']:.3f} -> {res['final_metrics']['success_rate']:.3f} ===",
        flush=True,
    )

  summary = {
      "experiment": "v3_w7_ppo_4c",
      "backend": jax.default_backend(),
      "n_chips": 4,
      "runs": runs,
  }
  with open(args.out_json, "w") as f:
    json.dump(summary, f, indent=2)
  print(f"\nALL RUNS SAVED -> {args.out_json}", flush=True)


if __name__ == "__main__":
  main()
