"""Phase 3 End-to-End Zero-Copy Vision PPO Closed-Loop Training on JAX/TPU."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from typing import Any, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from tpu_mjx_render.envs import CartpoleBalanceVisionEnv, PandaPickCubeVisionEnv
from tpu_mjx_render.roofline import inspect_hlo_ops


class VisionActorCritic(nn.Module):
  """Flax CNN + MLP Actor-Critic matching mujoco_playground/brax networks_vision."""

  action_dim: int

  @nn.compact
  def __call__(self, obs_rgb: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
    # obs_rgb: (B, H, W, 3) in [0, 1]
    x = obs_rgb.astype(jnp.bfloat16) - 0.5
    x = nn.relu(nn.Conv(32, (3, 3), strides=(2, 2), dtype=jnp.bfloat16, param_dtype=jnp.float32)(x))
    x = nn.relu(nn.Conv(64, (3, 3), strides=(2, 2), dtype=jnp.bfloat16, param_dtype=jnp.float32)(x))
    x = nn.relu(nn.Conv(64, (3, 3), strides=(2, 2), dtype=jnp.bfloat16, param_dtype=jnp.float32)(x))
    x = x.reshape((x.shape[0], -1))
    h = nn.relu(nn.Dense(256, dtype=jnp.bfloat16, param_dtype=jnp.float32)(x)).astype(jnp.float32)
    mean = nn.Dense(self.action_dim, kernel_init=nn.initializers.orthogonal(0.01))(h)
    log_std = self.param("log_std", nn.initializers.constant(-0.5), (self.action_dim,))
    value = nn.Dense(1)(h)[..., 0]
    return mean, jnp.clip(log_std, -2.5, 0.5), value


def run_ppo(
    env_name: str,
    num_envs: int = 32,
    rollout_len: int = 16,
    num_iters: int = 20,
    res: int = 64,
    neural_ckpt: str | None = None,
    render_mode: str = "tiled",
    seed: int = 0,
) -> dict:
  neural_arch, neural_params, cal_scale = None, None, 1.0
  if neural_ckpt and os.path.exists(neural_ckpt):
    ck = pickle.load(open(neural_ckpt, "rb"))
    neural_arch = ck["arch"]
    neural_params = jax.tree_util.tree_map(jnp.asarray, ck["params"])
    cal_scale = float(ck["cal_scale"])

  if env_name == "cartpole":
    env = CartpoleBalanceVisionEnv(height=res, width=res, episode_length=rollout_len * 2, render_mode=render_mode)
  else:
    env = PandaPickCubeVisionEnv(
        height=res,
        width=res,
        episode_length=rollout_len * 2,
        render_mode=render_mode,
        neural_arch=neural_arch,
        neural_params=neural_params,
        cal_scale=cal_scale,
    )

  model = VisionActorCritic(action_dim=env.action_size)
  rng = jax.random.PRNGKey(seed)
  rng, init_k, env_k = jax.random.split(rng, 3)

  pallas_interp = (jax.default_backend() == "cpu")
  use_batched_render = hasattr(env, "step_physics") and render_mode in ("v2_soa", "pallas_v2")
  if use_batched_render:
    def _reset_b(keys):
      st_p = jax.vmap(env.reset_physics)(keys)
      return st_p._replace(obs=env.render_obs_batch(st_p.data, st_p.info["brightness"], pallas_interpret=pallas_interp))
    v_reset = jax.jit(_reset_b)
  else:
    v_reset = jax.jit(jax.vmap(env.reset))
    v_step = jax.vmap(env.step)

  env_keys = jax.random.split(env_k, num_envs)
  state0 = v_reset(env_keys)
  params = model.init(init_k, state0.obs["pixels/view_0"])["params"]

  tx = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(3e-4))
  opt_state = tx.init(params)

  @jax.jit
  def ppo_iter(p, opt_st, env_st, iter_rng):
    def _rollout_step(carry, step_rng):
      st = carry
      obs = st.obs["pixels/view_0"]
      mean, log_std, val = model.apply({"params": p}, obs)
      std = jnp.exp(log_std)
      eps = jax.random.normal(step_rng, mean.shape)
      act = mean + std * eps
      logp = -0.5 * jnp.sum(((act - mean) / std) ** 2 + 2.0 * log_std + jnp.log(2.0 * jnp.pi), axis=-1)

      r_keys = jax.random.split(step_rng, num_envs)
      if use_batched_render:
        next_st = jax.vmap(env.step_physics)(st, act)
        fresh = jax.vmap(env.reset_physics)(r_keys)
        d_flag = next_st.done > 0.5
        def _where_leaf(f_leaf, c_leaf):
          mask = d_flag.reshape((num_envs,) + (1,) * (c_leaf.ndim - 1))
          return jnp.where(mask, f_leaf, c_leaf)
        reset_st = jax.tree_util.tree_map(_where_leaf, fresh, next_st)
        reset_st = reset_st._replace(
            obs=env.render_obs_batch(reset_st.data, reset_st.info["brightness"], pallas_interpret=pallas_interp)
        )
      else:
        next_st = v_step(st, act)
        # Auto-reset finished environments inside JIT graph
        fresh = jax.vmap(env.reset)(r_keys)
        d_flag = next_st.done > 0.5
        def _where_leaf(f_leaf, c_leaf):
          mask = d_flag.reshape((num_envs,) + (1,) * (c_leaf.ndim - 1))
          return jnp.where(mask, f_leaf, c_leaf)
        reset_st = jax.tree_util.tree_map(_where_leaf, fresh, next_st)
      return reset_st, (obs, act, logp, next_st.reward, next_st.done, val)

    step_keys = jax.random.split(iter_rng, rollout_len)
    env_st_out, traj = jax.lax.scan(_rollout_step, env_st, step_keys)
    obs_t, act_t, logp_t, rew_t, done_t, val_t = traj  # (T, B, ...)

    _, _, last_val = model.apply({"params": p}, env_st_out.obs["pixels/view_0"])

    # Compute GAE inside scan
    gamma, lam = 0.99, 0.95
    def _gae(adv_next, xs):
      r, d, v, v_next = xs
      nonterm = 1.0 - d
      delta = r + gamma * v_next * nonterm - v
      adv = delta + gamma * lam * nonterm * adv_next
      return adv, adv

    v_next_all = jnp.concatenate([val_t[1:], last_val[None, :]], axis=0)
    _, adv_t = jax.lax.scan(
        _gae,
        jnp.zeros((num_envs,), dtype=jnp.float32),
        (rew_t, done_t, val_t, v_next_all),
        reverse=True,
    )
    ret_t = adv_t + val_t
    adv_norm = (adv_t - jnp.mean(adv_t)) / (jnp.std(adv_t) + 1e-6)

    # Flatten (T, B, ...) -> (T*B, ...)
    obs_f = obs_t.reshape((-1, *obs_t.shape[2:]))
    act_f = act_t.reshape((-1, act_t.shape[-1]))
    old_lp_f = logp_t.reshape((-1,))
    adv_f = adv_norm.reshape((-1,))
    ret_f = ret_t.reshape((-1,))

    def _ppo_loss(pr):
      mean, log_std, val = model.apply({"params": pr}, obs_f)
      std = jnp.exp(log_std)
      new_lp = -0.5 * jnp.sum(((act_f - mean) / std) ** 2 + 2.0 * log_std + jnp.log(2.0 * jnp.pi), axis=-1)
      ratio = jnp.exp(new_lp - old_lp_f)
      surr1 = ratio * adv_f
      surr2 = jnp.clip(ratio, 0.8, 1.2) * adv_f
      pi_loss = -jnp.mean(jnp.minimum(surr1, surr2))
      v_loss = 0.5 * jnp.mean((val - ret_f) ** 2)
      ent = jnp.mean(jnp.sum(log_std + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e), axis=-1))
      return pi_loss + 0.5 * v_loss - 0.01 * ent

    def _update_step(carry, _):
      pr, ost = carry
      loss, grads = jax.value_and_grad(_ppo_loss)(pr)
      ups, ost = tx.update(grads, ost, pr)
      pr = optax.apply_updates(pr, ups)
      return (pr, ost), loss

    (p_new, opt_new), losses = jax.lax.scan(_update_step, (p, opt_st), None, length=4)
    metrics = {
        "mean_reward": jnp.mean(rew_t),
        "mean_loss": jnp.mean(losses),
        "last_done_frac": jnp.mean(done_t),
    }
    if "success" in env_st_out.info:
      metrics["success_rate"] = jnp.mean(env_st_out.info["success"])
      metrics["d_grip_box"] = jnp.mean(env_st_out.info["d_grip_box"])
    return p_new, opt_new, env_st_out, metrics

  # Verify zero host callbacks in full PPO rollout + update HLO
  hlo_text = ppo_iter.lower(params, opt_state, state0, rng).as_text()
  hlo_info = inspect_hlo_ops(hlo_text)
  assert hlo_info["host_callbacks"] == 0, hlo_info

  history = []
  cur_state = state0
  t_start = time.time()
  t_compile_end = None
  for it in range(num_iters):
    rng, k = jax.random.split(rng)
    params, opt_state, cur_state, m_out = ppo_iter(params, opt_state, cur_state, k)
    jax.block_until_ready(m_out["mean_reward"])
    if it == 0:
      t_compile_end = time.time()
    rec = {"iter": it + 1, **{k_: float(v_) for k_, v_ in m_out.items()}}
    history.append(rec)
    print(f"[{env_name}][iter {it+1}/{num_iters}] {rec}", flush=True)

  t_total = time.time() - t_start
  t_steady = time.time() - (t_compile_end or t_start)
  steady_steps = max(1, (num_iters - 1) * num_envs * rollout_len)
  return {
      "env": env_name,
      "backend": jax.default_backend(),
      "num_envs": num_envs,
      "rollout_len": rollout_len,
      "num_iters": num_iters,
      "neural_arch": neural_arch,
      "render_mode": render_mode,
      "hlo_zero_copy": hlo_info["host_callbacks"] == 0,
      "steady_env_steps_per_sec": steady_steps / max(t_steady, 1e-6),
      "total_wall_s": t_total,
      "history": history,
  }


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--env", default="cartpole", choices=["cartpole", "panda_pick_cube"])
  ap.add_argument("--num_envs", type=int, default=32)
  ap.add_argument("--rollout_len", type=int, default=16)
  ap.add_argument("--num_iters", type=int, default=20)
  ap.add_argument("--res", type=int, default=64)
  ap.add_argument("--neural_ckpt", default=None)
  ap.add_argument("--render_mode", default="tiled", choices=["tiled", "v2_soa", "pallas_v2"])
  ap.add_argument("--out_json", default="out/metrics/ppo_run.json")
  args = ap.parse_args()

  os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
  res = run_ppo(
      env_name=args.env,
      num_envs=args.num_envs,
      rollout_len=args.rollout_len,
      num_iters=args.num_iters,
      res=args.res,
      neural_ckpt=args.neural_ckpt,
      render_mode=args.render_mode,
  )
  with open(args.out_json, "w") as f:
    json.dump(res, f, indent=2)


if __name__ == "__main__":
  main()
