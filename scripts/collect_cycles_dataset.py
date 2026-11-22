"""Sample 384 diverse PandaPickCubeCartesian dynamic states & Stage 1 G-Buffers for Cycles GT."""

from __future__ import annotations

import argparse
import os
import time

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np

from tpu_mjx_render.renderer import RenderSceneSpec, render_batch
from tpu_mjx_render.scenes import get_scene_xml


def sample_panda_states(
    m: mujoco.MjModel,
    mx: mjx.Model,
    spec: RenderSceneSpec,
    n_total: int = 384,
    seed: int = 2026,
    tight_grasp_only: bool = False,
    all_train: bool = False,
):
  """Generate diverse reach / hover / grasp / lift Panda + Cube states via batched vmap."""
  t0 = time.time()
  rng = np.random.default_rng(seed)
  d = mujoco.MjData(m)

  box_gid = m.geom("box").id
  lf_gid = m.geom("left_finger_pad").id
  rf_gid = m.geom("right_finger_pad").id
  hand_gid = m.geom("hand_capsule").id
  grip_sid = m.site("gripper").id

  q_low = m.keyframe("low_home").qpos.copy()
  q_pick = m.keyframe("picked").qpos.copy()
  qpos_all = np.zeros((n_total, m.nq), dtype=np.float32)

  for i in range(n_total):
    mode_i = 2 if tight_grasp_only else (i % 4)
    if mode_i == 0:
      q = q_low.copy()
      q[:7] += rng.normal(0.0, 0.18, size=7)
      q[7:9] = rng.uniform(0.015, 0.04)
      q[9:11] += rng.uniform(-0.12, 0.12, size=2)
      q[11] = 0.03
    elif mode_i == 1:
      q = q_pick.copy()
      q[:7] += rng.normal(0.0, 0.14, size=7)
      q[7:9] = rng.uniform(0.016, 0.024)
      q[9:11] += rng.uniform(-0.10, 0.10, size=2)
      q[11] = rng.uniform(0.03, 0.22)
    elif mode_i == 2:
      alpha = rng.uniform(0.3, 1.0)
      q = (1.0 - alpha) * q_low + alpha * q_pick
      q[:7] += rng.normal(0.0, 0.08, size=7)
      q[7:9] = rng.uniform(0.018, 0.025)
      mujoco.mj_resetData(m, d)
      d.qpos[:] = q
      mujoco.mj_forward(m, d)
      g_site = d.site_xpos[grip_sid].copy()
      q[9:12] = g_site + np.array([rng.uniform(-0.015, 0.015), rng.uniform(-0.015, 0.015), rng.uniform(-0.03, 0.01)])
      q[11] = max(q[11], 0.03)
    else:
      alpha = rng.uniform(0.0, 1.0)
      q = (1.0 - alpha) * q_low + alpha * q_pick
      q[:7] += rng.uniform(-0.28, 0.28, size=7)
      q[7:9] = rng.uniform(0.0, 0.04)
      q[9:11] += rng.uniform(-0.16, 0.16, size=2)
      q[11] = rng.uniform(0.03, 0.18)

    yaw = rng.uniform(-0.6, 0.6)
    q[12:16] = np.array([np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)], dtype=np.float64)
    qpos_all[i] = q.astype(np.float32)

  # Single JIT-compiled batched forward + render across all 384 states
  dx0 = mjx.make_data(m, impl="jax")
  dx_batch = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x[None, ...], (n_total, *x.shape)).copy(), dx0)
  dx_batch = dx_batch.replace(qpos=jnp.asarray(qpos_all, dtype=jnp.float32))

  @jax.jit
  def batch_fwd_and_render(d_b):
    d_fwd = jax.vmap(lambda x: mjx.forward(mx, x))(d_b)
    gb = render_batch(spec, d_fwd, mode="tiled")
    return d_fwd.geom_xpos, d_fwd.geom_xmat, d_fwd.cam_xpos, d_fwd.cam_xmat, gb

  g_xpos, g_xmat, c_xpos, c_xmat, gb = batch_fwd_and_render(dx_batch)
  jax.block_until_ready(gb.rgb)
  print(f"Batched MJX forward + Stage 1 render for {n_total} poses finished in {time.time()-t0:.2f}s", flush=True)

  s1_geom_id = np.asarray(gb.geom_id, dtype=np.int32)
  split = np.zeros((n_total,), dtype=np.int32)
  if not all_train:
    split[288:320] = 1
    split[320:] = 2
  roi_mask = np.isin(s1_geom_id, [box_gid, lf_gid, rf_gid, hand_gid])

  return {
      "geom_type": spec.geom_type,
      "geom_size": spec.geom_size,
      "geom_rgba": spec.geom_rgba,
      "geom_is_checker": spec.geom_is_checker.astype(np.int32),
      "enabled_geom_ids": np.asarray(spec.enabled_geom_ids, dtype=np.int32),
      "checker_rgb1": np.asarray(spec.checker_rgb1, dtype=np.float32),
      "checker_rgb2": np.asarray(spec.checker_rgb2, dtype=np.float32),
      "sky_rgb1": np.asarray(spec.sky_rgb1, dtype=np.float32),
      "sky_rgb2": np.asarray(spec.sky_rgb2, dtype=np.float32),
      "cam_fovy": np.float32(spec.cam_fovy),
      "height": np.int32(spec.height),
      "width": np.int32(spec.width),
      "box_geom_id": np.int32(box_gid),
      "finger_geom_ids": np.asarray([lf_gid, rf_gid], dtype=np.int32),
      "geom_xpos": np.asarray(g_xpos, dtype=np.float32),
      "geom_xmat": np.asarray(g_xmat, dtype=np.float32),
      "cam_xpos": np.asarray(c_xpos[:, spec.cam_id], dtype=np.float32),
      "cam_xmat": np.asarray(c_xmat[:, spec.cam_id], dtype=np.float32),
      "s1_rgb": np.asarray(gb.rgb, dtype=np.float32),
      "s1_albedo": np.asarray(gb.albedo, dtype=np.float32),
      "s1_depth": np.asarray(gb.depth, dtype=np.float32),
      "s1_z_depth": np.asarray(gb.z_depth, dtype=np.float32),
      "s1_normal": np.asarray(gb.normal, dtype=np.float32),
      "s1_view_dir": np.asarray(gb.view_dir, dtype=np.float32),
      "s1_world_pos": np.asarray(gb.world_pos, dtype=np.float32),
      "s1_geom_id": s1_geom_id,
      "s1_shadow": np.asarray(gb.direct_shadow, dtype=np.float32),
      "roi_mask": roi_mask,
      "split": split,
  }


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--n_total", type=int, default=384)
  ap.add_argument("--res", type=int, default=64)
  ap.add_argument("--seed", type=int, default=2026)
  ap.add_argument("--tight_grasp_only", action="store_true")
  ap.add_argument("--all_train", action="store_true")
  ap.add_argument("--out", default="out/data/panda_cycles_dataset.npz")
  args = ap.parse_args()

  os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
  m = mujoco.MjModel.from_xml_string(get_scene_xml("panda_pick_cube"))
  mx = mjx.put_model(m, impl="jax")
  spec = RenderSceneSpec.from_mjmodel(m, cam_id=0, height=args.res, width=args.res)
  ds = sample_panda_states(
      m,
      mx,
      spec,
      n_total=args.n_total,
      seed=args.seed,
      tight_grasp_only=args.tight_grasp_only,
      all_train=args.all_train,
  )
  np.savez_compressed(args.out, **ds)
  print(f"Saved {args.out}: n_total={args.n_total}, res={args.res}x{args.res}")


if __name__ == "__main__":
  main()
