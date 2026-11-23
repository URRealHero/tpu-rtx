"""Worker w6: MuJoCo Menagerie 1024x1024 Animated 360-deg Turntables (G1, Go2, Aero Hand)."""

from __future__ import annotations

import argparse
import copy
import gc
import math
import os
import subprocess
import sys
import time

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
  sys.path.insert(0, REPO_ROOT)

import tpu_rtx
from experiments.showcase_videos.video_utils import look_at_mjx_cmat, save_frames_tar
from experiments.w0_wave5_menagerie.run_menagerie_showcase import (
    ROBOT_CONFIGS,
    _build_studio_model,
    _pose_robot_for_gallery,
    studio_agx_post_tpu,
)


def _ensure_menagerie(menagerie_dir: str) -> None:
  if os.path.exists(os.path.join(menagerie_dir, "unitree_g1/scene_mjx.xml")):
    return
  os.makedirs(menagerie_dir, exist_ok=True)
  subprocess.run(
      [
          "git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
          "https://github.com/google-deepmind/mujoco_menagerie.git", menagerie_dir,
      ],
      check=True,
  )
  subprocess.run(
      ["git", "-C", menagerie_dir, "sparse-checkout", "set", "unitree_g1", "unitree_go2", "franka_emika_panda", "tetheria_aero_hand_open"],
      check=True,
  )


def _render_robot_turntable(cfg: dict, menagerie_dir: str, out_dir: str, clip_name: str, n_frames: int = 90, res: int = 1024) -> None:
  c = copy.deepcopy(cfg)
  rel = c["xml"].split("/menagerie/", 1)[1]
  c["xml"] = os.path.join(menagerie_dir, rel)

  m = _build_studio_model(c)
  mx = mjx.put_model(m, impl="jax")
  d_host = _pose_robot_for_gallery(c["key"], m)
  qpos_base = np.asarray(d_host.qpos, dtype=np.float32).copy()
  floor_gid = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor"))

  ao_s = 8 if m.ngeom <= 40 else 4
  sh_s = 8 if m.ngeom <= 40 else 4
  block_r = 512 if m.ngeom <= 40 else 256
  sc = tpu_rtx.compile(
      m, shader="fused_ao_ggx", res=(res, res), sharding=1, cam_id="hero", dtype="float32",
      outputs=("rgb", "normal", "seg_id"), block_b=8, block_r=block_r,
      n_ao_samples=ao_s, n_shadow_samples=sh_s, ao_radius=c["ao_radius"], ao_strength=0.82, shadow_softness=0.05,
  )
  dx_base = mjx.make_data(m, impl="jax")

  @jax.jit
  def _render_pose(qpos_vec, cpos, cmat):
    dx = mjx.forward(mx, dx_base.replace(qpos=qpos_vec))
    dx = dx.replace(
        cam_xpos=dx.cam_xpos.at[int(sc.cam_ids[0])].set(cpos),
        cam_xmat=dx.cam_xmat.at[int(sc.cam_ids[0])].set(cmat),
    )
    out = tpu_rtx.render(dx, sc)
    return studio_agx_post_tpu(out["rgb"], out["normal"], out["seg_id"], floor_gid)

  base_eye = np.asarray(c["cam_pos"], dtype=np.float64)
  target = np.asarray(c["cam_target"], dtype=np.float64)
  rel0 = base_eye - target
  r_xy = float(np.linalg.norm(rel0[:2]))
  ang0 = math.atan2(rel0[1], rel0[0])
  z0 = float(base_eye[2])

  t0 = time.perf_counter()
  warm = _render_pose(jnp.asarray(qpos_base), jnp.asarray(base_eye, jnp.float32), jnp.asarray(look_at_mjx_cmat(base_eye, target), jnp.float32))
  jax.block_until_ready(warm)
  print(f"[w6:{c['key']}] Compiled {res}x{res} in {time.perf_counter() - t0:.1f}s", flush=True)

  frames_u8 = []
  t1 = time.perf_counter()
  for i in range(n_frames):
    u = i / float(n_frames)
    phase = 2.0 * math.pi * u
    ang = ang0 + phase
    eye = np.array([
        target[0] + r_xy * math.cos(ang),
        target[1] + r_xy * math.sin(ang),
        z0 + 0.06 * r_xy * math.sin(phase),
    ], dtype=np.float64)
    cmat = look_at_mjx_cmat(eye, target)

    q = qpos_base.copy()
    if c["key"] == "tetheria_aero_hand":
      # Rhythmic finger wave + wrist tilt while orbiting
      wave = 0.5 + 0.5 * math.sin(phase * 2.0)
      for fi in range(min(12, len(q))):
        q[fi] = max(0.05, qpos_base[fi] + 0.45 * math.sin(phase * 2.0 - 0.35 * (fi // 3)))
      if len(q) >= 16:
        q[12] += 0.18 * math.sin(phase)
    elif c["key"] == "unitree_g1" and len(q) > 20:
      # Expressive upper-body waist/arm posture wave
      q[19] += 0.14 * math.sin(phase * 2.0)
      q[26] -= 0.14 * math.sin(phase * 2.0) if len(q) > 26 else 0.0
    elif c["key"] == "unitree_go2" and len(q) >= 19:
      # Subtle trot leg breathing
      q[8] += 0.08 * math.sin(phase * 2.0)
      q[11] -= 0.08 * math.sin(phase * 2.0)
      q[14] -= 0.08 * math.sin(phase * 2.0)
      q[17] += 0.08 * math.sin(phase * 2.0)

    elif c["key"] == "franka_emika_panda" and len(q) >= 9:
      q[0] += 0.35 * math.sin(phase * 1.0)
      q[1] += 0.22 * math.sin(phase * 2.0)
      q[3] += 0.28 * math.cos(phase * 1.0)
      q[5] += 0.30 * math.sin(phase * 2.0)
      q[7] = 0.02 + 0.018 * (0.5 + 0.5 * math.sin(phase * 2.0))
      q[8] = q[7]

    rgb = _render_pose(jnp.asarray(q, jnp.float32), jnp.asarray(eye, jnp.float32), jnp.asarray(cmat, jnp.float32))
    jax.block_until_ready(rgb)
    frames_u8.append(np.clip(np.round(np.asarray(rgb) * 255.0), 0, 255).astype(np.uint8))

  print(f"[w6:{c['key']}] Rendered {n_frames} turntable frames in {time.perf_counter() - t1:.1f}s", flush=True)
  save_frames_tar(frames_u8, out_dir, clip_name)


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--menagerie_dir", default="/dev/shm/zaychik-tpu-rtx/menagerie")
  ap.add_argument("--out_dir", default="/dev/shm/zaychik-tpu-rtx/out_videos")
  ap.add_argument("--n_frames", type=int, default=120)
  args = ap.parse_args()

  os.makedirs(args.out_dir, exist_ok=True)
  _ensure_menagerie(args.menagerie_dir)

  targets = [
      ("unitree_g1", "menagerie_g1_orbit"),
      ("unitree_go2", "menagerie_go2_orbit"),
      ("franka_emika_panda", "menagerie_panda_orbit"),
      ("tetheria_aero_hand", "menagerie_aerohand_wave"),
  ]
  cfg_by_key = {c["key"]: c for c in ROBOT_CONFIGS}
  for key, clip_name in targets:
    _render_robot_turntable(cfg_by_key[key], args.menagerie_dir, args.out_dir, clip_name, n_frames=args.n_frames)
    jax.clear_caches()
    gc.collect()


if __name__ == "__main__":
  main()
