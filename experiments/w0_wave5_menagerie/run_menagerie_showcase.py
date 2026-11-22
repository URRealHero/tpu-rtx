"""Task A: Official google-deepmind/mujoco_menagerie Universal TPU-RTX Showcase.

1. Loads 4 diverse real robots from mujoco_menagerie (`unitree_g1`, `unitree_go2`,
   `franka_emika_panda`, `tetheria_aero_hand_open`).
2. Verifies HLO zero-copy invariants (host_callbacks == 0, send_recv_ops == 0, gather_ops == 0)
   and interpret=False across 4 TPU v5p chips.
3. Benchmarks 4-chip batched MJX physics + tpu_rtx.render at 64x64 and 128x128.
4. Renders 1024x1024 studio-lit GGX PBR + SoftShadow + Hemisphere AO + AgX gallery frames.
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


MENAGERIE_DIR = "/dev/shm/zaychik-tpu-mjx-w0/menagerie"

ROBOT_CONFIGS = [
    {
        "key": "unitree_g1",
        "title": "Unitree G1 Humanoid (35-DOF)",
        "xml": f"{MENAGERIE_DIR}/unitree_g1/scene_mjx.xml",
        "cam_pos": [1.55, -1.45, 1.05],
        "cam_target": [0.0, 0.0, 0.64],
        "cam_fovy": 40.0,
        "ao_radius": 0.35,
        "key_light_pos": [2.2, -2.1, 3.4],
        "rim_light_pos": [-2.0, 1.9, 2.1],
        "fill_light_pos": [-1.8, -1.6, 1.9],
        "floor_z": None,
        "bench_batch": 128,
    },
    {
        "key": "unitree_go2",
        "title": "Unitree Go2 Quadruped (18-DOF)",
        "xml": f"{MENAGERIE_DIR}/unitree_go2/scene_mjx.xml",
        "cam_pos": [0.78, -0.72, 0.46],
        "cam_target": [0.03, 0.0, 0.19],
        "cam_fovy": 36.0,
        "ao_radius": 0.25,
        "key_light_pos": [1.4, -1.4, 2.2],
        "rim_light_pos": [-1.3, 1.2, 1.4],
        "fill_light_pos": [-1.1, -1.1, 1.3],
        "floor_z": None,
        "bench_batch": 128,
    },
    {
        "key": "franka_emika_panda",
        "title": "Franka Emika Panda (9-DOF Arm)",
        "xml": f"{MENAGERIE_DIR}/franka_emika_panda/mjx_scene.xml",
        "cam_pos": [1.05, -0.98, 0.72],
        "cam_target": [0.14, 0.0, 0.42],
        "cam_fovy": 40.0,
        "ao_radius": 0.30,
        "key_light_pos": [1.8, -1.8, 2.8],
        "rim_light_pos": [-1.6, 1.5, 1.7],
        "fill_light_pos": [-1.4, -1.3, 1.6],
        "floor_z": None,
        "bench_batch": 128,
    },
    {
        "key": "tetheria_aero_hand",
        "title": "Tetheria Aero Hand Open (16-DOF)",
        "xml": f"{MENAGERIE_DIR}/tetheria_aero_hand_open/scene_right.xml",
        "cam_pos": [0.30, 0.34, 0.17],
        "cam_target": [0.08, -0.005, -0.02],
        "cam_fovy": 40.0,
        "ao_radius": 0.10,
        "key_light_pos": [0.6, 0.6, 0.9],
        "rim_light_pos": [-0.5, -0.5, 0.5],
        "fill_light_pos": [-0.5, 0.5, 0.5],
        "floor_z": -0.10,
        "bench_batch": 128,
    },
]


def _look_at_xyaxes(pos: List[float], target: List[float]) -> List[float]:
  p = np.asarray(pos, dtype=np.float64)
  t = np.asarray(target, dtype=np.float64)
  fwd = t - p
  fwd = fwd / np.linalg.norm(fwd)
  world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
  right = np.cross(fwd, world_up)
  if np.linalg.norm(right) < 1e-6:
    right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
  right = right / np.linalg.norm(right)
  up = np.cross(right, fwd)
  up = up / np.linalg.norm(up)
  return [float(x) for x in np.concatenate([right, up])]


def _build_bench_model(cfg: Dict[str, Any]) -> mujoco.MjModel:
  """Load official Menagerie MJCF via MjSpec and attach only a hero camera."""
  spec = mujoco.MjSpec.from_file(cfg["xml"])
  xyaxes = _look_at_xyaxes(cfg["cam_pos"], cfg["cam_target"])
  spec.worldbody.add_camera(
      name="hero",
      pos=cfg["cam_pos"],
      xyaxes=xyaxes,
      fovy=cfg["cam_fovy"],
  )
  return spec.compile()


def _build_studio_model(cfg: Dict[str, Any]) -> mujoco.MjModel:
  """Load official Menagerie MJCF via MjSpec and configure crisp 3-point studio lighting & slate floor."""
  spec = mujoco.MjSpec.from_file(cfg["xml"])
  xyaxes = _look_at_xyaxes(cfg["cam_pos"], cfg["cam_target"])
  spec.worldbody.add_camera(
      name="hero",
      pos=cfg["cam_pos"],
      xyaxes=xyaxes,
      fovy=cfg["cam_fovy"],
  )
  # Dim headlight so 3-point studio lights & shadows shape the silhouette cleanly
  spec.visual.headlight.ambient = [0.05, 0.055, 0.065]
  spec.visual.headlight.diffuse = [0.12, 0.12, 0.13]
  spec.visual.headlight.specular = [0.05, 0.05, 0.05]

  # Disable existing lights so our calibrated 3-point studio rig controls shading
  for l in spec.lights:
    l.active = False

  def _dir_to(src, dst):
    v = np.asarray(dst, dtype=np.float64) - np.asarray(src, dtype=np.float64)
    v = v / np.linalg.norm(v)
    return [float(x) for x in v]

  l_key = spec.worldbody.add_light(
      name="studio_key",
      pos=cfg["key_light_pos"],
      dir=_dir_to(cfg["key_light_pos"], cfg["cam_target"]),
      diffuse=[0.96, 0.90, 0.82],
      specular=[0.52, 0.48, 0.42],
      ambient=[0.04, 0.04, 0.045],
      castshadow=True,
  )
  l_key.type = mujoco.mjtLightType.mjLIGHT_SPOT
  l_key.cutoff = 65.0
  l_key.exponent = 1.5

  l_rim = spec.worldbody.add_light(
      name="studio_rim",
      pos=cfg["rim_light_pos"],
      dir=_dir_to(cfg["rim_light_pos"], cfg["cam_target"]),
      diffuse=[0.34, 0.58, 0.92],
      specular=[0.52, 0.72, 0.98],
      ambient=[0.0, 0.0, 0.0],
      castshadow=False,
  )
  l_rim.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL

  l_fill = spec.worldbody.add_light(
      name="studio_fill",
      pos=cfg["fill_light_pos"],
      dir=_dir_to(cfg["fill_light_pos"], cfg["cam_target"]),
      diffuse=[0.32, 0.35, 0.40],
      specular=[0.12, 0.12, 0.14],
      ambient=[0.0, 0.0, 0.0],
      castshadow=False,
  )
  l_fill.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL

  # Update floor plane height if specified (e.g. raise floor under Aero Hand for crisp shadow)
  if cfg["floor_z"] is not None:
    for g in spec.geoms:
      if g.name == "floor":
        g.pos = [0.0, 0.0, float(cfg["floor_z"])]

  # Replace skybox & groundplane textures with dark slate studio grid
  for tx in spec.textures:
    if tx.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
      tx.rgb1 = [0.15, 0.20, 0.28]
      tx.rgb2 = [0.04, 0.055, 0.085]
    elif tx.type == mujoco.mjtTexture.mjTEXTURE_2D:
      tx.rgb1 = [0.14, 0.17, 0.22]
      tx.rgb2 = [0.095, 0.115, 0.15]
      tx.markrgb = [0.28, 0.34, 0.44]

  return spec.compile()


def _pose_robot_for_gallery(key: str, m: mujoco.MjModel) -> mujoco.MjData:
  d = mujoco.MjData(m)
  if m.nkey > 0:
    # Prefer dynamic knees_bent keyframe if present for G1
    kid = 0
    if key == "unitree_g1" and m.nkey > 1:
      kid = 1
    mujoco.mj_resetDataKeyframe(m, d, kid)
  else:
    mujoco.mj_resetData(m, d)

  if key == "franka_emika_panda":
    d.qpos[:9] = np.array([0.28, -0.42, 0.18, -2.02, 0.12, 1.68, 0.78, 0.032, 0.032], dtype=np.float64)
  elif key == "unitree_go2":
    # Slight alert head/leg stance if default qpos has legs straight
    if m.nkey == 0 and m.nq >= 19:
      d.qpos[2] = 0.33
      # FL, FR, RL, RR: (hip, thigh, calf)
      leg = np.array([0.05, 0.78, -1.52, -0.05, 0.78, -1.52, 0.05, 0.85, -1.58, -0.05, 0.85, -1.58])
      d.qpos[7:19] = leg
  elif key == "tetheria_aero_hand":
    # Slightly open expressive finger curl
    if m.nkey > 0:
      d.qpos[0:3] = [0.45, 0.25, 0.25]
      d.qpos[3:6] = [0.35, 0.20, 0.20]
      d.qpos[6:9] = [0.55, 0.30, 0.30]
      d.qpos[9:12] = [0.75, 0.35, 0.35]

  mujoco.mj_forward(m, d)
  return d


def studio_agx_post_tpu(
    rgb: jax.Array,
    normal: jax.Array,
    seg_id: jax.Array,
    floor_gid: int,
    exposure: float = 1.42,
    rim_strength: float = 0.22,
) -> jax.Array:
  """Zero-copy TPU studio Fresnel rim sheen + Blender 4.x AgX Filmic Tonemap + sRGB OETF."""
  rgb_f = rgb.astype(jnp.float32)
  nrm_f = normal.astype(jnp.float32)

  is_fg = (seg_id >= 0) & (seg_id != floor_gid)
  nrm_xy = jnp.sqrt(jnp.maximum(nrm_f[..., 0] ** 2 + nrm_f[..., 1] ** 2, 0.0))
  rim = jnp.where(is_fg, (nrm_xy ** 2.6) * rim_strength, 0.0)[..., None]
  rim_col = jnp.array([0.42, 0.64, 0.98], dtype=jnp.float32)[None, None, :]
  hdr = jnp.maximum(rgb_f * exposure + rim * rim_col, 0.0)

  H, W, _ = hdr.shape
  yy = jnp.linspace(-1.0, 1.0, H, dtype=jnp.float32)[:, None]
  xx = jnp.linspace(-1.0, 1.0, W, dtype=jnp.float32)[None, :]
  r2 = (xx * xx + yy * yy) * 0.5
  hdr = hdr * (1.0 - 0.12 * r2[..., None])

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

  hdr_lin = jnp.power(jnp.maximum(hdr, 1e-6), 1.65)
  val = jnp.einsum("ij,hwj->hwi", agx_inset, jnp.maximum(hdr_lin, 1e-10), precision=jax.lax.Precision.HIGHEST)
  min_ev, max_ev = -6.0, 4.2
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


def benchmark_4chip_menagerie(res_list=(64, 128), n_steps: int = 8) -> List[Dict[str, Any]]:
  """Benchmark 4-chip batched MJX physics + tpu_rtx.render across all 4 Menagerie robots."""
  n_chips = len(jax.devices())
  assert n_chips == 4, f"Expected 4 TPU chips on w0, got {n_chips}"
  rows: List[Dict[str, Any]] = []

  for cfg in ROBOT_CONFIGS:
    key = cfg["key"]
    B = int(cfg["bench_batch"])
    m = _build_bench_model(cfg)
    mx = mjx.put_model(m, impl="jax")
    d_host = _pose_robot_for_gallery(key, m)
    qpos_np = np.tile(np.asarray(d_host.qpos, dtype=np.float32)[None, :], (B, 1))
    # Small perturbation across batch envs so every env has distinct pose
    rng = np.random.default_rng(42)
    qpos_np += rng.uniform(-0.04, 0.04, size=qpos_np.shape).astype(np.float32)
    if m.nq >= 7 and int(m.jnt_type[0]) == 0:  # freejoint root quat normalize
      q = qpos_np[:, 3:7]
      qpos_np[:, 3:7] = q / np.linalg.norm(q, axis=-1, keepdims=True)
    d0 = jax.vmap(lambda q: mjx.make_data(m, impl="jax").replace(qpos=q))(jnp.asarray(qpos_np))
    d0 = jax.jit(jax.vmap(lambda di: mjx.forward(mx, di)))(d0)
    ctrl_seq = jnp.zeros((B, n_steps, m.nu), dtype=jnp.float32)

    for res in res_list:
      sc = tpu_rtx.compile(
          m,
          shader="fast_fused",
          res=(res, res),
          sharding=4,
          cam_id="hero",
          dtype="bfloat16",
          outputs=("rgb", "depth"),
          use_shadows=False,
          enabled_geom_groups=(0, 1, 2),
          interpret=False,
      )
      d_sh = tpu_rtx.shard_data(d0, sc)
      c_sh = tpu_rtx.shard_data(ctrl_seq, sc)

      hlo = tpu_rtx.audit_hlo(tpu_rtx.lower_render(d_sh, sc))
      assert hlo["host_callbacks"] == 0, f"Host callback detected for {key}@{res}: {hlo}"
      assert hlo["send_recv_ops"] == 0, f"Send/recv detected for {key}@{res}: {hlo}"

      # Warmup + measure render-only
      obs_w = tpu_rtx.render(d_sh, sc)
      jax.block_until_ready(obs_w["rgb"])
      n_iters = 10
      t0 = time.perf_counter()
      for _ in range(n_iters):
        obs_r = tpu_rtx.render(d_sh, sc)
      jax.block_until_ready(obs_r["rgb"])
      dt_render_s = (time.perf_counter() - t0) / n_iters
      render_fps = float(B / dt_render_s)

      # Warmup x2 + measure fused mjx.step + tpu_rtx.render (rollout_scan)
      df_w, of_w = tpu_rtx.rollout(mx, d_sh, c_sh, sc, substeps=1, return_frames=False)
      jax.block_until_ready(of_w["rgb"])
      df_w2, of_w2 = tpu_rtx.rollout(mx, d_sh, c_sh, sc, substeps=1, return_frames=False)
      jax.block_until_ready(of_w2["rgb"])
      n_rep = 8
      t1 = time.perf_counter()
      for _ in range(n_rep):
        df, of = tpu_rtx.rollout(mx, d_sh, c_sh, sc, substeps=1, return_frames=False)
      jax.block_until_ready(of["rgb"])
      dt_roll_s = (time.perf_counter() - t1) / n_rep
      coupled_fps = float((B * n_steps) / dt_roll_s)

      n_tris = int(sc.mesh_bvh.total_tris) if sc.mesh_bvh is not None else 0
      n_mesh_geoms = len(sc.mesh_geom_ids)
      row = {
          "robot": key,
          "title": cfg["title"],
          "res": f"{res}x{res}",
          "res_int": int(res),
          "batch_size": B,
          "n_chips": n_chips,
          "nq": int(m.nq),
          "nu": int(m.nu),
          "n_mesh_geoms": n_mesh_geoms,
          "total_tris": n_tris,
          "render_only_ms": round(dt_render_s * 1e3, 3),
          "render_only_fps": round(render_fps, 1),
          "coupled_mjx_rtx_steps": n_steps,
          "coupled_ms_per_rollout": round(dt_roll_s * 1e3, 3),
          "coupled_fps": round(coupled_fps, 1),
          "hlo_audit": hlo,
      }
      print(
          f"[4-Chip Bench] {key:20s} @ {res}x{res} B={B}: "
          f"render={render_fps:,.1f} FPS ({dt_render_s*1e3:.2f} ms), "
          f"MJX+RTX={coupled_fps:,.1f} FPS, tris={n_tris:,}, hc={hlo['host_callbacks']}",
          flush=True,
      )
      rows.append(row)
  return rows


def render_studio_gallery(
    res: int = 1024,
    n_ao: int = 8,
    n_shadow: int = 8,
    out_npz: str = "/dev/shm/zaychik-tpu-mjx-w0/menagerie_gallery.npz",
) -> Dict[str, Any]:
  """Render 1024x1024 GGX PBR + SoftShadow + Hemisphere AO + AgX frames for all 4 Menagerie robots."""
  renders: Dict[str, np.ndarray] = {}
  scenes_meta: Dict[str, Any] = {}

  for cfg in ROBOT_CONFIGS:
    key = cfg["key"]
    m = _build_studio_model(cfg)
    mx = mjx.put_model(m, impl="jax")
    d_host = _pose_robot_for_gallery(key, m)
    dx = mjx.make_data(m, impl="jax").replace(qpos=jnp.asarray(d_host.qpos, jnp.float32))
    dx = mjx.forward(mx, dx)
    floor_gid = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor"))

    sc = tpu_rtx.compile(
        m,
        shader="fused_ao_ggx",
        res=(res, res),
        sharding=1,
        cam_id="hero",
        dtype="float32",
        outputs=("rgb", "depth", "normal", "seg_id"),
        n_ao_samples=n_ao,
        n_shadow_samples=n_shadow,
        ao_radius=float(cfg["ao_radius"]),
        ao_strength=0.82,
        shadow_softness=0.055,
        enabled_geom_groups=(0, 1, 2),
        interpret=False,
    )
    hlo = tpu_rtx.audit_hlo(tpu_rtx.lower_render(dx, sc))
    assert hlo["zero_copy_verified"], f"Gallery HLO check failed for {key}: {hlo}"

    @jax.jit
    def _render_agx(d_in):
      out = tpu_rtx.render(d_in, sc)
      agx = studio_agx_post_tpu(out["rgb"], out["normal"], out["seg_id"], floor_gid=floor_gid)
      return agx, out["depth"], out["seg_id"]

    agx_w, d_w, s_w = _render_agx(dx)
    jax.block_until_ready(agx_w)
    t0 = time.perf_counter()
    agx_img, depth_img, seg_img = _render_agx(dx)
    jax.block_until_ready(agx_img)
    dt_ms = (time.perf_counter() - t0) * 1e3

    img_u8 = np.clip(np.asarray(agx_img) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    fg_frac = float(np.mean((np.asarray(seg_img) >= 0) & (np.asarray(seg_img) != floor_gid)))
    renders[f"{key}_rgb"] = img_u8
    n_tris = int(sc.mesh_bvh.total_tris) if sc.mesh_bvh is not None else 0
    scenes_meta[key] = {
        "title": cfg["title"],
        "res": f"{res}x{res}",
        "n_mesh_geoms": len(sc.mesh_geom_ids),
        "total_tris": n_tris,
        "n_ao_samples": n_ao,
        "n_shadow_samples": n_shadow,
        "dt_render_ms": round(dt_ms, 2),
        "fg_pixel_fraction": round(fg_frac, 4),
        "hlo_audit": hlo,
    }
    print(
        f"[Gallery {res}x{res}] {key:20s}: {n_tris:,} tris, {dt_ms:.1f} ms, "
        f"fg={100*fg_frac:.1f}%, hc={hlo['host_callbacks']}",
        flush=True,
    )

  os.makedirs(os.path.dirname(out_npz) or ".", exist_ok=True)
  np.savez_compressed(out_npz, **renders)
  return {"res": f"{res}x{res}", "n_ao_samples": n_ao, "n_shadow_samples": n_shadow, "scenes": scenes_meta}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--gallery_res", type=int, default=1024)
  ap.add_argument("--n_ao", type=int, default=8)
  ap.add_argument("--n_shadow", type=int, default=8)
  ap.add_argument("--out_npz", default="/dev/shm/zaychik-tpu-mjx-w0/menagerie_gallery.npz")
  ap.add_argument("--out_json", default="/dev/shm/zaychik-tpu-mjx-w0/taskA_menagerie.json")
  args = ap.parse_args()

  bench_rows = benchmark_4chip_menagerie(res_list=(64, 128), n_steps=8)
  gallery_meta = render_studio_gallery(res=args.gallery_res, n_ao=args.n_ao, n_shadow=args.n_shadow, out_npz=args.out_npz)
  payload = {
      "task": "A_mujoco_menagerie_showcase",
      "hardware": "tpu-v5p-4chips-w0",
      "robots": [c["key"] for c in ROBOT_CONFIGS],
      "benchmark_4chip": bench_rows,
      "gallery_pbr": gallery_meta,
  }
  os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
  with open(args.out_json, "w") as f:
    json.dump(payload, f, indent=2)
  print("Saved Task A JSON ->", args.out_json, flush=True)


if __name__ == "__main__":
  main()
