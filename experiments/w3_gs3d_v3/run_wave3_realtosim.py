"""Wave 3 Pillar 3: `tpu_gsplat` Adapter Parity/HLO, Pure `jax.grad` Inverse 3DGS, and Native 4K Hybrid `MJX Panda + 10M 3DGS Sponza` Real-to-Sim Compositor."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from typing import Any, Dict, Tuple

import jax
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np
from PIL import Image

from tpu_mjx_render.adapters.tpu_gsplat import (
    cam_pose_to_viewmat_K,
    composite_mjx_and_3dgs,
    rasterization,
)
from tpu_mjx_render.camera import compute_camera_rays
from tpu_mjx_render.renderer import RenderSceneSpec, _cast_rays_tiled, _shade_scene
from tpu_mjx_render.roofline import inspect_hlo_ops
from tpu_mjx_render.scenes import get_scene_xml

from gltf_to_3dgs import load_gltf_triangles, look_at_cam_mat, sample_anisotropic_3dgs_from_gltf
from gs3d_wave2_10m import make_massive_scene_4c_fn, render_massive_scene_1c


SPONZA_GLTF = "/dev/shm/zaychik-tpu-mjx-w3/assets/gltf-assets/Models/Sponza/glTF/Sponza.gltf"


def _check_hlo(lowered, require_zero_gather: bool = True) -> Dict[str, Any]:
  txt = lowered.as_text()
  info = inspect_hlo_ops(txt)
  info["send_recv_ops"] = len(re.findall(r"\bsend\(|\brecv\(", txt, re.I))
  assert info["host_callbacks"] == 0, f"Found host_callbacks: {info}"
  assert info["send_recv_ops"] == 0, f"Found send_recv_ops: {info}"
  if require_zero_gather:
    assert info["gather_ops"] == 0, f"Expected gather_ops == 0, got: {info}"
  info["zero_copy_verified"] = True
  return info


def _bench_fn(fn, args, n_warmup: int = 2, n_steps: int = 10):
  t0 = time.perf_counter()
  out = fn(*args)
  jax.block_until_ready(out)
  compile_s = time.perf_counter() - t0
  for _ in range(n_warmup):
    out = fn(*args)
  jax.block_until_ready(out)
  t1 = time.perf_counter()
  for _ in range(n_steps):
    out = fn(*args)
  jax.block_until_ready(out)
  ms = (time.perf_counter() - t1) * 1e3 / n_steps
  return ms, compile_s, out


def _save_png(rgb_hwc: np.ndarray, path: str) -> None:
  arr = np.clip(np.round(rgb_hwc * 255.0), 0, 255).astype(np.uint8)
  Image.fromarray(arr).save(path)


def _look_at_opencv_viewmat(cam_pos: np.ndarray, target: np.ndarray, up=(0.0, 1.0, 0.0)) -> np.ndarray:
  fwd = target - cam_pos
  fwd = fwd / max(float(np.linalg.norm(fwd)), 1e-12)
  right = np.cross(fwd, np.array(up, dtype=np.float32))
  right = right / max(float(np.linalg.norm(right)), 1e-12)
  down = np.cross(fwd, right)
  R_wc = np.stack([right, down, fwd], axis=0).astype(np.float32)
  t_wc = -(R_wc @ cam_pos.astype(np.float32))
  V = np.eye(4, dtype=np.float32)
  V[:3, :3] = R_wc
  V[:3, 3] = t_wc
  return V


def _make_synthetic_gaussians(n_g: int, seed: int = 42) -> Dict[str, np.ndarray]:
  rng = np.random.default_rng(seed)
  # Multi-part torus + sphere + pedestal object centered around (0, 0, 0)
  n1 = n_g // 2
  n2 = n_g - n1
  th = rng.uniform(0.0, 2.0 * math.pi, size=n1).astype(np.float32)
  ph = rng.uniform(0.0, 2.0 * math.pi, size=n1).astype(np.float32)
  R_t, r_t = 0.55, 0.18
  x1 = (R_t + r_t * np.cos(ph)) * np.cos(th)
  y1 = r_t * np.sin(ph) + 0.12 * np.sin(2.0 * th)
  z1 = (R_t + r_t * np.cos(ph)) * np.sin(th)
  mu1 = np.stack([x1, y1, z1], axis=-1)
  col1 = np.stack([
      0.5 + 0.45 * np.sin(th),
      0.5 + 0.45 * np.sin(th + 2.094),
      0.5 + 0.45 * np.sin(th + 4.188),
  ], axis=-1).astype(np.float32)

  # Central golden-crimson knot / core
  u = rng.uniform(-1.0, 1.0, size=n2).astype(np.float32)
  ang = rng.uniform(0.0, 2.0 * math.pi, size=n2).astype(np.float32)
  rad_s = 0.28 * np.sqrt(1.0 - u * u)
  mu2 = np.stack([rad_s * np.cos(ang), 0.28 * u, rad_s * np.sin(ang)], axis=-1)
  col2 = np.stack([
      0.85 + 0.10 * u,
      0.35 + 0.30 * np.cos(ang),
      0.15 + 0.15 * np.sin(ang),
  ], axis=-1).astype(np.float32)

  means = np.concatenate([mu1, mu2], axis=0).astype(np.float32)
  colors = np.clip(np.concatenate([col1, col2], axis=0), 0.05, 0.98).astype(np.float32)
  quats = rng.normal(0.0, 1.0, size=(n_g, 4)).astype(np.float32)
  quats = quats / np.maximum(np.linalg.norm(quats, axis=-1, keepdims=True), 1e-8)
  scales = np.full((n_g, 3), 0.032 * math.sqrt(2048.0 / max(n_g, 512)), dtype=np.float32)
  opacities = np.full((n_g,), 0.90, dtype=np.float32)
  return {"means": means, "quats": quats, "scales": scales, "opacities": opacities, "colors": colors}


def run_gsplat_bench(data: Dict[str, Any]) -> None:
  print("--- Running tpu_gsplat Benchmark & Parity ---", flush=True)
  runs = []
  hlo_checks = {}

  # Parity check: compare Pallas (use_approx_topk=True) vs exact top_k (use_approx_topk=False, interpret=False)
  gs_p = _make_synthetic_gaussians(2048, seed=7)
  V0 = _look_at_opencv_viewmat(np.array([0.0, 0.75, -1.65], np.float32), np.array([0.0, 0.0, 0.0], np.float32))
  K0 = np.array([[140.0, 0.0, 64.0], [0.0, 140.0, 64.0], [0.0, 0.0, 1.0]], dtype=np.float32)
  bg = jnp.array([0.1, 0.15, 0.22], dtype=jnp.float32)
  args_p = (
      jnp.asarray(gs_p["means"]), jnp.asarray(gs_p["quats"]), jnp.asarray(gs_p["scales"]),
      jnp.asarray(gs_p["opacities"]), jnp.asarray(gs_p["colors"]),
      jnp.asarray(V0[None]), jnp.asarray(K0[None]),
  )
  fn_exact = jax.jit(lambda m, q, s, o, c, v, k: rasterization(m, q, s, o, c, v, k, 128, 128, backgrounds=bg, use_approx_topk=False, use_bf16_mxu=False))
  fn_fast = jax.jit(lambda m, q, s, o, c, v, k: rasterization(m, q, s, o, c, v, k, 128, 128, backgrounds=bg, use_approx_topk=True, use_bf16_mxu=False))
  fn_bf16 = jax.jit(lambda m, q, s, o, c, v, k: rasterization(m, q, s, o, c, v, k, 128, 128, backgrounds=bg, use_approx_topk=True, use_bf16_mxu=True))

  rgb_ex, alp_ex, dep_ex = fn_exact(*args_p)
  rgb_fa, alp_fa, dep_fa = fn_fast(*args_p)
  rgb_bf, alp_bf, dep_bf = fn_bf16(*args_p)
  jax.block_until_ready((rgb_ex, rgb_fa, rgb_bf))

  mse_fa = float(jnp.mean((rgb_ex - rgb_fa) ** 2))
  mse_bf = float(jnp.mean((rgb_ex - rgb_bf) ** 2))
  data["tpu_gsplat_parity"] = {
      "fp32_approx_vs_exact_max_rgb_diff": round(float(jnp.max(jnp.abs(rgb_ex - rgb_fa))), 6),
      "fp32_approx_vs_exact_depth_mae": round(float(jnp.mean(jnp.abs(dep_ex - dep_fa))), 6),
      "fp32_approx_psnr_db": round(10.0 * math.log10(1.0 / max(mse_fa, 1e-12)), 2),
      "bf16_mxu_max_rgb_diff": round(float(jnp.max(jnp.abs(rgb_ex - rgb_bf))), 6),
      "bf16_mxu_psnr_db": round(10.0 * math.log10(1.0 / max(mse_bf, 1e-12)), 2),
  }
  print("tpu_gsplat parity:", data["tpu_gsplat_parity"], flush=True)

  for n_g in (2048, 8192):
    gs = _make_synthetic_gaussians(n_g, seed=42)
    for C in (1, 4):
      Vs = np.stack([
          _look_at_opencv_viewmat(
              np.array([1.7 * math.cos(a), 0.75, 1.7 * math.sin(a)], np.float32),
              np.array([0.0, 0.0, 0.0], np.float32),
          )
          for a in np.linspace(0.0, 2.0 * math.pi, C, endpoint=False)
      ], axis=0)
      for res_hw in (64, 256):
        fx = float(res_hw) * 1.1
        Ks = np.broadcast_to(
            np.array([[fx, 0.0, 0.5 * res_hw], [0.0, fx, 0.5 * res_hw], [0.0, 0.0, 1.0]], np.float32)[None],
            (C, 3, 3),
        ).copy()
        args = (
            jnp.asarray(gs["means"]), jnp.asarray(gs["quats"]), jnp.asarray(gs["scales"]),
            jnp.asarray(gs["opacities"]), jnp.asarray(gs["colors"]),
            jnp.asarray(Vs), jnp.asarray(Ks),
        )
        for use_bf16 in (False, True):
          fn = jax.jit(
              lambda m, q, s, o, c, v, k, H=res_hw, bf=use_bf16: rasterization(
                  m, q, s, o, c, v, k, H, H, backgrounds=bg, use_bf16_mxu=bf, use_approx_topk=True
              )
          )
          if n_g == 2048 and C == 4 and res_hw == 64 and not use_bf16:
            hlo_checks["tpu_gsplat_2048_c4_64x64_fp32"] = _check_hlo(fn.lower(*args), require_zero_gather=True)
          if n_g == 8192 and C == 1 and res_hw == 256 and not use_bf16:
            hlo_checks["tpu_gsplat_8192_c1_256x256_fp32"] = _check_hlo(fn.lower(*args), require_zero_gather=True)
          ms, comp_s, _ = _bench_fn(fn, args, n_warmup=3, n_steps=15)
          fps = (C * 1000.0) / ms
          rec = {
              "n_gaussians": n_g, "n_views": C, "res": f"{res_hw}x{res_hw}",
              "dtype": "bf16_mxu" if use_bf16 else "fp32",
              "ms_batch": round(ms, 3), "fps_views": round(fps, 1), "compile_s": round(comp_s, 2),
          }
          runs.append(rec)
          print("GSPLAT:", rec, flush=True)

  data["tpu_gsplat_bench"] = runs
  data.setdefault("hlo_checks", {}).update(hlo_checks)


def run_grad_fit(data: Dict[str, Any], export_dir: str) -> None:
  print("--- Running Pure jax.grad Differentiable 3DGS Inverse Fit ---", flush=True)
  C = 8
  H = W = 128
  N_gt = 1024
  N_fit = 1024

  gs_gt = _make_synthetic_gaussians(N_gt, seed=123)
  Vs = np.stack([
      _look_at_opencv_viewmat(
          np.array([1.65 * math.cos(a), 0.65 + 0.25 * math.sin(2 * a), 1.65 * math.sin(a)], np.float32),
          np.array([0.0, 0.0, 0.0], np.float32),
      )
      for a in np.linspace(0.0, 2.0 * math.pi, C, endpoint=False)
  ], axis=0)
  fx = 140.0
  Ks = np.broadcast_to(
      np.array([[fx, 0.0, 0.5 * W], [0.0, fx, 0.5 * H], [0.0, 0.0, 1.0]], np.float32)[None],
      (C, 3, 3),
  ).copy()
  bg = jnp.array([0.08, 0.10, 0.15], dtype=jnp.float32)

  viewmats_d = jnp.asarray(Vs)
  Ks_d = jnp.asarray(Ks)

  # Render exact multi-view GT targets via tpu_gsplat.rasterization
  gt_rgb, _, _ = jax.jit(
      lambda m, q, s, o, c: rasterization(m, q, s, o, c, viewmats_d, Ks_d, W, H, backgrounds=bg, use_approx_topk=False)
  )(
      jnp.asarray(gs_gt["means"]), jnp.asarray(gs_gt["quats"]), jnp.asarray(gs_gt["scales"]),
      jnp.asarray(gs_gt["opacities"]), jnp.asarray(gs_gt["colors"]),
  )
  jax.block_until_ready(gt_rgb)

  # Initialize noisy point cloud
  rng = np.random.default_rng(999)
  init_means = gs_gt["means"] + rng.normal(0.0, 0.14, size=(N_fit, 3)).astype(np.float32)
  init_quats = rng.normal(0.0, 1.0, size=(N_fit, 4)).astype(np.float32)
  init_quats /= np.maximum(np.linalg.norm(init_quats, axis=-1, keepdims=True), 1e-8)
  init_log_scales = np.full((N_fit, 3), math.log(0.045), dtype=np.float32)
  init_logit_op = np.full((N_fit,), 0.4, dtype=np.float32)
  init_logit_col = rng.normal(0.0, 0.5, size=(N_fit, 3)).astype(np.float32)

  params = {
      "means": jnp.asarray(init_means),
      "quats": jnp.asarray(init_quats),
      "log_scales": jnp.asarray(init_log_scales),
      "logit_opacities": jnp.asarray(init_logit_op),
      "logit_colors": jnp.asarray(init_logit_col),
  }

  def _render_from_params(p):
    rgb, _, _ = rasterization(
        p["means"],
        p["quats"],
        jnp.exp(jnp.clip(p["log_scales"], -8.0, -1.5)),
        jax.nn.sigmoid(p["logit_opacities"]),
        jax.nn.sigmoid(p["logit_colors"]),
        viewmats_d,
        Ks_d,
        W,
        H,
        backgrounds=bg,
    )
    return rgb

  def _loss_fn(p):
    pred = _render_from_params(p)
    l1 = jnp.mean(jnp.abs(pred - gt_rgb))
    mse = jnp.mean((pred - gt_rgb) ** 2)
    return 0.8 * l1 + 0.2 * mse, mse

  # Pure JAX Adam optimizer
  lr = {"means": 0.012, "quats": 0.015, "log_scales": 0.03, "logit_opacities": 0.05, "logit_colors": 0.06}
  m_state = jax.tree.map(jnp.zeros_like, params)
  v_state = jax.tree.map(jnp.zeros_like, params)

  @jax.jit
  def _train_step(p, m_s, v_s, step_idx):
    (loss_val, mse_val), grads = jax.value_and_grad(_loss_fn, has_aux=True)(p)
    b1, b2, eps = 0.9, 0.999, 1e-8
    t_f = step_idx.astype(jnp.float32)
    new_p, new_m, new_v = {}, {}, {}
    for k in p:
      g = grads[k]
      mk = b1 * m_s[k] + (1.0 - b1) * g
      vk = b2 * v_s[k] + (1.0 - b2) * (g * g)
      m_hat = mk / (1.0 - (b1 ** t_f))
      v_hat = vk / (1.0 - (b2 ** t_f))
      new_p[k] = p[k] - lr[k] * m_hat / (jnp.sqrt(v_hat) + eps)
      new_m[k] = mk
      new_v[k] = vk
    return new_p, new_m, new_v, loss_val, mse_val

  # HLO zero-copy / zero-gather check on full jax.value_and_grad step!
  hlo_grad = _check_hlo(_train_step.lower(params, m_state, v_state, jnp.array(1, jnp.int32)), require_zero_gather=True)
  data.setdefault("hlo_checks", {})["jax_grad_inverse_3dgs_step"] = hlo_grad
  print("jax.grad HLO verified:", hlo_grad, flush=True)

  init_rgb = jax.jit(_render_from_params)(params)
  jax.block_until_ready(init_rgb)
  init_mse = float(jnp.mean((init_rgb - gt_rgb) ** 2))
  init_psnr = 10.0 * math.log10(1.0 / max(init_mse, 1e-12))

  # Warmup compile
  t0 = time.perf_counter()
  params, m_state, v_state, loss0, mse0 = _train_step(params, m_state, v_state, jnp.array(1, jnp.int32))
  jax.block_until_ready((params, loss0))
  compile_s = time.perf_counter() - t0

  n_steps = 350
  history = [{"step": 0, "psnr_db": round(init_psnr, 2), "mse": round(init_mse, 6)}]
  t_run0 = time.perf_counter()
  for st in range(2, n_steps + 1):
    params, m_state, v_state, loss_v, mse_v = _train_step(params, m_state, v_state, jnp.array(st, jnp.int32))
    if st in (25, 50, 100, 150, 200, 250, 300, 350):
      jax.block_until_ready((loss_v, mse_v))
      psnr_v = 10.0 * math.log10(1.0 / max(float(mse_v), 1e-12))
      history.append({"step": st, "psnr_db": round(psnr_v, 2), "loss": round(float(loss_v), 5)})
      print(f"Grad step {st:3d}/{n_steps}: loss={float(loss_v):.4f}, PSNR={psnr_v:.2f} dB", flush=True)
  jax.block_until_ready(params)
  total_s = time.perf_counter() - t_run0
  steps_per_s = (n_steps - 1) / total_s

  final_rgb = jax.jit(_render_from_params)(params)
  jax.block_until_ready(final_rgb)
  final_mse = float(jnp.mean((final_rgb - gt_rgb) ** 2))
  final_psnr = 10.0 * math.log10(1.0 / max(final_mse, 1e-12))

  data["inverse_3dgs_grad"] = {
      "n_views": C,
      "res": f"{W}x{H}",
      "n_gt_gaussians": N_gt,
      "n_fit_gaussians": N_fit,
      "n_steps": n_steps,
      "compile_s": round(compile_s, 2),
      "total_train_s": round(total_s, 2),
      "steps_per_s": round(steps_per_s, 1),
      "ms_per_step": round(1000.0 / steps_per_s, 2),
      "init_psnr_db": round(init_psnr, 2),
      "final_psnr_db": round(final_psnr, 2),
      "psnr_gain_db": round(final_psnr - init_psnr, 2),
      "history": history,
  }

  # Save side-by-side comparison figure (3 views: Init | Fitted | Ground Truth)
  rows = []
  for v_idx in (0, 2, 5):
    row = np.concatenate([
        np.asarray(init_rgb[v_idx]),
        np.asarray(final_rgb[v_idx]),
        np.asarray(gt_rgb[v_idx]),
    ], axis=1)
    rows.append(row)
  grid = np.concatenate(rows, axis=0)
  out_png = os.path.join(export_dir, "v3_w3_realtosim_grad_fit.png")
  _save_png(grid, out_png)
  print(f"Saved Inverse 3DGS Grad Fit visual: {out_png} (PSNR {init_psnr:.2f} -> {final_psnr:.2f} dB @ {steps_per_s:.1f} step/s)", flush=True)


def run_hybrid_4k(data: Dict[str, Any], export_dir: str) -> None:
  print("--- Running Native 4K Real-to-Sim Hybrid Z-Buffer Compositor (MJX Panda + 10M 3DGS Sponza) ---", flush=True)
  # 1. Load MJX PandaPickCube without floor/init_space so Panda rests directly on Sponza stone floor
  mj_model = mujoco.MjModel.from_xml_string(get_scene_xml("panda_pick_cube"))
  for gname in ("floor", "init_space"):
    gid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, gname)
    if gid >= 0:
      mj_model.geom_rgba[gid, 3] = 0.0
      mj_model.geom_matid[gid] = -1

  mjx_model = mjx.put_model(mj_model, impl="jax")
  key_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, "picked")
  d0 = mjx.make_data(mj_model, impl="jax").replace(
      qpos=jnp.asarray(mj_model.key_qpos[key_id], dtype=jnp.float32),
      ctrl=jnp.asarray(mj_model.key_ctrl[key_id], dtype=jnp.float32),
  )

  # 2. Load 10M Anisotropic 3DGS Sponza
  sc = load_gltf_triangles(SPONZA_GLTF)
  gs = sample_anisotropic_3dgs_from_gltf(sc, 10_000_000, seed=42, splat_scale=1.25, thick_ratio=0.22)

  # Coordinate transform: MJX Z-up -> Sponza Y-up
  # e_x -> (1, 0, 0), e_y -> (0, 0, -1), e_z -> (0, 1, 0), scale 1.45x so Panda is hero-sized in courtyard
  panda_scale = 1.45
  R_ms = jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=jnp.float32)
  # Rotate around Y by -30 deg so Panda faces diagonally toward the camera & courtyard
  ang_y = -0.45
  R_rot_y = jnp.array([
      [math.cos(ang_y), 0.0, math.sin(ang_y)],
      [0.0, 1.0, 0.0],
      [-math.sin(ang_y), 0.0, math.cos(ang_y)],
  ], dtype=jnp.float32)
  R_total = R_rot_y @ R_ms
  # Place Panda near [-4.6, 0.0, 1.25] right beside a foreground colonnade pillar so pillar edge & courtyard interact!
  t_offset = jnp.array([-4.4, 0.0, 0.85], dtype=jnp.float32)

  cam_pos = np.array([-6.9, 1.55, -0.75], dtype=np.float32)
  target = np.array([1.2, 1.85, 0.45], dtype=np.float32)
  cam_mat = look_at_cam_mat(cam_pos, target, world_up=(0.0, 1.0, 0.0))
  fovy, znear = 50.0, 0.10

  devs = jax.devices()[:4]
  mesh = jax.sharding.Mesh(np.array(devs), ("dp",))
  rep = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
  P = jax.sharding.PartitionSpec

  def _make_hybrid_fn(W: int, H: int, k_tile: int = 512, macro_rep: int = 4):
    spec = RenderSceneSpec.from_mjmodel(mj_model, cam_id=0, height=H, width=W, use_shadows=True)
    # Use 4-chip shard_map for 3DGS Sponza + unified analytic MJX raycaster in the same JIT graph!
    gs_4c_fn = make_massive_scene_4c_fn(
        mesh, height=H, width=W, cam_fovy=fovy, znear=znear,
        k_macro=4096, use_bf16_mxu=False, has_nrm=True,
        k_tile=k_tile, macro_rep=macro_rep, return_alpha_depth=True,
    )

    @jax.jit
    def _unified_hybrid(d_in, mu, cov6, rgb, op, nrm, cxp, cxm):
      # A. Live MJX kinematics + physics step inside XLA
      d_step = mjx.step(mjx_model, d_in)
      g_xpos = (d_step.geom_xpos * panda_scale) @ R_total.T + t_offset[None, :]
      g_xmat = jnp.einsum("ij,gjk->gik", R_total, d_step.geom_xmat, precision=jax.lax.Precision.HIGHEST)
      g_size = jnp.asarray(spec.geom_size, dtype=jnp.float32) * panda_scale
      g_rgba = jnp.asarray(spec.geom_rgba, dtype=jnp.float32)

      # Sun light inside Sponza courtyard
      l_pos = jnp.broadcast_to(jnp.array([-2.0, 8.0, -1.5], dtype=jnp.float32), (max(spec.nlight, 1), 3))
      l_dir = jnp.broadcast_to(jnp.array([0.25, -0.92, 0.28], dtype=jnp.float32), (max(spec.nlight, 1), 3))

      ray_orig, ray_dir, neg_z = compute_camera_rays(cxp, cxm, fovy, H, W, znear=znear)
      dist, normal, geom_id = _cast_rays_tiled(
          spec, g_xpos, g_xmat, g_size, ray_orig, ray_dir,
          jnp.full((H * W,), spec.zfar, dtype=jnp.float32), tile_size=32768,
      )
      gb = _shade_scene(
          spec, g_xpos, g_xmat, g_size, g_rgba, cxp, cxm, l_pos, l_dir,
          ray_orig, ray_dir, neg_z, dist, normal, geom_id, mode="tiled",
      )
      mjx_z = jnp.where(gb.geom_id >= 0, gb.z_depth, -1.0)

      # B. 10M 3DGS Sponza (4-chip shard_map)
      gs_rgb, gs_z, gs_alp = gs_4c_fn(mu, cov6, rgb, op, cxp, cxm, nrm)

      # C. Exact per-pixel Z-buffer compositor
      comp_rgb, comp_depth = composite_mjx_and_3dgs(gb.rgb, mjx_z, gs_rgb, gs_alp, gs_z)
      return comp_rgb, comp_depth, mjx_z, gs_z

    return _unified_hybrid

  args_4c = (
      jax.device_put(d0, rep),
      jax.device_put(jnp.asarray(gs["mu"]), rep),
      jax.device_put(jnp.asarray(gs["cov6"]), rep),
      jax.device_put(jnp.asarray(gs["rgb"]), rep),
      jax.device_put(jnp.asarray(gs["opacity"]), rep),
      jax.device_put(jnp.asarray(gs["nrm"]), rep),
      jax.device_put(jnp.asarray(cam_pos), rep),
      jax.device_put(jnp.asarray(cam_mat), rep),
  )

  # Benchmark 1080p and Native 4K
  hybrid_runs = []
  for res_label, W, H in (("1080p", 1920, 1080), ("4K", 3840, 2160)):
    fn_h = _make_hybrid_fn(W, H, k_tile=512, macro_rep=4)
    if res_label == "4K":
      hlo_hyb = _check_hlo(fn_h.lower(*args_4c), require_zero_gather=False)
      data.setdefault("hlo_checks", {})["hybrid_mjx_10m_3dgs_4k"] = hlo_hyb
      print("Hybrid 4K HLO verified:", hlo_hyb, flush=True)
    ms, comp_s, out_t = _bench_fn(fn_h, args_4c, n_warmup=1, n_steps=3)
    comp_rgb, comp_depth, mjx_z, gs_z = out_t
    mjx_pixels = int(jnp.sum(mjx_z > 0))
    occluded_by_3dgs = int(jnp.sum((mjx_z > 0) & (gs_z > 0) & (gs_z < mjx_z)))
    rec = {
        "n_chips": 4, "res_label": res_label, "res": f"{W}x{H}",
        "n_gaussians": 10_000_000, "k_tile": 512, "macro_rep": 4,
        "ms": round(ms, 2), "fps": round(1000.0 / ms, 2), "compile_s": round(comp_s, 2),
        "mjx_robot_pixels": mjx_pixels,
        "mjx_pixels_occluded_by_3dgs_fg": occluded_by_3dgs,
    }
    hybrid_runs.append(rec)
    print(f"HYBRID {res_label}:", rec, flush=True)

    if res_label == "4K":
      out_png = os.path.join(export_dir, "v3_w3_hybrid_mjx_3dgs_4k.png")
      _save_png(np.asarray(comp_rgb), out_png)
      print(f"Saved Native 4K Hybrid MJX + 3DGS Sponza render: {out_png}", flush=True)

  data["hybrid_realtosim_4k"] = hybrid_runs


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--part", default="all", choices=["all", "gsplat_bench", "grad_fit", "hybrid_4k"])
  ap.add_argument("--out_json", default="/dev/shm/zaychik-tpu-mjx-w3/out/v3_w3_wave3_realtosim_hybrid.json")
  ap.add_argument("--export_dir", default="/dev/shm/zaychik-tpu-mjx-w3/out")
  args = ap.parse_args()

  os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
  os.makedirs(args.export_dir, exist_ok=True)
  if os.path.exists(args.out_json):
    with open(args.out_json) as f:
      data = json.load(f)
  else:
    data = {
        "backend": jax.default_backend(),
        "device_kind": jax.devices()[0].device_kind,
        "jax_version": jax.__version__,
    }

  if args.part in ("all", "gsplat_bench"):
    run_gsplat_bench(data)
    with open(args.out_json, "w") as f:
      json.dump(data, f, indent=2)

  if args.part in ("all", "grad_fit"):
    run_grad_fit(data, args.export_dir)
    with open(args.out_json, "w") as f:
      json.dump(data, f, indent=2)

  if args.part in ("all", "hybrid_4k"):
    run_hybrid_4k(data, args.export_dir)
    with open(args.out_json, "w") as f:
      json.dump(data, f, indent=2)


if __name__ == "__main__":
  main()
