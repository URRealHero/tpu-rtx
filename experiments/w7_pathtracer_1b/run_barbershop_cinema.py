"""Task B: Film-Studio Cinema Showcase — Blender 'Barbershop Interior' on TPU v5p.

Dual-mode script:
  1. Inside Blender 5.3.0 (`blender -b barbershop_interior.blend --python ... -- --export_only`):
     Compiles the `.blend` via `blend_compiler.py` (with texture-size alignment on
     mixed procedural/image shaders) and renders Cycles 5.3.0 GT + EXR Z-depth for
     24 orbit views and 4 hero views.
  2. On TPU v5p (`train_venv/bin/python`):
     Runs Stage-1 Wide-BVH-64 UV-Textured ray tracing + Stage-2 Neural Relighting
     (`IrradianceNRC3S_Demod_W128`), evaluates PSNR/SSIM/Z-MAE vs Cycles 5.3.0 GT,
     benchmarks 4-chip 1080p & Native 4K throughput (`host_callbacks == 0`), and
     produces `out/figures/v3_w7_barbershop_cinema.png`.
"""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
import os
import re
import sys
import time
from typing import Any, Dict, Tuple

import numpy as np

try:
  import bpy  # type: ignore
  HAS_BPY = True
except Exception:
  bpy = None  # type: ignore
  HAS_BPY = False


# ==============================================================================
# Part 1: Blender 5.3.0 Export & Cycles 5.3.0 GT + Z-Depth Generator
# ==============================================================================

def _blender_export_main() -> None:
  adapter_dir = "/dev/shm/zaychik-tpu-mjx-w7/tpu_mjx_render/adapters"
  if adapter_dir not in sys.path:
    sys.path.insert(0, adapter_dir)
  import blend_compiler as bc
  from blend_compiler import _ShaderGraphWalker, _resize_bilinear, compile_active_blend, render_cycles_gt_and_depth

  # Monkey-patch MIX_SHADER / ADD_SHADER / MIX_FLOAT to safely align mixed procedural & image texture shapes
  orig_eval_shader = _ShaderGraphWalker._eval_shader_socket
  orig_eval_scalar = _ShaderGraphWalker._eval_scalar

  def _align_tex(ta: np.ndarray, tb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if ta.shape[:2] == tb.shape[:2]:
      return ta, tb
    th = max(ta.shape[0], tb.shape[0])
    tw = max(ta.shape[1], tb.shape[1])
    if ta.shape[:2] != (th, tw):
      ta = _resize_bilinear(ta, th, tw)
    if tb.shape[:2] != (th, tw):
      tb = _resize_bilinear(tb, th, tw)
    return ta, tb

  def _safe_eval_scalar(self, sock, default, group_stack, depth=0):
    val, tex = orig_eval_scalar(self, sock, default, group_stack, depth)
    return val, tex

  # Wrap _eval_shader_socket at MIX_SHADER so mismatched procedural+image textures resize cleanly
  def _safe_eval_shader_socket(self, sock, group_stack, depth=0):
    if sock is not None and sock.is_linked and depth <= 12:
      resolved, new_stack = self._resolve_socket_link(sock, group_stack)
      if resolved is not None and isinstance(resolved, tuple):
        fn, _fs = resolved
        if fn.type == "MIX_SHADER":
          fac, _ = self._eval_scalar(fn.inputs[0], 0.5, new_stack, depth + 1)
          fac = float(np.clip(fac, 0.0, 1.0))
          s1 = _safe_eval_shader_socket(self, fn.inputs[1], new_stack, depth + 1)
          s2 = _safe_eval_shader_socket(self, fn.inputs[2], new_stack, depth + 1)
          if s2["alpha"] < 0.05 and s1["alpha"] > 0.5:
            s1["alpha"] *= (1.0 - fac)
            return s1
          if s1["alpha"] < 0.05 and s2["alpha"] > 0.5:
            s2["alpha"] *= fac
            return s2
          out = dict(s1)
          for k in ("base_color", "roughness", "metallic", "transmission", "ior", "emissive_color", "emissive_strength", "alpha"):
            out[k] = (1.0 - fac) * s1[k] + fac * s2[k]
          t1, t2 = s1["tex_albedo"], s2["tex_albedo"]
          if t1 is not None and t2 is not None:
            t1, t2 = _align_tex(t1, t2)
            out["tex_albedo"] = (1.0 - fac) * t1 + fac * t2
          elif t2 is not None and fac > 0.35:
            out["tex_albedo"] = (1.0 - fac) * s1["base_color"][None, None, :] + fac * t2
            out["uv_scale_offset"] = s2["uv_scale_offset"]
            out["uv_name"] = s2["uv_name"]
          elif t1 is not None:
            out["tex_albedo"] = (1.0 - fac) * t1 + fac * s2["base_color"][None, None, :]
          return out
    return orig_eval_shader(self, sock, group_stack, depth)

  _ShaderGraphWalker._eval_shader_socket = _safe_eval_shader_socket

  # Also guard MIX FLOAT texture blending inside _eval_scalar if needed
  def _patched_eval_scalar(self, sock, default, group_stack, depth=0):
    try:
      return orig_eval_scalar(self, sock, default, group_stack, depth)
    except ValueError:
      return float(default), None

  _ShaderGraphWalker._eval_scalar = _patched_eval_scalar

  argv = sys.argv
  argv = argv[argv.index("--") + 1 :] if "--" in argv else []
  ap = argparse.ArgumentParser()
  ap.add_argument("--out_dir", required=True)
  ap.add_argument("--scene_name", default="barbershop_interior")
  ap.add_argument("--max_tex_res", type=int, default=512)
  ap.add_argument("--orbit_views", type=int, default=32)
  ap.add_argument("--orbit_spp", type=int, default=24)
  ap.add_argument("--hero_views", type=int, default=4)
  ap.add_argument("--hero_spp", type=int, default=64)
  ap.add_argument("--skip_pack", action="store_true")
  ap.add_argument("--skip_orbit", action="store_true")
  ap.add_argument("--skip_hero", action="store_true")
  args = ap.parse_args(argv)

  os.makedirs(args.out_dir, exist_ok=True)
  pack_p = os.path.join(args.out_dir, "barbershop_pack.npz")
  orb_p = os.path.join(args.out_dir, "barbershop_gt_orbit.npz")
  hero_p = os.path.join(args.out_dir, "barbershop_gt_hero.npz")
  hero_png = os.path.join(args.out_dir, "barbershop_cycles_hero0.png")

  if args.skip_pack and os.path.exists(pack_p):
    pack = bc.ScenePack.load(pack_p)
  else:
    # Hide pure-volume domains (e.g. ENV-Fog.001 where Material Output has no Surface link)
    # so they do not appear as opaque spheres blocking the room.
    def _is_pure_volume_mat(mat) -> bool:
      if mat is None or not getattr(mat, "use_nodes", False) or mat.node_tree is None:
        return False
      out_nd = next((n for n in mat.node_tree.nodes if n.type == "OUTPUT_MATERIAL" and n.is_active_output), None)
      if out_nd is None:
        out_nd = next((n for n in mat.node_tree.nodes if n.type == "OUTPUT_MATERIAL"), None)
      if out_nd is None:
        return False
      surf_in = out_nd.inputs.get("Surface")
      vol_in = out_nd.inputs.get("Volume")
      return (surf_in is None or not surf_in.is_linked) and (vol_in is not None and vol_in.is_linked)

    hidden_objs = []
    for o in bpy.data.objects:
      if o.type != "MESH" or o.hide_render:
        continue
      mats = [s.material for s in o.material_slots if s.material is not None]
      if o.name.startswith("ENV-Fog") or (len(mats) > 0 and all(_is_pure_volume_mat(m) for m in mats)):
        o.hide_render = True
        hidden_objs.append(o)

    pack = compile_active_blend(scene_name=args.scene_name, max_tex_res=args.max_tex_res)
    for o in hidden_objs:
      o.hide_render = False

    # Strip mostly-transparent glass/overlay panes (alpha < 0.35, e.g. display counter glass & picture frame glass)
    alpha_by_mid = np.array([float(m.get("alpha", 1.0)) for m in pack.materials], dtype=np.float32)
    keep_tri = alpha_by_mid[np.clip(pack.tri_mat_id, 0, len(alpha_by_mid) - 1)] >= 0.35
    # Also strip pure Volume/fog material triangles if mixed on an object
    vol_mids = {i for i, m in enumerate(pack.materials) if "fog" in m.get("name", "").lower()}
    if vol_mids:
      for vm in vol_mids:
        keep_tri &= (pack.tri_mat_id != vm)
    if not np.all(keep_tri) and np.any(keep_tri):
      print(f"FILTERED {int(np.sum(~keep_tri)):,} transparent/volume tris -> keeping {int(np.sum(keep_tri)):,}", flush=True)
      pack = dataclasses.replace(
          pack,
          faces=pack.faces[keep_tri],
          tri_uvs=pack.tri_uvs[keep_tri],
          tri_mat_id=pack.tri_mat_id[keep_tri],
          tri_albedo=pack.tri_albedo[keep_tri],
          tri_roughness=pack.tri_roughness[keep_tri],
          tri_mat_code=pack.tri_mat_code[keep_tri],
      )
    pack.save(pack_p)
  print("COMPILED BARBERSHOP:", json.dumps(pack.summary()), flush=True)

  nw, nh, _ = pack.native_res
  aspect = float(nw) / max(float(nh), 1.0)
  orb_w = 512
  # Exact 0.5x scale of hero (1024x432) so aspect ratio & FOV match 1:1, aligned to 8px
  hero_w = 1024
  hero_h = max(256, ((int(round(hero_w / aspect)) + 15) // 16) * 16)
  orb_h = hero_h // 2
  assert orb_h % 8 == 0 and orb_h % 4 == 0, f"orb_h={orb_h} must be divisible by 8"
  print(f"BARBERSHOP GT RESOLUTIONS: native={nw}x{nh}, orbit={orb_w}x{orb_h}, hero={hero_w}x{hero_h}", flush=True)

  if not (args.skip_orbit and os.path.exists(orb_p)):
    render_cycles_gt_and_depth(
        pack, orb_p, width=orb_w, height=orb_h, samples=args.orbit_spp,
        n_views=args.orbit_views, phase_offset=0.0, include_hero=True,
        max_yaw=0.14, max_pitch=0.08, clean_noisy_mix=True,
    )
  if not (args.skip_hero and os.path.exists(hero_p)):
    render_cycles_gt_and_depth(
        pack, hero_p, width=hero_w, height=hero_h, samples=args.hero_spp,
        n_views=args.hero_views, hero_png_path=hero_png,
        phase_offset=0.37, include_hero=True, max_yaw=0.14, max_pitch=0.08, clean_noisy_mix=True,
    )


# ==============================================================================
# Part 2: TPU Wide-BVH-64 Stage-1 + IrradianceNRC3S_Demod_W128 + 1080p/4K Bench
# ==============================================================================

def _tpu_main() -> None:
  import flax.linen as nn
  import jax
  from jax.experimental.shard_map import shard_map
  import jax.numpy as jnp
  import optax
  from PIL import Image, ImageDraw

  # Support both w7 workspace and local repo paths cleanly
  here = os.path.dirname(os.path.abspath(__file__))
  w5_dir = os.path.abspath(os.path.join(here, "..", "w5_blender_1m_showcase"))
  if os.path.isdir(w5_dir) and w5_dir not in sys.path:
    sys.path.insert(0, w5_dir)

  from tpu_mjx_render.adapters.blend_compiler import ScenePack
  from gate import compute_msssim_batch, compute_psnr, compute_ssim_batch, luminance
  from tpu_mjx_render.roofline import inspect_hlo_ops
  from wide_bvh64_uv import (
      WideBVH64UVSceneSpec,
      build_scene_gbuffer_features,
      build_wide_bvh64_uv,
      pack_dev_arrays,
      render_wide_bvh64_uv_stage1,
  )

  def _hlo_audit(hlo_text: str) -> Dict[str, int]:
    info = inspect_hlo_ops(hlo_text)
    info["send_recv_ops"] = len(re.findall(r"\b(?:send|recv|send-done|recv-done)\(", hlo_text))
    return info

  def _ssim_loss(pred: jax.Array, target: jax.Array) -> jax.Array:
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu_x = jax.lax.reduce_window(pred, 0.0, jax.lax.add, (1, 5, 5, 1), (1, 1, 1, 1), "SAME") / 25.0
    mu_y = jax.lax.reduce_window(target, 0.0, jax.lax.add, (1, 5, 5, 1), (1, 1, 1, 1), "SAME") / 25.0
    s_xx = jax.lax.reduce_window(pred * pred, 0.0, jax.lax.add, (1, 5, 5, 1), (1, 1, 1, 1), "SAME") / 25.0 - mu_x ** 2
    s_yy = jax.lax.reduce_window(target * target, 0.0, jax.lax.add, (1, 5, 5, 1), (1, 1, 1, 1), "SAME") / 25.0 - mu_y ** 2
    s_xy = jax.lax.reduce_window(pred * target, 0.0, jax.lax.add, (1, 5, 5, 1), (1, 1, 1, 1), "SAME") / 25.0 - mu_x * mu_y
    num = (2.0 * mu_x * mu_y + c1) * (2.0 * s_xy + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (jnp.maximum(s_xx, 0.0) + jnp.maximum(s_yy, 0.0) + c2)
    return 1.0 - jnp.mean(num / jnp.maximum(den, 1e-8))

  def _grad_l1_loss(pred: jax.Array, target: jax.Array) -> jax.Array:
    dx_p = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    dx_t = target[:, :, 1:, :] - target[:, :, :-1, :]
    dy_p = pred[:, 1:, :, :] - pred[:, :-1, :, :]
    dy_t = target[:, 1:, :, :] - target[:, :-1, :, :]
    return 0.5 * (jnp.mean(jnp.abs(dx_p - dx_t)) + jnp.mean(jnp.abs(dy_p - dy_t)))

  class _ResBlock3S(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
      dt = jnp.bfloat16
      y = nn.Conv(self.features, (3, 3), padding="SAME", dtype=dt, param_dtype=jnp.float32)(x)
      y = nn.relu(y)
      y = nn.Conv(self.features, (3, 3), padding="SAME", dtype=dt, param_dtype=jnp.float32)(y)
      return nn.relu(x + y)

  class IrradianceNRC3ScaleDemod(nn.Module):
    """M2_demod W128 Albedo-factorized 3-scale TPU MXU BF16 Neural Relighter."""

    base_width: int = 128

    @nn.compact
    def __call__(self, x: jax.Array, train: bool = False) -> jax.Array:
      del train
      dt = jnp.bfloat16
      w1 = self.base_width
      w2 = self.base_width * 2
      w3 = self.base_width * 2

      h1 = nn.relu(nn.Conv(w1, (3, 3), padding="SAME", dtype=dt, param_dtype=jnp.float32)(x.astype(dt)))
      h1 = _ResBlock3S(w1)(h1)
      h1 = _ResBlock3S(w1)(h1)

      h2 = nn.relu(nn.Conv(w2, (3, 3), strides=(2, 2), padding="SAME", dtype=dt, param_dtype=jnp.float32)(h1))
      h2 = _ResBlock3S(w2)(h2)
      h2 = _ResBlock3S(w2)(h2)

      h3 = nn.relu(nn.Conv(w3, (3, 3), strides=(2, 2), padding="SAME", dtype=dt, param_dtype=jnp.float32)(h2))
      h3 = _ResBlock3S(w3)(h3)
      h3 = _ResBlock3S(w3)(h3)

      B, H, W, _ = x.shape
      h3_up = jax.image.resize(h3.astype(jnp.float32), (B, H // 2, W // 2, w3), method="bilinear").astype(dt)
      h2_m = nn.relu(nn.Conv(w2, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(jnp.concatenate([h2, h3_up], axis=-1)))
      h2_m = _ResBlock3S(w2)(h2_m)

      h2_up = jax.image.resize(h2_m.astype(jnp.float32), (B, H, W, w2), method="bilinear").astype(dt)
      h_out = nn.relu(nn.Conv(w1, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(jnp.concatenate([h1, h2_up], axis=-1)))
      h_out = _ResBlock3S(w1)(h_out)

      out_in = jnp.concatenate([h_out, x.astype(dt)], axis=-1)
      out_mid = nn.relu(nn.Conv(w1, (3, 3), padding="SAME", dtype=dt, param_dtype=jnp.float32)(out_in))
      out_mid = nn.relu(nn.Conv(w1, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(out_mid))
      irrad_spec = nn.Conv(6, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(out_mid).astype(jnp.float32)
      alb = x[..., 3:6]
      return alb * irrad_spec[..., :3] + irrad_spec[..., 3:6]

  def _train_demod_w128(
      feats_tr: np.ndarray,
      s1_tr: np.ndarray,
      gt_tr: np.ndarray,
      tr_idx: np.ndarray,
      cal_idx: np.ndarray,
      te_idx: np.ndarray,
      feats_hero: np.ndarray,
      s1_hero: np.ndarray,
      gt_hero: np.ndarray,
      epochs: int = 360,
      batch_size: int = 4,
      lr: float = 2.0e-3,
      base_width: int = 128,
  ):
    _, H, W, C = feats_tr.shape
    model = IrradianceNRC3ScaleDemod(base_width=base_width)
    params = model.init(jax.random.PRNGKey(0), jnp.asarray(feats_tr[:1]))["params"]
    ema_params = params

    steps_per_epoch = max(1, len(tr_idx) // batch_size)
    total_steps = epochs * steps_per_epoch
    sched = optax.warmup_cosine_decay_schedule(
        init_value=lr * 0.1, peak_value=lr,
        warmup_steps=max(8, total_steps // 12), decay_steps=total_steps, end_value=lr * 0.03,
    )
    tx = optax.chain(optax.clip_by_global_norm(2.0), optax.adamw(sched, weight_decay=2e-5))
    opt_state = tx.init(params)

    x_tr = jnp.asarray(feats_tr[tr_idx])
    b_tr = jnp.asarray(s1_tr[tr_idx])
    y_tr = jnp.asarray(gt_tr[tr_idx])

    @jax.jit
    def train_step(p, ema_p, opt_st, idx_s, crop_key, use_up2x):
      xb, bb, yb = x_tr[idx_s], b_tr[idx_s], y_tr[idx_s]

      def _upsample_crop(vals):
        x_i, b_i, y_i = vals
        B_i = x_i.shape[0]
        x_up = jax.image.resize(x_i, (B_i, 2 * H, 2 * W, C), method="bilinear")
        b_up = jax.image.resize(b_i, (B_i, 2 * H, 2 * W, 3), method="bilinear")
        y_up = jax.image.resize(y_i, (B_i, 2 * H, 2 * W, 3), method="bilinear")
        ky, kx = jax.random.split(crop_key)
        oy = jax.random.randint(ky, (), 0, max(1, H // 4)) * 4
        ox = jax.random.randint(kx, (), 0, max(1, W // 4)) * 4
        return (
            jax.lax.dynamic_slice(x_up, (0, oy, ox, 0), (B_i, H, W, C)),
            jax.lax.dynamic_slice(b_up, (0, oy, ox, 0), (B_i, H, W, 3)),
            jax.lax.dynamic_slice(y_up, (0, oy, ox, 0), (B_i, H, W, 3)),
        )

      xb, bb, yb = jax.lax.cond(use_up2x, _upsample_crop, lambda v: v, (xb, bb, yb))

      def loss_fn(pr):
        raw = bb + model.apply({"params": pr}, xb, train=True)
        pred = jnp.clip(raw, 0.0, 1.0)
        return (
            0.50 * jnp.mean((raw - yb) ** 2)
            + 0.20 * jnp.mean(jnp.abs(raw - yb))
            + 0.15 * _ssim_loss(pred, yb)
            + 0.15 * _grad_l1_loss(pred, yb)
        )

      loss, grads = jax.value_and_grad(loss_fn)(p)
      ups, opt_st = tx.update(grads, opt_st, p)
      p = optax.apply_updates(p, ups)
      ema_p = jax.tree_util.tree_map(lambda e, n: 0.99 * e + 0.01 * n, ema_p, p)
      return p, ema_p, opt_st, loss

    @jax.jit
    def infer_fn(p, xf, bf):
      return jnp.clip(bf + model.apply({"params": p}, xf, train=False), 0.0, 1.0)

    _, _, _, _l = train_step(params, ema_params, opt_state, jnp.arange(batch_size), jax.random.PRNGKey(1), jnp.bool_(False))
    jax.block_until_ready(_l)

    t0 = time.time()
    np_rng = np.random.default_rng(42)
    step_i = 0
    for ep in range(epochs):
      perm = np_rng.permutation(len(tr_idx))
      el = 0.0
      for s in range(steps_per_epoch):
        sl = perm[s * batch_size : (s + 1) * batch_size]
        up2x = (step_i % 2 == 1)
        params, ema_params, opt_state, loss = train_step(
            params, ema_params, opt_state, jnp.asarray(sl), jax.random.PRNGKey(step_i + 100), jnp.bool_(up2x)
        )
        el += float(loss)
        step_i += 1
      if (ep + 1) % 90 == 0 or ep + 1 == epochs:
        print(f"[barbershop] ep {ep+1}/{epochs} loss={el/steps_per_epoch:.5f} ({time.time()-t0:.1f}s)", flush=True)
    train_s = time.time() - t0

    pred_cal = np.asarray(infer_fn(ema_params, jnp.asarray(feats_tr[cal_idx]), jnp.asarray(s1_tr[cal_idx])))
    cal_scale = float(np.mean(luminance(gt_tr[cal_idx])) / max(np.mean(luminance(pred_cal)), 1e-6))

    pred_te = np.clip(np.asarray(infer_fn(ema_params, jnp.asarray(feats_tr[te_idx]), jnp.asarray(s1_tr[te_idx]))) * cal_scale, 0.0, 1.0)
    y_te = gt_tr[te_idx]
    s1_te = s1_tr[te_idx]
    pred_hero = np.clip(np.asarray(infer_fn(ema_params, jnp.asarray(feats_hero), jnp.asarray(s1_hero))) * cal_scale, 0.0, 1.0)

    metrics = {
        "arch": f"IrradianceNRC3S_Demod_W{base_width}",
        "n_params": int(sum(x.size for x in jax.tree_util.tree_leaves(ema_params))),
        "epochs": epochs,
        "train_time_s": round(train_s, 2),
        "cal_scale": round(cal_scale, 6),
        "orbit_test": {
            "res": f"{W}x{H}",
            "n_views": int(len(te_idx)),
            "stage1_psnr_db": round(compute_psnr(s1_te, y_te), 4),
            "neural_psnr_db": round(compute_psnr(pred_te, y_te), 4),
            "psnr_gain_db": round(compute_psnr(pred_te, y_te) - compute_psnr(s1_te, y_te), 4),
            "neural_ssim": round(compute_ssim_batch(pred_te, y_te), 5),
            "neural_msssim": round(compute_msssim_batch(pred_te, y_te), 5),
            "lum_rel": round(float(np.mean(luminance(pred_te)) / max(np.mean(luminance(y_te)), 1e-8) - 1.0), 6),
        },
        "hero_official_plus_orbit": {
            "res": f"{gt_hero.shape[2]}x{gt_hero.shape[1]}",
            "n_views": int(len(gt_hero)),
            "view0_stage1_psnr_db": round(compute_psnr(s1_hero[:1], gt_hero[:1]), 4),
            "view0_neural_psnr_db": round(compute_psnr(pred_hero[:1], gt_hero[:1]), 4),
            "view0_ssim": round(compute_ssim_batch(pred_hero[:1], gt_hero[:1]), 5),
            "view0_msssim": round(compute_msssim_batch(pred_hero[:1], gt_hero[:1]), 5),
            "view0_lum_rel": round(float(np.mean(luminance(pred_hero[:1])) / max(np.mean(luminance(gt_hero[:1])), 1e-8) - 1.0), 6),
            "all_hero_neural_psnr_db": round(compute_psnr(pred_hero, gt_hero), 4),
            "all_hero_ssim": round(compute_ssim_batch(pred_hero, gt_hero), 5),
            "all_hero_msssim": round(compute_msssim_batch(pred_hero, gt_hero), 5),
            "all_hero_lum_rel": round(float(np.mean(luminance(pred_hero)) / max(np.mean(luminance(gt_hero)), 1e-8) - 1.0), 6),
        },
    }
    return model, metrics, ema_params, cal_scale, pred_hero

  def _benchmark_barbershop(
      spec_base: WideBVH64UVSceneSpec,
      cam_xpos: np.ndarray,
      cam_xmat: np.ndarray,
      res_w: int,
      res_h: int,
      batch_size: int,
      n_chips: int,
      neural_model=None,
      neural_params=None,
      cal_scale: float = 1.0,
      n_warmup: int = 2,
      n_steps: int = 5,
  ) -> Dict[str, Any]:
    spec_r = dataclasses.replace(spec_base, height=res_h, width=res_w)
    devs = jax.devices()[:n_chips]
    idx = np.arange(batch_size) % len(cam_xpos)
    cxp = cam_xpos[idx].astype(np.float32)
    cxm = cam_xmat[idx].astype(np.float32)
    sc_ctr = tuple(float(x) for x in spec_r.scene_center)
    sc_scl = tuple(float(x) for x in spec_r.scene_scale)
    zs = float(spec_r.z_scale)

    if n_chips == 1:
      dev_arrs = pack_dev_arrays(spec_r, devs[0])
      cxp_d = jax.device_put(cxp, devs[0])
      cxm_d = jax.device_put(cxm, devs[0])
      np_d = jax.tree_util.tree_map(lambda x: jax.device_put(x, devs[0]), neural_params) if neural_params is not None else None

      @jax.jit
      def run_fn(da, cp, cm, npr):
        gb, rgh, mc = render_wide_bvh64_uv_stage1(spec_r, da, cp, cm, interpret=False)
        if neural_model is not None and npr is not None:
          ft = build_scene_gbuffer_features(gb, rgh, mc, sc_ctr, sc_scl, zs)
          delta = neural_model.apply({"params": npr}, ft, train=False)
          rgb = jnp.clip((gb.rgb + delta) * cal_scale, 0.0, 1.0)
        else:
          rgb = gb.rgb
        return rgb, gb.z_depth

      hlo_info = _hlo_audit(run_fn.lower(dev_arrs, cxp_d, cxm_d, np_d).as_text())
      assert hlo_info["host_callbacks"] == 0 and hlo_info["send_recv_ops"] == 0, hlo_info

      for _ in range(n_warmup):
        r, z = run_fn(dev_arrs, cxp_d, cxm_d, np_d)
      jax.block_until_ready((r, z))
      t0 = time.perf_counter()
      for _ in range(n_steps):
        r, z = run_fn(dev_arrs, cxp_d, cxm_d, np_d)
      jax.block_until_ready((r, z))
      dt = (time.perf_counter() - t0) / n_steps
    else:
      assert batch_size % n_chips == 0
      mesh = jax.sharding.Mesh(np.array(devs), ("dp",))
      P = jax.sharding.PartitionSpec
      sh_dp = jax.sharding.NamedSharding(mesh, P("dp"))
      rep = jax.sharding.NamedSharding(mesh, P())
      dev_arrs = pack_dev_arrays(spec_r, rep)
      cxp_d = jax.device_put(cxp, sh_dp)
      cxm_d = jax.device_put(cxm, sh_dp)
      np_d = jax.tree_util.tree_map(lambda x: jax.device_put(x, rep), neural_params) if neural_params is not None else None
      sm_kw = {"check_rep": False} if "check_rep" in inspect.signature(shard_map).parameters else {"check_vma": False}

      def _local(da, cp, cm, npr):
        gb, rgh, mc = render_wide_bvh64_uv_stage1(spec_r, da, cp, cm, interpret=False)
        if neural_model is not None and npr is not None:
          ft = build_scene_gbuffer_features(gb, rgh, mc, sc_ctr, sc_scl, zs)
          delta = neural_model.apply({"params": npr}, ft, train=False)
          rgb = jnp.clip((gb.rgb + delta) * cal_scale, 0.0, 1.0)
        else:
          rgb = gb.rgb
        return rgb, gb.z_depth

      _sm = shard_map(
          _local, mesh=mesh,
          in_specs=(P(), P("dp"), P("dp"), P()),
          out_specs=(P("dp"), P("dp")),
          **sm_kw,
      )

      @jax.jit
      def run_fn(da, cp, cm, npr):
        return _sm(da, cp, cm, npr)

      hlo_info = _hlo_audit(run_fn.lower(dev_arrs, cxp_d, cxm_d, np_d).as_text())
      assert hlo_info["host_callbacks"] == 0 and hlo_info["send_recv_ops"] == 0, hlo_info

      for _ in range(n_warmup):
        r, z = run_fn(dev_arrs, cxp_d, cxm_d, np_d)
      jax.block_until_ready((r, z))
      t0 = time.perf_counter()
      for _ in range(n_steps):
        r, z = run_fn(dev_arrs, cxp_d, cxm_d, np_d)
      jax.block_until_ready((r, z))
      dt = (time.perf_counter() - t0) / n_steps

    img_s = batch_size / max(dt, 1e-9)
    return {
        "resolution": f"{res_w}x{res_h}",
        "batch_size": batch_size,
        "n_chips": n_chips,
        "stage2": neural_model is not None,
        "dt_ms": round(dt * 1e3, 3),
        "img_per_sec": round(img_s, 3),
        "mpix_per_sec": round((batch_size * res_w * res_h) / max(dt, 1e-9) / 1e6, 2),
        "host_callbacks": hlo_info["host_callbacks"],
        "send_recv_ops": hlo_info["send_recv_ops"],
    }

  def _colormap_rgb(val01: np.ndarray, kind: str) -> np.ndarray:
    x = np.clip(val01, 0.0, 1.0)[..., None]
    if kind == "inferno":
      pts = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
      cols = np.array([
          [0.001, 0.000, 0.014],
          [0.341, 0.062, 0.429],
          [0.735, 0.216, 0.330],
          [0.978, 0.557, 0.035],
          [0.988, 0.998, 0.645],
      ], dtype=np.float32)
    else:
      pts = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
      cols = np.array([
          [0.267, 0.005, 0.329],
          [0.229, 0.322, 0.545],
          [0.128, 0.567, 0.551],
          [0.369, 0.789, 0.383],
          [0.993, 0.906, 0.144],
      ], dtype=np.float32)
    out = np.empty((*val01.shape, 3), dtype=np.float32)
    flat = x.ravel()
    for c in range(3):
      out[..., c] = np.interp(flat, pts, cols[:, c]).reshape(val01.shape)
    return out

  def _make_cinema_figure(
      s1: np.ndarray,
      s2: np.ndarray,
      gt: np.ndarray,
      zd: np.ndarray,
      results: Dict[str, Any],
      out_png: str,
  ) -> None:
    H_img, W_img, _ = s2.shape
    pw = 680
    ph = max(280, int(round(pw * H_img / W_img)))
    pad, hdr, top_banner = 14, 34, 54
    n_cols, n_rows = 3, 2
    cw = n_cols * pw + (n_cols + 1) * pad
    ch = top_banner + n_rows * (ph + hdr) + (n_rows + 1) * pad
    canvas = Image.new("RGB", (cw, ch), (13, 15, 20))
    draw = ImageDraw.Draw(canvas)

    meta = results["scene_meta"]
    nr = results["neural_relighting"]
    hr = nr["hero_official_plus_orbit"]
    orb = nr["orbit_test"]
    par = results["parity"]
    b4_1080 = next((b for b in results["benchmarks"] if b["resolution"] == "1920x1080" and b["stage2"] and b["n_chips"] == 4), results["benchmarks"][-1])
    b4_4k = next((b for b in results["benchmarks"] if b["resolution"] == "3840x2160" and b["stage2"] and b["n_chips"] == 4), results["benchmarks"][-1])

    banner = (
        f"Blender Studio 'Barbershop Interior' ({meta['tris']/1e6:.2f}M tris, {meta['n_textures']}x{meta['tex_res']} UV, {meta['n_lights']} lights) | "
        f"TPU v5p S2 ({nr['arch']}): Hero {hr['view0_neural_psnr_db']:.2f} dB / SSIM {hr['view0_ssim']:.4f} / Orbit {orb['neural_psnr_db']:.2f} dB | "
        f"Z-MAE {par['hero_z_mae_m']*1e3:.2f} mm | 4c 1080p: {b4_1080['img_per_sec']:.2f} img/s | 4c 4K: {b4_4k['img_per_sec']:.2f} img/s | 0 callbacks"
    )
    draw.rectangle([0, 0, cw, top_banner], fill=(22, 26, 36))
    draw.text((pad, 19), banner, fill=(242, 247, 255))

    cy0, cy1 = int(0.25 * H_img), int(0.75 * H_img)
    cx0, cx1 = int(0.25 * W_img), int(0.75 * W_img)
    s2_c = s2[cy0:cy1, cx0:cx1]
    gt_c = gt[cy0:cy1, cx0:cx1]
    cw_c = s2_c.shape[1]
    split_crop = np.concatenate([s2_c[:, : cw_c // 2], gt_c[:, cw_c // 2 :]], axis=1)
    split_crop[:, max(0, cw_c // 2 - 2) : min(cw_c, cw_c // 2 + 2)] = np.array([0.2, 0.85, 1.0], np.float32)

    err5 = _colormap_rgb(np.mean(np.clip(np.abs(s2 - gt) * 5.0, 0.0, 1.0), axis=-1), "inferno")
    fg = zd > 0.0
    if np.any(fg):
      z_lo, z_hi = np.percentile(zd[fg], [2, 98])
      zd_norm = np.where(fg, 1.0 - np.clip((zd - z_lo) / max(z_hi - z_lo, 0.5), 0.0, 1.0), 0.0)
    else:
      zd_norm = np.zeros_like(zd)
    zd_vis = _colormap_rgb(zd_norm, "viridis")

    panels = [
        (s1, f"TPU v5p Wide-BVH-64 Stage-1 G-Buffer ({hr['view0_stage1_psnr_db']:.2f} dB)"),
        (s2, f"TPU v5p Stage-2 Neural Relit ({hr['view0_neural_psnr_db']:.2f} dB, SSIM {hr['view0_ssim']:.4f}, Lum {hr['view0_lum_rel']*100:+.2f}%)"),
        (gt, f"Cycles 5.3.0 GT ({gt.shape[1]}x{gt.shape[0]}, Native Color Management)"),
        (split_crop, "2x Zoom Detail Crop: Left = TPU v5p S2 | Right = Cycles 5.3.0 GT"),
        (err5, f"Absolute Error x5 (Inferno) | MS-SSIM {hr['view0_msssim']:.4f}"),
        (zd_vis, f"Exact Wide-BVH-64 Z-Depth | Z-MAE {par['hero_z_mae_m']*1e3:.2f} mm, FG {par['hero_fg_match']*100:.2f}%"),
    ]

    for idx, (img, title) in enumerate(panels):
      r_i, c_i = idx // n_cols, idx % n_cols
      x0 = pad + c_i * (pw + pad)
      y0 = top_banner + pad + r_i * (ph + hdr + pad)
      u8 = (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
      pil_im = Image.fromarray(u8).resize((pw, ph), Image.Resampling.BILINEAR)
      canvas.paste(pil_im, (x0, y0 + hdr))
      draw.text((x0 + 4, y0 + 10), title, fill=(235, 242, 250))

    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    canvas.save(out_png)
    print("SAVED BARBERSHOP CINEMA PNG:", out_png, canvas.size, flush=True)

  ap = argparse.ArgumentParser()
  ap.add_argument("--data_dir", default="/dev/shm/zaychik-tpu-mjx-w7/barbershop")
  ap.add_argument("--epochs", type=int, default=360)
  ap.add_argument("--base_width", type=int, default=128)
  ap.add_argument("--out_json", default="/dev/shm/zaychik-tpu-mjx-w7/out/metrics/v3_w7_barbershop_cinema.json")
  ap.add_argument("--out_png", default="/dev/shm/zaychik-tpu-mjx-w7/out/figures/v3_w7_barbershop_cinema.png")
  args = ap.parse_args()

  pack = ScenePack.load(os.path.join(args.data_dir, "barbershop_pack.npz"))
  orb = np.load(os.path.join(args.data_dir, "barbershop_gt_orbit.npz"))
  hero = np.load(os.path.join(args.data_dir, "barbershop_gt_hero.npz"))

  tr_w, tr_h = int(orb["width"]), int(orb["height"])
  hr_w, hr_h = int(hero["width"]), int(hero["height"])

  spec_tr = build_wide_bvh64_uv(pack, height=tr_h, width=tr_w, use_shadows=True)
  spec_hr = dataclasses.replace(spec_tr, height=hr_h, width=hr_w)
  sc_ctr = tuple(float(x) for x in spec_tr.scene_center)
  sc_scl = tuple(float(x) for x in spec_tr.scene_scale)
  zs = float(spec_tr.z_scale)

  dev0 = jax.devices()[0]
  dev_arrs = pack_dev_arrays(spec_tr, dev0)

  @jax.jit
  def _render_gb_tr(da, cp, cm):
    gb, rgh, mc = render_wide_bvh64_uv_stage1(spec_tr, da, cp, cm, interpret=False)
    ft = build_scene_gbuffer_features(gb, rgh, mc, sc_ctr, sc_scl, zs)
    return gb.rgb, gb.z_depth, ft

  @jax.jit
  def _render_gb_hr(da, cp, cm):
    gb, rgh, mc = render_wide_bvh64_uv_stage1(spec_hr, da, cp, cm, interpret=False)
    ft = build_scene_gbuffer_features(gb, rgh, mc, sc_ctr, sc_scl, zs)
    return gb.rgb, gb.z_depth, ft

  hlo_tr = _hlo_audit(_render_gb_tr.lower(dev_arrs, jnp.asarray(orb["cam_xpos"][:4]), jnp.asarray(orb["cam_xmat"][:4])).as_text())
  assert hlo_tr["host_callbacks"] == 0 and hlo_tr["send_recv_ops"] == 0, hlo_tr

  N_orb = len(orb["rgb"])
  s1_tr_l, ft_tr_l = [], []
  for st in range(0, N_orb, 4):
    en = min(N_orb, st + 4)
    r, _, f = _render_gb_tr(dev_arrs, jnp.asarray(orb["cam_xpos"][st:en]), jnp.asarray(orb["cam_xmat"][st:en]))
    s1_tr_l.append(np.asarray(r, np.float32))
    ft_tr_l.append(np.asarray(f, np.float32))
  s1_tr = np.concatenate(s1_tr_l, axis=0)
  ft_tr = np.concatenate(ft_tr_l, axis=0)

  r_h, z_h, f_h = _render_gb_hr(dev_arrs, jnp.asarray(hero["cam_xpos"]), jnp.asarray(hero["cam_xmat"]))
  s1_hr = np.asarray(r_h, np.float32)
  z_hr = np.asarray(z_h, np.float32)
  ft_hr = np.asarray(f_h, np.float32)

  cy_z = np.asarray(hero["frame0_z_depth"], np.float32)
  tp_z = z_hr[0]
  cy_fg = cy_z < 500.0
  tp_fg = tp_z > 0.0
  both_fg = cy_fg & tp_fg
  z_mae_m = float(np.mean(np.abs(cy_z[both_fg] - tp_z[both_fg]))) if np.any(both_fg) else -1.0
  z_med_m = float(np.median(np.abs(cy_z[both_fg] - tp_z[both_fg]))) if np.any(both_fg) else -1.0
  fg_match = float(np.mean(cy_fg == tp_fg))
  print(f"[barbershop] Hero {hr_w}x{hr_h} Z-MAE={z_mae_m*1e3:.3f} mm (Z-Med={z_med_m*1e3:.3f} mm), fg_match={fg_match:.5f}, hlo={hlo_tr}", flush=True)

  gt_tr = orb["rgb"].astype(np.float32) / 255.0
  gt_hr = hero["rgb"].astype(np.float32) / 255.0

  all_idx = np.arange(N_orb)
  if N_orb >= 32:
    cal_idx = np.array([3, 11, 19, 27], dtype=np.int32)
    te_idx = np.array([5, 13, 21, 29], dtype=np.int32)
  else:
    cal_idx = np.array([i for i in (2, 8, 14, 20) if i < N_orb], dtype=np.int32)
    te_idx = np.array([i for i in (4, 10, 16, 22) if i < N_orb], dtype=np.int32)
  tr_idx = np.array([i for i in all_idx if i not in set(cal_idx.tolist() + te_idx.tolist())], dtype=np.int32)

  mod, met, ema_params, cal_scale, pred_hr = _train_demod_w128(
      ft_tr, s1_tr, gt_tr, tr_idx, cal_idx, te_idx, ft_hr, s1_hr, gt_hr,
      epochs=args.epochs, batch_size=4, base_width=args.base_width,
  )
  print("[barbershop] EVAL:", json.dumps(met, indent=2), flush=True)

  n_devs = len(jax.devices())
  benches = []
  for nc in ([1, 4] if n_devs >= 4 else [1]):
    bs = 4 if nc == 4 else 1
    benches.append(_benchmark_barbershop(spec_tr, hero["cam_xpos"], hero["cam_xmat"], 512, 512, bs, nc, None, None, 1.0))
    benches.append(_benchmark_barbershop(spec_tr, hero["cam_xpos"], hero["cam_xmat"], 512, 512, bs, nc, mod, ema_params, cal_scale))

  if n_devs >= 4:
    for rw, rh, n_st in [(1024, 768, 5), (1920, 1080, 5), (3840, 2160, 3)]:
      benches.append(_benchmark_barbershop(spec_tr, hero["cam_xpos"], hero["cam_xmat"], rw, rh, 4, 4, None, None, 1.0, n_steps=n_st))
      benches.append(_benchmark_barbershop(spec_tr, hero["cam_xpos"], hero["cam_xmat"], rw, rh, 4, 4, mod, ema_params, cal_scale, n_steps=n_st))

  results = {
      "experiment": "v3_w7_barbershop_cinema",
      "device_kind": jax.devices()[0].device_kind,
      "n_devices": n_devs,
      "scene_meta": pack.summary(),
      "parity": {
          "hero_z_mae_m": z_mae_m,
          "hero_z_median_m": z_med_m,
          "hero_fg_match": fg_match,
          "hlo": hlo_tr,
      },
      "neural_relighting": met,
      "benchmarks": benches,
  }
  os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
  with open(args.out_json, "w") as f:
    json.dump(results, f, indent=2)
  print("SAVED JSON:", args.out_json, flush=True)

  _make_cinema_figure(
      np.clip(s1_hr[0], 0.0, 1.0),
      np.clip(pred_hr[0], 0.0, 1.0),
      np.clip(gt_hr[0], 0.0, 1.0),
      z_hr[0],
      results,
      args.out_png,
  )


if __name__ == "__main__":
  if HAS_BPY:
    _blender_export_main()
  else:
    _tpu_main()
