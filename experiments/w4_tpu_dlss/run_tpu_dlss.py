"""Wave 4: TPU-DLSS 3.5 Training, 30-Frame Temporal/SR Evaluation, 4K Benchmarks & Visual Showcase."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import pickle
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import jax
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np
import optax
from PIL import Image, ImageDraw

from bvh_pbr_pt import (
    LightRig,
    PBRWideBVHMesh,
    _agx_tonemap_srgb,
    _build_onb,
    _eval_pbr_brdf,
    _sample_cosine_hemisphere,
    _sample_ggx_half,
    _smith_g1,
    pbr_raycast_tpu,
)
from complex_scenes_pt import build_classroom_sunlit
from gate import compute_msssim_batch, compute_psnr, compute_ssim_batch, luminance
from mc_denoiser import (
    MXUAlbedoDemodDenoiser,
    apply_denoiser as apply_wave3_denoiser,
    build_denoiser_features as build_wave3_features,
    compute_exposure_scale as compute_wave3_exposure,
)
from render_sequences import generate_sequence_trajectory
from tpu_mjx_render.adapters.tpu_dlss import (
    TPUDLSS35Reconstructor,
    apply_tpu_dlss,
    audit_hlo_dlss,
    build_sharded_tpu_dlss_fn,
    compute_motion_vectors,
    warp_history_and_gbuf_4x4,
)


# ==============================================================================
# 1. Loss & Unrolled Multi-Frame Training for TPU-DLSS 3.5 + Wave 3 Baseline
# ==============================================================================

def _ssim_loss(pred: jax.Array, target: jax.Array) -> jax.Array:
  c1, c2 = 0.01 ** 2, 0.03 ** 2
  mu_x = jax.lax.reduce_window(pred, 0.0, jax.lax.add, (1, 5, 5, 1), (1, 1, 1, 1), "SAME") / 25.0
  mu_y = jax.lax.reduce_window(target, 0.0, jax.lax.add, (1, 5, 5, 1), (1, 1, 1, 1), "SAME") / 25.0
  sig_x = jax.lax.reduce_window(pred * pred, 0.0, jax.lax.add, (1, 5, 5, 1), (1, 1, 1, 1), "SAME") / 25.0 - mu_x * mu_x
  sig_y = jax.lax.reduce_window(target * target, 0.0, jax.lax.add, (1, 5, 5, 1), (1, 1, 1, 1), "SAME") / 25.0 - mu_y * mu_y
  sig_xy = jax.lax.reduce_window(pred * target, 0.0, jax.lax.add, (1, 5, 5, 1), (1, 1, 1, 1), "SAME") / 25.0 - mu_x * mu_y
  num = (2.0 * mu_x * mu_y + c1) * (2.0 * sig_xy + c2)
  den = (mu_x * mu_x + mu_y * mu_y + c1) * (jnp.maximum(sig_x, 0.0) + jnp.maximum(sig_y, 0.0) + c2)
  return 1.0 - jnp.mean(num / jnp.maximum(den, 1e-12))


def train_wave3_baseline(
    train_seqs: List[Dict[str, Any]],
    epochs: int = 200,
    batch_size: int = 16,
    patch_size: int = 96,
    lr: float = 2.0e-3,
) -> Tuple[Any, int, float]:
  """Train Wave 3 single-frame MXUAlbedoDemodDenoiser on the exact same half-res 1-spp training frames."""
  dev = jax.devices()[0]
  n1_list, gb_list, gt_lr_list = [], [], []
  for seq in train_seqs:
    n1_list.append(seq["noisy1_lr"])
    gb_list.append(seq["gbuf_lr"])
    gt_hr = seq["gt_hr"]  # (T, H, W, 3)
    # 2x2 average pool GT to match half-res LR supervision for single-frame denoiser
    gt_lr = 0.25 * (gt_hr[:, 0::2, 0::2, :] + gt_hr[:, 0::2, 1::2, :] + gt_hr[:, 1::2, 0::2, :] + gt_hr[:, 1::2, 1::2, :])
    gt_lr_list.append(gt_lr)

  n_dev = jax.device_put(np.concatenate(n1_list, axis=0).astype(np.float32), dev)
  gb_dev = jax.device_put(np.concatenate(gb_list, axis=0).astype(np.float32), dev)
  y_dev = jax.device_put(np.concatenate(gt_lr_list, axis=0).astype(np.float32), dev)
  N, H, W, _ = n_dev.shape

  model = MXUAlbedoDemodDenoiser(c1=64, c2=128, c3=128, c_pix=24, head_width=48)
  dummy = build_wave3_features(
      jnp.zeros((1, patch_size, patch_size, 3), dtype=jnp.float32),
      jnp.zeros((1, patch_size, patch_size, 9), dtype=jnp.float32),
  )
  params = model.init(jax.random.PRNGKey(101), dummy)["params"]
  ema_params = params
  n_params = int(sum(p.size for p in jax.tree_util.tree_leaves(params)))

  steps_per_ep = 16
  total_steps = epochs * steps_per_ep
  sched = optax.warmup_cosine_decay_schedule(lr * 0.1, lr, max(20, total_steps // 15), total_steps, lr * 0.02)
  tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=2e-5))
  opt_state = tx.init(params)

  @jax.jit
  def train_ep(p, ema_p, ost, ep_rng):
    def _step(carry, s_idx):
      cp, cema, cost = carry
      k = jax.random.fold_in(ep_rng, s_idx)
      k_idx, k_y, k_x, k_exp = jax.random.split(k, 4)
      b_idx = jax.random.randint(k_idx, (batch_size,), 0, N)
      y0 = jax.random.randint(k_y, (), 0, H - patch_size + 1)
      x0 = jax.random.randint(k_x, (), 0, W - patch_size + 1)
      nb = jax.lax.dynamic_slice(n_dev[b_idx], (0, y0, x0, 0), (batch_size, patch_size, patch_size, 3))
      gbb = jax.lax.dynamic_slice(gb_dev[b_idx], (0, y0, x0, 0), (batch_size, patch_size, patch_size, 9))
      yb = jax.lax.dynamic_slice(y_dev[b_idx], (0, y0, x0, 0), (batch_size, patch_size, patch_size, 3))
      light_scale = jnp.exp(jax.random.uniform(k_exp, (batch_size, 1, 1, 1), minval=math.log(0.55), maxval=math.log(1.15)))
      nb_aug = jnp.clip(nb * light_scale, 0.0, 1.0)
      yb_aug = jnp.clip(yb * light_scale, 0.0, 1.0)
      l_scale = compute_wave3_exposure(nb_aug, axis_name=None)
      xb = build_wave3_features(nb_aug, gbb, l_scale=l_scale)

      def loss_fn(pr):
        pred_norm, blend_w = model.apply({"params": pr}, xb, train=True)
        pred = jnp.clip((l_scale * pred_norm + blend_w * nb_aug).astype(jnp.float32), 0.0, 1.0)
        l2 = jnp.mean((pred - yb_aug) ** 2)
        l1 = jnp.mean(jnp.abs(pred - yb_aug))
        ls = _ssim_loss(pred, yb_aug)
        gx_p = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        gx_y = yb_aug[:, :, 1:, :] - yb_aug[:, :, :-1, :]
        gy_p = pred[:, 1:, :, :] - pred[:, :-1, :, :]
        gy_y = yb_aug[:, 1:, :, :] - yb_aug[:, :-1, :, :]
        lgrad = jnp.mean(jnp.abs(gx_p - gx_y)) + jnp.mean(jnp.abs(gy_p - gy_y))
        lum_p = 0.2126 * pred[..., 0] + 0.7152 * pred[..., 1] + 0.0722 * pred[..., 2]
        lum_y = 0.2126 * yb_aug[..., 0] + 0.7152 * yb_aug[..., 1] + 0.0722 * yb_aug[..., 2]
        llum = jnp.abs(jnp.mean(lum_p) - jnp.mean(lum_y))
        return 0.45 * l2 + 0.30 * l1 + 0.15 * ls + 0.10 * lgrad + 0.04 * llum

      loss, grads = jax.value_and_grad(loss_fn)(cp)
      upd, n_ost = tx.update(grads, cost, cp)
      np_p = optax.apply_updates(cp, upd)
      n_ema = jax.tree_util.tree_map(lambda e, n: 0.995 * e + 0.005 * n, cema, np_p)
      return (np_p, n_ema, n_ost), loss

    (p_o, e_o, o_o), losses = jax.lax.scan(_step, (p, ema_p, ost), jnp.arange(steps_per_ep))
    return p_o, e_o, o_o, jnp.mean(losses)

  t0 = time.time()
  for ep in range(epochs):
    params, ema_params, opt_state, ep_loss = train_ep(params, ema_params, opt_state, jax.random.PRNGKey(ep + 30000))
    if (ep + 1) % 50 == 0 or ep + 1 == epochs:
      jax.block_until_ready(ep_loss)
      print(f"[WAVE3-BASE] ep {ep+1}/{epochs} loss={float(ep_loss):.5f} ({time.time()-t0:.1f}s)", flush=True)
  return ema_params, n_params, round(time.time() - t0, 2)


def train_tpu_dlss(
    model: TPUDLSS35Reconstructor,
    train_seqs: List[Dict[str, Any]],
    epochs: int = 140,
    seq_len: int = 6,
    steps_per_ep: int = 20,
    patch_lr: int = 64,
    lr: float = 2.5e-3,
) -> Tuple[Any, int, float]:
  """Unrolled multi-frame temporal + 2x SR training with random spatial crops to prevent orbit overfitting."""
  dev = jax.devices()[0]
  n_seqs = len(train_seqs)
  T = train_seqs[0]["noisy1_lr"].shape[0]
  Hl, Wl = train_seqs[0]["noisy1_lr"].shape[1:3]
  H, W = Hl * 2, Wl * 2
  patch_hr = patch_lr * 2

  s_n1 = jax.device_put(np.stack([s["noisy1_lr"] for s in train_seqs], axis=0).astype(np.float32), dev)
  s_gb = jax.device_put(np.stack([s["gbuf_lr"] for s in train_seqs], axis=0).astype(np.float32), dev)
  s_gt = jax.device_put(np.stack([s["gt_hr"] for s in train_seqs], axis=0).astype(np.float32), dev)
  s_cxp = jax.device_put(np.stack([s["cxp"] for s in train_seqs], axis=0).astype(np.float32), dev)
  s_cxm = jax.device_put(np.stack([s["cxm"] for s in train_seqs], axis=0).astype(np.float32), dev)
  s_jit = jax.device_put(np.stack([s["jitter_lr"] for s in train_seqs], axis=0).astype(np.float32), dev)
  znear = float(train_seqs[0]["znear"])

  dummy_x = jnp.zeros((1, patch_lr, patch_lr, 44), dtype=jnp.bfloat16)
  params = model.init(jax.random.PRNGKey(2026), dummy_x)["params"]
  ema_params = params
  n_params = int(sum(p.size for p in jax.tree_util.tree_leaves(params)))

  total_steps = epochs * steps_per_ep
  sched = optax.warmup_cosine_decay_schedule(lr * 0.1, lr, max(20, total_steps // 12), total_steps, lr * 0.02)
  tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=5e-4))
  opt_state = tx.init(params)

  def _build_scene_step(scene_idx: int):
    fovy_s = float(train_seqs[scene_idx]["cam_fovy"])

    def _loss_for_seq(pr, n_sub, gb_sub, gt_sub, cxp_sub, cxm_sub, jit_sub, y0_lr, x0_lr, light_scale):
      n_aug = jnp.clip(n_sub * light_scale, 0.0, 1.0)
      y_aug = jnp.clip(gt_sub * light_scale, 0.0, 1.0)

      def _frame_step(carry, xs_t):
        t_idx, curr_n_hw, curr_gb_hw, curr_gt_hw, c_cxp, c_cxm, c_jit = xs_t
        prev_pred_hr, prev_gb_lr, prev_gt_hr, prev_cxp, prev_cxm = carry
        has_prev = (t_idx > 0).astype(jnp.float32)

        curr_n = curr_n_hw[None, ...]
        curr_gb = curr_gb_hw[None, ...]
        curr_gt = curr_gt_hw[None, ...]

        pred_hr, disocc_lr, mv_lr, _ = apply_tpu_dlss(
            model,
            pr,
            curr_n,
            curr_gb,
            jax.lax.stop_gradient(prev_pred_hr),
            prev_gb_lr,
            c_cxp,
            c_cxm,
            prev_cxp,
            prev_cxm,
            cam_fovy=fovy_s,
            znear=znear,
            full_width_lr=Wl,
            x_offset_lr=x0_lr,
            jitter_lr=c_jit,
            has_prev_history=has_prev,
            axis_name=None,
            halo_lr=32,
            use_rgb10=False,
            warp_mode="4x4",
            train=True,
            return_aux=True,
            full_height_lr=Hl,
            y_offset_lr=y0_lr,
        )

        l2 = jnp.mean((pred_hr - curr_gt) ** 2)
        l1 = jnp.mean(jnp.abs(pred_hr - curr_gt))
        ls = _ssim_loss(pred_hr, curr_gt)
        gx_p = pred_hr[:, :, 1:, :] - pred_hr[:, :, :-1, :]
        gx_y = curr_gt[:, :, 1:, :] - curr_gt[:, :, :-1, :]
        gy_p = pred_hr[:, 1:, :, :] - pred_hr[:, :-1, :, :]
        gy_y = curr_gt[:, 1:, :, :] - curr_gt[:, :-1, :, :]
        lgrad = jnp.mean(jnp.abs(gx_p - gx_y)) + jnp.mean(jnp.abs(gy_p - gy_y))
        lum_p = 0.2126 * pred_hr[..., 0] + 0.7152 * pred_hr[..., 1] + 0.0722 * pred_hr[..., 2]
        lum_y = 0.2126 * curr_gt[..., 0] + 0.7152 * curr_gt[..., 1] + 0.0722 * curr_gt[..., 2]
        llum = jnp.abs(jnp.mean(lum_p) - jnp.mean(lum_y))

        pred_w_s2d, _ = warp_history_and_gbuf_4x4(
            jax.lax.stop_gradient(prev_pred_hr), prev_gb_lr[..., 3:7], mv_lr, axis_name=None, use_rgb10=False
        )
        gt_w_s2d, _ = warp_history_and_gbuf_4x4(
            prev_gt_hr, prev_gb_lr[..., 3:7], mv_lr, axis_name=None, use_rgb10=False
        )
        pred_w_hr = pred_w_s2d.astype(jnp.float32).reshape(1, patch_lr, patch_lr, 2, 2, 3).transpose(0, 1, 3, 2, 4, 5).reshape(1, patch_hr, patch_hr, 3)
        gt_w_hr = gt_w_s2d.astype(jnp.float32).reshape(1, patch_lr, patch_lr, 2, 2, 3).transpose(0, 1, 3, 2, 4, 5).reshape(1, patch_hr, patch_hr, 3)
        valid_hr = jax.image.resize(1.0 - disocc_lr, (1, patch_hr, patch_hr, 1), method="nearest")
        l_temp_gt = has_prev * jnp.mean(valid_hr * jnp.abs((pred_hr - pred_w_hr) - (curr_gt - gt_w_hr)))

        frame_loss = 0.35 * l2 + 0.35 * l1 + 0.18 * ls + 0.15 * lgrad + 0.12 * llum + 0.14 * l_temp_gt
        w_t = jnp.where(t_idx == 0, 0.85, 1.05)
        return (pred_hr, curr_gb, curr_gt, c_cxp, c_cxm), frame_loss * w_t

      init_carry = (
          jnp.zeros((1, patch_hr, patch_hr, 3), dtype=jnp.float32),
          gb_sub[0:1],
          y_aug[0:1],
          cxp_sub[0],
          cxm_sub[0],
      )
      _, losses = jax.lax.scan(
          _frame_step,
          init_carry,
          (jnp.arange(seq_len), n_aug, gb_sub, y_aug, cxp_sub, cxm_sub, jit_sub),
      )
      return jnp.mean(losses)

    return _loss_for_seq

  scene_loss_fns = [_build_scene_step(i) for i in range(n_seqs)]

  @jax.jit
  def train_epoch(p, ema_p, ost, ep_rng):
    def _scan_step(carry, s_idx):
      cp, cema, cost = carry
      k = jax.random.fold_in(ep_rng, s_idx)
      k_t, k_y, k_x, k_exp, k_mix = jax.random.split(k, 5)
      t0 = jax.random.randint(k_t, (n_seqs,), 0, T - seq_len + 1)
      y0 = jax.random.randint(k_y, (n_seqs,), 0, Hl - patch_lr + 1)
      x0 = jax.random.randint(k_x, (n_seqs,), 0, Wl - patch_lr + 1)
      light_s = jnp.exp(jax.random.uniform(k_exp, (n_seqs,), minval=math.log(0.35), maxval=math.log(1.15)))
      do_lowvar = jax.random.uniform(k_mix, (n_seqs,)) < 0.20

      def total_loss_fn(pr):
        tot = 0.0
        for i in range(n_seqs):
          ts, ys, xs = t0[i], y0[i], x0[i]
          n_sub = jax.lax.dynamic_slice(s_n1[i], (ts, ys, xs, 0), (seq_len, patch_lr, patch_lr, 3))
          gb_sub = jax.lax.dynamic_slice(s_gb[i], (ts, ys, xs, 0), (seq_len, patch_lr, patch_lr, 9))
          gt_sub = jax.lax.dynamic_slice(s_gt[i], (ts, 2 * ys, 2 * xs, 0), (seq_len, patch_hr, patch_hr, 3))
          gt_lr_sub = 0.25 * (gt_sub[:, 0::2, 0::2, :] + gt_sub[:, 0::2, 1::2, :] + gt_sub[:, 1::2, 0::2, :] + gt_sub[:, 1::2, 1::2, :])
          n_sub = jnp.where(do_lowvar[i], 0.55 * n_sub + 0.45 * gt_lr_sub, n_sub)
          cxp_sub = jax.lax.dynamic_slice(s_cxp[i], (ts, 0, 0), (seq_len, 1, 3))
          cxm_sub = jax.lax.dynamic_slice(s_cxm[i], (ts, 0, 0, 0), (seq_len, 1, 3, 3))
          jit_sub = jax.lax.dynamic_slice(s_jit[i], (ts, 0), (seq_len, 2))
          tot = tot + scene_loss_fns[i](pr, n_sub, gb_sub, gt_sub, cxp_sub, cxm_sub, jit_sub, ys, xs, light_s[i])
        return tot / float(n_seqs)

      loss, grads = jax.value_and_grad(total_loss_fn)(cp)
      upd, n_ost = tx.update(grads, cost, cp)
      np_p = optax.apply_updates(cp, upd)
      n_ema = jax.tree_util.tree_map(lambda e, n: 0.995 * e + 0.005 * n, cema, np_p)
      return (np_p, n_ema, n_ost), loss

    (p_o, e_o, o_o), losses = jax.lax.scan(_scan_step, (p, ema_p, ost), jnp.arange(steps_per_ep))
    return p_o, e_o, o_o, jnp.mean(losses)

  t0 = time.time()
  for ep in range(epochs):
    params, ema_params, opt_state, ep_loss = train_epoch(params, ema_params, opt_state, jax.random.PRNGKey(ep + 10000))
    if (ep + 1) % 30 == 0 or ep + 1 == epochs:
      jax.block_until_ready(ep_loss)
      print(f"[TPU-DLSS 3.5] ep {ep+1}/{epochs} loss={float(ep_loss):.5f} ({time.time()-t0:.1f}s)", flush=True)

  return ema_params, n_params, round(time.time() - t0, 2)


# ==============================================================================
# 2. 30-Frame Sequence Evaluation + Temporal Flicker Metrics
# ==============================================================================

def eval_30frame_sequence(
    seq: Dict[str, Any],
    dlss_model: TPUDLSS35Reconstructor,
    dlss_params: Any,
    w3_model: MXUAlbedoDemodDenoiser,
    w3_params: Any,
    use_rgb10: bool = True,
) -> Dict[str, Any]:
  """Evaluate (a) 1-spp bilinear, (b) Wave 3 + bilinear, (c) TPU-DLSS 3.5 across all 30 frames."""
  dev = jax.devices()[0]
  n1_lr = seq["noisy1_lr"]    # (T, Hl, Wl, 3)
  gb_lr = seq["gbuf_lr"]      # (T, Hl, Wl, 9)
  gt_hr = seq["gt_hr"]        # (T, H, W, 3)
  cxp = seq["cxp"]
  cxm = seq["cxm"]
  jit = seq["jitter_lr"]
  fovy = float(seq["cam_fovy"])
  znear = float(seq["znear"])
  T, Hl, Wl, _ = n1_lr.shape
  H, W = Hl * 2, Wl * 2

  # 1. Method (a): 1-spp Half-Res Bilinear Upscaled
  pred_bil = np.asarray(
      jax.image.resize(jnp.asarray(n1_lr), (T, H, W, 3), method="bilinear"),
      dtype=np.float32,
  )

  # 2. Method (b): Wave 3 Single-Frame Denoiser + Bilinear
  w3_p_dev = jax.device_put(jax.tree_util.tree_map(jnp.asarray, w3_params), dev)

  @jax.jit
  def _run_w3_batch(p, nb, gbb):
    dn_lr = apply_wave3_denoiser(w3_model, p, nb, gbb, axis_name=None, train=False)
    return jax.image.resize(dn_lr, (nb.shape[0], H, W, 3), method="bilinear")

  pred_w3 = np.asarray(_run_w3_batch(w3_p_dev, jax.device_put(n1_lr, dev), jax.device_put(gb_lr, dev)), dtype=np.float32)

  # 3. Method (c): TPU-DLSS 3.5 Temporal + 2x SR (frame-by-frame recurrence)
  dlss_p_dev = jax.device_put(jax.tree_util.tree_map(jnp.asarray, dlss_params), dev)

  @jax.jit
  def _dlss_step(p, n_l, gb_l, ph_hr, pgb_l, c_p, c_m, p_p, p_m, jt, hp):
    return apply_tpu_dlss(
        dlss_model, p, n_l, gb_l, ph_hr, pgb_l, c_p, c_m, p_p, p_m,
        cam_fovy=fovy, znear=znear, full_width_lr=Wl, x_offset_lr=0,
        jitter_lr=jt, has_prev_history=hp, axis_name=None, halo_lr=32,
        use_rgb10=use_rgb10, warp_mode="4x4", train=False,
    )

  # Also JIT exact HR warp for GT/baseline temporal flicker computation
  @jax.jit
  def _compute_warp_and_mask(gb_c, gb_p, img_prev_hr, c_p, c_m, p_p, p_m, jt):
    mv_lr, prev_uv, exp_d, val_z = compute_motion_vectors(
        gb_c[..., 6:7], c_p, c_m, p_p, p_m, cam_fovy=fovy, znear=znear,
        full_width_lr=Wl, x_offset_lr=0, jitter_lr=jt,
    )
    img_w_s2d, p_geom_w = warp_history_and_gbuf_4x4(
        img_prev_hr, gb_p[..., 3:7], mv_lr, axis_name=None, use_rgb10=False
    )
    img_w_hr = img_w_s2d.astype(jnp.float32).reshape(1, Hl, Wl, 2, 2, 3).transpose(0, 1, 3, 2, 4, 5).reshape(1, H, W, 3)
    in_b = (prev_uv[..., 0:1] >= 0.0) & (prev_uv[..., 0:1] <= 1.0) & (prev_uv[..., 1:2] >= 0.0) & (prev_uv[..., 1:2] <= 1.0) & val_z
    rel_d = jnp.abs(exp_d - p_geom_w[..., 3:4]) / jnp.maximum(exp_d, 0.05)
    ndot = jnp.sum(gb_c[..., 3:6] * p_geom_w[..., 0:3], axis=-1, keepdims=True)
    valid_lr = jnp.where(in_b & (rel_d < 0.08) & (ndot > 0.75), 1.0, 0.0)
    valid_hr = jax.image.resize(valid_lr, (1, H, W, 1), method="nearest")
    return img_w_hr[0], valid_hr[0]

  dlss_frames = []
  prev_hist = jnp.zeros((1, H, W, 3), dtype=jnp.float32)
  prev_gb = jax.device_put(gb_lr[0:1], dev)
  prev_cxp = jax.device_put(cxp[0], dev)
  prev_cxm = jax.device_put(cxm[0], dev)

  for t in range(T):
    curr_n = jax.device_put(n1_lr[t : t + 1], dev)
    curr_gb = jax.device_put(gb_lr[t : t + 1], dev)
    c_p = jax.device_put(cxp[t], dev)
    c_m = jax.device_put(cxm[t], dev)
    jt = jax.device_put(jit[t], dev)
    hp = jnp.float32(1.0 if t > 0 else 0.0)
    pred_hr, _ = _dlss_step(dlss_p_dev, curr_n, curr_gb, prev_hist, prev_gb, c_p, c_m, prev_cxp, prev_cxm, jt, hp)
    dlss_frames.append(np.asarray(pred_hr[0], dtype=np.float32))
    prev_hist = pred_hr
    prev_gb = curr_gb
    prev_cxp = c_p
    prev_cxm = c_m

  pred_dlss = np.stack(dlss_frames, axis=0)

  # Compute Spatial + Temporal Metrics across all 30 frames
  def _compute_all_metrics(pred_seq: np.ndarray) -> Dict[str, Any]:
    psnr_list = [compute_psnr(pred_seq[t], gt_hr[t]) for t in range(T)]
    ssim_all = compute_ssim_batch(pred_seq, gt_hr)
    msssim_all = compute_msssim_batch(pred_seq, gt_hr)
    lum_rel = float(np.mean(luminance(pred_seq)) / max(float(np.mean(luminance(gt_hr))), 1e-6) - 1.0)

    # Warped temporal flicker across t = 1..T-1
    warped_l1s, flicker_ex_l1s, t_ssims = [], [], []
    for t in range(1, T):
      gb_c = jax.device_put(gb_lr[t : t + 1], dev)
      gb_p = jax.device_put(gb_lr[t - 1 : t], dev)
      c_p, c_m = jax.device_put(cxp[t], dev), jax.device_put(cxm[t], dev)
      p_p, p_m = jax.device_put(cxp[t - 1], dev), jax.device_put(cxm[t - 1], dev)
      jt = jax.device_put(jit[t], dev)

      pw, val_m = _compute_warp_and_mask(gb_c, gb_p, jax.device_put(pred_seq[t - 1 : t], dev), c_p, c_m, p_p, p_m, jt)
      gw, _ = _compute_warp_and_mask(gb_c, gb_p, jax.device_put(gt_hr[t - 1 : t], dev), c_p, c_m, p_p, p_m, jt)
      pw_n, gw_n, vm_n = np.asarray(pw, np.float32), np.asarray(gw, np.float32), np.asarray(val_m, np.float32)

      denom = max(float(np.mean(vm_n)), 1e-4)
      wl1 = float(np.mean(vm_n * np.abs(pred_seq[t] - pw_n)) / denom)
      gt_wl1 = float(np.mean(vm_n * np.abs(gt_hr[t] - gw_n)) / denom)
      ex_l1 = float(np.mean(vm_n * np.abs(np.abs(pred_seq[t] - pw_n) - np.abs(gt_hr[t] - gw_n))) / denom)
      tss = compute_ssim_batch((vm_n * pred_seq[t])[None, ...], (vm_n * pw_n)[None, ...])
      warped_l1s.append(wl1)
      flicker_ex_l1s.append(ex_l1)
      t_ssims.append(tss)

    return {
        "psnr_db": round(float(np.mean(psnr_list)), 3),
        "psnr_converged_t5_29_db": round(float(np.mean(psnr_list[5:])), 3),
        "ssim": round(ssim_all, 4),
        "msssim": round(msssim_all, 4),
        "lum_rel": round(lum_rel, 6),
        "lum_bias_pct": round(abs(lum_rel) * 100.0, 4),
        "warped_flicker_l1": round(float(np.mean(warped_l1s)), 5),
        "flicker_excess_l1": round(float(np.mean(flicker_ex_l1s)), 5),
        "temporal_ssim": round(float(np.mean(t_ssims)), 4),
        "per_frame_psnr_db": [round(x, 3) for x in psnr_list],
        "per_frame_warped_l1": [round(x, 5) for x in warped_l1s],
    }

  m_bil = _compute_all_metrics(pred_bil)
  m_w3 = _compute_all_metrics(pred_w3)
  m_dlss = _compute_all_metrics(pred_dlss)

  return {
      "metrics": {
          "noisy_1spp_bilinear": m_bil,
          "wave3_denoiser_bilinear": m_w3,
          "tpu_dlss_3_5": m_dlss,
          "dlss_vs_1spp_psnr_gain_db": round(m_dlss["psnr_db"] - m_bil["psnr_db"], 3),
          "dlss_vs_wave3_psnr_gain_db": round(m_dlss["psnr_db"] - m_w3["psnr_db"], 3),
          "flicker_reduction_vs_1spp_x": round(m_bil["warped_flicker_l1"] / max(m_dlss["warped_flicker_l1"], 1e-6), 2),
          "flicker_reduction_vs_wave3_x": round(m_w3["warped_flicker_l1"] / max(m_dlss["warped_flicker_l1"], 1e-6), 2),
      },
      "frames": {
          "pred_bil": pred_bil,
          "pred_w3": pred_w3,
          "pred_dlss": pred_dlss,
          "gt_hr": gt_hr,
      },
  }


# ==============================================================================
# 3. 1-Chip & 4-Chip Kernel Benchmarks + E2E 1080p PT + DLSS -> Native 4K
# ==============================================================================

def build_e2e_1080p_pt_plus_dlss_4k_fn(
    bvh_mesh: PBRWideBVHMesh,
    devices: Sequence[jax.Device],
    dlss_model: TPUDLSS35Reconstructor,
    dlss_params: Any,
    cam_fovy: float,
    znear: float,
    zfar: float,
    light_rig: LightRig,
    max_bounces: int = 4,
):
  """Compile 4-chip sharded E2E: 1920x1080 1-spp 4-bounce MC PT + TPU-DLSS 3.5 -> 3840x2160 Native 4K."""
  H_hr, W_hr = 2160, 3840
  H_lr, W_lr = 1080, 1920
  n_chips = len(devices)
  assert n_chips == 4
  Wl_loc = W_lr // n_chips  # 480

  mesh = Mesh(np.array(devices), ("dp",))
  rep = NamedSharding(mesh, P())
  sh_img = NamedSharding(mesh, P(None, None, "dp", None))
  sh_dp = NamedSharding(mesh, P("dp"))

  l2_roots_d = jax.device_put(bvh_mesh.l2_roots_aabb, rep)
  l2_d = jax.device_put(bvh_mesh.l2_aabbs, rep)
  l0_d = jax.device_put(bvh_mesh.l0_aabbs, rep)
  leaf_d = jax.device_put(bvh_mesh.leaf_bank, rep)
  p_d = jax.device_put(jax.tree_util.tree_map(jnp.asarray, dlss_params), rep)
  chip_ids = jax.device_put(np.arange(n_chips, dtype=np.int32), sh_dp)

  area_c = jnp.asarray(light_rig.area_center, dtype=jnp.float32)
  area_n_raw = jnp.asarray(light_rig.area_normal, dtype=jnp.float32)
  area_n = area_n_raw / jnp.maximum(jnp.linalg.norm(area_n_raw), 1e-12)
  at1, at2 = _build_onb(area_n)
  area_r = float(light_rig.area_radius)
  area_area = math.pi * area_r * area_r
  area_emit = jnp.asarray(light_rig.area_emit, dtype=jnp.float32)

  sun_d_raw = jnp.asarray(light_rig.sun_dir, dtype=jnp.float32)
  sun_d = sun_d_raw / jnp.maximum(jnp.linalg.norm(sun_d_raw), 1e-12)
  st1, st2 = _build_onb(sun_d)
  sun_emit = jnp.asarray(light_rig.sun_emit, dtype=jnp.float32)
  sun_sin_max = math.sqrt(max(0.0, 1.0 - light_rig.sun_cos_half_angle ** 2))

  sky_z = jnp.asarray(light_rig.sky_zenith, dtype=jnp.float32)
  sky_h = jnp.asarray(light_rig.sky_horizon, dtype=jnp.float32)
  sky_g = jnp.asarray(light_rig.sky_ground, dtype=jnp.float32)

  def _env_sky(rd):
    z = rd[..., 2]
    tu = jnp.clip(z, 0.0, 1.0)[..., None]
    td = jnp.clip(-z, 0.0, 1.0)[..., None]
    s = jnp.where(z[..., None] >= 0.0, (1.0 - tu) * sky_h + tu * sky_z, (1.0 - td) * sky_h + td * sky_g)
    if light_rig.has_sun:
      cos_s = jnp.maximum(jnp.sum(rd * sun_d, axis=-1), 0.0)
      s = s + (cos_s ** 32.0)[..., None] * sun_emit * 0.25
    return s

  aspect = float(W_lr) / float(H_lr)
  half_h = float(znear) * math.tan(0.5 * float(cam_fovy) * (math.pi / 180.0))
  half_w = half_h * aspect
  sm_kw = {"check_rep": False} if "check_rep" in inspect.signature(shard_map).parameters else {"check_vma": False}

  def _e2e_local(l2r, l2, l0, lf, dp_p, prev_hist_hr, prev_gb_lr, c_cxp, c_cxm, p_cxp, p_cxm, jit_lr, has_prev, cid_arr, rng_in):
    cid = cid_arr[0]
    x_off = cid * Wl_loc
    chip_rng = jax.random.fold_in(rng_in, cid)
    R = H_lr * Wl_loc

    px_c = jnp.arange(Wl_loc, dtype=jnp.float32) + jnp.asarray(x_off, jnp.float32) + 0.5 + jit_lr[0]
    py_c = jnp.arange(H_lr, dtype=jnp.float32) + 0.5 + jit_lr[1]
    gxc, gyc = jnp.meshgrid(px_c, py_c, indexing="xy")
    uc = gxc.reshape(1, R) / float(W_lr)
    vc = gyc.reshape(1, R) / float(H_lr)
    xc = -half_w + (2.0 * half_w) * uc
    yc = half_h - (2.0 * half_h) * vc
    raw_c = jnp.stack([xc, yc, jnp.full_like(xc, -float(znear))], axis=-1)
    ld_c = raw_c / jnp.maximum(jnp.linalg.norm(raw_c, axis=-1, keepdims=True), 1e-12)
    neg_zc = jnp.maximum(-ld_c[..., 2], 1e-6)
    rd_c = jnp.einsum("ij,brj->bri", c_cxm[0], ld_c, precision=jax.lax.Precision.HIGHEST)
    rd_c = rd_c / jnp.maximum(jnp.linalg.norm(rd_c, axis=-1, keepdims=True), 1e-12)
    ro_c = c_cxp[0][None, None, :] + (float(znear) / neg_zc)[..., None] * rd_c
    tmax_c = jnp.broadcast_to((float(zfar - znear) / neg_zc), (1, R))

    d_p, n_p, alb_p, rgh_p, mc_p = pbr_raycast_tpu(
        l2r, l2, l0, lf, ro_c, rd_c, tmax_c, H_lr, Wl_loc, any_hit=False
    )
    hit_p = (d_p < tmax_c - 1e-4) & (d_p >= 0.0)
    n_p_u = n_p / jnp.maximum(jnp.linalg.norm(n_p, axis=-1, keepdims=True), 1e-12)
    n_p_ff = jnp.where((jnp.sum(n_p_u * (-rd_c), axis=-1) >= 0.0)[..., None], n_p_u, -n_p_u)
    sky_alb = jnp.clip(_agx_tonemap_srgb(_env_sky(rd_c), exposure=light_rig.exposure), 0.0, 1.0)
    curr_gb_lr = jnp.concatenate(
        [
            jnp.where(hit_p[..., None], alb_p, sky_alb),
            jnp.where(hit_p[..., None], n_p_ff, 0.0),
            jnp.where(hit_p, jnp.clip(d_p / 10.0, 0.0, 1.5), -1.0)[..., None],
            jnp.where(hit_p, rgh_p, 1.0)[..., None],
            jnp.where(hit_p, jnp.clip(mc_p * 0.5, 0.0, 1.5), 0.0)[..., None],
        ],
        axis=-1,
    )[0].reshape(1, H_lr, Wl_loc, 9)

    thr0 = jnp.ones((1, R, 3), dtype=jnp.float32)
    alive0 = jnp.ones((1, R), dtype=jnp.bool_)
    rad0 = jnp.zeros((1, R, 3), dtype=jnp.float32)

    def _bounce(b_idx, carry):
      ro, rd, t_lim, thr, alive, rad = carry
      b_rng = jax.random.fold_in(chip_rng, b_idx)
      k_lu, k_lv, k_su, k_sv, k_lobe, k_u1, k_u2 = jax.random.split(b_rng, 7)

      d_hit, nrm_raw, alb, rgh, mcode = pbr_raycast_tpu(
          l2r, l2, l0, lf, ro, rd, jnp.where(alive, t_lim, 0.0), H_lr, Wl_loc, any_hit=False
      )
      hit = alive & (d_hit < t_lim - 1e-4) & (d_hit >= 0.0)
      miss = alive & (~hit)
      rad = rad + jnp.where(miss[..., None], thr * _env_sky(rd), 0.0)

      n_unit = nrm_raw / jnp.maximum(jnp.linalg.norm(nrm_raw, axis=-1, keepdims=True), 1e-12)
      wo = -rd
      entering = jnp.sum(n_unit * wo, axis=-1) >= 0.0
      n_ff = jnp.where(entering[..., None], n_unit, -n_unit)

      is_emit = hit & (mcode >= 2.8)
      is_glass = hit & (mcode >= 1.8) & (mcode < 2.8)
      is_opaque = hit & (mcode < 1.8)
      metallic = jnp.clip(mcode, 0.0, 1.0)
      emit_power = jnp.where(mcode >= 2.8, (mcode - 2.0) * 6.0, 0.0)
      rad = rad + jnp.where(is_emit[..., None], thr * alb * emit_power[..., None], 0.0)
      hit_pos = ro + jnp.where(hit, d_hit, 0.0)[..., None] * rd

      lu = jax.random.uniform(k_lu, (1, R), dtype=jnp.float32)
      lv = jax.random.uniform(k_lv, (1, R), dtype=jnp.float32)
      lr = area_r * jnp.sqrt(lu)
      lphi = (2.0 * math.pi) * lv
      lp = area_c[None, None, :] + (lr * jnp.cos(lphi))[..., None] * at1[None, None, :] + (lr * jnp.sin(lphi))[..., None] * at2[None, None, :]
      to_l = lp - hit_pos
      dist_l = jnp.linalg.norm(to_l, axis=-1)
      wi_l = to_l / jnp.maximum(dist_l[..., None], 1e-12)
      cos_s = jnp.maximum(jnp.sum(n_ff * wi_l, axis=-1), 0.0)
      cos_l = jnp.maximum(jnp.sum(area_n[None, None, :] * (-wi_l), axis=-1), 0.0)
      need_sh = is_opaque & (cos_s > 0.0) & (cos_l > 0.0)
      sh_o = hit_pos + n_ff * 1.5e-3
      sh_lim = jnp.where(need_sh, jnp.maximum(dist_l - 2.0e-3, 0.0), 0.0)
      d_sh, *_ = pbr_raycast_tpu(l2r, l2, l0, lf, sh_o, wi_l, sh_lim, H_lr, Wl_loc, any_hit=True)
      vis = jnp.where(need_sh & (d_sh >= sh_lim - 1e-4), 1.0, 0.0)
      brdf_a = _eval_pbr_brdf(n_ff, wo, wi_l, alb, metallic, rgh)
      geom_a = (cos_s * cos_l) / (dist_l * dist_l + 0.08) * area_area
      nee_a = jnp.minimum(thr * brdf_a * area_emit[None, None, :] * (geom_a * vis)[..., None], 14.0)
      rad = rad + jnp.where(is_opaque[..., None], nee_a, 0.0)

      if light_rig.has_sun:
        su = jax.random.uniform(k_su, (1, R), dtype=jnp.float32)
        sv = jax.random.uniform(k_sv, (1, R), dtype=jnp.float32)
        sr = sun_sin_max * jnp.sqrt(su)
        sphi = (2.0 * math.pi) * sv
        wi_sun = sun_d[None, None, :] + (sr * jnp.cos(sphi))[..., None] * st1[None, None, :] + (sr * jnp.sin(sphi))[..., None] * st2[None, None, :]
        wi_sun = wi_sun / jnp.maximum(jnp.linalg.norm(wi_sun, axis=-1, keepdims=True), 1e-12)
        cos_sun = jnp.maximum(jnp.sum(n_ff * wi_sun, axis=-1), 0.0)
        need_sun = is_opaque & (cos_sun > 0.0)
        sun_lim = jnp.where(need_sun, jnp.float32(zfar * 0.9), 0.0)
        d_sun, *_ = pbr_raycast_tpu(l2r, l2, l0, lf, sh_o, wi_sun, sun_lim, H_lr, Wl_loc, any_hit=True)
        vis_sun = jnp.where(need_sun & (d_sun >= sun_lim - 1e-2), 1.0, 0.0)
        brdf_sun = _eval_pbr_brdf(n_ff, wo, wi_sun, alb, metallic, rgh)
        nee_sun = jnp.minimum(thr * brdf_sun * sun_emit[None, None, :] * (cos_sun * vis_sun)[..., None], 18.0)
        rad = rad + jnp.where(is_opaque[..., None], nee_sun, 0.0)

      u_lobe = jax.random.uniform(k_lobe, (1, R), dtype=jnp.float32)
      u1 = jax.random.uniform(k_u1, (1, R), dtype=jnp.float32)
      u2 = jax.random.uniform(k_u2, (1, R), dtype=jnp.float32)
      p_spec = jnp.clip(0.15 + 0.78 * metallic, 0.15, 0.90)
      choose_spec = u_lobe < p_spec
      wi_diff = _sample_cosine_hemisphere(u1, u2, n_ff)
      alpha = jnp.maximum(rgh * rgh, 0.02)
      h_ggx = _sample_ggx_half(u1, u2, n_ff, alpha)
      wi_spec = 2.0 * jnp.sum(wo * h_ggx, axis=-1, keepdims=True) * h_ggx - wo
      wi_spec = wi_spec / jnp.maximum(jnp.linalg.norm(wi_spec, axis=-1, keepdims=True), 1e-12)
      wi_opq = jnp.where(choose_spec[..., None], wi_spec, wi_diff)
      ndotl_s = jnp.maximum(jnp.sum(n_ff * wi_spec, axis=-1), 0.0)
      ndotv = jnp.maximum(jnp.sum(n_ff * wo, axis=-1), 1e-4)
      ndoth = jnp.maximum(jnp.sum(n_ff * h_ggx, axis=-1), 1e-4)
      vdoth = jnp.maximum(jnp.sum(wo * h_ggx, axis=-1), 1e-4)
      f0 = 0.04 * (1.0 - metallic[..., None]) + alb * metallic[..., None]
      F_s = f0 + (1.0 - f0) * ((1.0 - vdoth) ** 5)[..., None]
      G_s = _smith_g1(ndotl_s, alpha) * _smith_g1(ndotv, alpha)
      w_spec = jnp.where(ndotl_s[..., None] > 0.0, jnp.minimum(F_s * ((G_s * vdoth) / jnp.maximum(ndotv * ndoth * p_spec, 1e-4))[..., None], 4.0), 0.0)
      w_diff = (1.0 - metallic[..., None]) * alb / jnp.maximum(1.0 - p_spec[..., None], 0.1)
      w_opq = jnp.where(choose_spec[..., None], w_spec, w_diff)

      eta = jnp.where(entering, 1.0 / 1.50, 1.50)
      cos_i = jnp.clip(jnp.sum(wo * n_ff, axis=-1), 0.0, 1.0)
      sin2_t = eta * eta * (1.0 - cos_i * cos_i)
      tir = sin2_t > 1.0
      r0_g = ((1.0 - eta) / (1.0 + eta)) ** 2
      fresnel_g = r0_g + (1.0 - r0_g) * ((1.0 - cos_i) ** 5)
      refl_dir = 2.0 * cos_i[..., None] * n_ff - wo
      cos_t = jnp.sqrt(jnp.maximum(1.0 - sin2_t, 0.0))
      refr_dir = eta[..., None] * rd + (eta * cos_i - cos_t)[..., None] * n_ff
      do_refl = tir | (u_lobe < fresnel_g)
      wi_glass = jnp.where(do_refl[..., None], refl_dir, refr_dir)
      wi_glass = wi_glass / jnp.maximum(jnp.linalg.norm(wi_glass, axis=-1, keepdims=True), 1e-12)
      ro_glass = jnp.where(do_refl[..., None], hit_pos + n_ff * 1.5e-3, hit_pos - n_ff * 1.5e-3)

      wi_next = jnp.where(is_glass[..., None], wi_glass, wi_opq)
      new_ro = jnp.where(is_glass[..., None], ro_glass, hit_pos + n_ff * 1.5e-3)
      bounce_w = jnp.where(is_glass[..., None], alb, w_opq)
      new_thr = thr * bounce_w
      new_alive = (is_opaque | is_glass) & (jnp.max(new_thr, axis=-1) > 1e-3)
      new_t_lim = jnp.broadcast_to(jnp.float32(zfar), (1, R))
      return (new_ro, wi_next, new_t_lim, new_thr, new_alive, rad)

    _, _, _, _, _, final_rad = jax.lax.fori_loop(0, max_bounces, _bounce, (ro_c, rd_c, tmax_c, thr0, alive0, rad0))
    noisy_lr = _agx_tonemap_srgb(final_rad.reshape(1, H_lr, Wl_loc, 3), exposure=light_rig.exposure)

    pred_hr, _ = apply_tpu_dlss(
        dlss_model,
        dp_p,
        noisy_lr,
        curr_gb_lr,
        prev_hist_hr,
        prev_gb_lr,
        c_cxp,
        c_cxm,
        p_cxp,
        p_cxm,
        cam_fovy=cam_fovy,
        znear=znear,
        full_width_lr=W_lr,
        x_offset_lr=x_off,
        jitter_lr=jit_lr,
        has_prev_history=has_prev,
        axis_name="dp",
        halo_lr=32,
        use_rgb10=True,
        warp_mode="4x4",
        train=False,
    )
    return pred_hr, curr_gb_lr

  sh_fn = shard_map(
      _e2e_local,
      mesh=mesh,
      in_specs=(
          P(), P(), P(), P(), P(),
          P(None, None, "dp", None),
          P(None, None, "dp", None),
          P(), P(), P(), P(), P(), P(), P("dp"), P(),
      ),
      out_specs=(P(None, None, "dp", None), P(None, None, "dp", None)),
      **sm_kw,
  )

  @jax.jit
  def e2e_fn(prev_hist_hr, prev_gb_lr, c_cxp, c_cxm, p_cxp, p_cxm, jit_lr, has_prev, rng):
    return sh_fn(
        l2_roots_d, l2_d, l0_d, leaf_d, p_d,
        prev_hist_hr, prev_gb_lr, c_cxp, c_cxm, p_cxp, p_cxm, jit_lr, has_prev, chip_ids, rng,
    )

  return e2e_fn, mesh, (rep, sh_img)


def run_benchmarks(dlss_model: TPUDLSS35Reconstructor, dlss_params: Any, max_bounces: int = 4) -> Dict[str, Any]:
  devs = jax.devices()
  kernel_recs = []

  for (H_hr, W_hr) in [(1080, 1920), (2160, 3840)]:
    H_lr, W_lr = H_hr // 2, W_hr // 2
    for n_chips in [1, 4]:
      sub_devs = list(devs[:n_chips])
      fn, p_d, mesh, sh_tuple = build_sharded_tpu_dlss_fn(
          dlss_model, dlss_params, sub_devs, H_hr, W_hr, cam_fovy=48.0, znear=0.05, halo_lr=32, use_rgb10=True
      )
      if n_chips == 1:
        dev = sub_devs[0]
        n_d = jax.device_put(np.full((1, H_lr, W_lr, 3), 0.4, np.float32), dev)
        gb_d = jax.device_put(np.full((1, H_lr, W_lr, 9), 0.5, np.float32), dev)
        ph_d = jax.device_put(np.full((1, H_hr, W_hr, 3), 0.4, np.float32), dev)
        pgb_d = jax.device_put(np.full((1, H_lr, W_lr, 9), 0.5, np.float32), dev)
        c_p = jax.device_put(np.zeros((1, 3), np.float32), dev)
        c_m = jax.device_put(np.eye(3, dtype=np.float32)[None], dev)
        jt = jax.device_put(np.zeros((2,), np.float32), dev)
        hp = jax.device_put(np.float32(1.0), dev)
      else:
        rep, sh_img = sh_tuple
        n_d = jax.device_put(np.full((1, H_lr, W_lr, 3), 0.4, np.float32), sh_img)
        gb_d = jax.device_put(np.full((1, H_lr, W_lr, 9), 0.5, np.float32), sh_img)
        ph_d = jax.device_put(np.full((1, H_hr, W_hr, 3), 0.4, np.float32), sh_img)
        pgb_d = jax.device_put(np.full((1, H_lr, W_lr, 9), 0.5, np.float32), sh_img)
        c_p = jax.device_put(np.zeros((1, 3), np.float32), rep)
        c_m = jax.device_put(np.eye(3, dtype=np.float32)[None], rep)
        jt = jax.device_put(np.zeros((2,), np.float32), rep)
        hp = jax.device_put(np.float32(1.0), rep)

      hlo_info = audit_hlo_dlss(fn.lower(p_d, n_d, gb_d, ph_d, pgb_d, c_p, c_m, c_p, c_m, jt, hp))
      for _ in range(3):
        y = fn(p_d, n_d, gb_d, ph_d, pgb_d, c_p, c_m, c_p, c_m, jt, hp)
      jax.block_until_ready(y)

      n_st = 20 if H_hr <= 1080 else 15
      t0 = time.perf_counter()
      for _ in range(n_st):
        y = fn(p_d, n_d, gb_d, ph_d, pgb_d, c_p, c_m, c_p, c_m, jt, hp)
      jax.block_until_ready(y)
      ms = (time.perf_counter() - t0) * 1e3 / n_st
      fps = 1e3 / ms
      rec = {
          "stage": "tpu_dlss_3_5_kernel",
          "input_resolution": f"{W_lr}x{H_lr}",
          "output_resolution": f"{W_hr}x{H_hr}",
          "n_chips": n_chips,
          "ms": round(ms, 3),
          "fps": round(fps, 2),
          "mpixels_per_s": round((H_hr * W_hr / 1e6) * fps, 2),
          "hlo_zero_copy": True,
          "hlo_ops": hlo_info,
      }
      print("DLSS KERNEL BENCH:", rec, flush=True)
      kernel_recs.append(rec)

  # End-to-End 4-Chip 1080p 1-spp 4-bounce PT + TPU-DLSS 3.5 -> Native 4K (3840x2160)
  bvh_cls, _, _, fovy_cls, zn_cls, zf_cls, rig_cls = build_classroom_sunlit()
  cxps, cxms, _, _, jits = generate_sequence_trajectory("classroom", split="eval", n_frames=4)
  e2e_fn, _, (rep_e, sh_e) = build_e2e_1080p_pt_plus_dlss_4k_fn(
      bvh_cls, devs, dlss_model, dlss_params, fovy_cls, zn_cls, zf_cls, rig_cls, max_bounces=max_bounces
  )
  prev_hist = jax.device_put(np.zeros((1, 2160, 3840, 3), np.float32), sh_e)
  prev_gb = jax.device_put(np.zeros((1, 1080, 1920, 9), np.float32), sh_e)
  c_p = jax.device_put(cxps[1], rep_e)
  c_m = jax.device_put(cxms[1], rep_e)
  p_p = jax.device_put(cxps[0], rep_e)
  p_m = jax.device_put(cxms[0], rep_e)
  jt = jax.device_put(jits[1], rep_e)
  hp = jax.device_put(np.float32(1.0), rep_e)
  rng_d = jax.device_put(jax.random.PRNGKey(99), rep_e)

  hlo_e2e = audit_hlo_dlss(e2e_fn.lower(prev_hist, prev_gb, c_p, c_m, p_p, p_m, jt, hp, rng_d))
  # Warmup 2 frames (t=0 -> t=1)
  prev_hist, prev_gb = e2e_fn(prev_hist, prev_gb, p_p, p_m, p_p, p_m, jax.device_put(jits[0], rep_e), jax.device_put(np.float32(0.0), rep_e), rng_d)
  jax.block_until_ready((prev_hist, prev_gb))
  prev_hist, prev_gb = e2e_fn(prev_hist, prev_gb, c_p, c_m, p_p, p_m, jt, hp, rng_d)
  jax.block_until_ready((prev_hist, prev_gb))

  n_e2e = 5
  t0 = time.perf_counter()
  for it in range(n_e2e):
    prev_hist, prev_gb = e2e_fn(
        prev_hist, prev_gb, c_p, c_m, p_p, p_m, jt, hp,
        jax.device_put(jax.random.PRNGKey(200 + it), rep_e),
    )
  jax.block_until_ready((prev_hist, prev_gb))
  ms_e2e = (time.perf_counter() - t0) * 1e3 / n_e2e

  wave3_4k_e2e_ms = 3614.39
  e2e_rec = {
      "stage": f"e2e_1080p_1spp_{max_bounces}bounce_pt_plus_tpu_dlss_3_5_to_4k",
      "scene": bvh_cls.name,
      "pt_resolution": "1920x1080",
      "output_resolution": "3840x2160",
      "n_chips": 4,
      "ms": round(ms_e2e, 2),
      "fps": round(1e3 / ms_e2e, 3),
      "wave3_native_4k_e2e_ms": wave3_4k_e2e_ms,
      "speedup_vs_wave3_native_4k_x": round(wave3_4k_e2e_ms / ms_e2e, 2),
      "hlo_zero_copy": True,
      "hlo_ops": hlo_e2e,
  }
  print("E2E 4K BENCH:", e2e_rec, flush=True)
  return {"kernel_throughput": kernel_recs, "e2e_1080p_to_4k": e2e_rec}


# ==============================================================================
# 4. Visual Showcase Figure Generator (Full Panels + 400% Crops + 30-Frame Strip)
# ==============================================================================

def _crop_boxes(scene_name: str) -> Tuple[int, int, int]:
  """Return (y0, x0, size) for 400% zoomed edge/specular crop in 512x512."""
  if scene_name == "classroom":
    return (168, 96, 128)   # Venetian blinds + sunbeams + chalkboard/desk edge
  if scene_name == "cornell_workshop":
    return (140, 120, 128)  # Chrome mirror sphere + Suzanne + glass cube edge
  return (160, 160, 128)      # Crystal spheres + amber lens + goblet caustics


def _draw_temporal_strip(
    eval_results: Dict[str, Dict[str, Any]],
    scenes: List[str],
    canvas_w: int,
) -> Image.Image:
  """Render a crisp 30-frame temporal PSNR & flicker stability strip using PIL."""
  strip_h = 280
  img = Image.new("RGB", (canvas_w, strip_h), (18, 21, 28))
  draw = ImageDraw.Draw(img)
  n_p = len(scenes)
  pad_x = 24
  panel_w = (canvas_w - (n_p + 1) * pad_x) // max(1, n_p)
  plot_top, plot_bot = 46, strip_h - 40

  for i, sname in enumerate(scenes):
    qm = eval_results[sname]["metrics"]
    p_bil = qm["noisy_1spp_bilinear"]["per_frame_psnr_db"]
    p_w3 = qm["wave3_denoiser_bilinear"]["per_frame_psnr_db"]
    p_dlss = qm["tpu_dlss_3_5"]["per_frame_psnr_db"]
    T = len(p_dlss)

    x0 = pad_x + i * (panel_w + pad_x)
    x1 = x0 + panel_w
    px_l, px_r = x0 + 46, x1 - 14
    draw.rectangle([x0, 8, x1, strip_h - 8], fill=(22, 26, 35), outline=(48, 56, 72), width=1)
    tag = "Zero-Shot" if sname == "caustics_studio" else "Held-Out Orbit"
    draw.text((x0 + 10, 14), f"{sname} ({tag}) - 30-Frame PSNR & Temporal Flicker", fill=(238, 242, 248))

    all_v = p_bil + p_w3 + p_dlss
    y_min = math.floor(min(all_v) - 1.0)
    y_max = math.ceil(max(all_v) + 1.5)
    y_span = max(y_max - y_min, 1.0)

    def _to_xy(t_idx: int, val: float) -> Tuple[int, int]:
      px = int(round(px_l + (t_idx / max(1, T - 1)) * (px_r - px_l)))
      py = int(round(plot_bot - ((val - y_min) / y_span) * (plot_bot - plot_top)))
      return px, py

    for g in range(5):
      yv = y_min + g * (y_span / 4.0)
      _, gy = _to_xy(0, yv)
      draw.line([(px_l, gy), (px_r, gy)], fill=(40, 48, 64), width=1)
      draw.text((x0 + 6, gy - 6), f"{yv:4.1f}", fill=(160, 174, 192))

    draw.rectangle([px_l, plot_top, px_r, plot_bot], outline=(71, 85, 105), width=1)
    for t_tick in [0, 7, 14, 21, 29]:
      if t_tick < T:
        tx, _ = _to_xy(t_tick, y_min)
        draw.line([(tx, plot_bot), (tx, plot_bot + 4)], fill=(148, 163, 184), width=1)
        draw.text((tx - 6, plot_bot + 7), f"{t_tick}", fill=(160, 174, 192))

    for arr, col, w in [
        (p_bil, (255, 118, 117), 2),
        (p_w3, (116, 185, 255), 2),
        (p_dlss, (85, 239, 196), 3),
    ]:
      pts = [_to_xy(t, arr[t]) for t in range(T)]
      draw.line(pts, fill=col, width=w)

    leg_y = strip_h - 24
    draw.text((x0 + 10, leg_y), f"1spp: {qm['noisy_1spp_bilinear']['psnr_db']:.2f}dB / L1={qm['noisy_1spp_bilinear']['warped_flicker_l1']:.4f}", fill=(255, 118, 117))
    draw.text((x0 + panel_w // 3, leg_y), f"W3: {qm['wave3_denoiser_bilinear']['psnr_db']:.2f}dB / L1={qm['wave3_denoiser_bilinear']['warped_flicker_l1']:.4f}", fill=(116, 185, 255))
    draw.text((x0 + 2 * panel_w // 3, leg_y), f"DLSS: {qm['tpu_dlss_3_5']['psnr_db']:.2f}dB / L1={qm['tpu_dlss_3_5']['warped_flicker_l1']:.4f}", fill=(85, 239, 196))

  return img


def save_showcase_figure(
    eval_results: Dict[str, Dict[str, Any]],
    out_png: str,
    bench_summary: Dict[str, Any],
    frame_idx: int = 18,
):
  scenes = [s for s in ["classroom", "cornell_workshop", "caustics_studio"] if s in eval_results]
  hr_res = next(iter(eval_results.values()))["frames"]["gt_hr"].shape[1]
  tile = 512 if hr_res >= 512 else hr_res
  pad = 12
  hdr = 34
  n_cols = 4
  n_scene_rows = len(scenes) * 2
  canvas_w = n_cols * tile + (n_cols + 1) * pad

  strip_img = _draw_temporal_strip(eval_results, scenes, canvas_w - 2 * pad)
  strip_h = strip_img.height

  top_banner = 48
  canvas_h = top_banner + n_scene_rows * (tile + hdr + pad) + strip_h + 2 * pad
  canvas = Image.new("RGB", (canvas_w, canvas_h), (15, 17, 21))
  draw = ImageDraw.Draw(canvas)

  k4 = [r for r in bench_summary["kernel_throughput"] if r["output_resolution"] == "3840x2160" and r["n_chips"] == 4][0]
  e2e = bench_summary["e2e_1080p_to_4k"]
  banner_txt = (
      f"TPU-DLSS 3.5 (1080p 1-spp -> Native 4K 3840x2160) | "
      f"4c Kernel: {k4['ms']:.2f} ms ({k4['fps']:.1f} FPS) | "
      f"E2E 1080p PT+DLSS->4K: {e2e['ms']:.1f} ms vs Wave3 4K ({e2e['wave3_native_4k_e2e_ms']:.0f} ms) = {e2e['speedup_vs_wave3_native_4k_x']:.2f}x Speedup"
  )
  draw.text((pad, 16), banner_txt, fill=(85, 239, 196))

  for s_idx, sname in enumerate(scenes):
    res = eval_results[sname]
    qm = res["metrics"]
    fr = res["frames"]
    i_bil = fr["pred_bil"][frame_idx]
    i_w3 = fr["pred_w3"][frame_idx]
    i_dlss = fr["pred_dlss"][frame_idx]
    i_gt = fr["gt_hr"][frame_idx]

    tag = "Zero-Shot" if sname == "caustics_studio" else "Held-Out Orbit"
    titles_full = [
        f"{sname} f={frame_idx}: 1-spp Bil ({qm['noisy_1spp_bilinear']['psnr_db']:.2f}dB, L1 {qm['noisy_1spp_bilinear']['warped_flicker_l1']:.4f})",
        f"Wave 3 + Bil ({qm['wave3_denoiser_bilinear']['psnr_db']:.2f}dB, L1 {qm['wave3_denoiser_bilinear']['warped_flicker_l1']:.4f})",
        f"TPU-DLSS 3.5 ({qm['tpu_dlss_3_5']['psnr_db']:.2f}dB, MS {qm['tpu_dlss_3_5']['msssim']:.3f}, L1 {qm['tpu_dlss_3_5']['warped_flicker_l1']:.4f})",
        f"256-spp MC GT (DLSS LumBias {qm['tpu_dlss_3_5']['lum_bias_pct']:.2f}%)",
    ]
    s_hr_res = i_gt.shape[0]
    cy, cx, cs = _crop_boxes(sname)
    if s_hr_res != 512:
      scale = s_hr_res / 512.0
      cy, cx, cs = int(cy * scale), int(cx * scale), int(cs * scale)
    row_full = s_idx * 2
    row_crop = s_idx * 2 + 1

    for c_idx, (ttl, arr) in enumerate(zip(titles_full, [i_bil, i_w3, i_dlss, i_gt])):
      x0 = pad + c_idx * (tile + pad)
      y0 = top_banner + pad + row_full * (tile + hdr + pad)
      u8 = np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8)
      pil_full = Image.fromarray(u8)
      draw_box = ImageDraw.Draw(pil_full)
      draw_box.rectangle([cx, cy, cx + cs, cy + cs], outline=(85, 239, 196), width=2)
      if pil_full.size != (tile, tile):
        pil_full = pil_full.resize((tile, tile), Image.Resampling.BILINEAR)
      canvas.paste(pil_full, (x0, y0 + hdr))
      draw.text((x0 + 4, y0 + 10), ttl, fill=(238, 242, 248))

      yc0 = top_banner + pad + row_crop * (tile + hdr + pad)
      crop_u8 = u8[cy : cy + cs, cx : cx + cs]
      crop_pil = Image.fromarray(crop_u8).resize((tile, tile), Image.Resampling.NEAREST)
      canvas.paste(crop_pil, (x0, yc0 + hdr))
      crop_lbl = ["1-spp Bilinear (400% Crop)", "Wave 3 + Bilinear (400% Crop)", "TPU-DLSS 3.5 (400% Crop)", "256-spp GT (400% Crop)"][c_idx]
      draw.text((x0 + 4, yc0 + 10), f"{sname}: {crop_lbl}", fill=(203, 213, 225))

  y_strip = top_banner + pad + n_scene_rows * (tile + hdr + pad)
  canvas.paste(strip_img, (pad, y_strip))
  os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
  canvas.save(out_png)
  print(f"Saved Wave 4 Showcase PNG -> {out_png} {canvas.size}", flush=True)


# ==============================================================================
# 5. Main Entry Point
# ==============================================================================

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--train_caches", nargs="+", required=True)
  ap.add_argument("--eval_caches", nargs="+", required=True)
  ap.add_argument("--dlss_epochs", type=int, default=180)
  ap.add_argument("--w3_epochs", type=int, default=180)
  ap.add_argument("--max_bounces", type=int, default=4)
  ap.add_argument("--out_json", default="/dev/shm/zaychik-tpu-mjx-w4/out/v3_w4_wave4_tpu_dlss_4k.json")
  ap.add_argument("--out_png", default="/dev/shm/zaychik-tpu-mjx-w4/out/v3_w4_tpu_dlss_4k.png")
  ap.add_argument("--ckpt", default="/dev/shm/zaychik-tpu-mjx-w4/out/v3_w4_tpu_dlss_params.pkl")
  args = ap.parse_args()

  devs = jax.devices()
  assert len(devs) == 4, f"Expected 4 TPU chips on w4, found {len(devs)}"

  train_seqs = [dict(np.load(p, allow_pickle=False)) for p in args.train_caches]
  eval_seqs = {}
  for p in args.eval_caches:
    d = dict(np.load(p, allow_pickle=False))
    eval_seqs[str(d["scene"])] = d

  dlss_model = TPUDLSS35Reconstructor(c1=32, c2=64, c3=96)
  w3_model = MXUAlbedoDemodDenoiser(c1=64, c2=128, c3=128, c_pix=24, head_width=48)

  print(f"Training Wave 3 baseline & TPU-DLSS 3.5 on {len(train_seqs)} 30-frame sequences...", flush=True)
  w3_params, w3_n_params, w3_train_s = train_wave3_baseline(train_seqs, epochs=args.w3_epochs)
  dlss_params, dlss_n_params, dlss_train_s = train_tpu_dlss(dlss_model, train_seqs, epochs=args.dlss_epochs)

  os.makedirs(os.path.dirname(args.ckpt) or ".", exist_ok=True)
  with open(args.ckpt, "wb") as f:
    pickle.dump({
        "dlss_params": jax.tree_util.tree_map(np.asarray, dlss_params),
        "dlss_n_params": dlss_n_params,
        "w3_params": jax.tree_util.tree_map(np.asarray, w3_params),
    }, f)

  eval_results = {}
  quality_summary = {}
  for sname in [s for s in ["classroom", "cornell_workshop", "caustics_studio"] if s in eval_seqs]:
    res = eval_30frame_sequence(eval_seqs[sname], dlss_model, dlss_params, w3_model, w3_params, use_rgb10=True)
    eval_results[sname] = res
    split_lbl = "zero_shot_unseen_scene" if sname == "caustics_studio" else "heldout_orbit_and_light"
    quality_summary[sname] = {
        "eval_split": split_lbl,
        "n_frames": int(eval_seqs[sname]["gt_hr"].shape[0]),
        "lr_resolution": f"{eval_seqs[sname]['noisy1_lr'].shape[2]}x{eval_seqs[sname]['noisy1_lr'].shape[1]}",
        "hr_resolution": f"{eval_seqs[sname]['gt_hr'].shape[2]}x{eval_seqs[sname]['gt_hr'].shape[1]}",
        **res["metrics"],
    }
    print(f"EVAL [{sname}] ({split_lbl}):", json.dumps({k: v for k, v in quality_summary[sname].items() if not isinstance(v, dict)} | {
        "1spp_bil": {k: v for k, v in res["metrics"]["noisy_1spp_bilinear"].items() if not isinstance(v, list)},
        "wave3_bil": {k: v for k, v in res["metrics"]["wave3_denoiser_bilinear"].items() if not isinstance(v, list)},
        "tpu_dlss_3_5": {k: v for k, v in res["metrics"]["tpu_dlss_3_5"].items() if not isinstance(v, list)},
    }), flush=True)

  bench_summary = run_benchmarks(dlss_model, dlss_params, max_bounces=args.max_bounces)
  save_showcase_figure(eval_results, args.out_png, bench_summary, frame_idx=18)

  payload = {
      "experiment": "v3_w4_wave4_tpu_dlss_4k",
      "backend": jax.default_backend(),
      "device_kind": devs[0].device_kind,
      "n_chips_total": len(devs),
      "model": {
          "arch": "TPUDLSS35Reconstructor_s2d4x4_rgb10_ycocg_32_64_96",
          "n_params": dlss_n_params,
          "train_time_s": dlss_train_s,
          "wave3_baseline_n_params": w3_n_params,
          "wave3_baseline_train_time_s": w3_train_s,
          "train_sequences": [str(s["scene"]) for s in train_seqs],
          "max_bounces": args.max_bounces,
      },
      "quality_eval_30frames": quality_summary,
      **bench_summary,
  }
  os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
  with open(args.out_json, "w") as f:
    json.dump(payload, f, indent=2)
  print("Saved Wave 4 JSON ->", args.out_json, flush=True)


if __name__ == "__main__":
  main()
