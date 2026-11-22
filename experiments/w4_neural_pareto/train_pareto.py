"""Train and gate Stage 2 Neural Relighter Pareto candidates with teacher KD on TPU v5p."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import time
from typing import Any, Dict, List

import jax
import jax.numpy as jnp
import numpy as np
import optax

from gate import dilate_mask, evaluate_stage2_neural, luminance
from models import make_candidate_model
from tpu_mjx_render.neural_shader import build_gbuffer_features
from tpu_mjx_render.renderer import GBuffer
from tpu_mjx_render.roofline import inspect_hlo_ops


def _gbuffer_from_npz(ds, idx):
  return GBuffer(
      rgb=jnp.asarray(ds["s1_rgb"][idx]),
      albedo=jnp.asarray(ds["s1_albedo"][idx]),
      depth=jnp.asarray(ds["s1_depth"][idx]),
      z_depth=jnp.asarray(ds["s1_z_depth"][idx]),
      normal=jnp.asarray(ds["s1_normal"][idx]),
      view_dir=jnp.asarray(ds["s1_view_dir"][idx]),
      world_pos=jnp.asarray(ds["s1_world_pos"][idx]),
      geom_id=jnp.asarray(ds["s1_geom_id"][idx]),
      direct_shadow=jnp.asarray(ds["s1_shadow"][idx]),
      direct_rgb=jnp.asarray(ds["s1_rgb"][idx]),
  )


def _ssim_loss(pred: jax.Array, target: jax.Array) -> jax.Array:
  """Differentiable 5x5 local SSIM loss for crisp contact edges & soft shadows."""
  c1 = 0.01 ** 2
  c2 = 0.03 ** 2
  mu_x = jax.lax.reduce_window(pred, 0.0, jax.lax.add, (1, 5, 5, 1), (1, 1, 1, 1), "SAME") / 25.0
  mu_y = jax.lax.reduce_window(target, 0.0, jax.lax.add, (1, 5, 5, 1), (1, 1, 1, 1), "SAME") / 25.0
  sig_x = jax.lax.reduce_window(pred * pred, 0.0, jax.lax.add, (1, 5, 5, 1), (1, 1, 1, 1), "SAME") / 25.0 - mu_x * mu_x
  sig_y = jax.lax.reduce_window(target * target, 0.0, jax.lax.add, (1, 5, 5, 1), (1, 1, 1, 1), "SAME") / 25.0 - mu_y * mu_y
  sig_xy = jax.lax.reduce_window(pred * target, 0.0, jax.lax.add, (1, 5, 5, 1), (1, 1, 1, 1), "SAME") / 25.0 - mu_x * mu_y
  num = (2.0 * mu_x * mu_y + c1) * (2.0 * sig_xy + c2)
  den = (mu_x * mu_x + mu_y * mu_y + c1) * (jnp.maximum(sig_x, 0.0) + jnp.maximum(sig_y, 0.0) + c2)
  ssim = jnp.mean(num / jnp.maximum(den, 1e-12))
  return 1.0 - ssim


def _count_params(params: Any) -> int:
  return int(sum(x.size for x in jax.tree_util.tree_leaves(params)))


def _measure_stage2_only(model, params, cal_scale: float, batches: List[int]) -> Dict[str, Any]:
  """Benchmark pure Stage 2 ConvNet forward pass (excluding Stage 1) on 1 TPU chip."""
  out = {}
  dev = jax.devices()[0]
  params_dev = jax.device_put(jax.tree_util.tree_map(jnp.asarray, params), dev)

  @jax.jit
  def fwd(p, x_in, base_in):
    delta = model.apply({"params": p}, x_in, train=False)
    return jnp.clip((base_in + delta) * cal_scale, 0.0, 1.0)

  hlo_info = None
  for B in batches:
    x_dummy = jax.device_put(jnp.zeros((B, 64, 64, 24), dtype=jnp.float32), dev)
    b_dummy = jax.device_put(jnp.zeros((B, 64, 64, 3), dtype=jnp.float32), dev)
    if hlo_info is None:
      hlo_text = fwd.lower(params_dev, x_dummy, b_dummy).as_text()
      hlo_info = inspect_hlo_ops(hlo_text)
      hlo_info["send_recv_ops"] = len(re.findall(r"\b(send|recv|send-done|recv-done)\(", hlo_text))
      assert hlo_info["host_callbacks"] == 0, hlo_info
      assert hlo_info["send_recv_ops"] == 0, hlo_info

    for _ in range(3):
      y = fwd(params_dev, x_dummy, b_dummy)
    jax.block_until_ready(y)

    t0 = time.perf_counter()
    n_steps = 30
    for _ in range(n_steps):
      y = fwd(params_dev, x_dummy, b_dummy)
    jax.block_until_ready(y)
    dt_ms = (time.perf_counter() - t0) * 1e3 / n_steps
    out[f"B{B}"] = {
        "stage2_ms": round(dt_ms, 3),
        "stage2_only_fps": round(B * 1e3 / dt_ms, 1),
    }
  out["hlo_ops"] = hlo_info
  return out


def evaluate_checkpoint(arch: str, ckpt: Dict[str, Any], ds, gt_rgb: np.ndarray) -> Dict[str, Any]:
  test_idx = np.where(ds["split"] == 2)[0]
  gb_test = _gbuffer_from_npz(ds, test_idx)
  x_test = jnp.asarray(build_gbuffer_features(gb_test), dtype=jnp.float32)
  b_test = jnp.asarray(ds["s1_rgb"][test_idx], dtype=jnp.float32)
  y_test = gt_rgb[test_idx].astype(np.float32)

  model = make_candidate_model(arch)
  params = jax.tree_util.tree_map(jnp.asarray, ckpt["params"])
  cal_scale = float(ckpt["cal_scale"])

  @jax.jit
  def infer_fn(p, x_in, base_in):
    return jnp.clip(base_in + model.apply({"params": p}, x_in, train=False), 0.0, 1.0)

  pred_uncal = np.asarray(infer_fn(params, x_test, b_test))
  pred_cal = np.clip(pred_uncal * cal_scale, 0.0, 1.0)

  gate_cal = evaluate_stage2_neural(
      rgb_pred=pred_cal,
      rgb_ref=y_test,
      rgb_stage1_baseline=ds["s1_rgb"][test_idx],
      roi_mask=ds["roi_mask"][test_idx],
      geom_id=ds["s1_geom_id"][test_idx],
      box_geom_id=int(ds["box_geom_id"]),
      finger_geom_ids=tuple(int(x) for x in ds["finger_geom_ids"]),
  )
  return {
      "arch": arch,
      "n_params": _count_params(ckpt["params"]),
      "cal_scale": cal_scale,
      "gate_cal": gate_cal,
      "stage2_bench_1c": _measure_stage2_only(model, ckpt["params"], cal_scale, [1024, 2048, 4096]),
  }


def train_one_candidate(
    arch: str,
    ds,
    gt_rgb: np.ndarray,
    teacher_raw_train: np.ndarray,
    extra_ds=None,
    extra_gt_rgb: np.ndarray | None = None,
    epochs: int = 240,
    batch_size: int = 32,
    lr: float = 2.0e-3,
    kd_weight: float = 0.25,
    w_bnd: float = 250.0,
    bnd_weight: float = 6000.0,
    bnd_margin: float = 0.05,
    ema_decay: float = 0.995,
    eval_every: int = 20,
    seeds: List[int] = (0, 1, 2),
) -> Dict[str, Any]:
  split = ds["split"]
  train_idx = np.where(split == 0)[0]
  calib_idx = np.where(split == 1)[0]
  test_idx = np.where(split == 2)[0]
  assert len(train_idx) == 544 and len(calib_idx) == 32 and len(test_idx) == 64, (
      len(train_idx), len(calib_idx), len(test_idx)
  )

  gb_train = _gbuffer_from_npz(ds, train_idx)
  x_train = np.asarray(build_gbuffer_features(gb_train), dtype=np.float32)
  b_train = ds["s1_rgb"][train_idx].astype(np.float32)
  y_train = gt_rgb[train_idx].astype(np.float32)
  roi_train = ds["roi_mask"][train_idx].astype(np.float32)[..., None]
  gid_tr = ds["s1_geom_id"][train_idx]
  bnd_train = (
      dilate_mask(gid_tr == int(ds["box_geom_id"]), radius=1)
      & dilate_mask(np.isin(gid_tr, [int(x) for x in ds["finger_geom_ids"]]), radius=1)
  ).astype(np.float32)[..., None]

  if extra_ds is not None and extra_gt_rgb is not None:
    ex_idx = np.where(extra_ds["split"] == 0)[0]
    gb_ex = _gbuffer_from_npz(extra_ds, ex_idx)
    x_ex = np.asarray(build_gbuffer_features(gb_ex), dtype=np.float32)
    b_ex = extra_ds["s1_rgb"][ex_idx].astype(np.float32)
    y_ex = extra_gt_rgb[ex_idx].astype(np.float32)
    roi_ex = extra_ds["roi_mask"][ex_idx].astype(np.float32)[..., None]
    gid_ex = extra_ds["s1_geom_id"][ex_idx]
    bnd_ex = (
        dilate_mask(gid_ex == int(extra_ds["box_geom_id"]), radius=1)
        & dilate_mask(np.isin(gid_ex, [int(x) for x in extra_ds["finger_geom_ids"]]), radius=1)
    ).astype(np.float32)[..., None]
    x_train = np.concatenate([x_train, x_ex], axis=0)
    b_train = np.concatenate([b_train, b_ex], axis=0)
    y_train = np.concatenate([y_train, y_ex], axis=0)
    roi_train = np.concatenate([roi_train, roi_ex], axis=0)
    bnd_train = np.concatenate([bnd_train, bnd_ex], axis=0)

  n_train = x_train.shape[0]
  assert teacher_raw_train.shape[0] == n_train, (teacher_raw_train.shape[0], n_train)

  gb_calib = _gbuffer_from_npz(ds, calib_idx)
  x_calib = jnp.asarray(build_gbuffer_features(gb_calib), dtype=jnp.float32)
  b_calib = jnp.asarray(ds["s1_rgb"][calib_idx], dtype=jnp.float32)
  y_calib = gt_rgb[calib_idx].astype(np.float32)
  gid_cal = ds["s1_geom_id"][calib_idx]
  bnd_cal = (
      dilate_mask(gid_cal == int(ds["box_geom_id"]), radius=1)
      & dilate_mask(np.isin(gid_cal, [int(x) for x in ds["finger_geom_ids"]]), radius=1)
  )
  roi_cal = ds["roi_mask"][calib_idx]

  gb_test = _gbuffer_from_npz(ds, test_idx)
  x_test = jnp.asarray(build_gbuffer_features(gb_test), dtype=jnp.float32)
  b_test = jnp.asarray(ds["s1_rgb"][test_idx], dtype=jnp.float32)
  y_test = gt_rgb[test_idx].astype(np.float32)

  model = make_candidate_model(arch)
  steps_per_epoch = max(1, n_train // batch_size)
  total_steps = epochs * steps_per_epoch
  sched = optax.warmup_cosine_decay_schedule(
      init_value=lr * 0.1,
      peak_value=lr,
      warmup_steps=max(20, total_steps // 20),
      decay_steps=total_steps,
      end_value=lr * 0.02,
  )
  tx = optax.chain(
      optax.clip_by_global_norm(1.0),
      optax.adamw(learning_rate=sched, weight_decay=3e-5, eps=1e-8),
  )

  x_tr_dev = jnp.asarray(x_train)
  b_tr_dev = jnp.asarray(b_train)
  y_tr_dev = jnp.asarray(y_train)
  t_tr_dev = jnp.asarray(teacher_raw_train.astype(np.float32))
  roi_tr_dev = jnp.asarray(roi_train)
  bnd_tr_dev = jnp.asarray(bnd_train)

  def _one_step(p, ema_p, opt_st, xb, base_b, yb, tb, roib, bndb):
    def loss_fn(pr):
      delta = model.apply({"params": pr}, xb, train=True)
      raw = base_b + delta
      pred = jnp.clip(raw, 0.0, 1.0)

      w = 1.0 + 2.0 * roib + w_bnd * bndb
      l2_gt = jnp.mean(w * (raw - yb) ** 2)
      l1_gt = jnp.mean(w * jnp.abs(raw - yb))
      ls_gt = _ssim_loss(pred, yb)
      yp_lum = 0.2126 * raw[..., 0] + 0.7152 * raw[..., 1] + 0.0722 * raw[..., 2]
      yg_lum = 0.2126 * yb[..., 0] + 0.7152 * yb[..., 1] + 0.0722 * yb[..., 2]
      bnd_excess_gt = jnp.mean(bndb[..., 0] * jnp.maximum(jnp.abs(yp_lum - yg_lum) - bnd_margin, 0.0) ** 2)
      loss_gt = 0.6 * l2_gt + 0.25 * l1_gt + 0.15 * ls_gt + bnd_weight * bnd_excess_gt

      if kd_weight > 0.0:
        t_lum = 0.2126 * tb[..., 0] + 0.7152 * tb[..., 1] + 0.0722 * tb[..., 2]
        teacher_ok = (jnp.abs(t_lum - yg_lum) < 0.08)[..., None].astype(jnp.float32)
        w_kd = (1.0 - bndb * (1.0 - teacher_ok)) * (1.0 + 2.0 * roib)
        tb_clip = jnp.clip(tb, 0.0, 1.0)
        l2_kd = jnp.mean(w_kd * (raw - tb) ** 2)
        l1_kd = jnp.mean(w_kd * jnp.abs(raw - tb))
        ls_kd = _ssim_loss(pred, tb_clip)
        loss_kd = 0.6 * l2_kd + 0.25 * l1_kd + 0.15 * ls_kd
        return loss_gt + kd_weight * loss_kd
      return loss_gt

    loss, grads = jax.value_and_grad(loss_fn)(p)
    updates, opt_st = tx.update(grads, opt_st, p)
    p = optax.apply_updates(p, updates)
    ema_p = jax.tree_util.tree_map(lambda e, n: ema_decay * e + (1.0 - ema_decay) * n, ema_p, p)
    return p, ema_p, opt_st, loss

  @jax.jit
  def train_epoch(p, ema_p, opt_st, batch_idx):
    def _scan_body(carry, idx_s):
      cp, cema, cost = carry
      np_p, n_ema, n_ost, loss = _one_step(
          cp,
          cema,
          cost,
          x_tr_dev[idx_s],
          b_tr_dev[idx_s],
          y_tr_dev[idx_s],
          t_tr_dev[idx_s],
          roi_tr_dev[idx_s],
          bnd_tr_dev[idx_s],
      )
      return (np_p, n_ema, n_ost), loss

    (p_out, ema_out, opt_out), losses = jax.lax.scan(_scan_body, (p, ema_p, opt_st), batch_idx)
    return p_out, ema_out, opt_out, jnp.mean(losses)

  @jax.jit
  def infer_fn(p, x_in, base_in):
    return jnp.clip(base_in + model.apply({"params": p}, x_in, train=False), 0.0, 1.0)

  def _calib_score(ema_p):
    pred_c = np.asarray(infer_fn(ema_p, x_calib, b_calib))
    c_scale = float(np.mean(luminance(y_calib)) / max(np.mean(luminance(pred_c)), 1e-6))
    pred_c_cal = np.clip(pred_c * c_scale, 0.0, 1.0)
    cal_occ = float(np.mean(np.abs(luminance(pred_c_cal)[bnd_cal] - luminance(y_calib)[bnd_cal]) > 0.12)) if np.any(bnd_cal) else 0.0
    cal_roi_psnr = float(10.0 * np.log10(1.0 / max(float(np.mean((pred_c_cal[roi_cal] - y_calib[roi_cal]) ** 2)), 1e-12)))
    cal_psnr = float(10.0 * np.log10(1.0 / max(float(np.mean((pred_c_cal - y_calib) ** 2)), 1e-12)))
    # Select by highest ROI PSNR + full PSNR on calib (54-px calib occ is too noisy; teacher itself has 0.13 on calib!)
    return (0, 0.0, -cal_roi_psnr, -cal_psnr, cal_occ), c_scale

  best_score = (9, 999.0, 0.0, 0.0, 999.0)
  best_ema = None
  best_seed = seeds[0]
  best_ep = epochs
  best_cal_scale = 1.0
  n_params = 0

  t0 = time.time()
  n_use = steps_per_epoch * batch_size
  for seed in seeds:
    rng = jax.random.PRNGKey(seed)
    params = model.init(rng, jnp.asarray(x_train[:1]))["params"]
    ema_params = params
    n_params = _count_params(params)
    opt_state = tx.init(params)
    np_rng = np.random.default_rng(seed)

    for ep in range(epochs):
      perm = np_rng.permutation(n_train)[:n_use].reshape(steps_per_epoch, batch_size)
      params, ema_params, opt_state, ep_loss = train_epoch(params, ema_params, opt_state, jnp.asarray(perm))
      do_eval = (eval_every > 0 and (ep + 1) >= (epochs // 2) and (ep + 1) % eval_every == 0) or (ep + 1 == epochs)
      if (ep + 1) % 80 == 0 or ep + 1 == epochs:
        jax.block_until_ready(ep_loss)
        print(
            f"[{arch} s={seed}] ep {ep+1}/{epochs} loss={float(ep_loss):.5f} ({time.time()-t0:.1f}s)",
            flush=True,
        )
      if do_eval:
        sc, c_scale = _calib_score(ema_params)
        if sc < best_score:
          best_score = sc
          best_ema = jax.tree_util.tree_map(np.asarray, ema_params)
          best_seed = seed
          best_ep = ep + 1
          best_cal_scale = c_scale
        if ep + 1 == epochs:
          print(
              f"[{arch} s={seed}] FINAL calib: occ={sc[4]:.5f} roi_psnr={-sc[2]:.2f} psnr={-sc[3]:.2f} cal_scale={c_scale:.4f}",
              flush=True,
          )
  train_time_s = time.time() - t0

  best_ema_dev = jax.tree_util.tree_map(jnp.asarray, best_ema)
  cal_scale = best_cal_scale
  pred_test_uncal = np.asarray(infer_fn(best_ema_dev, x_test, b_test))
  pred_test_cal = np.clip(pred_test_uncal * cal_scale, 0.0, 1.0)

  gate_uncal = evaluate_stage2_neural(
      rgb_pred=pred_test_uncal,
      rgb_ref=y_test,
      rgb_stage1_baseline=ds["s1_rgb"][test_idx],
      roi_mask=ds["roi_mask"][test_idx],
      geom_id=ds["s1_geom_id"][test_idx],
      box_geom_id=int(ds["box_geom_id"]),
      finger_geom_ids=tuple(int(x) for x in ds["finger_geom_ids"]),
  )
  gate_cal = evaluate_stage2_neural(
      rgb_pred=pred_test_cal,
      rgb_ref=y_test,
      rgb_stage1_baseline=ds["s1_rgb"][test_idx],
      roi_mask=ds["roi_mask"][test_idx],
      geom_id=ds["s1_geom_id"][test_idx],
      box_geom_id=int(ds["box_geom_id"]),
      finger_geom_ids=tuple(int(x) for x in ds["finger_geom_ids"]),
  )

  # Also check exact apply_neural_shader formula: clip((base + delta) * cal_scale, 0, 1)
  @jax.jit
  def infer_apply_fn(p, x_in, base_in):
    delta = model.apply({"params": p}, x_in, train=False)
    return jnp.clip((base_in + delta) * cal_scale, 0.0, 1.0)

  pred_test_apply = np.asarray(infer_apply_fn(best_ema_dev, x_test, b_test))
  gate_apply = evaluate_stage2_neural(
      rgb_pred=pred_test_apply,
      rgb_ref=y_test,
      rgb_stage1_baseline=ds["s1_rgb"][test_idx],
      roi_mask=ds["roi_mask"][test_idx],
      geom_id=ds["s1_geom_id"][test_idx],
      box_geom_id=int(ds["box_geom_id"]),
      finger_geom_ids=tuple(int(x) for x in ds["finger_geom_ids"]),
  )

  ema_np = best_ema
  s2_bench = _measure_stage2_only(model, ema_np, cal_scale, [1024, 2048, 4096])

  return {
      "arch": arch,
      "n_params": n_params,
      "epochs": epochs,
      "batch_size": batch_size,
      "lr": lr,
      "kd_weight": kd_weight,
      "best_seed": best_seed,
      "best_epoch": best_ep,
      "best_calib_occ": best_score[4],
      "best_calib_roi_psnr": -best_score[2],
      "train_time_s": round(train_time_s, 2),
      "cal_scale": cal_scale,
      "gate_uncal": gate_uncal,
      "gate_cal": gate_cal,
      "gate_apply": gate_apply,
      "stage2_bench_1c": s2_bench,
      "ema_params": ema_np,
  }


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--dataset", default="/dev/shm/zaychik-tpu-mjx/out/data/panda_cycles_dataset.npz")
  ap.add_argument("--gt_rgb", default="/dev/shm/zaychik-tpu-mjx/out/data/panda_cycles_gt_rgb.npz")
  ap.add_argument("--extra_train", default="")
  ap.add_argument("--extra_train_gt", default="")
  ap.add_argument("--teacher_ckpt", default="/dev/shm/zaychik-tpu-mjx/out/metrics/M2_spatial_128_params.pkl")
  ap.add_argument(
      "--archs",
      nargs="+",
      default=["M2_demod_64", "M2_demod_fast_64", "M2_demod_convnext_lite", "M2_demod_s2d_mxu_64"],
  )
  ap.add_argument("--epochs", type=int, default=240)
  ap.add_argument("--eval_every", type=int, default=20)
  ap.add_argument("--batch_size", type=int, default=32)
  ap.add_argument("--lr", type=float, default=2.0e-3)
  ap.add_argument("--kd_weight", type=float, default=0.25)
  ap.add_argument("--w_bnd", type=float, default=250.0)
  ap.add_argument("--bnd_weight", type=float, default=6000.0)
  ap.add_argument("--bnd_margin", type=float, default=0.05)
  ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
  ap.add_argument("--out_dir", default="/dev/shm/zaychik-tpu-mjx-w4/out")
  ap.add_argument("--out_json", default="")
  ap.add_argument("--eval_only", action="store_true")
  ap.add_argument("--merge_json", action="store_true")
  args = ap.parse_args()

  os.makedirs(args.out_dir, exist_ok=True)
  ds = np.load(args.dataset)
  gt_rgb = np.load(args.gt_rgb)["rgb"].astype(np.float32)
  extra_ds = np.load(args.extra_train) if args.extra_train else None
  extra_gt_rgb = np.load(args.extra_train_gt).astype(np.float32) if args.extra_train_gt else None

  summary_path = args.out_json or os.path.join(args.out_dir, "v3_w4_train_summary.json")
  if args.merge_json and os.path.exists(summary_path):
    with open(summary_path) as f:
      results = json.load(f)
  else:
    results = {}

  # Load frozen teacher M2_spatial_128 and generate teacher raw targets on train split only
  with open(args.teacher_ckpt, "rb") as f:
    teacher_ck = pickle.load(f)
  if "M2_spatial_128_teacher" not in results:
    teacher_eval = evaluate_checkpoint("M2_spatial_128", teacher_ck, ds, gt_rgb)
    print("=== TEACHER M2_spatial_128 VERIFICATION ===", flush=True)
    print(json.dumps(teacher_eval, indent=2), flush=True)
    assert teacher_eval["gate_cal"]["passed"], "Teacher M2_spatial_128 failed gate!"
    results["backend"] = jax.default_backend()
    results["device_kind"] = jax.devices()[0].device_kind
    results["M2_spatial_128_teacher"] = teacher_eval
    results.setdefault("candidates", {})

  if args.eval_only:
    for arch in args.archs:
      ckpt_path = os.path.join(args.out_dir, f"v3_w4_{arch}_params.pkl")
      if not os.path.exists(ckpt_path):
        print(f"SKIP {arch}: {ckpt_path} not found", flush=True)
        continue
      with open(ckpt_path, "rb") as f:
        ck = pickle.load(f)
      rec = evaluate_checkpoint(arch, ck, ds, gt_rgb)
      rec["ckpt_path"] = ckpt_path
      results["candidates"][arch] = rec
      g = rec["gate_cal"]
      b = rec["stage2_bench_1c"]
      print(
          f"[EVAL] {arch}: params={rec['n_params']:,} passed={g['passed']} "
          f"PSNR={g['psnr_db']:.2f} SSIM={g['ssim']:.4f} MS-SSIM={g['msssim']:.4f} "
          f"lum_rel={g['lum_rel']:.5f} ROI_PSNR={g['roi_psnr_db']:.2f} "
          f"edge={g['roi_edge_px']:.3f} occ={g['roi_occ_mismatch']:.5f} | "
          f"S2_B1024={b['B1024']['stage2_ms']}ms S2_B4096={b['B4096']['stage2_ms']}ms",
          flush=True,
      )
    with open(summary_path, "w") as f:
      json.dump(results, f, indent=2)
    print(f"Saved eval summary to {summary_path}", flush=True)
    return

  train_idx = np.where(ds["split"] == 0)[0]
  gb_train = _gbuffer_from_npz(ds, train_idx)
  x_train_np = np.asarray(build_gbuffer_features(gb_train), dtype=np.float32)
  b_train_np = ds["s1_rgb"][train_idx].astype(np.float32)
  if extra_ds is not None and extra_gt_rgb is not None:
    ex_idx = np.where(extra_ds["split"] == 0)[0]
    gb_ex = _gbuffer_from_npz(extra_ds, ex_idx)
    x_train_np = np.concatenate([x_train_np, np.asarray(build_gbuffer_features(gb_ex), dtype=np.float32)], axis=0)
    b_train_np = np.concatenate([b_train_np, extra_ds["s1_rgb"][ex_idx].astype(np.float32)], axis=0)
  x_train_j = jnp.asarray(x_train_np, dtype=jnp.float32)
  b_train_j = jnp.asarray(b_train_np, dtype=jnp.float32)
  teacher_model = make_candidate_model("M2_spatial_128")
  teacher_params = jax.tree_util.tree_map(jnp.asarray, teacher_ck["params"])

  @jax.jit
  def teacher_raw_fn(p, x_in, base_in):
    return base_in + teacher_model.apply({"params": p}, x_in, train=False)

  teacher_raw_train = np.asarray(teacher_raw_fn(teacher_params, x_train_j, b_train_j))

  for arch in args.archs:
    res = train_one_candidate(
        arch=arch,
        ds=ds,
        gt_rgb=gt_rgb,
        teacher_raw_train=teacher_raw_train,
        extra_ds=extra_ds,
        extra_gt_rgb=extra_gt_rgb,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        kd_weight=args.kd_weight,
        w_bnd=args.w_bnd,
        bnd_weight=args.bnd_weight,
        bnd_margin=args.bnd_margin,
        eval_every=args.eval_every,
        seeds=args.seeds,
    )
    ckpt_path = os.path.join(args.out_dir, f"v3_w4_{arch}_params.pkl")
    with open(ckpt_path, "wb") as f:
      pickle.dump({"arch": arch, "cal_scale": res["cal_scale"], "params": res["ema_params"]}, f)

    rec = {k: v for k, v in res.items() if k != "ema_params"}
    rec["ckpt_path"] = ckpt_path
    results["candidates"][arch] = rec
    print(f"=== {arch} GATE (passed={rec['gate_cal']['passed']}) ===", flush=True)
    print(json.dumps(rec["gate_cal"], indent=2), flush=True)
    print(f"=== {arch} STAGE2 BENCH 1C ===", flush=True)
    print(json.dumps(rec["stage2_bench_1c"], indent=2), flush=True)
    with open(summary_path, "w") as f:
      json.dump(results, f, indent=2)

  print(f"Saved training summary to {summary_path}", flush=True)


if __name__ == "__main__":
  main()
