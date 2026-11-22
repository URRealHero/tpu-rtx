"""Train, calibrate (held-out), and gate Stage 2 MXU neural shaders against Blender Cycles GT."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from gate import dilate_mask, evaluate_stage2_neural, luminance
from tpu_mjx_render.neural_shader import build_gbuffer_features, make_neural_shader_model
from tpu_mjx_render.renderer import GBuffer


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
  # Use 5x5 average pool as fast differentiable window
  k = (5, 5)
  mu_x = jax.lax.reduce_window(pred, 0.0, jax.lax.add, (1, 5, 5, 1), (1, 1, 1, 1), "SAME") / 25.0
  mu_y = jax.lax.reduce_window(target, 0.0, jax.lax.add, (1, 5, 5, 1), (1, 1, 1, 1), "SAME") / 25.0
  sig_x = jax.lax.reduce_window(pred * pred, 0.0, jax.lax.add, (1, 5, 5, 1), (1, 1, 1, 1), "SAME") / 25.0 - mu_x * mu_x
  sig_y = jax.lax.reduce_window(target * target, 0.0, jax.lax.add, (1, 5, 5, 1), (1, 1, 1, 1), "SAME") / 25.0 - mu_y * mu_y
  sig_xy = jax.lax.reduce_window(pred * target, 0.0, jax.lax.add, (1, 5, 5, 1), (1, 1, 1, 1), "SAME") / 25.0 - mu_x * mu_y
  num = (2.0 * mu_x * mu_y + c1) * (2.0 * sig_xy + c2)
  den = (mu_x * mu_x + mu_y * mu_y + c1) * (jnp.maximum(sig_x, 0.0) + jnp.maximum(sig_y, 0.0) + c2)
  ssim = jnp.mean(num / jnp.maximum(den, 1e-12))
  return 1.0 - ssim


def train_one_arch(arch: str, ds, gt_rgb: np.ndarray, epochs: int = 80, batch_size: int = 16, lr: float = 2e-3, seed: int = 0):
  split = ds["split"]
  train_idx = np.where(split == 0)[0]
  calib_idx = np.where(split == 1)[0]
  test_idx = np.where(split == 2)[0]

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

  gb_calib = _gbuffer_from_npz(ds, calib_idx)
  x_calib = jnp.asarray(build_gbuffer_features(gb_calib), dtype=jnp.float32)
  b_calib = jnp.asarray(ds["s1_rgb"][calib_idx], dtype=jnp.float32)
  y_calib = gt_rgb[calib_idx].astype(np.float32)

  gb_test = _gbuffer_from_npz(ds, test_idx)
  x_test = jnp.asarray(build_gbuffer_features(gb_test), dtype=jnp.float32)
  b_test = jnp.asarray(ds["s1_rgb"][test_idx], dtype=jnp.float32)
  y_test = gt_rgb[test_idx].astype(np.float32)

  model = make_neural_shader_model(arch)
  rng = jax.random.PRNGKey(seed)
  params = model.init(rng, jnp.asarray(x_train[:1]))["params"]
  ema_params = params

  steps_per_epoch = max(1, len(train_idx) // batch_size)
  total_steps = epochs * steps_per_epoch
  sched = optax.warmup_cosine_decay_schedule(
      init_value=lr * 0.1,
      peak_value=lr,
      warmup_steps=max(10, total_steps // 20),
      decay_steps=total_steps,
      end_value=lr * 0.05,
  )
  tx = optax.chain(
      optax.clip_by_global_norm(1.0),
      optax.adamw(learning_rate=sched, weight_decay=3e-5, eps=1e-8),
  )
  opt_state = tx.init(params)

  x_tr_dev = jnp.asarray(x_train)
  b_tr_dev = jnp.asarray(b_train)
  y_tr_dev = jnp.asarray(y_train)
  roi_tr_dev = jnp.asarray(roi_train)
  bnd_tr_dev = jnp.asarray(bnd_train)

  def _one_step(p, ema_p, opt_st, xb, base_b, yb, roib, bndb):
    def loss_fn(pr):
      delta = model.apply({"params": pr}, xb, train=True)
      raw = base_b + delta
      pred = jnp.clip(raw, 0.0, 1.0)
      # Compute L2/L1/outlier terms on unclipped `raw` so gradients never die if delta overshoots [0, 1]
      w = 1.0 + 2.0 * roib + 100.0 * bndb
      l2 = jnp.mean(w * (raw - yb) ** 2)
      l1 = jnp.mean(w * jnp.abs(raw - yb))
      ls = _ssim_loss(pred, yb)
      yp_lum = 0.2126 * raw[..., 0] + 0.7152 * raw[..., 1] + 0.0722 * raw[..., 2]
      yg_lum = 0.2126 * yb[..., 0] + 0.7152 * yb[..., 1] + 0.0722 * yb[..., 2]
      bnd_excess = jnp.mean(bndb[..., 0] * jnp.maximum(jnp.abs(yp_lum - yg_lum) - 0.04, 0.0) ** 2)
      return 0.6 * l2 + 0.25 * l1 + 0.15 * ls + 1200.0 * bnd_excess

    loss, grads = jax.value_and_grad(loss_fn)(p)
    updates, opt_st = tx.update(grads, opt_st, p)
    p = optax.apply_updates(p, updates)
    decay = 0.995
    ema_p = jax.tree_util.tree_map(lambda e, n: decay * e + (1.0 - decay) * n, ema_p, p)
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
          roi_tr_dev[idx_s],
          bnd_tr_dev[idx_s],
      )
      return (np_p, n_ema, n_ost), loss

    (p_out, ema_out, opt_out), losses = jax.lax.scan(_scan_body, (p, ema_p, opt_st), batch_idx)
    return p_out, ema_out, opt_out, jnp.mean(losses)

  @jax.jit
  def infer_fn(p, x_in, base_in):
    return jnp.clip(base_in + model.apply({"params": p}, x_in, train=False), 0.0, 1.0)

  t0 = time.time()
  np_rng = np.random.default_rng(seed)
  n_use = steps_per_epoch * batch_size
  for ep in range(epochs):
    perm = np_rng.permutation(len(train_idx))[:n_use].reshape(steps_per_epoch, batch_size)
    params, ema_params, opt_state, ep_loss = train_epoch(params, ema_params, opt_state, jnp.asarray(perm))
    if (ep + 1) % 20 == 0 or ep + 1 == epochs:
      jax.block_until_ready(ep_loss)
      print(f"[{arch}] epoch {ep+1}/{epochs} loss={float(ep_loss):.5f} ({time.time()-t0:.1f}s)", flush=True)

  # Held-out calibration on 32 calib frames (NEVER touches test_gate!)
  pred_calib = np.asarray(infer_fn(ema_params, x_calib, b_calib))
  cal_scale = float(np.mean(luminance(y_calib)) / max(np.mean(luminance(pred_calib)), 1e-6))

  pred_test_uncal = np.asarray(infer_fn(ema_params, x_test, b_test))
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

  return {
      "arch": arch,
      "train_time_s": time.time() - t0,
      "cal_scale": cal_scale,
      "gate_uncal": gate_uncal,
      "gate_cal": gate_cal,
      "ema_params": jax.tree_util.tree_map(np.asarray, ema_params),
      "pred_test_cal": pred_test_cal,
  }


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--dataset", default="out/data/panda_cycles_dataset.npz")
  ap.add_argument("--gt_rgb", default="out/data/panda_cycles_gt_rgb.npy")
  ap.add_argument("--archs", nargs="+", default=["M1_pixel_128", "M2_spatial_32", "M2_spatial_128"])
  ap.add_argument("--epochs", type=int, default=80)
  ap.add_argument("--batch_size", type=int, default=32)
  ap.add_argument("--lr", type=float, default=2e-3)
  ap.add_argument("--merge_existing", action="store_true")
  ap.add_argument("--out_dir", default="out/metrics")
  args = ap.parse_args()

  os.makedirs(args.out_dir, exist_ok=True)
  ds = np.load(args.dataset)
  if args.gt_rgb.endswith(".npz"):
    gt_rgb = np.load(args.gt_rgb)["rgb"].astype(np.float32)
  else:
    gt_rgb = np.load(args.gt_rgb).astype(np.float32)
  test_idx = np.where(ds["split"] == 2)[0]

  summary_path = os.path.join(args.out_dir, "stage2_summary.json")
  preds_path = os.path.join(args.out_dir, "stage2_test_preds.npz")
  summary = json.load(open(summary_path)) if (args.merge_existing and os.path.exists(summary_path)) else {}
  preds_bundle = {
      "s1_rgb": ds["s1_rgb"][test_idx],
      "gt_rgb": gt_rgb[test_idx],
      "roi_mask": ds["roi_mask"][test_idx],
      "geom_id": ds["s1_geom_id"][test_idx],
  }
  if args.merge_existing and os.path.exists(preds_path):
    prev_preds = np.load(preds_path)
    for k in prev_preds.files:
      if k.startswith("pred_"):
        preds_bundle[k] = prev_preds[k]

  for arch in args.archs:
    res = train_one_arch(arch, ds, gt_rgb, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    ckpt_path = os.path.join(args.out_dir, f"{arch}_params.pkl")
    with open(ckpt_path, "wb") as f:
      pickle.dump({"arch": arch, "cal_scale": res["cal_scale"], "params": res["ema_params"]}, f)

    preds_bundle[f"pred_{arch}"] = res["pred_test_cal"]
    summary[arch] = {
        "train_time_s": res["train_time_s"],
        "cal_scale": res["cal_scale"],
        "gate_uncal": res["gate_uncal"],
        "gate_cal": res["gate_cal"],
        "ckpt_path": ckpt_path,
    }
    print(f"=== {arch} GATE (cal_scale={res['cal_scale']:.4f}) ===")
    print(json.dumps(res["gate_cal"], indent=2))

  np.savez_compressed(os.path.join(args.out_dir, "stage2_test_preds.npz"), **preds_bundle)
  with open(os.path.join(args.out_dir, "stage2_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)


if __name__ == "__main__":
  main()
