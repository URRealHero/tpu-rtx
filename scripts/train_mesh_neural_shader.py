"""Train Stage 2 MXU SpatialRelighter on Real Blender Curved Triangle Meshes (Suzanne + Torus)."""

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

from gate import compute_msssim_batch, compute_psnr, compute_ssim_batch, luminance
from tpu_mjx_render.mesh_ray import build_mesh_scene_spec, render_mesh_batch
from tpu_mjx_render.neural_shader import SpatialRelighter, build_gbuffer_features


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


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--scene_npz", default="out/assets/blender_real_scene.npz")
  ap.add_argument("--gt_npz", default="out/assets/blender_real_scene_gt.npz")
  ap.add_argument("--epochs", type=int, default=80)
  ap.add_argument("--batch_size", type=int, default=12)
  ap.add_argument("--lr", type=float, default=3e-3)
  ap.add_argument("--base_width", type=int, default=32)
  ap.add_argument("--out_dir", default="out/metrics")
  args = ap.parse_args()

  os.makedirs(args.out_dir, exist_ok=True)
  ds = np.load(args.scene_npz)
  gt = np.load(args.gt_npz)
  gt_rgb = gt["rgb"].astype(np.float32)

  M = int(ds["n_meshes"])
  meshes = [
      {
          "name": str(ds[f"mesh_{m}_name"]),
          "verts": ds[f"mesh_{m}_verts"],
          "faces": ds[f"mesh_{m}_faces"],
          "rgba": ds[f"mesh_{m}_rgba"],
      }
      for m in range(M)
  ]
  mspec = build_mesh_scene_spec(
      meshes,
      height=int(ds["height"]),
      width=int(ds["width"]),
      cam_fovy=float(ds["cam_fovy"]),
      znear=float(ds["znear"]),
      zfar=float(ds["zfar"]),
      light_pos=tuple(float(x) for x in ds["light_pos"]),
  )

  t0 = time.time()
  gb_all = jax.jit(lambda p, m, cp, cm: render_mesh_batch(mspec, p, m, cp, cm, mode="jax_cluster"))(
      jnp.asarray(ds["obj_xpos"]),
      jnp.asarray(ds["obj_xmat"]),
      jnp.asarray(ds["cam_xpos"]),
      jnp.asarray(ds["cam_xmat"]),
  )
  feats_all = np.asarray(build_gbuffer_features(gb_all), dtype=np.float32)
  s1_all = np.asarray(gb_all.rgb, dtype=np.float32)
  print(f"Rendered {len(gt_rgb)} Blender mesh G-buffers in {time.time()-t0:.2f}s", flush=True)

  split = ds["split"]
  tr_idx = np.where(split == 0)[0]
  cal_idx = np.where(split == 1)[0]
  te_idx = np.where(split == 2)[0]

  model = SpatialRelighter(base_width=args.base_width)
  rng = jax.random.PRNGKey(0)
  params = model.init(rng, jnp.asarray(feats_all[:1]))["params"]
  ema_params = params

  steps_per_epoch = max(1, len(tr_idx) // args.batch_size)
  total_steps = args.epochs * steps_per_epoch
  sched = optax.warmup_cosine_decay_schedule(
      init_value=args.lr * 0.1,
      peak_value=args.lr,
      warmup_steps=max(4, total_steps // 15),
      decay_steps=total_steps,
      end_value=args.lr * 0.05,
  )
  tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=3e-5))
  opt_state = tx.init(params)

  x_tr = jnp.asarray(feats_all[tr_idx])
  b_tr = jnp.asarray(s1_all[tr_idx])
  y_tr = jnp.asarray(gt_rgb[tr_idx])

  @jax.jit
  def train_step(p, ema_p, opt_st, idx_s):
    xb, bb, yb = x_tr[idx_s], b_tr[idx_s], y_tr[idx_s]

    def loss_fn(pr):
      raw = bb + model.apply({"params": pr}, xb, train=True)
      pred = jnp.clip(raw, 0.0, 1.0)
      return 0.6 * jnp.mean((raw - yb) ** 2) + 0.25 * jnp.mean(jnp.abs(raw - yb)) + 0.15 * _ssim_loss(pred, yb)

    loss, grads = jax.value_and_grad(loss_fn)(p)
    ups, opt_st = tx.update(grads, opt_st, p)
    p = optax.apply_updates(p, ups)
    ema_p = jax.tree_util.tree_map(lambda e, n: 0.99 * e + 0.01 * n, ema_p, p)
    return p, ema_p, opt_st, loss

  @jax.jit
  def infer(p, xf, bf):
    return jnp.clip(bf + model.apply({"params": p}, xf, train=False), 0.0, 1.0)

  t_tr = time.time()
  np_rng = np.random.default_rng(0)
  for ep in range(args.epochs):
    perm = np_rng.permutation(len(tr_idx))
    el = 0.0
    for s in range(steps_per_epoch):
      sl = perm[s * args.batch_size : (s + 1) * args.batch_size]
      params, ema_params, opt_state, loss = train_step(params, ema_params, opt_state, jnp.asarray(sl))
      el += float(loss)
    if (ep + 1) % 20 == 0 or ep + 1 == args.epochs:
      print(f"[Mesh SpatialRelighter-W{args.base_width}] epoch {ep+1}/{args.epochs} loss={el/steps_per_epoch:.5f} ({time.time()-t_tr:.1f}s)", flush=True)

  pred_cal = np.asarray(infer(ema_params, jnp.asarray(feats_all[cal_idx]), jnp.asarray(s1_all[cal_idx])))
  cal_scale = float(np.mean(luminance(gt_rgb[cal_idx])) / max(np.mean(luminance(pred_cal)), 1e-6))

  pred_te_uncal = np.asarray(infer(ema_params, jnp.asarray(feats_all[te_idx]), jnp.asarray(s1_all[te_idx])))
  pred_te = np.clip(pred_te_uncal * cal_scale, 0.0, 1.0)

  y_te = gt_rgb[te_idx]
  s1_te = s1_all[te_idx]
  psnr_s1 = compute_psnr(s1_te, y_te)
  psnr_m2 = compute_psnr(pred_te, y_te)
  ssim_m2 = compute_ssim_batch(pred_te, y_te)
  msssim_m2 = compute_msssim_batch(pred_te, y_te)
  lum_rel = float(np.mean(luminance(pred_te)) / max(np.mean(luminance(y_te)), 1e-8) - 1.0)

  summary = {
      "scene": "blender_suzanne_torus_pedestal",
      "base_width": args.base_width,
      "n_train": int(len(tr_idx)),
      "n_calib": int(len(cal_idx)),
      "n_test": int(len(te_idx)),
      "train_time_s": round(time.time() - t_tr, 2),
      "cal_scale": cal_scale,
      "stage1_mesh_psnr_db": psnr_s1,
      "neural_mesh_psnr_db": psnr_m2,
      "psnr_gain_db": psnr_m2 - psnr_s1,
      "neural_mesh_ssim": ssim_m2,
      "neural_mesh_msssim": msssim_m2,
      "lum_rel": lum_rel,
  }
  with open(os.path.join(args.out_dir, "v2_mesh_neural_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
  with open(os.path.join(args.out_dir, "v2_mesh_relighter_params.pkl"), "wb") as f:
    pickle.dump({"arch": f"SpatialRelighter_W{args.base_width}", "cal_scale": cal_scale, "params": jax.tree_util.tree_map(np.asarray, ema_params)}, f)
  np.savez_compressed(
      os.path.join(args.out_dir, "v2_mesh_test_preds.npz"),
      test_idx=te_idx,
      s1_rgb=s1_te,
      pred_rgb=pred_te,
      gt_rgb=y_te,
      s1_z_depth=np.asarray(gb_all.z_depth[te_idx], np.float32),
  )
  print("MESH NEURAL RELIGHTER SUMMARY:", json.dumps(summary, indent=2))


if __name__ == "__main__":
  main()
