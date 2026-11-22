"""Run V3 Blender Open Data (classroom, monster, junkshop) Head-to-Head vs Cycles 5.3.0 & wenhaochai/rtx-h100 on w5 TPU v5p."""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
import os
import pickle
import re
import time
from typing import Any, Dict, List, Tuple

import jax
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optax
from PIL import Image

from blend_compiler import ScenePack
from gate import compute_msssim_batch, compute_psnr, compute_ssim_batch, luminance
from run_w5_showcase import SpatialRelighter3Scale, _hlo_audit, _ssim_loss
from wide_bvh64_uv import (
    WideBVH64UVSceneSpec,
    build_scene_gbuffer_features,
    build_wide_bvh64_uv,
    pack_dev_arrays,
    render_wide_bvh64_uv_stage1,
)


# Reference baselines from wenhaochai/rtx-h100 (Blender Open Data Benchmark)
RTX_H100_BASELINES = {
    "stock_blender_5_3_0_h100_spm": 7041,
    "exact_h100_spm": 14106,
    "rtx_5090_optix_spm": 19202,
    "nrc_scatter_h100_spm": 55148,
}

SCENE_RESOLUTIONS = {
    "classroom": {"train": (480, 272), "hero": (960, 544), "native": (1920, 1088)},
    "monster":   {"train": (256, 256), "hero": (512, 512), "native": (1024, 1024)},
    "junkshop":  {"train": (512, 256), "hero": (1024, 512), "native": (2048, 1024)},
}


def _train_relighter_26ch(
    scene_name: str,
    feats_tr: np.ndarray,
    s1_tr: np.ndarray,
    gt_tr: np.ndarray,
    tr_idx: np.ndarray,
    cal_idx: np.ndarray,
    te_idx: np.ndarray,
    feats_hero: np.ndarray,
    s1_hero: np.ndarray,
    gt_hero: np.ndarray,
    epochs: int = 240,
    batch_size: int = 6,
    lr: float = 2.5e-3,
):
  _, H, W, C = feats_tr.shape
  model = SpatialRelighter3Scale(base_width=96)
  rng = jax.random.PRNGKey(0)
  params = model.init(rng, jnp.asarray(feats_tr[:1]))["params"]
  ema_params = params

  steps_per_epoch = max(1, len(tr_idx) // batch_size)
  total_steps = epochs * steps_per_epoch
  sched = optax.warmup_cosine_decay_schedule(
      init_value=lr * 0.1,
      peak_value=lr,
      warmup_steps=max(8, total_steps // 12),
      decay_steps=total_steps,
      end_value=lr * 0.03,
  )
  tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=2e-5))
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
      oy = jax.random.randint(ky, (), 0, H)
      ox = jax.random.randint(kx, (), 0, W)
      return (
          jax.lax.dynamic_slice(x_up, (0, oy, ox, 0), (B_i, H, W, C)),
          jax.lax.dynamic_slice(b_up, (0, oy, ox, 0), (B_i, H, W, 3)),
          jax.lax.dynamic_slice(y_up, (0, oy, ox, 0), (B_i, H, W, 3)),
      )

    xb, bb, yb = jax.lax.cond(use_up2x, _upsample_crop, lambda v: v, (xb, bb, yb))

    def loss_fn(pr):
      raw = bb + model.apply({"params": pr}, xb, train=True)
      pred = jnp.clip(raw, 0.0, 1.0)
      return 0.60 * jnp.mean((raw - yb) ** 2) + 0.25 * jnp.mean(jnp.abs(raw - yb)) + 0.15 * _ssim_loss(pred, yb)

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
      up2x = (step_i % 3 == 1)
      params, ema_params, opt_state, loss = train_step(
          params, ema_params, opt_state, jnp.asarray(sl), jax.random.PRNGKey(step_i + 100), jnp.bool_(up2x)
      )
      el += float(loss)
      step_i += 1
    if (ep + 1) % 80 == 0 or ep + 1 == epochs:
      print(f"[{scene_name}] ep {ep+1}/{epochs} loss={el/steps_per_epoch:.5f} ({time.time()-t0:.1f}s)", flush=True)
  train_s = time.time() - t0

  pred_cal = np.asarray(infer_fn(ema_params, jnp.asarray(feats_tr[cal_idx]), jnp.asarray(s1_tr[cal_idx])))
  cal_scale = float(np.mean(luminance(gt_tr[cal_idx])) / max(np.mean(luminance(pred_cal)), 1e-6))

  pred_te = np.clip(np.asarray(infer_fn(ema_params, jnp.asarray(feats_tr[te_idx]), jnp.asarray(s1_tr[te_idx]))) * cal_scale, 0.0, 1.0)
  y_te = gt_tr[te_idx]
  s1_te = s1_tr[te_idx]

  pred_hero = np.clip(np.asarray(infer_fn(ema_params, jnp.asarray(feats_hero), jnp.asarray(s1_hero))) * cal_scale, 0.0, 1.0)

  metrics = {
      "arch": "SpatialRelighter3S_W96_MS",
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
  return metrics, ema_params, cal_scale, pred_hero


def _benchmark_scene(
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
      "equiv_spm_64spp": round(img_s * 60.0 * 64.0, 1) if neural_model is not None else round(img_s * 60.0, 1),
      "raw_frames_per_min": round(img_s * 60.0, 1),
      "mpix_per_sec": round((batch_size * res_w * res_h) / max(dt, 1e-9) / 1e6, 2),
      "host_callbacks": hlo_info["host_callbacks"],
      "send_recv_ops": hlo_info["send_recv_ops"],
  }


def run_scene(
    scene_name: str,
    data_dir: str,
    epochs: int = 240,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
  pack_path = os.path.join(data_dir, f"{scene_name}_pack.npz")
  orb_path = os.path.join(data_dir, f"{scene_name}_gt_orbit.npz")
  hero_path = os.path.join(data_dir, f"{scene_name}_gt_hero.npz")

  pack = ScenePack.load(pack_path)
  orb = np.load(orb_path)
  hero = np.load(hero_path)

  tr_w, tr_h = int(orb["width"]), int(orb["height"])
  hr_w, hr_h = int(hero["width"]), int(hero["height"])
  nat_w, nat_h = SCENE_RESOLUTIONS[scene_name]["native"]

  spec_tr = build_wide_bvh64_uv(pack, height=tr_h, width=tr_w, use_shadows=True)
  spec_hr = dataclasses.replace(spec_tr, height=hr_h, width=hr_w)
  sc_ctr = tuple(float(x) for x in spec_tr.scene_center)
  sc_scl = tuple(float(x) for x in spec_tr.scene_scale)
  zs = float(spec_tr.z_scale)

  dev0 = jax.devices()[0]
  dev_arrs_tr = pack_dev_arrays(spec_tr, dev0)

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

  hlo_tr = _hlo_audit(_render_gb_tr.lower(dev_arrs_tr, jnp.asarray(orb["cam_xpos"][:4]), jnp.asarray(orb["cam_xmat"][:4])).as_text())
  assert hlo_tr["host_callbacks"] == 0 and hlo_tr["send_recv_ops"] == 0, hlo_tr

  # Render Stage 1 orbit GBuffers
  N_orb = len(orb["rgb"])
  s1_tr_l, z_tr_l, ft_tr_l = [], [], []
  for st in range(0, N_orb, 4):
    en = min(N_orb, st + 4)
    r, z, f = _render_gb_tr(dev_arrs_tr, jnp.asarray(orb["cam_xpos"][st:en]), jnp.asarray(orb["cam_xmat"][st:en]))
    s1_tr_l.append(np.asarray(r, np.float32))
    z_tr_l.append(np.asarray(z, np.float32))
    ft_tr_l.append(np.asarray(f, np.float32))
  s1_tr = np.concatenate(s1_tr_l, axis=0)
  z_tr = np.concatenate(z_tr_l, axis=0)
  ft_tr = np.concatenate(ft_tr_l, axis=0)

  # Render Stage 1 hero GBuffers
  r_h, z_h, f_h = _render_gb_hr(dev_arrs_tr, jnp.asarray(hero["cam_xpos"]), jnp.asarray(hero["cam_xmat"]))
  s1_hr = np.asarray(r_h, np.float32)
  z_hr = np.asarray(z_h, np.float32)
  ft_hr = np.asarray(f_h, np.float32)

  # Z-depth parity vs Cycles 5.3.0 EXR on Frame 0
  cy_z = np.asarray(hero["frame0_z_depth"], np.float32)
  tp_z = z_hr[0]
  cy_fg = cy_z < 500.0
  tp_fg = tp_z > 0.0
  both_fg = cy_fg & tp_fg
  z_mae_m = float(np.mean(np.abs(cy_z[both_fg] - tp_z[both_fg]))) if np.any(both_fg) else -1.0
  fg_match = float(np.mean(cy_fg == tp_fg))
  print(f"[{scene_name}] Hero {hr_w}x{hr_h} Z-MAE={z_mae_m*1e3:.3f} mm, fg_match={fg_match:.5f}, hlo={hlo_tr}", flush=True)

  gt_tr = orb["rgb"].astype(np.float32) / 255.0
  gt_hr = hero["rgb"].astype(np.float32) / 255.0

  # Split 24 orbit views: 18 train / 2 calib / 4 held-out test
  all_idx = np.arange(N_orb)
  cal_idx = np.array([3, 15], dtype=np.int32)
  te_idx = np.array([5, 11, 17, 23], dtype=np.int32)
  tr_idx = np.array([i for i in all_idx if i not in set(cal_idx.tolist() + te_idx.tolist())], dtype=np.int32)

  met, ema_params, cal_scale, pred_hr = _train_relighter_26ch(
      scene_name, ft_tr, s1_tr, gt_tr, tr_idx, cal_idx, te_idx, ft_hr, s1_hr, gt_hr, epochs=epochs, batch_size=6
  )
  print(f"[{scene_name}] EVAL:", json.dumps(met, indent=2), flush=True)

  mod = SpatialRelighter3Scale(base_width=96)
  n_devs = len(jax.devices())
  benches = []
  # Hero & Native res benchmarks (1c + 4c)
  for nc in ([1, 4] if n_devs >= 4 else [1]):
    bs_hr = 4 if nc == 4 else 1
    benches.append(_benchmark_scene(spec_tr, hero["cam_xpos"], hero["cam_xmat"], hr_w, hr_h, bs_hr, nc, None, None, 1.0))
    benches.append(_benchmark_scene(spec_tr, hero["cam_xpos"], hero["cam_xmat"], hr_w, hr_h, bs_hr, nc, mod, ema_params, cal_scale))
    benches.append(_benchmark_scene(spec_tr, hero["cam_xpos"], hero["cam_xmat"], nat_w, nat_h, bs_hr, nc, None, None, 1.0, n_steps=3))
    benches.append(_benchmark_scene(spec_tr, hero["cam_xpos"], hero["cam_xmat"], nat_w, nat_h, bs_hr, nc, mod, ema_params, cal_scale, n_steps=3))

  scene_rec = {
      "scene_meta": pack.summary(),
      "parity": {
          "hero_z_mae_m": z_mae_m,
          "hero_fg_match": fg_match,
          "hlo": hlo_tr,
      },
      "neural_relighting": met,
      "benchmarks": benches,
  }
  visuals = {
      "s1_hero0": np.clip(s1_hr[0], 0.0, 1.0),
      "s2_hero0": np.clip(pred_hr[0], 0.0, 1.0),
      "gt_hero0": np.clip(gt_hr[0], 0.0, 1.0),
      "z_hero0": z_hr[0],
      "s2_orbit_heldout": np.clip(pred_hr[1], 0.0, 1.0),
      "gt_orbit_heldout": np.clip(gt_hr[1], 0.0, 1.0),
      "met": met,
      "z_mae_m": z_mae_m,
      "benches": benches,
      "tris": pack.n_tris,
  }
  return scene_rec, visuals


def make_comparison_figure(all_vis: Dict[str, Dict[str, Any]], out_png: str) -> None:
  scenes = list(all_vis.keys())
  n_rows = len(scenes)
  fig, axes = plt.subplots(n_rows, 5, figsize=(21.0, 4.0 * n_rows), dpi=170)
  if n_rows == 1:
    axes = axes[None, :]

  for r_i, sc in enumerate(scenes):
    v = all_vis[sc]
    s1 = v["s1_hero0"]
    s2 = v["s2_hero0"]
    gt = v["gt_hero0"]
    zd = v["z_hero0"]
    met = v["met"]
    hr = met["hero_official_plus_orbit"]
    orb = met["orbit_test"]
    b4_s2 = next((b for b in v["benches"] if b["n_chips"] == 4 and b["stage2"] and b["resolution"].startswith(str(s2.shape[1]))), None)
    if b4_s2 is None:
      b4_s2 = v["benches"][-1]

    err5 = np.mean(np.clip(np.abs(s2 - gt) * 5.0, 0.0, 1.0), axis=-1)
    fg = zd > 0.0
    if np.any(fg):
      z_lo, z_hi = np.percentile(zd[fg], [2, 98])
      zd_vis = np.where(fg, 1.0 - np.clip((zd - z_lo) / max(z_hi - z_lo, 0.5), 0.0, 1.0), 0.0)
    else:
      zd_vis = np.zeros_like(zd)

    panels = [
        (s1, f"{sc} ({v['tris']/1e6:.2f}M tris) TPU S1\nUV Wide-BVH-64 ({hr['view0_stage1_psnr_db']:.2f} dB)", None),
        (s2, f"TPU S2 NRC ({hr['view0_neural_psnr_db']:.2f} dB | SSIM {hr['view0_ssim']:.4f})\nOrbit Held-Out: {orb['neural_psnr_db']:.2f} dB | {b4_s2['img_per_sec']:.1f} img/s (4c)", None),
        (gt, f"Blender 5.3.0 Cycles GT\n({gt.shape[1]}x{gt.shape[0]}, LumRel {hr['view0_lum_rel']*100:+.2f}%)", None),
        (err5, f"Error Heatmap (x5)\n(MS-SSIM {hr['view0_msssim']:.4f} | 0 callbacks)", "inferno"),
        (zd_vis, f"Exact Mesh Z-Depth\n(Z-MAE {v['z_mae_m']*1e3:.2f} mm vs EXR)", "viridis"),
    ]
    for c_i, (img, title, cmap) in enumerate(panels):
      ax = axes[r_i, c_i]
      ax.imshow(img, cmap=cmap, vmin=0 if cmap else None, vmax=1 if cmap else None)
      ax.set_title(title, fontsize=9.0, fontweight="bold")
      ax.axis("off")

  plt.suptitle(
      "V3 Blender 5.3.0 Open Data Auto-Compiler (.blend Shader Graph -> Wide-BVH-64 + TPU NRC) vs Cycles GT & wenhaochai/rtx-h100",
      fontsize=12.5, fontweight="bold", y=1.01,
  )
  plt.tight_layout()
  os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
  plt.savefig(out_png, bbox_inches="tight")
  plt.close(fig)
  print("SAVED FIGURE:", out_png, flush=True)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--data_dir", default="/dev/shm/zaychik-tpu-mjx-w5/opendata")
  ap.add_argument("--scenes", nargs="+", default=["classroom", "monster", "junkshop"])
  ap.add_argument("--epochs", type=int, default=240)
  ap.add_argument("--out_json", default="/dev/shm/zaychik-tpu-mjx-w5/v3_w5_opendata_vs_rtxh100.json")
  ap.add_argument("--out_png", default="/dev/shm/zaychik-tpu-mjx-w5/v3_w5_opendata_vs_rtxh100.png")
  args = ap.parse_args()

  results = {
      "experiment": "v3_w5_opendata_vs_rtxh100",
      "device_kind": jax.devices()[0].device_kind,
      "n_devices": len(jax.devices()),
      "rtx_h100_baselines": RTX_H100_BASELINES,
      "scenes": {},
  }
  all_vis = {}
  for sc in args.scenes:
    rec, vis = run_scene(sc, args.data_dir, epochs=args.epochs)
    results["scenes"][sc] = rec
    all_vis[sc] = vis

  make_comparison_figure(all_vis, args.out_png)
  os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
  with open(args.out_json, "w") as f:
    json.dump(results, f, indent=2)
  print("SAVED JSON:", args.out_json, flush=True)


if __name__ == "__main__":
  main()
