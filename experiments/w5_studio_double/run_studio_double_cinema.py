"""Wave 6 Studio Cinema Double-Feature: Blender Cycles `fishy_cat` + `monster_sss` on TPU v5p.

Reuses `wide_bvh64_uv_128.py` + `blend_compiler.py` + `IrradianceNRC3S_Demod_W128`
to evaluate exact Z-depth parity, held-out orbit & hero view PSNR/SSIM/MS-SSIM/LumBias
vs Cycles 5.3.0 GT, and 4-chip 1080p & Native 4K throughput (`host_callbacks == 0`,
`send_recv_ops == 0`), producing `v3_w5_wave6_studio_double.json` and
`v3_w5_studio_double_cinema.png`.
"""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
import os
import sys
import time
from typing import Any, Dict, Optional, Tuple

import jax
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
import numpy as np
import optax
from PIL import Image, ImageDraw

from blend_compiler import ScenePack
from gate import compute_msssim_batch, compute_psnr, compute_ssim_batch, luminance
from run_lone_monk_cinema import IrradianceNRC3ScaleDemod, _colormap_rgb, _grad_l1_loss
from run_w5_showcase import _hlo_audit, _ssim_loss
from wide_bvh64_uv import (
    WideBVH64UVSceneSpec,
    build_scene_gbuffer_features,
    pack_dev_arrays,
)
from wide_bvh64_uv_128 import (
    build_wide_bvh64_uv_128,
    render_wide_bvh64_uv128_stage1,
)


SCENE_CONFIGS = {
    "fishy_cat": {
        "title": "Blender Fishy Cat (Stylized DOF/Fur/Water)",
        "active_lights": None,
        "crop_box": (0.24, 0.88, 0.42, 0.93),  # cat face, eyes, nose, whiskers + bokeh background
    },
    "monster_sss": {
        "title": "Blender Monster Under the Bed (Cycles SSS/Wood)",
        "active_lights": None,
        "crop_box": (0.36, 0.81, 0.10, 0.90),  # SSS pink/purple monster + bed frame + wood floor
    },
}


def _train_demod_w128(
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
    epochs: int = 280,
    batch_size: int = 4,
    lr: float = 2.0e-3,
    base_width: int = 128,
):
  _, H, W, C = feats_tr.shape
  model = IrradianceNRC3ScaleDemod(base_width=base_width)
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
      oy = jax.random.randint(ky, (), 0, H // 4) * 4
      ox = jax.random.randint(kx, (), 0, W // 4) * 4
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

  _, _, _, _l = train_step(
      params, ema_params, opt_state, jnp.arange(batch_size), jax.random.PRNGKey(1), jnp.bool_(False)
  )
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
    if (ep + 1) % 70 == 0 or ep + 1 == epochs:
      print(f"[{scene_name}] ep {ep+1}/{epochs} loss={el/steps_per_epoch:.5f} ({time.time()-t0:.1f}s)", flush=True)
  train_s = time.time() - t0

  pred_cal = np.asarray(infer_fn(ema_params, jnp.asarray(feats_tr[cal_idx]), jnp.asarray(s1_tr[cal_idx])))
  cal_scale = float(np.mean(luminance(gt_tr[cal_idx])) / max(np.mean(luminance(pred_cal)), 1e-6))

  pred_te = np.clip(
      np.asarray(infer_fn(ema_params, jnp.asarray(feats_tr[te_idx]), jnp.asarray(s1_tr[te_idx]))) * cal_scale, 0.0, 1.0
  )
  y_te = gt_tr[te_idx]
  s1_te = s1_tr[te_idx]

  pred_hero_l = []
  for i in range(len(gt_hero)):
    ph_i = np.asarray(infer_fn(ema_params, jnp.asarray(feats_hero[i : i + 1]), jnp.asarray(s1_hero[i : i + 1])))
    pred_hero_l.append(np.clip(ph_i * cal_scale, 0.0, 1.0))
  pred_hero = np.concatenate(pred_hero_l, axis=0)

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
          "view0_lum_rel": round(
              float(np.mean(luminance(pred_hero[:1])) / max(np.mean(luminance(gt_hero[:1])), 1e-8) - 1.0), 6
          ),
          "all_hero_neural_psnr_db": round(compute_psnr(pred_hero, gt_hero), 4),
          "all_hero_ssim": round(compute_ssim_batch(pred_hero, gt_hero), 5),
          "all_hero_msssim": round(compute_msssim_batch(pred_hero, gt_hero), 5),
          "all_hero_lum_rel": round(
              float(np.mean(luminance(pred_hero)) / max(np.mean(luminance(gt_hero)), 1e-8) - 1.0), 6
          ),
      },
  }
  return model, metrics, ema_params, cal_scale, pred_hero


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
    dev_arrs_pre: Optional[Tuple[jax.Array, ...]] = None,
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
    dev_arrs = dev_arrs_pre if dev_arrs_pre is not None else pack_dev_arrays(spec_r, devs[0])
    cxp_d = jax.device_put(cxp, devs[0])
    cxm_d = jax.device_put(cxm, devs[0])
    np_d = jax.tree_util.tree_map(lambda x: jax.device_put(x, devs[0]), neural_params) if neural_params is not None else None

    @jax.jit
    def run_fn(da, cp, cm, npr):
      gb, rgh, mc = render_wide_bvh64_uv128_stage1(spec_r, da, cp, cm, interpret=False)
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
    dev_arrs = dev_arrs_pre if dev_arrs_pre is not None else pack_dev_arrays(spec_r, rep)
    cxp_d = jax.device_put(cxp, sh_dp)
    cxm_d = jax.device_put(cxm, sh_dp)
    np_d = jax.tree_util.tree_map(lambda x: jax.device_put(x, rep), neural_params) if neural_params is not None else None
    sm_kw = {"check_rep": False} if "check_rep" in inspect.signature(shard_map).parameters else {"check_vma": False}

    def _local(da, cp, cm, npr):
      gb, rgh, mc = render_wide_bvh64_uv128_stage1(spec_r, da, cp, cm, interpret=False)
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


def _z_stats(cy_z: np.ndarray, tp_z: np.ndarray) -> Dict[str, float]:
  cy_fg = cy_z < 500.0
  tp_fg = tp_z > 0.0
  both_fg = cy_fg & tp_fg
  if np.any(both_fg):
    diff_m = np.abs(cy_z[both_fg] - tp_z[both_fg])
    z_mae_m = float(np.mean(diff_m))
    z_med_m = float(np.median(diff_m))
  else:
    z_mae_m, z_med_m = -1.0, -1.0
  fg_match = float(np.mean(cy_fg == tp_fg))
  return {
      "z_mae_m": z_mae_m,
      "z_med_m": z_med_m,
      "fg_match": fg_match,
  }


def run_single_scene(
    scene_key: str,
    data_dir: str,
    epochs: int = 280,
    base_width: int = 128,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
  cfg = SCENE_CONFIGS[scene_key]
  pack = ScenePack.load(os.path.join(data_dir, f"{scene_key}_pack.npz"))
  orb = np.load(os.path.join(data_dir, f"{scene_key}_gt_orbit.npz"))
  hero = np.load(os.path.join(data_dir, f"{scene_key}_gt_hero.npz"))

  tr_w, tr_h = int(orb["width"]), int(orb["height"])
  hr_w, hr_h = int(hero["width"]), int(hero["height"])

  t_bvh = time.perf_counter()
  spec_tr = build_wide_bvh64_uv_128(
      pack, height=tr_h, width=tr_w, use_shadows=True, active_light_names=cfg["active_lights"]
  )
  bvh_build_s = time.perf_counter() - t_bvh
  spec_hr = dataclasses.replace(spec_tr, height=hr_h, width=hr_w)
  sc_ctr = tuple(float(x) for x in spec_tr.scene_center)
  sc_scl = tuple(float(x) for x in spec_tr.scene_scale)
  zs = float(spec_tr.z_scale)
  print(
      f"[{scene_key}] Built Wide-BVH-64-128: tris={spec_tr.bvh.n_tris_orig:,}, "
      f"leaves={spec_tr.bvh.n_leaves:,}, l1={spec_tr.bvh.n_l1:,}, l2={spec_tr.bvh.n_l2} in {bvh_build_s:.2f}s",
      flush=True,
  )

  dev0 = jax.devices()[0]
  dev_arrs_1c = pack_dev_arrays(spec_tr, dev0)

  @jax.jit
  def _render_gb_tr(da, cp, cm):
    gb, rgh, mc = render_wide_bvh64_uv128_stage1(spec_tr, da, cp, cm, interpret=False)
    ft = build_scene_gbuffer_features(gb, rgh, mc, sc_ctr, sc_scl, zs)
    return gb.rgb, gb.z_depth, ft

  @jax.jit
  def _render_gb_hr(da, cp, cm):
    gb, rgh, mc = render_wide_bvh64_uv128_stage1(spec_hr, da, cp, cm, interpret=False)
    ft = build_scene_gbuffer_features(gb, rgh, mc, sc_ctr, sc_scl, zs)
    return gb.rgb, gb.z_depth, ft

  hlo_tr = _hlo_audit(
      _render_gb_tr.lower(dev_arrs_1c, jnp.asarray(orb["cam_xpos"][:4]), jnp.asarray(orb["cam_xmat"][:4])).as_text()
  )
  assert hlo_tr["host_callbacks"] == 0 and hlo_tr["send_recv_ops"] == 0, hlo_tr

  N_orb = len(orb["rgb"])
  s1_tr_l, ft_tr_l = [], []
  for st in range(0, N_orb, 4):
    en = min(N_orb, st + 4)
    r, _, f = _render_gb_tr(dev_arrs_1c, jnp.asarray(orb["cam_xpos"][st:en]), jnp.asarray(orb["cam_xmat"][st:en]))
    s1_tr_l.append(np.asarray(r, np.float32))
    ft_tr_l.append(np.asarray(f, np.float32))
  s1_tr = np.concatenate(s1_tr_l, axis=0)
  ft_tr = np.concatenate(ft_tr_l, axis=0)

  s1_hr_l, z_hr_l, ft_hr_l = [], [], []
  for i in range(len(hero["rgb"])):
    r_h, z_h, f_h = _render_gb_hr(
        dev_arrs_1c, jnp.asarray(hero["cam_xpos"][i : i + 1]), jnp.asarray(hero["cam_xmat"][i : i + 1])
    )
    s1_hr_l.append(np.asarray(r_h, np.float32))
    z_hr_l.append(np.asarray(z_h, np.float32))
    ft_hr_l.append(np.asarray(f_h, np.float32))
  s1_hr = np.concatenate(s1_hr_l, axis=0)
  z_hr = np.concatenate(z_hr_l, axis=0)
  ft_hr = np.concatenate(ft_hr_l, axis=0)

  tp_z = z_hr[0]
  z_raw = _z_stats(np.asarray(hero["frame0_z_depth"], np.float32), tp_z)
  if "frame0_z_depth_pinhole" in hero:
    z_pin = _z_stats(np.asarray(hero["frame0_z_depth_pinhole"], np.float32), tp_z)
    z_mae_m = z_pin["z_mae_m"]
    z_med_m = z_pin["z_med_m"]
    fg_match = z_pin["fg_match"]
  else:
    z_pin = None
    z_mae_m = z_raw["z_mae_m"]
    z_med_m = z_raw["z_med_m"]
    fg_match = z_raw["fg_match"]

  print(
      f"[{scene_key}] Hero {hr_w}x{hr_h} Z-MAE={z_mae_m*1e3:.3f} mm, Z-MedAE={z_med_m*1e3:.3f} mm, "
      f"fg_match={fg_match:.5f}, hlo={hlo_tr}",
      flush=True,
  )

  gt_tr = orb["rgb"].astype(np.float32) / 255.0
  gt_hr = hero["rgb"].astype(np.float32) / 255.0

  all_idx = np.arange(N_orb)
  if N_orb >= 40:
    cal_idx = np.array([3, 13, 23, 33], dtype=np.int32)
    te_idx = np.array([5, 11, 17, 25, 31, 37], dtype=np.int32)
  else:
    cal_idx = np.array([3, 11, 19, 27], dtype=np.int32)
    te_idx = np.array([5, 13, 21, 29, 34], dtype=np.int32)
  tr_idx = np.array([i for i in all_idx if i not in set(cal_idx.tolist() + te_idx.tolist())], dtype=np.int32)

  mod, met, ema_params, cal_scale, pred_hr = _train_demod_w128(
      scene_key, ft_tr, s1_tr, gt_tr, tr_idx, cal_idx, te_idx, ft_hr, s1_hr, gt_hr,
      epochs=epochs, batch_size=4, base_width=base_width,
  )
  print(f"[{scene_key}] EVAL:", json.dumps(met, indent=2), flush=True)

  del s1_tr, ft_tr, gt_tr, dev_arrs_1c
  jax.clear_caches()

  n_devs = len(jax.devices())
  benches = []
  benches.append(_benchmark_scene(spec_tr, hero["cam_xpos"], hero["cam_xmat"], hr_w, hr_h, 1, 1, None, None, 1.0))
  benches.append(_benchmark_scene(spec_tr, hero["cam_xpos"], hero["cam_xmat"], hr_w, hr_h, 1, 1, mod, ema_params, cal_scale))

  if n_devs >= 4:
    devs = jax.devices()[:4]
    mesh = jax.sharding.Mesh(np.array(devs), ("dp",))
    rep = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    dev_arrs_4c = pack_dev_arrays(spec_tr, rep)
    for rw, rh, n_st in [(hr_w, hr_h, 5), (1920, 1080, 5), (3840, 2160, 3)]:
      benches.append(
          _benchmark_scene(spec_tr, hero["cam_xpos"], hero["cam_xmat"], rw, rh, 4, 4, None, None, 1.0, n_steps=n_st, dev_arrs_pre=dev_arrs_4c)
      )
      benches.append(
          _benchmark_scene(spec_tr, hero["cam_xpos"], hero["cam_xmat"], rw, rh, 4, 4, mod, ema_params, cal_scale, n_steps=n_st, dev_arrs_pre=dev_arrs_4c)
      )
    del dev_arrs_4c

  jax.clear_caches()

  scene_rec = {
      "title": cfg["title"],
      "scene_meta": pack.summary(),
      "bvh_stats": {
          "n_tris_orig": spec_tr.bvh.n_tris_orig,
          "n_leaves": spec_tr.bvh.n_leaves,
          "n_l1": spec_tr.bvh.n_l1,
          "n_l2": spec_tr.bvh.n_l2,
          "bvh_build_s": round(bvh_build_s, 3),
      },
      "parity": {
          "hero_z_mae_m": z_mae_m,
          "hero_z_med_m": z_med_m,
          "hero_fg_match": fg_match,
          "hero_z_dof_raw": z_raw if z_pin is not None else None,
          "hlo": hlo_tr,
      },
      "neural_relighting": met,
      "benchmarks": benches,
  }
  visuals = {
      "s1": np.clip(s1_hr[0], 0.0, 1.0),
      "s2": np.clip(pred_hr[0], 0.0, 1.0),
      "gt": np.clip(gt_hr[0], 0.0, 1.0),
      "zd": z_hr[0],
      "rec": scene_rec,
      "crop_box": cfg["crop_box"],
  }
  return scene_rec, visuals


def _paste_aspect_fit(canvas: Image.Image, img_arr: np.ndarray, x0: int, y0: int, pw: int, ph: int) -> None:
  u8 = (np.clip(img_arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
  ih, iw = u8.shape[:2]
  scale = min(pw / float(iw), ph / float(ih))
  nw = max(1, int(round(iw * scale)))
  nh = max(1, int(round(ih * scale)))
  pil_im = Image.fromarray(u8).resize((nw, nh), Image.Resampling.BILINEAR)
  ox = x0 + (pw - nw) // 2
  oy = y0 + (ph - nh) // 2
  canvas.paste(pil_im, (ox, oy))


def make_double_feature_cinema_figure(
    all_vis: Dict[str, Dict[str, Any]],
    out_png: str,
) -> None:
  scenes = list(all_vis.keys())
  pw, ph = 640, 360
  pad, hdr, row_hdr, top_banner = 14, 32, 34, 54
  n_cols = 5
  n_rows = len(scenes)
  cw = n_cols * pw + (n_cols + 1) * pad
  ch = top_banner + n_rows * (row_hdr + ph + hdr) + (n_rows + 1) * pad
  canvas = Image.new("RGB", (cw, ch), (13, 15, 20))
  draw = ImageDraw.Draw(canvas)

  top_text = (
      "Blender Cycles Studio Demo Double-Feature on TPU v5p (4 chips, Mosaic Wide-BVH-64-128 + IrradianceNRC3S_Demod_W128) | "
      "Exact Cycles 5.3.0 EXR Z-Depth | 0 host callbacks | 0 send/recv ops"
  )
  draw.rectangle([0, 0, cw, top_banner], fill=(22, 26, 36))
  draw.text((pad, 19), top_text, fill=(242, 247, 255))

  for r_i, sc in enumerate(scenes):
    v = all_vis[sc]
    s1, s2, gt, zd = v["s1"], v["s2"], v["gt"], v["zd"]
    rec = v["rec"]
    meta = rec["scene_meta"]
    nr = rec["neural_relighting"]
    hr = nr["hero_official_plus_orbit"]
    orb = nr["orbit_test"]
    par = rec["parity"]
    b4_1080 = next(
        (b for b in rec["benchmarks"] if b["resolution"] == "1920x1080" and b["stage2"] and b["n_chips"] == 4),
        rec["benchmarks"][-1],
    )
    b4_4k = next(
        (b for b in rec["benchmarks"] if b["resolution"] == "3840x2160" and b["stage2"] and b["n_chips"] == 4),
        rec["benchmarks"][-1],
    )

    y_row = top_banner + pad + r_i * (row_hdr + ph + hdr + pad)
    z_lbl = f"Z-MedAE {par['hero_z_med_m']*1e3:.2f} mm" if par["hero_z_mae_m"] > 0.1 else f"Z-MAE {par['hero_z_mae_m']*1e3:.2f} mm"
    row_banner = (
        f"{rec['title']} ({meta['tris']/1e6:.2f}M tris, {meta['n_textures']}x{meta['tex_res']} UV) | "
        f"Hero {hr['view0_neural_psnr_db']:.2f} dB / SSIM {hr['view0_ssim']:.4f} / MS-SSIM {hr['view0_msssim']:.4f} | "
        f"Held-Out Orbit {orb['neural_psnr_db']:.2f} dB (+{orb['psnr_gain_db']:.2f} dB) | "
        f"{z_lbl} | 4c 1080p: {b4_1080['img_per_sec']:.2f} img/s | 4c 4K: {b4_4k['img_per_sec']:.2f} img/s"
    )
    draw.rectangle([pad, y_row, cw - pad, y_row + row_hdr - 4], fill=(28, 34, 48))
    draw.text((pad + 10, y_row + 9), row_banner, fill=(130, 225, 255))

    H_img, W_img, _ = s2.shape
    yf0, yf1, xf0, xf1 = v["crop_box"]
    cy0, cy1 = int(yf0 * H_img), int(yf1 * H_img)
    cx0, cx1 = int(xf0 * W_img), int(xf1 * W_img)
    s2_c = s2[cy0:cy1, cx0:cx1]
    gt_c = gt[cy0:cy1, cx0:cx1]
    cw_c = s2_c.shape[1]
    split_crop = np.concatenate([s2_c[:, : cw_c // 2], gt_c[:, cw_c // 2 :]], axis=1)
    split_crop[:, max(0, cw_c // 2 - 2) : min(cw_c, cw_c // 2 + 2)] = np.array([0.2, 0.85, 1.0], np.float32)

    fg = zd > 0.0
    if np.any(fg):
      z_lo, z_hi = np.percentile(zd[fg], [2, 98])
      zd_norm = np.where(fg, 1.0 - np.clip((zd - z_lo) / max(z_hi - z_lo, 0.5), 0.0, 1.0), 0.0)
    else:
      zd_norm = np.zeros_like(zd)
    zd_vis = _colormap_rgb(zd_norm, "viridis")

    z_col_lbl = (
        f"Exact Z-Depth | MedAE {par['hero_z_med_m']*1e3:.2f} mm, FG {par['hero_fg_match']*100:.2f}%"
        if par["hero_z_mae_m"] > 0.1
        else f"Exact Z-Depth | Z-MAE {par['hero_z_mae_m']*1e3:.2f} mm, FG {par['hero_fg_match']*100:.2f}%"
    )
    panels = [
        (s1, f"TPU S1 G-Buffer ({hr['view0_stage1_psnr_db']:.2f} dB)"),
        (s2, f"TPU v5p S2 Neural Relit ({hr['view0_neural_psnr_db']:.2f} dB, SSIM {hr['view0_ssim']:.4f})"),
        (gt, f"Cycles 5.3.0 GT ({gt.shape[1]}x{gt.shape[0]}, Lum {hr['view0_lum_rel']*100:+.2f}%)"),
        (split_crop, "2x Split Zoom: Left = TPU S2 | Right = Cycles GT"),
        (zd_vis, z_col_lbl),
    ]

    y_panel = y_row + row_hdr
    for c_i, (img, title) in enumerate(panels):
      x0 = pad + c_i * (pw + pad)
      draw.rectangle([x0, y_panel + hdr, x0 + pw, y_panel + hdr + ph], fill=(18, 21, 28))
      _paste_aspect_fit(canvas, img, x0, y_panel + hdr, pw, ph)
      draw.text((x0 + 4, y_panel + 10), title, fill=(235, 242, 250))

  os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
  canvas.save(out_png)
  print("SAVED CINEMA FIGURE:", out_png, canvas.size, flush=True)


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--data_dir", default="/dev/shm/zaychik-tpu-mjx-w5")
  ap.add_argument("--scenes", nargs="+", default=["fishy_cat", "monster_sss"])
  ap.add_argument("--epochs", type=int, default=280)
  ap.add_argument("--base_width", type=int, default=128)
  ap.add_argument("--out_json", default="/dev/shm/zaychik-tpu-mjx-w5/v3_w5_wave6_studio_double.json")
  ap.add_argument("--out_png", default="/dev/shm/zaychik-tpu-mjx-w5/v3_w5_studio_double_cinema.png")
  args = ap.parse_args()

  results: Dict[str, Any] = {
      "experiment": "v3_w5_wave6_studio_double",
      "device_kind": jax.devices()[0].device_kind,
      "n_devices": len(jax.devices()),
      "scenes": {},
  }
  all_vis: Dict[str, Dict[str, Any]] = {}
  for sc in args.scenes:
    rec, vis = run_single_scene(sc, args.data_dir, epochs=args.epochs, base_width=args.base_width)
    results["scenes"][sc] = rec
    all_vis[sc] = vis

  os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
  with open(args.out_json, "w") as f:
    json.dump(results, f, indent=2)
  print("SAVED JSON:", args.out_json, flush=True)

  make_double_feature_cinema_figure(all_vis, args.out_png)


if __name__ == "__main__":
  main()
