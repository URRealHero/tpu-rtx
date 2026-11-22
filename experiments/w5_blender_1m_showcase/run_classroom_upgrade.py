"""Task B: OpenData `classroom` Sharpness Upgrade on TPU v5p.

Upgrades `classroom` using:
  1. Higher-res 512x512 baked textures (`max_tex_res=512`).
  2. Dual shadow-ray features (`shadow_top2=True` -> both `exterior_fillLight` AND `sun` cast Wide-BVH-64 `any_hit` shadows, 27-ch G-Buffer).
  3. 768x432 orbit training + `IrradianceNRC3ScaleDemod` (`W128` with 3x3 detail fusion + gradient-preserving loss).
  4. Hero 960x544 + Native 1920x1088 benchmarks (`host_callbacks == 0`).

Saves standalone `v3_w5_classroom_upgrade.json` and `v3_w5_classroom_upgrade.png`,
and provides `--update_opendata` to safely update `classroom` in `v3_w5_opendata_vs_rtxh100.{json,png}`
only when strictly improved over the baseline (`19.0971 dB` hero / `19.4163 dB` orbit / `0.67579` SSIM).
"""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
import os
import time
from typing import Any, Dict

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
    build_wide_bvh64_uv,
    pack_dev_arrays,
    render_wide_bvh64_uv_stage1,
)


def _train_classroom_w128(
    feats_tr: np.ndarray,
    s1_tr: np.ndarray,
    gt_tr: np.ndarray,
    tr_idx: np.ndarray,
    cal_idx: np.ndarray,
    te_idx: np.ndarray,
    feats_hero: np.ndarray,
    s1_hero: np.ndarray,
    gt_hero: np.ndarray,
    epochs: int = 400,
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
      print(f"[classroom] ep {ep+1}/{epochs} loss={el/steps_per_epoch:.5f} ({time.time()-t0:.1f}s)", flush=True)
  train_s = time.time() - t0

  pred_cal = np.asarray(infer_fn(ema_params, jnp.asarray(feats_tr[cal_idx]), jnp.asarray(s1_tr[cal_idx])))
  cal_scale = float(np.mean(luminance(gt_tr[cal_idx])) / max(np.mean(luminance(pred_cal)), 1e-6))

  pred_te = np.clip(np.asarray(infer_fn(ema_params, jnp.asarray(feats_tr[te_idx]), jnp.asarray(s1_tr[te_idx]))) * cal_scale, 0.0, 1.0)
  y_te = gt_tr[te_idx]
  s1_te = s1_tr[te_idx]

  pred_hero = np.clip(np.asarray(infer_fn(ema_params, jnp.asarray(feats_hero), jnp.asarray(s1_hero))) * cal_scale, 0.0, 1.0)

  metrics = {
      "arch": f"IrradianceNRC3S_Demod_W{base_width}_27chSh2",
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


def _benchmark_classroom(
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
      gb, rgh, mc, vis1 = render_wide_bvh64_uv_stage1(spec_r, da, cp, cm, interpret=False, shadow_top2=True, return_sh2=True)
      if neural_model is not None and npr is not None:
        ft = build_scene_gbuffer_features(gb, rgh, mc, sc_ctr, sc_scl, zs, extra_sh=vis1)
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
      gb, rgh, mc, vis1 = render_wide_bvh64_uv_stage1(spec_r, da, cp, cm, interpret=False, shadow_top2=True, return_sh2=True)
      if neural_model is not None and npr is not None:
        ft = build_scene_gbuffer_features(gb, rgh, mc, sc_ctr, sc_scl, zs, extra_sh=vis1)
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


def render_classroom_row(
    s1: np.ndarray,
    s2: np.ndarray,
    gt: np.ndarray,
    zd: np.ndarray,
    rec: Dict[str, Any],
) -> Image.Image:
  pw, ph = 512, 320
  pad, hdr = 12, 42
  n_cols = 5
  cw = n_cols * pw + (n_cols + 1) * pad
  rh = ph + hdr + pad
  row_im = Image.new("RGB", (cw, rh + pad), (15, 17, 22))
  draw = ImageDraw.Draw(row_im)

  met = rec["neural_relighting"]
  hr = met["hero_official_plus_orbit"]
  orb = met["orbit_test"]
  b4_s2 = next((b for b in rec["benchmarks"] if b["n_chips"] == 4 and b["stage2"] and b["resolution"].startswith(str(s2.shape[1]))), rec["benchmarks"][-1])

  err5 = _colormap_rgb(np.mean(np.clip(np.abs(s2 - gt) * 5.0, 0.0, 1.0), axis=-1), "inferno")
  fg = zd > 0.0
  if np.any(fg):
    z_lo, z_hi = np.percentile(zd[fg], [2, 98])
    zd_norm = np.where(fg, 1.0 - np.clip((zd - z_lo) / max(z_hi - z_lo, 0.5), 0.0, 1.0), 0.0)
  else:
    zd_norm = np.zeros_like(zd)
  zd_vis = _colormap_rgb(zd_norm, "viridis")

  tris = rec["scene_meta"]["tris"]
  z_mae_m = rec["parity"]["hero_z_mae_m"]
  panels = [
      (s1, f"classroom ({tris/1e6:.2f}M) TPU S1 Sh2 | {hr['view0_stage1_psnr_db']:.2f} dB"),
      (s2, f"TPU S2 NRC W128: {hr['view0_neural_psnr_db']:.2f} dB (SSIM {hr['view0_ssim']:.4f}) | Orbit {orb['neural_psnr_db']:.2f} dB | {b4_s2['img_per_sec']:.1f} img/s"),
      (gt, f"Cycles 5.3.0 GT ({gt.shape[1]}x{gt.shape[0]}) | Lum {hr['view0_lum_rel']*100:+.2f}%"),
      (err5, f"Error x5 | MS-SSIM {hr['view0_msssim']:.4f} | 0 callbacks"),
      (zd_vis, f"Exact Z-Depth | Z-MAE {z_mae_m*1e3:.2f} mm vs EXR"),
  ]
  for c_i, (img, title) in enumerate(panels):
    x0 = pad + c_i * (pw + pad)
    u8 = (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    pil_im = Image.fromarray(u8).resize((pw, ph), Image.Resampling.BILINEAR)
    row_im.paste(pil_im, (x0, pad + hdr))
    draw.text((x0 + 4, pad + 14), title, fill=(238, 244, 250))
  return row_im


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--data_dir", default="/dev/shm/zaychik-tpu-mjx-cache/classroom_512")
  ap.add_argument("--epochs", type=int, default=650)
  ap.add_argument("--base_width", type=int, default=128)
  ap.add_argument("--out_json", default="/dev/shm/zaychik-tpu-mjx-w5/v3_w5_classroom_upgrade.json")
  ap.add_argument("--out_png", default="/dev/shm/zaychik-tpu-mjx-w5/v3_w5_classroom_upgrade.png")
  ap.add_argument("--opendata_json", default=None)
  ap.add_argument("--opendata_png", default=None)
  args = ap.parse_args()

  pack = ScenePack.load(os.path.join(args.data_dir, "classroom_pack.npz"))
  orb = np.load(os.path.join(args.data_dir, "classroom_gt_orbit.npz"))
  hero = np.load(os.path.join(args.data_dir, "classroom_gt_hero.npz"))

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
    gb, rgh, mc, vis1 = render_wide_bvh64_uv_stage1(spec_tr, da, cp, cm, interpret=False, shadow_top2=True, return_sh2=True)
    ft = build_scene_gbuffer_features(gb, rgh, mc, sc_ctr, sc_scl, zs, extra_sh=vis1)
    return gb.rgb, gb.z_depth, ft

  @jax.jit
  def _render_gb_hr(da, cp, cm):
    gb, rgh, mc, vis1 = render_wide_bvh64_uv_stage1(spec_hr, da, cp, cm, interpret=False, shadow_top2=True, return_sh2=True)
    ft = build_scene_gbuffer_features(gb, rgh, mc, sc_ctr, sc_scl, zs, extra_sh=vis1)
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
  fg_match = float(np.mean(cy_fg == tp_fg))
  print(f"[classroom] Hero {hr_w}x{hr_h} Z-MAE={z_mae_m*1e3:.3f} mm, fg_match={fg_match:.5f}, hlo={hlo_tr}", flush=True)

  gt_tr = orb["rgb"].astype(np.float32) / 255.0
  gt_hr = hero["rgb"].astype(np.float32) / 255.0

  all_idx = np.arange(N_orb)
  if N_orb >= 48:
    cal_idx = np.array([3, 15, 27, 39], dtype=np.int32)
    te_idx = np.array([5, 11, 17, 23, 29, 35, 41, 47], dtype=np.int32)
    bs_tr = 6
  else:
    cal_idx = np.array([3, 15], dtype=np.int32)
    te_idx = np.array([5, 11, 17, 23], dtype=np.int32)
    bs_tr = 4
  tr_idx = np.array([i for i in all_idx if i not in set(cal_idx.tolist() + te_idx.tolist())], dtype=np.int32)

  mod, met, ema_params, cal_scale, pred_hr = _train_classroom_w128(
      ft_tr, s1_tr, gt_tr, tr_idx, cal_idx, te_idx, ft_hr, s1_hr, gt_hr,
      epochs=args.epochs, batch_size=bs_tr, lr=2.5e-3, base_width=args.base_width,
  )
  print("[classroom] EVAL:", json.dumps(met, indent=2), flush=True)

  n_devs = len(jax.devices())
  benches = []
  for nc in ([1, 4] if n_devs >= 4 else [1]):
    bs_hr = 4 if nc == 4 else 1
    benches.append(_benchmark_classroom(spec_tr, hero["cam_xpos"], hero["cam_xmat"], hr_w, hr_h, bs_hr, nc, None, None, 1.0))
    benches.append(_benchmark_classroom(spec_tr, hero["cam_xpos"], hero["cam_xmat"], hr_w, hr_h, bs_hr, nc, mod, ema_params, cal_scale))
  if n_devs >= 4:
    benches.append(_benchmark_classroom(spec_tr, hero["cam_xpos"], hero["cam_xmat"], 1920, 1088, 4, 4, None, None, 1.0, n_steps=3))
    benches.append(_benchmark_classroom(spec_tr, hero["cam_xpos"], hero["cam_xmat"], 1920, 1088, 4, 4, mod, ema_params, cal_scale, n_steps=3))

  rec = {
      "scene_meta": pack.summary(),
      "parity": {
          "hero_z_mae_m": z_mae_m,
          "hero_fg_match": fg_match,
          "hlo": hlo_tr,
      },
      "neural_relighting": met,
      "benchmarks": benches,
  }
  with open(args.out_json, "w") as f:
    json.dump(rec, f, indent=2)
  print("SAVED CLASSROOM JSON:", args.out_json, flush=True)

  row_im = render_classroom_row(
      np.clip(s1_hr[0], 0.0, 1.0),
      np.clip(pred_hr[0], 0.0, 1.0),
      np.clip(gt_hr[0], 0.0, 1.0),
      z_hr[0],
      rec,
  )
  row_im.save(args.out_png)
  print("SAVED CLASSROOM ROW PNG:", args.out_png, row_im.size, flush=True)

  # Safely patch opendata json/png only if strictly improved
  if args.opendata_json and os.path.exists(args.opendata_json):
    od = json.load(open(args.opendata_json))
    old_hr = od["scenes"]["classroom"]["neural_relighting"]["hero_official_plus_orbit"]
    new_hr = met["hero_official_plus_orbit"]
    improved = (
        new_hr["view0_neural_psnr_db"] > old_hr["view0_neural_psnr_db"] + 1.0
        and new_hr["view0_ssim"] > old_hr["view0_ssim"] + 0.02
    )
    print(f"Classroom vs old: PSNR {old_hr['view0_neural_psnr_db']} -> {new_hr['view0_neural_psnr_db']}, SSIM {old_hr['view0_ssim']} -> {new_hr['view0_ssim']}, improved={improved}", flush=True)
    if improved:
      od["scenes"]["classroom"] = rec
      with open(args.opendata_json, "w") as f:
        json.dump(od, f, indent=2)
      print("UPDATED OPENDATA JSON:", args.opendata_json, flush=True)
      if args.opendata_png and os.path.exists(args.opendata_png):
        base_im = Image.open(args.opendata_png).convert("RGB")
        # Row 0 occupies top (12 + 320 + 42) = 374 px; paste cleanly at (0, 0)
        crop_row = row_im.crop((0, 0, row_im.width, 374))
        base_im.paste(crop_row, (0, 0))
        base_im.save(args.opendata_png)
        print("UPDATED OPENDATA PNG:", args.opendata_png, base_im.size, flush=True)


if __name__ == "__main__":
  main()
