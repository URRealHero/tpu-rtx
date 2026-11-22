"""Run V3 101.63M-Triangle (100M+) Blender 5.2 AgX Showcase, Multi-Scale Relighter, 1c/4c & Native 4K UHD on live w5 TPU."""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
import os
import pickle
import re
import time
from typing import Any, Dict

import jax
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
import numpy as np
import optax

from gate import compute_msssim_batch, compute_psnr, compute_ssim_batch, luminance
from run_w5_showcase import SpatialRelighter3Scale, _hlo_audit, _ssim_loss
from tpu_10m_render import (
    InstancedBVHSceneSpec,
    build_instanced_bvh_scene_spec,
    render_instanced_bvh_batch,
)
from tpu_mjx_render.neural_shader import SpatialRelighter, apply_neural_shader, build_gbuffer_features


def _pack_inst_spec_arrays(spec: InstancedBVHSceneSpec):
  return (
      [(m.root_box, m.leaf_boxes, m.leaf_tris) for m in spec.meshes],
      spec.tex_pack,
      spec.metallic_tbl,
  )


def _unpack_inst_spec(spec_meta: InstancedBVHSceneSpec, packed) -> InstancedBVHSceneSpec:
  m_arrs, tex_pack, metallic_tbl = packed
  new_meshes = tuple(
      dataclasses.replace(m, root_box=rb, leaf_boxes=lb, leaf_tris=lt)
      for m, (rb, lb, lt) in zip(spec_meta.meshes, m_arrs)
  )
  return dataclasses.replace(spec_meta, meshes=new_meshes, tex_pack=tex_pack, metallic_tbl=metallic_tbl)


def benchmark_100m(
    spec_meta: InstancedBVHSceneSpec,
    spec_arrs,
    inst_xpos_ord: np.ndarray,
    inst_xmat_ord: np.ndarray,
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
  spec_r = dataclasses.replace(spec_meta, height=res_h, width=res_w)
  devs = jax.devices()[:n_chips]
  idx = np.arange(batch_size) % len(cam_xpos)
  oxp = jnp.asarray(inst_xpos_ord[idx])
  oxm = jnp.asarray(inst_xmat_ord[idx])
  cxp = jnp.asarray(cam_xpos[idx])
  cxm = jnp.asarray(cam_xmat[idx])

  if n_chips == 1:
    oxp, oxm, cxp, cxm = [jax.device_put(x, devs[0]) for x in (oxp, oxm, cxp, cxm)]
    sa = jax.tree_util.tree_map(lambda x: jax.device_put(x, devs[0]), spec_arrs)
    np_dev = jax.tree_util.tree_map(lambda x: jax.device_put(x, devs[0]), neural_params) if neural_params is not None else None

    @jax.jit
    def run_fn(sa_in, p, m, cp, cm, npr):
      sp = _unpack_inst_spec(spec_r, sa_in)
      gb = render_instanced_bvh_batch(sp, p, m, cp, cm, mode="pallas_bvh", interpret=False)
      if neural_model is not None and npr is not None:
        rgb = apply_neural_shader(neural_model, npr, gb, cal_scale=cal_scale)
      else:
        rgb = gb.rgb
      return rgb, gb.z_depth

    hlo_info = _hlo_audit(run_fn.lower(sa, oxp, oxm, cxp, cxm, np_dev).as_text())
    assert hlo_info["host_callbacks"] == 0 and hlo_info["send_recv_ops"] == 0, hlo_info

    for _ in range(n_warmup):
      r, z = run_fn(sa, oxp, oxm, cxp, cxm, np_dev)
    jax.block_until_ready((r, z))
    t0 = time.perf_counter()
    for _ in range(n_steps):
      r, z = run_fn(sa, oxp, oxm, cxp, cxm, np_dev)
    jax.block_until_ready((r, z))
    dt = (time.perf_counter() - t0) / n_steps
  else:
    assert batch_size % n_chips == 0
    mesh = jax.sharding.Mesh(np.array(devs), ("dp",))
    P = jax.sharding.PartitionSpec
    sharding = jax.sharding.NamedSharding(mesh, P("dp"))
    replicate = jax.sharding.NamedSharding(mesh, P())
    oxp, oxm, cxp, cxm = [jax.device_put(x, sharding) for x in (oxp, oxm, cxp, cxm)]
    sa = jax.tree_util.tree_map(lambda x: jax.device_put(x, replicate), spec_arrs)
    np_dev = jax.tree_util.tree_map(lambda x: jax.device_put(x, replicate), neural_params) if neural_params is not None else None
    sm_kw = {"check_rep": False} if "check_rep" in inspect.signature(shard_map).parameters else {"check_vma": False}

    def _local(sa_in, p, m, cp, cm, npr):
      sp = _unpack_inst_spec(spec_r, sa_in)
      gb = render_instanced_bvh_batch(sp, p, m, cp, cm, mode="pallas_bvh", interpret=False)
      if neural_model is not None and npr is not None:
        rgb = apply_neural_shader(neural_model, npr, gb, cal_scale=cal_scale)
      else:
        rgb = gb.rgb
      return rgb, gb.z_depth

    _sm = shard_map(
        _local,
        mesh=mesh,
        in_specs=(P(), P("dp"), P("dp"), P("dp"), P("dp"), P()),
        out_specs=(P("dp"), P("dp")),
        **sm_kw,
    )

    @jax.jit
    def run_fn(sa_in, p, m, cp, cm, npr):
      return _sm(sa_in, p, m, cp, cm, npr)

    hlo_info = _hlo_audit(run_fn.lower(sa, oxp, oxm, cxp, cxm, np_dev).as_text())
    assert hlo_info["host_callbacks"] == 0 and hlo_info["send_recv_ops"] == 0, hlo_info

    for _ in range(n_warmup):
      r, z = run_fn(sa, oxp, oxm, cxp, cxm, np_dev)
    jax.block_until_ready((r, z))
    t0 = time.perf_counter()
    for _ in range(n_steps):
      r, z = run_fn(sa, oxp, oxm, cxp, cxm, np_dev)
    jax.block_until_ready((r, z))
    dt = (time.perf_counter() - t0) / n_steps

  return {
      "resolution": f"{res_w}x{res_h}",
      "batch_size": batch_size,
      "n_chips": n_chips,
      "stage2": neural_model is not None,
      "dt_ms": round(dt * 1e3, 3),
      "fps": round(batch_size / max(dt, 1e-9), 3),
      "mpix_per_sec": round((batch_size * res_w * res_h) / max(dt, 1e-9) / 1e6, 2),
      "host_callbacks": hlo_info["host_callbacks"],
      "send_recv_ops": hlo_info["send_recv_ops"],
  }


def train_multiscale_relighter(
    arch_name: str,
    model,
    feats_256: np.ndarray,
    s1_256: np.ndarray,
    gt_256: np.ndarray,
    tr_idx: np.ndarray,
    cal_idx: np.ndarray,
    te_idx: np.ndarray,
    feats_512: np.ndarray,
    s1_512: np.ndarray,
    gt_512: np.ndarray,
    epochs: int = 240,
    batch_size: int = 9,
    lr: float = 2.5e-3,
):
  rng = jax.random.PRNGKey(0)
  params = model.init(rng, jnp.asarray(feats_256[:1]))["params"]
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

  x_tr = jnp.asarray(feats_256[tr_idx])
  b_tr = jnp.asarray(s1_256[tr_idx])
  y_tr = jnp.asarray(gt_256[tr_idx])

  @jax.jit
  def train_step(p, ema_p, opt_st, idx_s, crop_key, use_up2x):
    xb, bb, yb = x_tr[idx_s], b_tr[idx_s], y_tr[idx_s]

    def _upsample_crop(vals):
      x_i, b_i, y_i = vals
      B = x_i.shape[0]
      x_up = jax.image.resize(x_i, (B, 512, 512, x_i.shape[-1]), method="bilinear")
      b_up = jax.image.resize(b_i, (B, 512, 512, 3), method="bilinear")
      y_up = jax.image.resize(y_i, (B, 512, 512, 3), method="bilinear")
      ky, kx = jax.random.split(crop_key)
      oy = jax.random.randint(ky, (), 0, 256)
      ox = jax.random.randint(kx, (), 0, 256)
      return (
          jax.lax.dynamic_slice(x_up, (0, oy, ox, 0), (B, 256, 256, x_i.shape[-1])),
          jax.lax.dynamic_slice(b_up, (0, oy, ox, 0), (B, 256, 256, 3)),
          jax.lax.dynamic_slice(y_up, (0, oy, ox, 0), (B, 256, 256, 3)),
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
  def infer_256(p, xf, bf):
    return jnp.clip(bf + model.apply({"params": p}, xf, train=False), 0.0, 1.0)

  @jax.jit
  def infer_512(p, xf, bf):
    return jnp.clip(bf + model.apply({"params": p}, xf, train=False), 0.0, 1.0)

  _p, _ep, _os, _l = train_step(params, ema_params, opt_state, jnp.arange(batch_size), jax.random.PRNGKey(1), jnp.bool_(False))
  jax.block_until_ready(_l)

  t_tr = time.time()
  np_rng = np.random.default_rng(42)
  step_i = 0
  for ep in range(epochs):
    perm = np_rng.permutation(len(tr_idx))
    el = 0.0
    for s in range(steps_per_epoch):
      sl = perm[s * batch_size : (s + 1) * batch_size]
      up2x = (step_i % 3 == 1)  # 33% 2x-upsampled 512x512 scale crops + 67% native 256x256
      params, ema_params, opt_state, loss = train_step(
          params, ema_params, opt_state, jnp.asarray(sl), jax.random.PRNGKey(step_i + 100), jnp.bool_(up2x)
      )
      el += float(loss)
      step_i += 1
    if (ep + 1) % 60 == 0 or ep + 1 == epochs:
      print(f"[{arch_name}] epoch {ep+1}/{epochs} loss={el/steps_per_epoch:.5f} ({time.time()-t_tr:.1f}s)", flush=True)
  train_s = time.time() - t_tr

  pred_cal = np.asarray(infer_256(ema_params, jnp.asarray(feats_256[cal_idx]), jnp.asarray(s1_256[cal_idx])))
  cal_scale = float(np.mean(luminance(gt_256[cal_idx])) / max(np.mean(luminance(pred_cal)), 1e-6))

  pred_te_256 = np.clip(np.asarray(infer_256(ema_params, jnp.asarray(feats_256[te_idx]), jnp.asarray(s1_256[te_idx]))) * cal_scale, 0.0, 1.0)
  y_te_256 = gt_256[te_idx]
  s1_te_256 = s1_256[te_idx]
  pred_512 = np.clip(np.asarray(infer_512(ema_params, jnp.asarray(feats_512), jnp.asarray(s1_512))) * cal_scale, 0.0, 1.0)

  metrics = {
      "arch": arch_name,
      "n_params": int(sum(x.size for x in jax.tree_util.tree_leaves(ema_params))),
      "epochs": epochs,
      "train_time_s": round(train_s, 2),
      "cal_scale": round(cal_scale, 6),
      "test_256": {
          "stage1_psnr_db": round(compute_psnr(s1_te_256, y_te_256), 4),
          "neural_psnr_db": round(compute_psnr(pred_te_256, y_te_256), 4),
          "psnr_gain_db": round(compute_psnr(pred_te_256, y_te_256) - compute_psnr(s1_te_256, y_te_256), 4),
          "neural_ssim": round(compute_ssim_batch(pred_te_256, y_te_256), 5),
          "neural_msssim": round(compute_msssim_batch(pred_te_256, y_te_256), 5),
          "lum_rel": round(float(np.mean(luminance(pred_te_256)) / max(np.mean(luminance(y_te_256)), 1e-8) - 1.0), 6),
      },
      "hero_512": {
          "stage1_psnr_db": round(compute_psnr(s1_512, gt_512), 4),
          "neural_psnr_db": round(compute_psnr(pred_512, gt_512), 4),
          "psnr_gain_db": round(compute_psnr(pred_512, gt_512) - compute_psnr(s1_512, gt_512), 4),
          "neural_ssim": round(compute_ssim_batch(pred_512, gt_512), 5),
          "neural_msssim": round(compute_msssim_batch(pred_512, gt_512), 5),
          "lum_rel": round(float(np.mean(luminance(pred_512)) / max(np.mean(luminance(gt_512)), 1e-8) - 1.0), 6),
      },
  }
  return metrics, ema_params, cal_scale, pred_te_256, pred_512


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--work_dir", default="/dev/shm/zaychik-tpu-mjx-w5")
  ap.add_argument("--stage", choices=["gb_256", "gb_512", "train", "uhd4k", "bench1c", "bench4c"], required=True)
  ap.add_argument("--epochs", type=int, default=240)
  args = ap.parse_args()

  scene_npz = os.path.join(args.work_dir, "v3_w5_blender_100m_scene.npz")
  gt256_npz = os.path.join(args.work_dir, "v3_w5_blender_100m_gt_256.npz")
  gt512_npz = os.path.join(args.work_dir, "v3_w5_blender_100m_gt_512.npz")
  gt4k_npz = os.path.join(args.work_dir, "v3_w5_blender_100m_gt_4k.npz")
  gb256_cache = os.path.join(args.work_dir, "v3_w5_100m_gb256.npz")
  gb512_cache = os.path.join(args.work_dir, "v3_w5_100m_gb512.npz")

  ds = dict(np.load(scene_npz))
  gt256 = dict(np.load(gt256_npz))
  gt512 = dict(np.load(gt512_npz))

  N_256 = len(ds["split_256"])
  N_hero = len(gt512["rgb"])
  spec_256, perm = build_instanced_bvh_scene_spec(scene_npz, height=256, width=256)
  spec_512 = dataclasses.replace(spec_256, height=512, width=512)
  spec_arrs = _pack_inst_spec_arrays(spec_256)

  inst_xpos_ord = ds["inst_xpos"][:, perm, :]
  inst_xmat_ord = ds["inst_xmat"][:, perm, :, :]
  cam_xpos = ds["cam_xpos"]
  cam_xmat = ds["cam_xmat"]

  out_json_path = os.path.join(args.work_dir, "v3_w5_blender_100m_showcase.json")
  summary: Dict[str, Any] = json.load(open(out_json_path)) if os.path.exists(out_json_path) else {}
  summary["scene_meta"] = {
      "unique_tris": int(ds["unique_tris"]),
      "effective_tris": int(ds["effective_tris"]),
      "n_unique_blas": int(ds["n_meshes"]),
      "n_instances": int(ds["n_instances"]),
      "blas_tris": {m.name: m.n_tris_orig for m in spec_256.meshes},
      "blas_instances": {m.name: c for m, c in zip(spec_256.meshes, spec_256.inst_counts)},
      "device_kind": jax.devices()[0].device_kind,
      "n_visible_devices": len(jax.devices()),
  }

  if args.stage == "gb_256":
    t_gb = time.time()
    fn_b16 = jax.jit(lambda p, m, cp, cm: render_instanced_bvh_batch(spec_256, p, m, cp, cm, mode="pallas_bvh", interpret=False))
    hlo_256 = _hlo_audit(fn_b16.lower(jnp.asarray(inst_xpos_ord[:16]), jnp.asarray(inst_xmat_ord[:16]), jnp.asarray(cam_xpos[:16]), jnp.asarray(cam_xmat[:16])).as_text())
    assert hlo_256["host_callbacks"] == 0 and hlo_256["send_recv_ops"] == 0, hlo_256

    gb_256_list = []
    for st in range(0, N_256, 16):
      en = min(N_256, st + 16)
      gb_c = fn_b16(jnp.asarray(inst_xpos_ord[st:en]), jnp.asarray(inst_xmat_ord[st:en]), jnp.asarray(cam_xpos[st:en]), jnp.asarray(cam_xmat[st:en]))
      gb_256_list.append(jax.device_get(gb_c))
    feats_256 = np.concatenate([np.asarray(build_gbuffer_features(g), np.float32) for g in gb_256_list], axis=0)
    s1_256 = np.concatenate([np.asarray(g.rgb, np.float32) for g in gb_256_list], axis=0)
    z_256 = np.concatenate([np.asarray(g.z_depth, np.float32) for g in gb_256_list], axis=0)

    cy256 = np.asarray(gt256["frame0_z_depth"], np.float32)
    tp256 = z_256[0]
    b256 = (cy256 < 20.0) & (tp256 > 0.0)
    par = summary.get("parity", {})
    par.update({
        "pallas_vs_cycles_256_z_mae_m": float(np.mean(np.abs(cy256[b256] - tp256[b256]))),
        "pallas_vs_cycles_256_fg_match": float(np.mean((cy256 < 20.0) == (tp256 > 0.0))),
        "hlo_256_host_callbacks": hlo_256["host_callbacks"],
        "hlo_256_send_recv_ops": hlo_256["send_recv_ops"],
    })
    print("256x256 PARITY (101.63M tris):", json.dumps(par, indent=2), flush=True)
    assert par["pallas_vs_cycles_256_z_mae_m"] < 1e-4
    summary["parity"] = par
    print(f"Rendered 48x256 (101.63M tris) in {time.time()-t_gb:.2f}s", flush=True)
    np.savez(gb256_cache, feats_256=feats_256.astype(np.float16), s1_256=s1_256, z_256=z_256)

  elif args.stage == "gb_512":
    t_gb = time.time()
    fn_b4 = jax.jit(lambda p, m, cp, cm: render_instanced_bvh_batch(spec_512, p, m, cp, cm, mode="pallas_bvh", interpret=False))
    hlo_512 = _hlo_audit(fn_b4.lower(jnp.asarray(inst_xpos_ord[N_256 : N_256 + N_hero]), jnp.asarray(inst_xmat_ord[N_256 : N_256 + N_hero]), jnp.asarray(cam_xpos[N_256 : N_256 + N_hero]), jnp.asarray(cam_xmat[N_256 : N_256 + N_hero])).as_text())
    assert hlo_512["host_callbacks"] == 0 and hlo_512["send_recv_ops"] == 0, hlo_512

    gb_512_all = fn_b4(
        jnp.asarray(inst_xpos_ord[N_256 : N_256 + N_hero]),
        jnp.asarray(inst_xmat_ord[N_256 : N_256 + N_hero]),
        jnp.asarray(cam_xpos[N_256 : N_256 + N_hero]),
        jnp.asarray(cam_xmat[N_256 : N_256 + N_hero]),
    )
    feats_512 = np.asarray(build_gbuffer_features(gb_512_all), np.float32)
    s1_512 = np.asarray(gb_512_all.rgb, np.float32)
    z_512 = np.asarray(gb_512_all.z_depth, np.float32)

    cy512 = np.asarray(gt512["frame0_z_depth"], np.float32)
    tp512 = z_512[0]
    b512 = (cy512 < 20.0) & (tp512 > 0.0)
    par = summary.get("parity", {})
    par.update({
        "pallas_vs_cycles_512_z_mae_m": float(np.mean(np.abs(cy512[b512] - tp512[b512]))),
        "pallas_vs_cycles_512_fg_match": float(np.mean((cy512 < 20.0) == (tp512 > 0.0))),
        "hlo_512_host_callbacks": hlo_512["host_callbacks"],
        "hlo_512_send_recv_ops": hlo_512["send_recv_ops"],
    })
    print("512x512 PARITY (101.63M tris):", json.dumps(par, indent=2), flush=True)
    assert par["pallas_vs_cycles_512_z_mae_m"] < 1e-4
    summary["parity"] = par
    print(f"Rendered 4x512 (101.63M tris) in {time.time()-t_gb:.2f}s", flush=True)
    np.savez(gb512_cache, feats_512=feats_512.astype(np.float16), s1_512=s1_512, z_512=z_512)

  elif args.stage == "train":
    g2 = np.load(gb256_cache)
    g5 = np.load(gb512_cache)
    feats_256, s1_256, z_256 = g2["feats_256"].astype(np.float32), g2["s1_256"].astype(np.float32), g2["z_256"].astype(np.float32)
    feats_512, s1_512, z_512 = g5["feats_512"].astype(np.float32), g5["s1_512"].astype(np.float32), g5["z_512"].astype(np.float32)

    split = ds["split_256"]
    tr_idx = np.where(split == 0)[0]
    cal_idx = np.where(split == 1)[0]
    te_idx = np.where(split == 2)[0]
    gt_256_rgb = gt256["rgb"].astype(np.float32)
    gt_512_rgb = gt512["rgb"].astype(np.float32)

    mod = SpatialRelighter3Scale(base_width=96)
    met, ep, cs, p256, p512 = train_multiscale_relighter(
        "SpatialRelighter3S_W96_MS",
        mod,
        feats_256, s1_256, gt_256_rgb, tr_idx, cal_idx, te_idx,
        feats_512, s1_512, gt_512_rgb,
        epochs=args.epochs, batch_size=9, lr=2.5e-3,
    )
    print("100M+ EVAL:", json.dumps(met, indent=2), flush=True)
    summary["neural_relighting"] = met

    np.savez_compressed(
        os.path.join(args.work_dir, "v3_w5_100m_preds_512.npz"),
        s1_rgb=s1_512,
        pred_rgb=p512,
        gt_rgb=gt_512_rgb,
        s1_z_depth=z_512,
    )
    np.savez_compressed(
        os.path.join(args.work_dir, "v3_w5_100m_preds_256.npz"),
        test_idx=te_idx,
        s1_rgb=s1_256[te_idx],
        pred_rgb=p256,
        gt_rgb=gt_256_rgb[te_idx],
        s1_z_depth=z_256[te_idx],
    )
    with open(os.path.join(args.work_dir, "v3_w5_100m_relighter.pkl"), "wb") as f:
      pickle.dump({"arch": "SpatialRelighter3S_W96_MS", "cal_scale": cs, "params": jax.tree_util.tree_map(np.asarray, ep)}, f)

  elif args.stage == "uhd4k":
    ck = pickle.load(open(os.path.join(args.work_dir, "v3_w5_100m_relighter.pkl"), "rb"))
    cs = float(ck["cal_scale"])
    ep = jax.tree_util.tree_map(jnp.asarray, ck["params"])
    mod = SpatialRelighter3Scale(base_width=96)

    uhd_idx = N_256 + N_hero
    spec_4k = dataclasses.replace(spec_256, height=2160, width=3840)

    @jax.jit
    def render_4k_s1_s2(p, m, cp, cm, npr):
      gb = render_instanced_bvh_batch(spec_4k, p, m, cp, cm, mode="pallas_bvh", interpret=False)
      rgb_s2 = apply_neural_shader(mod, npr, gb, cal_scale=cs)
      return gb.rgb, rgb_s2, gb.z_depth

    oxp = jnp.asarray(inst_xpos_ord[uhd_idx : uhd_idx + 1])
    oxm = jnp.asarray(inst_xmat_ord[uhd_idx : uhd_idx + 1])
    cxp = jnp.asarray(cam_xpos[uhd_idx : uhd_idx + 1])
    cxm = jnp.asarray(cam_xmat[uhd_idx : uhd_idx + 1])

    hlo_4k = _hlo_audit(render_4k_s1_s2.lower(oxp, oxm, cxp, cxm, ep).as_text())
    assert hlo_4k["host_callbacks"] == 0 and hlo_4k["send_recv_ops"] == 0, hlo_4k
    print("4K UHD HLO audit:", hlo_4k, flush=True)

    s1_4k, s2_4k, z_4k = render_4k_s1_s2(oxp, oxm, cxp, cxm, ep)
    jax.block_until_ready((s1_4k, s2_4k, z_4k))

    s1_4k_np = np.asarray(s1_4k[0], np.float32)
    s2_4k_np = np.asarray(s2_4k[0], np.float32)
    z_4k_np = np.asarray(z_4k[0], np.float32)

    uhd_metrics: Dict[str, Any] = {"resolution": "3840x2160", "hlo": hlo_4k}
    if os.path.exists(gt4k_npz):
      gt4k = np.load(gt4k_npz)
      gt4k_rgb = gt4k["rgb"].astype(np.float32) / 255.0
      cy4k_z = np.asarray(gt4k["frame0_z_depth"], np.float32)
      b4k = (cy4k_z < 20.0) & (z_4k_np > 0.0)
      z_mae_4k = float(np.mean(np.abs(cy4k_z[b4k] - z_4k_np[b4k])))
      fg_4k = float(np.mean((cy4k_z < 20.0) == (z_4k_np > 0.0)))
      psnr_s1_4k = compute_psnr(s1_4k_np[None], gt4k_rgb[None])
      psnr_s2_4k = compute_psnr(s2_4k_np[None], gt4k_rgb[None])
      ssim_4k = compute_ssim_batch(s2_4k_np[None], gt4k_rgb[None])
      msssim_4k = compute_msssim_batch(s2_4k_np[None], gt4k_rgb[None])
      lum_4k = float(np.mean(luminance(s2_4k_np)) / max(np.mean(luminance(gt4k_rgb)), 1e-8) - 1.0)
      uhd_metrics.update({
          "z_mae_m": z_mae_4k,
          "fg_match": fg_4k,
          "stage1_psnr_db": round(psnr_s1_4k, 4),
          "neural_psnr_db": round(psnr_s2_4k, 4),
          "psnr_gain_db": round(psnr_s2_4k - psnr_s1_4k, 4),
          "neural_ssim": round(ssim_4k, 5),
          "neural_msssim": round(msssim_4k, 5),
          "lum_rel": round(lum_4k, 6),
      })
      np.savez_compressed(
          os.path.join(args.work_dir, "v3_w5_100m_4k_preds.npz"),
          s1_u8=(np.clip(s1_4k_np, 0, 1) * 255.0 + 0.5).astype(np.uint8),
          pred_u8=(np.clip(s2_4k_np, 0, 1) * 255.0 + 0.5).astype(np.uint8),
          gt_u8=gt4k["rgb"],
          z_depth=z_4k_np.astype(np.float16),
      )
    else:
      np.savez_compressed(
          os.path.join(args.work_dir, "v3_w5_100m_4k_preds.npz"),
          s1_u8=(np.clip(s1_4k_np, 0, 1) * 255.0 + 0.5).astype(np.uint8),
          pred_u8=(np.clip(s2_4k_np, 0, 1) * 255.0 + 0.5).astype(np.uint8),
          z_depth=z_4k_np.astype(np.float16),
      )
    print("4K UHD QUALITY:", json.dumps(uhd_metrics, indent=2), flush=True)
    summary["uhd_4k_quality"] = uhd_metrics

  elif args.stage == "bench1c":
    ck = pickle.load(open(os.path.join(args.work_dir, "v3_w5_100m_relighter.pkl"), "rb"))
    cs = float(ck["cal_scale"])
    ep = jax.tree_util.tree_map(jnp.asarray, ck["params"])
    mod = SpatialRelighter3Scale(base_width=96)

    b1c = []
    for rw, rh, bs, stp in [(256, 256, 16, 6), (512, 512, 4, 5), (3840, 2160, 1, 3)]:
      r1 = benchmark_100m(spec_256, spec_arrs, inst_xpos_ord, inst_xmat_ord, cam_xpos, cam_xmat, rw, rh, bs, n_chips=1, n_steps=stp)
      print("100M+ BENCH 1c S1:", json.dumps(r1), flush=True)
      b1c.append(r1)
    for rw, rh, bs, stp in [(256, 256, 16, 6), (512, 512, 4, 5), (3840, 2160, 1, 3)]:
      r2 = benchmark_100m(spec_256, spec_arrs, inst_xpos_ord, inst_xmat_ord, cam_xpos, cam_xmat, rw, rh, bs, n_chips=1, neural_model=mod, neural_params=ep, cal_scale=cs, n_steps=stp)
      print("100M+ BENCH 1c S1+S2:", json.dumps(r2), flush=True)
      b1c.append(r2)
    summary["benchmarks_1chip"] = b1c

  elif args.stage == "bench4c":
    assert len(jax.devices()) >= 4, f"Need >= 4 devices, got {len(jax.devices())}"
    ck = pickle.load(open(os.path.join(args.work_dir, "v3_w5_100m_relighter.pkl"), "rb"))
    cs = float(ck["cal_scale"])
    ep = jax.tree_util.tree_map(jnp.asarray, ck["params"])
    mod = SpatialRelighter3Scale(base_width=96)

    b4c = []
    for rw, rh, bs, stp in [(256, 256, 16, 6), (512, 512, 4, 5), (3840, 2160, 4, 3)]:
      r1 = benchmark_100m(spec_256, spec_arrs, inst_xpos_ord, inst_xmat_ord, cam_xpos, cam_xmat, rw, rh, bs, n_chips=4, n_steps=stp)
      print("100M+ BENCH 4c S1:", json.dumps(r1), flush=True)
      b4c.append(r1)
    for rw, rh, bs, stp in [(256, 256, 16, 6), (512, 512, 4, 5), (3840, 2160, 4, 3)]:
      r2 = benchmark_100m(spec_256, spec_arrs, inst_xpos_ord, inst_xmat_ord, cam_xpos, cam_xmat, rw, rh, bs, n_chips=4, neural_model=mod, neural_params=ep, cal_scale=cs, n_steps=stp)
      print("100M+ BENCH 4c S1+S2:", json.dumps(r2), flush=True)
      b4c.append(r2)
    summary["benchmarks_4chip_shard_map"] = b4c

  with open(out_json_path, "w") as f:
    json.dump(summary, f, indent=2)
  print("SAVED SUMMARY:", out_json_path, flush=True)


if __name__ == "__main__":
  main()
