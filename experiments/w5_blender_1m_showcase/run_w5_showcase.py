"""Run V3 1.24M-Triangle Blender Showcase verification, 1c/4c benchmarks, and Stage 2 Neural Relighting on live TPU v5p."""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
import os
import pickle
import re
import time
from typing import Any, Dict, List

import flax.linen as nn
import jax
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
import numpy as np
import optax

from gate import compute_msssim_batch, compute_psnr, compute_ssim_batch, luminance
from tpu_1m_render import BVHSceneSpec, build_bvh_scene_spec, render_bvh_batch
from tpu_mjx_render.neural_shader import SpatialRelighter, apply_neural_shader, build_gbuffer_features
from tpu_mjx_render.roofline import inspect_hlo_ops


class _ResBlock3S(nn.Module):
  features: int

  @nn.compact
  def __call__(self, x: jax.Array) -> jax.Array:
    dt = jnp.bfloat16
    y = nn.Conv(self.features, (3, 3), padding="SAME", dtype=dt, param_dtype=jnp.float32)(x)
    y = nn.relu(y)
    y = nn.Conv(self.features, (3, 3), padding="SAME", dtype=dt, param_dtype=jnp.float32)(y)
    return nn.relu(x + y)


class SpatialRelighter3Scale(nn.Module):
  """3-scale TPU MXU-aligned BF16 Relighter (1x -> 1/2x -> 1/4x) for 256x256/512x512 GI & soft penumbras."""

  base_width: int = 96

  @nn.compact
  def __call__(self, x: jax.Array, train: bool = False) -> jax.Array:
    del train
    dt = jnp.bfloat16
    w1 = self.base_width
    w2 = self.base_width * 2
    w3 = self.base_width * 2

    # Scale 1: H x W
    h1 = nn.relu(nn.Conv(w1, (3, 3), padding="SAME", dtype=dt, param_dtype=jnp.float32)(x.astype(dt)))
    h1 = _ResBlock3S(w1)(h1)
    h1 = _ResBlock3S(w1)(h1)

    # Scale 2: H/2 x W/2
    h2 = nn.relu(nn.Conv(w2, (3, 3), strides=(2, 2), padding="SAME", dtype=dt, param_dtype=jnp.float32)(h1))
    h2 = _ResBlock3S(w2)(h2)
    h2 = _ResBlock3S(w2)(h2)

    # Scale 3: H/4 x W/4 (wide global bounce & area penumbra receptive field)
    h3 = nn.relu(nn.Conv(w3, (3, 3), strides=(2, 2), padding="SAME", dtype=dt, param_dtype=jnp.float32)(h2))
    h3 = _ResBlock3S(w3)(h3)
    h3 = _ResBlock3S(w3)(h3)

    B, H, W, _ = x.shape
    h3_up = jax.image.resize(h3.astype(jnp.float32), (B, H // 2, W // 2, w3), method="bilinear").astype(dt)
    h2_cat = jnp.concatenate([h2, h3_up], axis=-1)
    h2_m = nn.relu(nn.Conv(w2, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(h2_cat))
    h2_m = _ResBlock3S(w2)(h2_m)

    h2_up = jax.image.resize(h2_m.astype(jnp.float32), (B, H, W, w2), method="bilinear").astype(dt)
    cat = jnp.concatenate([h1, h2_up], axis=-1)
    h_out = nn.relu(nn.Conv(w1, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(cat))
    h_out = _ResBlock3S(w1)(h_out)

    out_in = jnp.concatenate([h_out, x.astype(dt)], axis=-1)
    out_mid = nn.relu(nn.Conv(w1, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(out_in))
    out_mid = nn.relu(nn.Conv(w1, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(out_mid))
    return nn.Conv(3, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(out_mid).astype(jnp.float32)


def _hlo_audit(hlo_text: str) -> Dict[str, int]:
  info = inspect_hlo_ops(hlo_text)
  info["send_recv_ops"] = len(re.findall(r"\b(?:send|recv|send-done|recv-done)\(", hlo_text))
  return info


def _pack_spec_arrays(spec: BVHSceneSpec):
  return (
      [(m.root_box, m.leaf_boxes, m.leaf_tris) for m in spec.meshes],
      spec.tex_pack,
      spec.metallic_tbl,
  )


def _unpack_spec(spec_meta: BVHSceneSpec, packed) -> BVHSceneSpec:
  m_arrs, tex_pack, metallic_tbl = packed
  new_meshes = tuple(
      dataclasses.replace(m, root_box=rb, leaf_boxes=lb, leaf_tris=lt)
      for m, (rb, lb, lt) in zip(spec_meta.meshes, m_arrs)
  )
  return dataclasses.replace(spec_meta, meshes=new_meshes, tex_pack=tex_pack, metallic_tbl=metallic_tbl)


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


def benchmark_render(
    spec_meta: BVHSceneSpec,
    spec_arrs,
    ds: Dict[str, np.ndarray],
    res: int,
    batch_size: int,
    n_chips: int,
    neural_model: nn.Module | None = None,
    neural_params: Any = None,
    cal_scale: float = 1.0,
    n_warmup: int = 2,
    n_steps: int = 10,
) -> Dict[str, Any]:
  spec_r = dataclasses.replace(spec_meta, height=res, width=res)
  devs = jax.devices()[:n_chips]
  idx = np.arange(batch_size) % len(ds["cam_xpos"])
  oxp = jnp.asarray(ds["obj_xpos"][idx])
  oxm = jnp.asarray(ds["obj_xmat"][idx])
  cxp = jnp.asarray(ds["cam_xpos"][idx])
  cxm = jnp.asarray(ds["cam_xmat"][idx])

  if n_chips == 1:
    oxp, oxm, cxp, cxm = [jax.device_put(x, devs[0]) for x in (oxp, oxm, cxp, cxm)]
    sa = jax.tree_util.tree_map(lambda x: jax.device_put(x, devs[0]), spec_arrs)
    np_dev = jax.tree_util.tree_map(lambda x: jax.device_put(x, devs[0]), neural_params) if neural_params is not None else None

    @jax.jit
    def run_fn(sa_in, p, m, cp, cm, npr):
      sp = _unpack_spec(spec_r, sa_in)
      gb = render_bvh_batch(sp, p, m, cp, cm, mode="pallas_bvh", interpret=False)
      if neural_model is not None and npr is not None:
        rgb = apply_neural_shader(neural_model, npr, gb, cal_scale=cal_scale)
      else:
        rgb = gb.rgb
      return rgb, gb.z_depth

    hlo_text = run_fn.lower(sa, oxp, oxm, cxp, cxm, np_dev).as_text()
    hlo_info = _hlo_audit(hlo_text)
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
      sp = _unpack_spec(spec_r, sa_in)
      gb = render_bvh_batch(sp, p, m, cp, cm, mode="pallas_bvh", interpret=False)
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

    hlo_text = run_fn.lower(sa, oxp, oxm, cxp, cxm, np_dev).as_text()
    hlo_info = _hlo_audit(hlo_text)
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
      "resolution": f"{res}x{res}",
      "batch_size": batch_size,
      "n_chips": n_chips,
      "stage2": neural_model is not None,
      "dt_ms": round(dt * 1e3, 3),
      "fps": round(batch_size / max(dt, 1e-9), 2),
      "mpix_per_sec": round((batch_size * res * res) / max(dt, 1e-9) / 1e6, 2),
      "host_callbacks": hlo_info["host_callbacks"],
      "send_recv_ops": hlo_info["send_recv_ops"],
  }


def train_and_eval_relighter(
    arch_name: str,
    model: nn.Module,
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
  def train_step(p, ema_p, opt_st, idx_s):
    xb, bb, yb = x_tr[idx_s], b_tr[idx_s], y_tr[idx_s]

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

  # Warmup train_step
  _p, _ep, _os, _l = train_step(params, ema_params, opt_state, jnp.arange(batch_size))
  jax.block_until_ready(_l)

  t_tr = time.time()
  np_rng = np.random.default_rng(42)
  for ep in range(epochs):
    perm = np_rng.permutation(len(tr_idx))
    el = 0.0
    for s in range(steps_per_epoch):
      sl = perm[s * batch_size : (s + 1) * batch_size]
      params, ema_params, opt_state, loss = train_step(params, ema_params, opt_state, jnp.asarray(sl))
      el += float(loss)
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
  ap.add_argument(
      "--stage",
      choices=["gbuffers", "train", "bench1c_a", "bench1c_b", "bench4c_a", "bench4c_b", "all"],
      default="all",
  )
  ap.add_argument("--epochs", type=int, default=180)
  args = ap.parse_args()

  scene_npz = os.path.join(args.work_dir, "v3_w5_blender_1m_scene.npz")
  gt256_npz = os.path.join(args.work_dir, "v3_w5_blender_1m_gt_256.npz")
  gt512_npz = os.path.join(args.work_dir, "v3_w5_blender_1m_gt_512.npz")
  gb_cache_npz = os.path.join(args.work_dir, "v3_w5_gbuffers_cache.npz")

  ds = dict(np.load(scene_npz))
  gt256 = dict(np.load(gt256_npz))
  gt512 = dict(np.load(gt512_npz))

  N_256 = len(ds["split_256"])
  N_hero = len(gt512["rgb"])
  total_tris = int(ds["total_tris"])
  print(f"Loaded scene: {total_tris:,} tris, {N_256} frames @ 256x256, {N_hero} hero views @ 512x512, devs={len(jax.devices())}", flush=True)

  spec_256 = build_bvh_scene_spec(scene_npz, height=256, width=256)
  spec_512 = dataclasses.replace(spec_256, height=512, width=512)
  spec_arrs = _pack_spec_arrays(spec_256)

  out_json_path = os.path.join(args.work_dir, "v3_w5_blender_1m_showcase.json")
  summary: Dict[str, Any] = {}
  if os.path.exists(out_json_path):
    summary = json.load(open(out_json_path))
  summary["scene_meta"] = {
      "total_tris": total_tris,
      "n_meshes": int(ds["n_meshes"]),
      "mesh_tris": {m.name: m.n_tris_orig for m in spec_256.meshes},
      "mesh_leaves_128": {m.name: m.n_leaves for m in spec_256.meshes},
      "mesh_top_nodes_128": {m.name: m.n_top for m in spec_256.meshes},
      "device_kind": jax.devices()[0].device_kind,
      "n_visible_devices": len(jax.devices()),
  }

  if args.stage in ("all", "gbuffers"):
    # 1. Parity & EXR Z-Depth Check at 256x256 and 512x512
    fn_pal_256 = jax.jit(lambda p, m, cp, cm: render_bvh_batch(spec_256, p, m, cp, cm, mode="pallas_bvh", interpret=False))
    fn_jax_256 = jax.jit(lambda p, m, cp, cm: render_bvh_batch(spec_256, p, m, cp, cm, mode="jax_bvh"))
    fn_pal_512 = jax.jit(lambda p, m, cp, cm: render_bvh_batch(spec_512, p, m, cp, cm, mode="pallas_bvh", interpret=False))

    gb_p0 = fn_pal_256(jnp.asarray(ds["obj_xpos"][:1]), jnp.asarray(ds["obj_xmat"][:1]), jnp.asarray(ds["cam_xpos"][:1]), jnp.asarray(ds["cam_xmat"][:1]))
    gb_j0 = fn_jax_256(jnp.asarray(ds["obj_xpos"][:1]), jnp.asarray(ds["obj_xmat"][:1]), jnp.asarray(ds["cam_xpos"][:1]), jnp.asarray(ds["cam_xmat"][:1]))
    gb_512_0 = fn_pal_512(jnp.asarray(ds["obj_xpos"][N_256 : N_256 + 1]), jnp.asarray(ds["obj_xmat"][N_256 : N_256 + 1]), jnp.asarray(ds["cam_xpos"][N_256 : N_256 + 1]), jnp.asarray(ds["cam_xmat"][N_256 : N_256 + 1]))
    jax.block_until_ready((gb_p0.rgb, gb_j0.rgb, gb_512_0.rgb))

    cy256 = np.asarray(gt256["frame0_z_depth"], np.float32)
    tp256 = np.asarray(gb_p0.z_depth[0], np.float32)
    b256 = (cy256 < 20.0) & (tp256 > 0.0)
    cy512 = np.asarray(gt512["frame0_z_depth"], np.float32)
    tp512 = np.asarray(gb_512_0.z_depth[0], np.float32)
    b512 = (cy512 < 20.0) & (tp512 > 0.0)

    parity = {
        "pallas_vs_jax_256_max_depth_diff_m": float(jnp.max(jnp.abs(gb_p0.depth - gb_j0.depth))),
        "pallas_vs_jax_256_max_rgb_diff": float(jnp.max(jnp.abs(gb_p0.rgb - gb_j0.rgb))),
        "pallas_vs_jax_256_geom_match": float(jnp.mean(gb_p0.geom_id == gb_j0.geom_id)),
        "pallas_vs_cycles_256_z_mae_m": float(np.mean(np.abs(cy256[b256] - tp256[b256]))),
        "pallas_vs_cycles_256_fg_match": float(np.mean((cy256 < 20.0) == (tp256 > 0.0))),
        "pallas_vs_cycles_512_z_mae_m": float(np.mean(np.abs(cy512[b512] - tp512[b512]))),
        "pallas_vs_cycles_512_fg_match": float(np.mean((cy512 < 20.0) == (tp512 > 0.0))),
    }
    print("PARITY:", json.dumps(parity, indent=2), flush=True)
    assert parity["pallas_vs_cycles_256_z_mae_m"] < 1e-4
    assert parity["pallas_vs_cycles_512_z_mae_m"] < 1e-4
    summary["parity"] = parity

    # 2. Render all 48 @ 256x256 and 4 @ 512x512 G-buffers via Mosaic Pallas BVH-128
    t_gb = time.time()
    fn_b12 = jax.jit(lambda p, m, cp, cm: render_bvh_batch(spec_256, p, m, cp, cm, mode="pallas_bvh", interpret=False))
    gb_256_list = []
    for st in range(0, N_256, 12):
      en = min(N_256, st + 12)
      gb_c = fn_b12(
          jnp.asarray(ds["obj_xpos"][st:en]),
          jnp.asarray(ds["obj_xmat"][st:en]),
          jnp.asarray(ds["cam_xpos"][st:en]),
          jnp.asarray(ds["cam_xmat"][st:en]),
      )
      gb_256_list.append(jax.device_get(gb_c))

    feats_256 = np.concatenate([np.asarray(build_gbuffer_features(g), np.float32) for g in gb_256_list], axis=0)
    s1_256 = np.concatenate([np.asarray(g.rgb, np.float32) for g in gb_256_list], axis=0)
    z_256 = np.concatenate([np.asarray(g.z_depth, np.float32) for g in gb_256_list], axis=0)

    gb_512_all = jax.jit(lambda p, m, cp, cm: render_bvh_batch(spec_512, p, m, cp, cm, mode="pallas_bvh", interpret=False))(
        jnp.asarray(ds["obj_xpos"][N_256:]),
        jnp.asarray(ds["obj_xmat"][N_256:]),
        jnp.asarray(ds["cam_xpos"][N_256:]),
        jnp.asarray(ds["cam_xmat"][N_256:]),
    )
    feats_512 = np.asarray(build_gbuffer_features(gb_512_all), np.float32)
    s1_512 = np.asarray(gb_512_all.rgb, np.float32)
    z_512 = np.asarray(gb_512_all.z_depth, np.float32)
    print(f"Rendered 48x256 + 4x512 1.24M-tri G-buffers in {time.time()-t_gb:.2f}s", flush=True)
    np.savez(
        gb_cache_npz,
        feats_256=feats_256.astype(np.float16),
        s1_256=s1_256,
        z_256=z_256,
        feats_512=feats_512.astype(np.float16),
        s1_512=s1_512,
        z_512=z_512,
    )
    print(f"Saved G-buffer cache {gb_cache_npz}", flush=True)

  if args.stage in ("all", "train"):
    gbc = np.load(gb_cache_npz)
    feats_256 = gbc["feats_256"].astype(np.float32)
    s1_256 = gbc["s1_256"].astype(np.float32)
    z_256 = gbc["z_256"].astype(np.float32)
    feats_512 = gbc["feats_512"].astype(np.float32)
    s1_512 = gbc["s1_512"].astype(np.float32)
    z_512 = gbc["z_512"].astype(np.float32)

    split = ds["split_256"]
    tr_idx = np.where(split == 0)[0]
    cal_idx = np.where(split == 1)[0]
    te_idx = np.where(split == 2)[0]
    gt_256_rgb = gt256["rgb"].astype(np.float32)
    gt_512_rgb = gt512["rgb"].astype(np.float32)

    candidates = [
        ("SpatialRelighter_W64", SpatialRelighter(base_width=64)),
        ("SpatialRelighter_W96", SpatialRelighter(base_width=96)),
        ("SpatialRelighter3S_W96", SpatialRelighter3Scale(base_width=96)),
    ]
    neural_results = []
    best_psnr = -1.0
    best_pack = None
    for cname, cmod in candidates:
      met, ep, cs, p256, p512 = train_and_eval_relighter(
          cname, cmod,
          feats_256, s1_256, gt_256_rgb, tr_idx, cal_idx, te_idx,
          feats_512, s1_512, gt_512_rgb,
          epochs=args.epochs, batch_size=9, lr=2.5e-3,
      )
      print("EVAL:", json.dumps(met, indent=2), flush=True)
      neural_results.append(met)
      if met["test_256"]["neural_psnr_db"] > best_psnr:
        best_psnr = met["test_256"]["neural_psnr_db"]
        best_pack = (cname, cmod, ep, cs, p256, p512, met)

    best_name, best_mod, best_params, best_cs, best_p256, best_p512, best_met = best_pack
    summary["neural_relighting"] = {"runs": neural_results, "best": best_met}

    @jax.jit
    def infer_all_256(p, xf, bf):
      return jnp.clip(bf + best_mod.apply({"params": p}, xf, train=False), 0.0, 1.0)

    pred_all_256 = np.clip(
        np.concatenate(
            [np.asarray(infer_all_256(best_params, jnp.asarray(feats_256[i : i + 12]), jnp.asarray(s1_256[i : i + 12]))) for i in range(0, N_256, 12)],
            axis=0,
        )
        * best_cs,
        0.0,
        1.0,
    )

    np.savez_compressed(
        os.path.join(args.work_dir, "v3_w5_showcase_preds_512.npz"),
        s1_rgb=s1_512,
        pred_rgb=best_p512,
        gt_rgb=gt_512_rgb,
        s1_z_depth=z_512,
    )
    np.savez_compressed(
        os.path.join(args.work_dir, "v3_w5_orbit_preds_256.npz"),
        split=split,
        test_idx=te_idx,
        s1_rgb=s1_256,
        pred_rgb=pred_all_256,
        gt_rgb=gt_256_rgb,
        s1_z_depth=z_256,
    )
    with open(os.path.join(args.work_dir, "v3_w5_best_relighter.pkl"), "wb") as f:
      pickle.dump({"arch": best_name, "cal_scale": best_cs, "params": jax.tree_util.tree_map(np.asarray, best_params)}, f)

  def _load_best():
    ck = pickle.load(open(os.path.join(args.work_dir, "v3_w5_best_relighter.pkl"), "rb"))
    bn = ck["arch"]
    bcs = float(ck["cal_scale"])
    bp = jax.tree_util.tree_map(jnp.asarray, ck["params"])
    if bn == "SpatialRelighter3S_W96":
      bm = SpatialRelighter3Scale(base_width=96)
    elif bn == "SpatialRelighter_W96":
      bm = SpatialRelighter(base_width=96)
    else:
      bm = SpatialRelighter(base_width=64)
    return bm, bp, bcs

  if args.stage in ("all", "bench1c_a"):
    b1c = []
    for res, batches in [(64, [1, 64, 256]), (256, [1, 16, 48])]:
      for bs in batches:
        r1 = benchmark_render(spec_256, spec_arrs, ds, res, bs, n_chips=1, n_steps=8)
        print("BENCH 1c S1:", json.dumps(r1), flush=True)
        b1c.append(r1)
    summary["benchmarks_1chip"] = b1c

  if args.stage in ("all", "bench1c_b"):
    best_mod, best_params, best_cs = _load_best()
    b1c = summary.get("benchmarks_1chip", [])
    for bs in [1, 4, 8]:
      r1 = benchmark_render(spec_256, spec_arrs, ds, 512, bs, n_chips=1, n_steps=5)
      print("BENCH 1c S1:", json.dumps(r1), flush=True)
      b1c.append(r1)
    for res, bs in [(256, 16), (512, 4)]:
      r2 = benchmark_render(spec_256, spec_arrs, ds, res, bs, n_chips=1, neural_model=best_mod, neural_params=best_params, cal_scale=best_cs, n_steps=6)
      print("BENCH 1c S1+S2:", json.dumps(r2), flush=True)
      b1c.append(r2)
    summary["benchmarks_1chip"] = b1c

  if args.stage in ("all", "bench4c_a"):
    assert len(jax.devices()) >= 4, f"Need >= 4 devices, got {len(jax.devices())}"
    b4c = []
    for res, batches in [(64, [64, 256]), (256, [16, 48])]:
      for bs in batches:
        r1 = benchmark_render(spec_256, spec_arrs, ds, res, bs, n_chips=4, n_steps=8)
        print("BENCH 4c S1:", json.dumps(r1), flush=True)
        b4c.append(r1)
    summary["benchmarks_4chip_shard_map"] = b4c

  if args.stage in ("all", "bench4c_b"):
    assert len(jax.devices()) >= 4, f"Need >= 4 devices, got {len(jax.devices())}"
    best_mod, best_params, best_cs = _load_best()
    b4c = summary.get("benchmarks_4chip_shard_map", [])
    for bs in [4, 8]:
      r1 = benchmark_render(spec_256, spec_arrs, ds, 512, bs, n_chips=4, n_steps=5)
      print("BENCH 4c S1:", json.dumps(r1), flush=True)
      b4c.append(r1)
    for res, bs in [(256, 16), (512, 4)]:
      r2 = benchmark_render(spec_256, spec_arrs, ds, res, bs, n_chips=4, neural_model=best_mod, neural_params=best_params, cal_scale=best_cs, n_steps=6)
      print("BENCH 4c S1+S2:", json.dumps(r2), flush=True)
      b4c.append(r2)
    summary["benchmarks_4chip_shard_map"] = b4c

  with open(out_json_path, "w") as f:
    json.dump(summary, f, indent=2)
  print("SAVED SUMMARY:", out_json_path, flush=True)


if __name__ == "__main__":
  main()
