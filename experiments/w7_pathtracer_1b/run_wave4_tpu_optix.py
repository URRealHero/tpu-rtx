"""Priority #3: `tpu_optix` SDK Showcase + `jax.grad` Inverse Rendering vs Blender 5.3.0 Cycles."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw

from tpu_mjx_render.adapters.tpu_optix import (
    OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
    OptixBuildInputInstanceArray,
    OptixBuildInputTriangleArray,
    OptixDeviceContext,
    OptixHitPayload,
    OptixInstance,
    ShaderBindingTable,
    _build_onb,
    _safe_norm,
    _safe_unit,
    audit_hlo_zero_copy,
    build_optix_launcher,
    eval_ggx_brdf,
    make_default_pbr_sbt,
    optixTrace,
)


def stitch_strip(out_arr: jax.Array) -> np.ndarray:
  arr = np.asarray(jax.device_get(out_arr), dtype=np.float32)
  n_c, H, W_loc, C = arr.shape
  return arr.transpose(1, 0, 2, 3).reshape(H, n_c * W_loc, C)


def compute_psnr(img_a: np.ndarray, img_b: np.ndarray) -> float:
  mse = float(np.mean((img_a.astype(np.float32) - img_b.astype(np.float32)) ** 2))
  return float(10.0 * math.log10(1.0 / max(mse, 1e-10)))


def make_cycles_matched_diff_sbt(
    cam_fovy: float,
    znear: float,
    zfar: float,
    spp: int = 16,
    max_bounces: int = 3,
    target_mat_tag: float = 0.45,
) -> ShaderBindingTable:
  """Differentiable SBT matched to Cycles Standard sRGB + Principled BSDF (IOR 1.5, F0=0.04)."""

  def _miss(payload: OptixHitPayload, params: Dict[str, Any]) -> jax.Array:
    return jnp.broadcast_to(jnp.array([0.032, 0.040, 0.056], dtype=jnp.float32), payload.rd.shape)

  def _anyhit(payload: OptixHitPayload, params: Dict[str, Any]) -> jax.Array:
    return payload.hit

  def _closesthit(payload: OptixHitPayload, params: Dict[str, Any], aux: Dict[str, Any]) -> jax.Array:
    H_l, W_l = aux["H"], aux["W"]
    accel = aux["accel"]
    rng = aux["rng"]
    n_unit = _safe_unit(payload.normal)
    wo = -payload.rd
    entering = jnp.sum(n_unit * wo, axis=-1) >= 0.0
    n_ff = jnp.where(entering[..., None], n_unit, -n_unit)

    is_hero = jnp.abs(payload.mat_code - target_mat_tag) < 0.08
    hero_alb = params["hero_albedo"]
    hero_rgh = params["hero_roughness"]
    alb = jnp.where(is_hero[..., None], hero_alb[None, None, :], payload.albedo)
    rgh = jnp.where(is_hero, hero_rgh, payload.roughness)
    metallic = jnp.zeros_like(rgh)

    hit_pos = payload.ro + jnp.where(payload.hit, payload.dist, 0.0)[..., None] * payload.rd
    l_pos = params["light_pos"]
    l_emit = params["light_emit"]
    l_rad = params["light_radius"]
    l_norm = jnp.array([0.0, 0.0, -1.0], dtype=jnp.float32)
    lt1, lt2 = _build_onb(l_norm)

    k_u, k_v = jax.random.split(rng)
    lu = jax.random.uniform(k_u, payload.hit.shape, minval=0.0, maxval=1.0, dtype=jnp.float32)
    lv = jax.random.uniform(k_v, payload.hit.shape, minval=0.0, maxval=1.0, dtype=jnp.float32)
    lr = l_rad * jnp.sqrt(lu + 1e-8)
    lphi = (2.0 * math.pi) * lv
    lp = l_pos[None, None, :] + (lr * jnp.cos(lphi))[..., None] * lt1[None, None, :] + (lr * jnp.sin(lphi))[..., None] * lt2[None, None, :]

    to_l = lp - hit_pos
    dist_l = _safe_norm(to_l)
    wi_l = _safe_unit(to_l)
    cos_s = jnp.maximum(jnp.sum(n_ff * wi_l, axis=-1), 0.0)
    cos_l = jnp.maximum(jnp.sum(l_norm[None, None, :] * (-wi_l), axis=-1), 0.0)

    sh_o = hit_pos + n_ff * 1.5e-3
    sh_lim = jnp.where(payload.hit & (cos_s > 0.0) & (cos_l > 0.0), jnp.maximum(dist_l - 2.0e-3, 0.0), 0.0)
    sh_occ = optixTrace(
        accel, sh_o, wi_l, 0.0, sh_lim, H_l, W_l,
        ray_flags=OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
        sbt=ShaderBindingTable(lambda *_: None, _miss, lambda *_: None, _anyhit),
        params=params,
    )
    vis = jnp.where(payload.hit & (cos_s > 0.0) & (cos_l > 0.0) & (~sh_occ), 1.0, 0.0)
    brdf = eval_ggx_brdf(n_ff, wo, wi_l, alb, metallic, rgh)
    l_area = math.pi * l_rad * l_rad
    geom = (cos_s * cos_l) / jnp.maximum(dist_l * dist_l, 0.02) * l_area
    return jnp.minimum(brdf * l_emit[None, None, :] * (geom * vis)[..., None], 16.0)

  def _raygen(accel, params, H, W_local, W_total, x_off, rng_in):
    aspect = float(W_total) / float(H)
    half_h = float(znear) * math.tan(0.5 * float(cam_fovy) * (math.pi / 180.0))
    half_w = half_h * aspect
    R = H * W_local
    spp_par = 16 if (spp >= 16 and R <= 32768) else (8 if (spp >= 8 and R <= 65536) else 4)
    spp_seq = max(1, spp // spp_par)
    cxp = params["cam_pos"]
    cxm = params["cam_xmat"]

    def _sample_batch(s_idx, rad_acc):
      s_rng = jax.random.fold_in(rng_in, s_idx)
      k_jit, k_bnc = jax.random.split(s_rng)
      jitter = jax.random.uniform(k_jit, (spp_par, R, 2), minval=-0.5, maxval=0.5, dtype=jnp.float32)
      px = jnp.arange(W_local, dtype=jnp.float32) + jnp.asarray(x_off, jnp.float32) + 0.5
      py = jnp.arange(H, dtype=jnp.float32) + 0.5
      gx, gy = jnp.meshgrid(px, py, indexing="xy")
      u = (gx.reshape(1, R) + jitter[..., 0]) / float(W_total)
      v = (gy.reshape(1, R) + jitter[..., 1]) / float(H)
      x = -half_w + (2.0 * half_w) * u
      y = half_h - (2.0 * half_h) * v
      raw = jnp.stack([x, y, jnp.full_like(x, -float(znear))], axis=-1)
      ld = _safe_unit(raw)
      neg_z = jnp.maximum(-ld[..., 2], 1e-6)
      rd0 = _safe_unit(jnp.einsum("ij,brj->bri", cxm, ld, precision=jax.lax.Precision.HIGHEST))
      ro0 = cxp[None, None, :] + (float(znear) / neg_z)[..., None] * rd0
      t_max0 = jnp.broadcast_to((float(zfar - znear) / neg_z), (spp_par, R))
      thr0 = jnp.ones((spp_par, R, 3), dtype=jnp.float32)
      alive0 = jnp.ones((spp_par, R), dtype=jnp.bool_)
      rad0 = jnp.zeros((spp_par, R, 3), dtype=jnp.float32)

      def _bounce(b_idx, carry):
        ro, rd, t_lim, thr, alive, rad = carry
        b_rng = jax.random.fold_in(k_bnc, b_idx)
        k_nee, k_u1, k_u2 = jax.random.split(b_rng, 3)

        pl_hit = optixTrace(accel, ro, rd, 0.0, jnp.where(alive, t_lim, 0.0), H, W_local)
        hit = alive & pl_hit.hit
        miss = alive & (~hit)
        rad = rad + jnp.where(miss[..., None], thr * _miss(pl_hit, params), 0.0)

        nee = _closesthit(pl_hit, params, {"H": H, "W": W_local, "accel": accel, "rng": k_nee})
        rad = rad + jnp.where(hit[..., None], thr * nee, 0.0)

        n_unit = _safe_unit(pl_hit.normal)
        wo = -rd
        entering = jnp.sum(n_unit * wo, axis=-1) >= 0.0
        n_ff = jnp.where(entering[..., None], n_unit, -n_unit)
        is_hero = jnp.abs(pl_hit.mat_code - target_mat_tag) < 0.08
        alb = jnp.where(is_hero[..., None], params["hero_albedo"][None, None, :], pl_hit.albedo)

        u1 = jax.random.uniform(k_u1, (spp_par, R), minval=0.0, maxval=1.0, dtype=jnp.float32)
        u2 = jax.random.uniform(k_u2, (spp_par, R), minval=0.0, maxval=1.0, dtype=jnp.float32)
        r_cos = jnp.sqrt(jnp.clip(u1, 1e-6, 1.0))
        phi = 2.0 * math.pi * u2
        t1, t2 = _build_onb(n_ff)
        wi_next = _safe_unit(t1 * (r_cos * jnp.cos(phi))[..., None] + t2 * (r_cos * jnp.sin(phi))[..., None] + n_ff * jnp.sqrt(jnp.maximum(1.0 - u1, 1e-6))[..., None])

        hit_pos = ro + jnp.where(hit, pl_hit.dist, 0.0)[..., None] * rd
        new_ro = hit_pos + n_ff * 1.5e-3
        new_thr = thr * alb
        new_alive = hit & (jnp.max(new_thr, axis=-1) > 1e-3)
        new_t_lim = jnp.broadcast_to(jnp.float32(zfar), (spp_par, R))
        return (new_ro, wi_next, new_t_lim, new_thr, new_alive, rad)

      _, _, _, _, _, final_rad = jax.lax.fori_loop(0, max_bounces, _bounce, (ro0, rd0, t_max0, thr0, alive0, rad0))
      return rad_acc + jnp.mean(final_rad, axis=0)

    rad_sum = jax.lax.fori_loop(0, spp_seq, _sample_batch, jnp.zeros((R, 3), dtype=jnp.float32))
    rad_avg = (rad_sum / float(spp_seq)).reshape(H, W_local, 3)
    # Soft differentiable linear -> sRGB (IEC 61966-2-1 gamma 2.2) matching Cycles Standard
    lin_c = jnp.clip(rad_avg, 1e-4, 1.0)
    rgb = jnp.power(lin_c, 1.0 / 2.2)
    return rgb[None, ...]

  return ShaderBindingTable(raygen=_raygen, miss=_miss, closesthit=_closesthit, anyhit=_anyhit)


def _draw_convergence_plot(
    w: int, h: int, losses: List[float], psnrs: List[float],
    param_errs: Dict[str, List[float]],
) -> Image.Image:
  im = Image.new("RGB", (w, h), (18, 22, 28))
  draw = ImageDraw.Draw(im)
  draw.text((12, 8), "jax.grad End-to-End Inverse Rendering Convergence", fill=(240, 245, 252))

  # Plot box
  x0, y0, x1, y1 = 44, 34, w - 18, h - 38
  draw.rectangle([x0, y0, x1, y1], outline=(70, 80, 95), width=1)
  n = max(len(losses) - 1, 1)

  # Normalize loss curve (blue/cyan) & PSNR curve (gold)
  l_max = max(max(losses), 1e-6)
  p_min, p_max = min(psnrs) - 1.0, max(psnrs) + 1.0

  def _plot_series(vals, vmin, vmax, col):
    pts = []
    for i, val in enumerate(vals):
      px = x0 + int((i / n) * (x1 - x0))
      norm = np.clip((val - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
      py = y1 - int(norm * (y1 - y0))
      pts.append((px, py))
    if len(pts) >= 2:
      draw.line(pts, fill=col, width=2)

  _plot_series(losses, 0.0, l_max, (80, 195, 255))
  _plot_series(psnrs, p_min, p_max, (255, 210, 75))
  _plot_series(param_errs["albedo_l2"], 0.0, max(param_errs["albedo_l2"]) + 1e-4, (110, 235, 130))
  _plot_series(param_errs["light_pos_l2"], 0.0, max(param_errs["light_pos_l2"]) + 1e-4, (255, 120, 140))

  draw.text((x0 + 6, y0 + 6), f"Cyan: Image Loss ({losses[0]:.4f} -> {losses[-1]:.4f})", fill=(80, 195, 255))
  draw.text((x0 + 6, y0 + 22), f"Gold: PSNR vs Cycles GT ({psnrs[0]:.1f} -> {psnrs[-1]:.1f} dB)", fill=(255, 210, 75))
  draw.text((x0 + 6, y0 + 38), f"Green: Albedo L2 ({param_errs['albedo_l2'][0]:.3f} -> {param_errs['albedo_l2'][-1]:.3f})", fill=(110, 235, 130))
  draw.text((x0 + 6, y0 + 54), f"Red: Light Pos L2 ({param_errs['light_pos_l2'][0]:.3f} -> {param_errs['light_pos_l2'][-1]:.3f} m)", fill=(255, 120, 140))
  draw.text((x0, y1 + 8), "Iter 0", fill=(180, 190, 205))
  draw.text((x1 - 48, y1 + 8), f"Iter {n}", fill=(180, 190, 205))
  return im


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--scene_npz", default="/dev/shm/zaychik-tpu-mjx-w7/opendata/exported/inv_scene.npz")
  ap.add_argument("--cycles_png", default="/dev/shm/zaychik-tpu-mjx-w7/opendata/exported/cycles_inv_ref.png")
  ap.add_argument("--steps", type=int, default=65)
  ap.add_argument("--out_json", default="out/metrics/v3_w7_wave4_tpu_optix_sdk.json")
  ap.add_argument("--out_png", default="out/figures/v3_w7_tpu_optix_inverse.png")
  args = ap.parse_args()

  ctx = OptixDeviceContext()
  assert ctx.n_chips == 4

  # --------------------------------------------------------------------------
  # Part A: tpu_optix SDK AccelBuild (GAS Wide-BVH-64 + IAS 2-Level TLAS) Bench
  # --------------------------------------------------------------------------
  ds = np.load(args.scene_npz, allow_pickle=False)
  verts = ds["verts"]
  faces = ds["faces"]
  vnorm = ds["vnorm"]
  tri_alb = ds["tri_albedo"]
  tri_rgh = ds["tri_roughness"]
  tri_mc = ds["tri_mat_code"]
  cam_pos_np = ds["cam_pos"]
  cam_xmat_np = ds["cam_xmat"]
  cam_fovy = float(ds["cam_fovy"])
  gt_meta = json.loads(bytes(ds["gt_meta"]).decode("utf-8"))

  t_gas0 = time.perf_counter()
  gas_scene = ctx.accelBuild(
      OptixBuildInputTriangleArray(
          name="cornell_hero_gas",
          vertices=verts,
          indices=faces,
          vert_normals=vnorm,
          tri_albedo=tri_alb,
          tri_roughness=tri_rgh,
          tri_mat_code=tri_mc,
      )
  )
  gas_build_ms = (time.perf_counter() - t_gas0) * 1e3

  # Also build a 2-Level TLAS (IAS) over 8 rotated instances of gas_scene
  instances = []
  for k in range(8):
    ang = 2.0 * math.pi * k / 8.0
    rmat = np.array([[math.cos(ang), -math.sin(ang), 0], [math.sin(ang), math.cos(ang), 0], [0, 0, 1]], np.float32)
    pos = np.array([3.5 * math.cos(ang), 3.5 * math.sin(ang), 0.0], np.float32)
    instances.append(OptixInstance(gas=gas_scene, transform_pos=pos, transform_mat=rmat, instance_id=k))
  t_ias0 = time.perf_counter()
  ias_8x = ctx.accelBuild(OptixBuildInputInstanceArray(name="tlas_8x_instances", instances=tuple(instances)))
  ias_build_ms = (time.perf_counter() - t_ias0) * 1e3

  sbt_bench = make_default_pbr_sbt(cam_fovy, 0.05, 30.0, spp=16, max_bounces=3)
  launch_gas_512 = build_optix_launcher(ctx, sbt_bench, gas_scene, 512, 512)
  launch_ias_512 = build_optix_launcher(ctx, sbt_bench, ias_8x, 512, 512)

  bench_params = jax.device_put({
      "cam_pos": jnp.asarray(cam_pos_np, jnp.float32),
      "cam_xmat": jnp.asarray(cam_xmat_np, jnp.float32),
      "light_pos": jnp.asarray(gt_meta["light_pos_gt"], jnp.float32),
      "light_emit": jnp.asarray(gt_meta["light_emit_gt"], jnp.float32),
      "light_radius": jnp.float32(gt_meta["light_radius"]),
      "hero_albedo": jnp.asarray(gt_meta["hero_albedo_gt"], jnp.float32),
      "hero_roughness": jnp.float32(gt_meta["hero_roughness_gt"]),
      "hero_metallic": jnp.float32(0.0),
      "exposure": jnp.float32(1.0),
  }, ctx.rep)
  rng0 = jax.device_put(jax.random.PRNGKey(0), ctx.rep)

  hlo_gas = ctx.audit_hlo(launch_gas_512.lower(bench_params, rng0))
  y_g = launch_gas_512(bench_params, rng0); jax.block_until_ready(y_g)
  t0 = time.perf_counter()
  for i in range(3):
    y_g = launch_gas_512(bench_params, jax.device_put(jax.random.PRNGKey(10 + i), ctx.rep))
  jax.block_until_ready(y_g)
  gas_512_ms = (time.perf_counter() - t0) / 3.0 * 1e3

  hlo_ias = ctx.audit_hlo(launch_ias_512.lower(bench_params, rng0))
  y_i = launch_ias_512(bench_params, rng0); jax.block_until_ready(y_i)
  t0 = time.perf_counter()
  for i in range(2):
    y_i = launch_ias_512(bench_params, jax.device_put(jax.random.PRNGKey(20 + i), ctx.rep))
  jax.block_until_ready(y_i)
  ias_512_ms = (time.perf_counter() - t0) / 2.0 * 1e3

  print(f"tpu_optix SDK GAS 512x512 16spp 3b: {gas_512_ms:.2f}ms ({1000.0/gas_512_ms*16*60:.1f} spm) | HLO={hlo_gas}", flush=True)
  print(f"tpu_optix SDK IAS(2-level TLAS 8x) 512x512 16spp 3b: {ias_512_ms:.2f}ms | HLO={hlo_ias}", flush=True)

  # --------------------------------------------------------------------------
  # Part B: End-to-End jax.grad Inverse Rendering vs Cycles 5.3.0 Reference
  # --------------------------------------------------------------------------
  gt_pil = Image.open(args.cycles_png).convert("RGB")
  H_inv, W_inv = gt_pil.size[1], gt_pil.size[0]
  gt_np = np.asarray(gt_pil, dtype=np.float32) / 255.0
  # Shard target across 4 chips along width to match sharded output directly
  W_loc = W_inv // ctx.n_chips
  gt_sharded_np = gt_np.reshape(H_inv, ctx.n_chips, W_loc, 3).transpose(1, 0, 2, 3)
  gt_dev = jax.device_put(gt_sharded_np, ctx.sh_dp)

  sbt_inv = make_cycles_matched_diff_sbt(cam_fovy, 0.05, 30.0, spp=16, max_bounces=3, target_mat_tag=0.45)
  launch_inv = build_optix_launcher(ctx, sbt_inv, gas_scene, W_inv, H_inv)

  # Verify forward match at GT parameters first!
  gt_params = jax.device_put({
      "cam_pos": jnp.asarray(cam_pos_np, jnp.float32),
      "cam_xmat": jnp.asarray(cam_xmat_np, jnp.float32),
      "light_pos": jnp.asarray(gt_meta["light_pos_gt"], jnp.float32),
      "light_emit": jnp.asarray(gt_meta["light_emit_gt"], jnp.float32),
      "light_radius": jnp.float32(gt_meta["light_radius"]),
      "hero_albedo": jnp.asarray(gt_meta["hero_albedo_gt"], jnp.float32),
      "hero_roughness": jnp.float32(gt_meta["hero_roughness_gt"]),
  }, ctx.rep)
  img_fwd_gt = stitch_strip(launch_inv(gt_params, jax.device_put(jax.random.PRNGKey(99), ctx.rep)))
  fwd_psnr_vs_cycles = compute_psnr(img_fwd_gt, gt_np)
  print(f"Forward Parity vs Cycles 5.3.0 GT at exact params: {fwd_psnr_vs_cycles:.2f} dB", flush=True)

  # Unconstrained differentiable latent vector theta = [lp_x, lp_y, lp_z, log_le(3), logit_alb(3), logit_rgh(1)]
  def _unpack_theta(theta: jax.Array) -> Dict[str, Any]:
    lp = jnp.stack([
        jnp.tanh(theta[0]) * 0.90,
        jnp.tanh(theta[1]) * 0.90,
        1.75 + 0.45 * jax.nn.sigmoid(theta[2]),
    ])
    le = jnp.exp(jnp.clip(theta[3:6], 0.5, 4.0))
    alb = 0.05 + 0.90 * jax.nn.sigmoid(theta[6:9])
    rgh = 0.04 + 0.85 * jax.nn.sigmoid(theta[9])
    return {
        "cam_pos": jnp.asarray(cam_pos_np, jnp.float32),
        "cam_xmat": jnp.asarray(cam_xmat_np, jnp.float32),
        "light_pos": lp,
        "light_emit": le,
        "light_radius": jnp.float32(gt_meta["light_radius"]),
        "hero_albedo": alb,
        "hero_roughness": rgh,
    }

  def _loss_fn(theta: jax.Array, rng_k: jax.Array) -> jax.Array:
    prms = _unpack_theta(theta)
    pred = launch_inv(prms, rng_k)  # (4, H, W_loc, 3)
    diff = pred - gt_dev
    l2 = jnp.mean(diff * diff)
    l1 = jnp.mean(jnp.abs(diff))
    return l2 + 0.15 * l1

  val_and_grad_fn = jax.jit(jax.value_and_grad(_loss_fn))

  # Severely wrong initial guess: light on wrong side/dimmer/blue-tinted, sphere is rough cobalt blue!
  theta_init_np = np.array([-0.55, 0.35, -0.30, math.log(8.0), math.log(14.0), math.log(19.0), -1.3, -0.3, 1.8, 1.1], dtype=np.float32)
  theta = jax.device_put(theta_init_np, ctx.rep)

  hlo_grad = ctx.audit_hlo(val_and_grad_fn.lower(theta, jax.device_put(jax.random.PRNGKey(1), ctx.rep)))
  print("jax.value_and_grad HLO zero-copy verified:", hlo_grad, flush=True)

  # Render initial guess image
  p_init = jax.device_get(_unpack_theta(theta))
  img_init = stitch_strip(launch_inv(jax.device_put(_unpack_theta(theta), ctx.rep), jax.device_put(jax.random.PRNGKey(101), ctx.rep)))

  # Adam optimizer on TPU
  m = jnp.zeros_like(theta)
  v = jnp.zeros_like(theta)
  lr = 0.085
  b1, b2, eps = 0.9, 0.999, 1e-8

  lp_gt = np.array(gt_meta["light_pos_gt"], np.float32)
  le_gt = np.array(gt_meta["light_emit_gt"], np.float32)
  alb_gt = np.array(gt_meta["hero_albedo_gt"], np.float32)
  rgh_gt = float(gt_meta["hero_roughness_gt"])

  losses, psnrs = [], []
  param_errs = {"light_pos_l2": [], "light_emit_rel": [], "albedo_l2": [], "roughness_abs": []}
  step_times_ms = []

  for step in range(args.steps):
    rng_step = jax.device_put(jax.random.PRNGKey(500 + step), ctx.rep)
    ts0 = time.perf_counter()
    loss_val, grads = val_and_grad_fn(theta, rng_step)
    jax.block_until_ready((loss_val, grads))
    dt_ms = (time.perf_counter() - ts0) * 1e3
    if step > 0:
      step_times_ms.append(dt_ms)

    g_np = np.asarray(jax.device_get(grads), dtype=np.float32)
    assert np.all(np.isfinite(g_np)), f"Non-finite gradient at step {step}: {g_np}"

    # Adam update
    t_step = step + 1
    m = b1 * m + (1.0 - b1) * grads
    v = b2 * v + (1.0 - b2) * (grads * grads)
    m_hat = m / (1.0 - b1 ** t_step)
    v_hat = v / (1.0 - b2 ** t_step)
    lr_cur = lr * (0.5 * (1.0 + math.cos(math.pi * step / max(args.steps, 1))))
    theta = theta - lr_cur * m_hat / (jnp.sqrt(v_hat) + eps)

    p_cur = jax.device_get(_unpack_theta(theta))
    lp_err = float(np.linalg.norm(np.asarray(p_cur["light_pos"]) - lp_gt))
    le_err = float(np.linalg.norm(np.asarray(p_cur["light_emit"]) - le_gt) / np.linalg.norm(le_gt))
    alb_err = float(np.linalg.norm(np.asarray(p_cur["hero_albedo"]) - alb_gt))
    rgh_err = float(abs(float(p_cur["hero_roughness"]) - rgh_gt))

    losses.append(float(jax.device_get(loss_val)))
    param_errs["light_pos_l2"].append(lp_err)
    param_errs["light_emit_rel"].append(le_err)
    param_errs["albedo_l2"].append(alb_err)
    param_errs["roughness_abs"].append(rgh_err)

    if step % 10 == 0 or step == args.steps - 1:
      im_cur = stitch_strip(launch_inv(jax.device_put(_unpack_theta(theta), ctx.rep), jax.device_put(jax.random.PRNGKey(99), ctx.rep)))
      psnr_cur = compute_psnr(im_cur, gt_np)
      print(
          f"Step {step:02d} | loss={losses[-1]:.5f} | PSNR={psnr_cur:.2f}dB | "
          f"lp_err={lp_err:.3f}m | alb_err={alb_err:.3f} | rgh={float(p_cur['hero_roughness']):.3f} (gt {rgh_gt:.3f}) | {dt_ms:.1f}ms",
          flush=True,
      )
    else:
      # Fast linear PSNR estimate from L2 loss
      psnrs.append(float(10.0 * math.log10(1.0 / max(losses[-1] * 0.7, 1e-8))))
      continue
    psnrs.append(psnr_cur)

  img_rec = stitch_strip(launch_inv(jax.device_put(_unpack_theta(theta), ctx.rep), jax.device_put(jax.random.PRNGKey(99), ctx.rep)))
  psnr_init = compute_psnr(img_init, gt_np)
  psnr_rec = compute_psnr(img_rec, gt_np)
  p_final = jax.device_get(_unpack_theta(theta))

  # Save 2x2 Comparison + Convergence Hero PNG
  pad, hdr = 14, 40
  pw, ph = W_inv, H_inv
  plot_im = _draw_convergence_plot(pw, ph, losses, psnrs, param_errs)
  canvas = Image.new("RGB", (2 * pw + 3 * pad, 2 * (ph + hdr) + 3 * pad), (15, 17, 21))
  draw = ImageDraw.Draw(canvas)
  panels = [
      (f"Initial Wrong Guess (PSNR {psnr_init:.2f} dB)", Image.fromarray(np.clip(img_init * 255.0 + 0.5, 0, 255).astype(np.uint8))),
      (f"tpu_optix jax.grad Recovered (PSNR {psnr_rec:.2f} dB)", Image.fromarray(np.clip(img_rec * 255.0 + 0.5, 0, 255).astype(np.uint8))),
      ("Blender 5.3.0 Cycles Reference GT (256 spp)", gt_pil),
      (f"Optimization Convergence ({np.mean(step_times_ms):.1f} ms/grad-step)", plot_im),
  ]
  for idx, (title, pim) in enumerate(panels):
    r, c = idx // 2, idx % 2
    x0 = pad + c * (pw + pad)
    y0 = pad + r * (ph + hdr + pad)
    canvas.paste(pim, (x0, y0 + hdr))
    draw.text((x0 + 4, y0 + 12), title, fill=(242, 246, 252))

  os.makedirs(os.path.dirname(args.out_png) or ".", exist_ok=True)
  canvas.save(args.out_png)
  print("Saved tpu_optix Inverse PNG ->", args.out_png, canvas.size, flush=True)

  out_data = {
      "experiment": "v3_w7_wave4_tpu_optix_sdk",
      "backend": ctx.backend,
      "n_chips": ctx.n_chips,
      "sdk_bench": {
          "gas_name": gas_scene.name,
          "gas_tris": gas_scene.n_tris,
          "gas_hbm_mb": gas_scene.hbm_mb,
          "gas_build_ms": round(gas_build_ms, 2),
          "gas_512_16spp_3b_ms": round(gas_512_ms, 2),
          "gas_512_16spp_3b_spm": round(1000.0 / gas_512_ms * 16 * 60.0, 1),
          "gas_hlo": hlo_gas,
          "ias_name": ias_8x.name,
          "ias_effective_tris": ias_8x.total_effective_tris,
          "ias_build_ms": round(ias_build_ms, 2),
          "ias_512_16spp_3b_ms": round(ias_512_ms, 2),
          "ias_hlo": hlo_ias,
      },
      "inverse_rendering": {
          "res": f"{H_inv}x{W_inv}",
          "spp_per_grad_step": 16,
          "max_bounces": 3,
          "steps": args.steps,
          "avg_grad_step_ms": round(float(np.mean(step_times_ms)), 2),
          "grad_hlo": hlo_grad,
          "forward_parity_vs_cycles_db": round(fwd_psnr_vs_cycles, 2),
          "psnr_init_db": round(psnr_init, 2),
          "psnr_recovered_db": round(psnr_rec, 2),
          "loss_init": round(losses[0], 6),
          "loss_final": round(losses[-1], 6),
          "gt": gt_meta,
          "init": {
              "light_pos": [round(float(x), 4) for x in p_init["light_pos"]],
              "light_emit": [round(float(x), 4) for x in p_init["light_emit"]],
              "hero_albedo": [round(float(x), 4) for x in p_init["hero_albedo"]],
              "hero_roughness": round(float(p_init["hero_roughness"]), 4),
          },
          "recovered": {
              "light_pos": [round(float(x), 4) for x in p_final["light_pos"]],
              "light_emit": [round(float(x), 4) for x in p_final["light_emit"]],
              "hero_albedo": [round(float(x), 4) for x in p_final["hero_albedo"]],
              "hero_roughness": round(float(p_final["hero_roughness"]), 4),
          },
          "final_errors": {
              "light_pos_l2_m": round(param_errs["light_pos_l2"][-1], 4),
              "light_emit_rel": round(param_errs["light_emit_rel"][-1], 4),
              "albedo_l2": round(param_errs["albedo_l2"][-1], 4),
              "roughness_abs": round(param_errs["roughness_abs"][-1], 4),
          },
      },
  }
  os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
  with open(args.out_json, "w") as f:
    json.dump(out_data, f, indent=2)
  print("Saved tpu_optix JSON ->", args.out_json, flush=True)


if __name__ == "__main__":
  main()
