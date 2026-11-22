"""Wave 2A: Dynamic Moving-Light Generalization on PandaPickCube (TPU v5p).

Tests how Stage 2 albedo-demodulated neural shaders generalize to randomized moving light
directions, elevations, intensities, and colors against a 64-spp multi-bounce Monte Carlo
path-traced reference on the exact MuJoCo PandaPickCube analytic primitives.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import pickle
import time
from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import optax

from gate import dilate_mask, evaluate_stage2_neural, luminance
from models import make_candidate_model
from tpu_mjx_render.neural_shader import build_gbuffer_features
from tpu_mjx_render.primitives import (
    GEOM_BOX,
    GEOM_CAPSULE,
    GEOM_CYLINDER,
    GEOM_ELLIPSOID,
    GEOM_PLANE,
    GEOM_SPHERE,
    MJ_MAXVAL,
    RAY_FN_BY_TYPE,
)
from tpu_mjx_render.renderer import GBuffer, RenderSceneSpec, render_exact_gbuffer
from tpu_mjx_render.scenes import get_scene_xml


def _cast_world_rays(
    spec: RenderSceneSpec,
    geom_xpos: jax.Array,
    geom_xmat: jax.Array,
    geom_size: jax.Array,
    ray_orig: jax.Array,  # (R, 3)
    ray_dir: jax.Array,   # (R, 3)
    max_dist: jax.Array,  # (R,)
) -> Tuple[jax.Array, jax.Array, jax.Array]:
  R = ray_orig.shape[0]
  best_dist = max_dist
  best_gid = jnp.full((R,), -1, dtype=jnp.int32)
  best_nrm = jnp.zeros((R, 3), dtype=jnp.float32)

  for gt, gid_tuple in spec.type_groups:
    idx = np.asarray(gid_tuple, dtype=np.int32)
    g_pos = geom_xpos[idx][None, :, :]
    g_mat = geom_xmat[idx][None, :, :, :]
    g_sz = geom_size[idx][None, :, :]
    fn = RAY_FN_BY_TYPE[gt]
    d_grp, n_grp = fn(g_pos, g_mat, g_sz, ray_orig[:, None, :], ray_dir[:, None, :])
    valid = (d_grp >= 0.0) & (d_grp < best_dist[:, None])
    d_masked = jnp.where(valid, d_grp, MJ_MAXVAL)
    local_idx = jnp.argmin(d_masked, axis=1)
    min_d = jnp.take_along_axis(d_masked, local_idx[:, None], axis=1)[:, 0]
    improved = min_d < best_dist
    gids_arr = jnp.asarray(idx, dtype=jnp.int32)
    cand_gid = gids_arr[local_idx]
    cand_nrm = jnp.take_along_axis(n_grp, local_idx[:, None, None], axis=1)[:, 0, :]
    best_dist = jnp.where(improved, min_d, best_dist)
    best_gid = jnp.where(improved, cand_gid, best_gid)
    best_nrm = jnp.where(improved[:, None], cand_nrm, best_nrm)

  hit = best_gid >= 0
  return jnp.where(hit, best_dist, -1.0), best_nrm, best_gid


def _sample_albedo_world(
    spec: RenderSceneSpec,
    geom_id: jax.Array,
    hit_pos: jax.Array,
    geom_xpos: jax.Array,
    geom_xmat: jax.Array,
) -> jax.Array:
  safe_gid = jnp.maximum(geom_id, 0)
  base = jnp.asarray(spec.geom_rgba, dtype=jnp.float32)[safe_gid, :3]
  is_chk = jnp.asarray(spec.geom_is_checker, dtype=jnp.bool_)[safe_gid]
  g_pos = geom_xpos[safe_gid]
  g_mat = geom_xmat[safe_gid]
  local_p = jnp.einsum("nji,nj->ni", g_mat, hit_pos - g_pos, precision=jax.lax.Precision.HIGHEST)
  rep_x, rep_y = spec.checker_repeat
  cx = jnp.floor(local_p[:, 0] * rep_x).astype(jnp.int32)
  cy = jnp.floor(local_p[:, 1] * rep_y).astype(jnp.int32)
  even = ((cx + cy) & 1) == 0
  rgb1 = jnp.asarray(spec.checker_rgb1, dtype=jnp.float32)
  rgb2 = jnp.asarray(spec.checker_rgb2, dtype=jnp.float32)
  chk_rgb = jnp.where(even[:, None], rgb1[None, :], rgb2[None, :])
  return jnp.where((geom_id >= 0)[:, None] & is_chk[:, None], chk_rgb, base)


def _build_onb(n: jax.Array) -> Tuple[jax.Array, jax.Array]:
  nz = n[..., 2:3]
  sign = jnp.where(nz >= 0.0, 1.0, -1.0)
  a = -1.0 / (sign + nz)
  b = n[..., 0:1] * n[..., 1:2] * a
  t1 = jnp.concatenate([1.0 + sign * (n[..., 0:1] ** 2) * a, sign * b, -sign * n[..., 0:1]], axis=-1)
  near_neg_z = nz < -0.999999
  t1 = jnp.where(near_neg_z, jnp.broadcast_to(jnp.array([0.0, -1.0, 0.0], jnp.float32), t1.shape), t1)
  t1 = t1 / jnp.maximum(jnp.linalg.norm(t1, axis=-1, keepdims=True), 1e-12)
  t2 = jnp.cross(n, t1)
  return t1, t2


def render_panda_mc_gi_single(
    spec: RenderSceneSpec,
    geom_xpos: jax.Array,
    geom_xmat: jax.Array,
    cam_pos: jax.Array,
    cam_mat: jax.Array,
    light_pos: jax.Array,      # (3,)
    light_diffuse: jax.Array,  # (3,)
    spp: int = 64,
    max_bounces: int = 3,
    rng_seed: int = 0,
) -> jax.Array:
  """Render multi-bounce soft-area-light MC GI reference on exact Panda analytic primitives."""
  H, W = spec.height, spec.width
  R = H * W
  geom_size = jnp.asarray(spec.geom_size, dtype=jnp.float32)
  geom_spec = jnp.asarray(spec.geom_spec, dtype=jnp.float32)
  geom_shin = jnp.asarray(spec.geom_shin, dtype=jnp.float32)

  aspect = float(W) / float(H)
  half_h = float(spec.znear) * math.tan(0.5 * float(spec.cam_fovy) * (math.pi / 180.0))
  half_w = half_h * aspect
  px = jnp.arange(W, dtype=jnp.float32) + 0.5
  py = jnp.arange(H, dtype=jnp.float32) + 0.5
  gx, gy = jnp.meshgrid(px, py, indexing="xy")
  gx = gx.reshape(-1)
  gy = gy.reshape(-1)

  sky_top = jnp.asarray(spec.sky_rgb1, dtype=jnp.float32)
  sky_bot = jnp.asarray(spec.sky_rgb2, dtype=jnp.float32)
  area_radius = 0.42
  light_dir_t = jnp.array([0.55, 0.0, 0.12], dtype=jnp.float32) - light_pos
  light_n = light_dir_t / jnp.maximum(jnp.linalg.norm(light_dir_t), 1e-12)
  lt1, lt2 = _build_onb(light_n)

  # Secondary cool studio fill overhead-side (matches studio balanced illumination)
  fill_pos = jnp.array([0.95, -1.1, 1.05], dtype=jnp.float32)
  fill_col = jnp.array([0.35, 0.40, 0.48], dtype=jnp.float32)

  base_rng = jax.random.PRNGKey(rng_seed)

  def _one_sample(s_idx, acc):
    s_rng = jax.random.fold_in(base_rng, s_idx)
    k_jit, k_bnc = jax.random.split(s_rng)
    jit = jax.random.uniform(k_jit, (R, 2), minval=-0.5, maxval=0.5, dtype=jnp.float32)
    u = (gx + jit[:, 0]) / float(W)
    v = (gy + jit[:, 1]) / float(H)
    x = -half_w + (2.0 * half_w) * u
    y = half_h - (2.0 * half_h) * v
    raw = jnp.stack([x, y, jnp.full_like(x, -float(spec.znear))], axis=-1)
    ld = raw / jnp.maximum(jnp.linalg.norm(raw, axis=-1, keepdims=True), 1e-12)
    neg_z = jnp.maximum(-ld[:, 2], 1e-6)
    rd0 = jnp.einsum("ij,rj->ri", cam_mat, ld, precision=jax.lax.Precision.HIGHEST)
    rd0 = rd0 / jnp.maximum(jnp.linalg.norm(rd0, axis=-1, keepdims=True), 1e-12)
    ro0 = cam_pos[None, :] + (float(spec.znear) / neg_z)[:, None] * rd0
    tmax0 = (float(spec.zfar - spec.znear) / neg_z)

    thr0 = jnp.ones((R, 3), dtype=jnp.float32)
    alive0 = jnp.ones((R,), dtype=jnp.bool_)
    rad0 = jnp.zeros((R, 3), dtype=jnp.float32)

    def _bounce(b_idx, carry):
      ro, rd, t_lim, thr, alive, rad = carry
      b_rng = jax.random.fold_in(k_bnc, b_idx)
      k_lu, k_lv, k_u1, k_u2 = jax.random.split(b_rng, 4)

      d_hit, nrm, gid = _cast_world_rays(
          spec, geom_xpos, geom_xmat, geom_size, ro, rd, jnp.where(alive, t_lim, 0.0)
      )
      hit = alive & (gid >= 0)
      miss = alive & (~hit)
      sky_t = jnp.clip(0.5 * (rd[:, 2] + 1.0), 0.0, 1.0)[:, None]
      sky = sky_t * sky_top[None, :] + (1.0 - sky_t) * sky_bot[None, :]
      rad = rad + jnp.where(miss[:, None], thr * sky * 0.42, 0.0)

      hit_pos = ro + jnp.where(hit, d_hit, 0.0)[:, None] * rd
      wo = -rd
      n_ff = jnp.where((jnp.sum(nrm * wo, axis=-1, keepdims=True) >= 0.0), nrm, -nrm)
      alb = _sample_albedo_world(spec, gid, hit_pos, geom_xpos, geom_xmat)
      safe_gid = jnp.maximum(gid, 0)
      spec_c = geom_spec[safe_gid]
      shin_e = geom_shin[safe_gid] * 128.0

      # NEE toward dynamic moving primary area light
      lu = jax.random.uniform(k_lu, (R,), minval=0.0, maxval=1.0, dtype=jnp.float32)
      lv = jax.random.uniform(k_lv, (R,), minval=0.0, maxval=1.0, dtype=jnp.float32)
      lr = area_radius * jnp.sqrt(lu)
      lphi = (2.0 * math.pi) * lv
      lp = light_pos[None, :] + (lr * jnp.cos(lphi))[:, None] * lt1[None, :] + (lr * jnp.sin(lphi))[:, None] * lt2[None, :]
      to_l = lp - hit_pos
      dist_l = jnp.linalg.norm(to_l, axis=-1)
      wi_l = to_l / jnp.maximum(dist_l[:, None], 1e-12)
      ndotl = jnp.maximum(jnp.sum(n_ff * wi_l, axis=-1), 0.0)
      sh_o = hit_pos + n_ff * 2.0e-4
      sh_lim = jnp.where(hit & (ndotl > 0.0), jnp.maximum(dist_l - 1.0e-3, 0.0), 0.0)
      _, _, sh_gid = _cast_world_rays(spec, geom_xpos, geom_xmat, geom_size, sh_o, wi_l, sh_lim)
      vis = jnp.where(hit & (ndotl > 0.0) & (sh_gid < 0), 1.0, 0.0)

      att = 1.0 / (1.0 + 0.08 * dist_l * dist_l)
      h_vec = (wi_l + wo) / jnp.maximum(jnp.linalg.norm(wi_l + wo, axis=-1, keepdims=True), 1e-12)
      ndoth = jnp.maximum(jnp.sum(n_ff * h_vec, axis=-1), 0.0)
      spec_term = jnp.where((spec_c > 0.0) & (shin_e > 0.0), 0.25 * spec_c * (ndoth ** shin_e), 0.0)
      nee_key = thr * (alb * ndotl[:, None] + spec_term[:, None]) * light_diffuse[None, :] * (att * vis * 1.35)[:, None]

      # Subtle ambient fill light (keeps dark shadowed side readable)
      to_f = fill_pos[None, :] - hit_pos
      dist_f = jnp.linalg.norm(to_f, axis=-1)
      wi_f = to_f / jnp.maximum(dist_f[:, None], 1e-12)
      ndotf = jnp.maximum(jnp.sum(n_ff * wi_f, axis=-1), 0.0)
      nee_fill = thr * alb * fill_col[None, :] * (ndotf / (1.0 + 0.15 * dist_f * dist_f) * 0.45)[:, None]

      rad = rad + jnp.where(hit[:, None], nee_key + nee_fill, 0.0)

      # Cosine hemisphere indirect bounce (diffuse GI + color bleeding)
      u1 = jax.random.uniform(k_u1, (R,), minval=0.0, maxval=1.0, dtype=jnp.float32)
      u2 = jax.random.uniform(k_u2, (R,), minval=0.0, maxval=1.0, dtype=jnp.float32)
      r_h = jnp.sqrt(jnp.clip(u1, 0.0, 1.0))
      phi_h = (2.0 * math.pi) * u2
      t1, t2 = _build_onb(n_ff)
      wi_next = t1 * (r_h * jnp.cos(phi_h))[:, None] + t2 * (r_h * jnp.sin(phi_h))[:, None] + n_ff * jnp.sqrt(jnp.maximum(1.0 - u1, 0.0))[:, None]
      wi_next = wi_next / jnp.maximum(jnp.linalg.norm(wi_next, axis=-1, keepdims=True), 1e-12)

      new_thr = thr * alb * 0.85
      new_alive = hit & (jnp.max(new_thr, axis=-1) > 1e-3)
      new_ro = hit_pos + n_ff * 2.0e-4
      new_tlim = jnp.full((R,), float(spec.zfar), dtype=jnp.float32)
      return (new_ro, wi_next, new_tlim, new_thr, new_alive, rad)

    _, _, _, _, _, final_rad = jax.lax.fori_loop(0, max_bounces, _bounce, (ro0, rd0, tmax0, thr0, alive0, rad0))
    return acc + final_rad

  rad_avg = jax.lax.fori_loop(0, spp, _one_sample, jnp.zeros((R, 3), dtype=jnp.float32)) / float(spp)
  # Soft Reinhard/gamma-matched [0, 1] output
  rgb = jnp.clip(rad_avg / (1.0 + 0.15 * rad_avg), 0.0, 1.0)
  return rgb.reshape(H, W, 3)


def make_randomized_lights(n: int, seed: int = 777) -> Tuple[np.ndarray, np.ndarray]:
  """Sample randomized light positions (spherical arc above Panda workspace) & diffuse colors."""
  rng = np.random.default_rng(seed)
  az = rng.uniform(-math.pi, math.pi, size=n)
  el = rng.uniform(math.radians(35.0), math.radians(82.0), size=n)
  dist = rng.uniform(1.3, 2.3, size=n)
  cx, cy, cz = 0.55, 0.0, 0.12
  lx = cx + dist * np.cos(el) * np.cos(az)
  ly = cy + dist * np.cos(el) * np.sin(az)
  lz = cz + dist * np.sin(el)
  l_pos = np.stack([lx, ly, lz], axis=-1).astype(np.float32)

  base_int = rng.uniform(0.65, 1.35, size=(n, 1)).astype(np.float32)
  tint = rng.uniform(0.85, 1.05, size=(n, 3)).astype(np.float32)
  l_diff = np.clip(base_int * tint * np.array([[0.85, 0.83, 0.78]], np.float32), 0.35, 1.45)
  return l_pos, l_diff


def render_dynamic_light_dataset(
    m: mujoco.MjModel,
    base_spec: RenderSceneSpec,
    geom_xpos: np.ndarray,
    geom_xmat: np.ndarray,
    cam_xpos: np.ndarray,
    cam_xmat: np.ndarray,
    l_pos: np.ndarray,
    l_diff: np.ndarray,
    spp: int = 64,
) -> Tuple[GBuffer, np.ndarray]:
  """Render both Stage 1 GBuffer (dynamic light) and 64-spp MC GI GT on TPU via vmap."""
  N = geom_xpos.shape[0]
  # Dynamic light shading inside Stage 1 exact raytracer:
  # We override spec.light_pos / light_dir / light_diffuse per frame via jax.vmap!
  geom_size = jnp.asarray(base_spec.geom_size, dtype=jnp.float32)

  def _render_one(gxp, gxm, cxp, cxm, lp, ld, idx):
    # Stage 1 raycast + dynamic light shading
    from tpu_mjx_render.camera import compute_camera_rays
    from tpu_mjx_render.renderer import _shade_scene

    ray_orig, ray_dir, neg_z = compute_camera_rays(
        cam_pos=cxp,
        cam_mat=cxm,
        fovy_deg=base_spec.cam_fovy,
        height=base_spec.height,
        width=base_spec.width,
        projection=base_spec.cam_projection,
        sensorsize=base_spec.cam_sensorsize,
        intrinsic=base_spec.cam_intrinsic,
        znear=base_spec.znear,
    )
    max_dist = (base_spec.zfar - base_spec.znear) / neg_z
    dist, normal, geom_id = _cast_world_rays(base_spec, gxp, gxm, geom_size, ray_orig, ray_dir, max_dist)

    # Temporarily patch spec light_diffuse? Since _shade_scene reads spec.light_diffuse as static numpy,
    # scale Stage 1 direct diffuse/specular cleanly or run with fixed unit diffuse and scale!
    # Better: use unit diffuse=(1,1,1) in spec_unit and pass light_xpos=lp[None], light_xdir=ldir[None].
    spec_unit = dataclasses.replace(
        base_spec,
        light_diffuse=np.ones((1, 3), dtype=np.float32),
        light_specular=np.full((1, 3), 0.35, dtype=np.float32),
        light_ambient=np.zeros((1, 3), dtype=np.float32),
    )
    ldir = jnp.array([0.55, 0.0, 0.12], dtype=jnp.float32) - lp
    ldir = ldir / jnp.maximum(jnp.linalg.norm(ldir), 1e-12)
    # Use _shade_scene with unit light, then scale direct light contribution by ld
    # Wait: let's inject ld directly into spec_unit? No, _shade_scene uses spec.light_diffuse static!
    # So we compute _shade_scene with headlight_active=False -> pure 1-light RGB + albedo*headlight!
    spec_nosh = dataclasses.replace(spec_unit, headlight_active=False)
    gb_l = _shade_scene(
        spec=spec_nosh,
        geom_xpos=gxp,
        geom_xmat=gxm,
        geom_size=geom_size,
        geom_rgba=jnp.asarray(base_spec.geom_rgba, dtype=jnp.float32),
        cam_pos=cxp,
        cam_mat=cxm,
        light_xpos=lp[None, :],
        light_xdir=ldir[None, :],
        ray_orig=ray_orig,
        ray_dir=ray_dir,
        neg_z=neg_z,
        dist=dist,
        normal=normal,
        geom_id=geom_id,
        mode="tiled",
    )
    hit = (gb_l.geom_id >= 0)[..., None]
    # Add dynamic light intensity/tint + subtle studio headlight
    ndotv = jnp.clip(jnp.sum(gb_l.normal * gb_l.view_dir, axis=-1, keepdims=True), 0.0, 1.0)
    hl = gb_l.albedo * (0.20 + 0.32 * ndotv)
    rgb_s1 = jnp.where(hit, jnp.clip(gb_l.rgb * ld[None, None, :] + hl, 0.0, 1.0), gb_l.rgb)
    gb_out = gb_l._replace(rgb=rgb_s1, direct_rgb=rgb_s1)

    gt_mc = render_panda_mc_gi_single(base_spec, gxp, gxm, cxp, cxm, lp, ld, spp=spp, max_bounces=3, rng_seed=1000)
    return gb_out, gt_mc

  @jax.jit
  def batch_fn(gxp, gxm, cxp, cxm, lp, ld, idxs):
    return jax.vmap(_render_one)(gxp, gxm, cxp, cxm, lp, ld, idxs)

  # Chunk in batches of 64 to keep compile & memory small
  gbs, gts = [], []
  for i0 in range(0, N, 64):
    i1 = min(i0 + 64, N)
    gb_c, gt_c = batch_fn(
        jnp.asarray(geom_xpos[i0:i1]),
        jnp.asarray(geom_xmat[i0:i1]),
        jnp.asarray(cam_xpos[i0:i1]),
        jnp.asarray(cam_xmat[i0:i1]),
        jnp.asarray(l_pos[i0:i1]),
        jnp.asarray(l_diff[i0:i1]),
        jnp.arange(i0, i1, dtype=jnp.int32),
    )
    jax.block_until_ready(gt_c)
    gbs.append(jax.tree_util.tree_map(np.asarray, gb_c))
    gts.append(np.asarray(gt_c))

  gb_all = jax.tree_util.tree_map(lambda *xs: np.concatenate(xs, axis=0), *gbs)
  gt_all = np.concatenate(gts, axis=0)
  return gb_all, gt_all


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--dataset", default="/dev/shm/zaychik-tpu-mjx/out/data/panda_cycles_dataset.npz")
  ap.add_argument("--ckpts", nargs="+", required=True)
  ap.add_argument("--arch_train", default="M2_demod_s2d_mxu_64")
  ap.add_argument("--spp", type=int, default=64)
  ap.add_argument("--epochs", type=int, default=160)
  ap.add_argument("--out_json", required=True)
  ap.add_argument("--out_npz", default="/dev/shm/zaychik-tpu-mjx-w4/out/v3_w4_dynlight_samples.npz")
  args = ap.parse_args()

  ds = np.load(args.dataset)
  m = mujoco.MjModel.from_xml_string(get_scene_xml("panda_pick_cube"))
  spec = RenderSceneSpec.from_mjmodel(m, cam_id=0, height=int(ds["height"]), width=int(ds["width"]), use_shadows=True)

  train_idx = np.where(ds["split"] == 0)[0]
  test_idx = np.where(ds["split"] == 2)[0]

  t0 = time.time()
  lp_tr, ld_tr = make_randomized_lights(len(train_idx), seed=101)
  lp_te, ld_te = make_randomized_lights(len(test_idx), seed=202)

  print(f"Rendering {len(train_idx)} train + {len(test_idx)} test dynamic-light frames ({args.spp}-spp 3-bounce MC GI on TPU)...", flush=True)
  gb_tr_np, gt_tr_np = render_dynamic_light_dataset(
      m, spec, ds["geom_xpos"][train_idx], ds["geom_xmat"][train_idx],
      ds["cam_xpos"][train_idx], ds["cam_xmat"][train_idx], lp_tr, ld_tr, spp=args.spp
  )
  gb_te_np, gt_te_np = render_dynamic_light_dataset(
      m, spec, ds["geom_xpos"][test_idx], ds["geom_xmat"][test_idx],
      ds["cam_xpos"][test_idx], ds["cam_xmat"][test_idx], lp_te, ld_te, spp=args.spp
  )
  print(f"Dynamic-light MC GI dataset rendered in {time.time()-t0:.2f}s", flush=True)

  gb_te_j = jax.tree_util.tree_map(jnp.asarray, gb_te_np)
  x_te = jnp.asarray(build_gbuffer_features(gb_te_j), dtype=jnp.float32)
  b_te = jnp.asarray(gb_te_np.rgb, dtype=jnp.float32)

  results: Dict[str, Any] = {
      "n_train": int(len(train_idx)),
      "n_test": int(len(test_idx)),
      "mc_spp": int(args.spp),
      "zero_shot_eval": {},
      "dynamic_light_trained": {},
  }

  # 1. Zero-shot evaluation of fixed-light Stage 2 checkpoints on unseen dynamic moving lights
  best_pred_zeroshot = None
  for ckpt_path in args.ckpts:
    with open(ckpt_path, "rb") as f:
      ck = pickle.load(f)
    arch = ck["arch"]
    model = make_candidate_model(arch)
    params = jax.tree_util.tree_map(jnp.asarray, ck["params"])
    cal_scale = float(ck["cal_scale"])

    @jax.jit
    def _infer(p, xin, bin_):
      return jnp.clip((bin_ + model.apply({"params": p}, xin, train=False)) * cal_scale, 0.0, 1.0)

    pred = np.asarray(_infer(params, x_te, b_te))
    g = evaluate_stage2_neural(
        rgb_pred=pred,
        rgb_ref=gt_te_np,
        rgb_stage1_baseline=gb_te_np.rgb,
        roi_mask=ds["roi_mask"][test_idx],
        geom_id=gb_te_np.geom_id,
        box_geom_id=int(ds["box_geom_id"]),
        finger_geom_ids=tuple(int(x) for x in ds["finger_geom_ids"]),
    )
    results["zero_shot_eval"][arch] = g
    print(f"[DYN-LIGHT ZERO-SHOT] {arch}: passed={g['passed']} PSNR={g['psnr_db']:.2f} (S1={g['stage1_psnr_db']:.2f}, +{g['psnr_gain_db']:.2f}) SSIM={g['ssim']:.4f} ROI_PSNR={g['roi_psnr_db']:.2f} occ={g['roi_occ_mismatch']:.5f}", flush=True)
    if best_pred_zeroshot is None:
      best_pred_zeroshot = pred

  # 2. Train/Fine-tune demodulated Stage 2 model directly on randomized moving lights
  gb_tr_j = jax.tree_util.tree_map(jnp.asarray, gb_tr_np)
  x_tr_np = np.asarray(build_gbuffer_features(gb_tr_j), dtype=np.float32)
  b_tr_np = gb_tr_np.rgb.astype(np.float32)
  roi_tr_np = ds["roi_mask"][train_idx].astype(np.float32)[..., None]
  gid_tr = gb_tr_np.geom_id
  bnd_tr_np = (
      dilate_mask(gid_tr == int(ds["box_geom_id"]), radius=1)
      & dilate_mask(np.isin(gid_tr, [int(x) for x in ds["finger_geom_ids"]]), radius=1)
  ).astype(np.float32)[..., None]

  dyn_model = make_candidate_model(args.arch_train)
  rng = jax.random.PRNGKey(42)
  params = dyn_model.init(rng, jnp.asarray(x_tr_np[:1]))["params"]
  ema_params = params
  batch_size = 32
  steps_per_epoch = len(train_idx) // batch_size
  total_steps = args.epochs * steps_per_epoch
  sched = optax.warmup_cosine_decay_schedule(2e-4, 2e-3, max(20, total_steps // 15), total_steps, 5e-5)
  tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=3e-5))
  opt_state = tx.init(params)

  x_tr_d = jnp.asarray(x_tr_np)
  b_tr_d = jnp.asarray(b_tr_np)
  y_tr_d = jnp.asarray(gt_tr_np)
  roi_tr_d = jnp.asarray(roi_tr_np)
  bnd_tr_d = jnp.asarray(bnd_tr_np)

  @jax.jit
  def train_ep(p, ema_p, ost, batch_idx):
    def _step(carry, idxs):
      cp, cema, cost = carry
      xb, bb, yb, roib, bndb = x_tr_d[idxs], b_tr_d[idxs], y_tr_d[idxs], roi_tr_d[idxs], bnd_tr_d[idxs]

      def loss_fn(pr):
        raw = bb + dyn_model.apply({"params": pr}, xb, train=True)
        w = 1.0 + 2.0 * roib + 200.0 * bndb
        return 0.7 * jnp.mean(w * (raw - yb) ** 2) + 0.3 * jnp.mean(w * jnp.abs(raw - yb))

      loss, grads = jax.value_and_grad(loss_fn)(cp)
      upd, n_ost = tx.update(grads, cost, cp)
      np_p = optax.apply_updates(cp, upd)
      n_ema = jax.tree_util.tree_map(lambda e, n: 0.995 * e + 0.005 * n, cema, np_p)
      return (np_p, n_ema, n_ost), loss

    (p_o, e_o, o_o), losses = jax.lax.scan(_step, (p, ema_p, ost), batch_idx)
    return p_o, e_o, o_o, jnp.mean(losses)

  np_rng = np.random.default_rng(42)
  t_tr0 = time.time()
  for ep in range(args.epochs):
    perm = np_rng.permutation(len(train_idx))[: steps_per_epoch * batch_size].reshape(steps_per_epoch, batch_size)
    params, ema_params, opt_state, loss_v = train_ep(params, ema_params, opt_state, jnp.asarray(perm))
    if (ep + 1) % 40 == 0 or ep + 1 == args.epochs:
      jax.block_until_ready(loss_v)
      print(f"[DYN-TRAIN {args.arch_train}] ep {ep+1}/{args.epochs} loss={float(loss_v):.5f}", flush=True)

  @jax.jit
  def dyn_infer(p, xin, bin_):
    return jnp.clip(bin_ + dyn_model.apply({"params": p}, xin, train=False), 0.0, 1.0)

  pred_dyn = np.asarray(dyn_infer(ema_params, x_te, b_te))
  g_dyn = evaluate_stage2_neural(
      rgb_pred=pred_dyn,
      rgb_ref=gt_te_np,
      rgb_stage1_baseline=gb_te_np.rgb,
      roi_mask=ds["roi_mask"][test_idx],
      geom_id=gb_te_np.geom_id,
      box_geom_id=int(ds["box_geom_id"]),
      finger_geom_ids=tuple(int(x) for x in ds["finger_geom_ids"]),
  )
  results["dynamic_light_trained"][args.arch_train] = {
      "train_time_s": round(time.time() - t_tr0, 2),
      "gate": g_dyn,
  }
  print(f"[DYN-LIGHT TRAINED] {args.arch_train}: passed={g_dyn['passed']} PSNR={g_dyn['psnr_db']:.2f} (S1={g_dyn['stage1_psnr_db']:.2f}, +{g_dyn['psnr_gain_db']:.2f}) SSIM={g_dyn['ssim']:.4f} ROI_PSNR={g_dyn['roi_psnr_db']:.2f} occ={g_dyn['roi_occ_mismatch']:.5f}", flush=True)

  os.makedirs(os.path.dirname(args.out_npz) or ".", exist_ok=True)
  np.savez_compressed(
      args.out_npz,
      s1_rgb=gb_te_np.rgb[:8],
      gt_mc=gt_te_np[:8],
      pred_zeroshot=best_pred_zeroshot[:8],
      pred_dyn=pred_dyn[:8],
      l_pos=lp_te[:8],
      l_diff=ld_te[:8],
  )
  os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
  with open(args.out_json, "w") as f:
    json.dump(results, f, indent=2)
  print("Saved dynamic-light results ->", args.out_json, flush=True)


if __name__ == "__main__":
  main()
