"""Render 30-frame smooth camera-orbit + moving-sunlight/keylight sequences on TPU v5p (4-chip shard_map).

Generates paired datasets for TPU-DLSS 3.5 training and evaluation:
  - Half-res (H/2 x W/2) 1-spp 4-bounce MC noisy radiance + primary G-buffer at Halton (2,3) sub-pixel jitter
  - Full-res (H x W) 256-spp 4-bounce MC GT at canonical anti-aliased pixel centers
  - Exact per-frame camera extrinsics (cxp, cxm), Halton jitter, and light trajectory metadata.
"""

from __future__ import annotations

import argparse
import inspect
import math
import os
import time
from typing import Sequence, Tuple

import jax
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np

from bvh_pbr_pt import (
    LightRig,
    PBRWideBVHMesh,
    _agx_tonemap_srgb,
    _build_onb,
    _eval_pbr_brdf,
    _sample_cosine_hemisphere,
    _sample_ggx_half,
    _smith_g1,
    pbr_raycast_tpu,
)
from complex_scenes_pt import (
    _look_at_mat,
    build_classroom_sunlit,
    build_cornell_workshop,
)
from physical_fx_pt import build_caustics_studio_scene


def halton_sequence(n: int, base: int) -> np.ndarray:
  out = np.zeros((n,), dtype=np.float32)
  for i in range(n):
    f = 1.0
    r = 0.0
    idx = i + 1
    while idx > 0:
      f /= float(base)
      r += f * float(idx % base)
      idx //= base
    out[i] = r
  return out


def generate_halton_jitter(n_frames: int) -> np.ndarray:
  """Centered Halton (2, 3) sub-pixel offsets in [-0.5, 0.5] LR pixels."""
  jx = halton_sequence(n_frames, 2) - 0.5
  jy = halton_sequence(n_frames, 3) - 0.5
  return np.stack([jx, jy], axis=-1).astype(np.float32)


def _build_scene(scene_name: str):
  if scene_name == "classroom":
    return build_classroom_sunlit()
  if scene_name == "cornell_workshop":
    return build_cornell_workshop()
  if scene_name == "caustics_studio":
    return build_caustics_studio_scene()
  raise ValueError(f"Unknown scene: {scene_name}")


def generate_sequence_trajectory(
    scene_name: str, split: str = "eval", n_frames: int = 30
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Generate smooth 30-frame camera orbit + moving sunlight/area-light trajectory."""
  cxps, cxms, sun_dirs, area_cs = [], [], [], []
  for t in range(n_frames):
    s = float(t) / float(max(1, n_frames - 1))
    if scene_name == "classroom":
      if split == "eval":
        pos = (
            2.12 - 1.08 * s,
            -2.02 + 0.42 * math.sin(math.pi * s),
            1.52 - 0.24 * s,
        )
        tgt = (
            -0.46 + 0.30 * s,
            0.82 + 0.14 * math.sin(math.pi * s),
            0.84 + 0.10 * s,
        )
        sdir = np.array([-0.82, -0.36 + 0.62 * s, 0.48 - 0.10 * s], np.float32)
        ac = np.array([0.1, 0.4, 2.92], np.float32)
      else:
        pos = (
            0.88 + 1.24 * s,
            -2.08 + 0.68 * s,
            1.64 - 0.44 * s,
        )
        tgt = (
            -0.55 + 0.36 * s,
            0.66 + 0.38 * s,
            0.80 + 0.16 * s,
        )
        sdir = np.array([-0.84, 0.28 - 0.64 * s, 0.40 + 0.14 * s], np.float32)
        ac = np.array([0.1, 0.4, 2.92], np.float32)
    elif scene_name == "cornell_workshop":
      if split == "eval":
        th = -0.52 + 1.04 * s
        pos = (
            0.78 * math.sin(th),
            -2.55 + 0.30 * (1.0 - math.cos(th)),
            1.04 + 0.18 * math.sin(2.0 * math.pi * s),
        )
        tgt = (0.04 * math.sin(th), 0.20, 0.80)
        ac = np.array([0.38 * math.sin(math.pi * (s - 0.5)), 0.05 + 0.22 * math.cos(math.pi * s), 2.18], np.float32)
      else:
        th = 0.62 - 1.24 * s
        pos = (
            0.82 * math.sin(th),
            -2.35 - 0.22 * math.cos(th),
            1.22 - 0.28 * s,
        )
        tgt = (0.05, 0.26, 0.70)
        ac = np.array([-0.35 * math.cos(math.pi * s), 0.10 - 0.24 * math.sin(math.pi * s), 2.18], np.float32)
      sdir = np.array([0.0, 0.0, 1.0], np.float32)
    else:  # caustics_studio (zero-shot)
      th = -0.48 + 0.96 * s
      pos = (
          0.72 * math.sin(th),
          -2.52 + 0.24 * (1.0 - math.cos(th)),
          1.12 + 0.16 * math.cos(th),
      )
      tgt = (0.02, 0.04, 0.24)
      ac = np.array([-1.05 + 0.45 * s, 0.80 - 0.28 * math.sin(math.pi * s), 1.56], np.float32)
      sdir = np.array([0.0, 0.0, 1.0], np.float32)

    cxp, cxm = _look_at_mat(pos, tgt)
    cxps.append(cxp)
    cxms.append(cxm)
    sun_dirs.append((sdir / max(float(np.linalg.norm(sdir)), 1e-12)).astype(np.float32))
    area_cs.append(ac)

  jit = generate_halton_jitter(n_frames)
  return (
      np.stack(cxps, axis=0),
      np.stack(cxms, axis=0),
      np.stack(sun_dirs, axis=0),
      np.stack(area_cs, axis=0),
      jit,
  )


def build_dynamic_pt_fn(
    bvh_mesh: PBRWideBVHMesh,
    devices: Sequence[jax.Device],
    height: int,
    width: int,
    cam_fovy: float,
    znear: float,
    zfar: float,
    light_rig: LightRig,
    spp: int,
    max_bounces: int = 4,
    use_external_jitter: bool = False,
    aim_area_at_origin: bool = False,
):
  """Compile 4-chip sharded MC path tracer with dynamic camera, sun_dir, area_center, and jitter."""
  n_chips = len(devices)
  assert width % n_chips == 0 and (width // n_chips) % 16 == 0 and height % 8 == 0
  W_local = width // n_chips

  mesh = Mesh(np.array(devices), ("dp",))
  rep = NamedSharding(mesh, P())
  sh_dp = NamedSharding(mesh, P("dp"))

  l2_roots_d = jax.device_put(bvh_mesh.l2_roots_aabb, rep)
  l2_d = jax.device_put(bvh_mesh.l2_aabbs, rep)
  l0_d = jax.device_put(bvh_mesh.l0_aabbs, rep)
  leaf_d = jax.device_put(bvh_mesh.leaf_bank, rep)
  chip_ids = jax.device_put(np.arange(n_chips, dtype=np.int32), sh_dp)

  area_n_fixed = jnp.asarray(light_rig.area_normal, dtype=jnp.float32)
  area_n_fixed = area_n_fixed / jnp.maximum(jnp.linalg.norm(area_n_fixed), 1e-12)
  area_r = float(light_rig.area_radius)
  area_area = math.pi * area_r * area_r
  area_emit = jnp.asarray(light_rig.area_emit, dtype=jnp.float32)

  sun_emit = jnp.asarray(light_rig.sun_emit, dtype=jnp.float32)
  sun_sin_max = math.sqrt(max(0.0, 1.0 - light_rig.sun_cos_half_angle ** 2))
  sky_z = jnp.asarray(light_rig.sky_zenith, dtype=jnp.float32)
  sky_h = jnp.asarray(light_rig.sky_horizon, dtype=jnp.float32)
  sky_g = jnp.asarray(light_rig.sky_ground, dtype=jnp.float32)

  aspect = float(width) / float(height)
  half_h = float(znear) * math.tan(0.5 * float(cam_fovy) * (math.pi / 180.0))
  half_w = half_h * aspect

  def _trace_core(l2r, l2, l0, lf, cxp, cxm, sun_d_in, area_c_in, jit_in, cid, rng_in):
    x_off = cid * W_local
    chip_rng = jax.random.fold_in(rng_in, cid)

    sun_d = sun_d_in / jnp.maximum(jnp.linalg.norm(sun_d_in), 1e-12)
    st1, st2 = _build_onb(sun_d)

    if aim_area_at_origin:
      aim_vec = jnp.array([0.12, -0.22, 0.18], dtype=jnp.float32) - area_c_in
      area_n = aim_vec / jnp.maximum(jnp.linalg.norm(aim_vec), 1e-12)
    else:
      area_n = area_n_fixed
    at1, at2 = _build_onb(area_n)

    def _env_sky(rd):
      z = rd[..., 2]
      tu = jnp.clip(z, 0.0, 1.0)[..., None]
      td = jnp.clip(-z, 0.0, 1.0)[..., None]
      s = jnp.where(z[..., None] >= 0.0, (1.0 - tu) * sky_h + tu * sky_z, (1.0 - td) * sky_h + td * sky_g)
      if light_rig.has_sun:
        cos_s = jnp.maximum(jnp.sum(rd * sun_d, axis=-1), 0.0)
        s = s + (cos_s ** 32.0)[..., None] * sun_emit * 0.25
      return s

    local_px = height * W_local
    if spp >= 16 and local_px <= 32768:
      spp_par = 16
    elif spp >= 8 and local_px <= 131072:
      spp_par = 8
    elif spp >= 4 and local_px <= 262144:
      spp_par = 4
    else:
      spp_par = 1
    spp_seq = spp // spp_par
    R = height * W_local

    # Primary G-buffer rays (jittered by jit_in when use_external_jitter=True!)
    jx0 = jit_in[0] if use_external_jitter else jnp.float32(0.0)
    jy0 = jit_in[1] if use_external_jitter else jnp.float32(0.0)
    px_c = jnp.arange(W_local, dtype=jnp.float32) + jnp.asarray(x_off, jnp.float32) + 0.5 + jx0
    py_c = jnp.arange(height, dtype=jnp.float32) + 0.5 + jy0
    gxc, gyc = jnp.meshgrid(px_c, py_c, indexing="xy")

    uc = gxc.reshape(1, R) / float(width)
    vc = gyc.reshape(1, R) / float(height)
    xc = -half_w + (2.0 * half_w) * uc
    yc = half_h - (2.0 * half_h) * vc
    raw_c = jnp.stack([xc, yc, jnp.full_like(xc, -float(znear))], axis=-1)
    ld_c = raw_c / jnp.maximum(jnp.linalg.norm(raw_c, axis=-1, keepdims=True), 1e-12)
    neg_zc = jnp.maximum(-ld_c[..., 2], 1e-6)
    rd_c = jnp.einsum("ij,brj->bri", cxm[0], ld_c, precision=jax.lax.Precision.HIGHEST)
    rd_c = rd_c / jnp.maximum(jnp.linalg.norm(rd_c, axis=-1, keepdims=True), 1e-12)
    ro_c = cxp[0][None, None, :] + (float(znear) / neg_zc)[..., None] * rd_c
    tmax_c = jnp.broadcast_to((float(zfar - znear) / neg_zc), (1, R))

    d_p, n_p, alb_p, rgh_p, mc_p = pbr_raycast_tpu(
        l2r, l2, l0, lf, ro_c, rd_c, tmax_c, height, W_local, any_hit=False
    )
    hit_p = (d_p < tmax_c - 1e-4) & (d_p >= 0.0)
    n_p_u = n_p / jnp.maximum(jnp.linalg.norm(n_p, axis=-1, keepdims=True), 1e-12)
    n_p_ff = jnp.where((jnp.sum(n_p_u * (-rd_c), axis=-1) >= 0.0)[..., None], n_p_u, -n_p_u)
    sky_alb = jnp.clip(_agx_tonemap_srgb(_env_sky(rd_c), exposure=light_rig.exposure), 0.0, 1.0)
    gbuf_flat = jnp.concatenate(
        [
            jnp.where(hit_p[..., None], alb_p, sky_alb),
            jnp.where(hit_p[..., None], n_p_ff, 0.0),
            jnp.where(hit_p, jnp.clip(d_p / 10.0, 0.0, 1.5), -1.0)[..., None],
            jnp.where(hit_p, rgh_p, 1.0)[..., None],
            jnp.where(hit_p, jnp.clip(mc_p * 0.5, 0.0, 1.5), 0.0)[..., None],
        ],
        axis=-1,
    )
    gbuf_hw = gbuf_flat[0].reshape(height, W_local, 9)

    def _trace_batch_samples(s_idx, rad_acc):
      s_rng = jax.random.fold_in(chip_rng, s_idx)
      k_jit, k_bnc = jax.random.split(s_rng)
      if use_external_jitter:
        u = uc
        v = vc
      else:
        jitter = jax.random.uniform(k_jit, (spp_par, R, 2), minval=-0.5, maxval=0.5, dtype=jnp.float32)
        u = (gxc.reshape(1, R) + jitter[..., 0]) / float(width)
        v = (gyc.reshape(1, R) + jitter[..., 1]) / float(height)
      x = -half_w + (2.0 * half_w) * u
      y = half_h - (2.0 * half_h) * v
      raw = jnp.stack([x, y, jnp.full_like(x, -float(znear))], axis=-1)
      ld = raw / jnp.maximum(jnp.linalg.norm(raw, axis=-1, keepdims=True), 1e-12)
      neg_z = jnp.maximum(-ld[..., 2], 1e-6)
      rd0 = jnp.einsum("ij,brj->bri", cxm[0], ld, precision=jax.lax.Precision.HIGHEST)
      rd0 = rd0 / jnp.maximum(jnp.linalg.norm(rd0, axis=-1, keepdims=True), 1e-12)
      ro0 = cxp[0][None, None, :] + (float(znear) / neg_z)[..., None] * rd0
      t_max0 = jnp.broadcast_to((float(zfar - znear) / neg_z), (spp_par, R))
      if use_external_jitter:
        rd0 = jnp.broadcast_to(rd0, (spp_par, R, 3))
        ro0 = jnp.broadcast_to(ro0, (spp_par, R, 3))
      thr0 = jnp.ones((spp_par, R, 3), dtype=jnp.float32)
      alive0 = jnp.ones((spp_par, R), dtype=jnp.bool_)
      rad0 = jnp.zeros((spp_par, R, 3), dtype=jnp.float32)

      def _bounce(b_idx, carry):
        ro, rd, t_lim, thr, alive, rad = carry
        b_rng = jax.random.fold_in(k_bnc, b_idx)
        k_lu, k_lv, k_su, k_sv, k_lobe, k_u1, k_u2 = jax.random.split(b_rng, 7)

        d_hit, nrm_raw, alb, rgh, mcode = pbr_raycast_tpu(
            l2r, l2, l0, lf, ro, rd, jnp.where(alive, t_lim, 0.0), height, W_local, any_hit=False
        )
        hit = alive & (d_hit < t_lim - 1e-4) & (d_hit >= 0.0)
        miss = alive & (~hit)
        rad = rad + jnp.where(miss[..., None], thr * _env_sky(rd), 0.0)

        n_unit = nrm_raw / jnp.maximum(jnp.linalg.norm(nrm_raw, axis=-1, keepdims=True), 1e-12)
        wo = -rd
        entering = jnp.sum(n_unit * wo, axis=-1) >= 0.0
        n_ff = jnp.where(entering[..., None], n_unit, -n_unit)

        is_emit = hit & (mcode >= 2.8)
        is_glass = hit & (mcode >= 1.8) & (mcode < 2.8)
        is_opaque = hit & (mcode < 1.8)
        metallic = jnp.clip(mcode, 0.0, 1.0)

        emit_power = jnp.where(mcode >= 2.8, (mcode - 2.0) * 6.0, 0.0)
        rad = rad + jnp.where(is_emit[..., None], thr * alb * emit_power[..., None], 0.0)

        hit_pos = ro + jnp.where(hit, d_hit, 0.0)[..., None] * rd

        lu = jax.random.uniform(k_lu, (spp_par, R), minval=0.0, maxval=1.0, dtype=jnp.float32)
        lv = jax.random.uniform(k_lv, (spp_par, R), minval=0.0, maxval=1.0, dtype=jnp.float32)
        lr = area_r * jnp.sqrt(lu)
        lphi = (2.0 * math.pi) * lv
        lp = area_c_in[None, None, :] + (lr * jnp.cos(lphi))[..., None] * at1[None, None, :] + (lr * jnp.sin(lphi))[..., None] * at2[None, None, :]
        to_l = lp - hit_pos
        dist_l = jnp.linalg.norm(to_l, axis=-1)
        wi_l = to_l / jnp.maximum(dist_l[..., None], 1e-12)
        cos_s = jnp.maximum(jnp.sum(n_ff * wi_l, axis=-1), 0.0)
        cos_l = jnp.maximum(jnp.sum(area_n[None, None, :] * (-wi_l), axis=-1), 0.0)
        need_sh = is_opaque & (cos_s > 0.0) & (cos_l > 0.0)
        sh_o = hit_pos + n_ff * 1.5e-3
        sh_lim = jnp.where(need_sh, jnp.maximum(dist_l - 2.0e-3, 0.0), 0.0)

        d_sh, *_ = pbr_raycast_tpu(l2r, l2, l0, lf, sh_o, wi_l, sh_lim, height, W_local, any_hit=True)
        vis = jnp.where(need_sh & (d_sh >= sh_lim - 1e-4), 1.0, 0.0)
        brdf_a = _eval_pbr_brdf(n_ff, wo, wi_l, alb, metallic, rgh)
        geom_a = (cos_s * cos_l) / (dist_l * dist_l + 0.08) * area_area
        nee_a = jnp.minimum(thr * brdf_a * area_emit[None, None, :] * (geom_a * vis)[..., None], 14.0)
        rad = rad + jnp.where(is_opaque[..., None], nee_a, 0.0)

        if light_rig.has_sun:
          su = jax.random.uniform(k_su, (spp_par, R), minval=0.0, maxval=1.0, dtype=jnp.float32)
          sv = jax.random.uniform(k_sv, (spp_par, R), minval=0.0, maxval=1.0, dtype=jnp.float32)
          sr = sun_sin_max * jnp.sqrt(su)
          sphi = (2.0 * math.pi) * sv
          wi_sun = sun_d[None, None, :] + (sr * jnp.cos(sphi))[..., None] * st1[None, None, :] + (sr * jnp.sin(sphi))[..., None] * st2[None, None, :]
          wi_sun = wi_sun / jnp.maximum(jnp.linalg.norm(wi_sun, axis=-1, keepdims=True), 1e-12)
          cos_sun = jnp.maximum(jnp.sum(n_ff * wi_sun, axis=-1), 0.0)
          need_sun = is_opaque & (cos_sun > 0.0)
          sun_lim = jnp.where(need_sun, jnp.float32(zfar * 0.9), 0.0)
          d_sun, *_ = pbr_raycast_tpu(l2r, l2, l0, lf, sh_o, wi_sun, sun_lim, height, W_local, any_hit=True)
          vis_sun = jnp.where(need_sun & (d_sun >= sun_lim - 1e-2), 1.0, 0.0)
          brdf_sun = _eval_pbr_brdf(n_ff, wo, wi_sun, alb, metallic, rgh)
          nee_sun = jnp.minimum(thr * brdf_sun * sun_emit[None, None, :] * (cos_sun * vis_sun)[..., None], 18.0)
          rad = rad + jnp.where(is_opaque[..., None], nee_sun, 0.0)

        u_lobe = jax.random.uniform(k_lobe, (spp_par, R), minval=0.0, maxval=1.0, dtype=jnp.float32)
        u1 = jax.random.uniform(k_u1, (spp_par, R), minval=0.0, maxval=1.0, dtype=jnp.float32)
        u2 = jax.random.uniform(k_u2, (spp_par, R), minval=0.0, maxval=1.0, dtype=jnp.float32)

        p_spec = jnp.clip(0.15 + 0.78 * metallic, 0.15, 0.90)
        choose_spec = u_lobe < p_spec
        wi_diff = _sample_cosine_hemisphere(u1, u2, n_ff)
        alpha = jnp.maximum(rgh * rgh, 0.02)
        h_ggx = _sample_ggx_half(u1, u2, n_ff, alpha)
        wi_spec = 2.0 * jnp.sum(wo * h_ggx, axis=-1, keepdims=True) * h_ggx - wo
        wi_spec = wi_spec / jnp.maximum(jnp.linalg.norm(wi_spec, axis=-1, keepdims=True), 1e-12)
        wi_opq = jnp.where(choose_spec[..., None], wi_spec, wi_diff)

        ndotl_s = jnp.maximum(jnp.sum(n_ff * wi_spec, axis=-1), 0.0)
        ndotv = jnp.maximum(jnp.sum(n_ff * wo, axis=-1), 1e-4)
        ndoth = jnp.maximum(jnp.sum(n_ff * h_ggx, axis=-1), 1e-4)
        vdoth = jnp.maximum(jnp.sum(wo * h_ggx, axis=-1), 1e-4)
        f0 = 0.04 * (1.0 - metallic[..., None]) + alb * metallic[..., None]
        F_s = f0 + (1.0 - f0) * ((1.0 - vdoth) ** 5)[..., None]
        G_s = _smith_g1(ndotl_s, alpha) * _smith_g1(ndotv, alpha)
        w_spec = jnp.where(ndotl_s[..., None] > 0.0, jnp.minimum(F_s * ((G_s * vdoth) / jnp.maximum(ndotv * ndoth * p_spec, 1e-4))[..., None], 4.0), 0.0)
        w_diff = (1.0 - metallic[..., None]) * alb / jnp.maximum(1.0 - p_spec[..., None], 0.1)
        w_opq = jnp.where(choose_spec[..., None], w_spec, w_diff)

        eta = jnp.where(entering, 1.0 / 1.50, 1.50)
        cos_i = jnp.clip(jnp.sum(wo * n_ff, axis=-1), 0.0, 1.0)
        sin2_t = eta * eta * (1.0 - cos_i * cos_i)
        tir = sin2_t > 1.0
        r0_g = ((1.0 - eta) / (1.0 + eta)) ** 2
        fresnel_g = r0_g + (1.0 - r0_g) * ((1.0 - cos_i) ** 5)
        refl_dir = 2.0 * cos_i[..., None] * n_ff - wo
        cos_t = jnp.sqrt(jnp.maximum(1.0 - sin2_t, 0.0))
        refr_dir = eta[..., None] * rd + (eta * cos_i - cos_t)[..., None] * n_ff
        do_refl = tir | (u_lobe < fresnel_g)
        wi_glass = jnp.where(do_refl[..., None], refl_dir, refr_dir)
        wi_glass = wi_glass / jnp.maximum(jnp.linalg.norm(wi_glass, axis=-1, keepdims=True), 1e-12)
        ro_glass = jnp.where(do_refl[..., None], hit_pos + n_ff * 1.5e-3, hit_pos - n_ff * 1.5e-3)

        wi_next = jnp.where(is_glass[..., None], wi_glass, wi_opq)
        new_ro = jnp.where(is_glass[..., None], ro_glass, hit_pos + n_ff * 1.5e-3)
        bounce_w = jnp.where(is_glass[..., None], alb, w_opq)

        new_thr = thr * bounce_w
        new_alive = (is_opaque | is_glass) & (jnp.max(new_thr, axis=-1) > 1e-3)
        new_t_lim = jnp.broadcast_to(jnp.float32(zfar), (spp_par, R))
        return (new_ro, wi_next, new_t_lim, new_thr, new_alive, rad)

      _, _, _, _, _, final_rad = jax.lax.fori_loop(0, max_bounces, _bounce, (ro0, rd0, t_max0, thr0, alive0, rad0))
      return rad_acc + jnp.mean(final_rad, axis=0)

    rad_sum = jax.lax.fori_loop(0, spp_seq, _trace_batch_samples, jnp.zeros((R, 3), dtype=jnp.float32))
    rad_avg = (rad_sum / float(spp_seq)).reshape(height, W_local, 3)
    rgb = _agx_tonemap_srgb(rad_avg, exposure=light_rig.exposure)
    return rgb, gbuf_hw

  sm_kw = {"check_rep": False} if "check_rep" in inspect.signature(shard_map).parameters else {"check_vma": False}

  def _render_local(l2r, l2, l0, lf, cxp, cxm, sdir, ac, jit_in, cid_arr, rng_in):
    rgb, gbuf_hw = _trace_core(l2r, l2, l0, lf, cxp, cxm, sdir, ac, jit_in, cid_arr[0], rng_in)
    return rgb[None, ...], gbuf_hw[None, ...]

  sh_fn = shard_map(
      _render_local,
      mesh=mesh,
      in_specs=(P(), P(), P(), P(), P(), P(), P(), P(), P(), P("dp"), P()),
      out_specs=(P("dp"), P("dp")),
      **sm_kw,
  )

  @jax.jit
  def render_fn(cxp, cxm, sdir, ac, jit_in, rng):
    return sh_fn(l2_roots_d, l2_d, l0_d, leaf_d, cxp, cxm, sdir, ac, jit_in, chip_ids, rng)

  return render_fn, mesh, rep


def _stitch(arr_j: jax.Array) -> np.ndarray:
  arr = np.asarray(jax.device_get(arr_j), dtype=np.float32)
  n_c, H, W_loc, C = arr.shape
  return arr.transpose(1, 0, 2, 3).reshape(H, n_c * W_loc, C)


def render_sequence(
    scene_name: str,
    split: str,
    hr_res: int,
    n_frames: int,
    gt_spp: int,
    max_bounces: int,
    out_npz: str,
    seed_base: int = 10000,
):
  devs = jax.devices()
  assert len(devs) == 4, f"Expected 4 TPU chips, found {len(devs)}"
  lr_res = hr_res // 2
  bvh, _, _, fovy, zn, zf, rig = _build_scene(scene_name)
  cxps, cxms, sdirs, acs, jits = generate_sequence_trajectory(scene_name, split=split, n_frames=n_frames)

  aim_origin = (scene_name == "caustics_studio")
  fn_lr, _, rep = build_dynamic_pt_fn(
      bvh, devs, lr_res, lr_res, fovy, zn, zf, rig,
      spp=1, max_bounces=max_bounces, use_external_jitter=True, aim_area_at_origin=aim_origin,
  )
  fn_gt, _, _ = build_dynamic_pt_fn(
      bvh, devs, hr_res, hr_res, fovy, zn, zf, rig,
      spp=gt_spp, max_bounces=max_bounces, use_external_jitter=False, aim_area_at_origin=aim_origin,
  )

  n1_l, gb_l, gt_l = [], [], []
  t0 = time.perf_counter()
  for t in range(n_frames):
    cxp_d = jax.device_put(cxps[t], rep)
    cxm_d = jax.device_put(cxms[t], rep)
    sd_d = jax.device_put(sdirs[t], rep)
    ac_d = jax.device_put(acs[t], rep)
    jit_d = jax.device_put(jits[t], rep)
    r1, gb = fn_lr(cxp_d, cxm_d, sd_d, ac_d, jit_d, jax.device_put(jax.random.PRNGKey(seed_base + t), rep))
    rgt, _ = fn_gt(cxp_d, cxm_d, sd_d, ac_d, jit_d, jax.device_put(jax.random.PRNGKey(seed_base + 5000 + t), rep))
    jax.block_until_ready((r1, gb, rgt))
    n1_l.append(_stitch(r1))
    gb_l.append(_stitch(gb))
    gt_l.append(_stitch(rgt))
    if (t + 1) % 5 == 0 or t == 0 or t + 1 == n_frames:
      print(f"[{scene_name}/{split}] frame {t+1}/{n_frames} ({time.perf_counter()-t0:.1f}s)", flush=True)

  os.makedirs(os.path.dirname(out_npz) or ".", exist_ok=True)
  np.savez(
      out_npz,
      scene=np.array(scene_name),
      split=np.array(split),
      noisy1_lr=np.stack(n1_l, axis=0),
      gbuf_lr=np.stack(gb_l, axis=0),
      gt_hr=np.stack(gt_l, axis=0),
      cxp=cxps,
      cxm=cxms,
      sun_dir=sdirs,
      area_center=acs,
      jitter_lr=jits,
      cam_fovy=np.float32(fovy),
      znear=np.float32(zn),
      zfar=np.float32(zf),
  )
  print(f"[{scene_name}/{split}] saved -> {out_npz} ({time.perf_counter()-t0:.1f}s total)", flush=True)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--scene", required=True)
  ap.add_argument("--split", choices=["train", "eval"], required=True)
  ap.add_argument("--hr_res", type=int, default=512)
  ap.add_argument("--n_frames", type=int, default=30)
  ap.add_argument("--gt_spp", type=int, default=256)
  ap.add_argument("--max_bounces", type=int, default=4)
  ap.add_argument("--out", required=True)
  ap.add_argument("--seed_base", type=int, default=10000)
  args = ap.parse_args()
  render_sequence(
      args.scene, args.split, args.hr_res, args.n_frames, args.gt_spp, args.max_bounces, args.out, args.seed_base
  )


if __name__ == "__main__":
  main()
