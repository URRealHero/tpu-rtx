"""Multi-Precision (FP32 / BF16 / INT8 W8A8) FiLM-Conditioned MXU Neural Relighter.

Implements:
  - Zero-raytrace physical feature builder `build_relight_features(gb, cond)` with
    live hero material override, GGX/Schlick BRDF proxies, and multi-frequency
    Fourier spatial/reflection encodings.
  - SpaceToDepth(2) 3-scale MXU U-Net + 1x1 `dot_general` pixel BRDF head with
    continuous Time-of-Day & Material FiLM conditioning at every scale.
  - Exact shared-weight execution across three hardware precision modes:
      * "fp32" : `float32` + `Precision.HIGHEST` (true FP32 multi-pass MXU)
      * "bf16" : `bfloat16` native MXU
      * "int8" : Static W8A8 (`int8` weights + `int8` activations -> `int32` acc
                 via `jax.lax.conv_general_dilated` & `jax.lax.dot_general`)
  - QAT (Straight-Through Estimator) fine-tuning + PTQ activation calibration.
  - 1-chip and 4-chip `shard_map` wrappers with strict zero-copy HLO auditing.
"""

from __future__ import annotations

import inspect
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import jax
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np

from bvh_pbr_pt import _agx_tonemap_srgb
from tpu_mjx_render.roofline import inspect_hlo_ops


def audit_hlo_strict(lowered: Any) -> Dict[str, Any]:
  txt = lowered.as_text()
  info = inspect_hlo_ops(txt)
  info["send_recv_ops"] = len(re.findall(r"\bsend\(|\brecv\(", txt, re.I))
  assert info["host_callbacks"] == 0, f"HLO contains host callbacks: {info}"
  assert info["send_recv_ops"] == 0, f"HLO contains send/recv ops: {info}"
  assert info["gather_ops"] == 0, f"HLO contains gather ops: {info}"
  return info


# ==============================================================================
# 1. Zero-Raytrace Physical Feature Builder + Exact Sky Evaluator
# ==============================================================================

def _eval_sky_tonemapped(
    rd: jax.Array,       # (B, H, W, 3) pointing away from camera
    sun_dir: jax.Array,  # (B, 1, 1, 3)
    sun_emit: jax.Array, # (B, 1, 1, 3)
    sky_z: jax.Array,    # (B, 1, 1, 3)
    sky_h: jax.Array,    # (B, 1, 1, 3)
    exposure: float = 0.95,
) -> jax.Array:
  z = rd[..., 2]
  tu = jnp.clip(z, 0.0, 1.0)[..., None]
  td = jnp.clip(-z, 0.0, 1.0)[..., None]
  sky_g = jnp.asarray([0.06, 0.055, 0.05], dtype=jnp.float32)
  s = jnp.where(z[..., None] >= 0.0, (1.0 - tu) * sky_h + tu * sky_z, (1.0 - td) * sky_h + td * sky_g)
  cos_s = jnp.maximum(jnp.sum(rd * sun_dir, axis=-1), 0.0)
  s = s + (cos_s ** 32.0)[..., None] * sun_emit * 0.25
  return jnp.clip(_agx_tonemap_srgb(s, exposure=exposure), 0.0, 1.0)


def apply_live_material_override(gb: jax.Array, cond: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
  """Override hero-mask albedo/roughness/metallic on-device with zero raytracing."""
  albedo = gb[..., 0:3]
  normal = gb[..., 3:6]
  depth_norm = gb[..., 6:7]
  roughness = gb[..., 7:8]
  mcode_norm = gb[..., 8:9]
  hero_mask = gb[..., 15:16] > 0.5

  mat_rgh = cond[:, None, None, 16:17]
  mat_met = cond[:, None, None, 17:18]
  mat_alb = cond[:, None, None, 18:21]

  albedo = jnp.where(hero_mask, mat_alb, albedo)
  roughness = jnp.where(hero_mask, mat_rgh, roughness)
  metallic = jnp.where(hero_mask, mat_met, jnp.clip(mcode_norm * 2.0, 0.0, 1.0))
  is_glass = (mcode_norm >= 0.85) & (mcode_norm < 1.35) & (~hero_mask)
  is_emit = (mcode_norm >= 1.35) & (~hero_mask)
  return albedo, roughness, metallic, is_glass.astype(jnp.float32), is_emit.astype(jnp.float32)


def build_relight_features(gb: jax.Array, cond: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
  """Build (B, H, W, 48) physical + Fourier G-buffer features and auxiliary shading masks.

  Returns:
    feats_48  : (B, H, W, 48) float32
    albedo    : (B, H, W, 3)
    metallic  : (B, H, W, 1)
    hit_mask  : (B, H, W, 1)
    is_emit   : (B, H, W, 1)
    sky_rgb   : (B, H, W, 3)
  """
  albedo, roughness, metallic, is_glass, is_emit = apply_live_material_override(gb, cond)
  normal = gb[..., 3:6]
  depth_norm = gb[..., 6:7]
  world_pos = gb[..., 9:12]
  view_dir = gb[..., 12:15]  # points toward camera (-rd)
  hit_mask = (depth_norm >= 0.0).astype(jnp.float32)

  sun_dir_raw = cond[:, None, None, 0:3]
  sun_dir = sun_dir_raw / jnp.maximum(jnp.linalg.norm(sun_dir_raw, axis=-1, keepdims=True), 1e-12)
  sun_emit_n = cond[:, None, None, 3:6]
  sky_z = cond[:, None, None, 6:9]
  sky_h = cond[:, None, None, 9:12]
  area_emit_n = cond[:, None, None, 12:15]

  sky_rgb = _eval_sky_tonemapped(-view_dir, sun_dir, sun_emit_n * 30.0, sky_z, sky_h, exposure=0.95)
  albedo = jnp.where(hit_mask > 0.5, albedo, sky_rgb)

  ndotv_raw = jnp.sum(normal * view_dir, axis=-1, keepdims=True)
  ndotv = jnp.clip(ndotv_raw, 1e-4, 1.0)
  refl_dir = 2.0 * ndotv_raw * normal - view_dir
  refl_dir = refl_dir / jnp.maximum(jnp.linalg.norm(refl_dir, axis=-1, keepdims=True), 1e-12)

  ndotl_sun = jnp.maximum(jnp.sum(normal * sun_dir, axis=-1, keepdims=True), 0.0)
  h_sun = view_dir + sun_dir
  h_sun = h_sun / jnp.maximum(jnp.linalg.norm(h_sun, axis=-1, keepdims=True), 1e-12)
  ndoth_sun = jnp.maximum(jnp.sum(normal * h_sun, axis=-1, keepdims=True), 0.0)
  vdoth_sun = jnp.maximum(jnp.sum(view_dir * h_sun, axis=-1, keepdims=True), 0.0)

  alpha = jnp.maximum(roughness * roughness, 0.02)
  a2 = alpha * alpha
  denom_d = ndoth_sun * ndoth_sun * (a2 - 1.0) + 1.0
  ggx_d = jnp.clip(a2 / (math.pi * denom_d * denom_d + 1e-4), 0.0, 16.0) * 0.125

  f0 = 0.04 * (1.0 - metallic) + albedo * metallic
  fresnel_v = f0 + (1.0 - f0) * ((1.0 - ndotv) ** 5)

  sky_hemi = 0.5 + 0.5 * normal[..., 2:3]
  direct_diff = albedo * (1.0 - metallic) * (
      sun_emit_n * ndotl_sun + 0.45 * area_emit_n * jnp.clip(0.35 + 0.65 * normal[..., 2:3], 0.0, 1.0) + 0.35 * sky_z * sky_hemi
  )
  direct_spec = fresnel_v * (ggx_d * sun_emit_n + 0.40 * sky_z)

  # 4-neighbor normal & depth edge discontinuity features
  B, H, W, _ = gb.shape
  npad = jnp.pad(normal, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="edge")
  zpad = jnp.pad(jnp.where(hit_mask > 0.5, depth_norm, 2.0), ((0, 0), (1, 1), (1, 1), (0, 0)), mode="edge")
  dn_y = npad[:, 2 : H + 2, 1 : W + 1] - npad[:, :H, 1 : W + 1]
  dn_x = npad[:, 1 : H + 1, 2 : W + 2] - npad[:, 1 : H + 1, :W]
  n_edge = jnp.clip(
      0.5 * (
          jnp.sqrt(jnp.maximum(jnp.sum(dn_y * dn_y, axis=-1, keepdims=True), 1e-6))
          + jnp.sqrt(jnp.maximum(jnp.sum(dn_x * dn_x, axis=-1, keepdims=True), 1e-6))
      ),
      0.0,
      2.0,
  )
  z_edge = jnp.clip(
      jnp.abs(zpad[:, 2 : H + 2, 1 : W + 1] - zpad[:, :H, 1 : W + 1])
      + jnp.abs(zpad[:, 1 : H + 1, 2 : W + 2] - zpad[:, 1 : H + 1, :W]),
      0.0,
      1.0,
  )

  # Multi-frequency Fourier positional & reflection encodings (resolves slatted shadows & reflections)
  wp_n = world_pos * 0.35
  pe_wp1_s = jnp.sin(math.pi * wp_n)
  pe_wp1_c = jnp.cos(math.pi * wp_n)
  pe_wp2_s = jnp.sin(2.0 * math.pi * wp_n)
  pe_wp2_c = jnp.cos(2.0 * math.pi * wp_n)
  pe_wp3_s = jnp.sin(4.0 * math.pi * wp_n)
  pe_wp3_c = jnp.cos(4.0 * math.pi * wp_n)
  pe_rf_s = jnp.sin(math.pi * refl_dir)
  pe_rf_c = jnp.cos(math.pi * refl_dir)

  temp_b = jnp.broadcast_to(cond[:, None, None, 15:16], (B, H, W, 1))
  sun_d_b = jnp.broadcast_to(sun_dir, (B, H, W, 3))

  feats = jnp.concatenate(
      [
          albedo,        # 3
          normal,        # 3
          depth_norm,    # 1
          roughness,     # 1
          metallic,      # 1
          is_glass,      # 1
          is_emit,       # 1
          hit_mask,      # 1  -> 12
          wp_n,          # 3
          view_dir,      # 3
          refl_dir,      # 3  -> 21
          ndotl_sun,     # 1
          ndotv,         # 1
          ndoth_sun,     # 1
          ggx_d,         # 1
          fresnel_v,     # 3
          direct_diff,   # 3
          direct_spec,   # 3  -> 34
          n_edge,        # 1
          z_edge,        # 1  -> 36
          pe_wp1_s,      # 3
          pe_wp1_c,      # 3
          pe_wp2_s,      # 3
          pe_wp2_c,      # 3  -> 48
          pe_wp3_s,      # 3
          pe_wp3_c,      # 3
          pe_rf_s,       # 3
          pe_rf_c,       # 3  -> 60
          sun_d_b,       # 3
          temp_b,        # 1  -> 64
      ],
      axis=-1,
  )
  return feats, albedo, metallic, hit_mask, is_emit, sky_rgb


# ==============================================================================
# 2. Explicit Parameter Initialization & Multi-Precision Primitives
# ==============================================================================

def _init_conv(key: jax.Array, kh: int, kw: int, ci: int, co: int) -> Dict[str, jax.Array]:
  std = math.sqrt(2.0 / max(kh * kw * ci, 1))
  w = jax.random.normal(key, (kh, kw, ci, co), dtype=jnp.float32) * std
  b = jnp.zeros((co,), dtype=jnp.float32)
  return {"w": w, "b": b}


def _init_linear(key: jax.Array, ci: int, co: int, scale: float = 1.0) -> Dict[str, jax.Array]:
  std = math.sqrt(2.0 / max(ci, 1)) * scale
  w = jax.random.normal(key, (ci, co), dtype=jnp.float32) * std
  b = jnp.zeros((co,), dtype=jnp.float32)
  return {"w": w, "b": b}


def init_relighter_params(
    seed: int = 2026,
    c1: int = 64,
    c2: int = 128,
    c3: int = 128,
    c_pix: int = 32,
    head_w: int = 64,
    cond_dim: int = 24,
) -> Dict[str, Any]:
  k = jax.random.PRNGKey(seed)
  keys = jax.random.split(k, 28)
  it = iter(keys)

  film_h = 64
  # FiLM MLP maps cond(24) -> 64 -> per-scale (gamma, beta)
  film = {
      "fc0": _init_linear(next(it), cond_dim, film_h),
      "fc1": _init_linear(next(it), film_h, film_h),
      "p_s1": _init_linear(next(it), film_h, 2 * c1, scale=0.15),
      "p_s2": _init_linear(next(it), film_h, 2 * c2, scale=0.15),
      "p_s3": _init_linear(next(it), film_h, 2 * c3, scale=0.15),
      "p_hd": _init_linear(next(it), film_h, 2 * head_w, scale=0.15),
  }
  # Zero-init FiLM gamma bias -> 1.0, beta bias -> 0.0
  for pk, c_out in [("p_s1", c1), ("p_s2", c2), ("p_s3", c3), ("p_hd", head_w)]:
    film[pk]["b"] = jnp.concatenate([jnp.ones((c_out,), jnp.float32), jnp.zeros((c_out,), jnp.float32)])

  # 13 main spatial/head layers + activation scale parameters for QAT/PTQ
  params = {
      "film": film,
      "stem": _init_conv(next(it), 3, 3, 256, c1),
      "res1_a": _init_conv(next(it), 3, 3, c1, c1),
      "res1_b": _init_conv(next(it), 3, 3, c1, c1),
      "down2": _init_conv(next(it), 3, 3, c1, c2),
      "res2_a": _init_conv(next(it), 3, 3, c2, c2),
      "res2_b": _init_conv(next(it), 3, 3, c2, c2),
      "down3": _init_conv(next(it), 3, 3, c2, c3),
      "res3_a": _init_conv(next(it), 3, 3, c3, c3),
      "res3_b": _init_conv(next(it), 3, 3, c3, c3),
      "up3": _init_conv(next(it), 3, 3, c2 + c3, c2),
      "up2": _init_conv(next(it), 3, 3, c1 + c2, 4 * c_pix),
      "head1": _init_linear(next(it), c_pix + 32, head_w),
      "head2": _init_linear(next(it), head_w, 6, scale=0.5),
  }
  # Learnable positive log activation clip scales for QAT (initialized to ~6.0)
  act_names = [
      "s2d", "stem", "res1_a", "res1_b", "down2", "res2_a", "res2_b",
      "down3", "res3_a", "res3_b", "cat2", "up3", "cat1", "up2", "head_in", "head1",
  ]
  params["act_scales"] = {name: jnp.asarray(6.0, dtype=jnp.float32) for name in act_names}
  return params


def _eval_film(film_params: Dict[str, Any], cond: jax.Array) -> Dict[str, Tuple[jax.Array, jax.Array]]:
  h = jax.nn.silu(cond @ film_params["fc0"]["w"] + film_params["fc0"]["b"])
  h = jax.nn.silu(h @ film_params["fc1"]["w"] + film_params["fc1"]["b"])
  out = {}
  for pk, name in [("p_s1", "s1"), ("p_s2", "s2"), ("p_s3", "s3"), ("p_hd", "hd")]:
    gb = h @ film_params[pk]["w"] + film_params[pk]["b"]
    c = gb.shape[-1] // 2
    gamma = gb[..., :c][:, None, None, :]
    beta = gb[..., c:][:, None, None, :]
    out[name] = (gamma, beta)
  return out


def _fake_quant_sym(x: jax.Array, scale: jax.Array, qmax: float = 127.0) -> jax.Array:
  """Straight-Through Estimator (STE) symmetric fake quantization with clamped gradient."""
  s = jnp.maximum(scale, 1e-6)
  lim = qmax * s
  x_c = jnp.clip(x, -lim, lim)
  q = jnp.round(x_c / s)
  x_q = q * s
  return x_c + jax.lax.stop_gradient(x_q - x_c)


def _quant_weight_per_ch_ste(w: jax.Array) -> jax.Array:
  axes = tuple(range(w.ndim - 1))
  s_w = jnp.maximum(jnp.max(jnp.abs(w), axis=axes, keepdims=True) / 127.0, 1e-6)
  q = jnp.clip(jnp.round(w / s_w), -127.0, 127.0)
  w_q = q * s_w
  return w + jax.lax.stop_gradient(w_q - w)


def quantize_params_static_int8(params: Dict[str, Any]) -> Dict[str, Any]:
  """Convert trained params + act_scales into static INT8 W8A8 weights & scales."""
  q_layers: Dict[str, Any] = {}
  # Quantize layer parameters for ALL 13 conv + dot_general layers
  layer_names = [
      "stem", "res1_a", "res1_b", "down2", "res2_a", "res2_b",
      "down3", "res3_a", "res3_b", "up3", "up2", "head1", "head2",
  ]
  for lname in layer_names:
    w = np.asarray(params[lname]["w"], dtype=np.float32)
    b = np.asarray(params[lname]["b"], dtype=np.float32)
    axes = tuple(range(w.ndim - 1))
    s_w = np.maximum(np.max(np.abs(w), axis=axes) / 127.0, 1e-7).astype(np.float32)
    w_q = np.clip(np.rint(w / s_w), -127.0, 127.0).astype(np.int8)
    q_layers[lname] = {"w_q": jnp.asarray(w_q), "s_w": jnp.asarray(s_w), "b": jnp.asarray(b)}

  act_s = {k: jnp.maximum(jnp.asarray(v, dtype=jnp.float32), 1e-5) for k, v in params["act_scales"].items()}
  return {
      "film": params["film"],
      "layers": q_layers,
      "act_scales": act_s,
  }


def _conv2d_exec(
    x: jax.Array,
    w: jax.Array,
    stride: int,
    precision_mode: str,
) -> jax.Array:
  dn = jax.lax.conv_dimension_numbers(x.shape, w.shape, ("NHWC", "HWIO", "NHWC"))
  if precision_mode == "fp32":
    return jax.lax.conv_general_dilated(
        x.astype(jnp.float32),
        w.astype(jnp.float32),
        (stride, stride),
        "SAME",
        dimension_numbers=dn,
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
  elif precision_mode == "bf16":
    return jax.lax.conv_general_dilated(
        x.astype(jnp.bfloat16),
        w.astype(jnp.bfloat16),
        (stride, stride),
        "SAME",
        dimension_numbers=dn,
        preferred_element_type=jnp.bfloat16,
    )
  else:
    raise ValueError(f"Unsupported conv mode: {precision_mode}")


def _dot1x1_exec(x: jax.Array, w: jax.Array, precision_mode: str) -> jax.Array:
  if precision_mode == "fp32":
    return jax.lax.dot_general(
        x.astype(jnp.float32),
        w.astype(jnp.float32),
        (((3,), (0,)), ((), ())),
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
  elif precision_mode == "bf16":
    return jax.lax.dot_general(
        x.astype(jnp.bfloat16),
        w.astype(jnp.bfloat16),
        (((3,), (0,)), ((), ())),
        preferred_element_type=jnp.bfloat16,
    )
  else:
    raise ValueError(f"Unsupported dot mode: {precision_mode}")


def relight_forward_float_or_qat(
    params: Dict[str, Any],
    gb: jax.Array,
    cond: jax.Array,
    precision_mode: str = "bf16",
    qat: bool = False,
) -> jax.Array:
  """Execute Relighter in FP32, BF16, or QAT-simulated W8A8 mode."""
  feats, albedo, metallic, hit_mask, is_emit, sky_rgb = jax.lax.optimization_barrier(
      build_relight_features(gb, cond)
  )
  B, H, W, C = feats.shape
  dt = jnp.float32 if precision_mode == "fp32" else jnp.bfloat16
  act_s = params["act_scales"]

  def _q_act(t, name):
    if not qat:
      return t
    s = jnp.maximum(jnp.abs(act_s[name]), 1e-4) / 127.0
    return _fake_quant_sym(t.astype(jnp.float32), s).astype(dt)

  def _q_w(w):
    if not qat:
      return w
    return _quant_weight_per_ch_ste(w)

  film = _eval_film(params["film"], cond)

  def _conv_act(x, lname, stride=1, film_key=None, res_add=None, out_act_name=None):
    w = _q_w(params[lname]["w"])
    b = params[lname]["b"].astype(dt)[None, None, None, :]
    y = _conv2d_exec(x, w, stride, precision_mode) + b
    if film_key is not None:
      gm, bt = film[film_key]
      y = y * gm.astype(dt) + bt.astype(dt)
    if res_add is not None:
      y = y + res_add.astype(dt)
    y = jax.nn.relu(y)
    if out_act_name is not None:
      y = _q_act(y, out_act_name)
    return y

  # SpaceToDepth(2): (B, H, W, 48) -> (B, H/2, W/2, 192)
  s2d = (
      feats.astype(dt)
      .reshape(B, H // 2, 2, W // 2, 2, C)
      .transpose(0, 1, 3, 2, 4, 5)
      .reshape(B, H // 2, W // 2, 4 * C)
  )
  s2d = _q_act(s2d, "s2d")

  h1 = _conv_act(s2d, "stem", 1, "s1", None, "stem")
  r1 = _conv_act(h1, "res1_a", 1, "s1", None, "res1_a")
  h1 = _conv_act(r1, "res1_b", 1, None, h1, "res1_b")

  h2 = _conv_act(h1, "down2", 2, "s2", None, "down2")
  r2 = _conv_act(h2, "res2_a", 1, "s2", None, "res2_a")
  h2 = _conv_act(r2, "res2_b", 1, None, h2, "res2_b")

  h3 = _conv_act(h2, "down3", 2, "s3", None, "down3")
  r3 = _conv_act(h3, "res3_a", 1, "s3", None, "res3_a")
  h3 = _conv_act(r3, "res3_b", 1, None, h3, "res3_b")

  h3_up = jnp.repeat(jnp.repeat(h3, 2, axis=1), 2, axis=2)
  cat2 = _q_act(jnp.concatenate([h2, h3_up], axis=-1), "cat2")
  d2 = _conv_act(cat2, "up3", 1, "s2", None, "up3")

  d2_up = jnp.repeat(jnp.repeat(d2, 2, axis=1), 2, axis=2)
  cat1 = _q_act(jnp.concatenate([h1, d2_up], axis=-1), "cat1")
  d1 = _conv_act(cat1, "up2", 1, None, None, "up2")

  c_pix = d1.shape[-1] // 4
  h_pix = (
      d1.reshape(B, H // 2, W // 2, 2, 2, c_pix)
      .transpose(0, 1, 3, 2, 4, 5)
      .reshape(B, H, W, c_pix)
  )

  # Pixel-wise 1x1 BRDF head via dot_general
  head_in = jnp.concatenate([h_pix, feats[..., :32].astype(dt)], axis=-1)
  head_in = _q_act(head_in, "head_in")

  w_h1 = _q_w(params["head1"]["w"])
  b_h1 = params["head1"]["b"].astype(dt)[None, None, None, :]
  gm_h, bt_h = film["hd"]
  h_hd = _dot1x1_exec(head_in, w_h1, precision_mode) + b_h1
  h_hd = jax.nn.relu(h_hd * gm_h.astype(dt) + bt_h.astype(dt))
  h_hd = _q_act(h_hd, "head1")

  w_h2 = _q_w(params["head2"]["w"])
  raw_out = _dot1x1_exec(h_hd, w_h2, precision_mode) + params["head2"]["b"].astype(dt)[None, None, None, :]
  raw_f32 = raw_out.astype(jnp.float32)

  irrad_diff = raw_f32[..., 0:3]
  irrad_spec = raw_f32[..., 3:6]
  kd = albedo * (1.0 - metallic)
  f0 = 0.04 * (1.0 - metallic) + albedo * metallic
  pred_hit = kd * irrad_diff + f0 * irrad_spec + is_emit * albedo
  return jnp.clip(jnp.where(hit_mask > 0.5, pred_hit, sky_rgb), 0.0, 1.0)


def relight_forward_int8(
    q_params: Dict[str, Any],
    gb: jax.Array,
    cond: jax.Array,
) -> jax.Array:
  """Execute static W8A8 Relighter on TPU v5p INT8 MXU (conv_general_dilated + dot_general -> int32)."""
  feats, albedo, metallic, hit_mask, is_emit, sky_rgb = jax.lax.optimization_barrier(
      build_relight_features(gb, cond)
  )
  B, H, W, C = feats.shape
  layers = q_params["layers"]
  act_s = q_params["act_scales"]
  film = _eval_film(q_params["film"], cond)
  dt = jnp.bfloat16

  def _quant_tensor(x: jax.Array, scale_max: jax.Array) -> jax.Array:
    inv_s = (127.0 / scale_max).astype(dt)
    return jnp.clip(jnp.rint(x.astype(dt) * inv_s), -127.0, 127.0).astype(jnp.int8)

  def _conv_i8_dequant(x_q: jax.Array, lname: str, in_act_name: str, stride: int = 1,
                       film_key: Optional[str] = None, res_bf: Optional[jax.Array] = None) -> jax.Array:
    ly = layers[lname]
    dn = jax.lax.conv_dimension_numbers(x_q.shape, ly["w_q"].shape, ("NHWC", "HWIO", "NHWC"))
    acc_i32 = jax.lax.conv_general_dilated(
        x_q, ly["w_q"], (stride, stride), "SAME", dimension_numbers=dn, preferred_element_type=jnp.int32
    )
    sc = ((act_s[in_act_name] / 127.0) * ly["s_w"]).astype(dt)[None, None, None, :]
    y = acc_i32.astype(dt) * sc + ly["b"].astype(dt)[None, None, None, :]
    if film_key is not None:
      gm, bt = film[film_key]
      y = y * gm.astype(dt) + bt.astype(dt)
    if res_bf is not None:
      y = y + res_bf
    return jax.nn.relu(y)

  s2d = (
      feats.astype(dt)
      .reshape(B, H // 2, 2, W // 2, 2, C)
      .transpose(0, 1, 3, 2, 4, 5)
      .reshape(B, H // 2, W // 2, 4 * C)
  )
  s2d_q = _quant_tensor(s2d, act_s["s2d"])

  h1 = _conv_i8_dequant(s2d_q, "stem", "s2d", 1, "s1", None)
  r1 = _conv_i8_dequant(_quant_tensor(h1, act_s["stem"]), "res1_a", "stem", 1, "s1", None)
  h1 = _conv_i8_dequant(_quant_tensor(r1, act_s["res1_a"]), "res1_b", "res1_a", 1, None, h1)

  h2 = _conv_i8_dequant(_quant_tensor(h1, act_s["res1_b"]), "down2", "res1_b", 2, "s2", None)
  r2 = _conv_i8_dequant(_quant_tensor(h2, act_s["down2"]), "res2_a", "down2", 1, "s2", None)
  h2 = _conv_i8_dequant(_quant_tensor(r2, act_s["res2_a"]), "res2_b", "res2_a", 1, None, h2)

  h3 = _conv_i8_dequant(_quant_tensor(h2, act_s["res2_b"]), "down3", "res2_b", 2, "s3", None)
  r3 = _conv_i8_dequant(_quant_tensor(h3, act_s["down3"]), "res3_a", "down3", 1, "s3", None)
  h3 = _conv_i8_dequant(_quant_tensor(r3, act_s["res3_a"]), "res3_b", "res3_a", 1, None, h3)

  h3_up = jnp.repeat(jnp.repeat(h3, 2, axis=1), 2, axis=2)
  cat2_q = _quant_tensor(jnp.concatenate([h2, h3_up], axis=-1), act_s["cat2"])
  d2 = _conv_i8_dequant(cat2_q, "up3", "cat2", 1, "s2", None)

  d2_up = jnp.repeat(jnp.repeat(d2, 2, axis=1), 2, axis=2)
  cat1_q = _quant_tensor(jnp.concatenate([h1, d2_up], axis=-1), act_s["cat1"])
  d1 = _conv_i8_dequant(cat1_q, "up2", "cat1", 1, None, None)

  c_pix = d1.shape[-1] // 4
  h_pix = (
      d1.reshape(B, H // 2, W // 2, 2, 2, c_pix)
      .transpose(0, 1, 3, 2, 4, 5)
      .reshape(B, H, W, c_pix)
  )
  head_in_q = _quant_tensor(jnp.concatenate([h_pix, feats[..., :32].astype(dt)], axis=-1), act_s["head_in"])

  # 1x1 pixel head via INT8 jax.lax.dot_general -> int32!
  ly_h1 = layers["head1"]
  acc_h1 = jax.lax.dot_general(
      head_in_q, ly_h1["w_q"], (((3,), (0,)), ((), ())), preferred_element_type=jnp.int32
  )
  sc_h1 = ((act_s["head_in"] / 127.0) * ly_h1["s_w"]).astype(dt)[None, None, None, :]
  gm_h, bt_h = film["hd"]
  h_hd = jax.nn.relu((acc_h1.astype(dt) * sc_h1 + ly_h1["b"].astype(dt)[None, None, None, :]) * gm_h.astype(dt) + bt_h.astype(dt))
  h_hd_q = _quant_tensor(h_hd, act_s["head1"])

  # Final 6-ch projection via INT8 jax.lax.dot_general -> int32 -> float32 (zero 8-bit output banding)
  ly_h2 = layers["head2"]
  acc_h2 = jax.lax.dot_general(
      h_hd_q, ly_h2["w_q"], (((3,), (0,)), ((), ())), preferred_element_type=jnp.int32
  )
  sc_h2 = ((act_s["head1"] / 127.0) * ly_h2["s_w"])[None, None, None, :]
  raw_out = acc_h2.astype(jnp.float32) * sc_h2 + ly_h2["b"][None, None, None, :]

  irrad_diff = raw_out[..., 0:3]
  irrad_spec = raw_out[..., 3:6]
  kd = albedo * (1.0 - metallic)
  f0 = 0.04 * (1.0 - metallic) + albedo * metallic
  pred_hit = kd * irrad_diff + f0 * irrad_spec + is_emit * albedo
  return jnp.clip(jnp.where(hit_mask > 0.5, pred_hit, sky_rgb), 0.0, 1.0)



def calibrate_act_scales(
    params: Dict[str, Any],
    gb_batch: jax.Array,
    cond_batch: jax.Array,
    percentile: float = 99.95,
) -> Dict[str, jax.Array]:
  """Collect per-layer activation clip scales (99.95th percentile) for PTQ / QAT init."""
  feats, *_ = build_relight_features(gb_batch, cond_batch)
  B, H, W, C = feats.shape
  dt = jnp.float32
  film = _eval_film(params["film"], cond_batch)
  stats: Dict[str, float] = {}

  def _rec(name: str, t: jax.Array) -> jax.Array:
    arr = np.abs(np.asarray(jax.device_get(t), dtype=np.float32))
    val = float(np.percentile(arr, percentile))
    stats[name] = max(val, 1e-3)
    return t

  def _conv_f(x, lname, stride=1, film_key=None, res_add=None):
    w = params[lname]["w"]
    b = params[lname]["b"][None, None, None, :]
    y = _conv2d_exec(x, w, stride, "fp32") + b
    if film_key is not None:
      gm, bt = film[film_key]
      y = y * gm + bt
    if res_add is not None:
      y = y + res_add
    return jax.nn.relu(y)

  s2d = (
      feats.astype(dt)
      .reshape(B, H // 2, 2, W // 2, 2, C)
      .transpose(0, 1, 3, 2, 4, 5)
      .reshape(B, H // 2, W // 2, 4 * C)
  )
  _rec("s2d", s2d)

  h1 = _rec("stem", _conv_f(s2d, "stem", 1, "s1", None))
  r1 = _rec("res1_a", _conv_f(h1, "res1_a", 1, "s1", None))
  h1 = _rec("res1_b", _conv_f(r1, "res1_b", 1, None, h1))

  h2 = _rec("down2", _conv_f(h1, "down2", 2, "s2", None))
  r2 = _rec("res2_a", _conv_f(h2, "res2_a", 1, "s2", None))
  h2 = _rec("res2_b", _conv_f(r2, "res2_b", 1, None, h2))

  h3 = _rec("down3", _conv_f(h2, "down3", 2, "s3", None))
  r3 = _rec("res3_a", _conv_f(h3, "res3_a", 1, "s3", None))
  h3 = _rec("res3_b", _conv_f(r3, "res3_b", 1, None, h3))

  h3_up = jnp.repeat(jnp.repeat(h3, 2, axis=1), 2, axis=2)
  cat2 = _rec("cat2", jnp.concatenate([h2, h3_up], axis=-1))
  d2 = _rec("up3", _conv_f(cat2, "up3", 1, "s2", None))

  d2_up = jnp.repeat(jnp.repeat(d2, 2, axis=1), 2, axis=2)
  cat1 = _rec("cat1", jnp.concatenate([h1, d2_up], axis=-1))
  d1 = _rec("up2", _conv_f(cat1, "up2", 1, None, None))

  c_pix = d1.shape[-1] // 4
  h_pix = (
      d1.reshape(B, H // 2, W // 2, 2, 2, c_pix)
      .transpose(0, 1, 3, 2, 4, 5)
      .reshape(B, H, W, c_pix)
  )
  head_in = _rec("head_in", jnp.concatenate([h_pix, feats[..., :32]], axis=-1))
  gm_h, bt_h = film["hd"]
  h_hd = jax.nn.relu((_dot1x1_exec(head_in, params["head1"]["w"], "fp32") + params["head1"]["b"][None, None, None, :]) * gm_h + bt_h)
  _rec("head1", h_hd)
  return {k: jnp.asarray(v, dtype=jnp.float32) for k, v in stats.items()}


def benchmark_precision_pareto(
    params_fp32: Dict[str, Any],
    q_params_int8: Dict[str, Any],
    devs_all: Sequence[jax.Device],
    resolutions: Sequence[Tuple[int, int]] = ((1080, 1920), (2160, 3840)),
    chip_counts: Sequence[int] = (1, 4),
    precisions: Sequence[str] = ("fp32", "bf16", "int8"),
) -> List[Dict[str, Any]]:
  """Benchmark FP32 vs BF16 vs INT8 W8A8 on 1-chip and 4-chip at 1080p and Native 4K."""
  import time

  records: List[Dict[str, Any]] = []
  sm_kw = {"check_rep": False} if "check_rep" in inspect.signature(shard_map).parameters else {"check_vma": False}

  for H, W in resolutions:
    for n_chips in chip_counts:
      devs = list(devs_all[:n_chips])
      for prec in precisions:
        n_steps = 12 if H <= 1080 else 8
        if prec == "fp32" and H >= 2160 and n_chips == 1:
          n_steps = 5

        cond_np = np.full((1, 24), 0.25, dtype=np.float32)
        gb_np = np.full((1, H, W, 16), 0.4, dtype=np.float32)
        gb_np[..., 3:6] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        gb_np[..., 12:15] = np.array([0.0, -1.0, 0.0], dtype=np.float32)

        if n_chips == 1:
          dev = devs[0]
          p_src = q_params_int8 if prec == "int8" else params_fp32
          p_d = jax.device_put(jax.tree_util.tree_map(jnp.asarray, p_src), dev)
          gb_d = jax.device_put(gb_np, dev)
          cond_d = jax.device_put(cond_np, dev)

          if prec == "int8":
            @jax.jit
            def run_1c(p_in, gb_in, c_in):
              return relight_forward_int8(p_in, gb_in, c_in)
          else:
            @jax.jit
            def run_1c(p_in, gb_in, c_in, _pm=prec):
              return relight_forward_float_or_qat(p_in, gb_in, c_in, precision_mode=_pm, qat=False)

          hlo_info = audit_hlo_strict(run_1c.lower(p_d, gb_d, cond_d))
          for _ in range(3):
            y = run_1c(p_d, gb_d, cond_d)
          jax.block_until_ready(y)
          t0 = time.perf_counter()
          for _ in range(n_steps):
            y = run_1c(p_d, gb_d, cond_d)
          jax.block_until_ready(y)
          ms = (time.perf_counter() - t0) * 1e3 / n_steps
        else:
          mesh = Mesh(np.array(devs), ("dp",))
          rep = NamedSharding(mesh, P())
          sh_w = NamedSharding(mesh, P(None, None, "dp", None))
          p_src = q_params_int8 if prec == "int8" else params_fp32
          p_d = jax.device_put(jax.tree_util.tree_map(jnp.asarray, p_src), rep)
          gb_d = jax.device_put(gb_np, sh_w)
          cond_d = jax.device_put(cond_np, rep)

          if prec == "int8":
            def _local_fn(p_in, gb_in, c_in):
              return relight_forward_int8(p_in, gb_in, c_in)
          else:
            def _local_fn(p_in, gb_in, c_in, _pm=prec):
              return relight_forward_float_or_qat(p_in, gb_in, c_in, precision_mode=_pm, qat=False)

          sh_fn = shard_map(
              _local_fn,
              mesh=mesh,
              in_specs=(P(), P(None, None, "dp", None), P()),
              out_specs=P(None, None, "dp", None),
              **sm_kw,
          )

          @jax.jit
          def run_4c(p_in, gb_in, c_in):
            return sh_fn(p_in, gb_in, c_in)

          hlo_info = audit_hlo_strict(run_4c.lower(p_d, gb_d, cond_d))
          for _ in range(3):
            y = run_4c(p_d, gb_d, cond_d)
          jax.block_until_ready(y)
          t0 = time.perf_counter()
          for _ in range(n_steps):
            y = run_4c(p_d, gb_d, cond_d)
          jax.block_until_ready(y)
          ms = (time.perf_counter() - t0) * 1e3 / n_steps

        fps = 1e3 / ms
        mpix_s = (H * W / 1e6) * fps
        rec = {
            "precision": prec,
            "resolution": f"{W}x{H}",
            "n_chips": n_chips,
            "ms": round(ms, 3),
            "fps": round(fps, 2),
            "mpixels_per_s": round(mpix_s, 2),
            "hlo_zero_copy": True,
            "hlo_ops": hlo_info,
        }
        print(f"[PARETO BENCH] {prec.upper():4s} {W}x{H} {n_chips}c: {ms:7.3f} ms ({fps:7.2f} FPS, {mpix_s:7.2f} MP/s) hlo={hlo_info}", flush=True)
        records.append(rec)
  return records
