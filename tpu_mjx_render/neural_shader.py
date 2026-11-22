"""Stage 2 TPU-Co-Designed MXU Neural Shaders (BF16, zero HashGrid, single-JIT compatible)."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp

from tpu_mjx_render.pallas_render import pallas_mxu_pixel_mlp
from tpu_mjx_render.renderer import GBuffer


def _local_edges(geom_id: jax.Array, z_depth: jax.Array) -> Tuple[jax.Array, jax.Array]:
  """Compute 4-neighbor geom_id silhouette discontinuity and depth gradient edges."""
  squeeze = geom_id.ndim == 2
  g = geom_id[None, ...] if squeeze else geom_id
  z = z_depth[None, ...] if squeeze else z_depth

  gp = jnp.pad(g, ((0, 0), (1, 1), (1, 1)), mode="edge")
  zp = jnp.pad(jnp.where(z > 0.0, z, 10.0), ((0, 0), (1, 1), (1, 1)), mode="edge")
  H, W = g.shape[1], g.shape[2]
  g_edge = (
      (g != gp[:, :H, 1 : W + 1])
      | (g != gp[:, 2 : H + 2, 1 : W + 1])
      | (g != gp[:, 1 : H + 1, :W])
      | (g != gp[:, 1 : H + 1, 2 : W + 2])
  ).astype(jnp.float32)
  z_edge = jnp.clip(
      jnp.abs(zp[:, 2 : H + 2, 1 : W + 1] - zp[:, :H, 1 : W + 1])
      + jnp.abs(zp[:, 1 : H + 1, 2 : W + 2] - zp[:, 1 : H + 1, :W]),
      0.0,
      1.0,
  )
  if squeeze:
    return g_edge[0, ..., None], z_edge[0, ..., None]
  return g_edge[..., None], z_edge[..., None]


def build_gbuffer_features(gb: GBuffer) -> jax.Array:
  """Pack exact Stage 1 GBuffer + BRDF terms into normalized (..., H, W, 24) float32 tensor."""
  hit = (gb.geom_id >= 0).astype(jnp.float32)[..., None]
  pos_norm = jnp.clip((gb.world_pos - jnp.array([-0.8, -1.0, -0.1])) / jnp.array([2.2, 2.0, 1.5]), -1.0, 2.0)
  z_norm = jnp.where(gb.z_depth > 0.0, jnp.clip(gb.z_depth / 3.0, 0.0, 1.5), 1.5)[..., None]
  sh = gb.direct_shadow[..., None]
  g_edge, z_edge = _local_edges(gb.geom_id, gb.z_depth)
  ndotv_raw = jnp.sum(gb.view_dir * gb.normal, axis=-1, keepdims=True)
  ndotv = jnp.clip(-ndotv_raw, 0.0, 1.0)
  refl = gb.view_dir - 2.0 * ndotv_raw * gb.normal
  return jnp.concatenate(
      [
          gb.rgb,         # 3
          gb.albedo,      # 3
          gb.normal,      # 3
          gb.view_dir,    # 3
          pos_norm,       # 3
          z_norm,         # 1
          sh,             # 1
          hit,            # 1
          g_edge,         # 1
          z_edge,         # 1
          refl,           # 3
          ndotv,          # 1
      ],
      axis=-1,
  )


class PixelResMLP128(nn.Module):
  """M1: Pixel-wise 4-layer W=128 BF16 MLP (can execute via Pallas VMEM kernel)."""

  width: int = 128

  @nn.compact
  def __call__(self, x: jax.Array, train: bool = False) -> jax.Array:
    del train
    compute_dtype = jnp.bfloat16
    h = x.astype(compute_dtype)
    for _ in range(3):
      h = nn.Dense(self.width, use_bias=True, dtype=compute_dtype, param_dtype=jnp.float32)(h)
      h = nn.relu(h)
    delta = nn.Dense(3, use_bias=True, dtype=compute_dtype, param_dtype=jnp.float32)(h).astype(jnp.float32)
    return delta


class _ResBlock(nn.Module):
  features: int

  @nn.compact
  def __call__(self, x: jax.Array) -> jax.Array:
    dt = jnp.bfloat16
    y = nn.Conv(self.features, (3, 3), padding="SAME", dtype=dt, param_dtype=jnp.float32)(x)
    y = nn.relu(y)
    y = nn.Conv(self.features, (3, 3), padding="SAME", dtype=dt, param_dtype=jnp.float32)(y)
    return nn.relu(x + y)


class SpatialRelighter(nn.Module):
  """M2: TPU MXU-aligned 2-scale BF16 Convolutional Relighter (base_width=64 -> 128 -> 64).

  Captures soft penumbra shadows, contact AO around gripper/cube, and indirect bounce
  while keeping every matrix/channel dimension aligned for BF16 MXU execution.
  """

  base_width: int = 64

  @nn.compact
  def __call__(self, x: jax.Array, train: bool = False) -> jax.Array:
    del train
    dt = jnp.bfloat16
    w1 = self.base_width
    w2 = self.base_width * 2

    # Scale 1: (B, H, W, C_in) -> (B, H, W, w1)
    h1 = nn.relu(nn.Conv(w1, (3, 3), padding="SAME", dtype=dt, param_dtype=jnp.float32)(x.astype(dt)))
    h1 = _ResBlock(w1)(h1)
    h1 = _ResBlock(w1)(h1)

    # Scale 2: downsample 2x -> (B, H/2, W/2, w2)
    h2 = nn.relu(nn.Conv(w2, (3, 3), strides=(2, 2), padding="SAME", dtype=dt, param_dtype=jnp.float32)(h1))
    h2 = _ResBlock(w2)(h2)
    h2 = _ResBlock(w2)(h2)

    # Upsample 2x & fuse skip + raw pixel G-buffer edge-preserving skip
    B, H, W, _ = x.shape
    h2_up = jax.image.resize(h2.astype(jnp.float32), (B, H, W, w2), method="bilinear").astype(dt)
    cat = jnp.concatenate([h1, h2_up], axis=-1)
    h_out = nn.relu(nn.Conv(w1, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(cat))
    h_out = _ResBlock(w1)(h_out)
    # Direct pixel-wise 2-layer 1x1 BRDF skip prevents cross-object blur & resolves Fresnel specular
    out_in = jnp.concatenate([h_out, x.astype(dt)], axis=-1)
    out_mid = nn.relu(nn.Conv(w1, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(out_in))
    out_mid = nn.relu(nn.Conv(w1, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(out_mid))
    delta = nn.Conv(3, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(out_mid).astype(jnp.float32)
    return delta


def make_neural_shader_model(arch: str) -> nn.Module:
  if arch == "M1_pixel_128":
    return PixelResMLP128(width=128)
  if arch == "M2_spatial_32":
    return SpatialRelighter(base_width=16)  # 16 -> 32 narrow ablation
  if arch == "M2_spatial_128":
    return SpatialRelighter(base_width=64)  # 64 -> 128 MXU-aligned
  raise ValueError(f"Unknown arch: {arch}")


def apply_neural_shader(
    model: nn.Module,
    params: Any,
    gb: GBuffer,
    cal_scale: float = 1.0,
) -> jax.Array:
  """Apply frozen Stage 2 neural relighter inside JIT graph (zero host copy)."""
  feats = build_gbuffer_features(gb)
  squeezed = feats.ndim == 3
  if squeezed:
    feats = feats[None, ...]
    base = gb.rgb[None, ...]
  else:
    base = gb.rgb
  delta = model.apply({"params": params}, feats, train=False)
  # Combine Stage 1 direct base + learned GI/soft-shadow/AO delta, scaled by held-out calibration
  rgb = jnp.clip((base + delta) * cal_scale, 0.0, 1.0)
  return rgb[0] if squeezed else rgb


def apply_pallas_pixel_mlp_128(
    params: Any,
    gb: GBuffer,
    cal_scale: float = 1.0,
    interpret: bool = True,
) -> jax.Array:
  """Execute PixelResMLP-128 directly via VMEM-resident Pallas MXU kernel (no bias fold)."""
  feats = build_gbuffer_features(gb)
  orig_shape = feats.shape[:-1]
  flat = feats.reshape(-1, feats.shape[-1])
  # Pad 18 features -> 128 (with bias=1 in col 18) for 128x128 MXU kernel
  pad_c = 128 - flat.shape[-1]
  x128 = jnp.pad(flat, ((0, 0), (0, pad_c)))
  d0 = params["Dense_0"]["kernel"]
  d1 = params["Dense_1"]["kernel"]
  d2 = params["Dense_2"]["kernel"]
  d3 = params["Dense_3"]["kernel"]
  w0 = jnp.pad(d0, ((0, 128 - d0.shape[0]), (0, 0)))
  w1 = d1
  w2 = d2
  w3 = jnp.pad(d3, ((0, 0), (0, 128 - d3.shape[1])))
  delta = pallas_mxu_pixel_mlp(x128, w0, w1, w2, w3, interpret=interpret)
  delta = delta.reshape(*orig_shape, 3)
  return jnp.clip((gb.rgb + delta) * cal_scale, 0.0, 1.0)
