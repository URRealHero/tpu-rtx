"""Stage 2 Neural Relighter Pareto Candidate Architectures (BF16 MXU-oriented)."""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

from tpu_mjx_render.neural_shader import SpatialRelighter, _ResBlock


class ConvNeXtLiteRelighter(nn.Module):
  """Compact 3-stage MXU-friendly U-Net (32 -> 64 -> 128) with 1x1 pixel-BRDF head.

  Reduces expensive full-resolution (64x64) 3x3 convolutions from 7 (in M2_spatial_128)
  down to 3 while keeping a 16x16x128 bottleneck for soft penumbra & indirect GI, and
  preserving crisp contact edges via a 2-layer 1x1 pixel-BRDF head directly on raw GBuffer.
  """

  c1: int = 32
  c2: int = 64
  c3: int = 128
  head_width: int = 64

  @nn.compact
  def __call__(self, x: jax.Array, train: bool = False) -> jax.Array:
    del train
    dt = jnp.bfloat16
    B, H, W, _ = x.shape
    x_bf = x.astype(dt)

    # Stage 1: 64x64 (only 3 3x3 convs total at full res)
    h1 = nn.relu(nn.Conv(self.c1, (3, 3), padding="SAME", dtype=dt, param_dtype=jnp.float32)(x_bf))
    h1 = _ResBlock(self.c1)(h1)

    # Stage 2: 32x32
    h2 = nn.relu(nn.Conv(self.c2, (3, 3), strides=(2, 2), padding="SAME", dtype=dt, param_dtype=jnp.float32)(h1))
    h2 = _ResBlock(self.c2)(h2)

    # Stage 3: 16x16 bottleneck for wide soft shadows & bounce
    h3 = nn.relu(nn.Conv(self.c3, (3, 3), strides=(2, 2), padding="SAME", dtype=dt, param_dtype=jnp.float32)(h2))
    h3 = _ResBlock(self.c3)(h3)

    # Lightweight bilinear upsample decoder (1x1 fusion only, no 3x3 spatial blur at full res)
    h3_up = jax.image.resize(h3.astype(jnp.float32), (B, H // 2, W // 2, self.c3), method="bilinear").astype(dt)
    d2 = nn.relu(
        nn.Conv(self.c2, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(
            jnp.concatenate([h2, h3_up], axis=-1)
        )
    )
    d2_up = jax.image.resize(d2.astype(jnp.float32), (B, H, W, self.c2), method="bilinear").astype(dt)

    # Direct pixel-wise 1x1 BRDF skip head
    out_in = jnp.concatenate([h1, d2_up, x_bf], axis=-1)
    out_mid = nn.relu(nn.Conv(self.head_width, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(out_in))
    out_mid = nn.relu(nn.Conv(self.head_width, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(out_mid))
    delta = nn.Conv(3, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(out_mid).astype(jnp.float32)
    return delta


class _ConvNeXtDWBlock(nn.Module):
  """Depthwise-separable ConvNeXt-style residual block (for TPU VPU vs MXU ablation)."""

  features: int
  kernel_size: int = 5
  expand_ratio: int = 2

  @nn.compact
  def __call__(self, x: jax.Array) -> jax.Array:
    dt = jnp.bfloat16
    y = nn.Conv(
        self.features,
        (self.kernel_size, self.kernel_size),
        padding="SAME",
        feature_group_count=self.features,
        dtype=dt,
        param_dtype=jnp.float32,
    )(x)
    y = nn.Conv(self.features * self.expand_ratio, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(y)
    y = nn.relu(y)
    y = nn.Conv(self.features, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(y)
    return x + y


class ConvNeXtDWRelighter(nn.Module):
  """3-stage depthwise ConvNeXt U-Net (32 -> 64 -> 128) with 1x1 BRDF head."""

  c1: int = 32
  c2: int = 64
  c3: int = 128
  head_width: int = 64

  @nn.compact
  def __call__(self, x: jax.Array, train: bool = False) -> jax.Array:
    del train
    dt = jnp.bfloat16
    B, H, W, _ = x.shape
    x_bf = x.astype(dt)

    h1 = nn.relu(nn.Conv(self.c1, (3, 3), padding="SAME", dtype=dt, param_dtype=jnp.float32)(x_bf))
    h1 = _ConvNeXtDWBlock(self.c1, kernel_size=5)(h1)

    h2 = nn.relu(nn.Conv(self.c2, (3, 3), strides=(2, 2), padding="SAME", dtype=dt, param_dtype=jnp.float32)(h1))
    h2 = _ConvNeXtDWBlock(self.c2, kernel_size=5)(h2)

    h3 = nn.relu(nn.Conv(self.c3, (3, 3), strides=(2, 2), padding="SAME", dtype=dt, param_dtype=jnp.float32)(h2))
    h3 = _ConvNeXtDWBlock(self.c3, kernel_size=5)(h3)

    h3_up = jax.image.resize(h3.astype(jnp.float32), (B, H // 2, W // 2, self.c3), method="bilinear").astype(dt)
    d2 = nn.relu(
        nn.Conv(self.c2, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(
            jnp.concatenate([h2, h3_up], axis=-1)
        )
    )
    d2_up = jax.image.resize(d2.astype(jnp.float32), (B, H, W, self.c2), method="bilinear").astype(dt)

    out_in = jnp.concatenate([h1, d2_up, x_bf], axis=-1)
    out_mid = nn.relu(nn.Conv(self.head_width, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(out_in))
    out_mid = nn.relu(nn.Conv(self.head_width, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(out_mid))
    delta = nn.Conv(3, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(out_mid).astype(jnp.float32)
    return delta


class Compact2ScaleRelighter(nn.Module):
  """2-scale MXU-aligned U-Net with 1 ResBlock/scale and 1x1-only decoder/BRDF head.

  Eliminates 4 out of 7 full-res 3x3 convolutions compared to M2_spatial_128 while keeping
  channel widths aligned to TPU v5p MXU tiles.
  """

  w1: int = 64
  w2: int = 128
  head_width: int = 64

  @nn.compact
  def __call__(self, x: jax.Array, train: bool = False) -> jax.Array:
    del train
    dt = jnp.bfloat16
    B, H, W, _ = x.shape
    x_bf = x.astype(dt)

    h1 = nn.relu(nn.Conv(self.w1, (3, 3), padding="SAME", dtype=dt, param_dtype=jnp.float32)(x_bf))
    h1 = _ResBlock(self.w1)(h1)

    h2 = nn.relu(nn.Conv(self.w2, (3, 3), strides=(2, 2), padding="SAME", dtype=dt, param_dtype=jnp.float32)(h1))
    h2 = _ResBlock(self.w2)(h2)

    h2_up = jax.image.resize(h2.astype(jnp.float32), (B, H, W, self.w2), method="bilinear").astype(dt)
    out_in = jnp.concatenate([h1, h2_up, x_bf], axis=-1)
    out_mid = nn.relu(nn.Conv(self.head_width, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(out_in))
    out_mid = nn.relu(nn.Conv(self.head_width, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(out_mid))
    delta = nn.Conv(3, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(out_mid).astype(jnp.float32)
    return delta


class EdgeGatedSpatialRelighter(nn.Module):
  """TPU MXU-friendly 2-scale Relighter with learned pixel-edge gate on spatial features.

  Prevents 3x3 cross-object feature bleed across 1-px contact boundaries (box vs fingers)
  by gating spatial features `h_out` with `sigmoid(1x1(x))` before the width-64 1x1 BRDF head.
  """

  base_width: int = 32
  head_width: int = 64

  @nn.compact
  def __call__(self, x: jax.Array, train: bool = False) -> jax.Array:
    del train
    dt = jnp.bfloat16
    w1 = self.base_width
    w2 = self.base_width * 2
    B, H, W, _ = x.shape
    x_bf = x.astype(dt)

    h1 = nn.relu(nn.Conv(w1, (3, 3), padding="SAME", dtype=dt, param_dtype=jnp.float32)(x_bf))
    h1 = _ResBlock(w1)(h1)
    h1 = _ResBlock(w1)(h1)

    h2 = nn.relu(nn.Conv(w2, (3, 3), strides=(2, 2), padding="SAME", dtype=dt, param_dtype=jnp.float32)(h1))
    h2 = _ResBlock(w2)(h2)
    h2 = _ResBlock(w2)(h2)

    h2_up = jax.image.resize(h2.astype(jnp.float32), (B, H, W, w2), method="bilinear").astype(dt)
    cat = jnp.concatenate([h1, h2_up], axis=-1)
    h_out = nn.relu(nn.Conv(w1, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(cat))
    h_out = _ResBlock(w1)(h_out)

    # Learned pixel-edge gate modulates spatial features without cross-pixel blur
    edge_gate = jax.nn.sigmoid(
        nn.Conv(w1, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(x_bf)
    )
    h_gated = h_out * edge_gate

    out_in = jnp.concatenate([h_gated, x_bf], axis=-1)
    out_mid = nn.relu(nn.Conv(self.head_width, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(out_in))
    out_mid = nn.relu(nn.Conv(self.head_width, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(out_mid))
    delta = nn.Conv(3, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(out_mid).astype(jnp.float32)
    return delta


class EdgeGatedConvNeXtLite(nn.Module):
  """3-stage compact U-Net (32 -> 64 -> 128) with pixel-edge gate and width-64 1x1 BRDF head."""

  c1: int = 32
  c2: int = 64
  c3: int = 128
  head_width: int = 64

  @nn.compact
  def __call__(self, x: jax.Array, train: bool = False) -> jax.Array:
    del train
    dt = jnp.bfloat16
    B, H, W, _ = x.shape
    x_bf = x.astype(dt)

    h1 = nn.relu(nn.Conv(self.c1, (3, 3), padding="SAME", dtype=dt, param_dtype=jnp.float32)(x_bf))
    h1 = _ResBlock(self.c1)(h1)

    h2 = nn.relu(nn.Conv(self.c2, (3, 3), strides=(2, 2), padding="SAME", dtype=dt, param_dtype=jnp.float32)(h1))
    h2 = _ResBlock(self.c2)(h2)

    h3 = nn.relu(nn.Conv(self.c3, (3, 3), strides=(2, 2), padding="SAME", dtype=dt, param_dtype=jnp.float32)(h2))
    h3 = _ResBlock(self.c3)(h3)

    h3_up = jax.image.resize(h3.astype(jnp.float32), (B, H // 2, W // 2, self.c3), method="bilinear").astype(dt)
    d2 = nn.relu(
        nn.Conv(self.c2, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(
            jnp.concatenate([h2, h3_up], axis=-1)
        )
    )
    d2_up = jax.image.resize(d2.astype(jnp.float32), (B, H, W, self.c2), method="bilinear").astype(dt)

    spatial = jnp.concatenate([h1, d2_up], axis=-1)
    edge_gate = jax.nn.sigmoid(
        nn.Conv(self.c1 + self.c2, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(x_bf)
    )
    out_in = jnp.concatenate([spatial * edge_gate, x_bf], axis=-1)
    out_mid = nn.relu(nn.Conv(self.head_width, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(out_in))
    out_mid = nn.relu(nn.Conv(self.head_width, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(out_mid))
    delta = nn.Conv(3, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(out_mid).astype(jnp.float32)
    return delta


class SpatialRelighterH64(nn.Module):
  """M2_spatial_64 with MXU-aligned width-64 1x1 pixel-BRDF skip head."""

  base_width: int = 32
  head_width: int = 64

  @nn.compact
  def __call__(self, x: jax.Array, train: bool = False) -> jax.Array:
    del train
    dt = jnp.bfloat16
    w1 = self.base_width
    w2 = self.base_width * 2
    B, H, W, _ = x.shape
    x_bf = x.astype(dt)

    h1 = nn.relu(nn.Conv(w1, (3, 3), padding="SAME", dtype=dt, param_dtype=jnp.float32)(x_bf))
    h1 = _ResBlock(w1)(h1)
    h1 = _ResBlock(w1)(h1)

    h2 = nn.relu(nn.Conv(w2, (3, 3), strides=(2, 2), padding="SAME", dtype=dt, param_dtype=jnp.float32)(h1))
    h2 = _ResBlock(w2)(h2)
    h2 = _ResBlock(w2)(h2)

    h2_up = jax.image.resize(h2.astype(jnp.float32), (B, H, W, w2), method="bilinear").astype(dt)
    cat = jnp.concatenate([h1, h2_up], axis=-1)
    h_out = nn.relu(nn.Conv(w1, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(cat))
    h_out = _ResBlock(w1)(h_out)

    out_in = jnp.concatenate([h_out, x_bf], axis=-1)
    out_mid = nn.relu(nn.Conv(self.head_width, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(out_in))
    out_mid = nn.relu(nn.Conv(self.head_width, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(out_mid))
    delta = nn.Conv(3, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(out_mid).astype(jnp.float32)
    return delta


class PixelShuffleMXURelighter(nn.Module):
  """TPU v5p MXU-aligned SpaceToDepth(2) -> 128-ch U-Net -> DepthToSpace(2) + 1x1 BRDF head.

  Eliminates all 3x3 convolutions at full resolution (H x W) and avoids bilinear upsample
  half-pixel phase shift across 1-px contact boundaries, while aligning spatial convolutions
  to 128-channel MXU tiles at (H/2 x W/2) and (H/4 x W/4).
  """

  c_mid: int = 128
  c_bot: int = 128
  head_width: int = 64

  @nn.compact
  def __call__(self, x: jax.Array, train: bool = False) -> jax.Array:
    del train
    dt = jnp.bfloat16
    B, H, W, C = x.shape
    x_bf = x.astype(dt)

    # SpaceToDepth(2): (B, H, W, C) -> (B, H/2, W/2, 4*C)
    s2d = (
        x_bf.reshape(B, H // 2, 2, W // 2, 2, C)
        .transpose(0, 1, 3, 2, 4, 5)
        .reshape(B, H // 2, W // 2, 4 * C)
    )

    h1 = nn.relu(nn.Conv(self.c_mid, (3, 3), padding="SAME", dtype=dt, param_dtype=jnp.float32)(s2d))
    h1 = _ResBlock(self.c_mid)(h1)

    h2 = nn.relu(
        nn.Conv(self.c_bot, (3, 3), strides=(2, 2), padding="SAME", dtype=dt, param_dtype=jnp.float32)(h1)
    )
    h2 = _ResBlock(self.c_bot)(h2)

    h2_up = jax.image.resize(h2.astype(jnp.float32), (B, H // 2, W // 2, self.c_bot), method="bilinear").astype(dt)
    h_fuse = nn.relu(
        nn.Conv(self.c_mid, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(
            jnp.concatenate([h1, h2_up], axis=-1)
        )
    )
    h_fuse = _ResBlock(self.c_mid)(h_fuse)

    # DepthToSpace(2): (B, H/2, W/2, c_mid) -> (B, H, W, c_mid // 4)
    c_out = self.c_mid // 4
    h_pix = (
        h_fuse.reshape(B, H // 2, W // 2, 2, 2, c_out)
        .transpose(0, 1, 3, 2, 4, 5)
        .reshape(B, H, W, c_out)
    )

    out_in = jnp.concatenate([h_pix, x_bf], axis=-1)
    out_mid = nn.relu(nn.Conv(self.head_width, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(out_in))
    out_mid = nn.relu(nn.Conv(self.head_width, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(out_mid))
    delta = nn.Conv(3, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(out_mid).astype(jnp.float32)
    return delta


class SpatialRelighter1RB(nn.Module):
  """2-scale BF16 Relighter (64 -> 128 -> 64) with 1 ResBlock per stage (10 convs vs 16)."""

  base_width: int = 64

  @nn.compact
  def __call__(self, x: jax.Array, train: bool = False) -> jax.Array:
    del train
    dt = jnp.bfloat16
    w1 = self.base_width
    w2 = self.base_width * 2
    B, H, W, _ = x.shape
    x_bf = x.astype(dt)

    h1 = nn.relu(nn.Conv(w1, (3, 3), padding="SAME", dtype=dt, param_dtype=jnp.float32)(x_bf))
    h1 = _ResBlock(w1)(h1)

    h2 = nn.relu(nn.Conv(w2, (3, 3), strides=(2, 2), padding="SAME", dtype=dt, param_dtype=jnp.float32)(h1))
    h2 = _ResBlock(w2)(h2)

    h2_up = jax.image.resize(h2.astype(jnp.float32), (B, H, W, w2), method="bilinear").astype(dt)
    cat = jnp.concatenate([h1, h2_up], axis=-1)
    h_out = nn.relu(nn.Conv(w1, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(cat))
    h_out = _ResBlock(w1)(h_out)

    out_in = jnp.concatenate([h_out, x_bf], axis=-1)
    out_mid = nn.relu(nn.Conv(w1, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(out_in))
    out_mid = nn.relu(nn.Conv(w1, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(out_mid))
    delta = nn.Conv(3, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(out_mid).astype(jnp.float32)
    return delta


class DemodSpatialRelighter(nn.Module):
  """Albedo-demodulated 2-scale BF16 Relighter: delta = albedo * delta_irrad + delta_spec.

  Separates smooth spatial irradiance/GI from sharp per-pixel GBuffer albedo discontinuities,
  eliminating cross-object color bleed across 1-px box-finger contact boundaries.
  """

  base_width: int = 32
  head_width: int = 64

  @nn.compact
  def __call__(self, x: jax.Array, train: bool = False) -> jax.Array:
    del train
    dt = jnp.bfloat16
    w1 = self.base_width
    w2 = self.base_width * 2
    B, H, W, _ = x.shape
    x_bf = x.astype(dt)
    albedo = x_bf[..., 3:6]

    h1 = nn.relu(nn.Conv(w1, (3, 3), padding="SAME", dtype=dt, param_dtype=jnp.float32)(x_bf))
    h1 = _ResBlock(w1)(h1)
    h1 = _ResBlock(w1)(h1)

    h2 = nn.relu(nn.Conv(w2, (3, 3), strides=(2, 2), padding="SAME", dtype=dt, param_dtype=jnp.float32)(h1))
    h2 = _ResBlock(w2)(h2)
    h2 = _ResBlock(w2)(h2)

    h2_up = jax.image.resize(h2.astype(jnp.float32), (B, H, W, w2), method="bilinear").astype(dt)
    cat = jnp.concatenate([h1, h2_up], axis=-1)
    h_out = nn.relu(nn.Conv(w1, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(cat))
    h_out = _ResBlock(w1)(h_out)

    out_in = jnp.concatenate([h_out, x_bf], axis=-1)
    out_mid = nn.relu(nn.Conv(self.head_width, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(out_in))
    out_mid = nn.relu(nn.Conv(self.head_width, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(out_mid))
    irrad_spec = nn.Conv(6, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(out_mid)
    delta_irrad = irrad_spec[..., :3]
    delta_spec = jnp.asarray(0.25, dt) * jnp.tanh(irrad_spec[..., 3:6].astype(jnp.float32)).astype(dt)
    return (albedo * delta_irrad + delta_spec).astype(jnp.float32)


class DemodCompactRelighter(nn.Module):
  """Ultra-fast 2-scale (1 ResBlock/scale, 1x1 decoder) albedo-demodulated BF16 Relighter."""

  w1: int = 32
  w2: int = 64
  head_width: int = 64

  @nn.compact
  def __call__(self, x: jax.Array, train: bool = False) -> jax.Array:
    del train
    dt = jnp.bfloat16
    B, H, W, _ = x.shape
    x_bf = x.astype(dt)
    albedo = x_bf[..., 3:6]

    h1 = nn.relu(nn.Conv(self.w1, (3, 3), padding="SAME", dtype=dt, param_dtype=jnp.float32)(x_bf))
    h1 = _ResBlock(self.w1)(h1)

    h2 = nn.relu(nn.Conv(self.w2, (3, 3), strides=(2, 2), padding="SAME", dtype=dt, param_dtype=jnp.float32)(h1))
    h2 = _ResBlock(self.w2)(h2)

    h2_up = jax.image.resize(h2.astype(jnp.float32), (B, H, W, self.w2), method="bilinear").astype(dt)
    out_in = jnp.concatenate([h1, h2_up, x_bf], axis=-1)
    out_mid = nn.relu(nn.Conv(self.head_width, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(out_in))
    out_mid = nn.relu(nn.Conv(self.head_width, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(out_mid))
    irrad_spec = nn.Conv(6, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(out_mid)
    delta_spec = jnp.asarray(0.25, dt) * jnp.tanh(irrad_spec[..., 3:6].astype(jnp.float32)).astype(dt)
    return (albedo * irrad_spec[..., :3] + delta_spec).astype(jnp.float32)


class DemodConvNeXtLiteRelighter(nn.Module):
  """3-stage compact U-Net (32 -> 64 -> 128) with albedo-demodulated 1x1 BRDF head."""

  c1: int = 32
  c2: int = 64
  c3: int = 128
  head_width: int = 64

  @nn.compact
  def __call__(self, x: jax.Array, train: bool = False) -> jax.Array:
    del train
    dt = jnp.bfloat16
    B, H, W, _ = x.shape
    x_bf = x.astype(dt)
    albedo = x_bf[..., 3:6]

    h1 = nn.relu(nn.Conv(self.c1, (3, 3), padding="SAME", dtype=dt, param_dtype=jnp.float32)(x_bf))
    h1 = _ResBlock(self.c1)(h1)

    h2 = nn.relu(nn.Conv(self.c2, (3, 3), strides=(2, 2), padding="SAME", dtype=dt, param_dtype=jnp.float32)(h1))
    h2 = _ResBlock(self.c2)(h2)

    h3 = nn.relu(nn.Conv(self.c3, (3, 3), strides=(2, 2), padding="SAME", dtype=dt, param_dtype=jnp.float32)(h2))
    h3 = _ResBlock(self.c3)(h3)

    h3_up = jax.image.resize(h3.astype(jnp.float32), (B, H // 2, W // 2, self.c3), method="bilinear").astype(dt)
    d2 = nn.relu(
        nn.Conv(self.c2, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(
            jnp.concatenate([h2, h3_up], axis=-1)
        )
    )
    d2_up = jax.image.resize(d2.astype(jnp.float32), (B, H, W, self.c2), method="bilinear").astype(dt)

    out_in = jnp.concatenate([h1, d2_up, x_bf], axis=-1)
    out_mid = nn.relu(nn.Conv(self.head_width, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(out_in))
    out_mid = nn.relu(nn.Conv(self.head_width, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(out_mid))
    irrad_spec = nn.Conv(6, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(out_mid)
    delta_spec = jnp.asarray(0.25, dt) * jnp.tanh(irrad_spec[..., 3:6].astype(jnp.float32)).astype(dt)
    return (albedo * irrad_spec[..., :3] + delta_spec).astype(jnp.float32)


class DemodPixelShuffleMXURelighter(nn.Module):
  """Albedo-demodulated SpaceToDepth(2) -> MXU U-Net -> DepthToSpace(2) + 1x1 BRDF head.

  Combines 200K+ FPS SpaceToDepth(2) MXU throughput with albedo demodulation
  (`delta = albedo * delta_irrad + delta_spec`) to prevent 1-px contact boundary bleed.
  """

  c_mid: int = 64
  c_bot: int = 128
  c_pix: int = 32
  head_width: int = 64
  fuse_resblock: bool = True

  @nn.compact
  def __call__(self, x: jax.Array, train: bool = False) -> jax.Array:
    del train
    dt = jnp.bfloat16
    B, H, W, C = x.shape
    x_bf = x.astype(dt)
    albedo = x_bf[..., 3:6]

    s2d = (
        x_bf.reshape(B, H // 2, 2, W // 2, 2, C)
        .transpose(0, 1, 3, 2, 4, 5)
        .reshape(B, H // 2, W // 2, 4 * C)
    )

    h1 = nn.relu(nn.Conv(self.c_mid, (3, 3), padding="SAME", dtype=dt, param_dtype=jnp.float32)(s2d))
    h1 = _ResBlock(self.c_mid)(h1)

    h2 = nn.relu(
        nn.Conv(self.c_bot, (3, 3), strides=(2, 2), padding="SAME", dtype=dt, param_dtype=jnp.float32)(h1)
    )
    h2 = _ResBlock(self.c_bot)(h2)

    h2_up = jax.image.resize(h2.astype(jnp.float32), (B, H // 2, W // 2, self.c_bot), method="bilinear").astype(dt)
    h_fuse = nn.relu(
        nn.Conv(self.c_mid, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(
            jnp.concatenate([h1, h2_up], axis=-1)
        )
    )
    if self.fuse_resblock:
      h_fuse = _ResBlock(self.c_mid)(h_fuse)

    # Project to 4 * c_pix before DepthToSpace(2) so c_pix channels are rich without expensive 3x3s
    h_exp = nn.relu(
        nn.Conv(4 * self.c_pix, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(h_fuse)
    )
    h_pix = (
        h_exp.reshape(B, H // 2, W // 2, 2, 2, self.c_pix)
        .transpose(0, 1, 3, 2, 4, 5)
        .reshape(B, H, W, self.c_pix)
    )

    out_in = jnp.concatenate([h_pix, x_bf], axis=-1)
    out_mid = nn.relu(nn.Conv(self.head_width, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(out_in))
    out_mid = nn.relu(nn.Conv(self.head_width, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(out_mid))
    irrad_spec = nn.Conv(6, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(out_mid)
    delta_spec = jnp.asarray(0.25, dt) * jnp.tanh(irrad_spec[..., 3:6].astype(jnp.float32)).astype(dt)
    return (albedo * irrad_spec[..., :3] + delta_spec).astype(jnp.float32)


class EdgeGatedS2DMXURelighter(nn.Module):
  """200K-FPS SpaceToDepth(2) MXU U-Net with per-pixel 1x1 edge gate & albedo-demodulated BRDF head.

  Eliminates 2x2 SpaceToDepth super-pixel cross-object bleed across 1-px contact boundaries
  via a zero-spatial-mixing 1x1 edge gate on raw GBuffer features (including exact geom_id edges).
  """

  c_mid: int = 64
  c_bot: int = 128
  c_pix: int = 32
  head_width: int = 64

  @nn.compact
  def __call__(self, x: jax.Array, train: bool = False) -> jax.Array:
    del train
    dt = jnp.bfloat16
    B, H, W, C = x.shape
    x_bf = x.astype(dt)
    albedo = x_bf[..., 3:6]

    s2d = (
        x_bf.reshape(B, H // 2, 2, W // 2, 2, C)
        .transpose(0, 1, 3, 2, 4, 5)
        .reshape(B, H // 2, W // 2, 4 * C)
    )

    h1 = nn.relu(nn.Conv(self.c_mid, (3, 3), padding="SAME", dtype=dt, param_dtype=jnp.float32)(s2d))
    h1 = _ResBlock(self.c_mid)(h1)

    h2 = nn.relu(
        nn.Conv(self.c_bot, (3, 3), strides=(2, 2), padding="SAME", dtype=dt, param_dtype=jnp.float32)(h1)
    )
    h2 = _ResBlock(self.c_bot)(h2)

    h2_up = jax.image.resize(h2.astype(jnp.float32), (B, H // 2, W // 2, self.c_bot), method="bilinear").astype(dt)
    h_fuse = nn.relu(
        nn.Conv(self.c_mid, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(
            jnp.concatenate([h1, h2_up], axis=-1)
        )
    )
    h_fuse = _ResBlock(self.c_mid)(h_fuse)

    h_exp = nn.relu(
        nn.Conv(4 * self.c_pix, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(h_fuse)
    )
    h_pix = (
        h_exp.reshape(B, H // 2, W // 2, 2, 2, self.c_pix)
        .transpose(0, 1, 3, 2, 4, 5)
        .reshape(B, H, W, self.c_pix)
    )

    # Pixel-wise 1x1 edge gate suppresses 2x2 super-pixel bleed at exact Stage 1 geom_id boundaries
    edge_gate = jax.nn.sigmoid(
        nn.Conv(self.c_pix, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(x_bf).astype(jnp.float32)
    ).astype(dt)
    out_in = jnp.concatenate([h_pix * edge_gate, x_bf], axis=-1)
    out_mid = nn.relu(nn.Conv(self.head_width, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(out_in))
    out_mid = nn.relu(nn.Conv(self.head_width, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(out_mid))
    irrad_spec = nn.Conv(6, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(out_mid)
    delta_irrad = irrad_spec[..., :3]
    delta_spec = jnp.asarray(0.35, dt) * jnp.tanh(irrad_spec[..., 3:6].astype(jnp.float32)).astype(dt)
    return (albedo * delta_irrad + delta_spec).astype(jnp.float32)


class HybridDemodConvNeXtLite(nn.Module):
  """3-stage compact U-Net (32 -> 64 -> 128) with hybrid albedo-demodulated + direct BRDF head."""

  c1: int = 32
  c2: int = 64
  c3: int = 128
  head_width: int = 64

  @nn.compact
  def __call__(self, x: jax.Array, train: bool = False) -> jax.Array:
    del train
    dt = jnp.bfloat16
    B, H, W, _ = x.shape
    x_bf = x.astype(dt)
    albedo = x_bf[..., 3:6]

    h1 = nn.relu(nn.Conv(self.c1, (3, 3), padding="SAME", dtype=dt, param_dtype=jnp.float32)(x_bf))
    h1 = _ResBlock(self.c1)(h1)

    h2 = nn.relu(nn.Conv(self.c2, (3, 3), strides=(2, 2), padding="SAME", dtype=dt, param_dtype=jnp.float32)(h1))
    h2 = _ResBlock(self.c2)(h2)

    h3 = nn.relu(nn.Conv(self.c3, (3, 3), strides=(2, 2), padding="SAME", dtype=dt, param_dtype=jnp.float32)(h2))
    h3 = _ResBlock(self.c3)(h3)

    h3_up = jax.image.resize(h3.astype(jnp.float32), (B, H // 2, W // 2, self.c3), method="bilinear").astype(dt)
    d2 = nn.relu(
        nn.Conv(self.c2, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(
            jnp.concatenate([h2, h3_up], axis=-1)
        )
    )
    d2_up = jax.image.resize(d2.astype(jnp.float32), (B, H, W, self.c2), method="bilinear").astype(dt)

    out_in = jnp.concatenate([h1, d2_up, x_bf], axis=-1)
    out_mid = nn.relu(nn.Conv(self.head_width, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(out_in))
    out_mid = nn.relu(nn.Conv(self.head_width, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(out_mid))
    out6 = nn.Conv(6, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(out_mid)
    return (albedo * out6[..., :3] + out6[..., 3:6]).astype(jnp.float32)


class HybridDemodS2DMXU(nn.Module):
  """200K-FPS SpaceToDepth(2) MXU U-Net with hybrid albedo-demodulated + direct 1x1 BRDF head."""

  c_mid: int = 64
  c_bot: int = 128
  c_pix: int = 32
  head_width: int = 64

  @nn.compact
  def __call__(self, x: jax.Array, train: bool = False) -> jax.Array:
    del train
    dt = jnp.bfloat16
    B, H, W, C = x.shape
    x_bf = x.astype(dt)
    albedo = x_bf[..., 3:6]

    s2d = (
        x_bf.reshape(B, H // 2, 2, W // 2, 2, C)
        .transpose(0, 1, 3, 2, 4, 5)
        .reshape(B, H // 2, W // 2, 4 * C)
    )
    h1 = nn.relu(nn.Conv(self.c_mid, (3, 3), padding="SAME", dtype=dt, param_dtype=jnp.float32)(s2d))
    h1 = _ResBlock(self.c_mid)(h1)
    h2 = nn.relu(
        nn.Conv(self.c_bot, (3, 3), strides=(2, 2), padding="SAME", dtype=dt, param_dtype=jnp.float32)(h1)
    )
    h2 = _ResBlock(self.c_bot)(h2)
    h2_up = jax.image.resize(h2.astype(jnp.float32), (B, H // 2, W // 2, self.c_bot), method="bilinear").astype(dt)
    h_fuse = nn.relu(
        nn.Conv(self.c_mid, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(
            jnp.concatenate([h1, h2_up], axis=-1)
        )
    )
    h_fuse = _ResBlock(self.c_mid)(h_fuse)
    h_exp = nn.relu(nn.Conv(4 * self.c_pix, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(h_fuse))
    h_pix = (
        h_exp.reshape(B, H // 2, W // 2, 2, 2, self.c_pix)
        .transpose(0, 1, 3, 2, 4, 5)
        .reshape(B, H, W, self.c_pix)
    )
    out_in = jnp.concatenate([h_pix, x_bf], axis=-1)
    out_mid = nn.relu(nn.Conv(self.head_width, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(out_in))
    out_mid = nn.relu(nn.Conv(self.head_width, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(out_mid))
    out6 = nn.Conv(6, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(out_mid)
    return (albedo * out6[..., :3] + out6[..., 3:6]).astype(jnp.float32)


class DemodEdgeProtS2DMXU(nn.Module):
  """Ultra-fast SpaceToDepth(2) BF16 MXU U-Net with hard geom-edge bleed protection + albedo demod.

  At 1-px Stage-1 `geom_id` silhouette discontinuities (`g_edge = x[..., 18:19]`), attenuates
  spatial 2x2/3x3 features by `(1 - 0.85 * g_edge)` and relies on a pure 1x1 pixel-BRDF
  residual branch + `albedo * delta_irrad + delta_spec`, eliminating cross-object bleed while
  sustaining ~170K-185K FPS on 1 TPU v5p chip.
  """

  c_mid: int = 64
  c_bot: int = 128
  c_pix: int = 32
  head_width: int = 64

  @nn.compact
  def __call__(self, x: jax.Array, train: bool = False) -> jax.Array:
    del train
    dt = jnp.bfloat16
    B, H, W, C = x.shape
    x_bf = x.astype(dt)
    albedo = x_bf[..., 3:6]
    g_edge = x_bf[..., 18:19]

    s2d = (
        x_bf.reshape(B, H // 2, 2, W // 2, 2, C)
        .transpose(0, 1, 3, 2, 4, 5)
        .reshape(B, H // 2, W // 2, 4 * C)
    )
    h1 = nn.relu(nn.Conv(self.c_mid, (3, 3), padding="SAME", dtype=dt, param_dtype=jnp.float32)(s2d))
    h1 = _ResBlock(self.c_mid)(h1)
    h2 = nn.relu(
        nn.Conv(self.c_bot, (3, 3), strides=(2, 2), padding="SAME", dtype=dt, param_dtype=jnp.float32)(h1)
    )
    h2 = _ResBlock(self.c_bot)(h2)
    h2_up = jax.image.resize(h2.astype(jnp.float32), (B, H // 2, W // 2, self.c_bot), method="bilinear").astype(dt)
    h_fuse = nn.relu(
        nn.Conv(self.c_mid, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(
            jnp.concatenate([h1, h2_up], axis=-1)
        )
    )
    h_fuse = _ResBlock(self.c_mid)(h_fuse)
    h_exp = nn.relu(nn.Conv(4 * self.c_pix, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(h_fuse))
    h_pix = (
        h_exp.reshape(B, H // 2, W // 2, 2, 2, self.c_pix)
        .transpose(0, 1, 3, 2, 4, 5)
        .reshape(B, H, W, self.c_pix)
    )

    # Hard geom_id edge gate blocks cross-object bleed at 1-px boundaries
    learn_g = jax.nn.sigmoid(nn.Conv(1, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(x_bf).astype(jnp.float32)).astype(dt)
    keep_sp = (jnp.asarray(1.0, dt) - jnp.asarray(0.95, dt) * g_edge) * learn_g
    out_in = jnp.concatenate([h_pix * keep_sp, x_bf], axis=-1)
    out_mid = nn.relu(nn.Conv(self.head_width, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(out_in))
    out_mid = nn.relu(nn.Conv(self.head_width, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(out_mid))
    out6 = nn.Conv(6, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(out_mid)
    return (albedo * out6[..., :3] + out6[..., 3:6]).astype(jnp.float32)


class DemodEdgeProtConvNeXtLite(nn.Module):
  """3-stage compact U-Net (32 -> 64 -> 128) with hard geom-edge bleed protection + albedo demod."""

  c1: int = 32
  c2: int = 64
  c3: int = 128
  head_width: int = 64

  @nn.compact
  def __call__(self, x: jax.Array, train: bool = False) -> jax.Array:
    del train
    dt = jnp.bfloat16
    B, H, W, _ = x.shape
    x_bf = x.astype(dt)
    albedo = x_bf[..., 3:6]
    g_edge = x_bf[..., 18:19]

    h1 = nn.relu(nn.Conv(self.c1, (3, 3), padding="SAME", dtype=dt, param_dtype=jnp.float32)(x_bf))
    h1 = _ResBlock(self.c1)(h1)
    h2 = nn.relu(nn.Conv(self.c2, (3, 3), strides=(2, 2), padding="SAME", dtype=dt, param_dtype=jnp.float32)(h1))
    h2 = _ResBlock(self.c2)(h2)
    h3 = nn.relu(nn.Conv(self.c3, (3, 3), strides=(2, 2), padding="SAME", dtype=dt, param_dtype=jnp.float32)(h2))
    h3 = _ResBlock(self.c3)(h3)

    h3_up = jax.image.resize(h3.astype(jnp.float32), (B, H // 2, W // 2, self.c3), method="bilinear").astype(dt)
    d2 = nn.relu(
        nn.Conv(self.c2, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(
            jnp.concatenate([h2, h3_up], axis=-1)
        )
    )
    d2_up = jax.image.resize(d2.astype(jnp.float32), (B, H, W, self.c2), method="bilinear").astype(dt)

    spatial = jnp.concatenate([h1, d2_up], axis=-1)
    learn_g = jax.nn.sigmoid(nn.Conv(1, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(x_bf).astype(jnp.float32)).astype(dt)
    keep_sp = (jnp.asarray(1.0, dt) - jnp.asarray(0.95, dt) * g_edge) * learn_g
    out_in = jnp.concatenate([spatial * keep_sp, x_bf], axis=-1)
    out_mid = nn.relu(nn.Conv(self.head_width, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(out_in))
    out_mid = nn.relu(nn.Conv(self.head_width, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(out_mid))
    out6 = nn.Conv(6, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(out_mid)
    return (albedo * out6[..., :3] + out6[..., 3:6]).astype(jnp.float32)


class SpatialFast128Relighter(nn.Module):
  """120K-FPS MXU-aligned 2-scale BF16 Relighter (64 -> 128 -> 64) with 2 full-res 3x3s.

  Retains full 64/128 MXU channel widths for subpixel contact sharpness & GI capacity while
  cutting full-resolution (64x64) 3x3 convolutions from 7 (in M2_spatial_128) down to 2.
  """

  w1: int = 64
  w2: int = 128

  @nn.compact
  def __call__(self, x: jax.Array, train: bool = False) -> jax.Array:
    del train
    dt = jnp.bfloat16
    B, H, W, _ = x.shape
    x_bf = x.astype(dt)

    h1 = nn.relu(nn.Conv(self.w1, (3, 3), padding="SAME", dtype=dt, param_dtype=jnp.float32)(x_bf))
    h2 = nn.relu(nn.Conv(self.w2, (3, 3), strides=(2, 2), padding="SAME", dtype=dt, param_dtype=jnp.float32)(h1))
    h2 = _ResBlock(self.w2)(h2)
    h2 = _ResBlock(self.w2)(h2)

    h2_up = jax.image.resize(h2.astype(jnp.float32), (B, H, W, self.w2), method="bilinear").astype(dt)
    cat = jnp.concatenate([h1, h2_up], axis=-1)
    h_out = nn.relu(nn.Conv(self.w1, (3, 3), padding="SAME", dtype=dt, param_dtype=jnp.float32)(cat))

    out_in = jnp.concatenate([h_out, x_bf], axis=-1)
    out_mid = nn.relu(nn.Conv(self.w1, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(out_in))
    out_mid = nn.relu(nn.Conv(self.w1, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(out_mid))
    delta = nn.Conv(3, (1, 1), padding="SAME", dtype=dt, param_dtype=jnp.float32)(out_mid).astype(jnp.float32)
    return delta


def make_candidate_model(arch: str) -> nn.Module:
  if arch == "M2_spatial_128":
    return SpatialRelighter(base_width=64)
  if arch == "M2_spatial_128_1rb":
    return SpatialRelighter1RB(base_width=64)
  if arch == "M2_mxu_64_fast":
    return SpatialFast128Relighter(w1=64, w2=128)
  if arch == "M2_spatial_64":
    return SpatialRelighter(base_width=32)
  if arch == "M2_spatial_64_h64":
    return SpatialRelighterH64(base_width=32, head_width=64)
  if arch == "M2_spatial_80":
    return SpatialRelighter(base_width=40)
  if arch == "M2_spatial_96":
    return SpatialRelighter(base_width=48)
  if arch == "M2_hybrid_convnext_lite":
    return HybridDemodConvNeXtLite(c1=32, c2=64, c3=128, head_width=64)
  if arch == "M2_hybrid_s2d_64":
    return HybridDemodS2DMXU(c_mid=64, c_bot=128, c_pix=32, head_width=64)
  if arch == "M2_demod_edgeprot_s2d":
    return DemodEdgeProtS2DMXU(c_mid=64, c_bot=128, c_pix=32, head_width=64)
  if arch == "M2_demod_edgeprot_s2d_96":
    return DemodEdgeProtS2DMXU(c_mid=96, c_bot=128, c_pix=32, head_width=64)
  if arch == "M2_demod_edgeprot_lite":
    return DemodEdgeProtConvNeXtLite(c1=32, c2=64, c3=128, head_width=64)
  if arch == "M2_demod_edgeprot_lite_w48":
    return DemodEdgeProtConvNeXtLite(c1=48, c2=64, c3=128, head_width=64)
  if arch == "M2_convnext_lite":
    return ConvNeXtLiteRelighter(c1=32, c2=64, c3=128, head_width=64)
  if arch == "M2_convnext_dw":
    return ConvNeXtDWRelighter(c1=32, c2=64, c3=128, head_width=64)
  if arch == "M2_unet_compact_64":
    return Compact2ScaleRelighter(w1=64, w2=128, head_width=64)
  if arch == "M2_v3_fast_64":
    return Compact2ScaleRelighter(w1=32, w2=64, head_width=64)
  if arch == "M2_v3_fast_96":
    return Compact2ScaleRelighter(w1=48, w2=96, head_width=64)
  if arch == "M2_s2d_mxu_128":
    return PixelShuffleMXURelighter(c_mid=128, c_bot=128, head_width=64)
  if arch == "M2_s2d_mxu_64":
    return PixelShuffleMXURelighter(c_mid=64, c_bot=128, head_width=64)
  if arch == "M2_demod_64":
    return DemodSpatialRelighter(base_width=32, head_width=64)
  if arch == "M2_demod_96":
    return DemodSpatialRelighter(base_width=48, head_width=64)
  if arch == "M2_demod_fast_64":
    return DemodCompactRelighter(w1=32, w2=64, head_width=64)
  if arch == "M2_demod_convnext_lite":
    return DemodConvNeXtLiteRelighter(c1=32, c2=64, c3=128, head_width=64)
  if arch == "M2_demod_s2d_mxu_64":
    return DemodPixelShuffleMXURelighter(c_mid=64, c_bot=128, c_pix=32, head_width=64, fuse_resblock=True)
  if arch == "M2_demod_s2d_fast_64":
    return DemodPixelShuffleMXURelighter(c_mid=64, c_bot=128, c_pix=32, head_width=64, fuse_resblock=False)
  if arch == "M2_demod_s2d_mxu_96":
    return DemodPixelShuffleMXURelighter(c_mid=96, c_bot=128, c_pix=32, head_width=64, fuse_resblock=True)
  if arch == "M2_demod_s2d_mxu_128":
    return DemodPixelShuffleMXURelighter(c_mid=128, c_bot=128, c_pix=32, head_width=64, fuse_resblock=True)
  if arch == "M2_edgegate_s2d_64":
    return EdgeGatedS2DMXURelighter(c_mid=64, c_bot=128, c_pix=32, head_width=64)
  if arch == "M2_edgegate_s2d_96":
    return EdgeGatedS2DMXURelighter(c_mid=96, c_bot=128, c_pix=32, head_width=64)
  if arch == "M2_edgegate_64":
    return EdgeGatedSpatialRelighter(base_width=32, head_width=64)
  if arch == "M2_convnext_edge":
    return EdgeGatedConvNeXtLite(c1=32, c2=64, c3=128, head_width=64)
  raise ValueError(f"Unknown candidate arch: {arch}")
