"""Flax/JAX BF16 ViT-Tiny Vision-Language-Action (VLA) Encoder + AdamW Optimizer.

Designed for Zero-Copy HBM Fusion inside the same compiled XLA graph as MJX physics
and Mosaic Pallas TiledCamera rendering (`host_callbacks == 0`, `gather_ops == 0`).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import flax
from flax import linen as nn
import jax
import jax.numpy as jnp
import optax


class TransformerBlockBF16(nn.Module):
  emb_dim: int = 192
  num_heads: int = 3
  mlp_dim: int = 768

  @nn.compact
  def __call__(self, x: jax.Array) -> jax.Array:
    B, N, D = x.shape
    H = self.num_heads
    dh = D // H

    h = nn.LayerNorm(dtype=jnp.bfloat16, param_dtype=jnp.float32)(x)
    qkv = nn.Dense(3 * D, use_bias=True, dtype=jnp.bfloat16, param_dtype=jnp.float32)(h)
    qkv = qkv.reshape(B, N, 3, H, dh)
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # (B, N, H, dh)

    scale = jnp.asarray(1.0 / (dh ** 0.5), dtype=jnp.float32)
    logits = jnp.einsum("bnhd,bmhd->bhnm", q.astype(jnp.float32), k.astype(jnp.float32)) * scale
    attn = jax.nn.softmax(logits, axis=-1).astype(jnp.bfloat16)
    out = jnp.einsum("bhnm,bmhd->bnhd", attn, v).reshape(B, N, D)
    x = x + nn.Dense(D, use_bias=True, dtype=jnp.bfloat16, param_dtype=jnp.float32)(out)

    h2 = nn.LayerNorm(dtype=jnp.bfloat16, param_dtype=jnp.float32)(x)
    h2 = nn.Dense(self.mlp_dim, dtype=jnp.bfloat16, param_dtype=jnp.float32)(h2)
    h2 = nn.gelu(h2, approximate=True)
    h2 = nn.Dense(D, dtype=jnp.bfloat16, param_dtype=jnp.float32)(h2)
    return x + h2


class ViTTinyVLA(nn.Module):
  """BF16 Patch-Conv ViT-Tiny Vision-Language-Action Policy Head."""
  action_dim: int
  emb_dim: int = 192
  num_heads: int = 3
  num_layers: int = 4
  mlp_dim: int = 768
  patch_size: int = 8

  @nn.compact
  def __call__(
      self,
      obs_rgbd: jax.Array,    # (B, N_cam, H, W, 4) bfloat16
      proprio: jax.Array,     # (B, P) float32/bfloat16
      task_onehot: jax.Array, # (B, T) bfloat16
  ) -> jax.Array:
    B, N_cam, H, W, C = obs_rgbd.shape
    flat_img = obs_rgbd.reshape(B * N_cam, H, W, C).astype(jnp.bfloat16)

    patches = nn.Conv(
        features=self.emb_dim,
        kernel_size=(self.patch_size, self.patch_size),
        strides=(self.patch_size, self.patch_size),
        padding="VALID",
        dtype=jnp.bfloat16,
        param_dtype=jnp.float32,
        name="patch_conv",
    )(flat_img)  # (B*N_cam, Hp, Wp, D)

    n_vis_per_cam = patches.shape[1] * patches.shape[2]
    vis_tok = patches.reshape(B, N_cam * n_vis_per_cam, self.emb_dim)

    pos_emb = self.param(
        "vis_pos_emb",
        nn.initializers.normal(stddev=0.02),
        (1, N_cam * n_vis_per_cam, self.emb_dim),
        jnp.float32,
    ).astype(jnp.bfloat16)
    vis_tok = vis_tok + pos_emb

    prop_tok = nn.Dense(self.emb_dim, dtype=jnp.bfloat16, param_dtype=jnp.float32, name="proprio_proj")(
        proprio.astype(jnp.bfloat16)
    )[:, None, :]
    lang_tok = nn.Dense(self.emb_dim, dtype=jnp.bfloat16, param_dtype=jnp.float32, name="lang_proj")(
        task_onehot.astype(jnp.bfloat16)
    )[:, None, :]

    tokens = jnp.concatenate([lang_tok, prop_tok, vis_tok], axis=1)
    for li in range(self.num_layers):
      tokens = TransformerBlockBF16(
          emb_dim=self.emb_dim,
          num_heads=self.num_heads,
          mlp_dim=self.mlp_dim,
          name=f"block_{li}",
      )(tokens)

    tokens = nn.LayerNorm(dtype=jnp.bfloat16, param_dtype=jnp.float32)(tokens)
    pooled = jnp.mean(tokens, axis=1)  # (B, D)
    h = nn.Dense(self.emb_dim, dtype=jnp.bfloat16, param_dtype=jnp.float32)(pooled)
    h = nn.gelu(h, approximate=True)
    pred_action = nn.Dense(self.action_dim, dtype=jnp.float32, param_dtype=jnp.float32, name="action_head")(h)
    return pred_action


@flax.struct.dataclass
class VLATrainState:
  params: Any
  opt_state: Any
  step: jax.Array


def create_vla_train_state(
    rng: jax.Array,
    n_cam: int,
    height: int,
    width: int,
    proprio_dim: int,
    n_tasks: int,
    action_dim: int,
    learning_rate: float = 3e-4,
    emb_dim: int = 192,
    num_layers: int = 4,
) -> Tuple[ViTTinyVLA, optax.GradientTransformation, VLATrainState]:
  model = ViTTinyVLA(
      action_dim=action_dim,
      emb_dim=emb_dim,
      num_heads=3,
      num_layers=num_layers,
      mlp_dim=emb_dim * 4,
      patch_size=8,
  )
  dummy_rgbd = jnp.zeros((1, n_cam, height, width, 4), dtype=jnp.bfloat16)
  dummy_prop = jnp.zeros((1, proprio_dim), dtype=jnp.float32)
  dummy_task = jnp.zeros((1, n_tasks), dtype=jnp.bfloat16)
  params = model.init(rng, dummy_rgbd, dummy_prop, dummy_task)
  tx = optax.chain(
      optax.clip_by_global_norm(1.0),
      optax.adamw(learning_rate=learning_rate, weight_decay=1e-4),
  )
  opt_state = tx.init(params)
  state = VLATrainState(params=params, opt_state=opt_state, step=jnp.zeros((), dtype=jnp.int32))
  return model, tx, state
