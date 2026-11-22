"""TPU-ReSTIR DI: Real-Time Many-Light Direct Illumination via Spatiotemporal Reservoir Resampling.

Implements SIGGRAPH 2020 ReSTIR DI (Bitterli et al. 2020) natively in pure JAX +
TPU Mosaic Pallas (`tpu_optix.py` Wide-BVH-64) with zero host callbacks
(`host_callbacks == 0`, `send_recv_ops == 0`):
  1. Primary G-buffer raycast via `optixTrace`.
  2. Streaming M=32 Resampled Importance Sampling (RIS) over large dynamic light
     banks (e.g. N_L = 1,024 lights) using power-proportional or uniform source PDFs
     and full GGX + Lambert unshadowed target PDF p_hat(x, l).
  3. Initial visibility shadow ray (`any_hit=True`) discarding occluded reservoirs.
  4. Multi-pass Spatial Reservoir Reuse with cross-chip ICI `ppermute` halo
     exchange, geometric depth/normal similarity rejection, target-domain p_hat
     re-evaluation, and unbiased MIS normalization.
  5. Final visibility check + Joint A-Trous Wavelet Denoiser + AgX Filmic Tonemap.
"""

from __future__ import annotations

import dataclasses
import inspect
import math
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import jax
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np

from tpu_mjx_render.adapters.tpu_optix import (
    OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
    OptixDeviceContext,
    OptixGAS,
    OptixIAS,
    ShaderBindingTable,
    _safe_norm,
    _safe_unit,
    agx_tonemap_srgb,
    audit_hlo_zero_copy,
    eval_ggx_brdf,
    optixTrace,
)


# ==============================================================================
# Data Structures (PyTree-Registered for shard_map / JIT)
# ==============================================================================

@dataclasses.dataclass(frozen=True)
class ReSTIRConfig:
  m_candidates: int = 32
  m_chunk: int = 8
  spatial_passes: int = 2
  spatial_neighbors: int = 5
  spatial_radii: Tuple[int, ...] = (24, 12)
  depth_rel_thresh: float = 0.10
  normal_dot_thresh: float = 0.90
  m_cap: int = 640
  use_initial_visibility: bool = True
  use_final_visibility: bool = True
  use_unbiased_mis: bool = True
  use_power_sampling: bool = True
  firefly_clamp: float = 40.0
  denoise_sigma_l: float = 0.34
  denoise_passes: int = 4


@dataclasses.dataclass
class ReSTIRLightBank:
  pos: jax.Array           # (N_L, 3) float32
  normal: jax.Array        # (N_L, 3) float32
  emit: jax.Array          # (N_L, 3) float32
  radius: jax.Array        # (N_L,) float32
  power: jax.Array         # (N_L,) float32
  is_isotropic: jax.Array  # (N_L,) float32 (1.0=point/sphere bulb, 0.0=oriented area)
  cdf: jax.Array           # (N_L,) float32 normalized cumulative power
  alias_prob: jax.Array    # (N_L,) float32 Vose alias probability table
  alias_idx: jax.Array     # (N_L,) int32 Vose alias index table

  @property
  def n_lights(self) -> int:
    return int(self.pos.shape[0])


@dataclasses.dataclass
class ReSTIRGBuffer:
  hit: jax.Array        # (H, W) bool
  pos: jax.Array        # (H, W, 3) float32
  normal: jax.Array     # (H, W, 3) float32 (face-forwarded unit)
  wo: jax.Array         # (H, W, 3) float32 (view direction)
  albedo: jax.Array     # (H, W, 3) float32
  roughness: jax.Array  # (H, W) float32
  metallic: jax.Array   # (H, W) float32
  emissive: jax.Array   # (H, W, 3) float32
  depth: jax.Array      # (H, W) float32


@dataclasses.dataclass
class ReSTIRReservoir:
  y_idx: jax.Array  # (H, W) int32 selected light index
  w_sum: jax.Array  # (H, W) float32 sum of RIS weights
  W: jax.Array      # (H, W) float32 unbiased contribution weight
  M: jax.Array      # (H, W) float32 effective sample count
  p_hat: jax.Array  # (H, W) float32 target PDF at current pixel


def _register_restir_pytrees() -> None:
  try:
    jax.tree_util.register_pytree_node(
        OptixGAS,
        lambda g: (
            (g.l2_roots_dev, g.l2_dev, g.l0_dev, g.leaf_dev),
            (
                g.name, g.n_tris, g.n_leaves, g.n_l1, g.n_l2,
                tuple(np.asarray(g.obj_aabb_min, np.float32).tolist()),
                tuple(np.asarray(g.obj_aabb_max, np.float32).tolist()),
                g.hbm_mb,
            ),
        ),
        lambda aux, ch: OptixGAS(
            name=aux[0], n_tris=aux[1], n_leaves=aux[2], n_l1=aux[3], n_l2=aux[4],
            obj_aabb_min=np.array(aux[5], np.float32), obj_aabb_max=np.array(aux[6], np.float32),
            l2_roots_dev=ch[0], l2_dev=ch[1], l0_dev=ch[2], leaf_dev=ch[3], hbm_mb=aux[7],
        ),
    )
  except ValueError:
    pass
  try:
    jax.tree_util.register_pytree_node(
        OptixIAS,
        lambda ias: (
            (ias.unique_gases, ias.inst_xpos_per_gas, ias.inst_xmat_per_gas, ias.inst_ids_per_gas),
            (ias.name, ias.max_active_per_tile, ias.total_effective_tris),
        ),
        lambda aux, ch: OptixIAS(
            name=aux[0], unique_gases=ch[0], inst_xpos_per_gas=ch[1],
            inst_xmat_per_gas=ch[2], inst_ids_per_gas=ch[3],
            max_active_per_tile=aux[1], total_effective_tris=aux[2],
        ),
    )
  except ValueError:
    pass
  try:
    jax.tree_util.register_pytree_node(
        ReSTIRLightBank,
        lambda lb: ((lb.pos, lb.normal, lb.emit, lb.radius, lb.power, lb.is_isotropic, lb.cdf, lb.alias_prob, lb.alias_idx), None),
        lambda _, ch: ReSTIRLightBank(*ch),
    )
  except ValueError:
    pass
  try:
    jax.tree_util.register_pytree_node(
        ReSTIRGBuffer,
        lambda g: ((g.hit, g.pos, g.normal, g.wo, g.albedo, g.roughness, g.metallic, g.emissive, g.depth), None),
        lambda _, ch: ReSTIRGBuffer(*ch),
    )
  except ValueError:
    pass
  try:
    jax.tree_util.register_pytree_node(
        ReSTIRReservoir,
        lambda r: ((r.y_idx, r.w_sum, r.W, r.M, r.p_hat), None),
        lambda _, ch: ReSTIRReservoir(*ch),
    )
  except ValueError:
    pass


_register_restir_pytrees()


def make_restir_light_bank(
    pos: np.ndarray,
    emit: np.ndarray,
    radius: np.ndarray,
    normal: Optional[np.ndarray] = None,
    is_isotropic: Optional[np.ndarray] = None,
) -> ReSTIRLightBank:
  """Construct a `ReSTIRLightBank` from NumPy arrays."""
  pos_a = np.asarray(pos, dtype=np.float32)
  emit_a = np.asarray(emit, dtype=np.float32)
  rad_a = np.asarray(radius, dtype=np.float32)
  n_l = pos_a.shape[0]
  if normal is None:
    nrm_a = np.broadcast_to(np.array([0.0, 0.0, -1.0], dtype=np.float32), (n_l, 3)).copy()
  else:
    nrm_a = np.asarray(normal, dtype=np.float32)
    nrm_a = nrm_a / np.maximum(np.linalg.norm(nrm_a, axis=-1, keepdims=True), 1e-12)
  if is_isotropic is None:
    iso_a = np.ones((n_l,), dtype=np.float32)
  else:
    iso_a = np.asarray(is_isotropic, dtype=np.float32)

  lum = 0.2126 * emit_a[:, 0] + 0.7152 * emit_a[:, 1] + 0.0722 * emit_a[:, 2]
  area = math.pi * rad_a * rad_a
  power_a = np.maximum(lum * area * np.where(iso_a > 0.5, 4.0, 1.0), 1e-6).astype(np.float32)
  cdf_a = np.cumsum(power_a.astype(np.float64))
  cdf_a = (cdf_a / max(float(cdf_a[-1]), 1e-12)).astype(np.float32)
  cdf_a[-1] = 1.0

  # Precompute Vose's Alias Table for O(1) power-proportional candidate sampling on TPU
  scaled = power_a.astype(np.float64) * (float(n_l) / max(float(np.sum(power_a)), 1e-12))
  prob_a = np.ones((n_l,), dtype=np.float32)
  alias_a = np.arange(n_l, dtype=np.int32)
  small = [i for i in range(n_l) if scaled[i] < 1.0]
  large = [i for i in range(n_l) if scaled[i] >= 1.0]
  while small and large:
    s = small.pop()
    l = large.pop()
    prob_a[s] = float(scaled[s])
    alias_a[s] = l
    scaled[l] = (scaled[l] + scaled[s]) - 1.0
    if scaled[l] < 1.0:
      small.append(l)
    else:
      large.append(l)
  for i in large + small:
    prob_a[i] = 1.0
    alias_a[i] = i

  return ReSTIRLightBank(
      pos=jnp.asarray(pos_a),
      normal=jnp.asarray(nrm_a),
      emit=jnp.asarray(emit_a),
      radius=jnp.asarray(rad_a),
      power=jnp.asarray(power_a),
      is_isotropic=jnp.asarray(iso_a),
      cdf=jnp.asarray(cdf_a),
      alias_prob=jnp.asarray(prob_a),
      alias_idx=jnp.asarray(alias_a),
  )


# ==============================================================================
# Primary Camera Ray Generation & G-Buffer Pass
# ==============================================================================

def generate_primary_rays(
    cam_pos: jax.Array,   # (3,) or (1, 3)
    cam_xmat: jax.Array,  # (3, 3) or (1, 3, 3)
    cam_fovy: float,
    znear: float,
    zfar: float,
    H: int,
    W_local: int,
    W_total: int,
    x_off: Union[int, jax.Array],
    rng_key: Optional[jax.Array] = None,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
  """Generate primary pinhole rays `(ro, rd, t_lim)` shaped `(1, H * W_local, ...)`."""
  if cam_pos.ndim == 2:
    cam_pos = cam_pos[0]
  if cam_xmat.ndim == 3:
    cam_xmat = cam_xmat[0]
  aspect = float(W_total) / float(H)
  half_h = float(znear) * math.tan(0.5 * float(cam_fovy) * (math.pi / 180.0))
  half_w = half_h * aspect
  R = H * W_local

  if rng_key is not None:
    jitter = jax.random.uniform(rng_key, (1, R, 2), minval=-0.5, maxval=0.5, dtype=jnp.float32)
  else:
    jitter = jnp.zeros((1, R, 2), dtype=jnp.float32)

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
  rd = _safe_unit(jnp.einsum("ij,brj->bri", cam_xmat, ld, precision=jax.lax.Precision.HIGHEST))
  ro = cam_pos[None, None, :] + (float(znear) / neg_z)[..., None] * rd
  t_lim = jnp.broadcast_to((float(zfar - znear) / neg_z), (1, R))
  return ro, rd, t_lim


def trace_primary_gbuffer(
    accel: Union[OptixGAS, OptixIAS],
    ro: jax.Array,
    rd: jax.Array,
    t_lim: jax.Array,
    H: int,
    W_local: int,
    zfar: float = 120.0,
) -> ReSTIRGBuffer:
  """Trace primary rays via `optixTrace` and unpack a structured `ReSTIRGBuffer`."""
  pl_hit = optixTrace(accel, ro, rd, 0.0, t_lim, H, W_local)
  hit = pl_hit.hit[0].reshape(H, W_local)
  dist = pl_hit.dist[0].reshape(H, W_local)
  ro2 = ro[0].reshape(H, W_local, 3)
  rd2 = rd[0].reshape(H, W_local, 3)
  n_raw = _safe_unit(pl_hit.normal[0].reshape(H, W_local, 3))
  wo = -rd2
  entering = jnp.sum(n_raw * wo, axis=-1) >= 0.0
  n_ff = jnp.where(entering[..., None], n_raw, -n_raw)
  alb = pl_hit.albedo[0].reshape(H, W_local, 3)
  rgh = jnp.clip(pl_hit.roughness[0].reshape(H, W_local), 0.02, 0.99)
  mc = pl_hit.mat_code[0].reshape(H, W_local)

  is_emit = hit & (mc >= 2.8)
  emit_power = jnp.where(is_emit, (mc - 2.0) * 6.0, 0.0)
  emissive = jnp.where(is_emit[..., None], alb * emit_power[..., None], 0.0)
  metallic = jnp.where(mc < 1.8, jnp.clip(mc, 0.0, 1.0), 0.0)
  shade_hit = hit & (~is_emit)

  hit_pos = ro2 + jnp.where(hit, dist, 0.0)[..., None] * rd2
  depth = jnp.where(hit, dist, jnp.float32(zfar))

  return ReSTIRGBuffer(
      hit=shade_hit,
      pos=hit_pos,
      normal=n_ff,
      wo=wo,
      albedo=alb,
      roughness=rgh,
      metallic=metallic,
      emissive=emissive,
      depth=depth,
  )


# ==============================================================================
# Unshadowed Target PDF p_hat & Contribution Evaluation
# ==============================================================================

def eval_light_unshadowed(
    gb: ReSTIRGBuffer,
    lights: ReSTIRLightBank,
    l_idx: jax.Array,  # (H, W) or (H, W, K)
) -> Tuple[jax.Array, jax.Array]:
  """Evaluate unshadowed GGX + Lambert radiance `F_rgb` and luminance `p_hat`."""
  has_k = l_idx.ndim == 3
  l_pos = lights.pos[l_idx]
  l_nrm = lights.normal[l_idx]
  l_emit = lights.emit[l_idx]
  l_rad = lights.radius[l_idx]
  l_iso = lights.is_isotropic[l_idx]

  if has_k:
    pos = gb.pos[:, :, None, :]
    nrm = gb.normal[:, :, None, :]
    wo = gb.wo[:, :, None, :]
    alb = gb.albedo[:, :, None, :]
    rgh = gb.roughness[:, :, None]
    met = gb.metallic[:, :, None]
    hit = gb.hit[:, :, None]
  else:
    pos = gb.pos
    nrm = gb.normal
    wo = gb.wo
    alb = gb.albedo
    rgh = gb.roughness
    met = gb.metallic
    hit = gb.hit

  to_l = l_pos - pos
  dist2 = jnp.sum(to_l * to_l, axis=-1) + l_rad * l_rad
  inv_d = jax.lax.rsqrt(jnp.maximum(dist2, 1e-6))
  wi = to_l * inv_d[..., None]
  ndotl = jnp.maximum(jnp.sum(nrm * wi, axis=-1), 0.0)
  cos_l = jnp.where(l_iso > 0.5, 1.0, jnp.maximum(jnp.sum(l_nrm * (-wi), axis=-1), 0.0))
  area = math.pi * l_rad * l_rad
  geom = (ndotl * cos_l * area) / jnp.maximum(dist2, 0.025)

  brdf = eval_ggx_brdf(nrm, wo, wi, alb, met, rgh)
  f_rgb = brdf * l_emit * geom[..., None]
  active = hit & (ndotl > 0.0) & (cos_l > 0.0)
  f_rgb = jnp.where(active[..., None], f_rgb, 0.0)
  p_hat = jnp.where(
      active,
      0.2126 * f_rgb[..., 0] + 0.7152 * f_rgb[..., 1] + 0.0722 * f_rgb[..., 2],
      0.0,
  )
  return f_rgb, p_hat


# ==============================================================================
# Visibility Check Helper (Shadow Ray with Bulb-Radius Margin)
# ==============================================================================

def _dummy_miss(pl, params):
  del params
  return jnp.zeros(pl.rd.shape, dtype=jnp.float32)


def _dummy_anyhit(pl, params):
  del params
  return pl.hit


_SHADOW_SBT = ShaderBindingTable(
    raygen=lambda *_: None,
    miss=_dummy_miss,
    closesthit=lambda *_: None,
    anyhit=_dummy_anyhit,
)


def trace_light_visibility(
    accel: Union[OptixGAS, OptixIAS],
    gb: ReSTIRGBuffer,
    lights: ReSTIRLightBank,
    l_idx: jax.Array,   # (H, W_local)
    active_mask: jax.Array,  # (H, W_local) bool
) -> jax.Array:
  """Trace 1 shadow ray per pixel (`any_hit=True`), stopping outside bulb radius."""
  H, W_local = gb.hit.shape
  R = H * W_local
  l_pos = lights.pos[l_idx]
  l_rad = lights.radius[l_idx]
  sh_o = gb.pos + gb.normal * 2.0e-3
  to_l = l_pos - sh_o
  dist_l = _safe_norm(to_l)
  wi_l = to_l / jnp.maximum(dist_l[..., None], 1e-12)
  # Stop slightly before the light bulb surface so emitter meshes don't self-occlude.
  sh_lim = jnp.where(active_mask, jnp.maximum(dist_l - l_rad - 3.5e-3, 0.0), 0.0)

  occ = optixTrace(
      accel,
      sh_o.reshape(1, R, 3),
      wi_l.reshape(1, R, 3),
      0.0,
      sh_lim.reshape(1, R),
      H,
      W_local,
      ray_flags=OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
      sbt=_SHADOW_SBT,
      params=None,
  )
  return active_mask & (~occ[0].reshape(H, W_local))


# ==============================================================================
# Stage 1: Streaming M=32 RIS Candidate Generation
# ==============================================================================

def restir_initial_ris(
    gb: ReSTIRGBuffer,
    lights: ReSTIRLightBank,
    cfg: ReSTIRConfig,
    rng_key: jax.Array,
) -> ReSTIRReservoir:
  """Draw `M` initial light candidates via chunked streaming RIS on TPU."""
  H, W_local = gb.hit.shape
  n_lights = lights.n_lights
  total_power = jnp.maximum(jnp.sum(lights.power), 1e-12)

  m_chunk = min(cfg.m_chunk, cfg.m_candidates)
  n_chunks = max(1, cfg.m_candidates // m_chunk)

  init_carry = (
      jnp.zeros((H, W_local), dtype=jnp.int32),
      jnp.zeros((H, W_local), dtype=jnp.float32),
      jnp.zeros((H, W_local), dtype=jnp.float32),
      jnp.zeros((H, W_local), dtype=jnp.float32),
  )

  def _chunk_step(c_idx, carry):
    y_acc, w_sum_acc, m_acc, p_acc = carry
    c_rng = jax.random.fold_in(rng_key, c_idx)
    k_src, k_alias, k_pick, k_merge = jax.random.split(c_rng, 4)

    if cfg.use_power_sampling:
      k_bin = jax.random.randint(k_src, (H, W_local, m_chunk), 0, n_lights, dtype=jnp.int32)
      u_al = jax.random.uniform(k_alias, (H, W_local, m_chunk), minval=0.0, maxval=1.0, dtype=jnp.float32)
      cand_idx = jnp.where(u_al < lights.alias_prob[k_bin], k_bin, lights.alias_idx[k_bin])
      p_src = lights.power[cand_idx] / total_power
    else:
      cand_idx = jax.random.randint(k_src, (H, W_local, m_chunk), 0, n_lights, dtype=jnp.int32)
      p_src = jnp.full((H, W_local, m_chunk), 1.0 / float(n_lights), dtype=jnp.float32)

    _, p_hat_blk = eval_light_unshadowed(gb, lights, cand_idx)
    w_blk = jnp.where(p_src > 0.0, p_hat_blk / jnp.maximum(p_src, 1e-12), 0.0)
    w_blk_sum = jnp.sum(w_blk, axis=-1)

    cdf_blk = jnp.cumsum(w_blk, axis=-1)
    u_pick = jax.random.uniform(k_pick, (H, W_local, 1), minval=0.0, maxval=1.0, dtype=jnp.float32) * w_blk_sum[..., None]
    m_star = jnp.argmax(cdf_blk >= u_pick, axis=-1)
    y_blk = jnp.take_along_axis(cand_idx, m_star[..., None], axis=-1)[..., 0]
    p_blk = jnp.take_along_axis(p_hat_blk, m_star[..., None], axis=-1)[..., 0]

    new_w_sum = w_sum_acc + w_blk_sum
    u_merge = jax.random.uniform(k_merge, (H, W_local), minval=0.0, maxval=1.0, dtype=jnp.float32)
    accept = (w_blk_sum > 0.0) & (u_merge * jnp.maximum(new_w_sum, 1e-12) < w_blk_sum)
    return (
        jnp.where(accept, y_blk, y_acc),
        new_w_sum,
        m_acc + float(m_chunk),
        jnp.where(accept, p_blk, p_acc),
    )

  y_fin, w_sum_fin, m_fin, p_fin = jax.lax.fori_loop(0, n_chunks, _chunk_step, init_carry)
  W_fin = jnp.where((p_fin > 1e-10) & gb.hit, w_sum_fin / jnp.maximum(m_fin * p_fin, 1e-12), 0.0)
  return ReSTIRReservoir(y_idx=y_fin, w_sum=w_sum_fin, W=W_fin, M=m_fin, p_hat=p_fin)


# ==============================================================================
# Stage 2: Cross-Chip Halo Exchange & Spatial Reservoir Reuse
# ==============================================================================

def _exchange_x_halo(
    arr: jax.Array,  # (H, W_local, ...)
    halo: int,
    n_chips: int,
    axis_name: str = "dp",
) -> jax.Array:
  """Pad `(H, W_local, ...)` by `halo` columns left/right across ICI chip boundaries."""
  if halo <= 0:
    return arr
  if n_chips == 1:
    pad_spec = [(0, 0), (halo, halo)] + [(0, 0)] * (arr.ndim - 2)
    return jnp.pad(arr, pad_spec, mode="edge")
  perm_r = [(i, (i + 1) % n_chips) for i in range(n_chips)]
  perm_l = [(i, (i - 1) % n_chips) for i in range(n_chips)]
  left_h = jax.lax.ppermute(arr[:, -halo:], axis_name=axis_name, perm=perm_r)
  right_h = jax.lax.ppermute(arr[:, :halo], axis_name=axis_name, perm=perm_l)
  return jnp.concatenate([left_h, arr, right_h], axis=1)


def restir_spatial_reuse_pass(
    gb: ReSTIRGBuffer,
    res: ReSTIRReservoir,
    lights: ReSTIRLightBank,
    radius: int,
    n_neighbors: int,
    cfg: ReSTIRConfig,
    rng_key: jax.Array,
    n_chips: int,
    chip_id: Union[int, jax.Array],
    axis_name: str = "dp",
) -> ReSTIRReservoir:
  """One pass of SIGGRAPH 2020 Spatial Reservoir Reuse with depth/normal rejection."""
  H, W_local = gb.hit.shape
  W_total = W_local * n_chips
  halo = min(int(radius), W_local)

  # Exchange G-buffer geometry and reservoir state across chip borders
  p_hit = _exchange_x_halo(gb.hit, halo, n_chips, axis_name)
  p_pos = _exchange_x_halo(gb.pos, halo, n_chips, axis_name)
  p_nrm = _exchange_x_halo(gb.normal, halo, n_chips, axis_name)
  p_wo = _exchange_x_halo(gb.wo, halo, n_chips, axis_name)
  p_alb = _exchange_x_halo(gb.albedo, halo, n_chips, axis_name)
  p_rgh = _exchange_x_halo(gb.roughness, halo, n_chips, axis_name)
  p_met = _exchange_x_halo(gb.metallic, halo, n_chips, axis_name)
  p_dep = _exchange_x_halo(gb.depth, halo, n_chips, axis_name)

  p_y = _exchange_x_halo(res.y_idx, halo, n_chips, axis_name)
  p_W = _exchange_x_halo(res.W, halo, n_chips, axis_name)
  p_M = _exchange_x_halo(res.M, halo, n_chips, axis_name)

  k_ang, k_rad, k_pick = jax.random.split(rng_key, 3)
  ang = jax.random.uniform(k_ang, (H, W_local, n_neighbors), minval=0.0, maxval=2.0 * math.pi, dtype=jnp.float32)
  rad = float(radius) * jnp.sqrt(
      jax.random.uniform(k_rad, (H, W_local, n_neighbors), minval=0.0, maxval=1.0, dtype=jnp.float32)
  )
  dx = jnp.round(rad * jnp.cos(ang)).astype(jnp.int32)
  dy = jnp.round(rad * jnp.sin(ang)).astype(jnp.int32)

  gy = jnp.arange(H, dtype=jnp.int32)[:, None, None]
  gx_loc = jnp.arange(W_local, dtype=jnp.int32)[None, :, None]
  gx_glob = jnp.asarray(chip_id, jnp.int32) * W_local + gx_loc

  ny = gy + dy
  nx_glob = gx_glob + dx
  in_bounds = (ny >= 0) & (ny < H) & (nx_glob >= 0) & (nx_glob < W_total)
  ny_c = jnp.clip(ny, 0, H - 1)
  nx_pad = jnp.clip(halo + gx_loc + dx, 0, W_local + 2 * halo - 1)

  # Gather neighbor geometry + reservoir fields
  n_hit = p_hit[ny_c, nx_pad]
  n_nrm = p_nrm[ny_c, nx_pad]
  n_dep = p_dep[ny_c, nx_pad]
  n_y = p_y[ny_c, nx_pad]
  n_W = p_W[ny_c, nx_pad]
  n_M = p_M[ny_c, nx_pad]

  # Geometric depth + normal similarity rejection
  dep_rel = jnp.abs(gb.depth[:, :, None] - n_dep) / jnp.maximum(gb.depth[:, :, None], 1e-4)
  nrm_dot = jnp.sum(gb.normal[:, :, None, :] * n_nrm, axis=-1)
  geom_ok = (
      in_bounds
      & gb.hit[:, :, None]
      & n_hit
      & (dep_rel < cfg.depth_rel_thresh)
      & (nrm_dot > cfg.normal_dot_thresh)
  )

  # Re-evaluate neighbor samples at center pixel q's target domain p_hat_q(y_n)
  _, p_hat_q_yn = eval_light_unshadowed(gb, lights, n_y)
  w_nbr = jnp.where(geom_ok & (n_W > 0.0), p_hat_q_yn * n_W * n_M, 0.0)
  w_self = jnp.where(gb.hit & (res.W > 0.0), res.p_hat * res.W * res.M, 0.0)

  # Combine [self, n_0, ..., n_{K-1}]
  all_y = jnp.concatenate([res.y_idx[:, :, None], n_y], axis=-1)
  all_p = jnp.concatenate([res.p_hat[:, :, None], p_hat_q_yn], axis=-1)
  all_w = jnp.concatenate([w_self[:, :, None], w_nbr], axis=-1)
  w_sum_new = jnp.sum(all_w, axis=-1)

  cdf_all = jnp.cumsum(all_w, axis=-1)
  u_pick = jax.random.uniform(k_pick, (H, W_local, 1), minval=0.0, maxval=1.0, dtype=jnp.float32) * w_sum_new[..., None]
  win_idx = jnp.argmax(cdf_all >= u_pick, axis=-1)
  y_win = jnp.take_along_axis(all_y, win_idx[..., None], axis=-1)[..., 0]
  p_win = jnp.take_along_axis(all_p, win_idx[..., None], axis=-1)[..., 0]

  # Unbiased MIS Z normalization (Bitterli et al. 2020 Sec 4.3 / 5)
  m_self = jnp.where(gb.hit, res.M, 0.0)
  m_nbr_acc = jnp.where(geom_ok, n_M, 0.0)
  if cfg.use_unbiased_mis:
    nbr_gb = ReSTIRGBuffer(
        hit=geom_ok,
        pos=p_pos[ny_c, nx_pad],
        normal=n_nrm,
        wo=p_wo[ny_c, nx_pad],
        albedo=p_alb[ny_c, nx_pad],
        roughness=p_rgh[ny_c, nx_pad],
        metallic=p_met[ny_c, nx_pad],
        emissive=jnp.zeros((H, W_local, n_neighbors, 3), dtype=jnp.float32),
        depth=n_dep,
    )
    # Note: eval_light_unshadowed expects gb without K dim, so pass reshaped (H, W*K)
    nbr_gb_flat = ReSTIRGBuffer(
        hit=nbr_gb.hit.reshape(H, W_local * n_neighbors),
        pos=nbr_gb.pos.reshape(H, W_local * n_neighbors, 3),
        normal=nbr_gb.normal.reshape(H, W_local * n_neighbors, 3),
        wo=nbr_gb.wo.reshape(H, W_local * n_neighbors, 3),
        albedo=nbr_gb.albedo.reshape(H, W_local * n_neighbors, 3),
        roughness=nbr_gb.roughness.reshape(H, W_local * n_neighbors),
        metallic=nbr_gb.metallic.reshape(H, W_local * n_neighbors),
        emissive=nbr_gb.emissive.reshape(H, W_local * n_neighbors, 3),
        depth=nbr_gb.depth.reshape(H, W_local * n_neighbors),
    )
    y_win_nbr = jnp.broadcast_to(y_win[:, :, None], (H, W_local, n_neighbors)).reshape(H, W_local * n_neighbors)
    _, p_hat_nbr_ywin = eval_light_unshadowed(nbr_gb_flat, lights, y_win_nbr)
    p_hat_nbr_ywin = p_hat_nbr_ywin.reshape(H, W_local, n_neighbors)
    z_self = jnp.where(gb.hit & (p_win > 1e-10), m_self, 0.0)
    z_nbrs = jnp.sum(jnp.where(geom_ok & (p_hat_nbr_ywin > 1e-10), m_nbr_acc, 0.0), axis=-1)
    Z = jnp.maximum(z_self + z_nbrs, 1.0)
  else:
    Z = jnp.maximum(m_self + jnp.sum(m_nbr_acc, axis=-1), 1.0)

  m_new = jnp.minimum(m_self + jnp.sum(m_nbr_acc, axis=-1), float(cfg.m_cap))
  W_new = jnp.where((p_win > 1e-10) & gb.hit, w_sum_new / jnp.maximum(Z * p_win, 1e-12), 0.0)
  return ReSTIRReservoir(y_idx=y_win, w_sum=w_sum_new, W=W_new, M=m_new, p_hat=p_win)


# ==============================================================================
# Pure-JAX Cross-Chip Joint A-Trous Wavelet Denoiser (Zero Host Callbacks)
# ==============================================================================

def joint_atrous_denoise_sharded(
    rad_lin: jax.Array,  # (H, W_local, 3)
    gb: ReSTIRGBuffer,
    n_chips: int,
    sigma_l: float = 0.34,
    n_passes: int = 4,
    axis_name: str = "dp",
) -> jax.Array:
  """Sharded firefly clamp + multi-scale Joint A-Trous Wavelet Denoiser with ICI halo."""
  H, W_local, _ = rad_lin.shape
  spec_mask = jnp.where((gb.roughness < 0.14) | (gb.metallic > 0.70), 1.0, 0.0)

  # 1. 3x3 luminance firefly suppression
  lum = 0.2126 * rad_lin[..., 0] + 0.7152 * rad_lin[..., 1] + 0.0722 * rad_lin[..., 2]
  lum_h = _exchange_x_halo(lum, 1, n_chips, axis_name)
  lum_pad = jnp.pad(lum_h, ((1, 1), (0, 0)), mode="edge")
  nbr_sum = jnp.zeros_like(lum)
  nbr_sq = jnp.zeros_like(lum)
  for dy in (-1, 0, 1):
    for dx in (-1, 0, 1):
      if dy == 0 and dx == 0:
        continue
      sl = lum_pad[1 + dy : 1 + dy + H, 1 + dx : 1 + dx + W_local]
      nbr_sum = nbr_sum + sl
      nbr_sq = nbr_sq + sl * sl
  nbr_mu = nbr_sum / 8.0
  nbr_std = jnp.sqrt(jnp.maximum(nbr_sq / 8.0 - nbr_mu * nbr_mu, 0.0))
  # Preserve crisp glossy lantern reflections on wet floor while clamping rare spikes
  cap = jnp.where(spec_mask > 0.5, nbr_mu + 6.0 * nbr_std + 1.20, nbr_mu + 3.0 * nbr_std + 0.15)
  lum_cap = jnp.minimum(lum, cap)
  scale_ff = jnp.where(lum > 1e-6, lum_cap / jnp.maximum(lum, 1e-6), 1.0)
  cur = rad_lin * scale_ff[..., None]

  # 2. Multi-scale A-Trous bilateral passes with joint normal/depth/albedo/luma edge stopping
  max_s = min(1 << max(n_passes - 1, 0), W_local)
  p_nrm = jnp.pad(_exchange_x_halo(gb.normal, max_s, n_chips, axis_name), ((max_s, max_s), (0, 0), (0, 0)), mode="edge")
  p_alb = jnp.pad(_exchange_x_halo(gb.albedo, max_s, n_chips, axis_name), ((max_s, max_s), (0, 0), (0, 0)), mode="edge")
  p_dep = jnp.pad(_exchange_x_halo(gb.depth, max_s, n_chips, axis_name), ((max_s, max_s), (0, 0)), mode="edge")
  k_1d = [0.25, 0.50, 0.25]

  for p_idx in range(n_passes):
    step = 1 << p_idx
    p_cur = jnp.pad(_exchange_x_halo(cur, max_s, n_chips, axis_name), ((max_s, max_s), (0, 0), (0, 0)), mode="edge")
    log_c = jnp.log1p(jnp.maximum(cur, 0.0))
    acc_rad = jnp.zeros_like(cur)
    acc_w = jnp.zeros((H, W_local), dtype=jnp.float32)

    s_l_cur = sigma_l * (0.85 ** p_idx)
    inv_2n = 1.0 / (2.0 * 0.22 * 0.22)
    inv_2z = 1.0 / (2.0 * 0.16 * 0.16)
    inv_2a = 1.0 / (2.0 * 0.25 * 0.25)
    inv_2l = 1.0 / (2.0 * s_l_cur * s_l_cur)

    for iy, dy in enumerate((-step, 0, step)):
      for ix, dx in enumerate((-step, 0, step)):
        kw = k_1d[iy] * k_1d[ix]
        q_rad = p_cur[max_s + dy : max_s + dy + H, max_s + dx : max_s + dx + W_local]
        q_nrm = p_nrm[max_s + dy : max_s + dy + H, max_s + dx : max_s + dx + W_local]
        q_alb = p_alb[max_s + dy : max_s + dy + H, max_s + dx : max_s + dx + W_local]
        q_dep = p_dep[max_s + dy : max_s + dy + H, max_s + dx : max_s + dx + W_local]

        dn2 = jnp.sum((gb.normal - q_nrm) ** 2, axis=-1)
        dz2 = ((gb.depth - q_dep) / jnp.maximum(gb.depth, 0.25)) ** 2
        da2 = jnp.sum((gb.albedo - q_alb) ** 2, axis=-1)
        dl2 = jnp.sum((log_c - jnp.log1p(jnp.maximum(q_rad, 0.0))) ** 2, axis=-1)

        w = kw * jnp.exp(-dn2 * inv_2n - dz2 * inv_2z - da2 * inv_2a - dl2 * inv_2l)
        acc_rad = acc_rad + w[..., None] * q_rad
        acc_w = acc_w + w

    denoised = acc_rad / jnp.maximum(acc_w[..., None], 1e-6)
    # Keep glossy puddle highlights sharp while smoothing diffuse regions
    blend = jnp.where(spec_mask > 0.5, 0.14, 1.0)[..., None]
    cur = jnp.where(gb.hit[..., None], cur + blend * (denoised - cur), rad_lin)
  return cur


# ==============================================================================
# Complete Multi-Chip Renderers: Naive Uniform, ReSTIR DI, ReSTIR DI + Denoise, GT
# ==============================================================================

def _sky_bg(rd2: jax.Array, sky_zenith: jax.Array, sky_horizon: jax.Array) -> jax.Array:
  z = jnp.clip(rd2[..., 2], 0.0, 1.0)[..., None]
  return (1.0 - z) * sky_horizon[None, None, :] + z * sky_zenith[None, None, :]


def render_restir_di_local(
    accel: Union[OptixGAS, OptixIAS],
    lights: ReSTIRLightBank,
    cam_pos: jax.Array,
    cam_xmat: jax.Array,
    cam_fovy: float,
    znear: float,
    zfar: float,
    H: int,
    W_local: int,
    W_total: int,
    x_off: Union[int, jax.Array],
    chip_id: Union[int, jax.Array],
    n_chips: int,
    rng_key: jax.Array,
    cfg: ReSTIRConfig,
    sky_zenith: jax.Array,
    sky_horizon: jax.Array,
    exposure: float = 1.0,
    denoise: bool = True,
    jitter_primary: bool = False,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
  """Render one chip's tile via TPU-ReSTIR DI; returns `(tonemapped_rgb, linear_hdr, depth)`."""
  k_jit, k_ris, k_sp, k_fin = jax.random.split(rng_key, 4)
  ro, rd, t_lim = generate_primary_rays(
      cam_pos, cam_xmat, cam_fovy, znear, zfar, H, W_local, W_total, x_off,
      rng_key=(k_jit if jitter_primary else None),
  )
  gb = trace_primary_gbuffer(accel, ro, rd, t_lim, H, W_local, zfar=zfar)

  # 1. M=32 Initial RIS candidate resampling
  res = restir_initial_ris(gb, lights, cfg, k_ris)

  # 2. Initial visibility check (SIGGRAPH 2020 Sec 4.2)
  if cfg.use_initial_visibility:
    vis0 = trace_light_visibility(accel, gb, lights, res.y_idx, gb.hit & (res.W > 0.0))
    res = ReSTIRReservoir(
        y_idx=res.y_idx,
        w_sum=jnp.where(vis0, res.w_sum, 0.0),
        W=jnp.where(vis0, res.W, 0.0),
        M=res.M,
        p_hat=res.p_hat,
    )

  # 3. 2-Pass Spatial Reservoir Reuse with cross-chip ICI halo & depth/normal rejection
  for p_i in range(cfg.spatial_passes):
    rad_p = cfg.spatial_radii[min(p_i, len(cfg.spatial_radii) - 1)]
    res = restir_spatial_reuse_pass(
        gb, res, lights, rad_p, cfg.spatial_neighbors, cfg,
        jax.random.fold_in(k_sp, p_i), n_chips=n_chips, chip_id=chip_id,
    )

  # 4. Final visibility ray + direct lighting evaluation
  if cfg.use_final_visibility:
    vis_f = trace_light_visibility(accel, gb, lights, res.y_idx, gb.hit & (res.W > 0.0))
  else:
    vis_f = gb.hit & (res.W > 0.0)

  f_rgb, _ = eval_light_unshadowed(gb, lights, res.y_idx)
  direct_rad = jnp.where(vis_f[..., None], f_rgb * res.W[..., None], 0.0)
  direct_rad = jnp.minimum(direct_rad, cfg.firefly_clamp)

  # Subtle ambient sky fill on diffuse albedo + direct emissive bulbs + background sky
  rd2 = rd[0].reshape(H, W_local, 3)
  bg = _sky_bg(rd2, sky_zenith, sky_horizon)
  is_bg = gb.depth >= (zfar - 1.0)
  amb = jnp.where(gb.hit[..., None], 0.03 * gb.albedo * sky_horizon[None, None, :], 0.0)
  rad_lin = jnp.where(is_bg[..., None], bg, gb.emissive + amb + direct_rad)

  if denoise:
    rad_out = joint_atrous_denoise_sharded(
        rad_lin, gb, n_chips=n_chips, sigma_l=cfg.denoise_sigma_l, n_passes=cfg.denoise_passes
    )
  else:
    rad_out = rad_lin

  rgb = agx_tonemap_srgb(rad_out, exposure=exposure)
  return rgb, rad_out, gb.depth


def render_naive_uniform_local(
    accel: Union[OptixGAS, OptixIAS],
    lights: ReSTIRLightBank,
    cam_pos: jax.Array,
    cam_xmat: jax.Array,
    cam_fovy: float,
    znear: float,
    zfar: float,
    H: int,
    W_local: int,
    W_total: int,
    x_off: Union[int, jax.Array],
    rng_key: jax.Array,
    n_vis_rays: int,
    sky_zenith: jax.Array,
    sky_horizon: jax.Array,
    exposure: float = 1.0,
    firefly_clamp: float = 40.0,
    jitter_primary: bool = False,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
  """Render one chip's tile via Uniform Many-Light Sampling with `n_vis_rays` shadow rays/px."""
  k_jit, k_samp = jax.random.split(rng_key)
  ro, rd, t_lim = generate_primary_rays(
      cam_pos, cam_xmat, cam_fovy, znear, zfar, H, W_local, W_total, x_off,
      rng_key=(k_jit if jitter_primary else None),
  )
  gb = trace_primary_gbuffer(accel, ro, rd, t_lim, H, W_local, zfar=zfar)
  n_lights = lights.n_lights

  def _one_vis_sample(s_i, acc):
    l_idx = jax.random.randint(jax.random.fold_in(k_samp, s_i), (H, W_local), 0, n_lights, dtype=jnp.int32)
    f_rgb, p_hat = eval_light_unshadowed(gb, lights, l_idx)
    vis = trace_light_visibility(accel, gb, lights, l_idx, gb.hit & (p_hat > 0.0))
    contrib = jnp.where(vis[..., None], f_rgb * float(n_lights), 0.0)
    return acc + jnp.minimum(contrib, firefly_clamp)

  direct_sum = jax.lax.fori_loop(0, n_vis_rays, _one_vis_sample, jnp.zeros((H, W_local, 3), dtype=jnp.float32))
  direct_rad = direct_sum / float(max(n_vis_rays, 1))

  rd2 = rd[0].reshape(H, W_local, 3)
  bg = _sky_bg(rd2, sky_zenith, sky_horizon)
  is_bg = gb.depth >= (zfar - 1.0)
  amb = jnp.where(gb.hit[..., None], 0.03 * gb.albedo * sky_horizon[None, None, :], 0.0)
  rad_lin = jnp.where(is_bg[..., None], bg, gb.emissive + amb + direct_rad)
  rgb = agx_tonemap_srgb(rad_lin, exposure=exposure)
  return rgb, rad_lin, gb.depth


def build_restir_renderer(
    ctx: OptixDeviceContext,
    accel: Union[OptixGAS, OptixIAS],
    lights: ReSTIRLightBank,
    width: int,
    height: int,
    cam_fovy: float,
    znear: float,
    zfar: float,
    mode: str = "restir_denoise",  # "naive" | "restir" | "restir_denoise" | "gt"
    cfg: Optional[ReSTIRConfig] = None,
    n_vis_rays_naive: int = 2,
    gt_spp: int = 128,
    exposure: float = 1.0,
    sky_zenith: Tuple[float, float, float] = (0.015, 0.025, 0.055),
    sky_horizon: Tuple[float, float, float] = (0.030, 0.042, 0.075),
):
  """Compile a zero-copy multi-chip sharded ReSTIR / Naive / GT renderer on TPU."""
  if cfg is None:
    cfg = ReSTIRConfig()
  n_chips = ctx.n_chips
  assert width % n_chips == 0 and (width // n_chips) % 16 == 0 and height % 8 == 0
  W_local = width // n_chips
  sz = jnp.asarray(sky_zenith, dtype=jnp.float32)
  sh = jnp.asarray(sky_horizon, dtype=jnp.float32)

  def _local(ac, lb, cxp, cxm, cid_arr, rng_in):
    cid = cid_arr[0]
    x_off = cid * W_local
    chip_rng = jax.random.fold_in(rng_in, cid)

    if mode == "naive":
      rgb, hdr, dep = render_naive_uniform_local(
          ac, lb, cxp, cxm, cam_fovy, znear, zfar, height, W_local, width, x_off,
          chip_rng, n_vis_rays_naive, sz, sh, exposure=exposure, firefly_clamp=cfg.firefly_clamp,
      )
    elif mode in ("restir", "restir_denoise"):
      rgb, hdr, dep = render_restir_di_local(
          ac, lb, cxp, cxm, cam_fovy, znear, zfar, height, W_local, width, x_off,
          cid, n_chips, chip_rng, cfg, sz, sh, exposure=exposure, denoise=(mode == "restir_denoise"),
      )
    elif mode == "gt":
      # High-SPP Reference Ground Truth (anti-aliased + power-importance-sampled NEE + ReSTIR M=32)
      cfg_gt = dataclasses.replace(
          cfg, spatial_passes=0, use_initial_visibility=False, use_final_visibility=True, firefly_clamp=120.0
      )
      def _gt_step(s_i, acc):
        _, h_i, d_i = render_restir_di_local(
            ac, lb, cxp, cxm, cam_fovy, znear, zfar, height, W_local, width, x_off,
            cid, n_chips, jax.random.fold_in(chip_rng, s_i), cfg_gt, sz, sh,
            exposure=exposure, denoise=False, jitter_primary=True,
        )
        return (acc[0] + h_i, acc[1] + d_i)
      hdr_sum, dep_sum = jax.lax.fori_loop(
          0, gt_spp, _gt_step,
          (jnp.zeros((height, W_local, 3), jnp.float32), jnp.zeros((height, W_local), jnp.float32)),
      )
      hdr = hdr_sum / float(gt_spp)
      dep = dep_sum / float(gt_spp)
      rgb = agx_tonemap_srgb(hdr, exposure=exposure)
    else:
      raise ValueError(f"Unknown mode={mode}")

    return rgb[None, ...], hdr[None, ...], dep[None, ...]

  sm_kw = {"check_rep": False} if "check_rep" in inspect.signature(shard_map).parameters else {"check_vma": False}
  sharded = shard_map(
      _local,
      mesh=ctx.mesh,
      in_specs=(P(), P(), P(), P(), P("dp"), P()),
      out_specs=(P("dp"), P("dp"), P("dp")),
      **sm_kw,
  )

  @jax.jit
  def render_fn(lights_d: ReSTIRLightBank, cam_pos: jax.Array, cam_xmat: jax.Array, rng: jax.Array):
    return sharded(accel, lights_d, cam_pos, cam_xmat, ctx.chip_ids, rng)

  return render_fn
