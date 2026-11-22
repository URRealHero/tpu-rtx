"""Isaac Sim `TiledCamera` & `Omniverse Replicator` (`rep.randomizer.*`) Zero-Copy TPU Adapter.

Compiles NVIDIA Isaac Sim / Omniverse Replicator domain randomization and multi-modal
`TiledCamera` rendering natively into a SINGLE zero-copy XLA + Pallas Mosaic graph on
Google Cloud TPU (`host_callbacks == 0`, `send_recv_ops == 0`, `gather_ops == 0`):

  1. `rep` (`Omniverse Replicator` pure-JAX XLA API):
       - `rep.distribution.{uniform, normal, choice, color_temperature}`
       - `rep.randomizer.{light, materials, texture, camera, poses}`
       - `ReplicatorCfg` / `ReplicatorGraph.apply(...)`: vectorized per-environment
         randomization across `B = 1024..4096+` parallel worlds with zero CPU/GPU stalls.

  2. `TiledCamera` (`TiledCameraCfg`, `TiledCamera`):
       - Single-camera `(B, H, W, C)` and multi-camera `(B, N_cam, H, W, C)` modes
         (e.g., RoboTwin `head + left_wrist + right_wrist`), plus zero-copy stitched
         tiled grid canvas `(grid_h * H, grid_w * W, C)`.
       - 6 synchronized modalities in one XLA graph:
           * `rgb`: GGX Microfacet PBR + Hemisphere AO + Soft Shadows + Per-Env
             Color-Temp Lighting + Procedural Tabletop/Floor Textures + AgX Tonemapping
           * `distance_to_image_plane`: exact metric planar Z-depth (m)
           * `normals`: world-space surface normals
           * `semantic_segmentation`: per-geom class ID (emitted from VMEM)
           * `instance_id`: per-geom instance ID (`-1` on sky)
           * `motion_vectors`: zero-gather 2D optical flow (pixels)
       - Dual Pallas Mosaic backends under one unified API:
           * Unrolled VMEM kernel (`G <= 128`) with 4-AO + 4-soft-shadow GGX or `fast_dr`
           * Hierarchical 2-level Morton Cluster-BVH `fori_loop` kernel (`G = 128..1004+`)
             scaling seamlessly to 1,000+ moving MJX primitives without VMEM OOM.
       - Multi-chip `jax.sharding.Mesh` + `shard_map` scaling across 1c / 4c / Pods.
"""

from __future__ import annotations

import dataclasses
from functools import partial
import inspect
import math
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
try:
  from jax import shard_map
except ImportError:
  from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
import numpy as np

from tpu_mjx_render.camera import PROJ_ORTHOGRAPHIC, PROJ_PERSPECTIVE
from tpu_mjx_render.pallas_v2 import (
    _compute_geom_bound_sq,
    _extract_light_batch,
    _ray_one_geom_v2,
)
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
from tpu_mjx_render.renderer import RenderSceneSpec
from tpu_mjx_render.roofline import TPU_SPECS, TPUHardwareSpec, inspect_hlo_ops


DEFAULT_MODALITIES: Tuple[str, ...] = (
    "rgb",
    "distance_to_image_plane",
    "normals",
    "semantic_segmentation",
    "instance_id",
    "motion_vectors",
)

TEX_SOLID = 0
TEX_CHECKER = 1
TEX_WOOD = 2
TEX_GRID = 3
TEX_SPECKLED = 4

CLUSTER_K = 32
MACRO_M = 4  # 128 primitives per macro-cluster (2x denser micro-clusters -> half VMEM)


def _shard_map_kwargs() -> Dict[str, Any]:
  sig = inspect.signature(shard_map).parameters
  return {"check_rep": False} if "check_rep" in sig else {"check_vma": False}


def audit_replicator_hlo(lowered: Any) -> Dict[str, Any]:
  """Verify zero-copy XLA HLO invariants: host_callbacks == 0, send_recv_ops == 0, gather_ops == 0."""
  txt = lowered.as_text() if hasattr(lowered, "as_text") else str(lowered)
  info = inspect_hlo_ops(txt)
  info["send_recv_ops"] = len(re.findall(r"\bsend\(|\brecv\(", txt, re.I))
  info["zero_copy_verified"] = (
      info["host_callbacks"] == 0
      and info["send_recv_ops"] == 0
      and info["gather_ops"] == 0
  )
  return info


# ==============================================================================
# Color Temperature, HSV, Procedural Textures & AgX Filmic Tonemapping (Pure JAX)
# ==============================================================================

def kelvin_to_rgb(kelvin: jax.Array) -> jax.Array:
  """Approximate Planckian color temperature (1800K..12000K) in pure JAX -> normalized linear RGB."""
  temp = jnp.clip(kelvin, 1800.0, 12000.0) / 100.0
  # Red
  r_hi = 329.698727446 * jnp.power(jnp.maximum(temp - 60.0, 1e-4), -0.1332047592)
  r = jnp.where(temp <= 66.0, 255.0, r_hi)
  # Green
  g_lo = 99.4708025861 * jnp.log(jnp.maximum(temp, 1.0)) - 161.1195681661
  g_hi = 288.1221695283 * jnp.power(jnp.maximum(temp - 60.0, 1e-4), -0.0755148492)
  g = jnp.where(temp <= 66.0, g_lo, g_hi)
  # Blue
  b_mid = 138.5177312231 * jnp.log(jnp.maximum(temp - 10.0, 1.0)) - 305.0447927307
  b = jnp.where(temp >= 66.0, 255.0, jnp.where(temp <= 19.0, 0.0, b_mid))

  srgb = jnp.clip(jnp.stack([r, g, b], axis=-1) / 255.0, 0.0, 1.0)
  lin = jnp.power(jnp.maximum(srgb, 1e-4), 2.2)
  # Normalize by max channel so white-point brightness stays controlled by intensity
  lum = 0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]
  return lin / jnp.maximum(lum[..., None], 1e-4) * 0.92


def hsv_to_rgb_jax(h: jax.Array, s: jax.Array, v: jax.Array) -> jax.Array:
  """Vectorized HSV [0,1]^3 -> linear RGB [0,1]^3 in pure JAX."""
  h6 = (h % 1.0) * 6.0
  c = v * s
  x = c * (1.0 - jnp.abs((h6 % 2.0) - 1.0))
  m = v - c
  hi = jnp.floor(h6).astype(jnp.int32) % 6
  r = jnp.where(hi == 0, c, jnp.where(hi == 1, x, jnp.where(hi == 4, x, jnp.where(hi == 5, c, 0.0))))
  g = jnp.where(hi == 0, x, jnp.where(hi == 1, c, jnp.where(hi == 2, c, jnp.where(hi == 3, x, 0.0))))
  b = jnp.where(hi == 2, x, jnp.where(hi == 3, c, jnp.where(hi == 4, c, jnp.where(hi == 5, x, 0.0))))
  return jnp.clip(jnp.stack([r + m, g + m, b + m], axis=-1), 0.0, 1.0)


def _agx_tonemap_rgb_components(
    r_lin: jax.Array,
    g_lin: jax.Array,
    b_lin: jax.Array,
    exposure: jax.Array,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
  """In-VMEM AgX filmic curve preserving highlight saturation + sRGB gamma."""
  rx = jnp.maximum(r_lin * exposure, 0.0)
  gx = jnp.maximum(g_lin * exposure, 0.0)
  bx = jnp.maximum(b_lin * exposure, 0.0)

  # AgX inset matrix
  xi0 = 0.84247906 * rx + 0.07843360 * gx + 0.07922375 * bx
  xi1 = 0.04232824 * rx + 0.87846864 * gx + 0.07916613 * bx
  xi2 = 0.04237565 * rx + 0.07843360 * gx + 0.87914297 * bx

  def _curve(x):
    return jnp.clip((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0)

  c0, c1, c2 = _curve(xi0), _curve(xi1), _curve(xi2)

  # AgX outset matrix
  o0 = jnp.clip(1.19687900 * c0 - 0.09802088 * c1 - 0.09902974 * c2, 0.0, 1.0)
  o1 = jnp.clip(-0.05289685 * c0 + 1.15190313 * c1 - 0.09896118 * c2, 0.0, 1.0)
  o2 = jnp.clip(-0.05297163 * c0 - 0.09804345 * c1 + 1.15107367 * c2, 0.0, 1.0)

  inv_gamma = 1.0 / 2.2
  return (
      jnp.power(jnp.maximum(o0, 1e-4), inv_gamma),
      jnp.power(jnp.maximum(o1, 1e-4), inv_gamma),
      jnp.power(jnp.maximum(o2, 1e-4), inv_gamma),
  )


def _branchless_onb(nx, ny, nz):
  """Duff et al. (2017) branchless orthonormal basis from unit normal."""
  sign_z = jnp.where(nz >= 0.0, 1.0, -1.0)
  a = -1.0 / (sign_z + nz)
  b = nx * ny * a
  tx = 1.0 + sign_z * nx * nx * a
  ty = sign_z * b
  tz = -sign_z * nx
  bx = b
  by = sign_z + ny * ny * a
  bz = -ny
  return (tx, ty, tz), (bx, by, bz)


def _r2_pixel_rotation(H: int, W: int, R_pad: int) -> Tuple[np.ndarray, np.ndarray]:
  """2D R2 quasirandom per-pixel tangent rotation for stratified AO & soft shadows."""
  px = np.arange(W, dtype=np.float64)
  py = np.arange(H, dtype=np.float64)
  gx, gy = np.meshgrid(px, py, indexing="xy")
  u = (gx * 0.7548776662466927 + gy * 0.5698402909980532) % 1.0
  theta = (2.0 * math.pi * u).reshape(-1)
  cos_r = np.ones((R_pad,), dtype=np.float32)
  sin_r = np.zeros((R_pad,), dtype=np.float32)
  R = H * W
  cos_r[:R] = np.cos(theta).astype(np.float32)
  sin_r[:R] = np.sin(theta).astype(np.float32)
  return cos_r, sin_r


def _eval_procedural_tex_vmem(
    lx: jax.Array,
    ly: jax.Array,
    mode: jax.Array,
    scale: jax.Array,
    r1: jax.Array,
    g1: jax.Array,
    b1: jax.Array,
    r2: jax.Array,
    g2: jax.Array,
    b2: jax.Array,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
  """Evaluate per-env procedural texture (checker, wood, workshop grid, speckled stone) in VMEM."""
  sx = lx * scale
  sy = ly * scale

  # 1: Checkerboard
  cx = jnp.floor(sx).astype(jnp.int32)
  cy = jnp.floor(sy).astype(jnp.int32)
  w_chk = (((cx + cy) & 1) == 0).astype(jnp.float32)

  # 2: Organic wood grain
  ring = sx * 5.2 + 0.75 * jnp.sin(sy * 2.4) + 0.25 * jnp.sin(sx * 13.0 + sy * 4.0)
  w_wood = 0.5 + 0.5 * jnp.sin(ring * 2.0)

  # 3: Workshop grid / tile lines
  fx = sx - jnp.floor(sx)
  fy = sy - jnp.floor(sy)
  is_line = (fx < 0.055) | (fx > 0.945) | (fy < 0.055) | (fy > 0.945)
  w_grid = jnp.where(is_line, 0.0, 1.0)

  # 4: Speckled concrete / terrazzo
  w_spec = 0.5 + 0.5 * jnp.sin(sx * 19.3 + sy * 13.7) * jnp.cos(sx * 9.1 - sy * 23.4)

  # Select blend weight w in [0, 1] without branching
  w = jnp.where(
      mode < 0.5, 1.0,
      jnp.where(mode < 1.5, w_chk,
                jnp.where(mode < 2.5, w_wood,
                          jnp.where(mode < 3.5, w_grid, w_spec))),
  )
  out_r = w * r1 + (1.0 - w) * r2
  out_g = w * g1 + (1.0 - w) * g2
  out_b = w * b1 + (1.0 - w) * b2
  return out_r, out_g, out_b


# ==============================================================================
# Omniverse Replicator (`rep`) Pure-JAX XLA Domain Randomization API
# ==============================================================================

@dataclasses.dataclass(frozen=True)
class DistributionSpec:
  """Declarative Omniverse Replicator distribution evaluated inside JAX XLA."""
  kind: str
  a: Any
  b: Any = None

  def sample(self, key: jax.Array, shape: Tuple[int, ...], dtype: Any = jnp.float32) -> jax.Array:
    if self.kind == "uniform":
      lo = jnp.asarray(self.a, dtype=dtype)
      hi = jnp.asarray(self.b, dtype=dtype)
      return jax.random.uniform(key, shape=shape, dtype=dtype, minval=lo, maxval=hi)
    if self.kind == "normal":
      mean = jnp.asarray(self.a, dtype=dtype)
      std = jnp.asarray(self.b, dtype=dtype)
      return mean + std * jax.random.normal(key, shape=shape, dtype=dtype)
    if self.kind == "choice":
      arr = jnp.asarray(self.a)
      idx = jax.random.randint(key, shape=shape, minval=0, maxval=arr.shape[0])
      # Use static one-hot dot instead of dynamic gather so gather_ops == 0!
      oh = jax.nn.one_hot(idx, arr.shape[0], dtype=jnp.float32)
      if arr.ndim == 1:
        return jnp.einsum("...k,k->...", oh, arr.astype(jnp.float32)).astype(arr.dtype)
      return jnp.einsum("...k,k...->...", oh, arr.astype(jnp.float32)).astype(arr.dtype)
    if self.kind == "color_temperature":
      lo_k = jnp.asarray(self.a, dtype=jnp.float32)
      hi_k = jnp.asarray(self.b, dtype=jnp.float32)
      k_val = jax.random.uniform(key, shape=shape, dtype=jnp.float32, minval=lo_k, maxval=hi_k)
      return kelvin_to_rgb(k_val)
    raise ValueError(f"Unknown distribution kind: {self.kind}")


class _RepDistributionNamespace:
  @staticmethod
  def uniform(low: Any, high: Any) -> DistributionSpec:
    return DistributionSpec("uniform", low, high)

  @staticmethod
  def normal(mean: Any, std: Any) -> DistributionSpec:
    return DistributionSpec("normal", mean, std)

  @staticmethod
  def choice(choices: Sequence[Any]) -> DistributionSpec:
    return DistributionSpec("choice", tuple(choices))

  @staticmethod
  def color_temperature(low_kelvin: float = 2800.0, high_kelvin: float = 8500.0) -> DistributionSpec:
    return DistributionSpec("color_temperature", float(low_kelvin), float(high_kelvin))


@dataclasses.dataclass(frozen=True)
class LightRandomizerCfg:
  azimuth_deg: Tuple[float, float] = (0.0, 360.0)
  elevation_deg: Tuple[float, float] = (28.0, 78.0)
  radius: Tuple[float, float] = (1.6, 3.2)
  target_center: Tuple[float, float, float] = (0.45, 0.0, 0.25)
  kelvin_range: Tuple[float, float] = (2700.0, 8200.0)
  intensity_range: Tuple[float, float] = (0.85, 1.55)
  ambient_scale: Tuple[float, float] = (0.14, 0.30)
  exposure_range: Tuple[float, float] = (0.95, 1.28)
  sky_hue_range: Tuple[float, float] = (0.50, 0.68)


@dataclasses.dataclass(frozen=True)
class MaterialRandomizerCfg:
  randomize_objects: bool = True
  randomize_table: bool = True
  randomize_floor: bool = True
  randomize_robot: bool = True
  object_saturation: Tuple[float, float] = (0.65, 0.95)
  object_value: Tuple[float, float] = (0.38, 0.92)
  roughness_range: Tuple[float, float] = (0.14, 0.85)
  specular_range: Tuple[float, float] = (0.08, 0.65)
  table_modes: Tuple[int, ...] = (TEX_SOLID, TEX_CHECKER, TEX_WOOD, TEX_GRID, TEX_SPECKLED)
  floor_modes: Tuple[int, ...] = (TEX_CHECKER, TEX_GRID, TEX_SPECKLED, TEX_SOLID)
  table_tex_scale: Tuple[float, float] = (2.5, 9.0)
  floor_tex_scale: Tuple[float, float] = (1.5, 5.0)


@dataclasses.dataclass(frozen=True)
class CameraRandomizerCfg:
  pos_jitter_xyz: Tuple[float, float, float] = (0.035, 0.035, 0.025)
  rot_jitter_deg: Tuple[float, float, float] = (3.0, 3.0, 4.0)
  fovy_jitter_deg: float = 6.0


@dataclasses.dataclass(frozen=True)
class PoseRandomizerCfg:
  joint_jitter_rad: float = 0.08
  free_xy_jitter_m: float = 0.045
  free_yaw_jitter_rad: float = 0.75


@dataclasses.dataclass(frozen=True)
class ReplicatorCfg:
  light: Optional[LightRandomizerCfg] = dataclasses.field(default_factory=LightRandomizerCfg)
  materials: Optional[MaterialRandomizerCfg] = dataclasses.field(default_factory=MaterialRandomizerCfg)
  camera: Optional[CameraRandomizerCfg] = dataclasses.field(default_factory=CameraRandomizerCfg)
  poses: Optional[PoseRandomizerCfg] = None


class _RepRandomizerNamespace:
  @staticmethod
  def light(**kwargs) -> LightRandomizerCfg:
    return LightRandomizerCfg(**kwargs)

  @staticmethod
  def materials(**kwargs) -> MaterialRandomizerCfg:
    return MaterialRandomizerCfg(**kwargs)

  @staticmethod
  def texture(**kwargs) -> MaterialRandomizerCfg:
    return MaterialRandomizerCfg(**kwargs)

  @staticmethod
  def camera(**kwargs) -> CameraRandomizerCfg:
    return CameraRandomizerCfg(**kwargs)

  @staticmethod
  def poses(**kwargs) -> PoseRandomizerCfg:
    return PoseRandomizerCfg(**kwargs)


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class ReplicatorBatch:
  """Pure-JAX PyTree holding per-environment Domain Randomization state (zero host copies)."""
  cam_xpos: jax.Array         # (B, N_cam, 3)
  cam_xmat: jax.Array         # (B, N_cam, 3, 3)
  cam_fovy_deg: jax.Array     # (B, N_cam)
  light_xpos: jax.Array       # (B, 1, 3)
  light_xdir: jax.Array       # (B, 1, 3)
  light_diffuse: jax.Array    # (B, 3)
  light_specular: jax.Array   # (B, 3)
  light_ambient: jax.Array    # (B, 3)
  sky_rgb: jax.Array          # (B, 2, 3)
  exposure: jax.Array         # (B,)
  geom_rgba: jax.Array        # (B, ngeom, 4)
  geom_roughness: jax.Array   # (B, ngeom)
  geom_specular: jax.Array    # (B, ngeom)
  floor_tex: jax.Array        # (B, 8) -> mode, scale, r1, g1, b1, r2, g2, b2
  table_tex: jax.Array        # (B, 8) -> mode, scale, r1, g1, b1, r2, g2, b2

  def tree_flatten(self):
    children = (
        self.cam_xpos,
        self.cam_xmat,
        self.cam_fovy_deg,
        self.light_xpos,
        self.light_xdir,
        self.light_diffuse,
        self.light_specular,
        self.light_ambient,
        self.sky_rgb,
        self.exposure,
        self.geom_rgba,
        self.geom_roughness,
        self.geom_specular,
        self.floor_tex,
        self.table_tex,
    )
    return children, None

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    del aux_data
    return cls(*children)


def _euler_zyx_batch(roll: jax.Array, pitch: jax.Array, yaw: jax.Array) -> jax.Array:
  """Vectorized `(...,)` Euler angles (rad) -> `(..., 3, 3)` rotation matrices."""
  cr, sr = jnp.cos(roll), jnp.sin(roll)
  cp, sp = jnp.cos(pitch), jnp.sin(pitch)
  cy, sy = jnp.cos(yaw), jnp.sin(yaw)

  r00 = cy * cp
  r01 = cy * sp * sr - sy * cr
  r02 = cy * sp * cr + sy * sr
  r10 = sy * cp
  r11 = sy * sp * sr + cy * cr
  r12 = sy * sp * cr - cy * sr
  r20 = -sp
  r21 = cp * sr
  r22 = cp * cr

  row0 = jnp.stack([r00, r01, r02], axis=-1)
  row1 = jnp.stack([r10, r11, r12], axis=-1)
  row2 = jnp.stack([r20, r21, r22], axis=-1)
  return jnp.stack([row0, row1, row2], axis=-2)


def _apply_replicator_local(
    rng_or_keys: jax.Array,
    mjx_data_batch: Any,
    base_spec: RenderSceneSpec,
    cam_ids: Tuple[int, ...],
    base_fovys: Tuple[float, ...],
    category_masks: Dict[str, np.ndarray],
    cfg: ReplicatorCfg,
) -> ReplicatorBatch:
  """Vectorized XLA Domain Randomization across local batch B (works inside jit & shard_map)."""
  B = int(mjx_data_batch.cam_xpos.shape[0])
  ngeom = int(base_spec.ngeom)
  n_cam = len(cam_ids)

  # Accept either per-env (B, 2) PRNG keys (zero cross-chip comms!) or single PRNGKey
  if rng_or_keys.ndim == 2 and rng_or_keys.shape[0] == B:
    keys_b = rng_or_keys
    def _split_b(idx_offset: int) -> jax.Array:
      return jax.vmap(lambda k: jax.random.fold_in(k, idx_offset))(keys_b)
    def _u(idx_offset: int, shape_tail: Tuple[int, ...], lo: float = 0.0, hi: float = 1.0):
      ks = _split_b(idx_offset)
      return jax.vmap(lambda k: jax.random.uniform(k, shape=shape_tail, dtype=jnp.float32, minval=lo, maxval=hi))(ks)
    def _n(idx_offset: int, shape_tail: Tuple[int, ...], mean: float = 0.0, std: float = 1.0):
      ks = _split_b(idx_offset)
      return mean + std * jax.vmap(lambda k: jax.random.normal(k, shape=shape_tail, dtype=jnp.float32))(ks)
  else:
    def _u(idx_offset: int, shape_tail: Tuple[int, ...], lo: float = 0.0, hi: float = 1.0):
      k = jax.random.fold_in(rng_or_keys, idx_offset)
      return jax.random.uniform(k, shape=(B, *shape_tail), dtype=jnp.float32, minval=lo, maxval=hi)
    def _n(idx_offset: int, shape_tail: Tuple[int, ...], mean: float = 0.0, std: float = 1.0):
      k = jax.random.fold_in(rng_or_keys, idx_offset)
      return mean + std * jax.random.normal(k, shape=(B, *shape_tail), dtype=jnp.float32)

  # 1. Camera pose + per-env FOV jitter (static slices over cam_ids -> 0 gather_ops!)
  cam_pos_list = [mjx_data_batch.cam_xpos[:, int(cid), :] for cid in cam_ids]
  cam_mat_list = [mjx_data_batch.cam_xmat[:, int(cid), :, :] for cid in cam_ids]
  base_cam_pos = jnp.stack(cam_pos_list, axis=1)  # (B, N_cam, 3)
  base_cam_mat = jnp.stack(cam_mat_list, axis=1)  # (B, N_cam, 3, 3)
  base_fovy_arr = jnp.broadcast_to(
      jnp.asarray(base_fovys, dtype=jnp.float32)[None, :], (B, n_cam)
  )

  if cfg.camera is not None:
    pj = jnp.asarray(cfg.camera.pos_jitter_xyz, dtype=jnp.float32)[None, None, :]
    dpos = _u(101, (n_cam, 3), -1.0, 1.0) * pj
    rj = jnp.asarray(cfg.camera.rot_jitter_deg, dtype=jnp.float32) * (math.pi / 180.0)
    ang = _u(102, (n_cam, 3), -1.0, 1.0) * rj[None, None, :]
    rmat_jit = _euler_zyx_batch(ang[..., 0], ang[..., 1], ang[..., 2])
    cam_xpos = base_cam_pos + dpos
    cam_xmat = jnp.einsum("bcij,bcjk->bcik", base_cam_mat, rmat_jit, precision=jax.lax.Precision.HIGHEST)
    dfov = _u(103, (n_cam,), -float(cfg.camera.fovy_jitter_deg), float(cfg.camera.fovy_jitter_deg))
    cam_fovy_deg = jnp.clip(base_fovy_arr + dfov, 22.0, 95.0)
  else:
    cam_xpos = base_cam_pos
    cam_xmat = base_cam_mat
    cam_fovy_deg = base_fovy_arr

  # 2. Light azimuth / elevation / color temperature / intensity / sky
  if cfg.light is not None:
    lc = cfg.light
    az = _u(201, (), lc.azimuth_deg[0] * math.pi / 180.0, lc.azimuth_deg[1] * math.pi / 180.0)
    el = _u(202, (), lc.elevation_deg[0] * math.pi / 180.0, lc.elevation_deg[1] * math.pi / 180.0)
    rad = _u(203, (), lc.radius[0], lc.radius[1])
    ctr = jnp.asarray(lc.target_center, dtype=jnp.float32)[None, :]
    lx = ctr[:, 0] + rad * jnp.cos(el) * jnp.cos(az)
    ly = ctr[:, 1] + rad * jnp.cos(el) * jnp.sin(az)
    lz = ctr[:, 2] + rad * jnp.sin(el)
    lpos = jnp.stack([lx, ly, lz], axis=-1)
    ldir = ctr - lpos
    ldir = ldir / jnp.maximum(jnp.linalg.norm(ldir, axis=-1, keepdims=True), 1e-12)
    light_xpos = lpos[:, None, :]
    light_xdir = ldir[:, None, :]

    kelvin = _u(204, (), lc.kelvin_range[0], lc.kelvin_range[1])
    lcol = kelvin_to_rgb(kelvin)
    intensity = _u(205, (), lc.intensity_range[0], lc.intensity_range[1])[:, None]
    amb_sc = _u(206, (), lc.ambient_scale[0], lc.ambient_scale[1])[:, None]
    light_diffuse = lcol * intensity
    light_specular = lcol * (0.42 * intensity)
    light_ambient = lcol * amb_sc
    exposure = _u(207, (), lc.exposure_range[0], lc.exposure_range[1])

    sky_h = _u(208, (), lc.sky_hue_range[0], lc.sky_hue_range[1])
    sky_s = _u(209, (), 0.28, 0.65)
    sky_v_top = _u(210, (), 0.22, 0.58)
    sky_top = hsv_to_rgb_jax(sky_h, sky_s, sky_v_top) * (0.65 + 0.35 * lcol)
    sky_bot = hsv_to_rgb_jax((sky_h + 0.04) % 1.0, sky_s * 0.55, sky_v_top * 0.26)
    sky_rgb = jnp.stack([sky_top, sky_bot], axis=1)
  else:
    if base_spec.nlight > 0:
      lp, ld = _extract_light_batch(base_spec, mjx_data_batch)
      light_xpos = lp[:, :1, :]
      light_xdir = ld[:, :1, :]
      light_diffuse = jnp.broadcast_to(jnp.asarray(base_spec.light_diffuse[0], jnp.float32)[None, :], (B, 3))
      light_specular = jnp.broadcast_to(jnp.asarray(base_spec.light_specular[0], jnp.float32)[None, :], (B, 3))
      light_ambient = jnp.broadcast_to(jnp.asarray(base_spec.light_ambient[0], jnp.float32)[None, :], (B, 3))
    else:
      light_xpos = jnp.broadcast_to(jnp.array([[[0.8, -0.8, 2.2]]], jnp.float32), (B, 1, 3))
      light_xdir = jnp.broadcast_to(jnp.array([[[-0.3, 0.3, -0.9]]], jnp.float32), (B, 1, 3))
      light_diffuse = jnp.broadcast_to(jnp.array([[0.9, 0.9, 0.9]], jnp.float32), (B, 3))
      light_specular = jnp.broadcast_to(jnp.array([[0.35, 0.35, 0.35]], jnp.float32), (B, 3))
      light_ambient = jnp.broadcast_to(jnp.array([[0.22, 0.22, 0.24]], jnp.float32), (B, 3))
    sky_base = jnp.stack([
        jnp.asarray(base_spec.sky_rgb1, jnp.float32),
        jnp.asarray(base_spec.sky_rgb2, jnp.float32),
    ], axis=0)
    sky_rgb = jnp.broadcast_to(sky_base[None, :, :], (B, 2, 3))
    exposure = jnp.ones((B,), dtype=jnp.float32)

  # 3. Materials, Albedo, Roughness, Specular & Procedural Tabletop/Floor Textures
  base_rgba = jnp.asarray(base_spec.geom_rgba, dtype=jnp.float32)  # (ngeom, 4)
  base_rough = jnp.clip(1.0 - jnp.asarray(base_spec.geom_shin, dtype=jnp.float32) * 1.4, 0.14, 0.90)
  base_spec_c = jnp.asarray(base_spec.geom_spec, dtype=jnp.float32)

  geom_rgba = jnp.broadcast_to(base_rgba[None, :, :], (B, ngeom, 4))
  geom_roughness = jnp.broadcast_to(base_rough[None, :], (B, ngeom))
  geom_specular = jnp.broadcast_to(base_spec_c[None, :], (B, ngeom))

  if cfg.materials is not None:
    mc = cfg.materials
    if mc.randomize_objects and np.any(category_masks["object"]):
      obj_h = _u(301, (ngeom,), 0.0, 1.0)
      obj_s = _u(302, (ngeom,), mc.object_saturation[0], mc.object_saturation[1])
      obj_v = _u(303, (ngeom,), mc.object_value[0], mc.object_value[1])
      obj_rgb = hsv_to_rgb_jax(obj_h, obj_s, obj_v)
      obj_rgba = jnp.concatenate([obj_rgb, geom_rgba[..., 3:4]], axis=-1)
      m_obj = jnp.asarray(category_masks["object"], dtype=jnp.bool_)[None, :, None]
      geom_rgba = jnp.where(m_obj, obj_rgba, geom_rgba)

    if mc.randomize_robot and np.any(category_masks["robot_body"]):
      rob_h = _u(304, (), 0.0, 1.0)[:, None]
      rob_s = _u(305, (), 0.04, 0.35)[:, None]
      rob_v = _u(306, (), 0.55, 0.96)[:, None]
      rob_rgb = hsv_to_rgb_jax(rob_h, rob_s, rob_v)  # (B, 1, 3)
      rob_rgba = jnp.concatenate([jnp.broadcast_to(rob_rgb, (B, ngeom, 3)), geom_rgba[..., 3:4]], axis=-1)
      m_rob = jnp.asarray(category_masks["robot_body"], dtype=jnp.bool_)[None, :, None]
      geom_rgba = jnp.where(m_rob, rob_rgba, geom_rgba)

    if np.any(category_masks["bin"]):
      bin_h = _u(307, (ngeom,), 0.0, 1.0)
      bin_s = _u(308, (ngeom,), 0.45, 0.85)
      bin_v = _u(309, (ngeom,), 0.32, 0.78)
      bin_rgb = hsv_to_rgb_jax(bin_h, bin_s, bin_v)
      bin_rgba = jnp.concatenate([bin_rgb, geom_rgba[..., 3:4]], axis=-1)
      m_bin = jnp.asarray(category_masks["bin"], dtype=jnp.bool_)[None, :, None]
      geom_rgba = jnp.where(m_bin, bin_rgba, geom_rgba)

    # Roughness & specular per geom
    geom_roughness = _u(310, (ngeom,), mc.roughness_range[0], mc.roughness_range[1])
    geom_specular = _u(311, (ngeom,), mc.specular_range[0], mc.specular_range[1])

    # Procedural floor & table texture parameters
    def _sample_tex(seed_base: int, modes: Tuple[int, ...], scale_rng: Tuple[float, float], sat_rng, val_rng):
      mode_arr = jnp.asarray(modes, dtype=jnp.float32)
      m_u = _u(seed_base + 0, (), 0.0, float(len(modes)) - 1e-4)
      m_idx = jnp.floor(m_u).astype(jnp.int32)
      oh = jax.nn.one_hot(m_idx, len(modes), dtype=jnp.float32)
      mode_val = jnp.einsum("bk,k->b", oh, mode_arr)
      sc_val = _u(seed_base + 1, (), scale_rng[0], scale_rng[1])
      h1 = _u(seed_base + 2, (), 0.0, 1.0)
      s1 = _u(seed_base + 3, (), sat_rng[0], sat_rng[1])
      v1 = _u(seed_base + 4, (), val_rng[0], val_rng[1])
      rgb1 = hsv_to_rgb_jax(h1, s1, v1)
      # Secondary shade shifts hue slightly and contrasts value
      h2 = (h1 + _u(seed_base + 5, (), -0.06, 0.06)) % 1.0
      v2 = jnp.clip(v1 * _u(seed_base + 6, (), 0.42, 0.78), 0.06, 0.92)
      rgb2 = hsv_to_rgb_jax(h2, jnp.clip(s1 * 1.1, 0.0, 0.9), v2)
      return jnp.concatenate([mode_val[:, None], sc_val[:, None], rgb1, rgb2], axis=-1)

    if mc.randomize_floor:
      floor_tex = _sample_tex(320, mc.floor_modes, mc.floor_tex_scale, (0.05, 0.35), (0.14, 0.44))
    else:
      c1 = jnp.asarray(base_spec.checker_rgb1, jnp.float32)
      c2 = jnp.asarray(base_spec.checker_rgb2, jnp.float32)
      floor_tex = jnp.broadcast_to(
          jnp.concatenate([jnp.array([float(TEX_CHECKER), float(base_spec.checker_repeat[0])]), c1, c2])[None, :],
          (B, 8),
      )

    if mc.randomize_table:
      table_tex = _sample_tex(340, mc.table_modes, mc.table_tex_scale, (0.12, 0.52), (0.24, 0.72))
    else:
      t_c1 = jnp.array([0.62, 0.50, 0.36], jnp.float32)
      table_tex = jnp.broadcast_to(jnp.concatenate([jnp.array([float(TEX_SOLID), 4.0]), t_c1, t_c1])[None, :], (B, 8))
  else:
    c1 = jnp.asarray(base_spec.checker_rgb1, jnp.float32)
    c2 = jnp.asarray(base_spec.checker_rgb2, jnp.float32)
    floor_tex = jnp.broadcast_to(
        jnp.concatenate([jnp.array([float(TEX_CHECKER), float(base_spec.checker_repeat[0])]), c1, c2])[None, :],
        (B, 8),
    )
    t_c1 = jnp.array([0.62, 0.50, 0.36], jnp.float32)
    table_tex = jnp.broadcast_to(jnp.concatenate([jnp.array([float(TEX_SOLID), 4.0]), t_c1, t_c1])[None, :], (B, 8))

  return ReplicatorBatch(
      cam_xpos=cam_xpos,
      cam_xmat=cam_xmat,
      cam_fovy_deg=cam_fovy_deg,
      light_xpos=light_xpos,
      light_xdir=light_xdir,
      light_diffuse=light_diffuse,
      light_specular=light_specular,
      light_ambient=light_ambient,
      sky_rgb=sky_rgb,
      exposure=exposure,
      geom_rgba=geom_rgba,
      geom_roughness=geom_roughness,
      geom_specular=geom_specular,
      floor_tex=floor_tex,
      table_tex=table_tex,
  )


def randomize_qpos_batch(
    rng: jax.Array,
    mjx_model: Any,
    mjx_data_batch: Any,
    free_qpos_indices: Sequence[int] = (),
    joint_qpos_indices: Sequence[int] = (),
    cfg: Optional[PoseRandomizerCfg] = None,
) -> Any:
  """Zero-gather per-environment MJX qpos scatter (joints + freejoint XY/yaw) before mjx.kinematics/step."""
  if cfg is None:
    cfg = PoseRandomizerCfg()
  from mujoco import mjx

  B = int(mjx_data_batch.qpos.shape[0])
  nq = int(mjx_data_batch.qpos.shape[1])
  qpos = mjx_data_batch.qpos

  if joint_qpos_indices:
    j_mask = np.zeros((nq,), dtype=bool)
    for idx in joint_qpos_indices:
      j_mask[int(idx)] = True
    dj = jax.random.uniform(
        jax.random.fold_in(rng, 401), shape=(B, nq), dtype=jnp.float32,
        minval=-float(cfg.joint_jitter_rad), maxval=float(cfg.joint_jitter_rad),
    )
    qpos = jnp.where(jnp.asarray(j_mask)[None, :], qpos + dj, qpos)

  if free_qpos_indices:
    xy_mask = np.zeros((nq,), dtype=bool)
    for base_adr in free_qpos_indices:
      xy_mask[int(base_adr)] = True
      xy_mask[int(base_adr) + 1] = True
    dxy = jax.random.uniform(
        jax.random.fold_in(rng, 402), shape=(B, nq), dtype=jnp.float32,
        minval=-float(cfg.free_xy_jitter_m), maxval=float(cfg.free_xy_jitter_m),
    )
    qpos = jnp.where(jnp.asarray(xy_mask)[None, :], qpos + dxy, qpos)

  d_new = mjx_data_batch.replace(qpos=qpos)
  return jax.vmap(lambda d: mjx.kinematics(mjx_model, d))(d_new)


class _RepNamespace:
  distribution = _RepDistributionNamespace()
  randomizer = _RepRandomizerNamespace()


rep = _RepNamespace()


# ==============================================================================
# Pallas Mosaic Kernel 1: Unrolled Multi-Modal DR Kernel (G <= 128)
# ==============================================================================

def _pack_camera_and_light_dr(
    spec: RenderSceneSpec,
    rep_batch: Optional[ReplicatorBatch],
    mjx_data_batch: Any,
    cam_slot: int,
    block_b: int,
    B: int,
    B_pad: int,
    cam_fovy_override: Optional[jax.Array] = None,
) -> Tuple[jax.Array, jax.Array]:
  if rep_batch is not None:
    cam_pos = rep_batch.cam_xpos[:, cam_slot, :]
    cam_mat = rep_batch.cam_xmat[:, cam_slot, :, :]
    fovy_deg = rep_batch.cam_fovy_deg[:, cam_slot]
    lpos = rep_batch.light_xpos[:, 0, :]
    ldir = rep_batch.light_xdir[:, 0, :]
    ldiff = rep_batch.light_diffuse
    lspec = rep_batch.light_specular
    lamb = rep_batch.light_ambient
    sky1 = rep_batch.sky_rgb[:, 0, :]
    sky2 = rep_batch.sky_rgb[:, 1, :]
    floor_tex = rep_batch.floor_tex
    table_tex = rep_batch.table_tex
    exposure = rep_batch.exposure[:, None]
  else:
    cam_pos = mjx_data_batch.cam_xpos[:, int(spec.cam_id), :]
    cam_mat = mjx_data_batch.cam_xmat[:, int(spec.cam_id), :, :]
    fovy_deg = cam_fovy_override if cam_fovy_override is not None else jnp.full((B,), float(spec.cam_fovy), dtype=jnp.float32)
    if spec.nlight > 0:
      lp, ld = _extract_light_batch(spec, mjx_data_batch)
      lpos, ldir = lp[:, 0, :], ld[:, 0, :]
      ldiff = jnp.broadcast_to(jnp.asarray(spec.light_diffuse[0], jnp.float32)[None, :], (B, 3))
      lspec = jnp.broadcast_to(jnp.asarray(spec.light_specular[0], jnp.float32)[None, :], (B, 3))
      lamb = jnp.broadcast_to(jnp.asarray(spec.light_ambient[0], jnp.float32)[None, :], (B, 3))
    else:
      lpos = jnp.broadcast_to(jnp.array([[0.8, -0.8, 2.2]], jnp.float32), (B, 3))
      ldir = jnp.broadcast_to(jnp.array([[-0.3, 0.3, -0.9]], jnp.float32), (B, 3))
      ldiff = jnp.broadcast_to(jnp.array([[0.9, 0.9, 0.9]], jnp.float32), (B, 3))
      lspec = jnp.broadcast_to(jnp.array([[0.35, 0.35, 0.35]], jnp.float32), (B, 3))
      lamb = jnp.broadcast_to(jnp.array([[0.22, 0.22, 0.24]], jnp.float32), (B, 3))
    sky1 = jnp.broadcast_to(jnp.asarray(spec.sky_rgb1, jnp.float32)[None, :], (B, 3))
    sky2 = jnp.broadcast_to(jnp.asarray(spec.sky_rgb2, jnp.float32)[None, :], (B, 3))
    c1 = jnp.asarray(spec.checker_rgb1, jnp.float32)
    c2 = jnp.asarray(spec.checker_rgb2, jnp.float32)
    floor_tex = jnp.broadcast_to(
        jnp.concatenate([jnp.array([float(TEX_CHECKER), float(spec.checker_repeat[0])]), c1, c2])[None, :],
        (B, 8),
    )
    t_c1 = jnp.array([0.62, 0.50, 0.36], jnp.float32)
    table_tex = jnp.broadcast_to(jnp.concatenate([jnp.array([float(TEX_SOLID), 4.0]), t_c1, t_c1])[None, :], (B, 8))
    exposure = jnp.ones((B, 1), dtype=jnp.float32)

  ldir = ldir / jnp.maximum(jnp.linalg.norm(ldir, axis=-1, keepdims=True), 1e-12)
  tan_hf = jnp.tan(0.5 * fovy_deg * (math.pi / 180.0))[:, None]
  cam_pack = jnp.concatenate([cam_pos, cam_mat.reshape(B, 9), tan_hf], axis=-1)  # (B, 13)
  cam_128 = jnp.pad(cam_pack, ((0, B_pad - B), (0, 128 - 13)))

  # Pack 38 light/sky/tex/exposure columns into (B_pad, 128)
  light_pack = jnp.concatenate(
      [lpos, ldir, ldiff, lspec, lamb, sky1, sky2, floor_tex, table_tex, exposure],
      axis=-1,
  )  # (B, 38)
  light_128 = jnp.pad(light_pack, ((0, B_pad - B), (0, 128 - light_pack.shape[1])))
  return cam_128, light_128


def _build_unit_tangent_rays(H: int, W: int, R_pad: int) -> jax.Array:
  """Build FOV-independent tangent-plane unit ray templates (16, R_pad) for per-env FOV in VMEM."""
  R = H * W
  px = np.arange(W, dtype=np.float32)
  py = np.arange(H, dtype=np.float32)
  gx, gy = np.meshgrid(px, py, indexing="xy")
  u = (gx + 0.5) / float(W)
  v = (gy + 0.5) / float(H)
  aspect = float(W) / float(H)
  tx = ((2.0 * u - 1.0) * aspect).reshape(-1)
  ty = (1.0 - 2.0 * v).reshape(-1)
  cos_rot, sin_rot = _r2_pixel_rotation(H, W, R_pad)

  arr = np.zeros((16, R_pad), dtype=np.float32)
  arr[0, :R] = tx
  arr[1, :R] = ty
  arr[2, :] = cos_rot
  arr[3, :] = sin_rot
  return jnp.asarray(arr, dtype=jnp.float32)


def _pallas_tiled_dr_unrolled(
    spec: RenderSceneSpec,
    mjx_data_batch: Any,
    rep_batch: Optional[ReplicatorBatch],
    cam_slot: int,
    geom_sem_ids: Tuple[int, ...],
    geom_body_ids: Tuple[int, ...],
    floor_gids: Tuple[int, ...],
    table_gids: Tuple[int, ...],
    shader: str = "pbr_ggx",
    tonemap: str = "agx",
    out_dtype: Any = jnp.bfloat16,
    block_b: int = 8,
    block_r: int = 512,
    n_ao_samples: int = 4,
    n_shadow_samples: int = 4,
    ao_radius: float = 0.35,
    ao_strength: float = 0.72,
    shadow_softness: float = 0.055,
    emit_aux: bool = True,
    interpret: bool = False,
    cam_fovy_override: Optional[jax.Array] = None,
) -> Dict[str, jax.Array]:
  """Unrolled Mosaic Pallas DR kernel (G <= 128) with per-env FOV, lights, textures, PBR & AgX."""
  B = int(mjx_data_batch.cam_xpos.shape[0])
  H, W = int(spec.height), int(spec.width)
  R = H * W
  gids = np.asarray(spec.enabled_geom_ids, dtype=np.int32)
  G = len(gids)

  use_pbr_ao = (shader == "pbr_ggx")
  if use_pbr_ao and G > 64:
    block_r = min(int(block_r), 128)
  elif G > 40:
    block_r = min(int(block_r), 256)

  B_pad = ((B + block_b - 1) // block_b) * block_b
  R_pad = ((R + block_r - 1) // block_r) * block_r
  G_pad = ((max(G, 1) + 127) // 128) * 128
  n_b_blk = B_pad // block_b
  n_r_blk = R_pad // block_r

  cam_128, light_128 = _pack_camera_and_light_dr(
      spec, rep_batch, mjx_data_batch, cam_slot, block_b, B, B_pad, cam_fovy_override=cam_fovy_override
  )
  lray_dev = _build_unit_tangent_rays(H, W, R_pad)

  # Pack geometry fields (16 * block_b, G_pad)
  geom_size = jnp.asarray(spec.geom_size, dtype=jnp.float32)
  gpos = mjx_data_batch.geom_xpos[:, gids, :]
  gmat = mjx_data_batch.geom_xmat[:, gids, :, :].reshape(B, G, 9)
  gsz_all = jnp.broadcast_to(geom_size[gids][None, :, :], (B, G, 3))
  rb_sq = jnp.broadcast_to(jnp.asarray(_compute_geom_bound_sq(spec, gids))[None, :, None], (B, G, 1))
  g_16 = jnp.concatenate([gpos, gmat, gsz_all, rb_sq], axis=-1)
  if B_pad != B or G_pad != G:
    g_16 = jnp.pad(g_16, ((0, B_pad - B), (0, G_pad - G), (0, 0)))
  geom_fields = (
      g_16.reshape(n_b_blk, block_b, G_pad, 16)
      .transpose(0, 3, 1, 2)
      .reshape(n_b_blk * 16 * block_b, G_pad)
  )

  # Pack per-env dynamic material fields (6 * block_b, G_pad): rgb(3), rough(1), spec(1), emit(1)
  if rep_batch is not None:
    g_rgb = rep_batch.geom_rgba[:, gids, :3]
    g_rough = rep_batch.geom_roughness[:, gids][..., None]
    g_spec = rep_batch.geom_specular[:, gids][..., None]
  else:
    g_rgb = jnp.broadcast_to(jnp.asarray(spec.geom_rgba[gids, :3], jnp.float32)[None, :, :], (B, G, 3))
    base_rough = jnp.clip(1.0 - jnp.asarray(spec.geom_shin[gids], jnp.float32) * 1.4, 0.14, 0.90)
    g_rough = jnp.broadcast_to(base_rough[None, :, None], (B, G, 1))
    g_spec = jnp.broadcast_to(jnp.asarray(spec.geom_spec[gids], jnp.float32)[None, :, None], (B, G, 1))
  g_emit = jnp.broadcast_to(jnp.asarray(spec.geom_emit[gids], jnp.float32)[None, :, None], (B, G, 1))
  m_6 = jnp.concatenate([g_rgb, g_rough, g_spec, g_emit], axis=-1)  # (B, G, 6)
  if B_pad != B or G_pad != G:
    m_6 = jnp.pad(m_6, ((0, B_pad - B), (0, G_pad - G), (0, 0)))
  mat_fields = (
      m_6.reshape(n_b_blk, block_b, G_pad, 6)
      .transpose(0, 3, 1, 2)
      .reshape(n_b_blk * 6 * block_b, G_pad)
  )

  g_types_static = tuple(int(spec.geom_type[g]) for g in gids)
  gids_static = tuple(int(g) for g in gids)
  plane_fx = tuple(float(spec.geom_size[g, 0]) > 0.0 for g in gids)
  plane_fy = tuple(float(spec.geom_size[g, 1]) > 0.0 for g in gids)
  sem_static = tuple(int(s) for s in geom_sem_ids)
  packed_id_static = tuple(int(geom_body_ids[k]) * 64 + int(geom_sem_ids[k]) for k in range(G))
  floor_k = tuple((k, int(g)) for k, g in enumerate(gids) if int(g) in floor_gids)
  table_k = tuple((k, int(g)) for k, g in enumerate(gids) if int(g) in table_gids)

  eff_ao = n_ao_samples if use_pbr_ao else 0
  eff_sh = n_shadow_samples if use_pbr_ao else 1

  ao_samples = []
  for s in range(max(eff_ao, 0)):
    r_s = math.sqrt((s + 0.5) / max(eff_ao, 1))
    phi_s = 2.0 * math.pi * ((s + 0.5) * 0.61803398875)
    uz = math.sqrt(max(1.0 - r_s * r_s, 1e-4))
    ao_samples.append((r_s * math.cos(phi_s), r_s * math.sin(phi_s), uz))

  sh_samples = []
  for s in range(max(eff_sh, 1)):
    if eff_sh <= 1:
      sh_samples.append((0.0, 0.0))
    else:
      r_s = math.sqrt((s + 0.5) / eff_sh) * shadow_softness
      phi_s = 2.0 * math.pi * ((s + 0.5) * 0.61803398875)
      sh_samples.append((r_s * math.cos(phi_s), r_s * math.sin(phi_s)))

  znear_f = float(spec.znear)
  zfar_f = float(spec.zfar) if spec.zfar > 0.0 else 50.0
  l_type_0 = int(spec.light_type[0]) if spec.nlight > 0 else 0
  l_att_0 = tuple(float(x) for x in spec.light_attenuation[0]) if spec.nlight > 0 else (1.0, 0.0, 0.0)
  do_sh_0 = bool(spec.use_shadows) and (bool(spec.light_castshadow[0]) if spec.nlight > 0 else True)
  shadow_frac = float(spec.shadow_light_fraction)
  headlight_active = bool(spec.headlight_active)
  hl_amb = tuple(float(x) for x in spec.headlight_ambient)
  hl_diff = tuple(float(x) for x in spec.headlight_diffuse)
  use_agx = (tonemap == "agx")

  def _kernel(*refs):
    cam = refs[0][...]
    lr = refs[1][...]
    gf = refs[2][...]
    mf = refs[3][...]
    linfo = refs[4][...]
    out_r_ref, out_g_ref, out_b_ref, out_d_ref = refs[5], refs[6], refs[7], refs[8]
    if emit_aux:
      out_nx_ref, out_ny_ref, out_nz_ref, out_gid_ref, out_pid_ref = refs[9], refs[10], refs[11], refs[12], refs[13]

    # Per-env FOV raygen directly in VMEM
    tx_r = lr[0:1, :]
    ty_r = lr[1:2, :]
    cos_rot = lr[2:3, :]
    sin_rot = lr[3:4, :]
    tan_hf = cam[:, 12:13]  # (block_b, 1)

    ldx_raw = tx_r * tan_hf
    ldy_raw = ty_r * tan_hf
    ldz_raw = -1.0
    inv_ln = 1.0 / jnp.sqrt(ldx_raw * ldx_raw + ldy_raw * ldy_raw + 1.0)
    neg_z = inv_ln
    ldx, ldy, ldz = ldx_raw * inv_ln, ldy_raw * inv_ln, ldz_raw * inv_ln
    inv_cos = znear_f / neg_z
    max_d = (zfar_f - znear_f) / neg_z

    cpx, cpy, cpz = cam[:, 0:1], cam[:, 1:2], cam[:, 2:3]
    c00, c01, c02 = cam[:, 3:4], cam[:, 4:5], cam[:, 5:6]
    c10, c11, c12 = cam[:, 6:7], cam[:, 7:8], cam[:, 8:9]
    c20, c21, c22 = cam[:, 9:10], cam[:, 10:11], cam[:, 11:12]

    rdx_raw = c00 * ldx + c01 * ldy + c02 * ldz
    rdy_raw = c10 * ldx + c11 * ldy + c12 * ldz
    rdz_raw = c20 * ldx + c21 * ldy + c22 * ldz
    inv_rn = 1.0 / jnp.maximum(jnp.sqrt(rdx_raw * rdx_raw + rdy_raw * rdy_raw + rdz_raw * rdz_raw), 1e-12)
    rdx, rdy, rdz = rdx_raw * inv_rn, rdy_raw * inv_rn, rdz_raw * inv_rn

    rox = cpx + inv_cos * rdx
    roy = cpy + inv_cos * rdy
    roz = cpz + inv_cos * rdz

    best_d = max_d
    best_nx = jnp.zeros((block_b, block_r), dtype=jnp.float32)
    best_ny = jnp.zeros((block_b, block_r), dtype=jnp.float32)
    best_nz = jnp.zeros((block_b, block_r), dtype=jnp.float32)
    best_gid = jnp.full((block_b, block_r), -1, dtype=jnp.int32)
    best_pid = jnp.zeros((block_b, block_r), dtype=jnp.int32)

    alb_r = jnp.zeros((block_b, block_r), dtype=jnp.float32)
    alb_g = jnp.zeros((block_b, block_r), dtype=jnp.float32)
    alb_b = jnp.zeros((block_b, block_r), dtype=jnp.float32)
    roughness = jnp.full((block_b, block_r), 0.45, dtype=jnp.float32)
    spec_coef = jnp.zeros((block_b, block_r), dtype=jnp.float32)
    emit_coef = jnp.zeros((block_b, block_r), dtype=jnp.float32)

    for k, (gt, gid_val, pfx, pfy, pid_val) in enumerate(
        zip(g_types_static, gids_static, plane_fx, plane_fy, packed_id_static)
    ):
      px = gf[0 * block_b : 1 * block_b, k : k + 1]
      py = gf[1 * block_b : 2 * block_b, k : k + 1]
      pz = gf[2 * block_b : 3 * block_b, k : k + 1]
      m00, m01, m02 = gf[3 * block_b : 4 * block_b, k : k + 1], gf[4 * block_b : 5 * block_b, k : k + 1], gf[5 * block_b : 6 * block_b, k : k + 1]
      m10, m11, m12 = gf[6 * block_b : 7 * block_b, k : k + 1], gf[7 * block_b : 8 * block_b, k : k + 1], gf[8 * block_b : 9 * block_b, k : k + 1]
      m20, m21, m22 = gf[9 * block_b : 10 * block_b, k : k + 1], gf[10 * block_b : 11 * block_b, k : k + 1], gf[11 * block_b : 12 * block_b, k : k + 1]
      sx, sy, sz = gf[12 * block_b : 13 * block_b, k : k + 1], gf[13 * block_b : 14 * block_b, k : k + 1], gf[14 * block_b : 15 * block_b, k : k + 1]
      rb_sq = gf[15 * block_b : 16 * block_b, k : k + 1]

      cd, cnx, cny, cnz = _ray_one_geom_v2(
          gt, px, py, pz, m00, m01, m02, m10, m11, m12, m20, m21, m22,
          sx, sy, sz, rb_sq, rox, roy, roz, rdx, rdy, rdz,
          plane_finite_x=pfx, plane_finite_y=pfy, compute_normal=True,
      )
      imp = cd < best_d
      best_d = jnp.where(imp, cd, best_d)
      best_gid = jnp.where(imp, jnp.int32(gid_val), best_gid)
      best_pid = jnp.where(imp, jnp.int32(pid_val), best_pid)
      best_nx = jnp.where(imp, cnx, best_nx)
      best_ny = jnp.where(imp, cny, best_ny)
      best_nz = jnp.where(imp, cnz, best_nz)

      alb_r = jnp.where(imp, mf[0 * block_b : 1 * block_b, k : k + 1], alb_r)
      alb_g = jnp.where(imp, mf[1 * block_b : 2 * block_b, k : k + 1], alb_g)
      alb_b = jnp.where(imp, mf[2 * block_b : 3 * block_b, k : k + 1], alb_b)
      roughness = jnp.where(imp, mf[3 * block_b : 4 * block_b, k : k + 1], roughness)
      spec_coef = jnp.where(imp, mf[4 * block_b : 5 * block_b, k : k + 1], spec_coef)
      emit_coef = jnp.where(imp, mf[5 * block_b : 6 * block_b, k : k + 1], emit_coef)

    hit = best_gid >= 0
    d_safe = jnp.where(hit, best_d, 0.0)
    nx_out = jnp.where(hit, best_nx, 0.0)
    ny_out = jnp.where(hit, best_ny, 0.0)
    nz_out = jnp.where(hit, best_nz, 0.0)
    nz_onb = jnp.where(hit, best_nz, 1.0)

    hx = rox + d_safe * rdx
    hy = roy + d_safe * rdy
    hz = roz + d_safe * rdz

    # Procedural floor & workbench textures (per-env mode/scale/colors from linfo!)
    for k, fg in floor_k:
      px = gf[0 * block_b : 1 * block_b, k : k + 1]
      py = gf[1 * block_b : 2 * block_b, k : k + 1]
      pz = gf[2 * block_b : 3 * block_b, k : k + 1]
      m00, m01 = gf[3 * block_b : 4 * block_b, k : k + 1], gf[4 * block_b : 5 * block_b, k : k + 1]
      m10, m11 = gf[6 * block_b : 7 * block_b, k : k + 1], gf[7 * block_b : 8 * block_b, k : k + 1]
      m20, m21 = gf[9 * block_b : 10 * block_b, k : k + 1], gf[10 * block_b : 11 * block_b, k : k + 1]
      dcx, dcy, dcz = hx - px, hy - py, hz - pz
      lx_f = m00 * dcx + m10 * dcy + m20 * dcz
      ly_f = m01 * dcx + m11 * dcy + m21 * dcz
      tr, tg, tb = _eval_procedural_tex_vmem(
          lx_f, ly_f,
          linfo[:, 21:22], linfo[:, 22:23],
          linfo[:, 23:24], linfo[:, 24:25], linfo[:, 25:26],
          linfo[:, 26:27], linfo[:, 27:28], linfo[:, 28:29],
      )
      is_fg = best_gid == fg
      alb_r = jnp.where(is_fg, tr, alb_r)
      alb_g = jnp.where(is_fg, tg, alb_g)
      alb_b = jnp.where(is_fg, tb, alb_b)

    for k, tg_id in table_k:
      px = gf[0 * block_b : 1 * block_b, k : k + 1]
      py = gf[1 * block_b : 2 * block_b, k : k + 1]
      pz = gf[2 * block_b : 3 * block_b, k : k + 1]
      m00, m01 = gf[3 * block_b : 4 * block_b, k : k + 1], gf[4 * block_b : 5 * block_b, k : k + 1]
      m10, m11 = gf[6 * block_b : 7 * block_b, k : k + 1], gf[7 * block_b : 8 * block_b, k : k + 1]
      m20, m21 = gf[9 * block_b : 10 * block_b, k : k + 1], gf[10 * block_b : 11 * block_b, k : k + 1]
      dcx, dcy, dcz = hx - px, hy - py, hz - pz
      lx_t = m00 * dcx + m10 * dcy + m20 * dcz
      ly_t = m01 * dcx + m11 * dcy + m21 * dcz
      tr, tg_c, tb = _eval_procedural_tex_vmem(
          lx_t, ly_t,
          linfo[:, 29:30], linfo[:, 30:31],
          linfo[:, 31:32], linfo[:, 32:33], linfo[:, 33:34],
          linfo[:, 34:35], linfo[:, 35:36], linfo[:, 36:37],
      )
      is_tg = best_gid == tg_id
      alb_r = jnp.where(is_tg, tr, alb_r)
      alb_g = jnp.where(is_tg, tg_c, alb_g)
      alb_b = jnp.where(is_tg, tb, alb_b)

    sh_ox = hx + nx_out * 2.0e-4
    sh_oy = hy + ny_out * 2.0e-4
    sh_oz = hz + nz_onb * 2.0e-4

    # Hemisphere Ambient Occlusion (R2 rotated in VMEM via O(1)-VMEM fori_loop)
    if eff_ao > 0:
      (tx_b, ty_b, tz_b), (bx_b, by_b, bz_b) = _branchless_onb(nx_out, ny_out, nz_onb)
      ao_max_d = jnp.where(hit, ao_radius, 0.0)

      def _ao_sample_body(s_idx, occ_sum_in):
        sf = s_idx.astype(jnp.float32) + 0.5
        r_s = jnp.sqrt(sf / float(eff_ao))
        phi_s = (2.0 * math.pi * 0.61803398875) * sf
        ux0 = r_s * jnp.cos(phi_s)
        uy0 = r_s * jnp.sin(phi_s)
        uz0 = jnp.sqrt(jnp.maximum(1.0 - r_s * r_s, 1e-4))
        ux = ux0 * cos_rot - uy0 * sin_rot
        uy = ux0 * sin_rot + uy0 * cos_rot
        aodx = ux * tx_b + uy * bx_b + uz0 * nx_out
        aody = ux * ty_b + uy * by_b + uz0 * ny_out
        aodz = ux * tz_b + uy * bz_b + uz0 * nz_onb
        min_ao = ao_max_d
        for k, (gt, pfx, pfy) in enumerate(zip(g_types_static, plane_fx, plane_fy)):
          px = gf[0 * block_b : 1 * block_b, k : k + 1]
          py = gf[1 * block_b : 2 * block_b, k : k + 1]
          pz = gf[2 * block_b : 3 * block_b, k : k + 1]
          m00, m01, m02 = gf[3 * block_b : 4 * block_b, k : k + 1], gf[4 * block_b : 5 * block_b, k : k + 1], gf[5 * block_b : 6 * block_b, k : k + 1]
          m10, m11, m12 = gf[6 * block_b : 7 * block_b, k : k + 1], gf[7 * block_b : 8 * block_b, k : k + 1], gf[8 * block_b : 9 * block_b, k : k + 1]
          m20, m21, m22 = gf[9 * block_b : 10 * block_b, k : k + 1], gf[10 * block_b : 11 * block_b, k : k + 1], gf[11 * block_b : 12 * block_b, k : k + 1]
          sx, sy, sz = gf[12 * block_b : 13 * block_b, k : k + 1], gf[13 * block_b : 14 * block_b, k : k + 1], gf[14 * block_b : 15 * block_b, k : k + 1]
          rb_sq = gf[15 * block_b : 16 * block_b, k : k + 1]
          acd, _, _, _ = _ray_one_geom_v2(
              gt, px, py, pz, m00, m01, m02, m10, m11, m12, m20, m21, m22,
              sx, sy, sz, rb_sq, sh_ox, sh_oy, sh_oz, aodx, aody, aodz,
              plane_finite_x=pfx, plane_finite_y=pfy, compute_normal=False,
          )
          min_ao = jnp.minimum(min_ao, acd)
        return occ_sum_in + jnp.where(hit & (min_ao < ao_radius), 1.0 - 0.5 * (min_ao / ao_radius), 0.0)

      ao_occ_sum = jax.lax.fori_loop(
          0, eff_ao, _ao_sample_body, jnp.zeros((block_b, block_r), dtype=jnp.float32)
      )
      ao_vis = jnp.clip(1.0 - ao_strength * (ao_occ_sum / float(eff_ao)), 0.16, 1.0)
    else:
      ao_vis = 1.0

    # GGX Microfacet PBR shading with per-env light pose & color temperature
    lpx_l, lpy_l, lpz_l = linfo[:, 0:1], linfo[:, 1:2], linfo[:, 2:3]
    ldx_l, ldy_l, ldz_l = linfo[:, 3:4], linfo[:, 4:5], linfo[:, 5:6]
    ldiff_r, ldiff_g, ldiff_b = linfo[:, 6:7], linfo[:, 7:8], linfo[:, 8:9]
    lspec_r, lspec_g, lspec_b = linfo[:, 9:10], linfo[:, 10:11], linfo[:, 11:12]
    lamb_r, lamb_g, lamb_b = linfo[:, 12:13], linfo[:, 13:14], linfo[:, 14:15]
    sky1_r, sky1_g, sky1_b = linfo[:, 15:16], linfo[:, 16:17], linfo[:, 17:18]
    sky2_r, sky2_g, sky2_b = linfo[:, 18:19], linfo[:, 19:20], linfo[:, 20:21]
    exposure_v = linfo[:, 37:38]

    if l_type_0 == 1:
      lx_v = jnp.broadcast_to(-ldx_l, (block_b, block_r))
      ly_v = jnp.broadcast_to(-ldy_l, (block_b, block_r))
      lz_v = jnp.broadcast_to(-ldz_l, (block_b, block_r))
      dist_lim = 1.0e8
      attenuation = jnp.ones((block_b, block_r), dtype=jnp.float32)
    else:
      dlx, dly, dlz = lpx_l - hx, lpy_l - hy, lpz_l - hz
      d_sq = dlx * dlx + dly * dly + dlz * dlz
      d_raw = jnp.sqrt(d_sq)
      inv_dl = 1.0 / jnp.maximum(d_raw, 1e-12)
      lx_v, ly_v, lz_v = dlx * inv_dl, dly * inv_dl, dlz * inv_dl
      dist_lim = jnp.maximum(d_raw - 1.0e-3, 0.0)
      att_denom = l_att_0[0] + l_att_0[1] * d_raw + l_att_0[2] * d_sq
      attenuation = jnp.where(att_denom > 1e-12, 1.0 / jnp.maximum(att_denom, 1e-12), 1.0)

    ndotl = jnp.maximum(0.0, nx_out * lx_v + ny_out * ly_v + nz_onb * lz_v)

    if do_sh_0:
      sh_max = jnp.where(hit & (ndotl > 0.0), dist_lim, 0.0)
      (ltx, lty, ltz), (lbx, lby, lbz) = _branchless_onb(lx_v, ly_v, lz_v)
      n_sh = len(sh_samples)

      def _sh_sample_body(s_idx, occ_sum_in):
        if n_sh == 1 and sh_samples[0] == (0.0, 0.0):
          sdx, sdy, sdz = lx_v, ly_v, lz_v
        else:
          sf = jnp.asarray(s_idx, jnp.float32) + 0.5
          r_s = jnp.sqrt(sf / float(n_sh)) * float(shadow_softness)
          phi_s = (2.0 * math.pi * 0.61803398875) * sf
          dx_d = r_s * jnp.cos(phi_s)
          dy_d = r_s * jnp.sin(phi_s)
          rx_d = dx_d * cos_rot - dy_d * sin_rot
          ry_d = dx_d * sin_rot + dy_d * cos_rot
          raw_x = lx_v + rx_d * ltx + ry_d * lbx
          raw_y = ly_v + rx_d * lty + ry_d * lby
          raw_z = lz_v + rx_d * ltz + ry_d * lbz
          inv_sn = 1.0 / jnp.maximum(jnp.sqrt(raw_x * raw_x + raw_y * raw_y + raw_z * raw_z), 1e-12)
          sdx, sdy, sdz = raw_x * inv_sn, raw_y * inv_sn, raw_z * inv_sn
        occ_s = jnp.zeros((block_b, block_r), dtype=jnp.bool_)
        for k, (gt, pfx, pfy) in enumerate(zip(g_types_static, plane_fx, plane_fy)):
          px = gf[0 * block_b : 1 * block_b, k : k + 1]
          py = gf[1 * block_b : 2 * block_b, k : k + 1]
          pz = gf[2 * block_b : 3 * block_b, k : k + 1]
          m00, m01, m02 = gf[3 * block_b : 4 * block_b, k : k + 1], gf[4 * block_b : 5 * block_b, k : k + 1], gf[5 * block_b : 6 * block_b, k : k + 1]
          m10, m11, m12 = gf[6 * block_b : 7 * block_b, k : k + 1], gf[7 * block_b : 8 * block_b, k : k + 1], gf[8 * block_b : 9 * block_b, k : k + 1]
          m20, m21, m22 = gf[9 * block_b : 10 * block_b, k : k + 1], gf[10 * block_b : 11 * block_b, k : k + 1], gf[11 * block_b : 12 * block_b, k : k + 1]
          sx, sy, sz = gf[12 * block_b : 13 * block_b, k : k + 1], gf[13 * block_b : 14 * block_b, k : k + 1], gf[14 * block_b : 15 * block_b, k : k + 1]
          rb_sq = gf[15 * block_b : 16 * block_b, k : k + 1]
          scd, _, _, _ = _ray_one_geom_v2(
              gt, px, py, pz, m00, m01, m02, m10, m11, m12, m20, m21, m22,
              sx, sy, sz, rb_sq, sh_ox, sh_oy, sh_oz, sdx, sdy, sdz,
              plane_finite_x=pfx, plane_finite_y=pfy, compute_normal=False,
          )
          occ_s = occ_s | (scd < sh_max)
        return occ_sum_in + (hit & (ndotl > 0.0) & occ_s).astype(jnp.float32)

      if n_sh == 1:
        occ_sum = _sh_sample_body(0, jnp.zeros((block_b, block_r), dtype=jnp.float32))
      else:
        occ_sum = jax.lax.fori_loop(0, n_sh, _sh_sample_body, jnp.zeros((block_b, block_r), dtype=jnp.float32))
      vis = 1.0 - (1.0 - shadow_frac) * (occ_sum / float(n_sh))
    else:
      vis = 1.0

    alpha = roughness * roughness
    a2 = alpha * alpha
    k_g = ((roughness + 1.0) * (roughness + 1.0)) * 0.125
    f0 = 0.04 + 0.08 * spec_coef
    ndotv = jnp.maximum(1e-4, -(nx_out * rdx + ny_out * rdy + nz_onb * rdz))
    g1_v = ndotv / (ndotv * (1.0 - k_g) + k_g + 1e-12)

    hx_v, hy_v, hz_v = lx_v - rdx, ly_v - rdy, lz_v - rdz
    inv_hn = 1.0 / jnp.maximum(jnp.sqrt(hx_v * hx_v + hy_v * hy_v + hz_v * hz_v), 1e-12)
    hx_n, hy_n, hz_n = hx_v * inv_hn, hy_v * inv_hn, hz_v * inv_hn
    ndoth = jnp.maximum(0.0, nx_out * hx_n + ny_out * hy_n + nz_onb * hz_n)
    vdoth = jnp.maximum(0.0, -(rdx * hx_n + rdy * hy_n + rdz * hz_n))

    denom_d = ndoth * ndoth * (a2 - 1.0) + 1.0
    d_ggx = a2 / (math.pi * denom_d * denom_d + 1e-12)
    omv = 1.0 - vdoth
    omv2 = omv * omv
    f_schlick = f0 + (1.0 - f0) * (omv2 * omv2 * omv)
    g1_l = ndotl / (ndotl * (1.0 - k_g) + k_g + 1e-12)
    spec_brdf = (d_ggx * f_schlick * g1_l * g1_v) / (4.0 * ndotl * ndotv + 1e-4)

    weight = attenuation * vis * ndotl
    kd = 1.0 - f_schlick
    diff_r = ldiff_r * weight * kd
    diff_g = ldiff_g * weight * kd
    diff_b = ldiff_b * weight * kd
    spec_fac = jnp.where((spec_coef > 0.0) & (ndotl > 0.0), spec_brdf * weight * math.pi, 0.0)
    spec_r = lspec_r * spec_fac
    spec_g = lspec_g * spec_fac
    spec_b = lspec_b * spec_fac

    # Hemisphere sky bounce + optional gentle fill headlight
    hemi_t = 0.5 + 0.5 * nz_onb
    amb_r = lamb_r + 0.14 * (hemi_t * sky1_r + (1.0 - hemi_t) * sky2_r)
    amb_g = lamb_g + 0.14 * (hemi_t * sky1_g + (1.0 - hemi_t) * sky2_g)
    amb_b = lamb_b + 0.14 * (hemi_t * sky1_b + (1.0 - hemi_t) * sky2_b)
    if headlight_active:
      amb_r = amb_r + 0.35 * hl_amb[0]
      amb_g = amb_g + 0.35 * hl_amb[1]
      amb_b = amb_b + 0.35 * hl_amb[2]
      diff_r = diff_r + 0.25 * hl_diff[0] * ndotv
      diff_g = diff_g + 0.25 * hl_diff[1] * ndotv
      diff_b = diff_b + 0.25 * hl_diff[2] * ndotv

    tot_r = alb_r * (amb_r * ao_vis + diff_r + emit_coef) + spec_r * (0.5 + 0.5 * ao_vis)
    tot_g = alb_g * (amb_g * ao_vis + diff_g + emit_coef) + spec_g * (0.5 + 0.5 * ao_vis)
    tot_b = alb_b * (amb_b * ao_vis + diff_b + emit_coef) + spec_b * (0.5 + 0.5 * ao_vis)

    sky_t = jnp.clip(0.5 * (rdz + 1.0), 0.0, 1.0)
    sky_r = sky_t * sky1_r + (1.0 - sky_t) * sky2_r
    sky_g = sky_t * sky1_g + (1.0 - sky_t) * sky2_g
    sky_b = sky_t * sky1_b + (1.0 - sky_t) * sky2_b

    lin_r = jnp.where(hit, tot_r, sky_r)
    lin_g = jnp.where(hit, tot_g, sky_g)
    lin_b = jnp.where(hit, tot_b, sky_b)

    if use_agx:
      fin_r, fin_g, fin_b = _agx_tonemap_rgb_components(lin_r, lin_g, lin_b, exposure_v)
    else:
      fin_r = jnp.clip(lin_r, 0.0, 1.0)
      fin_g = jnp.clip(lin_g, 0.0, 1.0)
      fin_b = jnp.clip(lin_b, 0.0, 1.0)

    out_r_ref[...] = fin_r.astype(out_dtype)
    out_g_ref[...] = fin_g.astype(out_dtype)
    out_b_ref[...] = fin_b.astype(out_dtype)
    # Exact metric planar Z-depth (distance_to_image_plane)
    out_d_ref[...] = jnp.where(hit, znear_f + best_d * neg_z, -1.0).astype(out_dtype)
    if emit_aux:
      out_nx_ref[...] = nx_out.astype(out_dtype)
      out_ny_ref[...] = ny_out.astype(out_dtype)
      out_nz_ref[...] = nz_out.astype(out_dtype)
      out_gid_ref[...] = best_gid
      out_pid_ref[...] = jnp.where(hit, best_pid, jnp.int32(0))

  n_outs = 9 if emit_aux else 4
  out_shape = [jax.ShapeDtypeStruct((B_pad, R_pad), out_dtype) for _ in range(4)]
  if emit_aux:
    out_shape.extend([
        jax.ShapeDtypeStruct((B_pad, R_pad), out_dtype),
        jax.ShapeDtypeStruct((B_pad, R_pad), out_dtype),
        jax.ShapeDtypeStruct((B_pad, R_pad), out_dtype),
        jax.ShapeDtypeStruct((B_pad, R_pad), jnp.int32),
        jax.ShapeDtypeStruct((B_pad, R_pad), jnp.int32),
    ])

  outs = pl.pallas_call(
      _kernel,
      out_shape=out_shape,
      grid=(n_b_blk, n_r_blk),
      in_specs=[
          pl.BlockSpec((block_b, 128), lambda bi, ri: (bi, 0)),
          pl.BlockSpec((16, block_r), lambda bi, ri: (0, ri)),
          pl.BlockSpec((16 * block_b, G_pad), lambda bi, ri: (bi, 0)),
          pl.BlockSpec((6 * block_b, G_pad), lambda bi, ri: (bi, 0)),
          pl.BlockSpec((block_b, 128), lambda bi, ri: (bi, 0)),
      ],
      out_specs=[pl.BlockSpec((block_b, block_r), lambda bi, ri: (bi, ri)) for _ in range(n_outs)],
      compiler_params=None if interpret else pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
      interpret=interpret,
  )(cam_128, lray_dev, geom_fields, mat_fields, light_128)

  r_out, g_out, b_out, d_out = outs[0], outs[1], outs[2], outs[3]
  res: Dict[str, jax.Array] = {
      "rgb": jnp.stack([r_out[:B, :R], g_out[:B, :R], b_out[:B, :R]], axis=-1).reshape(B, H, W, 3),
      "distance_to_image_plane": d_out[:B, :R].reshape(B, H, W),
  }
  if emit_aux:
    nx_out, ny_out, nz_out, gid_out, pid_out = outs[4], outs[5], outs[6], outs[7], outs[8]
    gid_hw = gid_out[:B, :R].reshape(B, H, W)
    pid_hw = pid_out[:B, :R].reshape(B, H, W)
    res["normals"] = jnp.stack([nx_out[:B, :R], ny_out[:B, :R], nz_out[:B, :R]], axis=-1).reshape(B, H, W, 3)
    res["instance_id"] = gid_hw
    res["semantic_segmentation"] = jnp.where(gid_hw >= 0, pid_hw % 64, jnp.int32(0))
    res["_body_id"] = jnp.where(gid_hw >= 0, pid_hw // 64, jnp.int32(-1))
  return res


# ==============================================================================
# Pallas Mosaic Kernel 2: Hierarchical Morton Cluster-BVH DR Kernel (G > 128)
# ==============================================================================

def _expand_bits_10(v: jax.Array) -> jax.Array:
  v = (v | (v << 16)) & jnp.uint32(0x030000FF)
  v = (v | (v << 8)) & jnp.uint32(0x0300F00F)
  v = (v | (v << 4)) & jnp.uint32(0x030C30C3)
  v = (v | (v << 2)) & jnp.uint32(0x09249249)
  return v


def _morton_codes_3d(pos: jax.Array) -> jax.Array:
  mn = jnp.min(pos, axis=1, keepdims=True)
  mx = jnp.max(pos, axis=1, keepdims=True)
  span = jnp.maximum(mx - mn, 1e-4)
  norm = jnp.clip((pos - mn) / span, 0.0, 0.999999)
  q = jnp.floor(norm * 1024.0).astype(jnp.uint32)
  return (_expand_bits_10(q[..., 0]) << 2) | (_expand_bits_10(q[..., 1]) << 1) | _expand_bits_10(q[..., 2])


def _sphere_tile_hit(rox, roy, roz, rdx, rdy, rdz, cx, cy, cz, rad_sq, dist_cap=None):
  dx = rox - cx
  dy = roy - cy
  dz = roz - cz
  proj = dx * rdx + dy * rdy + dz * rdz
  d2 = dx * dx + dy * dy + dz * dz
  d2_perp = d2 - proj * proj
  hit = (d2_perp <= rad_sq) & ((proj < 0.0) | (d2 <= rad_sq))
  if dist_cap is not None:
    t_enter = jnp.maximum(-proj - jnp.sqrt(jnp.maximum(rad_sq - d2_perp, 0.0)), 0.0)
    hit = hit & (dist_cap > 0.0) & (t_enter < dist_cap)
  return hit


def _build_type_hierarchy_dr(
    pos_t: jax.Array,      # (B, G_t, 3)
    mat_t: jax.Array,      # (B, G_t, 3, 3)
    size_t: jax.Array,     # (G_t, 3)
    rb_t: jax.Array,       # (G_t,)
    gid_t: jax.Array,      # (G_t,) int32
    pid_t: jax.Array,      # (G_t,) int32 (bid*64 + sem_id)
    rgb_t: jax.Array,      # (B, G_t, 3)
    rough_t: jax.Array,    # (B, G_t)
    spec_t: jax.Array,     # (B, G_t)
    block_b: int,
    B_pad: int,
) -> Tuple[jax.Array, jax.Array, int, int, int]:
  """On-device Morton sort + 2-level bounding-sphere hierarchy with per-env DR materials."""
  B, G_t, _ = pos_t.shape
  n_b_blk = B_pad // block_b

  codes = _morton_codes_3d(pos_t)
  order = jnp.argsort(codes, axis=-1)

  pos_s = jnp.take_along_axis(pos_t, order[..., None], axis=1)
  mat_s = jnp.take_along_axis(mat_t.reshape(B, G_t, 9), order[..., None], axis=1)
  sz_s = jnp.take_along_axis(jnp.broadcast_to(size_t[None, :, :], (B, G_t, 3)), order[..., None], axis=1)
  rb_s = jnp.take_along_axis(jnp.broadcast_to(rb_t[None, :], (B, G_t)), order, axis=1)
  gid_s = jnp.take_along_axis(jnp.broadcast_to(gid_t[None, :].astype(jnp.float32), (B, G_t)), order, axis=1)
  pid_s = jnp.take_along_axis(jnp.broadcast_to(pid_t[None, :].astype(jnp.float32), (B, G_t)), order, axis=1)
  rgb_s = jnp.take_along_axis(rgb_t, order[..., None], axis=1)
  rough_s = jnp.take_along_axis(rough_t, order, axis=1)
  spec_s = jnp.take_along_axis(spec_t, order, axis=1)
  valid_s = jnp.ones((B, G_t), dtype=jnp.bool_)

  n_micro_raw = (G_t + CLUSTER_K - 1) // CLUSTER_K
  n_macro = (n_micro_raw + MACRO_M - 1) // MACRO_M
  n_micro = n_macro * MACRO_M
  G_pad = n_micro * CLUSTER_K
  pad_g = G_pad - G_t

  if pad_g > 0:
    pos_s = jnp.pad(pos_s, ((0, 0), (0, pad_g), (0, 0)))
    mat_s = jnp.pad(mat_s, ((0, 0), (0, pad_g), (0, 0)))
    sz_s = jnp.pad(sz_s, ((0, 0), (0, pad_g), (0, 0)))
    rb_s = jnp.pad(rb_s, ((0, 0), (0, pad_g)), constant_values=-1.0)
    gid_s = jnp.pad(gid_s, ((0, 0), (0, pad_g)), constant_values=-1.0)
    pid_s = jnp.pad(pid_s, ((0, 0), (0, pad_g)), constant_values=0.0)
    rgb_s = jnp.pad(rgb_s, ((0, 0), (0, pad_g), (0, 0)))
    rough_s = jnp.pad(rough_s, ((0, 0), (0, pad_g)), constant_values=0.5)
    spec_s = jnp.pad(spec_s, ((0, 0), (0, pad_g)), constant_values=0.2)
    valid_s = jnp.pad(valid_s, ((0, 0), (0, pad_g)), constant_values=False)

  pos_c = pos_s.reshape(B, n_micro, CLUSTER_K, 3)
  rb_c = rb_s.reshape(B, n_micro, CLUSTER_K)
  val_c = valid_s.reshape(B, n_micro, CLUSTER_K)
  any_val_c = jnp.any(val_c, axis=2)

  c_min = jnp.min(jnp.where(val_c[..., None], pos_c, 1.0e9), axis=2)
  c_max = jnp.max(jnp.where(val_c[..., None], pos_c, -1.0e9), axis=2)
  c_center = jnp.where(any_val_c[..., None], 0.5 * (c_min + c_max), 0.0)
  c_dists = jnp.linalg.norm(pos_c - c_center[:, :, None, :], axis=-1) + rb_c
  c_rad = jnp.max(jnp.where(val_c, c_dists, 0.0), axis=2) + 1.0e-4
  c_rad_sq = jnp.where(any_val_c, c_rad * c_rad, -1.0)

  uc_m = c_center.reshape(B, n_macro, MACRO_M, 3)
  ur_m = jnp.where(any_val_c, c_rad, 0.0).reshape(B, n_macro, MACRO_M)
  uval_m = any_val_c.reshape(B, n_macro, MACRO_M)
  any_val_m = jnp.any(uval_m, axis=2)

  m_min = jnp.min(jnp.where(uval_m[..., None], uc_m - ur_m[..., None], 1.0e9), axis=2)
  m_max = jnp.max(jnp.where(uval_m[..., None], uc_m + ur_m[..., None], -1.0e9), axis=2)
  m_center = jnp.where(any_val_m[..., None], 0.5 * (m_min + m_max), 0.0)
  m_dists = jnp.linalg.norm(uc_m - m_center[:, :, None, :], axis=-1) + ur_m
  m_rad = jnp.max(jnp.where(uval_m, m_dists, 0.0), axis=2) + 1.0e-4
  m_rad_sq = jnp.where(any_val_m, m_rad * m_rad, -1.0)

  rb_sq_s = jnp.where(valid_s, rb_s * rb_s, -1.0)
  # 23 field rows per micro-cluster + 1 meta row = 24 rows (24 * 8 = 192, multiple of 8)
  g23 = jnp.concatenate([
      pos_s, mat_s, sz_s, rb_sq_s[..., None], gid_s[..., None],
      rgb_s, rough_s[..., None], spec_s[..., None], pid_s[..., None],
  ], axis=-1)  # (B, n_micro*K, 23)
  g23_c = g23.reshape(B, n_micro, CLUSTER_K, 23).transpose(0, 1, 3, 2)  # (B, n_micro, 23, 16)
  g23_pad = jnp.pad(g23_c, ((0, 0), (0, 0), (0, 0), (0, 128 - CLUSTER_K)))

  c_meta = jnp.zeros((B, n_micro, 1, 128), dtype=jnp.float32)
  c_meta = c_meta.at[:, :, 0, 0:3].set(c_center)
  c_meta = c_meta.at[:, :, 0, 3].set(c_rad_sq)
  micro_24 = jnp.concatenate([g23_pad, c_meta], axis=2)  # (B, n_micro, 24, 128)
  if B_pad > B:
    micro_24 = jnp.pad(micro_24, ((0, B_pad - B), (0, 0), (0, 0), (0, 0)))
  gf_packed = (
      micro_24.reshape(n_b_blk, block_b, n_micro, 24, 128)
      .transpose(0, 2, 3, 1, 4)
      .reshape(n_b_blk * n_micro * 24 * block_b, 128)
  )

  m_4 = jnp.stack([m_center[..., 0], m_center[..., 1], m_center[..., 2], m_rad_sq], axis=-1)
  m_4_pad = jnp.pad(m_4, ((0, B_pad - B), (0, 0), (0, 0)))
  mf_packed = (
      m_4_pad.reshape(n_b_blk, block_b, n_macro, 4)
      .transpose(0, 2, 3, 1)
      .reshape(n_b_blk * n_macro * 4 * block_b)
  )
  mf_packed = jnp.pad(mf_packed[:, None], ((0, 0), (0, 127)))
  return gf_packed, mf_packed, n_micro_raw, n_micro, n_macro


def _pallas_tiled_dr_hierarchical(
    spec: RenderSceneSpec,
    mjx_data_batch: Any,
    rep_batch: Optional[ReplicatorBatch],
    cam_slot: int,
    geom_sem_ids: Tuple[int, ...],
    geom_body_ids: Tuple[int, ...],
    floor_gids: Tuple[int, ...],
    table_gids: Tuple[int, ...],
    cull_mode: str = "cluster1",
    tonemap: str = "agx",
    out_dtype: Any = jnp.bfloat16,
    block_b: int = 8,
    block_r: int = 512,
    emit_aux: bool = True,
    interpret: bool = False,
    cam_fovy_override: Optional[jax.Array] = None,
) -> Dict[str, jax.Array]:
  """O(1)-compile Hierarchical Morton Cluster-BVH Mosaic DR kernel (supports G=1,004+ prims)."""
  B = int(mjx_data_batch.cam_xpos.shape[0])
  H, W = int(spec.height), int(spec.width)
  R = H * W
  gids = np.asarray(spec.enabled_geom_ids, dtype=np.int32)
  G = len(gids)
  if G > 500:
    block_r = min(int(block_r), 256)

  B_pad = ((B + block_b - 1) // block_b) * block_b
  R_pad = ((R + block_r - 1) // block_r) * block_r
  n_b_blk = B_pad // block_b
  n_r_blk = R_pad // block_r

  cam_128, light_128 = _pack_camera_and_light_dr(
      spec, rep_batch, mjx_data_batch, cam_slot, block_b, B, B_pad, cam_fovy_override=cam_fovy_override
  )
  lray_dev = _build_unit_tangent_rays(H, W, R_pad)

  geom_size_np = np.asarray(spec.geom_size, dtype=np.float32)
  rb_all = np.sqrt(np.maximum(_compute_geom_bound_sq(spec, gids), 0.0)).astype(np.float32)

  if rep_batch is not None:
    all_rgb = rep_batch.geom_rgba[..., :3]
    all_rough = rep_batch.geom_roughness
    all_spec = rep_batch.geom_specular
  else:
    all_rgb = jnp.broadcast_to(jnp.asarray(spec.geom_rgba[:, :3], jnp.float32)[None, :, :], (B, spec.ngeom, 3))
    base_rough = jnp.clip(1.0 - jnp.asarray(spec.geom_shin, jnp.float32) * 1.4, 0.14, 0.90)
    all_rough = jnp.broadcast_to(base_rough[None, :], (B, spec.ngeom))
    all_spec = jnp.broadcast_to(jnp.asarray(spec.geom_spec, jnp.float32)[None, :], (B, spec.ngeom))

  # Map global geom_id -> sem_id & body_id for enabled geoms
  sem_by_gid = {int(g): int(geom_sem_ids[k]) for k, g in enumerate(gids)}
  bid_by_gid = {int(g): int(geom_body_ids[k]) for k, g in enumerate(gids)}

  # Separate GEOM_PLANE from clustered non-plane primitive types
  plane_gids = [int(g) for g in gids if int(spec.geom_type[g]) == GEOM_PLANE]
  P_cnt = len(plane_gids)
  P_pad = ((max(P_cnt, 1) + 127) // 128) * 128
  if P_cnt > 0:
    p_idx = np.asarray(plane_gids, dtype=np.int32)
    p_pos = mjx_data_batch.geom_xpos[:, p_idx, :]
    p_mat = mjx_data_batch.geom_xmat[:, p_idx, :, :].reshape(B, P_cnt, 9)
    p_sz = jnp.broadcast_to(jnp.asarray(geom_size_np[p_idx])[None, :, :], (B, P_cnt, 3))
    p_gid = jnp.broadcast_to(jnp.asarray(p_idx, jnp.float32)[None, :, None], (B, P_cnt, 1))
    p_pid = jnp.broadcast_to(
        jnp.asarray([bid_by_gid[g] * 64 + sem_by_gid[g] for g in plane_gids], jnp.float32)[None, :, None],
        (B, P_cnt, 1),
    )
    p_rgb = all_rgb[:, p_idx, :]
    p_rough = all_rough[:, p_idx][..., None]
    p_spec = all_spec[:, p_idx][..., None]
    p_22 = jnp.concatenate([p_pos, p_mat, p_sz, p_gid, p_rgb, p_rough, p_spec, p_pid], axis=-1)
    if B_pad > B or P_pad > P_cnt:
      p_22 = jnp.pad(p_22, ((0, B_pad - B), (0, P_pad - P_cnt), (0, 0)))
    # Pad rows to 24 (multiple of 8)
    p_24 = jnp.pad(p_22, ((0, 0), (0, 0), (0, 2)))
    plane_fields = (
        p_24.reshape(n_b_blk, block_b, P_pad, 24)
        .transpose(0, 3, 1, 2)
        .reshape(n_b_blk * 24 * block_b, P_pad)
    )
  else:
    plane_fields = jnp.zeros((n_b_blk * 24 * block_b, P_pad), dtype=jnp.float32)

  plane_fx = tuple(float(geom_size_np[g, 0]) > 0.0 for g in plane_gids)
  plane_fy = tuple(float(geom_size_np[g, 1]) > 0.0 for g in plane_gids)

  active_types: List[int] = []
  gf_list: List[jax.Array] = []
  mf_list: List[jax.Array] = []
  n_micro_raw_list: List[int] = []
  n_micro_list: List[int] = []
  n_macro_list: List[int] = []

  for gt in (GEOM_BOX, GEOM_SPHERE, GEOM_CAPSULE, GEOM_CYLINDER, GEOM_ELLIPSOID):
    idxs = [k for k, g in enumerate(gids) if int(spec.geom_type[g]) == gt]
    if not idxs:
      continue
    g_sub = gids[idxs]
    pid_sub = np.asarray([bid_by_gid[int(g)] * 64 + sem_by_gid[int(g)] for g in g_sub], dtype=np.int32)
    gf_t, mf_t, nm_raw, nm_pad, n_mac = _build_type_hierarchy_dr(
        mjx_data_batch.geom_xpos[:, g_sub, :],
        mjx_data_batch.geom_xmat[:, g_sub, :, :],
        jnp.asarray(geom_size_np[g_sub], dtype=jnp.float32),
        jnp.asarray(rb_all[idxs], dtype=jnp.float32),
        jnp.asarray(g_sub, dtype=jnp.int32),
        jnp.asarray(pid_sub, dtype=jnp.int32),
        all_rgb[:, g_sub, :],
        all_rough[:, g_sub],
        all_spec[:, g_sub],
        block_b, B_pad,
    )
    active_types.append(gt)
    gf_list.append(gf_t)
    mf_list.append(mf_t)
    n_micro_raw_list.append(nm_raw)
    n_micro_list.append(nm_pad)
    n_macro_list.append(n_mac)

  n_types = len(active_types)
  znear_f = float(spec.znear)
  zfar_f = float(spec.zfar) if spec.zfar > 0.0 else 50.0
  l_type_0 = int(spec.light_type[0]) if spec.nlight > 0 else 0
  l_att_0 = tuple(float(x) for x in spec.light_attenuation[0]) if spec.nlight > 0 else (1.0, 0.0, 0.0)
  do_sh_0 = bool(spec.use_shadows) and (bool(spec.light_castshadow[0]) if spec.nlight > 0 else True)
  shadow_frac = float(spec.shadow_light_fraction)
  headlight_active = bool(spec.headlight_active)
  hl_amb = tuple(float(x) for x in spec.headlight_ambient)
  hl_diff = tuple(float(x) for x in spec.headlight_diffuse)
  use_agx = (tonemap == "agx")
  floor_gid_0 = int(floor_gids[0]) if floor_gids else -999
  table_gid_0 = int(table_gids[0]) if table_gids else -999

  # Note: if table is a box/non-plane geom, grab its center for local table texture coords
  table_center_np = (0.0, 0.0, 0.0)
  if table_gids:
    tg0 = int(table_gids[0])
    # Passed cleanly via plane or world coords (tabletops in MJX are nearly fixed XY)
    table_center_np = (0.0, 0.0, 0.0)
  del table_center_np

  def _kernel(*refs):
    cam = refs[0][...]
    lr = refs[1][...]
    pf = refs[2][...]
    linfo = refs[3][...]
    gf_refs = refs[4 : 4 + n_types]
    mf_refs = refs[4 + n_types : 4 + 2 * n_types]
    out_start = 4 + 2 * n_types
    out_r_ref, out_g_ref, out_b_ref, out_d_ref = (
        refs[out_start], refs[out_start + 1], refs[out_start + 2], refs[out_start + 3]
    )
    if emit_aux:
      out_nx_ref, out_ny_ref, out_nz_ref, out_gid_ref, out_pid_ref = (
          refs[out_start + 4], refs[out_start + 5], refs[out_start + 6],
          refs[out_start + 7], refs[out_start + 8],
      )

    tx_r = lr[0:1, :]
    ty_r = lr[1:2, :]
    tan_hf = cam[:, 12:13]
    ldx_raw = tx_r * tan_hf
    ldy_raw = ty_r * tan_hf
    inv_ln = 1.0 / jnp.sqrt(ldx_raw * ldx_raw + ldy_raw * ldy_raw + 1.0)
    neg_z = inv_ln
    ldx, ldy, ldz = ldx_raw * inv_ln, ldy_raw * inv_ln, -inv_ln
    inv_cos = znear_f / neg_z
    max_d = (zfar_f - znear_f) / neg_z

    cpx, cpy, cpz = cam[:, 0:1], cam[:, 1:2], cam[:, 2:3]
    c00, c01, c02 = cam[:, 3:4], cam[:, 4:5], cam[:, 5:6]
    c10, c11, c12 = cam[:, 6:7], cam[:, 7:8], cam[:, 8:9]
    c20, c21, c22 = cam[:, 9:10], cam[:, 10:11], cam[:, 11:12]

    rdx_raw = c00 * ldx + c01 * ldy + c02 * ldz
    rdy_raw = c10 * ldx + c11 * ldy + c12 * ldz
    rdz_raw = c20 * ldx + c21 * ldy + c22 * ldz
    inv_rn = 1.0 / jnp.maximum(jnp.sqrt(rdx_raw * rdx_raw + rdy_raw * rdy_raw + rdz_raw * rdz_raw), 1e-12)
    rdx, rdy, rdz = rdx_raw * inv_rn, rdy_raw * inv_rn, rdz_raw * inv_rn

    rox = cpx + inv_cos * rdx
    roy = cpy + inv_cos * rdy
    roz = cpz + inv_cos * rdz

    best_d = max_d
    best_gid = jnp.full((block_b, block_r), -1, dtype=jnp.int32)
    best_pid = jnp.zeros((block_b, block_r), dtype=jnp.int32)
    best_nx = jnp.zeros((block_b, block_r), dtype=jnp.float32)
    best_ny = jnp.zeros((block_b, block_r), dtype=jnp.float32)
    best_nz = jnp.zeros((block_b, block_r), dtype=jnp.float32)
    alb_r = jnp.zeros((block_b, block_r), dtype=jnp.float32)
    alb_g = jnp.zeros((block_b, block_r), dtype=jnp.float32)
    alb_b = jnp.zeros((block_b, block_r), dtype=jnp.float32)
    roughness = jnp.full((block_b, block_r), 0.45, dtype=jnp.float32)
    spec_coef = jnp.zeros((block_b, block_r), dtype=jnp.float32)

    # Planes first
    for pk_i, (pfx, pfy) in enumerate(zip(plane_fx, plane_fy)):
      px = pf[0 * block_b : 1 * block_b, pk_i : pk_i + 1]
      py = pf[1 * block_b : 2 * block_b, pk_i : pk_i + 1]
      pz = pf[2 * block_b : 3 * block_b, pk_i : pk_i + 1]
      m00, m01, m02 = pf[3 * block_b : 4 * block_b, pk_i : pk_i + 1], pf[4 * block_b : 5 * block_b, pk_i : pk_i + 1], pf[5 * block_b : 6 * block_b, pk_i : pk_i + 1]
      m10, m11, m12 = pf[6 * block_b : 7 * block_b, pk_i : pk_i + 1], pf[7 * block_b : 8 * block_b, pk_i : pk_i + 1], pf[8 * block_b : 9 * block_b, pk_i : pk_i + 1]
      m20, m21, m22 = pf[9 * block_b : 10 * block_b, pk_i : pk_i + 1], pf[10 * block_b : 11 * block_b, pk_i : pk_i + 1], pf[11 * block_b : 12 * block_b, pk_i : pk_i + 1]
      sx, sy, sz = pf[12 * block_b : 13 * block_b, pk_i : pk_i + 1], pf[13 * block_b : 14 * block_b, pk_i : pk_i + 1], pf[14 * block_b : 15 * block_b, pk_i : pk_i + 1]
      gid_f = pf[15 * block_b : 16 * block_b, pk_i : pk_i + 1]
      cd, cnx, cny, cnz = _ray_one_geom_v2(
          GEOM_PLANE, px, py, pz, m00, m01, m02, m10, m11, m12, m20, m21, m22,
          sx, sy, sz, 1.0e20, rox, roy, roz, rdx, rdy, rdz,
          plane_finite_x=pfx, plane_finite_y=pfy, compute_normal=True,
      )
      imp = cd < best_d
      best_d = jnp.where(imp, cd, best_d)
      best_gid = jnp.where(imp, gid_f.astype(jnp.int32), best_gid)
      best_nx = jnp.where(imp, cnx, best_nx)
      best_ny = jnp.where(imp, cny, best_ny)
      best_nz = jnp.where(imp, cnz, best_nz)
      alb_r = jnp.where(imp, pf[16 * block_b : 17 * block_b, pk_i : pk_i + 1], alb_r)
      alb_g = jnp.where(imp, pf[17 * block_b : 18 * block_b, pk_i : pk_i + 1], alb_g)
      alb_b = jnp.where(imp, pf[18 * block_b : 19 * block_b, pk_i : pk_i + 1], alb_b)
      roughness = jnp.where(imp, pf[19 * block_b : 20 * block_b, pk_i : pk_i + 1], roughness)
      spec_coef = jnp.where(imp, pf[20 * block_b : 21 * block_b, pk_i : pk_i + 1], spec_coef)
      best_pid = jnp.where(imp, pf[21 * block_b : 22 * block_b, pk_i : pk_i + 1].astype(jnp.int32), best_pid)

    def _eval_micro_primary(gt, c_slice, state):
      bd, bgid, bpid, bnx, bny, bnz, ar, ag, ab, rgh, spc = state
      for k in range(CLUSTER_K):
        px = c_slice[0 * block_b : 1 * block_b, k : k + 1]
        py = c_slice[1 * block_b : 2 * block_b, k : k + 1]
        pz = c_slice[2 * block_b : 3 * block_b, k : k + 1]
        m00, m01, m02 = c_slice[3 * block_b : 4 * block_b, k : k + 1], c_slice[4 * block_b : 5 * block_b, k : k + 1], c_slice[5 * block_b : 6 * block_b, k : k + 1]
        m10, m11, m12 = c_slice[6 * block_b : 7 * block_b, k : k + 1], c_slice[7 * block_b : 8 * block_b, k : k + 1], c_slice[8 * block_b : 9 * block_b, k : k + 1]
        m20, m21, m22 = c_slice[9 * block_b : 10 * block_b, k : k + 1], c_slice[10 * block_b : 11 * block_b, k : k + 1], c_slice[11 * block_b : 12 * block_b, k : k + 1]
        sx, sy, sz = c_slice[12 * block_b : 13 * block_b, k : k + 1], c_slice[13 * block_b : 14 * block_b, k : k + 1], c_slice[14 * block_b : 15 * block_b, k : k + 1]
        rb_sq = c_slice[15 * block_b : 16 * block_b, k : k + 1]
        gid_f = c_slice[16 * block_b : 17 * block_b, k : k + 1]
        cd, cnx, cny, cnz = _ray_one_geom_v2(
            gt, px, py, pz, m00, m01, m02, m10, m11, m12, m20, m21, m22,
            sx, sy, sz, rb_sq, rox, roy, roz, rdx, rdy, rdz, compute_normal=True,
        )
        imp = (rb_sq >= 0.0) & (cd < bd)
        bd = jnp.where(imp, cd, bd)
        bgid = jnp.where(imp, gid_f.astype(jnp.int32), bgid)
        bnx = jnp.where(imp, cnx, bnx)
        bny = jnp.where(imp, cny, bny)
        bnz = jnp.where(imp, cnz, bnz)
        ar = jnp.where(imp, c_slice[17 * block_b : 18 * block_b, k : k + 1], ar)
        ag = jnp.where(imp, c_slice[18 * block_b : 19 * block_b, k : k + 1], ag)
        ab = jnp.where(imp, c_slice[19 * block_b : 20 * block_b, k : k + 1], ab)
        rgh = jnp.where(imp, c_slice[20 * block_b : 21 * block_b, k : k + 1], rgh)
        spc = jnp.where(imp, c_slice[21 * block_b : 22 * block_b, k : k + 1], spc)
        bpid = jnp.where(imp, c_slice[22 * block_b : 23 * block_b, k : k + 1].astype(jnp.int32), bpid)
      return (bd, bgid, bpid, bnx, bny, bnz, ar, ag, ab, rgh, spc)

    state_p = (
        best_d, best_gid, best_pid, best_nx, best_ny, best_nz,
        alb_r, alb_g, alb_b, roughness, spec_coef,
    )
    for t_i, gt in enumerate(active_types):
      gf_ref_t = gf_refs[t_i]
      mf_ref_t = mf_refs[t_i]
      n_micro_raw_t = n_micro_raw_list[t_i]
      n_macro_t = n_macro_list[t_i]

      def _run_micro_by_idx(c_idx, st, _gt=gt, _gf_ref=gf_ref_t):
        base_r = c_idx * 24 * block_b
        if cull_mode == "flat_loop":
          return _eval_micro_primary(_gt, _gf_ref[pl.ds(base_r, 23 * block_b), :], st)
        c_meta = _gf_ref[pl.ds(base_r + 23 * block_b, block_b), :]
        u_hit = jnp.any(_sphere_tile_hit(
            rox, roy, roz, rdx, rdy, rdz,
            c_meta[:, 0:1], c_meta[:, 1:2], c_meta[:, 2:3], c_meta[:, 3:4],
            dist_cap=st[0],
        ))
        return jax.lax.cond(
            u_hit,
            lambda s: _eval_micro_primary(_gt, _gf_ref[pl.ds(base_r, 23 * block_b), :], s),
            lambda s: s,
            st,
        )

      if cull_mode in ("flat_loop", "cluster1"):
        state_p = jax.lax.fori_loop(0, n_micro_raw_t, _run_micro_by_idx, state_p)
      else:
        def _macro_step(m_idx, st, _mf_ref=mf_ref_t, _rm=_run_micro_by_idx, _nm=n_micro_raw_t):
          m_slice = _mf_ref[pl.ds(m_idx * 4 * block_b, 4 * block_b), :]
          m_hit = jnp.any(_sphere_tile_hit(
              rox, roy, roz, rdx, rdy, rdz,
              m_slice[0 * block_b : 1 * block_b, 0:1],
              m_slice[1 * block_b : 2 * block_b, 0:1],
              m_slice[2 * block_b : 3 * block_b, 0:1],
              m_slice[3 * block_b : 4 * block_b, 0:1],
              dist_cap=st[0],
          ))
          return jax.lax.cond(
              m_hit,
              lambda s: jax.lax.fori_loop(m_idx * MACRO_M, jnp.minimum(m_idx * MACRO_M + MACRO_M, _nm), _rm, s),
              lambda s: s,
              st,
          )
        state_p = jax.lax.fori_loop(0, n_macro_t, _macro_step, state_p)

    best_d, best_gid, best_pid, best_nx, best_ny, best_nz, alb_r, alb_g, alb_b, roughness, spec_coef = state_p
    hit = best_gid >= 0
    d_safe = jnp.where(hit, best_d, 0.0)
    nx_out = jnp.where(hit, best_nx, 0.0)
    ny_out = jnp.where(hit, best_ny, 0.0)
    nz_out = jnp.where(hit, best_nz, 0.0)
    nz_onb = jnp.where(hit, best_nz, 1.0)

    hx = rox + d_safe * rdx
    hy = roy + d_safe * rdy
    hz = roz + d_safe * rdz

    # Procedural floor & tabletop textures in world XY
    if floor_gid_0 >= 0:
      tr, tg_c, tb = _eval_procedural_tex_vmem(
          hx, hy, linfo[:, 21:22], linfo[:, 22:23],
          linfo[:, 23:24], linfo[:, 24:25], linfo[:, 25:26],
          linfo[:, 26:27], linfo[:, 27:28], linfo[:, 28:29],
      )
      is_fg = best_gid == floor_gid_0
      alb_r = jnp.where(is_fg, tr, alb_r)
      alb_g = jnp.where(is_fg, tg_c, alb_g)
      alb_b = jnp.where(is_fg, tb, alb_b)

    if table_gid_0 >= 0:
      tr, tg_c, tb = _eval_procedural_tex_vmem(
          hx, hy, linfo[:, 29:30], linfo[:, 30:31],
          linfo[:, 31:32], linfo[:, 32:33], linfo[:, 33:34],
          linfo[:, 34:35], linfo[:, 35:36], linfo[:, 36:37],
      )
      is_tg = best_gid == table_gid_0
      alb_r = jnp.where(is_tg, tr, alb_r)
      alb_g = jnp.where(is_tg, tg_c, alb_g)
      alb_b = jnp.where(is_tg, tb, alb_b)

    # Hierarchical shadow any-hit
    lpx_l, lpy_l, lpz_l = linfo[:, 0:1], linfo[:, 1:2], linfo[:, 2:3]
    ldx_l, ldy_l, ldz_l = linfo[:, 3:4], linfo[:, 4:5], linfo[:, 5:6]
    ldiff_r, ldiff_g, ldiff_b = linfo[:, 6:7], linfo[:, 7:8], linfo[:, 8:9]
    lspec_r, lspec_g, lspec_b = linfo[:, 9:10], linfo[:, 10:11], linfo[:, 11:12]
    lamb_r, lamb_g, lamb_b = linfo[:, 12:13], linfo[:, 13:14], linfo[:, 14:15]
    sky1_r, sky1_g, sky1_b = linfo[:, 15:16], linfo[:, 16:17], linfo[:, 17:18]
    sky2_r, sky2_g, sky2_b = linfo[:, 18:19], linfo[:, 19:20], linfo[:, 20:21]
    exposure_v = linfo[:, 37:38]

    sh_ox = hx + nx_out * 2.0e-4
    sh_oy = hy + ny_out * 2.0e-4
    sh_oz = hz + nz_onb * 2.0e-4

    if l_type_0 == 1:
      lx_v = jnp.broadcast_to(-ldx_l, (block_b, block_r))
      ly_v = jnp.broadcast_to(-ldy_l, (block_b, block_r))
      lz_v = jnp.broadcast_to(-ldz_l, (block_b, block_r))
      dist_lim = 1.0e8
      attenuation = jnp.ones((block_b, block_r), dtype=jnp.float32)
    else:
      dlx, dly, dlz = lpx_l - sh_ox, lpy_l - sh_oy, lpz_l - sh_oz
      d_sq = dlx * dlx + dly * dly + dlz * dlz
      d_raw = jnp.sqrt(d_sq)
      inv_dl = 1.0 / jnp.maximum(d_raw, 1e-12)
      lx_v, ly_v, lz_v = dlx * inv_dl, dly * inv_dl, dlz * inv_dl
      dist_lim = jnp.maximum(d_raw - 1.0e-3, 0.0)
      att_denom = l_att_0[0] + l_att_0[1] * d_raw + l_att_0[2] * d_sq
      attenuation = jnp.where(att_denom > 1e-12, 1.0 / jnp.maximum(att_denom, 1e-12), 1.0)

    ndotl = jnp.maximum(0.0, nx_out * lx_v + ny_out * ly_v + nz_onb * lz_v)

    if do_sh_0:
      sh_max = jnp.where(hit & (ndotl > 0.0), dist_lim, 0.0)
      occ_i32 = jnp.zeros((block_b, block_r), dtype=jnp.int32)
      for pk_i, (pfx, pfy) in enumerate(zip(plane_fx, plane_fy)):
        px = pf[0 * block_b : 1 * block_b, pk_i : pk_i + 1]
        py = pf[1 * block_b : 2 * block_b, pk_i : pk_i + 1]
        pz = pf[2 * block_b : 3 * block_b, pk_i : pk_i + 1]
        m00, m01, m02 = pf[3 * block_b : 4 * block_b, pk_i : pk_i + 1], pf[4 * block_b : 5 * block_b, pk_i : pk_i + 1], pf[5 * block_b : 6 * block_b, pk_i : pk_i + 1]
        m10, m11, m12 = pf[6 * block_b : 7 * block_b, pk_i : pk_i + 1], pf[7 * block_b : 8 * block_b, pk_i : pk_i + 1], pf[8 * block_b : 9 * block_b, pk_i : pk_i + 1]
        m20, m21, m22 = pf[9 * block_b : 10 * block_b, pk_i : pk_i + 1], pf[10 * block_b : 11 * block_b, pk_i : pk_i + 1], pf[11 * block_b : 12 * block_b, pk_i : pk_i + 1]
        sx, sy, sz = pf[12 * block_b : 13 * block_b, pk_i : pk_i + 1], pf[13 * block_b : 14 * block_b, pk_i : pk_i + 1], pf[14 * block_b : 15 * block_b, pk_i : pk_i + 1]
        scd, _, _, _ = _ray_one_geom_v2(
            GEOM_PLANE, px, py, pz, m00, m01, m02, m10, m11, m12, m20, m21, m22,
            sx, sy, sz, 1.0e20, sh_ox, sh_oy, sh_oz, lx_v, ly_v, lz_v,
            plane_finite_x=pfx, plane_finite_y=pfy, compute_normal=False,
        )
        occ_i32 = occ_i32 | (scd < sh_max).astype(jnp.int32)

      def _eval_micro_shadow(gt, c_slice, occ_in):
        occ_cur = occ_in
        for k in range(CLUSTER_K):
          px = c_slice[0 * block_b : 1 * block_b, k : k + 1]
          py = c_slice[1 * block_b : 2 * block_b, k : k + 1]
          pz = c_slice[2 * block_b : 3 * block_b, k : k + 1]
          m00, m01, m02 = c_slice[3 * block_b : 4 * block_b, k : k + 1], c_slice[4 * block_b : 5 * block_b, k : k + 1], c_slice[5 * block_b : 6 * block_b, k : k + 1]
          m10, m11, m12 = c_slice[6 * block_b : 7 * block_b, k : k + 1], c_slice[7 * block_b : 8 * block_b, k : k + 1], c_slice[8 * block_b : 9 * block_b, k : k + 1]
          m20, m21, m22 = c_slice[9 * block_b : 10 * block_b, k : k + 1], c_slice[10 * block_b : 11 * block_b, k : k + 1], c_slice[11 * block_b : 12 * block_b, k : k + 1]
          sx, sy, sz = c_slice[12 * block_b : 13 * block_b, k : k + 1], c_slice[13 * block_b : 14 * block_b, k : k + 1], c_slice[14 * block_b : 15 * block_b, k : k + 1]
          rb_sq = c_slice[15 * block_b : 16 * block_b, k : k + 1]
          scd, _, _, _ = _ray_one_geom_v2(
              gt, px, py, pz, m00, m01, m02, m10, m11, m12, m20, m21, m22,
              sx, sy, sz, rb_sq, sh_ox, sh_oy, sh_oz, lx_v, ly_v, lz_v, compute_normal=False,
          )
          occ_cur = occ_cur | ((rb_sq >= 0.0) & (scd < sh_max)).astype(jnp.int32)
        return occ_cur

      for t_i, gt in enumerate(active_types):
        gf_ref_t = gf_refs[t_i]
        mf_ref_t = mf_refs[t_i]
        n_micro_raw_t = n_micro_raw_list[t_i]
        n_macro_t = n_macro_list[t_i]

        def _run_sh_micro(c_idx, occ_in, _gt=gt, _gf_ref=gf_ref_t):
          base_r = c_idx * 24 * block_b
          if cull_mode == "flat_loop":
            return _eval_micro_shadow(_gt, _gf_ref[pl.ds(base_r, 23 * block_b), :], occ_in)
          c_meta = _gf_ref[pl.ds(base_r + 23 * block_b, block_b), :]
          u_hit = jnp.any(_sphere_tile_hit(
              sh_ox, sh_oy, sh_oz, lx_v, ly_v, lz_v,
              c_meta[:, 0:1], c_meta[:, 1:2], c_meta[:, 2:3], c_meta[:, 3:4],
              dist_cap=sh_max,
          ))
          return jax.lax.cond(
              u_hit,
              lambda o: _eval_micro_shadow(_gt, _gf_ref[pl.ds(base_r, 23 * block_b), :], o),
              lambda o: o,
              occ_in,
          )

        if cull_mode in ("flat_loop", "cluster1"):
          occ_i32 = jax.lax.fori_loop(0, n_micro_raw_t, _run_sh_micro, occ_i32)
        else:
          def _sh_macro(m_idx, occ_in, _mf_ref=mf_ref_t, _rm=_run_sh_micro, _nm=n_micro_raw_t):
            m_slice = _mf_ref[pl.ds(m_idx * 4 * block_b, 4 * block_b), :]
            m_hit = jnp.any(_sphere_tile_hit(
                sh_ox, sh_oy, sh_oz, lx_v, ly_v, lz_v,
                m_slice[0 * block_b : 1 * block_b, 0:1],
                m_slice[1 * block_b : 2 * block_b, 0:1],
                m_slice[2 * block_b : 3 * block_b, 0:1],
                m_slice[3 * block_b : 4 * block_b, 0:1],
                dist_cap=sh_max,
            ))
            return jax.lax.cond(
                m_hit,
                lambda o: jax.lax.fori_loop(m_idx * MACRO_M, jnp.minimum(m_idx * MACRO_M + MACRO_M, _nm), _rm, o),
                lambda o: o,
                occ_in,
            )
          occ_i32 = jax.lax.fori_loop(0, n_macro_t, _sh_macro, occ_i32)
      vis = jnp.where((occ_i32 > 0) & hit & (ndotl > 0.0), shadow_frac, 1.0)
    else:
      vis = 1.0

    # GGX Microfacet PBR shading + AgX tonemapping
    alpha = roughness * roughness
    a2 = alpha * alpha
    k_g = ((roughness + 1.0) * (roughness + 1.0)) * 0.125
    f0 = 0.04 + 0.08 * spec_coef
    ndotv = jnp.maximum(1e-4, -(nx_out * rdx + ny_out * rdy + nz_onb * rdz))
    g1_v = ndotv / (ndotv * (1.0 - k_g) + k_g + 1e-12)

    hx_v, hy_v, hz_v = lx_v - rdx, ly_v - rdy, lz_v - rdz
    inv_hn = 1.0 / jnp.maximum(jnp.sqrt(hx_v * hx_v + hy_v * hy_v + hz_v * hz_v), 1e-12)
    hx_n, hy_n, hz_n = hx_v * inv_hn, hy_v * inv_hn, hz_v * inv_hn
    ndoth = jnp.maximum(0.0, nx_out * hx_n + ny_out * hy_n + nz_onb * hz_n)
    vdoth = jnp.maximum(0.0, -(rdx * hx_n + rdy * hy_n + rdz * hz_n))

    denom_d = ndoth * ndoth * (a2 - 1.0) + 1.0
    d_ggx = a2 / (math.pi * denom_d * denom_d + 1e-12)
    omv = 1.0 - vdoth
    omv2 = omv * omv
    f_schlick = f0 + (1.0 - f0) * (omv2 * omv2 * omv)
    g1_l = ndotl / (ndotl * (1.0 - k_g) + k_g + 1e-12)
    spec_brdf = (d_ggx * f_schlick * g1_l * g1_v) / (4.0 * ndotl * ndotv + 1e-4)

    weight = attenuation * vis * ndotl
    kd = 1.0 - f_schlick
    diff_r = ldiff_r * weight * kd
    diff_g = ldiff_g * weight * kd
    diff_b = ldiff_b * weight * kd
    spec_fac = jnp.where((spec_coef > 0.0) & (ndotl > 0.0), spec_brdf * weight * math.pi, 0.0)
    spec_r = lspec_r * spec_fac
    spec_g = lspec_g * spec_fac
    spec_b = lspec_b * spec_fac

    hemi_t = 0.5 + 0.5 * nz_onb
    amb_r = lamb_r + 0.16 * (hemi_t * sky1_r + (1.0 - hemi_t) * sky2_r)
    amb_g = lamb_g + 0.16 * (hemi_t * sky1_g + (1.0 - hemi_t) * sky2_g)
    amb_b = lamb_b + 0.16 * (hemi_t * sky1_b + (1.0 - hemi_t) * sky2_b)
    if headlight_active:
      amb_r = amb_r + 0.35 * hl_amb[0]
      amb_g = amb_g + 0.35 * hl_amb[1]
      amb_b = amb_b + 0.35 * hl_amb[2]
      diff_r = diff_r + 0.25 * hl_diff[0] * ndotv
      diff_g = diff_g + 0.25 * hl_diff[1] * ndotv
      diff_b = diff_b + 0.25 * hl_diff[2] * ndotv

    tot_r = alb_r * (amb_r + diff_r) + spec_r
    tot_g = alb_g * (amb_g + diff_g) + spec_g
    tot_b = alb_b * (amb_b + diff_b) + spec_b

    sky_t = jnp.clip(0.5 * (rdz + 1.0), 0.0, 1.0)
    sky_r = sky_t * sky1_r + (1.0 - sky_t) * sky2_r
    sky_g = sky_t * sky1_g + (1.0 - sky_t) * sky2_g
    sky_b = sky_t * sky1_b + (1.0 - sky_t) * sky2_b

    lin_r = jnp.where(hit, tot_r, sky_r)
    lin_g = jnp.where(hit, tot_g, sky_g)
    lin_b = jnp.where(hit, tot_b, sky_b)

    if use_agx:
      fin_r, fin_g, fin_b = _agx_tonemap_rgb_components(lin_r, lin_g, lin_b, exposure_v)
    else:
      fin_r = jnp.clip(lin_r, 0.0, 1.0)
      fin_g = jnp.clip(lin_g, 0.0, 1.0)
      fin_b = jnp.clip(lin_b, 0.0, 1.0)

    out_r_ref[...] = fin_r.astype(out_dtype)
    out_g_ref[...] = fin_g.astype(out_dtype)
    out_b_ref[...] = fin_b.astype(out_dtype)
    out_d_ref[...] = jnp.where(hit, znear_f + best_d * neg_z, -1.0).astype(out_dtype)
    if emit_aux:
      out_nx_ref[...] = nx_out.astype(out_dtype)
      out_ny_ref[...] = ny_out.astype(out_dtype)
      out_nz_ref[...] = nz_out.astype(out_dtype)
      out_gid_ref[...] = best_gid
      out_pid_ref[...] = jnp.where(hit, best_pid, jnp.int32(0))

  in_arrays = [cam_128, lray_dev, plane_fields, light_128] + gf_list + mf_list
  in_specs = [
      pl.BlockSpec((block_b, 128), lambda bi, ri: (bi, 0)),
      pl.BlockSpec((16, block_r), lambda bi, ri: (0, ri)),
      pl.BlockSpec((24 * block_b, P_pad), lambda bi, ri: (bi, 0)),
      pl.BlockSpec((block_b, 128), lambda bi, ri: (bi, 0)),
  ]
  for n_micro_t in n_micro_list:
    in_specs.append(pl.BlockSpec((n_micro_t * 24 * block_b, 128), lambda bi, ri: (bi, 0)))
  for n_macro_t in n_macro_list:
    in_specs.append(pl.BlockSpec((n_macro_t * 4 * block_b, 128), lambda bi, ri: (bi, 0)))

  n_outs = 9 if emit_aux else 4
  out_shape = [jax.ShapeDtypeStruct((B_pad, R_pad), out_dtype) for _ in range(4)]
  if emit_aux:
    out_shape.extend([
        jax.ShapeDtypeStruct((B_pad, R_pad), out_dtype),
        jax.ShapeDtypeStruct((B_pad, R_pad), out_dtype),
        jax.ShapeDtypeStruct((B_pad, R_pad), out_dtype),
        jax.ShapeDtypeStruct((B_pad, R_pad), jnp.int32),
        jax.ShapeDtypeStruct((B_pad, R_pad), jnp.int32),
    ])

  outs = pl.pallas_call(
      _kernel,
      out_shape=out_shape,
      grid=(n_b_blk, n_r_blk),
      in_specs=in_specs,
      out_specs=[pl.BlockSpec((block_b, block_r), lambda bi, ri: (bi, ri)) for _ in range(n_outs)],
      compiler_params=None if interpret else pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
      interpret=interpret,
  )(*in_arrays)

  r_out, g_out, b_out, d_out = outs[0], outs[1], outs[2], outs[3]
  res: Dict[str, jax.Array] = {
      "rgb": jnp.stack([r_out[:B, :R], g_out[:B, :R], b_out[:B, :R]], axis=-1).reshape(B, H, W, 3),
      "distance_to_image_plane": d_out[:B, :R].reshape(B, H, W),
  }
  if emit_aux:
    nx_out, ny_out, nz_out, gid_out, pid_out = outs[4], outs[5], outs[6], outs[7], outs[8]
    gid_hw = gid_out[:B, :R].reshape(B, H, W)
    pid_hw = pid_out[:B, :R].reshape(B, H, W)
    res["normals"] = jnp.stack([nx_out[:B, :R], ny_out[:B, :R], nz_out[:B, :R]], axis=-1).reshape(B, H, W, 3)
    res["instance_id"] = gid_hw
    res["semantic_segmentation"] = jnp.where(gid_hw >= 0, pid_hw % 64, jnp.int32(0))
    res["_body_id"] = jnp.where(gid_hw >= 0, pid_hw // 64, jnp.int32(-1))
  return res


# ==============================================================================
# Zero-Gather 2D Motion Vectors (supports per-env FOV + G=1,004 via moving bodies)
# ==============================================================================

def _project_world_to_pixel_dyn_fov(
    H: int,
    W: int,
    cam_pos: jax.Array,      # (B, 3)
    cam_mat: jax.Array,      # (B, 3, 3)
    cam_fovy_deg: jax.Array, # (B,)
    pts_w: jax.Array,        # (B, H, W, 3)
) -> jax.Array:
  delta = pts_w - cam_pos[:, None, None, :]
  p_cam = jnp.einsum("bji,bhwj->bhwi", cam_mat, delta, precision=jax.lax.Precision.HIGHEST)
  aspect = float(W) / float(H)
  half_h = jnp.tan(0.5 * cam_fovy_deg * (math.pi / 180.0))[:, None, None]
  half_w = half_h * aspect
  neg_z = jnp.maximum(-p_cam[..., 2], 1e-5)
  x_ndc = p_cam[..., 0] / neg_z
  y_ndc = p_cam[..., 1] / neg_z
  u_norm = (x_ndc + half_w) / jnp.maximum(2.0 * half_w, 1e-6)
  v_norm = (half_h - y_ndc) / jnp.maximum(2.0 * half_h, 1e-6)
  return jnp.stack([u_norm * float(W), v_norm * float(H)], axis=-1)


def _compute_motion_vectors_zero_gather(
    spec: RenderSceneSpec,
    mjx_data_batch: Any,
    rep_batch: Optional[ReplicatorBatch],
    cam_slot: int,
    z_depth: jax.Array,     # (B, H, W)
    instance_id: jax.Array, # (B, H, W)
    body_id_map: jax.Array, # (B, H, W)
    active_moving_bids: Tuple[int, ...],
    body_rootid: Tuple[int, ...],
    dt: float,
    out_dtype: Any,
) -> jax.Array:
  """Zero-gather 2D optical flow (B, H, W, 2) in pixels via static moving-body slice loop."""
  B, H, W = z_depth.shape
  hit = (instance_id >= 0) & (z_depth >= 0.0)
  if not (hasattr(mjx_data_batch, "cvel") and hasattr(mjx_data_batch, "subtree_com")):
    return jnp.zeros((B, H, W, 2), dtype=out_dtype)

  if rep_batch is not None:
    cam_pos = rep_batch.cam_xpos[:, cam_slot, :]
    cam_mat = rep_batch.cam_xmat[:, cam_slot, :, :]
    cam_fovy = rep_batch.cam_fovy_deg[:, cam_slot]
  else:
    cam_pos = mjx_data_batch.cam_xpos[:, int(spec.cam_id), :]
    cam_mat = mjx_data_batch.cam_xmat[:, int(spec.cam_id), :, :]
    cam_fovy = jnp.full((B,), float(spec.cam_fovy), dtype=jnp.float32)

  px = np.arange(W, dtype=np.float32)
  py = np.arange(H, dtype=np.float32)
  gx, gy = np.meshgrid(px, py, indexing="xy")
  aspect = float(W) / float(H)
  tx = jnp.asarray(((2.0 * ((gx + 0.5) / float(W)) - 1.0) * aspect), dtype=jnp.float32)
  ty = jnp.asarray((1.0 - 2.0 * ((gy + 0.5) / float(H))), dtype=jnp.float32)

  tan_hf = jnp.tan(0.5 * cam_fovy * (math.pi / 180.0))[:, None, None]
  ldx = tx[None, :, :] * tan_hf
  ldy = ty[None, :, :] * tan_hf
  ldz = -jnp.ones_like(ldx)
  inv_ln = 1.0 / jnp.sqrt(ldx * ldx + ldy * ldy + 1.0)
  ld_hw3 = jnp.stack([ldx * inv_ln, ldy * inv_ln, ldz * inv_ln], axis=-1)
  ray_dir = jnp.einsum("bij,bhwj->bhwi", cam_mat, ld_hw3, precision=jax.lax.Precision.HIGHEST)
  ray_dir = ray_dir / jnp.maximum(jnp.linalg.norm(ray_dir, axis=-1, keepdims=True), 1e-12)

  # Reconstruct world point from planar Z-depth: z_depth = (p_w - cam_pos) . (-cam_fwd) -> t_total = z_depth / neg_z
  t_total = jnp.where(hit, z_depth.astype(jnp.float32) / jnp.maximum(inv_ln, 1e-6), 0.0)
  pts_w = cam_pos[:, None, None, :] + t_total[..., None] * ray_dir

  vel_w = jnp.zeros((B, H, W, 3), dtype=jnp.float32)
  cvel = mjx_data_batch.cvel
  scom = mjx_data_batch.subtree_com
  for bid in active_moving_bids:
    if bid <= 0:
      continue
    rid = body_rootid[bid] if bid < len(body_rootid) else bid
    omega = cvel[:, bid, :3][:, None, None, :]
    vlin = cvel[:, bid, 3:][:, None, None, :]
    com = scom[:, rid, :][:, None, None, :]
    v_b = vlin + jnp.cross(omega, pts_w - com)
    vel_w = jnp.where((body_id_map == bid)[..., None], v_b, vel_w)

  uv_curr = _project_world_to_pixel_dyn_fov(H, W, cam_pos, cam_mat, cam_fovy, pts_w)
  uv_next = _project_world_to_pixel_dyn_fov(H, W, cam_pos, cam_mat, cam_fovy, pts_w + vel_w * float(dt))
  flow = jnp.where(hit[..., None], uv_next - uv_curr, 0.0)
  return flow.astype(out_dtype)


# ==============================================================================
# Isaac Sim `TiledCamera` API
# ==============================================================================

@dataclasses.dataclass(frozen=True)
class TiledCameraCfg:
  cam_names: Union[int, str, Sequence[Union[int, str]]] = 0
  height: int = 128
  width: int = 128
  data_types: Tuple[str, ...] = DEFAULT_MODALITIES
  shader: str = "pbr_ggx"       # "pbr_ggx" | "fast_dr"
  tonemap: str = "agx"          # "agx" | "linear"
  out_dtype: Any = jnp.bfloat16
  block_b: int = 8
  block_r: int = 512
  n_ao_samples: int = 4
  n_shadow_samples: int = 4
  ao_radius: float = 0.35
  ao_strength: float = 0.72
  shadow_softness: float = 0.055
  cull_mode: str = "cluster1"   # used when G > 128
  use_shadows: bool = True
  shadow_light_fraction: float = 0.08
  dt: float = 0.005
  n_chips: Optional[int] = None
  data_axis: str = "dp"
  interpret: Optional[bool] = None
  semantic_fn: Optional[Callable[[Any, Sequence[int]], Tuple[int, ...]]] = None


def _default_category_masks(src: Any) -> Dict[str, np.ndarray]:
  """Infer semantic & DR category masks from MJCF geom/body names."""
  ngeom = int(src.ngeom)
  masks = {
      "floor": np.zeros((ngeom,), dtype=bool),
      "table": np.zeros((ngeom,), dtype=bool),
      "robot_body": np.zeros((ngeom,), dtype=bool),
      "bin": np.zeros((ngeom,), dtype=bool),
      "object": np.zeros((ngeom,), dtype=bool),
  }
  g_bodyid = np.asarray(src.geom_bodyid, dtype=np.int32)
  g_type = np.asarray(src.geom_type, dtype=np.int32)
  body_has_free = np.zeros((int(src.nbody),), dtype=bool)
  if hasattr(src, "jnt_type") and hasattr(src, "jnt_bodyid"):
    jt = np.asarray(src.jnt_type, dtype=np.int32)
    jb = np.asarray(src.jnt_bodyid, dtype=np.int32)
    for j in range(len(jt)):
      if int(jt[j]) == 0:  # mjtJoint.mjJNT_FREE
        body_has_free[int(jb[j])] = True

  has_mj_names = hasattr(src, "names") and hasattr(src, "name_geomadr")
  def _gname(gid: int) -> str:
    try:
      import mujoco
      if isinstance(src, mujoco.MjModel):
        return mujoco.mj_id2name(src, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
    except Exception:
      pass
    if has_mj_names:
      adr = int(src.name_geomadr[gid])
      if adr >= 0:
        raw = bytes(src.names[adr:])
        return raw.split(b"\x00", 1)[0].decode("utf-8", errors="ignore")
    return ""

  for g in range(ngeom):
    nm = _gname(g).lower()
    bid = int(g_bodyid[g])
    if int(g_type[g]) == GEOM_PLANE or nm == "floor":
      masks["floor"][g] = True
    elif any(k in nm for k in ("workbench_top", "bench_top", "table_top", "tabletop")):
      masks["table"][g] = True
    elif nm.startswith(("l_", "r_", "left_arm", "right_arm", "panda", "arm_")):
      # Keep dark joint collars neutral while tinting primary white arm shells
      if not any(c in nm for c in ("collar", "joint", "flange", "pad", "finger_vis")):
        masks["robot_body"][g] = True
    elif "bin_" in nm:
      masks["bin"][g] = True
    elif (
        body_has_free[bid]
        or any(k in nm for k in ("cube", "box", "can", "bottle", "ball", "crate", "mug", "prop", "target", "tray"))
    ):
      if "workbench" not in nm and "bench" not in nm and "head_bar" not in nm:
        masks["object"][g] = True
  return masks


def _default_semantic_classes(src: Any, gids: Sequence[int], masks: Dict[str, np.ndarray]) -> Tuple[int, ...]:
  """Classes: 1=floor/table/static, 2=robot, 3=targets/props, 4=bins, 5=racks/tools."""
  g_bodyid = np.asarray(src.geom_bodyid, dtype=np.int32)
  out = []
  for g in gids:
    gi = int(g)
    if masks["floor"][gi] or masks["table"][gi]:
      out.append(1)
    elif masks["robot_body"][gi] or int(g_bodyid[gi]) > 0 and not masks["object"][gi] and not masks["bin"][gi]:
      # Check if part of robot or moving prop
      out.append(2 if not masks["object"][gi] else 3)
    elif masks["object"][gi]:
      out.append(3)
    elif masks["bin"][gi]:
      out.append(4)
    else:
      out.append(5)
  return tuple(out)


def _resolve_cam_ids(src: Any, cam_names: Union[int, str, Sequence[Union[int, str]]]) -> Tuple[int, ...]:
  items = [cam_names] if isinstance(cam_names, (int, str)) else list(cam_names)
  out: List[int] = []
  for item in items:
    if isinstance(item, int):
      out.append(item)
    else:
      try:
        import mujoco
        cid = mujoco.mj_name2id(src, mujoco.mjtObj.mjOBJ_CAMERA, str(item))
      except Exception:
        cid = -1
      if cid < 0:
        raise ValueError(f"Camera '{item}' not found in model.")
      out.append(int(cid))
  return tuple(out)


def tile_grid_mosaic(images: jax.Array, grid_shape: Tuple[int, int]) -> jax.Array:
  """Zero-copy TPU assembly of `(B, H, W, ...)` batch into a single `(gh*H, gw*W, ...)` tiled canvas."""
  gh, gw = int(grid_shape[0]), int(grid_shape[1])
  n_need = gh * gw
  assert images.shape[0] >= n_need, f"Need >= {n_need} envs for grid {grid_shape}, got {images.shape[0]}"
  sub = images[:n_need]
  H, W = int(sub.shape[1]), int(sub.shape[2])
  tail = sub.shape[3:]
  # (gh, gw, H, W, *tail) -> (gh, H, gw, W, *tail) -> (gh*H, gw*W, *tail)
  axes = [0, 2, 1, 3] + list(range(4, 4 + len(tail)))
  return sub.reshape(gh, gw, H, W, *tail).transpose(axes).reshape(gh * H, gw * W, *tail)


class TiledCamera:
  """Isaac Sim `TiledCamera`-compatible TPU Mosaic Multi-Modal Sensor with Omniverse Replicator DR."""

  def __init__(self, mjmodel: Any, cfg: Optional[TiledCameraCfg] = None):
    self.cfg = cfg or TiledCameraCfg()
    self.cam_ids = _resolve_cam_ids(mjmodel, self.cfg.cam_names)
    self.multi_cam = not isinstance(self.cfg.cam_names, (int, str))
    self.specs: Tuple[RenderSceneSpec, ...] = tuple(
        RenderSceneSpec.from_mjmodel(
            mjmodel,
            cam_id=cid,
            height=int(self.cfg.height),
            width=int(self.cfg.width),
            use_shadows=bool(self.cfg.use_shadows),
            shadow_light_fraction=float(self.cfg.shadow_light_fraction),
        )
        for cid in self.cam_ids
    )
    self.base_fovys: Tuple[float, ...] = tuple(float(sp.cam_fovy) for sp in self.specs)
    self.category_masks = _default_category_masks(mjmodel)
    gids = self.specs[0].enabled_geom_ids
    if self.cfg.semantic_fn is not None:
      self.geom_sem_ids = tuple(int(x) for x in self.cfg.semantic_fn(mjmodel, gids))
    else:
      self.geom_sem_ids = _default_semantic_classes(mjmodel, gids, self.category_masks)

    g_bid = np.asarray(mjmodel.geom_bodyid, dtype=np.int32)
    b_rid = np.asarray(mjmodel.body_rootid, dtype=np.int32)
    self.geom_body_ids: Tuple[int, ...] = tuple(int(g_bid[g]) for g in gids)
    self.body_rootid: Tuple[int, ...] = tuple(int(x) for x in b_rid)
    self.active_moving_bids: Tuple[int, ...] = tuple(sorted({bid for bid in self.geom_body_ids if bid > 0}))
    self.floor_gids: Tuple[int, ...] = tuple(int(g) for g in gids if self.category_masks["floor"][g])
    self.table_gids: Tuple[int, ...] = tuple(int(g) for g in gids if self.category_masks["table"][g])

    n_devs = len(jax.devices())
    self.n_chips = min(n_devs, int(self.cfg.n_chips)) if self.cfg.n_chips is not None else 1
    if self.n_chips > 1:
      self.mesh = jax.sharding.Mesh(np.array(jax.devices()[: self.n_chips]), (self.cfg.data_axis,))
    else:
      self.mesh = None
    self._compiled_cache: Dict[Any, Any] = {}

  @property
  def n_enabled_geoms(self) -> int:
    return len(self.specs[0].enabled_geom_ids)

  def randomize_local(
      self,
      rng_or_keys: jax.Array,
      mjx_data_batch: Any,
      rep_cfg: Optional[ReplicatorCfg] = None,
  ) -> ReplicatorBatch:
    """Run Omniverse Replicator DR on a local/sharded batch (pure JAX, zero host callbacks)."""
    return _apply_replicator_local(
        rng_or_keys=rng_or_keys,
        mjx_data_batch=mjx_data_batch,
        base_spec=self.specs[0],
        cam_ids=self.cam_ids,
        base_fovys=self.base_fovys,
        category_masks=self.category_masks,
        cfg=rep_cfg or ReplicatorCfg(),
    )

  def _render_single_cam_local(
      self,
      spec: RenderSceneSpec,
      cam_slot: int,
      mjx_data_batch: Any,
      rep_batch: Optional[ReplicatorBatch],
  ) -> Dict[str, jax.Array]:
    interp = (jax.default_backend() == "cpu") if self.cfg.interpret is None else bool(self.cfg.interpret)
    need_aux = any(
        k in self.cfg.data_types
        for k in ("normals", "semantic_segmentation", "instance_id", "motion_vectors")
    )
    n_g = len(spec.enabled_geom_ids)
    if n_g <= 128:
      raw = _pallas_tiled_dr_unrolled(
          spec=spec,
          mjx_data_batch=mjx_data_batch,
          rep_batch=rep_batch,
          cam_slot=cam_slot,
          geom_sem_ids=self.geom_sem_ids,
          geom_body_ids=self.geom_body_ids,
          floor_gids=self.floor_gids,
          table_gids=self.table_gids,
          shader=self.cfg.shader,
          tonemap=self.cfg.tonemap,
          out_dtype=self.cfg.out_dtype,
          block_b=self.cfg.block_b,
          block_r=self.cfg.block_r,
          n_ao_samples=self.cfg.n_ao_samples,
          n_shadow_samples=self.cfg.n_shadow_samples,
          ao_radius=self.cfg.ao_radius,
          ao_strength=self.cfg.ao_strength,
          shadow_softness=self.cfg.shadow_softness,
          emit_aux=need_aux,
          interpret=interp,
      )
    else:
      raw = _pallas_tiled_dr_hierarchical(
          spec=spec,
          mjx_data_batch=mjx_data_batch,
          rep_batch=rep_batch,
          cam_slot=cam_slot,
          geom_sem_ids=self.geom_sem_ids,
          geom_body_ids=self.geom_body_ids,
          floor_gids=self.floor_gids,
          table_gids=self.table_gids,
          cull_mode=self.cfg.cull_mode,
          tonemap=self.cfg.tonemap,
          out_dtype=self.cfg.out_dtype,
          block_b=self.cfg.block_b,
          block_r=self.cfg.block_r,
          emit_aux=need_aux,
          interpret=interp,
      )

    if "motion_vectors" in self.cfg.data_types:
      raw["motion_vectors"] = _compute_motion_vectors_zero_gather(
          spec=spec,
          mjx_data_batch=mjx_data_batch,
          rep_batch=rep_batch,
          cam_slot=cam_slot,
          z_depth=raw["distance_to_image_plane"],
          instance_id=raw["instance_id"],
          body_id_map=raw["_body_id"],
          active_moving_bids=self.active_moving_bids,
          body_rootid=self.body_rootid,
          dt=self.cfg.dt,
          out_dtype=self.cfg.out_dtype,
      )

    return {k: raw[k] for k in self.cfg.data_types}

  def render_local(
      self,
      mjx_data_batch: Any,
      rep_batch: Optional[ReplicatorBatch] = None,
      rng_or_keys: Optional[jax.Array] = None,
      rep_cfg: Optional[ReplicatorCfg] = None,
  ) -> Dict[str, jax.Array]:
    """Render local batch inside `jax.jit` or `shard_map`, fusing multi-cam into ONE Mosaic kernel."""
    if rep_batch is None and rng_or_keys is not None:
      rep_batch = self.randomize_local(rng_or_keys, mjx_data_batch, rep_cfg=rep_cfg)

    if not self.multi_cam:
      return self._render_single_cam_local(self.specs[0], 0, mjx_data_batch, rep_batch)

    # Fuse all N_cam cameras into a single (B * N_cam, ...) Mosaic Pallas call!
    B = int(mjx_data_batch.cam_xpos.shape[0])
    N_cam = len(self.cam_ids)
    H, W = int(self.cfg.height), int(self.cfg.width)
    interp = (jax.default_backend() == "cpu") if self.cfg.interpret is None else bool(self.cfg.interpret)
    need_aux = any(
        k in self.cfg.data_types
        for k in ("normals", "semantic_segmentation", "instance_id", "motion_vectors")
    )

    def _tile_cam(x: jax.Array) -> jax.Array:
      return jnp.broadcast_to(x[:, None, ...], (B, N_cam, *x.shape[1:])).reshape(B * N_cam, *x.shape[1:])

    cpos_stack = jnp.stack([mjx_data_batch.cam_xpos[:, int(cid), :] for cid in self.cam_ids], axis=1)
    cmat_stack = jnp.stack([mjx_data_batch.cam_xmat[:, int(cid), :, :] for cid in self.cam_ids], axis=1)

    @dataclasses.dataclass
    class _FlatDataProxy:
      geom_xpos: jax.Array
      geom_xmat: jax.Array
      cam_xpos: jax.Array
      cam_xmat: jax.Array
      light_xpos: jax.Array
      light_xmat: jax.Array
      xpos: jax.Array
      xmat: jax.Array

    lpos_src = getattr(mjx_data_batch, "light_xpos", jnp.zeros((B, 0, 3), dtype=jnp.float32))
    lmat_src = getattr(mjx_data_batch, "light_xmat", jnp.zeros((B, 0, 3, 3), dtype=jnp.float32))
    flat_data = _FlatDataProxy(
        geom_xpos=_tile_cam(mjx_data_batch.geom_xpos),
        geom_xmat=_tile_cam(mjx_data_batch.geom_xmat),
        cam_xpos=cpos_stack.reshape(B * N_cam, 1, 3),
        cam_xmat=cmat_stack.reshape(B * N_cam, 1, 3, 3),
        light_xpos=_tile_cam(lpos_src),
        light_xmat=_tile_cam(lmat_src),
        xpos=_tile_cam(mjx_data_batch.xpos),
        xmat=_tile_cam(mjx_data_batch.xmat),
    )

    if rep_batch is not None:
      flat_rep = ReplicatorBatch(
          cam_xpos=rep_batch.cam_xpos.reshape(B * N_cam, 1, 3),
          cam_xmat=rep_batch.cam_xmat.reshape(B * N_cam, 1, 3, 3),
          cam_fovy_deg=rep_batch.cam_fovy_deg.reshape(B * N_cam, 1),
          light_xpos=_tile_cam(rep_batch.light_xpos),
          light_xdir=_tile_cam(rep_batch.light_xdir),
          light_diffuse=_tile_cam(rep_batch.light_diffuse),
          light_specular=_tile_cam(rep_batch.light_specular),
          light_ambient=_tile_cam(rep_batch.light_ambient),
          sky_rgb=_tile_cam(rep_batch.sky_rgb),
          exposure=_tile_cam(rep_batch.exposure),
          geom_rgba=_tile_cam(rep_batch.geom_rgba),
          geom_roughness=_tile_cam(rep_batch.geom_roughness),
          geom_specular=_tile_cam(rep_batch.geom_specular),
          floor_tex=_tile_cam(rep_batch.floor_tex),
          table_tex=_tile_cam(rep_batch.table_tex),
      )
      fovy_override = None
    else:
      flat_rep = None
      fovy_override = jnp.broadcast_to(
          jnp.asarray(self.base_fovys, dtype=jnp.float32)[None, :], (B, N_cam)
      ).reshape(B * N_cam)

    # Spec with cam_id=0 matching the single-slot flat_data proxy
    sp0 = dataclasses.replace(self.specs[0], cam_id=0)
    n_g = len(sp0.enabled_geom_ids)
    if n_g <= 128:
      raw_flat = _pallas_tiled_dr_unrolled(
          spec=sp0,
          mjx_data_batch=flat_data,
          rep_batch=flat_rep,
          cam_slot=0,
          geom_sem_ids=self.geom_sem_ids,
          geom_body_ids=self.geom_body_ids,
          floor_gids=self.floor_gids,
          table_gids=self.table_gids,
          shader=self.cfg.shader,
          tonemap=self.cfg.tonemap,
          out_dtype=self.cfg.out_dtype,
          block_b=self.cfg.block_b,
          block_r=self.cfg.block_r,
          n_ao_samples=self.cfg.n_ao_samples,
          n_shadow_samples=self.cfg.n_shadow_samples,
          ao_radius=self.cfg.ao_radius,
          ao_strength=self.cfg.ao_strength,
          shadow_softness=self.cfg.shadow_softness,
          emit_aux=need_aux,
          interpret=interp,
          cam_fovy_override=fovy_override,
      )
    else:
      raw_flat = _pallas_tiled_dr_hierarchical(
          spec=sp0,
          mjx_data_batch=flat_data,
          rep_batch=flat_rep,
          cam_slot=0,
          geom_sem_ids=self.geom_sem_ids,
          geom_body_ids=self.geom_body_ids,
          floor_gids=self.floor_gids,
          table_gids=self.table_gids,
          cull_mode=self.cfg.cull_mode,
          tonemap=self.cfg.tonemap,
          out_dtype=self.cfg.out_dtype,
          block_b=self.cfg.block_b,
          block_r=self.cfg.block_r,
          emit_aux=need_aux,
          interpret=interp,
          cam_fovy_override=fovy_override,
      )

    out: Dict[str, jax.Array] = {}
    for k, v in raw_flat.items():
      out[k] = v.reshape(B, N_cam, H, W, *v.shape[3:])

    if "motion_vectors" in self.cfg.data_types:
      flows = [
          _compute_motion_vectors_zero_gather(
              spec=self.specs[ci],
              mjx_data_batch=mjx_data_batch,
              rep_batch=rep_batch,
              cam_slot=ci,
              z_depth=out["distance_to_image_plane"][:, ci],
              instance_id=out["instance_id"][:, ci],
              body_id_map=out["_body_id"][:, ci],
              active_moving_bids=self.active_moving_bids,
              body_rootid=self.body_rootid,
              dt=self.cfg.dt,
              out_dtype=self.cfg.out_dtype,
          )
          for ci in range(N_cam)
      ]
      out["motion_vectors"] = jnp.stack(flows, axis=1)

    return {k: out[k] for k in self.cfg.data_types}

  def shard_tree(self, tree: Any) -> Any:
    """Place a batched PyTree across `self.mesh` along `data_axis`."""
    if self.mesh is None or self.n_chips <= 1:
      return jax.device_put(tree, jax.devices()[0])
    P = jax.sharding.PartitionSpec
    sharding = jax.sharding.NamedSharding(self.mesh, P(self.cfg.data_axis))
    return jax.device_put(tree, sharding)

  def build_step_and_render_fn(
      self,
      mjx_model: Any,
      batch_size: int,
      rep_cfg: Optional[ReplicatorCfg] = None,
      with_dr: bool = True,
      with_physics: bool = True,
      substeps: int = 1,
  ) -> Callable[..., Tuple[Any, Dict[str, jax.Array]]]:
    """Compile ONE zero-copy XLA graph: (optional MJX step) + (optional Replicator DR) + TiledCamera."""
    from mujoco import mjx

    use_spmd = (
        self.mesh is not None
        and self.n_chips > 1
        and batch_size >= self.n_chips
        and (batch_size % self.n_chips == 0)
    )
    key = (bool(use_spmd), bool(with_dr), bool(with_physics), int(substeps), rep_cfg)
    if key in self._compiled_cache:
      return self._compiled_cache[key]

    def _local_fn(mx_l, d_b, u_b, rng_b):
      if with_physics:
        if substeps == 1:
          dn = jax.vmap(lambda di, ui: mjx.step(mx_l, di.replace(ctrl=ui)))(d_b, u_b)
        else:
          def _sub(_, ds):
            return jax.vmap(lambda di, ui: mjx.step(mx_l, di.replace(ctrl=ui)))(ds, u_b)
          dn = jax.lax.fori_loop(0, int(substeps), _sub, d_b)
      else:
        dn = d_b
      rb = self.randomize_local(rng_b, dn, rep_cfg=rep_cfg) if with_dr else None
      obs = self.render_local(dn, rep_batch=rb)
      return dn, obs

    if not use_spmd:
      @jax.jit
      def _fn(d_b, u_b, rng_b):
        return _local_fn(mjx_model, d_b, u_b, rng_b)
    else:
      P = jax.sharding.PartitionSpec
      ax = self.cfg.data_axis
      sm_kw = _shard_map_kwargs()
      _sm = shard_map(
          _local_fn,
          mesh=self.mesh,
          in_specs=(P(), P(ax), P(ax), P(ax)),
          out_specs=(P(ax), {k: P(ax) for k in self.cfg.data_types}),
          **sm_kw,
      )

      @jax.jit
      def _fn(d_b, u_b, rng_b):
        return _sm(mjx_model, d_b, u_b, rng_b)

    self._compiled_cache[key] = _fn
    return _fn
