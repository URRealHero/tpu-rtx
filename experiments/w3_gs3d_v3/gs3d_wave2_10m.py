"""Wave 2 Massive Scene-Scale 3DGS (100k -> 1M -> 10M Gaussians) up to Native 4K (3840x2160) on TPU v5p.

Architecture (zero host callbacks, pure TPU HBM/VMEM):
1. Vectorized 3D->2D EWA projection & frustum culling across N in {100k, 1M, 10M}.
2. 2x2 screen-space 64x64 macro-tile assignment with packed uint32 key
   ((macro_id << 21) | (z_bits >> 11)) and `jax.lax.sort_key_val` front-to-back sorting.
3. Exact O(M_tiles * log L) `jnp.searchsorted` bucket starts/counts + fixed-capacity
   `K_macro` dynamic_slice window per 64x64 macro-tile.
4. Direct reuse of Wave 1 `pallas_gs3d_v3_fused_rasterize` (`t_grp=16`, zero output bloat)
   across all `M_tiles` 64x64 macro-tiles (16 macro-tiles @ 256x256, 64 @ 512x512,
   510 @ 1080p, 2,040 @ Native 4K 3840x2160), with 4-chip `shard_map` parallelization.
"""

from __future__ import annotations

import inspect
import math
from typing import Any, Dict, Tuple

import jax
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
import numpy as np

from gs3d_v3 import pallas_gs3d_v3_fused_rasterize


def build_massive_scene_arrays(
    n_gaussians: int,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
  """Build a rich multi-cluster 3D scene scaled to N in {100k, 1M, 10M} Gaussians.

  Combines the real Blender scene geometry (`blender_real_scene.npz`) replicated as
  foreground hero sculptures + dense procedural terrain/arch/foliage clusters in
  world coordinates so Native 4K renders show rich multi-scale depth & parallax.
  """
  rng = np.random.default_rng(seed)
  ds = np.load("/dev/shm/zaychik-tpu-mjx/out/assets/blender_real_scene.npz")

  # Allocate 20% to dense hero Blender sculptures (8 arranged instances) + 80% terrain/spheres/ribbons
  n_hero = min(n_gaussians // 5, 1_000_000)
  n_bg = n_gaussians - n_hero

  # Sample hero points from the real Blender mesh triangles
  M = int(ds["n_meshes"])
  all_v0, all_v1, all_v2, all_col, all_area = [], [], [], [], []
  for m in range(M):
    verts = np.asarray(ds[f"mesh_{m}_verts"], dtype=np.float32)
    faces = np.asarray(ds[f"mesh_{m}_faces"], dtype=np.int32)
    rgba = np.asarray(ds[f"mesh_{m}_rgba"], dtype=np.float32)[:3]
    # Transform by frame 0 pose
    pos0 = np.asarray(ds["obj_xpos"][0, m], dtype=np.float32)
    mat0 = np.asarray(ds["obj_xmat"][0, m], dtype=np.float32)
    vw = verts @ mat0.T + pos0[None, :]
    v0, v1, v2 = vw[faces[:, 0]], vw[faces[:, 1]], vw[faces[:, 2]]
    a = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=-1)
    all_v0.append(v0)
    all_v1.append(v1)
    all_v2.append(v2)
    all_col.append(np.broadcast_to(rgba[None, :], (len(faces), 3)))
    all_area.append(a)

  v0_c = np.concatenate(all_v0, axis=0)
  v1_c = np.concatenate(all_v1, axis=0)
  v2_c = np.concatenate(all_v2, axis=0)
  col_c = np.concatenate(all_col, axis=0)
  area_c = np.concatenate(all_area, axis=0)
  probs = area_c / np.maximum(np.sum(area_c), 1e-12)

  tri_idx = rng.choice(len(v0_c), size=n_hero, replace=True, p=probs)
  r1 = np.sqrt(rng.uniform(0.0, 1.0, size=(n_hero, 1))).astype(np.float32)
  r2 = rng.uniform(0.0, 1.0, size=(n_hero, 1)).astype(np.float32)
  w0, w1, w2 = 1.0 - r1, r1 * (1.0 - r2), r1 * r2
  mu_hero = (w0 * v0_c[tri_idx] + w1 * v1_c[tri_idx] + w2 * v2_c[tri_idx]).astype(np.float32)
  # Slight color variation + spatial shading tint
  rgb_hero = np.clip(col_c[tri_idx] * rng.uniform(0.85, 1.15, size=(n_hero, 3)), 0.0, 1.0).astype(np.float32)
  # Scale radius inversely with sqrt(n_hero) so surface remains crisp even at 4K
  rad_hero = float(np.clip(0.025 * math.sqrt(2048.0 / max(n_hero, 2048)), 0.0025, 0.025))
  scale_hero = np.full((n_hero, 3), rad_hero, dtype=np.float32)

  # Procedural multi-cluster background & ground plane in [-3.5, 3.5] x [-3.5, 3.5] x [0, 1.5]
  n_ground = n_bg // 2
  n_clouds = n_bg - n_ground
  gx = rng.uniform(-3.2, 3.2, size=n_ground).astype(np.float32)
  gy = rng.uniform(-3.2, 3.2, size=n_ground).astype(np.float32)
  gz = (0.04 * np.sin(2.5 * gx) * np.cos(2.5 * gy) - 0.05).astype(np.float32)
  mu_ground = np.stack([gx, gy, gz], axis=-1)
  checker = ((np.floor(gx * 2.0) + np.floor(gy * 2.0)) % 2 == 0).astype(np.float32)
  rgb_ground = np.stack(
      [0.18 + 0.14 * checker, 0.22 + 0.16 * checker, 0.28 + 0.18 * checker], axis=-1
  ).astype(np.float32)
  rad_gnd = float(np.clip(0.05 * math.sqrt(100_000.0 / max(n_ground, 1000)), 0.004, 0.05))
  scale_ground = np.stack(
      [
          np.full((n_ground,), rad_gnd, np.float32),
          np.full((n_ground,), rad_gnd, np.float32),
          np.full((n_ground,), rad_gnd * 0.25, np.float32),
      ],
      axis=-1,
  )

  # 12 colorful 3D torus/ring & pillar clusters framing the Blender centerpiece
  cluster_id = rng.integers(0, 12, size=n_clouds)
  ang_c = (cluster_id.astype(np.float32) / 12.0) * (2.0 * math.pi)
  cx_c = 1.65 * np.cos(ang_c)
  cy_c = 1.65 * np.sin(ang_c)
  cz_c = 0.35 + 0.25 * np.sin(3.0 * ang_c)
  theta = rng.uniform(0.0, 2.0 * math.pi, size=n_clouds).astype(np.float32)
  phi = rng.uniform(0.0, 2.0 * math.pi, size=n_clouds).astype(np.float32)
  R_t, r_t = 0.28, 0.10
  lx = (R_t + r_t * np.cos(phi)) * np.cos(theta)
  ly = (R_t + r_t * np.cos(phi)) * np.sin(theta)
  lz = r_t * np.sin(phi)
  mu_clouds = np.stack([cx_c + lx, cy_c + ly, cz_c + lz], axis=-1).astype(np.float32)
  # HSV-like palette per cluster
  rgb_clouds = np.stack(
      [
          0.45 + 0.45 * np.sin(ang_c + 0.0),
          0.45 + 0.45 * np.sin(ang_c + 2.094),
          0.45 + 0.45 * np.sin(ang_c + 4.188),
      ],
      axis=-1,
  ).astype(np.float32)
  rgb_clouds = np.clip(rgb_clouds * rng.uniform(0.85, 1.15, size=(n_clouds, 3)), 0.0, 1.0).astype(np.float32)
  rad_cld = float(np.clip(0.025 * math.sqrt(100_000.0 / max(n_clouds, 1000)), 0.0025, 0.025))
  scale_clouds = np.full((n_clouds, 3), rad_cld, dtype=np.float32)

  mu = np.concatenate([mu_hero, mu_ground, mu_clouds], axis=0)
  rgb = np.concatenate([rgb_hero, rgb_ground, rgb_clouds], axis=0)
  scale = np.concatenate([scale_hero, scale_ground, scale_clouds], axis=0)
  op = np.full((n_gaussians,), 0.88, dtype=np.float32)

  cam_xpos = np.asarray(ds["cam_xpos"][0], dtype=np.float32)
  cam_xmat = np.asarray(ds["cam_xmat"][0], dtype=np.float32)
  return {
      "mu": mu,
      "scale2": (scale * scale).astype(np.float32),
      "rgb": rgb,
      "opacity": op,
      "cam_xpos": cam_xpos,
      "cam_xmat": cam_xmat,
      "cam_fovy": np.float32(ds["cam_fovy"]),
      "znear": np.float32(ds["znear"]),
  }


def project_and_bucket_macro_tiles(
    mu: jax.Array,      # (N, 3)
    cov_in: jax.Array,  # (N, 6) symmetric (xx,xy,xz,yy,yz,zz) OR (N, 3) diagonal scale2
    rgb: jax.Array,     # (N, 3)
    opacity: jax.Array, # (N,)
    cam_xpos: jax.Array,  # (3,)
    cam_xmat: jax.Array,  # (3, 3)
    height: int,
    width: int,
    nrm: jax.Array | None = None,  # (N, 3) surface normal for dynamic Lambertian+sky shading
    cam_fovy: float = 42.0,
    znear: float = 0.05,
    k_macro: int = 4096,
) -> Tuple[jax.Array, jax.Array, jax.Array, int, int, int, int]:
  """Project N anisotropic 3D Gaussians, replicate across up to 3x3 macro-tiles (32x32), and bucket via uint32 sort + searchsorted."""
  N = mu.shape[0]
  M_SIZE = 32
  H_pad = ((height + M_SIZE - 1) // M_SIZE) * M_SIZE
  W_pad = ((width + M_SIZE - 1) // M_SIZE) * M_SIZE
  My = H_pad // M_SIZE
  Mx = W_pad // M_SIZE
  M_tiles = My * Mx
  assert M_tiles < 8191, f"M_tiles={M_tiles} must fit in 13 bits (<8191)"

  # 1. Camera transform (N, 3)
  p_cam = (mu - cam_xpos[None, :]) @ cam_xmat  # (N, 3)
  cx, cy, cz = p_cam[:, 0], p_cam[:, 1], p_cam[:, 2]
  z_d = -cz
  in_front = z_d > znear
  z_safe = jnp.maximum(z_d, znear)

  half_h = math.tan(0.5 * math.radians(float(cam_fovy)))
  fy = 0.5 * float(height) / half_h
  fx = fy
  mean_x = 0.5 * float(width) + fx * (cx / z_safe)
  mean_y = 0.5 * float(height) - fy * (cy / z_safe)

  inv_z = 1.0 / z_safe
  inv_z2 = inv_z * inv_z
  lim_x = 1.3 * (0.5 * float(width) / fx) * z_safe
  lim_y = 1.3 * (0.5 * float(height) / fy) * z_safe
  cx_c = jnp.clip(cx, -lim_x, lim_x)
  cy_c = jnp.clip(cy, -lim_y, lim_y)
  j00, j02 = fx * inv_z, fx * cx_c * inv_z2
  j11, j12 = -fy * inv_z, -fy * cy_c * inv_z2

  w0 = j00[:, None] * cam_xmat[None, :, 0] + j02[:, None] * cam_xmat[None, :, 2]
  w1 = j11[:, None] * cam_xmat[None, :, 1] + j12[:, None] * cam_xmat[None, :, 2]
  if cov_in.shape[-1] == 6:
    s00, s01, s02, s11, s12, s22 = [cov_in[:, i] for i in range(6)]
    u0, u1, u2 = w0[:, 0], w0[:, 1], w0[:, 2]
    v0, v1, v2 = w1[:, 0], w1[:, 1], w1[:, 2]
    sw0_0 = s00 * u0 + s01 * u1 + s02 * u2
    sw0_1 = s01 * u0 + s11 * u1 + s12 * u2
    sw0_2 = s02 * u0 + s12 * u1 + s22 * u2
    sw1_0 = s00 * v0 + s01 * v1 + s02 * v2
    sw1_1 = s01 * v0 + s11 * v1 + s12 * v2
    sw1_2 = s02 * v0 + s12 * v1 + s22 * v2
    cov2_xx = u0 * sw0_0 + u1 * sw0_1 + u2 * sw0_2 + 0.30
    cov2_xy = v0 * sw0_0 + v1 * sw0_1 + v2 * sw0_2
    cov2_yy = v0 * sw1_0 + v1 * sw1_1 + v2 * sw1_2 + 0.30
  else:
    cov2_xx = jnp.sum(w0 * w0 * cov_in, axis=-1) + 0.30
    cov2_xy = jnp.sum(w0 * w1 * cov_in, axis=-1)
    cov2_yy = jnp.sum(w1 * w1 * cov_in, axis=-1) + 0.30

  det = cov2_xx * cov2_yy - cov2_xy * cov2_xy
  valid_g = in_front & (det > 1e-6)
  inv_det = jnp.where(valid_g, 1.0 / jnp.maximum(det, 1e-6), 0.0)
  icxx = cov2_yy * inv_det
  icxy = -cov2_xy * inv_det
  icyy = cov2_xx * inv_det

  # Tight axis-aligned 2.6-sigma half-extents
  hx = 2.6 * jnp.sqrt(jnp.maximum(cov2_xx, 0.1))
  hy = 2.6 * jnp.sqrt(jnp.maximum(cov2_yy, 0.1))

  on_screen = (
      valid_g
      & (mean_x + hx >= 0.0)
      & (mean_x - hx < float(width))
      & (mean_y + hy >= 0.0)
      & (mean_y - hy < float(height))
  )

  # 2. Macro-tile 3x3 replication keys (packed uint32: [13b macro_id | 19b z_bits])
  tx0 = jnp.clip(jnp.floor((mean_x - hx) / float(M_SIZE)).astype(jnp.int32), 0, Mx - 1)
  tx1 = jnp.clip(jnp.floor((mean_x + hx) / float(M_SIZE)).astype(jnp.int32), 0, Mx - 1)
  ty0 = jnp.clip(jnp.floor((mean_y - hy) / float(M_SIZE)).astype(jnp.int32), 0, My - 1)
  ty1 = jnp.clip(jnp.floor((mean_y + hy) / float(M_SIZE)).astype(jnp.int32), 0, My - 1)

  z_u32 = jax.lax.bitcast_convert_type(jnp.maximum(z_safe, 1e-5), jnp.uint32) >> jnp.uint32(13)
  INV_MACRO = jnp.uint32(8191)

  def _make_key(v_mask, tx, ty):
    m_id = jnp.where(v_mask, (ty * Mx + tx).astype(jnp.uint32), INV_MACRO)
    return (m_id << jnp.uint32(19)) | z_u32

  key_parts = []
  for dy in (0, 1, 2):
    for dx in (0, 1, 2):
      vx = on_screen if dx == 0 else (on_screen & ((tx0 + dx) <= tx1))
      vy = vx if dy == 0 else (vx & ((ty0 + dy) <= ty1))
      key_parts.append(_make_key(vy, jnp.minimum(tx0 + dx, Mx - 1), jnp.minimum(ty0 + dy, My - 1)))

  keys = jnp.concatenate(key_parts, axis=0)
  idx_n = jnp.arange(N, dtype=jnp.int32)
  g_ids = jnp.tile(idx_n, 9)

  keys_s, gids_s = jax.lax.sort_key_val(keys, g_ids, dimension=0)
  macro_s = keys_s >> jnp.uint32(19)

  m_queries = jnp.arange(M_tiles, dtype=jnp.uint32)
  starts = jnp.searchsorted(macro_s, m_queries, side="left")
  ends = jnp.searchsorted(macro_s, m_queries, side="right")
  counts = jnp.minimum(ends - starts, k_macro)

  # Pad gids_s by k_macro at the tail so dynamic_slice is always in-bounds
  gids_pad = jnp.pad(gids_s, (0, k_macro), constant_values=0)
  slot_offs = jnp.arange(k_macro, dtype=jnp.int32)[None, :]

  def _slice_macro(st, cnt):
    sl = jax.lax.dynamic_slice(gids_pad, (st,), (k_macro,))
    return jnp.where(jnp.arange(k_macro, dtype=jnp.int32) < cnt, sl, -1)

  macro_gidx = jax.vmap(_slice_macro)(starts, counts)  # (M_tiles, k_macro)

  # Dynamic directional sun + sky hemisphere shading when normals are provided
  if nrm is not None:
    vdir = cam_xpos[None, :] - mu
    vdir = vdir / jnp.maximum(jnp.linalg.norm(vdir, axis=-1, keepdims=True), 1e-12)
    nrm_vis = jnp.where(jnp.sum(nrm * vdir, axis=-1, keepdims=True) < 0.0, -nrm, nrm)
    ldir = jnp.array([0.28, 0.90, 0.34], dtype=jnp.float32)
    ldir = ldir / jnp.linalg.norm(ldir)
    ndotl = jnp.maximum(0.0, jnp.sum(nrm_vis * ldir[None, :], axis=-1))
    hemi = 0.5 + 0.5 * nrm_vis[:, 1]
    shade = 0.42 + 0.22 * hemi + 0.78 * ndotl
    rgb_lit = jnp.clip(rgb * shade[:, None], 0.0, 1.0)
    nx, ny, nz = nrm_vis[:, 0], nrm_vis[:, 1], nrm_vis[:, 2]
  else:
    vdir = cam_xpos[None, :] - mu
    vdir = vdir / jnp.maximum(jnp.linalg.norm(vdir, axis=-1, keepdims=True), 1e-12)
    shade = 0.65 + 0.35 * jnp.clip(vdir[:, 2], 0.0, 1.0)
    rgb_lit = jnp.clip(rgb * shade[:, None], 0.0, 1.0)
    nx = jnp.zeros((N,), dtype=jnp.float32)
    ny = jnp.zeros((N,), dtype=jnp.float32)
    nz = jnp.ones((N,), dtype=jnp.float32)

  z_n = jnp.zeros((N,), dtype=jnp.float32)
  # Pack (N, 16) so row gather across macro_gidx is contiguous along minor axis 16!
  splats_n16 = jnp.stack(
      [
          mean_x, mean_y, icxx, icxy, icyy,
          rgb_lit[:, 0], rgb_lit[:, 1], rgb_lit[:, 2],
          opacity, z_d, nx, ny, nz,
          hx, hy, z_n,
      ],
      axis=1,
  )
  return splats_n16, macro_gidx, counts, H_pad, W_pad, My, Mx


def render_macro_tiles_subset(
    splats_n16: jax.Array,  # (N, 16)
    macro_gidx: jax.Array,  # (M_sub, k_macro) int32 (-1 invalid)
    tile_ox_m: jax.Array,   # (M_sub,) float32 (global pixel X origin of each 32x32 macro-tile)
    tile_oy_m: jax.Array,   # (M_sub,) float32 (global pixel Y origin of each 32x32 macro-tile)
    use_bf16_mxu: bool = False,
    interpret: bool = False,
) -> jax.Array:
  """Render a slice of 32x32 macro-tiles (`M_sub` tiles) using Wave 1 V3 Pallas kernel!

  Returns:
    rgb_macro: (M_sub, 32, 32, 3) float32
  """
  M_sub, k_macro = macro_gidx.shape
  K_pad = ((k_macro + 127) // 128) * 128
  safe_idx = jnp.maximum(macro_gidx, 0)
  valid_slot = macro_gidx >= 0

  # Gather (M_sub, k_macro, 16) -> transpose to (M_sub, 16, k_macro)
  g16 = splats_n16[safe_idx]  # (M_sub, k_macro, 16)
  mx = g16[..., 0]
  my = g16[..., 1]
  zd = g16[..., 9]
  hx = g16[..., 13]
  hy = g16[..., 14]

  splats_b16n = jnp.transpose(g16, (0, 2, 1))
  if K_pad > k_macro:
    splats_b16n = jnp.pad(splats_b16n, ((0, 0), (0, 0), (0, K_pad - k_macro)))

  # Each 32x32 macro-tile has 4x4 = 16 micro-tiles of 8x8 pixels
  t_ox_loc = jnp.asarray(np.tile(np.arange(4, dtype=np.float32) * 8.0, 4), dtype=jnp.float32)[None, :] + tile_ox_m[:, None]
  t_oy_loc = jnp.asarray(np.repeat(np.arange(4, dtype=np.float32) * 8.0, 4), dtype=jnp.float32)[None, :] + tile_oy_m[:, None]

  # Tight AABB overlap test against 8x8 tile
  overlap = (
      valid_slot[:, None, :]
      & (mx[:, None, :] + hx[:, None, :] >= t_ox_loc[:, :, None])
      & (mx[:, None, :] - hx[:, None, :] <= t_ox_loc[:, :, None] + 8.0)
      & (my[:, None, :] + hy[:, None, :] >= t_oy_loc[:, :, None])
      & (my[:, None, :] - hy[:, None, :] <= t_oy_loc[:, :, None] + 8.0)
  )

  score = jnp.where(overlap, -zd[:, None, :], -1.0e9)
  top_scores, top_idx = jax.lax.approx_max_k(score, 128, recall_target=1.0)
  neg_ts, top_idx = jax.lax.sort_key_val(-top_scores, top_idx, dimension=-1)
  top_scores = -neg_ts
  top_idx = jnp.where(top_scores > -1.0e8, top_idx, -1)

  # Shift mean_x/mean_y into local [0, 32) coords so every macro-tile shares identical 4x4 tile grid!
  splats_local = splats_b16n.at[:, 0, :k_macro].set(mx - tile_ox_m[:, None])
  splats_local = splats_local.at[:, 1, :k_macro].set(my - tile_oy_m[:, None])

  t_ox_0 = np.tile(np.arange(4, dtype=np.float32) * 8.0, 4)
  t_oy_0 = np.repeat(np.arange(4, dtype=np.float32) * 8.0, 4)
  rgb_t, _, _, alp_t = pallas_gs3d_v3_fused_rasterize(
      splats_local,
      top_idx,
      t_ox_0,
      t_oy_0,
      tile_size=8,
      use_bf16_mxu=use_bf16_mxu,
      t_grp=16,
      interpret=interpret,
  )

  rgb_m = rgb_t.reshape(M_sub, 4, 4, 8, 8, 3).transpose(0, 1, 3, 2, 4, 5).reshape(M_sub, 32, 32, 3)
  alp_m = alp_t.reshape(M_sub, 4, 4, 8, 8).transpose(0, 1, 3, 2, 4).reshape(M_sub, 32, 32)
  sky_bg = jnp.array([0.52, 0.72, 0.94], dtype=jnp.float32)[None, None, None, :]
  return jnp.clip(rgb_m + (1.0 - alp_m[..., None]) * sky_bg, 0.0, 1.0)


def render_massive_scene_1c(
    mu, cov_in, rgb, opacity, cam_xpos, cam_xmat,
    height: int, width: int, nrm: jax.Array | None = None,
    cam_fovy: float = 42.0, znear: float = 0.05,
    k_macro: int = 4096, use_bf16_mxu: bool = False, interpret: bool = False,
) -> jax.Array:
  splats_n16, macro_gidx, _, H_pad, W_pad, My, Mx = project_and_bucket_macro_tiles(
      mu, cov_in, rgb, opacity, cam_xpos, cam_xmat,
      height=height, width=width, nrm=nrm, cam_fovy=cam_fovy, znear=znear, k_macro=k_macro,
  )
  t_ox_m = jnp.asarray(np.tile(np.arange(Mx, dtype=np.float32) * 32.0, My), dtype=jnp.float32)
  t_oy_m = jnp.asarray(np.repeat(np.arange(My, dtype=np.float32) * 32.0, Mx), dtype=jnp.float32)
  rgb_macros = render_macro_tiles_subset(
      splats_n16, macro_gidx, t_ox_m, t_oy_m, use_bf16_mxu=use_bf16_mxu, interpret=interpret
  )
  img_pad = rgb_macros.reshape(My, Mx, 32, 32, 3).transpose(0, 2, 1, 3, 4).reshape(H_pad, W_pad, 3)
  return img_pad[:height, :width, :]


def make_massive_scene_4c_fn(
    mesh: jax.sharding.Mesh,
    height: int,
    width: int,
    cam_fovy: float = 42.0,
    znear: float = 0.05,
    k_macro: int = 4096,
    use_bf16_mxu: bool = False,
    has_nrm: bool = False,
):
  """4-chip shard_map massive-scene renderer: shards 32x32 macro-tiles across 4 TPU v5p chips!"""
  M_SIZE = 32
  H_pad = ((height + M_SIZE - 1) // M_SIZE) * M_SIZE
  W_pad = ((width + M_SIZE - 1) // M_SIZE) * M_SIZE
  My = H_pad // M_SIZE
  Mx = W_pad // M_SIZE
  M_tiles = My * Mx
  n_chips = len(mesh.devices)
  M_pad = ((M_tiles + n_chips - 1) // n_chips) * n_chips

  t_ox_np = np.zeros((M_pad,), dtype=np.float32)
  t_oy_np = np.zeros((M_pad,), dtype=np.float32)
  t_ox_np[:M_tiles] = np.tile(np.arange(Mx, dtype=np.float32) * 32.0, My)
  t_oy_np[:M_tiles] = np.repeat(np.arange(My, dtype=np.float32) * 32.0, Mx)

  P = jax.sharding.PartitionSpec
  sm_kw = {"check_rep": False} if "check_rep" in inspect.signature(shard_map).parameters else {"check_vma": False}

  def _render_macros_local(splats_n16, macro_gidx_sub, tox_sub, toy_sub):
    return render_macro_tiles_subset(
        splats_n16, macro_gidx_sub, tox_sub, toy_sub, use_bf16_mxu=use_bf16_mxu, interpret=False
    )

  _sm_raster = shard_map(
      _render_macros_local,
      mesh=mesh,
      in_specs=(P(), P("dp"), P("dp"), P("dp")),
      out_specs=P("dp"),
      **sm_kw,
  )

  t_ox_dev = jax.device_put(jnp.asarray(t_ox_np), jax.sharding.NamedSharding(mesh, P("dp")))
  t_oy_dev = jax.device_put(jnp.asarray(t_oy_np), jax.sharding.NamedSharding(mesh, P("dp")))

  @jax.jit
  def _run_4c(mu, cov_in, rgb, opacity, cam_xpos, cam_xmat, nrm=None):
    splats_n16, macro_gidx, _, _, _, _, _ = project_and_bucket_macro_tiles(
        mu, cov_in, rgb, opacity, cam_xpos, cam_xmat,
        height=height, width=width, nrm=nrm, cam_fovy=cam_fovy, znear=znear, k_macro=k_macro,
    )
    if M_pad > M_tiles:
      macro_gidx = jnp.pad(macro_gidx, ((0, M_pad - M_tiles), (0, 0)), constant_values=-1)
    rgb_macros = _sm_raster(splats_n16, macro_gidx, t_ox_dev, t_oy_dev)[:M_tiles]
    img_pad = rgb_macros.reshape(My, Mx, 32, 32, 3).transpose(0, 2, 1, 3, 4).reshape(H_pad, W_pad, 3)
    return img_pad[:height, :width, :]

  return _run_4c

