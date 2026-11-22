"""Stage 1 Exact Analytic Raycaster & G-Buffer Renderer for MJX on TPU."""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from tpu_mjx_render.camera import compute_camera_rays
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


class GBuffer(NamedTuple):
  """Exact Stage 1 G-Buffer tensors (all in TPU device memory, zero host copies)."""

  rgb: jax.Array            # (..., H, W, 3) float32 [0, 1]
  albedo: jax.Array         # (..., H, W, 3) float32 [0, 1]
  depth: jax.Array          # (..., H, W) float32 (Euclidean dist from camera, -1 on miss)
  z_depth: jax.Array        # (..., H, W) float32 (camera -Z depth, -1 on miss)
  normal: jax.Array         # (..., H, W, 3) float32 world normal (0 on miss)
  view_dir: jax.Array       # (..., H, W, 3) float32 unit dir toward camera (-ray_dir)
  world_pos: jax.Array      # (..., H, W, 3) float32 hit position
  geom_id: jax.Array        # (..., H, W) int32 (-1 on miss)
  direct_shadow: jax.Array  # (..., H, W) float32 (1.0 lit, shadow_frac occluded)
  direct_rgb: jax.Array     # (..., H, W, 3) float32 direct diffuse+specular irradiance


@dataclasses.dataclass(frozen=True)
class RenderSceneSpec:
  """Static compile-time scene descriptor extracted from mujoco.MjModel."""

  ngeom: int
  nlight: int
  cam_id: int
  height: int
  width: int
  cam_projection: int
  cam_fovy: float
  cam_sensorsize: Tuple[float, float]
  cam_intrinsic: Tuple[float, float, float, float]
  znear: float
  zfar: float
  use_shadows: bool
  shadow_light_fraction: float
  enable_specular: bool

  # Enabled visible geom indices partitioned statically by MuJoCo geom type
  enabled_geom_ids: Tuple[int, ...]
  type_groups: Tuple[Tuple[int, Tuple[int, ...]], ...]

  # Static arrays (kept in JAX constant/closure or passed directly)
  geom_type: np.ndarray        # (ngeom,) int32
  geom_size: np.ndarray        # (ngeom, 3) float32
  geom_rgba: np.ndarray        # (ngeom, 4) float32 (resolved material fallback)
  geom_spec: np.ndarray        # (ngeom,) float32
  geom_shin: np.ndarray        # (ngeom,) float32
  geom_emit: np.ndarray        # (ngeom,) float32
  geom_is_checker: np.ndarray  # (ngeom,) bool

  # Checker & skybox colors
  checker_rgb1: Tuple[float, float, float]
  checker_rgb2: Tuple[float, float, float]
  checker_markrgb: Tuple[float, float, float]
  checker_repeat: Tuple[float, float]
  sky_rgb1: Tuple[float, float, float]
  sky_rgb2: Tuple[float, float, float]

  # Light static properties
  light_type: np.ndarray        # (nlight,) int32
  light_castshadow: np.ndarray  # (nlight,) bool
  light_active: np.ndarray      # (nlight,) bool
  light_bodyid: np.ndarray      # (nlight,) int32
  light_pos: np.ndarray         # (nlight, 3) float32
  light_dir: np.ndarray         # (nlight, 3) float32
  light_attenuation: np.ndarray # (nlight, 3) float32
  light_cutoff_rad: np.ndarray  # (nlight,) float32
  light_exponent: np.ndarray    # (nlight,) float32
  light_ambient: np.ndarray     # (nlight, 3) float32
  light_diffuse: np.ndarray     # (nlight, 3) float32
  light_specular: np.ndarray    # (nlight, 3) float32
  headlight_active: bool
  headlight_ambient: Tuple[float, float, float]
  headlight_diffuse: Tuple[float, float, float]
  headlight_specular: Tuple[float, float, float]

  @classmethod
  def from_mjmodel(
      cls,
      m: Any,
      cam_id: int = 0,
      height: int = 64,
      width: int = 64,
      use_shadows: bool = True,
      shadow_light_fraction: float = 0.0,
      enable_specular: bool = True,
      enabled_geom_groups: Tuple[int, ...] = (0, 1, 2, 3),
      znear: Optional[float] = None,
      zfar: Optional[float] = None,
  ) -> "RenderSceneSpec":
    ngeom = int(m.ngeom)
    nlight = int(m.nlight)

    geom_type = np.asarray(m.geom_type, dtype=np.int32)
    geom_size = np.asarray(m.geom_size, dtype=np.float32)
    geom_matid = np.asarray(m.geom_matid, dtype=np.int32)
    geom_rgba_raw = np.asarray(m.geom_rgba, dtype=np.float32)
    geom_group = np.asarray(m.geom_group, dtype=np.int32)

    mat_rgba = np.asarray(m.mat_rgba, dtype=np.float32) if m.nmat > 0 else np.zeros((0, 4), np.float32)
    mat_spec = np.asarray(m.mat_specular, dtype=np.float32) if m.nmat > 0 else np.zeros((0,), np.float32)
    mat_shin = np.asarray(m.mat_shininess, dtype=np.float32) if m.nmat > 0 else np.zeros((0,), np.float32)
    mat_emit = np.asarray(m.mat_emission, dtype=np.float32) if m.nmat > 0 else np.zeros((0,), np.float32)
    mat_texid = np.asarray(m.mat_texid, dtype=np.int32) if m.nmat > 0 else np.zeros((0, 10), np.int32)

    resolved_rgba = geom_rgba_raw.copy()
    resolved_spec = np.full((ngeom,), 0.3, dtype=np.float32)
    resolved_shin = np.full((ngeom,), 0.3, dtype=np.float32)
    resolved_emit = np.zeros((ngeom,), dtype=np.float32)
    is_checker = np.zeros((ngeom,), dtype=bool)

    for g in range(ngeom):
      mid = int(geom_matid[g])
      if mid >= 0:
        if np.allclose(resolved_rgba[g], [0.5, 0.5, 0.5, 1.0]):
          resolved_rgba[g] = mat_rgba[mid]
        resolved_spec[g] = float(mat_spec[mid])
        resolved_shin[g] = float(mat_shin[mid])
        resolved_emit[g] = float(mat_emit[mid])
        if mat_texid.ndim == 2 and mat_texid[mid, 1] >= 0:
          tid = int(mat_texid[mid, 1])
          if int(m.tex_type[tid]) == 0:
            is_checker[g] = True
        elif mat_texid.ndim == 1 and mat_texid[mid] >= 0:
          tid = int(mat_texid[mid])
          if int(m.tex_type[tid]) == 0:
            is_checker[g] = True

    # Identify visible analytic geoms
    enabled = []
    groups_map: Dict[int, list[int]] = {}
    for g in range(ngeom):
      gt = int(geom_type[g])
      if gt not in RAY_FN_BY_TYPE:
        continue
      if int(geom_group[g]) not in enabled_geom_groups:
        continue
      if float(resolved_rgba[g, 3]) <= 0.0:
        continue
      enabled.append(g)
      groups_map.setdefault(gt, []).append(g)

    type_groups = tuple((gt, tuple(ids)) for gt, ids in sorted(groups_map.items()))

    # Extract checker/sky colors from MuJoCo baked tex_data
    checker_rgb1 = (0.24, 0.32, 0.40)
    checker_rgb2 = (0.14, 0.20, 0.28)
    checker_mark = (0.75, 0.78, 0.82)
    checker_rep = (1.0, 1.0)
    sky_rgb1 = (0.28, 0.46, 0.68)
    sky_rgb2 = (0.05, 0.08, 0.12)
    for t in range(int(m.ntex)):
      adr = int(m.tex_adr[t])
      th = int(m.tex_height[t])
      tw = int(m.tex_width[t])
      nc = int(m.tex_nchannel[t])
      if adr < 0 or th <= 0 or tw <= 0 or nc < 3:
        continue
      arr = np.asarray(m.tex_data[adr : adr + th * tw * nc], dtype=np.float32).reshape(th, tw, nc)[..., :3] / 255.0
      if int(m.tex_type[t]) == 2:  # skybox
        sky_rgb1 = tuple(float(x) for x in arr[0, 0])
        sky_rgb2 = tuple(float(x) for x in arr[-1, 0])
      elif int(m.tex_type[t]) == 0:  # 2D checker
        checker_rgb1 = tuple(float(x) for x in arr[th // 4, tw // 4])
        checker_rgb2 = tuple(float(x) for x in arr[th // 4, (3 * tw) // 4])
        checker_mark = tuple(float(x) for x in arr[0, 0])

    for g in range(ngeom):
      if is_checker[g] and geom_matid[g] >= 0:
        rep = m.mat_texrepeat[int(geom_matid[g])]
        checker_rep = (float(rep[0]), float(rep[1]))
        break

    extent = max(float(m.stat.extent), 1e-3)
    znear_val = float(m.vis.map.znear * extent) if znear is None else float(znear)
    zfar_val = float(m.vis.map.zfar * extent) if zfar is None else float(zfar)
    znear_val = max(znear_val, 1e-3)

    hl = m.vis.headlight
    return cls(
        ngeom=ngeom,
        nlight=nlight,
        cam_id=int(cam_id),
        height=int(height),
        width=int(width),
        cam_projection=int(m.cam_projection[cam_id]),
        cam_fovy=float(m.cam_fovy[cam_id]),
        cam_sensorsize=(float(m.cam_sensorsize[cam_id, 0]), float(m.cam_sensorsize[cam_id, 1])),
        cam_intrinsic=tuple(float(x) for x in m.cam_intrinsic[cam_id]),
        znear=znear_val,
        zfar=zfar_val,
        use_shadows=bool(use_shadows),
        shadow_light_fraction=float(shadow_light_fraction),
        enable_specular=bool(enable_specular),
        enabled_geom_ids=tuple(enabled),
        type_groups=type_groups,
        geom_type=geom_type,
        geom_size=geom_size,
        geom_rgba=resolved_rgba,
        geom_spec=resolved_spec,
        geom_shin=resolved_shin,
        geom_emit=resolved_emit,
        geom_is_checker=is_checker,
        checker_rgb1=checker_rgb1,
        checker_rgb2=checker_rgb2,
        checker_markrgb=checker_mark,
        checker_repeat=checker_rep,
        sky_rgb1=sky_rgb1,
        sky_rgb2=sky_rgb2,
        light_type=np.asarray(m.light_type, dtype=np.int32),
        light_castshadow=np.asarray(m.light_castshadow, dtype=bool),
        light_active=np.asarray(m.light_active, dtype=bool),
        light_bodyid=np.asarray(m.light_bodyid, dtype=np.int32),
        light_pos=np.asarray(m.light_pos, dtype=np.float32),
        light_dir=np.asarray(m.light_dir, dtype=np.float32),
        light_attenuation=np.asarray(m.light_attenuation, dtype=np.float32),
        light_cutoff_rad=np.asarray(m.light_cutoff, dtype=np.float32) * (np.pi / 180.0),
        light_exponent=np.asarray(m.light_exponent, dtype=np.float32),
        light_ambient=np.asarray(m.light_ambient, dtype=np.float32),
        light_diffuse=np.asarray(m.light_diffuse, dtype=np.float32),
        light_specular=np.asarray(m.light_specular, dtype=np.float32),
        headlight_active=bool(hl.active),
        headlight_ambient=tuple(float(x) for x in hl.ambient),
        headlight_diffuse=tuple(float(x) for x in hl.diffuse),
        headlight_specular=tuple(float(x) for x in hl.specular),
    )


def _cast_rays_grouped(
    spec: RenderSceneSpec,
    geom_xpos: jax.Array,
    geom_xmat: jax.Array,
    geom_size: jax.Array,
    ray_orig: jax.Array,
    ray_dir: jax.Array,
    max_dist: jax.Array | float,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
  """Cast (R, 3) rays against all static-grouped primitives.

  Returns:
    best_dist: (R,) float32 (-1 on miss)
    best_nrm:  (R, 3) float32
    best_gid:  (R,) int32 (-1 on miss)
  """
  R = ray_orig.shape[0]
  best_dist = jnp.broadcast_to(jnp.asarray(max_dist, dtype=jnp.float32), (R,))
  best_gid = jnp.full((R,), -1, dtype=jnp.int32)
  best_nrm = jnp.zeros((R, 3), dtype=jnp.float32)

  for gt, gid_tuple in spec.type_groups:
    idx = np.asarray(gid_tuple, dtype=np.int32)
    g_pos = geom_xpos[idx][None, :, :]    # (1, G, 3)
    g_mat = geom_xmat[idx][None, :, :, :] # (1, G, 3, 3)
    g_sz = geom_size[idx][None, :, :]     # (1, G, 3)

    r_o = ray_orig[:, None, :]            # (R, 1, 3)
    r_d = ray_dir[:, None, :]             # (R, 1, 3)

    fn = RAY_FN_BY_TYPE[gt]
    d_grp, n_grp = fn(g_pos, g_mat, g_sz, r_o, r_d)  # (R, G), (R, G, 3)

    valid = (d_grp >= 0.0) & (d_grp < best_dist[:, None])
    d_masked = jnp.where(valid, d_grp, MJ_MAXVAL)
    local_idx = jnp.argmin(d_masked, axis=1)  # (R,)
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


def _cast_rays_tiled(
    spec: RenderSceneSpec,
    geom_xpos: jax.Array,
    geom_xmat: jax.Array,
    geom_size: jax.Array,
    ray_orig: jax.Array,
    ray_dir: jax.Array,
    max_dist: jax.Array,
    tile_size: int = 128,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
  """Tile-scanned ray casting keeping working buffers bounded to (tile_size, N_g)."""
  R = ray_orig.shape[0]
  pad = (tile_size - (R % tile_size)) % tile_size
  if pad > 0:
    ray_orig = jnp.pad(ray_orig, ((0, pad), (0, 0)))
    ray_dir = jnp.pad(ray_dir, ((0, pad), (0, 0)), constant_values=1.0 / np.sqrt(3.0))
    max_dist = jnp.pad(max_dist, ((0, pad),), constant_values=1.0)

  n_tiles = ray_orig.shape[0] // tile_size
  ro_t = ray_orig.reshape(n_tiles, tile_size, 3)
  rd_t = ray_dir.reshape(n_tiles, tile_size, 3)
  md_t = max_dist.reshape(n_tiles, tile_size)

  def _scan_body(carry, xs):
    ro_b, rd_b, md_b = xs
    d_b, n_b, g_b = _cast_rays_grouped(spec, geom_xpos, geom_xmat, geom_size, ro_b, rd_b, md_b)
    return carry, (d_b, n_b, g_b)

  _, (d_all, n_all, g_all) = jax.lax.scan(_scan_body, None, (ro_t, rd_t, md_t))
  d_flat = d_all.reshape(-1)[:R]
  n_flat = n_all.reshape(-1, 3)[:R]
  g_flat = g_all.reshape(-1)[:R]
  return d_flat, n_flat, g_flat


def _sample_albedo(
    spec: RenderSceneSpec,
    geom_rgba: jax.Array,
    geom_id: jax.Array,
    hit_pos: jax.Array,
    geom_xpos: jax.Array,
    geom_xmat: jax.Array,
) -> jax.Array:
  """Evaluate base albedo with MuJoCo procedural checker on checker-textured planes."""
  safe_gid = jnp.maximum(geom_id, 0)
  base = geom_rgba[safe_gid, :3]
  is_chk = jnp.asarray(spec.geom_is_checker, dtype=jnp.bool_)[safe_gid]

  # Local coordinates on plane
  g_pos = geom_xpos[safe_gid]
  g_mat = geom_xmat[safe_gid]
  local_p = jnp.einsum("nji,nj->ni", g_mat, hit_pos - g_pos, precision=jax.lax.Precision.HIGHEST)
  rep_x, rep_y = spec.checker_repeat
  ux = local_p[:, 0] * rep_x
  uy = local_p[:, 1] * rep_y
  cx = jnp.floor(ux).astype(jnp.int32)
  cy = jnp.floor(uy).astype(jnp.int32)
  even = ((cx + cy) & 1) == 0
  rgb1 = jnp.asarray(spec.checker_rgb1, dtype=jnp.float32)
  rgb2 = jnp.asarray(spec.checker_rgb2, dtype=jnp.float32)
  chk_rgb = jnp.where(even[:, None], rgb1[None, :], rgb2[None, :])
  return jnp.where((geom_id >= 0)[:, None] & is_chk[:, None], chk_rgb, base)


def _shade_scene(
    spec: RenderSceneSpec,
    geom_xpos: jax.Array,
    geom_xmat: jax.Array,
    geom_size: jax.Array,
    geom_rgba: jax.Array,
    cam_pos: jax.Array,
    cam_mat: jax.Array,
    light_xpos: jax.Array,
    light_xdir: jax.Array,
    ray_orig: jax.Array,
    ray_dir: jax.Array,
    neg_z: jax.Array,
    dist: jax.Array,
    normal: jax.Array,
    geom_id: jax.Array,
    mode: str,
) -> GBuffer:
  H, W = spec.height, spec.width
  R = H * W
  hit = geom_id >= 0
  safe_gid = jnp.maximum(geom_id, 0)

  hit_pos = ray_orig + jnp.where(hit, dist, 0.0)[:, None] * ray_dir
  view_dir = -ray_dir
  albedo = _sample_albedo(spec, geom_rgba, geom_id, hit_pos, geom_xpos, geom_xmat)

  spec_coef = jnp.asarray(spec.geom_spec, dtype=jnp.float32)[safe_gid]
  shin_exp = jnp.asarray(spec.geom_shin, dtype=jnp.float32)[safe_gid] * 128.0
  emit_coef = jnp.asarray(spec.geom_emit, dtype=jnp.float32)[safe_gid]

  diff_accum = jnp.zeros((R, 3), dtype=jnp.float32)
  spec_accum = jnp.zeros((R, 3), dtype=jnp.float32)
  shadow_vis_accum = jnp.ones((R,), dtype=jnp.float32)

  # Scene lights
  for l_idx in range(spec.nlight):
    if not bool(spec.light_active[l_idx]):
      continue
    l_type = int(spec.light_type[l_idx])
    l_pos = light_xpos[l_idx]
    l_dir = light_xdir[l_idx]
    l_dir = l_dir / jnp.maximum(jnp.linalg.norm(l_dir), 1e-12)
    l_amb = jnp.asarray(spec.light_ambient[l_idx], dtype=jnp.float32)
    l_diff = jnp.asarray(spec.light_diffuse[l_idx], dtype=jnp.float32)
    l_spec = jnp.asarray(spec.light_specular[l_idx], dtype=jnp.float32)
    l_att = jnp.asarray(spec.light_attenuation[l_idx], dtype=jnp.float32)

    diff_accum = diff_accum + l_amb[None, :]
    if l_type == 1:  # directional
      L = jnp.broadcast_to(-l_dir[None, :], (R, 3))
      dist_to_light = jnp.full((R,), 1.0e8, dtype=jnp.float32)
      attenuation = jnp.ones((R,), dtype=jnp.float32)
    else:
      delta = l_pos[None, :] - hit_pos
      dist_to_light = jnp.linalg.norm(delta, axis=-1)
      L = delta / jnp.maximum(dist_to_light[:, None], 1e-12)
      att_denom = l_att[0] + l_att[1] * dist_to_light + l_att[2] * dist_to_light * dist_to_light
      attenuation = jnp.where(att_denom > 1e-12, 1.0 / jnp.maximum(att_denom, 1e-12), 1.0)
      if l_type == 0:  # spot
        cos_theta = jnp.sum(-L * l_dir[None, :], axis=-1)
        cos_cut = float(np.cos(spec.light_cutoff_rad[l_idx]))
        exp_val = float(spec.light_exponent[l_idx])
        in_cone = cos_theta >= cos_cut
        attenuation = jnp.where(in_cone, attenuation * (jnp.maximum(cos_theta, 0.0) ** exp_val), 0.0)

    ndotl = jnp.maximum(0.0, jnp.sum(normal * L, axis=-1))

    vis = jnp.ones((R,), dtype=jnp.float32)
    if spec.use_shadows and bool(spec.light_castshadow[l_idx]):
      sh_orig = hit_pos + normal * 1.0e-4
      sh_max = jnp.where(hit & (ndotl > 0.0), jnp.maximum(dist_to_light - 1.0e-3, 0.0), 0.0)
      if mode == "naive":
        sh_d, _, sh_gid = _cast_rays_grouped(spec, geom_xpos, geom_xmat, geom_size, sh_orig, L, sh_max)
      else:
        sh_d, _, sh_gid = _cast_rays_tiled(spec, geom_xpos, geom_xmat, geom_size, sh_orig, L, sh_max)
      del sh_d
      occluded = hit & (ndotl > 0.0) & (sh_gid >= 0)
      vis = jnp.where(occluded, float(spec.shadow_light_fraction), 1.0)
      if l_idx == 0:
        shadow_vis_accum = vis

    weight = attenuation * vis
    diff_accum = diff_accum + l_diff[None, :] * (ndotl * weight)[:, None]
    if spec.enable_specular:
      H_vec = L + view_dir
      H_vec = H_vec / jnp.maximum(jnp.linalg.norm(H_vec, axis=-1, keepdims=True), 1e-12)
      ndoth = jnp.maximum(0.0, jnp.sum(normal * H_vec, axis=-1))
      spec_w = jnp.where((spec_coef > 0.0) & (shin_exp > 0.0), spec_coef * (ndoth ** shin_exp) * weight, 0.0)
      spec_accum = spec_accum + l_spec[None, :] * spec_w[:, None]

  # Headlight (no shadow ray, attached to camera)
  if spec.headlight_active:
    hl_amb = jnp.asarray(spec.headlight_ambient, dtype=jnp.float32)
    hl_diff = jnp.asarray(spec.headlight_diffuse, dtype=jnp.float32)
    hl_spec = jnp.asarray(spec.headlight_specular, dtype=jnp.float32)
    L_hl = -ray_dir
    ndotl_hl = jnp.maximum(0.0, jnp.sum(normal * L_hl, axis=-1))
    diff_accum = diff_accum + hl_amb[None, :] + hl_diff[None, :] * ndotl_hl[:, None]
    if spec.enable_specular:
      ndoth_hl = ndotl_hl
      spec_w_hl = jnp.where((spec_coef > 0.0) & (shin_exp > 0.0), spec_coef * (ndoth_hl ** shin_exp), 0.0)
      spec_accum = spec_accum + hl_spec[None, :] * spec_w_hl[:, None]

  direct_rgb = albedo * diff_accum + spec_accum + albedo * emit_coef[:, None]
  direct_rgb = jnp.clip(direct_rgb, 0.0, 1.0)

  # Skybox gradient on miss
  sky_t = jnp.clip(0.5 * (ray_dir[:, 2] + 1.0), 0.0, 1.0)[:, None]
  sky_rgb = sky_t * jnp.asarray(spec.sky_rgb1, dtype=jnp.float32)[None, :] + (1.0 - sky_t) * jnp.asarray(spec.sky_rgb2, dtype=jnp.float32)[None, :]
  rgb = jnp.where(hit[:, None], direct_rgb, sky_rgb)

  inv_cos = 1.0 / neg_z
  euclid_depth = jnp.where(hit, dist + spec.znear * inv_cos, -1.0)
  z_depth = jnp.where(hit, spec.znear + dist * neg_z, -1.0)

  return GBuffer(
      rgb=rgb.reshape(H, W, 3),
      albedo=jnp.where(hit[:, None], albedo, sky_rgb).reshape(H, W, 3),
      depth=euclid_depth.reshape(H, W),
      z_depth=z_depth.reshape(H, W),
      normal=jnp.where(hit[:, None], normal, 0.0).reshape(H, W, 3),
      view_dir=view_dir.reshape(H, W, 3),
      world_pos=jnp.where(hit[:, None], hit_pos, 0.0).reshape(H, W, 3),
      geom_id=geom_id.reshape(H, W),
      direct_shadow=jnp.where(hit, shadow_vis_accum, 1.0).reshape(H, W),
      direct_rgb=jnp.where(hit[:, None], direct_rgb, sky_rgb).reshape(H, W, 3),
  )


def render_exact_gbuffer(
    spec: RenderSceneSpec,
    mjx_data: Any,
    geom_rgba_override: Optional[jax.Array] = None,
    mode: str = "tiled",
    pallas_interpret: bool = True,
) -> GBuffer:
  """Render a single world's exact G-Buffer directly from mjx.Data (zero-copy)."""
  cam_pos = mjx_data.cam_xpos[spec.cam_id]
  cam_mat = mjx_data.cam_xmat[spec.cam_id]
  geom_xpos = mjx_data.geom_xpos
  geom_xmat = mjx_data.geom_xmat
  if spec.nlight > 0:
    l_bid = np.asarray(spec.light_bodyid, dtype=np.int32)
    l_pos0 = jnp.asarray(spec.light_pos, dtype=jnp.float32)
    l_dir0 = jnp.asarray(spec.light_dir, dtype=jnp.float32)
    light_xpos = mjx_data.xpos[l_bid] + jnp.einsum("nij,nj->ni", mjx_data.xmat[l_bid], l_pos0, precision=jax.lax.Precision.HIGHEST)
    light_xdir = jnp.einsum("nij,nj->ni", mjx_data.xmat[l_bid], l_dir0, precision=jax.lax.Precision.HIGHEST)
  else:
    light_xpos = jnp.zeros((0, 3), dtype=jnp.float32)
    light_xdir = jnp.zeros((0, 3), dtype=jnp.float32)

  geom_size = jnp.asarray(spec.geom_size, dtype=jnp.float32)
  geom_rgba = jnp.asarray(spec.geom_rgba, dtype=jnp.float32) if geom_rgba_override is None else geom_rgba_override

  ray_orig, ray_dir, neg_z = compute_camera_rays(
      cam_pos=cam_pos,
      cam_mat=cam_mat,
      fovy_deg=spec.cam_fovy,
      height=spec.height,
      width=spec.width,
      projection=spec.cam_projection,
      sensorsize=spec.cam_sensorsize,
      intrinsic=spec.cam_intrinsic,
      znear=spec.znear,
  )
  max_dist = (spec.zfar - spec.znear) / neg_z if spec.zfar > 0.0 else jnp.full_like(neg_z, MJ_MAXVAL)

  if mode == "pallas":
    from tpu_mjx_render.pallas_render import pallas_raycast_scene
    dist, normal, geom_id = pallas_raycast_scene(
        spec, geom_xpos, geom_xmat, geom_size, ray_orig, ray_dir, max_dist, interpret=pallas_interpret
    )
  elif mode == "naive":
    dist, normal, geom_id = _cast_rays_grouped(
        spec, geom_xpos, geom_xmat, geom_size, ray_orig, ray_dir, max_dist
    )
  else:
    dist, normal, geom_id = _cast_rays_tiled(
        spec, geom_xpos, geom_xmat, geom_size, ray_orig, ray_dir, max_dist
    )

  return _shade_scene(
      spec=spec,
      geom_xpos=geom_xpos,
      geom_xmat=geom_xmat,
      geom_size=geom_size,
      geom_rgba=geom_rgba,
      cam_pos=cam_pos,
      cam_mat=cam_mat,
      light_xpos=light_xpos,
      light_xdir=light_xdir,
      ray_orig=ray_orig,
      ray_dir=ray_dir,
      neg_z=neg_z,
      dist=dist,
      normal=normal,
      geom_id=geom_id,
      mode=mode,
  )


def render_batch(
    spec: RenderSceneSpec,
    mjx_data_batch: Any,
    geom_rgba_batch: Optional[jax.Array] = None,
    mode: str = "tiled",
    pallas_interpret: bool = True,
) -> GBuffer:
  """Zero-copy batched rendering across B parallel MJX worlds via SoA or jax.vmap."""
  if mode == "v2_soa":
    from tpu_mjx_render.pallas_v2 import raycast_batch_soa
    return raycast_batch_soa(spec, mjx_data_batch, geom_rgba_batch=geom_rgba_batch)
  if mode == "pallas_v2":
    from tpu_mjx_render.pallas_v2 import pallas_raycast_batch_v2
    return pallas_raycast_batch_v2(
        spec, mjx_data_batch, geom_rgba_batch=geom_rgba_batch, interpret=pallas_interpret
    )
  in_ax_rgba = 0 if geom_rgba_batch is not None else None
  return jax.vmap(
      lambda d, rgba: render_exact_gbuffer(
          spec, d, geom_rgba_override=rgba, mode=mode, pallas_interpret=pallas_interpret
      ),
      in_axes=(0, in_ax_rgba),
  )(mjx_data_batch, geom_rgba_batch)
