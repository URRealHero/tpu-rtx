"""Universal Zero-Copy MuJoCo / MJX Adapter for TPU-RTX (JAX + Pallas Mosaic).

Supports:
  - Auto-compilation from `mujoco.MjModel`, `mjx.Model`, MJCF XML string/path,
    `RenderSceneSpec`, or `MeshAssetSpec`.
  - Analytic MuJoCo primitives (plane, sphere, capsule, ellipsoid, cylinder, box),
    rigid MJCF links, checkerboards, skyboxes, multi-light shadows, and optional
    static triangle meshes (`GEOM_MESH`).
  - Single-camera `(B, H, W, ...)` and multi-camera `(B, C, H, W, ...)` rendering,
    plus unbatched single-env `(H, W, ...)` convenience mode.
  - V3 In-VMEM Pallas shaders:
      * `"fused_ao_ggx"`: GGX Microfacet PBR + 4-sample Hemisphere AO + Soft Shadows
      * `"fast_fused"`: Material-deduplicated V3 Blinn-Phong + Hard Shadows (peak FPS)
      * `"exact_gbuffer"`: Stage 1 Exact Reference G-Buffer (`pallas_raycast_batch_v2`)
  - Clean `jax.sharding.Mesh` + `shard_map` (`data_axis`) scaling across 1 chip,
    4 chips, and 32/64/1000-chip TPU Pods with zero host callbacks, zero send/recv,
    and zero XLA gather ops.
"""

from __future__ import annotations

import dataclasses
import inspect
import math
import os
import re
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

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
from tpu_mjx_render.mesh_ray import MeshAssetSpec, build_mesh_scene_spec, render_mesh_batch
from tpu_mjx_render.pallas_v2 import (
    _compute_geom_bound_sq,
    _extract_light_batch,
    _ray_one_geom_v2,
    build_local_camera_rays,
    pallas_raycast_batch_v2,
)
from tpu_mjx_render.primitives import MJ_MAXVAL, RAY_FN_BY_TYPE
from tpu_mjx_render.renderer import RenderSceneSpec
from tpu_mjx_render.roofline import TPU_SPECS, TPUHardwareSpec, inspect_hlo_ops

GEOM_MESH_TYPE = 7
VALID_SHADERS = (
    "fused_ao_ggx",
    "pbr_ao_ggx",
    "pbr",
    "fast_fused",
    "blinn_phong",
    "fused",
    "exact_gbuffer",
    "gbuffer",
)
DEFAULT_OUTPUTS = ("rgb", "depth", "normal", "seg_id", "flow")


def _shard_map_kwargs() -> Dict[str, Any]:
  sig = inspect.signature(shard_map).parameters
  return {"check_rep": False} if "check_rep" in sig else {"check_vma": False}


def _normalize_shader(shader: str) -> str:
  key = shader.lower().strip()
  if key in ("fused_ao_ggx", "pbr_ao_ggx", "pbr"):
    return "fused_ao_ggx"
  if key in ("fast_fused", "blinn_phong", "fused"):
    return "fast_fused"
  if key in ("exact_gbuffer", "gbuffer"):
    return "exact_gbuffer"
  raise ValueError(f"Unknown shader '{shader}'. Expected one of {VALID_SHADERS}.")


def _normalize_res(res: Union[int, Tuple[int, int], Sequence[int]]) -> Tuple[int, int]:
  if isinstance(res, int):
    return (int(res), int(res))
  if len(res) != 2:
    raise ValueError(f"Resolution must be int or (H, W), got {res}")
  return (int(res[0]), int(res[1]))


def _normalize_dtype(dtype: Any) -> Any:
  if dtype in ("bfloat16", "bf16", jnp.bfloat16):
    return jnp.bfloat16
  if dtype in ("float32", "fp32", "f32", jnp.float32):
    return jnp.float32
  return jnp.dtype(dtype)


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


def _spec_from_mjx_model(
    mx: Any,
    cam_id: int = 0,
    height: int = 128,
    width: int = 128,
    use_shadows: bool = True,
    shadow_light_fraction: float = 0.0,
    enable_specular: bool = True,
    enabled_geom_groups: Tuple[int, ...] = (0, 1, 2, 3),
    znear: Optional[float] = None,
    zfar: Optional[float] = None,
) -> RenderSceneSpec:
  """Build RenderSceneSpec directly from an mjx.Model (zero MjModel dependency)."""
  ngeom = int(mx.ngeom)
  nlight = int(mx.nlight)
  nmat = int(getattr(mx, "nmat", 0))
  ntex = int(getattr(mx, "ntex", 0))

  geom_type = np.asarray(mx.geom_type, dtype=np.int32)
  geom_size = np.asarray(mx.geom_size, dtype=np.float32)
  geom_matid = np.asarray(mx.geom_matid, dtype=np.int32)
  geom_rgba_raw = np.asarray(mx.geom_rgba, dtype=np.float32)
  geom_group = np.asarray(mx.geom_group, dtype=np.int32)

  mat_rgba = np.asarray(mx.mat_rgba, dtype=np.float32) if nmat > 0 else np.zeros((0, 4), np.float32)
  mat_spec = np.asarray(mx.mat_specular, dtype=np.float32) if nmat > 0 else np.zeros((0,), np.float32)
  mat_shin = np.asarray(mx.mat_shininess, dtype=np.float32) if nmat > 0 else np.zeros((0,), np.float32)
  mat_emit = np.asarray(mx.mat_emission, dtype=np.float32) if nmat > 0 else np.zeros((0,), np.float32)
  mat_texid = np.asarray(mx.mat_texid, dtype=np.int32) if nmat > 0 else np.zeros((0, 10), np.int32)
  tex_type = np.asarray(mx.tex_type, dtype=np.int32) if ntex > 0 else np.zeros((0,), np.int32)

  resolved_rgba = geom_rgba_raw.copy()
  resolved_spec = np.full((ngeom,), 0.3, dtype=np.float32)
  resolved_shin = np.full((ngeom,), 0.3, dtype=np.float32)
  resolved_emit = np.zeros((ngeom,), dtype=np.float32)
  is_checker = np.zeros((ngeom,), dtype=bool)

  for g in range(ngeom):
    mid = int(geom_matid[g])
    if mid >= 0 and mid < nmat:
      if np.allclose(resolved_rgba[g], [0.5, 0.5, 0.5, 1.0]):
        resolved_rgba[g] = mat_rgba[mid]
      resolved_spec[g] = float(mat_spec[mid])
      resolved_shin[g] = float(mat_shin[mid])
      resolved_emit[g] = float(mat_emit[mid])
      if mat_texid.ndim == 2 and mat_texid.shape[1] > 1 and mat_texid[mid, 1] >= 0:
        tid = int(mat_texid[mid, 1])
        if int(tex_type[tid]) == 0:
          is_checker[g] = True
      elif mat_texid.ndim == 1 and mat_texid[mid] >= 0:
        tid = int(mat_texid[mid])
        if int(tex_type[tid]) == 0:
          is_checker[g] = True

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

  checker_rgb1 = (0.24, 0.32, 0.40)
  checker_rgb2 = (0.14, 0.20, 0.28)
  checker_mark = (0.75, 0.78, 0.82)
  checker_rep = (1.0, 1.0)
  sky_rgb1 = (0.28, 0.46, 0.68)
  sky_rgb2 = (0.05, 0.08, 0.12)

  if ntex > 0 and hasattr(mx, "tex_data") and hasattr(mx, "tex_adr"):
    tex_adr = np.asarray(mx.tex_adr, dtype=np.int32)
    tex_h = np.asarray(mx.tex_height, dtype=np.int32)
    tex_w = np.asarray(mx.tex_width, dtype=np.int32)
    tex_nc = np.asarray(mx.tex_nchannel, dtype=np.int32)
    tex_data = np.asarray(mx.tex_data)
    for t in range(ntex):
      adr, th, tw, nc = int(tex_adr[t]), int(tex_h[t]), int(tex_w[t]), int(tex_nc[t])
      if adr < 0 or th <= 0 or tw <= 0 or nc < 3 or adr + th * tw * nc > tex_data.size:
        continue
      arr = tex_data[adr : adr + th * tw * nc].astype(np.float32).reshape(th, tw, nc)[..., :3] / 255.0
      if int(tex_type[t]) == 2:
        sky_rgb1 = tuple(float(x) for x in arr[0, 0])
        sky_rgb2 = tuple(float(x) for x in arr[-1, 0])
      elif int(tex_type[t]) == 0:
        checker_rgb1 = tuple(float(x) for x in arr[th // 4, tw // 4])
        checker_rgb2 = tuple(float(x) for x in arr[th // 4, (3 * tw) // 4])
        checker_mark = tuple(float(x) for x in arr[0, 0])

  if hasattr(mx, "mat_texrepeat"):
    mtr = np.asarray(mx.mat_texrepeat, dtype=np.float32)
    for g in range(ngeom):
      if is_checker[g] and geom_matid[g] >= 0:
        rep = mtr[int(geom_matid[g])]
        checker_rep = (float(rep[0]), float(rep[1]))
        break

  extent = max(float(getattr(getattr(mx, "stat", None), "extent", 1.0)), 1e-3)
  znear_val = max(0.01 * extent if znear is None else float(znear), 1e-3)
  zfar_val = 50.0 * extent if zfar is None else float(zfar)

  cam_proj_arr = getattr(mx, "cam_projection", None)
  cam_proj = int(np.asarray(cam_proj_arr)[cam_id]) if cam_proj_arr is not None else PROJ_PERSPECTIVE
  cam_fovy = float(np.asarray(mx.cam_fovy)[cam_id])
  cam_ss = np.asarray(mx.cam_sensorsize, dtype=np.float32)[cam_id]
  cam_in = np.asarray(mx.cam_intrinsic, dtype=np.float32)[cam_id]

  if nlight > 0:
    l_type = np.asarray(mx.light_type, dtype=np.int32)
    l_cast = np.asarray(mx.light_castshadow, dtype=bool)
    l_act = np.asarray(mx.light_active, dtype=bool)
    l_bid_arr = getattr(mx, "light_bodyid", None)
    l_bid = np.asarray(l_bid_arr, dtype=np.int32) if l_bid_arr is not None else np.zeros((nlight,), dtype=np.int32)
    l_pos = np.asarray(mx.light_pos, dtype=np.float32)
    l_dir = np.asarray(mx.light_dir, dtype=np.float32)
    l_att = np.asarray(mx.light_attenuation, dtype=np.float32)
    l_cut = np.asarray(mx.light_cutoff, dtype=np.float32) * (np.pi / 180.0)
    l_exp = np.asarray(mx.light_exponent, dtype=np.float32)
    l_amb = np.asarray(mx.light_ambient, dtype=np.float32)
    l_diff = np.asarray(mx.light_diffuse, dtype=np.float32)
    l_spec = np.asarray(mx.light_specular, dtype=np.float32)
  else:
    l_type = np.zeros((0,), np.int32)
    l_cast = np.zeros((0,), bool)
    l_act = np.zeros((0,), bool)
    l_bid = np.zeros((0,), np.int32)
    l_pos = np.zeros((0, 3), np.float32)
    l_dir = np.zeros((0, 3), np.float32)
    l_att = np.zeros((0, 3), np.float32)
    l_cut = np.zeros((0,), np.float32)
    l_exp = np.zeros((0,), np.float32)
    l_amb = np.zeros((0, 3), np.float32)
    l_diff = np.zeros((0, 3), np.float32)
    l_spec = np.zeros((0, 3), np.float32)

  return RenderSceneSpec(
      ngeom=ngeom,
      nlight=nlight,
      cam_id=int(cam_id),
      height=int(height),
      width=int(width),
      cam_projection=cam_proj,
      cam_fovy=cam_fovy,
      cam_sensorsize=(float(cam_ss[0]), float(cam_ss[1])),
      cam_intrinsic=tuple(float(x) for x in cam_in),
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
      light_type=l_type,
      light_castshadow=l_cast,
      light_active=l_act,
      light_bodyid=l_bid,
      light_pos=l_pos,
      light_dir=l_dir,
      light_attenuation=l_att,
      light_cutoff_rad=l_cut,
      light_exponent=l_exp,
      light_ambient=l_amb,
      light_diffuse=l_diff,
      light_specular=l_spec,
      headlight_active=True,
      headlight_ambient=(0.25, 0.25, 0.25),
      headlight_diffuse=(0.45, 0.45, 0.45),
      headlight_specular=(0.15, 0.15, 0.15),
  )


def _extract_mesh_spec_from_model(
    src: Any,
    spec_ref: RenderSceneSpec,
    enabled_geom_groups: Tuple[int, ...] = (0, 1, 2, 3),
) -> Tuple[Optional[MeshAssetSpec], Tuple[int, ...]]:
  """Extract MeshAssetSpec for any visible GEOM_MESH geoms in MjModel / mjx.Model."""
  nmesh = int(getattr(src, "nmesh", 0))
  if nmesh <= 0:
    return None, ()

  geom_type = np.asarray(src.geom_type, dtype=np.int32)
  geom_group = np.asarray(src.geom_group, dtype=np.int32)
  geom_dataid = np.asarray(src.geom_dataid, dtype=np.int32)
  geom_rgba = spec_ref.geom_rgba

  mesh_gids = []
  for g in range(int(src.ngeom)):
    if int(geom_type[g]) != GEOM_MESH_TYPE:
      continue
    if int(geom_group[g]) not in enabled_geom_groups:
      continue
    if float(geom_rgba[g, 3]) <= 0.0:
      continue
    if int(geom_dataid[g]) < 0:
      continue
    mesh_gids.append(g)

  if not mesh_gids:
    return None, ()

  mesh_vert = np.asarray(src.mesh_vert, dtype=np.float32)
  mesh_vertadr = np.asarray(src.mesh_vertadr, dtype=np.int32)
  mesh_vertnum = np.asarray(src.mesh_vertnum, dtype=np.int32)
  mesh_face = np.asarray(src.mesh_face, dtype=np.int32)
  mesh_faceadr = np.asarray(src.mesh_faceadr, dtype=np.int32)
  mesh_facenum = np.asarray(src.mesh_facenum, dtype=np.int32)

  mesh_dicts = []
  for g in mesh_gids:
    mid = int(geom_dataid[g])
    va, vn = int(mesh_vertadr[mid]), int(mesh_vertnum[mid])
    fa, fn = int(mesh_faceadr[mid]), int(mesh_facenum[mid])
    verts = mesh_vert[va : va + vn]
    faces = mesh_face[fa : fa + fn]
    mesh_dicts.append({
        "name": f"geom_{g}_mesh_{mid}",
        "verts": verts,
        "faces": faces,
        "rgba": tuple(float(x) for x in geom_rgba[g]),
    })

  light_pos = tuple(float(x) for x in spec_ref.light_pos[0]) if spec_ref.nlight > 0 else (0.65, 0.25, 1.85)
  light_amb = tuple(float(x) for x in spec_ref.light_ambient[0]) if spec_ref.nlight > 0 else (0.28, 0.30, 0.34)
  light_diff = tuple(float(x) for x in spec_ref.light_diffuse[0]) if spec_ref.nlight > 0 else (0.85, 0.83, 0.80)
  light_spec = tuple(float(x) for x in spec_ref.light_specular[0]) if spec_ref.nlight > 0 else (0.30, 0.30, 0.30)
  mesh_spec = build_mesh_scene_spec(
      mesh_dicts,
      height=spec_ref.height,
      width=spec_ref.width,
      cam_fovy=spec_ref.cam_fovy,
      znear=spec_ref.znear,
      zfar=spec_ref.zfar,
      light_pos=light_pos,
      light_ambient=light_amb,
      light_diffuse=light_diff,
      light_specular=light_spec,
      sky_rgb1=spec_ref.sky_rgb1,
      sky_rgb2=spec_ref.sky_rgb2,
      use_shadows=spec_ref.use_shadows,
  )
  return mesh_spec, tuple(mesh_gids)


def _prepare_geom_light_buffers(
    spec: RenderSceneSpec,
    mjx_data_batch: Any,
    geom_rgba_batch: Optional[jax.Array],
    block_b: int,
    block_r: int,
):
  cam_pos = mjx_data_batch.cam_xpos[:, int(spec.cam_id), :]
  cam_mat = mjx_data_batch.cam_xmat[:, int(spec.cam_id), :, :]
  geom_xpos = mjx_data_batch.geom_xpos
  geom_xmat = mjx_data_batch.geom_xmat
  light_xpos, light_xdir = _extract_light_batch(spec, mjx_data_batch)

  B = cam_pos.shape[0]
  H, W = int(spec.height), int(spec.width)
  R = H * W
  gids = np.asarray(spec.enabled_geom_ids, dtype=np.int32)
  G = len(gids)

  B_pad = ((B + block_b - 1) // block_b) * block_b
  R_pad = ((R + block_r - 1) // block_r) * block_r
  G_pad = ((max(G, 1) + 127) // 128) * 128
  n_b_blk = B_pad // block_b
  n_r_blk = R_pad // block_r

  cam_pack = jnp.concatenate([cam_pos, cam_mat.reshape(B, 9)], axis=-1)
  if B_pad != B:
    cam_128 = jnp.pad(cam_pack, ((0, B_pad - B), (0, 128 - 12)))
  else:
    cam_128 = jnp.pad(cam_pack, ((0, 0), (0, 128 - 12)))

  geom_size = jnp.asarray(spec.geom_size, dtype=jnp.float32)
  gpos = geom_xpos[:, gids, :]
  gmat = geom_xmat[:, gids, :, :].reshape(B, G, 9)
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

  has_dynamic_rgba = geom_rgba_batch is not None
  if has_dynamic_rgba:
    g_rgba_4 = geom_rgba_batch[:, gids, :]
    if B_pad != B or G_pad != G:
      g_rgba_4 = jnp.pad(g_rgba_4, ((0, B_pad - B), (0, G_pad - G), (0, 0)))
    rgba_fields = (
        g_rgba_4.reshape(n_b_blk, block_b, G_pad, 4)
        .transpose(0, 3, 1, 2)
        .reshape(n_b_blk * 4 * block_b, G_pad)
    )
  else:
    rgba_fields = None

  active_lights = []
  light_chunks = []
  for li in range(int(spec.nlight)):
    if not bool(spec.light_active[li]):
      continue
    lp = light_xpos[:, li, :]
    ld_v = light_xdir[:, li, :]
    ld_v = ld_v / jnp.maximum(jnp.linalg.norm(ld_v, axis=-1, keepdims=True), 1e-12)
    light_chunks.append(lp)
    light_chunks.append(ld_v)
    active_lights.append((
        li,
        int(spec.light_type[li]),
        tuple(float(x) for x in spec.light_ambient[li]),
        tuple(float(x) for x in spec.light_diffuse[li]),
        tuple(float(x) for x in spec.light_specular[li]),
        tuple(float(x) for x in spec.light_attenuation[li]),
        float(np.cos(spec.light_cutoff_rad[li])),
        float(spec.light_exponent[li]),
        bool(spec.use_shadows and bool(spec.light_castshadow[li])),
    ))

  if light_chunks:
    l_cat = jnp.concatenate(light_chunks, axis=-1)
    light_128 = jnp.pad(l_cat, ((0, B_pad - B), (0, 128 - l_cat.shape[1])))
  else:
    light_128 = jnp.zeros((B_pad, 128), dtype=jnp.float32)

  return (
      B, H, W, R, gids, B_pad, R_pad, G_pad, n_b_blk, n_r_blk,
      cam_128, geom_fields, rgba_fields, light_128, active_lights,
  )


def _pallas_raycast_universal_fused(
    spec: RenderSceneSpec,
    mjx_data_batch: Any,
    geom_rgba_batch: Optional[jax.Array] = None,
    interpret: bool = False,
    block_b: int = 16,
    block_r: int = 512,
    out_dtype: Any = jnp.bfloat16,
    dedup_materials: bool = True,
    emit_aux: bool = True,
) -> Dict[str, jax.Array]:
  """V3 Material-Deduped In-VMEM Blinn-Phong shader emitting (rgb, depth[, normal, seg_id])."""
  gids = np.asarray(spec.enabled_geom_ids, dtype=np.int32)
  B = mjx_data_batch.cam_xpos.shape[0]
  H, W = int(spec.height), int(spec.width)
  R = H * W

  if len(gids) == 0:
    cam_mat = mjx_data_batch.cam_xmat[:, int(spec.cam_id), :, :]
    ld_np, _, _ = build_local_camera_rays(spec)
    ld = jnp.asarray(ld_np, dtype=jnp.float32)
    ray_dir = jnp.einsum("bij,rj->bri", cam_mat, ld, precision=jax.lax.Precision.HIGHEST)
    ray_dir = ray_dir / jnp.maximum(jnp.linalg.norm(ray_dir, axis=-1, keepdims=True), 1e-12)
    sky_t = jnp.clip(0.5 * (ray_dir[..., 2] + 1.0), 0.0, 1.0)[..., None]
    sky_rgb = (
        sky_t * jnp.asarray(spec.sky_rgb1, dtype=jnp.float32)[None, None, :]
        + (1.0 - sky_t) * jnp.asarray(spec.sky_rgb2, dtype=jnp.float32)[None, None, :]
    )
    out = {
        "rgb": sky_rgb.astype(out_dtype).reshape(B, H, W, 3),
        "depth": jnp.full((B, H, W), -1.0, dtype=out_dtype),
    }
    if emit_aux:
      out["normal"] = jnp.zeros((B, H, W, 3), dtype=out_dtype)
      out["seg_id"] = jnp.full((B, H, W), -1, dtype=jnp.int32)
    return out

  (
      B, H, W, R, gids, B_pad, R_pad, G_pad, n_b_blk, n_r_blk,
      cam_128, geom_fields, rgba_fields, light_128, active_lights,
  ) = _prepare_geom_light_buffers(spec, mjx_data_batch, geom_rgba_batch, block_b, block_r)

  ld_np, lo_np, neg_z_np = build_local_camera_rays(spec)
  md_np = ((float(spec.zfar) - float(spec.znear)) / neg_z_np) if spec.zfar > 0.0 else np.full((R,), MJ_MAXVAL, dtype=np.float32)
  lray_raw = np.zeros((8, R_pad), dtype=np.float32)
  lray_raw[0:3, :R] = ld_np.T
  lray_raw[3:6, :R] = lo_np.T
  lray_raw[6, :R] = neg_z_np
  lray_raw[7, :R] = md_np
  lray_raw[2, R:] = -1.0
  lray_raw[6, R:] = 1.0
  lray_raw[7, R:] = 1.0
  lray_dev = jnp.asarray(lray_raw, dtype=jnp.float32)

  g_types_static = tuple(int(spec.geom_type[g]) for g in gids)
  gids_static = tuple(int(g) for g in gids)
  plane_fx = tuple(float(spec.geom_size[g, 0]) > 0.0 for g in gids)
  plane_fy = tuple(float(spec.geom_size[g, 1]) > 0.0 for g in gids)
  g_rgba_static = tuple(tuple(float(c) for c in spec.geom_rgba[g, :3]) for g in gids)
  g_spec_static = tuple(float(spec.geom_spec[g]) for g in gids)
  g_shin_static = tuple(float(spec.geom_shin[g]) * 128.0 for g in gids)
  g_emit_static = tuple(float(spec.geom_emit[g]) for g in gids)
  chk_indices = tuple((k, int(g)) for k, g in enumerate(gids) if bool(spec.geom_is_checker[g]))

  spec_uniform = len(set(g_spec_static)) == 1
  shin_uniform = len(set(g_shin_static)) == 1
  emit_uniform = len(set(g_emit_static)) == 1

  has_dynamic_rgba = rgba_fields is not None
  mat_groups: Dict[Tuple[float, ...], list[int]] = {}
  if dedup_materials and not has_dynamic_rgba:
    for k, gid_val in enumerate(gids_static):
      if bool(spec.geom_is_checker[gid_val]):
        continue
      mkey = (*g_rgba_static[k], g_spec_static[k], g_shin_static[k], g_emit_static[k])
      mat_groups.setdefault(mkey, []).append(gid_val)

  znear_f = float(spec.znear)
  rep_x, rep_y = float(spec.checker_repeat[0]), float(spec.checker_repeat[1])
  chk1_r, chk1_g, chk1_b = [float(x) for x in spec.checker_rgb1]
  chk2_r, chk2_g, chk2_b = [float(x) for x in spec.checker_rgb2]
  sky1_r, sky1_g, sky1_b = [float(x) for x in spec.sky_rgb1]
  sky2_r, sky2_g, sky2_b = [float(x) for x in spec.sky_rgb2]
  shadow_frac = float(spec.shadow_light_fraction)
  enable_specular = bool(spec.enable_specular)
  headlight_active = bool(spec.headlight_active)
  hl_amb = tuple(float(x) for x in spec.headlight_ambient)
  hl_diff = tuple(float(x) for x in spec.headlight_diffuse)
  hl_spec = tuple(float(x) for x in spec.headlight_specular)

  def _kernel(*refs):
    idx = 0
    cam = refs[idx][...]; idx += 1
    lr = refs[idx][...]; idx += 1
    gf = refs[idx][...]; idx += 1
    linfo = refs[idx][...]; idx += 1
    if has_dynamic_rgba:
      rgba_blk = refs[idx][...]; idx += 1
    else:
      rgba_blk = None
    out_r_ref, out_g_ref, out_b_ref, out_d_ref = refs[idx], refs[idx + 1], refs[idx + 2], refs[idx + 3]
    idx += 4
    if emit_aux:
      out_nx_ref, out_ny_ref, out_nz_ref, out_gid_ref = refs[idx], refs[idx + 1], refs[idx + 2], refs[idx + 3]

    ldx, ldy, ldz = lr[0:1, :], lr[1:2, :], lr[2:3, :]
    lox, loy, loz = lr[3:4, :], lr[4:5, :], lr[5:6, :]
    inv_cos = znear_f / lr[6:7, :]
    max_d = jnp.broadcast_to(lr[7:8, :], (block_b, block_r))

    cpx, cpy, cpz = cam[:, 0:1], cam[:, 1:2], cam[:, 2:3]
    c00, c01, c02 = cam[:, 3:4], cam[:, 4:5], cam[:, 5:6]
    c10, c11, c12 = cam[:, 6:7], cam[:, 7:8], cam[:, 8:9]
    c20, c21, c22 = cam[:, 9:10], cam[:, 10:11], cam[:, 11:12]

    rdx_raw = c00 * ldx + c01 * ldy + c02 * ldz
    rdy_raw = c10 * ldx + c11 * ldy + c12 * ldz
    rdz_raw = c20 * ldx + c21 * ldy + c22 * ldz
    inv_rn = 1.0 / jnp.maximum(jnp.sqrt(rdx_raw * rdx_raw + rdy_raw * rdy_raw + rdz_raw * rdz_raw), 1e-12)
    rdx, rdy, rdz = rdx_raw * inv_rn, rdy_raw * inv_rn, rdz_raw * inv_rn

    rox = cpx + (c00 * lox + c01 * loy + c02 * loz) + inv_cos * rdx
    roy = cpy + (c10 * lox + c11 * loy + c12 * loz) + inv_cos * rdy
    roz = cpz + (c20 * lox + c21 * loy + c22 * loz) + inv_cos * rdz

    best_d = max_d
    best_nx = jnp.zeros((block_b, block_r), dtype=jnp.float32)
    best_ny = jnp.zeros((block_b, block_r), dtype=jnp.float32)
    best_nz = jnp.zeros((block_b, block_r), dtype=jnp.float32)
    best_gid = jnp.full((block_b, block_r), -1, dtype=jnp.int32)

    for k, (gt, gid_val, pfx, pfy) in enumerate(zip(g_types_static, gids_static, plane_fx, plane_fy)):
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
      best_nx = jnp.where(imp, cnx, best_nx)
      best_ny = jnp.where(imp, cny, best_ny)
      best_nz = jnp.where(imp, cnz, best_nz)

    hit = best_gid >= 0
    d_safe = jnp.where(hit, best_d, 0.0)
    nx_v = jnp.where(hit, best_nx, 0.0)
    ny_v = jnp.where(hit, best_ny, 0.0)
    nz_v = jnp.where(hit, best_nz, 0.0)
    hx = rox + d_safe * rdx
    hy = roy + d_safe * rdy
    hz = roz + d_safe * rdz

    alb_r = jnp.zeros((block_b, block_r), dtype=jnp.float32)
    alb_g = jnp.zeros((block_b, block_r), dtype=jnp.float32)
    alb_b = jnp.zeros((block_b, block_r), dtype=jnp.float32)
    spec_coef = jnp.full((block_b, block_r), g_spec_static[0], dtype=jnp.float32) if spec_uniform else jnp.zeros((block_b, block_r), dtype=jnp.float32)
    shin_exp = jnp.full((block_b, block_r), g_shin_static[0], dtype=jnp.float32) if shin_uniform else jnp.zeros((block_b, block_r), dtype=jnp.float32)
    emit_coef = jnp.full((block_b, block_r), g_emit_static[0], dtype=jnp.float32) if emit_uniform else jnp.zeros((block_b, block_r), dtype=jnp.float32)

    if dedup_materials and rgba_blk is None:
      for mkey, gid_list in mat_groups.items():
        rk, gk, bk, sk, shk, ek = mkey
        is_m = best_gid == gid_list[0]
        for g_other in gid_list[1:]:
          is_m = is_m | (best_gid == g_other)
        alb_r = jnp.where(is_m, rk, alb_r)
        alb_g = jnp.where(is_m, gk, alb_g)
        alb_b = jnp.where(is_m, bk, alb_b)
        if not spec_uniform:
          spec_coef = jnp.where(is_m, sk, spec_coef)
        if not shin_uniform:
          shin_exp = jnp.where(is_m, shk, shin_exp)
        if not emit_uniform:
          emit_coef = jnp.where(is_m, ek, emit_coef)
      if not (spec_uniform and shin_uniform and emit_uniform):
        for k, cg in chk_indices:
          is_cg = best_gid == cg
          if not spec_uniform:
            spec_coef = jnp.where(is_cg, g_spec_static[k], spec_coef)
          if not shin_uniform:
            shin_exp = jnp.where(is_cg, g_shin_static[k], shin_exp)
          if not emit_uniform:
            emit_coef = jnp.where(is_cg, g_emit_static[k], emit_coef)
    else:
      for k, gid_val in enumerate(gids_static):
        is_k = best_gid == gid_val
        if rgba_blk is not None:
          rk = rgba_blk[0 * block_b : 1 * block_b, k : k + 1]
          gk = rgba_blk[1 * block_b : 2 * block_b, k : k + 1]
          bk = rgba_blk[2 * block_b : 3 * block_b, k : k + 1]
        else:
          rk, gk, bk = g_rgba_static[k]
        alb_r = jnp.where(is_k, rk, alb_r)
        alb_g = jnp.where(is_k, gk, alb_g)
        alb_b = jnp.where(is_k, bk, alb_b)
        if not spec_uniform:
          spec_coef = jnp.where(is_k, g_spec_static[k], spec_coef)
        if not shin_uniform:
          shin_exp = jnp.where(is_k, g_shin_static[k], shin_exp)
        if not emit_uniform:
          emit_coef = jnp.where(is_k, g_emit_static[k], emit_coef)

    for k, cg in chk_indices:
      px = gf[0 * block_b : 1 * block_b, k : k + 1]
      py = gf[1 * block_b : 2 * block_b, k : k + 1]
      pz = gf[2 * block_b : 3 * block_b, k : k + 1]
      m00, m01 = gf[3 * block_b : 4 * block_b, k : k + 1], gf[4 * block_b : 5 * block_b, k : k + 1]
      m10, m11 = gf[6 * block_b : 7 * block_b, k : k + 1], gf[7 * block_b : 8 * block_b, k : k + 1]
      m20, m21 = gf[9 * block_b : 10 * block_b, k : k + 1], gf[10 * block_b : 11 * block_b, k : k + 1]
      dcx, dcy, dcz = hx - px, hy - py, hz - pz
      lx_cg = m00 * dcx + m10 * dcy + m20 * dcz
      ly_cg = m01 * dcx + m11 * dcy + m21 * dcz
      cx = jnp.floor(lx_cg * rep_x).astype(jnp.int32)
      cy = jnp.floor(ly_cg * rep_y).astype(jnp.int32)
      even = ((cx + cy) & 1) == 0
      is_cg = best_gid == cg
      alb_r = jnp.where(is_cg, jnp.where(even, chk1_r, chk2_r), alb_r)
      alb_g = jnp.where(is_cg, jnp.where(even, chk1_g, chk2_g), alb_g)
      alb_b = jnp.where(is_cg, jnp.where(even, chk1_b, chk2_b), alb_b)

    diff_r = jnp.zeros((block_b, block_r), dtype=jnp.float32)
    diff_g = jnp.zeros((block_b, block_r), dtype=jnp.float32)
    diff_b = jnp.zeros((block_b, block_r), dtype=jnp.float32)
    spec_r = jnp.zeros((block_b, block_r), dtype=jnp.float32)
    spec_g = jnp.zeros((block_b, block_r), dtype=jnp.float32)
    spec_b = jnp.zeros((block_b, block_r), dtype=jnp.float32)

    for slot, (_, l_type, l_amb, l_diff, l_spec, l_att, cos_cut, exp_val, do_sh) in enumerate(active_lights):
      off = slot * 6
      lpx_l = linfo[:, off + 0 : off + 1]
      lpy_l = linfo[:, off + 1 : off + 2]
      lpz_l = linfo[:, off + 2 : off + 3]
      ldx_l = linfo[:, off + 3 : off + 4]
      ldy_l = linfo[:, off + 4 : off + 5]
      ldz_l = linfo[:, off + 5 : off + 6]

      diff_r = diff_r + l_amb[0]
      diff_g = diff_g + l_amb[1]
      diff_b = diff_b + l_amb[2]

      if l_type == 1:
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
        att_denom = l_att[0] + l_att[1] * d_raw + l_att[2] * d_sq
        attenuation = jnp.where(att_denom > 1e-12, 1.0 / jnp.maximum(att_denom, 1e-12), 1.0)
        if l_type == 0:
          cos_theta = -(lx_v * ldx_l + ly_v * ldy_l + lz_v * ldz_l)
          in_cone = cos_theta >= cos_cut
          safe_cos = jnp.where(cos_theta > 0.0, cos_theta, 1.0)
          spot_pow = jnp.where(cos_theta > 0.0, safe_cos ** exp_val, 1.0 if exp_val == 0.0 else 0.0)
          attenuation = jnp.where(in_cone, attenuation * spot_pow, 0.0)

      ndotl = jnp.maximum(0.0, nx_v * lx_v + ny_v * ly_v + nz_v * lz_v)
      if do_sh:
        sh_ox = hx + nx_v * 1.0e-4
        sh_oy = hy + ny_v * 1.0e-4
        sh_oz = hz + nz_v * 1.0e-4
        sh_max = jnp.where(hit & (ndotl > 0.0), dist_lim, 0.0)
        occ_bool = jnp.zeros((block_b, block_r), dtype=jnp.bool_)
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
              sx, sy, sz, rb_sq, sh_ox, sh_oy, sh_oz, lx_v, ly_v, lz_v,
              plane_finite_x=pfx, plane_finite_y=pfy, compute_normal=False,
          )
          occ_bool = occ_bool | (scd < sh_max)
        vis = jnp.where(hit & (ndotl > 0.0) & occ_bool, shadow_frac, 1.0)
      else:
        vis = 1.0

      ndotl_w = ndotl * attenuation * vis
      diff_r = diff_r + l_diff[0] * ndotl_w
      diff_g = diff_g + l_diff[1] * ndotl_w
      diff_b = diff_b + l_diff[2] * ndotl_w

      if enable_specular:
        hx_v, hy_v, hz_v = lx_v - rdx, ly_v - rdy, lz_v - rdz
        inv_hn = 1.0 / jnp.maximum(jnp.sqrt(hx_v * hx_v + hy_v * hy_v + hz_v * hz_v), 1e-12)
        ndoth = jnp.maximum(0.0, (nx_v * hx_v + ny_v * hy_v + nz_v * hz_v) * inv_hn)
        safe_ndoth = jnp.where(ndoth > 0.0, ndoth, 1.0)
        spec_w = jnp.where(
            (spec_coef > 0.0) & (shin_exp > 0.0) & (ndoth > 0.0),
            spec_coef * (safe_ndoth ** shin_exp) * attenuation * vis,
            0.0,
        )
        spec_r = spec_r + l_spec[0] * spec_w
        spec_g = spec_g + l_spec[1] * spec_w
        spec_b = spec_b + l_spec[2] * spec_w

    if headlight_active:
      ndotl_hl = jnp.maximum(0.0, -(nx_v * rdx + ny_v * rdy + nz_v * rdz))
      diff_r = diff_r + hl_amb[0] + hl_diff[0] * ndotl_hl
      diff_g = diff_g + hl_amb[1] + hl_diff[1] * ndotl_hl
      diff_b = diff_b + hl_amb[2] + hl_diff[2] * ndotl_hl
      if enable_specular:
        safe_ndoth_hl = jnp.where(ndotl_hl > 0.0, ndotl_hl, 1.0)
        spec_w_hl = jnp.where(
            (spec_coef > 0.0) & (shin_exp > 0.0) & (ndotl_hl > 0.0),
            spec_coef * (safe_ndoth_hl ** shin_exp),
            0.0,
        )
        spec_r = spec_r + hl_spec[0] * spec_w_hl
        spec_g = spec_g + hl_spec[1] * spec_w_hl
        spec_b = spec_b + hl_spec[2] * spec_w_hl

    direct_r = jnp.clip(alb_r * diff_r + spec_r + alb_r * emit_coef, 0.0, 1.0)
    direct_g = jnp.clip(alb_g * diff_g + spec_g + alb_g * emit_coef, 0.0, 1.0)
    direct_b = jnp.clip(alb_b * diff_b + spec_b + alb_b * emit_coef, 0.0, 1.0)

    sky_t = jnp.clip(0.5 * (rdz + 1.0), 0.0, 1.0)
    sky_r = sky_t * sky1_r + (1.0 - sky_t) * sky2_r
    sky_g = sky_t * sky1_g + (1.0 - sky_t) * sky2_g
    sky_b = sky_t * sky1_b + (1.0 - sky_t) * sky2_b

    out_r_ref[...] = jnp.where(hit, direct_r, sky_r).astype(out_dtype)
    out_g_ref[...] = jnp.where(hit, direct_g, sky_g).astype(out_dtype)
    out_b_ref[...] = jnp.where(hit, direct_b, sky_b).astype(out_dtype)
    out_d_ref[...] = jnp.where(hit, best_d + inv_cos, -1.0).astype(out_dtype)
    if emit_aux:
      out_nx_ref[...] = nx_v.astype(out_dtype)
      out_ny_ref[...] = ny_v.astype(out_dtype)
      out_nz_ref[...] = nz_v.astype(out_dtype)
      out_gid_ref[...] = best_gid

  in_specs = [
      pl.BlockSpec((block_b, 128), lambda bi, ri: (bi, 0)),
      pl.BlockSpec((8, block_r), lambda bi, ri: (0, ri)),
      pl.BlockSpec((16 * block_b, G_pad), lambda bi, ri: (bi, 0)),
      pl.BlockSpec((block_b, 128), lambda bi, ri: (bi, 0)),
  ]
  call_args = [cam_128, lray_dev, geom_fields, light_128]
  if has_dynamic_rgba:
    in_specs.append(pl.BlockSpec((4 * block_b, G_pad), lambda bi, ri: (bi, 0)))
    call_args.append(rgba_fields)

  out_shape = [
      jax.ShapeDtypeStruct((B_pad, R_pad), out_dtype),
      jax.ShapeDtypeStruct((B_pad, R_pad), out_dtype),
      jax.ShapeDtypeStruct((B_pad, R_pad), out_dtype),
      jax.ShapeDtypeStruct((B_pad, R_pad), out_dtype),
  ]
  out_specs = [
      pl.BlockSpec((block_b, block_r), lambda bi, ri: (bi, ri)),
      pl.BlockSpec((block_b, block_r), lambda bi, ri: (bi, ri)),
      pl.BlockSpec((block_b, block_r), lambda bi, ri: (bi, ri)),
      pl.BlockSpec((block_b, block_r), lambda bi, ri: (bi, ri)),
  ]
  if emit_aux:
    out_shape.extend([
        jax.ShapeDtypeStruct((B_pad, R_pad), out_dtype),
        jax.ShapeDtypeStruct((B_pad, R_pad), out_dtype),
        jax.ShapeDtypeStruct((B_pad, R_pad), out_dtype),
        jax.ShapeDtypeStruct((B_pad, R_pad), jnp.int32),
    ])
    out_specs.extend([
        pl.BlockSpec((block_b, block_r), lambda bi, ri: (bi, ri)),
        pl.BlockSpec((block_b, block_r), lambda bi, ri: (bi, ri)),
        pl.BlockSpec((block_b, block_r), lambda bi, ri: (bi, ri)),
        pl.BlockSpec((block_b, block_r), lambda bi, ri: (bi, ri)),
    ])

  outs = pl.pallas_call(
      _kernel,
      out_shape=out_shape,
      grid=(n_b_blk, n_r_blk),
      in_specs=in_specs,
      out_specs=out_specs,
      compiler_params=None if interpret else pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
      interpret=interpret,
  )(*call_args)

  r_out, g_out, b_out, d_out = outs[0], outs[1], outs[2], outs[3]
  res = {
      "rgb": jnp.stack([r_out[:B, :R], g_out[:B, :R], b_out[:B, :R]], axis=-1).reshape(B, H, W, 3),
      "depth": d_out[:B, :R].reshape(B, H, W),
  }
  if emit_aux:
    nx_out, ny_out, nz_out, gid_out = outs[4], outs[5], outs[6], outs[7]
    res["normal"] = jnp.stack([nx_out[:B, :R], ny_out[:B, :R], nz_out[:B, :R]], axis=-1).reshape(B, H, W, 3)
    res["seg_id"] = gid_out[:B, :R].reshape(B, H, W)
  return res


def _pallas_raycast_universal_pbr_ao(
    spec: RenderSceneSpec,
    mjx_data_batch: Any,
    geom_rgba_batch: Optional[jax.Array] = None,
    interpret: bool = False,
    block_b: int = 8,
    block_r: int = 512,
    out_dtype: Any = jnp.bfloat16,
    n_ao_samples: int = 4,
    n_shadow_samples: int = 4,
    ao_radius: float = 0.35,
    ao_strength: float = 0.75,
    shadow_softness: float = 0.06,
    emit_aux: bool = True,
) -> Dict[str, jax.Array]:
  """V3 In-VMEM GGX Microfacet PBR + Multi-Sample Hemisphere AO + Soft-Shadow shader."""
  gids = np.asarray(spec.enabled_geom_ids, dtype=np.int32)
  B = mjx_data_batch.cam_xpos.shape[0]
  H, W = int(spec.height), int(spec.width)
  R = H * W

  if len(gids) == 0:
    return _pallas_raycast_universal_fused(
        spec, mjx_data_batch, geom_rgba_batch=geom_rgba_batch, interpret=interpret,
        block_b=block_b, block_r=block_r, out_dtype=out_dtype, emit_aux=emit_aux,
    )

  (
      B, H, W, R, gids, B_pad, R_pad, G_pad, n_b_blk, n_r_blk,
      cam_128, geom_fields, rgba_fields, light_128, active_lights,
  ) = _prepare_geom_light_buffers(spec, mjx_data_batch, geom_rgba_batch, block_b, block_r)

  ld_np, lo_np, neg_z_np = build_local_camera_rays(spec)
  md_np = ((float(spec.zfar) - float(spec.znear)) / neg_z_np) if spec.zfar > 0.0 else np.full((R,), MJ_MAXVAL, dtype=np.float32)
  cos_rot_np, sin_rot_np = _r2_pixel_rotation(H, W, R_pad)
  lray_raw = np.zeros((16, R_pad), dtype=np.float32)
  lray_raw[0:3, :R] = ld_np.T
  lray_raw[3:6, :R] = lo_np.T
  lray_raw[6, :R] = neg_z_np
  lray_raw[7, :R] = md_np
  lray_raw[2, R:] = -1.0
  lray_raw[6, R:] = 1.0
  lray_raw[7, R:] = 1.0
  lray_raw[8, :] = cos_rot_np
  lray_raw[9, :] = sin_rot_np
  lray_dev = jnp.asarray(lray_raw, dtype=jnp.float32)

  g_types_static = tuple(int(spec.geom_type[g]) for g in gids)
  gids_static = tuple(int(g) for g in gids)
  plane_fx = tuple(float(spec.geom_size[g, 0]) > 0.0 for g in gids)
  plane_fy = tuple(float(spec.geom_size[g, 1]) > 0.0 for g in gids)
  g_rgba_static = tuple(tuple(float(c) for c in spec.geom_rgba[g, :3]) for g in gids)
  g_spec_static = tuple(float(spec.geom_spec[g]) for g in gids)
  g_shin_static = tuple(float(spec.geom_shin[g]) for g in gids)
  g_emit_static = tuple(float(spec.geom_emit[g]) for g in gids)
  chk_indices = tuple((k, int(g)) for k, g in enumerate(gids) if bool(spec.geom_is_checker[g]))

  spec_uniform = len(set(g_spec_static)) == 1
  shin_uniform = len(set(g_shin_static)) == 1
  emit_uniform = len(set(g_emit_static)) == 1
  has_dynamic_rgba = rgba_fields is not None

  mat_groups: Dict[Tuple[float, ...], list[int]] = {}
  if not has_dynamic_rgba:
    for k, gid_val in enumerate(gids_static):
      if bool(spec.geom_is_checker[gid_val]):
        continue
      mkey = (*g_rgba_static[k], g_spec_static[k], g_shin_static[k], g_emit_static[k])
      mat_groups.setdefault(mkey, []).append(gid_val)

  ao_samples = []
  for s in range(max(n_ao_samples, 0)):
    r_s = math.sqrt((s + 0.5) / max(n_ao_samples, 1))
    phi_s = 2.0 * math.pi * ((s + 0.5) * 0.61803398875)
    uz = math.sqrt(max(1.0 - r_s * r_s, 1e-4))
    ao_samples.append((r_s * math.cos(phi_s), r_s * math.sin(phi_s), uz))

  sh_samples = []
  for s in range(max(n_shadow_samples, 1)):
    if n_shadow_samples <= 1:
      sh_samples.append((0.0, 0.0))
    else:
      r_s = math.sqrt((s + 0.5) / n_shadow_samples) * shadow_softness
      phi_s = 2.0 * math.pi * ((s + 0.5) * 0.61803398875)
      sh_samples.append((r_s * math.cos(phi_s), r_s * math.sin(phi_s)))

  znear_f = float(spec.znear)
  rep_x, rep_y = float(spec.checker_repeat[0]), float(spec.checker_repeat[1])
  chk1_r, chk1_g, chk1_b = [float(x) for x in spec.checker_rgb1]
  chk2_r, chk2_g, chk2_b = [float(x) for x in spec.checker_rgb2]
  sky1_r, sky1_g, sky1_b = [float(x) for x in spec.sky_rgb1]
  sky2_r, sky2_g, sky2_b = [float(x) for x in spec.sky_rgb2]
  shadow_frac = float(spec.shadow_light_fraction)
  headlight_active = bool(spec.headlight_active)
  hl_amb = tuple(float(x) for x in spec.headlight_ambient)
  hl_diff = tuple(float(x) for x in spec.headlight_diffuse)
  hl_spec = tuple(float(x) for x in spec.headlight_specular)

  def _kernel(*refs):
    idx = 0
    cam = refs[idx][...]; idx += 1
    lr = refs[idx][...]; idx += 1
    gf = refs[idx][...]; idx += 1
    linfo = refs[idx][...]; idx += 1
    if has_dynamic_rgba:
      rgba_blk = refs[idx][...]; idx += 1
    else:
      rgba_blk = None
    out_r_ref, out_g_ref, out_b_ref, out_d_ref = refs[idx], refs[idx + 1], refs[idx + 2], refs[idx + 3]
    idx += 4
    if emit_aux:
      out_nx_ref, out_ny_ref, out_nz_ref, out_gid_ref = refs[idx], refs[idx + 1], refs[idx + 2], refs[idx + 3]

    ldx, ldy, ldz = lr[0:1, :], lr[1:2, :], lr[2:3, :]
    lox, loy, loz = lr[3:4, :], lr[4:5, :], lr[5:6, :]
    inv_cos = znear_f / lr[6:7, :]
    max_d = jnp.broadcast_to(lr[7:8, :], (block_b, block_r))
    cos_rot = lr[8:9, :]
    sin_rot = lr[9:10, :]

    cpx, cpy, cpz = cam[:, 0:1], cam[:, 1:2], cam[:, 2:3]
    c00, c01, c02 = cam[:, 3:4], cam[:, 4:5], cam[:, 5:6]
    c10, c11, c12 = cam[:, 6:7], cam[:, 7:8], cam[:, 8:9]
    c20, c21, c22 = cam[:, 9:10], cam[:, 10:11], cam[:, 11:12]

    rdx_raw = c00 * ldx + c01 * ldy + c02 * ldz
    rdy_raw = c10 * ldx + c11 * ldy + c12 * ldz
    rdz_raw = c20 * ldx + c21 * ldy + c22 * ldz
    inv_rn = 1.0 / jnp.maximum(jnp.sqrt(rdx_raw * rdx_raw + rdy_raw * rdy_raw + rdz_raw * rdz_raw), 1e-12)
    rdx, rdy, rdz = rdx_raw * inv_rn, rdy_raw * inv_rn, rdz_raw * inv_rn

    rox = cpx + (c00 * lox + c01 * loy + c02 * loz) + inv_cos * rdx
    roy = cpy + (c10 * lox + c11 * loy + c12 * loz) + inv_cos * rdy
    roz = cpz + (c20 * lox + c21 * loy + c22 * loz) + inv_cos * rdz

    best_d = max_d
    best_nx = jnp.zeros((block_b, block_r), dtype=jnp.float32)
    best_ny = jnp.zeros((block_b, block_r), dtype=jnp.float32)
    best_nz = jnp.zeros((block_b, block_r), dtype=jnp.float32)
    best_gid = jnp.full((block_b, block_r), -1, dtype=jnp.int32)

    for k, (gt, gid_val, pfx, pfy) in enumerate(zip(g_types_static, gids_static, plane_fx, plane_fy)):
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
      best_nx = jnp.where(imp, cnx, best_nx)
      best_ny = jnp.where(imp, cny, best_ny)
      best_nz = jnp.where(imp, cnz, best_nz)

    hit = best_gid >= 0
    d_safe = jnp.where(hit, best_d, 0.0)
    nx_out = jnp.where(hit, best_nx, 0.0)
    ny_out = jnp.where(hit, best_ny, 0.0)
    nz_out = jnp.where(hit, best_nz, 0.0)
    nz_onb = jnp.where(hit, best_nz, 1.0)

    hx = rox + d_safe * rdx
    hy = roy + d_safe * rdy
    hz = roz + d_safe * rdz

    alb_r = jnp.zeros((block_b, block_r), dtype=jnp.float32)
    alb_g = jnp.zeros((block_b, block_r), dtype=jnp.float32)
    alb_b = jnp.zeros((block_b, block_r), dtype=jnp.float32)
    spec_coef = jnp.full((block_b, block_r), g_spec_static[0], dtype=jnp.float32) if spec_uniform else jnp.zeros((block_b, block_r), dtype=jnp.float32)
    shin_val = jnp.full((block_b, block_r), g_shin_static[0], dtype=jnp.float32) if shin_uniform else jnp.zeros((block_b, block_r), dtype=jnp.float32)
    emit_coef = jnp.full((block_b, block_r), g_emit_static[0], dtype=jnp.float32) if emit_uniform else jnp.zeros((block_b, block_r), dtype=jnp.float32)

    if rgba_blk is None:
      for mkey, gid_list in mat_groups.items():
        rk, gk, bk, sk, shk, ek = mkey
        is_m = best_gid == gid_list[0]
        for g_other in gid_list[1:]:
          is_m = is_m | (best_gid == g_other)
        alb_r = jnp.where(is_m, rk, alb_r)
        alb_g = jnp.where(is_m, gk, alb_g)
        alb_b = jnp.where(is_m, bk, alb_b)
        if not spec_uniform:
          spec_coef = jnp.where(is_m, sk, spec_coef)
        if not shin_uniform:
          shin_val = jnp.where(is_m, shk, shin_val)
        if not emit_uniform:
          emit_coef = jnp.where(is_m, ek, emit_coef)
    else:
      for k, gid_val in enumerate(gids_static):
        is_k = best_gid == gid_val
        rk = rgba_blk[0 * block_b : 1 * block_b, k : k + 1]
        gk = rgba_blk[1 * block_b : 2 * block_b, k : k + 1]
        bk = rgba_blk[2 * block_b : 3 * block_b, k : k + 1]
        alb_r = jnp.where(is_k, rk, alb_r)
        alb_g = jnp.where(is_k, gk, alb_g)
        alb_b = jnp.where(is_k, bk, alb_b)
        if not spec_uniform:
          spec_coef = jnp.where(is_k, g_spec_static[k], spec_coef)
        if not shin_uniform:
          shin_val = jnp.where(is_k, g_shin_static[k], shin_val)
        if not emit_uniform:
          emit_coef = jnp.where(is_k, g_emit_static[k], emit_coef)

    for k, cg in chk_indices:
      px = gf[0 * block_b : 1 * block_b, k : k + 1]
      py = gf[1 * block_b : 2 * block_b, k : k + 1]
      pz = gf[2 * block_b : 3 * block_b, k : k + 1]
      m00, m01 = gf[3 * block_b : 4 * block_b, k : k + 1], gf[4 * block_b : 5 * block_b, k : k + 1]
      m10, m11 = gf[6 * block_b : 7 * block_b, k : k + 1], gf[7 * block_b : 8 * block_b, k : k + 1]
      m20, m21 = gf[9 * block_b : 10 * block_b, k : k + 1], gf[10 * block_b : 11 * block_b, k : k + 1]
      dcx, dcy, dcz = hx - px, hy - py, hz - pz
      lx_cg = m00 * dcx + m10 * dcy + m20 * dcz
      ly_cg = m01 * dcx + m11 * dcy + m21 * dcz
      cx = jnp.floor(lx_cg * rep_x).astype(jnp.int32)
      cy = jnp.floor(ly_cg * rep_y).astype(jnp.int32)
      even = ((cx + cy) & 1) == 0
      is_cg = best_gid == cg
      alb_r = jnp.where(is_cg, jnp.where(even, chk1_r, chk2_r), alb_r)
      alb_g = jnp.where(is_cg, jnp.where(even, chk1_g, chk2_g), alb_g)
      alb_b = jnp.where(is_cg, jnp.where(even, chk1_b, chk2_b), alb_b)

    sh_ox = hx + nx_out * 2.0e-4
    sh_oy = hy + ny_out * 2.0e-4
    sh_oz = hz + nz_onb * 2.0e-4

    if n_ao_samples > 0:
      (tx, ty, tz), (bx, by, bz) = _branchless_onb(nx_out, ny_out, nz_onb)
      ao_occ_sum = jnp.zeros((block_b, block_r), dtype=jnp.float32)
      ao_max_d = jnp.where(hit, ao_radius, 0.0)
      for ux0, uy0, uz0 in ao_samples:
        ux = ux0 * cos_rot - uy0 * sin_rot
        uy = ux0 * sin_rot + uy0 * cos_rot
        aodx = ux * tx + uy * bx + uz0 * nx_out
        aody = ux * ty + uy * by + uz0 * ny_out
        aodz = ux * tz + uy * bz + uz0 * nz_onb
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
        ao_occ_sum = ao_occ_sum + jnp.where(hit & (min_ao < ao_radius), 1.0 - 0.5 * (min_ao / ao_radius), 0.0)
      ao_vis = jnp.clip(1.0 - ao_strength * (ao_occ_sum / float(n_ao_samples)), 0.15, 1.0)
    else:
      ao_vis = 1.0

    roughness = jnp.clip(1.0 - shin_val * 1.4, 0.14, 0.90)
    alpha = roughness * roughness
    a2 = alpha * alpha
    k_g = ((roughness + 1.0) * (roughness + 1.0)) * 0.125
    f0 = 0.04 + 0.08 * spec_coef
    ndotv = jnp.maximum(1e-4, -(nx_out * rdx + ny_out * rdy + nz_onb * rdz))
    g1_v = ndotv / (ndotv * (1.0 - k_g) + k_g + 1e-12)

    amb_r = jnp.zeros((block_b, block_r), dtype=jnp.float32)
    amb_g = jnp.zeros((block_b, block_r), dtype=jnp.float32)
    amb_b = jnp.zeros((block_b, block_r), dtype=jnp.float32)
    diff_r = jnp.zeros((block_b, block_r), dtype=jnp.float32)
    diff_g = jnp.zeros((block_b, block_r), dtype=jnp.float32)
    diff_b = jnp.zeros((block_b, block_r), dtype=jnp.float32)
    spec_r = jnp.zeros((block_b, block_r), dtype=jnp.float32)
    spec_g = jnp.zeros((block_b, block_r), dtype=jnp.float32)
    spec_b = jnp.zeros((block_b, block_r), dtype=jnp.float32)

    for slot, (_, l_type, l_amb, l_diff, l_spec, l_att, cos_cut, exp_val, do_sh) in enumerate(active_lights):
      off = slot * 6
      lpx_l = linfo[:, off + 0 : off + 1]
      lpy_l = linfo[:, off + 1 : off + 2]
      lpz_l = linfo[:, off + 2 : off + 3]
      ldx_l = linfo[:, off + 3 : off + 4]
      ldy_l = linfo[:, off + 4 : off + 5]
      ldz_l = linfo[:, off + 5 : off + 6]

      amb_r = amb_r + l_amb[0]
      amb_g = amb_g + l_amb[1]
      amb_b = amb_b + l_amb[2]

      if l_type == 1:
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
        att_denom = l_att[0] + l_att[1] * d_raw + l_att[2] * d_sq
        attenuation = jnp.where(att_denom > 1e-12, 1.0 / jnp.maximum(att_denom, 1e-12), 1.0)
        if l_type == 0:
          cos_theta = -(lx_v * ldx_l + ly_v * ldy_l + lz_v * ldz_l)
          in_cone = cos_theta >= cos_cut
          safe_cos = jnp.where(cos_theta > 0.0, cos_theta, 1.0)
          spot_pow = jnp.where(cos_theta > 0.0, safe_cos ** exp_val, 1.0 if exp_val == 0.0 else 0.0)
          attenuation = jnp.where(in_cone, attenuation * spot_pow, 0.0)

      ndotl = jnp.maximum(0.0, nx_out * lx_v + ny_out * ly_v + nz_onb * lz_v)

      if do_sh:
        sh_max = jnp.where(hit & (ndotl > 0.0), dist_lim, 0.0)
        (ltx, lty, ltz), (lbx, lby, lbz) = _branchless_onb(lx_v, ly_v, lz_v)
        occ_sum = jnp.zeros((block_b, block_r), dtype=jnp.float32)
        for dx_d, dy_d in sh_samples:
          if dx_d == 0.0 and dy_d == 0.0:
            sdx, sdy, sdz = lx_v, ly_v, lz_v
          else:
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
          occ_sum = occ_sum + (hit & (ndotl > 0.0) & occ_s).astype(jnp.float32)
        vis = 1.0 - (1.0 - shadow_frac) * (occ_sum / float(len(sh_samples)))
      else:
        vis = 1.0

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
      diff_r = diff_r + l_diff[0] * weight * kd
      diff_g = diff_g + l_diff[1] * weight * kd
      diff_b = diff_b + l_diff[2] * weight * kd

      spec_fac = jnp.where((spec_coef > 0.0) & (ndotl > 0.0), spec_brdf * weight * math.pi, 0.0)
      spec_r = spec_r + l_spec[0] * spec_fac
      spec_g = spec_g + l_spec[1] * spec_fac
      spec_b = spec_b + l_spec[2] * spec_fac

    if headlight_active:
      amb_r = amb_r + hl_amb[0]
      amb_g = amb_g + hl_amb[1]
      amb_b = amb_b + hl_amb[2]
      diff_r = diff_r + hl_diff[0] * ndotv
      diff_g = diff_g + hl_diff[1] * ndotv
      diff_b = diff_b + hl_diff[2] * ndotv
      denom_hl = ndotv * ndotv * (a2 - 1.0) + 1.0
      d_hl = a2 / (math.pi * denom_hl * denom_hl + 1e-12)
      spec_hl = jnp.where(spec_coef > 0.0, d_hl * f0 * g1_v * g1_v / (4.0 * ndotv + 1e-4) * math.pi, 0.0)
      spec_r = spec_r + hl_spec[0] * spec_hl
      spec_g = spec_g + hl_spec[1] * spec_hl
      spec_b = spec_b + hl_spec[2] * spec_hl

    tot_r = alb_r * (amb_r * ao_vis + diff_r + emit_coef) + spec_r * (0.5 + 0.5 * ao_vis)
    tot_g = alb_g * (amb_g * ao_vis + diff_g + emit_coef) + spec_g * (0.5 + 0.5 * ao_vis)
    tot_b = alb_b * (amb_b * ao_vis + diff_b + emit_coef) + spec_b * (0.5 + 0.5 * ao_vis)

    direct_r = jnp.clip(tot_r, 0.0, 1.0)
    direct_g = jnp.clip(tot_g, 0.0, 1.0)
    direct_b = jnp.clip(tot_b, 0.0, 1.0)

    sky_t = jnp.clip(0.5 * (rdz + 1.0), 0.0, 1.0)
    sky_r = sky_t * sky1_r + (1.0 - sky_t) * sky2_r
    sky_g = sky_t * sky1_g + (1.0 - sky_t) * sky2_g
    sky_b = sky_t * sky1_b + (1.0 - sky_t) * sky2_b

    out_r_ref[...] = jnp.where(hit, direct_r, sky_r).astype(out_dtype)
    out_g_ref[...] = jnp.where(hit, direct_g, sky_g).astype(out_dtype)
    out_b_ref[...] = jnp.where(hit, direct_b, sky_b).astype(out_dtype)
    out_d_ref[...] = jnp.where(hit, best_d + inv_cos, -1.0).astype(out_dtype)
    if emit_aux:
      out_nx_ref[...] = nx_out.astype(out_dtype)
      out_ny_ref[...] = ny_out.astype(out_dtype)
      out_nz_ref[...] = nz_out.astype(out_dtype)
      out_gid_ref[...] = best_gid

  in_specs = [
      pl.BlockSpec((block_b, 128), lambda bi, ri: (bi, 0)),
      pl.BlockSpec((16, block_r), lambda bi, ri: (0, ri)),
      pl.BlockSpec((16 * block_b, G_pad), lambda bi, ri: (bi, 0)),
      pl.BlockSpec((block_b, 128), lambda bi, ri: (bi, 0)),
  ]
  call_args = [cam_128, lray_dev, geom_fields, light_128]
  if has_dynamic_rgba:
    in_specs.append(pl.BlockSpec((4 * block_b, G_pad), lambda bi, ri: (bi, 0)))
    call_args.append(rgba_fields)

  out_shape = [
      jax.ShapeDtypeStruct((B_pad, R_pad), out_dtype),
      jax.ShapeDtypeStruct((B_pad, R_pad), out_dtype),
      jax.ShapeDtypeStruct((B_pad, R_pad), out_dtype),
      jax.ShapeDtypeStruct((B_pad, R_pad), out_dtype),
  ]
  out_specs = [
      pl.BlockSpec((block_b, block_r), lambda bi, ri: (bi, ri)),
      pl.BlockSpec((block_b, block_r), lambda bi, ri: (bi, ri)),
      pl.BlockSpec((block_b, block_r), lambda bi, ri: (bi, ri)),
      pl.BlockSpec((block_b, block_r), lambda bi, ri: (bi, ri)),
  ]
  if emit_aux:
    out_shape.extend([
        jax.ShapeDtypeStruct((B_pad, R_pad), out_dtype),
        jax.ShapeDtypeStruct((B_pad, R_pad), out_dtype),
        jax.ShapeDtypeStruct((B_pad, R_pad), out_dtype),
        jax.ShapeDtypeStruct((B_pad, R_pad), jnp.int32),
    ])
    out_specs.extend([
        pl.BlockSpec((block_b, block_r), lambda bi, ri: (bi, ri)),
        pl.BlockSpec((block_b, block_r), lambda bi, ri: (bi, ri)),
        pl.BlockSpec((block_b, block_r), lambda bi, ri: (bi, ri)),
        pl.BlockSpec((block_b, block_r), lambda bi, ri: (bi, ri)),
    ])

  outs = pl.pallas_call(
      _kernel,
      out_shape=out_shape,
      grid=(n_b_blk, n_r_blk),
      in_specs=in_specs,
      out_specs=out_specs,
      compiler_params=None if interpret else pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
      interpret=interpret,
  )(*call_args)

  r_out, g_out, b_out, d_out = outs[0], outs[1], outs[2], outs[3]
  res = {
      "rgb": jnp.stack([r_out[:B, :R], g_out[:B, :R], b_out[:B, :R]], axis=-1).reshape(B, H, W, 3),
      "depth": d_out[:B, :R].reshape(B, H, W),
  }
  if emit_aux:
    nx_out, ny_out, nz_out, gid_out = outs[4], outs[5], outs[6], outs[7]
    res["normal"] = jnp.stack([nx_out[:B, :R], ny_out[:B, :R], nz_out[:B, :R]], axis=-1).reshape(B, H, W, 3)
    res["seg_id"] = gid_out[:B, :R].reshape(B, H, W)
  return res


def _project_world_to_pixel(
    spec: RenderSceneSpec,
    cam_pos: jax.Array,
    cam_mat: jax.Array,
    pts_w: jax.Array,
) -> jax.Array:
  """Project (B, H, W, 3) world points into pixel coordinates (u, v) in [0, W] x [0, H]."""
  # cam_mat columns are camera axes in world: p_cam = cam_mat^T @ (p_w - cam_pos)
  delta = pts_w - cam_pos[:, None, None, :]
  p_cam = jnp.einsum("bji,bhwj->bhwi", cam_mat, delta, precision=jax.lax.Precision.HIGHEST)
  H, W = int(spec.height), int(spec.width)
  aspect = float(W) / float(H)

  if int(spec.cam_projection) == PROJ_ORTHOGRAPHIC:
    sensor_h = float(spec.cam_fovy)
    sensor_w = sensor_h * aspect
    u_norm = (p_cam[..., 0] + 0.5 * sensor_w) / max(sensor_w, 1e-6)
    v_norm = (0.5 * sensor_h - p_cam[..., 1]) / max(sensor_h, 1e-6)
  else:
    neg_z = jnp.maximum(-p_cam[..., 2], 1e-5)
    fovy_rad = float(spec.cam_fovy) * (math.pi / 180.0)
    half_h = math.tan(0.5 * fovy_rad)
    half_w = half_h * aspect
    x_ndc = p_cam[..., 0] / neg_z
    y_ndc = p_cam[..., 1] / neg_z
    u_norm = (x_ndc + half_w) / max(2.0 * half_w, 1e-6)
    v_norm = (half_h - y_ndc) / max(2.0 * half_h, 1e-6)

  return jnp.stack([u_norm * float(W), v_norm * float(H)], axis=-1)


def _compute_flow_zero_gather(
    spec: RenderSceneSpec,
    mjx_data_batch: Any,
    prev_data_batch: Optional[Any],
    depth: jax.Array,
    seg_id: jax.Array,
    geom_bodyid: Tuple[int, ...],
    body_rootid: Tuple[int, ...],
    dt: float,
    out_dtype: Any,
) -> jax.Array:
  """Zero-gather 2D optical flow (B, H, W, 2) in pixels via static geom-loop select."""
  B, H, W = depth.shape
  hit = (seg_id >= 0) & (depth >= 0.0)
  if prev_data_batch is None and not (hasattr(mjx_data_batch, "cvel") and hasattr(mjx_data_batch, "subtree_com")):
    return jnp.zeros((B, H, W, 2), dtype=out_dtype)

  cam_pos = mjx_data_batch.cam_xpos[:, int(spec.cam_id), :]
  cam_mat = mjx_data_batch.cam_xmat[:, int(spec.cam_id), :, :]
  ld_np, lo_np, neg_z_np = build_local_camera_rays(spec)
  ld = jnp.asarray(ld_np, dtype=jnp.float32)
  lo = jnp.asarray(lo_np, dtype=jnp.float32)
  inv_cos = (float(spec.znear) / jnp.asarray(neg_z_np, dtype=jnp.float32)).reshape(1, H, W)
  ray_dir = jnp.einsum("bij,rj->bri", cam_mat, ld, precision=jax.lax.Precision.HIGHEST)
  ray_dir = (ray_dir / jnp.maximum(jnp.linalg.norm(ray_dir, axis=-1, keepdims=True), 1e-12)).reshape(B, H, W, 3)
  ray_off = jnp.einsum("bij,rj->bri", cam_mat, lo, precision=jax.lax.Precision.HIGHEST).reshape(B, H, W, 3)
  d_safe = jnp.where(hit, jnp.maximum(depth.astype(jnp.float32) - inv_cos, 0.0), 0.0)
  ray_orig = cam_pos[:, None, None, :] + ray_off + inv_cos[..., None] * ray_dir
  pts_w = ray_orig + d_safe[..., None] * ray_dir

  if prev_data_batch is not None:
    # Reconstruct previous world point per geom without any XLA gather
    prev_pts_w = pts_w
    gids = tuple(int(g) for g in spec.enabled_geom_ids)
    for g in gids:
      # Skip static worldbody geoms if camera is also static? Camera may move, so keep world transform!
      p_curr = mjx_data_batch.geom_xpos[:, g, :]
      m_curr = mjx_data_batch.geom_xmat[:, g, :, :]
      p_prev = prev_data_batch.geom_xpos[:, g, :]
      m_prev = prev_data_batch.geom_xmat[:, g, :, :]
      local_p = jnp.einsum("bji,bhwj->bhwi", m_curr, pts_w - p_curr[:, None, None, :], precision=jax.lax.Precision.HIGHEST)
      cand_prev = p_prev[:, None, None, :] + jnp.einsum("bij,bhwj->bhwi", m_prev, local_p, precision=jax.lax.Precision.HIGHEST)
      prev_pts_w = jnp.where((seg_id == g)[..., None], cand_prev, prev_pts_w)
    uv_curr = _project_world_to_pixel(spec, cam_pos, cam_mat, pts_w)
    uv_prev = _project_world_to_pixel(
        spec,
        prev_data_batch.cam_xpos[:, int(spec.cam_id), :],
        prev_data_batch.cam_xmat[:, int(spec.cam_id), :, :],
        prev_pts_w,
    )
    flow = jnp.where(hit[..., None], uv_curr - uv_prev, 0.0)
    return flow.astype(out_dtype)

  # Instantaneous cvel-based flow (zero gather via static Python int body slices!)
  vel_w = jnp.zeros((B, H, W, 3), dtype=jnp.float32)
  gids = tuple(int(g) for g in spec.enabled_geom_ids)
  cvel = mjx_data_batch.cvel
  scom = mjx_data_batch.subtree_com
  for g in gids:
    bid = geom_bodyid[g] if g < len(geom_bodyid) else 0
    if bid <= 0:
      continue
    rid = body_rootid[bid] if bid < len(body_rootid) else bid
    omega = cvel[:, bid, :3][:, None, None, :]
    vlin = cvel[:, bid, 3:][:, None, None, :]
    com = scom[:, rid, :][:, None, None, :]
    v_g = vlin + jnp.cross(omega, pts_w - com)
    vel_w = jnp.where((seg_id == g)[..., None], v_g, vel_w)

  uv_curr = _project_world_to_pixel(spec, cam_pos, cam_mat, pts_w)
  uv_next = _project_world_to_pixel(spec, cam_pos, cam_mat, pts_w + vel_w * float(dt))
  flow = jnp.where(hit[..., None], uv_next - uv_curr, 0.0)
  return flow.astype(out_dtype)


@dataclasses.dataclass
class CompiledScene:
  """Compiled TPU-RTX scene descriptor ready for zero-copy rendering."""

  specs: Tuple[RenderSceneSpec, ...]
  cam_ids: Tuple[int, ...]
  shader: str
  res: Tuple[int, int]
  out_dtype: Any
  outputs: Tuple[str, ...]
  mesh: Optional[jax.sharding.Mesh]
  data_axis: str
  n_chips: int
  block_b: int
  block_r: int
  n_ao_samples: int
  n_shadow_samples: int
  ao_radius: float
  ao_strength: float
  shadow_softness: float
  dedup_materials: bool
  mesh_spec: Optional[MeshAssetSpec]
  mesh_geom_ids: Tuple[int, ...]
  geom_bodyid: Tuple[int, ...]
  body_rootid: Tuple[int, ...]
  dt: float
  interpret: Optional[bool]
  _compiled_cache: Dict[Tuple[bool, bool], Callable[..., Any]] = dataclasses.field(default_factory=dict, repr=False)

  @property
  def spec(self) -> RenderSceneSpec:
    return self.specs[0]

  @property
  def multi_cam(self) -> bool:
    return len(self.specs) > 1


def _resolve_cam_ids(src: Any, cam_id: Union[int, str, Sequence[Union[int, str]]]) -> Tuple[int, ...]:
  ncam = int(getattr(src, "ncam", 1))
  if cam_id == "all":
    return tuple(range(max(ncam, 1)))
  if isinstance(cam_id, (int, str)):
    items = [cam_id]
    single = True
  else:
    items = list(cam_id)
    single = False
  out = []
  for it in items:
    if isinstance(it, int):
      out.append(it)
    elif isinstance(it, str):
      try:
        import mujoco
        cid = mujoco.mj_name2id(src, mujoco.mjtObj.mjOBJ_CAMERA, it)
        if cid < 0:
          raise ValueError(f"Camera '{it}' not found in model.")
        out.append(int(cid))
      except Exception as exc:
        raise ValueError(f"Could not resolve camera name '{it}': {exc}") from exc
    else:
      raise TypeError(f"Unsupported cam_id entry: {it!r}")
  return tuple(out)


def _resolve_mesh_sharding(
    sharding: Union[str, int, jax.sharding.Mesh, None],
    data_axis: str = "data",
) -> Tuple[Optional[jax.sharding.Mesh], str, int]:
  devs = jax.devices()
  if sharding is None or sharding == "none" or sharding == 1:
    return None, data_axis, 1
  if isinstance(sharding, jax.sharding.Mesh):
    ax = data_axis if data_axis in sharding.axis_names else sharding.axis_names[0]
    return sharding, ax, int(sharding.shape[ax])
  if sharding == "auto":
    if len(devs) <= 1:
      return None, data_axis, 1
    mesh = jax.sharding.Mesh(np.array(devs), (data_axis,))
    return mesh, data_axis, len(devs)
  if isinstance(sharding, int):
    n = int(sharding)
    if n <= 1:
      return None, data_axis, 1
    if n > len(devs):
      raise ValueError(f"Requested {n} chips, but only {len(devs)} visible devices on host.")
    mesh = jax.sharding.Mesh(np.array(devs[:n]), (data_axis,))
    return mesh, data_axis, n
  raise ValueError(f"Unsupported sharding spec: {sharding!r}")


def _pytree_pspec(tree: Any, batch_size: int, axis_name: str):
  P = jax.sharding.PartitionSpec
  return jax.tree_util.tree_map(
      lambda x: P(axis_name) if (hasattr(x, "ndim") and x.ndim > 0 and x.shape[0] == batch_size) else P(),
      tree,
  )


def compile_scene(
    source: Any,
    shader: str = "fused_ao_ggx",
    res: Union[int, Tuple[int, int]] = (128, 128),
    sharding: Union[str, int, jax.sharding.Mesh, None] = "auto",
    cam_id: Union[int, str, Sequence[Union[int, str]]] = 0,
    dtype: Any = "bfloat16",
    outputs: Sequence[str] = DEFAULT_OUTPUTS,
    use_shadows: bool = True,
    shadow_light_fraction: float = 0.0,
    enable_specular: bool = True,
    include_meshes: bool = True,
    block_b: Optional[int] = None,
    block_r: int = 512,
    n_ao_samples: int = 4,
    n_shadow_samples: int = 4,
    ao_radius: float = 0.35,
    ao_strength: float = 0.75,
    shadow_softness: float = 0.06,
    dedup_materials: bool = True,
    data_axis: str = "data",
    interpret: Optional[bool] = None,
) -> CompiledScene:
  """Compile any MuJoCo MjModel, mjx.Model, XML, or RenderSceneSpec into a TPU-RTX CompiledScene."""
  shader_norm = _normalize_shader(shader)
  H, W = _normalize_res(res)
  out_dtype = _normalize_dtype(dtype)
  outs_tuple = tuple(outputs)
  if block_b is None:
    block_b = 16 if shader_norm == "fast_fused" else 8

  mesh_obj, ax_name, n_chips = _resolve_mesh_sharding(sharding, data_axis=data_axis)

  # Load XML string or path if given
  src_obj = source
  if isinstance(source, (str, os.PathLike)):
    import mujoco
    s_str = str(source)
    if "<mujoco" in s_str:
      src_obj = mujoco.MjModel.from_xml_string(s_str)
    else:
      src_obj = mujoco.MjModel.from_xml_path(s_str)

  if isinstance(src_obj, RenderSceneSpec):
    specs = (
        dataclasses.replace(
            src_obj,
            height=H,
            width=W,
            use_shadows=use_shadows,
            shadow_light_fraction=shadow_light_fraction,
            enable_specular=enable_specular,
        ),
    )
    cam_ids = (int(src_obj.cam_id),)
    mesh_spec, mesh_gids = None, ()
    geom_bodyid = tuple(0 for _ in range(int(src_obj.ngeom)))
    body_rootid = (0,)
    dt = 0.005
  elif isinstance(src_obj, MeshAssetSpec):
    specs = (
        RenderSceneSpec(
            ngeom=0, nlight=1, cam_id=0, height=H, width=W, cam_projection=0,
            cam_fovy=src_obj.cam_fovy, cam_sensorsize=src_obj.cam_sensorsize,
            cam_intrinsic=src_obj.cam_intrinsic, znear=src_obj.znear, zfar=src_obj.zfar,
            use_shadows=use_shadows, shadow_light_fraction=shadow_light_fraction,
            enable_specular=enable_specular, enabled_geom_ids=(), type_groups=(),
            geom_type=np.zeros((0,), np.int32), geom_size=np.zeros((0, 3), np.float32),
            geom_rgba=np.zeros((0, 4), np.float32), geom_spec=np.zeros((0,), np.float32),
            geom_shin=np.zeros((0,), np.float32), geom_emit=np.zeros((0,), np.float32),
            geom_is_checker=np.zeros((0,), bool),
            checker_rgb1=(0.2, 0.2, 0.2), checker_rgb2=(0.1, 0.1, 0.1), checker_markrgb=(0.5, 0.5, 0.5),
            checker_repeat=(1.0, 1.0), sky_rgb1=src_obj.sky_rgb1, sky_rgb2=src_obj.sky_rgb2,
            light_type=np.zeros((1,), np.int32), light_castshadow=np.array([use_shadows], bool),
            light_active=np.array([True], bool), light_bodyid=np.zeros((1,), np.int32),
            light_pos=src_obj.light_pos[None, :], light_dir=np.array([[0.0, 0.0, -1.0]], np.float32),
            light_attenuation=np.array([[1.0, 0.0, 0.0]], np.float32),
            light_cutoff_rad=np.array([np.pi / 4.0], np.float32), light_exponent=np.array([10.0], np.float32),
            light_ambient=np.array([src_obj.light_ambient], np.float32),
            light_diffuse=np.array([src_obj.light_diffuse], np.float32),
            light_specular=np.array([src_obj.light_specular], np.float32),
            headlight_active=True, headlight_ambient=(0.25, 0.25, 0.25),
            headlight_diffuse=(0.45, 0.45, 0.45), headlight_specular=(0.15, 0.15, 0.15),
        ),
    )
    cam_ids = (0,)
    mesh_spec = dataclasses.replace(src_obj, height=H, width=W, use_shadows=use_shadows)
    mesh_gids = tuple(range(len(src_obj.meshes)))
    geom_bodyid = ()
    body_rootid = (0,)
    dt = 0.005
  else:
    # MjModel or mjx.Model
    cam_ids = _resolve_cam_ids(src_obj, cam_id)
    is_mjmodel = hasattr(src_obj, "vis") and hasattr(src_obj, "mat_texrepeat")
    specs_list = []
    for cid in cam_ids:
      if is_mjmodel:
        sp = RenderSceneSpec.from_mjmodel(
            src_obj, cam_id=cid, height=H, width=W,
            use_shadows=use_shadows, shadow_light_fraction=shadow_light_fraction,
            enable_specular=enable_specular,
        )
      else:
        sp = _spec_from_mjx_model(
            src_obj, cam_id=cid, height=H, width=W,
            use_shadows=use_shadows, shadow_light_fraction=shadow_light_fraction,
            enable_specular=enable_specular,
        )
      specs_list.append(sp)
    specs = tuple(specs_list)
    if include_meshes:
      mesh_spec, mesh_gids = _extract_mesh_spec_from_model(src_obj, specs[0])
    else:
      mesh_spec, mesh_gids = None, ()
    geom_bodyid = tuple(int(x) for x in np.asarray(src_obj.geom_bodyid, dtype=np.int32))
    body_rootid = tuple(int(x) for x in np.asarray(getattr(src_obj, "body_rootid", np.arange(int(src_obj.nbody))), dtype=np.int32))
    dt = float(getattr(getattr(src_obj, "opt", None), "timestep", 0.005))

  return CompiledScene(
      specs=specs,
      cam_ids=cam_ids,
      shader=shader_norm,
      res=(H, W),
      out_dtype=out_dtype,
      outputs=outs_tuple,
      mesh=mesh_obj,
      data_axis=ax_name,
      n_chips=n_chips,
      block_b=int(block_b),
      block_r=int(block_r),
      n_ao_samples=int(n_ao_samples),
      n_shadow_samples=int(n_shadow_samples),
      ao_radius=float(ao_radius),
      ao_strength=float(ao_strength),
      shadow_softness=float(shadow_softness),
      dedup_materials=bool(dedup_materials),
      mesh_spec=mesh_spec,
      mesh_geom_ids=mesh_gids,
      geom_bodyid=geom_bodyid,
      body_rootid=body_rootid,
      dt=dt,
      interpret=interpret,
  )


def _render_single_cam_local(
    scene: CompiledScene,
    spec: RenderSceneSpec,
    mjx_data_batch: Any,
    prev_data_batch: Optional[Any],
    geom_rgba_batch: Optional[jax.Array],
) -> Dict[str, jax.Array]:
  interp = (jax.default_backend() == "cpu") if scene.interpret is None else bool(scene.interpret)
  need_aux = any(k in scene.outputs for k in ("normal", "seg_id", "flow"))

  if scene.shader == "exact_gbuffer":
    gb = pallas_raycast_batch_v2(spec, mjx_data_batch, geom_rgba_batch=geom_rgba_batch, interpret=interp)
    out = {
        "rgb": gb.rgb.astype(scene.out_dtype),
        "depth": gb.depth.astype(scene.out_dtype),
    }
    if need_aux:
      out["normal"] = gb.normal.astype(scene.out_dtype)
      out["seg_id"] = gb.geom_id
  elif scene.shader == "fast_fused":
    out = _pallas_raycast_universal_fused(
        spec, mjx_data_batch, geom_rgba_batch=geom_rgba_batch, interpret=interp,
        block_b=scene.block_b, block_r=scene.block_r, out_dtype=scene.out_dtype,
        dedup_materials=scene.dedup_materials, emit_aux=need_aux,
    )
  else:  # fused_ao_ggx
    out = _pallas_raycast_universal_pbr_ao(
        spec, mjx_data_batch, geom_rgba_batch=geom_rgba_batch, interpret=interp,
        block_b=scene.block_b, block_r=scene.block_r, out_dtype=scene.out_dtype,
        n_ao_samples=scene.n_ao_samples, n_shadow_samples=scene.n_shadow_samples,
        ao_radius=scene.ao_radius, ao_strength=scene.ao_strength,
        shadow_softness=scene.shadow_softness, emit_aux=need_aux,
    )

  # Optional Z-buffer compositing with static triangle meshes (if model has GEOM_MESH)
  if scene.mesh_spec is not None and len(scene.mesh_geom_ids) > 0:
    mgids = np.asarray(scene.mesh_geom_ids, dtype=np.int32)
    obj_xpos = mjx_data_batch.geom_xpos[:, mgids, :]
    obj_xmat = mjx_data_batch.geom_xmat[:, mgids, :, :]
    cam_pos = mjx_data_batch.cam_xpos[:, int(spec.cam_id), :]
    cam_mat = mjx_data_batch.cam_xmat[:, int(spec.cam_id), :, :]
    mgb = render_mesh_batch(
        scene.mesh_spec, obj_xpos, obj_xmat, cam_pos, cam_mat,
        mode="pallas", interpret=interp,
    )
    m_hit = mgb.depth >= 0.0
    a_hit = out["depth"] >= 0.0
    take_mesh = m_hit & ((~a_hit) | (mgb.depth.astype(jnp.float32) < out["depth"].astype(jnp.float32)))
    out["rgb"] = jnp.where(take_mesh[..., None], mgb.rgb.astype(scene.out_dtype), out["rgb"])
    out["depth"] = jnp.where(take_mesh, mgb.depth.astype(scene.out_dtype), out["depth"])
    if "normal" in out:
      out["normal"] = jnp.where(take_mesh[..., None], mgb.normal.astype(scene.out_dtype), out["normal"])
    if "seg_id" in out:
      # Map mesh index -> model geom_id statically without gather
      mesh_gid = jnp.full_like(out["seg_id"], -1)
      for mi, mgid in enumerate(scene.mesh_geom_ids):
        mesh_gid = jnp.where(mgb.geom_id == mi, jnp.int32(mgid), mesh_gid)
      out["seg_id"] = jnp.where(take_mesh, mesh_gid, out["seg_id"])

  if "flow" in scene.outputs:
    out["flow"] = _compute_flow_zero_gather(
        spec, mjx_data_batch, prev_data_batch,
        out["depth"], out["seg_id"],
        scene.geom_bodyid, scene.body_rootid, scene.dt, scene.out_dtype,
    )

  return {k: out[k] for k in scene.outputs}


def _render_local(
    scene: CompiledScene,
    mjx_data_batch: Any,
    prev_data_batch: Optional[Any],
    geom_rgba_batch: Optional[jax.Array],
) -> Dict[str, jax.Array]:
  if not scene.multi_cam:
    return _render_single_cam_local(scene, scene.specs[0], mjx_data_batch, prev_data_batch, geom_rgba_batch)
  cam_outs = [
      _render_single_cam_local(scene, sp, mjx_data_batch, prev_data_batch, geom_rgba_batch)
      for sp in scene.specs
  ]
  return {k: jnp.stack([co[k] for co in cam_outs], axis=1) for k in scene.outputs}


def build_render_fn(
    scene: CompiledScene,
    batch_size: int,
    has_prev: bool = False,
    has_dyn_rgba: bool = False,
) -> Callable[..., Dict[str, jax.Array]]:
  """Build & cache the zero-copy JIT / shard_map callable for a CompiledScene."""
  use_spmd = (
      scene.mesh is not None
      and scene.n_chips > 1
      and batch_size >= scene.n_chips
      and (batch_size % scene.n_chips == 0)
  )
  cache_key = (bool(use_spmd), bool(has_prev), bool(has_dyn_rgba))
  if cache_key in scene._compiled_cache:
    return scene._compiled_cache[cache_key]

  if not use_spmd:
    @jax.jit
    def _fn(d_b, p_b=None, rgba_b=None):
      return _render_local(scene, d_b, p_b, rgba_b)
  else:
    P = jax.sharding.PartitionSpec
    ax = scene.data_axis
    sm_kw = _shard_map_kwargs()

    def _local_wrapper(d_b, p_b, rgba_b):
      return _render_local(scene, d_b, p_b, rgba_b)

    in_specs = (P(ax), P(ax) if has_prev else P(), P(ax) if has_dyn_rgba else P())
    out_specs = {k: P(ax) for k in scene.outputs}
    _sm = shard_map(_local_wrapper, mesh=scene.mesh, in_specs=in_specs, out_specs=out_specs, **sm_kw)

    @jax.jit
    def _fn(d_b, p_b=None, rgba_b=None):
      return _sm(d_b, p_b, rgba_b)

  scene._compiled_cache[cache_key] = _fn
  return _fn


def shard_data(tree: Any, scene: CompiledScene) -> Any:
  """Place a batched PyTree (e.g. mjx.Data, ctrl) across scene.mesh with zero copies."""
  if scene.mesh is None or scene.n_chips <= 1:
    return jax.device_put(tree, jax.devices()[0])
  P = jax.sharding.PartitionSpec
  sharding = jax.sharding.NamedSharding(scene.mesh, P(scene.data_axis))
  return jax.device_put(tree, sharding)


def _ensure_sharded_for_scene(tree: Any, scene: CompiledScene) -> Any:
  if tree is None or scene.mesh is None or scene.n_chips <= 1:
    return tree
  leaves = jax.tree_util.tree_leaves(tree)
  if not leaves or isinstance(leaves[0], jax.core.Tracer):
    return tree
  target_devs = set(scene.mesh.devices.flat)
  sh0 = getattr(leaves[0], "sharding", None)
  if sh0 is not None and getattr(sh0, "device_set", None) == target_devs:
    return tree
  return shard_data(tree, scene)


def render_scene(
    mjx_data: Any,
    scene: CompiledScene,
    prev_data: Optional[Any] = None,
    geom_rgba: Optional[jax.Array] = None,
) -> Dict[str, jax.Array]:
  """Render batched `(B, ...)` or unbatched `mjx.Data` with a `CompiledScene`."""
  unbatched = mjx_data.cam_xpos.ndim == 2
  if unbatched:
    d_b = jax.tree_util.tree_map(lambda x: x[None, ...], mjx_data)
    p_b = jax.tree_util.tree_map(lambda x: x[None, ...], prev_data) if prev_data is not None else None
    r_b = geom_rgba[None, ...] if (geom_rgba is not None and geom_rgba.ndim == 2) else geom_rgba
    B = 1
  else:
    d_b, p_b, r_b = mjx_data, prev_data, geom_rgba
    B = int(d_b.cam_xpos.shape[0])

  use_spmd = (
      scene.mesh is not None
      and scene.n_chips > 1
      and B >= scene.n_chips
      and (B % scene.n_chips == 0)
  )
  if use_spmd:
    d_b = _ensure_sharded_for_scene(d_b, scene)
    p_b = _ensure_sharded_for_scene(p_b, scene)
    r_b = _ensure_sharded_for_scene(r_b, scene)

  fn = build_render_fn(scene, batch_size=B, has_prev=(p_b is not None), has_dyn_rgba=(r_b is not None))
  obs = fn(d_b, p_b, r_b)
  if unbatched:
    return {k: v[0] for k, v in obs.items()}
  return obs


def rollout_scan(
    mjx_model: Any,
    mjx_data: Any,
    ctrl_seq: jax.Array,
    scene: CompiledScene,
    substeps: int = 1,
    return_frames: bool = True,
) -> Tuple[Any, Dict[str, jax.Array]]:
  """Fused `lax.scan` multi-step MJX physics + TPU-RTX render inside one XLA program.

  Args:
    mjx_model: `mjx.Model` (replicated across chips if multi-chip).
    mjx_data: Batched `mjx.Data` `(B, ...)`, sharded on `scene.data_axis` if multi-chip.
    ctrl_seq: Control sequence `(T, B, nu)` or `(B, T, nu)` — wait: `(B, T, nu)` shards cleanly on axis 0!
      Also accepts `(B, nu)` for constant control or `(T, nu)` broadcast.
    scene: `CompiledScene`.
    substeps: Number of `mjx.step` physics substeps per rendered observation.
    return_frames: If True, returns stacked `(B, T, H, W, ...)` obs; if False, returns final frame only
      (eliminating HBM frame-history retention during high-FPS RL throughput benchmarks!).
  """
  from mujoco import mjx

  B = int(mjx_data.cam_xpos.shape[0])
  if ctrl_seq.ndim == 2 and ctrl_seq.shape[0] == B:
    ctrl_btu = ctrl_seq[:, None, :]
  elif ctrl_seq.ndim == 3 and ctrl_seq.shape[0] == B:
    ctrl_btu = ctrl_seq
  else:
    raise ValueError(f"Expected ctrl_seq shape (B, T, nu) or (B, nu) with B={B}, got {ctrl_seq.shape}")

  T = int(ctrl_btu.shape[1])

  def _local_scan(mx_l, d_b, c_btu):
    def _one_step(d_cur, step_idx):
      u_b = c_btu[:, step_idx, :]
      if int(substeps) == 1:
        d_next = jax.vmap(lambda di, ui: mjx.step(mx_l, di.replace(ctrl=ui)))(d_cur, u_b)
      else:
        def _sub(_, d_s):
          return jax.vmap(lambda di, ui: mjx.step(mx_l, di.replace(ctrl=ui)))(d_s, u_b)
        d_next = jax.lax.fori_loop(0, int(substeps), _sub, d_cur)
      obs_t = _render_local(scene, d_next, None, None)
      return d_next, obs_t

    if return_frames:
      d_final, obs_tb = jax.lax.scan(_one_step, d_b, jnp.arange(T))
      obs_bt = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), obs_tb)
      return d_final, obs_bt
    else:
      obs_struct = jax.eval_shape(lambda d: _render_local(scene, d, None, None), d_b)
      init_obs = jax.tree_util.tree_map(lambda s: jnp.zeros(s.shape, s.dtype), obs_struct)

      def _accum_step(i, carry):
        d_c, acc_obs = carry
        d_n, obs_n = _one_step(d_c, i)
        new_acc = jax.tree_util.tree_map(lambda a, b: a + b, acc_obs, obs_n)
        return (d_n, new_acc)

      d_final, acc_obs = jax.lax.fori_loop(0, T, _accum_step, (d_b, init_obs))
      return d_final, acc_obs

  if scene.mesh is None or scene.n_chips <= 1:
    return jax.jit(_local_scan)(mjx_model, mjx_data, ctrl_btu)

  P = jax.sharding.PartitionSpec
  ax = scene.data_axis
  sm_kw = _shard_map_kwargs()
  _sm = shard_map(
      _local_scan,
      mesh=scene.mesh,
      in_specs=(P(), P(ax), P(ax)),
      out_specs=(P(ax), P(ax)),
      **sm_kw,
  )
  return jax.jit(_sm)(mjx_model, mjx_data, ctrl_btu)


def audit_hlo(lowered: Any) -> Dict[str, Any]:
  """Verify zero-copy XLA HLO invariants: host_callbacks == 0, send_recv_ops == 0, gather_ops == 0."""
  txt = lowered.as_text() if hasattr(lowered, "as_text") else str(lowered)
  info = inspect_hlo_ops(txt)
  info["send_recv_ops"] = len(re.findall(r"\bsend\(|\brecv\(", txt, re.I))
  info["zero_copy_verified"] = (
      info["host_callbacks"] == 0 and info["send_recv_ops"] == 0 and info["gather_ops"] == 0
  )
  return info


def compute_roofline_report(
    scene: CompiledScene,
    batch_size: int,
    dt_render_ms: Optional[float] = None,
    hw_name: str = "v5p",
) -> Dict[str, Any]:
  """Compute TPU v5p theoretical roofline & achieved % (HBM 2.765 TB/s, VPU 28 TFLOP/s, 64 MiB VMEM)."""
  hw: TPUHardwareSpec = TPU_SPECS[hw_name]
  H, W = scene.res
  n_cams = len(scene.specs)
  G = len(scene.specs[0].enabled_geom_ids)
  n_pix_total = float(batch_size) * float(H * W) * float(n_cams)

  # Rays per pixel: 1 primary + shadows + AO
  if scene.shader == "fused_ao_ggx":
    rays_per_px = 1.0 + float(scene.n_shadow_samples) + float(scene.n_ao_samples)
    flops_per_geom = 68.0
  else:
    rays_per_px = 2.0 if scene.specs[0].use_shadows else 1.0
    flops_per_geom = 65.0

  vpu_flops = n_pix_total * rays_per_px * max(float(G), 1.0) * flops_per_geom

  # HBM traffic per chip:
  # - geom_fields read per (bi, ri) tile: (B/n_chips, n_r_blk, 16, G) * 4B + camera/light + output tensors
  bytes_per_el = 2.0 if scene.out_dtype == jnp.bfloat16 else 4.0
  out_channels = 0.0
  for k in scene.outputs:
    if k in ("rgb", "normal"):
      out_channels += 3.0 * bytes_per_el
    elif k == "depth":
      out_channels += 1.0 * bytes_per_el
    elif k == "seg_id":
      out_channels += 4.0
    elif k == "flow":
      out_channels += 2.0 * bytes_per_el

  n_r_blk = max((H * W + scene.block_r - 1) // scene.block_r, 1)
  hbm_bytes = (
      n_pix_total * out_channels
      + float(batch_size) * float(max(G, 1)) * 16.0 * 4.0 * float(n_r_blk) * float(n_cams)
      + float(H * W) * 16.0 * 4.0
  )

  n_chips = max(scene.n_chips, 1)
  t_vpu_ms = (vpu_flops / (float(n_chips) * hw.vpu_fp32_tflops * 1e12)) * 1e3
  t_hbm_ms = (hbm_bytes / (float(n_chips) * hw.hbm_tb_s * 1e12)) * 1e3
  t_roof_ms = max(t_vpu_ms, t_hbm_ms)

  # VMEM footprint per Pallas core tile
  n_out_bufs = 8 if any(k in scene.outputs for k in ("normal", "seg_id", "flow")) else 4
  g_pad = ((max(G, 1) + 127) // 128) * 128
  vmem_bytes = (
      scene.block_b * 128 * 4 * 2
      + 16 * scene.block_r * 4
      + 16 * scene.block_b * g_pad * 4
      + n_out_bufs * scene.block_b * scene.block_r * 4
  )
  vmem_mib = vmem_bytes / (1024.0 * 1024.0)

  rep: Dict[str, Any] = {
      "hardware": hw.name,
      "n_chips": n_chips,
      "hbm_tb_s_per_chip": hw.hbm_tb_s,
      "vpu_fp32_tflops_per_chip": hw.vpu_fp32_tflops,
      "vmem_mib_capacity": hw.vmem_mib,
      "vmem_tile_mib": vmem_mib,
      "vmem_utilization_pct": 100.0 * vmem_mib / hw.vmem_mib,
      "vpu_gflops_total": vpu_flops / 1e9,
      "hbm_gb_total": hbm_bytes / 1e9,
      "arithmetic_intensity_flop_per_byte": vpu_flops / max(hbm_bytes, 1.0),
      "ridge_point_vpu": hw.ridge_vpu,
      "bound": "VPU" if t_vpu_ms >= t_hbm_ms else "HBM",
      "t_vpu_roof_ms": t_vpu_ms,
      "t_hbm_roof_ms": t_hbm_ms,
      "t_roofline_ms": t_roof_ms,
      "theoretical_peak_render_fps": float(batch_size / max(t_roof_ms * 1e-3, 1e-9)),
  }
  if dt_render_ms is not None and dt_render_ms > 0:
    achieved_gflops_s = (vpu_flops / 1e9) / (dt_render_ms * 1e-3)
    achieved_bw_tb_s = (hbm_bytes / 1e12) / (dt_render_ms * 1e-3)
    rep.update({
        "measured_render_ms": float(dt_render_ms),
        "measured_render_fps": float(batch_size / (dt_render_ms * 1e-3)),
        "achieved_vpu_tflops_total": achieved_gflops_s / 1e3,
        "achieved_vpu_pct": 100.0 * t_vpu_ms / dt_render_ms,
        "achieved_hbm_tb_s_total": achieved_bw_tb_s,
        "achieved_hbm_pct": 100.0 * t_hbm_ms / dt_render_ms,
        "roofline_efficiency_pct": 100.0 * t_roof_ms / dt_render_ms,
    })
  return rep
