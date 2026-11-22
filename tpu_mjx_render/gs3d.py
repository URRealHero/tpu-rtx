"""V2 Rigid-Body-Bound Tile-Based 3D Gaussian Splatting (TPU Pallas + JAX).

Designed strictly for TPU MXU/VPU constraints (zero dynamic atomic per-tile lists):
- `mesh_scene_to_rigid_3dgs` samples anisotropic surface-aligned 3D Gaussians
  anchored cleanly to each rigid body's local frame (or loads pre-baked splats).
- `render_gs3d_batch` applies dynamic rigid poses (B, M, 3/3x3), computes EWA 2D
  covariances, statically bins top-K front-most overlapping Gaussians per 8x8 tile
  via `jax.lax.top_k`, and executes front-to-back alpha compositing either via pure
  JAX or a VMEM-resident Pallas kernel (`pallas_gs3d_tile_rasterize`).
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
import numpy as np

from tpu_mjx_render.mesh_ray import _compute_smooth_vertex_normals


@dataclasses.dataclass(frozen=True)
class Rigid3DGSSpec:
  n_bodies: int
  n_gaussians: int
  height: int
  width: int
  cam_fovy: float
  znear: float
  mu_local: np.ndarray        # (N, 3) float32
  cov_local: np.ndarray       # (N, 3, 3) float32 symmetric covariance R_loc S^2 R_loc^T
  nrm_local: np.ndarray       # (N, 3) float32 surface normal
  rgb_base: np.ndarray        # (N, 3) float32
  opacity: np.ndarray         # (N,) float32 in (0, 1]
  body_id: np.ndarray         # (N,) int32
  light_pos: np.ndarray       # (3,) float32
  light_ambient: Tuple[float, float, float]
  light_diffuse: Tuple[float, float, float]
  sky_rgb1: Tuple[float, float, float]
  sky_rgb2: Tuple[float, float, float]


def mesh_scene_to_rigid_3dgs(
    mesh_dicts: Sequence[Dict[str, Any]],
    total_gaussians: int = 1024,
    height: int = 64,
    width: int = 64,
    cam_fovy: float = 42.0,
    znear: float = 0.05,
    light_pos: Tuple[float, float, float] = (0.65, 0.25, 1.85),
    light_ambient: Tuple[float, float, float] = (0.32, 0.34, 0.38),
    light_diffuse: Tuple[float, float, float] = (0.85, 0.83, 0.80),
    sky_rgb1: Tuple[float, float, float] = (0.28, 0.36, 0.48),
    sky_rgb2: Tuple[float, float, float] = (0.10, 0.14, 0.20),
    seed: int = 42,
) -> Rigid3DGSSpec:
  """Convert M arbitrary rigid triangle meshes into rigid-body-anchored 3D Gaussians."""
  rng = np.random.default_rng(seed)
  n_bodies = len(mesh_dicts)

  # Allocate budget proportionally to log/sqrt or linear surface area (keep min 64 per mesh)
  areas_list = []
  normals_list = []
  total_area = 0.0
  for md in mesh_dicts:
    verts = np.asarray(md["verts"], dtype=np.float32)
    faces = np.asarray(md["faces"], dtype=np.int32)
    vn = (
        np.asarray(md["vert_normals"], dtype=np.float32)
        if ("vert_normals" in md and md["vert_normals"] is not None)
        else _compute_smooth_vertex_normals(verts, faces)
    )
    normals_list.append(vn)
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    a = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=-1)
    areas_list.append(a)
    total_area += float(np.sum(a))

  raw_counts = []
  for a in areas_list:
    frac = float(np.sum(a)) / max(total_area, 1e-8)
    raw_counts.append(max(48, int(round(frac * total_gaussians))))
  # Normalize count exactly to total_gaussians
  counts = np.asarray(raw_counts, dtype=np.int32)
  diff = total_gaussians - int(np.sum(counts))
  counts[int(np.argmax(counts))] += diff

  mu_all, cov_all, nrm_all, rgb_all, op_all, bid_all = [], [], [], [], [], []
  for mid, (md, a, vn, cnt) in enumerate(zip(mesh_dicts, areas_list, normals_list, counts)):
    verts = np.asarray(md["verts"], dtype=np.float32)
    faces = np.asarray(md["faces"], dtype=np.int32)
    rgba = np.asarray(md.get("rgba", [0.7, 0.7, 0.7, 1.0]), dtype=np.float32)[:3]

    probs = a / np.maximum(np.sum(a), 1e-12)
    tri_idx = rng.choice(len(faces), size=int(cnt), replace=True, p=probs)

    # Low-discrepancy jittered barycentric sampling
    r1 = np.sqrt(rng.uniform(0.0, 1.0, size=(cnt, 1))).astype(np.float32)
    r2 = rng.uniform(0.0, 1.0, size=(cnt, 1)).astype(np.float32)
    w0 = 1.0 - r1
    w1 = r1 * (1.0 - r2)
    w2 = r1 * r2

    f_sel = faces[tri_idx]
    v0 = verts[f_sel[:, 0]]
    v1 = verts[f_sel[:, 1]]
    v2 = verts[f_sel[:, 2]]
    mu = w0 * v0 + w1 * v1 + w2 * v2

    nrm = w0 * vn[f_sel[:, 0]] + w1 * vn[f_sel[:, 1]] + w2 * vn[f_sel[:, 2]]
    nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-12)

    # Tangent basis (t1, t2, nrm)
    ref = np.where(np.abs(nrm[:, 2:3]) < 0.9, np.array([[0.0, 0.0, 1.0]], np.float32), np.array([[1.0, 0.0, 0.0]], np.float32))
    t1 = np.cross(ref, nrm)
    t1 = t1 / np.maximum(np.linalg.norm(t1, axis=-1, keepdims=True), 1e-12)
    t2 = np.cross(nrm, t1)

    # Surface disk radius matching mesh area / cnt so splats overlap cleanly without holes
    mesh_area = max(float(np.sum(a)), 1e-6)
    area_per_g = mesh_area / float(cnt)
    rad = float(np.sqrt(area_per_g / math.pi) * 1.85)
    thick = max(rad * 0.18, 1e-3)

    rot = np.stack([t1, t2, nrm], axis=-1)  # (cnt, 3, 3)
    s2 = np.array([rad * rad, rad * rad, thick * thick], dtype=np.float32)
    # cov = rot @ diag(s2) @ rot^T
    cov = np.einsum("nij,j,nkj->nik", rot, s2, rot).astype(np.float32)

    mu_all.append(mu.astype(np.float32))
    cov_all.append(cov)
    nrm_all.append(nrm.astype(np.float32))
    rgb_all.append(np.broadcast_to(rgba[None, :], (cnt, 3)).copy())
    op_all.append(np.full((cnt,), 0.92, dtype=np.float32))
    bid_all.append(np.full((cnt,), mid, dtype=np.int32))

  return Rigid3DGSSpec(
      n_bodies=n_bodies,
      n_gaussians=int(total_gaussians),
      height=int(height),
      width=int(width),
      cam_fovy=float(cam_fovy),
      znear=float(znear),
      mu_local=np.concatenate(mu_all, axis=0),
      cov_local=np.concatenate(cov_all, axis=0),
      nrm_local=np.concatenate(nrm_all, axis=0),
      rgb_base=np.concatenate(rgb_all, axis=0),
      opacity=np.concatenate(op_all, axis=0),
      body_id=np.concatenate(bid_all, axis=0),
      light_pos=np.asarray(light_pos, dtype=np.float32),
      light_ambient=light_ambient,
      light_diffuse=light_diffuse,
      sky_rgb1=sky_rgb1,
      sky_rgb2=sky_rgb2,
  )


def _pallas_gs3d_rasterize(
    mx: jax.Array,       # (B, T, K)
    my: jax.Array,       # (B, T, K)
    icxx: jax.Array,     # (B, T, K)
    icxy: jax.Array,     # (B, T, K)
    icyy: jax.Array,     # (B, T, K)
    sr: jax.Array,       # (B, T, K)
    sg: jax.Array,       # (B, T, K)
    sb: jax.Array,       # (B, T, K)
    sop: jax.Array,      # (B, T, K)
    sz: jax.Array,       # (B, T, K)
    snx: jax.Array,      # (B, T, K)
    sny: jax.Array,      # (B, T, K)
    snz: jax.Array,      # (B, T, K)
    tile_ox: jax.Array,  # (B, T)
    tile_oy: jax.Array,  # (B, T)
    tile_size: int,
    interpret: bool = True,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """VMEM MXU-accelerated Pallas 3DGS tile rasterizer (zero dynamic_slice / zero loops).

  Processes all (64 tile pixels x 128 splats) simultaneously in VMEM registers
  using an upper-triangular MXU matmul for exact exclusive transmittance prefix-sums
  followed by a second MXU matmul to accumulate (RGB, depth, normal, alpha).
  """
  B, T, K = mx.shape
  BT = B * T
  P_t = tile_size * tile_size  # 64 pixels for 8x8 tile
  assert P_t == 64, "Mosaic MXU 3DGS kernel expects 8x8 (64-pixel) tiles"
  K_pad = ((K + 127) // 128) * 128
  assert K_pad == 128, "Mosaic MXU 3DGS kernel expects K <= 128"

  def _pbt(arr):
    return jnp.pad(arr.reshape(BT, K), ((0, 0), (0, K_pad - K)))

  # Pack 16 attributes per tile into (BT * 16, 128) -> each grid step loads (16, 128)
  z_k = jnp.zeros((BT, K_pad), jnp.float32)
  splat_rows = jnp.stack(
      [
          _pbt(mx), _pbt(my), _pbt(icxx), _pbt(icxy), _pbt(icyy),
          _pbt(sr), _pbt(sg), _pbt(sb), _pbt(sop), _pbt(sz),
          _pbt(snx), _pbt(sny), _pbt(snz),
          jnp.broadcast_to(tile_ox.reshape(BT, 1), (BT, K_pad)),
          jnp.broadcast_to(tile_oy.reshape(BT, 1), (BT, K_pad)),
          z_k,
      ],
      axis=1,  # (BT, 16, 128)
  ).reshape(BT * 16, K_pad)

  # Precompute local pixel grid (64, 128) + strict upper-triangular (128, 128) prefix matrix
  lx = np.tile(np.arange(tile_size, dtype=np.float32), tile_size) + 0.5
  ly = np.repeat(np.arange(tile_size, dtype=np.float32), tile_size) + 0.5
  loc_x_dev = jnp.asarray(np.broadcast_to(lx[:, None], (P_t, K_pad)).copy(), jnp.float32)
  loc_y_dev = jnp.asarray(np.broadcast_to(ly[:, None], (P_t, K_pad)).copy(), jnp.float32)
  # U[j, k] = 1.0 iff j < k -> (log_1ma @ U)[:, k] = sum_{j < k} log(1 - alpha[:, j])
  u_128_dev = jnp.asarray(np.triu(np.ones((K_pad, K_pad), dtype=np.float32), k=1), jnp.float32)

  def _gs_kernel(spl_ref, lx_ref, ly_ref, u_ref, out_ref):
    sf = spl_ref[...]   # (16, 128)
    lx_m = lx_ref[...]  # (64, 128)
    ly_m = ly_ref[...]  # (64, 128)
    u_m = u_ref[...]    # (128, 128)

    # Pixel coordinates broadcast across (64, 128)
    px = jnp.broadcast_to(sf[13:14, 0:1], (P_t, K_pad)) + lx_m
    py = jnp.broadcast_to(sf[14:15, 0:1], (P_t, K_pad)) + ly_m

    dx = px - jnp.broadcast_to(sf[0:1, :], (P_t, K_pad))
    dy = py - jnp.broadcast_to(sf[1:2, :], (P_t, K_pad))
    cxx = jnp.broadcast_to(sf[2:3, :], (P_t, K_pad))
    cxy = jnp.broadcast_to(sf[3:4, :], (P_t, K_pad))
    cyy = jnp.broadcast_to(sf[4:5, :], (P_t, K_pad))
    gop = jnp.broadcast_to(sf[8:9, :], (P_t, K_pad))

    power = -0.5 * (cxx * dx * dx + 2.0 * cxy * dx * dy + cyy * dy * dy)
    valid_p = (power <= 0.0) & (power > -16.0)
    alpha = jnp.where(valid_p, jnp.minimum(0.99, gop * jnp.exp(jnp.maximum(power, -16.0))), 0.0)
    alpha = jnp.where(alpha >= (1.0 / 255.0), alpha, 0.0)

    # Exact transmittance via MXU prefix-sum in log space (zero loops!)
    log_1ma = jnp.log(jnp.maximum(1.0 - alpha, 1.0e-7))
    log_tr = jnp.matmul(log_1ma, u_m, precision=jax.lax.Precision.HIGHEST)
    trans = jnp.exp(jnp.maximum(log_tr, -30.0))
    w = alpha * trans  # (64, 128)

    # Pack attributes of 128 splats into (128, 8) padded to (128, 128) -> single MXU matmul
    ones_k = jnp.ones((1, K_pad), jnp.float32)
    attr_top = jnp.concatenate(
        [sf[5:8, :], sf[9:13, :], ones_k], axis=0
    )  # (8, 128) -> transpose to (128, 8) & pad to (128, 128)
    attr_128 = jnp.pad(jnp.transpose(attr_top, (1, 0)), ((0, 0), (0, K_pad - 8)))
    out_ref[...] = jnp.matmul(w, attr_128, precision=jax.lax.Precision.HIGHEST)  # (64, 128)

  out_raw = pl.pallas_call(
      _gs_kernel,
      out_shape=jax.ShapeDtypeStruct((BT * P_t, K_pad), jnp.float32),
      grid=(BT,),
      in_specs=[
          pl.BlockSpec((16, K_pad), lambda i: (i, 0)),
          pl.BlockSpec((P_t, K_pad), lambda i: (0, 0)),
          pl.BlockSpec((P_t, K_pad), lambda i: (0, 0)),
          pl.BlockSpec((K_pad, K_pad), lambda i: (0, 0)),
      ],
      out_specs=pl.BlockSpec((P_t, K_pad), lambda i: (i, 0)),
      compiler_params=None if interpret else pltpu.CompilerParams(dimension_semantics=("parallel",)),
      interpret=interpret,
  )(splat_rows, loc_x_dev, loc_y_dev, u_128_dev)

  unp = out_raw.reshape(B, T, tile_size, tile_size, K_pad)
  rgb_t = unp[..., 0:3]
  depth_t = unp[..., 3]
  nrm_t = unp[..., 4:7]
  alpha_t = unp[..., 7]
  return rgb_t, depth_t, nrm_t, alpha_t


def render_gs3d_batch(
    spec: Rigid3DGSSpec,
    obj_xpos: jax.Array,
    obj_xmat: jax.Array,
    cam_xpos: jax.Array,
    cam_xmat: jax.Array,
    tile_size: int = 8,
    top_k: int = 64,
    mode: str = "jax_tile",
    interpret: bool = True,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Render batched rigid-body-bound 3D Gaussians via static top-K tile binning.

  Returns:
    rgb:   (B, H, W, 3)
    depth: (B, H, W)
    normal:(B, H, W, 3)
    alpha: (B, H, W)
  """
  B = obj_xpos.shape[0]
  H, W = spec.height, spec.width
  assert H % tile_size == 0 and W % tile_size == 0

  bids = jnp.asarray(spec.body_id, dtype=jnp.int32)
  mu_loc = jnp.asarray(spec.mu_local, dtype=jnp.float32)    # (N, 3)
  cov_loc = jnp.asarray(spec.cov_local, dtype=jnp.float32)  # (N, 3, 3)
  nrm_loc = jnp.asarray(spec.nrm_local, dtype=jnp.float32)  # (N, 3)
  rgb_b = jnp.asarray(spec.rgb_base, dtype=jnp.float32)     # (N, 3)
  op_b = jnp.asarray(spec.opacity, dtype=jnp.float32)       # (N,)

  pos_g = obj_xpos[:, bids, :]        # (B, N, 3)
  mat_g = obj_xmat[:, bids, :, :]     # (B, N, 3, 3)
  mu_w = pos_g + jnp.einsum("bnij,nj->bni", mat_g, mu_loc, precision=jax.lax.Precision.HIGHEST)
  cov_w = jnp.einsum("bnij,njk,bnlk->bnil", mat_g, cov_loc, mat_g, precision=jax.lax.Precision.HIGHEST)
  nrm_w = jnp.einsum("bnij,nj->bni", mat_g, nrm_loc, precision=jax.lax.Precision.HIGHEST)
  nrm_w = nrm_w / jnp.maximum(jnp.linalg.norm(nrm_w, axis=-1, keepdims=True), 1e-12)

  # Dynamic Lambertian shading aligned with scene light
  l_pos = jnp.asarray(spec.light_pos, dtype=jnp.float32)
  ldir = l_pos[None, None, :] - mu_w
  ldir = ldir / jnp.maximum(jnp.linalg.norm(ldir, axis=-1, keepdims=True), 1e-12)
  # Ensure normals face camera for consistent shading
  vdir = cam_xpos[:, None, :] - mu_w
  vdir = vdir / jnp.maximum(jnp.linalg.norm(vdir, axis=-1, keepdims=True), 1e-12)
  nrm_vis = jnp.where(jnp.sum(nrm_w * vdir, axis=-1, keepdims=True) < 0.0, -nrm_w, nrm_w)
  ndotl = jnp.maximum(0.0, jnp.sum(nrm_vis * ldir, axis=-1))
  l_amb = jnp.asarray(spec.light_ambient, dtype=jnp.float32)
  l_diff = jnp.asarray(spec.light_diffuse, dtype=jnp.float32)
  rgb_lit = jnp.clip(rgb_b[None, :, :] * (l_amb[None, None, :] + l_diff[None, None, :] * ndotl[..., None]), 0.0, 1.0)

  # Camera projection (MuJoCo/Blender: visible in front of camera has cz < 0, z_d = -cz)
  p_cam = jnp.einsum("bji,bnj->bni", cam_xmat, mu_w - cam_xpos[:, None, :], precision=jax.lax.Precision.HIGHEST)  # (B, N, 3)
  cov_cam = jnp.einsum("bji,bnjk,bkl->bnil", cam_xmat, cov_w, cam_xmat, precision=jax.lax.Precision.HIGHEST)      # (B, N, 3, 3)

  cx, cy, cz = p_cam[..., 0], p_cam[..., 1], p_cam[..., 2]
  z_d = -cz
  in_front = z_d > spec.znear
  z_safe = jnp.maximum(z_d, spec.znear)

  half_h = math.tan(0.5 * math.radians(spec.cam_fovy))
  fy = 0.5 * float(H) / half_h
  fx = fy
  mean_x = 0.5 * float(W) + fx * (cx / z_safe)
  mean_y = 0.5 * float(H) - fy * (cy / z_safe)

  # EWA 2D Jacobian: d(px, py)/d(cx, cy, cz_cam)
  inv_z = 1.0 / z_safe
  inv_z2 = inv_z * inv_z
  # Clamp extreme frustum slopes to avoid EWA explosion outside view
  lim_x = 1.3 * (0.5 * float(W) / fx) * z_safe
  lim_y = 1.3 * (0.5 * float(H) / fy) * z_safe
  cx_c = jnp.clip(cx, -lim_x, lim_x)
  cy_c = jnp.clip(cy, -lim_y, lim_y)

  j00 = fx * inv_z
  j02 = fx * cx_c * inv_z2
  j11 = -fy * inv_z
  j12 = -fy * cy_c * inv_z2

  s00 = j00 * cov_cam[..., 0, 0] + j02 * cov_cam[..., 2, 0]
  s01 = j00 * cov_cam[..., 0, 1] + j02 * cov_cam[..., 2, 1]
  s02 = j00 * cov_cam[..., 0, 2] + j02 * cov_cam[..., 2, 2]
  s10 = j11 * cov_cam[..., 1, 0] + j12 * cov_cam[..., 2, 0]
  s11 = j11 * cov_cam[..., 1, 1] + j12 * cov_cam[..., 2, 1]
  s12 = j11 * cov_cam[..., 1, 2] + j12 * cov_cam[..., 2, 2]

  cov2_xx = s00 * j00 + s02 * j02 + 0.30
  cov2_xy = s01 * j11 + s02 * j12
  cov2_yy = s11 * j11 + s12 * j12 + 0.30

  det = cov2_xx * cov2_yy - cov2_xy * cov2_xy
  valid_g = in_front & (det > 1e-6)
  inv_det = jnp.where(valid_g, 1.0 / jnp.maximum(det, 1e-6), 0.0)
  icxx = cov2_yy * inv_det
  icxy = -cov2_xy * inv_det
  icyy = cov2_xx * inv_det

  # 3-sigma bounding radius from largest eigenvalue of 2x2 covariance
  mid = 0.5 * (cov2_xx + cov2_yy)
  disc = jnp.sqrt(jnp.maximum(mid * mid - det, 0.0))
  lam_max = mid + disc
  radius = 3.0 * jnp.sqrt(jnp.maximum(lam_max, 0.1))

  # Static tile grid (Ty x Tx)
  Ty = H // tile_size
  Tx = W // tile_size
  T = Ty * Tx
  t_ox = np.tile(np.arange(Tx, dtype=np.float32) * tile_size, Ty)
  t_oy = np.repeat(np.arange(Ty, dtype=np.float32) * tile_size, Tx)
  tox_dev = jnp.asarray(t_ox, jnp.float32)[None, :, None]  # (1, T, 1)
  toy_dev = jnp.asarray(t_oy, jnp.float32)[None, :, None]

  # Overlap test between Gaussian circle (mean_x, mean_y, radius) and tile [tox, tox+ts] x [toy, toy+ts]
  near_x = jnp.clip(mean_x[:, None, :], tox_dev, tox_dev + float(tile_size))
  near_y = jnp.clip(mean_y[:, None, :], toy_dev, toy_dev + float(tile_size))
  dx_t = mean_x[:, None, :] - near_x
  dy_t = mean_y[:, None, :] - near_y
  overlap = valid_g[:, None, :] & ((dx_t * dx_t + dy_t * dy_t) <= (radius * radius)[:, None, :])

  # Static top-K closest Gaussians overlapping each tile (higher -z_d = closer to camera)
  score = jnp.where(overlap, -z_d[:, None, :], -1.0e9)
  k_use = min(top_k, spec.n_gaussians)
  top_scores, top_idx = jax.lax.top_k(score, k_use)  # (B, T, K_use)
  top_valid = top_scores > -1.0e8

  def _gather_bt(arr_bn):
    # arr_bn: (B, N) -> (B, T, K_use)
    b_ix = jnp.arange(B, dtype=jnp.int32)[:, None, None]
    return arr_bn[b_ix, top_idx]

  g_mx = _gather_bt(mean_x)
  g_my = _gather_bt(mean_y)
  g_icxx = _gather_bt(icxx)
  g_icxy = _gather_bt(icxy)
  g_icyy = _gather_bt(icyy)
  g_sr = _gather_bt(rgb_lit[..., 0])
  g_sg = _gather_bt(rgb_lit[..., 1])
  g_sb = _gather_bt(rgb_lit[..., 2])
  g_op = jnp.where(top_valid, _gather_bt(jnp.broadcast_to(op_b[None, :], (B, spec.n_gaussians))), 0.0)
  g_sz = _gather_bt(z_d)
  g_nx = _gather_bt(nrm_vis[..., 0])
  g_ny = _gather_bt(nrm_vis[..., 1])
  g_nz = _gather_bt(nrm_vis[..., 2])

  if mode == "pallas":
    t_ox_bt = jnp.broadcast_to(tox_dev[:, :, 0], (B, T))
    t_oy_bt = jnp.broadcast_to(toy_dev[:, :, 0], (B, T))
    rgb_t, dep_t, nrm_t, alp_t = _pallas_gs3d_rasterize(
        g_mx, g_my, g_icxx, g_icxy, g_icyy,
        g_sr, g_sg, g_sb, g_op, g_sz,
        g_nx, g_ny, g_nz,
        t_ox_bt, t_oy_bt, tile_size=tile_size, interpret=interpret,
    )
  else:
    # Pure JAX tile rasterization across (B, T, tile_h, tile_w, K_use)
    lx = jnp.arange(tile_size, dtype=jnp.float32) + 0.5
    ly = jnp.arange(tile_size, dtype=jnp.float32) + 0.5
    px = tox_dev[..., None] + lx[None, None, None, :]  # (1, T, 1, tw)
    py = toy_dev[..., None] + ly[None, None, :, None]  # (1, T, th, 1)

    dx = px[..., None] - g_mx[:, :, None, None, :]
    dy = py[..., None] - g_my[:, :, None, None, :]
    power = -0.5 * (
        g_icxx[:, :, None, None, :] * dx * dx
        + 2.0 * g_icxy[:, :, None, None, :] * dx * dy
        + g_icyy[:, :, None, None, :] * dy * dy
    )
    valid_p = (power <= 0.0) & (power > -16.0)
    alpha_k = jnp.where(
        valid_p,
        jnp.minimum(0.99, g_op[:, :, None, None, :] * jnp.exp(jnp.maximum(power, -16.0))),
        0.0,
    )
    alpha_k = jnp.where(alpha_k >= (1.0 / 255.0), alpha_k, 0.0)
    one_m_a = 1.0 - alpha_k
    trans_excl = jnp.cumprod(
        jnp.concatenate([jnp.ones_like(one_m_a[..., :1]), one_m_a[..., :-1]], axis=-1),
        axis=-1,
    )
    w_k = alpha_k * trans_excl
    alp_t = 1.0 - jnp.prod(one_m_a, axis=-1)
    rgb_t = jnp.stack(
        [
            jnp.sum(w_k * g_sr[:, :, None, None, :], axis=-1),
            jnp.sum(w_k * g_sg[:, :, None, None, :], axis=-1),
            jnp.sum(w_k * g_sb[:, :, None, None, :], axis=-1),
        ],
        axis=-1,
    )
    dep_t = jnp.sum(w_k * g_sz[:, :, None, None, :], axis=-1)
    nrm_t = jnp.stack(
        [
            jnp.sum(w_k * g_nx[:, :, None, None, :], axis=-1),
            jnp.sum(w_k * g_ny[:, :, None, None, :], axis=-1),
            jnp.sum(w_k * g_nz[:, :, None, None, :], axis=-1),
        ],
        axis=-1,
    )

  # Assemble (B, Ty, Tx, tile_h, tile_w, ...) -> (B, H, W, ...)
  rgb_img = rgb_t.reshape(B, Ty, Tx, tile_size, tile_size, 3).transpose(0, 1, 3, 2, 4, 5).reshape(B, H, W, 3)
  dep_raw = dep_t.reshape(B, Ty, Tx, tile_size, tile_size).transpose(0, 1, 3, 2, 4).reshape(B, H, W)
  nrm_img = nrm_t.reshape(B, Ty, Tx, tile_size, tile_size, 3).transpose(0, 1, 3, 2, 4, 5).reshape(B, H, W, 3)
  alp_img = alp_t.reshape(B, Ty, Tx, tile_size, tile_size).transpose(0, 1, 3, 2, 4).reshape(B, H, W)

  # Sky background composite (matching camera row gradient)
  sky_v = jnp.linspace(1.0, 0.0, H, dtype=jnp.float32)[None, :, None, None]
  sky = sky_v * jnp.asarray(spec.sky_rgb1, jnp.float32)[None, None, None, :] + (1.0 - sky_v) * jnp.asarray(spec.sky_rgb2, jnp.float32)[None, None, None, :]
  rgb_final = jnp.clip(rgb_img + (1.0 - alp_img[..., None]) * sky, 0.0, 1.0)
  depth_final = jnp.where(alp_img > 0.1, dep_raw / jnp.maximum(alp_img, 1e-4), -1.0)
  return rgb_final, depth_final, nrm_img, alp_img
