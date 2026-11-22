"""Profile V2 Rigid-Body 3DGS stages on live TPU v5p."""

from __future__ import annotations

import math
import time
import jax
import jax.numpy as jnp
import numpy as np

from tpu_mjx_render.gs3d import (
    _pallas_gs3d_rasterize,
    mesh_scene_to_rigid_3dgs,
    render_gs3d_batch,
)


def load_spec(total_gaussians: int = 2048, height: int = 64, width: int = 64):
  ds = np.load("/dev/shm/zaychik-tpu-mjx/out/assets/blender_real_scene.npz")
  M = int(ds["n_meshes"])
  meshes = [
      {
          "name": str(ds[f"mesh_{m}_name"]),
          "verts": ds[f"mesh_{m}_verts"],
          "faces": ds[f"mesh_{m}_faces"],
          "rgba": ds[f"mesh_{m}_rgba"],
      }
      for m in range(M)
  ]
  spec = mesh_scene_to_rigid_3dgs(
      meshes,
      total_gaussians=total_gaussians,
      height=height,
      width=width,
      cam_fovy=float(ds["cam_fovy"]),
      znear=float(ds["znear"]),
      light_pos=tuple(float(x) for x in ds["light_pos"]),
  )
  return ds, spec


def _tile_inputs(ds, B: int):
  def _tile(arr):
    rep = (B + arr.shape[0] - 1) // arr.shape[0]
    return jnp.asarray(np.tile(arr, (rep,) + (1,) * (arr.ndim - 1))[:B])

  return (
      _tile(ds["obj_xpos"]),
      _tile(ds["obj_xmat"]),
      _tile(ds["cam_xpos"]),
      _tile(ds["cam_xmat"]),
  )


def bench_fn(fn, args, n_warmup: int = 3, n_steps: int = 10):
  t_c0 = time.perf_counter()
  out = fn(*args)
  jax.block_until_ready(out)
  compile_s = time.perf_counter() - t_c0
  for _ in range(n_warmup):
    out = fn(*args)
  jax.block_until_ready(out)
  t0 = time.perf_counter()
  for _ in range(n_steps):
    out = fn(*args)
  jax.block_until_ready(out)
  ms = (time.perf_counter() - t0) * 1e3 / n_steps
  return ms, compile_s, out


def stage_frontend_and_binning(spec, obj_xpos, obj_xmat, cam_xpos, cam_xmat, tile_size=8, top_k=128):
  B = obj_xpos.shape[0]
  H, W = spec.height, spec.width
  bids = jnp.asarray(spec.body_id, dtype=jnp.int32)
  mu_loc = jnp.asarray(spec.mu_local, dtype=jnp.float32)
  cov_loc = jnp.asarray(spec.cov_local, dtype=jnp.float32)
  nrm_loc = jnp.asarray(spec.nrm_local, dtype=jnp.float32)
  rgb_b = jnp.asarray(spec.rgb_base, dtype=jnp.float32)
  op_b = jnp.asarray(spec.opacity, dtype=jnp.float32)

  pos_g = obj_xpos[:, bids, :]
  mat_g = obj_xmat[:, bids, :, :]
  mu_w = pos_g + jnp.einsum("bnij,nj->bni", mat_g, mu_loc, precision=jax.lax.Precision.HIGHEST)
  cov_w = jnp.einsum("bnij,njk,bnlk->bnil", mat_g, cov_loc, mat_g, precision=jax.lax.Precision.HIGHEST)
  nrm_w = jnp.einsum("bnij,nj->bni", mat_g, nrm_loc, precision=jax.lax.Precision.HIGHEST)
  nrm_w = nrm_w / jnp.maximum(jnp.linalg.norm(nrm_w, axis=-1, keepdims=True), 1e-12)

  l_pos = jnp.asarray(spec.light_pos, dtype=jnp.float32)
  ldir = l_pos[None, None, :] - mu_w
  ldir = ldir / jnp.maximum(jnp.linalg.norm(ldir, axis=-1, keepdims=True), 1e-12)
  vdir = cam_xpos[:, None, :] - mu_w
  vdir = vdir / jnp.maximum(jnp.linalg.norm(vdir, axis=-1, keepdims=True), 1e-12)
  nrm_vis = jnp.where(jnp.sum(nrm_w * vdir, axis=-1, keepdims=True) < 0.0, -nrm_w, nrm_w)
  ndotl = jnp.maximum(0.0, jnp.sum(nrm_vis * ldir, axis=-1))
  l_amb = jnp.asarray(spec.light_ambient, dtype=jnp.float32)
  l_diff = jnp.asarray(spec.light_diffuse, dtype=jnp.float32)
  rgb_lit = jnp.clip(rgb_b[None, :, :] * (l_amb[None, None, :] + l_diff[None, None, :] * ndotl[..., None]), 0.0, 1.0)

  p_cam = jnp.einsum("bji,bnj->bni", cam_xmat, mu_w - cam_xpos[:, None, :], precision=jax.lax.Precision.HIGHEST)
  cov_cam = jnp.einsum("bji,bnjk,bkl->bnil", cam_xmat, cov_w, cam_xmat, precision=jax.lax.Precision.HIGHEST)

  cx, cy, cz = p_cam[..., 0], p_cam[..., 1], p_cam[..., 2]
  z_d = -cz
  in_front = z_d > spec.znear
  z_safe = jnp.maximum(z_d, spec.znear)

  half_h = math.tan(0.5 * math.radians(spec.cam_fovy))
  fy = 0.5 * float(H) / half_h
  fx = fy
  mean_x = 0.5 * float(W) + fx * (cx / z_safe)
  mean_y = 0.5 * float(H) - fy * (cy / z_safe)

  inv_z = 1.0 / z_safe
  inv_z2 = inv_z * inv_z
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

  mid = 0.5 * (cov2_xx + cov2_yy)
  disc = jnp.sqrt(jnp.maximum(mid * mid - det, 0.0))
  lam_max = mid + disc
  radius = 3.0 * jnp.sqrt(jnp.maximum(lam_max, 0.1))

  return mean_x, mean_y, icxx, icxy, icyy, rgb_lit, op_b, z_d, nrm_vis, valid_g, radius


def stage_bin_only(mean_x, mean_y, z_d, valid_g, radius, H, W, tile_size, top_k, N):
  Ty = H // tile_size
  Tx = W // tile_size
  T = Ty * Tx
  t_ox = np.tile(np.arange(Tx, dtype=np.float32) * tile_size, Ty)
  t_oy = np.repeat(np.arange(Ty, dtype=np.float32) * tile_size, Tx)
  tox_dev = jnp.asarray(t_ox, jnp.float32)[None, :, None]
  toy_dev = jnp.asarray(t_oy, jnp.float32)[None, :, None]
  near_x = jnp.clip(mean_x[:, None, :], tox_dev, tox_dev + float(tile_size))
  near_y = jnp.clip(mean_y[:, None, :], toy_dev, toy_dev + float(tile_size))
  dx_t = mean_x[:, None, :] - near_x
  dy_t = mean_y[:, None, :] - near_y
  overlap = valid_g[:, None, :] & ((dx_t * dx_t + dy_t * dy_t) <= (radius * radius)[:, None, :])
  score = jnp.where(overlap, -z_d[:, None, :], -1.0e9)
  k_use = min(top_k, N)
  top_scores, top_idx = jax.lax.top_k(score, k_use)
  return top_scores, top_idx


def main():
  ds, spec = load_spec(2048, 64, 64)
  B = 1024
  args = _tile_inputs(ds, B)

  # 1. Full E2E baselines
  for mode, ts, tk in [
      ("jax_tile", 8, 128),
      ("pallas", 8, 128),
      ("jax_tile", 8, 64),
      ("jax_tile", 8, 32),
      ("jax_tile", 16, 128),
      ("jax_tile", 16, 64),
  ]:
    fn = jax.jit(lambda a, b, c, d: render_gs3d_batch(spec, a, b, c, d, tile_size=ts, top_k=tk, mode=mode, interpret=False)[0])
    ms, comp_s, _ = bench_fn(fn, args, n_warmup=2, n_steps=5)
    print(f"E2E {mode:10s} ts={ts:2d} k={tk:3d}: {ms:8.2f} ms ({B*1e3/ms:8.1f} FPS), compile={comp_s:.2f}s", flush=True)

  # 2. Quality comparison against ts=8, k=128
  ref_rgb, ref_dep, _, ref_alp = jax.jit(lambda a, b, c, d: render_gs3d_batch(spec, a, b, c, d, tile_size=8, top_k=128, mode="jax_tile"))(*args)
  for ts, tk in [(8, 64), (8, 32), (16, 128), (16, 64), (16, 32)]:
    r_rgb, r_dep, _, r_alp = jax.jit(lambda a, b, c, d: render_gs3d_batch(spec, a, b, c, d, tile_size=ts, top_k=tk, mode="jax_tile"))(*args)
    mae = float(jnp.mean(jnp.abs(ref_rgb[:32] - r_rgb[:32])))
    mx = float(jnp.max(jnp.abs(ref_rgb[:32] - r_rgb[:32])))
    mse = float(jnp.mean((ref_rgb[:32] - r_rgb[:32]) ** 2))
    psnr = -10.0 * math.log10(max(mse, 1e-12))
    print(f"PARITY vs ts=8,k=128 | ts={ts:2d} k={tk:3d}: rgb_mae={mae:.6f}, max_diff={mx:.6f}, psnr={psnr:.2f} dB", flush=True)

  # 3. Stage-by-stage timing breakdown
  fn_ewa = jax.jit(lambda a, b, c, d: stage_frontend_and_binning(spec, a, b, c, d))
  ms_ewa, _, ewa_out = bench_fn(fn_ewa, args, n_warmup=2, n_steps=10)
  print(f"STAGE EWA_PROJ (B={B}, N=2048): {ms_ewa:.2f} ms", flush=True)

  mean_x, mean_y, icxx, icxy, icyy, rgb_lit, op_b, z_d, nrm_vis, valid_g, radius = ewa_out
  for ts, tk in [(8, 128), (8, 64), (8, 32), (16, 128), (16, 64)]:
    fn_bin = jax.jit(lambda mx_, my_, zd_, vg_, rad_: stage_bin_only(mx_, my_, zd_, vg_, rad_, 64, 64, ts, tk, 2048))
    ms_bin, _, bin_out = bench_fn(fn_bin, (mean_x, mean_y, z_d, valid_g, radius), n_warmup=2, n_steps=10)
    print(f"STAGE TOPK_ONLY ts={ts:2d} k={tk:3d}: {ms_bin:.2f} ms", flush=True)

    # Also measure gather + topk
    def _bin_and_gather(mx_, my_, zd_, vg_, rad_, icxx_, icxy_, icyy_, rgbl_, op_, nv_):
      ts_, ti_ = stage_bin_only(mx_, my_, zd_, vg_, rad_, 64, 64, ts, tk, 2048)
      tv_ = ts_ > -1.0e8
      b_ix = jnp.arange(B, dtype=jnp.int32)[:, None, None]
      g_op = jnp.where(tv_, jnp.broadcast_to(op_[None, :], (B, 2048))[b_ix, ti_], 0.0)
      return (
          mx_[b_ix, ti_], my_[b_ix, ti_],
          icxx_[b_ix, ti_], icxy_[b_ix, ti_], icyy_[b_ix, ti_],
          rgbl_[b_ix, ti_, 0], rgbl_[b_ix, ti_, 1], rgbl_[b_ix, ti_, 2],
          g_op, zd_[b_ix, ti_],
          nv_[b_ix, ti_, 0], nv_[b_ix, ti_, 1], nv_[b_ix, ti_, 2],
      )
    fn_bg = jax.jit(_bin_and_gather)
    ms_bg, _, g_out = bench_fn(fn_bg, (mean_x, mean_y, z_d, valid_g, radius, icxx, icxy, icyy, rgb_lit, op_b, nrm_vis), n_warmup=2, n_steps=10)
    print(f"STAGE TOPK+GATHER ts={ts:2d} k={tk:3d}: {ms_bg:.2f} ms", flush=True)


if __name__ == "__main__":
  main()
