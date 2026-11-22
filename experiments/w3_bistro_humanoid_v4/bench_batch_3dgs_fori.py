"""Wave 4 Real-Time Batch 3DGS E2E `fori_loop` Scaling (B=1024..4096, 4-Chip Vision RL Ready).

Benchmarks end-to-end `jax.lax.fori_loop` camera-orbiting 3DGS rollouts inside 4-chip
`shard_map` at 64x64 and 128x128 across:
  - B in {1024, 2048, 4096} parallel environments (256..1024 per chip)
  - N in {8192, 16384, 32768, 65536} Gaussians per env (sampled from Bistro!)
Verifies `host_callbacks == 0`, `send_recv_ops == 0`, `gather_ops == 0` in HLO.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import re
import time
from typing import Any, Dict, Tuple

import jax
try:
  from jax import shard_map
except ImportError:
  from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
import numpy as np

from tpu_mjx_render.roofline import inspect_hlo_ops
from tpu_gsplat import (
    _pallas_gs3d_v3_fused_rasterize,
    _quats_to_rotmat,
)
from cinema_3dgs import load_gltf_scene_cinema, sample_cinema_3dgs

BISTRO_GLTF = "/dev/shm/zaychik-tpu-mjx-w3/assets/niagara_bistro/bistro.gltf"


def _check_hlo_zero(lowered, name: str) -> Dict[str, Any]:
  txt = lowered.as_text()
  info = inspect_hlo_ops(txt)
  info["send_recv_ops"] = len(re.findall(r"\bsend\(|\brecv\(", txt, re.I))
  assert info["host_callbacks"] == 0, f"{name} host_callbacks != 0: {info}"
  assert info["send_recv_ops"] == 0, f"{name} send_recv_ops != 0: {info}"
  assert info["gather_ops"] == 0, f"{name} gather_ops != 0: {info}"
  info["zero_copy_verified"] = True
  return info


def _project_static_scene_microbatch(
    means: jax.Array,      # (N, 3)
    M_world: jax.Array,    # (N, 3, 3)
    opacities: jax.Array,  # (N,)
    colors: jax.Array,     # (N, 3)
    viewmats: jax.Array,   # (C, 4, 4)
    Ks: jax.Array,         # (C, 3, 3)
    width: int,
    height: int,
    near_plane: float = 0.05,
    far_plane: float = 100.0,
    eps2d: float = 0.30,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
  """Project static 3DGS scene across a microbatch of C orbiting cameras (zero gathers)."""
  C = viewmats.shape[0]
  N = means.shape[0]
  N_pad = ((N + 127) // 128) * 128

  R_wc = viewmats[:, :3, :3]  # (C, 3, 3)
  t_wc = viewmats[:, :3, 3]   # (C, 3)
  p_cam = jnp.einsum("cij,nj->cni", R_wc, means, precision=jax.lax.Precision.HIGHEST) + t_wc[:, None, :]

  cx, cy, cz = p_cam[..., 0], p_cam[..., 1], p_cam[..., 2]
  z_d = cz
  in_front = (z_d > near_plane) & (z_d < far_plane)
  z_safe = jnp.maximum(z_d, near_plane)

  fx = Ks[:, 0, 0][:, None]
  fy = Ks[:, 1, 1][:, None]
  cx0 = Ks[:, 0, 2][:, None]
  cy0 = Ks[:, 1, 2][:, None]

  mean_x = fx * (cx / z_safe) + cx0
  mean_y = fy * (cy / z_safe) + cy0

  inv_z = 1.0 / z_safe
  inv_z2 = inv_z * inv_z
  tan_x = 0.5 * float(width) / jnp.maximum(fx, 1e-6)
  tan_y = 0.5 * float(height) / jnp.maximum(fy, 1e-6)
  lim_x = 1.3 * tan_x * z_safe
  lim_y = 1.3 * tan_y * z_safe
  tx = jnp.clip(cx, -lim_x, lim_x)
  ty = jnp.clip(cy, -lim_y, lim_y)

  j00 = fx * inv_z
  j02 = -fx * tx * inv_z2
  j11 = fy * inv_z
  j12 = -fy * ty * inv_z2

  w0 = j00[..., None] * R_wc[:, None, 0, :] + j02[..., None] * R_wc[:, None, 2, :]
  w1 = j11[..., None] * R_wc[:, None, 1, :] + j12[..., None] * R_wc[:, None, 2, :]

  a0 = jnp.einsum("cnj,njk->cnk", w0, M_world, precision=jax.lax.Precision.HIGHEST)
  a1 = jnp.einsum("cnj,njk->cnk", w1, M_world, precision=jax.lax.Precision.HIGHEST)

  cov2_xx = jnp.sum(a0 * a0, axis=-1) + eps2d
  cov2_xy = jnp.sum(a0 * a1, axis=-1)
  cov2_yy = jnp.sum(a1 * a1, axis=-1) + eps2d

  det = cov2_xx * cov2_yy - cov2_xy * cov2_xy
  valid_g = in_front & (det > 1e-6)
  inv_det = jnp.where(valid_g, 1.0 / jnp.maximum(det, 1e-6), 0.0)
  icxx = cov2_yy * inv_det
  icxy = -cov2_xy * inv_det
  icyy = cov2_xx * inv_det

  mid = 0.5 * (cov2_xx + cov2_yy)
  disc = jnp.sqrt(jnp.maximum(mid * mid - det, 0.0))
  radius = 3.0 * jnp.sqrt(jnp.maximum(mid + disc, 0.1))

  op_cn = jnp.where(valid_g, jnp.broadcast_to(jnp.clip(opacities, 0.0, 1.0)[None, :], (C, N)), 0.0)
  col_cn = jnp.broadcast_to(colors[None, :, :], (C, N, 3))
  z_cn = jnp.zeros((C, N), dtype=jnp.float32)

  splats_c16n = jnp.stack(
      [
          mean_x, mean_y, icxx, icxy, icyy,
          col_cn[..., 0], col_cn[..., 1], col_cn[..., 2],
          op_cn, z_d, z_cn, z_cn, z_cn,
          z_cn, z_cn, z_cn,
      ],
      axis=1,
  )
  if N_pad > N:
    splats_c16n = jnp.pad(splats_c16n, ((0, 0), (0, 0), (0, N_pad - N)))
  return splats_c16n, mean_x, mean_y, z_d, valid_g, radius


def make_fori_orbit_3dgs_4c_fn(
    mesh: jax.sharding.Mesh,
    height: int,
    width: int,
    n_gaussians: int,
    batch_total: int,
    n_steps: int = 8,
    micro_b: int = 64,
    top_k: int = 128,
    use_bf16_mxu: bool = False,
):
  """Build 4-chip sharded E2E `fori_loop` camera-orbiting 3DGS rollout benchmark."""
  n_chips = len(mesh.devices)
  assert batch_total % n_chips == 0
  b_loc = batch_total // n_chips
  assert b_loc % micro_b == 0
  n_mb = b_loc // micro_b

  tile_size = 8
  assert height % 32 == 0 and width % 32 == 0
  Ty, Tx = height // tile_size, width // tile_size
  T = Ty * Tx
  t_ox = np.tile(np.arange(Tx, dtype=np.float32) * tile_size, Ty)
  t_oy = np.repeat(np.arange(Ty, dtype=np.float32) * tile_size, Tx)
  tox_dev = jnp.asarray(t_ox, dtype=jnp.float32)[None, :, None]
  toy_dev = jnp.asarray(t_oy, dtype=jnp.float32)[None, :, None]

  half_h = math.tan(0.5 * math.radians(50.0))
  fy = 0.5 * float(height) / half_h
  fx = fy
  K_single = np.array(
      [[fx, 0.0, 0.5 * float(width)], [0.0, fy, 0.5 * float(height)], [0.0, 0.0, 1.0]],
      dtype=np.float32,
  )
  K_mb = jnp.asarray(np.broadcast_to(K_single[None, :, :], (micro_b, 3, 3)))

  P = jax.sharding.PartitionSpec
  sm_kw = {"check_rep": False} if "check_rep" in inspect.signature(shard_map).parameters else {"check_vma": False}

  def _render_microbatch(means, M_world, opacities, colors, cam_pos_mb, target_mb):
    # Compute OpenCV world-to-camera viewmats for microbatch (micro_b, 4, 4)
    fwd = target_mb - cam_pos_mb
    fwd = fwd / jnp.maximum(jnp.linalg.norm(fwd, axis=-1, keepdims=True), 1e-12)
    up_w = jnp.broadcast_to(jnp.array([[0.0, 1.0, 0.0]], dtype=jnp.float32), fwd.shape)
    right = jnp.cross(fwd, up_w)
    right = right / jnp.maximum(jnp.linalg.norm(right, axis=-1, keepdims=True), 1e-12)
    down = jnp.cross(fwd, right)
    R_wc = jnp.stack([right, down, fwd], axis=1)  # (micro_b, 3, 3)
    t_wc = -jnp.einsum("cij,cj->ci", R_wc, cam_pos_mb, precision=jax.lax.Precision.HIGHEST)
    vm = jnp.broadcast_to(jnp.eye(4, dtype=jnp.float32), (micro_b, 4, 4)).copy()
    vm = vm.at[:, :3, :3].set(R_wc).at[:, :3, 3].set(t_wc)

    splats_c16n, mean_x, mean_y, z_d, valid_g, radius = _project_static_scene_microbatch(
        means, M_world, opacities, colors, vm, K_mb, width=width, height=height,
    )
    near_x = jnp.clip(mean_x[:, None, :], tox_dev, tox_dev + float(tile_size))
    near_y = jnp.clip(mean_y[:, None, :], toy_dev, toy_dev + float(tile_size))
    dx_t = mean_x[:, None, :] - near_x
    dy_t = mean_y[:, None, :] - near_y
    overlap = valid_g[:, None, :] & ((dx_t * dx_t + dy_t * dy_t) <= (radius * radius)[:, None, :])

    score = jnp.where(overlap, -z_d[:, None, :], -1.0e9)
    top_scores, top_idx = jax.lax.approx_max_k(score, top_k, recall_target=1.0)
    neg_ts, top_idx = jax.lax.sort_key_val(-top_scores, top_idx, dimension=-1)
    top_scores = -neg_ts
    top_idx = jnp.where(top_scores > -1.0e8, top_idx, -1)

    rgb_t, _, alp_t = _pallas_gs3d_v3_fused_rasterize(
        splats_c16n, top_idx, t_ox, t_oy,
        tile_size=tile_size, use_bf16_mxu=use_bf16_mxu, t_grp=16, interpret=False,
    )
    rgb_img = rgb_t.reshape(micro_b, Ty, Tx, tile_size, tile_size, 3).transpose(0, 1, 3, 2, 4, 5).reshape(micro_b, height, width, 3)
    return jnp.mean(rgb_img)

  def _shard_rollout(means, quats, scales, opacities, colors, theta0, radius0, y0, omega, target0):
    R_q = _quats_to_rotmat(quats)
    M_world = R_q * jnp.maximum(scales, 1e-6)[:, None, :]

    def _step_fn(step_idx, carry):
      th, acc_sum = carry
      th_next = th + omega
      cam_x = target0[:, 0] + radius0 * jnp.cos(th_next)
      cam_z = target0[:, 2] + radius0 * jnp.sin(th_next)
      cam_pos = jnp.stack([cam_x, y0, cam_z], axis=-1)

      if n_mb == 1:
        step_mean = _render_microbatch(means, M_world, opacities, colors, cam_pos, target0)
      else:
        cp_mb = cam_pos.reshape(n_mb, micro_b, 3)
        tg_mb = target0.reshape(n_mb, micro_b, 3)

        def _mb_scan(c_sum, xs):
          cp_i, tg_i = xs
          m_val = _render_microbatch(means, M_world, opacities, colors, cp_i, tg_i)
          return c_sum + m_val, None

        step_mean, _ = jax.lax.scan(_mb_scan, jnp.array(0.0, dtype=jnp.float32), (cp_mb, tg_mb))
        step_mean = step_mean / float(n_mb)
      return (th_next, acc_sum + step_mean)

    th_end, total_sum = jax.lax.fori_loop(
        0, n_steps, _step_fn, (theta0, jnp.array(0.0, dtype=jnp.float32))
    )
    return th_end, total_sum / float(n_steps)

  sm = shard_map(
      _shard_rollout,
      mesh=mesh,
      in_specs=(P(), P(), P(), P(), P(), P("dp"), P("dp"), P("dp"), P("dp"), P("dp")),
      out_specs=(P("dp"), P("dp")),
      **sm_kw,
  )
  return jax.jit(sm)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--out_json", default="/dev/shm/zaychik-tpu-mjx-w3/out/v3_w3_wave4_bistro_humanoid.json")
  ap.add_argument("--n_steps", type=int, default=8)
  args = ap.parse_args()

  devs = jax.devices()[:4]
  mesh = jax.sharding.Mesh(np.array(devs), ("dp",))
  rep = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
  dp_sh = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("dp"))

  sc = load_gltf_scene_cinema(BISTRO_GLTF, load_normal_maps=False)
  cam_pos_ref = np.array([-19.4, 1.88, 12.60], dtype=np.float32)
  target_ref = np.array([-13.0, 1.2, 14.5], dtype=np.float32)
  cam_fwd_ref = (target_ref - cam_pos_ref) / np.linalg.norm(target_ref - cam_pos_ref)

  if os.path.exists(args.out_json):
    with open(args.out_json) as f:
      data = json.load(f)
  else:
    data = {
        "backend": jax.default_backend(),
        "device_kind": jax.devices()[0].device_kind,
        "jax_version": jax.__version__,
    }

  results = []
  n_list = [8192, 16384, 32768, 65536]
  b_list = [1024, 2048, 4096]
  res_list = [(64, 64), (128, 128)]

  for N in n_list:
    gs = sample_cinema_3dgs(
        sc, n_gaussians=N, seed=42, splat_scale=1.1, thick_ratio=0.20,
        cam_pos=cam_pos_ref, cam_fwd=cam_fwd_ref, focus_radius=12.0,
    )
    # Extract simple isotropic/anisotropic scales & identity quats from cov6 diagonal for tpu_gsplat benchmark
    scales_np = np.sqrt(np.maximum(np.stack([gs["cov6"][:, 0], gs["cov6"][:, 3], gs["cov6"][:, 5]], axis=-1), 1e-6)).astype(np.float32)
    quats_np = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (N, 1))
    cols_np = np.clip(gs["albedo_lin"] * 1.4, 0.0, 1.0).astype(np.float32)

    means_d = jax.device_put(jnp.asarray(gs["mu"]), rep)
    quats_d = jax.device_put(jnp.asarray(quats_np), rep)
    scales_d = jax.device_put(jnp.asarray(scales_np), rep)
    op_d = jax.device_put(jnp.asarray(gs["opacity"]), rep)
    cols_d = jax.device_put(jnp.asarray(cols_np), rep)

    for H, W in res_list:
      for B in b_list:
        micro_b = 64 if (N >= 32768 or H == 128) else 128
        if B // 4 < micro_b:
          micro_b = B // 4

        rng = np.random.default_rng(7)
        th0 = jax.device_put(jnp.asarray(rng.uniform(0.0, 2.0 * math.pi, (B,)).astype(np.float32)), dp_sh)
        rad0 = jax.device_put(jnp.asarray(rng.uniform(4.5, 7.5, (B,)).astype(np.float32)), dp_sh)
        y0 = jax.device_put(jnp.asarray(rng.uniform(1.4, 2.2, (B,)).astype(np.float32)), dp_sh)
        omega = jax.device_put(jnp.full((B,), 0.05, dtype=np.float32), dp_sh)
        tgt0 = jax.device_put(jnp.asarray(np.broadcast_to(target_ref[None, :], (B, 3))), dp_sh)

        fn = make_fori_orbit_3dgs_4c_fn(
            mesh, height=H, width=W, n_gaussians=N, batch_total=B,
            n_steps=args.n_steps, micro_b=micro_b, top_k=128, use_bf16_mxu=False,
        )
        call_args = (means_d, quats_d, scales_d, op_d, cols_d, th0, rad0, y0, omega, tgt0)

        if (N == 8192 and B == 1024 and H == 64) or (N == 65536 and B == 4096 and H == 128):
          hlo = _check_hlo_zero(fn.lower(*call_args), f"fori_3dgs_N{N}_B{B}_{H}x{W}")
          data.setdefault("hlo_checks", {})[f"fori_3dgs_N{N}_B{B}_{H}x{W}"] = hlo

        t0 = time.perf_counter()
        out = fn(*call_args)
        jax.block_until_ready(out)
        compile_s = time.perf_counter() - t0

        out = fn(*call_args)
        jax.block_until_ready(out)

        t1 = time.perf_counter()
        n_rep = 3
        for _ in range(n_rep):
          out = fn(*call_args)
        jax.block_until_ready(out)
        total_s = (time.perf_counter() - t1) / n_rep
        ms_per_step = (total_s * 1e3) / args.n_steps
        env_steps_per_s = (B * args.n_steps) / total_s

        rec = {
            "n_chips": 4,
            "n_gaussians": N,
            "res": f"{W}x{H}",
            "batch_B": B,
            "micro_b": micro_b,
            "n_fori_steps": args.n_steps,
            "ms_per_batch_step": round(ms_per_step, 3),
            "env_steps_per_s": round(env_steps_per_s, 1),
            "compile_s": round(compile_s, 2),
        }
        results.append(rec)
        print(f"3DGS FORI N={N:5d} {W}x{H:3d} B={B:4d}: {ms_per_step:7.3f} ms/step | {env_steps_per_s:10.1f} env-steps/s (comp {compile_s:.1f}s)", flush=True)

        data["batch_3dgs_fori_scaling"] = results
        with open(args.out_json, "w") as f:
          json.dump(data, f, indent=2)


if __name__ == "__main__":
  main()
