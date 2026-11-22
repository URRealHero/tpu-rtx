"""Full V3 Wide-BVH Verification & Multi-Scale Stress Benchmark (1-chip & 4-chip TPU v5p)."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import jax
import jax.numpy as jnp
import numpy as np

from tpu_mjx_render.mesh_ray import build_mesh_scene_spec, render_mesh_batch
from tpu_mjx_render.roofline import inspect_hlo_ops
from wide_bvh_v3 import (
    _compute_smooth_vertex_normals,
    build_wide_bvh_scene_spec,
    make_sharded_wide_bvh_renderer,
    render_wide_bvh_batch,
    to_device_arrays,
)


def subdivide_mesh_barycentric(verts: np.ndarray, faces: np.ndarray, k: int):
  if k <= 1:
    return verts, faces, _compute_smooth_vertex_normals(verts, faces)
  vnorm = _compute_smooth_vertex_normals(verts, faces)
  pts_uv = []
  idx_map = {}
  for i in range(k + 1):
    for j in range(k + 1 - i):
      idx_map[(i, j)] = len(pts_uv)
      pts_uv.append((i / k, j / k))
  pts_uv = np.array(pts_uv, dtype=np.float32)
  P = pts_uv.shape[0]
  sub_faces = []
  for i in range(k):
    for j in range(k - i):
      a = idx_map[(i, j)]
      b = idx_map[(i + 1, j)]
      c = idx_map[(i, j + 1)]
      sub_faces.append((a, b, c))
      if i + j + 1 < k:
        d = idx_map[(i + 1, j + 1)]
        sub_faces.append((b, d, c))
  sub_faces = np.array(sub_faces, dtype=np.int32)
  F = faces.shape[0]
  v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
  n0, n1, n2 = vnorm[faces[:, 0]], vnorm[faces[:, 1]], vnorm[faces[:, 2]]
  u = pts_uv[None, :, 0:1]
  v = pts_uv[None, :, 1:2]
  w = 1.0 - u - v
  new_verts = (w * v0[:, None, :] + u * v1[:, None, :] + v * v2[:, None, :]).reshape(F * P, 3)
  new_norms = (w * n0[:, None, :] + u * n1[:, None, :] + v * n2[:, None, :]).reshape(F * P, 3)
  new_norms = (new_norms / np.maximum(np.linalg.norm(new_norms, axis=-1, keepdims=True), 1e-12)).astype(np.float32)
  offsets = (np.arange(F, dtype=np.int32) * P)[:, None, None]
  new_faces = (sub_faces[None, :, :] + offsets).reshape(F * k * k, 3)
  return new_verts.astype(np.float32), new_faces, new_norms


def load_base_scene():
  ds = np.load("/dev/shm/zaychik-tpu-mjx/out/assets/blender_real_scene.npz")
  gt = np.load("/dev/shm/zaychik-tpu-mjx/out/assets/blender_real_scene_gt.npz")
  M = int(ds["n_meshes"])
  base_meshes = [
      {
          "name": str(ds[f"mesh_{m}_name"]),
          "verts": ds[f"mesh_{m}_verts"],
          "faces": ds[f"mesh_{m}_faces"],
          "rgba": ds[f"mesh_{m}_rgba"],
      }
      for m in range(M)
  ]
  return ds, gt, base_meshes


def make_scaled_meshes(base_meshes, k: int):
  out = []
  for m in base_meshes:
    nv, nf, nn = subdivide_mesh_barycentric(m["verts"], m["faces"], k)
    out.append({"name": m["name"], "verts": nv, "faces": nf, "vert_normals": nn, "rgba": m["rgba"]})
  return out


def spec_stats(spec):
  total_tris = sum(it.n_tris_orig for it in spec.meshes)
  total_leaves = sum(it.n_leaves for it in spec.meshes)
  total_l1 = sum(it.n_l1 for it in spec.meshes)
  total_l2 = sum(it.n_l2 for it in spec.meshes)
  hbm_bytes = sum(
      it.leaf_bank.nbytes + it.l0_aabbs.nbytes + it.l2_aabbs.nbytes
      for it in spec.meshes
  )
  return {
      "total_tris": int(total_tris),
      "total_leaves_128": int(total_leaves),
      "total_l1_64": int(total_l1),
      "total_l2_64": int(total_l2),
      "bvh_hbm_mb": round(hbm_bytes / (1024 * 1024), 3),
  }


def tile_poses(ds, B: int):
  def _tile(arr):
    rep = (B + arr.shape[0] - 1) // arr.shape[0]
    return jnp.asarray(np.tile(arr, (rep,) + (1,) * (arr.ndim - 1))[:B])
  return _tile(ds["obj_xpos"]), _tile(ds["obj_xmat"]), _tile(ds["cam_xpos"]), _tile(ds["cam_xmat"])


def run_1chip_suite(out_path: str):
  ds, gt, base_meshes = load_base_scene()
  devs = jax.devices()
  assert len(devs) == 1, f"Expected 1 device for 1c suite, got {devs}"
  results = {
      "backend": jax.default_backend(),
      "jax_version": jax.__version__,
      "device_kind": devs[0].device_kind,
      "v2_1_ref_4620_1c_64x64_fps": 1802.8,
      "parity": {},
      "runs_1c": [],
      "runs_4c": [],
  }

  kw = dict(
      cam_fovy=float(ds["cam_fovy"]),
      znear=float(ds["znear"]),
      zfar=float(ds["zfar"]),
      light_pos=tuple(float(x) for x in ds["light_pos"]),
  )

  # 1. Exactness Parity on 4,620-tri Blender scene
  mspec_v2 = build_mesh_scene_spec(base_meshes, height=64, width=64, **kw)
  mspec_v3 = build_wide_bvh_scene_spec(base_meshes, height=64, width=64, **kw)
  dev_v3 = to_device_arrays(mspec_v3)
  idx_eval = np.array([0] + list(range(len(ds["split"]) - 7, len(ds["split"]))), dtype=np.int32)
  oxp = jnp.asarray(ds["obj_xpos"][idx_eval])
  oxm = jnp.asarray(ds["obj_xmat"][idx_eval])
  cxp = jnp.asarray(ds["cam_xpos"][idx_eval])
  cxm = jnp.asarray(ds["cam_xmat"][idx_eval])

  fn_v2 = jax.jit(lambda p, m, cp, cm: render_mesh_batch(mspec_v2, p, m, cp, cm, mode="jax_cluster"))
  fn_v3 = jax.jit(lambda d, p, m, cp, cm: render_wide_bvh_batch(mspec_v3, d, p, m, cp, cm, interpret=False))
  gb_v2 = fn_v2(oxp, oxm, cxp, cxm)
  gb_v3 = fn_v3(dev_v3, oxp, oxm, cxp, cxm)
  jax.block_until_ready(gb_v3.rgb)

  f0_cy_z = np.asarray(gt["frame0_z_depth"], np.float32)
  f0_v3_z = np.asarray(gb_v3.z_depth[0], np.float32)
  fg_cy = f0_cy_z < 20.0
  fg_v3 = f0_v3_z > 0.0
  both_fg = fg_cy & fg_v3

  hlo_txt = fn_v3.lower(dev_v3, oxp, oxm, cxp, cxm).as_text()
  hlo_info = inspect_hlo_ops(hlo_txt)
  hlo_info["send_recv_ops"] = len(re.findall(r"\bsend\(|\brecv\(", hlo_txt, re.I))

  p_4620 = {
      "total_tris": 4620,
      "B_eval": 8,
      "max_depth_diff_vs_jax_cluster_m": float(jnp.max(jnp.abs(gb_v2.depth - gb_v3.depth))),
      "depth_mae_vs_jax_cluster_m": float(jnp.mean(jnp.abs(gb_v2.depth - gb_v3.depth))),
      "max_rgb_diff_vs_jax_cluster": float(jnp.max(jnp.abs(gb_v2.rgb - gb_v3.rgb))),
      "geom_match_vs_jax_cluster": float(jnp.mean(gb_v2.geom_id == gb_v3.geom_id)),
      "frame0_z_depth_mae_vs_cycles_m": float(np.mean(np.abs(f0_cy_z[both_fg] - f0_v3_z[both_fg]))),
      "frame0_fg_mask_match_vs_cycles": float(np.mean(fg_cy == fg_v3)),
      "hlo": hlo_info,
  }
  assert p_4620["max_depth_diff_vs_jax_cluster_m"] < 1e-4
  assert p_4620["geom_match_vs_jax_cluster"] >= 0.999
  assert p_4620["frame0_z_depth_mae_vs_cycles_m"] < 1e-4
  assert hlo_info["host_callbacks"] == 0 and hlo_info["send_recv_ops"] == 0
  results["parity"]["blender_4620"] = p_4620
  print("PARITY 4620 PASS:", p_4620, flush=True)

  # 2. 1.18M-tri self-consistency parity: spatial_median BVH vs morton BVH + Cycles Z-depth
  meshes_1m = make_scaled_meshes(base_meshes, 16)
  spec_sm = build_wide_bvh_scene_spec(meshes_1m, height=64, width=64, build_method="spatial_median", **kw)
  spec_mo = build_wide_bvh_scene_spec(meshes_1m, height=64, width=64, build_method="morton", **kw)
  dev_sm = to_device_arrays(spec_sm)
  dev_mo = to_device_arrays(spec_mo)
  fn_sm = jax.jit(lambda d, p, m, cp, cm: render_wide_bvh_batch(spec_sm, d, p, m, cp, cm, interpret=False))
  fn_mo = jax.jit(lambda d, p, m, cp, cm: render_wide_bvh_batch(spec_mo, d, p, m, cp, cm, interpret=False))
  gb_sm = fn_sm(dev_sm, oxp, oxm, cxp, cxm)
  gb_mo = fn_mo(dev_mo, oxp, oxm, cxp, cxm)
  jax.block_until_ready(gb_mo.rgb)

  f0_1m_z = np.asarray(gb_sm.z_depth[0], np.float32)
  fg_1m = f0_1m_z > 0.0
  both_1m = fg_cy & fg_1m
  p_1m = {
      "total_tris": 1182720,
      "B_eval": 8,
      "spatial_median_vs_morton_max_depth_diff_m": float(jnp.max(jnp.abs(gb_sm.depth - gb_mo.depth))),
      "spatial_median_vs_morton_max_rgb_diff": float(jnp.max(jnp.abs(gb_sm.rgb - gb_mo.rgb))),
      "spatial_median_vs_morton_geom_match": float(jnp.mean(gb_sm.geom_id == gb_mo.geom_id)),
      "frame0_z_depth_mae_vs_cycles_m": float(np.mean(np.abs(f0_cy_z[both_1m] - f0_1m_z[both_1m]))),
      "frame0_fg_mask_match_vs_cycles": float(np.mean(fg_cy == fg_1m)),
  }
  assert p_1m["spatial_median_vs_morton_max_depth_diff_m"] < 1e-4
  assert p_1m["spatial_median_vs_morton_geom_match"] >= 0.9999
  results["parity"]["blender_1182720"] = p_1m
  print("PARITY 1.18M PASS:", p_1m, flush=True)
  del dev_sm, dev_mo, spec_sm, spec_mo, meshes_1m

  # 3. 1-Chip Multi-Scale Sweep
  for k in [1, 5, 11, 16]:
    t_s0 = time.perf_counter()
    meshes = make_scaled_meshes(base_meshes, k)
    t_sub = time.perf_counter() - t_s0
    for res in [64, 128]:
      t_b0 = time.perf_counter()
      spec = build_wide_bvh_scene_spec(meshes, height=res, width=res, **kw)
      bvh_build_s = time.perf_counter() - t_b0
      st = spec_stats(spec)
      dev_arr = to_device_arrays(spec)
      batches = [64, 256, 1024] if res == 64 else [64, 256]
      for B in batches:
        oxp_b, oxm_b, cxp_b, cxm_b = tile_poses(ds, B)
        fn = jax.jit(lambda da, a, b, c, d: render_wide_bvh_batch(spec, da, a, b, c, d, interpret=False).rgb)
        t_c0 = time.perf_counter()
        y = fn(dev_arr, oxp_b, oxm_b, cxp_b, cxm_b)
        jax.block_until_ready(y)
        compile_s = time.perf_counter() - t_c0
        # Extra warmup step
        jax.block_until_ready(fn(dev_arr, oxp_b, oxm_b, cxp_b, cxm_b))

        n_steps = 10 if (st["total_tris"] < 200000 or B <= 256) else 5
        t0 = time.perf_counter()
        for _ in range(n_steps):
          y = fn(dev_arr, oxp_b, oxm_b, cxp_b, cxm_b)
        jax.block_until_ready(y)
        ms = (time.perf_counter() - t0) * 1e3 / n_steps
        fps = B * 1e3 / ms
        rec = {
            **st,
            "n_chips": 1,
            "res": f"{res}x{res}",
            "B": B,
            "subdivide_s": round(t_sub, 3),
            "bvh_build_s": round(bvh_build_s, 3),
            "compile_s": round(compile_s, 2),
            "ms": round(ms, 2),
            "fps": round(fps, 1),
            "mrays_per_s": round((B * res * res * 2) / (ms * 1e-3) / 1e6, 2),
        }
        results["runs_1c"].append(rec)
        print("1C BENCH:", rec, flush=True)
      del dev_arr, spec
    del meshes

  with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
  print(f"Saved 1c results to {out_path}", flush=True)


def run_4chip_suite(out_path: str):
  ds, _, base_meshes = load_base_scene()
  devs = jax.devices()
  assert len(devs) == 4, f"Expected 4 devices for 4c suite, got {devs}"
  with open(out_path, "r") as f:
    results = json.load(f)

  kw = dict(
      cam_fovy=float(ds["cam_fovy"]),
      znear=float(ds["znear"]),
      zfar=float(ds["zfar"]),
      light_pos=tuple(float(x) for x in ds["light_pos"]),
  )

  hlo_4c_verified = False
  for k in [1, 5, 11, 16]:
    meshes = make_scaled_meshes(base_meshes, k)
    for res in [64, 128]:
      spec = build_wide_bvh_scene_spec(meshes, height=res, width=res, **kw)
      st = spec_stats(spec)
      fn_4c, _mesh = make_sharded_wide_bvh_renderer(spec, devs, interpret=False)
      batches = [256, 1024]
      for B in batches:
        oxp_b, oxm_b, cxp_b, cxm_b = tile_poses(ds, B)
        if not hlo_4c_verified:
          txt = fn_4c.lower(oxp_b, oxm_b, cxp_b, cxm_b).as_text()
          info = inspect_hlo_ops(txt)
          info["send_recv_ops"] = len(re.findall(r"\bsend\(|\brecv\(", txt, re.I))
          assert info["host_callbacks"] == 0 and info["send_recv_ops"] == 0, info
          results["parity"]["hlo_4c_shard_map"] = info
          hlo_4c_verified = True

        t_c0 = time.perf_counter()
        y = fn_4c(oxp_b, oxm_b, cxp_b, cxm_b).rgb
        jax.block_until_ready(y)
        compile_s = time.perf_counter() - t_c0
        jax.block_until_ready(fn_4c(oxp_b, oxm_b, cxp_b, cxm_b).rgb)

        n_steps = 10
        t0 = time.perf_counter()
        for _ in range(n_steps):
          y = fn_4c(oxp_b, oxm_b, cxp_b, cxm_b).rgb
        jax.block_until_ready(y)
        ms = (time.perf_counter() - t0) * 1e3 / n_steps
        fps = B * 1e3 / ms
        rec = {
            **st,
            "n_chips": 4,
            "res": f"{res}x{res}",
            "B": B,
            "compile_s": round(compile_s, 2),
            "ms": round(ms, 2),
            "fps": round(fps, 1),
            "mrays_per_s": round((B * res * res * 2) / (ms * 1e-3) / 1e6, 2),
        }
        results["runs_4c"].append(rec)
        print("4C BENCH:", rec, flush=True)
      del spec, fn_4c
    del meshes

  with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
  print(f"Saved full 1c+4c results to {out_path}", flush=True)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--phase", choices=["1c", "4c"], required=True)
  ap.add_argument("--out", required=True)
  args = ap.parse_args()
  if args.phase == "1c":
    run_1chip_suite(args.out)
  else:
    run_4chip_suite(args.out)


if __name__ == "__main__":
  main()
