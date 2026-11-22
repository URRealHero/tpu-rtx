"""Comprehensive local verification of V2 Pallas SoA, Cluster-BVH Meshes & Rigid 3DGS.

Runs standalone on CPU (JAX 0.10.2, interpret=True for Pallas) without needing
`mujoco` installed locally, and also runs identically on live TPU (interpret=False).
"""

from __future__ import annotations

import json
import math
import os
import re
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from tpu_mjx_render.gs3d import mesh_scene_to_rigid_3dgs, render_gs3d_batch
from tpu_mjx_render.mesh_ray import build_mesh_scene_spec, render_mesh_batch
from tpu_mjx_render.primitives import (
    GEOM_BOX,
    GEOM_CAPSULE,
    GEOM_CYLINDER,
    GEOM_ELLIPSOID,
    GEOM_PLANE,
    GEOM_SPHERE,
)
from tpu_mjx_render.renderer import RenderSceneSpec, render_batch
from tpu_mjx_render.roofline import inspect_hlo_ops


def _rot_z(a: float) -> np.ndarray:
  c, s = math.cos(a), math.sin(a)
  return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def _look_at_mat(cam_pos: np.ndarray, target: np.ndarray) -> np.ndarray:
  fwd = target - cam_pos
  fwd = fwd / max(np.linalg.norm(fwd), 1e-12)
  right = np.cross(fwd, np.array([0.0, 0.0, 1.0], np.float32))
  right = right / max(np.linalg.norm(right), 1e-12)
  up = np.cross(right, fwd)
  return np.stack([right, up, -fwd], axis=1).astype(np.float32)


class _MockMjxData(NamedTuple):
  cam_xpos: jax.Array
  cam_xmat: jax.Array
  geom_xpos: jax.Array
  geom_xmat: jax.Array
  xpos: jax.Array
  xmat: jax.Array


def _make_synthetic_all_prims_batch(B: int = 16, H: int = 64, W: int = 64):
  """Construct a 6-primitive synthetic RenderSceneSpec + batched MjData-like namespace."""
  geom_type = np.array([GEOM_PLANE, GEOM_SPHERE, GEOM_CAPSULE, GEOM_ELLIPSOID, GEOM_CYLINDER, GEOM_BOX], np.int32)
  geom_size = np.array([
      [4.0, 4.0, 0.1],
      [0.24, 0.0, 0.0],
      [0.10, 0.25, 0.0],
      [0.22, 0.14, 0.18],
      [0.14, 0.22, 0.0],
      [0.20, 0.16, 0.18],
  ], np.float32)
  geom_rgba = np.array([
      [0.3, 0.4, 0.5, 1.0],
      [0.9, 0.2, 0.2, 1.0],
      [0.2, 0.8, 0.3, 1.0],
      [0.2, 0.4, 0.9, 1.0],
      [0.9, 0.7, 0.2, 1.0],
      [0.8, 0.3, 0.8, 1.0],
  ], np.float32)
  spec = RenderSceneSpec(
      ngeom=6,
      nlight=1,
      cam_id=0,
      height=H,
      width=W,
      cam_projection=0,
      cam_fovy=45.0,
      cam_sensorsize=(0.0, 0.0),
      cam_intrinsic=(0.0, 0.0, 0.0, 0.0),
      znear=0.05,
      zfar=50.0,
      use_shadows=True,
      shadow_light_fraction=0.0,
      enable_specular=True,
      enabled_geom_ids=(0, 1, 2, 3, 4, 5),
      type_groups=(
          (GEOM_PLANE, (0,)),
          (GEOM_SPHERE, (1,)),
          (GEOM_CAPSULE, (2,)),
          (GEOM_ELLIPSOID, (3,)),
          (GEOM_CYLINDER, (4,)),
          (GEOM_BOX, (5,)),
      ),
      geom_type=geom_type,
      geom_size=geom_size,
      geom_rgba=geom_rgba,
      geom_spec=np.full((6,), 0.3, np.float32),
      geom_shin=np.full((6,), 0.3, np.float32),
      geom_emit=np.zeros((6,), np.float32),
      geom_is_checker=np.array([True, False, False, False, False, False], bool),
      checker_rgb1=(0.24, 0.32, 0.40),
      checker_rgb2=(0.14, 0.20, 0.28),
      checker_markrgb=(0.75, 0.78, 0.82),
      checker_repeat=(2.0, 2.0),
      sky_rgb1=(0.28, 0.46, 0.68),
      sky_rgb2=(0.05, 0.08, 0.12),
      light_type=np.array([1], np.int32),  # directional
      light_castshadow=np.array([True], bool),
      light_active=np.array([True], bool),
      light_bodyid=np.array([0], np.int32),
      light_pos=np.array([[0.8, -0.5, 2.5]], np.float32),
      light_dir=np.array([[-0.35, 0.25, -0.90]], np.float32),
      light_attenuation=np.array([[1.0, 0.0, 0.0]], np.float32),
      light_cutoff_rad=np.array([math.radians(45.0)], np.float32),
      light_exponent=np.array([10.0], np.float32),
      light_ambient=np.array([[0.15, 0.15, 0.15]], np.float32),
      light_diffuse=np.array([[0.80, 0.80, 0.80]], np.float32),
      light_specular=np.array([[0.25, 0.25, 0.25]], np.float32),
      headlight_active=True,
      headlight_ambient=(0.2, 0.2, 0.2),
      headlight_diffuse=(0.4, 0.4, 0.4),
      headlight_specular=(0.1, 0.1, 0.1),
  )

  g_xpos = np.zeros((B, 6, 3), np.float32)
  g_xmat = np.zeros((B, 6, 3, 3), np.float32)
  c_xpos = np.zeros((B, 1, 3), np.float32)
  c_xmat = np.zeros((B, 1, 3, 3), np.float32)
  base_pos = np.array([
      [0.0, 0.0, 0.0],
      [-0.6, 0.1, 0.30],
      [-0.2, -0.2, 0.35],
      [0.2, 0.15, 0.25],
      [0.55, -0.1, 0.30],
      [0.0, 0.55, 0.25],
  ], np.float32)
  for b in range(B):
    ang = 0.08 * b
    cp = np.array([1.8 * math.cos(ang), -1.8 * math.sin(ang), 1.1], np.float32)
    c_xpos[b, 0] = cp
    c_xmat[b, 0] = _look_at_mat(cp, np.array([0.0, 0.1, 0.25], np.float32))
    for g in range(6):
      g_xpos[b, g] = base_pos[g] + (np.array([0.02 * b, -0.01 * b, 0.0], np.float32) if g > 0 else 0.0)
      g_xmat[b, g] = _rot_z(0.15 * b * (g % 2)) if g > 0 else np.eye(3, dtype=np.float32)

  data = _MockMjxData(
      cam_xpos=jnp.asarray(c_xpos),
      cam_xmat=jnp.asarray(c_xmat),
      geom_xpos=jnp.asarray(g_xpos),
      geom_xmat=jnp.asarray(g_xmat),
      xpos=jnp.zeros((B, 1, 3), dtype=jnp.float32),
      xmat=jnp.broadcast_to(jnp.eye(3, dtype=jnp.float32)[None, None, :, :], (B, 1, 3, 3)),
  )
  return spec, data


def _load_blender_meshes_and_data():
  ds = np.load("out/assets/blender_real_scene.npz")
  gt = np.load("out/assets/blender_real_scene_gt.npz")
  M = int(ds["n_meshes"])
  meshes = []
  for m in range(M):
    meshes.append({
        "name": str(ds[f"mesh_{m}_name"]),
        "verts": ds[f"mesh_{m}_verts"],
        "faces": ds[f"mesh_{m}_faces"],
        "rgba": ds[f"mesh_{m}_rgba"],
    })
  return ds, gt, meshes


def main():
  interpret = (jax.default_backend() == "cpu")
  results = {"backend": jax.default_backend(), "interpret": interpret}

  # --- TEST 1: V2 SoA & Native (8, 128) Pallas V2 Parity vs V1 Tiled (all 6 geom types) ---
  spec, d_batch = _make_synthetic_all_prims_batch(B=16, H=64, W=64)
  fn_tiled = jax.jit(lambda d: render_batch(spec, d, mode="tiled"))
  fn_v2 = jax.jit(lambda d: render_batch(spec, d, mode="v2_soa"))
  fn_pv2 = jax.jit(lambda d: render_batch(spec, d, mode="pallas_v2", pallas_interpret=interpret))

  gb_t = fn_tiled(d_batch)
  gb_v2 = fn_v2(d_batch)
  gb_pv2 = fn_pv2(d_batch)

  v2_depth_mae = float(jnp.mean(jnp.abs(gb_t.depth - gb_v2.depth)))
  v2_depth_diff = float(jnp.max(jnp.abs(gb_t.depth - gb_v2.depth)))
  v2_rgb_diff = float(jnp.max(jnp.abs(gb_t.rgb - gb_v2.rgb)))
  v2_geom_match = float(jnp.mean(gb_t.geom_id == gb_v2.geom_id))

  pv2_depth_mae = float(jnp.mean(jnp.abs(gb_v2.depth - gb_pv2.depth)))
  pv2_depth_diff = float(jnp.max(jnp.abs(gb_v2.depth - gb_pv2.depth)))
  pv2_rgb_diff = float(jnp.max(jnp.abs(gb_v2.rgb - gb_pv2.rgb)))
  pv2_geom_match = float(jnp.mean(gb_v2.geom_id == gb_pv2.geom_id))

  assert v2_depth_mae < 1e-5 and v2_depth_diff < 5e-4 and v2_geom_match >= 0.9999 and v2_rgb_diff < 1e-3, (
      v2_depth_mae, v2_depth_diff, v2_geom_match, v2_rgb_diff
  )
  assert pv2_depth_mae < 1e-5 and pv2_depth_diff < 5e-4 and pv2_geom_match == 1.0 and pv2_rgb_diff < 1e-3, (
      pv2_depth_mae, pv2_depth_diff, pv2_geom_match, pv2_rgb_diff
  )

  results["test1_v2_pallas_parity"] = {
      "B": 16,
      "res": "64x64",
      "v2_soa_vs_tiled_depth_mae_m": v2_depth_mae,
      "v2_soa_vs_tiled_max_depth_diff_m": v2_depth_diff,
      "v2_soa_vs_tiled_max_rgb_diff": v2_rgb_diff,
      "v2_soa_vs_tiled_geom_match": v2_geom_match,
      "pallas_v2_vs_v2_soa_depth_mae_m": pv2_depth_mae,
      "pallas_v2_vs_v2_soa_max_depth_diff_m": pv2_depth_diff,
      "pallas_v2_vs_v2_soa_max_rgb_diff": pv2_rgb_diff,
      "pallas_v2_vs_v2_soa_geom_match": pv2_geom_match,
      "passed": True,
  }
  print("TEST 1 PASS:", results["test1_v2_pallas_parity"])

  # --- TEST 2: Real Blender 5.2 Multi-Mesh Triangle Raytracer Parity vs Cycles Z-depth ---
  ds, gt, meshes = _load_blender_meshes_and_data()
  mspec = build_mesh_scene_spec(
      meshes,
      height=int(ds["height"]),
      width=int(ds["width"]),
      cam_fovy=float(ds["cam_fovy"]),
      znear=float(ds["znear"]),
      zfar=float(ds["zfar"]),
      light_pos=tuple(float(x) for x in ds["light_pos"]),
  )
  B_sub = 8  # Frame 0 + 7 held-out dynamic poses
  idx_eval = np.array([0] + list(range(len(ds["split"]) - 7, len(ds["split"]))), dtype=np.int32)
  oxp = jnp.asarray(ds["obj_xpos"][idx_eval])
  oxm = jnp.asarray(ds["obj_xmat"][idx_eval])
  cxp = jnp.asarray(ds["cam_xpos"][idx_eval])
  cxm = jnp.asarray(ds["cam_xmat"][idx_eval])

  fn_mesh_jax = jax.jit(lambda p, m, cp, cm: render_mesh_batch(mspec, p, m, cp, cm, mode="jax_cluster"))
  fn_mesh_pal = jax.jit(lambda p, m, cp, cm: render_mesh_batch(mspec, p, m, cp, cm, mode="pallas", interpret=interpret))
  gb_mj = fn_mesh_jax(oxp, oxm, cxp, cxm)
  gb_mp = fn_mesh_pal(oxp, oxm, cxp, cxm)

  # Verify Frame-0 Z-depth alignment against Blender 5.2 Cycles EXR Z-depth
  f0_cy_z = np.asarray(gt["frame0_z_depth"], np.float32)
  f0_mj_z = np.asarray(gb_mj.z_depth[0], np.float32)
  fg_cy = f0_cy_z < 20.0
  fg_mj = f0_mj_z > 0.0
  fg_iou = float(np.mean(fg_cy == fg_mj))
  both_fg = fg_cy & fg_mj
  z_mae = float(np.mean(np.abs(f0_cy_z[both_fg] - f0_mj_z[both_fg])))
  mp_depth_diff = float(jnp.max(jnp.abs(gb_mj.depth - gb_mp.depth)))
  mp_geom_match = float(jnp.mean(gb_mj.geom_id == gb_mp.geom_id))

  assert z_mae < 0.01 and fg_iou >= 0.99, (z_mae, fg_iou)
  assert mp_depth_diff < 1e-5 and mp_geom_match == 1.0, (mp_depth_diff, mp_geom_match)

  results["test2_blender_mesh_ray_parity"] = {
      "n_meshes": len(meshes),
      "total_tris": int(sum(it.n_tris_orig for it in mspec.meshes)),
      "total_clusters_k16": int(sum(it.n_clusters for it in mspec.meshes)),
      "frame0_vs_cycles_z_depth_mae_m": z_mae,
      "frame0_vs_cycles_fg_mask_match": fg_iou,
      "pallas_vs_jax_cluster_max_depth_diff_m": mp_depth_diff,
      "pallas_vs_jax_cluster_geom_match": mp_geom_match,
      "passed": True,
  }
  print("TEST 2 PASS:", results["test2_blender_mesh_ray_parity"])

  # --- TEST 3: Rigid-Body-Bound Tile-Based 3DGS Parity & Pallas Kernel ---
  gs_spec = mesh_scene_to_rigid_3dgs(
      meshes,
      total_gaussians=2048,
      height=int(ds["height"]),
      width=int(ds["width"]),
      cam_fovy=float(ds["cam_fovy"]),
      znear=float(ds["znear"]),
      light_pos=tuple(float(x) for x in ds["light_pos"]),
  )
  fn_gs_jax = jax.jit(lambda p, m, cp, cm: render_gs3d_batch(gs_spec, p, m, cp, cm, tile_size=8, top_k=128, mode="jax_tile"))
  fn_gs_pal = jax.jit(lambda p, m, cp, cm: render_gs3d_batch(gs_spec, p, m, cp, cm, tile_size=8, top_k=128, mode="pallas", interpret=interpret))
  rgb_gj, dep_gj, _, alp_gj = fn_gs_jax(oxp, oxm, cxp, cxm)
  rgb_gp, dep_gp, _, alp_gp = fn_gs_pal(oxp, oxm, cxp, cxm)

  gs_rgb_diff = float(jnp.max(jnp.abs(rgb_gj - rgb_gp)))
  gs_dep_diff = float(jnp.max(jnp.abs(dep_gj - dep_gp)))
  fg_gs = np.asarray(alp_gj[0]) > 0.3
  gs_fg_match = float(np.mean(fg_gs == fg_cy))
  assert gs_rgb_diff < 1e-4 and gs_dep_diff < 1e-4 and gs_fg_match >= 0.90

  results["test3_rigid_3dgs_parity"] = {
      "n_gaussians": gs_spec.n_gaussians,
      "pallas_vs_jax_tile_max_rgb_diff": gs_rgb_diff,
      "pallas_vs_jax_tile_max_depth_diff_m": gs_dep_diff,
      "frame0_vs_cycles_fg_match": gs_fg_match,
      "passed": True,
  }
  print("TEST 3 PASS:", results["test3_rigid_3dgs_parity"])

  # --- TEST 4: Zero-Copy XLA HLO Audit for All 4 V2 Pipelines ---
  def _check_hlo(lowered):
    txt = lowered.as_text()
    info = inspect_hlo_ops(txt)
    info["send_recv_ops"] = len(re.findall(r"\bsend\(|\brecv\(", txt, re.I))
    assert info["host_callbacks"] == 0 and info["send_recv_ops"] == 0, info
    info["zero_copy_verified"] = True
    return info

  results["test4_v2_zero_copy_hlo"] = {
      "v2_soa": _check_hlo(fn_v2.lower(d_batch)),
      "pallas_v2": _check_hlo(fn_pv2.lower(d_batch)),
      "mesh_ray_jax_cluster": _check_hlo(fn_mesh_jax.lower(oxp, oxm, cxp, cxm)),
      "mesh_ray_pallas": _check_hlo(fn_mesh_pal.lower(oxp, oxm, cxp, cxm)),
      "gs3d_jax_tile": _check_hlo(fn_gs_jax.lower(oxp, oxm, cxp, cxm)),
      "gs3d_pallas": _check_hlo(fn_gs_pal.lower(oxp, oxm, cxp, cxm)),
  }
  print("TEST 4 PASS: zero host_callbacks & send_recv across all 6 lowered HLO graphs")

  os.makedirs("out/metrics", exist_ok=True)
  out_json = "out/metrics/v2_local_verification.json"
  with open(out_json, "w") as f:
    json.dump(results, f, indent=2)
  print(f"Saved {out_json}")


if __name__ == "__main__":
  main()
