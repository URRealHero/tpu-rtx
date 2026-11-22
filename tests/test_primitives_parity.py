"""Phase 1 Parity Test: Compare Stage 1 Analytic Primitives & Scenes vs official C mujoco.mj_multiRay."""

from __future__ import annotations

import json
import os
import sys

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np

from gate import evaluate_stage1
from tpu_mjx_render.camera import compute_camera_rays
from tpu_mjx_render.renderer import RenderSceneSpec, render_batch
from tpu_mjx_render.scenes import get_scene_xml


def mujoco_c_multiray_reference(m: mujoco.MjModel, d: mujoco.MjData, cam_id: int, H: int, W: int, spec: RenderSceneSpec):
  """Cast all H*W rays in C via mujoco.mj_multiRay()."""
  cam_pos = np.asarray(d.cam_xpos[cam_id], dtype=np.float64)
  cam_mat = np.asarray(d.cam_xmat[cam_id], dtype=np.float64).reshape(3, 3)

  _, rd, _ = compute_camera_rays(
      jnp.asarray(cam_pos, jnp.float32),
      jnp.asarray(cam_mat, jnp.float32),
      spec.cam_fovy,
      H,
      W,
      projection=spec.cam_projection,
      sensorsize=spec.cam_sensorsize,
      intrinsic=spec.cam_intrinsic,
      znear=spec.znear,
  )
  rd_np = np.ascontiguousarray(np.asarray(rd, dtype=np.float64).reshape(-1))
  nray = H * W

  geomgroup = np.array([1, 1, 1, 1, 0, 0], dtype=np.uint8)
  geomid_out = np.full((nray,), -1, dtype=np.int32)
  dist_out = np.full((nray,), -1.0, dtype=np.float64)
  normal_out = np.zeros((nray * 3,), dtype=np.float64)

  cutoff = spec.zfar if spec.zfar > 0.0 else 1.0e8
  mujoco.mj_multiRay(
      m,
      d,
      cam_pos,
      rd_np,
      geomgroup,
      True,  # flg_static
      -1,    # bodyexclude
      geomid_out,
      dist_out,
      normal_out,
      nray,
      cutoff,
  )

  # Mask out invisible geoms (alpha == 0)
  nrm_3 = normal_out.reshape(nray, 3).astype(np.float32)
  for gid in range(m.ngeom):
    if spec.geom_rgba[gid, 3] <= 0.0:
      mask = geomid_out == gid
      geomid_out[mask] = -1
      dist_out[mask] = -1.0
      nrm_3[mask] = 0.0

  miss = (geomid_out < 0) | (dist_out < 0.0)
  geomid_out[miss] = -1
  dist_out[miss] = -1.0
  nrm_3[miss] = 0.0
  return dist_out.astype(np.float32).reshape(H, W), nrm_3.reshape(H, W, 3), geomid_out.reshape(H, W)


def check_scene_parity(scene_name: str, cam_id: int = 0, H: int = 64, W: int = 64, n_poses: int = 16):
  xml = get_scene_xml(scene_name)
  m = mujoco.MjModel.from_xml_string(xml)
  mx = mjx.put_model(m, impl="jax")
  spec = RenderSceneSpec.from_mjmodel(m, cam_id=cam_id, height=H, width=W)

  rng = np.random.default_rng(42)
  depth_refs, nrm_refs, gid_refs = [], [], []
  qpos_list = []

  d = mujoco.MjData(m)
  dx_list = []
  for pose_i in range(n_poses):
    if m.nkey > 0:
      mujoco.mj_resetDataKeyframe(m, d, pose_i % m.nkey)
    else:
      mujoco.mj_resetData(m, d)
    if m.nq > 0:
      if scene_name == "cartpole":
        d.qpos[0] = rng.uniform(-1.2, 1.2)
        d.qpos[1] = rng.uniform(-0.8, 0.8)
      elif scene_name == "panda_pick_cube":
        d.qpos[:7] += rng.uniform(-0.25, 0.25, size=7)
        d.qpos[7:9] = rng.uniform(0.0, 0.04)
        d.qpos[9] += rng.uniform(-0.12, 0.12)
        d.qpos[10] += rng.uniform(-0.15, 0.15)
    mujoco.mj_forward(m, d)
    d_ref, n_ref, g_ref = mujoco_c_multiray_reference(m, d, cam_id, H, W, spec)
    depth_refs.append(d_ref)
    nrm_refs.append(n_ref)
    gid_refs.append(g_ref)
    if m.nq > 0:
      qpos_list.append(d.qpos.copy())
    else:
      dx_list.append((
          d.geom_xpos.copy(),
          d.geom_xmat.copy().reshape(m.ngeom, 3, 3),
          d.cam_xpos.copy(),
          d.cam_xmat.copy().reshape(m.ncam, 3, 3),
          d.xpos.copy(),
          d.xmat.copy().reshape(m.nbody, 3, 3),
      ))

  dx0 = mjx.make_data(m, impl="jax")
  dx_batch = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x[None, ...], (n_poses, *x.shape)).copy(), dx0)
  if m.nq > 0:
    qpos_arr = jnp.asarray(np.stack(qpos_list), jnp.float32)
    dx_batch = dx_batch.replace(qpos=qpos_arr)
    dx_batch = jax.jit(jax.vmap(lambda x: mjx.forward(mx, x)))(dx_batch)
  else:
    dx_batch = dx_batch.replace(
        geom_xpos=jnp.asarray(np.stack([x[0] for x in dx_list]), jnp.float32),
        geom_xmat=jnp.asarray(np.stack([x[1] for x in dx_list]), jnp.float32),
        cam_xpos=jnp.asarray(np.stack([x[2] for x in dx_list]), jnp.float32),
        cam_xmat=jnp.asarray(np.stack([x[3] for x in dx_list]), jnp.float32),
        xpos=jnp.asarray(np.stack([x[4] for x in dx_list]), jnp.float32),
        xmat=jnp.asarray(np.stack([x[5] for x in dx_list]), jnp.float32),
    )

  gb_batch = jax.jit(lambda d_b: render_batch(spec, d_b, mode="tiled"))(dx_batch)

  depth_p = np.asarray(gb_batch.depth)
  depth_r = np.stack(depth_refs)
  nrm_p = np.asarray(gb_batch.normal)
  nrm_r = np.stack(nrm_refs)
  gid_p = np.asarray(gb_batch.geom_id)
  gid_r = np.stack(gid_refs)

  valid = (gid_p >= 0) & (gid_r >= 0)
  depth_mae = float(np.mean(np.abs(depth_p[valid] - depth_r[valid])))
  ndot = np.clip(np.sum(nrm_p[valid] * nrm_r[valid], axis=-1), -1.0, 1.0)
  nrm_deg = float(np.mean(np.degrees(np.arccos(ndot))))
  geom_match = float(np.mean(gid_p == gid_r))

  res = {
      "scene": scene_name,
      "n_poses": n_poses,
      "depth_mae_m": depth_mae,
      "normal_deg": nrm_deg,
      "geom_match": geom_match,
  }
  if scene_name in ("cartpole", "panda_pick_cube"):
    gate_res = evaluate_stage1(scene_name, depth_p, depth_r, nrm_p, nrm_r, gid_p, gid_r)
    res["gate_passed"] = gate_res["passed"]
  else:
    res["gate_passed"] = (depth_mae <= 1e-4) and (nrm_deg <= 0.5) and (geom_match >= 0.998)
  return res


def main():
  results = []
  for sc, n_p in [("all_primitives", 4), ("cartpole", 64), ("panda_pick_cube", 64)]:
    r = check_scene_parity(sc, cam_id=0, H=64, W=64, n_poses=n_p)
    print(json.dumps(r), flush=True)
    results.append(r)
    if not r["gate_passed"]:
      print(f"FAILED parity on {sc}", file=sys.stderr)
      sys.exit(1)

  os.makedirs("out/metrics", exist_ok=True)
  with open("out/metrics/stage1_parity.json", "w") as f:
    json.dump(results, f, indent=2)
  print("ALL STAGE 1 PARITY TESTS PASSED")


if __name__ == "__main__":
  main()
