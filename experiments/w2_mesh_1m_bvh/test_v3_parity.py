"""Test V3 Wide-BVH parity vs V2 jax_cluster, V2 pallas, and Blender Cycles EXR + HLO audit."""

import json
import re
import time
import jax
import jax.numpy as jnp
import numpy as np

from tpu_mjx_render.mesh_ray import build_mesh_scene_spec, render_mesh_batch
from tpu_mjx_render.roofline import inspect_hlo_ops
from wide_bvh_v3 import build_wide_bvh_scene_spec, render_wide_bvh_batch, to_device_arrays


def main():
  print("Backend:", jax.default_backend(), "Devices:", jax.devices())
  ds = np.load("/dev/shm/zaychik-tpu-mjx/out/assets/blender_real_scene.npz")
  gt = np.load("/dev/shm/zaychik-tpu-mjx/out/assets/blender_real_scene_gt.npz")
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

  kw = dict(
      height=int(ds["height"]),
      width=int(ds["width"]),
      cam_fovy=float(ds["cam_fovy"]),
      znear=float(ds["znear"]),
      zfar=float(ds["zfar"]),
      light_pos=tuple(float(x) for x in ds["light_pos"]),
  )
  mspec_v2 = build_mesh_scene_spec(meshes, **kw)
  mspec_v3 = build_wide_bvh_scene_spec(meshes, **kw)
  dev_v3 = to_device_arrays(mspec_v3)

  idx_eval = np.array([0] + list(range(len(ds["split"]) - 7, len(ds["split"]))), dtype=np.int32)
  oxp = jnp.asarray(ds["obj_xpos"][idx_eval])
  oxm = jnp.asarray(ds["obj_xmat"][idx_eval])
  cxp = jnp.asarray(ds["cam_xpos"][idx_eval])
  cxm = jnp.asarray(ds["cam_xmat"][idx_eval])

  fn_v2 = jax.jit(lambda p, m, cp, cm: render_mesh_batch(mspec_v2, p, m, cp, cm, mode="jax_cluster"))
  fn_v3 = jax.jit(lambda d, p, m, cp, cm: render_wide_bvh_batch(mspec_v3, d, p, m, cp, cm, interpret=False))

  print("Compiling & running V2 jax_cluster...")
  gb_v2 = fn_v2(oxp, oxm, cxp, cxm)
  jax.block_until_ready(gb_v2.rgb)

  print("Compiling & running V3 Wide-BVH Pallas (interpret=False)...")
  t0 = time.perf_counter()
  gb_v3 = fn_v3(dev_v3, oxp, oxm, cxp, cxm)
  jax.block_until_ready(gb_v3.rgb)
  print(f"V3 compile+run: {time.perf_counter() - t0:.2f}s")

  max_depth_diff = float(jnp.max(jnp.abs(gb_v2.depth - gb_v3.depth)))
  mean_depth_diff = float(jnp.mean(jnp.abs(gb_v2.depth - gb_v3.depth)))
  max_rgb_diff = float(jnp.max(jnp.abs(gb_v2.rgb - gb_v3.rgb)))
  geom_match = float(jnp.mean(gb_v2.geom_id == gb_v3.geom_id))

  f0_cy_z = np.asarray(gt["frame0_z_depth"], np.float32)
  f0_v3_z = np.asarray(gb_v3.z_depth[0], np.float32)
  fg_cy = f0_cy_z < 20.0
  fg_v3 = f0_v3_z > 0.0
  fg_iou = float(np.mean(fg_cy == fg_v3))
  both_fg = fg_cy & fg_v3
  z_mae = float(np.mean(np.abs(f0_cy_z[both_fg] - f0_v3_z[both_fg])))

  hlo_txt = fn_v3.lower(dev_v3, oxp, oxm, cxp, cxm).as_text()
  hlo_info = inspect_hlo_ops(hlo_txt)
  hlo_info["send_recv_ops"] = len(re.findall(r"\bsend\(|\brecv\(", hlo_txt, re.I))

  res = {
      "max_depth_diff_vs_v2": max_depth_diff,
      "mean_depth_diff_vs_v2": mean_depth_diff,
      "max_rgb_diff_vs_v2": max_rgb_diff,
      "geom_match_vs_v2": geom_match,
      "z_depth_mae_vs_cycles": z_mae,
      "fg_mask_match_vs_cycles": fg_iou,
      "hlo": hlo_info,
  }
  print("PARITY RESULT:", json.dumps(res, indent=2))
  assert max_depth_diff < 1e-4, max_depth_diff
  assert max_rgb_diff < 1e-3, max_rgb_diff
  assert geom_match >= 0.999, geom_match
  assert z_mae < 1e-4, z_mae
  assert hlo_info["host_callbacks"] == 0 and hlo_info["send_recv_ops"] == 0, hlo_info
  print("ALL PARITY & HLO ASSERTIONS PASSED!")


if __name__ == "__main__":
  main()
