"""Benchmark V3 Wide-BVH throughput across 4.6k -> 100k+ -> 500k+ -> 1M+ tris on TPU v5p."""

import json
import time
import jax
import jax.numpy as jnp
import numpy as np

from wide_bvh_v3 import (
    _compute_smooth_vertex_normals,
    build_wide_bvh_scene_spec,
    render_wide_bvh_batch,
    to_device_arrays,
)


def subdivide_mesh_barycentric(verts: np.ndarray, faces: np.ndarray, k: int):
  """Subdivide each triangle into k^2 sub-triangles with interpolated smooth vertex normals."""
  if k <= 1:
    return verts, faces, _compute_smooth_vertex_normals(verts, faces)

  vnorm = _compute_smooth_vertex_normals(verts, faces)
  # Build barycentric grid for one canonical triangle: (i, j) with i+j <= k
  pts_uv = []
  idx_map = {}
  for i in range(k + 1):
    for j in range(k + 1 - i):
      idx_map[(i, j)] = len(pts_uv)
      pts_uv.append((i / k, j / k))
  pts_uv = np.array(pts_uv, dtype=np.float32)  # (P, 2)
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
  sub_faces = np.array(sub_faces, dtype=np.int32)  # (k^2, 3)

  F = faces.shape[0]
  v0 = verts[faces[:, 0]]  # (F, 3)
  v1 = verts[faces[:, 1]]
  v2 = verts[faces[:, 2]]
  n0 = vnorm[faces[:, 0]]
  n1 = vnorm[faces[:, 1]]
  n2 = vnorm[faces[:, 2]]

  u = pts_uv[None, :, 0:1]  # (1, P, 1)
  v = pts_uv[None, :, 1:2]
  w = 1.0 - u - v

  new_verts = (w * v0[:, None, :] + u * v1[:, None, :] + v * v2[:, None, :]).reshape(F * P, 3)
  new_norms = (w * n0[:, None, :] + u * n1[:, None, :] + v * n2[:, None, :]).reshape(F * P, 3)
  new_norms = (new_norms / np.maximum(np.linalg.norm(new_norms, axis=-1, keepdims=True), 1e-12)).astype(np.float32)

  offsets = (np.arange(F, dtype=np.int32) * P)[:, None, None]
  new_faces = (sub_faces[None, :, :] + offsets).reshape(F * k * k, 3)
  return new_verts.astype(np.float32), new_faces, new_norms


def load_scaled_meshes(base_meshes, k: int):
  out = []
  for m in base_meshes:
    nv, nf, nn = subdivide_mesh_barycentric(m["verts"], m["faces"], k)
    out.append({"name": m["name"], "verts": nv, "faces": nf, "vert_normals": nn, "rgba": m["rgba"]})
  return out


def main():
  ds = np.load("/dev/shm/zaychik-tpu-mjx/out/assets/blender_real_scene.npz")
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

  def _tile(arr, B):
    rep = (B + arr.shape[0] - 1) // arr.shape[0]
    return jnp.asarray(np.tile(arr, (rep,) + (1,) * (arr.ndim - 1))[:B])

  for k in [1, 5, 11, 16]:
    t_b0 = time.perf_counter()
    meshes = load_scaled_meshes(base_meshes, k)
    total_tris = sum(m["faces"].shape[0] for m in meshes)
    for res in [64, 128]:
      spec = build_wide_bvh_scene_spec(
          meshes,
          height=res,
          width=res,
          cam_fovy=float(ds["cam_fovy"]),
          znear=float(ds["znear"]),
          zfar=float(ds["zfar"]),
          light_pos=tuple(float(x) for x in ds["light_pos"]),
      )
      dev_arr = to_device_arrays(spec)
      for B in [64, 256]:
        if res == 64 and B == 256:
          pass
        oxp = _tile(ds["obj_xpos"], B)
        oxm = _tile(ds["obj_xmat"], B)
        cxp = _tile(ds["cam_xpos"], B)
        cxm = _tile(ds["cam_xmat"], B)

        fn = jax.jit(lambda da, a, b, c, d: render_wide_bvh_batch(spec, da, a, b, c, d, interpret=False).rgb)
        t_c0 = time.perf_counter()
        y = fn(dev_arr, oxp, oxm, cxp, cxm)
        jax.block_until_ready(y)
        compile_s = time.perf_counter() - t_c0

        n_steps = 10
        t0 = time.perf_counter()
        for _ in range(n_steps):
          y = fn(dev_arr, oxp, oxm, cxp, cxm)
        jax.block_until_ready(y)
        ms = (time.perf_counter() - t0) * 1e3 / n_steps
        fps = B * 1e3 / ms
        print(
            f"tris={total_tris:>9d} res={res}x{res} B={B:>4d} | compile={compile_s:5.2f}s | {ms:7.2f} ms | {fps:8.1f} FPS",
            flush=True,
        )


if __name__ == "__main__":
  main()
