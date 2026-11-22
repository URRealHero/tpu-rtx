"""Inspect Sponza glTF bounds and test camera pose + 1M anisotropic 3DGS on w3."""

from __future__ import annotations

import time
import jax
import jax.numpy as jnp
import numpy as np

from gltf_to_3dgs import load_gltf_triangles, sample_anisotropic_3dgs_from_gltf, look_at_cam_mat
from gs3d_wave2_10m import render_massive_scene_1c


def main():
  t0 = time.perf_counter()
  sc = load_gltf_triangles("/dev/shm/zaychik-tpu-mjx-w3/assets/gltf-assets/Models/Sponza/glTF/Sponza.gltf")
  print(f"Loaded Sponza: {len(sc['v0'])} tris, {len(sc['materials'])} materials in {time.perf_counter()-t0:.2f}s", flush=True)
  print("bbox_min:", sc["bbox_min"], "bbox_max:", sc["bbox_max"], flush=True)

  t1 = time.perf_counter()
  gs = sample_anisotropic_3dgs_from_gltf(sc, 10_000_000, seed=42)
  print(f"Sampled 10M anisotropic Gaussians in {time.perf_counter()-t1:.2f}s (rad={float(gs['splat_radius']):.5f} m)", flush=True)

  # Inside the sunlit open Sponza courtyard looking down the grand colonnade
  cam_pos = np.array([-7.2, 2.3, -1.1], dtype=np.float32)
  target = np.array([7.5, 3.8, 0.5], dtype=np.float32)
  cam_mat = look_at_cam_mat(cam_pos, target, world_up=(0.0, 1.0, 0.0))
  print("cam_pos:", cam_pos, "target:", target, flush=True)

  fn = jax.jit(
      lambda a, b, c, d, e, f, g: render_massive_scene_1c(
          a, b, c, d, e, f, height=1080, width=1920, nrm=g, cam_fovy=52.0, znear=0.25, k_macro=4096, use_bf16_mxu=False
      )
  )
  img = fn(
      jnp.asarray(gs["mu"]), jnp.asarray(gs["cov6"]), jnp.asarray(gs["rgb"]),
      jnp.asarray(gs["opacity"]), jnp.asarray(cam_pos), jnp.asarray(cam_mat), jnp.asarray(gs["nrm"])
  )
  jax.block_until_ready(img)
  img_np = np.asarray(img)
  print(f"Preview 512x512 1M Sponza: mean={img_np.mean():.4f}, std={img_np.std():.4f}, min={img_np.min():.4f}, max={img_np.max():.4f}", flush=True)
  from PIL import Image
  Image.fromarray(np.clip(np.round(img_np * 255.0), 0, 255).astype(np.uint8)).save("/dev/shm/zaychik-tpu-mjx-w3/out/preview_sponza.png")


if __name__ == "__main__":
  main()
