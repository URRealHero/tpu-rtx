"""Run TPU-RTX's own Open Data path tracer (bvh_pbr_pt.build_complex_pbr_pathtracer) on CPU.

The Pallas kernels run in TPU interpret mode; the image is split across N fake CPU
devices exactly as TPU-RTX splits it across chips. PSNR/SSIM use TPU-RTX's own
compute_psnr/compute_ssim from run_opendata_pt.py.

python render_real_cpu.py --npz classroom.npz --gt classroom_cycles_gt.png --out out_dir \
    --devices 40 --spp 16 --bounces 3
"""

import argparse
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--npz", required=True)
ap.add_argument("--gt", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--scene", default="classroom")
ap.add_argument("--devices", type=int, default=40)
ap.add_argument("--spp", type=int, default=16)
ap.add_argument("--bounces", type=int, default=3)
ap.add_argument("--seed", type=int, default=100)
args = ap.parse_args()

os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + f" --xla_force_host_platform_device_count={args.devices}"

import common  # noqa: E402,F401

import json  # noqa: E402
import time  # noqa: E402

import jax  # noqa: E402
import jax.experimental.pallas.tpu as pltpu  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from bvh_pbr_pt import build_complex_pbr_pathtracer, build_pbr_wide_bvh  # noqa: E402
from run_opendata_pt import _align_res, _build_rig_from_lights, compute_psnr, compute_ssim, stitch_strip  # noqa: E402


def main():
  os.makedirs(args.out, exist_ok=True)
  ds = np.load(args.npz, allow_pickle=False)
  verts = ds["verts"]
  lights = json.loads(bytes(ds["lights_json"]).decode("utf-8"))
  bmin, bmax = verts.min(0), verts.max(0)
  rig = _build_rig_from_lights(lights, args.scene, bmin, bmax)
  t0 = time.perf_counter()
  bvh = build_pbr_wide_bvh(args.scene, verts, ds["faces"], ds["tri_albedo"], ds["tri_roughness"], ds["tri_mat_code"], vert_normals=ds["vnorm"])
  t_bvh = time.perf_counter() - t0
  gt = np.asarray(Image.open(args.gt).convert("RGB"), dtype=np.float32) / 255.0
  gt_h, gt_w = gt.shape[:2]
  W, H = _align_res(gt_w, gt_h, 4)
  devs = jax.devices()
  assert len(devs) == args.devices, devs
  fn, mesh, rep = build_complex_pbr_pathtracer(bvh, devs, H, W, float(ds["cam_fovy"]), 0.05, 120.0, rig, spp=args.spp, max_bounces=args.bounces)
  cxp = jax.device_put(ds["cam_pos"][None, :], rep)
  cxm = jax.device_put(ds["cam_xmat"][None, :, :], rep)
  rng = jax.device_put(jax.random.PRNGKey(args.seed), rep)
  with pltpu.force_tpu_interpret_mode():
    t0 = time.perf_counter()
    y = fn(cxp, cxm, rng)
    jax.block_until_ready(y)
    t_render = time.perf_counter() - t0
  img = stitch_strip(y)
  assert img.shape[:2] == (gt_h, gt_w), (img.shape, gt.shape)
  psnr = compute_psnr(img, gt)
  ssim = compute_ssim(img, gt)
  Image.fromarray(np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(os.path.join(args.out, "tpurtx_real.png"))
  np.save(os.path.join(args.out, "tpurtx_real.npy"), img)
  rec = {
      "what": "TPU-RTX bvh_pbr_pt path tracer, unmodified, Pallas TPU interpret mode on CPU",
      "scene": args.scene, "res": f"{H}x{W}", "spp": args.spp, "max_bounces": args.bounces,
      "cpu_devices": args.devices, "tris": bvh.n_tris_orig, "rig": {k: (list(v) if isinstance(v, tuple) else v) for k, v in rig.__dict__.items()},
      "bvh_build_s": round(t_bvh, 2), "cpu_render_s_incl_compile": round(t_render, 1),
      "psnr_vs_gt_db": round(psnr, 3), "ssim_vs_gt": round(ssim, 4), "gt": os.path.abspath(args.gt),
  }
  with open(os.path.join(args.out, "result.json"), "w") as f:
    json.dump(rec, f, indent=1)
  print("RESULT", json.dumps(rec), flush=True)


if __name__ == "__main__":
  main()
