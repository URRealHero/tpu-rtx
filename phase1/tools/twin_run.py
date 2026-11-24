"""Render a scene with the CPU twin of TPU-RTX's path tracer and score it against a Cycles image.

python twin_run.py --npz classroom.npz --gt classroom_cycles_gt.png --out dir --spp 16 --bounces 3 [--tag name]
  [--no-soften] [--no-clamps] [--compare path.npy ...]
"""

import argparse
import json
import os
import time

import common  # noqa: F401
import numpy as np
from PIL import Image

from run_opendata_pt import _build_rig_from_lights, compute_psnr, compute_ssim
from twin import Twin, TwinOptions, agx_tonemap_srgb, standard_view_srgb


def load_rig(npz_path, scene):
  ds = np.load(npz_path, allow_pickle=False)
  lights = json.loads(bytes(ds["lights_json"]).decode("utf-8"))
  v = ds["verts"]
  return _build_rig_from_lights(lights, scene, v.min(0), v.max(0))


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--npz", required=True)
  ap.add_argument("--gt", required=True)
  ap.add_argument("--out", required=True)
  ap.add_argument("--scene", default="classroom")
  ap.add_argument("--spp", type=int, default=16)
  ap.add_argument("--bounces", type=int, default=3)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--tag", default=None)
  ap.add_argument("--rig", help="light rig JSON to use instead of deriving it from the .npz")
  ap.add_argument("--no-soften", action="store_true")
  ap.add_argument("--no-clamps", action="store_true")
  ap.add_argument("--aovs", action="store_true")
  ap.add_argument("--dump-rays", action="store_true", help="save per-bounce rays of sample 0")
  ap.add_argument("--compare", nargs="*", default=[])
  args = ap.parse_args()
  os.makedirs(args.out, exist_ok=True)
  tag = args.tag or f"twin_{args.spp}spp_{args.bounces}b_s{args.seed}"

  gt = np.asarray(Image.open(args.gt).convert("RGB"), dtype=np.float32) / 255.0
  H, W = gt.shape[:2]
  if args.rig:  # a fixed rig (e.g. when the .npz was edited and its bounding box changed)
    from run_opendata_pt import LightRig
    rig = LightRig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in json.load(open(args.rig)).items()})
  else:
    rig = load_rig(args.npz, args.scene)
  t0 = time.perf_counter()
  tw = Twin(args.npz, rig, H, W)
  t_build = time.perf_counter() - t0
  opts = TwinOptions(dump_rays_sample=0 if args.dump_rays else None)
  if args.no_soften:
    opts.falloff_soften = 0.0
  if args.no_clamps:
    opts.clamp_area = opts.clamp_sun = opts.clamp_spec_weight = None
  t0 = time.perf_counter()
  res = tw.render(args.spp, args.bounces, seed=args.seed, opts=opts, primary_aovs=args.aovs)
  t_render = time.perf_counter() - t0
  lin = res["linear"]
  img = agx_tonemap_srgb(lin, rig.exposure).astype(np.float32)
  np.save(os.path.join(args.out, f"{tag}_linear.npy"), lin)
  np.save(os.path.join(args.out, f"{tag}_srgb.npy"), img)
  Image.fromarray(np.clip(img * 255 + 0.5, 0, 255).astype(np.uint8)).save(os.path.join(args.out, f"{tag}.png"))
  std = standard_view_srgb(lin).astype(np.float32)
  Image.fromarray(np.clip(std * 255 + 0.5, 0, 255).astype(np.uint8)).save(os.path.join(args.out, f"{tag}_standardview.png"))
  if res["aovs"] is not None:
    np.savez_compressed(os.path.join(args.out, f"{tag}_aovs.npz"), **res["aovs"])
  if res["dumps"]:
    np.savez_compressed(os.path.join(args.out, f"{tag}_rays.npz"),
                        **{f"b{d['bounce']}_{k}": d[k] for d in res["dumps"] for k in d if k != "bounce"})
  rec = {
      "tag": tag, "what": "CPU twin of TPU-RTX bvh_pbr_pt (NumPy shading port + Embree ray casts)",
      "scene": args.scene, "res": f"{H}x{W}", "spp": args.spp, "max_bounces": args.bounces, "seed": args.seed,
      "falloff_soften": opts.falloff_soften, "clamps": opts.clamp_area is not None,
      "build_s": round(t_build, 1), "render_s": round(t_render, 1),
      "psnr_vs_gt_db": round(compute_psnr(img, gt), 3), "ssim_vs_gt": round(compute_ssim(img, gt), 4),
      "psnr_standardview_vs_gt_db": round(compute_psnr(std, gt), 3),
      "mean_linear_rgb": [round(float(x), 5) for x in lin.reshape(-1, 3).mean(0)],
      "stats": res["stats"],
      "compare": {},
  }
  for c in args.compare:
    other = np.load(c)
    rec["compare"][c] = {"psnr_db": round(compute_psnr(img, other), 3), "ssim": round(compute_ssim(img, other), 4)}
  with open(os.path.join(args.out, f"{tag}.json"), "w") as f:
    json.dump(rec, f, indent=1)
  print("RESULT", json.dumps({k: v for k, v in rec.items() if k != "stats"}), flush=True)


if __name__ == "__main__":
  main()
