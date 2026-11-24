"""Where does TPU-RTX's 9.4 dB on Classroom go? Score each Cycles variant and twin render
against the Cycles reference TPU-RTX's own exporter makes, with TPU-RTX's own PSNR/SSIM code.

python gap_breakdown.py --gt classroom_cycles_gt.png --variants DIR --twin DIR --out gap.json
"""

import argparse
import glob
import json
import os

import common  # noqa: F401
import numpy as np
import OpenEXR
from PIL import Image

from run_opendata_pt import compute_psnr, compute_ssim
from twin import agx_tonemap_srgb, standard_view_srgb


def png(p):
  return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0


def exr_passes(p):
  f = OpenEXR.File(p)
  out = {}
  for part in f.parts:
    for name, ch in part.channels.items():
      out[name.replace("ViewLayer.", "")] = np.asarray(ch.pixels, dtype=np.float32)
  return out


def twin_linear(d, prefix):
  """Average every '<prefix>*_linear.npy' (independent seeds) into one linear image."""
  fs = sorted(glob.glob(os.path.join(d, prefix + "*_linear.npy")))
  if not fs:
    return None, 0
  return np.mean([np.load(f) for f in fs], axis=0), len(fs)


def lum(x):
  return 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]


def score(img, ref):
  return {"psnr_db": round(compute_psnr(img, ref), 3), "ssim": round(compute_ssim(img, ref), 4)}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--gt", required=True)
  ap.add_argument("--variants", required=True)
  ap.add_argument("--twin", required=True)
  ap.add_argument("--exposure", type=float, default=0.55)
  ap.add_argument("--out", required=True)
  args = ap.parse_args()
  gt = png(args.gt)
  V = args.variants
  res = {"gt": os.path.abspath(args.gt), "steps": [], "single_changes": [], "pairs": [], "aov_checks": {}, "luminance": {}}

  cy = {}
  for name in ("cy_ref_s2", "cy_orig_set", "cy_lights_set", "cy_scene_set", "cy_pin", "cy_b2", "cy_noclamp", "cy_lights"):
    p = os.path.join(V, name + ".png")
    if os.path.exists(p):
      cy[name] = {"srgb": png(p), "exr": exr_passes(os.path.join(V, name + ".exr"))}
  tw = {}
  for tag in ("twin_def_", "twin_raw_", "twin_nosoft_"):
    lin, n = twin_linear(args.twin, tag)
    if lin is not None:
      tw[tag] = {"lin": lin, "n_runs": n, "std": standard_view_srgb(lin).astype(np.float32),
                 "aces": agx_tonemap_srgb(lin, args.exposure).astype(np.float32)}
  t16 = sorted(glob.glob(os.path.join(args.twin, "t16_*_srgb.npy")))

  def step(label, img, note):
    if img is None:
      return
    res["steps"].append({"step": label, "note": note, **score(img, gt)})

  # Cumulative chain: each step adds one group of TPU-RTX's choices.
  step("0 Cycles, same settings, new seed", cy.get("cy_ref_s2", {}).get("srgb"), "noise floor of the 1024-spp reference")
  step("1 + TPU camera/bounces/no clamp", cy.get("cy_orig_set", {}).get("srgb"), "pinhole (no depth of field), 2 bounces after the first hit, box pixel filter, no sample clamping")
  step("2 + TPU lights", cy.get("cy_lights_set", {}).get("srgb"), "scene's 10 lamps and world replaced by TPU-RTX's 1 disk light, sun and gradient sky")
  step("3 + TPU materials/geometry export", cy.get("cy_scene_set", {}).get("srgb"), "TPU-RTX's exported triangles: one colour/roughness/material code per triangle, smooth vertex normals, no textures or bump")
  if "twin_def_" in tw:
    step("4 + TPU light transport (Standard view)", tw["twin_def_"]["std"], f"TPU-RTX's path tracer (CPU twin, {64 * tw['twin_def_']['n_runs']} spp), shown with Blender's Standard view")
    step("5 + TPU tone curve (ACES-fit, exposure 0.55)", tw["twin_def_"]["aces"], "what TPU-RTX outputs, converged")
  if t16:
    vals = [score(np.load(f), gt) for f in t16]
    res["steps"].append({"step": "6 + 16 spp noise", "note": f"TPU-RTX's output at 16 spp, mean of {len(vals)} seeds (reported: 9.44 dB)",
                         "psnr_db": round(float(np.mean([v['psnr_db'] for v in vals])), 3), "ssim": round(float(np.mean([v['ssim'] for v in vals])), 4)})

  # One change at a time against the reference.
  for name, note in (("cy_pin", "depth of field off"), ("cy_b2", "bounces capped at 2 after the first hit"),
                     ("cy_noclamp", "no sample clamping (scene clamps indirect at 3.0)"), ("cy_lights", "TPU-RTX's lights only")):
    if name in cy:
      res["single_changes"].append({"change": note, **score(cy[name]["srgb"], gt)})

  # Renderer-only comparisons on the same (TPU-exported) scene, both in Standard view.
  if "cy_scene_set" in cy:
    ref = cy["cy_scene_set"]["srgb"]
    for tag, note in (("twin_def_", "TPU-RTX estimator as shipped"), ("twin_nosoft_", "without the +0.08 light falloff softening"),
                      ("twin_raw_", "without softening and clamps")):
      if tag in tw:
        res["pairs"].append({"a": "Cycles on TPU-RTX's exported scene and lights", "b": f"CPU twin: {note}", **score(tw[tag]["std"], ref),
                             "mean_lum_ratio_b_over_a": round(float(lum(tw[tag]["lin"]).mean() / lum(cy["cy_scene_set"]["exr"]["Combined"][..., :3]).mean()), 4)})

  # Linear mean luminance of each image relative to the reference render.
  if "cy_ref_s2" in cy:
    L0 = float(lum(cy["cy_ref_s2"]["exr"]["Combined"][..., :3]).mean())
    for name, d in cy.items():
      res["luminance"][name] = round(float(lum(d["exr"]["Combined"][..., :3]).mean()) / L0, 4)
    for tag, d in tw.items():
      res["luminance"][tag] = round(float(lum(d["lin"]).mean()) / L0, 4)

  # Primary-hit checks: geometry/camera (depth) and materials (albedo) against Cycles passes.
  aov_f = sorted(glob.glob(os.path.join(args.twin, "*_aovs.npz")))
  if aov_f and "cy_orig_set" in cy:
    a = np.load(aov_f[0])
    ex = cy["cy_orig_set"]["exr"]
    zc = ex.get("Depth.Z", ex.get("Depth"))
    if zc is not None:
      zc = zc.reshape(gt.shape[:2])
      zt = a["depth"]
      fin = np.isfinite(zt) & np.isfinite(zc) & (zc < 1e5)
      rel = np.abs(zt[fin] - zc[fin]) / np.maximum(zc[fin], 1e-6)
      res["aov_checks"]["depth"] = {"pixels_compared": int(fin.sum()), "within_1pct": round(float(np.mean(rel < 0.01)), 4),
                                    "median_rel_err": float(np.median(rel)),
                                    "note": "twin depth is distance from the near plane along the ray; Cycles Depth pass"}
    dc = ex.get("Diffuse Color")
    if dc is not None:
      dc = dc[..., :3]
      opq = (a["mcode"] >= 0) & (a["mcode"] < 0.5)
      alb = a["albedo"]
      both = opq & (dc.max(-1) > 0)
      res["aov_checks"]["albedo"] = {"pixels_compared": int(both.sum()),
                                     "psnr_db_linear": round(compute_psnr(np.clip(alb[both], 0, 1), np.clip(dc[both], 0, 1)), 3),
                                     "mean_tpu": [round(float(x), 4) for x in alb[both].mean(0)],
                                     "mean_cycles": [round(float(x), 4) for x in dc[both].mean(0)],
                                     "note": "non-metal opaque pixels; Cycles Diffuse Color pass vs TPU-RTX per-triangle albedo"}
  with open(args.out, "w") as f:
    json.dump(res, f, indent=1)
  print(json.dumps(res, indent=1))


if __name__ == "__main__":
  main()
