"""Write an edited copy of a TPU-RTX scene export, for isolating renderer differences.

python edit_npz.py in.npz out.npz [--no-metal] [--no-glass]
  --no-metal  opaque triangles get mat_code 0 (TPU-RTX's dielectric GGX + Lambert)
  --no-glass  glass triangles (mat_code in [1.8, 2.8)) are removed
"""

import argparse

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("dst")
ap.add_argument("--no-metal", action="store_true")
ap.add_argument("--no-glass", action="store_true")
a = ap.parse_args()
d = dict(np.load(a.src, allow_pickle=False))
mc = d["tri_mat_code"].copy()
keep = np.ones(len(mc), bool)
if a.no_glass:
  keep &= ~((mc >= 1.8) & (mc < 2.8))
if a.no_metal:
  mc[mc < 1.8] = 0.0
d["tri_mat_code"] = mc
for k in ("faces", "tri_albedo", "tri_roughness", "tri_mat_code"):
  d[k] = d[k][keep]
np.savez(a.dst, **d)
print("EDITED", a.dst, "tris", int(keep.sum()), "of", len(keep))
