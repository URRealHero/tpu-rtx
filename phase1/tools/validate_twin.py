"""Compare TPU-RTX's real renderer output with CPU-twin outputs at the same settings.

python validate_twin.py --real real.npy --twins a_srgb.npy b_srgb.npy --ref ref_srgb.npy --out validation.json
If the twin is faithful, real-vs-ref PSNR equals twin-vs-ref PSNR, and real-vs-twin PSNR equals the
twin-vs-twin noise floor (images differ only by Monte Carlo noise).
"""

import argparse
import json

import common  # noqa: F401
import numpy as np

from run_opendata_pt import compute_psnr, compute_ssim


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--real", required=True)
  ap.add_argument("--twins", nargs=2, required=True)
  ap.add_argument("--ref", required=True)
  ap.add_argument("--out", required=True)
  a = ap.parse_args()
  real, t1, t2, ref = (np.load(p) for p in (a.real, *a.twins, a.ref))
  res = {
      "real_vs_ref": {"psnr_db": compute_psnr(real, ref), "ssim": compute_ssim(real, ref)},
      "twin1_vs_ref": {"psnr_db": compute_psnr(t1, ref), "ssim": compute_ssim(t1, ref)},
      "twin2_vs_ref": {"psnr_db": compute_psnr(t2, ref), "ssim": compute_ssim(t2, ref)},
      "real_vs_twin1_psnr_db": compute_psnr(real, t1),
      "real_vs_twin2_psnr_db": compute_psnr(real, t2),
      "twin1_vs_twin2_psnr_db (noise floor)": compute_psnr(t1, t2),
      "mean_srgb": {"real": real.mean((0, 1)).tolist(), "twin1": t1.mean((0, 1)).tolist(), "ref": ref.mean((0, 1)).tolist()},
      "inputs": vars(a),
  }
  json.dump(res, open(a.out, "w"), indent=1)
  print(json.dumps(res, indent=1))


if __name__ == "__main__":
  main()
