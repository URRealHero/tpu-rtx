"""Summarize WP1 fetch-count JSONs (count_run.py output) for one scene as a Markdown table.

python summarize_wp1.py scene_name counts1.json [counts2.json ...]

Per ray set and bounce: leaf fetches with TPU-RTX's 8x16 pixel packets, with the best sort key
found, and the 'perfect sharing' floor = (ray-leaf pairs needed) / 128, i.e. if every leaf fetch
served 128 rays that each needed that leaf. 'needed' is counted with the packet's traversal order,
so the floor uses the smallest count over all orders.
"""

import json
import sys
from collections import defaultdict


def main():
  scene = sys.argv[1]
  runs = []
  for p in sys.argv[2:]:
    for r in json.load(open(p))["runs"]:
      r.setdefault("ray_set", "closest")
      runs.append(r)
  by = defaultdict(dict)
  for r in runs:
    by[(r["ray_set"], r["bounce"])][r["order"]] = r
  print(f"### {scene}: leaf fetches per sample (thousands)\n")
  print("| Rays | Bounce | Live rays | Pixel packets (as shipped) | Leaf fetches per packet | Best sort | Best sort key | Saving | Perfect-sharing floor | Floor saving |")
  print("|---|---|---|---|---|---|---|---|---|---|")
  tot = defaultdict(float)
  for (rs, b) in sorted(by, key=lambda k: (["closest", "sun_shadow", "area_shadow"].index(k[0]), k[1])):
    d = by[(rs, b)]
    if "pixel_tiles" not in d or d["pixel_tiles"]["live_rays"] == 0:
      continue
    base = d["pixel_tiles"]
    cands = {k: v for k, v in d.items() if k.startswith("sort_") or k == "compact"}
    best_k = min(cands, key=lambda k: cands[k]["leaf_fetches"]) if cands else "pixel_tiles"
    best = d[best_k] if b > 0 or rs != "closest" else base  # camera rays keep pixel packets
    if b == 0 and rs == "closest":
      best_k = "pixel_tiles"
    floor = min(v["ray_leaf_pairs_needed"] for v in d.values()) / 128.0
    tot["base"] += base["leaf_fetches"]
    tot["best"] += best["leaf_fetches"]
    tot["floor"] += floor
    print(f"| {rs} | {b} | {base['live_rays']:,} | {base['leaf_fetches'] / 1e3:,.1f} | {base['leaf_fetches_per_packet_mean']:.0f} | "
          f"{best['leaf_fetches'] / 1e3:,.1f} | `{best_k}` | {base['leaf_fetches'] / max(best['leaf_fetches'], 1):.2f}x | "
          f"{floor / 1e3:,.1f} | {base['leaf_fetches'] / max(floor, 1):.1f}x |")
  print(f"| **all** | | | **{tot['base'] / 1e3:,.1f}** | | **{tot['best'] / 1e3:,.1f}** | | **{tot['base'] / tot['best']:.2f}x** | **{tot['floor'] / 1e3:,.1f}** | **{tot['base'] / tot['floor']:.1f}x** |")
  agree = min(r["hit_agreement_with_embree"] for r in runs)
  print(f"\nLowest hit agreement with Embree over all {len(runs)} runs: {agree:.6f}")


if __name__ == "__main__":
  main()
