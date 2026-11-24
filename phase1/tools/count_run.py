"""WP1 experiment: how many BVH fetches does TPU-RTX's packet traversal make for each bounce of
Classroom, and how many does ray sorting or compaction save?

Rays come from the CPU twin (twin_run.py --dump-rays): every closest-hit ray of sample 0 at each
bounce. Each ordering packs the rays into 128-ray packets and runs TPU-RTX's kernel with fetch
counters (counted_kernel.py) in Pallas interpret mode, spread over worker processes. Hit distances
are checked against Embree.

python count_run.py --npz classroom.npz --rays twin_rays.npz --out out.json --workers 48 \
    [--bounces 0 1 2] [--orders pixel_tiles compact sort_origin ...] [--max-rays N]
"""

import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np

CHUNK = 1024  # rays per kernel call (8 packets); fixed so each worker compiles once

_G = {}


def _worker_init(npz_path):
  os.environ["JAX_PLATFORMS"] = "cpu"
  os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
  import common  # noqa: F401
  import jax
  import jax.numpy as jnp
  from bvh_pbr_pt import build_pbr_wide_bvh
  from counted_kernel import pbr_raycast_counted
  ds = np.load(npz_path, allow_pickle=False)
  bvh = build_pbr_wide_bvh("s", ds["verts"], ds["faces"], ds["tri_albedo"], ds["tri_roughness"], ds["tri_mat_code"], vert_normals=ds["vnorm"])
  _G["bv"] = [jnp.asarray(x) for x in (bvh.l2_roots_aabb, bvh.l2_aabbs, bvh.l0_aabbs, bvh.leaf_bank)]
  _G["fn"] = jax.jit(lambda a, b, c, d, o, r, t: pbr_raycast_counted(a, b, c, d, o, r, t, 1, CHUNK, interpret=True))


def _worker_run(task):
  import jax.numpy as jnp
  key, ro, rd, tl = task
  n = len(ro)
  if n < CHUNK:  # pad with dead rays (t_limit 0), which never drive traversal
    pad = CHUNK - n
    ro = np.concatenate([ro, np.zeros((pad, 3), np.float32)])
    rd = np.concatenate([rd, np.tile(np.array([[0.0, 0.0, 1.0]], np.float32), (pad, 1))])
    tl = np.concatenate([tl, np.zeros(pad, np.float32)])
  dist, cnt = _G["fn"](*_G["bv"], jnp.asarray(ro[None]), jnp.asarray(rd[None]), jnp.asarray(tl[None]))
  return key, np.asarray(dist)[0, :n], np.asarray(cnt)


def morton30(p, lo, hi):
  q = np.clip(((p - lo) / np.maximum(hi - lo, 1e-9) * 1023.0).astype(np.int64), 0, 1023)

  def spread(x):
    x = (x | (x << 16)) & 0x030000FF
    x = (x | (x << 8)) & 0x0300F00F
    x = (x | (x << 4)) & 0x030C30C3
    x = (x | (x << 2)) & 0x09249249
    return x

  return (spread(q[:, 0]) << 2) | (spread(q[:, 1]) << 1) | spread(q[:, 2])


def octahedral_bin(d, n=16):
  """Direction -> one of n*n octahedral-map cells."""
  a = d / np.maximum(np.abs(d).sum(1, keepdims=True), 1e-12)
  x, y, z = a[:, 0], a[:, 1], a[:, 2]
  ox = np.where(z >= 0, x, (1 - np.abs(y)) * np.sign(x + 1e-30))
  oy = np.where(z >= 0, y, (1 - np.abs(x)) * np.sign(y + 1e-30))
  ix = np.clip(((ox + 1) * 0.5 * n).astype(np.int64), 0, n - 1)
  iy = np.clip(((oy + 1) * 0.5 * n).astype(np.int64), 0, n - 1)
  return iy * n + ix


def orderings(ro, rd, tl, H, W, lo, hi, names):
  R = H * W
  tile_order = np.arange(R).reshape(H // 8, 8, W // 16, 16).transpose(0, 2, 1, 3).ravel()
  live_t = tile_order[tl[tile_order] > 0]
  live = np.nonzero(tl > 0)[0]
  m = morton30(ro[live], lo, hi)
  octant = ((rd[live, 0] < 0).astype(np.int64) << 2) | ((rd[live, 1] < 0).astype(np.int64) << 1) | (rd[live, 2] < 0).astype(np.int64)
  ob = octahedral_bin(rd[live], 16)
  out = {}
  for nm in names:
    if nm == "pixel_tiles":
      out[nm] = tile_order          # exactly the renderer's 8x16 packets, dead rays included
    elif nm == "compact":
      out[nm] = live_t              # dead rays removed, otherwise tile order
    elif nm == "sort_origin":
      out[nm] = live[np.argsort(m, kind="stable")]
    elif nm == "sort_octant_origin":
      out[nm] = live[np.argsort((octant << 30) | m, kind="stable")]
    elif nm == "sort_dir256_origin":
      out[nm] = live[np.argsort((ob << 30) | m, kind="stable")]
    elif nm == "sort_cell_dir256":
      out[nm] = live[np.argsort(((m >> 12) << 8) | ob, kind="stable")]  # 6-bit-per-axis cells, direction within a cell
    elif nm == "random":
      out[nm] = live[np.random.default_rng(0).permutation(len(live))]
    elif nm == "sort_zsign_origin":   # 2 direction classes: up/down
      out[nm] = live[np.argsort(((rd[live, 2] < 0).astype(np.int64) << 30) | m, kind="stable")]
    elif nm == "sort_cube_origin":    # 6 direction classes: dominant axis and sign
      ax = np.argmax(np.abs(rd[live]), 1)
      sg = (rd[live][np.arange(len(live)), ax] < 0).astype(np.int64)
      out[nm] = live[np.argsort(((ax * 2 + sg) << 30) | m, kind="stable")]
    elif nm.startswith("sort_oct") and nm.endswith("_origin") and nm[8:-7].isdigit():
      n = int(nm[8:-7])               # n x n octahedral direction cells, then hit-point Morton
      out[nm] = live[np.argsort((octahedral_bin(rd[live], n) << 30) | m, kind="stable")]
    elif nm.startswith("sort_octant_m") and nm[13:].isdigit():
      k = int(nm[13:])                # octant, then only the top k bits of the Morton code, then direction cell
      out[nm] = live[np.argsort((octant << 40) | ((m >> (30 - k)) << 8) | ob, kind="stable")]
    else:
      raise ValueError(nm)
  return out


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--npz", required=True)
  ap.add_argument("--rays", required=True)
  ap.add_argument("--out", required=True)
  ap.add_argument("--H", type=int, default=360)
  ap.add_argument("--W", type=int, default=640)
  ap.add_argument("--workers", type=int, default=48)
  ap.add_argument("--bounces", type=int, nargs="*", default=[0, 1, 2])
  ap.add_argument("--orders", nargs="*", default=["pixel_tiles", "compact", "sort_origin", "sort_octant_origin", "sort_dir256_origin", "sort_cell_dir256"])
  ap.add_argument("--max-rays", type=int, default=None, help="debug: only the first N rays of each ordering")
  args = ap.parse_args()

  import common
  ds = np.load(args.npz, allow_pickle=False)
  lo, hi = ds["verts"].min(0), ds["verts"].max(0)
  rays = np.load(args.rays)
  ctx = mp.get_context("spawn")
  results = {"what": "TPU-RTX Wide-BVH-64 closest-hit kernel with fetch counters, Pallas interpret mode on CPU",
             "scene_npz": os.path.abspath(args.npz), "rays": os.path.abspath(args.rays), "chunk": CHUNK, "runs": []}
  with ctx.Pool(args.workers, initializer=_worker_init, initargs=(os.path.abspath(args.npz),)) as pool:
    for b in args.bounces:
      ro, rd, tl = rays[f"b{b}_ro"], rays[f"b{b}_rd"], rays[f"b{b}_tlim"]
      t_ref, _ = common.embree_closest_hit(ds["verts"], ds["faces"], ro[tl > 0], rd[tl > 0])
      t_ref_full = np.full(len(ro), np.inf, np.float32)
      t_ref_full[tl > 0] = t_ref
      for nm, idx in orderings(ro, rd, tl, args.H, args.W, lo, hi, args.orders).items():
        if args.max_rays:
          idx = idx[: args.max_rays]
        tasks = [((i,), ro[idx[i:i + CHUNK]], rd[idx[i:i + CHUNK]], tl[idx[i:i + CHUNK]]) for i in range(0, len(idx), CHUNK)]
        t0 = time.perf_counter()
        cnt_all, dist_all = [], np.empty(len(idx), np.float32)
        for key, dist, cnt in pool.imap_unordered(_worker_run, tasks, chunksize=1):
          i = key[0]
          dist_all[i:i + len(dist)] = dist
          cnt_all.append(cnt)
        wall = time.perf_counter() - t0
        cnt = np.concatenate(cnt_all, 0)
        live_mask = tl[idx] > 0
        hit_k = (dist_all < tl[idx] - 1e-4) & live_mask
        hit_e = np.isfinite(t_ref_full[idx]) & (t_ref_full[idx] < tl[idx]) & live_mask
        packets = cnt[cnt[:, 4] > 0]
        rec = {
            "bounce": b, "order": nm, "live_rays": int(live_mask.sum()), "packets_with_live_rays": int(len(packets)),
            "packets_total": int(len(cnt)),
            "l2_fetches": int(cnt[:, 0].sum()), "l1_fetches": int(cnt[:, 1].sum()), "leaf_fetches": int(cnt[:, 2].sum()),
            "ray_leaf_pairs_needed": int(cnt[:, 3].sum()),
            "leaf_fetches_per_live_ray": float(cnt[:, 2].sum() / max(live_mask.sum(), 1)),
            "leaf_fetches_per_packet_mean": float(packets[:, 2].mean()) if len(packets) else 0.0,
            "lane_utilization": float(cnt[:, 3].sum() / max(128.0 * cnt[:, 2].sum(), 1.0)),
            "hit_agreement_with_embree": float(np.mean(hit_k[live_mask] == hit_e[live_mask])) if live_mask.any() else 1.0,
            "cpu_wall_s": round(wall, 1),
        }
        results["runs"].append(rec)
        print("COUNT", json.dumps(rec), flush=True)
        with open(args.out, "w") as f:
          json.dump(results, f, indent=1)


if __name__ == "__main__":
  main()
