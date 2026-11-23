#!/bin/bash
set -euo pipefail

# Fail-closed V2 TPU benchmark launcher for active TPU idle windows:
# - Benchmarks Stage 1 V1 tiled vs V2 v2_soa vs V2 native (8,128) pallas_v2 (interpret=False)
# - Benchmarks V2 Static Cluster-BVH Triangle Mesh Raytracer (jax_cluster & pallas)
# - Benchmarks V2 Rigid-Body Tile-3DGS (jax_tile & pallas)
# NEVER interferes with any active active workload.

MODE="${1:-run}"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PKG_DIR="/tmp/tpu-rtx-pkgs"
PY="${PYTHON_BIN:-$(command -v python3)}"
[ -x "$PY" ] || PY="$(command -v python3)"

echo "=== TPU-MJX V2 Live Silicon Benchmark (mode=$MODE) ==="

if [ -f /tmp/libtpu_lockfile ] && fuser /tmp/libtpu_lockfile >/dev/null 2>&1; then
  echo "GUARD BLOCKED: /tmp/libtpu_lockfile held by active process." >&2
  exit 2
fi

ACTIVE_PROCS=$(ps -u "${USER:-$(whoami)}" -o args= | grep -cE "[t]rain_unified_tokenizer|[e]val_unified_tokenizer|[u]mp_eval|[q]c_g4_" || true)
if [ "$ACTIVE_PROCS" -gt 0 ]; then
  echo "GUARD BLOCKED: $ACTIVE_PROCS active active process(es) detected." >&2
  exit 2
fi

if [ "$MODE" = "--dry-run" ]; then
  echo "V2 DRY-RUN OK: guard passed, ready at $ROOT_DIR (python=$PY)"
  exit 0
fi

unset $(env | awk -F= '/^MEGASCALE_/ {print $1}') 2>/dev/null || true
export TPU_SKIP_MDS_QUERY=1
export TPU_HOST_BOUNDS=1,1,1
export TPU_CHIPS_PER_HOST_BOUNDS=1,1,1
export TPU_VISIBLE_DEVICES=0
export TPU_PROCESS_ADDRESSES=127.0.0.1:8476
export TPU_PROCESS_PORT=8476
export TPU_WORKER_ID=0
export CLOUD_TPU_TASK_ID=0
export JAX_COMPILATION_CACHE_DIR=/tmp/tpu_rtx_jax_cache
export PYTHONPATH="${PKG_DIR}:${ROOT_DIR}:${PYTHONPATH:-}"
cd "$ROOT_DIR"

echo "1. Verify V2 parity & zero-copy HLO on live TPU (interpret=False):"
"$PY" -u tests/test_v2_pallas_and_assets.py

echo "2. Stage 1 V1 tiled vs V2 v2_soa vs V2 native Pallas (B=1024, 2048, 4096 x 64x64):"
for M in tiled v2_soa pallas_v2; do
  "$PY" -u bench.py --scene panda_pick_cube --batches 1024 2048 4096 --res 64 \
    --mode "$M" --n_chips 1 --steps 20 --out_json "out/metrics/v2_live_${M}_1c.json"
done

echo "3. Real Blender Mesh Raytracer & Rigid-Body 3DGS E2E Throughput (B=1024, 64x64):"
"$PY" - << "PYEOF"
import json, time
import jax, jax.numpy as jnp, numpy as np
from tpu_mjx_render.mesh_ray import build_mesh_scene_spec, render_mesh_batch
from tpu_mjx_render.gs3d import mesh_scene_to_rigid_3dgs, render_gs3d_batch

ds = np.load("out/assets/blender_real_scene.npz")
M = int(ds["n_meshes"])
meshes = [{"name": str(ds[f"mesh_{m}_name"]), "verts": ds[f"mesh_{m}_verts"], "faces": ds[f"mesh_{m}_faces"], "rgba": ds[f"mesh_{m}_rgba"]} for m in range(M)]
mspec = build_mesh_scene_spec(meshes, height=64, width=64, cam_fovy=float(ds["cam_fovy"]), znear=float(ds["znear"]), zfar=float(ds["zfar"]), light_pos=tuple(float(x) for x in ds["light_pos"]))
gs_spec = mesh_scene_to_rigid_3dgs(meshes, total_gaussians=2048, height=64, width=64, cam_fovy=float(ds["cam_fovy"]), znear=float(ds["znear"]), light_pos=tuple(float(x) for x in ds["light_pos"]))

B, N_STEPS = 1024, 20
def _tile(arr):
  rep = (B + arr.shape[0] - 1) // arr.shape[0]
  return jnp.asarray(np.tile(arr, (rep,) + (1,) * (arr.ndim - 1))[:B])

oxp, oxm, cxp, cxm = _tile(ds["obj_xpos"]), _tile(ds["obj_xmat"]), _tile(ds["cam_xpos"]), _tile(ds["cam_xmat"])
out = {"backend": jax.default_backend(), "B": B, "res": "64x64", "runs": []}

for label, fn in [
    ("mesh_jax_cluster", jax.jit(lambda a, b, c, d: render_mesh_batch(mspec, a, b, c, d, mode="jax_cluster").rgb)),
    ("mesh_pallas", jax.jit(lambda a, b, c, d: render_mesh_batch(mspec, a, b, c, d, mode="pallas", interpret=False).rgb)),
    ("gs3d_jax_tile", jax.jit(lambda a, b, c, d: render_gs3d_batch(gs_spec, a, b, c, d, tile_size=8, top_k=128, mode="jax_tile")[0])),
    ("gs3d_pallas", jax.jit(lambda a, b, c, d: render_gs3d_batch(gs_spec, a, b, c, d, tile_size=8, top_k=128, mode="pallas", interpret=False)[0])),
]:
  y = fn(oxp, oxm, cxp, cxm)
  jax.block_until_ready(y)
  t0 = time.perf_counter()
  for _ in range(N_STEPS):
    y = fn(oxp, oxm, cxp, cxm)
  jax.block_until_ready(y)
  ms = (time.perf_counter() - t0) * 1e3 / N_STEPS
  rec = {"pipeline": label, "ms": round(ms, 2), "fps": round(B * 1e3 / ms, 1)}
  out["runs"].append(rec)
  print("V2 ASSET BENCH:", rec, flush=True)

with open("out/metrics/v2_live_assets_1c.json", "w") as f:
  json.dump(out, f, indent=2)
PYEOF
echo "=== V2 LIVE BENCHMARK COMPLETE ==="
