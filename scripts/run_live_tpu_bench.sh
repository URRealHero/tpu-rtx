#!/bin/bash
set -euo pipefail

# Fail-closed live TPU silicon benchmark for TPU-MJX Batch Renderer.
# NEVER touches an active UMP TPU process or UMP venv.

MODE="${1:-run}"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PKG_DIR="/tmp/zaychik-tpu-mjx-pkgs"
PY="/home/zaychik/ump-v5p-64-1-env/train_venv/bin/python"
[ -x "$PY" ] || PY="$(command -v python3)"

echo "=== TPU-MJX Live Silicon Benchmark (mode=$MODE) ==="

# 1. Fail-closed guard against interfering with active UMP workloads
if [ -f /tmp/libtpu_lockfile ]; then
  if fuser /tmp/libtpu_lockfile >/dev/null 2>&1; then
    echo "GUARD BLOCKED: /tmp/libtpu_lockfile is held by an active process. Refusing to touch TPU." >&2
    exit 2
  fi
fi

UMP_PROCS=$(ps -u "${USER:-zaychik}" -o args= | grep -cE "[t]rain_unified_tokenizer|[e]val_unified_tokenizer|[u]mp_eval|[q]c_g4_" || true)
if [ "$UMP_PROCS" -gt 0 ]; then
  echo "GUARD BLOCKED: $UMP_PROCS UMP process(es) active on this host. Refusing to touch TPU." >&2
  exit 2
fi

if [ "$MODE" = "--dry-run" ]; then
  echo "DRY-RUN OK: guard passed, environment ready at $ROOT_DIR"
  exit 0
fi

# Single-host 4-chip isolation topology (when running on 1 host of a multi-host Pod)
unset $(env | awk -F= '/^MEGASCALE_/ {print $1}') 2>/dev/null || true
export TPU_SKIP_MDS_QUERY=1
export TPU_HOST_BOUNDS=1,1,1
export TPU_CHIPS_PER_HOST_BOUNDS=2,2,1
export TPU_VISIBLE_DEVICES=0,1,2,3
export TPU_PROCESS_ADDRESSES=127.0.0.1:8476
export TPU_PROCESS_PORT=8476
export TPU_WORKER_ID=0
export CLOUD_TPU_TASK_ID=0

export PYTHONPATH="${PKG_DIR}:${ROOT_DIR}:${PYTHONPATH:-}"
cd "$ROOT_DIR"

echo "1. Native TPU device check:"
"$PY" -c "import jax; devs=jax.devices(); print('Devices:', len(devs), devs[0].platform, devs[0].device_kind)"

echo "2. Native Pallas TPU kernel verification (interpret=False):"
"$PY" -c "
import jax, jax.numpy as jnp, mujoco
from mujoco import mjx
from tpu_mjx_render.renderer import RenderSceneSpec, render_exact_gbuffer
from tpu_mjx_render.scenes import get_scene_xml
m = mujoco.MjModel.from_xml_string(get_scene_xml('panda_pick_cube'))
mx = mjx.put_model(m, impl='jax')
dx = mjx.forward(mx, mjx.make_data(m, impl='jax'))
spec = RenderSceneSpec.from_mjmodel(m, 0, 64, 64)
gb_t = jax.jit(lambda x: render_exact_gbuffer(spec, x, mode='tiled'))(dx)
gb_p = jax.jit(lambda x: render_exact_gbuffer(spec, x, mode='pallas', pallas_interpret=False))(dx)
print('Native Pallas vs Tiled max depth diff:', float(jnp.max(jnp.abs(gb_t.depth - gb_p.depth))))
"

echo "3. End-to-End MJX+Render FPS Benchmark on 1 & 4 TPU chips (B=1024, 2048, 4096 x 64x64, 128x128):"
"$PY" bench.py --scene panda_pick_cube --batches 1024 2048 4096 --res 64 128 --mode tiled --n_chips 1 --steps 20 --out_json out/metrics/live_tpu_tiled_1c.json
"$PY" bench.py --scene panda_pick_cube --batches 1024 2048 4096 --res 64 128 --mode tiled --n_chips 4 --steps 20 --out_json out/metrics/live_tpu_tiled_4c.json

echo "=== LIVE TPU BENCHMARK COMPLETE ==="
