#!/bin/bash
set -euo pipefail

# Complete TPU v5p Live Silicon Suite (Single-Host 4-Chip Isolated)
# Guard: refuses to run if active training is active.
ACTIVE_PROCS=$(ps -u "${USER:-$(whoami)}" -o args= | grep -cE "[t]rain_unified_tokenizer|[e]val_unified_tokenizer|[u]mp_eval|[q]c_g4_" || true)
if [ "$ACTIVE_PROCS" -gt 0 ]; then
  echo "GUARD BLOCKED: $ACTIVE_PROCS active process(es) active." >&2
  exit 2
fi

for v in $(env | awk -F= '/^MEGASCALE_/ {print $1}'); do
  unset "$v"
done
export TPU_SKIP_MDS_QUERY=1
export TPU_HOST_BOUNDS=1,1,1
export TPU_CHIPS_PER_HOST_BOUNDS=2,2,1
export TPU_VISIBLE_DEVICES=0,1,2,3
export TPU_PROCESS_ADDRESSES=127.0.0.1:8476
export TPU_PROCESS_PORT=8476
export TPU_WORKER_ID=0
export CLOUD_TPU_TASK_ID=0
export JAX_COMPILATION_CACHE_DIR=/tmp/tpu_rtx_jax_cache
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

cd ${ROOT_DIR}
export PYTHONPATH=/tmp/tpu-rtx-pkgs:${ROOT_DIR}
PY="${PYTHON_BIN:-$(command -v python3)}"

rm -f out/metrics/SUITE_DONE out/metrics/SUITE_FAIL

echo "=== [1/4] Stage 2 M2_spatial_128 (200 ep, 50x train boundary weight) on TPU v5p ==="
"$PY" -u scripts/train_neural_shader.py \
  --dataset out/data/panda_cycles_dataset.npz \
  --gt_rgb out/data/panda_cycles_gt_rgb.npz \
  --archs M2_spatial_128 --epochs 200 --batch_size 32 --merge_existing
"$PY" -u tests/test_stage2_hlo.py

echo "=== [2/4] Live TPU v5p E2E Benchmarks: Cartpole (1 chip) ==="
"$PY" -u bench.py --scene cartpole --batches 1024 2048 4096 --res 64 --mode tiled --n_chips 1 --steps 50 --out_json out/metrics/live_tpu_cartpole_1c.json

echo "=== [3/4] Live TPU v5p E2E Benchmarks: PandaPickCube (1 chip & 4 chips) ==="
"$PY" -u bench.py --scene panda_pick_cube --batches 1024 2048 4096 --res 64 --mode tiled --n_chips 1 --steps 30 --out_json out/metrics/live_tpu_panda_tiled_1c.json
"$PY" -u bench.py --scene panda_pick_cube --batches 1024 --res 64 --mode naive --n_chips 1 --steps 30 --out_json out/metrics/live_tpu_panda_naive_1c.json
"$PY" -u bench.py --scene panda_pick_cube --batches 1024 2048 4096 --res 64 --mode tiled --neural_ckpt out/metrics/M2_spatial_128_params.pkl --n_chips 1 --steps 30 --out_json out/metrics/live_tpu_panda_m2_1c.json
"$PY" -u bench.py --scene panda_pick_cube --batches 1024 2048 4096 --res 64 --mode tiled --neural_ckpt out/metrics/M2_spatial_128_params.pkl --n_chips 4 --steps 30 --out_json out/metrics/live_tpu_panda_m2_4c.json
"$PY" -u bench.py --scene panda_pick_cube --batches 1024 --res 128 --mode tiled --neural_ckpt out/metrics/M2_spatial_128_params.pkl --n_chips 1 --steps 20 --out_json out/metrics/live_tpu_panda_128_1c.json
"$PY" -u bench.py --scene panda_pick_cube --batches 1024 --res 128 --mode tiled --neural_ckpt out/metrics/M2_spatial_128_params.pkl --n_chips 4 --steps 20 --out_json out/metrics/live_tpu_panda_128_4c.json

echo "=== [4/4] Phase 3 Zero-Copy Vision PPO on Real TPU v5p (B=512) ==="
"$PY" -u scripts/run_vision_ppo.py --env cartpole --num_envs 512 --rollout_len 16 --num_iters 40 --res 64 --out_json out/metrics/ppo_cartpole.json
"$PY" -u scripts/run_vision_ppo.py --env panda_pick_cube --num_envs 512 --rollout_len 16 --num_iters 40 --res 64 --neural_ckpt out/metrics/M2_spatial_128_params.pkl --out_json out/metrics/ppo_panda.json

touch out/metrics/SUITE_DONE
echo "=== ALL TPU v5p LIVE SUITE TASKS COMPLETE ==="
