#!/bin/bash
set -euo pipefail

SRC_DIR="${SRC_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
TPU_NAME="${TPU_NAME:?Set TPU_NAME (your Cloud TPU VM name)}"
TPU_ZONE="${TPU_ZONE:-us-east5-a}"
WORKER="${WORKER:-0}"
DST_DIR="${DST_DIR:-/tmp/tpu-rtx}"

echo "Syncing $SRC_DIR -> $TPU_NAME (worker=$WORKER, zone=$TPU_ZONE):$DST_DIR"
gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$TPU_ZONE" --worker="$WORKER" \
  --command="mkdir -p '$DST_DIR'"

COPYFILE_DISABLE=1 tar -C "$SRC_DIR" --exclude='out/*' --exclude='docs/assets/*' --exclude='__pycache__' --exclude='.git' -czf /tmp/tpu_rtx_sync.tar.gz .
gcloud compute tpus tpu-vm scp /tmp/tpu_rtx_sync.tar.gz "$TPU_NAME:/tmp/tpu_rtx_sync.tar.gz" --zone="$TPU_ZONE" --worker="$WORKER"
rm -f /tmp/tpu_rtx_sync.tar.gz

gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$TPU_ZONE" --worker="$WORKER" \
  --command="tar -C '$DST_DIR' -xzf /tmp/tpu_rtx_sync.tar.gz && rm -f /tmp/tpu_rtx_sync.tar.gz && ls -la '$DST_DIR'"
