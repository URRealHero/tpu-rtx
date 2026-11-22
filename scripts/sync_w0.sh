#!/bin/bash
set -euo pipefail

SRC_DIR="/Users/zaychik/Desktop/Projects/UMP/research/tpu-mjx-renderer"
DST_DIR="/tmp/zaychik-tpu-mjx"
FSSH="python3 -B /Users/zaychik/ump_perf_shared/infra/release/current/tools/ops/ump/bin/fssh.py"
NODE_ID="511895978032286846"
POD="tpu-v5p-64-2"

TAR_B64=$(COPYFILE_DISABLE=1 tar -C "$SRC_DIR" --exclude='out/*' --exclude='__pycache__' -czf - . | base64 | tr -d '\n')

# Send in safe chunks or direct command if small (< 100KB)
LEN=${#TAR_B64}
echo "Syncing $SRC_DIR -> $POD:$DST_DIR (b64 len=$LEN)"
$FSSH "$POD" 0 --mutating --fresh --expected-node-id "$NODE_ID" --zone us-east5-a -- \
  "mkdir -p $DST_DIR && rm -f /tmp/zaychik_tpu_mjx_sync.b64" >/dev/null

CHUNK=40000
OFFSET=0
while [ $OFFSET -lt $LEN ]; do
  PART="${TAR_B64:$OFFSET:$CHUNK}"
  $FSSH "$POD" 0 --mutating --fresh --expected-node-id "$NODE_ID" --zone us-east5-a -- \
    "printf '%s' '$PART' >> /tmp/zaychik_tpu_mjx_sync.b64" >/dev/null
  OFFSET=$((OFFSET + CHUNK))
done

$FSSH "$POD" 0 --mutating --fresh --expected-node-id "$NODE_ID" --zone us-east5-a -- \
  "base64 -d /tmp/zaychik_tpu_mjx_sync.b64 | tar -C $DST_DIR -xzf - && rm -f /tmp/zaychik_tpu_mjx_sync.b64 && ls -la $DST_DIR"
