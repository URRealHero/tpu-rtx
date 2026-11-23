#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" != "--go" ]]; then
  echo "Refusing to touch TPU hosts without --go (wait for UMP ROOT GO signal)." >&2
  exit 2
fi

REPO_DIR="/Users/zaychik/Desktop/Projects/UMP/research/tpu-mjx-renderer"
OUT_LOCAL="/Users/zaychik/Desktop/Projects/UMP/research/tpu-mjx-renderer/out/videos"
SSH_KEY="${HOME}/.ssh/google_compute_engine"
PY_BIN="/home/zaychik/ump-v5p-64-1-env/train_venv/bin/python"
SHM_BASE="/dev/shm/zaychik-tpu-rtx"
TMP_TGZ="/tmp/tpu_rtx_showcase_$$.tgz"

HOSTS=(
  "136.79.194.202"
  "34.186.255.126"
  "34.162.179.74"
  "34.186.229.150"
  "34.186.244.145"
  "136.79.141.193"
  "136.79.173.204"
  "136.79.177.210"
)

WORKERS=(
  "experiments/showcase_videos/worker_w0_4dgs_robowild.py"
  "experiments/showcase_videos/worker_w1_cloth_robotwin.py"
  "experiments/showcase_videos/worker_w2_mipnerf_garden.py"
  "experiments/showcase_videos/worker_w3_mipnerf_bicycle_bonsai.py"
  "experiments/showcase_videos/worker_w4_interactive_relight.py"
  "experiments/showcase_videos/worker_w5_diffphys_bokeh.py"
  "experiments/showcase_videos/worker_w6_menagerie_turntable.py"
  "experiments/showcase_videos/worker_w7_wave_optics_cinema.py"
)

HOSTS_TOUCHED=0
JOB_PIDS=()

ssh_host() {
  local ip="$1"
  shift
  ssh -n -i "${SSH_KEY}" \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -o ConnectTimeout=15 \
    -o LogLevel=ERROR \
    "zaychik@${ip}" "$@"
}

scrub_host() {
  local idx="$1"
  local ip="${HOSTS[$idx]}"
  ssh_host "${ip}" "
    unset CLOUDSDK_CORE_ACCOUNT || true
    if [ -f '${SHM_BASE}/worker.pid' ]; then
      wpid=\$(cat '${SHM_BASE}/worker.pid' 2>/dev/null || true)
      if [ -n \"\${wpid}\" ] && kill -0 \"\${wpid}\" 2>/dev/null; then
        kill -TERM \"\${wpid}\" 2>/dev/null || true
        sleep 1
        kill -KILL \"\${wpid}\" 2>/dev/null || true
      fi
    fi
    rm -rf '${SHM_BASE}'
    if [ -e /tmp/libtpu_lockfile ] && [ \"\$(stat -c '%U' /tmp/libtpu_lockfile 2>/dev/null || echo '')\" = 'zaychik' ]; then
      rm -f /tmp/libtpu_lockfile || true
    fi
  " >/dev/null 2>&1 || true
}

scrub_all_hosts() {
  local pids=()
  local idx
  for idx in "${!HOSTS[@]}"; do
    scrub_host "${idx}" &
    pids+=("$!")
  done
  local pid
  for pid in "${pids[@]}"; do
    wait "${pid}" || true
  done
}

cleanup() {
  local rc=$?
  set +e
  local pid
  for pid in "${JOB_PIDS[@]:-}"; do
    if [ -n "${pid}" ] && kill -0 "${pid}" 2>/dev/null; then
      kill -TERM "${pid}" 2>/dev/null || true
    fi
  done
  rm -f "${TMP_TGZ}"
  if [ "${HOSTS_TOUCHED}" -eq 1 ]; then
    echo "[cleanup] Scrubbing all 8 TPU hosts back to IDLE..." >&2
    scrub_all_hosts
    verify_hosts_idle || true
  fi
  exit "${rc}"
}
trap cleanup EXIT

preflight_check_host() {
  local idx="$1"
  local ip="${HOSTS[$idx]}"
  local out
  out=$(ssh_host "${ip}" "
    unset CLOUDSDK_CORE_ACCOUNT || true
    vf=\$(sudo lsof -w /dev/vfio/* 2>/dev/null | wc -l || echo 0)
    lk=0
    [ -e /tmp/libtpu_lockfile ] && lk=1
    echo \"VF=\${vf} LOCK=\${lk}\"
  ")
  echo "[preflight w${idx}] ${ip}: ${out}"
  if [[ "${out}" != *"VF=0 LOCK=0"* ]]; then
    echo "[ERROR] Host w${idx} (${ip}) not clean: ${out}" >&2
    return 1
  fi
}

verify_hosts_idle() {
  local idx ip out clean=0
  for idx in "${!HOSTS[@]}"; do
    ip="${HOSTS[$idx]}"
    out=$(ssh_host "${ip}" "
      unset CLOUDSDK_CORE_ACCOUNT || true
      vf=\$(sudo lsof -w /dev/vfio/* 2>/dev/null | wc -l || echo 0)
      lk=0
      [ -e /tmp/libtpu_lockfile ] && lk=1
      shm=0
      [ -d '${SHM_BASE}' ] && shm=1
      echo \"VF=\${vf} LOCK=\${lk} SHM=\${shm}\"
    " || echo "SSH_ERR")
    echo "[verify w${idx}] ${ip}: ${out}"
    if [[ "${out}" == *"VF=0 LOCK=0 SHM=0"* ]]; then
      clean=$((clean + 1))
    fi
  done
  echo "[verify] Clean IDLE hosts: ${clean}/8"
  [[ "${clean}" -eq 8 ]]
}

deploy_and_run_host() {
  local idx="$1"
  local ip="${HOSTS[$idx]}"
  local worker="${WORKERS[$idx]}"
  local log_local="${OUT_LOCAL}/w${idx}.log"

  scp -i "${SSH_KEY}" \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -o ConnectTimeout=15 \
    -o LogLevel=ERROR \
    "${TMP_TGZ}" "zaychik@${ip}:/dev/shm/tpu_rtx_showcase_w${idx}.tgz"

  local ssh_rc=0
  ssh_host "${ip}" "
    set -euo pipefail
    unset CLOUDSDK_CORE_ACCOUNT || true
    unset JAX_COORDINATOR_ADDRESS JAX_PROCESS_COUNT JAX_PROCESS_INDEX JAX_LOCAL_DEVICE_IDS || true
    unset TPU_CHIPS_PER_HOST_BOUNDS TPU_HOST_BOUNDS CLOUD_TPU_TASK_ID || true

    rm -rf '${SHM_BASE}/repo' '${SHM_BASE}/out_videos' '${SHM_BASE}/worker.pid'
    mkdir -p '${SHM_BASE}/repo' '${SHM_BASE}/tmp' '${SHM_BASE}/logs' '${SHM_BASE}/jax_cache' '${SHM_BASE}/xdg' '${SHM_BASE}/mpl' '${SHM_BASE}/pkgs' '${SHM_BASE}/out_videos'
    tar -xzf '/dev/shm/tpu_rtx_showcase_w${idx}.tgz' -C '${SHM_BASE}/repo'
    rm -f '/dev/shm/tpu_rtx_showcase_w${idx}.tgz'

    for v in \$(env | awk -F= '/^MEGASCALE_/ {print \$1}'); do
      unset \"\${v}\" || true
    done
    export TPU_SKIP_MDS_QUERY=1
    export TPU_HOST_BOUNDS=1,1,1
    export TPU_CHIPS_PER_HOST_BOUNDS=2,2,1
    export TPU_VISIBLE_DEVICES=0,1,2,3
    export TPU_PROCESS_ADDRESSES=127.0.0.1:8476
    export TPU_PROCESS_PORT=8476
    export TPU_WORKER_ID=0
    export CLOUD_TPU_TASK_ID=0

    export TMPDIR='${SHM_BASE}/tmp'
    export TEMP='${SHM_BASE}/tmp'
    export TMP='${SHM_BASE}/tmp'
    export TPU_LOG_DIR='${SHM_BASE}/logs'
    export JAX_COMPILATION_CACHE_DIR='${SHM_BASE}/jax_cache'
    export XDG_CACHE_HOME='${SHM_BASE}/xdg'
    export MPLCONFIGDIR='${SHM_BASE}/mpl'
    export PYTHONPATH='${SHM_BASE}/repo:${SHM_BASE}/pkgs'

    need_pkgs=()
    for mod_pkg in 'mujoco:mujoco' 'mujoco.mjx:mujoco-mjx' 'PIL:Pillow' 'optax:optax' 'trimesh:trimesh' 'etils:etils'; do
      mod=\${mod_pkg%%:*}
      pkg=\${mod_pkg##*:}
      if ! '${PY_BIN}' -c \"import \${mod}\" >/dev/null 2>&1; then
        need_pkgs+=(\"\${pkg}\")
      fi
    done
    if [ \${#need_pkgs[@]} -gt 0 ]; then
      '${PY_BIN}' -m pip install --quiet --no-cache-dir --target='${SHM_BASE}/pkgs' \"\${need_pkgs[@]}\"
      rm -rf '${SHM_BASE}/pkgs'/jax* '${SHM_BASE}/pkgs'/libtpu* '${SHM_BASE}/pkgs'/numpy* '${SHM_BASE}/pkgs'/scipy* '${SHM_BASE}/pkgs'/ml_dtypes*
    fi

    cd '${SHM_BASE}/repo'
    timeout 780 '${PY_BIN}' '${worker}' --out_dir '${SHM_BASE}/out_videos' &
    wpid=\$!
    echo \"\${wpid}\" > '${SHM_BASE}/worker.pid'
    wait \"\${wpid}\"
    rm -f '${SHM_BASE}/worker.pid'
  " > "${log_local}" 2>&1 || ssh_rc=$?

  echo "[harvest w${idx}] Downloading rendered .tar archives from ${ip} (rc=${ssh_rc})..."
  scp -i "${SSH_KEY}" \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -o ConnectTimeout=15 \
    -o LogLevel=ERROR \
    "zaychik@${ip}:${SHM_BASE}/out_videos/*.tar" "${OUT_LOCAL}/" >/dev/null 2>&1 || true
  echo "[done w${idx}] Finished ${worker} (rc=${ssh_rc})"
  return "${ssh_rc}"
}

mkdir -p "${OUT_LOCAL}"

echo "[stage 1/4] Preflight check across 8 hosts..."
for idx in "${!HOSTS[@]}"; do
  preflight_check_host "${idx}"
done

echo "[stage 2/4] Packing lightweight repo tarball (code + out/assets only)..."
COPYFILE_DISABLE=1 tar -czf "${TMP_TGZ}" \
  -C "${REPO_DIR}" \
  --exclude='.git' \
  --exclude='docs' \
  --exclude='out/data' \
  --exclude='out/figures' \
  --exclude='out/metrics' \
  --exclude='out/renders' \
  --exclude='out/hlo' \
  --exclude='out/videos' \
  --exclude='experiments/w1_tile_cull' \
  --exclude='**/__pycache__' \
  .

HOSTS_TOUCHED=1

echo "[stage 3/4] Launching 8 parallel workers (w0..w7)..."
for idx in "${!HOSTS[@]}"; do
  deploy_and_run_host "${idx}" &
  JOB_PIDS+=("$!")
done

FAIL_COUNT=0
for idx in "${!JOB_PIDS[@]}"; do
  if ! wait "${JOB_PIDS[$idx]}"; then
    echo "[ERROR] Worker w${idx} (${WORKERS[$idx]}) failed; inspect ${OUT_LOCAL}/w${idx}.log" >&2
    FAIL_COUNT=$((FAIL_COUNT + 1))
  fi
done
JOB_PIDS=()

echo "[stage 4/4] Scrubbing all 8 hosts and verifying clean release..."
scrub_all_hosts
HOSTS_TOUCHED=0
verify_hosts_idle

if [ "${FAIL_COUNT}" -ne 0 ]; then
  echo "[FAIL] ${FAIL_COUNT} worker(s) failed." >&2
  exit 1
fi
echo "[SUCCESS] All 8 workers completed and harvested to ${OUT_LOCAL}"
