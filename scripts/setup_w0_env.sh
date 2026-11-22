#!/bin/bash
set -euo pipefail

# Isolated setup on tpu-v5p-64-2 w0 in /tmp/zaychik-tpu-mjx-pkgs
# Never touches /home/zaychik/ump-v5p-64-1-env or any UMP directories.
ROOT_DIR="/tmp/zaychik-tpu-mjx"
PKG_DIR="/tmp/zaychik-tpu-mjx-pkgs"
PY="/home/zaychik/ump-v5p-64-1-env/train_venv/bin/python"

mkdir -p "$ROOT_DIR" "$PKG_DIR"

if [ ! -f "$PKG_DIR/mujoco/_version.py" ]; then
  "$PY" -m pip install --no-cache-dir --no-deps --upgrade --target "$PKG_DIR" \
    mujoco mujoco-core mujoco-mjx PyOpenGL glfw
fi

PYTHONPATH="$PKG_DIR:$ROOT_DIR" JAX_PLATFORMS=cpu "$PY" -c "
import jax, flax, optax, mujoco
from mujoco import mjx
print('OK jax:', jax.__version__, 'flax:', flax.__version__, 'optax:', optax.__version__, 'mujoco:', mujoco.__version__)
"
