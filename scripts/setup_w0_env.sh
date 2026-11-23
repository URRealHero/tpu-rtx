#!/bin/bash
set -euo pipefail

# Isolated setup on Cloud TPU v5p host in /tmp/tpu-rtx-pkgs
# Never touches system Python environments or any active directories.
ROOT_DIR="${ROOT_DIR}"
PKG_DIR="/tmp/tpu-rtx-pkgs"
PY="${PYTHON_BIN:-$(command -v python3)}"

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
