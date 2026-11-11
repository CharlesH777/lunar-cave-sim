#!/usr/bin/env bash
set -euo pipefail
SIM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAAC_ENV="${ISAAC_CAVE_ENV:-/home/charles/miniconda3/envs/env_isaaclab}"
export LD_LIBRARY_PATH="$ISAAC_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_PRELOAD="$ISAAC_ENV/lib/libstdc++.so.6:$ISAAC_ENV/lib/libgcc_s.so.1${LD_PRELOAD:+:$LD_PRELOAD}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-6}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-6}"
exec "$ISAAC_ENV/bin/python" "$SIM_DIR/run_demo.py" "$@"
