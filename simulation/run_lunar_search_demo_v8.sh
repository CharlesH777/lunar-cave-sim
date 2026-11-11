#!/usr/bin/env bash
# Build an independent seeded demo. Rendering is explicitly selected with --render.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
seed="${1:-20261118}"
if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
    echo 'Usage: bash run_lunar_search_demo_v8.sh SEED [--render]' >&2
    exit 2
fi
if [[ $# -gt 2 || ( $# -eq 2 && "$2" != '--render' ) ]]; then
    echo 'Usage: bash run_lunar_search_demo_v8.sh SEED [--render]' >&2
    exit 2
fi
destination="outputs/lunar_search_v8/demo_seed_${seed}"
isaac_python='/home/charles/miniconda3/envs/env_isaaclab/bin/python'
usd_libraries='/home/charles/.local/share/ov/data/exts/v2/omni.usd.libs-4fde11c8f289f1f4'
isaac_libraries='/home/charles/miniconda3/envs/env_isaaclab/lib'
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
PYTHONPATH="$usd_libraries${PYTHONPATH:+:$PYTHONPATH}" LD_LIBRARY_PATH="$isaac_libraries:$usd_libraries/bin" \
"$isaac_python" build_lunar_search_demo_v8_cpu.py --seed "$seed" --output "$destination"
if [[ "${2:-}" == '--render' ]]; then
    LD_LIBRARY_PATH="$isaac_libraries" LD_PRELOAD="$isaac_libraries/libstdc++.so.6:$isaac_libraries/libgcc_s.so.1" \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 \
    "$isaac_python" render_lunar_cave_v4_isaac.py --source "$destination" \
        --output "$destination/isaac_render_01" --width 1600 --subframes 32
fi
