#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Edit these lists and settings, then run: bash experiments/run_planning.sh
robots=(panda fetch)
envs=(table cage shelf)
methods=(grr opt dmp rrtc vamp lightning ertconnect)
python_bin=python3
ik=neighbor
planner=RRTConnect
neighbors=1000
samples=1000
timeout=3.0

for robot in "${robots[@]}"; do
  for env in "${envs[@]}"; do
    "$python_bin" experiments/benchmark_planning.py \
      --robot "$robot" --env "$env" --ik "$ik" \
      --planner "$planner" --n-neighbors "$neighbors" \
      --samples "$samples" --timeout "$timeout" \
      --methods "${methods[@]}" "$@"
  done
done
