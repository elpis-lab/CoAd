#!/usr/bin/env bash
set -euo pipefail

PYTHON=python3
SCRIPT="experiments/benchmark_baselines.py"

robots=(panda fetch)
envs=(table cage shelf)
ik="neighbor"
planner="RRTConnect"
n_neighbors=1000

OVERWRITE=false
if [[ "${1:-}" == "--overwrite" ]]; then
  OVERWRITE=true
fi

for r in "${robots[@]}"; do
  for e in "${envs[@]}"; do

    out="data/baseline_results_${r}_${e}.npz"

    if [[ -f "$out" && "$OVERWRITE" == "false" ]]; then
      echo "[Skip] $out exists. (Pass --overwrite to re-run)"
      continue
    fi

    echo "Running: robot=$r env=$e -> $out"
    cmd=(
      $PYTHON "$SCRIPT"
      --robot "$r"
      --env "$e"
      --ik "$ik"
      --planner "$planner"
      --n_neighbors "$n_neighbors"
    )

    if [[ "$OVERWRITE" == "true" ]]; then
      cmd+=(--overwrite)
    fi

    "${cmd[@]}"
  done
done