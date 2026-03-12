#!/usr/bin/env bash
set -euo pipefail

root_dir="$(dirname "$(realpath "$0")")/.."
script_path="$root_dir/experiments/benchmark_baselines.py"

# Start timer
start_time=$(date +%s)

robots=(
    panda
    fetch
)
envs=(
    table
    cage
    shelf
)

ik="neighbor"
planner="RRTConnect"
n_neighbors=1000
overwrite_results=true

if [[ "${1:-}" == "--overwrite" ]]; then
  overwrite_results=true
fi

for robot in "${robots[@]}"; do
  for env in "${envs[@]}"; do
    echo "=== Running baseline: ${robot} in ${env} ==="

    if [ "$overwrite_results" = true ]; then
      python3 "$script_path" \
        --robot "$robot" \
        --env "$env" \
        --ik "$ik" \
        --planner "$planner" \
        --n_neighbors "$n_neighbors" \
        --overwrite
    else
      python3 "$script_path" \
        --robot "$robot" \
        --env "$env" \
        --ik "$ik" \
        --planner "$planner" \
        --n_neighbors "$n_neighbors"
    fi

    echo "Finished: robot=${robot}, env=${env}"
  done
done

end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
printf "=== All baseline jobs completed in %02d:%02d:%02d ===\n" \
  $((elapsed/3600)) $((elapsed%3600/60)) $((elapsed%60))