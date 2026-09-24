#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Edit these lists and settings before running this file.
robots=(panda fetch)
envs=(table cage shelf)
methods=(grr opt dmp)
ik=neighbor
planner=RRTConnect
neighbors=1000
workers=2
generate_data=false
compress=false
overwrite_data=false

data_flags=()
if [[ "$overwrite_data" == true ]]; then
  data_flags=(--overwrite)
fi

for robot in "${robots[@]}"; do
  for env in "${envs[@]}"; do
    if [[ "$generate_data" == true ]]; then
      python3 coad/generate_task_set.py \
        --robot "$robot" --env "$env" "${data_flags[@]}"
      python3 coad/generate_joint_goal_set_parallelized.py \
        --robot "$robot" --env "$env" --ik "$ik" \
        --num_workers "$workers" "${data_flags[@]}"
      python3 coad/generate_task_paths_parallelized.py \
        --robot "$robot" --env "$env" --ik "$ik" --planner "$planner" \
        --num_workers "$workers" "${data_flags[@]}"
    fi

    if [[ "$compress" == true ]]; then
      for method in "${methods[@]}"; do
        if [[ "$method" == full ]]; then continue; fi
        python3 coad/generate_condensed_task_paths.py \
          --robot "$robot" --env "$env" --ik "$ik" --planner "$planner" \
          --adaptation "$method" --n_neighbors "$neighbors" "${data_flags[@]}"
      done
    fi

    python3 experiments/benchmark_adaptations.py \
      --robot "$robot" --env "$env" --ik "$ik" --planner "$planner" \
      --n-neighbors "$neighbors" --methods "${methods[@]}" "$@"
  done
done
