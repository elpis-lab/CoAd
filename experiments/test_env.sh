#!/usr/bin/env bash
set -euo pipefail
root_dir="$(dirname "$(realpath "$0")")/.."
script_dir="$root_dir/coad"
# Start timer
start_time=$(date +%s)

robots=(
    panda
    # fetch
    # ur10
)
envs=(
    table
    # cage
    # shelf
    # free
)
ik = "grr"
overwrite_task_set = true
overwrite_joint_goal_set = true

for robot in "${robots[@]}"; do
  for env in "${envs[@]}"; do
    echo "=== Building environments: ${robot} in ${env} ==="

    # 1 Generate task set
    if [ "$overwrite_task_set" = true ]; then
      python "$script_dir/generate_task_set.py" \
        --robot "$robot" \
        --env "$env" \
        --overwrite
    else
      python "$script_dir/generate_task_set.py" \
        --robot "$robot" \
        --env "$env"
    fi

    # 2 Generate joint goal set
    if [ "$overwrite_joint_goal_set" = true ]; then
      python "$script_dir/generate_joint_goal_set.py" \
        --robot "$robot" \
        --env "$env" \
        --ik "$ik" \
        --overwrite
    else
      python "$script_dir/generate_joint_goal_set.py" \
        --robot "$robot" \
        --env "$env" \
        --ik "$ik"
    fi

    echo "Finished building environment"
  done
done

end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
printf "=== All planning jobs completed in %02d:%02d:%02d ===\n" \
    $((elapsed/3600)) $((elapsed%3600/60)) $((elapsed%60))
