\#!/usr/bin/env bash
set -euo pipefail
root_dir="$(dirname "$(realpath "$0")")/.."
script_dir="$root_dir/coad"

# Start timer
start_time=$(date +%s)

robots=(
    #panda
    #ur10
    fetch
)
envs=(
    table
    cage
    shelf
)
adaptations=(
    grr
    opt
    dmp
)
adaptation="opt"
ik="neighbor"
planner="RRTConnect"
overwrite_task_set=false
overwrite_joint_goal_set=false
overwrite_task_paths=true
overwrite_condensed_graph=true

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

    # 3 Generate task path set
    if [ "$overwrite_task_paths" = true ]; then
      python "$script_dir/generate_task_paths.py" \
        --robot "$robot" \
        --env "$env" \
        --ik "$ik" \
        --planner "$planner" \
        --overwrite
    else
      python "$script_dir/generate_task_paths.py" \
        --robot "$robot" \
        --env "$env" \
        --ik "$ik" \
        --planner "$planner"
    fi

    # 4 Condense for each adaptation
    for adaptation in "${adaptations[@]}"; do
      echo "=== Condensing: ${robot} in ${env}, adaptation=${adaptation} ==="

      if [ "$overwrite_condensed_graph" = true ]; then
        python "$script_dir/condense_task_paths.py" \
          --robot "$robot" \
          --env "$env" \
          --ik "$ik" \
          --planner "$planner" \
          --adaptation "$adaptation" \
          --overwrite
      else
        python "$script_dir/condense_task_paths.py" \
          --robot "$robot" \
          --env "$env" \
          --ik "$ik" \
          --planner "$planner" \
          --adaptation "$adaptation"
      fi

      echo "Finished adaptation=${adaptation}"
    done

  done
done

end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
printf "=== All planning jobs completed in %02d:%02d:%02d ===\n" \
  $((elapsed/3600)) $((elapsed%3600/60)) $((elapsed%60))
