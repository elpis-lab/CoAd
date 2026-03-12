\#!/usr/bin/env bash
set -euo pipefail
root_dir="$(dirname "$(realpath "$0")")/.."
script_dir="$root_dir/plan_load"

# Start timer
start_time=$(date +%s)

robots=(
    #panda
    #ur10
    fetch
)
envs=(
    cage
    table
    shelf
)
adaptations=(
    grr
    opt
    dmp
)
ik="neighbor"
planner="RRTConnect"
overwrite_condensed_graph=false

for robot in "${robots[@]}"; do
  for env in "${envs[@]}"; do
    # Condense for each adaptation
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
