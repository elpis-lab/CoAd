# Experiments

The entry points below share existing planning and adaptation implementations.
The conveyor runner adds timed physical execution to the static benchmarks. Implementations are
shared rather than copied into each script.

| Task | Entry point |
| --- | --- |
| Adaptation performance and compression | `benchmark_adaptations.py` |
| Random-problem planning, including baselines | `benchmark_planning.py` |
| Adaptation/compression table | `summarize_adaptations.py` |
| Planning table | `summarize_planning.py` |
| Planning quality and time plots | `plot_planning.py` |
| Moving conveyor pick-and-sort experiment | `run_conveyor_experiment.py` |
| Environment viewer / transparent screenshot | `visualize_env.py` |
| Live path comparison / saved-path replay | `visualize_paths.py` |

## Benchmarks

```bash
python experiments/benchmark_adaptations.py --robot panda --env table --methods grr opt dmp
python experiments/benchmark_planning.py --robot fetch --env conveyor --methods rrtc vamp lightning ertconnect --samples 100 --timeout 3 --save-paths
```

Planning methods: `full`, `grr`, `opt`, `dmp`, `linear`, `rrtc`, `prmstar`,
`vamp`, `lightning`, `ertconnect`. Adaptation benchmarking supports the first five.
Choose only methods whose libraries/dependencies you have installed. ERTConnect
requires OMPL bindings exposing ERTConnect; VAMP is only required when selected.

`--planner` identifies the **saved dataset's planner**, while `--methods` selects
what to benchmark. `--ik`, `--n-neighbors` (also `--n_neighbors`), and `--data-root`
select files such as `root_paths_neighbor_RRTConnect_grr_1000.pkl` under
`data/table_panda/`. The default reference library is GRR; change it with
`--reference-method`. Baseline queries still need a reference map for saved regions
and goal configurations. `full` additionally loads the original task-path files.

Adaptation benchmarking samples once per usable reference region by default;
`--samples` limits the number of regions. Planning draws random poses from the
reference regions and rejects queries with colliding/out-of-bounds starts or goals.
The query set is saved and reused when adding methods to the same result file.

Lightning and ERTConnect share the sampled experience library and nearest-neighbor
lookup. The default size is the largest compressed library for the selected
dataset, capped at 50,000 for microwave scenes. Override with `--library-size`.
`--library-k` controls Lightning's neighbor count. Paths are stored by reference
until needed, avoiding copies of every trajectory.

Online times include retrieval/planning and path extraction. Library loading,
index construction, native experience conversion, and independent validation are
excluded. `validation_seconds` is saved separately. VAMP retains its configured
iteration budget; `--timeout` applies to the OMPL/experience planners. Adapters
retain their stored (possibly IK-refined) joint goals; baselines target the reference
library's joint goal. All methods are independently checked for collisions, joint
bounds, and endpoints. Old files may have counted success differently.

Outputs retain the familiar `adaptation_results_ROBOT_ENV.npz` and
`baseline_results_ROBOT_ENV.npz` filenames. New files contain `metadata`, `queries`,
`home`, and per-method arrays under `methods`. One success/time/length record is
written for every query, including failures. Completed methods are saved atomically.
`--overwrite` reruns selected methods in a matching configuration. With a different
configuration it replaces the result file; use `--output` to keep multiple runs.
Legacy files must be explicitly overwritten or a new output selected.

## Two shell scripts

Edit the arrays directly in `run_adaptations.sh` and `run_planning.sh`:

```bash
robots=(panda fetch)
envs=(table cage shelf)
methods=(grr opt dmp)
```

Each script loops over the selected robots and environments. Other settings, such
as planner, neighbors, samples, and timeout, are ordinary variables in the script.

```bash
bash experiments/run_adaptations.sh
bash experiments/run_planning.sh --save-paths
```

In `run_adaptations.sh`, set `generate_data=true` to generate tasks, goals, and full
paths, and `compress=true` to generate compressed libraries before evaluation.
Compression uses the serial global algorithm; worker count only affects the
parallel goal/path stages. `overwrite_data=true` regenerates offline artifacts;
`--overwrite` passed to the shell script applies to benchmark results.

## Tables and plots

```bash
python experiments/summarize_adaptations.py --robots panda fetch --envs table cage --output results/adaptations.csv
python experiments/summarize_planning.py --robots panda fetch --output results/planning.csv
python experiments/plot_planning.py --robots panda fetch --envs table cage --output-dir plots
```

Both summaries print tables and optionally write CSV. They also read historical
NPZ layouts. Missing historical compression counts are shown as N/A. For new runs,
compression is `100 * (1 - root_count / usable_mapped_task_count)`.

Plotting writes **four files**: `planning_time.pdf`, `planning_time.png`,
`planning_quality.pdf`, and `planning_quality.png`. Time is logarithmic by default;
use `--linear-time` to change this. Both metrics use successful trials only.
`--methods`, `--files`, and `--data-root` also work for summaries and plots.
No LaTeX or specific font installation is required; `--font` selects a custom font.

## Visualization

```bash
python experiments/visualize_env.py --env shelf --robot fetch
python experiments/visualize_env.py --env conveyor --robot none
MUJOCO_GL=egl python experiments/visualize_env.py --env conveyor --robot panda --output images/conveyor.png --azimuth -130 --elevation -40 --distance 2.85 --lookat .4 1 1.1
python experiments/visualize_paths.py --env table --robot panda --methods grr dmp rrtc --query 0
python experiments/visualize_paths.py --results data/baseline_results_panda_table.npz --methods rrtc lightning --query 3
```

`visualize_env` opens MuJoCo by default. `--output` saves a transparent RGBA PNG;
add `--show` to also open the viewer. Use `--width`, `--height`, `--keep-floor`, and
`--environment-color R G B A` to customize screenshots. Transparency comes from
segmentation rather than removing white pixels. Camera arguments also work in
`visualize_paths`.

All environment/robot visualization combinations are accepted. Named, configured
pairs use their planning scene and home pose. UR10/conveyor and non-UR10/lab views
use the static scene with an uncalibrated robot placement, adjustable with
`--base-position X Y Z` and `--base-quat W X Y Z`. `--robot none` loads the static
scene. `--scene path/to/scene.yaml` accepts custom collision-object scenes.

Saved-path replay requires benchmarking with `--save-paths`. Without `--results`,
the path viewer samples a valid problem and runs the selected methods. There are
no blocking console prompts. Use `--loop`, `--playback-dt`, and `--pause` to control
playback; closing the viewer stops it.

## Shared implementation

- `benchmark.py`: shared benchmark setup, query sampling, method execution, validation.
- `common.py`: CLI, dataset loading, and result serialization.
- `indexing.py`: task-region lookup and sampling, including door and contact-face dimensions.
- `experience_library.py`: shared Lightning/ERTConnect experience indexing and Lightning repair.
- `reporting.py`: shared result discovery and table formatting.

`coad.env` continues exporting the same environment classes. Their implementations
are organized under `coad/environments`: `base.py` compiles models in memory,
`objects.py` handles object/scene geometry and motion, `task_regions.py` handles
TCR construction, and `swept_volume.py` handles swept meshes. Scene-specific modules
hold the environment classes. `compute_tcr=False` now works consistently when
`using_swept_volume=False`, which keeps evaluation and visualization startup light.

Regression tests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q experiments/tests
```
