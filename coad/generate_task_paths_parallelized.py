import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import multiprocessing as mp
import pickle
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

from coad.utils import set_seed, load_env_and_robot, get_data_folder
from coad.planning import OMPLPlanner, VAMPPlanner, euclidean_path_length


_WORKER_ENV = None
_WORKER_ROBOT = None
_WORKER_HOME_QPOS = None
_WORKER_PLANNER = None
_WORKER_FALLBACK_PLANNER = None
_WORKER_PLANNER_NAME = None


def _initialize_library_worker(
    env_name,
    robot_name,
    planner_name,
    worker_seed,
    batch_time_budget,
    progress_lock,
):
    """Create process-local MuJoCo and planner objects once per worker."""
    global _WORKER_ENV
    global _WORKER_ROBOT
    global _WORKER_HOME_QPOS
    global _WORKER_PLANNER
    global _WORKER_FALLBACK_PLANNER
    global _WORKER_PLANNER_NAME

    # All tqdm instances, including those in separate processes, use the
    # same lock so their terminal output does not overwrite one another.
    tqdm.set_lock(progress_lock)

    process_identity = mp.current_process()._identity
    process_index = process_identity[0] if process_identity else 0
    set_seed(worker_seed + process_index)

    # Each process must own its MuJoCo model/data and planner.
    _WORKER_ENV, _WORKER_ROBOT = load_env_and_robot(
        env_name,
        robot_name,
        visualize=False,
    )
    _WORKER_HOME_QPOS = _WORKER_ROBOT.get_joint_qpos().copy()
    _WORKER_PLANNER_NAME = planner_name

    if planner_name == "VAMP":
        _WORKER_PLANNER = VAMPPlanner(
            _WORKER_ROBOT,
            _WORKER_ENV,
            _WORKER_ROBOT.data,
            robot_name,
        )

        _WORKER_FALLBACK_PLANNER = OMPLPlanner(
            _WORKER_ROBOT,
            _WORKER_ROBOT.data,
            planner="RRTConnect",
        )

    elif "PRM" in planner_name:
        _WORKER_PLANNER = OMPLPlanner(
            _WORKER_ROBOT,
            _WORKER_ROBOT.data,
            planner=planner_name,
        )

        # Each worker owns a separate roadmap and reuses it for its chunk.
        _WORKER_PLANNER.construct_roadmap(
            _WORKER_HOME_QPOS,
            timeout=batch_time_budget,
        )

        _WORKER_FALLBACK_PLANNER = OMPLPlanner(
            _WORKER_ROBOT,
            _WORKER_ROBOT.data,
            planner="RRTConnect",
        )

    else:
        _WORKER_PLANNER = OMPLPlanner(
            _WORKER_ROBOT,
            _WORKER_ROBOT.data,
            planner=planner_name,
        )
        _WORKER_FALLBACK_PLANNER = None


def _solve_individual_key(index, key, ik_goal):
    """
    Solve one task with RRTConnect or VAMP.

    Result columns match the original solve_individual():
        [plan_success, solve_time, total_plan_time, path_length]
    """
    result = np.array(
        [False, np.nan, np.nan, np.nan],
        dtype=float,
    )

    if ik_goal is None:
        return index, key, None, result, None

    try:
        _WORKER_ENV.move_swept_volume(key)

        if _WORKER_PLANNER_NAME == "VAMP":
            # Measure the complete VAMP call duration.
            t0 = time.perf_counter()

            path, vamp_planning_time, _ = _WORKER_PLANNER.plan(
                start=_WORKER_HOME_QPOS,
                goal=ik_goal,
                smooth_path=True,
                num_waypoints=200,
                benchmark=True,
            )

            vamp_total_time = time.perf_counter() - t0

            planning_time = vamp_planning_time
            total_time = vamp_total_time
            fallback_used = False

            if path is None or len(path) == 0:
                fallback_used = True

                (
                    fallback_path,
                    fallback_total_time,
                    fallback_planning_time,
                ) = _WORKER_FALLBACK_PLANNER.plan(
                    start=_WORKER_HOME_QPOS,
                    goal=ik_goal,
                    timeout=3.0,
                    num_waypoints=200,
                    benchmark=True,
                )

                path = fallback_path

                # Include both the failed VAMP attempt and fallback attempt.
                planning_time = (
                    vamp_planning_time + fallback_planning_time
                )
                total_time = (
                    vamp_total_time + fallback_total_time
                )

        else:
            path, total_time, planning_time = _WORKER_PLANNER.plan(
                start=_WORKER_HOME_QPOS,
                goal=ik_goal,
                timeout=3.0,
                num_waypoints=200,
                benchmark=True,
            )

            fallback_used = False

        if path is None or len(path) == 0:
            return index, key, None, result, None

        result[:] = [
            True,
            planning_time,
            total_time,
            euclidean_path_length(path),
        ]
        return index, key, path, result, None

    except Exception:
        return index, key, None, result, traceback.format_exc()


def _solve_batch_key(index, key, ik_goal):
    """
    Solve one task with a process-local PRM roadmap and RRTConnect fallback.

    Result columns match the original solve_batch():
        [
            plan_success,
            batch_plan_success,
            total_plan_time,
            batch_total_time,
            fallback_total_time,
            batch_path_length,
            fallback_path_length,
        ]
    """
    result = np.array(
        [False, False, np.nan, np.nan, np.nan, np.nan, np.nan],
        dtype=float,
    )

    if ik_goal is None:
        return index, key, None, result, None

    try:
        _WORKER_ENV.move_swept_volume(key)

        t0 = time.perf_counter()
        path = _WORKER_PLANNER.graph_query(
            _WORKER_HOME_QPOS,
            ik_goal,
        )
        batch_total_time = time.perf_counter() - t0
        batch_planning_time = batch_total_time

        result[3] = batch_total_time

        if path is not None and len(path) > 0:
            result[:] = [
                True,
                True,
                batch_planning_time,
                batch_total_time,
                np.nan,
                euclidean_path_length(path),
                np.nan,
            ]
            return index, key, path, result, None

        fallback_path, fallback_total_time, fallback_planning_time = (
            _WORKER_FALLBACK_PLANNER.plan(
                start=_WORKER_HOME_QPOS,
                goal=ik_goal,
                timeout=3.0,
                num_waypoints=200,
                benchmark=True,
            )
        )

        result[4] = fallback_total_time

        combined_total_time = batch_total_time + fallback_total_time
        combined_planning_time = batch_planning_time + fallback_planning_time

        if fallback_path is None or len(fallback_path) == 0:
            result[2] = combined_planning_time
            return index, key, None, result, None

        result[:] = [
            True,
            False,
            combined_planning_time,
            batch_total_time,
            fallback_total_time,
            np.nan,
            euclidean_path_length(fallback_path),
        ]
        return index, key, fallback_path, result, None

    except Exception:
        return index, key, None, result, traceback.format_exc()


def _solve_chunk(chunk_id, chunk):
    """Solve one chunk while displaying a dedicated worker progress bar."""
    chunk_results = []
    num_successes = 0

    # position=chunk_id gives each submitted chunk its own terminal row.
    # Since the number of chunks is at most the number of workers, every
    # concurrently executing worker has a distinct progress-bar position.
    with tqdm(
        total=len(chunk),
        desc=f"Worker {chunk_id}",
        position=chunk_id,
        unit="task",
        leave=True,
        dynamic_ncols=True,
        mininterval=0.2,
    ) as worker_bar:
        for index, key, ik_goal in chunk:
            if "PRM" in _WORKER_PLANNER_NAME:
                item_result = _solve_batch_key(index, key, ik_goal)
            else:
                item_result = _solve_individual_key(index, key, ik_goal)

            chunk_results.append(item_result)

            if bool(item_result[3][0]):
                num_successes += 1

            worker_bar.set_postfix(
                solved=f"{num_successes}/{len(chunk_results)}",
                refresh=False,
            )
            worker_bar.update(1)

    return chunk_results


def _split_into_chunks(indexed_tasks, num_chunks):
    """Split tasks into approximately equal contiguous chunks."""
    if not indexed_tasks:
        return []

    num_chunks = max(1, min(num_chunks, len(indexed_tasks)))
    arrays = np.array_split(
        np.arange(len(indexed_tasks)),
        num_chunks,
    )

    return [
        [indexed_tasks[int(i)] for i in indices]
        for indices in arrays
        if len(indices) > 0
    ]


def solve_joint_goal_set_parallel(
    env_name,
    robot_name,
    joint_goal_set,
    planner_name,
    num_workers=2,
    batch_time_budget=30.0,
):
    """Generate the full path library using multiple worker processes."""
    keys = list(joint_goal_set.keys())
    indexed_tasks = [
        (index, key, joint_goal_set[key])
        for index, key in enumerate(keys)
    ]

    task_paths = {key: None for key in keys}

    if "PRM" in planner_name:
        results = np.full((len(keys), 7), np.nan, dtype=float)
        results[:, 0:2] = False
    else:
        results = np.full((len(keys), 4), np.nan, dtype=float)
        results[:, 0] = False

    chunks = _split_into_chunks(indexed_tasks, num_workers)

    print(f"Number of tasks: {len(keys)}")
    print(
        "Number of valid IK goals: "
        f"{sum(goal is not None for goal in joint_goal_set.values())}"
    )
    print(f"Number of workers: {len(chunks)}")
    print(f"Planner: {planner_name}")

    context = mp.get_context("spawn")

    completed_tasks = 0
    successful_tasks = 0

    # A shared lock keeps progress bars from different processes readable.
    progress_lock = context.RLock()
    tqdm.set_lock(progress_lock)

    with ProcessPoolExecutor(
        max_workers=len(chunks),
        mp_context=context,
        initializer=_initialize_library_worker,
        initargs=(
            env_name,
            robot_name,
            planner_name,
            42,
            batch_time_budget,
            progress_lock,
        ),
    ) as executor:
        future_to_chunk_id = {
            executor.submit(_solve_chunk, chunk_id, chunk): chunk_id
            for chunk_id, chunk in enumerate(chunks)
        }

        for future in as_completed(future_to_chunk_id):
            chunk_id = future_to_chunk_id[future]

            try:
                chunk_results = future.result()
            except Exception:
                tqdm.write(
                    f"Worker chunk {chunk_id} failed:\n"
                    f"{traceback.format_exc()}"
                )
                continue

            chunk_successes = 0

            for index, key, path, result_row, error in chunk_results:
                task_paths[key] = path
                results[index] = result_row

                completed_tasks += 1
                if bool(result_row[0]):
                    successful_tasks += 1
                    chunk_successes += 1

                if error is not None:
                    tqdm.write(
                        f"Error while solving key {key}:\n{error}"
                    )

            tqdm.write(
                f"Worker chunk {chunk_id} complete: "
                f"{chunk_successes}/{len(chunk_results)} solved | "
                f"Overall: {successful_tasks}/{completed_tasks} solved"
            )

    return task_paths, results


def main(args):
    """Generate a full dataset of task paths in parallel."""
    folder = get_data_folder(args.env, args.robot)
    suffix = f"{args.ik}_{args.planner}"

    output_data_path = f"{folder}/task_paths_data_{suffix}.npy"
    output_results_path = f"{folder}/task_paths_results_{suffix}.npy"
    output_keys_path = f"{folder}/task_paths_keys_{suffix}.pkl"

    if os.path.exists(output_data_path) and not args.overwrite:
        print(
            f"Task paths already exist at {folder} "
            f"with IK '{args.ik}' and planner '{args.planner}'. "
            "Use --overwrite to regenerate the task paths."
        )
        return

    joint_goal_path = f"{folder}/joint_goal_set_{args.ik}.pkl"

    try:
        with open(joint_goal_path, "rb") as file:
            joint_goal_set = pickle.load(file)
    except FileNotFoundError as error:
        print(error)
        print(
            f"Joint goal set with IK '{args.ik}' not found. "
            "Generate it first with generate_joint_goal_set.py."
        )
        return

    task_paths, results = solve_joint_goal_set_parallel(
        env_name=args.env,
        robot_name=args.robot,
        joint_goal_set=joint_goal_set,
        planner_name=args.planner,
        num_workers=args.num_workers,
        batch_time_budget=args.batch_time_budget,
    )

    # Preserve the original joint_goal_set insertion order.
    keys = list(task_paths.keys())
    data = np.array(
        [task_paths[key] for key in keys],
        dtype=object,
    )

    np.save(output_results_path, results)
    np.save(output_data_path, data)

    with open(output_keys_path, "wb") as file:
        pickle.dump(keys, file)

    print("Full library generation complete.")
    print(f"Saved paths to: {output_data_path}")
    print(f"Saved results to: {output_results_path}")
    print(f"Saved keys to: {output_keys_path}")


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--env",
        choices=[
            "table",
            "box",
            "cage",
            "shelf",
            "free",
            "real",
            "largeobj",
            "microwave",
            "allstable",
        ],
        default="table",
    )
    parser.add_argument(
        "--robot",
        choices=["panda", "ur10", "fetch"],
        default="panda",
    )
    parser.add_argument(
        "--ik",
        choices=["random", "neighbor", "grr"],
        default="neighbor",
    )
    parser.add_argument(
        "--planner",
        choices=["RRTConnect", "PRMstar", "LazyPRMstar", "VAMP"],
        default="RRTConnect",
    )
    parser.add_argument(
        "--num-workers",
        "--num_workers",
        dest="num_workers",
        type=int,
        default=2,
        help="Number of parallel planning worker processes.",
    )
    parser.add_argument(
        "--batch-time-budget",
        type=float,
        default=30.0,
        help=(
            "Roadmap construction time in seconds per worker for PRM planners."
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    set_seed(42)
    main(parse_arguments())
