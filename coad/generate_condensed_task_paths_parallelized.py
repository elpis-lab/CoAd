import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import time
import numpy as np
import pickle
from tqdm import tqdm

from coad.utils import set_seed, load_env_and_robot, get_data_folder
from coad.task_space import build_task_nn, key_to_center, split_key, has_contact_face
from coad.env import MujocoEnv
from coad.robot import MujocoRobot
from coad.mink_ik import get_ik_solver

from coad.adaptation import LinearAdapter, GRRAdapter
from coad.adaptation import DMPAdapter, TrajOptAdapter

from coad.planning import OMPLPlanner

from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import multiprocessing as mp
import traceback

_WORKER_ENV = None
_WORKER_ROBOT = None
_WORKER_ADAPTER = None
_WORKER_PLANNER = None
_WORKER_FALLBACK_PLANNER = None
_WORKER_HOME_QPOS = None
_WORKER_JOINT_GOAL_SET = None
_WORKER_PLANNER_NAME = None

def _initialize_condensation_worker(
    env_name,
    robot_name,
    planner_name,
    adaptation_name,
    joint_goal_set,
    worker_seed,
):
    """
    Initialize process-local MuJoCo, IK, adaptation, and planning objects.

    This function runs once inside each worker process.
    """
    global _WORKER_ENV
    global _WORKER_ROBOT
    global _WORKER_ADAPTER
    global _WORKER_PLANNER
    global _WORKER_FALLBACK_PLANNER
    global _WORKER_HOME_QPOS
    global _WORKER_JOINT_GOAL_SET
    global _WORKER_PLANNER_NAME

    # Different random stream in each process.
    process_identity = mp.current_process()._identity
    process_index = process_identity[0] if process_identity else 0
    set_seed(worker_seed + process_index)

    # Viewer must be disabled in worker processes.
    _WORKER_ENV, _WORKER_ROBOT = load_env_and_robot(
        env_name,
        robot_name,
        False,
    )

    _WORKER_HOME_QPOS = _WORKER_ROBOT.get_joint_qpos().copy()
    _WORKER_JOINT_GOAL_SET = joint_goal_set
    _WORKER_PLANNER_NAME = planner_name

    ik_solver = get_ik_solver(
        _WORKER_ROBOT,
        env_collision_geoms=_WORKER_ENV.env_details["collision_geoms"],
    )

    if adaptation_name == "linear":
        _WORKER_ADAPTER = LinearAdapter(_WORKER_ROBOT, ik_solver)
    elif adaptation_name == "grr":
        _WORKER_ADAPTER = GRRAdapter(_WORKER_ROBOT, ik_solver)
    elif adaptation_name == "dmp":
        _WORKER_ADAPTER = DMPAdapter(_WORKER_ROBOT, ik_solver)
    elif adaptation_name == "opt":
        _WORKER_ADAPTER = TrajOptAdapter(_WORKER_ROBOT, ik_solver)
    else:
        raise ValueError(
            f"Invalid adaptation method: {adaptation_name}"
        )

    _WORKER_PLANNER = OMPLPlanner(
        _WORKER_ROBOT,
        _WORKER_ROBOT.data,
        planner=planner_name,
    )

    if planner_name == "PRMstar":
        _WORKER_PLANNER.construct_roadmap(
            _WORKER_HOME_QPOS,
            timeout=60,
        )

        _WORKER_FALLBACK_PLANNER = OMPLPlanner(
            _WORKER_ROBOT,
            _WORKER_ROBOT.data,
            planner="RRTConnect",
        )
    else:
        _WORKER_FALLBACK_PLANNER = None

def _evaluate_root_candidate(center_key, candidate_neighbor_keys):
    """
    Plan a path for center_key and test whether its nearby keys can be
    compressed into that path.

    The returned coverage is only a proposal. The parent process decides
    which assignments are accepted.
    """
    global _WORKER_ENV
    global _WORKER_ADAPTER
    global _WORKER_PLANNER
    global _WORKER_FALLBACK_PLANNER
    global _WORKER_HOME_QPOS
    global _WORKER_JOINT_GOAL_SET
    global _WORKER_PLANNER_NAME

    result = {
        "center_key": center_key,
        "success": False,
        "adapted_center": None,
        "center_q_end": None,
        "build_center_time": np.nan,
        "covered_neighbors": [],
        "error": None,
    }

    try:
        center_ik = _WORKER_JOINT_GOAL_SET[center_key].copy()
        _WORKER_ENV.move_swept_volume(center_key)

        if _WORKER_PLANNER_NAME == "RRTConnect":
            center_path, _, _ = _WORKER_PLANNER.plan(
                start=_WORKER_HOME_QPOS,
                goal=center_ik,
                timeout=3.0,
                num_waypoints=200,
                benchmark=True,
            )
        else:
            center_path = _WORKER_PLANNER.graph_query(
                start=_WORKER_HOME_QPOS,
                goal=center_ik,
                num_waypoints=200,
            )

            if center_path is None or len(center_path) == 0:
                center_path, _, _ = _WORKER_FALLBACK_PLANNER.plan(
                    start=_WORKER_HOME_QPOS,
                    goal=center_ik,
                    timeout=3.0,
                    num_waypoints=200,
                    benchmark=True,
                )

        if center_path is None or len(center_path) == 0:
            return result

        t0 = time.perf_counter()
        adapted_center, center_q_end = _WORKER_ADAPTER.build_center(
            center_path
        )
        build_time = time.perf_counter() - t0

        covered_neighbors = []

        for nb_key in candidate_neighbor_keys:
            nb_goal = _WORKER_JOINT_GOAL_SET[nb_key].copy()

            _WORKER_ENV.move_swept_volume(nb_key)

            t0 = time.perf_counter()
            valid, q_nb_end = _WORKER_ADAPTER.compress(
                adapted_center,
                nb_goal,
            )
            compression_time = time.perf_counter() - t0

            if valid:
                covered_neighbors.append(
                    {
                        "key": nb_key,
                        "q_end": q_nb_end,
                        "compression_time": compression_time,
                    }
                )

        result.update(
            {
                "success": True,
                "adapted_center": adapted_center,
                "center_q_end": center_q_end,
                "build_center_time": build_time,
                "covered_neighbors": covered_neighbors,
            }
        )

        return result

    except Exception:
        result["error"] = traceback.format_exc()
        return result
    
def _get_neighbor_keys(
    center_key,
    n_neighbors,
    env_has_contact_face,
    nn,
    keys,
    nn_by_face=None,
    keys_by_face=None,
):
    if env_has_contact_face:
        face, numeric_key = split_key(center_key)
        face_nn = nn_by_face[face]

        key_arr = np.asarray(numeric_key)
        key_center = (key_arr[:, 0] + key_arr[:, 1]) / 2

        num_available = len(keys_by_face[face])
        k = min(n_neighbors + 1, num_available)

        neighbor_indices = face_nn.query(
            [key_center],
            k=k,
            return_distance=False,
            sort_results=True,
        )[0]

        neighbor_keys = []

        for nb_idx in neighbor_indices:
            numeric_nb_key = keys_by_face[face][int(nb_idx)]

            # face_task_set was created using key[1:], so reconstruct the
            # complete key here.
            nb_key = (face,) + tuple(numeric_nb_key)
            neighbor_keys.append(nb_key)

        return neighbor_keys

    key_arr = np.asarray(center_key)
    key_center = (key_arr[:, 0] + key_arr[:, 1]) / 2

    k = min(n_neighbors + 1, len(keys))

    neighbor_indices = nn.query(
        [key_center],
        k=k,
        return_distance=False,
        sort_results=True,
    )[0]

    return [keys[int(nb_idx)] for nb_idx in neighbor_indices]

def solve_joint_goal_set_condensed(
    env: MujocoEnv,
    robot: MujocoRobot,
    joint_goal_set,
    planner,
    adaptation,
    n_neighbors=100,
):
    model, data = robot.model, robot.data
    home_qpos = robot.get_joint_qpos()

    ik_solver = get_ik_solver(robot, env_collision_geoms=env.env_details['collision_geoms'])
    if adaptation == "linear":
        adapter = LinearAdapter(robot, ik_solver)
    elif adaptation == "grr":
        adapter = GRRAdapter(robot, ik_solver)
    elif adaptation == "dmp":
        adapter = DMPAdapter(robot, ik_solver)
    elif adaptation == "opt":
        adapter = TrajOptAdapter(robot, ik_solver)
    else:
        raise ValueError(f"Invalid adaptation method: {adaptation}")

    # Build BallTree for finding nearby neighbors

    if has_contact_face(env):
        nn_by_face = {}
        keys_by_face = {}

        for face in ["xy", "yz", "zx"]:
            face_task_set = {
                key[1:]: joint_goal_set[key]
                for key in joint_goal_set.keys()
                if key[0] == face
            }

            nn, _ = build_task_nn(face_task_set)

            nn_by_face[face] = nn
            keys_by_face[face] = list(face_task_set.keys())
    else:
        nn, _ = build_task_nn(joint_goal_set)

    keys = list(joint_goal_set.keys())
    coverage_success = np.zeros(len(joint_goal_set), dtype=bool)

    # If no solution for a task, we assume it is unsolvable.
    # Get the successfully solved tasks.
    remaining = set(
        [
            key
            for key, val in joint_goal_set.items()
            if val is not None
        ]
    )
    print(f"Number of tasks: {len(joint_goal_set)}")
    print(f"Number of successfully solved tasks: {len(remaining)}")

    # Skip keys with IK failures
    valid_keys = [
        key
        for key, goal in joint_goal_set.items()
        if goal is not None
    ]

    # Result containers
    root_paths = {}  # root_id -> root path
    # key_to_root = {
    #     key: (None, None) for key in joint_goal_set.keys()
    # }  # key -> (root_id, goal_q)
    # build_center_time = {key: np.nan for key in joint_goal_set.keys()}
    # compress_time = {key: np.nan for key in joint_goal_set.keys()}

    key_to_root = {
        key: (None, None) for key in valid_keys
    }
    build_center_time = {
        key: np.nan for key in valid_keys
    }
    compress_time = {
        key: np.nan for key in valid_keys
    }

    # Planner
    ompl_planner = OMPLPlanner(robot, data, planner=planner)

    if (planner == "PRMstar"):
        ompl_planner.construct_roadmap(
            home_qpos,
            timeout=10,
        )
        fallback_planner = OMPLPlanner(robot, data, "RRTConnect")
    else:
        fallback_planner = None

    # Greedy condensation loop
    pbar = tqdm(total=len(remaining))
    
    i = 0
    while remaining:

        # Pick a random center among remaining bins
        center_key = list(remaining)[np.random.randint(0, len(remaining))]
        env.move_swept_volume(center_key)
        center_ik = joint_goal_set[center_key].copy()
        
        if planner == "RRTConnect":
            center_path, total_time, planning_time = ompl_planner.plan(
                start=home_qpos,
                goal=center_ik,
                timeout=3.0,
                num_waypoints=200,
                benchmark=True,
            )
        else:
            center_path = ompl_planner.graph_query(
                start=home_qpos,
                goal=center_ik,
                num_waypoints=200
            )
            if center_path is None or len(center_path) == 0:
                center_path, _, _ = fallback_planner.plan(
                    start=home_qpos,
                    goal=center_ik,
                    timeout=3.0,
                    num_waypoints=200,
                    benchmark=True,
                )
        
        if center_path is None or len(center_path) == 0:
            coverage_success[i] = False
            remaining.remove(center_key)
            pbar.update(1)
            continue

        # Register new root path
        root_id = len(root_paths)
        # build adaptation around the center path
        t0 = time.perf_counter()
        adapted_center, q_end = adapter.build_center(center_path)
        t1 = time.perf_counter()
        build_center_time[center_key] = t1 - t0
        root_paths[root_id] = adapted_center
        key_to_root[center_key] = (root_id, q_end)
        remaining.remove(center_key)

        # Visualization
        if robot.viewer:
            robot.set_joint_qpos(joint_goal_set[center_key])
            robot.viewer.sync()
        pbar.update(1)

        # Query nearest neighbors and
        # Consider those that are still in remaining

        if has_contact_face(env):
            face, numeric_key = split_key(center_key)
            nn = nn_by_face[face]
            key_arr = np.array(numeric_key)

        else:
            key_arr = np.array(center_key)
        
        key_center = (key_arr[:, 0] + key_arr[:, 1]) / 2
        neighbor_indices = nn.query(
            [key_center],
            k=n_neighbors + 1,  # +1 for the center itself
            return_distance=True,
            sort_results=True,
        )[1][0]

        # Try to compress neighbors into this root.
        for nb_idx in neighbor_indices:
            nb_key = keys[nb_idx]
            if nb_key not in remaining or nb_key == center_key:
                continue
            # nb_path = task_paths[nb_key].copy()
            nb_goal = joint_goal_set[nb_key].copy()

            # Set up neighbor environment
            env.move_swept_volume(nb_key)
            # Try to compress neighbor into this root.
            t0 = time.perf_counter()
            valid, q_nb_end = adapter.compress(adapted_center, nb_goal)
            t1 = time.perf_counter()

            if not valid:
                continue

            # Neighbor successfully compressed into this root.
            compress_time[nb_key] = t1 - t0
            key_to_root[nb_key] = (root_id, q_nb_end)
            remaining.remove(nb_key)

            # Visualization
            if robot.viewer:
                robot.set_joint_qpos(q_nb_end)
                robot.viewer.sync()
            pbar.update(1)

        # Update tqdm message periodically
        print_interval = 50
        if len(root_paths) % print_interval == 0:
            tqdm.write(
                f"\nNumber of root paths:{len(root_paths)}."
                + f"\nNumber of completed tasks:{pbar.total - len(remaining)}."
            )

    print("Condensation complete.")
    print(f"Number of root paths: {len(root_paths)}")
    results = np.stack(
        [
            list(build_center_time.values()),
            list(compress_time.values()),
        ],
        axis=1,
    )
    return root_paths, key_to_root, results


def solve_joint_goal_set_condensed_parallel(
    env: MujocoEnv,
    robot: MujocoRobot,
    env_name,
    robot_name,
    joint_goal_set,
    planner,
    adaptation,
    n_neighbors=100,
    num_workers=2,
):
    valid_keys = [
        key
        for key, goal in joint_goal_set.items()
        if goal is not None
    ]

    remaining = set(valid_keys)

    print(f"Number of tasks: {len(joint_goal_set)}")
    print(f"Number of successfully solved tasks: {len(remaining)}")
    print(f"Number of workers: {num_workers}")

    root_paths = {}
    key_to_root = {
        key: (None, None)
        for key in valid_keys
    }
    build_center_time = {
        key: np.nan
        for key in valid_keys
    }
    compress_time = {
        key: np.nan
        for key in valid_keys
    }

    contact_face_environment = has_contact_face(env)

    nn = None
    keys = None
    nn_by_face = None
    keys_by_face = None

    if contact_face_environment:
        nn_by_face = {}
        keys_by_face = {}

        for face in ["xy", "yz", "zx"]:
            face_task_set = {
                key[1:]: joint_goal_set[key]
                for key in valid_keys
                if key[0] == face
            }

            if not face_task_set:
                continue

            face_nn, _ = build_task_nn(face_task_set)

            nn_by_face[face] = face_nn
            keys_by_face[face] = list(face_task_set.keys())
    else:
        valid_joint_goal_set = {
            key: joint_goal_set[key]
            for key in valid_keys
        }

        nn, _ = build_task_nn(valid_joint_goal_set)
        keys = list(valid_joint_goal_set.keys())

    # Keys currently being evaluated as roots.
    # Reserved keys cannot be claimed as neighbors by another result.
    in_progress = set()

    # Future -> center key
    pending = {}

    pbar = tqdm(total=len(valid_keys))

    # Use spawn rather than fork because MuJoCo and C++ objects may not be
    # safe after a fork.
    context = mp.get_context("spawn")

    with ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=context,
        initializer=_initialize_condensation_worker,
        initargs=(
            env_name,
            robot_name,
            planner,
            adaptation,
            joint_goal_set,
            42,
        ),
    ) as executor:

        while remaining or pending:

            # Keep workers supplied with distinct root candidates.
            while remaining and len(pending) < num_workers:
                selectable_keys = list(remaining - in_progress)

                if not selectable_keys:
                    break

                center_key = selectable_keys[
                    np.random.randint(0, len(selectable_keys))
                ]

                # Reserve this key as a root candidate.
                in_progress.add(center_key)

                neighbor_keys = _get_neighbor_keys(
                    center_key=center_key,
                    n_neighbors=n_neighbors,
                    env_has_contact_face=contact_face_environment,
                    nn=nn,
                    keys=keys,
                    nn_by_face=nn_by_face,
                    keys_by_face=keys_by_face,
                )

                # This is only a snapshot. Some of these neighbors may be
                # covered before the worker result returns.
                candidate_neighbors = [
                    nb_key
                    for nb_key in neighbor_keys
                    if (
                        nb_key != center_key
                        and nb_key in remaining
                        and nb_key not in in_progress
                    )
                ]

                future = executor.submit(
                    _evaluate_root_candidate,
                    center_key,
                    candidate_neighbors,
                )

                pending[future] = center_key

            if not pending:
                break

            completed, _ = wait(
                pending,
                return_when=FIRST_COMPLETED,
            )

            for future in completed:
                center_key = pending.pop(future)

                try:
                    result = future.result()
                except Exception:
                    # This generally catches serialization or executor-level
                    # failures. Worker-side failures are returned in result.
                    tqdm.write(
                        f"Worker failed while evaluating {center_key}:\n"
                        f"{traceback.format_exc()}"
                    )

                    in_progress.discard(center_key)

                    # Leave the center in remaining so it can be retried.
                    continue

                in_progress.discard(center_key)

                if result["error"] is not None:
                    tqdm.write(
                        f"Error while evaluating {center_key}:\n"
                        f"{result['error']}"
                    )

                    # Leave it in remaining for now.
                    # You may instead remove it permanently after N retries.
                    continue

                # If planning failed, mark this key as uncompressible and
                # remove it from the active problem.
                if not result["success"]:
                    if center_key in remaining:
                        remaining.remove(center_key)
                        pbar.update(1)

                    continue

                # Reserved center keys cannot be assigned by another root,
                # so center_key should still be in remaining here.
                if center_key not in remaining:
                    # Defensive check in case the commit policy changes later.
                    continue

                root_id = len(root_paths)

                root_paths[root_id] = result["adapted_center"]
                key_to_root[center_key] = (
                    root_id,
                    result["center_q_end"],
                )
                build_center_time[center_key] = result[
                    "build_center_time"
                ]

                remaining.remove(center_key)
                pbar.update(1)

                newly_covered = 0

                # Commit only neighbors that are still unclaimed.
                for proposal in result["covered_neighbors"]:
                    nb_key = proposal["key"]

                    # Another completed root may already have claimed it.
                    if nb_key not in remaining:
                        continue

                    # Protect keys currently reserved by other workers.
                    if nb_key in in_progress:
                        continue

                    key_to_root[nb_key] = (
                        root_id,
                        proposal["q_end"],
                    )
                    compress_time[nb_key] = proposal[
                        "compression_time"
                    ]

                    remaining.remove(nb_key)
                    pbar.update(1)
                    newly_covered += 1

                if len(root_paths) % 50 == 0:
                    tqdm.write(
                        f"\nNumber of root paths: {len(root_paths)}."
                        f"\nNumber of completed tasks: "
                        f"{pbar.total - len(remaining)}."
                        f"\nPending root evaluations: {len(pending)}."
                        f"\nLast root newly covered: {newly_covered}."
                    )

    pbar.close()

    print("Condensation complete.")
    print(f"Number of root paths: {len(root_paths)}")

    results = np.stack(
        [
            list(build_center_time.values()),
            list(compress_time.values()),
        ],
        axis=1,
    )

    return root_paths, key_to_root, results


def main(args):
    """Generate a dataset of task paths for a given environment and robot."""
    folder = get_data_folder(args.env, args.robot)
    suffix = f"{args.ik}_{args.planner}_{args.adaptation}_{args.n_neighbors}"

    # Check if graph data is already generated
    data_exists = os.path.exists(f"{folder}/root_paths_{suffix}.pkl")
    if data_exists and not args.overwrite:
        print(
            f"Compressed root paths already exists at {folder} "
            + f"with IK '{args.ik}', planner '{args.planner}', "
            + f"and adaptation '{args.adaptation}'. "
            + "Use --overwrite to regenerate the task paths."
        )
        return

    # Load environment and robot
    env, robot = load_env_and_robot(args.env, args.robot, False)

    # Solve problems
    # Load the joint space problem set
    try:
        joint_goal_set = pickle.load(
            open(f"{folder}/joint_goal_set_{args.ik}.pkl", "rb")
        )
    except FileNotFoundError as e:
        print(e)
        print(
            f"Joint goal set with IK '{args.ik}' not found! "
            + "Generate the joint goal set with generate_joint_goal_set.py."
        )
        robot.close()
        return

    # Solve joint goal set with compression

    root_paths, key_to_root, results = (
        solve_joint_goal_set_condensed_parallel(
            env=env,
            robot=robot,
            env_name=args.env,
            robot_name=args.robot,
            joint_goal_set=joint_goal_set,
            planner=args.planner,
            adaptation=args.adaptation,
            n_neighbors=args.n_neighbors,
            num_workers=args.num_workers,
        )
    )

    # Save results
    np.save(f"{folder}/root_paths_results_{suffix}.npy", results)
    pickle.dump(root_paths, open(f"{folder}/root_paths_{suffix}.pkl", "wb"))
    pickle.dump(key_to_root, open(f"{folder}/key_to_root_{suffix}.pkl", "wb"))
    robot.close()


def parse_arguments():
    parser = argparse.ArgumentParser()
    # envs
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--env", choices=[
        "table", 
        "box",
        "cage",
        "shelf",
        "free",
        "real",
        "largeobj",
        "microwave",
        "allstable"
        ], default="table",
    )
    parser.add_argument(
        "--robot", choices=["panda", "ur10", "fetch"], default="panda"
    )
    parser.add_argument(
        "--ik", choices=["random", "neighbor", "grr"], default="neighbor"
    )
    parser.add_argument(
        "--planner", choices=["RRTConnect", "PRMstar"], default="RRTConnect"
    )
    parser.add_argument(
        "--adaptation", choices=["linear", "grr", "dmp", "opt"], default="grr"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of parallel condensation worker processes.",
    )
    parser.add_argument("--n_neighbors", type=int, default=100)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    set_seed(42)
    args = parse_arguments()

    main(args)
