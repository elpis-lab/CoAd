import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import time
import numpy as np
import pickle
from tqdm import tqdm

from plan_load.utils import set_seed, load_env_and_robot, get_data_folder
from plan_load.task_space import build_task_nn
from plan_load.env import MujocoEnv
from plan_load.robot import MujocoRobot
from plan_load.mink_ik import get_ik_solver

from plan_load.adaptation import LinearAdapter, GRRAdapter
from plan_load.adaptation import DMPAdapter, TrajOptAdapter


def condense_dataset(
    env: MujocoEnv,
    robot: MujocoRobot,
    task_paths,
    adaptation,
    n_neighbors=1000,
):
    """
    Condense a dataset of joint-space paths by greedily picking root paths
    and compressing nearby neighbors
    """
    model, data = robot.model, robot.data
    ik_solver = get_ik_solver(robot, env_collision_geoms=env.collision_geoms)
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
    nn, bin_poses = build_task_nn(task_paths)
    keys = list(task_paths.keys())

    # If no solution for a task, we assume it is unsolvable.
    # Get the successfully solved tasks.
    remaining = set(
        [
            key
            for key, val in task_paths.items()
            if val is not None and len(val) > 1
        ]
    )
    print(f"Number of tasks: {len(task_paths)}")
    print(f"Number of successfully solved tasks: {len(remaining)}")

    # Result containers
    root_paths = {}  # root_id -> root path
    key_to_root = {
        key: (None, None) for key in task_paths.keys()
    }  # key -> (root_id, goal_q)
    build_center_time = {key: np.nan for key in task_paths.keys()}
    compress_time = {key: np.nan for key in task_paths.keys()}

    # Greedy condensation loop
    pbar = tqdm(total=len(remaining))
    while remaining:
        # Pick a random center among remaining bins
        center_key = list(remaining)[np.random.randint(0, len(remaining))]
        center_path = task_paths[center_key].copy()
        env.move_swept_volume(center_key)

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
            robot.set_joint_qpos(task_paths[center_key][-1])
            robot.viewer.sync()
        pbar.update(1)

        # Query nearest neighbors and
        # Consider those that are still in remaining
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
            nb_path = task_paths[nb_key].copy()

            # Set up neighbor environment
            env.move_swept_volume(nb_key)
            # Try to compress neighbor into this root.
            t0 = time.perf_counter()
            valid, q_nb_end = adapter.compress(adapted_center, nb_path)
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
            tqdm.write(f"[{len(root_paths)}] ")

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
    env, robot = load_env_and_robot(args.env, args.robot)

    # Solve problems
    # Load the joint space problem set
    try:
        d_name = f"{folder}/task_paths_data_{args.ik}_{args.planner}.npy"
        data = np.load(d_name, allow_pickle=True)
        k_name = f"{folder}/task_paths_keys_{args.ik}_{args.planner}.pkl"
        keys = pickle.load(open(k_name, "rb"))
        task_paths = {key: data for key, data in zip(keys, data)}
    except FileNotFoundError as e:
        print(e)
        print(
            f"Task paths with IK '{args.ik}' and planner '{args.planner}' "
            + "not found! "
            + "Generate the task paths with generate_task_paths.py."
        )
        robot.close()
        return

    # Condense dataset
    root_paths, key_to_root, results = condense_dataset(
        env, robot, task_paths, args.adaptation, args.n_neighbors
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
    parser.add_argument(
        "--env",
        choices=["table", "box", "cage", "shelf", "free", "real"],
        default="table",
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
    parser.add_argument("--n_neighbors", type=int, default=1000)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    set_seed(42)
    args = parse_arguments()

    main(args)
