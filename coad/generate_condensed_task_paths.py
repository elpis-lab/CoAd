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


def condense_dataset(
    env: MujocoEnv,
    robot: MujocoRobot,
    task_paths,
    adaptation,
    n_neighbors=100,
):
    """
    Condense a dataset of joint-space paths by greedily picking root paths
    and compressing nearby neighbors
    """
    model, data = robot.model, robot.data
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

        # # DEBUG
        # def visualize_path(path):
        #     for q in path:
        #         robot.set_joint_qpos(q)
        #         robot.viewer.sync()
        #         time.sleep(0.005)

        # input("PATH1")
        # # visualize_path(center_path)
        # input("PATH2")
        # center_path2 = adapter.adapt(adapted_center, q_end - 0.3)
        # print(center_path2.shape)
        # # center_path2[:, -1] = 0
        # # valid = adapter.path_validity_check(center_path2)
        # # print(valid)
        # print(adapted_center.start)
        # print(q_end)
        # print(len(center_path))
        # print(len(center_path2))
        # print(center_path[:20, 0])
        # print(center_path2[:20, 0])
        # visualize_path(center_path2)
        # # Plot 7 figures to compare center path and center path 2
        # import matplotlib.pyplot as plt

        # fig, axs = plt.subplots(7, 1)
        # for i in range(7):
        #     axs[i].plot(center_path[:, i])
        #     axs[i].plot(center_path2[:, i])
        #     axs[i].legend(["center path", "center path2"])
        #     axs[i].set_title(f"Joint {i}")
        #     axs[i].set_xlabel("Time")
        #     axs[i].set_ylabel("Joint Position")
        # plt.show()
        # if not valid:
        #     for i in range(len(center_path2) - 1):
        #         q1 = center_path2[i]
        #         q2 = center_path2[i + 1]
        #         if not adapter.segment_validity_check(q1, q2):
        #             input("INVALID")
        #             robot.set_joint_qpos(q1)
        #             robot.viewer.sync()
        #             input("INVALID")
        # input("DONE")

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

            # input(f"NEIGHBOR PATH {valid}")
            # adapted_nb = adapter.adapt(adapted_center, q_nb_end)
            # visualize_path(adapted_nb)
            # for i in range(7):
            #     axs[i].plot(center_path[:, i])
            #     axs[i].plot(adapted_nb[:, i])
            #     axs[i].legend(["center path", "center path2"])
            # plt.show()

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

    root_paths, key_to_root, results = solve_joint_goal_set_condensed(
        env, robot, joint_goal_set, args.planner, args.adaptation, args.n_neighbors
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
    parser.add_argument("--n_neighbors", type=int, default=100)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    set_seed(42)
    args = parse_arguments()

    main(args)
