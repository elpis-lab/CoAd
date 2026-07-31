import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pickle
import mujoco
import time
import argparse
from scipy.spatial import cKDTree

from tqdm import tqdm
from coad.env import MujocoEnv
from coad.robot import MujocoRobot
from coad.mink_ik import get_ik_solver

from coad.adaptation import LinearAdapter, GRRAdapter
from coad.adaptation import DMPAdapter, TrajOptAdapter

# from coad.task_space import deep_tuple
from experiments.visualize_paths import traj_len
from coad.utils import set_seed, load_env_and_robot, get_data_folder

from coad.env import FreeEnv, CageEnv, BoxEnv, TableEnv, ShelfEnv, LargeObjectEnv, MicrowaveEnv, AllStableEnv
from coad.robot import Panda, UR10, FetchArm

from experiments.evaluate_baselines import BoxGrid, sample_from_key

folder1 = "dataset/top_naive"
folder2 = "dataset/top"



def deep_tuple(x):
    # NumPy array
    if isinstance(x, np.ndarray):
        return tuple(deep_tuple(i) for i in x)
    # Python list or tuple
    elif isinstance(x, (list, tuple)):
        return tuple(deep_tuple(i) for i in x)
    # Base case (scalar)
    else:
        return x


def get_avg_path_length(root_path, key_map):
    lengths = np.zeros(len(key_map), dtype=float)
    for i, key in enumerate(key_map):
        key = deep_tuple(key)
        root_id, goal_q = key_map[key]
        path = list(root_path[root_id].copy())
        path.append(goal_q)

        lengths[i] = traj_len(path)
    return np.mean(lengths)


def wrap_pi(self, a):
    return (a + np.pi) % (2 * np.pi) - np.pi

from collections import defaultdict
import numpy as np


def inspect_grid(task_set):
    keys = list(task_set.keys())

    has_face = (
        len(keys[0]) == 5
        and isinstance(keys[0][0], str)
    )

    groups = defaultdict(list)

    for key in keys:
        face = key[0] if has_face else None
        numeric_key = key[1:] if has_face else key
        groups[face].append(numeric_key)

    for face, face_keys in groups.items():
        print(f"\n=== Face: {face} ===")
        print(f"Number of keys: {len(face_keys)}")

        for dim, name in enumerate(["x", "y", "z", "yaw"]):
            intervals = sorted({
                (
                    round(float(key[dim][0]), 10),
                    round(float(key[dim][1]), 10),
                )
                for key in face_keys
            })

            starts = np.array(sorted({lo for lo, _ in intervals}))
            widths = np.array(sorted({round(hi - lo, 10) for lo, hi in intervals}))

            print(f"\n{name}:")
            print(f"  unique intervals: {len(intervals)}")
            print(f"  unique starts:    {len(starts)}")
            print(f"  unique widths:    {widths}")

            if len(starts) > 1:
                steps = np.diff(starts)
                print(
                    "  unique start steps:",
                    np.unique(np.round(steps, 10)),
                )

        unique_per_dim = [
            {
                (
                    round(float(key[d][0]), 10),
                    round(float(key[d][1]), 10),
                )
                for key in face_keys
            }
            for d in range(4)
        ]

        expected_cartesian_size = np.prod(
            [len(values) for values in unique_per_dim]
        )

        actual_keys = {
            tuple(
                (
                    round(float(interval[0]), 10),
                    round(float(interval[1]), 10),
                )
                for interval in key
            )
            for key in face_keys
        }

        print("\nCartesian check:")
        print(f"  dimension counts: {[len(v) for v in unique_per_dim]}")
        print(f"  expected product: {expected_cartesian_size}")
        print(f"  actual key count: {len(actual_keys)}")
        print(
            "  full Cartesian:",
            len(actual_keys) == expected_cartesian_size,
        )




def evaluate_adaptations(
    args,
    env: MujocoEnv,
    robot: MujocoRobot,
    folder,
    task_set,
    task_paths,
    adaptations,
):
    model, data = robot.model, robot.data
    home_qpos = robot.get_joint_qpos()

    # ik_solver = get_ik_solver(robot, env_collision_geoms=env.collision_geoms)
    ik_solver = get_ik_solver(robot, env_collision_geoms=env.env_details['collision_geoms'])
    solved_task_paths_keys = [
        k
        for k, path in task_paths.items()
        if path is not None and len(path) > 0
    ]
    solved_task_paths = {
        k: v for k, v in task_paths.items() if v is not None and len(v) > 0
    }

    print(f"Number of solved paths: {len(solved_task_paths_keys)}")

    adapters = []
    key_to_roots = []
    root_paths_list = []

    # Setup grids for base library and adaptations
    indexers = []
    indexers.append(BoxGrid(task_set))

    full_lib_success = []
    full_lib_times = []
    full_lib_lengths = []

    adaptation_success = {}
    adaptation_times = {}
    adaptation_lengths = {}

    lib_sizes = {}
    lib_sizes["full"] = len(solved_task_paths_keys)

    for adaptation in adaptations:

        adaptation_success[f"{adaptation}"] = []
        adaptation_times[f"{adaptation}"] = []
        adaptation_lengths[f"{adaptation}"] = []

        suffix = f"{args.ik}_{args.planner}_{adaptation}_{args.n_neighbors}"
        root_path = f"{folder}/root_paths_{suffix}.pkl"
        map_path = f"{folder}/key_to_root_{suffix}.pkl"

        root_data = pickle.load(open(root_path, "rb"))
        map_data = pickle.load(open(map_path, "rb"))
        key_to_roots.append(map_data)
        root_paths_list.append(root_data)

        print(f"{adaptation} library size: {len(root_data)}")
        lib_sizes[f"{adaptation}"] = len(root_data)

        indexers.append(BoxGrid(map_data))

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
        adapters.append(adapter)

    pbar = tqdm(
        enumerate(solved_task_paths_keys), total=len(solved_task_paths_keys)
    )
    for i, key in enumerate(solved_task_paths_keys):

        # sample = []
        # for lo, hi in key:
        #     x = np.random.uniform(lo, hi)
        #     # x = np.nextafter(x, lo)
        #     sample.append(x)
        sample = sample_from_key(key)
        # env.move_cube_object(sample)
        env.move_object(sample)

        # Benchmark full library
        full_start = time.perf_counter()
        recovered_key_full = indexers[0].query_point(sample)

        if recovered_key_full is None:
            indexer = indexers[0]

            expected_indices = indexer.key_to_indices(key)
            sampled_indices = indexer._bin_indices(sample)

            print("\nFull library failure")
            print("Original key:", key)
            print("Sample:", sample)
            print("Key exists:", key in indexer.keys_list)
            print("Expected indices:", expected_indices)
            print("Sample indices:  ", sampled_indices)
            print(
                "Expected index exists:",
                expected_indices in indexer.index,
            )
            print(
                "Sample index exists:",
                sampled_indices in indexer.index
                if sampled_indices is not None
                else False,
            )

            if sampled_indices in indexer.index:
                candidate_idx = indexer.index[sampled_indices]
                candidate_key = indexer.keys_list[candidate_idx]
                print("Candidate key:", candidate_key)

            # break

            full_lib_success.append(False)
            full_end = time.perf_counter()
            full_lib_times.append(full_end - full_start)
            full_lib_lengths.append(np.nan)
        else:
            if key != recovered_key_full:
                print(f"Original key: {key}")
                print(f"Full library recovered key: {recovered_key_full}")

                input()

            full_lib_path = task_paths[recovered_key_full]
            full_end = time.perf_counter()
            full_lib_success.append(True)
            full_lib_times.append(full_end - full_start)
            full_lib_lengths.append(traj_len(full_lib_path))

        # Benchmark each adaptation
        for adaptation_ind, adaptation in enumerate(adaptations):
            adapt_start = time.perf_counter()
            recovered_key_adapt = indexers[adaptation_ind + 1].query_point(
                sample
            )
            if recovered_key_adapt is None:
                adaptation_success[adaptation].append(False)
                adapt_end = time.perf_counter()
                adaptation_times[adaptation].append(adapt_end - adapt_start)
                adaptation_lengths[adaptation].append(np.nan)
                continue
            root_id, curr_goal = key_to_roots[adaptation_ind][
                recovered_key_adapt
            ]

            if key != recovered_key_adapt:
                print(f"Original key: {key}")
                # print(sample)
                print(f"{adaptation} recovered key: {recovered_key_adapt}")
                input()

            curr_root = root_paths_list[adaptation_ind][root_id]
            adapted_path = adapters[adaptation_ind].adapt(curr_root, curr_goal)
            adapt_end = time.perf_counter()
            adapt_time = adapt_end - adapt_start

            curr_success = adapted_path is not None and len(adapted_path) > 0
            adaptation_success[adaptation].append(curr_success)
            adaptation_times[adaptation].append(adapt_time)
            adaptation_lengths[adaptation].append(traj_len(adapted_path))

        pbar.update(1)

    full_lib_success = np.array(full_lib_success)
    full_lib_times = np.array(full_lib_times)
    full_lib_lengths = np.array(full_lib_lengths)

    print()
    print(f"\nFull library results")
    print(f"Full library success rate: {np.mean(full_lib_success)*100}%")
    print(f"Full library mean time: {np.mean(full_lib_times)*1000} ms")
    print(f"Full library mean length: {np.nanmean(full_lib_lengths)}")

    for adaptation in adaptations:
        adaptation_success[adaptation] = np.array(
            adaptation_success[adaptation]
        )
        adaptation_times[adaptation] = np.array(adaptation_times[adaptation])
        adaptation_lengths[adaptation] = np.array(
            adaptation_lengths[adaptation]
        )

        print(f"\n{adaptation} results")
        curr_compression_ratio = lib_sizes[adaptation] / lib_sizes["full"]
        curr_comp_percent = (1 - curr_compression_ratio) * 100
        print(f"{adaptation} compression %: {curr_comp_percent}%")
        print(
            f"{adaptation} success rate: {np.mean(adaptation_success[adaptation])*100}%"
        )
        print(
            f"{adaptation} mean time: {np.mean(adaptation_times[adaptation])*1000} ms"
        )
        print(
            f"{adaptation} mean length: {np.nanmean(adaptation_lengths[adaptation])}"
        )

    results_path = f"data/adaptation_results_{args.robot}_{args.env}.npz"
    results = {
        "full": {
            "success": full_lib_success,
            "times": full_lib_times,
            "lengths": full_lib_lengths,
        },
        "adaptations": {
            "success": adaptation_success,
            "times": adaptation_times,
            "lengths": adaptation_lengths,
        },
    }

    np.savez(results_path, results=results)


def main(args):
    """Evaluate path quality and query time for graph"""
    folder = get_data_folder(args.env, args.robot)

    try:
        task_set = pickle.load(open(f"{folder}/task_set.pkl", "rb"))
        # inspect_grid(task_set)
        joint_goal_set = pickle.load(
            open(f"{folder}/joint_goal_set_{args.ik}.pkl", "rb")
        )
        d_name = f"{folder}/task_paths_data_{args.ik}_{args.planner}.npy"
        data = np.load(d_name, allow_pickle=True)
        k_name = f"{folder}/task_paths_keys_{args.ik}_{args.planner}.pkl"
        keys = pickle.load(open(k_name, "rb"))
        task_paths = {key: data for key, data in zip(keys, data)}

    except FileNotFoundError as e:
        print(e)
        print(f"One or more required files not found.")
        return

    IK_solved = sum(v is not None for v in joint_goal_set.values())
    planner_solved = sum(
        v is not None and len(v) > 1 for v in task_paths.values()
    )

    print(f"Number of generated tasks: {len(task_set)}")
    print(f"Number of tasks solved by IK: {IK_solved}")
    print(f"Number of paths solved by {args.planner}: {planner_solved}")

    adaptations_found = []
    for adaptation in ["grr", "opt", "dmp"]:
        suffix = f"{args.ik}_{args.planner}_{adaptation}_{args.n_neighbors}"
        root_path = f"{folder}/root_paths_{suffix}.pkl"
        map_path = f"{folder}/key_to_root_{suffix}.pkl"
        root_exists = os.path.exists(root_path)
        map_exists = os.path.exists(map_path)

        if root_exists and map_exists:
            adaptations_found.append(adaptation)
    if len(adaptations_found) == 0:
        raise FileNotFoundError(
            f"No adaptations found for problem: {args.robot} in {args.env}"
        )
    print(f"Adaptations found: {adaptations_found}")

    env_name = args.env
    robot_name = args.robot
    visualize = False

    if env_name == "table":
        env = TableEnv(robot_name, using_swept_volume=False)
    elif env_name == "box":
        env = BoxEnv(robot_name, using_swept_volume=False)
    elif env_name == "cage":
        env = CageEnv(robot_name, using_swept_volume=False)
    elif env_name == "shelf":
        env = ShelfEnv(robot_name, using_swept_volume=False)
    elif env_name == "free":
        env = FreeEnv(robot_name, using_swept_volume=False)
    elif env_name == "largeobj":
        env = LargeObjectEnv(robot_name, using_swept_volume=False)
    elif env_name == "allstable":
        env = AllStableEnv(robot_name, using_swept_volume=False)
    else:
        raise ValueError(f"Invalid environment: {env_name}")

    model, data = env.model, env.data
    if robot_name == "panda":
        robot = Panda(model, data, visualize)
    elif robot_name == "ur10":
        robot = UR10(model, data, visualize)
    elif robot_name == "fetch":
        robot = FetchArm(model, data, visualize)
    else:
        raise ValueError(f"Invalid robot: {robot_name}")

    robot_pos = env.env_details['robot_pos']
    robot_quat = env.env_details['robot_quat']
    robot.teleport_base(pos=robot_pos, quat=robot_quat)

    # root_data = pickle.load(open(root_path, "rb"))
    # map_data = pickle.load(open(map_path, "rb"))

    # planning_results_path = f"{folder}/task_paths_results_neighbor_RRTConnect.npy"
    # planning_results = np.load(planning_results_path)

    evaluate_adaptations(
        args, env, robot, folder, task_set, task_paths, adaptations_found
    )


def parse_arguments():
    parser = argparse.ArgumentParser()
    # envs
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--env", choices=[
            "table",
            "box",
            "cage",
            "shelf",
            "free",
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
        "--planner", choices=["RRTConnect", "PRMstar", "VAMP"], default="RRTConnect"
    )
    # parser.add_argument(
    #     "--adaptation", choices=["linear", "grr", "dmp", "opt"], default="grr"
    # )
    parser.add_argument("--n_neighbors", type=int, default=100)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
