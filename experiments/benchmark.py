import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pickle
import mujoco
import time
import argparse

from tqdm import tqdm
from plan_load.env import MujocoEnv
from plan_load.robot import MujocoRobot
from plan_load.mink_ik import get_ik_solver

from plan_load.adaptation import LinearAdapter, GRRAdapter
from plan_load.adaptation import DMPAdapter, TrajOptAdapter

# from plan_load.task_space import deep_tuple
from experiments.evaluate import traj_len
from plan_load.utils import set_seed, load_env_and_robot, get_data_folder
from plan_load.planning import OMPLPlanner, euclidean_path_length

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

def evaluate_graph(
    env: MujocoEnv,
    robot: MujocoRobot,
    root_paths,
    key_to_root,
    adaptation,
    planning_results
):
    model, data = robot.model, robot.data
    home_qpos = robot.get_joint_qpos()

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
    
    solved_keys = [k for k, (rid, _) in key_to_root.items() if rid is not None]
    print(f"Number of solved paths: {len(solved_keys)}")
    print(f"Number of compressed root paths: {len(root_paths)}")

    adaptation_times = []
    adaptation_lengths = []
    adaptation_success = []

    pbar = tqdm(enumerate(key_to_root), total=len(solved_keys))
    for i, key in enumerate(solved_keys):
        #print(f"key index: {i}")
        env.move_swept_volume(key)
        t0 = time.perf_counter()
        root_id, curr_goal = key_to_root[key]
        if root_id is None:
            adaptation_success.append(False)
            adaptation_lengths.append(np.nan)
            continue
        curr_root = root_paths[root_id]
        adapted_path = adapter.adapt(curr_root, curr_goal)
        t1 = time.perf_counter()
        dt = t1 - t0
        adaptation_times.append(dt)
        if adapted_path is None:
            adaptation_success.append(False)
            adaptation_lengths.append(np.nan)
        else:
            adaptation_success.append(True)
            adaptation_lengths.append(traj_len(adapted_path))

        pbar.update(1)

    adaptation_times = np.array(adaptation_times)
    adaptation_lengths = np.array(adaptation_lengths)
    adaptation_success = np.array(adaptation_success)

    planning_success = planning_results[:, 0]
    planning_times = planning_results[:, 1]
    planning_total_times = planning_results[:, 2]
    planning_lengths = planning_results[:, 3]

    mean_planning_time = np.nanmean(planning_times)
    mean_total_planning_time = np.nanmean(planning_total_times)
    mean_adaptation_time = np.nanmean(adaptation_times)

    mean_planning_length = np.nanmean(planning_lengths)
    mean_adaptation_length = np.nanmean(adaptation_lengths)

    np.savez(
        f"data/benchmark_results_{adaptation}.npz",
        planning_times=planning_times,
        planning_lengths=planning_lengths,
        planning_success=planning_success,
        adaptation_times=adaptation_times,
        adaptation_lengths=adaptation_lengths,
        adaptation_success=adaptation_success,
    )

    print(f"Mean planning time: {mean_planning_time}")
    print(f"Mean adaptation time: {mean_adaptation_time}")

    print(f"Mean planning length: {mean_planning_length}")
    print(f"Mean adaptation length: {mean_adaptation_length}")



def main(args):
    """Evaluate path quality and query time for graph"""
    folder = get_data_folder(args.env, args.robot)
    suffix = f"{args.ik}_{args.planner}_{args.adaptation}_{args.n_neighbors}"
    
    # Check if graph data exists
    root_path = f"{folder}/root_paths_{suffix}.pkl"
    map_path = f"{folder}/key_to_root_{suffix}.pkl"

    root_exists = os.path.exists(root_path)
    map_exists = os.path.exists(map_path)
    
    if not root_exists or not map_exists:
        print("Compressed root paths "
            + f"with IK '{args.ik}', planner '{args.planner}', "
            + f"and adaptation '{args.adaptation}' does NOT exist. "
            + "Use condense_task_paths.py to generate it."
        )
        return

    # Load environment and robot
    env, robot = load_env_and_robot(args.env, args.robot)

    root_data = pickle.load(open(root_path, "rb"))
    map_data = pickle.load(open(map_path, "rb"))

    task_set_path = f"{folder}/task_set.pkl"
    joint_goal_set_path = f"{folder}/joint_goal_set_neighbor.pkl"

    task_set_data = pickle.load(open(task_set_path, "rb"))
    joint_goal_set_data = pickle.load(open(joint_goal_set_path, "rb"))
    
    print(f"Number of generated tasks: {len(task_set_data)}")
    print(f"Number of solved IK: {len(joint_goal_set_data)}")

    planning_results_path = f"{folder}/task_paths_results_neighbor_RRTConnect.npy"
    planning_results = np.load(planning_results_path)

    evaluate_graph(env, robot, root_data, map_data, args.adaptation, planning_results)


def parse_arguments():
    parser = argparse.ArgumentParser()
    # envs
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--env",
        choices=["table", "box", "cage", "shelf", "free"],
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

if __name__=="__main__":
    args = parse_arguments()

    main(args)
