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

from plan_load.env import FreeEnv, CageEnv, BoxEnv, TableEnv, ShelfEnv
from plan_load.robot import Panda, UR10, FetchArm

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

def visualize_solution(
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


    try:
        while(True):
            #print("Sampling a key")
            key = list(key_to_root.keys())[np.random.randint(0, len(key_to_root))]
            root_id, curr_goal = key_to_root[key]
            if root_id is None:
                #print("Unsolved bin...")
                continue

            sample = []
            for lo, hi in key:
                sample.append(np.random.uniform(lo, hi))

            print(sample)
            env.move_cube_object(sample)
            robot.viewer.sync()

            proceed = input("Proceed?")

            if proceed !="":
                t0 = time.perf_counter()
                curr_root = root_paths[root_id]
                adapted_path = adapter.adapt(curr_root, curr_goal)
                t1 = time.perf_counter()
                dt = t1 - t0
                print(f"Time: {dt}")

                for waypoint in adapted_path:
                    robot.set_joint_qpos(waypoint)
                    robot.viewer.sync()
                    input()
                input("Trajectory complete.")
                continue
            else:
                continue

            #break

    except KeyboardInterrupt:
        pass
        

    # pbar = tqdm(enumerate(key_to_root), total=len(solved_keys))
    # for i, key in enumerate(solved_keys):
    #     #print(f"key index: {i}")
    #     env.move_swept_volume(key)
    #     t0 = time.perf_counter()
    #     root_id, curr_goal = key_to_root[key]
    #     if root_id is None:
    #         adaptation_success.append(False)
    #         adaptation_lengths.append(np.nan)
    #         continue
    #     curr_root = root_paths[root_id]
    #     adapted_path = adapter.adapt(curr_root, curr_goal)
    #     t1 = time.perf_counter()
    #     dt = t1 - t0
    #     adaptation_times.append(dt)
    #     if adapted_path is None:
    #         adaptation_success.append(False)
    #         adaptation_lengths.append(np.nan)
    #     else:
    #         adaptation_success.append(True)
    #         adaptation_lengths.append(traj_len(adapted_path))

    #     pbar.update(1)

def visualize_solution2(
    env: MujocoEnv,
    robot: MujocoRobot,
    root_paths,
    key_to_root,
    adaptation,
    planning_results,
    N_save: int = 10,
):
    """
    Phase 1 (record):
        - Cycle through random object poses.
        - Press Enter to skip.
        - Type anything to accept and save.
        - Stop after N_save successful adaptations.

    Phase 2 (playback):
        - Visualize each saved pose + adapted path frame-by-frame.
    """

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

    saved = []

    print("\n=== RECORD MODE ===")
    print(f"Press Enter to skip a pose.")
    print(f"Type ANYTHING + Enter to accept and save.")
    print(f"Need {N_save} successful saves.\n")

    try:
        while len(saved) < N_save:

            # sample a solved key
            while True:
                key = list(key_to_root.keys())[np.random.randint(0, len(key_to_root))]
                root_id, curr_goal = key_to_root[key]
                if root_id is not None:
                    break

            # sample continuous pose inside bin
            sample = [float(np.random.uniform(lo, hi)) for (lo, hi) in key]

            env.move_cube_object(sample)
            robot.viewer.sync()

            resp = input(f"[{len(saved)}/{N_save}] Accept? (Enter=skip, text=save): ").strip()
            if resp == "":
                continue

            curr_root = root_paths[root_id]

            t0 = time.perf_counter()
            adapted_path = adapter.adapt(curr_root, curr_goal)
            t1 = time.perf_counter()

            dt = t1 - t0

            if adapted_path is None or len(adapted_path) == 0:
                print(f"Adaptation failed (dt={dt:.4f}s). Not saved.\n")
                continue

            saved.append({
                "object_pose": np.array(sample),
                "adapted_path": np.array(adapted_path),
                "root_id": root_id,
                "adapt_time_s": dt,
            })

            print(f"Saved {len(saved)}/{N_save} "
                  f"(dt={dt:.4f}s, T={len(adapted_path)} waypoints)\n")

    except KeyboardInterrupt:
        print("\nInterrupted during record mode.")

    if len(saved) == 0:
        print("No samples saved. Exiting.")
        return

    print("\n=== PLAYBACK MODE ===")
    print("Press Enter to step through waypoints. Ctrl+C to quit.\n")

    try:
        for i, item in enumerate(saved):

            obj = item["object_pose"]
            path = item["adapted_path"]

            print(f"\n--- Sample {i+1}/{len(saved)} ---")
            print(f"root_id = {item['root_id']}, "
                  f"adapt_time_s = {item['adapt_time_s']:.4f}, "
                  f"T = {len(path)}")

            env.move_cube_object(obj)
            robot.viewer.sync()

            robot.set_joint_qpos(home_qpos)
            robot.viewer.sync()

            input("Press Enter to start trajectory...")

            for waypoint in path:
                robot.set_joint_qpos(waypoint)
                robot.viewer.sync()
                input()

            input("Trajectory complete. Press Enter for next sample...")

    except KeyboardInterrupt:
        print("\nPlayback interrupted.")



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
    #env, robot = load_env_and_robot(args.env, args.robot, visualize=True)

    env_name = args.env
    robot_name = args.robot

    if env_name == "table":
        env = TableEnv(robot_name, no_sv=True)
    elif env_name == "box":
        env = BoxEnv(robot_name, no_sv=True)
    elif env_name == "cage":
        env = CageEnv(robot_name, no_sv=True)
    elif env_name == "shelf":
        env = ShelfEnv(robot_name, no_sv=True)
    elif env_name == "free":
        env = FreeEnv(robot_name, no_sv=True)
    else:
        raise ValueError(f"Invalid environment: {env_name}")

    model, data = env.model, env.data
    if robot_name == "panda":
        robot = Panda(model, data, True)
    elif robot_name == "ur10":
        robot = UR10(model, data, True)
    elif robot_name == "fetch":
        robot = FetchArm(model, data, True)
    else:
        raise ValueError(f"Invalid robot: {robot_name}")

    robot.teleport_base(pos=env.robot_pos, quat=env.robot_quat)

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

    # Configure camera
    robot.viewer.cam.lookat[:] = [0.25, -0.25, 0.5]
    robot.viewer.cam.distance = 1.75
    robot.viewer.cam.azimuth = 120
    robot.viewer.cam.elevation = -20

    N_save = 3
    #visualize_solution(env, robot, root_data, map_data, args.adaptation, planning_results)
    visualize_solution2(env, robot, root_data, map_data, args.adaptation, planning_results, N_save)


def parse_arguments():
    parser = argparse.ArgumentParser()
    # envs
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--env",
        choices=["table", "box", "cage", "shelf", "free"],
        default="free",
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
