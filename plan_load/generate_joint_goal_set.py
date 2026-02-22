from typing import Any


import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import time
import numpy as np
import pickle
from tqdm import tqdm

from plan_load.env import MujocoEnv
from plan_load.robot import MujocoRobot
from plan_load.pose import matrix_to_flat, Pose
from plan_load.utils import set_seed, load_env_and_robot, get_data_folder
from plan_load.task_space import build_task_nn
from plan_load.mink_ik import get_ik_solver
from plan_load.mujoco_utils import sample_qpos


def get_ik_reference(robot: MujocoRobot, key, attempts, ik_method, **kwargs):
    """Get a reference for IK solving"""
    # If using random method
    # or we have already tried the given method (attempts > 1)
    # use Random
    if ik_method == "random" or attempts > 1:
        seed = sample_qpos(robot.model, robot.joint_ids)

    # If method is not random but we have already tried the given method
    # set None to use the default ik solver pos
    elif attempts == 1:
        seed = None

    # Use given method
    elif ik_method == "neighbor":
        nn = kwargs["nn"]
        k = kwargs["k"]
        joint_goal_set = kwargs["joint_goal_set"]
        keys = kwargs["keys"]

        # get the nearest neighbors
        # convert key to (x, y, z, rz)
        key_arr = np.array(key)
        key_center = (key_arr[:, 0] + key_arr[:, 1]) / 2
        neighbors = nn.query(
            [key_center],
            k * 10,  # query more neighbors for better accuracy
            return_distance=True,
            sort_results=True,
        )[1][0]
        neighbors = neighbors[:k]

        # get the averaged configuration of the neighbors
        configs = []
        for neighbor in neighbors:
            key = keys[neighbor]
            if joint_goal_set[key] is not None:
                configs.append(joint_goal_set[key])
        if len(configs) == 0:
            seed = None
        else:
            seed = np.mean(configs, axis=0)

    elif ik_method == "grr":
        raise NotImplementedError("GRR method not implemented yet")

    else:
        raise ValueError(f"Invalid IK method: {ik_method}")

    return seed


def convert_task_to_joint_goal(
    env: MujocoEnv,
    robot: MujocoRobot,
    task_set,
    ik_method,
    ik_max_attempts=20,
    use_col=False,
    **kwargs,
):
    """Convert task set to joint goal set."""
    # Get an IK solver
    model = robot.model
    data = robot.data
    ik_solver = get_ik_solver(robot, env_collision_geoms=env.collision_geoms)

    # Result containers
    ik_success = np.zeros(len(task_set), dtype=bool)
    ik_times = np.zeros(len(task_set), dtype=float)
    joint_goal_set = {key: None for key in task_set.keys()}

    # Prepare for different IK methods
    method_args = {}
    if ik_method == "random":
        pass
    elif ik_method == "neighbor":
        # build a Ball tree for finding the nearest task
        nn, bin_poses = build_task_nn(task_set)
        method_args = {
            "nn": nn,
            "k": kwargs.get("n_neighbors", 10),
            "joint_goal_set": joint_goal_set,
            "keys": list(task_set.keys()),
        }
    # TODO: GRR method
    elif ik_method == "grr":
        raise NotImplementedError("GRR method not implemented yet")
    else:
        raise ValueError(f"Invalid IK method: {ik_method}")

    # Start sovling IK one by one
    pbar = tqdm(enumerate[Any](task_set), total=len(task_set))
    for i, key in pbar:
        # Moving object (swept volume) to given key pose
        env.move_swept_volume(key)

        # Get the end effector target of the current task
        # object pose
        key_arr = np.array(key)
        key_center = (key_arr[:, 0] + key_arr[:, 1]) / 2
        obj_pose = Pose(key_center[:3], (0, 0, key_center[3])).matrix()

        # removing unreachable offset from env.ee_offset

        #if env.yaw_edges is not None and env.best_ee_offset_idx is not None:
        if env.yaw_edges is not None and env.worst_ee_offset_idx is not None: 
            yaw = (key_center[3] + np.pi) % (2*np.pi) - np.pi
            k = np.searchsorted(env.yaw_edges, yaw, side="right") - 1
            #k = np.clip(k, 0, len(env.best_ee_offset_idx) - 1)
            # Tz180 = np.array([
            #     [-1.0,  0.0, 0.0, 0.0],
            #     [ 0.0, -1.0, 0.0, 0.0],
            #     [ 0.0,  0.0, 1.0, 0.0],
            #     [ 0.0,  0.0, 0.0, 1.0],
            # ], dtype=float)
            # Tew1 = env.ee_offset[env.best_ee_offset_idx[k]]
            # Tew2 = Tew1 @ Tz180
            # ee_offsets = [Tew1, Tew2]
            k = np.clip(k, 0, len(env.worst_ee_offset_idx) - 1)
            bad = int(env.worst_ee_offset_idx[k])
            ee_offsets = [T for j, T in enumerate(env.ee_offset) if j != bad]

        else:
            ee_offsets = env.ee_offset.copy()
        # multiple potential ee targets
        targets = [
            #matrix_to_flat(obj_pose @ offset) for offset in env.ee_offset
            matrix_to_flat(obj_pose @ offset) for offset in ee_offsets
        ]
        # give each target the same number of attempts
        n_target_attempts = int(np.ceil(ik_max_attempts / len(targets)))
        # ik_max_attempts = len(targets) * n_target_attempts

        # Start solving IK
        t0 = time.perf_counter()
        valid_ik = False
        for target in targets:
            for target_attempts in range(n_target_attempts):
                # Solve IK
                reference = get_ik_reference(
                    robot, key, target_attempts, ik_method, **method_args
                )
                reached, solution = ik_solver.solve(
                    target, reference, use_col=use_col
                )

                if reached:
                    robot.set_joint_qpos(solution)
                    if not robot.in_contact():
                        valid_ik = True
                        break
            if valid_ik:
                break

        ik_times[i] = time.perf_counter() - t0
        ik_success[i] = valid_ik
        if valid_ik:
            joint_goal_set[key] = solution

        # Update viewer
        if robot.viewer is not None:
            robot.viewer.sync()

        #if i>=750:
        #    input(f"valid_ik: {valid_ik}, in_contact: {robot.in_contact()}")

        # Update tqdm message periodically
        print_interval = 500
        if (i + 1) % print_interval == 0:
            m_ik = np.nanmean(ik_times[np.array(ik_success)])
            tqdm.write(
                f"[{i+1}] "
                f"IK Success: {np.sum(ik_success)/(i+1):.3f} | "
                f"IK Time: {m_ik:.4f}s | "
            )

    # Stack results (n, 2)
    results = np.stack([ik_success, ik_times], axis=1)
    return joint_goal_set, results


def main(args):
    """Generate a dataset of task paths for a given environment and robot."""
    folder = get_data_folder(args.env, args.robot)
    suffix = f"{args.ik}"

    # Check if graph data is already generated
    data_exists = os.path.exists(f"{folder}/joint_goal_set_{suffix}.pkl")
    if data_exists and not args.overwrite:
        print(
            f"Joint goal set from IK '{args.ik}' already exists at {folder}. "
            + "Use --overwrite to regenerate the joint goal set."
        )
        return

    # Load environment and robot
    env, robot = load_env_and_robot(args.env, args.robot)

    # Solve problems
    # Load the task set
    try:
        task_set = pickle.load(open(f"{folder}/task_set.pkl", "rb"))
    except FileNotFoundError as e:
        print(e)
        print(
            f"Task set not found! "
            + "Generate the task set with generate_task_set.py."
        )
        robot.close()
        return

    # Convert task set to joint goal set
    joint_goal_set, results = convert_task_to_joint_goal(
        env, robot, task_set, args.ik, ik_max_attempts=20
    )

    # Save results
    np.save(f"{folder}/joint_goal_set_results_{suffix}.npy", results)
    pickle.dump(
        joint_goal_set, open(f"{folder}/joint_goal_set_{suffix}.pkl", "wb")
    )
    robot.close()


def parse_arguments():
    parser = argparse.ArgumentParser()
    # envs
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--env", choices=["table", "box", "cage", "shelf", "free"], default="table"
    )
    parser.add_argument(
        "--robot", choices=["panda", "ur10", "fetch"], default="panda"
    )
    parser.add_argument(
        "--ik", choices=["random", "neighbor", "grr"], default="neighbor"
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    set_seed(42)
    args = parse_arguments()

    main(args)
