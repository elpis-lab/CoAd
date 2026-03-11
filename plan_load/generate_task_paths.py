import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import time
import numpy as np
import pickle
from tqdm import tqdm

from plan_load.utils import set_seed, load_env_and_robot, get_data_folder
from plan_load.env import MujocoEnv
from plan_load.robot import MujocoRobot
from plan_load.planning import OMPLPlanner, euclidean_path_length


def solve_batch(
    env: MujocoEnv,
    robot: MujocoRobot,
    start,
    joint_goal_set,
    planner: OMPLPlanner,
    batch_time_budget=180.0,
):
    """Solve a joint goal set for a given robot at its home qpos."""
    raise NotImplementedError("Batch planning not implemented yet")
    model, data = robot.model, robot.data
    average_batch_time = batch_time_budget / len(joint_goal_set)

    # Result containers
    plan_success = np.zeros(len(joint_goal_set), dtype=bool)
    solve_times = np.zeros(len(joint_goal_set), dtype=float)
    total_plan_times = np.zeros(len(joint_goal_set), dtype=float)
    task_paths = {key: None for key in joint_goal_set.keys()}

    # Initial batch planning phrase
    print("Sampling for batch planning...")
    batch_planner.sample_for_batch_planning(
        start=start, timeout=batch_time_budget
    )

    # Start solving
    pbar = tqdm(
        enumerate(joint_goal_set), total=len(joint_goal_set), unit="task"
    )
    for i, key in pbar:
        # Moving object (swept volume) to key pose
        move_swept_volume(model, data, key)

        # Solve planning problem
        if joint_goal_set[key] is not None:
            ik_goal = joint_goal_set[key]
            if robot.viewer is not None:
                robot.set_joint_qpos(ik_goal)
                robot.viewer.sync()

            # TODO Think of this better for obstacle avoidance in the future
            # for now, if failed, we will fall back to individual planner
            path, total_time, planning_time = batch_planner.plan_batch(
                start=start,
                goal=ik_goal,
                timeout=3.0,
                num_waypoints=200,
                benchmark=True,
            )
            if not path:
                # if failed, fall back to individual planner
                path, total_time, planning_time = individual_planner.plan(
                    start=start,
                    goal=ik_goal,
                    timeout=3.0,
                    num_waypoints=200,
                    benchmark=True,
                )
                total_time -= average_batch_time
                planning_time -= average_batch_time

            if not path:
                print(f"Planning failure for key: {key}")
                plan_success[i] = False
                task_paths[key] = None
            else:
                plan_success[i] = True
                task_paths[key] = path
            solve_times[i] = planning_time + average_batch_time
            total_plan_times[i] = total_time + average_batch_time

        else:
            print(f"IK failure for key: {key}")
            plan_success[i] = False
            task_paths[key] = None
            solve_times[i] = np.nan
            total_plan_times[i] = np.nan

        # Update tqdm message periodically
        print_interval = 500
        if (i + 1) % print_interval == 0:
            st = np.array(solve_times, dtype=float)
            tt = np.array(total_plan_times, dtype=float)
            m_solve = np.nanmean(st[np.array(plan_success)])
            m_total = np.nanmean(tt[np.array(plan_success)])
            tqdm.write(
                f"[{i+1}] "
                f"Plan Success: {np.sum(plan_success)/(i+1):.3f} | "
                f"Planning Solving Time: {m_solve:.4f}s | "
                f"Total Planning Time: {m_total:.4f}s"
            )
    return task_paths


def solve_individual(
    env: MujocoEnv,
    robot: MujocoRobot,
    start,
    joint_goal_set,
    planner: OMPLPlanner,
):
    """Solve a joint goal set for a given robot at its home qpos."""
    model, data = robot.model, robot.data

    # Result containers
    plan_success = np.zeros(len(joint_goal_set), dtype=bool)
    solve_times = np.zeros(len(joint_goal_set), dtype=float)
    total_plan_times = np.zeros(len(joint_goal_set), dtype=float)
    path_length = np.zeros(len(joint_goal_set), dtype=float)
    task_paths = {key: None for key in joint_goal_set.keys()}

    # Start solving
    pbar = tqdm(enumerate(joint_goal_set), total=len(joint_goal_set))
    for i, key in pbar:
        # Moving object (swept volume) to key pose
        env.move_swept_volume(key)

        # Solve planning problem
        if joint_goal_set[key] is not None:
            ik_goal = joint_goal_set[key]
            if robot.viewer is not None:
                robot.set_joint_qpos(ik_goal)
                robot.viewer.sync()

            path, total_time, planning_time = planner.plan(
                start=start,
                goal=ik_goal,
                timeout=3.0,
                num_waypoints=200,
                benchmark=True,
            )
            if path is None:
                print(f"Planning failure for key: {key}")
                plan_success[i] = False
                task_paths[key] = None
                path_length[i] = np.nan
            else:
                plan_success[i] = True
                task_paths[key] = path
                path_length[i] = euclidean_path_length(path)
            solve_times[i] = planning_time
            total_plan_times[i] = total_time

        else:
            print(f"No valid joint goal for key: {key}")
            plan_success[i] = False
            task_paths[key] = None
            solve_times[i] = np.nan
            total_plan_times[i] = np.nan

        # Update tqdm message periodically
        print_interval = 500
        if (i + 1) % print_interval == 0:
            st = np.array(solve_times, dtype=float)
            tt = np.array(total_plan_times, dtype=float)
            m_solve = np.nanmean(st[np.array(plan_success)])
            m_total = np.nanmean(tt[np.array(plan_success)])
            tqdm.write(
                f"[{i+1}] "
                f"Plan Success: {np.sum(plan_success)/(i+1):.3f} | "
                f"Planning Solving Time: {m_solve:.4f}s | "
                f"Total Planning Time: {m_total:.4f}s"
            )

    results = np.stack(
        [plan_success, solve_times, total_plan_times, path_length], axis=1
    )
    return task_paths, results


def solve_joint_goal_set(
    env: MujocoEnv,
    robot: MujocoRobot,
    joint_goal_set,
    planner,
):
    """Solve a joint goal set for a given robot at its home qpos."""
    # Robot
    model, data = robot.model, robot.data
    home_qpos = robot.get_joint_qpos()

    # Planner
    ompl_planner = OMPLPlanner(robot, data, planner=planner)

    # Solve problems
    if "PRM" in planner:
        solution_paths, results = solve_batch(
            env, robot, home_qpos, joint_goal_set, ompl_planner, 180
        )
    else:
        solution_paths, results = solve_individual(
            env, robot, home_qpos, joint_goal_set, ompl_planner
        )
    return solution_paths, results


def main(args):
    """Generate a dataset of task paths for a given environment and robot."""
    folder = get_data_folder(args.env, args.robot)
    suffix = f"{args.ik}_{args.planner}"

    # Check if graph data is already generated
    data_exists = os.path.exists(f"{folder}/task_paths_data_{suffix}.npy")
    if data_exists and not args.overwrite:
        print(
            f"Task paths already exists at {folder} "
            + f"with IK '{args.ik}' and planner '{args.planner}'. "
            + "Use --overwrite to regenerate the task paths."
        )
        return

    # Load environment and robot
    env, robot = load_env_and_robot(args.env, args.robot)

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

    # Solve problems converted in joint space
    task_paths, results = solve_joint_goal_set(
        env, robot, joint_goal_set, args.planner
    )
    # split data and keys to two separate files
    keys = list(task_paths.keys())
    data = np.array([path for path in task_paths.values()], dtype=object)

    # Save results
    np.save(f"{folder}/task_paths_results_{suffix}.npy", results)
    np.save(f"{folder}/task_paths_data_{suffix}.npy", data)
    pickle.dump(keys, open(f"{folder}/task_paths_keys_{suffix}.pkl", "wb"))
    robot.close()


def parse_arguments():
    parser = argparse.ArgumentParser()
    # envs
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--env",
        choices=["table", "cage", "shelf", "free", "real"],
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

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    set_seed(42)
    args = parse_arguments()

    main(args)
