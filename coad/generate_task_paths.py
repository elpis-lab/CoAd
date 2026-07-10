import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import time
import numpy as np
import pickle
from tqdm import tqdm

from coad.utils import set_seed, load_env_and_robot, get_data_folder
from coad.env import MujocoEnv
from coad.robot import MujocoRobot
from coad.planning import OMPLPlanner, euclidean_path_length

from collections import Counter

def solve_batch(
    env: MujocoEnv,
    robot: MujocoRobot,
    start,
    joint_goal_set,
    planner: OMPLPlanner,
    batch_time_budget=180.0,
):
    """Solve a joint goal set for a given robot at its home qpos."""
    # raise NotImplementedError("Batch planning not implemented yet")
    model, data = robot.model, robot.data
    average_batch_time = batch_time_budget / len(joint_goal_set)

    # Result containers
    plan_success = np.zeros(len(joint_goal_set), dtype=bool)
    batch_plan_success = np.zeros(len(joint_goal_set), dtype=bool)
    solve_times = np.zeros(len(joint_goal_set), dtype=float)
    total_plan_times = np.zeros(len(joint_goal_set), dtype=float)
    task_paths = {key: None for key in joint_goal_set.keys()}

    batch_solve_times = np.full(len(joint_goal_set), np.nan)
    batch_total_times = np.full(len(joint_goal_set), np.nan)

    fallback_total_times = np.full(len(joint_goal_set), np.nan)

    batch_path_lengths = np.full(len(joint_goal_set), np.nan)
    fallback_path_lengths = np.full(len(joint_goal_set), np.nan)

    # Initialize PRM* and RRTConnect Fallback planners
    batch_planner = planner
    individual_planner = OMPLPlanner(robot, data, planner="RRTConnect")

    # Initial batch planning phrase
    print("Building roadmap...")
    batch_planner.construct_roadmap(
        start,
        timeout=batch_time_budget
    )

    # Start solving
    pbar = tqdm(
        enumerate(joint_goal_set), total=len(joint_goal_set), unit="task"
    )
    for i, key in pbar:
        # Moving object (swept volume) to key pose
        env.move_swept_volume(key)

        # Solve planning problem
        if joint_goal_set[key] is not None:
            ik_goal = joint_goal_set[key]
            if robot.viewer is not None:
                robot.set_joint_qpos(ik_goal)
                robot.viewer.sync()

            # TODO Think of this better for obstacle avoidance in the future
            # for now, if failed, we will fall back to individual planner

            t0 = time.perf_counter()
            path = batch_planner.graph_query(start, ik_goal)
            t1 = time.perf_counter()
            batch_total_time = t1 - t0
            batch_planning_time = batch_total_time

            # error_counts[error_code] += 1
            batch_total_times[i] = batch_total_time
            batch_solve_times[i] = batch_planning_time

            if path is None or len(path) == 0:
                # if failed, fall back to individual planner
                batch_plan_success[i] = False

                pfs_path, total_pfs_time, planning_pfs_time = individual_planner.plan(
                    start=start,
                    goal=ik_goal,
                    timeout=3.0,
                    num_waypoints=200,
                    benchmark=True,
                )
                fallback_total_times[i] = total_pfs_time
                
                if pfs_path is not None and len(pfs_path) > 0:
                    
                    path = pfs_path
                    total_time = batch_total_time + total_pfs_time
                    planning_time = batch_planning_time + planning_pfs_time
                else:
                    path = None
                    total_time = batch_total_time + total_pfs_time
                    planning_time = batch_planning_time + planning_pfs_time

            else:

                total_time = batch_total_time
                planning_time = batch_planning_time
                batch_plan_success[i] = True

                # # Visualize batch planner path
                # robot.set_joint_qpos(path[0])
                # robot.viewer.sync()
                
                # input("Visualize batch planner path?")
                # for waypoint in path:
                #     robot.set_joint_qpos(waypoint)
                #     robot.viewer.sync()
                #     robot.in_contact(verbose=True)
                # input("Proceed?")

            if path is None or len(path) == 0:
                print(f"Planning failure for key: {key}")
                plan_success[i] = False
                task_paths[key] = None
            else:

                path_length = euclidean_path_length(path)

                if batch_plan_success[i]:
                    batch_path_lengths[i] = path_length
                else:
                    fallback_path_lengths[i] = path_length

                plan_success[i] = True
                task_paths[key] = path
            # solve_times[i] = planning_time + average_batch_time
            # total_plan_times[i] = total_time + average_batch_time
            solve_times[i] = planning_time
            total_plan_times[i] = total_time

        else:
            print(f"IK failure for key: {key}")
            plan_success[i] = False
            task_paths[key] = None
            solve_times[i] = np.nan
            total_plan_times[i] = np.nan

        # Update tqdm message periodically
        print_interval = 100
        if (i + 1) % print_interval == 0:
            success_mask = np.array(plan_success, dtype=bool)
            batch_success_mask = np.array(batch_plan_success, dtype=bool)
            fallback_mask = success_mask & (~batch_success_mask)

            m_solve = np.nanmean(np.array(solve_times)[success_mask])
            m_total = np.nanmean(np.array(total_plan_times)[success_mask])

            m_batch_solve = np.nanmean(batch_solve_times[batch_success_mask])
            m_batch_total = np.nanmean(batch_total_times[batch_success_mask])
            m_fallback_total = np.nanmean(fallback_total_times[fallback_mask])

            m_batch_path = np.nanmean(batch_path_lengths[batch_success_mask])
            m_fallback_path = np.nanmean(fallback_path_lengths[fallback_mask])

            tqdm.write(
                f"[{i+1}] "
                f"Plan Success: {np.sum(plan_success)/(i+1):.3f} | "
                f"Batch Plan Success: {np.sum(batch_plan_success)/(i+1):.3f} | "
                f"Total Planning Time: {m_total:.4f}s | "
                f"Successful Batch Total Time: {m_batch_total:.4f}s | "
                f"Fallback Total Time: {m_fallback_total:.4f}s | "
                f"Batch Path Length: {m_batch_path:.3f} | "
                f"Fallback Path Length: {m_fallback_path:.3f}"
            )
    
    results = np.stack(
        [plan_success, batch_plan_success, total_plan_times, batch_total_times, fallback_total_times, batch_path_lengths, fallback_path_lengths], axis=1
    )
    return task_paths, results


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
    # path_lengths = np.zeros(len(joint_goal_set), dtype=float)

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
            if path is None or len(path) == 0:
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
        print_interval = 100
        if (i + 1) % print_interval == 0:
            st = np.array(solve_times, dtype=float)
            tt = np.array(total_plan_times, dtype=float)
            pl = np.array(path_length, dtype=float)
            m_solve = np.nanmean(st[np.array(plan_success)])
            m_total = np.nanmean(tt[np.array(plan_success)])
            m_length = np.nanmean(pl[np.array(plan_success)])
            tqdm.write(
                f"[{i+1}] "
                f"Plan Success: {np.sum(plan_success)/(i+1):.3f} | "
                f"Planning Solving Time: {m_solve:.4f}s | "
                f"Total Planning Time: {m_total:.4f}s | "
                f"Path Length: {m_length:.4f}rad "
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
            env, robot, home_qpos, joint_goal_set, ompl_planner, 30
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
    env, robot = load_env_and_robot(args.env, args.robot, visualize=False)

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
        "--planner", choices=["RRTConnect", "PRMstar", "LazyPRMstar"], default="RRTConnect"
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    set_seed(42)
    args = parse_arguments()

    main(args)
