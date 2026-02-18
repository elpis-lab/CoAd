import numpy as np
import time
import pickle
from tqdm import tqdm

from sklearn.neighbors import BallTree

from helpers.mujoco_utils import move_swept_volume

from plan_load.ik import get_joint_goal, make_ik_solver
from plan_load.robot import MujocoRobot
from plan_load.planning import OMPLPlanner
from plan_load.pose import angle_diff


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


def se2_distance(p1, p2, position_weight=1.0, rotation_weight=0.5):
    """Compute the distance between two workspace SE3 points."""
    # Position component
    d_position = ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
    # Rotation component
    d_rotation = abs(angle_diff(p1[2], p2[2]))
    return position_weight * d_position + rotation_weight * d_rotation


def build_task_nn(keys):
    """Build a Ball tree for finding the nearest bin."""
    if isinstance(keys, dict):
        bins = list(keys.keys())
    else:
        bins = np.asarray(keys)

    # Represent bin as a SE2 pose
    bin_poses = (bins[:, :, 0] + bins[:, :, 1]) / 2
    bin_poses = bin_poses[:, [0, 1, 3]]
    nn = BallTree(bin_poses, metric=se2_distance)
    return nn, bin_poses


def convert_task_to_joint_goal(
    robot: MujocoRobot,
    task_set,
    ee_offset,
    ik_solver,
    ik_max_attempts=20,
    grasp_idx=0,
):
    """Convert task set to joint goal set."""
    model = robot.model
    data = robot.data

    # Build a Ball tree for finding the nearest set
    nn, bin_poses = build_task_nn(task_set)

    # Container
    ik_success = np.zeros(len(task_set), dtype=bool)
    ik_times = np.zeros(len(task_set), dtype=float)
    joint_goal_set = {key: None for key in task_set.keys()}
    task_keys = np.array(list(task_set.keys()))

    # Start sovling IK ony by one
    pbar = tqdm(enumerate(task_set), total=len(task_set), unit="task")
    for i, key in pbar:
        # Moving object (swept volume) to key pose
        move_swept_volume(model, data, key)
        # time.sleep(0.01)

        # Solve IK
        valid_ik, ik_goal, ik_solve_time = get_joint_goal(
            robot,
            task_set,
            key,
            ee_offset,
            ik_solver,
            ik_max_attempts,
            grasp_idx,
            viewer=robot.viewer,
            benchmark=True,
            nn=nn,
            joint_goal_set=joint_goal_set,
            task_keys=task_keys,
        )
        ik_times[i] = ik_solve_time
        ik_success[i] = valid_ik
        # only store the valid IK
        if valid_ik:
            joint_goal_set[key] = ik_goal

        # Update tqdm message periodically
        print_interval = 500
        if (i + 1) % print_interval == 0:
            m_ik = np.nanmean(ik_times[np.array(ik_success)])
            tqdm.write(
                f"[{i+1}] "
                f"IK Success: {np.sum(ik_success)/(i+1):.3f} | "
                f"IK Time: {m_ik:.4f}s | "
            )

    return joint_goal_set, ik_times


def solve_batch(
    robot: MujocoRobot,
    start,
    joint_goal_set,
    individual_planner: OMPLPlanner,
    batch_planner: OMPLPlanner,
    batch_time_budget=180.0,
):
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
        if i > 3300:
            break
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
    robot: MujocoRobot,
    start,
    joint_goal_set,
    planner: OMPLPlanner,
):
    model, data = robot.model, robot.data

    # Result containers
    plan_success = np.zeros(len(joint_goal_set), dtype=bool)
    solve_times = np.zeros(len(joint_goal_set), dtype=float)
    total_plan_times = np.zeros(len(joint_goal_set), dtype=float)
    task_paths = {key: None for key in joint_goal_set.keys()}

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

            path, total_time, planning_time = planner.plan(
                start=start,
                goal=ik_goal,
                timeout=3.0,
                num_waypoints=200,
                benchmark=True,
            )
            if not path:
                print(f"Planning failure for key: {key}")
                plan_success[i] = False
                task_paths[key] = None
            else:
                plan_success[i] = True
                task_paths[key] = path
            solve_times[i] = planning_time
            total_plan_times[i] = total_time

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


def solve_task_set(
    robot: MujocoRobot,
    task_set,
    ee_offset,
    ik_max_attempts=20,
    planner="RRTConnect",
    env_collision_geoms=None,
):
    """Solve a task set for a given robot at its home qpos."""
    # IK
    ik_solver = make_ik_solver(robot, None, env_collision_geoms)
    # Planner
    planner = OMPLPlanner(robot, robot.data, planner=planner, log=False)
    # Robot
    model, data = robot.model, robot.data
    home_qpos = robot.get_joint_qpos()

    # Run IK to convert all task-space problems to joint-space problems
    print("Converting task set to joint goal set...")
    # joint_goal_set, ik_times = convert_task_to_joint_goal(
    #     robot, task_set, ee_offset, ik_solver, ik_max_attempts, grasp_idx=0
    # )
    # with open("joint_goal_set_near.pkl", "wb") as f:
    #     pickle.dump(joint_goal_set, f)
    joint_goal_set = pickle.load(open("joint_goal_set_near.pkl", "rb"))

    # TODO: Test script to run RRT* instead of RRTConnect to build graph
    batch_planner = OMPLPlanner(
        robot, robot.data, planner="PRMstar", log=False
    )
    task_paths = solve_batch(
        robot, home_qpos, joint_goal_set, planner, batch_planner
    )
    # task_paths = solve_individual(robot, home_qpos, joint_goal_set, planner)
    return task_paths
