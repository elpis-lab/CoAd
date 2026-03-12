import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import numpy as np
import pickle

from plan_load.env import MujocoEnv, RealEnv
from plan_load.robot import MujocoRobot, UR10
from plan_load.mink_ik import get_ik_solver

from plan_load.adaptation import LinearAdapter, GRRAdapter
from plan_load.adaptation import DMPAdapter, TrajOptAdapter
from plan_load.utils import set_seed, load_env_and_robot, get_data_folder
from plan_load.planning import OMPLPlanner
from geometry.trajectory import Trajectory, SplineTrajectory, TOPPRATrajectory

from real_world.physical_robot import PhysicalUR10
from experiments.benchmark_baselines import BoxGrid
from experiments.benchmark_baselines import Library


def load_library_and_adapter(
    env_name="real",
    robot_name="ur10",
    ik_used="neighbor",
    planner_used="RRTConnect",
    adaptation="grr",
    n_neighbors=1000,
):
    """Load library and adapter"""
    folder = get_data_folder(env_name, robot_name)
    suffix = f"{ik_used}_{planner_used}_{adaptation}_{n_neighbors}"

    # Get library and solution paths
    root_path = f"{folder}/root_paths_{suffix}.pkl"
    map_path = f"{folder}/key_to_root_{suffix}.pkl"
    root_data = pickle.load(open(root_path, "rb"))
    map_data = pickle.load(open(map_path, "rb"))
    d_name = f"{folder}/task_paths_data_{ik_used}_{planner_used}.npy"
    data = np.load(d_name, allow_pickle=True)
    k_name = f"{folder}/task_paths_keys_{ik_used}_{planner_used}.pkl"
    keys = pickle.load(open(k_name, "rb"))
    task_paths = {key: data for key, data in zip(keys, data)}

    # Load environment and robot
    env, robot = load_env_and_robot(env_name, robot_name)
    ik_solver = get_ik_solver(robot, env_collision_geoms=env.collision_geoms)

    # Get adapter
    indexer = BoxGrid(map_data)
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

    return indexer, map_data, root_data, adapter, env, robot, task_paths


def execute_sim(robot: MujocoRobot, waypoints):
    """Execute a trajectory in simulation"""
    for q in waypoints:
        robot.set_joint_qpos(q)
        robot.viewer.sync()
        time.sleep(0.008)


def execute_path(robot: MujocoRobot, real_robot: PhysicalUR10, path):
    """Execute a path"""
    # Execute path
    # traj_time = 5.0
    # time_states = np.linspace(0, traj_time, path.shape[0])
    # trajectory = SplineTrajectory(path, time_states)
    trajectory = TOPPRATrajectory(path, 2.0, 1.0)

    input("Run Sim?")
    waypoints = trajectory.to_step_waypoints(dt=0.008, type="p")
    execute_sim(robot, waypoints)

    input("Run Real?")
    real_robot.execute_trajectory(waypoints)

    # Move gripper down
    ee_pose = real_robot.get_ee_pose()
    z_offset = -0.02
    new_ee_pose = ee_pose.copy()
    new_ee_pose[2] += z_offset
    real_robot.move_tool(new_ee_pose)

    # Close gripper
    real_robot.control_gripper("close")
    time.sleep(2.0)

    # Execute reversed path
    path_reversed = path[::-1]
    trajectory_reversed = TOPPRATrajectory(path_reversed, 2.0, 1.0)
    waypoints_reversed = trajectory_reversed.to_step_waypoints(
        dt=0.008, type="p"
    )
    real_robot.execute_trajectory(waypoints_reversed)

    # Open gripper
    real_robot.control_gripper("open")
    time.sleep(2.0)


def main(method="adaptation", adaptation_method="grr"):
    """Main function"""
    # Load library and adapter
    indexer, map_data, root_data, adapter, env, robot, task_paths = (
        load_library_and_adapter(adaptation=adaptation_method)
    )
    data = robot.data
    home_qpos = np.array([2.0309, -1.095, 1.5799, -2.071, -1.5938, 0.5060])
    robot.set_joint_qpos(home_qpos)
    robot.viewer.sync()

    # Load planner if it is not adaptation
    if method != "adaptation":
        ompl_planner = OMPLPlanner(robot, data)

    # Physical robot
    real_robot = PhysicalUR10()
    real_robot.control_gripper("open")
    time.sleep(1.0)

    # Object
    object_height = list(map_data.keys())[0][2][0]
    object_intervals = [[-0.35, 0.07], [-1.02, -0.7]]

    # Start procedure
    try:
        while True:
            input("Start packing?")

            # Detect objects
            object_poses = real_robot.get_object_pose_top()
            # get object IDs that are within the specified bounds
            object_ids_in_bounds = []
            for obj_id, pose in object_poses.items():
                x, y = pose[0, 3], pose[1, 3]
                if not (
                    object_intervals[0][0] <= x <= object_intervals[0][1]
                    and object_intervals[1][0] <= y <= object_intervals[1][1]
                    and obj_id in [8, 9, 10]
                ):
                    continue
                object_ids_in_bounds.append(obj_id)

            # Select object
            if len(object_ids_in_bounds) == 0:
                print("No objects detected within bounds. Finished.")
                continue
            else:
                object_id = 0
            pose = object_poses[object_id]

            # Query
            pose_to_query = [pose[0, 3], pose[1, 3], object_height, 0.0]
            key = indexer.query_point(pose_to_query)
            if key is None:
                print("No key found for the given pose.")
                continue
            root_id, goal_q = map_data[key]

            # Get solution path
            if method == "adaptation":
                t0 = time.perf_counter()
                root_path = root_data[root_id]
                path = adapter.adapt(root_path, goal_q)
                t1 = time.perf_counter()

                print(f"Adaptation time: {(t1 - t0)*1000:.4f} ms")
                print("Adapted path size:", path.shape)

            elif method == "rrtconnect":
                path, total_time, planning_time = ompl_planner.plan(
                    start=home_qpos,
                    goal=goal_q,
                    timeout=3.0,
                    num_waypoints=200,
                    benchmark=True,
                    smooth_path=False,
                )

                print(f"Planning time: {total_time*1000:.4f} ms")
                if not path:
                    print("Planning failed.")
                    continue

            elif method == "library":
                solved_keys = [
                    k for k, path in task_paths.items() if path is not None
                ]
                library = Library(
                    len(solved_keys),
                    env,
                    robot,
                    home_qpos,
                    map_data,
                    solved_keys,
                    task_paths,
                    data,
                )
                path, library_time, lib_query_success = library.solve(
                    pose_to_query
                )

                print(f"Library query time: {library_time*1000:.4f} ms")
                if not lib_query_success:
                    print("Library query failed.")
                    continue

            else:
                print("No key found for the given pose.")
                continue

            # Execute path
            execute_path(real_robot, path)

    finally:
        robot.close()
        real_robot.close()


if __name__ == "__main__":
    set_seed(42)
    main(method="adaptation", adaptation_method="grr")
