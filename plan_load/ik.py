import numpy as np
import time

from plan_load.robot import MujocoRobot, Panda, UR10
from plan_load.mink_ik import IK, PandaIK, UR10IK
from plan_load.pose import Pose, matrix_to_quat, quat_to_matrix
from plan_load.mujoco_utils import geoms_in_contact, sample_qpos


def make_ik_solver(
    robot: MujocoRobot, default_q=None, env_collision_geoms=None
):
    """Build an IK solver from robot; extend for other robot types as needed."""
    if isinstance(robot, Panda):
        ik_class = PandaIK
    elif isinstance(robot, UR10):
        ik_class = UR10IK
    else:
        raise ValueError("No IK solver for robot type.")

    if default_q is None:
        default_q = np.asarray(robot.data.qpos.copy(), dtype=float)
    return ik_class(
        robot.model,
        ee_site=robot.ee_name,
        default_q=default_q,
        env_collision_geoms=env_collision_geoms,
    )


def get_neighbor_ref(
    nn, key, task_keys: np.ndarray, joint_goal_set: dict, k=5
):
    """Get the nearest neighbor reference for a task."""
    # Get the SE2 pose of the task
    key_pose = [
        (key[0][0] + key[0][1]) / 2,
        (key[1][0] + key[1][1]) / 2,
        (key[3][0] + key[3][1]) / 2,
    ]

    # Get the nearest neighbor (need to get more for better accuracy)
    neighbors = nn.query(
        [key_pose],
        k * 10,
        return_distance=True,
        sort_results=True,
    )[1][0]
    neighbors = neighbors[:k]

    # See if neighbors have solutions
    keys = task_keys[neighbors]
    # conver to tuple
    keys = [tuple(tuple(v for v in row) for row in key) for key in keys]
    configs = []
    for key in keys:
        if joint_goal_set[key] is not None:
            configs.append(joint_goal_set[key])
    if len(configs) == 0:
        return None

    # Get the mean configuration
    config = np.mean(configs, axis=0)
    return config


def get_joint_goal(
    robot: MujocoRobot,
    task_set,
    key,
    ee_offset,
    ik_solver: IK,
    ik_max_attempts=20,
    grasp_idx=0,
    viewer=None,
    pos_tol=1e-3,
    rot_tol=1e-2,
    benchmark=False,
    nn=None,
    joint_goal_set=None,
    task_keys=None,
):
    """"""
    # Robot
    model, data = robot.model, robot.data

    # Get current task
    if grasp_idx != 0:
        raise NotImplementedError("Only top grasp is supported.")
    task = task_set[key][grasp_idx]
    ee_offset_pose = Pose(ee_offset[:3, 3], matrix_to_quat(ee_offset[:3, :3]))

    # Object pose as center of task bounds
    s = (task[:, 0] + task[:, 1]) / 2
    obj_pose = Pose(s[:3], s[3:6])
    target_pose = obj_pose @ ee_offset_pose

    # Start solving IK
    t0 = time.perf_counter()
    valid_ik = False
    goal = None
    for ik_attempts in range(ik_max_attempts):
        # iterate through 4 directional grasps
        rot_offset = (ik_attempts * 4 // ik_max_attempts) * (np.pi / 2)
        target = rotate_pose_about_world_z(
            target_pose, rot_offset, obj_pose.position
        )
        target = np.concatenate([target.position, target.rotation])

        # For each grasp, try the following in order:
        seed = robot.data.qpos.copy()
        # 0, try nearby solution
        if ik_attempts % (ik_max_attempts // 4) == 0:
            neighbor_ref = get_neighbor_ref(nn, key, task_keys, joint_goal_set)
            # neighbor_ref = None
            if neighbor_ref is None:
                neighbor_ref = robot.get_joint_qpos().copy()
            seed[robot.joint_ids] = neighbor_ref
        # 1, try with previous solution
        # elif ik_attempts % (ik_max_attempts // 4) == 1:
        #     seed[robot.joint_ids] = robot.get_joint_qpos().copy()
        # 2, try with home position
        elif ik_attempts % (ik_max_attempts // 4) == 1:
            seed[robot.joint_ids] = ik_solver.default_q[robot.joint_ids]
        # 3, try with random position for all the rest
        else:
            t_random = sample_qpos(model, robot.joint_ids)
            seed[robot.joint_ids] = t_random

        # Solve IK
        reached, solution = ik_solver.solve(
            target,
            seed,
            use_col=False,
            pos_tol=pos_tol,
            rot_tol=rot_tol,
        )
        if solution is not None:
            goal = solution.copy()
            robot.set_joint_qpos(solution)
        # Update viewer
        if viewer is not None:
            viewer.sync()

        if reached:
            valid_ik = not robot.in_contact()
            if valid_ik:
                break

    if benchmark:
        ik_solve_time = time.perf_counter() - t0
        return valid_ik, goal, ik_solve_time
    return valid_ik, goal


def rotate_pose_about_world_z(pose: Pose, theta, p_center):
    """
    Rotate pose position and orientation
    about world Z axis by thetathrough p_center.
    """
    # to pivot
    t_p = Pose(p_center)
    # rotate around z axis
    t_r = Pose(rotation=np.array([0, 0, theta]))
    # conjugate to rotate about pivot point in world coords
    t_world = t_p @ t_r @ t_p.invert()
    return t_world @ pose
