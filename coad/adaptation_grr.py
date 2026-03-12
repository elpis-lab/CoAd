"""
GRR Continuity Check

This check gurantees that the joint trajectory is not only
straight in joint space, but the end effector trajectory is also
"nearly straight" in the work space.

Expansion-GRR: Efficient Generation of Smooth
Global Redundancy Resolution Roadmaps
https://ieeexplore.ieee.org/document/10801917
"""

import numpy as np
from collections import deque

from coad.robot import MujocoRobot
from coad.mink_ik import IK
from geometry.pose import Pose


def joint_distance(q1, q2):
    return np.linalg.norm(q1 - q2)


def joint_interpolate(q1, q2, alpha):
    return (1.0 - alpha) * q1 + alpha * q2


def workspace_interpolate(p1, p2, alpha):
    pose1 = Pose(position=p1[:3], rotation=p1[3:])
    pose2 = Pose(position=p2[:3], rotation=p2[3:])
    pose = pose1.interpolate(pose2, alpha)
    return pose.flat()


def segment_continuity_check(
    robot: MujocoRobot, ik_solver: IK, q1, q2, epsilon=0.0, deviation=0.0
):
    """Check if two configurations follow the continuous constraints

    Linearly interpolate and bisectionally visit and check
    with the help of a queue to avoid stack overflow issues
    """
    # if error not defined
    if epsilon <= 0.0:
        epsilon = np.sqrt(robot.n_joints) * 0.05
    if deviation <= 0.0:
        deviation = 1.8

    # Get workspace points
    robot.set_joint_qpos(q1)
    p1 = robot.get_ee_pose()
    robot.set_joint_qpos(q2)
    p2 = robot.get_ee_pose()

    # Divide path segments for bisecting continuity check
    n_divs = int(np.ceil(joint_distance(q1, q2) / epsilon))

    queue = deque()
    queue.append((q1, q2, 0, n_divs + 1))
    while len(queue) > 0:
        qa, qb, ia, ib = queue.popleft()
        d = joint_distance(qa, qb)

        # If the two points are already adjacent
        if ib == ia + 1:
            continue

        # Find the middle point and configuration
        im = (ia + ib) // 2
        reached, qm = ik_solver.solve(
            workspace_interpolate(p1, p2, im / (n_divs + 1)),
            joint_interpolate(qa, qb, (im - ia) / (ib - ia)),
            use_col=False,
        )
        if not reached:
            return False

        # Check middle's deviation from "straight path"
        d1 = joint_distance(qa, qm)
        if d1 > deviation * d:
            return False
        d2 = joint_distance(qm, qb)
        if d2 > deviation * d:
            return False

        # Add to queue for further bisection
        queue.append((qa, qm, ia, im))
        queue.append((qm, qb, im, ib))
    return True
