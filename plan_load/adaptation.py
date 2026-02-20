import mujoco
import numpy as np
from tqdm import tqdm
from collections import deque

from plan_load.mink_ik import IK
from plan_load.robot import MujocoRobot
from plan_load.pose import Pose


class Adapter:
    def __init__(self, robot: MujocoRobot, ik_solver: IK):
        self.robot = robot
        self.ik_solver = ik_solver

    def compress(self, center_path, nb_path, ik_for_new_end=True):
        """Compress the nb_path into the center_path"""
        raise NotImplementedError("Compress method not implemented")

    def adapt(self, center_path, q_goal):
        """Adapt the center_path for the q_goal"""
        raise NotImplementedError("Adapt method not implemented")

    def use_ik_for_new_end(self, center_path, nb_path):
        """
        Use IK to find the new end goal.
        hopefully, it can be closer to the center path's end
        """
        q_goal = nb_path[-1]

        # Recover task
        self.robot.set_joint_qpos(q_goal)
        target = self.robot.get_ee_pose()
        # Solve IK
        reached, q_nb_end = self.ik_solver.solve(
            target,
            current=center_path[-1],
            use_col=False,
        )
        if reached:
            q_goal = q_nb_end
        return reached, q_goal


class LinearAdapter(Adapter):
    def compress(self, center_path, nb_path, ik_for_new_end=True):
        """Compress the nb_path into the center_path"""
        # Self, keep
        if center_path is nb_path:
            return True, center_path, center_path[-1]

        # Get end goal for neighbor
        q_goal = nb_path[-1]
        if ik_for_new_end:
            reached, q = self.use_ik_for_new_end(center_path, nb_path)
            if reached:
                q_goal = q

        # Direct interpolation
        # Collision check
        if not segment_validity_check(self.robot, center_path[-1], q_goal):
            return False, center_path, None
        return True, center_path, q_goal

    def adapt(self, center_path, q_goal):
        """Adapt the center_path to the q_goal"""
        # Append q_goal to center_path
        if isinstance(center_path, np.ndarray):
            center_path = np.append(center_path, q_goal)
        else:
            center_path.append(list(q_goal))
        return center_path


class GRRAdapter(Adapter):
    def compress(self, center_path, nb_path, ik_for_new_end=True):
        """Compress the nb_path into the center_path"""
        # Self, keep
        if center_path is nb_path:
            return True, center_path, center_path[-1]

        # Get end goal for neighbor
        q_goal = nb_path[-1]  # original end goal
        if ik_for_new_end:
            reached, q = self.use_ik_for_new_end(center_path, nb_path)
            if reached:
                q_goal = q

        # GRR interpolation
        # Collision check
        if not segment_validity_check(self.robot, center_path[-1], q_goal):
            return False, center_path, None
        # GRR continuity check
        if not segment_continuity_check(
            self.robot, self.ik_solver, center_path[-1], q_goal
        ):
            return False, center_path, None

        return True, center_path, q_goal

    def adapt(self, center_path, q_goal):
        """Adapt the center_path to the q_goal"""
        pass


class DMPAdapter(Adapter):
    pass


def joint_distance(q1, q2):
    return np.linalg.norm(q1 - q2)


def joint_interpolate(q1, q2, alpha):
    return (1.0 - alpha) * q1 + alpha * q2


def workspace_interpolate(p1, p2, alpha):
    pose1 = Pose(position=p1[:3], rotation=p1[3:])
    pose2 = Pose(position=p2[:3], rotation=p2[3:])
    pose = pose1.interpolate(pose2, alpha)
    return pose.flat()


def segment_continuity_check(robot: MujocoRobot, ik_solver: IK, q1, q2):
    """Check if two configurations follow the continuous constraints

    Linearly interpolate and bisectionally visit and check
    with the help of a queue to avoid stack overflow issues
    """
    epsilon = np.sqrt(robot.n_joints) * 0.05
    deviation = 1.8
    robot.set_joint_qpos(q1)
    p1 = robot.get_ee_pose()
    robot.set_joint_qpos(q2)
    p2 = robot.get_ee_pose()

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


def segment_validity_check(
    robot: MujocoRobot,
    q_start: np.ndarray,
    q_end: np.ndarray,
    max_step: float = 0.05,
) -> bool:
    """
    Check if straight-line interpolation between q_start and q_end is
    collision free for the robot (including environment).
    """
    q_start = np.asarray(q_start, dtype=float)
    q_end = np.asarray(q_end, dtype=float)
    diff = q_end - q_start

    # Use infinity norm so that per-joint step is bounded by max_step
    dist = np.linalg.norm(diff, ord=np.inf)
    n_steps = max(int(np.ceil(dist / max_step)) + 1, 2)

    # Per-step collision check
    for alpha in np.linspace(0.0, 1.0, n_steps):
        q = (1.0 - alpha) * q_start + alpha * q_end
        robot.set_joint_qpos(q)
        if robot.in_contact():
            return False
    return True
