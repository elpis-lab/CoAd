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

    def build_center(self, center_path):
        """Build the adapter center for using the given center path"""
        raise NotImplementedError("Build center method not implemented")

    def compress(self, center, nb_path, ik_for_new_end=True):
        """Compress the nb_path into the adapter center"""
        raise NotImplementedError("Compress method not implemented")

    def adapt(self, center, q_goal):
        """Adapt the adapter center for the q_goal"""
        raise NotImplementedError("Adapt method not implemented")


class LinearAdapter(Adapter):
    def build_center(self, center_path):
        """
        Build the adapter center for using the given center path
        For this method, the adapter center is the same as the center path
        """
        return center_path, center_path[-1]

    def compress(self, center, nb_path, ik_for_new_end=True):
        """Compress the nb_path into the center_path"""
        # Get end goal for neighbor
        q_goal = nb_path[-1]
        if ik_for_new_end:
            reached, q = self.use_ik_for_new_end(center, nb_path)
            if reached:
                q_goal = q

        # Direct interpolation
        # Collision check
        if not segment_validity_check(self.robot, center[-1], q_goal):
            return False, None
        return True, q_goal

    def adapt(self, center, q_goal):
        """Adapt the center_path to the q_goal"""
        # Append q_goal to center_path
        if isinstance(center_path, np.ndarray):
            center_path = np.append(center_path, q_goal)
        else:
            center_path.append(list(q_goal))
        return center_path

    def use_ik_for_new_end(self, center, nb_path):
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
            target, current=center[-1], use_col=False
        )
        if reached:
            q_goal = q_nb_end
        return reached, q_goal


class GRRAdapter(LinearAdapter):
    def compress(self, center, nb_path, ik_for_new_end=True):
        """Compress the nb_path into the center_path"""
        # Get end goal for neighbor
        q_goal = nb_path[-1]  # original end goal
        if ik_for_new_end:
            reached, q = self.use_ik_for_new_end(center, nb_path)
            if reached:
                q_goal = q

        # GRR interpolation
        # Collision check
        if not segment_validity_check(self.robot, center[-1], q_goal):
            return False, None
        # GRR continuity check
        if not segment_continuity_check(
            self.robot, self.ik_solver, center[-1], q_goal
        ):
            return False, None

        return True, q_goal


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
