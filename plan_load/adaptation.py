import mujoco
import time
import numpy as np
from dataclasses import dataclass

from plan_load.robot import MujocoRobot
from plan_load.mink_ik import IK

from plan_load.adaptation_grr import segment_continuity_check
from plan_load.adaptation_dmp import DMPDiscete, DMP
from plan_load.adaptation_opt import TrajOpt


class Adapter:
    def __init__(self, robot: MujocoRobot, ik_solver: IK):
        self.robot = robot
        self.ik_solver = ik_solver

    def build_center(self, center_path):
        """Build the adapter center for using the given center path"""
        raise NotImplementedError("Build center method not implemented")

    def compress(self, center, nb_path, goal_refinement=True):
        """Compress the nb_path into the adapter center"""
        raise NotImplementedError("Compress method not implemented")

    def adapt(self, center, q_goal):
        """Adapt the adapter center for the q_goal"""
        raise NotImplementedError("Adapt method not implemented")

    def segment_validity_check(
        self, q_start: np.ndarray, q_end: np.ndarray, step_size: float = 0.05
    ) -> bool:
        """
        Check if straight-line interpolation between q_start and q_end is
        collision free for the robot (including environment).
        """
        q_start = np.asarray(q_start)
        q_end = np.asarray(q_end)
        diff = q_end - q_start

        # Use infinity norm so that per-joint step is bounded by step_size
        dist = np.linalg.norm(diff, ord=np.inf)
        n_steps = max(int(np.ceil(dist / step_size)) + 1, 2)

        # Per-step collision check
        for alpha in np.linspace(0.0, 1.0, n_steps):
            q = (1.0 - alpha) * q_start + alpha * q_end
            self.robot.set_joint_qpos(q)
            if self.robot.in_contact():
                return False
        return True

    def path_validity_check(self, path):
        """Check validity of the whole path"""
        for i in range(len(path) - 1):
            q1 = path[i]
            q2 = path[i + 1]
            if not self.segment_validity_check(q1, q2):
                return False
        return True

    def ik_refinement(self, ref_joint, curr_goal_joint):
        """
        Use IK to find the new end goal.
        hopefully, it can be closer to the center path's end
        """
        # Recover task
        self.robot.set_joint_qpos(curr_goal_joint)
        target = self.robot.get_ee_pose()
        # Solve IK
        reached, new_goal_joint = self.ik_solver.solve(
            target, current=ref_joint, use_col=False
        )
        return reached, new_goal_joint


class LinearAdapter(Adapter):
    def build_center(self, center_path):
        """
        Build the adapter center for using the given center path
        For this method, the adapter center is the same as the center path
        """
        return center_path, center_path[-1]

    def compress(self, center, nb_path, goal_refinement=True):
        """Compress the nb_path into the center_path"""
        # Get end goal for neighbor
        q_goal = nb_path[-1]
        if goal_refinement:
            reached, q = self.ik_refinement(center[-1], nb_path[-1])
            if reached:
                q_goal = q

        # Direct interpolation
        # Collision check
        path = self.adapt(center, q_goal)
        valid = self.path_validity_check(path)
        if not valid:
            return False, q_goal

        return True, q_goal

    def adapt(self, center, q_goal):
        """Adapt the center_path to the q_goal"""
        # Append q_goal to center_path with linear interpolation
        q_start = center[-1]
        alphas = np.linspace(0.0, 1.0, 10)[:, None]  # shape (N,1)
        interp = (1.0 - alphas) * q_start + alphas * q_goal
        path = np.concatenate([center, interp], axis=0)
        return path


class GRRAdapter(LinearAdapter):
    def compress(self, center, nb_path, goal_refinement=True):
        """Compress the nb_path into the center_path"""
        # Get end goal for neighbor
        q_goal = nb_path[-1]  # original end goal
        if goal_refinement:
            reached, q = self.ik_refinement(center[-1], nb_path[-1])
            if reached:
                q_goal = q

        # GRR interpolation
        # Collision check
        path = self.adapt(center, q_goal)
        valid = self.path_validity_check(path)
        if not valid:
            return False, q_goal
        # GRR continuity check
        if not segment_continuity_check(
            self.robot, self.ik_solver, center[-1], q_goal
        ):
            return False, q_goal

        return True, q_goal


@dataclass
class DMPCenter:
    """Compact center representation: DMP params + learned weights."""

    dmp: DMP  # DMP instance
    start: np.ndarray  # (n_joints,)
    goal: np.ndarray  # (n_joints,)
    timesteps: int  # number of trajectory timesteps


class DMPAdapter(Adapter):
    """
    Center = learned DMP weights/etc from center_path.
    Compression of neighbor = store only goal (optionally IK-refined),
    as long as DMP(center, new_goal) is feasible.
    """

    def __init__(
        self,
        robot: MujocoRobot,
        ik_solver: IK,
        # DMP parameters
        n_bfs=50,
        ay=None,
        by=None,
    ):
        super().__init__(robot, ik_solver)
        self.n_bfs = int(n_bfs)
        self.ay = ay
        self.by = by

    def build_center(self, center_path):
        """
        Fit a discrete DMP per joint to center_path (T, n_joints).
        Return (center_obj, center_goal).

        To avoid issues of force being too large
        when start and goal are too close,
        We instead fit DMP on the residual minus a traight line trend
        """
        center_path = np.asarray(center_path)
        steps, n_joints = center_path.shape

        # dmp = DMPDiscete(
        #     n_dmps=n_joints,
        #     n_bfs=self.n_bfs,
        #     dt=1 / (steps - 1),
        #     ay=None if self.ay is None else self.ay * np.ones(n_joints),
        #     by=None if self.by is None else self.by * np.ones(n_joints),
        # )
        # # expects y_des with shape (n_dmps, T)
        # y_des = center_path.T
        # dmp.imitate_path(y_des=y_des)

        # # DMP "center"
        # start = center_path[0].copy()
        # goal = center_path[-1].copy()
        # center = DMPCenter(dmp, start, goal, steps)
        # return center, goal

        y0 = center_path[0].copy()
        g = center_path[-1].copy()

        # linear trend
        s = np.linspace(0.0, 1.0, steps)[:, None]  # (steps, 1)
        line = y0[None, :] + s * (g - y0)[None, :]  # (steps, n_joints)
        # residual
        res = center_path - line  # (steps, n_joints)

        # train DMP on residuals, with y0=0, goal=0
        dmp = DMPDiscete(
            n_dmps=n_joints,
            n_bfs=self.n_bfs,
            dt=1 / (steps - 1),
            ay=None if self.ay is None else self.ay * np.ones(n_joints),
            by=None if self.by is None else self.by * np.ones(n_joints),
        )
        dmp.imitate_path(y_des=res.T)  # shape (D, T)

        # store y0, goal and dmp
        center = DMPCenter(dmp=dmp, start=y0, goal=g, timesteps=steps)
        return center, g

    def compress(self, center: DMPCenter, nb_path, goal_refinement=True):
        """
        Return (ok, q_goal).
        If ok=True, the neighbor is represented only by q_goal.
        """
        q_goal = nb_path[-1]  # original end goal
        if goal_refinement:
            reached, q = self.ik_refinement(center.goal, nb_path[-1])
            if reached:
                q_goal = q

        # Try to restore using DMP;
        # if not feasible, refuse compression
        path = self.adapt(center, q_goal)

        # Check collision
        valid = self.path_validity_check(path)
        if not valid:
            return False, q_goal
        # Check if the path is not far from desired goal
        if np.linalg.norm(path[-1] - q_goal) > 1e-2 * self.robot.n_joints:
            return False, q_goal

        return True, q_goal

    def adapt(self, center: DMPCenter, q_goal: np.ndarray) -> np.ndarray:
        """
        Get a full joint-space trajectory from center.y0 to q_goal.
        Similar as in the build_center method, we fit a DMP on the residual
        """
        q_goal = np.asarray(q_goal)
        n_joints = len(q_goal)

        # Get DMP instance
        dmp = center.dmp
        steps = center.timesteps
        # Set start and goal
        dmp.y0 = center.start.copy()
        dmp.goal = q_goal.copy()

        # # Rollout the DMP
        # y_track, _, _ = dmp.rollout(timesteps=center.timesteps)
        # return y_track

        # rollout residual with fixed endpoints
        dmp.y0 = np.zeros(n_joints)
        dmp.goal = np.zeros(n_joints)
        res_track, _, _ = dmp.rollout(timesteps=steps, residual=True)

        # line to the new goal
        s = np.linspace(0.0, 1.0, steps)[:, None]
        line_new = center.start[None, :] + s * (q_goal - center.start)[None, :]

        # add the line to the residual to get the full path
        y_track = line_new + res_track
        return y_track


class TrajOptAdapter(Adapter):
    """Adapter using TrajOpt optimization"""

    def __init__(
        self,
        robot: MujocoRobot,
        ik_solver: IK,
        # TrajOpt parameters
        w_seed=1.0,
        w_vel=1e-2,
        w_acc=1.0,
    ):
        """Initialize the TrajOptAdapter"""
        super().__init__(robot, ik_solver)
        self.optimizer = TrajOpt(w_seed, w_vel, w_acc, robot.joint_limits)

    def build_center(self, center_path):
        """No need to optimize the center path"""
        center_path = np.asarray(center_path)
        return center_path, center_path[-1].copy()

    def compress(self, center, nb_path, goal_refinement=True):
        """Compress the nb_path into the center_path"""
        q_goal = nb_path[-1]  # original end goal
        if goal_refinement:
            reached, q = self.ik_refinement(center[-1], nb_path[-1])
            if reached:
                q_goal = q

        # Get the optimized path
        path = self.adapt(center, q_goal)

        # Check collision
        valid = self.path_validity_check(path)
        if not valid:
            return False, q_goal
        # Check if the path is not far from desired goal
        if np.linalg.norm(path[-1] - q_goal) > 5e-3 * self.robot.n_joints:
            return False, q_goal

        return True, q_goal

    def adapt(self, center, q_goal):
        """Get the optimized path"""
        ok, path = self.optimizer.solve(seed_traj=center, q_goal=q_goal)
        # return the original path if the optimization failed
        if not ok:
            return center
        return path
