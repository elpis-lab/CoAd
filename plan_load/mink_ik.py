import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import numpy as np
import mujoco
import mink

from plan_load.mujoco_utils import sample_qpos
from plan_load.robot import MujocoRobot, Panda, UR10, FetchArm


class IK:
    """A meta IK package that uses mink"""

    def __init__(
        self,
        robot: MujocoRobot,
        solver: str = "daqp",
        collision_pairs: list[tuple[list[str], list[str]]] = (),
        max_velocities: dict[str, float] | None = None,
    ):
        """Initialize meta IK class"""
        # MuJoCo model
        self.robot = robot
        self.model = robot.model
        self.free_qpos_ids = robot.joint_qpos_ids
        self.free_dof_ids = robot.joint_dof_ids

        # Mink
        self.configuration = mink.Configuration(self.model, robot.data.qpos[:])
        self.default_q = self.configuration.q.copy()
        # solver
        self.solver = solver

        # Tasks
        # reach target
        self.ee_task = mink.FrameTask(
            frame_name=robot.ee_name,
            frame_type="site",
            position_cost=1.0,
            orientation_cost=0.5,
            lm_damping=1e-6,
        )
        # keep posture
        self.posture_task = mink.PostureTask(self.model, cost=1e-3)
        self.tasks = [self.ee_task, self.posture_task]

        # Constraints
        # freeze dof
        freeze_dof = list(
            np.setdiff1d(range(self.model.nv), self.free_dof_ids)
        )
        if len(freeze_dof) > 0:
            self.freeze_task = mink.DofFreezingTask(self.model, freeze_dof)
            self.constraints = [self.freeze_task]

        # Limits
        # joint limits
        self.joint_limit = mink.ConfigurationLimit(
            self.model, min_distance_from_limits=1e-6
        )
        self.limits = [self.joint_limit]
        # velocity limit
        if max_velocities is not None:
            self.velocity_limit = mink.VelocityLimit(
                self.model, max_velocities
            )
            self.limits.append(self.velocity_limit)
        # collision avoidance
        self.limits_col = list(self.limits)
        if collision_pairs:
            collision = mink.CollisionAvoidanceLimit(
                model=self.model, geom_pairs=collision_pairs
            )
            self.limits_col.append(collision)

    def solve(
        self,
        target: np.ndarray,
        current: np.ndarray | None = None,
        random_current: bool = False,
        random_sigma: float = 0.4,
        use_col: bool = False,
        iter_dt: float = 0.01,
        max_iters: int = 200,
        pos_tol: float = 1e-3,
        rot_tol: float = 1e-2,
    ):
        """Solve IK for target point [x y z qw qx qy qz]"""
        # Get a full size configuration to start with
        # Check reference configuration size
        config = self.default_q.copy()
        if current is not None:
            if len(current) == self.model.nq:
                config[:] = current
            elif len(current) == self.robot.n_joints:
                config[self.free_qpos_ids] = current
            else:
                raise ValueError(
                    f"Invalid current q length: {len(current)}. "
                    + f"Expected {self.model.nq} or {self.robot.n_joints}"
                )

        # Get random joint values for free joints
        if random_current:
            if current is not None:
                if len(current) == self.model.nq:
                    current = current[self.free_qpos_ids]
            # if current is None, sample randomly
            # if current is provided, sample around it
            config[self.free_qpos_ids] = sample_qpos(
                self.model, self.free_qpos_ids, None, current, random_sigma
            )

        # Set configuration
        self.configuration.update(q=config)
        self.posture_task.set_target(config)

        # Set target
        rot = target[3:]
        rot = rot / np.linalg.norm(rot)
        target_rot = mink.SO3(rot)
        self.ee_task.set_target(
            mink.SE3.from_rotation_and_translation(target_rot, target[:3])
        )

        # Run IK
        reached = False
        best_q = None
        best_err = np.inf
        for i in range(max_iters):
            # Run one IK step (joint velocities)
            try:
                qdot = mink.solve_ik(
                    configuration=self.configuration,
                    tasks=self.tasks,
                    constraints=self.constraints,
                    limits=self.limits_col if use_col else self.limits,
                    dt=iter_dt,
                    solver=self.solver,
                )
            except Exception as e:
                return False, best_q

            # Integrate to new configuration
            self.configuration.integrate_inplace(qdot, iter_dt)
            # clip to joint limits
            q = self.configuration.q.copy()
            q[self.free_qpos_ids] = np.clip(
                q[self.free_qpos_ids],
                self.joint_limit.lower[self.free_qpos_ids],
                self.joint_limit.upper[self.free_qpos_ids],
            )
            self.configuration.update(q=q)

            # Check convergence
            err = self.ee_task.compute_error(self.configuration)
            pos_err = np.linalg.norm(err[:3])
            rot_err = np.linalg.norm(err[3:])
            if pos_err < pos_tol and rot_err < rot_tol:
                reached = True
                best_q = self.configuration.q.copy()
                break

            if pos_err + rot_err < best_err:
                best_err = pos_err + rot_err
                best_q = self.configuration.q.copy()

        return reached, best_q[self.free_qpos_ids]


class PandaIK(IK):
    """Panda IK class"""

    # Collision geoms for IK
    GRIPPER = [
        "hand_c",
        "left_finger_c",
        "right_finger_c",
        "left_pad_c1",
        "left_pad_c2",
        "left_pad_c3",
        "left_pad_c4",
        "left_pad_c5",
        "right_pad_c1",
        "right_pad_c2",
        "right_pad_c3",
        "right_pad_c4",
        "right_pad_c5",
    ]
    ARM = [
        "link0_c",
        "link1_c",
        "link2_c",
        "link3_c",
        "link4_c",
        "link5_c0",
        "link5_c1",
        "link5_c2",
        "link6_c",
        "link7_c",
    ]

    def __init__(
        self,
        robot: Panda,
        solver: str = "daqp",
        env_collision_geoms: list[str] = None,
    ):
        """Initialize Panda IK class"""
        # Collision constraints
        # self collision
        collision_pairs = [(self.GRIPPER, self.ARM)]
        # environment collision
        if env_collision_geoms is not None:
            collision_pairs.append((self.GRIPPER, env_collision_geoms))
        # Velocity limits
        max_velocities = {name: np.pi for name in robot.joint_names}
        super().__init__(robot, solver, collision_pairs, max_velocities)


class UR10IK(IK):
    """UR10 IK class"""

    GRIPPER = [
        "gripper_base_collision",
        "r1_collision",
        "r2_collision",
        "l1_collision",
        "l2_collision",
    ]
    ARM = [
        "shoulder_collision",
        "upperarm_collision_0",
        "upperarm_collision_1",
        "forearm_collision_0",
        "forearm_collision_1",
        "wrist_1_collision",
        "wrist_2_collision_0",
        "wrist_2_collision_1",
    ]

    def __init__(
        self,
        robot: UR10,
        solver: str = "daqp",
        env_collision_geoms: list[str] = None,
    ):
        """Initialize UR10 IK class"""
        # Collision constraints
        # self collision
        collision_pairs = [(self.GRIPPER, self.ARM)]
        # environment collision
        if env_collision_geoms is not None:
            collision_pairs.append((self.GRIPPER, env_collision_geoms))
        # Velocity limits
        max_velocities = {name: np.pi for name in robot.joint_names}
        super().__init__(robot, solver, collision_pairs, max_velocities)


# TODO
class FetchArmIK(IK):
    """Fetch (arm only) IK class"""

    pass


# TODO
# Write some test cases for the IK classes
if __name__ == "__main__":
    pass
