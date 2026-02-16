import time
import numpy as np
import mujoco
import mink

from mujoco_utils import joint_names_to_qpos_dof_ids


class IK:
    """A meta IK package that uses mink"""

    def __init__(
        self,
        model: mujoco.MjModel,
        ee_site: str = "attachment_site",
        default_q: np.ndarray | None = None,
        free_joint_names: list[str] | None = None,
        solver: str = "daqp",
        collision_pairs: list[tuple[list[str], list[str]]] = (),
        max_velocities: dict[str, float] | None = None,
    ):
        """Initialize meta IK class"""
        # MuJoCo model
        self.model = model
        if free_joint_names is not None:
            self.free_qpos_ids, self.free_dof_ids = (
                joint_names_to_qpos_dof_ids(model, free_joint_names)
            )
        else:
            self.free_qpos_ids = np.arange(model.nq)
            self.free_dof_ids = np.arange(model.nv)

        # Mink
        self.configuration = mink.Configuration(self.model)
        if default_q is not None:
            self.configuration.update(q=default_q)
        self.default_q = self.configuration.q.copy()
        self.data = self.configuration.data
        # solver
        self.solver = solver

        # Tasks
        # reach target
        self.ee_site = ee_site
        self.ee_task = mink.FrameTask(
            frame_name=self.ee_site,
            frame_type="site",
            position_cost=1.0,
            orientation_cost=0.5,
            lm_damping=1e-6,
        )
        # keep posture
        self.posture_task = mink.PostureTask(self.model, cost=1e-3)
        self.tasks = [self.ee_task, self.posture_task]
        # freeze dof
        freeze_dof = np.setdiff1d(np.arange(model.nv), self.free_dof_ids)
        if len(freeze_dof) > 0:
            self.freeze_task = mink.DofFreezingTask(model, freeze_dof)
            self.tasks.append(self.freeze_task)

        # Joint limits
        joint_limit = mink.ConfigurationLimit(model=self.model)
        self.limits = [joint_limit]

        # Velocity limit
        if max_velocities is not None:
            velocity_limit = mink.VelocityLimit(self.model, max_velocities)
            self.limits.append(velocity_limit)

        # Collision avoidance
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
        use_col: bool = True,
        iter_dt: float = 0.01,
        max_iters: int = 200,
        pos_tol: float = 1e-3,
        rot_tol: float = 1e-2,
    ):
        """Solve IK for target point [x y z qw qx qy qz]"""
        # Set current configuration
        if current is None:
            if random_current:
                # TODO
                current
            else:
                current = self.default_q.copy()
        self.configuration.update(q=current)
        self.posture_task.set_target(current)

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
            qdot = mink.solve_ik(
                configuration=self.configuration,
                tasks=self.tasks,
                dt=iter_dt,
                solver=self.solver,
                limits=self.limits_col if use_col else self.limits,
            )
            # Integrate to new configuration
            self.configuration.integrate_inplace(qdot, iter_dt)

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

        return reached, best_q


class PandaIK(IK):
    """Panda IK class"""

    # Collision geoms
    HAND = ["hand_c", "left_finger_c", "right_finger_c"]
    PADS = [
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
    FREE_JOINTS = [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
    ]

    def __init__(
        self,
        model: mujoco.MjModel,
        ee_site: str = "attachment_site",
        default_q: np.ndarray | None = None,
        free_joint_names: list[str] | None = None,
        solver: str = "daqp",
        collision_pairs: list[tuple[list[str], list[str]]] = None,
        max_velocities: dict[str, float] | None = None,
        env_collision_geoms: list[str] = None,
    ):
        """Initialize Panda IK class"""
        # Make sure the finger joints are open
        if default_q is not None:
            if len(default_q) == 7:
                default_q = np.concatenate([default_q, np.array([0.04, 0.04])])
            elif len(default_q) == 9:
                default_q = default_q.copy()
                default_q[7:9] = [0.04, 0.04]

        if free_joint_names is None:
            free_joint_names = self.FREE_JOINTS

        # Self collision
        if collision_pairs is None:
            collision_pairs = [(self.HAND + self.PADS, self.ARM)]
            if env_collision_geoms is not None:
                collision_pairs.append(
                    (self.HAND + self.PADS, env_collision_geoms)
                )

        if max_velocities is None:
            max_velocities = {name: np.pi for name in self.FREE_JOINTS}

        super().__init__(
            model,
            ee_site,
            default_q,
            free_joint_names,
            solver,
            collision_pairs,
            max_velocities,
        )

    def solve(
        self,
        target: np.ndarray,
        current: np.ndarray | None = None,
        random_current: bool = False,
        use_col: bool = True,
        iter_dt: float = 0.01,
        max_iters: int = 200,
        pos_tol: float = 1e-3,
        rot_tol: float = 1e-2,
    ):
        """Override solve method to use 7-DOF input and output"""
        # Expand to full DOF
        ref = current
        if current is not None:
            ref = self.default_q.copy()
            ref[self.free_qpos_ids] = current

        reached, solution = super().solve(
            target,
            ref,
            random_current,
            use_col,
            iter_dt,
            max_iters,
            pos_tol,
            rot_tol,
        )
        # Crop to free joints' qpos
        return reached, solution[self.free_qpos_ids]


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
    FREE_JOINTS = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]

    def __init__(
        self,
        model: mujoco.MjModel,
        ee_site: str = "attachment_site",
        default_q: np.ndarray | None = None,
        free_joint_names: list[str] | None = None,
        solver: str = "daqp",
        collision_pairs: list[tuple[list[str], list[str]]] = None,
        max_velocities: dict[str, float] | None = None,
        env_collision_geoms: list[str] = None,
    ):
        """Initialize Panda IK class"""
        # Make sure the finger joints are open
        if default_q is not None:
            if len(default_q) == 6:
                default_q = np.concatenate([default_q, 1.12 * np.ones(4)])
            elif len(default_q) == 10:
                default_q = default_q.copy()
                default_q[6:10] = 1.12 * np.ones(4)

        if free_joint_names is None:
            free_joint_names = self.FREE_JOINTS

        # Self collision
        if collision_pairs is None:
            collision_pairs = [(self.GRIPPER, self.ARM)]
            if env_collision_geoms is not None:
                collision_pairs.append((self.GRIPPER, env_collision_geoms))

        if max_velocities is None:
            max_velocities = {
                "shoulder_pan_joint": np.pi,
                "shoulder_lift_joint": np.pi,
                "elbow_joint": np.pi,
                "wrist_1_joint": np.pi,
                "wrist_2_joint": np.pi,
                "wrist_3_joint": np.pi,
            }

        super().__init__(
            model,
            ee_site,
            default_q,
            free_joint_names,
            solver,
            collision_pairs,
            max_velocities,
        )

    def solve(
        self,
        target: np.ndarray,
        current: np.ndarray | None = None,
        random_current: bool = False,
        use_col: bool = True,
        iter_dt: float = 0.01,
        max_iters: int = 200,
        pos_tol: float = 1e-3,
        rot_tol: float = 1e-2,
    ):
        """Override solve method to use 6-DOF input and output"""
        # Expand to full DOF
        ref = current
        if current is not None:
            ref = self.default_q.copy()
            ref[self.free_qpos_ids] = current

        reached, solution = super().solve(
            target,
            ref,
            random_current,
            use_col,
            iter_dt,
            max_iters,
            pos_tol,
            rot_tol,
        )
        # Crop to free joints' qpos
        return reached, solution[self.free_qpos_ids]
