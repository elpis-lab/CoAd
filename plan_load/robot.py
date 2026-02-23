import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import numpy as np
import mujoco
import mujoco.viewer

from plan_load.mujoco_utils import joint_names_to_joint_ids
from plan_load.mujoco_utils import joints_to_limits, joints_to_qpos_dof_ids
from plan_load.mujoco_utils import get_geoms_from_group, geoms_in_contact


class MujocoRobot:
    """Generic robot wrapper for MuJoCo model/data access."""

    def __init__(
        self,
        model,
        joint_names,
        root_link,
        data=None,
        collision_geom_group=3,
        ee_name=None,
        visualize=False,
    ):
        """Initialize MujocoRobot"""
        self.model = model
        if data is None:
            self.data = mujoco.MjData(model)
        else:
            self.data = data
        self.joint_names = joint_names
        self.root_link = root_link
        self.viewer = None
        if visualize:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()

        # Resolve joint ids and corresponding qpos ids
        self.joint_ids = joint_names_to_joint_ids(model, self.joint_names)
        self.joint_qpos_ids, self.joint_dof_ids = joints_to_qpos_dof_ids(
            model, joint_names=self.joint_names
        )
        self.n_joints = len(self.joint_ids)
        # get joint limits
        self.joint_limits = joints_to_limits(model, self.joint_ids)

        # get robot geoms from subtree
        self.robot_geoms = self.get_robot_geoms(collision_geom_group)

        self.ee_name = ee_name

    def get_joint_qpos(self):
        """Get joint angles"""
        # allow passing in a different data object
        return self.data.qpos[self.joint_qpos_ids].copy()

    def set_joint_qpos(self, q):
        """Set joint angles"""
        # set joint values
        q = np.asarray(q)
        if len(q) != self.n_joints:
            raise ValueError("Expected q length %d" % self.n_joints)
        self.data.qpos[self.joint_qpos_ids] = q

        mujoco.mj_forward(self.model, self.data)

    def get_ee_pose(self):
        """Get end-effector pose"""
        ee_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, self.ee_name
        )
        pos = self.data.site_xpos[ee_id]
        mat = self.data.site_xmat[ee_id]
        quat = np.zeros(4, dtype=float)
        mujoco.mju_mat2Quat(quat, mat)
        return np.concatenate([pos.copy(), quat])

    def in_contact(self, verbose=False):
        """Check if the robot is in contact with the environment"""
        return geoms_in_contact(
            self.model, self.data, self.robot_geoms, verbose
        )

    def get_robot_geoms(self, geom_group):
        """Get robot geoms"""
        return get_geoms_from_group(self.model, geom_group, self.root_link)

    def teleport_base(self, pos=[0.0, 0.0, 0.0], quat=[1.0, 0.0, 0.0, 0.0]):
        """Teleport the robot base to the given position and orientation"""
        bid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, self.root_link
        )
        if bid < 0:
            raise ValueError(f"unknown body root: '{self.root_link}'")

        self.model.body_pos[bid] = pos
        self.model.body_quat[bid] = quat
        mujoco.mj_forward(self.model, self.data)

    def in_limits(self, q):
        """Check if a configuration is within joint limits"""
        lo, hi = self.joint_limits
        return np.all(q >= lo) and np.all(q <= hi)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()


class Panda(MujocoRobot):
    """Franka Panda specialization."""

    ARM = [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
    ]
    HOME_POS = [0, 0, 0, -np.pi / 2, 0, np.pi / 2, -np.pi / 4]
    FINGER = ["finger_joint1", "finger_joint2"]
    FINGER_OPEN = [0.04, 0.04]
    FINGER_CLOSED = [0.0, 0.0]

    def __init__(self, model, data=None, visualize=False):
        """Initialize PandaRobot"""
        MujocoRobot.__init__(
            self,
            model,
            joint_names=self.ARM,
            root_link="link0",
            data=data,
            collision_geom_group=3,
            ee_name="attachment_site",
            visualize=visualize,
        )
        # Send to home
        self.set_joint_qpos(self.HOME_POS)

        # Open the gripper
        for i, finger in enumerate(self.FINGER):
            j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, finger)
            self.data.qpos[model.jnt_qposadr[j_id]] = self.FINGER_OPEN[i]
        mujoco.mj_forward(model, self.data)


class UR10(MujocoRobot):
    """UR10 specialization."""

    ARM = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]
    HOME_POS = [0, -1.7, 2, -1.87, -np.pi / 2, 0]
    FINGER = ["rh_r1", "rh_l1", "rh_r2", "rh_l2"]
    FINGER_OPEN = [0, 0, 0, 0]
    FINGER_CLOSED = [1.12, 1.12, 1.12, 1.12]

    def __init__(self, model, data=None, visualize=False):
        """Initialize UR10Robot"""
        MujocoRobot.__init__(
            self,
            model,
            joint_names=self.ARM,
            root_link="base",
            data=data,
            collision_geom_group=3,
            ee_name="attachment_site",
            visualize=visualize,
        )
        # Send to home
        self.set_joint_qpos(self.HOME_POS)


class FetchArm(MujocoRobot):
    """Fetch specialization."""

    ARM = [
        "torso_lift_joint",
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "upperarm_roll_joint",
        "elbow_flex_joint",
        "forearm_roll_joint",
        "wrist_flex_joint",
        "wrist_roll_joint"
    ]
    FINGER = ["r_gripper_finger_joint", "l_gripper_finger_joint"]
    FINGER_CLOSED = [0, 0]
    FINGER_OPEN = [0.05, 0.05]
    HOME_POS = [0, -1.5, 0, -np.pi, -np.pi / 2, 0, 0, 0]

    def __init__(self, model, data=None, visualize=False):
        """Initialize FetchRobot"""
        MujocoRobot.__init__(
            self,
            model,
            joint_names=self.ARM,
            root_link="base_link",
            data=data,
            collision_geom_group=3,
            ee_name="attachment_site",
            visualize=visualize,
        )
        # Send to home
        self.set_joint_qpos(self.HOME_POS)

        # Open the gripper
        for i, finger in enumerate(self.FINGER):
            j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, finger)
            self.data.qpos[model.jnt_qposadr[j_id]] = self.FINGER_OPEN[i]
        mujoco.mj_forward(model, self.data)


if __name__ == "__main__":
    # Test Panda
    # model = mujoco.MjModel.from_xml_path("assets/franka_emika_panda/scene.xml")
    # robot = Panda(model, visualize=True)
    # robot.teleport_base(np.array([0.2, 0.0, 0.2]))

    # Test UR10
    # model = mujoco.MjModel.from_xml_path("assets/ur10/scene.xml")
    # robot = UR10(model, visualize=True)

    # TODO
    # Test Fetch
    model = mujoco.MjModel.from_xml_path("assets/fetch/scene.xml")
    robot = FetchArm(model, visualize=True)
    #robot.set_joint_qpos(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    robot.teleport_base(np.array([0.0, 0.0, 0.005]))

    # test collision checking
    in_contact = geoms_in_contact(model, robot.data, robot.robot_geoms, True)
    print(robot.robot_geoms)
    print("in_contact:", in_contact)
    robot.viewer.sync()

    # Keep the viewer
    try:
        while True:
            time.sleep(0.01)
    except KeyboardInterrupt:
        robot.close()
