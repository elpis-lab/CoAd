import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import numpy as np
import mujoco
import mujoco.viewer

from plan_load.mujoco_utils import joint_names_to_joint_ids
from plan_load.mujoco_utils import joints_to_qpos_dof_ids
from plan_load.mujoco_utils import joints_to_limits


class MujocoRobot:
    """Generic robot wrapper for MuJoCo model/data access."""

    def __init__(
        self,
        model,
        joint_names,
        root_root,
        collision_geom_group=3,
        ee_names=None,
        visualize=False,
    ):
        """Initialize MujocoRobot"""
        self.model = model
        self.data = mujoco.MjData(model)
        self.joint_names = joint_names
        self.root_root = root_root
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

        self.ee_body_names = []
        if ee_names is not None:
            self.ee_names = ee_names

        # get joint limits
        self.joint_limits = joints_to_limits(model, self.joint_ids)
        # get robot geoms from subtree
        self.robot_geoms = self.get_robot_geoms(collision_geom_group)

    def get_joint_qpos(self, data=None):
        """Get joint angles"""
        # allow passing in a different data object
        if data is None:
            data = self.data
        return data.qpos[self.joint_qpos_ids].copy()

    def set_joint_qpos(self, q, data=None):
        """Set joint angles"""
        # allow passing in a different data object
        if data is None:
            data = self.data
        # set joint values
        q = np.asarray(q)
        if len(q) != self.n_joints:
            raise ValueError("Expected q length %d" % self.n_joints)
        data.qpos[self.joint_qpos_ids] = q

        mujoco.mj_forward(self.model, data)

    def get_robot_geoms(self, geom_group=None):
        """Get robot geoms"""
        root_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, self.root_root
        )
        if root_body_id < 0:
            raise ValueError("Root body '%s' not found" % self.root_root)

        # Mark bodies in robot root subtree
        in_robot = [False] * self.model.nbody
        in_robot[root_body_id] = True
        for b in range(self.model.nbody):
            p = self.model.body_parentid[b]
            if p >= 0 and in_robot[p]:
                in_robot[b] = True

        # Get robot geoms that are in the subtree and are in given group
        robot_geoms = set()
        for g in range(self.model.ngeom):
            if in_robot[self.model.geom_bodyid[g]]:
                if (
                    geom_group is None
                    or self.model.geom_group[g] == geom_group
                ):
                    robot_geoms.add(g)

        return robot_geoms


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
    FINGERS = ["finger_joint1", "finger_joint2"]
    FINGERS_CLOSED = [0.0, 0.0]
    FINGERS_OPEN = [0.04, 0.04]

    def __init__(self, model, visualize=False):
        """Initialize PandaRobot"""
        MujocoRobot.__init__(
            self,
            model,
            joint_names=self.ARM,
            root_root="link0",
            collision_geom_group=3,
            ee_names=["attachment_site"],
            visualize=visualize,
        )

        # open the gripper
        for i, finger in enumerate(self.FINGERS):
            j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, finger)
            self.data.qpos[model.jnt_qposadr[j_id]] = self.FINGERS_OPEN[i]
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

    def __init__(self, model, visualize=False):
        """Initialize UR10Robot"""
        MujocoRobot.__init__(
            self,
            model,
            joint_names=self.ARM,
            root_root="base",
            collision_geom_group=3,
            ee_names=["attachment_site"],
            visualize=visualize,
        )


if __name__ == "__main__":
    # Test Robot
    model = mujoco.MjModel.from_xml_path("assets/franka_emika_panda/scene.xml")
    robot = Panda(model, visualize=True)
    robot.set_joint_qpos(
        np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
    )

    from plan_load.planning import geoms_in_contact

    in_contact = geoms_in_contact(model, robot.data, robot.robot_geoms, True)
    print(in_contact)
    robot.viewer.sync()
    input()

    # # Test UR10
    # model = mujoco.MjModel.from_xml_path("assets/ur10/ur10_robotis.xml")
    # robot = UR10(model, visualize=True)
    # robot.set_joint_qpos(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

    while True:
        time.sleep(0.01)

    # count = 0
    # value = 1.0
    # while True:
    #     robot.viewer.sync()
    #     if count % 300 == 0:
    #         value = 0.0 if value == 1.0 else 1.0
    #         count = 0
    #     robot.data.ctrl[6] = value
    #     mujoco.mj_step(robot.model, robot.data)
    #     count += 1
    #     time.sleep(0.01)
