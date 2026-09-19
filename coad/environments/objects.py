"""Object XML generation, scene YAML loading and object motion."""

import numpy as np
import yaml
import mujoco
from scipy.spatial.transform import Rotation


class SceneObjects:
    def build_xml(
        self, scene_yaml, parent_body_name="env_name", skip_ids=None, rgba=None
    ):
        """Return xml for environment"""

        if skip_ids is None:
            skip_ids = set()

        # Default to MuJoCo default gray if not provided
        if rgba is None:
            rgba = [0.5, 0.5, 0.5, 1]
            rgba = [0.15, 1, 0.15, 1]
            rgba = [0.133, 0.6, 0.329, 1]

        with open(scene_yaml, "r") as f:
            scene_yaml_data = yaml.safe_load(f)

        objs = scene_yaml_data["world"]["collision_objects"]
        lines = []
        lines.append(f'<body name="{parent_body_name}" pos="0 0 0">')

        for obj in objs:
            obj_id = obj.get("id", "")
            if obj_id in skip_ids:
                continue

            prim = obj["primitives"][0]
            pose = obj["primitive_poses"][0]

            pos = pose["position"]
            quat_xyzw = pose["orientation"]
            quat_wxyz = self.quat_xyzw_to_wxyz(quat_xyzw)

            prim_type = prim["type"].lower()
            dims = prim["dimensions"]
            self.env_details["collision_geoms"].append(obj_id)

            if prim_type == "box":
                size = [dims[0] / 2.0, dims[1] / 2.0, dims[2] / 2.0]
                mj_type = "box"
                mj_size = size

            elif prim_type == "cylinder":
                height, radius = dims[0], dims[1]
                mj_type = "cylinder"
                mj_size = [radius, height / 2.0]

            else:
                raise ValueError(
                    f"Unsupported primitive type: {prim_type} for id={obj_id}"
                )

            lines.append(
                f'  <geom name="{obj_id}" type="{mj_type}" '
                f'pos="{self.fmt(pos)}" quat="{self.fmt(quat_wxyz)}" '
                f'size="{self.fmt(mj_size)}" '
                f'contype="1" conaffinity="1" '
                f'rgba="{self.fmt(rgba)}"/>'
            )

        lines.append("</body>")
        return "\n".join(lines)

    def fmt(self, v):
        """Formatting for XML"""
        return " ".join(f"{x:.6g}" for x in v)

    def quat_xyzw_to_wxyz(self, quat_xyzw):
        """Convert to wxyz quats"""
        quat_wxyz = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
        return quat_wxyz

    def get_geom_pose(self, geom_name, model=None, data=None, as_matrix=False):
        """
        Return world pose of a MuJoCo geom.

        Does not call mj_forward; assumes data is already current.

        Returns:
            if as_matrix=True:
                4x4 homogeneous transform
            else:
                (pos, R)
        """
        if model is None:
            model = self.model
        if data is None:
            data = self.data

        gid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            geom_name,
        )

        if gid == -1:
            raise ValueError(f"Could not find geom: {geom_name}")

        pos = data.geom_xpos[gid].copy()
        R = data.geom_xmat[gid].reshape(3, 3).copy()

        rpy = Rotation.from_matrix(R).as_euler("xyz", degrees=False)

        if geom_name == "microwave_handle_sv":
            rpy = np.array([rpy[0], rpy[1], rpy[2] - np.pi / 2])
            rot_mat = Rotation.from_euler("xyz", rpy, degrees=False).as_matrix()
        else:
            rot_mat = R

        if not as_matrix:
            return pos, rpy

        T = np.eye(4)
        T[:3, :3] = rot_mat
        T[:3, 3] = pos
        return T

    def object_xml(
        self,
        object_dims,
        object_pose,
        fixed=False,
        temp=False,
        name=None,
    ):
        if self.object_details["type"] == "microwave":
            return self.microwave_object_xml(
                object_dims,
                object_pose,
                fixed,
                temp,
            )

        if self.object_details["type"] == "box":
            return self.cube_object_xml(
                object_dims,
                object_pose,
                fixed,
                temp,
                name=name,
            )

        if self.object_details["type"] == "cylinder":
            return self.cylinder_object_xml(
                object_dims,
                object_pose,
                fixed,
                temp,
                name=name,
            )

        raise ValueError(f"Unsupported object type: {self.object_details['type']}")

    def microwave_object_xml(self, object_dims, object_pose, fixed=False, temp=False):
        """Create articulated hollow microwave XML string.

        object_dims: full outer dimensions [lx, ly, lz]
        object_pose: [x, y, z, yaw]
        self.object_details["door_size"]: full door dimensions [lx, ly, lz]
        """

        rgba_shell = [0.7, 0.7, 0.7, 1.0]
        rgba_back = [0.55, 0.55, 0.55, 1.0]
        rgba_door = [0.2, 0.2, 0.25, 0.7]
        rgba_handle = [0.05, 0.05, 0.05, 1.0]

        name = "microwave_object"

        object_x, object_y, object_z, object_yaw = object_pose

        half = 0.5 * float(object_yaw)
        qw = np.cos(half)
        qx = 0.0
        qy = 0.0
        qz = np.sin(half)

        lx, ly, lz = map(float, object_dims)
        hx, hy, hz = lx / 2.0, ly / 2.0, lz / 2.0

        door_lx, door_ly, door_lz = map(float, self.object_details["door_size"])
        door_hx, door_hy, door_hz = door_lx / 2.0, door_ly / 2.0, door_lz / 2.0

        handle_lx, handle_ly, handle_lz = map(float, self.object_details["handle_size"])
        handle_hx, handle_hy, handle_hz = (
            handle_lx / 2.0,
            handle_ly / 2.0,
            handle_lz / 2.0,
        )

        wall_thickness = self.object_details.get("wall_thickness", 0.01)
        wt = wall_thickness / 2.0

        joint_xml = "" if fixed else f'<joint name="{name}_free" type="free"/>'

        # Door hinge at front-left edge.
        # Same convention as earlier:
        #   microwave x-axis: back/front
        #   microwave y-axis: left/right
        #   microwave z-axis: vertical
        front_x = -(hx + door_hx)
        hinge_y = -(-hy)
        door_center_y = -(door_ly / 2.0)

        handle_x = -door_lx
        handle_y = -0.35 * door_ly
        handle_z = 0.0

        self.object_details["hinge_pos_body"] = [front_x, hinge_y, 0]
        self.object_details["door_origin_closed_body"] = [0, door_center_y, 0]
        self.object_details["handle_pos_door"] = [handle_x, handle_y, handle_z]

        obj_xml = f"""
        <body name="{name}"
            pos="{object_x} {object_y} {object_z}"
            quat="{qw} {qx} {qy} {qz}">
            {joint_xml}

            <!-- Hollow microwave shell: open front -->

            <geom name="mw_bottom" type="box"
                pos="0 0 {-hz + wt}"
                size="{hx} {hy} {wt}"
                rgba="{rgba_shell[0]} {rgba_shell[1]} {rgba_shell[2]} {rgba_shell[3]}"/>

            <geom name="mw_top" type="box"
                pos="0 0 {hz - wt}"
                size="{hx} {hy} {wt}"
                rgba="{rgba_shell[0]} {rgba_shell[1]} {rgba_shell[2]} {rgba_shell[3]}"/>

            <geom name="mw_left" type="box"
                pos="0 {-hy + wt} 0"
                size="{hx} {wt} {hz}"
                rgba="{rgba_shell[0]} {rgba_shell[1]} {rgba_shell[2]} {rgba_shell[3]}"/>

            <geom name="mw_right" type="box"
                pos="0 {hy - wt} 0"
                size="{hx} {wt} {hz}"
                rgba="{rgba_shell[0]} {rgba_shell[1]} {rgba_shell[2]} {rgba_shell[3]}"/>

            <geom name="mw_back" type="box"
                pos="{-(-hx + wt)} 0 0"
                size="{wt} {hy} {hz}"
                rgba="{rgba_back[0]} {rgba_back[1]} {rgba_back[2]} {rgba_back[3]}"/>

            <!-- Door hinged at front-left edge -->
            <body name="microwave_door_hinge_frame" pos="{front_x} {hinge_y} 0">
                <joint name="microwave_door_hinge"
                    type="hinge"
                    axis="0 0 -1"
                    range="0 1.5708"
                    limited="true"/>

                <!-- Door center offset from hinge along +y -->
                <body name="microwave_door" pos="0 {door_center_y} 0">
                    <geom name="mw_door" type="box"
                        pos="0 0 0"
                        size="{door_hx} {door_hy} {door_hz}"
                        rgba="{rgba_door[0]} {rgba_door[1]} {rgba_door[2]} {rgba_door[3]}"/>

                    <geom name="mw_handle" type="box"
                        pos="{handle_x} {handle_y} {handle_z}"
                        size="{handle_hx} {handle_hy} {handle_hz}"
                        rgba="{rgba_handle[0]} {rgba_handle[1]} {rgba_handle[2]} {rgba_handle[3]}"/>
                </body>
            </body>
        </body>
        """

        if not temp:
            self.env_details["collision_geoms"].extend(
                [
                    "mw_bottom",
                    "mw_top",
                    "mw_left",
                    "mw_right",
                    "mw_back",
                    "mw_door",
                    "mw_handle",
                ]
            )

        return obj_xml

    def cube_object_xml(
        self, object_dims, object_pose, fixed=False, temp=False, name=None
    ):
        """Create cube object xml string"""
        rgba = [0.8, 0.2, 0.2, 1]

        if name is None:
            name = "cube_object"

        object_x, object_y, object_z, object_yaw = object_pose
        half = 0.5 * float(object_yaw)
        qw = np.cos(half)
        qx = 0.0
        qy = 0.0
        qz = np.sin(half)

        joint_xml = "" if fixed else f'<joint name="{name}_free" type="free"/>'

        obj_xml = f"""
        <body name="{name}"
            pos="{object_x} {object_y} {object_z}"
            quat="{qw} {qx} {qy} {qz}">
            {joint_xml}

            <!-- boxes centered at body origin -->
            <geom name="{name}_geom" type="box" pos="0 0 0"
                size="{object_dims[0]/2} {object_dims[1]/2} {object_dims[2]/2}"
                rgba="{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}"/>
        </body>
        """
        # self.collision_geoms.append(f"{name}_geom")
        if not temp:
            self.env_details["collision_geoms"].append(f"{name}_geom")
        return obj_xml

    def cylinder_object_xml(
        self,
        object_dims,
        object_pose,
        fixed=False,
        temp=False,
        name=None,
    ):
        """Create a cylinder-object XML fragment."""

        rgba = [0.8, 0.2, 0.2, 1.0]

        if name is None:
            name = "cube_object"

        radius, height = object_dims
        object_x, object_y, object_z, object_yaw = object_pose

        half_yaw = 0.5 * float(object_yaw)

        qw = np.cos(half_yaw)
        qx = 0.0
        qy = 0.0
        qz = np.sin(half_yaw)

        joint_xml = "" if fixed else f'<joint name="{name}_free" type="free"/>'

        object_xml = f"""
        <body name="{name}"
            pos="{object_x} {object_y} {object_z}"
            quat="{qw} {qx} {qy} {qz}">
            {joint_xml}

            <geom name="{name}_geom"
                type="cylinder"
                pos="0 0 0"
                size="{radius} {height / 2.0}"
                rgba="{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}"/>
        </body>
        """

        if not temp:
            self.env_details["collision_geoms"].append(f"{name}_geom")

        return object_xml

    def move_object(self, object_pose, model=None, data=None):

        if self.object_details["type"] == "box":
            self.move_cube_object(object_pose, model, data)

        elif self.object_details["type"] == "cylinder":
            self.move_xml_object(
                "cube_object",
                object_pose,
                model,
                data,
            )

        elif self.object_details["type"] == "microwave":
            self.move_xml_object(
                "microwave_object",
                object_pose,
                model,
                data,
            )

        else:
            raise ValueError(f"Unsupported object type: {self.object_details['type']}")

    def move_xml_object(
        self,
        object_name,
        object_pose,
        model=None,
        data=None,
    ):
        """
        Move a free-joint XML object to an x, y, z, yaw pose.

        The object's free joint must be named:
            <object_name>_free
        """

        if model is None:
            model = self.model

        if data is None:
            data = self.data

        x, y, z, yaw = object_pose

        joint_name = f"{object_name}_free"

        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )

        if joint_id == -1:
            raise ValueError(f"Could not find free joint: {joint_name}")

        qpos_address = model.jnt_qposadr[joint_id]
        qvel_address = model.jnt_dofadr[joint_id]

        half_yaw = 0.5 * float(yaw)

        quaternion_wxyz = [
            np.cos(half_yaw),
            0.0,
            0.0,
            np.sin(half_yaw),
        ]

        # Free-joint qpos:
        # [x, y, z, qw, qx, qy, qz]
        data.qpos[qpos_address : qpos_address + 7] = [
            x,
            y,
            z,
            *quaternion_wxyz,
        ]

        # Free-joint velocity:
        # [vx, vy, vz, wx, wy, wz]
        data.qvel[qvel_address : qvel_address + 6] = 0.0

        mujoco.mj_forward(model, data)

    def move_xml_joint(self, joint_name, joint_value, model=None, data=None):
        """
        Set a scalar joint value (hinge/slide).

        Example:
            move_xml_joint("microwave_door_hinge", np.pi/4)
        """

        if model is None:
            model = self.model
        if data is None:
            data = self.data

        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)

        if jid == -1:
            raise ValueError(f"Could not find joint: {joint_name}")

        qadr = model.jnt_qposadr[jid]
        vadr = model.jnt_dofadr[jid]

        data.qpos[qadr] = joint_value
        data.qvel[vadr] = 0.0

        mujoco.mj_forward(model, data)

    def move_cube_object(self, object_pose, model=None, data=None):
        """
        Move the active cube object to (x, y, z, yaw).

        Standard environments:
            object_pose = (x, y, z, yaw)

        AllStableEnv:
            object_pose = (face, x, y, z, yaw)

            The object corresponding to `face` is moved to the requested pose,
            while the other two face-specific objects are moved to a dummy pose.
        """
        if model is None:
            model = self.model

        if data is None:
            data = self.data

        # This is a pose, unlike move_swept_volume's interval-based dummy_config.
        dummy_pose = (1.0, 1.0, 0.0, 0.0)

        def move_one_object(joint_name, pose):
            if len(pose) != 4:
                raise ValueError(f"Expected pose (x, y, z, yaw), got {pose}")

            x, y, z, yaw = map(float, pose)

            jid = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_name,
            )

            if jid == -1:
                raise RuntimeError(f"Could not find free joint '{joint_name}'")

            qadr = model.jnt_qposadr[jid]
            vadr = model.jnt_dofadr[jid]

            half = 0.5 * yaw

            qw = np.cos(half)
            qx = 0.0
            qy = 0.0
            qz = np.sin(half)

            # MuJoCo free-joint qpos:
            # [x, y, z, qw, qx, qy, qz]
            data.qpos[qadr : qadr + 7] = [
                x,
                y,
                z,
                qw,
                qx,
                qy,
                qz,
            ]

            data.qvel[vadr : vadr + 6] = 0.0

        if self.environment_name == "allstable":
            if len(object_pose) != 5:
                raise ValueError(
                    "AllStableEnv expects object_pose in the form "
                    "(face, x, y, z, yaw), "
                    f"got {object_pose}"
                )

            face_in_contact = object_pose[0]
            real_object_pose = object_pose[1:]

            face_to_joint = {
                "xy": "cube_object_0_free",
                "yz": "cube_object_1_free",
                "zx": "cube_object_2_free",
            }

            if face_in_contact not in face_to_joint:
                raise ValueError(
                    f"Unknown face value: {face_in_contact}. "
                    f"Expected one of {tuple(face_to_joint)}"
                )

            for face, joint_name in face_to_joint.items():
                if face == face_in_contact:
                    move_one_object(joint_name, real_object_pose)
                else:
                    move_one_object(joint_name, dummy_pose)

        else:
            if len(object_pose) != 4:
                raise ValueError(
                    "Expected object_pose in the form "
                    "(x, y, z, yaw), "
                    f"got {object_pose}"
                )

            move_one_object(
                "cube_object_free",
                object_pose,
            )

        # Update all derived MuJoCo quantities once after moving every object.
        mujoco.mj_forward(model, data)
