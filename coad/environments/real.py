"""Real-lab scene and movable objects."""

import numpy as np
import mujoco
import re
from pathlib import Path
from .base import MujocoEnv


class RealEnv(MujocoEnv):
    """Real environment"""

    environment_name = "real"

    def __init__(self, robot, using_swept_volume=True, compute_tcr=True):
        """Initialize the real lab environment."""

        if robot != "ur10":
            raise NotImplementedError("RealEnv only supports UR10")

        super().__init__(robot, custom_base="lab_scene.xml")

        # --------------------------------------------------------------
        # Object details
        # --------------------------------------------------------------

        object_type = "cylinder"
        object_size = [0.045, 0.08]  # [radius, height]

        real_intervals = [
            [-0.35, 0.07],
            [-1.02, -0.70],
        ]

        object_variation = {
            "x": [real_intervals[0]],
            "y": [real_intervals[1]],
            "z": [
                [
                    object_size[1] / 2.0,
                    object_size[1] / 2.0,
                ]
            ],
            "yaw": [[0.0, 0.0]],
        }

        super().populate_object_details(
            object_type,
            object_size,
            object_variation,
        )

        # --------------------------------------------------------------
        # Environment details
        # --------------------------------------------------------------

        robot_pos = [0, 0, 0]
        robot_quat = [0, 0, 0, 1]  # WXYZ

        outer_rad = 1.0
        inner_rad = 0.3

        # RealEnv currently generates its scene programmatically, so it
        # does not need a scene YAML.
        super().populate_env_details(
            scene_yaml=None,
            robot_name=robot,
            env_name="real",
            robot_pos=robot_pos,
            robot_quat=robot_quat,
            outer_rad=outer_rad,
            inner_rad=inner_rad,
        )

        # RealEnv has manually specified placement intervals.
        self.env_details["intervals"] = real_intervals

        # --------------------------------------------------------------
        # Grasp details
        # --------------------------------------------------------------

        super().populate_grasp_details(
            alpha=0.5,
            grasp_type="top",
        )

        # --------------------------------------------------------------
        # Temporary compatibility with the old UR10 TSR implementation
        # --------------------------------------------------------------

        self.robot_pos = self.env_details["robot_pos"]
        self.robot_quat = self.env_details["robot_quat"]

        self.object_inner_rad = self.env_details["inner_rad"]
        self.object_outer_rad = self.env_details["outer_rad"]
        self.object_yaw = 0.0

        self.yaw_buffer = self.grasp_details["yaw_buffer"]
        self.alpha = self.grasp_details["alpha"]

        self.object_details["dist"] = [
            self.object_outer_rad,
            self.object_outer_rad,
            0.0,
            self.object_yaw,
        ]

        self.object_details["position"] = [
            self.robot_pos[0],
            self.robot_pos[1],
            object_size[1] / 2.0,
        ]

        # Keep this legacy dictionary until its consumers are migrated
        # to object_details and env_details.
        self.problem = {
            "name": "real",
            "intervals": real_intervals,
            "robot": robot,
        }

        if using_swept_volume:
            tcr_intervals = (
                super().construct_tcr() if (compute_tcr or using_swept_volume) else None
            )

            object_xml = super().create_swept_volume(
                tcr_intervals,
            )
        else:
            object_xml = super().cylinder_object_xml(
                self.object_details["size"],
                [0, 0, 0, 0],
            )

        xmls_to_add = [object_xml]

        wall1_dims = (0.24, 0.45, 0.29)
        wall_inflation = 0.07  # inflating by object's size (for return path)
        wall1_dims = np.array(wall1_dims) * 1.05
        wall1_dims = wall1_dims + wall_inflation / 2

        # build wall 1
        wall_1_xml = self.build_primitive_body_xml(
            body_name="wall1",
            prim_type="box",
            pos=(0.27, -0.82, 0.145),
            dims=wall1_dims.tolist(),
            quat_xyzw=(0, 0, 0, 1),
            make_free=False,
        )
        xmls_to_add.append(wall_1_xml)

        # build wall 2
        wall_2_xml = self.build_primitive_body_xml(
            body_name="wall2",
            prim_type="box",
            pos=(-0.07, -0.45, 0.135),
            dims=(0.48, 0.34, 0.27),
            quat_xyzw=(0, 0, 0, 1),
            make_free=False,
        )
        xmls_to_add.append(wall_2_xml)

        # build block 1
        block_1_xml = self.build_primitive_body_xml(
            body_name="block1",
            prim_type="box",
            pos=(0.205, -0.51, 0.1),
            dims=(0.09, 0.09, 0.2),
            quat_xyzw=(0, 0, 0, 1),
            make_free=False,
        )
        xmls_to_add.append(block_1_xml)

        # build packing
        packing_1_xml = self.build_primitive_body_xml(
            body_name="packing1",
            prim_type="box",
            pos=(0.54, -0.80, 0.01),
            dims=(0.21, 0.36, 0.02),
            quat_xyzw=(0, 0, 0, 1),
            make_free=False,
        )
        xmls_to_add.append(packing_1_xml)

        # --- parameters ---
        cx, cy, z_floor = 0.54, -0.80, 0.01
        Lx, Ly, t_floor = 0.21, 0.36, 0.02

        t_wall = 0.02
        h_wall = 0.13  # <-- change this to how tall you want the hollow box
        h_wall = 0.10 + wall_inflation

        z_top = z_floor + t_floor / 2.0
        z_wall = z_top + h_wall / 2.0

        x_off = (Lx / 2.0) - (t_wall / 2.0)
        y_off = (Ly / 2.0) - (t_wall / 2.0)

        # Left wall (thin in x, spans y, tall in z)
        packing_2_xml = self.build_primitive_body_xml(
            body_name="packing2_left",
            prim_type="box",
            pos=(cx - x_off, cy, z_wall),
            dims=(t_wall, Ly, h_wall),
            quat_xyzw=(0, 0, 0, 1),
            make_free=False,
        )

        # Right wall
        packing_3_xml = self.build_primitive_body_xml(
            body_name="packing3_right",
            prim_type="box",
            pos=(cx + x_off, cy, z_wall),
            dims=(t_wall, Ly, h_wall),
            quat_xyzw=(0, 0, 0, 1),
            make_free=False,
        )

        # Bottom wall (thin in y, spans x, tall in z)
        packing_4_xml = self.build_primitive_body_xml(
            body_name="packing4_bottom",
            prim_type="box",
            pos=(cx, cy - y_off, z_wall),
            dims=(Lx, t_wall, h_wall),
            quat_xyzw=(0, 0, 0, 1),
            make_free=False,
        )

        # Top wall
        packing_5_xml = self.build_primitive_body_xml(
            body_name="packing5_top",
            prim_type="box",
            pos=(cx, cy + y_off, z_wall),
            dims=(Lx, t_wall, h_wall),
            quat_xyzw=(0, 0, 0, 1),
            make_free=False,
        )

        xmls_to_add.append(packing_2_xml)
        xmls_to_add.append(packing_3_xml)
        xmls_to_add.append(packing_4_xml)
        xmls_to_add.append(packing_5_xml)

        # upper boundary
        upper_boundary_xml = self.build_primitive_body_xml(
            body_name="ub1",
            prim_type="box",
            pos=(0, -0.50, 1.02),
            dims=(1.5, 1.5, 0.02),
            quat_xyzw=(0, 0, 0, 1),
            make_free=False,
        )
        xmls_to_add.append(upper_boundary_xml)

        # upper boundary
        back_boundary_xml = self.build_primitive_body_xml(
            body_name="bb1",
            prim_type="box",
            pos=(0, 0.40, 0.8),
            dims=(2, 0.02, 1.60),
            quat_xyzw=(0, 0, 0, 1),
            make_free=False,
        )
        xmls_to_add.append(back_boundary_xml)

        left_boundary_xml = self.build_primitive_body_xml(
            body_name="lb1",
            prim_type="box",
            pos=(-0.82, -0.50, 0.8),
            dims=(0.02, 2, 1.60),
            quat_xyzw=(0, 0, 0, 1),
            make_free=False,
        )
        xmls_to_add.append(left_boundary_xml)

        right_boundary_xml = self.build_primitive_body_xml(
            body_name="rb1",
            prim_type="box",
            pos=(0.82, -0.50, 0.8),
            dims=(0.02, 2, 1.60),
            quat_xyzw=(0, 0, 0, 1),
            make_free=False,
        )
        xmls_to_add.append(right_boundary_xml)

        free_xml_path = f"{self.robot_dir}/real_scene.xml"
        self.xmls_to_add = xmls_to_add
        self.model, self.data = super().build_model(free_xml_path, xmls_to_add)

    def build_primitive_body_xml(
        self,
        body_name: str,
        geom_name: str | None = None,
        prim_type: str = "box",
        dims: list[float] | tuple[float, ...] = (1.0, 1.0, 1.0),
        pos: list[float] | tuple[float, float, float] = (0.0, 0.0, 0.0),
        quat_xyzw: list[float] | tuple[float, float, float, float] = (
            0.0,
            0.0,
            0.0,
            1.0,
        ),
        rgba: list[float] | tuple[float, float, float, float] | None = None,
        contype: int = 1,
        conaffinity: int = 1,
        make_free: bool = False,
    ) -> str:
        """
        Create XML for a single primitive inside its own <body>, similar to build_xml().

        - prim_type: "box" or "cylinder"
        * box dims = (lx, ly, lz)  -> mj_size = (lx/2, ly/2, lz/2)
        * cylinder dims = (height, radius) -> mj_size = (radius, height/2)
        - quat_xyzw is converted to MuJoCo quat order (w x y z)
        - If make_free=True, adds <joint type="free"> so you can move the body by setting qpos later.
        """

        if rgba is None:
            rgba = [0.133, 0.6, 0.329, 1.0]  # your last default
            rgba = [0.75, 0.75, 0.75, 1.0]

        if geom_name is None:
            geom_name = body_name

        prim_type = prim_type.lower()
        dims = list(dims)

        quat_wxyz = self.quat_xyzw_to_wxyz(quat_xyzw)

        if prim_type == "box":
            if len(dims) != 3:
                raise ValueError(f"box dims must be (lx, ly, lz), got {dims}")
            mj_type = "box"
            mj_size = [dims[0] / 2.0, dims[1] / 2.0, dims[2] / 2.0]

        elif prim_type == "cylinder":
            if len(dims) != 2:
                raise ValueError(f"cylinder dims must be (height, radius), got {dims}")
            height, radius = dims[0], dims[1]
            mj_type = "cylinder"
            mj_size = [radius, height / 2.0]

        else:
            raise ValueError(f"Unsupported primitive type: {prim_type}")

        lines = []
        lines.append(f'<body name="{body_name}" pos="0 0 0">')

        if make_free:
            # Free joint so the body's pose is controlled via qpos (7 values: x y z qw qx qy qz)
            lines.append(f'  <joint name="{body_name}_free" type="free"/>')

        # Match your pattern: pose on geom (not body)
        lines.append(
            f'  <geom name="{geom_name}" type="{mj_type}" '
            f'pos="{self.fmt(pos)}" quat="{self.fmt(quat_wxyz)}" '
            f'size="{self.fmt(mj_size)}" '
            f'contype="{contype}" conaffinity="{conaffinity}" '
            f'rgba="{self.fmt(rgba)}"/>'
        )

        lines.append("</body>")
        return "\n".join(lines)

    def randomize_object_positions(self, objects, object_heights):
        for i, object in enumerate(objects):
            joint_name = f"{object}_joint"

            x = np.random.uniform(-0.5, 0.5)
            y = np.random.uniform(-0.5, -0.1)
            z = object_heights[i] / 2
            yaw = np.random.uniform(-np.pi, np.pi)

            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            qadr = self.model.jnt_qposadr[jid]
            vadr = self.model.jnt_dofadr[jid]

            half = 0.5 * float(yaw)
            qw = np.cos(half)
            qx = 0.0
            qy = 0.0
            qz = np.sin(half)

            # free joint qpos layout: [x y z qw qx qy qz]
            self.data.qpos[qadr : qadr + 7] = [x, y, z, qw, qx, qy, qz]
            self.data.qvel[vadr : vadr + 6] = 0.0

            mujoco.mj_forward(self.model, self.data)

    def select_object(self, object_name):
        if object_name == "mug":
            object_path = "assets/ycb/mug.xml"
            self.object_details = {
                "size": [0.05, 0.052, 0.052],
                "type": "mug",
                "yaw": 0,
            }
            self.yaw_buffer = 6 * (np.pi / 180)
            return object_path
        elif object_name == "a_cups":
            object_path = "assets/ycb/a_cups.xml"
            self.object_details = {
                "size": [0.05, 0.052, 0.052],
                "type": "a_cups",
                "yaw": 0,
            }
            self.yaw_buffer = 6 * (np.pi / 180)
            return object_path
        elif object_name == "g_cups":
            object_path = "assets/ycb/g_cups.xml"
            self.object_details = {
                "size": [0.05, 0.052, 0.06],
                "type": "g_cups",
                "yaw": 0,
            }
            self.yaw_buffer = 6 * (np.pi / 180)
            return object_path

    def mjcf_file_to_fragment(self, path: str) -> str:
        """Read a standalone MJCF file and return an MJCF fragment:
        <asset>...</asset> + worldbody contents (bodies).
        """
        text = Path(path).read_text()

        # grab asset block (optional)
        m_asset = re.search(r"<asset\b[^>]*>.*?</asset>", text, flags=re.DOTALL)
        asset = m_asset.group(0) if m_asset else ""

        # grab contents inside <worldbody>...</worldbody>
        m_world = re.search(
            r"<worldbody\b[^>]*>(.*?)</worldbody>", text, flags=re.DOTALL
        )
        if not m_world:
            raise ValueError(f"No <worldbody> found in {path}")
        world_contents = m_world.group(1).strip()

        return (asset + "\n\n" + world_contents).strip()
