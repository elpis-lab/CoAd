"""Swept-volume meshes, primitive metadata and placement."""

import os
import numpy as np
import mujoco
import itertools
import trimesh


class SweptVolumes:
    def create_swept_volume(
        self,
        tcr_intervals,
        obj_size=None,
        sv_count=0,
        fixed=False,
    ):
        """
        Create a swept-volume mesh and save the cuboids used to construct it.

        The dictionary has the form:

            self.swept_volume_primitives[geom_name] = [
                {
                    "type": "cuboid",
                    "position": np.ndarray(shape=(3,)),
                    "orientation": np.ndarray(shape=(3, 3)),
                    "half_extents": np.ndarray(shape=(3,)),
                },
                ...
            ]

        `position` and `orientation` describe the cuboid pose relative to the
        mesh geom's local coordinate frame.

        During VAMP environment construction, the current MuJoCo geom pose
        should be composed with this local primitive pose.
        """

        if obj_size is None:
            obj_size = self.object_details["size"]

        def yaw_rot(theta):
            c = np.cos(theta)
            s = np.sin(theta)

            return np.array(
                [
                    [c, -s, 0.0],
                    [s, c, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=float,
            )

        def make_T(rotation, position):
            T = np.eye(4, dtype=float)
            T[:3, :3] = rotation
            T[:3, 3] = position
            return T

        def transform_points(points, T):
            points_h = np.c_[
                points,
                np.ones(len(points)),
            ]

            return (T @ points_h.T).T[:, :3]

        def box_corners(lx, ly, lz):
            hx = lx / 2.0
            hy = ly / 2.0
            hz = lz / 2.0

            return np.array(
                [
                    [-hx, -hy, -hz],
                    [-hx, -hy, hz],
                    [-hx, hy, -hz],
                    [-hx, hy, hz],
                    [hx, -hy, -hz],
                    [hx, -hy, hz],
                    [hx, hy, -hz],
                    [hx, hy, hz],
                ],
                dtype=float,
            )

        def make_cuboid_primitive(
            position,
            orientation,
            half_extents,
        ):
            """
            Save a cuboid pose in the mesh geom's local frame.
            """

            return {
                "type": "cuboid",
                "position": np.asarray(
                    position,
                    dtype=float,
                ).copy(),
                "orientation": np.asarray(
                    orientation,
                    dtype=float,
                )
                .reshape(3, 3)
                .copy(),
                "half_extents": np.asarray(
                    half_extents,
                    dtype=float,
                ).copy(),
            }

        def make_cylinder_primitive(
            position,
            orientation,
            radius,
            length,
        ):
            return {
                "type": "cylinder",
                "position": np.asarray(
                    position,
                    dtype=float,
                ).copy(),
                "orientation": np.asarray(
                    orientation,
                    dtype=float,
                )
                .reshape(3, 3)
                .copy(),
                "radius": float(radius),
                "length": float(length),
            }

        # ================================================================
        # Box swept volume
        # ================================================================
        if self.object_details["type"] == "box":

            if self.env_details["robot"] == "fetch":
                mesh_prefix = "assets/temp"
            else:
                mesh_prefix = "temp"

            lx, ly, lz = map(float, obj_size)

            half_extents = np.array(
                [
                    lx / 2.0,
                    ly / 2.0,
                    lz / 2.0,
                ],
                dtype=float,
            )

            base_box_mesh = trimesh.creation.box(
                extents=[lx, ly, lz],
            )

            box_vertices = []
            box_primitives = []

            yaw_values = np.asarray(
                tcr_intervals["yaw"],
                dtype=float,
            )

            if self.environment_name == "largeobj":
                num_yaw_samples = 100
            else:
                num_yaw_samples = 4

            yaw_samples = np.linspace(
                yaw_values.min(),
                yaw_values.max(),
                num_yaw_samples,
            )

            for x, y, yaw in itertools.product(
                tcr_intervals["x"],
                tcr_intervals["y"],
                yaw_samples,
            ):
                local_position = np.array(
                    [
                        float(x),
                        float(y),
                        0.0,
                    ],
                    dtype=float,
                )

                local_orientation = yaw_rot(float(yaw))

                # Save the cuboid pose relative to the mesh geom.
                box_primitives.append(
                    make_cuboid_primitive(
                        position=local_position,
                        orientation=local_orientation,
                        half_extents=half_extents,
                    )
                )

                # Construct the same transformed cuboid for the STL.
                T_geom_primitive = make_T(
                    local_orientation,
                    local_position,
                )

                box_mesh = base_box_mesh.copy()
                box_mesh.apply_transform(T_geom_primitive)

                box_vertices.append(box_mesh.vertices.copy())

            if not box_vertices:
                raise ValueError("No cuboids were generated for the box swept volume.")

            box_vertices = np.vstack(box_vertices)

            hull = trimesh.points.PointCloud(box_vertices).convex_hull

            mesh_file_name = f"sv_mesh_{sv_count}_{os.getpid()}.stl"

            self._generated_assets[mesh_file_name] = hull.export(file_type="stl")

            mesh_path_for_xml = f"{mesh_prefix}/{mesh_file_name}"

            mesh_name = f"swept_volume_mesh_{sv_count}"

            body_name = f"swept_volume_{sv_count}"

            geom_name = f"sv_mesh_{sv_count}"

            rgba = [0.8, 0.8, 0.8, 1.0]

            joint_xml = (
                "" if fixed else (f'<joint name="{body_name}_free" ' f'type="free"/>')
            )

            # geom name -> primitive list
            self.swept_volume_primitives[geom_name] = box_primitives

            return f"""
            <asset>
                <mesh
                    name="{mesh_name}"
                    file="{mesh_path_for_xml}"/>
            </asset>

            <body name="{body_name}" pos="0 0 0">
                {joint_xml}

                <geom
                    name="{geom_name}"
                    type="mesh"
                    mesh="{mesh_name}"
                    rgba="{' '.join(map(str, rgba))}"/>
            </body>
            """

        # ================================================================
        # Microwave swept volume
        # ================================================================
        elif self.object_details["type"] == "microwave":

            if self.env_details["robot"] == "fetch":
                mesh_prefix = "assets/temp"
            else:
                mesh_prefix = "temp"

            body_name = "swept_volume_0"

            joint_xml = (
                "" if fixed else (f'<joint name="{body_name}_free" ' f'type="free"/>')
            )

            rgba_body = [0.8, 0.8, 0.8, 1.0]
            rgba_door = [0.2, 0.2, 0.25, 1.0]
            rgba_handle = [0.8, 0.2, 0.2, 1.0]

            asset_xml = []
            geom_xml = []

            # ------------------------------------------------------------
            # Microwave body
            # ------------------------------------------------------------
            body_lx, body_ly, body_lz = map(
                float,
                self.object_details["size"],
            )

            body_half_extents = np.array(
                [
                    body_lx / 2.0,
                    body_ly / 2.0,
                    body_lz / 2.0,
                ],
                dtype=float,
            )

            body_corners = box_corners(
                body_lx,
                body_ly,
                body_lz,
            )

            body_points = []
            body_primitives = []

            for x, y, yaw in itertools.product(
                tcr_intervals["x"],
                tcr_intervals["y"],
                tcr_intervals["yaw"],
            ):
                local_position = np.array(
                    [
                        float(x),
                        float(y),
                        0.0,
                    ],
                    dtype=float,
                )

                local_orientation = yaw_rot(float(yaw))

                T_geom_primitive = make_T(
                    local_orientation,
                    local_position,
                )

                body_points.append(
                    transform_points(
                        body_corners,
                        T_geom_primitive,
                    )
                )

                body_primitives.append(
                    make_cuboid_primitive(
                        position=local_position,
                        orientation=local_orientation,
                        half_extents=body_half_extents,
                    )
                )

            if not body_points:
                raise ValueError(
                    "No cuboids were generated for the " "microwave body swept volume."
                )

            body_points = np.vstack(body_points)

            body_hull = trimesh.points.PointCloud(body_points).convex_hull

            body_mesh_name = "microwave_body_sv_mesh"

            body_mesh_filename = "microwave_body_sv.stl"

            body_mesh_file = f"{mesh_prefix}/{body_mesh_filename}"

            self._generated_assets[body_mesh_filename] = body_hull.export(
                file_type="stl"
            )

            asset_xml.append(f"""
                <mesh
                    name="{body_mesh_name}"
                    file="{body_mesh_file}"/>
                """)

            body_geom_name = "microwave_body_sv"

            geom_xml.append(f"""
                <geom
                    name="{body_geom_name}"
                    type="mesh"
                    mesh="{body_mesh_name}"
                    rgba="{' '.join(map(str, rgba_body))}"/>
                """)

            self.swept_volume_primitives[body_geom_name] = body_primitives

            # ------------------------------------------------------------
            # Microwave door and handle
            # ------------------------------------------------------------
            door_lx, door_ly, door_lz = map(
                float,
                self.object_details["door_size"],
            )

            handle_lx, handle_ly, handle_lz = map(
                float,
                self.object_details["handle_size"],
            )

            door_half_extents = np.array(
                [
                    door_lx / 2.0,
                    door_ly / 2.0,
                    door_lz / 2.0,
                ],
                dtype=float,
            )

            handle_half_extents = np.array(
                [
                    handle_lx / 2.0,
                    handle_ly / 2.0,
                    handle_lz / 2.0,
                ],
                dtype=float,
            )

            handle_pos_door = np.asarray(
                self.object_details["handle_pos_door"],
                dtype=float,
            )

            hinge_pos_body = np.asarray(
                self.object_details["hinge_pos_body"],
                dtype=float,
            )

            door_origin_closed_body = np.asarray(
                self.object_details["door_origin_closed_body"],
                dtype=float,
            )

            base_door_mesh = trimesh.creation.box(
                extents=[
                    door_lx,
                    door_ly,
                    door_lz,
                ],
            )

            base_handle_mesh = trimesh.creation.box(
                extents=[
                    handle_lx,
                    handle_ly,
                    handle_lz,
                ],
            )

            # Handle geometry is offset from the door origin.
            base_handle_mesh.apply_translation(handle_pos_door)

            door_meshes = []
            handle_meshes = []

            door_primitives = []
            handle_primitives = []

            for x, y, yaw, phi in itertools.product(
                tcr_intervals["x"],
                tcr_intervals["y"],
                tcr_intervals["yaw"],
                tcr_intervals["door"],
            ):
                R_geom_body = yaw_rot(float(yaw))

                p_geom_body = np.array(
                    [
                        float(x),
                        float(y),
                        0.0,
                    ],
                    dtype=float,
                )

                T_geom_body = make_T(
                    R_geom_body,
                    p_geom_body,
                )

                R_phi = yaw_rot(float(phi))

                T_body_hinge = make_T(
                    np.eye(3, dtype=float),
                    hinge_pos_body,
                )

                T_hinge_rotation = make_T(
                    R_phi,
                    np.zeros(3, dtype=float),
                )

                T_hinge_door = make_T(
                    np.eye(3, dtype=float),
                    door_origin_closed_body - hinge_pos_body,
                )

                T_body_door = T_body_hinge @ T_hinge_rotation @ T_hinge_door

                # This is the transform used to place the door cuboid
                # in the exported door mesh's coordinate frame.
                T_geom_door = T_geom_body @ T_body_door

                door_local_position = T_geom_door[:3, 3].copy()

                door_local_orientation = T_geom_door[:3, :3].copy()

                door_mesh = base_door_mesh.copy()
                door_mesh.apply_transform(T_geom_door)
                door_meshes.append(door_mesh)

                door_primitives.append(
                    make_cuboid_primitive(
                        position=door_local_position,
                        orientation=door_local_orientation,
                        half_extents=door_half_extents,
                    )
                )

                # The base handle mesh already contains handle_pos_door,
                # so applying T_geom_door matches the exported mesh.
                handle_mesh = base_handle_mesh.copy()
                handle_mesh.apply_transform(T_geom_door)
                handle_meshes.append(handle_mesh)

                handle_local_position = (
                    door_local_position + door_local_orientation @ handle_pos_door
                )

                handle_local_orientation = door_local_orientation

                handle_primitives.append(
                    make_cuboid_primitive(
                        position=handle_local_position,
                        orientation=handle_local_orientation,
                        half_extents=handle_half_extents,
                    )
                )

            if not door_meshes:
                raise ValueError(
                    "No cuboids were generated for the " "microwave door swept volume."
                )

            try:
                door_union = trimesh.boolean.union(
                    door_meshes,
                    engine="manifold",
                )

                handle_union = trimesh.boolean.union(
                    handle_meshes,
                    engine="manifold",
                )

            except Exception as e:
                raise RuntimeError(
                    "Boolean union failed. Install manifold3d with "
                    "`pip install manifold3d`, or adjust the sampling. "
                    f"Original error: {e}"
                ) from e

            door_mesh_name = "microwave_door_sv_mesh"

            door_mesh_filename = "microwave_door_sv_union.stl"

            door_mesh_file = f"{mesh_prefix}/{door_mesh_filename}"

            self._generated_assets[door_mesh_filename] = door_union.export(
                file_type="stl"
            )

            handle_mesh_name = "microwave_handle_sv_mesh"

            handle_mesh_filename = "microwave_handle_sv_union.stl"

            handle_mesh_file = f"{mesh_prefix}/{handle_mesh_filename}"

            self._generated_assets[handle_mesh_filename] = handle_union.export(
                file_type="stl"
            )

            asset_xml.append(f"""
                <mesh
                    name="{door_mesh_name}"
                    file="{door_mesh_file}"/>
                """)

            asset_xml.append(f"""
                <mesh
                    name="{handle_mesh_name}"
                    file="{handle_mesh_file}"/>
                """)

            door_geom_name = "microwave_door_sv"

            handle_geom_name = "microwave_handle_sv"

            hx, hy, hz = hinge_pos_body

            geom_xml.append(f"""
                <body
                    name="sv_door_hinge_frame"
                    pos="{hx} {hy} {hz}">

                    <joint
                        name="sv_door_hinge"
                        type="hinge"
                        axis="0 0 -1"
                        limited="false"/>

                    <geom
                        name="{door_geom_name}"
                        type="mesh"
                        mesh="{door_mesh_name}"
                        rgba="{' '.join(map(str, rgba_door))}"/>

                    <geom
                        name="{handle_geom_name}"
                        type="mesh"
                        mesh="{handle_mesh_name}"
                        rgba="{' '.join(map(str, rgba_handle))}"/>
                </body>
                """)

            self.swept_volume_primitives[door_geom_name] = door_primitives

            self.swept_volume_primitives[handle_geom_name] = handle_primitives

            return f"""
            <asset>
                {''.join(asset_xml)}
            </asset>

            <body name="{body_name}" pos="0 0 0">
                {joint_xml}

                {''.join(geom_xml)}
            </body>
            """

        # ================================================================
        # Cylinder swept volume
        # ================================================================
        elif self.object_details["type"] == "cylinder":

            if self.env_details["robot"] == "fetch":
                mesh_prefix = "assets/temp"
            else:
                mesh_prefix = "temp"

            radius, height = map(float, obj_size)

            x_values = np.asarray(
                tcr_intervals["x"],
                dtype=float,
            )

            y_values = np.asarray(
                tcr_intervals["y"],
                dtype=float,
            )

            if x_values.size == 0 or y_values.size == 0:
                raise ValueError(
                    "Cylinder swept volume requires nonempty " "x and y intervals."
                )

            x_min = float(x_values.min())
            x_max = float(x_values.max())
            y_min = float(y_values.min())
            y_max = float(y_values.max())

            x_center = (x_min + x_max) / 2.0
            y_center = (y_min + y_max) / 2.0

            x_span = x_max - x_min
            y_span = y_max - y_min

            identity_rotation = np.eye(3, dtype=float)

            cylinder_primitives = []

            # The swept cylinder is a rounded rectangular prism.
            # Represent its center using two overlapping cuboids.
            if x_span > 1e-12:
                cylinder_primitives.append(
                    make_cuboid_primitive(
                        position=[
                            x_center,
                            y_center,
                            0.0,
                        ],
                        orientation=identity_rotation,
                        half_extents=[
                            x_span / 2.0,
                            y_span / 2.0 + radius,
                            height / 2.0,
                        ],
                    )
                )

            if y_span > 1e-12:
                cylinder_primitives.append(
                    make_cuboid_primitive(
                        position=[
                            x_center,
                            y_center,
                            0.0,
                        ],
                        orientation=identity_rotation,
                        half_extents=[
                            x_span / 2.0 + radius,
                            y_span / 2.0,
                            height / 2.0,
                        ],
                    )
                )

            # Add cylinders at the corners of the x/y sweep.
            corner_positions = {
                (x_min, y_min),
                (x_min, y_max),
                (x_max, y_min),
                (x_max, y_max),
            }

            for x, y in corner_positions:
                cylinder_primitives.append(
                    make_cylinder_primitive(
                        position=[x, y, 0.0],
                        orientation=identity_rotation,
                        radius=radius,
                        length=height,
                    )
                )

            # Construct the STL from cylinders placed at the corners.
            # Their convex hull is the rounded rectangular sweep.
            base_cylinder_mesh = trimesh.creation.cylinder(
                radius=radius,
                height=height,
                sections=64,
            )

            cylinder_vertices = []

            for x, y in corner_positions:
                cylinder_mesh = base_cylinder_mesh.copy()

                cylinder_mesh.apply_translation([x, y, 0.0])

                cylinder_vertices.append(cylinder_mesh.vertices.copy())

            if not cylinder_vertices:
                raise ValueError(
                    "No cylinders were generated for the " "cylinder swept volume."
                )

            cylinder_vertices = np.vstack(cylinder_vertices)

            hull = trimesh.points.PointCloud(cylinder_vertices).convex_hull

            mesh_file_name = f"sv_mesh_{sv_count}_{os.getpid()}.stl"

            self._generated_assets[mesh_file_name] = hull.export(file_type="stl")

            mesh_path_for_xml = f"{mesh_prefix}/{mesh_file_name}"

            mesh_name = f"swept_volume_mesh_{sv_count}"

            body_name = f"swept_volume_{sv_count}"

            geom_name = f"sv_mesh_{sv_count}"

            rgba = [0.8, 0.8, 0.8, 1.0]

            joint_xml = (
                "" if fixed else (f'<joint name="{body_name}_free" ' f'type="free"/>')
            )

            self.swept_volume_primitives[geom_name] = cylinder_primitives

            return f"""
            <asset>
                <mesh
                    name="{mesh_name}"
                    file="{mesh_path_for_xml}"/>
            </asset>

            <body name="{body_name}" pos="0 0 0">
                {joint_xml}

                <geom
                    name="{geom_name}"
                    type="mesh"
                    mesh="{mesh_name}"
                    rgba="{' '.join(map(str, rgba))}"/>
            </body>
            """

        else:
            raise ValueError(
                "Unsupported object type for swept-volume generation: "
                f"{self.object_details['type']}"
            )

    def move_swept_volume(self, object_configs):
        """Move selected swept volume to desired bin, and dummy out unused SVs."""

        dummy_config = [[1, 1], [1, 1], [0, 0], [0, 0]]

        def move_one_sv(sv_joint_name, configs):
            svid = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                sv_joint_name,
            )

            if svid == -1:
                raise RuntimeError(f"Could not find joint {sv_joint_name}")

            sv_adr = self.model.jnt_qposadr[svid]
            sv_vadr = self.model.jnt_dofadr[svid]

            configs = np.asarray(configs, dtype=np.float64)

            if configs.ndim == 2:
                configs = configs[None, :, :]

            x_lower = configs[:, 0, 0]
            x_upper = configs[:, 0, 1]
            y_lower = configs[:, 1, 0]
            y_upper = configs[:, 1, 1]
            z = configs[:, 2, 0]
            yaw_lower = configs[:, 3, 0]
            yaw_upper = configs[:, 3, 1]

            cx = 0.5 * (x_upper + x_lower)
            cy = 0.5 * (y_upper + y_lower)
            cyaw = 0.5 * (yaw_lower + yaw_upper)

            new_pos = [cx[0], cy[0], z[0]]

            half = 0.5 * cyaw[0]
            new_quat = [
                np.cos(half),
                0.0,
                0.0,
                np.sin(half),
            ]

            self.data.qpos[sv_adr : sv_adr + 7] = [
                new_pos[0],
                new_pos[1],
                new_pos[2],
                new_quat[0],
                new_quat[1],
                new_quat[2],
                new_quat[3],
            ]

            self.data.qvel[sv_vadr : sv_vadr + 6] = 0

        if self.environment_name == "allstable":
            face_to_joint = {
                "xy": "swept_volume_0_free",
                "yz": "swept_volume_1_free",
                "zx": "swept_volume_2_free",
            }

            face_in_contact = object_configs[0]

            if face_in_contact not in face_to_joint:
                raise ValueError(f"Unknown face value: {face_in_contact}")

            real_object_configs = object_configs[1:]

            for face, sv_joint_name in face_to_joint.items():
                if face == face_in_contact:
                    move_one_sv(sv_joint_name, real_object_configs)
                else:
                    move_one_sv(sv_joint_name, dummy_config)

        else:
            move_one_sv("swept_volume_0_free", object_configs)

        if self.environment_name == "microwave":
            object_configs = np.asarray(object_configs, dtype=np.float64)

            if object_configs.ndim == 2:
                object_configs = object_configs[None, :, :]

            door_lower = object_configs[:, 4, 0]
            door_upper = object_configs[:, 4, 1]
            door_ang = 0.5 * (door_lower + door_upper)

            self.move_xml_joint("sv_door_hinge", door_ang[0])

        mujoco.mj_forward(self.model, self.data)
