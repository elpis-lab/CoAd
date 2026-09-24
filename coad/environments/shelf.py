"""Shelf scene and placement intervals."""

import numpy as np
import yaml
from .base import MujocoEnv


class ShelfEnv(MujocoEnv):
    """Thin shelf environment"""

    environment_name = "shelf"

    def __init__(self, robot, using_swept_volume=True, compute_tcr=True):
        """Initialize the thin shelf environment."""
        super().__init__(robot)

        # Object details
        object_type = "box"
        object_size = [0.03, 0.03, 0.15]

        yaw_variation = [
            -0.5 * np.pi,
            0.5 * np.pi,
        ]

        object_variation = {
            "x": [[-0.8, 0.8]],
            "y": [[-0.8, 0.8]],
            "z": [
                [
                    object_size[2] / 2.0,
                    object_size[2] / 2.0,
                ]
            ],
            "yaw": [yaw_variation],
        }

        super().populate_object_details(
            object_type,
            object_size,
            object_variation,
        )

        # Environment details
        if robot == "panda":
            config_yaml = "configs/problems/" "bookshelf_thin_panda.yaml"
            inner_rad = 0.3
            outer_rad = 0.75

        elif robot == "fetch":
            config_yaml = "configs/problems/" "bookshelf_thin_fetch.yaml"
            inner_rad = 0.3
            outer_rad = 0.75

        elif robot == "ur10":
            # Preserve the existing configuration filename.
            config_yaml = "configs/problems/" "bookshelf_thin_ur5.yaml"
            inner_rad = 0.3
            outer_rad = 0.65

        else:
            raise ValueError(f"Unsupported ShelfEnv robot: {robot}")

        scene_yaml = "configs/scenes/bookshelf/" "scene_thin.yaml"

        with open(config_yaml, "r") as file:
            config_data = yaml.safe_load(file)

        robot_pos = config_data["base_offset"]["position"]

        robot_quat = super().quat_xyzw_to_wxyz(
            config_data["base_offset"]["orientation"]
        )

        super().populate_env_details(
            scene_yaml=scene_yaml,
            robot_name=robot,
            env_name="shelf",
            robot_pos=robot_pos,
            robot_quat=robot_quat,
            outer_rad=outer_rad,
            inner_rad=inner_rad,
        )

        # Grasp details
        super().populate_grasp_details(
            yaw_buffer=6 * (np.pi / 180),
            grasp_type="front",
        )

        tcr_intervals = (
            super().construct_tcr() if (compute_tcr or using_swept_volume) else None
        )

        if using_swept_volume:
            object_xml = super().create_swept_volume(
                tcr_intervals,
            )
        else:
            object_xml = super().cube_object_xml(
                self.object_details["size"],
                [0, 0, 0, 0],
            )

        shelf_xml = super().build_xml(
            scene_yaml,
            parent_body_name="scene_shelf",
            skip_ids={"Cube1"},
        )

        xmls_to_add = [
            object_xml,
            shelf_xml,
        ]

        free_xml_path = f"{self.robot_dir}/shelf_scene.xml"

        self.model, self.data = super().build_model(
            free_xml_path,
            xmls_to_add,
        )

    def find_problem_intervals(
        self, scene_yaml, bases, wall_clearance, dividing_wall_clearance
    ):
        with open(scene_yaml, "r") as f:
            scene_yaml_data = yaml.safe_load(f)
        objs = scene_yaml_data["world"]["collision_objects"]

        base_dim = None
        base_pos = None

        for obj in objs:
            obj_id = obj.get("id", "")
            if obj_id != bases[0]:
                continue
            base_dim = obj["primitives"][0]["dimensions"]
            base_pos = obj["primitive_poses"][0]["position"]

        base_zpos = []
        base_zdim = []
        base_names = []
        for obj in objs:
            obj_id = obj.get("id", "")
            if obj_id not in bases:
                continue
            base_zpos.append(obj["primitive_poses"][0]["position"][2])
            base_zdim.append(obj["primitives"][0]["dimensions"][2])
            base_names.append(obj_id)

        z_correction = []
        for base_idx in range(len(base_zpos)):
            curr_zpos = base_zpos[base_idx]
            curr_zdim = base_zdim[base_idx]
            z_correction.append(curr_zpos + (curr_zdim / 2))

        self.env_details["z_correction"] = z_correction

        hx_int = base_dim[0] / 2 - wall_clearance / 2
        hy_int = base_dim[1] / 2 - wall_clearance / 2

        shelf_xmin = base_pos[0] - hx_int + self.object_details["size"][0] / 2
        shelf_xmax = base_pos[0] + hx_int - self.object_details["size"][0] / 2
        shelf_ymin = base_pos[1] - hy_int + self.object_details["size"][1] / 2
        shelf_ymax = base_pos[1] + hy_int - self.object_details["size"][1] / 2

        for obj in objs:
            obj_id = obj.get("id", "")
            if obj_id != "shelf_vert":
                continue
            wall_dim = obj["primitives"][0]["dimensions"]
            wall_pos = obj["primitive_poses"][0]["position"]
        wx_int = wall_dim[0] / 2 + dividing_wall_clearance / 2
        wy_int = wall_dim[1] / 2 + dividing_wall_clearance / 2

        wall_xmin = wall_pos[0] - wx_int - self.object_details["size"][0] / 2
        wall_xmax = wall_pos[0] + wx_int + self.object_details["size"][0] / 2
        wall_ymin = wall_pos[1] - wy_int - self.object_details["size"][1] / 2
        wall_ymax = wall_pos[1] + wy_int + self.object_details["size"][1] / 2

        fxmin = max(shelf_xmin, wall_xmin)
        fxmax = min(shelf_xmax, wall_xmax)
        fymin = max(shelf_ymin, wall_ymin)
        fymax = min(shelf_ymax, wall_ymax)

        regions = []

        # If wall doesn't overlap the cage at all, nothing to subtract
        if fxmin >= fxmax or fymin >= fymax:
            regions.append([[shelf_xmin, shelf_xmax], [shelf_ymin, shelf_ymax]])
        else:
            # Subtract forbidden rectangle from cage rectangle.
            # This can produce up to 4 rectangles; if your wall "cuts the shelf in half" you'll typically get 2.

            # Left slab
            if shelf_xmin < fxmin:
                regions.append([[shelf_xmin, fxmin], [shelf_ymin, shelf_ymax]])

            # Right slab
            if fxmax < shelf_xmax:
                regions.append([[fxmax, shelf_xmax], [shelf_ymin, shelf_ymax]])

            # Bottom slab
            if shelf_ymin < fymin:
                regions.append([[fxmin, fxmax], [shelf_ymin, fymin]])

            # Top slab
            if fymax < shelf_ymax:
                regions.append([[fxmin, fxmax], [fymax, shelf_ymax]])

        # Optional: keep only non-degenerate regions (numerical safety)
        eps = 1e-9
        regions = [
            r
            for r in regions
            if (r[0][1] - r[0][0] > eps) and (r[1][1] - r[1][0] > eps)
        ]
        return regions
