"""Free, box, cage, table and large-object scenes."""

import numpy as np
import yaml
from .base import MujocoEnv


class FreeEnv(MujocoEnv):
    """Free environment"""

    environment_name = "free"

    def __init__(self, robot, using_swept_volume=True, compute_tcr=True):
        """Initialize free environment"""
        super().__init__(robot)

        # Problem parameters
        object_type = "box"
        object_size = [0.03, 0.03, 0.15]
        yaw_variation = [-0.5 * np.pi, 0.5 * np.pi]
        yaw_buffer = 6 * (np.pi / 180)

        # Prepare target object
        object_variation = {
            "x": [[-0.8, 0.8]],
            "y": [[-0.8, 0.8]],
            "z": [[object_size[2] / 2, object_size[2] / 2]],
            "yaw": [yaw_variation],
        }
        super().populate_object_details(object_type, object_size, object_variation)

        # Prepare environment details
        if robot == "panda":
            robot_pos = [0, 0, 0]
            robot_quat = [1, 0, 0, 0]
            outer_rad = 0.7
            inner_rad = 0.3
        elif robot == "fetch":
            robot_pos = [0, 0, 0.005]
            robot_quat = [1, 0, 0, 0]
            outer_rad = 0.7
            inner_rad = 0.3
        elif robot == "ur10":
            robot_pos = [0, 0, 0]
            robot_quat = [1, 0, 0, 0]
            outer_rad = 0.8
            inner_rad = 0.3
        elif robot == "g1":
            robot_pos = [0, 0, 0]
            robot_quat = [1, 0, 0, 0]
            outer_rad = 0.8
            inner_rad = 0.3
        else:
            raise ValueError(f"Unsupported robot for FreeEnv: {robot}")

        super().populate_env_details(
            None, robot, "free", robot_pos, robot_quat, outer_rad, inner_rad
        )

        # Prepare grasp details
        super().populate_grasp_details(yaw_buffer=yaw_buffer, grasp_type="top")
        tcr_intervals = (
            super().construct_tcr() if (compute_tcr or using_swept_volume) else None
        )

        # Prepare swept volume (or object geom for validation)
        xmls_to_add = []
        if using_swept_volume:
            xml = super().create_swept_volume(tcr_intervals)

        else:
            if self.object_details["type"] == "box":
                xml = super().cube_object_xml(self.object_details["size"], [1, 1, 0, 0])
            else:
                raise ValueError(
                    f"Currently unsupported object type: {self.object_details['type']}"
                )
        xmls_to_add.append(xml)

        free_xml_path = f"{self.robot_dir}/free_scene.xml"
        self.model, self.data = super().build_model(free_xml_path, xmls_to_add)

        if using_swept_volume:
            self.move_swept_volume(
                [
                    [1, 1],
                    [1, 1],
                    [object_size[2] / 2, object_size[2] / 2],
                    [0, 0],
                ]
            )
        else:
            self.move_cube_object([1, 1, object_size[2] / 2, 0])


class BoxEnv(MujocoEnv):
    """Box environment"""

    environment_name = "box"

    def __init__(self, robot, using_swept_volume=True, compute_tcr=True):
        """Initialize box environment"""
        super().__init__(robot)

        # Problem parameters
        object_type = "box"
        object_size = [0.03, 0.03, 0.15]
        yaw_variation = [-0.5 * np.pi, 0.5 * np.pi]
        yaw_buffer = 6 * (np.pi / 180)

        # Prepare target object
        object_variation = {
            "x": [[-0.8, 0.8]],
            "y": [[-0.8, 0.8]],
            "z": [[object_size[2] / 2, object_size[2] / 2]],
            "yaw": [yaw_variation],
        }
        super().populate_object_details(object_type, object_size, object_variation)

        # Prepare environment details
        if robot == "panda":
            config_yaml = "configs/problems/box_panda.yaml"
        elif robot == "fetch":
            config_yaml = "configs/problems/box_fetch.yaml"
        elif robot == "ur10":
            config_yaml = "configs/problems/box_ur5.yaml"
        elif robot == "g1":
            config_yaml = "configs/problems/box_g1.yaml"

        scene_yaml = "configs/scenes/box/scene_box.yaml"
        with open(config_yaml, "r") as f:
            config_yaml_data = yaml.safe_load(f)

        robot_pos = config_yaml_data["base_offset"]["position"]
        robot_quat = super().quat_xyzw_to_wxyz(
            config_yaml_data["base_offset"]["orientation"]
        )
        outer_rad = 0.75
        inner_rad = 0.3

        super().populate_env_details(
            scene_yaml,
            robot,
            "box",
            robot_pos,
            robot_quat,
            outer_rad,
            inner_rad,
        )

        # Prepare grasp details
        super().populate_grasp_details(yaw_buffer=yaw_buffer, grasp_type="top")
        tcr_intervals = (
            super().construct_tcr() if (compute_tcr or using_swept_volume) else None
        )

        xmls_to_add = []
        if using_swept_volume:
            xml = super().create_swept_volume(tcr_intervals)
        else:
            if self.object_details["type"] == "box":
                xml = super().cube_object_xml(self.object_details["size"], [1, 1, 0, 0])
            else:
                raise ValueError(
                    f"Currently unsupported object type: {self.object_details['type']}"
                )
        xmls_to_add.append(xml)

        # Prepare environment xmls
        box_xml = super().build_xml(
            scene_yaml, parent_body_name="scene_box", skip_ids={"Can1"}
        )
        xmls_to_add.append(box_xml)

        free_xml_path = f"{self.robot_dir}/box_scene.xml"
        self.model, self.data = super().build_model(free_xml_path, xmls_to_add)


class CageEnv(MujocoEnv):
    """Cage environment"""

    environment_name = "cage"

    def __init__(self, robot, using_swept_volume=True, compute_tcr=True):
        """Initialize cage environment"""
        super().__init__(robot)

        # Problem parameters
        object_type = "box"
        object_size = [0.03, 0.03, 0.15]
        yaw_variation = [-0.5 * np.pi, 0.5 * np.pi]
        yaw_buffer = 6 * (np.pi / 180)

        # Prepare target object
        object_variation = {
            "x": [[-0.8, 0.8]],
            "y": [[-0.8, 0.8]],
            "z": [[object_size[2] / 2, object_size[2] / 2]],
            "yaw": [yaw_variation],
        }
        super().populate_object_details(object_type, object_size, object_variation)

        # Prepare environment details
        if robot == "panda":
            config_yaml = "configs/problems/cage_panda.yaml"
        elif robot == "fetch":
            config_yaml = "configs/problems/cage_fetch.yaml"
        elif robot == "ur10":
            config_yaml = "configs/problems/cage_ur5.yaml"

        scene_yaml = "configs/scenes/cage/scene_cage.yaml"
        with open(config_yaml, "r") as f:
            config_yaml_data = yaml.safe_load(f)

        robot_pos = config_yaml_data["base_offset"]["position"]
        robot_quat = super().quat_xyzw_to_wxyz(
            config_yaml_data["base_offset"]["orientation"]
        )
        outer_rad = 0.75
        inner_rad = 0.3

        super().populate_env_details(
            scene_yaml,
            robot,
            "cage",
            robot_pos,
            robot_quat,
            outer_rad,
            inner_rad,
        )

        # Prepare grasp details
        super().populate_grasp_details(yaw_buffer=yaw_buffer, grasp_type="top")
        tcr_intervals = (
            super().construct_tcr() if (compute_tcr or using_swept_volume) else None
        )

        xmls_to_add = []
        if using_swept_volume:
            xml = super().create_swept_volume(tcr_intervals)
        else:
            if self.object_details["type"] == "box":
                xml = super().cube_object_xml(self.object_details["size"], [1, 1, 0, 0])
            else:
                raise ValueError(
                    f"Currently unsupported object type: {self.object_details['type']}"
                )
        xmls_to_add.append(xml)

        # Prepare environment xmls
        cage_xml = super().build_xml(
            scene_yaml, parent_body_name="scene_cage", skip_ids={"Cube1"}
        )
        xmls_to_add.append(cage_xml)

        free_xml_path = f"{self.robot_dir}/cage_scene.xml"
        self.model, self.data = super().build_model(free_xml_path, xmls_to_add)


class TableEnv(MujocoEnv):
    """Table environment"""

    environment_name = "table"

    def __init__(self, robot, using_swept_volume=True, compute_tcr=True):
        """Initialize table environment"""
        super().__init__(robot)

        # Problem parameters
        object_type = "box"
        object_size = [0.03, 0.03, 0.15]
        yaw_variation = [-0.5 * np.pi, 0.5 * np.pi]
        yaw_buffer = 6 * (np.pi / 180)

        # Prepare target object
        object_variation = {
            "x": [[-0.8, 0.8]],
            "y": [[-0.8, 0.8]],
            "z": [[object_size[2] / 2, object_size[2] / 2]],
            "yaw": [yaw_variation],
        }
        super().populate_object_details(object_type, object_size, object_variation)

        # Prepare environment details
        if robot == "panda":
            config_yaml = "configs/problems/table_pick_panda.yaml"
        elif robot == "fetch":
            config_yaml = "configs/problems/table_pick_fetch.yaml"
        elif robot == "ur10":
            config_yaml = "configs/problems/table_pick_ur5.yaml"

        scene_yaml = "configs/scenes/table/scene_table.yaml"
        with open(config_yaml, "r") as f:
            config_yaml_data = yaml.safe_load(f)

        robot_pos = config_yaml_data["base_offset"]["position"]
        robot_quat = super().quat_xyzw_to_wxyz(
            config_yaml_data["base_offset"]["orientation"]
        )

        if robot == "panda" or robot == "fetch":
            inner_rad = 0.3
            outer_rad = 0.7
        elif robot == "ur10":
            inner_rad = 0.3
            outer_rad = 0.75

        super().populate_env_details(
            scene_yaml,
            robot,
            "table",
            robot_pos,
            robot_quat,
            outer_rad,
            inner_rad,
        )

        # Prepare grasp details
        super().populate_grasp_details(yaw_buffer=yaw_buffer, grasp_type="top")
        tcr_intervals = (
            super().construct_tcr() if (compute_tcr or using_swept_volume) else None
        )

        xmls_to_add = []
        if using_swept_volume:
            xml = super().create_swept_volume(tcr_intervals)
        else:
            if self.object_details["type"] == "box":
                xml = super().cube_object_xml(self.object_details["size"], [1, 1, 0, 0])
            else:
                raise ValueError(
                    f"Currently unsupported object type: {self.object_details['type']}"
                )
        xmls_to_add.append(xml)

        # Prepare environment xmls
        table_xml = super().build_xml(
            scene_yaml, parent_body_name="scene_table", skip_ids={"Cube1"}
        )
        xmls_to_add.append(table_xml)

        free_xml_path = f"{self.robot_dir}/table_scene.xml"
        self.model, self.data = super().build_model(free_xml_path, xmls_to_add)


class LargeObjectEnv(MujocoEnv):
    """Table environment with a large target object"""

    environment_name = "largeobj"

    def __init__(self, robot, using_swept_volume=True, compute_tcr=True):
        """Initialize large object environment"""
        super().__init__(robot)

        # Problem parameters
        object_type = "box"
        object_size = [0.3, 0.04, 0.04]
        yaw_variation = [-0.5 * np.pi, 0.5 * np.pi]
        yaw_buffer = 0.5 * (np.pi / 180)

        # Prepare target object
        object_variation = {
            "x": [[-0.8, 0.8]],
            "y": [[-0.8, 0.8]],
            "z": [[object_size[2] / 2, object_size[2] / 2]],
            "yaw": [yaw_variation],
        }
        super().populate_object_details(object_type, object_size, object_variation)

        # Prepare environment details
        if robot == "panda":
            config_yaml = "configs/problems/table_pick_panda.yaml"
        elif robot == "fetch":
            config_yaml = "configs/problems/table_pick_fetch.yaml"
        elif robot == "ur10":
            config_yaml = "configs/problems/table_pick_ur5.yaml"

        scene_yaml = "configs/scenes/table/scene_empty_table.yaml"
        with open(config_yaml, "r") as f:
            config_yaml_data = yaml.safe_load(f)

        robot_pos = config_yaml_data["base_offset"]["position"]
        robot_quat = super().quat_xyzw_to_wxyz(
            config_yaml_data["base_offset"]["orientation"]
        )

        if robot == "fetch":
            robot_pos = [robot_pos[0] - 0.05, robot_pos[1], robot_pos[2]]

        if robot == "panda":
            inner_rad = 0.3
            outer_rad = 0.7
        elif robot == "fetch":
            inner_rad = 0.3
            outer_rad = 0.75
        elif robot == "ur10":
            inner_rad = 0.3
            outer_rad = 0.75

        super().populate_env_details(
            scene_yaml,
            robot,
            "largeobj",
            robot_pos,
            robot_quat,
            outer_rad,
            inner_rad,
        )

        # Prepare grasp details
        super().populate_grasp_details(yaw_buffer=yaw_buffer, grasp_type="top")
        tcr_intervals = (
            super().construct_tcr() if (compute_tcr or using_swept_volume) else None
        )

        # Prepare swept volume (or object geom for validation)
        xmls_to_add = []
        if using_swept_volume:
            xml = super().create_swept_volume(tcr_intervals)
        else:
            if self.object_details["type"] == "box":
                xml = super().cube_object_xml(self.object_details["size"], [1, 1, 0, 0])
            else:
                raise ValueError(
                    f"Currently unsupported object type: {self.object_details['type']}"
                )
        xmls_to_add.append(xml)

        # Prepare environment xmls
        table_xml = super().build_xml(scene_yaml, parent_body_name="scene_table")
        xmls_to_add.append(table_xml)

        free_xml_path = f"{self.robot_dir}/empty_table_scene.xml"
        self.model, self.data = super().build_model(free_xml_path, xmls_to_add)
