"""Articulated microwave scene and grasp configuration."""

import numpy as np
import yaml
from .base import MujocoEnv


class MicrowaveEnv(MujocoEnv):
    """Table environment with a microwave object"""

    environment_name = "microwave"

    def __init__(self, robot, using_swept_volume=True, compute_tcr=True):
        """Initialize the microwave environment"""
        super().__init__(robot)

        # Problem parameters
        object_type = "microwave"
        object_size = [0.26, 0.24, 0.21]
        yaw_variation = [-0.5 * np.pi, 0.5 * np.pi]
        yaw_buffer = 4 * (np.pi / 180)
        door_buffer = 2 * (np.pi / 180)

        door_size = [0.012, 0.250, 0.210]

        # Prepare target object
        object_variation = {
            "x": [[-0.8, 0.8]],
            "y": [[-0.8, 0.8]],
            "z": [[object_size[2] / 2, object_size[2] / 2]],
            "yaw": [yaw_variation],
            "door": [[0, np.pi / 2]],
        }

        self.populate_object_details(object_type, object_size, object_variation)
        self.object_details["door_size"] = door_size
        self.object_details["handle_size"] = [
            door_size[0] / 0.75,
            door_size[1] * 0.08,
            door_size[2] * 0.6,
        ]
        self.object_details["handle_size"] = [
            door_size[0] / 0.25,
            door_size[1] * 0.1,
            door_size[2] * 0.7,
        ]

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
        robot_quat = self.quat_xyzw_to_wxyz(
            config_yaml_data["base_offset"]["orientation"]
        )

        if robot == "panda" or robot == "fetch":
            inner_rad = 0.3
            outer_rad = 0.8
        elif robot == "ur10":
            inner_rad = 0.3
            outer_rad = 0.75

        self.populate_env_details(
            scene_yaml,
            robot,
            "microwave",
            robot_pos,
            robot_quat,
            outer_rad,
            inner_rad,
        )

        # Prepare grasp details
        self.populate_grasp_details(
            yaw_buffer=yaw_buffer, door_buffer=door_buffer, grasp_type="front"
        )
        # Evaluation uses stored tasks and actual object geometry; no TCR search is needed.
        tcr_intervals = (
            self.construct_tcr() if (compute_tcr or using_swept_volume) else None
        )

        # Prepare swept volume (or object geom for validation)
        xmls_to_add = []
        if using_swept_volume:
            xml = super().create_swept_volume(tcr_intervals)
        else:
            xml = self.object_xml(self.object_details["size"], [1, 1, 0, 0])
        xmls_to_add.append(xml)

        # Prepare environment xmls
        table_xml = super().build_xml(scene_yaml, parent_body_name="scene_table")
        xmls_to_add.append(table_xml)

        # xmls_to_add.append(inner_xml)
        # xmls_to_add.append(outer_xml)

        free_xml_path = f"{self.robot_dir}/empty_table_scene.xml"
        self.model, self.data = super().build_model(free_xml_path, xmls_to_add)

    def populate_object_details(self, object_type, object_size, object_variation):

        if object_type != "microwave":
            raise ValueError(
                f"MicrowaveEnv only supports microwave objects. Unsupported object type: {object_type}"
            )

        for variation_axis in object_variation:
            if variation_axis not in ["x", "y", "z", "yaw", "door"]:
                raise ValueError(
                    f"Unsupported variation given for {object_type}: {variation_axis}"
                )

        self.object_details = {
            "variation": object_variation,
            "type": object_type,
            "size": object_size,
        }

    def populate_grasp_details(
        self,
        alpha=0.95,
        yaw_buffer=6 * (np.pi / 180),
        door_buffer=1 * (np.pi / 180),
        grasp_type="front",
    ):

        self.grasp_details = {
            "type": grasp_type,
            "alpha": alpha,
            "yaw_buffer": yaw_buffer,
            "door_buffer": door_buffer,
        }
