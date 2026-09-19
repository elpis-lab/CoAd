"""Cube scenes spanning all stable contact faces."""

import numpy as np
import yaml
from coad.tcr import create_tcr_set
from .base import MujocoEnv


class AllStableEnv(MujocoEnv):
    """Table environment with a cube object in all stable configurations"""

    environment_name = "allstable"

    def __init__(self, robot, using_swept_volume=True, compute_tcr=True):
        """Initialize the AllStable environment"""
        super().__init__(robot)

        # Problem parameters
        object_type = "box"
        object_size = [0.04, 0.06, 0.03]
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

        scene_yaml = "configs/scenes/table/scene_empty_table.yaml"
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
            "allstable",
            robot_pos,
            robot_quat,
            outer_rad,
            inner_rad,
        )
        self.env_details["tcr_batches"] = ["xy", "yz", "zx"]

        # Prepare grasp details
        super().populate_grasp_details(yaw_buffer=yaw_buffer, grasp_type="top")
        tcr_intervals = (
            super().construct_tcr() if (compute_tcr or using_swept_volume) else None
        )

        # Prepare swept volume (or object geom for validation)
        xmls_to_add = []
        if using_swept_volume:

            sv1_xml = super().create_swept_volume(
                tcr_intervals["xy"], object_size, sv_count=0
            )

            sv2_xml = super().create_swept_volume(
                tcr_intervals["yz"],
                [object_size[1], object_size[2], object_size[0]],
                sv_count=1,
            )

            sv3_xml = super().create_swept_volume(
                tcr_intervals["zx"],
                [object_size[2], object_size[0], object_size[1]],
                sv_count=2,
            )
            xmls_to_add.extend([sv1_xml, sv2_xml, sv3_xml])
        else:
            if self.object_details["type"] == "box":
                xml1 = super().cube_object_xml(
                    [object_size[0], object_size[1], object_size[2]],
                    [1, 1, 0, 0],
                    name="cube_object_0",
                )
                xml2 = super().cube_object_xml(
                    [object_size[1], object_size[2], object_size[0]],
                    [1, 1, 0, 0],
                    name="cube_object_1",
                )
                xml3 = super().cube_object_xml(
                    [object_size[2], object_size[0], object_size[1]],
                    [1, 1, 0, 0],
                    name="cube_object_2",
                )

                xmls_to_add.extend([xml1, xml2, xml3])
            else:
                raise ValueError(
                    f"Currently unsupported object type: {self.object_details['type']}"
                )

        # Prepare environment xmls
        table_xml = super().build_xml(scene_yaml, parent_body_name="scene_table")
        xmls_to_add.append(table_xml)

        free_xml_path = f"{self.robot_dir}/empty_table_scene.xml"
        self.model, self.data = super().build_model(free_xml_path, xmls_to_add)

    def generate_task_set(self):
        """Generate task set/TSRs"""

        TCR_set = {}

        for i, face_in_contact in enumerate(["xy", "yz", "zx"]):
            TCR_set.update(
                {
                    (face_in_contact,) + key: value
                    for key, value in create_tcr_set(
                        self, batch_idx=face_in_contact
                    ).items()
                }
            )

        self.task_set = TCR_set
        return self.task_set
