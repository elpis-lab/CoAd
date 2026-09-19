"""Conveyor scene and configured home poses."""

import numpy as np
import yaml
from .base import MujocoEnv


class ConveyorEnv(MujocoEnv):
    """Goal-varying top-grasp task over the conveyor-belt surface."""

    environment_name = "conveyor"

    HOME_QPOS = {
        "panda": [
            -0.014131995359767748,
            -0.13770028726210812,
            0.2653771985668798,
            -1.825910196727267,
            0.03627906223844136,
            1.692893099807465,
            1.0298256088739774,
        ],
        "fetch": [
            0.193075,
            0.7933781260752903,
            -0.37236089961590363,
            -2.0896605498174083,
            1.3211077116192844,
            1.2841872246175499,
            2.138250002656627,
            2.429858012425777,
        ],
    }

    def __init__(self, robot, using_swept_volume=True, compute_tcr=True):
        if robot not in self.HOME_QPOS:
            raise ValueError(
                f"Unsupported ConveyorEnv robot: {robot}. "
                "Expected 'panda' or 'fetch'."
            )

        super().__init__(robot)

        # This is the red cuboid used by the conveyor experiment figure.
        object_type = "box"
        object_size = [0.03, 0.03, 0.10]
        object_variation = {
            "x": [[-0.8, 0.8]],
            "y": [[-0.8, 0.8]],
            "z": [[object_size[2] / 2.0, object_size[2] / 2.0]],
            # A square footprint is unique over a half turn.
            "yaw": [[-0.5 * np.pi, 0.5 * np.pi]],
        }
        super().populate_object_details(
            object_type,
            object_size,
            object_variation,
        )

        config_yaml = f"configs/problems/conveyor_pick_{robot}.yaml"
        scene_yaml = "configs/scenes/conveyor/scene_conveyor.yaml"
        with open(config_yaml, "r") as file:
            config_data = yaml.safe_load(file)

        robot_pos = config_data["base_offset"]["position"]
        robot_quat = super().quat_xyzw_to_wxyz(
            config_data["base_offset"]["orientation"]
        )
        outer_rad = 0.75 if robot == "panda" else 0.80
        inner_rad = 0.20

        super().populate_env_details(
            scene_yaml,
            robot,
            "conveyor",
            robot_pos,
            robot_quat,
            outer_rad,
            inner_rad,
        )
        self.home_qpos = np.asarray(self.HOME_QPOS[robot], dtype=float)

        super().populate_grasp_details(
            yaw_buffer=6 * (np.pi / 180),
            grasp_type="top",
        )
        tcr_intervals = (
            super().construct_tcr() if (compute_tcr or using_swept_volume) else None
        )

        if using_swept_volume:
            object_xml = super().create_swept_volume(tcr_intervals)
        else:
            object_xml = super().cube_object_xml(
                self.object_details["size"],
                [1, 0, 0, 1],
            )

        environment_xml = super().build_xml(
            scene_yaml,
            parent_body_name="scene_conveyor",
        )
        free_xml_path = f"{self.robot_dir}/conveyor_scene.xml"
        self.model, self.data = super().build_model(
            free_xml_path,
            [object_xml, environment_xml],
        )
