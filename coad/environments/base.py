"""Shared MuJoCo environment state and in-memory model compilation."""

import mujoco
from pathlib import Path
from .task_regions import TaskRegions
from .objects import SceneObjects
from .swept_volume import SweptVolumes


class MujocoEnv(TaskRegions, SceneObjects, SweptVolumes):
    environment_name = None

    def __init__(self, robot, custom_base=None):
        """Initialize object dimensions and common parameters"""
        self.swept_volume_primitives = {}
        self._generated_assets = {}
        if robot == "panda":
            self.robot_dir = "assets/franka_emika_panda"
        else:
            self.robot_dir = f"assets/{robot}"
        if custom_base is None:
            if robot == "g1":
                self.base_xml = "scene_23dof.xml"
            else:
                self.base_xml = "scene.xml"
        else:
            self.base_xml = custom_base

    def populate_object_details(self, object_type, object_size, object_variation):

        if object_type == "box":
            if len(object_size) == 3:
                self.object_details = {
                    "type": object_type,
                    "size": object_size,
                }
            else:
                raise ValueError("Object size must be length 3 for boxes")

            for variation_axis in object_variation:
                if variation_axis not in ["x", "y", "z", "yaw"]:
                    raise ValueError(
                        f"Unsupported variation given for {object_type} type: {variation_axis}"
                    )
            self.object_details["variation"] = object_variation

        elif object_type == "cylinder":
            if len(object_size) == 2:
                self.object_details = {
                    "type": object_type,
                    "size": object_size,
                }
            else:
                raise ValueError("Object size must be length 2 for cylinders")

            for variation_axis in object_variation:
                if variation_axis == "yaw":
                    object_variation["yaw"] = [[0, 0]]
                if variation_axis not in ["x", "y", "z", "yaw"]:
                    raise ValueError(
                        f"Unsupported variation given for {object_type} type: {variation_axis}"
                    )
            self.object_details["variation"] = object_variation

        else:
            raise ValueError(f"Unsupported object type: {object_type}")

    def populate_env_details(
        self,
        scene_yaml,
        robot_name,
        env_name,
        robot_pos,
        robot_quat,
        outer_rad,
        inner_rad,
    ):

        self.env_details = {
            "robot_pos": robot_pos,
            "robot_quat": robot_quat,
            "robot": robot_name,
            "env_name": env_name,
            "outer_rad": outer_rad,
            "inner_rad": inner_rad,
            "collision_geoms": [],
        }

        if env_name in ["box", "cage"]:
            intervals = self.find_problem_intervals(
                scene_yaml, base_name="base", wall_clearance=0.18
            )
        elif env_name in ["table", "largeobj", "microwave", "allstable"]:
            intervals = self.find_problem_intervals(
                scene_yaml, base_name="table_top", wall_clearance=0.18
            )
        elif env_name == "conveyor":
            intervals = self.find_problem_intervals(
                scene_yaml,
                base_name="conveyor_top",
                wall_clearance=0.04,
            )
        elif env_name == "shelf":
            bases = [
                "shelf_bottom",
                "shelf_middle_bottom",
                "shelf_middle",
                "shelf_middle_top",
                "shelf_top",
            ]
            intervals = self.find_problem_intervals(scene_yaml, bases, 0.12, 0.14)
        else:
            intervals = None
            self.env_details["z_correction"] = [0]

        self.env_details["intervals"] = intervals

    def build_model(self, xml_path, xmls_to_add):
        """
        Build XML and compile the model entirely in memory.
        xml_path: virtual XML path used to resolve relative includes and assets
        xmls_to_add: list of xml fragments containing <asset> and/or <body>
        """

        # A virtual filename preserves relative include and asset resolution.
        xml_path = str(Path(xml_path).resolve())

        asset_blocks = []
        body_blocks = []

        for frag in xmls_to_add:
            # split fragment into asset + body parts
            if "<asset" in frag:
                start = frag.find("<asset")
                end = frag.find("</asset>") + len("</asset>")
                asset_blocks.append(frag[start:end])
                frag = frag[:start] + frag[end:]

            body_blocks.append(frag)

        curr_xml = f"""
        <mujoco model="test_world">
            <include file="{self.base_xml}"/>

            <asset>
        """
        for a in asset_blocks:
            # strip outer <asset> wrapper
            inner = a.replace("<asset>", "").replace("</asset>", "")
            curr_xml += inner + "\n"

        curr_xml += """
            </asset>

            <worldbody>
        """

        for b in body_blocks:
            curr_xml += b + "\n"

        curr_xml += """
            </worldbody>
        </mujoco>
        """

        assets = {**self._generated_assets, xml_path: curr_xml.encode("utf-8")}
        try:
            if hasattr(mujoco, "MjVfs"):
                with mujoco.MjVfs() as vfs:
                    for name, contents in assets.items():
                        vfs[name] = contents
                    model = mujoco.MjModel.from_xml_path(xml_path, vfs=vfs)
            else:
                # Compatibility with MuJoCo versions before MjVfs was exposed.
                model = mujoco.MjModel.from_xml_path(xml_path, assets=assets)
        finally:
            self._generated_assets.clear()

        data = mujoco.MjData(model)
        return model, data
