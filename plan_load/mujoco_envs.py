# TODO
import os
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R
from helpers import mujoco_utils

# PANDA Relocate
# pos="0.2 0.0 0.8" quat="0.70710678 0.0 0.0 -0.70710678"


def build_top_problem(args):
    """Build top problem environment configuration."""
    # small extra angular slack around yaw limits
    # when generating bounds
    yaw_buffer = 0.1047  # 6.0 * (np.pi / 180.0)
    # a coverage hyperparameter for graph generation
    alpha = 0.95
    # Safety margin for the robot
    robot_clearance = 0.3
    reachable_ws = 0.6 if args.mode == "top" else 0.9
    yaw_dist_rad = 0.5 * np.pi

    panda_dir = "assets/franka_emika_panda"
    panda_src = f"{panda_dir}/panda.xml"
    panda_dst = f"{panda_dir}/panda_relocated.xml"

    mujoco_utils.write_relocated_panda(
        panda_src,
        panda_dst,
        base_pos=[0.0, 0.0, 0.0],
        base_quat_wxyz=[1.0, 0.0, 0.0, 0.0],
    )
    base_xml, xml_path = mujoco_utils.init_xml(name="free_top_scene")

    object_size = [args.object_size_x, args.object_size_y, args.object_size_z]
    object_position = [0.0, 0.0, object_size[2] / 2.0]
    object_quat = [0, 0, 0, 1]
    _, _, object_yaw = R.from_quat(object_quat).as_euler("xyz", degrees=False)

    object_details = {
        "type": "cube",
        "size": object_size,
        "position": object_position,
        "yaw": object_yaw,
    }

    prefix = f"dataset/{args.mode}"

    result = {
        "base_xml": base_xml,
        "xml_path": xml_path,
        "xmls_to_add": [],
        "object_details": object_details,
        "problem": None,
        "robot_pos": [0.0, 0.0, 0.0],
        "prefix": prefix,
        "obj_geom_list": [
            "sv_box1",
            "sv_box2",
            "sv_cyl1",
            "sv_cyl2",
            "sv_cyl3",
            "sv_cyl4",
        ],
    }

    return result


def build_cube_problem(args):
    """Build cube/box problem environment configuration."""
    panda_dir = "assets/franka_emika_panda"
    panda_src = f"{panda_dir}/panda.xml"
    panda_dst = f"{panda_dir}/panda_relocated.xml"

    problem_config_path = "configs/problems/box_panda.yaml"
    problem_scene_path = "configs/scenes/box/scene_box.yaml"

    with open(problem_scene_path, "r") as f:
        scene_data = yaml.safe_load(f)
    objs = scene_data["world"]["collision_objects"]

    base_dim = [0.0, 0.0, 0.0]
    base_pos_scene = [0.0, 0.0, 0.0]
    box_thickness = 0.18

    for obj in objs:
        if obj.get("id", "") != "base":
            continue
        base_dim = obj["primitives"][0]["dimensions"]
        base_pos_scene = obj["primitive_poses"][0]["position"]

    object_size = [args.object_size_x, args.object_size_y, args.object_size_z]
    z_object = base_pos_scene[2] + (base_dim[2] / 2.0) + (object_size[2] / 2.0)

    hx_int = base_dim[0] / 2.0 - box_thickness / 2.0
    hy_int = base_dim[1] / 2.0 - box_thickness / 2.0

    box_xmin = base_pos_scene[0] - hx_int + object_size[0] / 2.0
    box_xmax = base_pos_scene[0] + hx_int - object_size[0] / 2.0
    box_ymin = base_pos_scene[1] - hy_int + object_size[1] / 2.0
    box_ymax = base_pos_scene[1] + hy_int - object_size[1] / 2.0

    problem = {
        "name": "box",
        "intervals": [[box_xmin, box_xmax], [box_ymin, box_ymax]],
    }

    with open(problem_config_path, "r") as f:
        yaml_data = yaml.safe_load(f)

    robot_pos = [
        yaml_data["base_offset"]["position"][0],
        yaml_data["base_offset"]["position"][1],
        yaml_data["base_offset"]["position"][2],
    ]
    robot_quat = [
        yaml_data["base_offset"]["orientation"][3],
        yaml_data["base_offset"]["orientation"][0],
        yaml_data["base_offset"]["orientation"][1],
        yaml_data["base_offset"]["orientation"][2],
    ]

    mujoco_utils.write_relocated_panda(
        panda_src,
        panda_dst,
        base_pos=robot_pos,
        base_quat_wxyz=robot_quat,
    )

    base_xml, xml_path = mujoco_utils.init_xml(name="box_scene")
    box_xml = mujoco_utils.yaml_to_xml(
        yaml_path="configs/scenes/box/scene_box2.yaml"
    )

    object_quat = [0, 0, 0, 1]
    _, _, object_yaw = R.from_quat(object_quat).as_euler("xyz", degrees=False)
    object_details = {
        "type": "cube",
        "size": object_size,
        "position": [0.0, 0.0, z_object],
        "yaw": object_yaw,
    }

    prefix = (
        f"TSRs/box1/cube_{object_size[0]}_{object_size[1]}_{object_size[2]}/"
        f"{args.robot_clearance}_{args.reachable_ws}_{round(args.yaw_dist_rad, 2)}"
    )

    result = {
        "base_xml": base_xml,
        "xml_path": xml_path,
        "xmls_to_add": [box_xml],
        "object_details": object_details,
        "problem": problem,
        "robot_pos": robot_pos,
        "prefix": prefix,
        "obj_geom_list": [
            "sv_box1",
            "sv_box2",
            "sv_cyl1",
            "sv_cyl2",
            "sv_cyl3",
            "sv_cyl4",
            "base",
            "side_left",
            "side_right",
            "side_front",
            "side_cap",
            "side_back",
        ],
    }

    return result
