import yaml
import numpy as np
from scipy.spatial.transform import Rotation as R  # type: ignore
import time
import torch  # type: ignore

import json
import os
import argparse
import math
import random

from functools import wraps


# from planning import omplPlanner

from helpers.helpers import (
    save_paths_npy_numeric,
    to_jsonable,
    load_store,
    get_path_by_index,
)

from helpers.mj_ik import mj_condense
from helpers.import mujoco_utils
from src.condense_paths_mj import MjSuffix, SparseBoxGrid4D

# from helpers.mujoco_utils import MujocoViewer


def to_tuple(x):
    if isinstance(x, list):
        return tuple(to_tuple(e) for e in x)
    if isinstance(x, np.ndarray):
        return to_tuple(x.tolist())
    return x


def main(args):

    object_size = [0.03, 0.03, 0.15]  # x, y, z
    object_type = "cube"

    load_condensed = args.con.lower() == "y"
    robot_clearance = args.inner
    reachable_ws = args.outer
    yaw = float(args.yaw)

    print("Problem parameters:")
    print(f"Robot clearance: {robot_clearance}")
    print(f"Reachable workspace: {reachable_ws}")
    print(f"Object yaw: {-yaw} to {yaw}")

    # Prepare panda.xml with box problem initial config
    panda_dir = "assets/franka_emika_panda"
    panda_src = f"{panda_dir}/panda.xml"
    panda_dst = f"{panda_dir}/panda_relocated.xml"

    base_pos = [0.0, 0.0, 0.0]
    base_quat = [1.0, 0.0, 0.0, 0.0]
    mujoco_utils.write_relocated_panda(
        panda_src, panda_dst, base_pos=base_pos, base_quat_wxyz=base_quat
    )

    # Mujoco initialization
    base_xml, xml_path = mujoco_utils.init_xml(name="free_top_scene")

    object_position = [robot_clearance, 0, 0 + object_size[2] / 2]
    object_quat = [1, 0, 0, 0]

    # Load compressed data structure
    prefix = f"TSRs/free_top/{object_type}_{object_size[0]}_{object_size[1]}_{object_size[2]}/{robot_clearance}_{reachable_ws}_{round(yaw, 2)}"
    root_prefix = prefix + "/roots"

    if load_condensed:
        root_data, offsets, root_keys_arr = load_store(
            root_prefix, mmap_data=True
        )
        with open(f"{prefix}" + "/map.json", "r") as f:
            map_data = json.load(f)

    else:
        return None

    prefix_map = {to_tuple(d["key"]): d["value"] for d in map_data}
    keys_arr = np.array(list(prefix_map), dtype=np.float64)
    indexer = SparseBoxGrid4D(keys_arr)

    cube_name = "cube"
    obj_geom_list = [f"{cube_name}_geom"]
    cube_xml = mujoco_utils.cube_to_xml(
        cube_name, object_position, object_quat, object_size
    )
    xmls_to_add = [cube_xml]
    model, data, _ = mujoco_utils.build_model(base_xml, xml_path, xmls_to_add)

    homePos = [0, -0.785, 0, -2.356, 0, 1.571, 0.785, 0.0399, 0.0399]
    mujoco_utils.set_panda_qpos(model, data, homePos)

    robot_geoms = mujoco_utils.get_robot_collision_geom_ids(model)
    mujoco_viewer = mujoco_utils.MujocoViewer(model, data, robot_geoms)
    mujoco_viewer.open()

    pathSuffix = MjSuffix(
        model, data, robot_geoms, obj_geom_list, mujoco_viewer, 200
    )
    ik_solver = mujoco_utils.PandaMinkIK(
        model, data, robot_geoms, obj_geom_list
    )

    while True:
        mujoco_utils.set_panda_qpos(model, data, homePos)

        # print("Sampling random cube pose...")
        theta = random.uniform(0, 2 * math.pi)
        r = math.sqrt(random.uniform(robot_clearance**2, reachable_ws**2))
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        z = object_size[2] / 2
        sampled_pos = [x, y, z]

        sampled_yaw = random.uniform(-yaw, yaw)
        sampled_quat = [
            math.cos(sampled_yaw / 2),
            0.0,
            0.0,
            math.sin(sampled_yaw / 2),
        ]

        mujoco_utils.move_cube(
            model, data, cube_name, sampled_pos, sampled_quat
        )

        start_query = time.perf_counter()
        bin_idx = indexer.query_point(x, y, z, sampled_yaw)
        sample_key = to_tuple(keys_arr[bin_idx])
        map_result = prefix_map[sample_key]

        # print(map_result)
        map_string = map_result[0]
        bin_goal = map_result[1]
        code_list = list(map(int, map_string.split("-")))
        root_idx = code_list[0]
        root_path = get_path_by_index(root_data, offsets, root_idx)

        root_time = time.perf_counter()

        if len(code_list) > 1:
            prefix_len = code_list[1]
            prefix = root_path[:prefix_len].tolist()

            suffix_start = time.perf_counter()
            suffix = pathSuffix.ik_suffix_single(
                root_path, prefix_len, bin_goal, ik_solver
            )

            if suffix is None:
                # input("Suffix generation failed. Resample?")
                continue

            suffix_end = time.perf_counter()

        else:
            prefix = root_path.tolist()
            suffix = []

            # input("Sampled non-suffix case. Resample?")
            continue

        solved_path = prefix + suffix

        print(f"Solution found")
        print(f"Root time: {root_time-start_query:.6f}")
        print(f"Suffix time: {suffix_end-suffix_start:.6f}")

        mujoco_viewer.play_qpos_traj(solved_path)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inner",
        type=float,
        default=0.3,
        help="Inner radius of cube positions",
    )
    parser.add_argument(
        "--outer",
        type=float,
        default=0.6,
        help="Outer radius of cube positions",
    )
    parser.add_argument(
        "--yaw", type=float, default=1.57, help="+/-Yaw of cube positions"
    )
    parser.add_argument(
        "--con",
        type=str,
        default="y",
        choices=["y", "Y", "n", "N"],
        help="Load condensed data structure? [Y/N]",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
