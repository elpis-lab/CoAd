import yaml
import numpy as np
from scipy.spatial.transform import Rotation as R  # type: ignore
import time

# import torch # type: ignore

import json
import os
import argparse
from functools import wraps

# from planning import omplPlanner

from helpers.helpers import save_paths_npy_numeric, to_jsonable, load_store
from helpers.TSR_generation import (
    panda_TSR_parameters,
    find_yaw_iTSR_set,
    find_iTSR_set,
)

# from helpers.ik import cover_iTSR, condense_paths, condense_paths2, condense_paths3
# from src.swept_volume import SweptVolumeCube

from helpers.mj_ik import mj_condense

from helpers.import mujoco_utils


def main():

    # Tunable TSR parameters
    yaw_buffer = 6 * (np.pi / 180)
    alpha = 0.95

    # Problem parameters

    # Test case 1: 0.3, 0.301, 0
    # Benchmarking case 1: 0.3, 0.8, pi

    robot_clearance = 0.3
    reachable_ws = 0.4
    object_dist = [reachable_ws, reachable_ws, 0, 0.5 * np.pi]
    object_dist_check = np.sign(np.array(object_dist))

    # Object parameters (size, position, orientation)
    object_size = [0.04, 0.04, 0.2]  # x, y, z
    object_size = [0.03, 0.03, 0.15]
    object_type = "cube"

    start = time.perf_counter()

    viewer = False

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

    object_position = [0, 0, 0 + object_size[2] / 2]
    object_quat = [0, 0, 0, 1]
    _, _, object_yaw = R.from_quat(object_quat).as_euler("xyz", degrees=False)

    object_details = {
        "type": object_type,
        "size": object_size,
        "position": object_position,
        "yaw": object_yaw,
    }

    object_details["dist"] = object_dist

    # Initializing general TSR parameters based on gripper geometry
    Tew, Bw, half_side, Tw2_w1, yaw_tw2_w1 = panda_TSR_parameters(
        object_details, yaw_buffer, alpha
    )

    # print("tw2_w1: ", tw2_w1)
    # print("yaw_tw2_w1: ", yaw_tw2_w1)

    problem_details = {
        "alpha": alpha,
        "Bw": Bw,
        "half_side": half_side,
        "yaw_buffer": yaw_buffer,
        "reachable_ws": reachable_ws,
        "robot_clearance": robot_clearance,
    }

    # prefix = f"TSRs/cube_limit1/{robot_clearance}_{reachable_ws}_{round(object_dist[3], 2)}"
    # prefix = f"TSRs/cube_limit1/{object_type}_{object_size[0]}_{object_size[1]}_{object_size[2]}/{robot_clearance}_{reachable_ws}_{round(object_dist[3], 2)}"

    if object_type == "cube":
        prefix = f"TSRs/free_top/{object_type}_{object_size[0]}_{object_size[1]}_{object_size[2]}/{robot_clearance}_{reachable_ws}_{round(object_dist[3], 2)}"
    else:  # cylinder
        prefix = f"TSRs/free_top/{object_type}_{object_size[0]}_{object_size[1]}/{robot_clearance}_{reachable_ws}_{round(object_dist[3], 2)}"

    if (
        os.path.exists(f"{prefix}/data.npy")
        and os.path.exists(f"{prefix}/offsets.npy")
        and os.path.exists(f"{prefix}/keys.npy")
    ):

        print("Data structure found. Condensing...")
    else:
        print("Data structure NOT found. Aborting...")
        return None

    yaw_iTSR_set, yaw_to_cover = find_yaw_iTSR_set(
        object_details, problem_details, Tw2_w1
    )
    iTSR_set, iterno = find_iTSR_set(
        object_details, problem_details, yaw_tw2_w1, yaw_iTSR_set
    )
    keylist = list(iTSR_set)

    # Create swept volume xml
    key = keylist[0]
    first_config = tuple(tuple(v for v in row) for row in key)
    sv_xml, sv_pos = mujoco_utils.cube_swept_volume_xml(
        object_size, first_config
    )
    xmls_to_add = [sv_xml]
    model, data, _ = mujoco_utils.build_model(base_xml, xml_path, xmls_to_add)

    # data, offsets, keys_arr = load_store(prefix, mmap_data=True)
    # root_paths, prefix_map = condense_paths3(SV, robot, scene, prefix, n_envs, object_details)

    obj_geom_list = [
        "sv_box1",
        "sv_box2",
        "sv_cyl1",
        "sv_cyl2",
        "sv_cyl3",
        "sv_cyl4",
    ]

    root_paths, prefix_map = mj_condense(
        model, data, prefix, object_details, obj_geom_list, viewer
    )

    save_paths_npy_numeric(
        f"{prefix}/roots",
        root_paths,
        path_dtype=np.float32,
        key_dtype=np.float64,
    )

    json_ready = [
        {"key": to_jsonable(k), "value": to_jsonable(v)}
        for k, v in prefix_map.items()
    ]

    with open(f"{prefix}/map.json", "w") as f:
        json.dump(json_ready, f, indent=4)


if __name__ == "__main__":
    main()
