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

from tqdm import tqdm  # type: ignore

from functools import wraps


from planning_mj import omplPlanner

from helpers.helpers import (
    save_paths_npy_numeric,
    to_jsonable,
    load_store,
    get_path_by_index,
)

from helpers.mj_ik import mj_condense
from helpers.import mujoco_utils

from src.condense_paths_mj import MjSuffix, SparseBoxGrid4D


def to_tuple(x):
    if isinstance(x, list):
        return tuple(to_tuple(e) for e in x)
    if isinstance(x, np.ndarray):
        return to_tuple(x.tolist())
    return x


def main(args):
    object_size = [0.03, 0.03, 0.15]  # x, y, z
    object_type = "cube"

    robot_clearance = args.inner
    reachable_ws = args.outer
    n_runs = args.runs
    yaw = float(args.yaw)

    print("Problem parameters:")
    print(f"Robot clearance: {robot_clearance}")
    print(f"Reachable workspace: {reachable_ws}")
    print(f"Object yaw: {-yaw} to {yaw}")

    # Prepare panda.xml with box problem initial config
    panda_dir = "assets/franka_emika_panda"
    panda_src = f"{panda_dir}/panda.xml"
    panda_dst = f"{panda_dir}/panda_relocated.xml"

    problem_config_path = "configs/problems/box_panda.yaml"
    problem_scene_path = "configs/scenes/box/scene_box.yaml"

    with open(problem_scene_path, "r") as f:
        scene_data = yaml.safe_load(f)
    objs = scene_data["world"]["collision_objects"]

    base_zdim = 0
    base_zpos = 0
    box_thickness = 0.04
    box_thickness = 0.18
    base_dim = [0, 0, 0]
    base_pos = [0, 0, 0]

    for obj in objs:
        obj_id = obj.get("id", "")
        if obj_id != "base":
            continue
        # base_zdim = obj["primitives"][0]['dimensions'][2]
        # base_zpos = obj["primitive_poses"][0]['position'][2]
        base_dim = obj["primitives"][0]["dimensions"]
        base_pos = obj["primitive_poses"][0]["position"]

    base_zdim = base_dim[2]
    base_zpos = base_pos[2]

    z_object = base_zpos + (base_zdim / 2) + (object_size[2] / 2)
    # print(z_object)
    hx_int = base_dim[0] / 2 - box_thickness / 2
    hy_int = base_dim[1] / 2 - box_thickness / 2

    box_xmin = base_pos[0] - hx_int + object_size[0] / 2
    box_xmax = base_pos[0] + hx_int - object_size[0] / 2
    box_ymin = base_pos[1] - hy_int + object_size[1] / 2
    box_ymax = base_pos[1] + hy_int - object_size[1] / 2

    box_intervals = [[box_xmin, box_xmax], [box_ymin, box_ymax]]

    problem = {"name": "box", "intervals": box_intervals}

    with open(problem_config_path, "r") as f:
        yaml_data = yaml.safe_load(f)
    base_pos = [
        yaml_data["base_offset"]["position"][0],
        yaml_data["base_offset"]["position"][1],
        yaml_data["base_offset"]["position"][2],
    ]
    base_quat = [
        yaml_data["base_offset"]["orientation"][3],
        yaml_data["base_offset"]["orientation"][0],
        yaml_data["base_offset"]["orientation"][1],
        yaml_data["base_offset"]["orientation"][2],
    ]
    mujoco_utils.write_relocated_panda(
        panda_src, panda_dst, base_pos=base_pos, base_quat_wxyz=base_quat
    )

    # Mujoco initialization
    base_xml, xml_path = mujoco_utils.init_xml(name="box_scene")

    # Prepare box xml
    box_xml = mujoco_utils.yaml_to_xml(
        yaml_path="configs/scenes/box/scene_box2.yaml"
    )

    object_position = [robot_clearance, 0, z_object]
    object_quat = [1, 0, 0, 0]

    # Load compressed data structure
    benchmark_file_path = f"benchmark/box1/{object_type}_{object_size[0]}_{object_size[1]}_{object_size[2]}/{robot_clearance}_{reachable_ws}_{round(yaw, 2)}"
    # prefix = f"TSRs/free_top/{object_type}_{object_size[0]}_{object_size[1]}_{object_size[2]}/{robot_clearance}_{reachable_ws}_{round(yaw, 2)}"
    prefix = f"TSRs/box1/{object_type}_{object_size[0]}_{object_size[1]}_{object_size[2]}/{robot_clearance}_{reachable_ws}_{round(yaw, 2)}"
    root_prefix = prefix + "/roots"

    root_data, offsets, root_keys_arr = load_store(root_prefix, mmap_data=True)
    with open(f"{prefix}" + "/map.json", "r") as f:
        map_data = json.load(f)

    prefix_map = {to_tuple(d["key"]): d["value"] for d in map_data}
    # keys_arr = np.array(list(prefix_map), dtype=np.float64)
    keys_list = list(prefix_map.keys())
    keys_arr = np.array(keys_list, dtype=np.float64)
    print(f"Number of bins: {len(keys_arr)}")
    # return

    indexer = SparseBoxGrid4D(keys_arr)

    cube_name = "cube"
    obj_geom_list = [f"{cube_name}_geom"]
    cube_xml = mujoco_utils.cube_to_xml(
        cube_name, object_position, object_quat, object_size
    )
    xmls_to_add = [box_xml, cube_xml]
    model, data, _ = mujoco_utils.build_model(base_xml, xml_path, xmls_to_add)

    homePos = [0, -0.785, 0, -2.356, 0, 1.571, 0.785, 0.0399, 0.0399]
    mujoco_utils.set_panda_qpos(model, data, homePos)

    robot_geoms = mujoco_utils.get_robot_collision_geom_ids(model)
    mujoco_viewer = None
    pathSuffix = MjSuffix(
        model, data, robot_geoms, obj_geom_list, mujoco_viewer, 200
    )

    planner = omplPlanner(model, data, robot_geoms, log=False)
    ik_solver = mujoco_utils.PandaMinkIK(
        model, data, robot_geoms, obj_geom_list
    )

    planning_success = []
    planning_total_times = []
    planning_solve_times = []
    ds_success = []
    ds_times = []
    ds_suffix_times = []

    root_samples = 0
    current_run = 0

    pbar = tqdm(total=n_runs, desc=f"Running {n_runs} samples")
    while current_run < n_runs:
        mujoco_utils.set_panda_qpos(model, data, homePos)

        # Sample a cube position and move cube
        theta = random.uniform(0, 2 * math.pi)
        r = math.sqrt(random.uniform(robot_clearance**2, reachable_ws**2))
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        z = z_object
        sampled_pos = [x + base_pos[0], y + base_pos[1], z]
        sampled_yaw = random.uniform(-yaw, yaw)
        sampled_quat = [
            math.cos(sampled_yaw / 2),
            0.0,
            0.0,
            math.sin(sampled_yaw / 2),
        ]

        if (
            box_xmin <= sampled_pos[0] <= box_xmax
            and box_ymin <= sampled_pos[1] <= box_ymax
        ):
            in_box = True
        else:
            in_box = False

        if in_box is False:
            # print("Sampled point out of box. Resampling...")
            continue

        mujoco_utils.move_cube(
            model, data, cube_name, sampled_pos, sampled_quat
        )

        # Datastructure query
        start_query = time.perf_counter()
        bin_idx = indexer.query_point(
            sampled_pos[0], sampled_pos[1], sampled_pos[2], sampled_yaw
        )
        # print(bin_idx)
        if bin_idx is None:
            continue
        # return
        # sample_key = to_tuple(keys_arr[bin_idx])
        sample_key = keys_list[bin_idx]
        # print(sample_key)
        # return None
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
                # print("Suffix generation failed.")
                curr_ds_success = False
            else:
                curr_ds_success = True

            suffix_end = time.perf_counter()
            curr_ds_suffix_time = round(suffix_end - suffix_start, 5)
            curr_ds_time = round(suffix_end - start_query, 5)

        else:
            if root_path.size == 0:
                # Empty path = failure
                curr_ds_success = False

            prefix = root_path.tolist()
            suffix = []
            root_samples += 1
            # input("Sampled non-suffix case. Resample?")
            continue

        # RRTConnect call
        mujoco_utils.set_panda_qpos(model, data, homePos)

        path = planner.omplPlan(
            qpos_goal=bin_goal,
            qpos_start=homePos,
            timeout=3.0,
            planner="RRTConnect",
        )

        if path:
            curr_planning_success = True
        else:
            curr_planning_success = False

        curr_planning_solve_time = planner.plan_time
        curr_planning_total_time = planner.total_time

        planning_success.append(curr_planning_success)
        planning_total_times.append(curr_planning_total_time)
        planning_solve_times.append(curr_planning_solve_time)
        ds_success.append(curr_ds_success)
        ds_times.append(curr_ds_time)
        ds_suffix_times.append(curr_ds_suffix_time)

        current_run += 1
        pbar.update(1)

    pbar.close()

    print(
        f"Benchmarking Complete - {n_runs} runs. Skipped {root_samples} root path samples."
    )
    print(
        f"RRTConnect success rate: {100.0 * sum(planning_success) / len(planning_success)}"
    )
    print(f"RRTConnect median solve time: {np.median(planning_solve_times)}")
    print(f"RRTConnect median total time: {np.median(planning_total_times)}")
    print(f"Query success rate: {100 * sum(ds_success) / len(ds_success)}")
    print(f"Query median suffix time: {np.median(ds_suffix_times)}")
    print(f"Query median total time: {np.median(ds_times)}")

    os.makedirs(benchmark_file_path, exist_ok=True)
    np.save(
        os.path.join(benchmark_file_path, "ds_times.npy"), np.asarray(ds_times)
    )
    np.save(
        os.path.join(benchmark_file_path, "ds_success.npy"),
        np.asarray(ds_success),
    )
    np.save(
        os.path.join(benchmark_file_path, "ds_suffix_times.npy"),
        np.asarray(ds_suffix_times),
    )

    np.save(
        os.path.join(benchmark_file_path, "planning_total_times.npy"),
        np.asarray(planning_total_times),
    )
    np.save(
        os.path.join(benchmark_file_path, "planning_solve_times.npy"),
        np.asarray(planning_solve_times),
    )
    np.save(
        os.path.join(benchmark_file_path, "planning_success.npy"),
        np.asarray(planning_success),
    )


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inner",
        type=float,
        default=0.3,
        help="Inner radius of cube position",
    )
    parser.add_argument(
        "--outer",
        type=float,
        default=0.75,
        help="Outer radius of cube position",
    )
    parser.add_argument(
        "--yaw", type=float, default=1.57, help="+/- Yaw of cube positions"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=100,
        help="Number of runs to benchmark with",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
