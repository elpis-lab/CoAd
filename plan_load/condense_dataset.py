# TODO
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import json
import pickle

import numpy as np

from helpers import mujoco_utils
from helpers.helpers import save_paths_npy_numeric, to_jsonable
from helpers.TSR_generation import (
    find_iTSR_set,
    find_yaw_iTSR_set,
    panda_TSR_parameters,
)
from helpers.helpers import load_store

# from helpers.mj_ik import mj_condense

from plan_load.mujoco_envs import build_top_problem, build_cube_problem
from plan_load.generate_dataset import first_itsr_key, build_common_context
from plan_load.condense import condense_dataset
from plan_load.robot import Panda
from plan_load.ik import make_ik_solver


def main(args):
    if args.mode == "top":
        scene = build_top_problem(args)
    else:
        scene = build_cube_problem(args)

    prefix = scene["prefix"]
    data_exists = all(
        os.path.exists(f"{prefix}/{name}")
        for name in ["data.npy", "offsets.npy", "keys.npy"]
    )
    if not data_exists:
        raise FileNotFoundError(
            f"Dataset not found at {prefix}. Run generate_dataset.py first."
        )

    common = build_common_context(args)
    object_details = scene["object_details"]
    object_details["dist"] = common["object_dist"]

    # yaw_iTSR_set, _ = find_yaw_iTSR_set(
    #     object_details, common["problem_details"], common["Tw2_w1"]
    # )
    # iTSR_set, _ = find_iTSR_set(
    #     object_details,
    #     common["problem_details"],
    #     common["yaw_tw2_w1"],
    #     yaw_iTSR_set,
    #     problem=scene["problem"],
    #     robot_pos=scene["robot_pos"],
    # )
    # pickle.dump(iTSR_set, open("task_set.pkl", "wb"))
    iTSR_set = pickle.load(open("task_set.pkl", "rb"))

    first_key = first_itsr_key(iTSR_set)
    first_config = tuple(tuple(v for v in row) for row in first_key)
    sv_xml, _ = mujoco_utils.cube_swept_volume_xml(
        object_details["size"], first_config
    )
    model, data, _ = mujoco_utils.build_model(
        scene["base_xml"],
        scene["xml_path"],
        scene["xmls_to_add"] + [sv_xml],
    )

    # Create robot instance
    robot = Panda(model, visualize=True)
    home_pos = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
    robot.set_joint_qpos(home_pos)
    ik_solver = make_ik_solver(robot, None, scene["obj_geom_list"])

    # Load dataset and condense
    graph_data, offsets, keys = load_store(prefix, mmap_data=True)
    root_paths, key_map = condense_dataset(
        robot, ik_solver, graph_data, offsets, keys
    )

    pickle.dump(root_paths, open(f"{prefix}/root_paths.pkl", "wb"))
    pickle.dump(key_map, open(f"{prefix}/key_map.pkl", "wb"))
    # save_paths_npy_numeric(
    #     f"{prefix}/roots",
    #     root_paths,
    #     path_dtype=np.float32,
    #     key_dtype=np.float64,
    # )
    # json_ready = [
    #     {"key": to_jsonable(k), "value": to_jsonable(v)}
    #     for k, v in prefix_map.items()
    # ]
    # with open(f"{prefix}/map.json", "w") as f:
    #     json.dump(json_ready, f, indent=4)
    # print(f"Saved condensed outputs to: {prefix}/roots and {prefix}/map.json")

    robot.close()


def parse_arguments():
    parser = argparse.ArgumentParser()
    # envs
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--mode", choices=["top", "cube"], default="top")
    # assume object to pick is a cube
    parser.add_argument("--object-size-x", type=float, default=0.03)
    parser.add_argument("--object-size-y", type=float, default=0.03)
    parser.add_argument("--object-size-z", type=float, default=0.15)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main(parse_arguments())
