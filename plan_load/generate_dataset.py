# TODO
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import time
import numpy as np
import pickle

from plan_load.task_space import solve_task_set
from plan_load.mujoco_envs import build_top_problem, build_cube_problem
from plan_load.robot import Panda, UR10

from helpers.mujoco_utils import cube_swept_volume_xml, build_model
from helpers.helpers import save_paths_npy_numeric
from helpers.TSR_generation import (
    find_iTSR_set,
    find_yaw_iTSR_set,
    panda_TSR_parameters,
)


def build_common_context(args):
    # small extra angular slack around yaw limits
    # when generating bounds
    yaw_buffer = 0.1047  # 6.0 * (np.pi / 180.0)
    # a coverage hyperparameter for graph generation
    alpha = 0.95
    # Safety margin for the robot
    robot_clearance = 0.3

    reachable_ws = 0.6 if args.mode == "top" else 0.9
    object_dist = [
        reachable_ws,
        reachable_ws,
        0.0,
        0.5 * np.pi,  # args.yaw_dist_rad,
    ]

    object_details = {
        "type": "cube",
        "size": [args.object_size_x, args.object_size_y, args.object_size_z],
    }

    tsr_object_details = dict(object_details)
    tsr_object_details["position"] = [0, 0, 0]
    tsr_object_details["yaw"] = 0.0
    tsr_object_details["dist"] = object_dist
    TSR_params = panda_TSR_parameters(tsr_object_details, yaw_buffer, alpha)
    Tew_top, Bw_top, half_side_top, Tw2_w1, yaw_tw2_w1_top = TSR_params["top"]
    Tew_front, _, _, _, yaw_tw2_w1_front = TSR_params["front"]

    problem_details = {
        "top": {
            "alpha": alpha,
            "Bw": Bw_top,
            "half_side": half_side_top,
            "yaw_buffer": yaw_buffer,
            "reachable_ws": reachable_ws,
            "robot_clearance": robot_clearance,
        }
    }

    return {
        "object_dist": object_dist,
        "problem_details": problem_details,
        "Tew": [Tew_top, Tew_front],
        "Tw2_w1": Tw2_w1,
        "yaw_tw2_w1": {
            "top": yaw_tw2_w1_top,
            "front": yaw_tw2_w1_front,
        },
    }


def first_itsr_key(iTSR_set_all):
    iTSR_dict = (
        iTSR_set_all[0] if isinstance(iTSR_set_all, list) else iTSR_set_all
    )
    if not iTSR_dict:
        raise RuntimeError("No iTSRs were generated; cannot continue.")
    return next(iter(iTSR_dict))


def main(args):
    # small extra angular slack around yaw limits
    # when generating bounds
    yaw_buffer = 0.1047  # 6.0 * (np.pi / 180.0)
    # a coverage hyperparameter for graph generation
    alpha = 0.95
    # Safety margin for the robot
    robot_clearance = 0.3

    reachable_ws = 0.6 if args.mode == "top" else 0.9
    yaw_dist_rad = 0.5 * np.pi

    obj_size = [args.object_size_x, args.object_size_y, args.object_size_z]
    # prefix = (
    #     f"TSRs/free_top/cube_{obj_size[0]}_{obj_size[1]}_{obj_size[2]}/"
    #     f"{robot_clearance}_{reachable_ws}_{round(yaw_dist_rad, 2)}"
    # )
    prefix = f"dataset/{args.mode}"

    # Check if data is already generated
    data_exists = os.path.exists(f"{prefix}/data.npy")
    if data_exists and not args.overwrite:
        print(
            f"Data structure already exists at {prefix}. "
            + "Use --overwrite to regenerate."
        )
        return

    # Build scene
    if args.mode == "top":
        scene = build_top_problem(args)
    elif args.mode == "cube":
        scene = build_cube_problem(args)
    else:
        raise ValueError(f"Invalid mode: {args.mode}")
    common = build_common_context(args)

    object_details = scene["object_details"]
    object_details["dist"] = common["object_dist"]

    # Build problems
    yaw_iTSR_set, _ = find_yaw_iTSR_set(
        object_details, common["problem_details"], common["Tw2_w1"]
    )
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
    sv_xml, _ = cube_swept_volume_xml(object_details["size"], first_config)

    # Build model and solve problems
    model, data, _ = build_model(
        scene["base_xml"],
        scene["xml_path"],
        scene["xmls_to_add"] + [sv_xml],
    )
    # Create robot instance
    robot = Panda(model, visualize=True)
    home_pos = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
    robot.set_joint_qpos(home_pos)

    # Solve task set
    task_paths = solve_task_set(
        robot,
        iTSR_set[0],
        common["Tew"][0],
        env_collision_geoms=scene["obj_geom_list"],
    )

    # Save results
    os.makedirs(prefix, exist_ok=True)
    # np.savez(
    #     f"{prefix}/planning_stats.npz",
    #     ik_success=np.array(ik_success, dtype=bool),
    #     plan_success=np.array(plan_success, dtype=bool),
    #     solve_times=np.array(solve_times, dtype=float),
    #     total_plan_times=np.array(total_plan_times, dtype=float),
    # )
    save_paths_npy_numeric(
        prefix, task_paths, path_dtype=float, key_dtype=float
    )
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
