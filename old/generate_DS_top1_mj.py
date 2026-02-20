import time
import numpy as np
import json
import os
import argparse
from scipy.spatial.transform import Rotation as R

from helpers.import mujoco_utils
from helpers.helpers import save_paths_npy_numeric, to_jsonable
from helpers.TSR_generation import find_yaw_iTSR_set, find_iTSR_set
from helpers.TSR_generation import panda_TSR_parameters
from helpers.mj_ik import cover_iTSR, mj_condense, cover_iTSR_test


def main(args):
    do_condense = args.condense.lower() == "y"
    only_condense = args.oc.lower() == "y"

    # Tunable TSR parameters
    yaw_buffer = 6 * (np.pi / 180)
    alpha = 0.95

    # Problem parameters
    # Test case 1: 0.3, 0.301, 0
    # Benchmarking case 1: 0.3, 0.8, pi
    robot_clearance = 0.3
    reachable_ws = 0.6
    object_dist = [reachable_ws, reachable_ws, 0, 0.5 * np.pi]
    object_dist_check = np.sign(np.array(object_dist))

    # Object parameters (size, position, orientation)
    object_size = [0.03, 0.03, 0.15]  # x, y, z
    object_type = "cube"

    start = time.perf_counter()

    # Parallelization parameters
    parallelize = False
    perf_mode = False
    cpu = 8
    viewer = True

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
    object_quat = [0, 0, 0, 1]  # ?
    _, _, object_yaw = R.from_quat(object_quat).as_euler("xyz", degrees=False)
    object_details = {
        "type": object_type,
        "size": object_size,
        "position": object_position,
        "yaw": object_yaw,
    }
    object_details["dist"] = object_dist

    # Initializing general TSR parameters based on gripper geometry
    # Tew, Bw, half_side, Tw2_w1, yaw_tw2_w1 = panda_TSR_parameters(object_details, yaw_buffer, alpha)
    TSR_params = panda_TSR_parameters(object_details, yaw_buffer, alpha)

    Tew_top, Bw, half_side, Tw2_w1, yaw_tw2_w1_top = TSR_params["top"]
    problem_details_top = {
        "alpha": alpha,
        "Bw": Bw,
        "half_side": half_side,
        "yaw_buffer": yaw_buffer,
        "reachable_ws": reachable_ws,
        "robot_clearance": robot_clearance,
    }
    Tew_front, Bw, half_side, Tw2_w1, yaw_tw2_w1_front = TSR_params["front"]
    # problem_details_front = {
    #     "alpha": alpha,
    #     "Bw": Bw,
    #     "half_side": half_side,
    #     "yaw_buffer": yaw_buffer,
    #     "reachable_ws": reachable_ws,
    #     "robot_clearance": robot_clearance,
    # }

    problem_details = {
        "top": problem_details_top,
        #'front': problem_details_front
    }
    grasping_strategies = list(problem_details)
    print(f"Grasping strategies: {grasping_strategies}")
    Tew = [Tew_top, Tew_front]
    yaw_tw2_w1 = {"top": yaw_tw2_w1_top, "front": yaw_tw2_w1_front}

    yaw_iTSR_set, yaw_to_cover = find_yaw_iTSR_set(
        object_details, problem_details, Tw2_w1
    )
    iTSR_set, iterno = find_iTSR_set(
        object_details, problem_details, yaw_tw2_w1, yaw_iTSR_set
    )
    if len(iTSR_set) == 1:
        print(f"Number of iTSRs: {len(iTSR_set[0])}")
    else:
        for i in range(len(iTSR_set)):
            print(
                f"Number of {grasping_strategies[i]} iTSRs: {len(iTSR_set[i])}"
            )

    # Initializing Mujoco xml with swept volume
    keylist = list(iTSR_set)
    key = keylist[0]
    first_config = tuple(tuple(v for v in row) for row in key)
    sv_xml, sv_pos = mujoco_utils.cube_swept_volume_xml(
        object_size, first_config
    )
    xmls_to_add = [sv_xml]
    model, data, _ = mujoco_utils.build_model(base_xml, xml_path, xmls_to_add)

    if object_type == "cube":
        prefix = f"TSRs/free_top/{object_type}_{object_size[0]}_{object_size[1]}_{object_size[2]}/{robot_clearance}_{reachable_ws}_{round(object_dist[3], 2)}"
    elif object_type == "cylinder":
        prefix = f"TSRs/free_top/{object_type}_{object_size[0]}_{object_size[1]}/{robot_clearance}_{reachable_ws}_{round(object_dist[3], 2)}"
    else:
        raise ValueError(f"Invalid object type: {object_type}")

    if (
        os.path.exists(f"{prefix}/data.npy")
        and os.path.exists(f"{prefix}/offsets.npy")
        and os.path.exists(f"{prefix}/keys.npy")
    ):
        if only_condense:
            generate_ds = "n"
        else:
            generate_ds = input(
                "Data structure for this problem already exists. Overwrite existing data structure? [Y/N]: "
            )
        # generate_ds = input("Data structure for this problem already exists. Overwrite existing data structure? [Y/N]: ")
    else:
        generate_ds = "y"

    if generate_ds.lower() == "y":

        # print(f"Iterations: {iterno}")
        # print("Object uncertainty covered.")

        homePos = [0, -0.785, 0, -2.356, 0, 1.571, 0.785, 0.0399, 0.0399]
        # iTSR_paths, SV = cover_iTSR(robot, scene, homePos, iTSR_set, n_envs, object_details, Tew)

        iTSR_paths, ik_success, plan_success, solve_times, total_plan_times = (
            cover_iTSR_test(
                model, data, homePos, iTSR_set, object_details, Tew, viewer
            )
        )

        os.makedirs(prefix, exist_ok=True)
        np.savez(
            f"{prefix}/planning_stats.npz",
            ik_success=np.array(ik_success, dtype=bool),
            plan_success=np.array(plan_success, dtype=bool),
            solve_times=np.array(solve_times, dtype=float),
            total_plan_times=np.array(total_plan_times, dtype=float),
        )

        # prefix = f"TSRs/cube_limit1/{robot_clearance}_{reachable_ws}_{round(object_dist[3], 2)}"
        save_paths_npy_numeric(
            prefix, iTSR_paths, path_dtype=np.float32, key_dtype=np.float64
        )
        end = time.perf_counter()
        print(f"Elapsed time: {end - start:.6f} seconds")

    else:
        print("Skipping data structure generation.")
        SV = None

    # TODO
    if do_condense:

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
    else:
        print("Not condensing paths.")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--condense",
        type=str,
        default="y",
        choices=["y", "n", "Y", "N"],
        help="Condense paths? [Y/N]",
    )
    parser.add_argument(
        "--oc",
        type=str,
        default="n",
        choices=["y", "n", "Y", "N"],
        help="Only condense paths? [Y/N]",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
