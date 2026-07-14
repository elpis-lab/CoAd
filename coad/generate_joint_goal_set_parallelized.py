from typing import Any


import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import time
import numpy as np
import pickle
from tqdm import tqdm

from coad.env import MujocoEnv
from coad.robot import MujocoRobot
from geometry.pose import Pose, matrix_to_flat, wrap_to_pi
from coad.utils import set_seed, load_env_and_robot, get_data_folder
from coad.task_space import build_task_nn, key_to_center, split_key, has_contact_face
from coad.mink_ik import get_ik_solver
from coad.mujoco_utils import sample_qpos

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

def get_ik_reference(robot: MujocoRobot, key, attempts, ik_method, **kwargs):
    """Get a reference for IK solving"""
    # If using random method
    # or we have already tried the given method (attempts > 1)
    # use Random
    if ik_method == "random" or attempts > 1:
        seed = sample_qpos(robot.model, robot.joint_ids)

    # If method is not random but we have already tried the given method
    # set None to use the default ik solver pos
    elif attempts == 1:
        seed = None

    # Use given method
    # elif ik_method == "neighbor":
    #     nn = kwargs["nn"]
    #     k = kwargs["k"]
    #     joint_goal_set = kwargs["joint_goal_set"]
    #     keys = kwargs["keys"]

    #     # get the nearest neighbors
    #     # convert key to (x, y, z, rz)
    #     key_arr = np.array(key)
    #     key_center = (key_arr[:, 0] + key_arr[:, 1]) / 2
    #     neighbors = nn.query(
    #         [key_center],
    #         k * 10,  # query more neighbors for better accuracy
    #         return_distance=True,
    #         sort_results=True,
    #     )[1][0]
    #     neighbors = neighbors[:k]

    #     # get the averaged configuration of the neighbors
    #     configs = []
    #     for neighbor in neighbors:
    #         key = keys[neighbor]
    #         if joint_goal_set[key] is not None:
    #             configs.append(joint_goal_set[key])
    #     if len(configs) == 0:
    #         seed = None
    #     else:
    #         seed = np.mean(configs, axis=0)

    elif ik_method == "neighbor":
        k = kwargs["k"]
        joint_goal_set = kwargs["joint_goal_set"]

        if kwargs.get("use_faces", False):
            face, numeric_key = split_key(key)

            nn = kwargs["nn_by_face"][face]
            keys = kwargs["keys_by_face"][face]

            key_arr = np.array(numeric_key)
            key_center = (key_arr[:, 0] + key_arr[:, 1]) / 2

            neighbors = nn.query(
                [key_center],
                k * 10,
                return_distance=True,
                sort_results=True,
            )[1][0]
            neighbors = neighbors[:k]

            configs = []
            for neighbor in neighbors:
                numeric_neighbor_key = keys[neighbor]
                original_neighbor_key = (face, *numeric_neighbor_key)

                if joint_goal_set[original_neighbor_key] is not None:
                    configs.append(joint_goal_set[original_neighbor_key])

            seed = None if len(configs) == 0 else np.mean(configs, axis=0)

        else:
            nn = kwargs["nn"]
            keys = kwargs["keys"]

            key_center = key_to_center(key)

            neighbors = nn.query(
                [key_center],
                min(k * 10, len(keys)),
                return_distance=True,
                sort_results=True,
            )[1][0]

            configs = []
            for neighbor in neighbors:
                neighbor_key = keys[neighbor]
                if joint_goal_set[neighbor_key] is not None:
                    configs.append(joint_goal_set[neighbor_key])
                if len(configs) >= k:
                    break

            seed = None if len(configs) == 0 else np.mean(configs, axis=0)

    elif ik_method == "grr":
        raise NotImplementedError("GRR method not implemented yet")

    else:
        raise ValueError(f"Invalid IK method: {ik_method}")

    return seed


def convert_task_to_joint_goal(
    env: MujocoEnv,
    robot: MujocoRobot,
    task_set,
    ik_method,
    ik_max_attempts=20,
    worker_id=0,
    use_col=False,
    **kwargs,
):
    """Convert task set to joint goal set."""
    # Get an IK solver
    model = robot.model
    data = robot.data
    ik_solver = get_ik_solver(robot, env_collision_geoms=env.env_details['collision_geoms'])

    # Result containers
    ik_success = np.zeros(len(task_set), dtype=bool)
    ik_times = np.zeros(len(task_set), dtype=float)
    joint_goal_set = {key: None for key in task_set.keys()}

    # Prepare for different IK methods
    method_args = {}
    if ik_method == "random":
        pass
    # elif ik_method == "neighbor":
    #     # build a Ball tree for finding the nearest task
    #     nn, bin_poses = build_task_nn(task_set)
    #     method_args = {
    #         "nn": nn,
    #         "k": kwargs.get("n_neighbors", 10),
    #         "joint_goal_set": joint_goal_set,
    #         "keys": list(task_set.keys()),
    #     }
    elif ik_method == "neighbor":
        if has_contact_face(env):
            nn_by_face = {}
            keys_by_face = {}

            for face in ["xy", "yz", "zx"]:
                face_task_set = {
                    key[1:]: task_set[key]
                    for key in task_set.keys()
                    if key[0] == face
                }

                if len(face_task_set) == 0:
                    continue

                nn, _ = build_task_nn(face_task_set)

                nn_by_face[face] = nn
                keys_by_face[face] = list(face_task_set.keys())

            method_args = {
                "use_faces": True,
                "nn_by_face": nn_by_face,
                "keys_by_face": keys_by_face,
                "k": kwargs.get("n_neighbors", 10),
                "joint_goal_set": joint_goal_set,
            }

        else:
            nn, _ = build_task_nn(task_set)
            method_args = {
                "use_faces": False,
                "nn": nn,
                "k": kwargs.get("n_neighbors", 10),
                "joint_goal_set": joint_goal_set,
                "keys": list(task_set.keys()),
            }
    # TODO: GRR method
    elif ik_method == "grr":
        raise NotImplementedError("GRR method not implemented yet")
    else:
        raise ValueError(f"Invalid IK method: {ik_method}")

    # Start sovling IK one by one
    # pbar = tqdm(enumerate[Any](task_set), total=len(task_set))

    pbar = tqdm(
        enumerate(task_set),
        total=len(task_set),
        desc=f"Worker {worker_id}",
        position=worker_id,
        dynamic_ncols=True,
    )

    for i, key in pbar:
        # Moving object (swept volume) to given key pose
        env.move_swept_volume(key)

        # if env.env_details['env_name'] == "allstable":
        #     key = key[1:]
        
        # # Get the end effector targets of the current task
        # # object pose
        # key_arr = np.array(key)
        # key_center = (key_arr[:, 0] + key_arr[:, 1]) / 2

        original_key = key
        env.move_swept_volume(original_key)

        if has_contact_face(env):
            face, numeric_key = split_key(original_key)
        else:
            numeric_key = original_key

        key_center = key_to_center(numeric_key)



        if (env.object_details['type'] == "microwave"):
            obj_pose = env.get_geom_pose("microwave_handle_sv", as_matrix=True)
        else:
            obj_pose = Pose(key_center[:3], (0, 0, key_center[3])).matrix()

        ee_offsets = env.grasp_details['ee_offsets'].copy()
        # print(f"ee_offsets: {ee_offsets}")
        
        # multiple potential ee targets
        targets = [matrix_to_flat(obj_pose @ offset) for offset in ee_offsets]
        # give each target the same number of attempts
        n_target_attempts = int(np.ceil(ik_max_attempts / len(targets)))
        # ik_max_attempts = len(targets) * n_target_attempts
        # print(f"targets: {targets}")
        # Start solving IK
        t0 = time.perf_counter()
        valid_ik = False
        for target in targets:
            for target_attempts in range(n_target_attempts):
                # Solve IK
                reference = get_ik_reference(
                    robot, key, target_attempts, ik_method, **method_args
                )
                reached, solution = ik_solver.solve(
                    target, reference, use_col=use_col
                )

                if reached:
                    robot.set_joint_qpos(solution)
                    if not robot.in_contact():
                        valid_ik = True
                        break
            if valid_ik:
                break

        ik_times[i] = time.perf_counter() - t0
        ik_success[i] = valid_ik
        if valid_ik:
            joint_goal_set[key] = solution

        # Update viewer
        if robot.viewer is not None:
            robot.viewer.sync()
            # input("Proceed?")

        # Update tqdm message periodically
        print_interval = 1000
        if (i + 1) % print_interval == 0:
            m_ik = np.nanmean(ik_times[np.array(ik_success)])
            tqdm.write(
                f"Worker {worker_id}: [{i+1}] "
                f"IK Success: {np.sum(ik_success)/(i+1):.3f} | "
                f"IK Time: {m_ik:.4f}s"
            )

    # Stack results (n, 2)
    results = np.stack([ik_success, ik_times], axis=1)
    return joint_goal_set, results

def solve_task_set_chunk(
    env_name,
    robot_name,
    task_set_chunk,
    ik_method,
    ik_max_attempts,
    worker_id
):
    set_seed(42 + worker_id)
    env, robot = load_env_and_robot(
        env_name,
        robot_name,
        visualize=False
    )

    try:
        joint_goal_set, results = convert_task_to_joint_goal(
            env,
            robot,
            task_set_chunk,
            ik_method,
            ik_max_attempts=ik_max_attempts,
            worker_id=worker_id
        )
    finally:
        robot.close()
    
    return list(task_set_chunk.keys()), joint_goal_set, results


def split_task_set(task_set, num_workers):
    
    if not isinstance(num_workers, int):
        raise TypeError(f"Unsupported data type for num_workers: {type(num_workers)}")
    if num_workers < 1:
        raise ValueError("num_workers must be at least 1")
    
    keys = list(task_set.keys())
    num_workers = min(num_workers, len(keys))

    print(f"Length of task set: {len(keys)}")
    index_chunks = np.array_split(
        np.arange(len(keys)),
        num_workers,
    )

    task_set_chunks = []
    for index_chunk in index_chunks:
        chunk = {
            keys[i]: task_set[keys[i]]
            for i in index_chunk
        }
        task_set_chunks.append(chunk)

    return task_set_chunks

def run_parallel_ik(
    task_set_chunks,
    args,
):
    mp_context = mp.get_context("spawn")

    with ProcessPoolExecutor(
        max_workers=len(task_set_chunks),
        mp_context=mp_context
    ) as executor:
        
        futures = []

        for worker_id, task_set_chunk in enumerate(task_set_chunks):
            future = executor.submit(
                solve_task_set_chunk,
                args.env,
                args.robot,
                task_set_chunk,
                args.ik,
                30,
                worker_id,
            )
            futures.append(future)
        
        worker_outputs = [
            future.result()
            for future in futures
        ]

    return worker_outputs

def consolidate_worker_outputs(task_set, worker_outputs):
    """Merge worker outputs while preserving the original task-set order."""

    joint_goal_set = {
        key: None
        for key in task_set
    }

    results_by_key = {}

    print(f"Original task count: {len(task_set)}")
    print(f"Worker output count: {len(worker_outputs)}")

    for worker_id, worker_output in enumerate(worker_outputs):
        chunk_keys, local_joint_goal_set, local_results = worker_output

        chunk_keys = list(chunk_keys)

        print(
            f"Worker {worker_id}: "
            f"{len(chunk_keys)} keys, "
            f"{len(local_joint_goal_set)} solutions, "
            f"{len(local_results)} result rows"
        )

        if len(chunk_keys) != len(local_results):
            raise RuntimeError(
                f"Worker {worker_id} returned "
                f"{len(chunk_keys)} keys but "
                f"{len(local_results)} result rows"
            )

        joint_goal_set.update(local_joint_goal_set)

        for key, result in zip(chunk_keys, local_results):
            results_by_key[key] = result

    print(f"Consolidated result count: {len(results_by_key)}")

    missing_keys = [
        key
        for key in task_set
        if key not in results_by_key
    ]

    if missing_keys:
        print(f"First missing key: {missing_keys[0]}")
        print(
            "First returned key: "
            f"{next(iter(results_by_key), None)}"
        )

        raise RuntimeError(
            f"Missing worker results for {len(missing_keys)} tasks"
        )

    results = np.asarray(
        [results_by_key[key] for key in task_set],
        dtype=float,
    )

    return joint_goal_set, results

def main(args):
    """Generate a dataset of task paths for a given environment and robot."""
    folder = get_data_folder(args.env, args.robot)
    suffix = f"{args.ik}"

    # Check if graph data is already generated
    data_exists = os.path.exists(f"{folder}/joint_goal_set_{suffix}.pkl")
    if data_exists and not args.overwrite:
        print(
            f"Joint goal set from IK '{args.ik}' already exists at {folder}. "
            + "Use --overwrite to regenerate the joint goal set."
        )
        return

    # Load environment and robot
    # env, robot = load_env_and_robot(args.env, args.robot, visualize=False)

    # Solve problems
    # Load the task set
    try:
        task_set = pickle.load(open(f"{folder}/task_set.pkl", "rb"))
    except FileNotFoundError as e:
        print(e)
        print(
            f"Task set not found! "
            + "Generate the task set with generate_task_set.py."
        )
        # robot.close()
        return

    # Batch task set and create parallel workers
    task_set_chunks = split_task_set(task_set, args.num_workers)
    worker_outputs = run_parallel_ik(task_set_chunks, args)

    joint_goal_set, results = consolidate_worker_outputs(
        task_set,
        worker_outputs
    )


    # Save results
    np.save(f"{folder}/joint_goal_set_results_{suffix}.npy", results)
    # pickle.dump(
    #     joint_goal_set, open(f"{folder}/joint_goal_set_{suffix}.pkl", "wb")
    # )
    with open(f"{folder}/joint_goal_set_{suffix}.pkl", "wb") as file:
        pickle.dump(joint_goal_set, file)
    # robot.close()


def parse_arguments():
    parser = argparse.ArgumentParser()
    # envs
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--env",
        choices=[
            "table",
            "box",
            "cage",
            "shelf",
            "free",
            "real",
            "largeobj",
            "microwave",
            "allstable"],
        default="table",
    )
    parser.add_argument(
        "--robot", choices=["panda", "ur10", "fetch"], default="panda"
    )
    parser.add_argument(
        "--ik", choices=["random", "neighbor", "grr"], default="neighbor"
    )
    parser.add_argument(
        "--num-workers", type=int, default=2, help="number of parallel IK worker processes"
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    set_seed(42)
    args = parse_arguments()

    main(args)
