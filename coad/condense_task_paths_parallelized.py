import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import multiprocessing as mp
import pickle
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

from coad.utils import set_seed, load_env_and_robot, get_data_folder
from coad.task_space import build_task_nn, split_key, has_contact_face
from coad.mink_ik import get_ik_solver

from coad.adaptation import LinearAdapter, GRRAdapter
from coad.adaptation import DMPAdapter, TrajOptAdapter


_WORKER_ENV = None
_WORKER_ROBOT = None
_WORKER_ADAPTER = None


def _initialize_condensation_worker(
    env_name,
    robot_name,
    adaptation_name,
    tqdm_lock,
):
    """Create process-local MuJoCo, IK, and adaptation objects."""
    global _WORKER_ENV
    global _WORKER_ROBOT
    global _WORKER_ADAPTER

    tqdm.set_lock(tqdm_lock)

    _WORKER_ENV, _WORKER_ROBOT = load_env_and_robot(
        env_name,
        robot_name,
        visualize=False,
    )

    ik_solver = get_ik_solver(
        _WORKER_ROBOT,
        env_collision_geoms=_WORKER_ENV.env_details["collision_geoms"],
    )

    if adaptation_name == "linear":
        _WORKER_ADAPTER = LinearAdapter(_WORKER_ROBOT, ik_solver)
    elif adaptation_name == "grr":
        _WORKER_ADAPTER = GRRAdapter(_WORKER_ROBOT, ik_solver)
    elif adaptation_name == "dmp":
        _WORKER_ADAPTER = DMPAdapter(_WORKER_ROBOT, ik_solver)
    elif adaptation_name == "opt":
        _WORKER_ADAPTER = TrajOptAdapter(_WORKER_ROBOT, ik_solver)
    else:
        raise ValueError(
            f"Invalid adaptation method: {adaptation_name}"
        )


def _build_chunk_neighbor_index(task_paths):
    """
    Build nearest-neighbor structures for one worker's chunk.

    Contact-face environments use one index per face so that keys from
    different support faces are never compared directly.
    """
    if has_contact_face(_WORKER_ENV):
        nn_by_face = {}
        keys_by_face = {}

        for face in ["xy", "yz", "zx"]:
            face_task_paths = {
                key[1:]: path
                for key, path in task_paths.items()
                if key[0] == face
            }

            if not face_task_paths:
                continue

            face_nn, _ = build_task_nn(face_task_paths)
            nn_by_face[face] = face_nn
            keys_by_face[face] = list(face_task_paths.keys())

        return {
            "has_contact_face": True,
            "nn_by_face": nn_by_face,
            "keys_by_face": keys_by_face,
        }

    nn, _ = build_task_nn(task_paths)

    return {
        "has_contact_face": False,
        "nn": nn,
        "keys": list(task_paths.keys()),
    }


def _get_neighbor_keys(
    center_key,
    n_neighbors,
    neighbor_index,
):
    """Return nearby keys from the current worker's chunk."""
    if neighbor_index["has_contact_face"]:
        face, numeric_key = split_key(center_key)

        face_nn = neighbor_index["nn_by_face"][face]
        face_keys = neighbor_index["keys_by_face"][face]

        key_arr = np.asarray(numeric_key)
        key_center = (key_arr[:, 0] + key_arr[:, 1]) / 2

        k = min(n_neighbors + 1, len(face_keys))

        neighbor_indices = face_nn.query(
            [key_center],
            k=k,
            return_distance=False,
            sort_results=True,
        )[0]

        return [
            (face,) + tuple(face_keys[int(index)])
            for index in neighbor_indices
        ]

    keys = neighbor_index["keys"]

    key_arr = np.asarray(center_key)
    key_center = (key_arr[:, 0] + key_arr[:, 1]) / 2

    k = min(n_neighbors + 1, len(keys))

    neighbor_indices = neighbor_index["nn"].query(
        [key_center],
        k=k,
        return_distance=False,
        sort_results=True,
    )[0]

    return [keys[int(index)] for index in neighbor_indices]


def _condense_chunk(
    chunk_id,
    task_paths,
    n_neighbors,
    seed,
):
    """
    Condense one independent chunk.

    Local root IDs begin at zero. The parent process remaps them when
    combining all worker outputs.
    """
    global _WORKER_ENV
    global _WORKER_ADAPTER

    rng = np.random.default_rng(seed + chunk_id)

    valid_task_paths = {
        key: path
        for key, path in task_paths.items()
        if path is not None and len(path) > 1
    }

    remaining = set(valid_task_paths.keys())

    local_root_paths = {}
    local_key_to_root = {
        key: (None, None)
        for key in task_paths.keys()
    }
    local_build_center_time = {
        key: np.nan
        for key in task_paths.keys()
    }
    local_compress_time = {
        key: np.nan
        for key in task_paths.keys()
    }

    if not valid_task_paths:
        return {
            "chunk_id": chunk_id,
            "root_paths": local_root_paths,
            "key_to_root": local_key_to_root,
            "build_center_time": local_build_center_time,
            "compress_time": local_compress_time,
            "num_tasks": len(task_paths),
            "num_valid_tasks": 0,
            "error": None,
        }

    try:
        neighbor_index = _build_chunk_neighbor_index(
            valid_task_paths
        )

        pbar = tqdm(
            total=len(remaining),
            position=chunk_id,
            desc=f"Worker {chunk_id}",
            unit="task",
            leave=True,
            dynamic_ncols=True,
        )

        while remaining:
            selectable_keys = tuple(remaining)
            center_key = selectable_keys[
                int(rng.integers(0, len(selectable_keys)))
            ]

            center_path = valid_task_paths[center_key].copy()
            _WORKER_ENV.move_swept_volume(center_key)

            local_root_id = len(local_root_paths)

            t0 = time.perf_counter()
            adapted_center, q_end = _WORKER_ADAPTER.build_center(
                center_path
            )
            build_time = time.perf_counter() - t0

            local_root_paths[local_root_id] = adapted_center
            local_key_to_root[center_key] = (
                local_root_id,
                q_end,
            )
            local_build_center_time[center_key] = build_time

            remaining.remove(center_key)
            pbar.update(1)

            neighbor_keys = _get_neighbor_keys(
                center_key=center_key,
                n_neighbors=n_neighbors,
                neighbor_index=neighbor_index,
            )

            newly_covered = 0

            for neighbor_key in neighbor_keys:
                if (
                    neighbor_key == center_key
                    or neighbor_key not in remaining
                ):
                    continue

                neighbor_path = valid_task_paths[
                    neighbor_key
                ].copy()

                _WORKER_ENV.move_swept_volume(neighbor_key)

                t0 = time.perf_counter()
                valid, q_neighbor_end = (
                    _WORKER_ADAPTER.compress(
                        adapted_center,
                        neighbor_path,
                    )
                )
                compression_time = time.perf_counter() - t0

                if not valid:
                    continue

                local_key_to_root[neighbor_key] = (
                    local_root_id,
                    q_neighbor_end,
                )
                local_compress_time[
                    neighbor_key
                ] = compression_time

                remaining.remove(neighbor_key)
                newly_covered += 1
                pbar.update(1)

            pbar.set_postfix(
                roots=len(local_root_paths),
                covered=(
                    len(valid_task_paths) - len(remaining)
                ),
                refresh=True,
            )

        pbar.close()

        return {
            "chunk_id": chunk_id,
            "root_paths": local_root_paths,
            "key_to_root": local_key_to_root,
            "build_center_time": local_build_center_time,
            "compress_time": local_compress_time,
            "num_tasks": len(task_paths),
            "num_valid_tasks": len(valid_task_paths),
            "error": None,
        }

    except Exception:
        return {
            "chunk_id": chunk_id,
            "root_paths": local_root_paths,
            "key_to_root": local_key_to_root,
            "build_center_time": local_build_center_time,
            "compress_time": local_compress_time,
            "num_tasks": len(task_paths),
            "num_valid_tasks": len(valid_task_paths),
            "error": traceback.format_exc(),
        }


def _split_task_paths(task_paths, num_chunks):
    """Split the path dictionary into contiguous, balanced chunks."""
    items = list(task_paths.items())

    if not items:
        return []

    num_chunks = max(1, min(num_chunks, len(items)))
    index_chunks = np.array_split(
        np.arange(len(items)),
        num_chunks,
    )

    return [
        {
            items[int(index)][0]: items[int(index)][1]
            for index in indices
        }
        for indices in index_chunks
        if len(indices) > 0
    ]


def condense_dataset_parallel(
    env_name,
    robot_name,
    task_paths,
    adaptation,
    n_neighbors=1000,
    num_workers=2,
    seed=42,
):
    """
    Condense an existing full path library in independent worker chunks.

    Roots are shared only within a chunk. Therefore, this produces a valid
    condensed representation but may use more roots than global serial
    condensation.
    """
    all_keys = list(task_paths.keys())

    global_root_paths = {}
    global_key_to_root = {
        key: (None, None)
        for key in all_keys
    }
    global_build_center_time = {
        key: np.nan
        for key in all_keys
    }
    global_compress_time = {
        key: np.nan
        for key in all_keys
    }

    chunks = _split_task_paths(task_paths, num_workers)

    num_valid_tasks = sum(
        path is not None and len(path) > 1
        for path in task_paths.values()
    )

    print(f"Number of tasks: {len(task_paths)}")
    print(
        "Number of successfully solved tasks: "
        f"{num_valid_tasks}"
    )
    print(f"Number of workers: {len(chunks)}")

    context = mp.get_context("spawn")
    tqdm_lock = context.RLock()

    with ProcessPoolExecutor(
        max_workers=len(chunks),
        mp_context=context,
        initializer=_initialize_condensation_worker,
        initargs=(
            env_name,
            robot_name,
            adaptation,
            tqdm_lock,
        ),
    ) as executor:
        future_to_chunk_id = {
            executor.submit(
                _condense_chunk,
                chunk_id,
                chunk,
                n_neighbors,
                seed,
            ): chunk_id
            for chunk_id, chunk in enumerate(chunks)
        }

        for future in as_completed(future_to_chunk_id):
            chunk_id = future_to_chunk_id[future]

            try:
                worker_result = future.result()
            except Exception:
                tqdm.write(
                    f"Worker {chunk_id} failed:\n"
                    f"{traceback.format_exc()}"
                )
                continue

            if worker_result["error"] is not None:
                tqdm.write(
                    f"Worker {chunk_id} failed:\n"
                    f"{worker_result['error']}"
                )
                continue

            root_offset = len(global_root_paths)

            for local_root_id, adapted_path in (
                worker_result["root_paths"].items()
            ):
                global_root_id = (
                    root_offset + local_root_id
                )
                global_root_paths[
                    global_root_id
                ] = adapted_path

            for key, assignment in (
                worker_result["key_to_root"].items()
            ):
                local_root_id, goal_q = assignment

                if local_root_id is None:
                    global_key_to_root[key] = (
                        None,
                        None,
                    )
                else:
                    global_key_to_root[key] = (
                        root_offset + local_root_id,
                        goal_q,
                    )

            global_build_center_time.update(
                worker_result["build_center_time"]
            )
            global_compress_time.update(
                worker_result["compress_time"]
            )

            tqdm.write(
                f"Worker {chunk_id} complete: "
                f"{worker_result['num_valid_tasks']} valid tasks, "
                f"{len(worker_result['root_paths'])} roots"
            )

    results = np.stack(
        [
            [
                global_build_center_time[key]
                for key in all_keys
            ],
            [
                global_compress_time[key]
                for key in all_keys
            ],
        ],
        axis=1,
    )

    print("Parallel condensation complete.")
    print(
        f"Number of root paths: "
        f"{len(global_root_paths)}"
    )

    return (
        global_root_paths,
        global_key_to_root,
        results,
    )


def main(args):
    """Condense an already-generated full task-path library."""
    folder = get_data_folder(args.env, args.robot)
    suffix = (
        f"{args.ik}_{args.planner}_"
        f"{args.adaptation}_{args.n_neighbors}"
    )

    root_path_file = (
        f"{folder}/root_paths_{suffix}.pkl"
    )

    if (
        os.path.exists(root_path_file)
        and not args.overwrite
    ):
        print(
            f"Compressed root paths already exist at "
            f"{folder} with IK '{args.ik}', "
            f"planner '{args.planner}', and adaptation "
            f"'{args.adaptation}'. Use --overwrite to "
            "regenerate them."
        )
        return

    data_name = (
        f"{folder}/task_paths_data_"
        f"{args.ik}_{args.planner}.npy"
    )
    keys_name = (
        f"{folder}/task_paths_keys_"
        f"{args.ik}_{args.planner}.pkl"
    )

    try:
        data = np.load(
            data_name,
            allow_pickle=True,
        )

        with open(keys_name, "rb") as file:
            keys = pickle.load(file)

        if len(keys) != len(data):
            raise ValueError(
                "Task-path key and data counts differ: "
                f"{len(keys)} keys versus {len(data)} paths."
            )

        task_paths = {
            key: path
            for key, path in zip(keys, data)
        }

    except FileNotFoundError as error:
        print(error)
        print(
            f"Task paths with IK '{args.ik}' and "
            f"planner '{args.planner}' were not found. "
            "Generate the full task-path library first."
        )
        return

    root_paths, key_to_root, results = (
        condense_dataset_parallel(
            env_name=args.env,
            robot_name=args.robot,
            task_paths=task_paths,
            adaptation=args.adaptation,
            n_neighbors=args.n_neighbors,
            num_workers=args.num_workers,
            seed=args.seed,
        )
    )

    np.save(
        f"{folder}/root_paths_results_{suffix}.npy",
        results,
    )

    with open(
        f"{folder}/root_paths_{suffix}.pkl",
        "wb",
    ) as file:
        pickle.dump(root_paths, file)

    with open(
        f"{folder}/key_to_root_{suffix}.pkl",
        "wb",
    ) as file:
        pickle.dump(key_to_root, file)


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
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
            "allstable",
        ],
        default="table",
    )
    parser.add_argument(
        "--robot",
        choices=["panda", "ur10", "fetch"],
        default="panda",
    )
    parser.add_argument(
        "--ik",
        choices=["random", "neighbor", "grr"],
        default="neighbor",
    )
    parser.add_argument(
        "--planner",
        choices=[
            "RRTConnect",
            "PRMstar",
            "LazyPRMstar",
            "VAMP",
        ],
        default="RRTConnect",
    )
    parser.add_argument(
        "--adaptation",
        choices=["linear", "grr", "dmp", "opt"],
        default="grr",
    )
    parser.add_argument(
        "--n_neighbors",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--num-workers",
        "--num_workers",
        dest="num_workers",
        type=int,
        default=2,
        help="Number of independent condensation workers.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


if __name__ == "__main__":
    set_seed(42)
    main(parse_arguments())
