"""Dataset, command-line and result helpers shared by experiment entry points."""

import argparse
import hashlib
import os
import pickle
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ENVS = (
    "table",
    "box",
    "cage",
    "shelf",
    "free",
    "real",
    "largeobj",
    "microwave",
    "allstable",
    "conveyor",
)
ROBOTS = ("panda", "fetch", "ur10")
ADAPTATIONS = ("grr", "opt", "dmp", "linear")
METHODS = (
    "full",
    "grr",
    "opt",
    "dmp",
    "linear",
    "rrtc",
    "prmstar",
    "vamp",
    "lightning",
    "ertconnect",
)
LABELS = dict(
    full="Full library",
    grr="GRR",
    opt="TrajOpt",
    dmp="DMP",
    linear="Linear",
    rrtc="RRTConnect",
    prmstar="PRMstar",
    vamp="VAMP",
    lightning="Lightning",
    ertconnect="ERTConnect",
)


class DatasetUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "plan_load" or module.startswith("plan_load."):
            module = "coad" + module[len("plan_load") :]
        return super().find_class(module, name)


def read_pickle(path):
    with Path(path).open("rb") as stream:
        return DatasetUnpickler(stream).load()


def hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def add_dataset_args(parser):
    parser.add_argument("--env", choices=ENVS, default="table")
    parser.add_argument("--robot", choices=ROBOTS, default="panda")
    parser.add_argument("--data-root", type=Path, default=REPO / "data")
    parser.add_argument(
        "--ik", choices=["random", "neighbor", "grr"], default="neighbor"
    )
    parser.add_argument(
        "--planner",
        choices=["RRTConnect", "PRMstar", "VAMP"],
        default="RRTConnect",
        help="Planner used to generate the saved dataset",
    )
    parser.add_argument(
        "--n-neighbors", "--n_neighbors", type=positive_int, default=1000
    )
    parser.add_argument(
        "--reference-method",
        choices=ADAPTATIONS,
        default="grr",
        help="Compressed library supplying query regions/goals and experiences",
    )
    parser.add_argument("--seed", type=int, default=42)


def dataset_folder(args):
    return args.data_root / f"{args.env}_{args.robot}"


def library_paths(args, method):
    suffix = f"{args.ik}_{args.planner}_{method}_{args.n_neighbors}"
    folder = dataset_folder(args)
    return folder / f"root_paths_{suffix}.pkl", folder / f"key_to_root_{suffix}.pkl"


def load_library(args, method):
    roots_path, map_path = library_paths(args, method)
    roots, mapping = read_pickle(roots_path), read_pickle(map_path)
    # Equal-length paths saved with dtype=object can also give their endpoint
    # arrays object dtype. Normalize in memory without rewriting the dataset.
    mapping = {
        key: (root_id, None if goal is None else np.asarray(goal, dtype=float))
        for key, (root_id, goal) in mapping.items()
    }
    return roots, mapping


def usable_keys(roots, mapping, *, allow_empty=False):
    keys = []
    for key, (root_id, goal) in mapping.items():
        if root_id is None or goal is None:
            continue
        root = roots[root_id]
        # DMP centers are objects rather than waypoint arrays.
        if root is not None and (not hasattr(root, "__len__") or len(root)):
            keys.append(key)
    if not keys and not allow_empty:
        raise ValueError("The selected reference library has no usable tasks")
    return keys


def move_object(env, sample):
    if env.object_details["type"] == "microwave":
        env.move_object(sample[:4])
        env.move_xml_joint("microwave_door_hinge", sample[4])
    else:
        env.move_object(sample)


def make_adapter(method, robot, solver):
    from coad.adaptation import GRRAdapter, TrajOptAdapter, DMPAdapter, LinearAdapter

    return {
        "grr": GRRAdapter,
        "opt": TrajOptAdapter,
        "dmp": DMPAdapter,
        "linear": LinearAdapter,
    }[method](robot, solver)


def load_full_paths(args):
    folder = dataset_folder(args)
    suffix = f"{args.ik}_{args.planner}"
    keys = read_pickle(folder / f"task_paths_keys_{suffix}.pkl")
    paths = np.load(folder / f"task_paths_data_{suffix}.npy", allow_pickle=True)
    if len(keys) != len(paths):
        raise ValueError("Task-path keys and data have different lengths")
    return {
        key: np.asarray(path, dtype=float)
        for key, path in zip(keys, paths) if path is not None and len(path)
    }


def load_results(path):
    with np.load(path, allow_pickle=True) as saved:
        return saved["results"].item()


def save_results(path, results):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(temporary, results=results)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def method_results(results):
    """Read both the consolidated format and historical experiment NPZ files."""
    if "methods" in results:
        return results["methods"]
    methods = {}
    for name in METHODS:
        key = "library" if name == "lightning" else name
        if (
            key in results
            and isinstance(results[key], dict)
            and "success" in results[key]
        ):
            methods[name] = results[key]
    adaptations = results.get("adaptations", {})
    for name, success in adaptations.get("success", {}).items():
        methods[name] = {
            field: adaptations[field][name] for field in ("success", "times", "lengths")
        }
        for field in ("compression", "library_sizes"):
            if name in adaptations.get(field, {}):
                methods[name][field] = adaptations[field][name]
    return methods


def metrics(values):
    success = np.asarray(values["success"], dtype=bool).reshape(-1)
    times = np.asarray(values["times"], dtype=float).reshape(-1)
    lengths = np.asarray(values["lengths"], dtype=float).reshape(-1)
    if not (len(success) == len(times) == len(lengths)):
        raise ValueError("success, times, and lengths arrays have different lengths")
    return success, times, lengths
