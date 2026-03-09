import os
import argparse
import numpy as np


def load_results(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    if "results" not in data:
        raise KeyError(f"'results' key not found in {npz_path}")
    return data["results"].item()


def replace_dmp_in_baseline(robot, env):
    baseline_path = f"data/baseline_results_{robot}_{env}.npz"
    dmp_path = f"data/dmp_results_{robot}_{env}.npz"

    if not os.path.exists(baseline_path):
        raise FileNotFoundError(f"Baseline results file not found: {baseline_path}")
    if not os.path.exists(dmp_path):
        raise FileNotFoundError(f"DMP results file not found: {dmp_path}")

    baseline_results = load_results(baseline_path)
    dmp_results = load_results(dmp_path)

    # Validate baseline structure
    if "adaptations" not in baseline_results:
        raise KeyError(f"'adaptations' missing in baseline results: {baseline_path}")
    for key in ["success", "times", "lengths"]:
        if key not in baseline_results["adaptations"]:
            raise KeyError(f"'adaptations/{key}' missing in baseline results")

    # Validate dmp-only structure
    if "adaptations" not in dmp_results:
        raise KeyError(f"'adaptations' missing in dmp results: {dmp_path}")
    for key in ["success", "times", "lengths"]:
        if key not in dmp_results["adaptations"]:
            raise KeyError(f"'adaptations/{key}' missing in dmp results")
        if "dmp" not in dmp_results["adaptations"][key]:
            raise KeyError(f"'adaptations/{key}/dmp' missing in dmp results")

    # Replace only dmp entries
    baseline_results["adaptations"]["success"]["dmp"] = dmp_results["adaptations"]["success"]["dmp"]
    baseline_results["adaptations"]["times"]["dmp"] = dmp_results["adaptations"]["times"]["dmp"]
    baseline_results["adaptations"]["lengths"]["dmp"] = dmp_results["adaptations"]["lengths"]["dmp"]

    # Overwrite baseline file
    np.savez(baseline_path, results=baseline_results)
    print(f"[Updated] Replaced DMP entries in: {baseline_path}")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        choices=["table", "box", "cage", "shelf", "free", "real"],
        required=True,
    )
    parser.add_argument(
        "--robot",
        choices=["panda", "ur10", "fetch"],
        required=True,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    replace_dmp_in_baseline(args.robot, args.env)