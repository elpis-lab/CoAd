"""Benchmark selected methods on identical random object-pose planning problems."""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import METHODS, add_dataset_args, positive_int


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    add_dataset_args(parser)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=["grr", "opt", "dmp", "rrtc", "vamp", "lightning", "ertconnect"],
    )
    parser.add_argument("--samples", "--num-samples", type=positive_int, default=1000)
    parser.add_argument(
        "--timeout",
        type=float,
        default=3.0,
        help="OMPL/experience planner timeout; VAMP uses its configured iteration limit",
    )
    parser.add_argument(
        "--library-size",
        type=positive_int,
        help="Default: largest compressed library for the selected dataset (microwave capped at 50000)",
    )
    parser.add_argument("--library-k", type=positive_int, default=5)
    parser.add_argument(
        "--save-paths",
        action="store_true",
        help="Include trajectories for visualize_paths --results",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    return args


def main(args):
    from experiments.benchmark import run

    return run(args, planning=True)


if __name__ == "__main__":
    args = parse_arguments()
    args.data_root = args.data_root.resolve()
    args.output = args.output.resolve() if args.output else None
    os.chdir(Path(__file__).resolve().parents[1])
    main(args)
