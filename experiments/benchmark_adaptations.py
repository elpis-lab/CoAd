"""Benchmark retrieval/adaptation over saved task regions and record compression."""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import ADAPTATIONS, add_dataset_args, positive_int


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    add_dataset_args(parser)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("full", *ADAPTATIONS),
        default=["grr", "opt", "dmp"],
    )
    parser.add_argument(
        "--samples",
        "--num-samples",
        type=positive_int,
        help="Limit the number of task regions; default: every usable reference region",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main(args):
    from experiments.benchmark import run

    return run(args, planning=False)


if __name__ == "__main__":
    args = parse_arguments()
    args.data_root = args.data_root.resolve()
    args.output = args.output.resolve() if args.output else None
    os.chdir(Path(__file__).resolve().parents[1])
    main(args)
