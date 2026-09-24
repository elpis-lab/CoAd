"""Replay saved benchmark paths or compare selected methods on sampled problems."""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from experiments.common import (
    METHODS,
    add_dataset_args,
    load_results,
    method_results,
    move_object,
    positive_int,
)
from experiments.visualize_env import add_camera_args, set_camera


def play_path(robot, path, dt=0.03):
    if path is None or not len(path):
        return
    for q in path:
        if not robot.viewer.is_running():
            break
        robot.set_joint_qpos(q)
        robot.viewer.sync()
        time.sleep(dt)


def main(args):
    from coad.utils import load_env_and_robot
    from experiments.benchmark import Benchmark, validate_path

    saved = load_results(args.results) if args.results else None
    if saved:
        metadata = saved.get("metadata", {})
        args.env = metadata.get("env", args.env)
        args.robot = metadata.get("robot", args.robot)
        if "queries" not in saved:
            raise ValueError(
                "This file has no saved queries; rerun benchmark_planning with --save-paths"
            )
    env, robot = load_env_and_robot(
        args.env,
        args.robot,
        visualize=True,
        using_swept_volume=False,
        compute_tcr=False,
    )
    set_camera(robot.viewer.cam, robot.model, args, robot.data)
    robot.viewer.opt.geomgroup[3] = False
    try:
        if saved:
            queries = saved["queries"]
            if args.query < 0 or args.query >= len(queries):
                raise ValueError(f"--query must be between 0 and {len(queries)-1}")
            sample, goal = queries[args.query]
            outputs = method_results(saved)
            selected = args.methods or list(outputs)
            paths = []
            for method in selected:
                if method not in outputs or "paths" not in outputs[method]:
                    raise ValueError(
                        f"No saved paths for {method}; benchmark with --save-paths"
                    )
                paths.append((method, outputs[method]["paths"][args.query]))
        else:
            args.methods = args.methods or ["grr", "opt", "dmp", "rrtc"]
            bench = Benchmark(args, env, robot)
            sample, goal = bench.queries(args.query + 1, planning=True)[args.query]
            bench.setup(args.methods)
            paths = []
            for method in args.methods:
                path, seconds, target = bench.solve(method, sample, goal)
                valid = validate_path(bench.validator, path, bench.home, target)
                print(f'{method}: {"success" if valid else "failed"}, {seconds:.6f} s')
                paths.append((method, path if valid else None))
        move_object(env, sample)
        print(f"Object pose: {sample}")
        while robot.viewer.is_running():
            for method, path in paths:
                if not robot.viewer.is_running():
                    return
                if path is None or not len(path):
                    print(f"{method}: no successful path to display")
                    continue
                print(f"Playing {method}: {len(path)} waypoints")
                robot.set_joint_qpos(path[0])
                robot.viewer.sync()
                time.sleep(args.pause)
                play_path(robot, path, args.playback_dt)
                time.sleep(args.pause)
            if not args.loop:
                break
    finally:
        robot.close()


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    add_dataset_args(parser)
    parser.add_argument("--methods", nargs="+", choices=METHODS)
    parser.add_argument(
        "--results",
        type=Path,
        help="Replay paths saved by benchmark_planning --save-paths",
    )
    parser.add_argument("--query", type=int, default=0, help="Zero-based query index")
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--library-size", type=positive_int)
    parser.add_argument("--library-k", type=positive_int, default=5)
    parser.add_argument("--playback-dt", "--playback_dt", type=float, default=0.03)
    parser.add_argument("--pause", type=float, default=0.5)
    parser.add_argument("--loop", action="store_true")
    add_camera_args(parser)
    args = parser.parse_args()
    if args.query < 0 or args.timeout <= 0 or args.playback_dt < 0 or args.pause < 0:
        parser.error(
            "query/playback/pause must be nonnegative and timeout must be positive"
        )
    return args


if __name__ == "__main__":
    args = parse_arguments()
    args.data_root = args.data_root.resolve()
    args.results = args.results.resolve() if args.results else None
    os.chdir(Path(__file__).resolve().parents[1])
    main(args)
