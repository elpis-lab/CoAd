"""Result discovery and tables, shared by the two summaries and planning plots."""

import argparse
import csv
from pathlib import Path

import numpy as np
from experiments.common import (
    ENVS,
    ROBOTS,
    METHODS,
    LABELS,
    REPO,
    load_results,
    method_results,
    metrics,
)


def parser(description):
    result = argparse.ArgumentParser(description=description)
    result.add_argument("--data-root", type=Path, default=REPO / "data")
    result.add_argument("--robots", "--robot", nargs="+", choices=ROBOTS)
    result.add_argument("--envs", "--env", nargs="+", choices=ENVS)
    result.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    result.add_argument(
        "--files", nargs="+", type=Path, help="Explicit NPZ files instead of discovery"
    )
    return result


def datasets(args, kind):
    paths = args.files or sorted(args.data_root.glob(f"{kind}_results_*.npz"))
    found = []
    for path in paths:
        # Exclude historical backup and partial files from automatic discovery.
        if not args.files and "." in path.stem:
            continue
        results = load_results(path)
        metadata = results.get("metadata", {})
        name = path.stem.removeprefix(f"{kind}_results_")
        robot, _, env = name.partition("_")
        robot, env = metadata.get("robot", robot), metadata.get("env", env)
        if args.robots and robot not in args.robots:
            continue
        if args.envs and env not in args.envs:
            continue
        found.append((robot, env, results))
    if not found:
        raise FileNotFoundError("No matching completed result files")
    return found


def finite_mean(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else np.nan


def table(args, adaptation=False):
    rows = []
    for robot, env, results in datasets(
        args, "adaptation" if adaptation else "baseline"
    ):
        for method, values in method_results(results).items():
            if method not in args.methods:
                continue
            success, times, lengths = metrics(values)
            row = dict(
                robot=robot,
                env=env,
                method=LABELS.get(method, method),
                trials=len(success),
                success_percent=100 * success.mean() if len(success) else np.nan,
                time_ms=1000 * finite_mean(times[success]),
                length=finite_mean(lengths[success]),
            )
            if adaptation:
                row.update(
                    library_size=values.get("library_size", np.nan),
                    task_count=values.get("task_count", np.nan),
                    compression_percent=values.get("compression", np.nan),
                )
            rows.append(row)
    if not rows:
        raise ValueError("No selected methods exist in these results")
    headers = list(rows[0])
    print(" | ".join(headers))
    print(" | ".join("---" for _ in headers))
    for row in rows:
        print(
            " | ".join(
                (
                    f"{value:.3f}"
                    if isinstance(value, (float, np.floating)) and np.isfinite(value)
                    else (
                        "N/A" if isinstance(value, (float, np.floating)) else str(value)
                    )
                )
                for value in row.values()
            )
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)
    return rows
