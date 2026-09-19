"""Shared execution for adaptation and random-problem planning benchmarks."""

import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from experiments.common import (
    ADAPTATIONS,
    dataset_folder,
    hash_file,
    library_paths,
    load_library,
    load_full_paths,
    load_results,
    make_adapter,
    move_object,
    save_results,
    usable_keys,
)
from experiments.indexing import BoxGrid, sample_from_key


def validate_path(planner, path, start, goal):
    if path is None:
        return False
    points = np.asarray(path, dtype=float)
    if (
        points.ndim != 2
        or points.shape[1] != len(start)
        or len(points) < 2
        or not np.isfinite(points).all()
        or not np.allclose(points[0], start, atol=1e-5)
        or not np.allclose(points[-1], goal, atol=1e-5)
    ):
        return False
    native = planner.np_path_to_path_geometric(points)
    return bool(
        all(planner.si.satisfiesBounds(s) for s in native.getStates())
        and planner.validate_path(native)
    )


class Benchmark:
    """Load only selected methods; keep library construction outside online timing."""

    def __init__(self, args, env, robot):
        from coad.planning import OMPLPlanner

        self.args, self.env, self.robot = args, env, robot
        self.home = robot.get_joint_qpos().copy()
        self.validator = OMPLPlanner(robot, robot.data)
        self.libraries = {}
        self.adapters = {}
        self.planners = {}
        self.full_paths = None
        self.full_index = None
        self.experiences = {}
        self.experience_library = None
        self.library_size = None
        self.reference_roots, self.reference_map = self.load(args.reference_method)
        self.keys = usable_keys(self.reference_roots, self.reference_map)
        self.reference_index = BoxGrid(self.reference_map)

    def load(self, method):
        if method not in self.libraries:
            roots, mapping = load_library(self.args, method)
            self.libraries[method] = roots, mapping
        return self.libraries[method]

    def setup(self, methods):
        from coad.planning import OMPLPlanner, VAMPPlanner, ERTConnectPlanner

        args = self.args
        if any(m in ADAPTATIONS for m in methods):
            from coad.mink_ik import get_ik_solver

            solver = get_ik_solver(
                self.robot, env_collision_geoms=self.env.env_details["collision_geoms"]
            )
            for method in methods:
                if method in ADAPTATIONS:
                    roots, mapping = self.load(method)
                    self.adapters[method] = make_adapter(
                        method, self.robot, solver
                    ), BoxGrid(mapping)
        if "full" in methods:
            self.full_paths = load_full_paths(args)
            self.full_index = BoxGrid(self.full_paths)
        for method in ("rrtc", "prmstar"):
            if method in methods:
                self.planners[method] = OMPLPlanner(
                    self.robot,
                    self.robot.data,
                    planner="RRTConnect" if method == "rrtc" else "PRMstar",
                    rrtc_range=(
                        0.1
                        if args.robot == "fetch" and args.env in ("cage", "shelf")
                        else None
                    ),
                )
        if "vamp" in methods:
            self.planners["vamp"] = VAMPPlanner(
                self.robot, self.env, self.robot.data, robot_name=args.robot
            )
        if "ertconnect" in methods:
            self.planners["ertconnect"] = ERTConnectPlanner(self.robot, self.robot.data)
        if {"lightning", "ertconnect"} & set(methods):
            from experiments.experience_library import (
                Library,
                ReferencedExperienceLibrary,
                count_pickled_roots,
            )

            if args.reference_method == "dmp":
                raise ValueError(
                    "Experience planners need a waypoint reference library; use --reference-method grr or opt"
                )
            counts = [
                count_pickled_roots(library_paths(args, method)[0])
                for method in ADAPTATIONS
                if library_paths(args, method)[0].exists()
            ]
            size = args.library_size or max(counts)
            if args.library_size is None and args.env == "microwave":
                size = min(size, 50000)
            # A local RNG state makes the experience library independent of selected methods.
            state = np.random.get_state()
            np.random.seed(args.seed + 1)
            try:
                self.experience_library = ReferencedExperienceLibrary(
                    size, self.reference_map, self.reference_roots, self.keys
                )
            finally:
                np.random.set_state(state)
            self.library_size = len(self.experience_library.library)
            if "lightning" in methods:
                # Reuse repair methods while materializing only the queried root paths.
                class Lightning(ReferencedExperienceLibrary, Library):
                    def query_library_nn(self, index, sample, n=1):
                        found = super().query_library_nn(index, sample, n)
                        return [
                            (pose, (value[0], self.get_path(pose)), distance)
                            for pose, value, distance in found
                        ]

                lib = Lightning.__new__(Lightning)
                lib.__dict__.update(self.experience_library.__dict__)
                lib.robot, lib.home_qpos = self.robot, self.home
                lib.key_to_root, lib.indexer = self.reference_map, self.reference_index
                lib.ompl_planner = self.validator
                self.lightning = lib

    def queries(self, count=None, planning=False):
        np.random.seed(self.args.seed)
        if not planning:
            keys = self.keys
            if count is not None and count < len(keys):
                keys = [
                    keys[i] for i in np.random.choice(len(keys), count, replace=False)
                ]
            return [(sample_from_key(key), self.reference_map[key][1]) for key in keys]
        queries = []
        for _ in range(count * 1000):
            if len(queries) == count:
                break
            key = self.keys[np.random.randint(len(self.keys))]
            sample = sample_from_key(key)
            recovered = self.reference_index.query_point(sample)
            if recovered is None:
                continue
            goal = self.reference_map[recovered][1]
            if goal is None:
                continue
            move_object(self.env, sample)
            valid = True
            for q in (self.home, goal):
                state = self.validator.numpy_to_state(q)
                if not self.validator.si.satisfiesBounds(
                    state
                ) or not self.validator.validity_checker(state):
                    valid = False
                    break
            if valid:
                queries.append((sample, np.asarray(goal).copy()))
        self.robot.set_joint_qpos(self.home)
        if len(queries) != count:
            raise RuntimeError(
                f"Only found {len(queries)}/{count} valid queries; check reference data and scene"
            )
        return queries

    def solve(self, method, sample, goal):
        """Return trajectory, online seconds and the method's stored target joint pose."""
        move_object(self.env, sample)
        self.robot.set_joint_qpos(self.home)
        begin = time.perf_counter()
        path = None
        target = goal
        if method in self.adapters:
            adapter, index = self.adapters[method]
            key = index.query_point(sample)
            roots, mapping = self.libraries[method]
            if key is not None:
                root_id, target = mapping[key]
                if (
                    root_id is not None
                    and target is not None
                    and roots[root_id] is not None
                ):
                    path = adapter.adapt(roots[root_id], target)
        elif method == "full":
            key = self.full_index.query_point(sample)
            if key is not None:
                path = self.full_paths[key]
                target = path[-1]
        elif method in ("rrtc", "prmstar"):
            path, _, _ = self.planners[method].plan(
                self.home,
                goal,
                timeout=self.args.timeout,
                smooth_path=False,
                num_waypoints=200,
                benchmark=True,
            )
        elif method == "vamp":
            path, _, _ = self.planners[method].plan(
                self.home, goal, smooth_path=False, num_waypoints=200, benchmark=True
            )
        elif method == "lightning":
            path, _, ok = self.lightning.solve(
                sample,
                k=self.args.library_k,
                timeout=self.args.timeout,
                goal=goal,
                start=self.home,
            )
            if not ok:
                path = None
        elif method == "ertconnect":
            nearest = self.experience_library.query_library_nn(
                self.experience_library.lib_index, sample, n=1
            )
            if nearest:
                pose = nearest[0][0]
                query_seconds = time.perf_counter() - begin
                planner = self.planners[method]
                if pose not in self.experiences:
                    self.experiences[pose] = planner.prepare_experience(
                        self.experience_library.get_path(pose)
                    )
                path, _, seconds, ok = planner.solve_experience(
                    self.home, goal, self.experiences[pose], self.args.timeout
                )
                return path if ok else None, query_seconds + seconds, target
        else:
            raise ValueError(f"Method was not initialized: {method}")
        return path, time.perf_counter() - begin, target


def run(args, planning):
    from coad.planning import euclidean_path_length
    from coad.utils import load_env_and_robot, set_seed
    from ompl import util as ou

    set_seed(args.seed)
    ou.RNG.setSeed(args.seed)
    path = (
        args.output
        or args.data_root
        / f'{"baseline" if planning else "adaptation"}_results_{args.robot}_{args.env}.npz'
    )
    methods = list(dict.fromkeys(args.methods))
    # Hash reference files so selected-method reruns cannot silently mix datasets.
    reference_paths = library_paths(args, args.reference_method)
    metadata = dict(
        schema=1,
        robot=args.robot,
        env=args.env,
        ik=args.ik,
        planner=args.planner,
        n_neighbors=args.n_neighbors,
        reference_method=args.reference_method,
        seed=args.seed,
        samples=args.samples,
        planning=planning,
        reference_sha256=[hash_file(p) for p in reference_paths],
        timeout=getattr(args, "timeout", None),
        library_size=getattr(args, "library_size", None),
        library_k=getattr(args, "library_k", None),
        times_definition="Online retrieval/planning/extraction; excludes dataset loading, library construction, native experience conversion and independent validation",
        goals_definition="Baselines use reference joint goals; adapters/full use their stored goals (which may be IK-refined)",
    )
    results = {}
    if path.exists():
        results = load_results(path)
        if results.get("metadata") != metadata:
            if not args.overwrite:
                raise ValueError(
                    f"{path} belongs to another configuration or legacy format; use --output or --overwrite"
                )
            results = {}
    results.setdefault("metadata", metadata)
    results.setdefault("methods", {})
    pending = [
        m
        for m in methods
        if args.overwrite
        or m not in results["methods"]
        or (getattr(args, "save_paths", False) and "paths" not in results["methods"][m])
    ]
    for method in methods:
        if (
            method in ADAPTATIONS
            and method in results["methods"]
            and not args.overwrite
        ):
            current_hashes = [hash_file(p) for p in library_paths(args, method)]
            if results["methods"][method].get("source_sha256") != current_hashes:
                raise ValueError(
                    f"{method} library has changed; rerun with --overwrite or a different --output"
                )
    if not pending:
        print(f"[Skip] Selected methods already saved: {path}")
        return
    env, robot = load_env_and_robot(
        args.env,
        args.robot,
        visualize=False,
        using_swept_volume=False,
        compute_tcr=False,
    )
    try:
        bench = Benchmark(args, env, robot)
        if "queries" in results:
            queries = results["queries"]
        else:
            queries = bench.queries(args.samples, planning)
            results["queries"] = queries
            results["home"] = bench.home
        bench.setup(pending)
        results["library_size"] = bench.library_size or results.get("library_size")
        for method in pending:
            set_seed(args.seed)
            output = {
                field: []
                for field in (
                    "success",
                    "reported_success",
                    "times",
                    "lengths",
                    "validation_seconds",
                )
            }
            if getattr(args, "save_paths", False):
                output["paths"] = []
            for sample, goal in tqdm(queries, desc=method):
                trajectory, seconds, target = bench.solve(method, sample, goal)
                reported = trajectory is not None and len(trajectory) > 0
                start = time.perf_counter()
                valid = (
                    validate_path(bench.validator, trajectory, bench.home, target)
                    if reported
                    else False
                )
                output["validation_seconds"].append(time.perf_counter() - start)
                output["reported_success"].append(reported)
                output["success"].append(valid)
                output["times"].append(seconds)
                output["lengths"].append(
                    euclidean_path_length(trajectory) if valid else np.nan
                )
                if "paths" in output:
                    output["paths"].append(trajectory if valid else None)
            for field in output:
                if field != "paths":
                    output[field] = np.asarray(output[field])
            if method in ADAPTATIONS:
                roots, mapping = bench.libraries[method]
                output["library_size"] = len(roots)
                output["task_count"] = len(
                    usable_keys(roots, mapping, allow_empty=True)
                )
                output["compression"] = (
                    100 * (1 - len(roots) / output["task_count"])
                    if output["task_count"]
                    else np.nan
                )
                output["source_sha256"] = [
                    hash_file(p) for p in library_paths(args, method)
                ]
            if method == "full":
                output.update(
                    library_size=len(bench.full_paths),
                    task_count=len(bench.full_paths),
                    compression=0.0,
                )
            results["methods"][method] = output
            save_results(path, results)
            print(
                f'[Saved] {method}: {sum(output["success"])}/{len(queries)} successful; {path}'
            )
    finally:
        robot.close()
