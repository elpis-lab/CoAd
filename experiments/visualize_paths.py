import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import pickle
import argparse
import numpy as np

from tqdm import tqdm
from scipy.spatial import cKDTree

from plan_load.env import MujocoEnv
from plan_load.robot import MujocoRobot
from plan_load.mink_ik import get_ik_solver
from plan_load.adaptation import LinearAdapter, DMPAdapter, TrajOptAdapter
from plan_load.planning import OMPLPlanner
from plan_load.utils import get_data_folder

from plan_load.env import FreeEnv, CageEnv, BoxEnv, TableEnv, ShelfEnv
from plan_load.robot import Panda, UR10, FetchArm


def traj_len(traj):
    """Compute the length of a trajectory."""
    traj = np.asarray(traj, dtype=np.float64)
    if traj.ndim != 2 or len(traj) < 2:
        return 0.0

    # Differences between consecutive states
    diffs = np.diff(traj, axis=0)
    # Euclidean norms of each segment
    segment_lengths = np.linalg.norm(diffs, axis=1)
    return np.sum(segment_lengths)


class BoxGrid:
    def __init__(self, keys_to_root, tol=1e-12):
        keys = np.asarray(list(keys_to_root.keys()), dtype=np.float64)
        self.keys_list = list(keys_to_root.keys())
        self.keys_arr = keys

        if keys.ndim != 3 or keys.shape[1:] != (4, 2):
            raise ValueError(f"Expected keys shape (N,4,2), got {keys.shape}")

        mins = keys[:, :, 0].copy()
        maxs = keys[:, :, 1].copy()

        mins[:, 3] = self._wrap_pi(mins[:, 3])
        maxs[:, 3] = self._wrap_pi(maxs[:, 3])

        self.x_mins = np.sort(np.unique(mins[:, 0]))
        self.y_mins = np.sort(np.unique(mins[:, 1]))

        self._x_to_ix = {float(v): i for i, v in enumerate(self.x_mins)}
        self._y_to_iy = {float(v): i for i, v in enumerate(self.y_mins)}

        self.nx = len(self.x_mins)
        self.ny = len(self.y_mins)

        self.z_values = np.sort(np.unique(mins[:, 2]))
        self.nz = len(self.z_values)

        yaw_mins = np.sort(np.unique(mins[:, 3]))
        self.nyaw = len(yaw_mins) if yaw_mins.size else 1
        self.yaw0 = float(yaw_mins[0]) if yaw_mins.size else 0.0

        def spacing(vals, default=1.0):
            vals = np.sort(np.unique(vals))
            diffs = np.diff(vals)
            diffs = diffs[diffs > tol]
            return float(diffs.min()) if diffs.size else float(default)

        self.dyaw = spacing(yaw_mins)
        self.index = {}

        for bin_idx, key in enumerate(self.keys_list):
            x_min, y_min, z_val, yaw_min = (
                key[0][0],
                key[1][0],
                key[2][0],
                key[3][0],
            )

            ix = self._x_to_ix[float(x_min)]
            iy = self._y_to_iy[float(y_min)]
            iz = int(np.argmin(np.abs(self.z_values - z_val)))

            rel = self._wrap_pi(yaw_min - self.yaw0)
            iyaw = int(np.floor(rel / self.dyaw + tol)) % self.nyaw

            indices = (ix, iy, iz, iyaw)
            if indices in self.index:
                raise RuntimeError(f"Duplicate bin index {indices}")

            self.index[indices] = bin_idx

    def _bin_indices(self, sample, eps=1e-12):
        x, y, z, yaw = sample
        yaw = self._wrap_pi(yaw)

        ix = int(np.searchsorted(self.x_mins, x, side="right") - 1)
        iy = int(np.searchsorted(self.y_mins, y, side="right") - 1)

        ix = min(max(ix, 0), len(self.x_mins) - 1)
        iy = min(max(iy, 0), len(self.y_mins) - 1)

        if ix < 0 or ix >= self.nx or iy < 0 or iy >= self.ny:
            return None

        iz = int(np.argmin(np.abs(self.z_values - z)))
        rel = self._wrap_pi(yaw - self.yaw0)
        iyaw = int(np.floor(rel / self.dyaw + eps)) % self.nyaw

        return ix, iy, iz, iyaw

    def query_point(self, sample):
        indices = self._bin_indices(sample)
        if indices is None:
            return None

        bin_idx = self.index.get(indices, None)
        if bin_idx is None:
            return None

        return self.keys_list[bin_idx]

    @staticmethod
    def _wrap_pi(a):
        return (a + np.pi) % (2 * np.pi) - np.pi


class Library:
    def __init__(
        self,
        N,
        env: MujocoEnv,
        robot: MujocoRobot,
        home_qpos,
        key_to_root,
        solved_keys,
        task_paths,
        data,
    ):
        self.key_to_root = key_to_root
        self.task_paths = task_paths
        self.indexer = BoxGrid(task_paths)
        self.robot = robot

        if isinstance(env, ShelfEnv) and isinstance(robot, FetchArm):
            self.ompl_planner = OMPLPlanner(robot, data, rrtc_range=0.1)
        elif isinstance(env, CageEnv) and isinstance(robot, FetchArm):
            self.ompl_planner = OMPLPlanner(robot, data, rrtc_range=0.1)
        else:
            self.ompl_planner = OMPLPlanner(robot, data)

        solved_key_list = list(solved_keys)

        self.library = {}
        pbar_library = tqdm(total=N, desc="Building library", leave=True)

        while len(self.library) < N:
            random_key = solved_key_list[
                np.random.randint(0, len(solved_key_list))
            ]
            sample = [np.random.uniform(lo, hi) for lo, hi in random_key]

            recovered_key = self.indexer.query_point(sample)
            if recovered_key is None:
                continue

            path = task_paths[recovered_key]
            if path is None or len(path) == 0:
                continue

            curr_goal = path[-1]

            if tuple(sample) not in self.library:
                self.library[tuple(sample)] = (curr_goal, path)
                pbar_library.update(1)

        self.lib_index = self.build_library_index(w_yaw=1.0)

    def build_library_index(self, z_tol=1e-6, w_yaw=1.0):
        keys = np.asarray(list(self.library.keys()), dtype=np.float64)
        if keys.ndim != 2 or keys.shape[1] != 4:
            raise ValueError(f"Expected keys shape (N,4), got {keys.shape}")

        z_vals = np.unique(keys[:, 2])
        z_vals = np.sort(z_vals)

        trees = {}
        key_lists = {}
        yaw_scale = np.sqrt(w_yaw)

        for z0 in z_vals:
            mask = np.abs(keys[:, 2] - z0) <= z_tol
            kz = keys[mask]
            if kz.size == 0:
                continue

            feats = np.column_stack(
                [
                    kz[:, 0],
                    kz[:, 1],
                    yaw_scale * np.cos(kz[:, 3]),
                    yaw_scale * np.sin(kz[:, 3]),
                ]
            )

            trees[float(z0)] = cKDTree(feats)
            key_lists[float(z0)] = [tuple(row) for row in kz]

        return {
            "z_vals": z_vals,
            "trees": trees,
            "key_lists": key_lists,
            "yaw_scale": yaw_scale,
            "z_tol": z_tol,
        }

    def query_library_nn(self, index, sample, n=1):
        x, y, z, yaw = map(float, sample)

        z_vals = index["z_vals"]
        zi = int(np.argmin(np.abs(z_vals - z)))
        z0 = float(z_vals[zi])

        if abs(z0 - z) > index["z_tol"]:
            return []

        tree = index["trees"].get(z0, None)
        if tree is None:
            return []

        ys = index["yaw_scale"]
        q = np.array(
            [x, y, ys * np.cos(yaw), ys * np.sin(yaw)], dtype=np.float64
        )

        num_points = len(index["key_lists"][z0])
        k = min(n, num_points)

        dists, idxs = tree.query(q, k=k)
        if k == 1:
            dists = np.array([dists])
            idxs = np.array([idxs])

        results = []
        key_list = index["key_lists"][z0]

        for dist, idx in zip(dists, idxs):
            key = key_list[int(idx)]
            results.append((key, self.library[key], float(dist)))

        return results

    def check_path_collision(self, path):
        T = len(path)
        waypoint_valid = np.zeros(T, dtype=bool)
        for i, q in enumerate(path):
            self.robot.set_joint_qpos(q)
            in_collision = self.robot.in_contact()
            waypoint_valid[i] = not in_collision
        return waypoint_valid

    def collision_buffer(self, waypoint_valid, b=0):
        if b <= 0:
            return waypoint_valid.copy()

        T = waypoint_valid.shape[0]
        buffered = waypoint_valid.copy()
        coll = np.flatnonzero(~waypoint_valid)
        if coll.size == 0:
            return buffered

        for i in coll:
            lo = max(0, i - b)
            hi = min(T, i + b + 1)
            buffered[lo:hi] = False

        return buffered

    def rewire_segments(
        self, path, validity_map, timeout=2.0, num_waypoints=20, max_repairs=20
    ):
        path = np.asarray(path, dtype=np.float64)
        valid = np.asarray(validity_map, dtype=bool)
        out = path.copy()

        prev_signature = None
        repairs = 0

        while True:
            invalid = ~valid
            n_invalid_before = int(invalid.sum())
            if n_invalid_before == 0:
                return out, True

            if repairs >= max_repairs:
                return out, True

            starts = np.flatnonzero(invalid & np.r_[True, ~invalid[:-1]])
            ends = np.flatnonzero(invalid & np.r_[~invalid[1:], True])

            rewired_any = False

            for s, e in zip(starts, ends):
                prev = s - 1
                while prev >= 0 and not valid[prev]:
                    prev -= 1

                nxt = e + 1
                while nxt < len(out) and not valid[nxt]:
                    nxt += 1

                if prev < 0 or nxt >= len(out):
                    continue

                sig = (prev, nxt, len(out))
                if sig == prev_signature:
                    return None, False
                prev_signature = sig

                q0, q1 = out[prev], out[nxt]

                rewired_segment, _, _ = self.ompl_planner.plan(
                    start=q0,
                    goal=q1,
                    timeout=timeout,
                    num_waypoints=num_waypoints,
                    benchmark=True,
                )

                if rewired_segment is None or len(rewired_segment) == 0:
                    return None, False

                rewired_segment = np.asarray(rewired_segment, dtype=np.float64)
                mid = (
                    rewired_segment[1:-1]
                    if rewired_segment.shape[0] >= 2
                    else rewired_segment
                )

                a = prev + 1
                out = np.vstack([out[:a], mid, out[nxt:]])
                valid = self.check_path_collision(out)

                n_invalid_after = int((~valid).sum())
                if n_invalid_after >= n_invalid_before:
                    return None, False

                repairs += 1
                rewired_any = True
                break

            if not rewired_any:
                return out, True

    def rewire_to_goal(self, path, goal, n_wps=20, timeout=1.0):
        path = np.asarray(path, dtype=np.float64)
        T = len(path)

        start_idx = None
        for i in range(T - 1, -1, -1):
            self.robot.set_joint_qpos(path[i])
            if not self.robot.in_contact():
                start_idx = i
                break

        if start_idx is None:
            return None, False

        q_start = path[start_idx]
        q_goal = np.asarray(goal, dtype=np.float64)

        t = np.linspace(0.0, 1.0, n_wps)[:, None]
        rewired_segment = (1.0 - t) * q_start + t * q_goal

        interpolation_valid = True
        for wp in rewired_segment:
            self.robot.set_joint_qpos(wp)
            if self.robot.in_contact():
                interpolation_valid = False
                break

        if not interpolation_valid:
            rewired_segment, _, _ = self.ompl_planner.plan(
                start=q_start,
                goal=q_goal,
                timeout=timeout,
                num_waypoints=n_wps,
                benchmark=True,
            )

        if rewired_segment is None or len(rewired_segment) == 0:
            return None, False

        rewired_segment = np.asarray(rewired_segment, dtype=np.float64)

        new_path = np.vstack([path[: start_idx + 1], rewired_segment[1:]])

        return new_path, True

    def solve(self, sample, k=5, timeout=3.0):
        nn_query_start = time.perf_counter()
        nn_results = self.query_library_nn(self.lib_index, sample, n=k)
        nn_query_end = time.perf_counter()
        nn_time = nn_query_end - nn_query_start

        recovered_key = self.indexer.query_point(sample)
        if recovered_key is None:
            return None, nn_time, False

        path = self.task_paths[recovered_key]
        curr_goal = path[-1]

        fix_start = time.perf_counter()
        final_path = None
        success = False

        for (
            neighbor_key,
            (neighbor_goal, neighbor_path),
            neighbor_dist,
        ) in nn_results:
            elapsed_fix = time.perf_counter() - fix_start
            if elapsed_fix > timeout:
                total_time = nn_time + elapsed_fix
                return None, total_time, False

            waypoints_valid = self.collision_buffer(
                self.check_path_collision(neighbor_path), b=0
            )

            remaining = timeout - (time.perf_counter() - fix_start)
            if remaining <= 0:
                total_time = nn_time + (time.perf_counter() - fix_start)
                return None, total_time, False

            rewired_path, ok = self.rewire_segments(
                neighbor_path,
                waypoints_valid,
                timeout=min(2.0, remaining),
                num_waypoints=20,
            )

            if not ok or rewired_path is None:
                continue

            remaining = timeout - (time.perf_counter() - fix_start)
            if remaining <= 0:
                total_time = nn_time + (time.perf_counter() - fix_start)
                return None, total_time, False

            candidate_path, ok = self.rewire_to_goal(
                rewired_path, curr_goal, n_wps=20, timeout=min(1.0, remaining)
            )

            if not ok or candidate_path is None:
                continue

            final_path = candidate_path
            success = True
            break

        fix_end = time.perf_counter()
        total_time = nn_time + (fix_end - fix_start)

        if not success:
            return None, total_time, False

        return final_path, total_time, True


def build_env_and_robot(env_name, robot_name, visualize=True):
    if env_name == "table":
        env = TableEnv(robot_name, no_sv=True)
    elif env_name == "box":
        env = BoxEnv(robot_name, no_sv=True)
    elif env_name == "cage":
        env = CageEnv(robot_name, no_sv=True)
    elif env_name == "shelf":
        env = ShelfEnv(robot_name, no_sv=True)
    elif env_name == "free":
        env = FreeEnv(robot_name, no_sv=True)
    else:
        raise ValueError(f"Invalid environment: {env_name}")

    model, data = env.model, env.data

    if robot_name == "panda":
        robot = Panda(model, data, visualize)
    elif robot_name == "ur10":
        robot = UR10(model, data, visualize)
    elif robot_name == "fetch":
        robot = FetchArm(model, data, visualize)
    else:
        raise ValueError(f"Invalid robot: {robot_name}")

    robot.teleport_base(pos=env.robot_pos, quat=env.robot_quat)
    return env, robot


def load_adapter_data(
    folder, ik, planner, adaptation, n_neighbors, env_name, robot_name
):
    if adaptation == "dmp" and env_name == "real" and robot_name == "ur10":
        suffix = f"{ik}_{planner}_{adaptation}_100"
    else:
        suffix = f"{ik}_{planner}_{adaptation}_{n_neighbors}"

    root_path = f"{folder}/root_paths_{suffix}.pkl"
    map_path = f"{folder}/key_to_root_{suffix}.pkl"

    if not os.path.exists(root_path) or not os.path.exists(map_path):
        raise FileNotFoundError(
            f"Missing files for {adaptation}: {root_path} / {map_path}"
        )

    root_data = pickle.load(open(root_path, "rb"))
    map_data = pickle.load(open(map_path, "rb"))
    return root_data, map_data


def set_camera(robot):
    robot.viewer.cam.lookat[:] = [0.25, -0.25, 0.5]
    robot.viewer.cam.distance = 1.75
    robot.viewer.cam.azimuth = 120
    robot.viewer.cam.elevation = -20


def sync_pause(robot, seconds=0.02):
    robot.viewer.sync()
    time.sleep(seconds)


def play_path(robot, path, dt=0.03):
    if path is None or len(path) == 0:
        print("Path is empty.")
        return

    for q in path:
        robot.set_joint_qpos(q)
        robot.viewer.sync()
        time.sleep(dt)


def sample_pose_supported_by_all(
    env,
    robot,
    solved_task_paths_keys,
    base_indexer,
    adapter_indexers,
):
    while True:
        key = solved_task_paths_keys[
            np.random.randint(0, len(solved_task_paths_keys))
        ]
        sample = [float(np.random.uniform(lo, hi)) for lo, hi in key]

        base_key = base_indexer.query_point(sample)
        if base_key is None:
            continue

        all_ok = True
        recovered = {}
        for name, indexer in adapter_indexers.items():
            k = indexer.query_point(sample)
            if k is None:
                all_ok = False
                break
            recovered[name] = k

        if not all_ok:
            continue

        env.move_cube_object(sample)
        robot.viewer.sync()
        return sample, base_key, recovered


def main(args):
    folder = get_data_folder(args.env, args.robot)

    task_set = pickle.load(open(f"{folder}/task_set.pkl", "rb"))

    d_name = f"{folder}/task_paths_data_{args.ik}_{args.planner}.npy"
    data = np.load(d_name, allow_pickle=True)
    k_name = f"{folder}/task_paths_keys_{args.ik}_{args.planner}.pkl"
    keys = pickle.load(open(k_name, "rb"))
    task_paths = {key: datum for key, datum in zip(keys, data)}

    solved_task_paths = {
        k: v for k, v in task_paths.items() if v is not None and len(v) > 0
    }
    solved_task_paths_keys = list(solved_task_paths.keys())

    coad_full_paths = solved_task_paths
    coad_full_indexer = BoxGrid(coad_full_paths)

    print(f"Generated tasks: {len(task_set)}")
    print(f"Solved planning tasks: {len(solved_task_paths_keys)}")

    print(f"Generated tasks: {len(task_set)}")
    print(f"Solved planning tasks: {len(solved_task_paths_keys)}")

    env, robot = build_env_and_robot(args.env, args.robot, visualize=True)
    set_camera(robot)

    model, mujoco_data = robot.model, robot.data
    home_qpos = robot.get_joint_qpos().copy()

    ik_solver = get_ik_solver(robot, env_collision_geoms=env.collision_geoms)

    method_specs = {
        "linear": {
            "file_tag": "grr",
            "adapter": LinearAdapter(robot, ik_solver),
        },
        "trajopt": {
            "file_tag": "opt",
            "adapter": TrajOptAdapter(robot, ik_solver),
        },
        "dmp": {
            "file_tag": "dmp",
            "adapter": DMPAdapter(robot, ik_solver),
        },
    }

    root_paths = {}
    key_to_roots = {}
    adapter_indexers = {}

    for method_name, spec in method_specs.items():
        root_data, map_data = load_adapter_data(
            folder,
            args.ik,
            args.planner,
            spec["file_tag"],
            args.n_neighbors,
            args.env,
            args.robot,
        )
        root_paths[method_name] = root_data
        key_to_roots[method_name] = map_data
        adapter_indexers[method_name] = BoxGrid(map_data)

        print(
            f"{method_name} ({spec['file_tag']} files): {len(root_data)} root paths"
        )
    base_indexer = BoxGrid(solved_task_paths)

    if isinstance(env, ShelfEnv) and isinstance(robot, FetchArm):
        ompl_planner = OMPLPlanner(robot, mujoco_data, rrtc_range=0.1)
    elif isinstance(env, CageEnv) and isinstance(robot, FetchArm):
        ompl_planner = OMPLPlanner(robot, mujoco_data, rrtc_range=0.1)
    else:
        ompl_planner = OMPLPlanner(robot, mujoco_data)

    print("\nBuilding library baseline...")
    library = Library(
        N=len(solved_task_paths_keys),
        env=env,
        robot=robot,
        home_qpos=home_qpos,
        key_to_root=key_to_roots["linear"],
        solved_keys=solved_task_paths_keys,
        task_paths=solved_task_paths,
        data=mujoco_data,
    )

    try:
        while True:
            print("\n=== RANDOM OBJECT POSE SAMPLING ===")
            print("Press Enter to keep cycling.")
            print("Type anything + Enter to lock the current object pose.\n")

            locked = False
            locked_sample = None
            locked_base_key = None
            locked_adapter_keys = None

            while not locked:
                sample, base_key, recovered = sample_pose_supported_by_all(
                    env=env,
                    robot=robot,
                    solved_task_paths_keys=solved_task_paths_keys,
                    base_indexer=base_indexer,
                    adapter_indexers=adapter_indexers,
                )

                print(f"Current sample: {np.array(sample)}")
                resp = input("Lock this pose? ").strip()

                if resp != "":
                    locked = True
                    locked_sample = sample
                    locked_base_key = base_key
                    locked_adapter_keys = recovered

            print("\nLocked object pose:")
            print(np.array(locked_sample))

            key_goal = np.asarray(
                solved_task_paths[locked_base_key][-1], dtype=float
            )

            print("\nShowing goal configuration for this object pose...")
            robot.set_joint_qpos(home_qpos)
            sync_pause(robot, 0.3)
            robot.set_joint_qpos(key_goal)
            sync_pause(robot, 1.0)

            print("Goal configuration q:")
            print(key_goal)

            resp = input(
                "\nType anything + Enter to confirm this pose and proceed to method visualization: "
            ).strip()
            if resp == "":
                print("Pose not confirmed. Returning to random sampling.")
                robot.set_joint_qpos(home_qpos)
                sync_pause(robot, 0.3)
                continue

            method_order = [
                "coad_full",
                "linear",
                "trajopt",
                "dmp",
                "rrtconnect",
                "library",
            ]
            # method_order = ["dmp"]
            method_order = [
                "dmp",
                "dmp",
                "coad_full",
                "linear",
                "trajopt",
                "rrtconnect",
                "library",
            ]

            for method in method_order:
                print(
                    f"\n==================== {method.upper()} ===================="
                )

                env.move_cube_object(locked_sample)
                robot.set_joint_qpos(home_qpos)
                sync_pause(robot, 0.5)

                path = None
                solve_time = None
                success = False

                if method == "coad_full":
                    recovered_key = coad_full_indexer.query_point(
                        locked_sample
                    )
                    if recovered_key is None:
                        path = None
                        solve_time = 0.0
                        success = False
                    else:
                        t0 = time.perf_counter()
                        path = coad_full_paths[recovered_key]
                        t1 = time.perf_counter()

                        solve_time = t1 - t0
                        success = path is not None and len(path) > 0

                        print(f"Recovered key: {recovered_key}")
                        print(f"goal q: {np.asarray(path[-1])}")

                elif method in ["linear", "trajopt", "dmp"]:
                    recovered_key = locked_adapter_keys[method]
                    root_id, curr_goal = key_to_roots[method][recovered_key]
                    curr_root = root_paths[method][root_id]

                    t0 = time.perf_counter()
                    path = method_specs[method]["adapter"].adapt(
                        curr_root, curr_goal
                    )
                    t1 = time.perf_counter()

                    solve_time = t1 - t0
                    success = path is not None and len(path) > 0

                    print(f"Recovered key: {recovered_key}")
                    print(f"root_id: {root_id}")
                    print(f"goal q: {np.asarray(curr_goal)}")

                elif method == "rrtconnect":
                    t0 = time.perf_counter()
                    path, total_time, planning_time = ompl_planner.plan(
                        start=home_qpos,
                        goal=key_goal,
                        timeout=args.rrt_timeout,
                        smooth_path=False,
                        num_waypoints=200,
                        benchmark=True,
                    )
                    t1 = time.perf_counter()

                    solve_time = (
                        planning_time
                        if planning_time is not None
                        else (t1 - t0)
                    )
                    success = path is not None and len(path) > 0
                    print(f"goal q: {key_goal}")

                elif method == "library":
                    t0 = time.perf_counter()
                    path, library_time, lib_success = library.solve(
                        locked_sample,
                        k=args.library_k,
                        timeout=args.library_timeout,
                    )
                    t1 = time.perf_counter()

                    solve_time = (
                        library_time if library_time is not None else (t1 - t0)
                    )
                    success = (
                        lib_success and path is not None and len(path) > 0
                    )

                    print(f"goal q: {key_goal}")

                if not success:
                    print(f"{method} failed.")
                    input("Press Enter to continue to the next method...")
                    robot.set_joint_qpos(home_qpos)
                    sync_pause(robot, 0.3)
                    continue

                path = np.asarray(path)
                print(f"{method} success.")
                print(f"time: {solve_time:.6f} s")
                print(f"num waypoints: {len(path)}")
                print(f"path length: {traj_len(path):.6f}")

                input("Press Enter to begin visualizing the path...")
                play_path(robot, path, dt=args.playback_dt)
                input("Trajectory complete. Press Enter for next method...")

                robot.set_joint_qpos(home_qpos)
                sync_pause(robot, 0.5)

            print("\nAll methods completed for this locked pose.")
            again = input(
                "Press Enter to sample a new pose, or type anything to quit: "
            ).strip()
            if again != "":
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user.")


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--env",
        choices=["table", "box", "cage", "shelf", "free"],
        default="shelf",
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
        choices=["RRTConnect", "PRMstar"],
        default="RRTConnect",
    )
    parser.add_argument("--n_neighbors", type=int, default=1000)

    parser.add_argument("--rrt_timeout", type=float, default=3.0)
    parser.add_argument("--library_timeout", type=float, default=3.0)
    parser.add_argument("--library_k", type=int, default=5)
    parser.add_argument("--playback_dt", type=float, default=0.03)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
