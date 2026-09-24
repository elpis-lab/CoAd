"""Lightning repair and shared experience-library indexing."""

import time
import pickletools
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm
from coad.env import MujocoEnv, ShelfEnv, CageEnv
from coad.robot import MujocoRobot, FetchArm
from coad.planning import OMPLPlanner
from experiments.indexing import BoxGrid, sample_from_key


class Library:
    def __init__(
        self,
        N: int,
        env: MujocoEnv,
        robot: MujocoRobot,
        home_qpos,
        key_to_root,
        root_paths,
        solved_keys,
        data,
    ):
        """Build nearest-neighbor Lightning from adaptation files only.

        This avoids loading task_paths_data_*.npy / task_paths_keys_*.pkl.
        Each stored path is reconstructed as:
            root_paths[root_id] + [goal_q]
        where (root_id, goal_q) comes from key_to_root.
        """
        self.key_to_root = key_to_root
        self.root_paths = root_paths
        self.indexer = BoxGrid(key_to_root)
        self.robot = robot
        self.home_qpos = np.asarray(home_qpos, dtype=float).copy()

        if isinstance(env, ShelfEnv) and isinstance(robot, FetchArm):
            self.ompl_planner = OMPLPlanner(robot, data, rrtc_range=0.1)
        elif isinstance(env, CageEnv) and isinstance(robot, FetchArm):
            self.ompl_planner = OMPLPlanner(robot, data, rrtc_range=0.1)
        else:
            self.ompl_planner = OMPLPlanner(robot, data)
        # self.ompl_planner = OMPLPlanner(robot, data)
        # solved_key_list = list(solved_keys.keys())
        solved_key_list = list(solved_keys)

        if N <= 0 or not solved_key_list:
            raise ValueError("Library size and usable keys must be nonempty")
        self.library = {}
        pbar_library = tqdm(total=N, desc="Building Lightning", leave=True)

        iters = 0
        while len(self.library) < N:
            iters += 1
            if iters > max(1000, N * 100):
                raise RuntimeError("Cannot sample enough distinct usable library poses")

            random_key = solved_key_list[np.random.randint(0, len(solved_key_list))]
            # sample = [np.random.uniform(lo, hi) for lo, hi in random_key]
            sample = sample_from_key(random_key)

            recovered_key = self.indexer.query_point(sample)
            if recovered_key is None:
                # print("Failed to find sample", flush=True)
                continue
            root_id, curr_goal = key_to_root[recovered_key]
            if root_id is None:
                continue
            root_path = root_paths[root_id]
            if root_path is None or len(root_path) == 0:
                continue

            path = list(root_path.copy())
            path.append(curr_goal)

            # path, total_time, planning_time = self.ompl_planner.plan(
            #     start=home_qpos,
            #     goal=curr_goal,
            #     timeout=5.0,
            #     num_waypoints=200,
            #     benchmark=True,
            # )

            if tuple(sample) not in self.library:
                # print(tuple(sample))
                self.library[tuple(sample)] = (curr_goal, path)
                pbar_library.update(1)

        pbar_library.close()
        self.lib_index = self.build_library_index(w_yaw=1.0, w_door=1.0)

    def build_library_index(
        self,
        z_tol=1e-6,
        w_yaw=1.0,
        w_door=1.0,
    ):
        """
        Library keys may be:

            Standard:
                (x, y, z, yaw)

            Microwave:
                (x, y, z, yaw, door)

            AllStable:
                (face, x, y, z, yaw)

        Standard and microwave keys are grouped by discrete z.

        AllStable keys are grouped by:
            (face, discrete z)
        """
        raw_keys = list(self.library.keys())

        if not raw_keys:
            raise ValueError("Cannot build an index from an empty library")

        first_key = raw_keys[0]

        has_face = len(first_key) == 5 and isinstance(first_key[0], str)

        if has_face:
            # AllStable:
            #     (face, x, y, z, yaw)
            faces = np.asarray(
                [key[0] for key in raw_keys],
                dtype=object,
            )

            keys = np.asarray(
                [key[1:] for key in raw_keys],
                dtype=np.float64,
            )

            ndim = 5
            has_door = False
        else:
            keys = np.asarray(
                raw_keys,
                dtype=np.float64,
            )

            if keys.ndim != 2 or keys.shape[1] not in (4, 5):
                raise ValueError(
                    f"Expected keys shape (N,4) or (N,5), " f"got {keys.shape}"
                )

            ndim = keys.shape[1]
            has_door = ndim == 5
            faces = None

        # Numeric keys must always contain:
        #     x, y, z, yaw
        #
        # and optionally:
        #     door
        if keys.ndim != 2 or keys.shape[1] not in (4, 5):
            raise ValueError(
                f"Expected numeric keys shape (N,4) or (N,5), " f"got {keys.shape}"
            )

        z_vals = np.sort(np.unique(keys[:, 2]))

        trees = {}
        key_lists = {}

        yaw_scale = np.sqrt(w_yaw)
        door_scale = np.sqrt(w_door)

        if has_face:
            face_values = tuple(
                face for face in ("xy", "yz", "zx") if np.any(faces == face)
            )

            for face in face_values:
                face_mask = faces == face

                for z0 in z_vals:
                    mask = face_mask & (np.abs(keys[:, 2] - z0) <= z_tol)

                    kz = keys[mask]

                    if kz.size == 0:
                        continue

                    # [x, y, cos(yaw), sin(yaw)]
                    feats = np.column_stack(
                        [
                            kz[:, 0],
                            kz[:, 1],
                            yaw_scale * np.cos(kz[:, 3]),
                            yaw_scale * np.sin(kz[:, 3]),
                        ]
                    )

                    group = (face, float(z0))

                    trees[group] = cKDTree(feats)

                    # Restore the original AllStable key format.
                    key_lists[group] = [
                        (
                            face,
                            float(row[0]),
                            float(row[1]),
                            float(row[2]),
                            float(row[3]),
                        )
                        for row in kz
                    ]

        else:
            face_values = None

            for z0 in z_vals:
                mask = np.abs(keys[:, 2] - z0) <= z_tol
                kz = keys[mask]

                if kz.size == 0:
                    continue

                if has_door:
                    # [x, y, cos(yaw), sin(yaw), door]
                    feats = np.column_stack(
                        [
                            kz[:, 0],
                            kz[:, 1],
                            yaw_scale * np.cos(kz[:, 3]),
                            yaw_scale * np.sin(kz[:, 3]),
                            door_scale * kz[:, 4],
                        ]
                    )
                else:
                    # [x, y, cos(yaw), sin(yaw)]
                    feats = np.column_stack(
                        [
                            kz[:, 0],
                            kz[:, 1],
                            yaw_scale * np.cos(kz[:, 3]),
                            yaw_scale * np.sin(kz[:, 3]),
                        ]
                    )

                group = float(z0)

                trees[group] = cKDTree(feats)

                key_lists[group] = [tuple(float(value) for value in row) for row in kz]

        return {
            "ndim": ndim,
            "has_face": has_face,
            "has_door": has_door,
            "face_values": face_values,
            "z_vals": z_vals,
            "trees": trees,
            "key_lists": key_lists,
            "yaw_scale": yaw_scale,
            "door_scale": door_scale,
            "z_tol": z_tol,
        }

    def query_library_nn(self, lib_index, sample, n=1):
        """
        Query nearest library entries.

        Supported sample formats:

            Standard:
                [x, y, z, yaw]

            Microwave:
                [x, y, z, yaw, door]

            AllStable:
                [face, x, y, z, yaw]
        """
        has_face = lib_index["has_face"]
        has_door = lib_index["has_door"]

        if has_face:
            if len(sample) != 5 or not isinstance(sample[0], str):
                raise ValueError(
                    "AllStable query expects " "[face, x, y, z, yaw], " f"got {sample}"
                )

            face = sample[0]

            if face not in lib_index["face_values"]:
                return []

            numeric_sample = np.asarray(
                sample[1:],
                dtype=np.float64,
            )
        else:
            face = None

            numeric_sample = np.asarray(
                sample,
                dtype=np.float64,
            )

        expected_numeric_dim = 5 if has_door else 4

        if numeric_sample.ndim != 1 or numeric_sample.shape[0] != expected_numeric_dim:
            raise ValueError(
                f"Expected numeric sample with "
                f"{expected_numeric_dim} values, "
                f"got shape {numeric_sample.shape}: {sample}"
            )

        x = float(numeric_sample[0])
        y = float(numeric_sample[1])
        z = float(numeric_sample[2])
        yaw = float(numeric_sample[3])

        z_vals = lib_index["z_vals"]

        if z_vals.size == 0:
            return []

        nearest_z = float(z_vals[np.argmin(np.abs(z_vals - z))])

        if has_face:
            group = (face, nearest_z)
        else:
            group = nearest_z

        tree = lib_index["trees"].get(group)
        key_list = lib_index["key_lists"].get(group)

        if tree is None or key_list is None:
            return []

        yaw_scale = lib_index["yaw_scale"]

        if has_door:
            door = float(numeric_sample[4])
            door_scale = lib_index["door_scale"]

            query_feature = np.array(
                [
                    x,
                    y,
                    yaw_scale * np.cos(yaw),
                    yaw_scale * np.sin(yaw),
                    door_scale * door,
                ],
                dtype=np.float64,
            )
        else:
            query_feature = np.array(
                [
                    x,
                    y,
                    yaw_scale * np.cos(yaw),
                    yaw_scale * np.sin(yaw),
                ],
                dtype=np.float64,
            )

        k = min(int(n), len(key_list))

        if k <= 0:
            return []

        distances, indices = tree.query(
            query_feature,
            k=k,
        )

        distances = np.atleast_1d(distances)
        indices = np.atleast_1d(indices)

        results = []

        for dist, idx in zip(distances, indices):
            idx = int(idx)

            if idx < 0 or idx >= len(key_list):
                continue

            key = key_list[idx]

            results.append(
                (
                    key,
                    self.library[key],
                    float(dist),
                )
            )

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

        # mark everything within +/- b of each collision as collision
        for i in coll:
            lo = max(0, i - b)
            hi = min(T, i + b + 1)  # +1 because slice end is exclusive
            buffered[lo:hi] = False

        return buffered

    def rewire_segments(
        self, path, validity_map, timeout=2.0, num_waypoints=20, max_repairs=20
    ):
        path = np.asarray(path, dtype=np.float64)
        valid = np.asarray(validity_map, dtype=bool)
        out = path.copy()

        deadline = time.perf_counter() + timeout
        prev_signature = None
        repairs = 0

        while True:
            invalid = ~valid
            n_invalid_before = int(invalid.sum())
            if n_invalid_before == 0:
                return out, True

            if repairs >= max_repairs or time.perf_counter() >= deadline:
                # prevent runaway time
                return None, False

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
                    continue  # edge run

                # detect "same segment again" (prevents looping on one stubborn gap)
                sig = (prev, nxt, len(out))
                if sig == prev_signature:
                    # give up on this neighbor path
                    return None, False
                prev_signature = sig

                q0, q1 = out[prev], out[nxt]

                t0 = time.perf_counter()
                rewired_segment, _, _ = self.ompl_planner.plan(
                    start=q0,
                    goal=q1,
                    timeout=max(0.0, deadline - time.perf_counter()),
                    num_waypoints=num_waypoints,
                    benchmark=True,
                )
                t1 = time.perf_counter()
                # tqdm.write(f"rewire [{prev}->{nxt}] plan_time={t1-t0:.2f}s invalid={n_invalid_before}")

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

                # full recompute (expensive but correct with variable length)
                valid = self.check_path_collision(out)

                n_invalid_after = int((~valid).sum())
                if n_invalid_after >= n_invalid_before:
                    # no improvement -> stop before infinite repair loop
                    return None, False

                repairs += 1
                rewired_any = True
                break

            if not rewired_any:
                return None, False

    def rewire_to_goal(self, path, goal, n_wps=20, timeout=1.0):
        path = np.asarray(path, dtype=np.float64)
        T = len(path)

        # ---- find last valid waypoint ----
        start_idx = None
        for i in range(T - 1, -1, -1):
            self.robot.set_joint_qpos(path[i])
            if not self.robot.in_contact():
                start_idx = i
                break

        if start_idx is None:
            return None, False  # entire path in collision

        q_start = path[start_idx]
        q_goal = np.asarray(goal, dtype=np.float64)

        # ---- straight-line interpolation ----
        t = np.linspace(0.0, 1.0, n_wps)[:, None]
        rewired_segment = (1.0 - t) * q_start + t * q_goal

        interpolation_valid = True
        for wp in rewired_segment:
            self.robot.set_joint_qpos(wp)
            if self.robot.in_contact():
                interpolation_valid = False
                break

        # ---- fallback to RRTConnect if needed ----
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

        # ---- splice new tail ----
        # keep original path up to start_idx
        # append rewired segment excluding duplicate start
        new_path = np.vstack([path[: start_idx + 1], rewired_segment[1:]])

        return new_path, True

    def solve(self, sample, k=5, timeout=3.0, goal=None, start=None):
        if start is None:
            start = getattr(self, "home_qpos", None)
        nn_query_start = time.perf_counter()
        nn_results = self.query_library_nn(self.lib_index, sample, n=k)
        nn_query_end = time.perf_counter()
        nn_time = nn_query_end - nn_query_start

        if goal is None:
            recovered_key = self.indexer.query_point(sample)
            if recovered_key is None:
                return None, nn_time, False
            _, curr_goal = self.key_to_root[recovered_key]
        else:
            curr_goal = np.asarray(goal, dtype=float)

        fix_start = time.perf_counter()
        fix_end = fix_start

        final_path = None
        success = False

        for (
            neighbor_key,
            (neighbor_goal, neighbor_path),
            neighbor_dist,
        ) in nn_results:

            # ---- Check global timeout BEFORE heavy work ----
            elapsed_fix = time.perf_counter() - fix_start
            if elapsed_fix > timeout:
                total_time = nn_time + elapsed_fix
                return None, total_time, False

            # 1) collision map + buffer
            waypoints_valid = self.collision_buffer(
                self.check_path_collision(neighbor_path), b=0
            )

            # ---- Remaining budget for this neighbor ----
            remaining = timeout - (time.perf_counter() - fix_start)
            if remaining <= 0:
                total_time = nn_time + (time.perf_counter() - fix_start)
                return None, total_time, False

            # 2) rewire internal segments
            rewired_path, ok = self.rewire_segments(
                neighbor_path,
                waypoints_valid,
                timeout=min(2.0, remaining),  # cap by remaining budget
                num_waypoints=20,
            )

            if not ok or rewired_path is None:
                continue

            # Experience starts can differ from the current query start.
            # Reuse tail repair in reverse, charging the same shared budget.
            if start is not None and not np.allclose(rewired_path[0], start, atol=1e-6):
                remaining = timeout - (time.perf_counter() - fix_start)
                if remaining <= 0:
                    return (
                        None,
                        nn_time + (time.perf_counter() - fix_start),
                        False,
                    )
                reversed_path, ok = self.rewire_to_goal(
                    np.asarray(rewired_path)[::-1],
                    start,
                    n_wps=20,
                    timeout=min(1.0, remaining),
                )
                if not ok or reversed_path is None:
                    continue
                rewired_path = np.asarray(reversed_path)[::-1]

            # ---- Recompute remaining budget ----
            remaining = timeout - (time.perf_counter() - fix_start)
            if remaining <= 0:
                total_time = nn_time + (time.perf_counter() - fix_start)
                return None, total_time, False

            # 3) rewire tail to goal
            candidate_path, ok = self.rewire_to_goal(
                rewired_path,
                curr_goal,
                n_wps=20,
                timeout=min(1.0, remaining),  # cap by remaining budget
            )

            if not ok or candidate_path is None:
                continue

            final_path = candidate_path
            success = True
            break

        fix_end = time.perf_counter()
        fix_time = fix_end - fix_start
        total_time = nn_time + fix_time

        if not success:
            return None, total_time, False

        return final_path, total_time, True

    def wrap_pi(self, a):
        return (a + np.pi) % (2 * np.pi) - np.pi


def count_pickled_roots(path):
    """Count entries in a root dict/list without loading trajectory byte payloads.

    Simulate pickle's stack only; no globals are imported or executed. This is for
    the generated root libraries, whose dictionary keys are unique root IDs.
    """
    opcodes = {ord(op.code): op for op in pickletools.opcodes}
    stack = []
    mark = object()
    value = object()
    count = 0
    root_kind = None
    with open(path, "rb") as stream:
        while code := stream.read(1):
            op = opcodes[code[0]]
            if op.arg:
                size = op.arg.n
                if size >= 0:
                    stream.read(size)
                elif size == -1:
                    stream.readline()
                    if op.name in ("GLOBAL", "INST"):
                        stream.readline()
                else:
                    prefix = {-2: 1, -3: 4, -4: 4, -5: 8}[size]
                    payload_size = int.from_bytes(stream.read(prefix), "little")
                    stream.seek(payload_size, 1)
            if op.name in ("PROTO", "FRAME"):
                continue
            if root_kind is None:
                if op.name not in ("EMPTY_DICT", "EMPTY_LIST"):
                    raise ValueError(f"{path}: expected a generated root dict/list")
                root_kind = op.name
            if op.name == "STOP":
                return count
            if op.name == "MARK":
                stack.append(mark)
                continue
            if any(item.name == "mark" for item in op.stack_before):
                index = len(stack) - 1 - stack[::-1].index(mark)
                if index == 1:
                    if op.name == "SETITEMS" and root_kind == "EMPTY_DICT":
                        count += (len(stack) - index - 1) // 2
                    elif op.name == "APPENDS" and root_kind == "EMPTY_LIST":
                        count += len(stack) - index - 1
                prefix = next(
                    i for i, item in enumerate(op.stack_before) if item.name == "mark"
                )
                del stack[index - prefix :]
            else:
                if (
                    op.name == "SETITEM"
                    and len(stack) == 3
                    and root_kind == "EMPTY_DICT"
                ):
                    count += 1
                if (
                    op.name == "APPEND"
                    and len(stack) == 2
                    and root_kind == "EMPTY_LIST"
                ):
                    count += 1
                if op.stack_before:
                    del stack[-len(op.stack_before) :]
            stack.extend(value for _ in op.stack_after)
    raise ValueError(f"{path}: missing pickle STOP")


class ReferencedExperienceLibrary:
    """Store all N poses, but reference root paths instead of duplicating arrays."""

    build_library_index = Library.build_library_index
    query_library_nn = Library.query_library_nn
    wrap_pi = Library.wrap_pi

    def __init__(self, count, key_map, roots, usable):
        if count <= 0 or not usable:
            raise ValueError("Library size and usable keys must be nonempty")
        self.roots = roots
        self.library = {}
        attempts = 0
        while len(self.library) < count:
            attempts += 1
            if attempts > max(1000, count * 100):
                raise RuntimeError("Cannot sample enough distinct usable library poses")
            key = usable[np.random.randint(len(usable))]
            pose = tuple(sample_from_key(key))
            root_id, goal = key_map[key]
            self.library[pose] = (goal, root_id)
        self.lib_index = self.build_library_index(w_yaw=1.0, w_door=1.0)

    def get_path(self, pose):
        goal, root_id = self.library[pose]
        return np.vstack([self.roots[root_id], goal])
