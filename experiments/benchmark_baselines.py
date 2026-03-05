import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pickle
import mujoco
import time
import argparse
from scipy.spatial import cKDTree

from tqdm import tqdm
from plan_load.env import MujocoEnv
from plan_load.robot import MujocoRobot
from plan_load.mink_ik import get_ik_solver

from plan_load.adaptation import LinearAdapter, GRRAdapter
from plan_load.adaptation import DMPAdapter, TrajOptAdapter

# from plan_load.task_space import deep_tuple
from experiments.evaluate import traj_len
from plan_load.utils import set_seed, load_env_and_robot, get_data_folder
from plan_load.planning import OMPLPlanner, euclidean_path_length

from plan_load.env import FreeEnv, CageEnv, BoxEnv, TableEnv, ShelfEnv
from plan_load.robot import Panda, UR10, FetchArm


folder1 = "dataset/top_naive"
folder2 = "dataset/top"

class BoxGrid:
    def __init__(self, keys_to_root, tol=1e-12):

        keys = np.asarray(list(keys_to_root.keys()), dtype=np.float64)
        self.keys_list = list(keys_to_root.keys())
        self.keys_arr = keys

        if keys.ndim != 3 or keys.shape[1:] != (4, 2):
            raise ValueError(f"Expected keys shape (N,4,2), got {keys.shape}")

        mins = keys[:, :, 0].copy()
        maxs = keys[:, :, 1].copy()

        # Wrap yaw into [-pi, pi)
        mins[:, 3] = self._wrap_pi(mins[:, 3])
        maxs[:, 3] = self._wrap_pi(maxs[:, 3])

        # ---------- X/Y: USE BIN STARTS DIRECTLY ----------
        self.x_mins = np.sort(np.unique(mins[:, 0]))
        self.y_mins = np.sort(np.unique(mins[:, 1]))

        # Build fast lookup maps for construction
        self._x_to_ix = {float(v): i for i, v in enumerate(self.x_mins)}
        self._y_to_iy = {float(v): i for i, v in enumerate(self.y_mins)}

        self.nx = len(self.x_mins)
        self.ny = len(self.y_mins)

        # ---------- Z: discrete levels ----------
        self.z_values = np.sort(np.unique(mins[:, 2]))
        self.nz = len(self.z_values)

        # ---------- YAW: periodic bins ----------
        yaw_mins = np.sort(np.unique(mins[:, 3]))
        self.nyaw = len(yaw_mins) if yaw_mins.size else 1
        self.yaw0 = float(yaw_mins[0]) if yaw_mins.size else 0.0

        def spacing(vals, default=1.0):
            vals = np.sort(np.unique(vals))
            diffs = np.diff(vals)
            diffs = diffs[diffs > tol]
            return float(diffs.min()) if diffs.size else float(default)

        self.dyaw = spacing(yaw_mins)

        # ---------- BUILD INDEX ----------
        self.index = {}

        for bin_idx, key in enumerate(self.keys_list):

            x_min, y_min, z_val, yaw_min = key[0][0], key[1][0], key[2][0], key[3][0]

            # X/Y via direct lookup
            ix = self._x_to_ix[float(x_min)]
            iy = self._y_to_iy[float(y_min)]

            # Z
            iz = int(np.argmin(np.abs(self.z_values - z_val)))

            # Yaw
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

        # clamp to valid range
        ix = min(max(ix, 0), len(self.x_mins) - 1)
        iy = min(max(iy, 0), len(self.y_mins) - 1)

        if ix < 0 or ix >= self.nx or iy < 0 or iy >= self.ny:
            return None

        iz = int(np.argmin(np.abs(self.z_values - z)))  # discrete levels

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

def deep_tuple(x):
    # NumPy array
    if isinstance(x, np.ndarray):
        return tuple(deep_tuple(i) for i in x)
    # Python list or tuple
    elif isinstance(x, (list, tuple)):
        return tuple(deep_tuple(i) for i in x)
    # Base case (scalar)
    else:
        return x


def get_avg_path_length(root_path, key_map):
    lengths = np.zeros(len(key_map), dtype=float)
    for i, key in enumerate(key_map):
        key = deep_tuple(key)
        root_id, goal_q = key_map[key]
        path = list(root_path[root_id].copy())
        path.append(goal_q)

        lengths[i] = traj_len(path)
    return np.mean(lengths)

class Library():
    def __init__(
        self,
        N: int,
        env: MujocoEnv,
        robot: MujocoRobot,
        home_qpos,
        key_to_root,
        solved_keys,
        task_paths,
        data
    ):
        """Build library"""
        self.key_to_root = key_to_root
        self.task_paths = task_paths
        #self.indexer = BoxGrid(key_to_root)
        self.indexer = BoxGrid(task_paths)
        self.robot = robot
        
        if isinstance(env, ShelfEnv) and isinstance(robot, FetchArm):
            self.ompl_planner = OMPLPlanner(robot, data, rrtc_range=0.1)
        elif isinstance(env, CageEnv) and isinstance(robot, FetchArm):
            self.ompl_planner = OMPLPlanner(robot, data, rrtc_range=0.1)
        else:
            self.ompl_planner = OMPLPlanner(robot, data)    
        #self.ompl_planner = OMPLPlanner(robot, data)
        #solved_key_list = list(solved_keys.keys())
        solved_key_list = list(solved_keys)

        self.library = {}
        pbar_library = tqdm(total=N, desc="Building library", leave=True)

        iters = 0
        while len(self.library) <= N:
            iters += 1
            
            random_key = solved_key_list[np.random.randint(0, len(solved_key_list))]
            sample = [np.random.uniform(lo, hi) for lo, hi in random_key]

            recovered_key = self.indexer.query_point(sample)
            if recovered_key is None:
                #print("Failed to find sample", flush=True)
                continue
            #_, curr_goal = key_to_root[recovered_key]
            path = task_paths[recovered_key]
            if path is None or len(path)==0:
                #print("none path", flush=True)
                continue
            curr_goal = path[-1]

            # path, total_time, planning_time = self.ompl_planner.plan(
            #     start=home_qpos,
            #     goal=curr_goal,
            #     timeout=5.0,
            #     num_waypoints=200,
            #     benchmark=True,
            # )
            
            if tuple(sample) not in self.library:
                #print(tuple(sample))
                self.library[tuple(sample)] = (curr_goal, path)
                pbar_library.update(1)

        
        self.lib_index = self.build_library_index(w_yaw=1.0)
        #return self.lib_index

    def build_library_index(self, z_tol=1e-6, w_yaw=1.0):
        """
        library: dict keyed by (x,y,z,yaw) -> (curr_goal, path)
        Returns an index object you can query quickly.
        """
        keys = np.asarray(list(self.library.keys()), dtype=np.float64)  # (N,4)
        if keys.ndim != 2 or keys.shape[1] != 4:
            raise ValueError(f"Expected keys shape (N,4), got {keys.shape}")

        # group by z (discrete)
        z_vals = np.unique(keys[:, 2])
        z_vals = np.sort(z_vals)

        trees = {}
        key_lists = {}

        # scaling for yaw embedding
        # Euclidean distance in (cos,sin) is in [0, 2]; w_yaw scales its influence.
        yaw_scale = np.sqrt(w_yaw)

        for z0 in z_vals:
            mask = np.abs(keys[:, 2] - z0) <= z_tol
            kz = keys[mask]                         # (Nz,4)
            if kz.size == 0:
                continue

            # Feature vector: [x, y, yaw_scale*cos(yaw), yaw_scale*sin(yaw)]
            feats = np.column_stack([
                kz[:, 0],
                kz[:, 1],
                yaw_scale * np.cos(kz[:, 3]),
                yaw_scale * np.sin(kz[:, 3]),
            ])

            trees[float(z0)] = cKDTree(feats)
            # store the exact tuple keys for retrieval (avoid float reconstruction issues)
            key_lists[float(z0)] = [tuple(row) for row in kz]

        return {
            "z_vals": z_vals,
            "trees": trees,
            "key_lists": key_lists,
            "yaw_scale": yaw_scale,
            "z_tol": z_tol,
        }

    # def query_library_nn(self, index, sample):
    #     """
    #     sample: [x,y,z,yaw]
    #     Returns: nearest_key, (curr_goal, path), distance
    #     """
    #     x, y, z, yaw = map(float, sample)

    #     # pick nearest z-slice
    #     z_vals = index["z_vals"]
    #     zi = int(np.argmin(np.abs(z_vals - z)))
    #     z0 = float(z_vals[zi])

    #     if abs(z0 - z) > index["z_tol"]:
    #         return None, None, np.inf

    #     tree = index["trees"].get(z0, None)
    #     if tree is None:
    #         return None, None, np.inf

    #     ys = index["yaw_scale"]
    #     q = np.array([x, y, ys * np.cos(yaw), ys * np.sin(yaw)], dtype=np.float64)

    #     dist, idx = tree.query(q, k=1)
    #     nearest_key = index["key_lists"][z0][int(idx)]
    #     return nearest_key, self.library[nearest_key], float(dist)

    def query_library_nn(self, index, sample, n=1):
        """
        sample: [x,y,z,yaw]
        n: number of nearest neighbors to return

        Returns:
            results: list of tuples
                [(key, (curr_goal, path), distance), ...]
            sorted from closest to farthest
        """
        x, y, z, yaw = map(float, sample)

        # ---- pick nearest z-slice ----
        z_vals = index["z_vals"]
        zi = int(np.argmin(np.abs(z_vals - z)))
        z0 = float(z_vals[zi])

        if abs(z0 - z) > index["z_tol"]:
            return []

        tree = index["trees"].get(z0, None)
        if tree is None:
            return []

        ys = index["yaw_scale"]
        q = np.array([x, y, ys * np.cos(yaw), ys * np.sin(yaw)], dtype=np.float64)

        # ---- Query k nearest ----
        # ensure n does not exceed available points
        num_points = len(index["key_lists"][z0])
        k = min(n, num_points)

        dists, idxs = tree.query(q, k=k)

        # If k == 1, tree.query returns scalars, so normalize to arrays
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

            # mark everything within +/- b of each collision as collision
            for i in coll:
                lo = max(0, i - b)
                hi = min(T, i + b + 1)   # +1 because slice end is exclusive
                buffered[lo:hi] = False

            return buffered

    def rewire_segments(self, path, validity_map, timeout=2.0, num_waypoints=20,
                    max_repairs=20):
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
                # prevent runaway time
                return out, True

            starts = np.flatnonzero(invalid & np.r_[True, ~invalid[:-1]])
            ends   = np.flatnonzero(invalid & np.r_[~invalid[1:], True])

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
                    start=q0, goal=q1,
                    timeout=timeout,
                    num_waypoints=num_waypoints,
                    benchmark=True,
                )
                t1 = time.perf_counter()
                # tqdm.write(f"rewire [{prev}->{nxt}] plan_time={t1-t0:.2f}s invalid={n_invalid_before}")

                if rewired_segment is None or len(rewired_segment)==0:
                    return None, False

                rewired_segment = np.asarray(rewired_segment, dtype=np.float64)
                mid = rewired_segment[1:-1] if rewired_segment.shape[0] >= 2 else rewired_segment

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
                return out, True

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

        if rewired_segment is None or len(rewired_segment)==0:
            return None, False

        rewired_segment = np.asarray(rewired_segment, dtype=np.float64)

        # ---- splice new tail ----
        # keep original path up to start_idx
        # append rewired segment excluding duplicate start
        new_path = np.vstack([
            path[:start_idx + 1],
            rewired_segment[1:]
        ])

        return new_path, True
        



    def solve(self, sample, k=5, timeout=3.0):
        nn_query_start = time.perf_counter()
        nn_results = self.query_library_nn(self.lib_index, sample, n=k)
        nn_query_end = time.perf_counter()
        nn_time = nn_query_end - nn_query_start

        recovered_key = self.indexer.query_point(sample)
        if recovered_key is None:
            #raise RuntimeError("Failed to find key")
            return None, nn_time, False
        #_, curr_goal = self.key_to_root[recovered_key]
        path = self.task_paths[recovered_key]
        curr_goal = path[-1]


        fix_start = time.perf_counter()
        fix_end = fix_start

        final_path = None
        success = False

        for neighbor_key, (neighbor_goal, neighbor_path), neighbor_dist in nn_results:

            # ---- Check global timeout BEFORE heavy work ----
            elapsed_fix = time.perf_counter() - fix_start
            if elapsed_fix > timeout:
                total_time = nn_time + elapsed_fix
                return None, total_time, False

            # 1) collision map + buffer
            waypoints_valid = self.collision_buffer(
                self.check_path_collision(neighbor_path),
                b=0
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
                timeout=min(2.0, remaining),   # cap by remaining budget
                num_waypoints=20
            )

            if not ok or rewired_path is None:
                continue

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
                timeout=min(1.0, remaining)   # cap by remaining budget
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
        return (a + np.pi) % (2*np.pi) - np.pi

def evaluate_graph(
    args,
    env: MujocoEnv,
    robot: MujocoRobot,
    folder,
    task_set,
    task_paths,
    adaptations,
    num_samples
):
    model, data = robot.model, robot.data
    home_qpos = robot.get_joint_qpos()
    ik_solver = get_ik_solver(robot, env_collision_geoms=env.collision_geoms)
    solved_task_paths_keys = [k for k, path in task_paths.items() if path is not None and len(path) > 0]
    solved_task_paths = {k: v for k, v in task_paths.items() if v is not None and len(v) > 0}

    print(f"Number of solved paths: {len(solved_task_paths_keys)}")

    adapters = []
    key_to_roots = []
    root_paths_list = []

    # Setup grids for base library and adaptations
    indexers = []
    indexers.append(BoxGrid(task_set))
    #indexers.append(BoxGrid(solved_task_paths))

    rrtc_success = []
    rrtc_times = []
    rrtc_lengths = []

    library_success = []
    library_lengths = []
    library_times = []

    adaptation_success = {}
    adaptation_times = {}
    adaptation_lengths = {}


    lib_sizes = {}
    lib_sizes['full'] = len(solved_task_paths_keys)

    for adaptation in adaptations:

        adaptation_success[f"{adaptation}"] = []
        adaptation_times[f"{adaptation}"] = []
        adaptation_lengths[f"{adaptation}"] = []

        suffix = f"{args.ik}_{args.planner}_{adaptation}_{args.n_neighbors}"
        root_path = f"{folder}/root_paths_{suffix}.pkl"
        map_path = f"{folder}/key_to_root_{suffix}.pkl"
        
        root_data = pickle.load(open(root_path, "rb"))
        map_data = pickle.load(open(map_path, "rb"))
        key_to_roots.append(map_data)
        root_paths_list.append(root_data)

        print(f"{adaptation} library size: {len(root_data)}")
        lib_sizes[f"{adaptation}"] = len(root_data)

        indexers.append(BoxGrid(map_data))

        if adaptation == "linear":
            adapter = LinearAdapter(robot, ik_solver)
        elif adaptation == "grr":
            adapter = GRRAdapter(robot, ik_solver)
        elif adaptation == "dmp":
            adapter = DMPAdapter(robot, ik_solver)
        elif adaptation == "opt":
            adapter = TrajOptAdapter(robot, ik_solver)
        else:
            raise ValueError(f"Invalid adaptation method: {adaptation}")
        adapters.append(adapter)


    rrtc_success = []
    rrtc_lengths = []
    rrtc_solve_times = []

    solved_keys = solved_task_paths_keys

    #indexer = BoxGrid(key_to_root)
    if isinstance(env, ShelfEnv) and isinstance(robot, FetchArm):
        ompl_planner = OMPLPlanner(robot, data, rrtc_range=0.1)
    elif isinstance(env, CageEnv) and isinstance(robot, FetchArm):
        ompl_planner = OMPLPlanner(robot, data, rrtc_range=0.1)
    else:
        ompl_planner = OMPLPlanner(robot, data) 
    #ompl_planner = OMPLPlanner(robot, data)

    # Building library baseline with N = full library size
    N = len(solved_task_paths_keys)
    print("\n=== Building library ===")
    library = Library(N, env, robot, home_qpos, key_to_roots[0], solved_keys, task_paths, data)

    print(f"\n=== Evaluating baselines over {num_samples} samples===")
    num_tested = 0
    
    #pbar = tqdm(enumerate(solved_keys), total=len(solved_keys))
    #for i, key in enumerate(solved_keys):
    pbar = tqdm(range(num_samples), total=num_samples)
    while (num_tested < num_samples):

        key_ind = np.random.randint(0, len(solved_keys))
        key = solved_keys[key_ind]

        sample = [] 
        for lo, hi in key:
            x = np.random.uniform(lo, hi)
            x = np.nextafter(x, lo)
            sample.append(x)

        env.move_cube_object(sample)
        recovered_key = indexers[0].query_point(sample)  # pick one
        #_, key_goal = key_to_roots[0][recovered_key]
        key_goal = solved_task_paths[recovered_key][-1]

        for adaptation_ind, adaptation in enumerate(adaptations):
            adapt_start = time.perf_counter()
            recovered_key_adapt = indexers[adaptation_ind+1].query_point(sample)
            if recovered_key_adapt is None:
                adaptation_success[adaptation].append(False)
                adapt_end = time.perf_counter()
                adaptation_times[adaptation].append(adapt_end - adapt_start)
                adaptation_lengths[adaptation].append(np.nan)
                continue
            root_id, curr_goal = key_to_roots[adaptation_ind][recovered_key_adapt]

            if key != recovered_key_adapt:
                print(f"Original key: {key}")
                # print(sample)
                print(f"{adaptation} recovered key: {recovered_key_adapt}")
                input()

            curr_root  = root_paths_list[adaptation_ind][root_id]
            adapted_path = adapters[adaptation_ind].adapt(curr_root, curr_goal)
            adapt_end = time.perf_counter()
            adapt_time = adapt_end - adapt_start

            curr_success = adapted_path is not None and len(adapted_path)>0
            adaptation_success[adaptation].append(curr_success)
            adaptation_times[adaptation].append(adapt_time)
            adaptation_lengths[adaptation].append(traj_len(adapted_path))

        # RRTConnect
        path, total_time, planning_time = ompl_planner.plan(
            start=home_qpos,
            goal=key_goal,
            timeout=3.0,
            smooth_path=False,
            num_waypoints=200,
            benchmark=True,
        )
        if path is None or len(path)==0:
            #print(f"Planning failure for key: {key}")
            rrtc_success.append(False)
            rrtc_lengths.append(np.nan)
        else:
            rrtc_success.append(True)
            rrtc_lengths.append(traj_len(path))
        
        rrtc_solve_times.append(planning_time)

        # Library baseline
        library_path, library_time, lib_query_success = library.solve(sample)

        library_success.append(lib_query_success)
        library_times.append(library_time)
        
        if lib_query_success is True:
            library_lengths.append(traj_len(library_path))
        else:
            library_lengths.append(np.nan)

        num_tested += 1
        pbar.update(1)

    rrtc_success = np.array(rrtc_success)
    rrtc_times = np.array(rrtc_solve_times)
    rrtc_lengths = np.array(rrtc_lengths)

    library_success = np.array(library_success)
    library_times = np.array(library_times)
    library_lengths = np.array(library_lengths)

    rrtc_success_rate = np.mean(rrtc_success)*100
    library_success_rate = np.mean(library_success)*100
    
    rrtc_times_succ = rrtc_times[rrtc_success]
    library_times_succ = library_times[library_success]

    # ---- RRTConnect ----
    mean_rrtc_time_ms = np.nanmean(rrtc_times_succ) * 1000
    std_rrtc_time_ms  = np.nanstd(rrtc_times_succ, ddof=1) * 1000
    
    mean_rrtc_length = np.nanmean(rrtc_lengths)
    std_rrtc_length  = np.nanstd(rrtc_lengths, ddof=1)

    # ---- Library baseline ----
    mean_library_time_ms = np.nanmean(library_times_succ) * 1000
    std_library_time_ms  = np.nanstd(library_times_succ, ddof=1) * 1000

    mean_library_length = np.nanmean(library_lengths)
    std_library_length  = np.nanstd(library_lengths, ddof=1)

    print("\n=== RRTConnect results ===")
    print(f"RRTConnect success rate: {rrtc_success_rate:.2f}%")
    print(f"Mean RRTConnect time: {mean_rrtc_time_ms:.3f} ± {std_rrtc_time_ms:.3f} ms")
    print(f"Mean RRTConnect length: {mean_rrtc_length:.6f} ± {std_rrtc_length:.6f}")

    print("\n=== Library baseline results ===")
    print(f"Library success rate: {library_success_rate:.2f}%")
    print(f"Mean library time: {mean_library_time_ms:.3f} ± {std_library_time_ms:.3f} ms")
    print(f"Mean library length: {mean_library_length:.6f} ± {std_library_length:.6f}")


    for adaptation in adaptations:

        times = np.asarray(adaptation_times[adaptation], dtype=float)
        succ  = np.asarray(adaptation_success[adaptation], dtype=bool)

        times_succ = times[succ]

        mean_time_ms = np.nanmean(times_succ) * 1000
        std_time_ms  = np.nanstd(times_succ, ddof=1) * 1000

        #times = adaptation_times[adaptation]
        lengths = adaptation_lengths[adaptation]

        #mean_time_ms = np.nanmean(times) * 1000
        #std_time_ms  = np.nanstd(times, ddof=1) * 1000

        mean_length = np.nanmean(lengths)
        std_length  = np.nanstd(lengths, ddof=1)

        success_rate = np.mean(adaptation_success[adaptation]) * 100

        print(f"\n{adaptation} results")
        print(f"{adaptation} success rate: {success_rate:.2f}%")
        print(f"{adaptation} mean time: {mean_time_ms:.3f} ± {std_time_ms:.3f} ms")
        print(f"{adaptation} mean length: {mean_length:.6f} ± {std_length:.6f}")

    results_path = f"data/baseline_results_{args.robot}_{args.env}.npz"
    results = {
        "rrtc": {
            "success": rrtc_success,
            "times": rrtc_times,
            "lengths": rrtc_lengths,
        },
        "library": {
            "success": library_success,
            "times": library_times,
            "lengths": library_lengths,
        },
        "adaptations": {
            "success": adaptation_success,
            "times": adaptation_times,
            "lengths": adaptation_lengths,
        }
    }

    np.savez(results_path, results=results)



def main(args):
    """Evaluate path quality and query time for graph"""
    folder = get_data_folder(args.env, args.robot)
    suffix = f"{args.ik}_{args.planner}_{args.adaptation}_{args.n_neighbors}"
    
    # ---- output path for this run ----
    results_path = f"data/baseline_results_{args.robot}_{args.env}.npz"

    # ---- skip if already computed ----
    if os.path.exists(results_path) and not args.overwrite:
        print(f"[Skip] Results already exist: {results_path}")
        print("       Use --overwrite to re-run benchmarking.")
        return

    try:
        task_set = pickle.load(open(f"{folder}/task_set.pkl", "rb"))
        joint_goal_set = pickle.load(open(f"{folder}/joint_goal_set_{args.ik}.pkl", "rb"))
        d_name = f"{folder}/task_paths_data_{args.ik}_{args.planner}.npy"
        data = np.load(d_name, allow_pickle=True)
        k_name = f"{folder}/task_paths_keys_{args.ik}_{args.planner}.pkl"
        keys = pickle.load(open(k_name, "rb"))
        task_paths = {key: data for key, data in zip(keys, data)}

    except FileNotFoundError as e:
        print(e)
        print(
            f"One or more required files not found.")
        return

    IK_solved = sum(v is not None for v in joint_goal_set.values())
    planner_solved = sum(v is not None and len(v)>1 for v in task_paths.values())

    print(f"Number of generated tasks: {len(task_set)}")
    print(f"Number of tasks solved by IK: {IK_solved}")
    print(f"Number of paths solved by {args.planner}: {planner_solved}")

    # Check if graph data exists
    root_path = f"{folder}/root_paths_{suffix}.pkl"
    map_path = f"{folder}/key_to_root_{suffix}.pkl"

    root_exists = os.path.exists(root_path)
    map_exists = os.path.exists(map_path)
    
    # if not root_exists or not map_exists:
    #     print("Compressed root paths "
    #         + f"with IK '{args.ik}', planner '{args.planner}', "
    #         + f"and adaptation '{args.adaptation}' does NOT exist. "
    #         + "Use condense_task_paths.py to generate it."
    #     )
    #     return

    # Load environment and robot
    #env, robot = load_env_and_robot(args.env, args.robot)

    adaptations_found = []
    for adaptation in ['grr', 'opt', 'dmp']:
        suffix = f"{args.ik}_{args.planner}_{adaptation}_{args.n_neighbors}"
        root_path = f"{folder}/root_paths_{suffix}.pkl"
        map_path = f"{folder}/key_to_root_{suffix}.pkl"
        root_exists = os.path.exists(root_path)
        map_exists = os.path.exists(map_path)

        if root_exists and map_exists:   
            adaptations_found.append(adaptation)
    if len(adaptations_found)==0:
        raise FileNotFoundError(f"No adaptations found for problem: {args.robot} in {args.env}")        
    print(f"Adaptations found: {adaptations_found}")

    env_name = args.env
    robot_name = args.robot
    visualize = False

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

    # root_data = pickle.load(open(root_path, "rb"))
    # map_data = pickle.load(open(map_path, "rb"))

    # task_set_path = f"{folder}/task_set.pkl"
    # joint_goal_set_path = f"{folder}/joint_goal_set_neighbor.pkl"

    # task_set_data = pickle.load(open(task_set_path, "rb"))
    # joint_goal_set_data = pickle.load(open(joint_goal_set_path, "rb"))
    
    # print(f"Number of generated tasks: {len(task_set_data)}")
    # print(f"Number of solved IK: {len(joint_goal_set_data)}")

    #planning_results_path = f"{folder}/task_paths_results_neighbor_RRTConnect.npy"
    #planning_results = np.load(planning_results_path)

    num_samples = 1000
    evaluate_graph(args, env, robot, folder, task_set, task_paths, adaptations_found, num_samples)


def parse_arguments():
    parser = argparse.ArgumentParser()
    # envs
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--env",
        choices=["table", "box", "cage", "shelf", "free"],
        default="table",
    )
    parser.add_argument(
        "--robot", choices=["panda", "ur10", "fetch"], default="panda"
    )
    parser.add_argument(
        "--ik", choices=["random", "neighbor", "grr"], default="neighbor"
    )
    parser.add_argument(
        "--planner", choices=["RRTConnect", "PRMstar"], default="RRTConnect"
    )
    parser.add_argument(
        "--adaptation", choices=["linear", "grr", "dmp", "opt"], default="grr"
    )
    parser.add_argument("--n_neighbors", type=int, default=1000)

    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = parse_arguments()

    main(args)
