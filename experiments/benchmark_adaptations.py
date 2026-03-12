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
from experiments.visualize_baselines import traj_len
from plan_load.utils import set_seed, load_env_and_robot, get_data_folder
from plan_load.planning import OMPLPlanner, euclidean_path_length

from plan_load.env import FreeEnv, CageEnv, BoxEnv, TableEnv, ShelfEnv, RealEnv
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

def wrap_pi(self, a):
    return (a + np.pi) % (2*np.pi) - np.pi

def evaluate_adaptations(
    args,
    env: MujocoEnv,
    robot: MujocoRobot,
    folder,
    task_set,
    task_paths,
    adaptations
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

    full_lib_success = []
    full_lib_times = []
    full_lib_lengths = []

    adaptation_success = {}
    adaptation_times = {}
    adaptation_lengths = {}


    lib_sizes = {}
    lib_sizes['full'] = len(solved_task_paths_keys)

    for adaptation in adaptations:

        adaptation_success[f"{adaptation}"] = []
        adaptation_times[f"{adaptation}"] = []
        adaptation_lengths[f"{adaptation}"] = []

        if adaptation == 'dmp' and args.env=="real" and args.robot == "ur10":
            suffix = f"{args.ik}_{args.planner}_{adaptation}_100"
        else:
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

    

    pbar = tqdm(enumerate(solved_task_paths_keys), total=len(solved_task_paths_keys))
    for i, key in enumerate(solved_task_paths_keys):

        sample = [] 
        for lo, hi in key:
            x = np.random.uniform(lo, hi)
            #x = np.nextafter(x, lo)
            sample.append(x)
        env.move_cube_object(sample)

        # Benchmark full library
        full_start = time.perf_counter()
        recovered_key_full = indexers[0].query_point(sample)

        if recovered_key_full is None:
            print("full library failure")
            # print(key)
            # print(sample)
            # print(recovered_key_full)
            # print(len(task_paths[key]))
            # input()
            full_lib_success.append(False)
            full_end = time.perf_counter()
            full_lib_times.append(full_end - full_start)
            full_lib_lengths.append(np.nan)
        else:
            # print(key)
            # print(sample)
            # print(recovered_key_full)
            if key != recovered_key_full:
                print(f"Original key: {key}")
                # print(sample)
                print(f"Full library recovered key: {recovered_key_full}")
                
                # y = float(sample[1])
                # print("y:", repr(y))
                # print("y0:", repr(indexers[0].y0), "dy:", repr(indexers[0].dy))
                # t = (y - indexers[0].y0) / indexers[0].dy
                # print("t:", t, "floor:", np.floor(t), "frac:", t - np.floor(t))

                # print("orig y_lo/hi:", repr(key[1][0]), repr(key[1][1]))
                # print("reco y_lo/hi:", repr(recovered_key_full[1][0]), repr(recovered_key_full[1][1]))
                
                input()

            full_lib_path = task_paths[recovered_key_full]
            full_end = time.perf_counter()
            full_lib_success.append(True)
            full_lib_times.append(full_end - full_start)
            full_lib_lengths.append(traj_len(full_lib_path))
        
        # Benchmark each adaptation
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

        pbar.update(1)

    full_lib_success = np.array(full_lib_success)
    full_lib_times = np.array(full_lib_times)
    full_lib_lengths = np.array(full_lib_lengths)

    print(f"\nFull library results")
    print(f"Full library success rate: {np.mean(full_lib_success)*100}%")
    print(f"Full library mean time: {np.mean(full_lib_times)*1000} ms")
    print(f"Full library mean length: {np.nanmean(full_lib_lengths)}")

    for adaptation in adaptations:
        adaptation_success[adaptation] = np.array(adaptation_success[adaptation])
        adaptation_times[adaptation] = np.array(adaptation_times[adaptation])
        adaptation_lengths[adaptation] = np.array(adaptation_lengths[adaptation])

        print(f"\n{adaptation} results")
        curr_compression_ratio = lib_sizes[adaptation]/lib_sizes['full']
        curr_comp_percent = (1 - curr_compression_ratio)*100
        print(f"{adaptation} compression %: {curr_comp_percent}%")
        print(f"{adaptation} success rate: {np.mean(adaptation_success[adaptation])*100}%")
        print(f"{adaptation} mean time: {np.mean(adaptation_times[adaptation])*1000} ms")
        print(f"{adaptation} mean length: {np.nanmean(adaptation_lengths[adaptation])}")


    results_path = f"data/adaptation_results_{args.robot}_{args.env}.npz"
    results = {
        "full": {
            "success": full_lib_success,
            "times": full_lib_times,
            "lengths": full_lib_lengths,
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

    adaptations_found = []
    for adaptation in ['grr', 'opt', 'dmp']:
        if adaptation == 'dmp' and args.env=="real" and args.robot == "ur10":
            suffix = f"{args.ik}_{args.planner}_{adaptation}_100"
        else:
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
    elif env_name == "real":
        env = RealEnv(robot_name, no_sv=True)
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

    #root_data = pickle.load(open(root_path, "rb"))
    #map_data = pickle.load(open(map_path, "rb"))

    #planning_results_path = f"{folder}/task_paths_results_neighbor_RRTConnect.npy"
    #planning_results = np.load(planning_results_path)

    evaluate_adaptations(args, env, robot, folder, task_set, task_paths, adaptations_found)


def parse_arguments():
    parser = argparse.ArgumentParser()
    # envs
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--env",
        choices=["table", "box", "cage", "shelf", "free", "real"],
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
