import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pickle
import mujoco
import time

from plan_load.robot import Panda

# from plan_load.task_space import deep_tuple
from experiments.evaluate import traj_len

folder1 = "dataset/top_naive"
folder2 = "dataset/top"


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


p1 = pickle.load(open(f"{folder1}/root_paths.pkl", "rb"))
p2 = pickle.load(open(f"{folder2}/root_paths.pkl", "rb"))
k1 = pickle.load(open(f"{folder1}/key_map.pkl", "rb"))
k2 = pickle.load(open(f"{folder2}/key_map.pkl", "rb"))

t1 = time.perf_counter()
print(get_avg_path_length(p1, k1))
t2 = time.perf_counter()
print(get_avg_path_length(p2, k2))
t3 = time.perf_counter()
print(f"Time taken for folder 1: {t2 - t1} seconds")
print(f"Time taken for folder 2: {t3 - t2} seconds")
