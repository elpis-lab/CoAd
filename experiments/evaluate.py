import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pickle
import mujoco
import time

from plan_load.robot import Panda


def get_avg_path_length(data, offsets):
    lengths = np.zeros(len(offsets), dtype=np.float64)
    for i in range(len(offsets)):
        lengths[i] = traj_len(get_path_by_index(data, offsets, i))
        if lengths[i] == 0.0:
            lengths[i] = np.nan
    return np.nanmean(lengths)


def get_path_by_index(data, offsets, i):
    s, e = int(offsets[i]), int(offsets[i + 1])
    return data[s:e]


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


def test():
    folder1 = "dataset/top_near"
    folder2 = "dataset/top"

    d1 = np.load(f"{folder1}/data.npy")
    d2 = np.load(f"{folder2}/data.npy")
    k1 = np.load(f"{folder1}/keys.npy")
    k2 = np.load(f"{folder2}/keys.npy")
    o1 = np.load(f"{folder1}/offsets.npy")
    o2 = np.load(f"{folder2}/offsets.npy")
    j1 = pickle.load(open(f"joint_goal_set_near.pkl", "rb"))
    j2 = pickle.load(open(f"joint_goal_set_prev.pkl", "rb"))

    k1_list = list(k1)
    k2_list = list(k2)
    key1 = k1_list[1]
    key2 = k2_list[1]

    # print(j1[tuple(tuple(v for v in row) for row in key1)])
    # print(j2[tuple(tuple(v for v in row) for row in key2)])
    # p1 = get_path_by_index(d1, o1, 1)
    # p2 = get_path_by_index(d2, o2, 1)
    # print(p1[-1])
    # print(p2[-1])

    # j1_list = list(j1.values())
    # j2_list = list(j2.values())
    # j1_case = j1_list[1]
    # j2_case = j2_list[1]
    # print(j1_case)
    # print(j2_case)

    t1 = time.perf_counter()
    p1 = get_path_by_index(d1, o1, 1)
    p2 = get_path_by_index(d2, o2, 1)
    t2 = time.perf_counter()
    print(f"Time taken: {t2 - t1} seconds")
    # print(p1[-1])
    # print(p2[-1])

    # print(np.allclose(k1, k2))
    # print(np.allclose(d1, d2))
    # print(np.allclose(o1, o2))
    # print(np.allclose(np.array(j1_list), np.array(j2_list)))
    # print(j1[tuple(tuple(v for v in row) for row in k1[100])])
    # print(j2[tuple(tuple(v for v in row) for row in k1[100])])

    print(get_avg_path_length(d1, o1))
    print(get_avg_path_length(d2, o2))

    # model = mujoco.MjModel.from_xml_path("assets/franka_emika_panda/scene.xml")
    # robot = Panda(model, visualize=True)
    # traj = get_path_by_index(d1, o1, 10000)
    # for state in traj:
    #     robot.set_joint_qpos(state)
    #     robot.viewer.sync()
    #     time.sleep(0.01)

    # robot.close()


if __name__ == "__main__":
    test()
