import mujoco
import numpy as np
from tqdm import tqdm
from collections import deque

from helpers.helpers import load_store, get_path_by_index
from helpers.mujoco_utils import move_swept_volume

from plan_load.mujoco_utils import geoms_in_contact
from plan_load.task_space import build_task_nn, deep_tuple
from plan_load.mink_ik import IK
from plan_load.robot import MujocoRobot
from plan_load.pose import Pose


def joint_distance(q1, q2):
    return np.linalg.norm(q1 - q2)


def joint_interpolate(q1, q2, alpha):
    return (1.0 - alpha) * q1 + alpha * q2


def workspace_interpolate(p1, p2, alpha):
    pose1 = Pose(position=p1[:3], rotation=p1[3:])
    pose2 = Pose(position=p2[:3], rotation=p2[3:])
    pose = pose1.interpolate(pose2, alpha)
    return pose.flat()


def segment_continuity_check(robot: MujocoRobot, ik_solver: IK, q1, q2):
    """Check if two configurations follow the continuous constraints

    Linearly interpolate and bisectionally visit and check
    with the help of a queue to avoid stack overflow issues
    """
    epsilon = np.sqrt(robot.n_joints) * 0.05
    deviation = 1.8
    robot.set_joint_qpos(q1)
    p1 = robot.get_ee_pose()
    robot.set_joint_qpos(q2)
    p2 = robot.get_ee_pose()

    n_divs = int(np.ceil(joint_distance(q1, q2) / epsilon))
    queue = deque()
    queue.append((q1, q2, 0, n_divs + 1))

    while len(queue) > 0:
        qa, qb, ia, ib = queue.popleft()
        d = joint_distance(qa, qb)

        # If the two points are already adjacent
        if ib == ia + 1:
            continue

        # Find the middle point and configuration
        im = (ia + ib) // 2
        reached, qm = ik_solver.solve(
            workspace_interpolate(p1, p2, im / (n_divs + 1)),
            joint_interpolate(qa, qb, (im - ia) / (ib - ia)),
            use_col=False,
        )
        if not reached:
            return False

        # Check middle's deviation from "straight path"
        d1 = joint_distance(qa, qm)
        if d1 > deviation * d:
            return False
        d2 = joint_distance(qm, qb)
        if d2 > deviation * d:
            return False

        # Add to queue for further bisection
        queue.append((qa, qm, ia, im))
        queue.append((qm, qb, im, ib))
    return True


def segment_validity_check(
    robot: MujocoRobot,
    q_start: np.ndarray,
    q_end: np.ndarray,
    max_step: float = 0.05,
) -> bool:
    """
    Check if straight-line interpolation between q_start and q_end is
    collision free for the robot (including environment).
    """
    q_start = np.asarray(q_start, dtype=float)
    q_end = np.asarray(q_end, dtype=float)
    diff = q_end - q_start

    # Use infinity norm so that per-joint step is bounded by max_step
    dist = np.linalg.norm(diff, ord=np.inf)
    n_steps = max(int(np.ceil(dist / max_step)) + 1, 2)

    # Per-step collision check
    for alpha in np.linspace(0.0, 1.0, n_steps):
        q = (1.0 - alpha) * q_start + alpha * q_end
        robot.set_joint_qpos(q, robot.data)
        if robot.in_contact():
            return False
    return True


def condense_dataset(
    robot: MujocoRobot,
    ik_solver: IK,
    graph_data,
    offsets,
    keys,
    max_num_neighbors=1000,
):
    """
    Condense a dataset of joint-space paths by greedily picking root paths
    and compressing nearby neighbors via direct interpolation using IK.

    Algorithm (simplified, no prefix/suffix graph search):
    - Load all paths and remaining bins.
    - While there are remaining bins:
        - Randomly pick a center bin as a new root.
        - Use a Ballnn (SE2 distance over bin centers) to find nearest
          neighbors among remaining bins.
        - For each neighbor:
            - Compute its end-effector pose from the stored solution.
            - Run IK using the center's final configuration as the seed.
            - If successful and the straight segment from the center's
              final configuration to this IK solution is collision free,
              mark the neighbor as compressed into this root.
    - Return:
        - root_paths: dict[int, np.ndarray] of root joint-space paths.
        - prefix_map: dict[key -> (root_id_str, goal_q_list)] mapping each
          workspace bin to its associated root and terminal joint goal.
    """
    model, data = robot.model, robot.data
    # Build BallTree on bin centers
    nn, bin_poses = build_task_nn(keys)

    # Get bins that don't have solutions in the graph
    no_sol_bins = set()
    for i in range(len(keys)):
        path_i = get_path_by_index(graph_data, offsets, i)
        if len(path_i) < 2:
            no_sol_bins.add(i)
    print(f"Number of bins: {len(keys)}")
    print(f"Non-empty bins: {len(keys) - len(no_sol_bins)}")

    # Container
    # root_id -> root path
    root_paths = {}
    # key -> (root_id, goal_q)
    key_map = {key: (None, None) for key in deep_tuple(keys)}

    # Greedy condensation loop
    pbar = tqdm(total=len(keys) - len(no_sol_bins))
    remaining = set(range(len(keys)))
    while remaining != no_sol_bins:
        # 1) Pick a random center among remaining bins
        center_idx = int(np.random.choice(list(remaining)))
        center_path = get_path_by_index(graph_data, offsets, center_idx)
        if len(center_path) < 2:
            continue

        # Register new root path
        root_id = len(root_paths)
        center_key = deep_tuple(keys[center_idx])
        root_paths[root_id] = center_path.copy()
        key_map[center_key] = (root_id, center_path[-1])
        remaining.remove(center_idx)

        robot.set_joint_qpos(center_path[-1])
        if robot.viewer:
            robot.viewer.sync()
        pbar.update(1)

        # 2) Query nearest neighbors among all bins
        # Consider those that are still in remaining
        neighbor_indices = nn.query(
            bin_poses[center_idx : center_idx + 1],
            k=max_num_neighbors + 1,
            return_distance=True,
            sort_results=True,
        )[1][0]

        # 3) Try to compress neighbors into this root.
        for nb_idx in neighbor_indices:
            if nb_idx not in remaining or nb_idx == center_idx:
                continue
            nb_key = deep_tuple(keys[nb_idx])

            nb_path = get_path_by_index(graph_data, offsets, nb_idx)
            if len(nb_path) < 2:
                remaining.discard(nb_idx)
                continue

            # Option1, interpolate directly
            # q_nb_end = nb_path[-1]

            # Option2, IK
            robot.set_joint_qpos(nb_path[-1])
            target = robot.get_ee_pose()
            reached, q_nb_end = ik_solver.solve(
                target,
                current=center_path[-1],
                use_col=False,
            )
            if not reached:
                continue

            # Check if simple interpolation is collision free.
            move_swept_volume(model, data, keys[nb_idx])
            if not segment_validity_check(robot, center_path[-1], q_nb_end):
                continue
            # Check if the segment is continuous
            if not segment_continuity_check(
                robot, ik_solver, center_path[-1], q_nb_end
            ):
                continue

            # Neighbor successfully compressed into this root.
            key_map[nb_key] = (root_id, q_nb_end)
            remaining.discard(nb_idx)

            robot.set_joint_qpos(q_nb_end)
            if robot.viewer:
                robot.viewer.sync()
            pbar.update(1)

    print("Condensation complete.")
    print(f"Number of root paths: {len(root_paths)}")
    return root_paths, key_map
