import math
import numpy as np

from sklearn.neighbors import BallTree
from geometry.pose import angle_diff


def env_distance(p1, p2, position_weight=1.0, rotation_weight=0.5):
    """
    Compute the distance between two workspace poses.
    Specifically, the workspace pose only varies in position and yaw.
    """
    # Position component
    d_position = math.hypot(p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2])
    # Rotation component
    d_rotation = abs(angle_diff(p1[3], p2[3]))
    return position_weight * d_position + rotation_weight * d_rotation


def build_task_nn(keys):
    """Build a Ball tree for finding the nearest bin."""
    if isinstance(keys, dict):
        bins = np.array(list(keys.keys()))
    else:
        bins = np.asarray(keys)

    # Represent bin as a SE2 pose
    bin_poses = (bins[:, :, 0] + bins[:, :, 1]) / 2
    nn = BallTree(bin_poses, metric=env_distance)
    return nn, bin_poses
