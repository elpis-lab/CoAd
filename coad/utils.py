import warnings
import random
import numpy as np
import argparse

from coad.env import (
    MujocoEnv,
    TableEnv,
    BoxEnv,
    CageEnv,
    ShelfEnv,
    FreeEnv,
    RealEnv,
    LargeObjectEnv,
    MicrowaveEnv,
    AllStableEnv,
    ConveyorEnv,
)
from coad.robot import MujocoRobot, Panda, UR10, FetchArm

# for better printing
warnings.filterwarnings("ignore", category=FutureWarning)
np.set_printoptions(precision=5, suppress=True)


def set_seed(seed: int):
    """Set seed for reproducibility"""
    np.random.seed(seed)
    random.seed(seed)


def parse_args(args: list[tuple[str, any, type]]) -> argparse.Namespace:
    """
    A simple wrapper for argument parser
    args is a list of arguments, each argument is
    a tuple of (name, default(optional), type(optional))
    """
    parser = argparse.ArgumentParser()
    for arg in args:
        kwargs = {"nargs": "?"}
        if len(arg) > 1:
            kwargs["default"] = arg[1]
        if len(arg) > 2:
            kwargs["type"] = arg[2]
        parser.add_argument(arg[0], **kwargs)

    args = parser.parse_args()
    return args


def get_data_folder(env_name: str, robot_name: str) -> str:
    return f"data/{env_name}_{robot_name}"


def load_env_and_robot(
    env_name: str,
    robot_name: str,
    visualize: bool = True,
    using_swept_volume: bool = True,
    compute_tcr: bool = True,
) -> tuple[MujocoEnv, MujocoRobot]:
    # Keep one constructor path so optional flags reach every environment.
    environments = {
        "table": TableEnv, "box": BoxEnv, "cage": CageEnv,
        "shelf": ShelfEnv, "free": FreeEnv, "real": RealEnv,
        "largeobj": LargeObjectEnv, "microwave": MicrowaveEnv,
        "allstable": AllStableEnv, "conveyor": ConveyorEnv,
    }
    if env_name not in environments:
        raise ValueError(f"Invalid environment: {env_name}")
    env = environments[env_name](
        robot_name, using_swept_volume=using_swept_volume, compute_tcr=compute_tcr
    )

    # Configure problem home pose
    NEW_ENVS = [LargeObjectEnv, AllStableEnv, MicrowaveEnv]

    # Change fetch_table start config
    if robot_name == "fetch":
        NEW_ENVS.append(TableEnv)

    NEW_ENVS = tuple(NEW_ENVS)
    home_pose_flag = "new" if isinstance(env, NEW_ENVS) else "default"

    # Create robot instance
    model, data = env.model, env.data
    if robot_name == "panda":
        robot = Panda(model, data, visualize, home_pose=home_pose_flag)
    elif robot_name == "ur10":
        robot = UR10(model, data, visualize)
    elif robot_name == "fetch":
        robot = FetchArm(model, data, visualize, home_pose=home_pose_flag)
    else:
        raise ValueError(f"Invalid robot: {robot_name}")

    robot_pos = env.env_details["robot_pos"]
    robot_quat = env.env_details["robot_quat"]
    robot.teleport_base(pos=robot_pos, quat=robot_quat)
    if hasattr(env, "home_qpos"):
        robot.set_joint_qpos(env.home_qpos)
        robot.home_pos = env.home_qpos.copy()
    return env, robot
