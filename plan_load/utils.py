import warnings
import random
import numpy as np
import argparse

from plan_load.env import MujocoEnv, TableEnv, BoxEnv, CageEnv, ShelfEnv, FreeEnv
from plan_load.robot import MujocoRobot, Panda, UR10, FetchArm

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
    env_name: str, robot_name: str, visualize: bool = False
) -> tuple[MujocoEnv, MujocoRobot]:
    # Build scene for given environment
    if env_name == "table":
        env = TableEnv(robot_name)
    elif env_name == "box":
        env = BoxEnv(robot_name)
    elif env_name == "cage":
        env = CageEnv(robot_name)
    elif env_name == "shelf":
        env = ShelfEnv(robot_name)
    elif env_name == "free":
        env = FreeEnv(robot_name)
    else:
        raise ValueError(f"Invalid environment: {env_name}")

    # Create robot instance
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
