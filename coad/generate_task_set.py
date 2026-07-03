import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import time
import numpy as np
import pickle
from tqdm import tqdm

from coad.utils import set_seed, load_env_and_robot, get_data_folder


def main(args):
    """Generate a task set for a given environment and robot."""
    folder = get_data_folder(args.env, args.robot)

    # Check if task set is already generated
    data_exists = os.path.exists(f"{folder}/task_set.pkl")
    if data_exists and not args.overwrite:
        print(
            f"Task set already exists at {folder}. "
            + "Use --overwrite to regenerate the task set."
        )
        return

    # Load environment and robot
    env, robot = load_env_and_robot(args.env, args.robot, False)

    # Solve task set
    task_set = env.generate_task_set()
    print(f"Length of task set: {len(task_set)}")

    # Save task set
    os.makedirs(folder, exist_ok=True)
    pickle.dump(task_set, open(f"{folder}/task_set.pkl", "wb"))
    robot.close()


def parse_arguments():
    parser = argparse.ArgumentParser()
    # envs
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--env", choices=[
            "table",
            "box",
            "cage",
            "shelf",
            "free",
            "real",
            "largeobj",
            "microwave",
            "allstable"], default="table"
    )
    parser.add_argument(
        "--robot", choices=[
            "panda",
            "ur10",
            "fetch",
            "g1"], default="panda"
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    set_seed(42)
    args = parse_arguments()

    main(args)
