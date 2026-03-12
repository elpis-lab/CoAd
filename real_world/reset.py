import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import numpy as np

from real_world.physical_robot import PhysicalUR10


def reset_robot():
    real_robot = PhysicalUR10()
    home_pos = np.array([2.0309, -1.095, 1.5799, -2.071, -1.5938, 0.5060])
    real_robot.control_gripper("open")
    time.sleep(1.0)

    real_robot.move_joint(
        home_pos, speed=0.5, acceleration=1.0, asynchronous=False
    )


if __name__ == "__main__":
    reset_robot()
