"""Automatically sample task_set.pkl, with an optional robot and task boundaries."""

import argparse
import os
from pathlib import Path
import pickle
import pickletools
import sys
import tempfile
import time

import mujoco
import mujoco.viewer
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experiments.common import ENVS, ROBOTS
from coad.utils import get_data_folder, load_env_and_robot

BOUNDARY_COLOR = [1, 0, 0, 1]
BOUNDARY_CLEARANCE = 0.005  # Keep the outline just above its support surface.
MICROWAVE_TASK_LIMIT = 20_000
_PICKLE_MARK = object()


def positive_seconds(value):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("interval must be finite and positive")
    return value


def set_camera(viewer):
    viewer.cam.lookat[:] = [0.25, -0.25, 0.5]
    viewer.cam.distance = 1.75
    viewer.cam.azimuth = 120
    viewer.cam.elevation = -20
    viewer.opt.geomgroup[3] = False


def split_tcr(key):
    return (key[0], key[1:]) if isinstance(key[0], str) else (None, key)


def sample_from_tcr(tcr):
    return [float(np.random.uniform(lo, hi)) for lo, hi in tcr]


def load_partial_microwave_tasks(path, limit=MICROWAVE_TASK_LIMIT):
    """Reservoir-sample complete microwave keys from a possibly truncated pickle.

    Microwave task dictionaries contain millions of entries. Reading pickle
    opcodes lets this viewer recover every complete key before a damaged tail
    without constructing the dictionary or regenerating the full task grid.
    """
    stack = []
    memo = []
    tasks = []
    complete = 0
    rng = np.random.default_rng()
    try:
        with path.open("rb") as stream:
            for opcode, argument, _ in pickletools.genops(stream):
                name = opcode.name
                if name == "EMPTY_DICT":
                    stack.append({})
                elif name == "MARK":
                    stack.append(_PICKLE_MARK)
                elif name in {"BINFLOAT", "BININT", "BININT1", "BININT2", "INT"}:
                    stack.append(argument)
                elif name == "TUPLE2":
                    second = stack.pop()
                    first = stack.pop()
                    stack.append((first, second))
                elif name == "TUPLE":
                    mark = len(stack) - 1 - stack[::-1].index(_PICKLE_MARK)
                    value = tuple(stack[mark + 1:])
                    del stack[mark:]
                    stack.append(value)
                    if (len(value) == 5 and all(
                            isinstance(bounds, tuple) and len(bounds) == 2
                            for bounds in value)):
                        complete += 1
                        if len(tasks) < limit:
                            tasks.append(value)
                        else:
                            replacement = int(rng.integers(complete))
                            if replacement < limit:
                                tasks[replacement] = value
                elif name == "NONE":
                    stack.append(None)
                elif name == "MEMOIZE":
                    memo.append(stack[-1])
                elif name in {"BINGET", "LONG_BINGET"}:
                    stack.append(memo[argument])
                elif name == "SETITEMS":
                    mark = len(stack) - 1 - stack[::-1].index(_PICKLE_MARK)
                    del stack[mark:]
                # PROTO, FRAME and STOP do not affect this simple
                # task-set pickle's value stack.
    except (EOFError, ValueError, pickle.UnpicklingError) as error:
        print(f"Microwave task pickle ends early: {error}", flush=True)
    if not tasks:
        raise ValueError(f"No complete microwave tasks could be recovered from {path}")
    print(f"Recovered {complete} complete microwave tasks; randomly selected "
          f"{len(tasks)} for visualization.")
    return tasks


def load_tasks(env_name, robot_name, env):
    path = Path(get_data_folder(env_name, robot_name)) / "task_set.pkl"
    if env_name == "microwave" and path.exists():
        try:
            return load_partial_microwave_tasks(path)
        except ValueError as error:
            print(f"Partial microwave recovery failed: {error}", flush=True)
    try:
        with path.open("rb") as stream:
            tasks = list(pickle.load(stream))
    except (FileNotFoundError, EOFError, pickle.UnpicklingError) as error:
        print(f"Cannot load {path}: {error}", flush=True)
        print("Generating a task set in memory; the saved file will not be changed.",
              flush=True)
        env.construct_tcr()
        tasks = list(env.generate_task_set())
        if not tasks:
            raise ValueError(f"No tasks generated for {env_name}/{robot_name}")
        print(f"Generated {len(tasks)} tasks.")
        return tasks
    if not tasks:
        raise ValueError(f"Empty task set: {path}")
    print(f"Loaded {len(tasks)} tasks from {path}")
    return tasks


def move_object(env, face, pose):
    if env.environment_name == "microwave":
        env.move_object(pose[:4])
        env.move_xml_joint("microwave_door_hinge", pose[4])
    else:
        env.move_object([face, *pose] if face else pose)
    if face:
        # The environment parks inactive stable-face objects nearby; hide them.
        active = ("xy", "yz", "zx").index(face)
        for i in range(3):
            body = env.model.body(f"cube_object_{i}").id
            env.model.geom_rgba[env.model.geom_bodyid == body, 3] = float(i == active)


def remove_robot(env, robot):
    """Compile a scene without the robot subtree or its dependent elements."""
    with tempfile.NamedTemporaryFile(suffix=".xml") as stream:
        mujoco.mj_saveLastXML(stream.name, env.model)
        spec = mujoco.MjSpec.from_file(stream.name)
    spec.meshdir = str(Path(env.robot_dir, spec.meshdir).resolve())
    spec.texturedir = str(Path(env.robot_dir, spec.texturedir).resolve())
    spec.delete(spec.body(robot.root_link))
    for key in list(spec.keys):
        spec.delete(key)
    env.model = spec.compile()
    env.data = mujoco.MjData(env.model)
    mujoco.mj_forward(env.model, env.data)


def line(scene, start, end, color, width=0.003):
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                       np.zeros(3), np.eye(3).ravel(), np.asarray(color, dtype=float))
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, width,
                         np.asarray(start), np.asarray(end))
    scene.ngeom += 1


def rectangle(scene, xy, z, color):
    (x0, x1), (y0, y1) = xy
    corners = [(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)]
    for i in range(4):
        line(scene, corners[i], corners[(i + 1) % 4], color)


def task_boundaries(env):
    """Draw XY outlines at the configured table/shelf support heights."""
    regions = env.env_details.get("intervals")
    if regions is None:
        variation = env.object_details["variation"]
        regions = [[x, y] for x in variation["x"] for y in variation["y"]]
    regions = np.asarray(regions, dtype=float).reshape(-1, 2, 2)
    regions = np.unique(regions, axis=0)
    heights = np.unique(env.env_details.get("z_correction", [0]))
    return [(region, float(z) + BOUNDARY_CLEARANCE)
            for region in regions for z in heights]


def main(args):
    source_robot = args.robot if args.robot != "none" else args.task_robot
    env, robot = load_env_and_robot(args.env, source_robot, False, False, False)
    if args.robot == "none":
        remove_robot(env, robot)
    else:
        robot.set_joint_qpos(robot.home_pos)
    tasks = load_tasks(args.env, source_robot, env)
    boundaries = task_boundaries(env)
    print("Red: XY task domain. Close the viewer to quit.")
    try:
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            set_camera(viewer)
            next_sample = 0.0
            while viewer.is_running():
                if time.monotonic() >= next_sample:
                    key = tasks[np.random.randint(len(tasks))]
                    face, tcr = split_tcr(key)
                    pose = sample_from_tcr(tcr)
                    with viewer.lock():
                        move_object(env, face, pose)
                        viewer.user_scn.ngeom = 0
                        for xy, z in boundaries:
                            rectangle(viewer.user_scn, xy, z, BOUNDARY_COLOR)
                    next_sample = time.monotonic() + args.interval
                viewer.sync()
                time.sleep(1 / 60)
    except KeyboardInterrupt:
        pass
    finally:
        robot.close()


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=ENVS, default="shelf")
    parser.add_argument("--robot", choices=(*ROBOTS, "none"), default="none")
    parser.add_argument("--task-robot", choices=ROBOTS, default="panda",
                        help="Task dataset/configuration when --robot=none (default: panda)")
    parser.add_argument("--interval", type=positive_seconds, default=1.0,
                        help="Seconds between samples (default: 1)")
    return parser.parse_args()


if __name__ == "__main__":
    os.chdir(REPO)
    main(parse_arguments())
