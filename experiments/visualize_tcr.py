"""Show TSR grasps and the inverse SE(2) object region with a robot.

Te = H(q) Delta Ts. In TSR mode H(q) is fixed and IK follows sampled Te.
In TCR mode Te is fixed and H(q) = Te inv(Ts) inv(Delta); object height,
contact face, and (for a microwave) door angle stay fixed. Samples must satisfy
the TSR and a MuJoCo penetration check. Boundary probes are sampled checks,
not a continuous-region certificate or a collision-free motion plan.
All samples are solved and checked before the viewer opens; playback loops
over the accepted configurations without running IK or collision checks.
"""

import argparse
import itertools
import json
import os
from pathlib import Path
import sys
import time

import mujoco.viewer
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experiments.common import ENVS, ROBOTS
from experiments.visualize_goals import (
    move_object,
    positive_seconds,
    sample_from_tcr,
    set_camera,
    split_tcr,
)
from coad.mink_ik import get_ik_solver
from coad.tcr import TSRBounds, reference_pose
from coad.utils import get_data_folder, load_env_and_robot
from geometry.pose import flat_to_matrix, matrix_to_flat


def sample_delta(bounds, planar=False, corner=None):
    """Sample the object-reference-frame displacement, before the grasp offset."""
    half = [
        bounds.x - bounds.position_slack,
        bounds.y - bounds.position_slack,
        0 if planar else max(0, bounds.z - bounds.position_slack),
        max(0, bounds.yaw - bounds.rotation_slack),
    ]
    half = np.asarray(half)
    if corner is not None:
        # Cycle through all corners as well as sampling the interior.
        axes = np.flatnonzero(half)
        displacement = np.zeros(4)
        for bit, axis in enumerate(axes):
            displacement[axis] = half[axis] * (
                1 if corner & (1 << bit) else -1
            )
        return reference_pose(displacement)
    return reference_pose(np.random.uniform(-half, half))


def inverse_object_pose(ee, offset, delta, anchor, details=None):
    """Invert the TSR, retaining the anchor's non-SE(2) coordinates."""
    H = ee @ np.linalg.inv(offset) @ np.linalg.inv(delta)
    if details is not None:
        local_handle = reference_pose([0, 0, 0, 0, anchor[4]], details)
        H = H @ np.linalg.inv(local_handle)
    pose = np.array(anchor, dtype=float).copy()
    pose[:2] = H[:2, 3]
    pose[3] = np.arctan2(H[1, 0], H[0, 0])
    return pose


def fits_tsr(ee, pose, offset, bounds, details=None):
    delta = np.linalg.solve(
        reference_pose(pose, details), ee @ np.linalg.inv(offset)
    )
    tilt = np.arccos(np.clip(delta[2, 2], -1, 1))
    return (
        np.all(
            np.abs(delta[:3, 3])
            <= np.array([bounds.x, bounds.y, bounds.z]) + 1e-12
        )
        and abs(np.arctan2(delta[1, 0], delta[0, 0])) <= bounds.yaw + 1e-12
        and tilt <= bounds.rotation_slack + 1e-12
    )


def collision_geoms(env, robot, face):
    """Robot collision proxies plus the active object's entire body subtree."""
    if face:
        name = f"cube_object_{('xy', 'yz', 'zx').index(face)}"
    else:
        name = (
            "microwave_object"
            if env.environment_name == "microwave"
            else "cube_object"
        )
    bodies = {env.model.body(name).id}
    for body in range(env.model.nbody):
        if env.model.body_parentid[body] in bodies:
            bodies.add(body)
    return set(robot.robot_geoms) | set(
        np.flatnonzero(np.isin(env.model.geom_bodyid, list(bodies)))
    )


def has_penetration(data, geoms, tolerance=1e-6):
    """Allow touching a support/grasp surface, but reject penetration > 1 micron."""
    return any(
        contact.dist < -tolerance
        and (contact.geom1 in geoms or contact.geom2 in geoms)
        for contact in data.contact[: data.ncon]
    )


def accept_sample(
    env, robot, face, pose, q, offset, bounds, geoms, details=None
):
    """Keep only a valid sample; restore the previous state on rejection.

    Used during precomputation. move_object/set_joint_qpos run mj_forward,
    so their contact buffer already contains the fast broad/narrow-phase result.
    """
    previous_qpos = env.data.qpos.copy()
    previous_rgba = env.model.geom_rgba.copy()
    move_object(env, face, pose)
    robot.set_joint_qpos(q)
    actual_ee = flat_to_matrix(robot.get_ee_pose())
    reason = None
    if not fits_tsr(actual_ee, pose, offset, bounds, details):
        reason = "TSR"
    elif has_penetration(env.data, geoms):
        reason = "collision"
    if reason:
        env.data.qpos[:] = previous_qpos
        env.model.geom_rgba[:] = previous_rgba
        mujoco.mj_forward(env.model, env.data)
    return reason


def reference_tasks(env):
    """Sample anchors from scene placement ranges without requiring saved IK/tasks."""
    variation = env.object_details["variation"]
    regions = env.env_details.get("intervals")
    if regions is None:
        regions = [[x, y] for x in variation["x"] for y in variation["y"]]
    regions = np.asarray(regions).reshape(-1, 2, 2)
    clipped = []
    for region, x, y in itertools.product(
        regions, variation["x"], variation["y"]
    ):
        box = np.array([x, y], dtype=float)
        box[:, 0] = np.maximum(box[:, 0], region[:, 0])
        box[:, 1] = np.minimum(box[:, 1], region[:, 1])
        if np.all(box[:, 0] <= box[:, 1]):
            clipped.append(box)
    regions = clipped
    faces = env.env_details.get("tcr_batches", [None])
    tasks = []
    for face, xy, z, dz, yaw, door in itertools.product(
        faces,
        regions,
        variation["z"],
        env.env_details.get("z_correction", [0]),
        variation["yaw"],
        variation.get("door", [None]),
    ):
        if face:
            height = (
                env.object_details["size"][("yz", "zx", "xy").index(face)] / 2
            )
            z = [height, height]
        key = (*map(tuple, xy), tuple(np.asarray(z) + dz), tuple(yaw))
        if door is not None:
            key += (tuple(door),)
        tasks.append((face, *key) if face else key)
    return tasks


def find_anchor(env, robot, solver, tasks, cells, attempts=200):
    if not tasks:
        raise ValueError(
            "No object placement ranges intersect the task domain"
        )
    for attempt in range(attempts):
        for _ in range(1000):
            face, key = split_tcr(tasks[np.random.randint(len(tasks))])
            pose = sample_from_tcr(key)
            distance = np.linalg.norm(
                np.asarray(pose[:2]) - env.env_details["robot_pos"][:2]
            )
            if (
                env.env_details["inner_rad"]
                <= distance
                <= env.env_details["outer_rad"]
            ):
                break
        else:
            raise RuntimeError(
                "Could not sample a pose in the robot's task annulus"
            )
        cell = cells[face or "default"]
        offsets = [np.asarray(T) for T in cell["ee_offsets"]]
        offset = offsets[np.random.randint(len(offsets))]
        details = (
            env.object_details if env.environment_name == "microwave" else None
        )
        target = reference_pose(pose, details) @ offset
        ok, q = solver.solve(
            matrix_to_flat(target),
            current=env.data.qpos.copy(),
            random_current=attempt % 3 == 2,
            max_iters=200,
            pos_tol=1e-6,
            rot_tol=1e-5,
        )
        if ok:
            reason = accept_sample(
                env,
                robot,
                face,
                pose,
                q,
                offset,
                TSRBounds(**cell["tsr_bounds"]),
                collision_geoms(env, robot, face),
                details,
            )
            if reason is None:
                return face, pose, cell, offset
    raise RuntimeError(
        f"No collision-free TSR anchor found in {attempts} IK attempts"
    )


def precompute_samples(env, robot, solver, anchor, modes, samples):
    """Build finite, validated playback buffers before launching any viewer."""
    face, pose, cell, offset = anchor
    bounds = TSRBounds(**cell["tsr_bounds"])
    geoms = collision_geoms(env, robot, face)
    details = (
        env.object_details if env.environment_name == "microwave" else None
    )
    anchor_qpos = env.data.qpos.copy()
    anchor_q = robot.get_joint_qpos()
    fixed_ee = flat_to_matrix(robot.get_ee_pose())
    offsets = [np.asarray(T) for T in cell["ee_offsets"]]
    buffers = {}
    try:
        for mode in modes:
            env.data.qpos[:] = anchor_qpos
            mujoco.mj_forward(env.model, env.data)
            frames = []
            rejected = {"IK": 0, "TSR": 0, "collision": 0}
            print(
                f"Preparing {mode.upper()}: {samples} candidates...",
                flush=True,
            )
            for count in range(samples):
                delta = sample_delta(
                    bounds,
                    planar=mode == "tcr",
                    corner=count // 2 if count % 2 == 0 else None,
                )
                if mode == "tsr":
                    grasp = offsets[np.random.randint(len(offsets))]
                    target = reference_pose(pose, details) @ delta @ grasp
                    ok, q = solver.solve(
                        matrix_to_flat(target),
                        current=env.data.qpos.copy(),
                        max_iters=60,
                        pos_tol=1e-6,
                        rot_tol=1e-5,
                    )
                    reason = (
                        accept_sample(
                            env,
                            robot,
                            face,
                            pose,
                            q,
                            grasp,
                            bounds,
                            geoms,
                            details,
                        )
                        if ok
                        else "IK"
                    )
                else:
                    candidate = inverse_object_pose(
                        fixed_ee, offset, delta, pose, details
                    )
                    reason = accept_sample(
                        env,
                        robot,
                        face,
                        candidate,
                        anchor_q,
                        offset,
                        bounds,
                        geoms,
                        details,
                    )
                if reason:
                    rejected[reason] += 1
                else:
                    # Full qpos preserves both robot and object/door configuration.
                    frames.append(env.data.qpos.copy())
                if (count + 1) % 25 == 0:
                    print(
                        f"  {mode.upper()}: {count + 1}/{samples} checked; "
                        f"{len(frames)} accepted",
                        flush=True,
                    )
            print(
                f"{mode.upper()} ready: {len(frames)} accepted, rejected={rejected}"
            )
            if not frames:
                raise RuntimeError(
                    f"No valid {mode.upper()} samples in {samples} attempts; "
                    "try increasing --samples or restarting for a new anchor"
                )
            buffers[mode] = frames
    finally:
        env.data.qpos[:] = anchor_qpos
        mujoco.mj_forward(env.model, env.data)
    return buffers


def display_frame(env, qpos):
    """Update only rendering transforms for a previously validated configuration."""
    env.data.qpos[:] = qpos
    mujoco.mj_kinematics(env.model, env.data)
    mujoco.mj_comPos(env.model, env.data)
    mujoco.mj_camlight(env.model, env.data)


def play_samples(env, buffers, interval):
    """Loop through each prepared mode without solving or validating new samples."""
    modes = list(buffers)
    original_flags = env.model.opt.disableflags
    # The passive viewer can run its own forward update when synchronizing UI.
    # Disable contacts only after validation so that update cannot redo checks.
    env.model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    try:
        display_frame(env, buffers[modes[0]][0])
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            set_camera(viewer)
            mode_index, frame_index = 0, 0
            next_sample = time.monotonic()
            while viewer.is_running():
                now = time.monotonic()
                if now >= next_sample:
                    mode = modes[mode_index]
                    if frame_index == 0:
                        print(
                            f"Playing {mode.upper()}: {len(buffers[mode])} validated samples"
                        )
                    with viewer.lock():
                        display_frame(env, buffers[mode][frame_index])
                    frame_index += 1
                    if frame_index == len(buffers[mode]):
                        frame_index = 0
                        mode_index = (mode_index + 1) % len(modes)
                        time.sleep(1.0)
                    # Schedule before rendering, without bursts after a pause.
                    next_sample = now + interval
                viewer.sync()
                time.sleep(
                    min(1 / 120, max(0, next_sample - time.monotonic()))
                )
    finally:
        env.model.opt.disableflags = original_flags


def main(args):
    env, robot = load_env_and_robot(args.env, args.robot, False, False, False)
    try:
        robot.set_joint_qpos(robot.home_pos)
        tasks = reference_tasks(env)
        metadata_path = (
            Path(get_data_folder(args.env, args.robot)) / "task_set.tcr.json"
        )
        if metadata_path.exists():
            with metadata_path.open() as stream:
                cells = json.load(stream)["cells"]
        else:
            print(
                "No saved TSR metadata; constructing the environment's TSR bounds."
            )
            env.construct_tcr()
            cells = env.grasp_details["tcr_metadata"]["cells"]
        solver = get_ik_solver(robot)
        # Keep posture weak enough to meet the inverse TSR's tight anchor tolerance.
        solver.posture_task.set_cost(1e-6)
        print(
            "Preparing TSR interiors and corners; rejecting penetrations over 1 micron."
        )
        print(
            "Sample checks do not certify the entire continuous TSR boundary."
        )
        print(
            "Finding a collision-free anchor before opening the viewer...",
            flush=True,
        )
        anchor = find_anchor(env, robot, solver, tasks, cells)
        modes = ("tsr", "tcr") if args.mode == "both" else (args.mode,)
        buffers = precompute_samples(
            env, robot, solver, anchor, modes, args.samples
        )
        print(
            "Preparation complete. Opening viewer; close it to stop playback."
        )
        play_samples(env, buffers, args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        robot.close()


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=ENVS, default="allstable")
    parser.add_argument("--robot", choices=ROBOTS, default="panda")
    parser.add_argument(
        "--mode", choices=("tsr", "tcr", "both"), default="both"
    )
    parser.add_argument(
        "--interval",
        type=positive_seconds,
        default=0.05,
        help="Seconds between samples (default: 0.05)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=300,
        help="Candidates to solve/check per mode before playback (default: 100)",
    )
    args = parser.parse_args()
    if args.samples <= 0:
        parser.error("--samples must be positive")
    return args


if __name__ == "__main__":
    os.chdir(REPO)
    main(parse_arguments())
