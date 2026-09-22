"""Timed conveyor picking with dynamic robot control and frictional grasps.

The belt runs along world +Y. Detection is simulated ground truth. Planning uses
an independent model: planners never write the live robot's joint positions.
"""

from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import METHODS, REPO, add_dataset_args, positive_int
from geometry.trajectory import SplineTrajectory

COLORS = ((1, 0.15, 0.15, 1), (0.15, 1, 0.15, 1), (0.15, 0.3, 1, 1))
# Three non-overlapping slots inside the existing 9 cm square container.
SLOTS = np.array(
    [[0.627, 0.708, 0.775], [0.673, 0.708, 0.775], [0.65, 0.754, 0.775]]
)
RADIUS = np.sqrt(2) * 0.015


def parse_arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    add_dataset_args(p)
    p.set_defaults(env="conveyor")
    p.add_argument("--methods", nargs="+", choices=METHODS, default=["grr"])
    p.add_argument("--objects", type=positive_int, default=9)
    p.add_argument(
        "--speed", type=float, default=0.15, help="Belt speed in m/s along +Y"
    )
    p.add_argument("--spawn-interval", type=float, default=4.0)
    p.add_argument(
        "--grasp-region",
        nargs=4,
        type=float,
        default=[0.53, 0.76, -0.05, 0.25],
        metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
    )
    p.add_argument(
        "--approach-seconds",
        type=float,
        default=0.7,
        help="Fixed approach duration; departure is scheduled to meet the intercept",
    )
    p.add_argument("--transfer-seconds", type=float, default=0.7)
    p.add_argument(
        "--return-seconds",
        type=float,
        default=0.4,
        help="Empty-arm return duration after release; loaded transfers use --transfer-seconds",
    )
    p.add_argument("--settle-seconds", type=float, default=0.1)
    p.add_argument(
        "--placement-align-seconds",
        type=float,
        default=0.1,
        help="Fixed duration of held-object alignment above its slot before release",
    )
    p.add_argument(
        "--base-lift",
        type=float,
        default=0.05,
        help="Raise the entire robot mounting base in world Z (m), preserving library joint poses",
    )
    p.add_argument(
        "--home-lift",
        type=float,
        default=0.0,
        help="Optional additional home-only lift (m); normally use --base-lift",
    )
    p.add_argument("--gripper-seconds", type=float, default=0.10)
    p.add_argument("--descent-seconds", type=float, default=0.10)
    p.add_argument(
        "--lift-height",
        type=float,
        default=0.10,
        help="Legacy compatibility option; lift now returns to the stored TCR height",
    )
    p.add_argument("--lift-seconds", type=float, default=0.5)
    p.add_argument("--timeout", type=float, default=3.0)
    p.add_argument("--library-size", type=positive_int)
    p.add_argument("--library-k", type=positive_int, default=5)
    p.add_argument("--headless", action="store_true")
    p.add_argument(
        "--realtime",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Pace physics to wall time (default on with viewer, off headless)",
    )
    p.add_argument(
        "--placement-cache",
        type=Path,
        help="NPZ cache of three fixed home-to-slot paths",
    )
    p.add_argument(
        "--output", type=Path, default=REPO / "results/conveyor.json"
    )
    args = p.parse_args(argv)
    if args.realtime is None:
        args.realtime = not args.headless
    if args.robot not in ("panda", "fetch") or args.env != "conveyor":
        p.error(
            "the calibrated conveyor supports --robot panda or fetch and --env conveyor"
        )
    for name in (
        "speed",
        "spawn_interval",
        "approach_seconds",
        "transfer_seconds",
        "return_seconds",
        "settle_seconds",
        "placement_align_seconds",
        "timeout",
        "gripper_seconds",
        "lift_height",
        "lift_seconds",
        "descent_seconds",
    ):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            p.error(f"--{name.replace('_', '-')} must be finite and positive")
    for name in ("base_lift", "home_lift"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            p.error(
                f"--{name.replace('_', '-')} must be finite and nonnegative"
            )
    x0, x1, y0, y1 = args.grasp_region
    if not all(np.isfinite(args.grasp_region)) or not (
        0.49 <= x0 < x1 <= 0.81 and -0.40 <= y0 < y1 <= 0.40
    ):
        p.error(
            "grasp region must lie inside belt bounds x=[.49,.81], y=[-.40,.40]"
        )
    if min(x1 - x0, y1 - y0) <= 2 * RADIUS:
        p.error("grasp region must fit the complete object footprint")
    if args.spawn_interval * args.speed <= y1 - y0 + 2 * RADIUS:
        p.error(
            "spawn spacing must exceed grasp-region length plus object diameter"
        )
    for name in ("data_root", "output", "placement_cache"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    return args


def isolated(index, poses, start, end, region, speed):
    """Conservative swept-footprint exclusion over the entire picking window."""
    x0, x1, y0, y1 = region
    for j, (x, spawn, _) in enumerate(poses):
        if j == index or not x0 - RADIUS <= x <= x1 + RADIUS or spawn > end:
            continue
        lo = -0.42 + speed * (max(start, spawn) - spawn)
        hi = -0.42 + speed * (end - spawn)
        if lo <= y1 + RADIUS and hi >= y0 - RADIUS:
            return False
    return True


def earliest_isolated_start(
    index, poses, start, end, region, speed, excluded=()
):
    """Wait for upstream misses to leave instead of rejecting every follower."""
    x0, x1, y0, y1 = region
    for j, (x, spawn, _) in enumerate(poses):
        if j == index or j in excluded or not x0 - RADIUS <= x <= x1 + RADIUS:
            continue
        enters = max(spawn, spawn + (y0 - RADIUS + 0.42) / speed)
        leaves = spawn + (y1 + RADIUS + 0.42) / speed
        if leaves < start or enters > end:
            continue
        if leaves >= end:
            return None
        start = max(start, leaves + 0.002)
    return start


def approach_timing(now, arrival, descent_seconds, motion_seconds):
    """Keep motion time fixed; schedule departure separately to avoid hovering."""
    available = arrival - descent_seconds - now
    if available < motion_seconds + 0.02:
        return None
    return motion_seconds


def interception_y(
    now, detected_y, ready_at, region, speed, approach, descent, closing
):
    """Earliest reachable center, with the whole parcel in-region through closure."""
    lower = region[2] + RADIUS + speed * closing / 2
    upper = region[3] - RADIUS - speed * closing / 2
    lead = approach + descent + closing / 2 + 0.02
    target = max(lower, detected_y + speed * (ready_at - now + lead))
    return target if target <= upper else None


def report_skip(method, record, status):
    record["status"] = status
    detail = record.get("failure_detail", "")
    print(
        f"[{method}] object {record['object']} ({record['color']}): {status} {detail}".rstrip(),
        flush=True,
    )


class ExecutionSpline:
    """Arc-length spline with quintic timing: zero velocity/acceleration at rest."""

    def __init__(self, path, seconds):
        points = np.asarray(path, dtype=float)
        keep = np.r_[
            True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-9
        ]
        points = points[keep]
        if len(points) == 1:
            points = np.repeat(points, 2, axis=0)
            knots = np.array([0.0, 1.0])
        else:
            knots = np.r_[
                0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))
            ]
            knots /= knots[-1]
        self.spline = SplineTrajectory(points, knots)
        self.seconds = seconds

    def state(self, elapsed):
        t = np.clip(elapsed / self.seconds, 0, 1)
        u = np.clip(10 * t**3 - 15 * t**4 + 6 * t**5, 0, 1)
        du = 30 * t**2 * (1 - t) ** 2 / self.seconds
        ddu = 60 * t * (1 - t) * (1 - 2 * t) / self.seconds**2
        q = self.spline.position(u)
        v = self.spline.velocity(u) * du
        a = self.spline.acceleration(u) * du**2 + self.spline.velocity(u) * ddu
        return q, v, a


def prepare_path(validator, path, start, goal):
    """Retarget library home, shortcut/smooth, then collision-check the spline."""
    from ompl import geometric as og
    from experiments.benchmark import validate_path

    if path is None or len(path) < 2:
        return None
    path = np.asarray(path, dtype=float).copy()
    if (
        path.ndim != 2
        or path.shape[1] != len(start)
        or not np.isfinite(path).all()
    ):
        return None
    # Existing CoAd libraries start at the old home. Fade the home correction
    # out along the path, preserving the original TCR goal, then revalidate.
    progress = np.r_[
        0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))
    ]
    if progress[-1] > 0:
        progress /= progress[-1]
    path += (1 - np.minimum(progress[:, None] / 0.5, 1)) * (
        np.asarray(start) - path[0]
    )
    if not validate_path(validator, path, start, goal):
        return None
    native = validator.np_path_to_path_geometric(path)
    original = path.copy()
    smoother = og.PathSimplifier(validator.si)
    if hasattr(smoother, "ropeShortcutPath"):
        smoother.ropeShortcutPath(native)
    else:
        smoother.shortcutPath(native)
    smoother.smoothBSpline(native)
    path = np.array(
        [[s[j] for j in range(len(start))] for s in native.getStates()]
    )
    if validate_spline(validator, path, start, goal):
        return path
    # Smoothing may cut too close to the object during the final descent.
    return (
        original if validate_spline(validator, original, start, goal) else None
    )


def validate_spline(validator, path, start, goal):
    from experiments.benchmark import validate_path

    spline = ExecutionSpline(path, 1.0)
    # Check interpolated segments as well as knots; cubic overshoot can collide.
    times = np.unique(np.r_[np.linspace(0, 1, 1001), spline.spline.t_states])
    samples = spline.spline.trajectory(times)
    return validate_path(validator, samples, start, goal)


def raise_base(args, env, robot):
    """Experiment-local mount offset shared by planning and live physics models."""
    position = np.asarray(env.env_details["robot_pos"], dtype=float).copy()
    position[2] += args.base_lift
    env.env_details["robot_pos"] = position
    robot.teleport_base(position, env.env_details["robot_quat"])


def raise_home(args, env, robot, validator):
    from coad.mink_ik import get_ik_solver

    target = robot.get_ee_pose()
    target[2] += args.home_lift
    ok, home = get_ik_solver(robot).solve(
        target, current=robot.get_joint_qpos()
    )
    if not ok or not validator.validity_checker(
        validator.numpy_to_state(home)
    ):
        raise RuntimeError("Could not find a collision-free raised home pose")
    env.home_qpos = np.asarray(home).copy()
    robot.set_joint_qpos(home)
    return env.home_qpos.copy()


def plan_pick(bench, method, sample, reference):
    """The selected method is the complete online planning operation."""
    started = time.perf_counter()
    path, seconds, target = bench.solve(method, sample, reference)
    return (
        path,
        target,
        dict(
            planning_seconds=seconds,
            planning_wall_seconds=time.perf_counter() - started,
        ),
    )


class Simulation:
    """Dynamic parcels: conveyor traction and finger contacts provide all motion."""

    def __init__(self, args, env, poses):
        import mujoco
        from coad.robot import Panda, FetchArm

        self.mj, self.args, self.poses = mujoco, args, poses
        scene = env.build_xml(
            "configs/scenes/conveyor/scene_conveyor.yaml",
            parent_body_name="scene_conveyor",
        )
        objects = []
        for i in range(len(poses)):
            rgba = " ".join(map(str, COLORS[i % 3]))
            objects.append(
                f'<body name="parcel_{i}" gravcomp="1" pos="0 0 {-2-i}">'
                f'<freejoint name="parcel_joint_{i}"/>'
                f'<geom name="parcel_geom_{i}" type="box" size=".015 .015 .05" mass=".04" '
                f'rgba="{rgba}" friction="1.5 .02 .002" condim="6" '
                'solref=".004 1" solimp=".95 .99 .001" contype="1" conaffinity="1"/></body>'
            )
        x0, x1, y0, y1 = args.grasp_region
        objects.append(
            f'<site name="grasp_region" type="box" pos="{(x0+x1)/2} {(y0+y1)/2} .723" '
            f'size="{(x1-x0)/2} {(y1-y0)/2} .001" rgba="1 1 0 .2"/>'
        )
        for slot, color in zip(SLOTS, COLORS):
            objects.append(
                f'<site type="box" pos="{slot[0]} {slot[1]} .727" '
                f'size=".017 .017 .001" rgba="{color[0]} {color[1]} {color[2]} .6"/>'
            )
        model, data = env.build_model(
            f"{env.robot_dir}/conveyor_live.xml", [scene, *objects]
        )
        self.robot = (Panda if args.robot == "panda" else FetchArm)(
            model, data
        )
        self.robot.teleport_base(
            env.env_details["robot_pos"], env.env_details["robot_quat"]
        )
        self.robot.set_joint_qpos(env.home_qpos)
        self.model, self.data = model, data
        from experiments.visualize_env import scene_color

        for name in set(env.env_details["collision_geoms"]):
            geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if geom >= 0:
                model.geom_rgba[geom] = np.fromstring(
                    scene_color(name), sep=" "
                )
        model.opt.timestep = 0.001
        model.opt.iterations = 100
        model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
        scalar = [
            j
            for j in range(model.njnt)
            if model.jnt_type[j]
            in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
        ]
        self.scalar_qpos = model.jnt_qposadr[scalar]
        self.scalar_dof = model.jnt_dofadr[scalar]
        self.rest = data.qpos.copy()
        self.command = env.home_qpos.copy()
        self.command_velocity = np.zeros(self.robot.n_joints)
        self.command_acceleration = np.zeros(self.robot.n_joints)
        fingers = [model.joint(name).id for name in self.robot.FINGER]
        self.finger_qpos = model.jnt_qposadr[fingers]
        self.finger_dof = model.jnt_dofadr[fingers]
        self.finger_geoms = []
        for joint in fingers:
            body = model.jnt_bodyid[joint]
            geoms = {
                g
                for g in range(model.ngeom)
                if model.geom_bodyid[g] == body and model.geom_group[g] == 3
            }
            self.finger_geoms.append(geoms)
            for geom in geoms:
                model.geom_priority[geom] = 2
                model.geom_friction[geom] = [1.5, 0.02, 0.002]
                model.geom_condim[geom] = 6
                model.geom_solref[geom] = [0.004, 1]
        self.belt_geom = model.geom("conveyor_top").id
        model.geom_priority[self.belt_geom] = 1
        model.geom_friction[self.belt_geom] = [0.01, 0.001, 0.0001]
        self.parcel_geoms = [
            model.geom(f"parcel_geom_{i}").id for i in range(len(poses))
        ]
        self.parcel_bodies = [
            model.body(f"parcel_{i}").id for i in range(len(poses))
        ]
        joints = [
            model.joint(f"parcel_joint_{i}").id for i in range(len(poses))
        ]
        self.parcel_qpos = model.jnt_qposadr[joints]
        self.parcel_dof = model.jnt_dofadr[joints]
        for geom, body in zip(self.parcel_geoms, self.parcel_bodies):
            model.geom_contype[geom] = model.geom_conaffinity[geom] = 0
            model.body_contype[body] = model.body_conaffinity[body] = 0
        self.mass_force = np.zeros(model.nv)
        self.active, self.done = set(), set()
        self.released = set()
        self.deposited = {}
        self.release_observations = {}
        self.held = (
            None  # Bookkeeping only: there is no weld or position attachment.
        )
        self.spawn_events, self.clear_events = [], []
        self.viewer = None
        if not args.headless:
            import mujoco.viewer

            self.viewer = mujoco.viewer.launch_passive(model, data)
            self.viewer.cam.lookat[:] = [0.5, 0.35, 0.8]
            self.viewer.cam.distance = 2.5
        self.wall_start = time.perf_counter()
        self.frames = 0

    def object_pose(self, index):
        adr = self.parcel_qpos[index]
        return self.data.qpos[adr : adr + 7].copy()

    def remove(self, index):
        """Lifecycle removal only (off belt or collected RGB set)."""
        adr, dof = self.parcel_qpos[index], self.parcel_dof[index]
        self.data.qpos[adr : adr + 7] = [0, 0, -2 - index, 1, 0, 0, 0]
        self.data.qvel[dof : dof + 6] = 0
        self.model.body_gravcomp[self.parcel_bodies[index]] = 1
        geom = self.parcel_geoms[index]
        self.model.geom_contype[geom] = self.model.geom_conaffinity[geom] = 0
        body = self.parcel_bodies[index]
        self.model.body_contype[body] = self.model.body_conaffinity[body] = 0
        self.active.discard(index)
        self.done.add(index)

    def update_objects(self):
        # Spawn independently of robot state. Never set an active parcel's pose.
        for i, (x, spawn, yaw) in enumerate(self.poses):
            if (
                i in self.done
                or i in self.active
                or self.data.time + 1e-9 < spawn
            ):
                continue
            adr, dof = self.parcel_qpos[i], self.parcel_dof[i]
            self.data.qpos[adr : adr + 7] = [
                x,
                -0.42,
                0.7702,
                np.cos(yaw / 2),
                0,
                0,
                np.sin(yaw / 2),
            ]
            self.data.qvel[dof : dof + 6] = [0, self.args.speed, 0, 0, 0, 0]
            geom = self.parcel_geoms[i]
            self.model.geom_contype[geom] = self.model.geom_conaffinity[
                geom
            ] = 1
            body = self.parcel_bodies[i]
            self.model.body_contype[body] = self.model.body_conaffinity[
                body
            ] = 1
            self.model.body_gravcomp[self.parcel_bodies[i]] = 0
            self.active.add(i)
            self.spawn_events.append(
                dict(object=i, scheduled=spawn, actual=float(self.data.time))
            )
            self.mj.mj_forward(self.model, self.data)
        for i in list(self.active):
            pos = self.object_pose(i)[:3]
            if (
                i != self.held
                and i not in self.deposited
                and i not in self.released
                and (pos[2] < 0.5 or (pos[1] > 0.45 and pos[2] < 0.8))
            ):
                self.remove(i)

    def finger_contacts(self, index):
        geom = self.parcel_geoms[index]
        touching = set()
        for contact in self.data.contact:
            other = (
                contact.geom2
                if contact.geom1 == geom
                else contact.geom1 if contact.geom2 == geom else -1
            )
            if contact.dist < 0.001:
                for side, geoms in enumerate(self.finger_geoms):
                    if other in geoms:
                        touching.add(side)
        return len(touching)

    def step(self):
        if self.viewer is not None and not self.viewer.is_running():
            raise KeyboardInterrupt
        self.update_objects()
        target = self.rest.copy()
        target[self.robot.joint_qpos_ids] = self.command
        velocity = np.zeros(self.model.nv)
        velocity[self.robot.joint_dof_ids] = self.command_velocity
        acceleration = np.zeros(self.model.nv)
        acceleration[self.scalar_dof] = 2500 * (
            target[self.scalar_qpos] - self.data.qpos[self.scalar_qpos]
        )
        acceleration[self.scalar_dof] += 100 * (
            velocity[self.scalar_dof] - self.data.qvel[self.scalar_dof]
        )
        acceleration[self.robot.joint_dof_ids] += self.command_acceleration
        if getattr(self, "cartesian_command", None) is not None:
            position, rotation, desired_velocity, desired_acceleration = (
                self.cartesian_command
            )
            site = self.model.site(self.robot.ee_name).id
            jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
            self.mj.mj_jacSite(self.model, self.data, jp, jr, site)
            jac = np.vstack((jp, jr))[:, self.robot.joint_dof_ids]
            actual_rotation = self.data.site_xmat[site].reshape(3, 3)
            orientation_error = 0.5 * sum(
                (
                    np.cross(actual_rotation[:, k], rotation[:, k])
                    for k in range(3)
                ),
                np.zeros(3),
            )
            error = np.r_[
                position - self.data.site_xpos[site], orientation_error
            ]
            qvelocity = self.data.qvel[self.robot.joint_dof_ids]
            previous = getattr(self, "cartesian_jacobian", jac)
            bias = ((jac - previous) / self.model.opt.timestep) @ qvelocity
            inverse = jac.T @ np.linalg.solve(
                jac @ jac.T + 1e-8 * np.eye(6), np.eye(6)
            )
            task_acceleration = (
                desired_acceleration
                + 2500 * error
                + 100 * (desired_velocity - jac @ qvelocity)
                - bias
            )
            posture = acceleration[self.robot.joint_dof_ids].copy()
            acceleration[self.robot.joint_dof_ids] = (
                inverse @ task_acceleration
                + (np.eye(self.robot.n_joints) - inverse @ jac) @ posture
            )
            self.cartesian_jacobian = jac

        self.mj.mj_mulM(self.model, self.data, self.mass_force, acceleration)
        self.data.qfrc_applied[:] = 0
        self.data.qfrc_applied[self.scalar_dof] = (
            self.mass_force + self.data.qfrc_bias - self.data.qfrc_passive
        )[self.scalar_dof]
        # Force-limited gripper squeeze; finger contacts, not a weld, carry the box.
        self.data.qfrc_applied[self.finger_dof] = np.clip(
            1200
            * (self.rest[self.finger_qpos] - self.data.qpos[self.finger_qpos])
            - 8 * self.data.qvel[self.finger_dof],
            -20,
            20,
        )
        self.data.xfrc_applied[:] = 0
        supported = set()
        lookup = {g: i for i, g in enumerate(self.parcel_geoms)}
        for contact in self.data.contact:
            other = (
                contact.geom2
                if contact.geom1 == self.belt_geom
                else contact.geom1 if contact.geom2 == self.belt_geom else -1
            )
            if other in lookup:
                supported.add(lookup[other])
        # A tangential force models belt traction, only while actually supported.
        for i in supported:
            body, dof = self.parcel_bodies[i], self.parcel_dof[i]
            mass = self.model.body_mass[body]
            force = mass * (
                80 * (self.args.speed - self.data.qvel[dof + 1]) + 0.01 * 9.81
            )
            self.data.xfrc_applied[body, 1] = np.clip(
                force, -mass * 9.81, mass * 9.81
            )
        self.mj.mj_step(self.model, self.data)
        self.mj.mj_forward(self.model, self.data)
        if not np.isfinite(self.data.qpos).all():
            raise RuntimeError("Non-finite robot state")
        # Finger/object contacts are expected. Robot/scene penetration is not.
        parcels = set(self.parcel_geoms)
        for contact in self.data.contact:
            if (
                contact.dist < -0.003
                and contact.geom1 not in parcels
                and contact.geom2 not in parcels
            ):
                if (
                    contact.geom1 in self.robot.robot_geoms
                    or contact.geom2 in self.robot.robot_geoms
                ):
                    raise RuntimeError(
                        "Robot penetrated scene geometry during physical execution"
                    )
        self.frames += 1
        if self.frames % 20 == 0 and self.viewer is not None:
            self.viewer.sync()
        if self.args.realtime:
            time.sleep(
                max(0, self.wall_start + self.data.time - time.perf_counter())
            )

    def wait_until(self, deadline):
        while self.data.time < deadline:
            self.step()

    def plan_while_running(self, executor, function, *args):
        """Keep physics/rendering on the main thread during background planning.

        Even fast headless runs use wall-clock pacing during planning so CPU
        simulation throughput does not determine the available planning time.
        """
        start_sim = float(self.data.time)
        start_wall = time.perf_counter()
        future = executor.submit(function, *args)
        while not future.done():
            if self.data.time < start_sim + time.perf_counter() - start_wall:
                self.step()
            else:
                time.sleep(0.001)
        # Account for the last polling interval, not a post-planning time jump.
        self.wait_until(start_sim + time.perf_counter() - start_wall)
        return future.result()

    def execute(self, path, seconds):
        trajectory = ExecutionSpline(path, seconds)
        begin = self.data.time
        while self.data.time < begin + seconds:
            self.command, self.command_velocity, self.command_acceleration = (
                trajectory.state(self.data.time - begin)
            )
            self.step()
        self.command = np.asarray(path[-1]).copy()
        self.command_velocity[:] = 0
        self.command_acceleration[:] = 0

    def vertical(self, height, seconds):
        """Physical Cartesian servo, not a new joint-space plan or IK goal."""
        site = self.model.site(self.robot.ee_name).id
        goal = self.data.site_xpos[site].copy()
        goal[2] = height
        self.move_tool(goal, seconds)

    def move_tool(self, goal, seconds):
        """Translate the tool with fixed-duration spline timing and physical torque."""
        site = self.model.site(self.robot.ee_name).id
        start = self.data.site_xpos[site].copy()
        rotation = self.data.site_xmat[site].reshape(3, 3).copy()
        curve = ExecutionSpline(np.array([start, goal]), seconds)
        begin = self.data.time
        while self.data.time < begin + seconds:
            position, velocity, acceleration = curve.state(
                self.data.time - begin
            )
            self.cartesian_command = (
                position,
                rotation,
                np.r_[velocity, np.zeros(3)],
                np.r_[acceleration, np.zeros(3)],
            )
            self.step()
        self.cartesian_command = None
        if hasattr(self, "cartesian_jacobian"):
            del self.cartesian_jacobian
        self.command = self.robot.get_joint_qpos().copy()
        self.command_velocity[:] = 0
        self.command_acceleration[:] = 0

    def gripper(self, closed):
        """Open or close the fingers while holding the arm at fixed joint targets."""
        start = self.data.qpos[self.finger_qpos].copy()
        target = (
            np.zeros(len(start))
            if closed
            else np.asarray(self.robot.FINGER_OPEN)
        )
        begin = self.data.time
        while self.data.time < begin + self.args.gripper_seconds:
            elapsed = self.data.time - begin
            u = elapsed / self.args.gripper_seconds
            blend = 10 * u**3 - 15 * u**4 + 6 * u**5
            self.rest[self.finger_qpos] = start + (target - start) * blend
            self.step()
        self.rest[self.finger_qpos] = target
        self.command_velocity[:] = 0
        self.command_acceleration[:] = 0

    def reached(self, q):
        return np.max(np.abs(self.robot.get_joint_qpos() - q)) < 0.035

    def confirm_grasp(self, index):
        """Record a candidate grasp only; no forces, constraints or pose edits."""
        if self.finger_contacts(index) == 2:
            self.held = index
            return True
        return False

    def release(self, index):
        """Verify a physical drop and collect the container after a complete RGB set."""
        self.held = None
        self.released.add(index)
        self.wait_until(self.data.time + 0.3)
        self.release_observations[index] = self.object_pose(index).tolist()
        pos = self.object_pose(index)[:3]
        if (
            np.linalg.norm(pos[:2] - SLOTS[index % 3, :2]) > 0.025
            or not 0.755 < pos[2] < 0.81
        ):
            return False
        self.deposited[index] = pos.copy()
        if {i % 3 for i in self.deposited} == {0, 1, 2}:
            collected = sorted(self.deposited)
            for i in collected:
                self.remove(i)
            self.deposited.clear()
            self.clear_events.append(
                dict(time=float(self.data.time), objects=collected)
            )
        return True

    def close(self):
        if self.viewer is not None:
            self.viewer.close()


def placements(args, env, robot, validator, home):
    """Generate/cache three fixed paths before the belt starts; validate on load."""
    from coad.mink_ik import get_ik_solver

    env.move_object([0, 0, -2, 0])
    # Fixed top-down tool pose; physical final alignment compensates grasp offset.
    targets = np.c_[
        SLOTS[:, :2],
        np.full(3, 0.85),
        np.tile([0, np.sqrt(0.5), np.sqrt(0.5), 0], (3, 1)),
    ]
    cache = (
        args.placement_cache
        or args.data_root / f"conveyor_{args.robot}/placement_paths.npz"
    )
    paths = []
    if cache.exists():
        with np.load(cache, allow_pickle=False) as saved:
            if (
                saved.get("schema", 0) == 6
                and np.array_equal(
                    saved["base_pos"], env.env_details["robot_pos"]
                )
                and np.array_equal(
                    saved["base_quat"], env.env_details["robot_quat"]
                )
                and np.allclose(saved["home"], home)
                and np.array_equal(saved["targets"], targets)
            ):
                paths = [saved[f"path_{i}"] for i in range(3)]
            else:
                print(
                    f"Rebuilding placement cache for raised home and spline execution: {cache}",
                    flush=True,
                )
    if not paths:
        solver = get_ik_solver(
            robot, env_collision_geoms=env.env_details["collision_geoms"]
        )
        for target in targets:
            path = None
            hover = target.copy()
            hover[2] = 1.05
            for attempt in range(20):
                ok, q = solver.solve(
                    hover, current=home, random_current=attempt > 0
                )
                if not ok:
                    continue
                candidate, _, _ = validator.plan(
                    home,
                    q,
                    timeout=args.timeout,
                    smooth_path=True,
                    num_waypoints=200,
                    benchmark=True,
                )
                path = prepare_path(validator, candidate, home, q)
                if path is not None:
                    descent = list(path)
                    for fraction in np.linspace(0, 1, 21)[1:]:
                        point = target.copy()
                        point[:3] = hover[:3] + fraction * (
                            target[:3] - hover[:3]
                        )
                        ok, q = solver.solve(point, current=descent[-1])
                        if not ok:
                            break
                        descent.append(q)
                    if ok and validate_spline(
                        validator, descent, home, descent[-1]
                    ):
                        path = np.asarray(descent)
                        break
                    path = None
            if path is None:
                raise RuntimeError(
                    f"Unable to preplan placement for {target[:3]}"
                )
            paths.append(path)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache,
            schema=6,
            base_pos=env.env_details["robot_pos"],
            base_quat=env.env_details["robot_quat"],
            home=home,
            targets=targets,
            **{f"path_{i}": path for i, path in enumerate(paths)},
        )
    for path, target in zip(paths, targets):
        if not validate_spline(validator, path, home, path[-1]):
            raise ValueError(f"Invalid placement path in {cache}")
        robot.set_joint_qpos(path[-1])
        actual = robot.get_ee_pose()
        if (
            np.linalg.norm(actual[:3] - target[:3]) > 0.005
            or abs(np.dot(actual[3:], target[3:])) < 0.999
        ):
            raise ValueError(f"Placement endpoint mismatch in {cache}")
    robot.set_joint_qpos(home)
    return paths


def run_method(args, method):
    from coad.utils import load_env_and_robot, set_seed
    from experiments.benchmark import Benchmark

    set_seed(args.seed)
    env, robot = load_env_and_robot(
        "conveyor",
        args.robot,
        visualize=False,
        using_swept_volume=False,
        compute_tcr=False,
    )
    sim = None
    records = []
    executor = ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="conveyor-planner"
    )
    try:
        raise_base(args, env, robot)
        bench = Benchmark(args, env, robot)
        home = (
            raise_home(args, env, robot, bench.validator)
            if args.home_lift
            else bench.home.copy()
        )
        # Libraries retain their original start. Connect the raised execution
        # home offline instead of deforming every method's returned path.
        if args.home_lift:
            bridge, _, _ = bench.validator.plan(
                home,
                bench.home,
                timeout=args.timeout,
                smooth_path=True,
                num_waypoints=50,
                benchmark=True,
            )
            if bridge is None:
                raise RuntimeError("Could not prepare raised-home connection")
            bridge = np.asarray(bridge)
        else:
            bridge = home[None, :]
        bench.setup([method])
        drop_paths = placements(args, env, robot, bench.validator, home)
        rng = np.random.default_rng(args.seed)
        x0, x1, y0, y1 = args.grasp_region
        poses = [
            (
                rng.uniform(x0 + RADIUS, x1 - RADIUS),
                i * args.spawn_interval,
                rng.uniform(-np.pi / 2, np.pi / 2),
            )
            for i in range(args.objects)
        ]
        sim = Simulation(args, env, poses)
        planning_budget = 0.005
        for i, (x, spawn, yaw) in enumerate(poses):
            record = dict(
                object=i, color="RGB"[i % 3], spawn=spawn, status="pending"
            )
            records.append(record)
            sim.wait_until(spawn)
            sim.update_objects()
            if i not in sim.active:
                report_skip(method, record, "missed_object")
                continue
            if any(previous % 3 == i % 3 for previous in sim.deposited):
                report_skip(method, record, "slot_occupied")
                continue
            # Replan downstream if measured computation consumes the intercept window.
            record["intercept_attempts"] = 0
            while True:
                detected = [*sim.object_pose(i)[:3], yaw]
                now = float(sim.data.time)
                record.update(detection=detected, detection_time=now)
                # Spend remaining slack on planning even when it is smaller than
                # the estimate; only actual elapsed time can invalidate the plan.
                upper = y1 - RADIUS - args.speed * args.gripper_seconds / 2
                latest_ready = (
                    now
                    + (upper - detected[1]) / args.speed
                    - (
                        args.approach_seconds
                        + args.descent_seconds
                        + args.gripper_seconds / 2
                        + 0.02
                    )
                )
                ready_at = min(
                    now + planning_budget, max(now, latest_ready - 0.003)
                )
                while True:
                    intercept_y = interception_y(
                        now,
                        detected[1],
                        ready_at,
                        args.grasp_region,
                        args.speed,
                        args.approach_seconds,
                        args.descent_seconds,
                        args.gripper_seconds,
                    )
                    if intercept_y is None:
                        report_skip(method, record, "missed_deadline")
                        break
                    arrival = now + (intercept_y - detected[1]) / args.speed
                    clear_at = earliest_isolated_start(
                        i,
                        poses,
                        now,
                        arrival + args.gripper_seconds + args.lift_seconds,
                        args.grasp_region,
                        args.speed,
                        sim.done | sim.released,
                    )
                    if clear_at is None:
                        report_skip(method, record, "region_occupied")
                        break
                    if clear_at <= ready_at:
                        break
                    ready_at = clear_at
                if record["status"] != "pending":
                    break
                closing_at = arrival - 0.5 * args.gripper_seconds
                sample = np.array([detected[0], intercept_y, 0.77, yaw])
                record.update(
                    detection=detected,
                    detection_time=now,
                    predicted_pose=sample.tolist(),
                    arrival=arrival,
                    region_clear_at=clear_at,
                    intercept_attempts=record["intercept_attempts"] + 1,
                )
                key = bench.reference_index.query_point(sample)
                if key is None or bench.reference_map[key][1] is None:
                    report_skip(method, record, "outside_library")
                    break
                path, target, diagnostics = sim.plan_while_running(
                    executor,
                    plan_pick,
                    bench,
                    method,
                    sample.copy(),
                    bench.reference_map[key][1],
                )
                record.update(diagnostics)
                planning_budget = max(
                    0.002, diagnostics["planning_wall_seconds"] * 1.25
                )
                if path is None or len(path) < 2:
                    report_skip(method, record, "planning_failed")
                    break
                # Keep the method output and its stored configuration-space goal
                # unchanged. Only prepend the fixed, offline home connection.
                path = np.vstack((bridge[:-1], np.asarray(path)))
                sim.wait_until(clear_at)
                approach_seconds = approach_timing(
                    sim.data.time,
                    closing_at,
                    args.descent_seconds,
                    args.approach_seconds,
                )
                if approach_seconds is not None:
                    break
            if record["status"] != "pending":
                continue
            record["approach_execution_seconds"] = approach_seconds
            departure = (
                closing_at - args.descent_seconds - approach_seconds - 0.02
            )
            record["wait_at_home_seconds"] = max(
                0.0, departure - sim.data.time
            )
            sim.wait_until(departure)
            record["approach_started_at"] = float(sim.data.time)
            sim.execute(np.asarray(path, dtype=float), approach_seconds)
            record["wait_at_grasp_seconds"] = max(
                0.0, closing_at - args.descent_seconds - sim.data.time
            )
            sim.wait_until(closing_at - args.descent_seconds)
            robot.set_joint_qpos(target)
            ee_target = robot.get_ee_pose()
            # TCRs may vary tool pose; check actual tool position against planned pose.
            ee_actual = sim.robot.get_ee_pose()
            if (
                not sim.reached(target)
                or np.linalg.norm(ee_actual[:3] - ee_target[:3]) > 0.015
                or np.linalg.norm(ee_actual[:2] - ee_target[:2]) > 0.015
            ):
                report_skip(method, record, "tracking_failed")
                sim.execute(path[::-1], args.transfer_seconds)
                sim.wait_until(sim.data.time + args.settle_seconds)
                if not sim.reached(home):
                    raise RuntimeError(
                        "Tracking recovery failed to reach home"
                    )
                continue
            else:
                record["tool_before_descent"] = (
                    sim.robot.get_ee_pose().tolist()
                )
                sim.vertical(0.795, args.descent_seconds)
                record["tool_after_descent"] = sim.robot.get_ee_pose().tolist()
                record["closing_started_at"] = float(sim.data.time)
                record["object_at_closing_start"] = sim.object_pose(i).tolist()
                sim.gripper(closed=True)
                record["tool_after_closure"] = sim.robot.get_ee_pose().tolist()
                record["gripper_closed_at"] = float(sim.data.time)
                record["object_at_closing_end"] = sim.object_pose(i).tolist()
                record["finger_contacts"] = sim.finger_contacts(i)
                record["status"] = (
                    "grasp_candidate"
                    if sim.confirm_grasp(i)
                    else "grasp_failed"
                )
            # Lift the payload clear of the belt, then return to the raised home.
            sim.vertical(float(ee_target[2]), args.lift_seconds)
            record["object_after_lift"] = sim.object_pose(i).tolist()
            if sim.held == i:
                if sim.object_pose(i)[2] < 0.78 or sim.finger_contacts(i) < 2:
                    record["status"] = "grasp_failed"
                    sim.held = None
                else:
                    record["status"] = "picked"
                    record["picked_at"] = float(sim.data.time)
            sim.execute(path[::-1], args.transfer_seconds)
            sim.wait_until(sim.data.time + args.settle_seconds)
            if not sim.reached(home):
                raise RuntimeError(
                    "Failed to return home; stopping instead of executing an invalid start"
                )
            if sim.held == i and (
                sim.finger_contacts(i) < 2
                or np.linalg.norm(
                    sim.object_pose(i)[:3] - sim.robot.get_ee_pose()[:3]
                )
                > 0.12
            ):
                record["status"] = "dropped_during_transport"
                sim.held = None
            if sim.held is not None:
                drop = drop_paths[i % 3]
                sim.execute(drop, args.transfer_seconds)
                sim.wait_until(sim.data.time + args.settle_seconds)
                if not sim.reached(drop[-1]):
                    raise RuntimeError("Placement tracking failed")
                record["object_before_alignment"] = sim.object_pose(i).tolist()
                desired_object = SLOTS[i % 3].copy()
                desired_object[
                    2
                ] += 0.05  # Box bottom clears the rim before release.
                correction = desired_object - sim.object_pose(i)[:3]
                record["placement_alignment"] = correction.tolist()
                sim.move_tool(
                    sim.robot.get_ee_pose()[:3] + correction,
                    args.placement_align_seconds,
                )
                sim.wait_until(sim.data.time + args.settle_seconds)
                record["object_before_release"] = sim.object_pose(i).tolist()
                sim.gripper(closed=False)
                placed = sim.release(i)
                record["object_after_release"] = sim.release_observations[i]
                if sim.viewer is not None:
                    sim.viewer.sync()
                record.update(
                    status="placed" if placed else "placement_failed",
                    slot=SLOTS[i % 3].tolist(),
                    placed_at=float(sim.data.time),
                    clear_events=list(sim.clear_events),
                )
                sim.execute(
                    np.vstack((sim.robot.get_joint_qpos(), drop[::-1])),
                    args.return_seconds,
                )
                sim.wait_until(sim.data.time + args.settle_seconds)
                if not sim.reached(home):
                    raise RuntimeError("Placement return tracking failed")
            else:
                sim.gripper(closed=False)
            print(
                f"[{method}] object {i} ({record['color']}): {record['status']}",
                flush=True,
            )
        for record in records:
            event = next(
                (
                    e
                    for e in sim.spawn_events
                    if e["object"] == record["object"]
                ),
                None,
            )
            record["spawn_actual"] = event["actual"] if event else None
        return records
    except RuntimeError as error:
        if sim is None:
            raise
        # A physical execution failure stops this trial; never teleport home to resume.
        if records:
            records[-1].update(status="execution_failed", error=str(error))
        for i in range(len(records), args.objects):
            records.append(
                dict(
                    object=i,
                    color="RGB"[i % 3],
                    status="aborted_after_execution_failure",
                )
            )
        print(f"[{method}] stopped: {error}", flush=True)
        return records
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        if sim is not None:
            sim.close()
        robot.close()


def main():
    args = parse_arguments()
    os.chdir(REPO)
    from ompl import util as ou

    ou.RNG.setSeed(args.seed)
    result = dict(
        configuration={
            k: str(v) if isinstance(v, Path) else v
            for k, v in vars(args).items()
        },
        assumptions="Ground-truth detection; force-driven conveyor traction at known speed; "
        "ideal arm torque servos and force-limited fingers; free-body parcels held only by "
        "frictional contact and released under gravity. Complete RGB sets are removed as collected. "
        "Contact/friction parameters are simulation settings, not hardware calibration.",
        methods={},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for method in dict.fromkeys(args.methods):
        result["methods"][method] = run_method(args, method)
        records = result["methods"][method]
        print(
            f"[{method}] placed {sum(r['status'] == 'placed' for r in records)}/{len(records)}"
        )
        temporary = args.output.with_suffix(".tmp.json")
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
