"""Grasp setup and task-region construction."""

import os
import numpy as np
import yaml
import itertools
from geometry.pose import Pose, matrix_to_flat, flat_to_matrix
from coad.tcr import (
    TSRBounds,
    calibrate_bounds,
    inner_halfwidths,
    microwave_radii,
    goal_fits_cell,
    create_tcr_set,
    make_tew_yaw_variants,
    translational_half_extents,
    valid_grasp_yaw_offsets,
)
from coad.mink_ik import get_ik_solver
from coad.mujoco_utils import sample_qpos
from coad.robot import Panda, UR10, FetchArm, G1


class TaskRegions:
    def populate_grasp_details(
        self, alpha=0.95, yaw_buffer=6 * (np.pi / 180), grasp_type="top"
    ):

        self.grasp_details = {
            "type": grasp_type,
            "alpha": alpha,
            "yaw_buffer": yaw_buffer,
        }

    def initial_tcr_construction(
        self,
        sx,
        sy,
        sz,
        ee_z_offset,
        ee_offset,
        half_finger_length,
        half_finger_clearance,
        min_contact_overlap,
    ):

        robot_name = self.env_details["robot"]

        if self.grasp_details["type"] == "top":
            Tew = np.eye(4)
            Tew[1, 1] = -1
            Tew[2, 2] = -1

            if robot_name == "g1":
                Tew = np.eye(4)

                R = np.array(
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, 0.0, -1.0],
                        [0.0, 1.0, 0.0],
                    ]
                )

                Tew[:3, :3] = R

            Tew[2, 3] = ee_z_offset + sz / 2.0

            if self.object_details["type"] == "cylinder":
                radius = sx / 2.0

                del_geom_x = half_finger_clearance - radius
                del_geom_y = half_finger_clearance - radius

                if del_geom_x < 0.0 or del_geom_y < 0.0:
                    raise ValueError(
                        "Cylinder does not fit inside the gripper: "
                        f"radius={radius}, "
                        f"half_clearance={half_finger_clearance}"
                    )

                # Rotation about the cylinder's symmetry axis does not
                # produce a distinct grasp.
                Tews = [Tew]

            else:
                # Existing box logic begins here.

                yaw_angles, x_fits, y_fits = valid_grasp_yaw_offsets(
                    {**self.object_details, "size": [sx, sy, sz]},
                    2.0 * half_finger_clearance,
                )
                if self.environment_name == "table":
                    offsets = [np.pi / 2, -np.pi / 2]
                else:
                    offsets = [0.0, np.pi]

                Tews = make_tew_yaw_variants(Tew, offsets)

                clearance_size_x = sx
                clearance_size_y = sy

                if y_fits and not x_fits:
                    # Long object along x, narrow along y.
                    long_axis_slide = max(
                        0.0, clearance_size_x / 2.0 - min_contact_overlap
                    )
                    del_geom_x = long_axis_slide
                    del_geom_x /= 2.0
                    del_geom_y = half_finger_clearance - clearance_size_y / 2.0
                    del_geom_y /= 1.0

                elif x_fits and not y_fits:
                    # Long object along y, narrow along x.
                    long_axis_slide = max(
                        0.0, clearance_size_y / 2.0 - min_contact_overlap
                    )
                    del_geom_y = long_axis_slide
                    del_geom_x = half_finger_clearance - clearance_size_x / 2.0
                    del_geom_x /= 1.0

                elif x_fits and y_fits:
                    # Object fits both ways. Pick a consistent convention.
                    long_axis_slide = max(
                        0.0, clearance_size_y / 2.0 - min_contact_overlap
                    )
                    del_geom_y = long_axis_slide
                    del_geom_x = half_finger_clearance - clearance_size_x / 2.0

                else:
                    raise ValueError(
                        "Object does not fit the gripper in either x or y."
                    )

                del_geom_x, del_geom_y = translational_half_extents(
                    del_geom_x,
                    del_geom_y,
                )

        elif self.grasp_details["type"] == "front":
            # Canonical front grasp: approach horizontally.

            Ry90 = np.array(
                [
                    [0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0],
                    [-1.0, 0.0, 0.0],
                ]
            )

            Rz180 = np.array(
                [
                    [-1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )

            R1 = Ry90
            R2 = Ry90 @ Rz180

            Tew = np.eye(4)
            Tew[:3, :3] = R1

            ee_offset_eeframe = np.array([0.0, 0.0, -ee_offset])
            Tew[:3, 3] = R1 @ ee_offset_eeframe

            Tew2 = np.eye(4)
            Tew2[:3, :3] = R2

            Tew2[:3, 3] = R2 @ ee_offset_eeframe

            x_fits = sx / 2 <= half_finger_clearance
            y_fits = sy / 2 <= half_finger_clearance

            offsets = []

            # Grasp across y dimension: object x can hang out
            if x_fits:
                offsets += [np.pi / 2, 3 * np.pi / 2]

            # Grasp across x dimension: object y can hang out
            if y_fits:
                offsets += [0.0, np.pi]

            # Override calculated offsets for microwave handle
            if self.environment_name == "microwave":
                offsets = [0.0]

            Tews = make_tew_yaw_variants(Tew, offsets)
            Tews2 = make_tew_yaw_variants(Tew2, offsets)
            # pprint(f"Tews: {Tews}")
            # pprint(f"Tews2: {Tews2}")

            Tews = Tews + Tews2

            if y_fits and not x_fits:
                # grasp object from ±x; y is between fingers
                del_geom_y = half_finger_clearance - sy / 2
                del_geom_x = half_finger_length - sx / 2

            elif x_fits and not y_fits:
                # grasp object from ±y; x is between fingers
                del_geom_x = half_finger_clearance - sx / 2
                del_geom_y = half_finger_length - sy / 2

            elif x_fits and y_fits:
                # The object fits between the fingers.
                del_geom_x = half_finger_clearance - sx / 2.0

                # Allow sliding along the finger length while retaining
                # at least min_contact_overlap of contact.
                del_geom_y = max(
                    0.0,
                    sy / 2.0 - min_contact_overlap,
                )

            else:
                raise ValueError("Object does not fit between fingers.")

            del_geom_x /= 2
            del_geom_y /= 2

            if self.env_details["robot"] == "fetch":
                del_geom_x /= 2

        else:
            raise ValueError(f"Unsupported grasp type: {self.grasp_details['type']}")

        self.grasp_details["ee_offsets"] = Tews
        return del_geom_x, del_geom_y

    def construct_tcr(
        self, min_contact_overlap=0.01, tsr_bounds=None, planar_shape="compatible"
    ):
        """Derive analytic inner cells, retaining historical sizes by default.

        The old nominal collision sampling only calibrates preferred dimensions.
        It is not a certificate. Each actual goal is checked against a continuous
        cell enclosure. Pass TSRBounds (or face -> TSRBounds) to impose independent
        task tolerances; planar_shape="square" uses the paper's equal xy widths.
        """
        if planar_shape not in {"compatible", "square"}:
            raise ValueError("planar_shape must be compatible or square")
        self.grasp_details["tcr_metadata"] = {
            "version": 2,
            "planar_shape": planar_shape,
            "environment": self.environment_name,
            "robot": self.env_details["robot"],
            "bounds_source": (
                "compatibility_calibration" if tsr_bounds is None else "explicit"
            ),
            "cells": {},
        }
        self.grasp_details["ee_offsets_by_face"] = {}

        yaw_buffer = self.grasp_details["yaw_buffer"]
        alpha = self.grasp_details["alpha"]
        env_name = self.env_details["env_name"]
        tcr_batches = self.env_details.get("tcr_batches", None)

        if self.env_details["robot"] not in {
            "panda",
            "fetch",
            "g1",
            "ur10",
        }:
            raise ValueError(f"Unsupported robot: {self.env_details['robot']}")

        if self.object_details["type"] not in {
            "box",
            "cylinder",
            "microwave",
        }:
            raise ValueError(f"Unsupported object type: {self.object_details['type']}")

        if self.env_details["robot"] == "panda":
            half_finger_clearance = 0.038
            half_finger_length = 0.015
            ee_z_offset = 0.0
            ee_offset = 0.0

        elif self.env_details["robot"] == "fetch":
            half_finger_clearance = 0.048
            half_finger_length = 0.03
            ee_z_offset = 0.02
            ee_offset = 0.005

        elif self.env_details["robot"] == "g1":
            half_finger_clearance = 0.038
            half_finger_length = 0.01
            ee_z_offset = 0.0
            ee_offset = 0.016
        elif self.env_details["robot"] == "ur10":
            half_finger_clearance = 0.05
            half_finger_length = 0.05
            ee_z_offset = -0.04
            ee_offset = 0.0

        if tcr_batches is not None:
            num_tcr_batches = len(tcr_batches)
        else:
            num_tcr_batches = 1

        tcr_intervals_batches = {}

        for tcr_batch_idx in range(num_tcr_batches):

            if self.environment_name == "allstable":
                contact_face = tcr_batches[tcr_batch_idx]
            else:
                contact_face = None

            if self.object_details["type"] == "box":
                sx, sy, sz = map(float, self.object_details["size"])

                if tcr_batch_idx == 1:
                    sx, sy, sz = sy, sz, sx
                elif tcr_batch_idx == 2:
                    sx, sy, sz = sz, sx, sy

            elif self.object_details["type"] == "microwave":
                # Use door dimensions
                sx, sy, sz = map(float, self.object_details["handle_size"])

            elif self.object_details["type"] == "cylinder":
                radius, height = map(
                    float,
                    self.object_details["size"],
                )

                sx = 2.0 * radius
                sy = 2.0 * radius
                sz = height

            del_geom_x, del_geom_y = self.initial_tcr_construction(
                sx,
                sy,
                sz,
                ee_z_offset,
                ee_offset,
                half_finger_length,
                half_finger_clearance,
                min_contact_overlap,
            )

            initial_tcr_intervals = {
                "x": (alpha * np.array([-del_geom_x, 0.0, del_geom_x])).tolist(),
                "y": (alpha * np.array([-del_geom_y, 0.0, del_geom_y])).tolist(),
            }

            if self.object_details["type"] != "cylinder":
                initial_tcr_intervals["yaw"] = (
                    alpha
                    * np.array(
                        [
                            -yaw_buffer / 2.0,
                            0.0,
                            yaw_buffer / 2.0,
                        ]
                    )
                ).tolist()

            if self.object_details["type"] == "microwave":
                door_buffer = self.grasp_details["door_buffer"]
                initial_tcr_intervals["door"] = (
                    alpha * np.array([-door_buffer / 2.0, 0.0, door_buffer / 2.0])
                ).tolist()

            Tews = self.grasp_details["ee_offsets"]
            if env_name == "microwave" or tsr_bounds is not None:
                # A nominal IK search can fail even when other cells are usable.
                # Articulated cells now use the analytic hinge bound directly.
                tcr_intervals = dict(initial_tcr_intervals)
                tcr_intervals["z"] = [
                    (
                        self.object_details["size"][2] / 2
                        if env_name == "microwave"
                        else sz / 2
                    )
                ]
                calibration = (
                    "analytic_hinge" if env_name == "microwave" else "explicit_bounds"
                )
                if self.object_details["type"] == "cylinder":
                    tcr_intervals["yaw"] = [0.0]
                if (
                    env_name == "microwave"
                    and "hinge_pos_body" not in self.object_details
                ):
                    self.object_xml(
                        self.object_details["size"], [0, 0, 0, 0], temp=True
                    )
            else:
                try:
                    tcr_intervals = self.find_tcr_intervals(
                        initial_tcr_intervals, Tews, contact_face
                    )
                    calibration = "nominal_samples"
                except ValueError as error:
                    if str(error) != "Unable to find IK solution":
                        raise
                    # Nominal reachability is not a precondition for an analytic
                    # grid. No goals are accepted without the per-cell checks.
                    tcr_intervals = dict(initial_tcr_intervals, z=[sz / 2])
                    if self.object_details["type"] == "cylinder":
                        tcr_intervals["yaw"] = [0.0]
                    calibration = "initial_preference_no_nominal_solution"

            # Keep the nominal sampled calibration only as a compatibility
            # preference; now derive dimensions from an explicit TSR contract.
            half = np.array(
                [
                    (max(tcr_intervals.get(d, [0])) - min(tcr_intervals.get(d, [0])))
                    / 2
                    for d in ("x", "y", "z", "yaw", "door")
                ]
            )
            radii = (
                microwave_radii(self.object_details)
                if env_name == "microwave"
                else None
            )
            bounds = tsr_bounds
            if isinstance(bounds, dict):
                bounds = bounds[contact_face or "default"]
            if bounds is None:
                bounds = calibrate_bounds(half, radii)
            aspect = half[:2] if planar_shape == "compatible" else (1.0, 1.0)
            derived = inner_halfwidths(bounds, half[3], half[4], aspect, radii)
            for dimension, h in zip(("x", "y", "z", "yaw", "door"), derived):
                if dimension in tcr_intervals and (dimension != "z" or h > 0):
                    tcr_intervals[dimension] = (
                        [-float(h), 0.0, float(h)] if h else [0.0]
                    )
            face_key = contact_face or "default"
            self.grasp_details["tcr_metadata"]["cells"][face_key] = {
                "tsr_bounds": bounds.to_dict(),
                "calibration": calibration,
                "preferred_halfwidths": half.tolist(),
                "halfwidths": derived.tolist(),
                "ee_offsets": [offset.tolist() for offset in Tews],
            }
            self.grasp_details["ee_offsets_by_face"][face_key] = [
                offset.copy() for offset in Tews
            ]

            if tcr_batches is None:
                self.grasp_details["tcr_intervals"] = tcr_intervals
                return tcr_intervals

            tcr_intervals_batches[tcr_batches[tcr_batch_idx]] = tcr_intervals

        self.grasp_details["tcr_intervals"] = tcr_intervals_batches
        return tcr_intervals_batches

    def configure_tcr_grid(self, planar_shape="compatible", cell_scale=1.0):
        """Choose square cells or uniformly refine; keep TSR tolerances fixed.

        Refining the whole grid preserves coverage. Shrinking individual cells
        while keeping their original keys would leave unrepresented gaps.
        """
        if planar_shape not in {"compatible", "square"} or not 0 < cell_scale <= 1:
            raise ValueError("Expected compatible/square shape and 0 < cell_scale <= 1")
        metadata = self.grasp_details["tcr_metadata"]
        radii = (
            microwave_radii(self.object_details)
            if self.environment_name == "microwave"
            else None
        )
        for face, cell in metadata["cells"].items():
            old = np.asarray(cell["halfwidths"])
            if planar_shape == "square":
                half = inner_halfwidths(
                    TSRBounds(**cell["tsr_bounds"]), old[3], old[4], radii=radii
                )
            else:
                half = old.copy()
            half *= cell_scale
            intervals = self.grasp_details["tcr_intervals"]
            if face != "default":
                intervals = intervals[face]
            for dim, h in zip(("x", "y", "z", "yaw", "door"), half):
                if dim in intervals and (dim != "z" or h > 0):
                    intervals[dim] = [-float(h), 0.0, float(h)] if h else [0.0]
            cell["halfwidths"] = half.tolist()
        metadata["planar_shape"] = planar_shape
        metadata["cell_scale"] = cell_scale

    def sample_tcr_intervals(self, intervals):
        dimensions = list(intervals)

        for values in itertools.product(
            *(intervals[dimension] for dimension in dimensions)
        ):
            yield dict(zip(dimensions, values))

    def find_tcr_intervals(self, initial_tcr_intervals, ee_offsets, contact_face=None):

        object_size = self.object_details["size"]

        if contact_face is None:
            new_object_size = object_size
        else:
            if contact_face == "xy":
                new_object_size = object_size
                face_idx = 0
            elif contact_face == "yz":
                new_object_size = [
                    object_size[1],
                    object_size[2],
                    object_size[0],
                ]
                face_idx = 1
            elif contact_face == "zx":
                new_object_size = [
                    object_size[2],
                    object_size[0],
                    object_size[1],
                ]
                face_idx = 2
            else:
                raise ValueError(f"Unknown contact face: {contact_face}")

        # Generate dummy environment for finding TCR
        robot_dir = f"assets/{self.env_details['robot']}"

        if self.env_details["robot"] == "panda":
            robot_dir = "assets/franka_emika_panda"
            robot_pos = [0, 0, 0]
        elif self.env_details["robot"] == "fetch":
            robot_pos = [0, 0, 0.005]
        elif self.env_details["robot"] == "g1":
            robot_pos = [0, 0, 0]
        elif self.env_details["robot"] == "ur10":
            robot_pos = [0, 0, 0]

        robot_quat = [1, 0, 0, 0]
        base_xml = "scene.xml"

        # Generalize to more objects later
        if self.environment_name == "allstable":
            object_xml = self.object_xml(
                new_object_size, [1, 1, 0, 0], temp=True, name=f"cube_object_0"
            )
            object_xml_dummy1 = self.object_xml(
                new_object_size, [1, 1, 0, 0], temp=True, name=f"cube_object_1"
            )
            object_xml_dummy2 = self.object_xml(
                new_object_size, [1, 1, 0, 0], temp=True, name=f"cube_object_2"
            )
            object_xmls = [object_xml, object_xml_dummy1, object_xml_dummy2]

        else:
            object_xml = self.object_xml(new_object_size, [1, 1, 0, 0], temp=True)
            object_xmls = [object_xml]

        free_xml_path = f"{robot_dir}/temp_scene_{os.getpid()}.xml"

        tempModel, tempData = self.build_model(
            free_xml_path,
            object_xmls,
        )

        visualizeTemp = False
        if self.env_details["robot"] == "panda":
            tempRobot = Panda(tempModel, tempData, visualizeTemp)
        elif self.env_details["robot"] == "fetch":
            tempRobot = FetchArm(tempModel, tempData, visualizeTemp)
        elif self.env_details["robot"] == "g1":
            tempRobot = G1(tempModel, tempData, visualizeTemp)
        elif self.env_details["robot"] == "ur10":
            tempRobot = UR10(tempModel, tempData, visualizeTemp)
        else:
            raise NotImplementedError(
                f"Robot not supported: {self.env_details['robot']}"
            )

        tempRobot.teleport_base(pos=robot_pos, quat=robot_quat)

        nominal_x = 0
        nominal_y = 0

        if self.object_details["type"] == "cylinder":
            nominal_z = new_object_size[1] / 2.0
        else:
            nominal_z = new_object_size[2] / 2.0

        nominal_yaw = 0

        if self.environment_name == "microwave":
            nominal_x = 0.75
            coll_geoms = [
                "mw_bottom",
                "mw_top",
                "mw_left",
                "mw_right",
                "mw_back",
                "mw_door",
                "mw_handle",
            ]

            if self.env_details["robot"] == "fetch":
                nominal_z += 0.5
                nominal_x = 1.0
        elif self.environment_name == "allstable":
            nominal_x = 0.5
            coll_geoms = [
                "cube_object_0_geom",
                "cube_object_1_geom",
                "cube_object_2_geom",
            ]
        else:
            nominal_x = 0.5
            coll_geoms = ["cube_object_geom"]

            if self.env_details["robot"] == "fetch":
                nominal_z += 0.5

        if self.env_details["robot"] == "g1":
            nominal_z += 0.75
            nominal_x = 0.4

        nominal_object_pose = [nominal_x, nominal_y, nominal_z, nominal_yaw]

        if contact_face is None:
            self.move_object(nominal_object_pose, tempModel, tempData)
        else:
            self.move_object(
                [contact_face, nominal_x, nominal_y, nominal_z, nominal_yaw],
                tempModel,
                tempData,
            )
        # self.move_xml_joint("microwave_door_hinge", np.pi/4, tempModel, tempData)

        if tempRobot.viewer is not None:
            tempRobot.viewer.sync()
            input("Proceed?")

        ik_solver = get_ik_solver(tempRobot, env_collision_geoms=coll_geoms)

        if self.environment_name == "microwave":
            pass
            door_pos, door_rpy = self.get_geom_pose("mw_handle", tempModel, tempData)
            obj_pose = Pose(tuple(door_pos), tuple(door_rpy)).matrix()
        else:
            obj_pose = Pose(
                tuple(nominal_object_pose[:3]), (0, 0, nominal_object_pose[3])
            ).matrix()

        targets = [matrix_to_flat(obj_pose @ offset) for offset in ee_offsets]
        n_attempts = 10

        nominal_grasp = None

        for target in targets:
            for attempt_no in range(n_attempts):
                seed = sample_qpos(tempRobot.model, tempRobot.joint_ids)

                reached, solution = ik_solver.solve(
                    target,
                    seed,
                    use_col=True,
                    pos_tol=1e-4,
                    rot_tol=1e-3,
                )

                if not reached or solution is None:
                    print(
                        f"PID {os.getpid()}: "
                        f"IK failed on attempt {attempt_no + 1}/{n_attempts}",
                        flush=True,
                    )
                    continue

                tempRobot.set_joint_qpos(solution)

                if tempRobot.in_contact():
                    continue

                nominal_grasp = solution.tolist()
                break

            if nominal_grasp is not None:
                break

        if nominal_grasp is not None:
            tempRobot.set_joint_qpos(nominal_grasp)
        else:
            raise ValueError("Unable to find IK solution")

        # Perturb object until collision

        current_intervals = {
            dimension: list(values)
            for dimension, values in initial_tcr_intervals.items()
        }

        cleared = False

        for refinement in range(100):
            collision_found = False

            for sample in self.sample_tcr_intervals(current_intervals):
                x_value = sample.get("x", 0.0)
                y_value = sample.get("y", 0.0)
                yaw_value = sample.get("yaw", 0.0)
                door_value = sample.get("door")

                object_perturbation = np.array(
                    [
                        x_value,
                        y_value,
                        0.0,
                        yaw_value,
                    ],
                    dtype=float,
                )

                test_pose = (
                    np.asarray(
                        nominal_object_pose,
                        dtype=float,
                    )
                    + object_perturbation
                ).tolist()

                if contact_face is None:
                    self.move_object(
                        test_pose,
                        tempModel,
                        tempData,
                    )
                else:
                    self.move_object(
                        [
                            contact_face,
                            test_pose[0],
                            test_pose[1],
                            test_pose[2],
                            test_pose[3],
                        ],
                        tempModel,
                        tempData,
                    )

                if door_value is not None:
                    self.move_xml_joint(
                        "microwave_door_hinge",
                        door_value,
                        tempModel,
                        tempData,
                    )

                if tempRobot.viewer is not None:
                    tempRobot.viewer.sync()
                    input("Proceed?")

                if tempRobot.in_contact():
                    collision_found = True
                    break

            if collision_found:
                current_intervals = {
                    dimension: (0.95 * np.asarray(values)).tolist()
                    for dimension, values in current_intervals.items()
                }
            else:
                cleared = True
                break
        if not cleared:
            raise RuntimeError("Nominal TCR calibration failed after 100 reductions")

        tcr_intervals = {
            dimension: values for dimension, values in current_intervals.items()
        }

        tcr_intervals["z"] = [nominal_object_pose[2]]

        if self.object_details["type"] == "cylinder":
            tcr_intervals["yaw"] = [0.0]

        if tempRobot.viewer is not None:
            input()
            tempRobot.close()
        return tcr_intervals

    def validate_cell_goal(self, robot, q):
        """Recheck endpoints after IK refinement or trajectory adaptation."""
        if not hasattr(self, "current_tcr_key"):
            return False
        key = self.current_tcr_key
        face = key[0] if self.environment_name == "allstable" else "default"
        robot.set_joint_qpos(q)
        if not self.goal_clears_cell(robot):
            return False
        actual_ee = flat_to_matrix(robot.get_ee_pose())
        return any(
            self.goal_satisfies_tsr(key, actual_ee, offset)
            for offset in self.grasp_details["ee_offsets_by_face"][face]
        )

    def goal_clears_cell(self, robot):
        """Strict collision check of the fixed goal against the enclosure."""
        return not any(
            contact.dist <= 0
            and (
                contact.geom1 in robot.robot_geoms or contact.geom2 in robot.robot_geoms
            )
            for contact in self.data.contact[: self.data.ncon]
        )

    def load_tcr_metadata(self, path):
        """Restore the task set's contract, independently of worker RNG seeds."""
        import json
        from pathlib import Path

        path = Path(path)
        if not path.exists():
            return False  # Historical datasets keep their original file format.
        metadata = json.loads(path.read_text())
        if metadata.get("version") != 2:
            raise ValueError("Unsupported task-region metadata version")
        if (
            metadata.get("environment") != self.environment_name
            or metadata.get("robot") != self.env_details["robot"]
        ):
            raise ValueError(
                "Task-region metadata belongs to another environment/robot"
            )
        self.grasp_details["tcr_metadata"] = metadata
        self.grasp_details["ee_offsets_by_face"] = {
            face: [np.asarray(offset) for offset in cell["ee_offsets"]]
            for face, cell in metadata["cells"].items()
        }
        return True

    def goal_satisfies_tsr(self, key, actual_ee, offset):
        """Check actual IK output against the analytic contract for this cell."""
        if self.environment_name == "allstable":
            face, numeric_key = key[0], key[1:]
        else:
            face, numeric_key = "default", key
        metadata = self.grasp_details["tcr_metadata"]["cells"][face]
        bounds = TSRBounds(**metadata["tsr_bounds"])
        details = self.object_details if self.environment_name == "microwave" else None
        return goal_fits_cell(bounds, numeric_key, actual_ee, offset, details)

    def find_problem_intervals(self, scene_yaml, base_name="base", wall_clearance=0.18):
        """Find x,y intervals for valid object positions in problem"""
        with open(scene_yaml, "r") as f:
            scene_yaml_data = yaml.safe_load(f)
        objs = scene_yaml_data["world"]["collision_objects"]

        base_dim = None
        base_pos = None
        for obj in objs:
            obj_id = obj.get("id", "")
            if obj_id != base_name:
                continue
            base_dim = obj["primitives"][0]["dimensions"]
            base_pos = obj["primitive_poses"][0]["position"]

        if base_dim is None or base_pos is None:
            raise RuntimeError(f"Did not find element in scene: {base_name}")

        # Update nominal object position with updated z
        self.object_details["position"] = [
            self.env_details["robot_pos"][0],
            self.env_details["robot_pos"][1],
            base_pos[2] + (base_dim[2] / 2) + self.object_details["size"][2] / 2,
        ]

        self.env_details["z_correction"] = [base_pos[2] + (base_dim[2] / 2)]

        hx_int = base_dim[0] / 2 - wall_clearance / 2
        hy_int = base_dim[1] / 2 - wall_clearance / 2

        env_name = self.env_details["env_name"]

        if env_name in {"largeobj"}:
            base_xmin = base_pos[0] - hx_int  # + self.object_details['size'][0]/2
            base_xmax = base_pos[0] + hx_int  # - self.object_details['size'][0]/2
            base_ymin = base_pos[1] - hy_int  # + self.object_details['size'][1]/2
            base_ymax = base_pos[1] + hy_int  # - self.object_details['size'][1]/2
        else:
            base_xmin = base_pos[0] - hx_int + self.object_details["size"][0] / 2
            base_xmax = base_pos[0] + hx_int - self.object_details["size"][0] / 2
            base_ymin = base_pos[1] - hy_int + self.object_details["size"][1] / 2
            base_ymax = base_pos[1] + hy_int - self.object_details["size"][1] / 2

        base_intervals = [[base_xmin, base_xmax], [base_ymin, base_ymax]]
        return base_intervals

    def generate_task_set(self):
        """Generate task set/TSRs"""

        TCR_set = create_tcr_set(self)
        self.task_set = TCR_set
        return self.task_set
