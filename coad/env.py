import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import numpy as np
import yaml
import mujoco
import re
from pathlib import Path

import itertools
import pickle
import re
from pathlib import Path
from pprint import pprint

import mujoco.viewer
import trimesh
from scipy.spatial.transform import Rotation
from mink.exceptions import NoSolutionFound


from geometry.pose import Pose, matrix_to_flat, wrap_to_pi

from coad.mink_ik import get_ik_solver
from coad.mujoco_utils import (
    joint_names_to_joint_ids,
    joints_to_limits,
    joints_to_qpos_dof_ids,
    sample_qpos,
)
from coad.robot import MujocoRobot, Panda, UR10, FetchArm, G1
from coad.task_generation import (
    create_TCR_set,
    fetch_TSR_parameters,
    find_iTSR_set,
    find_yaw_iTSR_set,
    make_Tew_yaw_variants,
    panda_TSR_parameters,
    translational_half_extents,
    ur10_TSR_parameters,
    valid_grasp_yaw_offsets,
)

def wrap_to_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

class MujocoEnv:
    def __init__(self, robot, custom_base=None):
        """Initialize object dimensions and common parameters"""
        self.swept_volume_primitives = {}
        if robot == "panda":
            self.robot_dir = "assets/franka_emika_panda"
        else:
            self.robot_dir = f"assets/{robot}"
        if custom_base is None:
            if robot == "g1":
                self.base_xml = "scene_23dof.xml"
            else:
                # self.base_xml = "spherized_scene.xml"
                self.base_xml = "scene.xml"
        else:
            self.base_xml = custom_base
    
    # Class function for populating object details
    def populate_object_details(self, object_type, object_size, object_variation):
        
        if object_type == "box":
            if len(object_size)==3:
                self.object_details = {
                    'type': object_type,
                    'size': object_size,
                }
            else:
                raise ValueError("Object size must be length 3 for boxes")
            
            for variation_axis in object_variation:
                if variation_axis not in ["x", "y", "z", "yaw"]:
                    raise ValueError(f"Unsupported variation given for {object_type} type: {variation_axis}")
            self.object_details['variation'] = object_variation

        elif object_type == "cylinder":
            if len(object_size)==2:
                self.object_details = {
                    'type': object_type,
                    'size': object_size,
                }
            else:
                raise ValueError("Object size must be length 2 for cylinders")

            for variation_axis in object_variation:
                if variation_axis == "yaw":
                    object_variation['yaw'] = [[0, 0]]
                if variation_axis not in ["x", "y", "z", "yaw"]:
                    raise ValueError(f"Unsupported variation given for {object_type} type: {variation_axis}")
            self.object_details['variation'] = object_variation

        else:
            raise ValueError(f"Unsupported object type: {object_type}")
        
    def populate_env_details(self, scene_yaml, robot_name, env_name, robot_pos, robot_quat, outer_rad, inner_rad):

        self.env_details = {
            'robot_pos': robot_pos,
            'robot_quat': robot_quat,
            'robot': robot_name,
            'env_name': env_name,
            'outer_rad': outer_rad,
            'inner_rad': inner_rad,
            'collision_geoms': []
        }

        if env_name in ['box', 'cage']:
            intervals = self.find_problem_intervals(scene_yaml, base_name="base", wall_clearance=0.18)
        elif env_name in ['table', 'largeobj', 'microwave', 'allstable']:
            intervals = self.find_problem_intervals(scene_yaml, base_name="table_top", wall_clearance=0.18)
        elif env_name == "shelf":
            #bases = ['shelf_bottom', 'shelf_middle_bottom', 'shelf_middle', 'shelf_middle_top', 'shelf_top']
            bases = ['shelf_middle']
            intervals = self.find_problem_intervals(scene_yaml, bases, 0.12, 0.14)
        else:
            intervals = None
            self.env_details['z_correction'] = [0]

        self.env_details['intervals'] = intervals

    def populate_grasp_details(
        self, 
        alpha=0.95,
        yaw_buffer=6*(np.pi/180),
        grasp_type="top"
    ):
        
        self.grasp_details = {
            'type': grasp_type,
            'alpha': alpha,
            'yaw_buffer': yaw_buffer
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
        min_contact_overlap
        ):

        robot_name = self.env_details['robot']

        if self.grasp_details["type"] == "top":
            Tew = np.eye(4)
            Tew[1, 1] = -1
            Tew[2, 2] = -1

            if robot_name == "g1":
                Tew = np.eye(4)

                R = np.array([
                    [1.0, 0.0,  0.0],
                    [0.0, 0.0, -1.0],
                    [0.0, 1.0,  0.0],
                ])

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
                    self.object_details,
                    2.0 * half_finger_clearance,
                )
                if isinstance(self, TableEnv):    
                    offsets = [np.pi/2, -np.pi/2]
                else:
                    offsets = [0.0, np.pi]

                Tews = make_Tew_yaw_variants(Tew, offsets)

                clearance_size_x = sx
                clearance_size_y = sy

                if y_fits and not x_fits:
                    # Long object along x, narrow along y.
                    long_axis_slide = max(0.0, clearance_size_x / 2.0 - min_contact_overlap)
                    del_geom_x = long_axis_slide
                    del_geom_x /= 2.0
                    del_geom_y = half_finger_clearance - clearance_size_y / 2.0
                    del_geom_y /= 1.0

                elif x_fits and not y_fits:
                    # Long object along y, narrow along x.
                    long_axis_slide = max(0.0, clearance_size_y / 2.0 - min_contact_overlap)
                    del_geom_y = long_axis_slide
                    del_geom_x = half_finger_clearance - clearance_size_x / 2.0
                    del_geom_x /= 1.0

                elif x_fits and y_fits:
                    # Object fits both ways. Pick a consistent convention.
                    long_axis_slide = max(0.0, clearance_size_y / 2.0 - min_contact_overlap)
                    del_geom_y = long_axis_slide
                    del_geom_x = half_finger_clearance - clearance_size_x / 2.0

                else:
                    raise ValueError("Object does not fit the gripper in either x or y.")

                # print(f"xfits: {x_fits}")
                # print(f"yfits: {y_fits}")

                del_geom_x, del_geom_y = translational_half_extents(
                    del_geom_x,
                    del_geom_y,
                )

        elif self.grasp_details["type"] == "front":
            # Canonical front grasp: approach horizontally.

            # print(sx, sy, sz)

            Ry90 = np.array([
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
            ])

            Rz180 = np.array([
                [-1.0,  0.0, 0.0],
                [ 0.0, -1.0, 0.0],
                [ 0.0,  0.0, 1.0],
            ])

            R1 = Ry90
            R2 = Ry90 @ Rz180

            Tew = np.eye(4)
            Tew[:3, :3] = R1

            # Keep your old convention: offset along -EEF z, mapped into world/object frame.
            # ee_offset = -0.03 + self.object_details['size'][1]
            # ee_offset = 0.016
            ee_offset_eeframe = np.array([0.0, 0.0, -ee_offset])
            Tew[:3, 3] = R1 @ ee_offset_eeframe

            Tew2 = np.eye(4)
            Tew2[:3, :3] = R2

            Tew2[:3, 3] = R2 @ ee_offset_eeframe            


            # Tews = make_Tew_yaw_variants(Tew, yaw_angles)

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
            if isinstance(self, MicrowaveEnv):
                offsets = [0.0]

            Tews = make_Tew_yaw_variants(Tew, offsets)
            Tews2 = make_Tew_yaw_variants(Tew2, offsets)
            # pprint(f"Tews: {Tews}")
            # pprint(f"Tews2: {Tews2}")

            Tews = Tews + Tews2
            

            # print(f"sx: {sx}")
            # print(f"sy: {sy}")

            if y_fits and not x_fits:
                # grasp object from ±x; y is between fingers
                del_geom_y = half_finger_clearance - sy / 2
                del_geom_x = half_finger_length - sx / 2

            elif x_fits and not y_fits:
                # grasp object from ±y; x is between fingers
                del_geom_x = half_finger_clearance - sx / 2
                del_geom_y = half_finger_length - sy / 2

            elif x_fits and y_fits:
                # choose convention, or create both candidate families
                del_geom_x = half_finger_clearance - sx / 2
                del_geom_y = half_finger_length - sy / 2

            else:
                raise ValueError("Object does not fit between fingers.")

            del_geom_x /= 2
            del_geom_y /= 2

            if self.env_details['robot'] == "fetch":
                del_geom_x /= 2

        else:
            raise ValueError(f"Unsupported grasp type: {self.grasp_details['type']}")

        # self.grasp_details["ee_offsets"] = Tews
        self.grasp_details["ee_offsets"] = Tews
        return del_geom_x, del_geom_y

    def construct_tcr(self, min_contact_overlap=0.01):

        yaw_buffer = self.grasp_details["yaw_buffer"]
        alpha = self.grasp_details["alpha"]
        env_name = self.env_details['env_name']
        tcr_batches = self.env_details.get('tcr_batches', None)

        if self.env_details["robot"] not in {
            "panda",
            "fetch",
            "g1",
            "ur10",
        }:
            raise ValueError(
                f"Unsupported robot: {self.env_details['robot']}"
            )

        if self.object_details["type"] not in {
            "box",
            "cylinder",
            "microwave",
        }:
            raise ValueError(
                f"Unsupported object type: {self.object_details['type']}"
            )
        
        if self.env_details['robot'] == "panda":    
            half_finger_clearance = 0.038
            half_finger_length = 0.015
            ee_z_offset = 0.0
            ee_offset = 0.016

        elif self.env_details['robot'] == 'fetch':
            half_finger_clearance = 0.048
            half_finger_length = 0.03
            ee_z_offset = 0.02
            ee_offset = 0.04
        
        elif self.env_details['robot'] == 'g1':
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

            if isinstance(self, AllStableEnv):
                contact_face = tcr_batches[tcr_batch_idx]
            else:
                contact_face = None

            if self.object_details["type"] == 'box':
                sx, sy, sz = map(float, self.object_details["size"])

                if tcr_batch_idx == 1:
                    sx, sy, sz = sz, sx, sy
                elif tcr_batch_idx == 2:
                    sx, sy, sz = sy, sz, sx

            elif self.object_details["type"] == 'microwave':
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
                min_contact_overlap
            )

            # initial_tcr_intervals = {
            #     "x": (alpha * np.array([-del_geom_x, 0.0, del_geom_x])).tolist(),
            #     "y": (alpha * np.array([-del_geom_y, 0.0, del_geom_y])).tolist(),
            #     "yaw": (alpha * np.array([-yaw_buffer / 2.0, 0.0, yaw_buffer / 2.0])).tolist(),
            # }

            initial_tcr_intervals = {
                "x": (
                    alpha * np.array(
                        [-del_geom_x, 0.0, del_geom_x]
                    )
                ).tolist(),
                "y": (
                    alpha * np.array(
                        [-del_geom_y, 0.0, del_geom_y]
                    )
                ).tolist(),
            }

            if self.object_details["type"] != "cylinder":
                initial_tcr_intervals["yaw"] = (
                    alpha * np.array(
                        [
                            -yaw_buffer / 2.0,
                            0.0,
                            yaw_buffer / 2.0,
                        ]
                    )
                ).tolist()

            if self.object_details['type'] == "microwave":
                door_buffer = self.grasp_details['door_buffer']
                initial_tcr_intervals["door"] = (alpha * np.array([-door_buffer / 2.0, 0.0, door_buffer / 2.0])).tolist()

            Tews = self.grasp_details["ee_offsets"]
            tcr_intervals = self.find_tcr_intervals(initial_tcr_intervals, Tews, contact_face)
        
            if tcr_batches is None:
                self.grasp_details['tcr_intervals'] = tcr_intervals
                return tcr_intervals
            
            tcr_intervals_batches[tcr_batches[tcr_batch_idx]] = tcr_intervals

        self.grasp_details['tcr_intervals'] = tcr_intervals_batches 
        return tcr_intervals_batches
        # return tcr_intervals
    
    # def sample_tcr_intervals(self, initial_tcr_intervals):
    #     dims = ["x", "y", "yaw"]

    #     if "door" in initial_tcr_intervals:
    #         dims.append("door")
        
    #     for values in itertools.product(*(initial_tcr_intervals[d] for d in dims)):
    #         yield dict(zip(dims, values))

    def sample_tcr_intervals(self, intervals):
        dimensions = list(intervals)

        for values in itertools.product(
            *(intervals[dimension] for dimension in dimensions)
        ):
            yield dict(zip(dimensions, values))

    def find_tcr_intervals(self, initial_tcr_intervals, ee_offsets, contact_face=None):

        object_size = self.object_details['size']

        if contact_face is None:
            new_object_size = object_size
        else:
            if contact_face == "xy":
                new_object_size = object_size
                face_idx = 0
            elif contact_face == "yz":
                new_object_size = [object_size[1], object_size[2], object_size[0]]
                face_idx = 1
            elif contact_face == "zx":
                new_object_size = [object_size[2], object_size[0], object_size[1]]
                face_idx = 2
            else:
                raise ValueError(f"Unknown contact face: {contact_face}")

        # Generate dummy environment for finding TCR
        robot_dir = f"assets/{self.env_details['robot']}"

        if self.env_details['robot'] == "panda":
            robot_dir = "assets/franka_emika_panda"
            robot_pos = [0, 0, 0]
        elif self.env_details['robot'] == "fetch":
            robot_pos = [0, 0, 0.005]
        elif self.env_details['robot'] == "g1":
            robot_pos = [0, 0, 0]
        elif self.env_details["robot"] == "ur10":
            robot_pos = [0, 0, 0]


        robot_quat = [1, 0, 0, 0]
        base_xml = "scene.xml"
        
        # Generalize to more objects later
        if isinstance(self, AllStableEnv):
            object_xml = self.object_xml(new_object_size, [1, 1, 0, 0], temp=True, name=f"cube_object_0")
            object_xml_dummy1 = self.object_xml(new_object_size, [1, 1, 0, 0], temp=True, name=f"cube_object_1")
            object_xml_dummy2 = self.object_xml(new_object_size, [1, 1, 0, 0], temp=True, name=f"cube_object_2")
            object_xmls = [object_xml, object_xml_dummy1, object_xml_dummy2]
        
        else:
            object_xml = self.object_xml(new_object_size, [1, 1, 0, 0], temp=True)
            object_xmls = [object_xml]

        # free_xml_path = f"{robot_dir}/temp_scene.xml"
        # tempModel, tempData = self.build_model(free_xml_path, [object_xml])
        free_xml_path = (
            f"{robot_dir}/temp_scene_{os.getpid()}.xml"
        )

        # tempModel, tempData = self.build_model(
        #     free_xml_path,
        #     [object_xml],
        # )
        tempModel, tempData = self.build_model(
            free_xml_path,
            object_xmls,
        )

        visualizeTemp = False
        if self.env_details['robot'] == "panda":
            tempRobot = Panda(tempModel, tempData, visualizeTemp)
        elif self.env_details['robot'] == "fetch":
            tempRobot = FetchArm(tempModel, tempData, visualizeTemp)
        elif self.env_details['robot'] == "g1":
            tempRobot = G1(tempModel, tempData, visualizeTemp)
        elif self.env_details['robot'] == "ur10":
            tempRobot = UR10(tempModel, tempData, visualizeTemp)
        else:
            raise NotImplemented(f"Robot not supported: {self.env_details['robot']}")

        tempRobot.teleport_base(pos=robot_pos, quat=robot_quat)
        
        nominal_x = 0
        nominal_y = 0

        if self.object_details["type"] == "cylinder":
            nominal_z = new_object_size[1] / 2.0
        else:
            nominal_z = new_object_size[2] / 2.0

        nominal_yaw = 0

        if isinstance(self, MicrowaveEnv):
            nominal_x = 0.75
            coll_geoms = [
                "mw_bottom",
                "mw_top",
                "mw_left",
                "mw_right",
                "mw_back",
                "mw_door",
                "mw_handle"
            ]

            if self.env_details['robot'] == "fetch":
                nominal_z += 0.5
                nominal_x = 1.0
        elif isinstance(self, AllStableEnv):
            nominal_x = 0.5
            coll_geoms = ['cube_object_0_geom', 'cube_object_1_geom', 'cube_object_2_geom']
        else:
            nominal_x = 0.5
            coll_geoms = ['cube_object_geom']

            if self.env_details['robot'] == "fetch":
                nominal_z += 0.5

        if self.env_details['robot'] == "g1":
            nominal_z += 0.75
            nominal_x = 0.4

        nominal_object_pose = [nominal_x, nominal_y, nominal_z, nominal_yaw]
        # nominal_object_pose = [nominal_x, 0, self.object_details['size'][2]/2, 0]
        
        if contact_face is None:
            self.move_object(nominal_object_pose, tempModel, tempData)
        else:
            self.move_object(
                [contact_face, nominal_x, nominal_y, nominal_z, nominal_yaw],
                tempModel, tempData)
        # self.move_xml_joint("microwave_door_hinge", np.pi/4, tempModel, tempData)

        if tempRobot.viewer is not None:
            tempRobot.viewer.sync()
            input("Proceed?")
        
        ik_solver = get_ik_solver(tempRobot, env_collision_geoms=coll_geoms)

        if isinstance(self, MicrowaveEnv):
            pass
            door_pos, door_rpy = self.get_geom_pose("mw_handle", tempModel, tempData)
            obj_pose = Pose(tuple(door_pos),tuple(door_rpy)).matrix()
        else:
            obj_pose = Pose(tuple(nominal_object_pose[:3]), (0, 0, nominal_object_pose[3])).matrix()

        targets = [matrix_to_flat(obj_pose @ offset) for offset in ee_offsets]
        n_attempts = 10

        # for i, offset in enumerate(ee_offsets):
        #     T_world_ee = obj_pose @ offset

        #     dist = np.linalg.norm(
        #         obj_pose[:3,3] - T_world_ee[:3,3]
        #     )

        #     obj_pos = obj_pose[:3, 3]
        #     ee_pos = T_world_ee[:3, 3]

        #     vec_ee_to_obj = obj_pos - ee_pos
        #     vec_ee_to_obj /= np.linalg.norm(vec_ee_to_obj)

        #     ee_z = T_world_ee[:3, :3][:, 2]

        #     # print(i, "dot(+z, ee_to_obj) =", np.dot(ee_z, vec_ee_to_obj))
        #     # print(i, dist)

        nominal_grasp = None
        # for target in targets:
        #     for attempt_no in range(n_attempts):
        #         seed = sample_qpos(tempRobot.model, tempRobot.joint_ids)
        #         reached, solution = ik_solver.solve(
        #             target, 
        #             seed, 
        #             use_col=True,
        #             pos_tol= 1e-4,
        #             rot_tol= 1e-3,
        #         )
        #         tempRobot.set_joint_qpos(solution)

        #         if tempRobot.viewer is not None:
        #             tempRobot.viewer.sync()
        #             print(f"Reached: {reached}, Solution: {solution}")
        #             input("Proceed?")
                
        #         if reached:
        #             # input()
        #             if not tempRobot.in_contact():
        #                 # print(f"Nominal grasp pose found: {solution}")
        #                 nominal_grasp = solution.tolist()
                        
        #                 if tempRobot.viewer is not None:
        #                     tempRobot.viewer.sync()
        #                     input()
        #                 break
                
        #     if nominal_grasp:
        #         break

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

        # x_vals = initial_tcr_intervals['x']
        # y_vals = initial_tcr_intervals['y']
        # yaw_vals = initial_tcr_intervals['yaw']
        # door_vals = initial_tcr_intervals.get("door", None)

        current_intervals = {
            dimension: list(values)
            for dimension, values
            in initial_tcr_intervals.items()
        }

        # print(f"Inital TCR bounds: {initial_tcr_intervals}")

        cleared = False

        while not cleared:
            collision_found = False

            for sample in self.sample_tcr_intervals(
                current_intervals
            ):
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
                    dimension: (
                        0.95 * np.asarray(values)
                    ).tolist()
                    for dimension, values
                    in current_intervals.items()
                }
            else:
                cleared = True
        
        # print("Found TCR Bounds:")
        # print(f"x: {x_vals}")
        # print(f"y: {y_vals}")
        # print(f"yaw: {yaw_vals}")
        # if door_vals is not None:
        #     print(f"door_vals: {door_vals}")

        # tcr_intervals = {
        #     'x': x_vals,
        #     'y': y_vals,
        #     'z': [nominal_object_pose[2]],
        #     'yaw': yaw_vals
        # }

        # if isinstance(self, MicrowaveEnv):
        #     tcr_intervals['door'] = door_vals

        tcr_intervals = {
            dimension: values
            for dimension, values
            in current_intervals.items()
        }

        tcr_intervals["z"] = [
            nominal_object_pose[2]
        ]

        if self.object_details["type"] == "cylinder":
            tcr_intervals["yaw"] = [0.0]

        if tempRobot.viewer is not None:
            input()
            tempRobot.close()
        # print(f"PID {os.getpid()}: {tcr_intervals}")
        return tcr_intervals

    def create_swept_volume(
        self,
        tcr_intervals,
        obj_size=None,
        sv_count=0,
        fixed=False,
    ):
        """
        Create a swept-volume mesh and save the cuboids used to construct it.

        The dictionary has the form:

            self.swept_volume_primitives[geom_name] = [
                {
                    "type": "cuboid",
                    "position": np.ndarray(shape=(3,)),
                    "orientation": np.ndarray(shape=(3, 3)),
                    "half_extents": np.ndarray(shape=(3,)),
                },
                ...
            ]

        `position` and `orientation` describe the cuboid pose relative to the
        mesh geom's local coordinate frame.

        During VAMP environment construction, the current MuJoCo geom pose
        should be composed with this local primitive pose.
        """

        if obj_size is None:
            obj_size = self.object_details["size"]

        def yaw_rot(theta):
            c = np.cos(theta)
            s = np.sin(theta)

            return np.array(
                [
                    [c, -s, 0.0],
                    [s,  c, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=float,
            )

        def make_T(rotation, position):
            T = np.eye(4, dtype=float)
            T[:3, :3] = rotation
            T[:3, 3] = position
            return T

        def transform_points(points, T):
            points_h = np.c_[
                points,
                np.ones(len(points)),
            ]

            return (T @ points_h.T).T[:, :3]

        def box_corners(lx, ly, lz):
            hx = lx / 2.0
            hy = ly / 2.0
            hz = lz / 2.0

            return np.array(
                [
                    [-hx, -hy, -hz],
                    [-hx, -hy,  hz],
                    [-hx,  hy, -hz],
                    [-hx,  hy,  hz],
                    [ hx, -hy, -hz],
                    [ hx, -hy,  hz],
                    [ hx,  hy, -hz],
                    [ hx,  hy,  hz],
                ],
                dtype=float,
            )

        def make_cuboid_primitive(
            position,
            orientation,
            half_extents,
        ):
            """
            Save a cuboid pose in the mesh geom's local frame.
            """

            return {
                "type": "cuboid",
                "position": np.asarray(
                    position,
                    dtype=float,
                ).copy(),
                "orientation": np.asarray(
                    orientation,
                    dtype=float,
                ).reshape(3, 3).copy(),
                "half_extents": np.asarray(
                    half_extents,
                    dtype=float,
                ).copy(),
            }

        def make_cylinder_primitive(
            position,
            orientation,
            radius,
            length,
        ):
            return {
                "type": "cylinder",
                "position": np.asarray(
                    position,
                    dtype=float,
                ).copy(),
                "orientation": np.asarray(
                    orientation,
                    dtype=float,
                ).reshape(3, 3).copy(),
                "radius": float(radius),
                "length": float(length),
            }

        # ================================================================
        # Box swept volume
        # ================================================================
        if self.object_details["type"] == "box":

            if self.env_details["robot"] == "fetch":
                mesh_prefix = "assets/temp"
            else:
                mesh_prefix = "temp"

            lx, ly, lz = map(float, obj_size)

            half_extents = np.array(
                [
                    lx / 2.0,
                    ly / 2.0,
                    lz / 2.0,
                ],
                dtype=float,
            )

            base_box_mesh = trimesh.creation.box(
                extents=[lx, ly, lz],
            )

            box_vertices = []
            box_primitives = []

            yaw_values = np.asarray(
                tcr_intervals["yaw"],
                dtype=float,
            )

            if isinstance(self, LargeObjectEnv):
                num_yaw_samples = 100
            else:
                num_yaw_samples = 4

            yaw_samples = np.linspace(
                yaw_values.min(),
                yaw_values.max(),
                num_yaw_samples,
            )

            for x, y, yaw in itertools.product(
                tcr_intervals["x"],
                tcr_intervals["y"],
                yaw_samples,
            ):
                local_position = np.array(
                    [
                        float(x),
                        float(y),
                        0.0,
                    ],
                    dtype=float,
                )

                local_orientation = yaw_rot(
                    float(yaw)
                )

                # Save the cuboid pose relative to the mesh geom.
                box_primitives.append(
                    make_cuboid_primitive(
                        position=local_position,
                        orientation=local_orientation,
                        half_extents=half_extents,
                    )
                )

                # Construct the same transformed cuboid for the STL.
                T_geom_primitive = make_T(
                    local_orientation,
                    local_position,
                )

                box_mesh = base_box_mesh.copy()
                box_mesh.apply_transform(
                    T_geom_primitive
                )

                box_vertices.append(
                    box_mesh.vertices.copy()
                )

            if not box_vertices:
                raise ValueError(
                    "No cuboids were generated for the box swept volume."
                )

            box_vertices = np.vstack(
                box_vertices
            )

            hull = trimesh.points.PointCloud(
                box_vertices
            ).convex_hull

            mesh_file_name = (
                f"sv_mesh_{sv_count}.stl"
            )

            out_path = (
                Path(self.robot_dir)
                / "assets"
                / "temp"
                / mesh_file_name
            )

            out_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            hull.export(out_path)

            mesh_path_for_xml = (
                f"{mesh_prefix}/{mesh_file_name}"
            )

            mesh_name = (
                f"swept_volume_mesh_{sv_count}"
            )

            body_name = (
                f"swept_volume_{sv_count}"
            )

            geom_name = (
                f"sv_mesh_{sv_count}"
            )

            rgba = [0.8, 0.8, 0.8, 1.0]

            joint_xml = (
                ""
                if fixed
                else (
                    f'<joint name="{body_name}_free" '
                    f'type="free"/>'
                )
            )

            # geom name -> primitive list
            self.swept_volume_primitives[
                geom_name
            ] = box_primitives

            return f"""
            <asset>
                <mesh
                    name="{mesh_name}"
                    file="{mesh_path_for_xml}"/>
            </asset>

            <body name="{body_name}" pos="0 0 0">
                {joint_xml}

                <geom
                    name="{geom_name}"
                    type="mesh"
                    mesh="{mesh_name}"
                    rgba="{' '.join(map(str, rgba))}"/>
            </body>
            """

        # ================================================================
        # Microwave swept volume
        # ================================================================
        elif self.object_details["type"] == "microwave":

            out_dir = (
                Path(self.robot_dir)
                / "assets"
                / "temp"
            )

            out_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            if self.env_details["robot"] == "fetch":
                mesh_prefix = "assets/temp"
            else:
                mesh_prefix = "temp"

            body_name = "swept_volume_0"

            joint_xml = (
                ""
                if fixed
                else (
                    f'<joint name="{body_name}_free" '
                    f'type="free"/>'
                )
            )

            rgba_body = [0.8, 0.8, 0.8, 1.0]
            rgba_door = [0.2, 0.2, 0.25, 1.0]
            rgba_handle = [0.8, 0.2, 0.2, 1.0]

            asset_xml = []
            geom_xml = []

            # ------------------------------------------------------------
            # Microwave body
            # ------------------------------------------------------------
            body_lx, body_ly, body_lz = map(
                float,
                self.object_details["size"],
            )

            body_half_extents = np.array(
                [
                    body_lx / 2.0,
                    body_ly / 2.0,
                    body_lz / 2.0,
                ],
                dtype=float,
            )

            body_corners = box_corners(
                body_lx,
                body_ly,
                body_lz,
            )

            body_points = []
            body_primitives = []

            for x, y, yaw in itertools.product(
                tcr_intervals["x"],
                tcr_intervals["y"],
                tcr_intervals["yaw"],
            ):
                local_position = np.array(
                    [
                        float(x),
                        float(y),
                        0.0,
                    ],
                    dtype=float,
                )

                local_orientation = yaw_rot(
                    float(yaw)
                )

                T_geom_primitive = make_T(
                    local_orientation,
                    local_position,
                )

                body_points.append(
                    transform_points(
                        body_corners,
                        T_geom_primitive,
                    )
                )

                body_primitives.append(
                    make_cuboid_primitive(
                        position=local_position,
                        orientation=local_orientation,
                        half_extents=body_half_extents,
                    )
                )

            if not body_points:
                raise ValueError(
                    "No cuboids were generated for the "
                    "microwave body swept volume."
                )

            body_points = np.vstack(
                body_points
            )

            body_hull = trimesh.points.PointCloud(
                body_points
            ).convex_hull

            body_mesh_name = (
                "microwave_body_sv_mesh"
            )

            body_mesh_filename = (
                "microwave_body_sv.stl"
            )

            body_mesh_file = (
                f"{mesh_prefix}/{body_mesh_filename}"
            )

            body_hull.export(
                out_dir / body_mesh_filename
            )

            asset_xml.append(
                f"""
                <mesh
                    name="{body_mesh_name}"
                    file="{body_mesh_file}"/>
                """
            )

            body_geom_name = (
                "microwave_body_sv"
            )

            geom_xml.append(
                f"""
                <geom
                    name="{body_geom_name}"
                    type="mesh"
                    mesh="{body_mesh_name}"
                    rgba="{' '.join(map(str, rgba_body))}"/>
                """
            )

            self.swept_volume_primitives[
                body_geom_name
            ] = body_primitives

            # ------------------------------------------------------------
            # Microwave door and handle
            # ------------------------------------------------------------
            door_lx, door_ly, door_lz = map(
                float,
                self.object_details["door_size"],
            )

            handle_lx, handle_ly, handle_lz = map(
                float,
                self.object_details["handle_size"],
            )

            door_half_extents = np.array(
                [
                    door_lx / 2.0,
                    door_ly / 2.0,
                    door_lz / 2.0,
                ],
                dtype=float,
            )

            handle_half_extents = np.array(
                [
                    handle_lx / 2.0,
                    handle_ly / 2.0,
                    handle_lz / 2.0,
                ],
                dtype=float,
            )

            handle_pos_door = np.asarray(
                self.object_details[
                    "handle_pos_door"
                ],
                dtype=float,
            )

            hinge_pos_body = np.asarray(
                self.object_details[
                    "hinge_pos_body"
                ],
                dtype=float,
            )

            door_origin_closed_body = np.asarray(
                self.object_details[
                    "door_origin_closed_body"
                ],
                dtype=float,
            )

            base_door_mesh = trimesh.creation.box(
                extents=[
                    door_lx,
                    door_ly,
                    door_lz,
                ],
            )

            base_handle_mesh = trimesh.creation.box(
                extents=[
                    handle_lx,
                    handle_ly,
                    handle_lz,
                ],
            )

            # Handle geometry is offset from the door origin.
            base_handle_mesh.apply_translation(
                handle_pos_door
            )

            door_meshes = []
            handle_meshes = []

            door_primitives = []
            handle_primitives = []

            for x, y, yaw, phi in itertools.product(
                tcr_intervals["x"],
                tcr_intervals["y"],
                tcr_intervals["yaw"],
                tcr_intervals["door"],
            ):
                R_geom_body = yaw_rot(
                    float(yaw)
                )

                p_geom_body = np.array(
                    [
                        float(x),
                        float(y),
                        0.0,
                    ],
                    dtype=float,
                )

                T_geom_body = make_T(
                    R_geom_body,
                    p_geom_body,
                )

                R_phi = yaw_rot(
                    float(phi)
                )

                T_body_hinge = make_T(
                    np.eye(3, dtype=float),
                    hinge_pos_body,
                )

                T_hinge_rotation = make_T(
                    R_phi,
                    np.zeros(3, dtype=float),
                )

                T_hinge_door = make_T(
                    np.eye(3, dtype=float),
                    door_origin_closed_body
                    - hinge_pos_body,
                )

                T_body_door = (
                    T_body_hinge
                    @ T_hinge_rotation
                    @ T_hinge_door
                )

                # This is the transform used to place the door cuboid
                # in the exported door mesh's coordinate frame.
                T_geom_door = (
                    T_geom_body
                    @ T_body_door
                )

                door_local_position = (
                    T_geom_door[:3, 3].copy()
                )

                door_local_orientation = (
                    T_geom_door[:3, :3].copy()
                )

                door_mesh = base_door_mesh.copy()
                door_mesh.apply_transform(
                    T_geom_door
                )
                door_meshes.append(door_mesh)

                door_primitives.append(
                    make_cuboid_primitive(
                        position=door_local_position,
                        orientation=door_local_orientation,
                        half_extents=door_half_extents,
                    )
                )

                # The base handle mesh already contains handle_pos_door,
                # so applying T_geom_door matches the exported mesh.
                handle_mesh = base_handle_mesh.copy()
                handle_mesh.apply_transform(
                    T_geom_door
                )
                handle_meshes.append(handle_mesh)

                handle_local_position = (
                    door_local_position
                    + door_local_orientation
                    @ handle_pos_door
                )

                handle_local_orientation = (
                    door_local_orientation
                )

                handle_primitives.append(
                    make_cuboid_primitive(
                        position=handle_local_position,
                        orientation=handle_local_orientation,
                        half_extents=handle_half_extents,
                    )
                )

            if not door_meshes:
                raise ValueError(
                    "No cuboids were generated for the "
                    "microwave door swept volume."
                )

            try:
                door_union = trimesh.boolean.union(
                    door_meshes,
                    engine="manifold",
                )

                handle_union = trimesh.boolean.union(
                    handle_meshes,
                    engine="manifold",
                )

            except Exception as e:
                raise RuntimeError(
                    "Boolean union failed. Install manifold3d with "
                    "`pip install manifold3d`, or adjust the sampling. "
                    f"Original error: {e}"
                ) from e

            door_mesh_name = (
                "microwave_door_sv_mesh"
            )

            door_mesh_filename = (
                "microwave_door_sv_union.stl"
            )

            door_mesh_file = (
                f"{mesh_prefix}/{door_mesh_filename}"
            )

            door_union.export(
                out_dir / door_mesh_filename
            )

            handle_mesh_name = (
                "microwave_handle_sv_mesh"
            )

            handle_mesh_filename = (
                "microwave_handle_sv_union.stl"
            )

            handle_mesh_file = (
                f"{mesh_prefix}/{handle_mesh_filename}"
            )

            handle_union.export(
                out_dir / handle_mesh_filename
            )

            asset_xml.append(
                f"""
                <mesh
                    name="{door_mesh_name}"
                    file="{door_mesh_file}"/>
                """
            )

            asset_xml.append(
                f"""
                <mesh
                    name="{handle_mesh_name}"
                    file="{handle_mesh_file}"/>
                """
            )

            door_geom_name = (
                "microwave_door_sv"
            )

            handle_geom_name = (
                "microwave_handle_sv"
            )

            hx, hy, hz = hinge_pos_body

            geom_xml.append(
                f"""
                <body
                    name="sv_door_hinge_frame"
                    pos="{hx} {hy} {hz}">

                    <joint
                        name="sv_door_hinge"
                        type="hinge"
                        axis="0 0 -1"
                        limited="false"/>

                    <geom
                        name="{door_geom_name}"
                        type="mesh"
                        mesh="{door_mesh_name}"
                        rgba="{' '.join(map(str, rgba_door))}"/>

                    <geom
                        name="{handle_geom_name}"
                        type="mesh"
                        mesh="{handle_mesh_name}"
                        rgba="{' '.join(map(str, rgba_handle))}"/>
                </body>
                """
            )

            self.swept_volume_primitives[
                door_geom_name
            ] = door_primitives

            self.swept_volume_primitives[
                handle_geom_name
            ] = handle_primitives

            return f"""
            <asset>
                {''.join(asset_xml)}
            </asset>

            <body name="{body_name}" pos="0 0 0">
                {joint_xml}

                {''.join(geom_xml)}
            </body>
            """

        # ================================================================
        # Cylinder swept volume
        # ================================================================
        elif self.object_details["type"] == "cylinder":

            if self.env_details["robot"] == "fetch":
                mesh_prefix = "assets/temp"
            else:
                mesh_prefix = "temp"

            radius, height = map(float, obj_size)

            x_values = np.asarray(
                tcr_intervals["x"],
                dtype=float,
            )

            y_values = np.asarray(
                tcr_intervals["y"],
                dtype=float,
            )

            if x_values.size == 0 or y_values.size == 0:
                raise ValueError(
                    "Cylinder swept volume requires nonempty "
                    "x and y intervals."
                )

            x_min = float(x_values.min())
            x_max = float(x_values.max())
            y_min = float(y_values.min())
            y_max = float(y_values.max())

            x_center = (x_min + x_max) / 2.0
            y_center = (y_min + y_max) / 2.0

            x_span = x_max - x_min
            y_span = y_max - y_min

            identity_rotation = np.eye(3, dtype=float)

            cylinder_primitives = []

            # The swept cylinder is a rounded rectangular prism.
            # Represent its center using two overlapping cuboids.
            if x_span > 1e-12:
                cylinder_primitives.append(
                    make_cuboid_primitive(
                        position=[
                            x_center,
                            y_center,
                            0.0,
                        ],
                        orientation=identity_rotation,
                        half_extents=[
                            x_span / 2.0,
                            y_span / 2.0 + radius,
                            height / 2.0,
                        ],
                    )
                )

            if y_span > 1e-12:
                cylinder_primitives.append(
                    make_cuboid_primitive(
                        position=[
                            x_center,
                            y_center,
                            0.0,
                        ],
                        orientation=identity_rotation,
                        half_extents=[
                            x_span / 2.0 + radius,
                            y_span / 2.0,
                            height / 2.0,
                        ],
                    )
                )

            # Add cylinders at the corners of the x/y sweep.
            corner_positions = {
                (x_min, y_min),
                (x_min, y_max),
                (x_max, y_min),
                (x_max, y_max),
            }

            for x, y in corner_positions:
                cylinder_primitives.append(
                    make_cylinder_primitive(
                        position=[x, y, 0.0],
                        orientation=identity_rotation,
                        radius=radius,
                        length=height,
                    )
                )

            # Construct the STL from cylinders placed at the corners.
            # Their convex hull is the rounded rectangular sweep.
            base_cylinder_mesh = trimesh.creation.cylinder(
                radius=radius,
                height=height,
                sections=64,
            )

            cylinder_vertices = []

            for x, y in corner_positions:
                cylinder_mesh = base_cylinder_mesh.copy()

                cylinder_mesh.apply_translation(
                    [x, y, 0.0]
                )

                cylinder_vertices.append(
                    cylinder_mesh.vertices.copy()
                )

            if not cylinder_vertices:
                raise ValueError(
                    "No cylinders were generated for the "
                    "cylinder swept volume."
                )

            cylinder_vertices = np.vstack(
                cylinder_vertices
            )

            hull = trimesh.points.PointCloud(
                cylinder_vertices
            ).convex_hull

            mesh_file_name = f"sv_mesh_{sv_count}.stl"

            # out_path = (
            #     Path(self.robot_dir)
            #     / "assets"
            #     / "temp"
            #     / mesh_file_name
            # )

            if self.env_details["robot"] == "fetch":
                out_dir = (
                    Path(self.robot_dir)
                    / "assets"
                    / "temp"
                )
            else:
                out_dir = (
                    Path(self.robot_dir)
                    / "temp"
                )

            out_path = out_dir / mesh_file_name

            out_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            hull.export(out_path)

            mesh_path_for_xml = (
                f"{mesh_prefix}/{mesh_file_name}"
            )

            mesh_name = (
                f"swept_volume_mesh_{sv_count}"
            )

            body_name = (
                f"swept_volume_{sv_count}"
            )

            geom_name = (
                f"sv_mesh_{sv_count}"
            )

            rgba = [0.8, 0.8, 0.8, 1.0]

            joint_xml = (
                ""
                if fixed
                else (
                    f'<joint name="{body_name}_free" '
                    f'type="free"/>'
                )
            )

            self.swept_volume_primitives[
                geom_name
            ] = cylinder_primitives

            return f"""
            <asset>
                <mesh
                    name="{mesh_name}"
                    file="{mesh_path_for_xml}"/>
            </asset>

            <body name="{body_name}" pos="0 0 0">
                {joint_xml}

                <geom
                    name="{geom_name}"
                    type="mesh"
                    mesh="{mesh_name}"
                    rgba="{' '.join(map(str, rgba))}"/>
            </body>
            """

        else:
            raise ValueError(
                "Unsupported object type for swept-volume generation: "
                f"{self.object_details['type']}"
            )

    def build_xml(self, scene_yaml, parent_body_name="env_name", skip_ids=None, rgba=None):
        """Return xml for environment"""

        if skip_ids is None:
            skip_ids = set()

        # Default to MuJoCo default gray if not provided
        if rgba is None:
            rgba = [0.5, 0.5, 0.5, 1]
            rgba = [0.15, 1, 0.15, 1]
            rgba = [0.133, 0.6, 0.329, 1]

        with open(scene_yaml, "r") as f:
            scene_yaml_data = yaml.safe_load(f)

        objs = scene_yaml_data["world"]["collision_objects"]
        lines = []
        lines.append(f'<body name="{parent_body_name}" pos="0 0 0">')

        for obj in objs:
            obj_id = obj.get("id", "")
            if obj_id in skip_ids:
                continue

            prim = obj["primitives"][0]
            pose = obj["primitive_poses"][0]

            pos = pose["position"]
            quat_xyzw = pose["orientation"]
            quat_wxyz = self.quat_xyzw_to_wxyz(quat_xyzw)

            prim_type = prim["type"].lower()
            dims = prim["dimensions"]
            self.env_details['collision_geoms'].append(obj_id)

            if prim_type == "box":
                size = [dims[0] / 2.0, dims[1] / 2.0, dims[2] / 2.0]
                mj_type = "box"
                mj_size = size

            elif prim_type == "cylinder":
                height, radius = dims[0], dims[1]
                mj_type = "cylinder"
                mj_size = [radius, height / 2.0]

            else:
                raise ValueError(
                    f"Unsupported primitive type: {prim_type} for id={obj_id}"
                )

            lines.append(
                f'  <geom name="{obj_id}" type="{mj_type}" '
                f'pos="{self.fmt(pos)}" quat="{self.fmt(quat_wxyz)}" '
                f'size="{self.fmt(mj_size)}" '
                f'contype="1" conaffinity="1" '
                f'rgba="{self.fmt(rgba)}"/>'
            )

        lines.append("</body>")
        return "\n".join(lines)

    def fmt(self, v):
        """Formatting for XML"""
        return " ".join(f"{x:.6g}" for x in v)

    def quat_xyzw_to_wxyz(self, quat_xyzw):
        """Convert to wxyz quats"""
        quat_wxyz = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
        return quat_wxyz

    def build_model(self, xml_path, xmls_to_add):
        """
        Build final xml and model
        xml_path: desired path for model's xml
        xmls_to_add: list of xml fragments containing <asset> and/or <body>
        """

        xml_path = Path(xml_path)

        unique_xml_path = xml_path.with_name(
            f"{xml_path.stem}_{os.getpid()}{xml_path.suffix}"
        )

        # Use the unique path from this point onward.
        xml_path = str(unique_xml_path)

        asset_blocks = []
        body_blocks = []

        for frag in xmls_to_add:
            # split fragment into asset + body parts
            if "<asset" in frag:
                start = frag.find("<asset")
                end = frag.find("</asset>") + len("</asset>")
                asset_blocks.append(frag[start:end])
                frag = frag[:start] + frag[end:]

            body_blocks.append(frag)

        curr_xml = f"""
        <mujoco model="test_world">
            <include file="{self.base_xml}"/>

            <asset>
        """
        for a in asset_blocks:
            # strip outer <asset> wrapper
            inner = a.replace("<asset>", "").replace("</asset>", "")
            curr_xml += inner + "\n"

        curr_xml += """
            </asset>

            <worldbody>
        """

        for b in body_blocks:
            curr_xml += b + "\n"

        curr_xml += """
            </worldbody>
        </mujoco>
        """

        with open(xml_path, "w") as f:
            f.write(curr_xml)
        model = mujoco.MjModel.from_xml_path(xml_path)
        os.remove(xml_path)
        # model = mujoco.MjModel.from_xml_string(curr_xml)

        data = mujoco.MjData(model)
        return model, data

    def compute_sv_params(self, object_dims, object_configs):
        """Compute swept volume dimensions"""
        object_configs = np.asarray(object_configs, dtype=np.float64)
        xdim, ydim, zdim = object_dims
        if object_configs.ndim == 2:
            object_configs = object_configs[None, :, :]

        x_lower = object_configs[:, 0, 0]
        x_upper = object_configs[:, 0, 1]
        y_lower = object_configs[:, 1, 0]
        y_upper = object_configs[:, 1, 1]
        z = object_configs[:, 2, 0]

        R_cyl = float(np.round(0.5 * np.sqrt(xdim**2 + ydim**2), 5))
        cx = 0.5 * (x_upper + x_lower)
        cy = 0.5 * (y_upper + y_lower)

        b1_size = np.array(
            [
                (x_upper - x_lower) + 2 * R_cyl,
                (y_upper - y_lower),
                np.full_like(cx, zdim),
            ]
        ).T  # (B,3)

        b2_size = np.array(
            [
                (x_upper - x_lower),
                (y_upper - y_lower) + 2 * R_cyl,
                np.full_like(cx, zdim),
            ]
        ).T  # (B,3)

        b_pos = np.stack([cx, cy, z], axis=1)  # (B,3)

        corners = np.stack(
            [
                np.stack([x_lower, y_lower, z], axis=1),
                np.stack([x_lower, y_upper, z], axis=1),
                np.stack([x_upper, y_lower, z], axis=1),
                np.stack([x_upper, y_upper, z], axis=1),
            ],
            axis=1,
        )  # (B,4,3)

        return R_cyl, b_pos, b1_size, b2_size, corners

    def compute_cyl_sv_params(self, object_dims, object_configs):
        """Compute swept volume dimensions"""
        object_configs = np.asarray(object_configs, dtype=np.float64)
        # xdim, ydim, zdim = object_dims
        rdim, zdim = object_dims
        if object_configs.ndim == 2:
            object_configs = object_configs[None, :, :]

        x_lower = object_configs[:, 0, 0]
        x_upper = object_configs[:, 0, 1]
        y_lower = object_configs[:, 1, 0]
        y_upper = object_configs[:, 1, 1]
        z = object_configs[:, 2, 0]

        # R_cyl = float(np.round(0.5 * np.sqrt(xdim**2 + ydim**2), 5))
        R_cyl = rdim
        cx = 0.5 * (x_upper + x_lower)
        cy = 0.5 * (y_upper + y_lower)

        b1_size = np.array(
            [
                (x_upper - x_lower) + 2 * R_cyl,
                (y_upper - y_lower),
                np.full_like(cx, zdim),
            ]
        ).T  # (B,3)

        b2_size = np.array(
            [
                (x_upper - x_lower),
                (y_upper - y_lower) + 2 * R_cyl,
                np.full_like(cx, zdim),
            ]
        ).T  # (B,3)

        b_pos = np.stack([cx, cy, z], axis=1)  # (B,3)

        corners = np.stack(
            [
                np.stack([x_lower, y_lower, z], axis=1),
                np.stack([x_lower, y_upper, z], axis=1),
                np.stack([x_upper, y_lower, z], axis=1),
                np.stack([x_upper, y_upper, z], axis=1),
            ],
            axis=1,
        )  # (B,4,3)

        return R_cyl, b_pos, b1_size, b2_size, corners

    def get_geom_pose(self, geom_name, model=None, data=None, as_matrix=False):
        """
        Return world pose of a MuJoCo geom.

        Does not call mj_forward; assumes data is already current.

        Returns:
            if as_matrix=True:
                4x4 homogeneous transform
            else:
                (pos, R)
        """
        if model is None:
            model = self.model
        if data is None:
            data = self.data

        gid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            geom_name,
        )

        if gid == -1:
            raise ValueError(f"Could not find geom: {geom_name}")

        pos = data.geom_xpos[gid].copy()
        R = data.geom_xmat[gid].reshape(3, 3).copy()

        rpy = Rotation.from_matrix(R).as_euler(
            "xyz",
            degrees=False
        )

        if geom_name == "microwave_handle_sv":
            rpy = np.array([rpy[0], rpy[1], rpy[2] - np.pi/2])
            rot_mat = Rotation.from_euler(
                "xyz",
                rpy,
                degrees=False
            ).as_matrix()
        else:
            rot_mat = R

        if not as_matrix:
            return pos, rpy

        T = np.eye(4)
        T[:3, :3] = rot_mat
        T[:3, 3] = pos
        return T

    def object_xml(
        self,
        object_dims,
        object_pose,
        fixed=False,
        temp=False,
        name=None,
    ):
        if self.object_details["type"] == "microwave":
            return self.microwave_object_xml(
                object_dims,
                object_pose,
                fixed,
                temp,
            )

        if self.object_details["type"] == "box":
            return self.cube_object_xml(
                object_dims,
                object_pose,
                fixed,
                temp,
                name=name,
            )

        if self.object_details["type"] == "cylinder":
            return self.cylinder_object_xml(
                object_dims,
                object_pose,
                fixed,
                temp,
                name=name,
            )

        raise ValueError(
            f"Unsupported object type: {self.object_details['type']}"
        )
        
    def microwave_object_xml(self, object_dims, object_pose, fixed=False, temp=False):
        """Create articulated hollow microwave XML string.

        object_dims: full outer dimensions [lx, ly, lz]
        object_pose: [x, y, z, yaw]
        self.object_details["door_size"]: full door dimensions [lx, ly, lz]
        """

        rgba_shell = [0.7, 0.7, 0.7, 1.0]
        rgba_back = [0.55, 0.55, 0.55, 1.0]
        rgba_door = [0.2, 0.2, 0.25, 0.7]
        rgba_handle = [0.05, 0.05, 0.05, 1.0]

        name = "microwave_object"

        object_x, object_y, object_z, object_yaw = object_pose

        half = 0.5 * float(object_yaw)
        qw = np.cos(half)
        qx = 0.0
        qy = 0.0
        qz = np.sin(half)

        lx, ly, lz = map(float, object_dims)
        hx, hy, hz = lx / 2.0, ly / 2.0, lz / 2.0

        door_lx, door_ly, door_lz = map(float, self.object_details["door_size"])
        door_hx, door_hy, door_hz = door_lx / 2.0, door_ly / 2.0, door_lz / 2.0

        handle_lx, handle_ly, handle_lz = map(float, self.object_details["handle_size"])
        handle_hx, handle_hy, handle_hz = handle_lx / 2.0, handle_ly / 2.0, handle_lz / 2.0


        wall_thickness = self.object_details.get("wall_thickness", 0.01)
        wt = wall_thickness / 2.0

        joint_xml = "" if fixed else f'<joint name="{name}_free" type="free"/>'

        # Door hinge at front-left edge.
        # Same convention as earlier:
        #   microwave x-axis: back/front
        #   microwave y-axis: left/right
        #   microwave z-axis: vertical
        front_x = -(hx + door_hx)
        hinge_y = -(-hy)
        door_center_y = -(door_ly / 2.0)

        handle_x = -door_lx
        handle_y = -0.35 * door_ly
        handle_z = 0.0

        self.object_details['hinge_pos_body'] = [front_x, hinge_y, 0]
        self.object_details['door_origin_closed_body'] = [0, door_center_y, 0]
        self.object_details['handle_pos_door'] = [handle_x, handle_y, handle_z]

        obj_xml = f"""
        <body name="{name}"
            pos="{object_x} {object_y} {object_z}"
            quat="{qw} {qx} {qy} {qz}">
            {joint_xml}

            <!-- Hollow microwave shell: open front -->

            <geom name="mw_bottom" type="box"
                pos="0 0 {-hz + wt}"
                size="{hx} {hy} {wt}"
                rgba="{rgba_shell[0]} {rgba_shell[1]} {rgba_shell[2]} {rgba_shell[3]}"/>

            <geom name="mw_top" type="box"
                pos="0 0 {hz - wt}"
                size="{hx} {hy} {wt}"
                rgba="{rgba_shell[0]} {rgba_shell[1]} {rgba_shell[2]} {rgba_shell[3]}"/>

            <geom name="mw_left" type="box"
                pos="0 {-hy + wt} 0"
                size="{hx} {wt} {hz}"
                rgba="{rgba_shell[0]} {rgba_shell[1]} {rgba_shell[2]} {rgba_shell[3]}"/>

            <geom name="mw_right" type="box"
                pos="0 {hy - wt} 0"
                size="{hx} {wt} {hz}"
                rgba="{rgba_shell[0]} {rgba_shell[1]} {rgba_shell[2]} {rgba_shell[3]}"/>

            <geom name="mw_back" type="box"
                pos="{-(-hx + wt)} 0 0"
                size="{wt} {hy} {hz}"
                rgba="{rgba_back[0]} {rgba_back[1]} {rgba_back[2]} {rgba_back[3]}"/>

            <!-- Door hinged at front-left edge -->
            <body name="microwave_door_hinge_frame" pos="{front_x} {hinge_y} 0">
                <joint name="microwave_door_hinge"
                    type="hinge"
                    axis="0 0 -1"
                    range="0 1.5708"
                    limited="true"/>

                <!-- Door center offset from hinge along +y -->
                <body name="microwave_door" pos="0 {door_center_y} 0">
                    <geom name="mw_door" type="box"
                        pos="0 0 0"
                        size="{door_hx} {door_hy} {door_hz}"
                        rgba="{rgba_door[0]} {rgba_door[1]} {rgba_door[2]} {rgba_door[3]}"/>

                    <geom name="mw_handle" type="box"
                        pos="{handle_x} {handle_y} {handle_z}"
                        size="{handle_hx} {handle_hy} {handle_hz}"
                        rgba="{rgba_handle[0]} {rgba_handle[1]} {rgba_handle[2]} {rgba_handle[3]}"/>
                </body>
            </body>
        </body>
        """

        if not temp:
            self.env_details["collision_geoms"].extend([
                "mw_bottom",
                "mw_top",
                "mw_left",
                "mw_right",
                "mw_back",
                "mw_door",
                "mw_handle",
            ])

        return obj_xml

    def debug_sphere_xml(
        self,
        name,
        pos,
        radius=0.01,
        rgba=(1.0, 0.0, 0.0, 1.0),
        contype=0,
        conaffinity=0,
    ):
        """Return MJCF XML for a small non-colliding debug sphere."""
        x, y, z = pos
        r, g, b, a = rgba

        return f"""
        <body name="{name}_body" pos="{x} {y} {z}">
            <geom name="{name}"
                type="sphere"
                size="{radius}"
                rgba="{r} {g} {b} {a}"
                contype="{contype}"
                conaffinity="{conaffinity}"/>
        </body>
        """

    def cube_object_xml(self, object_dims, object_pose, fixed=False, temp=False, name=None):
        """Create cube object xml string"""
        rgba = [0.8, 0.2, 0.2, 1]
        
        if name is None:
            name = "cube_object"
        
        object_x, object_y, object_z, object_yaw = object_pose
        half = 0.5 * float(object_yaw)
        qw = np.cos(half)
        qx = 0.0
        qy = 0.0
        qz = np.sin(half)

        joint_xml = "" if fixed else f'<joint name="{name}_free" type="free"/>'

        obj_xml = f"""
        <body name="{name}" 
            pos="{object_x} {object_y} {object_z}"
            quat="{qw} {qx} {qy} {qz}">
            {joint_xml}

            <!-- boxes centered at body origin -->
            <geom name="{name}_geom" type="box" pos="0 0 0"
                size="{object_dims[0]/2} {object_dims[1]/2} {object_dims[2]/2}"
                rgba="{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}"/>
        </body>
        """
        # self.collision_geoms.append(f"{name}_geom")
        if not temp:
            self.env_details['collision_geoms'].append(f"{name}_geom")
        return obj_xml

    def cylinder_object_xml(
        self,
        object_dims,
        object_pose,
        fixed=False,
        temp=False,
        name=None,
    ):
        """Create a cylinder-object XML fragment."""

        rgba = [0.8, 0.2, 0.2, 1.0]

        if name is None:
            name = "cube_object"

        radius, height = object_dims
        object_x, object_y, object_z, object_yaw = object_pose

        half_yaw = 0.5 * float(object_yaw)

        qw = np.cos(half_yaw)
        qx = 0.0
        qy = 0.0
        qz = np.sin(half_yaw)

        joint_xml = (
            ""
            if fixed
            else f'<joint name="{name}_free" type="free"/>'
        )

        object_xml = f"""
        <body name="{name}"
            pos="{object_x} {object_y} {object_z}"
            quat="{qw} {qx} {qy} {qz}">
            {joint_xml}

            <geom name="{name}_geom"
                type="cylinder"
                pos="0 0 0"
                size="{radius} {height / 2.0}"
                rgba="{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}"/>
        </body>
        """

        if not temp:
            self.env_details["collision_geoms"].append(
                f"{name}_geom"
            )

        return object_xml
    
    def move_object(self, object_pose, model=None, data=None):

        if self.object_details["type"] == "box":
            self.move_cube_object(object_pose, model, data)

        elif self.object_details["type"] == "cylinder":
            self.move_xml_object(
                "cube_object",
                object_pose,
                model,
                data,
            )

        elif self.object_details["type"] == "microwave":
            self.move_xml_object(
                "microwave_object",
                object_pose,
                model,
                data,
            )

        else:
            raise ValueError(
                f"Unsupported object type: {self.object_details['type']}"
            )
        
    # def move_xml_object(self, object_name, object_pose, model, data):
    #     """
    #     Move cube_object to (x, y, z, yaw) by writing into its free joint qpos.
    #     object_pose: iterable length-4: (x, y, z, yaw) in radians
    #     """
    #     if object_name == "microwave_object":
    #         x, y, z, yaw = object_pose 

    #         if model is None and data is None:
    #             model = self.model
    #             data = self.data

    #         # jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_object_free")
    #         # qadr = self.model.jnt_qposadr[jid]
    #         # vadr = self.model.jnt_dofadr[jid]

    #         jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{object_name}_free")
    #         qadr = model.jnt_qposadr[jid]
    #         vadr = model.jnt_dofadr[jid]

    #         half = 0.5 * float(yaw)
    #         qw = np.cos(half)
    #         qx = 0.0
    #         qy = 0.0
    #         qz = np.sin(half)

    #         # free joint qpos layout: [x y z qw qx qy qz]
    #         data.qpos[qadr:qadr+7] = [x, y, z, qw, qx, qy, qz]
    #         data.qvel[vadr:vadr+6] = 0.0

    #         mujoco.mj_forward(model, data)

    #     else:
    #         raise ValueError(f"Unsupported XML object: {object_name}")

    def move_xml_object(
        self,
        object_name,
        object_pose,
        model=None,
        data=None,
    ):
        """
        Move a free-joint XML object to an x, y, z, yaw pose.

        The object's free joint must be named:
            <object_name>_free
        """

        if model is None:
            model = self.model

        if data is None:
            data = self.data

        x, y, z, yaw = object_pose

        joint_name = f"{object_name}_free"

        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )

        if joint_id == -1:
            raise ValueError(
                f"Could not find free joint: {joint_name}"
            )

        qpos_address = model.jnt_qposadr[joint_id]
        qvel_address = model.jnt_dofadr[joint_id]

        half_yaw = 0.5 * float(yaw)

        quaternion_wxyz = [
            np.cos(half_yaw),
            0.0,
            0.0,
            np.sin(half_yaw),
        ]

        # Free-joint qpos:
        # [x, y, z, qw, qx, qy, qz]
        data.qpos[qpos_address:qpos_address + 7] = [
            x,
            y,
            z,
            *quaternion_wxyz,
        ]

        # Free-joint velocity:
        # [vx, vy, vz, wx, wy, wz]
        data.qvel[qvel_address:qvel_address + 6] = 0.0

        mujoco.mj_forward(model, data)
        
    def move_xml_joint(self, joint_name, joint_value, model=None, data=None):
        """
        Set a scalar joint value (hinge/slide).

        Example:
            move_xml_joint("microwave_door_hinge", np.pi/4)
        """

        if model is None:
            model = self.model
        if data is None:
            data = self.data

        jid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name
        )

        if jid == -1:
            raise ValueError(f"Could not find joint: {joint_name}")

        qadr = model.jnt_qposadr[jid]
        vadr = model.jnt_dofadr[jid]

        data.qpos[qadr] = joint_value
        data.qvel[vadr] = 0.0

        mujoco.mj_forward(model, data)

    # def move_cube_object(self, object_pose, model=None, data=None):
    #     """
    #     Move cube_object to (x, y, z, yaw) by writing into its free joint qpos.
    #     object_pose: iterable length-4: (x, y, z, yaw) in radians
    #     """
    #     x, y, z, yaw = object_pose 

    #     if model is None and data is None:
    #         model = self.model
    #         data = self.data

    #     # jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_object_free")
    #     # qadr = self.model.jnt_qposadr[jid]
    #     # vadr = self.model.jnt_dofadr[jid]

    #     jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_object_free")
    #     qadr = model.jnt_qposadr[jid]
    #     vadr = model.jnt_dofadr[jid]

    #     half = 0.5 * float(yaw)
    #     qw = np.cos(half)
    #     qx = 0.0
    #     qy = 0.0
    #     qz = np.sin(half)

    #     # free joint qpos layout: [x y z qw qx qy qz]
    #     data.qpos[qadr:qadr+7] = [x, y, z, qw, qx, qy, qz]
    #     data.qvel[vadr:vadr+6] = 0.0

    #     mujoco.mj_forward(model, data)

    def move_cube_object(self, object_pose, model=None, data=None):
        """
        Move the active cube object to (x, y, z, yaw).

        Standard environments:
            object_pose = (x, y, z, yaw)

        AllStableEnv:
            object_pose = (face, x, y, z, yaw)

            The object corresponding to `face` is moved to the requested pose,
            while the other two face-specific objects are moved to a dummy pose.
        """
        if model is None:
            model = self.model

        if data is None:
            data = self.data

        # This is a pose, unlike move_swept_volume's interval-based dummy_config.
        dummy_pose = (1.0, 1.0, 0.0, 0.0)

        def move_one_object(joint_name, pose):
            if len(pose) != 4:
                raise ValueError(
                    f"Expected pose (x, y, z, yaw), got {pose}"
                )

            x, y, z, yaw = map(float, pose)

            jid = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_name,
            )

            if jid == -1:
                raise RuntimeError(
                    f"Could not find free joint '{joint_name}'"
                )

            qadr = model.jnt_qposadr[jid]
            vadr = model.jnt_dofadr[jid]

            half = 0.5 * yaw

            qw = np.cos(half)
            qx = 0.0
            qy = 0.0
            qz = np.sin(half)

            # MuJoCo free-joint qpos:
            # [x, y, z, qw, qx, qy, qz]
            data.qpos[qadr:qadr + 7] = [
                x,
                y,
                z,
                qw,
                qx,
                qy,
                qz,
            ]

            data.qvel[vadr:vadr + 6] = 0.0

        if isinstance(self, AllStableEnv):
            if len(object_pose) != 5:
                raise ValueError(
                    "AllStableEnv expects object_pose in the form "
                    "(face, x, y, z, yaw), "
                    f"got {object_pose}"
                )

            face_in_contact = object_pose[0]
            real_object_pose = object_pose[1:]

            face_to_joint = {
                "xy": "cube_object_0_free",
                "yz": "cube_object_1_free",
                "zx": "cube_object_2_free",
            }

            if face_in_contact not in face_to_joint:
                raise ValueError(
                    f"Unknown face value: {face_in_contact}. "
                    f"Expected one of {tuple(face_to_joint)}"
                )

            for face, joint_name in face_to_joint.items():
                if face == face_in_contact:
                    move_one_object(joint_name, real_object_pose)
                else:
                    move_one_object(joint_name, dummy_pose)

        else:
            if len(object_pose) != 4:
                raise ValueError(
                    "Expected object_pose in the form "
                    "(x, y, z, yaw), "
                    f"got {object_pose}"
                )

            move_one_object(
                "cube_object_free",
                object_pose,
            )

        # Update all derived MuJoCo quantities once after moving every object.
        mujoco.mj_forward(model, data)

    # def get_geom_pose(self, geom_name, model=None, data=None):
        
    #     if model is None and data is None:
    #         model = self.model
    #         data = self.data

    #     geom_id = mujoco.mj_name2id(
    #         model,
    #         mujoco.mjtObj.mjOBJ_GEOM,
    #         geom_name
    #     )

    #     pos = data.geom_xpos[geom_id].copy()
    #     rot_mat = data.geom_xmat[geom_id].reshape(3, 3)

    #     rpy = Rotation.from_matrix(rot_mat).as_euler(
    #         "xyz",
    #         degrees=False
    #     )

    #     return pos, rpy

    def cube_swept_volume_xml(
        self, object_dims, object_configs, fixed=False, cyl=False
    ):
        """Create swept volume xml string"""
        rgba = [0.8, 0.8, 0.8, 1]
        name = "swept_volume"
        joint_xml = "" if fixed else f'<joint name="{name}_free" type="free"/>'

        if cyl == True:
            R_cyl, b_pos, b1_size, b2_size, corners = (
                self.compute_cyl_sv_params(object_dims, object_configs)
            )
            z_cyl = object_dims[1] / 2
        else:
            R_cyl, b_pos, b1_size, b2_size, corners = self.compute_sv_params(
                object_dims, object_configs
            )
            z_cyl = object_dims[2] / 2

        b_pos0 = b_pos[0]  # numpy (3,)
        b1_size0 = b1_size[0].tolist()
        b2_size0 = b2_size[0].tolist()
        corners0 = corners[0]  # (4,3) world
        corners_local = corners0 - b_pos0  # (4,3) local coords

        sv_xml = f"""
        <body name="{name}" pos="{b_pos0[0]} {b_pos0[1]} {b_pos0[2]}">
            {joint_xml}

            <!-- boxes centered at body origin -->
            <geom name="sv_box1" type="box" pos="0 0 0"
                size="{b1_size0[0]/2} {b1_size0[1]/2} {b1_size0[2]/2}"
                rgba="{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}"/>

            <geom name="sv_box2" type="box" pos="0 0 0"
                size="{b2_size0[0]/2} {b2_size0[1]/2} {b2_size0[2]/2}"
                rgba="{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}"/>

            <!-- cylinders at local corner offsets -->
            <geom name="sv_cyl1" type="cylinder"
                pos="{corners_local[0,0]} {corners_local[0,1]} {corners_local[0,2]}"
                size="{R_cyl} {z_cyl}"
                rgba="{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}"/>

            <geom name="sv_cyl2" type="cylinder"
                pos="{corners_local[1,0]} {corners_local[1,1]} {corners_local[1,2]}"
                size="{R_cyl} {z_cyl}"
                rgba="{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}"/>

            <geom name="sv_cyl3" type="cylinder"
                pos="{corners_local[2,0]} {corners_local[2,1]} {corners_local[2,2]}"
                size="{R_cyl} {z_cyl}"
                rgba="{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}"/>

            <geom name="sv_cyl4" type="cylinder"
                pos="{corners_local[3,0]} {corners_local[3,1]} {corners_local[3,2]}"
                size="{R_cyl} {z_cyl}"
                rgba="{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}"/>
        </body>
        """

        self.collision_geoms.extend(
            ["sv_box1", "sv_box2", "sv_cyl1", "sv_cyl2", "sv_cyl3", "sv_cyl4"]
        )

        return sv_xml

    # def move_swept_volume(self, object_configs):
    #     """Move swept volume to desired bin"""

    #     dummy_config = [[1, 1], [1, 1], [0, 0], [0, 0]]

    #     if isinstance(self, AllStableEnv):
    #         face_in_contact = object_configs[0]
    #         if face_in_contact == "xy":
    #             sv_joint_name = "swept_volume_0_free"
    #         elif face_in_contact == "yz":
    #             sv_joint_name = "swept_volume_1_free"
    #         elif face_in_contact == "zx":
    #             sv_joint_name = "swept_volume_2_free"
    #         else:
    #             raise ValueError(f"Unknown face value: {face_in_contact}")
    #         object_configs = object_configs[1:]
    #     else:
    #         sv_joint_name = "swept_volume_0_free"

    #     # svid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "swept_volume_free")
    #     svid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, sv_joint_name)

    #     if svid == -1:
    #         raise RuntimeError("Could not find joint swept_volume_free")

    #     sv_adr = self.model.jnt_qposadr[svid]
    #     sv_vadr = self.model.jnt_dofadr[svid]

    #     object_configs = np.asarray(object_configs, dtype=np.float64)
    #     if object_configs.ndim == 2:
    #         object_configs = object_configs[None, :, :]

    #     x_lower = object_configs[:, 0, 0]
    #     x_upper = object_configs[:, 0, 1]
    #     y_lower = object_configs[:, 1, 0]
    #     y_upper = object_configs[:, 1, 1]
    #     yaw_lower = object_configs[:, 3, 0]
    #     yaw_upper = object_configs[:, 3, 1]

    #     if isinstance(self, MicrowaveEnv):
    #         # hid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "sv_door_hinge")

    #         # if hid == -1:
    #         #     raise RuntimeError("Could not find joint sv_door_hinge")
            
    #         joint_name = "sv_door_hinge"
    #         door_lower = object_configs[:, 4, 0]
    #         door_upper = object_configs[:, 4, 1]
    #         door_ang = 0.5 * (door_lower + door_upper)

    #         self.move_xml_joint(joint_name, door_ang[0])

    #     z = object_configs[:, 2, 0] 

    #     cx = 0.5 * (x_upper + x_lower)
    #     cy = 0.5 * (y_upper + y_lower)
    #     cyaw = 0.5 * (yaw_lower + yaw_upper)

    #     new_pos = [cx[0], cy[0], z[0]]
        
    #     half = 0.5 * cyaw[0]
    #     new_quat = [
    #         np.cos(half),
    #         0.0,
    #         0.0,
    #         np.sin(half),
    #     ]
    #     # new_quat = [1, 0, 0, 0]

    #     self.data.qpos[sv_adr: sv_adr + 7] = [new_pos[0], new_pos[1], new_pos[2], new_quat[0], new_quat[1], new_quat[2], new_quat[3]]
    #     self.data.qvel[sv_vadr: sv_vadr + 6] = 0

    #     mujoco.mj_forward(self.model, self.data)

    def move_swept_volume(self, object_configs):
        """Move selected swept volume to desired bin, and dummy out unused SVs."""

        dummy_config = [[1, 1], [1, 1], [0, 0], [0, 0]]

        def move_one_sv(sv_joint_name, configs):
            svid = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                sv_joint_name,
            )

            if svid == -1:
                raise RuntimeError(f"Could not find joint {sv_joint_name}")

            sv_adr = self.model.jnt_qposadr[svid]
            sv_vadr = self.model.jnt_dofadr[svid]

            configs = np.asarray(configs, dtype=np.float64)

            if configs.ndim == 2:
                configs = configs[None, :, :]

            x_lower = configs[:, 0, 0]
            x_upper = configs[:, 0, 1]
            y_lower = configs[:, 1, 0]
            y_upper = configs[:, 1, 1]
            z = configs[:, 2, 0]
            yaw_lower = configs[:, 3, 0]
            yaw_upper = configs[:, 3, 1]

            cx = 0.5 * (x_upper + x_lower)
            cy = 0.5 * (y_upper + y_lower)
            cyaw = 0.5 * (yaw_lower + yaw_upper)

            new_pos = [cx[0], cy[0], z[0]]

            half = 0.5 * cyaw[0]
            new_quat = [
                np.cos(half),
                0.0,
                0.0,
                np.sin(half),
            ]

            self.data.qpos[sv_adr: sv_adr + 7] = [
                new_pos[0],
                new_pos[1],
                new_pos[2],
                new_quat[0],
                new_quat[1],
                new_quat[2],
                new_quat[3],
            ]

            self.data.qvel[sv_vadr: sv_vadr + 6] = 0

        if isinstance(self, AllStableEnv):
            face_to_joint = {
                "xy": "swept_volume_0_free",
                "yz": "swept_volume_1_free",
                "zx": "swept_volume_2_free",
            }

            face_in_contact = object_configs[0]

            if face_in_contact not in face_to_joint:
                raise ValueError(f"Unknown face value: {face_in_contact}")

            real_object_configs = object_configs[1:]

            for face, sv_joint_name in face_to_joint.items():
                if face == face_in_contact:
                    move_one_sv(sv_joint_name, real_object_configs)
                else:
                    move_one_sv(sv_joint_name, dummy_config)

        else:
            move_one_sv("swept_volume_0_free", object_configs)

        if isinstance(self, MicrowaveEnv):
            object_configs = np.asarray(object_configs, dtype=np.float64)

            if object_configs.ndim == 2:
                object_configs = object_configs[None, :, :]

            door_lower = object_configs[:, 4, 0]
            door_upper = object_configs[:, 4, 1]
            door_ang = 0.5 * (door_lower + door_upper)

            self.move_xml_joint("sv_door_hinge", door_ang[0])

        mujoco.mj_forward(self.model, self.data)

    def initialize_TSR_parameters(self, robot, grasp_strategy="top", skip_generation=False):
        if robot=="panda":
            TSR_params = panda_TSR_parameters(self.object_details, self.yaw_buffer, self.alpha, grasp_strategy)
        elif robot=="fetch":
            TSR_params = fetch_TSR_parameters(self.object_details, self.yaw_buffer, self.alpha, grasp_strategy)
        elif robot=="ur10":
            TSR_params = ur10_TSR_parameters(self.object_details, self.yaw_buffer, self.alpha, grasp_strategy)

        self.ee_offset, self.Bw, self.half_side, self.Tw2_w1, self.yaw_tw2_w1 = TSR_params
        
        
        self.problem_details_grasp = {
            "Bw": self.Bw,
            "half_side": self.half_side,
            "yaw_buffer": self.yaw_buffer,
            "alpha": self.alpha,
            "reachable_ws": self.object_outer_rad,
            "robot_clearance": self.object_inner_rad,
        }

        if grasp_strategy == "front":
            pass
            p_nom = np.array([self.object_outer_rad, 0, 0])
            from_robot_nom = p_nom - self.robot_pos
            from_robot_nom[2] = 0.0
            from_robot_nom /= np.linalg.norm(from_robot_nom) + 1e-12

            yaw1 = -self.object_yaw
            yaw2 = self.object_yaw
            yaw_edges = np.arange(
                yaw1, yaw2 + self.yaw_buffer, self.yaw_buffer
            )
            yaw_centers = (yaw_edges[:-1] + yaw_edges[1:]) * 0.5

            # best_idx = np.zeros(len(yaw_centers), dtype=np.int64)
            worst_idx = np.zeros(len(yaw_centers), dtype=np.int64)

            z_axis = np.array([0.0, 0.0, 1.0])
            Tews = self.ee_offset  # your 4 variants

            for k, yaw in enumerate(yaw_centers):
                # nominal obj rotation about world Z
                cy, sy = np.cos(yaw), np.sin(yaw)
                Rwo = np.array(
                    [[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]],
                    dtype=float,
                )

                # best_s = -float("inf")
                # best_i = 0
                worst_s = float("inf")
                worst_i = 0

                for i, Tew in enumerate(Tews):
                    Rwe = Rwo @ Tew[:3, :3]
                    approach = Rwe @ z_axis
                    approach[2] = 0.0
                    na = np.linalg.norm(approach)
                    if na > 1e-12:
                        approach /= na

                    s = float(np.dot(approach, from_robot_nom))
                    # if s > best_s:
                    #    best_s = s
                    #    best_i = i
                    if s < worst_s:
                        worst_s = s
                        worst_i = i

                # best_idx[k] = best_i
                worst_idx[k] = worst_i

            self.yaw_edges = yaw_edges
            # self.best_ee_offset_idx = best_idx
            self.worst_ee_offset_idx = worst_idx

        else:

            self.yaw_edges = None
            self.worst_ee_offset_idx = None

        self.problem_details = {
            f"{grasp_strategy}": self.problem_details_grasp
        }
        self.yaw_tw2_w1_dict = {f"{grasp_strategy}": self.yaw_tw2_w1}
        sv_config = [
            [0, round(self.yaw_tw2_w1[0], 5)],
            [0, round(self.yaw_tw2_w1[1], 5)],
            [
                round(self.object_details["position"][2], 5),
                round(self.object_details["position"][2], 5),
            ],
            [0, round(self.Tw2_w1[3], 5)],
        ]
        return sv_config

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
            base_dim = obj["primitives"][0]['dimensions']
            base_pos = obj["primitive_poses"][0]['position']
        
        if base_dim is None or base_pos is None:
            raise RuntimeError(f"Did not find element in scene: {base_name}")

        # Update nominal object position with updated z
        self.object_details['position'] = [
            self.env_details['robot_pos'][0],
            self.env_details['robot_pos'][1],
            base_pos[2] + (base_dim[2]/2) + self.object_details['size'][2]/2
        ]

        self.env_details['z_correction'] = [base_pos[2] + (base_dim[2]/2)]

        hx_int = base_dim[0]/2 - wall_clearance/2
        hy_int = base_dim[1]/2 - wall_clearance/2

        env_name = self.env_details['env_name']

        # Allow overhang for table problems
        # if base_name == "table_top":
        if env_name in {"largeobj"}:
            base_xmin = base_pos[0] - hx_int #+ self.object_details['size'][0]/2
            base_xmax = base_pos[0] + hx_int #- self.object_details['size'][0]/2
            base_ymin = base_pos[1] - hy_int #+ self.object_details['size'][1]/2
            base_ymax = base_pos[1] + hy_int #- self.object_details['size'][1]/2
        else:
            base_xmin = base_pos[0] - hx_int + self.object_details['size'][0]/2
            base_xmax = base_pos[0] + hx_int - self.object_details['size'][0]/2
            base_ymin = base_pos[1] - hy_int + self.object_details['size'][1]/2
            base_ymax = base_pos[1] + hy_int - self.object_details['size'][1]/2

        base_intervals = [
            [base_xmin, base_xmax],
            [base_ymin, base_ymax]
        ]
        # print(f"base intervals: {base_intervals}")
        return base_intervals

    def generate_task_set(self):
        """Generate task set/TSRs"""

        TCR_set = create_TCR_set(self)
        self.task_set = TCR_set
        return self.task_set

        yaw_iTSR_set, _ = find_yaw_iTSR_set(
            self.object_details, 
            self.problem_details, 
            self.Tw2_w1
        )

        iTSR_set, _ = find_iTSR_set(
            self.object_details, 
            self.problem_details, 
            self.yaw_tw2_w1_dict, 
            yaw_iTSR_set, 
            problem=self.problem, 
            robot_pos=self.robot_pos
        )

        self.task_set = iTSR_set[0]
        return self.task_set


class FreeEnv(MujocoEnv):
    """Free environment"""
    def __init__(self, robot, using_swept_volume=True):
        """Initialize free environment"""
        super().__init__(robot)

        # Problem parameters
        object_type = "box"
        object_size = [0.03, 0.03, 0.15]
        yaw_variation = [-0.5*np.pi, 0.5*np.pi]
        yaw_buffer = 6*(np.pi/180)

        # Prepare target object        
        object_variation = {
            'x': [[-0.8, 0.8]],
            'y': [[-0.8, 0.8]],
            'z': [[object_size[2]/2, object_size[2]/2]],
            'yaw': [yaw_variation]
        }
        super().populate_object_details(object_type, object_size, object_variation)

        # Prepare environment details
        if robot == "panda":
            robot_pos = [0, 0, 0]
            robot_quat = [1, 0, 0, 0]
            outer_rad = 0.7
            inner_rad = 0.3
        elif robot == "fetch":
            robot_pos = [0, 0, 0.005]
            robot_quat = [1, 0, 0, 0]
            outer_rad = 0.7
            inner_rad = 0.3
        elif robot == "ur5":
            robot_pos = [0, 0, 0]
            robot_quat = [1, 0, 0, 0]
            outer_rad = 0.8
            inner_rad = 0.3
        elif robot == "G1":
            robot_pos = [0, 0, 0]
            robot_quat = [1, 0, 0, 0]
            outer_rad = 0.8
            inner_rad = 0.3
        else:
            raise ValueError(f"Unsupported robot for FreeEnv: {robot}")
        
        super().populate_env_details(None, robot, "free", robot_pos, robot_quat, outer_rad, inner_rad)

        # Prepare grasp details
        super().populate_grasp_details(yaw_buffer=yaw_buffer, grasp_type="top")
        tcr_intervals = super().construct_tcr()

        # Prepare swept volume (or object geom for validation)
        xmls_to_add = []
        if using_swept_volume:
            xml = super().create_swept_volume(tcr_intervals)

        else:
            if self.object_details['type'] == "box":
                xml = super().cube_object_xml(self.object_details['size'], [1, 1, 0, 0])
            else:
                raise ValueError(f"Currently unsupported object type: {self.object_details['type']}")
        xmls_to_add.append(xml)

        free_xml_path = f"{self.robot_dir}/free_scene.xml"
        self.model, self.data = super().build_model(free_xml_path, xmls_to_add)

        if using_swept_volume:
            self.move_swept_volume([[1, 1], [1, 1], [object_size[2]/2, object_size[2]/2], [0, 0]])
        else:
            self.move_cube_object([1, 1, object_size[2]/2, 0])
        
class BoxEnv(MujocoEnv):
    """Box environment"""
    def __init__(self, robot, using_swept_volume=True):
        """Initialize box environment"""
        super().__init__(robot)

        # Problem parameters
        object_type = "box"
        object_size = [0.03, 0.03, 0.15]
        yaw_variation = [-0.5*np.pi, 0.5*np.pi]
        yaw_buffer = 6*(np.pi/180)

        # Prepare target object        
        object_variation = {
            'x': [[-0.8, 0.8]],
            'y': [[-0.8, 0.8]],
            'z': [[object_size[2]/2, object_size[2]/2]],
            'yaw': [yaw_variation]
        }
        super().populate_object_details(object_type, object_size, object_variation)

        # Prepare environment details
        if robot=="panda":
            config_yaml = "configs/problems/box_panda.yaml"
        elif robot=="fetch":
            config_yaml = "configs/problems/box_fetch.yaml"
        elif robot=="ur10":
            config_yaml = "configs/problems/box_ur5.yaml"
        elif robot=="g1":
            config_yaml = "configs/problems/box_g1.yaml"

        scene_yaml = "configs/scenes/box/scene_box.yaml"
        with open(config_yaml, "r") as f:
            config_yaml_data = yaml.safe_load(f)
        
        robot_pos = config_yaml_data['base_offset']['position']
        robot_quat = super().quat_xyzw_to_wxyz(config_yaml_data['base_offset']['orientation'])
        outer_rad = 0.75
        # outer_rad = 0.4
        inner_rad = 0.3

        super().populate_env_details(scene_yaml, robot, "box", robot_pos, robot_quat, outer_rad, inner_rad)

        # Prepare grasp details
        super().populate_grasp_details(yaw_buffer=yaw_buffer, grasp_type="top")
        tcr_intervals = super().construct_tcr()

        # # Find valid x,y intervals for object placements in problem
        # box_thickness = 0.18
        # box_intervals = super().find_problem_intervals(base_name="base", wall_clearance=box_thickness)
        # self.problem = {
        #     'name': "box",
        #     'intervals': box_intervals,
        #     'robot': f"{robot}"   
        # }

        # # Annulus of object positions
        # self.object_inner_rad = 0.3
        # self.object_outer_rad = 0.75
        # self.object_yaw = 0.25*np.pi #-yaw to +yaw
        # self.object_details['dist'] = [
        #     self.object_outer_rad, self.object_outer_rad, 0, self.object_yaw
        # ]
        # # Find TSR parameters
        # sv_config = super().initialize_TSR_parameters(robot, grasp_strategy="top")
        
        # # Add environment xmls and build model
        # #sv_xml = super().cube_swept_volume_xml(self.object_details['size'], sv_config)
        # if no_sv==False:
        #     sv_xml = super().cube_swept_volume_xml(self.object_details['size'], sv_config)
        # else:
        #     sv_xml = super().cube_object_xml(self.object_details['size'], [1, 1, 0, 0])
        # box_xml = super().build_xml(parent_body_name="scene_box", skip_ids={"Can1"})
        
        # xmls_to_add = [sv_xml, box_xml]
        # free_xml_path = f"{self.robot_dir}/box_scene.xml"
        # self.model, self.data = super().build_model(free_xml_path, xmls_to_add)

        # Prepare swept volume (or object geom for validation)
        xmls_to_add = []
        if using_swept_volume:
            xml = super().create_swept_volume(tcr_intervals)
        else:
            if self.object_details['type'] == "box":
                xml = super().cube_object_xml(self.object_details['size'], [1, 1, 0, 0])
            else:
                raise ValueError(f"Currently unsupported object type: {self.object_details['type']}")
        xmls_to_add.append(xml)

        # Prepare environment xmls
        box_xml = super().build_xml(scene_yaml, parent_body_name="scene_box", skip_ids={"Can1"})
        xmls_to_add.append(box_xml)

        free_xml_path = f"{self.robot_dir}/box_scene.xml"
        self.model, self.data = super().build_model(free_xml_path, xmls_to_add)


class CageEnv(MujocoEnv):
    """Cage environment"""
    def __init__(self, robot, using_swept_volume=True):
        """Initialize cage environment"""
        super().__init__(robot)

        # Problem parameters
        object_type = "box"
        object_size = [0.03, 0.03, 0.15]
        yaw_variation = [-0.5*np.pi, 0.5*np.pi]
        yaw_buffer = 6*(np.pi/180)

        # Prepare target object        
        object_variation = {
            'x': [[-0.8, 0.8]],
            'y': [[-0.8, 0.8]],
            'z': [[object_size[2]/2, object_size[2]/2]],
            'yaw': [yaw_variation]
        }
        super().populate_object_details(object_type, object_size, object_variation)

        # Prepare environment details
        if robot=="panda":
            config_yaml = "configs/problems/cage_panda.yaml"
        elif robot=="fetch":
            config_yaml = "configs/problems/cage_fetch.yaml"
        elif robot=="ur10":
            config_yaml = "configs/problems/cage_ur5.yaml"
        
        scene_yaml = "configs/scenes/cage/scene_cage.yaml"
        with open(config_yaml, "r") as f:
            config_yaml_data = yaml.safe_load(f)
        
        robot_pos = config_yaml_data['base_offset']['position']
        robot_quat = super().quat_xyzw_to_wxyz(config_yaml_data['base_offset']['orientation'])
        outer_rad = 0.75
        inner_rad = 0.3

        super().populate_env_details(scene_yaml, robot, "cage", robot_pos, robot_quat, outer_rad, inner_rad)

        # Prepare grasp details
        super().populate_grasp_details(yaw_buffer=yaw_buffer, grasp_type="top")
        tcr_intervals = super().construct_tcr()

        xmls_to_add = []
        if using_swept_volume:
            xml = super().create_swept_volume(tcr_intervals)
        else:
            if self.object_details['type'] == "box":
                xml = super().cube_object_xml(self.object_details['size'], [1, 1, 0, 0])
            else:
                raise ValueError(f"Currently unsupported object type: {self.object_details['type']}")
        xmls_to_add.append(xml)

        # Prepare environment xmls
        cage_xml = super().build_xml(scene_yaml, parent_body_name="scene_cage", skip_ids={"Cube1"})
        xmls_to_add.append(cage_xml)

        free_xml_path = f"{self.robot_dir}/cage_scene.xml"
        self.model, self.data = super().build_model(free_xml_path, xmls_to_add)

class TableEnv(MujocoEnv):
    """Table environment"""
    def __init__(self, robot, using_swept_volume=True):
        """Initialize table environment"""
        super().__init__(robot)

        # Problem parameters
        object_type = "box"
        object_size = [0.03, 0.03, 0.15]
        yaw_variation = [-0.5*np.pi, 0.5*np.pi]
        yaw_buffer = 6*(np.pi/180)

        # Prepare target object        
        object_variation = {
            'x': [[-0.8, 0.8]],
            'y': [[-0.8, 0.8]],
            'z': [[object_size[2]/2, object_size[2]/2]],
            'yaw': [yaw_variation]
        }
        super().populate_object_details(object_type, object_size, object_variation)

        # Prepare environment details
        if robot=="panda":
            config_yaml = "configs/problems/table_pick_panda.yaml"
        elif robot=="fetch":
            config_yaml = "configs/problems/table_pick_fetch.yaml"
        elif robot=="ur10":
            config_yaml = "configs/problems/table_pick_ur5.yaml"
        
        scene_yaml = "configs/scenes/table/scene_table.yaml"
        with open(config_yaml, "r") as f:
            config_yaml_data = yaml.safe_load(f)
        
        robot_pos = config_yaml_data['base_offset']['position']
        robot_quat = super().quat_xyzw_to_wxyz(config_yaml_data['base_offset']['orientation'])
        
        if robot == "panda" or robot == "fetch":
            inner_rad = 0.3
            outer_rad = 0.7
        elif robot == "ur10":
            inner_rad = 0.3
            outer_rad = 0.75
        
        super().populate_env_details(scene_yaml, robot, "table", robot_pos, robot_quat, outer_rad, inner_rad)

        # Prepare grasp details
        super().populate_grasp_details(yaw_buffer=yaw_buffer, grasp_type="top")
        tcr_intervals = super().construct_tcr()

        xmls_to_add = []
        if using_swept_volume:
            xml = super().create_swept_volume(tcr_intervals)
        else:
            if self.object_details['type'] == "box":
                xml = super().cube_object_xml(self.object_details['size'], [1, 1, 0, 0])
            else:
                raise ValueError(f"Currently unsupported object type: {self.object_details['type']}")
        xmls_to_add.append(xml)

        # Prepare environment xmls
        table_xml = super().build_xml(scene_yaml, parent_body_name="scene_table", skip_ids={"Cube1"})
        xmls_to_add.append(table_xml)

        free_xml_path = f"{self.robot_dir}/table_scene.xml"
        self.model, self.data = super().build_model(free_xml_path, xmls_to_add)


class ShelfEnv(MujocoEnv):
    """Thin shelf environment"""

    def __init__(self, robot, no_sv=False):
        """Initialize thin shelf environment"""
        super().__init__(robot)

        # Problem parameters
        object_type = "box"
        object_size = [0.03, 0.03, 0.15]
        yaw_variation = [-0.5*np.pi, 0.5*np.pi]
        yaw_buffer = 6*(np.pi/180)

        # Prepare target object        
        object_variation = {
            'x': [[-0.8, 0.8]],
            'y': [[-0.8, 0.8]],
            'z': [[object_size[2]/2, object_size[2]/2]],
            'yaw': [yaw_variation]
        }
        super().populate_object_details(object_type, object_size, object_variation)

        # Prepare environment details
        if robot=="panda":
            config_yaml = "configs/problems/bookshelf_thin_panda.yaml"
        elif robot=="fetch":
            config_yaml = "configs/problems/bookshelf_thin_fetch.yaml"
        elif robot=="ur10":
            config_yaml = "configs/problems/bookshelf_thin_ur5.yaml"
        
        scene_yaml = "configs/scenes/bookshelf/scene_thin.yaml"
        with open(config_yaml, "r") as f:
            config_yaml_data = yaml.safe_load(f)
        
        robot_pos = config_yaml_data['base_offset']['position']
        robot_quat = super().quat_xyzw_to_wxyz(config_yaml_data['base_offset']['orientation'])

        if robot == "panda":
            inner_rad = 0.3
            outer_rad = 0.75
        elif robot == "fetch":
            inner_rad = 0.3
            outer_rad = 0.75
        elif robot == "ur10":
            inner_rad = 0.3
            #self.object_outer_rad = 1.1
            outer_rad = 0.65

        super().populate_env_details(scene_yaml, robot, "shelf", robot_pos, robot_quat, outer_rad, inner_rad)

        # Prepare grasp details
        super().populate_grasp_details(yaw_buffer=yaw_buffer, grasp_type="front")
        tcr_intervals = super().construct_tcr()

        # shelf_thickness = 0.18
        shelf_thickness = 0.12
        dividing_wall_thickness = 0.14
        # bases = ['shelf_bottom', 'shelf_middle_bottom', 'shelf_middle', 'shelf_middle_top', 'shelf_top']
        bases = ["shelf_middle"]
        shelf_intervals = self.find_problem_intervals(
            bases, shelf_thickness, dividing_wall_thickness
        )
        self.problem = {
            "name": "shelf",
            "intervals": shelf_intervals,
            "robot": f"{robot}",
        }
        # Annulus of object positions
        if robot == "panda":
            self.object_inner_rad = 0.3
            self.object_outer_rad = 0.75
        elif robot == "fetch":
            self.object_inner_rad = 0.3
            self.object_outer_rad = 0.75
        elif robot == "ur10":
            self.object_inner_rad = 0.3
            # self.object_outer_rad = 1.1
            self.object_outer_rad = 0.65

            # self.robot_pos[0] = self.robot_pos[0]-0.8

        self.object_yaw = 0.01 * np.pi  # -yaw to +yaw
        self.object_details["dist"] = [
            self.object_outer_rad,
            self.object_outer_rad,
            0,
            self.object_yaw,
        ]
        # Find TSR parameters
        sv_config = super().initialize_TSR_parameters(
            robot, grasp_strategy="front"
        )

        # Add environment xmls and build model
        # sv_xml = super().cube_swept_volume_xml(self.object_details['size'], sv_config)
        if no_sv == False:
            sv_xml = super().cube_swept_volume_xml(
                self.object_details["size"], sv_config
            )
        else:
            sv_xml = super().cube_object_xml(
                self.object_details["size"], [1, 1, 0, 0]
            )
        shelf_xml = super().build_xml(
            parent_body_name="scene_shelf", skip_ids={"Cube1"}
        )

        xmls_to_add = [sv_xml, shelf_xml]
        free_xml_path = f"{self.robot_dir}/shelf_scene.xml"
        self.model, self.data = super().build_model(free_xml_path, xmls_to_add)

    def find_problem_intervals(self, scene_yaml, bases, wall_clearance, dividing_wall_clearance):
        with open(scene_yaml, "r") as f:
            scene_yaml_data = yaml.safe_load(f)
        objs = scene_yaml_data["world"]["collision_objects"]

        base_dim = None
        base_pos = None

        for obj in objs:
            obj_id = obj.get("id", "")
            if obj_id != bases[0]:
                continue
            base_dim = obj["primitives"][0]["dimensions"]
            base_pos = obj["primitive_poses"][0]["position"]

        # bases = ['shelf_bottom', 'shelf_middle_bottom', 'shelf_middle', 'shelf_middle_top', 'shelf_top']
        # bases = ['shelf_middle']
        base_zpos = []
        base_zdim = []
        base_names = []
        for obj in objs:
            obj_id = obj.get("id", "")
            if obj_id not in bases:
                continue
            base_zpos.append(obj["primitive_poses"][0]["position"][2])
            base_zdim.append(obj["primitives"][0]["dimensions"][2])
            base_names.append(obj_id)
        # self.base_zpos = base_zpos
        # self.base_zdim = base_zdim
        # self.base_names = base_names

        # self.base_dim = base_dim
        # self.base_pos = base_pos

        # # Update nominal object position with updated z
        # self.object_details['position'] = [
        #     self.robot_pos[0], self.robot_pos[1], base_pos[2] + (base_dim[2]/2) + self.object_details['size'][2]/2
        # ]

        z_correction = []
        for base_idx in range(len(base_zpos)):
            curr_zpos = base_zpos[base_idx]
            curr_zdim = base_zdim[base_idx]
            z_correction.append(curr_zpos + (curr_zdim/2))

        self.env_details['z_correction'] = z_correction

        hx_int = base_dim[0] / 2 - wall_clearance / 2
        hy_int = base_dim[1] / 2 - wall_clearance / 2

        shelf_xmin = base_pos[0] - hx_int + self.object_details["size"][0] / 2
        shelf_xmax = base_pos[0] + hx_int - self.object_details["size"][0] / 2
        shelf_ymin = base_pos[1] - hy_int + self.object_details["size"][1] / 2
        shelf_ymax = base_pos[1] + hy_int - self.object_details["size"][1] / 2

        for obj in objs:
            obj_id = obj.get("id", "")
            if obj_id != "shelf_vert":
                continue
            wall_dim = obj["primitives"][0]["dimensions"]
            wall_pos = obj["primitive_poses"][0]["position"]
        wx_int = wall_dim[0] / 2 + dividing_wall_clearance / 2
        wy_int = wall_dim[1] / 2 + dividing_wall_clearance / 2

        wall_xmin = wall_pos[0] - wx_int - self.object_details["size"][0] / 2
        wall_xmax = wall_pos[0] + wx_int + self.object_details["size"][0] / 2
        wall_ymin = wall_pos[1] - wy_int - self.object_details["size"][1] / 2
        wall_ymax = wall_pos[1] + wy_int + self.object_details["size"][1] / 2

        fxmin = max(shelf_xmin, wall_xmin)
        fxmax = min(shelf_xmax, wall_xmax)
        fymin = max(shelf_ymin, wall_ymin)
        fymax = min(shelf_ymax, wall_ymax)

        regions = []

        # If wall doesn't overlap the cage at all, nothing to subtract
        if fxmin >= fxmax or fymin >= fymax:
            regions.append(
                [[shelf_xmin, shelf_xmax], [shelf_ymin, shelf_ymax]]
            )
        else:
            # Subtract forbidden rectangle from cage rectangle.
            # This can produce up to 4 rectangles; if your wall "cuts the shelf in half" you'll typically get 2.

            # Left slab
            if shelf_xmin < fxmin:
                regions.append([[shelf_xmin, fxmin], [shelf_ymin, shelf_ymax]])

            # Right slab
            if fxmax < shelf_xmax:
                regions.append([[fxmax, shelf_xmax], [shelf_ymin, shelf_ymax]])

            # Bottom slab
            if shelf_ymin < fymin:
                regions.append([[fxmin, fxmax], [shelf_ymin, fymin]])

            # Top slab
            if fymax < shelf_ymax:
                regions.append([[fxmin, fxmax], [fymax, shelf_ymax]])

        # Optional: keep only non-degenerate regions (numerical safety)
        eps = 1e-9
        regions = [
            r
            for r in regions
            if (r[0][1] - r[0][0] > eps) and (r[1][1] - r[1][0] > eps)
        ]
        # print(regions)
        return regions

    def generate_task_set(self):
        """Generate task set/TSRs"""

        iTSR_dict = {}
        yaw_iTSR_set, _ = find_yaw_iTSR_set(
            self.object_details,
            self.problem_details,
            self.Tw2_w1,
        )

        for curr_base_ind in range(len(self.base_zpos)):
            curr_base_zdim = self.base_zdim[curr_base_ind]
            curr_base_zpos = self.base_zpos[curr_base_ind]
            # print(curr_base_zdim)
            # print(curr_base_zpos)
            z_object = (
                curr_base_zpos
                + (curr_base_zdim / 2)
                + (self.object_details["size"][2] / 2)
            )
            object_position = [self.robot_pos[0], self.robot_pos[1], z_object]
            # print(object_position)
            object_type = self.object_details["type"]
            object_size = self.object_details["size"]
            object_dist = self.object_details["dist"]
            self.object_details = {
                "type": object_type,
                "size": object_size,
                "position": object_position,
                "yaw": 0,
                "dist": object_dist,
            }
            print(f"Base: {self.base_names[curr_base_ind]}")
            curr_base_iTSR_set, _ = find_iTSR_set(
                self.object_details,
                self.problem_details,
                self.yaw_tw2_w1_dict,
                yaw_iTSR_set,
                problem=self.problem,
                robot_pos=self.robot_pos,
            )
            # print(len(curr_base_iTSR_set[0]))
            iTSR_dict.update(curr_base_iTSR_set[0])

        # iTSR_set, _ = find_iTSR_set(self.object_details, self.problem_details, self.yaw_tw2_w1_dict, yaw_iTSR_set, problem=self.problem, robot_pos=self.robot_pos)
        # iTSR_set = [iTSR_dict]
        # self.task_set = iTSR_set[0]
        self.task_set = iTSR_dict
        return self.task_set



class RealEnv(MujocoEnv):
    """Real environment"""

    def __init__(self, robot, using_swept_volume=True):
        """Initialize the real lab environment."""

        if robot != "ur10":
            raise NotImplementedError("RealEnv only supports UR10")

        super().__init__(robot, custom_base="lab_scene.xml")

        # --------------------------------------------------------------
        # Object details
        # --------------------------------------------------------------

        object_type = "cylinder"
        object_size = [0.045, 0.08]  # [radius, height]

        real_intervals = [
            [-0.35, 0.07],
            [-1.02, -0.70],
        ]

        # object_variation = {
        #     "x": [real_intervals[0]],
        #     "y": [real_intervals[1]],
        #     "z": [[object_size[1] / 2.0, object_size[1] / 2.0]],
        # }

        object_variation = {
            "x": [real_intervals[0]],
            "y": [real_intervals[1]],
            "z": [
                [
                    object_size[1] / 2.0,
                    object_size[1] / 2.0,
                ]
            ],
            "yaw": [[0.0, 0.0]],
        }

        super().populate_object_details(
            object_type,
            object_size,
            object_variation,
        )

        # --------------------------------------------------------------
        # Environment details
        # --------------------------------------------------------------

        robot_pos = [0, 0, 0]
        robot_quat = [0, 0, 0, 1]  # WXYZ

        outer_rad = 1.0
        inner_rad = 0.3

        # RealEnv currently generates its scene programmatically, so it
        # does not need a scene YAML.
        super().populate_env_details(
            scene_yaml=None,
            robot_name=robot,
            env_name="real",
            robot_pos=robot_pos,
            robot_quat=robot_quat,
            outer_rad=outer_rad,
            inner_rad=inner_rad,
        )

        # RealEnv has manually specified placement intervals.
        self.env_details["intervals"] = real_intervals

        # --------------------------------------------------------------
        # Grasp details
        # --------------------------------------------------------------

        super().populate_grasp_details(
            alpha=0.5,
            grasp_type="top",
        )

        # --------------------------------------------------------------
        # Temporary compatibility with the old UR10 TSR implementation
        # --------------------------------------------------------------

        self.robot_pos = self.env_details["robot_pos"]
        self.robot_quat = self.env_details["robot_quat"]

        self.object_inner_rad = self.env_details["inner_rad"]
        self.object_outer_rad = self.env_details["outer_rad"]
        self.object_yaw = 0.0

        self.yaw_buffer = self.grasp_details["yaw_buffer"]
        self.alpha = self.grasp_details["alpha"]

        self.object_details["dist"] = [
            self.object_outer_rad,
            self.object_outer_rad,
            0.0,
            self.object_yaw,
        ]

        self.object_details["position"] = [
            self.robot_pos[0],
            self.robot_pos[1],
            object_size[1] / 2.0,
        ]

        # Keep this legacy dictionary until its consumers are migrated
        # to object_details and env_details.
        self.problem = {
            "name": "real",
            "intervals": real_intervals,
            "robot": robot,
        }

        # sv_config = super().initialize_TSR_parameters(
        #     robot,
        #     grasp_strategy="top",
        # )

        # if using_swept_volume:
        #     object_xml = super().cube_swept_volume_xml(
        #         self.object_details["size"],
        #         sv_config,
        #         cyl=True,
        #     )
        # else:
        #     object_xml = super().cylinder_object_xml(
        #         self.object_details["size"],
        #         [1, 1, 0, 0],
        #     )

        if using_swept_volume:
            tcr_intervals = super().construct_tcr()

            object_xml = super().create_swept_volume(
                tcr_intervals,
            )
        else:
            object_xml = super().cylinder_object_xml(
                self.object_details["size"],
                [0, 0, 0, 0],
            )

        xmls_to_add = [object_xml]

        wall1_dims = (0.24, 0.45, 0.29)
        wall_inflation = 0.07  # inflating by object's size (for return path)
        wall1_dims = np.array(wall1_dims) * 1.05
        wall1_dims = wall1_dims + wall_inflation / 2

        # build wall 1
        wall_1_xml = self.build_primitive_body_xml(
            body_name="wall1",
            prim_type="box",
            pos=(0.27, -0.82, 0.145),
            dims=wall1_dims.tolist(),
            quat_xyzw=(0, 0, 0, 1),
            make_free=False,
        )
        xmls_to_add.append(wall_1_xml)

        # build wall 2
        wall_2_xml = self.build_primitive_body_xml(
            body_name="wall2",
            prim_type="box",
            pos=(-0.07, -0.45, 0.135),
            dims=(0.48, 0.34, 0.27),
            quat_xyzw=(0, 0, 0, 1),
            make_free=False,
        )
        xmls_to_add.append(wall_2_xml)

        # build block 1
        block_1_xml = self.build_primitive_body_xml(
            body_name="block1",
            prim_type="box",
            pos=(0.205, -0.51, 0.1),
            dims=(0.09, 0.09, 0.2),
            quat_xyzw=(0, 0, 0, 1),
            make_free=False,
        )
        xmls_to_add.append(block_1_xml)

        # build packing
        packing_1_xml = self.build_primitive_body_xml(
            body_name="packing1",
            prim_type="box",
            pos=(0.54, -0.80, 0.01),
            dims=(0.21, 0.36, 0.02),
            quat_xyzw=(0, 0, 0, 1),
            make_free=False,
        )
        xmls_to_add.append(packing_1_xml)

        # --- parameters ---
        cx, cy, z_floor = 0.54, -0.80, 0.01
        Lx, Ly, t_floor = 0.21, 0.36, 0.02

        t_wall = 0.02
        h_wall = 0.13  # <-- change this to how tall you want the hollow box
        h_wall = 0.10 + wall_inflation

        z_top = z_floor + t_floor / 2.0
        z_wall = z_top + h_wall / 2.0

        x_off = (Lx / 2.0) - (t_wall / 2.0)
        y_off = (Ly / 2.0) - (t_wall / 2.0)

        # Left wall (thin in x, spans y, tall in z)
        packing_2_xml = self.build_primitive_body_xml(
            body_name="packing2_left",
            prim_type="box",
            pos=(cx - x_off, cy, z_wall),
            dims=(t_wall, Ly, h_wall),
            quat_xyzw=(0, 0, 0, 1),
            make_free=False,
        )

        # Right wall
        packing_3_xml = self.build_primitive_body_xml(
            body_name="packing3_right",
            prim_type="box",
            pos=(cx + x_off, cy, z_wall),
            dims=(t_wall, Ly, h_wall),
            quat_xyzw=(0, 0, 0, 1),
            make_free=False,
        )

        # Bottom wall (thin in y, spans x, tall in z)
        packing_4_xml = self.build_primitive_body_xml(
            body_name="packing4_bottom",
            prim_type="box",
            pos=(cx, cy - y_off, z_wall),
            dims=(Lx, t_wall, h_wall),
            quat_xyzw=(0, 0, 0, 1),
            make_free=False,
        )

        # Top wall
        packing_5_xml = self.build_primitive_body_xml(
            body_name="packing5_top",
            prim_type="box",
            pos=(cx, cy + y_off, z_wall),
            dims=(Lx, t_wall, h_wall),
            quat_xyzw=(0, 0, 0, 1),
            make_free=False,
        )

        xmls_to_add.append(packing_2_xml)
        xmls_to_add.append(packing_3_xml)
        xmls_to_add.append(packing_4_xml)
        xmls_to_add.append(packing_5_xml)

        # upper boundary
        upper_boundary_xml = self.build_primitive_body_xml(
            body_name="ub1",
            prim_type="box",
            pos=(0, -0.50, 1.02),
            dims=(1.5, 1.5, 0.02),
            quat_xyzw=(0, 0, 0, 1),
            make_free=False,
        )
        xmls_to_add.append(upper_boundary_xml)

        # upper boundary
        back_boundary_xml = self.build_primitive_body_xml(
            body_name="bb1",
            prim_type="box",
            pos=(0, 0.40, 0.8),
            dims=(2, 0.02, 1.60),
            quat_xyzw=(0, 0, 0, 1),
            make_free=False,
        )
        xmls_to_add.append(back_boundary_xml)

        left_boundary_xml = self.build_primitive_body_xml(
            body_name="lb1",
            prim_type="box",
            pos=(-0.82, -0.50, 0.8),
            dims=(0.02, 2, 1.60),
            quat_xyzw=(0, 0, 0, 1),
            make_free=False,
        )
        xmls_to_add.append(left_boundary_xml)

        right_boundary_xml = self.build_primitive_body_xml(
            body_name="rb1",
            prim_type="box",
            pos=(0.82, -0.50, 0.8),
            dims=(0.02, 2, 1.60),
            quat_xyzw=(0, 0, 0, 1),
            make_free=False,
        )
        xmls_to_add.append(right_boundary_xml)

        free_xml_path = f"{self.robot_dir}/real_scene.xml"
        self.xmls_to_add = xmls_to_add
        self.model, self.data = super().build_model(free_xml_path, xmls_to_add)


    # def __init__(self, robot, no_sv=False):
    #     """Initialize real environment"""
    #     super().__init__(robot, custom_base="lab_scene.xml")
    #     if robot != "ur10":
    #         raise NotImplementedError("RealEnv only supports UR10")

    #     # Initialize object dimensions
    #     self.object_details = {
    #         "size": [0.045, 0.08],  # [r, h]
    #         "type": "cylinder",
    #         "yaw": 0,
    #     }
    #     # print(self.object_details['size'])
    #     # object_path = self.select_object("mug")
    #     # object_path = self.select_object("g_cups")

    #     self.robot_pos = [0, 0, 0]
    #     self.robot_quat = [1, 0, 0, 0]
    #     self.robot_quat = [0.70710678, 0.0, 0.0, 0.70710678]
    #     self.robot_quat = [0.0, 0.0, 0.0, 1.0]
    #     # self.robot_quat = [-0.70710678, 0.0, 0.0, 0.70710678]

    #     base_pos = [0, 0, 0]
    #     base_dim = [0, 0, 0]

    #     self.object_yaw = 0 * np.pi  # -yaw to +yaw
    #     self.object_inner_rad = 0.3
    #     self.object_outer_rad = 1.0

    #     self.object_details["dist"] = [
    #         self.object_outer_rad,
    #         self.object_outer_rad,
    #         0,
    #         self.object_yaw,
    #     ]

    #     self.object_details["position"] = [
    #         self.robot_pos[0],
    #         self.robot_pos[1],
    #         base_pos[2]
    #         + (base_dim[2] / 2)
    #         + self.object_details["size"][1] / 2,
    #     ]

    #     # box_intervals = super().find_problem_intervals(base_name="base", wall_clearance=box_thickness)

    #     # table intervals
    #     real_intervals = [[-0.35, 0.07], [-1.02, -0.70]]
    #     # real_intervals = [
    #     #     [-1, 1],
    #     #     [-1, 1]
    #     # ]
    #     self.problem = {
    #         "name": "box",
    #         "intervals": real_intervals,
    #         "robot": f"{robot}",
    #     }

    #     # Find TSR parameters
    #     sv_config = super().initialize_TSR_parameters(
    #         robot, grasp_strategy="top"
    #     )

    #     # print(self.object_details['size'])
    #     if no_sv is True:
    #         # pass
    #         # print(self.object_details['size'])
    #         # print(sv_config)
    #         sv_xml = super().cylinder_object_xml(
    #             self.object_details["size"], [1, 1, 0, 0]
    #         )
    #     else:
    #         sv_xml = super().cube_swept_volume_xml(
    #             self.object_details["size"], sv_config, cyl=True
    #         )
    #     # sv_xml = self.mjcf_file_to_fragment(object_path)
    #     xmls_to_add = [sv_xml]

    #     # Extra objects
    #     # apple_xml = self.mjcf_file_to_fragment("assets/ycb/apple.xml")
    #     # sugar_box_xml = self.mjcf_file_to_fragment("assets/ycb/sugar_box.xml")
    #     # a_cups_xml = self.mjcf_file_to_fragment("assets/ycb/a_cups.xml")

    #     # xmls_to_add.append(apple_xml)
    #     # xmls_to_add.append(sugar_box_xml)
    #     # xmls_to_add.append(a_cups_xml)

    #     # b_cups_xml = self.mjcf_file_to_fragment("assets/ycb/b_cups.xml")
    #     # xmls_to_add.append(b_cups_xml)

    #     # c_cups_xml = self.mjcf_file_to_fragment("assets/ycb/c_cups.xml")
    #     # xmls_to_add.append(c_cups_xml)

    #     # d_cups_xml = self.mjcf_file_to_fragment("assets/ycb/d_cups.xml")
    #     # xmls_to_add.append(d_cups_xml)

    #     # g_cups_xml = self.mjcf_file_to_fragment("assets/ycb/g_cups.xml")
    #     # xmls_to_add.append(g_cups_xml)

    #     wall1_dims = (0.24, 0.45, 0.29)
    #     wall_inflation = 0.07  # inflating by object's size (for return path)
    #     wall1_dims = np.array(wall1_dims) * 1.05
    #     wall1_dims = wall1_dims + wall_inflation / 2

    #     # build wall 1
    #     wall_1_xml = self.build_primitive_body_xml(
    #         body_name="wall1",
    #         prim_type="box",
    #         pos=(0.27, -0.82, 0.145),
    #         dims=wall1_dims.tolist(),
    #         quat_xyzw=(0, 0, 0, 1),
    #         make_free=False,
    #     )
    #     xmls_to_add.append(wall_1_xml)

    #     # build wall 2
    #     wall_2_xml = self.build_primitive_body_xml(
    #         body_name="wall2",
    #         prim_type="box",
    #         pos=(-0.07, -0.45, 0.135),
    #         dims=(0.48, 0.34, 0.27),
    #         quat_xyzw=(0, 0, 0, 1),
    #         make_free=False,
    #     )
    #     xmls_to_add.append(wall_2_xml)

    #     # build block 1
    #     block_1_xml = self.build_primitive_body_xml(
    #         body_name="block1",
    #         prim_type="box",
    #         pos=(0.205, -0.51, 0.1),
    #         dims=(0.09, 0.09, 0.2),
    #         quat_xyzw=(0, 0, 0, 1),
    #         make_free=False,
    #     )
    #     xmls_to_add.append(block_1_xml)

    #     # build packing
    #     packing_1_xml = self.build_primitive_body_xml(
    #         body_name="packing1",
    #         prim_type="box",
    #         pos=(0.54, -0.80, 0.01),
    #         dims=(0.21, 0.36, 0.02),
    #         quat_xyzw=(0, 0, 0, 1),
    #         make_free=False,
    #     )
    #     xmls_to_add.append(packing_1_xml)

    #     # --- parameters ---
    #     cx, cy, z_floor = 0.54, -0.80, 0.01
    #     Lx, Ly, t_floor = 0.21, 0.36, 0.02

    #     t_wall = 0.02
    #     h_wall = 0.13  # <-- change this to how tall you want the hollow box
    #     h_wall = 0.10 + wall_inflation

    #     z_top = z_floor + t_floor / 2.0
    #     z_wall = z_top + h_wall / 2.0

    #     x_off = (Lx / 2.0) - (t_wall / 2.0)
    #     y_off = (Ly / 2.0) - (t_wall / 2.0)

    #     # Left wall (thin in x, spans y, tall in z)
    #     packing_2_xml = self.build_primitive_body_xml(
    #         body_name="packing2_left",
    #         prim_type="box",
    #         pos=(cx - x_off, cy, z_wall),
    #         dims=(t_wall, Ly, h_wall),
    #         quat_xyzw=(0, 0, 0, 1),
    #         make_free=False,
    #     )

    #     # Right wall
    #     packing_3_xml = self.build_primitive_body_xml(
    #         body_name="packing3_right",
    #         prim_type="box",
    #         pos=(cx + x_off, cy, z_wall),
    #         dims=(t_wall, Ly, h_wall),
    #         quat_xyzw=(0, 0, 0, 1),
    #         make_free=False,
    #     )

    #     # Bottom wall (thin in y, spans x, tall in z)
    #     packing_4_xml = self.build_primitive_body_xml(
    #         body_name="packing4_bottom",
    #         prim_type="box",
    #         pos=(cx, cy - y_off, z_wall),
    #         dims=(Lx, t_wall, h_wall),
    #         quat_xyzw=(0, 0, 0, 1),
    #         make_free=False,
    #     )

    #     # Top wall
    #     packing_5_xml = self.build_primitive_body_xml(
    #         body_name="packing5_top",
    #         prim_type="box",
    #         pos=(cx, cy + y_off, z_wall),
    #         dims=(Lx, t_wall, h_wall),
    #         quat_xyzw=(0, 0, 0, 1),
    #         make_free=False,
    #     )

    #     xmls_to_add.append(packing_2_xml)
    #     xmls_to_add.append(packing_3_xml)
    #     xmls_to_add.append(packing_4_xml)
    #     xmls_to_add.append(packing_5_xml)

    #     # upper boundary
    #     upper_boundary_xml = self.build_primitive_body_xml(
    #         body_name="ub1",
    #         prim_type="box",
    #         pos=(0, -0.50, 1.02),
    #         dims=(1.5, 1.5, 0.02),
    #         quat_xyzw=(0, 0, 0, 1),
    #         make_free=False,
    #     )
    #     xmls_to_add.append(upper_boundary_xml)

    #     # upper boundary
    #     back_boundary_xml = self.build_primitive_body_xml(
    #         body_name="bb1",
    #         prim_type="box",
    #         pos=(0, 0.40, 0.8),
    #         dims=(2, 0.02, 1.60),
    #         quat_xyzw=(0, 0, 0, 1),
    #         make_free=False,
    #     )
    #     xmls_to_add.append(back_boundary_xml)

    #     left_boundary_xml = self.build_primitive_body_xml(
    #         body_name="lb1",
    #         prim_type="box",
    #         pos=(-0.82, -0.50, 0.8),
    #         dims=(0.02, 2, 1.60),
    #         quat_xyzw=(0, 0, 0, 1),
    #         make_free=False,
    #     )
    #     xmls_to_add.append(left_boundary_xml)

    #     right_boundary_xml = self.build_primitive_body_xml(
    #         body_name="rb1",
    #         prim_type="box",
    #         pos=(0.82, -0.50, 0.8),
    #         dims=(0.02, 2, 1.60),
    #         quat_xyzw=(0, 0, 0, 1),
    #         make_free=False,
    #     )
    #     xmls_to_add.append(right_boundary_xml)

    #     free_xml_path = f"{self.robot_dir}/real_scene.xml"
    #     self.xmls_to_add = xmls_to_add
    #     self.model, self.data = super().build_model(free_xml_path, xmls_to_add)

    #     # self.randomize_object_positions(["mug", "apple", "sugar_box", "a_cups"], [0.05, 0.05, 0.1, 0.1])
    #     # self.model, self.data = build_model_with_fragments(free_xml_path, xmls_to_add)

    # # def cup_object_xml(self, fixed=False):

    def build_primitive_body_xml(
        self,
        body_name: str,
        geom_name: str | None = None,
        prim_type: str = "box",
        dims: list[float] | tuple[float, ...] = (1.0, 1.0, 1.0),
        pos: list[float] | tuple[float, float, float] = (0.0, 0.0, 0.0),
        quat_xyzw: list[float] | tuple[float, float, float, float] = (
            0.0,
            0.0,
            0.0,
            1.0,
        ),
        rgba: list[float] | tuple[float, float, float, float] | None = None,
        contype: int = 1,
        conaffinity: int = 1,
        make_free: bool = False,
    ) -> str:
        """
        Create XML for a single primitive inside its own <body>, similar to build_xml().

        - prim_type: "box" or "cylinder"
        * box dims = (lx, ly, lz)  -> mj_size = (lx/2, ly/2, lz/2)
        * cylinder dims = (height, radius) -> mj_size = (radius, height/2)
        - quat_xyzw is converted to MuJoCo quat order (w x y z)
        - If make_free=True, adds <joint type="free"> so you can move the body by setting qpos later.
        """

        if rgba is None:
            rgba = [0.133, 0.6, 0.329, 1.0]  # your last default
            rgba = [0.75, 0.75, 0.75, 1.0]

        if geom_name is None:
            geom_name = body_name

        prim_type = prim_type.lower()
        dims = list(dims)

        quat_wxyz = self.quat_xyzw_to_wxyz(quat_xyzw)

        if prim_type == "box":
            if len(dims) != 3:
                raise ValueError(f"box dims must be (lx, ly, lz), got {dims}")
            mj_type = "box"
            mj_size = [dims[0] / 2.0, dims[1] / 2.0, dims[2] / 2.0]

        elif prim_type == "cylinder":
            if len(dims) != 2:
                raise ValueError(
                    f"cylinder dims must be (height, radius), got {dims}"
                )
            height, radius = dims[0], dims[1]
            mj_type = "cylinder"
            mj_size = [radius, height / 2.0]

        else:
            raise ValueError(f"Unsupported primitive type: {prim_type}")

        lines = []
        lines.append(f'<body name="{body_name}" pos="0 0 0">')

        if make_free:
            # Free joint so the body's pose is controlled via qpos (7 values: x y z qw qx qy qz)
            lines.append(f'  <joint name="{body_name}_free" type="free"/>')

        # Match your pattern: pose on geom (not body)
        lines.append(
            f'  <geom name="{geom_name}" type="{mj_type}" '
            f'pos="{self.fmt(pos)}" quat="{self.fmt(quat_wxyz)}" '
            f'size="{self.fmt(mj_size)}" '
            f'contype="{contype}" conaffinity="{conaffinity}" '
            f'rgba="{self.fmt(rgba)}"/>'
        )

        lines.append("</body>")
        return "\n".join(lines)

    def randomize_object_positions(self, objects, object_heights):
        for i, object in enumerate(objects):
            joint_name = f"{object}_joint"

            x = np.random.uniform(-0.5, 0.5)
            y = np.random.uniform(-0.5, -0.1)
            z = object_heights[i] / 2
            yaw = np.random.uniform(-np.pi, np.pi)

            jid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            qadr = self.model.jnt_qposadr[jid]
            vadr = self.model.jnt_dofadr[jid]

            half = 0.5 * float(yaw)
            qw = np.cos(half)
            qx = 0.0
            qy = 0.0
            qz = np.sin(half)

            # free joint qpos layout: [x y z qw qx qy qz]
            self.data.qpos[qadr : qadr + 7] = [x, y, z, qw, qx, qy, qz]
            self.data.qvel[vadr : vadr + 6] = 0.0

            mujoco.mj_forward(self.model, self.data)

    def select_object(self, object_name):
        if object_name == "mug":
            object_path = "assets/ycb/mug.xml"
            self.object_details = {
                "size": [0.05, 0.052, 0.052],
                "type": "mug",
                "yaw": 0,
            }
            self.yaw_buffer = 6 * (np.pi / 180)
            return object_path
        elif object_name == "a_cups":
            object_path = "assets/ycb/a_cups.xml"
            self.object_details = {
                "size": [0.05, 0.052, 0.052],
                "type": "a_cups",
                "yaw": 0,
            }
            self.yaw_buffer = 6 * (np.pi / 180)
            return object_path
        elif object_name == "g_cups":
            object_path = "assets/ycb/g_cups.xml"
            self.object_details = {
                "size": [0.05, 0.052, 0.06],
                "type": "g_cups",
                "yaw": 0,
            }
            self.yaw_buffer = 6 * (np.pi / 180)
            return object_path

    def mjcf_file_to_fragment(self, path: str) -> str:
        """Read a standalone MJCF file and return an MJCF fragment:
        <asset>...</asset> + worldbody contents (bodies).
        """
        text = Path(path).read_text()

        # grab asset block (optional)
        m_asset = re.search(
            r"<asset\b[^>]*>.*?</asset>", text, flags=re.DOTALL
        )
        asset = m_asset.group(0) if m_asset else ""

        # grab contents inside <worldbody>...</worldbody>
        m_world = re.search(
            r"<worldbody\b[^>]*>(.*?)</worldbody>", text, flags=re.DOTALL
        )
        if not m_world:
            raise ValueError(f"No <worldbody> found in {path}")
        world_contents = m_world.group(1).strip()

        # return fragment: assets + bodies (no <worldbody> wrapper)
        return (asset + "\n\n" + world_contents).strip()

class LargeObjectEnv(MujocoEnv):
    """ Table environment with a large target object"""
    def __init__(self, robot, using_swept_volume=True):
        """Initialize large object environment"""
        super().__init__(robot)

        # Problem parameters
        object_type = "box"
        # object_size = [0.6, 0.06, 0.06]
        object_size = [0.3, 0.04, 0.04]
        yaw_variation = [-0.5*np.pi, 0.5*np.pi]
        # yaw_variation = [-0.1*np.pi, 0.1*np.pi]
        yaw_buffer = 0.5*(np.pi/180)

        # Prepare target object        
        object_variation = {
            'x': [[-0.8, 0.8]],
            'y': [[-0.8, 0.8]],
            'z': [[object_size[2]/2, object_size[2]/2]],
            'yaw': [yaw_variation]
        }
        super().populate_object_details(object_type, object_size, object_variation)

        # Prepare environment details
        if robot=="panda":
            config_yaml = "configs/problems/table_pick_panda.yaml"
        elif robot=="fetch":
            config_yaml = "configs/problems/table_pick_fetch.yaml"
        elif robot=="ur10":
            config_yaml = "configs/problems/table_pick_ur5.yaml"

        scene_yaml = "configs/scenes/table/scene_empty_table.yaml"
        with open(config_yaml, "r") as f:
            config_yaml_data = yaml.safe_load(f)
        
        robot_pos = config_yaml_data['base_offset']['position']
        robot_quat = super().quat_xyzw_to_wxyz(config_yaml_data['base_offset']['orientation'])

        if robot=="fetch":
            robot_pos = [robot_pos[0]-0.05, robot_pos[1], robot_pos[2]]

        if robot == "panda":
            # inner_rad = 0.3
            # outer_rad = 0.7
            inner_rad = 0.3
            # outer_rad = 0.44
            outer_rad = 0.7
        elif robot == "fetch":
            inner_rad = 0.3
            outer_rad = 0.75
            # outer_rad = 0.5
        elif robot == "ur10":
            inner_rad = 0.3
            outer_rad = 0.75

        super().populate_env_details(scene_yaml, robot, "largeobj", robot_pos, robot_quat, outer_rad, inner_rad)

        # Prepare grasp details
        super().populate_grasp_details(yaw_buffer=yaw_buffer, grasp_type="top")
        tcr_intervals = super().construct_tcr()

        # Prepare swept volume (or object geom for validation)
        xmls_to_add = []
        if using_swept_volume:
            xml = super().create_swept_volume(tcr_intervals)
        else:
            if self.object_details['type'] == "box":
                xml = super().cube_object_xml(self.object_details['size'], [1, 1, 0, 0])
            else:
                raise ValueError(f"Currently unsupported object type: {self.object_details['type']}")
        xmls_to_add.append(xml)

        # Prepare environment xmls
        table_xml = super().build_xml(scene_yaml, parent_body_name="scene_table")
        xmls_to_add.append(table_xml)

        free_xml_path = f"{self.robot_dir}/empty_table_scene.xml"
        self.model, self.data = super().build_model(free_xml_path, xmls_to_add)

class MicrowaveEnv(MujocoEnv):
    """Table environment with a microwave object"""
    def __init__(self, robot, using_swept_volume=True):
        """Initialize the microwave environment"""
        super().__init__(robot)

        # Problem parameters
        object_type = "microwave"
        object_size = [0.26, 0.24, 0.21]
        yaw_variation = [-0.5*np.pi, 0.5*np.pi]
        yaw_buffer = 4*(np.pi/180)
        door_buffer = 2*(np.pi/180)

        door_size = [0.012, 0.250, 0.210]

        # Prepare target object
        object_variation = {
            'x': [[-0.8, 0.8]],
            'y': [[-0.8, 0.8]],
            'z': [[object_size[2]/2, object_size[2]/2]],
            'yaw': [yaw_variation],
            'door':[[0, np.pi/2]] 
        }

        self.populate_object_details(object_type, object_size, object_variation)
        self.object_details['door_size'] = door_size
        self.object_details['handle_size'] = [door_size[0]/0.75, door_size[1]*0.08, door_size[2]*0.6]
        self.object_details['handle_size'] = [door_size[0]/0.25, door_size[1]*0.1, door_size[2]*0.7]

        # Prepare environment details
        if robot=="panda":
            config_yaml = "configs/problems/table_pick_panda.yaml"
        elif robot=="fetch":
            config_yaml = "configs/problems/table_pick_fetch.yaml"
        elif robot=="ur10":
            config_yaml = "configs/problems/table_pick_ur5.yaml"

        scene_yaml = "configs/scenes/table/scene_empty_table.yaml"
        with open(config_yaml, "r") as f:
            config_yaml_data = yaml.safe_load(f)
        
        robot_pos = config_yaml_data['base_offset']['position']
        robot_quat = self.quat_xyzw_to_wxyz(config_yaml_data['base_offset']['orientation'])

        if robot == "panda" or robot == "fetch":
            inner_rad = 0.3
            outer_rad = 0.8
            # outer_rad = 0.5763
        elif robot == "ur10":
            inner_rad = 0.3
            outer_rad = 0.75
        
        self.populate_env_details(scene_yaml, robot, "microwave", robot_pos, robot_quat, outer_rad, inner_rad)

        # Prepare grasp details
        self.populate_grasp_details(yaw_buffer=yaw_buffer, door_buffer=door_buffer, grasp_type="front")
        tcr_intervals = self.construct_tcr()    


        # Prepare swept volume (or object geom for validation)
        xmls_to_add = []
        if using_swept_volume:
            xml = super().create_swept_volume(tcr_intervals)
        else:
            xml = self.object_xml(self.object_details['size'], [1, 1, 0, 0])
        xmls_to_add.append(xml)

        # Prepare environment xmls
        table_xml = super().build_xml(scene_yaml, parent_body_name="scene_table")
        xmls_to_add.append(table_xml)

        # Add debugging spheres
        # robot_x, robot_y = self.env_details["robot_pos"][:2]
        # z = self.env_details['z_correction'][0]

        # inner_xml = self.debug_sphere_xml(
        #     "inner_radius_debug",
        #     [robot_x + self.env_details["inner_rad"], robot_y, z],
        # )

        # outer_xml = self.debug_sphere_xml(
        #     "outer_radius_debug",
        #     [robot_x + self.env_details["outer_rad"], robot_y, z],
        # )

        # xmls_to_add.append(inner_xml)
        # xmls_to_add.append(outer_xml)

        free_xml_path = f"{self.robot_dir}/empty_table_scene.xml"
        self.model, self.data = super().build_model(free_xml_path, xmls_to_add)

    def populate_object_details(self, object_type, object_size, object_variation):
        
        if object_type != "microwave":
            raise ValueError(f"MicrowaveEnv only supports microwave objects. Unsupported object type: {object_type}")

        for variation_axis in object_variation:
            if variation_axis not in ["x", "y", "z", "yaw", "door"]:
                raise ValueError(f"Unsupported variation given for {object_type}: {variation_axis}")

        self.object_details = {
            'variation': object_variation,
            'type': object_type,
            'size': object_size
        }
    
    def populate_grasp_details(
        self, 
        alpha=0.95,
        yaw_buffer=6*(np.pi/180),
        door_buffer=1*(np.pi/180),
        grasp_type="front"
    ):
        
        self.grasp_details = {
            'type': grasp_type,
            'alpha': alpha,
            'yaw_buffer': yaw_buffer,
            'door_buffer': door_buffer 
        }

class AllStableEnv(MujocoEnv):
    """Table environment with a cube object in all stable configurations"""
    def __init__(self, robot, using_swept_volume=True):
        """Initialize the AllStable environment"""
        super().__init__(robot)

        # Problem parameters
        object_type = "box"
        object_size = [0.04, 0.06, 0.03]
        yaw_variation = [-0.5*np.pi, 0.5*np.pi]
        yaw_buffer = 6*(np.pi/180)

        # Prepare target object        
        object_variation = {
            'x': [[-0.8, 0.8]],
            'y': [[-0.8, 0.8]],
            'z': [[object_size[2]/2, object_size[2]/2]],
            'yaw': [yaw_variation]
        }
        super().populate_object_details(object_type, object_size, object_variation)

        # Prepare environment details
        if robot=="panda":
            config_yaml = "configs/problems/table_pick_panda.yaml"
        elif robot=="fetch":
            config_yaml = "configs/problems/table_pick_fetch.yaml"
        elif robot=="ur10":
            config_yaml = "configs/problems/table_pick_ur5.yaml"
        
        scene_yaml = "configs/scenes/table/scene_empty_table.yaml"
        with open(config_yaml, "r") as f:
            config_yaml_data = yaml.safe_load(f)
        
        robot_pos = config_yaml_data['base_offset']['position']
        robot_quat = super().quat_xyzw_to_wxyz(config_yaml_data['base_offset']['orientation'])

        if robot == "panda" or robot == "fetch":
            inner_rad = 0.3
            outer_rad = 0.7
            # outer_rad = 0.5
        elif robot == "ur10":
            inner_rad = 0.3
            outer_rad = 0.75

        super().populate_env_details(scene_yaml, robot, "allstable", robot_pos, robot_quat, outer_rad, inner_rad)
        self.env_details['tcr_batches'] = ['xy', 'yz', 'zx']

        # Prepare grasp details
        super().populate_grasp_details(yaw_buffer=yaw_buffer, grasp_type="top")
        tcr_intervals = super().construct_tcr()

        # Prepare swept volume (or object geom for validation)
        xmls_to_add = []
        if using_swept_volume:
            
            sv1_xml = super().create_swept_volume(
                tcr_intervals['xy'],
                object_size,
                sv_count=0
            )

            sv2_xml = super().create_swept_volume(
                tcr_intervals['yz'],
                [object_size[1], object_size[2], object_size[0]],
                sv_count=1
            )

            sv3_xml = super().create_swept_volume(
                tcr_intervals['zx'],
                [object_size[2], object_size[0], object_size[1]],
                sv_count=2
            )
            xmls_to_add.extend([sv1_xml, sv2_xml, sv3_xml])
            # print(f"xmls_to_add: {xmls_to_add}")
        else:
            if self.object_details['type'] == "box":
                xml1 = super().cube_object_xml(
                    [object_size[0], object_size[1], object_size[2]], [1, 1, 0, 0], name="cube_object_0"
                )
                xml2 = super().cube_object_xml(
                    [object_size[1], object_size[2], object_size[0]], [1, 1, 0, 0], name="cube_object_1"
                )
                xml3 = super().cube_object_xml(
                    [object_size[2], object_size[0], object_size[1]], [1, 1, 0, 0], name="cube_object_2"
                )
                
                xmls_to_add.extend([xml1, xml2, xml3])
            else:
                raise ValueError(f"Currently unsupported object type: {self.object_details['type']}")
        

        # Prepare environment xmls
        table_xml = super().build_xml(scene_yaml, parent_body_name="scene_table")
        xmls_to_add.append(table_xml)

        free_xml_path = f"{self.robot_dir}/empty_table_scene.xml"
        self.model, self.data = super().build_model(free_xml_path, xmls_to_add)

    def generate_task_set(self):
        """Generate task set/TSRs"""

        TCR_set = {}
        
        for i, face_in_contact in enumerate(['xy', 'yz', 'zx']):    
            TCR_set.update({
                (face_in_contact, ) + key: value
                for key, value in create_TCR_set(self, batch_idx=face_in_contact).items()
            })
            
        self.task_set = TCR_set
        return self.task_set


if __name__ == "__main__":
    # Load environment and generate task set
    robot_chosen = "ur10"
    environment = "real"
    ik = "neighbor"
    planner = "RRTConnect"

    if environment == "table":
        env = TableEnv(robot_chosen, no_sv=True)
    elif environment == "cage":
        env = CageEnv(robot_chosen, no_sv=True)
    elif environment == "shelf":
        env = ShelfEnv(robot_chosen, no_sv=True)
    elif environment == "largeobj":
        env = LargeObjectEnv(robot_chosen, no_sv=True)
    elif environment == "microwave":
        env = MicrowaveEnv(robot_chosen, using_swept_volume=True)
    elif environment == "real":
        env = RealEnv(robot_chosen, using_swept_volume=True)
    else:
        env = FreeEnv(robot_chosen, using_swept_volume=True)

    model, data = env.model, env.data
    # task_set = env.generate_task_set()
    # env.move_cube_object([-env.object_outer_rad, -env.object_outer_rad, env.object_details['position'][2], 0])

    folder = f"data/{environment}_{robot_chosen}"
    # task_set = pickle.load(open(f"{folder}/task_set.pkl", "rb"))

    # d_name = f"{folder}/task_paths_data_{ik}_{planner}.npy"
    # path_data = np.load(d_name, allow_pickle=True)
    # k_name = f"{folder}/task_paths_keys_{ik}_{planner}.pkl"
    # keys = pickle.load(open(k_name, "rb"))
    # task_paths = {key: data for key, data in zip(keys, path_data)}

    # solved_task_keys = [
    #     key for key, value in task_paths.items()
    #     if value is not None and len(value) > 1
    # ]

    # task_keys = list(task_set.keys())
    # task_keys = solved_task_keys
    # random_ind = np.random.randint(0, len(task_keys))
    # key = task_keys[random_ind]

    if robot_chosen == "panda":
        robot = Panda(model, data, visualize=True)
    elif robot_chosen == "fetch":
        robot = FetchArm(model, data, visualize=True)
    elif robot_chosen == "ur10":
        robot = UR10(model, data, visualize=True)
    else:
        robot = G1(model, data, visualize=True)

    robot.teleport_base(np.array(env.env_details['robot_pos']), np.array(env.env_details['robot_quat']))

    # env.move_swept_volume([[1, 1], [1, 1], [1, 1], [np.pi/2, np.pi/2], [np.pi/4, np.pi/4]])

    # if robot_chosen == "fetch":
    #     robot = FetchArm(model, data, visualize=True, prefix="f1_")
    #     robot2 = FetchArm(model, data, visualize=False, prefix="f2_")
    # elif robot_chosen == "panda":
    #     robot = Panda(model, data, visualize=True, prefix="f1_")
    #     robot2 = Panda(model, data, visualize=False, prefix="f2_")
    # robot.teleport_base(np.array([env.robot_pos]), np.array(env.robot_quat))
    # robot2.teleport_base(np.array([env.robot_pos]), np.array(env.robot_quat))

    # Configure camera
    robot.viewer.cam.lookat[:] = [0.25, -0.25, 0.5]
    robot.viewer.cam.distance = 1.75
    robot.viewer.cam.azimuth = 120
    robot.viewer.cam.elevation = -20
    camview = robot.viewer.cam
    # table
    if environment == "table":
        robot.viewer.cam.lookat[:] = [0.90100131, 0.15017647, 0.63007395]
        robot.viewer.cam.distance = 3.4766360559393026
        robot.viewer.cam.azimuth = -59.922579098753594
        robot.viewer.cam.elevation = -16.780680728667303

        camview.lookat = [1.0493371, 0.05479526, 0.52797369]
        camview.distance = 3.895318011088692
        camview.azimuth = 55.705417066155334
        camview.elevation = -40.769175455417056

    elif environment == "cage":
        robot.viewer.cam.lookat[:] = [0.52086037, -0.06213331, 0.49931841]
        robot.viewer.cam.distance = 3.079438735523704
        robot.viewer.cam.azimuth = -43.93624161073823
        robot.viewer.cam.elevation = -15.453259827420883

        camview.lookat = [0.45476833, 0.04653378, 0.75122542]
        camview.distance = 2.702403732778341
        camview.azimuth = 29.927612655800647
        camview.elevation = -17.351629913710447

        camview.lookat = [0.45352783, 0.04873283, 0.63760923]
        camview.distance = 3.178760106296739
        camview.azimuth = 29.08245445829346
        camview.elevation = -23.564477468839865

    else:
        robot.viewer.cam.lookat[:] = [0.84953477, 0.10998711, 0.76237909]
        robot.viewer.cam.distance = 3.4766360559393026
        robot.viewer.cam.azimuth = -50.630632790028756
        robot.viewer.cam.elevation = -18.232262703739217

        camview.lookat = [0.82747485, 0.12431182, 0.70116041]
        camview.distance = 3.7227554157167515
        camview.azimuth = 35.47267497603071
        camview.elevation = -14.435522531160109

    pos1 = [[1000, 1000], [0, 0], [0.05, 0.05], [0, 0]]
    #env.move_swept_volume(pos1)
    
    # new_home_pos = np.array([117.52, -61.10, 89.46, -119.21, -91.35, 30.14])
    # new_home_pos = np.array([116.36, -62.72, 90.52, -118.66, -91.32, 28.99])
    # new_home_pos = to_rad(new_home_pos)
    # print(new_home_pos)
    # robot.set_joint_qpos(new_home_pos)

    # env.move_swept_volume(task_keys[random_ind])
    # sample = [float(np.random.uniform(lo, hi)) for (lo, hi) in key]
    # env.move_cube_object(sample)

    # print(f"in contact: {robot.in_contact()}")
    # robot.viewer.sync()

    # cameraPos = None
    # while(cameraPos is None):
    #     cam = robot.viewer.cam
    #     cameraPos = input("print camera pos?")

    # print("lookat   :", cam.lookat)
    # print("distance :", cam.distance)
    # print("azimuth  :", cam.azimuth)
    # print("elevation:", cam.elevation)

    # while(True):
    #     random_ind = np.random.randint(0, len(task_keys))
    #     random_key = task_keys[random_ind]
    #     env.move_swept_volume(random_key)
    #     robot.viewer.sync()

    #     path = task_paths[random_key]
    #     goal = path[-1]

    #     action = input()

    #     if action != "":
    #         robot.set_joint_qpos(goal)
    #         robot.viewer.sync()
    #         input()

    print("gravity:", model.opt.gravity)
    print("timestep:", model.opt.timestep)
    print("nq/nv/nu:", model.nq, model.nv, model.nu)

    # robot.set_joint_qpos()

    # data.qvel[0] = 0.5   # x linear velocity
    # data.qvel[1] = 0.2   # y linear velocity
    # data.qvel[3] = 1.0   # angular velocity-ish depending on freejoint velocity convention

    # pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

    for i in range(model.njnt):
        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            i,
        )
        print(f"{i:2d}: {name}")

    # Keep the viewer
    try:
        while True:
            # data.xfrc_applied[pelvis_id, 0] = 50.0  # push in world +x
            mujoco.mj_forward(model, data)
            robot.viewer.sync()
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        robot.close()
