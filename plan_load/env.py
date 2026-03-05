import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import numpy as np
import yaml
import mujoco
import mujoco.viewer
import pickle

import re
from pathlib import Path


from plan_load.mujoco_utils import joint_names_to_joint_ids
from plan_load.mujoco_utils import joints_to_qpos_dof_ids
from plan_load.mujoco_utils import joints_to_limits

#from plan_load.TSR_generation import panda_TSR_parameters, fetch_TSR_parameters, ur10_TSR_parameters
#from plan_load.TSR_generation import find_yaw_iTSR_set
#from plan_load.TSR_generation import find_iTSR_set

from plan_load.task_generation import find_yaw_iTSR_set, find_iTSR_set
from plan_load.task_generation import panda_TSR_parameters, fetch_TSR_parameters, ur10_TSR_parameters


from plan_load.robot import MujocoRobot
from plan_load.robot import Panda
from plan_load.robot import UR10
from plan_load.robot import FetchArm

def wrap_to_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

class MujocoEnv:
    def __init__(self, robot, custom_base=None):
        """Initialize object dimensions and common parameters"""
        if robot == "panda":
            self.robot_dir = "assets/franka_emika_panda"
        else:
            self.robot_dir = f"assets/{robot}"
        #self.base_xml = f"{self.robot_dir}/scene.xml"
        if custom_base is None:
            self.base_xml = "scene.xml"
        else:
            self.base_xml = custom_base

        # Object parameters
        object_size = [0.03, 0.03, 0.15]
        object_type = "cube"
        self.object_details = {
            'size': object_size,
            'type': object_type,
            'yaw': 0
        }

        self.yaw_buffer = 6*(np.pi/180)
        self.alpha = 0.95
        
        # Updated during swept volume creation
        self.collision_geoms = []
    
    def build_xml(self, parent_body_name="env_name", skip_ids=None, rgba=None):
        """Return xml for environment"""

        if skip_ids is None:
            skip_ids = set()

        # Default to MuJoCo default gray if not provided
        if rgba is None:
            rgba = [0.5, 0.5, 0.5, 1]
            rgba = [0.15, 1, 0.15, 1]
            rgba = [0.133, 0.6, 0.329, 1]

        with open(self.scene_yaml, "r") as f:
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
            self.collision_geoms.append(obj_id)

            if prim_type == "box":
                size = [dims[0] / 2.0, dims[1] / 2.0, dims[2] / 2.0]
                mj_type = "box"
                mj_size = size

            elif prim_type == "cylinder":
                height, radius = dims[0], dims[1]
                mj_type = "cylinder"
                mj_size = [radius, height / 2.0]

            else:
                raise ValueError(f"Unsupported primitive type: {prim_type} for id={obj_id}")

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
        '''Formatting for XML'''
        return " ".join(f"{x:.6g}" for x in v)
    
    def quat_xyzw_to_wxyz(self, quat_xyzw):
        """Convert to wxyz quats"""
        quat_wxyz = [
            quat_xyzw[3],
            quat_xyzw[0],
            quat_xyzw[1],
            quat_xyzw[2]
        ]
        return quat_wxyz

    # def build_model(self, xml_path, xmls_to_add):
    #     """
    #     Build final xml and model
    #     xml_path: desired path for model's xml
    #     xmls_to_add: list of xml strings to add to model, prepared from build_xml()
    #     """
        
    #     curr_xml = f"""
    #     <mujoco model="test_world">
    #     <include file="{self.base_xml}"/>
    #     <worldbody>
    #     """
    #     for primitive_xml in xmls_to_add:
    #         curr_xml += f"{primitive_xml}"
    
    #     curr_xml += """
    #         </worldbody>
    #     </mujoco>
    #     """
    #     with open(xml_path, "w") as f:
    #         f.write(curr_xml)
    #     #print(xml_path)
    #     model = mujoco.MjModel.from_xml_path(xml_path)
    #     data = mujoco.MjData(model)
    #     return model, data

    def build_model(self, xml_path, xmls_to_add):
        """
        Build final xml and model
        xml_path: desired path for model's xml
        xmls_to_add: list of xml fragments containing <asset> and/or <body>
        """

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

        b1_size = np.array([ (x_upper - x_lower) + 2*R_cyl,
                            (y_upper - y_lower),
                            np.full_like(cx, zdim) ]).T   # (B,3)

        b2_size = np.array([ (x_upper - x_lower),
                            (y_upper - y_lower) + 2*R_cyl,
                            np.full_like(cx, zdim) ]).T   # (B,3)

        b_pos = np.stack([cx, cy, z], axis=1)  # (B,3)

        corners = np.stack([
            np.stack([x_lower, y_lower, z], axis=1),
            np.stack([x_lower, y_upper, z], axis=1),
            np.stack([x_upper, y_lower, z], axis=1),
            np.stack([x_upper, y_upper, z], axis=1),
        ], axis=1)  # (B,4,3)

        return R_cyl, b_pos, b1_size, b2_size, corners
    
    def compute_cyl_sv_params(self, object_dims, object_configs):
        """Compute swept volume dimensions"""
        object_configs = np.asarray(object_configs, dtype=np.float64)
        #xdim, ydim, zdim = object_dims
        rdim, zdim = object_dims  
        if object_configs.ndim == 2:
            object_configs = object_configs[None, :, :]

        x_lower = object_configs[:, 0, 0]
        x_upper = object_configs[:, 0, 1]
        y_lower = object_configs[:, 1, 0]
        y_upper = object_configs[:, 1, 1]
        z = object_configs[:, 2, 0] 

        #R_cyl = float(np.round(0.5 * np.sqrt(xdim**2 + ydim**2), 5))
        R_cyl = rdim
        cx = 0.5 * (x_upper + x_lower)
        cy = 0.5 * (y_upper + y_lower)

        b1_size = np.array([ (x_upper - x_lower) + 2*R_cyl,
                            (y_upper - y_lower),
                            np.full_like(cx, zdim) ]).T   # (B,3)

        b2_size = np.array([ (x_upper - x_lower),
                            (y_upper - y_lower) + 2*R_cyl,
                            np.full_like(cx, zdim) ]).T   # (B,3)

        b_pos = np.stack([cx, cy, z], axis=1)  # (B,3)

        corners = np.stack([
            np.stack([x_lower, y_lower, z], axis=1),
            np.stack([x_lower, y_upper, z], axis=1),
            np.stack([x_upper, y_lower, z], axis=1),
            np.stack([x_upper, y_upper, z], axis=1),
        ], axis=1)  # (B,4,3)

        return R_cyl, b_pos, b1_size, b2_size, corners

    def cube_object_xml(self, object_dims, object_pose, fixed=False):
        """Create cube object xml string"""
        rgba = [0.8, 0.2, 0.2, 1]
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
            <geom name="sv_box1" type="box" pos="0 0 0"
                size="{object_dims[0]/2} {object_dims[1]/2} {object_dims[2]/2}"
                rgba="{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}"/>
        </body>
        """
        return obj_xml
    
    def move_cube_object(self, object_pose):
        """
        Move cube_object to (x, y, z, yaw) by writing into its free joint qpos.
        object_pose: iterable length-4: (x, y, z, yaw) in radians
        """
        x, y, z, yaw = object_pose

        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_object_free")
        qadr = self.model.jnt_qposadr[jid]
        vadr = self.model.jnt_dofadr[jid]

        half = 0.5 * float(yaw)
        qw = np.cos(half)
        qx = 0.0
        qy = 0.0
        qz = np.sin(half)

        # free joint qpos layout: [x y z qw qx qy qz]
        self.data.qpos[qadr:qadr+7] = [x, y, z, qw, qx, qy, qz]
        self.data.qvel[vadr:vadr+6] = 0.0

        mujoco.mj_forward(self.model, self.data)

    def cube_swept_volume_xml(self, object_dims, object_configs, fixed=False, cyl=False):
        """Create swept volume xml string"""
        rgba = [0.8, 0.8, 0.8, 1]
        name = "swept_volume"
        joint_xml = "" if fixed else f'<joint name="{name}_free" type="free"/>'
        
        if cyl==True:
            R_cyl, b_pos, b1_size, b2_size, corners = self.compute_cyl_sv_params(object_dims, object_configs)
            z_cyl = object_dims[1]/2
        else:
            R_cyl, b_pos, b1_size, b2_size, corners = self.compute_sv_params(object_dims, object_configs)
            z_cyl = object_dims[2]/2

        b_pos0 = b_pos[0]              # numpy (3,)
        b1_size0 = b1_size[0].tolist()
        b2_size0 = b2_size[0].tolist()
        corners0 = corners[0]          # (4,3) world
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

        self.collision_geoms.extend(["sv_box1", "sv_box2", "sv_cyl1", "sv_cyl2", "sv_cyl3", "sv_cyl4"])

        return sv_xml

    def move_swept_volume(self, object_configs):
        """Move swept volume to desired bin"""
        svid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "swept_volume_free")
        sv_adr = self.model.jnt_qposadr[svid]
        sv_vadr = self.model.jnt_dofadr[svid]

        object_configs = np.asarray(object_configs, dtype=np.float64)
        if object_configs.ndim == 2:
            object_configs = object_configs[None, :, :]

        x_lower = object_configs[:, 0, 0]
        x_upper = object_configs[:, 0, 1]
        y_lower = object_configs[:, 1, 0]
        y_upper = object_configs[:, 1, 1]
        z = object_configs[:, 2, 0] 

        cx = 0.5 * (x_upper + x_lower)
        cy = 0.5 * (y_upper + y_lower)
        new_pos = [cx[0], cy[0], z[0]]
        new_quat = [1, 0, 0, 0]

        self.data.qpos[sv_adr: sv_adr + 7] = [new_pos[0], new_pos[1], new_pos[2], new_quat[0], new_quat[1], new_quat[2], new_quat[3]]
        self.data.qvel[sv_vadr: sv_vadr + 6] = 0

        mujoco.mj_forward(self.model, self.data)

    def initialize_TSR_parameters(self, robot, grasp_strategy="top"):
        if robot=="panda":
            TSR_params = panda_TSR_parameters(self.object_details, self.yaw_buffer, self.alpha)[grasp_strategy]
        elif robot=="fetch":
            TSR_params = fetch_TSR_parameters(self.object_details, self.yaw_buffer, self.alpha)[grasp_strategy]
        elif robot=="ur10":
            TSR_params = ur10_TSR_parameters(self.object_details, self.yaw_buffer, self.alpha)[grasp_strategy]

        self.ee_offset, self.Bw, self.half_side, self.Tw2_w1, self.yaw_tw2_w1 = TSR_params
        self.problem_details_grasp = {
            'Bw': self.Bw,
            'half_side': self.half_side,
            'yaw_buffer': self.yaw_buffer,
            'alpha': self.alpha,
            'reachable_ws': self.object_outer_rad,
            'robot_clearance': self.object_inner_rad
        }

        if grasp_strategy == "front":
            pass
            p_nom = np.array([self.object_outer_rad, 0, 0])
            from_robot_nom = p_nom - self.robot_pos
            from_robot_nom[2] = 0.0
            from_robot_nom /= (np.linalg.norm(from_robot_nom) + 1e-12)
            
            yaw1 = -self.object_yaw
            yaw2 = self.object_yaw
            yaw_edges = np.arange(yaw1, yaw2 + self.yaw_buffer, self.yaw_buffer)
            yaw_centers = (yaw_edges[:-1] + yaw_edges[1:])*0.5
            
            #best_idx = np.zeros(len(yaw_centers), dtype=np.int64)
            worst_idx = np.zeros(len(yaw_centers), dtype=np.int64)

            z_axis = np.array([0.0, 0.0, 1.0])
            Tews = self.ee_offset  # your 4 variants

            for k, yaw in enumerate(yaw_centers):
                # nominal obj rotation about world Z
                cy, sy = np.cos(yaw), np.sin(yaw)
                Rwo = np.array([[cy, -sy, 0.0],
                                [sy,  cy, 0.0],
                                [0.0, 0.0, 1.0]], dtype=float)

                #best_s = -float("inf")
                #best_i = 0
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
                    #if s > best_s:
                    #    best_s = s
                    #    best_i = i
                    if s < worst_s:
                        worst_s = s
                        worst_i = i

                #best_idx[k] = best_i
                worst_idx[k] = worst_i

            self.yaw_edges = yaw_edges
            #self.best_ee_offset_idx = best_idx
            self.worst_ee_offset_idx = worst_idx

        else:

            self.yaw_edges = None
            self.worst_ee_offset_idx = None

        self.problem_details = {
            f"{grasp_strategy}": self.problem_details_grasp
        }
        self.yaw_tw2_w1_dict = {
            f"{grasp_strategy}": self.yaw_tw2_w1
        }
        sv_config = [
            [0, round(self.yaw_tw2_w1[0], 5)],
            [0, round(self.yaw_tw2_w1[1], 5)],
            [round(self.object_details['position'][2], 5), round(self.object_details['position'][2], 5)],
            [0, round(self.Tw2_w1[3], 5)]
        ]
        return sv_config

    def find_problem_intervals(self, base_name="base", wall_clearance=0.18):
        """Find x,y intervals for valid object positions in problem"""
        with open(self.scene_yaml, "r") as f:
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
        
        # Update nominal object position with updated z
        self.object_details['position'] = [
            self.robot_pos[0], self.robot_pos[1], base_pos[2] + (base_dim[2]/2) + self.object_details['size'][2]/2
        ]

        hx_int = base_dim[0]/2 - wall_clearance/2
        hy_int = base_dim[1]/2 - wall_clearance/2

        base_xmin = base_pos[0] - hx_int + self.object_details['size'][0]/2
        base_xmax = base_pos[0] + hx_int - self.object_details['size'][0]/2
        base_ymin = base_pos[1] - hy_int + self.object_details['size'][1]/2
        base_ymax = base_pos[1] + hy_int - self.object_details['size'][1]/2

        base_intervals = [
            [base_xmin, base_xmax],
            [base_ymin, base_ymax]
        ]
        return base_intervals

    def generate_task_set(self):
        """Generate task set/TSRs"""
        yaw_iTSR_set, _ = find_yaw_iTSR_set(self.object_details, self.problem_details, self.Tw2_w1,) 
        iTSR_set, _ = find_iTSR_set(self.object_details, self.problem_details, self.yaw_tw2_w1_dict, yaw_iTSR_set, problem=self.problem, robot_pos=self.robot_pos)
        self.task_set = iTSR_set[0]
        return self.task_set


class FreeEnv(MujocoEnv):
    """Free environment"""
    def __init__(self, robot, no_sv=False):
        """Initialize free environment"""
        super().__init__(robot)
        self.robot_pos = [0, 0, 0]
        if robot == "fetch":
            self.robot_pos = [0, 0, 0.005]
        self.robot_quat = [1, 0, 0, 0] #w,x,y,z
        self.object_details['position'] = [self.robot_pos[0], self.robot_pos[1], self.object_details['size'][2]/2]

        # Annulus of object positions
        if robot=="panda":
            self.object_inner_rad = 0.3
            self.object_outer_rad = 0.8
        elif robot=="fetch":
            self.object_inner_rad = 0.3
            self.object_outer_rad = 0.7
        elif robot=="ur10":
            self.object_inner_rad = 0.3
            self.object_outer_rad = 0.8

        self.object_yaw = 0.25*np.pi #-yaw to +yaw
        self.object_details['dist'] = [
            self.object_outer_rad, self.object_outer_rad, 0, self.object_yaw
        ]
        self.problem = {
            'name': None,
            'intervals': None,
            'robot': f"{robot}"   
        }
        # Find TSR parameters
        sv_config = super().initialize_TSR_parameters(robot, grasp_strategy="top")

        # Add environment xmls and build model
        if no_sv==False:
            sv_xml = super().cube_swept_volume_xml(self.object_details['size'], sv_config)
        else:
            sv_xml = super().cube_object_xml(self.object_details['size'], [1, 1, 0, 0])
        xmls_to_add = [sv_xml]
        free_xml_path = f"{self.robot_dir}/free_scene.xml"
        self.model, self.data = super().build_model(free_xml_path, xmls_to_add)

        
class BoxEnv(MujocoEnv):
    """Box environment"""
    def __init__(self, robot, no_sv=False):
        """Initialize box environment"""
        super().__init__(robot)
        if robot=="panda":
            self.config_yaml = "configs/problems/box_panda.yaml"
        elif robot=="fetch":
            self.config_yaml = "configs/problems/box_fetch.yaml"
        elif robot=="ur10":
            self.config_yaml = "configs/problems/box_ur5.yaml"
        
        self.scene_yaml = "configs/scenes/box/scene_box.yaml"
        with open(self.config_yaml, "r") as f:
            config_yaml_data = yaml.safe_load(f)
        
        self.robot_pos = config_yaml_data['base_offset']['position']
        self.robot_quat = super().quat_xyzw_to_wxyz(config_yaml_data['base_offset']['orientation'])

        # Find valid x,y intervals for object placements in problem
        box_thickness = 0.18
        box_intervals = super().find_problem_intervals(base_name="base", wall_clearance=box_thickness)
        self.problem = {
            'name': "box",
            'intervals': box_intervals,
            'robot': f"{robot}"   
        }
        # Annulus of object positions
        self.object_inner_rad = 0.3
        self.object_outer_rad = 0.75
        self.object_yaw = 0.25*np.pi #-yaw to +yaw
        self.object_details['dist'] = [
            self.object_outer_rad, self.object_outer_rad, 0, self.object_yaw
        ]
        # Find TSR parameters
        sv_config = super().initialize_TSR_parameters(robot, grasp_strategy="top")
        
        # Add environment xmls and build model
        #sv_xml = super().cube_swept_volume_xml(self.object_details['size'], sv_config)
        if no_sv==False:
            sv_xml = super().cube_swept_volume_xml(self.object_details['size'], sv_config)
        else:
            sv_xml = super().cube_object_xml(self.object_details['size'], [1, 1, 0, 0])
        box_xml = super().build_xml(parent_body_name="scene_box", skip_ids={"Can1"})
        
        xmls_to_add = [sv_xml, box_xml]
        free_xml_path = f"{self.robot_dir}/box_scene.xml"
        self.model, self.data = super().build_model(free_xml_path, xmls_to_add)


class CageEnv(MujocoEnv):
    """Cage environment"""
    def __init__(self, robot, no_sv=False):
        """Initialize cage environment"""
        super().__init__(robot)
        if robot=="panda":
            self.config_yaml = "configs/problems/cage_panda.yaml"
        elif robot=="fetch":
            self.config_yaml = "configs/problems/cage_fetch.yaml"
        elif robot=="ur10":
            self.config_yaml = "configs/problems/cage_ur5.yaml"
        
        self.scene_yaml = "configs/scenes/cage/scene_cage.yaml"
        with open(self.config_yaml, "r") as f:
            config_yaml_data = yaml.safe_load(f)
        
        self.robot_pos = config_yaml_data['base_offset']['position']
        self.robot_quat = super().quat_xyzw_to_wxyz(config_yaml_data['base_offset']['orientation'])

        # Find valid x,y intervals for object placements in problem
        cage_thickness = 0.18
        cage_intervals = super().find_problem_intervals(base_name="base", wall_clearance=cage_thickness)
        self.problem = {
            'name': "box",
            'intervals': cage_intervals,
            'robot': f"{robot}"   
        }
        # Annulus of object positions
        self.object_inner_rad = 0.3
        self.object_outer_rad = 0.75
        self.object_yaw = 0.25*np.pi #-yaw to +yaw
        self.object_details['dist'] = [
            self.object_outer_rad, self.object_outer_rad, 0, self.object_yaw
        ]
        # Find TSR parameters
        sv_config = super().initialize_TSR_parameters(robot, grasp_strategy="top")
        
        # Add environment xmls and build model
        #sv_xml = super().cube_swept_volume_xml(self.object_details['size'], sv_config)
        if no_sv==False:
            sv_xml = super().cube_swept_volume_xml(self.object_details['size'], sv_config)
        else:
            sv_xml = super().cube_object_xml(self.object_details['size'], [1, 1, 0, 0])
        cage_xml = super().build_xml(parent_body_name="scene_cage", skip_ids={"Cube1"})
        
        xmls_to_add = [sv_xml, cage_xml]
        free_xml_path = f"{self.robot_dir}/cage_scene.xml"
        self.model, self.data = super().build_model(free_xml_path, xmls_to_add)
    

class TableEnv(MujocoEnv):
    """Table environment"""
    def __init__(self, robot, no_sv=False):
        """Initialize table environment"""
        super().__init__(robot)
        if robot=="panda":
            self.config_yaml = "configs/problems/table_pick_panda.yaml"
        elif robot=="fetch":
            self.config_yaml = "configs/problems/table_pick_fetch.yaml"
        elif robot=="ur10":
            self.config_yaml = "configs/problems/table_pick_ur5.yaml"
        
        self.scene_yaml = "configs/scenes/table/scene_table.yaml"
        with open(self.config_yaml, "r") as f:
            config_yaml_data = yaml.safe_load(f)
        
        self.robot_pos = config_yaml_data['base_offset']['position']
        self.robot_quat = super().quat_xyzw_to_wxyz(config_yaml_data['base_offset']['orientation'])

        # Find valid x,y intervals for object placements in problem
        table_thickness = 0.18
        table_intervals = super().find_problem_intervals(base_name="table_top", wall_clearance=table_thickness)
        self.problem = {
            'name': "table",
            'intervals': table_intervals,
            'robot': f"{robot}"    
        }
        # Annulus of object positions

        if robot == "panda" or robot == "fetch":
            self.object_inner_rad = 0.3
            self.object_outer_rad = 0.75
        elif robot == "ur10":
            self.object_inner_rad = 0.3
            self.object_outer_rad = 0.8

        self.object_yaw = 0.25*np.pi #-yaw to +yaw
        self.object_details['dist'] = [
            self.object_outer_rad, self.object_outer_rad, 0, self.object_yaw
        ]
        # Find TSR parameters
        sv_config = super().initialize_TSR_parameters(robot, grasp_strategy="top")
        
        # Add environment xmls and build model
        #sv_xml = super().cube_swept_volume_xml(self.object_details['size'], sv_config)
        if no_sv==False:
            sv_xml = super().cube_swept_volume_xml(self.object_details['size'], sv_config)
        else:
            sv_xml = super().cube_object_xml(self.object_details['size'], [1, 1, 0, 0])
        table_xml = super().build_xml(parent_body_name="scene_table", skip_ids={"Cube1"})
        
        xmls_to_add = [sv_xml, table_xml]
        free_xml_path = f"{self.robot_dir}/table_scene.xml"
        self.model, self.data = super().build_model(free_xml_path, xmls_to_add)

class ShelfEnv(MujocoEnv):
    """Thin shelf environment"""
    def __init__(self, robot, no_sv=False):
        """Initialize thin shelf environment"""
        super().__init__(robot)
        if robot=="panda":
            self.config_yaml = "configs/problems/bookshelf_thin_panda.yaml"
        elif robot=="fetch":
            self.config_yaml = "configs/problems/bookshelf_thin_fetch.yaml"
        elif robot=="ur10":
            self.config_yaml = "configs/problems/bookshelf_thin_ur5.yaml"
        #TODO: Add other robots
        self.scene_yaml = "configs/scenes/bookshelf/scene_thin.yaml"
        with open(self.config_yaml, "r") as f:
            config_yaml_data = yaml.safe_load(f)
        self.robot_pos = config_yaml_data['base_offset']['position']
        self.robot_quat = super().quat_xyzw_to_wxyz(config_yaml_data['base_offset']['orientation'])

        #shelf_thickness = 0.18
        shelf_thickness = 0.12
        dividing_wall_thickness = 0.14
        #bases = ['shelf_bottom', 'shelf_middle_bottom', 'shelf_middle', 'shelf_middle_top', 'shelf_top']
        bases = ['shelf_middle']
        shelf_intervals = self.find_problem_intervals(bases, shelf_thickness, dividing_wall_thickness)
        self.problem = {
            'name': "shelf",
            'intervals': shelf_intervals,
            'robot': f"{robot}"    
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
            #self.object_outer_rad = 1.1
            self.object_outer_rad = 0.65

            #self.robot_pos[0] = self.robot_pos[0]-0.8

        self.object_yaw = 0.01*np.pi #-yaw to +yaw
        self.object_details['dist'] = [
            self.object_outer_rad, self.object_outer_rad, 0, self.object_yaw
        ]
        # Find TSR parameters
        sv_config = super().initialize_TSR_parameters(robot, grasp_strategy="front")

        # Add environment xmls and build model
        #sv_xml = super().cube_swept_volume_xml(self.object_details['size'], sv_config)
        if no_sv==False:
            sv_xml = super().cube_swept_volume_xml(self.object_details['size'], sv_config)
        else:
            sv_xml = super().cube_object_xml(self.object_details['size'], [1, 1, 0, 0])
        shelf_xml = super().build_xml(parent_body_name="scene_shelf", skip_ids={"Cube1"})
        
        xmls_to_add = [sv_xml, shelf_xml]
        free_xml_path = f"{self.robot_dir}/shelf_scene.xml"
        self.model, self.data = super().build_model(free_xml_path, xmls_to_add)

    def find_problem_intervals(self, bases, wall_clearance, dividing_wall_clearance):
        with open(self.scene_yaml, "r") as f:
            scene_yaml_data = yaml.safe_load(f)
        objs = scene_yaml_data["world"]["collision_objects"]
        
        base_dim = None
        base_pos = None

        for obj in objs:
            obj_id = obj.get("id", "")
            if obj_id != bases[0]:
                continue
            base_dim = obj["primitives"][0]['dimensions']
            base_pos = obj["primitive_poses"][0]['position']
        

        #bases = ['shelf_bottom', 'shelf_middle_bottom', 'shelf_middle', 'shelf_middle_top', 'shelf_top']
        #bases = ['shelf_middle']
        base_zpos = []
        base_zdim = []
        base_names = []
        for obj in objs:
            obj_id = obj.get("id", "")
            if obj_id not in bases:
                continue
            base_zpos.append(obj["primitive_poses"][0]['position'][2])
            base_zdim.append(obj["primitives"][0]['dimensions'][2])
            base_names.append(obj_id)
        self.base_zpos = base_zpos
        self.base_zdim = base_zdim
        self.base_names = base_names

        self.base_dim = base_dim
        self.base_pos = base_pos

        # Update nominal object position with updated z
        self.object_details['position'] = [
            self.robot_pos[0], self.robot_pos[1], base_pos[2] + (base_dim[2]/2) + self.object_details['size'][2]/2
        ]

        hx_int = base_dim[0]/2 - wall_clearance/2
        hy_int = base_dim[1]/2 - wall_clearance/2

        shelf_xmin = base_pos[0] - hx_int + self.object_details['size'][0]/2
        shelf_xmax = base_pos[0] + hx_int - self.object_details['size'][0]/2
        shelf_ymin = base_pos[1] - hy_int + self.object_details['size'][1]/2
        shelf_ymax = base_pos[1] + hy_int - self.object_details['size'][1]/2

        for obj in objs:
            obj_id = obj.get("id", "")
            if obj_id != "shelf_vert":
                continue
            wall_dim = obj["primitives"][0]['dimensions']
            wall_pos = obj["primitive_poses"][0]['position']
        wx_int = wall_dim[0]/2 + dividing_wall_clearance/2
        wy_int = wall_dim[1]/2 + dividing_wall_clearance/2

        wall_xmin = wall_pos[0] - wx_int - self.object_details['size'][0] / 2
        wall_xmax = wall_pos[0] + wx_int + self.object_details['size'][0] / 2
        wall_ymin = wall_pos[1] - wy_int - self.object_details['size'][1] / 2
        wall_ymax = wall_pos[1] + wy_int + self.object_details['size'][1] / 2

        fxmin = max(shelf_xmin, wall_xmin)
        fxmax = min(shelf_xmax, wall_xmax)
        fymin = max(shelf_ymin, wall_ymin)
        fymax = min(shelf_ymax, wall_ymax)

        regions = []

        # If wall doesn't overlap the cage at all, nothing to subtract
        if fxmin >= fxmax or fymin >= fymax:
            regions.append([[shelf_xmin, shelf_xmax], [shelf_ymin, shelf_ymax]])
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
        regions = [r for r in regions if (r[0][1] - r[0][0] > eps) and (r[1][1] - r[1][0] > eps)]
        #print(regions)
        return regions
    
    
    def generate_task_set(self):
        """Generate task set/TSRs"""

        iTSR_dict = {}
        yaw_iTSR_set, _ = find_yaw_iTSR_set(self.object_details, self.problem_details, self.Tw2_w1,) 

        for curr_base_ind in range(len(self.base_zpos)):
            curr_base_zdim = self.base_zdim[curr_base_ind]
            curr_base_zpos = self.base_zpos[curr_base_ind]
            #print(curr_base_zdim)
            #print(curr_base_zpos)
            z_object = curr_base_zpos + (curr_base_zdim/2) + (self.object_details['size'][2]/2)
            object_position = [self.robot_pos[0], self.robot_pos[1], z_object]
            #print(object_position)
            object_type = self.object_details['type']
            object_size = self.object_details['size']
            object_dist = self.object_details['dist']
            self.object_details = {
                'type': object_type,
                'size': object_size,
                'position': object_position,
                'yaw': 0,
                'dist': object_dist
            }
            print(f"Base: {self.base_names[curr_base_ind]}")
            curr_base_iTSR_set, _ = find_iTSR_set(self.object_details, self.problem_details, self.yaw_tw2_w1_dict, yaw_iTSR_set, problem=self.problem, robot_pos=self.robot_pos)
            #print(len(curr_base_iTSR_set[0]))
            iTSR_dict.update(curr_base_iTSR_set[0])

        #iTSR_set, _ = find_iTSR_set(self.object_details, self.problem_details, self.yaw_tw2_w1_dict, yaw_iTSR_set, problem=self.problem, robot_pos=self.robot_pos)
        #iTSR_set = [iTSR_dict]
        #self.task_set = iTSR_set[0]
        self.task_set = iTSR_dict
        return self.task_set


class RealEnv(MujocoEnv):
    """Real environment"""
    def __init__(self, robot, no_sv=False):
        """Initialize real environment"""
        super().__init__(robot, custom_base="lab_scene.xml")
        if robot!="ur10":
            raise NotImplementedError("RealEnv only supports UR10")
        
        # Initialize object dimensions
        self.object_details = {
            'size': [0.045, 0.08], #[r, h]
            'type': "cylinder",
            'yaw': 0
        }
        #print(self.object_details['size'])
        #object_path = self.select_object("mug")
        #object_path = self.select_object("g_cups")
        
        self.robot_pos = [0, 0, 0]
        self.robot_quat = [1, 0, 0, 0]
        self.robot_quat = [0.70710678, 0.0, 0.0, 0.70710678]
        self.robot_quat = [0.0, 0.0, 0.0, 1.0]
        #self.robot_quat = [-0.70710678, 0.0, 0.0, 0.70710678]

        base_pos = [0, 0, 0]
        base_dim = [0, 0, 0]

        self.object_yaw = 0*np.pi #-yaw to +yaw
        self.object_inner_rad = 0.3
        self.object_outer_rad = 1.0

        self.object_details['dist'] = [
            self.object_outer_rad, self.object_outer_rad, 0, self.object_yaw
        ]

        self.object_details['position'] = [
            self.robot_pos[0], self.robot_pos[1], base_pos[2] + (base_dim[2]/2) + self.object_details['size'][1]/2
        ]

        #box_intervals = super().find_problem_intervals(base_name="base", wall_clearance=box_thickness)
        
        # table intervals
        real_intervals = [
            [-0.35, 0.07],
            [-1.02, -0.70]
        ]
        # real_intervals = [
        #     [-1, 1],
        #     [-1, 1]
        # ]
        self.problem = {
            'name': "box",
            'intervals': real_intervals,
            'robot': f"{robot}"   
        }

        # Find TSR parameters
        sv_config = super().initialize_TSR_parameters(robot, grasp_strategy="top")

        #print(self.object_details['size'])
        sv_xml = super().cube_swept_volume_xml(self.object_details['size'], sv_config, cyl=True)
        #sv_xml = self.mjcf_file_to_fragment(object_path)
        xmls_to_add = [sv_xml]
        
        # Extra objects
        #apple_xml = self.mjcf_file_to_fragment("assets/ycb/apple.xml")
        #sugar_box_xml = self.mjcf_file_to_fragment("assets/ycb/sugar_box.xml")
        #a_cups_xml = self.mjcf_file_to_fragment("assets/ycb/a_cups.xml")

        #xmls_to_add.append(apple_xml)
        #xmls_to_add.append(sugar_box_xml)
        #xmls_to_add.append(a_cups_xml)

        # b_cups_xml = self.mjcf_file_to_fragment("assets/ycb/b_cups.xml")
        # xmls_to_add.append(b_cups_xml)

        # c_cups_xml = self.mjcf_file_to_fragment("assets/ycb/c_cups.xml")
        # xmls_to_add.append(c_cups_xml)

        # d_cups_xml = self.mjcf_file_to_fragment("assets/ycb/d_cups.xml")
        # xmls_to_add.append(d_cups_xml)

        # g_cups_xml = self.mjcf_file_to_fragment("assets/ycb/g_cups.xml")
        # xmls_to_add.append(g_cups_xml)

        

        # build wall 1
        wall_1_xml = self.build_primitive_body_xml(
            body_name="wall1",
            prim_type="box",
            pos=(0.32, -0.82, 0.145),
            dims=(0.24, 0.45, 0.29),
            quat_xyzw=(0, 0, 0, 1),
            make_free=False,
        )
        xmls_to_add.append(wall_1_xml)

        # build wall 2
        wall_2_xml = self.build_primitive_body_xml(
            body_name="wall2",
            prim_type="box",
            pos=(-0.055, -0.46, 0.135),
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
            pos=(0.55, -0.80, 0.01),
            dims=(0.21, 0.36, 0.02),
            quat_xyzw=(0, 0, 0, 1),
            make_free=False,
        )
        xmls_to_add.append(packing_1_xml)

        # --- parameters ---
        cx, cy, z_floor = 0.55, -0.80, 0.01
        Lx, Ly, t_floor = 0.21, 0.36, 0.02

        t_wall = 0.02
        h_wall = 0.13  # <-- change this to how tall you want the hollow box

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
            pos=(0, -0.50, 1.1),
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

        #self.randomize_object_positions(["mug", "apple", "sugar_box", "a_cups"], [0.05, 0.05, 0.1, 0.1])
        #self.model, self.data = build_model_with_fragments(free_xml_path, xmls_to_add)
    #def cup_object_xml(self, fixed=False):



    def build_primitive_body_xml(
        self,
        body_name: str,
        geom_name: str | None = None,
        prim_type: str = "box",
        dims: list[float] | tuple[float, ...] = (1.0, 1.0, 1.0),
        pos: list[float] | tuple[float, float, float] = (0.0, 0.0, 0.0),
        quat_xyzw: list[float] | tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
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
                raise ValueError(f"cylinder dims must be (height, radius), got {dims}")
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
            y = np.random.uniform(-0.5,- 0.1)
            z = object_heights[i]/2
            yaw = np.random.uniform(-np.pi, np.pi)

            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            qadr = self.model.jnt_qposadr[jid]
            vadr = self.model.jnt_dofadr[jid]

            half = 0.5 * float(yaw)
            qw = np.cos(half)
            qx = 0.0
            qy = 0.0
            qz = np.sin(half)

            # free joint qpos layout: [x y z qw qx qy qz]
            self.data.qpos[qadr:qadr+7] = [x, y, z, qw, qx, qy, qz]
            self.data.qvel[vadr:vadr+6] = 0.0

            mujoco.mj_forward(self.model, self.data)



    def select_object(self, object_name):
        if object_name == "mug":
            object_path =  "assets/ycb/mug.xml"
            self.object_details = {
                'size': [0.05, 0.052, 0.052],
                'type': "mug",
                'yaw': 0
            }
            self.yaw_buffer = 6*(np.pi/180)
            return object_path
        elif object_name == "a_cups":
            object_path =  "assets/ycb/a_cups.xml"
            self.object_details = {
                'size': [0.05, 0.052, 0.052],
                'type': "a_cups",
                'yaw': 0
            }
            self.yaw_buffer = 6*(np.pi/180)
            return object_path
        elif object_name == "g_cups":
            object_path =  "assets/ycb/g_cups.xml"
            self.object_details = {
                'size': [0.05, 0.052, 0.06],
                'type': "g_cups",
                'yaw': 0
            }
            self.yaw_buffer = 6*(np.pi/180)
            return object_path

    def mjcf_file_to_fragment(self, path: str) -> str:
        """Read a standalone MJCF file and return an MJCF fragment:
        <asset>...</asset> + worldbody contents (bodies).
        """
        text = Path(path).read_text()

        # grab asset block (optional)
        m_asset = re.search(r"<asset\b[^>]*>.*?</asset>", text, flags=re.DOTALL)
        asset = m_asset.group(0) if m_asset else ""

        # grab contents inside <worldbody>...</worldbody>
        m_world = re.search(r"<worldbody\b[^>]*>(.*?)</worldbody>", text, flags=re.DOTALL)
        if not m_world:
            raise ValueError(f"No <worldbody> found in {path}")
        world_contents = m_world.group(1).strip()

        # return fragment: assets + bodies (no <worldbody> wrapper)
        return (asset + "\n\n" + world_contents).strip()


def to_rad(q_degrees):
    q_rad = []
    for q_deg in q_degrees:
        q_rad.append(q_deg*np.pi/180)
    
    return q_rad

if __name__=="__main__":
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
    elif environment == "real":
        env = RealEnv(robot_chosen, no_sv=True)
    else:
        env = FreeEnv(robot_chosen, no_sv=True)

    model, data = env.model, env.data
    task_set = env.generate_task_set()
    env.move_cube_object([-env.object_outer_rad, -env.object_outer_rad, env.object_details['position'][2], 0])    

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

    #task_keys = list(task_set.keys())
    # task_keys = solved_task_keys
    # random_ind = np.random.randint(0, len(task_keys))
    # key = task_keys[random_ind]


    if robot_chosen == "panda":
        robot = Panda(model, data, visualize=True)
    elif robot_chosen == "fetch":
        robot = FetchArm(model, data, visualize=True)
    else:
        robot = UR10(model, data, visualize=True)
    robot.teleport_base(np.array([env.robot_pos]), np.array(env.robot_quat))


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

        camview.lookat = [1.0493371,  0.05479526, 0.52797369]
        camview.distance = 3.895318011088692
        camview.azimuth = 55.705417066155334
        camview.elevation = -40.769175455417056


    elif environment == "cage":
        robot.viewer.cam.lookat[:] = [0.52086037, -0.06213331,  0.49931841]
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
    
    new_home_pos = np.array([117.52, -61.10, 89.46, -119.21, -91.35, 30.14])
    new_home_pos = np.array([116.36, -62.72, 90.52, -118.66, -91.32, 28.99])
    new_home_pos = to_rad(new_home_pos)
    print(new_home_pos)
    robot.set_joint_qpos(new_home_pos)

    #env.move_swept_volume(task_keys[random_ind])
    # sample = [float(np.random.uniform(lo, hi)) for (lo, hi) in key]
    # env.move_cube_object(sample)

    print(f"in contact: {robot.in_contact()}")
    robot.viewer.sync()

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

    # Keep the viewer
    try:
        while True:
            time.sleep(0.01)
    except KeyboardInterrupt:
        robot.close()

