import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import numpy as np
import yaml
import mujoco
import mujoco.viewer

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
    def __init__(self, robot):
        """Initialize object dimensions and common parameters"""
        if robot == "panda":
            self.robot_dir = "assets/franka_emika_panda"
        else:
            self.robot_dir = f"assets/{robot}"
        #self.base_xml = f"{self.robot_dir}/scene.xml"
        self.base_xml = "scene.xml"

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
    
    def build_xml(self, parent_body_name="env_name", skip_ids=None):
        '''Return xml for environment'''
        
        if skip_ids is None:
            skip_ids = set()
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
                # dims = [x, y, z]  -> size = half-dims
                size = [dims[0] / 2.0, dims[1] / 2.0, dims[2] / 2.0]
                mj_type = "box"
                mj_size = size

            elif prim_type == "cylinder":
                # cylinder dims = [height, radius]
                height, radius = dims[0], dims[1]
                mj_type = "cylinder"
                mj_size = [radius, height / 2.0]

            else:
                raise ValueError(f"Unsupported primitive type: {prim_type} for id={obj_id}")

            lines.append(
                f'  <geom name="{obj_id}" type="{mj_type}" '
                f'pos="{self.fmt(pos)}" quat="{self.fmt(quat_wxyz)}" '
                f'size="{self.fmt(mj_size)}" '
                f'contype="1" conaffinity="1" rgba="0.6 0.6 0.6 1"/>'
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

    def build_model(self, xml_path, xmls_to_add):
        """
        Build final xml and model
        xml_path: desired path for model's xml
        xmls_to_add: list of xml strings to add to model, prepared from build_xml()
        """
        
        curr_xml = f"""
        <mujoco model="test_world">
        <include file="{self.base_xml}"/>
        <worldbody>
        """
        for primitive_xml in xmls_to_add:
            curr_xml += f"{primitive_xml}"
    
        curr_xml += """
            </worldbody>
        </mujoco>
        """
        with open(xml_path, "w") as f:
            f.write(curr_xml)
        #print(xml_path)
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

    def cube_swept_volume_xml(self, object_dims, object_configs, fixed=False):
        """Create swept volume xml string"""
        rgba = [0.8, 0.8, 0.8, 1]
        name = "swept_volume"
        joint_xml = "" if fixed else f'<joint name="{name}_free" type="free"/>'
        
        R_cyl, b_pos, b1_size, b2_size, corners = self.compute_sv_params(object_dims, object_configs)

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
                size="{R_cyl} {object_dims[2]/2}"
                rgba="{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}"/>

            <geom name="sv_cyl2" type="cylinder"
                pos="{corners_local[1,0]} {corners_local[1,1]} {corners_local[1,2]}"
                size="{R_cyl} {object_dims[2]/2}"
                rgba="{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}"/>

            <geom name="sv_cyl3" type="cylinder"
                pos="{corners_local[2,0]} {corners_local[2,1]} {corners_local[2,2]}"
                size="{R_cyl} {object_dims[2]/2}"
                rgba="{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}"/>

            <geom name="sv_cyl4" type="cylinder"
                pos="{corners_local[3,0]} {corners_local[3,1]} {corners_local[3,2]}"
                size="{R_cyl} {object_dims[2]/2}"
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
    def __init__(self, robot):
        """Initialize free environment"""
        super().__init__(robot)
        self.robot_pos = [0, 0, 0]
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
        sv_xml = super().cube_swept_volume_xml(self.object_details['size'], sv_config)
        xmls_to_add = [sv_xml]
        free_xml_path = f"{self.robot_dir}/free_scene.xml"
        self.model, self.data = super().build_model(free_xml_path, xmls_to_add)

        
class BoxEnv(MujocoEnv):
    """Box environment"""
    def __init__(self, robot):
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
        sv_xml = super().cube_swept_volume_xml(self.object_details['size'], sv_config)
        box_xml = super().build_xml(parent_body_name="scene_box", skip_ids={"Can1"})
        
        xmls_to_add = [sv_xml, box_xml]
        free_xml_path = f"{self.robot_dir}/box_scene.xml"
        self.model, self.data = super().build_model(free_xml_path, xmls_to_add)


class CageEnv(MujocoEnv):
    """Cage environment"""
    def __init__(self, robot):
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
        sv_xml = super().cube_swept_volume_xml(self.object_details['size'], sv_config)
        cage_xml = super().build_xml(parent_body_name="scene_cage", skip_ids={"Cube1"})
        
        xmls_to_add = [sv_xml, cage_xml]
        free_xml_path = f"{self.robot_dir}/cage_scene.xml"
        self.model, self.data = super().build_model(free_xml_path, xmls_to_add)
    

class TableEnv(MujocoEnv):
    """Table environment"""
    def __init__(self, robot):
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
        sv_xml = super().cube_swept_volume_xml(self.object_details['size'], sv_config)
        table_xml = super().build_xml(parent_body_name="scene_table", skip_ids={"Cube1"})
        
        xmls_to_add = [sv_xml, table_xml]
        free_xml_path = f"{self.robot_dir}/table_scene.xml"
        self.model, self.data = super().build_model(free_xml_path, xmls_to_add)

class ShelfEnv(MujocoEnv):
    """Thin shelf environment"""
    def __init__(self, robot):
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
            self.object_outer_rad = 0.65

        self.object_yaw = 0.25*np.pi #-yaw to +yaw
        self.object_details['dist'] = [
            self.object_outer_rad, self.object_outer_rad, 0, self.object_yaw
        ]
        # Find TSR parameters
        sv_config = super().initialize_TSR_parameters(robot, grasp_strategy="front")

        # Add environment xmls and build model
        sv_xml = super().cube_swept_volume_xml(self.object_details['size'], sv_config)
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



if __name__=="__main__":
    # Load environment and generate task set
    robot = "fetch"
    #env = ShelfEnv(robot)
    env = BoxEnv(robot)
    model, data = env.model, env.data
    task_set = env.generate_task_set()
    print(f"Bins generated: {len(task_set)}")
    # Load robot and test
    
    #robot = Panda(model, visualize=True)
    #robot.set_joint_qpos(
    #    np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
    #)
    

    robot = FetchArm(model, visualize=True)

    robot.teleport_base(np.array(env.robot_pos), np.array(env.robot_quat))

    robot.viewer.sync()

    # Keep the viewer
    try:
        while True:
            time.sleep(0.01)
    except KeyboardInterrupt:
        robot.close()

