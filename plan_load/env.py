# TODO
import os
import sys
from pathlib import Path
import pickle
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R
import mujoco

from plan_load.task_generation import (
    find_iTSR_set,
    find_yaw_iTSR_set,
    panda_TSR_parameters,
)


class MujocoEnv:
    def __init__(self, robot):
        self.env_name = ""
        self.robot_name = robot

        self.model = None
        self.data = None
        self.collision_geoms = None
        self.ee_offset = None

    def generate_task_set(self):
        pass

    def move_swept_volume(self, key):
        move_swept_volume(self.model, self.data, key)


class TableEnv(MujocoEnv):
    def __init__(self, robot):
        super().__init__(robot)

        self.env_name = "table"
        self.scene = build_table_problem(robot)
        self.common = build_common_context("table")

        self.object_details = self.scene["object_details"]
        self.object_details["dist"] = self.common["object_dist"]

        key = self.get_one_task()
        sv_xml, _ = cube_swept_volume_xml(self.object_details["size"], key)

        # Build model and solve problems
        model, data, _ = build_model(
            self.scene["base_xml"],
            self.scene["xml_path"],
            self.scene["xmls_to_add"] + [sv_xml],
        )

        self.model = model
        self.data = data
        self.collision_geoms = self.scene["obj_geom_list"]
        self.ee_offset = self.common["Tew"][0]

    def get_one_task(self):
        pos = np.array(self.object_details["position"])
        dist = np.array(self.object_details["dist"])

        # Pick the top grasp strategy the same way
        yaw_tw2_w1 = np.array(self.common["yaw_tw2_w1"]["top"])

        # First yaw interval per find_yaw_iTSR_set:
        # yaw_1 = -dist[3], yaw_2 = yaw_1 + Tw2_w1[3]
        yaw_1 = float(round(-dist[3], 5))
        yaw_2 = float(round(yaw_1 + float(self.common["Tw2_w1"][3]), 5))

        tw1_0 = [pos[0] - dist[0], pos[1] - dist[1], pos[2] - dist[2]]
        tw2_0 = np.array(tw1_0) + yaw_tw2_w1
        intervals = (
            (round(float(tw1_0[0]), 5), round(float(tw2_0[0]), 5)),
            (round(float(tw1_0[1]), 5), round(float(tw2_0[1]), 5)),
            (round(float(tw1_0[2]), 5), round(float(tw2_0[2]), 5)),
            (round(yaw_1, 5), round(yaw_2, 5)),
        )
        return intervals

    def generate_task_set(self):
        # Build problems
        yaw_iTSR_set, _ = find_yaw_iTSR_set(
            self.object_details,
            self.common["problem_details"],
            self.common["Tw2_w1"],
        )
        iTSR_set, _ = find_iTSR_set(
            self.object_details,
            self.common["problem_details"],
            self.common["yaw_tw2_w1"],
            yaw_iTSR_set,
            problem=self.scene["problem"],
            robot_pos=self.scene["robot_pos"],
        )
        return iTSR_set[0]


class BoxEnv(MujocoEnv):
    def __init__(self, robot):
        pass


class CageEnv(MujocoEnv):
    def __init__(self, robot):
        pass


class ShelfEnv(MujocoEnv):
    def __init__(self, robot):
        pass


# PANDA Relocate
# pos="0.2 0.0 0.8" quat="0.70710678 0.0 0.0 -0.70710678"
def build_common_context(env_name: str):
    # small extra angular slack around yaw limits
    # when generating bounds
    yaw_buffer = 0.1047  # 6.0 * (np.pi / 180.0)
    # a coverage hyperparameter for graph generation
    alpha = 0.95
    # Safety margin for the robot
    robot_clearance = 0.3

    reachable_ws = 0.6 if env_name == "table" else 0.9
    object_dist = [
        reachable_ws,
        reachable_ws,
        0.0,
        0.5 * np.pi,  # args.yaw_dist_rad,
    ]

    object_details = {
        "type": "cube",
        "size": [0.03, 0.03, 0.15],
    }

    tsr_object_details = dict(object_details)
    tsr_object_details["position"] = [0, 0, 0]
    tsr_object_details["yaw"] = 0.0
    tsr_object_details["dist"] = object_dist
    TSR_params = panda_TSR_parameters(tsr_object_details, yaw_buffer, alpha)
    Tew_top, Bw_top, half_side_top, Tw2_w1, yaw_tw2_w1_top = TSR_params["top"]
    Tew_front, _, _, _, yaw_tw2_w1_front = TSR_params["front"]

    problem_details = {
        "top": {
            "alpha": alpha,
            "Bw": Bw_top,
            "half_side": half_side_top,
            "yaw_buffer": yaw_buffer,
            "reachable_ws": reachable_ws,
            "robot_clearance": robot_clearance,
        }
    }

    return {
        "object_dist": object_dist,
        "problem_details": problem_details,
        "Tew": [Tew_top, Tew_front],
        "Tw2_w1": Tw2_w1,
        "yaw_tw2_w1": {
            "top": yaw_tw2_w1_top,
            "front": yaw_tw2_w1_front,
        },
    }


def build_table_problem(robot):
    """Build top problem environment configuration."""
    # small extra angular slack around yaw limits
    # when generating bounds
    yaw_buffer = 0.1047  # 6.0 * (np.pi / 180.0)
    # a coverage hyperparameter for graph generation
    alpha = 0.95
    # Safety margin for the robot
    robot_clearance = 0.3
    reachable_ws = 0.6
    yaw_dist_rad = 0.5 * np.pi

    panda_dir = "assets/franka_emika_panda"
    panda_src = f"{panda_dir}/panda.xml"

    base_xml, xml_path = init_xml(name="free_top_scene")

    object_size = [0.03, 0.03, 0.15]
    object_position = [0.0, 0.0, object_size[2] / 2.0]
    object_quat = [0, 0, 0, 1]
    _, _, object_yaw = R.from_quat(object_quat).as_euler("xyz", degrees=False)

    object_details = {
        "type": "cube",
        "size": object_size,
        "position": object_position,
        "yaw": object_yaw,
    }

    prefix = f"dataset/{robot}_table"

    result = {
        "base_xml": base_xml,
        "xml_path": xml_path,
        "xmls_to_add": [],
        "object_details": object_details,
        "problem": None,
        "robot_pos": [0.0, 0.0, 0.0],
        "prefix": prefix,
        "obj_geom_list": [
            "sv_box1",
            "sv_box2",
            "sv_cyl1",
            "sv_cyl2",
            "sv_cyl3",
            "sv_cyl4",
        ],
    }

    return result


def build_cube_problem(args):
    """Build cube/box problem environment configuration."""
    panda_dir = "assets/franka_emika_panda"
    panda_src = f"{panda_dir}/panda.xml"
    panda_dst = f"{panda_dir}/panda_relocated.xml"

    problem_config_path = "configs/problems/box_panda.yaml"
    problem_scene_path = "configs/scenes/box/scene_box.yaml"

    with open(problem_scene_path, "r") as f:
        scene_data = yaml.safe_load(f)
    objs = scene_data["world"]["collision_objects"]

    base_dim = [0.0, 0.0, 0.0]
    base_pos_scene = [0.0, 0.0, 0.0]
    box_thickness = 0.18

    for obj in objs:
        if obj.get("id", "") != "base":
            continue
        base_dim = obj["primitives"][0]["dimensions"]
        base_pos_scene = obj["primitive_poses"][0]["position"]

    object_size = [args.object_size_x, args.object_size_y, args.object_size_z]
    z_object = base_pos_scene[2] + (base_dim[2] / 2.0) + (object_size[2] / 2.0)

    hx_int = base_dim[0] / 2.0 - box_thickness / 2.0
    hy_int = base_dim[1] / 2.0 - box_thickness / 2.0

    box_xmin = base_pos_scene[0] - hx_int + object_size[0] / 2.0
    box_xmax = base_pos_scene[0] + hx_int - object_size[0] / 2.0
    box_ymin = base_pos_scene[1] - hy_int + object_size[1] / 2.0
    box_ymax = base_pos_scene[1] + hy_int - object_size[1] / 2.0

    problem = {
        "name": "box",
        "intervals": [[box_xmin, box_xmax], [box_ymin, box_ymax]],
    }

    with open(problem_config_path, "r") as f:
        yaml_data = yaml.safe_load(f)

    robot_pos = [
        yaml_data["base_offset"]["position"][0],
        yaml_data["base_offset"]["position"][1],
        yaml_data["base_offset"]["position"][2],
    ]
    robot_quat = [
        yaml_data["base_offset"]["orientation"][3],
        yaml_data["base_offset"]["orientation"][0],
        yaml_data["base_offset"]["orientation"][1],
        yaml_data["base_offset"]["orientation"][2],
    ]

    write_relocated_panda(
        panda_src,
        panda_dst,
        base_pos=robot_pos,
        base_quat_wxyz=robot_quat,
    )

    base_xml, xml_path = init_xml(name="box_scene")
    box_xml = yaml_to_xml(yaml_path="configs/scenes/box/scene_box2.yaml")

    object_quat = [0, 0, 0, 1]
    _, _, object_yaw = R.from_quat(object_quat).as_euler("xyz", degrees=False)
    object_details = {
        "type": "cube",
        "size": object_size,
        "position": [0.0, 0.0, z_object],
        "yaw": object_yaw,
    }

    prefix = (
        f"TSRs/box1/cube_{object_size[0]}_{object_size[1]}_{object_size[2]}/"
        f"{args.robot_clearance}_{args.reachable_ws}_{round(args.yaw_dist_rad, 2)}"
    )

    result = {
        "base_xml": base_xml,
        "xml_path": xml_path,
        "xmls_to_add": [box_xml],
        "object_details": object_details,
        "problem": problem,
        "robot_pos": robot_pos,
        "prefix": prefix,
        "obj_geom_list": [
            "sv_box1",
            "sv_box2",
            "sv_cyl1",
            "sv_cyl2",
            "sv_cyl3",
            "sv_cyl4",
            "base",
            "side_left",
            "side_right",
            "side_front",
            "side_cap",
            "side_back",
        ],
    }

    return result


# Pass a robot name instead
def init_xml(name="test_scene"):
    # Initialize xml from scene.xml

    panda_dir = os.path.abspath("assets/franka_emika_panda")
    xml_path = os.path.join(panda_dir, f"{name}.xml")
    base_xml = "scene.xml"

    return base_xml, xml_path


def compute_sv_params(object_dims, object_configs):
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


def cube_swept_volume_xml(
    object_dims, object_configs, fixed=False, rgba=[0.8, 0.8, 0.8, 1]
):
    name = "swept_volume"
    joint_xml = "" if fixed else f'<joint name="{name}_free" type="free"/>'

    R_cyl, b_pos, b1_size, b2_size, corners = compute_sv_params(
        object_dims, object_configs
    )

    b_pos0 = b_pos[0]  # numpy (3,)
    b1_size0 = b1_size[0].tolist()
    b2_size0 = b2_size[0].tolist()
    corners0 = corners[0]  # (4,3) world
    corners_local = corners0 - b_pos0  # (4,3) local coords

    sv_xml_string = f"""
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
    return sv_xml_string, b_pos0


def move_swept_volume(model, data, object_configs):
    svid = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "swept_volume_free"
    )
    sv_adr = model.jnt_qposadr[svid]
    sv_vadr = model.jnt_dofadr[svid]

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

    data.qpos[sv_adr : sv_adr + 7] = [
        new_pos[0],
        new_pos[1],
        new_pos[2],
        new_quat[0],
        new_quat[1],
        new_quat[2],
        new_quat[3],
    ]
    data.qvel[sv_vadr : sv_vadr + 6] = 0

    mujoco.mj_forward(model, data)


def build_model(base_xml, xml_path, xmls_to_add):
    curr_xml = f"""
    <mujoco model="test_world">
    <include file="{base_xml}"/>
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

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    return model, data, xml_path


def yaml_to_xml(
    yaml_path="configs/scenes/box/scene_box.yaml",
    parent_body_name="scene_box",
    skip_ids={"Can1"},
):
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    def quat_xyzw_to_wxyz(q):
        x, y, z, w = q
        return [w, x, y, z]

    def fmt(v):
        # nice formatting for XML
        return " ".join(f"{x:.6g}" for x in v)

    objs = data["world"]["collision_objects"]
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
        quat_wxyz = quat_xyzw_to_wxyz(quat_xyzw)

        prim_type = prim["type"].lower()
        dims = prim["dimensions"]

        if prim_type == "box":
            # dims = [x, y, z]  -> size = half-dims
            size = [dims[0] / 2.0, dims[1] / 2.0, dims[2] / 2.0]
            mj_type = "box"
            mj_size = size

        elif prim_type == "cylinder":
            # MoveIt cylinder dims = [height, radius]
            height, radius = dims[0], dims[1]
            mj_type = "cylinder"
            mj_size = [radius, height / 2.0]

        else:
            raise ValueError(
                f"Unsupported primitive type: {prim_type} for id={obj_id}"
            )

        lines.append(
            f'  <geom name="{obj_id}" type="{mj_type}" '
            f'pos="{fmt(pos)}" quat="{fmt(quat_wxyz)}" '
            f'size="{fmt(mj_size)}" '
            f'contype="1" conaffinity="1" rgba="0.6 0.6 0.6 1"/>'
        )

    lines.append("</body>")
    return "\n".join(lines)


# This function is no longer needed
import re


def write_relocated_panda(
    panda_src: str,
    panda_dst: str,
    base_pos=(0.0, 0.0, 0.0),
    base_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
    root_body_name="link0",
):
    src = Path(panda_src).read_text()

    pos_str = f"{base_pos[0]} {base_pos[1]} {base_pos[2]}"
    quat_str = f"{base_quat_wxyz[0]} {base_quat_wxyz[1]} {base_quat_wxyz[2]} {base_quat_wxyz[3]}"

    # Match the opening tag of the root body: <body name="link0" ...>
    pattern = rf'(<body\b[^>]*\bname="{re.escape(root_body_name)}"[^>]*)(>)'
    m = re.search(pattern, src)
    if not m:
        raise ValueError(
            f'Could not find <body name="{root_body_name}"> in {panda_src}'
        )

    start_tag = m.group(1)  # everything before the closing '>'
    end = m.group(2)  # '>'

    # Remove existing pos/quat if present
    start_tag = re.sub(r'\spos="[^"]*"', "", start_tag)
    start_tag = re.sub(r'\squat="[^"]*"', "", start_tag)

    # Add pos + quat
    new_start_tag = f'{start_tag} pos="{pos_str}" quat="{quat_str}"{end}'

    out = src[: m.start()] + new_start_tag + src[m.end() :]
    Path(panda_dst).write_text(out)
