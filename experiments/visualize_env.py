"""View any named environment or scene YAML, optionally saving a transparent PNG."""

import argparse
import os
import sys
import time
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from experiments.common import ENVS, ROBOTS, positive_int

SCENES = {
    "table": "table/scene_table.yaml",
    "box": "box/scene_box.yaml",
    "cage": "cage/scene_cage.yaml",
    "shelf": "bookshelf/scene_thin.yaml",
    "conveyor": "conveyor/scene_conveyor.yaml",
    "largeobj": "table/scene_empty_table.yaml",
    "microwave": "table/scene_empty_table.yaml",
    "allstable": "table/scene_empty_table.yaml",
}


def scene_color(name):
    """Shared scene palette for visualization and live conveyor execution."""
    if name.startswith("box_"):
        color = "0.9 0.6 0.2 1"
    elif "splitter" in name:
        color = "0.5 0.5 0.5 1"
    elif name in (
        "conveyor_top",
        "conveyor_end1",
        "conveyor_end2",
    ):
        color = "0.15 0.18 0.2 1"
    elif name.startswith("table_"):
        color = "0.6 0.45 0.3 1"
    else:
        color = "0.5 0.55 0.6 1"
    return color


def _yaml_scene_root(scene_yaml):
    """Build XML using full box lengths, [height, radius] cylinders, XYZW poses."""
    with Path(scene_yaml).open() as stream:
        scene = yaml.safe_load(stream)

    root = ET.Element("mujoco", model="yaml_environment")
    world = ET.SubElement(root, "worldbody")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(
        visual,
        "headlight",
        diffuse="0.6 0.6 0.6",
        ambient="0.3 0.3 0.3",
        specular="0 0 0",
    )
    ET.SubElement(visual, "rgba", haze="0.15 0.25 0.35 1")
    ET.SubElement(
        visual,
        "global",
        azimuth="-130",
        elevation="-20",
        offwidth="1600",
        offheight="1200",
    )
    asset = ET.SubElement(root, "asset")
    ET.SubElement(
        asset,
        "texture",
        name="debug_ground_texture",
        type="2d",
        builtin="checker",
        mark="edge",
        rgb1="0.2 0.3 0.4",
        rgb2="0.1 0.2 0.3",
        markrgb="0.8 0.8 0.8",
        width="300",
        height="300",
    )
    ET.SubElement(
        asset,
        "material",
        name="debug_ground_material",
        texture="debug_ground_texture",
        texuniform="true",
        texrepeat="5 5",
        reflectance="0.2",
    )
    ET.SubElement(
        world,
        "geom",
        name="debug_floor",
        type="plane",
        size="20 20 0.1",
        material="debug_ground_material",
    )

    def values(items):
        return " ".join(str(value) for value in items)

    for obj in scene["world"]["collision_objects"]:
        name = obj["id"]
        primitives, poses = obj["primitives"], obj["primitive_poses"]
        if len(primitives) != len(poses):
            raise ValueError(f"{name}: each primitive needs a pose")
        color = scene_color(name)
        for index, (primitive, pose) in enumerate(zip(primitives, poses)):
            kind = primitive["type"].lower()
            dims = primitive["dimensions"]
            expected = {"box": 3, "cylinder": 2}.get(kind)
            if expected is None:
                raise ValueError(f"{name}: unsupported primitive {kind!r}")
            if len(dims) != expected or any(d <= 0 for d in dims):
                raise ValueError(f"{name}: invalid {kind} dimensions {dims}")
            size = [d / 2 for d in dims] if kind == "box" else [dims[1], dims[0] / 2]
            x, y, z, w = pose["orientation"]
            ET.SubElement(
                world,
                "geom",
                name=name if len(primitives) == 1 else f"{name}_{index}",
                type=kind,
                size=values(size),
                pos=values(pose["position"]),
                quat=values([w, x, y, z]),
                rgba=color,
            )

    return root


def add_camera_args(parser):
    parser.add_argument("--azimuth", type=float, default=120)
    parser.add_argument("--elevation", type=float, default=-20)
    parser.add_argument("--distance", type=float)
    parser.add_argument("--lookat", nargs=3, type=float, metavar=("X", "Y", "Z"))


def set_camera(camera, model, args, data=None):
    center, distance = model.stat.center, max(1, model.stat.extent * 1.5)
    if data is not None:
        visible = (
            (model.geom_type != mujoco.mjtGeom.mjGEOM_PLANE)
            & (model.geom_rgba[:, 3] > 0)
            & (model.geom_group != 3)
        )
        if np.any(visible):
            radii = model.geom_rbound[visible, None]
            lower = np.min(data.geom_xpos[visible] - radii, axis=0)
            upper = np.max(data.geom_xpos[visible] + radii, axis=0)
            center = (lower + upper) / 2
            distance = max(1, np.linalg.norm(upper - lower) * 1.1)
    camera.lookat[:] = args.lookat if args.lookat is not None else center
    camera.distance = args.distance if args.distance is not None else distance
    camera.azimuth = args.azimuth
    camera.elevation = args.elevation


def build_scene(args):
    """Use the planning scene when configured; raw YAML also permits arbitrary robot placement."""
    from coad.env import MujocoEnv
    from coad.robot import Panda, FetchArm, UR10
    from coad.utils import load_env_and_robot

    if args.robot != "none" and args.scene is None:
        # The real lab is UR10-only and conveyor planning has Panda/Fetch home poses.
        configured = (args.env != "real" or args.robot == "ur10") and (
            args.env != "conveyor" or args.robot in ("panda", "fetch")
        )
        if configured:
            env, robot = load_env_and_robot(
                args.env,
                args.robot,
                visualize=False,
                using_swept_volume=False,
                compute_tcr=False,
            )
            if args.base_position is not None or args.base_quat is not None:
                robot.teleport_base(
                    args.base_position or env.env_details["robot_pos"],
                    args.base_quat or env.env_details["robot_quat"],
                )
            return env.model, env.data, robot
    if args.env == "real" and args.scene is None:
        # The lab mesh can be viewed with any robot even though its planning
        # problem is calibrated for UR10. Use explicit base overrides as needed.
        lab_dir = REPO / "assets/ur10"
        root = ET.parse(lab_dir / "lab_scene.xml").getroot()
        for child in list(root):
            if child.tag in ("include", "compiler"):
                root.remove(child)
        for element in root.iter():
            if "file" in element.attrib:
                element.set("file", str(lab_dir / element.get("file")))
            if element.get("name") == "floor":
                element.set("name", "lab_floor")
        if args.robot == "none":
            model = mujoco.MjModel.from_xml_string(
                ET.tostring(root, encoding="unicode")
            )
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            return model, data, None
        env = MujocoEnv(args.robot)
        fragments = [
            ET.tostring(root.find("asset"), encoding="unicode")
            + "".join(
                ET.tostring(child, encoding="unicode")
                for child in root.find("worldbody")
            )
        ]
        model, data = env.build_model(
            f"{env.robot_dir}/visualization_scene.xml", fragments
        )
        robot = {"panda": Panda, "fetch": FetchArm, "ur10": UR10}[args.robot](
            model, data, False
        )
        robot.teleport_base(
            args.base_position or [0, 0, 0], args.base_quat or [1, 0, 0, 0]
        )
        return model, data, robot
    scene = args.scene or (
        REPO / "configs/scenes" / SCENES[args.env] if args.env in SCENES else None
    )
    if args.robot == "none":
        root = (
            _yaml_scene_root(scene)
            if scene
            else ET.fromstring(
                '<mujoco><worldbody><light pos="0 0 3"/><geom name="floor" type="plane" size="3 3 .1"/></worldbody></mujoco>'
            )
        )
        model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        return model, data, None
    env = MujocoEnv(args.robot)
    env.env_details = {"collision_geoms": []}
    fragments = [env.build_xml(scene)] if scene else []
    model, data = env.build_model(f"{env.robot_dir}/visualization_scene.xml", fragments)
    robot = {"panda": Panda, "fetch": FetchArm, "ur10": UR10}[args.robot](
        model, data, False
    )
    robot.teleport_base(args.base_position or [0, 0, 0], args.base_quat or [1, 0, 0, 0])
    return model, data, robot


def render_png(model, data, args):
    from PIL import Image

    model.vis.global_.offwidth = max(args.width, model.vis.global_.offwidth)
    model.vis.global_.offheight = max(args.height, model.vis.global_.offheight)
    camera = mujoco.MjvCamera()
    set_camera(camera, model, args, data)
    options = mujoco.MjvOption()
    options.geomgroup[3] = False  # Robot collision proxies overlap the visual meshes.
    floor_ids = np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
    original_groups = model.geom_group.copy()
    if not args.keep_floor:
        model.geom_group[floor_ids] = 5
        options.geomgroup[5] = False
    try:
        with mujoco.Renderer(model, height=args.height, width=args.width) as renderer:
            renderer.update_scene(data, camera=camera, scene_option=options)
            rgb = renderer.render().copy()
            renderer.enable_segmentation_rendering()
            renderer.update_scene(data, camera=camera, scene_option=options)
            segmentation = renderer.render()
            alpha = np.where(segmentation[..., 0] == -1, 0, 255).astype(np.uint8)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.dstack([rgb, alpha])).save(args.output)
    finally:
        model.geom_group[:] = original_groups
    print(f"Saved {args.output}")


def main(args):
    model, data, robot = build_scene(args)
    if args.environment_color is not None:
        excluded = set()
        if robot is not None:
            root_id = model.body(robot.root_link).id
            excluded.add(root_id)
            for i in range(root_id + 1, model.nbody):
                if model.body_parentid[i] in excluded:
                    excluded.add(i)
        ids = ~np.isin(model.geom_bodyid, list(excluded))
        model.geom_matid[ids] = -1
        model.geom_rgba[ids] = args.environment_color
    try:
        if args.output:
            render_png(model, data, args)
        if not args.output or args.show:
            from mujoco import viewer as mj_viewer

            with mj_viewer.launch_passive(model, data) as viewer:
                set_camera(viewer.cam, model, args, data)
                viewer.opt.geomgroup[3] = False
                while viewer.is_running():
                    viewer.sync()
                    time.sleep(1 / 60)
    finally:
        if robot is not None:
            robot.close()


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=ENVS, default="table")
    parser.add_argument("--robot", choices=(*ROBOTS, "none"), default="panda")
    parser.add_argument(
        "--scene",
        type=Path,
        help="Custom collision-object YAML; robot starts at the origin unless --base-position is supplied",
    )
    parser.add_argument("--base-position", nargs=3, type=float)
    parser.add_argument(
        "--base-quat", nargs=4, type=float, help="Base orientation in WXYZ order"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Save a transparent PNG instead of opening the viewer",
    )
    parser.add_argument(
        "--show", action="store_true", help="Also open the viewer when saving a PNG"
    )
    parser.add_argument(
        "--keep-floor",
        action="store_true",
        help="Include floor planes in the screenshot",
    )
    parser.add_argument("--width", type=positive_int, default=1200)
    parser.add_argument("--height", type=positive_int, default=900)
    parser.add_argument(
        "--environment-color", nargs=4, type=float, metavar=("R", "G", "B", "A")
    )
    add_camera_args(parser)
    args = parser.parse_args()
    if args.distance is not None and args.distance <= 0:
        parser.error("--distance must be positive")
    return args


if __name__ == "__main__":
    args = parse_arguments()
    args.scene = args.scene.resolve() if args.scene else None
    args.output = args.output.resolve() if args.output else None
    os.chdir(REPO)
    main(args)
