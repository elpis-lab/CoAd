"""Continuous cell enclosures, shared by MuJoCo and VAMP as primitives.

Each part is an oriented box inflated by an analytic displacement bound.
No finite pose sampling or mesh boolean union is used as a collision proof.
Enclosures are placed and sized from the *actual key*, including clipped bins.
"""

import numpy as np
import mujoco
from coad.tcr import chord, yaw_rotation


def rotated_half_extents(size, half_angle):
    """Exact axis extents of a centered box over a symmetric yaw interval."""
    a, b, z = np.asarray(size, dtype=float) / 2
    u = min(float(half_angle), np.pi / 2)
    tx, ty = min(u, np.arctan2(b, a)), min(u, np.arctan2(a, b))
    return np.array(
        [a * np.cos(tx) + b * np.sin(tx), b * np.cos(ty) + a * np.sin(ty), z]
    )


def enclosed_box(
    size, halfwidths, center_yaw, center_door=0.0, hinge=None, offset=None
):
    """Return body-local position, rotation and enclosing box half extents.

    The part center moves by at most chord(L,u)+chord(rho,v).
    Rotated vertex extents are bounded separately and inflated by that disk.
    Translation is transformed from world grid axes into the box frame.
    """
    size = np.asarray(size, dtype=float)
    h = np.asarray(halfwidths, dtype=float)
    if hinge is None:
        position, angle = np.zeros(3), 0.0
        motion = 0.0
        angular_extent = h[3]
    else:
        hinge, offset = np.asarray(hinge), np.asarray(offset)
        angle = -center_door
        position = hinge + yaw_rotation(angle) @ offset
        rho = np.linalg.norm(offset[:2])
        L = np.linalg.norm(hinge[:2]) + rho
        motion = chord(L, h[3]) + chord(rho, h[4])
        angular_extent = h[3] + h[4]
    rotation = yaw_rotation(angle)
    world_rotation = yaw_rotation(center_yaw + angle)
    translation = np.abs(world_rotation[:2, :2].T) @ h[:2]
    half_extents = (
        rotated_half_extents(size, angular_extent) + np.r_[translation + motion, h[2]]
    )
    return position, rotation, half_extents


class SweptVolumes:
    def create_swept_volume(
        self, tcr_intervals, obj_size=None, sv_count=0, fixed=False
    ):
        if obj_size is None:
            obj_size = self.object_details["size"]
        if not hasattr(self, "_cell_enclosures"):
            self._cell_enclosures = {}
        body = f"swept_volume_{sv_count}"
        kind = self.object_details["type"]
        if kind == "microwave":
            # object_xml initializes the exact hinge-frame geometry.
            if "hinge_pos_body" not in self.object_details:
                self.object_xml(obj_size, [0, 0, 0, 0], temp=True)
            c = np.asarray(self.object_details["hinge_pos_body"])
            d = np.asarray(self.object_details["door_origin_closed_body"])
            hp = np.asarray(self.object_details["handle_pos_door"])
            parts = [
                ("microwave_body_sv", "box", obj_size, None, None),
                ("microwave_door_sv", "box", self.object_details["door_size"], c, d),
                (
                    "microwave_handle_sv",
                    "box",
                    self.object_details["handle_size"],
                    c,
                    d + hp,
                ),
            ]
        else:
            parts = [(f"sv_mesh_{sv_count}", kind, obj_size, None, None)]
        self._cell_enclosures[body] = parts
        geoms = []
        for name, part_kind, size, hinge, offset in parts:
            if part_kind == "cylinder":
                geom_type, initial_size = "cylinder", [size[0], size[1] / 2]
            else:
                geom_type, initial_size = "box", np.asarray(size) / 2
            position = np.zeros(3) if hinge is None else hinge + offset
            geoms.append(
                f'<geom name="{name}" type="{geom_type}" '
                f'size="{" ".join(map(str, initial_size))}" '
                f'pos="{" ".join(map(str, position))}" rgba="0.8 0.8 0.8 1"/>'
            )
        joint = "" if fixed else f'<freejoint name="{body}_free"/>'
        return f'<body name="{body}">{joint}{"".join(geoms)}</body>'

    def move_swept_volume(self, object_configs):
        """Enclose the complete cell, with one separate grid per stable face."""
        if self.environment_name == "allstable":
            face = object_configs[0]
            faces = ["xy", "yz", "zx"]
            if face not in faces:
                raise ValueError(f"Unknown contact face: {face}")
            active = f"swept_volume_{faces.index(face)}"
            key = object_configs[1:]
        else:
            active, key = "swept_volume_0", object_configs
        key = np.asarray(key, dtype=float)
        if key.ndim == 3 and len(key) == 1:
            key = key[0]
        if (
            key.shape not in {(4, 2), (5, 2)}
            or not np.isfinite(key).all()
            or np.any(key[:, 1] < key[:, 0])
        ):
            raise ValueError("Expected ordered finite x/y/z/yaw[/door] cell bounds")
        self.current_tcr_key = object_configs
        center = key.mean(axis=1)
        half = (key[:, 1] - key[:, 0]) / 2
        # The compiled BVH describes the original sizes/positions. Disable it
        # because these primitives change per cell; broadphase uses rbound below.
        self.model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_MIDPHASE)
        for body, parts in self._cell_enclosures.items():
            joint = self.model.joint(f"{body}_free")
            adr, vadr = joint.qposadr[0], joint.dofadr[0]
            if body != active:
                self.data.qpos[adr : adr + 7] = [100, 100, 100, 1, 0, 0, 0]
                self.data.qvel[vadr : vadr + 6] = 0
                continue
            self.data.qpos[adr : adr + 7] = [
                *center[:3],
                np.cos(center[3] / 2),
                0,
                0,
                np.sin(center[3] / 2),
            ]
            self.data.qvel[vadr : vadr + 6] = 0
            for name, kind, size, hinge, offset in parts:
                gid = self.model.geom(name).id
                if kind == "cylinder":
                    radius = float(size[0]) + np.linalg.norm(half[:2])
                    height = float(size[1]) / 2 + half[2]
                    self.model.geom_size[gid] = [radius, height, 0]
                    position, rotation = np.zeros(3), np.eye(3)
                    rbound = np.hypot(radius, height)
                else:
                    door = center[4] if len(center) == 5 else 0.0
                    position, rotation, extents = enclosed_box(
                        size, half, center[3], door, hinge, offset
                    )
                    self.model.geom_size[gid] = extents
                    rbound = np.linalg.norm(extents)
                # The compiler may mark initially aligned geoms as SAMEFRAME
                # or SAMEROT; changing geom_quat must invalidate that shortcut.
                self.model.geom_sameframe[gid] = 0
                self.model.geom_pos[gid] = position
                mujoco.mju_mat2Quat(self.model.geom_quat[gid], rotation.ravel())
                self.model.geom_rbound[gid] = rbound
        mujoco.mj_forward(self.model, self.data)
