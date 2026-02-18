import numpy as np
import torch  # type: ignore


# from planning import omplPlanner
from helpers.helpers import _wrap_pi
from helpers.helpers import to_numpy, quat_angle_diff, quat_mul, quat_mul_t

from scipy.spatial.transform import Rotation as R, Slerp  # type: ignore

from helpers.mujoco_utils import (
    move_swept_volume,
    set_panda_qpos,
    robot_in_contact,
)
from helpers.mujoco_utils import MujocoViewer, mj_fk, mj_ik_multilink


def slerp(q0, q1, alpha):
    # q0, q1 are [w,x,y,z]
    q0 = np.array(q0, dtype=float)
    q1 = np.array(q1, dtype=float)

    # hemisphere fix (shortest path)
    if np.dot(q0, q1) < 0:
        q1 = -q1

    # normalize just in case
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)

    r0 = R.from_quat([q0[1], q0[2], q0[3], q0[0]])  # xyzw
    r1 = R.from_quat([q1[1], q1[2], q1[3], q1[0]])

    s = Slerp([0, 1], R.concatenate([r0, r1]))
    r = s([alpha])[0]  # returns a Rotation array
    q_xyzw = r.as_quat()
    return np.array(
        [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]
    )  # back to wxyz


class MjPrefix:
    def __init__(
        self,
        model,
        data,
        robot_geoms,
        mujoco_viewer,
        path_length,
        min_suffix_factor=0.1,
    ):
        self.path_length = path_length
        self.min_suffix = round(min_suffix_factor * self.path_length)
        self.model = model
        self.data = data
        self.robot_geoms = robot_geoms
        self.viz = mujoco_viewer

    def find_prefix_len(self, root_path, bin_bounds, backoff=20):
        k_max = max(0, self.path_length - self.min_suffix)
        # if self.viz.is_open() is False:
        #    self.viz.open()
        # input("Move sv?")
        move_swept_volume(self.model, self.data, bin_bounds)
        # self.viz.render_state()
        # input("moved sv")
        for i in range(k_max):
            q = np.array(root_path[i], dtype=np.float32, copy=True)
            set_panda_qpos(self.model, self.data, q)

            # Collision check
            in_contact, _ = robot_in_contact(
                self.model, self.data, self.robot_geoms
            )
            if in_contact:
                # print(f"index: {i} - collision")
                return max(0, (i - 1) - backoff)

        return k_max

    def visualize_prefix(self, prefix_qpath):
        if self.viz is not None:
            self.viz.play_qpos_traj(prefix_qpath)


class MjSuffix:
    def __init__(
        self,
        model,
        data,
        robot_geoms,
        obj_geom_list,
        mujoco_viewer,
        path_length,
    ):
        self.path_length = path_length
        self.model = model
        self.data = data
        self.robot_geoms = robot_geoms
        self.viz = mujoco_viewer
        self.obj_geom_list = obj_geom_list

    def ik_suffix(self, root_path, prefix_length, q_goal):

        # input('Visualize prefix end?')
        # set_panda_qpos(self.model, self.data, root_path[prefix_length])
        # self.viz.viewer.sync()
        # input('Visualize bin goal?')
        # set_panda_qpos(self.model, self.data, q_goal)
        # self.viz.viewer.sync()
        # input('Proceed?')

        prefix = root_path[:prefix_length]
        root_goal = root_path[-1]
        root_w = mj_fk(self.model, self.data, root_goal)
        goal_w = mj_fk(self.model, self.data, q_goal)
        # print(f"Goal W: {goal_w}")

        goal_f1_pos = goal_w[0][0]
        goal_f1_quat = goal_w[0][1]
        goal_f2_pos = goal_w[1][0]
        goal_f2_quat = goal_w[1][1]

        root_f1_pos = root_w[0][0]
        root_f1_quat = root_w[0][1]
        root_f2_pos = root_w[1][0]
        root_f2_quat = root_w[1][1]

        delta_f1_pos = np.array(goal_f1_pos) - np.array(root_f1_pos)
        delta_f2_pos = np.array(goal_f2_pos) - np.array(root_f2_pos)

        if np.dot(root_f1_quat, goal_f1_quat) < 0:
            goal_f1_quat = -goal_f1_quat

        if np.dot(root_f2_quat, goal_f2_quat) < 0:
            goal_f2_quat = -goal_f2_quat

        root_f1_quat_conj = np.array(
            [
                root_f1_quat[0],
                -root_f1_quat[1],
                -root_f1_quat[2],
                -root_f1_quat[3],
            ]
        )

        root_f2_quat_conj = np.array(
            [
                root_f2_quat[0],
                -root_f2_quat[1],
                -root_f2_quat[2],
                -root_f2_quat[3],
            ]
        )

        delta_f1_quat = quat_mul(goal_f1_quat, root_f1_quat_conj)
        delta_f2_quat = quat_mul(goal_f2_quat, root_f2_quat_conj)

        delta_w = [
            [delta_f1_pos, delta_f2_pos],
            [delta_f1_quat, delta_f2_quat],
        ]
        # print(delta_w)

        suffix = []

        for i in range(prefix_length, self.path_length):
            # print(i)

            # for i in range(1):
            alpha = (i - prefix_length) / (
                (self.path_length - 1) - prefix_length
            )
            # alpha = 1

            q_root_i = root_path[i]
            # q_root_i = root_goal
            w_root_i = mj_fk(self.model, self.data, q_root_i)

            w_pos_f1_i = w_root_i[0][0]
            w_quat_f1_i = w_root_i[0][1]
            w_pos_f2_i = w_root_i[1][0]
            w_quat_f2_i = w_root_i[1][1]

            recon_f1_pos = np.array(w_pos_f1_i) + alpha * delta_w[0][0]
            recon_f2_pos = np.array(w_pos_f2_i) + alpha * delta_w[0][1]

            w_quat_full_f1 = quat_mul(delta_w[1][0], w_quat_f1_i)
            recon_f1_quat = slerp(w_quat_f1_i, w_quat_full_f1, alpha)

            w_quat_full_f2 = quat_mul(delta_w[1][1], w_quat_f2_i)
            recon_f2_quat = slerp(w_quat_f2_i, w_quat_full_f2, alpha)

            # IK Call
            finger1_target = {
                "pos": recon_f1_pos,
                "quat": recon_f1_quat,
                "name": "left_finger",
            }
            finger2_target = {
                "pos": recon_f2_pos,
                "quat": recon_f2_quat,
                "name": "right_finger",
            }

            link_targets = [finger1_target, finger2_target]
            q_ik = mj_ik_multilink(
                self.model,
                self.data,
                link_targets,
                prefix[-1],
                self.robot_geoms,
                self.obj_geom_list,
            )

            if q_ik is not None:
                suffix.append(q_ik)
            else:
                print("IK returned none")
                return None

        # print(len(suffix))
        return suffix

    def ik_suffix_single(self, root_path, prefix_length, q_goal, ik_solver):

        suffix_length = self.path_length - prefix_length

        seed = (
            root_path[prefix_length - 1] if prefix_length > 0 else root_path[0]
        )
        goal_w = mj_fk(self.model, self.data, q_goal)
        # seed_w = mj_fk(self.model, self.data, seed)

        goal_f1_pos = goal_w[0][0]
        goal_f1_quat = goal_w[0][1]
        goal_f2_pos = goal_w[1][0]
        goal_f2_quat = goal_w[1][1]

        # IK Call
        finger1_target = {
            "pos": goal_f1_pos,
            "quat": goal_f1_quat,
            "name": "left_finger",
        }
        finger2_target = {
            "pos": goal_f2_pos,
            "quat": goal_f2_quat,
            "name": "right_finger",
        }

        link_targets = [finger1_target, finger2_target]
        # link_targets = [finger1_target]

        # q_ik_list = mj_ik_multilink(self.model, self.data, link_targets, seed, self.robot_geoms, self.obj_geom_list, single=True, dt=0.01, iter_count=60)
        q_ik_list = ik_solver.solve(
            link_targets=link_targets, seed=seed, ignore_col=True, single=True
        )

        if q_ik_list is not None:
            suffix = q_ik_list
            # print(len(suffix))
            return suffix
        else:
            # print("IK returned none")
            return None

        # print(len(suffix))
        # return suffix

    def visualize_suffix(self, suffix_qpath):
        if self.viz is not None:
            self.viz.play_qpos_traj(suffix_qpath)


class SparseBoxGrid4D:
    def __init__(self, keys_arr):

        mins = keys_arr[:, :, 0].astype(np.float64)
        maxs = keys_arr[:, :, 1].astype(np.float64)

        mins[:, 3] = _wrap_pi(mins[:, 3])
        maxs[:, 3] = _wrap_pi(maxs[:, 3])

        self.x0 = mins[:, 0].min()
        self.y0 = mins[:, 1].min()
        self.z0 = mins[:, 2].min()

        yaw_mins_sorted = np.sort(mins[:, 3])
        self.yaw0 = yaw_mins_sorted[0]

        """
        def spacing(vals):
            vals = np.sort(np.unique(vals))
            diffs = np.diff(vals)
            diffs = diffs[diffs > 1e-9]
            return diffs.min()
        """

        def spacing(vals, default=1.0):
            vals = np.sort(np.unique(vals))
            diffs = np.diff(vals)
            diffs = diffs[diffs > 1e-9]
            return diffs.min() if diffs.size else default

        self.dx = spacing(mins[:, 0])
        self.dy = spacing(mins[:, 1])

        z_range = np.ptp(mins[:, 2])  # <--- compute ONCE
        self.z_has_variation = z_range > 1e-9
        if self.z_has_variation:
            self.dz = spacing(mins[:, 2])
        else:
            self.dz = 1.0

        self.dyaw = spacing(mins[:, 3])

        yaw_max = mins[:, 3].max()
        self.nyaw = int(round((yaw_max - self.yaw0) / self.dyaw)) + 1

        self.index = {}
        for bin_idx, mn in enumerate(mins):
            x_min, y_min, z_min, yaw_min = mn

            ix = int(round((x_min - self.x0) / self.dx))
            iy = int(round((y_min - self.y0) / self.dy))
            if self.z_has_variation:
                iz = int(round((z_min - self.z0) / self.dz))
            else:
                iz = 0
            iyaw = (
                int(round((_wrap_pi(yaw_min) - self.yaw0) / self.dyaw))
                % self.nyaw
            )

            key = (ix, iy, iz, iyaw)
            self.index[key] = bin_idx

    def key_from_box(self, box):
        """
        box: shape (4,2) array-like with mins/maxs per dim: [[xmin,xmax],[ymin,ymax],[zmin,zmax],[yawmin,yawmax]]
        Returns the discrete grid key (ix,iy,iz,iyaw) used in self.index.
        """
        box = np.asarray(box, dtype=np.float64)
        x_min, y_min, z_min, yaw_min = box[:, 0]
        yaw_min = _wrap_pi(yaw_min)

        ix = int(round((x_min - self.x0) / self.dx))
        iy = int(round((y_min - self.y0) / self.dy))
        iz = (
            int(round((z_min - self.z0) / self.dz))
            if self.z_has_variation
            else 0
        )
        iyaw = int(round((yaw_min - self.yaw0) / self.dyaw)) % self.nyaw

        return (ix, iy, iz, iyaw)

    def query_point(self, x, y, z, yaw):
        yaw = _wrap_pi(yaw)

        ix = int(np.floor((x - self.x0) / self.dx))
        iy = int(np.floor((y - self.y0) / self.dy))
        iz = int(np.floor((z - self.z0) / self.dz)) if self.dz != 1.0 else 0
        iyaw = int(np.floor((yaw - self.yaw0) / self.dyaw)) % self.nyaw

        key = (ix, iy, iz, iyaw)
        return self.index.get(key, None)

    def query_box(self, box):
        key = self.key_from_box(box)
        return self.index.get(key, None)

    def adjacent_neighbors(self, box):
        ix0, iy0, iz0, iyaw0 = self.key_from_box(box)

        out = []
        for di_x in (-1, 0, 1):
            for di_y in (-1, 0, 1):
                for di_yaw in (-1, 0, 1):
                    if di_x == 0 and di_y == 0 and di_yaw == 0:
                        continue

                    iyaw = (iyaw0 + di_yaw) % self.nyaw
                    key = (ix0 + di_x, iy0 + di_y, iz0, iyaw)
                    idx = self.index.get(key, None)

                    if idx is not None:
                        out.append(idx)

        return out

    def neighbors_by_box(self, box, R_meters=0.1, R_yaw=None):

        Rx = int(np.ceil(R_meters / self.dx))
        Ry = int(np.ceil(R_meters / self.dy))

        if R_yaw is None:
            Ryaw = 0
        else:
            Ryaw = int(np.ceil(R_yaw / self.dyaw))

        ix0, iy0, iz0, iyaw0 = self.key_from_box(box)

        out = []
        R2 = R_meters * R_meters

        for dix in range(-Rx, Rx + 1):
            dx_m = dix * self.dx
            dx2 = dx_m * dx_m

            for diy in range(-Ry, Ry + 1):
                dy_m = diy * self.dy
                dy2 = dy_m * dy_m

                if dx2 + dy2 > R2:
                    continue

                for diyaw in range(-Ryaw, Ryaw + 1):
                    iyaw = (iyaw0 + diyaw) % self.nyaw
                    key = (ix0 + dix, iy0 + diy, iz0, iyaw)
                    idx = self.index.get(key, None)
                    if idx is not None:
                        out.append(idx)

        return out
