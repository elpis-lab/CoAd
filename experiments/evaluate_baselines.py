import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pickle
import mujoco
import time
import argparse
from scipy.spatial import cKDTree

from tqdm import tqdm
from coad.env import MujocoEnv
from coad.robot import MujocoRobot
from coad.mink_ik import get_ik_solver

from coad.adaptation import LinearAdapter, GRRAdapter
from coad.adaptation import DMPAdapter, TrajOptAdapter

# from coad.task_space import deep_tuple
from experiments.visualize_paths import traj_len
from coad.utils import set_seed, load_env_and_robot, get_data_folder
from coad.planning import OMPLPlanner, euclidean_path_length

from coad.planning import VAMPPlanner

from coad.env import FreeEnv, CageEnv, BoxEnv, TableEnv, ShelfEnv, LargeObjectEnv, MicrowaveEnv, AllStableEnv, RealEnv
from coad.robot import Panda, UR10, FetchArm


folder1 = "dataset/top_naive"
folder2 = "dataset/top"

# class BoxGrid:
#     FACE_VALUES = ("xy", "yz", "zx")

#     def __init__(self, keys_to_root, tol=1e-12):
#         self.keys_list = list(keys_to_root.keys())
#         self.tol = tol

#         if not self.keys_list:
#             raise ValueError("keys_to_root cannot be empty")

#         first_key = self.keys_list[0]

#         # AllStable:
#         #     (face, x, y, z, yaw)
#         #
#         # Standard:
#         #     (x, y, z, yaw)
#         #
#         # Microwave:
#         #     (x, y, z, yaw, door)
#         self.has_face = (
#             len(first_key) == 5
#             and isinstance(first_key[0], str)
#         )

#         if self.has_face:
#             if first_key[0] not in self.FACE_VALUES:
#                 raise ValueError(
#                     f"Unknown AllStable face value: {first_key[0]}"
#                 )

#             self.numeric_offset = 1
#             self.has_door = False
#             self.ndim = 5

#             numeric_keys = [
#                 key[1:]
#                 for key in self.keys_list
#             ]
#         else:
#             self.numeric_offset = 0
#             self.ndim = len(first_key)
#             self.has_door = self.ndim == 5

#             numeric_keys = self.keys_list

#         if self.ndim not in (4, 5):
#             raise ValueError(
#                 f"Expected 4D or 5D keys, got {self.ndim} dimensions"
#             )

#         self.keys_arr = np.asarray(
#             numeric_keys,
#             dtype=np.float64,
#         )

#         keys = self.keys_arr

#         if keys.ndim != 3 or keys.shape[2] != 2:
#             raise ValueError(
#                 f"Expected numeric key shape (N,D,2), got {keys.shape}"
#             )

#         # Numeric dimensions are always:
#         #     x, y, z, yaw
#         #
#         # and optionally:
#         #     door
#         mins = keys[:, :, 0].copy()
#         maxs = keys[:, :, 1].copy()

#         # Wrap yaw into [-pi, pi).
#         mins[:, 3] = self._wrap_pi(mins[:, 3])
#         maxs[:, 3] = self._wrap_pi(maxs[:, 3])

#         # ---------- FACE ----------
#         if self.has_face:
#             self.face_values = tuple(
#                 face
#                 for face in self.FACE_VALUES
#                 if any(key[0] == face for key in self.keys_list)
#             )

#             self._face_to_iface = {
#                 face: index
#                 for index, face in enumerate(self.face_values)
#             }

#         # ---------- GLOBAL BOUNDS ----------
#         self.x_global_min = float(np.min(mins[:, 0]))
#         self.x_global_max = float(np.max(maxs[:, 0]))

#         self.y_global_min = float(np.min(mins[:, 1]))
#         self.y_global_max = float(np.max(maxs[:, 1]))

#         if self.has_door:
#             self.door_global_min = float(np.min(mins[:, 4]))
#             self.door_global_max = float(np.max(maxs[:, 4]))

#         # ---------- X/Y BIN STARTS ----------
#         self.x_mins = np.sort(np.unique(mins[:, 0]))
#         self.y_mins = np.sort(np.unique(mins[:, 1]))

#         self._x_to_ix = {
#             float(value): index
#             for index, value in enumerate(self.x_mins)
#         }

#         self._y_to_iy = {
#             float(value): index
#             for index, value in enumerate(self.y_mins)
#         }

#         self.nx = len(self.x_mins)
#         self.ny = len(self.y_mins)

#         # ---------- Z DISCRETE LEVELS ----------
#         self.z_values = np.sort(np.unique(mins[:, 2]))
#         self.nz = len(self.z_values)

#         # ---------- YAW BIN STARTS ----------
#         self.yaw_mins = np.sort(np.unique(mins[:, 3]))
#         self.nyaw = len(self.yaw_mins)

#         # ---------- DOOR BIN STARTS ----------
#         if self.has_door:
#             self.door_mins = np.sort(np.unique(mins[:, 4]))
#             self.ndoor = len(self.door_mins)

#             self._door_to_idoor = {
#                 float(value): index
#                 for index, value in enumerate(self.door_mins)
#             }

#         # ---------- BUILD INDEX ----------
#         self.index = {}

#         for bin_idx, key in enumerate(self.keys_list):
#             if self.has_face:
#                 face = key[0]
#                 numeric_key = key[1:]

#                 if face not in self._face_to_iface:
#                     raise ValueError(
#                         f"Unknown AllStable face value: {face}"
#                     )

#                 iface = self._face_to_iface[face]
#             else:
#                 numeric_key = key

#             x_min = float(numeric_key[0][0])
#             y_min = float(numeric_key[1][0])
#             z_val = float(numeric_key[2][0])
#             yaw_min = float(
#                 self._wrap_pi(numeric_key[3][0])
#             )

#             ix = self._x_to_ix[x_min]
#             iy = self._y_to_iy[y_min]

#             iz = int(np.argmin(
#                 np.abs(self.z_values - z_val)
#             ))

#             iyaw = int(np.argmin(
#                 np.abs(self.yaw_mins - yaw_min)
#             ))

#             if not np.isclose(
#                 self.yaw_mins[iyaw],
#                 yaw_min,
#                 atol=self.tol,
#                 rtol=0.0,
#             ):
#                 raise RuntimeError(
#                     f"Could not match yaw minimum "
#                     f"{yaw_min:.17f}"
#                 )

#             if self.has_face:
#                 indices = (
#                     iface,
#                     ix,
#                     iy,
#                     iz,
#                     iyaw,
#                 )
#             elif self.has_door:
#                 door_min = float(numeric_key[4][0])
#                 idoor = self._door_to_idoor[door_min]

#                 indices = (
#                     ix,
#                     iy,
#                     iz,
#                     iyaw,
#                     idoor,
#                 )
#             else:
#                 indices = (
#                     ix,
#                     iy,
#                     iz,
#                     iyaw,
#                 )

#             if indices in self.index:
#                 previous_key = self.keys_list[
#                     self.index[indices]
#                 ]

#                 raise RuntimeError(
#                     f"Duplicate bin index {indices}: "
#                     f"{previous_key} and {key}"
#                 )

#             self.index[indices] = bin_idx

#     def _bin_indices(self, sample):
#         if len(sample) != self.ndim:
#             raise ValueError(
#                 f"Expected sample with {self.ndim} values, "
#                 f"got {len(sample)}: {sample}"
#             )

#         if self.has_face:
#             face = sample[0]

#             if face not in self._face_to_iface:
#                 return None

#             iface = self._face_to_iface[face]
#             numeric_sample = sample[1:]
#         else:
#             numeric_sample = sample

#         x = float(numeric_sample[0])
#         y = float(numeric_sample[1])
#         z = float(numeric_sample[2])
#         yaw = float(
#             self._wrap_pi(numeric_sample[3])
#         )

#         # ---------- X ----------
#         if (
#             x < self.x_global_min - self.tol
#             or x >= self.x_global_max
#         ):
#             return None

#         ix = int(
#             np.searchsorted(
#                 self.x_mins,
#                 x,
#                 side="right",
#             ) - 1
#         )

#         if ix < 0 or ix >= self.nx:
#             return None

#         # ---------- Y ----------
#         if (
#             y < self.y_global_min - self.tol
#             or y >= self.y_global_max
#         ):
#             return None

#         iy = int(
#             np.searchsorted(
#                 self.y_mins,
#                 y,
#                 side="right",
#             ) - 1
#         )

#         if iy < 0 or iy >= self.ny:
#             return None

#         # ---------- Z ----------
#         iz = int(np.argmin(
#             np.abs(self.z_values - z)
#         ))

#         # ---------- YAW ----------
#         iyaw = int(
#             np.searchsorted(
#                 self.yaw_mins,
#                 yaw,
#                 side="right",
#             ) - 1
#         )

#         if iyaw < 0 or iyaw >= self.nyaw:
#             return None

#         if self.has_face:
#             return (
#                 iface,
#                 ix,
#                 iy,
#                 iz,
#                 iyaw,
#             )

#         # ---------- DOOR ----------
#         if self.has_door:
#             door = float(numeric_sample[4])

#             if (
#                 door < self.door_global_min - self.tol
#                 or door >= self.door_global_max
#             ):
#                 return None

#             idoor = int(
#                 np.searchsorted(
#                     self.door_mins,
#                     door,
#                     side="right",
#                 ) - 1
#             )

#             if idoor < 0 or idoor >= self.ndoor:
#                 return None

#             return (
#                 ix,
#                 iy,
#                 iz,
#                 iyaw,
#                 idoor,
#             )

#         return (
#             ix,
#             iy,
#             iz,
#             iyaw,
#         )

#     def query_point(self, sample):
#         indices = self._bin_indices(sample)

#         if indices is None:
#             return None

#         bin_idx = self.index.get(indices)

#         if bin_idx is None:
#             return None

#         key = self.keys_list[bin_idx]

#         if self.has_face:
#             if sample[0] != key[0]:
#                 return None

#             numeric_sample = sample[1:]
#             numeric_key = key[1:]
#         else:
#             numeric_sample = sample
#             numeric_key = key

#         if not self._inside_nonperiodic(
#             numeric_sample[0],
#             numeric_key[0][0],
#             numeric_key[0][1],
#         ):
#             return None

#         if not self._inside_nonperiodic(
#             numeric_sample[1],
#             numeric_key[1][0],
#             numeric_key[1][1],
#         ):
#             return None

#         if not self._inside_yaw(
#             numeric_sample[3],
#             numeric_key[3][0],
#             numeric_key[3][1],
#         ):
#             return None

#         if self.has_door:
#             if not self._inside_nonperiodic(
#                 numeric_sample[4],
#                 numeric_key[4][0],
#                 numeric_key[4][1],
#             ):
#                 return None

#         return key

#     def _inside_nonperiodic(self, value, lo, hi):
#         value = float(value)
#         lo = float(lo)
#         hi = float(hi)

#         return (
#             value >= lo - self.tol
#             and value < hi
#         )

#     def _inside_yaw(self, value, lo, hi):
#         value = float(self._wrap_pi(value))
#         lo = float(self._wrap_pi(lo))
#         hi = float(self._wrap_pi(hi))

#         if lo <= hi:
#             return (
#                 value >= lo - self.tol
#                 and value < hi
#             )

#         return (
#             value >= lo - self.tol
#             or value < hi
#         )

#     @staticmethod
#     def _wrap_pi(angle):
#         return (
#             np.asarray(angle) + np.pi
#         ) % (2.0 * np.pi) - np.pi

import numpy as np

class BoxGrid:
    FACE_VALUES = ("xy", "yz", "zx")

    def __init__(self, keys_to_root, tol=1e-12):
        self.keys_list = list(keys_to_root.keys())
        self.tol = tol

        if not self.keys_list:
            raise ValueError("keys_to_root cannot be empty")

        first_key = self.keys_list[0]

        # AllStable:
        #     (face, x, y, z, yaw)
        #
        # Standard:
        #     (x, y, z, yaw)
        #
        # Microwave:
        #     (x, y, z, yaw, door)
        self.has_face = (
            len(first_key) == 5
            and isinstance(first_key[0], str)
        )

        if self.has_face:
            self.numeric_offset = 1
            self.has_door = False
            self.ndim = 5

            for key in self.keys_list:
                if (
                    len(key) != 5
                    or not isinstance(key[0], str)
                    or key[0] not in self.FACE_VALUES
                ):
                    raise ValueError(
                        f"Invalid AllStable key: {key}"
                    )

            numeric_keys = [key[1:] for key in self.keys_list]
        else:
            self.numeric_offset = 0
            self.ndim = len(first_key)
            self.has_door = self.ndim == 5
            numeric_keys = self.keys_list

        if self.ndim not in (4, 5):
            raise ValueError(
                f"Expected 4D or 5D keys, got {self.ndim} dimensions"
            )

        self.keys_arr = np.asarray(
            numeric_keys,
            dtype=np.float64,
        )

        if self.keys_arr.ndim != 3 or self.keys_arr.shape[2] != 2:
            raise ValueError(
                f"Expected numeric key shape (N,D,2), "
                f"got {self.keys_arr.shape}"
            )

        self.index = {}

        if self.has_face:
            self._build_face_grids()
        else:
            self._build_numeric_grid()

        self._build_index()

    def _make_grid_data(self, numeric_keys):
        keys = np.asarray(numeric_keys, dtype=np.float64)

        if keys.ndim != 3 or keys.shape[2] != 2:
            raise ValueError(
                f"Expected numeric key shape (N,D,2), got {keys.shape}"
            )

        mins = keys[:, :, 0].copy()
        maxs = keys[:, :, 1].copy()

        # Numeric dimensions:
        #     x, y, z, yaw
        # and optionally:
        #     door
        mins[:, 3] = self._wrap_pi(mins[:, 3])
        maxs[:, 3] = self._wrap_pi(maxs[:, 3])

        x_mins = np.sort(np.unique(mins[:, 0]))
        y_mins = np.sort(np.unique(mins[:, 1]))
        z_values = np.sort(np.unique(mins[:, 2]))
        yaw_mins = np.sort(np.unique(mins[:, 3]))

        grid = {
            "x_global_min": float(np.min(mins[:, 0])),
            "x_global_max": float(np.max(maxs[:, 0])),
            "y_global_min": float(np.min(mins[:, 1])),
            "y_global_max": float(np.max(maxs[:, 1])),
            "x_mins": x_mins,
            "y_mins": y_mins,
            "z_values": z_values,
            "yaw_mins": yaw_mins,
            "x_to_ix": {
                float(value): index
                for index, value in enumerate(x_mins)
            },
            "y_to_iy": {
                float(value): index
                for index, value in enumerate(y_mins)
            },
            "nx": len(x_mins),
            "ny": len(y_mins),
            "nz": len(z_values),
            "nyaw": len(yaw_mins),
        }

        if keys.shape[1] == 5:
            door_mins = np.sort(np.unique(mins[:, 4]))
            grid.update({
                "door_global_min": float(np.min(mins[:, 4])),
                "door_global_max": float(np.max(maxs[:, 4])),
                "door_mins": door_mins,
                "door_to_idoor": {
                    float(value): index
                    for index, value in enumerate(door_mins)
                },
                "ndoor": len(door_mins),
            })

        return grid

    def _build_face_grids(self):
        self.face_values = tuple(
            face
            for face in self.FACE_VALUES
            if any(key[0] == face for key in self.keys_list)
        )

        self._face_to_iface = {
            face: index
            for index, face in enumerate(self.face_values)
        }

        self.face_grids = {}

        for face in self.face_values:
            face_numeric_keys = [
                key[1:]
                for key in self.keys_list
                if key[0] == face
            ]

            self.face_grids[face] = self._make_grid_data(
                face_numeric_keys
            )
    
    def key_to_indices(self, key):
        if self.has_face:
            face = key[0]
            numeric_key = key[1:]
            iface = self._face_to_iface[face]
            grid = self.face_grids[face]
        else:
            numeric_key = key
            grid = self.grid

        x_min = float(numeric_key[0][0])
        y_min = float(numeric_key[1][0])
        z_val = float(numeric_key[2][0])
        yaw_min = float(self._wrap_pi(numeric_key[3][0]))

        ix = grid["x_to_ix"][x_min]
        iy = grid["y_to_iy"][y_min]

        iz = int(np.argmin(
            np.abs(grid["z_values"] - z_val)
        ))

        iyaw = int(np.argmin(
            np.abs(grid["yaw_mins"] - yaw_min)
        ))

        if self.has_face:
            return iface, ix, iy, iz, iyaw

        if self.has_door:
            door_min = float(numeric_key[4][0])
            idoor = grid["door_to_idoor"][door_min]
            return ix, iy, iz, iyaw, idoor

        return ix, iy, iz, iyaw

    def _build_numeric_grid(self):
        grid = self._make_grid_data(self.keys_list)
        self.grid = grid

        # Preserve the old public attributes for non-AllStable users.
        self.x_global_min = grid["x_global_min"]
        self.x_global_max = grid["x_global_max"]
        self.y_global_min = grid["y_global_min"]
        self.y_global_max = grid["y_global_max"]

        self.x_mins = grid["x_mins"]
        self.y_mins = grid["y_mins"]
        self.z_values = grid["z_values"]
        self.yaw_mins = grid["yaw_mins"]

        self._x_to_ix = grid["x_to_ix"]
        self._y_to_iy = grid["y_to_iy"]

        self.nx = grid["nx"]
        self.ny = grid["ny"]
        self.nz = grid["nz"]
        self.nyaw = grid["nyaw"]

        if self.has_door:
            self.door_global_min = grid["door_global_min"]
            self.door_global_max = grid["door_global_max"]
            self.door_mins = grid["door_mins"]
            self._door_to_idoor = grid["door_to_idoor"]
            self.ndoor = grid["ndoor"]

    def _build_index(self):
        for bin_idx, key in enumerate(self.keys_list):
            if self.has_face:
                face = key[0]
                numeric_key = key[1:]

                if face not in self._face_to_iface:
                    raise ValueError(
                        f"Unknown AllStable face value: {face}"
                    )

                iface = self._face_to_iface[face]
                grid = self.face_grids[face]
            else:
                numeric_key = key
                grid = self.grid

            x_min = float(numeric_key[0][0])
            y_min = float(numeric_key[1][0])
            z_val = float(numeric_key[2][0])
            yaw_min = float(
                self._wrap_pi(numeric_key[3][0])
            )

            ix = grid["x_to_ix"][x_min]
            iy = grid["y_to_iy"][y_min]

            iz = int(np.argmin(
                np.abs(grid["z_values"] - z_val)
            ))

            iyaw = int(np.argmin(
                np.abs(grid["yaw_mins"] - yaw_min)
            ))

            if not np.isclose(
                grid["yaw_mins"][iyaw],
                yaw_min,
                atol=self.tol,
                rtol=0.0,
            ):
                raise RuntimeError(
                    f"Could not match yaw minimum "
                    f"{yaw_min:.17f}"
                )

            if self.has_face:
                indices = (
                    iface,
                    ix,
                    iy,
                    iz,
                    iyaw,
                )
            elif self.has_door:
                door_min = float(numeric_key[4][0])
                idoor = grid["door_to_idoor"][door_min]

                indices = (
                    ix,
                    iy,
                    iz,
                    iyaw,
                    idoor,
                )
            else:
                indices = (
                    ix,
                    iy,
                    iz,
                    iyaw,
                )

            if indices in self.index:
                previous_key = self.keys_list[
                    self.index[indices]
                ]

                raise RuntimeError(
                    f"Duplicate bin index {indices}: "
                    f"{previous_key} and {key}"
                )

            self.index[indices] = bin_idx

    def _bin_indices(self, sample):
        if len(sample) != self.ndim:
            raise ValueError(
                f"Expected sample with {self.ndim} values, "
                f"got {len(sample)}: {sample}"
            )

        if self.has_face:
            face = sample[0]

            if face not in self._face_to_iface:
                return None

            iface = self._face_to_iface[face]
            numeric_sample = sample[1:]
            grid = self.face_grids[face]
        else:
            numeric_sample = sample
            grid = self.grid

        x = float(numeric_sample[0])
        y = float(numeric_sample[1])
        z = float(numeric_sample[2])
        yaw = float(
            self._wrap_pi(numeric_sample[3])
        )

        # ---------- X ----------
        if (
            x < grid["x_global_min"] - self.tol
            or x >= grid["x_global_max"]
        ):
            return None

        ix = int(
            np.searchsorted(
                grid["x_mins"],
                x,
                side="right",
            ) - 1
        )

        if ix < 0 or ix >= grid["nx"]:
            return None

        # ---------- Y ----------
        if (
            y < grid["y_global_min"] - self.tol
            or y >= grid["y_global_max"]
        ):
            return None

        iy = int(
            np.searchsorted(
                grid["y_mins"],
                y,
                side="right",
            ) - 1
        )

        if iy < 0 or iy >= grid["ny"]:
            return None

        # ---------- Z ----------
        iz = int(np.argmin(
            np.abs(grid["z_values"] - z)
        ))

        # ---------- YAW ----------
        iyaw = int(
            np.searchsorted(
                grid["yaw_mins"],
                yaw,
                side="right",
            ) - 1
        )

        # Yaw is periodic. A wrapped value below the first start
        # belongs to the final yaw bin.
        if iyaw < 0:
            iyaw = grid["nyaw"] - 1

        if iyaw >= grid["nyaw"]:
            return None

        if self.has_face:
            return (
                iface,
                ix,
                iy,
                iz,
                iyaw,
            )

        # ---------- DOOR ----------
        if self.has_door:
            door = float(numeric_sample[4])

            if (
                door < grid["door_global_min"] - self.tol
                or door >= grid["door_global_max"]
            ):
                return None

            idoor = int(
                np.searchsorted(
                    grid["door_mins"],
                    door,
                    side="right",
                ) - 1
            )

            if idoor < 0 or idoor >= grid["ndoor"]:
                return None

            return (
                ix,
                iy,
                iz,
                iyaw,
                idoor,
            )

        return (
            ix,
            iy,
            iz,
            iyaw,
        )

    def query_point(self, sample):
        indices = self._bin_indices(sample)

        if indices is None:
            return None

        bin_idx = self.index.get(indices)

        if bin_idx is None:
            return None

        key = self.keys_list[bin_idx]

        if self.has_face:
            if sample[0] != key[0]:
                return None

            numeric_sample = sample[1:]
            numeric_key = key[1:]
        else:
            numeric_sample = sample
            numeric_key = key

        if not self._inside_nonperiodic(
            numeric_sample[0],
            numeric_key[0][0],
            numeric_key[0][1],
        ):
            return None

        if not self._inside_nonperiodic(
            numeric_sample[1],
            numeric_key[1][0],
            numeric_key[1][1],
        ):
            return None

        if not self._inside_yaw(
            numeric_sample[3],
            numeric_key[3][0],
            numeric_key[3][1],
        ):
            return None

        if self.has_door:
            if not self._inside_nonperiodic(
                numeric_sample[4],
                numeric_key[4][0],
                numeric_key[4][1],
            ):
                return None

        return key

    def _inside_nonperiodic(self, value, lo, hi):
        value = float(value)
        lo = float(lo)
        hi = float(hi)

        return (
            value >= lo - self.tol
            and value < hi
        )

    def _inside_yaw(self, value, lo, hi):
        value = float(self._wrap_pi(value))
        lo = float(self._wrap_pi(lo))
        hi = float(self._wrap_pi(hi))

        # A zero-width interval represents a fixed yaw.
        if np.isclose(lo, hi, atol=self.tol, rtol=0.0):
            return np.isclose(value, lo, atol=self.tol, rtol=0.0)

        if lo < hi:
            return (
                value >= lo - self.tol
                and value < hi
            )

        return (
            value >= lo - self.tol
            or value < hi
        )

    @staticmethod
    def _wrap_pi(angle):
        return (
            np.asarray(angle) + np.pi
        ) % (2.0 * np.pi) - np.pi


# class BoxGrid:
#     FACE_VALUES = ("xy", "yz", "zx")

#     def __init__(self, keys_to_root, tol=1e-12):
#         self.keys_list = list(keys_to_root.keys())
#         self.tol = tol

#         if not self.keys_list:
#             raise ValueError("keys_to_root cannot be empty")

#         first_key = self.keys_list[0]

#         # AllStable:
#         #     (face, x, y, z, yaw)
#         #
#         # Standard:
#         #     (x, y, z, yaw)
#         #
#         # Microwave:
#         #     (x, y, z, yaw, door)
#         self.has_face = (
#             len(first_key) == 5
#             and isinstance(first_key[0], str)
#         )

#         if self.has_face:
#             if first_key[0] not in self.FACE_VALUES:
#                 raise ValueError(
#                     f"Unknown AllStable face value: {first_key[0]}"
#                 )

#             self.numeric_offset = 1
#             self.has_door = False
#             self.ndim = 5

#             numeric_keys = [
#                 key[1:]
#                 for key in self.keys_list
#             ]
#         else:
#             self.numeric_offset = 0
#             self.ndim = len(first_key)
#             self.has_door = self.ndim == 5

#             numeric_keys = self.keys_list

#         if self.ndim not in (4, 5):
#             raise ValueError(
#                 f"Expected 4D or 5D keys, got {self.ndim} dimensions"
#             )

#         self.keys_arr = np.asarray(
#             numeric_keys,
#             dtype=np.float64,
#         )

#         keys = self.keys_arr

#         if keys.ndim != 3 or keys.shape[2] != 2:
#             raise ValueError(
#                 f"Expected numeric key shape (N,D,2), got {keys.shape}"
#             )

#         # Numeric dimensions are always:
#         #     x, y, z, yaw
#         #
#         # and optionally:
#         #     door
#         mins = keys[:, :, 0].copy()
#         maxs = keys[:, :, 1].copy()

#         # Wrap yaw into [-pi, pi).
#         mins[:, 3] = self._wrap_pi(mins[:, 3])
#         maxs[:, 3] = self._wrap_pi(maxs[:, 3])

#         # ---------- FACE ----------
#         if self.has_face:
#             self.face_values = tuple(
#                 face
#                 for face in self.FACE_VALUES
#                 if any(key[0] == face for key in self.keys_list)
#             )

#             self._face_to_iface = {
#                 face: index
#                 for index, face in enumerate(self.face_values)
#             }

#         # ---------- GLOBAL BOUNDS ----------
#         self.x_global_min = float(np.min(mins[:, 0]))
#         self.x_global_max = float(np.max(maxs[:, 0]))

#         self.y_global_min = float(np.min(mins[:, 1]))
#         self.y_global_max = float(np.max(maxs[:, 1]))

#         if self.has_door:
#             self.door_global_min = float(np.min(mins[:, 4]))
#             self.door_global_max = float(np.max(maxs[:, 4]))

#         # ---------- X/Y BIN STARTS ----------
#         self.x_mins = np.sort(np.unique(mins[:, 0]))
#         self.y_mins = np.sort(np.unique(mins[:, 1]))

#         self._x_to_ix = {
#             float(value): index
#             for index, value in enumerate(self.x_mins)
#         }

#         self._y_to_iy = {
#             float(value): index
#             for index, value in enumerate(self.y_mins)
#         }

#         self.nx = len(self.x_mins)
#         self.ny = len(self.y_mins)

#         # ---------- Z DISCRETE LEVELS ----------
#         self.z_values = np.sort(np.unique(mins[:, 2]))
#         self.nz = len(self.z_values)

#         # ---------- YAW BIN STARTS ----------
#         self.yaw_mins = np.sort(np.unique(mins[:, 3]))
#         self.nyaw = len(self.yaw_mins)

#         # ---------- DOOR BIN STARTS ----------
#         if self.has_door:
#             self.door_mins = np.sort(np.unique(mins[:, 4]))
#             self.ndoor = len(self.door_mins)

#             self._door_to_idoor = {
#                 float(value): index
#                 for index, value in enumerate(self.door_mins)
#             }

#         # ---------- BUILD INDEX ----------
#         self.index = {}

#         for bin_idx, key in enumerate(self.keys_list):
#             if self.has_face:
#                 face = key[0]
#                 numeric_key = key[1:]

#                 if face not in self._face_to_iface:
#                     raise ValueError(
#                         f"Unknown AllStable face value: {face}"
#                     )

#                 iface = self._face_to_iface[face]
#             else:
#                 numeric_key = key

#             x_min = float(numeric_key[0][0])
#             y_min = float(numeric_key[1][0])
#             z_val = float(numeric_key[2][0])
#             yaw_min = float(
#                 self._wrap_pi(numeric_key[3][0])
#             )

#             ix = self._x_to_ix[x_min]
#             iy = self._y_to_iy[y_min]

#             iz = int(np.argmin(
#                 np.abs(self.z_values - z_val)
#             ))

#             iyaw = int(np.argmin(
#                 np.abs(self.yaw_mins - yaw_min)
#             ))

#             if not np.isclose(
#                 self.yaw_mins[iyaw],
#                 yaw_min,
#                 atol=self.tol,
#                 rtol=0.0,
#             ):
#                 raise RuntimeError(
#                     f"Could not match yaw minimum "
#                     f"{yaw_min:.17f}"
#                 )

#             if self.has_face:
#                 indices = (
#                     iface,
#                     ix,
#                     iy,
#                     iz,
#                     iyaw,
#                 )
#             elif self.has_door:
#                 door_min = float(numeric_key[4][0])
#                 idoor = self._door_to_idoor[door_min]

#                 indices = (
#                     ix,
#                     iy,
#                     iz,
#                     iyaw,
#                     idoor,
#                 )
#             else:
#                 indices = (
#                     ix,
#                     iy,
#                     iz,
#                     iyaw,
#                 )

#             if indices in self.index:
#                 previous_key = self.keys_list[
#                     self.index[indices]
#                 ]

#                 raise RuntimeError(
#                     f"Duplicate bin index {indices}: "
#                     f"{previous_key} and {key}"
#                 )

#             self.index[indices] = bin_idx

#     def _bin_indices(self, sample):
#         if len(sample) != self.ndim:
#             raise ValueError(
#                 f"Expected sample with {self.ndim} values, "
#                 f"got {len(sample)}: {sample}"
#             )

#         if self.has_face:
#             face = sample[0]

#             if face not in self._face_to_iface:
#                 return None

#             iface = self._face_to_iface[face]
#             numeric_sample = sample[1:]
#         else:
#             numeric_sample = sample

#         x = float(numeric_sample[0])
#         y = float(numeric_sample[1])
#         z = float(numeric_sample[2])
#         yaw = float(
#             self._wrap_pi(numeric_sample[3])
#         )

#         # ---------- X ----------
#         if (
#             x < self.x_global_min - self.tol
#             or x >= self.x_global_max
#         ):
#             return None

#         ix = int(
#             np.searchsorted(
#                 self.x_mins,
#                 x,
#                 side="right",
#             ) - 1
#         )

#         if ix < 0 or ix >= self.nx:
#             return None

#         # ---------- Y ----------
#         if (
#             y < self.y_global_min - self.tol
#             or y >= self.y_global_max
#         ):
#             return None

#         iy = int(
#             np.searchsorted(
#                 self.y_mins,
#                 y,
#                 side="right",
#             ) - 1
#         )

#         if iy < 0 or iy >= self.ny:
#             return None

#         # ---------- Z ----------
#         iz = int(np.argmin(
#             np.abs(self.z_values - z)
#         ))

#         # ---------- YAW ----------
#         iyaw = int(
#             np.searchsorted(
#                 self.yaw_mins,
#                 yaw,
#                 side="right",
#             ) - 1
#         )

#         if iyaw < 0 or iyaw >= self.nyaw:
#             return None

#         if self.has_face:
#             return (
#                 iface,
#                 ix,
#                 iy,
#                 iz,
#                 iyaw,
#             )

#         # ---------- DOOR ----------
#         if self.has_door:
#             door = float(numeric_sample[4])

#             if (
#                 door < self.door_global_min - self.tol
#                 or door >= self.door_global_max
#             ):
#                 return None

#             idoor = int(
#                 np.searchsorted(
#                     self.door_mins,
#                     door,
#                     side="right",
#                 ) - 1
#             )

#             if idoor < 0 or idoor >= self.ndoor:
#                 return None

#             return (
#                 ix,
#                 iy,
#                 iz,
#                 iyaw,
#                 idoor,
#             )

#         return (
#             ix,
#             iy,
#             iz,
#             iyaw,
#         )

#     def query_point(self, sample):
#         indices = self._bin_indices(sample)

#         if indices is None:
#             return None

#         bin_idx = self.index.get(indices)

#         if bin_idx is None:
#             return None

#         key = self.keys_list[bin_idx]

#         if self.has_face:
#             if sample[0] != key[0]:
#                 return None

#             numeric_sample = sample[1:]
#             numeric_key = key[1:]
#         else:
#             numeric_sample = sample
#             numeric_key = key

#         if not self._inside_nonperiodic(
#             numeric_sample[0],
#             numeric_key[0][0],
#             numeric_key[0][1],
#         ):
#             return None

#         if not self._inside_nonperiodic(
#             numeric_sample[1],
#             numeric_key[1][0],
#             numeric_key[1][1],
#         ):
#             return None

#         if not self._inside_yaw(
#             numeric_sample[3],
#             numeric_key[3][0],
#             numeric_key[3][1],
#         ):
#             return None

#         if self.has_door:
#             if not self._inside_nonperiodic(
#                 numeric_sample[4],
#                 numeric_key[4][0],
#                 numeric_key[4][1],
#             ):
#                 return None

#         return key

#     def _inside_nonperiodic(self, value, lo, hi):
#         value = float(value)
#         lo = float(lo)
#         hi = float(hi)

#         return (
#             value >= lo - self.tol
#             and value < hi
#         )

#     def _inside_yaw(self, value, lo, hi):
#         value = float(self._wrap_pi(value))
#         lo = float(self._wrap_pi(lo))
#         hi = float(self._wrap_pi(hi))

#         if lo <= hi:
#             return (
#                 value >= lo - self.tol
#                 and value < hi
#             )

#         return (
#             value >= lo - self.tol
#             or value < hi
#         )

#     @staticmethod
#     def _wrap_pi(angle):
#         return (
#             np.asarray(angle) + np.pi
#         ) % (2.0 * np.pi) - np.pi

# class BoxGrid:
#     def __init__(self, keys_to_root, tol=1e-12):
#         self.keys_list = list(keys_to_root.keys())
#         self.keys_arr = np.asarray(self.keys_list, dtype=np.float64)
#         self.tol = tol

#         keys = self.keys_arr

#         if keys.ndim != 3 or keys.shape[2] != 2:
#             raise ValueError(
#                 f"Expected keys shape (N,D,2), got {keys.shape}"
#             )

#         self.ndim = keys.shape[1]

#         if self.ndim not in (4, 5):
#             raise ValueError(
#                 f"Expected 4D or 5D keys, got {self.ndim} dimensions"
#             )

#         self.has_door = self.ndim == 5

#         mins = keys[:, :, 0].copy()
#         maxs = keys[:, :, 1].copy()

#         # Wrap yaw bounds into [-pi, pi).
#         mins[:, 3] = self._wrap_pi(mins[:, 3])
#         maxs[:, 3] = self._wrap_pi(maxs[:, 3])

#         # ---------- GLOBAL BOUNDS ----------
#         self.x_global_min = float(np.min(mins[:, 0]))
#         self.x_global_max = float(np.max(maxs[:, 0]))

#         self.y_global_min = float(np.min(mins[:, 1]))
#         self.y_global_max = float(np.max(maxs[:, 1]))

#         if self.has_door:
#             self.door_global_min = float(np.min(mins[:, 4]))
#             self.door_global_max = float(np.max(maxs[:, 4]))

#         # ---------- X/Y: BIN STARTS ----------
#         self.x_mins = np.sort(np.unique(mins[:, 0]))
#         self.y_mins = np.sort(np.unique(mins[:, 1]))

#         self._x_to_ix = {
#             float(value): index
#             for index, value in enumerate(self.x_mins)
#         }

#         self._y_to_iy = {
#             float(value): index
#             for index, value in enumerate(self.y_mins)
#         }

#         self.nx = len(self.x_mins)
#         self.ny = len(self.y_mins)

#         # ---------- Z: DISCRETE LEVELS ----------
#         self.z_values = np.sort(np.unique(mins[:, 2]))
#         self.nz = len(self.z_values)

#         # ---------- YAW: ACTUAL BIN STARTS ----------
#         self.yaw_mins = np.sort(np.unique(mins[:, 3]))
#         self.nyaw = len(self.yaw_mins)

#         # ---------- DOOR: NON-PERIODIC BINS ----------
#         if self.has_door:
#             self.door_mins = np.sort(np.unique(mins[:, 4]))
#             self.ndoor = len(self.door_mins)

#             self._door_to_idoor = {
#                 float(value): index
#                 for index, value in enumerate(self.door_mins)
#             }

#         # ---------- BUILD INDEX ----------
#         self.index = {}

#         for bin_idx, key in enumerate(self.keys_list):
#             x_min = float(key[0][0])
#             y_min = float(key[1][0])
#             z_val = float(key[2][0])
#             yaw_min = float(self._wrap_pi(key[3][0]))

#             ix = self._x_to_ix[x_min]
#             iy = self._y_to_iy[y_min]

#             iz = int(np.argmin(
#                 np.abs(self.z_values - z_val)
#             ))

#             iyaw = int(np.argmin(
#                 np.abs(self.yaw_mins - yaw_min)
#             ))

#             if not np.isclose(
#                 self.yaw_mins[iyaw],
#                 yaw_min,
#                 atol=self.tol,
#                 rtol=0.0,
#             ):
#                 raise RuntimeError(
#                     f"Could not match yaw minimum "
#                     f"{yaw_min:.17f}"
#                 )

#             if self.has_door:
#                 door_min = float(key[4][0])
#                 idoor = self._door_to_idoor[door_min]
#                 indices = (ix, iy, iz, iyaw, idoor)
#             else:
#                 indices = (ix, iy, iz, iyaw)

#             if indices in self.index:
#                 previous_key = self.keys_list[
#                     self.index[indices]
#                 ]

#                 raise RuntimeError(
#                     f"Duplicate bin index {indices}: "
#                     f"{previous_key} and {key}"
#                 )

#             self.index[indices] = bin_idx

#     def _bin_indices(self, sample):
#         if len(sample) != self.ndim:
#             raise ValueError(
#                 f"Expected sample with {self.ndim} values, "
#                 f"got {len(sample)}: {sample}"
#             )

#         x = float(sample[0])
#         y = float(sample[1])
#         z = float(sample[2])
#         yaw = float(self._wrap_pi(sample[3]))

#         # ---------- X ----------
#         if (
#             x < self.x_global_min - self.tol
#             or x >= self.x_global_max
#         ):
#             return None

#         ix = int(
#             np.searchsorted(
#                 self.x_mins,
#                 x,
#                 side="right",
#             ) - 1
#         )

#         if ix < 0 or ix >= self.nx:
#             return None

#         # ---------- Y ----------
#         if (
#             y < self.y_global_min - self.tol
#             or y >= self.y_global_max
#         ):
#             return None

#         iy = int(
#             np.searchsorted(
#                 self.y_mins,
#                 y,
#                 side="right",
#             ) - 1
#         )

#         if iy < 0 or iy >= self.ny:
#             return None

#         # ---------- Z ----------
#         iz = int(np.argmin(
#             np.abs(self.z_values - z)
#         ))

#         # ---------- YAW ----------
#         iyaw = int(
#             np.searchsorted(
#                 self.yaw_mins,
#                 yaw,
#                 side="right",
#             ) - 1
#         )

#         # This assumes your yaw region is bounded rather than a grid
#         # covering the entire periodic circle.
#         if iyaw < 0 or iyaw >= self.nyaw:
#             return None

#         # ---------- DOOR ----------
#         if self.has_door:
#             door = float(sample[4])

#             if (
#                 door < self.door_global_min - self.tol
#                 or door >= self.door_global_max
#             ):
#                 return None

#             idoor = int(
#                 np.searchsorted(
#                     self.door_mins,
#                     door,
#                     side="right",
#                 ) - 1
#             )

#             if idoor < 0 or idoor >= self.ndoor:
#                 return None

#             return ix, iy, iz, iyaw, idoor

#         return ix, iy, iz, iyaw

#     def query_point(self, sample):
#         indices = self._bin_indices(sample)

#         if indices is None:
#             return None

#         bin_idx = self.index.get(indices)

#         if bin_idx is None:
#             return None

#         key = self.keys_list[bin_idx]

#         # Verify that the selected key actually contains the sample.
#         if not self._inside_nonperiodic(
#             sample[0], key[0][0], key[0][1]
#         ):
#             return None

#         if not self._inside_nonperiodic(
#             sample[1], key[1][0], key[1][1]
#         ):
#             return None

#         if not self._inside_yaw(
#             sample[3], key[3][0], key[3][1]
#         ):
#             return None

#         if self.has_door:
#             if not self._inside_nonperiodic(
#                 sample[4], key[4][0], key[4][1]
#             ):
#                 return None

#         return key

#     def _inside_nonperiodic(self, value, lo, hi):
#         value = float(value)
#         lo = float(lo)
#         hi = float(hi)

#         return (
#             value >= lo - self.tol
#             and value < hi
#         )

#     def _inside_yaw(self, value, lo, hi):
#         value = float(self._wrap_pi(value))
#         lo = float(self._wrap_pi(lo))
#         hi = float(self._wrap_pi(hi))

#         if lo <= hi:
#             return (
#                 value >= lo - self.tol
#                 and value < hi
#             )

#         # Interval crosses the -pi/pi boundary.
#         return (
#             value >= lo - self.tol
#             or value < hi
#         )

#     @staticmethod
#     def _wrap_pi(angle):
#         return (
#             np.asarray(angle) + np.pi
#         ) % (2.0 * np.pi) - np.pi


def deep_tuple(x):
    # NumPy array
    if isinstance(x, np.ndarray):
        return tuple(deep_tuple(i) for i in x)
    # Python list or tuple
    elif isinstance(x, (list, tuple)):
        return tuple(deep_tuple(i) for i in x)
    # Base case (scalar)
    else:
        return x


def get_avg_path_length(root_path, key_map):
    lengths = np.zeros(len(key_map), dtype=float)
    for i, key in enumerate(key_map):
        key = deep_tuple(key)
        root_id, goal_q = key_map[key]
        path = list(root_path[root_id].copy())
        path.append(goal_q)

        lengths[i] = traj_len(path)
    return np.mean(lengths)

def sample_from_key(key):
    """
    Sample a configuration from either:

    Standard:
        ((x_lo, x_hi), (y_lo, y_hi), (z_lo, z_hi), (yaw_lo, yaw_hi))

    Microwave:
        ((x_lo, x_hi), (y_lo, y_hi), (z_lo, z_hi),
         (yaw_lo, yaw_hi), (door_lo, door_hi))

    AllStable:
        (face, (x_lo, x_hi), (y_lo, y_hi),
         (z_lo, z_hi), (yaw_lo, yaw_hi))
    """
    has_face = (
        len(key) == 5
        and isinstance(key[0], str)
    )

    if has_face:
        face = key[0]
        numeric_key = key[1:]
    else:
        face = None
        numeric_key = key

    numeric_sample = []

    for lo, hi in numeric_key:
        lo = float(lo)
        hi = float(hi)

        # Prevent sampling the upper boundary of a half-open bin.
        upper = np.nextafter(hi, lo)

        if np.isclose(lo, hi):
            value = lo
        else:
            value = np.random.uniform(lo, upper)

        numeric_sample.append(value)

    if has_face:
        return [face, *numeric_sample]

    return numeric_sample


class Library:
    def __init__(
        self,
        N: int,
        env: MujocoEnv,
        robot: MujocoRobot,
        home_qpos,
        key_to_root,
        root_paths,
        solved_keys,
        data,
    ):
        """Build nearest-neighbor library baseline from adaptation files only.

        This avoids loading task_paths_data_*.npy / task_paths_keys_*.pkl.
        Each stored path is reconstructed as:
            root_paths[root_id] + [goal_q]
        where (root_id, goal_q) comes from key_to_root.
        """
        self.key_to_root = key_to_root
        self.root_paths = root_paths
        self.indexer = BoxGrid(key_to_root)
        self.robot = robot

        if isinstance(env, ShelfEnv) and isinstance(robot, FetchArm):
            self.ompl_planner = OMPLPlanner(robot, data, rrtc_range=0.1)
        elif isinstance(env, CageEnv) and isinstance(robot, FetchArm):
            self.ompl_planner = OMPLPlanner(robot, data, rrtc_range=0.1)
        else:
            self.ompl_planner = OMPLPlanner(robot, data)
        # self.ompl_planner = OMPLPlanner(robot, data)
        # solved_key_list = list(solved_keys.keys())
        solved_key_list = list(solved_keys)

        self.library = {}
        pbar_library = tqdm(total=N, desc="Building library", leave=True)

        iters = 0
        while len(self.library) <= N:
            iters += 1

            random_key = solved_key_list[
                np.random.randint(0, len(solved_key_list))
            ]
            # sample = [np.random.uniform(lo, hi) for lo, hi in random_key]
            sample = sample_from_key(random_key)

            recovered_key = self.indexer.query_point(sample)
            if recovered_key is None:
                # print("Failed to find sample", flush=True)
                continue
            root_id, curr_goal = key_to_root[recovered_key]
            if root_id is None:
                continue
            root_path = root_paths[root_id]
            if root_path is None or len(root_path) == 0:
                continue

            path = list(root_path.copy())
            path.append(curr_goal)

            # path, total_time, planning_time = self.ompl_planner.plan(
            #     start=home_qpos,
            #     goal=curr_goal,
            #     timeout=5.0,
            #     num_waypoints=200,
            #     benchmark=True,
            # )

            if tuple(sample) not in self.library:
                # print(tuple(sample))
                self.library[tuple(sample)] = (curr_goal, path)
                pbar_library.update(1)

        self.lib_index = self.build_library_index(w_yaw=1.0, w_door=1.0)
        # return self.lib_index

    def build_library_index(
        self,
        z_tol=1e-6,
        w_yaw=1.0,
        w_door=1.0,
    ):
        """
        Library keys may be:

            Standard:
                (x, y, z, yaw)

            Microwave:
                (x, y, z, yaw, door)

            AllStable:
                (face, x, y, z, yaw)

        Standard and microwave keys are grouped by discrete z.

        AllStable keys are grouped by:
            (face, discrete z)
        """
        raw_keys = list(self.library.keys())

        if not raw_keys:
            raise ValueError("Cannot build an index from an empty library")

        first_key = raw_keys[0]

        has_face = (
            len(first_key) == 5
            and isinstance(first_key[0], str)
        )

        if has_face:
            # AllStable:
            #     (face, x, y, z, yaw)
            faces = np.asarray(
                [key[0] for key in raw_keys],
                dtype=object,
            )

            keys = np.asarray(
                [key[1:] for key in raw_keys],
                dtype=np.float64,
            )

            ndim = 5
            has_door = False
        else:
            keys = np.asarray(
                raw_keys,
                dtype=np.float64,
            )

            if keys.ndim != 2 or keys.shape[1] not in (4, 5):
                raise ValueError(
                    f"Expected keys shape (N,4) or (N,5), "
                    f"got {keys.shape}"
                )

            ndim = keys.shape[1]
            has_door = ndim == 5
            faces = None

        # Numeric keys must always contain:
        #     x, y, z, yaw
        #
        # and optionally:
        #     door
        if keys.ndim != 2 or keys.shape[1] not in (4, 5):
            raise ValueError(
                f"Expected numeric keys shape (N,4) or (N,5), "
                f"got {keys.shape}"
            )

        z_vals = np.sort(np.unique(keys[:, 2]))

        trees = {}
        key_lists = {}

        yaw_scale = np.sqrt(w_yaw)
        door_scale = np.sqrt(w_door)

        if has_face:
            face_values = tuple(
                face
                for face in ("xy", "yz", "zx")
                if np.any(faces == face)
            )

            for face in face_values:
                face_mask = faces == face

                for z0 in z_vals:
                    mask = (
                        face_mask
                        & (np.abs(keys[:, 2] - z0) <= z_tol)
                    )

                    kz = keys[mask]

                    if kz.size == 0:
                        continue

                    # [x, y, cos(yaw), sin(yaw)]
                    feats = np.column_stack(
                        [
                            kz[:, 0],
                            kz[:, 1],
                            yaw_scale * np.cos(kz[:, 3]),
                            yaw_scale * np.sin(kz[:, 3]),
                        ]
                    )

                    group = (face, float(z0))

                    trees[group] = cKDTree(feats)

                    # Restore the original AllStable key format.
                    key_lists[group] = [
                        (
                            face,
                            float(row[0]),
                            float(row[1]),
                            float(row[2]),
                            float(row[3]),
                        )
                        for row in kz
                    ]

        else:
            face_values = None

            for z0 in z_vals:
                mask = np.abs(keys[:, 2] - z0) <= z_tol
                kz = keys[mask]

                if kz.size == 0:
                    continue

                if has_door:
                    # [x, y, cos(yaw), sin(yaw), door]
                    feats = np.column_stack(
                        [
                            kz[:, 0],
                            kz[:, 1],
                            yaw_scale * np.cos(kz[:, 3]),
                            yaw_scale * np.sin(kz[:, 3]),
                            door_scale * kz[:, 4],
                        ]
                    )
                else:
                    # [x, y, cos(yaw), sin(yaw)]
                    feats = np.column_stack(
                        [
                            kz[:, 0],
                            kz[:, 1],
                            yaw_scale * np.cos(kz[:, 3]),
                            yaw_scale * np.sin(kz[:, 3]),
                        ]
                    )

                group = float(z0)

                trees[group] = cKDTree(feats)

                key_lists[group] = [
                    tuple(float(value) for value in row)
                    for row in kz
                ]

        return {
            "ndim": ndim,
            "has_face": has_face,
            "has_door": has_door,
            "face_values": face_values,
            "z_vals": z_vals,
            "trees": trees,
            "key_lists": key_lists,
            "yaw_scale": yaw_scale,
            "door_scale": door_scale,
            "z_tol": z_tol,
        }

    # def build_library_index(
    #     self,
    #     z_tol=1e-6,
    #     w_yaw=1.0,
    #     w_door=1.0,
    # ):
    #     """
    #     Library keys may be:

    #         (x, y, z, yaw)
    #         (x, y, z, yaw, door)

    #     Returns a nearest-neighbor index grouped by discrete z.
    #     """
    #     keys = np.asarray(
    #         list(self.library.keys()),
    #         dtype=np.float64,
    #     )

    #     if keys.ndim != 2 or keys.shape[1] not in (4, 5):
    #         raise ValueError(
    #             f"Expected keys shape (N,4) or (N,5), got {keys.shape}"
    #         )

    #     ndim = keys.shape[1]
    #     has_door = ndim == 5

    #     # Group by discrete z.
    #     z_vals = np.sort(np.unique(keys[:, 2]))

    #     trees = {}
    #     key_lists = {}

    #     yaw_scale = np.sqrt(w_yaw)
    #     door_scale = np.sqrt(w_door)

    #     for z0 in z_vals:
    #         mask = np.abs(keys[:, 2] - z0) <= z_tol
    #         kz = keys[mask]

    #         if kz.size == 0:
    #             continue

    #         if has_door:
    #             # [x, y, cos(yaw), sin(yaw), door]
    #             feats = np.column_stack(
    #                 [
    #                     kz[:, 0],
    #                     kz[:, 1],
    #                     yaw_scale * np.cos(kz[:, 3]),
    #                     yaw_scale * np.sin(kz[:, 3]),
    #                     door_scale * kz[:, 4],
    #                 ]
    #             )
    #         else:
    #             # [x, y, cos(yaw), sin(yaw)]
    #             feats = np.column_stack(
    #                 [
    #                     kz[:, 0],
    #                     kz[:, 1],
    #                     yaw_scale * np.cos(kz[:, 3]),
    #                     yaw_scale * np.sin(kz[:, 3]),
    #                 ]
    #             )

    #         trees[float(z0)] = cKDTree(feats)

    #         key_lists[float(z0)] = [
    #             tuple(row)
    #             for row in kz
    #         ]

    #     return {
    #         "ndim": ndim,
    #         "has_door": has_door,
    #         "z_vals": z_vals,
    #         "trees": trees,
    #         "key_lists": key_lists,
    #         "yaw_scale": yaw_scale,
    #         "door_scale": door_scale,
    #         "z_tol": z_tol,
    #     }

    # def query_library_nn(self, index, sample):
    #     """
    #     sample: [x,y,z,yaw]
    #     Returns: nearest_key, (curr_goal, path), distance
    #     """
    #     x, y, z, yaw = map(float, sample)

    #     # pick nearest z-slice
    #     z_vals = index["z_vals"]
    #     zi = int(np.argmin(np.abs(z_vals - z)))
    #     z0 = float(z_vals[zi])

    #     if abs(z0 - z) > index["z_tol"]:
    #         return None, None, np.inf

    #     tree = index["trees"].get(z0, None)
    #     if tree is None:
    #         return None, None, np.inf

    #     ys = index["yaw_scale"]
    #     q = np.array([x, y, ys * np.cos(yaw), ys * np.sin(yaw)], dtype=np.float64)

    #     dist, idx = tree.query(q, k=1)
    #     nearest_key = index["key_lists"][z0][int(idx)]
    #     return nearest_key, self.library[nearest_key], float(dist)

    # def query_library_nn(self, index, sample, n=1):
    #     """
    #     sample:
    #         [x, y, z, yaw]
    #         or
    #         [x, y, z, yaw, door]

    #     Returns:
    #         [(key, (curr_goal, path), distance), ...]
    #     """
    #     sample = np.asarray(sample, dtype=np.float64)

    #     expected_dim = index["ndim"]

    #     if sample.ndim != 1 or sample.shape[0] != expected_dim:
    #         raise ValueError(
    #             f"Expected sample shape ({expected_dim},), "
    #             f"got {sample.shape}"
    #         )

    #     x = float(sample[0])
    #     y = float(sample[1])
    #     z = float(sample[2])
    #     yaw = float(sample[3])

    #     # Pick nearest z slice.
    #     z_vals = index["z_vals"]

    #     if len(z_vals) == 0:
    #         return []

    #     zi = int(np.argmin(np.abs(z_vals - z)))
    #     z0 = float(z_vals[zi])

    #     if abs(z0 - z) > index["z_tol"]:
    #         return []

    #     tree = index["trees"].get(z0)

    #     if tree is None:
    #         return []

    #     yaw_scale = index["yaw_scale"]

    #     if index["has_door"]:
    #         door = float(sample[4])
    #         door_scale = index["door_scale"]

    #         query_features = np.array(
    #             [
    #                 x,
    #                 y,
    #                 yaw_scale * np.cos(yaw),
    #                 yaw_scale * np.sin(yaw),
    #                 door_scale * door,
    #             ],
    #             dtype=np.float64,
    #         )
    #     else:
    #         query_features = np.array(
    #             [
    #                 x,
    #                 y,
    #                 yaw_scale * np.cos(yaw),
    #                 yaw_scale * np.sin(yaw),
    #             ],
    #             dtype=np.float64,
    #         )

    #     num_points = len(index["key_lists"][z0])
    #     k = min(n, num_points)

    #     if k <= 0:
    #         return []

    #     dists, idxs = tree.query(query_features, k=k)

    #     dists = np.atleast_1d(dists)
    #     idxs = np.atleast_1d(idxs)

    #     results = []
    #     key_list = index["key_lists"][z0]

    #     for dist, idx in zip(dists, idxs):
    #         key = key_list[int(idx)]
    #         results.append(
    #             (
    #                 key,
    #                 self.library[key],
    #                 float(dist),
    #             )
    #         )

    #     return results

    def query_library_nn(self, lib_index, sample, n=1):
        """
        Query nearest library entries.

        Supported sample formats:

            Standard:
                [x, y, z, yaw]

            Microwave:
                [x, y, z, yaw, door]

            AllStable:
                [face, x, y, z, yaw]
        """
        has_face = lib_index["has_face"]
        has_door = lib_index["has_door"]

        if has_face:
            if len(sample) != 5 or not isinstance(sample[0], str):
                raise ValueError(
                    "AllStable query expects "
                    "[face, x, y, z, yaw], "
                    f"got {sample}"
                )

            face = sample[0]

            if face not in lib_index["face_values"]:
                return []

            numeric_sample = np.asarray(
                sample[1:],
                dtype=np.float64,
            )
        else:
            face = None

            numeric_sample = np.asarray(
                sample,
                dtype=np.float64,
            )

        expected_numeric_dim = 5 if has_door else 4

        if (
            numeric_sample.ndim != 1
            or numeric_sample.shape[0] != expected_numeric_dim
        ):
            raise ValueError(
                f"Expected numeric sample with "
                f"{expected_numeric_dim} values, "
                f"got shape {numeric_sample.shape}: {sample}"
            )

        x = float(numeric_sample[0])
        y = float(numeric_sample[1])
        z = float(numeric_sample[2])
        yaw = float(numeric_sample[3])

        z_vals = lib_index["z_vals"]

        if z_vals.size == 0:
            return []

        nearest_z = float(
            z_vals[np.argmin(np.abs(z_vals - z))]
        )

        if has_face:
            group = (face, nearest_z)
        else:
            group = nearest_z

        tree = lib_index["trees"].get(group)
        key_list = lib_index["key_lists"].get(group)

        if tree is None or key_list is None:
            return []

        yaw_scale = lib_index["yaw_scale"]

        if has_door:
            door = float(numeric_sample[4])
            door_scale = lib_index["door_scale"]

            query_feature = np.array(
                [
                    x,
                    y,
                    yaw_scale * np.cos(yaw),
                    yaw_scale * np.sin(yaw),
                    door_scale * door,
                ],
                dtype=np.float64,
            )
        else:
            query_feature = np.array(
                [
                    x,
                    y,
                    yaw_scale * np.cos(yaw),
                    yaw_scale * np.sin(yaw),
                ],
                dtype=np.float64,
            )

        k = min(int(n), len(key_list))

        if k <= 0:
            return []

        distances, indices = tree.query(
            query_feature,
            k=k,
        )

        distances = np.atleast_1d(distances)
        indices = np.atleast_1d(indices)

        results = []

        for dist, idx in zip(distances, indices):
            idx = int(idx)

            if idx < 0 or idx >= len(key_list):
                continue

            key = key_list[idx]

            results.append(
                (
                    key,
                    self.library[key],
                    float(dist),
                )
            )

        return results

    def check_path_collision(self, path):
        T = len(path)
        waypoint_valid = np.zeros(T, dtype=bool)
        for i, q in enumerate(path):
            self.robot.set_joint_qpos(q)
            in_collision = self.robot.in_contact()
            waypoint_valid[i] = not in_collision

        return waypoint_valid

    def collision_buffer(self, waypoint_valid, b=0):
        if b <= 0:
            return waypoint_valid.copy()

        T = waypoint_valid.shape[0]
        buffered = waypoint_valid.copy()

        coll = np.flatnonzero(~waypoint_valid)
        if coll.size == 0:
            return buffered

        # mark everything within +/- b of each collision as collision
        for i in coll:
            lo = max(0, i - b)
            hi = min(T, i + b + 1)  # +1 because slice end is exclusive
            buffered[lo:hi] = False

        return buffered

    def rewire_segments(
        self, path, validity_map, timeout=2.0, num_waypoints=20, max_repairs=20
    ):
        path = np.asarray(path, dtype=np.float64)
        valid = np.asarray(validity_map, dtype=bool)
        out = path.copy()

        prev_signature = None
        repairs = 0

        while True:
            invalid = ~valid
            n_invalid_before = int(invalid.sum())
            if n_invalid_before == 0:
                return out, True

            if repairs >= max_repairs:
                # prevent runaway time
                return out, True

            starts = np.flatnonzero(invalid & np.r_[True, ~invalid[:-1]])
            ends = np.flatnonzero(invalid & np.r_[~invalid[1:], True])

            rewired_any = False

            for s, e in zip(starts, ends):
                prev = s - 1
                while prev >= 0 and not valid[prev]:
                    prev -= 1

                nxt = e + 1
                while nxt < len(out) and not valid[nxt]:
                    nxt += 1

                if prev < 0 or nxt >= len(out):
                    continue  # edge run

                # detect "same segment again" (prevents looping on one stubborn gap)
                sig = (prev, nxt, len(out))
                if sig == prev_signature:
                    # give up on this neighbor path
                    return None, False
                prev_signature = sig

                q0, q1 = out[prev], out[nxt]

                t0 = time.perf_counter()
                rewired_segment, _, _ = self.ompl_planner.plan(
                    start=q0,
                    goal=q1,
                    timeout=timeout,
                    num_waypoints=num_waypoints,
                    benchmark=True,
                )
                t1 = time.perf_counter()
                # tqdm.write(f"rewire [{prev}->{nxt}] plan_time={t1-t0:.2f}s invalid={n_invalid_before}")

                if rewired_segment is None or len(rewired_segment) == 0:
                    return None, False

                rewired_segment = np.asarray(rewired_segment, dtype=np.float64)
                mid = (
                    rewired_segment[1:-1]
                    if rewired_segment.shape[0] >= 2
                    else rewired_segment
                )

                a = prev + 1
                out = np.vstack([out[:a], mid, out[nxt:]])

                # full recompute (expensive but correct with variable length)
                valid = self.check_path_collision(out)

                n_invalid_after = int((~valid).sum())
                if n_invalid_after >= n_invalid_before:
                    # no improvement -> stop before infinite repair loop
                    return None, False

                repairs += 1
                rewired_any = True
                break

            if not rewired_any:
                return out, True

    def rewire_to_goal(self, path, goal, n_wps=20, timeout=1.0):
        path = np.asarray(path, dtype=np.float64)
        T = len(path)

        # ---- find last valid waypoint ----
        start_idx = None
        for i in range(T - 1, -1, -1):
            self.robot.set_joint_qpos(path[i])
            if not self.robot.in_contact():
                start_idx = i
                break

        if start_idx is None:
            return None, False  # entire path in collision

        q_start = path[start_idx]
        q_goal = np.asarray(goal, dtype=np.float64)

        # ---- straight-line interpolation ----
        t = np.linspace(0.0, 1.0, n_wps)[:, None]
        rewired_segment = (1.0 - t) * q_start + t * q_goal

        interpolation_valid = True
        for wp in rewired_segment:
            self.robot.set_joint_qpos(wp)
            if self.robot.in_contact():
                interpolation_valid = False
                break

        # ---- fallback to RRTConnect if needed ----
        if not interpolation_valid:
            rewired_segment, _, _ = self.ompl_planner.plan(
                start=q_start,
                goal=q_goal,
                timeout=timeout,
                num_waypoints=n_wps,
                benchmark=True,
            )

        if rewired_segment is None or len(rewired_segment) == 0:
            return None, False

        rewired_segment = np.asarray(rewired_segment, dtype=np.float64)

        # ---- splice new tail ----
        # keep original path up to start_idx
        # append rewired segment excluding duplicate start
        new_path = np.vstack([path[: start_idx + 1], rewired_segment[1:]])

        return new_path, True

    def solve(self, sample, k=5, timeout=3.0):
        nn_query_start = time.perf_counter()
        nn_results = self.query_library_nn(self.lib_index, sample, n=k)
        nn_query_end = time.perf_counter()
        nn_time = nn_query_end - nn_query_start

        recovered_key = self.indexer.query_point(sample)
        if recovered_key is None:
            # raise RuntimeError("Failed to find key")
            return None, nn_time, False
        _, curr_goal = self.key_to_root[recovered_key]

        fix_start = time.perf_counter()
        fix_end = fix_start

        final_path = None
        success = False

        for (
            neighbor_key,
            (neighbor_goal, neighbor_path),
            neighbor_dist,
        ) in nn_results:

            # ---- Check global timeout BEFORE heavy work ----
            elapsed_fix = time.perf_counter() - fix_start
            if elapsed_fix > timeout:
                total_time = nn_time + elapsed_fix
                return None, total_time, False

            # 1) collision map + buffer
            waypoints_valid = self.collision_buffer(
                self.check_path_collision(neighbor_path), b=0
            )

            # ---- Remaining budget for this neighbor ----
            remaining = timeout - (time.perf_counter() - fix_start)
            if remaining <= 0:
                total_time = nn_time + (time.perf_counter() - fix_start)
                return None, total_time, False

            # 2) rewire internal segments
            rewired_path, ok = self.rewire_segments(
                neighbor_path,
                waypoints_valid,
                timeout=min(2.0, remaining),  # cap by remaining budget
                num_waypoints=20,
            )

            if not ok or rewired_path is None:
                continue

            # ---- Recompute remaining budget ----
            remaining = timeout - (time.perf_counter() - fix_start)
            if remaining <= 0:
                total_time = nn_time + (time.perf_counter() - fix_start)
                return None, total_time, False

            # 3) rewire tail to goal
            candidate_path, ok = self.rewire_to_goal(
                rewired_path,
                curr_goal,
                n_wps=20,
                timeout=min(1.0, remaining),  # cap by remaining budget
            )

            if not ok or candidate_path is None:
                continue

            final_path = candidate_path
            success = True
            break

        fix_end = time.perf_counter()
        fix_time = fix_end - fix_start
        total_time = nn_time + fix_time

        if not success:
            return None, total_time, False

        return final_path, total_time, True

    def wrap_pi(self, a):
        return (a + np.pi) % (2 * np.pi) - np.pi


def evaluate_graph(
    args,
    env: MujocoEnv,
    robot: MujocoRobot,
    folder,
    task_set,
    adaptations,
    num_samples,
    robot_name
):
    model, data = robot.model, robot.data
    home_qpos = robot.get_joint_qpos()
    ik_solver = get_ik_solver(robot, env_collision_geoms=env.env_details['collision_geoms'])
    # No task_paths/base-library file is loaded here.
    # Sampling and baseline goals are derived from the first adaptation's key_to_root.

    adapters = []
    key_to_roots = []
    root_paths_list = []

    # Setup grids for adaptations only.
    indexers = []

    rrtc_success = []
    rrtc_times = []
    rrtc_lengths = []

    library_success = []
    library_lengths = []
    library_times = []

    adaptation_success = {}
    adaptation_times = {}
    adaptation_lengths = {}

    lib_sizes = {}

    for adaptation in adaptations:

        adaptation_success[f"{adaptation}"] = []
        adaptation_times[f"{adaptation}"] = []
        adaptation_lengths[f"{adaptation}"] = []

        suffix = f"{args.ik}_{args.planner}_{adaptation}_{args.n_neighbors}"
        root_path = f"{folder}/root_paths_{suffix}.pkl"
        map_path = f"{folder}/key_to_root_{suffix}.pkl"

        root_data = pickle.load(open(root_path, "rb"))
        map_data = pickle.load(open(map_path, "rb"))
        key_to_roots.append(map_data)
        root_paths_list.append(root_data)

        print(f"{adaptation} library size: {len(root_data)}")
        lib_sizes[f"{adaptation}"] = len(root_data)

        indexers.append(BoxGrid(map_data))

        if adaptation == "linear":
            adapter = LinearAdapter(robot, ik_solver)
        elif adaptation == "grr":
            adapter = GRRAdapter(robot, ik_solver)
        elif adaptation == "dmp":
            adapter = DMPAdapter(robot, ik_solver)
        elif adaptation == "opt":
            adapter = TrajOptAdapter(robot, ik_solver)
        else:
            raise ValueError(f"Invalid adaptation method: {adaptation}")
        adapters.append(adapter)
    print(f"Length of key_to_roots: {len(key_to_roots)}")
    # input()
    rrtc_success = []
    rrtc_lengths = []
    rrtc_solve_times = []

    vamp_success = []
    vamp_lengths = []
    vamp_solve_times = []

    # Use the first available adaptation as the reference key/goal source.
    # This avoids loading the original full base-library task_paths file.
    reference_key_to_root = key_to_roots[0]
    reference_root_paths = root_paths_list[0]
    reference_indexer = indexers[0]
    solved_keys = list(reference_key_to_root.keys())

    solved_keys = [
        key
        for key, (root_id, goal_q) in reference_key_to_root.items()
        if (
            root_id is not None
            and 0 <= root_id < len(reference_root_paths)
            and reference_root_paths[root_id] is not None
            and len(reference_root_paths[root_id]) > 0
        )
    ]

    print(
        f"Usable reference keys: {len(solved_keys)} / "
        f"{len(reference_key_to_root)}"
    )

    print(f"Number of reference adaptation keys: {len(solved_keys)}")

    # indexer = BoxGrid(key_to_root)
    if isinstance(env, ShelfEnv) and isinstance(robot, FetchArm):
        ompl_planner = OMPLPlanner(robot, data, rrtc_range=0.1)
    elif isinstance(env, CageEnv) and isinstance(robot, FetchArm):
        ompl_planner = OMPLPlanner(robot, data, rrtc_range=0.1)
    else:
        ompl_planner = OMPLPlanner(robot, data)

    vamp_planner = VAMPPlanner(robot, env, data, robot_name=robot_name)

    # ompl_planner = OMPLPlanner(robot, data)

    # Building library baseline with N = reference adaptation key count.
    # This keeps the library baseline while avoiding task_paths_data_*.npy.
    # N = len(solved_keys)
    N = min(100_000, max(len(root_paths) for root_paths in root_paths_list))
    print("\n=== Building library ===")
    library = Library(
        N,
        env,
        robot,
        home_qpos,
        reference_key_to_root,
        reference_root_paths,
        solved_keys,
        data,
    )

    print(f"\n=== Evaluating baselines over {num_samples} samples===")
    num_tested = 0

    # pbar = tqdm(enumerate(solved_keys), total=len(solved_keys))
    # for i, key in enumerate(solved_keys):
    pbar = tqdm(range(num_samples), total=num_samples)
    while num_tested < num_samples:

        key_ind = np.random.randint(0, len(solved_keys))
        key = solved_keys[key_ind]

        # print(f"Key sampled: {key}")


        # sample = []
        # for lo, hi in key:
        #     x = np.random.uniform(lo, hi)
        #     x = np.nextafter(x, lo)
        #     sample.append(x)
        
        sample = sample_from_key(key)

        # print(f"Sample: {sample}")

        # env.move_cube_object(sample)
        if isinstance(env, MicrowaveEnv):
            pos_sample = sample[0:4]
            door_sample = sample[4]
            env.move_object(pos_sample)
            env.move_xml_joint("microwave_door_hinge", door_sample)
        else:
            env.move_object(sample)

        # env.move_object(sample)

        recovered_key = reference_indexer.query_point(sample)
        if recovered_key is None:
            continue
        # recovered_key = key
        
        _, key_goal = reference_key_to_root[recovered_key]
        
        # Task without a valid path
        if key_goal is None:
            continue

        robot.set_joint_qpos(key_goal)
        if robot.in_contact():
            # print("Bad sample. Skipping...")
            continue

        if robot.viewer is not None:
            # print(f"Key: {recovered_key}")
            # print(f"key goal: {key_goal}")
            
            robot.viewer.sync()
            # input()

        for adaptation_ind, adaptation in enumerate(adaptations):
            adapt_start = time.perf_counter()
            recovered_key_adapt = indexers[adaptation_ind].query_point(
                sample
            )
            if recovered_key_adapt is None:
                adaptation_success[adaptation].append(False)
                adapt_end = time.perf_counter()
                adaptation_times[adaptation].append(adapt_end - adapt_start)
                adaptation_lengths[adaptation].append(np.nan)
                continue
            root_id, curr_goal = key_to_roots[adaptation_ind][
                recovered_key_adapt
            ]

            if key != recovered_key_adapt:
                print(f"Original key: {key}")
                # print(sample)
                print(f"{adaptation} recovered key: {recovered_key_adapt}")
                # input()
                continue

            if root_id is None:
                continue

            curr_root = root_paths_list[adaptation_ind][root_id]
            adapted_path = adapters[adaptation_ind].adapt(curr_root, curr_goal)
            adapt_end = time.perf_counter()
            adapt_time = adapt_end - adapt_start

            curr_success = adapted_path is not None and len(adapted_path) > 0
            adaptation_success[adaptation].append(curr_success)
            adaptation_times[adaptation].append(adapt_time)
            if curr_success:
                adaptation_lengths[adaptation].append(traj_len(adapted_path))
            else:
                adaptation_lengths[adaptation].append(np.nan)

        # RRTConnect
        path, total_time, planning_time = ompl_planner.plan(
            start=home_qpos,
            goal=key_goal,
            timeout=3.0,
            smooth_path=False,
            num_waypoints=200,
            benchmark=True,
        )
        if path is None or len(path) == 0:
            # print(f"Planning failure for key: {key}")
            rrtc_success.append(False)
            rrtc_lengths.append(np.nan)
        else:
            rrtc_success.append(True)
            rrtc_lengths.append(traj_len(path))

        rrtc_solve_times.append(planning_time)

        # Library baseline
        library_path, library_time, lib_query_success = library.solve(sample)

        library_success.append(lib_query_success)
        library_times.append(library_time)

        if lib_query_success is True:
            library_lengths.append(traj_len(library_path))
        else:
            library_lengths.append(np.nan)

        # VAMP-RRTConnect baseline
        vamp_path, vamp_time, vamp_status = vamp_planner.plan(
            start=home_qpos,
            goal=key_goal,
            smooth_path=False,
            num_waypoints=200,
            benchmark=True
        )
        if vamp_path.size == 0:
            
            if vamp_status == "invalid_goal" or vamp_status == "invalid_start":
                # Resample if goal invalid for VAMP
                # print(f"vamp status: {vamp_status}")
                continue

            vamp_success.append(False)
            vamp_lengths.append(np.nan)
        else:
            vamp_success.append(True)
            vamp_lengths.append(traj_len(vamp_path))
        vamp_solve_times.append(vamp_time)

        num_tested += 1
        pbar.update(1)

    rrtc_success = np.array(rrtc_success)
    rrtc_times = np.array(rrtc_solve_times)
    rrtc_lengths = np.array(rrtc_lengths)

    library_success = np.array(library_success)
    library_times = np.array(library_times)
    library_lengths = np.array(library_lengths)

    vamp_success = np.array(vamp_success)
    vamp_times = np.array(vamp_solve_times)
    vamp_lengths = np.array(vamp_lengths)

    rrtc_success_rate = np.mean(rrtc_success) * 100
    library_success_rate = np.mean(library_success) * 100
    vamp_success_rate = np.mean(vamp_success) * 100

    rrtc_times_succ = rrtc_times[rrtc_success]
    library_times_succ = library_times[library_success]
    vamp_times_succ = vamp_times[vamp_success]

    # ---- RRTConnect ----
    mean_rrtc_time_ms = np.nanmean(rrtc_times_succ) * 1000
    std_rrtc_time_ms = np.nanstd(rrtc_times_succ, ddof=1) * 1000

    mean_rrtc_length = np.nanmean(rrtc_lengths)
    std_rrtc_length = np.nanstd(rrtc_lengths, ddof=1)

    # ---- Library baseline ----
    mean_library_time_ms = np.nanmean(library_times_succ) * 1000
    std_library_time_ms = np.nanstd(library_times_succ, ddof=1) * 1000

    mean_library_length = np.nanmean(library_lengths)
    std_library_length = np.nanstd(library_lengths, ddof=1)

        # ---- RRTConnect ----
    mean_vamp_time_ms = np.nanmean(vamp_times_succ) * 1000
    std_vamp_time_ms = np.nanstd(vamp_times_succ, ddof=1) * 1000

    mean_vamp_length = np.nanmean(vamp_lengths)
    std_vamp_length = np.nanstd(vamp_lengths, ddof=1)

    print("\n=== RRTConnect results ===")
    print(f"RRTConnect success rate: {rrtc_success_rate:.2f}%")
    print(
        f"Mean RRTConnect time: {mean_rrtc_time_ms:.3f} ± {std_rrtc_time_ms:.3f} ms"
    )
    print(
        f"Mean RRTConnect length: {mean_rrtc_length:.6f} ± {std_rrtc_length:.6f}"
    )

    print("\n=== Library baseline results ===")
    print(f"Library success rate: {library_success_rate:.2f}%")
    print(
        f"Mean library time: {mean_library_time_ms:.3f} ± {std_library_time_ms:.3f} ms"
    )
    print(
        f"Mean library length: {mean_library_length:.6f} ± {std_library_length:.6f}"
    )

    print("\n=== VAMP-RRTC results ===")
    print(f"VAMP-RRTC success rate: {vamp_success_rate:.2f}%")
    print(
        f"Mean VAMP-RRTC time: {mean_vamp_time_ms:.3f} ± {std_vamp_time_ms:.3f} ms"
    )
    print(
        f"Mean VAMP-RRTC length: {mean_vamp_length:.6f} ± {std_vamp_length:.6f}"
    )

    for adaptation in adaptations:

        times = np.asarray(adaptation_times[adaptation], dtype=float)
        succ = np.asarray(adaptation_success[adaptation], dtype=bool)

        times_succ = times[succ]

        mean_time_ms = np.nanmean(times_succ) * 1000
        std_time_ms = np.nanstd(times_succ, ddof=1) * 1000

        # times = adaptation_times[adaptation]
        lengths = adaptation_lengths[adaptation]

        # mean_time_ms = np.nanmean(times) * 1000
        # std_time_ms  = np.nanstd(times, ddof=1) * 1000

        mean_length = np.nanmean(lengths)
        std_length = np.nanstd(lengths, ddof=1)

        success_rate = np.mean(adaptation_success[adaptation]) * 100

        print(f"\n{adaptation} results")
        print(f"{adaptation} success rate: {success_rate:.2f}%")
        print(
            f"{adaptation} mean time: {mean_time_ms:.3f} ± {std_time_ms:.3f} ms"
        )
        print(
            f"{adaptation} mean length: {mean_length:.6f} ± {std_length:.6f}"
        )

    results_path = f"data/baseline_results_{args.robot}_{args.env}.npz"
    results = {
        "rrtc": {
            "success": rrtc_success,
            "times": rrtc_times,
            "lengths": rrtc_lengths,
        },
        "library": {
            "success": library_success,
            "times": library_times,
            "lengths": library_lengths,
        },
        "vamp": {
            "success": vamp_success,
            "times": vamp_times,
            "lengths": vamp_lengths,
        },
        "adaptations": {
            "success": adaptation_success,
            "times": adaptation_times,
            "lengths": adaptation_lengths,
        },
    }

    np.savez(results_path, results=results)


def main(args):
    """Evaluate path quality and query time for graph"""
    folder = get_data_folder(args.env, args.robot)
    suffix = f"{args.ik}_{args.planner}_{args.adaptation}_{args.n_neighbors}"

    # ---- output path for this run ----
    results_path = f"data/baseline_results_{args.robot}_{args.env}.npz"

    # ---- skip if already computed ----
    if os.path.exists(results_path) and not args.overwrite:
        print(f"[Skip] Results already exist: {results_path}")
        print("       Use --overwrite to re-run benchmarking.")
        return

    try:
        task_set = pickle.load(open(f"{folder}/task_set.pkl", "rb"))
        joint_goal_set = pickle.load(
            open(f"{folder}/joint_goal_set_{args.ik}.pkl", "rb")
        )

    except FileNotFoundError as e:
        print(e)
        print(f"One or more required files not found.")
        return

    IK_solved = sum(v is not None for v in joint_goal_set.values())
    print(f"Number of generated tasks: {len(task_set)}")
    print(f"Number of tasks solved by IK: {IK_solved}")
    print("Skipping load of full base-library task_paths file.")

    # Check if graph data exists
    root_path = f"{folder}/root_paths_{suffix}.pkl"
    map_path = f"{folder}/key_to_root_{suffix}.pkl"

    root_exists = os.path.exists(root_path)
    map_exists = os.path.exists(map_path)

    # if not root_exists or not map_exists:
    #     print("Compressed root paths "
    #         + f"with IK '{args.ik}', planner '{args.planner}', "
    #         + f"and adaptation '{args.adaptation}' does NOT exist. "
    #         + "Use condense_task_paths.py to generate it."
    #     )
    #     return

    # Load environment and robot
    # env, robot = load_env_and_robot(args.env, args.robot)

    adaptations_found = []
    for adaptation in ["grr", "opt", "dmp"]:
        suffix = f"{args.ik}_{args.planner}_{adaptation}_{args.n_neighbors}"
        root_path = f"{folder}/root_paths_{suffix}.pkl"
        map_path = f"{folder}/key_to_root_{suffix}.pkl"
        root_exists = os.path.exists(root_path)
        map_exists = os.path.exists(map_path)

        if root_exists and map_exists:
            adaptations_found.append(adaptation)
    if len(adaptations_found) == 0:
        raise FileNotFoundError(
            f"No adaptations found for problem: {args.robot} in {args.env}"
        )
    print(f"Adaptations found: {adaptations_found}")

    env_name = args.env
    robot_name = args.robot
    visualize = False

    if env_name == "table":
        env = TableEnv(robot_name, using_swept_volume=False)
    elif env_name == "box":
        env = BoxEnv(robot_name, using_swept_volume=False)
    elif env_name == "cage":
        env = CageEnv(robot_name, using_swept_volume=False)
    elif env_name == "shelf":
        env = ShelfEnv(robot_name, using_swept_volume=False)
    elif env_name == "free":
        env = FreeEnv(robot_name, using_swept_volume=False)
    elif env_name == "largeobj":
        env = LargeObjectEnv(robot_name, using_swept_volume=False)
    elif env_name == "microwave":
        env = MicrowaveEnv(robot_name, using_swept_volume=False)
    elif env_name == "allstable":
        env = AllStableEnv(robot_name, using_swept_volume=False)
    elif env_name == "real":
        env = RealEnv(robot_name, using_swept_volume=False)
    else:
        raise ValueError(f"Invalid environment: {env_name}")

     # Configure problem home pose
    NEW_ENVS = (LargeObjectEnv, AllStableEnv, MicrowaveEnv)
    home_pose_flag = "new" if isinstance(env, NEW_ENVS) else "default"


    model, data = env.model, env.data
    if robot_name == "panda":
        robot = Panda(model, data, visualize, home_pose=home_pose_flag)
    elif robot_name == "ur10":
        robot = UR10(model, data, visualize)
    elif robot_name == "fetch":
        robot = FetchArm(model, data, visualize, home_pose=home_pose_flag)
    else:
        raise ValueError(f"Invalid robot: {robot_name}")

    robot_pos = env.env_details['robot_pos']
    robot_quat = env.env_details['robot_quat']
    robot.teleport_base(pos=robot_pos, quat=robot_quat)

    # root_data = pickle.load(open(root_path, "rb"))
    # map_data = pickle.load(open(map_path, "rb"))

    # task_set_path = f"{folder}/task_set.pkl"
    # joint_goal_set_path = f"{folder}/joint_goal_set_neighbor.pkl"

    # task_set_data = pickle.load(open(task_set_path, "rb"))
    # joint_goal_set_data = pickle.load(open(joint_goal_set_path, "rb"))

    # print(f"Number of generated tasks: {len(task_set_data)}")
    # print(f"Number of solved IK: {len(joint_goal_set_data)}")

    # planning_results_path = f"{folder}/task_paths_results_neighbor_RRTConnect.npy"
    # planning_results = np.load(planning_results_path)

    num_samples = 1000
    evaluate_graph(
        args,
        env,
        robot,
        folder,
        task_set,
        adaptations_found,
        num_samples,
        robot_name
    )


def parse_arguments():
    parser = argparse.ArgumentParser()
    # envs
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--env", choices=[
            "table",
            "box",
            "cage",
            "shelf",
            "free",
            "largeobj",
            "microwave",
            "allstable",
            "real"
            ], default="table",
    )
    parser.add_argument(
        "--robot", choices=["panda", "ur10", "fetch"], default="panda"
    )
    parser.add_argument(
        "--ik", choices=["random", "neighbor", "grr"], default="neighbor"
    )
    parser.add_argument(
        "--planner", choices=["RRTConnect", "PRMstar","VAMP"], default="RRTConnect"
    )
    parser.add_argument(
        "--adaptation", choices=["linear", "grr", "dmp", "opt"], default="grr"
    )
    parser.add_argument("--n_neighbors", type=int, default=100)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
