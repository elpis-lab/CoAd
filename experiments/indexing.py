"""Shared task-region lookup and pose sampling for experiments."""

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
        self.has_face = len(first_key) == 5 and isinstance(first_key[0], str)

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
                    raise ValueError(f"Invalid AllStable key: {key}")

            numeric_keys = [key[1:] for key in self.keys_list]
        else:
            self.numeric_offset = 0
            self.ndim = len(first_key)
            self.has_door = self.ndim == 5
            numeric_keys = self.keys_list

        if self.ndim not in (4, 5):
            raise ValueError(f"Expected 4D or 5D keys, got {self.ndim} dimensions")

        self.keys_arr = np.asarray(
            numeric_keys,
            dtype=np.float64,
        )

        if self.keys_arr.ndim != 3 or self.keys_arr.shape[2] != 2:
            raise ValueError(
                f"Expected numeric key shape (N,D,2), " f"got {self.keys_arr.shape}"
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
            raise ValueError(f"Expected numeric key shape (N,D,2), got {keys.shape}")

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
            "x_to_ix": {float(value): index for index, value in enumerate(x_mins)},
            "y_to_iy": {float(value): index for index, value in enumerate(y_mins)},
            "nx": len(x_mins),
            "ny": len(y_mins),
            "nz": len(z_values),
            "nyaw": len(yaw_mins),
        }

        if keys.shape[1] == 5:
            door_mins = np.sort(np.unique(mins[:, 4]))
            grid.update(
                {
                    "door_global_min": float(np.min(mins[:, 4])),
                    "door_global_max": float(np.max(maxs[:, 4])),
                    "door_mins": door_mins,
                    "door_to_idoor": {
                        float(value): index for index, value in enumerate(door_mins)
                    },
                    "ndoor": len(door_mins),
                }
            )

        return grid

    def _build_face_grids(self):
        self.face_values = tuple(
            face
            for face in self.FACE_VALUES
            if any(key[0] == face for key in self.keys_list)
        )

        self._face_to_iface = {
            face: index for index, face in enumerate(self.face_values)
        }

        self.face_grids = {}

        for face in self.face_values:
            face_numeric_keys = [key[1:] for key in self.keys_list if key[0] == face]

            self.face_grids[face] = self._make_grid_data(face_numeric_keys)

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

        iz = int(np.argmin(np.abs(grid["z_values"] - z_val)))

        iyaw = int(np.argmin(np.abs(grid["yaw_mins"] - yaw_min)))

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
                    raise ValueError(f"Unknown AllStable face value: {face}")

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

            iz = int(np.argmin(np.abs(grid["z_values"] - z_val)))

            iyaw = int(np.argmin(np.abs(grid["yaw_mins"] - yaw_min)))

            if not np.isclose(
                grid["yaw_mins"][iyaw],
                yaw_min,
                atol=self.tol,
                rtol=0.0,
            ):
                raise RuntimeError(f"Could not match yaw minimum " f"{yaw_min:.17f}")

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
                previous_key = self.keys_list[self.index[indices]]

                raise RuntimeError(
                    f"Duplicate bin index {indices}: " f"{previous_key} and {key}"
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
        yaw = float(self._wrap_pi(numeric_sample[3]))

        # ---------- X ----------
        if not self._inside_nonperiodic(x, grid["x_global_min"], grid["x_global_max"]):
            return None

        ix = int(
            np.searchsorted(
                grid["x_mins"],
                x,
                side="right",
            )
            - 1
        )

        if ix < 0 or ix >= grid["nx"]:
            return None

        # ---------- Y ----------
        if not self._inside_nonperiodic(y, grid["y_global_min"], grid["y_global_max"]):
            return None

        iy = int(
            np.searchsorted(
                grid["y_mins"],
                y,
                side="right",
            )
            - 1
        )

        if iy < 0 or iy >= grid["ny"]:
            return None

        # ---------- Z ----------
        iz = int(np.argmin(np.abs(grid["z_values"] - z)))

        # ---------- YAW ----------
        iyaw = int(
            np.searchsorted(
                grid["yaw_mins"],
                yaw,
                side="right",
            )
            - 1
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

            if not self._inside_nonperiodic(
                door, grid["door_global_min"], grid["door_global_max"]
            ):
                return None

            idoor = int(
                np.searchsorted(
                    grid["door_mins"],
                    door,
                    side="right",
                )
                - 1
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

        if not self._inside_nonperiodic(
            numeric_sample[2], numeric_key[2][0], numeric_key[2][1]
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

        if abs(hi - lo) <= self.tol:
            return abs(value - lo) <= self.tol
        return value >= lo - self.tol and value < hi

    def _inside_yaw(self, value, lo, hi):
        if float(hi) - float(lo) >= 2 * np.pi - self.tol:
            return True
        value = float(self._wrap_pi(value))
        lo = float(self._wrap_pi(lo))
        hi = float(self._wrap_pi(hi))

        # A zero-width interval represents a fixed yaw.
        if np.isclose(lo, hi, atol=self.tol, rtol=0.0):
            return np.isclose(value, lo, atol=self.tol, rtol=0.0)

        if lo < hi:
            return value >= lo - self.tol and value < hi

        return value >= lo - self.tol or value < hi

    @staticmethod
    def _wrap_pi(angle):
        return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


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
    has_face = len(key) == 5 and isinstance(key[0], str)

    if has_face:
        face = key[0]
        numeric_key = key[1:]
    else:
        face = None
        numeric_key = key

    numeric_sample = []

    for dimension, (lo, hi) in enumerate(numeric_key):
        lo = float(lo)
        hi = float(hi)

        if dimension == 3 and hi < lo:
            hi += 2 * np.pi

        # Prevent sampling the upper boundary of a half-open bin.
        upper = np.nextafter(hi, lo)

        if lo == hi:
            value = lo
        else:
            value = np.random.uniform(lo, upper)

        if dimension == 3:
            value = float((value + np.pi) % (2 * np.pi) - np.pi)
        numeric_sample.append(value)

    if has_face:
        return [face, *numeric_sample]

    return numeric_sample
