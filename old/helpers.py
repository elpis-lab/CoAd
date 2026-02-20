import numpy as np
import json
import torch  # type: ignore
import os


def to_jsonable(x):
    if isinstance(x, tuple):
        return [to_jsonable(e) for e in x]
    if isinstance(x, list):
        return [to_jsonable(e) for e in x]
    return x  # int/float/str/bool/None etc.


def rpy_to_R(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    R = np.array(
        [
            [cp * cy, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [cp * sy, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )
    return R


def _wrap_pi(x):
    y = (np.asarray(x, dtype=float) + np.pi) % (2 * np.pi) - np.pi
    # enforce half-open: map any +π to −π (handles rare fp cases)
    return np.where(y == np.pi, -np.pi, y)


def _list_to_tuple(x):
    if isinstance(x, list):
        return tuple(_list_to_tuple(e) for e in x)
    return x


def _to_numpy_path(x, dtype):

    if x is None:
        return np.empty((0, 0), dtype=dtype)

    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(dtype, copy=False)

    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            return np.empty((0, 0), dtype=dtype)

        rows = []
        for xi in x:
            if isinstance(xi, torch.Tensor):
                xi = xi.detach().cpu().numpy()
            else:
                xi = np.asarray(xi)
            rows.append(xi)

        arr = np.stack(rows, axis=0)
        return arr.astype(dtype, copy=False)

    return np.asarray(x, dtype=dtype)


def to_numpy(x):
    # torch tensor (CPU or GPU)
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()

    # numpy array
    if isinstance(x, np.ndarray):
        # object arrays may contain tensors
        if x.dtype == object:
            return np.array([to_numpy(e) for e in x.tolist()])
        return x

    # python containers
    if isinstance(x, (list, tuple)):
        return np.array([to_numpy(e) for e in x])

    # scalars / fallback
    return np.array(x)


def quat_angle_diff(q1, q2):
    # q1, q2 (w, x, y, z)

    q1 = np.asarray(q1, dtype=float)
    q2 = np.asarray(q2, dtype=float)

    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)

    dot = np.dot(q1, q2)
    dot = abs(dot)
    dot = np.clip(dot, -1.0, 1.0)
    angle_rad = 2.0 * np.arccos(dot)
    return angle_rad


def load_store(prefix, mmap_data=None):
    data = np.load(
        f"{prefix}/data.npy", mmap_mode=("r" if mmap_data else None)
    )
    offsets = np.load(f"{prefix}/offsets.npy")
    keys = np.load(f"{prefix}/keys.npy")  # shape (K,4,2)
    return data, offsets, keys


def get_path_by_index(data, offsets, i):
    s, e = int(offsets[i]), int(offsets[i + 1])
    return data[s:e]  # O(1) view; touching L elements is O(L)


def pack_paths_from_dict(path_dict, *, path_dtype=np.float32, key_dtype=None):

    keys_list = list(path_dict.keys())
    # raw_paths = [np.asarray(path_dict[k], dtype=path_dtype) for k in keys_list]
    raw_paths = [
        _to_numpy_path(path_dict[k], dtype=path_dtype) for k in keys_list
    ]

    # 1) Find first non-empty path to infer D
    nonempty = [p for p in raw_paths if p.size > 0]
    if len(nonempty) == 0:
        # All paths are empty: define a consistent "empty" pack
        D = 0
        data = np.empty((0, 0), dtype=path_dtype)
        offsets = np.array([0], dtype=np.int64)  # only start=0
        keys_arr = (
            np.array(keys_list, dtype=key_dtype)
            if key_dtype
            else np.asarray(keys_list)
        )
        if not keys_arr.flags["C_CONTIGUOUS"]:
            keys_arr = np.ascontiguousarray(keys_arr)
        return data, offsets, keys_arr, D

    # Infer D from the first non-empty path
    if nonempty[0].ndim != 2:
        raise ValueError(
            f"Non-empty path must be 2D, got shape {nonempty[0].shape}"
        )
    D = nonempty[0].shape[1]

    # 2) Normalize all paths: empty -> (0, D), non-empty must be (Ni, D)
    paths = []
    for p in raw_paths:
        if p.size == 0:
            # represent empty path as (0, D)
            p_norm = np.empty((0, D), dtype=path_dtype)
        else:
            if p.ndim != 2 or p.shape[1] != D:
                raise ValueError(
                    f"each path must be (Ni, {D}), got shape {p.shape}"
                )
            p_norm = p
        paths.append(p_norm)

    # 3) Pack as before
    data = np.concatenate(paths, axis=0)  # (sum Ni, D)
    lengths = np.array([p.shape[0] for p in paths], np.int64)
    offsets = np.concatenate(([0], np.cumsum(lengths)))  # (K+1,)

    keys_arr = (
        np.array(keys_list, dtype=key_dtype)
        if key_dtype
        else np.asarray(keys_list)
    )
    if not keys_arr.flags["C_CONTIGUOUS"]:
        keys_arr = np.ascontiguousarray(keys_arr)

    return data, offsets, keys_arr, D


def save_paths_npy_numeric(
    prefix, path_dict, *, path_dtype=np.float32, key_dtype=None
):
    data, offsets, keys_arr, D = pack_paths_from_dict(
        path_dict, path_dtype=path_dtype, key_dtype=key_dtype
    )
    os.makedirs(prefix, exist_ok=True)

    np.save(f"{prefix}/data.npy", data)
    np.save(f"{prefix}/offsets.npy", offsets)
    np.save(f"{prefix}/keys.npy", keys_arr)
    meta = {
        "dim": int(D),
        "path_dtype": str(data.dtype),
        "key_dtype": str(keys_arr.dtype),
        "key_tail_shape": list(keys_arr.shape[1:]),
    }
    with open(f"{prefix}/meta.json", "w") as f:
        json.dump(meta, f)
    print(f"Saved paths to {prefix}")


def quat_mul(q, p):
    w, x, y, z = q
    W, X, Y, Z = p
    return np.array(
        [
            w * W - x * X - y * Y - z * Z,
            w * X + x * W + y * Z - z * Y,
            w * Y - x * Z + y * W + z * X,
            w * Z + x * Y - y * X + z * W,
        ]
    )


def quat_mul_t(q, p):  # both (...,4) wxyz
    w, x, y, z = q.unbind(-1)
    W, X, Y, Z = p.unbind(-1)
    return torch.stack(
        [
            w * W - x * X - y * Y - z * Z,
            w * X + x * W + y * Z - z * Y,
            w * Y - x * Z + y * W + z * X,
            w * Z + x * Y - y * X + z * W,
        ],
        dim=-1,
    )
