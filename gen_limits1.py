import genesis as gs
import trimesh  # type: ignore
import yaml
import numpy as np
from scipy.spatial.transform import Rotation as R
import time
import torch # type: ignore

import igl
import json
from functools import wraps

from planning import omplPlanner

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp, os

_original_signed_distance = igl.signed_distance

@wraps(_original_signed_distance)
def _signed_distance_3out(*args, **kwargs):
    res = _original_signed_distance(*args, **kwargs)
    # Some pyigl builds return (S, I, C), others (S, I, C, N), sometimes more
    if isinstance(res, (tuple, list)):
        if len(res) >= 3:
            return res[0], res[1], res[2]   # trim to 3
    return res  # fallback (unlikely, but don't crash here)

# **Monkey-patch**
igl.signed_distance = _signed_distance_3out


def rpy_to_R(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    R = np.array([
        [cp*cy,             cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
        [cp*sy,             sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
        [-sp,               cp*sr,             cp*cr]
    ])
    return R

def find_intersection(s1, s2):
    #x
    xmin = max(s1[0], s2[0])
    xmax = min(s1[1], s2[1])
    if xmin>xmax:
        return None
    xlim = [xmin, xmax]

    return xlim

def extract_intervals(x):
    # yields [lo, hi] from arbitrarily nested lists/tuples
    if isinstance(x, (list, tuple)):
        if len(x) == 2 and all(isinstance(v, (int, float)) for v in x):
            yield [x[0], x[1]]
        else:
            for child in x:
                yield from extract_intervals(child)

def merge_intervals(intervals, *, touch_as_overlap=True):
    # accept [lo, hi] lists/tuples; [] is treated as empty/no-op
    normed = list(extract_intervals(intervals))  # flatten/normalize
    if not normed:
        return []

    normed.sort()                 # sort by (lo, hi)
    merged = [normed[0][:]]       # copy
    for a, b in normed[1:]:
        c, d = merged[-1]
        cond = (a <= d) if touch_as_overlap else (a < d)
        if cond:
            merged[-1][1] = max(d, b)   # extend hi
        else:
            merged.append([a, b])
    return merged

def to_genesis_quat(quat):
    genesis_quat = [quat[3], quat[0], quat[1], quat[2]]
    return genesis_quat

def find_B0(Bw, Tew, Tw0):
    Te0 = Tw0 @ Tew
    yaw_ang = np.arctan2(Tw0[1, 0], Tw0[0, 0])

    #if (yaw_ang==0):
    B0 = np.array([
        [Bw[0, 0]+Tw0[0, 3], Bw[0, 1]+Tw0[0, 3]],
        [Bw[1, 0]+Tw0[1, 3], Bw[1, 1]+Tw0[1, 3]],
        [Bw[2, 0]+Tw0[2, 3], Bw[2, 1]+Tw0[2, 3]],
        [Bw[3, 0], Bw[3, 1]],
        [Bw[4, 0], Bw[4, 1]],
        [Bw[5, 0], Bw[5, 1]]
    ])

    return B0, Te0
    
def find_B0_intersection(B1_0, B2_0):
    #x
    xmin = max(B1_0[0, 0], B2_0[0, 0])
    xmax = min(B1_0[0, 1], B2_0[0, 1])
    if xmin <= xmax:
        xlim = [xmin, xmax]
    else:
        xlim = [None, None]
        return None
    
    ymin = max(B1_0[1, 0], B2_0[1, 0])
    ymax = min(B1_0[1, 1], B2_0[1, 1])
    if ymin <= ymax:
        ylim = [ymin, ymax]
    else:
        ylim = [None, None]
        return None

    zmin = max(B1_0[2, 0], B2_0[2, 0])
    zmax = min(B1_0[2, 1], B2_0[2, 1])
    if zmin <= zmax:
        zlim = [zmin, zmax]
    else:
        zlim = [None, None]
        return None
    
    rmin = max(B1_0[3, 0], B2_0[3, 0])
    rmax = min(B1_0[3, 1], B2_0[3, 1])
    if rmin <= rmax:
        rlim = [rmin, rmax]
    else:
        rlim = [None, None]
        return None
    
    pmin = max(B1_0[4, 0], B2_0[4, 0])
    pmax = min(B1_0[4, 1], B2_0[4, 1])
    if pmin <= pmax:
        plim = [pmin, pmax]
    else:
        plim = [None, None]
        return None
    
    yawmin = max(B1_0[5, 0], B2_0[5, 0])
    yawmax = min(B1_0[5, 1], B2_0[5, 1])
    if zmin <= zmax:
        yawlim = [yawmin, yawmax]
    else:
        yawlim = [None, None]
        return None

    B0_intersect = np.array([
        xlim, ylim, zlim, rlim, plim, yawlim
    ])

    return B0_intersect

def find_swept_volume(scene, object_dims, object_configs, volumes=None, env_idx=None):
    
    # Approximating the entire swept volume with a box primitive
    x = object_dims[0]
    y = object_dims[1]
    
    x_upper = object_configs[0][1]
    x_lower = object_configs[0][0]
    y_upper = object_configs[1][1]
    y_lower = object_configs[1][0]
    z = object_configs[2][0]

    R = round(0.5*np.sqrt(x**2 + y**2), 5)

    if (volumes is None):

        sv_b1 = scene.add_entity(
            gs.morphs.Box(
                pos = [(x_upper+x_lower)/2, (y_upper+y_lower)/2, z],
                size = [x_upper-x_lower+2*R, (y_upper-y_lower), object_dims[2]],
                quat = [1, 0, 0, 0],
                fixed = True,
                collision = True
            )
        )

        sv_b2 = scene.add_entity(
            gs.morphs.Box(
                pos = [(x_upper+x_lower)/2, (y_upper+y_lower)/2, z],
                size = [(x_upper-x_lower), (y_upper-y_lower)+2*R, object_dims[2]],
                quat = [1, 0, 0, 0],
                fixed = True,
                collision = True
            )
        )

        sv_c1 = scene.add_entity(
            gs.morphs.Cylinder(
                pos = [x_lower, y_lower, object_dims[2]/2],
                height = object_dims[2],
                radius = R,
                quat = [1, 0, 0, 0],
                fixed = True,
                collision=True
            )
        )

        sv_c2 = scene.add_entity(
            gs.morphs.Cylinder(
                pos = [x_lower, y_upper, object_dims[2]/2],
                height = object_dims[2],
                radius = R,
                quat = [1, 0, 0, 0],
                fixed = True,
                collision=True
            )
        )

        sv_c3 = scene.add_entity(
            gs.morphs.Cylinder(
                pos = [x_upper, y_lower, object_dims[2]/2],
                height = object_dims[2],
                radius = R,
                quat = [1, 0, 0, 0],
                fixed = True,
                collision=True
            )
        )

        sv_c4 = scene.add_entity(
            gs.morphs.Cylinder(
                pos = [x_upper, y_upper, object_dims[2]/2],
                height = object_dims[2],
                radius = R,
                quat = [1, 0, 0, 0],
                fixed = True,
                collision=True
            )
        )
    
    else:
        sv_b1 = volumes[0]
        sv_b2 = volumes[1]
        
        sv_c1 = volumes[2]
        sv_c2 = volumes[3]
        sv_c3 = volumes[4]
        sv_c4 = volumes[5]

        if env_idx is not None:
            sv_b1.set_pos(pos=[[(x_upper+x_lower)/2, (y_upper+y_lower)/2, z]], envs_idx=env_idx)
            sv_b2.set_pos(pos=[[(x_upper+x_lower)/2, (y_upper+y_lower)/2, z]], envs_idx=env_idx)
            sv_c1.set_pos(pos=[[x_lower, y_lower, object_dims[2]/2]], envs_idx=env_idx)
            sv_c2.set_pos(pos=[[x_lower, y_upper, object_dims[2]/2]], envs_idx=env_idx)
            sv_c3.set_pos(pos=[[x_upper, y_lower, object_dims[2]/2]], envs_idx=env_idx)
            sv_c4.set_pos(pos=[[x_upper, y_upper, object_dims[2]/2]], envs_idx=env_idx)
        else:
            sv_b1.set_pos(pos=[(x_upper+x_lower)/2, (y_upper+y_lower)/2, z], envs_idx=env_idx)
            sv_b2.set_pos(pos=[(x_upper+x_lower)/2, (y_upper+y_lower)/2, z], envs_idx=env_idx)
            sv_c1.set_pos(pos=[x_lower, y_lower, object_dims[2]/2], envs_idx=env_idx)
            sv_c2.set_pos(pos=[x_lower, y_upper, object_dims[2]/2], envs_idx=env_idx)
            sv_c3.set_pos(pos=[x_upper, y_lower, object_dims[2]/2], envs_idx=env_idx)
            sv_c4.set_pos(pos=[x_upper, y_upper, object_dims[2]/2], envs_idx=env_idx)

    
    return [sv_b1, sv_b2, sv_c1, sv_c2, sv_c3, sv_c4]

def encode_key(tup):
    # tup can be a tuple of tuples (of ints/floats/strings, etc.)
    # json.dumps turns tuples into lists; we only need a string here.
    return json.dumps(tup, separators=(",", ":"))  # compact, deterministic

def _list_to_tuple(x):
    if isinstance(x, list):
        return tuple(_list_to_tuple(e) for e in x)
    return x

def decode_key(s):
    # returns the original tuple-of-tuples structure
    return _list_to_tuple(json.loads(s))

def to_strkeyed(d):
    # dict[(tuple_of_tuples)] -> dict[str]
    return {encode_key(k): v for k, v in d.items()}

def from_strkeyed(d):
    # dict[str] -> dict[(tuple_of_tuples)]
    return {decode_key(k): v for k, v in d.items()}

def to_py(obj):
    # NumPy scalar
    try:
        import numpy as np
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        # no numpy or not available; fall back to duck-typing below
        pass

    # Duck-typing for arrays without importing numpy explicitly
    if hasattr(obj, "shape") and hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:
            pass

    if isinstance(obj, dict):
        return {k: to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_py(v) for v in obj]
    return obj

def pack_paths_from_dict(path_dict, *, path_dtype=np.float32, key_dtype=None):

    keys_list = list(path_dict.keys())
    raw_paths = [np.asarray(path_dict[k], dtype=path_dtype) for k in keys_list]

    # 1) Find first non-empty path to infer D
    nonempty = [p for p in raw_paths if p.size > 0]
    if len(nonempty) == 0:
        # All paths are empty: define a consistent "empty" pack
        D = 0
        data = np.empty((0, 0), dtype=path_dtype)
        offsets = np.array([0], dtype=np.int64)  # only start=0
        keys_arr = (np.array(keys_list, dtype=key_dtype) if key_dtype
                    else np.asarray(keys_list))
        if not keys_arr.flags["C_CONTIGUOUS"]:
            keys_arr = np.ascontiguousarray(keys_arr)
        return data, offsets, keys_arr, D

    # Infer D from the first non-empty path
    if nonempty[0].ndim != 2:
        raise ValueError(f"Non-empty path must be 2D, got shape {nonempty[0].shape}")
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
    data = np.concatenate(paths, axis=0)                      # (sum Ni, D)
    lengths = np.array([p.shape[0] for p in paths], np.int64)
    offsets = np.concatenate(([0], np.cumsum(lengths)))       # (K+1,)

    keys_arr = np.array(keys_list, dtype=key_dtype) if key_dtype else np.asarray(keys_list)
    if not keys_arr.flags["C_CONTIGUOUS"]:
        keys_arr = np.ascontiguousarray(keys_arr)

    return data, offsets, keys_arr, D

def save_paths_npy_numeric(prefix, path_dict, *, path_dtype=np.float32, key_dtype=None):
    data, offsets, keys_arr, D = pack_paths_from_dict(path_dict, path_dtype=path_dtype, key_dtype=key_dtype)
    np.save(f"{prefix}.data.npy", data)
    np.save(f"{prefix}.offsets.npy", offsets)
    np.save(f"{prefix}.keys.npy", keys_arr)
    meta = {
        "dim": int(D),
        "path_dtype": str(data.dtype),
        "key_dtype": str(keys_arr.dtype),
        "key_tail_shape": list(keys_arr.shape[1:]),
    }
    with open(f"{prefix}.meta.json", "w") as f:
        json.dump(meta, f)

def rmin_rmax_from_square_corners(tw1, tw2, nominal_pose=[0, 0, 0]):
    x_min = tw1[0]
    y_min = tw1[1]
    x_max = tw2[0]
    y_max = tw2[1]

    if (x_min<=nominal_pose[0]<=x_max) and (y_min<=nominal_pose[1]<=y_max):
        r_min = 0.0
    else:
        nearest_x = np.clip(nominal_pose[0], x_min, x_max)
        nearest_y = np.clip(nominal_pose[1], y_min, y_max)
        r_min = np.sqrt(nearest_x**2 + nearest_y**2)

    corners = np.array([
        [x_min, y_min],
        [x_min, y_max],
        [x_max, y_min],
        [x_max, y_max],
    ])
    nominal_pose_mat = np.array([
        nominal_pose[0:2],
        nominal_pose[0:2],
        nominal_pose[0:2],
        nominal_pose[0:2],
    ])

    r2 = (corners**2).sum(axis=1)
    r_max = np.sqrt(r2.max())

    return r_min, r_max       

# Parallel OMPL stuff

# worker globals
_SCENE = None
_ROBOT = None
_PLANNER = None
_VOLUMES = None
_CACHE = None
_KEY = None
_WORKER_ID = None
_OBJECT_SIZE = [0.06, 0.06, 0.2]

def _init_worker(genesis_cfg: dict, planner_cfg: dict):

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    global _SCENE, _ROBOT, _PLANNER, _VOLUMES, _CACHE, _KEY, _WORKER_ID, _OBJECT_SIZE
    
    _CACHE = {}
    _KEY = genesis_cfg.get("key")
    _WORKER_ID = f"{mp.current_process().name}:{os.getpid()}"

    perf_mode = genesis_cfg.get("performance_mode")

    import genesis as gs

    gs.init(backend=gs.cpu, performance_mode=perf_mode)
    _SCENE = gs.Scene(show_viewer=False, show_FPS=False)
    robotURDF = "robowflex_resources/panda/urdf/panda.urdf"
    robot_morph = gs.morphs.URDF(
        file=robotURDF,
        pos=tuple(genesis_cfg.get("robot_base_position", (0,0,0))),
        quat=tuple(genesis_cfg.get("robot_base_orientation", (1,0,0,0))),
        fixed=True
    )
    _ROBOT = _SCENE.add_entity(robot_morph)

    _VOLUMES = find_swept_volume(
        _SCENE,
        object_dims=_OBJECT_SIZE,
        object_configs=_KEY,
        volumes=None,
        env_idx=None
    )

    _SCENE.add_entity(gs.morphs.Plane())
    _SCENE.build()

    from planning import omplPlanner
    _PLANNER = omplPlanner(_ROBOT)

def _plan_one(task):
    import numpy as np
    global _SCENE, _ROBOT, _PLANNER, _VOLUMES, _WORKER_ID

    key, q_start, q_goal, timeout, num_waypoints, key_idx = task

    _VOLUMES = find_swept_volume(
        _SCENE,
        object_dims=_OBJECT_SIZE,
        object_configs=key,
        volumes=_VOLUMES,
        env_idx=None
    )

    path = _PLANNER.omplPlan(
        qpos_goal=q_goal,
        qpos_start=q_start,
        timeout=float(timeout),
        num_waypoints=int(num_waypoints),
        env_idx=None,
        planner="RRTConnect",
        ignore_collision=False,
        ignore_joint_limit=False
    )

    if not path:
        print(f"Planning failed for {key_idx}. Empty path.")
        path_np = np.asarray([]) 
    elif (path[-1].tolist()!=q_goal.tolist()):
        print(f"Planning failed for {key_idx}. Incomplete path")
        path_np = np.asarray([])
    else:
        path_np = [
            w.detach().cpu().numpy().astype(np.float32) if hasattr(w, "detach") else np.asarray(w, dtype=np.float32)
            for w in path
        ]

    #print(f"[{_WORKER_ID}] solved key={key_idx}", flush=True)

    return key, path_np 

def start_pool(num_workers, genesis_kwargs, planner_kwargs):
    mp.set_start_method("spawn", force=True)
    for v in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"]:
        os.environ.setdefault(v, "1")
    return ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=mp.get_context("spawn"),
        initializer=_init_worker,
        initargs=(genesis_kwargs, planner_kwargs)
    )

def submit_plans(executor, jobs):
    """
    jobs: iterable of tuples (key, q_start, q_goal, timeout, num_waypoints, env_idx)
    returns: dict {key: path}
    """
    futures = [executor.submit(_plan_one, job) for job in jobs]
    results = {}
    for f in as_completed(futures):
        k, path = f.result()  # raise if worker errored so you see it
        results[k] = path
    return results

def main():
    #gs.init(backend=gs.cpu, logging_level='error')

    parallelize = True
    perf_mode = False
    cpu = 8
    n_envs = 6


    gs.init(backend=gs.gpu, performance_mode=perf_mode)

    scene = gs.Scene(show_viewer=(not parallelize), show_FPS=False)


    start = time.perf_counter()

    robotURDF = "/home/adilshiyas/ros1/src/robowflex_resources/panda/urdf/panda.urdf"
    robotURDF = "robowflex_resources/panda/urdf/panda.urdf"

    robot_base_position = (0, 0, 0)
    robot_base_orientation = (1, 0, 0, 0)

    robot = scene.add_entity(
        gs.morphs.URDF(
            file=robotURDF, 
            pos=robot_base_position, 
            quat=robot_base_orientation,
            fixed=True,
            #convexify=False,
        )
    )

    # Placing the object anywhere on the XY plane within the reachable workspace
    #object_radius = 0.04
    #object_height = 0.2
    object_size = [0.06, 0.06, 0.2]

    object_position = [0.4, 0.0, 0.1]
    object_quat = [0, 0, 0, 1]
    object_roll, object_pitch, object_yaw = R.from_quat(object_quat).as_euler('xyz', degrees=False)

    nominal_object_pose = [0, 0, 0+object_size[2]/2]
    object_position = nominal_object_pose

    robot_clearance = 0.3
    reachable_ws = 0.306

    object_dist = [0.1, 0.1, 0, np.pi]
    object_dist = [reachable_ws, reachable_ws, 0, 0*np.pi]

    object_dist_check = np.sign(np.array(object_dist))

    object_upper = [
        round(object_position[0]+object_dist[0], 5),
        round(object_position[1]+object_dist[1], 5),
        round(object_position[2]+object_dist[2], 5),
        round(object_yaw + object_dist[3], 5)
    ]
    object_lower = [
        round(object_position[0]-object_dist[0], 5),
        round(object_position[1]-object_dist[1], 5),
        round(object_position[2]-object_dist[2], 5),
        round(object_yaw - object_dist[3], 5)
    ]

    is_covered = False
    dist_to_cover = [[object_lower[0], object_upper[0]], [object_lower[1], object_upper[1]], [object_lower[2], object_upper[2]]]
    #print(dist_to_cover)

    print("Dist_to_cover: ", dist_to_cover)

    dims = []
    for i in range(len(dist_to_cover)):
        if dist_to_cover[i][0]!=dist_to_cover[i][1]:
            dims.append(1)
        else:
            dims.append(0)
    print(dims)

    # Making a TSR for just the nominal pose for now

    iTSR_set = {}
    curr_intervals = [[], [], []]
    dist_covered = [[], [], []]
    dist_covered_pivot = [[], [], []]

    Tw0 = np.eye(4)
    Tw0[0, 3] = object_position[0]
    Tw0[1, 3] = object_position[1]
    Tw0[2, 3] = object_position[2]

    ee_z_offset = 0.075
    Tew = np.eye(4)
    Tew[1, 1] = -1
    Tew[2, 2] = -1
    Tew[2, 3] = ee_z_offset + object_position[2]/2

    s_f = 0.042
    del_geom = s_f
    del_geom_x = s_f - (object_size[0]/2)
    del_geom_y = s_f - (object_size[1]/2)
    #del_geom = 0.01

    yaw_buffer = 6*(np.pi/180)


    Bw = np.array([
        [-del_geom_x, del_geom_x],
        [-del_geom_y, del_geom_y],
        [0, 0],
        [0, 0],
        [0, 0],
        [0-yaw_buffer, 0+yaw_buffer]
    ])

    # Approximating the intersection of all rotated TSR's with a conservative TSR
    half_side = 0.5*np.sqrt(2)*0.5*min(2*del_geom_x, 2*del_geom_y)

    del_Bw = np.array([
        (Bw[0, 1]-Bw[0, 0])/2, 
        (Bw[1, 1]-Bw[1, 0])/2, 
        (Bw[2, 1]-Bw[2, 0])/2,
        (Bw[5, 1]-Bw[5, 0])/2
    ])

    del_Bw = object_dist_check*del_Bw
    alpha = 0.95
    Tw2_w1 = alpha*del_Bw
    tw2_w1 = Tw2_w1[0:3]

    yaw_tw2_w1 = np.array([alpha*half_side, alpha*half_side, tw2_w1[2]])
    yaw_tw2_w1 = yaw_tw2_w1*object_dist_check[0:3]

    print("tw2_w1: ", tw2_w1)
    print("yaw_tw2_w1: ", yaw_tw2_w1)

    if (dims[0]==1 and dims[1]==1):
        yaw_tw2_w1_x = np.array([alpha*half_side, 0, 0])
        yaw_tw2_w1_y = np.array([0, alpha*half_side, 0])

    x_covered = True

    yaw_is_covered = False
    yaw_to_cover = [round(-object_dist[3], 5), round(object_dist[3], 5)]

    yaw_iTSR_set = {}
    yaw_covered = []
    yaw_key = []

    while (yaw_is_covered is False):
        first_pos = [
            object_position[0]-object_dist[0],
            object_position[1]-object_dist[1],
            object_position[2]-object_dist[2]
        ]
        
        if yaw_iTSR_set == {}:
            yaw_1 = round(-object_dist[3], 5)       
        else:
            prev_Tw2_0 = yaw_iTSR_set[yaw_key][2]
            yaw_1 = round(np.arctan2(prev_Tw2_0[1, 0], prev_Tw2_0[0, 0]), 5)
        
        Tw1_0 = np.eye(4)
        Tw1_0[0, 0] = np.cos(yaw_1)
        Tw1_0[0, 1] = -np.sin(yaw_1)
        Tw1_0[1, 0] = np.sin(yaw_1)
        Tw1_0[1, 1] = np.cos(yaw_1)
            
        Tw1_0[0, 3] = first_pos[0]
        Tw1_0[1, 3] = first_pos[1]
        Tw1_0[2, 3] = first_pos[2]

        tw2_0 = Tw1_0[:3, 3] #Same position as Tw1
        #yaw_1 = np.arctan2(Tw1_0[1, 0], Tw1_0[0, 0])
        yaw_2 = round(yaw_1 + Tw2_w1[3], 5)

        Tw2_0 = Tw1_0
        Tw2_0[0, 0] = np.cos(yaw_2)
        Tw2_0[0, 1] = -np.sin(yaw_2)
        Tw2_0[1, 0] = np.sin(yaw_2)
        Tw2_0[1, 1] = np.cos(yaw_2)

        B12_yaw_intersect = np.array([
            [tw2_0[0] - half_side, tw2_0[0] + half_side],
            [tw2_0[1] - half_side, tw2_0[1] + half_side],
            [0, 0],
            [0, 0],
            [0, 0],
            [yaw_2 - yaw_buffer, yaw_1 + yaw_buffer]
        ])

    #print(B12_yaw_intersect)
        curr_yaw_intervals = [yaw_1, yaw_2]
        yaw_covered.append(curr_yaw_intervals)
        yaw_covered = merge_intervals(yaw_covered)
        #print(yaw_covered)
        #print(yaw_to_cover)
        
        yaw_cover_check = find_intersection(yaw_covered[0], yaw_to_cover) == yaw_to_cover
        yaw_is_covered = yaw_cover_check

        #yaw_key = tuple(yaw_covered)
        yaw_key = tuple(curr_yaw_intervals)
        #yaw_iTSR_set[yaw_key] = [B12_yaw_intersect, Tw1_0, Tw2_0]
        yaw_iTSR_set[yaw_key] = [B12_yaw_intersect, Tw1_0, Tw2_0] 

        #print(yaw_key)
        #print(B12_yaw_intersect)
        

    print(f"Yaw covered: {yaw_to_cover}")
    print(f"Length of yaw TSR set: {len(yaw_iTSR_set)}")

    iterno = 0

    for curr_yaw_interval in yaw_iTSR_set:
        #print(curr_yaw_interval)
        curr_yaw_iTSR = yaw_iTSR_set[curr_yaw_interval][0]
        yaw_1 = curr_yaw_interval[0]
        yaw_2 = curr_yaw_interval[1]
        
        curr_yaw_iTSR_set = {}
        curr_yaw_dist_covered = [[], [], []]
        curr_yaw_is_covered = False

        while (curr_yaw_is_covered is False):
            if curr_yaw_iTSR_set == {}:
                Tw1_0 = np.eye(4)
                Tw1_0[0, 3] = object_position[0]-object_dist[0]
                Tw1_0[1, 3] = object_position[1]-object_dist[1]
                Tw1_0[2, 3] = object_position[2]-object_dist[2]
            
            else:

                if (dims[0]==1 and dims[1]==1):
                    if (x_covered):
                        Tw1_0 = x_pivot
                    else:
                        Tw1_0 = curr_yaw_iTSR_set[curr_yaw_key][3]
                else:
                    
                    Tw1_0 = curr_yaw_iTSR_set[curr_yaw_key][2]

            tw1_0 = Tw1_0[:3, 3] 
            tw2_0 = tw1_0 + yaw_tw2_w1

            Tw2_0 = np.eye(4)
            Tw2_0[:3, 3] = tw2_0
        
            B1_0 = np.array([
                [tw1_0[0]- half_side, tw1_0[0]+ half_side],
                [tw1_0[1]- half_side, tw1_0[1]+ half_side],
                [tw1_0[2]+Bw[2, 0], tw1_0[2]+Bw[2, 1]],
                [Bw[3, 0], Bw[3, 1]],
                [Bw[4, 0], Bw[4, 1]],
                [yaw_2 - yaw_buffer, yaw_1 + yaw_buffer]
            ])

            B2_0 = np.array([
                [tw2_0[0]- half_side, tw2_0[0]+ half_side],
                [tw2_0[1]- half_side, tw2_0[1]+ half_side],
                [tw2_0[2]+Bw[2, 0], tw2_0[2]+Bw[2, 1]],
                [Bw[3, 0], Bw[3, 1]],
                [Bw[4, 0], Bw[4, 1]],
                [yaw_2 - yaw_buffer, yaw_1 + yaw_buffer]
            ])

            B12_intersect = find_B0_intersection(B1_0, B2_0)

            if (dims[0]==1 and dims[1]==1):
                tw3_0 = tw1_0 + yaw_tw2_w1_x #x translation
                tw4_0 = tw1_0 + yaw_tw2_w1_y #y translation

                Tw3_0 = np.eye(4)
                Tw3_0[:3, 3] = tw3_0
                Tw4_0 = np.eye(4)
                Tw4_0[:3, 3] = tw4_0

                B3_0 = np.array([
                    [tw3_0[0]- half_side, tw3_0[0]+ half_side],
                    [tw3_0[1]- half_side, tw3_0[1]+ half_side],
                    [tw3_0[2]+Bw[2, 0], tw3_0[2]+Bw[2, 1]],
                    [Bw[3, 0], Bw[3, 1]],
                    [Bw[4, 0], Bw[4, 1]],
                    [yaw_2 - yaw_buffer, yaw_1 + yaw_buffer]
                ])

                B4_0 = np.array([
                    [tw4_0[0]- half_side, tw4_0[0]+ half_side],
                    [tw4_0[1]- half_side, tw4_0[1]+ half_side],
                    [tw4_0[2]+Bw[2, 0], tw4_0[2]+Bw[2, 1]],
                    [Bw[3, 0], Bw[3, 1]],
                    [Bw[4, 0], Bw[4, 1]],
                    [yaw_2 - yaw_buffer, yaw_1 + yaw_buffer]
                ])

                B34_intersect = find_B0_intersection(B3_0, B4_0)    
                B12_intersect = find_B0_intersection(B34_intersect, B12_intersect)

                if (x_covered):
                    x_pivot = Tw4_0
                    dist_covered_pivot = [[], [], []]
                    x_covered = False

            curr_intervals = [
                [round(tw1_0[0], 5), round(tw2_0[0], 5)],
                [round(tw1_0[1], 5), round(tw2_0[1], 5)], 
                [round(tw1_0[2], 5), round(tw2_0[2], 5)],
                [round(yaw_1, 5), round(yaw_2, 5)]
            ]

            if(dims[0]==1 and dims[1]==1):
                
                dist_covered_pivot[0].append(curr_intervals[0])
                dist_covered_pivot[1].append(curr_intervals[1])
                dist_covered_pivot[2].append(curr_intervals[2])

                dist_covered_pivot = [
                    merge_intervals(dist_covered_pivot[0]),
                    merge_intervals(dist_covered_pivot[1]),
                    merge_intervals(dist_covered_pivot[2])    
                ]

                x_covered = find_intersection(dist_to_cover[0], dist_covered_pivot[0][0])==dist_to_cover[0]

                if (x_covered):
                    curr_yaw_dist_covered[0].append(dist_covered_pivot[0])
                    curr_yaw_dist_covered[1].append(dist_covered_pivot[1])
                    curr_yaw_dist_covered[2].append(dist_covered_pivot[2])
                    #print(curr_yaw_dist_covered)
                    curr_yaw_dist_covered = [
                        merge_intervals(curr_yaw_dist_covered[0]),
                        merge_intervals(curr_yaw_dist_covered[1]),
                        merge_intervals(curr_yaw_dist_covered[2])    
                    ]
                    #print(curr_yaw_dist_covered)

                    curr_yaw_cover_check = [
                        find_intersection(dist_to_cover[0], curr_yaw_dist_covered[0][0]) == dist_to_cover[0],
                        find_intersection(dist_to_cover[1], curr_yaw_dist_covered[1][0]) == dist_to_cover[1],
                        find_intersection(dist_to_cover[2], curr_yaw_dist_covered[2][0]) == dist_to_cover[2]
                    ]
                    curr_yaw_is_covered = all(curr_yaw_cover_check)
                else:
                    curr_yaw_is_covered = False

            else:

                curr_yaw_dist_covered[0].append(curr_intervals[0])
                curr_yaw_dist_covered[1].append(curr_intervals[1])
                curr_yaw_dist_covered[2].append(curr_intervals[2]) 

                curr_yaw_dist_covered = [
                    merge_intervals(curr_yaw_dist_covered[0]),
                    merge_intervals(curr_yaw_dist_covered[1]),
                    merge_intervals(curr_yaw_dist_covered[2])
                ]

                curr_yaw_cover_check = [
                    find_intersection(dist_to_cover[0], curr_yaw_dist_covered[0][0]) == dist_to_cover[0],
                    find_intersection(dist_to_cover[1], curr_yaw_dist_covered[1][0]) == dist_to_cover[1],
                    find_intersection(dist_to_cover[2], curr_yaw_dist_covered[2][0]) == dist_to_cover[2]
                ]

                curr_yaw_is_covered = all(curr_yaw_cover_check)

            #print(f"Current yaw coverage check: {curr_yaw_is_covered}")

            key = tuple(tuple(row) for row in curr_intervals)
            curr_yaw_key = tuple(tuple(row) for row in curr_intervals[0:3])

            tw1_rvec = np.sqrt((tw1_0[0]-nominal_object_pose[0])**2 + (tw1_0[1]-nominal_object_pose[1])**2)

            rmin, rmax = rmin_rmax_from_square_corners(tw1_0, tw2_0)
            in_sample_space = (rmin <= reachable_ws) and (rmax >= robot_clearance)

            #if(tw1_rvec<=reachable_ws and tw1_rvec>=robot_clearance):
            #    in_sample_space = True
            #else:
            #    in_sample_space = False

            if (dims[0]==1 and dims[1]==1):
                curr_yaw_iTSR_set[curr_yaw_key] = [B12_intersect, Tw1_0, Tw2_0, Tw3_0, Tw4_0]
                if (in_sample_space):
                    iTSR_set[key] = [B12_intersect, Tw1_0, Tw2_0, Tw3_0, Tw4_0]
                #else:
                    #print("Current iTSR not in sample space. Skipping...")
                    #print(tw1_rvec)
            else:
                curr_yaw_iTSR_set[curr_yaw_key] = [B12_intersect, Tw1_0, Tw2_0]
                if (in_sample_space):
                    iTSR_set[key] = [B12_intersect, Tw1_0, Tw2_0]
                #else:
                    #print("Current iTSR not in sample space. Skipping...")
                    #print(tw1_rvec)
            
            '''
            if (in_sample_space):
                if (dims[0]==1 and dims[1]==1):
                    #print(f"t1: {tw1_0}")
                    #print(f"t2: {tw2_0}")
                    #print(f"t3: {tw3_0}")
                    #print(f"t4: {tw4_0}")
                    curr_yaw_iTSR_set[curr_yaw_key] = [B12_intersect, Tw1_0, Tw2_0, Tw3_0, Tw4_0]
                    iTSR_set[key] = [B12_intersect, Tw1_0, Tw2_0, Tw3_0, Tw4_0]
                else:
                    curr_yaw_iTSR_set[curr_yaw_key] = [B12_intersect, Tw1_0, Tw2_0]
                    iTSR_set[key] = [B12_intersect, Tw1_0, Tw2_0]
            else:
                print("Current iTSR not in sample space. Skipping...")
                print(tw1_rvec)
            '''

            iterno += 1

            #print(f"Current pivot dist covered: {dist_covered_pivot}")
            #print(f"Current yaw dist covered: {curr_yaw_dist_covered}")
        print(f"Yaw interval covered: {yaw_1} to {yaw_2}")
        
            #break

    print(f"Iterations: {iterno}")
    print(f"Number of iTSRs: {len(iTSR_set)}")
    print("Object uncertainty covered.")

    #return

    sorted_iTSR_items = sorted(iTSR_set.items(), key=lambda kv: kv[0])
    packed_iTSR = {encode_key(k): to_py(v) for k, v in sorted_iTSR_items}
    with open("iTSR_set.json", "w") as f:
            json.dump(packed_iTSR, f, indent=2)   

    print("JSON file saved.")



    homePos = [0, -0.785, 0, -2.356, 0, 1.571, 0.785, 0.065, 0.065]

    iTSR_paths = {}

    keylist = list(iTSR_set)
    #n_envs = int(len(iTSR_set)/len(yaw_iTSR_set))
    #n_envs = int(len(yaw_iTSR_set)/2)
    print(f"Creating {n_envs} environments")

    ik_failures = 0

    planner = omplPlanner(robot)
    planning_failures = 0

    if(parallelize):
        # Choose number of workers
        
        NUM_WORKERS = cpu -2
        current_keys = keylist[0:n_envs]

        initial_key_for_entity = current_keys[0]  # any representative key is fine
        genesis_cfg = {
            "robot_base_position": (0, 0, 0),
            "robot_base_orientation": (1, 0, 0, 0),  # verify your quat ordering
            "key": initial_key_for_entity,
            "performance_mode": perf_mode
        }
        planner_cfg = {
            # put static OMPL knobs here if you need (you already pass most per-call)
        }
        executor = start_pool(NUM_WORKERS, genesis_cfg, planner_cfg)

        try:
            for i, key in enumerate(iTSR_set):
                
                if i%n_envs!=0:
                    continue
                
                if(i==0):
                    #print(key)

                    volumes = find_swept_volume(scene, object_size, key, volumes=None, env_idx=None)
                    plane = scene.add_entity(gs.morphs.Plane())
                    scene.build(n_envs=n_envs, env_spacing=(1.0, 1.0))
                    
                    envs_to_use = min(n_envs, len(range(i, len(iTSR_set))))

                    for env_idx in range(envs_to_use):
                        volumes = find_swept_volume(scene, object_size, keylist[i+env_idx], volumes=volumes, env_idx=[env_idx])  

                else:
                    #volumes = find_swept_volume(object_size, key, volumes=volumes, env_idx=None)  
                    
                    envs_to_use = min(n_envs, len(range(i, len(iTSR_set))))

                    for env_idx in range(envs_to_use):
                        volumes = find_swept_volume(scene, object_size, keylist[i+env_idx], volumes=volumes, env_idx=[env_idx])  

                #if(envs_to_use==len(range(i, len(iTSR_set)))):
                #    print("USING REMAINING KEYS, LESS THAN N_ENVS")
                #    print(envs_to_use)

                robot.set_dofs_position(
                    torch.tile(
                        torch.tensor(homePos, device=gs.device), (n_envs, 1)
                    ),
                )
                scene.step()

                rng = np.random.default_rng()
                
                iTSR = iTSR_set[key][0]

                #current_keys = keylist[i : i+n_envs]
                current_keys = keylist[i : i+envs_to_use]
                iTSR_batch = [
                    iTSR_set[curr_key][0] for curr_key in current_keys
                ]
                iTSR = np.stack(iTSR_batch, axis=0)

                B = iTSR.shape[0]
                num_links = robot.n_links
                ee_link1_idx = num_links - 2
                ee_link2_idx = num_links - 1

                #seeds = np.tile(homePos, (B,1)).astype(np.float32)

                alive = np.ones(B, dtype=bool)     
                attempts = np.zeros(B, dtype=np.int32)

                sample_pos = np.zeros((B, 3), dtype=np.float32)
                sample_quat = np.zeros((B, 4), dtype=np.float32)
                q_batch = np.zeros((B, robot.n_dofs), dtype=np.float32)
                ik_ok = np.zeros(B, dtype=bool)

                have_target = np.zeros(B, dtype=bool)
                target_pos = np.zeros((B, 3), np.float32)
                target_quat = np.zeros((B, 4), np.float32)
                seeds = np.tile(homePos, (B, 1)).astype(np.float32)
                seeds = np.tile(homePos, (n_envs, 1)).astype(np.float32)
                #seeds = torch.empty((B, 9), dtype=q.dtype, device=q.device)
                max_local = 10

                #inCollision = True

                ik_attempt = 0
                curr_ik_failures = 0
                ik_max_attempts = 1000
                q_sol = np.zeros_like(q_batch, dtype=np.float32)

                while alive.any():
                
                    if ik_attempt==0:
                        old_q = np.zeros((B, robot.n_dofs), dtype=np.float32)
                    else:
                        old_q = q_batch

                
                    idx = np.nonzero(alive)[0]
                    #idx = np.array(list(range(n_envs)))
                    idx = np.array(list(range(envs_to_use)))
                    #print(idx)
                    B_temp = idx.size
                    attempts[idx] += 1

                    # Sample only envs without a fixed target

                    new_idx = idx[~have_target[idx]]
                    new_idx = idx
                    if new_idx.size:
                        #print("sampling new points in TSRs")
                        lo = np.minimum(iTSR[new_idx, :, 0], iTSR[new_idx, :, 1]).astype(np.float32)     # (b, 6)
                        hi = np.maximum(iTSR[new_idx, :, 0], iTSR[new_idx, :, 1]).astype(np.float32)     # (b, 6)
                    
                        #lo = np.minimum(iTSR[idx, :, 0], iTSR[idx, :, 1])     # (b, 6)
                        #hi = np.maximum(iTSR[idx, :, 0], iTSR[idx, :, 1])     # (b, 6)
                        S = rng.uniform(lo, hi)                               # (b, 6)
                        S[lo == hi] = lo[lo == hi] 

                        Rmats = np.stack([rpy_to_R(r, p, y) for (r, p, y) in S[:, 3:]], axis=0)  # (b,3,3)
                        Tsamples = np.repeat(np.eye(4)[None, :, :], B_temp, axis=0)
                        Tsamples[:, :3, :3] = Rmats.astype(np.float32)
                        Tsamples[:, :3, 3] = S[:, :3]   # shape (b,3) -> (b,3)

                        Ttargets = Tsamples @ Tew

                        Rblocks = Ttargets[:, :3, :3]
                        q_xyzw = R.from_matrix(Rblocks).as_quat()

                        pos_world = Ttargets[:, :3, 3].astype(np.float32)
                        quat_world = np.asarray([to_genesis_quat(q) for q in q_xyzw], dtype=np.float32)

                        target_pos[new_idx] = pos_world
                        target_quat[new_idx] = quat_world
                        have_target[new_idx] = True
                    
                    stale_idx = idx[(attempts[idx] % max_local) == 0]
                    have_target[stale_idx] = False

                    poss_list  = [target_pos[idx],  target_pos[idx]]
                    quats_list = [target_quat[idx], target_quat[idx]]
                    q_out, ik_error = robot.inverse_kinematics_multilink(
                        links=[robot.links[ee_link1_idx], robot.links[ee_link2_idx]],
                        poss=poss_list, quats=quats_list,
                        init_qpos=seeds,                # <<< USE THE SEEDS
                        return_error=True,
                        pos_tol=5e-4, rot_tol=5e-3,          # sane tolerances (yours were ~1e-10: too tight)
                        envs_idx=idx
                    )

                    moved = []
                    for j, env_id in enumerate(idx):
                        q = q_out[j]
                        if q is None:
                            print(f"q for {env_id} is None")
                            continue
                        #q = q.astype(np.float32)
                        
                        seeds[env_id] = q.detach().cpu().numpy()
                        #seeds[env_id]    = q          # warm-start next loop from this q
                        q[-1] = 0.065; q[-2] = 0.065
                        q_batch[env_id]  = q.detach().cpu().numpy()
                        moved.append(env_id)

                    q_delta = q_batch - old_q
                    q_delta_means = np.mean(q_delta, axis=1)


                    if moved:
                        moved = np.asarray(moved, dtype=int)
                        robot.set_dofs_position(q_batch[moved], envs_idx=moved)
                        scene.step()
                        for env_id in moved:
                            col = robot.detect_collision(env_idx=env_id)
                            if getattr(col, "size", 0)==0:
                                if not ik_ok[env_id]:
                                    # Cache the FIRST valid IK for this environment
                                    q_sol[env_id] = q_batch[env_id].copy()
                                ik_ok[env_id] = True
                                alive[env_id] = False
                                have_target[env_id] = False

                    ik_attempt += 1

                    if (ik_attempt>=ik_max_attempts):
                        
                        unsolved_idx = np.where(alive)[0]

                        alive[unsolved_idx] = False
                        have_target[unsolved_idx] = False  

                        failed_key_indices = np.arange(i, i+n_envs)[unsolved_idx]
                        print(f"Failed to solve {failed_key_indices} after {ik_max_attempts}. Giving up...")
                        curr_ik_failures += len(unsolved_idx)
                        q_sol[unsolved_idx] = q_batch[unsolved_idx]

                ik_failures += curr_ik_failures

                print(f"IK complete: {i}-{i+envs_to_use}")
                
                # Planning
                #with start_pool(NUM_WORKERS, genesis_cfg, planner_cfg) as executor:

                    # 4) Build plain-data jobs
                    #    env_idx inside each worker is local; use [0] since each worker has n_envs=1
                TIMEOUT = 5.0
                NUM_WAYPOINTS = 200
                planning_start = time.perf_counter()

                jobs = []
                for b, k in enumerate(current_keys):
    
                    if not ik_ok[b]:
                        print(f"Planning for IK failure. Planning failure expected. Key index:{i+b}")
                    

                    q_start = np.asarray(homePos, dtype=np.float32)
                    q_goal  = np.asarray(q_sol[b], dtype=np.float32)
                    jobs.append((k, q_start, q_goal, TIMEOUT, NUM_WAYPOINTS, i+b))

                # 5) Fire them off and gather results
                results = submit_plans(executor, jobs)
                iTSR_paths.update(results)

                #failed_keys = [k for k, path in results.items() if not (path.size>0)]   # empty list => failure
                failed_keys = [k for k, path in results.items() if len(path)==0]
                num_failed  = len(failed_keys)
                planning_failures += num_failed

                print(f"[batch] planned={len(results)-num_failed}, failed={num_failed}")
                print(f"Planning complete: {i}-{i+envs_to_use}")

                #path = planner.omplPlan(
                #    qpos_goal = q_batch,
                #    qpos_start = homePosStack,
                #    num_waypoints = 200,
                #    env_idx=np.array(list(range(n_envs)))
                #)

                current_time = time.perf_counter()
                print(f"Planning time: {current_time - planning_start:.6f} seconds")

                print(f"KEYS TO GO: {len(iTSR_set)-i-envs_to_use}")
                print("-"*50)


        finally:
            executor.shutdown(wait=True)

    else:
        
        for i, key in enumerate(iTSR_set):
            
            if i%n_envs!=0:
                continue
            
            if(i==0):
                #print(key)

                volumes = find_swept_volume(scene, object_size, key, volumes=None, env_idx=None)
                plane = scene.add_entity(gs.morphs.Plane())
                scene.build(n_envs=n_envs, env_spacing=(1.0, 1.0))
                
                for env_idx in range(n_envs):
                    volumes = find_swept_volume(scene, object_size, keylist[i+env_idx], volumes=volumes, env_idx=[env_idx])  

            else:
                #volumes = find_swept_volume(object_size, key, volumes=volumes, env_idx=None)  
                for env_idx in range(n_envs):
                    volumes = find_swept_volume(scene, object_size, keylist[i+env_idx], volumes=volumes, env_idx=[env_idx])  


            robot.set_dofs_position(
                torch.tile(
                    torch.tensor(homePos, device=gs.device), (n_envs, 1)
                ),
            )
            scene.step()

            rng = np.random.default_rng()
            
            iTSR = iTSR_set[key][0]

            current_keys = keylist[i : i+n_envs]
            iTSR_batch = [
                iTSR_set[curr_key][0] for curr_key in current_keys
            ]
            iTSR = np.stack(iTSR_batch, axis=0)

            B = iTSR.shape[0]
            num_links = robot.n_links
            ee_link1_idx = num_links - 2
            ee_link2_idx = num_links - 1

            #seeds = np.tile(homePos, (B,1)).astype(np.float32)

            alive = np.ones(B, dtype=bool)     
            attempts = np.zeros(B, dtype=np.int32)

            sample_pos = np.zeros((B, 3), dtype=np.float32)
            sample_quat = np.zeros((B, 4), dtype=np.float32)
            q_batch = np.zeros((B, robot.n_dofs), dtype=np.float32)
            ik_ok = np.zeros(B, dtype=bool)

            have_target = np.zeros(B, dtype=bool)
            target_pos = np.zeros((B, 3), np.float32)
            target_quat = np.zeros((B, 4), np.float32)
            seeds = np.tile(homePos, (B, 1)).astype(np.float32)
            #seeds = torch.empty((B, 9), dtype=q.dtype, device=q.device)
            max_local = 10

            #inCollision = True

            ik_attempt = 0
            curr_ik_failures = 0
            ik_max_attempts = 1000
            q_sol = np.zeros_like(q_batch, dtype=np.float32)

            while alive.any():
            
                if ik_attempt==0:
                    old_q = np.zeros((B, robot.n_dofs), dtype=np.float32)
                else:
                    old_q = q_batch

            
                idx = np.nonzero(alive)[0]
                idx = np.array(list(range(n_envs)))
                #print(idx)
                B_temp = idx.size
                attempts[idx] += 1

                # Sample only envs without a fixed target

                new_idx = idx[~have_target[idx]]
                new_idx = idx
                if new_idx.size:
                    #print("sampling new points in TSRs")
                    lo = np.minimum(iTSR[new_idx, :, 0], iTSR[new_idx, :, 1]).astype(np.float32)     # (b, 6)
                    hi = np.maximum(iTSR[new_idx, :, 0], iTSR[new_idx, :, 1]).astype(np.float32)     # (b, 6)
                
                    #lo = np.minimum(iTSR[idx, :, 0], iTSR[idx, :, 1])     # (b, 6)
                    #hi = np.maximum(iTSR[idx, :, 0], iTSR[idx, :, 1])     # (b, 6)
                    S = rng.uniform(lo, hi)                               # (b, 6)
                    S[lo == hi] = lo[lo == hi] 

                    Rmats = np.stack([rpy_to_R(r, p, y) for (r, p, y) in S[:, 3:]], axis=0)  # (b,3,3)
                    Tsamples = np.repeat(np.eye(4)[None, :, :], B_temp, axis=0)
                    Tsamples[:, :3, :3] = Rmats.astype(np.float32)
                    Tsamples[:, :3, 3] = S[:, :3]   # shape (b,3) -> (b,3)

                    Ttargets = Tsamples @ Tew

                    Rblocks = Ttargets[:, :3, :3]
                    q_xyzw = R.from_matrix(Rblocks).as_quat()

                    pos_world = Ttargets[:, :3, 3].astype(np.float32)
                    quat_world = np.asarray([to_genesis_quat(q) for q in q_xyzw], dtype=np.float32)

                    target_pos[new_idx] = pos_world
                    target_quat[new_idx] = quat_world
                    have_target[new_idx] = True
                
                stale_idx = idx[(attempts[idx] % max_local) == 0]
                have_target[stale_idx] = False

                poss_list  = [target_pos[idx],  target_pos[idx]]
                quats_list = [target_quat[idx], target_quat[idx]]
                q_out, ik_error = robot.inverse_kinematics_multilink(
                    links=[robot.links[ee_link1_idx], robot.links[ee_link2_idx]],
                    poss=poss_list, quats=quats_list,
                    init_qpos=seeds,                # <<< USE THE SEEDS
                    return_error=True,
                    pos_tol=5e-4, rot_tol=5e-3,          # sane tolerances (yours were ~1e-10: too tight)
                    envs_idx=idx
                )

                moved = []
                for j, env_id in enumerate(idx):
                    q = q_out[j]
                    if q is None:
                        print(f"q for {env_id} is None")
                        continue
                    #q = q.astype(np.float32)
                    
                    seeds[env_id] = q.detach().cpu().numpy()
                    #seeds[env_id]    = q          # warm-start next loop from this q
                    q[-1] = 0.065; q[-2] = 0.065
                    q_batch[env_id]  = q.detach().cpu().numpy()
                    moved.append(env_id)

                q_delta = q_batch - old_q
                q_delta_means = np.mean(q_delta, axis=1)


                if moved:
                    moved = np.asarray(moved, dtype=int)
                    robot.set_dofs_position(q_batch[moved], envs_idx=moved)
                    scene.step()
                    for env_id in moved:
                        col = robot.detect_collision(env_idx=env_id)
                        if getattr(col, "size", 0)==0:
                            if not ik_ok[env_id]:
                                # Cache the FIRST valid IK for this environment
                                q_sol[env_id] = q_batch[env_id].copy()
                            ik_ok[env_id] = True
                            alive[env_id] = False
                            have_target[env_id] = False

                ik_attempt += 1

                if (ik_attempt>=ik_max_attempts):
                    
                    unsolved_idx = np.where(alive)[0]

                    alive[unsolved_idx] = False
                    have_target[unsolved_idx] = False  

                    failed_key_indices = np.arange(i, i+n_envs)[unsolved_idx]
                    print(f"Failed to solve {failed_key_indices} after {ik_max_attempts}. Giving up...")
                    curr_ik_failures += len(unsolved_idx)
                    q_sol[unsolved_idx] = q_batch[unsolved_idx]

            ik_failures += curr_ik_failures

            print(f"IK complete: {i}-{i+n_envs}")
            
            # Planning
            curr_failed = 0
            planning_start = time.perf_counter()

            for b, k in enumerate(current_keys):
                
                if not ik_ok[b]:
                    print(f"Planning for IK failure. Planning failure expected. Key index:{i+b}")

                path = planner.omplPlan(
                    #qpos_goal = np.array(list(q_batch[b])),
                    qpos_goal = np.array(list(q_sol[b])),
                    qpos_start = np.array(homePos),
                    num_waypoints = 200,
                    timeout = 5.0,
                    env_idx=[b]
                )

                if not path:
                    print(f"Failed to plan path for key: {k}")
                    curr_failed += 1
                    #planning_failures += 1
                    iTSR_paths[k] = []

                elif (path[-1].cpu().tolist()!=list(q_sol[b])):
                    print(f"path end: {path[-1].cpu().tolist()}")
                    print(f"qpos_goal: {list(q_sol[b])}")
                    print(f"Failed to plan path for key: {k}")
                    curr_failed += 1
                    #planning_failures += 1

                    iTSR_paths[k] = []
                else:

                    iTSR_paths[k] = path
            
            planning_failures += curr_failed
            #print(f"Planning complete: {i}-{i+n_envs}")

            # 5) Fire them off and gather results
            #results = submit_plans(executor, jobs)
            #iTSR_paths.update(results)

            #failed_keys = [k for k, path in results.items() if not path]   # empty list => failure
            #num_failed  = len(failed_keys)
            #planning_failures += num_failed

            print(f"[batch] planned={len(current_keys)-curr_failed}, failed={curr_failed}")
            print(f"Planning complete: {i}-{i+n_envs}")

            #path = planner.omplPlan(
            #    qpos_goal = q_batch,
            #    qpos_start = homePosStack,
            #    num_waypoints = 200,
            #    env_idx=np.array(list(range(n_envs)))
            #)

            current_time = time.perf_counter()
            print(f"Planning time: {current_time - planning_start:.6f} seconds")

            print(f"KEYS TO GO: {len(iTSR_set)-i-n_envs}")


    print("Paths generated for all object positions")
    print(f"IK failures: {ik_failures}")
    print(f"Planning failures: {planning_failures}")

    print(f"{len(iTSR_paths)} paths saved. {len(iTSR_paths) - planning_failures} valid paths.")
    print(f"Total IK and planning time: {time.perf_counter() - start:.6f} seconds")

    prefix = f"TSRs/cube_limit1/{robot_clearance}_{reachable_ws}_{round(object_dist[3], 2)}"
    save_paths_npy_numeric(
        prefix,
        iTSR_paths,
        path_dtype=np.float32,
        key_dtype=np.float64
    )

    '''
    sorted_iTSR_paths = sorted(iTSR_paths.items(), key=lambda kv: kv[0])
    packed_iTSR_paths = {encode_key(k): to_py(v) for k, v in sorted_iTSR_paths}
    with open("iTSR_paths.json", "w") as f:
            json.dump(packed_iTSR_paths, f, indent=2)   

    print("JSON file saved.")
    '''


    end = time.perf_counter()
    print(f"Elapsed time: {end - start:.6f} seconds")


    homePos = [0, -0.785, 0, -2.356, 0, 1.571, 0.785, 0.065, 0.065]

    #while(True):
        #robot.set_dofs_position(homePos)
        #robot.set_dofs_position(q)
        #scene.step()
        #print(time.time())

if __name__=="__main__":
    main()