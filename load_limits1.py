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

def to_genesis_quat(quat):
    genesis_quat = [quat[3], quat[0], quat[1], quat[2]]
    return genesis_quat

def load_store(prefix, mmap_data=None):
    data    = np.load(f"{prefix}.data.npy", mmap_mode=("r" if mmap_data else None))
    offsets = np.load(f"{prefix}.offsets.npy")
    keys    = np.load(f"{prefix}.keys.npy")  # shape (K,4,2)
    return data, offsets, keys

def get_path_by_index(data, offsets, i):
    s, e = int(offsets[i]), int(offsets[i+1])
    return data[s:e]  # O(1) view; touching L elements is O(L)

def _wrap_pi(x):
    y = (np.asarray(x, dtype=float) + np.pi) % (2*np.pi) - np.pi
    # enforce half-open: map any +π to −π (handles rare fp cases)
    return np.where(y == np.pi, -np.pi, y)

class BoxIndex4D:
    def __init__(self, keys_arr):
        assert keys_arr.ndim == 3 and keys_arr.shape[1:] == (4, 2)
        self.mins = keys_arr[:, :, 0].astype(np.float64, copy=True)
        self.maxs = keys_arr[:, :, 1].astype(np.float64, copy=True)
        self.mins[:, 3] = _wrap_pi(self.mins[:, 3])
        self.maxs[:, 3] = _wrap_pi(self.maxs[:, 3])
        self.yaw_wrap = self.mins[:, 3] > self.maxs[:, 3]
    
    def query_first(self, x, y, z, yaw):
        p = np.array([x, y, z, yaw], dtype=np.float64)
        p[3] = _wrap_pi(p[3])

        ok_xyz = np.all((p[:3] >= self.mins[:, :3]) & (p[:3] <= self.maxs[:, :3]), axis=1)
        if not np.any(ok_xyz): return None

        mins_y, maxs_y = self.mins[:, 3], self.maxs[:, 3]
        wrap = self.yaw_wrap
        ok_yaw = ((~wrap) & (p[3] >= mins_y) & (p[3] <= maxs_y)) | (wrap & ((p[3] >= mins_y) | (p[3] <= maxs_y)))

        ok = ok_xyz & ok_yaw
        if not np.any(ok): return None
        return int(np.argmax(ok))

class SparseBoxGrid4D:
    def __init__(self, keys_arr):
        ...
        mins = keys_arr[:, :, 0].astype(np.float64)
        maxs = keys_arr[:, :, 1].astype(np.float64)

        mins[:, 3] = _wrap_pi(mins[:, 3])
        maxs[:, 3] = _wrap_pi(maxs[:, 3])

        self.x0 = mins[:, 0].min()
        self.y0 = mins[:, 1].min()
        self.z0 = mins[:, 2].min()

        yaw_mins_sorted = np.sort(mins[:, 3])
        self.yaw0 = yaw_mins_sorted[0]

        def spacing(vals):
            vals = np.sort(np.unique(vals))
            diffs = np.diff(vals)
            diffs = diffs[diffs > 1e-9]
            return diffs.min()

        self.dx = spacing(mins[:, 0])
        self.dy = spacing(mins[:, 1])

        z_range = np.ptp(mins[:, 2])              # <--- compute ONCE
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
            iyaw = int(round((_wrap_pi(yaw_min) - self.yaw0) / self.dyaw)) % self.nyaw

            key = (ix, iy, iz, iyaw)
            self.index[key] = bin_idx

    def query_first(self, x, y, z, yaw):
        yaw = _wrap_pi(yaw)

        ix   = int(np.floor((x   - self.x0) / self.dx))
        iy   = int(np.floor((y   - self.y0) / self.dy))
        iz   = int(np.floor((z   - self.z0) / self.dz)) if self.dz != 1.0 else 0
        iyaw = int(np.floor((yaw - self.yaw0) / self.dyaw)) % self.nyaw

        key = (ix, iy, iz, iyaw)
        return self.index.get(key, None)

# **Monkey-patch**
igl.signed_distance = _signed_distance_3out

def main():
    gs.init(backend=gs.cpu)
    #gs.init(backend=gs.cpu, logging_level='error')

    #scene = gs.Scene(show_viewer=True, show_FPS=True)
    scene = gs.Scene(show_viewer=True, show_FPS=False,
                    viewer_options = gs.options.ViewerOptions(
                        max_FPS = 200
                    ))

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

    object_size = [0.06, 0.06, 0.2]
    object_quat = [0, 0, 0, 1]
    object_roll, object_pitch, object_yaw = R.from_quat(object_quat).as_euler('xyz', degrees=False)

    nominal_object_pose = [0.4, 0.4, 0+object_size[2]/2]
    object_position = nominal_object_pose

    robot_clearance = 0.3
    reachable_ws = 0.6
    yaw_limit = 0.5*np.pi

    object = scene.add_entity(
        gs.morphs.Box(
            pos = object_position,
            #size = object_size,
            size = object_size,
            quat = to_genesis_quat(object_quat),
            fixed = True,
            collision=True
        )
    )

    plane = scene.add_entity(gs.morphs.Plane())
    scene.build()

    start_load = time.perf_counter()

    prefix = f"TSRs/cube_limit1/{robot_clearance}_{reachable_ws}_{round(yaw_limit, 2)}"

    data, offsets, keys_arr= load_store(prefix, mmap_data=True)
    #indexer = BoxIndex4D(keys_arr)
    indexer = SparseBoxGrid4D(keys_arr)

    time_load = time.perf_counter() - start_load
    print(f"Time to load datastructure: {time_load:.6f} seconds")

    while(True):
        sampled_yaw = round(np.random.uniform(-yaw_limit, yaw_limit), 5)
        sampled_z = nominal_object_pose[2]

        theta = np.random.uniform(0, 2*np.pi)
        r = np.sqrt(np.random.uniform(robot_clearance**2, reachable_ws**2))
        sampled_x = round(r * np.cos(theta), 5)
        sampled_y = round(r * np.sin(theta), 5)

        print(f"Sampled object position: {sampled_x, sampled_y, sampled_z, sampled_yaw}")

        sampled_quat = R.from_euler('xyz', [0, 0, sampled_yaw], degrees=False).as_quat()

        homePos = [0, -0.785, 0, -2.356, 0, 1.571, 0.785, 0.065, 0.065]
        robot.set_dofs_position(homePos)
        #scene.step()

        object.set_pos(pos= (sampled_x, sampled_y, sampled_z))
        object.set_quat(quat= to_genesis_quat(sampled_quat))
        scene.step()

        start_query = time.perf_counter()

        idx = indexer.query_first(sampled_x, sampled_y, sampled_z, sampled_yaw)
        if idx is None:
            #current_time = time.perf_counter()
            #print(f"Failed. Time for query: {current_time - start_query:.6f} seconds")
            input("Failed to find path. Continue?")

            

        else:
            path = get_path_by_index(data, offsets, idx)
            #current_time = time.perf_counter()
            query_time = (time.perf_counter() - start_query)
            print(f"Success. Time for query: {query_time:.6f} seconds")

            print(path[-1])

            #print(f"Total time: {query_time+time_load:.6f} seconds")
            '''
            goal = np.array([round(x, 3) for x in path[-1]])
            
            planner = omplPlanner(robot)
            
            planning_start = time.perf_counter()
            planned_path = planner.omplPlan(
                qpos_goal=goal,
                qpos_start=homePos,
                timeout=10.0,
                num_waypoints=200,
                planner="RRTConnect"
            )
            print(f"PFS time: {time.perf_counter()-planning_start:.6f} seconds")
            planned_goal = np.array([round(x, 3) for x in planned_path[-1].tolist()])

            print(planned_goal)
            print(goal)

            #print(planned_goal!=goal)
            if (list(planned_goal)!=list(goal)):
                print("PLANNING FAILED!")
            '''
            
            input("Visualize?")
            path = torch.from_numpy(path.copy())
            for waypoint in path:
                robot.set_dofs_position(waypoint, zero_velocity=True)
                scene.step()

                collisions = robot.detect_collision()
                if collisions.size>0:
                    print(f"Collisions: {collisions}")
            
            input("Path complete. Continue?")    



if __name__=='__main__':
    main()