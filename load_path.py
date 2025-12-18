import genesis as gs
import trimesh
import yaml
import numpy as np
from scipy.spatial.transform import Rotation as R
import time

import igl
import json
from functools import wraps

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

def to_genesis_quat(quat):
    genesis_quat = [quat[3], quat[0], quat[1], quat[2]]
    return genesis_quat

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

gs.init(backend=gs.cpu)
#gs.init(backend=gs.cpu, logging_level='error')

#scene = gs.Scene(show_viewer=True, show_FPS=True)
scene = gs.Scene(show_viewer=True, show_FPS=False,
                 viewer_options = gs.options.ViewerOptions(
                     max_FPS = 70
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

object_position = [0.4, 0.0, 0.1]
object_quat = [0, 0, 0, 1]
object_roll, object_pitch, object_yaw = R.from_quat(object_quat).as_euler('xyz', degrees=False)


object_radius = 0.04
object_height = 0.2

object_size = [0.06, 0.06, 0.2]

object_dist = [0.05, 0.05, 0, np.pi]
#object_dist = [0.1, 0.1, 0]
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

object1 = scene.add_entity(
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

scene.draw_debug_box([[object_lower[0], object_lower[1], 0], [object_upper[0], object_upper[1], 0]], color=(1.0, 0.0, 0.0, 1.0), wireframe=True, wireframe_radius=0.0015)

with open('JSONs/iTSR_paths.json') as f:
    packed = json.load(f)
iTSR_paths = {decode_key(k): v for k, v in packed.items()}

print(len(iTSR_paths))
homePos = [0, -0.785, 0, -2.356, 0, 1.571, 0.785, 0.065, 0.065]

while(True):
    rng = np.random.default_rng()
    lo = object_lower[0]
    hi = object_upper[0]
    x_sample = rng.uniform(lo, hi)

    lo = object_lower[1]
    hi = object_upper[1]
    y_sample = rng.uniform(lo, hi)

    lo = object_lower[2]
    hi = object_upper[2]
    z_sample = rng.uniform(lo, hi)

    lo = object_lower[3]
    hi = object_upper[3]
    yaw_sample = rng.uniform(lo, hi)

    sampled_object_pos = [x_sample, y_sample, z_sample]
    sampled_object_quat = R.from_euler('xyz', [0, 0, yaw_sample], degrees=False).as_quat()

    robot.set_dofs_position(homePos)

    object1.set_pos(pos= sampled_object_pos)
    object1.set_quat(quat= to_genesis_quat(sampled_object_quat))
    scene.step()

    print(f"Placing object at {sampled_object_pos} at yaw angle: {yaw_sample}")
    start = time.perf_counter()

    found_iTSR = False
    for idx, interval in enumerate(iTSR_paths):

        path = iTSR_paths[interval]
        if(interval[0][0]<=sampled_object_pos[0] and sampled_object_pos[0]<=interval[0][1]):
            if(interval[1][0]<=sampled_object_pos[1] and sampled_object_pos[1]<=interval[1][1]):
                if(interval[2][0]<=sampled_object_pos[2] and sampled_object_pos[2]<=interval[2][1]):
                    if(interval[3][0]<=yaw_sample and yaw_sample<=interval[3][1]):
                        found_iTSR = True
                        break
                    else:
                        continue
                else:
                    continue
            else:
                continue
        else:
            continue

    if (found_iTSR):
        current_time = time.perf_counter()
        print(f"Time for query: {current_time - start:.6f} seconds")

        input("Found path. Visualize?")
        
        for waypoint in path:
            robot.set_dofs_position(waypoint, zero_velocity=True)
            scene.step()

            collisions = robot.detect_collision()
            if collisions.size>0:
                print(f"Collisions: {collisions}")
        
        input("Path complete. Continue?")

    else:
        input("Did NOT find path. Continue?")
    
    
