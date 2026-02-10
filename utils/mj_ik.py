#import genesis as gs # type: ignore
import torch # type: ignore
import numpy as np
import time
import random

from tqdm import tqdm #type: ignore

from scipy.spatial.transform import Rotation as R #type: ignore
#from src.swept_volume import SweptVolumeCube
from utils.helpers import rpy_to_R, to_genesis_quat, load_store, get_path_by_index
#from src.condense_paths import PrefixBuilder, SuffixBuilder, SparseBoxGrid4D
from src.visualization import Plotter

from utils.mujoco_utils import get_robot_collision_geom_ids, MujocoViewer
from src.condense_paths_mj import MjPrefix, MjSuffix, SparseBoxGrid4D

from utils.mujoco_utils import move_swept_volume, set_panda_qpos, robot_in_contact, densify_q_traj
from utils.mujoco_utils import MujocoViewer, mj_fk, mj_ik_multilink, PandaMinkIK, get_panda_qpos_idxs, mj_ik_multilink_until_tol

from utils.mujoco_utils import qpos_within_limits, randomize_seed_within_limits

from planning_mj import omplPlanner

import numpy as np
import mujoco

def dbg(name, T):
    Rw = T[:3,:3]
    pw = T[:3,3]
    z_w = Rw @ np.array([0,0,1.0])   # EE local +Z in world
    x_w = Rw @ np.array([1,0,0.0])   # EE local +X in world
    print(name)
    print("  p_world:", pw)
    print("  z_world:", z_w)
    print("  x_world:", x_w)

def Rz(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c,-s,0],
                     [s, c,0],
                     [0, 0,1]], dtype=float)

def rotate_T_about_worldZ(T, theta, p_center):
    R = Rz(theta)
    T2 = T.copy()
    # rotate orientation in world
    T2[:3,:3] = R @ T[:3,:3]
    # rotate position about center
    T2[:3,3]  = p_center + R @ (T[:3,3] - p_center)
    return T2

def quat_conj_wxyz(q):
    # q = [w,x,y,z]
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)

def quat_mul_wxyz(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ], dtype=np.float64)

def quat_to_angle_wxyz(q):
    # assumes q normalized, returns angle in [0, pi]
    q = q / np.linalg.norm(q)
    w = np.clip(q[0], -1.0, 1.0)
    return 2.0 * np.arccos(abs(w))  # abs handles q vs -q

def ee_pose_error(model, data, body_name, target_pos, target_quat_wxyz):
    # assumes data.qpos already set to IK solution
    mujoco.mj_forward(model, data)

    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)

    # achieved pose in world
    pos = data.xpos[bid].copy()
    xmat = data.xmat[bid].reshape(3, 3).copy()

    # convert rotation matrix -> quat (MuJoCo gives xyzw!)
    quat_wxyz = np.zeros(4)
    mujoco.mju_mat2Quat(quat_wxyz, xmat.ravel())


    # position error
    pos_err = np.linalg.norm(pos - np.asarray(target_pos, dtype=np.float64))

    # orientation error: q_err = q_target * conj(q_achieved)
    q_t = np.asarray(target_quat_wxyz, dtype=np.float64)
    q_a = quat_wxyz
    q_err = quat_mul_wxyz(q_t, quat_conj_wxyz(q_a))
    ang_err = quat_to_angle_wxyz(q_err)

    return pos_err, ang_err, pos, quat_wxyz


'''
def plan_to_goal(robot, current_keys, q_sol, ik_ok, i, homePos):
    
    curr_failed = 0
    current_paths = {}
    planner = omplPlanner(robot)

    for b, k in enumerate(current_keys):
        if not ik_ok[b]:
            print(f"Planning for IK failure. Planning failure expected. Key index:{i+b}")
            
        #print(f"Goal to plan to: {np.array(list(q_sol[b]))}")
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
            current_paths[k] = []

        elif (path[-1].cpu().tolist()!=list(q_sol[b])):
            print(f"path end: {path[-1].cpu().tolist()}")
            print(f"qpos_goal: {list(q_sol[b])}")
            print(f"Failed to plan path for key: {k}")
            curr_failed += 1
            #planning_failures += 1
            current_paths[k] = []

        else:
            current_paths[k] = path
        
    return current_paths, curr_failed 
'''
def goal_from_ik(mj_model, mj_data, homePos, robot_geoms, sv_geoms, curr_iTSR, Tew, ik_max_attempts, viewer, grasp_idx=0):
    valid_ik = False
    ik_attempts = 0

    pos_tol=0.01          
    ang_tol=0.03

    while not valid_ik:
        rng = np.random.default_rng()

        lo = np.minimum(curr_iTSR[:, 0], curr_iTSR[:, 1]).astype(np.float32)
        hi = np.maximum(curr_iTSR[:, 0], curr_iTSR[:, 1]).astype(np.float32)
        S = rng.uniform(lo, hi)    
        S[lo == hi] = lo[lo == hi]

        mean_S = curr_iTSR.mean(axis=1)
        #S = mean_S
        #print(f"Sampled S: {S}")
        Rmat = rpy_to_R(S[3], S[4], S[5])
        Tsample = np.eye(4)
        Tsample[:3, :3] = Rmat
        Tsample[:3, 3] = S[:3]

        if grasp_idx==0:
            grasp = "top"
        else: #1
            grasp = "front"

        #grasp = "front"

        d = 0.04
        if grasp=="top":
            # offset in the Tsample frame (choose the axis that corresponds to "left/right")
            offset_local = np.array([0.0, -d, 0.0])   # example: +Y in Tsample
            offset_world = Tsample[:3,:3] @ offset_local

            Tsample1 = Tsample.copy()
            Tsample1[:3,3] += offset_world

            Tsample2 = Tsample.copy()
            Tsample2[:3,3] -= offset_world

            Tew1_top = Tew.copy()
            Ttarget1_top = Tsample1 @ Tew1_top

            p_center = np.array(Tsample[:3,3])

            T_top = Ttarget1_top
            T_top_90  = rotate_T_about_worldZ(T_top, +np.pi/2, p_center)
            T_top_m90 = rotate_T_about_worldZ(T_top, -np.pi/2, p_center)
            T_top_180  = rotate_T_about_worldZ(T_top,  np.pi,   p_center)

            if ik_attempts<=int(ik_max_attempts/4):
                Ttarget1 = T_top
            elif ik_attempts<=int(ik_max_attempts/2):
                Ttarget1 = T_top_90
            elif ik_attempts<=int(3*ik_max_attempts/4):
                Ttarget1 = T_top_m90
            else:
                Ttarget1 = T_top_180

        else: #front
            offset_local = np.array([0.0, 0.0, -d])   # example: +Y in Tsample
            #offset_local = np.array([0.0, 0.0, 0.0])
            offset_world = Tsample[:3,:3] @ offset_local

            Tsample1 = Tsample.copy()
            Tsample1[:3,3] += offset_world

            Tsample2 = Tsample.copy()
            Tsample2[:3,3] -= offset_world

            l_f = 0.054/2
            ee_offset = l_f*1.1
            offset_ee_frame = np.array([0.0, 0.0, -ee_offset])

            Tew1_front = Tew.copy()
            Ttarget1_front = Tsample1 @ Tew1_front
            
            p_center = np.array(Tsample[:3,3])   # or object center in world
            
            # Targets in world frame for front, left, right, back
            T_front = Ttarget1_front
            T_left  = rotate_T_about_worldZ(T_front, +np.pi/2, p_center)
            T_right = rotate_T_about_worldZ(T_front, -np.pi/2, p_center)
            T_back  = rotate_T_about_worldZ(T_front,  np.pi,   p_center)
            
            Ttarget1 = T_front

        # Second finger target
        Rz_pi = np.array([
            [-1,  0, 0],
            [ 0, -1, 0],
            [ 0,  0, 1]
        ])
        Tew2 = Tew.copy()
        Tew2[:3, :3] = Tew2[:3, :3] @ Rz_pi
        Ttarget2 = Tsample2 @ Tew2


        pos_target1 = np.array(Ttarget1[:3, 3])
        pos_target2 = np.array(Ttarget2[:3, 3])
        quat_xyzw1 = R.from_matrix(Ttarget1[:3, :3]).as_quat()
        quat_xyzw2 = R.from_matrix(Ttarget2[:3, :3]).as_quat()

        quat_wxyz1 = np.array([quat_xyzw1[3], quat_xyzw1[0], quat_xyzw1[1], quat_xyzw1[2]])
        quat_wxyz2 = np.array([quat_xyzw2[3], quat_xyzw2[0], quat_xyzw2[1], quat_xyzw2[2]])

        z_cmd = Ttarget1[:3,:3] @ np.array([0,0,1.0])

        #print(f"quat_wxyz1: {quat_wxyz1}")

        finger1_target = {
            'pos': pos_target1,
            'quat': quat_wxyz1,
            'name': 'left_finger'
        }
        finger2_target = {
            'pos': pos_target2,
            'quat': quat_wxyz2,
            'name': 'right_finger'
        }
        link_targets = [finger1_target]
        #ndof = 9
        #homePos = homePos + 0.1 * np.random.randn(ndof)

        #ik_path = mj_ik_multilink(mj_model, mj_data, link_targets, homePos, robot_geoms, sv_geoms, single=True, dt=0.001, iter_count=2000, ignore_col=True)
        seed = np.array(homePos, dtype=float)
        seed2 = randomize_seed_within_limits(mj_model, seed)

        ik_path = mj_ik_multilink_until_tol(
            mj_model,
            mj_data,
            link_targets,
            seed2,
            robot_geoms,
            sv_geoms,
            single=True,
            dt=0.01,
            max_iters=500,
            pos_tol=pos_tol,          
            ang_tol=ang_tol,         
            ignore_col=True,
        )

        if ik_path is not None:
            ik_goal = ik_path[-1]
            set_panda_qpos(mj_model, mj_data, ik_goal)
            
            in_contact, _ = robot_in_contact(mj_model, mj_data, robot_geoms)
            #print(in_contact)
            target_pos = link_targets[0].get('pos')
            target_quat = link_targets[0].get('quat')
            if target_quat is None:
                target_quat = [1, 0, 0, 0]
            target_name = link_targets[0].get('name')

            pos_err, ang_err, pos_ach, quat_ach = ee_pose_error(
                mj_model, mj_data,
                body_name=target_name,                 
                target_pos=target_pos,
                target_quat_wxyz=target_quat
            )

            #print("pos_err (m):", pos_err)
            #print("ang_err (rad):", ang_err)

            within_limits, _, _, _, _ = qpos_within_limits(mj_model, ik_goal)
            #print(f"IK solution within limits: {within_limits}")

            #input("G?")

            if in_contact is False and within_limits is True:

                #if pos_err <=0.01 and ang_err <=0.001:
                #    valid_ik = True
                #else:
                #    valid_ik = False
                valid_ik = True
            else:
                #if in_contact is True:
                    #print("IK returned Collision")
                #if within_limits is False:
                    #print("IK returned Out of Limits")
                #print("IK returned Collision")
                valid_ik = False
        else:
            #print("IK returned None")
            valid_ik = False
            
        ik_attempts += 1

        if ik_attempts>=ik_max_attempts and valid_ik is False:
            # Give up on bin
            ik_goal = None
            #print("Failed to find IK goal.")
            break
        if viewer is not None:
            viewer.viewer.sync()
        #input("IK again?")
    
    return ik_goal, valid_ik, link_targets 

def cover_iTSR_test(mj_model, mj_data, homePos, iTSR_set_all, object_details, Tew_all, viewer):

    #idxs = get_panda_qpos_idxs(mj_model)
    #print("panda qpos idxs:", idxs)
    #print("model.nq:", mj_model.nq)

    grasp_index = 0 # top grasps
    iTSR_set = iTSR_set_all[grasp_index] 
    Tew = Tew_all[grasp_index]

    robot_geoms = get_robot_collision_geom_ids(mj_model)
    mujocoViewer = MujocoViewer(mj_model, mj_data, robot_geoms, show_left_ui=True, show_right_ui=True) if viewer else None
    
    if viewer:
        mujocoViewer.open()

    sv_geoms = ["sv_box1","sv_box2","sv_cyl1","sv_cyl2","sv_cyl3","sv_cyl4"]

    ik_failures = 0
    planning_failures = 0
    ik_max_attempts = 200
    print_interval = 100

    planner = omplPlanner(mj_model, mj_data, robot_geoms, log=False)    
    object_size = object_details['size']
    keylist = list(iTSR_set)

    iTSR_paths = {}
    start = time.perf_counter()

    total_plan_times = []
    solve_times = []
    plan_success = []
    ik_success = []

    for i, key in enumerate(iTSR_set):
        #print(f"Moving swept volume to key {key}")
        move_swept_volume(mj_model, mj_data, key)
        set_panda_qpos(mj_model, mj_data, homePos)
        
        curr_iTSR = iTSR_set[key][0]
        ik_goal, valid_ik, link_targets = goal_from_ik(mj_model, mj_data, homePos, robot_geoms, sv_geoms, curr_iTSR, Tew, ik_max_attempts, mujocoViewer, grasp_idx=grasp_index)
        
        target_pos = link_targets[0]['pos']
        target_quat = link_targets[0]['quat']
        target_name = link_targets[0]['name']

        pos_err, ang_err, pos_ach, quat_ach = ee_pose_error(
            mj_model, mj_data,
            body_name=target_name,                 
            target_pos=target_pos,
            target_quat_wxyz=target_quat
        )
        #print("final pos_err (m):", pos_err)
        #print("final ang_err (rad):", ang_err)
        
        if mujocoViewer is not None:
            mujocoViewer.viewer.sync()
        #input("Plan?")
        if valid_ik:
            path = planner.omplPlan(
                qpos_goal = np.array(ik_goal),
                qpos_start = np.array(homePos),
                num_waypoints=200,
                timeout=3.0
            )
            if not path:
                print(f"Planning failure for key: {key}")
                planning_failures += 1
                iTSR_paths[key] = None
            else:
                iTSR_paths[key] = path

                #if viewer:
                #    mujocoViewer.play_qpos_traj(path)

            total_plan_times.append(planner.total_time)
            solve_times.append(planner.plan_time)

        else:
            ik_failures += 1
            print(f"IK failure for key: {key}")
            # Skipping planning for IK failure
            iTSR_paths[key] = None
            path = []
            planning_failures += 1

            total_plan_times.append(np.nan)
            solve_times.append(np.nan)
        
        plan_success.append(bool(path))
        ik_success.append(valid_ik)

        # Logging 
        if (i % print_interval == 0) or i == len(iTSR_set) - 1:
            print(f"Planned for {i+1}/{len(iTSR_set)} bins")
            print(f"IK success rate so far: {np.mean(ik_success):.3f}")
            print(f"Plan success rate so far (given IK): {np.mean([p for p,ik in zip(plan_success,ik_success) if ik]):.3f}")
            print(f"Planning failures: {planning_failures}")

            st = np.array(solve_times, dtype=float)
            tt = np.array(total_plan_times, dtype=float)
            print(f"Median solve time (successes only): {np.nanmedian(st[np.array(plan_success)]):.4f}s")
            print(f"Median total plan time (successes only): {np.nanmedian(tt[np.array(plan_success)]):.4f}s \n")
            
        #input("G?")


        #print(f"in_contact: {in_contact}")

        #print(pos_target1)
        #print(pos_target2)
        #print(quat_wxyz1)
        #print(quat_wxyz2)

        #fk_output = mj_fk(mj_model, mj_data, ik_goal)
        #print(fk_output)
        
        
        #input("G?")
        #return None

        # Planning
        #planning_start = time.perf_counter()

        #current_paths, curr_failed = plan_to_goal(robot, current_keys, q_sol, ik_ok, i, homePos)
        #iTSR_paths.update(current_paths)

        #planning_failures += curr_failed

        #print(f"[batch] planned={len(current_keys)-curr_failed}, failed={curr_failed}")
        #print(f"Planning complete: {i}-{i+n_envs}")
        #print(f"Planning complete: {i}-{i+envs_to_use}")

        #current_time = time.perf_counter()
        #print(f"Planning time: {current_time - planning_start:.6f} seconds")

        #print(f"KEYS TO GO: {len(iTSR_set)-i-n_envs}")
        #print(f"KEYS TO GO: {len(iTSR_set)-i-envs_to_use}")

    print("Paths generated for all object positions")
    print(f"IK failures: {ik_failures}")
    print(f"Planning failures: {planning_failures}")

    print(f"{len(iTSR_paths)} paths saved. {len(iTSR_paths) - planning_failures} valid paths.")
    print(f"Total IK and planning time: {time.perf_counter() - start:.6f} seconds")    

    return iTSR_paths, ik_success, plan_success, solve_times, total_plan_times

def cover_iTSR(mj_model, mj_data, homePos, iTSR_set, object_details, Tew, viewer):

    #idxs = get_panda_qpos_idxs(mj_model)
    #print("panda qpos idxs:", idxs)
    #print("model.nq:", mj_model.nq)

    robot_geoms = get_robot_collision_geom_ids(mj_model)
    mujocoViewer = MujocoViewer(mj_model, mj_data, robot_geoms, show_left_ui=True, show_right_ui=True) if viewer else None
    
    if viewer:
        mujocoViewer.open()

    sv_geoms = ["sv_box1","sv_box2","sv_cyl1","sv_cyl2","sv_cyl3","sv_cyl4"]

    ik_failures = 0
    planning_failures = 0
    ik_max_attempts = 20
    print_interval = 100

    planner = omplPlanner(mj_model, mj_data, robot_geoms)    
    object_size = object_details['size']
    keylist = list(iTSR_set)

    iTSR_paths = {}
    start = time.perf_counter()

    total_plan_times = []
    solve_times = []
    plan_success = []
    ik_success = []

    for i, key in enumerate(iTSR_set):
        
        move_swept_volume(mj_model, mj_data, key)
        set_panda_qpos(mj_model, mj_data, homePos)
        
        curr_iTSR = iTSR_set[key][0]
        ik_goal, valid_ik = goal_from_ik(mj_model, mj_data, homePos, robot_geoms, sv_geoms, curr_iTSR, Tew, ik_max_attempts)
        #mujocoViewer.viewer.sync()
        #input("Plan?")
        if valid_ik:
            path = planner.omplPlan(
                qpos_goal = np.array(ik_goal),
                qpos_start = np.array(homePos),
                num_waypoints=200,
                timeout=3.0
            )
            if not path:
                planning_failures += 1
                iTSR_paths[key] = None
            else:
                iTSR_paths[key] = path

                if viewer:
                    mujocoViewer.play_qpos_traj(path)

            total_plan_times.append(planner.total_time)
            solve_times.append(planner.plan_time)

        else:
            ik_failures += 1

            # Skipping planning for IK failure
            iTSR_paths[key] = None
            planning_failures += 1

            total_plan_times.append(np.nan)
            solve_times.append(np.nan)
        
        plan_success.append(bool(path))
        ik_success.append(valid_ik)

        # Logging 
        if (i % print_interval == 0) or i == len(iTSR_set) - 1:
            print(f"Planned for {i+1}/{len(iTSR_set)} bins")
            print(f"IK success rate so far: {np.mean(ik_success):.3f}")
            print(f"Plan success rate so far (given IK): {np.mean([p for p,ik in zip(plan_success,ik_success) if ik]):.3f}")
            print(f"Planning failures: {planning_failures}")

            st = np.array(solve_times, dtype=float)
            tt = np.array(total_plan_times, dtype=float)
            print(f"Median solve time (successes only): {np.nanmedian(st[np.array(plan_success)]):.4f}s")
            print(f"Median total plan time (successes only): {np.nanmedian(tt[np.array(plan_success)]):.4f}s")

            
        #input("G?")


        #print(f"in_contact: {in_contact}")

        #print(pos_target1)
        #print(pos_target2)
        #print(quat_wxyz1)
        #print(quat_wxyz2)

        #fk_output = mj_fk(mj_model, mj_data, ik_goal)
        #print(fk_output)
        
        
        #input("G?")
        #return None

        # Planning
        #planning_start = time.perf_counter()

        #current_paths, curr_failed = plan_to_goal(robot, current_keys, q_sol, ik_ok, i, homePos)
        #iTSR_paths.update(current_paths)

        #planning_failures += curr_failed

        #print(f"[batch] planned={len(current_keys)-curr_failed}, failed={curr_failed}")
        #print(f"Planning complete: {i}-{i+n_envs}")
        #print(f"Planning complete: {i}-{i+envs_to_use}")

        #current_time = time.perf_counter()
        #print(f"Planning time: {current_time - planning_start:.6f} seconds")

        #print(f"KEYS TO GO: {len(iTSR_set)-i-n_envs}")
        #print(f"KEYS TO GO: {len(iTSR_set)-i-envs_to_use}")

    print("Paths generated for all object positions")
    print(f"IK failures: {ik_failures}")
    print(f"Planning failures: {planning_failures}")

    print(f"{len(iTSR_paths)} paths saved. {len(iTSR_paths) - planning_failures} valid paths.")
    print(f"Total IK and planning time: {time.perf_counter() - start:.6f} seconds")    

    return iTSR_paths, ik_success, plan_success, solve_times, total_plan_times

def mj_condense(mj_model, mj_data, filename_prefix, object_details, obj_geom_list, viewer):
    '''
    Takes a generated data structure and condenses it using a Prefix-Suffix strategy
    Outputs a collection of root paths and a collection of subpaths as references to the root paths
    '''

    object_size = object_details['size']
    robot_geoms = get_robot_collision_geom_ids(mj_model)
    mujocoViewer = MujocoViewer(mj_model, mj_data, robot_geoms) if viewer else None
    ik_solver = PandaMinkIK(mj_model, mj_data, robot_geoms, obj_geom_list)

    #if viewer:
    #    mujocoViewer.open()

    # saved datastructures
    root_paths = {}
    prefix_map = {}

    roots_idx = []

    data, offsets, keys_arr = load_store(filename_prefix, mmap_data=True)
    print(f"Number of bins: {len(keys_arr)}")
    indexer = SparseBoxGrid4D(keys_arr)
    plotter = Plotter(keys_arr, indexer)

    covered_key_indices = set()
    uncovered_key_indices = set(range(len(keys_arr)))
    always_invalid = set(np.where(offsets[1:] == offsets[:-1])[0])
    blacklist = set()


    recent_coverages = []

    exit_factor = 0.05
    rho = 0.01
    G_best_min = 25
    W_max = 20
    max_checked_bins = min(10000, int(0.15 * len(keys_arr)))
    max_suffix_checks = min(1000, int(0.05 * len(keys_arr)))

    fail_streak = 0
    FAIL_STREAK_MAX = 100
    A_accept_max = min(2000, max(100, int(0.005 * len(keys_arr))))

    while uncovered_key_indices:
        
        if len(uncovered_key_indices) > int(0.75*len(keys_arr)):
            K_abs = 40
            fail_bridge_budget = 0
        elif len(uncovered_key_indices) > int(0.5*len(keys_arr)):
            K_abs = 20
            fail_bridge_budget = 500
        else:
            K_abs = 5
            fail_bridge_budget = 1000

        covered_idx = []
        prefix_lengths = []
        covered_goals = []
        checked_bins = set()
        suffix_checks = 0
        #fail_bridge_budget = 1000
        #fail_bridge_budget = 0

        print("Sampling a new root path...")
        
        # Base pool: uncovered, excluding invalid and blacklisted
        pool = list(uncovered_key_indices - always_invalid - blacklist)

        if not pool:
            idx = random.choice(tuple(uncovered_key_indices))
        else:
            cands = [
                i for i in pool
                if any((n in uncovered_key_indices) for n in indexer.adjacent_neighbors(keys_arr[i]))
            ]
            idx = random.choice(cands if cands else pool)

        '''
        cands = [i for i in uncovered_key_indices
                if any(n not in covered_key_indices
                    for n in indexer.adjacent_neighbors(keys_arr[i]))]

        if not cands:
            idx = random.choice(tuple(uncovered_key_indices))
        else:
            idx = random.choice(cands)
        '''

        path = get_path_by_index(data, offsets, idx)

        #if (path is None):
        if path is None or getattr(path, "size", 0) == 0:
            print("Empty candidate root. Skipping...")
            always_invalid.add(idx)
            fail_streak += 1
            continue

        #pathPrefix = PrefixBuilder(scene, robot, SV, len(path))
        #pathSuffix = SuffixBuilder(scene, robot, SV, len(path))

        pathPrefix = MjPrefix(mj_model, mj_data, robot_geoms, mujocoViewer, len(path))
        pathSuffix = MjSuffix(mj_model, mj_data, robot_geoms, obj_geom_list, mujocoViewer, len(path))

        # Uncovered neighbors of root
        #neighbor_idx = indexer.neighbors_by_box(keys_arr[idx], R_meters=0.1, R_yaw=0.3)
        neighbor_idx = indexer.adjacent_neighbors(keys_arr[idx])
        neighbor_idx = list(dict.fromkeys(neighbor_idx))

        #neighbor_idx = [n for n in neighbor_idx if n != idx and n not in covered_key_indices]
        neighbor_idx = [n for n in neighbor_idx if n != idx]
        neighbor_idx = [n for n in neighbor_idx if n not in checked_bins]
        #print("uncovered neighbor count:", len(neighbor_idx))

        cleared_queue = (len(neighbor_idx)==0)

        while (not cleared_queue):    
            #print(f"Checking {len(neighbor_idx)} neighbors...")
            to_check_suffix = [
                n for n in neighbor_idx
                if n not in covered_key_indices
                and n not in checked_bins
            ]
            print(
                f"Layer: total={len(neighbor_idx)}, "
                f"suffix_checks={len(to_check_suffix)}, "
                f"checked_total={len(checked_bins)}"
            )

            next_layer = []
            covered_before = len(covered_idx)

            for i in range(len(neighbor_idx)):

                curr_bin = keys_arr[neighbor_idx[i]]
                prefix_len = pathPrefix.find_prefix_len(path, curr_bin)
                #print(f"prefix_len {prefix_len}")
                #pathPrefix.visualize_prefix(path[:prefix_len])

                n_bin_path = get_path_by_index(data, offsets, neighbor_idx[i])
                
                if n_bin_path.size==0:
                    n_bin_goal = None
                else:
                    n_bin_goal = n_bin_path[-1]

                checked_bins.add(neighbor_idx[i])
                is_globally_covered = (neighbor_idx[i] in covered_key_indices)

                if n_bin_goal is not None:

                    if not is_globally_covered:

                        suffix_checks += 1

                        suffix_start = time.time()
                        #suffix = pathSuffix.ik_suffix(path, prefix_len, n_bin_goal)
                        suffix = pathSuffix.ik_suffix_single(path, prefix_len, n_bin_goal, ik_solver)
                        #print(f"Precollision check time: {time.time() - suffix_start}")
                        #print("Checking suffix for collisions")

                        if suffix is not None:
                            dense_suffix = densify_q_traj(suffix, max_step=0.02)
                            for q in dense_suffix:
                                set_panda_qpos(mj_model, mj_data, q)                
                                in_contact, _ = robot_in_contact(mj_model, mj_data, robot_geoms)
                                within_limits, _, _, _, _ = qpos_within_limits(mj_model, q)
                                #mujocoViewer.viewer.sync()
                                #input("Continue?")
                                if in_contact or not within_limits:
                                    suffix = None
                                    break

                    else:
                        suffix = None
                    
                else:
                    #print("Empty neighbor bin")
                    suffix = None

                if (suffix is not None) and (not is_globally_covered):
                    #print(f"Suffix found. Time with collision checks: {time.time() - suffix_start}")
                    #print(f"Suffix time: {time.time() - suffix_start}")
                    #pathSuffix.visualize_suffix(suffix)

                    if viewer is True:
                        print("Visualizing entire path")
                        prefix = path[:prefix_len]
                        qpos_traj = np.vstack([prefix, suffix])
                        mujocoViewer.play_qpos_traj(qpos_traj)
                    
                    covered_idx.append(neighbor_idx[i])
                    prefix_lengths.append(prefix_len)
                    covered_goals.append(n_bin_goal)
                    
                if (suffix is not None) or is_globally_covered:
                    next_layer.extend(indexer.adjacent_neighbors(keys_arr[neighbor_idx[i]]))

                else:
                    if fail_bridge_budget > 0:
                        next_layer.extend(indexer.adjacent_neighbors(keys_arr[neighbor_idx[i]]))
                        fail_bridge_budget -= 1

            print(f"Layer added. Covered: {len(covered_idx)-covered_before}")
            print(f"Current candidate coverage: {len(covered_idx)}")
            #plotter.plot_bin_and_neighbors_3d_shell(idx, covered_idx)

            if len(next_layer)>0:
                cleared_queue = False

                #next_layer = [n for n in next_layer if n != idx and n not in covered_key_indices]
                next_layer = [n for n in next_layer if n != idx]
                next_layer = list(dict.fromkeys(next_layer))
                next_layer = [n for n in next_layer if n not in checked_bins]
                neighbor_idx = next_layer
            else:
                cleared_queue = True
            
            if len(checked_bins) > max_checked_bins or suffix_checks > max_suffix_checks:
                cleared_queue = True
                if len(covered_idx)==0 and suffix_checks >= 0.8*max_suffix_checks and len(checked_bins)>=0.8*max_checked_bins:
                    print(f"Blacklisting bin: {idx}")
                    blacklist.add(idx)
                break

        print(f"Candidate root covered {len(covered_idx)} paths")

        is_root = len(covered_idx)>=1
        #is_root = len(covered_idx) >= max(K_abs, np.ceil(rho * len(checked_bins)))
        is_root = len(covered_idx)>= K_abs

        U = len(uncovered_key_indices)
        A_max = int(500 + 30*np.sqrt(U))   
        G_min = max(2, int(0.01 * len(keys_arr)))

        if is_root: 

            #print(f"Root path added {idx} - covers {len(covered_idx)} paths")
            curr_root_idx = len(roots_idx)
            root_paths[curr_root_idx] = path
            tuple_idx = tuple(tuple(row) for row in keys_arr[idx])
            prefix_map[tuple_idx] = (f"{curr_root_idx}", path[-1].tolist())

            for i, i_cover in enumerate(covered_idx):
                tuple_idx = tuple(tuple(row) for row in keys_arr[i_cover])
                #prefix_map[tuple(keys_arr[i_cover])] = f"{curr_root_idx}-{prefix_lengths[i]}"
                prefix_map[tuple_idx] = (f"{curr_root_idx}-{prefix_lengths[i]}", covered_goals[i].tolist())

            newly = set(covered_idx)
            newly.add(idx)
            covered_key_indices |= newly
            uncovered_key_indices -= newly

            #covered_key_indices.update(covered_idx)
            roots_idx.append(idx)
            print(f"Root path added - covers {len(covered_idx)} paths.")
            print(f"Current number of root paths: {len(roots_idx)}")
            print(f"Current number of covered paths: {len(covered_key_indices)}")

            recent_coverages.append(len(covered_idx))

            if len(recent_coverages) > W_max:
                recent_coverages.pop(0)
            
            fail_streak = 0

        else:
            print(f"Root path rejected {idx}")
            fail_streak += 1
        
        # Exit condition

        half_tail = len(uncovered_key_indices) <= 0.5 * len(keys_arr)
        if half_tail:
            print(f"Half tail reached. Adding remaining {len(uncovered_key_indices)} bins as root paths. Current number of root paths: {len(root_paths)}")
            root_paths, prefix_map = add_remaining_bins(data, offsets, uncovered_key_indices, root_paths, prefix_map, keys_arr, roots_idx, covered_key_indices)

            break
        
        tail = len(uncovered_key_indices) <= exit_factor * len(keys_arr)
        if tail:
            if len(recent_coverages) >= 5:
                W = min(W_max, max(5, len(recent_coverages)))
                window = recent_coverages[-W:]
                window_sorted = sorted(window)
                G_best = window_sorted[-2] if len(window_sorted) >= 2 else window_sorted[-1]
                
                if G_best <= G_best_min or fail_streak >= FAIL_STREAK_MAX:
                    print(f"Exit condition reached (uncovered={len(uncovered_key_indices)}, G_best={G_best:.1f}).")
                    root_paths, prefix_map = add_remaining_bins(data, offsets, uncovered_key_indices, root_paths, prefix_map, keys_arr, roots_idx, covered_key_indices)

                    break
        else:
            
            if fail_streak >= A_accept_max:
                print("No progress exit: promoting remaining uncovered bins to roots.") 
                root_paths, prefix_map = add_remaining_bins(data, offsets, uncovered_key_indices, root_paths, prefix_map, keys_arr, roots_idx, covered_key_indices)
                break
            
            if len(recent_coverages) >= 5:
                W = min(W_max, len(recent_coverages))
                window = recent_coverages[-W:]
                G_best = np.percentile(window, 90)

                if fail_streak >= A_max and G_best <= G_min:
                    print("No progress exit: promoting remaining uncovered bins to roots.") 
                    root_paths, prefix_map = add_remaining_bins(data, offsets, uncovered_key_indices, root_paths, prefix_map, keys_arr, roots_idx, covered_key_indices)
                    break
            



    #print(f"Number of root paths: {len(roots_idx)}")
    #print(f"Paths covered: {len(covered_key_indices)}/{len(keys_arr)}")     
    print("Condensation complete.")
    print(f"Number of root paths: {len(root_paths)}")
    print(f"Paths covered: {len(prefix_map)}/{len(keys_arr)}")

    #print(f"Length of root_paths: {len(root_paths)}")
    #print(f"Length of prefix_map: {len(prefix_map)}")

    return root_paths, prefix_map

def add_remaining_bins(data, offsets, uncovered_key_indices, root_paths, prefix_map, keys_arr, roots_idx, covered_key_indices):
    for i, i_uncov in enumerate(uncovered_key_indices):
        leftover_path = get_path_by_index(data, offsets, i_uncov)
        if leftover_path.size == 0:
            leftover_goal = []
        else:
            leftover_goal = leftover_path[-1].tolist()
        curr_root_idx = len(root_paths)
        root_paths[curr_root_idx] = leftover_path
        tuple_idx = tuple(tuple(row) for row in keys_arr[i_uncov])
        #prefix_map[tuple_idx] = f"{curr_root_idx}"
        prefix_map[tuple_idx] = (f"{curr_root_idx}", leftover_goal)

    print(f"Adding {len(uncovered_key_indices)} unreachable bins to root paths")
    newly = set(uncovered_key_indices)
    covered_key_indices |= newly
    roots_idx.extend(newly)
    uncovered_key_indices -= newly

    return root_paths, prefix_map
