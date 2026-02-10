import genesis as gs # type: ignore
import torch # type: ignore
import numpy as np
import time
import random

from tqdm import tqdm #type: ignore

from scipy.spatial.transform import Rotation as R #type: ignore
from src.swept_volume import SweptVolumeCube
from utils.helpers import rpy_to_R, to_genesis_quat, load_store, get_path_by_index
from src.condense_paths import PrefixBuilder, SuffixBuilder, SparseBoxGrid4D
from src.visualization import Plotter


from planning import omplPlanner

def batched_IK(robot, i, homePos, iTSR_batch, Tew, envs_to_use, ik_max_attempts=1000):

    rng = np.random.default_rng()

    iTSR = np.stack(iTSR_batch, axis=0)

    B = iTSR.shape[0]
    num_links = robot.n_links
    ee_link1_idx = num_links - 2
    ee_link2_idx = num_links - 1

    alive = np.ones(B, dtype=bool)     
    attempts = np.zeros(B, dtype=np.int32)

    q_batch = np.zeros((B, robot.n_dofs), dtype=np.float32)
    ik_ok = np.zeros(B, dtype=bool)

    have_target = np.zeros(B, dtype=bool)
    target_pos = np.zeros((B, 3), np.float32)
    target_quat = np.zeros((B, 4), np.float32)
    seeds = np.tile(homePos, (B, 1)).astype(np.float32)
    max_local = 10

    ik_attempt = 0
    curr_ik_failures = 0
    q_sol = np.zeros_like(q_batch, dtype=np.float32)

    while alive.any():
    
        if ik_attempt==0:
            old_q = np.zeros((B, robot.n_dofs), dtype=np.float32)
        else:
            old_q = q_batch

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
            envs_idx=idx,
            dofs_idx_local=[0, 1, 2, 3, 4, 5, 6]
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
            #q[-1] = 0.065; q[-2] = 0.065
            #q[-1] = 0.04; q[-2] = 0.04
            q_batch[env_id]  = q.detach().cpu().numpy()
            moved.append(env_id)

        q_delta = q_batch - old_q
        q_delta_means = np.mean(q_delta, axis=1)

        #print(f"Moved: {moved}")
        
        if moved:
            moved = np.asarray(moved, dtype=int)
            robot.set_dofs_position(q_batch[moved], envs_idx=moved, zero_velocity=True)
            #robot.set_qpos(q_batch[moved], envs_idx=moved, zero_velocity=True)
            #scene.step()
            #print(f"q_batch[moved]: {q_batch[moved]}")
            for env_id in moved:
                col = robot.detect_collision(env_idx=env_id)
                #print(f"Col: {col}")
                #input("Proceed?")

                if getattr(col, "size", 0)==0:
                    if not ik_ok[env_id]:
                        # Cache the FIRST valid IK for this environment
                        q_sol[env_id] = q_batch[env_id].copy()
                        robot.set_dofs_position(q_sol[env_id][None, :], envs_idx=[env_id], zero_velocity=True)
                        col2 = robot.detect_collision(env_idx=env_id)
                        #scene.step()
                        #print(f"q_sol[env_id]: {q_sol[env_id]}")
                        #print(f"Col2: {col2}")
                    ik_ok[env_id] = True
                    alive[env_id] = False
                    have_target[env_id] = False

        ik_attempt += 1

        if (ik_attempt>=ik_max_attempts):
            
            unsolved_idx = np.where(alive)[0]

            alive[unsolved_idx] = False
            have_target[unsolved_idx] = False  

            failed_key_indices = np.arange(i, i + envs_to_use)[unsolved_idx]
            print(f"Failed to solve {failed_key_indices} after {ik_max_attempts}. Giving up...")
            curr_ik_failures += len(unsolved_idx)
            q_sol[unsolved_idx] = q_batch[unsolved_idx]

    #ik_failures += curr_ik_failures
    print(f"IK complete: {i}-{i+envs_to_use}")

    return q_sol, ik_ok, curr_ik_failures

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

def cover_iTSR(robot, scene, homePos, iTSR_set, n_envs, object_details, Tew):

    ik_failures = 0
    planning_failures = 0
    planner = omplPlanner(robot)    
    object_size = object_details['size']
    keylist = list(iTSR_set)

    iTSR_paths = {}
    start = time.perf_counter()

    for i, key in enumerate(iTSR_set):
        if i%n_envs!=0:
            continue

        if(i==0):
            '''
            print("Initializing SV prebuild")
            print(key)
            if isinstance(key, np.ndarray):
                print("numpy array")
            elif isinstance(key, (list, tuple)):
                print("python sequence")
                print("outer:", type(key))
                print("inner:", type(key[0]))
            else:
                print("something else")
            '''
            SV = SweptVolumeCube(scene, object_size, n_envs=n_envs)
            SV.update(object_configs=key, env_idx=None)
            
        envs_to_use = min(n_envs, len(range(i, len(iTSR_set))))

        for env_idx in range(envs_to_use):
            SV.update(object_configs=keylist[i+env_idx], env_idx=[env_idx])

        robot.set_dofs_position(
            torch.tile(
                torch.tensor(homePos, device=gs.device), (n_envs, 1)
            ),
        )
        scene.step()
        
        keylist = list(iTSR_set)
        current_keys = keylist[i : i + envs_to_use]
        iTSR_batch = [
            iTSR_set[curr_key][0] for curr_key in current_keys
        ]

        # IK
        q_sol, ik_ok, curr_ik_failures = batched_IK(robot, i, homePos, iTSR_batch, Tew, envs_to_use, ik_max_attempts=1000)
        ik_failures += curr_ik_failures

        # Planning
        planning_start = time.perf_counter()

        current_paths, curr_failed = plan_to_goal(robot, current_keys, q_sol, ik_ok, i, homePos)
        iTSR_paths.update(current_paths)

        planning_failures += curr_failed

        print(f"[batch] planned={len(current_keys)-curr_failed}, failed={curr_failed}")
        #print(f"Planning complete: {i}-{i+n_envs}")
        print(f"Planning complete: {i}-{i+envs_to_use}")

        current_time = time.perf_counter()
        print(f"Planning time: {current_time - planning_start:.6f} seconds")

        #print(f"KEYS TO GO: {len(iTSR_set)-i-n_envs}")
        print(f"KEYS TO GO: {len(iTSR_set)-i-envs_to_use}")

    print("Paths generated for all object positions")
    print(f"IK failures: {ik_failures}")
    print(f"Planning failures: {planning_failures}")

    print(f"{len(iTSR_paths)} paths saved. {len(iTSR_paths) - planning_failures} valid paths.")
    print(f"Total IK and planning time: {time.perf_counter() - start:.6f} seconds")    

    return iTSR_paths, SV

def cover_iTSR_condensed(robot, scene, homePos, iTSR_set, n_envs, object_details, Tew):
    ik_failures = 0
    planning_failures = 0
    planner = omplPlanner(robot)    
    object_size = object_details['size']
    keylist = list(iTSR_set)

    covered_key_indices = set()
    uncovered_key_indices = set(range(len(keylist)))

    # saved datastructures
    root_paths = {}
    prefix_map = {}

    # Prefix score thresholds, root must score higher to cover a subpath     
    prefix_t_max = 0.75
    prefix_t_min = 0.5

    # Coverage fractions, potiential root must cover this fraction of their uncovered neighbors to be a viable root
    f_max = 0.5
    f_min = 0.05

    roots_idx = []
    sv = SweptVolumeCube(scene, object_size)

def condense_paths(SV, robot, scene, prefix, n_envs, object_details):
    '''
    Takes a generated data structure and condenses it using a Prefix-Suffix strategy
    Outputs a collection of root paths and a collection of subpaths as references to the root paths
    '''
    # Currently only uses 1 env, regardless of n_envs

    object_size = object_details['size']
    # saved datastructures
    root_paths = {}
    prefix_map = {}

    # Prefix score thresholds, root must score higher to cover a subpath     
    prefix_t_max = 0.75
    prefix_t_min = 0.5

    # Coverage fractions, potiential root must cover this fraction of their uncovered neighbors to be a viable root
    f_max = 0.5
    f_min = 0.05

    roots_idx = []

    if SV is None:
        regenerated = False
        SV = SweptVolumeCube(scene, object_size)
    else:
        regenerated = True

    data, offsets, keys_arr = load_store(prefix, mmap_data=True)
    indexer = SparseBoxGrid4D(keys_arr)
    plotter = Plotter(keys_arr)
 
    covered_key_indices = set()
    uncovered_key_indices = set(range(len(keys_arr)))

    while uncovered_key_indices:

        covered_idx = []
        prefix_lengths = []
        failed_sample = True
        print("Sampling a new root path...")

        idx = random.choice(tuple(uncovered_key_indices))
        path = get_path_by_index(data, offsets, idx)

        # Uncovered neighbors of root
        neighbor_idx = indexer.neighbors_by_box(keys_arr[idx], R_meters=0.1)
        neighbor_idx = [n for n in neighbor_idx if n not in covered_key_indices]
        print("uncovered neighbor count:", len(neighbor_idx))

        pathPrefix = PrefixBuilder(scene, robot, SV, len(path))

        for n_idx in neighbor_idx:

            prefix_len = pathPrefix.find_prefix_len(path, keys_arr[n_idx])
            bin_q_path = get_path_by_index(data, offsets, n_idx)

            if bin_q_path.size==0:
                prefix_score, pos_err, rot_err = [0, np.inf, np.inf]
            else:
                bin_q_goal = bin_q_path[-1]
                prefix_score, pos_err, rot_err = pathPrefix.prefix_quality(path, bin_q_goal, prefix_len)

            # Find prefix score and fraction threshold , tapers down as more paths are covered

            u = min(1.0, (len(uncovered_key_indices)/len(keys_arr)))
            u = max(0.0, u)
            gamma = 2.0
            
            prefix_score_threshold = prefix_t_min + (prefix_t_max - prefix_t_min) * (u ** gamma)
            frac_req = f_min + (f_max - f_min) * (u ** gamma)

            if prefix_score > prefix_score_threshold:
                covered_idx.append(n_idx)
                prefix_lengths.append(prefix_len)

            #print(f"Root path {idx} - Neighbor index {n_idx} - {round(prefix_score, 2)}")

        # Accept roots that cover atleast 1 path if uncovered set is ~20%         
        if u > 0.2:
            if len(covered_idx) >= max(1, int(np.ceil(frac_req * len(neighbor_idx)))):   
                accept_root = True
            else:
                accept_root = False
        else:
            if len(covered_idx) >= 1:
                accept_root = True
            else:
                accept_root = False

        if (accept_root):
            
            # sampled path is worth adding as a root
            
            curr_root_idx = len(roots_idx)
            root_paths[curr_root_idx] = path
            tuple_idx = tuple(tuple(row) for row in keys_arr[idx])
            #prefix_map[tuple(keys_arr[idx])] = f"{curr_root_idx}"
            prefix_map[tuple_idx] = f"{curr_root_idx}"

            for i, i_cover in enumerate(covered_idx):
                tuple_idx = tuple(tuple(row) for row in keys_arr[i_cover])
                #prefix_map[tuple(keys_arr[i_cover])] = f"{curr_root_idx}-{prefix_lengths[i]}"
                prefix_map[tuple_idx] = f"{curr_root_idx}-{prefix_lengths[i]}"


            #print(root_paths)
            #print(prefix_map)

            newly = set(covered_idx)
            newly.add(idx)
            covered_key_indices |= newly
            uncovered_key_indices -= newly

            #covered_key_indices.update(covered_idx)
            roots_idx.append(idx)
            print(f"Root path added. Covered {len(covered_idx)} paths.")
            #print(f"Added {len(newly)-1} + 1 paths to covered set")
            print(f"Current number of root paths: {len(roots_idx)}")
            print(f"Current number of covered paths: {len(covered_key_indices)}")

            #plotter.plot_root_and_covered(roots_idx, covered_key_indices, show_all=True)
        else:
            print(f"Root path rejected. Covered {len(covered_idx)} paths.")

            if len(uncovered_key_indices) <= 0.12 * len(keys_arr):
            
                for i, i_uncov in enumerate(uncovered_key_indices):
                    leftover_path = get_path_by_index(data, offsets, i_uncov)
                    curr_root_idx = len(root_paths)
                    root_paths[curr_root_idx] = leftover_path
                    tuple_idx = tuple(tuple(row) for row in keys_arr[i_uncov])
                    prefix_map[tuple_idx] = f"{curr_root_idx}"
                
            
                print("Adding unreachable bins to root paths")
                newly = set(uncovered_key_indices)
                covered_key_indices |= newly
                roots_idx.extend(newly)
                uncovered_key_indices -= newly
            
    print(f"Number of root paths: {len(roots_idx)}")
    print(f"Paths covered: {len(covered_key_indices)}/{len(keys_arr)}")

    covered_key_indices = list(covered_key_indices)
    plotter.plot_root_and_covered(roots_idx, covered_key_indices, show_all=True)        

    print(f"Length of root_paths: {len(root_paths)}")
    print(f"Length of prefix_map: {len(prefix_map)}")


    return root_paths, prefix_map

def condense_paths3(SV, robot, scene, filename_prefix, n_envs, object_details):
    '''
    Takes a generated data structure and condenses it using a Prefix-Suffix strategy
    Outputs a collection of root paths and a collection of subpaths as references to the root paths
    '''

    object_size = object_details['size']
    # saved datastructures
    root_paths = {}
    prefix_map = {}

    roots_idx = []

    data, offsets, keys_arr = load_store(filename_prefix, mmap_data=True)
    print(f"Number of bins: {len(keys_arr)}")
    indexer = SparseBoxGrid4D(keys_arr)
    plotter = Plotter(keys_arr, indexer)
 
    if SV is None:
        SV = SweptVolumeCube(scene, object_size, n_envs=n_envs)
        SV.update(object_configs=keys_arr[0], env_idx=None)

    covered_key_indices = set()
    uncovered_key_indices = set(range(len(keys_arr)))
    recent_coverages = []

    exit_factor = 0.05
    rho = 0.01
    G_best_min = 25
    W_max = 20

    fail_streak = 0
    FAIL_STREAK_MAX = 100
    A_accept_max = min(2000, max(100, int(0.01 * len(keys_arr))))

    while uncovered_key_indices:
        
        if len(uncovered_key_indices) > int(0.5*len(keys_arr)):
            K_abs = 50
        else:
            K_abs = 25

        covered_idx = []
        prefix_lengths = []
        covered_goals = []
        checked_bins = set()
        print("Sampling a new root path...")

        #idx = random.choice(tuple(uncovered_key_indices))
        
        # Sampling a candidate root 
        #S = 200
        #cands = random.sample(list(uncovered_key_indices), min(S, len(uncovered_key_indices)))
        #best = max(cands, key=lambda idx: len(indexer.adjacent_neighbors(keys_arr[idx])))
        #idx = best
        
        cands = [i for i in uncovered_key_indices
                if any(n not in covered_key_indices
                    for n in indexer.adjacent_neighbors(keys_arr[i]))]

        if not cands:
            idx = random.choice(tuple(uncovered_key_indices))
        else:
            idx = random.choice(cands)
        path = get_path_by_index(data, offsets, idx)

        #if (path is None):
        if path is None or getattr(path, "size", 0) == 0:
            print("Empty candidate root. Skipping...")
            fail_streak += 1
            continue

        pathPrefix = PrefixBuilder(scene, robot, SV, len(path))
        pathSuffix = SuffixBuilder(scene, robot, SV, len(path))

        # Uncovered neighbors of root
        #neighbor_idx = indexer.neighbors_by_box(keys_arr[idx], R_meters=0.1, R_yaw=0.3)
        neighbor_idx = indexer.adjacent_neighbors(keys_arr[idx])
        neighbor_idx = list(dict.fromkeys(neighbor_idx))

        neighbor_idx = [n for n in neighbor_idx if n != idx and n not in covered_key_indices]
        neighbor_idx = [n for n in neighbor_idx if n not in checked_bins]
        #print("uncovered neighbor count:", len(neighbor_idx))

        cleared_queue = (len(neighbor_idx)==0)

        while (not cleared_queue):    
            print(f"Checking {len(neighbor_idx)} neighbors...")
            next_layer = []
            covered_before = len(covered_idx)

            #plotter.plot_bin_and_neighbors(idx, neighbor_idx, show_all=True)
            #for n_idx in neighbor_idx:
            #for start in tqdm(range(0, len(neighbor_idx), n_envs)):
            for start in range(0, len(neighbor_idx), n_envs):

                chunk = neighbor_idx[start : start + n_envs]
                B = len(chunk)
                envs_idx = list(range(B))

                boxes_batch = keys_arr[chunk]
                prefix_lens = pathPrefix.find_prefix_lens(path, boxes_batch, envs_idx)

                batched_q_paths = []
                for n_idx in chunk:
                    curr_path = get_path_by_index(data, offsets, n_idx)
                    batched_q_paths.append(curr_path)
        

                batched_q_goals = []
                for bin_q_path in batched_q_paths:
                    if bin_q_path.size==0:
                        batched_q_goals.append(None)
                    else:
                        batched_q_goals.append(bin_q_path[-1])
                
                #print("start suffix")
                suffixes = pathSuffix.ik_suffix_batch(path, prefix_lens, batched_q_goals, boxes_batch, envs_idx)  

                for j, n_idx in enumerate(chunk):
                    suffix = suffixes[j]
                    checked_bins.add(n_idx)

                    if suffix is not None:
                        covered_idx.append(n_idx)
                        prefix_lengths.append(prefix_lens[j])
                        covered_goals.append(batched_q_goals[j])
                        #print(f"Potential root path {idx} covered neighbor path {n_idx}")

                        next_layer.extend(indexer.adjacent_neighbors(keys_arr[n_idx]))
                
            print(f"Layer added. Covered: {len(covered_idx)-covered_before}")
            print(f"Current candidate coverage: {len(covered_idx)}")
            #plotter.plot_bin_and_neighbors_3d_shell(idx, covered_idx)

            if len(next_layer)>0:
                cleared_queue = False

                next_layer = [n for n in next_layer if n != idx and n not in covered_key_indices]
                next_layer = list(dict.fromkeys(next_layer))
                next_layer = [n for n in next_layer if n not in checked_bins]
                neighbor_idx = next_layer
            else:
                cleared_queue = True

        print(f"Candidate root covered {len(covered_idx)} paths")

        #is_root = len(covered_idx)>=1
        is_root = len(covered_idx) >= max(K_abs, np.ceil(rho * len(checked_bins)))


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

        tail = len(uncovered_key_indices) <= exit_factor * len(keys_arr)
        if tail:
            if len(recent_coverages) >= 5:
                W = min(W_max, max(5, len(recent_coverages)))
                window = recent_coverages[-W:]
                window_sorted = sorted(window)
                G_best = window_sorted[-2] if len(window_sorted) >= 2 else window_sorted[-1]
                
                if G_best <= G_best_min or fail_streak >= FAIL_STREAK_MAX:
                    print(f"Exit condition reached (uncovered={len(uncovered_key_indices)}, G_best={G_best:.1f}).")

                    '''
                    for i, i_uncov in enumerate(uncovered_key_indices):
                        leftover_path = get_path_by_index(data, offsets, i_uncov)
                        curr_root_idx = len(root_paths)
                        root_paths[curr_root_idx] = leftover_path
                        tuple_idx = tuple(tuple(row) for row in keys_arr[i_uncov])
                        prefix_map[tuple_idx] = f"{curr_root_idx}"
                    
                
                    print(f"Adding {len(uncovered_key_indices)} unreachable bins to root paths")
                    newly = set(uncovered_key_indices)
                    covered_key_indices |= newly
                    roots_idx.extend(newly)
                    uncovered_key_indices -= newly
                    '''
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
