import numpy as np
import torch # type: ignore


#from planning import omplPlanner
from utils.helpers import _wrap_pi
from utils.helpers import to_numpy, fk_query, fk_query_batch, fk_query_env, quat_angle_diff, quat_mul, quat_mul_t

from scipy.spatial.transform import Rotation as R, Slerp #type: ignore

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
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])  # back to wxyz

def slerp_torch(q0, q1, t, eps=1e-8):
    # q0,q1: (...,4) wxyz, t: (...) or (...,1)
    q0 = q0 / (torch.linalg.norm(q0, dim=-1, keepdim=True) + eps)
    q1 = q1 / (torch.linalg.norm(q1, dim=-1, keepdim=True) + eps)

    dot = torch.sum(q0*q1, dim=-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)
    dot = torch.clamp(dot, -1.0, 1.0)

    # If very close, lerp to avoid numerical issues
    close = dot > 1.0 - 1e-5
    t_ = t[..., None] if t.ndim == dot.ndim - 1 else t  # ensure (...,1)

    lerp = q0 + t_*(q1 - q0)
    lerp = lerp / (torch.linalg.norm(lerp, dim=-1, keepdim=True) + eps)

    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    s0 = torch.sin((1 - t_) * theta) / (sin_theta + eps)
    s1 = torch.sin(t_ * theta) / (sin_theta + eps)

    out = s0*q0 + s1*q1
    out = out / (torch.linalg.norm(out, dim=-1, keepdim=True) + eps)

    return torch.where(close, lerp, out)


class PrefixBuilder:
    def __init__(self, scene, robot, swept_volume, path_length, min_suffix_factor=0.1):
        self.scene = scene
        self.robot = robot
        self.swept_volume = swept_volume
        self.path_length = path_length
        self.min_suffix = round(min_suffix_factor * self.path_length)
    
    def find_prefix_len(self, root_path, bin_box, backoff=20):
        '''
        returns the index upto which the root path is collision-free for new bin
        leaves min_suffix portion of waypoints unchecked for suffix
        '''
        k_max = max(0, self.path_length - self.min_suffix)
        self.swept_volume.update(bin_box, env_idx=None)

        for i in range(k_max):
            q = np.array(root_path[i], dtype=np.float32, copy=True)
            self.robot.set_dofs_position(q)

            if len(self.robot.detect_collision())>0:
                #return i
                return max(0, (i - 1) - backoff)

        return k_max
    
    def find_prefix_lens(self, root_path, boxes_batch, envs_idx, backoff=20):
        '''
        returns the index upto which the root path is collision-free for new bin
        leaves min_suffix portion of waypoints unchecked for suffix
        batched version of find_prefix_lens
        '''
        B = len(envs_idx)
        k_max = max(0, self.path_length - self.min_suffix)
        
        self.swept_volume.update(boxes_batch, env_idx=envs_idx)

        prefix_lens = np.full(B, k_max, dtype=np.int32)
        alive = np.ones(B, dtype=bool)

        for i in range(k_max):
            active_js   = [j for j in range(B) if alive[j]]
            if not active_js:
                break
            active_envs = [envs_idx[j] for j in active_js]

            q = np.asarray(root_path[i], dtype=np.float32)
            q_batch = np.tile(q[None, :], (len(active_envs), 1))
            self.robot.set_dofs_position(q_batch, envs_idx=active_envs)

            for j, e in zip(active_js, active_envs):
                if len(self.robot.detect_collision(env_idx=e)) > 0:
                    prefix_lens[j] = max(0, (i - 1) - backoff)
                    alive[j] = False

        return prefix_lens

    def visualize_prefix(self, root_path, q_goal, prefix_len):
        '''
        '''
        q_path_np = np.array(root_path, dtype=np.float32, copy=True)
        q_path_cut = torch.from_numpy(q_path_np)

        q_goal_np = np.array(q_goal, dtype=np.float32, copy=True)
        q_goal = torch.from_numpy(q_goal_np)

        input("Visualize prefix?")
        for i in range(prefix_len):
            self.robot.set_dofs_position(q_path_cut[i])
            self.scene.step()

        input("Visualize true goal?")
        self.robot.set_dofs_position(q_goal)
        self.scene.step()

    def prefix_quality(self, root_path, q_goal, prefix_len, 
                        p0=0.15, r0=np.deg2rad(15), w1=0.05, w2=0.15, reuse_frac=0.8):
        '''
        '''    
        #q_cut = torch.tensor(np.asarray(root_path[prefix_len-1]).copy(), dtype=torch.float32)
        #q_cut = torch.tensor(q_cut, dtype=torch.float32)

        if prefix_len==0:
            return 0, np.inf, np.inf

        q_np = np.array(root_path[prefix_len-1], dtype=np.float32, copy=True)
        q_cut = torch.from_numpy(q_np)

        q_goal_np = np.array(q_goal, dtype=np.float32, copy=True)
        q_goal = torch.from_numpy(q_goal_np)

        w_cut = fk_query(self.robot, q_cut)
        w_goal = fk_query(self.robot, q_goal)

        pos_err_arr = [np.inf, np.inf]
        rot_err_arr = [np.inf, np.inf]

        for f in [0, 1]:
            p = w_cut[0][f].detach().cpu().numpy()
            pg = w_goal[0][f].detach().cpu().numpy()
            pos_err = np.linalg.norm(pg - p)
            pos_err_arr[f] = pos_err

            q = w_cut[1][f].detach().cpu().numpy()
            qg = w_goal[1][f].detach().cpu().numpy()
            rot_err = quat_angle_diff(qg, q)
            rot_err_arr[f] = rot_err

        pos_err = max(pos_err_arr)
        rot_err = max(rot_err_arr)

        #S_align = np.exp(-pos_err/p0) * np.exp(-rot_err/r0)
        S_align = np.exp(-(pos_err/p0)**2) * np.exp(-(rot_err/r0)**2)

        return float(S_align), pos_err, rot_err

class SuffixBuilder:
    def __init__(self, scene, robot, swept_volume, path_length):
        self.scene = scene
        self.robot = robot
        self.swept_volume = swept_volume
        self.path_length = path_length

    def ik_suffix(self, root_path, prefix_length, q_goal, bin_box, n_envs):
        '''
        root_path: entire root path to evaluate
        prefix_length: length upto which root is valid
        q_goal: goal configuration of current bin
        bin_box: xy bounds of relevant bin
        '''

        num_links = self.robot.n_links
        ee_link1_idx = num_links - 2
        ee_link2_idx = num_links - 1
        
        prefix = root_path[:prefix_length]
        root_w = fk_query_env(self.robot, root_path[-1])
        goal_w = fk_query_env(self.robot, q_goal)
        #print(f"goal_w: {goal_w}")
        #print(f"root_w: {root_w}")

        goal_pos = goal_w[0]
        goal_f1_pos = goal_pos[0]
        goal_f2_pos = goal_pos[1]

        goal_quat = goal_w[1]
        goal_f1_quat = goal_quat[0]
        goal_f2_quat = goal_quat[1]

        root_pos = root_w[0]
        root_f1_pos = root_pos[0]
        root_f2_pos = root_pos[1]

        root_quat = root_w[1]
        root_f1_quat = root_quat[0]
        root_f2_quat = root_quat[1]

        delta_f1_pos = np.array(goal_f1_pos) - np.array(root_f1_pos)
        delta_f2_pos = np.array(goal_f2_pos) - np.array(root_f2_pos)
        
        if np.dot(root_f1_quat, goal_f1_quat) < 0:
            goal_f1_quat = -goal_f1_quat
        
        if np.dot(root_f2_quat, goal_f2_quat) < 0:
            goal_f2_quat = -goal_f2_quat
        
        root_f1_quat_conj = np.array([
            root_f1_quat[0],
            -root_f1_quat[1],
            -root_f1_quat[2],
            -root_f1_quat[3]
        ])

        root_f2_quat_conj = np.array([
            root_f2_quat[0],
            -root_f2_quat[1],
            -root_f2_quat[2],
            -root_f2_quat[3]
        ])

        delta_f1_quat = quat_mul(goal_f1_quat, root_f1_quat_conj)
        delta_f2_quat = quat_mul(goal_f2_quat, root_f2_quat_conj)

        delta_w = [[delta_f1_pos, delta_f2_pos], [delta_f1_quat, delta_f2_quat]]
        #print(delta_w)

        suffix = []

        for i in range(prefix_length, self.path_length):
            alpha = (i - prefix_length)/((self.path_length - 1) - prefix_length)

            q_root_i = root_path[i]
            w_root_i = fk_query_env(self.robot, q_root_i)
            w_pos_i = w_root_i[0]
            w_pos_f1_i = w_pos_i[0]
            w_pos_f2_i = w_pos_i[1]

            w_quat_i = w_root_i[1]
            w_quat_f1_i = w_quat_i[0]
            w_quat_f2_i = w_quat_i[1]

            recon_f1_pos = np.array(w_pos_f1_i) + alpha*delta_w[0][0]
            recon_f2_pos = np.array(w_pos_f2_i) + alpha*delta_w[0][1]

            w_quat_full_f1 = quat_mul(delta_w[1][0], w_quat_f1_i)
            recon_f1_quat = slerp(w_quat_f1_i, w_quat_full_f1, alpha)

            w_quat_full_f2 = quat_mul(delta_w[1][1], w_quat_f2_i)
            recon_f2_quat = slerp(w_quat_f2_i, w_quat_full_f2, alpha)

            # Batching for IK

            recon_f1_pos_tile = np.tile(recon_f1_pos.copy(), (n_envs, 1))
            recon_f2_pos_tile = np.tile(recon_f2_pos.copy(), (n_envs, 1))
            recon_f1_quat_tile = np.tile(recon_f1_quat.copy(), (n_envs, 1))
            recon_f2_quat_tile = np.tile(recon_f2_quat.copy(), (n_envs, 1))

            prefix_end_tile = np.tile(prefix[-1].copy(), (n_envs, 1))
            q_goal_tile = np.tile(q_goal.copy(), (n_envs, 1))

            q_recon = self.robot.inverse_kinematics_multilink(
                links=[self.robot.links[ee_link1_idx], self.robot.links[ee_link2_idx]],
                poss=[recon_f1_pos_tile, recon_f2_pos_tile], quats=[recon_f1_quat_tile, recon_f2_quat_tile],
                init_qpos=prefix_end_tile,             
                pos_tol=5e-4, rot_tol=5e-3,          # sane tolerances (yours were ~1e-10: too tight)
                dofs_idx_local=[0, 1, 2, 3, 4, 5, 6]
            )

            self.robot.set_dofs_position(q_recon)

            if (i==prefix_length):
                prev_f1_pos = recon_f1_pos
                prev_f1_quat = recon_f1_quat
                prev_f2_pos = recon_f2_pos
                prev_f2_quat = recon_f2_quat
                prev_q_recon = q_recon

            if len(self.robot.detect_collision())>0:
                #print("Suffix collision - aborting")
                return None
            
            q_dev = np.array(q_recon) - np.array(prev_q_recon)
            q_dev_wrap = (q_dev + np.pi) % (2*np.pi) - np.pi
            epsilon_joint = 0.35
            epsilon = 0.8
            #if (np.linalg.norm(q_dev)>epsilon):
            if np.max(np.abs(q_dev_wrap)) > epsilon_joint or np.linalg.norm(q_dev_wrap) > epsilon: 
                #print("Suffix discontinuity - aborting")
                return None

            suffix.append(q_recon)
        
            prev_f1_pos = recon_f1_pos
            prev_f1_quat = recon_f1_quat
            prev_f2_pos = recon_f2_pos
            prev_f2_quat = recon_f2_quat
            prev_q_recon = q_recon

        #print(suffix)
        return suffix

    def ik_suffix_query(self, root_path, prefix_lens, batched_q_goals, boxes_batch, envs_idx):
        B = len(envs_idx)
        num_links = self.robot.n_links
        ee_link1_idx = num_links - 2
        ee_link2_idx = num_links - 1
        epsilon_joint = 0.35
        epsilon = 0.8        
        #prefix = root_path[:prefix_length]

        root_w = fk_query_env(self.robot, root_path[-1], env=envs_idx[0])
    
        root_pos = root_w[0]
        root_f1_pos, root_f2_pos = root_pos

        root_quat = root_w[1]
        root_f1_quat, root_f2_quat = root_quat

        #print("norm q def")
        def normq(q):
            q = np.asarray(q, dtype=np.float64)
            return q / (np.linalg.norm(q) + 1e-12)
        
        def normq_torch(q: torch.Tensor, eps=1e-12):
            return q / (torch.linalg.norm(q) + eps)

        #print("norm q end")
        root_f1_quat = normq_torch(root_f1_quat)
        root_f2_quat = normq_torch(root_f2_quat)

        #root_f1_quat_conj = np.array([root_f1_quat[0], -root_f1_quat[1], -root_f1_quat[2], -root_f1_quat[3]])
        #root_f2_quat_conj = np.array([root_f2_quat[0], -root_f2_quat[1], -root_f2_quat[2], -root_f2_quat[3]])

        root_f1_quat_conj = root_f1_quat * torch.tensor([1,-1,-1,-1], device=root_f1_quat.device, dtype=root_f1_quat.dtype)
        root_f2_quat_conj = root_f2_quat * torch.tensor([1,-1,-1,-1], device=root_f2_quat.device, dtype=root_f2_quat.dtype)


        goal_ws = [None] * B
        for e in range(B):
            if batched_q_goals[e] is None:
                continue
            goal_ws[e] = fk_query_env(self.robot, batched_q_goals[e], env=envs_idx[0])
        #print("mid suffix")
        delta_w_batch = []
        for goal_w in goal_ws:
            if goal_w is None:
                delta_w_batch.append(None)
                continue

            goal_f1_pos, goal_f2_pos = goal_w[0]
            goal_f1_quat, goal_f2_quat = goal_w[1]

            goal_f1_quat = normq_torch(goal_f1_quat)
            goal_f2_quat = normq_torch(goal_f2_quat)

            #if np.dot(root_f1_quat, goal_f1_quat) < 0:
            #    goal_f1_quat = -goal_f1_quat
            #if np.dot(root_f2_quat, goal_f2_quat) < 0:
            #    goal_f2_quat = -goal_f2_quat

            if torch.dot(root_f1_quat, goal_f1_quat) < 0:
                goal_f1_quat = -goal_f1_quat
            if torch.dot(root_f2_quat, goal_f2_quat) < 0:
                goal_f2_quat = -goal_f2_quat

            #delta_f1_pos = np.asarray(goal_f1_pos) - np.asarray(root_f1_pos)
            #delta_f2_pos = np.asarray(goal_f2_pos) - np.asarray(root_f2_pos)

            delta_f1_pos = goal_f1_pos - root_f1_pos
            delta_f2_pos = goal_f2_pos - root_f2_pos

            delta_f1_quat = normq_torch(quat_mul_t(goal_f1_quat, root_f1_quat_conj))
            delta_f2_quat = normq_torch(quat_mul_t(goal_f2_quat, root_f2_quat_conj))

            delta_w_batch.append([[delta_f1_pos, delta_f2_pos], [delta_f1_quat, delta_f2_quat]])

        device = root_pos.device

        #print("mid suffix2")
        T_max = self.path_length
        valid = np.array([dw is not None for dw in delta_w_batch], dtype=bool)
        active = valid.copy()

        start_idx = np.array(prefix_lens, dtype=np.int32)
        end_idx = np.full(B, self.path_length, dtype=np.int32)

        suffixes = [None]*B
        suffix_lists = [[] for _ in range(B)]
        prev_q = [None]*B

        if active.any():
            max_steps = int(np.max(self.path_length - start_idx[active]))
        else:
            max_steps = 0


        dofs = 9
        seed_q_batch = np.zeros((B, dofs), dtype=np.float32)
        for e in range(B):
            if not valid[e]:   # e.g., delta_w_batch[e] is None
                continue
            seed_idx = max(0, int(start_idx[e]) - 1)   # prefix end index
            #seed_q_batch[e] = np.asarray(root_path[seed_idx], dtype=np.float32)
            seed_q_batch[e] = np.array(root_path[seed_idx], dtype=np.float32, copy=True)


        for t in range(max_steps):
            in_range = active & ((start_idx + t) < self.path_length)
            if not in_range.any():
                break

            active_envs = [envs_idx[e] for e in range(B) if in_range[e]]
            active_es = [e for e in range(B) if in_range[e]]
            Ba = len(active_es)

            i_batch = (start_idx[in_range] + t).astype(int)
            q_root_batch = np.stack([root_path[i] for i in i_batch], axis=0).astype(np.float32)

            pos_fk, quat_fk = fk_query_batch(self.robot, q_root_batch, envs_idx=active_envs)
            w_f1_pos = pos_fk[:, 0, :]   # [Ba,3]
            w_f2_pos = pos_fk[:, 1, :]   # [Ba,3]
            w_f1_quat = quat_fk[:, 0, :] # [Ba,4]
            w_f2_quat = quat_fk[:, 1, :] # [Ba,4]            

            # ---- alpha per env (depends on that env's suffix length) ----
            start_idx_t = torch.as_tensor(start_idx, device=device)
            in_range_t  = torch.as_tensor(in_range, device=device)
            
            #denom = ((self.path_length - 1) - start_idx[in_range]).astype(np.float32)  # [Ba]
            #denom = np.maximum(denom, 1.0)

            denom = (self.path_length - 1) - start_idx_t[in_range_t]
            denom = denom.to(dtype=torch.float32)
            denom = torch.clamp(denom, min=1.0)

            #alpha = (t / denom).astype(np.float32)  # [Ba]
            #alpha_col = alpha[:, None]              # [Ba,1]

            alpha = (t / denom)
            alpha = torch.as_tensor(alpha, device=device, dtype=torch.float32)
            alpha = alpha.to(device=device, dtype=torch.float32)
            alpha_col = alpha[:, None]

            # ---- build batched deltas for active envs ----
            #dp1 = np.stack([delta_w_batch[e][0][0] for e in active_es], axis=0).astype(np.float32)  # [Ba,3]
            #dp2 = np.stack([delta_w_batch[e][0][1] for e in active_es], axis=0).astype(np.float32)  # [Ba,3]
            #dq1 = np.stack([delta_w_batch[e][1][0] for e in active_es], axis=0).astype(np.float32)  # [Ba,4]
            #dq2 = np.stack([delta_w_batch[e][1][1] for e in active_es], axis=0).astype(np.float32)  # [Ba,4]

            dp1 = torch.stack([delta_w_batch[e][0][0] for e in active_es], dim=0).float()
            dp2 = torch.stack([delta_w_batch[e][0][1] for e in active_es], dim=0).float()
            dq1 = torch.stack([delta_w_batch[e][1][0] for e in active_es], dim=0).float()
            dq2 = torch.stack([delta_w_batch[e][1][1] for e in active_es], dim=0).float()


            # ---- positions: easy ----
            recon_f1_pos = w_f1_pos + alpha_col * dp1   # [Ba,3]
            recon_f2_pos = w_f2_pos + alpha_col * dp2   # [Ba,3]

            # ---- quats: per-row slerp (Ba is small, loop is fine) ----
            recon_f1_quat = np.zeros((Ba, 4), dtype=np.float32)
            recon_f2_quat = np.zeros((Ba, 4), dtype=np.float32)
            #for k in range(Ba):
            #    q1_full = quat_mul_t(dq1[k], w_f1_quat[k])
            #    q2_full = quat_mul_t(dq2[k], w_f2_quat[k])

                #q0_f1 = normq(w_f1_quat[k])
                #q1_f1 = normq(q1_full)
                #q0_f2 = normq(w_f2_quat[k])
                #q1_f2 = normq(q2_full)

                #if np.dot(q0_f1, q1_f1) < 0: q1 = -q1
                #if np.dot(q0_f2, q1_f1) < 0: q2 = -q2

                #recon_f1_quat[k] = slerp(q0_f1, q1_f1, float(alpha[k]))
                #recon_f2_quat[k] = slerp(q0_f2, q1_f2, float(alpha[k]))

            #    recon_f1_quat[k] = slerp_torch(normq_torch(w_f1_quat[k]), normq_torch(q1_full), float(alpha[k]))
            #    recon_f2_quat[k] = slerp_torch(normq_torch(w_f2_quat[k]), normq_torch(q2_full), float(alpha[k]))

            q1_full = quat_mul_t(dq1, w_f1_quat)      # (Ba,4)
            q2_full = quat_mul_t(dq2, w_f2_quat)      # (Ba,4)

            recon_f1_quat = slerp_torch(normq_torch(w_f1_quat), normq_torch(q1_full), alpha)  # (Ba,4)
            recon_f2_quat = slerp_torch(normq_torch(w_f2_quat), normq_torch(q2_full), alpha)  # (Ba,4)


            init_qpos = seed_q_batch[in_range].copy()   # shape [Ba, dofs]
            #in_range_t = torch.as_tensor(in_range, device=seed_q_batch.device)
            #init_qpos = seed_q_batch[in_range_t].clone() 
            # batched IK

            q_recon_batch = self.robot.inverse_kinematics_multilink(
                links=[self.robot.links[ee_link1_idx], self.robot.links[ee_link2_idx]],
                poss=[recon_f1_pos, recon_f2_pos],
                quats=[recon_f1_quat, recon_f2_quat],
                init_qpos=init_qpos,
                #pos_tol=5e-4, rot_tol=5e-3,
                pos_tol=1e-3, rot_tol=1e-2,
                dofs_idx_local=[0, 1, 2, 3, 4, 5, 6],
                envs_idx=active_envs,   # critical: first dim Ba must match len(active_envs)
                max_solver_iters=5,
                max_samples=1
            )

            self.robot.set_dofs_position(q_recon_batch, envs_idx=active_envs)

            # checking each env for suffix validity
            for k, e in enumerate(active_es):
                env = active_envs[k]
                #qk = np.asarray(q_recon_batch[k])
                qk = q_recon_batch[k]

                '''
                # collision
                if len(self.robot.detect_collision(env_idx=env)) > 0:
                    active[e] = False
                    suffix_lists[e] = None
                    continue

                # discontinuity
                if prev_q[e] is not None:
                    dq = qk - prev_q[e]
                    dq_wrap = (dq + np.pi) % (2*np.pi) - np.pi
                    #if np.max(np.abs(dq_wrap)) > epsilon_joint or np.linalg.norm(dq_wrap) > epsilon:
                    if torch.max(torch.abs(dq_wrap)) > epsilon_joint or torch.linalg.norm(dq_wrap) > epsilon:
                        active[e] = False
                        suffix_lists[e] = None
                        continue  
                '''
                if suffix_lists[e] is not None:
                    #suffix_lists[e].append(qk.copy())
                    #prev_q[e] = qk.copy()  
                    suffix_lists[e].append(qk.clone())
                    prev_q[e] = qk.clone()

        for e in range(B):
            if delta_w_batch[e] is None or suffix_lists[e] is None:
                suffixes[e] = None
            else:
                suffixes[e] = suffix_lists[e]
        return suffixes

    def ik_suffix_batch(self, root_path, prefix_lens, batched_q_goals, boxes_batch, envs_idx):
        '''
        root_path: entire root path to evaluate
        prefix_length: length upto which root is valid
        q_goal: goal configuration of current bin
        bin_box: xy bounds of relevant bin
        '''

        B = len(envs_idx)
        num_links = self.robot.n_links
        ee_link1_idx = num_links - 2
        ee_link2_idx = num_links - 1
        epsilon_joint = 0.35
        epsilon = 0.8        
        #prefix = root_path[:prefix_length]

        root_w = fk_query_env(self.robot, root_path[-1], env=envs_idx[0])
    
        root_pos = root_w[0]
        root_f1_pos, root_f2_pos = root_pos

        root_quat = root_w[1]
        root_f1_quat, root_f2_quat = root_quat

        #print("norm q def")
        def normq(q):
            q = np.asarray(q, dtype=np.float64)
            return q / (np.linalg.norm(q) + 1e-12)
        
        def normq_torch(q: torch.Tensor, eps=1e-12):
            return q / (torch.linalg.norm(q) + eps)

        #print("norm q end")
        root_f1_quat = normq_torch(root_f1_quat)
        root_f2_quat = normq_torch(root_f2_quat)

        #root_f1_quat_conj = np.array([root_f1_quat[0], -root_f1_quat[1], -root_f1_quat[2], -root_f1_quat[3]])
        #root_f2_quat_conj = np.array([root_f2_quat[0], -root_f2_quat[1], -root_f2_quat[2], -root_f2_quat[3]])

        root_f1_quat_conj = root_f1_quat * torch.tensor([1,-1,-1,-1], device=root_f1_quat.device, dtype=root_f1_quat.dtype)
        root_f2_quat_conj = root_f2_quat * torch.tensor([1,-1,-1,-1], device=root_f2_quat.device, dtype=root_f2_quat.dtype)


        goal_ws = [None] * B
        for e in range(B):
            if batched_q_goals[e] is None:
                continue
            goal_ws[e] = fk_query_env(self.robot, batched_q_goals[e], env=envs_idx[0])
        #print("mid suffix")
        delta_w_batch = []
        for goal_w in goal_ws:
            if goal_w is None:
                delta_w_batch.append(None)
                continue

            goal_f1_pos, goal_f2_pos = goal_w[0]
            goal_f1_quat, goal_f2_quat = goal_w[1]

            goal_f1_quat = normq_torch(goal_f1_quat)
            goal_f2_quat = normq_torch(goal_f2_quat)

            #if np.dot(root_f1_quat, goal_f1_quat) < 0:
            #    goal_f1_quat = -goal_f1_quat
            #if np.dot(root_f2_quat, goal_f2_quat) < 0:
            #    goal_f2_quat = -goal_f2_quat

            if torch.dot(root_f1_quat, goal_f1_quat) < 0:
                goal_f1_quat = -goal_f1_quat
            if torch.dot(root_f2_quat, goal_f2_quat) < 0:
                goal_f2_quat = -goal_f2_quat

            #delta_f1_pos = np.asarray(goal_f1_pos) - np.asarray(root_f1_pos)
            #delta_f2_pos = np.asarray(goal_f2_pos) - np.asarray(root_f2_pos)

            delta_f1_pos = goal_f1_pos - root_f1_pos
            delta_f2_pos = goal_f2_pos - root_f2_pos

            delta_f1_quat = normq_torch(quat_mul_t(goal_f1_quat, root_f1_quat_conj))
            delta_f2_quat = normq_torch(quat_mul_t(goal_f2_quat, root_f2_quat_conj))

            delta_w_batch.append([[delta_f1_pos, delta_f2_pos], [delta_f1_quat, delta_f2_quat]])

        device = root_pos.device

        #print("mid suffix2")
        T_max = self.path_length
        valid = np.array([dw is not None for dw in delta_w_batch], dtype=bool)
        active = valid.copy()

        start_idx = np.array(prefix_lens, dtype=np.int32)
        end_idx = np.full(B, self.path_length, dtype=np.int32)

        suffixes = [None]*B
        suffix_lists = [[] for _ in range(B)]
        prev_q = [None]*B

        if active.any():
            max_steps = int(np.max(self.path_length - start_idx[active]))
        else:
            max_steps = 0


        dofs = 9
        seed_q_batch = np.zeros((B, dofs), dtype=np.float32)
        for e in range(B):
            if not valid[e]:   # e.g., delta_w_batch[e] is None
                continue
            seed_idx = max(0, int(start_idx[e]) - 1)   # prefix end index
            #seed_q_batch[e] = np.asarray(root_path[seed_idx], dtype=np.float32)
            seed_q_batch[e] = np.array(root_path[seed_idx], dtype=np.float32, copy=True)


        for t in range(max_steps):
            in_range = active & ((start_idx + t) < self.path_length)
            if not in_range.any():
                break

            active_envs = [envs_idx[e] for e in range(B) if in_range[e]]
            active_es = [e for e in range(B) if in_range[e]]
            Ba = len(active_es)

            i_batch = (start_idx[in_range] + t).astype(int)
            q_root_batch = np.stack([root_path[i] for i in i_batch], axis=0).astype(np.float32)

            pos_fk, quat_fk = fk_query_batch(self.robot, q_root_batch, envs_idx=active_envs)
            w_f1_pos = pos_fk[:, 0, :]   # [Ba,3]
            w_f2_pos = pos_fk[:, 1, :]   # [Ba,3]
            w_f1_quat = quat_fk[:, 0, :] # [Ba,4]
            w_f2_quat = quat_fk[:, 1, :] # [Ba,4]            

            # ---- alpha per env (depends on that env's suffix length) ----
            start_idx_t = torch.as_tensor(start_idx, device=device)
            in_range_t  = torch.as_tensor(in_range, device=device)
            
            #denom = ((self.path_length - 1) - start_idx[in_range]).astype(np.float32)  # [Ba]
            #denom = np.maximum(denom, 1.0)

            denom = (self.path_length - 1) - start_idx_t[in_range_t]
            denom = denom.to(dtype=torch.float32)
            denom = torch.clamp(denom, min=1.0)

            #alpha = (t / denom).astype(np.float32)  # [Ba]
            #alpha_col = alpha[:, None]              # [Ba,1]

            alpha = (t / denom)
            alpha = torch.as_tensor(alpha, device=device, dtype=torch.float32)
            alpha = alpha.to(device=device, dtype=torch.float32)
            alpha_col = alpha[:, None]

            # ---- build batched deltas for active envs ----
            #dp1 = np.stack([delta_w_batch[e][0][0] for e in active_es], axis=0).astype(np.float32)  # [Ba,3]
            #dp2 = np.stack([delta_w_batch[e][0][1] for e in active_es], axis=0).astype(np.float32)  # [Ba,3]
            #dq1 = np.stack([delta_w_batch[e][1][0] for e in active_es], axis=0).astype(np.float32)  # [Ba,4]
            #dq2 = np.stack([delta_w_batch[e][1][1] for e in active_es], axis=0).astype(np.float32)  # [Ba,4]

            dp1 = torch.stack([delta_w_batch[e][0][0] for e in active_es], dim=0).float()
            dp2 = torch.stack([delta_w_batch[e][0][1] for e in active_es], dim=0).float()
            dq1 = torch.stack([delta_w_batch[e][1][0] for e in active_es], dim=0).float()
            dq2 = torch.stack([delta_w_batch[e][1][1] for e in active_es], dim=0).float()


            # ---- positions: easy ----
            recon_f1_pos = w_f1_pos + alpha_col * dp1   # [Ba,3]
            recon_f2_pos = w_f2_pos + alpha_col * dp2   # [Ba,3]

            # ---- quats: per-row slerp (Ba is small, loop is fine) ----
            recon_f1_quat = np.zeros((Ba, 4), dtype=np.float32)
            recon_f2_quat = np.zeros((Ba, 4), dtype=np.float32)
            #for k in range(Ba):
            #    q1_full = quat_mul_t(dq1[k], w_f1_quat[k])
            #    q2_full = quat_mul_t(dq2[k], w_f2_quat[k])

                #q0_f1 = normq(w_f1_quat[k])
                #q1_f1 = normq(q1_full)
                #q0_f2 = normq(w_f2_quat[k])
                #q1_f2 = normq(q2_full)

                #if np.dot(q0_f1, q1_f1) < 0: q1 = -q1
                #if np.dot(q0_f2, q1_f1) < 0: q2 = -q2

                #recon_f1_quat[k] = slerp(q0_f1, q1_f1, float(alpha[k]))
                #recon_f2_quat[k] = slerp(q0_f2, q1_f2, float(alpha[k]))

            #    recon_f1_quat[k] = slerp_torch(normq_torch(w_f1_quat[k]), normq_torch(q1_full), float(alpha[k]))
            #    recon_f2_quat[k] = slerp_torch(normq_torch(w_f2_quat[k]), normq_torch(q2_full), float(alpha[k]))

            q1_full = quat_mul_t(dq1, w_f1_quat)      # (Ba,4)
            q2_full = quat_mul_t(dq2, w_f2_quat)      # (Ba,4)

            recon_f1_quat = slerp_torch(normq_torch(w_f1_quat), normq_torch(q1_full), alpha)  # (Ba,4)
            recon_f2_quat = slerp_torch(normq_torch(w_f2_quat), normq_torch(q2_full), alpha)  # (Ba,4)


            init_qpos = seed_q_batch[in_range].copy()   # shape [Ba, dofs]
            #in_range_t = torch.as_tensor(in_range, device=seed_q_batch.device)
            #init_qpos = seed_q_batch[in_range_t].clone() 
            # batched IK

            q_recon_batch = self.robot.inverse_kinematics_multilink(
                links=[self.robot.links[ee_link1_idx], self.robot.links[ee_link2_idx]],
                poss=[recon_f1_pos, recon_f2_pos],
                quats=[recon_f1_quat, recon_f2_quat],
                init_qpos=init_qpos,
                pos_tol=5e-4, rot_tol=5e-3,
                dofs_idx_local=[0, 1, 2, 3, 4, 5, 6],
                envs_idx=active_envs,   # critical: first dim Ba must match len(active_envs)
            )

            self.robot.set_dofs_position(q_recon_batch, envs_idx=active_envs)

            # checking each env for suffix validity
            for k, e in enumerate(active_es):
                env = active_envs[k]
                #qk = np.asarray(q_recon_batch[k])
                qk = q_recon_batch[k]

                # collision
                if len(self.robot.detect_collision(env_idx=env)) > 0:
                    active[e] = False
                    suffix_lists[e] = None
                    continue

                # discontinuity
                if prev_q[e] is not None:
                    dq = qk - prev_q[e]
                    dq_wrap = (dq + np.pi) % (2*np.pi) - np.pi
                    #if np.max(np.abs(dq_wrap)) > epsilon_joint or np.linalg.norm(dq_wrap) > epsilon:
                    if torch.max(torch.abs(dq_wrap)) > epsilon_joint or torch.linalg.norm(dq_wrap) > epsilon:
                        active[e] = False
                        suffix_lists[e] = None
                        continue  

                if suffix_lists[e] is not None:
                    #suffix_lists[e].append(qk.copy())
                    #prev_q[e] = qk.copy()  
                    suffix_lists[e].append(qk.clone())
                    prev_q[e] = qk.clone()

        for e in range(B):
            if delta_w_batch[e] is None or suffix_lists[e] is None:
                suffixes[e] = None
            else:
                suffixes[e] = suffix_lists[e]
        return suffixes
        

    def find_suffix(self, prefix_path, q_goal, bin_box):
        '''
        '''
        suffix_length = self.path_length - len(prefix_path)
        prefix_path_tensor = torch.from_numpy(np.array(prefix_path, dtype=np.float32, copy=True))
        q_start_tensor = prefix_path_tensor[-1]
        q_goal_tensor = torch.from_numpy(np.array(q_goal, dtype=np.float32, copy=True))
        
        self.swept_volume.update(bin_box, env_idx=None)

        suffix_validity, suffix = self.suffix_interpolation(suffix_length, q_start_tensor, q_goal_tensor)

        if (suffix_validity):
            return suffix_validity, suffix
        else:
            print("Collision detected. Suffix generation failed.")
            return self.suffix_planning(suffix_length, q_start_tensor, q_goal_tensor)


    def suffix_interpolation(self, suffix_length, q_start_tensor, q_goal_tensor):    
        '''
        '''
        print("Interpolating for suffix...")
        if suffix_length <= 0:
            return []
        suffix = []
        for i in range(1, suffix_length + 1):
            t = i / suffix_length
            q = (1 - t) * q_start_tensor + t * q_goal_tensor
            
            self.robot.set_dofs_position(q)
            if len(self.robot.detect_collision())>0:
                print("Suffix interpolation failed")
                return (False, [])

            suffix.append(q)

        return (True, suffix)

    '''
    def suffix_planning(self, suffix_length, q_start_tensor, q_goal_tensor):
        planner = omplPlanner(self.robot)
        path = planner.omplPlan(
            qpos_goal = np.array(q_goal_tensor),
            qpos_start = np.array(q_start_tensor),
            num_waypoints = suffix_length,
            timeout = 1.0,
            planner="RRTConnect"
        )
        pathComplete = planner.checkCompletePath(path, np.array(q_goal_tensor))
        return pathComplete, path   
    '''

    def find_final_path(self, prefix, suffix):
        prefix = to_numpy(prefix)
        suffix = to_numpy(suffix)
        return np.concatenate((prefix, suffix), axis=0)
    
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

        '''
        def spacing(vals):
            vals = np.sort(np.unique(vals))
            diffs = np.diff(vals)
            diffs = diffs[diffs > 1e-9]
            return diffs.min()
        '''

        def spacing(vals, default=1.0):
            vals = np.sort(np.unique(vals))
            diffs = np.diff(vals)
            diffs = diffs[diffs > 1e-9]
            return diffs.min() if diffs.size else default

            
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
        iz = int(round((z_min - self.z0) / self.dz)) if self.z_has_variation else 0
        iyaw = int(round((yaw_min - self.yaw0) / self.dyaw)) % self.nyaw

        return (ix, iy, iz, iyaw)

    def query_point(self, x, y, z, yaw):
        yaw = _wrap_pi(yaw)

        ix   = int(np.floor((x   - self.x0) / self.dx))
        iy   = int(np.floor((y   - self.y0) / self.dy))
        iz   = int(np.floor((z   - self.z0) / self.dz)) if self.dz != 1.0 else 0
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
        R2 = R_meters*R_meters

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

