import mujoco #type: ignore
#from dm_control import mjcf #type: ignore
import mujoco.viewer #type: ignore
import time
import os
import numpy as np
import math
import yaml
import re
from pathlib import Path
from scipy.spatial.transform import Rotation as R # type: ignore

from mink import ( #type: ignore
    Configuration,
    ConfigurationLimit,
    FrameTask,
    PostureTask,
    VelocityLimit,
    solve_ik,
	SE3,
	SO3,
	DofFreezingTask,
	DampingTask,
	PostureTask,
	ConfigurationLimit,
	CollisionAvoidanceLimit
)
from mink.solve_ik import NoSolutionFound #type: ignore

def init_xml(name="test_scene"):
	# Initialize xml from scene.xml

	panda_dir = os.path.abspath("robots/mujoco_menagerie/franka_emika_panda")
	xml_path = os.path.join(panda_dir, f"{name}.xml")
	base_xml = "scene.xml"

	return base_xml, xml_path

def body_T(model, data, name):
    mujoco.mj_forward(model, data)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    Rw = data.xmat[bid].reshape(3,3).copy()
    pw = data.xpos[bid].copy()
    T = np.eye(4)
    T[:3,:3] = Rw
    T[:3,3] = pw
    return T

def panda_to_xml(pos=[0, 0, 0], quat=[1, 0, 0, 0]):
	panda_path = "panda.xml"
	panda_xml = f"""
	  <body name="robot_base" pos="{pos[0]} {pos[1]} {pos[2]}" quat="{quat[0]} {quat[1]} {quat[2]} {quat[3]}">
	    <include file="{panda_path}"/>
	  </body>
	"""
	return panda_xml

def write_relocated_panda(
    panda_src: str,
    panda_dst: str,
    base_pos=(0.0, 0.0, 0.0),
    base_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
    root_body_name="link0",
):
    src = Path(panda_src).read_text()

    pos_str = f'{base_pos[0]} {base_pos[1]} {base_pos[2]}'
    quat_str = f'{base_quat_wxyz[0]} {base_quat_wxyz[1]} {base_quat_wxyz[2]} {base_quat_wxyz[3]}'

    # Match the opening tag of the root body: <body name="link0" ...>
    pattern = rf'(<body\b[^>]*\bname="{re.escape(root_body_name)}"[^>]*)(>)'
    m = re.search(pattern, src)
    if not m:
        raise ValueError(f'Could not find <body name="{root_body_name}"> in {panda_src}')

    start_tag = m.group(1)  # everything before the closing '>'
    end = m.group(2)        # '>'

    # Remove existing pos/quat if present
    start_tag = re.sub(r'\spos="[^"]*"', '', start_tag)
    start_tag = re.sub(r'\squat="[^"]*"', '', start_tag)

    # Add pos + quat
    new_start_tag = f'{start_tag} pos="{pos_str}" quat="{quat_str}"{end}'

    out = src[:m.start()] + new_start_tag + src[m.end():]
    Path(panda_dst).write_text(out)

def cube_to_xml(name, pos, quat, size, fixed=False, rgba=[0.8, 0.8, 0.8, 1]): #size = [x, y, z] full lengths, converted to half lengths for xml
	joint_xml = "" if fixed else f'<joint name="{name}_free" type="free"/>'
	cube_xml_string = f"""
		<body name="{name}" quat= "{quat[0]} {quat[1]} {quat[2]} {quat[3]}" pos="{pos[0]} {pos[1]} {pos[2]}">
		{joint_xml}
		<geom name="{name}_geom" type="box" size="{size[0]/2} {size[1]/2} {size[2]/2}" rgba="{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}"/>
		</body>
	"""
	return cube_xml_string
  
def cylinder_to_xml(name, pos, quat, size, fixed=False, rgba=[0.8, 0.8, 0.8, 1]): #size = [r, h], converted to [r, h/2] for xml
	joint_xml = "" if fixed else f'<joint name="{name}_free" type="free"/>'
	cylinder_xml_string = f"""
		<body name="{name}" quat= "{quat[0]} {quat[1]} {quat[2]} {quat[3]}" pos="{pos[0]} {pos[1]} {pos[2]}">
		{joint_xml}
		<geom name="{name}_geom" type="cylinder" size="{size[0]} {size[1]/2}" rgba="{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}"/>
		</body>
	"""
	return cylinder_xml_string  

def move_cube(model, data, name, new_pos, new_quat):
	svid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
	sv_adr = model.jnt_qposadr[svid]
	sv_vadr = model.jnt_dofadr[svid]

	data.qpos[sv_adr: sv_adr + 7] = [new_pos[0], new_pos[1], new_pos[2], new_quat[0], new_quat[1], new_quat[2], new_quat[3]]
	data.qvel[sv_vadr: sv_vadr + 6] = 0

	mujoco.mj_forward(model, data)

def build_model(base_xml, xml_path, xmls_to_add):
	curr_xml = f"""
	<mujoco model="test_world">
	<include file="{base_xml}"/>
	<worldbody>
	"""
	for primitive_xml in xmls_to_add:
		curr_xml += f"{primitive_xml}"
  
	curr_xml += """
		</worldbody>
	</mujoco>
	"""

	with open(xml_path, "w") as f:
		f.write(curr_xml)
	
	model = mujoco.MjModel.from_xml_path(xml_path)
	data = mujoco.MjData(model)
	return model, data, xml_path

def compute_sv_params(object_dims, object_configs):
	object_configs = np.asarray(object_configs, dtype=np.float64)
	xdim, ydim, zdim = object_dims  
	if object_configs.ndim == 2:
		object_configs = object_configs[None, :, :]

	x_lower = object_configs[:, 0, 0]
	x_upper = object_configs[:, 0, 1]
	y_lower = object_configs[:, 1, 0]
	y_upper = object_configs[:, 1, 1]
	z = object_configs[:, 2, 0] 

	R_cyl = float(np.round(0.5 * np.sqrt(xdim**2 + ydim**2), 5))
	cx = 0.5 * (x_upper + x_lower)
	cy = 0.5 * (y_upper + y_lower)

	b1_size = np.array([ (x_upper - x_lower) + 2*R_cyl,
						(y_upper - y_lower),
						np.full_like(cx, zdim) ]).T   # (B,3)

	b2_size = np.array([ (x_upper - x_lower),
						(y_upper - y_lower) + 2*R_cyl,
						np.full_like(cx, zdim) ]).T   # (B,3)

	b_pos = np.stack([cx, cy, z], axis=1)  # (B,3)

	corners = np.stack([
		np.stack([x_lower, y_lower, z], axis=1),
		np.stack([x_lower, y_upper, z], axis=1),
		np.stack([x_upper, y_lower, z], axis=1),
		np.stack([x_upper, y_upper, z], axis=1),
	], axis=1)  # (B,4,3)

	return R_cyl, b_pos, b1_size, b2_size, corners

def cube_swept_volume_xml(object_dims, object_configs, fixed=False, rgba=[0.8, 0.8, 0.8, 1]):
	name = "swept_volume"
	joint_xml = "" if fixed else f'<joint name="{name}_free" type="free"/>'
	
	R_cyl, b_pos, b1_size, b2_size, corners = compute_sv_params(object_dims, object_configs)

	b_pos0 = b_pos[0]              # numpy (3,)
	b1_size0 = b1_size[0].tolist()
	b2_size0 = b2_size[0].tolist()
	corners0 = corners[0]          # (4,3) world
	corners_local = corners0 - b_pos0  # (4,3) local coords

	sv_xml_string = f"""
	<body name="{name}" pos="{b_pos0[0]} {b_pos0[1]} {b_pos0[2]}">
		{joint_xml}

		<!-- boxes centered at body origin -->
		<geom name="sv_box1" type="box" pos="0 0 0"
			size="{b1_size0[0]/2} {b1_size0[1]/2} {b1_size0[2]/2}"
			rgba="{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}"/>

		<geom name="sv_box2" type="box" pos="0 0 0"
			size="{b2_size0[0]/2} {b2_size0[1]/2} {b2_size0[2]/2}"
			rgba="{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}"/>

		<!-- cylinders at local corner offsets -->
		<geom name="sv_cyl1" type="cylinder"
			pos="{corners_local[0,0]} {corners_local[0,1]} {corners_local[0,2]}"
			size="{R_cyl} {object_dims[2]/2}"
			rgba="{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}"/>

		<geom name="sv_cyl2" type="cylinder"
			pos="{corners_local[1,0]} {corners_local[1,1]} {corners_local[1,2]}"
			size="{R_cyl} {object_dims[2]/2}"
			rgba="{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}"/>

		<geom name="sv_cyl3" type="cylinder"
			pos="{corners_local[2,0]} {corners_local[2,1]} {corners_local[2,2]}"
			size="{R_cyl} {object_dims[2]/2}"
			rgba="{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}"/>

		<geom name="sv_cyl4" type="cylinder"
			pos="{corners_local[3,0]} {corners_local[3,1]} {corners_local[3,2]}"
			size="{R_cyl} {object_dims[2]/2}"
			rgba="{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}"/>
	</body>
	"""
	return sv_xml_string, b_pos0

def move_swept_volume(model, data, object_configs):
	svid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "swept_volume_free")
	sv_adr = model.jnt_qposadr[svid]
	sv_vadr = model.jnt_dofadr[svid]

	object_configs = np.asarray(object_configs, dtype=np.float64)
	if object_configs.ndim == 2:
		object_configs = object_configs[None, :, :]

	x_lower = object_configs[:, 0, 0]
	x_upper = object_configs[:, 0, 1]
	y_lower = object_configs[:, 1, 0]
	y_upper = object_configs[:, 1, 1]
	z = object_configs[:, 2, 0] 

	cx = 0.5 * (x_upper + x_lower)
	cy = 0.5 * (y_upper + y_lower)
	new_pos = [cx[0], cy[0], z[0]]
	new_quat = [1, 0, 0, 0]

	data.qpos[sv_adr: sv_adr + 7] = [new_pos[0], new_pos[1], new_pos[2], new_quat[0], new_quat[1], new_quat[2], new_quat[3]]
	data.qvel[sv_vadr: sv_vadr + 6] = 0

	mujoco.mj_forward(model, data)

def quat_xyzw_to_wxyz(q):
    x, y, z, w = q
    return [w, x, y, z]

def fmt(v):
    # nice formatting for XML
    return " ".join(f"{x:.6g}" for x in v)

def yaml_to_xml(yaml_path="configs/scenes/box/scene_box.yaml", parent_body_name="scene_box", skip_ids={"Can1"}):
	with open(yaml_path, "r") as f:
		data = yaml.safe_load(f)

	objs = data["world"]["collision_objects"]
	lines = []
	lines.append(f'<body name="{parent_body_name}" pos="0 0 0">')

	for obj in objs:
		obj_id = obj.get("id", "")
		if obj_id in skip_ids:
			continue
		prim = obj["primitives"][0]
		pose = obj["primitive_poses"][0]

		pos = pose["position"]
		quat_xyzw = pose["orientation"]
		quat_wxyz = quat_xyzw_to_wxyz(quat_xyzw)

		prim_type = prim["type"].lower()
		dims = prim["dimensions"]

		if prim_type == "box":
			# dims = [x, y, z]  -> size = half-dims
			size = [dims[0] / 2.0, dims[1] / 2.0, dims[2] / 2.0]
			mj_type = "box"
			mj_size = size

		elif prim_type == "cylinder":
			# MoveIt cylinder dims = [height, radius]
			height, radius = dims[0], dims[1]
			mj_type = "cylinder"
			mj_size = [radius, height / 2.0]

		else:
			raise ValueError(f"Unsupported primitive type: {prim_type} for id={obj_id}")

		lines.append(
            f'  <geom name="{obj_id}" type="{mj_type}" '
            f'pos="{fmt(pos)}" quat="{fmt(quat_wxyz)}" '
            f'size="{fmt(mj_size)}" '
            f'contype="1" conaffinity="1" rgba="0.6 0.6 0.6 1"/>'
        )

	lines.append("</body>")
	return "\n".join(lines)


def get_robot_collision_geom_ids(model, robot_root_body="link0"):
	root_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, robot_root_body)
	
	in_robot = [False] * model.nbody
	in_robot[root_bid] = True
	for b in range(model.nbody):
		p = model.body_parentid[b]
		if p >= 0 and in_robot[p]:
			in_robot[b] = True

	robot_geoms = set()
	for g in range(model.ngeom):
		if in_robot[model.geom_bodyid[g]] and model.geom_group[g] == 3:
			robot_geoms.add(g)

	return robot_geoms

PANDA_JOINTS = [
    "joint1","joint2","joint3","joint4","joint5","joint6","joint7",
    "finger_joint1","finger_joint2"
]

def get_joint_limits(model, joint_names=PANDA_JOINTS):
	lows, highs = [], []
	for name in joint_names:
		jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
		if jid < 0:
			raise ValueError(f"Joint '{name}' not found")
		if model.jnt_limited[jid]:
			lo, hi = model.jnt_range[jid]
		else:
			lo, hi = -np.inf, np.inf
		lows.append(lo)
		highs.append(hi)
	return np.array(lows, dtype=np.float64), np.array(highs, dtype=np.float64)

		
def get_panda_qpos_idxs(model):
	joint_names = [
		"joint1","joint2","joint3","joint4","joint5","joint6","joint7",
		"finger_joint1","finger_joint2"
	]
	qpos_idxs = []
	for name in joint_names:
		jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
		qpos_idxs.append(model.jnt_qposadr[jid])
	
	return qpos_idxs

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


def get_panda_qpos(model, data):
	joint_names = [
		"joint1","joint2","joint3","joint4","joint5","joint6","joint7",
		"finger_joint1","finger_joint2"
	]
	qpos_idxs = []
	for name in joint_names:
		jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
		qpos_idxs.append(model.jnt_qposadr[jid])

	curr_config = data.qpos[qpos_idxs].copy()
	return curr_config

def robot_in_contact(model, data, robot_geoms, print_contact=False):
	for i in range(data.ncon):
		c = data.contact[i]
		if c.geom1 in robot_geoms or c.geom2 in robot_geoms:
			g1, g2 = contact_geom_names(model, c)
			if print_contact:
				print(f"contact {i}: {g1}, {g2}")
			return True, i
	return False, None

PANDA_JOINTS_9 = [f"joint{i}" for i in range(1, 8)] + ["finger_joint1", "finger_joint2"]

def get_joint_limits_by_names(model, joint_names=PANDA_JOINTS_9):
    """
    Returns (lo, hi, qpos_adrs) arrays aligned to joint_names.
    lo/hi are finite where limited, else +/-inf.
    """
    lo = np.full(len(joint_names), -np.inf, dtype=np.float64)
    hi = np.full(len(joint_names),  np.inf, dtype=np.float64)
    qpos_adrs = np.full(len(joint_names), -1, dtype=int)

    for i, name in enumerate(joint_names):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qadr = int(model.jnt_qposadr[jid])
        qpos_adrs[i] = qadr

        if int(model.jnt_limited[jid]) == 1:
            lo[i] = float(model.jnt_range[jid, 0])
            hi[i] = float(model.jnt_range[jid, 1])

    return lo, hi, qpos_adrs

def qpos_within_limits(model, qpos, joint_names=PANDA_JOINTS_9, tol=1e-6):
    lo, hi, qadr = get_joint_limits_by_names(model, joint_names)
    for name, low, high, adr in zip(joint_names, lo, hi, qadr):
        if adr < 0:
            continue
        v = float(qpos[adr])
        if v < low - tol or v > high + tol:
            return False, name, v, low, high
    return True, None, None, None, None

def randomize_seed_within_limits(model, seed9, sigma=0.4):
    seed = np.asarray(seed9, dtype=np.float64).copy()

    # arm joints
    arm_names = [f"joint{i}" for i in range(1, 8)]
    lo, hi, _ = get_joint_limits_by_names(model, arm_names)

    noise = np.random.randn(7) * sigma
    seed[:7] = np.clip(seed[:7] + noise, lo, hi)

    # keep fingers unchanged
    # seed[7:9] left as-is
    return seed


def densify_q_traj(traj, max_step=0.02):
	traj = np.asarray(traj, dtype=np.float32)
	dense = [traj[0]]
	for a, b in zip(traj[:-1], traj[1:]):
		dist = np.linalg.norm(b-a)
		n = max(1, int(np.ceil(dist / max_step)))
		for k in range(1, n + 1):
			t = k / n
			dense.append((1 - t) * a + t * b)
	return np.asarray(dense)

def contact_geom_names(model, contact):
    g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
    g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
    return g1, g2

def set_panda_qpos_idxs(model, data, panda_qpos_idxs, q):
	data.qpos[panda_qpos_idxs] = q
	mujoco.mj_forward(model, data)

def set_panda_qpos(model, data, q):
	joint_names = [
		"joint1","joint2","joint3","joint4","joint5","joint6","joint7",
		"finger_joint1","finger_joint2"
	]
	#qpos_idxs = []
	#for name in joint_names:
	#	jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
	#	qpos_idxs.append(model.jnt_qposadr[jid])
	#data.qpos[qpos_idxs] = q
	robot_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint1")
	qadr = model.jnt_qposadr[robot_jid]
	data.qpos[qadr:qadr+9] = q

	mujoco.mj_forward(model, data)

def mj_fk(model, data, q):
	left_finger_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
	right_finger_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")

	qpos_saved = data.qpos.copy()

	set_panda_qpos(model, data, q)

	left_finger_pos = data.xpos[left_finger_id].copy()
	left_finger_quat = data.xquat[left_finger_id].copy()

	right_finger_pos = data.xpos[right_finger_id].copy()
	right_finger_quat = data.xquat[right_finger_id].copy()

	# restore state
	data.qpos[:] = qpos_saved
	mujoco.mj_forward(model, data)

	return ((left_finger_pos, left_finger_quat), (right_finger_pos, right_finger_quat))

def mj_get_geom_names(model, data):
	print("ngeom =", model.ngeom)
	for i in range(model.ngeom):
		print(i, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i))


def mj_ik_multilink_until_tol(
    model,
    data,
    link_targets,
    seed,
    robot_geoms,
    obj_geom_list,
    single=False,
    dt=0.05,
    max_iters=200,
    pos_tol=0.01,          # meters
    ang_tol=0.02,          # radians
    ignore_col=False,
    require_ori_if_given=True,   # only check ang for targets that provide quat
    early_stop_patience=0,       # 0 disables; >0 allows a few non-improving iters
):
    configuration = Configuration(model)

    ik_seed = configuration.q.copy()
    ik_seed[:9] = seed
    configuration.update(ik_seed)

	# Hard reset dynamics + recompute kinematics
    data.qvel[:] = 0.0
    if hasattr(data, "qacc"):
        data.qacc[:] = 0.0
    mujoco.mj_forward(model, data)

    tasks = []
    gain = 1.0 - 0.01 ** (1.0 / max_iters)  # ok even if you stop early

    # Build tasks
    for link_target in link_targets:
        target_pos = np.asarray(link_target["pos"], dtype=np.float64)
        target_quat = link_target.get("quat", None)
        target_name = link_target["name"]

        if target_quat is not None:
            rotation = SO3(wxyz=np.asarray(target_quat, dtype=np.float64))
            target = SE3.from_translation(target_pos) @ SE3.from_rotation(rotation)
            ori_cost = 1.0
        else:
            target = SE3.from_translation(target_pos)
            ori_cost = 0.0

        task = FrameTask(
            frame_name=target_name,
            frame_type="body",
            position_cost=1.0,
            orientation_cost=ori_cost,
            gain=gain if single else None  # mink ignores None; or just always pass gain
        )
        # If FrameTask doesn't accept gain=None, just always pass gain. Mink will be fine.
        if not single:
            # rebuild without gain if your version complains
            task = FrameTask(
                frame_name=target_name,
                frame_type="body",
                position_cost=1.0,
                orientation_cost=ori_cost
            )

        task.set_target(target)
        tasks.append(task)
	
    #posture_task = PostureTask(model, cost=0.005)
    #posture_task.set_target(ik_seed)
    #tasks.append(posture_task)


    finger_freezing_task = DofFreezingTask(model=model, dof_indices=[7, 8])

    hand = ["panda_hand_col", "panda_leftfinger_colmesh", "panda_rightfinger_colmesh"]
    pads = ["panda_leftpad_1","panda_leftpad_2","panda_leftpad_3","panda_leftpad_4","panda_leftpad_5",
            "panda_rightpad_1","panda_rightpad_2","panda_rightpad_3","panda_rightpad_4","panda_rightpad_5"]
    arm  = ["panda_link4_col","panda_link5_col0","panda_link5_col1","panda_link5_col2",
            "panda_link6_col","panda_link7_col"]

    collision_limit = CollisionAvoidanceLimit(
        model,
        geom_pairs=[(hand + pads, obj_geom_list), (hand + pads, arm)],
        minimum_distance_from_collisions=0.01
    )

    if ignore_col:
        #limits = [ConfigurationLimit(model)]
        limits = []
    else:
        limits = [ConfigurationLimit(model), collision_limit]

    looped_wp = []

    # For early-stop patience (optional)
    best_score = np.inf
    non_improve = 0

    for it in range(max_iters):
        # --- solve one IK step ---
        try:
            vel = solve_ik(
                configuration=configuration,
                tasks=tasks,
                constraints=[finger_freezing_task],
                limits=limits,
                dt=dt,
                solver="daqp"
            )
        except NoSolutionFound:
            print("IK failed")
            return None

        configuration.integrate_inplace(vel, dt)
        set_panda_qpos(model, data, configuration.q[:9])

        # --- collision check ---
        if not ignore_col:
            in_contact, _ = robot_in_contact(model, data, robot_geoms)
            if in_contact:
                return None

        # --- track trajectory ---
        if single:
            looped_wp.append(configuration.q[:9].copy())

        # --- compute errors across targets ---
        # Assume you have a helper: pose_error(model, data, body_name, target_pos, target_quat_or_None)
        # that returns (pos_err, ang_err)
        max_pos = 0.0
        max_ang = 0.0

        for link_target in link_targets:
            body_name = link_target["name"]
            target_pos = link_target["pos"]
            target_quat = link_target.get("quat", None)

            pos_err_i, ang_err_i, _, _ = ee_pose_error(model, data, body_name, target_pos, target_quat)

            max_pos = max(max_pos, float(pos_err_i))

            # Only enforce ang tol if quat is provided (recommended)
            if require_ori_if_given and target_quat is not None:
                max_ang = max(max_ang, float(ang_err_i))
            else:
                # If quat absent, ignore angular error
                pass

        # --- success condition ---
        pos_ok = (max_pos <= pos_tol)
        ang_ok = (max_ang <= ang_tol) if require_ori_if_given else True

        if pos_ok and ang_ok:
            return looped_wp if single else configuration.q[:9].copy()

        # --- optional “no progress” bailout ---
        if early_stop_patience > 0:
            # A simple scalar score: prioritize position, then angle
            score = max_pos + max_ang  # tweak weighting if you want
            if score + 1e-12 < best_score:
                best_score = score
                non_improve = 0
            else:
                non_improve += 1
                if non_improve >= early_stop_patience:
                    return None

    # max_iters elapsed
    #print(f"pos err: {pos_err_i}")
    #print(f"ang err: {ang_err_i}")
    return None

def mj_ik_multilink(model, data, link_targets, seed, robot_geoms, obj_geom_list, single=False, dt=0.05, iter_count=15, ignore_col=False):

	configuration = Configuration(model)
	#mj_get_geom_names(model, data)
	#current_configuration = configuration.q
	ik_seed = configuration.q.copy()
	ik_seed[:9] = seed
	#configuration.update_from_keyframe("home2")
	configuration.update(ik_seed)
	
	#print(f"q property: {configuration.q}")
	#print(f"nv property: {configuration.nv}")
	#print(f"nq property: {configuration.nq}")

	tasks = []
	gain = 1.0 - 0.01 ** (1.0/iter_count)

	for link_target in link_targets:
		target_pos = link_target['pos']
		#target_quat = link_target['quat']
		target_quat = link_target.get('quat', None)
		target_name = link_target['name']

		if target_quat is not None:
			rotation = SO3(wxyz=target_quat)
			target = SE3.from_translation(target_pos) @ SE3.from_rotation(rotation)
			ori_cost = 1.0
		else:
			target = SE3.from_translation(target_pos)
			ori_cost = 0.0

		if (single):
		# Build mink task
			task = FrameTask(
				frame_name=target_name,
				frame_type="body",
				position_cost=1.0,
				orientation_cost=ori_cost,
				gain=gain
			)
		else:
			task = FrameTask(
				frame_name=target_name,
				frame_type="body",
				position_cost=1.0,
				orientation_cost=ori_cost
			)

		task.set_target(target)
		tasks.append(task)

	finger_freezing_task = DofFreezingTask(
		model=model,
		dof_indices=[7, 8]
	)

	#damping_task = DampingTask(model, cost=0.5)
	#posture_task = PostureTask(model, cost=0.1)
	#posture_task.set_target_from_configuration(configuration)

	hand = ["panda_hand_col", "panda_leftfinger_colmesh", "panda_rightfinger_colmesh"]
	pads = ["panda_leftpad_1","panda_leftpad_2","panda_leftpad_3","panda_leftpad_4","panda_leftpad_5",
			"panda_rightpad_1","panda_rightpad_2","panda_rightpad_3","panda_rightpad_4","panda_rightpad_5"]

	arm = ["panda_link4_col","panda_link5_col0","panda_link5_col1","panda_link5_col2","panda_link6_col","panda_link7_col"]
	#sv  = ["sv_box1","sv_box2","sv_cyl1","sv_cyl2","sv_cyl3","sv_cyl4"]

	collision_limit = CollisionAvoidanceLimit(
		model,
		geom_pairs=[(hand + pads, obj_geom_list), (hand + pads, arm)],
		minimum_distance_from_collisions=0.01
	)

	if ignore_col:
		limits = [ConfigurationLimit(model)]
	else:
		limits = [ConfigurationLimit(model), collision_limit]

	looped_wp = []

	#dt = 0.05
	for it in range(iter_count):
		try:
			vel = solve_ik(
				configuration=configuration, 
				tasks=tasks,
				constraints=[finger_freezing_task],
				limits=limits, 
				dt=dt, 
				solver="daqp"
			)
		except NoSolutionFound:
			#print("IK failed")
			return None
		
		configuration.integrate_inplace(vel, dt)
		set_panda_qpos(model, data, configuration.q[:9])

		if ignore_col:
			in_contact = False
		else:
			in_contact, _ = robot_in_contact(model, data, robot_geoms)
		
		if in_contact:
			#print("Collision: Suffix = None")
			return None
		
		looped_wp.append(configuration.q[0:9].copy())
	#print(configuration.q[0:9])

	if single is True:
		#print(f"looped wp: {len(looped_wp)}")
		return looped_wp
	else:
		#print(f"conf: {len(configuration.q[0:9])}")
		return configuration.q[0:9].copy()

class PandaMinkIK:
	def __init__(self, model, data, robot_geoms, obj_geom_list, dt=0.05, iter_count=20, min_dist=0.02):
		self.model = model
		self.data = data
		self.robot_geoms = robot_geoms
		self.obj_geom_list = obj_geom_list
		self.dt = dt
		self.iter_count = iter_count

		self.configuration = Configuration(model)

		self.finger_freezing_task = DofFreezingTask(model=model, dof_indices=[7, 8])

		hand = ["panda_hand_col", "panda_leftfinger_colmesh", "panda_rightfinger_colmesh"]
		pads = ["panda_leftpad_1","panda_leftpad_2","panda_leftpad_3","panda_leftpad_4","panda_leftpad_5",
				"panda_rightpad_1","panda_rightpad_2","panda_rightpad_3","panda_rightpad_4","panda_rightpad_5"]
		arm = ["panda_link4_col","panda_link5_col0","panda_link5_col1","panda_link5_col2","panda_link6_col","panda_link7_col"]

		self.collision_limit = CollisionAvoidanceLimit(
			model,
			geom_pairs=[(hand + pads, obj_geom_list), (hand + pads, arm)],
			minimum_distance_from_collisions=min_dist
		)
		self.config_limit = ConfigurationLimit(model)

		self.tasks_by_name = {}
		gain = 1.0 - 0.01 ** (1.0 / iter_count)
		for name in ["left_finger", "right_finger"]:  # add others if needed
			t = FrameTask(frame_name=name, frame_type="body",
							position_cost=1.0, orientation_cost=1.0, gain=gain)
			self.tasks_by_name[name] = t

	def solve(self, link_targets, seed, ignore_col=False, single=False):
		
		q = self.configuration.q.copy()
		q[:9] = seed
		self.configuration.update(q)

		tasks = []
		for lt in link_targets:
			rotation = SO3(wxyz=lt["quat"])
			target = SE3.from_translation(lt["pos"]) @ SE3.from_rotation(rotation)
			task = self.tasks_by_name[lt["name"]]
			task.set_target(target)
			tasks.append(task)
		
		#limits = [self.config_limit] if ignore_col else [self.config_limit, self.collision_limit]
		limits = [self.config_limit] if ignore_col else [self.config_limit, self.collision_limit]
		

		looped_wp = []
		for _ in range(self.iter_count):
			try:
				vel = solve_ik(self.configuration, tasks,
								constraints=[self.finger_freezing_task],
								limits=limits,
								dt=self.dt,
								solver="daqp")
			except NoSolutionFound:
				#print("IK failed")
				return None

			self.configuration.integrate_inplace(vel, self.dt)
		
			if single:
				looped_wp.append(self.configuration.q[:9].copy())

		return looped_wp if single else self.configuration.q[:9].copy()
	

class MujocoViewer:
	def __init__(self, model, data, robot_geoms, fps=60, show_left_ui=False, show_right_ui=False):
		self.model = model
		self.data = data
		self.fps = fps
		self.show_left_ui = show_left_ui
		self.show_right_ui = show_right_ui
		self.viewer = None
		self.robot_geoms = robot_geoms
		#self.robot_qposadr = robot_qposadr
	
	def open(self):
		if self.viewer is None:
			self.viewer = mujoco.viewer.launch_passive(
				self.model, self.data,
				show_left_ui=self.show_left_ui,
				show_right_ui=self.show_right_ui
			)
		return self.viewer

	def is_open(self):
		return self.viewer is not None and self.viewer.is_running()

	def close(self):
		if self.viewer is not None:
			self.viewer.close()
			self.viewer = None

	def render_state(self):
		if not self.is_open():
			return False
		self.viewer.sync()
		return True

	def play_qpos_traj(self, qpos_traj, dt=None, wait_for_enter=True, loop=False):

		self.open()
		#print("s0")
		set_panda_qpos(self.model, self.data, qpos_traj[0])
		#print("s1")
		self.viewer.sync()

		if wait_for_enter:
			input("Begin visualization?")
			#print("Starting visualization...")
			#print(qpos_traj)

		qpos_traj = np.asarray(qpos_traj)
		step_dt = (1.0 / self.fps) if dt is None else float(dt)

		# Basic sanity
		#if qpos_traj.ndim != 2 or qpos_traj.shape[1] != self.model.nq:
		#	raise ValueError(f"qpos_traj must be (T, {self.model.nq}), got {qpos_traj.shape}")

		while True:
			for qpos in qpos_traj:
				if not self.is_open():
					print("Viewer closed.")
					return
				#self.data.qpos[:] = qpos
				#mujoco.mj_forward(self.model, self.data)
				#print(qpos)
				set_panda_qpos(self.model, self.data, qpos)
				in_contact, _ = robot_in_contact(self.model, self.data, self.robot_geoms)

				self.viewer.sync()
				time.sleep(step_dt)

			if wait_for_enter:
				input("Visualization complete. Proceed?")
				#print("Visualization Complete")
			if not loop:
				return
