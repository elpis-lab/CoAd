import multiprocessing as mp, os
from concurrent.futures import ProcessPoolExecutor, as_completed

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

    import genesis as gs #type: ignore

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