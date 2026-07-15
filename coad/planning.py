import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import numpy as np

import ompl.base as ob
import ompl.geometric as og
import ompl.util as ou

import mujoco
from coad.robot import MujocoRobot
from scipy.spatial import cKDTree

import vamp
from scipy.spatial.transform import Rotation

class VAMPPlanner:
    def __init__(
        self,
        robot: MujocoRobot,
        env,
        data=None,
        robot_name="panda",
        sampler_name="halton",
        log=False,
    ):
        self.robot = robot
        self.model = robot.model
        self.data = data if data is not None else mujoco.MjData(self.model)

        if data is None:
            self.data.qpos[:] = robot.data.qpos[:]

        self.n_dof = robot.n_joints
        self.robot_name = robot_name

        (
            self.vamp_robot,
            self.planner_fn,
            self.plan_settings,
            self.simplify_settings,
        ) = vamp.configure_robot_and_planner_with_kwargs(
            robot_name,
            "rrtc",
        )

        self.sampler = getattr(
            self.vamp_robot,
            sampler_name,
        )()

        # Robot base transforms
        self.base_pos_world = np.asarray(
            env.env_details["robot_pos"],
            dtype=float,
        )

        self.base_quat_world = np.asarray(
            env.env_details["robot_quat"],
            dtype=float,
        )
        quat_wxyz = self.base_quat_world
        quat_xyzw = np.array(
            [
                quat_wxyz[1],
                quat_wxyz[2],
                quat_wxyz[3],
                quat_wxyz[0],
            ],
            dtype=float,
        )
        self.base_rot_world = Rotation.from_quat(quat_xyzw).as_matrix()        

        self.environment = self.build_environment()
        # raise NotImplementedError("VAMP not yet implemented.")

    def world_pose_to_base(self, world_pos, world_rot):
        world_pos = np.asarray(world_pos, dtype=float)
        world_rot = np.asarray(world_rot, dtype=float).reshape(3, 3)

        # R_WB maps base-frame vectors into world frame.
        # Therefore, R_BW = R_WB.T.
        base_rot_inv = self.base_rot_world.T

        base_pos = base_rot_inv @ (
            world_pos - self.base_pos_world
        )

        base_rot = base_rot_inv @ world_rot

        return base_pos, base_rot

    
    def build_environment(self):
        vamp_env = vamp.Environment()

        mujoco.mj_forward(self.model, self.data)

        for geom_id in range(self.model.ngeom):
            if geom_id in self.robot.robot_geoms:
                continue

            geom_type = self.model.geom_type[geom_id]

            world_pos = self.data.geom_xpos[geom_id].copy()
            world_rot = (
                self.data.geom_xmat[geom_id]
                .reshape(3, 3)
                .copy()
            )
            size = self.model.geom_size[geom_id].copy()

            if geom_type == mujoco.mjtGeom.mjGEOM_PLANE:
                floor_half_height = 0.05

                # Create the finite floor cuboid in world coordinates.
                floor_world_pos = world_pos.copy()
                floor_normal_world = world_rot[:, 2]

                # Place the cuboid beneath the original plane.
                floor_world_pos -= (
                    floor_half_height * floor_normal_world
                )

                base_pos, base_rot = self.world_pose_to_base(
                    floor_world_pos,
                    world_rot,
                )

                base_euler = Rotation.from_matrix(
                    base_rot
                ).as_euler("xyz")

                vamp_env.add_cuboid(
                    vamp.Cuboid(
                        base_pos.tolist(),
                        base_euler.tolist(),
                        [5.0, 5.0, floor_half_height],
                    )
                )

            elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
                base_pos, base_rot = self.world_pose_to_base(
                    world_pos,
                    world_rot,
                )

                base_euler = Rotation.from_matrix(
                    base_rot
                ).as_euler("xyz")

                vamp_env.add_cuboid(
                    vamp.Cuboid(
                        base_pos.tolist(),
                        base_euler.tolist(),
                        size[:3].tolist(),
                    )
                )
            else:
                continue
                
            # print(
            #     geom_id,
            #     self.model.geom_type[geom_id],
            #     self.model.geom_size[geom_id],
            #     mujoco.mj_id2name(
            #         self.model,
            #         mujoco.mjtObj.mjOBJ_GEOM,
            #         geom_id,
            #     ),
            # )

        return vamp_env

    def plan(
        self,
        start,
        goal,
        smooth_path=True,
        num_waypoints=200,
        benchmark=False,
        log=False,
    ):
        
        # Setup environment for VAMP again (moved goal object)
        self.environment = self.build_environment()

        start = np.asarray(start, dtype=float).tolist()
        goal = np.asarray(goal, dtype=float).tolist()

        start_valid = self.vamp_robot.validate(start, self.environment)
        goal_valid = self.vamp_robot.validate(goal, self.environment)

        # print(f"VAMP start valid: {start_valid}", flush=True)
        # print(f"VAMP goal valid: {goal_valid}", flush=True)

        if len(start) != self.vamp_robot.dimension():
            raise ValueError(
                f"Expected {self.vamp_robot.dimension()} joints, "
                f"got {len(start)}"
            )

        if not start_valid:
            empty = np.empty((0, self.n_dof), dtype=np.float32)
            return empty, 0.0

        if not goal_valid:
            empty = np.empty((0, self.n_dof), dtype=np.float32)
            return empty, 0.0

        t0 = time.perf_counter()

        result = self.planner_fn(
            start,
            goal,
            self.environment,
            self.plan_settings,
            self.sampler,
        )

        # planning_time = time.perf_counter() - t0
        planning_time = result.nanoseconds * 1e-9

        if not result.solved:
            empty = np.empty((0, self.n_dof), dtype=np.float32)
            return empty, planning_time if benchmark else empty

        if result is None or result.path is None:
            return (empty, planning_time) if benchmark else empty
        
        path = result.path

        if smooth_path:
            simplified = self.vamp_robot.simplify(
                path,
                self.environment,
                self.simplify_settings,
                self.sampler,
            )
            path = simplified.path

        if num_waypoints is not None:
            path.interpolate_to_n_states(int(num_waypoints))

        waypoints = np.asarray(
            path.numpy(),
            dtype=np.float32,
        )

        if waypoints.ndim != 2 or waypoints.shape[1] != self.n_dof:
            raise RuntimeError(
                f"Unexpected VAMP path shape: {waypoints.shape}"
            )

        total_time = time.perf_counter() - t0

        if log:
            print(f"VAMP planning time: {planning_time:.6f} s")
            print(f"VAMP total time: {total_time:.6f} s")
            print(f"VAMP path shape: {waypoints.shape}")

        return waypoints, planning_time


class OMPLPlanner:
    """
    OMPL Planner class for planning paths.
    Using Mujoco Robot with itsmodel and data for planning.
    """

    def __init__(
        self,
        robot: MujocoRobot,
        data=None,
        planner="RRTConnect",
        rrtc_range=None,
        log=False,
    ):
        """Initialize Planner"""
        # Mujoco Robot with its model and data
        self.robot = robot
        self.model = robot.model
        # create a new data for this planning instead of
        # using the robot instance's data
        if data is None:
            self.data = mujoco.MjData(self.model)
            self.data.qpos[:] = robot.data.qpos[:]
        else:
            self.data = data

        self.robot_geoms = self.robot.robot_geoms
        self.n_dof = self.robot.n_joints
        self.joint_limits = self.robot.joint_limits
        self.range = rrtc_range

        # Set up OMPL planner
        self.planner_name = planner
        self.ss, self.si = self.set_up_ompl()
        self.si.setStateValidityCheckingResolution(0.005)

        self.planner = self.ss.getPlanner()
        if not log:
            ou.setLogLevel(ou.LOG_ERROR)

        self.query_states = []
        self.goal_vertices = []
        

    def set_up_ompl(self):
        """Setup OMPL planner"""
        # Define space
        space = ob.RealVectorStateSpace(self.n_dof)
        # Set bounds
        bounds = ob.RealVectorBounds(self.n_dof)
        for i in range(self.n_dof):
            bounds.setLow(i, self.joint_limits[0][i])
            bounds.setHigh(i, self.joint_limits[1][i])
        space.setBounds(bounds)

        # Simple setup and Space Information
        ss = og.SimpleSetup(space)
        si = ss.getSpaceInformation()
        si.setStateValidityCheckingResolution(0.01)

        # State validity checker
        ss.setStateValidityChecker(
            ob.StateValidityCheckerFn(self.validity_checker)
        )

        # Optimization objective (default path length)
        ss.setOptimizationObjective(ob.PathLengthOptimizationObjective(si))

        planner = getattr(og, self.planner_name)(si)

        # ---- Adaptive range ----
        if self.planner_name == "RRTConnect" and self.range is not None:
            extent = si.getMaximumExtent()
            planner.setRange(self.range * extent)

        # Setting up OMPL planner
        # ss.setPlanner(getattr(og, self.planner_name)(si))
        ss.setPlanner(planner)
        return ss, si

    def plan(
        self,
        start,
        goal,
        timeout=10.0,
        smooth_path=True,
        num_waypoints=200,
        benchmark=False,
        log=False,
    ):

        # Set up start and goal states
        start_state = ob.State(self.si.getStateSpace())
        goal_state = ob.State(self.si.getStateSpace())
        for i_q in range(self.n_dof):
            start_state[i_q] = float(start[i_q])
            goal_state[i_q] = float(goal[i_q])
        self.ss.setStartAndGoalStates(start_state, goal_state)

        # Solve
        waypoints = np.empty((0, self.n_dof), dtype=np.float32)
        t0 = time.perf_counter()
        status = self.ss.solve(float(timeout))

        if status.asString() == "Exact solution":
            if log:
                print("Path solution found.")
            path = self.ss.getSolutionPath()

            # smooth path
            if smooth_path:
                ps = og.PathSimplifier(self.si)
                try:
                    ps.ropeShortcutPath(path)
                except:
                    ps.shortcutPath(path)
                ps.smoothBSpline(path)
            # interpolate path
            if num_waypoints is not None:
                path.interpolate(int(num_waypoints))
            # extract waypoints
            states = path.getStates()
            waypoints = np.array(
                [[s[i] for i in range(self.n_dof)] for s in states],
                dtype=np.float32,  # save memory
            )
        else:
            if log:
                print("Path planning failed.")

        self.ss.clear()
        if benchmark:
            t1 = time.perf_counter()
            # total time: plan + simplify + interpolation
            total_time = round(t1 - t0, 5)
            # planning time: time spent in planning
            planning_time = self.ss.getLastPlanComputationTime()
            return waypoints, total_time, planning_time
        return waypoints
    

    def construct_roadmap(
        self,
        start,
        timeout=30.0
    ):
        """
        Sample uniformly in the state space and build a roadmap.
        The roadmap is kept for subsequent planning calls.
        """
        assert (
            "PRM" in self.planner_name
        ), f"Planner {self.planner_name} is not supported."

        self.start_np = np.asarray(start, dtype=float).copy()
        
        self.start_state = ob.State(self.si.getStateSpace())
        for i in range(self.n_dof):
            self.start_state[i] = float(start[i])
        
        # Start growing the roadmap
        self.ss.setStartAndGoalStates(self.start_state, self.start_state)
        self.ss.setup()
        ter = ob.timedPlannerTerminationCondition(float(timeout))
        self.planner.constructRoadmap(ter)

        # Add persistent start milestone AFTER roadmap construction.
        n0 = self.planner.milestoneCount()
        e0 = self.planner.edgeCount()

        self.v_start = self.planner.addMilestone(self.start_state())

        # print("start milestone added")
        # print("milestones:", n0, "->", self.planner.milestoneCount())
        # print("edges:", e0, "->", self.planner.edgeCount())
        print(f"Roadmap edge count after construction: {self.planner.edgeCount()}")

        # print("Building python graph...")

        self.build_graph_from_planner_data()

        # print("Done building Python graph.")


    def validate_path(self, path):
        states = path.getStates()

        for s in states:
            if not self.validity_checker(s):
                return False

        for s1, s2 in zip(states[:-1], states[1:]):
            if not self.si.checkMotion(s1, s2):
                return False

        return True

    def validity_checker(self, state):
        """Check if the state is valid"""
        # set robot joint positions
        q = np.array([state[i] for i in range(self.n_dof)], dtype=float)
        self.robot.set_joint_qpos(q)

        # Check for collisions
        in_contact = self.robot.in_contact()
        return not in_contact

    def build_graph_from_planner_data(self):
        pd = ob.PlannerData(self.si)
        self.planner.getPlannerData(pd)
        pd.computeEdgeWeights()

        n = pd.numVertices()
        vertices = np.zeros((n, self.n_dof), dtype=np.float64)
        adjacency = [[] for _ in range(n)]

        # Extract vertices
        for i in range(n):
            s = pd.getVertex(i).getState()
            vertices[i] = [s[j] for j in range(self.n_dof)]

        # Extract edges
        import ompl.util as ou

        for i in range(n):
            edge_list = ou.vectorUint()
            pd.getEdges(i, edge_list)

            for j in edge_list:
                j = int(j)
                try:
                    w = pd.getEdgeWeight(i, j).value()
                except Exception:
                    w = np.linalg.norm(vertices[i] - vertices[j])

                adjacency[i].append((j, float(w)))

        self.planner_data = pd
        self.graph_vertices = vertices
        self.graph_adj = adjacency
        self.vertex_tree = cKDTree(self.graph_vertices)

        print("Graph vertices:", n)
        print("Graph edges directed:", sum(len(a) for a in adjacency))

    def connect_temp_config(self, q, k=30):
        q = np.asarray(q, dtype=float)

        k = min(k, len(self.graph_vertices))
        dists, nbrs = self.vertex_tree.query(q, k=k)

        # cKDTree returns scalars when k == 1
        dists = np.atleast_1d(dists)
        nbrs = np.atleast_1d(nbrs)

        edges = []
        q_state = self.numpy_to_state(q)

        for dist, j in zip(dists, nbrs):
            j = int(j)
            qj_state = self.numpy_to_state(self.graph_vertices[j])

            if self.si.checkMotion(q_state(), qj_state()):
                edges.append((j, float(dist)))

        return edges
    
    def make_query_graph(self, start_q, goal_q, k=30):
        vertices = self.graph_vertices
        base_adj = self.graph_adj
        n = len(vertices)

        start_idx = n
        goal_idx = n + 1

        query_adj = [list(a) for a in base_adj]
        query_adj.append([])
        query_adj.append([])

        start_edges = self.connect_temp_config(start_q, k=k)
        goal_edges = self.connect_temp_config(goal_q, k=k)

        for j, w in start_edges:
            query_adj[start_idx].append((j, w))
            query_adj[j].append((start_idx, w))

        for j, w in goal_edges:
            query_adj[goal_idx].append((j, w))
            query_adj[j].append((goal_idx, w))

        return query_adj, start_idx, goal_idx

    def graph_query(
        self,
        start,
        goal,
        max_attempts=5,
        k=30,
        smooth_path=True,
        num_waypoints=200
    ):
        query_adj, s_idx, g_idx = self.make_query_graph(start, goal, k=k)
        blocked_edges = set()
        blocked_vertices = set()

        for _ in range(max_attempts):
            # idx_path = dijkstra(
            #     query_adj,
            #     s_idx,
            #     g_idx,
            #     blocked_edges=blocked_edges,
            #     blocked_vertices=blocked_vertices,
            # )
            idx_path = astar(
                query_adj,
                s_idx,
                g_idx,
                self.graph_vertices,
                np.asarray(start, dtype=float),
                np.asarray(goal, dtype=float),
                blocked_edges=blocked_edges,
                blocked_vertices=blocked_vertices
            )
            if idx_path is None:
                return None

            q_path = self.idx_path_to_waypoints(idx_path, start, goal)
            valid, failure = self.validate_np_path_and_bad_edge(q_path)

            if valid:
                path = self.np_path_to_path_geometric(q_path)

                if smooth_path:
                    ps = og.PathSimplifier(self.si)
                    try:
                        ps.ropeShortcutPath(path)
                    except KeyboardInterrupt:
                        raise
                    except Exception:
                        if hasattr(ps, "shortcutPath"):
                            ps.shortcutPath(path)

                    ps.smoothBSpline(path)

                if num_waypoints is not None:
                    path.interpolate(int(num_waypoints))

                # # Recheck final path after smoothing/interpolation.
                # validation = self.validate_path_with_prefix(path)
                valid = self.validate_path(path)
                # if not validation["valid"]:
                if not valid:
                    return None

                return np.array([
                    [s[i] for i in range(self.n_dof)]
                    for s in path.getStates()
                ], dtype=float)

            if failure is None:
                return None

            if failure[0] == "edge":
                _, path_i, path_j = failure
                u = idx_path[path_i]
                v = idx_path[path_j]
                blocked_edges.add((u, v))
                blocked_edges.add((v, u))

            elif failure[0] == "state":
                _, path_i = failure
                bad_v = idx_path[path_i]

                if bad_v in (s_idx, g_idx):
                    return None

                blocked_vertices.add(bad_v)

        return None

    def numpy_to_state(self, q):
        s = ob.State(self.si.getStateSpace())
        for i in range(self.n_dof):
            s[i] = float(q[i])
        return s
    
    def idx_path_to_waypoints(self, idx_path, start_q, goal_q):
        n = len(self.graph_vertices)
        out = []

        for idx in idx_path:
            if idx < n:
                out.append(self.graph_vertices[idx])
            elif idx == n:
                out.append(np.asarray(start_q, dtype=float))
            elif idx == n + 1:
                out.append(np.asarray(goal_q, dtype=float))

        return np.asarray(out)
    
    def np_path_to_path_geometric(self, q_path):
        path = og.PathGeometric(self.si)

        for q in q_path:
            s = self.numpy_to_state(q)
            path.append(s())

        return path

    def validate_np_path_and_bad_edge(self, q_path):
        if q_path is None or len(q_path) == 0:
            return False, None

        states = [self.numpy_to_state(q) for q in q_path]

        for i, s in enumerate(states):
            if not self.validity_checker(s()):
                return False, ("state", i)

        for i in range(len(states) - 1):
            if not self.si.checkMotion(states[i](), states[i + 1]()):
                return False, ("edge", i, i + 1)

        return True, None

import heapq
import math

def dijkstra(adj, start_idx, goal_idx, blocked_edges=None, blocked_vertices=None):
    if blocked_edges is None:
        blocked_edges = set()
    if blocked_vertices is None:
        blocked_vertices = set()

    n = len(adj)
    dist = np.full(n, np.inf)
    parent = np.full(n, -1, dtype=np.int64)

    dist[start_idx] = 0.0
    pq = [(0.0, start_idx)]

    while pq:
        d, u = heapq.heappop(pq)

        if u in blocked_vertices and u not in (start_idx, goal_idx):
            continue

        if d != dist[u]:
            continue
        if u == goal_idx:
            break

        for v, w in adj[u]:
            if v in blocked_vertices and v not in (start_idx, goal_idx):
                continue
            if (u, v) in blocked_edges or (v, u) in blocked_edges:
                continue

            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                parent[v] = u
                heapq.heappush(pq, (nd, v))

    if not np.isfinite(dist[goal_idx]):
        return None

    path = []
    cur = goal_idx
    while cur != -1:
        path.append(cur)
        cur = parent[cur]

    return path[::-1]

def astar(adj, start_idx, goal_idx, vertices, start_q, goal_q, blocked_edges=None, blocked_vertices=None):
    if blocked_edges is None:
        blocked_edges = set()
    if blocked_vertices is None:
        blocked_vertices = set()
    
    n_graph = len(vertices)
    n_total = len(adj)

    def q_of(idx):
        if idx < n_graph:
            return vertices[idx]
        elif idx == start_idx:
            return start_q
        elif idx == goal_idx:
            return goal_q
        raise IndexError(idx)

    def heuristic(idx):
        return np.linalg.norm(q_of(idx) - goal_q)
    
    g_score = np.full(n_total, np.inf)
    parent = np.full(n_total, -1, dtype=np.int64)

    g_score[start_idx] = 0.0
    pq = [(heuristic(start_idx), 0.0, start_idx)]

    while pq:
        f, g, u = heapq.heappop(pq)

        if g != g_score[u]:
            continue

        if u == goal_idx:
            break

        if u in blocked_vertices and u not in (start_idx, goal_idx):
            continue

        for v, w in adj[u]:
            if v in blocked_vertices and v not in (start_idx, goal_idx):
                continue
            if (u, v) in blocked_edges or (v, u) in blocked_edges:
                continue
            new_g = g + w
            if new_g < g_score[v]:
                g_score[v] = new_g
                parent[v] = u
                heapq.heappush(pq, (new_g + heuristic(v), new_g, v))
    
    if not np.isfinite(g_score[goal_idx]):
        return None
    
    path = []
    cur = goal_idx
    while cur != -1:
        path.append(cur)
        cur = parent[cur]

    return path[::-1]

def euclidean_path_length(traj):
    """Compute the length of a trajectory."""
    if traj is None:
        return 0.0
    traj = np.asarray(traj, dtype=np.float64)
    if traj.ndim != 2 or len(traj) < 2:
        return 0.0

    # Differences between consecutive states
    diffs = np.diff(traj, axis=0)
    # Euclidean norms of each segment
    segment_lengths = np.linalg.norm(diffs, axis=1)
    return np.sum(segment_lengths)


# TODO
# Write a test case for the planner
if __name__ == "__main__":
    pass
