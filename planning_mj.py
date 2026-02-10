import numpy as np
import torch #type: ignore
import time
import mujoco #type: ignore

from utils.mujoco_utils import robot_in_contact, set_panda_qpos, get_panda_qpos, get_joint_limits, get_panda_qpos_idxs

from ompl import base as ob #type: ignore
from ompl import geometric as og #type: ignore
from ompl import util as ou #type: ignore
from ompl import base as ob #type: ignore

from ompl import util as ou #type: ignore


class omplPlanner():

    def __init__(self, model, data, robot_geoms, log=True):
        self.model = model
        self.data = data
        #self.data_plan = mujoco.MjData(model)

        self.robot_geoms = robot_geoms
        #self.panda_qpos_idxs = get_panda_qpos_idxs(model)
        if not log:
            ou.setLogLevel(ou.LOG_ERROR)

    def _omplValidityChecker(self, state):
        q = np.array([state[i] for i in range(9)], dtype=np.float64)
        set_panda_qpos(self.model, self.data, q)
        in_contact, _ = robot_in_contact(self.model, self.data, self.robot_geoms)

        return (not in_contact)

    def omplPlan(
        self,
        qpos_goal,
        qpos_start=None,
        timeout=10.0,
        smooth_path=True,
        num_waypoints=1000,
        ignore_collision=False,
        ignore_joint_limit=False,
        planner="RRTConnect",
        log=False
    ):
        supported_planners = [
            "PRM",
            "RRT",
            "RRTConnect",
            "RRTstar",
            "EST",
            "FMT",
            "BITstar",
            "ABITstar",
        ]
        if planner not in supported_planners:
            raise ValueError(
            f"Unsupported planner '{planner}'. "
            f"Supported planners are: {supported_planners}"
        )
        if qpos_start is None:
            qpos_start = get_panda_qpos(self.model, self.data)
        
        qpos_start = np.asarray(qpos_start, dtype=np.float64)
        qpos_goal  = np.asarray(qpos_goal,  dtype=np.float64)

        if ignore_joint_limit:
            q_limit_lower = np.full_like(qpos_start, -1e6)
            q_limit_upper = np.full_like(qpos_start, 1e6)
        else:
            q_limit_lower, q_limit_upper = get_joint_limits(self.model) 

        if (qpos_start < q_limit_lower).any() or (qpos_start > q_limit_upper).any():
            if log:
                print("qpos_start out of limits. adjusting...")
            q_limit_lower = np.minimum(q_limit_lower, qpos_start)
            q_limit_upper = np.maximum(q_limit_upper, qpos_start)

        if (qpos_goal < q_limit_lower).any() or (qpos_goal > q_limit_upper).any():
            q_limit_lower = np.minimum(q_limit_lower, qpos_goal)
            q_limit_upper = np.maximum(q_limit_upper, qpos_goal)
            if log:
                print("qpos_goal out of limits. adjusting...")

        # Setting up OMPL StateSpace
        n_qs = len(qpos_start)
        space = ob.RealVectorStateSpace(n_qs)
        bounds = ob.RealVectorBounds(n_qs)

        for i_q in range(n_qs):
            bounds.setLow(int(i_q), float(q_limit_lower[i_q]))
            bounds.setHigh(int(i_q), float(q_limit_upper[i_q]))
        space.setBounds(bounds)
        ss = og.SimpleSetup(space)
        si = ss.getSpaceInformation()
        si.setStateValidityCheckingResolution(0.0035)

        if ignore_collision:
            ss.setStateValidityChecker(ob.StateValidityCheckerFn(lambda state: True))
        else:
            ss.setStateValidityChecker(ob.StateValidityCheckerFn(self._omplValidityChecker))

        # Setting up OMPL planner
        ss.setPlanner(getattr(og, planner)(ss.getSpaceInformation()))

        # Setup start and goal states
        state_start = ob.State(space)
        state_goal = ob.State(space)
        for i_q in range(n_qs):
            state_start[i_q] = float(qpos_start[i_q])
            state_goal[i_q] = float(qpos_goal[i_q])
        ss.setStartAndGoalStates(state_start, state_goal)

        # Solve
        t0 = time.perf_counter()
        solved = ss.solve(float(timeout))
        waypoints = []
        if solved:
            if log:
                print("Path solution found.")
            path = ss.getSolutionPath()

            if smooth_path:
                ps = og.PathSimplifier(ss.getSpaceInformation())
                try:
                    ps.ropeShortcutPath(path)
                except:
                    ps.shortcutPath(path)
                ps.smoothBSpline(path)
            
            if num_waypoints is not None:
                path.interpolate(int(num_waypoints))
            states = path.getStates()
            waypoints = [np.array([s[i] for i in range(n_qs)], dtype=np.float64) for s in states]
            
        else:
            if log:
                print("Path planning failed.")

        t1 = time.perf_counter()
        self.total_time = round(t1 - t0, 5)
        self.plan_time = ss.getLastPlanComputationTime()
        return waypoints
