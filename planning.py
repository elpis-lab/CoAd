
import genesis as gs
from genesis.utils.misc import tensor_to_array
import numpy as np
import torch

class omplPlanner:
    def __init__(self, robot):
         self.robot = robot

    def _omplStateToTensor(self, state):
        tensor = torch.empty(self.robot.n_qs, dtype=gs.tc_float, device=gs.device)
        for i in range(self.robot.n_qs):
            tensor[i] = state[i]
        return tensor
        
    def _omplStatesToTensorList(self, states):
        tensor_list = []
        for state in states:
            tensor_list.append(self._omplStateToTensor(state))
        return tensor_list

    def _omplValidityChecker(self, state):
        ...
        env_idx = self.env_idx
        #(env_idx)
        if env_idx is None:
            self.robot.set_qpos(self._omplStateToTensor(state), zero_velocity=True)
            collision_pairs = self.robot.detect_collision()
        else:
            self.robot.set_qpos(self._omplStateToTensor(state), envs_idx=torch.tensor(env_idx), zero_velocity=True)
            collision_pairs = self.robot.detect_collision(env_idx=env_idx[0])

        if len(collision_pairs)>0:
            return False
        else:
            return True
        
    # OMPL planning with SimpleSetup, based on RigidEntity.plan_path() 

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
        env_idx=None
    ):
        try:
            from ompl import base as ob
            from ompl import geometric as og
            from ompl import util as ou

            ou.setLogLevel(ou.LOG_ERROR)
        except:
            gs.raise_exception(
                "Failed to import OMPL. Did you install? (For installation instructions, see https://genesis-world.readthedocs.io/en/latest/user_guide/overview/installation.html#optional-motion-planning)"
            )
        
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
            gs.raise_exception(f"Planner {planner} is not supported. Supported planners: {supported_planners}.")

        #print(self.robot.n_qs)
        #print(self.robot.n_dofs)
        if self.robot.n_qs != self.robot.n_dofs:
            gs.raise_exception("Motion planning is not yet supported for rigid entities with free joints.")

        self.env_idx = env_idx

        if qpos_start is None:
            qpos_start = self.robot.get_qpos()
            qpos_start = tensor_to_array(qpos_start)
            qpos_goal = tensor_to_array(qpos_goal)

        qpos_start = np.asarray(qpos_start, dtype=np.float64)
        qpos_goal  = np.asarray(qpos_goal,  dtype=np.float64)

        if qpos_start.shape != (self.robot.n_qs,) or qpos_goal.shape != (self.robot.n_qs,):
                gs.raise_exception("Invalid shape for `qpos_start` or `qpos_goal`.")

        if ignore_joint_limit:
                q_limit_lower = np.full_like(self.robot.q_limit[0], -1e6)
                q_limit_upper = np.full_like(self.robot.q_limit[1], 1e6)
        else:
            q_limit_lower = self.robot.q_limit[0]
            q_limit_upper = self.robot.q_limit[1]

        if (qpos_start < q_limit_lower).any() or (qpos_start > q_limit_upper).any():
            #gs.logger.warning(
            #    "`qpos_start` exceeds joint limit. Relaxing joint limit to contain `qpos_start` for planning."
            #)
            q_limit_lower = np.minimum(q_limit_lower, qpos_start)
            q_limit_upper = np.maximum(q_limit_upper, qpos_start)

        if (qpos_goal < q_limit_lower).any() or (qpos_goal > q_limit_upper).any():
            gs.logger.warning(
                "`qpos_goal` exceeds joint limit. Relaxing joint limit to contain `qpos_goal` for planning."
            )
            q_limit_lower = np.minimum(q_limit_lower, qpos_goal)
            q_limit_upper = np.maximum(q_limit_upper, qpos_goal)

        # Setting up OMPL StateSpace
        space = ob.RealVectorStateSpace(self.robot.n_qs)
        bounds = ob.RealVectorBounds(self.robot.n_qs)

        for i_q in range(self.robot.n_qs):
            #bounds.setLow(i_q, q_limit_lower[i_q])
            #bounds.setHigh(i_q, q_limit_upper[i_q])
            bounds.setLow(int(i_q),  float(q_limit_lower[i_q]))  # <<< cast to Python float
            bounds.setHigh(int(i_q), float(q_limit_upper[i_q]))
        space.setBounds(bounds)
        ss = og.SimpleSetup(space)
        space_dim = space.getDimension()

        # Setting up OMPL collision checker
        if ignore_collision:
            ss.setStateValidityChecker(ob.StateValidityCheckerFn(lambda state: True))
        else:
            ss.setStateValidityChecker(ob.StateValidityCheckerFn(self._omplValidityChecker))
        #print(dir(og))
        # Setting up OMPL planner
        ss.setPlanner(getattr(og, planner)(ss.getSpaceInformation()))

        # Setting up OMPL start and goal states
        state_start = ob.State(space)
        state_goal = ob.State(space)
        for i_q in range(self.robot.n_qs):
            state_start[i_q] = float(qpos_start[i_q])
            state_goal[i_q] = float(qpos_goal[i_q])
        ss.setStartAndGoalStates(state_start, state_goal)

        # Solving with OMPL
        qpos_cur = self.robot.get_qpos()
        solved = ss.solve(float(timeout))
        waypoints = []
        if solved:
            if self.env_idx is not None:
                gs.logger.info(f"Path solution found successfully for {self.env_idx}.")
            #else:
            #    gs.logger.info(f"Path solution found.")
            path = ss.getSolutionPath()

            if smooth_path:
                ps = og.PathSimplifier(ss.getSpaceInformation())
                try:
                    ...
                    #ps.partialShortcutPath(path)
                    ps.ropeShortcutPath(path)
                except:
                    print("Using shortcut instead")
                    ps.shortcutPath(path)
                ps.smoothBSpline(path)
            
            if num_waypoints is not None:
                path.interpolate(int(num_waypoints))
            waypoints = self._omplStatesToTensorList(path.getStates())
        else:
            gs.logger.warning("Path planning failed. Returning empty path.")
        
        self.robot.set_qpos(qpos_cur)

        return waypoints


'''
def omplStateToTensor(state, n_qs):
    #n_qs = robot.n_qs
    tensor = torch.empty(n_qs, dtype=gs.tc_float, device=gs.device)
    for i in range(n_qs):
        tensor[i] = state[i]
    return tensor
    
def omplStatesToTensorList(states, n_qs):
    #tensor_list = []
    #for state in states:
    #    tensor_list.append(omplStateToTensor(state))
    #return tensor_list
    return [omplStateToTensor(st, n_qs) for st in states]

def omplValidityChecker(robot, state, env_idx):
    #env_idx = self.env_idx
    #(env_idx)
    robot.set_qpos(omplStateToTensor(state), envs_idx=torch.tensor(env_idx), zero_velocity=True)
    collision_pairs = robot.detect_collision(env_idx=env_idx[0])

    if len(collision_pairs)>0:
        return False
    else:
        return True

def omplPlan(
        robot,
        qpos_goal,
        qpos_start=None,
        timeout=10.0,
        smooth_path=True,
        num_waypoints=1000,
        ignore_collision=False,
        ignore_joint_limit=False,
        planner="RRTConnect",
        env_idx=None
    ):
        try:
            from ompl import base as ob
            from ompl import geometric as og
            from ompl import util as ou

            ou.setLogLevel(ou.LOG_ERROR)
        except:
            gs.raise_exception(
                "Failed to import OMPL. Did you install? (For installation instructions, see https://genesis-world.readthedocs.io/en/latest/user_guide/overview/installation.html#optional-motion-planning)"
            )
        
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
            gs.raise_exception(f"Planner {planner} is not supported. Supported planners: {supported_planners}.")

        #print(self.robot.n_qs)
        #print(self.robot.n_dofs)
        if robot.n_qs != robot.n_dofs:
            gs.raise_exception("Motion planning is not yet supported for rigid entities with free joints.")

        #self.env_idx = env_idx

        if qpos_start is None:
            qpos_start = robot.get_qpos()
            qpos_start = tensor_to_array(qpos_start)
            qpos_goal = tensor_to_array(qpos_goal)

        if qpos_start.shape != (robot.n_qs,) or qpos_goal.shape != (robot.n_qs,):
                gs.raise_exception("Invalid shape for `qpos_start` or `qpos_goal`.")

        if ignore_joint_limit:
                q_limit_lower = np.full_like(robot.q_limit[0], -1e6)
                q_limit_upper = np.full_like(robot.q_limit[1], 1e6)
        else:
            q_limit_lower = robot.q_limit[0]
            q_limit_upper = robot.q_limit[1]

        if (qpos_start < q_limit_lower).any() or (qpos_start > q_limit_upper).any():
            #gs.logger.warning(
            #    "`qpos_start` exceeds joint limit. Relaxing joint limit to contain `qpos_start` for planning."
            #)
            q_limit_lower = np.minimum(q_limit_lower, qpos_start)
            q_limit_upper = np.maximum(q_limit_upper, qpos_start)

        if (qpos_goal < q_limit_lower).any() or (qpos_goal > q_limit_upper).any():
            gs.logger.warning(
                "`qpos_goal` exceeds joint limit. Relaxing joint limit to contain `qpos_goal` for planning."
            )
            q_limit_lower = np.minimum(q_limit_lower, qpos_goal)
            q_limit_upper = np.maximum(q_limit_upper, qpos_goal)

        # Setting up OMPL StateSpace
        space = ob.RealVectorStateSpace(robot.n_qs)
        bounds = ob.RealVectorBounds(robot.n_qs)

        for i_q in range(robot.n_qs):
            bounds.setLow(i_q, q_limit_lower[i_q])
            bounds.setHigh(i_q, q_limit_upper[i_q])
        space.setBounds(bounds)
        ss = og.SimpleSetup(space)
        space_dim = space.getDimension()

        # Setting up OMPL collision checker
        if ignore_collision:
            ss.setStateValidityChecker(ob.StateValidityCheckerFn(lambda state: True))
        else:
            ss.setStateValidityChecker(ob.StateValidityCheckerFn(omplValidityChecker))
        #print(dir(og))
        # Setting up OMPL planner
        ss.setPlanner(getattr(og, planner)(ss.getSpaceInformation()))

        # Setting up OMPL start and goal states
        state_start = ob.State(space)
        state_goal = ob.State(space)
        for i_q in range(robot.n_qs):
            state_start[i_q] = float(qpos_start[i_q])
            state_goal[i_q] = float(qpos_goal[i_q])
        ss.setStartAndGoalStates(state_start, state_goal)

        # Solving with OMPL
        qpos_cur = robot.get_qpos()
        solved = ss.solve(timeout)
        waypoints = []
        if solved:
            gs.logger.info(f"Path solution found successfully for {env_idx}.")
            path = ss.getSolutionPath()

            if smooth_path:
                ps = og.PathSimplifier(ss.getSpaceInformation())
                try:
                    ...
                    #ps.partialShortcutPath(path)
                    ps.ropeShortcutPath(path)
                except:
                    print("Using shortcut instead")
                    ps.shortcutPath(path)
                ps.smoothBSpline(path)
            
            if num_waypoints is not None:
                path.interpolate(num_waypoints)
            waypoints = omplStatesToTensorList(path.getStates())
        else:
            gs.logger.warning("Path planning failed. Returning empty path.")
        
        robot.set_qpos(qpos_cur)

        return waypoints
'''
