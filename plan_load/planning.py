import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import numpy as np

import ompl.base as ob
import ompl.geometric as og
import ompl.util as ou

import mujoco
from plan_load.mujoco_utils import geoms_in_contact
from plan_load.robot import MujocoRobot


class OMPLPlanner:
    """
    OMPL Planner class for planning paths.
    Using Mujoco Robot with itsmodel and data for planning.
    """

    def __init__(
        self, robot: MujocoRobot, data=None, planner="RRTConnect", log=True
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
        self.n_dof = self.robot.n_dof
        self.joint_limits = self.robot.joint_limits

        # Set up OMPL planner
        self.planner = planner
        self.ss, self.si = self.set_up_ompl()
        if not log:
            ou.setLogLevel(ou.LOG_ERROR)

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

        # Setting up OMPL planner
        ss.setPlanner(getattr(og, self.planner)(si))
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
        t0 = time.perf_counter()
        waypoints = []
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

            states = path.getStates()
            waypoints = [
                np.array([s[i] for i in range(self.n_dof)], dtype=float)
                for s in states
            ]
        else:
            if log:
                print("Path planning failed.")

        if benchmark:
            t1 = time.perf_counter()
            # total time: plan + simplify + interpolation
            total_time = round(t1 - t0, 5)
            # planning time: time spent in planning
            planning_time = self.ss.getLastPlanComputationTime()
            return waypoints, total_time, planning_time
        return waypoints

    def validity_checker(self, state):
        """Check if the state is valid"""
        # set robot joint positions
        q = np.array([state[i] for i in range(self.robot.n_dof)], dtype=float)
        self.robot.set_joint_qpos(q, self.data)

        # Check for collisions
        in_contact = geoms_in_contact(self.model, self.data, self.robot_geoms)
        return not in_contact


if __name__ == "__main__":
    # Write a test case for the planner
    pass
