import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import numpy as np

import ompl.base as ob
import ompl.geometric as og
import ompl.util as ou

import mujoco
from plan_load.robot import MujocoRobot




class OMPLPlanner:
    """
    OMPL Planner class for planning paths.
    Using Mujoco Robot with itsmodel and data for planning.
    """

    def __init__(
        self, robot: MujocoRobot, data=None, planner="RRTConnect", rrtc_range=None, log=False
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
        self.planner = self.ss.getPlanner()
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

        planner = getattr(og, self.planner_name)(si)

        # ---- Adaptive range ----
        if self.planner_name == "RRTConnect" and self.range is not None:
            extent = si.getMaximumExtent()
            planner.setRange(self.range * extent)

        # Setting up OMPL planner
        #ss.setPlanner(getattr(og, self.planner_name)(si))
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

    def construct_roadmap(self, start, timeout=30.0):
        """
        Sample uniformly in the state space and build a roadmap.
        The roadmap is kept for subsequent planning calls.
        """
        assert (
            "PRM" in self.planner_name
        ), f"Planner {self.planner_name} is not supported."

        # Set start and a dummy goal (same as start)
        # so ProblemDefinition is valid.
        start_state = ob.State(self.si.getStateSpace())
        for i in range(self.n_dof):
            start_state[i] = float(start[i])
        self.ss.setStartAndGoalStates(start_state, start_state)

        # Start growing the roadmap
        self.ss.setup()
        ter = ob.timedPlannerTerminationCondition(float(timeout))
        self.planner.constructRoadmap(ter)

    def query(
        self,
        start,
        goal,
        timeout=10.0,
        smooth_path=True,
        num_waypoints=200,
        benchmark=False,
        check_time_freq=1e-3,
    ):
        """
        Solve a start-goal query using the pre-built roadmap from
        sample_for_batch_planning. Reuses the roadmap; call clearQuery
        between queries to clear only the previous start/goal.
        """
        assert (
            "PRM" in self.planner_name
        ), f"Planner {self.planner_name} is not supported."

        # Clear previous query (start/goal) but keep the roadmap
        self.planner.clearQuery()

        # Set new start and goal
        start_state = ob.State(self.si.getStateSpace())
        goal_state = ob.State(self.si.getStateSpace())
        for i in range(self.n_dof):
            start_state[i] = float(start[i])
            goal_state[i] = float(goal[i])
        self.ss.setStartAndGoalStates(start_state, goal_state)

        # TODO
        # need customized OMPL implementation here
        t0 = time.perf_counter()
        waypoints = []
        timeout_c = time.perf_counter()
        status_str = ""
        while status_str != "Exact solution":
            status = self.ss.solve(check_time_freq)
            status_str = status.asString()
            if time.perf_counter() - timeout_c > float(timeout):
                break

        # Check collision since environment is changed
        valid_path = False
        if status.asString() == "Exact solution":
            valid_path = True
            path = self.ss.getSolutionPath()
            for state in path.getStates():
                if not self.validity_checker(state):
                    valid_path = False
                    break

        if status.asString() == "Exact solution" and valid_path:
            path = self.ss.getSolutionPath()
            if smooth_path:
                ps = og.PathSimplifier(self.si)
                try:
                    ps.ropeShortcutPath(path)
                except Exception:
                    ps.shortcutPath(path)
                ps.smoothBSpline(path)
            if num_waypoints is not None:
                path.interpolate(int(num_waypoints))
            states = path.getStates()
            waypoints = [
                np.array([s[i] for i in range(self.n_dof)], dtype=float)
                for s in states
            ]

        if benchmark:
            t1 = time.perf_counter()
            total_time = round(t1 - t0, 5)
            planning_time = self.ss.getLastPlanComputationTime()
            return waypoints, total_time, planning_time
        return waypoints

    def validity_checker(self, state):
        """Check if the state is valid"""
        # set robot joint positions
        q = np.array([state[i] for i in range(self.n_dof)], dtype=float)
        self.robot.set_joint_qpos(q)

        # Check for collisions
        in_contact = self.robot.in_contact()
        return not in_contact


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
