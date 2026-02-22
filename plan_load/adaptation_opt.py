"""
TrajOpt

TrajOpt is a sequential convex optimization framework to solve
motion planning problems. It implements a penalty method to
optimize for joint velocities while satisfying a set of constraints.
Internally, it makes use of convex solvers that are able to
solve linearly constrained quadratic problems.
Full implementation in:
https://github.com/tesseract-robotics/trajopt

Here is a simplified implementation of TrajOpt using OSQP.
No obstacles avoidance and kinematic constraints are considered.
"""

import numpy as np
import scipy.sparse as sp
import osqp


class TrajOpt:
    """
    TrajOpt is a method for joint-space trajectory optimization
    using sequential convex QP solved by OSQP.
    Given that no obstacles avoidance or kinematic constraints are considered,
    this can be simplified to a single convex optimization.

    Constraints:
        q_0 == seed[0]
        q_{n_p-1} == q_goal
        joint limits as box bounds
    Soft cost:
        similar to seed
        velocity smoothness
        acceleration smoothness
    """

    def __init__(
        self,
        w_seed=1.0,
        w_vel=1e-2,
        w_acc=1.0,
        joint_limits=None,
        verbose=False,
    ):
        self.w_seed = w_seed
        self.w_vel = w_vel
        self.w_acc = w_acc
        joint_limits = np.asarray(joint_limits)
        self.joint_lb = joint_limits[0, :]
        self.joint_ub = joint_limits[1, :]
        self.verbose = verbose

    def solve(self, seed_traj, q_goal):
        """Solve the TrajOpt optimization problem"""
        seed = np.asarray(seed_traj)
        n_p, n_j = seed.shape
        n = n_p * n_j

        # Flatten the seed trajectory
        x_seed = seed.reshape(-1)

        # Build Hessian penalty
        penalty = sp.csc_matrix((n, n))
        eye_j = sp.eye(n_j, format="csc")
        # smoothness
        if self.w_vel > 0:
            d1 = self.diff_matrix(n_p, 1)
            penalty += 2 * self.w_vel * sp.kron(d1.T @ d1, eye_j)
        if self.w_acc > 0:
            d2 = self.diff_matrix(n_p, 2)
            penalty += 2 * self.w_acc * sp.kron(d2.T @ d2, eye_j)
        # similarity
        if self.w_seed > 0:
            penalty += 2 * self.w_seed * sp.eye(n)

        # Equality constraints (start + goal)
        rows, cols, data = [], [], []

        # fix first waypoint
        for j in range(n_j):
            rows.append(j)
            cols.append(j)
            data.append(1.0)
        # fix last waypoint
        base = (n_p - 1) * n_j
        for j in range(n_j):
            rows.append(n_j + j)
            cols.append(base + j)
            data.append(1.0)

        a_eq = sp.csc_matrix((data, (rows, cols)), shape=(2 * n_j, n))
        beq = np.concatenate([seed[0], q_goal])

        # Joint bounds
        lb = np.tile(self.joint_lb, n_p)
        ub = np.tile(self.joint_ub, n_p)

        a = sp.vstack([a_eq, sp.eye(n)], format="csc")
        l = np.concatenate([beq, lb])
        u = np.concatenate([beq, ub])

        # Linear term
        q = -2 * self.w_seed * x_seed

        # Solve QP
        prob = osqp.OSQP()
        prob.setup(P=penalty, q=q, A=a, l=l, u=u, verbose=self.verbose)
        prob.warm_start(x=x_seed)
        res = prob.solve()

        if res.x is None:
            return False, None
        return True, res.x.reshape(n_p, n_j)

    @staticmethod
    def diff_matrix(n, order):
        """Build the difference matrix"""
        if order == 1:
            m = n - 1
            return sp.diags([-1, 1], [0, 1], shape=(m, n), format="csc")
        elif order == 2:
            m = n - 2
            return sp.diags([1, -2, 1], [0, 1, 2], shape=(m, n), format="csc")
        else:
            raise ValueError("Only order 1 or 2 supported")
