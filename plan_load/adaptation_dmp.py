"""
Dynamic Movement Primitives

DMP is a method of trajectory control / planning from Stefan Schaal's lab.
This method is first proposed in 2002 and later refined in:
Dynamical Movement Primitives: Learning Attractor Models for Motor Behaviors
https://ieeexplore.ieee.org/document/6797340

Implementation from pydmps:
https://github.com/studywolf/pydmps?tab=readme-ov-file

Modified to improve efficiency and stability
"""

import numpy as np


class CanonicalSystem:
    """Implementation of the canonical dynamical system
    as described in Dr. Stefan Schaal's (2002) paper"""

    def __init__(self, dt, ax=1.0, pattern="discrete"):
        """Default values from Schaal (2012)

        dt float: the timestep
        ax float: a gain term on the dynamical system
        pattern string: either 'discrete' or 'rhythmic'
        """
        self.ax = ax

        self.pattern = pattern
        if pattern == "discrete":
            self.step = self.step_discrete
            self.run_time = 1.0
        elif pattern == "rhythmic":
            self.step = self.step_rhythmic
            self.run_time = 2 * np.pi
        else:
            raise Exception(
                "Invalid pattern type specified: \
                Please specify rhythmic or discrete."
            )

        self.dt = dt
        # Include start
        self.timesteps = int(round(self.run_time / self.dt)) + 1

        self.reset_state()

    def rollout(self, **kwargs):
        """Generate x for open loop movements."""
        if "tau" in kwargs:
            timesteps = int(self.timesteps / kwargs["tau"])
        else:
            timesteps = self.timesteps
        self.x_track = np.zeros(timesteps)

        self.reset_state()

        # Include start
        self.x_track[0] = self.x
        for t in range(1, timesteps):
            self.x_track[t] = self.x
            self.step(**kwargs)

        return self.x_track

    def reset_state(self):
        """Reset the system state"""
        self.x = 1.0

    def step_discrete(self, tau=1.0, error_coupling=1.0):
        """Generate a single step of x for discrete
        (potentially closed) loop movements.
        Decaying from 1 to 0 according to dx = -ax*x.

        tau float: gain on execution time
                   increase tau to make the system execute faster
        error_coupling float: slow down if the error is > 1
        """
        self.x += (-self.ax * self.x * error_coupling) * tau * self.dt
        return self.x

    def step_rhythmic(self, tau=1.0, error_coupling=1.0):
        """Generate a single step of x for rhythmic
        closed loop movements. Decaying from 1 to 0
        according to dx = -ax*x.

        tau float: gain on execution time
                   increase tau to make the system execute faster
        error_coupling float: slow down if the error is > 1
        """
        self.x += (1 * error_coupling * tau) * self.dt
        return self.x


class DMP(object):
    """Implementation of Dynamic Motor Primitives,
    as described in Dr. Stefan Schaal's (2002) paper."""

    def __init__(
        self,
        n_dmps,
        n_bfs,
        dt=0.01,
        y0=0,
        goal=1,
        w=None,
        ay=None,
        by=None,
        **kwargs
    ):
        """
        n_dmps int: number of dynamic motor primitives
        n_bfs int: number of basis functions per DMP
        dt float: timestep for simulation
        y0 list: initial state of DMPs
        goal list: goal state of DMPs
        w list: tunable parameters, control amplitude of basis functions
        ay int: gain on attractor term y dynamics
        by int: gain on attractor term y dynamics
        """

        self.n_dmps = n_dmps
        self.n_bfs = n_bfs
        self.dt = dt
        if isinstance(y0, (int, float)):
            y0 = np.ones(self.n_dmps) * y0
        self.y0 = y0
        if isinstance(goal, (int, float)):
            goal = np.ones(self.n_dmps) * goal
        self.goal = goal
        if w is None:
            # default is f = 0
            w = np.zeros((self.n_dmps, self.n_bfs))
        self.w = w

        self.ay = np.ones(n_dmps) * 25.0 if ay is None else ay  # Schaal 2012
        self.by = self.ay / 4.0 if by is None else by  # Schaal 2012

        # set up the CS
        self.cs = CanonicalSystem(dt=self.dt, **kwargs)
        self.timesteps = self.cs.timesteps

        # set up the DMP system
        self.reset_state()

    def check_offset(self):
        """Check to see if initial position and goal are the same
        if they are, offset slightly so that the forcing term is not 0"""

        for d in range(self.n_dmps):
            if abs(self.y0[d] - self.goal[d]) < 1e-4:
                self.goal[d] += 1e-4

    def gen_front_term(self, x, dmp_num):
        raise NotImplementedError()

    def gen_goal(self, y_des):
        raise NotImplementedError()

    def gen_psi(self):
        raise NotImplementedError()

    def gen_weights(self, f_target):
        raise NotImplementedError()

    def imitate_path(self, y_des, plot=False):
        """Takes in a desired trajectory and generates the set of
        system parameters that best realize this path.

        y_des list/array: the desired trajectories of each DMP
                          should be shaped [n_dmps, run_time]
        """

        # set initial state and goal
        if y_des.ndim == 1:
            y_des = y_des.reshape(1, len(y_des))
        self.y0 = y_des[:, 0].copy()
        self.y_des = y_des.copy()
        self.goal = self.gen_goal(y_des)

        # self.check_offset()

        # generate function to interpolate the desired trajectory
        import scipy.interpolate

        path = np.zeros((self.n_dmps, self.timesteps))
        x = np.linspace(0, self.cs.run_time, y_des.shape[1])
        for d in range(self.n_dmps):
            path_gen = scipy.interpolate.interp1d(x, y_des[d])
            for t in range(self.timesteps):
                path[d, t] = path_gen(t * self.dt)
        y_des = path

        # calculate velocity of y_des with central differences
        dy_des = np.gradient(y_des, axis=1) / self.dt

        # calculate acceleration of y_des with central differences
        ddy_des = np.gradient(dy_des, axis=1) / self.dt

        f_target = np.zeros((y_des.shape[1], self.n_dmps))
        # find the force required to move along this trajectory
        for d in range(self.n_dmps):
            f_target[:, d] = ddy_des[d] - self.ay[d] * (
                self.by[d] * (self.goal[d] - y_des[d]) - dy_des[d]
            )

        # efficiently generate weights to realize f_target
        self.gen_weights(f_target)

        if plot is True:
            # plot the basis function activations
            import matplotlib.pyplot as plt

            plt.figure()
            plt.subplot(211)
            psi_track = self.gen_psi(self.cs.rollout())
            plt.plot(psi_track)
            plt.title("basis functions")

            # plot the desired forcing function vs approx
            for ii in range(self.n_dmps):
                plt.subplot(2, self.n_dmps, self.n_dmps + 1 + ii)
                plt.plot(f_target[:, ii], "--", label="f_target %i" % ii)
            for ii in range(self.n_dmps):
                plt.subplot(2, self.n_dmps, self.n_dmps + 1 + ii)
                plt.plot(
                    np.sum(psi_track * self.w[ii], axis=1) * self.dt,
                    label="w*psi %i" % ii,
                )
                plt.legend()
            plt.title("DMP forcing function")
            plt.tight_layout()
            plt.show()

        self.reset_state()
        return y_des

    # Original rollout
    # def rollout(self, timesteps=None, **kwargs):
    #     """Generate a system trial, no feedback is incorporated."""

    #     self.reset_state()

    #     if timesteps is None:
    #         if "tau" in kwargs:
    #             timesteps = int(self.timesteps / kwargs["tau"])
    #         else:
    #             timesteps = self.timesteps

    #     # set up tracking vectors
    #     y_track = np.zeros((timesteps, self.n_dmps))
    #     dy_track = np.zeros((timesteps, self.n_dmps))
    #     ddy_track = np.zeros((timesteps, self.n_dmps))

    #     # FIX
    #     # record initial state
    #     y_track[0] = self.y.copy()
    #     dy_track[0] = self.dy.copy()
    #     ddy_track[0] = self.ddy.copy()
    #     for t in range(1, timesteps):

    #         # run and record timestep
    #         y_track[t], dy_track[t], ddy_track[t] = self.step(**kwargs)

    #     return y_track, dy_track, ddy_track

    def rollout(self, timesteps=None, **kwargs):
        """Fast open-loop rollout for discrete DMPs."""
        self.reset_state()

        # Hanlde residual fitting
        if "residual" in kwargs:
            residual = kwargs["residual"]
        else:
            residual = False

        if "tau" in kwargs:
            tau = kwargs["tau"]
        else:
            tau = 1.0
        if timesteps is None:
            timesteps = (
                int(self.timesteps / tau) if tau != 1.0 else self.timesteps
            )

        # T = timesteps
        # D = self.n_dmps

        # Precompute canonical system x(t) by stepping once
        x_track = np.empty(timesteps, dtype=float)
        x = 1.0
        x_track[0] = x
        ax_dt_tau = self.cs.ax * self.dt * tau  # constant for open loop
        for t in range(1, timesteps):
            x = x + (-ax_dt_tau) * x  # error_coupling=1
            x_track[t] = x

        # Precompute psi, normalized psi, and f_hat(t,d)
        # ensure h and c are 1D arrays shape (B,)
        diff = x_track[:, None] - self.c[None, :]
        psi = np.exp(-self.h[None, :] * diff * diff)

        # normalized basis times x(psi / sumpsi) * x
        psi_sum = np.sum(psi, axis=1, keepdims=True) + 1e-12
        phi = (psi / psi_sum) * x_track[:, None]  # (T, B)

        # f_hat: (T, D) = phi @ w.T (equals (psi@w.T)/sumpsi * x)
        f_hat = phi @ self.w.T  # (T, D)

        # front term for discrete: x*(goal - y0), vectorized over D
        k = self.goal - self.y0  # (D,)
        if residual:
            k = np.where(np.abs(k) < 1e-6, 1.0, k)

        # Outputs
        y_track = np.empty((timesteps, self.n_dmps), dtype=float)
        dy_track = np.empty((timesteps, self.n_dmps), dtype=float)
        ddy_track = np.empty((timesteps, self.n_dmps), dtype=float)

        # initial
        y = self.y.copy()
        dy = self.dy.copy()
        y_track[0] = y
        dy_track[0] = dy
        ddy_track[0] = 0.0

        # loops over time, but vectorized over D
        ay = self.ay
        by = self.by
        dt_tau = self.dt * tau
        for t in range(1, timesteps):
            # forcing term: front * (psi·w/sumpsi)
            # f_hat already includes x and normalization
            f = k * f_hat[t]  # (D,)

            ddy = ay * (by * (self.goal - y) - dy) + f
            dy = dy + ddy * dt_tau
            y = y + dy * dt_tau
            y_track[t] = y
            dy_track[t] = dy
            ddy_track[t] = ddy

        # write back state
        self.y, self.dy, self.ddy = y, dy, ddy
        return y_track, dy_track, ddy_track

    def reset_state(self):
        """Reset the system state"""
        self.y = self.y0.copy()
        self.dy = np.zeros(self.n_dmps)
        self.ddy = np.zeros(self.n_dmps)
        self.cs.reset_state()

    def step(self, tau=1.0, error=0.0, external_force=None):
        """Run the DMP system for a single timestep.

        tau float: scales the timestep
                   increase tau to make the system execute faster
        error float: optional system feedback
        """
        error_coupling = 1.0 / (1.0 + error)

        # Run canonical system later
        x = self.cs.x

        # generate basis function activation
        psi = self.gen_psi(x)

        for d in range(self.n_dmps):

            # generate the forcing term
            f = self.gen_front_term(x, d) * (np.dot(psi, self.w[d]))
            sum_psi = np.sum(psi)
            if np.abs(sum_psi) > 1e-6:
                f /= sum_psi

            # DMP acceleration
            self.ddy[d] = (
                self.ay[d]
                * (self.by[d] * (self.goal[d] - self.y[d]) - self.dy[d])
                + f
            )
            if external_force is not None:
                self.ddy[d] += external_force[d]
            self.dy[d] += self.ddy[d] * tau * self.dt * error_coupling
            self.y[d] += self.dy[d] * tau * self.dt * error_coupling

        # run canonical system
        x = self.cs.step(tau=tau, error_coupling=error_coupling)

        return self.y, self.dy, self.ddy


class DMPDiscete(DMP):
    """An implementation of discrete DMPs"""

    def __init__(self, **kwargs):
        """ """

        # call super class constructor
        super(DMPDiscete, self).__init__(pattern="discrete", **kwargs)

        self.gen_centers()

        # set variance of Gaussian basis functions
        # trial and error to find this spacing
        self.h = np.ones(self.n_bfs) * self.n_bfs**1.5 / self.c / self.cs.ax

        self.check_offset()

    def gen_centers(self):
        """Set the centre of the Gaussian basis
        functions be spaced evenly throughout run time"""

        """x_track = self.cs.discrete_rollout()
        t = np.arange(len(x_track))*self.dt
        # choose the points in time we'd like centers to be at
        c_des = np.linspace(0, self.cs.run_time, self.n_bfs)
        self.c = np.zeros(len(c_des))
        for ii, point in enumerate(c_des):
            diff = abs(t - point)
            self.c[ii] = x_track[np.where(diff == min(diff))[0][0]]"""

        # desired activations throughout time
        des_c = np.linspace(0, self.cs.run_time, self.n_bfs)

        self.c = np.ones(len(des_c))
        for n in range(len(des_c)):
            # finding x for desired times t
            self.c[n] = np.exp(-self.cs.ax * des_c[n])

    def gen_front_term(self, x, dmp_num):
        """Generates the diminishing front term on
        the forcing term.

        x float: the current value of the canonical system
        dmp_num int: the index of the current dmp
        """
        return x * (self.goal[dmp_num] - self.y0[dmp_num])

    def gen_goal(self, y_des):
        """Generate the goal for path imitation.
        For rhythmic DMPs the goal is the average of the
        desired trajectory.

        y_des np.array: the desired trajectory to follow
        """

        return np.copy(y_des[:, -1])

    def gen_psi(self, x):
        """Generates the activity of the basis functions for a given
        canonical system rollout.

        x float, array: the canonical system state or path
        """

        if isinstance(x, np.ndarray):
            x = x[:, None]
        return np.exp(-self.h * (x - self.c) ** 2)

    def gen_weights(self, f_target):
        """Generate a set of weights over the basis functions such
        that the target forcing term trajectory is matched.

        f_target np.array: the desired forcing term trajectory
        """
        # Per-basis Implementation
        # # calculate x and psi
        # x_track = self.cs.rollout()
        # psi_track = self.gen_psi(x_track)

        # # efficiently calculate BF weights using weighted linear regression
        # self.w = np.zeros((self.n_dmps, self.n_bfs))
        # for d in range(self.n_dmps):
        #     # spatial scaling term
        #     k = self.goal[d] - self.y0[d]
        #     for b in range(self.n_bfs):
        #         numer = np.sum(x_track * psi_track[:, b] * f_target[:, d])
        #         denom = np.sum(x_track**2 * psi_track[:, b]) + 1e-9
        #         self.w[d, b] = numer / denom
        #         if abs(k) > 1e-5:
        #             self.w[d, b] /= k

        # Solve weights with ridge least squares
        x_track = self.cs.rollout()
        psi = self.gen_psi(x_track)  # (T, B)
        psi_sum = np.sum(psi, axis=1, keepdims=True) + 1e-12

        # features: (T, B)
        phi = (psi / psi_sum) * x_track[:, None]

        for d in range(self.n_dmps):
            k = self.goal[d] - self.y0[d]
            y = f_target[:, d]
            if abs(k) > 1e-6:
                y = y / k

            # ridge
            lam = 1e-6
            a = phi.T @ phi + lam * np.eye(self.n_bfs)
            b = phi.T @ y
            self.w[d] = np.linalg.solve(a, b)

        self.w = np.nan_to_num(self.w)


class DMPRhythmic(DMP):
    """An implementation of discrete DMPs"""

    def __init__(self, **kwargs):
        """ """

        # call super class constructor
        super(DMPRhythmic, self).__init__(pattern="rhythmic", **kwargs)

        self.gen_centers()

        # set variance of Gaussian basis functions
        # trial and error to find this spacing
        self.h = np.ones(self.n_bfs) * self.n_bfs  # 1.75

        self.check_offset()

    def gen_centers(self):
        """Set the centre of the Gaussian basis
        functions be spaced evenly throughout run time"""

        c = np.linspace(0, 2 * np.pi, self.n_bfs + 1)
        c = c[0:-1]
        self.c = c

    def gen_front_term(self, x, dmp_num):
        """Generates the front term on the forcing term.
        For rhythmic DMPs it's non-diminishing, so this
        function is just a placeholder to return 1.

        x float: the current value of the canonical system
        dmp_num int: the index of the current dmp
        """

        if isinstance(x, np.ndarray):
            return np.ones(x.shape)
        return 1

    def gen_goal(self, y_des):
        """Generate the goal for path imitation.
        For rhythmic DMPs the goal is the average of the
        desired trajectory.

        y_des np.array: the desired trajectory to follow
        """

        goal = np.zeros(self.n_dmps)
        for n in range(self.n_dmps):
            num_idx = ~np.isnan(y_des[n])  # ignore nan's when calculating goal
            goal[n] = 0.5 * (y_des[n, num_idx].min() + y_des[n, num_idx].max())

        return goal

    def gen_psi(self, x):
        """Generates the activity of the basis functions for a given
        canonical system state or path.

        x float, array: the canonical system state or path
        """

        if isinstance(x, np.ndarray):
            x = x[:, None]
        return np.exp(self.h * (np.cos(x - self.c) - 1))

    def gen_weights(self, f_target):
        """Generate a set of weights over the basis functions such
        that the target forcing term trajectory is matched.

        f_target np.array: the desired forcing term trajectory
        """

        # calculate x and psi
        x_track = self.cs.rollout()
        psi_track = self.gen_psi(x_track)

        # efficiently calculate BF weights using weighted linear regression
        for d in range(self.n_dmps):
            for b in range(self.n_bfs):
                self.w[d, b] = np.dot(psi_track[:, b], f_target[:, d]) / (
                    np.sum(psi_track[:, b]) + 1e-10
                )
