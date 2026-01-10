import numpy as np
import casadi as ca
from control import dlqr
from typing import Tuple
from mpt4py import Polyhedron
from .utils import LIMIT, Z, ALPHA, BETA, GAMA, WX, WY, LB_X, UB_X, LB_U, UB_U, X, Y, Z, DP, DR, P_AVG


class NmpcCtrl:
    """
    Nonlinear MPC controller.
    get_u should provide this functionality: u0, x_ol, u_ol, t_ol = mpc_z_rob.get_u(t0, x0).
    - x_ol shape: (12, N+1); u_ol shape: (4, N); t_ol shape: (N+1,)
    You are free to modify other parts
    """

    def __init__(self, rocket, Ts, H, xs, us):
        """
        Hint: As in our NMPC exercise, you can evaluate the dynamics of the rocket using
            CASADI variables x and u via the call rocket.f_symbolic(x,u).
            We create a self.f for you: x_dot = self.f(x,u)
        """
        # symbolic dynamics f(x,u) from rocket
        self.rocket = rocket
        self.f = lambda x, u: self.rocket.f_symbolic(x, u)[0]
        self.Ts = Ts
        self.H = H
        self.N = int(self.H / self.Ts)
        self.xs = xs
        self.us = us
        self.nx = len(self.xs)
        self.nu = len(self.us)
        self._setup_controller()

    def _setup_controller(self) -> None:
        N = self.N
        nx = self.nx
        nu = self.nu
        xs = self.xs
        us = self.us
        opti = ca.Opti()

        self.x = opti.variable(nx, N + 1)
        self.u = opti.variable(nu, N)
        self.x0 = opti.parameter(nx, 1)
        x = self.x
        u = self.u
        x0 = self.x0

        Q = ca.DM(np.eye(self.nx))
        R = ca.DM(np.eye(self.nu))
        Q[X, X] = 60
        Q[Y, Y] = 60
        Q[Z, Z] = 100
        Q[GAMA, GAMA] = 10

        R[DR, DR] = 700
        R[DP, DP] = 700
        A, B = self.rocket.linearize(xs, us)
        K, Qf, _ = dlqr(A, B, Q, R)
        K = -K
        Qf = ca.DM(Qf)
        A_cl = A - B @ K

        # cost function
        cost = 0
        for k in range(N):
            dx = x[:, k] - xs
            du = u[:, k] - us
            cost += dx.T @ Q @ dx
            cost += du.T @ R @ du

        dxN = x[:, N] - xs
        cost += dxN.T @ Qf @ dxN

        # initial constraint
        opti.subject_to(x[:, 0] == x0)

        # dynamics constraints
        for k in range(N):
            opti.subject_to(x[:, k + 1] == self.f_d(x[:, k], u[:, k]))

        # state constraints
        X_ = get_polyhedron(LB_X, UB_X)
        XA = ca.DM(X_.A)
        Xb = ca.DM(X_.b).reshape((-1, 1))
        opti.subject_to(ca.vec(XA @ x[:, :-1] - Xb) <= 0)
        O_inv = X_

        # input constraints
        U = get_polyhedron(LB_U, UB_U)
        UA = ca.DM(U.A)
        Ub = ca.DM(U.b)
        opti.subject_to(ca.vec(UA @ u - Ub) <= 0)

        # terminal constraint
        KU = Polyhedron.from_Hrep(U.A @ K, U.b - U.A @ us)
        O_inv = O_inv.intersect(KU)
        O_inv = max_invariant_set(A_cl, O_inv)
        O_invA = ca.DM(O_inv.A)
        O_invb = ca.DM(O_inv.b).reshape((-1, 1))
        opti.subject_to(ca.vec(O_invA @ self.x[:, -1] - O_invb) <= 0)

        # set solver
        options = {
            "expand": True,
            "print_time": False,
            "ipopt": {"sb": "yes", "print_level": 0, "tol": 1e-3},
        }
        opti.solver("ipopt", options)
        opti.minimize(cost)

        self.ocp = opti

    def get_u(
        self, t0: float, x0: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        self.ocp.set_value(self.x0, x0)
        self.ocp.set_initial(self.x, ca.repmat(x0, 1, self.N + 1))
        self.ocp.set_initial(self.u, ca.repmat(self.us, 1, self.N))

        sol = self.ocp.solve()

        x_ol = sol.value(self.x)
        u_ol = sol.value(self.u)
        u0 = u_ol[:, 0]
        t_ol = t0 + self.Ts * np.array(range(self.N + 1))

        return u0, x_ol, u_ol, t_ol

    def f_d(self, x, u):
        return rk4(self.f, x, u, self.Ts)


def rk4(f, x, u, h):
    k1 = h * f(x, u)
    k2 = h * f(x + k1 / 2, u)
    k3 = h * f(x + k2 / 2, u)
    k4 = h * f(x + k3, u)
    return x + (k1 + 2 * k2 + 2 * k3 + k4) / 6


def get_polyhedron(lb, ub):
    n = len(lb)
    M = np.vstack((-np.eye(n), np.eye(n)))
    m = np.hstack((-lb, ub))
    idx = m != LIMIT
    return Polyhedron.from_Hrep(np.array(M[idx]), np.array(m[idx]))


def max_invariant_set(A_cl, X, max_iter=100):
    O = X
    itr = 1
    converged = False
    while itr < max_iter:
        Oprev = O
        F, f = O.A, O.b
        # Compute the pre-set
        O = Polyhedron.from_Hrep(
            np.vstack((F, F @ A_cl)), np.vstack((f, f)).reshape((-1,))
        )
        O.minHrep()
        if O == Oprev:
            converged = True
            break
        itr += 1

    if converged:
        print(
            "Maximum invariant set successfully computed after {0} iterations.\n".format(
                itr
            )
        )
    return O
