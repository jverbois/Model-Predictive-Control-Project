import cvxpy as cp
import numpy as np
from control import dlqr
from mpt4py import Polyhedron
from scipy.signal import cont2discrete
from scipy.signal import place_poles
import matplotlib.pyplot as plt
from .utils import LIMIT, X_TO_STRING, U_TO_STRING, BD, NB_D, LB_W, UB_W, VZ


class MPCControl_base:
    """Complete states indices"""

    x_ids: np.ndarray
    u_ids: np.ndarray

    """Optimization system"""
    A: np.ndarray
    B: np.ndarray
    xs: np.ndarray
    us: np.ndarray
    nx: int
    nu: int
    Ts: float
    H: float
    N: int

    """Optimization problem"""
    ocp: cp.Problem

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        x_ids: np.ndarray,
        u_ids: np.ndarray,
        xs: np.ndarray,
        us: np.ndarray,
        Ts: float,
        H: float,
    ) -> None:
        self.Ts = Ts
        self.H = H
        self.N = int(H / Ts)
        self.x_ids = x_ids
        self.u_ids = u_ids
        self.nx = len(self.x_ids)
        self.nu = len(self.u_ids)
        self.nd = NB_D
        self.Q = np.eye(self.nx)
        self.R = np.eye(self.nu)
        self.lb_u = None
        self.ub_u = None
        self.lb_x = None
        self.ub_x = None
        self.x_hat_next = None
        self.d_hat = None

        # System definition
        A_red = A[np.ix_(self.x_ids, self.x_ids)]
        B_red = B[np.ix_(self.x_ids, self.u_ids)]

        self.A, self.B = self._discretize(A_red, B_red, Ts)

        self.Bd = BD[self.x_ids, :]

        self.xs = xs[self.x_ids]
        self.us = us[self.u_ids]

        self._setup_controller()

    def _setup_controller(self) -> None:
        #################################################
        # YOUR CODE HERE

        # Cost matrices
        K, Qf, _ = dlqr(self.A, self.B, self.Q, self.R)
        K = -K
        self.A_cl = np.array(self.A + self.B @ K)

        # Variables and parameters
        nx, nu, N = self.nx, self.nu, self.N
        self.x_var = cp.Variable((nx, N + 1), name="x")
        self.s_var = cp.Variable((nx, N + 1), name="s")
        self.u_var = cp.Variable((nu, N), name="u")
        self.x0_par = cp.Parameter(nx, name="x0_hat")
        self.d_hat_par = cp.Parameter(self.nd, name="d_hat")

        # Stage cost
        cost = 0
        for i in range(N):
            cost += cp.quad_form(self.x_var[:, i], self.Q)
            cost += cp.quad_form(self.u_var[:, i], self.R)
            if not np.any(VZ == self.x_ids):
                cost += 1e6 * cp.sum_squares(self.s_var[:, i])
        # Terminal cost
        cost += cp.quad_form(self.x_var[:, N], Qf)

        constraints = []
        if np.any(VZ == self.x_ids):
            V = np.vstack((-np.eye(nu), np.eye(nu)))
            v = np.hstack((-LB_W[self.u_ids], UB_W[self.u_ids]))
            W = Polyhedron.from_Hrep(V, v)
            E = self.min_robust_invariant_set(self.B @ W)

        # initial condition
        constraints.append(self.x_var[:, 0] == self.x0_par)

        # dynamics
        for k in range(N):
            constraints.append(
                self.x_var[:, k + 1]
                == self.A @ self.x_var[:, k]
                + self.B @ self.u_var[:, k]
                + self.Bd @ self.d_hat_par
            )

        constr_set = None
        if self.lb_u is not None and self.ub_u is not None:
            # U = { u | M @ u <= m } / -u <= -lb_u, u <= ub_u
            M = np.vstack((-np.eye(nu), np.eye(nu)))
            m = np.hstack((-self.lb_u, self.ub_u))
            idx = m != LIMIT
            U = Polyhedron.from_Hrep(M[idx, :], m[idx] - M[idx, :] @ self.us)
            # Input constraints
            constraints.append(U.A @ self.u_var[:, :-1] <= U.b.reshape((-1, 1)))
            if np.any(VZ == self.x_ids):
                KE = K @ E
                U = U - KE
                # Map to state constraints via feedback K
                KU = Polyhedron.from_Hrep(U.A @ K, U.b)
                constr_set = KU

        if self.lb_x is not None and self.ub_x is not None:
            # X = { x | F @ x <= f } / -x <= -lb_x, x <= ub_x
            F = np.vstack((-np.eye(nx), np.eye(nx)))
            f = np.hstack((-self.lb_x, self.ub_x))
            idx = f != LIMIT
            X = Polyhedron.from_Hrep(F[idx], f[idx] - F[idx] @ self.xs)
            # State constraints

            if np.any(VZ == self.x_ids):
                X = X - E
                constraints.append(X.A @ self.x_var[:, :-1] <= X.b.reshape((-1, 1)))
                if constr_set is None:
                    constr_set = X
                else:
                    constr_set = constr_set.intersect(X)
            else:
                constraints.append(
                    X.A @ self.x_var <= X.b.reshape((-1, 1)) + X.A @ self.s_var
                )

        if constr_set is not None:
            # Terminal invariant set
            self.max_invariant_set(constr_set)
            # Terminal constraints
            constraints.append(
                self.O_inf.A @ self.x_var[:, -1] <= self.O_inf.b.reshape((-1, 1))
            )

        self.ocp = cp.Problem(cp.Minimize(cost), constraints)

        # YOUR CODE HERE
        #################################################

    @staticmethod
    def _discretize(A: np.ndarray, B: np.ndarray, Ts: float):
        nx, nu = B.shape
        C = np.zeros((1, nx))
        D = np.zeros((1, nu))
        A_discrete, B_discrete, _, _, _ = cont2discrete(system=(A, B, C, D), dt=Ts)
        return A_discrete, B_discrete

    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        #################################################
        # YOUR CODE HERE

        if x_target is None:
            x_target = self.xs
        if u_target is None:
            u_target = self.us

        self.x0_par.value = x0 - x_target
        if np.any(VZ == self.x_ids):
            if self.d_hat is None:
                self.d_hat = np.zeros(self.nd)
            self.d_hat_par.value = self.d_hat
        else:
            self.d_hat_par.value = np.zeros(self.nd)

        self.ocp.solve(solver=cp.PIQP, verbose=False)
        assert self.ocp.status == cp.OPTIMAL

        x_traj = np.array(self.x_var.value)
        u_traj = np.array(self.u_var.value)

        if np.any(VZ == self.x_ids):
            if self.x_hat_next is None:
                self.x_hat_next = x_traj[:, 0]
            innovation = x_traj[:, 0] - self.x_hat_next
            self.d_hat += 0.1 * innovation[VZ == self.x_ids]
            self.x_hat_next = (
                self.A @ x_traj[:, 0] + self.B @ u_traj[:, 0] + self.Bd @ self.d_hat
            )
        x_traj += x_target.reshape(-1, 1)
        u_traj += u_target.reshape(-1, 1)

        # YOUR CODE HERE
        #################################################

        return u_traj[:, 0], x_traj, u_traj

    def max_invariant_set(self, X: Polyhedron, max_iter=1000):
        """
        Compute invariant set for an autonomous linear time invariant system x^+ = A_cl x
        """
        O = X
        itr = 1
        converged = False
        while itr < max_iter:
            Oprev = O
            F, f = O.A, O.b
            # Compute the pre-set
            O = Polyhedron.from_Hrep(
                np.vstack((F, F @ self.A_cl)), np.vstack((f, f)).reshape((-1,))
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
        self.O_inf = O

    def min_robust_invariant_set(
        self, W: Polyhedron, max_iter: int = 300
    ) -> Polyhedron:
        Omega = W
        itr = 1
        while itr < max_iter:
            A_cl_ith_power = np.linalg.matrix_power(self.A_cl, itr)
            Omega = Omega + A_cl_ith_power @ W
            Omega.minHrep()  # optionally: Omega_next.minVrep()
            if np.linalg.matrix_norm(A_cl_ith_power, ord=2) < 1e-2:
                print(
                    "Minimal robust invariant set computation converged after {0} iterations.".format(
                        itr
                    )
                )
                break

            if itr == max_iter:
                print(
                    "Minimal robust invariant set computation did NOT converge after {0} iterations.".format(
                        itr
                    )
                )

            itr += 1
        title = "Min robust invariant set"
        legend = X_TO_STRING[self.x_ids]
        plot_polyhedron_2D(Omega, title, legend)
        return Omega

    def plot_max_invariant_set(self) -> None:
        P = self.O_inf
        legend = X_TO_STRING[self.x_ids]
        if P.dim == 3:
            title = "Max invariant set projection"
            plot_polyhedron_2D(P.projection(dims=(0, 1)), title, legend[0:2])
            plot_polyhedron_2D(P.projection(dims=(1, 2)), title, legend[1:3])
        elif P.dim == 2:
            title = "Max invariant set"
            plot_polyhedron_2D(P.projection(dims=(0, 1)), title, legend)
        elif P.dim == 1:
            print_constraints(P, legend)


def plot_polyhedron_2D(P: Polyhedron, title, legend):
    ax = plt.figure().gca()
    P.plot(ax)
    ax.set_title(title)
    ax.set_xlabel(legend[0])
    ax.set_ylabel(legend[1])
    plt.show()


def print_constraints(P: Polyhedron, legend):
    for a, b in zip(P.A, P.b):
        expr = " + ".join(f"{ai:.3f}{name}" for ai, name in zip(a, legend))
        print(f"{expr} <= {b:.3f}")
