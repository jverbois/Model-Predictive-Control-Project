import cvxpy as cp
import numpy as np
from control import dlqr
from mpt4py import Polyhedron
from scipy.signal import cont2discrete
import matplotlib.pyplot as plt

LIMIT = 1e50


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

        # System definition
        A_red = A[np.ix_(self.x_ids, self.x_ids)]
        B_red = B[np.ix_(self.x_ids, self.u_ids)]

        self.A, self.B = self._discretize(A_red, B_red, Ts)

        self.xs = xs[self.x_ids]
        self.us = us[self.u_ids]

        self._setup_controller()

    def _setup_controller(self) -> None:
        #################################################
        # YOUR CODE HERE

        # Cost matrices
        Q = 0.1 * np.eye(self.nx)
        R = 0.01 * np.eye(self.nu)
        K, Qf, _ = dlqr(self.A, self.B, Q, R)
        K = -K  # Ensure K is state feedback

        # Variables and parameters
        nx, nu, N = self.nx, self.nu, self.N
        x_var = cp.Variable((nx, N + 1), name="x")
        u_var = cp.Variable((nu, N), name="u")
        x0_var = cp.Parameter((nx,), name="x0")

        # Stage cost
        cost = 0
        for i in range(N):
            cost += cp.quad_form(x_var[:, i] - self.xs, Q)
            cost += cp.quad_form(u_var[:, i] - self.us, R)
        cost += cp.quad_form(x_var[:, N] - self.xs, Qf)  # Terminal cost

        constraints = []
        # Initial condition
        constraints.append(x_var[:, 0] == x0_var)

        # System dynamics
        constraints.append(x_var[:, 1:] == self.A @ x_var[:, :-1] + self.B @ u_var)

        # Inputs constraints [dR, dP, Pavg, Pdiff]
        lb_u = np.array([-np.deg2rad(15), -np.deg2rad(15), 40.0, -20.0])[self.u_ids]
        ub_u = np.array([np.deg2rad(15), np.deg2rad(15), 80.0, 20.0])[self.u_ids]

        # States constraints [wx, wy, wz, alpha, beta, gamma, vx, vy, vz, x, y, z]
        lb_x = np.array(
            [
                -LIMIT,
                -LIMIT,
                -LIMIT,
                -np.deg2rad(10),
                -np.deg2rad(10),
                -LIMIT,
                -LIMIT,
                -LIMIT,
                -LIMIT,
                -LIMIT,
                -LIMIT,
                0.0,
            ]
        )[self.x_ids]
        ub_x = np.array(
            [
                LIMIT,
                LIMIT,
                LIMIT,
                np.deg2rad(10),
                np.deg2rad(10),
                LIMIT,
                LIMIT,
                LIMIT,
                LIMIT,
                LIMIT,
                LIMIT,
                LIMIT,
            ]
        )[self.x_ids]

        # U = { u | Mu <= m }
        M = np.vstack((-np.eye(nu), np.eye(nu)))
        m = np.hstack((-lb_u, ub_u))
        U = Polyhedron.from_Hrep(M, m)  # -u <= -lb_u, u <= ub_u
        # Map to state constraints via feedback K
        KU = Polyhedron.from_Hrep(U.A @ K, U.b)

        # X = { x | Fx <= f }
        F = np.vstack((-np.eye(nx), np.eye(nx)))
        f = np.hstack((-lb_x, ub_x))
        X = Polyhedron.from_Hrep(F, f)  # -x <= -lb_x, x <= ub_x

        # Terminal invariant set
        O_inf = max_invariant_set(self.A + self.B @ K, X.intersect(KU))
        constraints.append(O_inf.A @ x_var[:, -1] <= O_inf.b)

        # Input constraints for P_avg (index 2)
        # if 2 in self.u_ids:
        #     # Define input bounds
        #     M = np.zeros((2, nu))
        #     M[0, 0] = -1
        #     M[1, 0] = 1
        #     m = np.array([-40, 80])
        #     U = Polyhedron.from_Hrep(M, m)
        #     KU = Polyhedron.from_Hrep(U.A @ K, U.b)
        #     O_inf = max_invariant_set(self.A + self.B @ K, KU)
        #     constraints.append(O_inf.A @ x_var[:, -1] <= O_inf.b)
        # if 3 in self.x_ids or 4 in self.x_ids:  # alpha or beta
        #     # x in X = { x | Fx <= f }
        #     F = np.zeros((2, self.nx))
        #     F[0, 1] = -1
        #     F[1, 1] = 1
        #     f = np.array([np.deg2rad(10), np.deg2rad(10)])
        #     X = Polyhedron.from_Hrep(F, f)
        #     O_inf = max_invariant_set(self.A + self.B @ K, X)
        #     constraints.append(O_inf.A @ x_var[:, -1] <= O_inf.b)

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

        N = self.N

        x_traj = np.zeros((self.nx, N + 1))
        u_traj = np.zeros((self.nu, N))

        u_var = self.ocp.variables()[1]
        x0_var = self.ocp.parameters()[0]
        # Closed-loop simulation
        x_traj[:, 0] = x0
        xk = x0
        for k in range(N):
            x0_var.value = xk
            self.ocp.solve(solver=cp.PIQP, verbose=False)
            # assert self.ocp.status == cp.OPTIMAL
            uk = u_var.value[:, 0]
            xk = self.A @ xk + self.B @ uk
            x_traj[:, k + 1] = xk.flatten()
            u_traj[:, k] = uk.flatten()

        # YOUR CODE HERE
        #################################################

        return u_traj[:, 0], x_traj, u_traj


def max_invariant_set(A_cl, X: Polyhedron, max_iter=100) -> Polyhedron:
    """
    Compute invariant set for an autonomous linear time invariant system x^+ = A_cl x
    """
    O = X
    itr = 1
    converged = False
    print("Computing maximum invariant set ...")
    while itr < max_iter:
        Oprev = O
        F, f = O.A, O.b
        # Compute the pre-set
        O = Polyhedron.from_Hrep(
            np.vstack((F, F @ A_cl)), np.vstack((f, f)).reshape((-1,))
        )
        O.minHrep(True)
        if O == Oprev:
            converged = True
            break
        print(f"Iteration {itr}... not yet converged")
        itr += 1

    if converged:
        print(f"Maximum invariant set successfully computed after {itr} iterations.")

    return O
