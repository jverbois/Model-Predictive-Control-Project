import cvxpy as cp
import numpy as np
from control import dlqr
from mpt4py import Polyhedron
from scipy.signal import cont2discrete


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

        Q = 0.01 * np.eye(self.nx)
        R = 0.01 * np.eye(self.nu)
        K, Qf, _ = dlqr(self.A, self.B, Q, R)
        K = -K
        # Define variables
        nx, nu, N = self.nx, self.nu, self.N
        x_var = cp.Variable((nx, N + 1), name="x")
        u_var = cp.Variable((nu, N), name="u")
        x0_var = cp.Parameter((nx,), name="x0")

        # Costs
        cost = 0
        for i in range(N):
            cost += cp.quad_form(x_var[:, i], Q)
            cost += cp.quad_form(u_var[:, i], R)

        # Terminal cost
        cost += cp.quad_form(x_var[:, N], Qf)

        constraints = []

        # Initial condition
        constraints.append(x_var[:, 0] == x0_var)

        # System dynamics
        constraints.append(x_var[:, 1:] == self.A @ x_var[:, :-1] + self.B @ u_var)

        # Constraints
        if 2 in self.u_ids:  # P_avg
            # u in U = { u | Mu <= m }
            M = np.zeros((2, self.nu))
            M[0, 0] = -1
            M[1, 0] = 1
            m = np.array([-40, 80])
            U = Polyhedron.from_Hrep(M, m)
            KU = Polyhedron.from_Hrep(U.A @ K, U.b)
            O_inf = max_invariant_set(self.A + self.B @ K, KU)
            constraints.append(O_inf.A @ x_var[:, -1] <= O_inf.b.reshape(-1, 1))

        if 3 in self.x_ids or 4 in self.x_ids:  # alpha or beta
            # x in X = { x | Fx <= f }
            F = np.zeros((1, self.nx))
            F[0, 1] = 1
            f = np.array([0.1745])
            X = Polyhedron.from_Hrep(F, f)
            O_inf = max_invariant_set(self.A + self.B @ K, X)
            constraints.append(O_inf.A @ x_var[:, -1] <= O_inf.b.reshape(-1, 1))

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

        x_traj, u_traj = self.open_loop_prediction(x0)
        u0 = u_traj[0]

        # YOUR CODE HERE
        #################################################

        return u0, x_traj, u_traj.T

    def bellman(self):
        Q = 0.001 * np.eye(self.nx)
        R = 0.001 * np.eye(self.nu)
        # control gain matrices
        Klist = []

        # Bellman/Riccati recursion
        H = Q
        for i in range(self.N - 1, -1, -1):
            # K = -(R + B.T @ H @ B) @ np.linalg.inv(B.T @ H @ A)
            K = -np.linalg.solve(R + self.B.T @ H @ self.B, self.B.T @ H @ self.A)
            A_cl = self.A + self.B @ K
            H = Q + K.T @ R @ K + A_cl.T @ H @ A_cl

            Klist.append(K)

        Klist = Klist[::-1]
        return Klist

    def open_loop_prediction(self, x0) -> tuple[np.ndarray, np.ndarray]:
        Klist = self.bellman()
        x = [x0]
        u = []
        for i in range(self.N):
            u.append(Klist[i] @ x[-1])
            x.append(self.A @ x[-1] + self.B @ u[i])
        x = np.column_stack(x)
        return np.array(x), np.array(u)


def max_invariant_set(A_cl, X: Polyhedron, max_iter=30) -> Polyhedron:
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
        print("Maximum invariant set successfully computed after {itr} iterations.")
    return O
