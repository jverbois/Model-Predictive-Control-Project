import numpy as np

from .MPCControl_base import MPCControl_base

from .utils import WZ, GAMA
from .utils import P_DIFF
from .utils import LB_U, UB_U


class MPCControl_roll(MPCControl_base):
    x_ids: np.ndarray = np.array([WZ, GAMA])
    u_ids: np.ndarray = np.array([P_DIFF])

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        xs: np.ndarray,
        us: np.ndarray,
        Ts: float,
        H: float,
    ) -> None:
        super().__init__(A, B, self.x_ids, self.u_ids, xs, us, Ts, H)

    def _setup_controller(self) -> None:
        #################################################
        # YOUR CODE HERE
        idx = self.x_ids == GAMA
        self.Q[idx, idx] *= 250

        self.lb_u = LB_U[self.u_ids]
        self.ub_u = UB_U[self.u_ids]

        super()._setup_controller()

        # YOUR CODE HERE
        #################################################

    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        #################################################
        # YOUR CODE HERE

        u0, x_traj, u_traj = super().get_u(x0, x_target, u_target)

        # YOUR CODE HERE
        #################################################

        return u0, x_traj, u_traj
