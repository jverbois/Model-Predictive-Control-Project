import numpy as np

from .MPCControl_base import MPCControl_base
from .utils import VZ, Z
from .utils import P_AVG
from .utils import LB_U, UB_U, LB_X, UB_X


class MPCControl_z(MPCControl_base):
    x_ids: np.ndarray = np.array([VZ, Z])
    u_ids: np.ndarray = np.array([P_AVG])

    # only useful for part 5 of the project
    d_estimate: np.ndarray
    d_gain: float

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
        idx = self.x_ids == VZ
        self.Q[idx, idx] *= 100
        idx = self.x_ids == Z
        self.Q[idx, idx] *= 100
        
        self.lb_x = LB_X[self.x_ids]
        self.ub_x = UB_X[self.x_ids]

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

    def setup_estimator(self):
        # FOR PART 5 OF THE PROJECT
        ##################################################
        # YOUR CODE HERE

        self.d_estimate = ...
        self.d_gain = ...

        # YOUR CODE HERE
        ##################################################

    def update_estimator(self, x_data: np.ndarray, u_data: np.ndarray) -> None:
        # FOR PART 5 OF THE PROJECT
        ##################################################
        # YOUR CODE HERE
        self.d_estimate = ...
        # YOUR CODE HERE
        ##################################################
