import numpy as np

from .MPCControl_base import MPCControl_base


class MPCControl_yvel(MPCControl_base):
    x_ids: np.ndarray = np.array([0, 3, 7])
    u_ids: np.ndarray = np.array([0])

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
