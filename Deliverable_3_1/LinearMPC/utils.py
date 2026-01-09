import numpy as np
import matplotlib.pyplot as plt

WX, WY, WZ, ALPHA, BETA, GAMA, VX, VY, VZ, X, Y, Z = range(12)
DR, DP, P_AVG, P_DIFF = range(4)


def plot_trajectory(t_cl, x_cl, u_cl, t_ol, x_ol, u_ol, mpc):
    legend = X_TO_STRING[mpc.x_ids]
    plt.figure()
    plt.plot(t_cl, np.transpose(x_cl[mpc.x_ids]))
    plt.title("Closed-loop states trajectory")
    plt.legend(legend)
    plt.xlabel("time [s]")
    plt.ylabel("state value")
    plt.show()
    plt.figure()
    plt.plot(t_ol[:, 0], np.transpose(x_ol[mpc.x_ids, :, 0]))
    plt.title("Open-loop states trajectory")
    plt.legend(legend)
    plt.xlabel("time [s]")
    plt.ylabel("state value")
    plt.show()
    legend = U_TO_STRING[mpc.u_ids]
    plt.figure()
    plt.plot(t_cl[:-1], np.transpose(u_cl[mpc.u_ids]))
    plt.title("Closed-loop input trajectory")
    plt.legend(legend)
    plt.xlabel("time [s]")
    plt.ylabel("input value")
    plt.show()
    plt.figure()
    plt.plot(t_ol[:-1, 0], np.transpose(u_ol[mpc.u_ids, :, 0]))
    plt.title("Open-loop input trajectory")
    plt.legend(legend)
    plt.xlabel("time [s]")
    plt.ylabel("input value")
    plt.show()


X_TO_STRING = np.array(
    [
        "WX",
        "WY",
        "WZ",
        "ALPHA",
        "BETA",
        "GAMA",
        "VX",
        "VY",
        "VZ",
        "X",
        "Y",
        "Z",
        "DR",
        "DP",
        "P_AVG",
        "P_DIFF",
    ]
)
U_TO_STRING = np.array(
    [
        "DR",
        "DP",
        "P_AVG",
        "P_DIFF",
    ]
)

LIMIT = 1e50

# States constraints [WX, WY, WZ, ALPHA, BETA, GAMA, VX, VY, VZ, X, Y, Z]
LB_X = np.array(
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
)
UB_X = np.array(
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
)

# Inputs constraints [DR, DP, P_AVG, P_DIFF]
LB_U = np.array([-np.deg2rad(15), -np.deg2rad(15), 40, -20.0])
UB_U = np.array([np.deg2rad(15), np.deg2rad(15), 80, 20.0])
