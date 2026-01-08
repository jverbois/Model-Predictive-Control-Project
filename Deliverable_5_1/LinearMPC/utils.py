import numpy as np
import matplotlib.pyplot as plt

NB_X = 12
NB_U = 4
NB_D = 1
WX, WY, WZ, ALPHA, BETA, GAMA, VX, VY, VZ, X, Y, Z = range(NB_X)
DR, DP, P_AVG, P_DIFF = range(NB_U)

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

BD = np.zeros((NB_X, NB_D))
BD[VZ] = 1


def plot_trajectory(t_cl, x_cl, u_cl, t_ol, x_ol, u_ol, mpc=None):

    if mpc is None:
        x_ids = np.array(range(len(X_TO_STRING)))[
            np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        ]
        u_ids = np.array(range(len(U_TO_STRING)))[np.array([0, 1, 2, 3])]
        print(u_ids)
    else:
        x_ids = mpc.x_ids
        u_ids = mpc.u_ids
    legend = X_TO_STRING[x_ids]
    plt.figure()
    plt.plot(t_cl, np.transpose(x_cl[x_ids]))
    plt.title("Closed-loop states trajectory")
    plt.legend(legend)
    plt.xlabel("time [s]")
    plt.ylabel("state value")
    plt.show()
    plt.figure()
    plt.plot(t_ol[:, 0], np.transpose(x_ol[x_ids, :, 0]))
    plt.title("Open-loop states trajectory")
    plt.legend(legend)
    plt.xlabel("time [s]")
    plt.ylabel("state value")
    plt.show()
    legend = U_TO_STRING[u_ids]
    plt.figure()
    plt.plot(t_cl[:-1], np.transpose(u_cl[u_ids]))
    plt.title("Closed-loop input trajectory")
    plt.legend(legend)
    plt.xlabel("time [s]")
    plt.ylabel("input value")
    plt.show()
    plt.figure()
    plt.plot(t_ol[:-1, 0], np.transpose(u_ol[u_ids, :, 0]))
    plt.title("Open-loop input trajectory")
    plt.legend(legend)
    plt.xlabel("time [s]")
    plt.ylabel("input value")
    plt.show()
