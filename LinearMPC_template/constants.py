import numpy as np

WX, WY, WZ, ALPHA, BETA, GAMA, VX, VY, VZ, X, Y, Z = range(12)
DR, DP, P_AVG, P_DIFF = range(4)

TO_STRING = np.array(
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
