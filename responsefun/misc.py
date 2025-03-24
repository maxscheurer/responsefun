import numpy as np
from scipy.constants import physical_constants


# Levi-Civita tensor
epsilon = np.zeros((3, 3, 3))
epsilon[0, 1, 2] = epsilon[1, 2, 0] = epsilon[2, 0, 1] = 1
epsilon[2, 1, 0] = epsilon[0, 2, 1] = epsilon[1, 0, 2] = -1


def ev2au(ev):
    return ev / physical_constants["hartree-electron volt relationship"][0]
