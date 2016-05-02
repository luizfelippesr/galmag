# -*- coding: utf-8 -*-

import numpy as np


def simple_radial_shear(r_cylindrical, V0, s0, Rmax=None):
    """ Shear rate compatible with simple_V """

    r = r_cylindrical
    rs0 = r_cylindrical/s0

    # Traps negligible radii
    if isinstance(r, np.ndarray):
        r = np.copy(r)
        r[rs0 < 1e-10] = 1e-10
    elif rs0 < 1e-10:
        r = 1e-10

    V = V0 * (1.0 - np.exp(-rs0))
    S = V0/s0 * np.exp(-rs0) - V/r
    # The following renormalisation makes S(r=1, 1,1) = -1
    S /= 0.26424111765711533

    return S
