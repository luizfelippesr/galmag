# Copyright (C) 2017  Luiz Felippe S. Rodrigues
#
# This file is part of GalMag.
#
# GalMag is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# GalMag is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GalMag.  If not, see <http://www.gnu.org/licenses/>.
#
"""
GalMag

Contains the definitions of the halo rotation curve and alpha profile.
"""
import numpy as np

def simple_ne(rho, theta, phi, R_h=1.0, V0=1.0, s0=0.5):
    """
    Simple electron density profile
    """
    raise NotImplementedError


def constant_ne(rho, theta, phi, ne0=1.0):
    """
    Dummy for constant cosmic ray electron density
    """
    ne = rho.copy_empty()
    ne[:,:,:] = ne0
    return ne


def constant_ncr(rho, theta, phi, ncr0=1.0):
    """
    Dummy for constant cosmic ray electron density
    """
    return np.ones_like(rho)*ncr0



