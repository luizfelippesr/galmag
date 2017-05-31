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

def simple_V(rho, theta, phi, r_h=1.0, Vh=220, fraction=3./15., normalize=True):
    """
    Simple form of the rotation curve to be used for the halo
    NB This simple form has no z dependence
    V(r,theta,phi) = V0 (1-exp(-r sin(theta) / s0))
    Input: rho, theta, phi -> NxNxN grid in spherical coordinates
           fraction -> fraction of the halo radius corresponding to the turnover
                       of the rotation curve.
           r_h -> halo radius in the same units as rho. Default: 1.0
           Vh -> Normalization of the rotation curve. Default: 220
           normalize, optional -> if True, the rotation curve will be normalized
                                  to one at rho=r_h
    Ouput: V -> rotation curve
    """
    Vr, Vt = [np.zeros_like(rho) for i in range(2)]


    Vp = (1.0-np.exp(-rho*np.sin(theta)/(fraction*r_h)))
    Vp /= (1.0-np.exp(-1./fraction))

    if not normalize:
        Vp *= Vh

    return Vr, Vt, Vp


def simple_alpha(rho, theta, phi, alpha0=1.0):
    """ Simple profile for alpha"""

    alpha = np.cos(theta)
    alpha[rho>1.] = 0.

    return alpha*alpha0



