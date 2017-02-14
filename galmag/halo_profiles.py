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
import numpy as N

def simple_V(rho, theta, phi, R_h=1.0, V0=1.0, s0=0.5):
    """
    Simple form of the rotation curve to be used for the halo
    NB This simple form has no z dependence
    V(r,theta,phi) = V0 (1-exp(-r sin(theta) / s0))
    Input: rho, theta, phi -> NxNxN grid in spherical coordinates
           s0 -> s0 parameter Default: 0.5
           R_h -> unit of rho in kpc [e.g. R_d=(halo radius in kpc)
                  for r=0..1 within the halo]. Default: 1.0
       V0 -> Normalization of the rotation curve. Default: 1.0
    Ouput: V -> rotation curve in the units of V0
    """
    Vr, Vt = [N.zeros_like(rho) for i in range(2)]

    Vp = V0 * (1.0 - N.exp(-rho*N.sin(theta)/s0/R_h))

    return Vr, Vt, Vp


def simple_alpha(rho, theta, phi, alpha0=1.0):
    """ Simple profile for alpha"""

    alpha = N.cos(theta)
    alpha[rho>1.] = 0.

    return alpha*alpha0



