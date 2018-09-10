# Copyright (C) 2017, 2018  Luiz Felippe S. Rodrigues <luiz.rodrigues@ncl.ac.uk>
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

def simple_V(rho, theta, phi, r_h=1.0, Vh=220, fraction=3./15., normalize=True,
             fraction_z=None, legacy=False):
    r"""
    Simple form of the rotation curve to be used for the halo

    .. math::
        V(r,\theta,\phi) \propto [1-\exp(-r \sin(\theta) / s_v) ]

    Note
    ----
        This simple form has no z dependence

    Parameters
    ----------
    rho : array
        Spherical radial coordinate, :math:`r`
    theta : array
        Polar coordinate, :math:`\theta`
    phi : array
        Azimuthal coordinate, :math:`\phi`
    fraction :
        fraction of the halo radius corresponding to the turnover
        of the rotation curve.
    r_h :
        halo radius in the same units as rho. Default: 1.0
    Vh :
        Value of the rotation curve at rho=r_h. Default: 220 km/s
    normalize : bool, optional
        if True, the rotation curve will be normalized to one at rho=r_h

    Returns
    -------
    list
        List containing three `numpy` arrays corresponding to:
        :math:`V_r`, :math:`V_\theta`, :math:`V_\phi`
    """
    Vr, Vt = [np.zeros_like(rho) for i in range(2)]


    Vp = (1.0-np.exp(-np.abs(rho*np.sin(theta))/(fraction*r_h)))
    if not legacy:
        Vp /= (1.0-np.exp(-1./fraction))

    if not normalize:
        Vp *= Vh

    return Vr, Vt, Vp


def simple_V_legacy(rho, theta, phi, r_h=1.0, Vh=220, fraction=0.5,
                    fraction_z=None, normalize=True):
    """
    Rotation curve employed in version 0.1 and in the MMath Final Report of
    James Hollins. Same as simple_V but with a slight change in the way it
    is normalized.
    """
    return simple_V(rho, theta, phi, r_h, Vh, fraction, normalize, legacy=True)


def simple_V_exp(rho, theta, phi, r_h=1.0, Vh=220, fraction=3./15.,
                    fraction_z=11./15., normalize=True,
                    legacy=False):
    r"""
    Variation on simple_V which decays exponentially with z

    .. math::
        V(r,\theta,\phi) \propto (1-\exp(-r \sin(\theta) / s_v)) \exp(-r \cos(\theta)/z_v)

    Parameters
    ----------
    rho : array
        Spherical radial coordinate, :math:`r`
    theta : array
        Polar coordinate, :math:`\theta`
    phi : array
        Azimuthal coordinate, :math:`\phi`
    fraction :
        fraction of the halo radius corresponding to the turnover of the
        rotation curve.
    fraction_z :
        fraction of the halo radius corresponding to the characteristic
        vertical decay length of the rotation
    r_h :
        halo radius in the same units as rho. Default: 1.0
    Vh :
        Value of the rotation curve at rho=r_h. Default: 220 km/s
    normalize : bool, optional
        if True, the rotation curve will be normalized to one at rho=r_h

    Returns
    -------
    list
        List containing three `numpy` arrays corresponding to:
        :math:`V_r`, :math:`V_\theta`, :math:`V_\phi`
    """
    Vr, Vt, Vp = simple_V(rho, theta, phi, r_h, Vh, fraction, normalize)

    z = np.abs(rho/r_h * np.cos(theta))
    decay_factor = np.exp(-z/fraction_z)

    return Vr, Vt, Vp*decay_factor


def simple_V_linear(rho, theta, phi, r_h=1.0, Vh=220, fraction=3./15.,
                    fraction_z=11./15, normalize=True,
                    legacy=False):
    r"""
    Variation on simple_V which decays linearly with z, reaching 0 at
    z=(halo radius), and V_h at z=0

    .. math::
        V(r,\theta,\phi) \propto [1-\exp(-r \sin(\theta) / s_v)] (1-z/z_v)

    Parameters
    ----------
    rho : array
        Spherical radial coordinate, :math:`r`
    theta : array
        Polar coordinate, :math:`\theta`
    phi : array
        Azimuthal coordinate, :math:`\phi`
    fraction :
        fraction of the halo radius corresponding to the turnover of the
        rotation curve. (s_v = fraction*r_h)
    fraction_z :
        fraction of the halo radius controling the "lag" of the rotation curve.
        (z_v = fraction_z*r_h)
    r_h :
        halo radius in the same units as rho. Default: 1.0
    Vh :
        Value of the rotation curve at rho=r_h. Default: 220 km/s
    normalize : bool, optional
        if True, the rotation curve will be normalized to one at rho=r_h
    Returns:
        List containing three `numpy` arrays corresponding to:
        :math:`V_r`, :math:`V_\theta`, :math:`V_\phi`
    """
    Vr, Vt, Vp = simple_V(rho, theta, phi, r_h, Vh, fraction, normalize)

    z = np.abs(rho/r_h * np.cos(theta)) # Dimensionless z
    decay_factor = (1-z/fraction_z)
    Vp[decay_factor<0.] = 0.

    return Vr, Vt, Vp*decay_factor


def simple_alpha(rho, theta, phi, alpha0=1.0):
    r"""
    Simple profile for alpha

    .. math::
        \alpha(\mathbf{r}) = \alpha_0\cos(\theta)

    Parameters
    ----------
    rho : array
        Spherical radial coordinate, :math:`r`
    theta : array
        Polar coordinate, :math:`\theta`
    phi : array
        Azimuthal coordinate, :math:`\phi`
    alpha0 : float, optional
        Normalization. Default: 1.0

    """

    alpha = np.cos(theta)
    alpha[rho>1.] = 0.

    return alpha*alpha0



