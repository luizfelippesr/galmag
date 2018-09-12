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
Contains the definitions of the disk rotation curve, radial shear,
alpha profile and disk scale height.
"""
import numpy as np
from galmag.util import distribute_function

def solid_body_rotation_curve(R, R_d=1.0, Rsun=8.5, V0=220, normalize=True):
    """
    Solid body rotation curve for testing.

    .. math::
        V(R) = R

    Parameters
    ----------
    R : float or array
        cylindrical radius
    """
    V = R * R_d / Rsun
    if not normalize:
        V *= V0
    return V


def constant_shear_rate(R, R_d=1.0, Rsun=8.5, S0=25, normalize=True):
    """
    Constant shear for testing.

    .. math::
        V(R) = cte

    Parameters
    ----------
    R : float or array
        cylindrical radius
    """
    S = np.ones_like(R)
    if not normalize:
        S *= S0
    return S


def simple_rotation_curve(R, R_d=1.0, Rsun=8.5, V0=220, normalize=True,
                          fraction=0.25/8.5):
    r"""
    Simple flat rotation curve

    .. math::
        V = V_0  \left[1-\exp(-R/(f R_\odot)\right]

    with fraction :math:`= f`, Rsun :math:`= R_\odot`, V0 :math:`= V_0`

    Parameters
    ----------
    R : float or array
        radial coordinate
    Rsun : float or array
        sun's radius in kpc. Default: 8.5 kpc
    R_d : float
        unit of R in kpc [e.g. R_d=(disk radius in kpc)
        for r=0..1 within the disk]. Default: 1.0
    V0 : float
        Circular velocity at infinity (i.e. at the flat part).
        Default: 220 (km/s)
    fraction : float
        Fraction of the solar radius at which the profile decays
        exponentially. Default: 0.03 (i.e. 250 pc for Rsun=8.5).
    normalize : bool
        If True , with result normalized to unit at solar radius,
        if False, the result will be in units of `V0`.

    Returns
    -------
    same as `R`
        rotation curve
    """
    V = V0*(1.0-np.exp(-R*R_d/(fraction*Rsun)))
    if normalize:
        Vsol = simple_rotation_curve(Rsun, R_d=1.0, Rsun=Rsun, V0=V0,
                                     normalize=False, fraction=fraction)
        V /= Vsol
    return V


def simple_shear_rate(R, R_d=1.0, Rsun=8.5, V0=220, normalize=True,
                      fraction=0.25/8.5):
    """
    A simple shear rate profile, compatible with the
    :func:`simple_rotation_curve` .


    Parameters
    ----------
    R : float or array
        radial coordinate
    Rsun : float or array
        sun's radius in kpc. Default: 8.5 kpc
    R_d : float
        unit of R in kpc [e.g. R_d=(disk radius in kpc)
        for r=0..1 within the disk]. Default: 1.0
    V0 : float
        Circular velocity at infinity (i.e. at the flat part).
        Default: 220 (km/s)
    fraction : float
        Fraction of the solar radius at which the profile decays
        exponentially. Default: 0.03 (i.e. 250 pc for Rsun=8.5).
    normalize : bool
        If True , with result normalized to unit at solar radius,
        if False, the result will be in km/s for R and Rsun in kpc

    Returns
    -------
    same as `R`
        shear rate
    """
    # Computes the shear rate ( rdOmega/dr = dV/dr - V/r )
    x = R*R_d/(fraction*Rsun)
    dVdr = V0/(fraction*Rsun)*np.exp(-x)
    Omega = V0*(1.0-np.exp(-x))/(R*R_d)
    S = dVdr - Omega
    if normalize:
        Ssol = simple_shear_rate(Rsun, R_d=1.0, Rsun=Rsun, V0=V0,
                                 normalize=False, fraction=fraction)
        S /= Ssol
    return S

# Coefficients used in the polynomial fit of Clemens (1985)
coef_Clemens = {
      'A': [-17731.0,54904.0, -68287.3, 43980.1, -15809.8, 3069.81, 0.0],
      'B': [-2.110625, 25.073006, -110.73531, 231.87099, -248.1467, 325.0912],
      'C': [0.00129348, -0.08050808, 2.0697271, -28.4080026, 224.562732,
                          -1024.068760,2507.60391, -2342.6564],
      'D': [234.88] }
# Ranges used in the polynomial fit of Clemens (1985)
ranges_Clemens = {
      'A': [0,0.09],
      'B': [0.09,0.45],
      'C': [0.45,1.60],
      'D': [1.60,1000]
      }

def Clemens_Milky_Way_rotation_curve(R, R_d=1.0, Rsun=8.5, normalize=True):
    """
    Rotation curve of the Milky Way obtained by `Clemens (1985)
    <http://adsabs.harvard.edu/abs/1985ApJ...295..422C>`_

    Parameters
    ----------
    R : float or array
        radial coordinate
    Rsun : float or array
        sun's radius in kpc. Default: 8.5 kpc
    R_d : float
        unit of R in kpc [e.g. R_d=(disk radius in kpc)
        for r=0..1 within the disk]. Default: 1.0
    normalize : bool
        If True , with result normalized to unit at solar radius,
        if False, the result will be in km/s for R and Rsun in kpc

    Returns
    -------
    same as `R`
        rotation curve
    """

    # If the function was called for a scalar
    if not hasattr(R, "__len__"):
        R = np.array([R,])
        scalar = True
    else:
        scalar = False
    V = R.copy()

    for x in coef_Clemens:
        # Construct polynomials
        pol_V = np.poly1d(coef_Clemens[x])
        # Reads the ranges
        min_r, max_r = ranges_Clemens[x]
        # Sets the index (selects the relevant range)
        idx  = R*R_d/Rsun >= min_r
        idx *= R*R_d/Rsun < max_r
        # Computes the shear rate ( rdOmega/dr = dV/dr - V/r )
        V[idx] = pol_V(R[idx]*R_d)

    if normalize:
        # Normalizes at solar radius
        Vsol = Clemens_Milky_Way_rotation_curve(Rsun, R_d=1.0, Rsun=Rsun,
                                                normalize=False)
        V /= Vsol

    if scalar:
        V = V[0]

    return V


def Clemens_Milky_Way_shear_rate(R, R_d=1.0, Rsun=8.5, normalize=True):
    """
    Shear rate of the Milky Way based on the rotation curve obtained by
    `Clemens (1985) <http://adsabs.harvard.edu/abs/1985ApJ...295..422C>`_

    Parameters
    ----------
    R : float or array
        radial coordinate
    Rsun : float or array
        sun's radius in kpc. Default: 8.5 kpc
    R_d : float
        unit of R in kpc [e.g. R_d=(disk radius in kpc)
        for r=0..1 within the disk]. Default: 1.0
    normalize : bool
        If True , with result normalized to unit at solar radius,
        if False, the result will be in km/s for R and Rsun in kpc

    Returns
    -------
    same as `R`
        shear rate profile, with
    """

    # If the function was called for a scalar
    if not hasattr(R, "__len__"):
        R = np.array([R,])
        scalar = True
    else:
        scalar = False
    S = R.copy()

    for x in coef_Clemens:
        # Construct polynomials
        pol_V = np.poly1d(coef_Clemens[x])
        dVdr = pol_V.deriv()

        # Reads the ranges
        min_r, max_r = ranges_Clemens[x]
        # Sets the index (selects the relevant range)
        idx  = R*R_d/Rsun >= min_r
        idx *= R*R_d/Rsun < max_r
        # Computes the shear rate ( rdOmega/dr = dV/dr - V/r )
        S[idx] = dVdr(R[idx]*R_d) - pol_V(R[idx]*R_d)/(R[idx]*R_d)

    if normalize:
        # Normalizes at solar radius
        Ssol = Clemens_Milky_Way_shear_rate(Rsun, R_d=1.0, Rsun=Rsun,
                                          normalize=False)
        S /= Ssol

    if scalar:
        S = S[0]

    return S


def constant_scale_height(R, h_d=1.0, R_d=1.0, Rsun=8.5):
    """ Constant scale height for testing."""
    return np.ones_like(R)*h_d


def exponential_scale_height(R, h_d=1.0, R_HI=5, R_d=1.0, Rsun=8.5):
    r"""
    Exponential disk scale-heigh profile profile

    .. math::
        h(R)=\exp\left(\frac{R-R_\odot}{R_{\rm sh}}\right)

    Parameters
    ----------
    R : float or array
        radial coordinate
    h_d : float
        normalization of the scaleheight. Default: 1.0
    R_d : float
        unit of R in kpc [e.g. R_d=(disk radius in kpc)
        for r=0..1 within the disk]. Default: 1.0
    R_HI : float
        Parameter :math:`R_{\rm sh}`, characterizes how "flared" the disc is.
    Rsun : float
        Sun's radius in kpc. Default: 8.5 kpc

    Returns
    ----------
    same as `R`
        scale height normalized to h_d at the solar radius
    """
    # Makes sure we are dealing with an array
    return h_d * np.exp((R*R_d - Rsun)/R_HI)


def Omega(rotation_curve, R, Rsun=8.5, R_d=1.0, normalize=True, **kwargs):
    """
    Simple wrapper to avoid making mistakes when using dimensionless
    (aka normalized) quantities.
    """

    V = rotation_curve(R, R_d=R_d, Rsun=Rsun, normalize=normalize, **kwargs)

    Om = V/(R*R_d)
    if normalize:
        Om = Om * Rsun
    return Om

def regularize(r, Om, S, r_reg, Om_reg, k=4):
    """
    Avoids unphysically large values of Omega and Shear near the origin
    applying an exponential cutoff on Omega to prevent it.

    .. math::
        \Omega(r) = \exp\left[-(r_\\xi/r)^k\\right]\left[ \\tilde\Omega(r)-\Omega_\\xi\\right] + \Omega_\\xi\,,

    and

    .. math::
        S(r) = e^{-(r_\\xi/r)^k}\left\{k\left(\\frac{r_\\xi}{r}\\right)^k\\left[\\tilde\Omega(r)
          -\Omega_\\xi \\right] +\\tilde S \\right\}


    Parameters
    ----------
    R : array
        radial coordinate
    Om : float
        Angular velocity profile
    r_reg : float
        regularization radius
    Om_reg : float
        Value of Omega to be used for math:`r \lesssim r_{reg`
    k : float
        How sharp is the cutoff. Default: 4

    """
    # Sets up exponential cutoff function
    f = lambda x: np.exp(-(x/r_reg)**-k)
    # Applies it in a d2o-compatible manner
    exp_cut = distribute_function(f, r)

    Om_new = exp_cut*(Om-Om_reg) + Om_reg

    S_new =  exp_cut*( k*(r_reg/r)**k*(Om-Om_reg) + S)

    return Om_new, S_new

