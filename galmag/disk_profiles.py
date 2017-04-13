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

Contains the definitions of the disk rotation curve, radial shear,
alpha profile and disk scale height.
"""
import numpy as np

def solid_body_rotation_curve(R, R_d=1.0, Rsun=8.5, V0=220, normalize=True):
    """ Solid body rotation curve for testing. V(R) = R """
    V = R * R_d / Rsun
    if not normalize:
        V *= V0
    return V

def constant_shear_rate(R, R_d=1.0, Rsun=8.5, S0=25, normalize=True):
    """ Constant shear for testing. V(R) = cte """
    S = np.ones_like(R)
    if not normalize:
        S *= S0
    return S


def simple_rotation_curve(R, R_d=1.0, Rsun=8.5, V0=220, normalize=False,
                          fraction=0.25/8.5):
    """
    Simple flat rotation curve
    V = V0 * (1-exp(-R/(fraction*R_sun))

    Input: R -> radial coordinate
           Rsun -> sun's radius in kpc. Default: 8.5 kpc
           R_d -> unit of R in kpc [e.g. R_d=(disk radius in kpc)
                      for r=0..1 within the disk]. Default: 1.0
           V0 -> Circular velocity at infinity (i.e. at the flat part).
              Default: 220 (km/s)
           fraction -> Fraction of the solar radius at which the profile decays
              exponentially. Default: 0.03 (i.e. 250 pc for Rsun=8.5).
    Ouput: V -> rotation curve, with:
           results normalized to unit at solar radius, if normalize==True
           results in km/s for R and Rsun in kpc, if normalize==False
    """
    V = V0*(1.0-np.exp(-R*R_d/(fraction*Rsun)))
    if normalize:
        V /= V0*(1.0-np.exp(-1./fraction))
    return V


def simple_shear_rate(R, R_d=1.0, Rsun=8.5, V0=220, normalize=True,
                      fraction=0.25/8.5):
    """
    A simple shear rate profile, compatible with the simple flat
    rotation curve.

    Input: R -> radial coordinate
           Rsun -> sun's radius in kpc. Default: 8.5 kpc
           R_d -> unit of R in kpc [e.g. R_d=(disk radius in kpc)
                      for r=0..1 within the disk]. Default: 1.0
           V0 -> Circular velocity at infinity (i.e. at the flat part).
              Default: 220 (km/s)
           fraction -> Fraction of the solar radius at which the profile decays
              exponentially. Default: 0.03 (i.e. 250 pc for Rsun=8.5).
    Ouput: V -> rotation curve, with:
           results normalized to unit at solar radius, if normalize==True
           results in km/s for R and Rsun in kpc, if normalize==False
    """
    # Computes the shear rate ( rdOmega/dr = dV/dr - V/r )
    x = R*R_d/(fraction*Rsun)
    dVdr = V0/(fraction*Rsun)*np.exp(-x)
    Omega = V0*(1.0-np.exp(-x))/R/R_d
    S = dVdr - Omega
    if normalize:
        dVdr_sun = V0/(fraction*Rsun)*np.exp(-1./fraction)
        Omega_sun = V0*(1.0-np.exp(-1./fraction))/Rsun
        S /= (dVdr_sun - Omega_sun)
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
    Rotation curve of the Milky Way obtained by Clemens (1985)
    Input: R -> radial coordinate
           Rsun -> sun's radius in kpc. Default: 8.5 kpc
           R_d -> unit of R in kpc [e.g. R_d=(disk radius in kpc)
                      for r=0..1 within the disk]. Default: 1.0
    Ouput: V -> rotation curve, with:
           results normalized to unit at solar radius, if normalize==True
           results in km/s for R and Rsun in kpc, if normalize==False
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
        Vsol = np.poly1d(coef_Clemens['C'])(Rsun)
        V = V/Vsol

    if scalar:
        V = V[0]

    return V


def Clemens_Milky_Way_shear_rate(R, R_d=1.0, Rsun=8.5, normalize=True):
    """
    Shear rate of the Milky Way based on the rotation curve
    obtained by Clemens (1985)
    Input: R -> radial coordinate
           R_d -> unit of R in kpc [e.g. R_d=(disk radius in kpc)
                  for r=0..1 within the disk]. Default: 1.0
           Rsun -> sun's radius in kpc. Default: 8.5 kpc
           Normalize -> Normalizes if True. Default: False
    Ouput: S -> shear rate profile curve, with:
                results normalized to unit at solar radius, if normalize==True
                results in km/s/kpc for R and Rsun in kpc, if normalize==False
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
        pol_V = np.poly1d(coef_Clemens['C'])
        dVdr = pol_V.deriv()
        S_sol = dVdr(Rsun) - pol_V(Rsun)/Rsun
        S = S/S_sol

    if scalar:
        S = S[0]

    return S


def constant_scale_height(R, h_d=1.0, R_d=1.0, Rsun=8.5):
    """ Constant scale height for testing."""
    return np.ones_like(R)*h_d


def exponential_scale_height(R, h_d=1.0, R_HI=5, R_d=1.0, Rsun=8.5):
    """
    Exponential disk scale-heigh profile profile
    Input: R -> radial coordinate
           R_d -> unit of R in kpc [e.g. R_d=(disk radius in kpc)
                  for r=0..1 within the disk]. Default: 1.0
           Rsun -> sun's radius in kpc. Default: 8.5 kpc
    Ouput: h -> scale height normalized to h_d at the solar radius
    """
    # Makes sure we are dealing with an array
    return h_d * np.exp((R*R_d - Rsun)/R_HI)

