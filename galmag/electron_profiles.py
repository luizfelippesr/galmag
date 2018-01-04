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
Distributions of electron densities and cosmic ray densities
"""
import numpy as np

def simple_ne(rho, theta, phi, ne0=1.0, Rs=2.6, R_d=1.0, Rsun=8.5, **kwargs):
    r"""
    Simple electron density profile

    Comprises a exponential flared disc.

    .. math::
        n = n_0 \exp\left[-\frac{r \sin(\theta)-s_\odot}{s_s}\right] \exp\left[\frac{-r \cos(\theta)}{h(s)}\right]

    where :math:`h(s)` is given by
    :func:`galmag.disk_profiles.exponential_scale_height`

    Parameters
    ----------
    rho : array
        Spherical radial coordinate, :math:`r`
    theta : array
        Polar coordinate, :math:`\theta`
    phi : array
        Azimuthal coordinate, :math:`\phi`
    ne0 : float
        Number density at `Rsun`. Default: 1.0
    R_d : float
        unit of `rho` in kpc [e.g. R_d=(disk radius in kpc)
        for r=0..1 within the disk]. Default: 1.0
    Rs : float
        Scale radius, :math:`s_s`, of the disc in kpc. Default: 2.6 kpc
    Rsun : float
        Reference radius, :math:`s_\odot`, in kpc. Default: 8.5 kpc
    """
    from galmag.disk_profiles import exponential_scale_height
    # Cylindrical radius
    R = rho*np.sin(theta)*R_d
    z = rho*np.cos(theta)*R_d
    h = exponential_scale_height(R, R_d=1.0, Rsun=8.5, **kwargs)

    ne = ne0 * np.exp(-(R-Rsun)/Rs) * np.exp(-np.abs(z)/h)

    return ne


def constant_ne(rho, theta, phi, ne0=1.0):
    """
    Dummy for constant cosmic ray electron density
    """
    ne = np.empty_like(rho)
    ne[:,:,:] = ne0
    return ne


def constant_ncr(rho, theta, phi, ncr0=1.0):
    """
    Dummy for constant cosmic ray electron density
    """
    ncr = np.empty_like(rho)
    ncr[:,:,:] = ncr0
    return ncr



