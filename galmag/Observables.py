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

Contains a class for computing
"""
from B_generators.B_generator import B_generator
import electron_profiles as prof
import numpy as np

class Observables(B_generator):
    def __init__(self, B_field, default_parameters={},
                 dtype=np.float64, direction=None, **kwargs):


        if B_field.grid_type != 'cartesian':
            raise NotImplementedError, 'At the moment, only cartesian grids are supported.'

        self.B_field = B_field
        resolution = B_field.resolution

        if direction == 'x' or direction == 'edge-on':
            self._integration_axis = 0
        elif direction == 'y':
            self._integration_axis = 1
        elif direction == 'z' or direction == 'face-on':
            self._integration_axis = 2
        else:
            raise NotImplementedError, 'Only "x", "y" and "z" directions are currently supported."'

        resolution[self._integration_axis] = 1

        super(Observables, self).__init__(box=B_field.box,
                                          resolution=resolution,
                                          default_parameters=default_parameters,
                                          dtype=dtype)

        # Reads in the parameters
        self.parameters = self._parse_parameters(kwargs)
        # Place holders
        self._synchrotron_emissivity = None
        self._instrinsic_polarization = None
        self._psi = None

    @property
    def _builtin_parameter_defaults(self):
        builtin_defaults = {
            'obs_electron_density_function': prof.simple_ne, # n_e [cm^-3]
            'obs_cosmic_ray_function': prof.constant_ncr, # n_{cr} [cm^-3]
            'obs_wavelength_cm': 1.0, # cm
            'obs_gamma': 1.0, # cm
            }
        return builtin_defaults

    def get_B_field(self):
        return self.B_field

    @property
    def synchrotron_emissivity(self):
        """
        Returns the synchrotron emissivity, computing it if necessary.
        """
        if self._synchrotron_emissivity is None:
            lamb = self.parameters['obs_wavelength_cm']
            gamma = self.parameters['obs_gamma']
            self._synchrotron_emissivity = \
              self._compute_synchrotron_emissivity(lamb, gamma)
        return self._synchrotron_emissivity


    def _compute_synchrotron_emissivity(self, lamb, gamma):
        """
        Input: B, ecr, lamb
        Output: synchrotron emissivity (along z)
        """

        # Bperp^2 = Bx^2 + By^2
        if self._integration_axis == 0:
            Bperp2 = self.B_field.y**2 + self.B_field.z**2
        elif self._integration_axis == 1:
            Bperp2 = self.B_field.x**2 + self.B_field.z**2
        elif self._integration_axis == 2:
            Bperp2 = self.B_field.x**2 + self.B_field.y**2

        ncr = self.parameters['obs_cosmic_ray_function'](
                                                 self.B_field.grid.r_spherical,
                                                 self.B_field.grid.theta,
                                                 self.B_field.grid.phi)

        return ncr * Bperp2**((gamma+1)/4) * lamb**((gamma-1)/2)

    @property
    def instrinsic_polarization():
        """
        Computes the (intrinsic) degree of polarization, p0
        """
        if self._instrinsic_polarization is None:
            gamma = self.parameters['obs_gamma']
            self._instrinsic_polarization = (gamma+1.0)/(gamma+7./3.)
        return self._instrinsic_polarization


    @property
    def intrinsic_polarization_angle():
        if self._instrinsic_polarization_angle is None:

            psi0 = np.pi/2.0 + np.arctan2(By[:,:,i],Bx[:,:,i])
            self._instrinsic_polarization_angle = psi0
        return self._instrinsic_polarization_angle

