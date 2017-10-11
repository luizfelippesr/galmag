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
import util

class Observables(B_generator):
    def __init__(self, B_field, default_parameters={},
                 dtype=np.float64, direction=None, **kwargs):


        if B_field.grid_type != 'cartesian':
            raise NotImplementedError, 'At the moment, only cartesian grids are supported.'

        self.B_field = B_field
        resolution = B_field.resolution
        self.direction = direction
        if direction == 'x' or direction == 'edge-on':
            self._integration_axis = 0
            self._Bp = self.B_field.x
            self._depths = self.B_field.grid.x[:,0,0]
        elif direction == 'y':
            self._integration_axis = 1
            self._Bp = self.B_field.y
            self._depths = self.B_field.grid.y[0,:,0]
        elif direction == 'z' or direction == 'face-on':
            self._integration_axis = 2
            self._Bp = self.B_field.z
            self._depths = self.B_field.grid.z[0,0,:]
        else:
            raise NotImplementedError, 'Only "x", "y" and "z" directions are currently supported."'
        self._ddepth = np.abs(self._depths[0]-self._depths[1])



        resolution[self._integration_axis] = 1

        super(Observables, self).__init__(box=B_field.box,
                                          resolution=resolution,
                                          default_parameters=default_parameters,
                                          dtype=dtype)

        # Reads in the parameters
        self.parameters = self._parse_parameters(kwargs)
        # Place holders
        self._synchrotron_emissivity = None
        self._instrinsic_polarization_angle = None
        self._intrinsic_polarization_degree = None
        self._psi = None
        self._Stokes_I = None
        self._Stokes_Q = None
        self._Stokes_U = None

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
        if self.direction == 'x':
            Bperp2 = self.B_field.y**2 + self.B_field.z**2
        elif self.direction == 'y':
            Bperp2 = self.B_field.x**2 + self.B_field.z**2
        elif self.direction == 'z':
            Bperp2 = self.B_field.x**2 + self.B_field.y**2

        ncr = self.parameters['obs_cosmic_ray_function'](
                                                 self.B_field.grid.r_spherical,
                                                 self.B_field.grid.theta,
                                                 self.B_field.grid.phi)

        return ncr * Bperp2**((gamma+1)/4) * lamb**((gamma-1)/2)

    @property
    def intrinsic_polarization_degree(self):
        """
        Computes the (intrinsic) degree of polarization, p0
        """
        if self._intrinsic_polarization_degree is None:
            gamma = self.parameters['obs_gamma']
            self._intrinsic_polarization_degree = (gamma+1.0)/(gamma+7./3.)
        return self._intrinsic_polarization_degree


    @property
    def intrinsic_polarization_angle(self):
        if self._instrinsic_polarization_angle is None:
            if self.direction == 'x':
                # Order needs to be checked
                B1 = self.B_field.y
                B2 = self.B_field.z
            elif self.direction == 'y':
                # Order needs to be checked
                B1 = self.B_field.z
                B2 = self.B_field.x
            elif self.direction == 'z':
                B1 = self.B_field.y
                B2 = self.B_field.x
            else:
                raise ValueError

            psi0 = np.pi/2.0 + util.arctan2(B1,B2)
            self._instrinsic_polarization_angle = psi0
        return self._instrinsic_polarization_angle

    @property
    def psi(self):
        """
        Polarization angle of radiation emitted at a given depth Faraday
        rotated over the line of sight
        """

        if self._psi is None:
            lamb = self.parameters['obs_wavelength_cm']

            ne =  self.parameters['obs_electron_density_function'](
                                                self.B_field.grid.r_spherical,
                                                self.B_field.grid.theta,
                                                self.B_field.grid.phi)

            self._psi = self._compute_psi(lamb, ne)
        return self._psi


    def _compute_psi(self, lamb, ne, from_bottom=False):
        """
        Computes the Faraday rotated polarization angle of radiation emmited at
        each depth
        """

        psi = self.intrinsic_polarization_angle.copy()
        # Creates an empty d2o array

        Bp = self._Bp
        ddepth = self._ddepth * 1000 # pc

        axis = [slice(None),]*3
        ax_n = self._integration_axis
        slices = [slice(None),]*3

        for i, depth in enumerate(self._depths):
            axis[ax_n] = slice(i,i+1) # e.g. for z, this will select psi[:,:,i]

            # The observer can be at the top or bottom
            if not from_bottom:
                # Will integrate from 0 to i; e.g. for z, Bp[:,:,0:i]
                slices[ax_n] = slice(0,i)
            else:
                # Will integrate from i to the end
                slices[ax_n] = slice(i,-1)

            integrand = ne[slices] * Bp[slices] * ddepth
            integral = integrand.sum(axis=ax_n)
            # Adjust the axes and uses full data (to make sure to avoid mistakes)
            integral = np.expand_dims(integral.get_full_data(), ax_n)
            # psi(z) = psi0(z) + 0.81\lambda^2 \int_-\infty^z ne(z') Bpara(z') dz'
            psi[axis] += 0.81*lamb**2*integral

        return psi

    @property
    def Stokes_I(self):
        """
        Computes Stokes parameter I (total emmission)
        """
        if self._Stokes_I is None:
            em = self.synchrotron_emissivity
            self._Stokes_I = em.sum(axis=self._integration_axis)*self._ddepth

        return self._Stokes_I

