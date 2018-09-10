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
# -*- coding: utf-8 -*-
from galmag.B_field import B_field_component
import numpy as np
from B_generator import B_generator
from galmag.util import curl_spherical, simpson
import galmag.halo_free_decay_modes as halo_free_decay_modes
from galmag.halo_profiles import simple_V, simple_alpha
from galmag.galerkin import Galerkin_expansion_coefficients

class B_generator_halo(B_generator):
    """
    Generator for the halo field

    Parameters
    ----------
    box : 3x2-array_like
         Box limits
    resolution : 3-array_like
         containing the resolution along each axis.
    grid_type : str, optional
        Choice between 'cartesian', 'spherical' and 'cylindrical' *uniform*
        coordinate grids. Default: 'cartesian'
    dtype : numpy.dtype, optional
        Data type used. Default: np.dtype(np.float)

    """
    def __init__(self, grid=None, box=None, resolution=None,
                 grid_type='cartesian', default_parameters={},
                 dtype=np.float):

        super(B_generator_halo, self).__init__(
                                        grid=grid,
                                        box=box,
                                        resolution=resolution,
                                        grid_type=grid_type,
                                        default_parameters=default_parameters,
                                        dtype=dtype)
    @property
    def _builtin_parameter_defaults(self):
        builtin_defaults = {
                            'halo_symmetric_field': True,
                            'halo_n_free_decay_modes': 4,
                            'halo_dynamo_type': 'alpha2-omega',
                            'halo_rotation_function': simple_V,
                            'halo_alpha_function': simple_alpha,
                            'halo_Galerkin_ngrid': 501,
                            'halo_growing_mode_only': False,
                            'halo_compute_only_one_quadrant': True,
                            'halo_turbulent_induction': 4.331,
                            'halo_rotation_induction': 203.65472,
                            'halo_radius': 15.0,
                            'halo_ref_radius': 8.5, # kpc (approx. solar radius)
                            'halo_ref_z': 0.02, # kpc (approx. solar z)
                            'halo_ref_Bphi': -0.5, # muG
                            'halo_rotation_characteristic_radius': 3.0, # kpc
                            'halo_rotation_characteristic_height': 1000,
                            'halo_manually_specified_coefficients': None,
                            'halo_do_not_normalize': False, # For testing
                            }
        return builtin_defaults


    def get_B_field(self, **kwargs):
        """
        Constructs a B_field component object containing a solution of the
        dynamo equation for the halo field.
        """
        parsed_parameters = self._parse_parameters(kwargs)

        if parsed_parameters['halo_manually_specified_coefficients'] is None:
            # Finds the coefficients
            values, vect = Galerkin_expansion_coefficients(parsed_parameters)

            # Selects fastest growing mode
            ok = np.argmax(values.real)
            # Stores the growth rate and expansion coefficients
            growth_rate = values[ok]
            # Stores coefficient (ignores any imaginary part)
            coefficients = vect[:,ok].real
        else:
            # For testing purposes, it is possible to mannually specify the
            # the coefficients...
            coefficients = parsed_parameters['halo_manually_specified_coefficients']
            # Overrides some parameters
            growth_rate = None
            parsed_parameters['halo_growing_mode_only'] = False
            parsed_parameters['halo_n_free_decay_modes'] = len(coefficients)


        local_r_sph_grid = self.grid.r_spherical.get_local_data()
        local_theta_grid = self.grid.theta.get_local_data()
        local_phi_grid = self.grid.phi.get_local_data()

        local_arrays = [np.zeros_like(local_r_sph_grid) for i in range(3)]

        if not (parsed_parameters['halo_growing_mode_only'] and
                growth_rate<0):

            ref_radius = parsed_parameters['halo_ref_radius']
            ref_theta = np.arccos(parsed_parameters['halo_ref_z']/ref_radius)

            ref_radius = np.array([ref_radius])
            ref_theta = np.array([ref_theta])

            halo_radius = parsed_parameters['halo_radius']
            symmetric = parsed_parameters['halo_symmetric_field']

            # Computes the normalization at the reference radius
            Bsun_p = np.array([0.])

            for i, coefficient in enumerate(coefficients):
                # Calculates free-decay modes locally
                Bmode = halo_free_decay_modes.get_mode(
                  local_r_sph_grid/halo_radius, local_theta_grid,
                  local_phi_grid, i+1, symmetric)

                for j in range(3):
                    local_arrays[j] += Bmode[j] * coefficient

                Brefs = halo_free_decay_modes.get_mode(
                    ref_radius/halo_radius, ref_theta, np.array([0.]), i+1,
                    symmetric)

                Bsun_p += Brefs[2] * coefficient

            Bnorm = parsed_parameters['halo_ref_Bphi']/Bsun_p[0]

            if not parsed_parameters['halo_do_not_normalize']:
                for i in range(3):
                    local_arrays[i] *= Bnorm

        # Initializes global arrays
        global_arrays = \
            [self.grid.get_prototype(dtype=self.dtype) for i in xrange(3)]

        # Bring the local array data into the d2o's
        for (g, l) in zip(global_arrays, local_arrays):
            g.set_local_data(l, copy=False)

        parsed_parameters['halo_field_growth_rate'] = growth_rate
        parsed_parameters['halo_field_coefficients'] = coefficients
        # Prepares the result field
        result_field = B_field_component(grid=self.grid,
                                         r_spherical=global_arrays[0],
                                         theta=global_arrays[1],
                                         phi=global_arrays[2],
                                         dtype=self.dtype,
                                         generator=self,
                                         parameters=parsed_parameters)

        result_field.growth_rate = growth_rate
        result_field.coefficients = coefficients

        return result_field
