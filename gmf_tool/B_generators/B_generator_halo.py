# -*- coding: utf-8 -*-
from gmf_tool.B_field import B_field_component
import numpy as np
from B_generator import B_generator
from gmf_tool.util import curl_spherical, simpson
import gmf_tool.halo_free_decay_modes as halo_free_decay_modes
from gmf_tool.halo_profiles import simple_V, simple_alpha
from gmf_tool.galerkin import Galerkin_expansion_coefficients

class B_generator_halo(B_generator):
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

        self.growth_rate = np.NaN
        self.component_count = 0

    @property
    def _builtin_parameter_defaults(self):
        builtin_defaults = {
                            'halo_symmetric_field': True,
                            'halo_n_free_decay_modes': 4,
                            'halo_dynamo_type': 'alpha-omega',
                            'halo_rotation_function': simple_V,
                            'halo_alpha_function': simple_alpha,
                            'halo_Galerkin_ngrid': 501,
                            'halo_growing_mode_only': False,
                            'halo_compute_only_one_quadrant': True,
                            'halo_turbulent_induction': 0.6,
                            'halo_rotation_induction': 200.0,
                            'halo_radius': 20.0,
                            'halo_ref_radius': 8.5, # kpc (approx. solar radius)
                            'halo_ref_z': 0.02, # kpc (approx. solar z)
                            'halo_ref_Bphi': 0.1 # \muG
                            }
        return builtin_defaults


    def get_B_field(self, **kwargs):
        """
        Constructs a B_field component object containing a solution of the
        dynamo equation for the halo field.

        """
        parsed_parameters = self._parse_parameters(kwargs)

        # Finds the coefficients
        values, vect = Galerkin_expansion_coefficients(parsed_parameters)

        # Selects fastest growing mode
        ok = np.argmax(values.real)
        # Stores the growth rate and expansion coefficients
        self.growth_rate = values[ok]

        self.coefficients = vect[ok].real

        local_r_sph_grid = self.grid.r_spherical.get_local_data()
        local_theta_grid = self.grid.theta.get_local_data()
        local_phi_grid = self.grid.phi.get_local_data()

        local_arrays = [np.zeros_like(local_r_sph_grid) for i in range(3)]

        if not (parsed_parameters['halo_growing_mode_only'] and
                self.growth_rate<0):

            ref_radius = parsed_parameters['halo_ref_radius']


            ref_theta = np.arccos(parsed_parameters['halo_ref_z']/ref_radius)
            ref_radius = np.array([ref_radius])
            ref_theta = np.array([ref_theta])

            halo_radius = parsed_parameters['halo_radius']
            symmetric = parsed_parameters['halo_symmetric_field']

            # Computes the normalization at the solar radius
            Bsun_p = np.array([0.])

            for i, coefficient in enumerate(self.coefficients):
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

            for i in range(3):
                local_arrays[i] *= Bnorm

        # Initializes global arrays
        global_arrays = \
            [self.grid.get_prototype(dtype=self.dtype) for i in xrange(3)]

        # Bring the local array data into the d2o's
        for (g, l) in zip(global_arrays, local_arrays):
            g.set_local_data(l, copy=False)

        # Prepares the result field
        result_field = B_field_component(grid=self.grid,
                                         r_spherical=global_arrays[0],
                                         theta=global_arrays[1],
                                         phi=global_arrays[2],
                                         dtype=self.dtype,
                                         generator=self,
                                         parameters=parsed_parameters)

        return result_field
