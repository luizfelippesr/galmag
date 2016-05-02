# -*- coding: utf-8 -*-

import numpy as np
import scipy.special

from gmf_tool.B_field import B_field

from B_generator import B_generator
from shear import simple_radial_shear


class B_generator_disk(B_generator):
    def __init__(self, box, resolution, default_parameters={},
                 dtype=np.float):

        super(B_generator_disk, self).__init__(
                                        box=box,
                                        resolution=resolution,
                                        default_parameters=default_parameters,
                                        dtype=dtype)

        component_count = len(self._builtin_parameter_defaults[
                                    'disk_component_normalization'])
        self._bessel_jn_zeros = scipy.special.jn_zeros(1, component_count)

    @property
    def _builtin_parameter_defaults(self):
        builtin_defaults = {
            'disk_height': 0.4,  # h_d
            'disk_radius': 20,  # R_d
            'disk_induction': 0.6,  # Ralpha_d
            'disk_component_normalization': np.array([1., 1., 1.]),  # Cn_d
            'disk_dynamo_number': -20,  # D_d
            'disk_rotation_curve_V0': 1.,
            'disk_rotation_curve_s0': 1.,
            'shear': simple_radial_shear,
                    }
        return builtin_defaults

    def get_B_field(self, **kwargs):
        parsed_parameters = self._parse_parameters(kwargs)

        local_r_cylindrical_grid = self.grid.r_cylindrical.get_local_data()
        local_phi_grid = self.grid.phi.get_local_data()
        local_z_grid = self.grid.z.get_local_data()

        # local_B_r_cylindrical, local_B_phi, local_B_z
        local_arrays = \
            self._convert_coordinates_to_B_values(local_r_cylindrical_grid,
                                                  local_phi_grid,
                                                  local_z_grid,
                                                  parsed_parameters)

        # global_r_cylindrical, global_phi, global_z
        global_arrays = \
            [self.grid.get_prototype(dtype=self.dtype) for i in xrange(3)]

        # bring the local array data into the d2o's
        for (g, l) in zip(global_arrays, local_arrays):
            g.set_local_data(l, copy=False)

        result_field = B_field(grid=self.grid,
                               r_cylindrical=global_arrays[0],
                               phi=global_arrays[1],
                               theta=global_arrays[2],
                               dtype=self.dtype,
                               generator=self,
                               parameters=parsed_parameters)
        return result_field

    def _convert_coordinates_to_B_values(self, local_r_cylindrical_grid,
                                         local_phi_grid, local_z_grid,
                                         parameters):
        return (np.copy(local_z_grid) for i in xrange(3))
