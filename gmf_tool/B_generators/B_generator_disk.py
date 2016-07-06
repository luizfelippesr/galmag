# -*- coding: utf-8 -*-

import numpy as np
import scipy.integrate
import scipy.special

from gmf_tool.B_field import B_field

from B_generator import B_generator
from shear import simple_radial_shear

import threading
lock = threading.Lock()


class B_generator_disk(B_generator):
    def __init__(self, box, resolution, grid_type='cartesian',
                 default_parameters={}, dtype=np.float):

        super(B_generator_disk, self).__init__(
                                        box=box,
                                        resolution=resolution,
                                        grid_type=grid_type,
                                        default_parameters=default_parameters,
                                        dtype=dtype)

        self.component_count = len(self._builtin_parameter_defaults[
                                   'disk_component_normalization'])
        self._bessel_jn_zeros = scipy.special.jn_zeros(1, self.component_count)

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
            'disk_shear_function': simple_radial_shear,
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
        result_fields = \
            [np.zeros_like(local_r_cylindrical_grid, dtype=self.dtype)
             for i in xrange(3)]

        separator = abs(local_z_grid) <= parameters['disk_height']

        item_list = result_fields + \
            [local_r_cylindrical_grid, local_phi_grid, local_z_grid]

        inner_objects = [item[separator] for item in item_list]
        outer_objects = [item[~separator] for item in item_list]

        for component_number in xrange(self.component_count):
            temp_inner_fields = self._get_B_component(inner_objects[3:],
                                                      component_number,
                                                      parameters,
                                                      mode='inner')
            temp_outer_fields = self._get_B_component(outer_objects[3:],
                                                      component_number,
                                                      parameters,
                                                      mode='outer')
            temp_normalization = self._get_normalization(component_number,
                                                         parameters)
            for i in xrange(3):
                inner_objects[i] += temp_inner_fields[i]*temp_normalization
                outer_objects[i] += temp_outer_fields[i]*temp_normalization

        for i in xrange(3):
            result_fields[i][separator] += inner_objects[i]
            result_fields[i][~separator] += outer_objects[i]

        return result_fields

    def _get_B_component(self, grid_arrays, component_number, parameters,
                         mode):

        # rescale grid to unity
        r_grid = grid_arrays[0] / parameters['disk_radius']
        phi_grid = grid_arrays[1]
        z_grid = grid_arrays[2] / parameters['disk_height']
        if mode == 'outer':
            z_grid = 1

        # compute the magnetic field components
        [induction, dynamo_number, V0, s0, shear_function,
         component_normalization] = \
            [parameters[item] for item in
             ['disk_induction', 'disk_dynamo_number', 'disk_rotation_curve_V0',
              'disk_rotation_curve_s0', 'disk_shear_function',
              'disk_component_normalization']]

        kn = self._bessel_jn_zeros[component_number]
        Cn = component_normalization[component_number]

        # calculate reoccuring quantities
        four_pi_sqrt_DS = (4.0*np.pi**1.5) * \
            np.sqrt(dynamo_number * shear_function(r_grid, V0, s0))
        knr = kn*r_grid
        j0_knr = scipy.special.j0(knr)
        j1_knr = scipy.special.j1(knr)
        jv_knr = scipy.special.jv(2, knr)

        piz_half = (np.pi/2.) * z_grid
        sin_piz_half = np.sin(piz_half)
        cos_piz_half = np.cos(piz_half)

        Br = Cn * induction * j1_knr * \
            (cos_piz_half + 3.*np.cos(3*piz_half)/four_pi_sqrt_DS)

        Bphi = -0.5*Cn/(np.pi**2) * four_pi_sqrt_DS * j1_knr * cos_piz_half

        Bz = -2.*Cn*induction/np.pi * (j1_knr + 0.5*knr*(j0_knr-jv_knr)) * \
            (sin_piz_half + np.sin(3*piz_half)/four_pi_sqrt_DS)

        return Br, Bphi, Bz

    def _get_normalization(self, component_number, parameters):
        r_range = [0, parameters['disk_radius']]
        phi_range = [0, 2.*np.pi]
        z_range = [-parameters['disk_height'], parameters['disk_height']]

        # Integrates
        lock.acquire()
        normalization = scipy.integrate.nquad(self._normalization_integrand,
                                              [r_range, phi_range, z_range],
                                              args=(component_number,
                                                    parameters))
        lock.release()
        volume_correction = parameters['disk_radius'] * \
                            parameters['disk_height']
        return (normalization[0]/volume_correction)**(-0.5)

    def _normalization_integrand(self, r, phi, z, component_number,
                                 parameters):
        Br, Bphi, Bz = self._get_B_component([r, phi, z],
                                             component_number,
                                             parameters,
                                             mode='inner')
        return r*Br*Br/parameters['disk_radius'] + Bphi*Bphi + Bz*Bz
















