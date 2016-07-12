# -*- coding: utf-8 -*-

import numpy as np
import scipy.integrate
import scipy.special

from gmf_tool.B_field import B_field

from B_generator import B_generator
from disk_profiles import Clemens_Milky_Way_shear_rate
from disk_profiles import Clemens_Milky_Way_rotation_curve
from disk_profiles import exponenial_scale_height

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
            'disk_component_normalization': np.array([1., 1., 1.]),  # Cn_d
            'disk_height': 0.4,  # h_d
            'disk_radius': 20,  # R_d
            'disk_turbulent_induction': 0.6,  # Ralpha_d
            'disk_dynamo_number': -20,  # D_d
            'disk_shear_function': Clemens_Milky_Way_shear_rate, # S(r)
            'disk_rotation_function': Clemens_Milky_Way_rotation_curve, # V(r)
            'disk_height_function': exponenial_scale_height, # h(r)
            'solar_radius': 8.5, # kpc
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
        # Initializes local variables
        result_fields = \
            [np.zeros_like(local_r_cylindrical_grid, dtype=self.dtype)
             for i in xrange(3)]

        # Computes disk scaleheight
        Rsun = parameters['solar_radius']
        height_function = parameters['disk_height_function']
        disk_height = height_function(local_r_cylindrical_grid, Rsun=Rsun,
                                      R_d=parameters['disk_radius'],
                                      h_d=parameters['disk_height']
                                      )
        # Separates inner and outer parts of the disk solutions
        separator = abs(local_z_grid) <= disk_height

        # List of objects required for the computation which includes
        # the result-arrays and the dimensionless local coordinate grid
        item_list = result_fields + [
            # Radial coordinate will be written in units of disk radius
            local_r_cylindrical_grid/parameters['disk_radius'],
            local_phi_grid,
            # Vertical coordinate will be written in units of disk scale height
            local_z_grid/disk_height]
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

            # Normalizes each mode, making |B_mode| unity at Rsun
            Br_sun, Bphi_sun, Bz_sun = self._get_B_component([Rsun, 0.0, 0.0],
                                                              component_number,
                                                              parameters,
                                                              mode='inner')
            temp_normalization = (Br_sun**2 + Bphi_sun**2 + Bz_sun**2)**(-0.5)

            for i in xrange(3):
                inner_objects[i] += temp_inner_fields[i]*temp_normalization
                outer_objects[i] += temp_outer_fields[i]*temp_normalization

        for i in xrange(3):
            result_fields[i][separator] += inner_objects[i]
            result_fields[i][~separator] += outer_objects[i]

        return result_fields

    def _get_B_component(self, grid_arrays, component_number, parameters, mode):

        # Unpacks some parameters (for convenience)
        disk_radius = parameters['disk_radius']
        induction = parameters['disk_turbulent_induction']
        dynamo_number = parameters['disk_dynamo_number']
        shear_function = parameters['disk_shear_function']
        rotation_function = parameters['disk_rotation_function']
        solar_radius = parameters['solar_radius']
        component_normalization = parameters['disk_component_normalization']

        r_grid = grid_arrays[0]
        phi_grid = grid_arrays[1]

        if mode == 'inner':
            z_grid = grid_arrays[2]
        elif mode == 'outer':
            # Assumes field constant for z>disk_scaleheight
            z_grid = 1

        # Computes angular velocity and shear
        Omega = rotation_function(r_grid, R_d=disk_radius,
                                  Rsun=solar_radius)/r_grid
        Shear = shear_function(r_grid, R_d=disk_radius,
                                  Rsun=solar_radius)

        # Calculates reoccuring quantities
        kn = self._bessel_jn_zeros[component_number]
        four_pi_sqrt_DS = (4.0*np.pi**1.5) \
            * np.sqrt(-dynamo_number * Shear * Omega)
        knr = kn*r_grid
        j0_knr = scipy.special.j0(knr)
        j1_knr = scipy.special.j1(knr)
        jv_knr = scipy.special.jv(2, knr)

        piz_half = (np.pi/2.) * z_grid
        sin_piz_half = np.sin(piz_half)
        cos_piz_half = np.cos(piz_half)

        # Computes the magnetic field components
        Cn = component_normalization[component_number]

        Br = Cn * induction * j1_knr * \
            (cos_piz_half + 3.*np.cos(3*piz_half)/four_pi_sqrt_DS)

        Bphi = -0.5*Cn/(np.pi**2) * four_pi_sqrt_DS * j1_knr * cos_piz_half

        Bz = -2.*Cn*induction/np.pi * (j1_knr + 0.5*knr*(j0_knr-jv_knr)) * \
            (sin_piz_half + np.sin(3*piz_half)/four_pi_sqrt_DS)

        return Br, Bphi, Bz







