# -*- coding: utf-8 -*-
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
"""
import numpy as np
import scipy.integrate
import scipy.special
from numpy import linalg as LA
from galmag.B_field import B_field_component

from B_generator import B_generator
import galmag.disk_profiles as prof

class B_generator_disk(B_generator):
    def __init__(self, grid=None, box=None, resolution=None,
                 grid_type='cartesian', default_parameters={},
                 dtype=np.float):

        super(B_generator_disk, self).__init__(
                                        grid=grid,
                                        box=box,
                                        resolution=resolution,
                                        grid_type=grid_type,
                                        default_parameters=default_parameters,
                                        dtype=dtype)

        self.modes_count = 0

    @property
    def _builtin_parameter_defaults(self):
        builtin_defaults = {
            'disk_modes_normalization': np.array([1., 1., 1.]),  # Cn_d
            'disk_height': 0.4,  # h_d
            'disk_radius': 20,  # R_d
            'disk_turbulent_induction': 0.6,  # Ralpha_d
            'disk_dynamo_number': -20,  # D_d
            'disk_shear_function': prof.Clemens_Milky_Way_shear_rate, # S(r)
            'disk_rotation_function': prof.Clemens_Milky_Way_rotation_curve, # V(r)
            'disk_height_function': prof.exponential_scale_height, # h(r)
            'solar_radius': 8.5, # kpc
            'disk_field_decay': True
            }
        return builtin_defaults


    def find_B_field(self, B_phi_solar_radius=-3, reversals=None,
                     number_of_modes=0, **kwargs):
        """
        Constructs B_field objects for the disk field based on constraints
        Input:
              B_phi_solar_radius -> Magnetic field intensity at the solar
                                    radius. Default: 10
              reversals -> a list containing the r-positions of field
                           reversals over the midplane (units consitent
                           with the grid).
              dr, dz -> the minimal r and z intervals used in the
                        calculation of the reversals
              number_of_modes -> Minimum of modes to be used.
                                 NB: modes_count = max(number_of_modes,
                                                        len(reversals)+1)

        Output: A B_field object satisfying the criteria
        """
        parsed_parameters = self._parse_parameters(kwargs)
        self.modes_count = max(len(reversals)+1, number_of_modes)
        self._bessel_jn_zeros = scipy.special.jn_zeros(1, self.modes_count)

        # The calculation is done solving the problem
        # A C = R
        # where A_{ij} = Bi(x_j) and C_i = disk_modes_normalization[i]
        # with x_j = reversals[j], for j=0..modes_count
        # and x_j = solar_radius for j=modes_count+1
        # R_i = Bsun for j=modes_count+1, otherwise R_i=0

        A = np.empty((len(reversals)+1, self.modes_count))
        tmp_parameters = parsed_parameters.copy()
        i=-1 # For the case of no-reversals

        for i, r_reversal in enumerate(reversals):
            r_reversal = np.float64(r_reversal)
            for j in range(self.modes_count):
                tmp_parameters['disk_modes_normalization'] = \
                                                np.zeros(self.modes_count)
                tmp_parameters['disk_modes_normalization'][j] = 1
                # Computes Bphi at each reversal (this should be 0)
                Br, Bphi, Bz = self._convert_coordinates_to_B_values(
                                                  np.array([r_reversal,]),
                                                  np.array([0.0,]),
                                                  np.array([0.0,]),
                                                  tmp_parameters)
                A[i,j] = Bphi[0]

        # The system of equations also accounts for the value at the Rsun
        for j in range(self.modes_count):
          tmp_parameters['disk_modes_normalization'] = \
                                                np.zeros(self.modes_count)
          tmp_parameters['disk_modes_normalization'][j] = 1
          # Computes Bphi at the solar radius (this should be Bsun)
          Br, Bphi, Bz = self._convert_coordinates_to_B_values(
              np.array([parsed_parameters['solar_radius'],]),
              np.array([0.0,]), np.array([0.0,]), tmp_parameters)
          A[i+1,j] = Bphi[0]

        results = np.zeros(len(reversals)+1)
        results[i+1] = B_phi_solar_radius

        # Uses a least squares fit to find the solution
        Cns, residuals, rank, s = LA.lstsq(A, results)
        parsed_parameters['disk_modes_normalization'] = Cns

        return self.get_B_field(**parsed_parameters)


    def get_B_field(self, **kwargs):
        """
        Returns a B_field object containing the specified disk field.
        Note: the coefficients for the modes have to be specified
        explicitly through the parameter disk_modes_normalization.
        """
        parsed_parameters = self._parse_parameters(kwargs)

        self.modes_count = len(parsed_parameters['disk_modes_normalization'])
        self._bessel_jn_zeros = scipy.special.jn_zeros(1, self.modes_count)

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

        result_field = B_field_component(grid=self.grid,
                                         r_cylindrical=global_arrays[0],
                                         phi=global_arrays[1],
                                         z=global_arrays[2],
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

        # Radial coordinate will be written in units of disk radius
        disk_radius = parameters['disk_radius']
        local_r_cylindrical_grid_dimensionless = local_r_cylindrical_grid \
                                                                / disk_radius
        # Computes disk scaleheight
        Rsun = parameters['solar_radius']
        height_function = parameters['disk_height_function']
        disk_height = height_function(local_r_cylindrical_grid_dimensionless,
                                      Rsun=Rsun,
                                      R_d=disk_radius,
                                      h_d=parameters['disk_height']
                                      )
        # Vertical coordinate will be written in units of disk scale height
        local_z_grid_dimensionless = local_z_grid/disk_height


        # Separates inner and outer parts of the disk solutions
        separator = abs(local_z_grid_dimensionless) <= 1.0
        # Separator which focuses on the dynamo active region
        active_separator = abs(local_r_cylindrical_grid_dimensionless <= 1.0)

        # List of objects required for the computation which includes
        # the result-arrays and the dimensionless local coordinate grid
        item_list = result_fields + [
            local_r_cylindrical_grid_dimensionless,
            local_phi_grid,
            local_z_grid_dimensionless]

        inner_objects = [item[separator * active_separator] for item in item_list]
        outer_objects = [item[~separator * active_separator] for item in item_list]

        for mode_number in xrange(self.modes_count):

            mode_normalization = \
                parameters['disk_modes_normalization'][mode_number]
            if mode_normalization==0:
                continue

            # Normalizes each mode, making |B_mode| unity at Rsun
            Br_sun, Bphi_sun, Bz_sun = self._get_B_mode([Rsun/disk_radius,
                                                              0.0, 0.0],
                                                              mode_number,
                                                              1.0,
                                                              parameters,
                                                              mode='inner')
            renormalization = (Br_sun**2 + Bphi_sun**2 + Bz_sun**2)**-0.5

            mode_normalization *= renormalization

            temp_inner_fields = self._get_B_mode(inner_objects[3:],
                                                      mode_number,
                                                      mode_normalization,
                                                      parameters,
                                                      mode='inner')
            temp_outer_fields = self._get_B_mode(outer_objects[3:],
                                                      mode_number,
                                                      mode_normalization,
                                                      parameters,
                                                      mode='outer')

            for i in xrange(3):
                inner_objects[i] += temp_inner_fields[i]
                outer_objects[i] += temp_outer_fields[i]

        for i in xrange(3):
            result_fields[i][separator * active_separator] += inner_objects[i]
            result_fields[i][~separator * active_separator] += outer_objects[i]

        return result_fields

    def _get_B_mode(self, grid_arrays, mode_number,
                         mode_normalization, parameters, mode):

        # Unpacks some parameters (for convenience)
        disk_radius = parameters['disk_radius']
        induction = parameters['disk_turbulent_induction']
        dynamo_number = parameters['disk_dynamo_number']
        shear_function = parameters['disk_shear_function']
        rotation_function = parameters['disk_rotation_function']
        solar_radius = parameters['solar_radius']
        Cn = mode_normalization

        r_grid = grid_arrays[0]
        phi_grid = grid_arrays[1]

        if mode == 'inner':
            z_grid = grid_arrays[2]
        elif mode == 'outer':
            # Assumes field constant outside the disk height
            # i.e. z_grid=1, for z>h; z_grid = -1 for z<-h
            z_grid = grid_arrays[2]/abs(grid_arrays[2])

        # Computes angular velocity and shear
        Omega = rotation_function(r_grid, R_d=disk_radius,
                                  Rsun=solar_radius)/r_grid
        Shear = shear_function(r_grid, R_d=disk_radius,
                                  Rsun=solar_radius)

        # Calculates reoccuring quantities
        kn = self._bessel_jn_zeros[mode_number]
        four_pi_sqrt_DS = (4.0*np.pi**1.5) \
            * np.sqrt(-dynamo_number * Shear * Omega)
        knr = kn*r_grid
        j0_knr = scipy.special.j0(knr)
        j1_knr = scipy.special.j1(knr)
        jv_knr = scipy.special.jv(2, knr)

        piz_half = (np.pi/2.) * z_grid
        sin_piz_half = np.sin(piz_half)
        cos_piz_half = np.cos(piz_half)

        # Computes the magnetic field modes
        Br = Cn * induction * j1_knr * \
            (cos_piz_half + 3.*np.cos(3*piz_half)/four_pi_sqrt_DS)

        Bphi = -0.5*Cn/(np.pi**2) * four_pi_sqrt_DS * j1_knr * cos_piz_half

        Bz = -2.*Cn*induction/np.pi * (j1_knr + 0.5*knr*(j0_knr-jv_knr)) * \
            (sin_piz_half + np.sin(3*piz_half)/four_pi_sqrt_DS)

        if mode == 'outer' and parameters['disk_field_decay']:
            # Makes the exernal field decay with r^-3
            Br /= abs(grid_arrays[2])**3
            Bphi /= abs(grid_arrays[2])**3
            Bz /= abs(grid_arrays[2])**3

        return Br, Bphi, Bz

