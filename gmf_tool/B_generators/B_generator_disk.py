# -*- coding: utf-8 -*-

import numpy as np
import scipy.integrate
import scipy.special
from numpy import linalg as LA
from gmf_tool.B_field import B_field_component

from B_generator import B_generator
from disk_profiles import Clemens_Milky_Way_shear_rate
from disk_profiles import Clemens_Milky_Way_rotation_curve
from disk_profiles import exponential_scale_height

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

        self.component_count = 0

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
            'disk_height_function': exponential_scale_height, # h(r)
            'solar_radius': 8.5, # kpc
            'disk_field_decay': True
            }
        return builtin_defaults


    def find_B_field(self, B_phi_solar_radius=-3, reversals=None,
                     number_of_components=0, **kwargs):
        """ Constructs B_field objects for the disk field based on constraints
            Input:
                  B_phi_solar_radius -> Magnetic field intensity at the solar
                                        radius. Default: 10
                  reversals -> a list containing the r-positions of field
                               reversals over the midplane (units consitent
                               with the grid).
                  dr, dz -> the minimal r and z intervals used in the
                            calculation of the reversals
                  number_of_components -> Minimum of components to be used.
                              NB: component_count = max(number_of_components,
                                                        len(reversals)+1)

            Output: A B_field object satisfying the criteria
        """
        parsed_parameters = self._parse_parameters(kwargs)

        self.component_count = max(len(reversals)+1, number_of_components)
        self._bessel_jn_zeros = scipy.special.jn_zeros(1, self.component_count)

        # The calculation is done solving the problem
        # A C = R
        # where A_{ij} = Bi(x_j) and C_i = disk_component_normalization[i]
        # with x_j = reversals[j], for j=0..component_count
        # and x_j = solar_radius for j=component_count+1
        # R_i = Bsun for j=component_count+1, otherwise R_i=0

        A = np.empty((len(reversals)+1, self.component_count))
        tmp_parameters = parsed_parameters.copy()

        for i, r_reversal in enumerate(reversals):
            r_reversal = np.float64(r_reversal)
            for j in range(self.component_count):
                tmp_parameters['disk_component_normalization'] = \
                                                np.zeros(self.component_count)
                tmp_parameters['disk_component_normalization'][j] = 1
                # Computes Bphi at each reversal (this should be 0)
                Br, Bphi, Bz = self._convert_coordinates_to_B_values(
                                                  np.array([r_reversal,]),
                                                  np.array([0.0,]),
                                                  np.array([0.0,]),
                                                  tmp_parameters)
                A[i,j] = Bphi[0]

        # The system of equations also accounts for the value at the Rsun
        for j in range(self.component_count):
          tmp_parameters['disk_component_normalization'] = \
                                                np.zeros(self.component_count)
          tmp_parameters['disk_component_normalization'][j] = 1
          # Computes Bphi at the solar radius (this should be Bsun)
          Br, Bphi, Bz = self._convert_coordinates_to_B_values(
              np.array([parsed_parameters['solar_radius'],]),
              np.array([0.0,]), np.array([0.0,]), tmp_parameters)
          A[i+1,j] = Bphi[0]

        results = np.zeros(len(reversals)+1)
        results[i+1] = B_phi_solar_radius

        # Uses a least squares fit to find the solution
        Cns, residuals, rank, s = LA.lstsq(A, results)
        parsed_parameters['disk_component_normalization'] = Cns

        return self.get_B_field(**parsed_parameters)


    def get_B_field(self, **kwargs):
        """ Returns a B_field object containing the specified disk field.
            Note: the coefficients for the components have to be specified
            explicitly through the parameter disk_component_normalization.
        """
        parsed_parameters = self._parse_parameters(kwargs)

        self.component_count = len(parsed_parameters['disk_component_normalization'])
        self._bessel_jn_zeros = scipy.special.jn_zeros(1, self.component_count)

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

        # List of objects required for the computation which includes
        # the result-arrays and the dimensionless local coordinate grid
        item_list = result_fields + [
            local_r_cylindrical_grid_dimensionless,
            local_phi_grid,
            local_z_grid_dimensionless]

        inner_objects = [item[separator] for item in item_list]
        outer_objects = [item[~separator] for item in item_list]

        for component_number in xrange(self.component_count):

            component_normalization = \
                parameters['disk_component_normalization'][component_number]
            if component_normalization==0:
                continue

            temp_inner_fields = self._get_B_component(inner_objects[3:],
                                                      component_number,
                                                      component_normalization,
                                                      parameters,
                                                      mode='inner')
            temp_outer_fields = self._get_B_component(outer_objects[3:],
                                                      component_number,
                                                      component_normalization,
                                                      parameters,
                                                      mode='outer')

            # Normalizes each mode, making |B_mode| unity at Rsun
            Br_sun, Bphi_sun, Bz_sun = self._get_B_component([Rsun/disk_radius,
                                                              0.0, 0.0],
                                                              component_number,
                                                              1.0,
                                                              parameters,
                                                              mode='inner')
            mode_normalization = (Br_sun**2 + Bphi_sun**2 + Bz_sun**2)**-0.5

            for i in xrange(3):
                inner_objects[i] += temp_inner_fields[i]*mode_normalization
                outer_objects[i] += temp_outer_fields[i]*mode_normalization

        for i in xrange(3):
            result_fields[i][separator] += inner_objects[i]
            result_fields[i][~separator] += outer_objects[i]

        return result_fields

    def _get_B_component(self, grid_arrays, component_number,
                         component_normalization, parameters, mode):

        # Unpacks some parameters (for convenience)
        disk_radius = parameters['disk_radius']
        induction = parameters['disk_turbulent_induction']
        dynamo_number = parameters['disk_dynamo_number']
        shear_function = parameters['disk_shear_function']
        rotation_function = parameters['disk_rotation_function']
        solar_radius = parameters['solar_radius']
        Cn = component_normalization

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
        Br = Cn * induction * j1_knr * \
            (cos_piz_half + 3.*np.cos(3*piz_half)/four_pi_sqrt_DS)

        Bphi = -0.5*Cn/(np.pi**2) * four_pi_sqrt_DS * j1_knr * cos_piz_half

        Bz = -2.*Cn*induction/np.pi * (j1_knr + 0.5*knr*(j0_knr-jv_knr)) * \
            (sin_piz_half + np.sin(3*piz_half)/four_pi_sqrt_DS)

        if mode == 'outer' and parameters['disk_field_decay']:
            # Makes the exernal field decay with r^-3
            Br /= grid_arrays[2]**3
            Bphi /= grid_arrays[2]**3
            Bz /= grid_arrays[2]**3

        return Br, Bphi, Bz

