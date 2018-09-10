# -*- coding: utf-8 -*-
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
import numpy as np
import scipy.integrate
import scipy.special
from numpy import linalg as LA
from galmag.B_field import B_field_component

from B_generator import B_generator
import galmag.disk_profiles as prof

class B_generator_disk(B_generator):
    """
    Generator for the disk field


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
            'disk_height': 0.5,  # h_d
            'disk_radius': 17.0,  # R_d
            'disk_turbulent_induction': 0.386392513,  # Ralpha_d
            'disk_dynamo_number': -20.4924192,  # D_d
            'disk_shear_function': prof.Clemens_Milky_Way_shear_rate, # S(r)
            'disk_rotation_function': prof.Clemens_Milky_Way_rotation_curve, # V(r)
            'disk_height_function': prof.exponential_scale_height, # h(r)
            'disk_regularization_radius': None, # kpc
            'disk_ref_r_cylindrical': 8.5, # kpc
            'disk_field_decay': True,
            'disk_newman_boundary_condition_envelope': False
            }
        return builtin_defaults


    def find_B_field(self, B_phi_ref=-3.0, reversals=None,
                     number_of_modes=0, **kwargs):
        """
        Constructs B_field objects for the disk field based on constraints.

        Parameters
        ----------
        B_phi_ref : float
            Magnetic field intensity at the reference radius. Default: -3
        reversals : list_like
            a list containing the r-positions of field reversals over the
            midplane (units consitent with the grid).
        number_of_modes : int, optional
            Minimum of modes to be used.
            NB: modes_count = max(number_of_modes, len(reversals)+1)

        Note
        ----
        Other disc parameters should be specified as keyword arguments.

        Returns
        -------
        B_field_component
            The computed disc component.
        """
        parsed_parameters = self._parse_parameters(kwargs)
        self.modes_count = max(len(reversals)+1, number_of_modes)
        self._bessel_jn_zeros = scipy.special.jn_zeros(1, self.modes_count)

        # The calculation is done solving the problem
        # A C = R
        # where A_{ij} = Bi(x_j) and C_i = disk_modes_normalization[i]
        # with x_j = reversals[j], for j=0..modes_count
        # and x_j = disk_ref_r_cylindrical for j=modes_count+1
        # R_i = B_phi_ref for j=modes_count+1, otherwise R_i=0

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
          # Computes Bphi at the reference radius (this should be B_phi_ref)
          Br, Bphi, Bz = self._convert_coordinates_to_B_values(
              np.array([parsed_parameters['disk_ref_r_cylindrical'],]),
              np.array([0.0,]), np.array([0.0,]), tmp_parameters)
          A[i+1,j] = Bphi[0]

        results = np.zeros(len(reversals)+1)
        results[i+1] = B_phi_ref

        # Uses a least squares fit to find the solution
        Cns, residuals, rank, s = LA.lstsq(A, results)
        parsed_parameters['disk_modes_normalization'] = Cns

        return self.get_B_field(**parsed_parameters)


    def get_B_field(self, **kwargs):
        """
        Computes a B_field object containing the specified disk field.

        Parameters
        ----------
        disk_modes_normalization : array_like
             array of coefficients for the disc modes

        Note
        ----
        Other disc parameters should be specified as keyword arguments.

        Returns
        -------
        B_field_component
            The computed disc component.
        """
        parsed_parameters = self._parse_parameters(kwargs)

        self.modes_count = len(parsed_parameters['disk_modes_normalization'])
        if not parsed_parameters['disk_newman_boundary_condition_envelope']:
            # Uses Q(s_d) = 0 as boundary condition, which corresponds to
            # k_n being a root of the J_1 Bessel function
            self._bessel_jn_zeros = scipy.special.jn_zeros(1, self.modes_count)
        else:
            # Uses d[sQ(s)]/ds = 0 at s_d as boundary condition, which
            # corresponds to k_n being a root of the J_0 Bessel function
            self._bessel_jn_zeros = scipy.special.jn_zeros(0, self.modes_count)

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
        """Args:
          local_r_cylindrical_grid:
          local_phi_grid:
          local_z_grid:

        Parameters
        ----------
        local_r_cylindrical_grid :

        local_phi_grid :

        local_z_grid :

        parameters :


        Returns
        -------
        type


        """
        # Initializes local variables
        result_fields = \
            [np.zeros_like(local_r_cylindrical_grid, dtype=self.dtype)
             for i in xrange(3)]

        # Radial coordinate will be written in units of disk radius
        disk_radius = parameters['disk_radius']
        local_r_cylindrical_grid_dimensionless = local_r_cylindrical_grid \
                                                                / disk_radius
        # Computes disk scaleheight
        Rsun = parameters['disk_ref_r_cylindrical']
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

    def _get_B_mode(self, grid_arrays, mode_number, mode_normalization,
                    parameters, mode):
        """
        Computes a given disk mode

        Parameters
        ----------
        grid_arrays : array_like
          array containig r_cylindrical, phi and z
        mode_number : int
          the index of the requested mode
        mode_normalization :
          normalization of the mode
        parameters : dict
          dictionary containin parameters
        mode : str
          'inner' for computing the field inside the disc height,
          'outer' for it externally

        Returns
        -------
        list
            List containing d2o's for Br, Bphi, Bz
        """

        # Unpacks some parameters (for convenience)
        disk_radius = parameters['disk_radius']
        shear_function = parameters['disk_shear_function']
        rotation_function = parameters['disk_rotation_function']
        height_function = parameters['disk_height_function']
        disk_ref_r_cylindrical = parameters['disk_ref_r_cylindrical']
        disk_height_ref = parameters['disk_height']
        # Switches reference within dynamo number and R_\alpha
        # from s_0 to s_d
        dynamo_number = parameters['disk_dynamo_number']  \
                        * disk_ref_r_cylindrical / disk_radius
        Ralpha = parameters['disk_turbulent_induction']  \
                    * disk_ref_r_cylindrical / disk_radius

        Cn = mode_normalization

        r_grid = grid_arrays[0]
        phi_grid = grid_arrays[1]
        pi = np.pi

        if mode == 'inner':
            z_grid = grid_arrays[2] # Note: this is actually z/h(s)
        elif mode == 'outer':
            # Assumes field constant outside the disk height
            # i.e. z_grid=1, for z>h; z_grid = -1 for z<-h
            z_grid = grid_arrays[2]/abs(grid_arrays[2])

        # Computes angular velocity, shear and scaleheight
        Omega = prof.Omega(rotation_function, r_grid,
                           R_d=disk_radius, Rsun=disk_ref_r_cylindrical)
        Shear = shear_function(r_grid, R_d=disk_radius,
                               Rsun=disk_ref_r_cylindrical)
        disk_height = height_function(r_grid, Rsun=disk_ref_r_cylindrical,
                                      R_d=disk_radius)

        if parameters['disk_regularization_radius'] is not None:
            rreg = parameters['disk_regularization_radius']/disk_radius

            # Finds the value of Omega at the regularization radios
            # (i.e. the value that will remain constant until 0
            Om_reg = prof.Omega(rotation_function, rreg,
                                R_d=disk_radius, Rsun=disk_ref_r_cylindrical)
            # Regularises the Omega and Shear profiles
            Omega, Shear = prof.regularize(r_grid, Omega, Shear, rreg, Om_reg)

        # Scaleheight in disk radius units (same units as s)
        h = disk_height *disk_height_ref/disk_radius
        # Local dynamo number
        Dlocal = dynamo_number * Shear * Omega * disk_height**2
        sqrt_Dlocal = np.sqrt(-Dlocal)
        # Normalization correction
        K0 =  (-Dlocal*4./pi -Dlocal*9./pi**3/16 + 1.0)**(-0.5)
        # Other reoccuring quantities
        kn = self._bessel_jn_zeros[mode_number]
        four_pi32 = (4.0*pi**(3./2.))
        knr = kn*r_grid
        j0_knr = scipy.special.j0(knr)
        j1_knr = scipy.special.j1(knr)
        piz_half = (pi/2.) * z_grid
        sin_piz_half = np.sin(piz_half)
        cos_piz_half = np.cos(piz_half)
        Ralpha_local = Omega * Ralpha

        # Computes the magnetic field modes
        Br = Cn*K0 * Ralpha_local * j1_knr * \
            (cos_piz_half + 3./four_pi32*sqrt_Dlocal*np.cos(3.*piz_half))

        Bphi = -2.* Cn*K0 * sqrt_Dlocal/np.sqrt(pi) * j1_knr * cos_piz_half

        Bz = -2.* kn*h /pi *Cn*K0 * Ralpha_local * j0_knr  \
            * (sin_piz_half + np.sin(3*piz_half)*sqrt_Dlocal/four_pi32)

        if mode == 'outer' and parameters['disk_field_decay']:
            # Makes the exernal field decay with z^-3
            Br /= abs(grid_arrays[2])**3
            Bphi /= abs(grid_arrays[2])**3
            Bz /= abs(grid_arrays[2])**3

        return Br, Bphi, Bz
