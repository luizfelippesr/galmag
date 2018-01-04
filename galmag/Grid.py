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
"""
Contains the definition of the Grid class.
"""
import numpy as np
from d2o import distributed_data_object


class Grid(object):
    """
    Defines a 3D grid object for a given choice of box dimensions
    and resolution.

    Calling the attributes does the conversion between different coordinate
    systems automatically.

    Parameters
    ----------
    box : 3x2-array_like
         Box limits
    resolution : 3-array_like
         containing the resolution along each axis.
    grid_type : str, optional
        Choice between 'cartesian', 'spherical' and 'cylindrical' *uniform*
        coordinate grids. Default: 'cartesian'
    """
    def __init__(self, box, resolution, grid_type='cartesian'):

        self.box = np.empty((3, 2), dtype=np.float)
        self.resolution = np.empty((3,), dtype=np.int)

        # use numpy upcasting of scalars and dtype conversion
        self.box[:] = box
        self.resolution[:] = resolution
        self.grid_type=grid_type

        self._coordinates = None
        self._prototype_source = None

    @property
    def coordinates(self):
        """A dictionary contaning all the coordinates"""
        if self._coordinates is None:
            self._coordinates = self._generate_coordinates()
        return self._coordinates

    @property
    def x(self):
        """Horizontal coordinate, :math:`x`"""
        return self.coordinates['x']

    @property
    def y(self):
        """Horizontal coordinate, :math:`y`"""
        return self.coordinates['y']

    @property
    def z(self):
        """Vertical coordinate, :math:`z`"""
        return self.coordinates['z']

    @property
    def r_spherical(self):
        """Spherical radial coordinate, :math:`r`"""
        return self.coordinates['r_spherical']

    @property
    def r_cylindrical(self):
        """Cylindrical radial coordinate, :math:`s`"""
        return self.coordinates['r_cylindrical']

    @property
    def theta(self):
        r"""Polar coordinate, :math:`\theta`"""
        return self.coordinates['theta']

    @property
    def phi(self):
        r"""Azimuthal coordinate, :math:`\phi`"""
        return self.coordinates['phi']

    @property
    def sin_theta(self):
        r""":math:`\sin(\theta)`"""
        return self.r_cylindrical / self.r_spherical

    @property
    def cos_theta(self):
        r""":math:`\cos(\theta)`"""
        return self.z / self.r_spherical

    @property
    def sin_phi(self):
        r""":math:`\sin(\phi)`"""
        return self.y / self.r_cylindrical

    @property
    def cos_phi(self):
        r""":math:`\cos(\phi)`"""
        return self.x / self.r_cylindrical

    def _generate_coordinates(self):
        # Initializes all coordinate arrays
        [x_array, y_array, z_array, r_spherical_array, r_cylindrical_array,
         theta_array, phi_array] = [self.get_prototype() for i in xrange(7)]

        local_start = x_array.distributor.local_start
        local_end = x_array.distributor.local_end
        box = self.box
        local_slice = (slice(local_start, local_end),
                       slice(box[1, 0], box[1, 1], self.resolution[1]*1j),
                       slice(box[2, 0], box[2, 1], self.resolution[2]*1j))

        local_coordinates = np.mgrid[local_slice]

        local_coordinates[0] *= (box[0, 1]-box[0, 0])/(self.resolution[0]-1.)
        local_coordinates[0] += box[0, 0]

        if self.grid_type=='cartesian':
            # Prepares an uniform cartesian grid
            x_array.set_local_data(local_coordinates[0], copy=False)
            y_array.set_local_data(local_coordinates[1], copy=False)
            z_array.set_local_data(local_coordinates[2], copy=False)

            x2y2 = local_coordinates[0]**2 + local_coordinates[1]**2
            local_r_spherical = np.sqrt(x2y2 + local_coordinates[2]**2)
            local_r_cylindrical = np.sqrt(x2y2)
            local_theta = np.arccos(local_coordinates[2]/local_r_spherical)
            local_phi = np.arctan2(local_coordinates[1], local_coordinates[0])

            r_spherical_array.set_local_data(local_r_spherical, copy=False)
            r_cylindrical_array.set_local_data(local_r_cylindrical, copy=False)
            theta_array.set_local_data(local_theta, copy=False)
            phi_array.set_local_data(local_phi, copy=False)

        elif self.grid_type=='spherical':
            # Uniform spherical grid
            r_spherical_array.set_local_data(local_coordinates[0], copy=False)
            theta_array.set_local_data(local_coordinates[1], copy=False)
            phi_array.set_local_data(local_coordinates[2], copy=False)

            local_sin_theta = np.sin(local_coordinates[1])
            local_cos_theta = np.cos(local_coordinates[1])
            local_sin_phi = np.sin(local_coordinates[2])
            local_cos_phi = np.cos(local_coordinates[2])

            local_x = local_coordinates[0] * local_sin_theta * local_cos_phi
            local_y = local_coordinates[0] * local_sin_theta * local_sin_phi
            local_z = local_coordinates[0] * local_cos_theta

            local_r_cylindrical = local_coordinates[0] * local_sin_theta

            r_cylindrical_array.set_local_data(local_r_cylindrical, copy=False)
            x_array.set_local_data(local_x, copy=False)
            y_array.set_local_data(local_y, copy=False)
            z_array.set_local_data(local_z, copy=False)

        elif self.grid_type=='cylindrical':
            # Uniform cylindrical grid
            r_cylindrical_array.set_local_data(local_coordinates[0], copy=False)
            phi_array.set_local_data(local_coordinates[1], copy=False)
            z_array.set_local_data(local_coordinates[2], copy=False)

            local_sin_phi = np.sin(local_coordinates[1])
            local_cos_phi = np.cos(local_coordinates[1])

            local_x = local_coordinates[0] * local_cos_phi
            local_y = local_coordinates[0] * local_sin_phi

            local_r_spherical = np.sqrt(local_coordinates[0]**2
                                        + local_coordinates[2]**2)
            local_theta = np.arccos(local_coordinates[2]/local_r_spherical)

            r_spherical_array.set_local_data(local_r_spherical, copy=False)
            theta_array.set_local_data(local_theta, copy=False)
            x_array.set_local_data(local_x, copy=False)
            y_array.set_local_data(local_y, copy=False)

        else:
            raise ValueError


        result_dict = {'x': x_array,
                       'y': y_array,
                       'z': z_array,
                       'r_spherical': r_spherical_array,
                       'r_cylindrical': r_cylindrical_array,
                       'theta': theta_array,
                       'phi': phi_array}

        return result_dict

    def get_prototype(self, dtype=None):
        if self._prototype_source is None:
            self._prototype_source = distributed_data_object(
                                                global_shape=self.resolution,
                                                distribution_strategy='equal',
                                                dtype=np.float)
        return self._prototype_source.copy_empty(dtype=dtype)
