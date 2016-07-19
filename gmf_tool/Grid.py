# -*- coding: utf-8 -*-

import numpy as np
from d2o import distributed_data_object


class Grid(object):
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
        if self._coordinates is None:
            self._coordinates = self._generate_coordinates()
        return self._coordinates

    @property
    def x(self):
        return self.coordinates['x']

    @property
    def y(self):
        return self.coordinates['y']

    @property
    def z(self):
        return self.coordinates['z']

    @property
    def r_spherical(self):
        return self.coordinates['r_spherical']

    @property
    def r_cylindrical(self):
        return self.coordinates['r_cylindrical']

    @property
    def theta(self):
        return self.coordinates['theta']

    @property
    def phi(self):
        return self.coordinates['phi']

    @property
    def sin_theta(self):
        return self.r_cylindrical / self.r_spherical

    @property
    def cos_theta(self):
        return self.z / self.r_spherical

    @property
    def sin_phi(self):
        return self.y / self.r_cylindrical

    @property
    def cos_phi(self):
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
