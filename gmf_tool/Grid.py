# -*- coding: utf-8 -*-

import numpy as np
from d2o_injector import d2o
from d2o import distributed_data_object


class Grid(object):
    def __init__(self, box, resolution):

        self.box = np.empty((3, 2), dtype=np.float)
        self.resolution = np.empty((3,), dtype=np.int)

        # use numpy upcasting of scalars and dtype conversion
        self.box[:] = box
        self.resolution[:] = resolution

        self._coordinates = None

    @property
    def coordinates(self):
        if self._coordinates is None:
            self._coordinates = self._generate_coordinates()
        return self._coordinates

    @property
    def x(self):
        return self.coordinates[0]

    @property
    def y(self):
        return self.coordinates[1]

    @property
    def z(self):
        return self.coordinates[2]

    @property
    def r(self):
        return self.coordinates[3]

    @property
    def theta(self):
        return self.coordinates[4]

    @property
    def phi(self):
        return self.coordinates[5]

    def _generate_coordinates(self):
        x_array = distributed_data_object(
                            global_shape=self.resolution,
                            distribution_strategy='equal',
                            dtype=np.float)
        [y_array, z_array, r_array, theta_array, phi_array] = \
            [x_array.copy_empty() for i in xrange(5)]

        local_start = x_array.distributor.local_start
        local_end = x_array.distributor.local_end
        box = self.box
        local_slice = (slice(local_start, local_end),
                       slice(box[1, 0], box[1, 1], self.resolution[1]*1j),
                       slice(box[2, 0], box[2, 1], self.resolution[2]*1j))

        local_coordinates = np.mgrid[local_slice]

        local_coordinates[0] *= (box[0, 1]-box[0, 0])/(self.resolution[0]-1.)
        local_coordinates[0] += box[0, 0]

        local_r = np.sqrt(local_coordinates[0]**2 + local_coordinates[1]**2)
        local_theta = np.arccos(local_coordinates[2]/local_r)
        local_phi = np.arctan2(local_coordinates[1], local_coordinates[0])

        x_array.set_local_data(local_coordinates[0], copy=False)
        y_array.set_local_data(local_coordinates[1], copy=False)
        z_array.set_local_data(local_coordinates[2], copy=False)
        r_array.set_local_data(local_r, copy=False)
        theta_array.set_local_data(local_theta, copy=False)
        phi_array.set_local_data(local_phi, copy=False)

        return [x_array, y_array, z_array, r_array, theta_array, phi_array]
