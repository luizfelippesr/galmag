# -*- coding: utf-8 -*-

import numpy as np
from gmf_tool.Grid import Grid


class B_generator(object):
    def __init__(self, box, resolution, default_parameters=None):
        self.box = np.empty((3, 2), dtype=np.float)
        self.resolution = np.empty((3,), dtype=np.int)

        # use numpy upcasting of scalars and dtype conversion
        self.box[:] = box
        self.resolution[:] = resolution

        self.grid = Grid(box=self.box,
                         resolution=self.resolution)

        self._builtin_parameter_defaults = \
            {'disk_height': 0.4,  # h_d
             'disk_radius': 20,  # R_d
             'disk_induction': 0.6,  # Ralpha_d
             'disk_normalization': np.array([1., 1., 1.]),  # Cn_d
             'disk_dynamo_number': -20,  # D_d
             'disk_rotation_curve_V0': 1.,
             'disk_rotation_curve_s0': 1.,
             }

        self._init_default_parameters(default_parameters)

    @property
    def _builtin_parameter_defaults(self):
        return {}

    def _init_default_parametesrs(self, parameters):
        default_parameters = {}
        for [key, value] in self._builtin_parameter_defaults.iteritems():
            default_parameters = parameters.get(key, value)

        self.default_parameters = default_parameters

    def get_B_field(self):
        raise NotImplementedError
