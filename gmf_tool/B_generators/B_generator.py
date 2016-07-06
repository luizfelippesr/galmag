# -*- coding: utf-8 -*-

import numpy as np
from gmf_tool.Grid import Grid


class B_generator(object):
    def __init__(self, box, resolution, grid_type='cartesian',
                 default_parameters={}, dtype=np.float):
        self.dtype = dtype
        self.box = np.empty((3, 2), dtype=self.dtype)
        self.resolution = np.empty((3,), dtype=np.int)

        # use numpy upcasting of scalars and dtype conversion
        self.grid_type = grid_type
        self.box[:] = box
        self.resolution[:] = resolution

        self.grid = Grid(box=self.box,
                         resolution=self.resolution,
                         grid_type=self.grid_type)

        self._init_default_parameters(default_parameters)

    @property
    def _builtin_parameter_defaults(self):
        return {}

    def _init_default_parameters(self, parameters):
        default_parameters = {}
        for [key, value] in self._builtin_parameter_defaults.iteritems():
            default_parameters[key] = parameters.get(key, value)

        self.default_parameters = default_parameters

    def _parse_parameters(self, parameters):
        parsed_parameters = {}
        for [key, value] in self.default_parameters.iteritems():
            parsed_parameters[key] = parameters.get(key, value)

        return parsed_parameters

    def get_B_field(self):
        raise NotImplementedError
