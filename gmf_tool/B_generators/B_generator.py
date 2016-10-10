# -*- coding: utf-8 -*-

import numpy as np
from gmf_tool.Grid import Grid


class B_generator(object):
    def __init__(self, grid=None, box=None, resolution=None,
                 grid_type='cartesian', default_parameters={},
                 dtype=np.float):

        self.dtype = dtype
        self._init_default_parameters(default_parameters)

        if grid is not None:
            self.grid = grid
            self.box = grid.box
            self.resolution = grid.resolution
            self.grid_type = grid.grid_type

        elif box is not None and resolution is not None:
            self.box = np.empty((3, 2), dtype=self.dtype)
            self.resolution = np.empty((3,), dtype=np.int)

            # use numpy upcasting of scalars and dtype conversion
            self.grid_type = grid_type
            self.box[:] = box
            self.resolution[:] = resolution

            self.grid = Grid(box=self.box,
                            resolution=self.resolution,
                          grid_type=self.grid_type)
        else:
            raise ValueError, 'Must specify either a valid Grid object or its properties (box and resolution).'

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
