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
from galmag.Grid import Grid


class B_generator(object):
    """
    Base class for B-field generators

    Note
    ----
    This class does not work on its own.
    """
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
