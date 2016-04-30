# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import jn_zeros

from B_generator import B_generator
from gmf_tool.core.rotation_curve import simple_radial_Shear


class B_generator_disk(B_generator):
    def __init__(self, box, resolution, default_parameters=None,
                 shear_function=simple_radial_Shear):

        super(B_generator_disk, self).__init__(
                                        box=box,
                                        resolution=resolution,
                                        default_parameters=default_parameters)

        assert(callable(shear_function))
        self.shear_function = shear_function

        self._kns = jn_zeros(
                         1, len(self.default_parameters['disk_normalization']))

    @property
    def _builtin_parameter_defaults(self):
        defaults = {'disk_height': 0.4,  # h_d
                    'disk_radius': 20,  # R_d
                    'disk_induction': 0.6,  # Ralpha_d
                    'disk_normalization': np.array([1., 1., 1.]),  # Cn_d
                    'disk_dynamo_number': -20,  # D_d
                    'disk_rotation_curve_V0': 1.,
                    'disk_rotation_curve_s0': 1.,
                    }
        return defaults

    def get_B_field(self):
        pass

    def _convert_coordinate_to_B_value(self):
        pass