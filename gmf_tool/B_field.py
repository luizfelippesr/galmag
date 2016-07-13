# -*- coding: utf-8 -*-

import numpy as np

from gmf_tool.Grid import Grid


class B_field(object):
    def __init__(self, grid, x=None, y=None, z=None, r_spherical=None,
                 r_cylindrical=None, theta=None, phi=None, copy=True,
                 dtype=np.dtype(np.float), generator=None, parameters=None):

        assert(isinstance(grid, Grid))
        self.grid = grid

        self.dtype = np.dtype(dtype)

        self.generator = generator
        self.parameters = parameters

        for component in ['x', 'y', 'z', 'r_spherical', 'r_cylindrical',
                          'theta', 'phi']:
            data = eval(component)
            setattr(self, '_'+component, None)
            self.set_field_data(component, data, copy=copy)

    @property
    def x(self):
        if self._x is None:
            raiseQ = False
            if (self._phi is not None and self._theta is not None):
                # V_r*x/r - V_phi*sin(phi) + V_theta*cos(phi)*cos(theta)
                if self._r_spherical is not None:
                    r_contribution = (self.r_spherical * self.grid.x /
                                      self.grid.r_spherical)

                elif self._r_cylindrical is not None:
                    r_contribution = (self.r_cylindrical * self.grid.x /
                                      self.grid.r_cylindrical)
                else:
                    raiseQ = True

                self._x = (r_contribution -
                           self.phi * self.grid.sin_phi +
                           self.theta * self.grid.cos_phi *
                           self.grid.cos_theta)
            else:
                raiseQ = True
            if raiseQ:
                raise ValueError(
                        "ERROR: x is neither directly nor indirectly defined.")
        return self._x

    @x.setter
    def x(self, data):
        self.set_field_data('x', data)

    @property
    def y(self):
        if self._y is None:
            raiseQ = False
            if (self._phi is not None and self._theta is not None):
                # V_r*y/r + V_phi*cos(phi) + V_theta*sin(phi)*cos(theta)
                if self._r_spherical is not None:
                    r_contribution = (self.r_spherical * self.grid.y /
                                      self.grid.r_spherical)
                elif self._r_cylindrical is not None:
                    r_contribution = (self.r_cylindrical * self.grid.y /
                                      self.grid.r_cylindrical)
                else:
                    raiseQ = True

                self._y = (r_contribution +
                           self.phi * self.grid.cos_phi +
                           self.theta * self.grid.sin_phi *
                           self.grid.cos_theta)
            else:
                raiseQ = True
            if raiseQ:
                raise ValueError(
                        "ERROR: y is neither directly nor indirectly defined.")
        return self._y

    @y.setter
    def y(self, data):
        self.set_field_data('y', data)

    @property
    def z(self):
        if self._z is None:
            raiseQ = False
            if self._theta is not None:
                # V_r*z/r - V_theta*sin(theta)
                if self._r_spherical is not None:
                    r_contribution = (self.r_spherical * self.grid.z /
                                      self.grid.r_spherical)
                elif self._r_cylindrical is not None:
                    r_contribution = (self.r_cylindrical * self.grid.z /
                                      self.grid.r_cylindrical)
                else:
                    raiseQ = True

                self._z = (r_contribution - self.theta * self.grid.sin_theta)
            else:
                raiseQ = True
            if raiseQ:
                raise ValueError(
                        "ERROR: z is neither directly nor indirectly defined.")
        return self._z

    @z.setter
    def z(self, data):
        self.set_field_data('z', data)

    @property
    def r_spherical(self):
        if self._r_spherical is None:
            if (self._x is not None and self._y is not None and
                    self._z is not None):
                # 1/r(V_x*x + V_y*y + V_z*z)
                self._r_spherical = (self.x*self.grid.x +
                                     self.y*self.grid.y +
                                     self.z*self.grid.z)/self.grid.r_spherical
            else:
                raise ValueError(
                    "ERROR: r_spherical is neither directly nor indirectly " +
                    "defined.")
        return self._r_spherical

    @r_spherical.setter
    def r_spherical(self, data):
        self.set_field_data('r_spherical', data)

    @property
    def r_cylindrical(self):
        if self._r_cylindrical is None:
            if (self._x is not None and self._y is not None):
                # 1/r(V_x*x + V_y*y)
                self._r_cylindrical = ((self.x*self.grid.x +
                                       self.y*self.grid.y) /
                                       self.grid.r_cylindrical)
            else:
                raise ValueError(
                    "ERROR: r is neither directly nor indirectly defined.")
        return self._r_cylindrical

    @r_cylindrical.setter
    def r_cylindrical(self, data):
        self.set_field_data('r_cylindrical', data)

    @property
    def theta(self):
        if self._theta is None:
            if (self._x is not None and self._y is not None and
                    self._z is not None):
                # V_x*cos(phi)*cos(theta) + V_y*sin(phi)*cos(theta) -
                # V_z*sin(theta)
                cos_theta = self.grid.cos_theta
                self._theta = (self.x*self.grid.cos_phi*cos_theta +
                               self.y*self.grid.sin_phi*cos_theta -
                               self.z*self.grid.sin_theta)
            else:
                raise ValueError(
                    "ERROR: theta is neither directly nor indirectly defined.")
        return self._theta

    @theta.setter
    def theta(self, data):
        self.set_field_data('theta', data)

    @property
    def phi(self):
        if self._phi is None:
            if (self._x is not None and self._y is not None):
                # -V_x*sin(phi) + V_y*cos(phi)
                self._phi = (-self.x*self.grid.sin_phi +
                             self.y*self.grid.cos_phi)
            else:
                raise ValueError(
                    "ERROR: phi is neither directly nor indirectly defined.")
        return self._phi

    @phi.setter
    def phi(self, data):
        self.set_field_data('phi', data)

    def set_field_data(self, name, data, copy=True):
        internal_field = getattr(self, "_"+name)

        if data is None:
            internal_field = None
        else:
            if internal_field is None:
                setattr(self, '_'+name,
                        self.grid.get_prototype(dtype=self.dtype))
                internal_field = getattr(self, "_"+name)
            internal_field.set_full_data(data, copy=copy)



