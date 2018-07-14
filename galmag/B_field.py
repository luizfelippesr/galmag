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
Contains the definition of the B_field and B_field_component classes.
"""
import numpy as np
from galmag.Grid import Grid

class B_field_component(object):
    """
    A single galactic magnetic field component.

    `B_field_component` objects store data for a single galactic magnetic
    field component (e.g. the galactic *halo* magnetic field).

    The properties x, y, z, r_spherical, r_cylindrical, theta and phi return
    the magnetic field along each of these axis (with any required coordinate
    transformation being performed on the fly).

    Parameters
    ----------
    grid : grid_object
        See :class: `galmag.Grid`
    x, y, z, r_spherical, r_cylindrical, theta, phi : 3D array_like
        Values of the magnetic field in coordinates compatible with the grid
        parameter should be supplied at initialization.
        Unused coordinates should be set to None.
        E.g. r_cylindrical=[[..],[..],[..]], z=[[..],[..],[..]],
        phi=[[..],[..],[..]]
    copy : bool
        whether a fresh copy of the field data should be made upon
        initialization. Default: True
    """
    def __init__(self, grid, x=None, y=None, z=None, r_spherical=None,
                 r_cylindrical=None, theta=None, phi=None, copy=True,
                 dtype=np.dtype(np.float), generator=None, parameters={}):

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
        """Horizontal coordinate component, :math:`x`"""
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
            elif (self._phi is not None and self._r_cylindrical is not None):
                self._x = (self.r_cylindrical * self.grid.cos_phi
                           - self.phi * self.grid.sin_phi)
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
        """Horizontal coordinate component, :math:`y`"""
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

            elif (self._phi is not None and self._r_cylindrical is not None):
                self._y = (self.r_cylindrical * self.grid.sin_phi
                           + self.phi * self.grid.cos_phi)

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
        """Vertical coordinate component, :math:`z`"""
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
        """Spherical radial coordinate component, :math:`r`"""
        if self._r_spherical is None:
            # 1/r(V_x*x + V_y*y + V_z*z)
            self._r_spherical = (self.x*self.grid.x +
                                 self.y*self.grid.y +
                                 self.z*self.grid.z)/self.grid.r_spherical
        return self._r_spherical

    @r_spherical.setter
    def r_spherical(self, data):
        self.set_field_data('r_spherical', data)

    @property
    def r_cylindrical(self):
        """Cylindrical radial coordinate component, :math:`s`"""
        if self._r_cylindrical is None:
            # 1/r(V_x*x + V_y*y)
            self._r_cylindrical = ((self.x*self.grid.x +
                                    self.y*self.grid.y) /
                                    self.grid.r_cylindrical)
        return self._r_cylindrical

    @r_cylindrical.setter
    def r_cylindrical(self, data):
        self.set_field_data('r_cylindrical', data)

    @property
    def theta(self):
        r"""Polar coordinate component, :math:`\theta`"""
        if self._theta is None:
            # V_x*cos(phi)*cos(theta) + V_y*sin(phi)*cos(theta) -
            # V_z*sin(theta)
            cos_theta = self.grid.cos_theta
            self._theta = (self.x*self.grid.cos_phi*cos_theta +
                            self.y*self.grid.sin_phi*cos_theta -
                              self.z*self.grid.sin_theta)
        return self._theta

    @theta.setter
    def theta(self, data):
        self.set_field_data('theta', data)

    @property
    def phi(self):
        r"""Azimuthal coordinate component, :math:`\phi`"""
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
        """
        Includes fresh data in a particular component

        Parameters
        ----------
        name : str
            Name of the coordinate component
        data : array_like
            Data
        copy : bool, optional
            Copy (usual) or reference the data? Default: True.
        """
        internal_field = getattr(self, "_"+name)

        if data is None:
            internal_field = None
        else:
            if internal_field is None:
                setattr(self, '_'+name,
                        self.grid.get_prototype(dtype=self.dtype))
                internal_field = getattr(self, "_"+name)
            internal_field.set_full_data(data, copy=copy)


class B_field(object):
    """
    Galactic magnetic field

    `B_field` objects store data for the whole galactic magnetic field.

    Individual galactic magnetic field components (e.g. disk or halo) are
    stored as attributes containing `B_field_component` objects.

    The properties x, y, z, r_spherical, r_cylindrical, theta and phi return
    the magnetic field along each of these axis. These are actually the sum of
    all the galactic components available.

    A disk or halo component can be added using the `add_disk_field` and
    `add_halo_field` methods.

    A custom new magnetic field component can be set using the
    `set_field_component` method. Alternatively, the custom component can be
    added during initialization as an extra keyword
    (i.e. B_field.set_field_component(name, component) is equivalent to
    B_field(..., name=component).

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
    def __init__(self, box, resolution, grid_type='cartesian',
                 dtype=np.float, **kwargs):

        self.dtype = dtype
        self.box = np.empty((3, 2), dtype=self.dtype)
        self.resolution = np.empty((3,), dtype=np.int)
        # Uses numpy upcasting of scalars and dtype conversion
        self.grid_type = grid_type
        self.box[:] = box
        self.resolution[:] = resolution

        self.grid = Grid(box=self.box,
                         resolution=self.resolution,
                         grid_type=self.grid_type)

        # If the user supplies pre-generated field components, add them
        # to a list containing components names (for internal use)
        self._components = []
        self.parameters = {}

        for key in kwargs:
            assert(isinstance(kwargs[key], B_field_component))
            self.set_field_component(key, kwargs[key])

        # Includes the coordinate component attributes
        for component in ['x', 'y', 'z', 'r_spherical', 'r_cylindrical',
                          'theta', 'phi']:
            setattr(self, '_'+component, None)

    @property
    def x(self):
        """Horizontal coordinate component, :math:`x`"""
        if self._x is None:
            self._set_data('x')
        return self._x

    @property
    def y(self):
        """Horizontal coordinate component, :math:`y`"""
        if self._y is None:
            self._set_data('y')
        return self._y

    @property
    def z(self):
        """Vertical coordinate component, :math:`z`"""
        if self._z is None:
            self._set_data('z')
        return self._z

    @property
    def r_spherical(self):
        """Spherical radial coordinate component, :math:`r`"""
        if self._r_spherical is None:
            self._set_data('r_spherical')
        return self._r_spherical

    @property
    def r_cylindrical(self):
        """Cylindrical radial coordinate component, :math:`s`"""
        if self._r_cylindrical is None:
            self._set_data('r_cylindrical')
        return self._r_cylindrical

    @property
    def theta(self):
        r"""Polar coordinate component, :math:`\theta`"""

        if self._theta is None:
            self._set_data('theta')
        return self._theta

    @property
    def phi(self):
        r"""Azimuthal coordinate component, :math:`\phi`"""

        if self._phi is None:
            self._set_data('phi')
        return self._phi

    def add_disk_field(self, name='disk', **kwargs):
        """
        Includes a disc magnetic field component.

        Parameters
        ----------
        name : str, optional
            Name of the disc component. Default value: 'disk'
        **kwargs :
            See :class:`galmag.B_generator_disk` for the list of disc parameters
        """
        import galmag.B_generators as Bgen
        # First, prepares the generator
        Bgen_disk = Bgen.B_generator_disk(grid=self.grid)
        # Decides between find_B_field and get_B_field and computes
        if 'reversals' in kwargs:
            component = Bgen_disk.find_B_field(**kwargs)
        elif 'disk_modes_normalization' in kwargs:
            component = Bgen_disk.get_B_field(**kwargs)
        else:
            raise ValueError, 'Must specify either the positions of the reversals or the disk_modes_normalization.'
        self.set_field_component(name, component)

    def add_halo_field(self, name='halo', **kwargs):
        """

        Parameters
        ----------
        name :
            (Default value = 'halo')
        **kwargs :


        Returns
        -------

        """
        import galmag.B_generators as Bgen
        # First, prepares the generator
        Bgen_halo = Bgen.B_generator_halo(grid=self.grid)
        # Gets the field
        component = Bgen_halo.get_B_field(**kwargs)
        self.set_field_component(name, component)


    def set_field_component(self, name, component):
        """

        Parameters
        ----------
        name :

        component :


        Returns
        -------

        """
        # Checks whether grid settings are compatible
        if (np.any(component.grid.box != self.box) or
            np.any(component.grid.resolution != self.resolution)):
            raise ValueError, 'Incompatible grid geometry.'
        # Adds the component (overwrites if it existent)
        if name not in self._components:
            self._components.append(name)
        self.reset_cache()
        setattr(self, name, component)
        self.parameters.update(component.parameters)

    def reset_cache(self):
        """
        Erases cached total field values. Major components as 'disk' or 'halo'
        are preserved and coordinate component values will be generated on next
        time they are called.
        """
        for component in ['x', 'y', 'z', 'r_spherical', 'r_cylindrical',
                          'theta', 'phi']:
            setattr(self, '_'+component, None)

    def _set_data(self, name):
        """
        Updates the a coordinate component with the information in the various
        field components (e.g. B.x = B.disk.x + B.halo.x)

        Parameters
        ----------
        name : str
            Name of the coordinate component
        """
        internal_field = None
        for component in self._components:
            component_field = getattr(self, component)
            component_field_values = getattr(component_field, name)
            if internal_field is None:
                internal_field = component_field_values.copy()
            else:
                internal_field += component_field_values

        setattr(self, '_'+name, internal_field)
