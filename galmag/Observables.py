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
Contains a class for computing a selection of observables
"""
from B_generators.B_generator import B_generator
import electron_profiles as prof
import numpy as np
import util

class Observables(B_generator):
    """
    Synchrotron emmission and Faraday rotation observables

    Class properties compute and return a range of observables computed along
    the axis specified in the initialization.

    Parameters
    ----------
    B_field : B_field
        The magnetic field based on which the observables should be computed.
        At the moment, only B_field constructed using a uniform cartesian grid
        is supported
    direction : str
        The coordinate axis parallel to the line-of-sight (i.e. the axis along
        which the integrations are performed).
        Valid values: 'x'/'edge-on', 'y' or 'z'/'face-on'.

    obs_electron_density_function : func
        A function which receives three coordinate arrays r_spherical, theta, phi
        and returns the electron density
    obs_cosmic_ray_function : func
        A function which receives three coordinate arrays r_spherical, theta, phi
        and returns the cosmic ray density
    obs_wavelength_m : float
        The wavelength of used in the synchrotron emission calculations.
        Default: 5 cm
    obs_emissivivty_normalization : float
        Needs to be adjusted
    """
    def __init__(self, B_field, default_parameters={},
                 dtype=np.float64, direction=None, **kwargs):


        if B_field.grid.grid_type != 'cartesian':
            raise NotImplementedError, 'At the moment, only cartesian grids are supported.'

        self.B_field = B_field
        resolution = B_field.resolution.copy()
        self.direction = direction
        if direction == 'x' or direction == 'edge-on':
            self._integration_axis = 0
            self._Bp = self.B_field.x
            self._depths = self.B_field.grid.x[:,0,0]
        elif direction == 'y':
            self._integration_axis = 1
            self._Bp = self.B_field.y
            self._depths = self.B_field.grid.y[0,:,0]
        elif direction == 'z' or direction == 'face-on':
            self._integration_axis = 2
            self._Bp = self.B_field.z
            self._depths = self.B_field.grid.z[0,0,:]
        else:
            raise NotImplementedError, 'Only "x", "y" and "z" directions are currently supported."'
        self._ddepth = np.abs(self._depths[0]-self._depths[1])



        resolution[self._integration_axis] = 1

        super(Observables, self).__init__(box=B_field.box,
                                          resolution=resolution,
                                          default_parameters=default_parameters,
                                          dtype=dtype)

        # Reads in the parameters
        self.parameters = self._parse_parameters(kwargs)
        # Cached quantities dictionary
        self._cache = {}

    @property
    def _builtin_parameter_defaults(self):
        builtin_defaults = {
            'obs_electron_density_function': prof.simple_ne, # n_e
            'obs_electron_at_reference_radius': 1, # n_e0 [cm^-3]
            'obs_cosmic_ray_function': prof.constant_ncr, # n_{cr} [cm^-3]
            'obs_wavelength_m': 5e-2, # 1 cm
            'obs_gamma': 1.0, # cm
            'obs_emissivivty_normalization': 1, # This needs to be updated
            }
        return builtin_defaults

    def get_B_field(self):
        """B_field based on which this Observables object was constructed"""
        return self.B_field

    @property
    def synchrotron_emissivity(self):
        r"""
        Synchrotron emissivity along a coordinate axis

        .. math::
           \epsilon = C\,  n_{cr} B_\perp^{(\gamma+1)/2)} \lambda^{(\gamma-1)/2}

        Returns
        -------
        3D-d2o
            Syncrotron emmissivity, along a coordinate axis
        """
        if 'synchrotron_emissivity' not in self._cache:
            lamb = self.parameters['obs_wavelength_m']
            gamma = self.parameters['obs_gamma']
            norm = self.parameters['obs_emissivivty_normalization']
            self._cache['synchrotron_emissivity'] = \
              self._compute_synchrotron_emissivity(lamb, gamma, norm)
        return self._cache['synchrotron_emissivity']


    def _compute_synchrotron_emissivity(self, lamb, gamma, norm):
        r"""
        Helper for synchrotron_emissivity

        Parameters
        ----------
        lamb : float
            wavelength
        gamma : float
            spectral index of the cosmic ray energy distribution
        norm : float
            normalization of the emissivity

        Returns
        -------
        synchrotron emissivity (along a given axis)
        """


        if self.direction == 'x':
            Bperp2 = self.B_field.y**2 + self.B_field.z**2
        elif self.direction == 'y':
            Bperp2 = self.B_field.x**2 + self.B_field.z**2
        elif self.direction == 'z':
            # Bperp^2 = Bx^2 + By^2
            Bperp2 = self.B_field.x**2 + self.B_field.y**2
        ncr = self.parameters['obs_cosmic_ray_function'](
                                                 self.B_field.grid.r_spherical,
                                                 self.B_field.grid.theta,
                                                 self.B_field.grid.phi)
        return ncr * Bperp2**((gamma+1)/4) * lamb**((gamma-1)/2)

    @property
    def intrinsic_polarization_degree(self):
        r"""
        Intrinsic degree of polarization

        .. math::
            p_0 = \frac{\gamma+1}{\gamma + 7/3}
        """
        if 'intrinsic_polarization_degree' not in self._cache:
            gamma = self.parameters['obs_gamma']
            self._cache['intrinsic_polarization_degree']=(gamma+1.0)/(gamma+7./3.)
        return self._cache['intrinsic_polarization_degree']


    @property
    def intrinsic_polarization_angle(self):
        r"""
        Intrinsic polarization angle

        .. math::
            \psi_0 = \frac{\pi}{2} + \tan^{-1}\left(\frac{B_y}{B_x}\right)
        """
        if 'intrinsic_polarization_angle' not in self._cache:
            if self.direction == 'x':
                B1 = self.B_field.z
                B2 = self.B_field.y
            elif self.direction == 'y':
                B1 = self.B_field.x
                B2 = self.B_field.z
            elif self.direction == 'z':
                B1 = self.B_field.y
                B2 = self.B_field.x
            else:
                raise ValueError

            psi0 = np.pi/2.0 + util.arctan2(B1,B2)
            psi0[psi0>np.pi] = psi0[psi0>np.pi]-2.*np.pi
            self._cache['intrinsic_polarization_angle'] = psi0
        return self._cache['intrinsic_polarization_angle']

    @property
    def electron_density(self):
        r"""
        Thermal electron density evaluated on this grid used for calculations.

        This is set through the parameter obs_electron_density_function,
        chosen during initialization.
        """
        if 'electron_density' not in self._cache:
            # Gets local grid (aka beginning of d2o gymnastics)
            local_r_sph_grid = self.B_field.grid.r_spherical.get_local_data()
            local_theta_grid = self.B_field.grid.theta.get_local_data()
            local_phi_grid = self.B_field.grid.phi.get_local_data()
            ne0 = self.parameters['obs_electron_at_reference_radius']
            # Evaluate obs_electron_density_function on the local grid
            local_ne = self.parameters['obs_electron_density_function'](
                                                              local_r_sph_grid,
                                                              local_theta_grid,
                                                              local_phi_grid,
                                                              ne0=ne0)

            # Initializes global array and set local data into a d2o
            global_ne = self.B_field.grid.get_prototype(dtype=self.dtype)
            global_ne.set_local_data(local_ne, copy=False)
            self._cache['electron_density'] = global_ne
        return self._cache['electron_density']

    @property
    def psi(self):
        r"""
        Polarization angle of radiation emitted at a given depth Faraday
        rotated over the line of sight.

        .. math::
            \psi(z) = \psi_0(z)
            + 0.812\,{\rm rad}\left(\frac{\lambda}{1\,\rm m}\right)^2 \int_z^\infty
            \left(\frac{n_e(z')}{1\,\rm cm^{-3}}\right)
            \left(\frac{B_\parallel(z')}{1\,\mu\rm G}\right)
            \frac{{\rm d} z'}{1\,\rm pc}
        """

        if 'psi' not in self._cache:
            lamb = self.parameters['obs_wavelength_m']

            ne = self.electron_density

            self._cache['psi'] = self._compute_psi(lamb, ne)
        return self._cache['psi']


    def _compute_psi(self, lamb, ne, from_bottom=False):
        """Computes the Faraday rotated polarization angle of radiation emmited
        at each depth

        Parameters
        ----------
        lamb : float
            Wavelength used for the computation (in meters)

        ne : 3D d2o
            Array containing the electron density in the galaxy

        from_bottom : bool
            Whether the observation is done "from bottom" (integration starting
            from  negative values) or "from top" (positive). Default: False
        """
        Bp = self._Bp
        ddepth = self._ddepth * 1000

        axis = [slice(None),slice(None),slice(None)]
        ax_n = self._integration_axis
        slices = [slice(None),slice(None),slice(None)]

        psi = self.intrinsic_polarization_angle.copy()

        for i, depth in enumerate(self._depths):
            axis[ax_n] = slice(i,i+1) # e.g. for z, this will select psi[:,:,i]

            # The observer can be at the top or bottom
            if not from_bottom:
                # Will integrate from 0 to i; e.g. for z, Bp[:,:,0:i]
                slices[ax_n] = slice(0,i)
            else:
                # Will integrate from i to the end
                slices[ax_n] = slice(i,-1)

            integrand = ne[slices] * Bp[slices] * ddepth
            integral = integrand.sum(axis=ax_n)
            # Adjust the axes and uses full data (to make sure to avoid mistakes)
            integral = np.expand_dims(integral.get_full_data(), ax_n)
            # psi(z) = psi0(z) + 0.812\lambda^2 \int_-\infty^z ne(z') Bpara(z') dz'
            psi[axis] += 0.812*lamb**2 * integral

        return psi


    @property
    def Stokes_I(self):
        r"""
        Stokes parameter I (total emmission)

        .. math::
           I(\lambda) = \int_{-\infty}^\infty \epsilon(w,\lambda) dw

        where :math:`w` is the specified coordinate axis.
        """
        if 'Stokes_I' not in self._cache:
            self._cache['Stokes_I'] = self._compute_Stokes('I')

        return self._cache['Stokes_I']


    @property
    def Stokes_Q(self):
        r"""
        Stokes parameter Q

        .. math::
           Q(\lambda) = \int_{-\infty}^\infty \epsilon(w,\lambda) p_0(w) \cos[2 \psi(w)] dw

        where :math:`w` is the specified coordinate axis.
        """
        if 'Stokes_Q' not in self._cache:
            self._cache['Stokes_Q'] = self._compute_Stokes('Q')

        return self._cache['Stokes_Q']


    @property
    def Stokes_U(self):
        r"""
        Stokes parameter U

        .. math::
           U(\lambda) = \int_{-\infty}^\infty \epsilon(w,\lambda) p_0(w) \sin[2 \psi(w)] dw

        where :math:`w` is the specified coordinate axis.
        """
        if 'Stokes_U' not in self._cache:
            self._cache['Stokes_U'] = self._compute_Stokes('U')

        return self._cache['Stokes_U']


    def _compute_Stokes(self, parameter):
        r"""
        Computes Stokes parameters I, Q, U

        .. math::
           I(\lambda) = \int_{-\infty}^\infty \epsilon(w,\lambda) dw
        .. math::
           Q(\lambda) = \int_{-\infty}^\infty \epsilon(w,\lambda) p_0(w) \cos[2 \psi(w)] dw
        .. math::
           U(\lambda) = \int_{-\infty}^\infty \epsilon(w,\lambda) p_0(w) \sin[2 \psi(w)] dw

        where :math:`w` is the specified coordinate axis.

        Parameters
        ----------
        parameter : str
          Either 'I', 'Q' or 'U'

        Returns
        -------
        The specified Stokes parameter
        """

        emissivity = self.synchrotron_emissivity

        # Computes the integrand
        if parameter == 'I':
            integrand = emissivity * self._ddepth
        elif parameter == 'Q':
            p0 = self.intrinsic_polarization_degree
            cos2psi = util.distribute_function(np.cos, 2.0*self.psi)
            integrand = emissivity * p0 * cos2psi * self._ddepth
        elif parameter == 'U':
            p0 = self.intrinsic_polarization_degree
            sin2psi = util.distribute_function(np.sin, 2.0*self.psi)
            integrand = emissivity * p0 * sin2psi * self._ddepth
        else:
            raise ValueError

        # Sums/integrates along the specified axis and returns
        return integrand.sum(axis=self._integration_axis)


    @property
    def observed_polarization_angle(self):
        r"""
        Observed integrated polarization angle

        .. math::
           \Psi = \frac{1}{2} \arctan\left(\frac{U}{Q}\right)
        """
        if 'observed_polarization_angle' not in self._cache:
            angle = 0.5 * util.arctan2(self.Stokes_U,self.Stokes_Q)
            self._cache['observed_polarization_angle'] = angle
        return self._cache['observed_polarization_angle']


    @property
    def polarized_intensity(self):
        r"""
        Polarized intensity

        .. math::
           P = \sqrt{Q^2 + U^2}
        """
        if 'polarized_intensity' not in self._cache:
            P = (self.Stokes_U**2 + self.Stokes_Q**2 )**0.5
            self._cache['polarized_intensity'] = P
        return self._cache['polarized_intensity']

    @property
    def rotation_measure(self):
        r"""
        Rotation measure

        .. math::
            RM = (0.812\,{\rm rad\,m}^{-2}) \int \frac{n_e(z)}{1\,\rm cm^{-3}}
                                                 \frac{B_z}{1\,\mu\rm G}
                                                 \frac{{\rm d} z}{1\,\rm pc}
        """

        if 'RM' not in self._cache:
            ne = self.electron_density
            self._cache['RM'] = self._compute_RM(ne)

        return self._cache['RM']


    def _compute_RM(self, ne, from_bottom=False):
        """
        Computes the Faraday rotation measure

        Parameters
        ----------
        ne : 3D d2o
            Array containing the electron density in the galaxy (in cm^{-3})
        """
        Bp = self._Bp
        ddepth = self._ddepth * 1000 # Converts from
        ax_n = self._integration_axis
        print 'RM details'
        print 'ddepth',ddepth
        print 'ne', ne.max(), ne.min(), ne.mean()
        integrand = ne * Bp * ddepth

        return 0.812*integrand.sum(axis=ax_n)

