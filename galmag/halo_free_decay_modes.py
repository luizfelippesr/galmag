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
Contains functions to compute the free decay modes of the magnetic
of the halo of a galaxy.

They should be accessed through the function :func:`get_mode`.
"""
from scipy.special import j0, j1, jv, jn_zeros
import numpy as N
from sympy import besselj
from mpmath import mp, findroot
import os.path
pi=N.pi
cos = N.cos
tan = N.tan
sin = N.sin
sqrt = N.sqrt

def get_B_a_1(r, theta, phi, C=0.346, k=pi):
    r"""
    Computes the first (pure poloidal) antisymmetric free decay mode.
    Purely poloidal.

    Parameters
    ----------
    r : array_like
        NxNxN array containing azimuthal coordinates.
    theta : array_like
        NxNxN array containing azimuthal coordinates.
    phi : array_like
        NxNxN array containing azimuthal coordinates.
    C : float, optional
        Mode normalization.
    k : float, optional
        :math:`k`

    Returns
    -------
    list
        List containing three `numpy` arrays corresponding to:
        :math:`B_r`, :math:`B_\theta`, :math:`B_\phi`
    """
    # Computes radial component
    Q = N.empty_like(r)
    separator = r <=1
    Q[separator] = r[separator]**(-0.5)*jv(3.0/2.0,k*r[separator])
    Q[~separator] = r[~separator]**(-2.0)*jv(3.0/2.0,k)

    Br = C*(2.0/r)*Q*cos(theta)

    # Computes polar component
    # X = d(rQ1)/dr
    X = N.empty_like(r)
    y = k*r[separator]
    X[separator] = sqrt(2.0/pi)*(y**2*sin(y) - sin(y) +
                                 y*cos(y)) / (k**(-0.5)*y**2)
    X[~separator] = -1.*r[~separator]**(-2.0)*jv(3.0/2.0,k)

    Btheta = C*(-sin(theta)/r)*X

    # Sets azimuthal component
    Bphi   = N.zeros_like(r)

    return Br, Btheta, Bphi

def get_B_a_2(r, theta, phi, C=0.250, k=5.763):
    r"""
    Computes the second antisymmetric free decay mode - one of a
    degenerate pair, with eigenvalue :math:`\gamma_2=-(5.763)^2`.
    Purely poloidal.

    Parameters
    ----------
    r : array_like
        NxNxN array containing azimuthal coordinates.
    theta : array_like
        NxNxN array containing azimuthal coordinates.
    phi : array_like
        NxNxN array containing azimuthal coordinates.
    C : float, optional
        Mode normalization.
    k : float, optional
        :math:`k`

    Returns
    -------
    list
        List containing three `numpy` arrays corresponding to:
        :math:`B_r`, :math:`B_\theta`, :math:`B_\phi`
    """

    # Computes radial component
    separator = r <=1.
    Q = N.empty_like(r)
    Q[separator] = r[separator]**(-0.5)*jv(7.0/2.0, k*r[separator])
    Q[~separator] = r[~separator]**(-4.0)*jv(7.0/2.0, k)

    Br = Q*cos(theta)*(5.0*cos(2.*theta)-1.)*C*(2.0/r)

    # Computes polar component
    # X = d(rQ1)/dr
    # r<=1, first define some auxiliary quantities
    y = k*r[separator]
    X = N.empty_like(r)

    X[separator]  = k**(0.5)*y**(0.5)*(jv(2.5,y) - jv(4.5,y))/2.0 \
                    + 0.5*r[separator]**(-0.5)*jv(3.5,y)
    X[~separator] = -3*jv(3.5,k)*r[~separator]**(-4)


    # Sets Btheta
    Btheta = C*(-sin(theta)/r)*(5.*(cos(theta))**2-1.)*X

    # Sets azimuthal component
    Bphi   = N.zeros_like(r)

    return Br, Btheta, Bphi


def get_B_a_3(r, theta, phi, C=3.445, k=5.763):
    r"""
    Computes the third antisymmetric free decay mode - one of a
    degenerate pair, with eigenvalue :math:`\gamma_2=-(5.763)^2`.
    Purely toroidal.

    Parameters
    ----------
    r : array_like
        NxNxN array containing azimuthal coordinates.
    theta : array_like
        NxNxN array containing azimuthal coordinates.
    phi : array_like
        NxNxN array containing azimuthal coordinates.
    C : float, optional
        Mode normalization.
    k : float, optional
        :math:`k`

    Returns
    -------
    list
        List containing three `numpy` arrays corresponding to:
        :math:`B_r`, :math:`B_\theta`, :math:`B_\phi`
    """
    separator = r <=1.

    # Sets radial component
    Br = N.zeros_like(r)

    # Sets polar component
    Btheta = N.zeros_like(r)

    # Computes azimuthal component
    Q = N.empty_like(r)
    Q[separator] = r[separator]**(-0.5)*jv(5.0/2.0, k*r[separator])
    Q[~separator] = r[~separator]**(-3.0)*jv(5.0/2.0, k)

    Bphi = C*Q*sin(theta)*cos(theta)

    return Br, Btheta, Bphi


def get_B_a_4(r, theta, phi, C=0.244, k=(2.*pi)):
    r"""
    Computes the forth antisymmetric free decay mode.
    Purely poloidal.

    Parameters
    ----------
    r : array_like
        NxNxN array containing azimuthal coordinates.
    theta : array_like
        NxNxN array containing azimuthal coordinates.
    phi : array_like
        NxNxN array containing azimuthal coordinates.
    C : float, optional
        Mode normalization.
    k : float, optional
        :math:`k`

    Returns
    -------
    list
        List containing three `numpy` arrays corresponding to:
        :math:`B_r`, :math:`B_\theta`, :math:`B_\phi`
    """
    # This happens to have the same form as the 1st antisymmetric mode
    return get_B_a_1(r, theta, phi, C=C, k=k)


def get_B_s_1(r, theta, phi, C=0.653646562698, k=4.493409457909):
    r"""
    Computes the first (poloidal) symmetric free decay mode.
    Purely poloidal.

    Parameters
    ----------
    r : array_like
        NxNxN array containing azimuthal coordinates.
    theta : array_like
        NxNxN array containing azimuthal coordinates.
    phi : array_like
        NxNxN array containing azimuthal coordinates.
    C : float, optional
        Mode normalization.
    k : float, optional
        :math:`k`

    Returns
    -------
    list
        List containing three `numpy` arrays corresponding to:
        :math:`B_r`, :math:`B_\theta`, :math:`B_\phi`
    """
    separator = r <=1.

    # Computes radial component
    Q = N.empty_like(r)
    Q[separator] = r[separator]**(-0.5)*jv(5.0/2.0,k*r[separator])
    Q[~separator]  = r[~separator]**(-3.0)*jv(5.0/2.0,k)

    Br = C*Q*(3.0*cos(theta)**2-1)/r

    # Computes theta component
    X = N.zeros_like(r)
    y = k*r[separator]
    siny = sin(y)
    cosy = cos(y)
    X[separator] = (3*siny/y**2 - siny - 3.*cosy/y)/sqrt(2*pi*y*r[r<=1]) \
      - k*sqrt(r[r<=1])*(3.*siny/y**2-siny-3.*cosy/y)/sqrt(2*pi)*y**(-3./2.)\
      + sqrt(2./pi*r[r<=1])*(-6.*siny*k/y**3+6.*cosy*k/y**2+3*siny*k/y-k*cosy)/(
                                                                        sqrt(y))
    X[~separator] = -2.0*r[~separator]**(-3.0)*jv(5.0/2.0,k)

    Btheta = C*(-sin(theta)*cos(theta)/r)*X

    # Sets azimuthal component
    Bphi   = N.zeros_like(r)

    return Br, Btheta, Bphi


def get_B_s_2(r, theta, phi, C=1.32984358196, k=4.493409457909):
    r"""
    Computes the second symmetric free decay mode (one of a
    degenerate pair, with eigenvalue gamma_2=-(5.763)^2 .
    Purely toroidal.

    Parameters
    ----------
    r : array_like
        NxNxN array containing azimuthal coordinates.
    theta : array_like
        NxNxN array containing azimuthal coordinates.
    phi : array_like
        NxNxN array containing azimuthal coordinates.
    C : float, optional
        Mode normalization.
    k : float, optional
        :math:`k`

    Returns
    -------
    list
        List containing three `numpy` arrays corresponding to:
        :math:`B_r`, :math:`B_\theta`, :math:`B_\phi`
    """
    separator = r <=1.

    # Sets radial component
    Br = N.zeros_like(r)

    # Sets theta component
    Btheta = N.zeros_like(r)

    # Computes azimuthal component
    Q = N.empty_like(r)
    Q[separator] = r[separator]**(-0.5)*jv(3.0/2.0, k*r[separator])
    Q[~separator] = r[~separator]**(-2.0)*jv(3.0/2.0, k)

    Bphi = C*Q*sin(theta)

    return Br, Btheta, Bphi

def get_B_s_3(r, theta, phi, C=0.0169610298034, k=6.987932000501):
    r"""
    Computes the third symmetric free decay mode.
    Purely poloidal

    Parameters
    ----------
    r : array_like
        NxNxN array containing azimuthal coordinates.
    theta : array_like
        NxNxN array containing azimuthal coordinates.
    phi : array_like
        NxNxN array containing azimuthal coordinates.
    C : float, optional
        Mode normalization.
    k : float, optional
        :math:`k`

    Returns
    -------
    list
        List containing three `numpy` arrays corresponding to:
        :math:`B_r`, :math:`B_\theta`, :math:`B_\phi`
    """
    separator = r <=1.

    # Auxiliary
    cost = cos(theta)
    sint = sin(theta)
    y = k*r[separator]

    # Computes radial component
    Q = N.empty_like(r)
    Q[separator] = r[separator]**(-0.5)*jv(9.0/2.0,y)
    Q[~separator]  = r[~separator]**(-5.0)*jv(9.0/2.0,k)
    S = -700.*cost**4+600.*cost**2-60

    Br = C*Q*S/r

    # Computes theta component
    Q[separator] = Q[separator]/2.0 + r[separator]**(0.5)/2.0 * k  \
                                        * (jv(7.0/2.0,y) - jv(11.0/2.0,y))
    Q[~separator] = -4*r[~separator]**(-5.0)*jv(9.0/2.0,k)
    S = -140.0*cost**3*sint+60*cost*sint

    Btheta = -C * Q * S/r

    # Sets azimuthal component
    Bphi = N.zeros_like(r)

    return Br, Btheta, Bphi


def get_B_s_4(r, theta, phi, C=0.539789362061, k=6.987932000501):
    r"""
    Computes the fourth symmetric free decay mode.
    Purely toroidal.

    Parameters
    ----------
    r : array_like
        NxNxN array containing azimuthal coordinates.
    theta : array_like
        NxNxN array containing azimuthal coordinates.
    phi : array_like
        NxNxN array containing azimuthal coordinates.
    C : float, optional
        Mode normalization.
    k : float, optional
        :math:`k`

    Returns
    -------
    list
        List containing three `numpy` arrays corresponding to:
        :math:`B_r`, :math:`B_\theta`, :math:`B_\phi`
    """
    separator = r <=1.

    # Sets radial component
    Br = N.zeros_like(r)

    # Sets theta component
    Btheta = N.zeros_like(r)

    # Computes azimuthal component
    Q = N.empty_like(r)
    Q[separator] = r[separator]**(-0.5)*jv(7.0/2.0, k*r[separator])
    Q[~separator] = r[~separator]**(-4.0)*jv(7.0/2.0, k)
    S = 3.*sin(theta)*(1.-5.*(cos(theta))**2)

    Bphi = -C*Q*S

    return Br, Btheta, Bphi

# Useful global variables
gamma_s = [-4.493409457909**2, -4.493409457909**2,
           -6.987932000501**2, -6.987932000501**2]
gamma_a = [-pi**2, -5.763**2, -5.763**2, -(2.*pi)**2]

gamma_m = [gamma_a[0], gamma_s[1], gamma_s[0], gamma_a[2],
           gamma_a[1], gamma_s[3], gamma_s[2], gamma_a[3]]

symmetric_modes_list = [get_B_s_1, get_B_s_2, get_B_s_3, get_B_s_4]
antisymmetric_modes_list = [get_B_a_1, get_B_a_2, get_B_a_3, get_B_a_4]

# Ordered by decay rate
mixed_modes_list = [get_B_a_1, get_B_s_2, get_B_s_1, get_B_a_3,
                    get_B_a_2, get_B_s_4, get_B_s_3, get_B_a_4]

def get_mode(r, theta, phi, n_mode, symmetric):
    r"""
    Computes the n_mode'th free decay mode.

    (This is actually a wrapper invoking other functions)

    Parameters
    ----------
    r : array_like
        NxNxN array containing azimuthal coordinates.
    theta : array_like
        NxNxN array containing azimuthal coordinates.
    phi : array_like
        NxNxN array containing azimuthal coordinates.
    nmode : int
        index of the mode to be computed
    symmetric :
        If `True`, the mode is assumed to be symmetric over the midplane
        (i.e. quadrupolar). If `False` the mode is assumed to be antisymmetric
        (dipolar). If `None` (or anything else), than the mode is drawn from a
        mixture of symmetric and antisymmetric modes.

    Note
    ----
    At the moment, only the first 4 symmetric and first 4 antisymmetric modes
    are available. Later versions will support arbitrary choice of mode.

    Returns
    -------
    list
        List containing three `numpy` arrays corresponding to:
        :math:`B_r`, :math:`B_\theta`, :math:`B_\phi`
    """
    if (n_mode>4 and symmetric in (True, False)) or (n_mode>8):
        raise NotImplementedError

    if symmetric == True:
        return symmetric_modes_list[n_mode-1](r, theta, phi)
    elif symmetric == False:
        return antisymmetric_modes_list[n_mode-1](r, theta, phi)
    else:
        return mixed_modes_list[n_mode-1](r, theta, phi)

class xi_lookup_table(object):
    r"""
    Stores a look-up table of the roots of the equation

    .. math::
            J_{n-1/2}(\xi_{nl}) J_{n+1/2}(\xi_{nl}) = 0

    which can be accessed through the method :func:`get_xi`.

    These are related to the decay rates through:
    :math:`\gamma_{nl} = -(\xi_{nl})^2`

    which can be access through the method :func:`get_gamma`.

    Parameters
    ----------
    filepath : str
        Path to the file where the lookup table will be stored.
        Default: '.xilookup.npy'
    regenerate : bool
        If True, the original file on hard disk will be overwritten.
    """
    def __init__(self, filepath='.xilookup.npy', regenerate=False,
                 **kwargs):

        self.filepath = filepath

        if regenerate or (not os.path.isfile(filepath)):
            self.generate_xi_lookup_table(**kwargs)
        else:
            self.table = N.load(filepath)
            self.max_n, self.max_l = N.shape(self.table)


    def get_xi(self, n, l, regenerate=False, **kwargs):
        r"""
        Computes or reads the root of
        :math:`J_{n-1/2}(\xi_{nl}) J_{n+1/2}(\xi_{nl}) = 0`

        Parameters
        ----------
        n, l : int
          indices of the free decay mode

        regenerate :
             (Default value = False)
        **kwargs :

        Returns
        -------
        float
            :math:`\xi_{nl}`
        """
        if regenerate or (n > self.max_n) or (l>self.max_l):
            self.generate_xi_lookup_table(max_n=n+1, max_l=l+1)
        return self.table[n-1,l-1]


    def get_gamma(self, n, l, **kwargs):
        r"""
        Returns the free-decay-mode decay rate, :math:`\gamma_{nl}`

        Parameters
        ----------
        n, l : int
          indices of the free decay mode

        Returns
        -------
        float
            :math:`\gamma_{nl}`
        """
        return -(self.get_xi(n, l, **kwargs))**2


    def generate_xi_lookup_table(self, max_n=4, max_l=4,
                                 number_of_guesses=150, max_guess=25,
                                 save=True):
        r"""
        Produces a (max_n,max_l)-array containing containing the roots of

        .. math::
            J_{n-1/2}(\xi_{nl}) J_{n+1/2}(\xi_{nl}) = 0

        These are related to the decay rates through:

        .. math::
            \gamma_{nl} = -(\xi_{nl})^2
        
        The root are found using the mpmath.findroot function. The search
        for roots supplying findroot with number_of_guesses initial guesses
        uniformly distributed in the interval [3,max_guess].

        Parameters
        ----------
        max_n, max_l : int
            Maximum values for n and l in the table. Default: 4
        number_of_guesses : int
            Number of uniformly distributed guesses. Default 150
        max_guess : float
             Maximum guess value for root.
        save :
             Saves the lookup table to file

        Returns
        -------
        numpy.ndarray
            A (max_n,max_l)-array containing containing the roots of
        """
        self.table = N.empty((max_n,max_l))
        guesses = linspace(3,max_guess,number_of_guesses)

        for n in range(1,max_n+1):
            # The following should be zero in order to have a free decay mode
            f = lambda x: besselj(n+0.5, x)*besselj(n-0.5,x)
            results = []
            for guess in guesses:
                try:

                    # Stores every root found
                    results.append(N.float64(findroot(f, guess)))
                except ValueError:
                    # Ignores failures in finding the root
                    pass
            # Excludes identical results
            results = N.unique(results)
            # Avoids spurious results close to 0
            results = results[results>1.0]
            # Updates table
            self.table[n-1,:] = results[:max_l]
        if save:
            N.save(self.filepath, self.table)
        self.max_n, self.max_l = N.shape(self.table)
