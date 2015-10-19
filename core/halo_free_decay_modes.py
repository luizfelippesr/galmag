""" Contains functions to compute the free decay modes of the magnetic
    of the halo of a galaxy """
from scipy.special import j0, j1, jv, jn_zeros
import numpy as N

pi=N.pi
cos = N.cos
sin = N.sin
sqrt = N.sqrt


def get_B_a_1(r, theta, phi, C=0.346, k=pi):
    """ Computes the first (pure poloidal) antisymmetric free decay mode.
        Purely poloidal.
        Input:
              r, theta, phi: NxNxN arrays containing, repectively,
              the radial, polar and azimuthal coordinates.
              optional: C, kspherical polar coordinates
        Output: B_r, B_\theta, B_\phi """

    # Computes radial component
    Q = N.empty_like(r)
    Q[r<=1.] = r[r<=1.]**(-0.5)*jv(3.0/2.0,k*r[r<=1.])
    Q[r>1.] = r[r>1.]**(-2.0)*jv(3.0/2.0,k)

    Br = C*(2.0/r)*Q*cos(theta)


    # Computes polar component
    # X = d(rQ1)/dr
    # http://is.gd/oBIDQo
    X = N.empty_like(r)
    y = k*r[r<=1.]
    X[r<=1.] = sqrt(2.0/pi)*(k**2*r[r<=1.]**2*sin(y) - sin(y) +\
          y*cos(y))/ (y**(3.0/2.0)*sqrt(y))
    X[r>1.] = -jv(3.0/2.0,k)/r[r>1.]**2

    Btheta = C*(-sin(theta)/r)*X*cos(theta)


    # Sets azimuthal component
    Bphi   = N.zeros_like(r)

    return Br, Btheta, Bphi

def get_B_a_2(r, theta, phi, C=0.346, k=5.763):
    """ Computes the second antisymmetric free decay mode (one of a
        degenerate pair, with eigenvalue gamma_2=-(5.763)^2 .
        Purely poloidal.
        Input:
              r, theta, phi: NxNxN arrays containing, repectively,
              the radial, polar and azimuthal coordinates.
              optional: C, k
        Output:
              B_r, B_\theta, B_\phi """

    # Computes radial component
    Q = N.empty_like(r)
    Q[r<=1.] = r[r<=1.]**(-0.5)*jv(7.0/2.0, k*r[r<=1.])
    Q[r>1.] = r[r>1.]**(-4.0)*jv(7.0/2.0, k)

    Br = C*(2.0/r)*Q*cos(theta)


    # Computes polar component
    # X = d(rQ1)/dr
    # http://is.gd/B56vg9   k=a, r=x
    # r<=1, first define some auxiliary quantities
    y = k*r[r<=1.]
    siny = sin(y)
    cosy=cos(y)
    A = 15.*siny/y**3 - 15.*cosy/y**2 - 6.*siny/y + cosy
    B = -45.*siny/y**3/r[r<=1.] +45.*cosy/y**2/r[r<=1.] \
             +21.*siny/y/r[r<=1.]  -k*siny - 6.*siny/r[r<=1.]
    # Now proceeds analogously to the first decay mode
    X = N.empty_like(r)
    X[r<=1.] = A/sqrt(2.*pi*r[r<=1.]*y) - \
                k*sqrt(r[r<=1.])*A/sqrt(2.*pi)/y**(3./2.) + \
                sqrt(2./pi)*B/sqrt(k)
    X[r>1] = -3.*jv(7.0/2.0,k)*r[r>1]**(-4)
    # Sets Btheta
    Btheta = C*(-sin(theta)/r)*X*cos(theta)


    # Sets azimuthal component
    Bphi   = N.zeros_like(r)

    return Br, Btheta, Bphi


def get_B_a_3(r, theta, phi, C=3.445, k=5.763):
    """ Computes the third antisymmetric free decay mode (one of a
        degenerate pair, with eigenvalue gamma_2=-(5.763)^2 .
        Purely toroidal.
        Input:
              r, theta, phi: NxNxN arrays containing, repectively,
              the radial, polar and azimuthal coordinates.
              optional: C, k
        Output: B_r, B_\theta, B_\phi """

    # Sets radial component
    Br = N.zeros_like(r)

    # Sets polar component
    Btheta = N.zeros_like(r)

    # Computes azimuthal component
    Q = N.empty_like(r)
    Q[r<=1.] = r[r<=1.]**(-0.5)*jv(5.0/2.0, k*r[r<=1.])
    Q[r>1.] = r[r>1.]**(-3.0)*jv(5.0/2.0, k)

    Bphi = C*(-sin(theta)/r)*X*cos(theta)

    return Br, Btheta, Bphi


def get_B_a_4(r, theta, phi, C=0.346, k=(2.*pi)):
    """ Computes the forth antisymmetric free decay mode (one of a
        degenerate pair, with eigenvalue gamma_2=-(5.763)^2 .
        Purely poloidal.
        Input:
              r, theta, phi: NxNxN arrays containing, repectively,
              the radial, polar and azimuthal coordinates.
              optional: C, k
        Output: B_r, B_\theta, B_\phi """

        # This happens to have the same form as the 1st antisymmetric mode
        return get_B_a_1(r, theta, phi,C=C,k=k)

