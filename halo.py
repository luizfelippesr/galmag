""" Computes the magnetic field of a galaxy halo.
    Initial sketch implementation. """

import halo_free_decay_modes
import numpy as N

pi=N.pi
cos = N.cos
sin = N.sin
sqrt = N.sqrt

def curl_spherical(r, B):
    """ Computes the curl of a vector field in spherical coordinates.
        Input:
            r, B: position vector and magnetic field, respectively,
            expressed as 3xNxNxN arrays containing the r, theta and phi
            components in [0,...], [1,...] and [2,...], respectively.
        Output:
            Returns a 3xNxNxN array, containing the radial, polar
            and azimuthal components of the curl of the field.
        NB the coordinates are expected to be an uniform grid.
    """
    Br, Btheta, Bphi = B[0,...], B[1,...], B[2,...]
    rr, theta, phi = r[0,...], r[1,...], r[2,...]

    # Gets grid spacing (assuming uniform grid spacing)
    dr = rr[1]-rr[0]
    dtheta  = theta[1] - theta[0]
    dphi = phi[1] - phi[0]

    # Computes partial derivatives
    dBr_dr, dBr_dtheta, dBr_dphi = N.gradient(Br,dr,dtheta,dphi)
    dBtheta_dr, dBtheta_dtheta, dBtheta_dphi = N.gradient(Btheta,dr,dtheta,dphi)
    dBphi_dr, dBphi_dtheta, dBphi_dphi = N.gradient(Bphi,dr,dtheta,dphi)

    # Auxiliary
    sint = sin(theta)
    cost = cos(theta)

    # Radial component of the curl
    cBr = 1.0/(rr*sint) * (Bphi*cos(theta) + sint*dBphi_dtheta - dBtheta_dphi)
    cBtheta = 1.0/rr * (dBr_dphi/sint - Bphi - rr*dBphi_dr)
    cBphi = 1.0/rr * (Btheta + rr*dBtheta_dr - dBr_dtheta)

    cB = N.empty_like(B)
    cB[0,...] = cBr
    cB[1,...] = cBtheta
    cB[2,...] = cBphi

    return cB

def perturbation_operator(r, B, alpha, V, p, dynamo_type='alpha-omega'):
    """ Applies the perturbation operator associated with an
        an alpha-omega dynamo to a magnetic field in spherical coordinates.

        Input:
            r, B, alpha, V: position vector (not radius!), magnetic field,
            alpha and rotaion curve, respectively, expressed as 3xNxNxN arrays
            containing the r, theta and phi components in [0,...], [1,...]
            and [2,...], respectively.
            p: dictionary of parameters containing 'Ralpha'.
        Output:
            Returns a 3xNxNxN array containing W(B)
    """

    # Makes sure the input is consistent (fails otherwise)
    assert B.shape == V.shape
    assert B[0,...].shape == alpha.shape
    assert 'Ralpha' in p
    Ra = p['Ralpha']

    if dynamo_type=='alpha2-omega':
        assert 'Romega' in p
        Ro = p['Romega']


    # Computes \nabla \times (\alpha B)
    aB = empty_like(B)
    for i in range(3):
        aB[i,:,:,:] = alpha*B[i,:,:,:]
    curl_aB = curl_spherical(r, aB)
    del aB

    # Computes \nabla \times (V \times B)
    VcrossB = N.cross(V, B, axis=0)
    curl_VcrossB = curl_spherical(r, VcrossB)
    del VcrossB

    if dynamo_type=='alpha-omega':
        WB = Ra*(curlaB - curlaB[2,...]) + Ra*curl_VcrossB

    elif dynamo_type=='alpha2-omega':
        WB = Ra*curlaB + Ro*curl_VcrossB

    else:
        raise AssertionError('Invalid option: dynamo_type={0}'.format(dynamo_type))

    return WB


