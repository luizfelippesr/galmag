# Copyright (C) 2017,2018,2019,2020 Luiz Felippe S. Rodrigues <luizfelippesr@alumni.usp.br>
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
Contains functions which deal with the computation the coefficients
associated with a Galerkin expansion of the solution of the mean field dynamo
equation.
"""
import numpy as np
import galmag.halo_free_decay_modes as halo_free_decay_modes
from .Grid import Grid
from .util import curl_spherical, simpson
#from numba import njit, jit

def Galerkin_expansion_coefficients(parameters, return_matrix=False,
                                    dtype=np.float64):
    r"""
    Calculates the Galerkin expansion coefficients.

    First computes the transformation M defined by:

    .. math::
        M_{ij} = W_{ij}, \text{ for } i \neq j

    .. math::
        M_{ij} = \gamma_j, \text{ for } i=j

    where:

    .. math::
        W_{ij} = \int B_j \cdot \hat{W} B_i

    Then, solves the eigenvalue/eigenvector problem using
    :func:`numpy.linalg.eig` , computing thus all the coefficients
    :math:`a_i` and the growth rates (eigenvalues) :math:`\Gamma_i`.

    Parameters
    ----------
    return_matrix : bool, optional
        If True, the matrix :math:`W_{ij}` will be returned as well.
    p : dict
        A dictionary of parameters dictionary of parameters containing the parameters:
            - halo_Galerkin_ngrid -> Number of grid points used in the
                                   calculation of the Galerkin expansion
            - halo_rotation_function -> a function specifying the halo rotation
                                     curve
            - halo_alpha_function -> a function specifying the dependence of
                                   the alpha effect
            - halo_turbulent_induction -> :math:`R_{\alpha}`
            - halo_rotation_induction -> :math:`R_{\omega}`
            - halo_n_free_decay_modes -> number of free decay modes to be used
                                       in the expansion
            - halo_symmetric_field -> Symmetric or anti-symmetric field
                                    solution
            - halo_rotation_characteristic_radius -> turn-over radius of the flat
                                                   rotation curve
            - halo_rotation_characteristic_height -> characteristic z used in some
                                                   rotation curve prescriptions


    Returns
    -------
    val : array
        n-array containing growth rates (the eigenvalues of :math:`Mij`).
    vec :  array
        :math:`a_i`'s: nx3 array containing the Galerkin coefficients associated.
    Wij : array
        The matrix :math:`W_{ij}`. Only present if return_matrix is True.
    """
    nGalerkin = parameters['halo_Galerkin_ngrid']
    function_V = parameters['halo_rotation_function']
    s_v = parameters['halo_rotation_characteristic_radius']
    z_v = parameters['halo_rotation_characteristic_height']
    function_alpha = parameters['halo_alpha_function']
    Ralpha = parameters['halo_turbulent_induction']
    Romega = parameters['halo_rotation_induction']
    nmodes = parameters['halo_n_free_decay_modes']
    symmetric = parameters['halo_symmetric_field']

    # Prepares a spherical grid for the Galerkin expansion
    galerkin_grid = Grid(box=[[0.00001,1.0], # r range
                          [0.01,np.pi],  # theta range
                          [0.0,0.0]], # phi range
                          resolution=[nGalerkin,nGalerkin,1],
                          grid_type='spherical')

    r_sph_grid = galerkin_grid.r_spherical
    phi_grid = galerkin_grid.phi
    theta_grid = galerkin_grid.theta

    # local_B_r_spherical, local_B_phi, local_B_theta (for each mode)
    Bmodes = [halo_free_decay_modes.get_mode(r_sph_grid,
                                             theta_grid,
                                             phi_grid,
                                             imode, symmetric)
              for imode in range(1,nmodes+1)]

    # Computes sintheta
    sintheta = np.sin(theta_grid)
    # Computes alpha
    alpha = function_alpha(r_sph_grid,
                           theta_grid,
                           phi_grid)
    # Computes the various components of V (locally)
    Vs = function_V(r_sph_grid,
                    theta_grid,
                    phi_grid,
                    fraction=s_v/parameters['halo_radius'],
                    fraction_z=z_v/parameters['halo_radius'])

    # Applies the perturbation operator
    WBmodes = [perturbation_operator(r_sph_grid,
                                     theta_grid,
                                     phi_grid,
                                     Bmode[0], Bmode[1], Bmode[2],
                                     Vs[0], Vs[1], Vs[2], alpha,
                                     Ralpha, Romega,
                                     parameters['halo_dynamo_type'])
               for Bmode in Bmodes]

    Wij = np.zeros((nmodes,nmodes))
    for i in range(nmodes):
        for j in range(nmodes):
            if i==j:
                continue

            integrand = np.zeros_like(r_sph_grid)

            for k in range(3):
                integrand += Bmodes[i][k]*WBmodes[j][k]

            integrand *= r_sph_grid**2 * sintheta

            # Integrates over phi assuming axisymmetry
            integrand = integrand[:,:,0]*2.0*np.pi
            # Integrates over theta
            integrand = simpson(integrand, theta_grid[:,:,0])
            # Integrates over r
            Wij[i,j] += simpson(integrand,r_sph_grid[:,0,0])

    # Overwrites the diagonal with its correct (gamma) values
    if symmetric == True:
        gamma = halo_free_decay_modes.gamma_s
    elif symmetric == False:
        gamma = halo_free_decay_modes.gamma_a
    elif symmetric == 'mixed':
        gamma = halo_free_decay_modes.gamma_m
    else:
        raise ValueError

    for i in range(nmodes):
        Wij[i,i] = gamma[i]

    # Solves the eigenvector problem and returns the result
    val, vec = np.linalg.eig(Wij)
    if not return_matrix:
        return val, vec
    else:
        return val, vec, Wij


#@jit
def perturbation_operator(r, theta, phi, Br, Bt, Bp, Vr, Vt, Vp,
                          alpha, Ra, Ro, dynamo_type='alpha-omega'):
    r"""
    Applies the perturbation operator associated with an dynamo
    to a magnetic field in uniform spherical coordinates.

    Parameters
    --------
    r/B/alpha/V : array
        position vector (not radius!), magnetic field, alpha and rotation
        curve, respectively, expressed as 3xNxNxN arrays containing the
        :math:`r`, :math:`\theta` and :math:`\phi` components in [0,...], [1,...]
        and [2,...], respectively.
    p : dict
        A dictionary of parameters containing 'Ralpha_h'.

    Returns
    -------
    list
        A list containing NxNxN arrays corresponding to the 3 components of
        :math:`\hat W(\mathbf{B})`
    """
    # Computes \nabla \times (\alpha B)
    curl_aB = curl_spherical(r, theta, phi, Br*alpha, Bt*alpha, Bp*alpha)

    # Computes \nabla \times (V \times B)
    VcrossBr = (Vt*Bp - Vp*Bt)
    VcrossBt = (Vp*Br - Vr*Bp)
    VcrossBp = (Vr*Bt - Vt*Br)

    curl_VcrossB = curl_spherical(r, theta, phi,
                                  VcrossBr, VcrossBt, VcrossBp)
    WBs = []
    for i in range(3):
        if dynamo_type=='alpha-omega':
            WBs.append(Ra*(curl_aB[i] - curl_aB[2])  \
                          + Ro*curl_VcrossB[i])

        elif dynamo_type=='alpha2-omega':
            WBs.append(Ra*curl_aB[i] + Ro*curl_VcrossB[i])

        else:
            raise AssertionError('Invalid option: dynamo_type={0}'.format(dynamo_type))

    return WBs
