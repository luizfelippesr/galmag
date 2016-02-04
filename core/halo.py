""" Computes the magnetic field of a galaxy halo.
    Initial sketch implementation. """

import halo_free_decay_modes as free
import tools
import numpy as N
from rotation_curve import simple_V, simple_alpha
import scipy.integrate as integrate

def curl_spherical(r, B):
    """ Computes the curl of a vector field in spherical coordinates.
        Input:
            r, B: position vector and magnetic field, respectively,
            expressed as 3xNxNxNp  arrays containing the r, theta and phi
            components in [0,...], [1,...] and [2,...], respectively.
            If Np=1, assumes axisymmetry.
        Output:
            Returns a 3xNxNxNp array, containing the components of the curl of
            the field.
        NB the coordinates are expected to be an uniform grid.
    """
    Br, Btheta, Bphi = B[0,...], B[1,...], B[2,...]
    rr, theta, phi = r[0,...], r[1,...], r[2,...]

    # Gets grid spacing (assuming uniform grid spacing)
    dr = rr[1,0,0]-rr[0,0,0]
    if dr==0:
        raise ValueError('Invalid spacing for dr')
    dtheta  = theta[0,1,0] - theta[0,0,0]
    if dtheta==0: raise ValueError('Invalid spacing for dtheta')

    n, n, np = phi.shape
    if np==1:
        # Assuming axisymmetry
        dphi = None
    else:
        dphi = phi[0,0,1] - phi[0,0,0]

    if dphi==0:
        raise ValueError('Invalid spacing for dphi')
    # Computes partial derivatives
    #gradient = my_gradient
    gradient = N.gradient
    if dphi:
        dBr_dr, dBr_dtheta, dBr_dphi = gradient(Br, dr, dtheta, dphi)
        dBtheta_dr, dBtheta_dtheta, dBtheta_dphi = gradient(Btheta,
                                                            dr, dtheta, dphi)
        dBphi_dr, dBphi_dtheta, dBphi_dphi = gradient(Bphi, dr, dtheta, dphi)
    else:
        # Assuming axisymmetry
        dBr_dr = N.zeros_like(Br)
        dBr_dtheta = N.zeros_like(Br)
        dBr_dphi = N.zeros_like(Br)

        dBtheta_dr = N.zeros_like(Br)
        dBtheta_dtheta = N.zeros_like(Br)
        dBtheta_dphi = N.zeros_like(Br)

        dBphi_dr = N.zeros_like(Br)
        dBphi_dtheta = N.zeros_like(Br)
        dBphi_dphi = N.zeros_like(Br)

        dBr_dr[...,0], dBr_dtheta[...,0] = gradient(Br[...,0], dr, dtheta)
        dBtheta_dr[...,0], dBtheta_dtheta[...,0] = gradient(Btheta[...,0],
                                                            dr, dtheta)
        dBphi_dr[...,0], dBphi_dtheta[...,0] = gradient(Bphi[...,0],
                                                        dr,dtheta)
    # Auxiliary
    sint = N.sin(theta)
    tant = N.tan(theta)

    # Components of the curl
    cBr = dBphi_dtheta/rr + Bphi/tant/rr - dBtheta_dphi/sint/rr
    cBtheta = (dBr_dphi/sint - Bphi)/rr - dBphi_dr
    cBphi = (Btheta + rr*dBtheta_dr - dBr_dtheta)/rr

    cB = N.empty_like(B)
    cB[0,:,:,:] = cBr
    cB[1,:,:,:] = cBtheta
    cB[2,:,:,:] = cBphi

    return cB

def perturbation_operator(r, B, alpha, V, p, dynamo_type='alpha-omega'):
    """ Applies the perturbation operator associated with an
        a dynamo to a magnetic field in spherical coordinates.

        Input:
            r, B, alpha, V: position vector (not radius!), magnetic field,
            alpha and rotation curve, respectively, expressed as 3xNxNxN arrays
            containing the r, theta and phi components in [0,...], [1,...]
            and [2,...], respectively.
            p: dictionary of parameters containing 'Ralpha_h'.
        Output:
            Returns a 3xNxNxN array containing W(B)
    """

    # Makes sure the input is consistent (fails otherwise)
    assert B.shape == V.shape
    assert B[0,...].shape == alpha.shape
    assert 'Ralpha_h' in p
    Ra = p['Ralpha_h']
    assert 'Romega_h' in p
    Ro = p['Romega_h']

    # Computes \nabla \times (\alpha B)
    aB = N.empty_like(B)
    for i in range(3):
        aB[i,:,:,:] = alpha*B[i,:,:,:]

    curl_aB = curl_spherical(r, aB)
    del aB
    # Computes \nabla \times (V \times B)
    VcrossB = N.cross(V, B, axis=0)
    curl_VcrossB = curl_spherical(r, VcrossB)
    del VcrossB

    WB = N.empty_like(curl_aB)*N.nan
    if dynamo_type=='alpha-omega':
        for i in range(3):
            WB[i,...] = Ra*(curl_aB[i,...] - curl_aB[2,...])  \
                          + Ro*curl_VcrossB[i,...]

    elif dynamo_type=='alpha2-omega':
        WB = Ra*curl_aB + Ro*curl_VcrossB

    else:
        raise AssertionError('Invalid option: dynamo_type={0}'.format(dynamo_type))

    return WB

def Galerkin_expansion_coefficients(r, alpha, V, p,
                                    symmetric=False,
                                    dr = None,
                                    dtheta = None,
                                    dphi = None,
                                    dynamo_type='alpha-omega',
                                    n_free_decay_modes=4,
                                    return_matrix=False):
    """ Calculates the Galerkin expansion coefficients.

        First computes the transformation M defined by:
        Mij = gamma_j, for i=j
        Mij = Wij, for i!=j
         where:
         W_{ij} = \int B_j \cdot \hat{W} B_i
        Then, solves the eigenvalue/eigenvector problem.

        Input:
            r, B, alpha, V: position vector (not radius!), magnetic field,
            alpha and rotation curve, respectively, expressed as 3xNxNxNp arrays
            containing the r, theta and phi components in [0,...], [1,...]
            and [2,...], respectively. Np=1 implies the assumption of
            axisymmetry.
            p: dictionary of parameters containing 'Ralpha_h' and 'Romega_h'.

        Output (Same as the output of numpy.linalg.eig)
          Gammas: n-array containing growth rates (the eigenvalues of Mij)
          ai's: nx3 array containing the Galerkin coefficients associated
                with each growth rate (the eigenvectors)
    """

    # Translate coordinate grid (for convenience only)
    radius = r[0,:,:,:]
    theta  = r[1,:,:,:]
    phi    = r[2,:,:,:]

    #Initializes Bi, WBi
    nc, nr, ntheta, nphi = r.shape
    Bi = N.empty((n_free_decay_modes, nc, nr, ntheta, nphi))
    WBj = N.empty_like(Bi)

    # These are the pre-computed gamma_j's
    if symmetric:
        gamma = free.gamma_s
    else:
        gamma = free.gamma_a

    for i in range(n_free_decay_modes):
        # Computes the halo free decay modes
        # Symmetric modes
        if symmetric and i==0:
            Bi[i,0,...], Bi[i,1,...], Bi[i,2,...] = free.get_B_s_1(radius, theta, phi)
        elif symmetric and i==1:
            Bi[i,0,...], Bi[i,1,...], Bi[i,2,...] = free.get_B_s_2(radius, theta, phi)
        elif symmetric and i==2:
            Bi[i,0,...], Bi[i,1,...], Bi[i,2,...] = free.get_B_s_3(radius, theta, phi)
        elif symmetric and i==3:
            Bi[i,0,...], Bi[i,1,...], Bi[i,2,...] = free.get_B_s_4(radius, theta, phi)
        # Antisymmetric modes
        elif i==0 and not symmetric:
            Bi[i,0,...], Bi[i,1,...], Bi[i,2,...] = free.get_B_a_1(radius, theta, phi)
        elif i==1 and not symmetric:
            Bi[i,0,...], Bi[i,1,...], Bi[i,2,...] = free.get_B_a_2(radius, theta, phi)
        elif i==2 and not symmetric:
            Bi[i,0,...], Bi[i,1,...], Bi[i,2,...] = free.get_B_a_3(radius, theta, phi)
        elif i==3 and not symmetric:
            Bi[i,0,...], Bi[i,1,...], Bi[i,2,...] = free.get_B_a_4(radius, theta, phi)

        # Applies the perturbation operator
        WBj[i] = perturbation_operator(r, Bi[i], alpha, V, p,
                                       dynamo_type=dynamo_type)

    Wij = N.zeros((n_free_decay_modes,n_free_decay_modes))
    for i in range(n_free_decay_modes):
        for j in range(n_free_decay_modes):
            if i==j:
                continue

            integrand = N.sum(Bi[i,...] * WBj[j,...],axis=0)

            integrand *= radius**2 * N.sin(theta)

            # Integrates over phi
            if dphi:
                integrand = integrate.simps(integrand, dx=dphi)
            else:
                # Assuming axisymmetry
                integrand = integrand[...,0]*2.0*N.pi
            # Integrates over theta
            integrand = integrate.simps(integrand, dx=dtheta)
            # Integrates over r
            Wij[i,j] += integrate.simps(integrand, dx=dr)

    # Overwrites the diagonal with its correct values
    for i in range(n_free_decay_modes):
        Wij[i,i] = gamma[i]

    # Solves the eigenvector problem and returns the result
    val, vec = N.linalg.eig(Wij)
    if not return_matrix:
        return val, vec
    else:
        return val, vec, Wij

def get_B_halo(r, p, no_spherical=True):
    """ Computes the magnetic field associated with a galaxy halo. Will choose
        the fastest growing solution compatible with the input parameters.
        Input:
            r: 3xNxNxN array containing the cartesian coordinates
            p: dictionary containing the parameters (see module doc)
        Output:
            B: 3xNxNxN array containing the components of the
                        disk magnetic field
        NB the p-dictionary will be altered, having:
           - 'halo_field_growth_rate' set to the magnetic field growth rate
           - '

        parameters in the 'p' dictionary:
        halo_symmetric_field -> 'True' if the field is symmetric over theta
        halo_n_free_decay_modes -> number of free decay modes to be used
        halo_dynamo_type -> 'alpha-omega' or 'alpha2-omega'
        rotation_curve -> name of the routine used to compute the rot. curve
        alpha -> name of the routine used to compute alpha
        Galerkin_n_grid -> square-root of the number of grid points used for
                           computing the coefficients in the Galerkin expansion
        Ralpha_h -> a measure of mean induction by interstellar turbulence
        Romega_h -> a measure of induction by differential rotation
    """
    # Reads parameters (using default values when they are absent
    symmetric = tools.get_param(p, 'halo_symmetric_field', default=True)
    n_modes = tools.get_param(p, 'halo_n_free_decay_modes', default=4)
    dynamo_type = tools.get_param(p, 'halo_dynamo_type', default='alpha-omega')
    rotation_curve = tools.get_param(p, 'rotation_curve', default=simple_V)
    V0 = tools.get_param(p, 'rotation_curve_V0', default=1.0)
    s0 = tools.get_param(p, 'rotation_curve_s0', default=1.0)
    alpha = tools.get_param(p, 'alpha', default=simple_alpha)
    n_grid = tools.get_param(p, 'Galerkin_n_grid ', default=250)
    growing_only = tools.get_param(p, 'halo_growing_mode_only', default=False)

    # Sets up the grid used to compute the Galerkin expansion coefficients
    r_range  = [0.001,2.0]
    theta_range = [0.1,N.pi]

    r_tmp, dr, dt, dp = tools.generate_grid(n_grid, return_dxdydz=True,
                                xlim=r_range,
                                ylim=theta_range,
                                zlim=None)

    #Computes the rotation curve and alpha
    V = rotation_curve(r_tmp, V0, s0)
    a = alpha(r_tmp)

    # Finds the coefficients
    values, vect = Galerkin_expansion_coefficients(r_tmp, a, V, p,
                                                   symmetric=symmetric,
                                                   dynamo_type=dynamo_type,
                                                   n_free_decay_modes=n_modes,
                                                   dr=dr, dtheta=dt, dphi=dp)

    # Selects fastest growing mode
    ok = N.argmax(values.real)
    growth_rate = values[ok]
    coefficients = vect[ok].real
    # Normalizes coefficients
    coefficients = coefficients/(abs(coefficients)).max()

    # Allocates final storage
    B = N.zeros_like(r)
    Bsph = N.zeros_like(r)
    if not no_spherical:
        Bsph_f = N.zeros_like(r)

    p['halo_field_growth_rate'] = growth_rate
    p['halo_field_coefficients'] = coefficients

    if growing_only and growth_rate<0:
        if not no_spherical:
            return B, Bsph
        else:
            return B

    # Selects the relevant free modes list
    if symmetric:
        modeslist = free.symmetric_modes_list
    else:
        modeslist = free.antisymmetric_modes_list

    # Converts the grid to spherical coordinates
    rho = N.sqrt(r[0,...]**2+r[1,...]**2+r[2,...]**2)/p['R_h']
    phi = N.arctan2(r[1,...],r[0,...])
    theta = N.arccos(r[2,...]/p['R_h']/rho)

    for coeff, mode in zip(coefficients, modeslist):
        # Computes the resulting field on this grid, in spherical coordinates
        Bsph[0,...], Bsph[1,...], Bsph[2,...] = mode(rho, theta, phi)
        Bsph *= coeff

        if not no_spherical:
            Bsph_f += Bsph

        # Recasts in cartesian coordinates
        Bx, By, Bz = tools.spherical_to_cartesian(rho, theta, phi, Bsph[0,...],
                                                  Bsph[1,...], Bsph[2,...])
        B[0,...] += Bx; B[1,...] += By; B[2,...] += Bz
    if not no_spherical:
        return B, Bsph
    else:
        return B

