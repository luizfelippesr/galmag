""" Computes the magnetic field of a galaxy halo.
    Initial sketch implementation. """

import halo_free_decay_modes as free
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
        a dynamo to a magnetic field in spherical coordinates.

        Input:
            r, B, alpha, V: position vector (not radius!), magnetic field,
            alpha and rotation curve, respectively, expressed as 3xNxNxN arrays
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

def Galerkin_expansion_coefficients(r, alpha, V, p, symmetric=False,
                                    dynamo_type='alpha-omega', 
                                    n_free_decay_modes=3):
    """ Calculates the Galerkin expansion coefficients. 
        
        First computes the transformation M defined by:
        Mij = gamma_j, for i=j
        Mij = Wij, for i!=j
         where:
         W_{ij} = \int B_j \cdot \hat{W} B_i
        Then, solves the eigenvalue/eigenvector problem.
      
        Input:
      
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
    Bi = empty(n_free_decay_modes, nc, nr, ntheta, nphi)
    WBj = N.empty_like(B)
    
    # These are the pre-computed gamma_j's
    gamma = [-N.pi**2, -(5.763)**2, -(5.763)**2, -(2*N.pi**2)]
    
    for i in range(n_free_decay_modes):
        # Computes the halo free decay modes
        # Symmetric modes
        if symmetric and i==0:
            Bi[i,0,...], B[i,1,...], B[i,2,...] = free.get_B_s_1(r, theta, phi)
        elif symmetric and i==1:
            Bi[i,0,...], B[i,1,...], B[i,2,...] = free.get_B_s_2(r, theta, phi)
        elif symmetric and i==2:
            Bi[i,0,...], B[i,1,...], B[i,2,...] = free.get_B_s_3(r, theta, phi)    
        # Antisymmetric modes
        elif i==0:
            Bi[i,0,...], B[i,1,...], B[i,2,...] = free.get_B_a_1(r, theta, phi)
        elif i==1:
            Bi[i,0,...], B[i,1,...], B[i,2,...] = free.get_B_a_2(r, theta, phi)
        elif i==2:
            Bi[i,0,...], B[i,1,...], B[i,2,...] = free.get_B_a_3(r, theta, phi)
        
        # Applies the perturbation operator
        WBj[i] = perturbation_operator(r, B, alpha, V, p, 
                                       dynamo_type=dynamo_type)
    
    # Computes volume elements (associated with each grid point)
    dr = radius[1]-radius[0]
    dtheta = theta[1]-theta[0]
    dphi = phi[1]-phi[0]
    dV = radius**2 * N.sin(theta) * dr * dtheta * dphi
    
    # Computes the Wij elements.
    #   indices lmn label the grid positions
    #   indices k label difference components
    #   indices i/j label free decay modes
    # W_{ij} = \sum_{l}\sum_{m}\sum_{n} \sum_{k} B_{iklmn} WB_{jklmn} dVlmn
    Wij = N.einsum('iklmn,jklmn,lmn', Bi, WBj, dV)
    # Overwrites the diagonal with its correct values
    for i in range(n_free_decay_modes):
        Wij[i,i] = gamma[i]
    
    # Solves the eigenvector problem and returns the result
    return N.linalg.eig(Wij)

