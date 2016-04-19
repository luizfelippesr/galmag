import numpy as N

def simple_V(r_sph, V0, s0):
    V = N.zeros_like(r_sph)
    rho = r_sph[0,:,:,:]
    theta = r_sph[1,:,:,:]

    V[2,:,:,:] = V0 * (1.0 - N.exp(-rho*N.sin(theta)/s0))

    return V


def simple_radial_Shear(r_cyl, V0, s0):

    V = V0 * ( 1.0 - N.exp(-r_cyl/s0) )

    S = V0/s0 * N.exp(-r_cyl/s0) - V/r_cyl
    # The following renormalisation makes S(r=1, 1,1) = -1
    S /= 0.26424111765711533

    # Traps negligible radii
    if isinstance(r_cyl, N.ndarray):
        small_r = r_cyl/s0<1e-7
        S[small_r]=0.0
    elif r_cyl/s0<1e-7:
        S = 0.0
    return S

def unity_Shear(r_cyl, V0, s0):
    return -1.0

def unity_dummy(r_sph, V0, s0):
    S = N.zeros_like(r_sph)
    S[2,:,:,:] = 1.0
    return S


def simple_alpha(r):
    rho = r[0,:,:,:]
    theta = r[1,:,:,:]
    
    alpha = N.cos(theta)
    alpha[rho>1.] = 0.
    
    return alpha
  
