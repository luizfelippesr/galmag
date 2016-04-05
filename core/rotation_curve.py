import numpy as N

def simple_V(r_sph, V0, s0):
    V = N.zeros_like(r_sph)
    rho = r_sph[0,:,:,:]
    theta = r_sph[1,:,:,:]

    V[2,:,:,:] = V0 * ( 1.0 - N.exp(-rho*N.sin(theta)/s0) )

    return V


def simple_S(r_sph, V0, s0):
    S = N.zeros_like(r_sph)
    rho = r_sph[0,:,:,:]
    theta = r_sph[1,:,:,:]
    sin_theta = N.sin(theta)

    S[2,:,:,:] = V0*(sin_theta/s0)*N.exp(-rho*sin_theta/s0)  \
                    - simple_V(r_sph,V0,s0)/rho
    return S


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
  
