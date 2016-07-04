""" Contains the definitions of the halo rotation curve and alpha profile. """
import numpy as N

def simple_V(r_sph, R_h=1, V0=1, s0=5, R_ref=10):
    """ Simple form of the rotation curve to be used for the halo
        Input: r_sph -> 3xNxNxN grid in spherical coordinates
               Rref -> Radius of the reference halo. Default: 10 kpc
               s0 -> s0 parameter in the reference halo. Default: 5 kpc
               R -> Radius of the halo in the units of r_sph. Default: 1
               V0 -> Normalization of the rotation curve. Default: 1
        Ouput: V -> rotation curve in the units of V0
    """
    s0 = s0/R_ref * R

    V = N.zeros_like(r_sph)
    rho = r_sph[0,:,:,:] * R_ref/R
    theta = r_sph[1,:,:,:]

    V_norm = (1.0 - N.exp(-R_ref/s0))
    V[2,:,:,:] = Vref * (1.0 - N.exp(-rho*N.sin(theta)/s0))/V_norm

    return V


def simple_alpha(r, alpha0=1):
    """ Simple profile for alpha"""
    rho = r[0,:,:,:]
    theta = r[1,:,:,:]

    alpha = N.cos(theta)
    alpha[rho>1.] = 0.

    return alpha*alpha0




