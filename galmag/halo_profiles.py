"""
This module is part of GMF tool

Contains the definitions of the halo rotation curve and alpha profile.
"""
import numpy as N

def simple_V(rho, theta, phi, R_h=1.0, V0=1.0, s0=0.5):
    """
    Simple form of the rotation curve to be used for the halo
    NB This simple form has no z dependence
    V(r,theta,phi) = V0 (1-exp(-r sin(theta) / s0))
    Input: rho, theta, phi -> NxNxN grid in spherical coordinates
           s0 -> s0 parameter Default: 0.5
           R_h -> unit of rho in kpc [e.g. R_d=(halo radius in kpc)
                  for r=0..1 within the halo]. Default: 1.0
       V0 -> Normalization of the rotation curve. Default: 1.0
    Ouput: V -> rotation curve in the units of V0
    """
    Vr, Vt = [N.zeros_like(rho) for i in range(2)]

    Vp = V0 * (1.0 - N.exp(-rho*N.sin(theta)/s0/R_h))

    return Vr, Vt, Vp


def simple_alpha(rho, theta, phi, alpha0=1.0):
    """ Simple profile for alpha"""

    alpha = N.cos(theta)
    alpha[rho>1.] = 0.

    return alpha*alpha0



