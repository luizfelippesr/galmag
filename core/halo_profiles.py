""" Contains the definitions of the halo rotation curve and alpha profile. """
import numpy as N

def simple_V(r_sph, R_h=1.0, V0=1.0, s0=5.0, R_ref=10.0):
    """ Simple form of the rotation curve to be used for the halo
        Input: r_sph -> 3xNxNxN grid in spherical coordinates
               Rref -> Radius of the reference halo. Default: 10 kpc
               s0 -> s0 parameter in the reference halo. Default: 5 kpc
               R -> Radius of the halo in the units of r_sph. Default: 1
               V0 -> Normalization of the rotation curve. Default: 1
        Ouput: V -> rotation curve in the units of V0
    """
    s0 = s0/R_ref * R_h

    V = N.zeros_like(r_sph)
    rho = r_sph[0,:,:,:]
    theta = r_sph[1,:,:,:]

    V[2,:,:,:] = V0 * (1.0 - N.exp(-rho*N.sin(theta)/s0))

    return V


def simple_alpha(r, alpha0=1):
    """ Simple profile for alpha"""
    rho = r[0,:,:,:]
    theta = r[1,:,:,:]

    alpha = N.cos(theta)
    alpha[rho>1.] = 0.

    return alpha*alpha0


if __name__ == "__main__"  :
    # If invoked as a script, prepare some testing plots
    import pylab as P
    import tools

    # Generates an uniform grid, varying only the r coordinate
    r_sph = tools.generate_grid(15, xlim=[0, 1.5], ylim=None, zlim=None)
    r_sph[1,...] = N.pi/2.0
    V = simple_V(r_sph)
    P.plot(r_sph[0,:,0,0], V[2,:,0,0])
    P.title('Rotation curve')
    P.ylabel(r'$V$')
    P.xlabel(r'$r\quad(\theta=\pi/2)$')
    P.show()

