
import pylab as P
import numpy as N

def simple_V(r_sph, V0, s0):
    """ Simple form of the rotation curve to be used for the halo """

    V = N.zeros_like(r_sph)
    rho = r_sph[0,:,:,:]
    theta = r_sph[1,:,:,:]

    V[2,:,:,:] = V0 * (1.0 - N.exp(-rho*N.sin(theta)/s0))

    return V


def simple_radial_Shear(r_cyl, V0, s0, Rmax=None):
    """ Shear rate compatible with simple_V """

    # Traps negligible radii
    if isinstance(rho_cyl, N.ndarray):
        if N.any(rho_cyl/s0 < 1e-10):
            print 'simple_radial_Shear: Warning, negligible disk radius detected.'
    elif rho_cyl/s0 < 1e-10:
        rho_cyl = 1e-10

    V = V0 * ( 1.0 - N.exp(-rho_cyl/s0) )
    S = - V0/s0 * N.exp(-rho_cyl/s0) - V/rho_cyl

    return S


def unity_Shear(r_cyl, V0, s0, Rmax=None):
    return -1.0


def simple_alpha(r):
    """ Simple profile for alpha"""
    rho = r[0,:,:,:]
    theta = r[1,:,:,:]
    
    alpha = N.cos(theta)
    alpha[rho>1.] = 0.
    
    return alpha
  

# Coefficients used in the polynomial fit of Clemens (1985)
coef_Clemens = {
      'A': [-17731.0,54904.0, -68287.3, 43980.1, -15809.8, 3069.81, 0.0],
      'B': [-2.110625, 25.073006, -110.73531, 231.87099, -248.1467, 325.0912],
      'C': [0.00129348, -0.08050808, 2.0697271, -28.4080026, 224.562732,
                          -1024.068760,2507.60391, -2342.6564],
      'D': [234.88] }
# Ranges used in the polynomial fit of Clemens (1985)
ranges_Clemens = {
      'A': [0,0.09],
      'B': [0.09,0.45],
      'C': [0.45,1.60],
      'D': [1.60,1000]
      }

def Clemens_Milky_way_rotation_curve(r_cyl, V0, s0, Rmax=16, R0=8.5):
    """ Rotation curve of the Milky Way obtained by Clemens (1985) """

    R0 /= Rmax

    # Makes sure we are dealing with an array
    if not isinstance(r_cyl, N.ndarray):
        r_cyl = N.array([r_cyl])

    V = N.empty_like(r_cyl)
    for x in coef_Clemens:
        # Construct polynomials
        pol_V = N.poly1d(coef_Clemens[x])
        # Reads the ranges
        min_r, max_r = ranges_Clemens[x]
        # Sets the index (selects the relevant range)
        idx  = r_cyl/R0 >= min_r
        idx *= r_cyl/R0 < max_r
        # Computes the shear rate ( rdOmega/dr = dV/dr - V/r )
        V[idx] = pol_V(r_cyl[idx])

    return V

def Clemens_Milky_way_shear_rate(r_cyl, V0, R0):
    """ Shear rate of the Milky Way based on the rotation curve
        obtained by Clemens (1985) """

    # Makes sure we are dealing with an array
    if not isinstance(r_cyl, N.ndarray):
        r_cyl = N.array([r_cyl])
        scalar = True
    else:
        scalar = False

    S = N.empty_like(r_cyl)
    for x in coef_Clemens:
        # Construct polynomials
        pol_V = N.poly1d(coef_Clemens[x])
        dVdr = pol_V.deriv()
        # Reads the ranges
        min_r, max_r = ranges_Clemens[x]
        # Sets the index (selects the relevant range)
        idx  = r_cyl/R0 >= min_r
        idx *= r_cyl/R0 < max_r
        # Computes the shear rate ( rdOmega/dr = dV/dr - V/r )
        S[idx] = dVdr(r_cyl[idx]) - pol_V(r_cyl[idx])/r_cyl[idx]
    if scalar:
        S = S[0]

    return S

if __name__ == "__main__"  :
    P.figure(figsize=(8.27,11.69))
    R = N.linspace(1e-4,15,1000)
    P.subplot(3,1,1)
    P.plot(R, Clemens_Milky_way_rotation_curve(R,None,8.5), linewidth=3)
    P.xlim([0,15])
    P.ylabel(r'$V\,[{\rm km}\,{\rm s}^{-1}]$')
    P.xlabel(r'$r\,[\rm kpc]$')
    P.vlines(0.09*8.5, 0, 300, linestyle='--', alpha=0.35)
    P.subplot(3,1,2)
    P.plot(R, Clemens_Milky_way_rotation_curve(R,None,8.5)/R, linewidth=3)
    P.xlim([0,15])
    P.ylabel(r'$\Omega\,[{\rm km}\,{\rm s}^{-1}\,{\rm kpc}^{-1}]$')
    P.xlabel(r'$r\,[\rm kpc]$')
    P.vlines(0.09*8.5, 0, 1000, linestyle='--', alpha=0.35)
    P.ylim([0,800])
    P.subplot(3,1,3)
    P.plot(R, Clemens_Milky_way_shear_rate(R,None,8.5), linewidth=3)
    P.xlim([0,15])
    P.ylabel(r'$\Omega\,[{\rm km}\,{\rm s}^{-1}\,{\rm kpc}^{-1}]$')
    P.xlabel(r'$r\,[\rm kpc]$')
    P.vlines(0.09*8.5, -1500, 0, linestyle='--', alpha=0.35)
    P.ylim([-1000,0])
    P.savefig('Clemens_rotation_curve.pdf')

