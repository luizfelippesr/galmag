""" Contains the definitions of the disk rotation curve, radial shear,
    alpha profile and disk scale height. """
import numpy as N

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

def Clemens_Milky_Way_rotation_curve(R, R_d=1.0, Rsun=8.5, normalize=False):
    """ Rotation curve of the Milky Way obtained by Clemens (1985)
        Input: R -> radial coordinate
               Rsun -> sun's radius in kpc. Default: 8.5 kpc
               R_d -> unit of R in kpc [e.g. R_d=(disk radius in kpc)
                      for r=0..1 within the disk]. Default: 1.0
        Ouput: V -> rotation curve, with:
                results normalized to unit at solar radius, if normalize==True
                results in km/s for R and Rsun in kpc, if normalize==False
    """

    # Makes sure we are dealing with an array
    if not isinstance(R, N.ndarray):
        R = N.array([R,])

    V = N.empty_like(R)
    for x in coef_Clemens:
        # Construct polynomials
        pol_V = N.poly1d(coef_Clemens[x])
        # Reads the ranges
        min_r, max_r = ranges_Clemens[x]
        # Sets the index (selects the relevant range)
        idx  = R*R_d/Rsun >= min_r
        idx *= R*R_d/Rsun < max_r
        # Computes the shear rate ( rdOmega/dr = dV/dr - V/r )
        V[idx] = pol_V(R[idx]*R_d)

    if normalize:
        # Normalizes at solar radius
        Vsol = N.poly1d(coef_Clemens['C'])(Rsun)
        V = V/Vsol

    return V


def Clemens_Milky_Way_shear_rate(R, R_d=1.0, Rsun=8.5, normalize=False):
    """ Shear rate of the Milky Way based on the rotation curve
        obtained by Clemens (1985)
        NB the maximum shear is normalized to 1
        Input: rho_cyl -> radial coordinate (in cylindrical coordinates
               R -> Radius of the halo in the units of rho_cyl
               optional, Rsun -> sun's radius. Default: 8.5 kpc
               optional, Ssun -> Shear
        Ouput: Shear rate in units of [Vref/R] """

    # Makes sure we are dealing with an array
    if not isinstance(R, N.ndarray):
        R = N.array([R,])
        scalar = True
    else:
        scalar = False

    S = N.empty_like(R)
    for x in coef_Clemens:
        # Construct polynomials
        pol_V = N.poly1d(coef_Clemens[x])
        dVdr = pol_V.deriv()

        # Reads the ranges
        min_r, max_r = ranges_Clemens[x]
        # Sets the index (selects the relevant range)
        idx  = R*R_d/Rsun >= min_r
        idx *= R*R_d/Rsun < max_r
        # Computes the shear rate ( rdOmega/dr = dV/dr - V/r )
        S[idx] = dVdr(R[idx]*R_d) - pol_V(R[idx]*R_d)/(R[idx]*R_d)

    if normalize:
        # Normalizes at solar radius
        pol_V = N.poly1d(coef_Clemens['C'])
        dVdr = pol_V.deriv()
        S_sol = dVdr(Rsun) - pol_V(Rsun)/Rsun
        S = S/S_sol

    if scalar:
        S = S[0]

    return S



if __name__ == "__main__"  :
    # If invoked as a script, prepare some testing plots
    import pylab as P
    import tools

    Rsun = 8.5
    fig = P.figure(1)
    fig.set_size_inches((10,12), forward=True)

    # Radial coordinate
    R = N.linspace(0,17., 250)

    P.suptitle('Disk profiles')

    P.subplot(2,1,1)
    Vsun = Clemens_Milky_Way_rotation_curve(Rsun, Rsun=Rsun)
    P.title('Rotation curve')
    P.ylabel(r'$V(R) \; [{{\rm km}}/{{\rm s}}]$')
    V = Clemens_Milky_Way_rotation_curve(R, R_d=1.0, Rsun=Rsun)
    P.plot(R, V)
    P.plot(Rsun, Vsun,'yo')
    P.xlabel(r'$R \; [{{\rm kpc}}]$')

    P.subplot(2,1,2)
    Ssun = Clemens_Milky_Way_shear_rate(Rsun, Rsun=Rsun)
    S = Clemens_Milky_Way_shear_rate(R, Rsun=Rsun)
    P.plot(R, S)
    P.plot(Rsun, Ssun,'yo')
    P.title('Shear rate profile')
    P.ylabel(r'$S(R)=\frac{{ {{\rm d}} V }}{{ {{\rm d}} t}} -'
             r'\frac{{ V}}{{R}} \; $')
    P.xlabel(r'$R\;[{{\rm kpc}}]$')



    P.show()
