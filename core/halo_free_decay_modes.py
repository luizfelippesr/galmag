""" Contains functions to compute the free decay modes of the magnetic
    of the halo of a galaxy """
from scipy.special import j0, j1, jv, jn_zeros
import numpy as N

pi=N.pi
cos = N.cos
sin = N.sin
sqrt = N.sqrt


def get_B_a_1(r, theta, phi, C=0.346, k=pi):
    """ Computes the first (pure poloidal) antisymmetric free decay mode.
        Purely poloidal.
        Input:
              r, theta, phi: NxNxN arrays containing, repectively,
              the radial, polar and azimuthal coordinates.
              optional: C, kspherical polar coordinates
        Output: B_r, B_\theta, B_\phi """

    # Computes radial component
    Q = N.empty_like(r)
    Q[r<=1.] = r[r<=1.]**(-0.5)*jv(3.0/2.0,k*r[r<=1.])
    Q[r>1.] = r[r>1.]**(-2.0)*jv(3.0/2.0,k)

    Br = C*(2.0/r)*Q*cos(theta)

    # Computes polar component
    # X = d(rQ1)/dr
    # http://is.gd/oBIDQo
    X = N.empty_like(r)
    y = k*r[r<=1.]
    X[r<=1.] = sqrt(2.0/pi)*(k**2*r[r<=1.]**2*sin(y) - sin(y) +\
          y*cos(y))/ (y**(3.0/2.0)*sqrt(y))
    X[r>1.] = -jv(3.0/2.0,k)/r[r>1.]**2

    Btheta = C*(-sin(theta)/r)*X*cos(theta)

    # Sets azimuthal component
    Bphi   = N.zeros_like(r)

    return Br, Btheta, Bphi

def get_B_a_2(r, theta, phi, C=0.346, k=5.763):
    """ Computes the second antisymmetric free decay mode (one of a
        degenerate pair, with eigenvalue gamma_2=-(5.763)^2 .
        Purely poloidal.
        Input:
              r, theta, phi: NxNxN arrays containing, repectively,
              the radial, polar and azimuthal coordinates.
              optional: C, k
        Output:
              B_r, B_\theta, B_\phi """

    # Computes radial component
    Q = N.empty_like(r)
    Q[r<=1.] = r[r<=1.]**(-0.5)*jv(7.0/2.0, k*r[r<=1.])
    Q[r>1.] = r[r>1.]**(-4.0)*jv(7.0/2.0, k)

    Br = C*(2.0/r)*Q*cos(theta)


    # Computes polar component
    # X = d(rQ1)/dr
    # http://is.gd/B56vg9   k=a, r=x
    # r<=1, first define some auxiliary quantities
    y = k*r[r<=1.]
    siny = sin(y)
    cosy=cos(y)
    A = 15.*siny/y**3 - 15.*cosy/y**2 - 6.*siny/y + cosy
    B = -45.*siny/y**3/r[r<=1.] +45.*cosy/y**2/r[r<=1.] \
             +21.*siny/y/r[r<=1.]  -k*siny - 6.*siny/r[r<=1.]
    # Now proceeds analogously to the first decay mode
    X = N.empty_like(r)
    X[r<=1.] = A/sqrt(2.*pi*r[r<=1.]*y) - \
                k*sqrt(r[r<=1.])*A/sqrt(2.*pi)/y**(3./2.) + \
                sqrt(2./pi)*B/sqrt(k)
    X[r>1] = -3.*jv(7.0/2.0,k)*r[r>1]**(-4)
    # Sets Btheta
    Btheta = C*(-sin(theta)/r)*X*cos(theta)


    # Sets azimuthal component
    Bphi   = N.zeros_like(r)

    return Br, Btheta, Bphi


def get_B_a_3(r, theta, phi, C=3.445, k=5.763):
    """ Computes the third antisymmetric free decay mode (one of a
        degenerate pair, with eigenvalue gamma_2=-(5.763)^2 .
        Purely toroidal.
        Input:
              r, theta, phi: NxNxN arrays containing, repectively,
              the radial, polar and azimuthal coordinates.
              optional: C, k
        Output: B_r, B_\theta, B_\phi """

    # Sets radial component
    Br = N.zeros_like(r)

    # Sets polar component
    Btheta = N.zeros_like(r)
    
    print r,theta,phi
    # Computes azimuthal component
    Q = N.empty_like(r)
    Q[r<=1.] = r[r<=1.]**(-0.5)*jv(5.0/2.0, k*r[r<=1.])
    Q[r>1.] = r[r>1.]**(-3.0)*jv(5.0/2.0, k)

    Bphi = C*Q*sin(theta)*cos(theta)

    return Br, Btheta, Bphi


def get_B_a_4(r, theta, phi, C=0.346, k=(2.*pi)):
    """ Computes the forth antisymmetric free decay mode (one of a
        degenerate pair, with eigenvalue gamma_2=-(5.763)^2 .
        Purely poloidal.
        Input:
              r, theta, phi: NxNxN arrays containing, repectively,
              the radial, polar and azimuthal coordinates.
              optional: C, k
        Output: B_r, B_\theta, B_\phi """

    # This happens to have the same form as the 1st antisymmetric mode
    return get_B_a_1(r, theta, phi,C=C,k=k)


def get_B_s_1(r, theta, phi, C=0.662, k=4.493):
    """ Computes the first (poloidal) symmetric free decay mode.
        Purely poloidal
        Input:
              r, theta, phi: NxNxN arrays containing, repectively,
              the radial, polar and azimuthal coordinates.
              optional: C, kspherical polar coordinates
        Output: B_r, B_\theta, B_\phi """

    # Computes radial component
    Q = N.empty_like(r)
    Q[r<=1.] = r[r<=1.]**(-0.5)*jv(5.0/2.0,k*r[r<=1.])
    Q[r>1.]  = r[r>1.]**(-3.0)*jv(5.0/2.0,k)

    Br = C*Q*(3.0*cos(theta)**2-1)/r

    # Computes polar component
    # X = d(rQ1)/dr
    # http://is.gd/oBIDQo
    X = N.zeros_like(r)
    y = k*r[r<=1.]
    siny = sin(y)
    cosy = cos(y)
    X[r<=1.] = (3*siny/y**2 - siny - 3.*cosy/y)/sqrt(2*pi*y*r[r<=1]) \
      - k*sqrt(r[r<=1])*(3.*siny/y**2-siny-3.*cosy/y)/sqrt(2*pi)*y**(-3./2.)\
    + sqrt(2./pi*r[r<=1])*(-6.*siny*k/y**3+6.*cosy*k/y**2+3*siny*k/y-k*cosy)/(
                                                                       sqrt(y))
                             
    #X[r>1.] = -jv(3.0/2.0,k)/r[r>1.]**2 #TODO!

    Btheta = C*(-sin(theta)*cos(theta)/r)*X

    # Sets azimuthal component
    Bphi   = N.zeros_like(r)

    return Br, Btheta, Bphi
  

def get_B_s_2(r, theta, phi, C=1.330, k=4.493):
    """ Computes the second symmetric free decay mode (one of a
        degenerate pair, with eigenvalue gamma_2=-(5.763)^2 .
        Purely toroidal.
        Input:
              r, theta, phi: NxNxN arrays containing, repectively,
              the radial, polar and azimuthal coordinates.
              optional: C, k
        Output: B_r, B_\theta, B_\phi """

    # Sets radial component
    Br = N.zeros_like(r)

    # Sets polar component
    Btheta = N.zeros_like(r)
    
    # Computes azimuthal component
    Q = N.empty_like(r)
    Q[r<=1.] = r[r<=1.]**(-0.5)*jv(3.0/2.0, k*r[r<=1.])
    Q[r>1.] = r[r>1.]**(-2.0)*jv(3.0/2.0, k)

    Bphi = C*Q*sin(theta)

    return Br, Btheta, Bphi
  


def get_B_s_3(r, theta, phi, C=0.133, k=6.988):
    """ Computes the first (poloidal) symmetric free decay mode.
        Purely poloidal
        Input:
              r, theta, phi: NxNxN arrays containing, repectively,
              the radial, polar and azimuthal coordinates.
              optional: C, kspherical polar coordinates
        Output: B_r, B_\theta, B_\phi """

    # Computes radial component
    S = 35.*cos(theta)**4 - 30.*cos(theta)**2 + 3.0
    
    Q = N.empty_like(r)
    Q[r<=1.] = r[r<=1.]**(-0.5)*jv(9.0/2.0,k*r[r<=1.])
    Q[r>1.]  = r[r>1.]**(-5.0)*jv(9.0/2.0,k)

    Br = C*Q*S/r

    # Computes polar component
    # X = d(rQ1)/dr
    # http://www.wolframalpha.com/input/?i=derivative%28r*r^%28-1%2F2%29*BesselJ%289%2F2%2Ca*r%29+%2C+r%29  
    X = N.zeros_like(r)
    y = k*r[r<=1.]
    siny = sin(y)
    cosy = cos(y)
    A = (105.*siny/y**4 - 105.*cosy/y**3 - 45.*siny/y**2 + siny
                + 10*cosy/y)
    X[r<=1.] =  A/sqrt(2.*pi*y*r[r<=1.])                \
              - k*sqrt(r[r<=1.])*A/sqrt(2.*pi*y**(1.5)) \
              +sqrt(2./pi*r[r<=1.])*(-420.*siny*k/y**5 + 420.*cosy*k/y**4
                                    +195.*siny*k/y**3 - 55.*cosy*k/y**2
                                    - 10.*siny*k/y + k*cosy)/sqrt(y)
              
    #X[r>1.] = -jv(3.0/2.0,k)/r[r>1.]**2 #TODO!
    dSdth = sin(theta)*cos(theta)*(60.-140*cos(theta)**2)

    Btheta = C * X * dSdth/r

    # Sets azimuthal component
    Bphi   = N.zeros_like(r)

    return Br, Btheta, Bphi
  
  
def get_B_s_4(r, theta, phi, C=0.763, k=6.988):
    """ Computes the fourth symmetric free decay mode (one of a
        degenerate pair, with eigenvalue gamma_2=-(5.763)^2 .
        Purely toroidal.
        Input:
              r, theta, phi: NxNxN arrays containing, repectively,
              the radial, polar and azimuthal coordinates.
              optional: C, k
        Output: B_r, B_\theta, B_\phi """

    # Sets radial component
    Br = N.zeros_like(r)

    # Sets polar component
    Btheta = N.zeros_like(r)
    
    # Computes azimuthal component
    Q = N.empty_like(r)
    Q[r<=1.] = r[r<=1.]**(-0.5)*jv(7.0/2.0, k*r[r<=1.])
    Q[r>1.] = r[r>1.]**(-4.0)*jv(7.0/2.0, k)
    S = 3.*sin(theta)*(1.-5.*(cos(theta))**2)
  
    Bphi = -C*Q*sin(theta)

    return Br, Btheta, Bphi
  
  



def spherical_to_cartesian(r, theta, phi, Vr, Vtheta, Vphi):
    from numpy import sin, cos
    
    sin_phi = sin(phi)
    cos_phi = cos(phi)
    sin_theta = sin(theta)
    cos_theta = cos(theta)
    
    x = r*sin_theta*cos_phi
    y = r*sin_theta*sin_phi
    z = r*cos_theta
    
    Vx =       Vr*sin_theta*cos_phi  \
         + Vtheta*cos_theta*cos_phi  \
         -   Vphi*sin_phi
       
    Vy =       Vr*sin_theta*sin_phi  \
         + Vtheta*cos_theta*sin_phi  \
         +   Vphi*cos_phi
       
    
    Vz =       Vr*cos_theta \
         - Vtheta*sin_theta
       
    return x , y, z, Vx, Vy, Vz

# If running as a script, do some test plots
if __name__ == "__main__"  :
    import numpy as N
    import pylab as P

    No = 50

    x = N.linspace(-1.,1.,No)
    y = N.linspace(-1.,1.,No)
    z = N.linspace(-1.,1.,No)

    yy,xx,zz = N.meshgrid(y,x,z) # KEEP THIS BLOODY ORDER IN MIND!

    rr = N.sqrt(xx**2+yy**2+zz**2)
    pp = N.arctan2(yy,xx)  # Chooses the quadrant correctly!
    tt = N.arccos(zz/rr)


    for get_B in [get_B_a_1,get_B_a_2,get_B_a_3,get_B_a_4,
                  get_B_s_1,get_B_s_2,get_B_s_3,get_B_s_4]:
            
        Br, Btheta, Bphi = get_B(rr, tt, pp)

        xx,yy,zz, Bx, By, Bz =  spherical_to_cartesian(rr, tt, pp, Br, Btheta, Bphi)

        imid = No/2
        P.quiver(xx[:,:,imid], yy[:,:,imid], Bx[:,:,imid], By[:,:,imid])
        Z = By[:,:,imid]**2+ Bx[:,:,imid]**2+ Bz[:,:,imid]**2
        X = xx[:,:,imid]
        Y = yy[:,:,imid]
        P.contour(X,Y,Z)
        P.title(get_B.__name__)
        P.show()

        P.quiver(yy[imid,:,:], zz[imid,:,:], By[imid,:,:], Bz[imid,:,:])
        Z = By[imid,:,:]**2+ Bx[imid,:,:]**2+ Bz[imid,:,:]**2
        X = yy[imid,:,:]
        Y = zz[imid,:,:]
        P.contour(X,Y,Z)
        P.title(get_B.__name__)
        P.show()

        P.quiver(xx[:,imid,:], zz[:,imid,:], Bx[:,imid,:], Bz[:,imid,:])
        Z = By[:,imid,:]**2+ Bx[:,imid,:]**2+ Bz[:,imid,:]**2
        X = xx[:,imid,:]
        Y = zz[:,imid,:]
        P.contour(X,Y,Z)
        P.title(get_B.__name__)
        P.show()
