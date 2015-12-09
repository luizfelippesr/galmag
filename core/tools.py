
def spherical_to_cartesian(r, theta, phi, Vr, Vtheta, Vphi):
    """ Simple routine to convert a field in spherical coordinates
        to cartesian coordinates. 
        Input: r, theta, phi -> radial, polar and azimuthal coordinates,
                                respectively.
               Vr, Vtheta, Vphi -> components of the field in spherical
                                   coordinates. 
        Output: x, y, z, Vx, Vy, Vz
    """ 
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
       
    return x, y, z, Vx, Vy, Vz
  

def cylindrical_to_cartesian(r, phi, z, Vr, Vphi, Vz):
    """ Simple routine to convert a field in cylindrical coordinates
        to cartesian coordinates. 
        Input: r, phi, z -> radial, azimuthal and vertical coordinates,
                                respectively.
               Vr, Vphi, Vz -> components of the field in spherical
                                   coordinates. 
        Output: x, y, z, Vx, Vy, Vz
    """ 
    from numpy import sin, cos
    
    sin_phi = sin(phi) 
    cos_phi = cos(phi) 
    
    x = r*cos_phi
    y = r*sin_phi
    
    Vx = (Vr*cos_phi - Vphi*sin_phi)
    Vy = (Vr*sin_phi + Vphi*cos_phi)    
              
    return x, y, z, Vx, Vy, Vz
