
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
       
    return x, y, z, Vx, Vy, Vz
  

def cylindrical_to_cartesian(r, phi, z, Vr, Vphi, Vz):
    from numpy import sin, cos
    
    sin_phi = sin(phi) 
    cos_phi = cos(phi) 
    
    x = r*cos_phi
    y = r*sin_phi
    
    Vx = (Vr*cos_phi - Vphi*sin_phi)
    Vy = (Vr*sin_phi + Vphi*cos_phi)    
              
    return x, y, z, Vx, Vy, Vz
