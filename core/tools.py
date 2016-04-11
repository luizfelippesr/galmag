import numpy as N

def cartesian_to_spherical_grid(r):
    """ Converts a cartesian grid to a spherical one. """
    r_sph = N.empty_like(r)
    # rho
    r_sph[0,...] = N.sqrt(r[0,...]**2+r[1,...]**2+r[2,...]**2)
    # theta
    r_sph[1,...] = N.arccos(r[2,...]/r_sph[0,...])
    # phi
    r_sph[2,...] = N.arctan2(r[1,...],r[0,...])

    return r_sph


def spherical_to_cartesian_grid(r_sph):
    """ Converts a cartesian grid to a spherical one. """
    r = N.empty_like(r_sph)
    sin_phi = N.sin(r_sph[2,...])
    cos_phi = N.cos(r_sph[2,...])
    sin_theta = N.sin(r_sph[1,...])
    cos_theta = N.cos(r_sph[1,...])
    # x
    r[0,...] = r_sph[0,...]*sin_theta*cos_phi
    # y
    r[1,...] = r_sph[0,...]*sin_theta*sin_phi
    # z
    r[2,...] = r_sph[0,...]*cos_theta

    return r

def cylindrical_to_cartesian_grid(r_cyl):
    """ Converts a cartesian grid to a spherical one. """
    r = N.empty_like(r_cyl)
    sin_phi = N.sin(r_cyl[1,...])
    cos_phi = N.cos(r_cyl[1,...])
    # x
    r[0,...] = r_cyl[0,...]*cos_phi
    # y
    r[1,...] = r_cyl[0,...]*sin_phi
    # z
    r[2,...] = r_cyl[2,...]

    return r


def cartesian_to_spherical(r_sph, V):
    """ Converts a vector field V to spherical coordinates.
        Inputs: r_sph -> spherical coords grid
                V -> Vector field
        Output: V_sph
    """
    V_sph = N.empty_like(V)
    cos_theta = N.cos(r_sph[1])
    sin_theta = N.sin(r_sph[1])
    cos_phi = N.cos(r_sph[2])
    sin_phi = N.sin(r_sph[2])

    V_sph[0] = V[0]*sin_theta*cos_phi \
             + V[1]*sin_theta*sin_phi \
             + V[2]*cos_theta
    V_sph[1] = V[0]*cos_theta*cos_phi \
             + V[1]*cos_theta*sin_phi \
             - V[2]*sin_theta
    V_sph[2] = - V[0]*sin_phi \
               + V[1]*cos_phi 

    return V_sph


def spherical_to_cartesian(r, theta, phi, Vr, Vtheta, Vphi, return_coord=False):
    """ Simple routine to convert a field in spherical coordinates
        to cartesian coordinates.
        Input: r, theta, phi -> radial, polar and azimuthal coordinates,
                                respectively.
               Vr, Vtheta, Vphi -> components of the field in spherical
                                   coordinates.
               optional: return_coord -> switch to retun the coordinates as well
        Output: Vx, Vy, Vz
                or
                x, y, z, Vx, Vy, Vz if return_coord=True
    """

    sin_phi = N.sin(phi)
    cos_phi = N.cos(phi)
    sin_theta = N.sin(theta)
    cos_theta = N.cos(theta)
    Vx =       Vr*sin_theta*cos_phi  \
         + Vtheta*cos_theta*cos_phi  \
         -   Vphi*sin_phi

    Vy =       Vr*sin_theta*sin_phi  \
         + Vtheta*cos_theta*sin_phi  \
         +   Vphi*cos_phi


    Vz =       Vr*cos_theta \
         - Vtheta*sin_theta

    if return_coord:
        x = r*sin_theta*cos_phi
        y = r*sin_theta*sin_phi
        z = r*cos_theta

        return x, y, z, Vx, Vy, Vz
    else:
        return Vx, Vy, Vz


def cylindrical_to_cartesian(r, phi, z, Vr, Vphi, Vz):
    """ Simple routine to convert a field in cylindrical coordinates
        to cartesian coordinates.
        Input: r, phi, z -> radial, azimuthal and vertical coordinates,
                                respectively.
               Vr, Vphi, Vz -> components of the field in spherical
                                   coordinates.
        Output: x, y, z, Vx, Vy, Vz
    """
    sin_phi = N.sin(phi)
    cos_phi = N.cos(phi)

    x = r*cos_phi
    y = r*sin_phi

    Vx = (Vr*cos_phi - Vphi*sin_phi)
    Vy = (Vr*sin_phi + Vphi*cos_phi)

    return x, y, z, Vx, Vy, Vz

def generate_grid(n_grid, xlim=[-1.,1.], ylim=[-1.,1.], zlim=[-1.,1.],
                  return_dxdydz=False):
    """ Generates a uniform grid.
        Input: n_grid -> number of points
               optional: xlim/ylim/zlim -> a list/array contaning the limits of
                         the grid in the corresponding coordinate.
        Output: if zlim!=None, a 3x(n_grid)x(n_grid)x(n_grid) array containing
                coordinates.
                elif zlim=None, a 3x(n_grid)x(n_grid)x(1) array containing
                coordinates.
    """
    x = N.linspace(xlim[0],xlim[1], n_grid)
    if ylim:
        n_gridy = n_grid
        y = N.linspace(ylim[0],ylim[1], n_grid)
    else:
        n_gridy = 1
        y = N.array([0.,])
    if zlim:
        n_gridz = n_grid
        z = N.linspace(zlim[0],zlim[1], n_grid)
    else:
        n_gridz = 1
        z = N.array([0.,])
    r = N.empty((3, n_grid, n_gridy, n_gridz))
    r[1,:,:,:], r[0,:,:,:], r[2,:,:,:] = N.meshgrid(y,x,z)

    if return_dxdydz:
        dx = x[1]-x[0]
        if ylim:
            dy = y[1]-y[0]
        else:
            dy = None
        if zlim:
            dz = z[1]-z[0]
        else:
            dz = None
        return r, dx, dy, dz
    return r


def get_param(pdict, param, default=None):
    """ Convenience auxiliary function: if param is in pdict,
        returns its value, otherwise returns default (and stores it)  .
    """
    if param not in pdict:
        pdict[param] = default
    return pdict[param]
