import numpy as np
from d2o import distributed_data_object

def distribute_function(f, x):
    """Evaluates a function f using only local data from the d2o x.
       After that, collects the data into a single d2o and returns
       the results.
       If x is not a d2o, simple evaluates the funtion for x.
    """
    if not isinstance(x, distributed_data_object):
        return f(x)
    local_x = x.get_local_data()
    local_y = f(local_x)
    y = x.copy_empty()
    y.set_local_data(local_y)
    return y

def derive(V, dx, axis=0):
    """Computes the numerical derivative of a function specified over a
       3 dimensional uniform grid. Uses second order finite differences.
       Input: V -> NxNxN array (either numpy or d2o)
              dx -> grid spacing
              axis -> specifies ove which axis the derivative should be
                      performed. Default: 0.
       Output: the derivative, dV/dx
    """
    if isinstance(V, distributed_data_object):
        dVdx = V.copy_empty()
    else:
        dVdx = np.empty_like(V)

    if axis==0:
        dVdx[1:-1,:,:] = (V[2:,:,:] - V[:-2,:,:])/2.0/dx
        dVdx[0,:,:]  = (-3.0*V[0,:,:]  +4.0*V[1,:,:]  - V[2,:,:])/dx/2.0
        dVdx[-1,:,:] = ( 3.0*V[-1,:,:] -4.0*V[-2,:,:] + V[-3,:,:])/dx/2.0
    elif axis==1:
        dVdx[:,1:-1,:] = (V[:,2:,:] - V[:,:-2,:])/2.0/dx
        dVdx[:,0,:]  = (-3.0*V[:,0,:]  +4.0*V[:,1,:]  - V[:,2,:])/dx/2.0
        dVdx[:,-1,:] = ( 3.0*V[:,-1,:] -4.0*V[:,-2,:] + V[:,-3,:])/dx/2.0
    elif axis==2:
        dVdx[:,:,1:-1] = (V[:,:,2:] - V[:,:,:-2])/2.0/dx
        dVdx[:,:,0]  = (-3.0*V[:,:,0]  +4.0*V[:,:,1]  - V[:,:,2])/dx/2.0
        dVdx[:,:,-1] = ( 3.0*V[:,:,-1] -4.0*V[:,:,-2] + V[:,:,-3])/dx/2.0

    return dVdx

def curl_spherical(rr, tt, pp, Br, Bt, Bp):
    """Computes the curl of a vector in spherical coordinates.
       Input: rr, tt, pp -> NxNxN arrays containing the r, theta and phi coords
              Br, Bt, Bp -> NxNxN arrays r, theta and phi components of the
                            vector in the same coordinate grid.
       Return: the components of the curl.
    """
    # Gets grid spacing (assuming uniform grid spacing)
    dr = rr[1,0,0]-rr[0,0,0]
    if dr==0:
        raise ValueError('Invalid spacing for dr')
    dtheta  = tt[0,1,0] - tt[0,0,0]
    if dtheta==0:
        raise ValueError('Invalid spacing for dtheta')
    n, n, n_p = pp.shape
    if n_p==1:
        # Assuming axisymmetry
        dphi = None
    else:
        dphi = pp[0,0,1] - pp[0,0,0]
    if dphi==0:
        raise ValueError('Invalid spacing for dphi')

    # Computes partial derivatives
    dBr_dr = derive(Br, dr)
    dBr_dtheta = derive(Br, dtheta, axis=1)

    dBtheta_dr = derive(Bt, dr)
    dBtheta_dtheta = derive(Bt, dtheta, axis=1)

    dBphi_dr = derive(Bp, dr)
    dBphi_dtheta = derive(Bp, dtheta, axis=1)

    # Auxiliary
    tant = distribute_function(np.tan, tt)

    if dphi:
        dBr_dphi = derive(Br, dphi, axis=2)
        dBphi_dphi = derive(Bp, dphi, axis=2)
        dBtheta_dphi = derive(Bt, dphi, axis=2)
        sint = distribute_function(np.sin, tt)

    # Components of the curl
    cBr = dBphi_dtheta/rr + Bp/tant/rr
    if dphi:
        cBr -= dBtheta_dphi/sint/rr

    cBtheta = -Bp/rr - dBphi_dr
    if dphi:
      cBtheta += (dBr_dphi/sint)/rr

    cBphi = (Bt + rr*dBtheta_dr - dBr_dtheta)/rr

    return cBr, cBtheta, cBphi
