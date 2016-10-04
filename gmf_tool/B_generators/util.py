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

def derive(V, dx, axis=0, order=2):
    """Computes the numerical derivative of a function specified over a
       3 dimensional uniform grid. Uses second order finite differences.
       Input: V -> NxNxN array (either numpy or d2o)
              dx -> grid spacing
              axis -> specifies ove which axis the derivative should be
                      performed. Default: 0.
              order -> order of the finite difference method. Default: 2
       Output: the derivative, dV/dx

       Obs: extremities will use forward or backwards finite differences.
    """
    if isinstance(V, distributed_data_object):
        dVdx = V.copy_empty()
    else:
        dVdx = np.empty_like(V)

    if axis==0:
        if order==2:
            dVdx[1:-1,:,:] = (V[2:,:,:] - V[:-2,:,:])/2.0/dx
            dVdx[0,:,:]  = (-3.0*V[0,:,:]  +4.0*V[1,:,:]  - V[2,:,:])/dx/2.0
            dVdx[-1,:,:] = ( 3.0*V[-1,:,:] -4.0*V[-2,:,:] + V[-3,:,:])/dx/2.0
        elif order==4:
            dVdx[2:-2,:,:] = ( V[:-4,:,:]/12.0 - V[4:,:,:]/12.0
                             - V[1:-3,:,:]*(2./3.) + V[3:-1,:,:]*(2./3.) )/dx

            a0 = -25./12.; a1=4.0; a2=-3.0; a3=4./3.; a4=-1./4.
            dVdx[0:2,:,:] = ( V[0:2,:,:]*a0 + V[1:3,:,:]*a1
                            + V[2:4,:,:]*a2 + V[3:5,:,:]*a3
                            + V[3:5,:,:]*a4 )/dx

            dVdx[-2:,:,:] = - ( V[-2:,:,:]*a0 + V[-3:-1,:,:]*a1
                              + V[-4:-2,:,:]*a2 + V[-5:-3,:,:]*a3
                              + V[-6:-4,:,:]*a4 )/dx
        else:
            raise ValueError('Only order 2 and 4 are currently implemented.')
    elif axis==1:
        if order==2:
            dVdx[:,1:-1,:] = (V[:,2:,:] - V[:,:-2,:])/2.0/dx
            dVdx[:,0,:]  = (-3.0*V[:,0,:]  +4.0*V[:,1,:]  - V[:,2,:])/dx/2.0
            dVdx[:,-1,:] = ( 3.0*V[:,-1,:] -4.0*V[:,-2,:] + V[:,-3,:])/dx/2.0
        elif order==4:
            dVdx[:,2:-2,:] = ( V[:,:-4,:]/12.0 - V[:,4:,:]/12.0
                             - V[:,1:-3,:]*(2./3.) + V[:,3:-1,:]*(2./3.) )/dx

            a0 = -25./12.; a1=4.0; a2=-3.0; a3=4./3.; a4=-1./4.
            dVdx[:,0:2,:] = ( V[:,0:2,:]*a0 + V[:,1:3,:]*a1
                            + V[:,2:4,:]*a2 + V[:,3:5,:]*a3
                            + V[:,3:5,:]*a4 )/dx

            dVdx[:,-2:,:] = - ( V[:,-2:,:]*a0 + V[:,-3:-1,:]*a1
                              + V[:,-4:-2,:]*a2 + V[:,-5:-3,:]*a3
                              + V[:,-6:-4,:]*a4 )/dx
        else:
            raise ValueError('Only order 2 and 4 are currently implemented.')
    elif axis==2:
        if order==2:
            dVdx[:,:,1:-1] = (V[:,:,2:] - V[:,:,:-2])/2.0/dx
            dVdx[:,:,0]  = (-3.0*V[:,:,0]  +4.0*V[:,:,1]  - V[:,:,2])/dx/2.0
            dVdx[:,:,-1] = ( 3.0*V[:,:,-1] -4.0*V[:,:,-2] + V[:,:,-3])/dx/2.0
        elif order==4:
            dVdx[:,:,2:-2] = ( V[:,:,:-4]/12.0 - V[:,:,4:]/12.0
                             - V[:,:,1:-3]*(2./3.) + V[:,:,3:-1]*(2./3.) )/dx

            a0 = -25./12.; a1=4.0; a2=-3.0; a3=4./3.; a4=-1./4.
            dVdx[:,:,0:2] = ( V[:,:,0:2]*a0 + V[:,:,1:3]*a1
                            + V[:,:,2:4]*a2 + V[:,:,3:5]*a3
                            + V[:,:,3:5]*a4 )/dx

            dVdx[:,:,-2:] = - ( V[:,:,-2:]*a0 + V[:,:,-3:-1]*a1
                              + V[:,:,-4:-2]*a2 + V[:,:,-5:-3]*a3
                              + V[:,:,-6:-4]*a4 )/dx
        else:
            raise ValueError('Only order 2 and 4 are currently implemented.')

    return dVdx

def curl_spherical(rr, tt, pp, Br, Bt, Bp, order=2):
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
    dBr_dr = derive(Br, dr, order=order)
    dBr_dtheta = derive(Br, dtheta, axis=1, order=order)

    dBtheta_dr = derive(Bt, dr, order=order)
    dBtheta_dtheta = derive(Bt, dtheta, axis=1, order=order)

    dBphi_dr = derive(Bp, dr, axis=0, order=order)
    dBphi_dtheta = derive(Bp, dtheta, axis=1, order=order)

    # Auxiliary
    tant = distribute_function(np.tan, tt)

    if dphi:
        dBr_dphi = derive(Br, dphi, axis=2, order=order)
        dBtheta_dphi = derive(Bt, dphi, axis=2, order=order)
        dBphi_dphi = derive(Bp, dphi, axis=2, order=order)
        sint = distribute_function(np.sin, tt)

    # Components of the curl
    cBr = dBphi_dtheta/rr + Bp/tant/rr
    if dphi:
        cBr -= dBtheta_dphi/sint/rr

    cBtheta = - dBphi_dr - Bp/rr
    if dphi:
      cBtheta += dBr_dphi/sint/rr

    cBphi = Bt/rr + dBtheta_dr - dBr_dtheta/rr

    return cBr, cBtheta, cBphi

def simpson(f, r):
    """Integrates over the last axis"""
    shape_r = r.shape
    integ = f.copy()

    if len(shape_r)==1:
        h = r[1]-r[0]
        integ[1:-1] += f[1:-1]
        integ[1:-1:2] += f[1:-1:2]*2.0
        return integ.sum()*h/3.0
    elif len(shape_r)==2:
        h = r[0,1]-r[0,0]
        integ[:,1:-1] += f[:,1:-1]
        integ[:,1:-1:2] += f[:,1:-1:2]*2.0
        return integ.sum(axis=-1)*h/3.0

    elif len(shape_r)==3:
        h = r[0,0,1]-r[0,0,0]
        integ[:,:,1:-1] += f[:,:,1:-1]
        integ[:,:,1:-1:2] += f[:,:,1:-1:2]*2.0
        return integ.sum(axis=-1)*h/3.0
    else:
        raise NotImplementedError
