# Copyright (C) 2017, 2018  Luiz Felippe S. Rodrigues <luiz.rodrigues@ncl.ac.uk>
#
# This file is part of GalMag.
#
# GalMag is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# GalMag is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GalMag.  If not, see <http://www.gnu.org/licenses/>.
#
"""
Auxiliary functions.
"""
import numpy as np
from d2o import distributed_data_object

def distribute_function(f, x):
    """
    Evaluates a function f using only local data from the d2o x.
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
    """
    Computes the numerical derivative of a function specified over a
    3 dimensional uniform grid. Uses second order finite differences.

    Obs: extremities will use forward or backwards finite differences.

    Parameters
    ----------
    V : array_like
        NxNxN array (either numpy or d2o)
    dx : float
        grid spacing
    axis : int
        specifies over which axis the derivative should be performed. Default: 0.
    order : int
        order of the finite difference method. Default: 2

    Returns
    -------
    same as V
        The derivative, dV/dx

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
    r"""
    Computes the curl of a vector in spherical coordinates.

    Parameters
    ----------
    rr/tt/pp : array_like
        NxNxN arrays containing the :math:`r`, :math:`\theta` and :math:`\phi`
        coordinates
    Br/Bt/Bp : array_like
        NxNxN arrays :math:`r`, :math:`\theta` and :math:`\phi` components of
        the vector in the same coordinate grid.

    Returns
    -------
    list
        three NxNxN arrays containing the components of the curl.
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

def arctan2(B1, B2):
    """
    A distributed version of numpy.arctan2

    (which works efficiently with a `distributed_data_object`)
    """
    if ((not isinstance(B1, distributed_data_object)) or
        (not isinstance(B2, distributed_data_object))):
        return np.arctan2(B1,B2)

    # Gets local data
    local_B1 = B1.get_local_data()
    local_B2 = B2.get_local_data()
    # Does the computation (locally)
    local_atan = np.arctan2(local_B1,local_B2)
    # Prepares the distributed object and returns
    global_atan = B1.copy_empty()
    global_atan.set_local_data(local_atan, copy=False)
    return global_atan
