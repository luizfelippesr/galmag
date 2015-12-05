import numpy as N
def simple_V(r,V0,s0):
    
    V = N.zeros_like(r)
    rho = r[0,:,:,:]
    theta = r[1,:,:,:]
    
    V[2,:,:,:] = V0 * ( 1.0 - N.exp(-rho*N.sin(theta)/s0) )
    
    return V
  
  
    