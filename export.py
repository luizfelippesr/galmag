""" Interfaces to facilitate exporting the results to different software. """
import numpy as N
from core.disk import get_B_disk
from core.halo import get_B_halo

def get_B_IMAGINE(params, 
                  coordinates='cartesian',
                  no_halo=True,
                  no_disk=False,
                  r_grid=None,
                  n_grid=100
                  ):
    """ It is more convenient in the IMAGINE project to have an array 
        of parameters, instead of a dictionary. This function provides 
        this interface.
        
        Input: params -> a Numpy array of parameters
            
    
               optional: r_grid -> 3xNxNxN array of coordinates
                                   Default: N=n_grid
                         n_grid -> In the absence of r_grid, a uniform 
                                   cartesian grid is used, with N=n_grid
                         coordinates -> chooses between 'spherical', 
                                        'cartesian' and 'cylindical'
                         no_halo -> omits the halo component
                         
        Output: a 3xNxNxN array containing the magnetic field.
    """
    
    if isinstance(params, dict):
        p = params
    else:
        # Reads the parameters into the dictionary
        # (Be careful! unfortunately, for this approach, order matters.)
        p = dict()
        p['Cn'] = params[:3]
        p['D'] = params[3]
        p['Rgamma'] = params[4]
        p['Ralpha'] = params[5]
        p['h'] = params[6]
    
    if r_grid is None:
        x = N.linspace(-p['Rgamma'], p['Rgamma'], n_grid)
        y = N.linspace(-p['Rgamma'], p['Rgamma'], n_grid)
        z = N.linspace(-p['Rgamma'], p['Rgamma'], n_grid)
        
        r = N.empty((3, n_grid, n_grid, n_grid))
        r[0,:,:,:], r[1,:,:,:], r[2,:,:,:] = N.meshgrid(y,x,z) 
        
    if not no_disk:
        B = get_B_disk(r, p)
        
    if not no_halo:
        exit('Not implemented yet')
        B += get_B_halo(r, p)
    
    return B
  
  
# If running as a script, do some tests
if __name__ == "__main__"  :
    print get_B_IMAGINE(N.array([1, 0, 0, -15, 15, 1.0, 0.5]))