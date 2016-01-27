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
        
        Input: params -> a Numpy array of parameters or a dictionary
                  Cn_d -> 3-array containing the coefficients for the disk field
                  D_d -> Dynamo number of the disk
                  Ralpha_d -> a measure of mean induction by interstellar
                            turbulence at the disk
                            $R_\alpha = L \alpha_0 / \eta$
                  h_d -> scale height of the dynamo active disk
                  R_d -> radius of the dynamo active disk
                  Ralpha_h -> a measure of mean induction by interstellar
                            turbulence at the halo
                  Romega_h -> a measure of induction by differential rotation
                            $R_\omega = L^2 S / \eta$
                  R_h -> radius of the dynamo active halo


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
        p['Cn_d'] = params[:3]
        p['D_d'] = params[3]
        p['Ralpha_d'] = params[4]
        p['h_d'] = params[5]
        p['R_d'] = params[6]
        p['Ralpha_h'] = params[7]
        p['Romega_h'] = params[8]
        p['R_h'] = params[9]
    
    if r_grid is None:
        x = N.linspace(-p['R_h'], p['R_h'], n_grid)
        y = N.linspace(-p['R_h'], p['R_h'], n_grid)
        z = N.linspace(-p['R_h'], p['R_h'], n_grid)
        
        r = N.empty((3, n_grid, n_grid, n_grid))
        r[0,:,:,:], r[1,:,:,:], r[2,:,:,:] = N.meshgrid(y,x,z) 
    else:
        r = r_grid

    if not no_disk:
        B = get_B_disk(r, p)
        
    if not no_halo:
        exit('Not implemented yet')
        B += get_B_halo(r, p)
    
    return B
  
  
# If running as a script, do some tests
if __name__ == "__main__"  :
    print get_B_IMAGINE(N.array([1, 0, 0, -15, 15, 1.0, 0.5]))
