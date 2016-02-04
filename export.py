""" Interfaces to facilitate exporting the results to different software. """
import numpy as N
from core.disk import get_B_disk
from core.halo import get_B_halo

paramter_names = ['Cn_d',
                  'D_d',
                  'Ralpha_d',
                  'h_d',
                  'Ralpha_h',
                  'Romega_h',
                  'B_h',
                  'R_h',
                  'R_d']

recommended_paramter_ranges = {'Cn_d': [[-25,25],[-25,25],[-25,25]],
                               'D_d': [-10,0],
                               'Ralpha_d': [1,3],
                               'h_d': [0.25,1],
                               'Ralpha_h': [0,10],
                               'Romega_h': [150,250],
                               'B_h': [0,25]}

def get_B_IMAGINE(p,
                  coordinates='cartesian',
                  no_halo=False,
                  no_disk=False,
                  r_grid=None,
                  n_grid=100
                  ):
    """ This function provides an interface to the IMAGINE project.

        Input: p -> a dictionary
                  Cn_d -> 3-array containing the normalisation of the disk
                          field components (in $\mu$G)
                          recommended range: [[-25,25],[-25,25],[-25,25]]
                  D_d -> Dynamo number of the disk
                          recommended range: [-10,0]
                  Ralpha_d -> a measure of mean induction by interstellar
                            turbulence at the disk
                            $R_\alpha = L \alpha_0 / \eta$
                            recommended range: [1,3]
                  h_d -> scale height of the dynamo active disk (in kpc)
                         recommended range: [0.25,1]
                  R_d -> radius of the dynamo active disk (in kpc)
                         recommended value: 20 kpc
                  Ralpha_h -> a measure of mean induction by interstellar
                              turbulence at the halo
                             recommended range: [0,10]
                  Romega_h -> a measure of induction by differential rotation
                            $R_\omega = L^2 S / \eta$
                            recommended range: [150,250]
                  R_h -> radius of the dynamo active halo (in kpc)
                         recommended value: 20 kpc
                  B_h -> Normalization of the halo field (in $\mu$G)
                         recommended range: [0,25]

               optional: r_grid -> 3xNxNxN array of coordinates
                                   Default: N=n_grid
                         n_grid -> In the absence of r_grid, a uniform
                                   cartesian grid is used, with N=n_grid
                         coordinates -> chooses between 'spherical',
                                        'cartesian' and 'cylindical'
                         no_halo -> omits the halo component

        Output: a 3xNxNxN array containing the magnetic field.
    """

    assert isinstance(p, dict)
    assert 'R_h' in p

    if r_grid is None:
        x = N.linspace(-p['R_h'], p['R_h'], n_grid)
        y = N.linspace(-p['R_h'], p['R_h'], n_grid)
        z = N.linspace(-p['R_h'], p['R_h'], n_grid)

        r = N.empty((3, n_grid, n_grid, n_grid))
        r[0,:,:,:], r[1,:,:,:], r[2,:,:,:] = N.meshgrid(y,x,z)
    else:
        r = r_grid

    B = N.zeros_like(r)

    if not no_disk:
        assert N.all([item in p for item in ['Cn_d', 'D_d', 'Ralpha_d', 'R_d',
                                             'h_d']])
        B += get_B_disk(r, p)

    if not no_halo:
        assert N.all([item in p for item in ['Ralpha_h', 'Romega_h', 'B_h', 
                                             'R_h']])
        B += get_B_halo(r, p) * p['B_h']

    return B

