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
                  Ctheta_d/Cphi_d -> Parametrisation for the relative
                           contribution of each of the disk modes
                           ranges: [0,pi], [0,2*pi]
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
                  B_d -> Normalization of the disk field (in $\mu G$)
                         recommended range: [0,25]

                  Ralpha_h -> a measure of mean induction by interstellar
                              turbulence at the halo
                             recommended range: [0,10]
                  Romega_h -> a measure of induction by differential rotation
                            $R_\omega = L^2 S / \eta$
                            recommended range: [150,250]
                  R_h -> radius of the dynamo active halo (in kpc)
                         recommended value: 20 kpc
                  B_h -> Normalization of the halo field (in $\mu G$)
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

    if isinstance(params, dict):
        p = params
        assert 'B_d' in p
        assert 'B_h' in p
    else:
        # Reads the parameters into the dictionary
        # (Be careful! unfortunately, for this approach, order matters.)
        p = dict()
        p['Ctheta_d'] = params[0]
        p['Cphi_d'] = params[1]
        p['D_d'] = params[2]
        p['Ralpha_d'] = params[3]
        p['h_d'] = params[4]
        p['R_d'] = params[5]
        p['B_d'] = params[6]
        p['Ralpha_h'] = params[7]
        p['Romega_h'] = params[8]
        p['R_h'] = params[9]
        p['B_h'] = params[10]

    if r_grid is None:
        x = N.linspace(-p['R_h'], p['R_h'], n_grid)
        y = N.linspace(-p['R_h'], p['R_h'], n_grid)
        z = N.linspace(-p['R_h'], p['R_h'], n_grid)

        r = N.empty((3, n_grid, n_grid, n_grid))
        r[0,:,:,:], r[1,:,:,:], r[2,:,:,:] = N.meshgrid(y,x,z)
    else:
        r = r_grid

    if not no_disk:
        p['Cn_d'] = N.array([N.sin(p['Ctheta_d']) * N.cos(p['Cphi_d']),
                             N.sin(p['Ctheta_d']) * N.sin(p['Cphi_d']),
                             N.cos(p['Cphi_d']) ])

        B = get_B_disk(r, p) * p['B_d']

    if not no_halo:
        exit('Not implemented yet')
        B += get_B_halo(r, p) * p['B_h']

    return B

