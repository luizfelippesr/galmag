#! /usb/bin/env python
r""" Script that tests the free decay modes
     This is done by calculating:
     $ - \nabla \times \nabla \times \bm{B} / \bm{B} $
     and comparing it with the decay rates $\gamma_l$
     (they should be exactly equal).

     The test is done for increasing grid resolutions.
 """
import galmag
from galmag.util import curl_spherical
from galmag import halo_free_decay_modes as free
import numpy as N
from galmag.Grid import Grid

pi = N.pi
Nop=10
for No in range(350,901,100):

    print '\n\nChecking the free decay modes. Will compute '
    print r'$x = - \nabla \times \nabla \times \bm{B} / \bm{B}/\gamma $'
    print '\nGrid shape: N_r x N_\\theta x N_\\phi = {0}x{0}x{1}'.format(No,Nop)
    print

    grid = Grid([[0.01,1.0], # r range
                [0.000001,N.pi],  # theta range
                [17e-5,2.*N.pi]], # phi range
                resolution=[No,No,Nop],
                grid_type='spherical')
      
    rr = N.array(grid.r_spherical)
    tt = N.array(grid.theta)
    pp = N.array(grid.phi)
    for symmetric in (True, False):
        for n_mode in range(1,5):
            if symmetric:
                symmetry = 's'
                gamma = free.gamma_s[n_mode-1]
            else:
                symmetry = 'a'
                gamma = free.gamma_a[n_mode-1]

            print r'B_{0}_{1}'.format(symmetry,n_mode)
            print '\t', '     ', 'mean(x)      ', ' std(x)'

            B = free.get_mode(rr, tt, pp, n_mode, symmetric)

            curl_B = curl_spherical(rr, tt, pp, B[0], B[1], B[2], order=2)

            curl_curl_B = curl_spherical(rr,
                                         tt,
                                         pp,
                                         curl_B[0],
                                         curl_B[1],
                                         curl_B[2], order=2)

            gamma_computed = [-(curl_curl_B[i]/B[i])
                    for i in range(3)]

            for i, name in zip([0,1,2],['r    ','theta','phi  ']):
                if N.all(B[i]==B[i]*0.0):
                    # If the magnetic field is 0, there is nothing to compare.
                    print '\t', name, 'zero'
                else:
                    #
                    print '\t', name, gamma_computed[i][3:-3,3:-3,3].mean()/gamma,
                    print (gamma_computed[i][3:-3,3:-3,3]/gamma).std()
