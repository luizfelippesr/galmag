#! /usb/bin/env python
r""" Script that tests the free decay modes
     This is done by calculating:
     $ - \nabla \times \nabla \times \bm{B} / \bm{B} $
     and comparing it with the decay rates $\gamma_l$
     (they should be exactly equal).

     The test is done for increasing grid resolutions.
 """
#! /usb/bin/env python
r""" Script that tests the free decay modes
     This is done by calculating:
     $ - \nabla \times \nabla \times \bm{B} / \bm{B} $
     and comparing it with the decay rates $\gamma_l$
     (they should be exactly equal).

     The test is done for increasing grid resolutions.
 """
import galmag
import matplotlib.pyplot as plt
from galmag.util import curl_spherical
from galmag import halo_free_decay_modes as free
import numpy as np
from galmag.Grid import Grid

class TestHaloFreeDecayModes():
    def test_s1(self):
        self.__check_nmode(1, True)
    def test_s2(self):
        self.__check_nmode(2, True)
    def test_s3(self):
        self.__check_nmode(3, True)
    def test_s4(self):
        self.__check_nmode(4, True)

    def test_a1(self):
        self.__check_nmode(1, False)
    def test_a2(self):
        self.__check_nmode(2, False)
    def test_a3(self):
        self.__check_nmode(3, False)
    def test_a4(self):
        self.__check_nmode(4, False)

    def __check_nmode(self, n_mode, symmetric):
        Nop=21
        No = 201

        grid = Grid([[0.01,1.0], # r range
                    [0.000001,np.pi],  # theta range
                    [17e-5,2.*np.pi]], # phi range
                    resolution=[No,No,Nop],
                    grid_type='spherical')

        self.rr = grid.r_spherical
        self.tt = grid.theta
        self.pp = grid.phi

        if symmetric:
            symmetry = 's'
            gamma = free.gamma_s[n_mode-1]
        else:
            symmetry = 'a'
            gamma = free.gamma_a[n_mode-1]

        print(r'B_{0}_{1}'.format(symmetry,n_mode), flush=True)
        print('\t', 'comp\t max err')

        B = free.get_mode(self.rr, self.tt, self.pp, n_mode, symmetric)

        curl_B = curl_spherical(self.rr, self.tt, self.pp, B[0], B[1], B[2])
        curl_curl_B = curl_spherical(self.rr, self.tt, self.pp,
                                      curl_B[0], curl_B[1], curl_B[2])

        for i, name in zip([0,1,2],['r    ','theta','phi  ']):
            if np.all(B[i]==B[i]*0.0):
                # If the magnetic field is 0, there is nothing to compare.
                print('\t', name, '\t zero')
            else:
                # Removes the boundaries, as the double curl accumulates errors
                # which are unlikely to be present in the actual GalMag use
                valid_slice = (slice(4,-4),slice(4,-4),4)
                curl_curl_Bi = curl_curl_B[i][valid_slice]
                Bi = B[i][valid_slice]
                max_error = (-curl_curl_Bi/gamma/Bi-1).max()
                print('\t {0}\t {1:.2%}'.format(name, max_error))
                assert np.allclose(curl_curl_Bi, -gamma*Bi, atol=5e-5,
                                   rtol=1e-5)


if __name__ == "__main__":
    test_free = TestHaloFreeDecayModes()
    for test in dir(test_free):
        if 'test' in test:
            getattr(test_free, test)()
