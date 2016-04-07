import numpy as N
import core.disk as d 
import core.halo as h
import core.tools as tools

class field(N.ndarray):
    """ The magnetic field class.
        Stores the numpy array containing the magnetic field as well as the
        parameters and potentially useful intermediate quantities.
    """ 
    def __new__(cls, params, no_disk=False, no_halo=False, info=None,
                grid_geometry='cartesian', store_spherical=True, r=None):
        
        if grid_geometry=='cartesian':
            # Generates a cartesian grid
            n_grid = tools.get_param(params, 'ngrid', default=300)
            R_h = tools.get_param(params, 'R_h', default=20)
            r = tools.generate_grid(n_grid, xlim=[-R_h, R_h],
                                            ylim=[-R_h, R_h],
                                            zlim=[-R_h, R_h])
            if store_spherical:
                r_sph = tools.cartesian_to_spherical_grid(r)
        elif grid_geometry=='spherical':
            n_grid = tools.get_param(params, 'ngrid', default=300)
            R_h = tools.get_param(params, 'R_h', default=20)
            r_sph = tools.generate_grid(n_grid, xlim=[1e-2*R_h, R_h],
                                                ylim=[1e-1, N.pi],
                                                zlim=[1e-1, 2*N.pi])
            r = tools.spherical_to_cartesian_grid(r_sph)
        elif grid_geometry=='spherical_no_phi':
            n_grid = tools.get_param(params, 'ngrid', default=300)
            R_h = tools.get_param(params, 'R_h', default=20)
            r_sph = tools.generate_grid(n_grid, xlim=[1e-2*R_h, R_h],
                                                ylim=[1e-1, N.pi],
                                                zlim=None)
            r = tools.spherical_to_cartesian_grid(r_sph)
        elif grid_geometry=='custom':
            R_h = tools.get_param(params, 'R_h', default=20)
            if not isinstance(r, N.ndarray):
                raise ValueError('Please provide the customized grid.')
            if store_spherical:
                r_sph = tools.cartesian_to_spherical_grid(r)
        else:
            raise ValueError('Unknown grid_geometry option.')

        # B array is an already formed ndarray instance
        # We first cast to be our class type
        B = N.zeros_like(r).view(cls)
        # add the new attribute to the created instance
        B.grid = r
        if store_spherical:
            B.grid_sph = r_sph
        B.info = info
        B.parameters = params
        # Finally, we must return the newly created object:
        if not no_disk:
            tools.get_param(params, 'h_d', default=0.4)
            tools.get_param(params, 'Ralpha_d', default=0.6)
            tools.get_param(params, 'D_d', default=-20.0)
            tools.get_param(params, 'R_d', default=10.0)
            tools.get_param(params, 'Cn_d', default=N.array([1,1,1.]))
            field.compute_disk_field(B)
            if store_spherical:
                B.sph = tools.cartesian_to_spherical(B.grid_sph, B)
        if not no_halo:
            tools.get_param(params, 'Ralpha_h', default=0.6)
            tools.get_param(params, 'Romega_h', default=200.0)
            field.compute_halo_field(B)

        return B

    def compute_disk_field(self, params=None):
        """ Produces the magnetic field of a disk.
            If a parameters dictionary is supplied, updates the parameters
            atribute.
        """
        if params is not None:
            self.parameters.update(params)
        self.disk = d.get_B_disk(self.grid, self.parameters)
        self += self.disk
        self.__clean_derived_quantities()

    def compute_halo_field(self, params=None):
        """ Produces the magnetic field of a halo.
            If a parameters dictionary is supplied, updates the parameters
            atribute.
        """
        if params is not None:
            self.parameters.update(params)

        self.halo = h.get_B_halo(self.grid, self.parameters)
        self += self.halo
        self.__clean_derived_quantities()

    def Rsun(self, store=True, return_position=False, sph=False,
             solar_rmax=8.4, solar_rmin=7.2,
             solar_zmax=29e-3, solar_zmin=23e-3 ):
        """ Returns the magnetic field at the solar radius.
            Optional inputs:
              store -> stores the result for later use. Default: True
              solar_rmax -> maximum distance from the MW centre, in kpc.
                Default: 8.4 kpc
              solar_rmin -> minimum distance from the MW centre, in kpc.
                Default: 7.2 kpc
              solar_zmax -> maximum distance from the MW plane, in kpc.
                Default: 0.029 kpc
              solar_zmin -> minimum distance from the MW plane, in kpc.
                Default: 0.023 kpc
              return_position -> if True returns two (3xNxNxN) arrays
                containing the field and and its cartesian coordinates.

        """
        B_Rsun = None
        if not sph and hasattr(self, '__Rsun__'):
            B_Rsun = self.__Rsun__
        elif sph and hasattr(self, '__Rsun_sph__'):
            B_Rsun = self.__Rsun_sph__
            
        if B_Rsun == None:
            if hasattr(self, 'grid_sph'):
                rr = self.grid_sph[0]
            else:
                rr = N.sqrt(self.grid[0]**2 +
                            self.grid[1]**2 +
                            self.grid[2]**2)

            zz = self.grid[2]

            Rsun_idx  = (rr <= solar_rmax)
            Rsun_idx *= (rr >= solar_rmin)
            Rsun_idx *= (zz <= solar_zmax)
            Rsun_idx *= (zz >= solar_zmin)
            if not sph:
                B_Rsun = N.array([self[i][Rsun_idx] for i in range(3)])
                self.__Rsun__ = B_Rsun
            else:
                B_Rsun = N.array([self.sph[i][Rsun_idx] for i in range(3)])
                self.__Rsun_sph__ = B_Rsun

        if not return_position:
            return B_Rsun
        else:
            pos_Rsun = N.array([self.grid[i][Rsun_idx] for i in range(3)])
            return B_Rsun, pos_Rsun

    def B2(self, store=True):
        b2 = None
        if hasattr(self, '__B2__'):
            b2 = self.__B2__
        if b2 == None:
            b2 = N.array(self[0,...]**2+self[1,...]**2+self[2,...]**2)
            if store:
                self.__B2__ = b2
        return b2

    def __clean_derived_quantities(self):
        """ Removes any derived quantities """
        derived_quantities = ['__B2__','__Rsun__','__Rsun_sph__']
        for quant in derived_quantities:
            if hasattr(self, quant):
                setattr(self, None)




