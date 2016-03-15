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
            r_sph = tools.generate_grid(n_grid, xlim=[1e-2, 1.0],
                                                ylim=[1e-1, N.pi],
                                                zlim=None)
            r = tools.spherical_to_cartesian_grid(r_sph)
        elif grid_geometry=='custom':
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

    def B2(self, store=True):
        b2 = None
        if hasattr(self, '__B2__'):
            b2 = self.__B2__
        if b2==None:
            b2 = N.array(self[0,...]**2+self[1,...]**2+self[2,...]**2)
            if store:
                self.__B2__ = b2
        return b2

    def __clean_derived_quantities(self):
        """ Removes any derived quantities """
        derived_quantities = ['__B2__',]
        for quant in derived_quantities:
            if hasattr(self, quant):
                setattr(self, None)




