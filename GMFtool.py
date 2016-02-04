import numpy as N
import core.disk as d 
import core.halo as h
import core.tools as tools

class field(N.ndarray):
    """ The magnetic field class.
        Stores the numpy array containing the magnetic field as well as the
        parameters and potentially useful intermediate quantities.
    """ 
    def __new__(cls, params, no_disk=False, no_halo=False,info=None):
        
        # Generates a cartesian grid
        n_grid = tools.get_param(params, 'ngrid', default=300)
        R_h = tools.get_param(params, 'R_h', default=20)
        r = tools.generate_grid(n_grid, xlim=[-R_h, R_h],
                                        ylim=[-R_h, R_h],
                                        zlim=[-R_h, R_h])

        # B array is an already formed ndarray instance
        # We first cast to be our class type
        B = N.zeros_like(r).view(cls)
        # add the new attribute to the created instance
        B.grid = r
        B.info = info
        B.parameters = params
        # Finally, we must return the newly created object:
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
        
    def compute_halo_field(self, params=None):
        """ Produces the magnetic field of a halo. 
            If a parameters dictionary is supplied, updates the parameters
            atribute.
        """
        if params is not None:
            self.parameters.update(params)
        
        self.halo = h.get_B_halo(self.grid, self.parameters)
        self += self.halo
     
