import numpy as N
import core.disk as d 
import core.halo as h
import core.tools as tools

class field(N.ndarray):
    """ 
        The magnetic field class. 
        Stores the numpy array containing the magnetic field as well as the parameters and potentially useful intermediate quantities.
    """ 
    def __new__(cls, params, no_disk=False, no_halo=False,info=None):
        
        if 'n_grid' in params:
            n_grid = params['n_grid']
        else:
            n_grid = 50
        r = tools.generate_grid(n_grid)
        
        # Adapted from NumPy's manual
        # B array is an already formed ndarray instance
        # We first cast to be our class type
        obj = N.zeros_like(r).view(cls)
        # add the new attribute to the created instance
        obj.grid = r
        obj.info = info
        obj.parameters = params
        # Finally, we must return the newly created object:
        return obj
    
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
     