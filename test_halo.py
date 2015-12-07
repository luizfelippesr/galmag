import pylab as P
from core import halo
from core import halo_free_decay_modes as f

from core.rotation_curve import simple_V

pi = P.pi
cos = P.cos

# Sets the grid
No = 40 # Avoid even numbers

rho = P.linspace(1e-10,1.1,No)
thetha = P.linspace(-pi/2.,pi/2.,No)
phi = P.linspace(0.,pi,No)

dr = rho[1]-rho[0]
dt = thetha[1]-thetha[0]
dp = phi[1]-phi[0]
d3sp = dr*dt*dp

tt, rr, pp = P.meshgrid(thetha,rho,phi) 

r = P.empty((3,No, No, No))
alpha = P.empty_like(r)

r[0,:,:,:] = rr 
r[1,:,:,:] = tt
r[2,:,:,:] = pp

#Computes the rotation curve and alpha
V = simple_V(r,10,10)

alpha = cos(tt)
alpha[rr>1.] = 0.

# Parameters
p = {'Ralpha':3, 'Romega': 200}

# Computes coefficients
values, vect = halo.Galerkin_expansion_coefficients(r, alpha, V, p,
                                                    dV_s = d3sp,
                                                    symmetric=True,
                                                    dynamo_type='alpha-omega',
                                                    n_free_decay_modes=4)

growth = []
Ras = P.linspace(0.001,10,15)
for Ra in Ras:
    p = {'Ralpha':Ra, 'Romega': 200}
    
    values, vect = halo.Galerkin_expansion_coefficients(r, alpha, V, p,
                                                    symmetric=True,
                                                    dV_s = d3sp,
                                                    dynamo_type='alpha-omega',
                                                    n_free_decay_modes=4)
    growth.append( values.real.max())
    
P.plot(Ras, growth)
P.xlabel(r'$R_\alpha$')
P.ylabel(r'$max(Re(\gamma))$')
P.title('Figure 19')
P.show()  