import pylab as P

def std_setup():
    from cycler import cycler
    P.rc('image', cmap='viridis')
    P.rc('xtick', labelsize=14)
    P.rc('ytick', labelsize=14)
    P.rc('axes', labelsize=15, titlesize=15)
    P.rcParams['axes.prop_cycle'] = cycler('color',['#1f78b4','#a6cee3','#33a02c','#b2df8a',
                                      '#e31a1c','#fb9a99','#ff7f00','#fdbf6f',
                                      '#6a3d9a','#cab2d6'])
    P.rcParams['lines.linewidth'] = 1.65

def plot_r_z_uniform(B,skipr=3,skipz=5, quiver=True, contour=True,
                     quiver_color = '0.25', cmap='viridis',
                     vmin=None, vmax=None, **kwargs):
    """
    Plots a r-z slice of the field. Assumes B is created using a cylindrical
    grid - for a more sophisticated/flexible plotting script which does not
    rely on the grid structure check the plot_slice.

    The plot consists of:
      1) a coloured contourplot of B_\phi
      2) quivers showing the x-z projection of the field

    Input:
      B -> a B_field or B_field_component object
      quiver, contour -> booleans controlling which type of plot
                                      should be displayed. (Default: all True.)
      skipx, skipz -> arguments to tweak the the displaying of the
                      quivers. (Default: skipz=5, skipx=5)
    """
    # Requires a cylindrical grid
    assert B.grid.grid_type == 'cylindrical'

    # Makes a color contour plot
    if contour:
        CP = P.contourf(B.grid.r_cylindrical[:,0,:], B.grid.z[:,0,:],
                      -B.phi[:,0,:], alpha=0.5, vmin=vmin, vmax=vmax, cmap=cmap)
        CB = P.colorbar(CP, label=r'$B_\phi\,[\mu{{\rm G}}]$',)
        P.setp(CP.collections , linewidth=2)

    if quiver:
        P.quiver(B.grid.r_cylindrical[::skipr,0,::skipz], B.grid.z[::skipr,0,::skipz],
                 B.r_cylindrical[::skipr,0,::skipz],B.z[::skipr,0,::skipz],
                 color=quiver_color, alpha=0.75, **kwargs)

    P.ylim([B.grid.z[:,0,:].min(),
          B.grid.z[:,0,:].max()])
    P.xlim([B.grid.r_cylindrical[:,0,:].min(),
          B.grid.r_cylindrical[:,0,:].max()])

    P.xlabel(r'$R\,[{{\rm kpc}}]$')
    P.ylabel(r'$z\,[{{\rm kpc}}]$')


def plot_x_z_uniform(B,skipx=1,skipz=5,iy=0, quiver=True, contour=True,
                     quiver_color='0.25', cmap='viridis',
                     vmin=None, vmax=None, no_colorbar=False, **kwargs):
    """
    Plots a x-z slice of the field. Assumes B is created using a cartesian
    grid - for a more sophisticated/flexible plotting script which does not
    rely on the grid structure check the plot_slice.

    The plot consists of:
      1) a coloured contourplot of B_\phi
      2) quivers showing the x-z projection of the field

    Input:
      B -> a B_field or B_field_component object
      quiver, contour -> booleans controlling which type of plot
                         should be displayed. (Default: all True.)
      skipx, skipz -> arguments to tweak the the displaying of the
                      quivers. (Default: skipz=5, skipx=5)
    """
    # Requires a Cartesian grid
    assert B.grid.grid_type == 'cartesian'

    # Makes a color contour plot
    if contour:
        CP = P.contourf(B.grid.x[:,iy,:], B.grid.z[:,iy,:], B.phi[:,iy,:],
                        alpha=0.5, cmap=cmap, vmin=vmin, vmax=vmax)
        if not no_colorbar:
            CB = P.colorbar(CP, label=r'$B_\phi\,[\mu{{\rm G}}]$',)
            P.setp(CP.collections , linewidth=2)

    if quiver:
        P.quiver(B.grid.x[::skipx,iy,::skipz], B.grid.z[::skipx,iy,::skipz],
                 B.x[::skipx,iy,::skipz],B.z[::skipx,iy,::skipz],
                 color=quiver_color, alpha=0.75,**kwargs)

    P.ylim([B.grid.z[:,iy,:].min(),
          B.grid.z[:,iy,:].max()])
    P.xlim([B.grid.x[:,iy,:].min(),
          B.grid.x[:,iy,:].max()])

    P.xlabel(r'$x\,[{{\rm kpc}}]$')
    P.ylabel(r'$z\,[{{\rm kpc}}]$')


def plot_y_z_uniform(B, skipy=5, skipz=5, ix=0, quiver=True, contour=True,
                     quiver_color='0.25', cmap='viridis',
                     vmin=None, vmax=None, **kwargs):
    """
    Plots a y-z slice of the field. Assumes B is created using a cartesian
    grid - for a more sophisticated/flexible plotting script which does not
    rely on the grid structure check the plot_slice.

    The plot consists of:
      1) a coloured contourplot of B_\phi
      2) Quivers showing the y-z projection of the field

    Input:
      B -> a B_field or B_field_component object
      quiver, contour -> booleans controlling which type of plot
                         should be displayed. (Default: all True.)
      skipy, skipz -> arguments to tweak the the displaying of the
                      quivers. (Default: skipz=5, skipy=5)
    """
    # Requires a Cartesian grid
    assert B.grid.grid_type == 'cartesian'

    # Makes a color contour plot
    CP = P.contourf(B.grid.y[ix,:,:], B.grid.z[ix,:,:], B.phi[ix,:,:], 
                    alpha=0.5, cmap=cmap, vmin=vmin, vmax=vmax)
    CB = P.colorbar(CP, label=r'$B_\phi\,[\mu{{\rm G}}]$',)
    P.setp(CP.collections , linewidth=2)

    if quiver:
        P.quiver(B.grid.y[ix,::skipy,::skipz], B.grid.z[ix,::skipy,::skipz],
                 B.y[ix,::skipy,::skipz],B.z[ix,::skipy,::skipz],
                 color=quiver_color, alpha=0.75,**kwargs)

    P.ylim([B.grid.z[ix,:,:].min(),
          B.grid.z[ix,:,:].max()])
    P.xlim([B.grid.y[ix,:,:].min(),
          B.grid.y[ix,:,:].max()])

    P.xlabel(r'$y\,[{{\rm kpc}}]$')
    P.ylabel(r'$z\,[{{\rm kpc}}]$')


def plot_x_y_uniform(B, skipx=5, skipy=5, iz=0, field_lines=True, quiver=True,
                     contour=True,quiver_color='0.25',cmap='viridis',**kwargs):
    """
    Plots a x-y slice of the field. Assumes B is created using a cartesian
    grid - for a more sophisticated/flexible plotting script which does not
    rely on the grid structure check the plot_slice.

    The plot consists of:
      1) a coloured contourplot of |B|^2
      2) Field lines of the B_x and B_y field
      3) Quivers showing the B_x and B_y field

    Input:
      B -> a B_field or B_field_component object
      field_lines, quiver, contour -> booleans controlling which type of plot
                                      should be displayed. (Default: all True.)
      skipx, skipy -> arguments to tweak the the displaying of the
                      quivers. (Default: skipx=5, skipy=5)
    """
    # Requires a Cartesian grid
    assert B.grid.grid_type == 'cartesian'

    if contour:
        CP = P.contourf(B.grid.x[:,:,iz], B.grid.y[:,:,iz],
                        P.sqrt(B.x[:,:,iz]**2+B.y[:,:,iz]**2+B.z[:,:,iz]**2),
                        alpha=0.5, cmap=cmap)
        CB = P.colorbar(CP, label=r'$B\,[\mu{{\rm G}}]$',)
        P.setp(CP.collections , linewidth=2)

    if field_lines:
        P.streamplot(P.array(B.grid.x[:,0,iz]), P.array(B.grid.y[0,:,iz]),
                    -P.array(B.y[:,:,iz]), -P.array(B.x[:,:,iz]),color='r')
    if quiver:
        P.quiver(B.grid.x[::skipx,::skipy,iz], B.grid.y[::skipx,::skipy,iz],
              B.x[::skipx,::skipy,iz],B.y[::skipx,::skipy,iz],
              color=quiver_color,**kwargs)

    P.ylim([B.grid.y[:,:,iz].min(),B.grid.y[:,:,iz].max()])
    P.xlim([B.grid.x[:,:,iz].min(),B.grid.x[:,:,iz].max()])
    P.xlabel(r'$x\,[{{\rm kpc}}]$')
    P.ylabel(r'$y\,[{{\rm kpc}}]$')

    return

def plot_slice():
    raise NotImplemented
