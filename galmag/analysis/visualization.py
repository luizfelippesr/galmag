import matplotlib.pyplot as plt
import numpy as np
""" Contains functions to facilitate simple ploting tasks """


def std_setup():
    """ Adjusts matplotlib default settings"""
    from cycler import cycler
    plt.rc('image', cmap='viridis')
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)
    plt.rc('axes', labelsize=15, titlesize=15)
    plt.rcParams['axes.prop_cycle'] = cycler('color',['#1f78b4','#a6cee3','#33a02c','#b2df8a',
                                      '#e31a1c','#fb9a99','#ff7f00','#fdbf6f',
                                      '#6a3d9a','#cab2d6'])
    plt.rcParams['lines.linewidth'] = 1.65

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

    Parameters
    ----------
    B : B_field
        a B_field or B_field_component object
    quiver : bool
        If True, shows quivers. Default True
    contour : bool
        If True, shows contours. Default: True
    skipx/skipz : int
        Tweaks the display of quivers (Default: skipz=5, skipr=3)
    """
    # Requires a cylindrical grid
    assert B.grid.grid_type == 'cylindrical'

    # Makes a color contour plot
    if contour:
        CP = plt.contourf(B.grid.r_cylindrical[:,0,:], B.grid.z[:,0,:],
                      -B.phi[:,0,:], alpha=0.75, vmin=vmin, vmax=vmax, cmap=cmap)
        CB = plt.colorbar(CP, label=r'$B_\phi\,[\mu{{\rm G}}]$',)
        plt.setp(CP.collections , linewidth=2)

    if quiver:
        plt.quiver(B.grid.r_cylindrical[::skipr,0,::skipz], B.grid.z[::skipr,0,::skipz],
                 B.r_cylindrical[::skipr,0,::skipz],B.z[::skipr,0,::skipz],
                 color=quiver_color, alpha=0.75, **kwargs)

    plt.ylim([B.grid.z[:,0,:].min(),
          B.grid.z[:,0,:].max()])
    plt.xlim([B.grid.r_cylindrical[:,0,:].min(),
          B.grid.r_cylindrical[:,0,:].max()])

    plt.xlabel(r'$R\,[{{\rm kpc}}]$')
    plt.ylabel(r'$z\,[{{\rm kpc}}]$')


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

    Parameters
    ----------
    B : B_field
        a B_field or B_field_component object
    quiver : bool
        If True, shows quivers. Default True
    contour : bool
        If True, shows contours. Default: True
    skipx/skipz : int
        Tweaks the display of quivers (Default: skipz=5, skipx=1)
    """
    # Requires a Cartesian grid
    assert B.grid.grid_type == 'cartesian'

    # Makes a color contour plot
    if contour:
        CP = plt.contourf(B.grid.x[:,iy,:], B.grid.z[:,iy,:], B.phi[:,iy,:],
                        alpha=0.75, cmap=cmap, vmin=vmin, vmax=vmax)
        if not no_colorbar:
            CB = plt.colorbar(CP, label=r'$B_\phi\,[\mu{{\rm G}}]$',)
            plt.setp(CP.collections , linewidth=2)

    if quiver:
        plt.quiver(B.grid.x[::skipx,iy,::skipz], B.grid.z[::skipx,iy,::skipz],
                 B.x[::skipx,iy,::skipz],B.z[::skipx,iy,::skipz],
                 color=quiver_color, alpha=0.75,**kwargs)

    plt.ylim([B.grid.z[:,iy,:].min(),
          B.grid.z[:,iy,:].max()])
    plt.xlim([B.grid.x[:,iy,:].min(),
          B.grid.x[:,iy,:].max()])

    plt.xlabel(r'$x\,[{{\rm kpc}}]$')
    plt.ylabel(r'$z\,[{{\rm kpc}}]$')


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

    Parameters
    ----------
    B : B_field
        a B_field or B_field_component object
    quiver : bool
        If True, shows quivers. Default True
    contour : bool
        If True, shows contours. Default: True
    skipy/skipz : int
        Tweaks the display of quivers (Default: skipz=5, skipy=5)
    """
    # Requires a Cartesian grid
    assert B.grid.grid_type == 'cartesian'

    # Makes a color contour plot
    CP = plt.contourf(B.grid.y[ix,:,:], B.grid.z[ix,:,:], B.phi[ix,:,:],
                    alpha=0.75, cmap=cmap, vmin=vmin, vmax=vmax)
    CB = plt.colorbar(CP, label=r'$B_\phi\,[\mu{{\rm G}}]$',)
    plt.setp(CP.collections , linewidth=2)

    if quiver:
        plt.quiver(B.grid.y[ix,::skipy,::skipz], B.grid.z[ix,::skipy,::skipz],
                 B.y[ix,::skipy,::skipz],B.z[ix,::skipy,::skipz],
                 color=quiver_color, alpha=0.75,**kwargs)

    plt.ylim([B.grid.z[ix,:,:].min(),
          B.grid.z[ix,:,:].max()])
    plt.xlim([B.grid.y[ix,:,:].min(),
          B.grid.y[ix,:,:].max()])

    plt.xlabel(r'$y\,[{{\rm kpc}}]$')
    plt.ylabel(r'$z\,[{{\rm kpc}}]$')


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

    Parameters
    ----------
    B : B_field
        a B_field or B_field_component object
    field_lines : bool
        If True, shows field lines. Default: True
    quiver : bool
        If True, shows quivers. Default True
    contour : bool
        If True, shows contours. Default: True
    skipx/skipy : int
        Tweaks the display of quivers (Default: skipx=5, skipy=5)
    """
    # Requires a Cartesian grid
    assert B.grid.grid_type == 'cartesian'

    if contour:
        CP = plt.contourf(B.grid.x[:,:,iz], B.grid.y[:,:,iz],
                        np.sqrt(B.x[:,:,iz]**2+B.y[:,:,iz]**2+B.z[:,:,iz]**2),
                        alpha=0.75, cmap=cmap)
        CB = plt.colorbar(CP, label=r'$B\,[\mu{{\rm G}}]$',)
        plt.setp(CP.collections , linewidth=2)

    if field_lines:
        plt.streamplot(np.array(B.grid.x[:,0,iz]), np.array(B.grid.y[0,:,iz]),
                    -np.array(B.y[:,:,iz]), -np.array(B.x[:,:,iz]),color='r')
    if quiver:
        plt.quiver(B.grid.x[::skipx,::skipy,iz], B.grid.y[::skipx,::skipy,iz],
              B.x[::skipx,::skipy,iz],B.y[::skipx,::skipy,iz],
              color=quiver_color,**kwargs)

    plt.ylim([B.grid.y[:,:,iz].min(),B.grid.y[:,:,iz].max()])
    plt.xlim([B.grid.x[:,:,iz].min(),B.grid.x[:,:,iz].max()])
    plt.xlabel(r'$x\,[{{\rm kpc}}]$')
    plt.ylabel(r'$y\,[{{\rm kpc}}]$')

    return

def plot_slice():
    """
    Plots slice of arbitrary orientation

    Note
    ----
    Not implemented yet
    """
    raise NotImplemented
