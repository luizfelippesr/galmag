# -*- coding: utf-8 -*-
# GalMag - A Python tool for computing realistic galactic magnetic fields
# Copyright (C) 2017, 2018  Luiz Felippe S. Rodrigues <luiz.rodrigues@ncl.ac.uk>
#
# GalMag is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# GalMag is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GalMag.  If not, see <http://www.gnu.org/licenses/>.
"""
-----------------------------------------------------------------------
GalMag - A Python tool for computing realistic galactic magnetic fields
-----------------------------------------------------------------------

Generates realistic galactic magnetic field based on the dynamo equation

Example:
   Initializes a B_field object on a uniform cartesian 100x100x100 grid::

      from galmag import B_field

      box_limits = [[-15, 15],[-15, 15],[-15, 15]] # kpc
      box_resolution = [100,100,100]
      B = B_field(box_limits, box_resolution)


   Adds a disc magnetic field with reversals at 4.7 kpc and 12.25 kpc
   and a field strength of -3 microgauss close to the Sun::

      B.add_disk_field(reversals=[4.7,12.25] ,
                       B_phi_solar_radius=-3, # muG
                       number_of_modes=number_of_modes)


   Adds a halo magnetic field with intensity 0.1 microgauss close to the Sun::

      B.add_halo_field(halo_ref_Bphi=Bphi_sun)

   Each coordinate component of the produced magnetic can then be accessed
   through the relevant coordinate attributes, e.g.::

      B.r_spherical
      B.theta

For a quick introduction to usage, please check the
`tutorial <http://nbviewer.jupyter.org/url/www.mas.ncl.ac.uk/~nlfsr/galmag/galmag_tutorial.ipynb>`_
jupyter notebook.
"""

# Imports version number
from version import __version__

# Imports the relevant classes
from B_generators import *
from Grid import Grid
from B_field import *
#from Observables import Observables
