=========================================================================
 GalMag - A Python tool for computing realistic galactic magnetic fields
=========================================================================

`GalMag <https://github.com/luizfelippesr/galmag>`_ is a
`Python <http://www.python.org>`_  package for computing galactic magnetic
fields based on mean field dynamo theory.

The code computes two separate components of the galactic magnetic
field: a disc and a halo component. Both are solutions to the mean field
dynamo equation for the choice of parameters and galaxy rotation curves
specified by the user.

GalMag is open-source software available under the GNU General Public License v3 (GPL-3).


Instalation
------------

To install, one needs first to clone this repository,

.. code:: console

    git clone git@github.com:luizfelippesr/galmag.git

and then run the setup script

.. code:: console

    sudo python setup.py install

or, without root privileges,

.. code:: console

    python setup.py install --user


Usage / Quick start / Tutorial
-------------------------------

We refer the reader to the the ipython/jupyter notebook containing 
`the tutorial <galmag_tutorial.ipynb>`_ 
which contains both a quick tour through the functionality and a 
description of the features and usage of the package. 



Pre-requisites
-------------------------------


- `Scipy <http://www.scipy.org/scipylib/index.html>`_
- `Numpy <http://www.numpy.org) (version 1.7 or later>`_
- `D2O <https://gitlab.mpcdf.mpg.de/ift/D2O/tree/master>`_
- `Sympy <http://www.sympy.org/en/index.html>`_ (optional)


Authors
-------------------------------


- `Luiz Felippe S. Rodrigues <http://www.mas.ncl.ac.uk/~nlfsr>`_
- Theo Steiniger
- and others!
