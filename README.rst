=========================================================================
 GalMag - A Python tool for computing realistic galactic magnetic fields
=========================================================================
|doi| |rtd| |arxiv|

`GalMag <https://github.com/luizfelippesr/galmag>`_ is a
`Python <http://www.python.org>`_  package for computing galactic magnetic
fields based on mean field dynamo theory. 

The code computes two separate components of the galactic magnetic
field: a disc and a halo component. Both are solutions to the mean field
dynamo equation for the choice of parameters and galaxy rotation curves
specified by the user.

For more details about the physics, we refer the reader to the code paper [1]_ .

GalMag is open-source software available under the GNU General Public License v3 (GPL-3).


Usage / Quick start / Tutorial
-------------------------------

We refer the reader to the `the tutorial <galmag_tutorial.ipynb>`_ 
jupyter notebook which contains both a quick tour through the functionality and a 
description of the features and usage of the package. 

For more detailed code documentation see http://galmag.readthedocs.io/

Installation
------------

Requirements
============

- `Python <http://python.org/>`_ (v2.7.x)
    - `Scipy <http://www.scipy.org/scipylib/index.html>`_
    - `Numpy <http://www.numpy.org) (version 1.7 or later>`_ 
    - `Sympy <http://www.sympy.org/en/index.html>`_ (optional)
- `D2O <https://gitlab.mpcdf.mpg.de/ift/D2O/tree/master>`_


Download
========

One can download the latest release from
https://github.com/luizfelippesr/galmag/releases

The current (development) version can be obtained by cloning the repository::

    git clone git@github.com:luizfelippesr/galmag.git
          
Installation
============

To install, one can simply run the command::

    sudo -P python setup.py install

or  without root privileges:: 

    python setup.py install --user
   
and the `galmag` python package will be available in one's system.


References
----------

.. [1] Shukurov et al., "A physical model of the galactic large-scale magnetic field",
    MNRAS (submitted), `arXiv:1809.03595 <https://arxiv.org/abs/1809.03595/>`_.

.. |doi| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.1135246.svg
   :target: https://doi.org/10.5281/zenodo.1135246
   :alt: Latest DOI
   
.. |rtd| image:: https://readthedocs.org/projects/galmag/badge/?version=latest
   :target: http://galmag.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status
   
.. |arxiv| image:: http://img.shields.io/badge/arXiv-1809.03595-B31B1B.svg
   :target: https://arxiv.org/abs/1809.03595
   :alt: Code Paper ArXiv ID 
