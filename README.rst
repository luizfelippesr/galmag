=========================================================================
 GalMag - A Python tool for computing realistic galactic magnetic fields
=========================================================================
|test| |rtd| |doi| |ascl|

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

Download
========

One can download the latest release from
https://github.com/luizfelippesr/galmag/releases

The current (development) version can be obtained by cloning the repository::

    git clone git@github.com:luizfelippesr/galmag.git
          
Installation
============

To install, one can simply run the command::

    pip install .

if you are (both) not using conda and do not have root privileges, the following may help:: 

    python setup.py install --user
   
and the `galmag` python package will be available in one's system.


References
----------

.. [1] Shukurov et al., "A physical approach to modelling large-scale galactic magnetic fields",
    `A&A 623, A113 (2019) <https://doi.org/10.1051/0004-6361/201834642>`_.


.. |test| image:: https://github.com/luizfelippesr/galmag/workflows/Python%20package/badge.svg
   :target: https://github.com/luizfelippesr/galmag/actions?query=workflow%3A%22Python+package%22
   :alt: Autotest results

.. |doi| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.1135246.svg
   :target: https://doi.org/10.5281/zenodo.1135246
   :alt: DOI:10.5281/zenodo.1135246
   
.. |rtd| image:: https://readthedocs.org/projects/galmag/badge/?version=latest
   :target: http://galmag.readthedocs.io/en/latest/?badge=latest
   :alt: galmag.readthedocs.io

.. |ascl| image:: https://img.shields.io/badge/ascl-1903.005-blue.svg?colorB=262255
   :target: http://ascl.net/1903.005
   :alt: ascl:1903.005
  
