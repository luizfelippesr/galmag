---------------------------------------------------------------------------
 GalMag - A Python tool for computing realistic galactic magnetic fields
---------------------------------------------------------------------------

[GalMag](http://www.mas.ncl.ac.uk/~nlfsr/galmag) is a
[Python](http://www.python.org)  package for computing galactic magnetic
fields based on mean field dynamo theory.

The code computes two separate components of the galactic magnetic
field: a disc and a halo component. Both are solutions to the mean field
dynamo equation for the choice of parameters and galaxy rotation curves
specified by the user.

GalMag is open-source software available under the GNU General Public License v3 (GPL-3).

## Instalation ##

To install, one needs first to clone this repository,

```git clone git@github.com:luizfelippesr/galmag.git```

and then run the setup script

```sudo python setup.py install```

or, without root privileges,

```python setup.py install --user```


## Usage / Quick start / Tutorial ##

We refer the reader to the the ipython/jupyter notebook containing 
the tutorial in 
[this link](http://nbviewer.jupyter.org/url/www.mas.ncl.ac.uk/%7Enlfsr/galmag/galmag_tutorial.ipynb).

The tutorial contains both a quick tour through the functionality and a 
description of the features and usage of the package. The same notebook 
is also available in the file 
[galmag_tutorial.ipnb](galmag_tutorial.ipynb).



## Pre-requisites ##

- [Scipy](http://www.scipy.org/scipylib/index.html)
- [Numpy](http://www.numpy.org) (version 1.7 or later)
- [D2O](https://gitlab.mpcdf.mpg.de/ift/D2O/tree/master)
- [Sympy](http://www.sympy.org/en/index.html) (optional)


## Authors ##

- [Luiz Felippe S. Rodrigues](http://www.mas.ncl.ac.uk/~nlfsr/)
- Theo Steiniger
- and others!
