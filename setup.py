# -*- coding: utf-8 -*-
# GalMag
# Copyright (C) 2017  Luiz Felippe S. Rodrigues
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

from setuptools import setup
import os, os.path
exec(open('galmag/version.py').read())

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(name="galmag",
      version=__version__,
      author="Luiz Felippe S. Rodrigues <luiz.rodrigues@newcastle.ac.uk>, "
             "Theo Steininger <theos@mpa-garching.mpg.de>",
      author_email="luiz.rodrigues@newcastle.ac.uk",
      keywords = "magnetic field, galaxy, Galaxy",
      description= ("Generates realistic galactic magnetic field based on the "
                    "dynamo equation"),
      url="https://github.com/luizfelippesr/galmag",
      packages=["galmag", "galmag.analysis", "galmag.B_generators"],
      license="GPLv3",
      zip_safe=False,
      long_description=read('README.rst'),
      dependency_links=[
        'git+https://gitlab.mpcdf.mpg.de/ift/d2o.git#egg=d2o'],
      install_requires=requirements,
      )
