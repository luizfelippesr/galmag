# -*- coding: utf-8 -*-

from setuptools import setup
import os, os.path
exec(open('galmag/version.py').read())

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(name="galmag",
      version=__version__,
      author="Luiz Felippe S. Rodrigues <luiz.rodrigues@newcastle.ac.uk>, "
             "Theo Steininger <theos@mpa-garching.mpg.de>",
      author_email="luiz.rodrigues@newcastle.ac.uk",
      keywords = "magnetic field, galaxy, Galaxy",
      description= ("Generates realistic galactic magnetic field based on the "
                    "dynamo equation"),
      url="https://bitbucket.org/luizfelippe/galmag",
      packages=["galmag", "galmag.analysis", "galmag.B_generators"],
      license="GPLv3",
      zip_safe=False,
      long_description=read('README.md'),
      dependency_links=[
        'git+https://gitlab.mpcdf.mpg.de/ift/d2o.git#egg=d2o'],
      install_requires=['d2o'],
      )
