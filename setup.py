# -*- coding: utf-8 -*-

from setuptools import setup

exec(open('gmf_tool/version.py').read())

setup(name="gmf_tool",
      version=__version__,
      author="Luiz Rodrigues <luiz.rodrigues@newcastle.ac.uk>, "
             "Theo Steininger <theos@mpa-garching.mpg.de>",
      author_email="luiz.rodrigues@newcastle.ac.uk",
      description="Galactic magnetic fields based on the dynamo equation",
      url="",
      packages=["gmf_tool", "gmf_tool.analysis", "gmf_tool.B_generators"],
      license="",
      zip_safe=False,
      dependency_links=[
        'git+https://gitlab.mpcdf.mpg.de/ift/d2o.git#egg=d2o'],
      install_requires=['d2o'],
      )
