# -*- coding: utf-8 -*-

from distutils.core import setup
import os

setup(name="gmf_tool",
      version="0.1",
      author="Luiz Rodrigues <luiz.rodrigues@newcastle.ac.uk>, " + \
             "Theo Steininger <theos@mpa-garching.mpg.de>",
      author_email="luiz.rodrigues@newcastle.ac.uk",
      description="Galactic magnetic fields based on the dynamo equation",
      url="",
      packages=["gmf_tool", "gmf_tool.analysis", "gmf_tool.B_generator",
                "gmf_tool.core"],
      license="")