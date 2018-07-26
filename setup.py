#!/usr/bin/env python

from os.path import exists
from setuptools import setup

DISTNAME = 'tsg_toolbox'
PACKAGES = ['tsg']
TESTS = [p + '.tests' for p in PACKAGES]
INSTALL_REQUIRES = ['numpy >= 1.11', 'matplotlib>=1.5']
TESTS_REQUIRE = ['pytest >= 2.7.1']

URL = 'http://github.com/serazing/tsg_toolbox'
AUTHOR = 'Guillaume Serazin'
AUTHOR_EMAIL = 'guillaume.serazin@legos.obs-mip.fr'
LICENSE = 'MIT'
DESCRIPTION = 'Collection of functions to study ocean surface tracer from thermosalinograph data'
LONG_DESCRIPTION = (open('README.rst').read() if exists('README.rst') else '')
VERSION = 0.1
setup(name=DISTNAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      url=URL,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      license=LICENSE,
      scripts=['bin/compute_tracer_gradients.py'],
      package_data = {'tsg': ['data/*.nc']},
      keywords=['TSG', 'oceanography'],
      packages=PACKAGES + TESTS,
      install_requires=INSTALL_REQUIRES,
      zip_safe=False)
