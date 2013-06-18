#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Setup script for srs2d.

Read INSTALL for instructions.
"""

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(name='srs2d',
      version='0.1',
      description='Swarm Robotics Simulator 2D',
      license='GPLv3',
      author='Eduardo L. Buratti',
      author_email='eburatti09@gmail.com',
      url='http://github.com/eburatti09/srs2d',
      download_url='http://github.com/eburatti09/srs2d',
      packages=['srs2d'],
      scripts=[],
      test_suite='tests',
      install_requires=[]
     )
