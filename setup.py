#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#
# This file is part of srs2d.
#
# srs2d is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# trooper-simulator is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with srs2d. If not, see <http://www.gnu.org/licenses/>.

"""
Setup script for srs2d.

Read INSTALL for instructions.
"""

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='srs2d',
    version='0.0.3',
    description='Swarm Robotics Simulator 2D',
    license='GPLv3',
    author='Eduardo L. Buratti',
    author_email='eburatti09@gmail.com',
    url='http://github.com/eburatti09/srs2d',
    download_url='http://github.com/eburatti09/srs2d',
    packages=['srs2d'],
    package_data = {
        'srs2d': ['kernels/*.cl']
    },
    scripts=[],
    test_suite='tests',
    install_requires=[],
    entry_points = {
        'console_scripts': [
            'srs2d-pso = srs2d.pso:main',
            'srs2d-ga = srs2d.ga:main',
        ],
    }
)
