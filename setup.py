#!/usr/bin/env python

import os
import setuptools

here = os.path.abspath(os.path.dirname(__file__))

setuptools.setup(
    name='lagrangian_analyses',
    description='Lagrangian Analyses codes for python using OpenDrift as particle tracker ',
    author='Mireya Montano',
    url='https://github.com/MireyaMMO/Lagrangian_Analyses',
    download_url='https://github.com/MireyaMMO/Lagrangian_Analyses',
    version='1.0.0',
    license='',
    packages=['Lagrangian_Analyses'],
    long_description=open('README.md').read(),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'netCDF4',
        'pyproj',
        'cartopy',
        'calendar',
        'pickle',
        'datetime',
        'xarray',
    ],
    include_package_data=True
)