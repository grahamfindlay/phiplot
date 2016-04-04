#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# setup.py

try:
  from setuptools import setup
except ImportError:
  from distutils.core import setup

setup(
    name='phiplot',
    version='0.0.1',
    description='Python tools for plotting IIT figures',
    author='Graham Findlay',
    url='http://github.com/grahamfindlay/phiplot',
    license='GNU General Public License v3.0',
    packages=['phiplot'],
    classifiers=[
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Developers',
      'Natural Language :: English',
      'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
      'Programming Language :: Python',
      'Programming Language :: Python :: 3',
      'Topic :: Scientific/Engineering',
    ]
)
