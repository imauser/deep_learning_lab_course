#!/usr/bin/env python

import os
import numpy as np
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

extensions = [
    Extension('im2col_cython', ['nn/conv/im2col_cython.pyx'], include_dirs = [np.get_include()]),
    Extension('conv_classic', ['nn/conv/conv_classic.pyx'], include_dirs = [np.get_include()]),
    Extension('pool_classic', ['nn/conv/pool_classic.pyx'], include_dirs = [np.get_include()])
]

setup(
    name = 'nn',
    version = '0.1',
    author = 'Jost Tobias Springenberg',
    author_email = 'springj@cs.informatik.uni-freiburg.de',
    description = "Neural nets in NumPy only",
    license = 'MIT',
    url = '',
    install_requires = ['numpy', 'scipy', 'cython'],
    long_description = read('README.md'),
    classifiers = [
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
    ],
    ext_modules = cythonize(extensions)
)

