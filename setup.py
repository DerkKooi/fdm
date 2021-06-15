# Fixed Diagonal Matrices, a python package to calculate dispersion C6 coefficients.
# Copyright (C) 2020  Derk Pieter Kooi

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.


import sys

import setuptools
from distutils.util import get_platform
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "fdm.cython",
        ["fdm/cython.pyx"],
    )
]

print(f'You are using platform: {get_platform()}')

if "parallel" in sys.argv:
    print('Compiling for parallel usage')
    ext_modules.append(Extension("fdm.cython_parallel",
        ["fdm/cython_parallel.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp']))
    sys.argv.remove('parallel')
    


long_description = '''A package to compute dispersion via the formalism of KooGor-JPCL-19, used in KooWecGor-arXiv-20
'''

setup(
    name='fdm',
    version='1.0.0',
    author='Derk Pieter Kooi',
    author_email='derkkooi@gmail.com',
    ext_modules=cythonize(ext_modules,compiler_directives={'language_level': sys.version_info[0]}),
    description='A package to compute dispersion via the formalism of KooGor-JPCL-19, used in KooWecGor-arXiv-20',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://gitlab.com/DerkKooi/dispersion",
    packages=setuptools.find_packages(),
    extras_require={
        "pyscf": ["pyscf>=1.6.1"],
        "numba": ["numba"],
    },
    license="BSD 3-Clause",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6.7',
    install_requires=[
          'numpy', 
          'scipy',
          'cython',
      ]
)
