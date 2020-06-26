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


print('General import')
import fdm
print('Import cython')
import fdm.cython
import importlib.util
loader = importlib.util.find_spec('fdm.cython_parallel')
if loader is not None:
    print('Import cython_parallel')
    import fdm.cython_parallel
print('Import numba')
import fdm.numba
print('Import pyscf_horton')
import fdm.pyscf_horton
