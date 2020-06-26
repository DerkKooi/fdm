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

# Import pyscf and numpy
from pyscf import gto
import numpy as np

# Import fdm modules
import fdm.cython
import fdm.numba
try:
    import fdm.cython_parallel
    parallel = True
except ImportError:
    parallel = False
from fdm import pyscf_horton

# Set the number of dispersals. We take all dispersals x^i y^j z^k up to i+j+k <= nbase
nbase = 22

# We then need the moments up to 2 * nbase -1 to calculate all the matrix elements
nmax = 2 * nbase - 1

# Hardcoded Helium atom
mol_pyscf = gto.M(atom='He 0 0 0', basis='def2-tzvpp')

# Reference values from the SI
si_vals = {'HF': 1.618906, 'MP2': 1.433029, 'CCSD': 1.427269}

# Run with methods also used in KooWecGor (2020)
methods = ['HF', 'MP2', 'CCSD']

for method in methods:
    if method == 'HF':
        # Run R(O)HF
        mf, mo_coeff, gb = pyscf_horton.RHF(mol_pyscf)

        # Compute multipoles for R(O)HF
        xyz, xyz2 = pyscf_horton.compute_multipole_RHF(sum(mol_pyscf.nelec), nmax, mo_coeff, gb, center=np.array([0.,0.,0.]))

        # Set the format of xyz2 to xc-only, so not including Hartree and negative
        twordm = 'xc-only'
    elif method == 'MP2':
        # Run MP2
        e_mp2, rdm1, rdm2 = pyscf_horton.MP2(mol_pyscf, mf)

        # Compute multipoles for MP2
        xyz, xyz2 = pyscf_horton.compute_multipole_rdm12(nmax, rdm1, rdm2, mo_coeff, gb, center=np.array([0.,0.,0.]))

        # Set the format of xyz2 to full, including Hartree
        twordm = 'full'
    elif method == 'CCSD':
        # Run CCSD
        e_ccsd, rdm1, rdm2 = pyscf_horton.CCSD(mol_pyscf, mf)

        # Compute multipoles for CCSD
        xyz, xyz2 = pyscf_horton.compute_multipole_rdm12(nmax, rdm1, rdm2, mo_coeff, gb, center=np.array([0.,0.,0.]))

        # Set the format of xyz2 to full, including Hartree        
        twordm = 'full'

    # numba
    # Compute d and tau elements in the eigenbasis of tau with numba, then compute the C6 and isotropic C6
    dlist, taulist = fdm.numba.compute_elements(sum(mol_pyscf.nelec), xyz, xyz2, nbase=nbase, spherical=True, twordm=twordm)
    C6 = fdm.numba.compute_C6(dlist, taulist, dlist, taulist)
    isotropic_C6_HF = fdm.numba.compute_isotropic_C6(dlist, taulist, dlist, taulist)
    print(f'C6 {method} with numba: {C6:.{6}f}, isotropic C6 {method}  with numba: {isotropic_C6_HF:.{6}f} (should match for an atom), value from SI: {si_vals[method]}')

    # Now run it with cython instead of numba
    dlist, taulist = fdm.cython.compute_elements(sum(mol_pyscf.nelec), xyz, xyz2, nbase=nbase, spherical=True, twordm=twordm)
    C6_HF = fdm.cython.compute_C6(dlist, taulist, dlist, taulist)
    isotropic_C6_HF = fdm.cython.compute_isotropic_C6(dlist, taulist, dlist, taulist)
    print(f'C6 {method}  with cython: {C6_HF:.{6}f}, isotropic C6 {method}  with cython: {isotropic_C6_HF:.{6}f} (should match for an atom), value from SI: {si_vals[method]}')

    # Now run it with parallel cython instead of numba
    if parallel:
        dlist, taulist = fdm.cython_parallel.compute_elements(sum(mol_pyscf.nelec), xyz, xyz2, nbase=nbase, spherical=True, twordm=twordm)
        C6_HF = fdm.cython_parallel.compute_C6(dlist, taulist, dlist, taulist)
        isotropic_C6_HF = fdm.cython_parallel.compute_isotropic_C6(dlist, taulist, dlist, taulist)
        print(f'C6 {method}  with cython parallel: {C6_HF:.{6}f}, isotropic C6 {method}  with cython parallel: {isotropic_C6_HF:.{6}f} (should match for an atom), value from SI: {si_vals[method]}')
