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

# Import dispersion modules
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

# Hardcoded H2 molecule
mol_pyscf = gto.M(atom='H 0 0 0.368495; H 0 0 -0.368495', basis='def2-tzvpp')

# Reference values from the SI
si_valsC6 = {'HF': 16.417556, 'MP2': 12.890010, 'CCSD': 11.600815}
si_valsGamma6AB = {'HF': 0.141587, 'MP2': 0.109860, 'CCSD': 0.102089}
si_valsGamma6BA = {'HF': 0.141587, 'MP2': 0.109860, 'CCSD': 0.102089}
si_valsDelta6 = {'HF':  0.021399, 'MP2': 0.012751, 'CCSD': 0.010987}

# Run with methods also used in KooWecGor (2020)
methods = ['HF', 'MP2', 'CCSD']

for method in methods:
    if method == 'HF':
        # Run R(O)HF
        mf, mo_coeff, gb =  pyscf_horton.RHF(mol_pyscf)

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

    # Compute d and tau elements in the eigenbasis of tau with numba, then compute the anisotropies at order 1/R^6
    dlist, taulist = fdm.numba.compute_elements(sum(mol_pyscf.nelec), xyz, xyz2, nbase=nbase, twordm=twordm)
    C6, Gamma6AB, Gamma6BA, Delta6 = fdm.numba.compute_anisotropic_C6(dlist, taulist, dlist, taulist)
    print(f'isotropic C6 {method} with numba: {C6:.{6}f}, value from SI: {si_valsC6[method]}')
    print(f'Gamma6AB {method} with numba: {Gamma6AB:.{6}f}, value from SI: {si_valsGamma6AB[method]}')
    print(f'Gamma6BA {method} with numba: {Gamma6BA:.{6}f},  value from SI: {si_valsGamma6BA[method]}')
    print(f'Delta6 {method} with numba: {Delta6:.{6}f}, value from SI: {si_valsDelta6[method]}')
    
    # Now run it with cython instead of numba
    dlist, taulist = fdm.cython.compute_elements(sum(mol_pyscf.nelec), xyz, xyz2, nbase=nbase, twordm=twordm)
    C6, Gamma6AB, Gamma6BA, Delta6  = fdm.cython.compute_anisotropic_C6(dlist, taulist, dlist, taulist)
    print(f'isotropic C6 {method} with cython: {C6:.{6}f}, value from SI: {si_valsC6[method]}')
    print(f'Gamma6AB {method} with cython: {Gamma6AB:.{6}f}, value from SI: {si_valsGamma6AB[method]}')
    print(f'Gamma6BA {method} with cython: {Gamma6BA:.{6}f},  value from SI: {si_valsGamma6BA[method]}')
    print(f'Delta6 {method} with cython: {Delta6:.{6}f}, value from SI: {si_valsDelta6[method]}')
    
    # Now run it with parallel cython instead of numba
    if parallel:
        dlist, taulist = fdm.cython_parallel.compute_elements(sum(mol_pyscf.nelec), xyz, xyz2, nbase=nbase, twordm=twordm)
        C6, Gamma6AB, Gamma6BA, Delta6 = fdm.cython_parallel.compute_anisotropic_C6(dlist, taulist, dlist, taulist)
        print(f'isotropic C6 {method} with cython parallel: {C6:.{6}f}, value from SI: {si_valsC6[method]}')
        print(f'Gamma6AB {method} with cython parallel: {Gamma6AB:.{6}f}, value from SI: {si_valsGamma6AB[method]}')
        print(f'Gamma6BA {method} with cython parallel: {Gamma6BA:.{6}f},  value from SI: {si_valsGamma6BA[method]}')
        print(f'Delta6 {method} with cython parallel: {Delta6:.{6}f}, value from SI: {si_valsDelta6[method]}')
