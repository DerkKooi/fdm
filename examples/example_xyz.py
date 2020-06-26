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

# Two xyz files to be loaded and charge and spin. Note that spin is 2S, not 2S+1, so you put a zero for a singlet
xyz_fileA = sys.argv[1]
xyz_fileB = sys.argv[2]
chargeA = int(sys.argv[3])
spinA = int(sys.argv[4])
chargeB = int(sys.argv[5])
spinB = int(sys.argv[6])

atomsA = open(xyz_fileA, 'r').readlines()[2:]
atoms_listA = "".join([i[:-1] + "; " for i in atomsA])

atomsB = open(xyz_fileB, 'r').readlines()[2:]
atoms_listB = "".join([i[:-1] + "; " for i in atomsB])

mol_pyscfA = gto.M(atom=atoms_listA, spin=spinA, basis='def2-TZVPP', charge=chargeA)
mol_pyscfB = gto.M(atom=atoms_listB, spin=spinB, basis='def2-TZVPP', charge=chargeB)

# Run with methods used in KooWecGor (2020)
methods = ['HF', 'MP2', 'CCSD']

for method in methods:
    if method == 'HF':
        #Run R(O)HF
        mfA, mo_coeffA, gbA = pyscf_horton.RHF(mol_pyscfA)
        mfB, mo_coeffB, gbB = pyscf_horton.RHF(mol_pyscfB)

        # Compute multipoles for R(O)HF
        if mol_pyscfA.spin == 0:
            xyzA, xyz2A = pyscf_horton.compute_multipole_RHF(sum(mol_pyscfA.nelec), nmax, mo_coeffA, gbA, center=np.array([0.,0.,0.]))
        else:
            xyzA, xyz2A = pyscf_horton.compute_multipole_ROHF(mol_pyscfA.nelec, nmax, mo_coeffA, gbA, center=np.array([0.,0.,0.]))

        if mol_pyscfB.spin == 0:
            xyzB, xyz2B = pyscf_horton.compute_multipole_RHF(sum(mol_pyscfB.nelec), nmax, mo_coeffB, gbB, center=np.array([0.,0.,0.]))
        else:
            xyzB, xyz2B = pyscf_horton.compute_multipole_ROHF(mol_pyscfB.nelec, nmax, mo_coeffB, gbB, center=np.array([0.,0.,0.]))


        # Set the format of xyz2 to xc-only, so not including Hartree and negative
        twordm = 'xc-only'
    elif method == 'MP2':
        # Run MP2
        e_mp2A, rdm1A, rdm2A = pyscf_horton.MP2(mol_pyscfA, mfA)
        e_mp2B, rdm1B, rdm2B = pyscf_horton.MP2(mol_pyscfB, mfB)

        # Compute multipoles for MP2
        xyzA, xyz2A = pyscf_horton.compute_multipole_rdm12(nmax, rdm1A, rdm2A, mo_coeffA, gbA, center=np.array([0.,0.,0.]))
        xyzB, xyz2B = pyscf_horton.compute_multipole_rdm12(nmax, rdm1B, rdm2B, mo_coeffB, gbB, center=np.array([0.,0.,0.]))

        # Set the format of xyz2 to full, including Hartree
        twordm='full'
    elif method == 'CCSD':
        # Run CCSD
        e_ccsdA, rdm1A, rdm2A = pyscf_horton.CCSD(mol_pyscfA, mfA)
        e_ccsdA, rdm1B, rdm2B = pyscf_horton.CCSD(mol_pyscfB, mfB)

        # Compute multipoles for CCSD
        xyzA, xyz2A = pyscf_horton.compute_multipole_rdm12(nmax, rdm1A, rdm2A, mo_coeffA, gbA, center=np.array([0.,0.,0.]))
        xyzB, xyz2B = pyscf_horton.compute_multipole_rdm12(nmax, rdm1B, rdm2B, mo_coeffB, gbB, center=np.array([0.,0.,0.]))

        # Set the format of xyz2 to full, including Hartree
        twordm='full'

    # numba
    # Compute d and tau elements in the eigenbasis of tau with numba, then compute the anisotropies
    dlistA, taulistA = fdm.numba.compute_elements(sum(mol_pyscfA.nelec), xyzA, xyz2A, nbase=nbase, spherical=False, twordm=twordm)
    dlistB, taulistB = fdm.numba.compute_elements(sum(mol_pyscfB.nelec), xyzB, xyz2B, nbase=nbase, spherical=False, twordm=twordm)
    C6 = fdm.numba.compute_C6(dlistA, taulistA, dlistB, taulistB)
    isoC6, Gamma6AB, Gamma6BA, Delta6 = fdm.numba.compute_anisotropic_C6(dlistA, taulistA, dlistB, taulistB)
    print(f'C6 aligned on z-axis {method} with cython: {C6:.{6}f}')   
    print(f'isotropic C6 {method} with numba: {isoC6:.{6}f}')
    print(f'Gamma6AB {method} with numba: {Gamma6AB:.{6}f}')
    print(f'Gamma6BA {method} with numba: {Gamma6BA:.{6}f}')
    print(f'Delta6 {method} with numba: {Delta6:.{6}f}')
    

    # Now run it with cython instead of numba
    dlistA, taulistA = fdm.cython.compute_elements(sum(mol_pyscfA.nelec), xyzA, xyz2A, nbase=nbase, spherical=False, twordm=twordm)
    dlistB, taulistB = fdm.cython.compute_elements(sum(mol_pyscfB.nelec), xyzB, xyz2B, nbase=nbase, spherical=False, twordm=twordm)
    C6 = fdm.cython.compute_C6(dlistA, taulistA, dlistB, taulistB)
    isoC6, Gamma6AB, Gamma6BA, Delta6 = fdm.cython.compute_anisotropic_C6(dlistA, taulistA, dlistB, taulistB)
    print(f'C6 aligned on z-axis {method} with cython: {C6:.{6}f}')
    print(f'isotropic C6 {method} with cython: {isoC6:.{6}f}')
    print(f'Gamma6AB {method} with cython: {Gamma6AB:.{6}f}')
    print(f'Gamma6BA {method} with cython: {Gamma6BA:.{6}f}')
    print(f'Delta6 {method} with cython: {Delta6:.{6}f}')

    # Now run it with parallel cython instead of numba
    if parallel:
        dlistA, taulistA = fdm.cython_parallel.compute_elements(sum(mol_pyscfA.nelec), xyzA, xyz2A, nbase=nbase, spherical=False, twordm=twordm)
        dlistB, taulistB = fdm.cython_parallel.compute_elements(sum(mol_pyscfB.nelec), xyzB, xyz2B, nbase=nbase, spherical=False, twordm=twordm)       
        C6 = fdm.cython_parallel.compute_isotropic_C6(dlistA, taulistA, dlistB, taulistB)
        isoC6, Gamma6AB, Gamma6BA, Delta6 = fdm.cython_parallel.compute_anisotropic_C6(dlistA, taulistA, dlistB, taulistB)
        print(f'C6 aligned on z-axis {method} with cython parallel: {C6:.{6}f}')
        print(f'isotropic C6 {method} with cython parallel: {isoC6:.{6}f}')
        print(f'Gamma6AB {method} with cython parallel: {Gamma6AB:.{6}f}')
        print(f'Gamma6BA {method} with cython parallel: {Gamma6BA:.{6}f}')
        print(f'Delta6 {method} with cython parallel: {Delta6:.{6}f}')
