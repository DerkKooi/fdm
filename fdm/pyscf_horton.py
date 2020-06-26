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


import numpy as np

from gbasis.wrappers import from_pyscf
from gbasis.integrals import overlap, moment

from pyscf import scf, ao2mo, fci, cc, mp


def write_string(filename, string):
    file = open(filename, 'w+')
    file.write(string)
    file.close()
    return


def RHF(mol_pyscf):
    '''Run a R(O)HF calculation on mol_pyscf, does a few sanity checks between
    horton and pyscf.

    Arguments:
    mol_pyscf: The molecule object from pyscf.

    Returns:
    mf: pyscf mean-field object.
    mo_coeff_pyscf: pyscf MO coefficients.
    gobasis: HORTON3 basiset object.
    '''

    # Calculate the overlaps and dipoles using pyscf
    overlap_pyscf = mol_pyscf.intor('int1e_ovlp')
    dipole = mol_pyscf.intor('int1e_r')

    # Run a RHF calculation on the molecule/atom
    mf = scf.RHF(mol_pyscf).run()  # Run RHF

    if mol_pyscf.spin == 0:
        # Make HF 1-RDM in AO basis
        rdm1_HF = mf.make_rdm1()

        # Number of basis states
        nbasis = rdm1_HF.shape[0]

        # Number of occupied spatial orbitals is equal to the number of alpha electrons is equal to the number of beta electrons
        nocc = mol_pyscf.nelec[0]

        # Get the MO coefficients C_{ai} from pyscf
        mo_coeff_pyscf = mf.mo_coeff

        # Make HF 1-RDM in MO basis
        rdm1_MO = np.zeros((nbasis, nbasis))
        for i in range(nocc):
            rdm1_MO[i, i] = 2.

        # Check if the MO and AO basis RDM's are equal
        assert np.allclose(mo_coeff_pyscf.dot(rdm1_MO.dot(mo_coeff_pyscf.T)), rdm1_HF) 

    else:
        # Make HF 1-RDM in AO basis
        rdm1_HF = mf.make_rdm1()

        # Number of basis states
        nbasis = rdm1_HF[0].shape[0]

        # Get the MO coefficients C_{ai} from pyscf
        mo_coeff_pyscf = mf.mo_coeff

        # Make HF 1-RDM in MO basis
        rdm1_MO = []
        rdm1_MO.append(np.zeros((nbasis, nbasis)))
        rdm1_MO.append(np.zeros((nbasis, nbasis)))
        for i in range(mol_pyscf.nelec[0]):
            rdm1_MO[0][i, i] = 1
        for i in range(mol_pyscf.nelec[1]):
            rdm1_MO[1][i, i] = 1

        # Check if the MO and AO basis RDM's are equal
        assert np.allclose(mo_coeff_pyscf.dot(rdm1_MO[0].dot(mo_coeff_pyscf.T)), rdm1_HF[0])
        assert np.allclose(mo_coeff_pyscf.dot(rdm1_MO[1].dot(mo_coeff_pyscf.T)), rdm1_HF[1])

    # Transform the pyscf overlap and dipole integrals to the MO basis
    overlap_MO = mo_coeff_pyscf.T.dot(overlap_pyscf.dot(mo_coeff_pyscf))
    dipole_MO = []
    for i in range(3):
        dipole_MO.append(mo_coeff_pyscf.T.dot(dipole[i].dot(mo_coeff_pyscf)))

    # Construct the molecule in Horton
    gobasis = from_pyscf(mol_pyscf)

    # Compute the overlap and dipole integrals from Horton, transform to MO basis
    overlap_gb = overlap.overlap_integral(gobasis)
    center = np.array([0., 0., 0.])
    overlap_MO_gb = mo_coeff_pyscf.T.dot(overlap_gb).dot(mo_coeff_pyscf)
    dipole_array = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    dipole_gb = []
    dipole_MO_gb = []
    for i in range(3):
        dipole_gb.append(moment.moment_integral(gobasis, center, np.array([dipole_array[i]]))[:, :, 0])
        dipole_MO_gb.append(mo_coeff_pyscf.T.dot(dipole_gb[i].dot(mo_coeff_pyscf)))

    # Check that the overlap and dipole integrals are the same in the MO basis coming from both pyscf and Horton
    assert np.allclose(overlap_MO, overlap_MO_gb)
    assert np.allclose(dipole_MO, dipole_MO_gb)

    return mf, mo_coeff_pyscf, gobasis


def UHF(mol_pyscf):
    '''Run a UHF calculation on mol_pyscf, does a few sanity checks between
    horton and pyscf.

    Arguments:
    mol_pyscf: The molecule object from pyscf.

    Returns:
    mf: pyscf mean-field object.
    mo_coeff_pyscf: pyscf MO coefficients.
    gobasis: HORTON3 basiset object.'''

    # Calculate the overlaps and dipoles using pyscf
    overlap_pyscf = mol_pyscf.intor('int1e_ovlp')
    dipole = mol_pyscf.intor('int1e_r')

    # Run a RHF calculation on the molecule/atom
    mf = scf.UHF(mol_pyscf).run()  #Run UHF

    # Make HF 1-RDM in AO basis
    rdm1_HF = mf.make_rdm1()

    # Number of basis states
    nbasis = rdm1_HF[0].shape[0]

    # Get the MO coefficients C_{ai} from pyscf
    mo_coeff_pyscf = mf.mo_coeff

    # Make HF 1-RDM in MO basis
    rdm1_MO = []
    rdm1_MO.append(np.zeros((nbasis, nbasis)))
    rdm1_MO.append(np.zeros((nbasis, nbasis)))
    for i in range(mol_pyscf.nelec[0]):
        rdm1_MO[0][i, i] = 1
    for i in range(mol_pyscf.nelec[1]):
        rdm1_MO[1][i, i] = 1

    # Check if the MO and AO basis RDM's are equal
    assert np.allclose(mo_coeff_pyscf[0].dot(rdm1_MO[0].dot(mo_coeff_pyscf[0].T)),rdm1_HF[0]) 
    assert np.allclose(mo_coeff_pyscf[1].dot(rdm1_MO[1].dot(mo_coeff_pyscf[1].T)),rdm1_HF[1]) 

    # Transform the pyscf overlap and dipole integrals to the MO basis
    overlap_MO = [mo_coeff_pyscf[0].T.dot(overlap_pyscf.dot(mo_coeff_pyscf[0])),mo_coeff_pyscf[1].T.dot(overlap_pyscf.dot(mo_coeff_pyscf[1]))]
    dipole_MO = []
    for sigma in range(2):
        dipole_MO.append([])
        for i in range(3):
            dipole_MO[sigma].append(mo_coeff_pyscf[sigma].T.dot(dipole[i].dot(mo_coeff_pyscf[sigma]))) 

    # Construct the molecule in Horton
    gobasis = from_pyscf(mol_pyscf)

    # Compute the overlap and dipole integrals from Horton, transform to MO basis
    overlap_MO_gb = []
    overlap_gb = overlap.overlap_integral(gobasis)
    center = np.array([0.,0.,0.])
    dipole_list = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    dipole_gb = []
    dipole_MO_gb = []

    for sigma in range(2):
        overlap_MO_gb.append(mo_coeff_pyscf[sigma].T.dot(overlap_gb).dot(mo_coeff_pyscf[sigma]))
        dipole_MO_gb.append([])
        for i in range(3):
            dipole_gb.append(moment.moment_integral(gobasis, center, np.array([dipole_list[i]]))[:,:,0])
            dipole_MO_gb[sigma].append(mo_coeff_pyscf[sigma].T.dot(dipole_gb[i].dot(mo_coeff_pyscf[sigma])))

    # Check that the overlap and dipole integrals are the same in the MO basis coming from both pyscf and Horton
    assert np.allclose(overlap_MO, overlap_MO_gb)
    assert np.allclose(dipole_MO, dipole_MO_gb)

    return mf, mo_coeff_pyscf, gobasis

def fullCI(mol_pyscf, mf):
    ''' Run a fullCI calculation using pyscf on mol_pyscf

    Arguments:
    mol_pyscf: The molecule object from pyscf.
    mf: pyscf mean-field object (e.g. RHF or UHF)

    Returns:
    e: fullCI energy,
    rdm1: one-body reduced densitymatrix.
    rdm2: two-body reduced densitymatrix.'''

    # Extract the MO coefficients in the PySCF AO ordering
    mo_coeff_pyscf = mf.mo_coeff

    # Get the number of AO's
    nbasis = mo_coeff_pyscf.shape[0]

    # Construct one-body Hamiltonian elements in the MO basis
    h1 = mo_coeff_pyscf.T.dot(mf.get_hcore()).dot(mo_coeff_pyscf)  #One-body matrix elements

    # Construct the 2-body Hamiltonian elements int he MO basis, Chemist's notation (pq|rs)
    eri = ao2mo.kernel(mol_pyscf, mf.mo_coeff)  #Get the electron repulsion integrals.

    # Construct the FullCI solver, selecting explicitly the spin=0 (singlet solver) or spin=1 (general spin symmetry solver)
    if mol_pyscf.spin == 0:
        cisolver = fci.direct_spin0.FCI(mol_pyscf)
    else:
        cisolver = fci.direct_spin1.FCI(mol_pyscf)

    # Do the actual solution, return only the lowest energy solution and the corresponding ci vector
    e, fcivec = cisolver.kernel(h1, eri, nbasis, mol_pyscf.nelec, ecore=mol_pyscf.energy_nuc())
    rdm1, rdm2 = cisolver.make_rdm12(fcivec, nbasis, mol_pyscf.nelec, reorder=True)
    return e, rdm1, rdm2


def CCSD(mol_pyscf, mf):
    ''' Run a CCSD calculation using pyscf on mol_pyscf

    Arguments:
    mol_pyscf: The molecule object from pyscf.
    mf: pyscf mean-field object (e.g. RHF or UHF)

    Returns:
    e: CCSD energy,
    rdm1: one-body reduced densitymatrix.
    rdm2: two-body reduced densitymatrix.'''

    # Run CCSD (one-line!)
    ccsd = cc.CCSD(mf).run()

    # Make reduced density matrices
    rdm1s = ccsd.make_rdm1()
    rdm2s = ccsd.make_rdm2()

    # If spin=0, then you get directly the spatial density matrices, otherwise sum the spin density matrices to get the spatial ones
    if mol_pyscf.spin == 0:
        rdm1 = rdm1s
        rdm2 = rdm2s
    else:  
        rdm1 = rdm1s[0]+rdm1s[1]
        rdm2 = rdm2s[0]+rdm2s[1]+rdm2s[1].transpose(2, 3, 0, 1)+rdm2s[2]

    return ccsd.e_tot, rdm1, rdm2


def MP2(mol_pyscf, mf):
    ''' Run a MP2 calculation using pyscf on mol_pyscf

    Arguments:
    mol_pyscf: The molecule object from pyscf.
    mf: pyscf mean-field object (e.g. RHF or UHF)

    Returns:
    e: MP2 energy,
    rdm1: one-body reduced densitymatrix.
    rdm2: two-body reduced densitymatrix.'''

    # Run MP2
    mp2 = mp.MP2(mf)
    e = mp2.kernel()[0]

    # Make reduced density matrices
    rdm1s = mp2.make_rdm1()
    rdm2s = mp2.make_rdm2()

    # If spin=0, then you get directly the spatial density matrices, otherwise sum the spin density matrices to get the spatial ones
    if mol_pyscf.spin == 0:
        rdm1 = rdm1s
        rdm2 = rdm2s
    else:
        rdm1 = rdm1s[0]+rdm1s[1]
        rdm2 = rdm2s[0]+rdm2s[1]+rdm2s[1].transpose(2, 3, 0, 1)+rdm2s[2]

    return e, rdm1, rdm2


def compute_multipole_rdm12(nmax, rdm1, rdm2, mo_coeff, gobasis, center=np.array([0.,0.,0.])):
    '''Given a 1- and 2-rdm calculate the expectation values of the multipole moments up to power nmax given the mo_coefficients corresponding to horton, the 
    that is calculate <x^i y^j z^k> up to i + j + k = nmax and <x_1^i y_1^j z_1^k x_2^l y_2^m z_2^n> up to i+j+k=nmax/2 (even nmax) or i+j+k=(nmax+1)/2 (odd nmax) and same for l+m+n

    Arguments:
    nmax: Maximum multipole powers,
    rdm1: One-body reduced density matrix,
    rdm2: Two-body reduced density matrix,
    mo_coeff: Molecular Orbital coefficients from MF calculation,
    gobasis: HORTON3 basisset object,
    center: The multipole center to expand around. Default: [0.,0.,0.]

    Returns:
    xyz: expectation values of multipoles.
    xyz2: expectation values of interaction multipoles.
    '''

    # Extract the number of basis states from the 1-RDM
    nbasis = rdm1.shape[0]

    # See above, we don't need to calculate xyz2 up to nmax, but only halfnmax
    halfnmax = np.ceil(nmax/2).astype(np.int)

    # Build list of moments to compute.
    momentlist = np.zeros((nmax**3, 3), dtype=np.int64)
    n = 0
    for i in range(nmax):
        for j in range(nmax):
            for k in range(nmax):
                momentlist[n, :] = [i, j, k]
                n += 1

    #Compute multipoles with immediate AO to MO transform
    xyzmatMO = moment.moment_integral(gobasis, center, momentlist, transform=mo_coeff.T).transpose(2, 0, 1).reshape(nmax, nmax, nmax, nbasis, nbasis)

    #Calculate expectation value from 1-RDM, do this in loop since we are looping anyway...
    xyz = np.einsum('ij,abcij->abc', rdm1, xyzmatMO, optimize=True)

    #One-liner for xyz2
    xyz2 = np.einsum('ijkl,pqrij,stvkl->pqrstv', rdm2, xyzmatMO[0:halfnmax, 0:halfnmax, 0:halfnmax, :, :], xyzmatMO[0:halfnmax, 0:halfnmax, 0:halfnmax, :, :], optimize=True)
    
    return xyz, xyz2


def compute_multipole_RHF(Nel, nmax, mo_coeff, gobasis, center=np.array([0., 0., 0.])):
    '''Given the number of electrons and the MO_coefficients for a RHF calculation 
    calculate the expectation values of the multipole moments up to power nmax given 
    the mo_coefficients corresponding to horton, that is calculate <x^i y^j z^k> 
    up to i + j + k = nmax and <x_1^i y_1^j z_1^k x_2^l y_2^m z_2^n> up to 
    i+j+k=nmax/2 (even nmax) or i+j+k=(nmax+1)/2 (odd nmax) and same for l+m+n
    
    Arguments:
    Nel: number of electrons,
    nmax: Maximum multipole powers,
    mo_coeff: Molecular Orbital coefficients from MF calculation,
    gobasis: HORTON3 basisset object,
    center: The multipole center to expand around. Default: [0.,0.,0.].
    
    Returns:
    xyz: expectation values of multipoles.
    xyz2: expectation values of interaction multipoles. 
    CAVE: It's the expectation value with *minus* the exchange-hole.
    '''
    # Build overlap matrix Smat
    Smat = overlap.overlap_integral(gobasis)

    # Extract the number of basis states from the overlap matrix
    nbasis = Smat.shape[0]

    # Number of occupied orbitals is half the number of electrons
    nocc = int(Nel//2)

    # See above, we don't need to calculate xyz2 up to nmax, but only halfnmax
    halfnmax = np.ceil(nmax/2).astype(np.int)

    # Build list of moments to compute
    momentlist = np.zeros((nmax**3, 3 ), dtype=np.int64)
    n = 0
    for i in range(nmax):
        for j in range(nmax):
            for k in range(nmax):
                momentlist[n, :] = [i, j, k]
                n += 1

    # Compute multipoles with immediate AO to MO transform
    xyzmatMO = moment.moment_integral(gobasis, center, momentlist, transform=mo_coeff.T).transpose(2, 0, 1).reshape(nmax, nmax, nmax, nbasis, nbasis)

    xyz = 2*np.einsum('abcpp->abc', xyzmatMO[:, :, :, :nocc, :nocc], optimize=True)

    #One-liner for xyz2, coefficient is 1/2 (to get rid of opposite-spin exchange) * 2 (2 electrons per orbital) * 2 (2 electrons per orbital)
    xyz2 = 2*np.einsum('ijkab,lmnab->ijklmn',xyzmatMO[0:halfnmax, 0:halfnmax, 0:halfnmax, :nocc, :nocc],xyzmatMO[0:halfnmax, 0:halfnmax, 0:halfnmax, :nocc, :nocc], optimize=True)

    #print("\n Computed xyz and xyz2 ")
    return xyz, xyz2


def compute_multipole_ROHF(nelec, nmax, mo_coeff, gobasis, center=np.array([0.,0.,0.])):
    '''Given the number of electrons and the MO_coefficients for a ROHF calculation
    calculate the expectation values of the multipole moments up to power nmax given 
    the mo_coefficients corresponding to horton, that is calculate <x^i y^j z^k>
    up to i + j + k = nmax and <x_1^i y_1^j z_1^k x_2^l y_2^m z_2^n> up to
    i+j+k=nmax/2 (even nmax) or i+j+k=(nmax+1)/2 (odd nmax) and same for l+m+n

    Arguments:
    Nel: number of electrons,
    nmax: Maximum multipole powers,
    mo_coeff: Molecular Orbital coefficients from MF calculation,
    gobasis: HORTON3 basisset object,
    center: The multipole center to expand around. Default: [0.,0.,0.].

    Returns:
    xyz: expectation values of multipoles.
    xyz2: expectation values of interaction multipoles. NOTE: It's the
    expectation value with *minus* the exchange-hole.
    '''
    if type(gobasis) == 'tuple':
        gobasis = gobasis[0]

    # Build overlap matrix Smat
    Smat = overlap.overlap_integral(gobasis)

    # Extract number of basis states from overlap matrix
    nbasis = Smat.shape[0]

    # Extract separately number of alpha and beta electrons
    neleca, nelecb = nelec

    # See above, we don't need to calculate xyz2 up to nmax, but only halfnmax
    halfnmax = np.ceil(nmax/2).astype(np.int)

    # Build list of moments to compute
    momentlist = np.zeros((nmax**3, 3 ), dtype=np.int64)
    n = 0
    for i in range(nmax):
        for j in range(nmax):
            for k in range(nmax):
                momentlist[n,:] = [i, j, k]
                n += 1

    # Compute multipoles with immediate AO to MO transform
    xyzmatMO = moment.moment_integral(gobasis, center, momentlist, transform=mo_coeff.T).transpose(2, 0, 1).reshape(nmax, nmax, nmax, nbasis, nbasis)
    xyz = np.einsum('abcpp->abc',xyzmatMO[:, :, :, :neleca, :neleca], optimize=True)+np.einsum('abcpp->abc', xyzmatMO[:, :, :, :nelecb, :nelecb],optimize=True)

    # Two-liner for xyz2
    xyz2 = np.einsum('ijkab,lmnab->ijklmn',xyzmatMO[0:halfnmax, 0:halfnmax, 0:halfnmax, :neleca, :neleca], xyzmatMO[0:halfnmax, 0:halfnmax, 0:halfnmax, :neleca, :neleca],optimize=True)
    xyz2 += np.einsum('ijkab,lmnab->ijklmn',xyzmatMO[0:halfnmax, 0:halfnmax, 0:halfnmax, :nelecb, :nelecb],xyzmatMO[0:halfnmax, 0:halfnmax, 0:halfnmax, :nelecb, :nelecb],optimize=True)

    # print("\n Computed xyz and xyz2 ")
    return xyz, xyz2
