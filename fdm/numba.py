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
import os
import numpy as np
import scipy
import numba

@numba.njit
def build_b(nbase):
    """Builds the list of exponents from the highest power allowed

    Arguments:
    nbase: Highest power allowed in the dispersals x^i y^j z^k, where i+j+k < nbase

    Returns:
    bbasis: list of dispersals"""

    # Pre-allocate the array, number of basis function is as below
    nb = nbase*(nbase+1)*(nbase+2)//6-1
    bbasis = np.zeros((nb, 3),dtype=numba.int64)
    i = 0
    for s in range(nbase):
        for t in range(nbase-s):
            for u in range(nbase-s-t):
                if s+t+u != 0:
                    bbasis[i, :] = [s, t, u]
                    i+=1
    return bbasis


@numba.njit
def build_b_spherical(nbase):
    """Builds the list of exponents from the highest power allowed.
    Version for spherical systems, where i+j+k even is ignored, 
    because they do not contribute due to symmetry.
    
    Arguments:
    nbase: Highest power allowed in the dispersals x^i y^j z^k, where i+j+k < nbase
    
    Returns:
    bbasis: list of dispersals"""
    nb = (2*nbase+5)*(nbase+2)*nbase//24
    bbasis = np.zeros((nb, 3), dtype=numba.int64)
    i = 0
    for s in range(nbase):
        for t in range(nbase-s):
            for u in range(nbase-s-t):
                if (s+t+u) % 2 == 1:
                    bbasis[i, :] = [s, t, u]
                    i += 1
    return bbasis


@numba.njit(parallel=True)
def build_elements_full(Nel, xyz, xyz2, bbasis):
    """Builds the dlist, taulist and Slist for bbasis
    from the expectation values of the moments for the 1-RDM, 2-RDM
    assumes xyz2 is calculated with the full 2-RDM

    Arguments:
    Nel: number of electrons.
    xyz: expectation values of moments.
    xyz2: expectation values of moments with 2-rdm.
    bbasis: list of dispsersals

    Returns:
    dlist: List of expectation values of the dipole moment with the dispersals.
    taulist: Matrix of kinetic matrix elements of the dispersals.
    Slist: Matrix of overlaps of the dispersals.
    """
    Nelinv = Nel**-1
    nb = bbasis.shape[0]

    # Get the dipole moments
    d0list = np.array([xyz[1, 0, 0], xyz[0, 1, 0], xyz[0, 0, 1]])


    # Allocate plist and norms
    norm = np.empty(nb)
    plist = np.empty(nb)

    # Calculate plist and norms
    for i in numba.prange(nb):
        plist[i] = xyz[bbasis[i, 0],bbasis[i, 1],bbasis[i, 2]]*Nelinv
        norm[i] = np.power(xyz[bbasis[i,0]*2,bbasis[i,1]*2,bbasis[i,2]*2]-Nel*plist[i]*plist[i]+xyz2[bbasis[i,0],bbasis[i,1],bbasis[i,2],bbasis[i,0],bbasis[i,1],bbasis[i,2]]-Nel*(Nel-1)*plist[i]*plist[i],-1/2)

    # Allocate dlist, taulist and Slist
    dlist = np.empty((nb, 3))
    taulist = np.empty((nb,nb))
    Slist = np.empty((nb,nb))

    # Calculate dlist, taulist and Slist
    for i in numba.prange(nb):
        dlist[i, 0] = (xyz[bbasis[i,0]+1,bbasis[i,1],bbasis[i,2]]+xyz2[1,0,0,bbasis[i,0],bbasis[i,1],bbasis[i,2]]-Nel*d0list[0]*plist[i])*norm[i]
        dlist[i, 1] = (xyz[bbasis[i,0],bbasis[i,1]+1,bbasis[i,2]]+xyz2[0,1,0,bbasis[i,0],bbasis[i,1],bbasis[i,2]]-Nel*d0list[1]*plist[i])*norm[i]
        dlist[i, 2] = (xyz[bbasis[i,0],bbasis[i,1],bbasis[i,2]+1]+xyz2[0,0,1,bbasis[i,0],bbasis[i,1],bbasis[i,2]]-Nel*d0list[2]*plist[i])*norm[i]
        for j in numba.prange(nb):
            Slist[i, j] = (xyz[bbasis[i,0]+bbasis[j,0],bbasis[i,1]+bbasis[j,1],bbasis[i,2]+bbasis[j,2]]+xyz2[bbasis[i,0],bbasis[i,1],bbasis[i,2],bbasis[j,0],bbasis[j,1],bbasis[j,2]]-Nel**2*plist[i]*plist[j])*norm[i]*norm[j]
            taulist[i, j] = norm[i]*norm[j]*(bbasis[i,0]*bbasis[j,0]*xyz[bbasis[i,0]+bbasis[j,0]-2,bbasis[i,1]+bbasis[j,1],bbasis[i,2]+bbasis[j,2]]+bbasis[i,1]*bbasis[j,1]*xyz[bbasis[i,0]+bbasis[j,0],bbasis[i,1]+bbasis[j,1]-2,bbasis[i,2]+bbasis[j,2]]+bbasis[i,2]*bbasis[j,2]*xyz[bbasis[i,0]+bbasis[j,0],bbasis[i,1]+bbasis[j,1],bbasis[i,2]+bbasis[j,2]-2])
    
    return dlist, taulist, Slist


@numba.njit(parallel=True)
def build_elements_xconly(Nel, xyz, xyz2, bbasis):
    """Builds the dlist, taulist and Slist for bbasis 
    from the expectation values of the moments for the 1-RDM, 2-RDM
    assumes xyz2 is calculated with MINUS the xc part.
    
    Arguments:
    Nel: number of electrons.
    xyz: expectation values of moments.
    xyz2: expectation values of moments with 2-rdm.
    bbasis: list of dispsersals

    Returns:
    dlist: List of expectation values of the dipole moment with the dispersals.
    taulist: Matrix of kinetic matrix elements of the dispersals.
    Slist: Matrix of overlaps of the dispersals.
    """
    nb = bbasis.shape[0]

    # Allocate plist and norms
    norm = np.empty(nb)

    # Calculate plist and norms
    for i in numba.prange(nb):
        norm[i] = np.power(xyz[bbasis[i,0]*2,bbasis[i,1]*2,bbasis[i,2]*2]-xyz2[bbasis[i,0],bbasis[i,1],bbasis[i,2],bbasis[i,0],bbasis[i,1],bbasis[i,2]],-1/2)

    # Allocate dlist, taulist and Slist
    dlist = np.empty((nb, 3))
    taulist = np.empty((nb,nb))
    Slist = np.empty((nb,nb))

    # calculate dlist, taulist and Slist
    for i in numba.prange(nb):
        dlist[i, 0] = (xyz[bbasis[i,0]+1,bbasis[i,1],bbasis[i,2]]-xyz2[1,0,0,bbasis[i,0],bbasis[i,1],bbasis[i,2]])*norm[i]
        dlist[i, 1] = (xyz[bbasis[i,0],bbasis[i,1]+1,bbasis[i,2]]-xyz2[0,1,0,bbasis[i,0],bbasis[i,1],bbasis[i,2]])*norm[i]
        dlist[i, 2] = (xyz[bbasis[i,0],bbasis[i,1],bbasis[i,2]+1]-xyz2[0,0,1,bbasis[i,0],bbasis[i,1],bbasis[i,2]])*norm[i]
        for j in numba.prange(nb):
            Slist[i, j] = (xyz[bbasis[i,0]+bbasis[j,0],bbasis[i,1]+bbasis[j,1],bbasis[i,2]+bbasis[j,2]]-xyz2[bbasis[i,0],bbasis[i,1],bbasis[i,2],bbasis[j,0],bbasis[j,1],bbasis[j,2]])*norm[i]*norm[j]
            taulist[i, j] = norm[i]*norm[j]*(bbasis[i,0]*bbasis[j,0]*xyz[bbasis[i,0]+bbasis[j,0]-2,bbasis[i,1]+bbasis[j,1],bbasis[i,2]+bbasis[j,2]]+bbasis[i,1]*bbasis[j,1]*xyz[bbasis[i,0]+bbasis[j,0],bbasis[i,1]+bbasis[j,1]-2,bbasis[i,2]+bbasis[j,2]]+bbasis[i,2]*bbasis[j,2]*xyz[bbasis[i,0]+bbasis[j,0],bbasis[i,1]+bbasis[j,1],bbasis[i,2]+bbasis[j,2]-2])
    return dlist, taulist, Slist


@numba.njit(parallel=True)
def build_elements_no2rdm(Nel, xyz, bbasis):
    """Builds the dlist, taulist and Slist for bbasis 
    from the expectation values of the moments for the 1-RDM
    assumes the pair density is the 'renormalized Hartree pair density'
    or self-interaction corrected Hartree P = rho * rho /N*(N-1)
    
    Arguments:
    Nel: number of electrons.
    xyz: expectation values of moments.
    bbasis: list of dispsersals

    Returns:
    dlist: List of expectation values of the dipole moment with the dispersals.
    taulist: Matrix of kinetic matrix elements of the dispersals.
    Slist: Matrix of overlaps of the dispersals."""

    Nelinv = Nel**-1
    nb = bbasis.shape[0]

    # Get the dipole moments
    d0list = np.array([xyz[1,0,0],xyz[0,1,0],xyz[0,0,1]])


    # Allocate plist and norms
    norm = np.empty(nb)
    plist = np.empty(nb)
    for i in numba.prange(nb):
        plist[i] = xyz[bbasis[i, 0],bbasis[i, 1],bbasis[i, 2]]*Nelinv
        norm[i] = np.power(xyz[bbasis[i,0]*2,bbasis[i,1]*2,bbasis[i,2]*2]-Nel*plist[i]*plist[i],-1/2)

    # Allocate dlist, taulist and Slist
    dlist = np.empty((nb, 3))
    taulist = np.empty((nb,nb))
    Slist = np.empty((nb,nb))

    # Calculate dlist, taulist and Slist
    for i in numba.prange(nb):
        dlist[i, 0] = (xyz[bbasis[i,0]+1,bbasis[i,1],bbasis[i,2]]-d0list[0]*plist[i])*norm[i]
        dlist[i, 1] = (xyz[bbasis[i,0],bbasis[i,1]+1,bbasis[i,2]]-d0list[1]*plist[i])*norm[i]
        dlist[i, 2] = (xyz[bbasis[i,0],bbasis[i,1],bbasis[i,2]+1]-d0list[2]*plist[i])*norm[i]
        for j in numba.prange(nb):
            Slist[i, j] = (xyz[bbasis[i,0]+bbasis[j,0],bbasis[i,1]+bbasis[j,1],bbasis[i,2]+bbasis[j,2]]-Nel*plist[i]*plist[j])*norm[i]*norm[j]
            taulist[i, j] = norm[i]*norm[j]*(bbasis[i,0]*bbasis[j,0]*xyz[bbasis[i,0]+bbasis[j,0]-2,bbasis[i,1]+bbasis[j,1],bbasis[i,2]+bbasis[j,2]]+bbasis[i,1]*bbasis[j,1]*xyz[bbasis[i,0]+bbasis[j,0],bbasis[i,1]+bbasis[j,1]-2,bbasis[i,2]+bbasis[j,2]]+bbasis[i,2]*bbasis[j,2]*xyz[bbasis[i,0]+bbasis[j,0],bbasis[i,1]+bbasis[j,1],bbasis[i,2]+bbasis[j,2]-2])
    return dlist, taulist, Slist


@numba.njit(parallel=True)
def build_elements_hartree(Nel, xyz, bbasis):
    """Builds the dlist, taulist and Slist for bbasis 
    from the expectation values of the moments for the 1-RDM
    assumes the pair density is Hartree pair density, 
    which exactly cancels plist parts

    Arguments:
    Nel: number of electrons.
    xyz: expectation values of moments.
    bbasis: list of dispsersals

    Returns:
    dlist: List of expectation values of the dipole moment with the dispersals.
    taulist: Matrix of kinetic matrix elements of the dispersals.
    Slist: Matrix of overlaps of the dispersals.
    """
    nb = bbasis.shape[0]


    #Allocate plist and norms
    norm = np.empty(nb)
    for i in numba.prange(nb):
        norm[i] = np.power(xyz[bbasis[i,0]*2,bbasis[i,1]*2,bbasis[i,2]*2],-1/2)

    # Allocate dlist, taulist and Slist
    dlist = np.empty((nb, 3))
    taulist = np.empty((nb,nb))
    Slist = np.empty((nb,nb))

    # calculate dlist, taulist and Slist
    for i in numba.prange(nb):
        dlist[i, 0] = (xyz[bbasis[i,0]+1,bbasis[i,1],bbasis[i,2]])*norm[i]
        dlist[i, 1] = (xyz[bbasis[i,0],bbasis[i,1]+1,bbasis[i,2]])*norm[i]
        dlist[i, 2] = (xyz[bbasis[i,0],bbasis[i,1],bbasis[i,2]+1])*norm[i]
        for j in numba.prange(nb):
            Slist[i, j] = (xyz[bbasis[i,0]+bbasis[j,0],bbasis[i,1]+bbasis[j,1],bbasis[i,2]+bbasis[j,2]])*norm[i]*norm[j]

            taulist[i, j] = norm[i]*norm[j]*(bbasis[i,0]*bbasis[j,0]*xyz[bbasis[i,0]+bbasis[j,0]-2,bbasis[i,1]+bbasis[j,1],bbasis[i,2]+bbasis[j,2]]+bbasis[i,1]*bbasis[j,1]*xyz[bbasis[i,0]+bbasis[j,0],bbasis[i,1]+bbasis[j,1]-2,bbasis[i,2]+bbasis[j,2]]+bbasis[i,2]*bbasis[j,2]*xyz[bbasis[i,0]+bbasis[j,0],bbasis[i,1]+bbasis[j,1],bbasis[i,2]+bbasis[j,2]-2])
    return dlist, taulist, Slist

def compute_elements(Nel, xyz, xyz2=None, nbase=3, spherical=False, twordm='full'):
    """ Compute the elements of the (non-orthogonal but normalized) atomic dispersals u
    p to maximum x^i y^j z^k with i+j+k=nbase-1 using xyz and xyz2, which are the expectation 
    values of the multipoles with the 1-RDM and 2-RDM, then diagonalise tau and return 
    dlist and taulist in the basis in which tau is diagonal.
    Arguments:
    Nel: number of electrons.
    xyz: expectation values of moments.
    xyz2: expectation values of moments with 2-RDM. Default: none, for Hartree or renormalized Hartree
    nbase: include all dispersals x^i y^j z^k up to i + j + k < nbase. Default: 3, 

    Returns:
    dlist: List of expectation values of the dipole moment with the dispersals diagonalising tau.
    taulist: List of diagonal kinetic matrix elements of the dispersals diagonalising tau."""
    
    #Construct multipole function basis
    bbasis = build_b_spherical(nbase) if spherical else build_b(nbase)
    Nel = float(Nel)
    #
    if twordm == 'full':
        dlist, taulist, Slist = build_elements_full(Nel, xyz, xyz2, bbasis)
    elif twordm == 'xc-only':
        dlist, taulist, Slist = build_elements_xconly(Nel, xyz, xyz2, bbasis)
    elif twordm == 'no':
        dlist, taulist, Slist = build_elements_no2rdm(Nel, xyz, bbasis)
    elif twordm == 'hartree':
        dlist, taulist, Slist = build_elements_hartree(Nel, xyz, bbasis)

    #Diagonalize tau, then we only need to transform dlist to the same basis and need to only keep dlist and the eigenvalues of tau
    taulist, tauvecs = scipy.linalg.eigh(taulist, Slist)
    dlist = tauvecs.T.dot(dlist)

    return dlist,  taulist

@numba.njit(parallel=True)
def compute_C6(dlistA, taulistA, dlistB, taulistB):
    '''Computes  C6 at orientation fixed from dlist and taulist of the two systems
    
    Arguments:
    dlistA: List of dipole expectation values of the dispersals for system A.
    taulistA: List of diagonal kinetic matrix elements of the dispersals diagonalising tau for system A.
    dlistB: List of dipole expectation values of the dispersals for system B.
    taulistA: List of diagonal kinetic matrix elements of the dispersals diagonalising tau for system B.
    
    Returns:
    C6: C6 at fixed orientation'''
   
    # Obtain number of b-states
    nbA = dlistA.shape[0]
    nbB = dlistB.shape[0]

    # Build w as matrix
    wvec = np.zeros((nbA, nbB))
    cvec = np.zeros((nbA, nbB))
    C6 = 0.
    for i in numba.prange(nbA):
        for j in numba.prange(nbB):
            C6 +=  2*(dlistA[i, 0]*dlistB[j,0]+dlistA[i, 1]*dlistB[j,1]-2*dlistA[i, 2]*dlistB[j,2])**2\
            / (taulistA[i]+taulistB[j])
    return C6

@numba.njit(parallel=True)
def compute_isotropic_C6(dlistA, taulistA, dlistB, taulistB):
    '''Computes isotropic C6 from dlist and taulist of the two systems
    
    Arguments:
    dlistA: List of dipole expectation values of the dispersals for system A.
    taulistA: List of diagonal kinetic matrix elements of the dispersals diagonalising tau for system A.
    dlistB: List of dipole expectation values of the dispersals for system B.
    taulistA: List of diagonal kinetic matrix elements of the dispersals diagonalising tau for system B.
    
    Returns:
    C6avg: C6 averaged over orientations (isotropic).'''
    #Obtain number of b-states
    nbA = dlistA.shape[0]
    nbB = dlistB.shape[0]

    dlistA_square = np.zeros((nbA))
    dlistB_square = np.zeros((nbB))
    
    for i in numba.prange(nbA):
        dlistA_square[i] = dlistA[i, 0]**2+dlistA[i, 1]**2+dlistA[i, 2]**2

    for j in numba.prange(nbB):
        dlistB_square[j]= dlistB[j, 0]**2+dlistB[j, 1]**2+dlistB[j, 2]**2

    # Directly compute C6avg = sum_{ij} 4/3*|d_i|**2*|d_j|**2/(taulistA[i] + taulistB[j])
    C6avg = 0.
    for i in numba.prange(nbA):
        for j in numba.prange(nbB):
            C6avg +=  dlistA_square[i]*dlistB_square[j]/ (taulistA[i]+taulistB[j])
    C6avg = C6avg * 4/3
    return C6avg

@numba.njit(parallel=True)
def compute_anisotropic_C6(dlistA, taulistA, dlistB, taulistB):
    '''Computes isotropic C6, first anistropies Gamma6AB, Gamma6BA, Delta6

    Arguments:
    dlistA: List of dipole expectation values of the dispersals for system A.
    taulistA: List of diagonal kinetic matrix elements of the dispersals diagonalising tau for system A.
    dlistB: List of dipole expectation values of the dispersals for system B.
    taulistA: List of diagonal kinetic matrix elements of the dispersals diagonalising tau for system B.
    
    Returns:
    C6: C6 averaged over orientations (isotropic).
    Gamma6AB: Anisotropic coefficient in units of C6.
    Gamma6BA: Anisotropic coefficient in units of C6.
    Delta6: Anisotropic coefficient in units of C6.'''
    print('WARNING: anisotropies only valid when A and B are linear molecules (or atoms) and aligned along the z-axis.')
    #Obtain number of b-states
    nbA = dlistA.shape[0]
    nbB = dlistB.shape[0]


    dlistA_square = np.zeros((nbA))
    dlistA_diff = np.zeros((nbA))
    dlistB_square = np.zeros((nbB))
    dlistB_diff = np.zeros((nbB))

    for i in numba.prange(nbA):
        dlistA_square[i] = dlistA[i, 0]**2+dlistA[i, 1]**2+dlistA[i, 2]**2
        dlistA_diff[i] = -dlistA[i, 0]**2-dlistA[i, 1]**2+2*dlistA[i, 2]**2

    for j in numba.prange(nbB):
        dlistB_square[j]= dlistB[j, 0]**2+dlistB[j, 1]**2+dlistB[j, 2]**2
        dlistB_diff[j] = -dlistB[j, 0]**2-dlistB[j, 1]**2+2*dlistB[j, 2]**2
    
    C6 = 0.
    Gamma6AB = 0.
    Gamma6BA = 0.
    Delta6 = 0.
    for i in numba.prange(nbA):
        for j in numba.prange(nbB):
            C6 +=  dlistA_square[i]*dlistB_square[j]/ (taulistA[i]+taulistB[j])
            Gamma6AB +=  dlistA_diff[i]*dlistB_square[j]/ (taulistA[i]+taulistB[j])
            Gamma6BA +=  dlistA_square[i]*dlistB_diff[j]/ (taulistA[i]+taulistB[j])
            Delta6 += dlistA_diff[i]*dlistB_diff[j]/ (taulistA[i]+taulistB[j])
            
    C6 = C6*4./3.
    Gamma6AB = Gamma6AB*2./3./C6
    Gamma6BA = Gamma6BA*2./3./C6
    Delta6 = Delta6*1./3./C6
    return C6, Gamma6AB, Gamma6BA, Delta6
