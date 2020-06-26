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


# Python imports
import numpy as np
import scipy.linalg

#Cython imports
from libc.math cimport sqrt
cimport cython
from cython.parallel import prange


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def build_b(Py_ssize_t nbase):
    """Builds the list of exponents from the highest power allowed
    
    Arguments:
    nbase: Highest power allowed in the dispersals x^i y^j z^k, where i+j+k < nbase
    
    Returns:
    bbasis: list of dispersals"""

    #Pre-allocate the array, number of basis function is as below
    nb = nbase*(nbase+1)*(nbase+2)//6-1
    bbasis = np.zeros((nb, 3),dtype=np.int64)
    #b_sum = np.zeros(nb, dtype=np.int64)
    cdef Py_ssize_t i = 0
    cdef Py_ssize_t s = 0
    cdef Py_ssize_t t = 0
    cdef Py_ssize_t u = 0
    cdef long[:, :] bbasis_view = bbasis
    #cdef long[:] b_sum_view = b_sum

    for s in range(nbase):
        for t in range(nbase-s):
            for u in range(nbase-s-t):
                if s+t+u != 0:
                    bbasis_view[i, 0] = s
                    bbasis_view[i, 1] = t
                    bbasis_view[i, 2] = u
                    #b_sum_view[i] = s+t+u
                    i+=1
    #sorted_indices = np.argsort(b_sum)
    
    #return bbasis[sorted_indices, :]
    return bbasis


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def build_b_spherical(Py_ssize_t nbase):
    """Builds the list of exponents from the highest power allowed.
    Version for spherical systems, where i+j+k even is ignored, 
    because they do not contribute due to symmetry.
    
    Arguments:
    nbase: Highest power allowed in the dispersals x^i y^j z^k, where i+j+k < nbase
    
    Returns:
    bbasis: list of dispersals"""

    if nbase%2 == 1:
        nbase -= 1
    
    nb = (2*nbase+5)*(nbase+2)*nbase//24 
    bbasis = np.zeros((nb, 3),dtype=np.int64)
    cdef Py_ssize_t i = 0
    cdef Py_ssize_t s = 0
    cdef Py_ssize_t t = 0
    cdef Py_ssize_t u = 0
    cdef long[:, :] bbasis_view = bbasis
    
    for s in range(nbase):
        for t in range(nbase-s):
            for u in range(nbase-s-t):
                if (s+t+u)%2 == 1:
                    bbasis_view[i, 0] = s
                    bbasis_view[i, 1] = t
                    bbasis_view[i, 2] = u
                    i += 1
    return bbasis


@cython.boundscheck(False)
@cython.cdivision(True)
def build_elements_full_unsafe(double Nel, double[:,:,:] xyz, double[:,:,:,:,:,:] xyz2, long[:,:] bbasis):
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
    Slist: Matrix of overlaps of the dispersals."""
    Nelinv = Nel**-1
    
    cdef Py_ssize_t nb = bbasis.shape[0]
    cdef long[:, :] bbasis_view = bbasis
    
    cdef double[:, :, :] xyz_view = xyz
    cdef double[:, :, :, :, :, :] xyz2_view = xyz2

    #Get the dipole moments
    d0list = np.array([xyz[1,0,0],xyz[0,1,0],xyz[0,0,1]])
    
    cdef double[:] d0list_view = d0list

    #Allocate plist and norms
    norm = np.empty(nb)
    plist = np.empty(nb)
    
    cdef double[:] norm_view = norm   
    cdef double[:] plist_view = plist
    
    cdef Py_ssize_t i = 0

    #Calculate plist and norms
    for i in prange(nb, nogil=True):
        plist_view[i] = xyz_view[bbasis_view[i, 0],bbasis_view[i, 1],bbasis_view[i, 2]]*Nelinv
        norm_view[i] = 1./sqrt(xyz_view[bbasis_view[i,0]*2,bbasis_view[i,1]*2,bbasis_view[i,2]*2]-Nel*plist_view[i]*plist_view[i]+xyz2_view[bbasis_view[i,0],bbasis_view[i,1],bbasis_view[i,2],bbasis_view[i,0],bbasis_view[i,1],bbasis_view[i,2]]-Nel*(Nel-1.)*plist_view[i]*plist_view[i])

    #Allocate dlist, taulist and Slist
    dlist = np.empty((nb, 3))
    taulist = np.empty((nb,nb))
    Slist = np.empty((nb,nb))
    
    cdef double[:, :] dlist_view = dlist 
    cdef double[:, :] taulist_view = taulist
    cdef double[:, :] Slist_view = Slist

    i = 0
    cdef Py_ssize_t j = 0
    
    #Calculate dlist, taulist and Slist
    for i in prange(nb, nogil=True):
        dlist_view[i, 0] = (xyz_view[bbasis_view[i,0]+1,bbasis_view[i,1],bbasis_view[i,2]]-d0list_view[0]*plist_view[i]\
                            +xyz2_view[1,0,0,bbasis_view[i,0],bbasis_view[i,1],bbasis_view[i,2]]-(Nel-1.)*d0list_view[0]*plist_view[i])*norm_view[i]
        dlist_view[i, 1] = (xyz_view[bbasis_view[i,0],bbasis_view[i,1]+1,bbasis_view[i,2]]-d0list_view[1]*plist_view[i]  \
                            +xyz2_view[0,1,0,bbasis_view[i,0],bbasis_view[i,1],bbasis_view[i,2]]-(Nel-1.)*d0list_view[1]*plist_view[i])*norm_view[i]
        dlist_view[i, 2] = (xyz_view[bbasis_view[i,0],bbasis_view[i,1],bbasis_view[i,2]+1]-d0list_view[2]*plist_view[i]\
                            +xyz2_view[0,0,1,bbasis_view[i,0],bbasis_view[i,1],bbasis_view[i,2]]-(Nel-1.)*d0list_view[2]*plist_view[i])*norm_view[i]        
        for j in range(nb):
            Slist_view[i, j] = (xyz_view[bbasis_view[i,0]+bbasis_view[j,0],bbasis_view[i,1]+bbasis_view[j,1],bbasis_view[i,2]+bbasis_view[j,2]]-Nel*plist_view[i]*plist_view[j] \
                                + xyz2_view[bbasis_view[i,0],bbasis_view[i,1],bbasis_view[i,2],bbasis_view[j,0],bbasis_view[j,1],bbasis_view[j,2]]-Nel*(Nel-1.)*plist_view[i]*plist_view[j])*norm_view[i]*norm_view[j]
            taulist_view[i, j] = norm_view[i]*norm_view[j]*(bbasis_view[i,0]*bbasis_view[j,0]*xyz_view[bbasis_view[i,0]+bbasis_view[j,0]-2,bbasis_view[i,1]+bbasis_view[j,1],bbasis_view[i,2]+bbasis_view[j,2]]+bbasis_view[i,1]*bbasis_view[j,1]*xyz_view[bbasis_view[i,0]+bbasis_view[j,0],bbasis_view[i,1]+bbasis_view[j,1]-2,bbasis_view[i,2]+bbasis_view[j,2]]+bbasis_view[i,2]*bbasis_view[j,2]*xyz_view[bbasis_view[i,0]+bbasis_view[j,0],bbasis_view[i,1]+bbasis_view[j,1],bbasis_view[i,2]+bbasis_view[j,2]-2])
    
    return dlist, taulist, Slist


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def build_elements_full(double Nel, double[:,:,:] xyz, double[:,:,:,:,:,:] xyz2, long[:,:] bbasis):
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
    Slist: Matrix of overlaps of the dispersals."""
    Nelinv = Nel**-1
    
    cdef Py_ssize_t nb = bbasis.shape[0]
    cdef long[:, :] bbasis_view = bbasis
    
    cdef double[:, :, :] xyz_view = xyz
    cdef double[:, :, :, :, :, :] xyz2_view = xyz2

    #Get the dipole moments
    d0list = np.array([xyz[1,0,0],xyz[0,1,0],xyz[0,0,1]])
    
    cdef double[:] d0list_view = d0list

    #Allocate plist and norms
    norm = np.empty(nb)
    plist = np.empty(nb)
    
    cdef double[:] norm_view = norm   
    cdef double[:] plist_view = plist
    
    cdef Py_ssize_t i = 0

    #Calculate plist and norms
    for i in prange(nb, nogil=True):
        plist_view[i] = xyz_view[bbasis_view[i, 0],bbasis_view[i, 1],bbasis_view[i, 2]]*Nelinv
        norm_view[i] = 1./sqrt(xyz_view[bbasis_view[i,0]*2,bbasis_view[i,1]*2,bbasis_view[i,2]*2]-Nel*plist_view[i]*plist_view[i]+xyz2_view[bbasis_view[i,0],bbasis_view[i,1],bbasis_view[i,2],bbasis_view[i,0],bbasis_view[i,1],bbasis_view[i,2]]-Nel*(Nel-1.)*plist_view[i]*plist_view[i])

    #Allocate dlist, taulist and Slist
    dlist = np.empty((nb, 3))
    taulist = np.empty((nb,nb))
    Slist = np.empty((nb,nb))
    
    cdef double[:, :] dlist_view = dlist 
    cdef double[:, :] taulist_view = taulist
    cdef double[:, :] Slist_view = Slist

    i = 0
    cdef Py_ssize_t j = 0
    
    #Calculate dlist, taulist and Slist
    for i in prange(nb, nogil=True):
        dlist_view[i, 0] = (xyz_view[bbasis_view[i,0]+1,bbasis_view[i,1],bbasis_view[i,2]]-d0list_view[0]*plist_view[i]\
                            +xyz2_view[1,0,0,bbasis_view[i,0],bbasis_view[i,1],bbasis_view[i,2]]-(Nel-1.)*d0list_view[0]*plist_view[i])*norm_view[i]
        dlist_view[i, 1] = (xyz_view[bbasis_view[i,0],bbasis_view[i,1]+1,bbasis_view[i,2]]-d0list_view[1]*plist_view[i]  \
                            +xyz2_view[0,1,0,bbasis_view[i,0],bbasis_view[i,1],bbasis_view[i,2]]-(Nel-1.)*d0list_view[1]*plist_view[i])*norm_view[i]
        dlist_view[i, 2] = (xyz_view[bbasis_view[i,0],bbasis_view[i,1],bbasis_view[i,2]+1]-d0list_view[2]*plist_view[i]\
                            +xyz2_view[0,0,1,bbasis_view[i,0],bbasis_view[i,1],bbasis_view[i,2]]-(Nel-1.)*d0list_view[2]*plist_view[i])*norm_view[i]        
        for j in range(nb):
            Slist_view[i, j] = (xyz_view[bbasis_view[i,0]+bbasis_view[j,0],bbasis_view[i,1]+bbasis_view[j,1],bbasis_view[i,2]+bbasis_view[j,2]]-Nel*plist_view[i]*plist_view[j] \
                                + xyz2_view[bbasis_view[i,0],bbasis_view[i,1],bbasis_view[i,2],bbasis_view[j,0],bbasis_view[j,1],bbasis_view[j,2]]-Nel*(Nel-1.)*plist_view[i]*plist_view[j])*norm_view[i]*norm_view[j]
            taulist_view[i, j] = 0.
            if bbasis_view[i, 0] > 0 and bbasis_view[j, 0] > 0:
                taulist_view[i, j] += norm_view[i]*norm_view[j]*bbasis_view[i,0]*bbasis_view[j,0]*xyz_view[bbasis_view[i,0]+bbasis_view[j,0]-2,bbasis_view[i,1]+bbasis_view[j,1],bbasis_view[i,2]+bbasis_view[j,2]]
            if bbasis_view[i, 1] > 0 and bbasis_view[j, 1] > 0:
                taulist_view[i, j] += norm_view[i]*norm_view[j]*bbasis_view[i,1]*bbasis_view[j,1]*xyz_view[bbasis_view[i,0]+bbasis_view[j,0],bbasis_view[i,1]+bbasis_view[j,1]-2,bbasis_view[i,2]+bbasis_view[j,2]]
            if bbasis_view[i, 2] > 0 and bbasis_view[j, 2] > 0:
                taulist_view[i, j] += norm_view[i]*norm_view[j]*bbasis_view[i,2]*bbasis_view[j,2]*xyz_view[bbasis_view[i,0]+bbasis_view[j,0],bbasis_view[i,1]+bbasis_view[j,1],bbasis_view[i,2]+bbasis_view[j,2]-2]

    
    return dlist, taulist, Slist


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def build_elements_full(double Nel, double[:,:,:] xyz, double[:,:,:,:,:,:] xyz2, long[:,:] bbasis):
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
    Slist: Matrix of overlaps of the dispersals."""
    Nelinv = Nel**-1
    
    cdef Py_ssize_t nb = bbasis.shape[0]
    cdef long[:, :] bbasis_view = bbasis
    
    cdef double[:, :, :] xyz_view = xyz
    cdef double[:, :, :, :, :, :] xyz2_view = xyz2

    #Get the dipole moments
    d0list = np.array([xyz[1,0,0],xyz[0,1,0],xyz[0,0,1]])
    
    cdef double[:] d0list_view = d0list

    #Allocate plist and norms
    norm = np.empty(nb)
    plist = np.empty(nb)
    
    cdef double[:] norm_view = norm   
    cdef double[:] plist_view = plist
    
    cdef Py_ssize_t i = 0

    #Calculate plist and norms
    for i in prange(nb, nogil=True):
        plist_view[i] = xyz_view[bbasis_view[i, 0],bbasis_view[i, 1],bbasis_view[i, 2]]*Nelinv
        norm_view[i] = 1./sqrt(xyz_view[bbasis_view[i,0]*2,bbasis_view[i,1]*2,bbasis_view[i,2]*2]-Nel*plist_view[i]*plist_view[i]+xyz2_view[bbasis_view[i,0],bbasis_view[i,1],bbasis_view[i,2],bbasis_view[i,0],bbasis_view[i,1],bbasis_view[i,2]]-Nel*(Nel-1.)*plist_view[i]*plist_view[i])

    #Allocate dlist, taulist and Slist
    dlist = np.empty((nb, 3))
    taulist = np.empty((nb,nb))
    Slist = np.empty((nb,nb))
    
    cdef double[:, :] dlist_view = dlist 
    cdef double[:, :] taulist_view = taulist
    cdef double[:, :] Slist_view = Slist

    i = 0
    cdef Py_ssize_t j = 0
    
    #Calculate dlist, taulist and Slist
    for i in prange(nb, nogil=True):
        dlist_view[i, 0] = (xyz_view[bbasis_view[i,0]+1,bbasis_view[i,1],bbasis_view[i,2]]-d0list_view[0]*plist_view[i]\
                            +xyz2_view[1,0,0,bbasis_view[i,0],bbasis_view[i,1],bbasis_view[i,2]]-(Nel-1.)*d0list_view[0]*plist_view[i])*norm_view[i]
        dlist_view[i, 1] = (xyz_view[bbasis_view[i,0],bbasis_view[i,1]+1,bbasis_view[i,2]]-d0list_view[1]*plist_view[i]  \
                            +xyz2_view[0,1,0,bbasis_view[i,0],bbasis_view[i,1],bbasis_view[i,2]]-(Nel-1.)*d0list_view[1]*plist_view[i])*norm_view[i]
        dlist_view[i, 2] = (xyz_view[bbasis_view[i,0],bbasis_view[i,1],bbasis_view[i,2]+1]-d0list_view[2]*plist_view[i]\
                            +xyz2_view[0,0,1,bbasis_view[i,0],bbasis_view[i,1],bbasis_view[i,2]]-(Nel-1.)*d0list_view[2]*plist_view[i])*norm_view[i]        
        for j in range(nb):
            Slist_view[i, j] = (xyz_view[bbasis_view[i,0]+bbasis_view[j,0],bbasis_view[i,1]+bbasis_view[j,1],bbasis_view[i,2]+bbasis_view[j,2]]-Nel*plist_view[i]*plist_view[j] \
                                + xyz2_view[bbasis_view[i,0],bbasis_view[i,1],bbasis_view[i,2],bbasis_view[j,0],bbasis_view[j,1],bbasis_view[j,2]]-Nel*(Nel-1.)*plist_view[i]*plist_view[j])*norm_view[i]*norm_view[j]
            taulist_view[i, j] = 0.
            if bbasis_view[i, 0] > 0 and bbasis_view[j, 0] > 0:
                taulist_view[i, j] += norm_view[i]*norm_view[j]*bbasis_view[i,0]*bbasis_view[j,0]*xyz_view[bbasis_view[i,0]+bbasis_view[j,0]-2,bbasis_view[i,1]+bbasis_view[j,1],bbasis_view[i,2]+bbasis_view[j,2]]
            if bbasis_view[i, 1] > 0 and bbasis_view[j, 1] > 0:
                taulist_view[i, j] += norm_view[i]*norm_view[j]*bbasis_view[i,1]*bbasis_view[j,1]*xyz_view[bbasis_view[i,0]+bbasis_view[j,0],bbasis_view[i,1]+bbasis_view[j,1]-2,bbasis_view[i,2]+bbasis_view[j,2]]
            if bbasis_view[i, 2] > 0 and bbasis_view[j, 2] > 0:
                taulist_view[i, j] += norm_view[i]*norm_view[j]*bbasis_view[i,2]*bbasis_view[j,2]*xyz_view[bbasis_view[i,0]+bbasis_view[j,0],bbasis_view[i,1]+bbasis_view[j,1],bbasis_view[i,2]+bbasis_view[j,2]-2]

    
    return dlist, taulist, Slist


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def build_elements_xconly(double Nel, double[:,:,:] xyz, double[:,:,:,:,:,:] xyz2, long[:,:] bbasis):
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
    Slist: Matrix of overlaps of the dispersals."""
    Nelinv = Nel**-1
    
    cdef Py_ssize_t nb = bbasis.shape[0]
    cdef long[:, :] bbasis_view = bbasis
    
    cdef double[:, :, :] xyz_view = xyz
    cdef double[:, :, :, :, :, :] xyz2_view = xyz2

    #Get the dipole moments
    d0list = np.array([xyz[1,0,0],xyz[0,1,0],xyz[0,0,1]])
    
    cdef double[:] d0list_view = d0list

    #Allocate plist and norms
    norm = np.empty(nb)
    plist = np.empty(nb)
    
    cdef double[:] norm_view = norm   
    cdef double[:] plist_view = plist
    
    cdef Py_ssize_t i = 0

    #Calculate plist and norms
    for i in prange(nb, nogil=True):
        plist_view[i] = xyz_view[bbasis_view[i, 0],bbasis_view[i, 1],bbasis_view[i, 2]]*Nelinv
        norm_view[i] = 1/sqrt(xyz_view[bbasis[i,0]*2,bbasis_view[i,1]*2,bbasis_view[i,2]*2]-xyz2_view[bbasis_view[i,0],bbasis_view[i,1],bbasis_view[i,2],bbasis_view[i,0],bbasis_view[i,1],bbasis_view[i,2]])

    #Allocate dlist, taulist and Slist
    dlist = np.empty((nb, 3))
    taulist = np.empty((nb,nb))
    Slist = np.empty((nb,nb))
    
    cdef double[:, :] dlist_view = dlist 
    cdef double[:, :] taulist_view = taulist
    cdef double[:, :] Slist_view = Slist

    i = 0
    cdef Py_ssize_t j = 0
    
    #Calculate dlist, taulist and Slist
    for i in prange(nb, nogil=True):
        dlist_view[i, 0] = (xyz_view[bbasis_view[i,0]+1,bbasis_view[i,1],bbasis_view[i,2]]\
                            -xyz2_view[1,0,0,bbasis_view[i,0],bbasis_view[i,1],bbasis_view[i,2]])*norm_view[i]
        dlist_view[i, 1] = (xyz_view[bbasis_view[i,0],bbasis_view[i,1]+1,bbasis_view[i,2]]  \
                            -xyz2_view[0,1,0,bbasis_view[i,0],bbasis_view[i,1],bbasis_view[i,2]])*norm_view[i]
        dlist_view[i, 2] = (xyz_view[bbasis_view[i,0],bbasis_view[i,1],bbasis_view[i,2]+1]\
                            -xyz2_view[0,0,1,bbasis_view[i,0],bbasis_view[i,1],bbasis_view[i,2]])*norm_view[i]        
        for j in range(nb):
            Slist_view[i, j] = (xyz_view[bbasis_view[i,0]+bbasis_view[j,0],bbasis_view[i,1]+bbasis_view[j,1],bbasis_view[i,2]+bbasis_view[j,2]] \
                                -xyz2_view[bbasis_view[i,0],bbasis_view[i,1],bbasis_view[i,2],bbasis_view[j,0],bbasis_view[j,1],bbasis_view[j,2]])*norm_view[i]*norm_view[j]
            taulist_view[i, j] = 0.
            if bbasis_view[i, 0] > 0 and bbasis_view[j, 0] > 0:
                taulist_view[i, j] += norm_view[i]*norm_view[j]*bbasis_view[i,0]*bbasis_view[j,0]*xyz_view[bbasis_view[i,0]+bbasis_view[j,0]-2,bbasis_view[i,1]+bbasis_view[j,1],bbasis_view[i,2]+bbasis_view[j,2]]
            if bbasis_view[i, 1] > 0 and bbasis_view[j, 1] > 0:
                taulist_view[i, j] += norm_view[i]*norm_view[j]*bbasis_view[i,1]*bbasis_view[j,1]*xyz_view[bbasis_view[i,0]+bbasis_view[j,0],bbasis_view[i,1]+bbasis_view[j,1]-2,bbasis_view[i,2]+bbasis_view[j,2]]
            if bbasis_view[i, 2] > 0 and bbasis_view[j, 2] > 0:
                taulist_view[i, j] += norm_view[i]*norm_view[j]*bbasis_view[i,2]*bbasis_view[j,2]*xyz_view[bbasis_view[i,0]+bbasis_view[j,0],bbasis_view[i,1]+bbasis_view[j,1],bbasis_view[i,2]+bbasis_view[j,2]-2]

    
    return dlist, taulist, Slist


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def build_elements_hartree(double Nel, double[:,:,:] xyz, long[:,:] bbasis):
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
    Slist: Matrix of overlaps of the dispersals."""
    Nelinv = Nel**-1
    
    cdef Py_ssize_t nb = bbasis.shape[0]
    cdef long[:, :] bbasis_view = bbasis
    
    cdef double[:, :, :] xyz_view = xyz

    #Get the dipole moments
    d0list = np.array([xyz[1,0,0],xyz[0,1,0],xyz[0,0,1]])
    
    cdef double[:] d0list_view = d0list

    #Allocate plist and norms
    norm = np.empty(nb)
    plist = np.empty(nb)
    
    cdef double[:] norm_view = norm   
    cdef double[:] plist_view = plist
    
    cdef Py_ssize_t i = 0

    #Calculate plist and norms
    for i in prange(nb, nogil=True):
        plist_view[i] = xyz_view[bbasis_view[i, 0],bbasis_view[i, 1],bbasis_view[i, 2]]*Nelinv
        norm_view[i] = 1/sqrt(xyz_view[bbasis[i,0]*2,bbasis_view[i,1]*2,bbasis_view[i,2]*2])

    #Allocate dlist, taulist and Slist
    dlist = np.empty((nb, 3))
    taulist = np.empty((nb,nb))
    Slist = np.empty((nb,nb))
    
    cdef double[:, :] dlist_view = dlist 
    cdef double[:, :] taulist_view = taulist
    cdef double[:, :] Slist_view = Slist

    i = 0
    cdef Py_ssize_t j = 0
    
    #Calculate dlist, taulist and Slist
    for i in prange(nb, nogil=True):
        dlist_view[i, 0] = xyz_view[bbasis_view[i,0]+1,bbasis_view[i,1],bbasis_view[i,2]]*norm_view[i]
        dlist_view[i, 1] = xyz_view[bbasis_view[i,0],bbasis_view[i,1]+1,bbasis_view[i,2]]*norm_view[i]
        dlist_view[i, 2] = xyz_view[bbasis_view[i,0],bbasis_view[i,1],bbasis_view[i,2]+1]*norm_view[i]        
        for j in range(nb):
            Slist_view[i, j] = xyz_view[bbasis_view[i,0]+bbasis_view[j,0],bbasis_view[i,1]+bbasis_view[j,1],bbasis_view[i,2]+bbasis_view[j,2]]*norm_view[i]*norm_view[j]
            taulist_view[i, j] = 0.
            if bbasis_view[i, 0] > 0 and bbasis_view[j, 0] > 0:
                taulist_view[i, j] += norm_view[i]*norm_view[j]*bbasis_view[i,0]*bbasis_view[j,0]*xyz_view[bbasis_view[i,0]+bbasis_view[j,0]-2,bbasis_view[i,1]+bbasis_view[j,1],bbasis_view[i,2]+bbasis_view[j,2]]
            if bbasis_view[i, 1] > 0 and bbasis_view[j, 1] > 0:
                taulist_view[i, j] += norm_view[i]*norm_view[j]*bbasis_view[i,1]*bbasis_view[j,1]*xyz_view[bbasis_view[i,0]+bbasis_view[j,0],bbasis_view[i,1]+bbasis_view[j,1]-2,bbasis_view[i,2]+bbasis_view[j,2]]
            if bbasis_view[i, 2] > 0 and bbasis_view[j, 2] > 0:
                taulist_view[i, j] += norm_view[i]*norm_view[j]*bbasis_view[i,2]*bbasis_view[j,2]*xyz_view[bbasis_view[i,0]+bbasis_view[j,0],bbasis_view[i,1]+bbasis_view[j,1],bbasis_view[i,2]+bbasis_view[j,2]-2]

    
    return dlist, taulist, Slist


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def build_elements_no2rdm(double Nel, double[:,:,:] xyz, long[:,:] bbasis):
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
    
    cdef Py_ssize_t nb = bbasis.shape[0]
    cdef long[:, :] bbasis_view = bbasis
    
    cdef double[:, :, :] xyz_view = xyz

    #Get the dipole moments
    d0list = np.array([xyz[1,0,0],xyz[0,1,0],xyz[0,0,1]])
    
    cdef double[:] d0list_view = d0list

    #Allocate plist and norms
    norm = np.empty(nb)
    plist = np.empty(nb)
    
    cdef double[:] norm_view = norm   
    cdef double[:] plist_view = plist
    
    cdef Py_ssize_t i = 0

    #Calculate plist and norms
    for i in range(nb):
        plist_view[i] = xyz_view[bbasis_view[i, 0],bbasis_view[i, 1],bbasis_view[i, 2]]*Nelinv
        norm_view[i] = 1./sqrt(xyz_view[bbasis_view[i,0]*2,bbasis_view[i,1]*2,bbasis_view[i,2]*2]-Nel*plist_view[i]*plist_view[i])

    #Allocate dlist, taulist and Slist
    dlist = np.empty((nb, 3))
    taulist = np.empty((nb,nb))
    Slist = np.empty((nb,nb))
    
    cdef double[:, :] dlist_view = dlist 
    cdef double[:, :] taulist_view = taulist
    cdef double[:, :] Slist_view = Slist

    i = 0
    cdef Py_ssize_t j = 0
    
    #Calculate dlist, taulist and Slist
    for i in prange(nb, nogil=True):
        dlist_view[i, 0] = (xyz_view[bbasis_view[i,0]+1,bbasis_view[i,1],bbasis_view[i,2]]-d0list_view[0]*plist_view[i])*norm_view[i]
        dlist_view[i, 1] = (xyz_view[bbasis_view[i,0],bbasis_view[i,1]+1,bbasis_view[i,2]]-d0list_view[1]*plist_view[i])*norm_view[i]
        dlist_view[i, 2] = (xyz_view[bbasis_view[i,0],bbasis_view[i,1],bbasis_view[i,2]+1]-d0list_view[2]*plist_view[i])*norm_view[i]        
        for j in range(nb):
            Slist_view[i, j] = (xyz_view[bbasis_view[i,0]+bbasis_view[j,0],bbasis_view[i,1]+bbasis_view[j,1],bbasis_view[i,2]+bbasis_view[j,2]]-Nel*plist_view[i]*plist_view[j])*norm_view[i]*norm_view[j]
            taulist_view[i, j] = 0.
            if bbasis_view[i, 0] > 0 and bbasis_view[j, 0] > 0:
                taulist_view[i, j] += norm_view[i]*norm_view[j]*bbasis_view[i,0]*bbasis_view[j,0]*xyz_view[bbasis_view[i,0]+bbasis_view[j,0]-2,bbasis_view[i,1]+bbasis_view[j,1],bbasis_view[i,2]+bbasis_view[j,2]]
            if bbasis_view[i, 1] > 0 and bbasis_view[j, 1] > 0:
                taulist_view[i, j] += norm_view[i]*norm_view[j]*bbasis_view[i,1]*bbasis_view[j,1]*xyz_view[bbasis_view[i,0]+bbasis_view[j,0],bbasis_view[i,1]+bbasis_view[j,1]-2,bbasis_view[i,2]+bbasis_view[j,2]]
            if bbasis_view[i, 2] > 0 and bbasis_view[j, 2] > 0:
                taulist_view[i, j] += norm_view[i]*norm_view[j]*bbasis_view[i,2]*bbasis_view[j,2]*xyz_view[bbasis_view[i,0]+bbasis_view[j,0],bbasis_view[i,1]+bbasis_view[j,1],bbasis_view[i,2]+bbasis_view[j,2]-2]

    
    return dlist, taulist, Slist


def compute_elements(double Nel, double[:,:,:] xyz, double[:,:,:,:,:,:] xyz2=None, long nbase=3, bint spherical=False, str twordm='full'):
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
    taulist: List of diagonal kinetic matrix elements of the dispersals diagonalising tau. """
    
    #Construct multipole function basis
    bbasis = build_b_spherical(nbase) if spherical else build_b(nbase)

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

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def compute_C6(double[:,:] dlistA, double[:] taulistA, double[:,:] dlistB, double[:] taulistB):
    '''Computes  C6 at orientation fixed from dlist and taulist of the two systems
    
    Arguments:
    dlistA: List of dipole expectation values of the dispersals for system A.
    taulistA: List of diagonal kinetic matrix elements of the dispersals diagonalising tau for system A.
    dlistB: List of dipole expectation values of the dispersals for system B.
    taulistA: List of diagonal kinetic matrix elements of the dispersals diagonalising tau for system B.
    
    Returns:
    C6: C6 at fixed orientation'''

    #Obtain number of b-states
    cdef Py_ssize_t nbA = dlistA.shape[0]
    cdef Py_ssize_t nbB = dlistB.shape[0]

    cdef double[:,:] dlistA_view = dlistA
    cdef double[:,:] dlistB_view = dlistB
    cdef double[:] taulistA_view = taulistA
    cdef double[:] taulistB_view = taulistB
    
    #Directly compute C6 = sum_{ij} 2*(wvec[i, j])**2/(taulistA[i] + taulistB[j])
    cdef double C6 = 0.
    
    cdef Py_ssize_t i = 0
    cdef Py_ssize_t j = 0
    
    for i in prange(nbA, nogil=True):
        for j in range(nbB):
            C6 +=  2*(dlistA_view[i, 0]*dlistB_view[j,0]+dlistA_view[i, 1]*dlistB_view[j,1]-2*dlistA_view[i, 2]*dlistB_view[j,2])**2\
            / (taulistA_view[i]+taulistB_view[j])
    return C6


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def compute_isotropic_C6(double[:,:] dlistA, double[:] taulistA, double[:,:] dlistB, double[:] taulistB):
    '''Computes isotropic C6 from dlist and taulist of the two systems
    
    Arguments:
    dlistA: List of dipole expectation values of the dispersals for system A.
    taulistA: List of diagonal kinetic matrix elements of the dispersals diagonalising tau for system A.
    dlistB: List of dipole expectation values of the dispersals for system B.
    taulistA: List of diagonal kinetic matrix elements of the dispersals diagonalising tau for system B.
    
    Returns:
    C6avg: C6 averaged over orientations (isotropic).'''
    #Obtain number of b-states
    cdef Py_ssize_t nbA = dlistA.shape[0]
    cdef Py_ssize_t nbB = dlistB.shape[0]
    
    cdef double[:,:] dlistA_view = dlistA
    cdef double[:,:] dlistB_view = dlistB
    cdef double[:] taulistA_view = taulistA
    cdef double[:] taulistB_view = taulistB
    
    #Directly compute C6 = sum_{ij} 2*(wvec[i, j])**2/(taulistA[i] + taulistB[j])

    dlistA_square = np.zeros((nbA))
    dlistB_square = np.zeros((nbB))
    
    cdef double[:] dlistA_square_view = dlistA_square
    cdef double[:] dlistB_square_view = dlistB_square
    
    cdef Py_ssize_t i = 0
    cdef Py_ssize_t j = 0
    
    for i in prange(nbA, nogil=True):
        dlistA_square_view[i] = dlistA_view[i, 0]**2+dlistA_view[i, 1]**2+dlistA_view[i, 2]**2

    for j in prange(nbB, nogil=True):
        dlistB_square_view[j]= dlistB_view[j, 0]**2+dlistB_view[j, 1]**2+dlistB_view[j, 2]**2
    
    i = 0
    j = 0
    cdef double C6avg = 0.
    for i in prange(nbA, nogil=True):
        for j in range(nbB):
            C6avg +=  dlistA_square_view[i]*dlistB_square_view[j]/ (taulistA_view[i]+taulistB_view[j])
    
    C6avg = C6avg*4./3.
    return C6avg


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def compute_anisotropic_C6(double[:,:] dlistA, double[:] taulistA, double[:,:] dlistB, double[:] taulistB):
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
    cdef Py_ssize_t nbA = dlistA.shape[0]
    cdef Py_ssize_t nbB = dlistB.shape[0]
    
    cdef double[:,:] dlistA_view = dlistA
    cdef double[:,:] dlistB_view = dlistB
    cdef double[:] taulistA_view = taulistA
    cdef double[:] taulistB_view = taulistB
    
    #Directly compute C6 = sum_{ij} 2*(wvec[i, j])**2/(taulistA[i] + taulistB[j])

    dlistA_square = np.zeros((nbA))
    dlistA_diff = np.zeros((nbA))
    dlistB_square = np.zeros((nbB))
    dlistB_diff = np.zeros((nbB))
    
    cdef double[:] dlistA_square_view = dlistA_square
    cdef double[:] dlistA_diff_view = dlistA_diff
    cdef double[:] dlistB_square_view = dlistB_square
    cdef double[:] dlistB_diff_view = dlistB_diff
    
    cdef Py_ssize_t i = 0
    cdef Py_ssize_t j = 0
        
    for i in prange(nbA, nogil=True):
        dlistA_square_view[i] = dlistA_view[i, 0]**2+dlistA_view[i, 1]**2+dlistA_view[i, 2]**2
        dlistA_diff_view[i] = -dlistA_view[i, 0]**2-dlistA_view[i, 1]**2+2*dlistA_view[i, 2]**2

    for j in prange(nbA, nogil=True):
        dlistB_square_view[j]= dlistB_view[j, 0]**2+dlistB_view[j, 1]**2+dlistB_view[j, 2]**2
        dlistB_diff_view[j] = -dlistB_view[j, 0]**2-dlistB_view[j, 1]**2+2*dlistB_view[j, 2]**2
    
    i = 0
    j = 0
    cdef double C6 = 0.
    cdef double Gamma6AB = 0.
    cdef double Gamma6BA = 0.
    cdef double Delta6 = 0.
    for i in prange(nbA, nogil=True):
        for j in range(nbB):
            C6 +=  dlistA_square_view[i]*dlistB_square_view[j]/ (taulistA_view[i]+taulistB_view[j])
            Gamma6AB +=  dlistA_diff_view[i]*dlistB_square_view[j]/ (taulistA_view[i]+taulistB_view[j])
            Gamma6BA +=  dlistA_square_view[i]*dlistB_diff_view[j]/ (taulistA_view[i]+taulistB_view[j])
            Delta6 += dlistA_diff_view[i]*dlistB_diff_view[j]/ (taulistA_view[i]+taulistB_view[j])
    C6 = C6*4./3.
    Gamma6AB = Gamma6AB*2./3./C6
    Gamma6BA = Gamma6BA*2./3./C6
    Delta6 = Delta6*1./3./C6
    return C6, Gamma6AB, Gamma6BA, Delta6


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def compute_coeffs(double[:,:] dlistA, double[:] taulistA, double[:,:] dlistB, double[:] taulistB):
    '''Computes coefficients c in the expansion of J from dlist and taulist of the two systems'''
    #Obtain number of b-states
    nbA = dlistA.shape[0]
    nbB = dlistB.shape[0]
    
    cdef double[:,:] dlistA_view = dlistA
    cdef double[:,:] dlistB_view = dlistB
    cdef double[:] taulistA_view = taulistA
    cdef double[:] taulistB_view = taulistB
    
    #Directly compute C6 = sum_{ij} 2*(wvec[i, j])**2/(taulistA[i] + taulistB[j])
    coeffs = np.empty((nbA, nbB))
    cdef double[:,:] coeffs_view = coeffs
    
    cdef Py_ssize_t i = 0
    cdef Py_ssize_t j = 0
    
    for i in prange(nbA, nogil=True):
        for j in range(nbB):
            coeffs_view[i, j] =  -4*(dlistA_view[i, 0]*dlistB_view[j,0]+dlistA_view[i, 1]*dlistB_view[j,1]-2*dlistA_view[i, 2]*dlistB_view[j,2])\
            / (taulistA_view[i]+taulistB_view[j])
    return coeffs


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def compute_coeffs_x(double[:, :] dlistA, double[:] taulistA, double[:, :] dlistB, double[:] taulistB):
    '''Computes coefficients c along x in the expansion of J from dlist and taulist of the two systems'''

    #Obtain number of b-states
    nbA = dlistA.shape[0]
    nbB = dlistB.shape[0]
    
    cdef double[:,:] dlistA_view = dlistA
    cdef double[:,:] dlistB_view = dlistB
    cdef double[:] taulistA_view = taulistA
    cdef double[:] taulistB_view = taulistB
    
    #Directly compute C6 = sum_{ij} 2*(wvec[i, j])**2/(taulistA[i] + taulistB[j])
    coeffs = np.empty((nbA, nbB))
    cdef double[:,:] coeffs_view = coeffs
    
    cdef Py_ssize_t i = 0
    cdef Py_ssize_t j = 0
    
    for i in prange(nbA, nogil=True):
        for j in range(nbB):
            coeffs_view[i, j] =  -4*(dlistA_view[i, 0]*dlistB_view[j,0])\
            / (taulistA_view[i]+taulistB_view[j])
    return coeffs
