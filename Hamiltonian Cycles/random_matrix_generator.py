"""
Generating a renctangular matrix of size mxn for given density and 
vector of singular values

author:
  Hamid Nazari
  Yuyuan Ouyang
"""

import numpy as np
import random as rnd
import math
from numpy import linalg as LA
from numpy.linalg import eig, eigvals
import scipy
from scipy.sparse import diags
from scipy.sparse.linalg import svds
import scipy.special as sc


# ------------------------------------------------------------------------
# Random Jacobi Rotation matrix
# Required libraries: random as rnd, numpy as np, math
def rjr(A):
    A_copy = A.copy()
    A_copy = scipy.sparse.lil_matrix(A_copy)
    [m,n] = [A_copy.get_shape()[0], A_copy.get_shape()[1]]
    if m != n:
        B_copy = float('NaN')
        print('A must be squared matrix (array)')
    else:
        pi = math.pi
        theta = (2*rnd.random()-1)*pi
        c = math.cos(theta)
        s = math.sin(theta)
        i = int(rnd.random()*m)
        j = i
        while (j==i):
            j = int(rnd.random() * m)
        B_copy = A_copy.copy()
        B_copy[[i, j]] = np.array([[c, s], [-s, c]]) @ A_copy[[i,j]]
        A_copy = B_copy.copy()
        B_copy[:,[i,j]] = A_copy[:,[i,j]] @ np.array([[c, -s], [s, c]])

    return B_copy

  
# ------------------------------------------------------------------------
# sprandsym function returns a sparse psd matrix for given size, density, and vector of eigenvalues
# required libraries: numpy, scipy
def sprandsym(n, density, rcond):
    lrc = len(rcond)
    density = min(density, 1)
    nnzwanted = np.round(density*n*n)

    if lrc == 1:
        ration = - rcond**(1/(n-1))
        anz = np.power(ratio, np.arange(n))
    else:
        anz = rcond
    R = scipy.sparse.diags(anz)
    nnzr = n

    while nnzr < .95*nnzwanted:
        R = rjr(R)
        nnzr = scipy.sparse.lil_matrix.count_nonzero(R)
    R = .5 * (R + R.T)
    return R


# ------------------------------------------------------------------------
# sprandrect function returns a sparse rectangular martix for given size, density, and vector of sigular
# values. The matrix must be a fat metrix and not tall
def sprandrect(m, n, density, sv):
    p = np.random.permutation(range(n))
    P = scipy.sparse.identity(n)
    P = scipy.sparse.lil_matrix(P)
    P = P[:,p]
    Q = scipy.sparse.random(n,m)
    Q = scipy.sparse.lil_matrix(Q)
    fac = n/m
    idxP = 0
    idxQ = 1

    while idxQ<=m:
        newidxP = int(np.round(idxQ*fac))
        Q[:, idxQ-1] = P[:,list(range(idxP,newidxP))]@(np.ones((newidxP-idxP,1))/math.sqrt(newidxP-idxP))
        idxQ += 1
        idxP = newidxP

    R = sprandsym(m, density, sv)
    A = R.T @ Q.T
    return A
