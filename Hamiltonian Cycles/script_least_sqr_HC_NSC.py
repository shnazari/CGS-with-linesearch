"""
Performing algorithms:
    - Conditional Gradient (CG)
    - Conditional Gradient Sliding (CGS)
    - Conditional Gradient Sliding with Linsearch (CGS-ls) 
for solving min {.5||Ax-b||^2; x \in HC} where 
- A is a random rectangular matrix and objective function is not strongly
  convex. 
- Feasible region is the convex hull of all  Hamiltonian cycles of an undirected
  complete graph with n nodes
- set X in this script is the set of all vertices of the feasible region in the
  paper "Backtracking linesearch for conditional gradient sliding"
authors: Hamid Nazari and Yuyuan Ouyang
snazari,yuyuano @ clemson.edu
"""

import math
import numpy as np
import scipy
from scipy.sparse import random
from CG_LEAST_SQR_HC import cg
from CGS_LEAST_SQR_HC import cgs
from CGS_ls_LEAST_SQR_HC import cgs_ls
from PAGD_LEAST_SQR_HC import pagd
from scipy.sparse.linalg import norm
from TSP import tsp
import random as rnd
from itertools import permutations
from random_matrix_generator import sprandrect


# ------------------------------------------------------------------------
# tspGen fn generates a feasible point for TSP for a given random seed and
# dimension n. With the default arguments it returns the point corresponds
# to the trivial path (0 1 ... n 0)
def tspGen(n, seed=1, isrand=False):
    p = np.array(range(n))
    if isrand:
        seeds = np.random.RandomState(seed)
        p = seeds.permutation(p)
#         print(p)
    p = np.append(p,p[0])
#     print(p)
    x = np.zeros((n, n))
    for i in range(n):
        if p[i] > p[i + 1]:
            x[p[i], p[i + 1]] = 1
        else:
            x[p[i + 1], p[i]] = 1
    x = x[np.tril_indices(n, k=-1)]
    return x
  
  
# ------------------------------------------------------------------------
# generating the parameters and initials
seed = 120
rnd.seed(seed)
m = 500
n = 25
n2 = int((n*(n-1))/2)

density = .6
m_half1 = int(n2/2)
m_half2 = n2 - m_half1

sv1 = np.zeros((1,m_half1))
sv2 = np.random.rand(m_half2)
sv = np.concatenate((sv1,sv2), axis=None)

A = sprandrect(n2, m, density, sv)
L_temp = norm(A.T @ A)
L = 1.5e4
A = (L*A)/L_temp
A = A.T

# finding an interior feasible point
x0 = tspGen(n, seed=123, isrand=True)
x1 = tspGen(n, seed=321, isrand=True)
lmd = .5 # convex combination parameter
y0 = lmd*x0+(1-lmd)*x1 # interior feasible pint

b = A @ y0
L0 = .001*L
diameter = math.sqrt(n2)
MaxIter = math.inf
MaxCPUTime = 30*60
Tol = 1e-2
c_mult = .05

# ------------------------------------------------------------------------
# CG
# ------------------------------------------------------------------------
CG_Opt_Sol, CG_Objective, CG_Gap, gaps, CG_Iter, CG_CPU_time, CG_Total_time\
    = cg(A, b, MaxIter, MaxCPUTime, Tol)

CG_end = CG_Iter - 1
print('')
print('CG:')
print('     iterations:', CG_Iter)
print('     Obj: ', CG_Objective[CG_end])
print('     CPU time:', CG_Total_time)
# print('     opt sln:', CG_Opt_Sol)
print('')

# ------------------------------------------------------------------------
# CGS
# ------------------------------------------------------------------------
CGS_Opt_Sol, CGS_Objective, CGS_Gap, CGS_Gaps, CGS_Iter, CGS_Inner_Iter, CGS_Inner_time,\
CGS_CPU_time, CGS_Total_time = cgs(A, b, MaxIter, MaxCPUTime, L, diameter, Tol)

CGS_end = CGS_Iter - 1
print('')
print('CGS:')
print('     Outer iterations:', CGS_Iter)
print('     Inner iterations:', sum(CGS_Inner_Iter))
print('     Obj:', CGS_Objective[CGS_end])
print('     CPU time:', CGS_Total_time)
print('     gap:', CGS_Gap)
# print('     sln:', CGS_Opt_Sol)
print('')

# ------------------------------------------------------------------------
# CGS-ls
# ------------------------------------------------------------------------
CGSls_Opt_Sol, CGSls_Objective, xiGap, CGSls_Gap, CGSls_Ls, LCount, CGSls_Iter, \
CGSls_Inner_Iter, CGSls_Inner_time, CGSls_CPU_time, CGSls_Total_time\
    = cgs_ls(A, b, MaxIter, MaxCPUTime, diameter, L0, c_mult, Tol)

CGSls_end = CGSls_Iter - 1
print('')
print('CGSls:')
print('     Outer iterations:', CGSls_Iter)
print('     Inner iterations:', sum(CGSls_Inner_Iter))
print('     xi Gap:', xiGap)
print('     Obj:', CGSls_Objective[CGSls_end])
print('     CPU time:', CGSls_Total_time)
print('     L counts:', LCount)
print('')
