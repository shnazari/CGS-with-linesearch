"""
Performing algorithm
    - Projected Accelerated Gradient Descent (PAGD) 
for solving min {.5||Ax-b||^2; x \in HC} where 
- A is a random rectangular matrix and objective function is not (neccessarily)
  strongly convex. 
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
from PAGD_LEAST_SQR_HC import pagd
from scipy.sparse.linalg import norm
from TSP import tsp
import random as rnd
from itertools import permutations
from random_matrix_generator import sprandrect
import matplotlib.pyplot as plt


# ------------------------------------------------------------------------
# tspGen fn generates a feasible point for TSP for a given random seed and
# dimension n. With the default arguments it returns the point corresponds
# to the trivial path (0 1 ... n 0)
def tspGen(n, seed=1, isrand=False):
    p = np.array(range(n))
    if isrand:
        seeds = np.random.RandomState(seed)
        p = seeds.permutation(p)
    p = np.append(p,p[0])
    x = np.zeros((n, n))
    for i in range(n):
        if p[i] > p[i + 1]:
            x[p[i], p[i + 1]] = 1
        else:
            x[p[i + 1], p[i]] = 1
    x = x[np.tril_indices(n, k=-1)]
    return x
  

# ------------------------------------------------------------------------
def cycGen(n):
  l = int(n*(n-1)/2)
  cycles = np.zeros((1,l))
  for p in permutations(range(n)):
      p = np.append(p,p[0])
      x = np.zeros((n,n))
      for i in range(n):
          if p[i] > p[i + 1]:
              x[p[i], p[i + 1]] = 1
          else:
              x[p[i + 1], p[i]] = 1
      x = x[np.tril_indices(n, k=-1)]
      cnt = 0
      for j in range(len(cycles)):
          if np.array_equal(x, cycles[j]):
              cnt += 1
      if cnt == 0:
          cycles = np.concatenate((cycles,[x]), axis=0)
  cycles = cycles[1:]
  return cycles

# ------------------------------------------------------------------------
# generating the parameters and initials
seed = 120
rnd.seed(seed)

m = 100
n = 10
n2 = int((n*(n-1))/2)
density = .2
m_half1 = int(n2/2)
m_half2 = n2 - m_half1
sv1 = np.zeros((1,m_half1))
sv2 = np.random.rand(m_half2)
sv = np.concatenate((sv1,sv2), axis=None)

A = sprandrect(n2, m, density, sv)
L_temp = norm(A.T @ A)
L = 1000
A = (L*A)/L_temp
A = A.T
X = cycGen(n)

# finding an interior feasible point
x0 = tspGen(n, seed=123, isrand=True)
x1 = tspGen(n, seed=321, isrand=True)
lmd = .5 # convex combination parameter
y0 = lmd*x0+(1-lmd)*x1 # interior feasible pint

b = A @ y0
L0 = .01*L

diameter = math.sqrt(n2)
MaxIter = math.inf
MaxCPUTime = 60*60
Tol = 1e-2
c_mult = .05

# ------------------------------------------------------------------------
PAGD_Opt_Sol, PAGD_Objective, PAGD_Gap, PAGD_gaps, PAGD_Iter, PAGD_CPU_time, \
PAGD_Total_time = pagd(A, b, X, seed, L, MaxIter, MaxCPUTime, Tol)

PAGD_end = PAGD_Iter - 1
print('')
print('PAGD:')
print('     Outer iterations:', PAGD_Iter)
print('     Obj:', PAGD_Objective[PAGD_end])
print('     gap:', PAGD_gaps[PAGD_end])
print('     CPU time:', PAGD_Total_time)
print('')

# ------------------------------------------------------------------------
plt.figure()
plt.plot(PAGD_Objective, 'r--', label='PAGD')
plt.xlabel('Iteration')
plt.ylabel('Obj')
plt.legend(loc='upper right')
