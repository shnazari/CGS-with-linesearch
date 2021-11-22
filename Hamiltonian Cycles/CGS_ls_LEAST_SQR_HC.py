"""
Conditional Gradient Sliding (CGS) Algorithm
for solving min {.5||Ax-b||^2; x \in HC} where 
- A is a random rectangular matrix and objective function is not strongly
  convex. 
- Feasible region is the convex hull of all  Hamiltonian cycles of an undirected
  complete graph with n nodes
- set X in this script is the set of all vertices of the feasible region in the
  paper "Backtracking linesearch for conditional gradient sliding"
  
Inputs:
  a_matrix - rectangular random matrix of size mxn with predefined density
             and singular values
  b - random vector of size mx1
  maxiter - maximum number of iteration
  max_cpu_time - time laps limit
  diam - diameter of the feasible region
  l0 - initial guess of Lipscitz constant
  c - parameter defined in the paper
  tol - tolerance for Wolfe gap stopping criterion
  
Outputs:
  x - solution at final iteration
  obj - objective value at final iteration
  xiGap - gap defined in the paper
  gaps - list of gaps in all iteration
  ell - list of updated Ls in all iterations
  lCount - number of distincts Ls
  k - number of iterations
  inn_iter - list of number inner iterations in all outer iterations
  inn_iter_time - list of inner iterations time lapses in all outer iterations
  cpu_time - time laps in each iteration
  elapsed - total time laps
  
author:
    - Hamid Nazari snazari@clemson.edu
"""

import numpy as np
import math
import time
from CndG import fun_cndg
from TSP import tsp


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


# CGSls function

def cgs_ls(a_matrix, b, max_iter, max_cpu_time, diam, l0, c, tol):

    t = time.time()
    nss = a_matrix.shape[1]  # X is n x m
    n = int((1 + math.sqrt(1 + 8 * nss)) / 2)
    w_gap = math.inf
    xiGap = math.inf

    x = tspGen(n)
    y = tspGen(n)

    a_x = a_matrix @ x
    a_y = a_matrix @ y

    a_mat_t = a_matrix.T
    l_var = l0
    lCount = 0
    xiMin = 0

    ell = np.array([])
    gaps = np.array([])
    obj = np.array([])
    inn_iter = np.array([])
    inn_iter_time = np.array([])
    cpu_time = np.array([])
    timeLapse = 0

    k = 0
    i = 2

    while timeLapse < max_cpu_time and xiGap > tol:# and w_gap > tol:#
        k += 1
        ell = np.append(ell, l_var)

        ax_prev = a_x
        ay_prev = a_y
        x_prev = x
        y_prev = y

        while True:
            l_var = (2**(i-2))*l_var

            if k == 1:
                gamma = 1
                gamma_cap = l_var
            else:
                p = gamma_cap/l_var
                q = p/2-p*math.sqrt(.25+p/27)
                if q >= 0:
                    gamma = (p / 2 + p * math.sqrt(.25 + p / 27)) ** (1. / 3.) + q ** (1. / 3.)
                    gamma_cap = l_var * (gamma ** 3)
                else:
                    gamma = (p / 2 + p * math.sqrt(.25 + p / 27)) ** (1. / 3.) - abs(q) ** (1. / 3.)
                    gamma_cap = l_var * (gamma ** 3)

            beta = l_var*gamma
            eta = (c*l_var*gamma*(diam**2))/k

            z = (1-gamma)*y_prev + gamma*x_prev
            a_z = (1-gamma)*ay_prev + gamma*ax_prev

            grad = a_mat_t @ (a_z - b)  # grad is 1 x n
            x, in_it, in_it_t = fun_cndg(grad, x, beta, eta, max_cpu_time)
            a_x = a_matrix@x

            inn_iter = np.append(inn_iter, in_it)
            inn_iter_time = np.append(inn_iter_time, in_it_t)

            y = (1-gamma)*y + gamma*x
            a_y = (1-gamma)*ay_prev + gamma*a_x

            fy = .5 * (np.linalg.norm(a_y - b) ** 2)
            fz = .5 * (np.linalg.norm(a_z - b) ** 2)

            inn = grad @ (y-z)
            ns = .5 * (np.linalg.norm(y-z)**2)
            if fy <= fz+inn+(l_var*ns):
                i = 2
                objective = fy
                obj = np.append(obj, objective)
                break
            else:
                i = 3
                lCount += 1

        # finding the xi gap
        xi = tsp(gamma*grad)
        xiMin = (1-gamma)*xiMin + gamma*(fz+grad@(xi-z))
        xiGap = fy - xiMin

        cpu_time = np.append(cpu_time, time.time() - t)
        timeLapse = time.time()-t

    elapsed = time.time() - t
    return x, obj, xiGap, gaps, ell, lCount, k, inn_iter, inn_iter_time, cpu_time, elapsed
