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
  lipschitz - Lipschitz constant of the objective value
  maxiter - maximum number of iteration
  max_cpu_time - time laps limit
  diam - diameter of the feasible region
  tol - tolerance for Wolfe gap stopping criterion

Outputs:
  y - solution at final iteration
  obj - objective value at final iteration
  w_gap - wolfe gap at final iteration
  gaps - list of Wolfe gaps in all iteration
  k - number of iterations
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
from numpy.linalg import norm


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


# CGS function

def cgs(a_matrix, b, max_iter, max_cpu_time, lipschitz, diam, tol):
    t = time.time()
    ns = a_matrix.shape[1]  # X is n x m
    n = int((1 + math.sqrt(1 + 8 * ns)) / 2)

    x = tspGen(n,123)
    y = tspGen(n,321)
    a_x = a_matrix@x
    a_y = a_matrix@y

    a_mat_t = a_matrix.T

    w_gap = math.inf
    gaps = np.array([])
    obj = np.array([])
    inn_iter = np.array([])
    inn_iter_time = np.array([])
    cpu_time = np.array([])
    timeLapse = 0

    k = 0

    while timeLapse < max_cpu_time and w_gap > tol:
        k += 1
        beta = (3 * lipschitz) / (k + 1)
        gamma = 3 / (k + 2)
        eta = (lipschitz * (diam ** 2)) / (k * (k + 1))

        z = (1-gamma)*y + gamma*x
        a_z = (1-gamma)*a_y + gamma*a_x

        grad = a_mat_t @ (a_z-b)  # grad is 1 x n
        x, in_it, in_it_t = fun_cndg(grad, x, beta, eta, max_cpu_time)
        a_x = a_matrix@x

        inn_iter = np.append(inn_iter, in_it)
        inn_iter_time = np.append(inn_iter_time, in_it_t)

        y = ((1-gamma)*y) + (gamma*x)
        a_y = (1-gamma)*a_y + gamma*a_x

        #   Wolfe gap is $max_{u\in X}<f'(z), z - u>$

        grad_y = a_mat_t @ (a_y-b)
        x_gap = tsp(grad_y)
        w_gap = grad_y @ (y - x_gap)
        if w_gap == 0:
            x_gap = X2
        w_gap = grad_y @ (y - x_gap)
        w_gap = abs(w_gap)
        gaps = np.append(gaps, w_gap)

        #   If about to terminate due to small gap, set the solution to z

        if w_gap < tol:
            y = z

        #    Finding the dual objective value

        objective = .5 * (np.linalg.norm(a_y - b) ** 2)
        obj = np.append(obj, objective)

        #   saving runtime data

        cpu_time = np.append(cpu_time, time.time() - t)
        timeLapse = time.time()-t

    elapsed = time.time() - t

    return x, obj, w_gap, gaps, k, inn_iter, inn_iter_time, cpu_time, elapsed
