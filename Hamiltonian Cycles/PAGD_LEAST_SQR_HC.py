"""
Projected Accelerated Gradient Descent (PAGD) Algorithm
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
  vertices - list of all possible vertices (cycles) in an undirected and
             complete graph of size n
  seed - value for random seed
  lipschitz - Lipschitz constant of the objective value
  maxiter - maximum number of iteration
  max_cpu_time - time laps limit
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
import random as rnd
import numpy as np
import math
import time
import gurobipy as gp
from gurobipy import GRB
from gurobipy import quicksum
# from scipy.sparse.linalg import norm
from TSP import tsp


# ------------------------------------------------------------------------
def projection_onto_HC(u, vertices):
    vertex_num = len(vertices)
    u_len = len(u)
    I = [i for i in range(u_len)]
    V = [i for i in range(vertex_num)]
    
    model = gp.Model()
    
    # variables
    x = model.addVars(I, vtype = GRB.CONTINUOUS, name = 'x')
    lam = model.addVars(V, vtype=GRB.CONTINUOUS, name = 'lambda')
    
    # objective
    obj = .5 * (quicksum(x[i]*x[i]+u[i]*u[i]-2*x[i]*u[i] for i in range(u_len)))
    model.setObjective(obj)
    
    # constraints
    model.addConstrs(x[j] - quicksum(lam[i]*vertices[i][j] for i in V)== 0 for j in I)
    model.addConstr(sum(lam[i] for i in V)==1)
    
    model.modelSense = GRB.MINIMIZE
    model.setParam('OutputFlag', 0)  # Muting the output
    model.optimize()
    
    gp_x = np.array(model.x[0:u_len])
    
    return gp_x


# ------------------------------------------------------------------------
def pagd(a_matrix, b, vertices, seed, lipschitz, max_iter, max_cpu_time, tol):
    
    t = time.time()
    n = vertices.shape[1]
    
    rnd.seed(seed)
    
    x = vertices[10]
    y = vertices[10]
    a_x = a_matrix@x
    a_y = a_matrix@y
    
    print(type(a_matrix))
    a_mat_t = a_matrix.T
    
    
    w_gap = math.inf
    gaps = np.array([])
    obj = np.array([])
    objective = 1
    cpu_time = np.array([])
    timeLapse = 0
    k = 0
    
    while timeLapse <= max_cpu_time and w_gap > tol:
        k += 1
        
        gamma = 2/(k+1)
        eta = 2*lipschitz/k
        
        z = (1-gamma)*y + gamma*x
        x = projection_onto_HC(x, vertices)
        a_x = a_matrix@x
        y = (1-gamma)*y + gamma*x
        a_y = (1-gamma)*a_y + gamma*a_x
        
        #   Wolfe gap is $max_{u\in X}<f'(z), z - u>$
        grad_y = a_mat_t @ (a_y - b)
        x_gap = tsp(grad_y)
        w_gap = grad_y @ (y - x_gap)
        gaps = np.append(gaps, w_gap)

        #   If about to terminate due to small gap, set the solution to z
        if w_gap < tol:
            y = z

        #    Finding the dual objective value
        objective = .5 * (np.linalg.norm(a_y - b) ** 2)
        obj = np.append(obj, objective)

        cpu_time = np.append(cpu_time, time.time() - t)
        timeLapse = time.time()-t

    elapsed = time.time() - t
    
    return 
    
