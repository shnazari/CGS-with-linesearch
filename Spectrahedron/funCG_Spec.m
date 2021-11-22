%{
Performing Conditional Gradient (CG) Algorithm
for solving min {.5||Ax-b||^2; x \in Spe} where 
    - A is a random rectangular matrix and objective function is not strongly
      convex. 
    - Feasible region is the standard spectrahedron {X \in R^nxn : Tr(X)=1, X>=0}
    - set X in this script is the set of all vertices of the feasible region
      in the paper "Backtracking linesearch for conditional gradient sliding"

inputs:
    A - random mxn matrix from sprandrect.m
    b - rand mx1 matrix
    par - structure of followings
        MaxIter - limit on maximum number of iteration
        MaxCPUtime - time limit
        Tol - tolerance on Wolfe gap

outputs:
    y - solution at final iteration
    etc - structure of following
        gap - array of Wolfe gaps in iterations
        obj - array objective values in iterations
        time_lapse - function timelaps
        objective - final objective value
        finalGap - final Wolfe gap
        CPUtime - array of each iteration timelaps

author: 
  Hamid Nazari - snazari@ clemson.edu
%}

function [y, etc] = funCG_Spec(A, b, par)

tStart = tic;
FWt = tic;

% Parameters
n2 = size(A,2);
n = sqrt(n2);
etc = [];
% etc.obj = nan(par.MaxIter, 1);
% etc.CPUTime = nan(par.MaxIter, 1);
% etc.gap = nan(par.MaxIter, 1);
MaxIter = par.MaxIter;
MaxCPUtime = par.MaxCPUtime;
CPUtime = 0;

% Main algorithm
At = A';
x = [1; zeros(n2-1,1)];
y = [1; zeros(n2-1,1)];
gap = Inf;
obj = Inf;
k = 0 ;

while obj>=par.Tol  && CPUtime<=MaxCPUtime && k<MaxIter% && gap>=par.Tol
    
    k = k+1;
    % parameter set up
    gamma = 3/(k+2);
    
    % FW body
    z = (1-gamma)*y + (gamma*x);
    
    grad = At*(A*z-b);
    
    matGrad = reshape(grad, n, n);
    symGrad = .5*(matGrad + matGrad');
    [evecs, ~] = eigs(symGrad, 1, 'smallestreal');
    
    xMat = evecs * evecs';
    x = reshape(xMat, n2, 1);
    
    y = (1-gamma)*y + (gamma*x);
    
    % Wolfe gap is $max_{u\in X}<f'(z), z - u>$
    gap = abs(grad'*(z - y));
    
    etc.gap(k) = gap;
    
    % If about to terminate due to small gap, set the solution to x_lb
    if gap<par.Tol
        y = z;
    end
    
    % Optimal objective
    obj = .5*(norm(A*y-b)^2);
    etc.obj(k) = obj;
    
    
    % Saving runtime data
    etc.CPUTime(k) = toc(tStart);
    CPUtime = toc(tStart);
    
end

etc.time_lapse = toc(FWt);
etc.Iterations = k;
etc.objective = etc.obj(k);
etc.finalGap = gap;
etc.CPUtime = CPUtime;

end
