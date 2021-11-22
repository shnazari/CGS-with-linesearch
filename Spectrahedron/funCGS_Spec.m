%{
Performing Conditional Gradient Sliding (CGS) Algorithm
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
        L - Lipschitz constant
        diam - diameter of feasible set
        MaxIter - limit on maximum number of iteration
        MaxCPUtime - time limit
        Tol - tolerance on Wolfe gap

outputs:
    y - solution at final iteration
    etc - structure of following
        inner_iter - numnber of inner iterations
        inner_iter_time_lapse - array of inner iterations timelaps
        gap - array of Wolfe gaps in iterations
        obj - array objective values in iterations
        time_lapse - function timelaps
        objective - final objective value
        finalGap - final Wolfe gap
        CPUtime - array of each iteration timelaps

author: 
  Hamid Nazari - snazari@ clemson.edu
%}
function  [y, etc] = funCGS_Spec(A, b, par)

tStart = tic;
CGS = tic;

% Parameters
n2 = size(A,2);
n = sqrt(n2);
etc = [];
L = par.L;
MaxIter = par.MaxIter;
D = par.diam;
MaxCPUtime = par.MaxCPUtime;
CPUtime = 0;

% Main algorithm
At = A';
x = [1;zeros(n2-1,1)];
y = [1;zeros(n2-1,1)];
gap = Inf;
obj = Inf;
k = 0 ;

while obj>par.Tol && k<MaxIter && CPUtime<=MaxCPUtime% && gap>=par.Tol
    
    k = k+1;
    
    %parameter et up
    beta = (3*L)/(k+1);
%     gamma = 2/(k+1);
    gamma = 3/(k+2);
    eta = (L*(D^2))/(k*(k+1));
    
    % CGS body
    z = (1-gamma)*y + gamma*x;
    
    grad = At*(A*z-b);
    [x, etc.inner_iter(k), etc.inner_iter_time_lapse(k)] = funCndG(grad ,x ,beta, eta, MaxCPUtime); % Inner iterations
    
    y = (1-gamma)*y + gamma*x;
    
    % Wolfe gap is $max_{u\in X}<f'(z), z - u>$
    gap = abs(grad'*(z - y));
    
    etc.gap(k) = gap;
    
    % If about to terminate due to small gap, set the solution to x_lb
    if gap<par.Tol
        y = z;
    end
    
    % objective
    obj = .5*norm(A*y-b)^2;
    etc.obj(k) = obj;
    
    % Save runtime data
    etc.CPUTime(k) = toc(tStart);
    CPUtime = toc(tStart);
    
end

etc.Outer_Time = toc(CGS);
etc.Iterations = k;
etc.objective = etc.obj(k);
etc.finalGap = gap;
etc.CPUtime = CPUtime;

end
