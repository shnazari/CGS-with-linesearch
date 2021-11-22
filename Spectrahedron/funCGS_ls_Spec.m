%{
Performing Conditional Gradient Sliding with linsearch (CGS-ls) Algorithm
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
        L0 - initial guess of Lipschitz constant
        diam - diameter of feasible set
        c - constant defined in CGS-ls algorithm
        MaxIter - limit on maximum number of iteration
        MaxCPUtime - time limit
        Tol - tolerance on Wolfe gap

outputs:
    y - solution at final iteration
    etc - structure of following
        L - array of distinct updats of first gues L0
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
function  [y, etc] = funCGS_ls_Spec(A, b, par)

tStart = tic;
CGSt = tic;

% Parameters
n2 = size(A,2);
n = sqrt(n2);
etc = [];
% etc.obj = nan(par.MaxIter, 1);
% etc.inner_iter = nan(par.MaxIter, 1);
% etc.inner_iter_time_lapse = nan(par.MaxIter, 1);
% etc.CPUTime = nan(par.MaxIter, 1);
% etc.gap = nan(par.MaxIter, 1);
L = par.L0;
MaxIter = par.MaxIter;
D = par.diam;
MaxCPUtime = par.MaxCPUtime;
CPUtime = 0;
% N = MaxIter;
% Main algorithm
At = A';
x = [1;zeros(n2-1,1)];
y = [1;zeros(n2-1,1)];
gap = Inf;
obj = Inf;
i=2;
k = 0 ;
while obj>par.Tol && CPUtime<=MaxCPUtime % && k<MaxIter && gap>=par.Tol
    
    k = k+1;
    etc.L(k) = L;
    
    x_prev = x;
    y_prev = y;
    while 1
        L = 2^(i-2)*L;
        if k==1
            gamma = 1;
        else
            % calculating gamma as a third root of  Gamma_cap = L*gamma^3
            p = Gamma_cap;
            sqrtDelta = sqrt(729*p^2 + 108*p^3);
            gamma = ((sqrtDelta + 27*p)^(1/3) - (sqrtDelta - 27*p)^(1/3))/3/2^(1/3);
            
            %             gamma = 3/(k+2);
            etc.gamma(k) = gamma:
            %             p = Gamma_cap;
            %             sqrtDelta = sqrt(p^2+4*L*p);
            %             gamma = (-Gamma_cap + sqrtDelta)/2*L;
        end
        % parameter set up
        beta = L*gamma;
        
        eta = (par.c*L*gamma*D^2)/k;
        %         eta = L*gamma*D^2/N;
        
        % CGS type body
        z = (1-gamma)*y_prev + gamma*x_prev;
        
        grad = At*(A*z-b);
        [x, etc.inner_iter(k), etc.inner_iter_time_lapse(k)] = funCndG(grad ,x, beta, eta, par); %Subproblem
        y = (1-gamma)*y_prev + gamma*x;
        
        % Wolfe gap is $max_{u\in X}<f'(z), z - u>$
        gap = grad'*(z - y);
        etc.gap(k) = gap;
        
        % If about to terminate due to small gap, set the solution to x_lb
        if gap<par.Tol
            y = z;
        end
        
        fy = .5*norm(A*y-b)^2;
        fz = .5*norm(A*z-b)^2;
        
        inn = grad'*(y-z);
        ns = norm(y-z)^2/2;
        if fy <= fz+inn+L*ns
            i = 2;
            etc.obj(k) = fy;
            obj = fy;
            break; % break point of inner while loop
        else
            i=3;
        end
    end
    
    Gamma_cap = L*gamma^3;
    %     Gamma_cap = L*gamma^2;
    
    % Save runtime data
    etc.CPUTime(k) = toc(tStart);
    CPUtime = toc(tStart);
    
end

etc.Iterations = k;
etc.objective = fy;
etc.finalGap = gap;
etc.finalL = L;
etc.Outer_Time = toc(CGSt);

end
