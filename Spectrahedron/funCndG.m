%{
  Conditional Gradient (CndG) function of CG, CGS, and CGS-ls
  Algorithms inner iterations
  
  inputs:
    grad - vector of the gradient value of the objective
    u - updated point from preious iteration of outer iterations of CGS
    beta - parameter beta of the CGS algorithm
    eta - parameter eta of the CGS algorithm
    MaxCPUtime - time limit for the function
  
  outputs:
    ut - updated point from inner iterations
    count - umber of inner iteations of CGS
    time - time laps of function
      
  author:
    Hamid Nazari - snazari@clemson.edu
%}

function  [ut , count, time] = funCndG(grad ,u, beta, eta,MaxCPUtime)
ut = u;
threshold = Inf;
count = 0;
n2 = size(grad,1);
n = sqrt(n2);
Cndt = tic;
CPUtime =0;

while threshold > eta%CPUtime< (MaxCPUtime/100) && count<1 
    
    count = count+1;
    r = (grad + beta*(ut-u));
    
    rMat = reshape(r, n,n);
    sym_r = .5*(rMat + rMat');
    [evec, ~] = eigs(sym_r, 1, 'smallestreal');
    vMat = evec*evec';
    v = reshape(vMat, n2,1);
    
    threshold = r' * (ut-v);
%     inn = ((beta*(u-ut)-grad)'*(v-ut)) / (beta*(norm(v-ut)^2));
    inn = threshold/(beta*(norm(v-ut)^2));
    alpha = min(1, inn);
%     alpha = 2/(count+1);
    ut = ((1-alpha)*ut) + (alpha*v);
    CPUtime = toc(Cndt);
    
end
time = toc(Cndt);
