%{
Performing algorithms:
    - Conditional Gradient (CG)
    - Conditional Gradient Sliding (CGS)
    - Conditional Gradient Sliding with Linsearch (CGS-ls) 
for solving min {.5||Ax-b||^2; x \in Spe} where 
- A is a random rectangular matrix and objective function is not strongly
  convex. 
- Feasible region is the standard spectrahedron {X \in R^nxn : Tr(X)=1, X>=0}
- set X in this script is the set of all vertices of the feasible region in the
  paper "Backtracking linesearch for conditional gradient sliding"
  
author: 
  Hamid Nazari - snazari@ clemson.edu

Notes:
  Matrices A are loaded for this script. As are generated in matrixGenerator.m
%}

tStart = tic;

rng(132)

for i = 1:2
    for j = 1:3
        for k = 1:3
            idx = 100*i + 10*j + k;
            file = sprintf('matrices/Mat%d.mat',idx);
            load(file);
            
            A = double(A);
            par.L = 1.5e4;
            sval = svds(A,1);
            A = (A*par.L)/(sval^2);
            
            diag_xtrue = rand(n, 1);
            diag_xtrue = diag_xtrue/sum(diag_xtrue);
            tmp = rand(n, n);
            [Q, ~] = qr(tmp'*tmp);
            xtrue = Q*diag(diag_xtrue)*Q';
            xtrue = xtrue(:);
            b = A*xtrue;
            
            par.L0 = .001*par.L;
            par.diam = sqrt(2);
            par.MaxIter = Inf;
            par.Tol = 1.00e-02;
            par.MaxInnerIter = Inf;
            par.MaxCPUtime = 30*60;
            par.c=.05;
            
            m, n , density
            [y_CG, CG]= funCG_Spec(A, b, par);
            CG
            [y_CGS, CGS]= funCGS_Spec(A, b, par);
            CGS
            [y_CGSls, CGSls] = funCGS_ls_Spec(A, b, par);
            CGSls
            % save(['all_NSC' num2str(idx)])
        end
    end
end
