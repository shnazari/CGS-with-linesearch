%{
  this function constructs a rectangular sparse random matrix of order 
  m-by-n (m < n), with prescribed density and singular values
    
  inputs:
    m - number of rows
    n - number of columns
    density - proportion of non-zero elements
    sv - vector of singular values
    
  outputs:
    A - Random matrix of size mxn 
  
  author:
    Hamid Nazari - snazari@clemson.edu
  
%}

function A = sprandrect(m,n,density,sv)
if m > n
    error('Input m must be less than n!\n');
end
if any(sv) < 0
    error('Input singular values must be non-negative!\n');
end
if length(sv) ~= m
    error('Input singular values count not equal to m!\n');
end
if density <= 0 || density > 1
    error('Input density must be in (0,1]!\n');
end

p = randperm(n);
P = speye(n);   P = P(:,p);     Q = sparse(n,m);
fac = n/m;      idxP = 0;       idxQ = 1;
while idxQ <= m
    newidxP = round(idxQ*fac);
    Q(:,idxQ) = P(:,idxP+1:newidxP)*(ones(newidxP-idxP,1)/sqrt(newidxP-idxP));
    idxQ = idxQ+1;
    idxP = newidxP;
end

R = sprandsym(m,density,sv);

A = R'*Q';
