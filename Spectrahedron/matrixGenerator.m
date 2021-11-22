%{
  This script generates a set of random matrices using function sprandrect
  and saves in given directory
%}

N = [100, 200, 300];
M = [500, 1000, 2000];
D = [.2, .6, .8];

for i = 1:length(N)
    for j = 1:length(M)
        for k = 1: length(D)
            rng(132)
            n = N(i);
            m = M(j);
            density = D(k);
            m_half1 = floor(m/2);
            m_half2 = m - m_half1;
            sv = [zeros(m_half1,1); rand(m_half2,1)];
            A = sprandrect(m,n^2, density, sv);
            
            idx = 100*i + 10*j + k;
            Matrix = ['Mat' num2str(idx)];
            filepath = '~\matrices'; % choose your file path
            save(fullfile(filepath, Matrix));
        end
    end
end
