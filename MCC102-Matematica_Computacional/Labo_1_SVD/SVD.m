function [U,S,V] = SVD(A)
  [m, n] = size(A);
  S = zeros(m,n);
  U = zeros(m,m);
  V = zeros(n,n);
  r = min(m,n);

  simetric_A = transpose(A)*A;
  [vectors, LAMBDAS] = eig(simetric_A);
  [vectors, LAMBDAS] = sort_eig(vectors,LAMBDAS);
  S(1:r,1:r) = sqrt(LAMBDAS(1:r,1:r));
  V(:,1:r) = vectors(:,1:r);
  for i=1:r
    if(S(i,i)!=0)
      U(:,i) = A*V(:,i)/S(i,i);
    endif
  endfor
endfunction
