function [X, Bsor] = SOR(A,x0,b,error, w)
  D = diag(diag(A));
  L = tril(A) - D;
  U = L';
  
  T  = -inv(L+D)*U;
  [Vu Du] = eig(T);
  pT = max(abs(diag(Du)));
%  w = 2 / (1+(1-pT^2)^0.5);
  
  Bsor = inv(D + w.*L)*((1-w).*D - w.*U)
  bsor = w.*inv(D+w.*L)*b;
  
  xk = x0;
  i = 1;
  
  while (i < 100)
    xnew = Bsor*xk + bsor;
    if norm(xnew-xk)/norm(xk) < error
      break
    end
    i = i + 1;
    for k = 1:length(A)
      fprintf('x%g = %f ', k, xnew(k));
    end
    fprintf('error = %f\n', norm(xnew-xk)/norm(xk));
    xk = xnew;
  end
  X = xnew;
end

function x=tridiag(A,b)
  n = length(A);
  D = diag(diag(A));
  L = tril(A) - D;
  U = L';
  U = U + D;
  L = L + eye(n);
  x = zeros(n,1);
  c = zeros(n,1);
  l = ones(n-1,1);
  u = ones(n-1,1);
  d = diag(D);
  c(1) = b(1);
  for i = 2:n
    c(i) = b(i) - l(i-1)*c(i-1);
  end
  for i = 1:n-1
    x(i) = (c(i) - u(i)*x(i+1))/d(i);
  end
end

