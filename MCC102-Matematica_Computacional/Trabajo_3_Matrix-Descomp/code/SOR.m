function [X, Bsor] = SOR(A,x0,b,error)
    D = diag(diag(A));
    L = tril(A) - D
    U = L'
    
    Bj = - inv(D)*(L + U);
    [Vu Du] = eig(Bj);
    pBj = max(abs(diag(Du)))
    w = 2 / (1+(1-pBj^2)^0.5);
    
    Bsor = inv(D + w.*L)*((1-w).*D - w.*U)
    bsor = w.*inv(D+w.*L)*b;
    
    xk = x0;
    i = 1
    
    while (i < 100)
      xnew = Bsor*xk + bsor;
      if norm(xnew-xk)/norm(xk) < error
        break
      end
      xk = xnew;
      i = i + 1
    end
    X = xnew;
end