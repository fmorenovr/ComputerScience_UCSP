#!/usr/bin/env octave

function  y=jacobi(A,b,x,tol)
  n     = length(A);
  error = 1;
  while error > tol
    xx = x;
    for i = 1:n
      s = 0;
      for j = 1:n
        if i != j 
          s = s + A(i,j)*xx(j);
        end
      end
      x(i) = (b(i)-s)/A(i,i);
    end
    error = norm(x - xx);
    for k = 1:n
      fprintf('x%g = %f ', k, xx(k));
    end
    fprintf('error = %f\n', error);
  end
  y = x;
end

function  y=gseidel(A,b,x,tol)
  n     = length(A);
  error = 1;
  while error > tol
    xx = x;
    for i = 1:n
      s = 0;
      for j = 1:n
        if i != j 
          s = s + A(i,j)*x(j);
        end
      end
      x(i) = (b(i)-s)/A(i,i);
    end
    error = norm(x - xx);
    for k = 1:n
      fprintf('x%g = %f ', k, xx(k));
    end
    fprintf('error = %f\n', error);
  end
  y = x;
end
