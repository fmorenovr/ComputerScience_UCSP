% A is a square matrix
function [L,U] = desLU(A)
n = length(A);
AA = A;
I = eye(n);
MMi = I;
for i = 1:n-1
    mi = zeros(n,1);
    for k = 1+i:n
      mi(k) = -AA(k,i)/AA(i,i);
    end
    ei = I(:,i)
    Mi = I + mi*ei';
    Ai = Mi * AA;
    MMi = Mi * MMi;
    AA = Ai; 
end
U = MMi*A;
L = inv(MMi);
end
