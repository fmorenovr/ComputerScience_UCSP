function [eigenValues, eigenMatrix]=eigQR(A)
  Ai=A;
  n=length(A);
  errTol=1.0;
  while errTol>0.0000001
    [Q,R]=desQR(Ai);
    QR = Q*R;
    A0=Ai;
    Ai=R*Q;
    errTol=norm(diag(Ai)-diag(A0),1)/norm(diag(Ai),1);
  end
  eigenMatrix=diag(Ai);
  eigenValues=Ai;
end
