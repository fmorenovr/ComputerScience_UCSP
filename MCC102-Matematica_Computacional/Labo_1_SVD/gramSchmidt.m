#!/usr/bin/env octave

function [Q R]=gramSchmidt(A)
  [m n]=size(A);
  Q=zeros(m,n);
  R=zeros(n);
  R(1,1)=norm(A(:,1));
  Q(:,1)=A(:,1)/R(1,1);
  for j=2:n
    Q(:,j)=A(:,j);
    for i=1:j-1
      Q(:,j)=Q(:,j)-A(:,j)'*Q(:,i)*Q(:,i);
      R(i,j)=A(:,j)'*Q(:,i);
    end
    R(j,j)=norm(Q(:,j));
    Q(:,j)=Q(:,j)/norm(Q(:,j));
  end
end
