function X=gradienteCJ(A,x0,b)
n=length(A);
r0=b-A*x0;p0=r0;
a0=(p0'*r0)/(p0'*A*p0);
ak=a0;pk=p0;xk=x0;rk=r0;
for k=1:n
    xk=xk+ak*pk;
    rk=rk-ak*A*pk;
    bk=-(pk'*A*rk)/(pk'*A*pk);
    pk=rk+bk*pk;
    ak=(pk'*rk)/(pk'*A*pk);
end
X=xk;