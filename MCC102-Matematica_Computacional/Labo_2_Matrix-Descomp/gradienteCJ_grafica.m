function X=gradienteCJ_grafica(A,x0,b)
n=2;%length(A);
r0=b-A*x0;p0=r0;
a0=(p0'*r0)/(p0'*A*p0);
ak=a0;pk=p0;xk=x0;rk=r0;


for k=1:n
    xk=xk+ak*pk;
    rk=rk-ak*A*pk;
    bk=-(pk'*A*rk)/(pk'*A*pk);
    pk=rk+bk*pk;
    ak=(pk'*rk)/(pk'*A*pk);
    xx(:,k)=xk;
    
end
h=norm(x0-xk);
xa=xk(1)-2*h:0.1:xk(1)+2*h;
ya=xk(2)-2*h:0.1:xk(2)+2*h;
[x,y]=meshgrid(xa,ya);
z=0.5*(A(1,1)*x.^2+2*A(1,2)*x.*y+A(2,2)*y.^2)-b(1)*x-b(2)*y;
surf(x,y,z)
camlight left
shading interp
alpha(.5)
colormap(winter)

hold on
contour(x,y,z)
plot(xx(1,:),xx(2,:),'or',xx(1,:),xx(2,:),'y')

hold off


X=xk;