function [l,V,Vn]=metodoPOTENCIA(A,y0,e)
E=e+1;i=0;
z=A*y0,a=norm(z,inf)
y=(1/a)*z;
aa=z./y;
while E>e
    %y=(1/a)*z
    y=z
    z1=A*y
    ak=norm(z1,inf)
    aak=z1./y;EE=abs(aak-aa)./abs(aak);
    E=norm(EE,inf);
    z=z1;
    aa=aak;
    a=ak;
        
end

l=a;V=z;Vn=y;

sigma = 4.2

AA= A - sigma*eye(3);

potenciaINVGERSCH(AA,y0,e,inf);
