function [autoValor,autoVetor]=potenciaINVGERSCH(A,y0,e,tipo)
n=length(A);R=zeros(1,n);r=zeros(1,n);s=r;
E=e+1;a0=norm(y0,tipo);
%%%%%%%%%%Gerschgorim%%%%%%%%%%%%radios
subplot(1,2,1)
for i=1:n
%s=0;
    for j=1:n
      if i~=j s(i)=s(i)+abs(A(i,j));
      end
    end
      R(i)=A(i,i)+s(i);
      r(i)=A(i,i)-s(i);
end
LI=min(r);
LS=max(R);

plot(LI,0,'*r',LS,0,'*g'),grid%%%%%%% grafica limites
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%circulos GERSCH
hold on
for i=1:n
t=0:0.01:2*pi+0.1;
xc=A(i,i)*ones(size(t))+s(i)*cos(t);
yc=s(i)*sin(t);
plot(xc,yc)%,grid
axis equal
end

hold off
%%%%%%%%%%%%%%%%%%%%%%%%grafico de autovalor inicial

subplot(1,2,2)
plot(a0,0,'*m')
grid
hold on
axis equal
%%%%%%%%%%%%%%%%%%%%%%%%%
INVA=inv(A);
z=INVA*y0;
a=norm(z,tipo);%a=max(abs(z));
k=0;
%%%%%%%%%%%%%%%%valores iniciales
   fprintf('|a(%2.0f)| = %5.4f   error =  \n',0,a0)
   fprintf('   z(0)=')
   for m=1:n
      fprintf(' %5.4f',z(m)) 
   end
   fprintf('\n   y(0)=')
   for m=1:n
      fprintf(' %5.4f',y0(m)) 
   end
   
   fprintf('\n__________________________________________________________\n')
%%%%%%%%%%%%%%%%    


%[L,U,P]=lu(A);Linv=inv(L);Uinv=inv(U);
while E>e
    k=k+1;
    y=(1/a)*z;
    %%%%%%%%%%%%% DESC LU
    %W=L\(P*y);%W=Linv*(P*y);
    %z1=U\W;%z1=Uinv*W;
    %%%%%%%%%%%%%%%%%%%%%
    z1=INVA*y;
    %pause(0.8)
    %z1=A*y;
    ak=norm(z1,tipo);%ak=max(abs(z1));
    E=norm(ak-a)/norm(ak);
    z=z1;
    a=ak;
   fprintf('|a(%2.0f)| = %5.4f   error = %5.4f\n',k,ak,E)
   fprintf('   z(%2.0f)=',k)
   for m=1:n
      fprintf(' %5.4f',z(m)) 
   end
   fprintf('\n   y(%2.0f)=',k)
   for m=1:n
      fprintf(' %5.4f',y(m)) 
   end
   
   fprintf('\n__________________________________________________________\n')
%%%%%%%%%%%%%%%%    
    
    subplot(1,2,2)
    plot(ak,0,'ob')
    axis equal
%%%%%%%%%%%%%%  sequencia de autovalores
end
hold off
autoValor=ak;autoVetor=y;%(1/norm(z1,inf))*z1;
end
