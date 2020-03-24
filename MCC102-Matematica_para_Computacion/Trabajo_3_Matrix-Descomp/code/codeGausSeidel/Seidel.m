% procedimiento seidel
function U = Seidel(Ax, Ay, Nx, Ny, h, Itmax, U)
%i, j, k;
U = zeros(Nx,Ny);
for k = 1:Itmax
	for j = 2:Ny-1
		y = Ay + j*h;
		for i = 2:Nx-1
			x = Ax + i*h;
			v = U(i+1,j) + U(i-1,j) + U(i,j+1) + U(i,j-1);
			U(i,j) = (v - (h*h)*g(x,y))/(4-(h*h)*f(x, y));
		end
	end
end

