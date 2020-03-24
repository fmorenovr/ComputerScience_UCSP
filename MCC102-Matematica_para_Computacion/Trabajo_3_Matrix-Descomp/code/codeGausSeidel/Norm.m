function norm=Norm(u,nx,ny)
  t = 0;
  for i=2:nx-1
    for j=2:ny-1
      t = t + (u(i,j)*u(i,j));
    end 
  end 
  norm = sqrt(t);
end 
