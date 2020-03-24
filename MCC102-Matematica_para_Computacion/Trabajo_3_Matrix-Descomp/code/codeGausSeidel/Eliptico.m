function u = Eliptico()
  #integer
  #i,j;
  n_x = 9;
  n_y = 9;
  itmax = 75;
  #real
  #h,x,y;
  a_x = 0.0;
  b_x = 1.0;
  a_y = 0.0;
  b_y = 1.0;
  h = (b_x-a_x)/n_x;
  #real array
  u = zeros(n_x,n_y);
  
  for j = 1:n_y
    y = a_y + j*h;
    u(1,j) = Bndy(a_x, y);
    u(n_x, j) = Bndy(b_x, y);
   end
   for i = 1:n_x
    x = a_x + i*h;
    u(i,1) = Bndy(x, a_y);
    u(i, n_y) = Bndy(x, b_y);
   end
  for j = 2:n_y-1
    y = a_y + j*h;
    for i = 2:n_x-1
      x = a_x + i*h;
      u(i,j) = ustart(x,y);
    end  
  end
  #output
  u = Seidel(a_x, a_y, n_x, n_y, h, itmax, u);
  #printf("%d", u);
  #output
  for j = 1: n_y
    y = a_y + j * h;
    for i = 1:n_x
      x = a_x + i * h;
      u(i,j) = abs(TrueSolution(x, y) - u(i,j));
    end
  end