A=magic(10)
x=ones(10,1)
v = zeros(10, 1);
for i = 1:10
  for j = 1:10
    v(i) = v(i) + A(i, j) * x(j);
  end
end


v=2*ones(7,1)
w=ones(7,1)
z = 0;
for i = 1:7
  z = z + v(i) * w(i)
end

z=v*w'

z=A^2