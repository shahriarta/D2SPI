function V = funBar(x)
    N = length(x);
    r = 1;
    V = zeros( N*(N+1)/2 , 1);
    for i = 1:N
        for j = i:N
            V(r) = x(i)*x(j);
            r = r+1;
        end
    end
end