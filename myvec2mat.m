function THETA = myvec2mat(X)
    N = floor( sqrt( 2*length(X) ) );
    r = 1;
    THETA = zeros(N,N);
    for i = 1:N
        THETA(i,i) = X(r);
        r = r+1;
        for j = i+1:N
            THETA(i,j) = X(r)/2;
            THETA(j,i) = X(r)/2;
            r = r + 1;
        end
    end
end