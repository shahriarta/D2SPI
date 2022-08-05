function P = lqr_cost(A,B,Q,R,K)
    P = dlyap((A+B*K)',K'*R*K + Q);
end
