clear all; close all; clc

% fixing the random seed
rng('default')
rand_var = 123;
rng(rand_var)

n   = 5; %state dimension
m   = 3; %input dimension

% Cost parameters
Q1 = eye(n);
Q2 = eye(n);
R  = eye(m);

% number of agents in the graph
max_num_of_agents = 10;
N = 10;

% Building the adjacency matrix for a path graph
Adj = zeros(N);
Adj(1,2)=1;
Adj(N,N-1)=1;
for i=2:N-1
    Adj(i,i-1)=1;
    Adj(i,i+1)=1;
end

Deg = diag(Adj*ones(N,1));
Lap   = Deg - Adj;
d = max(Adj*ones(N,1))+1;

% number of the random experiments
max_num_exp = 100;

% the random experiment data
P_data_poss = {};
P_data_act = {};

for exp=1:max_num_exp
    % Original continuous-time system parameters
    A = rand(n,n);
    B = rand(n,m);

    % discretization
    dt = 0.1;
    [A, B] = c2d(A, B, dt);

    % Optimal LQR controller for one agent
    [K_opt,P_opt] = dlqr(A,B,Q1,R);

    % Inserting uncertainty to the districized system while we assume the initial contoller is still stabilizing
    while(1)
        delta_A = 0.05*rand(n,n);
        A_new = A + delta_A;
        if max(abs(eig(A_new+B*(-K_opt))))<0.92
            A = A_new;
            break
        end
    end
    

    % The subgraph G_d to be involved in the learning process
    Q_t = Q1 + d*Q2;

    A_tilde  = kron(eye(d), A);
    B_tilde  = kron(eye(d), B);
    Q_tilde  = kron(eye(d), Q_t) - kron(ones(d), Q2);
    R_tilde  = kron(eye(d), R);

    [K_tilde_opt, P_tilde_opt] = dlqr(A_tilde, B_tilde, Q_tilde, R_tilde);


    % The entire graph G
    A_hat = kron(eye(N), A);
    B_hat = kron(eye(N), B);
    Q_hat = kron(eye(N), Q1) + kron(Lap, Q2);
    R_hat = kron(eye(N), R);

    [K_hat_opt_unconst, P_hat_opt_unconst] = dlqr(A_hat,B_hat,Q_hat,R_hat);


    % The residual graph G\G_d
    A_aux = kron(eye(N-d), A);
    B_aux = kron(eye(N-d), B);
    Q_aux = eye(n*(N-d));
    R_aux = eye(m*(N-d));

    [K_aux_opt, P_aux_opt] = dlqr(A_aux, B_aux, Q_aux, R_aux);

    % determining different matrix sizes
    SIZE    = size(B_tilde);
    n_tilde = SIZE(1);
    m_tilde = SIZE(2);
    l       = (n_tilde+m_tilde)*(n_tilde+m_tilde+1)/2;

    SIZE    = size(B_hat);
    n_hat = SIZE(1);
    m_hat = SIZE(2);


    % initial stabilizing controller
    K_tilde = -kron(eye(d), K_opt); 
    K_hat   = -kron(eye(N), K_opt); 

    if max(abs(eig(A_tilde+B_tilde*K_tilde)))>=1
        disp('K_0 is NOT stabilizing- reduce the uncertainty in the model or consider another initial stabilizing controller!')
        return
    end

    % The possible cost for the entire network before starting the algorithm
    P_hat_poss{1} = lqr_cost(A_hat,B_hat,Q_hat,R_hat,K_hat);
    P_hat_act{1} = P_hat_poss{1};


    % Initializing the system dynamics: x represents the states of G_d and y
    % represents the states of G\G_d
    x0 = 0.03*randn(n_tilde, 1);
    y0 = 0.03*randn(n_hat-n_tilde, 1);
    y0(1:n) = [0.15  0.035 -0.105 0.33 -0.02]';
    x = []; 
    y = [];
    x(:,1) = x0;
    y(:,1) = y0;


    % Initialization of the algorithm
    iteration_SPE  = 10000;
    iteration_main = 6;
    discount       = 1;
    err_tol_main   = 0.01;
    noise          = 300;
    beta           = 20;
    P0             = beta * eye(l);
    theta          = randn(l,1);
    t              = 1;

    for iter = 1:iteration_main
        iter

        %%%%%% The SPE subroutine %%%%%%%
        P = P0;
        for j = 1:iteration_SPE-1
            e = noise*randn(m_tilde,1);    
            u = K_tilde*x + e;
            x_new = A_tilde*x + B_tilde*u;

            zeta = funBar( [x ; u] ) - discount * funBar( [x_new ; K_tilde*x_new] );
            rt = x'*Q_tilde*x + u'*R_tilde*u;
            theta = theta + ( P * zeta * (rt - zeta'*theta) ) / ( 1 + zeta'*P*zeta );
            P = P - ( P * (zeta * zeta') * P ) / ( 1 + zeta'*P*zeta );

            x = x_new;
        end

        %%%%%%% Data matrix %%%%%%%%%%%
        H   = myvec2mat(theta);

        H11 = H(1:n_tilde, 1:n_tilde);
        H12 = H(1:n_tilde, n_tilde+1:end);
        H22 = H(n_tilde + 1:n_tilde + m_tilde, n_tilde + 1:n_tilde + m_tilde);
        H21 = H(n_tilde + 1:n_tilde + m_tilde, 1:n_tilde);

        %%%%%% recovering parameters essential for our controller design %%%%%%%

        X1 = H11(1:n, 1:n);       % = Q1 + (d-1)Q2 + A'*P1*A
        X2 = H11(1:n, n+1:2*n);   % = -Q2 + A'*P2*A
        dX = X1-X2;

        Y1 = H22(1:m, 1:m);       % = R + B'*P1*B
        Y2 = H22(1:m, m+1:2*m);   % = B'*P2*B
        dY = Y1-Y2;

        Z1 = H21(1:m, 1:n);       % = B'*P1*A
        Z2 = H21(1:m, n+1:2*n);   % = B'*P2*A
        dZ = Z1-Z2;

        %%%%% Computing individual (K) and cooperative (L) component of our controller %%%%%

        F_inv = (Y1 - (d-1)*Y2*((Y1 + (d-2)*Y2)\Y2));
        G = (Y1 + (d-1)*Y2)\(Y2/(Y1-Y2));

        K  = -F_inv\Z1 + (d-1)*G*Z2;
        L  = -F_inv\Z2 + G*Z1 + (d-2)*G*Z2;
        dK = K-L;

        %%%% Computing the stability margin (tau)
        Xi = dX - Q_t + dK'*dZ + dZ'*dK + dK'*(dY-R)*dK;
        gamma_k = min(svd(dK'*R*dK + Q_t)) / max(svd(Xi + L'*(dY-R)*L));
        tau_k = sqrt(gamma_k^2/(1+gamma_k));

        % checking Theorem 1 and Prop. 2
        if max(abs(eig(A_hat+B_hat*K_hat)))>=1
            disp('K_hat is NOT stabilizing at this iteration -VIOLATION of Theorem 1!')
        else
            disp('K_hat is stabilizing at this iteration -confirming Theorem 1-')
        end
        if max(abs(eig(A+B*(K-L))))>=1
            disp('K-L is NOT stabilizing at this iteration -VIOLATION of Prop. 2!')
        else
            disp('K-L is stabilizing at this iteration -confirmning Prop. 2-')
        end


        %%%%% Designing the controller for G\G_d   %%%%%%%%%%
        K_hat = (kron(eye(N), K) - kron(eye(N)-(tau_k/(d-1))*Adj, L));

        % The possible cost for the entire network WITHOUT temprory links
        P_hat_poss{iter+1} = lqr_cost(A_hat,B_hat,Q_hat,R_hat,K_hat);

        % Adjusting K_hat according to the algorithm
        K_hat(1:d*m, :) = zeros(d*m, N*n);
        K_hat(1:d*m, 1:d*n) = K_tilde;

        % The actual cost of the network WITH temprory links
        P_hat_act{iter+1} = lqr_cost(A_hat,B_hat,Q_hat,R_hat,K_hat);


        %%%%% Designing the controller for G_d   %%%%%%%%%%    
        K_tilde = kron(eye(d),K-L) + kron(ones(d),L);


        % checking Prop. 2
        if max(abs(eig(A_tilde+B_tilde*K_tilde)))>=1
            disp('K_tilde is NOT stabilizing yet -increase iterations of SPE subroutin and check the initial stabilizing conteroller.')
        else
            disp('K_tilde is stabilizing -confirming Prop. 2.')
        end   

        % checking Theorem 2
        K_tilde_cent = -H22\H21;
        if max(max(K_tilde - K_tilde_cent)) < 0.01
            disp('The structure of the controller for G_d_learn is as claimed- confirming Theorem 2.')
        else
            disp(['The structure of the controller for G_d_learn is NOT as claimed- VIOLATION of Theorem 2!----max entry error:',num2str(max(max(K_tilde - K_tilde_cent)))])
        end


%         %%%%%% instead of fixing the number of iterations, on could check
%                 the convergence of K_tilde as follows %%%%%
%         if norm(K_tilde+K_tilde_opt) < err_tol_main
%             break
%         end

    end

    % The final distributed policy
    K_hat_out = (kron(eye(N), K) - kron(eye(N)-(tau_k/(d-1))*Adj, L));
    
    P_data_poss{exp} = P_hat_poss;
    P_data_act{exp} = P_hat_act;
    P_data_lqr{exp} = P_hat_opt_unconst;
    K_data_lqr{exp} = K_hat_opt_unconst;
end

%% ploting the results
fig1 = figure;
hold on

trace_P_lqr = cellfun(@(P) trace(P), P_data_lqr);
normal_poss_cost = zeros(max_num_exp, iteration_main+1);
for exp = 1:max_num_exp
    normal_poss_cost(exp,:) = cellfun(@(P) (trace(P)- trace_P_lqr(exp))/trace_P_lqr(exp), P_data_poss{exp});
    p = plot(normal_poss_cost(exp,:),'LineWidth',2);
    p.Color(4) = 0.2;
    % normal_act_cost = cellfun(@(P)trace(P - P_hat_opt_unconst)/trace(P_hat_opt_unconst), P_hat_act);
    % plot(normal_act_cost,'LineWidth',2)
end 
p_1 = plot(max(normal_poss_cost),'-o','LineWidth',2,'Color','black','MarkerSize',8);
p_2 = plot(min(normal_poss_cost),'-*','LineWidth',2,'Color','black','MarkerSize',8);
p_3 = plot(mean(normal_poss_cost),'-^','LineWidth',2,'Color','black','MarkerSize',8);
set(gca, 'YScale', 'log','fontsize',17)
%     title('Normalized suboptimality of entire network w.r.t the (unfeasible) LQR ctr','fontsize',14)
ylabel('$\frac{\mathrm{tr}(\hat \mathbf{P}_k) - \mathrm{tr}(\hat \mathbf{P}_{LQR})}{\mathrm{tr}(\hat \mathbf{P}_{LQR})}$','Interpreter','latex','fontsize',25)
xlabel('Iteration (k)','fontsize',17)
legend([p_1, p_2, p_3],'max','min', 'mean','fontsize',15)


% saving files
saveas(fig1,'figures/Exp1-fig1','epsc')
clear fig1
Filename = sprintf('data/Exp1_%s', datestr(now,'mm-dd-yyyy-HH-MM'));
save(Filename)
