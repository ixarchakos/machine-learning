function [cluster, gamma, m, sigma, p] = em(K, X)
[N , D] = size(X);
gamma = zeros(N, K);
%%
% Initialize em parameters

p(1, 1:K) = 1/K;
m = X(randi(N, 1, K),:);
sigma = repmat(0.1 * var(X), K, 1);
likelihood = -realmax;
max_iterations = 50;
tol = 0.0001;
%%
% em iteration, if the algorithm converge the iteration breaks
for count = 1 : max_iterations
    %%
    % expectation step
    
    s = zeros(N, K);
    for k = 1 : K
        for d = 1 : D
            s(:, k) = s(:, k) + repmat(log(sqrt(2*pi*sigma(k,d))), N, 1)... 
            +(((X(:, d) - repmat(m(k, d), N, 1)).^2 ) ./ (2*sigma(k, d)));
        end
        s(:, k) = log(p(1, k)) - s(:, k);
    end
    gamma = softmax(s);
    %%
    % maximization step
    
    gamma_sums = sum(gamma);
    for k = 1 : K
        for d = 1 : D
            m(k,d) = (gamma(:, k)' * X(:, d)) / gamma_sums(1,k);
            sigma(k,d) = (gamma(1:N, k)' * (X(1:N,d) - repmat(m(k,d) , N, 1)).^2 )/gamma_sums(1,k);           
        end   
    end
    p = gamma_sums / N;
    %sigma array must not have zero values
    sigma(sigma < 1e-06) = 1e-06;
    
    %%
    % calculate maximum likelihood
    
    [N, D] = size(X);
    s = zeros(N, K);
    for k = 1 : K
        for d = 1 : D
            s(:, k) = s(:, k) + repmat(log(sqrt(2*pi*sigma(k,d))), N, 1)... 
            +(((X(:, d) - repmat(m(k, d), N, 1)).^2 ) ./ (2*sigma(k, d)));
        end
        s(:, k) = log(p(1, k)) - s(:, k);
    end
    maxF = max(s, [], 2);
    s = exp(s - repmat(maxF, 1, K));
    likelihoodNew = sum(maxF + log(sum(s, 2)));
    %%
    % check for convergence
    
    if(likelihoodNew < likelihood)
        fprintf('Error found! Stop the e-m algorithm!');
        break;
    else
        if (likelihoodNew - likelihood) < tol
            break;
        else
            likelihood = likelihoodNew;
            fprintf('likelihood = %d\n', likelihood);
        end
    end
end
%%
% assign each data to the closest cluster

[~ , maxCluster] = max(gamma , [] , 2);
cluster = m(maxCluster( : , 1) , :);
end