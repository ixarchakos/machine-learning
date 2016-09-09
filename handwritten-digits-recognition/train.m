function [W1, W2] = train(initW1, initW2, X, y, lambda, options)

iter = options(1); 
tol = options(2);
eta = options(3);
Ewold = -Inf;
for it=1:iter
    [Ew, grad1, grad2] = neuralCost(initW1,initW2,X,y,lambda);
    fprintf('Iteration: %d, Cost function: %f\n',it, Ew); 
    if abs(Ew - Ewold) < tol 
        break;
    end
    initW1 = initW1 + eta*grad1;
    initW2 = initW2 + eta*grad2;
    Ewold = Ew; 
end
W1 = initW1;
W2 = initW2;
end