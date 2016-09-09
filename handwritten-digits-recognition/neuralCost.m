function [J, Theta1_grad, Theta2_grad] = neuralCost(W1, W2, X, Y, lambda)
% Forward Propagation
z2 = X * W1';
a2 = [ones(size(z2, 1), 1) activationFunction(z2)];
z3 = a2 * W2';
a3 = softmax(z3);
% Cost function
M = max(z3, [], 2);
J = sum(sum( Y.*z3 )) - sum(M) - sum(log(sum(exp(z3 - repmat(M, 1, size(W2,1))), 2)))  - (0.5*lambda)*sum(sum(W2.*W2));
%Backpropagation
if nargout > 1 
    Theta2_grad = (Y - a3)' * a2 - lambda * W2;
    Theta1_grad = W2(:,2:end)' * (Y - a3)' .* activationFunctionGradient(z2)' * X - lambda * W1;
end