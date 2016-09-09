function [Ttest, a3] = predict(W1, W2, X)
% Forward Propagation
z2 = X * W1';
a2 = [ones(size(z2, 1), 1) activationFunction(z2)];
z3 = a2 * W2';
a3 = softmax(z3);
[~,Ttest] = max(a3,[],2);
end
