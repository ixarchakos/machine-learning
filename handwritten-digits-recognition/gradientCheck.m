function [diff1, diff2] = gradientCheck(W1,W2,X,T,lambda) 
[K1,D1] = size(W1);
[K2,D2] = size(W2);
% Compute the analytic gradient 
[Ew, gradEw1, gradEw2] = neuralCost(W1,W2,X,T,lambda);
epsilon = 1e-6; 

% numerical gradient for W1
numgradEw1 = zeros(K1,D1); 
for k=1:K1
    for d=1:D1
        Wtmp = W1; 
        Wtmp(k,d) = Wtmp(k,d) + epsilon;
        [Ewplus] = neuralCost(Wtmp,W2,X,T,lambda);
        Wtmp = W1; 
        Wtmp(k,d) = Wtmp(k,d) - epsilon; 
        [Ewminus] = neuralCost(Wtmp,W2,X,T,lambda);
        numgradEw1(k,d) = (Ewplus - Ewminus)/(2*epsilon);
    end
end

% Save the absolute norm as an indication of how close 
% the numerical gradients are to the analytic gradients
diff1 = abs(gradEw1 - numgradEw1);  

% numerical gradient for W2
numgradEw2 = zeros(K2,D2); 
for k=1:K2
    for d=1:D2
        Wtmp = W2; 
        Wtmp(k,d) = Wtmp(k,d) + epsilon;
        [Ewplus] = neuralCost(W1,Wtmp,X,T,lambda);
        Wtmp = W2; 
        Wtmp(k,d) = Wtmp(k,d) - epsilon; 
        [Ewminus] = neuralCost(W1,Wtmp,X,T,lambda);
        numgradEw2(k,d) = (Ewplus - Ewminus)/(2*epsilon);
    end
end

% Save the absolute norm as an indication of how close 
% the numerical gradients are to the analytic gradients
diff2 = abs(gradEw2 - numgradEw2);