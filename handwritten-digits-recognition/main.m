clear;close all;clc;
input_layer = 784;  % 28x28 Input Images of Digits
hidden_layer = 100;   % 25 hidden units
labels = 10;          % 10 labels, from 1 to 10 
lambda = 1.0;
[X, T, Xtest, Ntrain, Ntest, TtestTrue] = loadData();
% normalize the pixels to take values in [0,1]
X = X/255; 
Xtest = Xtest/255; 
[N, D] = size(X);
% Add 1 as the first for both the training input and test inputs 
X = [ones(sum(Ntrain),1), X];
Xtest = [ones(sum(Ntest),1), Xtest]; 
% Maximum number of iterations of the gradient ascend
options(1) = 1000; 
% Tolerance 
options(2) = 1e-6; 
% Learning rate 
options(3) = 1.0/N; 

%%%
%Random initialization
%%%
initW1 = randInitializeWeights(input_layer, hidden_layer);
initW2 = randInitializeWeights(hidden_layer, labels);

%%%
%Gradient checking
%%%
prompt = 'Do you want to execute gradient checking?(y/n) ';
answer = input(prompt);
threshold = 1e-6;
if strcmp(answer,'y')
    ch = randperm(N); 
    ch = ch(1:10);
    [diff1, diff2] = gradientCheck(initW1, initW2, X(ch,:), T(ch,:), lambda);
    disp(['The maximum abolute norm in the gradcheck for W1 is ' num2str(max(diff1(:)))]);
    disp(['The maximum abolute norm in the gradcheck for W2 is ' num2str(max(diff2(:)))]);
    if (diff1 < threshold)
        if (diff2 < threshold)
            disp('Gradient check passed');
        end
    end
end 

%%%
% Train
%%%
[W1, W2] = train(initW1, initW2, X, T, lambda, options);

%%%
% Predict
%%%
[Ttest, Ytest] = predict(W1, W2, Xtest);
[~, Ttrue] = max(TtestTrue,[],2); 
err = length(find(Ttest~=Ttrue))/10000;
disp(['The error of the method is: ' num2str(err)])
