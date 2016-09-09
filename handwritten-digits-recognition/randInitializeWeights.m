function W = randInitializeWeights(first, second)
epsilon = 0.1;
W = rand(second, 1+first) * 2 * epsilon - epsilon;
end
