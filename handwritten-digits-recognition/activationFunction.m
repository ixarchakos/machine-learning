function g = activationFunction(z)
m = max(0,z);
g = m +  log( exp(-m) + exp(z-m));  
end
