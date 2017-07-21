1;
#===============================================================================
### START FUNCTION: name                                                     ###
#===============================================================================
# Description

function functionName ()
    printf();
endfunction

#===============================================================================
### START FUNCTION: sigmoid                                                  ###
#===============================================================================
# Computes the sigmoid function for a given X and theta

function [g] = sigmoid (X, theta)
    z = X*theta;
    g = 1./(1+exp(-z);
endfunction

#===============================================================================
### START FUNCTION: log_cost                                                 ###
#===============================================================================
# Computes the logistic regression cost function

function [J] = log_cost(X, y, theta)
    m = length(y);
    h = sigmoid(X, theta);
    pos = -transpose(y)*log(h);
    neg = -transpose(1-y)*log(1-h);
    J = 1/m * (pos + neg);
endfunction

#===============================================================================
### START FUNCTION: grad_log_cost                                            ###
#===============================================================================
# Computes the gradient of the logistic regression cost function

function [del] = grad_log_cost(X, y, theta)
    m = length(y)
    h = g(X, theta);
    err = (h - y);
    del = 1/(m) * (X' * err);
endfunction

#===============================================================================
### START FUNCTION: gradDescLog                                              ###
#===============================================================================
# Calculates the gradient descent using logistic regression

function [theta, J_history] = gradDescLog (X, Y, Theta, alpha, iterations)
    J_history = zeros(iterations, 1);
    for iter = 1:iterations
        theta = Theta - alpha*grad_log_cost(X, Y, Theta);
        J_history(iter) = log_cost(X, Y, Theta);
    endfor
endfunction

#===============================================================================
### START FUNCTION: log_all_cost                                             ###
#===============================================================================
# Returns the cost and gradient of the cost function

function [J, grad_J] = log_all_cost()
    J = log_cost();
    grad_J = grad_log_cost();
endfunction

#===============================================================================
### START FUNCTION: optimized_descent                                        ###
#===============================================================================
# Uses the fminunc function to find optimized theta values

function optimized_descent (X, y)
    options = optimset('GradObj', 'on', 'MaxIter', 100);
    initialTheta = zeros(2,1);
    [optTheta, functionVal, exitFlag] = ...
        fminunc(@(theta)(log_all_cost(X,y,theta)), initialTheta, options);
endfunction
