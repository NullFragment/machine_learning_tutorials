1;
#===============================================================================
### START FUNCTION: name                                                     ###
#===============================================================================
# Description

function functionName ()
    printf();
endfunction

#===============================================================================
### START FUNCTION: meanNorm                                                 ###
#===============================================================================
# This function computes the mean normalization of an array

function [norm] = meanNorm (features)
    norm = (features - mean(features))./range(features);
endfunction

#===============================================================================
### START FUNCTION: componentMSE                                             ###
#===============================================================================
# Computes component-wise Mean Squared Error

function [J] = componentMSE (x, y, theta)
    m = size(x,1);                    # number of training examples
    predictions = x*theta;          # predicted values
    errors = (predictions - y).^2;  # squared errors
    J = 1/(2*m) * sum(errors);      # mean squared errors
endfunction

#===============================================================================
### START FUNCTION: vectorMSE                                                ###
#===============================================================================
# Computes the vectorized Mean Squared Error

function [J] = vectorMSE (X, Y, Theta)
    m = rows(X);            # number of training samples
    rhs = X*Theta - Y;      # error matrix
    lhs = rhs';             # inverse error matrix
    sqError = lhs * rhs;    # squared error matrix
    J = 1/(2*m) * sqError;  # mean squared error
endfunction

#===============================================================================
### START FUNCTION: gradvMSE                                                 ###
#===============================================================================
# Computes the vectorized gradient of the Mean Squared Error

function [del] = gradvMSE (X, Y, Theta)
    m = rows(X);            # number of training samples
    err = X*Theta - Y;      # error matrix
    del = 1/(m) * (X' * err);  # mean squared error
endfunction

#===============================================================================
### START FUNCTION: gradvDescMSE                                             ###
#===============================================================================
# Computes the vectorized gradient descent for one step using MSE

function [theta] = gradvDescMSE (X, Y, Theta, alpha, iterations)
    J_history = zeros(iterations, 1);
    for iter = 1:iterations
        theta = Theta - alpha*gradvMSE(X, Y, Theta);
        J_history(iter) = vectorMSE(X, Y, Theta);
endfunction

#===============================================================================
### START FUNCTION: normalEqn                                                ###
#===============================================================================
# Description

function [theta] = normalEqn (X, Y)
    theta = zeros(size(X, 2), 1);
    theta = pinv(X'*X) * X' * Y;
endfunction
