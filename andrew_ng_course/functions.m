1;

################################################################################
### START FUNCTION: name                                                     ###
################################################################################
# Description

function functionName ()
    printf();
endfunction


################################################################################
### START FUNCTION: meanNorm                                                 ###
################################################################################
# This function computes the mean normalization of an array

function [norm] = meanNorm (features)
    norm = (features - mean(features))./range(features);
endfunction

################################################################################
### START FUNCTION: componentMSE                                             ###
################################################################################
# Computes component-wise Mean Squared Error Function

function [J] = componentMSE (x, y, theta)
    m = size(x,1);                    # number of training examples
    predictions = x*theta;          # predicted values
    errors = (predictions - y).^2;  # squared errors
    J = 1/(2*m) * sum(errors);      # mean squared errors
endfunction

################################################################################
### START FUNCTION: vectorMSE                                                ###
################################################################################
# Computes the vectorized Mean Squared Error Function

function [J] = vectorMSE (X, Y, Theta)
    m = rows(X);            # number of training samples
    rhs = X*Theta - Y;      # error matrix
    lhs = rhs';             # inverse error matrix
    sqError = lhs * rhs;    # squared error matrix
    J = 1/(2*m) * sqError;  # mean squared error
endfunction
