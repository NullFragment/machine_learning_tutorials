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
    g = 1./(1+exp(-z));
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
    h = sigmoid(X, theta);
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
### START FUNCTION: log_cost_reg                                             ###
#===============================================================================
# Description

function log_cost_reg (X, y, theta, lambda)
    m = length(y);                          # number of training examples
    L = eye(length(theta));                 # Initialize L for regularization
    L(1,1) = 0;                             # Prevent regularization of Theta(0)
    z = X*theta;                            # Sigmoid input
    h = sigmoid(z);                         # Compute hypothesis probability
    reg_param = (lambda/(2*m))              # Regularization parameter
    reg_theta = (L*theta)'*(L*theta);       # Theta^2 without Theta(0)
    reg_val = reg_param*reg_theta;          # Regularization values
    
    pos = transpose(y)*log(h);              # Probability of 1
    neg = transpose(1-y)*log(1-h);          # Probability of 0
    J = -1/m * (pos + neg)+(reg_val);       # Regularized Cost

endfunction

#===============================================================================
### START FUNCTION: grad_log_cost_reg                                        ###
#===============================================================================
# Description

function grad_log_cost_reg (X,y,theta)
    m = length(y);                          # number of training examples
    L = eye(length(theta));                 # Initialize L for regularization
    L(1,1) = 0;                             # Prevent regularization of Theta(0)
    z = X*theta;                            # Compute sigmoid input
    h = sigmoid(z);                         # Compute hypothesis probability

    reg_val = (lambda)*L*theta;             # Create regularized value
    err = (h - y);                          # Compute error of hypothesis
    grad = 1/(m) * (X' * err + reg_val)     # Compute gradient of hypothesis
endfunction

