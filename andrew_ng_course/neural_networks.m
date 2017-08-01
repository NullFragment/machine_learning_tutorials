1;
#===============================================================================
### START FUNCTION: oneVsAll                                                 ###
#===============================================================================
# Returns the result of one-vs-all logistic regression

function [all_theta] = oneVsAll(X, y, num_labels, lambda)
    m = size(X, 1);
    n = size(X, 2);

    all_theta = zeros(num_labels, n + 1);

    X = [ones(m, 1) X];

    for c = 1:num_labels
        initial_theta = zeros(n + 1, 1);
        options = optimset('GradObj', 'on', 'MaxIter', 50);
        [theta] = fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
                        initial_theta, options);
        all_theta(c,:) = theta';
    endfor

    c = zeros(m,1);

endfunction

#===============================================================================
### START FUNCTION: ovaPredict                                               ###
#===============================================================================
# Outputs the prediction of one-vs-all logistic model

function functionName (all_theta, X)
    m = size(X, 1);
    num_labels = size(all_theta, 1);
    p = zeros(size(X, 1), 1);
    
    X = [ones(m, 1) X];
    z = X*all_theta';
    h = sigmoid(z);
    [pr, p] = max(h, [], 2);
endfunction

#===============================================================================
### START FUNCTION: nnPredict                                                ###
#===============================================================================
# Predicts 3-layer neural network output given trained parameters

function nnPredict (Theta1, Theta2, X)
    m = size(X, 1);
    num_labels = size(Theta2, 1);
    p = zeros(size(X, 1), 1);
    X = [ones(m, 1) X];
    
    a2 = sigmoid(X*Theta1');
    a2 = [ones(size(a2,1), 1) a2];
    a3 = sigmoid(a2*Theta2');
    [pr, p] = max(a3, [], 2);
endfunction
