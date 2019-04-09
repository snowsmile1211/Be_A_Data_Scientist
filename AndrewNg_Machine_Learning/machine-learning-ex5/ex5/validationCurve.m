function [lambda_vec, error_train, error_val] = ...
    validationCurve(X, y, Xval, yval)
%VALIDATIONCURVE Generate the train and validation errors needed to
%plot a validation curve that we can use to select lambda
%   [lambda_vec, error_train, error_val] = ...
%       VALIDATIONCURVE(X, y, Xval, yval) returns the train
%       and validation errors (in error_train, error_val)
%       for different values of lambda. You are given the training set (X,
%       y) and validation set (Xval, yval).
%

% Selected values of lambda (you should not change this)
lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';

% Number of training examples
m = size(X, 1);
% Number of validation examples(need to compute validation error)
k =size(Xval,1);

% You need to return these variables correctly.
error_train = zeros(length(lambda_vec), 1);
error_val = zeros(length(lambda_vec), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return training errors in 
%               error_train and the validation errors in error_val. The 
%               vector lambda_vec contains the different lambda parameters 
%               to use for each calculation of the errors, i.e, 
%               error_train(i), and error_val(i) should give 
%               you the errors obtained after training with 
%               lambda = lambda_vec(i)
%
% Note: You can loop over lambda_vec with the following:
%
%       for i = 1:length(lambda_vec)
%           lambda = lambda_vec(i);
%           % Compute train / val errors when training linear 
%           % regression with regularization parameter lambda
%           % You should store the result in error_train(i)
%           % and error_val(i)
%           ....
%           
%       end
%
%

 for i = 1:length(lambda_vec)
     lambda = lambda_vec(i);
     
    theta = trainLinearReg(X, y, lambda);
   
    %trainning error(pay attention to the size of X sample size, here, not m, but i, changing)
    % caculate y_bar using parameters
    y_bar = X*theta;
    %error between y_bar and y
    error_square = sum((y_bar-y).^2);
    %training error
    error_train(i) = (1/(2*m))*error_square;

    %cross-validition error (pay attention to the size of Xval size, here, not m, but k)
    % caculate y_bar using parameters
    y_bar_val = Xval*theta;
    %error between y_bar and y
    error_square_val = sum((y_bar_val-yval).^2);
    %cross_validation error
    error_val(i) = (1/(2*k))*error_square_val;

 end







% =========================================================================

end
