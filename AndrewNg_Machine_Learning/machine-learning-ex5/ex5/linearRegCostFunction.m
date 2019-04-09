function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

%X=[ones(m, 1) X];

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%compute J
% caculate y_bar using parameters
y_bar = X*theta;
%error between y_bar and y
error_square = sum((y_bar-y).^2);
%regulation term
reg = theta(2:end)'*theta(2:end);
%cost function
J = (1/(2*m))*error_square+(lambda/(2*m))*reg;

%compute grad
error = y_bar-y;
grad = (1/m)*X'*error+(lambda/m)*theta;
grad(1)=(1/m)*X(:,1)'*error;
















% =========================================================================

grad = grad(:);

end
