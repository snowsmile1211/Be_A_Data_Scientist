function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
temp_theta=zeros(size(theta,1),1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    
%     temp_theta0 = theta(1,1)-alpha*(1/m)*sum((X*theta-y).*X(:,1));
%     temp_theta1 = theta(2,1)-alpha*(1/m)*sum((X*theta-y).*X(:,2));
%     
%     theta(1,1)=temp_theta0;
%     theta(2,1)=temp_theta1;
    
    for i = 1:size(theta,1)
    temp_theta(i,1) = theta(i,1)-alpha*(1/m)*sum((X*theta-y).*X(:,i));
    end
    
    theta=temp_theta;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
