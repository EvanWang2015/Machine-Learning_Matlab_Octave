function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h = sigmoid(X*theta);
theta_re = theta(2:end);

% The cost function of regression is not same as the linear regression cost
%J = J + 1/(2*m)* (((h-y)'*(h-y)) + lambda * (theta_re'*theta_re));
J = J + sum((-y)'*log(sigmoid(X*theta)) + (y-1)' * log(1 - sigmoid(X*theta)))/m;
J = J + lambda/(2*m) * theta_re'*theta_re;

grad(1) = 1/m * X(:,1)'*(h-y);
grad(2:end) = 1/m *( X(:,2:end)'*(h-y) + lambda * theta_re);

% =============================================================

end
