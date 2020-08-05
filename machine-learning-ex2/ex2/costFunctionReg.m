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

h_theta = sigmoid(theta' * X');% dimensionsof 1 x m
J = 1/m * (-1 .* log(h_theta) * y - (log(1-h_theta) * (1 - y))) + lambda /(2*m)*(theta(2:end,1)' * theta(2:end,1));%y dimension is m x 1
grad_0 = (1/m) .* (h_theta - y') * X(:,1);% (1 x m ) x (m x 1) = 1
grad_rest = (1/m) .* (h_theta - y') * X(:,2:end) + (lambda / m * theta(2:end,:)');%(1 x m) x (m x (n-1)) = 1 x (n-1)
grad = [grad_0,grad_rest];



% =============================================================

end
