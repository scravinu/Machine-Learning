function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

num_layers = 3;
% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
% Feed forward
X = [ones(m, 1) X];%(m x n+1)(5000x401)
layer2 = (sigmoid( Theta1 * X'))';%[(25 x 401) * (401 x m)]' =[25 x m]'= m x 25
layer2 = [ones(m,1) layer2];%5000 x 26 (m x 25+1)
layer3 = (sigmoid( Theta2 * layer2'))'; %The output layer[](10 x 26) * (26 x m)]' = m x 10%

for k = 1 : num_labels % looping over k classes
  h_theta = (layer3(:,k))'; %layer 3 belonging to kth class, dimension of 1 x m
  kth_y = (y == k);%Temp Y for class k (m x 1)
  J = J + 1/m * (-1 .* log(h_theta) * kth_y - (log(1-h_theta) * (1 - kth_y)));
end
% regularization part
theta1_sumsquare = sum(sum(Theta1(:,2:end).^2));
theta2_sumsquare = sum(sum(Theta2(:,2:end).^2));

J = J + lambda/(2*m)*(theta1_sumsquare + theta2_sumsquare);

%  backpropagation
DELTA_1 = zeros(size(Theta1));
DELTA_2 = zeros(size(Theta2));
for t = 1:m
  a_1 = X(t,:)'; % n+1 x 1
  z_2 = Theta1 * a_1;% (25 x (n+1)) * ((n+1)x 1) = 25 x 1
  a_2 = sigmoid(z_2); % (25 x (n+1)) * ((n+1)x 1) = 25 x 1
  a_2 = [1;a_2];%(26 x 1)
  z_3 = Theta2 * a_2; % 1 (10 x 26) * (26 * 1) = (10 x 1)
  a_3 = sigmoid( z_3); % (10 x 26) * (26 * 1) = (10 x 1)

  %for each output unit in layer 3 setting error = (a_3 - yk), layer l = 3
  delta_3 = zeros(num_labels,1);%(10 x 1)
  mth_y = [1:num_labels]';
  mth_y = (mth_y == y(t));% (10 x 1)
  delta_3 = (a_3 - mth_y); % (10 x 1)

  %for hidden layer l = 2, back backpropagating the error term from layer 3
  delta_2 = Theta2'*delta_3.*sigmoidGradient([1;z_2]);%(26x10)*(10x1)=26x1
  % 1 is included above to match the dimensions , it is eventually discarded anyway
  % as seen below.
  DELTA_2 = DELTA_2 + delta_3 * (a_2)';% 10x1 * 1 x 26 => 10x26
  DELTA_1 = DELTA_1 + delta_2(2:end) * (a_1)'; %25x1 * 1 x n+1 = 25 x n+1









Theta1_grad = 1/m*DELTA_1;%25 x n + 1
%just adding the regularization term to non bias terms *i.e the first column
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m * Theta1(:,2:end));
Theta2_grad = 1/m*DELTA_2;% 10 x 26
%just adding the regularization term to non bias terms *i.e the first column
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda/m * Theta2(:,2:end));


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
