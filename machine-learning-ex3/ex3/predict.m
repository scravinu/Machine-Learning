function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);
num_layer2Nodes = size(Theta2, 2) - 1;
% You need to return the following variables correctly
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
X = [ones(m, 1) X];%(m x n+1)(5000x401)
layer2 = (sigmoid( Theta1 * X'))';%[(25 x 401) * (401 x m)]' =[25 x m]'= m x 25
layer2 = [ones(m,1) layer2];%5000 x 26
layer3 = (sigmoid( Theta2 * layer2'))'; %[](10 x 26) * (26 x m)]' = m x 10
[prob_max,p]=max(layer3,[],2);%columnwise max p dimension = m x 1, p captures the index of the location of prob_max






% =========================================================================


end
