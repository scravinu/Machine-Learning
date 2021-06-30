function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and
%   sigma. You should complete this function to return the optimal C and
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example,
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using
%        mean(double(predictions ~= yval))
%
C_list = [0.01,0.03,0.1,0.3,1,3,10,30];
sigma_list = [0.01,0.03,0.1,0.3,1,3,10,30];
C = C_list(1);
sigma = sigma_list(1);
model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
predictions = svmPredict(model, Xval);
p_error_min = mean(double(predictions ~= yval));

for cIndex = 1 : length(C_list)
  for sigmaIndex = 1 : length(sigma_list)
      C = C_list(cIndex);
      sigma = sigma_list(sigmaIndex);
      model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
      predictions = svmPredict(model, Xval);
      p_error = mean(double(predictions ~= yval));
      if p_error <= p_error_min
          p_error_min = p_error;
          C_opt = C;
          sigma_opt = sigma;
      end
  end
end

C = C_opt;
sigma = sigma_opt;












% =========================================================================

end
