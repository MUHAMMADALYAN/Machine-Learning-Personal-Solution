function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


 

 % Imp Not in matrix A*B != B*A so always check what u want to multi;lpy with whom and then select position of matrix multipliaction accordingy
h=X*theta;
J = (1/(2*m))*sum((h - y).^2) + (lambda/(2*m))*sum(theta(2:end).^2);



%grad(1)=(1/m)*((X(:,1))'*(h-y)); %for j =0 bias term 
%grad_reg=(lambda/m)*(theta(2:end));  
%grad(2)=(1/m)*((X(:,2:end)'*(h-y)))+ grad_reg;

grad=(1/m)*(X'*(h-y)); %2x1
grad_reg=(lambda/m)*(theta(2:end)); %for j=1 to so on 
grad(2:end)=grad(2:end)+grad_reg; % adding regularization only in bias






% =========================================================================

grad = grad(:);

end
