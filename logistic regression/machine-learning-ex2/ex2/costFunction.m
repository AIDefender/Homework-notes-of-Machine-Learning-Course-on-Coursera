function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples
n=length(theta);
% You need to return the following variables correctly 
J = 0;
h=sigmoid(X*theta);
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
summ=0;
for i=1:m
    summ=summ+y(i)*log(h(i))+(1-y(i))*log(1-h(i));
end
J=-summ/m;
summ=zeros(n,1);%注意zeros的使用方式
for i=1:m%原来打成i了，，，，，，巨坑
    summ=summ+(h(i)-y(i))*X(i,:)';%注意行向量与列向量！太坑了！
end
grad=summ/m;







% =============================================================

end
