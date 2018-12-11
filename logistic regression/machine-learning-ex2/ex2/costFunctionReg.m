function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n=length(theta);
% You need to return the following variables correctly 
J = 0;
grad = zeros(n,1);
h=sigmoid(X*theta);
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
summ=0;
for i=1:m
    summ=summ+y(i)*log(h(i))+(1-y(i))*log(1-h(i));
end
for i=2:n
    summ=summ-lambda/2*theta(i)^2;%注意定义时正则化项里面的除以二
end
J=-summ/m;
summ=zeros(n,1);%注意zeros的使用方式
for i=1:m%原来打成i了，，，，，，巨坑
    summ=summ+(h(i)-y(i))*X(i,:)';%注意行向量与列向量！太坑了！
end
summ=summ+lambda*theta;
summ(1)=summ(1)-lambda*theta(1);%注意第一项theta0不要改变梯度
grad=summ/m;






% =============================================================

end
