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
    summ=summ-lambda/2*theta(i)^2;%ע�ⶨ��ʱ����������ĳ��Զ�
end
J=-summ/m;
summ=zeros(n,1);%ע��zeros��ʹ�÷�ʽ
for i=1:m%ԭ�����i�ˣ������������޿�
    summ=summ+(h(i)-y(i))*X(i,:)';%ע������������������̫���ˣ�
end
summ=summ+lambda*theta;
summ(1)=summ(1)-lambda*theta(1);%ע���һ��theta0��Ҫ�ı��ݶ�
grad=summ/m;






% =============================================================

end
