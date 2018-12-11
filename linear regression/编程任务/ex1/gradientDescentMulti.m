function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
n=length(theta);%这边变量的细节不要错！
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
   temp=zeros(n,1);%搞清楚每步在做什么
    for j=1:n
       % temp(j,1)=0;%一定要注意初始化的条件的位置！
        for i=1:m %for语句的格式
            temp(j,1)=temp(j,1)+(X(i,:)*theta-y(i,:))*X(i,j);%sigema求和的部分
        end
    end
    %好像不能/=
    theta=theta-(alpha/m)*temp; 
    










    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
