syms theta0 theta1 theta2 J theta
fileIDX = fopen('logistic_x.txt', 'r');
sizeX = [2 Inf];
formatSpec = '%f';
X = fscanf(fileIDX, formatSpec, sizeX).';
% to include the intercept term
X = [ones(size(X, 1), 1) X].';

fileIDY  = fopen('logistic_y.txt', 'r');
Y = fscanf(fileIDY, formatSpec);

%plot the x and y
scatter3(X(2,:),X(3,:),Y, 16);
hold();

%the implementation of the Newton's Method for logistic regression
%serveral instance variables
THETA_INITIAL = zeros(3, 1);
ERROR_MARGINS = 1;
DELTA_THETA = 0.1;
%the sample space
m = size(Y, 1);




%the symbolic functions
theta = [theta0; theta1; theta2];
J(theta0, theta1, theta2) = cost(X, Y, theta);
gradLR(theta0, theta1, theta2) = gradient(J, theta);
HLR(theta0, theta1, theta2) = hessian(J, theta);
hypothesis(theta0, theta1, theta2) = hypo(X, theta);

%the search for the optimized theta
G = gradLR(THETA_INITIAL(1), THETA_INITIAL(2), THETA_INITIAL(3));
H = HLR(THETA_INITIAL(1), THETA_INITIAL(2), THETA_INITIAL(3));
theta = THETA_INITIAL;
while(norm(G) > ERROR_MARGINS)
disp(theta);
theta = theta - H^(-1) * G;
G = gradLR(theta(1), theta(2), theta(3));
H = HLR(theta(1), theta(2), theta(3));
end

hypoFinal = hypothesis(theta(1), theta(2), theta(3));
surf(hypoFinal, X(1,:), X(2,:));


%the cost function builder
function J = cost(X, Y, theta)
%the sample space
m = size(Y, 1);
J = 0;
for i = 1: m
    J = J + log(1 + exp(-Y(i) * theta.'* X(:, i)));
end
end

% hypotheis producer
function h = hypo(X, theta)
%the sample space
m = size(X, 2);
h = sym(zeros(m,1));
for i = 1:m
    h(i) = 1 + exp(theta.' * X(:,i) );
end
end























% % the cost function
% function J = costLR(X, Y, theta)
% m = size(Y);
% J = -1 / m * symsum(log(1 + exop(-y(i)* theta.' * x(i,:))), i, 1, m);
% end
% 
% %the gradient funcition
% function G = gradLR(h, X, Y, theta)
% gDim = size(theta);
% G = zeros(gDim);
% J0 = cost(X, Y, theta);
% 
% %the incremental for the gradient
% for i = 1:gDim
%     theta(i) = theta(i) + h;
%     Ji = cost(X, Y, theta);
%     G(i) = (Ji - J0) / h ;
% end
% end
% 
% %the hession function
% function H = hessianLR(h, X, Y, theta)
% H = zeros(size(theta)(1));
% G = gradLR(h, X, Y, theta);
% for i = 1:size(theta)
%     theta
% end
% end
% 
% 
% 
% 
% 
% % when error margin are met, output theta and the hypothesis function
