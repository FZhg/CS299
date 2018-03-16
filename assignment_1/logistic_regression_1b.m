close all; clear; clc;

% the rows of X is input variables
% the rows of Y is the response variables
fileIDX = fopen('logistic_x.txt', 'r');
sizeX = [2 Inf];
formatSpec = '%f';
X = fscanf(fileIDX, formatSpec, sizeX).';
% append the intercept term
X = [ones(size(X, 1), 1) X];

fileIDY  = fopen('logistic_y.txt', 'r');
Y = fscanf(fileIDY, formatSpec);

%plot the x and y
% the sub dataset for respective y is 1
Xp = X(1:50 ,:);
%the other half of sub dataset
Xn = X(51:size(X, 1) ,:);
sz = 25;
x1range = [0 8];
x2range = [-5 4];



%the implementation of the Newton's Method for logistic regression
%serveral instance variables
THETA_INITIAL = zeros(1, size(X, 2));
ERROR_MARGINS = 0.00001;
%the size of sample space
m = size(Y, 1);



%the sigmoid function
sigmoid =  @(z)1./(1 + exp(-z));
%the cost function
J = @(theta) 1 / m * sum(log(sigmoid(Y.*( X * theta'))));

thetaOptimized = getTheta(J, THETA_INITIAL,ERROR_MARGINS);
%the hypthesis funciton
h = @(X) sigmoid(X * thetaOptimized');

%plot the decision boundary
% step size for the accuracy of the boundary curve
inc = 0.01;

% generate grid coordinates
[x1, x2] = meshgrid(x1range(1):inc:x1range(2), x2range(1):inc:x2range(2));
imageSize = size(x1);

x1x2 = [x1(:) x2(:)]; % make the (x1, x2) pairs as row vectors

hypothesis = zeros(length(x1x2), 1);
for i = 1:length(x1x2)
    Xhypo= [1 x1x2(i,:)];
    htemp = h(Xhypo);
    if htemp > 0.5
        hypothesis(i) = 1;
    else
        hypothesis(i) = 0;
    end
end
%reshap the hypothesis to be positioned on each grid point
decisionMap = reshape(hypothesis, imageSize);

% plot the decision boundary
figure
imagesc(x1range, x2range, decisionMap);
hold on;
cmap = [1 0.8 0.8; 0.9 0.9 1];
colormap(cmap);

 
scatter(Xp(:,2), Xp(:, 3), sz, 'red', 'filled');
hold on;
scatter(Xn(:,2), Xn(:, 3), sz, 'blue', 'filled', 'd');
hold on;
title("Assign#1-1b: Logistic Regression Optimized with Newton's Method ");
xlim(x1range); ylim(x2range);
xlabel('0 < X1 < 8'); 
ylabel('-5 < X2 < 4');
legend('y = 1','y = -1', 'Location', 'Southwest')
hold on;

%save the image
 saveas(gcf,'1b.png')

%disp tests
disp(sum(sigmoid([1 2 3])));
disp(J([0 0 0]));
disp(getGradient(J, [0 0 0]));
disp(getHessian(J, [0 0 0]));
disp(getTheta(J, [0 0 0], 0.00001));

% get the optimized theta
function theta = getTheta(costfunc, thetaIni, errorMargins)
%% costfunc is the cost funciton for the logistic regression
%% thetaIni is the start popint to search for the optimized theta
%% return the optimized theta which can make the gradient down to zero
%% thus the cost function down to the minimal
theta = thetaIni;
grad = getGradient(costfunc, theta);
while norm(grad) > errorMargins
    grad = getGradient(costfunc, theta);
    H = getHessian(costfunc, theta);
    disp('H:'); disp(H); 
    disp('grad:'); disp(grad);
    theta = theta - grad / H; 
    disp('theta'); disp(theta);
end
end

%get the hessian
function H = getHessian(f, x)
%% f is a function
%% x is a input varibale
%% return the hessian for the function at x
gx = getGradient(f, x);
H = zeros(size(x, 2));
h = 0.00001;

%iterate over al indexes in x
for i = 1: size(x, 2)
    oldValues = x(i);
    x(i) = oldValues + h;
    gxh = getGradient(f, x); %get the grad f(x + h)
    x(i) = oldValues; % restore to previous value
    
    %compute the second partial derative
    H(:,i) = (gxh - gx)./ h;
    %iterate over to the next variable
end
end


% get the gradient
function grad = getGradient(f, x)
%% f is a function
%% x is a input varibale
%% return the gradient for the function at x

fx = f(x);
grad = zeros(size(x));
h = 0.00001;

%iterate  over all indexes in x
for i = 1:size(x, 2)
oldValues = x(i);
x(i) = oldValues + h; %increment by h
fxh  = f(x); % evaluate f(x + h)
x(i) = oldValues;%restore to the previous value for x(i)

%compute the partial derative
grad(i) = (fxh - fx) / h; %the slop
%iterate to the next index
end
end


