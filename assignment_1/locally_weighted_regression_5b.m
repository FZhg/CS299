%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%  5(b) visualize the data
%%%% i. demands a linear regression with
%%%% visualization and optimized parameter;
%%%% ii. visualize raw data and weighted 
%%%% linear regression;
%%%% iii. vary the bandwidth parameter
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all; clear; clc;

% define the query point
run 'load_quasar_data.m';
%load the first training example
trainEx1 = train_qso(1,:)';

       %%%%%%%% linear regression for the first training example %%%%%%%
%plot the raw data
figure
plot( lambdas, trainEx1);
title('5(b),i-Linear regression and raw data of the first training example');
xlabel('Wavelength \lambda (10^{-1} nm)');
ylabel('Flux');
hold on;

%compute theta using normal equation
X = [ones(size(lambdas)) lambdas];
theta = ( X' * X)^(-1) * X' * trainEx1;
disp(theta);

%plot the linear regression line
YHypoLin = X * theta;
p1 = plot(lambdas, YHypoLin);
legend('Raw Data', 'Linear Regression');
p1(1).LineWidth = 2;

saveas(gcf, '5b(i).png')


                %%%%%%%    weighted linear regression %%%%%%%%
%plot the raw data
figure
plot( lambdas, trainEx1);
title('5(b),ii-Weighted linear regression and raw data of the first training example');
xlabel('Wavelength \lambda (10^{-1} nm)');
ylabel('Flux');
hold on;

% for each lambda compute the predict value
LWRX = X;
%the demension of sample space
m = size(LWRX, 1);
% the bandwidth parameter
tau = 5;
LWRY = smoothLWR(X, trainEx1, LWRX, tau);

%plot the locally weighted regression
p2 = plot(lambdas, LWRY);
ylim([-2 8]);
legend('Raw Data', 'Locally Weighted Regression')
p2(1).LineWidth = 2;
saveas(gcf, '5b(ii).png')



                %%%%%% vary the bandwidthparameter %%%%%%

% when tau is 1
tau = 1;
LWRY = smoothLWR(X, trainEx1, LWRX, tau);

variedTau = figure(3);
set(variedTau, 'Position', [100, 100, 512, 1200]);
ax1 = subplot(4, 1, 1);
plot(ax1, lambdas, LWRY);
title(ax1, 'Locally weighted regression with \tau = 1');
ylabel('Flux');

% when tau is 10
tau = 10;
LWRY = smoothLWR(X, trainEx1, LWRX, tau);

ax2 = subplot(4, 1, 2);
plot(ax2, lambdas, LWRY);
title(ax2, 'Locally weighted regression with \tau = 10');
ylabel('Flux');

% when tau is 100
tau = 100;
LWRY = smoothLWR(X, trainEx1, LWRX, tau);

ax3 = subplot(4, 1, 3);
plot(ax3, lambdas, LWRY);
title(ax3, 'Locally weighted regression with \tau = 10');
ylabel('Flux');

% when tau is 100
tau = 1000;
LWRY = smoothLWR(X, trainEx1, LWRX, tau);

ax4 = subplot(4, 1, 4);
plot(ax4, lambdas, LWRY);
title(ax4, 'Locally weighted regression with \tau = 10');
ylabel('Flux');

saveas(gca, '5b(iii).png')
%%% comments on varied tau  %%%
% When tau get bigger, the weight decrease quickly. Thus no input will have
% much difference on the regression model.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
