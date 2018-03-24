close all; clear; clc;
load('smoothed');


%      %%%%%% 5(c), ii - Function Estimator %%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%TEST FOR FUNCITON%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % test for getDistance method
% distance = getDistance(smoothed_qso_test(2,:), smoothed_qso_test(3,:), true);
% disp(distance);
%
% % test for getNeighbors and getDistances 
% distances = getDistances(smoothed_qso_test, smoothed_qso_test(2,:));
% disp(distances);
% neighbors = getNeighbors(distances, 3);
% disp(neighbors);
% 
% % test for the kerl function
% temp = ker(2);
% disp(temp);
% 
% temp = ker(0.45);
% disp(temp);
% 
% % test for estimateLeft method
% lambdasLeft = lambdas(1:51);
% lambdasRight = lambdas(151:end);
% f = smoothed_qso_train(3,:);
% fright = f(:,151:end);
% fleft =  f(:,1: 51);
% fleftEstimated = estimateLeft(smoothed_qso_train, fright, 3);
% plot(lambdasLeft, fleftEstimated);
% hold on;
% plot(lambdasLeft, fleft);
% disp(getDistance(fleftEstimated, fleft, false));
% 
%%%%%%%%%%%%%%%%%%%END OF TEST%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% compute the training error
averageTrainError = getAverageTrainError(smoothed_qso_train, 3);
disp(averageTrainError);

% compute the test error
averageTestError = getAverageTestError(smoothed_qso_train, smoothed_qso_test, 3);
disp(averageTestError);

% plot for test example 1
figure
lambdasLeft = lambdas(1:51);
lambdasRight = lambdas(151:end);
f = smoothed_qso_train(1,:);
fright = f(:,151:end);
fleft =  f(:,1: 51);
fleftEstimated = estimateLeft(smoothed_qso_train, fright, 3);
plot(lambdasLeft, fleftEstimated);
hold on;
plot(lambdasLeft, fleft);
title('The estimated and observed left curve of text example 1');
xlabel('wavelength \lambda (10^{-1} nm) ');
ylabel('Flux');
legend('Estimated','Observed','Location', 'southeast');
saveas(gcf,'5c(iii)-testEx1.png');

%plot for the test example 6
figure
lambdasLeft = lambdas(1:51);
lambdasRight = lambdas(151:end);
f = smoothed_qso_train(6,:);
fright = f(:,151:end);
fleft =  f(:,1: 51);
fleftEstimated = estimateLeft(smoothed_qso_train, fright, 3);
plot(lambdasLeft, fleftEstimated);
hold on;
plot(lambdasLeft, fleft);
title('The estimated and observed left curve of text example 6');
xlabel('wavelength \lambda (10^{-1} nm) ');
ylabel('Flux');
legend('Estimated','Observed','Location', 'southeast');
saveas(gcf,'5c(iii)-testEx6.png');

function averageTestError = getAverageTestError(trainSet, testSet, k)
%%
% return the average test error for the entire test data set
% -'trainSet'; the training data set
% -'testSet': the test data set
% -'k': the number of nearest neighbors we want to focous on
%%
% the test size
m = size(testSet, 1);
testSetRight = trainSet(:,151: end);
testSetLeft = trainSet(:,1: 51);
testEstimated = zeros(size(testSetLeft));
testErrors = zeros(size(testSet, 1), 1);
for i = 1: m
    testEstimated(i,:) = estimateLeft(trainSet, testSetRight(i,:), k);
    testErrors(i,:) = getDistance(testSetLeft(i,:), testEstimated(i,:), false);
end
averageTestError = sum(testErrors) / m;
end

function averageTrainError = getAverageTrainError(trainSet, k)
%%
% return the estimated left funtion for the entire training right funciton;
% -'trainSet': the training data set;
% -'k': the number of nearest neighbors we want to foucus on.
%%
trainSetRight = trainSet(:,151: end);
trainSetLeft = trainSet(:,1: 51);
trainEstimated = zeros(size(trainSetLeft));
errors = zeros(size(trainSet, 1), 1);
% the size of training set
m = size(trainSet, 1);
for i = 1:m
    trainEstimated(i,:) = estimateLeft(trainSet, trainSetRight(i,:), k);
    errors(i) = getDistance(trainSetLeft(i,:), trainEstimated(i,:), false);
end
averageTrainError = sum(errors) / m;
end

function fleft = estimateLeft(trainSet, fright, k)
%%
% return the estimated left function
% -'trainSet'; the training data set
% -'fright': the observed right function
% -'k': the number of nearest neighbors
%%
% cut the training set into left and right part
trainSetRight = trainSet(:,151: end);
trainSetLeft = trainSet(:,1: 51);

distances = getDistances(trainSetRight, fright);
neighbors = getNeighbors(distances, k);
h = max(distances);

%comopute the numerator and xc
numerator = zeros(1, size(trainSetLeft, 2));
denominator = 0;
for i = 1:k
neighbor = neighbors(k);
numerator = numerator + ker(distances(neighbor) / h) * trainSetLeft(i,:);
denominator = denominator +ker(distances(neighbor) / h);

end

fleft = numerator / denominator;
end

function  temp = ker(t)
temp = max([1-t; 0]);
end

function neighbors = getNeighbors(distances, k)
%%
% return the K neighbors that is closet to our query funciton
%-'distances': is the vector containing distance of each training example
%-'K': the numbers of neighbors we want to find, must be a postitive
%integers
%%
neighbors = zeros(k, 1);
for i = 1:k
    [~, neighbors(i)] = min(distances,  [], 'omitnan');
    % update the the min to NaN for the next closet neighbor
    distances(neighbors(i)) = NaN;
end
end
     
 function distance = getDistance(f1, f2, right)
diff = (f1 - f2).^2;
if right == true
    % 151 is index of the wavelength 1300 among the lambdas vector
    distance =  sum(diff(151:end));
else
    % 51 is the index of the wavelength 1200 among the lambdas vector
    distance = sum(diff(1:51));
end
 end

function distances = getDistances(trainSet, f)
%%
% return the all samples' distances relative to the function f.
%-'trainSet': the training set that contains both left and right examples
%-'f': the smoothed funciton 
%%
diff = trainSet - f;
% only cumulate the right function distance
diff = diff(:,151:end).^2;
temp = cumsum(diff, 2);
distances = temp(:, end);
end