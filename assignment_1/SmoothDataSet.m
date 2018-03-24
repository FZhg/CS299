close all; clear; clc;
run 'load_quasar_data.m';

       %%%%%% 5(c),i-smooth the entire dataset %%%%%%%
% smooth the test and trian data set to get rid of random noise
smoothed_qso_train = smoothTrainSet(train_qso, lambdas);
smoothed_qso_test = smoothTrainSet(test_qso, lambdas);

save('smoothed');

function  smoothed_qso_train = smoothTrainSet(train_qso, lambdas)
%%
% return the smoothed training data set
%-'train_qso': the original training data set that contains random  noised
%-'lambdas': the training input wavelengths
%%
train_inputs = [ones(size(lambdas)) lambdas];
% bandwidth parameter
tau = 5;
%sample size 
m = size(train_qso, 1);
smoothed_qso_train = zeros(size(train_qso));

for i = 1:m
    smoothed_qso_train(i,:) = smoothLWR(train_inputs, train_qso(i,:)', train_inputs, tau)';
end
end

