%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% svm_test.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Construct test and train matrices
sampleSizes = [50, 100, 200, 400, 800, 1400];
errors = zeros(length(sampleSizes), 1);

[sparseTrainMatrix, tokenlist, testCategory] = ...
    readMatrix('MATRIX.TEST');
Xtest = full(sparseTrainMatrix);
Xtest = 1.0 * (Xtest > 0);
ytest = (2 * testCategory - 1)';
squared_X_test = sum(Xtest.^2, 2);

m_test = size(Xtest, 1);
gram_test = Xtest * Xtrain';
Ktest = full(exp(-(repmat(squared_X_test, 1, m_train) ...
                   + repmat(squared_X_train', m_test, 1) ...
                   - 2 * gram_test) / (2 * tau^2)));

% preds = Ktest * alpha;

% fprintf(1, 'Test error rate for final alpha: %1.4f\n', ...
%         sum(preds .* ytest <= 0) / length(ytest));
for i= 1:length(sampleSizes)
    avg_alpha = avg_alphas(i)
    preds = Ktest * avg_alpha;
    fprintf(1, 'Test error rate for average alpha: %1.4f\n', ...
            sum(preds .* ytest <= 0) / length(ytest));
    test_error = sum(preds .* ytest <= 0) / length(ytest);
    errors(i) = test_error;
end

figure()
plot(sampleSizes, errors)
title('SVM: Test Error V.s. Training Samples Size')
xlabel('Training Samples size')
ylabel('Test Error')
saveas(gcf, 'cs299_ps2_p5.jpg');