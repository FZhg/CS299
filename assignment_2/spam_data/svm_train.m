%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% svm_train.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
filenames = split("MATRIX.TRAIN.50 MATRIX.TRAIN.100 MATRIX.TRAIN.200 MATRIX.TRAIN.400 MATRIX.TRAIN.800 MATRIX.TRAIN.1400");
sampleSizes = [50, 100, 200, 400, 800, 1400];
errors = zeros(length(sampleSizes), 1);

[sparseTrainMatrix, tokenlist, testCategory] = ...
    readMatrix('MATRIX.TEST');
Xtest = full(sparseTrainMatrix);
Xtest = 1.0 * (Xtest > 0);
ytest = (2 * testCategory - 1)';
squared_X_test = sum(Xtest.^2, 2);

m_test = size(Xtest, 1);

               
for i= 1: length(filenames)
    [sparseTrainMatrix, tokenlist, trainCategory] = ...
    readMatrix(sprintf(filenames(1)));
    Xtrain = full(sparseTrainMatrix);
    m_train = size(Xtrain, 1);
    ytrain = (2 * trainCategory - 1)';
    Xtrain = 1.0 * (Xtrain > 0);

    squared_X_train = sum(Xtrain.^2, 2);
    gram_train = Xtrain * Xtrain';
    tau = 8;

    % Get full training matrix for kernels using vectorized code.
    Ktrain = full(exp(-(repmat(squared_X_train, 1, m_train) ...
                        + repmat(squared_X_train', m_train, 1) ...
                        - 2 * gram_train) / (2 * tau^2)));

    lambda = 1 / (64 * m_train);
    num_outer_loops = 40;
    alpha = zeros(m_train, 1);

    avg_alpha = zeros(m_train, 1);
    Imat = eye(m_train);

    count = 0;
    for ii = 1:(num_outer_loops * m_train)
      count = count + 1;
      ind = ceil(rand * m_train);
      margin = ytrain(ind) * Ktrain(ind, :) * alpha;
      g = -(margin < 1) * ytrain(ind) * Ktrain(:, ind) + ...
          m_train * lambda * (Ktrain(:, ind) * alpha(ind));
      % g(ind) = g(ind) + m_train * lambda * Ktrain(ind,:) * alpha;
      alpha = alpha - g / sqrt(count);
      avg_alpha = avg_alpha + alpha;
    end
    avg_alpha = avg_alpha / (num_outer_loops * m_train);
    
    % compute error
    gram_test = Xtest * Xtrain';
    Ktest = full(exp(-(repmat(squared_X_test, 1, m_train) ...
                       + repmat(squared_X_train', m_test, 1) ...
                       - 2 * gram_test) / (2 * tau^2)));
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
saveas(gcf, 'cs299_ps2_p5d.jpg');
