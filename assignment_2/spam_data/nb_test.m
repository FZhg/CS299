function [output, error] = nb_test(filename)
%% filename: a string of the filename of the matrix file
%% return y_hat: a vector(1 X numTestdocs) contains every predictive label for every test email.
    [spmatrix, ~, category] = readMatrix('MATRIX.TEST');

    testMatrix = full(spmatrix);
    numTestDocs = size(testMatrix, 1);
    numTokens = size(testMatrix, 2);

    % Assume nb_train.m has just been executed, and all the parameters computed/needed
    % by your classifier are in memory through that execution. You can also assume 
    % that the columns in the test set are arranged in exactly the same way as for the
    % training set (i.e., the j-th column represents the same token in the test data 
    % matrix as in the original training data matrix).

    % Write code below to classify each document in the test set (ie, each row
    % in the current document word matrix) as 1 for SPAM and 0 for NON-SPAM.

    % Construct the (numTestDocs x 1) vector 'output' such that the i-th entry 
    % of this vector is the predicted class (1/0) for the i-th  email (i-th row 
    % in testMatrix) in the test set.
    output = zeros(numTestDocs, 1);

    %---------------
    % YOUR CODE HERE
    logPHats = zeros(numTestDocs, 2);
    [phiI,pYeq1] = nb_train(filename);
    logPYeq1 = log(pYeq1);
    logPYeq0 = log(1 - pYeq1);
    for i = 1:numTestDocs
        x = testMatrix(i,:)';
        x(x == 0) = -1;
        x(x == 0) = 0;  %to convert to 1 - phiI when xj is not 1
        phiIi = phiI + x;
        logPhiIi = log(phiIi);
        sumEq1 = sum(logPhiIi(:,1)) + logPYeq1;
        sumEq0 = sum(logPhiIi(:,2)) + logPYeq0;
        logPHats(i,:) = [sumEq1, sumEq0];
    end
    output = logPHats(:,1) - logPHats(:,2);
    output(output > 0) = 1;
    output(output < 0) = 0;

    %---------------
    % Compute the error on the test set
    y = full(category);
    y = y(:);
    error = sum(y ~= output) / numTestDocs;

    %Print out the classification error on the test set
    fprintf(1, 'Test error: %1.4f\n', error);
end


