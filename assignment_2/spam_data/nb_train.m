function [phiI, pYeq1] =  nb_train(filename)
%% filename: a string for the matrix file
%% return a vector(trainCategory X 2), the first column vector is the probability when y =1
%% the second column vector is the probability when y=-1

    [spmatrix, ~, trainCategory] = readMatrix(filename);

    trainMatrix = full(spmatrix);
    numTrainDocs = size(trainMatrix, 1);
    numTokens = size(trainMatrix, 2);

    % trainMatrix is now a (numTrainDocs x numTokens) matrix.
    % Each row represents a unique document (email).
    % The j-th column of the row $i$ represents the number of times the j-th
    % token appeared in email $i$. 


    % tokenlist is a long string containing the list of all tokens (words).
    % These tokens are easily known by position in the file TOKENS_LIST

    % trainCategory is a (1 x numTrainDocs) vector containing the true 
    % classifications for the documents just read in. The i-th entry gives the 
    % correct class for the i-th email (which corresponds to the i-th row in 
    % the document word matrix).

    % Spam documents are indicated as class 1, and non-spam as class 0.
    % Note that for the SVM, you would want to convert these to +1 and -1.


    % YOUR CODE HERE
    numSpam = sum(trainCategory);
    numNonSpam = numTrainDocs - numSpam;
    pYeq1 = numSpam / numTrainDocs;
    phiI = zeros(numTokens, 2);
    for j=1:numTokens
        xj = trainMatrix(:,j);
        phiYeq1 = (sum((xj(:) == 1 & trainCategory(:) == 1)) + 1) / (numSpam + 2);
        phiYeq0 = (sum((xj(:) == 1 & trainCategory(:) == 0)) + 1) / (numNonSpam + 2); % Laplace smoothing
        phiI(j,:)= [phiYeq1, phiYeq0];
    end
end
