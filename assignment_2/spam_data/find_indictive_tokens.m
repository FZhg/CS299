function indictiveTokens = find_indictive_tokens(filename, k)
    %% filename: a string for matrixfile
    %% k: an interger to find the k top indictive tokens
    [phiI, ~] = nb_train(filename);
    metric = log(phiI(:,1)) ./ log(phiI(:,2));
    [~, indexes] = maxk(metric, k);
    [~, tokenlist, ~] = readMatrix(filename);
    tokenlist = split(tokenlist);
    indictiveTokens = string(tokenlist(indexes));
    indictiveTokensStr = join(indictiveTokens);
    fprintf( 'Top %i', k);
    fprintf('indictive tokens is \" %s \" .\n', indictiveTokensStr);
end