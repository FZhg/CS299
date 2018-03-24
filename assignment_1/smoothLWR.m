
function outputs = smoothLWR(trainInputs, trainOutputs, queryPoints, tau)
%%
% return smoothed curve
%-'trainInputs': training input data
%-'trainOutputs': training output data
%-'tau': also  called as bandwidth parameter for the gaussian kernel
%%
outputs = zeros([size(queryPoints, 1) 1]);
% the demension of the sample size
m = size(trainInputs, 1);
for i = 1:m
    outputs(i) = LWRPredict(trainInputs, trainOutputs, queryPoints(i,:), tau);
end
end

function Output = LWRPredict(trainInputs, trainOutputs, queryPoint, tau)
%%
% return the predicted value for the query point
%-'trainInputs': training input data
%-'trainOutputs': training output data
%- 'queryPoints': the data points we want to predict on
%-'tau': also  called as bandwidth parameter for the gaussian kernel
%%
W = getWeight(trainInputs, queryPoint, tau);
theta = (trainInputs' * W * trainInputs)^(-1)* trainInputs' * W * trainOutputs;
Output = queryPoint * theta;
end

function W = getWeight(trainInputs, queryPoint, tau)
%%
% get the diagonal weighted matrix
%-'trainInputs': training input data
%- 'queryPoints': the data points we want to predict on
%-'tau': also  called as bandwidth parameter for the gaussian kernel
%%

% the demension of the training input
m = size(trainInputs, 1);
W = eye(m);
for i = 1: m
    distance = trace((trainInputs(i,:) - queryPoint)' * (trainInputs(i,:) - queryPoint));
    W(i, i) = exp(-  distance / (2* tau^2));
end
end
