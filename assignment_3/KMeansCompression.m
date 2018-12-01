%% Function: KMeansCompressions
%% ----------------------------
%% original image with size M< N, C
%% return and show compressed image side with the original image
function result = KMeansCompression(~, ~)
%% Imenplementation Notes: This is a vetorized version of K-means.
%% I didn't compare the running time with non-vectorized version.
clear; clc; 
K = 16;
image = gpuArray(double(imread('mandrill-large.tiff')));
[M, N, C] = size(image);
result = gpuArray(zeros([M, N, C]));
labels = gpuArray(zeros([M, N]));

imageLarge = repmat(image, 1, 1, 1, K); % with same pixel piling up at the fourth demension
centeroids = datasample(reshape(image, [M*N, C]), K); %K, C shape random centeroids.
iterationCount = 0;
centeroidsUpdate = 0;

while((iterationCount < 31)|| (centeroidsUpdate > 1e-4))
    %increase iteration count
    iterationCount =  iterationCount +1;
    
    %assign labels
    centeroidsLarge = permute(repmat(centeroids, 1, 1, N, M), [4 3 2 1]); %M, N, C, K matrixs.
    distance = reshape(sum((centeroidsLarge - imageLarge).^2, 3), [M, N, K]);
    [~, labels] = min(distance, [], 3);
    
    preCenteroids = centeroids;
    %update centeroids
    for label = 1:K
        mask2d = labels==label;
        mask3d = repmat(mask2d, 1, 1, C);
        centeroids(label, :) = reshape(sum(sum(mask3d.*image, 1), 2), [1, 3]) ./ sum(sum(mask2d, 1), 2);
    end
    
    %calculate the delta for testing convergence
    centeroidsUpdate = sum(sum((preCenteroids - centeroids).^2, 1), 2);
    
     %for debug
    disp(['Iterations ', num2str(iterationCount), ': delta, ', num2str(centeroidsUpdate)]);   
end

%update the compressed image
result = reshape(centeroids(labels, :), [M, N, C]);
imshow(uint8(round(result)));
result_cpu = uint8(round(gather(result)));
imwrite(result_cpu, 'compressed-large-10.png');
end
