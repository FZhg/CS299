filenames = split("MATRIX.TRAIN.50 MATRIX.TRAIN.100 MATRIX.TRAIN.200 MATRIX.TRAIN.400 MATRIX.TRAIN.800 MATRIX.TRAIN.1400");
sampleSizes = [50, 100, 200, 400, 800, 1400];
errors = zeros(length(sampleSizes), 1);

for i= 1:length(sampleSizes)
    fprintf(filenames(i), ' ');
    [~, error] = nb_test(filenames(i));
    errors(i) = error;
end

figure()
plot(sampleSizes, errors)
title('Naive Bayes: Test Error V.s. Training Samples Size')
xlabel('Training Samples size')
ylabel('Test Error')
saveas(gcf, 'cs299_ps2_p5c.jpg');