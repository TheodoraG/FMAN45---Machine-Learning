clear;
close all;
clc;

%% exercise 6
set(0, 'DefaultLineLineWidth', 2.5);

% train
[pred, z_test] = mnist_starter();

% data for testing
x_test = loadMNISTImages('data/mnist/t10k-images.idx3-ubyte');

% kernels of the first convolutional layer
net = load('network_trained_with_momentum.mat'); 
kernels = net.net.layers{1, 2}.params.weights; 
figure() 
tile = tiledlayout(4,4); 
for k = 1:16 
    nexttile
    img = kernels(:,:,1,k); 
    imshow(img, 'InitialMagnification','fit')
    xlabel(sprintf('Kernel %d', k))
end

% some misclassified images
missclassified = (pred ~= z_test);
idx = find(missclassified); 
figure()
tile = tiledlayout(3,3); 
for k = 1:9
    nexttile
    prediction = pred(idx(k)); 
    img = reshape(x_test(:,idx(k)), 28,28); 
    imshow(img, 'InitialMagnification','fit')
    xlabel(sprintf('Predicted as: %d', prediction))
end 

% confusion matrix for the predictions on test set
figure() 
hold off;
C = confusionmat(z_test, pred); 
confusionchart(C) 
title("Consfusion matrix - test set")

% precision and recall for all digits
precis = zeros(1,10);
rec = zeros(1,10) ;
for k=1:10
    precis(k) = precision(k, C); 
    rec(k) = recall(k,C); 
end 

% precision
function prec = precision(class, confusion_matrix) 
    true_positives = confusion_matrix(class, class); 
    false_positives = sum(confusion_matrix(:,class));
    prec = true_positives/false_positives;
end

% recall
function rec = recall(class, confusion_matrix)
    true_positives = confusion_matrix(class, class) ;
    false_negatives = sum(confusion_matrix(class, :));
    rec = true_positives/false_negatives; 
end
