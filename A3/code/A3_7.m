%% exercise 7
clear;
close all;
clc;

set(0, 'DefaultLineLineWidth', 2.5);

% train
[pred, z_test_2] = cifar10_starter();

% load data
[x_train, z_train, x_test, z_test, classes] = load_cifar10(2);

% kernels of the first convolutional layer
net = load('cifar10_baseline_modif.mat'); 
kernels = net.net.layers{1, 2}.params.weights; 
figure() 
tile = tiledlayout(4,4); 
for k = 1:16 
    nexttile
    img = kernels(:,:,:,k); 
    imshow(img*10, 'InitialMagnification','fit')
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
    prediction = classes{prediction};
    true_val = z_test(idx(k));
    true_val = classes{true_val};
    %img = reshape(x_test(:,idx(k)), 28,28); 
    %imshow(img, 'InitialMagnification','fit')
    imagesc(x_test(:,:,:,idx(k))/255);
    xlabel(strcat(sprintf('Predicted as: %s', prediction), sprintf('. True class: %s',true_val)))
end 

% confusion matrix for the predictions on test set
figure() 
hold off;
true_val = {};
prediction = {};
for k = 1:length(z_test_2)
    true_val(k) = classes(z_test_2(k));
    prediction(k) = classes(pred(k));
end
C = confusionmat(true_val, prediction); 
confusionchart(true_val,prediction); 
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
