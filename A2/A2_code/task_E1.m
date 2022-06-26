clear 
close all
clc

%% task E1
load A2_data.mat

% subtract the mean
data = train_data_01 - mean(train_data_01,2);

% singular value decomposition
[U, S, V] = svd(data);
U_d = U(:, 1:2);

% X projection
X_proj = U_d'*data;

figure()
hold on
scatter(X_proj(1,train_labels_01 == 0), X_proj(2,train_labels_01 == 0),'p')
scatter(X_proj(1, train_labels_01 == 1), X_proj(2, train_labels_01 == 1))
xlabel("PC 1", "fontsize", 12)
ylabel("PC 2", "fontsize", 12)
title("PCA - 2 dimensions")
legend("Class 0", "Class 1","fontsize", 10)


%% task E2
% 2 clusters

[y2, C2] = K_means_clustering(train_data_01, 2);
figure() 
hold on
scatter(X_proj(1, y2 == 1), X_proj(2, y2 == 1)); 
scatter(X_proj(1, y2 == 2), X_proj(2, y2 == 2)); 
xlabel("PC 1", "fontsize", 12)
ylabel("PC 2", "fontsize", 12)
title("PCA - 2 dimensions")
legend("Cluster 1", "Cluster 2","fontsize", 10)

% 5 clusters
[y5, C5] = K_means_clustering(train_data_01, 5);
figure() 
hold on
scatter(X_proj(1, y5 == 1), X_proj(2, y5 == 1)); 
scatter(X_proj(1, y5 == 2), X_proj(2, y5 == 2)); 
scatter(X_proj(1, y5 == 3), X_proj(2, y5 == 3)); 
scatter(X_proj(1, y5 == 4), X_proj(2, y5 == 4)); 
scatter(X_proj(1, y5 == 5), X_proj(2, y5 == 5)); 
xlabel("PC 1", "fontsize", 12)
ylabel("PC 2", "fontsize", 12)
title("PCA - 5 dimensions")
legend("Cluster 1", "Cluster 2", "Cluster 3", "Cluster 4", "Cluster 5", "fontsize", 10)


%% task E3
% 2 clusters
C2 = reshape(C2, 28, 28, 2);
figure()
t = tiledlayout(2,3);
nexttile
imshow(C2(:,:,1),'InitialMagnification','fit' )
xlabel(sprintf("Cluster %d", 1))
nexttile
imshow(C2(:,:,2),'InitialMagnification','fit' )
xlabel(sprintf("Cluster %d", 2))

% 5 clusters
C5 = reshape(C5, 28, 28, 5);
figure()
t = tiledlayout(2,3);
nexttile
imshow(C5(:,:,1),'InitialMagnification','fit' )
xlabel(sprintf("Cluster %d", 1))
nexttile
imshow(C5(:,:,2),'InitialMagnification','fit' )
xlabel(sprintf("Cluster %d", 2))
nexttile
imshow(C5(:,:,3),'InitialMagnification','fit' )
xlabel(sprintf("Cluster %d", 3))
nexttile
imshow(C5(:,:,4),'InitialMagnification','fit' )
xlabel(sprintf("Cluster %d", 4))
nexttile
imshow(C5(:,:,5),'InitialMagnification','fit' )
xlabel(sprintf("Cluster %d", 5))

%% task E4
clear;
close all;
clc;

load A2_data.mat

% 2 clusters
[y_train, C2_train] = K_means_clustering(train_data_01, 2);
% classifiaction results
[perf_train, perf_test] = K_means_classification(y_train, train_labels_01, test_data_01, test_labels_01, C2_train);


%% task E5
clear;
close all;
clc;

load A2_data.mat
% try 15 clusters
K = 15;

% evaluate the evolution of the misclassiification rate
for k = 2:K    
   [y_train_k, C_k_train] = K_means_clustering(train_data_01, k);   
   [perf_train_k, perf_test_k] = K_means_classification(y_train_k, train_labels_01, test_data_01, test_labels_01, C_k_train);
   
   misclass_rate(k-1) = sum(perf_test_k(:,4))*100 / length(test_labels_01);
end

figure()
plot(2:K, misclass_rate,'linewidth', 2.5)
title('Evolution of the misclassification rate over the number of clusters')
xlabel('Number of clusters')
ylabel('Misclassification rate (%)')


%% task E6
clear;
close all;
clc;

load A2_data.mat

% train a binary soft-margin SVM 
svm_model = fitcsvm(train_data_01', train_labels_01);

% calculate class prediction
yp_train = predict(svm_model, train_data_01');
yp_test = predict(svm_model, test_data_01');

% miscalssification rate
train_perf_svm = SVM_classifier(yp_train, train_labels_01);
test_perf_svm = SVM_classifier(yp_test, test_labels_01);

%% task E7
clear 
close all
clc

load A2_data.mat

beta = 1:0.2:6;


for i = 1:length(beta)
    % train a SVM wih Gaussian kernel
    svm_gauss_model = fitcsvm(train_data_01',train_labels_01,'KernelFunction','gaussian','KernelScale', beta(i));

    % calculate class prediction
    yp_train = predict(svm_gauss_model, train_data_01');
    yp_test = predict(svm_gauss_model, test_data_01');

    % miscalssification rate
    train_perf_svm2 = SVM_classifier(yp_train, train_labels_01);
    rate_train_svm = (train_perf_svm2(2) + train_perf_svm2(3)) / length(yp_train)
    test_perf_svm2 = SVM_classifier(yp_test, test_labels_01);
    rate_test_svm(i) = (test_perf_svm2(2) + test_perf_svm2(3)) / length(yp_test)
    
    i
    
    if rate_test_svm(i) == 0
        break
    end
end

figure()
plot(1:0.2:beta(i), rate_test_svm,'linewidth', 2.5)
title('Evolution of the misclassification rate over different beta values')
xlabel('beta')
ylabel('Misclassification rate (%)')

%% task E7 - optimal results
clear 
close all
clc

load A2_data.mat

beta = 4.8;

% train a SVM wih Gaussian kernel
svm_gauss_model_opt = fitcsvm(train_data_01',train_labels_01,'KernelFunction','gaussian','KernelScale', beta);

% calculate class prediction
yp_train_opt = predict(svm_gauss_model_opt, train_data_01');
yp_test_opt = predict(svm_gauss_model_opt, test_data_01');

% miscalssification rate
train_perf_svm2_opt = SVM_classifier(yp_train_opt, train_labels_01);
rate_train_svm_opt = (train_perf_svm2_opt(2) + train_perf_svm2_opt(3)) / length(yp_train_opt);
test_perf_svm2_opt = SVM_classifier(yp_test_opt, test_labels_01);
rain_test_svm_opt = (test_perf_svm2_opt(2) + test_perf_svm2_opt(3)) / length(yp_test_opt);