clear;
close all;
clc;

%% task 4
load A1_data.mat

lambda = [0.1, 10, 2];
nonZeroCoords = [];

figure()
hold on
w_est = skeleton_lasso_ccd(t,X,lambda(1));
%interpolated reconstruction of data
plot(ninterp,Xinterp*w_est,'linewidth',2.5,'Color',"#0072BD");
%reconstructed data points
plot(n,X*w_est,'o','MarkerSize',8,'linewidth',2.5,'Color','#D95319');
nonZeroCoords(1) = sum(w_est~=0);
%the original plot
plot(n, t,'o','MarkerFaceColor', 'k', 'MarkerSize', 6);
legend('reconstruction of the data','reconstructed data points','original')
title('\lambda = 0.1')
xlabel('time')
ylabel('data points')


figure()
hold on
w_est = skeleton_lasso_ccd(t,X,lambda(2));
%interpolated reconstruction of data
plot(ninterp,Xinterp*w_est,'linewidth',2.5,'Color',"#77AC30");
%reconstructed data points
plot(n,X*w_est,'o','MarkerSize',8,'linewidth',2.5,'Color','#D95319');
nonZeroCoords(2) = sum(w_est~=0);
%the original plot
plot(n, t,'o','MarkerFaceColor', 'k', 'MarkerSize', 6);
legend('reconstruction of the data','reconstructed data points','original')
title('\lambda = 10')
xlabel('time')
ylabel('data points')


figure()
hold on
w_est = skeleton_lasso_ccd(t,X,lambda(3));
%interpolated reconstruction of data
plot(ninterp,Xinterp*w_est,'linewidth',2.5,'Color',"#A2142F");
%reconstructed data points
plot(n,X*w_est,'o','MarkerSize',8,'linewidth',2.5,'Color','#D95319');
nonZeroCoords(3) = sum(w_est~=0);
%the original plot
plot(n, t,'o','MarkerFaceColor', 'k', 'MarkerSize', 6);
legend('reconstruction of the data','reconstructed data points','original')
title('\lambda = 2')
xlabel('time')
ylabel('data points')


%% task 5
lambda_min = 0.1;
lambda_max = 10;
N_lambda = 5;
lambda_grid = exp(linspace(log(lambda_min), log(lambda_max)));
[wopt,lambdaopt,RMSEval,RMSEest] = skeleton_lasso_cv(t,X,lambda_grid,N_lambda);

figure()
hold on
plot(log(lambda_grid), RMSEval,'linewidth', 2.5);
plot(log(lambda_grid), RMSEest,'linewidth', 2.5);
xline(log(lambdaopt),'--','linewidth',1.5);
legend('RMSE validation','RMSE estimation', '\lambda optimal');
xlabel('log(\lambda)')

figure()
hold on
%interpolated reconstruction of data
plot(ninterp,Xinterp*wopt,'linewidth',2.5,'Color','#7E2F8E');
%reconstructed data points
plot(n,X*wopt,'o','MarkerSize',8,'linewidth',2.5,'Color','#D95319');
%the original plot
plot(n, t,'o','MarkerFaceColor', 'k', 'MarkerSize', 6);
legend('reconstruction of the data','reconstructed data points','original')
xlabel('time')
ylabel('data points')


%% task 6
lambda_min = 0.001;
lambda_max = 0.04;
N_lambda = 5;
lambda_grid = exp(linspace(log(lambda_min), log(lambda_max),50));
[wopt,lambdaopt,RMSEval,RMSEest] = skeleton_multiframe_lasso_cv(Ttrain, Xaudio,lambda_grid,N_lambda);

figure()
hold on
plot(log(lambda_grid), RMSEval,'linewidth', 2.5);
plot(log(lambda_grid), RMSEest,'linewidth', 2.5);
xline(log(lambdaopt),'--','linewidth',1.5);
legend('RMSE validation','RMSE estimation', '\lambda optimal');
xlabel('log(\lambda)')

figure()
hold on
plot((lambda_grid), RMSEval,'linewidth', 2.5);
plot((lambda_grid), RMSEest,'linewidth', 2.5);
xline((lambdaopt),'--','linewidth',1.5);
legend('RMSE validation','RMSE estimation', '\lambda optimal');
xlabel('\lambda')

%% task 7 
%lambdaopt = 0.0049;
Ytest = lasso_denoise(Ttest, Xaudio, lambdaopt);
soundsc(Ytest, fs)
save('denoised_audio','Ytest','fs')

%% task 7 -other values
lambda_new = 0.01; %almost no noise
Ytest = lasso_denoise(Ttest, Xaudio, lambda_new);
soundsc(Ytest, fs)

%% task 7 -other values - b
lambda_new = 0.02; %some strange sounds (the piano begins to hear less)
Ytest = lasso_denoise(Ttest, Xaudio, lambda_new);
soundsc(Ytest, fs)

%% task 7 -other values - c
lambda_new = 0.05; %the piano sound fades away
Ytest = lasso_denoise(Ttest, Xaudio, lambda_new);
soundsc(Ytest, fs)

%% task 7 -other values - d
lambda_new = 0.002; %more noise
Ytest = lasso_denoise(Ttest, Xaudio, lambda_new);
soundsc(Ytest, fs)

%% task 7 -other values - e
lambda_new = 0.0005; %even more noise
Ytest = lasso_denoise(Ttest, Xaudio, lambda_new);
soundsc(Ytest, fs)

%% task 7 -other values - f
lambda_new = 0.00008; %the noise hears better than the piano
Ytest = lasso_denoise(Ttest, Xaudio, lambda_new);
soundsc(Ytest, fs)