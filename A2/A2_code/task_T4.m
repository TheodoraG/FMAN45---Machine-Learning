clear;
close all;
clc;

%% task t4 
x = linspace(-3.5, 4.5);
y = (2/3)*x.^2 - (5/3); % g(x)

x2 = [-3 -2 -1 0 1 2 4];
y2 = [1 1 -1 -1 -1 1 1];

figure()
scatter(x2,y2,'LineWidth',2.5)
hold on
plot(x,y,'LineWidth',2.5);
hold off
legend("Data points (second dataset)", "g(x)")
xlabel("x")
ylabel("y")