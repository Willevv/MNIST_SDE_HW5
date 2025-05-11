clear;
clear all
rng(1234);
J = 6000;
[train_imgs, train_labels] = readMNIST('train-images.idx3-ubyte', 'train-labels.idx1-ubyte', J, 0);
[test_imgs, test_labels] = readMNIST('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte', 10000, 0);
train_imgs = reshape(train_imgs, [400,J]);
test_imgs = reshape(test_imgs, [400,10000]);
Y = OneHOT(train_labels);
% 
% % For problem 3
lambda = 0.01;
Ki = [2,4,8,16,32,64,128,256,512,1024];
persentage = zeros(1,length(Ki));
% 
% For problem 4
%lambdas = [0.0001, 0.001, 0.01, 0.1, 1, 10];
%persentage = zeros(1, length(lambdas));
% 
 i = 1;
for K = Ki % task 3
%for lambda = lambdas % task 4
    Sigma = 0.1^2*eye(400);
    W = mvnrnd(zeros(1,400),Sigma,K);
    %W = unifrnd(-10,10, K, 400);
    D = 1i*W*train_imgs;
    S = exp(1i*W*train_imgs)';
    A = S'*S;
    A1 = J*eye(K);
    beta = (S'*S+lambda*J*eye(K))\S'*Y; %Solving Least square problem

     S_test = exp(1i*W*test_imgs)';
     Y_pred = S_test*beta;
     [~, index] = max(Y_pred, [] ,2); %index for max value along 2nd dimension
     
     predicted_labels = index - 1; 
     
     persentage(i) = sum((predicted_labels-test_labels) == 0)/10000; %Find out how big percentage was correct classified
     
     i = i+1;
end
% % 
loglog(Ki, (1-persentage)*100, 'bx', Ki, 100*(Ki.^(-0.5)), 'r')
xlabel('K')
ylabel('Percent misstaken on the test set')
legend('Random Fourier Features','100*K^{(-0.5)}')
title('Generalization error of RFF on MNIST')

% 
% loglog(lambdas, (1-persentage), 'bx--')
% xlabel('\lambda')
% ylabel('error')
% title('Generalization error of RFF on MNIST')
