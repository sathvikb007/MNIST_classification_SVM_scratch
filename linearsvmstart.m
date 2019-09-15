%% Data preprocessing
load('test_and_train.mat'); %imported from mnist_test.csv and mnist_train.csv

%% To separate 1 from the other digits

% %Since one vs all multiclassification is being used, one svm has to be
% trained by separating one digit from the rest. This is done by making a
% particular digit 1 and all other digits -1. The process is repeated for 
% all 10 digits

X1train = X_train;
y1train = Y_train;
y1train(y1train==1)=1; 
y1train(y1train ~= 1) = -1;

X1train=X1train(1:995,:);
y1train=y1train(1:995,:);

X1test=X_test;
y1test = Y_test;
y1test(y1test==1) = 1;
y1test(y1test ~= 1) = -1;

%% To find the resultant matrix by classifying 1s

c=1;
% C is a parameter of the SVC learner and is the penalty for misclassifying 
% a data point. When C is small, the classifier is okay with misclassified 
% data points (high bias, low variance). When C is large, the classifier is
% heavily penalized for misclassified data and therefore bends over 
% backwards avoid any misclassified data points (low bias, high variance).

[w1 , b1] = classify(X1train,y1train,c);
%The function classify returns the optimum hyperplane and takes the value
%of c

res1 = (X1test*w1 +b1);
%returns the predictions on the test data.

k = find( res>=0);
l = find( res<0);
temp = zeros(length(y1test),1);
temp(k) = 1;
temp(l) = -1;

%%  To separate 2s from the rest
X2train = X_train;
y2train = Y_train;
y2train(y2train ~= 2) = -1;
y2train(y2train==2)=1;

X2train=X2train(1:995,:);
y2train=y2train(1:995,:);

X2test=X_test;
y2test = Y_test;
% y2test(y2test==1) = 1;
% y2test(y2test ~= 1) = -1;

%%
[w2 , b2] = classify(X2train,y2train,c);
res2 = (X2test*w2 +b2);

%%  To separate 3s from the rest
X3train = X_train;
y3train = Y_train;
y3train(y3train ~= 3) = -1;
y3train(y3train==3)=1;

X3train=X3train(1:995,:);
y3train=y3train(1:995,:);

X3test=X_test;
y3test = Y_test;
% y3test(y3test==1) = 1;
% y3test(y3test ~= 1) = -1;

%%
[w3 , b3] = classify(X3train,y3train,c);
res3 = (X3test*w3 +b3);

%%  To separate 4s from the rest
X4train = X_train;
y4train = Y_train;
y4train(y4train ~= 4) = -1;
y4train(y4train==4)=1;

X4train=X4train(1:995,:);
y4train=y4train(1:995,:);

X4test=X_test;
y4test = Y_test;
% y3test(y3test==1) = 1;
% y3test(y3test ~= 1) = -1;

%%
[w4 , b4] = classify(X4train,y4train,c);
res4 = (X4test*w4 +b4);

%%  To separate 5s from the rest
X5train = X_train;
y5train = Y_train;
y5train(y5train ~= 5) = -1;
y5train(y5train==5)=1;

X5train=X5train(1:995,:);
y5train=y5train(1:995,:);

X5test=X_test;
y5test = Y_test;
% y3test(y3test==1) = 1;
% y3test(y3test ~= 1) = -1;

%%
[w5 , b5] = classify(X5train,y5train,c);
res5 = (X5test*w5 +b5);

%%  To separate 6s from the rest
X6train = X_train;
y6train = Y_train;
y6train(y6train ~= 6) = -1;
y6train(y6train==6)=1;

X6train=X6train(1:995,:);
y6train=y6train(1:995,:);

X6test=X_test;
y6test = Y_test;
% y3test(y3test==1) = 1;
% y3test(y3test ~= 1) = -1;

%%
[w6 , b6] = classify(X6train,y6train,c);
res6 = (X6test*w6 +b6);

%%  To separate 7s from the rest
X7train = X_train;
y7train = Y_train;
y7train(y7train ~= 7) = -1;
y7train(y7train==7)=1;

X7train=X7train(1:995,:);
y7train=y7train(1:995,:);

X7test=X_test;
y7test = Y_test;
% y3test(y3test==1) = 1;
% y3test(y3test ~= 1) = -1;

%%
[w7 , b7] = classify(X7train,y7train,c);
res7 = (X7test*w7 +b7);

%%  To separate 8s from the rest
X8train = X_train;
y8train = Y_train;
y8train(y8train ~= 8) = -1;
y8train(y8train==8)=1;

X8train=X8train(1:995,:);
y8train=y8train(1:995,:);

X8test=X_test;
y8test = Y_test;
% y3test(y3test==1) = 1;
% y3test(y3test ~= 1) = -1;

%%
[w8 , b8] = classify(X8train,y8train,c);
res8 = (X8test*w8 +b8);

%%  To separate 9s from the rest
X9train = X_train;
y9train = Y_train;
y9train(y9train ~= 9) = -1;
y9train(y9train==9)=1;

X9train=X9train(1:995,:);
y9train=y9train(1:995,:);

X9test=X_test;
y9test = Y_test;
% y3test(y3test==1) = 1;
% y3test(y3test ~= 1) = -1;

%%
[w9 , b9] = classify(X9train,y9train,c);
res9 = (X9test*w9 +b9);

%%  To separate 0s from the rest
X0train = X_train;
y0train = Y_train;
y0train(y0train ~= 0) = -1;
y0train(y0train==0)=1;

X0train=X0train(1:995,:);
y0train=y0train(1:995,:);

X0test=X_test;
y0test = Y_test;
% y3test(y3test==1) = 1;
% y3test(y3test ~= 1) = -1;

%%
[w0 , b0] = classify(X0train,y0train,c);
res0 = (X0test*w0 +b0);

%% To make final predictions

% % The predictions for each row are found by finding the largest value of
% theta' * x, i.e, highest value of res


largest=0;
yPred = zeros(10000,1);
for i=1:10000
    v = [ res0(i); res1(i); res2(i); res3(i); res4(i); res5(i); res6(i); res7(i); res8(i); res9(i)];

    largest = max(v);
    
        if ( largest==res1(i) )
        yPred(i)=1;
        
    
        elseif  (largest==res2(i) )
        yPred(i)=2;
      
    
        elseif ( largest==res3(i) )
        yPred(i)=3;
        
        
        elseif ( largest==res4(i) )
        yPred(i)=4;
        
        
        elseif ( largest==res5(i) )
        yPred(i)=5;
        
        
        elseif ( largest==res6(i) )
        yPred(i)=6;
        
        
        elseif ( largest==res7(i) )
        yPred(i)=7;
        
        
        elseif (largest==res8(i) )
        yPred(i)=8;
        
        
        elseif ( largest==res9(i) )
        yPred(i)=9;
       
        
        else
        yPred(i)=0;
        end
        
end
        
%%
accuracy = sum(yPred == Y_test) / length(Y_test);
    

    


