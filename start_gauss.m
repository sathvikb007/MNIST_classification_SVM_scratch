%% Data preprocessing
load('test_and_train.mat'); 
%to load training and test data

%% To normalize the feature
X_train= normalize(X_train,'range'); 
%normalize features between 0 and 1

%% Separating 1s from the rest
X1train = X_train;   

y1train = Y_train;
y1train(y1train==1)=1;
y1train(y1train ~= 1) = 0;

% Since one vs all multiclassification is being used, one svm has to be
% trained by separating one digit from the rest. This is done by making a
% particular digit 1 and all other digits 0. The process is repeated for all
% 10 digits

X1train=X1train(1:1000,:);
y1train=y1train(1:1000,:);

X1test = X_test;
y1test = Y_test;
y1test(y1test==1) = 1;
y1test(y1test ~= 1) = 0;
    
%% To get the f matrix for 1s

%The matrix f is the matrix that holds the Radial Basis Function(RBF)
% or Guassian relations between different rows of the training data

sigma = 5; 

%Sigma is a parameter of the RBF kernel that decides the spread
% of the kernel and therefore the decision region. When gamma is low, the
%curve of the decision boundary is very low and thus the decision region 
%is very broad. When gamma is high, the curve of the decision boundary is
%high

m = size(X1train,1); 
n = size(X1train,2);
f1 = ones(m,m);

for i=1:m
    for j=1:m    
        f1(i,j) = GaussianKernel( X1train(i,:) ,  X1train(j,:) , sigma); 
    end
end

%% minimize theta1

c=10;
% C is a parameter of the SVC learner and is the penalty for misclassifying 
% a data point. When C is small, the classifier is okay with misclassified 
% data points (high bias, low variance). When C is large, the classifier is
% heavily penalized for misclassified data and therefore bends over 
% backwards avoid any misclassified data points (low bias, high variance).
theta1 = zeros(m,1);
theta1 = zeros(m,1);

theta1 = optimvar('theta1', m);
prob = optimproblem;
prob.Objective = c* ( sum ( y1train.* ( f1*theta1 ))  + sum ((1.-y1train).*(f1*theta1)) )   + sum(theta1.^2) ;
%The above objective shoud be minimized to obtain the weights vector theta
sol = solve(prob);

%% To store results of f*theta1
theta1 = double(sol.theta1);
res1 = (f1*theta1)./1000; 

%% Separating 2s from the rest
X2train = X_train;
y2train = Y_train;
y2train(y2train ~= 2) = 0;
y2train(y2train==2)=1;

X2train=X2train(1:1000,:);
y2train=y2train(1:1000,:);

X2test = X_test;
y2test = Y_test;
y2test(y2test~=2) = 1;
y2test(y2test ~= 2) = 0;

%% To get the f matrix for 2s

sigma = 5;
m = size(X2train,1); 
n = size(X2train,2);
f2 = ones(m,m);

for i=1:m
    for j=1:m    
        f2(i,j) = GaussianKernel( X2train(i,:) ,  X2train(j,:) , sigma); 
    end
end

%% minimize theta2

c=10;
theta2 = zeros(m,1);
theta2 = zeros(m,1);

theta2 = optimvar('theta2', m);
prob = optimproblem;
prob.Objective = c* ( sum ( y2train.* ( f2*theta2 ))  + sum ((1.-y2train).*(f2*theta2)) )   + sum(theta2.^2) ;
sol = solve(prob);

%% To store results of f*theta2
theta2 = double(sol.theta2);
res2 = (f2*theta2)./1000; 

%% Separating 3s from the rest
X3train = X_train;
y3train = Y_train;
y3train(y3train ~= 3) = 0;
y3train(y3train==3)=1;

X3train=X3train(1:1000,:);
y3train=y3train(1:1000,:);

X3test = X_test;
y3test = Y_test;
y3test(y3test~=3) = 1;
y3test(y3test ~= 3) = 0;

%% To get the f matrix for 3s

sigma = 5;
m = size(X3train,1); 
n = size(X3train,2);
f3 = ones(m,m);

for i=1:m
    for j=1:m    
        f3(i,j) = GaussianKernel( X3train(i,:) ,  X3train(j,:) , sigma); 
    end
end

%% minimize theta3

c=10;
theta3 = zeros(m,1);
theta3 = zeros(m,1);

theta3 = optimvar('theta3', m);
prob = optimproblem;
prob.Objective = c* ( sum ( y3train.* ( f3*theta3))  + sum ((1.-y3train).*(f3*theta3)) )   + sum(theta3.^2) ;
sol = solve(prob);

%% To store results of f*theta3
theta3 = double(sol.theta3);
res3 = (f3*theta3)./1000; 

%% Separating 4s from the rest
X4train = X_train;
y4train = Y_train;
y4train(y4train ~= 4) = 0;
y4train(y4train==4)=1;

X4train=X4train(1:1000,:);
y4train=y4train(1:1000,:);

X4test = X_test;
y4test = Y_test;
y4test(y4test~=4) = 1;
y4test(y4test ~= 4) = 0;

%% To get the f matrix for 4s

sigma = 5;
m = size(X4train,1); 
n = size(X4train,2);
f4 = ones(m,m);

for i=1:m
    for j=1:m    
        f4(i,j) = GaussianKernel( X4train(i,:) ,  X4train(j,:) , sigma); 
    end
end

%% minimize theta4

c=10;
theta4 = zeros(m,1);
theta4 = zeros(m,1);

theta4 = optimvar('theta4', m);
prob = optimproblem;
prob.Objective = c* ( sum ( y4train.* ( f4*theta4))  + sum ((1.-y4train).*(f4*theta4)) )   + sum(theta4.^2) ;
sol = solve(prob);

%% To store results of f*theta4
theta4 = double(sol.theta4);
res4 = (f4*theta4)./1000; 

%% Separating 5s

X5train = X_train;
y5train = Y_train;
y5train(y5train ~= 5) = 0;
y5train(y5train==5)=1;

X5train=X5train(1:1000,:);
y5train=y5train(1:1000,:);

X5test = X_test;
y5test = Y_test;
y5test(y5test~=5) = 1;
y5test(y5test ~= 5) = 0;

%% To get the f matrix for 5s

sigma = 5;
m = size(X5train,1); 
n = size(X5train,2);
f5 = ones(m,m);

for i=1:m
    for j=1:m    
        f5(i,j) = GaussianKernel( X5train(i,:) ,  X5train(j,:) , sigma); 
    end
end

%% minimize theta5

c=10;
theta5 = zeros(m,1);
theta5 = zeros(m,1);

theta5 = optimvar('theta5', m);
prob = optimproblem;
prob.Objective = c* ( sum ( y5train.* ( f5*theta5))  + sum ((1.-y5train).*(f5*theta5)) )   + sum(theta5.^2) ;
sol = solve(prob);

%% To store results of f*theta5
theta5 = double(sol.theta5);
res5 = (f5*theta5)./1000; 

%% To separate 6s
X6train = X_train;
y6train = Y_train;
y6train(y6train ~= 6) = 0;
y6train(y6train==6)=1;

X6train=X6train(1:1000,:);
y6train=y6train(1:1000,:);

X6test = X_test;
y6test = Y_test;
y6test(y6test~=6) = 0;
y6test(y6test == 6) = 1;

%% To get the f matrix for 6s

sigma = 5;
m = size(X6train,1); 
n = size(X6train,2);
f6 = ones(m,m);

for i=1:m
    for j=1:m    
        f6(i,j) = GaussianKernel( X6train(i,:) ,  X6train(j,:) , sigma); 
    end
end

%% minimize theta6

c=10;
theta6 = zeros(m,1);
theta6 = zeros(m,1);

theta6 = optimvar('theta6', m);
prob = optimproblem;
prob.Objective = c* ( sum ( y6train.* ( f6*theta6))  + sum ((1.-y6train).*(f6*theta6)) )   + sum(theta6.^2) ;
sol = solve(prob);

%% To store results of f*theta6
theta6 = double(sol.theta6);
res6 = (f6*theta6)./1000; 

%% To separate 7s
X7train = X_train;
y7train = Y_train;
y7train(y7train ~= 7) = 0;
y7train(y7train==7)=1;

X7train=X7train(1:1000,:);
y7train=y7train(1:1000,:);

X7test = X_test;
y7test = Y_test;
y7test(y7test~=7) = 0;
y7test(y7test == 7) = 1;

%% To get the f matrix for 7s

sigma = 5;
m = size(X7train,1); 
n = size(X7train,2);
f7 = ones(m,m);

for i=1:m
    for j=1:m    
        f7(i,j) = GaussianKernel( X7train(i,:) ,  X7train(j,:) , sigma); 
    end
end

%% minimize theta7

c=10;
theta7 = zeros(m,1);
theta7 = zeros(m,1);

theta7 = optimvar('theta7', m);
prob = optimproblem;
prob.Objective = c* ( sum ( y7train.* ( f7*theta7))  + sum ((1.-y7train).*(f7*theta7)) )   + sum(theta7.^2) ;
sol = solve(prob);

%% To store results of f*theta7
theta7 = double(sol.theta7);
res7 = (f7*theta7)./1000; 

%% Separting 8s

X8train = X_train;
y8train = Y_train;
y8train(y8train ~= 8) = 0;
y8train(y8train==8)=1;

X8train=X8train(1:1000,:);
y8train=y8train(1:1000,:);

X8test = X_test;
y8test = Y_test;
y8test(y8test~=8) = 0;
y8test(y8test == 8) = 1;

%% To get the f matrix for 8s

sigma = 5;
m = size(X8train,1); 
n = size(X8train,2);
f8 = ones(m,m);

for i=1:m
    for j=1:m    
        f8(i,j) = GaussianKernel( X8train(i,:) ,  X8train(j,:) , sigma); 
    end
end

%% minimize theta8

c=10;
theta8 = zeros(m,1);
theta8 = zeros(m,1);

theta8 = optimvar('theta8', m);
prob = optimproblem;
prob.Objective = c* ( sum ( y8train.* ( f8*theta8))  + sum ((1.-y8train).*(f8*theta8)) )   + sum(theta8.^2) ;
sol = solve(prob);

%% To store results of f*theta8
theta8 = double(sol.theta8);
res8 = (f8*theta8)./1000; 

%% To separate 9s

X9train = X_train;
y9train = Y_train;
y9train(y9train ~= 9) = 0;
y9train(y9train==9)=1;

X9train=X9train(1:1000,:);
y9train=y9train(1:1000,:);

X9test = X_test;
y9test = Y_test;
y9test(y9test~=9) = 0;
y9test(y9test == 9) = 1;

%% To get the f matrix for 9s

sigma = 5;
m = size(X9train,1); 
n = size(X9train,2);
f9 = ones(m,m);

for i=1:m
    for j=1:m    
        f9(i,j) = GaussianKernel( X9train(i,:) ,  X9train(j,:) , sigma); 
    end
end

%% minimize theta9

c=10;
theta9 = zeros(m,1);
theta9 = zeros(m,1);

theta9 = optimvar('theta9', m);
prob = optimproblem;
prob.Objective = c* ( sum ( y9train.* ( f9*theta9))  + sum ((1.-y9train).*(f9*theta9)) )   + sum(theta9.^2) ;
sol = solve(prob);

%% To store results of f*theta9
theta9 = double(sol.theta9);
res9 = (f9*theta9)./1000; 

%% To separate 0s

X0train = X_train;
y0train = Y_train;
y0train(y0train ~= 0) = 0;
y0train(y0train==0)=1;

X0train=X0train(1:1000,:);
y0train=y0train(1:1000,:);

X0test = X_test;
y0test = Y_test;
y0test(y0test~=0) = 0;
y0test(y0test == 0) = 1;

%% To get the f matrix for 0s

sigma = 5;
m = size(X0train,1); 
n = size(X0train,2);
f0 = ones(m,m);

for i=1:m
    for j=1:m    
        f0(i,j) = GaussianKernel( X0train(i,:) ,  X0train(j,:) , sigma); 
    end
end

%% minimize theta0

c=10;
theta0 = zeros(m,1);
theta0 = zeros(m,1);

theta0 = optimvar('theta0', m);
prob = optimproblem;
prob.Objective = c* ( sum ( y0train.* ( f0*theta0))  + sum ((1.-y0train).*(f0*theta0)) )   + sum(theta0.^2) ;
sol = solve(prob);

%% To store results of f*theta0
theta0 = double(sol.theta0);
res0 = (f0*theta0)./1000; 

%% To find the final predictions

% The predictions for each row are found by finding the largest value of
% theta' * x, i.e, highest value of res

largest=0;
yPred = zeros(1000,1);
for i=1:1000
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


accuracy = sum(yPred == Y_train(1:1000,:)) / length(Y_train(1:1000,:));




    

