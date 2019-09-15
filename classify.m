

function [w,b] = classify(X,y,c)

%The functions returns the optimal hyperplane.
% The assumed support vectors are wx + b = 1
%                             and wx + b= -1
% The margin, which is 2/ |w| is to be maximized.
% Alternatively, the expression on line 23 is to be minimized

n=size(X,2);
m=size(X,1);
c=1000;

%c is a parameter that decides the tradeoff between the number of instances
%misclassified and the margin 

%e is the slack variable. The slack variable is a variable that measures 
% the distance of the point to its marginal hyperplane



cvx_begin
        variable w(n)
        variable b
        variable e(m)
        minimize( (norm(w)^2)/2 + c*sum(e) )
        subject to
            (y.*(( X*w )+b)) + e >= 1; 
            
            % y(w'x+b)>=1  is one constraint as points are supposed to lie
            % outside the margin of the support vectors
            
            e>=0; 
            % slack variable greater than 0
cvx_end

end

