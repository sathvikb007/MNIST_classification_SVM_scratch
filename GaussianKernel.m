function sim = GaussianKernel(x1, x2, sigma)
% returns a radial basis function kernel between x1 and x2
%   sim = gaussianKernel(x1, x2) returns a gaussian kernel between x1 and x2
%   and returns the value in sim

x1 = x1(:); x2 = x2(:);

sim = 0;

temp = sum((x1-x2).^2)/(2*sigma^2);

sim = exp(-temp);
end
