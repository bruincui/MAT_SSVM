function [w,xi] = ssvm_nslack_pegasos( feature,margin,active_num,lambda,w0,maxIter,k )
% The modified pegasos algorithm for solving QP in n-slack structured svm
%
% Input:
% feature: n*1 cell, n is the number of training samples, and each cell
% contains active constraints for the corresponding training sample;
% margin: n*1 cell containing loss with corresponding training sample;
% active_num: n*1 vector representing the number of active constraints 
% in each training sample;
% w0: the initial value of w, it's a column vector;
% lambda, k: parameters in Pegasos algorithm (default: lambda=1, k=0.1*n);
% maxIter: maximum number of iterations for w convergence; (default:
% 10000); Training stops if maxIter is satisfied;
% 
% Output:
% w: weight vector in SVM primal problem;
% xi: n*1 vector representing the slack variable for each training sample;
% 
% References:
% [1] Pegasos-Primal Estimated sub-Gradient SOlver for SVM

n = length(feature);
if(length(margin) ~= n)
    error('Number of samples in feature and margin matrix must be same!');
end

d = size(feature{1},1);

if(nargin<4 || isempty(lambda)),  lambda = 1;  end
if(nargin<5), w0 = []; end
if(nargin<6 || isempty(maxIter)),   maxIter = 10000;  end
if(nargin<7 || isempty(k)), k = ceil(0.1*n);    end

if k > n
    k = n;
end

% intialization
if isempty(w0)
    w = zeros(d,1);
else
    if any(size(w0) ~= [d,1])
        error('The dimension of initial value is not correct!');
    else
        w = w0;
    end
end

for t = 1:maxIter
    % generating indexes uniformly at random without repetitions
    idx = randperm(n);
    idx = idx(1:k);
    
    subgradient = zeros(d,1);
    for i = 1:k
        j = idx(i);
        if active_num(j) ~= 0
            % choose the most violated constraint for current sample
            dis = margin{j}(:, 1:active_num(j)) - w' * feature{j}(:, 1:active_num(j));
            [mvc_value, mvc_idx] = max(dis);
            if mvc_value > 0
                subgradient = subgradient + feature{j}(:,mvc_idx);
            end
        end
    end
    
    eta_t = 1 / (lambda * t);
    
    % update w without projection
    w = (1 - 1/t) * w + (eta_t / k) * subgradient;
end

xi = zeros(n,1);
for j = 1:n
    if active_num(j) == 0
        xi(j) = 0;
    else
        dis = margin{j}(:, 1:active_num(j)) - w' * feature{j}(:, 1:active_num(j));
        mvc_value = max(dis);
        xi(j) = max([0 mvc_value]);
    end
end


end

