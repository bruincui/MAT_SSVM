function [ w,xi ] = ssvm_1slack_pegasos( feature,margin,active_num,lambda,w0,maxIter)
% A modified Pegasos algorithm (which actually is a standard subgradient 
% method) for solving QP in 1-slack structured svm
%
% Input:
% feature: d*m matrix containing active constraints in working set, where d 
% is the dimension of feature representation;
% margin: 1*m vector containing loss with corresponding active constraint;
% active_num: the number of active constraints;
% w0: the initial value of w, it's a column vector;
% lambda: parameters in Pegasos algorithm (default: lambda=1);
% maxIter: maximum number of iterations for w convergence; (default:
% 10000); Training stops if maxIter is satisfied;
% 
% Output:
% w: weight vector in SVM primal problem;
% xi: the slack variable;
% 
% References:
% [1] Pegasos-Primal Estimated sub-Gradient SOlver for SVM


d = size(feature,1);

if(nargin<4 || isempty(lambda)),  lambda = 1;  end
if(nargin<5), w0 = []; end
if(nargin<6 || isempty(maxIter)),   maxIter = 10000;  end

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
    
    subgradient = zeros(d,1);
    dis = margin(:, 1:active_num) - w' * feature(:, 1:active_num);
    [mvc_value, mvc_idx] = max(dis);
    if mvc_value > 0
        subgradient = subgradient + feature(:,mvc_idx);
    end
    
    eta_t = 1 / (lambda * t);
    % update w without projection
    w = (1 - 1/t) * w + eta_t * subgradient;
end

dis = margin(:, 1:active_num) - w' * feature(:, 1:active_num);
mvc_value = max(dis);
xi = max([0, mvc_value]);
    
end




