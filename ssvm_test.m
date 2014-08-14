function ssvm_test
% A demo function for ssvm_learn. 
% It shows how to use the nslack_ssvm or 1slack_ssvm to learn a standard 
% linear SVM.

% load data
load q1x.dat
load q1y.dat

% define variables
X = q1x;
Y = 2*(q1y-0.5);

[train_num, feature_num] = size(X);

patterns = {} ;
labels = {} ;
for i = 1:train_num
    patterns{i} = X(i,:) ;
    labels{i}   = Y(i);
end

param.patterns = patterns ;
param.labels = labels ;
param.lossFn = @lossCB ;
param.constraintFn  = @constraintCB ;
param.featureFn = @featureCB ;
param.dimension = feature_num;

C = 100;
args = sprintf('-c %g -e 0.01',C);

% model = ssvm_nslack_learn(args, param) ;
model = ssvm_1slack_learn(args, param) ;

w = model.w;

% reporting accuracy
t_num = sum(sign(X * w) == Y);
accuracy = 100 * t_num / train_num;
fprintf('Accuracy on training set is %.4f %%\n', accuracy);

% visualize
figure(1), clf
xp = linspace(min(X(:,1)), max(X(:,1)), 100);
yp = - w(1) * xp / w(2);
yp1 = - (w(1)*xp - 1) / w(2); % margin boundary for support vectors for y=1
yp0 = - (w(1)*xp + 1) / w(2); % margin boundary for support vectors for y=0

% index of negative samples
idx0 = find(q1y==0);
% index of positive samples
idx1 = find(q1y==1);

plot(q1x(idx0, 1), q1x(idx0, 2), 'rx'); 
hold on
plot(q1x(idx1, 1), q1x(idx1, 2), 'go');
plot(xp, yp, '-b', xp, yp1, '--g', xp, yp0, '--r');
hold off
title(sprintf('decision boundary for a linear SVM classifier with C = %g', C));

end

% ------------------------------------------------------------------
% Callback functions                                        
% ------------------------------------------------------------------

function delta = lossCB(param, y, ybar)
  delta = double(y ~= ybar) ;
end

function psi = featureCB(param, x, y)
  psi = y*x/2 ;
end

function yhat = constraintCB(param, model, x, y)
% margin rescaling: argmax_y delta(yi, y) + <psi(x,y), w>
  if dot(y*x, model.w) > 1, yhat = y ; else yhat = - y ; end
end

