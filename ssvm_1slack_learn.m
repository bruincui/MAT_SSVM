function model = ssvm_1slack_learn( args,param )
% It solves 1-slack structural SVM with margin-rescaling.
%
% References:
% [1] T. Joachims, T. Finley, Chun-Nam Yu, Cutting-Plane Training of Structural
%        SVMs, Machine Learning Journal.


% Setting default values of parameters
model.C = 0.01; % regularization term
model.eps = 0.1; % precision


% Handling parameters
args = strread(args, '%s');
for i = 1:2:size(args,1)
    str = args{i};
    if strcmp(str,'-c')
        model.C = str2num(args{i+1});
    elseif strcmp(str,'-e')
        model.eps = str2num(args{i+1});
    else
        error('Unknown paramters !')
    end
end

fprintf('Input parameter c is %4.2e, and e is %g.\n', model.C, model.eps);

% Initialization
if(length(param.patterns) ~= length(param.labels))
    error('Number of samples in patterns and labels must be same!');
end
size2train = length(param.patterns);
fprintf('There are %d training examples\n\n', size2train);

model.w = zeros(param.dimension, 1); % weight vector
model.xi = 0; % slack variables

% Constrcuting working sets
% For fast computation, we directly store feature map defferences and
% margins in the working set

expandstep = 25;
% feature map different of active constains
fdiffs = zeros(param.dimension, expandstep);
% loss of active constraints
margins = zeros(1, expandstep); 

% the number of active constraints in the working set
activeCons_num = 0;

iter = 1;
max_iter = 1000;
iterFlag = 1;
while (iter <= max_iter && iterFlag)
    fprintf('#%d iteration: ', iter)
    iterFlag = 0;
    
    fd = zeros(param.dimension, 1);
    loss = 0;
    for i = 1:size2train
        
        x_i = param.patterns{i};
        y_i = param.labels{i};
        
        % find the most violated constraint
        yhat = param.constraintFn(param, model, x_i, y_i);
        if ~mod(i, 10)
            fprintf('.') % mark printing
        end
        
        fd = fd + vec(param.featureFn(param, x_i,y_i) - param.featureFn(param, x_i,yhat));
        loss = loss + param.lossFn(param, y_i, yhat);
    end
    
    fprintf(' ') % mark printing
    
    fd = fd / size2train;
    loss = loss / size2train;
    cost = loss - dot(model.w, fd);
    
    if cost > model.xi + model.eps
            iterFlag = 1;
            
            if activeCons_num + 1 > size(fdiffs,2)
                % expanding the working set
                fdiffs = [fdiffs zeros(param.dimension, expandstep)];
                margins = [margins zeros(1, expandstep)];
            end
            
            activeCons_num = activeCons_num + 1;
            fdiffs(:, activeCons_num) = fd;
            margins(activeCons_num) = loss;
            
            %{
            % Option 1: Solving QP with cvx
            cvx_begin quiet
                variable w(param.dimension);
                variable xi nonnegative;
                
                minimize( 0.5*w'*w + model.C*xi);
                subject to
                    w' * fdiffs(:, 1:activeCons_num) >= ...
                        margins(:, 1:activeCons_num) - xi;
            cvx_end
            % end of cvx
            %}
            
            
            % Option 2: Solving QP with Pegasos algorithm
            lambda = 1 / model.C;
            % we restart the optimizer from current w
            [w,xi] = ssvm_1slack_pegasos(fdiffs,margins,activeCons_num,lambda,model.w);
            % end of Pegasos
            
            
            model.w(:) = w(:);
            model.xi(:) = xi(:);
            
            fprintf('*') % mark printing
    end
    
    iter = iter + 1;
    fprintf('\n\n')
end

end

