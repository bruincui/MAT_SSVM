function model = ssvm_nslack_learn(args,param)
% It solves n-slack structural SVM with margin-rescaling.
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
model.xi = zeros(size2train, 1); % slack variables

% Constrcuting working sets
% For fast computation, we directly store feature map defferences and
% margins in each working set
fdiffs = cell(size2train, 1); % feature map different of active constains
margins = cell(size2train, 1); % loss of active constraints

expandstep = 25;
for i = 1:size2train
    fdiffs{i} = zeros(param.dimension, expandstep);
    margins{i} = zeros(1, expandstep);
end

% the number of active constraints in each working set
activeCons_num = zeros(size2train, 1);

iter = 1;
max_iter = 1000;
iterFlag = 1;
while (iter <= max_iter && iterFlag)
    fprintf('#%d iteration: ', iter)
    iterFlag = 0;
    
    for i = 1:size2train
        
        x_i = param.patterns{i};
        y_i = param.labels{i};
        
        % find the most violated constraint
        yhat = param.constraintFn(param, model, x_i, y_i);
        fprintf('+') % mark printing
        
        fd = param.featureFn(param, x_i,y_i) - param.featureFn(param, x_i,yhat);
        loss = param.lossFn(param, y_i, yhat);
        cost = loss - dot(model.w, fd);
        
        if cost > model.xi(i) + model.eps
            iterFlag = 1;
                        
            if activeCons_num(i) + 1 > size(fdiffs{i},2)
                % expanding the current working set
                fdiffs{i} = [fdiffs{i} zeros(param.dimension, expandstep)];
                margins{i} = [margins{i} zeros(1, expandstep)];
            end
            
            activeCons_num(i) = activeCons_num(i) + 1;
            fdiffs{i}(:, activeCons_num(i)) = fd;
            margins{i}(activeCons_num(i)) = loss;
            
            % Here we solve the QP problem by CVX or Pegasos algorithm
            
            %{
            % Option 1: Solving QP with cvx
            cvx_begin quiet
                variable w(param.dimension);
                variable xi(size2train) nonnegative;
                
                minimize( 0.5*w'*w + (model.C/size2train)*sum(xi));
                subject to
                    for j = 1 : size2train
                        w' * fdiffs{j}(:, 1:activeCons_num(j)) >= ...
                            margins{j}(:, 1:activeCons_num(j)) - xi(j);
                    end
            cvx_end
            % end of cvx
            %}
            
            
            % Option 2: Solving QP with Pegasos algorithm
            lambda = 1 / model.C;
            % we restart the optimizer from current w
            [w,xi] = ssvm_nslack_pegasos(fdiffs,margins,activeCons_num,lambda,model.w);
            % end of Pegasos
                        
            model.w(:) = w(:);
            model.xi(:) = xi(:);
            
            fprintf('*') % mark printing
        end
        fprintf(' ') % mark printing
    end
    
    iter = iter + 1;
    fprintf('\n\n')
end


end

