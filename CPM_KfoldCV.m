clear;
clc;
% ------------ INPUTS -------------------

all_mats  = load('matrices.mat');  % Connectivity matrices in a 268 x 268 x N .mat file 
all_behav = load('behavior.mat'); % Behavioral variable in a N x 1 .mat file

% Set p-value threshold for feature selection (i.e., 0.05, 0.01, etc.)
thresh = 0.05;

% Define the number of folds and iterations
K = 5;
iterations = 100;

% ---------------------------------------

no_sub = size(all_mats, 3);
no_node = size(all_mats, 1);

% Initialize arrays to store predicted behavior for each fold and iteration
behav_pred_pos = zeros(no_sub, K, iterations);
behav_pred_neg = zeros(no_sub, K, iterations);
behav_pred_both = zeros(no_sub, K, iterations);

% Initialize arrays to store masks for each fold and iteration
pos_mask_all = zeros(no_node, no_node, K, iterations);
neg_mask_all = zeros(no_node, no_node, K, iterations);

% Initialize arrays to store correlation coefficients and p-values
R_pos = zeros(K, iterations);
P_pos = zeros(K, iterations);
R_neg = zeros(K, iterations);
P_neg = zeros(K, iterations);
R_both = zeros(K, iterations);
P_both = zeros(K, iterations);

% Initialize arrays to store sum of predicted behavior for each iteration
behav_pred_pos_sum = zeros(no_sub, iterations);
behav_pred_neg_sum = zeros(no_sub, iterations);
behav_pred_both_sum = zeros(no_sub, iterations);

% Initialize arrays to store fit coefficients for each iteration (i.e., slope and intercept)
fit_pos_all = zeros(2, K * iterations); 
fit_neg_all = zeros(2, K * iterations); 
fit_both_all = zeros(2, K * iterations); 

% Connectome Based Predictive Modeling (CPM) with K-fold Cross-Validation
for iter = 1:iterations

    % Display iteration number
    fprintf('\n Iteration # %d', iter);

    % Generate indices for K-fold cross-validation
    cv_indices = crossvalind('Kfold', no_sub, K);

    % Loop over folds
    for fold = 1:K
        fprintf('\n Processing fold # %d', fold);

        test_indices = (cv_indices == fold);
        train_indices = ~test_indices;

        % Training data
        train_mats = all_mats(:, :, train_indices);
        train_behav = all_behav(train_indices);

        % Testing data
        test_mats = all_mats(:, :, test_indices);
        test_behav = all_behav(test_indices);

        % Feature selection: Correlate all edges with behavior
        train_vcts = reshape(train_mats, [], sum(train_indices));
        [r_mat, p_mat] = corr(train_vcts', train_behav);
        r_mat = reshape(r_mat, no_node, no_node);
        p_mat = reshape(p_mat, no_node, no_node);

        % Initialize masks for this fold
        pos_mask = zeros(no_node, no_node);
        neg_mask = zeros(no_node, no_node);

        % Set threshold and define masks
        pos_edges = find(r_mat > 0 & p_mat < thresh);
        neg_edges = find(r_mat < 0 & p_mat < thresh);

        pos_mask(pos_edges) = 1;
        neg_mask(neg_edges) = 1;

        % Store masks for this fold and iteration
        pos_mask_all(:, :, fold, iter) = pos_mask;
        neg_mask_all(:, :, fold, iter) = neg_mask;

        % Initialize arrays to store network sums for training data
        train_sumpos = zeros(sum(train_indices), 1);
        train_sumneg = zeros(sum(train_indices), 1);
        train_sumboth = zeros(sum(train_indices), 1);

        % Calculate network sums for training data
        for ss = 1:size(train_sumpos)
            train_sumpos(ss) = sum(sum(train_mats(:, :, ss) .* pos_mask)) / 2; % Divided by 2 to control for matrix symmetry
            train_sumneg(ss) = sum(sum(train_mats(:, :, ss) .* neg_mask)) / 2; % Divided by 2 to control for matrix symmetry
            train_sumboth(ss) = train_sumpos(ss) - train_sumneg(ss);
        end

        % Fit linear models using positive, negative, and combined edges
        fit_pos = polyfit(train_sumpos, train_behav, 1);
        fit_neg = polyfit(train_sumneg, train_behav, 1);
        fit_both = polyfit(train_sumboth, train_behav, 1);

        % Initialize arrays to store predicted behavior for testing data
        test_sumpos = zeros(sum(test_indices), 1);
        test_sumneg = zeros(sum(test_indices), 1);
        test_sumboth = zeros(sum(test_indices), 1);

        % Calculate network sums for testing data
        for ss = 1:size(test_sumpos)
            test_sumpos(ss) = sum(sum(test_mats(:, :, ss) .* pos_mask)) / 2;
            test_sumneg(ss) = sum(sum(test_mats(:, :, ss) .* neg_mask)) / 2;
            test_sumboth(ss) = test_sumpos(ss) - test_sumneg(ss);
        end

        % Predict behavior for testing data using the fitted models
        behav_pred_pos(test_indices, fold, iter) = fit_pos(1) * test_sumpos + fit_pos(2);
        behav_pred_neg(test_indices, fold, iter) = fit_neg(1) * test_sumneg + fit_neg(2);
        behav_pred_both(test_indices, fold, iter) = fit_both(1) * test_sumboth + fit_both(2);

        % Store fit coefficients for this fold and iteration
        fit_pos_all(:, (fold - 1) * iterations + iter) = fit_pos;
        fit_neg_all(:, (fold - 1) * iterations + iter) = fit_neg;
        fit_both_all(:, (fold - 1) * iterations + iter) = fit_both;
    end
    
    % Calculate correlation coefficients and p-values, these are preliminary. P-values are calculated after permutation testing.
    behav_pred_pos_sum(:, iter) = sum(behav_pred_pos(:, :, iter), 2);
    behav_pred_neg_sum(:, iter) = sum(behav_pred_neg(:, :, iter), 2);
    behav_pred_both_sum(:, iter) = sum(behav_pred_both(:, :, iter), 2);

    [R_pos(:, iter), P_pos(:, iter)] = corr(behav_pred_pos_sum(:, iter), all_behav, 'type','Spearman');
    [R_neg(:, iter), P_neg(:, iter)] = corr(behav_pred_neg_sum(:, iter), all_behav, 'type','Spearman');
    [R_both(:, iter), P_both(:, iter)] = corr(behav_pred_both_sum(:, iter), all_behav, 'type','Spearman');
end

% Average correlation coefficients across iterations and folds
average_R_pos = mean(R_pos, 1); % Contains one rho value per iteration
average_R_neg = mean(R_neg, 1); % Contains one rho value per iteration
average_R_both = mean(R_both, 1); % Contains one rho value per iteration

fprintf('\n Average R value for positive edges: %f', mean(average_R_pos));
fprintf('\n Average R value for negative edges: %f', mean(average_R_neg));
fprintf('\n Average R value for both edges: %f', mean(average_R_both));

% Average p-values across iterations and folds
average_P_pos = mean(P_pos, 1); % Contains one p-value per iteration
average_P_neg = mean(P_neg, 1); % Contains one p-value per iteration
average_P_both = mean(P_both, 1); % Contains one p-value per iteration

fprintf('\n Average P value for positive edges: %f', mean(average_P_pos));
fprintf('\n Average P value for negative edges: %f', mean(average_P_neg));
fprintf('\n Average P value for both edges: %f', mean(average_P_both));

% ------------ OUTPUTS -------------------

% Save workspace in a .mat file 
save('workspace.mat');

% ---------------------------------------



















