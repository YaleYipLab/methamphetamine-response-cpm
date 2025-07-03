%%%%%%%%%%%%%% Leave-One-In Lesion Analysis for CPM %%%%%%%%%%%%%%

clc;
clear;

%%%% INPUTS %%%%

% ------------ CPM -------------------

all_mats  = load('matrices.mat');  % Connectivity matrices in a 268 x 268 x N .mat file 
all_behav = load('behavior.mat'); % Behavioral variable in a N x 1 .mat file

% threshold for feature selection
thresh = 0.05;

% Define the number of folds and iterations
K = 5;
iterations = 100;

% ----------- Node/Network Definitions ------------

% Read node number and network affiliations from network_definitions.txt
network_definitions = readtable('network_definition.txt', 'Delimiter', '\t');
node_numbers = network_definitions{:, 1}; 
node_affiliations = network_definitions{:, 2}; 

% Get unique network affiliations
networks = {'MF', 'FP', 'DMN', 'MS', 'VI', 'VII', 'VAs', 'SAL', 'SC', 'CBL'}';

% ---------------------------------------

% Calculate the total number of networks
num_networks = numel(networks);

% Calculate the total number of pairings
total_pairings = num_networks * (num_networks + 1) / 2;

% Store average R values for each pairing
average_R_pos_all = zeros(num_networks, iterations);
average_R_neg_all = zeros(num_networks, iterations);
average_R_both_all = zeros(num_networks, iterations);

% Store leasioned edges for each pairing
lesion_edges_all = zeros(268, 268, num_networks);

% Loop through network affiliations
for j = 1:num_networks

    % Print the current network affiliation
    fprintf('\n Processing network: %s', networks{j});

    % Get the indices of nodes belonging to the current network affiliations
    idx1 = find(strcmp(node_affiliations, networks{j}));

    % Create a lesioned version of all_mats 
    all_mats_lesioned = all_mats;

    % Lesioned edges for this pairing
    lesioned_edges = zeros(268, 268);
    lesioned_edges(idx1, :) = 1;
    lesioned_edges(:, idx1) = 1; % Assuming symmetric matrices

    % Store lesioned edges for this pairing
    lesion_edges_all(:, :, j) = lesioned_edges;

    % Lesion all_mats
    for i = 1:size(all_mats, 3)
        all_mats_lesioned(:, :, i) = all_mats(:, :, i) .* lesioned_edges;
    end

    no_sub = size(all_mats_lesioned, 3);
    no_node = size(all_mats_lesioned, 1);

    % Initialize arrays to store predicted behavior for each fold and iteration
    behav_pred_pos = zeros(no_sub, K, iterations);
    behav_pred_neg = zeros(no_sub, K, iterations);
    behav_pred_both = zeros(no_sub, K, iterations);

    % Initialize 3D arrays to store masks for each fold and iteration
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
            train_mats = all_mats_lesioned(:, :, train_indices);
            train_behav = all_behav(train_indices);

            % Testing data
            test_mats = all_mats_lesioned(:, :, test_indices);
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
                train_sumpos(ss) = sum(sum(train_mats(:, :, ss) .* pos_mask)) / 2;
                train_sumneg(ss) = sum(sum(train_mats(:, :, ss) .* neg_mask)) / 2;
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
        end
        
        % Calculate correlation coefficients and p-values, the important part are the rho distributions to compare with the original model.
        behav_pred_pos_sum(:, iter) = sum(behav_pred_pos(:, :, iter), 2);
        behav_pred_neg_sum(:, iter) = sum(behav_pred_neg(:, :, iter), 2);
        behav_pred_both_sum(:, iter) = sum(behav_pred_both(:, :, iter), 2);

        [R_pos(:, iter), P_pos(:, iter)] = corr(behav_pred_pos_sum(:, iter), all_behav, 'Type', 'Spearman');
        [R_neg(:, iter), P_neg(:, iter)] = corr(behav_pred_neg_sum(:, iter), all_behav, 'Type', 'Spearman');
        [R_both(:, iter), P_both(:, iter)] = corr(behav_pred_both_sum(:, iter), all_behav, 'Type', 'Spearman');
    end

    % Average correlation coefficients across iterations and folds
    average_R_pos = mean(R_pos, 1); % Contains one rho value per iteration
    average_R_neg = mean(R_neg, 1); % Contains one rho value per iteration
    average_R_both = mean(R_both, 1); % Contains one rho value per iteration

    % Store correlation coefficients for each network lesion
    average_R_pos_all(j, :) = average_R_pos;
    average_R_neg_all(j, :) = average_R_neg;
    average_R_both_all(j, :) = average_R_both;
end

% Save workspace
save('workspace.mat');