% Vicente Ordonez @ UNC Chapel Hill 2013
% Learning contains standard machine learning routines.
% addpath(genpath('../lib/liblinear-1.94'));
% addpath(genpath('../lib/libsvm-3.17'));
% addpath(genpath('../lib/util'));
classdef Learning
  properties
  end
  
  methods
  end
   
  methods(Static)
    % Create N-folds from labeled images.
    % This is not good if you have binary labels because it doesn't guarantee balance.
    function data = CreateNFoldSplits(images, labels, n, type)
      if nargin > 3 & strcmp(type, 'classification')
        % Separate them on positives and negatives.
        pos_images = images(labels == 1);
        neg_images = images(labels ~= 1);

        % Randomize each set of items.
        pos_images = pos_images(randperm(length(pos_images))); 
        neg_images = neg_images(randperm(length(neg_images)));

        step_size_pos = round(length(pos_images) / n);
        step_size_neg = round(length(neg_images) / n);
        % Now create the splits for the positives and negatives separately.
        for i = 1 : n
          if i ~= n
            data(i).pos_test_images = ...
              pos_images(1 + (i - 1) * step_size_pos : i * step_size_pos);
            data(i).neg_test_images = ...
              neg_images(1 + (i - 1) * step_size_neg : i * step_size_neg);
          else
            data(i).pos_test_images = pos_images(1 + (n - 1) * step_size_pos : end);
            data(i).neg_test_images = neg_images(1 + (n - 1) * step_size_neg : end);
          end
          [xx, data(i).pos_test_ids] = ismember(data(i).pos_test_images, images); 
          [xx, data(i).neg_test_ids] = ismember(data(i).neg_test_images, images);
           
          [pos_trainval_images, inds] = setdiff(pos_images, data(i).pos_test_images);
          cut_pos = round(length(pos_trainval_images) * 2 / 3);
          data(i).pos_train_images = pos_trainval_images(1:cut_pos);
          [xx, data(i).pos_train_ids] = ismember(data(i).pos_train_images, images);
          data(i).pos_val_images = pos_trainval_images(cut_pos + 1 : end); 
          [xx, data(i).pos_val_ids] = ismember(data(i).pos_val_images, images);

          [neg_trainval_images, inds] = setdiff(neg_images, data(i).neg_test_images);
          cut_neg = round(length(neg_trainval_images) * 2 / 3);
          data(i).neg_train_images = neg_trainval_images(1:cut_neg);
          [xx, data(i).neg_train_ids] = ismember(data(i).neg_train_images, images);
          data(i).neg_val_images = neg_trainval_images(cut_neg + 1 : end); 
          [xx, data(i).neg_val_ids] = ismember(data(i).neg_val_images, images);

          data(i).test_images = [data(i).pos_test_images; data(i).neg_test_images];
          data(i).test_ids = [data(i).pos_test_ids; data(i).neg_test_ids];
          data(i).test_labels = [ones(length(data(i).pos_test_images), 1); ...
                      -ones(length(data(i).neg_test_images), 1)];

          data(i).train_images = [data(i).pos_train_images; data(i).neg_train_images];
          data(i).train_ids = [data(i).pos_train_ids; data(i).neg_train_ids];
          data(i).train_labels = [ones(length(data(i).pos_train_images), 1); ...
                       -ones(length(data(i).neg_train_images), 1)];

          data(i).val_images = [data(i).pos_val_images; data(i).neg_val_images];
          data(i).val_ids = [data(i).pos_val_ids; data(i).neg_val_ids];
          data(i).val_labels = [ones(length(data(i).pos_val_images), 1); ...
                     -ones(length(data(i).neg_val_images), 1)];

          assert(sum(ismember(data(i).train_images, data(i).test_images)) == 0);
          assert(sum(ismember(data(i).val_images, data(i).test_images)) == 0);
          assert(sum(ismember(data(i).train_images, data(i).val_images)) == 0);
        end
      else
        rand_ids = randperm(length(images));
        all_images = images(rand_ids)';
        %all_labels = labels(rand_ids)';

        step_size = round(length(all_images) / n);
        
        for i = 1 : n
          if i ~= n
            data(i).test_ids = all_images(1 + (i - 1) * step_size : i * step_size);
            %data(i).test_labels = all_labels(1 + (i - 1) * step_size : i * step_size);
          else
            data(i).test_ids = all_images(1 + (n - 1) * step_size : end);
            %data(i).test_labels = all_labels(1 + (n - 1) * step_size : end);
          end
          %[xx, data(i).test_ids] = ismember(data(i).test_images, images);
          
          [trainval_images, inds] = setdiff(all_images, data(i).test_ids);
          %trainval_labels = all_labels(inds);
          cut_pos = round(length(trainval_images) * 2 / 3);
           
          data(i).train_ids = trainval_images(1:cut_pos);
          %data(i).train_labels = trainval_labels(1:cut_pos);
          %[xx, data(i).train_ids] = ismember(data(i).train_images, images);

          data(i).val_ids = trainval_images(cut_pos + 1 : end); 
          %data(i).val_labels = trainval_labels(cut_pos + 1 : end);
          %[xx, data(i).val_ids] = ismember(data(i).val_images, images);

          assert(sum(ismember(data(i).train_ids, data(i).test_ids)) == 0);
          assert(sum(ismember(data(i).val_ids, data(i).test_ids)) == 0);
          assert(sum(ismember(data(i).train_ids, data(i).val_ids)) == 0);
        end
      end 
    end
    % Given a set of positive and negative samples create a train,val,test split.
    function data = CreateRandomizedSplit(images, labels, type)
      if strcmp(type, 'classification')
        pos_images = images(labels == 1);
        neg_images = images(labels ~= 1);

        % Randomize each set of items.
        pos_images = pos_images(randperm(length(pos_images))); 
        neg_images = neg_images(randperm(length(neg_images)));

        % Set number of positive images in train, val and test.
        num_train_pos = round(length(pos_images) / 2);
        num_val_pos = round(length(pos_images) / 4);
        num_test_pos = length(pos_images) - num_train_pos - num_val_pos;

        % Set number of negative images in train, val and test.
        num_train_neg = round(length(neg_images) / 2);
        num_val_neg = round(length(neg_images) / 4);
        num_test_neg = length(neg_images) - num_train_neg - num_val_neg;

        % Now create the train data split.
        data.train_images = [pos_images(1:num_train_pos); ...
                   neg_images(1:num_train_neg)];
        data.train_labels = [ones(num_train_pos, 1); -ones(num_train_neg, 1)];

        % Now create the val data split.
        data.val_images = [pos_images(num_train_pos + 1: num_train_pos + num_val_pos); ...
                   neg_images(num_train_neg + 1: num_train_neg + num_val_neg)];
        data.val_labels = [ones(num_val_pos, 1); -ones(num_val_neg, 1)];
   
        % Now create the test data split.
        data.test_images = [pos_images(num_train_pos + num_val_pos + 1 : end); neg_images(num_train_neg + num_val_neg + 1 : end)];
        data.test_labels = [ones(num_test_pos, 1); -ones(num_test_neg, 1)];

        [xx, data.train_ids] = ismember(data.train_images, images);
        [xx, data.val_ids] = ismember(data.val_images, images);
        [xx, data.test_ids] = ismember(data.test_images, images);
      elseif strcmp(type, 'regression')
        % Randomize images and labels.
        rand_ids = randperm(length(images));
        all_images = images(rand_ids)';
        all_labels = labels(rand_ids)';

        mid_pos = floor(length(images) / 2);
        quart_pos = floor(length(images) / 4);

        % Now set train, val and test.
        data.train_images = all_images(1:mid_pos);
        data.train_labels = all_labels(1:mid_pos);
        [xx, data.train_ids] = ismember(data.train_images, images);
        
        data.val_images = all_images(mid_pos + 1: mid_pos + quart_pos);
        data.val_labels = all_labels(mid_pos + 1: mid_pos + quart_pos);
        [xx, data.val_ids] = ismember(data.val_images, images);

        data.test_images = all_images(end - quart_pos: end);
        data.test_labels = all_labels(end - quart_pos: end); 
        [xx, data.test_ids] = ismember(data.test_images, images);
      end
    end

    function test = TestRandomAssignment(data, features)
      decision_values = rand(size(data.test_labels));
      predicted_labels = 2 * (decision_values > 0.5) - 1;
      accuracy = sum(predicted_labels == data.test_labels) / length(data.test_labels); 
    end

    % Given a learned svm model with "TrainLinearSvm", run the model on the test features.
    function test = TestLinearSvm(data, features_set, model)
      if ~isfield(features_set, 'train_features')
        features.train_features = features_set(data.train_ids, :);
        features.val_features = features_set(data.val_ids, :);
        features.test_features = features_set(data.test_ids, :);
      else
        features = features_set;
      end

      tic;
      % Test on validation set and compute accuracy.
      [predicted_labels, accuracy, decision_values] = ...
        predict(data.test_labels, sparse(features.test_features), model, '-q'); 
      accuracy = accuracy(1);

      [sorted_decision_values, sinds] = sort(decision_values, 'descend');
      sorted_predicted_labels = predicted_labels(sinds);
      sorted_test_labels = data.test_labels(sinds);
      precisions = cumsum(sorted_test_labels > 0)' ./ [1:length(data.test_labels)];
      recalls = cumsum(sorted_test_labels > 0)' / sum(data.test_labels == 1);
      area_under_curve = trapz(recalls, precisions); 
      fprintf('Test-Results: acc = %2.2f, AUC = %2.2f, took = %f\n', accuracy, 100 * area_under_curve, toc);

      test.accuracy = accuracy;
      test.area_under_curve = area_under_curve;
      test.precisions = precisions;
      test.recalls = recalls;
      test.predicted_labels = predicted_labels;
      test.predicted_scores = decision_values;
    end
    
    function model = TrainRegression(data, features_set, config)
      if ~isfield(config, 'svm_method') || isempty(config.svm_method)
        config.svm_method = Learning.REGRESSION_TYPE('L2_REGULARIZED_L2_LOSS_REGRESSION_DUAL');
      end
      if ~isfield(config, 'bias_term') || isempty(config.bias_term)
        config.bias_term = 1;
      end
      if ~isfield(features_set, 'train_features')
        features.train_features = features_set(data.train_ids, :);
        features.val_features = features_set(data.val_ids, :);
        features.test_features = features_set(data.test_ids, :);
      else
        features = features_set;
      end

      % Find the best C-parameter on validation data.
      fprintf('\nFinding the best C parameter on validation data\n');
      %output_path = '/home/vicente/public_html/regress_tmp';
      if ~isfield(config, 'p_param_space')
        p_param_space = [0.1];
      else
        p_param_space = config.p_param_space;
      end
      if ~isfield(config, 'c_param_space')
        c_param_space = [0.1];
      else
        c_param_space = config.c_param_space;
      end
      %p_param_space = 10.^(-4:3);
      %c_param_space = 10.^(-4:3);
      metrics = zeros(length(c_param_space), length(p_param_space));
      %f = fopen(sprintf('%s/regress.html', output_path), 'w');
      for j = 1 : length(p_param_space)
      %fprintf(f, '<a href="regress_%d.html">%d. Param: %.6f</a><br/>',j,j,p_param_space(j));
      %ff = fopen(sprintf('%s/regress_%d.html', output_path, j), 'w');
      %fprintf(ff, '<table border>');
      for i = 1 : length(c_param_space)
        tic;
        c = c_param_space(i);
        p = p_param_space(j);
        
        train_labels = data.train_labels;
        val_labels = data.val_labels;
        test_labels = data.test_labels;
%        train_labels = 10 ./ (1 + exp(-(data.train_labels - mean(data.train_labels))));
%        val_labels = 10 ./ (1 + exp(-(data.val_labels - mean(data.val_labels))));
%        test_labels = 10 ./ (1 + exp(-(data.test_labels - mean(data.test_labels))));

        model_ = train(train_labels, sparse(features.train_features), ...
                 sprintf('-s %d -c %f -e %f -q', config.svm_method, c, p));
        [predicted_labels_train, perf_train, decision_values] = ...
          predict(train_labels, sparse(features.train_features), model_);

        [predicted_labels, perf, decision_values] = ...
          predict(val_labels, sparse(features.val_features), model_);
        metrics(i, j) = perf(3);
        fprintf('%d. c = %.5f, p = %.5f, MSE = %2.4f, Rsq = %2.4f took = %f\n', i, c, p, perf(2), perf(3), toc);

        %create_figure_regression(train_labels, predicted_labels_train, ...
        %  sprintf('traindata [c=%.6f, p=%.5f, mse=%.5f, scc=%.5f]', c, p, perf_train(2), perf_train(3)), ...
        %  sprintf('%s/train_%d_%d_train.png', output_path, j, i));
        
        %create_figure_regression(val_labels, predicted_labels, ...
        %  sprintf('traindata [c=%.6f, p=%.5f, mse=%.5f, scc=%.5f]', c, p, perf(2), perf(3)), ...
        %  sprintf('%s/train_%d_%d_val.png', output_path, j, i));
        %fprintf(ff, '<tr><td><img src="train_%d_%d_train.png"/></td>', j, i);
        %fprintf(ff, '<td><img src="train_%d_%d_val.png"/></td></tr>', j, i);
      end
      %fprintf(ff, '</table>');
      %fclose(ff);
      end
      %fclose(f);

      best_metric = max(metrics(:));
      [cind, pind] = find(metrics == best_metric);
      best_c = c_param_space(cind);
      best_p = p_param_space(pind);
       
      fprintf('Training final SVM model\n'); tic;
      trainval_labels = [data.train_labels; data.val_labels];
      %trainval_labels = 10 ./ (1 + exp(-(trainval_labels_ - mean(trainval_labels_))));
      trainval_features = [features.train_features; features.val_features]; 
      model = train(trainval_labels, sparse(trainval_features), ...
              sprintf('-s %d -c %f -e %f -q', config.svm_method, best_c, best_p));
      % Let's find out the amount of self-information.
      % Test on trainval set and compute accuracy.
      [predicted_labels, accuracy, decision_values] = ...
        predict(trainval_labels, sparse(trainval_features), model, '-q'); 
      train_mse_value = mean((trainval_labels - predicted_labels).^2);
      model.best_c = best_c;
      model.best_p = best_p;
      model.validation_Rsq = best_metric;
      model.training_Rsq = accuracy(3);
      model.training_MSE = train_mse_value;
      
      q_hat = mean(trainval_labels);
      sq_errs = sum((trainval_labels - predicted_labels).^2);
      sq_errs_hat = sum((trainval_labels - q_hat).^2);
      mRsq = 1 - (sq_errs / sq_errs_hat);
      
      fprintf('Final model: best_c = %.5f, best_p = %.5f, training_MSE = %2.2f, Rsq = %2.4f, mRsq = %2.4f, took = %f\n', ...
          best_c, best_p, train_mse_value, accuracy(3), mRsq, toc);
    end
    
    function test = TestRegression(data, features_set, model)
      if ~isfield(features_set, 'train_features')
        %features.train_features = features_set(data.train_ids, :);
        %features.val_features = features_set(data.val_ids, :);
        features.test_features = features_set(data.test_ids, :);
      else
        features = features_set;
      end

      % test_labels = 10 ./ (1 + exp(-(data.test_labels - mean(data.test_labels))));
      test_labels = data.test_labels;
      tic;
      % Test on validation set and compute accuracy.
      [predicted_labels, accuracy, decision_values] = ...
        predict(test_labels, sparse(features.test_features), model, '-q'); 
      mse_value = mean((test_labels - predicted_labels).^2);
      fprintf('Test-Results: MSE = %2.4f, Rsq = %2.4f, took = %f\n', ...
        mse_value, accuracy(3), toc);

      q_hat = mean(test_labels);
      sq_errs = sum((test_labels - predicted_labels).^2);
      sq_errs_hat = sum((test_labels - q_hat).^2);
      mRsq = 1 - (sq_errs / sq_errs_hat);

      test.RMSE = sqrt(mse_value);
      test.R = sqrt(accuracy(3));
      test.test_labels = test_labels;
      test.predicted_labels = predicted_labels;
      test.predicted_scores = decision_values;
      test.mRsq = mRsq;
      % Coefficient of determination.
      
    end

    % Learn a linear SVM given a dataset represented by data and features.
    % Required inputs:
    % data.train_labels, features.train_features
    % data.val_labels, features.val_features
    function model = TrainLinearSvm(data, features_set, config)
      if ~isfield(config, 'svm_method') || isempty(config.svm_method)
        config.svm_method = Learning.SVM_TYPE('L2_REGULARIZED_L2_LOSS_DUAL');
      end
      if ~isfield(config, 'bias_term') || isempty(config.bias_term)
        config.bias_term = 1;
      end
      if ~isfield(features_set, 'train_features')
        features.train_features = features_set(data.train_ids, :);
        features.val_features = features_set(data.val_ids, :);
        features.test_features = features_set(data.test_ids, :);
      else
        features = features_set;
      end
      
      % Find the best C-parameter on validation data.
      fprintf('\nFinding the best C parameter on validation data\n');
      c_param_space = [0.0001 0.001 0.01 0.1 1 10];
      metrics = zeros(length(c_param_space), 1);
      for i = 1 : length(c_param_space) 
        tic;
        c = c_param_space(i);
        wneg = length(find(data.train_labels == -1)) / length(find(data.train_labels == 1));
        model_ = train(data.train_labels, sparse(features.train_features), sprintf('-s %d -B %d -c %f -q -w-1 1 -w1 %f', config.svm_method, config.bias_term, c, wneg));
        % Test on validation set and compute accuracy.
        [predicted_labels, accuracy, decision_values] = predict(data.val_labels, sparse(features.val_features), model_, '-q');
        accuracy = accuracy(1);

        [sorted_decision_values, sinds] = sort(decision_values, 'descend');
        sorted_predicted_labels = predicted_labels(sinds);
        sorted_val_labels = data.val_labels(sinds);
        precisions = cumsum(sorted_val_labels > 0)' ./ [1:length(data.val_labels)];
        recalls = cumsum(sorted_val_labels > 0)' / sum(data.val_labels == 1);
        area_under_curve = trapz(recalls, precisions); 
        metrics(i) = area_under_curve;
        fprintf('%d. c = %.5f, acc = %2.2f, AUC = %2.2f, took = %f\n', ...
            i, c, accuracy, 100 * area_under_curve, toc);
      end

      fprintf('Training final SVM model\n'); tic;
      % Now pick the best C-parameter and learn the final model.
      [best_metric, mind] = max(metrics);
      best_c = c_param_space(mind);
      trainval_labels = [data.train_labels; data.val_labels];
      trainval_features = [features.train_features; features.val_features];
      wneg = length(find(trainval_labels == -1)) / length(find(trainval_labels == 1));
      model = train(trainval_labels, sparse(trainval_features), ...
              sprintf('-s %d -B %d -c %f -q -w-1 1 -w1 %f', ...
                  config.svm_method, config.bias_term, best_c, wneg));
      % Let's find out the amount of self-information.
      % Test on trainval set and compute accuracy.
      [predicted_labels, accuracy, decision_values] = ...
        predict(trainval_labels, sparse(trainval_features), model, '-q'); 
      accuracy = accuracy(1);
      fprintf('Final model: best_c = %.5f, training_acc = %2.2f, took = %f\n', ...
          best_c, accuracy, toc);
      model.best_c = best_c;
      model.validation_AUC = best_metric;
      model.training_accuracy = accuracy;
    end 

    % Given a learned svm model with "TrainLinearSvm", run the model on the test features.
    function test = TestKernelSvm(data, features, model)
      tic;
      % Test on validation set and compute accuracy.
      [predicted_labels, accuracy, decision_values] = ...
        svmpredict(data.test_labels, sparse(features.test_features), model.libsvm_model, '-q'); 
      accuracy = accuracy(1);
      decision_values = decision_values(:, 1);
      [sorted_decision_values, sinds] = sort(decision_values, 'descend');
      sorted_predicted_labels = predicted_labels(sinds);
      sorted_test_labels = data.test_labels(sinds);
      precisions = cumsum(sorted_test_labels > 0)' ./ [1:length(data.test_labels)];
      recalls = cumsum(sorted_test_labels > 0)' / sum(data.test_labels == 1);
      area_under_curve = trapz(recalls, precisions); 
      fprintf('Test-Results: acc = %2.2f, AUC = %2.2f, took = %f\n', ...
          accuracy, 100 * area_under_curve, toc);

      test.accuracy = accuracy;
      test.area_under_curve = area_under_curve;
      test.precisions = precisions;
      test.recalls = recalls;
      test.predicted_labels = predicted_labels;
      test.predicted_scores = decision_values;
    end

    % Learn an RBF SVM using libsvm given a dataset represented by data and features.
    % Required inputs:
    % data.train_labels, features.train_features
    % data.val_labels, features.val_features
    function model = TrainKernelSvm(data, features, config)
      config.svm_method = 0; % multiclass classification.
      config.kernel_type = 2; % RBF kernel.
      
      % Find the best C-parameter on validation data.
      fprintf('\nFinding the best C parameter on validation data\n');
      c_param_space = [0.000001 0.00001 0.0001 0.001 0.01 0.1 1 10 100 1000];
      g_param_space = [0.0001 0.001 0.01 0.1 1 10 100 1000];
      metrics = zeros(length(c_param_space), length(g_param_space));
      informative = zeros(length(c_param_space), length(g_param_space));
      for i = 1 : length(c_param_space)
        c = c_param_space(i);
      for j = 1 : length(g_param_space) tic;
        g = g_param_space(j);
        wneg = length(find(data.train_labels == -1)) / length(find(data.train_labels == 1));
        model_ = svmtrain(data.train_labels, features.train_features, ...
                  sprintf('-s %d -t %d -c %f -g %f -q -w-1 1 -w1 %f', ...
                      config.svm_method, config.kernel_type, c, g, wneg));
        % Test on validation set and compute accuracy.
        [predicted_labels, accuracy, decision_values] = ...
          svmpredict(data.val_labels, features.val_features, model_, '-q'); 
        accuracy = accuracy(1);

        [sorted_decision_values, sinds] = sort(decision_values, 'descend');
        sorted_predicted_labels = predicted_labels(sinds);
        sorted_val_labels = data.val_labels(sinds);
        precisions = cumsum(sorted_val_labels > 0)' ./ [1:length(data.val_labels)];
        recalls = cumsum(sorted_val_labels > 0)' / sum(data.val_labels == 1);
        area_under_curve = trapz(recalls, precisions); 
        metrics(i, j) = area_under_curve;
        informative(i, j) = length(unique(predicted_labels)) > 1;
        count = i * length(g_param_space) + j;
        fprintf('%d. c = %.5f, g = %.5f, acc = %2.2f, AUC = %2.2f, took = %f\n', ...
            count, c, g, accuracy, 100 * area_under_curve, toc);
      end
      end

      fprintf('Training final SVM model\n'); tic;
      % Now pick the best C-parameter and learn the final model.
      metrics = metrics .* informative;
      best_metric = max(metrics(:));
      [mind_c, mind_g] = find(max(metrics == best_metric));
      best_c = c_param_space(mind_c);
      best_g = g_param_space(mind_g);
      trainval_labels = [data.train_labels; data.val_labels];
      trainval_features = [features.train_features; features.val_features];
      wneg = length(find(trainval_labels == -1)) / length(find(trainval_labels == 1));
      libsvm_model = svmtrain(trainval_labels, sparse(trainval_features), ...
               sprintf('-s %d -t %d -c %f -g %f -w-1 1 -w1 %f -b 1 -q', ...
                   config.svm_method, config.kernel_type, best_c, best_g, wneg));
      % Let's find out the amount of self-information.
      % Test on trainval set and compute accuracy.
      [predicted_labels, accuracy, decision_values] = ...
        svmpredict(trainval_labels, sparse(trainval_features), libsvm_model, '-q -b 1'); 
      accuracy = accuracy(1);
      fprintf('Final model: best_c = %.5f, best_g = %.5f, training_acc = %2.2f, took = %f\n', ...
          best_c, best_g, accuracy, toc);
      model.best_c = best_c; model.best_g = best_g;
      model.validation_AUC = best_metric;
      model.training_accuracy = accuracy;
      model.libsvm_model = libsvm_model;
    end 

    function PlotTestPredictions(f, config, data, test, identifier, type_str)
      fprintf(f, '<h4>Predictions on %s:</h4>', identifier);
      if strcmp(type_str, 'regression')
        fprintf(f, '<table><tr>');
        [s_scores, s_inds] = sort(test.predicted_labels, 'descend');
        L1_errors = (test.predicted_labels - test.test_labels).^2;
        norm_errors = (L1_errors - min(L1_errors)) / (max(L1_errors) - min(L1_errors));
        for i = 1 : length(data.test_images)
          [path, image_name, ext] = fileparts(data.test_images{s_inds(i)});
          value = round(255.0 * (norm_errors(s_inds(i))));
          value_2 = round(255.0 * (1 - norm_errors(s_inds(i))));
          if norm_errors(s_inds(i)) > mean(norm_errors)
            style_str = ['border:4px solid rgb(' num2str(value) ', 0, 0)']; 
          else
            style_str = ['border:4px solid rgb(0, ' num2str(value_2) ', 0)']; 
          end
          fprintf(f, '<td><table><tr>');
          fprintf(f, '<td><img style="%s" src="%s/%s%s" height="180"/></td></tr>', ...
                 style_str, config.image_url, image_name, ext);
          fprintf(f, '<tr><td>%d. SVR: %2.4f, GT: %2.4f, ERR: %2.4f</td>', i, ...
                 s_scores(i), test.test_labels(s_inds(i)), sqrt(L1_errors(s_inds(i))));
          fprintf(f, '</tr></table></td>');
          if mod(i, 5) == 0, fprintf(f, '</tr><tr>'); end
        end
        fprintf(f, '</tr></table>');   
      else
        fprintf(f, '<table><tr>');
        [s_scores, s_inds] = sort(test.predicted_scores, 'descend');
        for i = 1 : length(data.test_images)
          [path, image_name, ext] = fileparts(data.test_images{s_inds(i)});
          if (data.test_labels(s_inds(i)) == 1), style_str = 'border:3px solid #00f';
          else, style_str = 'border:3px solid #f00'; end
          fprintf(f, '<td><table><tr>');
          fprintf(f, '<td><img style="%s" src="%s/%s%s" height="180"/></td></tr>', ...
                 style_str, config.image_url, image_name, ext);
          fprintf(f, '<tr><td>%d. svm-score: %.4f</td>', i, s_scores(i));
          fprintf(f, '</tr></table></td>');
          if mod(i, 5) == 0, fprintf(f, '</tr><tr>'); end
        end
        fprintf(f, '</tr></table>');
      end
    end

    function method = REGRESSION_TYPE(method_string)
      switch(method_string)
        case 'L2_REGULARIZED_L2_LOSS_REGRESSION_PRIMAL'
          method = 11;
        case 'L2_REGULARIZED_L2_LOSS_REGRESSION_DUAL'
          method = 12;
        case 'L2_REGULARIZED_L1_LOSS_LOGISTIC_REGRESSION_DUAL'
          method = 13;
        otherwise
          error('unknown regression');
      end
    end

    function method = SVM_TYPE(method_string)
      switch(method_string)
        case 'L2_REGULARIZED_L2_LOSS_DUAL'
          method = 1;
        case 'L2_REGULARIZED_L2_LOSS_PRIMAL'
          method = 2;
        otherwise
          method = 1;
      end
    end

    % Plot and save a precision-recall curve.
    function SavePrecisionRecallCurve(precisions, recalls, figure_path, color_str)
      [aa, bb, cc] = fileparts(figure_path);
      if ~exist(aa, 'dir'), mkdir(aa); end
      if isempty(color_str), color_str = 'red';end
      figure;
      hold on;
      if iscell(precisions)
        color_strings = {'blue', 'red', 'magenta', 'orange', 'brown', 'black'};
        for i = 1 : length(precisions);
          color = color_strings{i}; 
          plot(recalls{i}, precisions{i}, 'Color', color, 'LineWidth', 4);
        end
      else
        plot(recalls, precisions, 'Color', color_str, 'LineWidth', 4);
      end
      axis([0 1 0 1]);grid;
      print('-djpeg', '-r60', figure_path);
      close;
    end

  end
end

