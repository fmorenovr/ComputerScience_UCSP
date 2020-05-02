% Run standard computer vision whole-image classification on a dataset 
% loaded into the "data" data structure.
% Vicente Ordonez @ UNC Chapel Hill

current_path = pwd();

addpath(genpath([current_path '/lib/vlfeat-0.9.17/toolbox'])); vl_setup;
addpath(genpath([current_path '/lib/liblinear-2.30/matlab']));
addpath(genpath([current_path '/lib/libsvm-3.23/matlab']));
addpath(genpath([current_path '/lib/gist']));
addpath(genpath('utils'));

pkg load image
%pkg load parallel

% Set unique experiment identifier.
years = {2011, 2013, 2019};
metric_set = {'safer', 'unique', 'upperclass'};
features = {'cnn', 'cnn_gap', 'gist', 'fisher'};

for year_ind = 1 : length(years)
  config.data_year = years{year_ind};
  
  config.homedir = [current_path ];
  config.features_path = [current_path '/features/'];
  ensuredir(config.features_path);
  config.image_path = [config.homedir '/data/images/pp1/' num2str(config.data_year) '/'];
  config.image_url = config.image_path
  config.urban_data_file = [config.homedir '/data/consolidated_data.json'];
  config.urban_data_file_type = 'json';

  % Gist features configuration.
  config.gist_features_path = [config.features_path 'gist_' num2str(config.data_year) '.mat'];

  % CNN features configuration.
  config.cnn_layer = 'fc6_cudanet_out';
  config.cnn_features_path = [config.features_path '/cnn_' num2str(config.data_year) '.mat'];
  config.cnn_gap_features_path = [config.features_path '/cnn_gap_' num2str(config.data_year) '.mat'];

  % Sift features with Fisher Vectors settings.
  config.kCodebookSizeGmm = 128;
  config.lengthCodeBookGmm = 1000;
  config.gmm_dictionary_path = [config.features_path 'gmm_dictionary_' num2str(config.data_year) '.mat'];
  config.pyramid = {[1 1], [2 2]};

  config.fisher_features_path = [config.features_path 'fisher_' num2str(config.data_year) '.mat'];

  % Configure Learning parameters for Linear SVRs.
  config.splits_path = [config.homedir '/outputs/' num2str(config.data_year) '/regression/' ];
  ensuredir(config.splits_path);
  config.svm_method = Learning.REGRESSION_TYPE('L2_REGULARIZED_L2_LOSS_REGRESSION_DUAL');
  config.bias_term = 1;
  config.split_num = 10;
  config.normalization = 'unnormalized';
  config.p_param_space = [0.0001];
  config.c_param_space = 10.^(-6:2);

  % Configure output directory.
  config.output_path = [config.homedir '/outputs/' num2str(config.data_year) '/regression/'];
  ensuredir(config.output_path);
  
  f_table = fopen(sprintf('%s/table_%s.html', config.output_path, config.normalization), 'w');
  fprintf(f_table, '<html><body>');
  
  for feature_ind = 1 : length(features)
    config.feature_type = features{feature_ind};
  
    config.results_path = [config.output_path config.normalization '_' config.feature_type];
    ensuredir(config.results_path);

    % Load data from the Place Pulse dataset.
    urban = UrbanPerception(config.urban_data_file, config.urban_data_file_type, config.data_year);
    
    % Load list of cities in the dataset.
    cities = UrbanPerception.ListCities();
    cities_harder = cities(end:-1:1);
    fprintf(f_table, '<h3>%s</h3>', config.feature_type);
    fprintf(f_table, '<table border>');
    % Plot data for the two cities.
    for city_id = 1 : length(cities)
      city = cities{city_id};
      fprintf('%d. Plotting data for city: %s\n', city_id, city);
      urban.plotData(city, 'safer', [config.output_path '/view_data'], config.image_url);
      urban.plotData(city, 'unique', [config.output_path '/view_data'], config.image_url);
      urban.plotData(city, 'upperclass', [config.output_path '/view_data'], config.image_url);
    end

    % Compute or load features.
    fprintf('Computing features ... \n');
    data_city.images = cellfun(@(x){sprintf('%s/%s', config.image_path, x)}, urban.data.image_names);
    image_list = data_city.images;
    compute_features_streets;
    fprintf('Done \n');

    fprintf('Doing Experiments ... \n');

    % Now run experiments.
    for city_ind = 1 : length(cities)
      fprintf('\n\nREGRESSING %s..\n\n', cities{city_ind});
      city_string = cities{city_ind};
      city_string_harder = cities_harder{city_ind};
      city_identifier = regexprep(lower(city_string), ' ', '_');
      fprintf(f_table, '<tr><td rowspan=%d>%s</td><td></td>', 1+length(metric_set), city_string);
      fprintf(f_table, '<td width=100>%s</td><td width=100>%s</td>', city_string, city_string_harder);
      fprintf(f_table, '</tr>');
      for metric_ind = 1 : length(metric_set)
        metric_str = metric_set{metric_ind};
        ensuredir(sprintf('%s/%s/%s/', config.results_path, city_identifier, metric_str));
        fprintf(f_table, '<tr><td>%s</td>', metric_str);
        %output_results_splits = sprintf('%s/%s/splits', config.results_path, city_identifier);
        %ensuredir(output_results_splits);

        dataset = urban.getLabeledData(city_string, metric_str, config.normalization);
        [xx, inds] = ismember(dataset.images, urban.data.image_names);
        features_set = feature_matrix(inds, :);
          
        dataset_harder = urban.getLabeledData(city_string_harder, metric_str, config.normalization);
        [xx, inds_harder] = ismember(dataset_harder.images, urban.data.image_names);
        features_set_harder = feature_matrix(inds_harder, :);
        
       % splits_fname = sprintf('%s/split_traincity_%s_metric_%s.mat', config.splits_path, city_identifier, metric_str);
        
        %if ~exist(splits_fname, 'file')
        all_splits = Learning.CreateNFoldSplits(dataset.ids, dataset.labels, config.split_num);
        all_splits_harder = Learning.CreateNFoldSplits(dataset_harder.ids, dataset_harder.labels, 2);
        %  save(splits_fname, 'all_splits', 'all_splits_harder');
        %else
        %  load(splits_fname);
        %end
         
        %parpool('local', 5);
        results_split = zeros(3, config.split_num); 
        results_split_harder = zeros(3, config.split_num);
        SaveResults = @(location, model, test, test_harder, data, data_harder, urban) save(location, 'model', 'test', 'test_harder', 'data', 'data_harder', 'urban', '-v7');
        %parfor split_id = 1 : config.split_num
        for split_id = 1 : config.split_num
          fprintf('%s SPLIT %d...\n\n', city_string, split_id);
          rand('twister', split_id);
          % Load images in trainval, val and test sets.
          data = prepare_data(dataset, all_splits(split_id));
          % just use all images as test set.
          data_harder = prepare_data(dataset_harder, all_splits_harder(1)); 

          % Now learn models using Logistic Regression.
          model = Learning.TrainRegression(data, features_set, config);

          % Now run testing and present results in a webpage.
          test = Learning.TestRegression(data, features_set, model);
          results_split(:, split_id) = [test.RMSE.^2; test.R^2; test.mRsq];

          % Now run testing and present results in a webpage.
          test_harder = Learning.TestRegression(data_harder, features_set_harder, model);
          results_split_harder(:, split_id) = [test_harder.RMSE.^2; test_harder.R^2; test_harder.mRsq];
          
          % Save results.
          
          %SaveResults(sprintf('%s/results_split_%d.mat', output_results_splits, split_id), model, test, test_harder, data, data_harder, urban);

          % Plot output results.
          figure_path = sprintf('%s/%s/%s/%d_corr.png', config.results_path, city_identifier, metric_str, split_id);
          create_figure_regression(test.test_labels, test.predicted_labels, sprintf('[c=%.6f, p=%.5f, rmse=%.5f, R=%.5f]', model.best_c, model.best_p, test.RMSE, test.R), figure_path);

          figure_path = sprintf('%s/%s/%s/%d_corr_harder.png', config.results_path, city_identifier, metric_str, split_id);
          create_figure_regression(test_harder.test_labels, test_harder.predicted_labels, sprintf('[c=%.6f, p=%.5f, rmse=%.5f, R=%.5f]', model.best_c, model.best_p, test_harder.RMSE, test_harder.R), figure_path);

          f = fopen(sprintf('%s/%s/%s/%d.html', config.results_path, city_identifier, metric_str, split_id), 'w');
          fprintf(f, '<html><body><h3>[%s, %s]</h3>\n', city_identifier, metric_str);
          fprintf(f, '<b>[split = %d]</b><br/>\n', split_id);
          fprintf(f, '<b>[<span style="color:blue"/>');
          fprintf(f, 'Same Data: </span> RMSE = %2.4f, R = %2.4f<br/>', test.RMSE, test.R);
          fprintf(f, '<span style="color:red">Diff Data: </span>');
          fprintf(f, ' RMSE = %2.4f, R = %2.4f]</b><br/>', test_harder.RMSE, test_harder.R);
          fprintf(f, '<b>[#(train) = %d, #(test) = %d, #(total) = %d]</b><br/>', length(data.train_images) + length(data.val_images), length(data.test_images), sum(strcmp(urban.data.cities, city_string)));
          fprintf(f, '<img src="%d_corr.png"/>', split_id);
          fprintf(f, '<img src="%d_corr_harder.png"/>', split_id);

          Learning.PlotTestPredictions(f, config, data, test, city_string, 'regression');
          Learning.PlotTestPredictions(f, config, data_harder, test_harder, city_string_harder, 'regression');
          fprintf(f, '</body></html>');
          fclose(f);
        end
        %poolobj = gcp('nocreate');
        %delete(poolobj);
        clear all_splits; all_splits_harder;
        save(sprintf('%s/%s/%s/results.mat', config.results_path, city_identifier, metric_str), 'results_split', 'results_split_harder', 'config', 'dataset','dataset_harder', 'urban', '-v7');
        ff = fopen(sprintf('%s/%s/%s/results.html', config.results_path, city_identifier, metric_str), 'w');
        fprintf(ff, '<html><body>');
        fprintf(ff, '<h3>%s_%s</h3>', city_identifier, metric_str);
        fprintf(ff, '[<span style="color:blue"/>Same Data: </span>');
        rs = results_split;
        fprintf(ff, ' MSE = %2.4f (std: %2.4f), R^2 = %2.4f (std: %2.4f), mRsq = %2.4f]<br/>', mean(rs(1, :)), std(rs(1, :)), mean(rs(2, :)), std(rs(2, :)), mean(rs(3, :)));
        fprintf(ff, '[<span style="color:red">Diff Data: </span>');
        rs = results_split_harder;
        fprintf(ff, ' MSE = %2.4f (std: %2.4f), R^2 = %2.4f (std: %2.4f), mRsq = %2.4f]<br/>', mean(rs(1, :)), std(rs(1, :)), mean(rs(2, :)), std(rs(2, :)), mean(rs(3, :)));
        fprintf(ff, '<table border>');
        for split_id = 1 : config.split_num
          fprintf(ff, '<tr><td>split_%d</td><td><a href="%d.html">', split_id, split_id);
          fprintf(ff, '<img src="%d_corr.png"/></a><br/>MSE = %2.4f, R^2 = %2.4f</td>', split_id, results_split(1, split_id), results_split(2, split_id));
          fprintf(ff, '<td><a href="%d.html">', split_id);
          fprintf(ff, '<img src="%d_corr_harder.png"/></a><br/>MSE = %2.4f, R^2 = %2.4f</td></tr>', split_id, results_split_harder(1, split_id), results_split_harder(2, split_id));
        end
        fprintf(ff, '</table>');
        fprintf(ff, '</body></html>');
        fclose(ff);
        
        fprintf(f_table, '<td>%.4f</td>', sqrt(mean(results_split(2, :))));
        fprintf(f_table, '<td>%.4f</td>', sqrt(mean(results_split_harder(2, :))));
        fprintf(f_table, '</tr>\n');
      end
    end
    fprintf(f_table, '</table>');
  end
  fprintf(f_table, '</body></html>');
  fclose(f_table);
end
