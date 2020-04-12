% Run standard computer vision whole-image classification on a dataset 
% loaded into the "data" data structure.
% Vicente Ordonez @ UNC Chapel Hill

current_path = pwd();

addpath(genpath([current_path '/lib/vlfeat-0.9.17/toolbox'])); vl_setup;
addpath(genpath([current_path '/lib/liblinear-2.30/matlab']));
addpath(genpath([current_path '/lib/libsvm-3.23/matlab']));
addpath(genpath([current_path '/lib/gist']));
addpath(genpath('util'));

pkg load image
%pkg load parallel

% Set unique experiment identifier.
config.experiment_id = 'urbanperception';
config.data_year = 2013;

config.homedir = [current_path '/'];
config.features_path = [current_path '/features/'];
ensuredir(config.features_path);
config.datasource = ['placepulse_test_' num2str(config.data_year)];
config.image_path = [config.homedir '/data/images/pp1/' num2str(config.data_year) '/'];
config.image_url = config.image_path
%config.urban_data_file = [config.homedir '/data/' config.datasource '/consolidated_data.csv'];
%config.urban_data_file_type = 'csv';
config.urban_data_file = [config.homedir '/data/json/consolidated_data.json'];
config.urban_data_file_type = 'json';

% Configure feature type.
config.feature_type = 'gist';
%config.feature_type = 'fisher';
%config.feature_type = 'decaf';

% Gist features configuration.
config.gist_features_path = [config.features_path 'gist_features.mat'];

% Decaf features configuration.
config.decaf_layer = 'fc6_cudanet_out';
config.decaf_features_path = [config.features_path '/decaf_features.mat'];

% Sift features with Fisher Vectors settings.
config.kCodebookSizeGmm = 128;
config.lengthCodeBookGmm = 1000;
config.gmm_dictionary_path = [config.features_path 'gmm_dictionary.mat'];
config.pyramid = {[1 1], [2 2]};
config.fisher_features_path = [config.features_path 'fisher_features.mat'];

% Configure Learning parameters for Linear SVRs.
config.splits_path = [config.homedir '/output/split_info/regression'];
ensuredir(config.splits_path);
config.svm_method = Learning.REGRESSION_TYPE('L2_REGULARIZED_L2_LOSS_REGRESSION_DUAL');
config.bias_term = 1;
config.split_num = 10;
config.normalization = 'unnormalized';
config.p_param_space = [0.0001];
config.c_param_space = 10.^(-6:2);

% Configure output directory.
config.output_path = [config.homedir '/output/' config.datasource];
config.results_path = [config.output_path '/regression_results_' config.normalization '_' config.feature_type];
ensuredir(config.results_path);

% Load list of cities in the dataset.
cities = UrbanPerception.ListCities();
metric_set = {'safer', 'unique', 'upperclass'};

% Load data from the Place Pulse dataset.
urban = UrbanPerception(config.urban_data_file, config.urban_data_file_type, config.data_year);

% Plot data for the two cities.
for city_id = 1 : length(cities)
  city = cities{city_id};
  fprintf('%d. Plotting data for city: %s\n', city_id, city);
  urban.plotData(city, 'safer', [config.output_path '/view_data'], config.image_url);
  urban.plotData(city, 'unique', [config.output_path '/view_data'], config.image_url);
  urban.plotData(city, 'upperclass', [config.output_path '/view_data'], config.image_url);
end

fprintf('Computing features ... \n');

% Compute or load features.
compute_features_streets;

fprintf('Done \n');

fprintf('Doing Experiments ... \n');

% Now run experiments.
cities_harder = cities(end:-1:1);
for metric_ind = 1 : length(metric_set)
  % Now run regression.
  metric_str = metric_set{metric_ind};%metric_str = 'safer';
  for city_ind = 1 : length(cities)
    fprintf('\n\nREGRESSING %s..\n\n', cities{city_ind});
    city_string = cities{city_ind};
    city_string_harder = cities_harder{city_ind};
    city_identifier = regexprep(lower(city_string), ' ', '_');
    ensuredir(sprintf('%s/%s_%s/%s', config.results_path, config.experiment_id, city_identifier, metric_str));

    dataset = urban.getLabeledData(city_string, metric_str, config.normalization);
    [xx, inds] = ismember(dataset.images, urban.data.image_names);
    features_set = feature_matrix(inds, :);
      
    dataset_harder = urban.getLabeledData(city_string_harder, metric_str, config.normalization);
    [xx, inds_harder] = ismember(dataset_harder.images, urban.data.image_names);
    features_set_harder = feature_matrix(inds_harder, :);
    
    splits_fname = sprintf('%s/split_traincity_%s_metric_%s.mat', config.splits_path, city_identifier, metric_str);
    if ~exist(splits_fname, 'file')
      all_splits = Learning.CreateNFoldSplits(dataset.ids, dataset.labels, config.split_num);
      all_splits_harder = Learning.CreateNFoldSplits(dataset_harder.ids, dataset_harder.labels, 1);
      save(splits_fname, 'all_splits', 'all_splits_harder');
    else
      load(splits_fname);
    end
     
    %parpool('local', 5);
    results_split = zeros(3, config.split_num); 
    results_split_harder = zeros(3, config.split_num);
    SaveResults = @(location, model, test, test_harder, data, data_harder, urban)save(location, 'model', 'test', 'test_harder', 'data', 'data_harder', 'urban');
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
      results_path_ = sprintf('%s/%s_%s/%s', config.results_path, config.experiment_id, city_identifier, metric_str);
      ensuredir(results_path_);
      SaveResults(sprintf('%s/results_split_%d.mat', results_path_, split_id), model, test, test_harder, data, data_harder, urban);

      % Plot output results.
      figure_path = sprintf('%s/%s_%s/%s/%d_corr.jpg', config.results_path, config.experiment_id, city_identifier, metric_str, split_id);
      create_figure_regression(test.test_labels, test.predicted_labels, sprintf('[c=%.6f, p=%.5f, mse=%.5f, scc=%.5f]', model.best_c, model.best_p, test.RMSE, test.R), figure_path);

      figure_path = sprintf('%s/%s_%s/%s/%d_corr_harder.jpg', config.results_path, config.experiment_id, city_identifier, metric_str, split_id);
      create_figure_regression(test_harder.test_labels, test_harder.predicted_labels, sprintf('[c=%.6f, p=%.5f, mse=%.5f, scc=%.5f]', model.best_c, model.best_p, test_harder.RMSE, test_harder.R), figure_path);

      f = fopen(sprintf('%s/%s_%s/%s/%d.html', config.results_path, config.experiment_id, city_identifier, metric_str, split_id), 'w');
      fprintf(f, '<html><body><h3>[%s, %s, %s]</h3>\n', config.experiment_id, city_identifier, metric_str);
      fprintf(f, '<b>[split = %d]</b><br/>\n', split_id);
      fprintf(f, '<b>[<span style="color:blue"/>');
      fprintf(f, 'Same Data: </span> RMSE = %2.4f, R = %2.4f<br/>', test.RMSE, test.R);
      fprintf(f, '<span style="color:red">Diff Data: </span>');
      fprintf(f, ' RMSE = %2.4f, R = %2.4f]</b><br/>', test_harder.RMSE, test_harder.R);
      fprintf(f, '<b>[#(train) = %d, #(test) = %d, #(total) = %d]</b><br/>', length(data.train_images) + length(data.val_images), length(data.test_images), sum(strcmp(urban.data.cities, city_string)));
      fprintf(f, '<img src="%d_corr.jpg"/>', split_id);
      fprintf(f, '<img src="%d_corr_harder.jpg"/>', split_id);

      Learning.PlotTestPredictions(f, config, data, test, city_string, 'regression');
      Learning.PlotTestPredictions(f, config, data_harder, test_harder, city_string_harder, 'regression');
      fprintf(f, '</body></html>');
      fclose(f);
    end
    %poolobj = gcp('nocreate');
    %delete(poolobj);
    clear all_splits; all_splits_harder;
    save(sprintf('%s/%s_%s/%s/results.mat', config.results_path, config.experiment_id, city_identifier, metric_str), 'results_split', 'results_split_harder', 'config', 'dataset','dataset_harder', 'urban');
    ff = fopen(sprintf('%s/%s_%s/%s/results.html', config.results_path, config.experiment_id, city_identifier, metric_str), 'w');
    fprintf(ff, '<html><body>');
    fprintf(ff, '<h3>%s_%s_%s</h3>', config.experiment_id, city_identifier, metric_str);
    fprintf(ff, '[<span style="color:blue"/>Same Data: </span>');
    rs = results_split;
    fprintf(ff, ' MSE = %2.4f (std: %2.4f), Rsq = %2.4f (std: %2.4f), mRsq = %2.4f]<br/>', mean(rs(1, :)), std(rs(1, :)), mean(rs(2, :)), std(rs(2, :)), mean(rs(3, :)));
    fprintf(ff, '[<span style="color:red">Diff Data: </span>');
    rs = results_split_harder;
    fprintf(ff, ' MSE = %2.4f (std: %2.4f), Rsq = %2.4f (std: %2.4f), mRsq = %2.4f]<br/>', mean(rs(1, :)), std(rs(1, :)), mean(rs(2, :)), std(rs(2, :)), mean(rs(3, :)));
    fprintf(ff, '<table border>');
    for split_id = 1 : config.split_num
      fprintf(ff, '<tr><td>split_%d</td><td><a href="%d.html">', split_id, split_id);
      fprintf(ff, '<img src="%d_corr.jpg"/></a><br/>RMSE = %2.4f, R = %2.4f</td>', split_id, results_split(1, split_id), results_split(2, split_id));
      fprintf(ff, '<td><a href="%d.html">', split_id);
      fprintf(ff, '<img src="%d_corr_harder.jpg"/></a><br/>RMSE = %2.4f, R = %2.4f</td></tr>', split_id, results_split_harder(1, split_id), results_split_harder(2, split_id));
    end
    fprintf(ff, '</table>');
    fprintf(ff, '</body></html>');
    fclose(ff); 
  end
end
