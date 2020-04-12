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
config.data_year = 2019;

config.homedir = [current_path '/'];
config.features_path = [current_path '/features/'];
ensuredir(config.features_path);
config.datasource = ['placepulse_test_' num2str(config.data_year)];
config.image_path = [config.homedir '/data/images/pp1/' num2str(config.data_year) '/'];
config.image_url = config.image_path
%config.urban_data_file = [config.homedir '/data/' config.datasource '/consolidated_data.csv'];
%config.urban_data_file_type = 'csv';
config.urban_data_file = [config.homedir '/data/consolidated_data.json'];
config.urban_data_file_type = 'json';

% Configure feature type.
%config.feature_type = 'gist';
%config.feature_type = 'fisher';
config.feature_type = 'decaf';

% Gist features configuration.
config.gist_features_path = [config.features_path 'gist_features_' num2str(config.data_year) '.mat'];

% Decaf features configuration.
config.decaf_layer = 'fc6_cudanet_out';
config.decaf_features_path = [config.features_path '/decaf_features_' num2str(config.data_year) '.mat'];

% Sift features with Fisher Vectors settings.
config.kCodebookSizeGmm = 128;
config.lengthCodeBookGmm = 1000;
config.gmm_dictionary_path = [config.features_path 'gmm_dictionary' num2str(config.data_year) '.mat'];
config.pyramid = {[1 1], [2 2]};
config.fisher_features_path = [config.features_path 'fisher_features' num2str(config.data_year) '.mat'];

% Configure Learning parameters for Linear SVMs.
config.splits_path = [config.homedir '/output/split_info/binary_' num2str(config.data_year) ];
ensuredir(config.splits_path);
config.svm_method = Learning.SVM_TYPE('L2_REGULARIZED_L2_LOSS_DUAL');
config.bias_term = 1;
config.split_num = 10;

% Configure output directory.
config.output_path = [config.homedir '/output/' config.datasource];
config.results_path = [config.output_path '/classification_results_' config.feature_type];
ensuredir(config.results_path);

% Load list of cities in the dataset.
cities = UrbanPerception.ListCities();
metric_set = {'safer'};%, 'unique', 'upperclass'};

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
  % Now run classification.
  metric_str = metric_set{metric_ind};
  for city_ind = 1 : length(cities)
    city_string = cities{city_ind};
    city_string_harder = cities_harder{city_ind};
    city_identifier = regexprep(lower(city_string), ' ', '_');
    ensuredir(sprintf('%s/%s_%s/%s', config.results_path, config.experiment_id, city_identifier, metric_str));

    delta_set = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.5];
    #delta_set = [0.5]
    delta_aucs = zeros(length(delta_set), config.split_num);
    delta_aucs_harder = zeros(length(delta_set), config.split_num);
    for delta_ind = 1 : length(delta_set)
      % Define positive and negative set from scores.
      delta = delta_set(delta_ind);  % Take the top delta%/bottom delta% as positives/negatives.
      dataset = urban.getBinaryData(city_string, metric_str, delta);
      [xx, inds] = ismember(dataset.images, urban.data.image_names);
      features_set = feature_matrix(inds, :);
      
      dataset_harder = urban.getBinaryData(city_string_harder, metric_str, delta);
      [xx, inds_harder] = ismember(dataset_harder.images, urban.data.image_names);
      features_set_harder = feature_matrix(inds_harder, :);
      
      fprintf('\nSplitting data in %d-fold cross val\n', config.split_num);
      
      for split_id = 1 : config.split_num
        splits_fname = sprintf('%s/split_traincity_%s_metric_%s_d%02d_%02d.mat', config.splits_path, city_identifier, metric_str, round(delta * 100), split_id);

%        if ~exist(splits_fname, 'file') 
        rand('twister', split_id);
          % Load images in trainval, val and test sets.
        data = Learning.CreateRandomizedSplit(dataset.images, dataset.labels, 'classification');
        data_harder = Learning.CreateRandomizedSplit(dataset_harder.images, dataset_harder.labels, 'classification');
          % Save split data.
%          save(splits_fname, 'data', 'data_harder','-v7');
%        else
%          load(splits_fname);
%          data = fix_image_names(data, urban);
%          data_harder = fix_image_names(data_harder, urban);
%        end

        fprintf('Done\n');
        
        fprintf('\nTraining over city: %s, metric: %s with delta: %02d\n', city_string, metric_str, delta);
        
        % Now learn models using Linear SVMs'.
        model = Learning.TrainLinearSvm(data, features_set, config);

        fprintf('Done\n');

        fprintf('\nTrained over: %s, Testing over city: %s, metric: %s with delta: %02d\n', city_string, city_string, metric_str, delta);

        % Now run testing and present results in a webpage.
        test = Learning.TestLinearSvm(data, features_set, model);
        delta_aucs(delta_ind, split_id) = test.area_under_curve;

        fprintf('Done\n');
        
        fprintf('\nTrained over: %s, Testing over city: %s, metric: %s with delta: %02d\n', city_string, city_string_harder, metric_str, delta);

        % Now run testing and present results in a webpage.
        test_harder = Learning.TestLinearSvm(data_harder, features_set_harder, model);
        delta_aucs_harder(delta_ind, split_id) = test_harder.area_under_curve;
        
        fprintf('Done\n');

        % Plot output results.
        figure_path = sprintf('%s/%s_%s/%s/delta_%d/%d_pr.jpg', config.results_path, config.experiment_id, city_identifier, metric_str, round(100 * delta), split_id);
        precisions{1} = test.precisions; precisions{2} = test_harder.precisions;
        recalls{1} = test.recalls; recalls{2} = test_harder.recalls;
        Learning.SavePrecisionRecallCurve(precisions, recalls, figure_path, 'blue');
        f = fopen(sprintf('%s/%s_%s/%s/delta_%d/%d.html', config.results_path, config.experiment_id, city_identifier, metric_str, round(100 * delta), split_id), 'w');
        fprintf(f, '<html><body><h3>[%s, %s, %s]</h3>\n', config.experiment_id, city_identifier, metric_str);
        fprintf(f, '<b>[delta = %.2f, split = %d]</b><br/>\n', delta, split_id);
        fprintf(f, '<b>[<span style="color:blue"/>AUC_1</span> = %.2f,', test.area_under_curve);
        fprintf(f, '<span style="color:red">AUC_2</span> = %.2f]</b><br/>', test_harder.area_under_curve);
        fprintf(f, '<b>[#(train) = %d, #(test) = %d, #(total) = %d]</b><br/>', length(data.train_images) + length(data.val_images), length(data.test_images), sum(strcmp(urban.data.cities, city_string)));
        fprintf(f, '<img src="%d_pr.jpg"/>', split_id);

        Learning.PlotTestPredictions(f, config, data_harder, test_harder, city_string_harder, 'classification');
        Learning.PlotTestPredictions(f, config, data, test, city_string, 'classification');

        fprintf(f, '</body></html>');
        fclose(f);
        
        clear data; clear data_harder;
      end
    end
    namesave = [config.results_path, "/", config.experiment_id, "_", city_identifier, "/", metric_str, '/results.mat'];
    save(namesave, 'delta_aucs', 'delta_aucs_harder', 'delta_set', 'config', 'dataset', 'dataset_harder', 'urban', '-v7');
    ff = fopen(sprintf('%s/%s_%s/%s/results.html', config.results_path, config.experiment_id, city_identifier, metric_str), 'w');
    fprintf(ff, '<html><body>');
    figure;  hold on; 
    line(delta_set, mean(delta_aucs, 2), 'Color', 'blue'); 
    errorbar(delta_set, mean(delta_aucs, 2), std(delta_aucs, 1, 2), 'b');
    %errorbar(delta_set, mean(delta_aucs, 2), std(delta_aucs, 1, 2), 'Marker', 'x', 'markerEdgeColor', 'blue', 'linestyle', 'none', 'color', [0.6 0.3 0.1]);
    line(delta_set, mean(delta_aucs_harder, 2), 'Color', 'red'); 
    errorbar(delta_set, mean(delta_aucs_harder, 2), std(delta_aucs_harder, 1, 2), 'r');
    %errorbar(delta_set, mean(delta_aucs_harder, 2), std(delta_aucs_harder, 1, 2), 'marker', 'x', 'markerEdgeColor', 'red', 'linestyle', 'none', 'color', [0.6 0.3 0.1]);
    axis([0 0.55 0.5 1]);grid;
    print('-dpng', '-r60', sprintf('%s/%s_%s/%s/results.png', config.results_path, config.experiment_id, city_identifier, metric_str));
    close;
    fprintf(ff, '<h3>%s_%s_%s</h3>', config.experiment_id, city_identifier, metric_str);
    fprintf(ff, '<img src="results.png"/><br/>');
    fprintf(ff, '<table border>');
    for delta_ind = 1 : length(delta_set)
      delta = delta_set(delta_ind);
      fprintf(ff, '<tr><td><b>delta = %.2f</b></td>', delta);
      for split_id = 1 : config.split_num
        fprintf(ff, '<td><a href="delta_%d/%d.html">', round(100 * delta), split_id);
        fprintf(ff, '<img src="delta_%d/%d_pr.jpg" height="120"/></a><br/>%.4f</td>', round(100 * delta), split_id, delta_aucs(delta_ind, split_id));
      end
      fprintf(ff, '<td><br><span style="color:blue">mean(AUC)</span> = %.4f (%.4f)</br>', mean(delta_aucs(delta_ind, :)), std(delta_aucs(delta_ind, :)));
      fprintf(ff, '<span style="color:red">mean(AUC)</span> = %.4f (%.4f)</td>', mean(delta_aucs_harder(delta_ind, :)), std(delta_aucs_harder(delta_ind, :)));
      fprintf(ff, '</tr>');
    end
    fprintf(ff, '</table>');
    fprintf(ff, '</body></html>');
    fclose(ff); 
  end
end
