% Run standard computer vision whole-image classification on a dataset 
% loaded into the "data" data structure.
% Vicente Ordonez @ UNC Chapel Hill
addpath(genpath('../lib/vlfeat-0.9.17')); vl_setup;
addpath(genpath('../lib/liblinear-1.94/matlab'));
addpath(genpath('../lib/emat'));
addpath(genpath('util'));

% Set unique experiment identifier.
config.experiment_id = 'urbanperception';

% Configure experiment datasource for 2011 images.
config.homedir = '/mnt/raid/data/vicente/urbanperception/';
config.datasource = 'placepulse_2011';
config.image_url = ['http://tlberg.cs.unc.edu/vicente/urban/data/placepulse_2013/images/'];
config.image_path = [config.homedir '/data/' config.datasource '/images'];
config.urban_data_file = [config.homedir '/data/placepulse_2011/consolidated_data.csv'];
config.urban_data_file_type = 'csv';

% Configure output directory.
config.regression_output = [config.homedir '/output/' config.datasource '/regression_results_unnormalized_decaf'];
config.output_path = [config.homedir '/output/' config.datasource '/heatmaps'];
config.split_num = 10;
ensuredir(config.output_path);

% Load list of cities in the dataset.
cities = UrbanPerception.ListCities();
zoom_levels = UrbanPerception.ListCitiesZoomLevels();

% Load data from the Place Pulse dataset.
urban = UrbanPerception(config.urban_data_file, config.urban_data_file_type);

cities_harder = cities(end:-1:1);
metric_set = {'safer', 'unique', 'upperclass'};
for ii = 1 : length(cities)
    
    % Render heatmaps for ground truth scores.
    alldata = {};
    city_str = cities{ii};
    city_str_harder = cities_harder{ii};
    city_id = regexprep(lower(city_str), ' ', '_');
    city_id_harder = regexprep(lower(city_str_harder), ' ', '_');
    for jj = 1 : length(metric_set)
        metric_str = metric_set{jj};
        data = [];
        data = urban.getLabeledData(city_str, metric_str);
        data.labels = (data.labels - min(data.labels)) / (max(data.labels) - min(data.labels));
        data.title = sprintf('%s_%s_%s', config.experiment_id, city_id, metric_str);
        if strcmp(metric_str, 'safer')
            %data.labels = 1 - data.labels;
            %data.title = sprintf('%s_%s_UNSAFE', config.experiment_id, city_id);
        end
        data.labels = (7 * (data.labels)).^5;
        sorted_labels = sort(data.labels, 'descend');
        data.max_intensity = sorted_labels(1);
        data.zoom_level = zoom_levels(ii);
        data.id = data.title;
        data.center_latitude = mean(data.latitudes);
        data.center_longitude = mean(data.longitudes);
        output_file = sprintf('%s/%s_%s/%s/heatmap.html', config.output_path, ...
                              config.experiment_id, city_id, metric_str);
        obj = EMat('template.html');
        obj.render(output_file);
        alldata{jj} = data;
    end
    data = alldata;
    obj = EMat('template.html');
    output_file = sprintf('%s/%s_%s/heatmap.html', config.output_path, ...
                          config.experiment_id, city_id);
    obj.render(output_file);

    % Render heatmaps for predicted scores.
    alldata = {};
    for jj = 1 : length(metric_set)
        metric_str = metric_set{jj};
        data = [];
        data = urban.getLabeledData(city_str, metric_str);
        % Now load predicted labels for the city_str, metric_str pair.
        % Collect them across splits of the data.
        data.title = sprintf('%s_%s_%s_prediction', config.experiment_id, city_id, metric_str);
        image_list = {}; label_list = {};
        for kk = 1 : config.split_num
            result_file = sprintf('%s/urbanperception_%s/%s/results_split_%d.mat', ...
                                  config.regression_output, city_id, metric_str, kk);
            reg_result = load(result_file);
            image_list{kk} = reg_result.data.test_images;
            label_list{kk} = reg_result.test.predicted_labels;
        end
        image_list = cat(2, image_list{:})';
        label_list = cat(1, label_list{:});

        [xx, xinds] = ismember(data.images, image_list);
        assert(all(xx));  % Make sure all images are there.
        data.labels = label_list(xinds);
        data.labels = (data.labels - min(data.labels)) / (max(data.labels) - min(data.labels));
        data.labels = (8 * (data.labels)).^7;
        sorted_labels = sort(data.labels, 'descend');
        data.max_intensity = sorted_labels(1);
        data.zoom_level = zoom_levels(ii);
        data.id = data.title;
        data.center_latitude = mean(data.latitudes);
        data.center_longitude = mean(data.longitudes);
        output_file = sprintf('%s/%s_%s/%s/heatmap_predicted.html', config.output_path, ...
                              config.experiment_id, city_id, metric_str);
        obj = EMat('template.html');
        obj.render(output_file);
        alldata{jj} = data;
    end
    data = alldata;
    obj = EMat('template.html');
    output_file = sprintf('%s/%s_%s/heatmap_predicted.html', config.output_path, ...
                          config.experiment_id, city_id);
    obj.render(output_file);

    % Render heatmaps for predicted scores trained on a different city.
    alldata = {};
    for jj = 1 : length(metric_set)
        metric_str = metric_set{jj};
        data = [];
        data = urban.getLabeledData(city_str, metric_str);
        % Now load predicted labels for the city_str, metric_str pair.
        % Collect them across splits of the data.
        data.title = sprintf('%s_%s_%s_prediction_harder', config.experiment_id, city_id, metric_str);
        image_list = []; label_list = [];
        max_rsq = 0;
        for kk = 1 : config.split_num
            result_file = sprintf('%s/urbanperception_%s/%s/results_split_%d.mat', ...
                                  config.regression_output, city_id_harder, metric_str, kk);
            reg_result = load(result_file);
            if reg_result.test.R > max_rsq
                image_list = reg_result.data_harder.test_images;
                label_list = reg_result.test_harder.predicted_labels;
                max_rsq = reg_result.test.R;
            end
        end

        [xx, xinds] = ismember(data.images, image_list);
        assert(all(xx));  % Make sure all images are there.
        data.labels = label_list(xinds);
        data.labels = (data.labels - min(data.labels)) / (max(data.labels) - min(data.labels));
        data.labels = (8 * (data.labels)).^7;
        sorted_labels = sort(data.labels, 'descend');
        data.max_intensity = sorted_labels(1);
        data.zoom_level = zoom_levels(ii);
        data.id = data.title;
        data.center_latitude = mean(data.latitudes);
        data.center_longitude = mean(data.longitudes);
        output_file = sprintf('%s/%s_%s/%s/heatmap_predicted_harder.html', config.output_path, ...
                              config.experiment_id, city_id, metric_str);
        obj = EMat('template.html');
        obj.render(output_file);
        alldata{jj} = data;
    end
    data = alldata;
    obj = EMat('template.html');
    output_file = sprintf('%s/%s_%s/heatmap_predicted_harder.html', config.output_path, ...
                          config.experiment_id, city_id);
    obj.render(output_file);

    % Render heatmaps for predicted scores using linear SVM binary classification.


end

