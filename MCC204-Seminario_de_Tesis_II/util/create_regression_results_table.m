% Run standard computer vision whole-image regression on a dataset 
% loaded into the "data" data structure.
% Vicente Ordonez @ UNC Chapel Hill
addpath(genpath('../lib/liblinear-1.94/matlab'));
addpath(genpath('../lib/gist'));
addpath(genpath('util'));

% Set unique experiment identifier.
config.experiment_id = 'urbanperception';

% Configure experiment datasource for 2013 images.
config.homedir = '/mnt/raid/data/vicente/urbanperception/';
config.datasource = 'placepulse_2011';
%config.datasource = 'placepulse_2011';

config.normalization = 'unnormalized';

% Configure output directory.
config.output_path = [config.homedir '/output/' config.datasource];

% Load list of cities in the dataset.
cities = UrbanPerception.ListCities();

% Now collect results.
f = fopen(sprintf('table_%s_%s_2.html', config.datasource, config.normalization), 'w');
fprintf(f, '<html><body>');
feature_types = {'gist', 'fisher', 'decaf'};

for fx = 1 : length(feature_types)
config.feature_type = feature_types{fx};
config.results_path = [config.output_path '/regression_results_' config.normalization ...
                       '_' config.feature_type];
fprintf(f, '<h3>%s</h3>', config.feature_type);
metric_set = {'safer', 'unique', 'upperclass'};
cities_harder = cities(end:-1:1);
fprintf(f, '<table border>');
for city_ind = 1 : length(cities)
    city_string = cities{city_ind};
    city_string_harder = cities_harder{city_ind};
    city_identifier = regexprep(lower(city_string), ' ', '_');
    fprintf(f, '<tr><td rowspan=4>%s</td><td></td>', city_string);
    fprintf(f, '<td width=100>%s</td><td width=100>%s</td>', city_string, city_string_harder);
    fprintf(f, '</tr>');
    for metric_ind = 1 : length(metric_set)
        metric_str = metric_set{metric_ind};%metric_str = 'safer';
        % Now run regression.
        fprintf(f, '<tr><td>%s</td>', metric_str); 
        result_fname = sprintf('%s/%s_%s/%s/results.mat', config.results_path, ...
                               config.experiment_id, city_identifier, metric_str);
        fprintf('%s\n', result_fname);
        res = load(result_fname);
        fprintf(f, '<td>%.4f</td>', sqrt(mean(res.results_split(2, :))));
        fprintf(f, '<td>%.4f</td>', sqrt(mean(res.results_split_harder(2, :))));
        fprintf(f, '</tr>\n');
    end
end
fprintf(f, '</table>');
end
fprintf(f, '</body></html>');
fclose(f);


