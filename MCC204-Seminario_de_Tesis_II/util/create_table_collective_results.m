
% Configure metric values.
config.homedir = '/mnt/raid/data/vicente/urbanperception/';
config.datasource = 'placepulse_2011';

config.metrics = {'safer', 'unique', 'upperclass'};
config.features = {'fisher', 'decaf'};
city_id = 'new_york_city';
config.split_num = 5;

%f = fopen('collective_table.html', 'w');
for fi = 1 : length(config.features)

for mi = 1 : length(config.metrics)

metric_str = config.metrics{mi};
config.feature_type = config.features{fi};
config.results_path = sprintf('%s/output/%s/collective_results/%s/%s_%d_%s_modified.mat', ...
                              config.homedir, config.datasource, city_id, metric_str, ...
                              config.split_num, config.feature_type);
res = load(config.results_path);

fprintf('%s, %s:\t %.4f, %.4f\n', metric_str, config.feature_type, mean(res.f1scores_unary), mean(res.f1scores));

end
end
%fclose(f);
