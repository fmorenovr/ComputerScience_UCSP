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

data_years = {2011, 2013, 2019};
metric_set = {'safer', 'unique', 'upperclass'};

config.homedir = [current_path '/'];
config.features_path = [current_path '/features/'];
ensuredir(config.features_path);

config.urban_data_file = [config.homedir '/data/consolidated_data.json'];
config.urban_data_file_type = 'json';

% Sift features with Fisher Vectors settings.
config.kCodebookSizeGmm = 128;
config.lengthCodeBookGmm = 1000;
config.pyramid = {[1 1], [2 2]};

for year_ind = 1 : length(data_years)
  data_year = data_years{year_ind};

  config.image_path = [config.homedir 'data/images/pp1/' num2str(data_year)];
  config.image_url = config.image_path

  % Load data from the Place Pulse dataset.
  urban = UrbanPerception(config.urban_data_file, config.urban_data_file_type, data_year);

  cities = UrbanPerception.ListCities();
  for city_ind = 1:length(cities)
    city = cities{city_ind};
    data_city = urban.getData(city);

    disp(data_city);

    data_city.images = cellfun(@(x){sprintf('%s/%s', config.image_path, x)}, data_city.image_names);

    for metric_ind = 1:length(metric_set)
      metric = metric_set{metric_ind};

      scores = data_city.(['qs_' metric]); %qs_unique qs_upperclass
      image_list = data_city.images;

      % GIST
      gist_feature_matrix = double(VisionImage.ComputeGistFeatures(data_city));

      save([config.features_path 'gist_'  num2str(data_year) '_' city '_' metric '.mat'], 'gist_feature_matrix', 'image_list', 'scores', '-v7');



      % FISHER
      gmm_codebook = VisionImage.BuildSiftCodebookGmm(image_list(1:config.kCodebookSizeGmm), config.kCodebookSizeGmm,  config.kCodebookSizeGmm*config.lengthCodeBookGmm);

      fisher_feature_matrix = VisionImage.ComputeSiftFeatures(data_city, 'fisher', config.pyramid, gmm_codebook);
      
      save([config.features_path 'fisher_'  num2str(data_year) '_' city '_' metric '.mat'], 'fisher_feature_matrix', 'gmm_codebook', 'image_list', 'scores', '-v7');
    end
  end
end
