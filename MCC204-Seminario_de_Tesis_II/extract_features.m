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
config.image_path = [config.homedir '/data/images/pp1/' num2str(config.data_year)];
config.image_url = config.image_path
%config.urban_data_file = [config.homedir '/data/' config.datasource '/consolidated_data.csv'];
%config.urban_data_file_type = 'csv';
config.urban_data_file = [config.homedir '/data/consolidated_data.json'];
config.urban_data_file_type = 'json';

% Decaf features configuration.
config.decaf_layer = 'fc6_cudanet_out';

% Sift features with Fisher Vectors settings.
config.kCodebookSizeGmm = 128;
config.lengthCodeBookGmm = 1000;
config.pyramid = {[1 1], [2 2]};

config.output_features_path = [config.features_path 'features_' num2str(config.data_year) '.mat'];

% Load data from the Place Pulse dataset.
urban = UrbanPerception(config.urban_data_file, config.urban_data_file_type, config.data_year);

data_city = urban.getData("Boston");

disp(data_city);

data_city.images = cellfun(@(x){sprintf('%s/%s', config.image_path, x)}, data_city.image_names);

scores = data_city.(['qs_safer']);
image_list = data_city.images;

% GIST
gist_feature_matrix = double(VisionImage.ComputeGistFeatures(data_city));


% FISHER
gmm_codebook = VisionImage.BuildSiftCodebookGmm(image_list(1:config.kCodebookSizeGmm), config.kCodebookSizeGmm,  config.kCodebookSizeGmm*config.lengthCodeBookGmm);

fisher_feature_matrix = VisionImage.ComputeSiftFeatures(data_city, 'fisher', config.pyramid, gmm_codebook);
save(config.output_features_path, 'gist_feature_matrix', 'fisher_feature_matrix', 'image_list', 'scores', '-v7');

