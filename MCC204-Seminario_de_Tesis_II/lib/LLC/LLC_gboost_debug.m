%clear all; close all; clc;

% -------------------------------------------------------------------------
% parameter setting
pyramid = [1, 2, 4];                % spatial block structure for the SPM
knn = 0;                            % number of neighbors for local coding
c = 1;                             % regularization parameter for linear SVM
                                   % in Liblinear package

nRounds = 1;                       % number of random test on the dataset
tr_num  = 30;                       % training examples per category
mem_block = 100;                   % maxmum number of testing features
                                   % loaded each time  

coding_type=2

binarize=1;

do_train = 1;
test_on_train = 0;
options.T = 1;
options.use_misvm = 1;
options.misvm_C = 1000;
options.misvm_max_iter = 10;
options.misvm_epsilons = [ 0.1 0.1 0.1 ];
options.misvm_bias=1;
options.adaboost=1;

max_T_start = 1;
max_T_interval = 1;
max_nclass = 5;

dist_train=0;
options.dist_folder = sprintf('/memex/jiadeng/dclass/gboost_dist/%dc',max_nclass);


if ~exist(options.dist_folder) 
    mkdir(options.dist_folder);
end

% -------------------------------------------------------------------------
% set path
addpath('Liblinear/matlab');        % we use Liblinear package, you need 
                                    % download and compile the matlab codes

addpath('../');        % we use Liblinear package, you need 

img_dir = 'image/Caltech101';       % directory for the image database                             
data_dir = 'data/Caltech101';       % directory for saving SIFT descriptors
fea_dir =['features/Caltech101_raw'];
%% directory for saving final image
%features
%fea_dir = ['features/Caltech101'];    % directory for saving final image features

% -------------------------------------------------------------------------
% extract SIFT descriptors, we use Prof. Lazebnik's matlab codes in this package
% change the parameters for SIFT extraction inside function 'extr_sift'
% extr_sift(img_dir, data_dir);

%extr_sift(img_dir, data_dir);

% -------------------------------------------------------------------------
% retrieve the directory of the database and load the codebook
database = retr_database_dir(data_dir);

if isempty(database),
    error('Data directory error!');
end

Bpath = ['dictionary/Caltech101_SIFT_Kmeans_1024.mat'];

load(Bpath);
nCodebook = size(B, 2);              % size of the codebook
codebook = B';

% -------------------------------------------------------------------------
% extract image features

%dFea = sum(nCodebook*pyramid.^2);
dFea = 128;
nFea = length(database.path);

fdatabase = struct;
fdatabase.path = cell(nFea, 1);         % path for each image feature
fdatabase.label = zeros(nFea, 1);       % class label for each image feature
fdatabase.num_points = zeros(nFea,1);

if 0
    for iter1 = 1:nFea
        if ~mod(iter1, 5),
            fprintf('.');
        end
        if ~mod(iter1, 100),
            fprintf(' %d images processed\n', iter1);
        end
        fpath = database.path{iter1};
        flabel = database.label(iter1);


        load(fpath);
        [rtpath, fname] = fileparts(fpath);
        feaPath = fullfile(fea_dir, num2str(flabel), [fname '.mat']);

        fdatabase.num_points(iter1) = size(feaSet.feaArr,2);
        fdatabase.fea_dim = size(feaSet.feaArr,1);
        

        %fea = my_pooling(feaSet, B, pyramid, knn, binarize, coding_type);
        label = database.label(iter1);

        if ~isdir(fullfile(fea_dir, num2str(flabel))),
            mkdir(fullfile(fea_dir, num2str(flabel)));
        end      

        save(feaPath, 'feaSet', 'label');
        
        fdatabase.label(iter1) = flabel;
        fdatabase.path{iter1} = feaPath;
    end

    save('cal101_fdatabase.mat','fdatabase');
else
    load('cal101_fdatabase.mat','fdatabase');
end

% -------------------------------------------------------------------------
% evaluate the performance of the image feature using linear SVM
% we used Liblinear package in this example code

fprintf('\n Testing...\n');
clabel = unique(fdatabase.label);
nclass = length(clabel);
accuracy = zeros(nRounds, 1);


stream = RandStream('mrg32k3a');
tmp = randperm(stream,nclass);
clabel = clabel(tmp(1:max_nclass));
nclass = max_nclass;

for ii = 1:nRounds,
    fprintf('Round: %d...\n', ii);
    tr_idx = [];
    ts_idx = [];
    
    for jj = 1:nclass,
        idx_label = find(fdatabase.label == clabel(jj));
        num = length(idx_label);
        
        idx_rand = randperm(stream,num);
        
        tr_idx = [tr_idx; idx_label(idx_rand(1:tr_num))];
        if test_on_train == 1
            ts_idx = tr_idx;
        else
            ts_idx = [ts_idx; idx_label(idx_rand(tr_num+1:end))];
        end
        %ts_idx = [ts_idx; idx_label(idx_rand(tr_num+1:tr_num+tr_num))];
        
    end
    
    fprintf('Training number: %d\n', length(tr_idx));
    fprintf('Testing number:%d\n', length(ts_idx));
    
    % load the training features 

    L = 3;
    grids = get_spm_grids(L);

    if do_train
        %% TODO: get    
        [X, grp, coord, tr_label] = fea2x(fdatabase, tr_idx, dFea);

        fprintf('precompute resp...');
        %tic;
        %grp_codebook_resp = compute_max_resp_block(X, grp', coord, grids,
        %codebook);
        %grp_codebook_resp = compute_max_resp(X, grp', coord, grids, codebook);
        %toc;

        resp_file=sprintf('%s/grp_codebook_resp.mat',options.dist_folder);
        if exist(resp_file)
            resp_file
            load(resp_file);
        else
            tic;
            grp_codebook_resp = compute_max_resp_block_mex(X, grp', coord, grids, codebook);
            toc;
            save(sprintf('%s/grp_codebook_resp.mat',options.dist_folder),'grp_codebook_resp');
        toc;
        end

        %% TODO 

        fprintf('Start training...');    
        %options = ['-c ' num2str(c) '-s 5 '];
        %        if exist('dist_train','var') && dist_train == 1 

        %            tic;
                    model = gboost_train_multiclass_debug(X, grp, tr_label, coord, grids, codebook, grp_codebook_resp, options);
        %            toc;
        %        else
           

    end

end