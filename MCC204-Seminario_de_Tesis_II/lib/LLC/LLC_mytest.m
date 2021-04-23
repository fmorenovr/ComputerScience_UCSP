% =========================================================================
% An example code for the algorithm proposed in
%
%   Jinjun Wang, Jianchao Yang, Kai Yu, Fengjun Lv, Thomas Huang, and Yihong Gong.
%   "Locality-constrained Linear Coding for Image Classification", CVPR 2010.
%
%
% Written by Jianchao Yang @ IFP UIUC
% May, 2010.
% =========================================================================

% -------------------------------------------------------------------------
% parameter setting
pyramid = [1, 2, 4];                % spatial block structure for the SPM
knn = 0                            % number of neighbors for local coding
c = 1;                             % regularization parameter for linear SVM
                                   % in Liblinear package

nRounds = 1;                       % number of random test on the dataset
tr_num  = 30;                       % training examples per category
mem_block = 3000;                   % maxmum number of testing features
                                    % loaded each time  

coding_type=1
binarize=0

pooling_options.normalize=0
pooling_options.coding_type=coding_type;
pooling_options.binarize=binarize;

boosting=0
options.T = 1000;
options.weaklearner  = 0;
options.epsi         = 0.1;
options.lambda       = 1e-2;
options.max_ite      = 1000;
options.T            = 2000;

%dist_train=1
%dist_my_rank=1

% -------------------------------------------------------------------------
% set path
addpath('Liblinear/matlab');        % we use Liblinear package, you need 
                                    % download and compile the matlab
                                    % codes

addpath('../gentleboost');

img_dir = 'image/Caltech101';       % directory for the image database                             
data_dir = 'data/Caltech101';       % directory for saving SIFT descriptors
fea_dir =['features/Caltech101_my_','knn',num2str(knn),'_b',num2str(binarize),'_ct',num2str(coding_type),'_n',num2str(pooling_options.normalize)];
%% directory for saving final image


split_dir = ['exp/Caltech101/split'];
model_dir = [fea_dir,'_model'];

%features
%fea_dir = ['features/Caltech101'];    % directory for saving final image features

% -------------------------------------------------------------------------
% extract SIFT descriptors, we use Prof. Lazebnik's matlab codes in this package
% change the parameters for SIFT extraction inside function 'extr_sift'
% extr_sift(img_dir, data_dir);

%extr_sift(img_dir, data_dir);

% -------------------------------------------------------------------------
% retrieve the directory of the database and load the codebook

Bpath = ['dictionary/Caltech101_SIFT_Kmeans_1024.mat'];

load(Bpath);
nCodebook = size(B, 2);              % size of the codebook


meta_path = 'exp/Caltech101/meta.mat';

if exist(meta_path)
    load( meta_path);
else
    database = retr_database_dir(data_dir);
    save(meta_path,'database');
end

if isempty(database),
    error('Data directory error!');
end


% -------------------------------------------------------------------------
% extract image features

dFea = sum(nCodebook*pyramid.^2);
nFea = length(database.path);

fdatabase = struct;
fdatabase.path = cell(nFea, 1);         % path for each image feature
fdatabase.label = zeros(nFea, 1);       % class label for each image feature



for iter1 = 1:nFea,  
    if ~mod(iter1, 5),
        fprintf('.');
    end
    if ~mod(iter1, 100),
        fprintf(' %d images processed\n', iter1);
    end
    fpath = database.path{iter1};
    flabel = database.label(iter1);

    [rtpath, fname] = fileparts(fpath);
    feaPath = fullfile(fea_dir, num2str(flabel), [fname '.mat']);
    
    if 0
        load(fpath);
        fea = my_pooling(feaSet, B, pyramid, knn, pooling_options);
        label = database.label(iter1);

        if ~isdir(fullfile(fea_dir, num2str(flabel))),
            mkdir(fullfile(fea_dir, num2str(flabel)));
        end      
        save(feaPath, 'fea', 'label');
    end
    
    fdatabase.label(iter1) = flabel;
    fdatabase.path{iter1} = feaPath;
end;


% -------------------------------------------------------------------------
% evaluate the performance of the image feature using linear SVM
% we used Liblinear package in this example code

fprintf('\n Testing...\n');
clabel = unique(fdatabase.label);
nclass = length(clabel);
accuracy = zeros(nRounds, 1);

for ii = 1:nRounds,
    fprintf('Round: %d...\n', ii);
    tr_idx = [];
    ts_idx = [];

    split_path = [split_dir,'/',num2str(ii),'.mat'];
    if exist(split_path,'file')
        load(split_path);
    else
        for jj = 1:nclass,
            idx_label = find(fdatabase.label == clabel(jj));
            num = length(idx_label);
            
            idx_rand = randperm(num);
            
            tr_idx = [tr_idx; idx_label(idx_rand(1:tr_num))];
            ts_idx = [ts_idx; idx_label(idx_rand(tr_num+1:end))];

        end
        save(split_path, 'tr_idx', 'ts_idx');
    end
    
    fprintf('Training number: %d\n', length(tr_idx));
    fprintf('Testing number:%d\n', length(ts_idx));
    
    % load the training features 
    tr_fea = zeros(length(tr_idx), dFea);
    tr_label = zeros(length(tr_idx), 1);
    
    for jj = 1:length(tr_idx),
        fpath = fdatabase.path{tr_idx(jj)};
        load(fpath, 'fea', 'label');
        tr_fea(jj, :) = fea';
        tr_label(jj) = label;
    end
    
    %options = ['-c ' num2str(c) '-s 5 '];
    if exist('no_train','var') && no_train ==1
       for jj = 1:nclass
            my_model_path=[model_dir,'/',num2str(ii),'_',num2str(jj),'.mat'];
            load(my_model_path);
            if jj==1
                model_gentle = my_model_gentle;
            else
                model_gentle.featureIdx = [model_gentle.featureIdx, my_model_gentle.featureIdx];
                model_gentle.th = [model_gentle.th, my_model_gentle.th];
                model_gentle.a = [model_gentle.a, my_model_gentle.a];
                model_gentle.b = [model_gentle.b, my_model_gentle.b];
            end
       end
    else
    if boosting==1
        if exist('dist_train', 'var') && dist_train == 1
            my_model_path=[model_dir,'/',num2str(ii),'_',num2str(dist_my_rank),'.mat'];
            my_tr_label = tr_label;
            my_tr_label(tr_label==clabel(dist_my_rank))=1;
            my_tr_label(tr_label~=clabel(dist_my_rank))=-1;
            my_model_gentle  = gentleboost_model(tr_fea' , my_tr_label' , options.T, options);
            save(my_model_path,'my_model_gentle');
            exit
        else
            tic;
            model_gentle  = gentleboost_model(tr_fea' , tr_label' , options.T, options);
            toc;
        end
    else
        options = ['-c ' num2str(c) ];
        model = train(double(tr_label), sparse(tr_fea), options);
        clear tr_fea;
    end
    end

    % load the testing features
    ts_num = length(ts_idx);
    ts_label = [];
    
    if ts_num < mem_block,
        % load the testing features directly into memory for testing
        ts_fea = zeros(length(ts_idx), dFea);
        ts_label = zeros(length(ts_idx), 1);

        for jj = 1:length(ts_idx),
            fpath = fdatabase.path{ts_idx(jj)};
            load(fpath, 'fea', 'label');
            ts_fea(jj, :) = fea';
            ts_label(jj) = label;
        end

        if boosting==1
            [C, fxt_test] = gentleboost_predict(ts_fea', model_gentle, options);
            C=C';
        else
            [C] = predict(ts_label, sparse(ts_fea), model);
        end
    else
        % load the testing features block by block
        num_block = floor(ts_num/mem_block);
        rem_fea = rem(ts_num, mem_block);
        
        curr_ts_fea = zeros(mem_block, dFea);
        curr_ts_label = zeros(mem_block, 1);
        
        C = [];
        
        for jj = 1:num_block,
            block_idx = (jj-1)*mem_block + (1:mem_block);
            curr_idx = ts_idx(block_idx); 
            
            % load the current block of features
            for kk = 1:mem_block,
                fpath = fdatabase.path{curr_idx(kk)};
                load(fpath, 'fea', 'label');
                curr_ts_fea(kk, :) = fea';
                curr_ts_label(kk) = label;
            end    
            
            % test the current block features
            ts_label = [ts_label; curr_ts_label];
            if boosting==1
                [curr_C, fxt_test] = gentleboost_predict(curr_ts_fea',model_gentle, options);
                curr_C = curr_C' + 1;
            else
                [curr_C] = predict(curr_ts_label, sparse(curr_ts_fea),model);
            end
            C = [C; curr_C];

        end
        
        curr_ts_fea = zeros(rem_fea, dFea);
        curr_ts_label = zeros(rem_fea, 1);
        curr_idx = ts_idx(num_block*mem_block + (1:rem_fea));
        
        for kk = 1:rem_fea,
            fpath = fdatabase.path{curr_idx(kk)};
            load(fpath, 'fea', 'label');
            curr_ts_fea(kk, :) = fea';
            curr_ts_label(kk) = label;
        end  
        
        ts_label = [ts_label; curr_ts_label];
        if boosting==1
            [curr_C, fxt_test] = gentleboost_predict(curr_ts_fea',model_gentle, options);
            curr_C = curr_C' + 1;
        else
            [curr_C] = predict(curr_ts_label, sparse(curr_ts_fea), model);
        end
        C = [C; curr_C];        

    end
    
    % normalize the classification accuracy by averaging over different
    % classes
    acc = zeros(nclass, 1);

    for jj = 1 : nclass,
        c = clabel(jj);
        idx = find(ts_label == c);
        curr_pred_label = C(idx);
        curr_gnd_label = ts_label(idx);    
        acc(jj) = length(find(curr_pred_label == curr_gnd_label))/length(idx);
    end

    accuracy(ii) = mean(acc); 
    fprintf('Classification accuracy for round %d: %f\n', ii, accuracy(ii));
end

Ravg = mean(accuracy);                  % average recognition rate
Rstd = std(accuracy);                   % standard deviation of the recognition rate

fprintf('===============================================');
fprintf('Average classification accuracy: %f\n', Ravg);
fprintf('Standard deviation: %f\n', Rstd);    
fprintf('===============================================');

