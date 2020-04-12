% Vicente Ordonez @ UNC Chapel Hill 2013
% VisionImage with feature computation code routines.
% addpath(genpath('../lib/stuff_detectors_release')); 
% addpath(genpath('../lib/vlfeat-0.9.17')); vl_setup;
% addpath(genpath('../lib/LLC'));
% addpath(genpath('../lib/util'));
% addpath(genpath('../lib/gist'));
classdef VisionImage < BaseImage
    properties
		tc_handler = [];
    end
    
    methods
        function obj = VisionImage(image_fname)
            obj = obj@BaseImage(image_fname);
        end

		function handler = get.tc_handler(obj)
			if isempty(obj.tc_handler)
				obj.tc_handler = TCFeat;
				obj.tc_handler.process(obj.image);
				fprintf('Computing TC features\n');
			end
			handler = obj.tc_handler;
		end
        
        function feat = gist(obj)
            % param.imageSize = [256 256]; % it works also with non-square images
            param.orientationsPerScale = [8 8 8 8];
            param.numberBlocks = 4;
            param.fc_prefilt = 4;
            [gist1, param] = LMgist(obj.image, '', param);
            feat = gist1;
        end		

		function feat = texton_color_feature_endres(obj, varargin)
			%handler = TCFeat;
			%handler.process(obj.image);	
			feat.raw = obj.tc_handler.processed;
			if nargin == 1
				feat.histogram = obj.tc_handler.feat([1 1 size(obj.image, 2) size(obj.image, 1)]);
			else
				boundaries_map = varargin{1};
				labels = unique(boundaries_map(:));
				for i = 1 : length(labels)
					segment_map = boundaries_map == labels(i);
					feat.histogram(i, :) = obj.tc_handler.feat_segment(segment_map);
				end
			end
		end

        % Bag-of-words hog feature by Ian Endres.
		function feat = hog_feature_endres(obj)
			handler = HOGFeat;
			handler.process(obj.image);
			feat.raw = handler.processed;
			feat.histogram = handler.feat([1 1 size(obj.image, 2) size(obj.image, 1)]);
		end

        % Raw hog feature from Vlfeat implementation of Pedro's Raw Hog feature.
        % It quantizes the image aspect ratio to a set of predefined aspect ratios.
		function feat = hog_feature_exemplar(obj)
			h = size(obj.image, 1);
			w = size(obj.image, 2);
			ratio = h / w;
			
			% Find the closest ratio among a predetermined group of image ratios.
			ratios = [6/12, 7/12, 8/12, 9/12, 1, 12/9, 12/8, 12/7, 12/6];
			[xx, selected] = min(abs(ratios - ratio));
			quantized_ratio = ratios(selected);

			% Scale the image to the target image ratio dimensions.
			if abs(quantized_ratio - 1) < eps,
				h_dim = 10 * 8; w_dim = 10 * 8;
			else
				if w > h,
					w_dim = 12 * 8; h_dim = 12 * 8 * quantized_ratio;
				else
					h_dim = 12 * 8; w_dim = 12 * 8 / quantized_ratio;
				end
			end
			im = im2single(imresize(obj.image, [h_dim, w_dim]));
			kCellSize = 8;
			feat = vl_hog(im, kCellSize);
		end

        % Multi-scale hog feature from vl_feat. (Only 3 scales!)
		function feat = hog_feature_multiscale(obj)
			kCellSize = 8;
			img = im2single(obj.image);
			
			feat{1} = vl_hog(img, kCellSize);
			feat{2} = vl_hog(imresize(img, [obj.height/2 obj.width/2]), kCellSize);
			feat{3} = vl_hog(imresize(img, [obj.height/4 obj.width/4]), kCellSize);
		end

        % Single scale raw hog feature.
		function feat = hog_feature_pedro(obj)
			kCellSize = 8;
			feat = vl_hog(im2single(obj.image), kCellSize);	
		end

        % Sift feature descriptor using a codebook and an encoding method.
		function [betas, codes, sfeat] = sift_feature_encoding(obj, encoding, codebook, pyramid, boxes)
            fprintf('Computing raw features...');tic;
			sfeat_raw = obj.sift_feature();
            fprintf('took %f\n', toc);
 
			if isempty(boxes)
				boxes = [1, 1, sfeat_raw.height, sfeat_raw.width];  % Whole image.
			end
			sfeat.feaArr = double(sfeat_raw.descs);
			sfeat.x = sfeat_raw.frames(1, :);
			sfeat.y = sfeat_raw.frames(2, :);
	        sfeat.width = sfeat_raw.width;
            sfeat.height = sfeat_raw.height;
	
            % Encode all the features. 	
            fprintf('Running %s encoding...', encoding);tic;
            switch encoding
                case 'bow'
                    [codes] = LLC_encoding(sfeat, double(codebook.centers), 1, []);
                    betas = LLC_pooling(sfeat, codes, boxes, pyramid);
                case 'llc'
                    knn = 5;
			        [codes] = LLC_encoding(sfeat, double(codebook.centers), knn, []);
                    betas = LLC_pooling(sfeat, codes, boxes, pyramid);
                case 'fisher'
                    betas = FV_pooling(sfeat, boxes, pyramid, codebook);
                    codes = [];
            end
            fprintf('took %f\n', toc);
        end

        % Raw sift feature from vl_feat at 3 different resolutions.
		function sift = sift_feature(obj)
			imsizes = [1 0.75 0.5 0.38 0.25];
      dSiftOpts = {'norm', 'fast', 'floatdescriptors', 'step', 3, 'size', 8, 'geometry', [4 4 8]};
			img = rgb2gray(im2single(obj.image));
			s.width = size(img, 2); s.height = size(img, 1);
			s.descs = []; s.frames = []; s.scales = [];
			for j = 1:numel(imsizes)
				im = imresize(img, imsizes(j));
				[myframes mydescs] = vl_dsift(im, dSiftOpts{:});
				s.frames = [s.frames myframes/imsizes(j)];
				s.descs = [s.descs sqrt(mydescs)];  % sqrt = RootSift
				s.scales = [s.scales, 8 * ones(1, size(myframes, 2)) / imsizes(j)];
			end
			sift = s;
		end

    end
   
    methods(Static)
        % Collect features in a single matrix to compute a codebook either using kmeans or gmm.
        function allfeatures = CollectFeaturesForCodebook(data, kCodebookSize, feature_pool_size)
            if nargin < 3
                feature_pool_size = kCodebookSize * 100;
            end
            if isfield(data, 'train_images')
                image_list = data.train_images;
            else
                image_list = data;
            end

            % Use all the supplied images to build the codebook.
            use_this_many_images = round(length(image_list));
            if use_this_many_images < (kCodebookSize / 100)
                fprintf('Warning: You might not have enough images for a codebook this big');
            end
            fprintf('Using %d random images for building dictionary\n', use_this_many_images);
            
            % Compute features for the images selected for building the codebook.
            features = cell(use_this_many_images, 1);
            randseries = randperm(use_this_many_images);
            for i = 1 : use_this_many_images
                im = VisionImage(image_list{randseries(i)});
                feat = im.sift_feature();
                features{i} = feat.descs';
                fprintf('%d. Computing features for image %d\n', i, randseries(i));
            end
        
            fprintf('Using %d feature vectors for building dictionary\n', feature_pool_size);
            % Now will use  100 * kCodebookSize the number of features.
            allfeatures = cat(1, features{:});
            allfeatures = single(allfeatures(randperm(feature_pool_size), :))';
        end

        % Build a codebook using K-means.
        function bow_codebook = BuildSiftCodebookKmeans(data, kCodebookSize)
            allfeatures = VisionImage.CollectFeaturesForCodebook(data, kCodebookSize);

            fprintf('\nBuilding codebook using K-means...');tic;
            vocab = vl_kmeans(allfeatures, kCodebookSize, 'verbose',  'algorithm', 'elkan', 'MaxNumIterations', 30);

            fprintf('done in %f\n', toc);
            bow_codebook.centers = vocab;
            bow_codebook.codebook_size = kCodebookSize;
        end
       
        % Build a codebook using Gaussian Mixture Models.
        function fisher_codebook = BuildSiftCodebookGmm(data, kCodebookSize, kFeaturePoolSize)
            if nargin < 3
                kFeaturePoolSize = 100 * kCodebookSize;
            end
            allfeatures = VisionImage.CollectFeaturesForCodebook(data, kCodebookSize, kFeaturePoolSize);

            fprintf('\nBuilding codebook using GMMs...');tic;
            [means, covariances, priors] = vl_gmm(allfeatures, kCodebookSize); 
            fprintf('done in %f\n', toc);

            fisher_codebook.means = means;
            fisher_codebook.covariances = covariances;
            fisher_codebook.priors = priors;
            fisher_codebook.codebook_size = kCodebookSize;
        end

        % Compute Gist features for a set of images.
        function feature_set = ComputeGistFeatures(data)
            if isfield(data, 'train_images')
                image_list = [data.train_images; data.val_images; data.test_images];
                [thrash, train_ids] = ismember(data.train_images, image_list);
                [thrash, val_ids] = ismember(data.val_images, image_list);
                [thrash, test_ids] = ismember(data.test_images, image_list);
            else
                image_list = data.images;
            end
            
            fun = @(x) VisionImage(x).gist();

            features = cell(1, length(image_list));
            %features = pararrayfun(nproc, fun, image_list);
            %parpool('local', 12);
            %parfor i = 1 : length(image_list)
            for i = 1 : length(image_list)
                obj = VisionImage(image_list{i});
                fprintf('\n%d. Computing features for image %d of %d\n', i, i, length(image_list)); 
                features{i} = obj.gist(); 
            end
            features = cat(1, features{:});
            
            if isfield(data, 'train_images')
                feature_set.train_features = features(train_ids, :);
                feature_set.val_features = features(val_ids, :);
                feature_set.test_features = features(test_ids, :);
            else
                feature_set = features;
            end
        end 
	
        % Compute features for a set of images.
        function feature_set = ComputeSiftFeatures(data, encoding, pyramid, codebook)
            if isfield(data, 'train_images')
                image_list = [data.train_images; data.val_images; data.test_images];
                [thrash, train_ids] = ismember(data.train_images, image_list);
                [thrash, val_ids] = ismember(data.val_images, image_list);
                [thrash, test_ids] = ismember(data.test_images, image_list);
            else
                image_list = data.images;
            end

            feat_size = codebook.codebook_size;
            features = cell(1, length(image_list));
            for i = 1 : length(image_list)
                obj = VisionImage(image_list{i});
                fprintf('\n%d. Computing features for image %d of %d\n', i, i, length(image_list)); 
                features{i} = obj.sift_feature_encoding(encoding, codebook, pyramid, []); 
            end
            features = cat(2, features{:});
            
            if isfield(data, 'train_images')
                feature_set.train_features = features(:, train_ids)';
                feature_set.val_features = features(:, val_ids)';
                feature_set.test_features = features(:, test_ids)';
            else
                feature_set = features';
            end
        end 
	
        % Visualize Hog descriptor.	
        function imhog_jet = RawHog2Image(image, raw_hog)
			imhog = vl_hog('render', raw_hog, 'verbose');
			imhog = (imhog - min(imhog(:))) / (max(imhog(:)) - min(imhog(:)));	
			imhog_jet = ind2rgb(gray2ind(imhog, 255), jet(255));
			imhog_jet = imresize(imhog_jet, [size(image, 1), size(image, 2)]);
		end

        % 2D Gaussian function.
        function val = gaussC(x, y, sigma, center)
            xc = center(1);
            yc = center(2);
            exponent = ((x-xc).^2 + (y-yc).^2)./(2*sigma);
            val  = (exp(-exponent));
        end
    end
end

