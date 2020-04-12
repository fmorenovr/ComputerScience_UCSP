if strcmp(config.feature_type, 'decaf')
  fprintf("  DECAF FEATURES \n");
  if ~exist(config.decaf_features_path, 'file')
    image_list = cellfun(@(x){sprintf('%s/%s', config.image_path, x)}, urban.data.image_names);
    decaf_features = compute_decaf_features(config.image_path, image_list, config.decaf_features_path, config.decaf_layer);
    thefeatures = squeeze(decaf_features.features);
    feature_matrix = double(thefeatures);
    for i = 1 : length(image_list)
      [aa, bb, cc] = fileparts(image_list{i});
      assert(strcmp([bb cc], urban.data.image_names{i}));
    end
    save(config.decaf_features_path, 'feature_matrix', 'image_list', 'urban', '-v7');
  else
    load(config.decaf_features_path);
  end
elseif strcmp(config.feature_type, 'gist')
  fprintf("  GIST FEATURES \n");
  if ~exist(config.gist_features_path, 'file')
    data.images = cellfun(@(x){sprintf('%s/%s', config.image_path, x)}, urban.data.image_names);
    feature_matrix = double(VisionImage.ComputeGistFeatures(data));
    image_list = data.images;
    save(config.gist_features_path, 'feature_matrix', 'image_list', 'urban', '-v7');
  else
    load(config.gist_features_path);
  end
else
  fprintf("  FISHER FEATURES \n");
  if ~exist(config.gmm_dictionary_path, 'file')
    image_list = cellfun(@(x){sprintf('%s/%s', config.image_path, x)}, urban.data.image_names);
    image_list = image_list(randperm(length(image_list)));
    gmm_codebook = VisionImage.BuildSiftCodebookGmm(image_list(1:config.kCodebookSizeGmm), config.kCodebookSizeGmm,  config.kCodebookSizeGmm*config.lengthCodeBookGmm);
    save(config.gmm_dictionary_path, 'gmm_codebook');
  else
    load(config.gmm_dictionary_path);
  end

  % Now compute features for all images in our dataset using SIFT + Fisher Vectors.
  if ~exist(config.fisher_features_path, 'file')
    data.images = cellfun(@(x){sprintf('%s/%s', config.image_path, x)}, urban.data.image_names);
    feature_matrix = VisionImage.ComputeSiftFeatures(data, 'fisher', config.pyramid, gmm_codebook);
    image_list = data.images;
    save(config.fisher_features_path, 'feature_matrix', 'image_list', 'urban', '-v7');
  else
    if ~exist('feature_matrix', 'var')
      load(config.fisher_features_path);
    end
  end
end

