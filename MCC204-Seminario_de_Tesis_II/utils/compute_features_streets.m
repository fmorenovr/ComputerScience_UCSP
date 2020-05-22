if strcmp(config.feature_type, 'cnn')
  fprintf("CNN FEATURES \n");
  if ~exist(config.cnn_features_path, 'file')
    cnn_features = compute_cnn_features(config.image_path, image_list, config.cnn_features_path, config.cnn_layer, 0);
    thefeatures = squeeze(cnn_features.features);
    feature_matrix = double(thefeatures);
    for i = 1 : length(image_list)
      [aa, bb, cc] = fileparts(image_list{i});
      assert(strcmp([bb cc], urban.data.image_names{i}));
    end
    save(config.cnn_features_path, 'feature_matrix', '-v7');
  else
    obj_loaded = load(config.cnn_features_path);
    feature_matrix = obj_loaded.feature_matrix;
  end
elseif strcmp(config.feature_type, 'gist')
  fprintf("  GIST FEATURES \n");
  if ~exist(config.gist_features_path, 'file')
    feature_matrix = double(VisionImage.ComputeGistFeatures(data_city));
    save(config.gist_features_path, 'feature_matrix', '-v7');
  else
    obj_loaded = load(config.gist_features_path);
    feature_matrix = obj_loaded.feature_matrix;
  end
elseif strcmp(config.feature_type, 'fisher')
  fprintf("  FISHER FEATURES \n");
  if ~exist(config.gmm_dictionary_path, 'file')
    gmm_list = image_list(randperm(length(image_list)));
    gmm_codebook = VisionImage.BuildSiftCodebookGmm(gmm_list(1:config.kCodebookSizeGmm), config.kCodebookSizeGmm, config.kCodebookSizeGmm*config.lengthCodeBookGmm);
    save(config.gmm_dictionary_path, 'gmm_codebook');
  else
    obj_loaded = load(config.gmm_dictionary_path);
    gmm_codebook = obj_loaded.gmm_codebook;
  end

  % Now compute features for all images in our dataset using SIFT + Fisher Vectors.
  if ~exist(config.fisher_features_path, 'file')
    feature_matrix = VisionImage.ComputeSiftFeatures(data_city, 'fisher', config.pyramid, gmm_codebook);
    save(config.fisher_features_path, 'feature_matrix', '-v7');
  else
    obj_loaded = load(config.fisher_features_path);
    feature_matrix = obj_loaded.feature_matrix;
  end
else
  fprintf("CNN GAP FEATURES \n");
  if ~exist(config.cnn_gap_features_path, 'file')
    cnn_gap_features = compute_cnn_features(config.image_path, image_list, config.cnn_gap_features_path, config.cnn_layer, 1);
    thefeatures = squeeze(cnn_gap_features.features);
    feature_matrix = double(thefeatures);
    for i = 1 : length(image_list)
      [aa, bb, cc] = fileparts(image_list{i});
      assert(strcmp([bb cc], urban.data.image_names{i}));
    end
    save(config.cnn_gap_features_path, 'feature_matrix', '-v7');
  else
    obj_loaded = load(config.cnn_gap_features_path);
    feature_matrix = obj_loaded.feature_matrix;
  end
end

