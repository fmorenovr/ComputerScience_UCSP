function decaf_features = compute_decaf_features(image_path, image_list, features_path, layer)
  decaf_features = runDecaf(image_list, layer);
end
