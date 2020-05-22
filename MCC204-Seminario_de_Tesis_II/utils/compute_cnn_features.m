function decaf_features = compute_cnn_features(image_path, image_list, features_path, layer, gap)
  decaf_features = runCNN(image_list, layer, gap);
end
