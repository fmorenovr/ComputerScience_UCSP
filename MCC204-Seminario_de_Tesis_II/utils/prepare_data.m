function data = prepare_data(dataset, split_info)

    [~, train_ids] = ismember(split_info.train_ids, dataset.ids);
    assert(all(strcmp(dataset.ids(train_ids), split_info.train_ids')));
    
    [~, val_ids] = ismember(split_info.val_ids, dataset.ids);
    assert(all(strcmp(dataset.ids(val_ids), split_info.val_ids')));
    
    [~, test_ids] = ismember(split_info.test_ids, dataset.ids);
    assert(all(strcmp(dataset.ids(test_ids), split_info.test_ids')));

    data.train_ids = train_ids;
    data.val_ids = val_ids;
    data.test_ids = test_ids;
    
    data.train_labels = dataset.labels(data.train_ids)';
    data.val_labels = dataset.labels(data.val_ids)';
    data.test_labels = dataset.labels(data.test_ids)';

    data.train_images = dataset.images(data.train_ids);
    data.val_images = dataset.images(data.val_ids);
    data.test_images = dataset.images(data.test_ids); 
end
