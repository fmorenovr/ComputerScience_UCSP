function data = fix_image_names(data, urban)

    for w = 1 : length(data.train_images)
        toks = regexp(data.train_images{w}, '_', 'split');
        disp(toks);
        image_id = toks{2};
        ind = find(strcmp(image_id, urban.data.ids));
        data.train_images{w} = urban.data.image_names{ind};
    end

    for w = 1 : length(data.val_images)
        toks = regexp(data.val_images{w}, '_', 'split');
        disp(toks);
        image_id = toks{2};
        ind = find(strcmp(image_id, urban.data.ids));
        data.val_images{w} = urban.data.image_names{ind};
    end
    
    for w = 1 : length(data.test_images)
        toks = regexp(data.test_images{w}, '_', 'split');
        disp(toks);
        image_id = toks{2};
        ind = find(strcmp(image_id, urban.data.ids));
        data.test_images{w} = urban.data.image_names{ind};
    end
end
