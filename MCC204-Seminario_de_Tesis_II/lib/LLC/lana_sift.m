function [frames, desc]= lana_sift(I, gridSpacing, patchSize,nrml_threshold)

%I = imread(imgpath);
    if ndims(I) == 3,
        I = im2double(rgb2gray(I));
    else
        I = im2double(I);
    end;
    
    [im_h, im_w] = size(I);
    
    %    if max(im_h, im_w) > maxImSize,
    %        I = imresize(I, maxImSize/max(im_h, im_w), 'bicubic');
    %        [im_h, im_w] = size(I);
    %    end;

    % make grid sampling SIFT descriptors
    remX = mod(im_w-patchSize,gridSpacing);
    offsetX = floor(remX/2)+1;
    remY = mod(im_h-patchSize,gridSpacing);
    offsetY = floor(remY/2)+1;
    
    [gridX,gridY] = meshgrid(offsetX:gridSpacing:im_w-patchSize+1, offsetY:gridSpacing:im_h-patchSize+1);

    %    fprintf('Processing %s: wid %d, hgt %d, grid size: %d x %d, %d patches\n', ...
    %            frames(jj).name, im_w, im_h, size(gridX, 2), size(gridX, 1), numel(gridX));

    % find SIFT descriptors
    siftArr = sp_find_sift_grid(I, gridX, gridY, patchSize, 0.8);
    [siftArr, siftlen] = sp_normalize_sift(siftArr, nrml_threshold);
            
    %mean(siftlen)
    %siftLens = [siftLens; siftlen];

    desc = siftArr';
    x = gridX(:) + patchSize/2 - 0.5;
    y = gridY(:) + patchSize/2 - 0.5;    
            
    frames = [x,y, siftlen]';
