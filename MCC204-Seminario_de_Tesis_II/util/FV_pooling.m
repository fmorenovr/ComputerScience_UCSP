% Running Fisher Vector pooling.
% Based on Jianchao Yiang LLC Pooling.
% Vicente Ordonez @ UNC Chapel Hill
function [beta] = FV_pooling(feaSet, boxes, pyramid, gmm_codebook)
descriptors = single(feaSet.feaArr);  % transform to single.

nbox = size(boxes, 1);
beta = cell(nbox, 1);
%tic;
%fprintf('Max-pooling of FV spatial pyramid coding\n');
for i = 1:nbox
    box = boxes(i, :);
    box_beta = [];
    width = box(4)-box(2)+1;
    height = box(3)-box(1)+1;
    %pyramid = pyramids{1};
    pLevels = length(pyramid);
    %pBins = pyramid.^2;
    
    box_pts = feaSet.x >= box(2) & feaSet.x <= box(4) & feaSet.y >= box(1) & feaSet.y <= box(3);
    box_descriptors = descriptors(:, box_pts);

    box_x = feaSet.x(box_pts);
    box_y = feaSet.y(box_pts);
    for iter1 = 1:pLevels,
        
        %nBins = pBins(iter1);
        nBins = pyramid{iter1}(1) * pyramid{iter1}(2);    

        wUnit = width / pyramid{iter1}(2);
        hUnit = height / pyramid{iter1}(1);
    
        % find to which spatial bin each local descriptor belongs
        xBin = ceil((box_x-box(2)+1) / wUnit);
        yBin = ceil((box_y-box(1)+1) / hUnit);
        idxBin = (yBin - 1)*pyramid{iter1}(2) + xBin;
    
        for iter2 = 1:nBins,     
            sidxBin = find(idxBin == iter2);
            if isempty(sidxBin),
                continue;
            end      
            %box_beta(:, end+1) = max(box_code(:, sidxBin), [], 2);
            box_beta(:, end+1) = vl_fisher(box_descriptors(:, sidxBin), gmm_codebook.means, gmm_codebook.covariances, gmm_codebook.priors);%, 'SquareRoot', 'Normalized');
        end
    end
    
    box_beta = box_beta(:);
    %box_beta = box_beta ./ sqrt(sum(box_beta.^2));
    beta{i} = box_beta;
end
beta = cat(2, beta{:});
%fprintf('\n');
%toc;
