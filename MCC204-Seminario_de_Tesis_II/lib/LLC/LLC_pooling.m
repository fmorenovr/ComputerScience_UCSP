% ========================================================================
% Pooling the llc codes to form the image feature
% USAGE: [beta] = LLC_pooling(feaSet, B, pyramid, knn)
% Inputs
%       feaSet      -the coordinated local descriptors
%       B           -the codebook for llc coding
%       pyramid     -the spatial pyramid structure
%       knn         -the number of neighbors for llc coding
% Outputs
%       beta        -the output image feature
%
% Written by Jianchao Yang @ IFP UIUC
% May, 2010
% ========================================================================

function [beta] = LLC_pooling(feaSet, llc_codes, boxes, pyramid)

nbox = size(boxes, 1);
beta = cell(nbox, 1);
%tic;
%fprintf('Max-pooling of LLC codes & spatial pyramid coding\n');
for i = 1:nbox
    box = boxes(i, :);
    box_beta = [];
    width = box(4)-box(2)+1;
    height = box(3)-box(1)+1;
    %pyramid = pyramids{1};
    pLevels = length(pyramid);
    %pBins = pyramid.^2;
    
    box_pts = feaSet.x >= box(2) & feaSet.x <= box(4) & feaSet.y >= box(1) & feaSet.y <= box(3);
    box_code = llc_codes(:, box_pts);
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
            box_beta(:, end+1) = max(box_code(:, sidxBin), [], 2);
        end
    end
    
    box_beta = box_beta(:);
    box_beta = box_beta ./ sqrt(sum(box_beta.^2));
    beta{i} = box_beta;
end
beta = cat(2, beta{:});
%fprintf('\n');
%toc;
