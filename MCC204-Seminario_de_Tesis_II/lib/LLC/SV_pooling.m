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

function [beta] = SV_pooling(feaSet, B, pyramid, knn, svs)

%svs: parameter s 

%B: d * M bases
dSize = size(B, 2);
nSmp = size(feaSet.feaArr, 2);

fea_dim = size(feaSet.feaArr,1);

img_width = feaSet.width;
img_height = feaSet.height;
idxBin = zeros(nSmp, 1);

% llc coding
max_mat_size=20*1024*1024;
block_size = ceil(max_mat_size / dSize)
num_block = ceil(nSmp / block_size)

nn_idx = zeros(knn, nSmp);
for i = 1:num_block
    block_start = (i-1) * block_size + 1;
    block_end = min(nSmp, block_start + block_size -1 );
    nn_idx(:,block_start:block_end) = SV_nn(B', feaSet.feaArr(:,block_start:block_end)', knn)';
end

nn_i = repmat(1:nSmp,knn,1);

nn_prob = sparse(nn_idx(:), nn_i(:), 1, dSize, nSmp) / knn;

% spatial levels
pLevels = length(pyramid);
% spatial bins on each level
pBins = pyramid.^2;
% total spatial bins
tBins = sum(pBins);

beta = zeros(dSize * (fea_dim+1) , tBins);
bId = 0;

for iter1 = 1:pLevels,
    
    nBins = pBins(iter1);
    
    wUnit = img_width / pyramid(iter1);
    hUnit = img_height / pyramid(iter1);
    
    % find to which spatial bin each local descriptor belongs
    xBin = ceil(feaSet.x / wUnit);
    yBin = ceil(feaSet.y / hUnit);
    idxBin = (yBin - 1)*pyramid(iter1) + xBin;
    
    for iter2 = 1:nBins,     
        bId = bId + 1;
        sidxBin = find(idxBin == iter2);
        if isempty(sidxBin),
            continue;
        end      

        N = numel(sidxBin);
        chist = sum(nn_prob(:, sidxBin), 2) / N;

        chist2 = chist;
        chist2(chist==0) = 1;

        sumx = feaSet.feaArr(:,sidxBin) * nn_prob(:, sidxBin)';
        
        sumx = bsxfun(@times, sumx, 1./sqrt(chist2)') / N;
        
        V = bsxfun(@times, B, sqrt(chist)');
        
        phi = [(sumx - V); svs * sqrt(chist)'];
        
        beta(:, bId) = phi(:);
    end
end

if bId ~= tBins,
    error('Index number error!');
end

beta = beta(:);
beta = beta./sqrt(sum(beta.^2));
