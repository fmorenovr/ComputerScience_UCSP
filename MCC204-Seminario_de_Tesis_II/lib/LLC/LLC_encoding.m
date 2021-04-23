% ========================================================================
% Written by Jianchao Yang @ IFP UIUC
% May, 2010
% ========================================================================

function [llc_codes] = LLC_encoding(feaSet, B, knn, flann_options)


if ~exist('flann_options', 'var') || isempty(flann_options)
    flann_options=[];
end



dSize = size(B, 2);
nSmp = size(feaSet.feaArr, 2);

img_width = feaSet.width;
img_height = feaSet.height;
idxBin = zeros(nSmp, 1);

% llc coding
max_mat_size=20*1024*1024;

if ~isempty(flann_options)
    max_mat_size=200*1024*1024;
end


block_size = ceil(max_mat_size / dSize);
num_block = ceil(nSmp / block_size);

if ~isempty(flann_options)
    block_size = nSmp;
    num_block=1;
end

tic;
%fprintf('LLC encoding...\n');
llc_codes = sparse(dSize, nSmp);
for i = 1:num_block
    block_start = (i-1) * block_size + 1;
    block_end = min(nSmp, block_start + block_size -1 );
    llc_codes(:,block_start:block_end) = sparse(LLC_coding_appr_fast2(B', feaSet.feaArr(:,block_start:block_end)', knn, [], flann_options)');
end
%fprintf('\n');
%toc;
