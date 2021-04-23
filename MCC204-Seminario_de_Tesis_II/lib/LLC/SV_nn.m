% ========================================================================
% USAGE: [Coeff]=LLC_coding_appr(B,X,knn,lambda)
% Approximated Locality-constraint Linear Coding
%
% Inputs
%       B       -M x d codebook, M entries in a d-dim space
%       X       -N x d matrix, N data points in a d-dim space
%       knn     -number of nearest neighboring
%       lambda  -regulerization to improve condition
%
% Outputs
%       Coeff   -N x M matrix, each row is a code for corresponding X
%
% Jinjun Wang, march 19, 2010
% ========================================================================

function [IDX] = SV_nn(B, X, knn, beta, flann_options)

% IDX: each row is the nn idx for the example

if ~exist('knn', 'var') || isempty(knn),
    knn = 5;
end

if ~exist('beta', 'var') || isempty(beta),
    beta = 1e-4;
end

if ~exist('flann_options', 'var')
    use_flann = 0;
else
    use_flann = 1;
end

if use_flann
    path(path,'/memex/jiadeng/imagenet/hfeat/flann/flann-1.6.8-src/build/src/matlab/');
    path(path,'/memex/jiadeng/imagenet/hfeat/flann/flann-1.6.8-src/src/matlab/');
end

nframe=size(X,1);
nbase=size(B,1);

% find k nearest neighbors
%XX = sum(X.*X, 2);
%BB = sum(B.*B, 2);
%D  = repmat(XX, 1, nbase)-2*X*B'+repmat(BB', nframe, 1);
%IDX = zeros(nframe, knn);
if use_flann
    [d, IDX] = flann_search(flann_options.index, single(X'), knn, flann_options.paras);
    d = d';
    IDX = IDX';
else
    B_norm2 = sum(B.^2,2);
    X_norm2 = sum(X.^2,2);

    D = bsxfun(@plus, - 2 * X * B',B_norm2');
    D = bsxfun(@plus, D, X_norm2)';

if 0
disp('fast');
[d, IDX] = sort(D', 1, 'ascend');
IDX = IDX';
IDX = IDX(:, 1:knn);
else
IDX = zeros(nframe, knn);
for k=1:knn
[d, mi] = min(D,[], 1);
D(sub2ind(size(D), mi, 1:size(D,2))) = inf;
IDX(:,k) = mi';
end
end

end
%for i = 1:nframe,
%	d = D(i,:);
%	[dummy, idx] = sort(d, 'ascend');
%	IDX(i, :) = idx(1:knn);
%end

