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

function [Coeff] = temp_coding2(B, X, knn, binary)

Coeff = X * B';

if knn > 0
%[C,I]= sof(Coeff,[],2);
[C,I] = sort(Coeff,1, 'descend');

b = C(knn,:);

a = repmat(b, size(Coeff,1), 1);

Coeff(Coeff<a)=0;

if binary == 1
    Coeff(Coeff > 0 )=1;
end
end
