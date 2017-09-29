function [B,Ve,s2] = pgComputeBasisKPCA( W, d, perc, KernelType )

% KPCA using SVD instead of EIG, we get rid of negative eigenvalues
% from the possibly non-positive-semi-define similarity matrix Kww.
%
% This spectral transformation is associated with an inner product within
% a pseudo-Euclidean space, where the Representer theorem still holds.
%
% This implementation was inspired by the discussion in:
% Gang Wu, Edward Chang, Zhihua Zhang, "An Analysis of Transformation on 
% Non-Positive Semidefinite Similarity Matrix for Kernel Machines,"
% (TR, ECE, UCSB, Jue 2005).

[T,n] = size(W); T = T / 2;
t = mean(W,2);
Wc = W - repmat(t, 1, n);

% s2 is the kernel's (squared) scale parameter
[Kww,s2] = compute_RIKs_train( Wc, d, perc, KernelType );

[~,e,V] = svd( Kww ); e = diag(e);

pe = sum(e(1:d)) / sum(e);                        % percent energy retained
V(:,d+1:end)=[]; e(d+1:end)=[];                   % truncate spectrum

% Kww = V * diag(e) * V'
Ve = V * diag( e.^(-1/2) );
B  = Kww * Ve;

fprintf('\n KPCA basis: d = %d (%3.2fT, %d%% variance)\n', d, d/T, round(100*pe));

% ----------------------------------------------------------------------------
