% COMPUTER GENERATED versus NATURAL HUMAN FACES 
% Temporal-spatial-based discrimination
%
% dnductien, 2014
% -------------------------------------------------------------------------
% This code uses functions NRSFM from 
% NRSFM with RIKs from Onur C. Hamsici, Paulo F.U. Gotardo, Aleix M. Martinez
%   "Learning Spatially-Smooth Mappings in Non-Rigid Structure from Motion",
%   European Conference on Computer Vision, ECCV 2012, Firenze, Italy.
%
%
% function [S3] = Reconstruct3D(W2D, method, K, ratioD, kernel)
% Inputs:
% - W2D: a 2n x T matrix of n sets of T feature points.
% - method: 
%   + 'CSF2': Matrix factorization for non-rigid structure
%            from motion (no occlusion)
%   + 'A2': Iteratively-Reweighted NRSFM with RIKs (ECCV 2012)
% - K: A 3K-Rank solution M (see the paper for details). Should be 27.
% - kernel: 'RIKs' or 'aSFM', default: 'aSFM'
% - ratioD: to estimate the d-dimensional subspace. Should be 0.3
%
% Outputs:
% - 3D: 3D shapes reconstructed from NRSFM with RIKs method.

function [S3, Rf] = Reconstruct3D(W2D, method, K, ratioD, kernel)
    [n, ~] = size(W2D);
    d = ceil(ratioD * n / 2);
    scale = max(abs(W2D(:)));
    W = W2D / scale;

    %% Optimization parameters:
    opts = optimset( optimset('fminunc'), 'Display','iter', 'MaxIter', 300, ...
                     'TolFun', 1e-12, 'TolX', eps); opts.Method = 'pgDN';
    %% -----------------------------------------------------------------------------
    % NRSFM OPTIMIZATION: 

    % Estimate block-diagonal rotation matrix D
    [D, Rs] = pgComputeD(W);           % Rs has the rotations in the diagonal of D

    %% Estimate KPCA basis B
    [B, Ve, s2] = pgComputeBasisKPCA(W, d, 0.99, kernel);

    % Factorization W = M*S, with M = D*kron(BX,I): solve for X
    if (method == 'A2')
        [X, M, S, t, S3] = pgA2(W, D, K, B, opts);
    else
        [X, M, S, t, S3] = pgA1(W, D, K, B, [], opts );
    end

    %% Compute Rotation Matrices
    [F, P] = size(S3); F = F/3;

    % make shapes zero mean
    S = S3 - repmat(mean(S3, 2), 1, P );

    Rf = cell(F, 1);

    for f = 1:F
        f2 = 2*f-[1 0]; 
        r = Rs(f2, :);
        n = norm(r(1,:));
        r = r / n;
        r = n * [ r ; cross( r(1,:), r(2,:) ) ];

        if ( det(r) < 0 ), r(3,:) = -r(3,:); disp('alignStruct() det<0!'), end

        Rf{f} = r;
    end
end