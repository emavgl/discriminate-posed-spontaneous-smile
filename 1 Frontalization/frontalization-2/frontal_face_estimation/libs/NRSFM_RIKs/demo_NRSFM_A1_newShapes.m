%
% Demo A1: NRSFM with Rotation Invariant Kernels (cross-validation experiment)
% 
% -----------------------------------------------------------------------------
clear all

% Datasets:
strings = {'walking','face1','stretch','dance'};
NUM_DATASET = 2;

% Load data (original 3D shapes and observation matrix with 2D points)
[S0,W,T,n] = pgLoadDataNRSFM( strings{NUM_DATASET} );

%% -----------------------------------------------------------------------------
% Model parameters for each dataset:
Ks  = {  5 ,  5 ,  8 ,  7  };   % number of basis shapes (rank parameter)
ds1 = { 0.2, 0.3, 0.2, 0.2 };   % number of vectors (%T) in KPCA basis (2D RIK)
ds2 = { 0.1, 0.3, 0.2, 0.1 };   % (if using aSFM kernel)

kernels = {'RIK','aSFM'}; 
selK = kernels{2};

K = Ks{ NUM_DATASET };
switch selK
    case  'RIK', d = ceil( ds1{ NUM_DATASET } * T );
    case 'aSFM', d = ceil( ds2{ NUM_DATASET } * T );
end

% Optimization parameters:
opts = optimset( optimset('fminunc'), 'Display','iter', 'MaxIter', 300, ...
                 'TolFun', 1e-12, 'TolX', eps); opts.Method = 'pgDN';

%% -----------------------------------------------------------------------------
% NRSFM OPTIMIZATION: 

% Estimate block-diagonal rotation matrix D
[D,Rs] = pgComputeD( W );           % Rs has the rotations in the diagonal of D

%% nfolds = T;   % leave-one-out
nfolds = 30;

dt = ceil(T/nfolds);                % number of frames per fold
rp = randperm(T);                   % random permutation of frames
S3r = nan( size(S0) );              % recovered 3D shapes for all folds

for f = 1:nfolds
    fprintf('\n Fold %d of %d\n',f,nfolds);
     
    % (0) Select data for current fold
    if numel(rp) >= dt
        tf = rp(end-dt+1:end); rp(end-dt+1:end) = [];  % select,remove tailing frames
    else
        tf = rp(1:end);             % last fold may have less than dt frames
    end
    tf2 = [          2*tf-1 ; 2*tf ]; tf2 = tf2(:);
    tf3 = [ 3*tf-2 ; 3*tf-1 ; 3*tf ]; tf3 = tf3(:);
    Wt = W; Wt(tf2,:) = [];                   % remove 2D shapes from input
    Dt = D; Dt(tf2,:) = []; Dt(:,tf3) = [];   % remove cameras too
    
    % (1) Training: NRSFM

    % (1.a) Estimate KPCA basis B
    [B,Ve,s2] = pgComputeBasisKPCA( Wt, d, 0.99, selK );

    % (1.b) Factorization W = M*S, with M = D*kron(BX,I): solve for X
    [X,M,S,~,S3] = pgA1( Wt, Dt, K, B, [], opts );
    
    % (2) Testing: evaluate learned mapping to reconstruct 2D shapes that were left out
    for i = 1:numel(tf)
        w = W( 2*tf(i)-[1 0], :);
        c = compute_RIKs_test( w, Wt, Ve*X, s2, selK );
        
        S3r( 3*tf(i)-[2 1 0], :) = kron( c, eye(3) ) * S;
    end

    % (3) Display avera reconstruction error up til current fold
    rows  = all( isfinite(S3r), 2 );
    rows2 = rows; rows2(3:3:end) = [];

    if strcmpi(strings{NUM_DATASET}, 'stretch'), args = {}; else args = { Rs(rows2,:) }; end
    [err3D,S0R,S3R,errt] = pgCompare3DShapes( S0(rows,:), S3r(rows,:), args{:} );
    
    fprintf('(Dataset %d, %s), err3D (std) = %5.4f (%5.4f), (K = %d, d = %d)\n', ...
             NUM_DATASET, strings{NUM_DATASET}, err3D, std(errt(:)), K, d )
end

%%pgShowShapes3D(S0R, S3R);
% -----------------------------------------------------------------------------
