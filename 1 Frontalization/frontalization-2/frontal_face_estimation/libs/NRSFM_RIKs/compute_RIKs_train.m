function [Kww,s2,Kww0] = compute_RIKs_train( W, d, perc, type )
%function [Kww,s2,Kww0] = compute_RIKs_train( W, d, perc, type )
%
% Evaluate 2D RIK and aSFM rotation invariant kernels
% on all pairs of input 2D shapes in the observation matrix W
%
% d    : the desired number of dimensions of the KPCA basis B in C = BX
% perc : gives the percentage of total energy in the PCA step
% type : the kernel type, 'RIK' or 'aSFM'
%
if ~exist('type','var'), type = 'aSFM'; end

% Kww0 : square "distances" (dissimilarity) for all pairs of 2D shapes 
% Kww  : kernel matrix in the RIK kernel space, i.e.,
%        Kww = exp( Kww0 ./ (-2*sigma^2) );

if strcmpi( type, 'aSFM' )
    Kww0 = pgComputeASFM( W );
else
    Kww0 = pgCompute_2D_RIK( W );
end

% Set the kernel scale s2 automatically using d and perc
[s2,Kww] = pgFindSigmaFromd( Kww0, d, perc );

% -----------------------------------------------------------------------------

function [s2,Kww] = pgFindSigmaFromd( E2, d, perc )

NN = sort( E2, 2,'ascend');      % sort reprojection errors of affine rigid SFM
T  = sum(~isnan( NN(1,:) ));     % max # of computed neighbors (comparisons)

dk = 0;                          % d = f(s2) computed in current PCA step
p0 = 1; p1 = T;                  % initial interval for binary search of s2 = g(knn)

% (1) Given d and perc, define sigma^2 by binary search over nearest neighbors
while (dk ~= d && p1 > p0 )
    knn = ceil( (p0+p1)/2 );     % begins with median
    s2 = nanmean( NN(:,knn) );   % kernel scale parameter sigma^2
    
    % Evaluate kernel function
    Kww = exp( E2 / (-2*s2) );
    %Kww = 0.5 * (Kww + Kww');
    
    e = svd( Kww );

    % find number of dimensions d to represent perc % of total variation
    dk = find( cumsum(e) / sum(e) > perc, 1, 'first' );
    
    if (knn == p0 || knn == p1), break, end            % no further change
    
    if (dk > d), p0 = knn;       % need more smoothing (search upper half)
    else         p1 = knn;       % need less smoothing (search lower half)
    end
end

% -----------------------------------------------------------------------------

function Kww = pgComputeASFM( W )

numF = size(W,1) / 2;
df = 1;
numF2 = numel( 1:df:numF );
num2  = numF2*(numF2-1) / 2;               % # of combinations of 2 out of numF2

global str
%hbar = waitbar(0, 'Computing Kww...', 'Name', sprintf('getKww2 (%d pairs)', num2));
kb = 0;

Kww = nan(numF);
for f1 = 1:df:numF
    Kww(f1,f1) = 0;
    for f2 = (f1+df):df:numF

        rows = 2 * [ f1 f1 f2 f2 ] - [ 1 0 1 0 ];
        [~,~,err] = getAffineRigidSFMerror( W(rows,:) );      
        
        Kww(f1,f2) = err;
        Kww(f2,f1) = err;
                
        kb = kb + 1;
    end
    %waitbar(kb/num2, hbar, str)
end
%close(hbar), clear global str

% -----------------------------------------------------------------------------
% REPROJECTION ERROR (aSFM)

function [M,t,err] = getAffineRigidSFMerror( W )

% computer centered (zero-mean) 2D observations
t = mean(W,2);
W = W - repmat( t ,1,size(W,2));

% compute error of rank-3 rigid SFM solution
[V,D] = eig(W*W');
M = V(:, end:-1:end-2);
err = W - M*(M'*W);

err = err(:)' * err(:);            % REPROJECTION ERROR (Frobenius norm)

% -----------------------------------------------------------------------------

function Kww0 = pgCompute_2D_RIK( W )

% Represent 2D shapes as complex column vectors
Z = complex( W(1:2:end,:), W(2:2:end,:) )';

% Move shapes to their respective centrois (nullspace of 1's)
n = size(Z,1);
Z = Z - repmat( mean(Z), n, 1 );

% Normalize scale (project onto unit sphere)
sz = sqrt(sum( Z .* conj(Z) ));
Z = Z ./ repmat( sz, n, 1 );

% Gram matrix can be calculated from the inner products matrix
Kww0 = 2 - 2*abs(Z'*Z);         % will be divided by -2*sigma^2

% -----------------------------------------------------------------------------