
function [X,M,S,t,S3] = pgA2( W, D, N, B, options )
%function [X,M,S,t,S3] = pgA2( W, D, N, B, options )
%
% A2: Iteratively-Reweighted NRSFM with RIKs (ECCV 2012)
%
% Code by Paulo Gotardo and Onur C. Hamsici
%
% Inputs:
%
% W is the observation matrix (here it is assumed complete; no missing data)
% D is the block diagonal matrix of camera projections (orthographic model)
% N is the number of iterations for IRLS algorithm (number of basis shapes)
% B is the DCT basis or KPCA basis obtained with a RIK
%
% Outputs:
%
% X is a compact representation of shape coefficients w.r.t to basis B
% M is the 2T-by-3N motion factor
% S is the 3N-by-n factor with N basis shapes (each one is 3-by-n)
% t is the mean column vector of W (2D translations due to camera motion)
% S3 is 3T-by-n, the t^th triplet or rows has the recovered 3D shape for image t
%
%
% NOTE: This is a simplified version of the original A2 algorithm described in
%       the ECCV 2012 paper. The new version is simpler and offers similar performance. :)
%
%       The two main differences between this version and the original one are:
%       ( i) the same basis B is used in all iterations of the IRLS loop;
%       (ii) there is no need for A3, when directly recovering new 3D shapes, 
%            one simply evaluates function c_t = f(w_t) as done after A1.
%
options.A2 = true;

% (1) Initialization: estimate t and make W zero mean
n = size(W,2);
I3 = eye(3);

t = mean(W,2);
Wc = W - repmat(t, 1, n);

Ei = Wc;
Gi = eye(n);

X = cell(N,1);
K = 1;
i = 0;
while (i < N)
%while ( sqrt(mean( Ei(:).^2 )) > options.tolE )
  
    i = i + 1; fprintf('\n(A2 IRLS iteration %d of %d):\n', i, N)
    
    % (1) factorize Wc = Mi*Si
    Wc = Ei * Gi;
    [X{i},Mi,Si] = pgA1( Wc, D, K, B, [], options );
    
    % (2) update error matrix and weight matrix
    Ei = Ei - Mi*Si;
    Gi = pgWeightMask( Ei );
end
X = cat(2, X{:});
C = B * X;
M = D * kronmex( C, I3 );
S = pgGetS( W - repmat(t,1,n), M );

S3 = kronmex( C, I3 ) * S;

% return % (done!)
% ----------------------------------------------------------------------------

function G = pgWeightMask( E )

% find the landmarks that have maximum residue
[~,j] = max( sum(E.^2), [], 2 );

% computes squared distances to E(:,j)
n = size(E,2);
dej2 = sum((E - repmat( E(:,j), 1, n )).^2);

% computes scale of Gaussian weight
nn = sort( dej2 ); knn = ceil(0.1*n);
se2 = mean( nn(1:knn) );

% Gaussian weight mask
G = diag( exp( -dej2/se2 ));

% ----------------------------------------------------------------------------

function [S,piM] = pgGetS (Wc, M)

K = size(M,2) / 3;
n = size(Wc,2);
S = zeros(3*K,n);

piM = cell(K,1);

for k = 1:K, k3 = 3*k-[2 1 0];
    Mk = M(:,k3);
    piM{k} = pinv(Mk);
    Sk = piM{k} * Wc;
    Wc = Wc - Mk * Sk;
    S(k3,:) = Sk;
end

% ----------------------------------------------------------------------------
