function [c,kw] = compute_RIKs_test( w, W, VeX, s2, type )
% function [c,kw] = compute_RIKs_test( w, W, VeX, s2, type )
%
% w    : new 2D shape, the new "test sample"
% W    : all 2D shapes used in the NRSFM "training" stage
% VeX  : mapping learned during the NRSFM stage
% s2   : squared scale parameter of the kernel
% type : the kernel type, 'RIK' or 'aSFM'

if ~exist('type','var'), type = 'aSFM'; end

% evaluate the kernel function
if strcmpi( type, 'aSFM' )
    kw0 = pgComputeASFM( w, W );
else
    kw0 = pgCompute_2D_RIK( w, W );
end
kw = exp( kw0 / (-2*s2) );

% evaluate learned mapping function
c = kw * VeX;

% -----------------------------------------------------------------------------
function kw0 = pgCompute_2D_RIK( w, W )

T = size(W,1)/2;
kw0 = zeros(1,T);

Z = complex( W(1:2:end,:), W(2:2:end,:) )';
z = complex( w(1,:), w(2,:) )';
V = isfinite( z );

for t = 1:T
    zt = Z(:,t);
    V2 = V & isfinite( zt );      % consider only commonly visible points
    
    % center and normalize scale
    zt = zt(V2) - mean( zt(V2) ); zt = zt / sqrt(zt'*zt); % or / norm(zt)
    zi = z (V2) - mean( z (V2) ); zi = zi / sqrt(zi'*zi);
    
    kw0(t) = 2 - 2*abs(zi'*zt);   % we later divide by -2*s2
end

% -----------------------------------------------------------------------------

function kw0 = pgComputeASFM( w, W )

numF = size(W,1) / 2;

kw0 = nan(1,numF);
for f = 1:numF

    ww = [ w ; W(2*f-[1 0],:) ];
    cols = all(~isnan(ww));
    ww = ww(:,cols);                % select only common visible columns
    
    [~,~,err] = getAffineRigidSFMerror( ww );
    
    %err = err *  (n / sum(cols));
    
    kw0(f) = err;
end

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