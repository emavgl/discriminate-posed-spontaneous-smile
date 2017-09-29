function [err3D,S0R,S3R,errS3] = pgCompare3DShapes ( S0, S3, Rs, unrotate_flag )

if exist('unrotate_flag','var')
    [S0R,S3R] = alignStruct_unrotateS0( S0, S3, Rs );
elseif exist('Rs','var')
    [S0R,S3R] = alignStruct( S0, S3, Rs );
else
    [S0R,S3R] = alignStruct( S0, S3 );
end

% Compute 3D errors
[T,n] = size( S0 ); T = T / 3;
errS3 = zeros(T,n);                                   % 3D reconstruction error
for f = 1:T, f3 = 3*f-[2 1 0];
    errS3(f,:) = sqrt( sum( (S0R(f3,:) - S3R(f3,:)).^2 ) ); % 3D Euclidean dist
end
s = mean( std(S0R, 1, 2) );                           % scale (avg row std.dev.)
errS3 = errS3 / s;
err3D = mean( errS3(:) );

%err3D = norm(S0R - S3R, 'fro') / norm(S0R,'fro');    % as in Torresani et al.

% ------------------------------------------------------------------------------

function [S0R,SR,Y] = alignStruct_unrotateS0( S0, S3, Rs )

[F,P] = size(S0); F = F/3;

[S0R,SR,Y,R] = alignStruct (S0, S3, Rs);

% Remove 3D rotations R from all shapes
for f = 1:F
    f3 = 3*f-[2 1 0];
    Rf = R{f}' * Y';
    
    SR(f3,:) = Rf * SR(f3,:);  % SR = Y * R * S3
    S0R(f3,:) = Rf * S0R(f3,:);  % SR = Y * R * S3
end;

% ------------------------------------------------------------------------------

function [S0,SR,Y,rf] = alignStruct (S0, S, Rs )
%function [S0,SR,Y,rf] = alignStruct (S0, S, Rs )
%
% Based on code by Akhter, Khan, Sheikh, and Kanade:
% "Nonrigid Structure from Motion in Trajectory Domain", NIPS08
% http://cvlab.lums.edu.pk/nrsfm/
%
% Modified by Paulo Gotardo
%
[F,P] = size(S0); F = F/3;

% make shapes zero mean
S0 = S0 - repmat( mean(S0,2), 1, P );
S  = S  - repmat( mean(S ,2), 1, P );

rf = cell(F,1);
if ~exist('Rs','var')
    SR = S;                    % do not apply camera rotations before alignment
else
    SR = zeros(3*F,P);

    for f = 1:F
        f2 = 2*f-[1 0]; f3 = 3*f-[2 1 0];
    
        %r = [ Rs(f2, :) ; cross(Rs(f2(1),:), Rs(f2(2),:)) ];
        r = Rs(f2, :);
        n = norm(r(1,:));
        r = r / n;
        r = n * [ r ; cross( r(1,:), r(2,:) ) ];
        
        if ( det(r) < 0 ), r(3,:) = -r(3,:); disp('alignStruct() det<0!'), end
    
        % Apply each rotation f to its corresponding 3D shape in frame f
        SR(f3,:) = r * S(f3, :);
        rf{f} = r;
    end
end

% Procrust Alignment: find 3x3 rotation matrix Y
Y = findRotation(S0, SR);
for f = 1:F
    f3 = 3*f-[2 1 0];
    SR(f3,:) = Y * SR(f3,:);
end;

% ----------------------------------------------------------------------------

function [Y] = findRotation(S, Sh)

[F,P] = size(S); F = F / 3;

S1 = zeros(3,F*P);
S2 = zeros(3,F*P);
cols0 = 1:P;
for f = 1:F
    rows = 3*f - [2 1 0];
    cols = cols0 + (f-1)*P; 
    S1(:,cols) = S (rows,:);
    S2(:,cols) = Sh(rows,:);
end;

% This is not really a rotation, but a 3x3 afine alignment
Y = S1 / S2;             
% Y = inv(S2*S2')*S1*S2';

% Now make it a rotation:
[U, D, V] = svd(Y);
Y = U*V';

% ----------------------------------------------------------------------------
