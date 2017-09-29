% COMPUTER GENERATED versus NATURAL HUMAN FACES 
% Temporal-spatial-based discrimination
%
% dnductien, 2014
% -------------------------------------------------------------------------
% Rotate the 3D face to be frontal view with all poses = 0.
% 

function S3 = AlignFace3D(s3, R)
    [noOfFrames, ~] = size(s3);
    noOfFrames = noOfFrames/3;
    S3 = s3;
    for i = 1:noOfFrames
        t = 3*i - [2, 1, 0];
        
        %x_center = x(3);
        theta= 5*(pi/180);
        Rot=[cos(theta) -sin(theta) 0; sin(theta) cos(theta) 0; 0 0 1];

        S3(t, :) = R{i}*S3(t, :);
        
        S3(t, :) = Rot*S3(t, :);
        %S3(t(2), :) = -S3(t(2), :); % change the y-axis to Eulerian coord
    end
end