function [S0,W,T,n] = pgLoadDataNRSFM ( strName )

switch strName
    case {'walking'}                         % from Torresani et al.'s datasets
        [S0,W] = pgLoadDataNRSFM0( strName );
        
    case {'stretch','dance'}                 % Akhter et al.'s datasets
        mat = load([ './data/' strName ]); W = mat.W;
        if isfield(mat,'S'), S0 = mat.S;
        else S0 = mat.Shat;
        end
        
    case {'face1'}                           % Paladini et al.'s face1 dataset
        mat = load('./data/face1.mat');
        S0 = mat.GT.shape3D;
        mat.Rs = mat.GT.Rt;
        W  = mat.W2d;
        [T,n] = size(S0); T = T / 3;
        for f = 1:T, fff = 3*f - [2 1 0];
            S0(fff,:) = mat.GT.R3by3(fff,:) * S0(fff,:);
        end
end
[T,n] = size(W); T = T / 2;

%% Scale W and S0
scale = max(abs(W(:)));
W = W / scale;
S0 = S0 / scale;

% ------------------------------------------------------------------------------

function [ P3_gt, W ] = pgLoadDataNRSFM0( string )
%function [ P3_gt, W ] = pgLoadDataNRSFM0( string )
%
% where string = {'jaws','face2,'walking','capoeira'}
%
% Shark data has  240 frames, 91 points (P3_gt is 720x91)
% Face mocap has  316 frames, 40 points (P3_gt is 948x40)
% Walking mocap:  260 frames, 55 points (P3_gt is 780x55)
%
% loads the matrix P3_gt containing the ground thruth 3D shapes:
% P3_gt([t t+T t+2*T],:) contains the 3D coordinates of the J points at time t
% (T is the number of frames, J is the number of points)
% 
% [T, J] = size(P3_gt); T = T/3;
%
% 2D motion from orthographic projection (input to the non-rigid SFM algorithm)
%
% p2_obs = P3_gt([ 1 T+1 2 T+2 3 T+3 ... ], :);

load([ './data/' string '.mat' ]);

% For the datasets in Torresani et al (IEEE PAMI 2008):
T = size(P3_gt,1) / 3;
vect = 1:T;

% Adjusts matrix with 2D observations
rows = reshape([ vect ; vect+T ], [], 1); 
W = P3_gt(rows,:);
% W(2*t -[1 0], :) contains the 2D projection of the J points at time t

% Adjust matrix with 3D shapes [ 1 T+1 2 T+2 3 T+3 ... ]'
rows = reshape([ vect ; vect+T ; vect+T+T ], [], 1);  
P3_gt = P3_gt(rows,:);

% ------------------------------------------------------------------------------