function plot_3D_in_2D(S3)
%PLOT_3D_IN_2D Summary of this function goes here
%   Detailed explanation goes here
    
    [noOfFrames, ~] = size(S3);
    noOfFrames = noOfFrames/3;
    
    close all
    fig = figure;
    
    for i = 1:noOfFrames
        t = 3*i - [2, 1, 0];
        for p = 1:size(S3,2)
            scatter(S3(t(1), p), -S3(t(2), p), 'MarkerFaceColor', [0 0.5 0.5]);
            xlim([-0.5 0.5]);
            ylim([-0.5 0.5]);
            hold on;
        end
        pause(0.5);
        clf(fig,'reset');
    end
end

