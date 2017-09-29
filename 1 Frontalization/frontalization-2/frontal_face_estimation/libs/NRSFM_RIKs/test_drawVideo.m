
[noOfFrames, ~] = size(W);
noOfFrames = noOfFrames/2;

for i = 1:noOfFrames
    t = 2*i - [1, 0];
    plot(W(t(1),:), -W(t(2),:), 'ro', 'MarkerSize',  8);
    axis([300 1100 -900 -300]);
    pause(0.1);
end

[noOfFrames, ~] = size(S3);
noOfFrames = noOfFrames/3;

for i = 1:noOfFrames
    t = 3*i - [2, 1, 0];
    S3t = S3(t, :);
    plot(S3t(1,:), -S3t(2,:), 'bo', 'MarkerSize',  8);
    axis([-0.4 0.4 -0.4 0.4]);
    pause(0.1);
end