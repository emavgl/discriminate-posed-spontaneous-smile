%cleaning
clc
clear all

%loading
fprintf('Script started at: %s\n', datestr(now))
files = dir('**/**/landmarks/*.mat');
for file = files' %for each file in the path
    fprintf('Processing file: %s at %s\n', file.name, datestr(now))
    %fprintf('Current time: %s\n', datestr(now))
    
    %load the file
    var=load(file.name);
    nameVar = fieldnames(var);
    landmarks = var.((nameVar{1}));
    
    %elaborate the landmarks
    [S3, Rf] = Reconstruct3D(landmarks,'A2',27,0.3,'RIKs');
    Aligned_S3=AlignFace3D(S3, Rf);

    %save the result
    delete(strcat('landmarks/',file.name))
    save(strcat('alignedLandmarks/',file.name),'Aligned_S3')
end
fprintf('Script ended at: %s\n', datestr(now))