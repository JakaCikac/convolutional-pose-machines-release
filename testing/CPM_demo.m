%% Demo code of "Convolutional Pose Machines", 
% Shih-En Wei, Varun Ramakrishna, Takeo Kanade, Yaser Sheikh
% In CVPR 2016
% Please contact Shih-En Wei at shihenw@cmu.edu for any problems or questions
%%
close all;
addpath('src'); 
addpath('util');
addpath('util/ojwoodford-export_fig-5735e6d/');
param = config();
% 1 if video is provided, 0 if image is provided
video = 1; 
fprintf('Description of selected model: %s \n', param.model(param.modelID).description);
interestPart = 'head'; % to look across stages. check available names in config.m

path_to_files = '/home/nanorax/Desktop/video/';
videoType = '*.m4v';
allVideos = dir( [path_to_files videoType]);
% get all names of videos
videoAllNames = {allVideos(arrayfun(@(x) ~x.isdir, allVideos)).name};

tTotal = tic;

%% Run the actual model
if (video == 1)
    
    
    for k = 1:length(videoAllNames)
        
        tVid = tic;
        
        videoLoadString = sprintf('%s%s', path_to_files, videoAllNames{k});
        % load the movie for which we wish to compute the pose (ground truth)
        mov = VideoReader(videoLoadString);
        movName = videoAllNames{k};
        movNameFull = [movName, '.mat'];
        
        disp(['----------------------------------------------']);
        disp(['-------Working on Video: ' movName ' . -------']);
        disp(['----------------------------------------------']);
        
        % read the actual movie
        vidFrames = read(mov);
        % retrieve the number of frames in the video to use in the for loop
        nFrames = mov.NumberOfFrames;

        predictionMat = [];

        for i=1:nFrames

           close all;          % close all figures
           clear heatMaps;     % release a bit of memory
           clear prediction;   % release more memory
           caffe.reset_all()   % release GPU memory by clearing previous net
           
           disp(['------- Computing frame ' i ' out of ' nFrames ' frames. -------']);

           %figure(1); 
           %imshow(vidFrames(:,:,:,i),[]);  %frames are grayscale
           %hold on;
           %title('Drag a bounding box');
           rectangle = [0,0, size(vidFrames(:,:,:,i), 2) size(vidFrames(:,:,:,1))];
           [heatMaps, prediction] = applyModel(vidFrames(:,:,:,i), param, rectangle, 1);
           %% visualize, or extract variable heatMaps & prediction for your use
           %visualize(vidFrames(:,:,:,i), heatMaps, prediction, param, rectangle, interestPart, 1);

           % store prediction as ground truth in a mat file (2 x 14 x nFrames)
           predictionMat(:,:,i) = prediction';


        end

        save([path_to_files 'joint_positions_' movNameFull], 'predictionMat')

        toc(tVid);
        
    end
    
    toc(tTotal);
    
else 

    %% core: apply model on the image, to get heat maps and prediction coordinates
    figure(1); 
    imshow(test_image);
    hold on;
    title('Drag a bounding box');
    rectangle = getrect(1);
    [heatMaps, prediction] = applyModel(test_image, param, rectangle, 0);


    %% visualize, or extract variable heatMaps & prediction for your use
    visualize(test_image, heatMaps, prediction, param, rectangle, interestPart, 0);
end