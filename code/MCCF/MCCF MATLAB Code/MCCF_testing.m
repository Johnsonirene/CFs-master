
%---------------------------------------------------
%   Testing Multi-Channel Correlation Filters for Eye Detection
%   Multi-Channel Correlation Filters : ICCV'13
%   author    : Hamed Kiani
%   date      : 30 June 2014
%---------------------------------------------------

clear all; close all; clc;

%---------------------------------------------------
%   adding helper functions and images path.
%   we use multiPie face dataset for both training and testing, including
%   902 images of frontal faces with normal lighting.
%   the first 500 for training and the rest for testing.
%---------------------------------------------------

addpath('helper functions/');
imgsPath = 'multiPie Face Dataset/';
imgs     = dir(fullfile(imgsPath, '*.jpg'));

%---------------------------------------------------
%   parameters setting
%---------------------------------------------------

%   The "right eye" is located at [40 96].
%   We normalized all face images to the size of [128 128], where
%   the left and right eye are located at [40 32] and [40 96],
%   respectively.

im_sz      = [128 128];
target_pos = [40 96];

%   HoG parameters. Pleae refer to "calc_hog" function for more details.
nbins      = 5;
cell_size  = [5 5];
block_size = [5 5];

%   Cosine window applied on images to reduce the high frequenceis
%   caused by image borders.
%   Please refer to "get_cosine_window" function.
cos_window = get_cosine_window(im_sz,2);

%   loading the trained filter
load 'filt';

%   FFT2 of the filter. We perform detection in the Fourier domain for
%   its simple and super fast detection!
filt_f = fft2(filt);

%   uncomment this line if you want to save the detection results as video
% aviobj = avifile('example.avi','compression','None');

%   testing on 402 face images, from 501-902. Note that the first 500 face
%   images were used to train the MCCF filter.

%   to get the detection speed
proc_time = 0;

%   testing loop starts here!
for i = 501:902
    
    tic;
    %   loading images
    im     = imread([imgsPath imgs(i).name]);
    org_im = im;
    
    %   RGB to Gray
    if size(im,3) == 3
        im = double(rgb2gray(im));
    end;
    
    %   image power-normalization to have zero mean and unit variance
    %   to be robust against lighting variations
    nor_im = powerNormalise(double(im));     
    
    %   calculating dense HoG for the normalized face image. nbins,
    %   cell_size and block_size are the parameters required to
    %   compute HoG. Please refer to "calc_hog" for more details.
    hogs = calc_hog(nor_im, nbins, cell_size, block_size);
    
    %   applying cosine window on the HoG channels to reduce the
    %   high frequencies of image borders.
    hogs = bsxfun(@times, hogs, cos_window);
    
    
    %   FFT2 of HoG feature channels.
    %   "_f" postfix indicates the variable in the frequncy domain.
    hogs_f = fft2(hogs);
    
    %   correlation response of the given image and filter
    rsp_f  = sum(hogs_f.*filt_f,3);    
    
    %   correlation output in the spatial domain
    rsp    = circshift(real(ifft2(rsp_f)), -size(im)/2);
    
    
    %   predicting the right eye location. The global maximum over the
    %   entire correlation output.
    [x y] = find(rsp == max(max(rsp)));
    
    % total detection time for all testing images.
    proc_time = proc_time + toc;
    
    %   draw ground-truth (blue) and predicited (red) boxes
    subplot(1,2,1);
    imagesc(rsp); colormap gray;axis image ; axis off;title ('Correlation rsp.');
    
    subplot(1,2,2);
    imagesc(org_im); colormap gray;axis image ; axis off;title ('image');             
    hold on; plot(96,40, 'ob','MarkerSize',10,'LineWidth',3);
    hold on; plot(y,x, '*r','MarkerSize',10,'LineWidth',2);
    pause(.05);
    
%     aviobj = addframe(aviobj,gcf); 
    
end;
% close aviobj;
