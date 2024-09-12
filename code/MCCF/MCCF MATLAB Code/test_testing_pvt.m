clear all; close all; clc;

addpath('helper functions/');
imgsPath = 'multiPie Face Dataset/';
imgs     = dir(fullfile(imgsPath, '*.jpg'));

im_sz      = [128 128];
target_pos = [40 96];

nbins      = 5;
cell_size  = [5 5];
block_size = [5 5];

cos_window = get_cosine_window(im_sz,2);

load 'filt';
filt_f = fft2(filt);

% aviobj = avifile('example.avi','compression','None');

proc_time = 0;

for i = 501:902

    tic;

    im = imread([imgsPath imgs(i).name])
    org_im = im;
    
    if size(im,3) == 3
        im = double(rgb2gray(im));
    end;

    nor_im = powerNormalise(double(im));

    hogs = calc_hog(nor_im, nbins, cell_size, block_size);
    hogs = bsxfun(@times, hogs, cos_window);
    hogs_f = fft2(hogs);

    rsp_f = sum(hogs_f.*filt_f,3);
    rsp = circshift(real(ifft2(rsp_f)), -size(im)/2);

    [x y] = find(rsp == max(max(rsp)));

    proc_time = proc_time + toc;

    subplot(1,2,1);
    imagesc(rsp); colormap gray; axis image; axis off; title ('Correlation rsp.');
    
    subplot(1,2,2);
    imagesc(org_im); colormap gray; axis image; axis off; title ('image');
    hold on; plot(96,40, 'ob', 'MarkerSize',10,'LineWidth',3)
    hold on; plot(y,x, '*r','MarkerSize',10,'LineWidth',2);
    pause(.05);
%     aviobj = addframe(aviobj,gcf);

end;
    